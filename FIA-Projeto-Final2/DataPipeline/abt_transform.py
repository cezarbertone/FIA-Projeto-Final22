"""
Camada Silver -> Gold: construção da Analytical Base Table (ABT).

Transforma `Dados/Silver/clean_data.parquet` na ABT (`Dados/Gold/abt.parquet`) pronta para modelagem:
feature engineering, descarte de colunas muito vazias, imputação e encoding.

Os parâmetros aprendidos (colunas descartadas, valores de imputação e mapeamentos
de categóricas) são persistidos em `abt_artifacts.pkl` para que a mesma transformação
possa ser reproduzida em produção (predict.py) sobre novas solicitações.
"""
import os
import sys
import logging
import importlib.util

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLOps")))
import storage

# Carrega Model/config.py por caminho (nome de módulo único p/ evitar colisão com este config).
# Precisamos de RANDOM_STATE/TEST_SIZE p/ reproduzir EXATAMENTE o split do train.py e fitar o
# pré-processamento só no treino (sem data leakage para o holdout).
_mcfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model", "config.py")
_spec = importlib.util.spec_from_file_location("model_config", _mcfg_path)
model_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de razão financeira sem fragmentar o DataFrame e tratando inf."""
    new_features = {}
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(df.columns):
        new_features["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(df.columns):
        new_features["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_CREDIT"} <= set(df.columns):
        new_features["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    if new_features:
        ratios = pd.DataFrame(new_features, index=df.index)
        # Divisão por zero gera inf -> tratamos como nulo para imputação posterior
        ratios = ratios.replace([np.inf, -np.inf], np.nan)
        df = pd.concat([df, ratios], axis=1)
    return df


def create_abt(input_path: str, output_path: str, artifacts_path: str) -> pd.DataFrame:
    """Transforma os dados limpos em ABT e persiste os artefatos da transformação."""
    logging.info(f"Iniciando construção da ABT a partir de: {input_path}")

    if storage.STORAGE_BACKEND == "local" and not os.path.exists(input_path):
        logging.error(f"Arquivo não encontrado: {input_path}")
        raise FileNotFoundError(input_path)

    df = storage.read_parquet(input_path)

    # 0. Enriquecimento com histórico de crédito (agregações de bureau + previous_application)
    aux_feature_cols = []
    if getattr(config, "USE_AUX_AGGREGATIONS", False):
        from feature_aggregation import build_aux_features
        logging.info("Agregando tabelas auxiliares (bureau + previous_application)...")
        aux = build_aux_features(config.CLEAN_BUREAU_PATH, config.CLEAN_PREV_PATH)
        aux_feature_cols = list(aux.columns)
        df = df.merge(aux, how="left", left_on=config.ID_COL, right_index=True)
        # Cliente sem histórico não aparece nas auxiliares -> NaN -> 0 ("nenhum histórico")
        df[aux_feature_cols] = df[aux_feature_cols].fillna(0)
        logging.info(f"ABT enriquecida com {len(aux_feature_cols)} features de histórico de crédito.")

    # 1. Feature Engineering
    logging.info("Criando novas features...")
    df = _add_ratio_features(df)

    # Colunas que não são features (não entram em imputação/encoding)
    non_features = [c for c in (config.ID_COL, config.TARGET_COL) if c in df.columns]

    # A ABT existe para TREINAR o modelo supervisionado de risco (prever TARGET). Sem a coluna alvo
    # não há o que aprender aqui — scoring de base nova (sem rótulo) é responsabilidade do predict.py,
    # que reusa os artefatos deste treino. Por isso exigimos TARGET e falhamos explicitamente.
    if config.TARGET_COL not in df.columns:
        raise ValueError(
            f"Coluna alvo '{config.TARGET_COL}' ausente. A ABT é gerada para treino supervisionado; "
            f"para escorar base sem rótulo use Model/predict.py (reaproveita os artefatos do treino)."
        )

    # Split de treino IDÊNTICO ao do train.py: o pré-processamento (drop/imputação/encoding) é
    # APRENDIDO só nas linhas de treino e depois aplicado à ABT inteira — evita data leakage do
    # holdout. A partição depende apenas de (n_linhas, random_state, test_size, stratify=TARGET),
    # que são os mesmos do treino, garantindo o mesmo conjunto de treino.
    train_idx, _ = train_test_split(
        df.index, test_size=model_config.TEST_SIZE, random_state=model_config.RANDOM_STATE,
        stratify=df[config.TARGET_COL])
    df_fit = df.loc[train_idx]
    logging.info(f"Pré-processamento fitado SÓ no treino ({len(df_fit)} linhas) p/ evitar leakage.")

    # 2. Descarte de colunas com excesso de nulos
    missing_pct = df_fit.drop(columns=non_features).isnull().mean()
    cols_to_drop = [
        c for c in missing_pct[missing_pct > config.MISSING_THRESHOLD].index
        if c not in config.KEEP_ALWAYS
    ]
    logging.info(
        f"Removendo {len(cols_to_drop)} colunas com > {config.MISSING_THRESHOLD:.0%} de nulos "
        f"(preservando {config.KEEP_ALWAYS})."
    )
    df = df.drop(columns=cols_to_drop)
    df_fit = df_fit.drop(columns=cols_to_drop)

    feature_cols = [c for c in df.columns if c not in non_features]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # 3. Imputação (mediana/moda APRENDIDAS no treino, aplicadas à base inteira) -- persistidas
    logging.info("Tratando valores nulos...")
    impute_values = {}
    for col in num_cols:
        impute_values[col] = float(df_fit[col].median())
    for col in cat_cols:
        impute_values[col] = str(df_fit[col].mode(dropna=True)[0])
    # Features de histórico: ausência = "sem histórico" -> imputa 0 (e não a mediana),
    # para o predict.py reproduzir o mesmo default de clientes sem bureau/prev.
    for col in aux_feature_cols:
        if col in impute_values:
            impute_values[col] = 0.0
    df = df.fillna(impute_values)
    df_fit = df_fit.fillna(impute_values)

    # 4. Encoding de categóricas via mapeamento reproduzível (categorias do TREINO; unseen -> -1)
    logging.info("Codificando variáveis categóricas...")
    cat_mappings = {}
    for col in cat_cols:
        categories = sorted(df_fit[col].astype(str).unique())
        mapping = {cat: code for code, cat in enumerate(categories)}
        cat_mappings[col] = mapping
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # 5. Persistência: ABT (Gold, Parquet) + artefatos da transformação. O caminho plano é o
    #    "latest"; se a ingestão for datada, grava também o snapshot dt=<data>/ do mês.
    storage.write_parquet(df, output_path)
    if config.ABT_SNAPSHOT_PATH:
        storage.write_parquet(df, config.ABT_SNAPSHOT_PATH)
        logging.info(f"Snapshot Gold salvo em: {config.ABT_SNAPSHOT_PATH}")

    artifacts = {
        "dropped_columns": cols_to_drop,
        "impute_values": impute_values,
        "categorical_mappings": cat_mappings,
        "feature_columns": feature_cols,   # ordem das features (exclui ID e TARGET)
        "aux_feature_columns": aux_feature_cols,  # features de histórico (default 0 no predict)
        "target_col": config.TARGET_COL,
        "id_col": config.ID_COL,
    }
    storage.write_pickle(artifacts, artifacts_path)
    if config.ABT_ARTIFACTS_SNAPSHOT_PATH:
        storage.write_pickle(artifacts, config.ABT_ARTIFACTS_SNAPSHOT_PATH)
        logging.info(f"Snapshot de artefatos salvo em: {config.ABT_ARTIFACTS_SNAPSHOT_PATH}")
    logging.info(f"ABT criada e salva em: {output_path}. Shape final: {df.shape}")
    logging.info(f"Artefatos da transformação salvos em: {artifacts_path}")
    return df


if __name__ == "__main__":
    create_abt(config.CLEAN_PATH, config.ABT_PATH, config.ABT_ARTIFACTS_PATH)
