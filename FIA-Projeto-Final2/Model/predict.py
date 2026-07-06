"""
Serviço de predição (scoring) do Score de Risco de Inadimplência.

Aplica a MESMA transformação da ABT a novas solicitações (usando os artefatos persistidos
em `abt_artifacts.pkl`) e gera a probabilidade de inadimplência com o modelo treinado
(`model.pkl`), além de uma decisão de negócio baseada em um threshold configurável.

Uso:
    from Model.predict import predict
    resultado = predict({"SK_ID_CURR": 100002, "CODE_GENDER": "M", "AMT_INCOME_TOTAL": 202500, ...})

Ou via CLI (escora N linhas de um CSV no formato de application/raw):
    python Model/predict.py --input Dados/Bronze/application_train.csv --n 5
"""
import os
import sys
import argparse
import logging
import importlib.util
from typing import Union

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLOps")))
import storage

# Carrega Model/config.py por caminho explícito com nome de módulo único ("model_config").
# Evita colisão com DataPipeline/config.py quando ambos estão no sys.path e no cache de
# sys.modules (ex.: orquestração Airflow, onde DataPipeline é importado antes).
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
_spec = importlib.util.spec_from_file_location("model_config", _cfg_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Threshold de decisão de negócio (probabilidade acima da qual recomendamos NEGAR/revisar)
DECISION_THRESHOLD = 0.50

DAYS_EMPLOYED_ANOMALY = 365243
TIME_COLS = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]
# Caminho LÓGICO (relativo à raiz do projeto) — idêntico à key que `abt_transform` usa p/ GRAVAR o
# artefato (DataPipeline/config.ABT_ARTIFACTS_PATH). Precisa ser relativo p/ que, no backend MinIO, a
# object key bata com a gravada pelo pipeline (o storage usa o path como key). Os entry-points rodam
# com cwd = raiz do projeto — mesma premissa do `config.MODEL_PATH` (também relativo).
ARTIFACTS_PATH = os.path.join("DataPipeline", "abt_artifacts.pkl")


def _load_pickle(path: str):
    # Lê do filesystem ou do data lake MinIO conforme STORAGE_BACKEND (ver storage.py).
    return storage.read_pickle(path)


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Replica DataPipeline/data_sanitization.py para dados de entrada brutos."""
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    for col in TIME_COLS:
        if col in df.columns:
            df[col] = df[col].abs()
    return df


def _to_abt(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Replica DataPipeline/abt_transform.py usando os parâmetros aprendidos no treino."""
    df = df.copy()

    # Feature engineering (mesmas razões, inf -> NaN)
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_CREDIT"} <= set(df.columns):
        df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df = df.replace([np.inf, -np.inf], np.nan)

    impute_values = artifacts["impute_values"]
    cat_mappings = artifacts["categorical_mappings"]
    feature_cols = artifacts["feature_columns"]

    # Garante todas as colunas de feature; ausentes entram como NaN (serão imputadas).
    # Insere todas de uma vez (concat) p/ evitar fragmentação do DataFrame (PerformanceWarning).
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)

    # Imputação com os valores aprendidos no treino
    df = df.fillna(impute_values)

    # Encoding categórico com os mapeamentos do treino (categoria não vista -> -1)
    for col, mapping in cat_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Reordena exatamente como o modelo espera
    return df[feature_cols]


def predict(records: Union[dict, list, pd.DataFrame],
            model_path: str = None,
            artifacts_path: str = ARTIFACTS_PATH,
            threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """Recebe 1+ solicitações (dict/list/DataFrame no formato application) e retorna o score."""
    model_path = model_path or config.MODEL_PATH

    if isinstance(records, dict):
        df_in = pd.DataFrame([records])
    elif isinstance(records, list):
        df_in = pd.DataFrame(records)
    else:
        df_in = records.copy()

    artifacts = _load_pickle(artifacts_path)
    bundle = _load_pickle(model_path)
    model = bundle["model"]

    ids = df_in[artifacts["id_col"]] if artifacts["id_col"] in df_in.columns else pd.Series(range(len(df_in)))

    X = _to_abt(_sanitize(df_in), artifacts)
    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        artifacts["id_col"]: ids.values,
        "probabilidade_inadimplencia": proba,
        "decisao": np.where(proba >= threshold, "NEGAR/REVISAR", "APROVAR"),
    })
    return out


# Nomes amigáveis das features do modelo (p/ a explicação ao usuário). Fallback: nome cru.
FRIENDLY_NAMES = {
    "EXT_SOURCE_1": "Score de crédito externo 1",
    "EXT_SOURCE_2": "Score de crédito externo 2",
    "EXT_SOURCE_3": "Score de crédito externo 3",
    "CREDIT_INCOME_RATIO": "Crédito ÷ renda",
    "ANNUITY_INCOME_RATIO": "Parcela ÷ renda",
    "ANNUITY_CREDIT_RATIO": "Parcela ÷ crédito",
    "DAYS_EMPLOYED": "Tempo de emprego",
    "DAYS_BIRTH": "Idade",
    "CODE_GENDER": "Gênero",
    "NAME_EDUCATION_TYPE": "Escolaridade",
    "AMT_GOODS_PRICE": "Valor do bem",
    "AMT_CREDIT": "Valor do crédito",
    "AMT_ANNUITY": "Valor da parcela",
    "AMT_INCOME_TOTAL": "Renda anual",
    "NAME_FAMILY_STATUS": "Estado civil",
    "OCCUPATION_TYPE": "Ocupação",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Inadimplentes no círculo social",
}


def friendly_name(col: str) -> str:
    """Rótulo amigável de uma feature. PREV_*/BUREAU_* viram 'Histórico de crédito: ...'."""
    if col in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[col]
    if col.startswith(("PREV_", "BUREAU_")):
        return "Histórico de crédito (" + col.split("_", 1)[1].replace("_", " ").lower() + ")"
    return col.replace("_", " ").title()


def to_features(records: Union[dict, list, pd.DataFrame],
                artifacts_path: str = ARTIFACTS_PATH) -> pd.DataFrame:
    """Aplica a transformação da ABT e retorna o DataFrame de features que entra no modelo."""
    artifacts = _load_pickle(artifacts_path)
    if isinstance(records, dict):
        df_in = pd.DataFrame([records])
    elif isinstance(records, list):
        df_in = pd.DataFrame(records)
    else:
        df_in = records.copy()
    return _to_abt(_sanitize(df_in), artifacts)


def _reference_row(artifacts: dict, feature_cols: list) -> pd.Series:
    """Linha de referência 'neutra' (no espaço de features do modelo): mediana p/ numéricas e a
    moda codificada p/ categóricas. Usada como baseline na atribuição por occlusion."""
    impute = artifacts.get("impute_values", {})
    cat_maps = artifacts.get("categorical_mappings", {})
    ref = {}
    for col in feature_cols:
        iv = impute.get(col, 0)
        if col in cat_maps:                       # categórica: moda (string) -> código
            ref[col] = cat_maps[col].get(str(iv), -1)
        else:
            ref[col] = iv if iv is not None else 0
    return pd.Series(ref)[feature_cols]


def _explain_shap(model, X):
    """Contribuições SHAP (log-odds). Requer `shap` (e `numba`) — usado quando disponível."""
    import shap
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X))
    if sv.ndim == 3:                              # alguns modelos: (n, n_features, n_classes)
        sv = sv[..., -1]
    return sv


def _explain_occlusion(model, X, artifacts):
    """Atribuição local SEM dependências: para cada feature, mede quanto a probabilidade muda ao
    substituí-la pelo valor neutro (mediana/moda). delta>0 => o valor atual AUMENTA o risco.
    Fallback robusto quando o `shap` não está instalado (ex.: imagem Docker em Python 3.14)."""
    ref = _reference_row(artifacts, list(X.columns))
    base = model.predict_proba(X)[:, 1]
    out = np.zeros((len(X), X.shape[1]))
    cols = list(X.columns)
    for i in range(len(X)):
        # matriz: n_features cópias da linha i, cada uma com 1 feature trocada pelo valor neutro
        grid = pd.DataFrame(np.repeat(X.iloc[[i]].values, len(cols), axis=0), columns=cols)
        for j, c in enumerate(cols):
            grid.iat[j, j] = ref[c]
        out[i] = base[i] - model.predict_proba(grid)[:, 1]   # contribuição da feature atual vs neutro
    return out


def explain(records: Union[dict, list, pd.DataFrame],
            top_n: int = 8,
            model_path: str = None,
            artifacts_path: str = ARTIFACTS_PATH,
            only_cols: list = None) -> list:
    """Explica a decisão: principais fatores que empurraram o risco p/ cima/baixo.

    Usa **SHAP** quando disponível (contribuição em log-odds) e, se o `shap` não estiver instalado,
    cai para uma **atribuição por occlusion** (variação da probabilidade vs. valor neutro) — sem
    dependências extras. Retorna, por solicitação, uma lista ordenada de dicts:
        {"fator": <rótulo amigável>, "contribuicao": <float>, "efeito": "aumenta"|"reduz"}
    `contribuicao` > 0 aumenta o risco (puxa p/ inadimplência); < 0 reduz.

    `only_cols`: se informado, restringe a explicação a essas features (as demais — tipicamente as
    imputadas, que ficam num valor constante e dão a mesma contribuição p/ todo solicitante — são
    omitidas). Usado pelo app p/ mostrar só as variáveis que o usuário de fato informou.
    """
    model_path = model_path or config.MODEL_PATH
    model = _load_pickle(model_path)["model"]
    artifacts = _load_pickle(artifacts_path)
    X = _to_abt(_sanitize(pd.DataFrame([records]) if isinstance(records, dict)
                          else pd.DataFrame(records) if isinstance(records, list)
                          else records.copy()), artifacts)

    try:
        contribs_all = _explain_shap(model, X)
    except ImportError:
        contribs_all = _explain_occlusion(model, X, artifacts)

    keep = [c for c in X.columns if c in set(only_cols)] if only_cols else list(X.columns)

    explanations = []
    for i in range(len(X)):
        contribs = (pd.Series(contribs_all[i], index=X.columns)[keep]
                    .sort_values(key=np.abs, ascending=False))
        explanations.append([
            {"fator": friendly_name(col),
             "contribuicao": float(val),
             "efeito": "aumenta" if val > 0 else "reduz"}
            for col, val in contribs.head(top_n).items()
        ])
    return explanations


def _cli():
    parser = argparse.ArgumentParser(description="Score de risco de inadimplência")
    parser.add_argument("--input", default=os.path.join("Dados", "Bronze", "application_train.csv"),
                        help="CSV no formato application/raw")
    parser.add_argument("--n", type=int, default=5, help="Nº de linhas a escorar")
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    args = parser.parse_args()

    df = storage.read_csv(args.input, nrows=args.n)
    result = predict(df, threshold=args.threshold)
    print(result.to_string(index=False))


if __name__ == "__main__":
    _cli()
