"""
Camada Bronze -> Silver: limpeza e padronização dos dados brutos.

Limpa as TRÊS fontes do Kaggle, cada uma no seu NÍVEL DE REGISTRO ORIGINAL (Silver = representação
limpa e reutilizável, sem mudança de nível de registro):
- `application_train.csv`      -> `clean_data.parquet`                 (1 linha/solicitante)
- `bureau.csv`                 -> `clean_bureau.parquet`              (1 linha/crédito externo)
- `previous_application.csv`   -> `clean_previous_application.parquet` (1 linha/pedido anterior)

Corrige anomalias conhecidas do Home Credit e deduplica. NÃO faz feature engineering, imputação
nem agregação — a agregação por cliente (BUREAU_*/PREV_*) e o resto ficam na camada Gold
(feature_aggregation.py + abt_transform.py).
"""
import os
import sys
import logging

import pandas as pd
import numpy as np

import config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLOps")))
import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sanitize_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Executa a limpeza e padronização dos dados brutos (Bronze -> Silver)."""
    # Bronze é lida como a UNIÃO das partições dt=*/ (corpus acumulado da ingestão mensal).
    # No modo snapshot único recai no arquivo plano informado (uma partição só).
    paths = config.bronze_partition_paths(input_path)
    logging.info(f"Iniciando sanitização. Lendo {len(paths)} partição(ões) Bronze: {paths}")

    if storage.STORAGE_BACKEND == "local":
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            logging.error(f"Arquivo(s) Bronze não encontrado(s): {missing}")
            raise FileNotFoundError(missing[0])

    df = pd.concat([storage.read_csv(p) for p in paths], ignore_index=True)
    logging.info(f"Dados carregados. Shape inicial: {df.shape}")

    # 1. Padronização de nomes de colunas (caixa alta, sem espaços) -- feito primeiro
    #    para que as referências de colunas abaixo sejam consistentes.
    df.columns = [col.strip().upper() for col in df.columns]

    # 2. Remover duplicatas (linha idêntica)
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_rows:
        logging.info(f"Removidas {initial_rows - df.shape[0]} linhas duplicadas.")

    # 2b. Ingestão mensal: se o mesmo cliente reaparecer em drops diferentes (re-dump), o
    #     mais recente (última partição dt=) vence. No modo snapshot único é no-op.
    if config.ID_COL in df.columns:
        before = df.shape[0]
        df = df.drop_duplicates(subset=config.ID_COL, keep="last")
        if df.shape[0] < before:
            logging.info(f"Removidas {before - df.shape[0]} duplicatas por {config.ID_COL} "
                         f"(mês mais recente vence).")

    # 3. Corrigir anomalia conhecida: DAYS_EMPLOYED == 365243 representa "sem emprego" (nulo).
    if "DAYS_EMPLOYED" in df.columns:
        anomalies = int((df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_ANOMALY).sum())
        if anomalies > 0:
            logging.info(f"Corrigindo {anomalies} valores anômalos em DAYS_EMPLOYED.")
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(config.DAYS_EMPLOYED_ANOMALY, np.nan)

    # 4. Converter colunas de tempo (negativas na origem) para valor absoluto.
    for col in config.TIME_COLS:
        if col in df.columns:
            df[col] = df[col].abs()

    # Salva o arquivo limpo (Silver, Parquet — no filesystem ou no data lake). O caminho plano
    # é o "latest"; se a ingestão for datada, grava também o snapshot dt=<data>/ do mês.
    storage.write_parquet(df, output_path)
    if config.CLEAN_SNAPSHOT_PATH:
        storage.write_parquet(df, config.CLEAN_SNAPSHOT_PATH)
        logging.info(f"Snapshot Silver salvo em: {config.CLEAN_SNAPSHOT_PATH}")
    logging.info(f"Sanitização concluída. Arquivo salvo em: {output_path} com shape {df.shape}")
    return df


def _sanitize_aux(input_path: str, output_path: str, snapshot_path, dedup_col: str,
                  clip_debt: bool = False) -> pd.DataFrame:
    """Bronze -> Silver de uma tabela auxiliar, preservando o NÍVEL DE REGISTRO ORIGINAL (não agrega).

    Lê a Bronze como a UNIÃO das partições dt=*/ (corpus acumulado da ingestão mensal; no modo
    snapshot único é o arquivo plano), padroniza nomes p/ UPPER, remove duplicatas exatas e por
    `dedup_col` (re-dump do mesmo registro -> partição mais recente vence). Guarda a tabela limpa
    INTEIRA (todas as colunas). A agregação por cliente é responsabilidade da Gold.
    """
    paths = config.bronze_partition_paths(input_path)
    logging.info(f"Sanitizando auxiliar {input_path}: lendo {len(paths)} partição(ões) Bronze.")

    if storage.STORAGE_BACKEND == "local":
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            logging.error(f"Arquivo(s) Bronze não encontrado(s): {missing}")
            raise FileNotFoundError(missing[0])

    df = pd.concat([storage.read_csv(p) for p in paths], ignore_index=True)
    df.columns = [col.strip().upper() for col in df.columns]

    before = df.shape[0]
    df = df.drop_duplicates()
    # Ingestão mensal: mesma linha de detalhe reaparecendo em drops diferentes -> mais recente
    # vence (partições lidas em ordem crescente de data; keep="last"). No snapshot único é no-op.
    if dedup_col in df.columns:
        df = df.drop_duplicates(subset=dedup_col, keep="last")
    if df.shape[0] < before:
        logging.info(f"Removidas {before - df.shape[0]} duplicatas ({dedup_col} recente vence).")

    # AMT_CREDIT_SUM_DEBT vem negativo em ~0,5% das linhas do Kaggle (saldo a favor/pré-pagamento).
    # Dívida em aberto não pode ser < 0; piso em 0 = "sem passivo" (limpeza, não agregação).
    if clip_debt and "AMT_CREDIT_SUM_DEBT" in df.columns:
        df["AMT_CREDIT_SUM_DEBT"] = df["AMT_CREDIT_SUM_DEBT"].clip(lower=0)

    storage.write_parquet(df, output_path)
    if snapshot_path:
        storage.write_parquet(df, snapshot_path)
        logging.info(f"Snapshot Silver salvo em: {snapshot_path}")
    logging.info(f"Auxiliar limpo salvo em: {output_path} com shape {df.shape}")
    return df


def sanitize_bureau(input_path: str, output_path: str) -> pd.DataFrame:
    """Bronze -> Silver do bureau.csv (1 linha por crédito externo, SK_ID_BUREAU)."""
    return _sanitize_aux(input_path, output_path, config.CLEAN_BUREAU_SNAPSHOT_PATH,
                         dedup_col="SK_ID_BUREAU", clip_debt=True)


def sanitize_previous_application(input_path: str, output_path: str) -> pd.DataFrame:
    """Bronze -> Silver do previous_application.csv (1 linha por pedido, SK_ID_PREV)."""
    return _sanitize_aux(input_path, output_path, config.CLEAN_PREV_SNAPSHOT_PATH,
                         dedup_col="SK_ID_PREV", clip_debt=False)


if __name__ == "__main__":
    sanitize_data(config.APPLICATION_PATH, config.CLEAN_PATH)
    sanitize_bureau(config.BUREAU_PATH, config.CLEAN_BUREAU_PATH)
    sanitize_previous_application(config.PREV_APP_PATH, config.CLEAN_PREV_PATH)
