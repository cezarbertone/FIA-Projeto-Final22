"""
Agregação das tabelas auxiliares do Kaggle (SK_ID_CURR).

A ABT tem 1 linha por solicitante. As tabelas de histórico de crédito têm várias linhas
por cliente, então precisam ser agregadas antes de juntar à ABT. Aqui tratamos duas fontes,
lidas já LIMPAS da camada Silver (nível de registro original, deduplicadas em data_sanitization.py):

- `clean_bureau.parquet`               -> histórico EXTERNO: créditos do cliente em OUTRAS
                                          instituições, reportados ao birô (tipo Serasa/SPC).
- `clean_previous_application.parquet` -> histórico INTERNO: pedidos de crédito anteriores do
                                          cliente na própria Home Credit (aprovados, recusados...).

Cada função retorna um DataFrame indexado por SK_ID_CURR com features prefixadas
(`BUREAU_*` / `PREV_*`). Cliente sem histórico fica fora do índice; o merge na ABT trata
a ausência como "sem histórico" (preenchido com 0 em abt_transform.py).

Aqui é SÓ agregação (groupby) — a limpeza (dedup, clip) já foi feita na Silver. Só pandas/numpy.
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


def aggregate_bureau(bureau_path: str) -> pd.DataFrame:
    """Agrega a Silver do bureau por cliente: volume, dívida, atrasos e recência do crédito externo."""
    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG"]
    # Silver já limpa/deduplicada (nível de registro: crédito); aqui só selecionamos as colunas e agregamos.
    b = storage.read_parquet(bureau_path)[cols]
    logging.info(f"clean_bureau: {len(b):,} linhas, {b['SK_ID_CURR'].nunique():,} clientes")

    b["_active"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["_closed"] = (b["CREDIT_ACTIVE"] == "Closed").astype(int)

    agg = b.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("_active", "sum"),
        BUREAU_CLOSED_COUNT=("_closed", "sum"),
        BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_AMT_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),   # crédito mais antigo
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),   # crédito mais recente
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_CNT_PROLONG_TOTAL=("CNT_CREDIT_PROLONG", "sum"),
    )
    # Razões derivadas (proteção contra divisão por zero -> NaN)
    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_CREDIT_COUNT"]
    agg["BUREAU_DEBT_CREDIT_RATIO"] = agg["BUREAU_AMT_DEBT_TOTAL"] / agg["BUREAU_AMT_CREDIT_SUM_TOTAL"]
    agg = agg.replace([np.inf, -np.inf], np.nan)

    logging.info(f"bureau agregado: {agg.shape[1]} features para {len(agg):,} clientes")
    return agg


def aggregate_previous_application(prev_path: str) -> pd.DataFrame:
    """Agrega a Silver do previous_application por cliente: histórico interno (Home Credit)."""
    cols = ["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_CREDIT",
            "AMT_APPLICATION", "AMT_DOWN_PAYMENT", "DAYS_DECISION", "CNT_PAYMENT"]
    # Silver já limpa/deduplicada (nível de registro: pedido); aqui só selecionamos as colunas e agregamos.
    p = storage.read_parquet(prev_path)[cols]
    logging.info(f"clean_previous_application: {len(p):,} linhas, {p['SK_ID_CURR'].nunique():,} clientes")

    p["_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    # Quanto recebeu vs. quanto pediu (≈1 => recebeu o que pediu; <1 => recebeu menos)
    p["_credit_app_ratio"] = (p["AMT_CREDIT"] / p["AMT_APPLICATION"]).replace([np.inf, -np.inf], np.nan)

    agg = p.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("_approved", "sum"),
        PREV_REFUSED_COUNT=("_refused", "sum"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_TOTAL=("AMT_CREDIT", "sum"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_CREDIT_APP_RATIO_MEAN=("_credit_app_ratio", "mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),   # pedido mais recente
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),   # pedido mais antigo
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),     # prazo médio (parcelas)
    )
    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_APP_COUNT"]
    agg["PREV_REFUSED_RATE"] = agg["PREV_REFUSED_COUNT"] / agg["PREV_APP_COUNT"]
    agg = agg.replace([np.inf, -np.inf], np.nan)

    logging.info(f"previous_application agregado: {agg.shape[1]} features para {len(agg):,} clientes")
    return agg


def build_aux_features(bureau_path: str, prev_path: str) -> pd.DataFrame:
    """Junta as agregações de bureau + previous_application num único DataFrame por SK_ID_CURR."""
    bureau = aggregate_bureau(bureau_path)
    prev = aggregate_previous_application(prev_path)
    features = bureau.join(prev, how="outer")   # union de clientes; ausências viram NaN
    logging.info(f"Features auxiliares totais: {features.shape[1]} colunas, {len(features):,} clientes")
    return features
