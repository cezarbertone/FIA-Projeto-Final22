"""
Treinamento do modelo de Score de Risco de Inadimplência (Home Credit).

Fluxo:

    1. Separa a ABT em treino (80%) e holdout de teste (20%), estratificado pela TARGET.
    2. Compara 3 algoritmos do Scikit-Learn (LogisticRegression, RandomForest,
       GradientBoosting) com hiperparâmetros fixos, escolhendo o melhor pela AUC-ROC
       média em validação cruzada (StratifiedKFold) — feita só no treino.
    3. Faz a busca de hiperparâmetros (GridSearchCV) apenas no algoritmo vencedor.
    4. Re-treina o vencedor (já afinado) no conjunto de treino COMPLETO.
    5. Avalia UMA única vez no holdout (AUC-ROC, KS, recall) e salva modelo + métricas.

A seleção (passo 2) e a busca (passo 3) rodam sobre uma subamostra do treino só por
desempenho (ver config.SEARCH_SAMPLE_FRAC); o fit final e a avaliação usam tudo.
"""
import os
import sys
import logging
import importlib.util

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLOps")))
import storage

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# Carrega Model/config.py por caminho explícito com nome de módulo único ("model_config").
# Evita colisão com DataPipeline/config.py quando ambos estão no sys.path (ex.: Airflow).
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
_spec = importlib.util.spec_from_file_location("model_config", _cfg_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# GradientBoosting não tem `class_weight`: balanceamos via `sample_weight` no fit final.
NEEDS_SAMPLE_WEIGHT = {"gradient_boosting"}


def ks_statistic(y_true, y_score) -> float:
    """Estatística KS (separação entre bons e maus pagadores), padrão em crédito."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def build_candidates():
    """Modelos candidatos com hiperparâmetros fixos (de config.py).

    Scaling: só a LogisticRegression leva StandardScaler (modelo linear é sensível à escala). As
    árvores (RandomForest/GradientBoosting) são invariantes à escala — decidem por limiares, então
    padronizar seria desnecessário. O scaler fica DENTRO do Pipeline: é fitado só no treino (sem leakage) e
    salvo no model.pkl, de modo que o predict reaplica a mesma padronização automaticamente.
    LogReg e RandomForest tratam o desbalanceamento por `class_weight='balanced'`.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**config.LOGISTIC_PARAMS)),
        ]),
        "random_forest": RandomForestClassifier(**config.RANDOM_FOREST_PARAMS),
        "gradient_boosting": GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS),
    }


def fit_model(model, name, X, y):
    """Treina um candidato balanceando o GradientBoosting via sample_weight."""
    if name in NEEDS_SAMPLE_WEIGHT:
        sw = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Métricas no holdout (threshold padrão 0.5 para recall/precision)."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "ks": ks_statistic(y_test, y_proba),
        "recall_default": float(report["1"]["recall"]),
        "precision_default": float(report["1"]["precision"]),
        "accuracy": float(report["accuracy"]),
    }


def subsample(X, y, frac):
    """Subamostra estratificada para acelerar seleção/busca (None = usa tudo)."""
    if not frac:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=frac, random_state=config.RANDOM_STATE, stratify=y)
    return Xs, ys


def train():
    """Treina, seleciona, afina e persiste o melhor modelo (Model/model.pkl + metrics.json)."""
    # 1) Dados + split treino/holdout (estratificado)
    logging.info(f"Carregando ABT: {config.ABT_PATH}")
    df = storage.read_parquet(config.ABT_PATH)
    X = df.drop(columns=[c for c in (config.ID_COL, config.TARGET_COL) if c in df.columns])
    y = df[config.TARGET_COL]
    logging.info(f"Features: {X.shape[1]} | Amostras: {X.shape[0]} | Default rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    X_sel, y_sel = subsample(X_train, y_train, config.SEARCH_SAMPLE_FRAC)  # só p/ seleção/busca
    logging.info(f"Treino: {X_train.shape[0]} | Holdout: {X_test.shape[0]} | "
                 f"Subamostra seleção/busca: {X_sel.shape[0]}")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # 2) Seleção: AUC-ROC média em validação cruzada (no treino). O holdout NÃO é tocado aqui —
    #    só o vencedor será avaliado nele (passo 5), mantendo a estimativa final honesta.
    results = {}
    for name, model in build_candidates().items():
        cv_auc = cross_val_score(model, X_sel, y_sel, cv=cv, scoring="roc_auc").mean()
        results[name] = {"cv_auc": float(cv_auc)}
        logging.info(f"  {name} -> CV-AUC={cv_auc:.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_auc"])
    logging.info(f"Melhor algoritmo (CV-AUC): {best_name} ({results[best_name]['cv_auc']:.4f})")

    # 3) Busca de hiperparâmetros (GridSearchCV) só no vencedor, sobre a subamostra
    grid = config.SEARCH_GRIDS[best_name]
    logging.info(f"GridSearchCV em '{best_name}' ({grid})...")
    search = GridSearchCV(build_candidates()[best_name], grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_sel, y_sel)
    logging.info(f"Melhores params: {search.best_params_} (CV-AUC={search.best_score_:.4f})")

    # 4) Re-treina o vencedor afinado no TREINO COMPLETO
    best_model = fit_model(clone(search.best_estimator_), best_name, X_train, y_train)

    # 5) Avaliação final no holdout (atualiza as métricas do vencedor com a versão afinada)
    tuned = evaluate(best_model, X_test, y_test)
    tuned["cv_auc"] = float(search.best_score_)
    tuned["best_params"] = search.best_params_
    results[best_name] = tuned
    logging.info(f"'{best_name}' afinado -> holdout AUC={tuned['auc_roc']:.4f} | KS={tuned['ks']:.4f} "
                 f"| Recall(default)={tuned['recall_default']:.3f}")

    # Persistência (Model/model.pkl + Model/metrics.json — via camada de storage). Os caminhos
    # planos são o "latest" (app/api leem sempre eles); se a ingestão for datada, grava também
    # o snapshot dt=<data>/ do modelo e das métricas do mês (histórico p/ auditoria/rollback).
    model_bundle = {"model": best_model, "model_name": best_name,
                    "feature_columns": list(X.columns)}
    metrics_out = {"best_model": best_name, "results": results}
    storage.write_pickle(model_bundle, config.MODEL_PATH)
    storage.write_json(metrics_out, config.METRICS_PATH)
    logging.info(f"Modelo salvo em: {config.MODEL_PATH}")
    logging.info(f"Métricas salvas em: {config.METRICS_PATH}")
    if config.MODEL_SNAPSHOT_PATH:
        storage.write_pickle(model_bundle, config.MODEL_SNAPSHOT_PATH)
        storage.write_json(metrics_out, config.METRICS_SNAPSHOT_PATH)
        logging.info(f"Snapshot de modelo/métricas salvo em dt={config.INGESTION_DATE}/ "
                     f"({config.MODEL_SNAPSHOT_PATH})")


if __name__ == "__main__":
    train()
