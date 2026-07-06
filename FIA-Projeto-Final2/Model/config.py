"""
Configuração da camada de modelagem (Model).

Centraliza caminhos, variáveis-alvo, hiperparâmetros e metadados do treinamento.
Os scripts são executados a partir da raiz do projeto.
"""
import os

# --- Caminhos ---
DATA_DIR = "Dados"
ABT_PATH = os.path.join(DATA_DIR, "Gold", "abt.parquet")   # camada Gold (Parquet — data lake)

# Artefatos gerados por `train.py` (usados por predict/app/api).
MODEL_DIR = "Model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")          # melhor modelo serializado
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")     # métricas da avaliação

# --- Versionamento por data de ingestão (ver DataPipeline/config.py) ---
# Mesma env opt-in: preenchida => além do model.pkl/metrics.json planos ("latest"), grava
# um snapshot versionado dt=<data>/ do modelo e das métricas daquele mês (histórico/rollback).
INGESTION_DATE = os.environ.get("INGESTION_DATE", "").strip()


def _snap(path):
    """Caminho snapshot dt=<INGESTION_DATE>/ (None no modo snapshot único)."""
    if not INGESTION_DATE:
        return None
    head, fname = os.path.split(path)
    return os.path.join(head, f"dt={INGESTION_DATE}", fname)


MODEL_SNAPSHOT_PATH = _snap(MODEL_PATH)
METRICS_SNAPSHOT_PATH = _snap(METRICS_PATH)

# --- Metadados ---
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# --- Parâmetros de treino ---
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Hiperparâmetros dos modelos candidatos (controle de overfitting)
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

# --- Validação cruzada na seleção de modelos ---
# Os candidatos são comparados pela AUC-ROC média em StratifiedKFold (mais robusto que um
# holdout único). O modelo escolhido nunca "olha" o holdout para ser selecionado.
CV_FOLDS = 5

# --- Subamostra para a etapa de seleção/busca ---
# A seleção por CV e o GridSearchCV rodam sobre uma fração do treino apenas por DESEMPENHO
# (o GradientBoosting é single-thread e cada fit no treino completo leva minutos). Escolher
# o algoritmo e os hiperparâmetros não exige o dataset inteiro. O FIT FINAL do modelo vencedor
# e a avaliação no holdout usam sempre os conjuntos completos. (None = usar todo o treino.)
SEARCH_SAMPLE_FRAC = 0.30

# --- Busca de hiperparâmetros (GridSearchCV) ---
# Aplicada apenas ao modelo VENCEDOR da seleção por CV. Cada modelo tem seu próprio espaço de
# busca. Para o LogisticRegression (dentro de Pipeline), os params levam o prefixo do passo:
# `clf__`.
SEARCH_GRIDS = {
    "gradient_boosting": {
        "n_estimators": [150, 200],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
    },
    "random_forest": {
        "n_estimators": [200, 300],
        "max_depth": [8, 12],
        "min_samples_leaf": [20, 50],
    },
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
}
