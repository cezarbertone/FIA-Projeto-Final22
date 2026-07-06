"""
Configuração da camada de dados (DataPipeline).

Centraliza caminhos, parâmetros e metadados usados pelos scripts da
arquitetura medalhão (Bronze -> Silver -> Gold). Os scripts são executados
a partir da raiz do projeto, portanto os caminhos são relativos a ela.
"""
import os

# --- Caminhos (Arquitetura Medalhão — subpastas explícitas) ---
DATA_DIR = "Dados"
LANDING_DIR = os.path.join(DATA_DIR, "Landing")  # área de pouso do drop mensal (fonte da ingestão -> Bronze)
BRONZE_DIR = os.path.join(DATA_DIR, "Bronze")    # arquivos brutos do Kaggle (nomes originais)
SILVER_DIR = os.path.join(DATA_DIR, "Silver")    # dados limpos
GOLD_DIR = os.path.join(DATA_DIR, "Gold")        # ABT pronta p/ modelagem

APPLICATION_PATH = os.path.join(BRONZE_DIR, "application_train.csv")  # Bronze (base principal, CSV bruto)
CLEAN_PATH = os.path.join(SILVER_DIR, "clean_data.parquet")          # Silver (Parquet — data lake)
ABT_PATH = os.path.join(GOLD_DIR, "abt.parquet")                     # Gold (Parquet — data lake)

# Artefatos de transformação (para reproduzir a ABT em produção / predict.py)
ARTIFACTS_DIR = "DataPipeline"
ABT_ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR, "abt_artifacts.pkl")

# --- Tabelas auxiliares do Kaggle (histórico de crédito) ---
# Usadas para enriquecer a ABT com agregações por cliente (SK_ID_CURR). Ficam na Bronze
# junto da base principal (nomes originais do Kaggle).
BUREAU_PATH = os.path.join(BRONZE_DIR, "bureau.csv")                  # crédito em OUTRAS instituições (birô)
PREV_APP_PATH = os.path.join(BRONZE_DIR, "previous_application.csv")  # pedidos anteriores na Home Credit
# Silver dos auxiliares: tabelas limpas no NÍVEL DE REGISTRO ORIGINAL (1 linha/crédito, 1 linha/pedido).
# A limpeza (dedup, clip, UPPER) mora aqui; a agregação p/ nível de cliente é feita depois (Gold).
CLEAN_BUREAU_PATH = os.path.join(SILVER_DIR, "clean_bureau.parquet")
CLEAN_PREV_PATH = os.path.join(SILVER_DIR, "clean_previous_application.parquet")
# Metadado do Kaggle (dicionário de colunas: HomeCredit_columns_description.csv) — pequeno e VERSIONADO
# no git; usado só pelo gerador de dicionário (Tools/). Fica em Docs/ (documentação, fora das camadas
# de dados). NÃO confundir com LANDING_DIR (a fonte do drop mensal que a ingestão promove à Bronze).
AUX_DATA_DIR = os.path.join("Docs", "home-credit-default-risk")
# Liga/desliga o enriquecimento com agregações de bureau + previous_application.
USE_AUX_AGGREGATIONS = True

# --- Metadados do dataset ---
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# Valor anômalo do Home Credit que representa "sem emprego" (deve virar NaN)
DAYS_EMPLOYED_ANOMALY = 365243

# Colunas de tempo (em dias, negativas na origem) que serão convertidas para valor absoluto
TIME_COLS = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]

# --- Parâmetros de limpeza/transformação ---
# Limite de nulos para descartar uma coluna (fração)
MISSING_THRESHOLD = 0.50

# Colunas que NUNCA devem ser descartadas mesmo acima do limite de nulos
# (EXT_SOURCE_* estão entre as features mais preditivas do dataset)
KEEP_ALWAYS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# --- Versionamento por data de ingestão (cenário de ingestão mensal) ---
# Opt-in por env, no mesmo espírito de STORAGE_BACKEND:
#   INGESTION_DATE vazia (default) => comportamento de snapshot único (paths planos, como hoje).
#   INGESTION_DATE = "2026-07"      => além do arquivo plano ("latest"), grava um snapshot
#                                      versionado dt=<data>/ de cada camada DERIVADA, e a Bronze
#                                      é lida como a UNIÃO das partições dt=*/ (corpus acumulado).
# A Bronze é imutável e particionada por data (Dados/Bronze/dt=YYYY-MM/*.csv); as camadas
# derivadas mantêm o arquivo plano como ponteiro "latest" (serving lê o plano, sem saber da data).
INGESTION_DATE = os.environ.get("INGESTION_DATE", "").strip()


def _snap(path):
    """Caminho snapshot dt=<INGESTION_DATE>/ do arquivo (dt= inserido antes do nome).

    Ex.: "Dados/Gold/abt.parquet" -> "Dados/Gold/dt=2026-07/abt.parquet".
    Retorna None quando INGESTION_DATE está vazia (modo snapshot único = sem snapshot).
    """
    if not INGESTION_DATE:
        return None
    head, fname = os.path.split(path)
    return os.path.join(head, f"dt={INGESTION_DATE}", fname)


# Companheiros versionados das camadas derivadas (None no modo default = só arquivo plano)
CLEAN_SNAPSHOT_PATH = _snap(CLEAN_PATH)
CLEAN_BUREAU_SNAPSHOT_PATH = _snap(CLEAN_BUREAU_PATH)
CLEAN_PREV_SNAPSHOT_PATH = _snap(CLEAN_PREV_PATH)
ABT_SNAPSHOT_PATH = _snap(ABT_PATH)
ABT_ARTIFACTS_SNAPSHOT_PATH = _snap(ABT_ARTIFACTS_PATH)


def bronze_partition_paths(path):
    """Caminhos a ler para um arquivo Bronze: a UNIÃO das partições dt=*/ existentes.

    Ex.: "Dados/Bronze/application_train.csv" -> todos os
    "Dados/Bronze/dt=*/application_train.csv" (ordenados por data). Sem partições dt=*/
    (modo snapshot único) recai no arquivo plano informado.
    """
    head, fname = os.path.split(path)
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLOps")))
    import storage
    if storage.STORAGE_BACKEND == "minio":
        keys = storage.list_keys(head.replace(os.sep, "/") + "/dt=")
        parts = sorted(k for k in keys if k.rsplit("/", 1)[-1] == fname)
    else:
        import glob
        parts = sorted(glob.glob(os.path.join(head, "dt=*", fname)))
    return parts if parts else [path]
