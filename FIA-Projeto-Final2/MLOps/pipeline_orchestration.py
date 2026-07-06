"""
Orquestração do pipeline via Apache Airflow (PythonOperator).

DAG que executa a INGESTÃO + a arquitetura medalhão + o treino, um task por etapa:
    ingest_bronze -> bronze_to_silver (sanitize) -> silver_to_gold (ABT) -> train_model

Cada task chama diretamente as funções dos scripts do projeto, reaproveitando o código
de DataPipeline/ e Model/ (sem duplicar lógica). Os caminhos vêm dos respectivos config.py.

Os módulos do projeto são carregados por caminho explícito (importlib), e não por
`import config`/`from train import ...`. Isso evita ambiguidade entre DataPipeline/config.py
e Model/config.py e dispensa o analisador estático de resolver imports que só existem no
sys.path em runtime (caso contrário acusa "Cannot find module 'config'").

Para usar no Airflow, aponte este arquivo (ou um symlink) para a pasta `dags/` do Airflow
e garanta que a raiz do projeto esteja acessível (volume montado no docker-compose).

Ingestão mensal versionada (pipeline sustentável): a DAG roda `@monthly`. A 1ª task, `ingest_bronze`,
promove o drop mensal da área de origem (`Docs/home-credit-default-risk/`) para a Bronze IMUTÁVEL na
partição `dt=<mês>/`; as etapas seguintes leem a Bronze como a UNIÃO das partições `dt=*/` (corpus
acumulado) e gravam, além dos arquivos planos ("latest" que app/API consomem), um snapshot `dt=<mês>/`
de cada camada derivada + modelo/métricas.

O MÊS da partição vem da `execution_date` da run (template `{{ ds }}` -> `_set_month` define a env
`INGESTION_DATE` antes de carregar o config): run agendada usa o mês da janela (backfill escolhe a
partição certa), trigger MANUAL usa o mês corrente. Execução local direta (sem Airflow, `ds=None`)
fica em modo plano, sem `dt=`. A Bronze deixou de ser semeada pelo compose (bootstrap) — quem a
popula agora é esta ingestão.
"""
import os
import sys
import importlib
import importlib.util
from datetime import datetime, timedelta

# Raiz do projeto e pastas de código no path (DAG roda de dentro de MLOps/ ou de dags/).
# O sys.path é necessário porque os módulos carregados fazem imports internos
# (ex.: `import config` dentro de data_sanitization.py) resolvidos em runtime.
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
for sub in ("DataPipeline", "Model"):
    p = os.path.join(PROJECT_ROOT, sub)
    if p not in sys.path:
        sys.path.append(p)


def _load(name, rel_path):
    """Carrega um módulo do projeto por caminho de arquivo explícito."""
    path = os.path.join(PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_in_project(func):
    """Executa func() com o cwd na raiz do projeto (paths dos configs são relativos)."""
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        return func()
    finally:
        os.chdir(cwd)


# ----- Resolução do mês da partição (dt=) -----
def _set_month(ds=None):
    """Define a env INGESTION_DATE (YYYY-MM) que dirige a partição dt= das camadas.

    Fonte da verdade em produção: a `execution_date` da DAG, recebida como `ds`='YYYY-MM-DD' via
    template do Airflow -> mês da janela processada (backfill escolhe a partição certa sozinho).
    Trigger MANUAL -> ds ≈ hoje -> mês corrente. Execução LOCAL direta (ds=None, sem Airflow):
    NÃO define a env -> modo plano, sem dt= (igual ao resto do modo local). Chamar ANTES de
    carregar o config (que lê a env no import).
    """
    if not ds:
        return None
    month = ds[:7]
    os.environ["INGESTION_DATE"] = month
    return month


# ----- Callables das tasks (também usáveis fora do Airflow, p/ teste local) -----
def task_ingest(ds=None):
    # Ingestão: promove o drop mensal da LANDING ZONE (Dados/Landing/) para a Bronze IMUTÁVEL, na
    # partição dt=<mês> (via storage -> filesystem local OU MinIO). É o passo que "recebe o dado"
    # antes de qualquer transformação; a Bronze é o destino, nunca a fonte.
    _set_month(ds)
    dp_config = _load("dp_config", "DataPipeline/config.py")
    storage = _load("mlops_storage", "MLOps/storage.py")
    import pandas as pd

    # (arquivo na landing, caminho-base na Bronze). _snap() insere dt=<mês>/ quando datado.
    drops = [
        ("application_train.csv", dp_config.APPLICATION_PATH),
        ("bureau.csv", dp_config.BUREAU_PATH),
        ("previous_application.csv", dp_config.PREV_APP_PATH),
    ]

    def _run():
        for fname, bronze_path in drops:
            src = os.path.join(dp_config.LANDING_DIR, fname)     # Dados/Landing/<arquivo> (drop mensal)
            dest = dp_config._snap(bronze_path) or bronze_path   # Dados/Bronze/dt=<mês>/<arquivo> (ou plano)
            storage.write_csv(pd.read_csv(src), dest)

    _run_in_project(_run)


def task_sanitize(ds=None):
    # Bronze -> Silver das TRÊS fontes (cada uma no nível de registro original, limpa/deduplicada).
    _set_month(ds)
    dp_config = _load("dp_config", "DataPipeline/config.py")
    sanit = _load("dp_data_sanitization", "DataPipeline/data_sanitization.py")

    def _run():
        sanit.sanitize_data(dp_config.APPLICATION_PATH, dp_config.CLEAN_PATH)
        sanit.sanitize_bureau(dp_config.BUREAU_PATH, dp_config.CLEAN_BUREAU_PATH)
        sanit.sanitize_previous_application(dp_config.PREV_APP_PATH, dp_config.CLEAN_PREV_PATH)

    _run_in_project(_run)


def task_build_abt(ds=None):
    _set_month(ds)
    dp_config = _load("dp_config", "DataPipeline/config.py")
    abt = _load("dp_abt_transform", "DataPipeline/abt_transform.py")
    _run_in_project(lambda: abt.create_abt(dp_config.CLEAN_PATH, dp_config.ABT_PATH,
                                           dp_config.ABT_ARTIFACTS_PATH))


def task_train(ds=None):
    # Treina, seleciona (CV), afina (GridSearchCV) e grava o modelo + métricas
    # (Model/model.pkl + metrics.json) consumidos por predict/app/api.
    _set_month(ds)
    train_mod = _load("model_train", "Model/train.py")
    _run_in_project(lambda: train_mod.train())


# ----- Definição da DAG (só é construída se o Airflow estiver instalado) -----
# import dinâmico: evita `from airflow import ...` estático (airflow não está no venv
# do pipeline; é instalado só no contexto de deploy/docker).
if importlib.util.find_spec("airflow") is not None:
    DAG = importlib.import_module("airflow").DAG
    PythonOperator = importlib.import_module("airflow.operators.python").PythonOperator

    default_args = {
        "owner": "labdata",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="fia-projeto-final-pipeline",
        description="Ingestão mensal + medalhão (bronze->silver->gold) + treino do score de inadimplência",
        default_args=default_args,
        start_date=datetime(2026, 1, 1),
        schedule_interval="@monthly",   # pipeline sustentável: 1 ciclo por mês (ingestão + reprocesso)
        catchup=False,
        tags=["home-credit", "credit-risk"],
    ) as dag:

        # ds='{{ ds }}' (execution_date da run) dirige a partição dt=<mês>. Trigger manual -> mês corrente.
        _ds = {"ds": "{{ ds }}"}
        t0 = PythonOperator(task_id="ingest_bronze", python_callable=task_ingest, op_kwargs=_ds)
        t1 = PythonOperator(task_id="bronze_to_silver", python_callable=task_sanitize, op_kwargs=_ds)
        t2 = PythonOperator(task_id="silver_to_gold", python_callable=task_build_abt, op_kwargs=_ds)
        t3 = PythonOperator(task_id="train_model", python_callable=task_train, op_kwargs=_ds)

        t0 >> t1 >> t2 >> t3
else:
    # Airflow não instalado (ex.: execução local) — permite testar os callables manualmente.
    dag = None


if __name__ == "__main__":
    # Execução sequencial local, sem Airflow, para validar o fluxo ponta a ponta.
    # Sem `ds` -> modo plano (sem partição dt=), coerente com o modo local dos scripts.
    task_ingest()
    task_sanitize()
    task_build_abt()
    task_train()
