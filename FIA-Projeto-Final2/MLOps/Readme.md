# MLOps — Arquitetura da Solução (Home Credit Default Risk)

Etapa individual: deploy e arquitetura da solução de **Score de Risco de Inadimplência**.

## 1. Arquitetura funcional (origem dos dados → serviço de predição)

```
                ┌───────────────────────── Apache Airflow — DAG @monthly ─────────────────────────┐
                │                                                                                  │
 Landing zone   │  ingest_bronze       bronze_to_silver       silver_to_gold        train_model    │
 (drop mensal)  │  (Landing → Bronze   data_sanitization.py   abt_transform.py       train.py      │
 Dados/Landing/ ─┼─►  dt=<mês>)    ──►  (Silver: 3 tabelas) ─► (Gold/abt+artifacts) ─►(model+metrics)│
                └───────────────────────────────────────────────────────────────────────┬─────────┘
                                                                                          │
                                          ┌───────────────────────────────────────────────┘
                                          ▼
                                 Model/predict.py  ──►  Serviço de predição + explicação
                                 (aplica ABT via artifacts)   ├─ Streamlit (App/app.py)
                                                              └─ API FastAPI (App/api.py: /predict, /explain)
```

> A DAG roda **`@monthly`**; a 1ª task **`ingest_bronze`** promove o drop da *landing zone*
> (`Dados/Landing/`) para a Bronze imutável na partição `dt=<mês>` (a Bronze é o destino, nunca a
> fonte). O **mês** vem da `execution_date` (trigger manual = mês corrente).

**Camadas (arquitetura medalhão):**
- **Bronze** `Dados/Bronze/` — CSVs brutos do Kaggle, nomes originais (`application_train.csv`,
  `bureau.csv`, `previous_application.csv`); **plana** por padrão ou **particionada** `dt=<mês>/` no
  cenário de ingestão mensal. Populada pela task `ingest_bronze` (a partir de `Docs/`).
- **Silver** `Dados/Silver/` — **três** tabelas limpas, cada uma no **nível de registro original** (`data_sanitization.py`):
  `clean_data.parquet` (application), `clean_bureau.parquet` e `clean_previous_application.parquet`. Toda
  fonte passa pela Silver; a limpeza dos auxiliares (dedup, clip) mora aqui, a **agregação** só na Gold.
- **Gold** `Dados/Gold/abt.parquet` — ABT pronta p/ modelo (`abt_transform.py`) + `abt_artifacts.pkl`.
  Inclui **agregações de histórico de crédito** — as Silver `clean_bureau`/`clean_previous_application`
  (via `feature_aggregation.py`) juntadas por `SK_ID_CURR` → 28 features `BUREAU_*`/`PREV_*` (113 colunas no total).
- **Modelo** `Model/model.pkl` + `Model/metrics.json` (`train.py`).
- **Serviço** `Model/predict.py` reaplica a transformação via artefatos e escora novas solicitações.

**Persistência — data lake MinIO (object storage S3-compatível).** Todas as camadas medalhão e os
artefatos (`model.pkl`, `abt_artifacts.pkl`, `metrics.json`) são lidos/gravados via uma **camada de
I/O abstrata** (`MLOps/storage.py`) que troca o backend por env `STORAGE_BACKEND`:
- `STORAGE_BACKEND=local` (default) → filesystem, em `Dados/…` (modo de desenvolvimento).
- `STORAGE_BACKEND=minio` → **data lake MinIO**, mesmo caminho como *object key* num bucket único.

Bronze segue em **CSV** (bruto do Kaggle); Silver/Gold em **Parquet** (colunar, compacto). Os scripts
do pipeline não mudam de caminho — só o backend. Ver a seção [3 → Data lake (MinIO)](#data-lake-minio).

**Versionamento por data de ingestão (cenário de ingestão mensal).** Por padrão cada camada é
um arquivo único que a execução sobrescreve (snapshot único). Para o cenário de **receber a base 1×/mês**
(clientes novos já com `TARGET`), a env **`INGESTION_DATE=YYYY-MM`** liga o versionamento:
- **Bronze particionada e imutável** — cada drop mensal é uma pasta nova `Dados/Bronze/dt=YYYY-MM/`
  (populada pela task `ingest_bronze`); o pipeline **lê a união** das partições `dt=*/` (corpus
  **acumulado/crescente**) e regenera a ABT do zero.
- **Snapshot mensal das camadas derivadas** — Silver (×3)/Gold/artefatos/modelo/métricas gravam, além do
  arquivo plano (o **"latest"** que o serving lê), uma cópia `dt=YYYY-MM/` (histórico → auditoria e
  rollback: para voltar ao modelo do mês anterior, copie `Model/dt=<mês>/model.pkl` sobre o plano).
- **Mês da partição:** rodando **pela DAG do Airflow**, o `INGESTION_DATE` vem da **`execution_date`**
  (não é fixo): run `@monthly` usa o mês da janela (backfill escolhe a partição certa) e trigger manual
  usa o mês corrente. Rodando **os scripts à mão** (sem Airflow), a env vazia (default) mantém o
  comportamento **plano** (snapshot único). O serving (`predict.py`/`app.py`/`api.py`) sempre lê o
  caminho plano, sem saber da data.

## 2. Componentes
| Componente | Tecnologia | Papel |
|---|---|---|
| Orquestração | Apache Airflow (PythonOperator, SequentialExecutor + SQLite) | `pipeline_orchestration.py` — DAG `@monthly` ingest→bronze→silver→gold→treino (mês da `execution_date`) |
| Persistência | MinIO (object storage S3) + boto3 / pyarrow | data lake das camadas medalhão (Silver/Gold em Parquet) e artefatos; I/O abstrata em `MLOps/storage.py` |
| Processamento | pandas / numpy | limpeza, feature engineering, ABT |
| Modelagem | scikit-learn (seleção manual) | LogisticRegression / RandomForest / GradientBoosting; seleção por CV-AUC + `GridSearchCV` no vencedor |
| Serviço | Streamlit (`App/app.py`) + FastAPI (`App/api.py`) | interface de scoring + explicação; API REST (`/predict`, `/explain`) |
| Explicabilidade | SHAP (`TreeExplainer`) + fallback por occlusion | top fatores da decisão no app e no endpoint `/explain` |
| Infra | docker-compose | sobe Airflow + App + API |

### 2.1 Modelos e algoritmos

Três candidatos, todos de **Machine Learning clássico (supervisionado, classificação binária)** do
Scikit-Learn; seleção **manual** pelo **AUC-ROC** (sem AutoML). Hiperparâmetros explícitos em
`Model/config.py`.

| Algoritmo | Família | Paradigma | Hiperparâmetros-chave | Papel |
|---|---|---|---|---|
| **LogisticRegression** | Linear / GLM | ML clássico | `StandardScaler`, `class_weight='balanced'`, `max_iter=1000` | Baseline interpretável |
| **RandomForest** | Ensemble — *bagging* | ML clássico | `max_depth=8`, `min_samples_leaf=50`, `n_estimators=200`, `class_weight='balanced'` | Não-linearidades/interações |
| **GradientBoosting** | Ensemble — *boosting* | ML clássico | `max_depth=3`, `learning_rate=0.1`, `n_estimators=200`, `sample_weight` | **Melhor modelo** (AUC 0.773, KS 0.407) |

**Por que ML clássico e não Deep Learning:** problema **tabular** (~83 features), onde árvores/ensembles
igualam ou superam redes neurais com menor custo; **interpretabilidade** exigida em crédito (regulado).
O desbalanceamento (~8%) é tratado com `class_weight='balanced'` / `sample_weight`, não com reamostragem.
**Redes neurais (Deep Learning), avaliadas e não adotadas como principal:**
- **Densa (MLP):** aplicável a classificação tabular, mas ensembles de árvore costumam
  igualá-la/superá-la em dado tabular, com melhor interpretabilidade — fica como baseline comparativo futuro.
- **Convolucional (CNN):** não se aplica (dado espacial/imagem).
- **Recorrente (RNN/LSTM):** não se aplica (sequência/série temporal; aqui cada pedido é snapshot estático).

> A seleção do melhor é por **AUC-ROC em validação cruzada** (`StratifiedKFold`, 5 folds) e o vencedor
> passa por **`GridSearchCV`** (busca de hiperparâmetros), sendo então re-treinado no treino completo e
> avaliado uma única vez no holdout. Detalhamento completo (justificativa por modelo, controle de
> overfitting, busca de hiperparâmetros e a análise das redes neurais) no [`Readme.md`](../Readme.md)
> principal, seção "🤖 Modelos e Algoritmos".

## 3. Como executar

### Pipeline + treino (local, sem Airflow)
```bash
python MLOps/pipeline_orchestration.py   # ingest_bronze -> bronze_to_silver -> silver_to_gold -> train_model
```
> Local (sem Airflow) roda em **modo plano** (sem partição `dt=`): a ingestão promove `Docs/` → Bronze
> plana e as camadas gravam só o arquivo "latest".

### Infra completa (Airflow + App + API + MinIO) via Docker
```bash
cd MLOps
docker compose up --build
# MinIO Console: http://localhost:9001  (minioadmin/minioadmin)
# Airflow UI:    http://localhost:8080  (airflow/airflow)
# App Streamlit: http://localhost:8501
# API (Swagger): http://localhost:8000/docs
```
Rodar o pipeline pela DAG (após o compose subir):
```bash
# via UI: ativar/trigger a DAG fia-projeto-final-pipeline em http://localhost:8080
# ou via CLI:
docker compose exec airflow airflow dags trigger fia-projeto-final-pipeline
```
> A DAG (`@monthly`) **ingere a Bronze** (`ingest_bronze`), reconstrói Silver/Gold e roda o treino
> (seleção por CV-AUC + `GridSearchCV` no vencedor), gravando o `Model/model.pkl` + métricas no lake,
> consumidos por app/API. Rodando pela DAG, cada camada também ganha o snapshot `dt=<mês>` (mês da
> `execution_date`). O mesmo fluxo roda local (plano), sem Airflow, via
> `python MLOps/pipeline_orchestration.py`.

<a id="data-lake-minio"></a>
### Data lake (MinIO)
No `docker compose`, os serviços rodam com `STORAGE_BACKEND=minio` e leem/gravam **todas as camadas
medalhão + artefatos** no MinIO (bucket `fia-projeto-final`). O job `bootstrap` (imagem `minio/mc`;
renomeado de `createbuckets`), no boot: espera o MinIO (`mc ready`), cria o bucket (idempotente) **e**
sobe — via `mc cp`, a partir de volumes read-only — **apenas os artefatos de serving versionados**
(`Model/model.pkl`, `DataPipeline/abt_artifacts.pkl`). Assim o **app/API já escoram logo após o `up`**,
antes mesmo da DAG. **A Bronze não é semeada aqui:** quem a popula é a task `ingest_bronze` da DAG
(ingestão de verdade, `Docs/` → Bronze `dt=<mês>`). `airflow`/`app`/`api` dependem da **conclusão** do
`bootstrap` (`service_completed_successfully`), então só sobem com o bucket pronto e os artefatos no
lake. Fora do compose (scripts soltos sem as envs), o backend é `local` (filesystem) — nada muda nos
caminhos, só onde os bytes residem.

Subir tudo já apontando para o lake (bucket + artefatos de serving populados automaticamente):
```bash
cd MLOps
docker compose up -d   # minio -> bootstrap (bucket + model.pkl/abt_artifacts.pkl) -> airflow/app/api
```
- **Pré-requisito:** o **drop de origem** na *landing zone* `Dados/Landing/` (a task `ingest_bronze` o
  promove à Bronze quando a DAG rodar). O `bootstrap` não depende da Bronze — só do modelo/artefatos.
- Os artefatos `Model/model.pkl` e `DataPipeline/abt_artifacts.pkl` enviados no boot são os
  **versionados no repo**, que deixam o app/API funcionais de imediato. Ao **rodar a DAG**, a Bronze é
  ingerida na partição do mês, o `Model/model.pkl` do lake é **sobrescrito** pelo modelo recém-treinado,
  e as camadas `Dados/Silver/*` (as 3 tabelas), `Dados/Gold/abt.parquet`, `Model/metrics.json` passam a
  residir no bucket — com os snapshots `dt=<mês>/` ao lado dos planos (visíveis no console `:9001`).

### Apenas o app / a API de predição (local)
```bash
streamlit run App/app.py                 # app Streamlit (8501)
uvicorn App.api:app --port 8000          # API FastAPI (8000) -> /docs
```

### Deploy do app via Docker (imagem própria)
A imagem é definida em [`App/Dockerfile`](../App/Dockerfile) (base **`python:3.12-slim`**) e construída a
partir da **raiz** do projeto. Usa-se 3.12 (e não 3.14) porque `shap`/`numba` (explicação da decisão)
ainda não têm wheel para 3.14 no Linux; a compatibilidade do `pickle` é garantida pelas **versões das
libs** (não pelo Python), fixadas em [`App/requirements.txt`](../App/requirements.txt):
```bash
docker build -f App/Dockerfile -t hc-app .
docker run -p 8501:8501 hc-app                                   # Streamlit
docker run -p 8000:8000 hc-app uvicorn App.api:app --host 0.0.0.0 --port 8000   # API
```

**O que a imagem precisa conter (artefatos do modelo):**

| Artefato | Papel |
|---|---|
| `Model/model.pkl` | Modelo treinado (bundle `model` + `model_name`). |
| `DataPipeline/abt_artifacts.pkl` | Reproduz a transformação da ABT (colunas, imputação, encoding, ordem das features). **Indispensável.** |
| `Model/predict.py`, `Model/config.py`, `App/app.py`, `App/api.py` | Código de scoring/explicação + interface (Streamlit) + API (FastAPI). |

- **NÃO** entram os dados (`raw/clean/abt.csv`) nem os scripts de pipeline: o `predict.py` reimplementa
  a transformação e usa só os artefatos acima. Para uma solicitação nova, as features de histórico
  (`BUREAU_*`/`PREV_*`) entram como 0 (sem histórico).
- `model.pkl` e `abt_artifacts.pkl` devem vir do **mesmo treino**. A compatibilidade do `pickle.load`
  é garantida pelas **versões das libs** pinadas em [`App/requirements.txt`](../App/requirements.txt)
  (`scikit-learn`/`pandas` iguais às do treino; `numpy 2.4.6` no serving, pois `numba`/`shap` exigem
  `numpy<2.5`) — não pela versão do Python (treino em 3.14, serving em 3.12). Validado: a carga do
  `model.pkl` (feito em numpy 2.5) sob numpy 2.4.6 produz previsão idêntica.
- No `docker compose` (`STORAGE_BACKEND=minio`), os serviços `app`/`api` leem o `Model/model.pkl`
  **do data lake (MinIO)** — não do filesystem. Esse `model.pkl` é gravado no lake pela **DAG do
  Airflow** (task de treino). Ou seja: suba o compose → rode a DAG (gera o modelo no bucket) → o
  app/API já o consomem.

## 4. Monitoramento (dados e modelo em produção)
- **Qualidade de dados (entrada):** % de nulos por coluna, faixas válidas (renda/crédito > 0),
  categorias novas (encoding → -1), volume diário. Alerta se desviar do baseline da ABT.
- **Data drift:** comparar distribuição das features em produção vs. treino (ex.: PSI/KS por feature);
  alerta quando PSI > 0.2 nas variáveis mais importantes (`EXT_SOURCE_*`, razões financeiras).
- **Performance do modelo:** acompanhar AUC-ROC/KS quando o desfecho (pagou/não pagou) amadurece;
  monitorar taxa de aprovação e taxa de inadimplência aprovada (proxy de Falsos Negativos).
- **Operacional:** latência/erros do serviço de predição, sucesso/falha das DAGs no Airflow.

### Quando e como retreinar o modelo (ingestão mensal)

> **Ideia:** todo mês chega uma base nova de clientes **já com o resultado real** (pagou / não pagou).
> Como temos o gabarito, corrigimos a "prova" do modelo atual: se ele começou a errar mais, retreina;
> se continua acertando, mantém.

**Analogia:** o modelo é como um analista que aprova crédito. Todo mês conferimos as decisões dele
contra o que **de fato** aconteceu. Enquanto acerta, não se mexe; quando passa a errar demais, é sinal
de que o perfil dos clientes mudou e ele precisa **estudar de novo** com os dados mais recentes.

**Passo a passo:**
1. **Chega o mês novo** (clientes novos, já rotulados com quem pagou / não pagou).
2. **Aplica-se o modelo atual** nesses clientes e compara-se a previsão com o resultado real.
3. **Mede-se a "nota"** com as duas métricas que já usamos: **AUC-ROC** e **KS** (separação entre bom e
   mau pagador).
4. **Decisão (gatilho):** nota **continua boa** → **não retreina**; nota **caiu** além de uma margem →
   **retreina** com todos os dados acumulados (corpus crescente).
5. **Guarda-se a nota do mês** num histórico para acompanhar a evolução (gráfico de AUC/KS mês a mês).

**Por que não retreinar todo mês:** retreinar por reflexo gasta recurso e pode até **piorar** o modelo.
Retreinar **só quando há evidência de degradação** (vinda do resultado real que chega com a base nova)
é o que torna o pipeline **monitorado e sustentável**, não "treina e esquece".

> ℹ️ **Status:** o **versionamento** que sustenta isto (Bronze `dt=`, corpus acumulado, snapshots
> mensais) **já está implementado** (ver seção 1, "Versionamento por data de ingestão"). A **decisão
> automática de retreino** acima é o
> **próximo passo** (desenho) — hoje o pipeline retreina a cada execução, sem medir a degradação.

## 5. Ações automatizadas a partir das previsões (ML + automação + agentes)
- **Score alto (≥ threshold):** rota automática para revisão manual / pedido de garantias adicionais.
- **Score baixo:** aprovação automática e oferta de limite pré-aprovado.
- **Drift detectado:** trigger automático de re-treino da DAG e notificação ao time de risco.
- **Agente de IA:** geração automática de justificativa da decisão — já temos a base via **SHAP**
  (endpoint `/explain` e app retornam os principais fatores), pronta para o cliente e para
  auditoria/governança.

## 6. Próximos passos (itens iii e iv do escopo individual)
- **Decisão automática de retreino (ingestão mensal)** — medir AUC-ROC/KS do modelo atual no mês novo
  rotulado e retreinar só se degradar (ver seção 4). O versionamento que habilita isto já está pronto;
  falta a task de avaliação + gatilho na DAG. Cuidado de leakage: julgar o modelo retreinado num mês
  ainda não visto (o desafiante já treinou no mês do gatilho).
- Implementar coleta de métricas de drift (job agendado) e dashboard de monitoramento.
- Versionamento de modelo/artefatos (registry) e CI/CD do re-treino.
- Política formal de threshold por apetite de risco e testes A/B de cutoffs.
- **Camada transacional Postgres** (serving log + histórico de drift): registrar cada predição da
  API/app (auditoria) e provisionar a tabela de drift. Desenhada junto do data lake, mas fora do
  escopo desta entrega (só o MinIO foi implementado).
- ~~Endpoint FastAPI~~ ✅ **concluído** (`App/api.py`: `/predict`, `/predict/batch`, `/explain`).
- ~~Persistência em data lake (MinIO)~~ ✅ **concluído** (`MLOps/storage.py` + serviço `minio` no compose).
