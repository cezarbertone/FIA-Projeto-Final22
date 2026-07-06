# Home Credit Default Risk - Data Pipeline

Este repositório contém a estruturação inicial do projeto final do MBA em Big Data e Analytics, focado em resolver o problema de **Previsão de Risco de Inadimplência (Default Risk)** utilizando os dados do desafio do Kaggle (Home Credit).

## O Problema de Negócio

**Score de risco de inadimplência na concessão de crédito.**

O objetivo é prever, no momento da solicitação do empréstimo, se o cliente terá dificuldades para honrar com os pagamentos (variável alvo `TARGET`: `0` = paga em dia, `1` = inadimplente). A resolução deste problema ajuda a instituição que utilizará o modelo a diminuir prejuízos com não pagamentos e otimizar a concessão de crédito, apoiando a decisão de **aprovar / negar / ajustar o limite**.

A base é desbalanceada: a taxa de inadimplência (*default rate*) é de **~8%**.

### Métricas de sucesso

**Técnicas — medem a qualidade do modelo:**
- **AUC-ROC** (principal): o quanto o modelo acerta ao ordenar os clientes do menos para o mais arriscado.
- **Recall da classe inadimplente:** dos clientes que não pagariam, quantos o modelo consegue identificar.
- **KS:** o quanto o modelo separa bons e maus pagadores (métrica padrão em crédito).

**De negócio — mede o impacto real:**
- Reduzir os **Falsos Negativos** (aprovar quem não vai pagar) — é o prejuízo direto que queremos evitar.

> A base é desbalanceada (~8% de inadimplentes). Tratamos isso com `class_weight='balanced'`, que dá mais
> peso aos casos raros — sem criar dados artificiais.

---

## 🗂️ Dados utilizados (escopo do dataset Kaggle)

O desafio Home Credit no Kaggle disponibiliza **8 tabelas**. Usamos **três**: a base de solicitações
(`application_train`) mais duas tabelas de histórico de crédito (`bureau` e `previous_application`),
agregadas por cliente para enriquecer a ABT.

| Arquivo Kaggle | Usado? | Papel |
|---|---|---|
| **`application_train.csv`** | ✅ **Sim** | **Base** da ABT, na camada Bronze (`Dados/Bronze/application_train.csv`). Features da solicitação + rótulo `TARGET`. |
| **`bureau.csv`** | ✅ **Sim** | Histórico **externo**: créditos do cliente em **outras instituições** (birô de crédito, tipo Serasa/SPC). Agregado por cliente → 15 features `BUREAU_*`. |
| **`previous_application.csv`** | ✅ **Sim** | Histórico **interno**: pedidos de crédito anteriores na **própria Home Credit**. Agregado por cliente → 13 features `PREV_*`. |
| `application_test.csv` | ❌ Não | Conjunto de **submissão da competição, sem `TARGET`**. Avaliamos com **holdout estratificado (20%) do próprio train**, então não é necessário. |
| `bureau_balance.csv` | ❌ Não | Saldo **mensal** de cada crédito do birô (27M linhas) — exigiria agregação em 2 níveis. Extensão futura. |
| `credit_card_balance`, `installments_payments`, `POS_CASH_balance` | ❌ Não | Detalhe **mensal** dos contratos internos (saldo de cartão, parcelas, atrasos). Extensão futura. |

### Sobre as chaves (`SK_ID_*`)

O dicionário oficial define `SK_ID_CURR` como *"ID of loan in our sample"* — ou seja, é o **ID da
solicitação atual** (o empréstimo em análise), e **não** um ID de pessoa persistente. No
`application_train` há **exatamente 1 linha por `SK_ID_CURR`** (307.511 linhas = 307.511 ids), então
cada `SK_ID_CURR` representa **uma solicitação** e serve de chave para anexar o histórico daquele cliente.

| Chave | Identifica |
|---|---|
| `SK_ID_CURR` | A **solicitação atual** (1 por linha no `application_train`; chave de junção do histórico). |
| `SK_ID_PREV` | Um **pedido anterior** do cliente na Home Credit (`previous_application`). |
| `SK_ID_BUREAU` | Um **crédito registrado no birô** / outra instituição (`bureau`). |

> ⚠️ Não há ID global de pessoa: não dá para cruzar o mesmo indivíduo entre solicitações atuais
> diferentes. As agregações ligam o histórico **já pré-vinculado pelo Kaggle** a cada `SK_ID_CURR`.

---

## 🚀 Como Iniciar o Ambiente (Passo a Passo)

Siga estas instruções caso esteja executando pela primeira vez em sua máquina local:

### 1. Criar o Ambiente Virtual
O ambiente virtual isola as bibliotecas do projeto para não interferir no Python do seu sistema.
Abra o **PowerShell** na pasta do projeto (a raiz onde você clonou o repositório) e digite:
```powershell
python -m venv .venv
```

### 2. Ativar o Ambiente Virtual
```powershell
.venv\Scripts\Activate.ps1
```
*(Você verá um `(.venv)` no início da linha de comando, indicando que funcionou).*

### 3. Atualizar o Pip e Instalar as Bibliotecas Necessárias
Com o ambiente ativado, é recomendado primeiro atualizar o gerenciador de pacotes (`pip`):
```powershell
python.exe -m pip install --upgrade pip
```

Em seguida, instale as dependências do projeto executando:
```powershell
pip install -r requirements.txt
```

> Para rodar o **app Streamlit** localmente, instale também: `pip install streamlit`
> (ferramenta de deploy; fica fora do `requirements.txt` do pipeline de ML).

---

## 👥 Setup (após clonar o repositório)

Quem clona o repositório **não recebe os dados** (`Dados/**` — a *landing zone* `Dados/Landing/`, a
Bronze em `*.csv`, Silver/Gold em `*.parquet` — e as fontes em `Docs/home-credit-default-risk/` estão no
`.gitignore` por excederem o limite de 100 MB do GitHub). **Exceção versionada:** o metadado
`Docs/home-credit-default-risk/HomeCredit_columns_description.csv` (~37 KB, usado pelo gerador de dicionário).
Siga este checklist para deixar o ambiente pronto:

1. **Clonar e entrar na pasta** (troque pela URL do seu próprio repositório, se for o caso)
   ```powershell
   git clone https://github.com/guilhermepepa/FIA-Projeto-Final2.git
   cd FIA-Projeto-Final2
   ```
2. **Criar e ativar o venv + instalar dependências** (passos 1 a 3 da seção anterior).
3. **Baixar os dados do Kaggle** — competição *Home Credit Default Risk*
   (https://www.kaggle.com/c/home-credit-default-risk/data). As tabelas usadas são:
   `application_train.csv`, `bureau.csv` e `previous_application.csv`.
4. **Colocar os CSVs do Kaggle em `Dados/Landing/`** (a *landing zone* — nomes originais:
   `application_train.csv`, `bureau.csv`, `previous_application.csv`). Esta é a **área de pouso** do drop
   mensal. **Para rodar via Airflow (ou `python MLOps/pipeline_orchestration.py`), é só isto:** a task
   `ingest_bronze` lê daqui e promove os arquivos à camada Bronze (`Dados/Bronze/dt=<mês>/`)
   automaticamente — **não** é preciso copiar nada para `Dados/Bronze/` à mão.

   > **Exceção — rodar os scripts do `DataPipeline/` isoladamente** (Opção A abaixo, sem passar pela
   > ingestão): esses scripts leem a Bronze diretamente, então copie os CSVs de `Dados/Landing/` para
   > `Dados/Bronze/`:
   > ```powershell
   > Copy-Item "Dados/Landing/application_train.csv"     "Dados/Bronze/"
   > Copy-Item "Dados/Landing/bureau.csv"                "Dados/Bronze/"
   > Copy-Item "Dados/Landing/previous_application.csv"  "Dados/Bronze/"
   > ```

A partir daqui há **duas formas de persistir e rodar** o pipeline — escolha **uma**. A lógica é a
mesma; muda só **onde** os dados ficam, controlado pela env **`STORAGE_BACKEND`** (`local` | `minio`)
e implementado em [`MLOps/storage.py`](MLOps/storage.py).

#### 🅰️ Opção A — Desenvolvimento local (sem MinIO)  *(padrão, mais simples)*
Lê/grava as camadas direto no filesystem, em `Dados/` (Bronze CSV; Silver/Gold Parquet). **Não exige
Docker.** É o `STORAGE_BACKEND=local` (default — nenhuma env precisa ser setada).
```powershell
# a partir da raiz do projeto, com o venv ativo
python DataPipeline/data_sanitization.py   # Bronze -> Silver (limpa as 3 fontes: clean_data + clean_bureau + clean_previous_application)
python DataPipeline/abt_transform.py        # Silver -> Gold  (Dados/Gold/abt.parquet, agrega os auxiliares limpos)
python Model/train.py                        # (opcional) retreina; model.pkl/metrics.json já vêm versionados
```

#### 🅱️ Opção B — Data lake MinIO (object storage S3)
Lê/grava **as mesmas camadas + artefatos** como objetos no bucket `fia-projeto-final` do MinIO.
Os caminhos não mudam — só o backend (`STORAGE_BACKEND=minio`). Requer o MinIO no ar (via Docker).
```powershell
# 1) sobe o MinIO; o job bootstrap cria o bucket e sobe os artefatos de serving (model/abt_artifacts).
#    A Bronze NÃO é semeada aqui — quem a popula é a ingestão (passo 3).
cd MLOps; docker compose up -d minio bootstrap; cd ..

# 2) configura as envs do lake na sessão atual (PowerShell)
$env:STORAGE_BACKEND="minio"; $env:MINIO_ENDPOINT="http://localhost:9000"
$env:MINIO_ACCESS_KEY="minioadmin"; $env:MINIO_SECRET_KEY="minioadmin"; $env:MINIO_BUCKET="fia-projeto-final"

# 3) roda o pipeline COMPLETO (a 1ª etapa ingere Docs/ -> Bronze no bucket) — tudo passa a residir no lake
python MLOps/pipeline_orchestration.py   # ingest -> silver -> gold -> train (Bronze + Silver/Gold + artefatos)
```
> Console do MinIO em `http://localhost:9001` (minioadmin/minioadmin) para inspecionar os objetos.
> Pré-requisito: ter os CSVs do Kaggle em `Dados/Landing/` (a ingestão os promove ao lake).
> Para subir **tudo** já apontando para o lake (Airflow + app + API), veja a seção **🛰️ Serviço de
> Predição e Orquestração (MLOps)** abaixo. Detalhes do data lake em
> [`MLOps/Readme.md` → Data lake (MinIO)](MLOps/Readme.md#data-lake-minio).

---

## ⚙️ Executando o Pipeline Manualmente

Atualmente, o projeto utiliza a **Arquitetura Medalhão** (Camadas Bronze, Silver e Gold), que pode ser
orquestrado automaticamente pelo **Apache Airflow**. Por enquanto, a execução manual funciona assim:

> Os comandos abaixo detalham a **Opção A (local)**. Para rodar no **data lake MinIO (Opção B)**, basta
> definir as envs `STORAGE_BACKEND=minio` + `MINIO_*` antes destes mesmos comandos (ver Setup → Opção B);
> os scripts e caminhos são idênticos.

### Passo 1: Limpeza de Dados (Bronze para Silver)
O primeiro script limpa **as três fontes** do Kaggle, cada uma no seu **nível de registro original** (a
Silver não muda o nível de registro — só padroniza nomes, remove anomalias conhecidas e deduplica).
* **Comando:**
  ```powershell
  python DataPipeline\data_sanitization.py
  ```
* **O que acontece:** lê `Dados/Bronze/{application_train,bureau,previous_application}.csv` e gera as três
  tabelas limpas da Camada Silver: `clean_data.parquet` (1 linha/solicitante), `clean_bureau.parquet`
  (1 linha/crédito externo) e `clean_previous_application.parquet` (1 linha/pedido). A limpeza dos
  auxiliares (dedup por `SK_ID_BUREAU`/`SK_ID_PREV`, piso 0 na dívida) fica **aqui**; a **agregação** por
  cliente é o Passo 2 (Gold). Assim toda fonte passa pela Silver antes de virar feature.

### Passo 2: Geração da ABT (Silver para Gold)
O segundo script transforma o dado limpo na **Analytical Base Table (ABT)**, que é a tabela final já com a engenharia de features, tratamento de valores nulos e codificação de texto, pronta para entrar no modelo de Machine Learning.
* **Comando:**
  ```powershell
  python DataPipeline\abt_transform.py
  ```
* **O que acontece:** Ele lê `Dados/Silver/clean_data.parquet`, cria as razões financeiras, **agrega o histórico
  de crédito** (`Silver/clean_bureau.parquet` + `Silver/clean_previous_application.parquet`, via
  `feature_aggregation.py`) por `SK_ID_CURR`, trata nulos/encoding e gera `Dados/Gold/abt.parquet`
  (Camada Gold). Controlado por `USE_AUX_AGGREGATIONS`. (Os auxiliares já chegam limpos da Silver — aqui
  é só agregação.)

### Passo 3: Notebooks de exploração (duas visões)
O projeto tem **dois notebooks** gerados por script, com focos distintos — abra-os no VSCode/Jupyter
(selecione o kernel do `.venv`) e rode as células:

| Notebook | Camada | Foco |
|---|---|---|
| `DataPipeline/exp_analysis.ipynb` | **Silver** (`clean_data.parquet`) | EDA dos **dados limpos**: nulos, alvo, idade, `EXT_SOURCE`, categóricas, força de associação (η²/Cramér's V) |
| `DataPipeline/abt_overview.ipynb` | **Gold** (`abt.parquet`) | Visão da **ABT final**: blocos de features, histórico de crédito, cobertura, correlação com o alvo |

* **`exp_analysis.ipynb`** mostra o dado **antes** da engenharia/agregações — orienta as decisões da ABT.
* **`abt_overview.ipynb`** mostra a tabela **pronta para o modelo** (307.511 × 113), incluindo as
  28 features `BUREAU_*`/`PREV_*` e a cobertura do histórico (≈86% têm bureau, ≈95% têm pedido anterior).

> **Como editar os notebooks:** abra o `.ipynb` no VSCode/Jupyter (selecione o kernel do `.venv`) e
> edite as células **diretamente** — o notebook **é** o entregável e a fonte de verdade. Antes de
> commitar, **limpe os outputs** das células (no Jupyter: *Kernel → Restart & Clear Output*; no VSCode:
> *Clear All Outputs*) para manter o `git diff` legível, já que o `.ipynb` guarda também imagens e
> metadados de execução.

### Dicionário de dados (metadados)
Dicionários de dados de cada camada (tipo, % de nulos, nº de únicos, min/máx e descrição de cada coluna):
- [`DataPipeline/dicionario_clean_data.md`](DataPipeline/dicionario_clean_data.md) — camada **Silver** (as 3 tabelas: application 122 col, bureau 17 col, previous 37 col).
- [`DataPipeline/dicionario_abt.md`](DataPipeline/dicionario_abt.md) — camada **Gold / ABT** (113 colunas, com bloco de origem).

Descrições das colunas de `application` vêm do dicionário oficial do Kaggle; as geradas/agregadas
são descritas no projeto. Regenerar: `python Tools\generate_data_dictionary.py`.

### Passo 4: Treinamento do Modelo
Treina os modelos candidatos (Scikit-Learn), compara por métricas e salva o melhor.
* **Comando:**
  ```powershell
  python Model\train.py
  ```
* **O que acontece (em 5 passos):** Lê `Dados/Gold/abt.parquet` (ABT, 307.511 × 113), separa **features
  (X)** e **alvo (`TARGET`)** e executa:

  1. **Separação treino/teste** — `train_test_split` **80% treino / 20% teste (holdout)**, estratificado
     pela `TARGET` (preserva os ~8% de inadimplentes nos dois lados). O **teste fica guardado e intocado**
     até o passo 5 — é a nossa prova final de desempenho em dados nunca vistos.
  2. **Seleção do algoritmo (só no treino, por validação cruzada)** — os 3 candidatos
     (**LogisticRegression** com `StandardScaler`, **RandomForest**, **GradientBoosting**), com
     hiperparâmetros fixos de `config.py`, são comparados pela **CV-AUC**: a AUC-ROC **média em 5 recortes**
     do treino (`StratifiedKFold`, 5 folds). Vence quem tem a **maior CV-AUC**. Aqui **só** se calcula a
     CV-AUC — o teste **não** é tocado e os candidatos **não** são treinados na base inteira (isso fica
     reservado para o vencedor, no passo 4).
  3. **Afinação de hiperparâmetros do vencedor (`GridSearchCV`)** — busca em grade (espaço em
     `config.SEARCH_GRIDS`), também por AUC-ROC em validação cruzada, aplicada **apenas ao algoritmo
     vencedor** do passo 2.
  4. **Treino final do vencedor** — o vencedor já afinado é **re-treinado usando todo o treino** (246k
     linhas).
  5. **Avaliação final (uma única vez, no teste)** — mede **AUC-ROC, KS, recall, precisão e acurácia** do
     vencedor no holdout de 20% e salva os artefatos: **`Model/model.pkl`** (modelo + colunas) e
     **`Model/metrics.json`** (CV-AUC dos 3 + quadro completo de holdout e `best_params` do vencedor).

  **Notas:** os passos 2 e 3 rodam numa **amostra de 30% do treino** (`SEARCH_SAMPLE_FRAC`) só para ir mais
  rápido (o GradientBoosting é lento na base inteira); o **treino final (passo 4) e a avaliação (passo 5)
  usam tudo**. Como só o vencedor é treinado na base inteira e avaliado no teste, há **1 treino completo em
  vez de 4** e o **holdout é tocado literalmente 1× no fim** — os perdedores param na seleção por CV-AUC
  (por isso, no `metrics.json`, eles trazem só `cv_auc`). O desbalanceamento (~8%) é tratado com
  `class_weight='balanced'` (e `sample_weight` no GradientBoosting, que não tem `class_weight`). Algoritmos e
  hiperparâmetros são definidos manualmente — **sem AutoML**.

> **Por que validação cruzada em vez de uma rodada única?** Medir o modelo uma vez só (treina em 80%,
> testa em 20%) é como avaliar alguém por **uma única prova**: pode dar sorte ou azar no recorte. A
> validação cruzada divide o treino em **5 fatias**, testa em cada uma por vez e tira a **média**. Com
> isso a comparação entre modelos fica **justa** (vence quem é melhor de forma consistente), dá para ver
> se o resultado é **estável**, e o **teste de 20% continua intocado** para o final. O **"Stratified"**
> mantém os ~8% de inadimplentes em cada fatia.

### Passo 5: Avaliação do Modelo
* **Notebook:** `Model/evaluation.ipynb` — abra no VSCode/Jupyter (kernel do `.venv`) e rode as células.
* **O que contém:** comparação dos modelos (AUC-ROC, KS, acurácia, recall, precision, F1), curva ROC/KS,
  matriz de confusão (com acurácia/precisão e a avaliação de "bom ou ruim"), análise de threshold e
  interpretabilidade por **três métodos** que se validam entre si: importância nativa (impureza),
  **permutation importance** (queda na AUC ao embaralhar cada feature) e **SHAP**.

### Passo 6: Predição (scoring)
* **Comando (CLI):**
  ```powershell
  python Model\predict.py --input Dados\Bronze\application_train.csv --n 5
  ```
* **O que acontece:** `predict.py` aplica a mesma transformação da ABT (via `abt_artifacts.pkl`) a
  novas solicitações e retorna a probabilidade de inadimplência + decisão (APROVAR / NEGAR-REVISAR).

---

## 🛰️ Serviço de Predição e Orquestração (MLOps)

* **App Streamlit:** `streamlit run App/app.py` (interface para escorar **e explicar** uma solicitação).
  Expõe os **scores de crédito externo** (`EXT_SOURCE_*`, os preditores mais fortes) como sliders e
  mostra os **principais fatores** que puxaram a decisão (via SHAP, barras 🔴 aumenta / 🟢 reduz risco).
* **API REST (FastAPI):** `uvicorn App.api:app --port 8000` — `GET /health`, `POST /predict`,
  `POST /predict/batch` e **`POST /explain`** (score + top fatores SHAP). Swagger em `/docs`; coleção
  Postman em [`App/postman_collection.json`](App/postman_collection.json). Campos amigáveis (inclui
  `ext_source_1/2/3`, 0..1, maior=melhor). Serviço `api` no compose (porta 8000).
* **Pipeline orquestrado (Airflow, PythonOperator):** `MLOps/pipeline_orchestration.py`
  (DAG: bronze→silver→gold→treino). A task de treino grava o **modelo** (`Model/model.pkl`) e as
  métricas no lake, consumidos por app/API. Também roda local sem Airflow:
  `python MLOps/pipeline_orchestration.py`.
* **Data lake (MinIO):** persistência das camadas medalhão (Bronze CSV + Silver/Gold Parquet) e dos
  artefatos num object storage S3-compatível, via a camada de I/O abstrata `MLOps/storage.py`
  (`STORAGE_BACKEND=local|minio`). Detalhes e fluxo em [`MLOps/Readme.md`](MLOps/Readme.md).

### Subindo tudo via docker-compose

O [`MLOps/docker-compose.yml`](MLOps/docker-compose.yml) sobe os serviços abaixo (a raiz do projeto é
montada como volume, então todos enxergam `Dados/`, `DataPipeline/`, `Model/` e `App/`):

| Serviço | Imagem / build | Porta | Papel |
|---|---|---|---|
| `minio` | `minio/minio` | **9000/9001** | Data lake S3 (API 9000, console 9001); bucket `fia-projeto-final` |
| `bootstrap` | `minio/mc` | — | Job one-shot no boot: cria o bucket **e** sobe **só os artefatos de serving** (`model.pkl`, `abt_artifacts.pkl`) p/ o lake. A Bronze é populada pela DAG (task `ingest_bronze`), não aqui |
| `airflow` | `apache/airflow` (SequentialExecutor + SQLite) | **8080** | Orquestra ingestão + pipeline (DAG `@monthly` `fia-projeto-final-pipeline`) |
| `app` | build `App/Dockerfile` | **8501** | App Streamlit (scoring + explicação) |
| `api` | mesmo build do `app` (troca o CMD) | **8000** | API FastAPI (`/docs`, `/predict`, `/explain`) |

> No compose, `airflow`/`app`/`api` rodam com `STORAGE_BACKEND=minio` — leem/gravam dados e artefatos
> no MinIO. Eles dependem da **conclusão do `bootstrap`** (`service_completed_successfully`), então só
> sobem após o MinIO estar pronto, o bucket criado e os artefatos de serving no lake. Fora do compose
> (scripts soltos sem as envs), o backend é `local` (filesystem).

```bash
cd MLOps
docker compose up --build          # sobe minio (+ bootstrap: bucket + artefatos) + airflow + app + api
# MinIO Console: http://localhost:9001   (usuário/senha: minioadmin/minioadmin)
# Airflow UI:    http://localhost:8080   (usuário/senha: airflow/airflow)
# App Streamlit: http://localhost:8501
# API (Swagger): http://localhost:8000/docs
```
> No boot, o `bootstrap` envia ao lake **só os artefatos de serving versionados** (`Model/model.pkl`,
> `DataPipeline/abt_artifacts.pkl`) — com isso o **app/API já escoram logo após o `up`**, sem esperar a
> DAG. **A Bronze é populada pela DAG** (task `ingest_bronze`, que lê a *landing zone* `Dados/Landing/`),
> então garanta que os CSVs do Kaggle estejam **nessa pasta** antes de rodar a DAG. Ao disparar a DAG,
> a Bronze é ingerida na partição do mês e o `Model/model.pkl` do lake é **sobrescrito** pelo modelo recém-treinado.

**Rodar o pipeline (medalhão + treino):**

* **Via Airflow (UI):** abra `http://localhost:8080` e **despause** a DAG
  **`fia-projeto-final-pipeline`** (`@monthly`). As tasks rodam em sequência
  (`ingest_bronze → bronze_to_silver → silver_to_gold → train_model`), com o mês da partição vindo da
  `execution_date`. Ao **despausar**, o Airflow agenda a run do último mês fechado; um *Trigger DAG*
  manual cria uma run adicional do **mês corrente** (evite disparar manualmente se não quiser as duas).
* **Via Airflow (CLI):**
  ```bash
  docker compose exec airflow airflow dags trigger fia-projeto-final-pipeline
  ```
* **Manualmente (sem Airflow), a partir da raiz do projeto:**
  ```bash
  python MLOps/pipeline_orchestration.py     # ingestão + medalhão + treino (local, modo plano)
  ```

> Ambos os caminhos foram validados ponta a ponta. A task de treino grava o `Model/model.pkl` e as
> métricas no lake — consumidos por app/API.

Detalhes de arquitetura, monitoramento e ações automatizadas em [`MLOps/Readme.md`](MLOps/Readme.md).

---

## 🤖 Modelos e Algoritmos

Treinamos **três algoritmos candidatos**, todos de **Machine Learning clássico (supervisionado,
classificação binária)** do Scikit-Learn. A seleção do melhor é feita **manualmente** pela
métrica **AUC-ROC em validação cruzada (CV-AUC)** no treino — o **holdout de 20% fica reservado
para a avaliação final** do vencedor (sem AutoML). Os hiperparâmetros são explícitos em
`Model/config.py`.

### Por que Machine Learning clássico (e não Deep Learning)?
- **Natureza do dado:** é um problema **tabular** (~83 features estruturadas), domínio em que
  modelos de árvore/ensembles costumam **igualar ou superar redes neurais**, com muito menos
  custo de tuning e dados.
- **Interpretabilidade exigida:** crédito é um domínio regulado; precisamos **explicar a decisão**
  (importância de features, coeficientes, SHAP). Modelos clássicos entregam isso de forma direta.
- **Custo/benefício:** treino rápido em CPU, sem necessidade de GPU nem grandes volumes — adequado
  ao escopo do TCC e ao baseline.
- **Stack enxuta e padrão:** priorizamos `scikit-learn` (estável, interpretável, sem GPU). O
  desbalanceamento (~8%) é tratado com `class_weight='balanced'` (ou `sample_weight` no
  GradientBoosting), em vez de reamostragem (`SMOTE`).

#### E as Redes Neurais (Deep Learning)?

Avaliamos conceitualmente os três tipos de rede neural (Keras/TensorFlow). **Nenhum foi adotado como
modelo principal**, pelos motivos abaixo:

- **Redes Densas (MLP)** — *tecnicamente aplicável* a classificação tabular. **Não adotada como
  principal** porque, em dado **tabular** com
  features heterogêneas, valores nulos e relações monotônicas, **ensembles de árvore (GradientBoosting/
  RandomForest) costumam igualar ou superar MLPs** com muito menos tuning; além disso o MLP é **caixa-preta**
  (pior interpretabilidade, crítica em crédito regulado), mais sensível a pré-processamento/escala e mais
  propenso a overfit, exigindo tratamento manual extra do desbalanceamento (~8%). Fica registrada como
  **baseline comparativo futuro** (rede densa vs. GradientBoosting).
- **Redes Convolucionais (CNN)** — **não se aplicam**: foram projetadas para dados com estrutura
  **espacial/grid** (imagens). Nossa ABT é um vetor de features tabulares, sem vizinhança espacial.
- **Redes Recorrentes (RNN/LSTM)** — **não se aplicam**: foram projetadas para **sequências/séries
  temporais**. Cada solicitação é um **snapshot estático** no momento do pedido; não há sequência
  temporal por cliente nesta base (`application_train`).

### Classificação dos algoritmos escolhidos

| Algoritmo | Família | Paradigma | Papel no projeto |
|---|---|---|---|
| **LogisticRegression** | Linear / GLM | ML clássico | Baseline interpretável; coeficientes indicam direção/peso do risco. |
| **RandomForest** | Ensemble — *bagging* de árvores | ML clássico | Captura não-linearidades e interações; robusto a outliers/escala. |
| **GradientBoosting** | Ensemble — *boosting* de árvores | ML clássico | Maior poder preditivo; foi o **melhor modelo** (holdout: AUC **0.773**, KS **0.407**, recall 0.692). |

### Justificativa e hiperparâmetros (controle de overfitting)

- **LogisticRegression** — modelo linear, serve de **referência (baseline)** e é o mais fácil de
  explicar para área de negócio. Usa `StandardScaler` no pipeline (sensível a escala) e
  `class_weight='balanced'`; `max_iter=1000` para garantir convergência.
- **RandomForest** — *bagging*: várias árvores em subconjuntos aleatórios, agregadas por voto.
  Reduz variância e lida bem com features heterogêneas. Overfitting controlado com
  **`max_depth=8`** (árvores rasas) e **`min_samples_leaf=50`** (folhas não muito específicas);
  `n_estimators=200`, `class_weight='balanced'`.
- **GradientBoosting** — *boosting*: árvores treinadas em sequência, cada uma corrigindo o erro da
  anterior. Tende a ser o mais preciso em dado tabular. Regularizado com **`max_depth=3`**
  (árvores fracas) e **`learning_rate=0.1`** com `n_estimators=200`; desbalanceamento via
  `sample_weight` (não aceita `class_weight` diretamente).

#### Tratamento de variáveis quantitativas (scaling)

Aplicamos **padronização (`StandardScaler`: média 0, desvio 1) apenas na LogisticRegression**, e o
scaler fica **dentro do `Pipeline` do modelo** — não na camada de dados (a ABT guarda os valores
numéricos crus, só imputados). Motivos:

- **LogisticRegression é sensível à escala:** é um modelo linear regularizado; sem padronizar, uma
  feature em unidades grandes (ex.: `AMT_CREDIT`, na casa dos milhões) dominaria uma de escala pequena
  (ex.: uma razão entre 0 e 1), distorcendo coeficientes e convergência. Por isso o `StandardScaler`.
- **Árvores (RandomForest e GradientBoosting) são invariantes à escala:** decidem por limiares
  (`feature <= x`), e reescalar não altera a ordem dos valores nem os *splits* possíveis. Padronizar
  seria inócuo. Como o **campeão é o GradientBoosting**, o modelo final **não usa scaling** — e não precisa.
- **Consistência treino ↔ predict garantida:** por estar no `Pipeline`, o `StandardScaler` é **fitado só
  no treino** (aprende média/desvio do treino, sem *leakage*) e é **salvo junto no `model.pkl`**. No
  `predict.py` basta chamar `model.predict_proba(X)`: o pipeline reaplica a mesma padronização
  automaticamente — sem risco de esquecer de escalar em produção.

| Modelo | Scaling? | Onde |
|---|---|---|
| LogisticRegression | ✅ `StandardScaler` | dentro do `Pipeline` (salvo no `model.pkl`) |
| RandomForest | ❌ (não precisa) | — |
| **GradientBoosting** (campeão) | ❌ (não precisa) | — |

> A busca de hiperparâmetros (`GridSearchCV`) roda **automaticamente no algoritmo vencedor** da seleção
> por CV, com o espaço de busca de cada modelo em `config.SEARCH_GRIDS`. Para manter o runtime baixo, a
> busca usa a subamostra de treino (`SEARCH_SAMPLE_FRAC`); o vencedor afinado é re-treinado no treino completo.

---

## 📊 Estrutura da ABT (Analytical Base Table)

A **ABT** (`Dados/Gold/abt.parquet`) consolida, em **1 linha por solicitação (`SK_ID_CURR`)**, a visão completa
do cliente no momento do pedido. Dimensão final: **307.511 linhas × 113 colunas**
(111 features + `SK_ID_CURR` + `TARGET`).

### Composição das 113 colunas

| Bloco | Origem | Nº aprox. | Exemplos |
|---|---|---|---|
| **Identificador** | — | 1 | `SK_ID_CURR` |
| **Alvo** | — | 1 | `TARGET` (0 = paga, 1 = inadimplente) |
| **Solicitação (application)** | `application_train` | ~80 | demografia, renda, crédito, `EXT_SOURCE_*`, flags |
| **Razões financeiras** (engenharia) | derivadas | 3 | `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `ANNUITY_CREDIT_RATIO` |
| **Histórico externo (birô)** | `bureau` agregado | 15 | `BUREAU_*` |
| **Histórico interno (Home Credit)** | `previous_application` agregado | 13 | `PREV_*` |

> Partimos de ~83 features só com `application_train` (baseline) e chegamos a **111 features** ao
> incorporar as **28 agregações de histórico de crédito**.

### 1. Features da solicitação (`application_train`)
- **Demográficas/pessoais:** idade (`DAYS_BIRTH`), gênero, estado civil, escolaridade.
- **Financeiras base:** renda (`AMT_INCOME_TOTAL`), crédito solicitado (`AMT_CREDIT`), prestação (`AMT_ANNUITY`).
- **Scores externos:** `EXT_SOURCE_1/2/3` — entre as features mais preditivas (preservadas mesmo com muitos nulos).

### 2. Razões financeiras (engenharia)
- **`CREDIT_INCOME_RATIO`** = crédito / renda (alavancagem; quanto maior, mais arriscado).
- **`ANNUITY_INCOME_RATIO`** = parcela / renda (comprometimento da renda).
- **`ANNUITY_CREDIT_RATIO`** = parcela / crédito (proxy do prazo da dívida).

### 3. Histórico de crédito agregado (`DataPipeline/feature_aggregation.py`)
As tabelas auxiliares **já chegam limpas da Silver** (`clean_bureau` / `clean_previous_application`,
deduplicadas e no nível de registro original); aqui elas são apenas **agregadas** ao nível de cliente
(`SK_ID_CURR`) antes do *join*. Cliente **sem histórico** recebe **0** nessas colunas (= "nenhum
histórico"; default reproduzido
no `predict.py`).

**`BUREAU_*` (15) — crédito em outras instituições:**
- *Volume/recência:* `BUREAU_CREDIT_COUNT`, `BUREAU_ACTIVE_COUNT`, `BUREAU_CLOSED_COUNT`, `BUREAU_ACTIVE_RATIO`, `BUREAU_DAYS_CREDIT_MIN/MAX/MEAN`.
- *Valores/dívida:* `BUREAU_AMT_CREDIT_SUM_TOTAL/MEAN`, `BUREAU_AMT_DEBT_TOTAL`, `BUREAU_DEBT_CREDIT_RATIO`.
- *Atraso/risco:* `BUREAU_DAY_OVERDUE_MAX/MEAN`, `BUREAU_AMT_OVERDUE_TOTAL`, `BUREAU_CNT_PROLONG_TOTAL`.

**`PREV_*` (13) — pedidos anteriores na Home Credit:**
- *Volume/resultado:* `PREV_APP_COUNT`, `PREV_APPROVED_COUNT`, `PREV_REFUSED_COUNT`, `PREV_APPROVAL_RATE`, `PREV_REFUSED_RATE`.
- *Valores:* `PREV_AMT_CREDIT_MEAN/TOTAL`, `PREV_AMT_APPLICATION_MEAN`, `PREV_AMT_DOWN_PAYMENT_MEAN`, `PREV_CREDIT_APP_RATIO_MEAN` (recebido vs. pedido).
- *Recência/prazo:* `PREV_DAYS_DECISION_MIN/MAX`, `PREV_CNT_PAYMENT_MEAN`.

### 4. Tratamento aplicado (Silver → Gold)
- **Descarte** de colunas com > 50% de nulos (preservando `EXT_SOURCE_*`).
- **Imputação** de nulos: mediana (numéricas) / moda (categóricas); agregações de histórico → 0.
- **Label Encoding** reproduzível das categóricas (categoria não vista → -1).
- Artefatos da transformação persistidos em `DataPipeline/abt_artifacts.pkl` para reprodução no `predict.py`.

### Impacto no modelo
As 28 features de histórico melhoraram o **AUC-ROC** nos três modelos. Melhor modelo
(**GradientBoosting**): **0.7658 → 0.7727** (KS **0.3945 → 0.4069**). Ganho modesto e consistente —
os `EXT_SOURCE_*` já concentravam grande parte do sinal. Ver detalhes em
"📈 Resultados e impacto do enriquecimento".

> Para gerar a ABT **sem** o enriquecimento (baseline), defina `USE_AUX_AGGREGATIONS = False` em
> `DataPipeline/config.py` e rode `python DataPipeline\abt_transform.py` novamente.

---

## 📈 Resultados e impacto do enriquecimento

Comparação no **mesmo holdout estratificado (20%, `random_state=42`)**, antes e depois de incorporar
as 28 features de histórico de crédito (`bureau` + `previous_application`). Métrica principal: **AUC-ROC**.

| Modelo | AUC (baseline, só application) | AUC (com histórico) | Δ AUC |
|---|---|---|---|
| LogisticRegression | 0.7489 | 0.7608 | **+0.0119** |
| RandomForest | 0.7433 | 0.7498 | +0.0065 |
| **GradientBoosting** (melhor) | 0.7658 | **0.7727** | +0.0069 |

- **Melhor modelo (GradientBoosting):** AUC **0.7658 → 0.7727**, KS **0.3945 → 0.4069**, recall da
  classe inadimplente ~0.69. Coerente com a competição Kaggle (~0.74–0.80 nessa faixa de features).
- **Ganho modesto, porém consistente** nos três modelos: os `EXT_SOURCE_*` já capturavam grande
  parte do sinal; ainda assim, o histórico de crédito agrega informação real e de forte sentido de
  negócio (inadimplência/atraso passados e taxa de recusa em pedidos anteriores).
- O baseline anterior fica preservado em `Model/metrics_baseline.json` para auditoria da comparação.

### Entendendo as métricas

O problema é **decidir se aprova ou nega um empréstimo**, sabendo que ~8% das pessoas não pagam.
Dois erros possíveis: **Falso Negativo** = aprovamos quem **não** paga (o prejuízo mais caro pra nós);
**Falso Positivo** = negamos quem **pagaria** (cliente bom perdido).

| Métrica | O que responde, em linguagem simples | Nosso valor |
|---|---|---|
| **AUC-ROC** | *"O modelo sabe ordenar quem é mais arriscado?"* Quanto maior, melhor (0,50 = chute; 1,0 = perfeito). | **0,773** (bom) |
| **KS** | *"O quanto separa bons de maus pagadores?"* Padrão no crédito; acima de ~0,30 já é saudável. | **0,407** (bom) |
| **Recall (inadimplentes)** | *"Dos que não pagariam, quantos o modelo pega?"* A métrica que mais importa depois da AUC. | **0,692** (~7 em 10) |
| **Precision (inadimplentes)** | *"Quando ele aponta 'risco', quantas vezes acerta?"* Fica baixa porque inadimplentes são raros — e preferimos pecar pela cautela. | **~0,17** (esperado) |
| **Acurácia** | *"Quantos acertos no total?"* **Engana em base desbalanceada:** aprovar todos daria ~0,92 sem pegar nenhum calote. Ficamos em ~0,71 **de propósito** (trocamos acurácia por recall). | **0,710** (contexto, não critério) |
| **CV-AUC** | *"A nota é confiável ou foi sorte?"* AUC média em 5 recortes do treino; foi por ela que escolhemos o campeão. | **0,763** (estável) |

> **Recall × Precision é um cabo de guerra:** apertar o modelo para pegar mais quem não paga (↑recall)
> faz ele marcar mais gente boa por engano (↓precision). O equilíbrio é uma **decisão de negócio**,
> controlada pelo *threshold* (limiar de 0,50 que aparece no app/API e pode ser ajustado).

> **Modelo atual (26/06/2026 — fluxo simplificado: seleção por CV-AUC + `GridSearchCV` no vencedor).**
> Pré-processamento **sem data leakage** (imputação/encoding fitados só no treino). Resultado no holdout (20%):
>
> | Modelo | CV-AUC | Holdout AUC | KS | Recall(1) | Precision(1) | Acurácia |
> |---|---|---|---|---|---|---|
> | LogisticRegression | 0.7541 | 0.7608 | 0.389 | 0.688 | 0.167 | 0.697 |
> | RandomForest | 0.7448 | 0.7498 | 0.374 | 0.668 | 0.165 | 0.701 |
> | **GradientBoosting** (campeão) | **0.7628** | **0.7727** | **0.407** | **0.692** | **0.174** | **0.710** |
>
> *(Acurácia baixa **de propósito**: com `class_weight='balanced'` o modelo troca acurácia por recall.
> Um "aprova todos" teria ~0.92 de acurácia e recall 0 — por isso a decisão segue AUC-ROC/KS/recall,
> não acurácia.)*
>
> O GradientBoosting venceu a seleção por CV-AUC (5 folds). O `GridSearchCV` confirmou
> `{n_estimators=200, max_depth=3, learning_rate=0.1}` como melhor combinação do grid (CV-AUC 0.7628),
> que é re-treinada no treino completo e avaliada uma única vez no holdout (AUC 0.7727). Resultado
> consistente com as runs anteriores (CV + busca), agora com um pipeline de treino mais direto e fácil de explicar.
