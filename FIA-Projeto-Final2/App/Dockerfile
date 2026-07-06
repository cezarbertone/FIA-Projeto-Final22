# Imagem do serviço de predição (Streamlit + API) — Home Credit Default Risk.
#
# Base python:3.12-slim (e NÃO 3.14): o `shap`/`numba` (explicação da decisão) ainda não têm
# wheel para Python 3.14 no Linux. O que garante a compatibilidade do `pickle.load(model.pkl)`
# são as VERSÕES das libs (scikit-learn/numpy/pandas), fixadas em App/requirements.txt — não a
# versão do Python. O treino roda em 3.14; o serving em 3.12 com as MESMAS versões de libs.
#
# Build a partir da RAIZ do projeto (o contexto precisa enxergar Model/, DataPipeline/, App/):
#   docker build -f App/Dockerfile -t hc-app .
#   docker run -p 8501:8501 hc-app
FROM python:3.12-slim

WORKDIR /opt/project

# 1) Dependências (camada cacheada enquanto o requirements não muda).
#    timeout/retries maiores: numba/llvmlite/scipy são wheels grandes (rede pode estourar o default).
COPY App/requirements.txt App/requirements.txt
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r App/requirements.txt

# 2) Artefatos do modelo + código necessário em runtime.
#    NÃO copiamos os dados (raw/clean/abt.csv) nem os scripts de pipeline:
#    o predict.py reimplementa a transformação e usa apenas os artefatos abaixo.
COPY Model/model.pkl               Model/model.pkl
COPY Model/predict.py              Model/predict.py
COPY Model/config.py               Model/config.py
COPY DataPipeline/abt_artifacts.pkl DataPipeline/abt_artifacts.pkl
COPY App/app.py                    App/app.py
COPY App/api.py                    App/api.py
# Camada de I/O abstrata (filesystem local OU data lake MinIO) — usada por predict.py.
COPY MLOps/storage.py              MLOps/storage.py

EXPOSE 8501 8000

CMD ["streamlit", "run", "App/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
