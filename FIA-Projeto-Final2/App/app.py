"""
App Streamlit — Score de Risco de Inadimplência (Home Credit).

Interface para escorar uma solicitação de crédito usando o modelo treinado e **explicar a
decisão** (principais fatores via SHAP). Reutiliza Model/predict.py (mesma transformação da ABT).

Executar a partir da raiz do projeto:
    streamlit run App/app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

# Torna o pacote Model importável a partir da raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Model"))
from predict import predict, explain, DECISION_THRESHOLD  # noqa: E402

st.set_page_config(page_title="Score de Risco de Inadimplência", page_icon="💳", layout="centered")
st.title("💳 Score de Risco de Inadimplência")
st.caption("Home Credit Default Risk — previsão no momento da solicitação do crédito")

threshold = st.sidebar.slider(
    "Threshold de decisão (probabilidade p/ NEGAR/REVISAR)",
    min_value=0.05, max_value=0.95, value=float(DECISION_THRESHOLD), step=0.05,
)

st.subheader("Dados da solicitação")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gênero", ["M", "F"])
    age = st.number_input("Idade (anos)", min_value=18, max_value=100, value=35)
    income = st.number_input("Renda total anual", min_value=0.0, value=180000.0, step=1000.0)
    credit = st.number_input("Valor do crédito solicitado", min_value=0.0, value=600000.0, step=1000.0)
with col2:
    annuity = st.number_input("Valor da parcela (annuity)", min_value=0.0, value=27000.0, step=500.0)
    goods = st.number_input("Valor do bem (goods price)", min_value=0.0, value=540000.0, step=1000.0)
    years_employed = st.number_input("Anos de emprego", min_value=0.0, max_value=60.0, value=5.0)
    education = st.selectbox(
        "Escolaridade",
        ["Secondary / secondary special", "Higher education",
         "Incomplete higher", "Lower secondary", "Academic degree"],
    )

# Scores de crédito externo (EXT_SOURCE_*) — os preditores MAIS FORTES do modelo (~58% da
# importância). Escala 0..1, maior = melhor histórico/menor risco.
st.markdown("#### 📊 Scores de crédito externo (bureau)")
st.caption("São os fatores **mais determinantes** do modelo. 0 = histórico ruim · 1 = excelente. "
           "Se não souber, deixe em ~0,5 (valor típico).")
ec1, ec2, ec3 = st.columns(3)
with ec1:
    ext1 = st.slider("Score externo 1", 0.0, 1.0, 0.51, 0.01)
with ec2:
    ext2 = st.slider("Score externo 2", 0.0, 1.0, 0.57, 0.01)
with ec3:
    ext3 = st.slider("Score externo 3", 0.0, 1.0, 0.54, 0.01)

# Features dirigidas pelo formulário (informadas pelo usuário) + os ratios derivados delas.
# A explicação é restrita a estas — as demais são imputadas (valor constante p/ todo solicitante),
# então não são "fatores desta solicitação" e só confundiriam o usuário.
USER_FEATURES = [
    "CODE_GENDER", "DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "NAME_EDUCATION_TYPE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "ANNUITY_CREDIT_RATIO",
]

if st.button("Calcular score", type="primary"):
    record = {
        "SK_ID_CURR": 999999,
        "CODE_GENDER": gender,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "DAYS_BIRTH": -int(age * 365),
        "DAYS_EMPLOYED": -int(years_employed * 365),
        "NAME_EDUCATION_TYPE": education,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    }
    result = predict(pd.DataFrame([record]), threshold=threshold).iloc[0]
    proba = float(result["probabilidade_inadimplencia"])

    st.metric("Probabilidade de inadimplência", f"{proba:.1%}")
    if result["decisao"] == "APROVAR":
        st.success(f"✅ Decisão sugerida: **APROVAR** (abaixo do threshold {threshold:.0%})")
    else:
        st.error(f"⛔ Decisão sugerida: **NEGAR/REVISAR** (acima do threshold {threshold:.0%})")

    # Explicação da decisão: principais fatores que puxaram o risco p/ cima/baixo
    # (SHAP quando disponível; senão, atribuição por occlusion — ver Model/predict.explain).
    st.markdown("#### 🔎 Por que essa decisão? (principais fatores)")
    fatores = explain(record, top_n=7, only_cols=USER_FEATURES)[0]
    max_abs = max((abs(f["contribuicao"]) for f in fatores), default=1.0) or 1.0
    for f in fatores:
        sobe = f["efeito"] == "aumenta"
        icon = "🔴" if sobe else "🟢"
        texto = "aumenta o risco" if sobe else "reduz o risco"
        st.write(f"{icon} **{f['fator']}** — {texto}")
        st.progress(abs(f["contribuicao"]) / max_abs)
    st.caption("Tamanho da barra = força do fator nesta solicitação. "
               "🔴 empurra para inadimplência · 🟢 puxa para aprovação.")

    st.caption("As features não informadas são imputadas com os valores aprendidos no treino.")
