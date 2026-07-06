"""
API REST — Score de Risco de Inadimplência (Home Credit).

Casca HTTP fina sobre Model/predict.predict (que aplica a mesma transformação da ABT
via artefatos e gera probabilidade + decisão de negócio). Mesma lógica do app Streamlit,
exposta como serviço para integração (ex.: sistema de concessão de crédito).

Executar a partir da raiz do projeto:
    uvicorn App.api:app --reload --port 8000

Documentação interativa (Swagger): http://localhost:8000/docs
"""
import os
import sys
from enum import Enum

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

# Torna o pacote Model importável a partir da raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Model"))
from predict import predict, explain, DECISION_THRESHOLD  # noqa: E402
import config as model_config  # noqa: E402

app = FastAPI(
    title="Score de Risco de Inadimplência",
    description="Home Credit Default Risk — previsão no momento da solicitação do crédito.",
    version="1.0.0",
)


class Gender(str, Enum):
    """Gênero do solicitante (mapeia para CODE_GENDER no formato `application`)."""

    MALE = "M"
    FEMALE = "F"


class Education(str, Enum):
    """Escolaridade (mapeia para NAME_EDUCATION_TYPE; valores = categorias do treino)."""

    SECONDARY = "Secondary / secondary special"
    HIGHER = "Higher education"
    INCOMPLETE_HIGHER = "Incomplete higher"
    LOWER_SECONDARY = "Lower secondary"
    ACADEMIC_DEGREE = "Academic degree"


class CreditApplication(BaseModel):
    """Dados de uma solicitação de crédito, em formato amigável (alinhado ao app Streamlit).

    Todos os campos são opcionais; os ausentes são imputados com os valores aprendidos no
    treino (ver Model/predict.py). A tradução para o formato cru `application`
    (CODE_GENDER, DAYS_BIRTH, ...) é feita internamente por `to_application()`.
    """

    model_config = ConfigDict(extra="forbid")

    applicant_id: int | None = Field(default=None, description="ID da solicitação (opcional)", examples=[100001])
    gender: Gender | None = Field(default=None, description="Gênero do solicitante")
    age_years: float | None = Field(default=None, description="Idade em anos", examples=[35])
    total_income: float | None = Field(default=None, description="Renda total anual", examples=[180000])
    credit_amount: float | None = Field(default=None, description="Valor do crédito solicitado", examples=[600000])
    annuity_amount: float | None = Field(default=None, description="Valor da parcela (annuity)", examples=[27000])
    goods_price: float | None = Field(default=None, description="Valor do bem", examples=[540000])
    years_employed: float | None = Field(default=None, description="Anos de emprego", examples=[5])
    education: Education | None = Field(default=None, description="Escolaridade")
    # Scores de crédito externo (EXT_SOURCE_*) — os preditores MAIS FORTES do modelo (~58% da
    # importância). Escala 0..1, MAIOR = melhor histórico/menor risco. Ausentes -> imputados (~0.5).
    ext_source_1: float | None = Field(default=None, ge=0.0, le=1.0,
                                       description="Score de crédito externo 1 (0..1, maior=melhor)")
    ext_source_2: float | None = Field(default=None, ge=0.0, le=1.0,
                                       description="Score de crédito externo 2 (0..1, maior=melhor)")
    ext_source_3: float | None = Field(default=None, ge=0.0, le=1.0,
                                       description="Score de crédito externo 3 (0..1, maior=melhor)")

    def to_application(self) -> dict:
        """Converte para o dict no formato cru `application` (só campos informados)."""
        record: dict = {}
        if self.applicant_id is not None:
            record["SK_ID_CURR"] = self.applicant_id
        if self.gender is not None:
            record["CODE_GENDER"] = self.gender.value
        if self.age_years is not None:
            record["DAYS_BIRTH"] = -round(self.age_years * 365)
        if self.total_income is not None:
            record["AMT_INCOME_TOTAL"] = self.total_income
        if self.credit_amount is not None:
            record["AMT_CREDIT"] = self.credit_amount
        if self.annuity_amount is not None:
            record["AMT_ANNUITY"] = self.annuity_amount
        if self.goods_price is not None:
            record["AMT_GOODS_PRICE"] = self.goods_price
        if self.years_employed is not None:
            record["DAYS_EMPLOYED"] = -round(self.years_employed * 365)
        if self.education is not None:
            record["NAME_EDUCATION_TYPE"] = self.education.value
        if self.ext_source_1 is not None:
            record["EXT_SOURCE_1"] = self.ext_source_1
        if self.ext_source_2 is not None:
            record["EXT_SOURCE_2"] = self.ext_source_2
        if self.ext_source_3 is not None:
            record["EXT_SOURCE_3"] = self.ext_source_3
        return record


class Resultado(BaseModel):
    """Score de uma solicitação."""

    applicant_id: int
    default_probability: float = Field(description="Probabilidade de inadimplência (0..1)")
    default_probability_pct: float = Field(description="Probabilidade de inadimplência em % (0..100)")
    decision: str

    @classmethod
    def from_predict(cls, row: dict) -> "Resultado":
        """Monta a resposta a partir das chaves cruas retornadas por Model/predict.predict."""
        proba = row["probabilidade_inadimplencia"]
        return cls(
            applicant_id=row["SK_ID_CURR"],
            default_probability=proba,
            default_probability_pct=round(proba * 100, 2),
            decision=row["decisao"],
        )


class Fator(BaseModel):
    """Um fator que influenciou a decisão (contribuição SHAP em log-odds)."""

    fator: str = Field(description="Nome amigável da variável")
    contribuicao: float = Field(description="Contribuição SHAP (>0 aumenta risco, <0 reduz)")
    efeito: str = Field(description="'aumenta' ou 'reduz' o risco")


class Explicacao(BaseModel):
    """Score + principais fatores que puxaram a decisão (SHAP)."""

    applicant_id: int
    default_probability: float
    default_probability_pct: float
    decision: str
    fatores: list[Fator]


@app.get("/health")
def health() -> dict:
    """Status do serviço e caminho do modelo carregado."""
    return {"status": "ok", "model_path": model_config.MODEL_PATH, "threshold_default": DECISION_THRESHOLD}


@app.post("/predict", response_model=Resultado)
def predict_one(
    solicitacao: CreditApplication,
    threshold: float = Query(default=DECISION_THRESHOLD, ge=0.0, le=1.0),
) -> Resultado:
    """Escora uma única solicitação de crédito."""
    out = predict(pd.DataFrame([solicitacao.to_application()]), threshold=threshold)
    return Resultado.from_predict(out.iloc[0].to_dict())


@app.post("/predict/batch", response_model=list[Resultado])
def predict_batch(
    solicitacoes: list[CreditApplication],
    threshold: float = Query(default=DECISION_THRESHOLD, ge=0.0, le=1.0),
) -> list[Resultado]:
    """Escora um lote de solicitações."""
    records = [s.to_application() for s in solicitacoes]
    out = predict(pd.DataFrame(records), threshold=threshold)
    return [Resultado.from_predict(row) for row in out.to_dict(orient="records")]


@app.post("/explain", response_model=Explicacao)
def explain_one(
    solicitacao: CreditApplication,
    top_n: int = Query(default=8, ge=1, le=20, description="Nº de fatores a retornar"),
    threshold: float = Query(default=DECISION_THRESHOLD, ge=0.0, le=1.0),
) -> Explicacao:
    """Escora e explica a decisão: principais fatores que puxaram o risco (SHAP)."""
    record = solicitacao.to_application()
    out = predict(pd.DataFrame([record]), threshold=threshold).iloc[0].to_dict()
    # explain() usa SHAP se disponível, senão cai p/ atribuição por occlusion (sem deps).
    fatores = explain(record, top_n=top_n)[0]
    return Explicacao(
        applicant_id=out["SK_ID_CURR"],
        default_probability=out["probabilidade_inadimplencia"],
        default_probability_pct=round(out["probabilidade_inadimplencia"] * 100, 2),
        decision=out["decisao"],
        fatores=[Fator(**f) for f in fatores],
    )
