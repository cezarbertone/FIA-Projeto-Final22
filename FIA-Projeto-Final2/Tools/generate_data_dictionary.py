"""
Gera os dicionários de dados (metadados) das camadas Silver e Gold.

Produz dois arquivos Markdown:
    - DataPipeline/dicionario_clean_data.md  (Silver / clean_data.parquet)
    - DataPipeline/dicionario_abt.md          (Gold  / abt.parquet)

Para cada coluna documenta: tipo, % de nulos, nº de valores únicos, min/max (numéricas) e descrição.
As descrições das colunas de `application` vêm do dicionário oficial do Kaggle
(`HomeCredit_columns_description.csv`); as features engenheiradas/agregadas têm descrição própria.

Execute a partir da raiz do projeto:
    python Tools/generate_data_dictionary.py
"""
import os
import sys

import pandas as pd

# Este script vive em Tools/, mas usa os caminhos definidos em DataPipeline/config.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DataPipeline"))
import config

# --- Descrições das features criadas no projeto (não existem no dicionário do Kaggle) ---
DESC_ENGINEERED = {
    "CREDIT_INCOME_RATIO": "Razão crédito/renda (AMT_CREDIT / AMT_INCOME_TOTAL) — alavancagem.",
    "ANNUITY_INCOME_RATIO": "Razão parcela/renda (AMT_ANNUITY / AMT_INCOME_TOTAL) — comprometimento da renda.",
    "ANNUITY_CREDIT_RATIO": "Razão parcela/crédito (AMT_ANNUITY / AMT_CREDIT) — proxy do prazo da dívida.",
}

DESC_BUREAU = {
    "BUREAU_CREDIT_COUNT": "Nº de créditos do cliente em outras instituições (bureau/birô).",
    "BUREAU_ACTIVE_COUNT": "Nº de créditos ativos no bureau.",
    "BUREAU_CLOSED_COUNT": "Nº de créditos encerrados no bureau.",
    "BUREAU_AMT_CREDIT_SUM_TOTAL": "Soma do valor de crédito de todos os créditos no bureau.",
    "BUREAU_AMT_CREDIT_SUM_MEAN": "Valor médio de crédito por contrato no bureau.",
    "BUREAU_AMT_DEBT_TOTAL": "Soma da dívida atual (AMT_CREDIT_SUM_DEBT) no bureau.",
    "BUREAU_DAY_OVERDUE_MAX": "Máximo de dias em atraso (CREDIT_DAY_OVERDUE) no bureau.",
    "BUREAU_DAY_OVERDUE_MEAN": "Média de dias em atraso no bureau.",
    "BUREAU_AMT_OVERDUE_TOTAL": "Soma do valor em atraso (AMT_CREDIT_SUM_OVERDUE) no bureau.",
    "BUREAU_DAYS_CREDIT_MIN": "Dias desde o crédito mais antigo no bureau (DAYS_CREDIT mín).",
    "BUREAU_DAYS_CREDIT_MAX": "Dias desde o crédito mais recente no bureau (DAYS_CREDIT máx).",
    "BUREAU_DAYS_CREDIT_MEAN": "Recência média dos créditos no bureau (DAYS_CREDIT médio).",
    "BUREAU_CNT_PROLONG_TOTAL": "Total de prorrogações de crédito (CNT_CREDIT_PROLONG) no bureau.",
    "BUREAU_ACTIVE_RATIO": "Proporção de créditos ativos sobre o total no bureau.",
    "BUREAU_DEBT_CREDIT_RATIO": "Razão dívida/crédito no bureau (endividamento relativo).",
}

DESC_PREV = {
    "PREV_APP_COUNT": "Nº de pedidos de crédito anteriores na Home Credit.",
    "PREV_APPROVED_COUNT": "Nº de pedidos anteriores aprovados.",
    "PREV_REFUSED_COUNT": "Nº de pedidos anteriores recusados.",
    "PREV_AMT_CREDIT_MEAN": "Valor de crédito médio dos pedidos anteriores.",
    "PREV_AMT_CREDIT_TOTAL": "Soma do crédito dos pedidos anteriores.",
    "PREV_AMT_APPLICATION_MEAN": "Valor médio solicitado nos pedidos anteriores.",
    "PREV_CREDIT_APP_RATIO_MEAN": "Razão média concedido/solicitado (AMT_CREDIT / AMT_APPLICATION).",
    "PREV_AMT_DOWN_PAYMENT_MEAN": "Entrada média (AMT_DOWN_PAYMENT) nos pedidos anteriores.",
    "PREV_DAYS_DECISION_MAX": "Dias desde a decisão mais recente (DAYS_DECISION máx).",
    "PREV_DAYS_DECISION_MIN": "Dias desde a decisão mais antiga (DAYS_DECISION mín).",
    "PREV_CNT_PAYMENT_MEAN": "Prazo médio em parcelas (CNT_PAYMENT) dos pedidos anteriores.",
    "PREV_APPROVAL_RATE": "Taxa de aprovação dos pedidos anteriores (aprovados / total).",
    "PREV_REFUSED_RATE": "Taxa de recusa dos pedidos anteriores (recusados / total).",
}

DESC_KEYS = {
    "SK_ID_CURR": "Identificador da solicitação atual (1 por linha). Chave de junção do histórico.",
    "TARGET": "Alvo: 0 = paga em dia, 1 = inadimplente.",
}


def _load_kaggle_descriptions(table_filter: str | None = None) -> dict:
    """Lê HomeCredit_columns_description.csv (se existir) e retorna {coluna: descrição}.

    `table_filter=None` -> descrições da(s) tabela(s) `application_*` (default, base principal).
    `table_filter="bureau.csv"` / `"previous_application.csv"` -> a tabela auxiliar correspondente.
    """
    path = os.path.join(config.AUX_DATA_DIR, "HomeCredit_columns_description.csv")
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path, encoding="latin-1")
    if table_filter is None:
        d = d[d["Table"].str.startswith("application", na=False)]
    else:
        d = d[d["Table"] == table_filter]
    return dict(zip(d["Row"], d["Description"]))


def _describe(col: str, kaggle: dict) -> str:
    for mapping in (DESC_KEYS, DESC_ENGINEERED, DESC_BUREAU, DESC_PREV):
        if col in mapping:
            return mapping[col]
    return str(kaggle.get(col, "-")).replace("\n", " ").strip()


def _block(col: str) -> str:
    if col in DESC_KEYS:
        return "Chave/Alvo"
    if col in DESC_ENGINEERED:
        return "Razão (engenharia)"
    if col.startswith("BUREAU_"):
        return "Histórico externo (bureau)"
    if col.startswith("PREV_"):
        return "Histórico interno (previous)"
    return "Application"


def build_dictionary(df: pd.DataFrame, kaggle: dict, with_block: bool) -> str:
    """Monta a tabela Markdown do dicionário de dados a partir de um DataFrame."""
    n = len(df)
    header = "| Coluna | " + ("Bloco | " if with_block else "") + "Tipo | % Nulos | Únicos | Mín | Máx | Descrição |"
    sep = "|---|" + ("---|" if with_block else "") + "---|---|---|---|---|---|"
    lines = [header, sep]
    for col in df.columns:
        s = df[col]
        pct_null = f"{s.isnull().mean() * 100:.1f}%"
        nuniq = f"{s.nunique():,}"
        if pd.api.types.is_numeric_dtype(s):
            mn = f"{s.min():.2f}" if pd.notnull(s.min()) else "-"
            mx = f"{s.max():.2f}" if pd.notnull(s.max()) else "-"
        else:
            mn = mx = "-"
        desc = _describe(col, kaggle)
        bloco = f" {_block(col)} |" if with_block else ""
        lines.append(f"| `{col}` |{bloco} {s.dtype} | {pct_null} | {nuniq} | {mn} | {mx} | {desc} |")
    return "\n".join(lines)


def _table_section(title: str, subtitle: str, df: pd.DataFrame, kmap: dict) -> str:
    """Monta a seção Markdown de UMA tabela Silver (cabeçalho + métricas + dicionário)."""
    return (
        f"## {title}\n\n{subtitle}\n\n"
        f"- **Dimensão:** {df.shape[0]:,} linhas × {df.shape[1]} colunas\n"
        f"- **Numéricas:** {df.select_dtypes('number').shape[1]} | "
        f"**Categóricas:** {df.select_dtypes(exclude='number').shape[1]}\n\n"
        + build_dictionary(df, kmap, with_block=False) + "\n"
    )


def generate(clean_path: str, abt_path: str, out_dir: str = "DataPipeline"):
    kaggle = _load_kaggle_descriptions()
    if not kaggle:
        print("AVISO: HomeCredit_columns_description.csv não encontrado; "
              "descrições das colunas de application ficarão como '-'.")
    kaggle_bureau = _load_kaggle_descriptions("bureau.csv")
    kaggle_prev = _load_kaggle_descriptions("previous_application.csv")

    # --- Silver: TRÊS tabelas limpas (uma por fonte do Kaggle, nível de registro original) ---
    clean = pd.read_parquet(clean_path)
    bureau = pd.read_parquet(config.CLEAN_BUREAU_PATH)
    prev = pd.read_parquet(config.CLEAN_PREV_PATH)
    md_clean = (
        f"# Dicionário de Dados — Camada Silver\n\n"
        f"Dados limpos gerados por `data_sanitization.py` (nomes em UPPER, anomalias tratadas, "
        f"duplicatas removidas), cada fonte no seu **nível de registro original**. **Ainda não** passou por "
        f"engenharia de features, imputação, encoding ou agregações de histórico (isso é a Gold).\n\n"
        f"A Silver tem **três tabelas**, uma por fonte do Kaggle:\n\n"
        + _table_section(
            "1. `clean_data.parquet` — solicitações (application)",
            "Base principal: 1 linha por solicitante (`SK_ID_CURR`). Contém o alvo `TARGET`.",
            clean, kaggle)
        + "\n"
        + _table_section(
            "2. `clean_bureau.parquet` — histórico externo (bureau)",
            "Créditos do cliente em OUTRAS instituições (birô). 1 linha por crédito "
            "(`SK_ID_BUREAU`); vários por cliente. Agregado em `BUREAU_*` na Gold.",
            bureau, kaggle_bureau)
        + "\n"
        + _table_section(
            "3. `clean_previous_application.parquet` — histórico interno (previous)",
            "Pedidos de crédito anteriores do cliente na Home Credit. 1 linha por pedido "
            "(`SK_ID_PREV`); vários por cliente. Agregado em `PREV_*` na Gold.",
            prev, kaggle_prev)
    )
    out_clean = os.path.join(out_dir, "dicionario_clean_data.md")
    with open(out_clean, "w", encoding="utf-8") as f:
        f.write(md_clean)
    print(f"Dicionário Silver salvo em: {out_clean} "
          f"(application {clean.shape[1]} col, bureau {bureau.shape[1]} col, prev {prev.shape[1]} col)")

    # --- Gold (ABT) ---
    abt = pd.read_parquet(abt_path)
    md_abt = (
        f"# Dicionário de Dados — Camada Gold / ABT (`abt.parquet`)\n\n"
        f"Analytical Base Table pronta para modelagem, gerada por `abt_transform.py`: 1 linha por "
        f"`SK_ID_CURR`, **todas numéricas** (categóricas via Label Encoding), nulos imputados e "
        f"enriquecida com agregações de histórico de crédito (`BUREAU_*` + `PREV_*`).\n\n"
        f"> Categóricas estão **codificadas** (inteiros). A descrição reflete a variável original; "
        f"valor -1 indica categoria não vista no treino.\n\n"
        f"- **Dimensão:** {abt.shape[0]:,} linhas × {abt.shape[1]} colunas "
        f"({abt.shape[1] - 2} features + ID + TARGET)\n\n"
        + build_dictionary(abt, kaggle, with_block=True) + "\n"
    )
    out_abt = os.path.join(out_dir, "dicionario_abt.md")
    with open(out_abt, "w", encoding="utf-8") as f:
        f.write(md_abt)
    print(f"Dicionário Gold salvo em: {out_abt} ({abt.shape[1]} colunas)")


if __name__ == "__main__":
    generate(config.CLEAN_PATH, config.ABT_PATH)
