# Dicionário de Dados — Camada Gold / ABT (`abt.parquet`)

Analytical Base Table pronta para modelagem, gerada por `abt_transform.py`: 1 linha por `SK_ID_CURR`, **todas numéricas** (categóricas via Label Encoding), nulos imputados e enriquecida com agregações de histórico de crédito (`BUREAU_*` + `PREV_*`).

> Categóricas estão **codificadas** (inteiros). A descrição reflete a variável original; valor -1 indica categoria não vista no treino.

- **Dimensão:** 307,511 linhas × 113 colunas (111 features + ID + TARGET)

| Coluna | Bloco | Tipo | % Nulos | Únicos | Mín | Máx | Descrição |
|---|---|---|---|---|---|---|---|
| `SK_ID_CURR` | Chave/Alvo | int64 | 0.0% | 307,511 | 100002.00 | 456255.00 | Identificador da solicitação atual (1 por linha). Chave de junção do histórico. |
| `TARGET` | Chave/Alvo | int64 | 0.0% | 2 | 0.00 | 1.00 | Alvo: 0 = paga em dia, 1 = inadimplente. |
| `NAME_CONTRACT_TYPE` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Identification if loan is cash or revolving |
| `CODE_GENDER` | Application | int64 | 0.0% | 3 | 0.00 | 2.00 | Gender of the client |
| `FLAG_OWN_CAR` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if the client owns a car |
| `FLAG_OWN_REALTY` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client owns a house or flat |
| `CNT_CHILDREN` | Application | int64 | 0.0% | 15 | 0.00 | 19.00 | Number of children the client has |
| `AMT_INCOME_TOTAL` | Application | float64 | 0.0% | 2,548 | 25650.00 | 117000000.00 | Income of the client |
| `AMT_CREDIT` | Application | float64 | 0.0% | 5,603 | 45000.00 | 4050000.00 | Credit amount of the loan |
| `AMT_ANNUITY` | Application | float64 | 0.0% | 13,672 | 1615.50 | 258025.50 | Loan annuity |
| `AMT_GOODS_PRICE` | Application | float64 | 0.0% | 1,002 | 40500.00 | 4050000.00 | For consumer loans it is the price of the goods for which the loan is given |
| `NAME_TYPE_SUITE` | Application | int64 | 0.0% | 7 | 0.00 | 6.00 | Who was accompanying client when he was applying for the loan |
| `NAME_INCOME_TYPE` | Application | int64 | 0.0% | 8 | 0.00 | 7.00 | Clients income type (businessman, working, maternity leave,) |
| `NAME_EDUCATION_TYPE` | Application | int64 | 0.0% | 5 | 0.00 | 4.00 | Level of highest education the client achieved |
| `NAME_FAMILY_STATUS` | Application | int64 | 0.0% | 6 | 0.00 | 5.00 | Family status of the client |
| `NAME_HOUSING_TYPE` | Application | int64 | 0.0% | 6 | 0.00 | 5.00 | What is the housing situation of the client (renting, living with parents, ...) |
| `REGION_POPULATION_RELATIVE` | Application | float64 | 0.0% | 81 | 0.00 | 0.07 | Normalized population of region where client lives (higher number means the client lives in more populated region) |
| `DAYS_BIRTH` | Application | int64 | 0.0% | 17,460 | 7489.00 | 25229.00 | Client's age in days at the time of application |
| `DAYS_EMPLOYED` | Application | float64 | 0.0% | 12,573 | 0.00 | 17912.00 | How many days before the application the person started current employment |
| `DAYS_REGISTRATION` | Application | float64 | 0.0% | 15,688 | 0.00 | 24672.00 | How many days before the application did client change his registration |
| `DAYS_ID_PUBLISH` | Application | int64 | 0.0% | 6,168 | 0.00 | 7197.00 | How many days before the application did client change the identity document with which he applied for the loan |
| `FLAG_MOBIL` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide mobile phone (1=YES, 0=NO) |
| `FLAG_EMP_PHONE` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide work phone (1=YES, 0=NO) |
| `FLAG_WORK_PHONE` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide home phone (1=YES, 0=NO) |
| `FLAG_CONT_MOBILE` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Was mobile phone reachable (1=YES, 0=NO) |
| `FLAG_PHONE` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide home phone (1=YES, 0=NO) |
| `FLAG_EMAIL` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide email (1=YES, 0=NO) |
| `OCCUPATION_TYPE` | Application | int64 | 0.0% | 18 | 0.00 | 17.00 | What kind of occupation does the client have |
| `CNT_FAM_MEMBERS` | Application | float64 | 0.0% | 17 | 1.00 | 20.00 | How many family members does client have |
| `REGION_RATING_CLIENT` | Application | int64 | 0.0% | 3 | 1.00 | 3.00 | Our rating of the region where client lives (1,2,3) |
| `REGION_RATING_CLIENT_W_CITY` | Application | int64 | 0.0% | 3 | 1.00 | 3.00 | Our rating of the region where client lives with taking city into account (1,2,3) |
| `WEEKDAY_APPR_PROCESS_START` | Application | int64 | 0.0% | 7 | 0.00 | 6.00 | On which day of the week did the client apply for the loan |
| `HOUR_APPR_PROCESS_START` | Application | int64 | 0.0% | 24 | 0.00 | 23.00 | Approximately at what hour did the client apply for the loan |
| `REG_REGION_NOT_LIVE_REGION` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level) |
| `REG_REGION_NOT_WORK_REGION` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match work address (1=different, 0=same, at region level) |
| `LIVE_REGION_NOT_WORK_REGION` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's contact address does not match work address (1=different, 0=same, at region level) |
| `REG_CITY_NOT_LIVE_CITY` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) |
| `REG_CITY_NOT_WORK_CITY` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match work address (1=different, 0=same, at city level) |
| `LIVE_CITY_NOT_WORK_CITY` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's contact address does not match work address (1=different, 0=same, at city level) |
| `ORGANIZATION_TYPE` | Application | int64 | 0.0% | 58 | 0.00 | 57.00 | Type of organization where client works |
| `EXT_SOURCE_1` | Application | float64 | 0.0% | 114,584 | 0.01 | 0.96 | Normalized score from external data source |
| `EXT_SOURCE_2` | Application | float64 | 0.0% | 119,831 | 0.00 | 0.85 | Normalized score from external data source |
| `EXT_SOURCE_3` | Application | float64 | 0.0% | 814 | 0.00 | 0.90 | Normalized score from external data source |
| `YEARS_BEGINEXPLUATATION_AVG` | Application | float64 | 0.0% | 285 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMAX_AVG` | Application | float64 | 0.0% | 403 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BEGINEXPLUATATION_MODE` | Application | float64 | 0.0% | 221 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMAX_MODE` | Application | float64 | 0.0% | 25 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BEGINEXPLUATATION_MEDI` | Application | float64 | 0.0% | 245 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMAX_MEDI` | Application | float64 | 0.0% | 49 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `TOTALAREA_MODE` | Application | float64 | 0.0% | 5,116 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `EMERGENCYSTATE_MODE` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `OBS_30_CNT_SOCIAL_CIRCLE` | Application | float64 | 0.0% | 33 | 0.00 | 348.00 | How many observation of client's social surroundings with observable 30 DPD (days past due) default |
| `DEF_30_CNT_SOCIAL_CIRCLE` | Application | float64 | 0.0% | 10 | 0.00 | 34.00 | How many observation of client's social surroundings defaulted on 30 DPD (days past due) |
| `OBS_60_CNT_SOCIAL_CIRCLE` | Application | float64 | 0.0% | 33 | 0.00 | 344.00 | How many observation of client's social surroundings with observable 60 DPD (days past due) default |
| `DEF_60_CNT_SOCIAL_CIRCLE` | Application | float64 | 0.0% | 9 | 0.00 | 24.00 | How many observation of client's social surroundings defaulted on 60 (days past due) DPD |
| `DAYS_LAST_PHONE_CHANGE` | Application | float64 | 0.0% | 3,773 | -4292.00 | 0.00 | How many days before application did client change phone |
| `FLAG_DOCUMENT_2` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 2 |
| `FLAG_DOCUMENT_3` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 3 |
| `FLAG_DOCUMENT_4` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 4 |
| `FLAG_DOCUMENT_5` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 5 |
| `FLAG_DOCUMENT_6` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 6 |
| `FLAG_DOCUMENT_7` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 7 |
| `FLAG_DOCUMENT_8` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 8 |
| `FLAG_DOCUMENT_9` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 9 |
| `FLAG_DOCUMENT_10` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 10 |
| `FLAG_DOCUMENT_11` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 11 |
| `FLAG_DOCUMENT_12` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 12 |
| `FLAG_DOCUMENT_13` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 13 |
| `FLAG_DOCUMENT_14` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 14 |
| `FLAG_DOCUMENT_15` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 15 |
| `FLAG_DOCUMENT_16` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 16 |
| `FLAG_DOCUMENT_17` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 17 |
| `FLAG_DOCUMENT_18` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 18 |
| `FLAG_DOCUMENT_19` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 19 |
| `FLAG_DOCUMENT_20` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 20 |
| `FLAG_DOCUMENT_21` | Application | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 21 |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | Application | float64 | 0.0% | 5 | 0.00 | 4.00 | Number of enquiries to Credit Bureau about the client one hour before application |
| `AMT_REQ_CREDIT_BUREAU_DAY` | Application | float64 | 0.0% | 9 | 0.00 | 9.00 | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | Application | float64 | 0.0% | 9 | 0.00 | 8.00 | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) |
| `AMT_REQ_CREDIT_BUREAU_MON` | Application | float64 | 0.0% | 24 | 0.00 | 27.00 | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) |
| `AMT_REQ_CREDIT_BUREAU_QRT` | Application | float64 | 0.0% | 11 | 0.00 | 261.00 | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | Application | float64 | 0.0% | 25 | 0.00 | 25.00 | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) |
| `BUREAU_CREDIT_COUNT` | Histórico externo (bureau) | float64 | 0.0% | 61 | 0.00 | 116.00 | Nº de créditos do cliente em outras instituições (bureau/birô). |
| `BUREAU_ACTIVE_COUNT` | Histórico externo (bureau) | float64 | 0.0% | 23 | 0.00 | 32.00 | Nº de créditos ativos no bureau. |
| `BUREAU_CLOSED_COUNT` | Histórico externo (bureau) | float64 | 0.0% | 54 | 0.00 | 108.00 | Nº de créditos encerrados no bureau. |
| `BUREAU_AMT_CREDIT_SUM_TOTAL` | Histórico externo (bureau) | float64 | 0.0% | 205,644 | 0.00 | 1017957917.38 | Soma do valor de crédito de todos os créditos no bureau. |
| `BUREAU_AMT_CREDIT_SUM_MEAN` | Histórico externo (bureau) | float64 | 0.0% | 209,604 | 0.00 | 198072344.25 | Valor médio de crédito por contrato no bureau. |
| `BUREAU_AMT_DEBT_TOTAL` | Histórico externo (bureau) | float64 | 0.0% | 152,956 | 0.00 | 334498331.21 | Soma da dívida atual (AMT_CREDIT_SUM_DEBT) no bureau. |
| `BUREAU_DAY_OVERDUE_MAX` | Histórico externo (bureau) | float64 | 0.0% | 868 | 0.00 | 2792.00 | Máximo de dias em atraso (CREDIT_DAY_OVERDUE) no bureau. |
| `BUREAU_DAY_OVERDUE_MEAN` | Histórico externo (bureau) | float64 | 0.0% | 1,541 | 0.00 | 2776.00 | Média de dias em atraso no bureau. |
| `BUREAU_AMT_OVERDUE_TOTAL` | Histórico externo (bureau) | float64 | 0.0% | 1,218 | 0.00 | 3756681.00 | Soma do valor em atraso (AMT_CREDIT_SUM_OVERDUE) no bureau. |
| `BUREAU_DAYS_CREDIT_MIN` | Histórico externo (bureau) | float64 | 0.0% | 2,922 | -2922.00 | 0.00 | Dias desde o crédito mais antigo no bureau (DAYS_CREDIT mín). |
| `BUREAU_DAYS_CREDIT_MAX` | Histórico externo (bureau) | float64 | 0.0% | 2,923 | -2922.00 | 0.00 | Dias desde o crédito mais recente no bureau (DAYS_CREDIT máx). |
| `BUREAU_DAYS_CREDIT_MEAN` | Histórico externo (bureau) | float64 | 0.0% | 64,556 | -2922.00 | 0.00 | Recência média dos créditos no bureau (DAYS_CREDIT médio). |
| `BUREAU_CNT_PROLONG_TOTAL` | Histórico externo (bureau) | float64 | 0.0% | 10 | 0.00 | 9.00 | Total de prorrogações de crédito (CNT_CREDIT_PROLONG) no bureau. |
| `BUREAU_ACTIVE_RATIO` | Histórico externo (bureau) | float64 | 0.0% | 296 | 0.00 | 1.00 | Proporção de créditos ativos sobre o total no bureau. |
| `BUREAU_DEBT_CREDIT_RATIO` | Histórico externo (bureau) | float64 | 0.0% | 183,572 | 0.00 | 7.79 | Razão dívida/crédito no bureau (endividamento relativo). |
| `PREV_APP_COUNT` | Histórico interno (previous) | float64 | 0.0% | 66 | 0.00 | 73.00 | Nº de pedidos de crédito anteriores na Home Credit. |
| `PREV_APPROVED_COUNT` | Histórico interno (previous) | float64 | 0.0% | 26 | 0.00 | 27.00 | Nº de pedidos anteriores aprovados. |
| `PREV_REFUSED_COUNT` | Histórico interno (previous) | float64 | 0.0% | 46 | 0.00 | 68.00 | Nº de pedidos anteriores recusados. |
| `PREV_AMT_CREDIT_MEAN` | Histórico interno (previous) | float64 | 0.0% | 210,634 | 0.00 | 4050000.00 | Valor de crédito médio dos pedidos anteriores. |
| `PREV_AMT_CREDIT_TOTAL` | Histórico interno (previous) | float64 | 0.0% | 194,245 | 0.00 | 41461128.00 | Soma do crédito dos pedidos anteriores. |
| `PREV_AMT_APPLICATION_MEAN` | Histórico interno (previous) | float64 | 0.0% | 191,770 | 0.00 | 4050000.00 | Valor médio solicitado nos pedidos anteriores. |
| `PREV_CREDIT_APP_RATIO_MEAN` | Histórico interno (previous) | float64 | 0.0% | 243,634 | 0.00 | 10.18 | Razão média concedido/solicitado (AMT_CREDIT / AMT_APPLICATION). |
| `PREV_AMT_DOWN_PAYMENT_MEAN` | Histórico interno (previous) | float64 | 0.0% | 54,129 | -0.23 | 2025000.00 | Entrada média (AMT_DOWN_PAYMENT) nos pedidos anteriores. |
| `PREV_DAYS_DECISION_MAX` | Histórico interno (previous) | float64 | 0.0% | 2,923 | -2922.00 | 0.00 | Dias desde a decisão mais recente (DAYS_DECISION máx). |
| `PREV_DAYS_DECISION_MIN` | Histórico interno (previous) | float64 | 0.0% | 2,922 | -2922.00 | 0.00 | Dias desde a decisão mais antiga (DAYS_DECISION mín). |
| `PREV_CNT_PAYMENT_MEAN` | Histórico interno (previous) | float64 | 0.0% | 2,838 | 0.00 | 72.00 | Prazo médio em parcelas (CNT_PAYMENT) dos pedidos anteriores. |
| `PREV_APPROVAL_RATE` | Histórico interno (previous) | float64 | 0.0% | 362 | 0.00 | 1.00 | Taxa de aprovação dos pedidos anteriores (aprovados / total). |
| `PREV_REFUSED_RATE` | Histórico interno (previous) | float64 | 0.0% | 397 | 0.00 | 1.00 | Taxa de recusa dos pedidos anteriores (recusados / total). |
| `CREDIT_INCOME_RATIO` | Razão (engenharia) | float64 | 0.0% | 48,697 | 0.00 | 84.74 | Razão crédito/renda (AMT_CREDIT / AMT_INCOME_TOTAL) — alavancagem. |
| `ANNUITY_INCOME_RATIO` | Razão (engenharia) | float64 | 0.0% | 88,811 | 0.00 | 1.88 | Razão parcela/renda (AMT_ANNUITY / AMT_INCOME_TOTAL) — comprometimento da renda. |
| `ANNUITY_CREDIT_RATIO` | Razão (engenharia) | float64 | 0.0% | 39,539 | 0.02 | 0.12 | Razão parcela/crédito (AMT_ANNUITY / AMT_CREDIT) — proxy do prazo da dívida. |
