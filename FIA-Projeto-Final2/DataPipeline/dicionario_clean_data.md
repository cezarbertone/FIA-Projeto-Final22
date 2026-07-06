# Dicionário de Dados — Camada Silver

Dados limpos gerados por `data_sanitization.py` (nomes em UPPER, anomalias tratadas, duplicatas removidas), cada fonte no seu **nível de registro original**. **Ainda não** passou por engenharia de features, imputação, encoding ou agregações de histórico (isso é a Gold).

A Silver tem **três tabelas**, uma por fonte do Kaggle:

## 1. `clean_data.parquet` — solicitações (application)

Base principal: 1 linha por solicitante (`SK_ID_CURR`). Contém o alvo `TARGET`.

- **Dimensão:** 307,511 linhas × 122 colunas
- **Numéricas:** 106 | **Categóricas:** 16

| Coluna | Tipo | % Nulos | Únicos | Mín | Máx | Descrição |
|---|---|---|---|---|---|---|
| `SK_ID_CURR` | int64 | 0.0% | 307,511 | 100002.00 | 456255.00 | Identificador da solicitação atual (1 por linha). Chave de junção do histórico. |
| `TARGET` | int64 | 0.0% | 2 | 0.00 | 1.00 | Alvo: 0 = paga em dia, 1 = inadimplente. |
| `NAME_CONTRACT_TYPE` | str | 0.0% | 2 | - | - | Identification if loan is cash or revolving |
| `CODE_GENDER` | str | 0.0% | 3 | - | - | Gender of the client |
| `FLAG_OWN_CAR` | str | 0.0% | 2 | - | - | Flag if the client owns a car |
| `FLAG_OWN_REALTY` | str | 0.0% | 2 | - | - | Flag if client owns a house or flat |
| `CNT_CHILDREN` | int64 | 0.0% | 15 | 0.00 | 19.00 | Number of children the client has |
| `AMT_INCOME_TOTAL` | float64 | 0.0% | 2,548 | 25650.00 | 117000000.00 | Income of the client |
| `AMT_CREDIT` | float64 | 0.0% | 5,603 | 45000.00 | 4050000.00 | Credit amount of the loan |
| `AMT_ANNUITY` | float64 | 0.0% | 13,672 | 1615.50 | 258025.50 | Loan annuity |
| `AMT_GOODS_PRICE` | float64 | 0.1% | 1,002 | 40500.00 | 4050000.00 | For consumer loans it is the price of the goods for which the loan is given |
| `NAME_TYPE_SUITE` | str | 0.4% | 7 | - | - | Who was accompanying client when he was applying for the loan |
| `NAME_INCOME_TYPE` | str | 0.0% | 8 | - | - | Clients income type (businessman, working, maternity leave,) |
| `NAME_EDUCATION_TYPE` | str | 0.0% | 5 | - | - | Level of highest education the client achieved |
| `NAME_FAMILY_STATUS` | str | 0.0% | 6 | - | - | Family status of the client |
| `NAME_HOUSING_TYPE` | str | 0.0% | 6 | - | - | What is the housing situation of the client (renting, living with parents, ...) |
| `REGION_POPULATION_RELATIVE` | float64 | 0.0% | 81 | 0.00 | 0.07 | Normalized population of region where client lives (higher number means the client lives in more populated region) |
| `DAYS_BIRTH` | int64 | 0.0% | 17,460 | 7489.00 | 25229.00 | Client's age in days at the time of application |
| `DAYS_EMPLOYED` | float64 | 18.0% | 12,573 | 0.00 | 17912.00 | How many days before the application the person started current employment |
| `DAYS_REGISTRATION` | float64 | 0.0% | 15,688 | 0.00 | 24672.00 | How many days before the application did client change his registration |
| `DAYS_ID_PUBLISH` | int64 | 0.0% | 6,168 | 0.00 | 7197.00 | How many days before the application did client change the identity document with which he applied for the loan |
| `OWN_CAR_AGE` | float64 | 66.0% | 62 | 0.00 | 91.00 | Age of client's car |
| `FLAG_MOBIL` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide mobile phone (1=YES, 0=NO) |
| `FLAG_EMP_PHONE` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide work phone (1=YES, 0=NO) |
| `FLAG_WORK_PHONE` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide home phone (1=YES, 0=NO) |
| `FLAG_CONT_MOBILE` | int64 | 0.0% | 2 | 0.00 | 1.00 | Was mobile phone reachable (1=YES, 0=NO) |
| `FLAG_PHONE` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide home phone (1=YES, 0=NO) |
| `FLAG_EMAIL` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide email (1=YES, 0=NO) |
| `OCCUPATION_TYPE` | str | 31.3% | 18 | - | - | What kind of occupation does the client have |
| `CNT_FAM_MEMBERS` | float64 | 0.0% | 17 | 1.00 | 20.00 | How many family members does client have |
| `REGION_RATING_CLIENT` | int64 | 0.0% | 3 | 1.00 | 3.00 | Our rating of the region where client lives (1,2,3) |
| `REGION_RATING_CLIENT_W_CITY` | int64 | 0.0% | 3 | 1.00 | 3.00 | Our rating of the region where client lives with taking city into account (1,2,3) |
| `WEEKDAY_APPR_PROCESS_START` | str | 0.0% | 7 | - | - | On which day of the week did the client apply for the loan |
| `HOUR_APPR_PROCESS_START` | int64 | 0.0% | 24 | 0.00 | 23.00 | Approximately at what hour did the client apply for the loan |
| `REG_REGION_NOT_LIVE_REGION` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level) |
| `REG_REGION_NOT_WORK_REGION` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match work address (1=different, 0=same, at region level) |
| `LIVE_REGION_NOT_WORK_REGION` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's contact address does not match work address (1=different, 0=same, at region level) |
| `REG_CITY_NOT_LIVE_CITY` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level) |
| `REG_CITY_NOT_WORK_CITY` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's permanent address does not match work address (1=different, 0=same, at city level) |
| `LIVE_CITY_NOT_WORK_CITY` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if client's contact address does not match work address (1=different, 0=same, at city level) |
| `ORGANIZATION_TYPE` | str | 0.0% | 58 | - | - | Type of organization where client works |
| `EXT_SOURCE_1` | float64 | 56.4% | 114,584 | 0.01 | 0.96 | Normalized score from external data source |
| `EXT_SOURCE_2` | float64 | 0.2% | 119,831 | 0.00 | 0.85 | Normalized score from external data source |
| `EXT_SOURCE_3` | float64 | 19.8% | 814 | 0.00 | 0.90 | Normalized score from external data source |
| `APARTMENTS_AVG` | float64 | 50.7% | 2,339 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `BASEMENTAREA_AVG` | float64 | 58.5% | 3,780 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BEGINEXPLUATATION_AVG` | float64 | 48.8% | 285 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BUILD_AVG` | float64 | 66.5% | 149 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `COMMONAREA_AVG` | float64 | 69.9% | 3,181 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `ELEVATORS_AVG` | float64 | 53.3% | 257 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `ENTRANCES_AVG` | float64 | 50.3% | 285 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMAX_AVG` | float64 | 49.8% | 403 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMIN_AVG` | float64 | 67.8% | 305 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LANDAREA_AVG` | float64 | 59.4% | 3,527 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LIVINGAPARTMENTS_AVG` | float64 | 68.4% | 1,868 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LIVINGAREA_AVG` | float64 | 50.2% | 5,199 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `NONLIVINGAPARTMENTS_AVG` | float64 | 69.4% | 386 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `NONLIVINGAREA_AVG` | float64 | 55.2% | 3,290 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `APARTMENTS_MODE` | float64 | 50.7% | 760 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `BASEMENTAREA_MODE` | float64 | 58.5% | 3,841 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BEGINEXPLUATATION_MODE` | float64 | 48.8% | 221 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BUILD_MODE` | float64 | 66.5% | 154 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `COMMONAREA_MODE` | float64 | 69.9% | 3,128 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `ELEVATORS_MODE` | float64 | 53.3% | 26 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `ENTRANCES_MODE` | float64 | 50.3% | 30 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMAX_MODE` | float64 | 49.8% | 25 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMIN_MODE` | float64 | 67.8% | 25 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LANDAREA_MODE` | float64 | 59.4% | 3,563 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LIVINGAPARTMENTS_MODE` | float64 | 68.4% | 736 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LIVINGAREA_MODE` | float64 | 50.2% | 5,301 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `NONLIVINGAPARTMENTS_MODE` | float64 | 69.4% | 167 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `NONLIVINGAREA_MODE` | float64 | 55.2% | 3,327 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `APARTMENTS_MEDI` | float64 | 50.7% | 1,148 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `BASEMENTAREA_MEDI` | float64 | 58.5% | 3,772 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BEGINEXPLUATATION_MEDI` | float64 | 48.8% | 245 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `YEARS_BUILD_MEDI` | float64 | 66.5% | 151 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `COMMONAREA_MEDI` | float64 | 69.9% | 3,202 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `ELEVATORS_MEDI` | float64 | 53.3% | 46 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `ENTRANCES_MEDI` | float64 | 50.3% | 46 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMAX_MEDI` | float64 | 49.8% | 49 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FLOORSMIN_MEDI` | float64 | 67.8% | 47 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LANDAREA_MEDI` | float64 | 59.4% | 3,560 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LIVINGAPARTMENTS_MEDI` | float64 | 68.4% | 1,097 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `LIVINGAREA_MEDI` | float64 | 50.2% | 5,281 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `NONLIVINGAPARTMENTS_MEDI` | float64 | 69.4% | 214 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `NONLIVINGAREA_MEDI` | float64 | 55.2% | 3,323 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `FONDKAPREMONT_MODE` | str | 68.4% | 4 | - | - | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `HOUSETYPE_MODE` | str | 50.2% | 3 | - | - | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `TOTALAREA_MODE` | float64 | 48.3% | 5,116 | 0.00 | 1.00 | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `WALLSMATERIAL_MODE` | str | 50.8% | 7 | - | - | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `EMERGENCYSTATE_MODE` | str | 47.4% | 2 | - | - | Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor |
| `OBS_30_CNT_SOCIAL_CIRCLE` | float64 | 0.3% | 33 | 0.00 | 348.00 | How many observation of client's social surroundings with observable 30 DPD (days past due) default |
| `DEF_30_CNT_SOCIAL_CIRCLE` | float64 | 0.3% | 10 | 0.00 | 34.00 | How many observation of client's social surroundings defaulted on 30 DPD (days past due) |
| `OBS_60_CNT_SOCIAL_CIRCLE` | float64 | 0.3% | 33 | 0.00 | 344.00 | How many observation of client's social surroundings with observable 60 DPD (days past due) default |
| `DEF_60_CNT_SOCIAL_CIRCLE` | float64 | 0.3% | 9 | 0.00 | 24.00 | How many observation of client's social surroundings defaulted on 60 (days past due) DPD |
| `DAYS_LAST_PHONE_CHANGE` | float64 | 0.0% | 3,773 | -4292.00 | 0.00 | How many days before application did client change phone |
| `FLAG_DOCUMENT_2` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 2 |
| `FLAG_DOCUMENT_3` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 3 |
| `FLAG_DOCUMENT_4` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 4 |
| `FLAG_DOCUMENT_5` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 5 |
| `FLAG_DOCUMENT_6` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 6 |
| `FLAG_DOCUMENT_7` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 7 |
| `FLAG_DOCUMENT_8` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 8 |
| `FLAG_DOCUMENT_9` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 9 |
| `FLAG_DOCUMENT_10` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 10 |
| `FLAG_DOCUMENT_11` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 11 |
| `FLAG_DOCUMENT_12` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 12 |
| `FLAG_DOCUMENT_13` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 13 |
| `FLAG_DOCUMENT_14` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 14 |
| `FLAG_DOCUMENT_15` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 15 |
| `FLAG_DOCUMENT_16` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 16 |
| `FLAG_DOCUMENT_17` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 17 |
| `FLAG_DOCUMENT_18` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 18 |
| `FLAG_DOCUMENT_19` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 19 |
| `FLAG_DOCUMENT_20` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 20 |
| `FLAG_DOCUMENT_21` | int64 | 0.0% | 2 | 0.00 | 1.00 | Did client provide document 21 |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | float64 | 13.5% | 5 | 0.00 | 4.00 | Number of enquiries to Credit Bureau about the client one hour before application |
| `AMT_REQ_CREDIT_BUREAU_DAY` | float64 | 13.5% | 9 | 0.00 | 9.00 | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | float64 | 13.5% | 9 | 0.00 | 8.00 | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) |
| `AMT_REQ_CREDIT_BUREAU_MON` | float64 | 13.5% | 24 | 0.00 | 27.00 | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) |
| `AMT_REQ_CREDIT_BUREAU_QRT` | float64 | 13.5% | 11 | 0.00 | 261.00 | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | float64 | 13.5% | 25 | 0.00 | 25.00 | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application) |

## 2. `clean_bureau.parquet` — histórico externo (bureau)

Créditos do cliente em OUTRAS instituições (birô). 1 linha por crédito (`SK_ID_BUREAU`); vários por cliente. Agregado em `BUREAU_*` na Gold.

- **Dimensão:** 1,716,428 linhas × 17 colunas
- **Numéricas:** 14 | **Categóricas:** 3

| Coluna | Tipo | % Nulos | Únicos | Mín | Máx | Descrição |
|---|---|---|---|---|---|---|
| `SK_ID_CURR` | int64 | 0.0% | 305,811 | 100001.00 | 456255.00 | Identificador da solicitação atual (1 por linha). Chave de junção do histórico. |
| `SK_ID_BUREAU` | int64 | 0.0% | 1,716,428 | 5000000.00 | 6843457.00 | - |
| `CREDIT_ACTIVE` | str | 0.0% | 4 | - | - | Status of the Credit Bureau (CB) reported credits |
| `CREDIT_CURRENCY` | str | 0.0% | 4 | - | - | Recoded currency of the Credit Bureau credit |
| `DAYS_CREDIT` | int64 | 0.0% | 2,923 | -2922.00 | 0.00 | How many days before current application did client apply for Credit Bureau credit |
| `CREDIT_DAY_OVERDUE` | int64 | 0.0% | 942 | 0.00 | 2792.00 | Number of days past due on CB credit at the time of application for related loan in our sample |
| `DAYS_CREDIT_ENDDATE` | float64 | 6.1% | 14,096 | -42060.00 | 31199.00 | Remaining duration of CB credit (in days) at the time of application in Home Credit |
| `DAYS_ENDDATE_FACT` | float64 | 36.9% | 2,917 | -42023.00 | 0.00 | Days since CB credit ended at the time of application in Home Credit (only for closed credit) |
| `AMT_CREDIT_MAX_OVERDUE` | float64 | 65.5% | 68,251 | 0.00 | 115987185.00 | Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample) |
| `CNT_CREDIT_PROLONG` | int64 | 0.0% | 10 | 0.00 | 9.00 | How many times was the Credit Bureau credit prolonged |
| `AMT_CREDIT_SUM` | float64 | 0.0% | 236,708 | 0.00 | 585000000.00 | Current credit amount for the Credit Bureau credit |
| `AMT_CREDIT_SUM_DEBT` | float64 | 15.0% | 221,035 | 0.00 | 170100000.00 | Current debt on Credit Bureau credit |
| `AMT_CREDIT_SUM_LIMIT` | float64 | 34.5% | 51,726 | -586406.11 | 4705600.32 | Current credit limit of credit card reported in Credit Bureau |
| `AMT_CREDIT_SUM_OVERDUE` | float64 | 0.0% | 1,616 | 0.00 | 3756681.00 | Current amount overdue on Credit Bureau credit |
| `CREDIT_TYPE` | str | 0.0% | 15 | - | - | Type of Credit Bureau credit (Car, cash,...) |
| `DAYS_CREDIT_UPDATE` | int64 | 0.0% | 2,982 | -41947.00 | 372.00 | How many days before loan application did last information about the Credit Bureau credit come |
| `AMT_ANNUITY` | float64 | 71.5% | 40,321 | 0.00 | 118453423.50 | Annuity of the Credit Bureau credit |

## 3. `clean_previous_application.parquet` — histórico interno (previous)

Pedidos de crédito anteriores do cliente na Home Credit. 1 linha por pedido (`SK_ID_PREV`); vários por cliente. Agregado em `PREV_*` na Gold.

- **Dimensão:** 1,670,214 linhas × 37 colunas
- **Numéricas:** 21 | **Categóricas:** 16

| Coluna | Tipo | % Nulos | Únicos | Mín | Máx | Descrição |
|---|---|---|---|---|---|---|
| `SK_ID_PREV` | int64 | 0.0% | 1,670,214 | 1000001.00 | 2845382.00 | - |
| `SK_ID_CURR` | int64 | 0.0% | 338,857 | 100001.00 | 456255.00 | Identificador da solicitação atual (1 por linha). Chave de junção do histórico. |
| `NAME_CONTRACT_TYPE` | str | 0.0% | 4 | - | - | Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application |
| `AMT_ANNUITY` | float64 | 22.3% | 357,959 | 0.00 | 418058.15 | Annuity of previous application |
| `AMT_APPLICATION` | float64 | 0.0% | 93,885 | 0.00 | 6905160.00 | For how much credit did client ask on the previous application |
| `AMT_CREDIT` | float64 | 0.0% | 86,803 | 0.00 | 6905160.00 | Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT |
| `AMT_DOWN_PAYMENT` | float64 | 53.6% | 29,278 | -0.90 | 3060045.00 | Down payment on the previous application |
| `AMT_GOODS_PRICE` | float64 | 23.1% | 93,885 | 0.00 | 6905160.00 | Goods price of good that client asked for (if applicable) on the previous application |
| `WEEKDAY_APPR_PROCESS_START` | str | 0.0% | 7 | - | - | On which day of the week did the client apply for previous application |
| `HOUR_APPR_PROCESS_START` | int64 | 0.0% | 24 | 0.00 | 23.00 | Approximately at what day hour did the client apply for the previous application |
| `FLAG_LAST_APPL_PER_CONTRACT` | str | 0.0% | 2 | - | - | Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract |
| `NFLAG_LAST_APPL_IN_DAY` | int64 | 0.0% | 2 | 0.00 | 1.00 | Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice |
| `RATE_DOWN_PAYMENT` | float64 | 53.6% | 207,033 | -0.00 | 1.00 | Down payment rate normalized on previous credit |
| `RATE_INTEREST_PRIMARY` | float64 | 99.6% | 148 | 0.03 | 1.00 | Interest rate normalized on previous credit |
| `RATE_INTEREST_PRIVILEGED` | float64 | 99.6% | 25 | 0.37 | 1.00 | Interest rate normalized on previous credit |
| `NAME_CASH_LOAN_PURPOSE` | str | 0.0% | 25 | - | - | Purpose of the cash loan |
| `NAME_CONTRACT_STATUS` | str | 0.0% | 4 | - | - | Contract status (approved, cancelled, ...) of previous application |
| `DAYS_DECISION` | int64 | 0.0% | 2,922 | -2922.00 | -1.00 | Relative to current application when was the decision about previous application made |
| `NAME_PAYMENT_TYPE` | str | 0.0% | 4 | - | - | Payment method that client chose to pay for the previous application |
| `CODE_REJECT_REASON` | str | 0.0% | 9 | - | - | Why was the previous application rejected |
| `NAME_TYPE_SUITE` | str | 49.1% | 7 | - | - | Who accompanied client when applying for the previous application |
| `NAME_CLIENT_TYPE` | str | 0.0% | 4 | - | - | Was the client old or new client when applying for the previous application |
| `NAME_GOODS_CATEGORY` | str | 0.0% | 28 | - | - | What kind of goods did the client apply for in the previous application |
| `NAME_PORTFOLIO` | str | 0.0% | 5 | - | - | Was the previous application for CASH, POS, CAR, |
| `NAME_PRODUCT_TYPE` | str | 0.0% | 3 | - | - | Was the previous application x-sell o walk-in |
| `CHANNEL_TYPE` | str | 0.0% | 8 | - | - | Through which channel we acquired the client on the previous application |
| `SELLERPLACE_AREA` | int64 | 0.0% | 2,097 | -1.00 | 4000000.00 | Selling area of seller place of the previous application |
| `NAME_SELLER_INDUSTRY` | str | 0.0% | 11 | - | - | The industry of the seller |
| `CNT_PAYMENT` | float64 | 22.3% | 49 | 0.00 | 84.00 | Term of previous credit at application of the previous application |
| `NAME_YIELD_GROUP` | str | 0.0% | 5 | - | - | Grouped interest rate into small medium and high of the previous application |
| `PRODUCT_COMBINATION` | str | 0.0% | 17 | - | - | Detailed product combination of the previous application |
| `DAYS_FIRST_DRAWING` | float64 | 40.3% | 2,838 | -2922.00 | 365243.00 | Relative to application date of current application when was the first disbursement of the previous application |
| `DAYS_FIRST_DUE` | float64 | 40.3% | 2,892 | -2892.00 | 365243.00 | Relative to application date of current application when was the first due supposed to be of the previous application |
| `DAYS_LAST_DUE_1ST_VERSION` | float64 | 40.3% | 4,605 | -2801.00 | 365243.00 | Relative to application date of current application when was the first due of the previous application |
| `DAYS_LAST_DUE` | float64 | 40.3% | 2,873 | -2889.00 | 365243.00 | Relative to application date of current application when was the last due date of the previous application |
| `DAYS_TERMINATION` | float64 | 40.3% | 2,830 | -2874.00 | 365243.00 | Relative to application date of current application when was the expected termination of the previous application |
| `NFLAG_INSURED_ON_APPROVAL` | float64 | 40.3% | 2 | 0.00 | 1.00 | Did the client requested insurance during the previous application |
