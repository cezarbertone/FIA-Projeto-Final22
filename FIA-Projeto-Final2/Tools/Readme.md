# Tools/ — Utilitários de desenvolvimento

Scripts **auxiliares de desenvolvimento** que ficam fora de `DataPipeline/`/`Model/` para manter
ali só os arquivos exigidos pela especificação. Não fazem parte do pipeline de execução — nenhum
módulo os importa em runtime.

| Script | Saída | Por que é um script |
|---|---|---|
| `generate_data_dictionary.py` | `DataPipeline/dicionario_clean_data.md` + `dicionario_abt.md` | Calcula os metadados (tipo, % de nulos, nº de únicos, min/máx) de **todas as colunas** das camadas Silver/Gold e cruza com o dicionário oficial do Kaggle — inviável manter à mão. |

```powershell
# a partir da raiz do projeto, com o venv ativo
python Tools/generate_data_dictionary.py
```