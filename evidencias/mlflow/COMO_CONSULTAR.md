# Exportación de MLflow

- `all_runs.json`: todos los runs FINISHED (parámetros, métricas, tags).
- `runs/<run_id>/`: run usado en `experimentos.csv`, con `config.json` de la solución (umbral, hash del modelo, commit, DVC) y métricas de validación.
- Cruce: la columna `run_id` de `evidencias/experimentos.csv` es el nombre de la carpeta.

Para abrir la base original: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
