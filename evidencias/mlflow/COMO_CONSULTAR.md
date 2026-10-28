# Exportación de MLflow

- `all_runs.json`: todos los runs FINISHED (parámetros, métricas, tags).
- `runs/<run_id>/`: run usado en `experimentos.csv`, con `config.json` de la solución (umbral, hash del modelo, commit, DVC) y métricas de validación.
- `runs/<run_id>/artifacts.json`: manifiesto de **todos** los artefactos del run (`run/...`) y de su modelo registrado en MLflow (`logged_model/<model_id>/...`), con tamaño y SHA-256. Los de hasta 5 MB se copian en `runs/<run_id>/artifacts/` (`copied: true`); el modelo de E3 (~110 MB) solo queda con su hash.
- Cruce: la columna `run_id` de `evidencias/experimentos.csv` es el nombre de la carpeta, y `model_sha256` coincide con el `sha256` de `run/solution/model.joblib` en `artifacts.json`.
- Cargar una solución exportada y predecir: `python src/predict.py --model-dir evidencias/mlflow/runs/<run_id>/artifacts/run/solution --input evidencias/recuperacion/entrada.csv --output salida.csv` (comprueba el hash del modelo antes de cargarlo).

Para abrir la base original: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
