# Defiende el servidor — UNSW-NB15 con Git, DVC, MLflow y Hugging Face

Clasificación binaria **normal (0) / ataque (1)** sobre las particiones publicadas de UNSW-NB15,
con trazabilidad completa:

```
Código: commit Git → Datos: referencia DVC → Experimento: run ID de MLflow → Modelo: revisión (SHA) de Hugging Face
```

El índice de la entrega (identificadores, SHAs, ubicación de evidencias) está en [entrega.md](entrega.md).

## Requisitos

- Python **3.10.12** (probado en Linux/WSL2, CPU, 8 núcleos, 8 GB RAM).
- Dependencias fijadas en `requirements.txt` (scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6,
  mlflow 3.17.2, dvc 3.67.1, huggingface_hub 2.3.0).

## Instalación

```bash
git clone https://github.com/c851458/Taller_visu_2.git defiende-servidor && cd defiende-servidor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recuperar los datos (DVC)

El remote DVC está configurado en `.dvc/config`: https://dagshub.com/c851458/Taller_visu_2 (repositorio
público). DagsHub exige iniciar sesión para leer el almacenamiento DVC aunque el repositorio sea público,
así que cada persona usa **su propia** cuenta gratuita de DagsHub (*Settings → Tokens* para copiar el token).
Las credenciales quedan en `.dvc/config.local`, que Git ignora; no se comparten tokens.

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <TU_USUARIO_DAGSHUB>
dvc remote modify origin --local password <TU_TOKEN_DAGSHUB>
dvc pull                      # data/raw (originales) y data/prepared (particiones)
python -m src.recovery verify-data   # comprueba SHA-256 contra configs/params.yaml y evidencias/auditoria.json
```

Versiones de datos:
- **Originales**: `data/raw.dvc` (CSV recibidos + `procedencia.json`).
- **Preparados**: salida de la etapa `prepare` en `dvc.lock` (`train.csv`, `val.csv`, `test.csv`, `splits.csv`).

Para volver a generarlos desde cero: `python -m src.download_data && dvc repro`.

## Ejecutar un experimento

```bash
# Requiere árbol de trabajo limpio (git commit) para que el run quede ligado a un commit
python -m src.train configs/experiments/e1_logreg_base.yaml
python -m src.export_mlflow          # actualiza evidencias/experimentos.csv y evidencias/mlflow/
mlflow ui --backend-store-uri sqlite:///mlflow.db   # opcional, para explorar los runs locales
```

Configuraciones: `configs/experiments/e1_logreg_base.yaml`, `e2_logreg_log1p.yaml`,
`e3_random_forest.yaml`, `e4_hgb_threshold.yaml`. Cada una tiene hipótesis, cambio, parámetros y
política de umbral (elegido solo con validación).

## Descargar cada modelo y predecir

Los SHAs completos están en `evidencias/hf_versions.json` y en `entrega.md`.

```bash
python src/predict.py --repo-id c851458/defiende-servidor-unsw --revision ca61db0e29f5c1816f68790f44d6af65a23ab082 \
  --input evidencias/recuperacion/entrada.csv --output salida_v1.csv
python src/predict.py --repo-id c851458/defiende-servidor-unsw --revision 9fb15e97dc936b6ba90bbdc73a8de92781462f75 \
  --input evidencias/recuperacion/entrada.csv --output salida_v2.csv
```

Salida: `row_id, score_attack, y_pred`, con la regla `y_pred = 1 si score_attack >= umbral`.

## Prueba de recuperación en entorno nuevo

```bash
DAGSHUB_USER=<TU_USUARIO_DAGSHUB> DAGSHUB_TOKEN=<TU_TOKEN_DAGSHUB> \
  bash scripts/prueba_recuperacion.sh https://github.com/c851458/Taller_visu_2.git "$(git rev-parse HEAD)"
```

Clona en una carpeta nueva, crea un entorno nuevo, configura tus credenciales DagsHub solo en el
`.dvc/config.local` del clon, hace `dvc pull` con caché vacía, verifica huellas,
descarga v1 y v2 por SHA con una caché de Hugging Face vacía y compara con las salidas guardadas
(tolerancia absoluta 1e-9 en `score_attack`).

## Evaluación final (una sola vez)

```bash
python -m src.evaluate_test       # exige evidencias/seleccion.json confirmado en Git
python -m src.recompute_metrics   # recalcula métricas desde predicciones-test.csv
```

## Estructura

```
src/            código (download_data, prepare, train, export_mlflow, publish_hf,
                select_model, evaluate_test, recompute_metrics, recovery, predict)
configs/        parámetros, 4 experimentos, regla de selección, conclusiones
scripts/        prueba_recuperacion.sh
dvc.yaml/.lock  etapa prepare (datos preparados)
data/raw.dvc    datos originales
evidencias/     auditoria.json, experimentos.csv, seleccion.json, metricas-test.json,
                predicciones-test.csv, hf_versions.json, mlflow/, recuperacion/
```
