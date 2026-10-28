# Índice de la entrega — MLOps — Defiende el servidor

> Completa todo lo que está entre `< >` con los valores reales (cópialos de `evidencias/hf_versions.json`,
> `evidencias/seleccion.json` y `git log`). No dejes ningún `< >`.

## Integrantes y repositorios
- Integrantes y grupo: `<Nombre completo>` — Grupo `<A/B>`
- Repositorio de código (GitHub): `<https://github.com/USUARIO/defiende-servidor>`
- Commit final de la entrega: `<SHA completo de 40 caracteres>`
- Repositorio de modelos (Hugging Face): `<https://huggingface.co/USUARIO_HF/defiende-servidor-unsw>`
- Almacenamiento DVC: `https://dagshub.com/c851458/Taller_visu_2.dvc` (configurado en `.dvc/config`; repositorio público).
  DagsHub exige iniciar sesión para leer DVC aunque el repo sea público: el revisor usa **su propia** cuenta
  gratuita de DagsHub y su token (`dvc remote modify origin --local auth basic`, `... user <usuario>`,
  `... password <token>`; ver README). No se comparten tokens.

## Versiones

| Campo | v1 | v2 |
|---|---|---|
| Experimento | `e1_logreg_base` | `e4_hgb_threshold` |
| Commit de código que la produjo (tag Git) | `<sha>` (tag `v1`) | `<sha>` (tag `v2`) |
| Datos DVC originales (md5 de `data/raw.dvc`) | `<md5>` | `<md5>` |
| Datos DVC preparados (md5 en `dvc.lock`) | `<md5>` | `<md5>` |
| Run ID de MLflow | `<run_id>` | `<run_id>` |
| Revisión Hugging Face (SHA completo, tag) | `<sha>` (`v1`) | `<sha>` (`v2`) |
| sha256 de `model.joblib` | `<sha256>` | `<sha256>` |
| Umbral | `0.5` | `<umbral>` |

Cambio v1 → v2: familia (regresión logística → gradient boosting de árboles) y umbral elegido en
validación con FPR ≤ 5 %. Mejora precision, FPR y macro-F1; empeora recall de ataque (ver informe).

## Ubicación de las evidencias
- Auditoría de datos y particiones: `evidencias/auditoria.json`
- Tabla de experimentos: `evidencias/experimentos.csv`
- Exportación de MLflow (4 runs): `evidencias/mlflow/` (cruce por `run_id`; ver `COMO_CONSULTAR.md`)
- Selección congelada antes del test: `evidencias/seleccion.json` (commit `<sha>` anterior al de `metricas-test.json`)
- Versiones publicadas: `evidencias/hf_versions.json`
- Métricas de test: `evidencias/metricas-test.json`
- Predicciones de test (82.332 filas): `evidencias/predicciones-test.csv`
- Prueba de recuperación: `evidencias/recuperacion/` (`entrada.csv`, `antes_v*.csv`, `despues_v*.csv`, `comparacion.json`, `recuperacion.log`)
- Informe: `informe.pdf` (adjunto aparte)

## Comandos
Ver [README.md](README.md). Resumen:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
dvc pull && python -m src.recovery verify-data
python -m src.train configs/experiments/e1_logreg_base.yaml
python src/predict.py --repo-id <USUARIO_HF>/defiende-servidor-unsw --revision <SHA_V2> --input evidencias/recuperacion/entrada.csv --output salida.csv
DAGSHUB_USER=<usuario> DAGSHUB_TOKEN=<token> bash scripts/prueba_recuperacion.sh <URL_GITHUB> <COMMIT>
```

## Tiempo y recursos (Intel i5-10300H, 8 hilos, 8 GB RAM, WSL2, sin GPU)
- `prepare` (auditoría + particiones): ~16 s
- Entrenamiento (ajuste del modelo, `fit_seconds` en MLflow): E1 ~19 s, E2 ~21 s, E3 ~29 s, E4 ~32 s (~55 s por experimento con carga de datos y registro; RAM máx. ~0,9 GB)
- Prueba de recuperación completa (clonar, instalar dependencias, `dvc pull`, descargar v1/v2 y comparar): ~146 s, RAM máx. ~0,2 GB

## Limitaciones conocidas
- UNSW-NB15 es un dataset histórico de laboratorio (2015): no demuestra eficacia en una red actual.
- ~10 % de las filas de test tienen entradas idénticas a filas de train/val (ver `auditoria.json`).
- El umbral de v2 se eligió y se reporta sobre la misma validación (estimación algo optimista).
- `model.joblib` es pickle: requiere la misma versión de scikit-learn y una fuente confiable.

## Uso de IA, código externo y contribuciones
- IA: `<Claude (Anthropic), asistente de código en VS Code>` — usado para `<estructurar el proyecto, escribir scripts y documentación>`. Verifiqué: `<huellas SHA-256, ausencia de fuga entre train/val, ejecución de los 4 experimentos, prueba de recuperación, recálculo de métricas>`.
- Código externo: ninguno copiado; se usaron las APIs documentadas de scikit-learn, MLflow, DVC y huggingface_hub.
- Datos: UNSW-NB15 (Moustafa & Slay, 2015), copia mirror verificada por SHA-256.
- Contribución de cada integrante: `<...>`
