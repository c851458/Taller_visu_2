# Índice de la entrega — MLOps — Defiende el servidor

## Integrantes y repositorios
- Integrante y grupo: Francy Carolina Umbarila Rodriguez — Grupo A (entrega individual)
- Repositorio de código (GitHub): https://github.com/c851458/Taller_visu_2
- Commit final de la entrega: el commit con tag `entrega` (`git rev-parse entrega`); su SHA completo va también en el mensaje de envío
- Repositorio de modelos (Hugging Face): https://huggingface.co/c851458/defiende-servidor-unsw
- Almacenamiento DVC: `https://dagshub.com/c851458/Taller_visu_2.dvc` (configurado en `.dvc/config`; repositorio público).
  DagsHub exige iniciar sesión para leer DVC aunque el repo sea público: el revisor usa **su propia** cuenta
  gratuita de DagsHub y su token (`dvc remote modify origin --local auth basic`, `... user <usuario>`,
  `... password <token>`; ver README). No se comparten tokens.

## Versiones

| Campo | v1 | v2 |
|---|---|---|
| Experimento | `e1_logreg_base` | `e4_hgb_threshold` |
| Commit de código que entrenó el modelo | `bec239b75d48d6df9311545c4456c78339f4344e` | `057957069ef92fba2a9ebef2c8b70f61fd405c48` |
| Tag Git de la solución | `v1` = `f97b4200f0bb469e4c938b2da5337b1220b9a4d6` | `v2` = `269c2ab085012903f1e79b69ffebd137404c005c` |
| Datos DVC originales (md5 de `data/raw.dvc`) | `688fd73cceff08e318906ecd85de60c4.dir` | `688fd73cceff08e318906ecd85de60c4.dir` |
| Datos DVC preparados (md5 en `dvc.lock`) | `5832cbec6fbaff34eaf128c733c18932.dir` | `5832cbec6fbaff34eaf128c733c18932.dir` |
| Tags Git de datos | `datos-original`, `datos-preparados` | ídem (mismos datos y particiones) |
| Run ID de MLflow | `52c5225915704ba3bbe1c91635ea6846` | `c5503f3f886f4309927f98bac0288479` |
| Revisión Hugging Face (SHA completo, tag) | `ca61db0e29f5c1816f68790f44d6af65a23ab082` (`v1`) | `9fb15e97dc936b6ba90bbdc73a8de92781462f75` (`v2`) |
| sha256 de `model.joblib` | `cd0d72260f75535658480f24ab1b3b2a725cae1b64c526b48e690642b4756807` | `689fb002a0eaa577af78ee78cec88043a53e93fc17ae96a4f8aba4c93dda03e9` |
| Umbral (`y_pred = 1 si score_attack >= umbral`) | `0.5` | `0.6574` |

Revisión posterior de la model card de v2 (solo `README.md`, mismo `model.joblib`):
`b6b3a5950a22bbde67b237771cef3667c27da5f1`. El tag `v2` sigue apuntando a la revisión evaluada.

Cambio v1 → v2: familia (regresión logística → gradient boosting de árboles) y umbral elegido en
validación con FPR ≤ 5 %. En validación mejora precision (0,928 → 0,977), FPR (0,170 → 0,050) y
macro-F1 (0,926 → 0,951); empeora recall de ataque (0,989 → 0,960). Ver informe.

## Ubicación de las evidencias
- Auditoría de datos y particiones: `evidencias/auditoria.json`
- Tabla de experimentos: `evidencias/experimentos.csv`
- Exportación de MLflow (4 runs): `evidencias/mlflow/` (cruce por `run_id`; `runs/<run_id>/artifacts.json` lista cada artefacto con su SHA-256 y `artifacts/` copia los de hasta 5 MB, incluidos los modelos de E1, E2 y E4; ver `COMO_CONSULTAR.md`)
- Interpretación de errores en validación (4 configuraciones, por `attack_cat` y falsos positivos por `service`/`state`/`sttl`; posterior, no usado para decidir): `evidencias/errores-validacion.csv`, `evidencias/errores-validacion.json`, predicciones de validación por fila en `evidencias/predicciones-val.csv`
- Selección congelada antes del test: `evidencias/seleccion.json` (commit `269c2ab085012903f1e79b69ffebd137404c005c`, anterior al commit de `metricas-test.json`, `d71d7f66addf2349281c0217b48e99f0bbbe0590`)
- Versiones publicadas: `evidencias/hf_versions.json`
- Métricas de test: `evidencias/metricas-test.json`
- Predicciones de test (82.332 filas): `evidencias/predicciones-test.csv`
- Diagnóstico de falsos positivos en test (posterior, no usado para decidir): `evidencias/analisis-fp-test.json`
- Prueba de recuperación: `evidencias/recuperacion/` (`entrada.csv`, `antes_v*.csv`, `despues_v*.csv`, `comparacion.json`, `recuperacion.log`)
- Informe: `informe.pdf` (en el repositorio y adjunto aparte)

## Comandos
Ver [README.md](README.md). Resumen:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <TU_USUARIO_DAGSHUB>
dvc remote modify origin --local password <TU_TOKEN_DAGSHUB>
dvc pull && python -m src.recovery verify-data
python -m src.train configs/experiments/e1_logreg_base.yaml
python src/predict.py --repo-id c851458/defiende-servidor-unsw --revision 9fb15e97dc936b6ba90bbdc73a8de92781462f75 --input evidencias/recuperacion/entrada.csv --output salida_v2.csv
DAGSHUB_USER=<TU_USUARIO_DAGSHUB> DAGSHUB_TOKEN=<TU_TOKEN_DAGSHUB> bash scripts/prueba_recuperacion.sh https://github.com/c851458/Taller_visu_2.git "$(git rev-parse entrega)"
bash scripts/empaquetar.sh entrega   # ZIP de la entrega desde el commit final
```

## Tiempo y recursos (Intel i5-10300H, 8 hilos, 8 GB RAM, WSL2, sin GPU)
- `prepare` (auditoría + particiones): ~16 s
- Entrenamiento (ajuste del modelo, `fit_seconds` en MLflow): E1 ~19 s, E2 ~21 s, E3 ~29 s, E4 ~32 s (~55 s por experimento con carga de datos y registro; RAM máx. ~0,9 GB)
- Prueba de recuperación completa (clonar, instalar dependencias, `dvc pull`, descargar v1/v2 y comparar): ~146 s, RAM máx. ~0,2 GB

## Limitaciones conocidas
- UNSW-NB15 es un dataset histórico de laboratorio (2015): no demuestra eficacia en una red actual.
- En test la FPR de v2 sube de 0,050 (validación) a 0,192: el tráfico normal del archivo de test es
  distinto (40,7 % de normales con `sttl=254` frente a 19,5 % en el archivo de entrenamiento) y ahí se
  concentra el 88 % de las falsas alarmas (`evidencias/analisis-fp-test.json`).
- ~10 % de las filas de test tienen entradas idénticas a filas de train/val (ver `auditoria.json`).
- El umbral de v2 se eligió y se reporta sobre la misma validación (estimación algo optimista).
- `model.joblib` es pickle: requiere la misma versión de scikit-learn y una fuente confiable.
- Leer el almacenamiento DVC de DagsHub requiere una cuenta (gratuita) de DagsHub.

## Uso de IA, código externo y contribuciones
- IA: Claude (Anthropic), asistente de código en VS Code (Claude Code). Usado para estructurar el
  proyecto, escribir los scripts de `src/` y `scripts/`, ejecutar los pasos de la guía, analizar los
  falsos positivos del test y redactar la documentación y el informe. Verifiqué: huellas SHA-256 de los
  CSV contra el PDF del taller, ausencia de grupos compartidos entre train y val, que los 4 runs usan
  la misma partición, que los hashes de los modelos descargados de Hugging Face coinciden, la prueba de
  recuperación en entorno nuevo (`RESULTADO: OK`) y el recálculo de métricas de test.
- Código externo: ninguno copiado; se usaron las APIs documentadas de scikit-learn, MLflow, DVC y huggingface_hub.
- Datos: UNSW-NB15 (Moustafa & Slay, 2015), copia mirror verificada por SHA-256.
- Contribución: trabajo individual; Francy Carolina Umbarila Rodriguez realizó y es responsable de todas las partes.
