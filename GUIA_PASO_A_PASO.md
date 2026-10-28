# Guía paso a paso — Defiende el servidor

**El orden importa.** Cada paso deja un commit que demuestra *cuándo* pasó cada cosa (por ejemplo,
que la selección se congeló antes del test). Ejecuta todo desde la carpeta del proyecto con el
entorno activado:

```bash
cd ~/code/taller_visu2_yo
source .venv/bin/activate
export MLFLOW_DISABLE_AGENT_HINT=1
```

---

## Paso 0 — Cuentas y accesos (hazlo primero: puede tardar)

| Plataforma | Qué necesitas |
|---|---|
| GitHub | Repositorio **vacío** (sin README), p. ej. `defiende-servidor`. Público, o privado invitando al docente. |
| DagsHub (remote DVC) | Entra en dagshub.com **con tu cuenta de GitHub** → *Create → New repository → Connect a repository → GitHub* → elige `defiende-servidor`. Luego, en *Settings → Tokens*, copia tu token. |
| Hugging Face | En *Settings → Access Tokens*, crea un token con permiso **write**. |
| Docente | Pide **ya** sus usuarios de GitHub/HF por Teams si vas a usar repos privados. Lo más sencillo es dejar los tres repositorios públicos. |

**Nunca** pongas tokens en archivos del repositorio.

## Paso 1 — Identidad Git y primer commit (solo código)

```bash
git config user.name "TU NOMBRE COMPLETO"
git config user.email "tu_correo@ejemplo.com"

git add .gitignore .dvcignore .dvc README.md entrega.md GUIA_PASO_A_PASO.md requirements.txt \
        dvc.yaml src scripts configs/params.yaml configs/seleccion.yaml \
        configs/experiments/e1_logreg_base.yaml
git commit -m "Estructura del proyecto: descarga, auditoría, entrenamiento y publicación"

git remote add origin https://github.com/TU_USUARIO/defiende-servidor.git
git push -u origin main
```

## Paso 2 — Versión DVC 1: datos originales

```bash
python -m src.download_data        # ya están descargados; verifica las huellas SHA-256 del PDF
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "Datos: versión original (CSV recibidos UNSW-NB15 + procedencia)"
git tag datos-original
```

## Paso 3 — Versión DVC 2: datos preparados y particiones

```bash
dvc repro                          # ejecuta src/prepare.py: auditoría + train/val por grupos
git add dvc.lock evidencias/auditoria.json
git commit -m "Datos: versión preparada (sin contradicciones, train/val por grupos, semilla 2026)"
git tag datos-preparados
```

Revisa `evidencias/auditoria.json`. Deberías ver: 940 filas contradictorias eliminadas, train=139.166,
val=35.235, test=82.332, 0 grupos compartidos entre train y val y un solapamiento del test del 10,1 %.

## Paso 4 — Remote DVC en DagsHub (no sirve uno local)

```bash
dvc remote add -d origin https://dagshub.com/TU_USUARIO/defiende-servidor.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user TU_USUARIO
dvc remote modify origin --local password TU_TOKEN_DAGSHUB   # queda en .dvc/config.local (ignorado)
dvc push
git add .dvc/config
git commit -m "DVC: remote en DagsHub"
git push origin main --tags
```

## Paso 5 — Experimento E1 y publicación de v1

```bash
python -m src.train configs/experiments/e1_logreg_base.yaml

hf auth login                       # pega el token write de Hugging Face
python -m src.publish_hf --version v1 --exp e1_logreg_base --repo-id TU_USUARIO_HF/defiende-servidor-unsw
git add evidencias/hf_versions.json
git commit -m "v1: regresión logística base publicada en Hugging Face"
git tag v1
git push origin main --tags
```

## Paso 6 — Experimentos E2, E3 y E4 (un commit por hipótesis)

```bash
git add configs/experiments/e2_logreg_log1p.yaml && git commit -m "E2: hipótesis log1p en variables de cola larga"
python -m src.train configs/experiments/e2_logreg_log1p.yaml

git add configs/experiments/e3_random_forest.yaml && git commit -m "E3: hipótesis cambio de familia a Random Forest"
python -m src.train configs/experiments/e3_random_forest.yaml

git add configs/experiments/e4_hgb_threshold.yaml && git commit -m "E4: HistGradientBoosting + umbral con FPR<=5% en validación"
python -m src.train configs/experiments/e4_hgb_threshold.yaml
```

## Paso 7 — Exportar MLflow y escribir conclusiones

1. Compara tus números con `configs/conclusiones.yaml` y ajusta los textos si hace falta.
2. Exporta y guarda:

```bash
python -m src.export_mlflow
git add configs/conclusiones.yaml evidencias/experimentos.csv evidencias/mlflow
git commit -m "Experimentos: exportación de 4 runs de MLflow y conclusiones de validación"
```

## Paso 8 — Elegir v2 con validación, publicarla y congelar la selección

```bash
python -m src.select_model          # indica el experimento elegido y pide publicarlo
python -m src.publish_hf --version v2 --exp e4_hgb_threshold --repo-id TU_USUARIO_HF/defiende-servidor-unsw
git add evidencias/hf_versions.json && git commit -m "v2: HistGradientBoosting con umbral de validación publicado"

python -m src.select_model          # ahora escribe evidencias/seleccion.json
git add evidencias/seleccion.json
git commit -m "Congela selección de v2 (modelo, umbral y hash) antes de evaluar test"
git tag v2
git push origin main --tags
```

## Paso 9 — Prueba de recuperación en entorno nuevo

```bash
python -m src.recovery sample       # 30 filas fijas de validación + salidas "antes" de v1 y v2
git add evidencias/recuperacion && git commit -m "Recuperación: entrada fija y salidas de referencia"
git push origin main

bash scripts/prueba_recuperacion.sh https://github.com/TU_USUARIO/defiende-servidor.git "$(git rev-parse HEAD)"
```

El script clona en `~/recuperacion_FECHA/`, crea un `.venv` nuevo, hace `dvc pull` **sin tus credenciales**
(así prueba el acceso del docente), verifica las huellas, descarga v1 y v2 por SHA con una caché de HF
vacía y compara. Al final debe mostrar `RESULTADO: OK`. Después:

```bash
cp ~/recuperacion_FECHA/salidas/* evidencias/recuperacion/
git add evidencias/recuperacion && git commit -m "Recuperación: log y comparación en entorno nuevo (OK)"
```

Si `dvc pull` pide credenciales, el repo de DagsHub está privado: hazlo público (o invita al docente).

## Paso 10 — Evaluación final en test (UNA sola vez)

```bash
python -m src.evaluate_test         # se niega a correr si seleccion.json no está en un commit
python -m src.recompute_metrics     # debe decir OK
python -m src.publish_hf --version v2 --repo-id TU_USUARIO_HF/defiende-servidor-unsw --card-only
git add evidencias/metricas-test.json evidencias/predicciones-test.csv evidencias/hf_versions.json
git commit -m "Test final de v2 congelada; model card actualizada sin cambiar el modelo"
git push origin main --tags
```

Si el test sale peor que la validación, **no cambies nada**: explícalo en el informe (solapamiento,
cambio de distribución entre los archivos, etc.).

## Paso 11 — Documentos y envío

1. Completa todos los `< >` de `entrega.md` con los valores de `evidencias/hf_versions.json`,
   `evidencias/seleccion.json` y `git log --oneline --decorate`. Haz commit y push; **ese** es el commit final.
2. Informe PDF (máximo 5 páginas): datos y limpieza, 4 experimentos (tabla), v1 frente a v2, umbral,
   test y limitaciones. Usa exactamente las cifras de `evidencias/`.
3. ZIP solo con lo versionado (sin `.venv`, datos ni cachés):
   ```bash
   git archive --format=zip --prefix=proyecto/ -o ../entrega_defiende_servidor.zip HEAD
   ```
4. Comprueba que en otra sesión (incógnito) se ven el repo de GitHub, el de HF y el de DagsHub.
5. Envía **por un solo canal** (Teams o correo). Asunto: `MLOps — Defiende el servidor — Grupo X — Nombre`.
   Incluye: nombres y grupo, URL de GitHub, commit final completo, URL de HF, SHA de v1 y v2, URL del
   remote DagsHub y cómo acceder, el ZIP y el PDF.

---

## Lista de comprobación de la rúbrica

- [ ] Git (15): historial con commits explicados; tags `v1`/`v2`; `requirements.txt` fijado; sin tokens.
- [ ] DVC (20): `auditoria.json`; `data/raw.dvc` + `dvc.lock`; `dvc pull` desde DagsHub con huellas OK.
- [ ] MLflow (20): 4 runs de 2 familias con hipótesis; `evidencias/mlflow/` + `experimentos.csv`; umbral elegido en validación.
- [ ] Hugging Face (20): 2 revisiones con SHA, model card, enlaces a commit/DVC/run.
- [ ] Recuperación (15): `RESULTADO: OK` en entorno nuevo; test completo con 82.332 predicciones; métricas recalculadas.
- [ ] Entrega (10): informe coherente; README; `entrega.md` completo con la declaración de uso de IA.
