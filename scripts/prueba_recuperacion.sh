#!/usr/bin/env bash
# Prueba de recuperación en carpeta y entorno NUEVOS (sin cachés de la ejecución original).
# Uso: bash scripts/prueba_recuperacion.sh <URL_GITHUB> <COMMIT_COMPLETO>
# Deja el log y las salidas en <carpeta_nueva>/proyecto/evidencias/recuperacion/.
set -euo pipefail

REPO_URL="$1"
COMMIT="$2"
WORK="${RECOVERY_DIR:-$HOME/recuperacion_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$WORK"
LOG="$WORK/recuperacion.log"
exec > >(tee -a "$LOG") 2>&1

echo "== Inicio: $(date -Is) | host: $(hostname) | carpeta: $WORK"
echo "== 1. Código: clonar y fijar la revisión"
git clone "$REPO_URL" "$WORK/proyecto"
cd "$WORK/proyecto"
git checkout "$COMMIT"
git log -1 --format='commit %H%n%s'

echo "== 1b. Entorno nuevo con dependencias fijadas"
python3 --version
python3 -m venv .venv-recuperacion
.venv-recuperacion/bin/pip install -q --upgrade pip
.venv-recuperacion/bin/pip install -q -r requirements.txt
PY=.venv-recuperacion/bin/python

echo "== 2. Datos: dvc pull desde el remote entregado (caché nueva)"
.venv-recuperacion/bin/dvc remote list
.venv-recuperacion/bin/dvc pull
.venv-recuperacion/bin/dvc status
$PY -m src.recovery verify-data

echo "== 3-4. Modelos: descargar v1 y v2 por SHA completo (caché HF nueva) y predecir"
export HF_HOME="$WORK/hf_home"
V1=$($PY -c "import json;print(json.load(open('evidencias/hf_versions.json'))['v1']['revision'])")
V2=$($PY -c "import json;print(json.load(open('evidencias/hf_versions.json'))['v2']['revision'])")
REPO=$($PY -c "import json;print(json.load(open('evidencias/hf_versions.json'))['v1']['repo_id'])")
echo "repo=$REPO v1=$V1 v2=$V2"
OUT="$WORK/salidas"
mkdir -p "$OUT"
$PY src/predict.py --repo-id "$REPO" --revision "$V1" --input evidencias/recuperacion/entrada.csv --output "$OUT/despues_v1.csv"
$PY src/predict.py --repo-id "$REPO" --revision "$V2" --input evidencias/recuperacion/entrada.csv --output "$OUT/despues_v2.csv"

echo "== 5. Comparar con las salidas guardadas"
$PY -m src.recovery compare --after-dir "$OUT"
sha256sum evidencias/recuperacion/entrada.csv evidencias/recuperacion/antes_v*.csv "$OUT"/despues_v*.csv
cp "$LOG" "$OUT/"
echo "== Fin: $(date -Is). Copia $OUT/* a evidencias/recuperacion/ del repositorio original."
