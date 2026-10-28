#!/usr/bin/env bash
# ZIP de la entrega a partir de un commit (por defecto HEAD) con `git archive`: solo archivos
# versionados en Git, así que no incluye .venv, cachés, data/ (se recupera con DVC), modelos
# locales, mlflow.db/mlruns ni credenciales (.dvc/config.local está en .gitignore).
# Uso: bash scripts/empaquetar.sh [commit]   ->  dist/proyecto-<commit corto>.zip
set -euo pipefail

REF="${1:-HEAD}"
COMMIT=$(git rev-parse "$REF^{commit}")
if [ "$REF" = "HEAD" ] && [ -n "$(git status --porcelain)" ]; then
  echo "Árbol de trabajo con cambios sin commit: el ZIP no los incluiría" >&2
  exit 1
fi

mkdir -p dist
OUT="dist/proyecto-${COMMIT:0:7}.zip"
rm -f "$OUT"
git archive --format=zip --prefix=proyecto/ -o "$OUT" "$COMMIT"

# Comprobaciones: piezas obligatorias presentes y nada que no deba ir en el ZIP.
LIST=$(python3 -c "import sys, zipfile; print('\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))" "$OUT")
for f in README.md entrega.md requirements.txt .dvc/config .dvcignore data/raw.dvc \
         evidencias/auditoria.json evidencias/experimentos.csv evidencias/seleccion.json \
         evidencias/metricas-test.json evidencias/predicciones-test.csv; do
  grep -qx "proyecto/$f" <<< "$LIST" || { echo "Falta en el ZIP: $f" >&2; exit 1; }
done
for d in src/ configs/ evidencias/mlflow/ evidencias/recuperacion/; do
  grep -q "^proyecto/$d" <<< "$LIST" || { echo "Falta en el ZIP: $d" >&2; exit 1; }
done
if grep -E '(^|/)(\.venv[^/]*|data/raw|data/prepared|mlruns|\.env|config\.local)(/|$)' <<< "$LIST"; then
  echo "El ZIP contiene archivos que no deben entregarse (arriba)" >&2
  exit 1
fi
if git grep -nE 'hf_[A-Za-z0-9]{30,}|(password|token)[[:space:]]*=[[:space:]]*[A-Za-z0-9]{20,}' "$COMMIT"; then
  echo "Posible token en archivos versionados (arriba)" >&2
  exit 1
fi

echo "$OUT ($(du -h "$OUT" | cut -f1), $(wc -l <<< "$LIST") entradas) desde el commit $COMMIT"
sha256sum "$OUT"
