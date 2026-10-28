"""Predicción con una solución completa (preprocesamiento + modelo + umbral).

Autocontenido a propósito: este mismo archivo se publica en cada revisión de
Hugging Face. Regla: y_pred = 1 (ataque) si score_attack >= threshold.

Uso:
  python src/predict.py --repo-id USUARIO/REPO --revision <SHA> --input X.csv --output Y.csv
  python src/predict.py --model-dir models/e1_logreg_base --input X.csv --output Y.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_solution(model_dir) -> tuple:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_path = model_dir / config["model_file"]
    actual = sha256_file(model_path)
    if actual != config["model_sha256"]:
        raise SystemExit(f"Hash del modelo no coincide: {actual} != {config['model_sha256']}")
    return joblib.load(model_path), config


def download_solution(repo_id: str, revision: str, cache_dir=None) -> Path:
    from huggingface_hub import snapshot_download
    if len(revision) != 40:
        raise SystemExit("Usa el SHA completo (40 caracteres) de la revisión de Hugging Face.")
    return Path(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir))


def predict(model, config: dict, df: pd.DataFrame) -> pd.DataFrame:
    scores = model.predict_proba(df[config["features"]])[:, 1]
    out = pd.DataFrame({"row_id": df["row_id"]}) if "row_id" in df else pd.DataFrame(index=df.index)
    out["score_attack"] = scores
    out["y_pred"] = (scores >= config["threshold"]).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-id")
    src.add_argument("--model-dir")
    ap.add_argument("--revision", help="SHA completo de la revisión en Hugging Face")
    ap.add_argument("--cache-dir", help="caché de Hugging Face (usa una nueva para la prueba de recuperación)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model_dir = (download_solution(args.repo_id, args.revision, args.cache_dir)
                 if args.repo_id else Path(args.model_dir))
    model, config = load_solution(model_dir)
    df = pd.read_csv(args.input)
    out = predict(model, config, df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.17g")
    print(f"{len(out)} filas -> {args.output} | modelo sha256={config['model_sha256']} "
          f"| umbral={config['threshold']} | ataques={int(np.sum(out['y_pred']))}")


if __name__ == "__main__":
    main()
