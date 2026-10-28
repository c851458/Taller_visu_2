"""Evaluación ÚNICA de la solución congelada (seleccion.json) sobre todo el test.

Descarga la revisión de Hugging Face indicada en seleccion.json, verifica el hash
del modelo y escribe evidencias/predicciones-test.csv y evidencias/metricas-test.json.

Uso: python -m src.evaluate_test
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess

import pandas as pd

from src.common import ROOT, load_params, sha256_file, write_json
from src.metrics import compute_metrics
from src.predict import download_solution, load_solution, predict


def main() -> None:
    sel_path = ROOT / "evidencias" / "seleccion.json"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(sel_path)], cwd=ROOT,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(sel_path)], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise SystemExit("seleccion.json debe estar confirmado en Git (congelado) antes de evaluar test.")
    if (ROOT / "evidencias" / "metricas-test.json").exists():
        raise SystemExit("El test ya fue evaluado; no se repite la evaluación.")

    sel = json.loads(sel_path.read_text())
    params = load_params()
    model_dir = download_solution(sel["hf_repo_id"], sel["hf_revision"])
    model, config = load_solution(model_dir)
    assert config["model_sha256"] == sel["model_sha256"], "El modelo descargado no es el seleccionado"
    assert config["threshold"] == sel["threshold"], "El umbral descargado no es el seleccionado"

    test_path = ROOT / params["data"]["prepared_dir"] / "test.csv"
    test = pd.read_csv(test_path)
    pred = predict(model, config, test)
    pred.insert(1, "y_true", test[params["target"]].astype(int))
    pred = pred[["row_id", "y_true", "score_attack", "y_pred"]]
    out_csv = ROOT / "evidencias" / "predicciones-test.csv"
    pred.to_csv(out_csv, index=False, float_format="%.17g")

    write_json({
        "version": sel["version"],
        "experiment_id": sel["experiment_id"],
        "mlflow_run_id": sel["mlflow_run_id"],
        "hf_repo_id": sel["hf_repo_id"],
        "hf_revision": sel["hf_revision"],
        "model_sha256": config["model_sha256"],
        "dataset_file": "data/prepared/test.csv (= UNSW_NB15_testing-set.csv completo + row_id/group_id)",
        "dataset_sha256": sha256_file(test_path),
        "raw_test_sha256": params["data"]["sources"]["test"]["expected_sha256"],
        "rows_evaluated": int(len(test)),
        "threshold": config["threshold"],
        "threshold_rule": "y_pred = 1 si score_attack >= threshold",
        "positive_class": "1 = ataque",
        "zero_division_policy": "cociente indefinido -> 0.0, listado en undefined_metrics",
        "metrics": compute_metrics(pred["y_true"], pred["y_pred"]),
        "predictions_file": "evidencias/predicciones-test.csv",
        "predictions_sha256": sha256_file(out_csv),
        "evaluated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, ROOT / "evidencias" / "metricas-test.json")
    print(json.dumps(compute_metrics(pred["y_true"], pred["y_pred"]), indent=2))


if __name__ == "__main__":
    main()
