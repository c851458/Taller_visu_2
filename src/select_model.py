"""Selección de v2 usando EXCLUSIVAMENTE validación y congelación antes del test.

Regla (configs/seleccion.yaml): entre los experimentos con FPR de validación <= max_fpr,
el de mayor macro-F1 de validación. Escribe evidencias/seleccion.json; ese archivo
se confirma con un commit ANTES de ejecutar src.evaluate_test.

Uso: python -m src.select_model
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from src.common import ROOT, git_commit, load_yaml, write_json


def main() -> None:
    rule = load_yaml("configs/seleccion.yaml")
    table = pd.read_csv(ROOT / "evidencias" / "experimentos.csv")
    eligible = table[table["val_fpr"] <= rule["max_fpr"]]
    if eligible.empty:
        raise SystemExit("Ningún experimento cumple la restricción de FPR.")
    best = eligible.sort_values(["val_macro_f1", "val_recall_attack"], ascending=False).iloc[0]
    print(f"Elegido por validación: {best['id']} (macro-F1={best['val_macro_f1']:.4f}, "
          f"FPR={best['val_fpr']:.4f}, umbral={best['threshold']})")

    versions_file = ROOT / "evidencias" / "hf_versions.json"
    versions = json.loads(versions_file.read_text()) if versions_file.exists() else {}
    v2 = versions.get("v2")
    if v2 is None or v2["experiment_id"] != best["id"]:
        raise SystemExit(f"Publica primero v2 con: python -m src.publish_hf --version v2 "
                         f"--exp {best['id']} --repo-id <USUARIO/REPO>")
    assert v2["model_sha256"] == best["model_sha256"], "El modelo publicado no es el del run elegido"

    git = git_commit()
    write_json({
        "version": "v2",
        "experiment_id": best["id"],
        "mlflow_run_id": best["run_id"],
        "git_commit_model": best["git_commit"],
        "git_commit_at_selection": git["commit"],
        "dvc_prepared_md5": best["dvc_prepared_md5"],
        "dvc_raw_md5": v2["dvc"]["raw_md5"],
        "hf_repo_id": v2["repo_id"],
        "hf_revision": v2["revision"],
        "model_sha256": best["model_sha256"],
        "threshold": float(best["threshold"]),
        "selection_rule": rule,
        "reason": rule["reason"].strip(),
        "validation_metrics": {k: float(best[k]) for k in
                               ("val_recall_attack", "val_precision_attack", "val_fpr", "val_macro_f1")},
        "compared_against": "v1 = e1_logreg_base (solo validación)",
        "test_used_for_selection": False,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "freeze_evidence": "commit de Git que añade este archivo, anterior al commit de metricas-test.json",
    }, ROOT / "evidencias" / "seleccion.json")
    print("Escrito evidencias/seleccion.json -> haz commit AHORA, antes de evaluar test.")


if __name__ == "__main__":
    main()
