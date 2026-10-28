"""Exporta los runs de MLflow a evidencias/mlflow/ y construye evidencias/experimentos.csv.

Para cada configuración se toma su run FINISHED más reciente. La exportación
permite cruzar la tabla con los registros sin depender del servidor local.

Uso: python -m src.export_mlflow
"""
from __future__ import annotations

import shutil

import pandas as pd
from mlflow.tracking import MlflowClient

from src.common import ROOT, load_params, load_yaml, write_json


def main() -> None:
    params = load_params()
    client = MlflowClient(tracking_uri=params["mlflow"]["tracking_uri"])
    exp = client.get_experiment_by_name(params["mlflow"]["experiment_name"])
    runs = client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"])
    conclusions = load_yaml("configs/conclusiones.yaml")
    out = ROOT / "evidencias" / "mlflow"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    all_runs, rows, seen = [], [], set()
    for r in runs:
        record = {
            "run_id": r.info.run_id, "run_name": r.info.run_name,
            "experiment_id": exp.experiment_id, "experiment_name": exp.name,
            "status": r.info.status, "start_time_ms": r.info.start_time, "end_time_ms": r.info.end_time,
            "params": r.data.params, "metrics": r.data.metrics,
            "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.log-model")},
        }
        all_runs.append(record)
        cfg_id = r.data.tags.get("experiment_id")
        if cfg_id is None or cfg_id in seen:
            continue
        seen.add(cfg_id)
        run_dir = out / "runs" / r.info.run_id
        write_json(record, run_dir / "run.json")
        for art in ("solution/config.json", "validation_metrics.json"):
            client.download_artifacts(r.info.run_id, art, str(run_dir))
        cfg = load_yaml(r.data.tags["config_path"])
        m, p = r.data.metrics, r.data.params
        rows.append({
            "id": cfg_id,
            "hypothesis": cfg["hypothesis"].strip(),
            "change": cfg["change"],
            "family": cfg["family"],
            "estimator": p["estimator"],
            "params_path": r.data.tags["config_path"],
            "seed": p["seed"],
            "git_commit": r.data.tags["git_commit"],
            "dvc_prepared_md5": r.data.tags["dvc_prepared_md5"],
            "run_id": r.info.run_id,
            "threshold": float(p["threshold"]),
            "threshold_policy": p["threshold_policy"],
            "val_recall_attack": m["val_recall_attack"],
            "val_precision_attack": m["val_precision_attack"],
            "val_fpr": m["val_fpr"],
            "val_macro_f1": m["val_macro_f1"],
            "val_accuracy": m["val_accuracy"],
            "val_tn": int(m["val_tn"]), "val_fp": int(m["val_fp"]),
            "val_fn": int(m["val_fn"]), "val_tp": int(m["val_tp"]),
            "model_sha256": r.data.tags["model_sha256"],
            "conclusion": conclusions.get(cfg_id, "").strip(),
        })
    write_json(all_runs, out / "all_runs.json")
    table = pd.DataFrame(rows).sort_values("id")
    table.to_csv(ROOT / "evidencias" / "experimentos.csv", index=False)
    (out / "COMO_CONSULTAR.md").write_text(
        "# Exportación de MLflow\n\n"
        "- `all_runs.json`: todos los runs FINISHED (parámetros, métricas, tags).\n"
        "- `runs/<run_id>/`: run usado en `experimentos.csv`, con `config.json` de la "
        "solución (umbral, hash del modelo, commit, DVC) y métricas de validación.\n"
        "- Cruce: la columna `run_id` de `evidencias/experimentos.csv` es el nombre de la carpeta.\n\n"
        "Para abrir la base original: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.\n",
        encoding="utf-8")
    print(table[["id", "run_id", "threshold", "val_recall_attack", "val_precision_attack",
                 "val_fpr", "val_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
