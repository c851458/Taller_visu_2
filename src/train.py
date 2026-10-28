"""Entrena una configuración, elige su umbral en validación y la registra en MLflow.

Uso: python -m src.train configs/experiments/e1_logreg_base.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import mlflow
import pandas as pd
import sklearn

from src.common import ROOT, dvc_data_refs, git_commit, load_params, load_yaml, sha256_file, write_json
from src.metrics import apply_threshold, choose_threshold, compute_metrics
from src.model import build_pipeline

SKOPS_TRUSTED = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def load_split(name: str, params: dict) -> pd.DataFrame:
    return pd.read_csv(ROOT / params["data"]["prepared_dir"] / f"{name}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="permite entrenar con cambios sin commit (no recomendado)")
    args = ap.parse_args()

    params = load_params()
    cfg = load_yaml(args.config)
    git = git_commit()
    if git["dirty"] and not args.allow_dirty:
        raise SystemExit("Hay cambios sin commit: haz commit antes de entrenar para que el run sea trazable.")
    data_refs = dvc_data_refs()

    train, val = load_split("train", params), load_split("val", params)
    excluded = set(params["excluded_columns"]) | {"row_id", "group_id"}
    features = [c for c in train.columns if c not in excluded]
    categorical = [c for c in params["categorical_columns"] if c in features]
    numeric = [c for c in features if c not in categorical]
    y_train, y_val = train[params["target"]], val[params["target"]]

    pipe = build_pipeline(cfg, categorical, numeric, params["seed"])
    t0 = time.time()
    pipe.fit(train[features], y_train)
    fit_seconds = time.time() - t0
    val_scores = pipe.predict_proba(val[features])[:, 1]
    threshold = choose_threshold(y_val, val_scores, cfg["threshold_policy"])
    m = compute_metrics(y_val, apply_threshold(val_scores, threshold))
    m05 = compute_metrics(y_val, apply_threshold(val_scores, 0.5))

    out = ROOT / "models" / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    model_sha = sha256_file(out / "model.joblib")
    solution_cfg = {
        "experiment_id": cfg["id"],
        "family": cfg["family"],
        "threshold": threshold,
        "threshold_rule": "y_pred = 1 (ataque) si score_attack >= threshold",
        "threshold_policy": cfg["threshold_policy"],
        "positive_class": 1,
        "features": features,
        "categorical_features": categorical,
        "model_file": "model.joblib",
        "model_sha256": model_sha,
        "preprocessing": cfg.get("preprocessing", {}),
        "estimator": cfg["model"],
        "seed": params["seed"],
        "git_commit": git["commit"],
        "dvc": data_refs,
        "validation_metrics": m,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=cfg["id"], description=cfg["hypothesis"]) as run:
        mlflow.set_tags({
            "experiment_id": cfg["id"], "family": cfg["family"],
            "hypothesis": cfg["hypothesis"].strip(), "change": cfg["change"],
            "git_commit": git["commit"], "git_dirty": str(git["dirty"]),
            "dvc_raw_md5": str(data_refs["raw_md5"]), "dvc_prepared_md5": str(data_refs["prepared_md5"]),
            "model_sha256": model_sha, "config_path": args.config,
        })
        mlflow.log_params({
            "seed": params["seed"], "val_fraction": params["val_fraction"],
            "estimator": cfg["model"]["estimator"],
            **{f"model.{k}": v for k, v in cfg["model"].get("params", {}).items()},
            **{f"pre.{k}": v for k, v in cfg.get("preprocessing", {}).items()},
            "threshold_policy": json.dumps(cfg["threshold_policy"]),
            "threshold": threshold,
            "n_train": len(train), "n_val": len(val), "n_features": len(features),
        })
        scalar = {k: v for k, v in m.items() if isinstance(v, float)}
        mlflow.log_metrics({f"val_{k}": v for k, v in scalar.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in m["confusion_matrix"].items()})
        mlflow.log_metrics({f"val05_{k}": v for k, v in m05.items() if isinstance(v, float)})
        mlflow.log_metric("fit_seconds", fit_seconds)
        solution_cfg["mlflow_run_id"] = run.info.run_id
        write_json(solution_cfg, out / "config.json")
        mlflow.log_artifact(str(out / "config.json"), artifact_path="solution")
        mlflow.log_artifact(str(out / "model.joblib"), artifact_path="solution")
        mlflow.log_artifact(str(ROOT / args.config), artifact_path="solution")
        mlflow.log_dict(m, "validation_metrics.json")
        # Tipos internos de numpy/sklearn (dtype de codificadores y árboles); revisados como seguros.
        mlflow.sklearn.log_model(pipe, name="sklearn_model", skops_trusted_types=SKOPS_TRUSTED,
                                 input_example=val[features].head(5))
        print(f"run_id={run.info.run_id} umbral={threshold:.4f} "
              f"recall={m['recall_attack']:.4f} precision={m['precision_attack']:.4f} "
              f"fpr={m['fpr']:.4f} macro_f1={m['macro_f1']:.4f} ({fit_seconds:.1f}s)")


if __name__ == "__main__":
    main()
