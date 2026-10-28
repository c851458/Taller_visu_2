"""Prueba de recuperación: muestra fija de validación, salidas "antes" y comparación.

  python -m src.recovery sample    # entrada.csv (30 filas fijas de val) + salidas antes_v1/v2 (modelos locales)
  python -m src.recovery verify-data  # comprueba huellas de datos recuperados por DVC
  python -m src.recovery compare --after-dir DIR   # compara antes vs después (descargados por SHA)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import ROOT, load_params, sha256_file, write_json
from src.predict import load_solution, predict

REC = ROOT / "evidencias" / "recuperacion"
N_ROWS = 30
TOLERANCE = 1e-9  # diferencia absoluta máxima aceptada en score_attack


def sample() -> None:
    params = load_params()
    val = pd.read_csv(ROOT / params["data"]["prepared_dir"] / "val.csv")
    # Estratificada para tener normales y ataques; semilla fija.
    rows = val.groupby(params["target"], group_keys=False).sample(n=N_ROWS // 2, random_state=params["seed"])
    rows = rows.sort_values("row_id")
    REC.mkdir(parents=True, exist_ok=True)
    rows.to_csv(REC / "entrada.csv", index=False)
    versions = json.loads((ROOT / "evidencias" / "hf_versions.json").read_text())
    for v, info in versions.items():
        model, cfg = load_solution(ROOT / "models" / info["experiment_id"])
        predict(model, cfg, rows).to_csv(REC / f"antes_{v}.csv", index=False, float_format="%.17g")
    print(f"{len(rows)} filas en {REC / 'entrada.csv'}; salidas antes_*.csv escritas")


def verify_data() -> None:
    params = load_params()
    audit = json.loads((ROOT / "evidencias" / "auditoria.json").read_text())
    checks = {}
    for split, src in params["data"]["sources"].items():
        p = ROOT / params["data"]["raw_dir"] / src["file"]
        checks[f"raw/{src['file']}"] = sha256_file(p) == src["expected_sha256"]
    for name, expected in audit["splits"]["sha256"].items():
        fname = name if name.endswith(".csv") else f"{name}.csv"
        checks[f"prepared/{fname}"] = sha256_file(ROOT / params["data"]["prepared_dir"] / fname) == expected
    entrada = pd.read_csv(REC / "entrada.csv")
    val = pd.read_csv(ROOT / params["data"]["prepared_dir"] / "val.csv")
    merged = entrada.merge(val, on="row_id", suffixes=("", "_val"))
    checks["entrada_rows_in_recovered_val"] = len(merged) == len(entrada)
    for k, ok in checks.items():
        print(f"{'OK ' if ok else 'FALLA'} {k}")
    if not all(checks.values()):
        raise SystemExit("Huellas de datos NO coinciden.")


def compare(after_dir: Path) -> None:
    versions = json.loads((ROOT / "evidencias" / "hf_versions.json").read_text())
    result = {"tolerance_abs_score": TOLERANCE, "entrada_sha256": sha256_file(REC / "entrada.csv"),
              "versions": {}}
    for v, info in versions.items():
        before = pd.read_csv(REC / f"antes_{v}.csv")
        after = pd.read_csv(after_dir / f"despues_{v}.csv")
        same_rows = before["row_id"].tolist() == after["row_id"].tolist()
        max_diff = float((before["score_attack"] - after["score_attack"]).abs().max())
        same_labels = bool((before["y_pred"] == after["y_pred"]).all())
        result["versions"][v] = {
            "hf_revision": info["revision"], "model_sha256": info["model_sha256"],
            "threshold": info["threshold"], "rows": len(after), "same_row_ids": same_rows,
            "same_y_pred": same_labels, "max_abs_diff_score": max_diff,
            "antes_sha256": sha256_file(REC / f"antes_{v}.csv"),
            "despues_sha256": sha256_file(after_dir / f"despues_{v}.csv"),
            "passed": same_rows and same_labels and max_diff <= TOLERANCE,
        }
        print(f"{v}: filas={len(after)} y_pred_iguales={same_labels} max_diff={max_diff:.3e}")
    result["passed"] = all(x["passed"] for x in result["versions"].values())
    write_json(result, after_dir / "comparacion.json")
    print("RESULTADO:", "OK" if result["passed"] else "FALLA")
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "verify-data", "compare"])
    ap.add_argument("--after-dir", type=Path, default=REC)
    args = ap.parse_args()
    {"sample": sample, "verify-data": verify_data}.get(args.cmd, lambda: compare(args.after_dir))()


if __name__ == "__main__":
    main()
