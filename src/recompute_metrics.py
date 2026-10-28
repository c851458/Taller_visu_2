"""Recalcula las métricas de test desde predicciones-test.csv y las compara con metricas-test.json.

Uso: python -m src.recompute_metrics
"""
from __future__ import annotations

import json

import pandas as pd

from src.common import ROOT
from src.metrics import compute_metrics


def main() -> None:
    saved = json.loads((ROOT / "evidencias" / "metricas-test.json").read_text())
    pred = pd.read_csv(ROOT / "evidencias" / "predicciones-test.csv")
    assert len(pred) == saved["rows_evaluated"] and pred["row_id"].is_unique
    rule_ok = bool(((pred["score_attack"] >= saved["threshold"]).astype(int) == pred["y_pred"]).all())
    m = compute_metrics(pred["y_true"], pred["y_pred"])
    diffs = {k: abs(m[k] - saved["metrics"][k]) for k in m if isinstance(m[k], float)}
    same_cm = m["confusion_matrix"] == saved["metrics"]["confusion_matrix"]
    print(f"filas={len(pred)} regla_umbral_ok={rule_ok} matriz_igual={same_cm} "
          f"max_diff_metricas={max(diffs.values()):.2e}")
    if not (rule_ok and same_cm and max(diffs.values()) < 1e-12):
        raise SystemExit("Las métricas recalculadas NO coinciden.")
    print("OK: métricas recalculadas coinciden con metricas-test.json")


if __name__ == "__main__":
    main()
