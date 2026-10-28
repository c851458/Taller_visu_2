"""Métricas con ataque (1) como clase positiva.

Denominadores cero: si un cociente no está definido (p. ej. precision sin
predicciones positivas) se reporta 0.0 y se anota en `undefined_metrics`.
"""
from __future__ import annotations

import numpy as np


def _ratio(num: int, den: int, name: str, undefined: list) -> float:
    if den == 0:
        undefined.append(name)
        return 0.0
    return num / den


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    undefined: list[str] = []
    recall = _ratio(tp, tp + fn, "recall_attack", undefined)
    precision = _ratio(tp, tp + fp, "precision_attack", undefined)
    fpr = _ratio(fp, fp + tn, "fpr", undefined)
    recall_normal = _ratio(tn, tn + fp, "recall_normal", undefined)
    precision_normal = _ratio(tn, tn + fn, "precision_normal", undefined)
    f1_attack = _ratio(2 * precision * recall, precision + recall, "f1_attack", undefined)
    f1_normal = _ratio(2 * precision_normal * recall_normal, precision_normal + recall_normal,
                       "f1_normal", undefined)
    return {
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": fpr,
        "f1_attack": f1_attack,
        "f1_normal": f1_normal,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": (tp + tn) / len(y_true),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(len(y_true)),
        "undefined_metrics": undefined,
    }


def apply_threshold(scores, threshold: float) -> np.ndarray:
    """Regla del umbral: ataque si score_attack >= umbral."""
    return (np.asarray(scores) >= threshold).astype(int)


def choose_threshold(y_true, scores, policy: dict) -> float:
    """Elige el umbral solo con validación según la política del config."""
    kind = policy["type"]
    if kind == "fixed":
        return float(policy["value"])
    candidates = np.unique(np.round(np.asarray(scores), 4))
    candidates = np.unique(np.concatenate([candidates, [0.5]]))
    best_t, best_key = 0.5, None
    for t in candidates:
        m = compute_metrics(y_true, apply_threshold(scores, t))
        if kind == "max_recall_at_fpr":
            if m["fpr"] > policy["max_fpr"]:
                continue
            key = (m["recall_attack"], m["precision_attack"])
        elif kind == "max_macro_f1":
            key = (m["macro_f1"],)
        else:
            raise ValueError(f"Política de umbral desconocida: {kind}")
        if best_key is None or key > best_key:
            best_t, best_key = float(t), key
    return best_t
