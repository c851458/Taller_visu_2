"""Diagnóstico POSTERIOR al test: dónde se concentran los falsos positivos de v2.

No modifica modelo, umbral ni selección (seleccion.json ya está congelado y el test
ya fue evaluado). Solo explica el resultado: compara la FPR por servicio, estado,
protocolo y sttl entre validación y test, y la FPR de las filas de test cuyas
entradas ya aparecían en train/val.

Uso: python -m src.analyze_fp_test
"""
from __future__ import annotations

import json

import pandas as pd

from src.common import ROOT, load_params, sha256_file, write_json
from src.predict import load_solution, predict

MIN_NORMALES = 200  # grupos más pequeños no se reportan (FPR inestable)


def fpr_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    normal = df[df["y_true"] == 0]
    g = normal.groupby(col).agg(normales=("y_pred", "size"), falsos_positivos=("y_pred", "sum"))
    g["fpr"] = g["falsos_positivos"] / g["normales"]
    g["pct_de_normales"] = g["normales"] / len(normal)
    g["pct_de_fp"] = g["falsos_positivos"] / max(int(normal["y_pred"].sum()), 1)
    return g


def compare(val: pd.DataFrame, test: pd.DataFrame, col: str) -> list[dict]:
    v, t = fpr_by(val, col), fpr_by(test, col)
    m = t.join(v, lsuffix="_test", rsuffix="_val", how="left")
    m = m[m["normales_test"] >= MIN_NORMALES].sort_values("falsos_positivos_test", ascending=False)
    out = []
    for key, r in m.iterrows():
        out.append({
            col: str(key),
            "normales_test": int(r["normales_test"]), "fp_test": int(r["falsos_positivos_test"]),
            "fpr_test": round(float(r["fpr_test"]), 4),
            "pct_normales_test": round(float(r["pct_de_normales_test"]), 4),
            "pct_fp_test": round(float(r["pct_de_fp_test"]), 4),
            "normales_val": None if pd.isna(r["normales_val"]) else int(r["normales_val"]),
            "fpr_val": None if pd.isna(r["fpr_val"]) else round(float(r["fpr_val"]), 4),
            "pct_normales_val": None if pd.isna(r["pct_de_normales_val"]) else round(float(r["pct_de_normales_val"]), 4),
        })
    return out


def main() -> None:
    params = load_params()
    sel = json.loads((ROOT / "evidencias" / "seleccion.json").read_text())
    prep = ROOT / params["data"]["prepared_dir"]

    model, cfg = load_solution(ROOT / "models" / sel["experiment_id"])
    assert cfg["model_sha256"] == sel["model_sha256"], "El modelo local no es el congelado"
    val = pd.read_csv(prep / "val.csv")
    val = val.merge(predict(model, cfg, val)[["row_id", "y_pred"]], on="row_id")
    val["y_true"] = val[params["target"]]

    test = pd.read_csv(prep / "test.csv")
    preds = pd.read_csv(ROOT / "evidencias" / "predicciones-test.csv")
    test = test.merge(preds[["row_id", "y_true", "y_pred"]], on="row_id")

    seen = set(pd.read_csv(prep / "train.csv", usecols=["group_id"])["group_id"]) | set(val["group_id"])
    test["entrada_vista_en_train_val"] = test["group_id"].isin(seen)

    report = {
        "nota": "Diagnóstico posterior a la evaluación única de test; no se usó para elegir modelo ni umbral.",
        "modelo": {k: sel[k] for k in ("version", "experiment_id", "hf_revision", "model_sha256", "threshold")},
        "predicciones_test_sha256": sha256_file(ROOT / "evidencias" / "predicciones-test.csv"),
        "min_normales_por_grupo": MIN_NORMALES,
        "fpr_global": {"val": round(float(val.loc[val.y_true == 0, "y_pred"].mean()), 4),
                       "test": round(float(test.loc[test.y_true == 0, "y_pred"].mean()), 4)},
        "por_service": compare(val, test, "service"),
        "por_state": compare(val, test, "state"),
        "por_proto": compare(val, test, "proto")[:10],
        "por_sttl": compare(val, test, "sttl")[:10],
        "por_solapamiento": compare(val.assign(entrada_vista_en_train_val=False), test,
                                    "entrada_vista_en_train_val"),
    }
    write_json(report, ROOT / "evidencias" / "analisis-fp-test.json")
    for key in ("por_service", "por_state", "por_sttl", "por_solapamiento"):
        print(f"\n== {key}")
        print(pd.DataFrame(report[key]).head(8).to_string(index=False))
    print("\nFPR global:", report["fpr_global"])


if __name__ == "__main__":
    main()
