"""Interpretación de errores en VALIDACIÓN de las cuatro configuraciones.

Análisis posterior: no modifica modelos, umbrales ni seleccion.json (ya congelado) y no
usa test. Carga cada solución registrada (hash verificado contra experimentos.csv),
predice validación, comprueba que la matriz de confusión coincide con la de MLflow y
desglosa los errores por attack_cat (que NO es entrada del modelo; solo se usa aquí
para explicar los errores) y los falsos positivos por service, state y sttl.

Uso: python -m src.analyze_val_errors
Salidas: evidencias/errores-validacion.csv, evidencias/errores-validacion.json,
         evidencias/predicciones-val.csv
"""
from __future__ import annotations

import json

import pandas as pd

from src.common import ROOT, load_params, sha256_file, write_json
from src.predict import load_solution, predict

MIN_NORMALES = 100  # grupos de tráfico normal más pequeños no se reportan (FPR inestable)


def solution_dir(row: pd.Series):
    """Solución exportada de MLflow si se copió; si no, la carpeta local de entrenamiento."""
    exported = ROOT / "evidencias" / "mlflow" / "runs" / row["run_id"] / "artifacts" / "run" / "solution"
    return exported if (exported / "model.joblib").exists() else ROOT / "models" / row["id"]


def by_attack_cat(df: pd.DataFrame, exp_id: str) -> pd.DataFrame:
    g = df.groupby("attack_cat").agg(filas=("y_pred", "size"), pred_ataque=("y_pred", "sum"),
                                     y_true=("y_true", "first"))
    # Normal: los errores son falsos positivos; resto de categorías: ataques no detectados.
    g["errores"] = g["pred_ataque"].where(g["y_true"] == 0, g["filas"] - g["pred_ataque"])
    g["tasa_error"] = g["errores"] / g["filas"]
    g["tipo_error"] = g["y_true"].map({0: "falso_positivo", 1: "falso_negativo"})
    total = g.groupby("tipo_error")["errores"].transform("sum")
    g["pct_de_errores_del_tipo"] = (g["errores"] / total.where(total > 0)).fillna(0.0)
    g = g.reset_index().assign(id=exp_id).sort_values(["y_true", "filas"], ascending=[True, False])
    return g[["id", "attack_cat", "tipo_error", "filas", "pred_ataque", "errores", "tasa_error",
              "pct_de_errores_del_tipo"]]


def fp_by(df: pd.DataFrame, col: str, top: int = 5) -> list[dict]:
    normal = df[df["y_true"] == 0]
    g = normal.groupby(col).agg(normales=("y_pred", "size"), falsos_positivos=("y_pred", "sum"))
    g = g[g["normales"] >= MIN_NORMALES].sort_values("falsos_positivos", ascending=False).head(top)
    n_fp = max(int(normal["y_pred"].sum()), 1)
    return [{col: str(k), "normales": int(r["normales"]), "falsos_positivos": int(r["falsos_positivos"]),
             "fpr": round(r["falsos_positivos"] / r["normales"], 4),
             "pct_de_fp": round(r["falsos_positivos"] / n_fp, 4)} for k, r in g.iterrows()]


def main() -> None:
    params = load_params()
    exps = pd.read_csv(ROOT / "evidencias" / "experimentos.csv")
    sel = json.loads((ROOT / "evidencias" / "seleccion.json").read_text())
    hf = json.loads((ROOT / "evidencias" / "hf_versions.json").read_text())
    val_path = ROOT / params["data"]["prepared_dir"] / "val.csv"
    val = pd.read_csv(val_path)
    preds = val[["row_id", "attack_cat"]].assign(y_true=val[params["target"]])

    tables, summary = [], {}
    for _, row in exps.iterrows():
        model, cfg = load_solution(solution_dir(row))
        assert cfg["model_sha256"] == row["model_sha256"], f"{row['id']}: modelo distinto al del run"
        p = predict(model, cfg, val)
        df = preds.assign(score_attack=p["score_attack"].values, y_pred=p["y_pred"].values)
        cm = {"tn": int(((df.y_true == 0) & (df.y_pred == 0)).sum()), "fp": int(((df.y_true == 0) & (df.y_pred == 1)).sum()),
              "fn": int(((df.y_true == 1) & (df.y_pred == 0)).sum()), "tp": int(((df.y_true == 1) & (df.y_pred == 1)).sum())}
        expected = {k: int(row[f"val_{k}"]) for k in cm}
        assert cm == expected, f"{row['id']}: matriz {cm} distinta de MLflow {expected}"

        t = by_attack_cat(df, row["id"])
        tables.append(t)
        fn = t[t.tipo_error == "falso_negativo"].sort_values("errores", ascending=False)
        summary[row["id"]] = {
            "run_id": row["run_id"], "model_sha256": row["model_sha256"], "threshold": float(row["threshold"]),
            "matriz_confusion_val": cm, "coincide_con_mlflow": True,
            "falsos_negativos_por_attack_cat": [
                {"attack_cat": r.attack_cat, "ataques": int(r.filas), "no_detectados": int(r.errores),
                 "tasa_no_detectados": round(r.tasa_error, 4), "pct_de_fn": round(r.pct_de_errores_del_tipo, 4)}
                for r in fn.itertuples()],
            "falsos_positivos_por_service": fp_by(df.assign(service=val["service"]), "service"),
            "falsos_positivos_por_state": fp_by(df.assign(state=val["state"]), "state"),
            "falsos_positivos_por_sttl": fp_by(df.assign(sttl=val["sttl"]), "sttl"),
        }
        preds[f"score_{row['id']}"] = df["score_attack"].values
        preds[f"y_pred_{row['id']}"] = df["y_pred"].values

    table = pd.concat(tables, ignore_index=True)
    # v1 -> v2 por categoría: qué gana y qué cede la versión elegida (solo validación).
    v1_id, v2_id = hf["v1"]["experiment_id"], sel["experiment_id"]
    wide = table.pivot(index="attack_cat", columns="id", values="errores")
    filas = table.drop_duplicates("attack_cat").set_index("attack_cat")["filas"]
    tipo = table.drop_duplicates("attack_cat").set_index("attack_cat")["tipo_error"]
    cambio = [{"attack_cat": c, "tipo_error": tipo[c], "filas": int(filas[c]),
               "errores_v1": int(wide.at[c, v1_id]), "errores_v2": int(wide.at[c, v2_id]),
               "diferencia_v2_menos_v1": int(wide.at[c, v2_id] - wide.at[c, v1_id])}
              for c in wide.sort_values(v2_id, ascending=False).index]

    out = ROOT / "evidencias"
    table.to_csv(out / "errores-validacion.csv", index=False, float_format="%.6f")
    preds.to_csv(out / "predicciones-val.csv", index=False, float_format="%.17g")
    write_json({
        "nota": ("Análisis posterior sobre validación; no se usó para elegir modelo ni umbral y no usa test. "
                 "attack_cat no es entrada del modelo: solo explica los errores."),
        "regla": "y_pred = 1 (ataque) si score_attack >= threshold; errores de Normal = falsos positivos, "
                 "errores de categorías de ataque = falsos negativos",
        "dataset_val": {"file": "data/prepared/val.csv", "sha256": sha256_file(val_path), "filas": int(len(val))},
        "predicciones_val_sha256": sha256_file(out / "predicciones-val.csv"),
        "min_normales_por_grupo": MIN_NORMALES,
        "v1_vs_v2": {"v1": v1_id, "v2": v2_id, "por_attack_cat": cambio},
        "experimentos": summary,
    }, out / "errores-validacion.json")

    print(wide.assign(filas=filas)[["filas"] + list(exps["id"])].sort_values("filas", ascending=False).to_string())


if __name__ == "__main__":
    main()
