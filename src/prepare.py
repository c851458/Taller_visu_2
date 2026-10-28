"""Auditoría de los CSV originales y creación de particiones sin fuga.

- Agrupa filas con entradas (variables predictoras) idénticas: group_id.
- Elimina de train/val los grupos con etiquetas contradictorias (política documentada).
- Asigna grupos completos a train o val con un hash determinista (semilla fija),
  así ningún grupo de entradas idénticas queda repartido entre ambas particiones.
- El test se conserva completo (82.332 filas) y solo se audita.

Uso: python -m src.prepare   (etapa `prepare` de dvc.yaml)
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.common import ROOT, load_params, sha256_file, write_json


def read_raw(path) -> pd.DataFrame:
    # utf-8-sig elimina el BOM que trae el CSV de entrenamiento.
    return pd.read_csv(path, encoding="utf-8-sig")


def feature_columns(df: pd.DataFrame, params: dict) -> list[str]:
    return [c for c in df.columns if c not in params["excluded_columns"]]


def group_ids(df: pd.DataFrame, features: list[str]) -> pd.Series:
    joined = df[features].astype(str).agg("|".join, axis=1)
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])


def audit_file(df: pd.DataFrame, path, features: list[str], params: dict) -> dict:
    target = params["target"]
    sizes = df.groupby("group_id").size()
    labels_per_group = df.groupby("group_id")[target].nunique()
    contradictory = labels_per_group[labels_per_group > 1].index
    numeric = [c for c in features if c not in params["categorical_columns"]]
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": int(len(df)),
        "columns": int(df.shape[1] - 1),  # sin contar group_id
        "dtypes": {c: str(t) for c, t in df.drop(columns="group_id").dtypes.items()},
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "class_counts": {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "attack_cat_counts": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
        "id_unique": bool(df["id"].is_unique),
        "negative_values_in_numeric": int((df[numeric] < 0).sum().sum()),
        "duplicates_by_features": {
            "unique_input_groups": int(len(sizes)),
            "groups_with_more_than_one_row": int((sizes > 1).sum()),
            "rows_in_duplicate_groups": int(sizes[sizes > 1].sum()),
            "redundant_rows": int(len(df) - len(sizes)),
        },
        "contradictory_label_groups": {
            "groups": int(len(contradictory)),
            "rows": int(df["group_id"].isin(contradictory).sum()),
        },
    }


def main() -> None:
    params = load_params()
    seed, val_fraction = params["seed"], params["val_fraction"]
    raw_dir = ROOT / params["data"]["raw_dir"]
    out_dir = ROOT / params["data"]["prepared_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    src = params["data"]["sources"]

    train_path, test_path = raw_dir / src["train"]["file"], raw_dir / src["test"]["file"]
    train_raw, test_raw = read_raw(train_path), read_raw(test_path)
    for name, df in (("train", train_raw), ("test", test_raw)):
        if len(df) != src[name]["expected_rows"]:
            raise SystemExit(f"{name}: {len(df)} filas, se esperaban {src[name]['expected_rows']}")

    features = feature_columns(train_raw, params)
    assert features == feature_columns(test_raw, params), "Los esquemas de train y test difieren"

    for prefix, df in (("train", train_raw), ("test", test_raw)):
        df.insert(0, "row_id", prefix + ":" + df["id"].astype(str))
        df["group_id"] = group_ids(df, features)

    audit = {
        "target": params["target"],
        "positive_class": "1 = ataque",
        "features_used": features,
        "features_excluded": params["excluded_columns"],
        "excluded_reason": {
            "label": "es la respuesta",
            "attack_cat": "revela la respuesta (Normal <=> label 0)",
            "id": "identificador de fila; solo se usa para row_id y auditoría",
        },
        "files": {
            "train": audit_file(train_raw.drop(columns="row_id"), train_path, features, params),
            "test": audit_file(test_raw.drop(columns="row_id"), test_path, features, params),
        },
    }

    # Política de limpieza: fuera de train/val los grupos con etiquetas contradictorias;
    # los duplicados exactos se conservan (frecuencia real del tráfico) pero agrupados.
    n_labels = train_raw.groupby("group_id")[params["target"]].transform("nunique")
    clean = train_raw[n_labels == 1].copy()

    # Asignación determinista de grupos: hash(seed, group_id) -> [0, 1).
    u = clean["group_id"].map(
        lambda g: int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()[:8], 16) / 2**32)
    clean["split"] = np.where(u < val_fraction, "val", "train")
    train_df = clean[clean["split"] == "train"].drop(columns="split")
    val_df = clean[clean["split"] == "val"].drop(columns="split")
    test_df = test_raw

    shared = set(train_df["group_id"]) & set(val_df["group_id"])
    assert not shared, "Hay grupos repartidos entre train y val"

    paths = {"train": out_dir / "train.csv", "val": out_dir / "val.csv", "test": out_dir / "test.csv"}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        df.to_csv(paths[name], index=False)
    assignment = pd.concat([
        train_df[["row_id", "group_id"]].assign(split="train"),
        val_df[["row_id", "group_id"]].assign(split="val"),
        train_raw.loc[n_labels > 1, ["row_id", "group_id"]].assign(split="excluded_contradictory"),
    ]).sort_values("row_id", key=lambda s: s.str.split(":").str[1].astype(int))
    assignment.to_csv(out_dir / "splits.csv", index=False)

    test_groups = test_df["group_id"]
    tv_groups = set(train_df["group_id"]) | set(val_df["group_id"])
    audit["cleaning_policy"] = {
        "contradictory_groups": "se eliminan de train/val todas las filas de grupos de entradas "
                                "idénticas con etiquetas distintas (no son aprendibles)",
        "exact_duplicates": "se conservan; cada grupo va completo a una sola partición",
        "test": "no se elimina ninguna fila; solo se audita",
        "rows_removed_from_training_file": int((n_labels > 1).sum()),
    }
    audit["splits"] = {
        "method": "hash determinista sha256(f'{seed}:{group_id}') < val_fraction, por grupo",
        "seed": seed,
        "val_fraction": val_fraction,
        "sizes": {k: int(len(v)) for k, v in (("train", train_df), ("val", val_df), ("test", test_df))},
        "class_counts": {
            k: {str(c): int(n) for c, n in v[params["target"]].value_counts().sort_index().items()}
            for k, v in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "groups_shared_train_val": len(shared),
        "sha256": {k: sha256_file(p) for k, p in paths.items()}
                  | {"splits.csv": sha256_file(out_dir / "splits.csv")},
    }
    audit["cross_partition_overlap"] = {
        "test_rows_with_inputs_seen_in_train": int(test_groups.isin(set(train_df["group_id"])).sum()),
        "test_rows_with_inputs_seen_in_val": int(test_groups.isin(set(val_df["group_id"])).sum()),
        "test_rows_with_inputs_seen_in_train_or_val": int(test_groups.isin(tv_groups).sum()),
        "test_rows_total": int(len(test_df)),
        "test_unique_groups_seen_in_train_or_val": int(len(set(test_groups) & tv_groups)),
        "test_unique_groups": int(test_groups.nunique()),
        "note": "El test no se usó para elegir modelo ni umbral; el solapamiento limita "
                "la interpretación del test como datos totalmente nuevos.",
    }
    audit["cross_partition_overlap"]["test_overlap_fraction"] = round(
        audit["cross_partition_overlap"]["test_rows_with_inputs_seen_in_train_or_val"] / len(test_df), 4)
    write_json(audit, ROOT / "evidencias" / "auditoria.json")
    print("Tamaños:", audit["splits"]["sizes"])
    print("Solapamiento test:", audit["cross_partition_overlap"]["test_overlap_fraction"])


if __name__ == "__main__":
    main()
