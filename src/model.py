"""Construye la solución completa: preprocesamiento + estimador en un Pipeline.

Todo se ajusta solo con la partición de entrenamiento (Pipeline.fit).
Solo se usan funciones de numpy/sklearn para que el artefacto se cargue
con joblib sin depender del código de este repositorio.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def build_pipeline(cfg: dict, categorical: list[str], numeric: list[str], seed: int) -> Pipeline:
    pre_cfg = cfg.get("preprocessing", {})
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if pre_cfg.get("log1p", False):
        # Los predictores numéricos son >= 0 (ver auditoria.json); log1p reduce colas largas.
        num_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if pre_cfg.get("scale", False):
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=pre_cfg.get("min_frequency", 20),
                                 sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", cat, categorical)])
    params = dict(cfg["model"].get("params", {}))
    est_cls = ESTIMATORS[cfg["model"]["estimator"]]
    if "random_state" in est_cls().get_params():
        params.setdefault("random_state", seed)
    return Pipeline([("preprocess", pre), ("model", est_cls(**params))])
