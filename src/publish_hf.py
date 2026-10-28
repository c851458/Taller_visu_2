"""Publica una solución (v1 o v2) en un repositorio de modelos de Hugging Face Hub.

Cada revisión contiene: model.joblib, config.json (umbral, features, hashes,
referencias Git/DVC/MLflow), requirements.txt, predict.py y README.md (model card).
Guarda el SHA completo de la revisión en evidencias/hf_versions.json y crea un tag.

Uso:
  python -m src.publish_hf --exp e1_logreg_base --version v1 --repo-id USUARIO/defiende-servidor-unsw
  python -m src.publish_hf --version v2 --repo-id USUARIO/... --card-only   # solo actualiza la card
"""
from __future__ import annotations

import argparse
import json
import shutil

from huggingface_hub import HfApi

from src.common import ROOT, git_commit, load_params, sha256_file, write_json

VERSIONS_FILE = ROOT / "evidencias" / "hf_versions.json"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def model_card(version: str, cfg: dict, repo_id: str, test: dict | None) -> str:
    m, cm = cfg["validation_metrics"], cfg["validation_metrics"]["confusion_matrix"]
    audit = json.loads((ROOT / "evidencias" / "auditoria.json").read_text(encoding="utf-8"))
    sizes = audit["splits"]["sizes"]
    test_section = "Aún no evaluado en test (solo la versión final se evalúa una vez en test).\n"
    if test:
        tm, tcm = test["metrics"], test["metrics"]["confusion_matrix"]
        test_section = (
            f"Evaluación única sobre las {test['rows_evaluated']} filas del test publicado, con la "
            f"revisión `{test['hf_revision']}` y modelo sha256 `{test['model_sha256']}` "
            f"(no modificado al actualizar esta card).\n\n"
            f"| recall ataque | precision ataque | FPR | macro-F1 | accuracy |\n|---|---|---|---|---|\n"
            f"| {fmt(tm['recall_attack'])} | {fmt(tm['precision_attack'])} | {fmt(tm['fpr'])} | "
            f"{fmt(tm['macro_f1'])} | {fmt(tm['accuracy'])} |\n\n"
            f"Matriz de confusión: TN={tcm['tn']}, FP={tcm['fp']}, FN={tcm['fn']}, TP={tcm['tp']}.\n")
    return f"""---
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlops-course]
license: other
---

# Defiende el servidor — {version} ({cfg['experiment_id']})

## Propósito
Clasificación binaria de conexiones de red: **0 = normal, 1 = ataque (clase positiva)**.
Proyecto académico de MLOps (versionado de código, datos, experimentos y modelos). No es un
sistema de detección listo para producción.

## Solución
- Familia: {cfg['family']} · estimador: `{cfg['estimator']['estimator']}` · parámetros: `{json.dumps(cfg['estimator'].get('params', {}))}`
- Preprocesamiento (ajustado solo con entrenamiento): `{json.dumps(cfg['preprocessing'])}` + one-hot de {', '.join(cfg['categorical_features'])}
- **Umbral: {cfg['threshold']}** — regla `y_pred = 1 si score_attack >= umbral`; política `{json.dumps(cfg['threshold_policy'])}` aplicada solo sobre validación.
- Archivo `model.joblib` (Pipeline de scikit-learn completo), sha256 `{cfg['model_sha256']}`

## Datos
UNSW-NB15, particiones publicadas (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`).
Entrenamiento/validación salen solo del archivo de entrenamiento, por grupos de entradas idénticas
(semilla {cfg['seed']}): train={sizes['train']}, val={sizes['val']}. Se excluyen `id`, `attack_cat`, `label`.
Se eliminaron {audit['cleaning_policy']['rows_removed_from_training_file']} filas con entradas idénticas y etiquetas contradictorias.

## Métricas de validación ({m['n']} filas)
| recall ataque | precision ataque | FPR | macro-F1 | accuracy |
|---|---|---|---|---|
| {fmt(m['recall_attack'])} | {fmt(m['precision_attack'])} | {fmt(m['fpr'])} | {fmt(m['macro_f1'])} | {fmt(m['accuracy'])} |

Matriz de confusión: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.
Denominadores cero se reportan como 0.0 y se listan en `undefined_metrics`.

## Resultado en test
{test_section}
## Trazabilidad
- Código (Git commit): `{cfg['git_commit']}`
- Datos DVC: originales md5 `{cfg['dvc']['raw_md5']}` · preparados md5 `{cfg['dvc']['prepared_md5']}`
- Run de MLflow: `{cfg['mlflow_run_id']}`
- Python {cfg['python']} · scikit-learn {cfg['sklearn']} (ver `requirements.txt`)

## Uso
```bash
pip install -r requirements.txt
python predict.py --repo-id {repo_id} --revision <SHA_COMPLETO> --input entrada.csv --output salida.csv
```
```python
from huggingface_hub import snapshot_download
import joblib, json
d = snapshot_download("{repo_id}", revision="<SHA_COMPLETO>")
cfg = json.load(open(f"{{d}}/config.json")); model = joblib.load(f"{{d}}/model.joblib")
score = model.predict_proba(df[cfg["features"]])[:, 1]; y_pred = (score >= cfg["threshold"]).astype(int)
```

## Limitaciones
- Dataset histórico (2015, tráfico sintético de laboratorio): no demuestra eficacia en una red actual.
- Variables como `sttl`/`ct_state_ttl` son muy predictivas por artefactos de la captura.
- ~10 % de las filas de test tienen entradas idénticas a train/val; el test no es totalmente "nuevo".
- `joblib` usa pickle: carga solo archivos de fuentes confiables y con la misma versión de scikit-learn.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1", "v2"])
    ap.add_argument("--exp", help="id del experimento (carpeta en models/)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--card-only", action="store_true",
                    help="actualiza solo README.md (p. ej. con el resultado de test)")
    args = ap.parse_args()

    versions = json.loads(VERSIONS_FILE.read_text()) if VERSIONS_FILE.exists() else {}
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    stage = ROOT / "hf_release" / args.version
    params = load_params()

    if args.card_only:
        info = versions[args.version]
        cfg = json.loads((ROOT / "models" / info["experiment_id"] / "config.json").read_text())
        test_file = ROOT / "evidencias" / "metricas-test.json"
        test = json.loads(test_file.read_text()) if test_file.exists() else None
        (stage / "README.md").write_text(model_card(args.version, cfg, args.repo_id, test), encoding="utf-8")
        commit = api.upload_file(path_or_fileobj=str(stage / "README.md"), path_in_repo="README.md",
                                 repo_id=args.repo_id,
                                 commit_message=f"{args.version}: model card con resultado de test")
        info["card_update_revision"] = commit.oid
        write_json(versions, VERSIONS_FILE)
        print(f"Card actualizada en {commit.oid}; el modelo evaluado sigue en {info['revision']}")
        return

    git = git_commit()
    if git["dirty"]:
        raise SystemExit("Haz commit antes de publicar para que la versión quede ligada a un commit.")
    src_dir = ROOT / "models" / args.exp
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for f in ("model.joblib", "config.json"):
        shutil.copy2(src_dir / f, stage / f)
    cfg = json.loads((stage / "config.json").read_text())
    assert sha256_file(stage / "model.joblib") == cfg["model_sha256"]
    shutil.copy2(ROOT / "requirements.txt", stage / "requirements.txt")
    shutil.copy2(ROOT / "src" / "predict.py", stage / "predict.py")
    (stage / "README.md").write_text(model_card(args.version, cfg, args.repo_id, None), encoding="utf-8")

    commit = api.upload_folder(folder_path=str(stage), repo_id=args.repo_id, delete_patterns="*",
                               commit_message=f"{args.version}: {cfg['experiment_id']} "
                                              f"(run {cfg['mlflow_run_id']}, git {cfg['git_commit'][:7]})")
    api.create_tag(args.repo_id, tag=args.version, revision=commit.oid, exist_ok=False)
    versions[args.version] = {
        "repo_id": args.repo_id,
        "revision": commit.oid,
        "tag": args.version,
        "experiment_id": cfg["experiment_id"],
        "mlflow_run_id": cfg["mlflow_run_id"],
        "git_commit_model": cfg["git_commit"],
        "git_commit_publish": git["commit"],
        "dvc": cfg["dvc"],
        "model_sha256": cfg["model_sha256"],
        "threshold": cfg["threshold"],
        "mlflow_tracking_uri": params["mlflow"]["tracking_uri"],
    }
    write_json(versions, VERSIONS_FILE)
    print(f"{args.version} publicado: https://huggingface.co/{args.repo_id}/tree/{commit.oid}")


if __name__ == "__main__":
    main()
