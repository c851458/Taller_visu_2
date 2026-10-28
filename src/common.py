"""Utilidades compartidas: configuración, hashes, Git y referencias DVC."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_params() -> dict:
    return load_yaml("configs/params.yaml")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def git_commit() -> dict:
    """Commit actual y si el árbol de trabajo tiene cambios sin confirmar."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty = "sin-git", True
    return {"commit": sha, "dirty": dirty}


def dvc_data_refs() -> dict:
    """Hashes DVC de los datos originales (data/raw.dvc) y preparados (dvc.lock)."""
    refs = {"raw_md5": None, "prepared_md5": None}
    raw_dvc = ROOT / "data" / "raw.dvc"
    if raw_dvc.exists():
        refs["raw_md5"] = yaml.safe_load(raw_dvc.read_text())["outs"][0]["md5"]
    lock = ROOT / "dvc.lock"
    if lock.exists():
        outs = yaml.safe_load(lock.read_text())["stages"]["prepare"]["outs"]
        for out in outs:
            if out["path"] == "data/prepared":
                refs["prepared_md5"] = out["md5"]
    return refs
