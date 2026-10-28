"""Descarga las particiones publicadas de UNSW-NB15 y verifica su SHA-256.

Uso: python -m src.download_data
"""
from __future__ import annotations

import datetime as dt
import urllib.request
from pathlib import Path

from src.common import ROOT, load_params, sha256_file, write_json


def main() -> None:
    params = load_params()
    raw_dir = ROOT / params["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = {}
    for split, src in params["data"]["sources"].items():
        dest = raw_dir / src["file"]
        if not dest.exists():
            print(f"Descargando {src['mirror_url']}")
            urllib.request.urlretrieve(src["mirror_url"], dest)
        digest = sha256_file(dest)
        ok = digest == src["expected_sha256"]
        print(f"{dest.name}: sha256={digest} {'OK' if ok else 'NO COINCIDE'}")
        if not ok:
            raise SystemExit(f"Huella inesperada para {dest.name}; no se continúa.")
        log[split] = {
            "file": src["file"],
            "source_url": src["mirror_url"],
            "official_page": src["official_page"],
            "sha256": digest,
            "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
    # Registro de procedencia; viaja con los datos originales en DVC.
    # No se sobrescribe para no cambiar la versión DVC de los datos originales.
    if not (raw_dir / "procedencia.json").exists():
        write_json(log, raw_dir / "procedencia.json")


if __name__ == "__main__":
    main()
