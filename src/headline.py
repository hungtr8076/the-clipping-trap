"""Collapse test and result loading shared by verify_paper.py.

collapsed = reconstruction no better than the constant image of its source,
            i.e. PSNR(reconstruction) - PSNR(constant image) < 5 dB.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DIR = {"natural": "kodak", "screen": "screen", "screen_real": "screen_real"}

_flat: dict = {}
_STATS: dict = {}


def _stats() -> dict:
    """Per-image statistics (mean, std, constant-image PSNR) computed from the original files."""
    if not _STATS:
        _STATS.update(json.loads((ROOT / "results" / "image_stats.json").read_text(encoding="utf-8")))
    return _STATS


def flat_psnr(dataset: str, image: str) -> float:
    k = (dataset, image)
    if k not in _flat:
        p = ROOT / "data" / DIR[dataset] / f"{image}.png"
        if p.exists():
            a = np.asarray(Image.open(p).convert("RGB"), np.float64)
            mse = float(((a - a.mean(axis=(0, 1))) ** 2).mean())
            _flat[k] = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0
        else:  # images not redistributed (Kodak, web screenshots): use the precomputed statistics
            _flat[k] = _stats()[f"{DIR[dataset]}/{image}"]["flat_psnr"]
    return _flat[k]


def collapsed(r) -> bool:
    """Collapse = reconstruction within 5 dB of the constant image."""
    return (r["psnr"] - flat_psnr(r["dataset"], r["image"])) < 5.0


def load() -> list[dict]:
    """The trained-codec sweep (bmshj2018-hp, mbt2018-mean, cheng2020)."""
    f = RES / "sweep_trained.jsonl"
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
