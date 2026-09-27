#!/usr/bin/env python3
"""Detect a collapsed encode by comparing a reconstruction with the constant image.

    python src/detect_collapse.py SOURCE.png RECONSTRUCTION.png [--threshold 5]

margin = PSNR(x, x_hat) - PSNR(x, x_bar), where x_bar is the constant image whose per-channel
value is the mean of the source. A reconstruction whose margin is below the threshold (5 dB by
default) is no better than a flat color field. In the paper, collapsed and working encodes are
separated by an empty band from +0.79 dB to +10.82 dB, so any threshold in that band gives the
same labels. An absolute PSNR/SSIM threshold cannot do this: on bright, flat content the
constant image already scores high.

Exit status: 0 = working encode, 1 = collapsed, 2 = bad input.
"""
import argparse
import sys

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("reconstruction")
    ap.add_argument("--threshold", type=float, default=5.0, help="margin in dB below which an encode is collapsed")
    a = ap.parse_args()
    x = np.asarray(Image.open(a.source).convert("RGB"), np.float64)
    y = np.asarray(Image.open(a.reconstruction).convert("RGB"), np.float64)
    if x.shape != y.shape:
        print(f"shape mismatch: {x.shape} vs {y.shape}", file=sys.stderr)
        return 2
    flat = np.broadcast_to(x.mean(axis=(0, 1)), x.shape)
    p_rec, p_flat = psnr(x, y), psnr(x, flat)
    margin = p_rec - p_flat
    collapsed = margin < a.threshold
    print(f"PSNR(reconstruction) = {p_rec:.2f} dB")
    print(f"PSNR(constant image) = {p_flat:.2f} dB")
    print(f"margin               = {margin:+.2f} dB")
    print(f"reconstruction std   = {y.std():.2f}")
    print("COLLAPSED: no better than a flat color field" if collapsed else "ok")
    return 1 if collapsed else 0


if __name__ == "__main__":
    sys.exit(main())
