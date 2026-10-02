"""Figure 3: (a) Kodak rate-distortion curves of stock Cool-chic and the fix; (b) the encoder log
of a collapsed stock encode of code01.

(a) averages bpp and PSNR over the 24 Kodak images at each of the four lambdas in
    results/arena.jsonl (arm "base" = stock, arm "st" = straight-through fix).
(b) parses logs/diag_code01.log: PSNR and latent rate in each of its 71 rows (seven warm-up
    candidates, training, then quantization of the decoder), with the PSNR of the constant image.

    python src/plot_rd_and_log.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8,
                            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "figures" / "fig_rd_log"


def rd_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = [json.loads(l) for l in (RES / "arena.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "_error" in r or not str(r.get("image", "")).startswith("kodim"):
            continue
        if r["arm"] in ("base", "st"):
            acc[r["arm"]][r["lmbda"]].append((r["bpp"], r["psnr"]))
    out = {}
    for arm, by in acc.items():
        lams = sorted(by)
        if any(len(by[l]) != 24 for l in lams):
            raise SystemExit(f"arm {arm}: expected 24 Kodak images per lambda")
        out[arm] = (np.array([np.mean([b for b, _ in by[l]]) for l in lams]),
                    np.array([np.mean([p for _, p in by[l]]) for l in lams]))
    return out


def log_rows() -> list[tuple[float, float]]:
    rows = []
    for line in (ROOT / "logs" / "diag_code01.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
                     r"([0-9.]+)\s+(\d+)\s", line)
        if m:
            rows.append((float(m.group(5)), float(m.group(3))))
    if len(rows) != 71:
        raise SystemExit(f"expected 71 logged rows, found {len(rows)}")
    return rows


def main() -> None:
    curves = rd_curves()
    rows = log_rows()
    flat = next(json.loads(l)["flat_psnr"] for l in (RES / "arena.jsonl").read_text().splitlines()
                if l.strip() and json.loads(l)["image"] == "code01")

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 1.7))
    for arm, style, label in (("base", dict(marker="o", ls="-", color="black", mfc="white", ms=4), "stock"),
                              ("st", dict(marker="s", ls="--", color="0.45", ms=3), "with the fix")):
        x, y = curves[arm]
        a.plot(x, y, lw=1.0, label=label, **style)
    a.set_xlabel("rate (bpp), mean over 24 Kodak images")
    a.set_ylabel("PSNR (dB)")
    a.set_title("(a) Kodak, four values of $\\lambda$", fontsize=8)
    a.grid(alpha=0.25, linewidth=0.6)
    a.legend(fontsize=7, frameon=False, loc="lower right")

    idx = np.arange(1, len(rows) + 1)
    psnr = np.array([p for p, _ in rows]); rate = np.array([r for _, r in rows])
    b.plot(idx, psnr, color="black", lw=1.2)
    b.axhline(flat, color="0.4", ls=":", lw=1.0)
    b.set_ylim(17.5, 21.25)
    b.text(len(rows), flat + 0.06, "PSNR of the constant image", ha="right", va="bottom", fontsize=7)
    b.text(len(rows), psnr[-1] - 0.08, "PSNR of the reconstruction", ha="right", va="top", fontsize=7)
    b.set_xlabel("row of the encoder log")
    b.set_ylabel("PSNR (dB)")
    b.set_title("(b) stock encode of code01, $\\lambda = 0.0002$", fontsize=8)
    b2 = b.twinx()
    b2.semilogy(idx, rate, color="0.45", lw=1.0, ls="--")
    b2.set_ylim(1e-5, 30.0)
    b2.text(45, 1.5e-4, "latent rate (right axis)", ha="center", va="top", fontsize=7, color="0.3")
    b2.set_ylabel("latent rate (bpp)")
    b.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
