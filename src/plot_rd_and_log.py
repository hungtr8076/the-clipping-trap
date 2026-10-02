"""Figure 3: (a) Kodak rate-distortion curves of stock Cool-chic and the fix; (b) the encoder log
of a collapsed stock encode of code01.

(a) averages bpp and PSNR over the 24 Kodak images at each of the four lambdas in
    results/arena.jsonl (arm "base" = stock, arm "st" = straight-through fix).
(b) parses logs/diag_code01.log: PSNR and latent rate of the 71 logged rows by phase -- warm-up
    (five candidates, then the best two trained further; markers only, since each candidate is a
    separate run), training (line), quantization of the decoder (crosses) -- with the PSNR of the
    constant image. The two "Results at the end of the phase" rows repeat a training row and are
    not drawn.

    python src/plot_rd_and_log.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker
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


def log_rows() -> list[tuple[str, float, float]]:
    """(phase, PSNR, latent bpp) for every logged row, phase in warmup/train/summary/quant."""
    rows, phase, after_results = [], "warmup", False
    for line in (ROOT / "logs" / "diag_code01.log").read_text(encoding="utf-8").splitlines():
        if "Training phase" in line:
            phase = "train"
        elif "Results at the end of the phase" in line:
            after_results = True
        elif "quantize_model" in line:
            phase = "quant"
        m = re.match(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
                     r"([0-9.]+)\s+(\d+)\s", line)
        if m:
            rows.append(("summary" if after_results else phase, float(m.group(5)), float(m.group(3))))
            after_results = False
    n = {k: sum(r[0] == k for r in rows) for k in ("warmup", "train", "summary", "quant")}
    if len(rows) != 71 or n != {"warmup": 7, "train": 53, "summary": 2, "quant": 9}:
        raise SystemExit(f"unexpected log structure: {len(rows)} rows, {n}")
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

    drawn = [r for r in rows if r[0] != "summary"]
    idx = np.arange(1, len(drawn) + 1)
    ph = np.array([r[0] for r in drawn])
    psnr = np.array([r[1] for r in drawn]); rate = np.array([r[2] for r in drawn])
    b.axhline(flat, color="0.4", ls=":", lw=1.0)
    b.plot(idx, psnr, color="black", lw=1.2)
    b.set_ylim(17.5, 21.25)
    b.text(idx[ph == "train"][-1] - 1, flat + 0.06, "PSNR of the constant image", ha="right", va="bottom", fontsize=7)
    b.text(idx[ph == "train"][-1] - 1, psnr[-1] - 0.08, "PSNR of the reconstruction", ha="right", va="top", fontsize=7)
    b.set_xlabel("row of the encoder log")
    b.set_ylabel("PSNR (dB)")
    b.set_title("(b) stock encode of code01, $\\lambda = 0.0002$", fontsize=8)
    b2 = b.twinx()
    w, t, q = ph == "warmup", ph == "train", ph == "quant"
    b2.semilogy(idx[w], rate[w], ls="none", marker="o", ms=2.5, mfc="white", color="0.3")
    b2.semilogy(idx[t], rate[t], color="0.45", lw=1.0, ls="--")
    b2.semilogy(idx[q], rate[q], ls="none", marker="x", ms=3, color="0.3")
    b2.set_ylim(1e-5, 30.0)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    b2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    b2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b2.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1"], fontsize=7.5)
    for edge in (idx[w][-1] + 0.5, idx[t][-1] + 0.5):
        b.axvline(edge, color="0.6", lw=0.6, ls="-")
    b2.text(idx[w][-1] + 2, 1.2e-3, "latent rate (right axis)", ha="left", va="top", fontsize=7, color="0.3")
    b2.set_ylabel("latent rate (bpp)")
    b.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
