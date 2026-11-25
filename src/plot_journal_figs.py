"""Figures 2 and 4 of the paper, drawn from the raw results only, into figures/.

fig_log.pdf  the encoder log of a collapsed stock encode of code01 (logs/diag_code01.log,
             lambda 0.0002, 6000 iterations, CPU): PSNR and latent rate per logged row.
fig_rd.pdf   (a) mean rate and PSNR over the 24 Kodak images at 10,000 iterations
             (results/kodak_10k_rd.jsonl), shipped clamp and straight-through clamp;
             (b) the same over the screenshots of results/screenshots_rd.jsonl at 2000 iterations
             on which neither mapping collapsed at any rate (the images of the BD-rate average).

    python src/plot_journal_figs.py
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
OUT = ROOT / "figures"


def jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in (RES / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def mean_curves(rows: list[dict], images: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["image"] in images:
            acc[r["arm"]][r["lmbda"]].append((r["bpp"], r["psnr"]))
    out = {}
    for arm, by in acc.items():
        lams = sorted(by)
        if any(len(by[l]) != len(images) for l in lams):
            raise SystemExit(f"arm {arm}: missing encodes")
        out[arm] = (np.array([np.mean([b for b, _ in by[l]]) for l in lams]),
                    np.array([np.mean([p for _, p in by[l]]) for l in lams]))
    return out


def fig_rd() -> None:
    kod = jsonl("kodak_10k_rd.jsonl")
    scr = jsonl("screenshots_rd.jsonl")
    bad = {r["image"] for r in scr if r["collapsed"]}
    ok = {r["image"] for r in scr} - bad
    panels = ((mean_curves(kod, {r["image"] for r in kod}),
               f"(a) 24 Kodak images, 10,000 iterations"),
              (mean_curves(scr, ok), f"(b) {len(ok)} screenshots, 2000 iterations"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
    for ax, (cur, title) in zip(axes, panels):
        for arm, style, label in (("stock", dict(marker="o", ls="-", color="black", mfc="white", ms=4),
                                   "clamp (shipped)"),
                                  ("ste", dict(marker="s", ls="--", color="0.45", ms=3),
                                   "straight-through clamp")):
            x, y = cur[arm]
            ax.plot(x, y, lw=1.0, label=label, **style)
        ax.set_xlabel("rate (bpp), mean over images")
        ax.set_ylabel("PSNR (dB), mean over images")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_rd.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_rd.pdf; screenshots:", sorted(ok))


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


def fig_log() -> None:
    rows = log_rows()
    flat = next(r["flat_psnr"] for r in jsonl("arena.jsonl") if r["image"] == "code01")
    drawn = [r for r in rows if r[0] != "summary"]
    idx = np.arange(1, len(drawn) + 1)
    ph = np.array([r[0] for r in drawn])
    psnr = np.array([r[1] for r in drawn])
    rate = np.array([r[2] for r in drawn])
    fig, b = plt.subplots(figsize=(5.0, 2.2))
    b.axhline(flat, color="0.4", ls=":", lw=1.0)
    b.plot(idx, psnr, color="black", lw=1.2)
    b.set_ylim(17.5, 21.25)
    b.text(idx[ph == "train"][-1] - 1, flat + 0.06, "PSNR of the constant image",
           ha="right", va="bottom", fontsize=8)
    b.text(idx[ph == "train"][-1] - 1, psnr[-1] - 0.08, "PSNR of the reconstruction",
           ha="right", va="top", fontsize=8)
    b.set_xlabel("row of the encoder log")
    b.set_ylabel("PSNR (dB)")
    b2 = b.twinx()
    w, t, q = ph == "warmup", ph == "train", ph == "quant"
    b2.semilogy(idx[w], rate[w], ls="none", marker="o", ms=2.5, mfc="white", color="0.3")
    b2.semilogy(idx[t], rate[t], color="0.45", lw=1.0, ls="--")
    b2.semilogy(idx[q], rate[q], ls="none", marker="x", ms=3, color="0.3")
    b2.set_ylim(1e-5, 30.0)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    b2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    b2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b2.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1"], fontsize=8)
    for edge in (idx[w][-1] + 0.5, idx[t][-1] + 0.5):
        b.axvline(edge, color="0.6", lw=0.6, ls="-")
    b2.text(idx[w][-1] + 2, 1.2e-3, "latent rate (right axis)", ha="left", va="top", fontsize=8, color="0.3")
    b2.set_ylabel("latent rate (bpp)")
    b.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_log.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_log.pdf")


if __name__ == "__main__":
    fig_rd()
    fig_log()
