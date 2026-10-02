"""Figure 2: the trigger region, in mean-brightness x contrast space.

The figure shows two things:

  * the failure needs BOTH high mean intensity AND low contrast - full0138 and full0502 are
    two DIV2K photographs whose mean differs by 1.8 levels but whose standard deviation
    differs by a factor of 2.4, and only the low-contrast one collapses;
  * the standard benchmarks do NOT clear the region. The vertical marks are the
    brightest full image of each benchmark: 170.5 over Kodak's 24, and 225.5 over DIV2K's
    800. That DIV2K image is full0138, and it collapses - 11 of 40 encodes at Kodak
    resolution, 5 of 8 at its native 2040x1356. The brightest member of a standard benchmark
    is inside the failure region, not outside it.
    Nothing is claimed about Kodak: at mean 170.5 no encode has been run, so its status is
    untested rather than safe.

Selection rules:
  * STOCK encodes only. Every patched arm (straight-through, mean/meanstd init, clip
    disabled) is excluded - a patched run that does not collapse says nothing about the
    trigger.
  * Cool-chic only. The C3 runs use a different codec, crop size and iteration budget.
  * One rate point only, lambda = 0.0006. Collapse probability depends on lambda, so
    pooling rates would put a second variable on a two-variable plot.
  * Collapse uses the same rule as verify_paper.py, margin < 5 dB, recomputed for every row
    rather than read from the stored `collapsed` flag (one probe_ramp row has a stale flag).

Nothing is encoded by color, so the figure survives black-and-white printing: the marker
SHAPE gives the content type and the marker FILL gives the measured collapse rate.

    python src/plot_trigger_map.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8.5,
                            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import headline                                  # same directory

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LMBDA = 0.0006
THRESHOLD = 5.0
OUT_PNG = ROOT / "figures" / "fig_trigger_map.png"
OUT_PDF = ROOT / "figures" / "fig_trigger_map.pdf"
# Image statistics are precomputed from the original files (Kodak, DIV2K and the web screenshots
# are not redistributed here); see results/image_stats.json.
STATS = json.loads((RES / "image_stats.json").read_text(encoding="utf-8"))

# file -> predicate selecting the stock Cool-chic rows in it
SOURCES = {
    "natural_seeds.jsonl":     lambda r: True,
    "arena.jsonl":             lambda r: r.get("arm") == "base",
    "metrics_coolchic.jsonl":  lambda r: r.get("patched") is False,
    "metrics_full0138.jsonl":  lambda r: r.get("patched") is False,
    "expand_real.jsonl":       lambda r: r.get("codec", "coolchic") == "coolchic",
    "repeatability.jsonl":     lambda r: True,
    "validate_predictor.jsonl": lambda r: True,
    "probe_brightness.jsonl":  lambda r: True,
    "probe_ramp.jsonl":        lambda r: True,
}
# excluded on purpose: c3_real.jsonl (different codec), verify_fix.jsonl (patched only)

STAT_GROUPS = ("kodak", "screen", "screen_real", "natural_bright")   # key prefixes in image_stats.json

# content type -> (marker, label).  Shape carries the category, never color.
STYLE = {
    "natural": ("o", "natural photograph"),
    "screen":  ("^", "screen content, synthetic"),
    "screen_real": ("s", "screen content, real screenshot"),
    "derived": ("D", "brightness-shifted variant"),
}


def image_stats() -> dict[str, tuple[float, float, str]]:
    """name -> (mean, std, content type), from results/image_stats.json."""
    out = {}
    for key, st in STATS.items():
        d, stem = key.split("/")
        if d not in STAT_GROUPS:
            continue
        if d == "kodak":
            kind = "natural"
        elif d == "screen_real":
            kind = "screen_real"
        elif d == "screen":
            kind = "screen"
        else:                                       # natural_bright: photos and ramps
            kind = "derived" if stem.startswith("ramp") else "natural"
        out[stem] = (st["mean"], st["std"], kind)
    return out


def collect(stats):
    """(mean, std, kind, label) -> [collapse flags], over stock Cool-chic encodes only."""
    agg = collections.defaultdict(list)
    kept = dropped = 0
    for fn, ok in SOURCES.items():
        p = RES / fn
        if not p.exists():
            raise SystemExit(f"missing {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not ok(r):
                dropped += 1
                continue
            if "lmbda" in r and abs(r["lmbda"] - LMBDA) > 1e-12:
                dropped += 1
                continue
            name = r.get("image")
            if name in stats:
                mean, std, kind = stats[name]
            elif "mean" in r and "std" in r:          # ramp / inverted probes, no file on disk
                mean, std, kind = r["mean"], r["std"], "derived"
                name = name or f"ramp{r.get('target', '')}"
            else:
                dropped += 1
                continue
            margin = r["margin"] if "margin" in r \
                else r["psnr"] - headline.flat_psnr(r["dataset"], r["image"])
            agg[(round(mean, 2), round(std, 2), kind, name)].append(margin < THRESHOLD)
            kept += 1
    print(f"kept {kept} stock encodes at lambda={LMBDA}; dropped {dropped}")
    return agg


def main() -> None:
    stats = image_stats()
    agg = collect(stats)

    fig, ax = plt.subplots(figsize=(5.05, 2.24))  # drawn at print size (4.8 in wide after the tight bbox)

    # Background: the natural-image cloud, so the reader can see where photographs live.
    bg = []
    for key, st in STATS.items():
        if key.split("/")[0] in ("kodak", "div2k_crops"):
            bg.append((st["mean"], st["std"]))   # RGB, the same statistic as the markers
    bg = np.array(bg)
    ax.scatter(bg[:, 0], bg[:, 1], s=4, c="0.78", marker=".", linewidths=0, zorder=1,
               label=f"natural images, not encoded (n={len(bg)})")
    # Brightest FULL image of each benchmark. The DIV2K mark lands on full0138, which collapses.
    # Labels sit above the axes so that no marker can cover them.
    for x, txt, ha in ((170.5, "Kodak max", "right"), (225.5, "DIV2K max", "left")):
        ax.axvline(x, color="0.35", linestyle=":", linewidth=1.1, zorder=2)
        ax.text(x, 1.01, txt, transform=ax.get_xaxis_transform(), va="bottom", ha=ha,
                fontsize=7, color="0.25")
    seen_kinds = set()
    for (mean, std, kind, name), flags in sorted(agg.items()):
        n, k = len(flags), sum(flags)
        rate = k / n
        marker, _ = STYLE[kind]
        # fill encodes the collapse rate: open = never, gray = sometimes, black = always
        face = "white" if rate == 0 else ("black" if rate == 1 else "0.55")
        ax.scatter(mean, std, marker=marker, s=0.5 * (34 + 9 * min(n, 12)), facecolors=face,
                   edgecolors="black", linewidths=0.7, zorder=3)
        seen_kinds.add(kind)

    # The two photographs that separate brightness from contrast.
    for name, xy, ha in (("full0138", (-75, -12), "right"), ("full0502", (-14, 18), "right"),
                         ("div127", (-12, -3), "right")):
        for (mean, std, kind, nm), flags in agg.items():
            if nm == name:
                ax.annotate(f"{name}  {sum(flags)}/{len(flags)}", (mean, std),
                            textcoords="offset points", xytext=xy, ha=ha, fontsize=7.5,
                            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                            arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3"))
    ax.set_xlabel("mean intensity of the source image (0-255)")
    ax.set_ylabel("standard deviation (contrast)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    shape_legend = [Line2D([], [], marker=STYLE[k][0], linestyle="none", markerfacecolor="white",
                           markeredgecolor="black", markersize=5, label=STYLE[k][1])
                    for k in ("natural", "screen", "screen_real", "derived") if k in seen_kinds]
    fill_legend = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=f,
                          markeredgecolor="black", markersize=5, label=l)
                   for f, l in (("white", "never collapsed"), ("0.55", "collapsed sometimes"),
                                ("black", "collapsed every encode"))]
    ax.set_xlim(-8, 268); ax.set_ylim(-4, 92)
    # Both legends sit outside the axes, to the right, so that they cover no data.
    first = ax.legend(handles=shape_legend, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      fontsize=7, frameon=False, borderpad=0.2, labelspacing=0.4)
    ax.add_artist(first)
    ax.legend(handles=fill_legend, loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=7,
              frameon=False, borderpad=0.2, labelspacing=0.4)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(exist_ok=True)
    # bbox_extra_artists: the first legend is added with add_artist, so the tight bounding box
    # would ignore it and clip its labels.
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.02, bbox_extra_artists=[first, *ax.texts])
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02, bbox_extra_artists=[first, *ax.texts])
    print("wrote", OUT_PNG, "and", OUT_PDF)


if __name__ == "__main__":
    main()
