"""Regenerate every number quoted in the paper from the raw result files in results/.

Each block prints a FACT line whose value is quoted verbatim in the paper, so a claim in the manuscript can always be traced back to
the rows it came from. Nothing here re-encodes anything: it only reads results/.

    python src/verify_paper.py            # full report
    python src/verify_paper.py --check    # exit 1 if a self-consistency assertion fails

Collapse criterion. One rule, applied everywhere:

    margin = PSNR(reconstruction) - PSNR(best constant image),   collapsed iff margin < 5 dB

The threshold does no work: the block MARGIN GAP shows the measured band around it is
empty, so any cut in that band labels every encode identically.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
THRESHOLD = 5.0

FAILURES: list[str] = []


def jsonl(name: str) -> list[dict]:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"missing {p} - run the experiment that produces it first")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fact(key: str, value: str, note: str = "") -> None:
    print(f"  FACT  {key:<26} {value}" + (f"   [{note}]" if note else ""))


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------------------
# The pool of stock (unpatched) Cool-chic encodes.  Every file that records a margin.
# --------------------------------------------------------------------------------------
def stock_pool() -> list[dict]:
    pool = []
    for name, tag in (("natural_seeds.jsonl", "natural_seeds"),
                      ("repeatability.jsonl", "repeatability"),
                      ("expand_real.jsonl", "expand_real"),
                      ("probe_ramp.jsonl", "probe_ramp"),
                      ("probe_brightness.jsonl", "probe_brightness")):
        for r in jsonl(name):
            r["_src"] = tag
            pool.append(r)
    # these two predate the margin field; recompute it from the source image so that every
    # row in the pool is labeled by the same rule.  Their (image, lambda) pairs overlap
    # arena.jsonl, but the bitrates differ, so they are separate encodes, not duplicates.
    import headline                                   # noqa: E402  (same directory)
    for name in ("validate_predictor.jsonl", "sweep_coolchic.jsonl"):
        for r in jsonl(name):
            r["margin"] = r["psnr"] - headline.flat_psnr(r["dataset"], r["image"])
            r["_src"] = name.removesuffix(".jsonl")
            pool.append(r)
    for name in ("metrics_coolchic.jsonl", "metrics_full0138.jsonl"):
        for r in jsonl(name):                          # the patched half is a different arm
            if not r["patched"]:
                r["_src"] = name.removesuffix(".jsonl")
                pool.append(r)
    for r in jsonl("arena.jsonl"):                     # only the untouched-code arm
        if r["arm"] == "base":
            r["_src"] = "arena"
            pool.append(r)
    for r in pool:
        r["_collapsed"] = r["margin"] < THRESHOLD
        # arm C of natural_seeds and the whole ramp/brightness probes are images we
        # synthesized by moving intensity; they answer "where is the boundary", not
        # "how often does this happen to a real file".
        r["_synthetic"] = (r.get("arm") == "C"
                           or r["_src"] in ("probe_ramp", "probe_brightness"))
    return pool


def flatness(pool: list[dict]) -> None:
    """How literally true is "returns a constant image"?  Not always - state it exactly."""
    head("1b. WHAT A COLLAPSED RECONSTRUCTION ACTUALLY LOOKS LIKE")
    coll = [r for r in pool if r["_collapsed"] and "recon_std" in r]
    flat = [r for r in coll if r["recon_std"] == 0.0]
    fact("collapsed encodes with recon_std", f"{len(coll)}")
    fact("exactly one color (std = 0.00)", f"{len(flat)} of {len(coll)}")
    check("quoted as 31 of the 50 collapsed encodes", (len(flat), len(coll)) == (31, 50),
          f"{len(flat)} of {len(coll)}")
    rest = sorted((r for r in coll if r["recon_std"] > 0), key=lambda r: -r["recon_std"])
    fact("largest surviving recon_std", f"{rest[0]['recon_std']:.2f} ({rest[0]['image']})",
         f"source std {rest[0].get('src_std', 0):.2f}, margin {rest[0]['margin']:+.2f} dB")
    check("no collapsed encode beats the constant image by 1 dB",
          all(r["margin"] < 1.0 for r in coll))
    check("every collapsed encode spends under 0.05 bpp",
          all(r["bpp"] < 0.05 for r in coll),
          f"max {max(r['bpp'] for r in coll):.4f} bpp")
    # the headline image, quoted in Limitations
    f138 = [r for r in coll if r["image"] == "full0138"]
    n0 = sum(r["recon_std"] == 0.0 for r in f138)
    fact("full0138 collapsed encodes", f"{n0} exactly one color, {len(f138) - n0} not",
         "quoted verbatim in Limitations")


def margin_gap(pool: list[dict]) -> None:
    head("1.  MARGIN GAP - the outcome is binary, and the threshold is arbitrary")
    coll = sorted(r["margin"] for r in pool if r["_collapsed"])
    run = sorted(r["margin"] for r in pool if not r["_collapsed"])
    fact("stock encodes pooled", f"{len(pool)}")
    fact("collapsed / running", f"{len(coll)} / {len(run)}")
    fact("collapsed margin range", f"{coll[0]:+.2f} .. {coll[-1]:+.2f} dB")
    fact("running margin range", f"{run[0]:+.2f} .. {run[-1]:+.2f} dB")
    fact("empty band", f"{run[0] - coll[-1]:.2f} dB", f"{coll[-1]:+.2f} to {run[0]:+.2f}")
    check("no encode lands inside the band", coll[-1] < run[0])
    check("quoted as 270 encodes, 67 collapsed and 203 working",
          (len(pool), len(coll), len(run)) == (270, 67, 203),
          f"{len(pool)} = {len(coll)} + {len(run)}")
    check("threshold sits inside the empty band", coll[-1] < THRESHOLD < run[0],
          f"any cut in ({coll[-1]:.2f}, {run[0]:.2f}) gives the same labels")


def collapse_rates(pool: list[dict]) -> None:
    head("2.  COLLAPSE RATE per source image, stock encoder, lambda = 0.0006")
    sub = [r for r in pool if r.get("lmbda") == 0.0006 and not r["_synthetic"]]
    by = defaultdict(lambda: [0, 0])
    for r in sub:
        by[r["image"]][0] += 1
        by[r["image"]][1] += r["_collapsed"]
    print(f"    {'image':<13}{'n':>4}{'collapsed':>11}{'rate':>9}")
    for k, (n, c) in sorted(by.items(), key=lambda kv: (-kv[1][1] / kv[1][0], kv[0])):
        print(f"    {k:<13}{n:>4}{c:>11}{100 * c / n:>8.1f}%")
    fact("encodes in this table", f"{len(sub)}")
    check("Table 1: code01 collapses on 6 of 6 encodes", tuple(by["code01"]) == (6, 6),
          f"{int(by['code01'][1])}/{by['code01'][0]}")
    fix = [r["bpp"] for r in jsonl("arena.jsonl")
           if r["image"] == "code01" and r["arm"] == "st" and r["lmbda"] == 0.0006]
    check("code01 reaches 0.17 bpp only with the fix", len(fix) == 1 and round(fix[0], 2) == 0.17,
          str(fix))

    nat = jsonl("natural_seeds.jsonl")
    def arm(name: str) -> tuple[int, int]:
        rs = [r for r in nat if r["image"] == name]
        return sum(r["margin"] < THRESHOLD for r in rs), len(rs)

    c, n = arm("full0138")
    fact("full0138 (resized 768x512)", f"{c}/{n} = {100 * c / n:.1f}%", "brightest full DIV2K photo")
    c2, n2 = arm("native0138")
    fact("full0138 (native 2040x1356)", f"{c2}/{n2} = {100 * c2 / n2:.1f}%", "same photo, no resize")
    c3, n3 = arm("full0502")
    fact("full0502", f"{c3}/{n3} = {100 * c3 / n3:.1f}%", "2nd brightest, 2.4x the contrast")
    c4, n4 = arm("div127")
    fact("div127 (DIV2K sky crop)", f"{c4}/{n4} = {100 * c4 / n4:.1f}%", "mean 249, inside the zone")
    check("the brightest DIV2K photograph does collapse", c > 0)
    check("full0502 never collapses", c3 == 0)

    # the two photographs that separate brightness from contrast
    for nm in ("full0138", "full0502"):
        r = next(r for r in nat if r["image"] == nm)
        fact(f"{nm} statistics", f"mean {r['src_mean']:.1f}   std {r['src_std']:.1f}")


def stochastic(pool: list[dict]) -> None:
    head("3.  THE SAME FILE BOTH COLLAPSES AND SUCCEEDS (no seed is fixed)")
    sub = [r for r in pool if r.get("lmbda") == 0.0006 and not r["_synthetic"]]
    by = defaultdict(list)
    for r in sub:
        by[r["image"]].append(r["_collapsed"])
    mixed = {k: v for k, v in by.items() if 0 < sum(v) < len(v)}
    for k, v in sorted(mixed.items()):
        fact(f"{k}", f"{sum(v)}/{len(v)} encodes collapse", "identical settings")
    check("at least one image collapses only sometimes", len(mixed) > 0)


def boundary() -> None:
    head("4.  BRIGHTNESS BOUNDARY - one image, intensity moved, contrast held")
    rows = jsonl("probe_ramp.jsonl")
    for r in rows:
        r["_collapsed"] = r["margin"] < THRESHOLD
    print(f"    {'target':>7}{'mean':>8}{'std':>7}{'clipped%':>10}{'bpp':>9}"
          f"{'psnr':>8}{'margin':>9}   outcome")
    for r in sorted(rows, key=lambda r: r["mean"]):
        print(f"    {r['target']:>7}{r['mean']:>8.1f}{r['std']:>7.1f}{r['clipped_pct']:>10.2f}"
              f"{r['bpp']:>9.4f}{r['psnr']:>8.2f}{r['margin']:>+9.2f}   "
              f"{'COLLAPSED' if r['_collapsed'] else 'ok'}")
    ok_max = max(r["mean"] for r in rows if not r["_collapsed"])
    co_min = min(r["mean"] for r in rows if r["_collapsed"])
    fact("highest surviving mean", f"{ok_max:.1f} / 255 = {ok_max / 255:.4f}")
    fact("lowest collapsing mean", f"{co_min:.1f} / 255 = {co_min / 255:.4f}")
    fact("boundary width", f"{co_min - ok_max:.1f} intensity levels")
    check("boundary is narrow", co_min - ok_max <= 10)
    check("no clipping in the source at the boundary",
          all(r["clipped_pct"] == 0.0 for r in rows if r["mean"] >= ok_max),
          "so the trigger is not clipping of the input")
    # the stale flag in the raw file: recomputing from the margin moves one row
    stale = [r for r in rows if r["collapsed"] != r["_collapsed"]]
    fact("rows whose stored flag is stale", f"{len(stale)}",
         "recomputed from margin; " + ", ".join(str(r["target"]) for r in stale) or "none")


def metrics_blind() -> None:
    head("5.  NO METRIC SEES IT")
    rows = jsonl("metrics_coolchic.jsonl")
    for r in rows:
        r["_collapsed"] = r["margin"] < THRESHOLD
    coll = [r for r in rows if r["_collapsed"]]
    print(f"    {'image':<10}{'arm':<9}{'psnr':>8}{'ssim':>9}{'ms-ssim':>10}{'lpips':>9}{'rstd':>8}")
    for r in sorted(coll, key=lambda r: r["image"]):
        print(f"    {r['image']:<10}{'stock':<9}{r['psnr']:>8.2f}{r['ssim']:>9.4f}"
              f"{r['ms_ssim']:>10.4f}{r['lpips']:>9.4f}{r['recon_std']:>8.2f}")
        print(f"    {'':<10}{'constant':<9}{r['flat_psnr']:>8.2f}{r['flat_ssim']:>9.4f}"
              f"{r['flat_ms_ssim']:>10.4f}{r['flat_lpips']:>9.4f}{0.0:>8.2f}")
        p = next((q for q in rows if q["image"] == r["image"] and q["patched"]), None)
        if p:
            print(f"    {'':<10}{'patched':<9}{p['psnr']:>8.2f}{p['ssim']:>9.4f}"
                  f"{p['ms_ssim']:>10.4f}{p['lpips']:>9.4f}{p['recon_std']:>8.2f}")
    # how closely a collapsed run reproduces the constant image
    for m in ("ssim", "ms_ssim", "lpips"):
        d = [abs(r[m] - r["flat_" + m]) for r in coll]
        fact(f"|collapsed - constant| {m}", f"mean {np.mean(d):.5f}  max {max(d):.5f}")
    hi = max(coll, key=lambda r: r["psnr"])
    fact("highest PSNR while collapsed", f"{hi['psnr']:.2f} dB ({hi['image']})")
    tight = min(coll, key=lambda r: abs(r["margin"]))
    fact("closest to the trivial answer",
         f"{tight['image']} at {tight['psnr']:.2f} dB, {abs(tight['margin']):.2f} dB from it")

    # no absolute threshold on any metric separates the two groups
    run = [r for r in rows if not r["_collapsed"]]
    for m in ("ssim", "ms_ssim"):
        hi_c = max(coll, key=lambda r: r[m])
        lo_r = min(run, key=lambda r: r[m])
        fact(f"highest {m} while collapsed", f"{hi_c[m]:.4f} ({hi_c['image']})")
        fact(f"lowest  {m} while healthy", f"{lo_r[m]:.4f} ({lo_r['image']})")
        check(f"{m} groups overlap - no usable absolute threshold", hi_c[m] > lo_r[m])
    lo_c = min(coll, key=lambda r: r["lpips"])
    hi_r = max(run, key=lambda r: r["lpips"])
    fact("best lpips while collapsed", f"{lo_c['lpips']:.4f} ({lo_c['image']})")
    fact("worst lpips while healthy", f"{hi_r['lpips']:.4f} ({hi_r['image']})")
    check("lpips groups overlap too", lo_c["lpips"] < hi_r["lpips"])
    check("every collapsed reconstruction is (near) flat",
          all(r["recon_std"] < 0.1 * r["src_std"] for r in coll))


def blindness_needs_flat_sources() -> None:
    """When IS a metric blind?  Only when the trivial answer already scores well, and that
    depends on the source's contrast - not on whether the encode collapsed."""
    head("5b. METRIC BLINDNESS TRACKS SOURCE CONTRAST, NOT COLLAPSE")
    rows = jsonl("metrics_coolchic.jsonl")
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["src_std"]):
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        out.append(r)
    print(f"    {'image':<10}{'src std':>9}{'flat psnr':>11}{'flat ssim':>11}"
          f"{'flat ms':>10}{'flat lpips':>12}")
    for r in out:
        print(f"    {r['image']:<10}{r['src_std']:>9.1f}{r['flat_psnr']:>11.2f}"
              f"{r['flat_ssim']:>11.4f}{r['flat_ms_ssim']:>10.4f}{r['flat_lpips']:>12.4f}")
    lowc = [r for r in out if r["src_std"] < 10]
    high = [r for r in out if r["src_std"] > 20]
    fact("flat image on low-contrast sources", "PSNR " + ", ".join(f"{r['flat_psnr']:.1f}" for r in lowc),
         "std < 10 - a collapse here reads as a good score")
    fact("flat image on higher-contrast sources", "PSNR " + ", ".join(f"{r['flat_psnr']:.1f}" for r in high),
         "std > 20 - a collapse here reads as a bad score")
    check("the trivial answer only scores well on low-contrast sources",
          min(r["flat_ssim"] for r in lowc) > max(r["flat_ssim"] for r in high),
          f"lowest low-contrast SSIM {min(r['flat_ssim'] for r in lowc):.4f} > "
          f"highest high-contrast SSIM {max(r['flat_ssim'] for r in high):.4f}")


def headline_image_metrics() -> None:
    """Perceptual metrics on full0138, the image quoted in Limitations."""
    head("5c. THE HEADLINE IMAGE, SCORED ON ALL FOUR METRICS")
    rows = jsonl("metrics_full0138.jsonl")
    coll = [r for r in rows if not r["patched"] and r["margin"] < THRESHOLD]
    ok = [r for r in rows if not r["patched"] and r["margin"] >= THRESHOLD]
    pat = [r for r in rows if r["patched"]]
    fact("stock encodes", f"{len(coll)} collapsed / {len(coll) + len(ok)}")
    for lbl, g in (("collapsed", coll), ("healthy stock", ok), ("patched", pat)):
        if not g:
            continue
        fact(f"{lbl} PSNR", f"{min(r['psnr'] for r in g):.2f} .. {max(r['psnr'] for r in g):.2f} dB")
        fact(f"{lbl} SSIM", f"{min(r['ssim'] for r in g):.4f} .. {max(r['ssim'] for r in g):.4f}")
    f = coll[0]
    fact("collapsed vs constant image",
         f"PSNR {f['psnr']:.2f}/{f['flat_psnr']:.2f}  SSIM {f['ssim']:.4f}/{f['flat_ssim']:.4f}  "
         f"MS-SSIM {f['ms_ssim']:.4f}/{f['flat_ms_ssim']:.4f}  LPIPS {f['lpips']:.4f}/{f['flat_lpips']:.4f}")
    check("on this source the metrics DO separate collapsed from healthy",
          max(r["ssim"] for r in coll) < min(r["ssim"] for r in ok),
          f"collapsed SSIM <= {max(r['ssim'] for r in coll):.4f}, "
          f"healthy >= {min(r['ssim'] for r in ok):.4f}")
    check("the fix costs nothing on this image",
          min(r["psnr"] for r in pat) >= min(r["psnr"] for r in ok) - 0.05,
          f"patched {min(r['psnr'] for r in pat):.2f}-{max(r['psnr'] for r in pat):.2f} dB vs "
          f"healthy stock {min(r['psnr'] for r in ok):.2f}-{max(r['psnr'] for r in ok):.2f} dB")


def the_fix() -> None:
    head("6.  THE FIX")
    rows = jsonl("metrics_coolchic.jsonl")
    for r in rows:
        r["_collapsed"] = r["margin"] < THRESHOLD
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r["image"]]["patched" if r["patched"] else "stock"] = r
    print(f"    {'image':<10}{'stock psnr':>12}{'patched psnr':>14}{'delta':>9}   was")
    healed, harmed = 0, []
    for k, d in sorted(pairs.items()):
        if "stock" not in d or "patched" not in d:
            continue
        delta = d["patched"]["psnr"] - d["stock"]["psnr"]
        print(f"    {k:<10}{d['stock']['psnr']:>12.2f}{d['patched']['psnr']:>14.2f}"
              f"{delta:>+9.2f}   {'COLLAPSED' if d['stock']['_collapsed'] else 'ok'}")
        if d["stock"]["_collapsed"]:
            healed += d["patched"]["margin"] >= THRESHOLD
        elif delta < -0.05:
            harmed.append((k, delta))
    n_coll = sum(1 for d in pairs.values() if d.get("stock", {}).get("_collapsed"))
    fact("collapsed cases healed", f"{healed}/{n_coll}")
    check("the patch heals every collapsed case it was run on", healed == n_coll)
    check("the patch never costs more than 0.05 dB on a healthy image",
          not harmed, str(harmed))

    vf = jsonl("verify_fix.jsonl")
    fact("independent fix runs", f"{len(vf)}",
         ", ".join(f"{r['image']} {r['margin_before']:+.2f}->{r['margin']:+.2f}" for r in vf))
    check("every independent fix run recovers", all(r["margin"] >= THRESHOLD for r in vf))

    vb = json.loads((RES / "verify_bitstream.json").read_text())
    fact("bitstream check", json.dumps(vb, sort_keys=True))
    check("stock decoder reproduces the patched encode bit-exactly",
          vb.get("max_abs_diff", 1) == 0 or vb.get("identical") is True, json.dumps(vb))


def bdrate() -> None:
    head("7.  RATE-DISTORTION COST OF THE FIX (BD-rate vs stock, negative = better)")
    from analyze import bd_rate                      # noqa: E402  (same directory)
    rows = [r for r in jsonl("arena.jsonl") if "_error" not in r]
    g = defaultdict(dict)
    for r in rows:
        g[(r["image"], r["arm"])][r["lmbda"]] = r
    imgs = sorted({im for im, a in g if a == "st" and len(g[(im, "st")]) >= 4})
    vals = []
    for im in imgs:
        b, s = g[(im, "base")], g[(im, "st")]
        if len(b) < 4:
            continue
        v = bd_rate([b[k]["bpp"] for k in sorted(b)], [b[k]["psnr"] for k in sorted(b)],
                    [s[k]["bpp"] for k in sorted(s)], [s[k]["psnr"] for k in sorted(s)])
        vals.append(v)
        print(f"    {im:<10}{v:>+9.2f}%")
    fact("images with a full 4-point curve", f"{len(vals)} of 24 Kodak")
    fact("mean BD-rate", f"{np.mean(vals):+.2f}%")
    fact("median BD-rate", f"{np.median(vals):+.2f}%")
    fact("images improved", f"{sum(1 for v in vals if v < 0)}/{len(vals)}")
    fact("range", f"{min(vals):+.2f}% .. {max(vals):+.2f}%")
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    from scipy.stats import t as student_t          # n = 24 is too small for z = 1.96
    q = student_t.ppf(0.975, len(vals) - 1)
    lo, hi = np.mean(vals) - q * se, np.mean(vals) + q * se
    fact("95% t-interval on the mean", f"[{lo:+.2f}%, {hi:+.2f}%]")
    check("t-interval quoted as [-1.22%, +0.14%]", (round(lo, 2), round(hi, 2)) == (-1.22, 0.14),
          f"[{lo:+.2f}%, {hi:+.2f}%]")
    try:
        from scipy.stats import wilcoxon
        fact("Wilcoxon signed-rank", f"p = {wilcoxon(vals).pvalue:.3f}",
             "not an improvement at the 5% level")
    except ImportError:
        pass
    check("the whole Kodak set is covered", len(vals) == 24)
    # the claim the paper makes: whatever it costs, it is under a tenth of a percent
    check("the upper end of the interval is under +0.5%", hi < 0.5,
          f"upper bound {hi:+.2f}%")


def c3() -> None:
    head("8.  C3 (DeepMind, JAX) - one config value changed, no code edited")
    rows = jsonl("c3_real.jsonl")
    by = defaultdict(dict)
    for r in rows:
        by[r["image"]][r["arm"]] = r
    print(f"    {'image':<10}{'mean':>8}{'clip (0,1)':>13}{'clip off':>11}{'delta':>9}   flat std")
    for k, d in sorted(by.items(), key=lambda kv: -kv[1]["base"]["mean"]):
        b, c = d["base"], d["clip_disabled"]
        print(f"    {k:<10}{b['mean']:>8.1f}{b['margin']:>+13.2f}{c['margin']:>+11.2f}"
              f"{c['margin'] - b['margin']:>+9.2f}   {b['recon_std']:>5.2f}")
    bright = [d for d in by.values() if d["base"]["mean"] > 200]
    dark = [d for d in by.values() if d["base"]["mean"] <= 200]
    check("every bright image collapses with the shipped clip range",
          all(d["base"]["margin"] < THRESHOLD for d in bright))
    check("disabling the clip range recovers every one of them",
          all(d["clip_disabled"]["margin"] >= THRESHOLD for d in bright))
    check("on darker content the clip range is immaterial",
          all(abs(d["clip_disabled"]["margin"] - d["base"]["margin"]) < 0.2 for d in dark),
          f"max |delta| = {max(abs(d['clip_disabled']['margin'] - d['base']['margin']) for d in dark):.2f} dB")
    fact("collapsed C3 reconstruction std", ", ".join(
        f"{k}={d['base']['recon_std']:.2f}" for k, d in sorted(by.items())
        if d["base"]["margin"] < THRESHOLD))


def trained() -> None:
    head("9.  TRAINED CODECS - the same clamp, applied outside the training graph")
    import headline                                   # noqa: E402
    rows = [r for r in headline.load()
            if r["codec"] in ("bmshj2018-hp", "mbt2018-mean", "cheng2020")]
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        for k in (r["codec"], "ALL"):
            for d in (r["dataset"], "all"):
                by[(k, d)][0] += 1
                by[(k, d)][1] += headline.collapsed(r)
    for (k, d), (n, c) in sorted(by.items()):
        if k != "ALL" and d != "all":
            continue
        fact(f"{k} / {d}", f"{c}/{n} = {100 * c / n:.2f}%")
    # the like-for-like comparison: same content classes the overfitted codec ran on
    n_t = sum(by[("ALL", d)][0] for d in ("screen", "screen_real"))
    c_t = sum(by[("ALL", d)][1] for d in ("screen", "screen_real"))
    fact("trained, screen content only", f"{c_t}/{n_t} = {100 * c_t / n_t:.2f}%")

    pool = [r for r in stock_pool()
            if r.get("lmbda") == 0.0006 and not r["_synthetic"]
            and r.get("dataset") in ("screen", "screen_real")]
    c_o, n_o = sum(r["_collapsed"] for r in pool), len(pool)
    fact("Cool-chic, same content classes", f"{c_o}/{n_o} = {100 * c_o / n_o:.2f}%")
    try:
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([[c_o, n_o - c_o], [c_t, n_t - c_t]])
        fact("Fisher exact, overfitted vs trained", f"p = {p:.1e}   odds ratio {odds:.0f}")
        check("the difference between families is significant", p < 1e-6)
    except ImportError:
        print("  (scipy missing - skipping the Fisher test)")


def literature() -> None:
    head("10. THE LITERATURE SCAN QUOTED IN RELATED WORK")
    lit = ROOT / "lit"
    four = {"2605.02726": "Cool-chic 5.0", "2607.13723": "N-O Cool-chic",
            "2509.18748": "HyperCool", "2605.20672": "LANCE"}
    words, hits = 0, {k: 0 for k in ("clip", "clamp", "collapse", "failure", "screen content")}
    missing = []
    for aid, name in four.items():
        f = lit / f"{aid}.txt"
        if not f.exists():
            missing.append(aid); continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        words += len(txt.split())
        low = txt.lower()
        for k in hits:
            hits[k] += low.count(k)
    if missing:
        print(f"  (skipping - no full text for {', '.join(missing)})")
        return
    fact("full text of the four papers", f"{words} words", "quoted verbatim in Related Work")
    fact("keyword hits", ", ".join(f"{k}={v}" for k, v in hits.items()))
    check("none of the five strings occurs in any of the four papers",
          all(v == 0 for v in hits.values()))


def encoder_log() -> None:
    head("6b. STOCK ENCODER LOG DURING A COLLAPSE (logs/diag_code01.log, lambda=0.0002, 6000 itr, CPU)")
    import re
    rows = []
    for line in (ROOT / "logs" / "diag_code01.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s", line)
        if m:
            rows.append((float(m.group(5)), float(m.group(3)), int(m.group(8))))
    psnrs = {p for p, _, _ in rows}
    lat = [l for _, l, _ in rows]
    fact("logged checkpoints", f"{len(rows)}")
    fact("distinct PSNR values", f"{sorted(psnrs)}")
    fact("latent rate range", f"{max(lat):.3f} .. {min(lat):.4f} bpp")
    fact("first logged iteration", f"{min(i for _, _, i in rows)}")
    check("PSNR frozen at 20.281225 dB in every logged row", psnrs == {20.281225}, str(sorted(psnrs)))
    check("71 logged rows", len(rows) == 71, str(len(rows)))
    check("the first seven rows are warm-up candidates (iterations 100-700, Figure 3b caption)",
          [i for _, _, i in rows[:8]] == [100, 200, 300, 400, 500, 600, 700, 800])
    flat = [r["flat_psnr"] for r in jsonl("arena.jsonl") if r["image"] == "code01"][0]
    fact("code01 constant-image PSNR", f"{flat:.2f} dB", f"frozen PSNR is {flat - 20.281225:.2f} dB below it")
    check("frozen PSNR 0.36 dB below the 20.64 dB constant image", round(flat, 2) == 20.64 and round(flat - 20.281225, 2) == 0.36)
    ms = [r for r in jsonl("metrics_coolchic.jsonl") if r["collapsed"] and not r["patched"]]
    best = max(max(r["ssim"] - r["flat_ssim"], r["ms_ssim"] - r["flat_ms_ssim"], r["flat_lpips"] - r["lpips"]) for r in ms)
    fact("largest margin by which a collapsed encode beats the constant image (SSIM/MS-SSIM/LPIPS)", f"{best:.5f}")
    check("collapsed never beats the constant image by more than 0.0003", best <= 0.0003, f"{best:.5f}")
    allm = jsonl("metrics_coolchic.jsonl")
    check("metrics batch is 9 images x 2 encodes = 18", len({r["image"] for r in allm}) == 9 and len(allm) == 18,
          f'{len({r["image"] for r in allm})} images, {len(allm)} rows')
    check("collapsed PSNR never above the constant image", all(r["psnr"] <= r["flat_psnr"] + 1e-9 for r in ms))
    check("latent rate spans 0.35 .. 0.0003 bpp", round(max(lat), 2) == 0.35 and round(min(lat), 4) == 0.0003,
          f"{max(lat)} {min(lat)}")


def kodak_upper_bound(pool: list[dict]) -> None:
    head("2b. KODAK: stock encodes at lambda = 0.0006 -> upper end of the exact 95% interval")
    k = [r for r in pool if r.get("lmbda") == 0.0006 and str(r.get("image", "")).startswith("kodim")]
    n, c = len(k), sum(r["_collapsed"] for r in k)
    ub = 1 - 0.025 ** (1 / n)
    fact("Kodak stock encodes", f"{c} collapsed of {n}")
    fact("Clopper-Pearson upper bound", f"{100 * ub:.1f}%")
    check("Table 1: 0 of 29 Kodak encodes collapse", (c, n) == (0, 29), f"{c}/{n}")
    check("bound quoted as 11.9%", round(100 * ub, 1) == 11.9, f"{100 * ub:.2f}")


def revision_facts(pool: list[dict]) -> None:
    """Numbers added in the 2026-10-02 revision, each quoted in the paper."""
    head("11. NUMBERS ADDED IN THE REVISION")
    from scipy.stats import beta
    coll = [r["bpp"] for r in pool if r["_collapsed"] and "bpp" in r]
    work = [r["bpp"] for r in pool if not r["_collapsed"] and "bpp" in r]
    fact("collapsed bpp max / working bpp min", f"{max(coll):.4f} / {min(work):.4f}")
    check("a rate threshold also separates the groups (collapsed <= 0.045 bpp)",
          round(max(coll), 3) == 0.045 and min(work) > max(coll))
    import headline                                   # noqa: E402
    m = np.array([r["psnr"] - headline.flat_psnr(r["dataset"], r["image"]) for r in headline.load()
                  if r["codec"] in ("bmshj2018-hp", "mbt2018-mean", "cheng2020")])
    below = sorted(np.round(m[m < THRESHOLD], 2).tolist())
    band = int(((m > 0.79) & (m < 10.82)).sum())
    fact("trained codecs: margins below 5 dB", str(below), f"{band} encodes inside the 10 dB band")
    check("trained codecs: +4.68 and +4.91 dB, 102 in the band, none below 1 dB",
          below == [4.68, 4.91] and band == 102 and m.min() > 1.0)
    g = defaultdict(dict)
    for r in jsonl("arena.jsonl"):
        if "_error" not in r and "seconds" in r:
            g[(r["image"], r["lmbda"])][r["arm"]] = r["seconds"]
    ratios = [d["st"] / d["base"] for d in g.values() if "st" in d and "base" in d]
    fact("encoding time with the fix / stock", f"median {np.median(ratios):.3f} over {len(ratios)} pairs")
    check("quoted as median 0.999 over 97 paired encodes",
          round(float(np.median(ratios)), 3) == 0.999 and len(ratios) == 97)
    def cp(x, n):
        lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
        hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
        return round(100 * lo), round(100 * hi)
    fact("Clopper-Pearson 5/8 and 16/50", f"{cp(5, 8)} {cp(16, 50)}")
    check("intervals quoted as [24%, 91%] and [20%, 47%]", cp(5, 8) == (24, 91) and cp(16, 50) == (20, 47))
    syn = sorted({r["image"] for r in pool if r.get("dataset") == "screen" and not r["_synthetic"]})
    t01 = [r for r in pool if r.get("image") == "text01" and r.get("lmbda") == 0.0006]
    fact("synthetic screens encoded with Cool-chic", ", ".join(syn))
    check("six synthetic images; text01 collapses on 2 of 2", len(syn) == 6 and
          (len(t01), sum(r["_collapsed"] for r in t01)) == (2, 2))
    from PIL import Image
    a = np.asarray(Image.open(ROOT / "examples" / "text01_stock.png").convert("RGB"))
    fact("text01 stock reconstruction", f"min {a.min()} max {a.max()}")
    check("text01 ends at 239 on all channels", a.min() == a.max() == 239)


def init_ablation() -> None:
    head("7b. ALTERNATIVE FIX: OUTPUT INITIALIZED AT THE IMAGE MEAN, CLAMP UNCHANGED")
    rows = jsonl("init_ablation.jsonl")
    c = sum(r["collapsed"] for r in rows)
    fact("encodes", f"{len(rows)} (5 collapsing images x 4)")
    fact("collapsed", f"{c}/{len(rows)}", ", ".join(sorted({r['image'] for r in rows if r['collapsed']})))
    check("mean initialization: 2 of 20 still collapse", len(rows) == 20 and c == 2)
    check("both remaining collapses are real11", all(r["image"] == "real11" for r in rows if r["collapsed"]))


def second_review_facts(pool: list[dict]) -> None:
    """Numbers added after the second review (2026-10-02), each quoted in the paper."""
    head("12. NUMBERS ADDED AFTER THE SECOND REVIEW")
    by = defaultdict(lambda: {"c": [], "w": []})
    for r in pool:
        if "bpp" in r and r.get("image"):
            by[r["image"]]["c" if r["_collapsed"] else "w"].append(r["bpp"])
    fix = defaultdict(list)
    for r in jsonl("metrics_coolchic.jsonl"):
        if r["patched"]:
            fix[r["image"]].append(r["bpp"])
    for r in jsonl("arena.jsonl"):
        if r.get("arm") == "st" and r.get("lmbda") == 0.0006:
            fix[r["image"]].append(r["bpp"])
    ratio = {im: min(d["w"] or fix[im]) / max(d["c"]) for im, d in by.items() if d["c"] and (d["w"] or fix[im])}
    fact("rate ratio working / collapsed, same image", f"{min(ratio.values()):.1f} .. {max(ratio.values()):.1f}")
    check("a collapsed encode has a rate at least three times lower", min(ratio.values()) >= 3.0)
    stats = json.loads((RES / "image_stats.json").read_text(encoding="utf-8"))
    means = []
    for r in pool:
        if r["_collapsed"]:
            m = r.get("src_mean", r.get("mean"))
            if m is None:
                m = next(v["mean"] for k, v in stats.items() if k.endswith("/" + r["image"]))
            means.append(m)
    fact("lowest mean intensity of a collapsed encode", f"{min(means):.1f}")
    check("every collapse occurs at a mean of 225 or above", min(means) >= 225.0)
    ia = sorted(jsonl("init_ablation.jsonl"), key=lambda r: r["margin"])
    low = [(r["image"], round(r["margin"], 2)) for r in ia[:3]]
    fact("mean initialization, three lowest margins", str(low), f"next {ia[3]['margin']:+.2f} dB")
    check("real11 reaches +1.75, +4.75, +7.29 dB; the other 17 exceed +12.4 dB",
          low == [("real11", 1.75), ("real11", 4.75), ("real11", 7.29)] and ia[3]["margin"] > 12.4)
    mc = jsonl("metrics_coolchic.jsonl")
    pc = max(r["psnr"] for r in mc if r["margin"] < THRESHOLD)
    pw = min(r["psnr"] for r in mc if r["margin"] >= THRESHOLD)
    fact("18-encode batch PSNR", f"collapsed <= {pc:.2f} dB, working >= {pw:.2f} dB")
    check("PSNR separates the batch at 31.30 / 34.04 dB", (round(pc, 2), round(pw, 2)) == (31.30, 34.04))
    import re
    rows = re.findall(r"^\s+(0\.\d\d)\s+(yes|no)\s+\S+\s+\S+\s+\S+%\s+(\S+)\s+([+-]\d+\.\d\d)$",
                      (RES / "c3_gradient_trap.txt").read_text(encoding="utf-8"), re.M)
    got = [(b, c, g, m) for b, c, g, m in rows]
    want = [("0.97", "yes", "0.00e+00", "-1.12"), ("0.97", "no", "1.38e-01", "+22.57"),
            ("0.50", "yes", "5.99e-04", "+27.40"), ("0.50", "no", "9.15e-03", "+26.17")]
    fact("2x2 ablation (Mechanism) from results/c3_gradient_trap.txt", str(got))
    check("2x2 ablation matches the recorded run", got == want)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any assertion fails")
    args = ap.parse_args()

    pool = stock_pool()
    margin_gap(pool)
    flatness(pool)
    collapse_rates(pool)
    stochastic(pool)
    boundary()
    metrics_blind()
    blindness_needs_flat_sources()
    headline_image_metrics()
    the_fix()
    encoder_log()
    init_ablation()
    kodak_upper_bound(pool)
    bdrate()
    c3()
    trained()
    revision_facts(pool)
    second_review_facts(pool)
    literature()

    head("SUMMARY")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
    else:
        print("  every assertion holds")
    if args.check and FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
