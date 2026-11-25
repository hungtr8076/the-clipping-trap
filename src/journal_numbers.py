"""Regenerate every number of the paper from results/.

Each number in the manuscript is a LaTeX macro \\V{name}; this script writes their values to
numbers/numbers.tex and the three generated tables to numbers/tab_rates.tex,
numbers/tab_remedies.tex and numbers/tab_metrics.tex.
  - numbers shared with the earlier version are read from the FACT lines of verify_paper.py
    (one source of truth, same assertions);
  - numbers from the later encodes (collapse rates with ten encodes per image, inverted images,
    output mappings, Kodak at 10,000 iterations, screenshots) are computed here.

    python src/journal_numbers.py
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA = ROOT / "data"
OUT = ROOT / "numbers"
sys.path.insert(0, str(ROOT / "src"))
import verify_paper as vp  # noqa: E402

THRESHOLD = vp.THRESHOLD
LAMBDAS = (0.0002, 0.0006, 0.0015, 0.004)
V: dict[str, str] = {}


# ---------------------------------------------------------------- helpers
def facts() -> dict[str, str]:
    """Run every block of verify_paper.py and collect its FACT lines (key -> value text)."""
    got: dict[str, str] = {}
    orig = vp.fact
    vp.fact = lambda key, value, note="": got.__setitem__(key.strip(), f"{value} [[{note}]]" if note else str(value))
    argv = sys.argv
    sys.argv = ["verify_paper.py", "--check"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            vp.main()
    finally:
        vp.fact, sys.argv = orig, argv
    return got


def num(s: str, i: int = 0) -> float:
    return float(re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", s)[i])


def sgn(x: float, d: int = 2) -> str:
    """Signed number in math mode: $-5.09$, $+0.79$."""
    return f"${x:+.{d}f}$"


def pct(x: float, d: int = 2) -> str:
    return f"${x:+.{d}f}$"


def jsonl(name: str) -> list[dict]:
    p = RES / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def pending(name: str) -> None:
    raise SystemExit(f"incomplete result files: cannot compute {name}")


def bd_rate(anchor: dict, test: dict) -> float:
    """BD-rate (%) of test vs anchor; each a {lambda: (bpp, psnr)}; PCHIP on log-rate vs PSNR."""
    def curve(c):
        pts = sorted(c.values(), key=lambda t: t[1])
        p = np.array([t[1] for t in pts])
        r = np.log(np.array([t[0] for t in pts]))
        return PchipInterpolator(p, r), p.min(), p.max()
    fa, la, ha = curve(anchor)
    ft, lt, ht = curve(test)
    lo, hi = max(la, lt), min(ha, ht)
    x = np.linspace(lo, hi, 1001)
    trap = getattr(np, "trapezoid", None) or np.trapz
    diff = trap(ft(x) - fa(x), x) / (hi - lo)
    return (np.exp(diff) - 1) * 100


def t_interval(v: list[float]) -> tuple[float, float]:
    m, s, n = float(np.mean(v)), float(np.std(v, ddof=1)), len(v)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m - h, m + h


def ci_text(lo: float, hi: float) -> str:
    return f"$[{lo:+.2f}\\%, {hi:+.2f}\\%]$"


def cp(k: int, n: int) -> tuple[float, float]:
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo * 100, hi * 100


IMAGE_STATS = json.loads((RES / "image_stats.json").read_text(encoding="utf-8"))


def img_stats(rel: str) -> tuple[float, float]:
    """RGB mean and std of a source, from results/image_stats.json (the DIV2K images are not
    redistributed in this repository, so their statistics are read from the stored file)."""
    d = IMAGE_STATS[rel.removesuffix(".png")]
    return float(d["mean"]), float(d["std"])


# ---------------------------------------------------------------- pool of stock encodes
def extended_pool() -> list[dict]:
    """The pool of verify_paper.py plus every later stock RGB encode, each labeled by margin."""
    pool = vp.stock_pool()
    # metrics_* rows stored their margin against the constant image ROUNDED to 8 bits; every
    # other row uses the unrounded constant image (Section 3).  Relabel them by the same rule.
    sub = {"natural": "kodak", "natural_bright": "natural_bright", "screen": "screen",
           "screen_real": "screen_real"}
    for r in pool:
        if r["_src"] in ("metrics_coolchic", "metrics_full0138"):
            r["margin"] = r["psnr"] - IMAGE_STATS[f"{sub[r['dataset']]}/{r['image']}"]["flat_psnr"]
            r["_collapsed"] = r["margin"] < THRESHOLD
    later = [r for r in jsonl("schedule_yuv_remedies.jsonl") if r["arm"] == "stock" and r["input"] == "rgb"]
    later += [r for r in jsonl("div2k_bright_regions.jsonl")]
    later += [r for r in jsonl("screenshots_rd.jsonl") + jsonl("kodak_10k_rd.jsonl") if r["arm"] == "stock"]
    for name in ("collapse_rates.jsonl", "collapse_rates_inverted.jsonl", "output_mappings_kodak_rd.jsonl", "screenshots_rd_second_stock.jsonl"):
        later += [r for r in jsonl(name) if r["arm"] == "stock"]
    for r in later:
        r["_collapsed"] = r["margin"] < THRESHOLD
    for r in pool + later:   # inverted sources (collapse_rates_inverted and two early probes)
        r["_dark"] = r.get("image", "").endswith("_inv")
    return pool + later


# ---------------------------------------------------------------- blocks
def shared_facts(F: dict[str, str]) -> None:
    V["roundingLoss"] = f"{num(F['PSNR lost by rounding the constant image to 8 bits']):.3f}"
    V["codeOneLogPsnr"] = F["distinct PSNR values"].strip("[] ")
    V["codeOneLogPsnrShort"] = f"{float(V['codeOneLogPsnr']):.2f}"
    V["logRows"] = F["logged checkpoints"]
    lo_hi = F["latent rate range"]
    V["logRateHigh"], V["logRateLow"] = f"{num(lo_hi, 0):.3f}", f"{num(lo_hi, 1):.4f}"
    V["whitePsnr"] = f"{num(F['PSNR of an all-white image against code01']):.6f}"
    e1 = {k: F[f"10,000 iterations, {k}"] for k in ("full0138", "div127", "code01")}
    for key, im in (("eOneFull", "full0138"), ("eOneDiv", "div127"), ("eOneCode", "code01")):
        k, n = (int(x) for x in re.findall(r"(\d+)/(\d+)", e1[im])[0])
        V[key] = f"{k} of {n}"
    k, n = (int(x) for x in re.findall(r"(\d+)/(\d+)", F["YUV420 path, all"])[0])
    V["eTwoAll"] = f"{k} of {n}"
    V["highestWorkingMean"] = f"{num(F['highest surviving mean']):.1f}"
    V["lowestCollapsingMean"] = f"{num(F['lowest collapsing mean']):.1f}"
    V["regionPhotos"] = F["photographs whose brightest Kodak-sized window has a mean >= 225"].split()[0]
    V["metricConstGap"] = f"{num(F['largest margin by which a collapsed encode beats the constant image (SSIM/MS-SSIM/LPIPS)']):.4f}"
    for key, fk in (("fullCollapsedPsnr", "collapsed PSNR"), ("fullWorkingPsnr", "healthy stock PSNR")):
        V[key] = f"{num(F[fk], 0):.2f} to {num(F[fk], 1):.2f}"
    for key, fk in (("fullCollapsedSsim", "collapsed SSIM"), ("fullWorkingSsim", "healthy stock SSIM")):
        V[key] = f"{num(F[fk], 0):.4f} to {num(F[fk], 1):.4f}"
    V["ssimCollapsedMax"] = f"{num(F['highest ssim while collapsed']):.4f}"
    V["ssimWorkingMin"] = f"{num(F['lowest  ssim while healthy']):.4f}"
    V["msssimCollapsedMax"] = f"{num(F['highest ms_ssim while collapsed']):.4f}"
    V["msssimWorkingMin"] = f"{num(F['lowest  ms_ssim while healthy']):.4f}"
    V["lpipsCollapsedBest"] = f"{num(F['best lpips while collapsed']):.4f}"
    V["lpipsWorkingWorst"] = f"{num(F['worst lpips while healthy']):.4f}"
    b = F["18-encode batch PSNR"]
    V["psnrCollapsedMax"], V["psnrWorkingMin"] = f"{num(b, 0):.2f}", f"{num(b, 1):.2f}"
    V["cThreeCode"] = sgn(num(F["code01 base"]))
    V["cThreeText"] = sgn(num(F["text01 base"]))
    V["cThreeCodeWide"] = sgn(num(F["code01 clip_disabled"]))
    V["cThreeTextWide"] = sgn(num(F["text01 clip_disabled"]))
    four = F["four further screenshot crops"]
    shipped = [float(x) for x in re.findall(r"shipped clip range \[([^\]]+)\]", four)[0].split(",")]
    wide = [float(x) for x in re.findall(r"widened \[([^\]]+)\]", four)[0].split(",")]
    V["cThreeFourLow"], V["cThreeFourHigh"] = sgn(min(shipped)), sgn(max(shipped))
    V["cThreeFourWideLow"], V["cThreeFourWideHigh"] = sgn(min(wide)), sgn(max(wide))
    V["trainedEncodes"] = F["ALL / all"].split("/")[1].split()[0]
    lows = [float(x) for x in re.findall(r"[\d.]+", F["trained codecs: margins below 5 dB"].split("[[")[0])]
    V["trainedLowest"], V["trainedSecond"] = sgn(lows[0]), sgn(lows[1])
    V["trainedInBand"] = re.findall(r"\[\[(\d+) encodes inside", F["trained codecs: margins below 5 dB"])[0]
    init = re.findall(r"([\d.]+)\)", F["mean initialization, three lowest margins"])
    V["initLowOne"], V["initLowTwo"], V["initLowThree"] = (sgn(float(x)) for x in init[:3])
    V["initNext"] = sgn(float(re.findall(r"next \+?([\d.]+)", F["mean initialization, three lowest margins"])[0]))
    t = F["encoding time with the fix / stock"]
    V["timeRatio"], V["timePairs"] = f"{num(t, 0):.3f}", str(int(num(t, 1)))
    k = F["PCHIP interpolation: mean and 95% t-interval"]
    V["kodakTwokMean"] = pct(num(k, 0))
    V["kodakTwokCI"] = ci_text(num(k, 1), num(k, 2))


def pool_numbers(pool: list[dict]) -> None:
    light = [r for r in pool if not r.get("_dark")]
    col = [r for r in light if r["_collapsed"]]
    wrk = [r for r in light if not r["_collapsed"]]
    V["poolEncodes"], V["collapsedPool"] = str(len(light)), str(len(col))
    cm = [r["margin"] for r in col]
    wm = [r["margin"] for r in wrk]
    V["collapsedMarginLow"], V["collapsedMarginHigh"] = sgn(min(cm)), sgn(max(cm))
    V["workingMarginLow"], V["workingMarginHigh"] = sgn(min(wm)), sgn(max(wm))
    V["maxCollapsedMargin"] = f"{max(cm):.2f}"
    V["gapWidth"] = f"{min(wm) - max(cm):.2f}"
    V["collapsedUnderCentibit"] = str(sum(1 for r in col if r.get("bpp", 1) < 0.01))
    withstd = [r for r in col if r.get("recon_std") is not None]
    V["stdRecorded"] = str(len(withstd))
    V["singleColor"] = str(sum(1 for r in withstd if round(r["recon_std"], 2) == 0.0))
    top = max(withstd, key=lambda r: r["recon_std"])
    V["maxSurvivingStd"] = f"{top['recon_std']:.1f}"
    V["sourceStdFull"] = f"{top['src_std']:.1f}"
    V["lowestCollapsedMean"] = f"{min(r['src_mean'] for r in col if r.get('src_mean')):.1f}"
    # rate ratio at lambda 0.0006: cheapest working encode (stock or straight-through) of an image
    # over its most expensive collapsed encode
    by = defaultdict(lambda: {"c": [], "w": []})
    for r in light:
        if r.get("lmbda", 0.0006) == 0.0006 and "bpp" in r and r.get("image"):
            by[r["image"]]["c" if r["_collapsed"] else "w"].append(r["bpp"])
    for r in jsonl("metrics_coolchic.jsonl"):
        if r["patched"]:
            by[r["image"]]["w"].append(r["bpp"])
    ratio = [min(d["w"]) / max(d["c"]) for d in by.values() if d["c"] and d["w"]]
    V["rateRatioLow"], V["rateRatioHigh"] = f"{min(ratio):.1f}", f"{max(ratio):.0f}"
    V["rateRatioImages"] = str(len(ratio))
    # the trained codecs' in-band count is a FACT note; make sure the stock pool has none in band
    assert not [m for m in cm + wm if max(cm) < m < min(wm)]


def images() -> None:
    k = [img_stats(f"kodak/kodim{i:02d}.png")[0] for i in range(1, 25)]
    V["kodakMinMean"], V["kodakMaxMean"] = f"{min(k):.1f}", f"{max(k):.1f}"
    m1, s1 = img_stats("natural_bright/full0138.png")
    m2, s2 = img_stats("natural_bright/full0502.png")
    V["meanGapFull"], V["stdRatioFull"] = f"{m1 - m2:.1f}", f"{s2 / s1:.1f}"
    V["stdDiv"] = f"{img_stats('natural_bright/div127.png')[1]:.1f}"
    V["stdRealEleven"] = f"{img_stats('screen_real/real11.png')[1]:.1f}"
    V["stdTextOne"] = f"{img_stats('screen/text01.png')[1]:.1f}"
    V["stdCodeOne"] = f"{img_stats('screen/code01.png')[1]:.1f}"
    for key, pre in (("brightReal", "screen_real/"), ("brightSynth", "screen/")):
        ks = [k for k in IMAGE_STATS if k.startswith(pre)]
        V[key] = f"{sum(IMAGE_STATS[k]['mean'] >= 225 for k in ks)} of {len(ks)}"


def metrics_table() -> None:
    rows = {(r["image"], r["patched"]): r for r in jsonl("metrics_coolchic.jsonl")}
    s, f = rows[("text01", False)], rows[("text01", True)]
    V["figStockBpp"], V["figStockPsnr"] = f"{s['bpp']:.4f}", f"{s['psnr']:.2f}"
    V["figFixBpp"], V["figFixPsnr"] = f"{f['bpp']:.2f}", f"{f['psnr']:.2f}"
    body = []
    coll = [r for r in rows.values() if not r["patched"] and r["collapsed"]]
    assert sorted(r["image"] for r in coll) == ["code01", "div127", "real11", "text01"]
    V["metricConstAbs"] = f"{max(abs(r[m] - r['flat_' + m]) for r in coll for m in ('ssim', 'ms_ssim', 'lpips')):.4f}"
    V["flatSsimText"] = f"{rows[('text01', False)]['flat_ssim']:.2f}"
    V["flatSsimReal"] = f"{rows[('real11', False)]['flat_ssim']:.2f}"
    for im in ("div127", "real11", "text01", "code01"):
        r = rows[(im, False)]
        body.append(f"\\texttt{{{im}}} & collapsed & {r['psnr']:.2f} & {r['ssim']:.4f} & {r['ms_ssim']:.4f} "
                    f"& {r['lpips']:.4f} & {r['recon_std']:.2f} \\\\")
        # constant image rounded to 8 bits (scored by the same metric code)
        body.append(f" & $\\bar x$ & {r['flat_psnr']:.2f} & {r['flat_ssim']:.4f} & {r['flat_ms_ssim']:.4f} "
                    f"& {r['flat_lpips']:.4f} & 0.00 \\\\")
        body.append("\\midrule")
    body = body[:-1]
    (OUT / "tab_metrics.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{\\label{tab:metrics}%\n"
        "The four collapsed stock encodes of the 18-encode batch and the constant image $\\bar x$ of\n"
        "the same source, rounded to 8 bits as a decoded image is and scored by the same code.\n"
        "\\textit{std} is the spatial standard deviation of the reconstruction; the sources have\n"
        "\\V{stdDiv}, \\V{stdRealEleven}, \\V{stdTextOne} and \\V{stdCodeOne}.}\n"
        "\\small\n\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "image & reconstruction & PSNR (dB) & SSIM & MS-SSIM & LPIPS & std \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def rd_ste(name: str, prefix: str, n_itr_label: str):
    rows = jsonl(name)
    by = defaultdict(dict)
    for r in rows:
        by[(r["image"], r["arm"])][r["lmbda"]] = r
    bds, collapsed = [], []
    for im in sorted({k[0] for k in by}):
        s, t = by[(im, "stock")], by[(im, "ste")]
        cs = [l for l, r in s.items() if r["collapsed"]]
        ct = [l for l, r in t.items() if r["collapsed"]]
        if cs or ct:
            collapsed.append((im, cs, ct))
            continue
        bds.append(bd_rate({l: (r["bpp"], r["psnr"]) for l, r in s.items()},
                           {l: (r["bpp"], r["psnr"]) for l, r in t.items()}))
    lo, hi = t_interval(bds)
    V[f"{prefix}Mean"] = pct(np.mean(bds))
    V[f"{prefix}Median"] = pct(np.median(bds))
    V[f"{prefix}Min"], V[f"{prefix}Max"] = pct(min(bds)), pct(max(bds))
    V[f"{prefix}CI"] = ci_text(lo, hi)
    V[f"{prefix}P"] = f"{stats.wilcoxon(bds).pvalue:.2f}"
    V[f"{prefix}Images"] = str(len(bds))
    V[f"{prefix}Improved"] = str(sum(1 for b in bds if b < 0))
    return bds, collapsed, rows


def rd_numbers() -> None:
    bds, col, rows = rd_ste("kodak_10k_rd.jsonl", "kodakTenk", "10000")
    assert not col and len(bds) == 24 and len(rows) == 192
    bds, col, rows = rd_ste("screenshots_rd.jsonl", "screen", "2000")
    assert len(rows) == 112
    V["screenSaving"] = f"{-np.mean(bds):.1f}"
    V["screenPoints"] = str(sum(1 for r in rows if r["arm"] == "stock"))
    V["screenStockCollapsed"] = str(sum(len(c[1]) for c in col))
    assert sum(len(c[2]) for c in col) == 0
    V["screenCollapsedList"] = ", ".join(
        f"\\texttt{{{im}}} at $\\lambda = {', '.join(str(l) for l in cs)}$" for im, cs, _ in col)


def later_numbers() -> None:
    t1 = jsonl("collapse_rates.jsonl")
    dark = jsonl("collapse_rates_inverted.jsonl")
    rem = jsonl("output_mappings.jsonl")
    cost = jsonl("output_mappings_kodak_rd.jsonl")
    noise = jsonl("screenshots_rd_second_stock.jsonl")
    done = {"t1": len(t1) >= 130, "dark": len(dark) >= 80, "remedy": len(rem) >= 150,
            "cost": len(cost) >= 192, "noise": len(noise) >= 56}
    gpus = sorted({r.get("gpu", "") for r in t1 + dark + rem + cost + noise if r.get("gpu")})
    six = sorted({r.get("gpu", "") for r in jsonl("screenshots_rd.jsonl") + jsonl("kodak_10k_rd.jsonl")})
    assert len(six) == 1 and six[0], six
    V["rSixGpu"] = six[0].replace("NVIDIA GeForce ", "NVIDIA ")
    if all(done.values()) and gpus:
        assert len(gpus) == 1, gpus
        V["rSevenGpu"] = " and ".join(g.replace("NVIDIA GeForce ", "NVIDIA ") for g in gpus)
    else:
        pending("rSevenGpu")
    if done["t1"]:
        bright = ("full0138", "full0502", "div127", "code01", "text01", "text04", "real09", "real11")
        kn = {im: (sum(r["collapsed"] for r in t1 if r["image"] == im),
                   sum(1 for r in t1 if r["image"] == im)) for im in bright}
        assert all(n == 10 for _, n in kn.values())
        always = [im for im in bright if kn[im][0] == 10]
        never = [im for im in bright if kn[im][0] == 0]
        some = [im for im in bright if 0 < kn[im][0] < 10]
        tt = lambda ims: ", ".join(f"\\texttt{{{i}}}" for i in ims[:-1]) + f" and \\texttt{{{ims[-1]}}}" \
            if len(ims) > 1 else f"\\texttt{{{ims[0]}}}"
        word = "zero one two three four five six seven eight".split()
        V["tOneAlwaysList"], V["tOneAlwaysCount"] = tt(always), word[len(always)]
        V["tOneSomeList"], V["tOneSomeCount"] = tt(some), word[len(some)]
        V["tOneSomeCountCap"] = word[len(some)].capitalize()
        V["tOneSomeLow"] = word[min(kn[i][0] for i in some)]
        V["tOneSomeHigh"] = word[max(kn[i][0] for i in some)]
        V["tOneNeverList"] = tt(never)
        kod = [r for r in t1 if r["image"].startswith("kodim")]
        V["tOneKodakEncodes"], V["tOneKodakCollapsed"] = str(len(kod)), str(sum(r["collapsed"] for r in kod))
    else:
        for k in ("tOneAlwaysList", "tOneAlwaysCount", "tOneSomeList", "tOneSomeCount", "tOneSomeLow",
                  "tOneSomeHigh", "tOneNeverList", "tOneKodakEncodes", "tOneKodakCollapsed"):
            pending(k)
    if done["dark"]:
        V["darkEncodes"], V["darkCollapsed"] = str(len(dark)), str(sum(r["collapsed"] for r in dark))
        V["darkCIHigh"] = f"{cp(sum(r['collapsed'] for r in dark), len(dark))[1]:.1f}"
    else:
        for k in ("darkEncodes", "darkCollapsed", "darkCIHigh"):
            pending(k)
    if done["remedy"] and done["cost"]:
        arm_k = {a: (sum(r["collapsed"] for r in rem if r["arm"] == a), sum(1 for r in rem if r["arm"] == a))
                 for a in ("ste", "noclamp", "leaky", "sigmoid", "init")}
        assert all(n == 30 for _, n in arm_k.values())
        assert all(arm_k[a][0] == 0 for a in ("ste", "noclamp", "leaky"))
        V["sigCollapsed"], V["initCollapsed"] = str(arm_k["sigmoid"][0]), str(arm_k["init"][0])
        V["sigRecovered"], V["initRecovered"] = str(30 - arm_k["sigmoid"][0]), str(30 - arm_k["init"][0])
        per = {im: sum(r["collapsed"] for r in rem if r["arm"] == "sigmoid" and r["image"] == im)
               for im in sorted({r["image"] for r in rem})}
        V["sigAllFive"] = " and ".join(f"\\texttt{{{im}}}" for im, k in per.items() if k == 5)
        part = [(im, k) for im, k in per.items() if 0 < k < 5]
        V["sigPartList"] = " and ".join(f"\\texttt{{{im}}} on {'zero one two three four'.split()[k]}" for im, k in part)
        ic = [r for r in rem if r["arm"] == "init" and r["collapsed"]]
        V["initCollapsedImage"] = " and ".join(f"\\texttt{{{r['image']}}}" for r in ic)
        V["initCollapsedMargin"] = ", ".join(sgn(r["margin"]) for r in ic)
        stock5 = [r for r in t1 if r["image"] in COLLAPSE6 and r["rep"] <= 5]
        V["stockFiveRecovered"] = str(sum(not r["collapsed"] for r in stock5))
        V["stockFiveCollapsed"] = str(sum(r["collapsed"] for r in stock5))
        V["textOneCollapsed"] = str(sum(r["collapsed"] for r in t1 if r["image"] == "text01"))
        rdiff = []
        for lam in LAMBDAS:
            s = [r["bpp"] for r in cost if r["arm"] == "stock" and r["rep"] == 1 and r["lmbda"] == lam]
            g = [r["bpp"] for r in cost if r["arm"] == "sigmoid" and r["lmbda"] == lam]
            rdiff.append(abs(np.mean(g) / np.mean(s) - 1) * 100)
        V["sigRateDiff"] = f"{np.ceil(max(rdiff)):.0f}"
        # PSNR of the sigmoid against the shipped clamp at each lambda (Kodak, mean over images)
        drop = []
        for lam in LAMBDAS:
            s = [r["psnr"] for r in cost if r["arm"] == "stock" and r["rep"] == 1 and r["lmbda"] == lam]
            g = [r["psnr"] for r in cost if r["arm"] == "sigmoid" and r["lmbda"] == lam]
            drop.append(float(np.mean(s) - np.mean(g)))
        V["sigPsnrDropLow"], V["sigPsnrDropHigh"] = f"{min(drop):.1f}", f"{max(drop):.1f}"
    else:
        for k in ("sigCollapsed", "initCollapsed", "sigRecovered", "initRecovered", "sigAllFive",
                  "sigPartList", "initCollapsedImage", "initCollapsedMargin", "stockFiveRecovered",
                  "sigPsnrDropLow", "sigPsnrDropHigh"):
            pending(k)
    if done["t1"]:
        def rate(im):
            rs = [r for r in t1 if r["image"] == im]
            return sum(r["collapsed"] for r in rs), len(rs)
        k, n = rate("full0138")
        V["tOneFullCollapsed"], V["tOneFullEncodes"] = str(k), str(n)
        k, n = rate("full0502")
        V["tOneFullFiveCollapsed"], V["tOneFullFiveEncodes"] = str(k), str(n)
        old = [r for r in vp.stock_pool() if r.get("image") == "native0138"]
        k, n = rate("native0138")
        V["tOneNativeCollapsed"] = str(k + sum(r["_collapsed"] for r in old))
        V["tOneNativeEncodes"] = str(n + len(old))
    else:
        for k in ("tOneFullCollapsed", "tOneFullEncodes", "tOneFullFiveCollapsed", "tOneFullFiveEncodes",
                  "tOneNativeCollapsed", "tOneNativeEncodes"):
            pending(k)
    if done["noise"]:
        # stock run 2 against stock run 1, images where neither run collapsed
        one = [r for r in jsonl("screenshots_rd.jsonl") if r["arm"] == "stock"]
        a, b = defaultdict(dict), defaultdict(dict)
        for r in one:
            a[r["image"]][r["lmbda"]] = r
        for r in noise:
            b[r["image"]][r["lmbda"]] = r
        ok = [im for im in a if not any(r["collapsed"] for r in list(a[im].values()) + list(b[im].values()))]
        v = [bd_rate({l: (r["bpp"], r["psnr"]) for l, r in a[im].items()},
                     {l: (r["bpp"], r["psnr"]) for l, r in b[im].items()}) for im in ok]
        lo, hi = t_interval(v)
        V["screenNoiseMean"], V["screenNoiseCI"] = pct(float(np.mean(v))), ci_text(lo, hi)
        V["screenNoiseImages"] = str(len(ok))
        V["screenNoiseCollapsed"] = str(sum(r["collapsed"] for r in noise))
        # straight-through clamp against each stock run on the same images
        st = defaultdict(dict)
        for r in jsonl("screenshots_rd.jsonl"):
            if r["arm"] == "ste":
                st[r["image"]][r["lmbda"]] = r
        assert not [r for d in st.values() for r in d.values() if r["collapsed"]]
        for key, ref in (("steVsOne", a), ("steVsTwo", b)):
            v = [bd_rate({l: (r["bpp"], r["psnr"]) for l, r in ref[im].items()},
                         {l: (r["bpp"], r["psnr"]) for l, r in st[im].items()}) for im in ok]
            lo, hi = t_interval(v)
            V[f"{key}Mean"], V[f"{key}CI"] = pct(float(np.mean(v))), ci_text(lo, hi)
    else:
        for k in ("screenNoiseMean", "screenNoiseCI", "screenNoiseImages", "screenNoiseCollapsed",
                  "steVsOneMean", "steVsOneCI", "steVsTwoMean", "steVsTwoCI"):
            pending(k)
    rates_table(t1, dark, done)
    remedies_table(rem, cost, done)


COLLAPSE6 = ("div127", "code01", "text01", "real09", "real11", "full0138")
CONTENT = {"native0138": "photo, native", "full0138": "photo", "full0502": "photo",
           "div127": "sky crop", "code01": "synthetic", "text01": "synthetic", "text04": "synthetic",
           "real09": "screenshot", "real11": "screenshot"}
REL = {"native0138": "natural_bright/native0138.png", "full0138": "natural_bright/full0138.png",
       "full0502": "natural_bright/full0502.png", "div127": "natural_bright/div127.png",
       "code01": "screen/code01.png", "text01": "screen/text01.png", "text04": "screen/text04.png",
       "real09": "screen_real/real09.png", "real11": "screen_real/real11.png"}


def rates_table(t1: list[dict], dark: list[dict], done: dict) -> None:
    def cell(rows):
        if not rows:
            return r"\pending{} & & "
        k, n = sum(r["collapsed"] for r in rows), len(rows)
        lo, hi = cp(k, n)
        return f"{k}/{n} & [{lo:.0f}, {hi:.0f}]"
    old_native = [dict(collapsed=r["_collapsed"]) for r in vp.stock_pool() if r.get("image") == "native0138"]
    lines = []
    for im in ("native0138", "full0138", "full0502", "div127", "code01", "text01", "text04", "real09", "real11"):
        m, s = img_stats(REL[im])
        b = [r for r in t1 if r["image"] == im] if done["t1"] else []
        if im == "native0138" and b:
            b = b + old_native
        if im == "native0138":
            dcell = "--- & ---"
            dm = ""
        else:
            d = [r for r in dark if r["image"] == f"{im}_inv"] if done["dark"] else []
            dm = f"{255 - m:.1f}"
            dcell = cell(d) if d else r"\pending{} & "
        lines.append(f"\\texttt{{{im}}} & {m:.1f} & {s:.1f} & {cell(b) if b else chr(92)+'pending{} & '} "
                     f"& {dm} & {dcell} \\\\")
    kod = [r for r in t1 if r["image"].startswith("kodim")] if done["t1"] else []
    kcell = cell(kod) if kod else r"\pending{} & "
    lines.append("\\midrule")
    lines.append(f"24 Kodak images & \\multicolumn{{2}}{{c}}{{\\V{{kodakMinMean}}--\\V{{kodakMaxMean}}}} & {kcell} & & & \\\\")
    (OUT / "tab_rates.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{\\label{tab:rates}%\n"
        "Collapse rate of stock Cool-chic 5.0.1 per encode at $\\lambda = 0.0006$ and 2000 iterations,\n"
        "for bright images and for the same images inverted ($255 - x$). Within a row the encodes of\n"
        "one file use the same settings; their outcomes differ because the optimization is not\n"
        "seeded. Mean and std are RGB statistics of the source; brackets give the 95\\%\n"
        "Clopper--Pearson interval (\\%). Eight of the ten \\texttt{native0138} encodes ran in an\n"
        "earlier batch whose GPU was not recorded; the Kodak row has two encodes per image.}\n"
        "\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrr@{\\hspace{10pt}}rrr}\n\\toprule\n"
        " & \\multicolumn{4}{c}{bright source} & \\multicolumn{3}{c}{inverted} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}\n"
        "image & mean & std & collapsed & 95\\% CI & mean & collapsed & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


ARMS = (("stock", "Clamp (shipped)"), ("stock2", "Clamp, second encode"),
        ("ste", "Straight-through clamp"), ("noclamp", "No clamp"),
        ("leaky", "Leaky clamp"), ("sigmoid", "Sigmoid"), ("init", "Clamp, mean initialization"))


def remedies_table(rem: list[dict], cost: list[dict], done: dict) -> None:
    t1 = jsonl("collapse_rates.jsonl")
    coll = ("div127", "code01", "text01", "real09", "real11", "full0138")
    lines = []
    for arm, label in ARMS:
        src = [r for r in t1 if r["image"] in coll and r["rep"] <= 5] if arm == "stock" else \
              [r for r in rem if r["arm"] == arm]
        if arm == "stock2":
            c1 = "--- & "
        elif done["remedy"] and done["t1"] and src:
            k = sum(not r["collapsed"] for r in src)
            med = float(np.median([r["margin"] for r in src]))
            c1 = f"{k}/{len(src)} & {'$0.0$' if round(med, 1) == 0 else f'${med:+.1f}$'}"
        else:
            c1 = r"\pending{} & "
        if arm == "stock":
            c2 = "0 (anchor) & "
        elif arm == "init":
            c2 = "--- & "
        elif done["cost"]:
            # Kodak never collapses; enforce it rather than assume it (Section 4 exclusion rule)
            assert not [r for r in cost if r["collapsed"]], "a Kodak cost encode collapsed"
            by = defaultdict(dict)
            for r in cost:
                key = "stock2" if r["arm"] == "stock" and r["rep"] == 2 else r["arm"]
                by[(r["image"], key)][r["lmbda"]] = (r["bpp"], r["psnr"])
            ims = sorted({r["image"] for r in cost})
            v = [bd_rate(by[(im, "stock")], by[(im, arm)]) for im in ims]
            lo, hi = t_interval(v)
            V[f"{arm}KodakBd"] = f"{np.mean(v):+.2f}"
            c2 = f"${np.mean(v):+.2f}$ & $[{lo:+.2f}, {hi:+.2f}]$"
        else:
            c2 = r"\pending{} & "
        lines.append(f"{label} & {c1} & {c2} \\\\")
    (OUT / "tab_remedies.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{\\label{tab:remedies}%\n"
        "Output mappings of Section~\\ref{sec:mappings} in Cool-chic 5.0.1, 2000 iterations. Left:\n"
        "encodes that did not collapse out of 30 (six collapsing images, five encodes each,\n"
        "$\\lambda = 0.0006$) and their median margin (dB). Right: BD-rate (\\%) against the shipped\n"
        "clamp on eight Kodak images (\\texttt{kodim02}, \\texttt{kodim05}, \\ldots, \\texttt{kodim23})\n"
        "at four rates, mean over images and 95\\% $t$-interval; the second encode of the shipped clamp\n"
        "gives the run-to-run variation. The mean initialization changes only the start of the\n"
        "optimization and was not run on Kodak.}\n"
        "\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{six collapsing images} & \\multicolumn{2}{c}{eight Kodak images} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "mapping & not collapsed & median margin & BD-rate & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    F = facts()
    if vp.FAILURES:
        raise SystemExit(f"verify_paper assertions failed: {vp.FAILURES}")
    shared_facts(F)
    pool_numbers(extended_pool())
    images()
    metrics_table()
    rd_numbers()
    later_numbers()
    lines = ["% Generated by src/journal_numbers.py -- do not edit."]
    for k in sorted(V):
        lines.append(f"\\expandafter\\def\\csname V@{k}\\endcsname{{{V[k]}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(V)} numbers written to numbers/numbers.tex; tables in numbers/tab_*.tex")


if __name__ == "__main__":
    main()
