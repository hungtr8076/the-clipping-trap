"""Generate a controlled screen-content set at Kodak dimensions (768x512).

Why generate instead of downloading SIQAD:
  SIQAD has no stable direct download link. Generating them means (a) no external
  dependency, (b) full reproducibility, (c) control over the content axis - we know
  exactly which category each image belongs to, so results can be broken down per
  category instead of only in aggregate.

Four categories, 6 images each -> 24 images, matching the Kodak count:
  text      plain text at several font sizes
  code      source code with syntax highlighting
  chart     plots, tables, vector graphics
  mixed     a UI: text + flat color blocks + rules + a small photo
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 768, 512
OUT = Path(__file__).resolve().parent.parent / "data" / "screen"

FONTS = {
    "sans": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "serif": "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "mono": "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
}

LOREM = (
    "the quick brown fox jumps over the lazy dog while distant thunder rolls across "
    "the valley and the river carries silt toward the delta where fishermen mend nets "
    "under a pale sky that promises neither rain nor sun but only the slow patience of "
    "afternoon light on water and stone and the worn planks of an old wooden pier "
    "extending past the reeds into shallow brackish water"
).split()

CODE = '''import numpy as np
from pathlib import Path

def band_limit(stack, keep):
    """Keep only the channels listed in `keep`."""
    spec = np.fft.rfft(stack, axis=0)
    mask = np.zeros(spec.shape[0], dtype=bool)
    for h in keep:
        if h < spec.shape[0]:
            mask[h] = True
    spec[~mask] = 0
    return np.fft.irfft(spec, n=stack.shape[0], axis=0)

class Sequence:
    def __init__(self, paths, degrees):
        self.paths = tuple(paths)
        self.degrees = tuple(degrees)

    @property
    def uniform(self) -> bool:
        d = np.diff(np.array(self.degrees))
        return bool(np.ptp(d) <= 1.0)

    def stack(self, size=150):
        frames = [load(p, size) for p in self.paths]
        return np.stack(frames)          # (N, H, W)

if __name__ == "__main__":
    seqs = discover(Path("data/"))
    print(f"{len(seqs)} sequences, {sum(map(len, seqs))} images")
'''

TOKEN_COLORS = {
    "kw": (197, 60, 140), "str": (30, 130, 60), "num": (20, 90, 190),
    "cmt": (120, 125, 130), "fn": (150, 90, 20), "def": (30, 35, 42),
}
KEYWORDS = {"import", "from", "def", "class", "return", "if", "for", "in", "not",
            "None", "True", "False", "self", "as", "with", "print", "property"}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS[kind], size)


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(OUT / f"{name}.png", optimize=True)


# ─────────────────────────────────────────── text
def make_text(idx: int, rng: random.Random) -> None:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    kind = ["serif", "sans", "mono"][idx % 3]
    size = [11, 13, 16, 20][idx % 4]
    f = font(kind, size)
    fh = font(kind, size + 8)

    d.text((40, 30), "Section " + str(idx + 1) + ". Angular sampling", font=fh, fill=(15, 15, 20))
    d.line((40, 30 + size + 18, W - 40, 30 + size + 18), fill=(80, 80, 90), width=1)

    y = 30 + size + 30
    line, words = [], LOREM[:]
    rng.shuffle(words)
    wi = 0
    while y < H - 40:
        line, wpx = [], 0
        while wi < len(words):
            w = words[wi]
            adv = d.textlength(w + " ", font=f)
            if wpx + adv > W - 80:
                break
            line.append(w)
            wpx += adv
            wi += 1
        if not line:
            wi = 0
            rng.shuffle(words)
            continue
        d.text((40, y), " ".join(line), font=f, fill=(25, 25, 30))
        y += int(size * 1.55)
    save(img, f"text{idx+1:02d}")


# ─────────────────────────────────────────── code
def make_code(idx: int, rng: random.Random) -> None:
    dark = idx % 2 == 1
    bg = (30, 33, 39) if dark else (252, 252, 253)
    gutter = (40, 44, 52) if dark else (243, 243, 245)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    size = [12, 13, 14][idx % 3]
    f = font("mono", size)
    d.rectangle([0, 0, 46, H], fill=gutter)

    lines = CODE.splitlines()
    off = (idx * 3) % max(1, len(lines) - 4)
    y = 14
    n = 1
    for raw in (lines[off:] + lines[:off]):
        if y > H - size - 6:
            break
        d.text((10, y), f"{n:>3}", font=f, fill=(120, 125, 135))
        x = 56
        for tok in raw.replace("(", " ( ").replace(")", " ) ").split(" "):
            if not tok:
                x += d.textlength(" ", font=f)
                continue
            if tok.startswith("#"):
                c = TOKEN_COLORS["cmt"]
            elif tok.strip('."\'') in KEYWORDS:
                c = TOKEN_COLORS["kw"]
            elif tok.startswith(('"', "'")):
                c = TOKEN_COLORS["str"]
            elif tok.strip("(),.").replace(".", "").isdigit():
                c = TOKEN_COLORS["num"]
            else:
                c = (215, 218, 224) if dark else TOKEN_COLORS["def"]
            d.text((x, y), tok, font=f, fill=c)
            x += d.textlength(tok + " ", font=f)
        y += int(size * 1.5)
        n += 1
    save(img, f"code{idx+1:02d}")


# ─────────────────────────────────────────── chart
def make_chart(idx: int, rng: random.Random) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(W / 100, H / 100), dpi=100)
    x = np.linspace(0, 10, 120)
    for k, ax in enumerate(axes.ravel()):
        mode = (idx + k) % 4
        if mode == 0:
            for j in range(3):
                ax.plot(x, np.sin(x * (j + 1) * 0.6 + idx) + j, lw=1.4)
            ax.grid(alpha=.3, lw=.5)
        elif mode == 1:
            ax.bar(range(7), rng.sample(range(2, 20), 7), color="#3b6ea5", edgecolor="k", lw=.6)
            ax.grid(axis="y", alpha=.3, lw=.5)
        elif mode == 2:
            r = np.random.default_rng(idx * 10 + k)
            ax.scatter(r.normal(size=90), r.normal(size=90), s=9, c="#a63a6e", alpha=.75)
            ax.grid(alpha=.3, lw=.5)
        else:
            ax.step(range(12), rng.sample(range(1, 25), 12), where="mid", lw=1.5, color="#2c6e49")
            ax.grid(alpha=.3, lw=.5)
        ax.set_title(f"panel {k+1}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Figure {idx+1}: rate-distortion sweep", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"chart{idx+1:02d}.png", dpi=100, facecolor="white")
    plt.close(fig)
    im = Image.open(OUT / f"chart{idx+1:02d}.png").convert("RGB").resize((W, H), Image.LANCZOS)
    im.save(OUT / f"chart{idx+1:02d}.png")


# ─────────────────────────────────────────── mixed / UI
def make_mixed(idx: int, rng: random.Random) -> None:
    img = Image.new("RGB", (W, H), (246, 247, 249))
    d = ImageDraw.Draw(img)
    f_s, f_m, f_b = font("sans", 12), font("sans", 14), font("sans", 18)

    d.rectangle([0, 0, W, 48], fill=(38, 46, 60))
    d.text((18, 15), "Dashboard  ›  Experiments  ›  run " + str(1000 + idx),
           font=f_m, fill=(238, 240, 244))
    d.rectangle([0, 48, 176, H], fill=(255, 255, 255))
    d.line((176, 48, 176, H), fill=(222, 226, 232), width=1)
    for i, item in enumerate(["Overview", "Datasets", "Codecs", "Results", "Logs", "Settings"]):
        yy = 70 + i * 34
        if i == idx % 6:
            d.rectangle([8, yy - 8, 168, yy + 20], fill=(232, 238, 248))
        d.text((22, yy), item, font=f_m, fill=(40, 46, 56))

    for c in range(3):
        x0 = 196 + c * 186
        d.rectangle([x0, 70, x0 + 166, 152], fill=(255, 255, 255), outline=(224, 228, 234))
        d.text((x0 + 14, 84), ["bitrate", "psnr", "encode"][c], font=f_s, fill=(120, 126, 136))
        d.text((x0 + 14, 104), [f"{0.12*(idx+1):.2f} bpp", f"{30+idx:.1f} dB", f"{18+idx*3} s"][c],
               font=f_b, fill=(24, 28, 36))

    d.rectangle([196, 172, W - 24, H - 24], fill=(255, 255, 255), outline=(224, 228, 234))
    cols = ["image", "codec", "bpp", "psnr", "ssim"]
    for j, cname in enumerate(cols):
        d.text((212 + j * 104, 186), cname, font=f_s, fill=(120, 126, 136))
    d.line((204, 204, W - 36, 204), fill=(232, 236, 240), width=1)
    rows = ["kodim01", "kodim07", "text03", "code02", "chart05", "mixed01", "kodim19"]
    for r, name in enumerate(rows):
        yy = 214 + r * 26
        if yy > H - 44:
            break
        if r % 2:
            d.rectangle([204, yy - 5, W - 36, yy + 19], fill=(250, 251, 252))
        vals = [name, ["cool-chic", "avif", "webp", "jxl"][r % 4],
                f"{0.08 + r * 0.07:.3f}", f"{28.4 + r * 1.3:.2f}", f"{0.90 + r * 0.012:.3f}"]
        for j, v in enumerate(vals):
            d.text((212 + j * 104, yy), v, font=f_s, fill=(34, 40, 50))

    # thumbnail slot: a patch of natural image, but inside a UI frame
    thumb = np.zeros((60, 90, 3), np.uint8)
    yy, xx = np.mgrid[0:60, 0:90]
    thumb[..., 0] = (128 + 90 * np.sin(xx / 9 + idx)).clip(0, 255)
    thumb[..., 1] = (128 + 90 * np.sin(yy / 7 + idx * 2)).clip(0, 255)
    thumb[..., 2] = (128 + 90 * np.cos((xx + yy) / 11)).clip(0, 255)
    img.paste(Image.fromarray(thumb), (W - 130, 78))
    save(img, f"mixed{idx+1:02d}")


def main() -> None:
    rng = random.Random(7)
    for i in range(6):
        make_text(i, rng)
        make_code(i, rng)
        make_chart(i, rng)
        make_mixed(i, rng)
    files = sorted(OUT.glob("*.png"))
    tot = sum(f.stat().st_size for f in files)
    print(f"generated {len(files)} screen-content images -> {OUT}")
    print(f"total {tot/1e6:.1f} MB")
    for f in files[:4]:
        print("  ", f.name, Image.open(f).size)


if __name__ == "__main__":
    main()
