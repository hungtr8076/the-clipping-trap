"""Demonstrate the gradient trap on C3's OWN Synthesis structure (DeepMind).

Reading the source showed that both Cool-chic and C3 put the output clamp INSIDE the
per-image optimization loop:

    Cool-chic  frame.py::forward()      torch.clamp(decoded_image, 0., 1.)
    C3         synthesis.py::__call__() jnp.clip(self._net(latents), 0., 1.)
               clip_range = (0.0, 1.0) in all three official configurations

This script does not run the whole codec and does not import C3: it builds a minimal network
with the structure of C3's Synthesis layer (convolutions followed by jnp.clip), optimizes the
latents to fit a target image, and measures whether the gradient dies when the image sits
near the edge of the dynamic range.

Two targets, identical content, differing only in brightness:
    bright: mean 0.97  (like code01, the image that collapsed in the Cool-chic experiments)
    middle: mean 0.50  (control group)

If the bright one gives gradient ~0 with a frozen loss while the middle one learns
normally, the mechanism is confirmed on a SECOND implementation, independent of
Cool-chic.
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

H, W = 64, 64
STEPS = 400


def target_image(mean: float, seed: int = 0) -> jnp.ndarray:
    """Synthetic "screen" content: flat background + thin glyph strokes."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), mean, np.float32)
    for _ in range(60):                        # glyph strokes
        y, x = rng.integers(2, H - 6), rng.integers(2, W - 10)
        img[y:y + 2, x:x + rng.integers(4, 9)] = mean - 0.75
    return jnp.asarray(np.clip(img, 0.0, 1.0))


def build(clip: bool):
    """Minimal synthesis network mirroring C3's structure: conv then clip."""
    def fn(latents):
        net = hk.Sequential([
            hk.Conv2D(12, 3), jax.nn.relu,
            hk.Conv2D(12, 3), jax.nn.relu,
            hk.Conv2D(3, 3),
        ])
        out = net(latents)
        # <- THIS LINE IS THE ONLY DIFFERENCE. C3: jnp.clip(self._net(latents), 0, 1)
        return jnp.clip(out, 0.0, 1.0) if clip else out
    return hk.without_apply_rng(hk.transform(fn))


def run(mean: float, clip: bool, seed: int = 0) -> dict:
    tgt = target_image(mean, seed)
    net = build(clip)
    key = jax.random.PRNGKey(seed)
    # latents initialized with an upward bias, mimicking the state the optimizer can fall into
    lat = jnp.full((1, H, W, 4), 0.6)
    params = net.init(key, lat)
    opt = optax.adam(1e-2)
    state = opt.init((params, lat))

    def loss_fn(pl, tgt):
        p, l = pl
        return jnp.mean((net.apply(p, l)[0] - tgt) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    losses, gnorms = [], []
    pl = (params, lat)
    for _ in range(STEPS):
        loss, g = grad_fn(pl, tgt)
        gn = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g))))
        losses.append(float(loss))
        gnorms.append(gn)
        upd, state = opt.update(g, state, pl)
        pl = optax.apply_updates(pl, upd)

    rec = net.apply(pl[0], pl[1])[0]
    mse = float(jnp.mean((jnp.clip(rec, 0, 1) - tgt) ** 2))
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
    flat = float(jnp.mean((tgt - tgt.mean()) ** 2))
    flat_psnr = 10 * np.log10(1.0 / flat) if flat > 0 else 99.0
    return dict(mean=mean, clip=clip, loss0=losses[0], loss_end=losses[-1],
                drop=100 * (1 - losses[-1] / losses[0]),
                gnorm0=gnorms[0], gnorm_end=gnorms[-1],
                psnr=psnr, flat_psnr=flat_psnr, margin=psnr - flat_psnr)


def main() -> None:
    print("GRADIENT TRAP - on C3's synthesis structure (clip inside the optimization loop)")
    print(f"  {H}x{W} image, {STEPS} steps, latents initialized biased upward (0.6)\n")
    print(f"  {'brightness':>11}{'clip?':>7}{'loss 1st':>11}{'loss last':>11}"
          f"{'drop':>8}{'final |grad|':>14}{'margin':>10}")
    print("  " + "-" * 70)
    rows = []
    for mean in (0.97, 0.50):
        for clip in (True, False):
            r = run(mean, clip)
            rows.append(r)
            print(f"  {r['mean']:>11.2f}{'yes' if clip else 'no':>7}"
                  f"{r['loss0']:>11.5f}{r['loss_end']:>11.5f}"
                  f"{r['drop']:>7.1f}%{r['gnorm_end']:>14.2e}"
                  f"{r['margin']:>+10.2f}")

    print("\n  HOW TO READ THIS - the criteria are GRADIENT and MARGIN, not the loss drop")
    print("  (the loss can fall 86% and then die, so the drop is not a valid signal)")
    for r in rows:
        tag = "COLLAPSED" if r["margin"] < 5 else "ok"
        dead = "gradient DEAD" if r["gnorm_end"] == 0.0 else ""
        print(f"    brightness={r['mean']:.2f}  clip={'yes' if r['clip'] else 'no':<4}"
              f"  margin={r['margin']:+7.2f} dB  {tag:<10} {dead}")

    bc = [r for r in rows if r["mean"] == 0.97 and r["clip"]][0]
    bn = [r for r in rows if r["mean"] == 0.97 and not r["clip"]][0]
    mc = [r for r in rows if r["mean"] == 0.50 and r["clip"]][0]
    ok = bc["margin"] < 5 and bn["margin"] > 5 and mc["margin"] > 5
    print()
    if ok:
        print("  -> MECHANISM CONFIRMED. Collapse needs BOTH conditions at once:")
        print("      (a) content near the edge of the dynamic range, AND")
        print("      (b) the clamp sitting inside the gradient path")
        print("    Remove either condition and it learns normally.")
        if bc["gnorm_end"] == 0.0:
            print("    The gradient in the collapsed cell is exactly 0.00e+00 - not small, ZERO.")
    else:
        print("  -> not reproduced at this scale")


if __name__ == "__main__":
    main()
