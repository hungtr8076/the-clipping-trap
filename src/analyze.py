"""BD-rate (Bjontegaard delta rate), used by verify_paper.py for the cost of the fix.

BD-rate = percentage of bitrate saved against a reference codec at equal quality.
Negative = better. It is the standard measure in the compression field.
"""

from __future__ import annotations

import numpy as np


def bd_rate(r1, p1, r2, p2) -> float:
    """BD-rate of curve 2 against curve 1, per Bjontegaard (cubic interpolation on log-rate).

    Returns the % change in bitrate. Negative = curve 2 is better.
    """
    r1, p1 = np.asarray(r1, float), np.asarray(p1, float)
    r2, p2 = np.asarray(r2, float), np.asarray(p2, float)
    o1, o2 = np.argsort(p1), np.argsort(p2)
    p1, r1 = p1[o1], r1[o1]
    p2, r2 = p2[o2], r2[o2]
    l1, l2 = np.log10(r1), np.log10(r2)

    lo = max(p1.min(), p2.min())
    hi = min(p1.max(), p2.max())
    if hi - lo < 1e-6:
        return float("nan")

    # cubic fit: log(rate) as a function of PSNR
    c1 = np.polyfit(p1, l1, min(3, len(p1) - 1))
    c2 = np.polyfit(p2, l2, min(3, len(p2) - 1))
    i1 = np.polyval(np.polyint(c1), hi) - np.polyval(np.polyint(c1), lo)
    i2 = np.polyval(np.polyint(c2), hi) - np.polyval(np.polyint(c2), lo)
    return float((10 ** ((i2 - i1) / (hi - lo)) - 1) * 100)
