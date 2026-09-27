"""Across-seed confidence intervals for a handful of independent training runs.

Each seed contributes one point estimate, and the interval is taken across those
estimates with Student's t, so it reflects run-to-run training variance rather
than the sampling noise within one checkpoint.
"""

from __future__ import annotations

import math

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}


def t_critical(df: int) -> float:
    if df in _T95:
        return _T95[df]
    tabulated = sorted(_T95)
    if df < tabulated[0]:
        return _T95[tabulated[0]]
    if df > tabulated[-1]:
        return 1.96
    return next(_T95[k] for k in tabulated if k >= df)


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width across seeds; the half-width is nan for a single seed."""
    k = len(values)
    mean = sum(values) / k
    if k < 2:
        return mean, float("nan")
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (k - 1))
    return mean, t_critical(k - 1) * sd / math.sqrt(k)


def format_mean_ci(mean: float, half_width: float, decimals: int = 3) -> str:
    if math.isnan(half_width):
        return f"{mean:.{decimals}f} (K=1)"
    return f"{mean:.{decimals}f}±{half_width:.{decimals}f}"
