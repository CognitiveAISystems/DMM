"""Across-seed statistics shared by the corridor figure and table scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}


def t_critical(df: int) -> float:
    if df in _T95:
        return _T95[df]
    for k in sorted(_T95):
        if k >= df:
            return _T95[k]
    return 1.96


def mean_ci(values: list[float]) -> tuple[float, float]:
    k = len(values)
    mean = sum(values) / k
    if k < 2:
        return mean, float("nan")
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (k - 1))
    return mean, t_critical(k - 1) * sd / math.sqrt(k)


def load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _per_seed(entry: dict, action: str) -> list[float]:
    if entry["k_seeds"] != 5:
        raise ValueError(f"expected 5-seed data, got k_seeds={entry['k_seeds']}")
    return entry["rows"][action]["per_seed_freq"]


def p_valid_mean_ci(entry: dict) -> tuple[float, float]:
    """Sum the two valid actions per seed first, then take the interval across seeds."""
    wl = _per_seed(entry, "(wait, left)")
    rw = _per_seed(entry, "(right, wait)")
    return mean_ci([a + b for a, b in zip(wl, rw)])


def invalid_mean_ci(entry: dict) -> tuple[float, float]:
    ww = _per_seed(entry, "(wait, wait)")
    rl = _per_seed(entry, "(right, left)")
    return mean_ci([a + b for a, b in zip(ww, rl)])


def action_mean_ci(entry: dict, action: str) -> tuple[float, float]:
    return mean_ci(_per_seed(entry, action))


def canonical(data_dir: Path) -> dict:
    """The reference estimate of the standard beta_0=beta_r=0.8, K_train=K_test=4 run.

    Every appearance of that configuration resolves here, so the figure, both
    tables and Figure 1 report one number for it rather than several.
    """
    return load(data_dir / "round_generalization_matrix.json")["results"]["train=4/test=4"]
