"""Bootstrap intervals and matched-versus-control effect summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    samples: int


def bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int = 1000,
    confidence_level: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.nanmean,
    random_seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap over finite scalar observations."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return ConfidenceInterval(np.nan, np.nan, np.nan, samples)
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = statistic(rng.choice(values, len(values), replace=True))
    alpha = (1.0 - confidence_level) / 2.0
    return ConfidenceInterval(
        estimate=float(statistic(values)),
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1.0 - alpha)),
        samples=samples,
    )


def matched_control_statistics(
    matched: np.ndarray,
    control: np.ndarray,
    *,
    permutations: int = 10000,
    random_seed: int = 0,
) -> dict[str, float]:
    """Effect size and paired sign-flip permutation test for equal-length arrays."""
    matched = np.asarray(matched, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    valid = np.isfinite(matched) & np.isfinite(control)
    differences = matched[valid] - control[valid]
    if len(differences) == 0:
        return {
            "median_difference": np.nan,
            "standardized_effect": np.nan,
            "permutation_p": np.nan,
        }
    standard_deviation = differences.std(ddof=1) if len(differences) > 1 else 0.0
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(random_seed)
    null = np.empty(permutations)
    for index in range(permutations):
        null[index] = abs(
            float(np.mean(differences * rng.choice((-1, 1), len(differences))))
        )
    return {
        "median_difference": float(np.median(differences)),
        "standardized_effect": float(differences.mean() / standard_deviation)
        if standard_deviation > 0
        else np.nan,
        "permutation_p": float(
            (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
        ),
    }
