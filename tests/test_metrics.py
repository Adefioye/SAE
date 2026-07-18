from __future__ import annotations

import numpy as np
from scipy import sparse

from sae_seed_similarity.metrics import (
    activation_overlap,
    linear_cka,
    svcca,
)


def _representations(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(240, 12))


def _assert_global_similarity(
    left: np.ndarray, right: np.ndarray, threshold: float = 0.999
) -> None:
    assert linear_cka(left, right) > threshold
    assert (
        svcca(
            left, right, explained_variance=1.0, max_components=12, ridge=1e-10
        ).mean_correlation
        > threshold
    )


def test_identical_representations() -> None:
    values = _representations()
    _assert_global_similarity(values, values)


def test_permuted_latent_axes() -> None:
    values = _representations()
    permutation = np.random.default_rng(1).permutation(values.shape[1])
    _assert_global_similarity(values, values[:, permutation])


def test_sparse_and_dense_cka_agree() -> None:
    rng = np.random.default_rng(11)
    left = rng.binomial(1, 0.1, size=(100, 20)) * rng.random((100, 20))
    right = rng.binomial(1, 0.1, size=(100, 15)) * rng.random((100, 15))
    dense = linear_cka(left, right)
    sparse_value = linear_cka(sparse.csr_matrix(left), sparse.csr_matrix(right))
    assert abs(dense - sparse_value) < 1e-12


def test_orthogonal_rotation() -> None:
    values = _representations()
    rotation, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(12, 12)))
    _assert_global_similarity(values, values @ rotation)


def test_shared_low_rank_signal_plus_noise() -> None:
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(500, 4))
    left = signal @ rng.normal(size=(4, 18)) + 0.01 * rng.normal(size=(500, 18))
    right = signal @ rng.normal(size=(4, 15)) + 0.01 * rng.normal(size=(500, 15))
    svcca_result = svcca(
        left, right, explained_variance=0.99, max_components=15, ridge=1e-8
    )
    assert svcca_result.mean_correlation > 0.98


def test_disjoint_activation_supports() -> None:
    left = np.zeros((20, 1))
    right = np.zeros((20, 1))
    left[:10] = 1
    right[10:] = 1
    result = activation_overlap(left, right, 0, 0)
    assert result["jaccard"] == 0
    assert result["intersection_count"] == 0


def test_empty_activation_sets_report_reason() -> None:
    values = np.zeros((20, 1))
    result = activation_overlap(values, values, 0, 0)
    assert np.isnan(result["jaccard"])
    assert result["empty_reason"] == "both_active_sets_empty"
