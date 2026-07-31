"""Notebook-friendly feature and representation metrics.

All public functions accept NumPy arrays. Representation functions additionally
accept SciPy sparse matrices so TopK SAE latents need not be densified.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, svds
from scipy.stats import pearsonr, spearmanr

from .utils import monitored_operation

Array: TypeAlias = np.ndarray | sparse.spmatrix
LOGGER = logging.getLogger(__name__)


def _column(matrix: Array, index: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.getcol(index).toarray()).ravel()
    return np.asarray(matrix[:, index]).ravel()


def _safe_correlation(
    left: np.ndarray, right: np.ndarray, method: Literal["pearson", "spearman"]
) -> tuple[float, str | None]:
    if len(left) < 2:
        return np.nan, "fewer_than_two_observations"
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan, "zero_variance"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = (
            pearsonr(left, right).statistic
            if method == "pearson"
            else spearmanr(left, right).statistic
        )
    return float(value), None if np.isfinite(value) else "numerical_failure"


def activation_overlap(
    activations_a: Array,
    activations_b: Array,
    feature_a: int,
    feature_b: int,
    *,
    threshold_a: float = 0.0,
    threshold_b: float = 0.0,
    sequence_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare a matched feature pair over shared token rows.

    Args:
        activations_a: Post-nonlinearity latents ``[tokens, d_sae_a]``.
        activations_b: Post-nonlinearity latents ``[tokens, d_sae_b]``.
        sequence_ids: Optional shared sequence identifier for every token row.

    Empty supports produce NaN set metrics and an explicit ``empty_reason``.
    """
    if activations_a.shape[0] != activations_b.shape[0]:
        raise ValueError("Activation matrices must contain identical token rows")
    left = _column(activations_a, feature_a).astype(np.float64, copy=False)
    right = _column(activations_b, feature_b).astype(np.float64, copy=False)
    active_a = left > threshold_a
    active_b = right > threshold_b
    count_a = int(active_a.sum())
    count_b = int(active_b.sum())
    intersection = active_a & active_b
    union = active_a | active_b
    count_intersection = int(intersection.sum())
    count_union = int(union.sum())
    empty_reason: str | None = None
    if count_a == 0 and count_b == 0:
        empty_reason = "both_active_sets_empty"
    elif count_a == 0:
        empty_reason = "feature_a_active_set_empty"
    elif count_b == 0:
        empty_reason = "feature_b_active_set_empty"

    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else np.nan

    pearson_all, pearson_all_reason = _safe_correlation(left, right, "pearson")
    spearman_all, spearman_all_reason = _safe_correlation(left, right, "spearman")
    pearson_union, pearson_union_reason = _safe_correlation(
        left[union], right[union], "pearson"
    )
    spearman_union, spearman_union_reason = _safe_correlation(
        left[union], right[union], "spearman"
    )
    pearson_intersection, pearson_intersection_reason = _safe_correlation(
        left[intersection], right[intersection], "pearson"
    )
    spearman_intersection, spearman_intersection_reason = _safe_correlation(
        left[intersection], right[intersection], "spearman"
    )
    weighted_denominator = np.maximum(left, right).sum()

    result: dict[str, Any] = {
        "feature_a": int(feature_a),
        "feature_b": int(feature_b),
        "n_tokens": int(len(left)),
        "active_count_a": count_a,
        "active_count_b": count_b,
        "activation_frequency_a": divide(count_a, len(left)),
        "activation_frequency_b": divide(count_b, len(left)),
        "intersection_count": count_intersection,
        "union_count": count_union,
        "jaccard": divide(count_intersection, count_union),
        "overlap_coefficient": divide(count_intersection, min(count_a, count_b)),
        "p_b_active_given_a": divide(count_intersection, count_a),
        "p_a_active_given_b": divide(count_intersection, count_b),
        "precision_a_to_b": divide(count_intersection, count_a),
        "recall_a_to_b": divide(count_intersection, count_b),
        "precision_b_to_a": divide(count_intersection, count_b),
        "recall_b_to_a": divide(count_intersection, count_a),
        "weighted_jaccard": divide(np.minimum(left, right).sum(), weighted_denominator),
        "pearson_all": pearson_all,
        "spearman_all": spearman_all,
        "pearson_union": pearson_union,
        "spearman_union": spearman_union,
        "pearson_intersection": pearson_intersection,
        "spearman_intersection": spearman_intersection,
        "empty_reason": empty_reason,
        "pearson_all_reason": pearson_all_reason,
        "spearman_all_reason": spearman_all_reason,
        "pearson_union_reason": pearson_union_reason,
        "spearman_union_reason": spearman_union_reason,
        "pearson_intersection_reason": pearson_intersection_reason,
        "spearman_intersection_reason": spearman_intersection_reason,
    }
    if sequence_ids is not None:
        sequence_ids = np.asarray(sequence_ids)
        if len(sequence_ids) != len(left):
            raise ValueError("sequence_ids length must equal the number of token rows")
        seq_a = set(sequence_ids[active_a].tolist())
        seq_b = set(sequence_ids[active_b].tolist())
        seq_intersection = len(seq_a & seq_b)
        seq_union = len(seq_a | seq_b)
        result.update(
            sequence_active_count_a=len(seq_a),
            sequence_active_count_b=len(seq_b),
            sequence_jaccard=divide(seq_intersection, seq_union),
            sequence_overlap_coefficient=divide(
                seq_intersection, min(len(seq_a), len(seq_b))
            ),
        )
    return result


def _scale_columns(matrix: Array, scales: np.ndarray) -> Array:
    if sparse.issparse(matrix):
        return matrix @ sparse.diags(scales)
    return np.asarray(matrix) * scales


def _centered_cross_frobenius_squared(left: Array, right: Array) -> float:
    """Compute ``||X.T H Y||_F^2`` without materializing centered matrices."""
    n_rows = left.shape[0]
    cross = left.T @ right
    sum_left = np.asarray(left.sum(axis=0)).ravel().astype(np.float64)
    sum_right = np.asarray(right.sum(axis=0)).ravel().astype(np.float64)
    if sparse.issparse(cross):
        raw_squared = float(
            np.dot(cross.data.astype(np.float64), cross.data.astype(np.float64))
        )
        cross_sum_product = float(sum_left @ np.asarray(cross @ sum_right).ravel())
    else:
        cross = np.asarray(cross, dtype=np.float64)
        raw_squared = float(np.square(cross).sum())
        cross_sum_product = float(sum_left @ cross @ sum_right)
    mean_outer_squared = float(
        np.dot(sum_left, sum_left) * np.dot(sum_right, sum_right) / n_rows**2
    )
    value = raw_squared - (2.0 / n_rows) * cross_sum_product + mean_outer_squared
    return max(value, 0.0)


def linear_cka(
    left: Array,
    right: Array,
    *,
    center: bool = True,
    standardize_features: bool = False,
    progress_label: str | None = None,
) -> float:
    """Linear CKA for shared rows in ``[samples, features]`` matrices.

    The float64 covariance-statistic computation works with different widths and
    sparse matrices. Feature standardization changes the estimand by giving rare,
    low-variance features equal scale; raw activation CKA is the default.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must be 2D matrices with identical sample rows")
    if left.shape[0] < 2:
        raise ValueError("CKA requires at least two samples")
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    if standardize_features:

        def inverse_std(matrix: Array) -> np.ndarray:
            mean = np.asarray(matrix.mean(axis=0)).ravel()
            second = (
                np.asarray(matrix.power(2).mean(axis=0)).ravel()
                if sparse.issparse(matrix)
                else np.mean(np.square(matrix), axis=0)
            )
            variance = np.maximum(second - mean**2, 0.0)
            scale = np.zeros_like(variance)
            valid = variance > np.finfo(np.float64).eps
            scale[valid] = 1.0 / np.sqrt(variance[valid])
            return scale

        left = _scale_columns(left, inverse_std(left))
        right = _scale_columns(right, inverse_std(right))
    label = progress_label or (
        "standardized linear CKA" if standardize_features else "linear CKA"
    )
    if not center:
        with monitored_operation(f"{label}: cross covariance (1/3)"):
            cross = left.T @ right
        with monitored_operation(f"{label}: left auto-covariance (2/3)"):
            auto_left = left.T @ left
        with monitored_operation(f"{label}: right auto-covariance (3/3)"):
            auto_right = right.T @ right

        def norm(value: Any) -> float:
            return (
                float(np.dot(value.data, value.data))
                if sparse.issparse(value)
                else float(np.square(value).sum())
            )

        numerator = norm(cross)
        denominator = np.sqrt(norm(auto_left) * norm(auto_right))
    else:
        with monitored_operation(f"{label}: centered cross covariance (1/3)"):
            numerator = _centered_cross_frobenius_squared(left, right)
        with monitored_operation(f"{label}: left auto-covariance (2/3)"):
            left_auto = _centered_cross_frobenius_squared(left, left)
        with monitored_operation(f"{label}: right auto-covariance (3/3)"):
            right_auto = _centered_cross_frobenius_squared(right, right)
        denominator = np.sqrt(left_auto * right_auto)
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("CKA is undefined for a zero-variance representation")
    return float(np.clip(numerator / denominator, 0.0, 1.0 + 1e-10))


@dataclass
class PCAResult:
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    retained_components: int
    explained_variance_retained: float
    target_reached: bool


def _centered_operator(matrix: sparse.spmatrix) -> LinearOperator:
    matrix = matrix.astype(np.float64).tocsr()
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    n_rows, n_columns = matrix.shape

    def matvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ vector).ravel() - float(mean @ vector)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix.T @ vector).ravel() - mean * vector.sum()

    def matmat(values: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ values) - np.outer(np.ones(n_rows), mean @ values)

    def rmatmat(values: np.ndarray) -> np.ndarray:
        return np.asarray(matrix.T @ values) - np.outer(mean, values.sum(axis=0))

    return LinearOperator(
        shape=(n_rows, n_columns),
        dtype=np.dtype(np.float64),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
    )


def _pca_reduce(
    matrix: Array,
    explained_variance: float,
    max_components: int,
    random_seed: int,
) -> PCAResult:
    """PCA-reduce ``[samples, features]`` with exact or sparse centered SVD."""
    n_rows, n_columns = matrix.shape
    maximum_rank = min(n_rows - 1, n_columns)
    if maximum_rank < 1:
        raise ValueError("PCA requires at least two samples and one feature")
    requested = min(max_components, maximum_rank)
    if sparse.issparse(matrix):
        matrix64 = matrix.astype(np.float64)
        mean = np.asarray(matrix64.mean(axis=0)).ravel()
        total_variance = float(matrix64.power(2).sum() - n_rows * np.dot(mean, mean))
        if requested >= maximum_rank and n_rows * n_columns <= 20_000_000:
            centered = matrix64.toarray() - mean
            u, singular, _ = np.linalg.svd(centered, full_matrices=False)
            singular = singular[:requested]
            u = u[:, :requested]
        else:
            operator = _centered_operator(matrix64)
            v0 = np.random.default_rng(random_seed).normal(size=min(operator.shape))
            u, singular, _ = svds(operator, k=requested, which="LM", v0=v0)
            order = np.argsort(singular)[::-1]
            u, singular = u[:, order], singular[order]
    else:
        dense = np.asarray(matrix, dtype=np.float64)
        centered = dense - dense.mean(axis=0, keepdims=True)
        total_variance = float(np.square(centered).sum())
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        u, singular = u[:, :requested], singular[:requested]
    if total_variance <= np.finfo(np.float64).eps:
        raise ValueError("PCA is undefined for a zero-variance representation")
    ratios = np.square(singular) / total_variance
    cumulative = np.cumsum(ratios)
    reached = bool(cumulative[-1] >= explained_variance - 1e-12)
    retained = (
        min(int(np.searchsorted(cumulative, explained_variance) + 1), len(singular))
        if reached
        else len(singular)
    )
    return PCAResult(
        scores=u[:, :retained] * singular[:retained],
        explained_variance_ratio=ratios,
        retained_components=retained,
        explained_variance_retained=float(cumulative[retained - 1]),
        target_reached=reached,
    )


def _inverse_sqrt(matrix: np.ndarray, ridge: float) -> np.ndarray:
    values, vectors = eigh((matrix + matrix.T) / 2)
    cutoff = max(
        ridge,
        np.finfo(np.float64).eps * max(matrix.shape) * max(float(values.max()), 1.0),
    )
    inverse = np.zeros_like(values)
    valid = values > cutoff
    inverse[valid] = 1.0 / np.sqrt(values[valid] + ridge)
    return (vectors * inverse) @ vectors.T


def _cca(
    left: np.ndarray, right: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stable ridge CCA for centered ``[samples, components]`` scores."""
    if left.shape[0] != right.shape[0]:
        raise ValueError("CCA inputs must share sample rows")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    scale = max(left.shape[0] - 1, 1)
    covariance_left = left.T @ left / scale
    covariance_right = right.T @ right / scale
    cross = left.T @ right / scale
    whiten_left = _inverse_sqrt(covariance_left, ridge)
    whiten_right = _inverse_sqrt(covariance_right, ridge)
    u, correlations, vt = np.linalg.svd(
        whiten_left @ cross @ whiten_right, full_matrices=False
    )
    correlations = np.clip(correlations, 0.0, 1.0)
    coefficient_left = whiten_left @ u
    coefficient_right = whiten_right @ vt.T
    return correlations, coefficient_left, coefficient_right, left, right


@dataclass
class SVCCAResult:
    correlations: np.ndarray
    mean_correlation: float
    median_correlation: float
    components_a: int
    components_b: int
    explained_variance_a: float
    explained_variance_b: float
    target_reached_a: bool
    target_reached_b: bool
    pca_curve_a: np.ndarray
    pca_curve_b: np.ndarray

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_arrays:
            for key in ("correlations", "pca_curve_a", "pca_curve_b"):
                result.pop(key)
        return result


def svcca(
    left: Array,
    right: Array,
    *,
    explained_variance: float = 0.99,
    max_components: int = 1024,
    ridge: float = 1e-6,
    random_seed: int = 0,
    progress_label: str | None = None,
) -> SVCCAResult:
    """SVCCA on dominant PCA subspaces of shared ``[samples, latents]`` rows."""
    if left.shape[0] != right.shape[0]:
        raise ValueError("SVCCA inputs must share sample rows")
    label = progress_label or "SVCCA"
    with monitored_operation(f"{label}: left PCA/SVD (1/3)"):
        pca_left = _pca_reduce(left, explained_variance, max_components, random_seed)
    LOGGER.info(
        "%s left PCA retained %d components (%.4f variance; target_reached=%s)",
        label,
        pca_left.retained_components,
        pca_left.explained_variance_retained,
        pca_left.target_reached,
    )
    with monitored_operation(f"{label}: right PCA/SVD (2/3)"):
        pca_right = _pca_reduce(
            right, explained_variance, max_components, random_seed + 1
        )
    LOGGER.info(
        "%s right PCA retained %d components (%.4f variance; target_reached=%s)",
        label,
        pca_right.retained_components,
        pca_right.explained_variance_retained,
        pca_right.target_reached,
    )
    with monitored_operation(f"{label}: canonical correlation analysis (3/3)"):
        correlations, _, _, _, _ = _cca(pca_left.scores, pca_right.scores, ridge)
    return SVCCAResult(
        correlations=correlations,
        mean_correlation=float(correlations.mean()),
        median_correlation=float(np.median(correlations)),
        components_a=pca_left.retained_components,
        components_b=pca_right.retained_components,
        explained_variance_a=pca_left.explained_variance_retained,
        explained_variance_b=pca_right.explained_variance_retained,
        target_reached_a=pca_left.target_reached,
        target_reached_b=pca_right.target_reached,
        pca_curve_a=np.cumsum(pca_left.explained_variance_ratio),
        pca_curve_b=np.cumsum(pca_right.explained_variance_ratio),
    )
