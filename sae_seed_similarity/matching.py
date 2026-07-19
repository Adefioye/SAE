"""Batched cosine similarities and exact or sparse linear assignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from tqdm.auto import tqdm

from .adapters import SAEAdapter
from .config import MatchingConfig
from .utils import monitored_operation

LOGGER = logging.getLogger(__name__)


def _normalized(weight: Any) -> torch.Tensor:
    tensor = weight.detach().to(dtype=torch.float32)
    return torch.nn.functional.normalize(tensor, dim=1, eps=1e-12)


def _activation_correlations(
    latents_a: Any,
    latents_b: Any,
    feature_a: np.ndarray,
    feature_b: np.ndarray,
) -> np.ndarray:
    """Pearson correlations for aligned arrays of feature indices."""
    feature_a = np.asarray(feature_a, dtype=np.int64)
    feature_b = np.asarray(feature_b, dtype=np.int64)
    if feature_a.shape != feature_b.shape:
        raise ValueError("Feature index arrays must have identical shapes")
    if len(feature_a) == 0:
        return np.empty(0, dtype=np.float32)
    n_rows = latents_a.shape[0]
    sum_a = np.asarray(latents_a.sum(axis=0)).ravel().astype(np.float64)
    sum_b = np.asarray(latents_b.sum(axis=0)).ravel().astype(np.float64)
    square_a = (
        np.asarray(latents_a.power(2).sum(axis=0)).ravel()
        if sparse.issparse(latents_a)
        else np.square(np.asarray(latents_a, dtype=np.float64)).sum(axis=0)
    )
    square_b = (
        np.asarray(latents_b.power(2).sum(axis=0)).ravel()
        if sparse.issparse(latents_b)
        else np.square(np.asarray(latents_b, dtype=np.float64)).sum(axis=0)
    )
    variance_a = np.maximum(square_a - np.square(sum_a) / n_rows, 0.0)
    variance_b = np.maximum(square_b - np.square(sum_b) / n_rows, 0.0)
    if len(feature_a) <= 100_000 and (
        sparse.issparse(latents_a) or n_rows * len(feature_a) <= 20_000_000
    ):
        selected_a = latents_a[:, feature_a]
        selected_b = latents_b[:, feature_b]
        products = (
            np.asarray(selected_a.multiply(selected_b).sum(axis=0)).ravel()
            if sparse.issparse(selected_a)
            else np.sum(np.asarray(selected_a) * np.asarray(selected_b), axis=0)
        )
        numerator = products - sum_a[feature_a] * sum_b[feature_b] / n_rows
        denominator = np.sqrt(variance_a[feature_a] * variance_b[feature_b])
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > np.finfo(np.float64).eps,
        ).astype(np.float32)
    result = np.zeros(len(feature_a), dtype=np.float32)
    order = (
        np.arange(len(feature_a), dtype=np.int64)
        if len(feature_a) < 2 or np.all(feature_a[:-1] <= feature_a[1:])
        else np.argsort(feature_a, kind="stable")
    )
    sorted_left = feature_a[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_left)) + 1, len(sorted_left)]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        locations = order[start:end]
        left_index = int(sorted_left[start])
        right_indices = feature_b[locations]
        products = latents_a[:, left_index].T @ latents_b[:, right_indices]
        products = (
            products.toarray().ravel()
            if sparse.issparse(products)
            else np.asarray(products).ravel()
        )
        numerator = products - sum_a[left_index] * sum_b[right_indices] / n_rows
        denominator = np.sqrt(variance_a[left_index] * variance_b[right_indices])
        result[locations] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > np.finfo(np.float64).eps,
        )
    return result


def _activation_correlation_matrix(latents_a: Any, latents_b: Any) -> np.ndarray:
    """Dense all-pairs Pearson matrix for exact assignment."""
    n_rows = latents_a.shape[0]
    sum_a = np.asarray(latents_a.sum(axis=0)).ravel().astype(np.float64)
    sum_b = np.asarray(latents_b.sum(axis=0)).ravel().astype(np.float64)
    if sparse.issparse(latents_a):
        square_a = np.asarray(latents_a.power(2).sum(axis=0)).ravel()
    else:
        square_a = np.square(np.asarray(latents_a, dtype=np.float64)).sum(axis=0)
    if sparse.issparse(latents_b):
        square_b = np.asarray(latents_b.power(2).sum(axis=0)).ravel()
    else:
        square_b = np.square(np.asarray(latents_b, dtype=np.float64)).sum(axis=0)
    cross = latents_a.T @ latents_b
    cross = cross.toarray() if sparse.issparse(cross) else np.asarray(cross)
    numerator = np.asarray(cross, dtype=np.float64) - np.outer(sum_a, sum_b) / n_rows
    variance_a = np.maximum(square_a - np.square(sum_a) / n_rows, 0.0)
    variance_b = np.maximum(square_b - np.square(sum_b) / n_rows, 0.0)
    denominator = np.sqrt(np.outer(variance_a, variance_b))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    ).astype(np.float32)


def _weights(config: MatchingConfig) -> tuple[float, float, float]:
    if config.method == "decoder_cosine":
        return 1.0, 0.0, 0.0
    if config.method == "encoder_cosine":
        return 0.0, 1.0, 0.0
    if config.method == "activation_correlation":
        return 0.0, 0.0, 1.0
    total = (
        config.decoder_weight
        + config.encoder_weight
        + config.activation_correlation_weight
    )
    return (
        config.decoder_weight / total,
        config.encoder_weight / total,
        config.activation_correlation_weight / total,
    )


def _score_block(
    decoder_a: torch.Tensor,
    decoder_b: torch.Tensor,
    encoder_a: torch.Tensor | None,
    encoder_b: torch.Tensor | None,
    start: int,
    end: int,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    decoder_weight, encoder_weight, _ = weights
    score = decoder_weight * (decoder_a[start:end] @ decoder_b.T)
    if encoder_weight:
        if encoder_a is None or encoder_b is None:
            raise ValueError(
                "Encoder matching requested but an SAE has no encoder directions"
            )
        score = score + encoder_weight * (encoder_a[start:end] @ encoder_b.T)
    return torch.nan_to_num(score, nan=0.0)


@dataclass
class MatchResult:
    feature_a: np.ndarray
    feature_b: np.ndarray
    matching_score: np.ndarray
    decoder_cosine: np.ndarray
    encoder_cosine: np.ndarray
    activation_correlation: np.ndarray
    maximum_feature_b: np.ndarray
    maximum_score: np.ndarray
    solver: str


@dataclass
class PaperMatchResult:
    """Separate encoder/decoder assignments and the paper's shared label."""

    feature_a: np.ndarray
    encoder_feature_b: np.ndarray
    decoder_feature_b: np.ndarray
    encoder_cosine: np.ndarray
    decoder_cosine: np.ndarray
    encoder_max_feature_b: np.ndarray
    decoder_max_feature_b: np.ndarray
    encoder_max_cosine: np.ndarray
    decoder_max_cosine: np.ndarray
    same_counterpart: np.ndarray
    is_shared: np.ndarray
    is_orphan: np.ndarray
    average_matched_cosine: np.ndarray
    shared_threshold: float
    encoder_match: MatchResult
    decoder_match: MatchResult


def match_adapters(
    sae_a: SAEAdapter,
    sae_b: SAEAdapter,
    config: MatchingConfig,
    *,
    latents_a: Any | None = None,
    latents_b: Any | None = None,
) -> MatchResult:
    """Maximize configured feature similarity without assuming aligned IDs.

    Exact assignment materializes a ``[d_sae_a, d_sae_b]`` matrix. In ``auto``
    mode, wide dictionaries use a sparse candidate graph (top-k candidates plus
    deterministic feasibility edges), then solve a global sparse linear assignment.
    This preserves one-to-one matching while avoiding a 4 GiB 32k-by-32k matrix.
    """
    weights = _weights(config)
    if weights[2] and (latents_a is None or latents_b is None):
        raise ValueError(
            "Activation-correlation matching requires both latent matrices"
        )
    decoder_a, decoder_b = _normalized(sae_a.decoder), _normalized(sae_b.decoder)
    encoder_a = _normalized(sae_a.encoder) if sae_a.encoder is not None else None
    encoder_b = _normalized(sae_b.encoder) if sae_b.encoder is not None else None
    n_a, n_b = decoder_a.shape[0], decoder_b.shape[0]
    solver = config.solver
    if solver == "auto":
        solver = "exact" if max(n_a, n_b) <= config.exact_max_features else "sparse"
    if solver == "sparse" and weights[2] > 0 and weights[0] == 0 and weights[1] == 0:
        raise ValueError(
            "Pure activation-correlation matching needs solver: exact. Sparse matching "
            "requires a decoder or encoder term to construct candidate edges."
        )
    if weights[2] and solver == "sparse":
        LOGGER.warning(
            "Sparse weighted matching chooses candidates by weight geometry before "
            "adding activation correlation; increase candidate_top_k for robustness"
        )

    if solver == "exact":
        score = np.empty((n_a, n_b), dtype=np.float32)
        for start in tqdm(
            range(0, n_a, config.similarity_batch_size), desc="similarity blocks"
        ):
            end = min(start + config.similarity_batch_size, n_a)
            score[start:end] = (
                _score_block(
                    decoder_a, decoder_b, encoder_a, encoder_b, start, end, weights
                )
                .cpu()
                .numpy()
            )
        if weights[2]:
            score += weights[2] * _activation_correlation_matrix(latents_a, latents_b)
        maximum_feature_b = np.argmax(score, axis=1).astype(np.int64)
        maximum_score = score[np.arange(n_a), maximum_feature_b]
        with monitored_operation(
            f"exact {config.method} Hungarian assignment {sae_a.name}/{sae_b.name}"
        ):
            feature_a, feature_b = linear_sum_assignment(score, maximize=True)
        matching_score = score[feature_a, feature_b]
    else:
        # Orient the candidate graph with the smaller dictionary as rows so a full
        # row matching exists for unequal SAE widths.
        swapped = n_a > n_b
        if swapped:
            row_decoder, col_decoder = decoder_b, decoder_a
            row_encoder, col_encoder = encoder_b, encoder_a
        else:
            row_decoder, col_decoder = decoder_a, decoder_b
            row_encoder, col_encoder = encoder_a, encoder_b
        n_rows, n_columns = row_decoder.shape[0], col_decoder.shape[0]
        maximum_score = np.full(n_a, -np.inf, dtype=np.float32)
        maximum_feature_b = np.full(n_a, -1, dtype=np.int64)
        k = min(config.candidate_top_k, n_columns)
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        for start in tqdm(
            range(0, n_rows, config.similarity_batch_size), desc="candidate blocks"
        ):
            end = min(start + config.similarity_batch_size, n_rows)
            block = _score_block(
                row_decoder, col_decoder, row_encoder, col_encoder, start, end, weights
            )
            if swapped:
                block_maximum, local_rows = block.max(dim=0)
                block_maximum_array = block_maximum.cpu().numpy()
                improved = block_maximum_array > maximum_score
                maximum_score[improved] = block_maximum_array[improved]
                maximum_feature_b[improved] = start + local_rows.cpu().numpy()[improved]
            else:
                block_maximum, block_columns = block.max(dim=1)
                maximum_score[start:end] = block_maximum.cpu().numpy()
                maximum_feature_b[start:end] = block_columns.cpu().numpy()
            values, columns = torch.topk(block, k=k, dim=1)
            rows = np.repeat(np.arange(start, end, dtype=np.int64), k)
            candidate_columns = columns.cpu().numpy().ravel().astype(np.int64)
            candidate_scores = values.cpu().numpy().ravel().astype(np.float32)
            # A unique deterministic edge for each row guarantees graph feasibility.
            feasibility_columns = np.arange(start, end, dtype=np.int64)
            feasibility_scores = (
                block[
                    torch.arange(end - start, device=block.device),
                    torch.as_tensor(feasibility_columns, device=block.device),
                ]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            candidate_matrix = columns.cpu().numpy()
            missing_feasibility = ~np.any(
                candidate_matrix == feasibility_columns[:, None], axis=1
            )
            row_parts.append(rows)
            column_parts.append(candidate_columns)
            score_parts.append(candidate_scores)
            if missing_feasibility.any():
                row_parts.append(
                    np.arange(start, end, dtype=np.int64)[missing_feasibility]
                )
                column_parts.append(feasibility_columns[missing_feasibility])
                score_parts.append(feasibility_scores[missing_feasibility])
        graph_rows = np.concatenate(row_parts)
        graph_columns = np.concatenate(column_parts)
        graph_scores = np.concatenate(score_parts)
        if swapped:
            corr_a, corr_b = graph_columns, graph_rows
        else:
            corr_a, corr_b = graph_rows, graph_columns
        if weights[2]:
            graph_scores += weights[2] * _activation_correlations(
                latents_a, latents_b, corr_a, corr_b
            )
            # The sparse candidate graph does not contain activation correlations
            # for every A/B pair, so a true all-pairs maximum is unavailable.
            maximum_score.fill(np.nan)
            maximum_feature_b.fill(-1)
        maximum = float(graph_scores.max())
        costs = maximum - graph_scores.astype(np.float64) + 1e-8
        graph = sparse.coo_matrix(
            (costs, (graph_rows, graph_columns)), shape=(n_rows, n_columns)
        ).tocsr()
        with monitored_operation(
            f"sparse {config.method} assignment {sae_a.name}/{sae_b.name}"
        ):
            matched_rows, matched_columns = min_weight_full_bipartite_matching(graph)
        score_lookup = sparse.coo_matrix(
            (graph_scores, (graph_rows, graph_columns)), shape=(n_rows, n_columns)
        ).tocsr()
        matching_score = np.asarray(score_lookup[matched_rows, matched_columns]).ravel()
        if swapped:
            feature_a, feature_b = matched_columns, matched_rows
        else:
            feature_a, feature_b = matched_rows, matched_columns

    feature_a = np.asarray(feature_a, dtype=np.int64)
    feature_b = np.asarray(feature_b, dtype=np.int64)
    with torch.no_grad():
        decoder_cosine = (
            (
                decoder_a[torch.as_tensor(feature_a, device=decoder_a.device)]
                * decoder_b[torch.as_tensor(feature_b, device=decoder_b.device)]
            )
            .sum(dim=1)
            .cpu()
            .numpy()
        )
        if encoder_a is not None and encoder_b is not None:
            encoder_cosine = (
                (
                    encoder_a[torch.as_tensor(feature_a, device=encoder_a.device)]
                    * encoder_b[torch.as_tensor(feature_b, device=encoder_b.device)]
                )
                .sum(dim=1)
                .cpu()
                .numpy()
            )
        else:
            encoder_cosine = np.full(len(feature_a), np.nan, dtype=np.float32)
    if latents_a is not None and latents_b is not None:
        with monitored_operation(
            f"matched activation correlations {sae_a.name}/{sae_b.name}"
        ):
            activation_correlation = _activation_correlations(
                latents_a, latents_b, feature_a, feature_b
            )
    else:
        activation_correlation = np.full(len(feature_a), np.nan, dtype=np.float32)
    return MatchResult(
        feature_a=feature_a,
        feature_b=feature_b,
        matching_score=np.asarray(matching_score, dtype=np.float32),
        decoder_cosine=decoder_cosine,
        encoder_cosine=encoder_cosine,
        activation_correlation=activation_correlation,
        maximum_feature_b=np.asarray(maximum_feature_b, dtype=np.int64),
        maximum_score=np.asarray(maximum_score, dtype=np.float32),
        solver=solver,
    )


def maximum_cosine_matches(
    directions_a: Any,
    directions_b: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each A direction's non-bijective nearest B direction and cosine."""
    left = _normalized(directions_a)
    right = _normalized(directions_b)
    feature_b = np.empty(left.shape[0], dtype=np.int64)
    cosine = np.empty(left.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(
            range(0, left.shape[0], batch_size), desc="maximum-cosine blocks"
        ):
            end = min(start + batch_size, left.shape[0])
            score = torch.nan_to_num(left[start:end] @ right.T, nan=0.0)
            values, indices = score.max(dim=1)
            feature_b[start:end] = indices.cpu().numpy()
            cosine[start:end] = values.cpu().numpy()
    return feature_b, cosine


def classify_shared_latents(
    encoder_feature_b: np.ndarray,
    decoder_feature_b: np.ndarray,
    encoder_cosine: np.ndarray,
    decoder_cosine: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return assignment agreement and the paper's inclusive shared mask."""
    same_counterpart = np.asarray(encoder_feature_b) == np.asarray(decoder_feature_b)
    is_shared = (
        same_counterpart
        & (np.asarray(encoder_cosine) >= threshold)
        & (np.asarray(decoder_cosine) >= threshold)
    )
    return same_counterpart, is_shared


def _ordered_full_match(
    result: MatchResult, expected_features: int, label: str
) -> MatchResult:
    """Put a full equal-width assignment in SAE-A feature order."""
    order = np.argsort(result.feature_a)
    feature_a = result.feature_a[order]
    expected = np.arange(expected_features, dtype=np.int64)
    if not np.array_equal(feature_a, expected):
        raise ValueError(
            f"Paper matching requires a full {label} assignment for every SAE-A latent"
        )
    return MatchResult(
        feature_a=feature_a,
        feature_b=result.feature_b[order],
        matching_score=result.matching_score[order],
        decoder_cosine=result.decoder_cosine[order],
        encoder_cosine=result.encoder_cosine[order],
        activation_correlation=result.activation_correlation[order],
        maximum_feature_b=result.maximum_feature_b,
        maximum_score=result.maximum_score,
        solver=result.solver,
    )


def match_paper_shared_latents(
    sae_a: SAEAdapter,
    sae_b: SAEAdapter,
    config: MatchingConfig,
    *,
    shared_threshold: float = 0.7,
    latents_a: Any | None = None,
    latents_b: Any | None = None,
) -> PaperMatchResult:
    """Apply the paper's two-Hungarian shared/orphan definition.

    Encoder and decoder directions are assigned independently. A latent is shared
    iff both assignments choose the same SAE-B counterpart and both independently
    matched cosine similarities are at least ``shared_threshold``. The complement
    is classified as orphan.
    """
    if sae_a.d_sae != sae_b.d_sae:
        raise ValueError(
            "Paper shared/orphan matching requires equal-width SAE dictionaries; "
            f"got {sae_a.d_sae} and {sae_b.d_sae}"
        )
    if sae_a.encoder is None or sae_b.encoder is None:
        raise ValueError("Paper shared/orphan matching requires encoder directions")

    decoder_match = _ordered_full_match(
        match_adapters(
            sae_a,
            sae_b,
            replace(config, method="decoder_cosine"),
            latents_a=latents_a,
            latents_b=latents_b,
        ),
        sae_a.d_sae,
        "decoder",
    )
    encoder_match = _ordered_full_match(
        match_adapters(sae_a, sae_b, replace(config, method="encoder_cosine")),
        sae_a.d_sae,
        "encoder",
    )
    same_counterpart, is_shared = classify_shared_latents(
        encoder_match.feature_b,
        decoder_match.feature_b,
        encoder_match.matching_score,
        decoder_match.matching_score,
        threshold=shared_threshold,
    )
    return PaperMatchResult(
        feature_a=decoder_match.feature_a,
        encoder_feature_b=encoder_match.feature_b,
        decoder_feature_b=decoder_match.feature_b,
        encoder_cosine=encoder_match.matching_score,
        decoder_cosine=decoder_match.matching_score,
        encoder_max_feature_b=encoder_match.maximum_feature_b,
        decoder_max_feature_b=decoder_match.maximum_feature_b,
        encoder_max_cosine=encoder_match.maximum_score,
        decoder_max_cosine=decoder_match.maximum_score,
        same_counterpart=same_counterpart,
        is_shared=is_shared,
        is_orphan=~is_shared,
        average_matched_cosine=(
            encoder_match.matching_score + decoder_match.matching_score
        )
        / 2.0,
        shared_threshold=float(shared_threshold),
        encoder_match=encoder_match,
        decoder_match=decoder_match,
    )
