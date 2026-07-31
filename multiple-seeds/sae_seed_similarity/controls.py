"""Feature-pair controls matched on activation and dictionary statistics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

from .config import EvaluationConfig
from .metrics import activation_overlap
from .storage import ArtifactStore
from .utils import monitored_operation

LOGGER = logging.getLogger(__name__)


def feature_statistics(matrix: Any, decoder: Any) -> pd.DataFrame:
    """Return frequency, mean-positive activation, and decoder norm per latent."""
    if sparse.issparse(matrix):
        counts = np.asarray(matrix.getnnz(axis=0)).ravel()
        sums = np.asarray(matrix.sum(axis=0)).ravel()
    else:
        dense = np.asarray(matrix)
        active = dense > 0
        counts = active.sum(axis=0)
        sums = np.where(active, dense, 0).sum(axis=0)
    mean_nonzero = np.divide(
        sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0
    )
    decoder_array = (
        decoder.detach().float().cpu().numpy()
        if hasattr(decoder, "detach")
        else np.asarray(decoder)
    )
    return pd.DataFrame(
        {
            "feature": np.arange(matrix.shape[1], dtype=np.int64),
            "activation_frequency": counts / matrix.shape[0],
            "mean_nonzero_activation": mean_nonzero,
            "decoder_norm": np.linalg.norm(decoder_array, axis=1),
        }
    )


def _random_pairs(
    matches: pd.DataFrame,
    stats_b: pd.DataFrame,
    config: EvaluationConfig,
    pair_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(pair_seed)
    frequencies = stats_b["activation_frequency"].to_numpy()
    positive = frequencies[frequencies > 0]
    if len(positive):
        edges = np.unique(
            np.quantile(positive, np.linspace(0, 1, config.controls.frequency_bins + 1))
        )
    else:
        edges = np.array([0.0, 1.0])
    if len(edges) < 2:
        edges = np.array([edges[0], edges[0] + 1e-12])
    frequency_bins = np.clip(np.digitize(frequencies, edges[1:-1]), 0, len(edges) - 2)
    records: list[dict[str, Any]] = []
    for row in tqdm(
        matches.itertuples(index=False),
        total=len(matches),
        desc="select frequency-matched controls",
    ):
        target = stats_b.iloc[int(row.feature_b)]
        target_bin = frequency_bins[int(row.feature_b)]
        candidates = np.flatnonzero(frequency_bins == target_bin)
        candidates = candidates[candidates != int(row.feature_b)]
        if len(candidates) == 0:
            candidates = np.setdiff1d(np.arange(len(stats_b)), [int(row.feature_b)])
        candidate_stats = stats_b.iloc[candidates]
        target_vector = np.log1p(
            [
                target.activation_frequency,
                target.mean_nonzero_activation,
                target.decoder_norm,
            ]
        )
        candidate_vectors = np.log1p(
            candidate_stats[
                ["activation_frequency", "mean_nonzero_activation", "decoder_norm"]
            ].to_numpy()
        )
        scale = np.maximum(candidate_vectors.std(axis=0), 1e-12)
        distances = np.linalg.norm((candidate_vectors - target_vector) / scale, axis=1)
        nearest = candidates[np.argsort(distances)[: min(50, len(candidates))]]
        for control_index in range(config.controls.random_pairs_per_match):
            random_b = int(rng.choice(nearest))
            records.append(
                {
                    "sae_a": row.sae_a,
                    "sae_b": row.sae_b,
                    "feature_a": int(row.feature_a),
                    "feature_b": random_b,
                    "matched_feature_b": int(row.feature_b),
                    "control_index": control_index,
                    "frequency_bin": int(target_bin),
                }
            )
    return pd.DataFrame(records)


def run_feature_controls(
    config: EvaluationConfig,
    matches: pd.DataFrame,
    adapters: dict[str, Any],
    latents: dict[str, Any],
    sequence_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate random-pair and shuffled-token activation-overlap controls."""
    store = ArtifactStore(config.output_path).ensure()
    pairs_path = store.root / "random_feature_pairs.parquet"
    overlap_path = store.root / "control_activation_overlap.parquet"
    if pairs_path.exists() and overlap_path.exists() and not config.force:
        return pd.read_parquet(pairs_path), pd.read_parquet(overlap_path)
    random_frames: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, Any]] = []
    for pair_index, ((sae_a, sae_b), frame) in enumerate(
        matches.groupby(["sae_a", "sae_b"], sort=False)
    ):
        with monitored_operation(f"control feature statistics {sae_b}"):
            stats_b = feature_statistics(latents[sae_b], adapters[sae_b].decoder)
        random_frame = _random_pairs(
            frame,
            stats_b,
            config,
            config.dataset.random_seed + 30_000 + pair_index,
        )
        random_frames.append(random_frame)
        with monitored_operation(f"shuffle activation rows {sae_b}"):
            shuffled_b = latents[sae_b][
                np.random.default_rng(
                    config.dataset.random_seed + 40_000 + pair_index
                ).permutation(latents[sae_b].shape[0])
            ]
        controls = [
            ("random_pair", random_frame),
            ("shuffled_tokens", frame[["sae_a", "sae_b", "feature_a", "feature_b"]]),
        ]
        for control_name, controls_frame in controls:
            right_matrix = (
                shuffled_b if control_name == "shuffled_tokens" else latents[sae_b]
            )
            for row in tqdm(
                controls_frame.itertuples(index=False),
                total=len(controls_frame),
                desc=f"{control_name} {sae_a}/{sae_b}",
            ):
                values = activation_overlap(
                    latents[sae_a],
                    right_matrix,
                    int(row.feature_a),
                    int(row.feature_b),
                    threshold_a=config.activations.active_threshold,
                    threshold_b=config.activations.active_threshold,
                    sequence_ids=sequence_ids,
                )
                values.update(sae_a=sae_a, sae_b=sae_b, control=control_name)
                if control_name == "random_pair":
                    values["matched_feature_b"] = int(row.matched_feature_b)
                    values["control_index"] = int(row.control_index)
                overlap_rows.append(values)
        if pair_index == 0:
            identity_features = (
                frame["feature_a"].drop_duplicates().to_numpy(dtype=np.int64)
            )
            for feature in tqdm(
                identity_features,
                total=len(identity_features),
                desc=f"identity control {sae_a}",
            ):
                values = activation_overlap(
                    latents[sae_a],
                    latents[sae_a],
                    int(feature),
                    int(feature),
                    threshold_a=config.activations.active_threshold,
                    threshold_b=config.activations.active_threshold,
                    sequence_ids=sequence_ids,
                )
                values.update(sae_a=sae_a, sae_b=sae_a, control="identity")
                overlap_rows.append(values)
    random_pairs = pd.concat(random_frames, ignore_index=True)
    overlaps = pd.DataFrame(overlap_rows)
    with monitored_operation("save activation-overlap control tables"):
        random_pairs.to_parquet(pairs_path, index=False)
        overlaps.to_parquet(overlap_path, index=False)
    return random_pairs, overlaps
