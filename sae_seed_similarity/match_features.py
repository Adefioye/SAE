"""Match all SAE seed pairs and compute matched-feature activation overlap."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .adapters import load_sae
from .config import EvaluationConfig, load_config
from .controls import run_feature_controls
from .matching import match_adapters
from .metrics import activation_overlap
from .storage import ArtifactStore
from .utils import configure_logging, pairwise, resolve_device, validate_cache_manifest

LOGGER = logging.getLogger(__name__)


def _threshold(matrix: Any, feature: int, config: EvaluationConfig) -> float:
    mode = config.activations.threshold_mode
    if mode in ("fixed", "positive"):
        return config.activations.active_threshold if mode == "fixed" else 0.0
    column = (
        matrix.getcol(feature).data
        if hasattr(matrix, "getcol")
        else np.asarray(matrix[:, feature]).ravel()
    )
    nonzero = np.asarray(column)[np.asarray(column) > 0]
    return (
        float(np.quantile(nonzero, config.activations.threshold_quantile))
        if len(nonzero)
        else np.inf
    )


def run(config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ArtifactStore(config.output_path).ensure()
    if not validate_cache_manifest(config.to_dict(), store.root, force=config.force):
        raise RuntimeError("Run collect_activations before feature matching")
    matches_path = store.root / "hungarian_matches.parquet"
    overlap_path = store.root / "activation_overlap.parquet"
    control_complete = not config.controls.enabled or (
        (store.root / "random_feature_pairs.parquet").exists()
        and (store.root / "control_activation_overlap.parquet").exists()
    )
    if (
        matches_path.exists()
        and overlap_path.exists()
        and control_complete
        and not config.force
    ):
        LOGGER.info("Feature comparison artifacts already exist; reusing cache")
        return pd.read_parquet(matches_path), pd.read_parquet(overlap_path)
    metadata = pd.read_parquet(store.row_metadata)
    sequence_ids = metadata["sequence_id"].to_numpy()
    device = resolve_device(config.base_model.device)
    adapters = {
        item.name: load_sae(item, device, config.base_model.dtype)
        for item in config.saes
    }
    latents = {item.name: store.load_latents(item.name) for item in config.saes}
    match_frames: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, object]] = []
    for sae_a, sae_b in pairwise(config.saes):
        LOGGER.info("Matching %s to %s", sae_a.name, sae_b.name)
        result = match_adapters(
            adapters[sae_a.name],
            adapters[sae_b.name],
            config.matching,
            latents_a=latents[sae_a.name],
            latents_b=latents[sae_b.name],
        )
        frame = pd.DataFrame(
            {
                "sae_a": sae_a.name,
                "sae_b": sae_b.name,
                "feature_a": result.feature_a,
                "feature_b": result.feature_b,
                "matching_score": result.matching_score,
                "decoder_cosine": result.decoder_cosine,
                "encoder_cosine_if_available": result.encoder_cosine,
                "activation_correlation": result.activation_correlation,
                "solver": result.solver,
                "passes_minimum_similarity": result.matching_score
                >= config.matching.minimum_similarity,
            }
        )
        match_frames.append(frame)
        for row in tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc=f"overlap {sae_a.name}/{sae_b.name}",
        ):
            threshold_a = _threshold(latents[sae_a.name], row.feature_a, config)
            threshold_b = _threshold(latents[sae_b.name], row.feature_b, config)
            metrics = activation_overlap(
                latents[sae_a.name],
                latents[sae_b.name],
                row.feature_a,
                row.feature_b,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
                sequence_ids=sequence_ids,
            )
            metrics.update(
                sae_a=sae_a.name,
                sae_b=sae_b.name,
                decoder_cosine=row.decoder_cosine,
                matching_score=row.matching_score,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
            )
            overlap_rows.append(metrics)
    matches = pd.concat(match_frames, ignore_index=True)
    overlap = pd.DataFrame(overlap_rows)
    matches.to_parquet(matches_path, index=False)
    overlap.to_parquet(overlap_path, index=False)
    if config.controls.enabled:
        run_feature_controls(config, matches, adapters, latents, sequence_ids)
    return matches, overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
