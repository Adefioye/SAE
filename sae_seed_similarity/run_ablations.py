"""Run matched and random-pair SAE feature ablations with per-SAE baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm.auto import tqdm

from .adapters import SAEAdapter, load_base_model, load_sae
from .config import EvaluationConfig, load_config
from .metrics import ablation_metrics, js_divergence_from_logits
from .statistics import bootstrap_ci
from .storage import ArtifactStore
from .utils import (
    configure_logging,
    resolve_device,
    seed_everything,
    validate_cache_manifest,
)

LOGGER = logging.getLogger(__name__)


def _dense_column(matrix: Any, feature: int) -> np.ndarray:
    return (
        matrix.getcol(feature).toarray().ravel()
        if sparse.issparse(matrix)
        else np.asarray(matrix[:, feature]).ravel()
    )


def select_rows(
    left: Any,
    right: Any,
    feature_a: int,
    feature_b: int,
    *,
    mode: str,
    threshold: float,
    maximum: int,
) -> np.ndarray:
    """Select shared activation rows for one matched pair, ordered by strength."""
    activation_a = _dense_column(left, feature_a)
    activation_b = _dense_column(right, feature_b)
    if mode == "both_active":
        candidates = np.flatnonzero(
            (activation_a > threshold) & (activation_b > threshold)
        )
    elif mode == "either_active":
        candidates = np.flatnonzero(
            (activation_a > threshold) | (activation_b > threshold)
        )
    else:
        count = min(maximum, len(activation_a))
        top_a = np.argpartition(activation_a, -count)[-count:]
        top_b = np.argpartition(activation_b, -count)[-count:]
        candidates = (
            np.union1d(top_a, top_b)
            if mode == "top_activating"
            else np.intersect1d(top_a, top_b)
        )
        candidates = candidates[
            (activation_a[candidates] > threshold)
            | (activation_b[candidates] > threshold)
        ]
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    strength = activation_a[candidates] + activation_b[candidates]
    order = np.argsort(strength)[::-1]
    return candidates[order[:maximum]].astype(np.int64)


@torch.inference_mode()
def _intervened_logits(
    model: Any,
    adapter: SAEAdapter,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    hook_point: str,
    feature: int,
    selected_positions: torch.Tensor,
    *,
    ablate: bool,
    scope: str,
    minimum_activation: float,
) -> torch.Tensor:
    """Replace the hook with this SAE's reconstruction, optionally zeroing one latent."""

    def replace(activation: torch.Tensor, _hook: Any) -> torch.Tensor:
        latents = adapter.encode(activation)
        if ablate:
            latents = latents.clone()
            if scope == "selected_token":
                batch = torch.arange(len(latents), device=latents.device)
                latents[batch, selected_positions.to(latents.device), feature] = 0
            elif scope == "active_positions":
                values = latents[..., feature]
                latents[..., feature] = torch.where(
                    values > minimum_activation, torch.zeros_like(values), values
                )
            elif scope == "all_positions":
                latents[..., feature] = 0
            else:
                raise ValueError(f"Unknown intervention scope: {scope}")
        return adapter.decode(latents)

    return model.run_with_hooks(
        tokens,
        attention_mask=attention_mask,
        fwd_hooks=[(hook_point, replace)],
        return_type="logits",
    )


def _evaluation_locations(
    positions: np.ndarray, mask: np.ndarray, horizon: int
) -> list[tuple[int, int, int]]:
    locations: list[tuple[int, int, int]] = []
    for batch, position in enumerate(positions):
        valid_length = int(mask[batch].sum())
        for offset in range(max(horizon, 1)):
            evaluation_position = int(position + offset)
            if evaluation_position < valid_length:
                locations.append((batch, evaluation_position, offset))
    return locations


def _pair_table(config: EvaluationConfig, store: ArtifactStore) -> pd.DataFrame:
    matches = pd.read_parquet(store.root / "hungarian_matches.parquet")
    matches = matches[matches["passes_minimum_similarity"]].copy()
    matches = (
        matches.sort_values("matching_score", ascending=False)
        .groupby(["sae_a", "sae_b"], as_index=False, group_keys=False)
        .head(config.ablation.max_feature_pairs)
    )
    matches["pair_type"] = "matched"
    tables = [matches[["sae_a", "sae_b", "feature_a", "feature_b", "pair_type"]]]
    if config.controls.enabled:
        identity = matches[["sae_a", "feature_a"]].drop_duplicates().copy()
        identity["sae_b"] = identity["sae_a"]
        identity["feature_b"] = identity["feature_a"]
        identity["pair_type"] = "identity_control"
        tables.append(
            identity[["sae_a", "sae_b", "feature_a", "feature_b", "pair_type"]]
        )
    if (
        config.controls.enabled
        and (store.root / "random_feature_pairs.parquet").exists()
    ):
        random_pairs = pd.read_parquet(store.root / "random_feature_pairs.parquet")
        keys = matches[["sae_a", "sae_b", "feature_a", "feature_b"]].rename(
            columns={"feature_b": "matched_feature_b"}
        )
        random_pairs = random_pairs.merge(
            keys, on=["sae_a", "sae_b", "feature_a", "matched_feature_b"]
        )
        random_pairs["pair_type"] = "random_control"
        tables.append(
            random_pairs[["sae_a", "sae_b", "feature_a", "feature_b", "pair_type"]]
        )
    return pd.concat(tables, ignore_index=True)


def _aggregate(
    prompt_level: pd.DataFrame,
    failures: list[dict[str, Any]],
    config: EvaluationConfig,
) -> pd.DataFrame:
    metrics = {
        "effect_jsd_a": "mean_effect_jsd_a",
        "effect_jsd_b": "mean_effect_jsd_b",
        "ablation_jsd_between_seeds": "mean_ablation_jsd_between_seeds",
        "baseline_adjusted_ablation_jsd": "mean_baseline_adjusted_ablation_jsd",
        "logit_delta_cosine": "mean_logit_delta_cosine",
        "probability_delta_cosine": "mean_probability_delta_cosine",
        "top1_disagreement": "top1_disagreement_rate",
        "topk_overlap": "mean_topk_overlap",
        "effect_norm_a": "mean_effect_norm_a",
        "effect_norm_b": "mean_effect_norm_b",
    }
    keys = ["sae_a", "sae_b", "feature_a", "feature_b", "pair_type"]
    rows: list[dict[str, Any]] = []
    for group_key, frame in prompt_level.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, group_key, strict=True))
        row["number_of_evaluated_examples"] = int(frame["activation_row"].nunique())
        row["number_of_evaluated_logits"] = len(frame)
        for source, destination in metrics.items():
            values = frame[source].to_numpy(dtype=np.float64)
            interval = bootstrap_ci(
                values,
                samples=config.bootstrap.samples,
                confidence_level=config.bootstrap.confidence_level,
                random_seed=config.dataset.random_seed + int(row["feature_a"]),
            )
            row[destination] = interval.estimate
            row[f"{destination}_ci_low"] = interval.low
            row[f"{destination}_ci_high"] = interval.high
        row["median_logit_delta_cosine"] = float(frame["logit_delta_cosine"].median())
        row["informative_effect_fraction"] = float(
            (frame["effect_status"] == "informative").mean()
        )
        row["selection_failure_reason"] = None
        rows.append(row)
    rows.extend(failures)
    return pd.DataFrame(rows)


@torch.inference_mode()
def run(config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = ArtifactStore(config.output_path).ensure()
    if not validate_cache_manifest(config.to_dict(), store.root, force=config.force):
        raise RuntimeError("Run collect_activations before causal ablations")
    prompt_path = store.root / "ablation_prompt_level.parquet"
    feature_path = store.root / "ablation_feature_level.parquet"
    if prompt_path.exists() and feature_path.exists() and not config.force:
        LOGGER.info("Ablation artifacts already exist; reusing cache")
        return pd.read_parquet(prompt_path), pd.read_parquet(feature_path)
    if not config.ablation.enabled:
        LOGGER.info("Ablations are disabled in configuration")
        empty = pd.DataFrame()
        empty.to_parquet(prompt_path, index=False)
        empty.to_parquet(feature_path, index=False)
        return empty, empty

    seed_everything(config.dataset.random_seed)
    device = resolve_device(config.base_model.device)
    config.base_model.device = device
    model = load_base_model(config.base_model)
    model.eval()
    adapters = {
        item.name: load_sae(item, device, config.base_model.dtype)
        for item in config.saes
    }
    latents = {item.name: store.load_latents(item.name) for item in config.saes}
    metadata = pd.read_parquet(store.row_metadata).set_index("activation_row")
    all_tokens = np.load(store.token_ids)
    all_masks = np.load(store.attention_mask)
    pairs = _pair_table(config, store)
    prompt_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for pair in tqdm(
        pairs.itertuples(index=False), total=len(pairs), desc="feature-pair ablations"
    ):
        selected_rows = select_rows(
            latents[pair.sae_a],
            latents[pair.sae_b],
            int(pair.feature_a),
            int(pair.feature_b),
            mode=config.ablation.selection_mode,
            threshold=config.ablation.minimum_activation,
            maximum=config.ablation.examples_per_pair,
        )
        if len(selected_rows) == 0:
            failures.append(
                {
                    "sae_a": pair.sae_a,
                    "sae_b": pair.sae_b,
                    "feature_a": int(pair.feature_a),
                    "feature_b": int(pair.feature_b),
                    "pair_type": pair.pair_type,
                    "number_of_evaluated_examples": 0,
                    "number_of_evaluated_logits": 0,
                    "selection_failure_reason": "empty_selection",
                }
            )
            continue
        selected_metadata = metadata.loc[selected_rows]
        sequence_ids = selected_metadata["sequence_id"].to_numpy(dtype=np.int64)
        positions = selected_metadata["token_position"].to_numpy(dtype=np.int64)
        tokens_np = all_tokens[sequence_ids]
        masks_np = all_masks[sequence_ids]
        tokens = torch.as_tensor(tokens_np, device=device)
        mask = torch.as_tensor(masks_np, device=device)
        position_tensor = torch.as_tensor(positions, device=device)
        adapter_a, adapter_b = adapters[pair.sae_a], adapters[pair.sae_b]
        baseline_a = _intervened_logits(
            model,
            adapter_a,
            tokens,
            mask,
            config.base_model.hook_point,
            int(pair.feature_a),
            position_tensor,
            ablate=False,
            scope=config.ablation.intervention_scope,
            minimum_activation=config.ablation.minimum_activation,
        )
        ablated_a = _intervened_logits(
            model,
            adapter_a,
            tokens,
            mask,
            config.base_model.hook_point,
            int(pair.feature_a),
            position_tensor,
            ablate=True,
            scope=config.ablation.intervention_scope,
            minimum_activation=config.ablation.minimum_activation,
        )
        baseline_b = _intervened_logits(
            model,
            adapter_b,
            tokens,
            mask,
            config.base_model.hook_point,
            int(pair.feature_b),
            position_tensor,
            ablate=False,
            scope=config.ablation.intervention_scope,
            minimum_activation=config.ablation.minimum_activation,
        )
        ablated_b = _intervened_logits(
            model,
            adapter_b,
            tokens,
            mask,
            config.base_model.hook_point,
            int(pair.feature_b),
            position_tensor,
            ablate=True,
            scope=config.ablation.intervention_scope,
            minimum_activation=config.ablation.minimum_activation,
        )
        clean = (
            model(tokens, attention_mask=mask, return_type="logits")
            if config.ablation.include_clean_logits
            else None
        )
        for batch_index, evaluation_position, offset in _evaluation_locations(
            positions, masks_np, config.ablation.evaluation_horizon
        ):
            vectors = [
                item[batch_index, evaluation_position].float().cpu().numpy()
                for item in (baseline_a, ablated_a, baseline_b, ablated_b)
            ]
            values = ablation_metrics(
                *vectors,
                top_k=config.ablation.top_k,
                minimum_effect_norm=config.ablation.minimum_effect_norm,
            )
            if clean is not None:
                clean_vector = (
                    clean[batch_index, evaluation_position].float().cpu().numpy()
                )
                values["clean_to_baseline_jsd_a"] = js_divergence_from_logits(
                    clean_vector, vectors[0]
                )
                values["clean_to_baseline_jsd_b"] = js_divergence_from_logits(
                    clean_vector, vectors[2]
                )
            values.update(
                {
                    "sae_a": pair.sae_a,
                    "sae_b": pair.sae_b,
                    "feature_a": int(pair.feature_a),
                    "feature_b": int(pair.feature_b),
                    "pair_type": pair.pair_type,
                    "activation_row": int(selected_rows[batch_index]),
                    "sequence_id": int(sequence_ids[batch_index]),
                    "dataset_index": int(
                        selected_metadata.iloc[batch_index]["dataset_index"]
                    ),
                    "token_position": int(positions[batch_index]),
                    "evaluation_position": evaluation_position,
                    "evaluation_offset": offset,
                    "token_id": int(tokens_np[batch_index, positions[batch_index]]),
                }
            )
            prompt_rows.append(values)
        del baseline_a, ablated_a, baseline_b, ablated_b, clean

    prompt_level = pd.DataFrame(prompt_rows)
    feature_level = (
        _aggregate(prompt_level, failures, config)
        if len(prompt_level)
        else pd.DataFrame(failures)
    )
    prompt_level.to_parquet(prompt_path, index=False)
    feature_level.to_parquet(feature_path, index=False)
    return prompt_level, feature_level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
