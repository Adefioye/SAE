"""Compute pairwise CKA, SVCCA, PWCCA, and global representation controls."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import EvaluationConfig, load_config
from .metrics import linear_cka, svcca_pwcca
from .storage import ArtifactStore
from .utils import (
    configure_logging,
    pairwise,
    stable_sample_indices,
    validate_cache_manifest,
)

LOGGER = logging.getLogger(__name__)


def _rows(matrix: Any, indices: np.ndarray) -> Any:
    selected = matrix[indices]
    return selected.tocsr() if sparse.issparse(selected) else np.asarray(selected)


def _matrix(names: list[str], rows: list[dict[str, Any]], value: str) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in rows:
        matrix.loc[row["sae_a"], row["sae_b"]] = row[value]
        matrix.loc[row["sae_b"], row["sae_a"]] = row[value]
    matrix.index.name = "sae"
    return matrix


def run(config: EvaluationConfig) -> pd.DataFrame:
    store = ArtifactStore(config.output_path).ensure()
    if not validate_cache_manifest(config.to_dict(), store.root, force=config.force):
        raise RuntimeError("Run collect_activations before representation comparison")
    summary_path = store.root / "seed_pair_summary.csv"
    required = [
        summary_path,
        store.root / "cka_matrix.csv",
        store.root / "svcca_summary.csv",
        store.root / "pwcca_summary.csv",
        store.root / "svcca_matrix.csv",
        store.root / "pwcca_matrix.csv",
    ]
    if config.controls.enabled:
        required.append(store.root / "controls_summary.csv")
    if all(path.exists() for path in required) and not config.force:
        LOGGER.info("Representation artifacts already exist; reusing cache")
        return pd.read_csv(summary_path)

    names = [item.name for item in config.saes]
    latents = {name: store.load_latents(name) for name in names}
    pair_rows: list[dict[str, Any]] = []
    svcca_rows: list[dict[str, Any]] = []
    pwcca_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for pair_index, (sae_a, sae_b) in enumerate(pairwise(config.saes)):
        left, right = latents[sae_a.name], latents[sae_b.name]
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"Shared row count differs for {sae_a.name} and {sae_b.name}"
            )
        cka_indices = stable_sample_indices(
            left.shape[0],
            config.cka.max_samples,
            config.dataset.random_seed + pair_index,
        )
        subspace_indices = stable_sample_indices(
            left.shape[0],
            config.svcca.max_samples,
            config.dataset.random_seed + pair_index,
        )
        cka_left, cka_right = _rows(left, cka_indices), _rows(right, cka_indices)
        subspace_left, subspace_right = (
            _rows(left, subspace_indices),
            _rows(right, subspace_indices),
        )
        LOGGER.info("CKA %s/%s on %d rows", sae_a.name, sae_b.name, len(cka_indices))
        cka_raw = linear_cka(cka_left, cka_right, center=config.cka.center)
        cka_standardized = linear_cka(
            cka_left, cka_right, center=config.cka.center, standardize_features=True
        )
        LOGGER.info("SVCCA/PWCCA %s/%s", sae_a.name, sae_b.name)
        svcca_result, pwcca_result = svcca_pwcca(
            subspace_left,
            subspace_right,
            explained_variance=config.svcca.explained_variance,
            max_components=config.svcca.max_components,
            ridge=config.svcca.ridge,
            random_seed=config.dataset.random_seed + pair_index,
        )
        common = {"sae_a": sae_a.name, "sae_b": sae_b.name}
        svcca_row = {
            **common,
            **svcca_result.to_dict(),
            "n_samples": len(subspace_indices),
        }
        pwcca_row = {
            **common,
            **pwcca_result.to_dict(),
            "n_samples": len(subspace_indices),
        }
        svcca_rows.append(svcca_row)
        pwcca_rows.append(pwcca_row)
        pair_rows.append(
            {
                **common,
                "n_cka_samples": len(cka_indices),
                "n_subspace_samples": len(subspace_indices),
                "cka": cka_raw,
                "cka_standardized": cka_standardized,
                "svcca_mean": svcca_result.mean_correlation,
                "svcca_median": svcca_result.median_correlation,
                "pwcca": pwcca_result.similarity,
            }
        )
        stem = f"{sae_a.name}__{sae_b.name}"
        np.savez_compressed(
            store.root / "svcca_correlations" / f"{stem}.npz",
            correlations=svcca_result.correlations,
            pca_curve_a=svcca_result.pca_curve_a,
            pca_curve_b=svcca_result.pca_curve_b,
            pwcca_correlations=pwcca_result.correlations,
            pwcca_weights=pwcca_result.projection_weights,
        )

        if config.controls.enabled:
            rng = np.random.default_rng(
                config.dataset.random_seed + 10_000 + pair_index
            )
            permutation = rng.permutation(len(cka_indices))
            shuffled_cka = linear_cka(cka_left, cka_right[permutation])
            shuffled_subspace = subspace_right[rng.permutation(len(subspace_indices))]
            shuffled_svcca, shuffled_pwcca = svcca_pwcca(
                subspace_left,
                shuffled_subspace,
                explained_variance=config.svcca.explained_variance,
                max_components=config.svcca.max_components,
                ridge=config.svcca.ridge,
                random_seed=config.dataset.random_seed + 20_000 + pair_index,
            )
            control_rows.append(
                {
                    **common,
                    "control": "shuffled_tokens",
                    "cka": shuffled_cka,
                    "svcca": shuffled_svcca.mean_correlation,
                    "pwcca": shuffled_pwcca.similarity,
                }
            )

    summary = pd.DataFrame(pair_rows)
    svcca_frame, pwcca_frame = pd.DataFrame(svcca_rows), pd.DataFrame(pwcca_rows)
    summary.to_csv(summary_path, index=False)
    svcca_frame.to_csv(store.root / "svcca_summary.csv", index=False)
    pwcca_frame.to_csv(store.root / "pwcca_summary.csv", index=False)
    _matrix(names, pair_rows, "cka").to_csv(store.root / "cka_matrix.csv")
    _matrix(
        names,
        [{**row, "value": row["mean_correlation"]} for row in svcca_rows],
        "value",
    ).to_csv(store.root / "svcca_matrix.csv")
    _matrix(
        names, [{**row, "value": row["similarity"]} for row in pwcca_rows], "value"
    ).to_csv(store.root / "pwcca_matrix.csv")

    if config.controls.enabled:
        # Identity and column-permutation controls need only one SAE; all three
        # global metrics are invariant to a latent-axis permutation.
        first = latents[names[0]]
        indices = stable_sample_indices(
            first.shape[0], config.cka.max_samples, config.dataset.random_seed
        )
        identity = _rows(first, indices)
        permutation = np.random.default_rng(config.dataset.random_seed).permutation(
            first.shape[1]
        )
        permuted = identity[:, permutation]
        for label, other in (("identity", identity), ("column_permutation", permuted)):
            sub_result, pw_result = svcca_pwcca(
                identity,
                other,
                explained_variance=config.svcca.explained_variance,
                max_components=config.svcca.max_components,
                ridge=config.svcca.ridge,
                random_seed=config.dataset.random_seed,
            )
            control_rows.append(
                {
                    "sae_a": names[0],
                    "sae_b": names[0],
                    "control": label,
                    "cka": linear_cka(identity, other),
                    "svcca": sub_result.mean_correlation,
                    "pwcca": pw_result.similarity,
                }
            )
        pd.DataFrame(control_rows).to_csv(
            store.root / "controls_summary.csv", index=False
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
