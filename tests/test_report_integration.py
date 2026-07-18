from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sae_seed_similarity.config import load_config
from sae_seed_similarity.make_report import run
from sae_seed_similarity.storage import ArtifactStore


def test_report_from_synthetic_cached_artifacts(tmp_path: Path) -> None:
    config = load_config("configs/pythia_160m_two_seed.yaml")
    config.output_dir = str(tmp_path)
    config.config_path = None
    config.bootstrap.samples = 20
    store = ArtifactStore(tmp_path).ensure()
    summary = pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "cka": 0.8,
                "cka_standardized": 0.75,
                "svcca_mean": 0.7,
                "svcca_median": 0.72,
                "pwcca": 0.73,
            }
        ]
    )
    summary.to_csv(store.root / "seed_pair_summary.csv", index=False)
    matrix = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["seed_0", "seed_1"],
        columns=["seed_0", "seed_1"],
    )
    for name in ("cka", "svcca", "pwcca"):
        matrix.to_csv(store.root / f"{name}_matrix.csv")
    overlap = pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "feature_a": 0,
                "feature_b": 1,
                "decoder_cosine": 0.9,
                "jaccard": 0.8,
                "activation_frequency_a": 0.1,
                "activation_frequency_b": 0.11,
            },
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "feature_a": 2,
                "feature_b": 3,
                "decoder_cosine": 0.7,
                "jaccard": 0.5,
                "activation_frequency_a": 0.2,
                "activation_frequency_b": 0.18,
            },
        ]
    )
    overlap.to_parquet(store.root / "activation_overlap.parquet", index=False)
    random_pairs = pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "feature_a": 0,
                "feature_b": 4,
                "matched_feature_b": 1,
                "control_index": 0,
                "frequency_bin": 0,
            },
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "feature_a": 2,
                "feature_b": 5,
                "matched_feature_b": 3,
                "control_index": 0,
                "frequency_bin": 1,
            },
        ]
    )
    random_pairs.to_parquet(store.root / "random_feature_pairs.parquet", index=False)
    control_overlap = pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "feature_a": 0,
                "feature_b": 4,
                "control": "random_pair",
                "jaccard": 0.1,
            },
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "feature_a": 2,
                "feature_b": 5,
                "control": "random_pair",
                "jaccard": 0.2,
            },
        ]
    )
    control_overlap.to_parquet(
        store.root / "control_activation_overlap.parquet", index=False
    )
    feature_rows = []
    for pair_type, features, cosine, jsd, disagreement in (
        ("matched", ((0, 1), (2, 3)), 0.8, 0.05, 0.1),
        ("random_control", ((0, 4), (2, 5)), 0.1, 0.3, 0.6),
    ):
        for feature_a, feature_b in features:
            feature_rows.append(
                {
                    "sae_a": "seed_0",
                    "sae_b": "seed_1",
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "pair_type": pair_type,
                    "mean_logit_delta_cosine": cosine,
                    "mean_ablation_jsd_between_seeds": jsd,
                    "top1_disagreement_rate": disagreement,
                    "mean_effect_jsd_a": 0.1,
                    "mean_effect_jsd_b": 0.1,
                    "informative_effect_fraction": 1.0,
                }
            )
    pd.DataFrame(feature_rows).to_parquet(
        store.root / "ablation_feature_level.parquet", index=False
    )
    prompt = pd.DataFrame(
        [
            {
                "pair_type": "matched",
                "effect_jsd_a": 0.1,
                "effect_jsd_b": 0.1,
                "ablation_jsd_between_seeds": 0.05,
                "top1_disagreement": 0,
            },
            {
                "pair_type": "random_control",
                "effect_jsd_a": 0.1,
                "effect_jsd_b": 0.1,
                "ablation_jsd_between_seeds": 0.3,
                "top1_disagreement": 1,
            },
        ]
    )
    prompt.to_parquet(store.root / "ablation_prompt_level.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "control": "identity",
                "cka": 1.0,
                "svcca": 1.0,
                "pwcca": 1.0,
            }
        ]
    ).to_csv(store.root / "controls_summary.csv", index=False)
    np.savez_compressed(
        store.root / "svcca_correlations" / "seed_0__seed_1.npz",
        correlations=np.array([0.9, 0.8]),
        pca_curve_a=np.array([0.7, 1.0]),
        pca_curve_b=np.array([0.6, 1.0]),
        pwcca_correlations=np.array([0.9, 0.8]),
        pwcca_weights=np.array([0.6, 0.4]),
    )

    report = run(config)
    assert report.exists()
    assert (store.root / "statistical_summary.csv").exists()
    assert (store.root / "plots" / "cka_heatmap.png").exists()
    assert (store.root / "plots" / "canonical_correlation_spectra.svg").exists()
