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
            }
        ]
    )
    summary.to_csv(store.root / "seed_pair_summary.csv", index=False)
    matrix = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["seed_0", "seed_1"],
        columns=["seed_0", "seed_1"],
    )
    for name in ("cka", "svcca"):
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
    paper_matches = pd.DataFrame(
        {
            "sae_a": ["seed_0"] * 4,
            "sae_b": ["seed_1"] * 4,
            "feature_a": [0, 1, 2, 3],
            "encoder_feature_b": [1, 0, 3, 2],
            "decoder_feature_b": [1, 0, 2, 3],
            "encoder_matched_cosine": [0.95, 0.91, 0.55, 0.45],
            "decoder_matched_cosine": [0.96, 0.88, 0.58, 0.42],
            "average_matched_cosine": [0.955, 0.895, 0.565, 0.435],
            "encoder_max_cosine": [0.95, 0.92, 0.61, 0.52],
            "decoder_max_cosine": [0.96, 0.90, 0.62, 0.49],
            "same_counterpart": [True, True, False, False],
            "is_shared": [True, True, False, False],
            "is_orphan": [False, False, True, True],
        }
    )
    paper_matches.to_parquet(
        store.root / "paper_hungarian_matches.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "shared_fraction": 0.5,
                "orphan_fraction": 0.5,
            }
        ]
    ).to_csv(store.root / "paper_seed_pair_summary.csv", index=False)
    pd.DataFrame(
        {
            "sae_a": ["seed_0"] * 3,
            "sae_b": ["seed_1"] * 3,
            "threshold": [0.0, 0.7, 1.0],
            "cosine_threshold_fraction": [1.0, 0.5, 0.0],
            "shared_fraction": [0.5, 0.5, 0.0],
            "max_cosine_fraction": [1.0, 0.5, 0.0],
        }
    ).to_csv(store.root / "paper_threshold_sweep.csv", index=False)
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
    pd.DataFrame(
        [
            {
                "sae_a": "seed_0",
                "sae_b": "seed_1",
                "control": "identity",
                "cka": 1.0,
                "svcca": 1.0,
            }
        ]
    ).to_csv(store.root / "controls_summary.csv", index=False)
    np.savez_compressed(
        store.root / "svcca_correlations" / "seed_0__seed_1.npz",
        correlations=np.array([0.9, 0.8]),
        pca_curve_a=np.array([0.7, 1.0]),
        pca_curve_b=np.array([0.6, 1.0]),
    )

    report = run(config)
    assert report.exists()
    assert (store.root / "statistical_summary.csv").exists()
    assert (store.root / "plots" / "cka_heatmap.png").exists()
    canonical_svg = (
        store.root / "plots" / "canonical_correlation_spectra.svg"
    ).read_text()
    assert 'id="legend_' not in canonical_svg
    pca_svg = (store.root / "plots" / "pca_explained_variance_curves.svg").read_text()
    assert "seed_0__seed_1" not in pca_svg
    assert "<!-- A -->" in pca_svg
    assert "<!-- B -->" in pca_svg
    assert (
        store.root / "plots" / "paper_figure_1_encoder_decoder_alignment.png"
    ).exists()
    assert (store.root / "plots" / "paper_figure_a1_threshold_sweep.svg").exists()
    assert (
        store.root / "plots" / "paper_figure_a2_decoder_hungarian_vs_max_cosine.png"
    ).exists()
    assert (store.root / "plots" / "paper_shared_orphan_fractions.svg").exists()
