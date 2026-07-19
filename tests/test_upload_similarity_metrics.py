from __future__ import annotations

from pathlib import Path

import pytest

from sae_seed_similarity.config import load_config
from scripts.upload_similarity_metrics_to_hf import (
    destination_repo,
    discover_metric_artifacts,
    expected_metric_files,
    validate_artifacts,
    validate_repo_path,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "pythia_160m_two_seed.yaml"


def test_discovers_metrics_but_excludes_raw_activation_artifacts(tmp_path: Path) -> None:
    expected = {
        "seed_pair_summary.csv",
        "activation_overlap.parquet",
        "run_manifest.json",
        "report.md",
        "svcca_correlations/seed_0__seed_1.npz",
    }
    for relative in expected | {
        "activation_rows.parquet",
        "activations/seed_0.npz",
        "plots/cka_heatmap.png",
    }:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    discovered = {
        path.relative_to(tmp_path).as_posix()
        for path in discover_metric_artifacts(tmp_path)
    }
    assert discovered == expected


def test_strict_validation_requires_complete_configured_outputs(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    artifacts = []
    for relative in expected_metric_files(config) | {
        "svcca_correlations/seed_0__seed_1.npz"
    }:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        artifacts.append(path)

    assert validate_artifacts(
        config, tmp_path, artifacts, allow_partial=False
    ) == artifacts

    (tmp_path / "cka_matrix.csv").unlink()
    with pytest.raises(RuntimeError, match="cka_matrix.csv"):
        validate_artifacts(
            config,
            tmp_path,
            [path for path in artifacts if path.exists()],
            allow_partial=False,
        )


def test_destination_defaults_to_common_sae_repo_and_rejects_unsafe_path() -> None:
    config = load_config(CONFIG_PATH)
    assert destination_repo(config, None) == "kokolamba/pythia-160m-seeds"
    assert validate_repo_path("similarity_metrics_data") == "similarity_metrics_data"
    with pytest.raises(ValueError, match="safe relative"):
        validate_repo_path("../other-folder")
