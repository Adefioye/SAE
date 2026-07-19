#!/usr/bin/env python3
"""Upload completed SAE seed-similarity metric artifacts to Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sae_seed_similarity.config import EvaluationConfig, load_config


REQUIRED_METRIC_FILES = {
    "activation_overlap.parquet",
    "cka_matrix.csv",
    "hungarian_matches.parquet",
    "run_manifest.json",
    "seed_pair_summary.csv",
    "svcca_matrix.csv",
    "svcca_summary.csv",
}
PAPER_METRIC_FILES = {
    "paper_hungarian_matches.parquet",
    "paper_seed_pair_summary.csv",
    "paper_threshold_sweep.csv",
}
CONTROL_METRIC_FILES = {
    "control_activation_overlap.parquet",
    "controls_summary.csv",
    "random_feature_pairs.parquet",
}
REPORT_FILES = {"report.md", "statistical_summary.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload completed CSV, Parquet, and SVCCA NPZ metric artifacts to "
            "the Hugging Face repository containing the evaluated SAE seeds."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Evaluation YAML used to produce the metric artifacts.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Override the config's output_dir when locating local artifacts.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "Destination model repository. By default, require and use the one "
            "common repo_id configured for all evaluated SAEs."
        ),
    )
    parser.add_argument(
        "--path-in-repo",
        default="similarity_metrics_data",
        help=(
            "Destination folder inside the model repository "
            "(default: similarity_metrics_data)."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch to upload to; defaults to the common SAE revision or main.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing HF_TOKEN (default: .env).",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Make a newly created repository private or public.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload SAE seed similarity metric artifacts",
        help="Hugging Face commit message.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Upload available metric files even when expected outputs are missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list selected artifacts without contacting Hugging Face.",
    )
    return parser.parse_args()


def validate_repo_path(value: str) -> str:
    path = PurePosixPath(value.strip())
    if not str(path) or str(path) == ".":
        raise ValueError("--path-in-repo must name a repository folder")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("--path-in-repo must be a safe relative repository path")
    return path.as_posix()


def destination_repo(config: EvaluationConfig, override: str | None) -> str:
    if override is not None and override.strip():
        return override.strip()
    repo_ids = {sae.repo_id for sae in config.saes if sae.repo_id}
    if len(repo_ids) != 1 or any(sae.repo_id is None for sae in config.saes):
        raise ValueError(
            "The evaluated SAEs do not share exactly one remote repo_id; pass "
            "--repo-id explicitly"
        )
    return next(iter(repo_ids))


def destination_revision(config: EvaluationConfig, override: str | None) -> str:
    if override is not None and override.strip():
        return override.strip()
    revisions = {sae.revision or "main" for sae in config.saes}
    return next(iter(revisions)) if len(revisions) == 1 else "main"


def expected_metric_files(config: EvaluationConfig) -> set[str]:
    expected = set(REQUIRED_METRIC_FILES) | REPORT_FILES
    if config.paper_matching.enabled:
        expected |= PAPER_METRIC_FILES
    if config.controls.enabled:
        expected |= CONTROL_METRIC_FILES
    return expected


def discover_metric_artifacts(metrics_dir: Path) -> list[Path]:
    """Return plot-reproducibility artifacts, excluding raw activation data."""
    artifacts: set[Path] = set()
    for pattern in ("*.csv", "*.parquet", "*.npz"):
        artifacts.update(path for path in metrics_dir.glob(pattern) if path.is_file())
    for name in ("run_manifest.json", "report.md"):
        path = metrics_dir / name
        if path.is_file():
            artifacts.add(path)
    spectra_dir = metrics_dir / "svcca_correlations"
    if spectra_dir.is_dir():
        artifacts.update(path for path in spectra_dir.rglob("*.npz") if path.is_file())

    # This root-level Parquet is token metadata rather than a computed metric.
    artifacts.discard(metrics_dir / "activation_rows.parquet")
    return sorted(artifacts, key=lambda path: path.relative_to(metrics_dir).as_posix())


def validate_artifacts(
    config: EvaluationConfig,
    metrics_dir: Path,
    artifacts: Iterable[Path],
    *,
    allow_partial: bool,
) -> list[Path]:
    selected = list(artifacts)
    if not selected:
        raise RuntimeError(f"No metric artifacts found under {metrics_dir}")
    if allow_partial:
        return selected

    relative = {path.relative_to(metrics_dir).as_posix() for path in selected}
    missing = sorted(expected_metric_files(config) - relative)
    if missing:
        formatted = "\n  - ".join(missing)
        raise RuntimeError(
            "The metric run is incomplete; expected artifacts are missing:\n"
            f"  - {formatted}\n"
            "Finish the metric/report stages or pass --allow-partial intentionally."
        )
    if not any(name.startswith("svcca_correlations/") for name in relative):
        raise RuntimeError(
            "No svcca_correlations/*.npz artifacts were found; finish the SVCCA "
            "stage or pass --allow-partial intentionally."
        )
    return selected


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics_dir = (
        args.metrics_dir.expanduser().resolve()
        if args.metrics_dir is not None
        else config.output_path
    )
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Metrics directory does not exist: {metrics_dir}")

    repo_id = destination_repo(config, args.repo_id)
    revision = destination_revision(config, args.revision)
    path_in_repo = validate_repo_path(args.path_in_repo)
    artifacts = validate_artifacts(
        config,
        metrics_dir,
        discover_metric_artifacts(metrics_dir),
        allow_partial=args.allow_partial,
    )
    total_bytes = sum(path.stat().st_size for path in artifacts)

    print(f"Local metrics directory: {metrics_dir}")
    print(f"Metric artifacts: {len(artifacts):,} ({format_bytes(total_bytes)})")
    print(f"Destination: model repo {repo_id}/{path_in_repo}")
    print(f"Revision: {revision}")
    for path in artifacts:
        print(f"  {path.relative_to(metrics_dir).as_posix()}")
    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return

    env_file = args.env_file.expanduser().resolve()
    if not env_file.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {env_file}")
    try:
        from dotenv import load_dotenv
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as error:
        raise RuntimeError(
            "python-dotenv and huggingface-hub are required for uploads; reinstall "
            "the project dependencies"
        ) from error
    load_dotenv(env_file, override=False)
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(f"HF_TOKEN is missing from {env_file}")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    prefix = PurePosixPath(path_in_repo)
    operations = [
        CommitOperationAdd(
            path_in_repo=(prefix / path.relative_to(metrics_dir).as_posix()).as_posix(),
            path_or_fileobj=str(path),
        )
        for path in artifacts
    ]

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    result = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        operations=operations,
        commit_message=args.commit_message,
    )
    print(f"Upload complete: {result.commit_url}")


if __name__ == "__main__":
    main()
