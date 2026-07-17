#!/usr/bin/env python3
"""Upload a complete SAELens checkpoint tree to a Hugging Face repo folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload all SAE checkpoints and optimizer states beneath a local "
            "checkpoint directory. Authentication is read from .env."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Local checkpoint root produced by train_two_seed_sae.py.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="checkpoints",
        help="Destination folder inside HF_REPO_ID (default: checkpoints).",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Hugging Face repository type (default: model).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch to upload to (default: main).",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Make a newly created repository private or public.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing HF_REPO_ID and HF_TOKEN (default: .env).",
    )
    return parser.parse_args()


def validate_repo_path(value: str) -> str:
    path = PurePosixPath(value.strip())
    if not str(path) or str(path) == ".":
        raise ValueError("--path-in-repo must name a repository folder")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("--path-in-repo must be a safe relative repository path")
    return path.as_posix()


def checkpoint_summary(checkpoint_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in checkpoint_dir.rglob("*"):
        if path.is_file() and ".cache" not in path.relative_to(checkpoint_dir).parts:
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()

    env_file = args.env_file.expanduser().resolve()
    if not env_file.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {env_file}")
    load_dotenv(env_file, override=False)

    repo_id = os.getenv("HF_REPO_ID", "").strip()
    token = os.getenv("HF_TOKEN", "").strip()
    if not repo_id:
        raise RuntimeError(f"HF_REPO_ID is missing from {env_file}")
    if not token:
        raise RuntimeError(f"HF_TOKEN is missing from {env_file}")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    path_in_repo = validate_repo_path(args.path_in_repo)
    file_count, total_bytes = checkpoint_summary(checkpoint_dir)
    if file_count == 0:
        raise RuntimeError(f"No checkpoint files found under {checkpoint_dir}")

    # Hugging Face recommends this setting for maximum Xet upload throughput.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    print(f"Local checkpoint directory: {checkpoint_dir}")
    print(f"Checkpoint files: {file_count:,} ({format_bytes(total_bytes)})")
    print(f"Destination: {args.repo_type} repo {repo_id}/{path_in_repo}")
    print(f"Revision: {args.revision}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type=args.repo_type,
        folder_path=checkpoint_dir,
        path_in_repo=path_in_repo,
        revision=args.revision,
        commit_message="Upload SAELens checkpoints and optimizer states",
        ignore_patterns=[".cache/**", "**/.cache/**"],
    )

    print(f"Upload complete: {result.commit_url}")


if __name__ == "__main__":
    main()
