"""Shared deterministic, logging, and filesystem helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

import numpy as np

LOGGER = logging.getLogger("sae_seed_similarity")
T = TypeVar("T")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and the RunPod environment report KiB.
    return peak if sys.platform == "darwin" else peak * 1024


@contextmanager
def monitored_operation(
    label: str,
    *,
    heartbeat_seconds: float = 30.0,
) -> Iterator[None]:
    """Log start, periodic liveness/resource heartbeats, and elapsed time."""
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    started = time.monotonic()
    process_started = time.process_time()
    finished = threading.Event()
    LOGGER.info("Started: %s", label)

    def heartbeat() -> None:
        while not finished.wait(heartbeat_seconds):
            elapsed = time.monotonic() - started
            cpu_percent = 100.0 * (time.process_time() - process_started) / max(
                elapsed, 1e-12
            )
            peak_rss = _peak_rss_bytes()
            memory = format_bytes(peak_rss) if peak_rss is not None else "unavailable"
            LOGGER.info(
                "Still running: %s | elapsed=%s | peak_ram=%s | avg_cpu=%.0f%%",
                label,
                format_duration(elapsed),
                memory,
                cpu_percent,
            )

    worker = threading.Thread(
        target=heartbeat,
        name=f"progress-heartbeat:{label}",
        daemon=True,
    )
    worker.start()
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started
        LOGGER.exception("Failed: %s | elapsed=%s", label, format_duration(elapsed))
        raise
    else:
        elapsed = time.monotonic() - started
        peak_rss = _peak_rss_bytes()
        memory = format_bytes(peak_rss) if peak_rss is not None else "unavailable"
        LOGGER.info(
            "Completed: %s | elapsed=%s | peak_ram=%s",
            label,
            format_duration(elapsed),
            memory,
        )
    finally:
        finished.set()
        worker.join(timeout=min(heartbeat_seconds, 1.0))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested in {"cuda", "mps"}:
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(f"Device {requested!r} requires PyTorch") from error
            available = (
                torch.cuda.is_available()
                if requested == "cuda"
                else torch.backends.mps.is_available()
            )
            if not available:
                raise RuntimeError(
                    f"Device {requested!r} was requested but is unavailable"
                )
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def pairwise(items: list[T]) -> Iterator[tuple[T, T]]:
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            yield items[left], items[right]


def stable_sample_indices(n_rows: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or n_rows <= maximum:
        return np.arange(n_rows, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n_rows, maximum, replace=False))


def config_digest(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def scientific_config_digest(data: dict[str, Any]) -> str:
    """Hash result-affecting configuration while ignoring cache-control flags."""
    normalized = dict(data)
    normalized.pop("force", None)
    return config_digest(normalized)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def create_manifest(
    config: dict[str, Any],
    output_dir: Path,
    *,
    resolved_revisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import platform

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": scientific_config_digest(config),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            distribution: _distribution_version(distribution)
            for distribution in (
                "datasets",
                "huggingface-hub",
                "numpy",
                "pandas",
                "pyarrow",
                "sae-lens",
                "scipy",
                "torch",
                "transformer-lens",
                "transformers",
            )
        },
        "resolved_revisions": resolved_revisions or {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_cache_manifest(
    config: dict[str, Any], output_dir: Path, *, force: bool
) -> bool:
    """Return whether an existing manifest matches, or reject stale cache reuse."""
    path = output_dir / "run_manifest.json"
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = scientific_config_digest(config)
    actual = manifest.get("config_sha256")
    if actual != expected and not force:
        raise RuntimeError(
            "The output directory contains artifacts from a different configuration. "
            "Choose a new output_dir or set force: true to recompute all stages."
        )
    return actual == expected
