"""Run every resumable SAE seed evaluation stage in dependency order."""

from __future__ import annotations

import argparse
from pathlib import Path

from .collect_activations import collect
from .compare_representations import run as compare_representations
from .config import load_config
from .make_report import run as make_report
from .match_features import run as match_features
from .utils import configure_logging, monitored_operation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    stages = (
        ("collect activations", collect),
        ("match features and activation overlap", match_features),
        ("compare representations", compare_representations),
        ("create report and plots", make_report),
    )
    for index, (name, operation) in enumerate(stages, start=1):
        with monitored_operation(
            f"pipeline stage {index}/{len(stages)}: {name}",
            heartbeat_seconds=60.0,
        ):
            operation(config)


if __name__ == "__main__":
    main()
