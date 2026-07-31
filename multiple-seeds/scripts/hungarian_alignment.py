#!/usr/bin/env python3
"""Compatibility entry point for the configured SAELens Hungarian stage.

The reusable implementation lives in :mod:`sae_seed_similarity.matching`; this
wrapper intentionally replaces the old EleutherAI ``sparsify``-specific script.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    from sae_seed_similarity.match_features import main as match_main

    match_main()


if __name__ == "__main__":
    main()
