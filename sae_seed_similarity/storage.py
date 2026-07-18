"""Cached artifact paths and matrix persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ArtifactStore:
    root: Path

    def ensure(self) -> "ArtifactStore":
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "activations").mkdir(exist_ok=True)
        (self.root / "svcca_correlations").mkdir(exist_ok=True)
        (self.root / "plots").mkdir(exist_ok=True)
        return self

    @property
    def row_metadata(self) -> Path:
        return self.root / "activation_rows.parquet"

    @property
    def token_ids(self) -> Path:
        return self.root / "token_ids.npy"

    @property
    def attention_mask(self) -> Path:
        return self.root / "attention_mask.npy"

    def latent_path(self, sae_name: str, sparse_matrix: bool = True) -> Path:
        suffix = ".npz" if sparse_matrix else ".npy"
        return self.root / "activations" / f"{sae_name}{suffix}"

    def load_latents(self, sae_name: str) -> Any:
        sparse_path = self.latent_path(sae_name, True)
        dense_path = self.latent_path(sae_name, False)
        if sparse_path.exists():
            return sparse.load_npz(sparse_path).tocsr()
        if dense_path.exists():
            return np.load(dense_path, mmap_mode="r")
        raise FileNotFoundError(f"No cached latent matrix for SAE {sae_name!r}")

    def save_latents(self, sae_name: str, matrix: Any, sparse_matrix: bool) -> Path:
        path = self.latent_path(sae_name, sparse_matrix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if sparse_matrix:
            sparse.save_npz(path, sparse.csr_matrix(matrix), compressed=True)
        else:
            np.save(path, np.asarray(matrix, dtype=np.float32))
        return path
