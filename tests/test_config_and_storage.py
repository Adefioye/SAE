from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from sae_seed_similarity.config import load_config
from sae_seed_similarity.storage import ArtifactStore


def test_production_config_loads() -> None:
    config = load_config(Path("configs/pythia_160m_two_seed.yaml"))
    assert [item.name for item in config.saes] == ["seed_0", "seed_1"]
    assert config.base_model.hook_point == "blocks.6.hook_mlp_out"
    assert config.saes[0].format == "sae_lens"


def test_sparse_activation_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path).ensure()
    matrix = sparse.csr_matrix(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))
    store.save_latents("seed", matrix, sparse_matrix=True)
    loaded = store.load_latents("seed")
    np.testing.assert_array_equal(loaded.toarray(), matrix.toarray())
