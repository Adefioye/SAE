from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sae_seed_similarity.config import MatchingConfig  # noqa: E402
from sae_seed_similarity.matching import match_adapters  # noqa: E402


@dataclass
class SyntheticAdapter:
    name: str
    decoder: object

    @property
    def encoder(self):
        return self.decoder

    @property
    def d_in(self) -> int:
        return self.decoder.shape[1]

    @property
    def d_sae(self) -> int:
        return self.decoder.shape[0]

    def encode(self, activations):
        return activations @ self.decoder.T

    def decode(self, latents):
        return latents @ self.decoder


@pytest.mark.parametrize("solver", ["exact", "sparse"])
def test_hungarian_matching_recovers_column_permutation(solver: str) -> None:
    generator = torch.Generator().manual_seed(7)
    decoder = torch.randn((24, 10), generator=generator)
    permutation = torch.randperm(24, generator=generator)
    first = SyntheticAdapter("a", decoder)
    second = SyntheticAdapter("b", decoder[permutation])
    config = MatchingConfig(
        solver=solver,
        exact_max_features=8,
        candidate_top_k=6,
        similarity_batch_size=7,
    )
    result = match_adapters(first, second, config)
    expected_b = torch.argsort(permutation).numpy()
    order = np.argsort(result.feature_a)
    np.testing.assert_array_equal(result.feature_b[order], expected_b)
    assert np.all(result.decoder_cosine > 0.99999)
