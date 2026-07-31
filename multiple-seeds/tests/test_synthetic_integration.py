from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse

from sae_seed_similarity.config import MatchingConfig
from sae_seed_similarity.matching import match_adapters
from sae_seed_similarity.metrics import activation_overlap, linear_cka


@dataclass
class SyntheticSAE:
    name: str
    decoder: torch.Tensor

    @property
    def encoder(self) -> torch.Tensor:
        return self.decoder

    @property
    def d_in(self) -> int:
        return self.decoder.shape[1]

    @property
    def d_sae(self) -> int:
        return self.decoder.shape[0]

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        return torch.relu(activation @ self.decoder.T)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.decoder


def test_synthetic_sae_pair_end_to_end() -> None:
    generator = torch.Generator().manual_seed(17)
    dictionary = torch.randn((16, 8), generator=generator)
    dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    permutation = torch.randperm(16, generator=generator)
    sae_a = SyntheticSAE("a", dictionary)
    sae_b = SyntheticSAE("b", dictionary[permutation])
    model_activations = torch.randn((200, 8), generator=generator)
    latents_a = sparse.csr_matrix(sae_a.encode(model_activations).numpy())
    latents_b = sparse.csr_matrix(sae_b.encode(model_activations).numpy())

    matches = match_adapters(
        sae_a,
        sae_b,
        MatchingConfig(solver="exact", similarity_batch_size=5),
        latents_a=latents_a,
        latents_b=latents_b,
    )
    expected = torch.argsort(permutation).numpy()
    order = np.argsort(matches.feature_a)
    np.testing.assert_array_equal(matches.feature_b[order], expected)
    assert linear_cka(latents_a, latents_b) > 0.999999
    first_a = int(matches.feature_a[order][0])
    first_b = int(matches.feature_b[order][0])
    assert activation_overlap(latents_a, latents_b, first_a, first_b)["jaccard"] == 1.0
