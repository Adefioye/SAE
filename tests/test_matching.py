from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sae_seed_similarity.config import MatchingConfig  # noqa: E402
from sae_seed_similarity.matching import (  # noqa: E402
    classify_shared_latents,
    match_adapters,
    match_paper_shared_latents,
)


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


@dataclass
class PaperSyntheticAdapter:
    name: str
    decoder: object
    encoder: object

    @property
    def d_in(self) -> int:
        return self.decoder.shape[1]

    @property
    def d_sae(self) -> int:
        return self.decoder.shape[0]


def test_paper_matching_recovers_shared_permutation() -> None:
    directions = torch.eye(5)
    permutation = torch.tensor([2, 4, 0, 1, 3])
    first = PaperSyntheticAdapter("a", directions, directions)
    second = PaperSyntheticAdapter(
        "b", directions[permutation], directions[permutation]
    )
    result = match_paper_shared_latents(
        first,
        second,
        MatchingConfig(solver="exact", similarity_batch_size=2),
    )
    expected = torch.argsort(permutation).numpy()
    np.testing.assert_array_equal(result.encoder_feature_b, expected)
    np.testing.assert_array_equal(result.decoder_feature_b, expected)
    assert result.same_counterpart.all()
    assert result.is_shared.all()
    assert not result.is_orphan.any()
    assert np.all(result.encoder_max_cosine > 0.99999)
    assert np.all(result.decoder_max_cosine > 0.99999)


def test_paper_matching_disagreement_is_orphan() -> None:
    directions = torch.eye(4)
    first = PaperSyntheticAdapter("a", directions, directions)
    second = PaperSyntheticAdapter(
        "b",
        directions,
        directions[torch.tensor([1, 2, 3, 0])],
    )
    result = match_paper_shared_latents(
        first,
        second,
        MatchingConfig(solver="exact", similarity_batch_size=2),
    )
    assert not result.same_counterpart.any()
    assert not result.is_shared.any()
    assert result.is_orphan.all()


def test_shared_threshold_is_inclusive_and_requires_both_cosines() -> None:
    same, shared = classify_shared_latents(
        np.array([3, 4, 5]),
        np.array([3, 4, 5]),
        np.array([0.7, 0.69, 0.9]),
        np.array([0.7, 0.9, 0.69]),
        threshold=0.7,
    )
    np.testing.assert_array_equal(same, [True, True, True])
    np.testing.assert_array_equal(shared, [True, False, False])
