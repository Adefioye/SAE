from __future__ import annotations

import pytest
import torch

from sparse_factorial_generator import FactorialConfig, SparseFactorialGenerator


def test_correlated_pair_amplitudes_and_support():
    generator = SparseFactorialGenerator(
        FactorialConfig(
            n_x=3,
            n_y=5,
            activation_dim=4,
            amplitude_correlation=1.0,
            singleton_probability=0.0,
            seed=7,
        )
    )

    hidden, coefficients = generator.sample(2_000)

    assert hidden.shape == (2_000, 4)
    assert coefficients.shape == (2_000, 8)
    assert torch.all((coefficients[:, :3] > 0).sum(dim=1) == 1)
    assert torch.all((coefficients[:, 3:] > 0).sum(dim=1) == 1)
    assert torch.allclose(
        coefficients[:, :3].sum(dim=1), coefficients[:, 3:].sum(dim=1)
    )


def test_singleton_probability_one_returns_exactly_one_feature():
    generator = SparseFactorialGenerator(
        FactorialConfig(
            n_x=4,
            n_y=4,
            activation_dim=4,
            amplitude_correlation=1.0,
            singleton_probability=1.0,
            seed=11,
        )
    )

    _, coefficients = generator.sample(4_000)

    assert torch.all((coefficients > 0).sum(dim=1) == 1)
    x_fraction = float((coefficients[:, :4].sum(dim=1) > 0).float().mean())
    assert 0.45 < x_fraction < 0.55


def test_external_dictionary_can_preserve_trained_norms():
    dictionary = torch.tensor(
        [[2.0, 0.0], [-2.0, 0.0], [0.0, 3.0], [0.0, -3.0]]
    )
    cfg = FactorialConfig(n_x=2, n_y=2, activation_dim=2)

    preserved = SparseFactorialGenerator(
        cfg, true_dictionary=dictionary, normalize_dictionary=False
    )
    normalized = SparseFactorialGenerator(cfg, true_dictionary=dictionary)

    assert torch.equal(preserved.true_dictionary, dictionary)
    assert torch.allclose(normalized.true_dictionary.norm(dim=1), torch.ones(4))


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_x": 0},
        {"n_y": 0},
        {"activation_dim": 0},
        {"amplitude_correlation": -0.1},
        {"amplitude_correlation": 1.1},
        {"singleton_probability": -0.1},
        {"singleton_probability": 1.1},
        {"noise_std": -0.1},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        SparseFactorialGenerator(FactorialConfig(**overrides))


def test_nonpositive_batch_size_is_rejected():
    generator = SparseFactorialGenerator(FactorialConfig())

    with pytest.raises(ValueError, match="batch_size"):
        generator.sample(0)
