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


def test_zipf_probabilities_follow_rank_frequency_law():
    generator = SparseFactorialGenerator(
        FactorialConfig(
            n_x=8,
            n_y=8,
            activation_dim=8,
            index_distribution="zipf",
            zipf_exponent=1.0,
            seed=17,
        )
    )

    assert torch.isclose(generator.x_probabilities.sum(), torch.tensor(1.0))
    assert torch.allclose(
        generator.x_probabilities,
        generator.y_probabilities,
    )
    assert torch.isclose(
        generator.x_probabilities[0] / generator.x_probabilities[-1],
        torch.tensor(8.0),
    )


def test_zipf_sampling_matches_expected_primitive_frequencies():
    generator = SparseFactorialGenerator(
        FactorialConfig(
            n_x=8,
            n_y=8,
            activation_dim=8,
            index_distribution="zipf",
            zipf_exponent=1.0,
            seed=23,
        )
    )

    _, coefficients = generator.sample(200_000)
    observed_x = (coefficients[:, :8] > 0).float().mean(dim=0)
    observed_y = (coefficients[:, 8:] > 0).float().mean(dim=0)
    x_indices = coefficients[:, :8].argmax(dim=1)
    y_indices = coefficients[:, 8:].argmax(dim=1)
    observed_pairs = torch.bincount(
        x_indices * 8 + y_indices,
        minlength=64,
    ).reshape(8, 8) / len(coefficients)
    expected_pairs = torch.outer(
        generator.x_probabilities,
        generator.y_probabilities,
    )

    assert torch.allclose(observed_x, generator.x_probabilities, atol=0.003)
    assert torch.allclose(observed_y, generator.y_probabilities, atol=0.003)
    assert torch.allclose(observed_pairs, expected_pairs, atol=0.003)


@pytest.mark.parametrize(
    "overrides",
    [
        {"index_distribution": "not-a-distribution"},
        {"zipf_exponent": -0.1},
        {"zipf_shift": -0.1},
    ],
)
def test_invalid_index_distribution_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        SparseFactorialGenerator(FactorialConfig(**overrides))
