from __future__ import annotations

import pytest
import torch

from sae import (
    SAETrainingConfig,
    TMSHiddenActivationIterator,
    train_sae,
)
from toy_model import (
    ToyTrainingConfig,
    default_device,
    make_correlated_amplitude_model,
    train_toy_model,
)
from visualization import data_feature_responses, decoder_cosine_similarity, evaluate_sae


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_default_device_precedence(
    monkeypatch, cuda_available, mps_available, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert default_device() == torch.device(expected)


def test_tms_batches_follow_model_device():
    model = make_correlated_amplitude_model(
        ToyTrainingConfig(batches=1), device="cpu"
    )

    assert next(model.parameters()).device.type == "cpu"
    assert model.get_batch(8).device.type == "cpu"


def test_correlated_amplitude_sampler_has_expected_support_and_values():
    model = make_correlated_amplitude_model(ToyTrainingConfig(batches=1, seed=3))
    batch = model.get_batch(2_000).cpu()

    assert torch.equal(model.get_prob_table().cpu(), torch.full((2, 2), 0.25))
    assert torch.all((batch[:, :2] > 0).sum(dim=1) == 1)
    assert torch.all((batch[:, 2:] > 0).sum(dim=1) == 1)
    assert torch.allclose(batch[:, :2].sum(dim=1), batch[:, 2:].sum(dim=1))
    assert batch.max() < 1
    assert batch.std() > 0


def test_hidden_activation_iterator_returns_tms_hidden_states():
    model = make_correlated_amplitude_model(ToyTrainingConfig(batches=1))
    hidden = next(TMSHiddenActivationIterator(model, batch_size=17))
    assert hidden.shape == (17, 2)
    assert not hidden.requires_grad


def test_tiny_toy_and_saelens_training_smoke():
    toy_cfg = ToyTrainingConfig(batches=2, batch_size=16, seed=1)
    model = make_correlated_amplitude_model(toy_cfg)
    losses = train_toy_model(model, toy_cfg)
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)))

    sae_cfg = SAETrainingConfig(
        training_samples=32,
        batch_size=16,
        d_sae=4,
        seed=1,
    )
    sae = train_sae(model, sae_cfg)
    metrics = evaluate_sae(model, sae, samples=32)
    assert set(metrics) == {"mse", "normalized_mse", "mean_l0", "mean_l1"}
    assert data_feature_responses(model, sae).shape == (4, 4)
    assert decoder_cosine_similarity(model, sae).shape == (4, 4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_tiny_toy_and_saelens_training_on_mps():
    toy_cfg = ToyTrainingConfig(batches=2, batch_size=16, seed=1)
    model = make_correlated_amplitude_model(toy_cfg, device="mps")
    losses = train_toy_model(model, toy_cfg)

    assert next(model.parameters()).device.type == "mps"
    assert model.get_batch(8).device.type == "mps"
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)))

    sae_cfg = SAETrainingConfig(
        training_samples=32,
        batch_size=16,
        d_sae=4,
        seed=1,
    )
    sae = train_sae(model, sae_cfg, device="mps")

    assert next(sae.parameters()).device.type == "mps"
