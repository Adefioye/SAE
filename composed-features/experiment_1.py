"""Experiment 1: a small composed model with correlated feature amplitudes.

The toy model and its data sampler are imported directly from
``superposition-geometry-toys/model_fns.py``.  Only the sparse autoencoder and
its training loop are supplied by SAELens.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import torch
from torch import Tensor


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
TOY_SOURCE_DIR = REPOSITORY_ROOT / "superposition-geometry-toys"


def _import_toy_source():
    """Import the original toy-model module without copying its implementation."""
    if not (TOY_SOURCE_DIR / "model_fns.py").is_file():
        raise FileNotFoundError(
            f"Expected the source repository at {TOY_SOURCE_DIR}. "
            "Keep composed-features next to superposition-geometry-toys."
        )
    source_path = str(TOY_SOURCE_DIR)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return importlib.import_module("model_fns")


toy_source = _import_toy_source()
ToyModelConfig = toy_source.ToyModelConfig
ComposedFeatureTMS = toy_source.ComposedFeatureTMS


def default_device() -> torch.device:
    """Select the best available accelerator, falling back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class ToyTrainingConfig:
    """Training settings for TMS"""

    batches: int = 10_000
    batch_size: int = 1_000
    lr: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0


@dataclass(frozen=True)
class SAETrainingConfig:
    """Settings for SAE training on hidden activations of """

    d_sae: int = 4
    training_samples: int = 128_000_000
    batch_size: int = 1_024
    lr: float = 3e-4
    l1_coefficient: float = 0.3
    seed: int = 0
    lr_warm_up_steps: int = 0
    lr_decay_steps: int = 0
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999
    apply_b_dec_to_input: bool = False


FULL_TOY_TRAINING = ToyTrainingConfig()
FULL_SAE_TRAINING = SAETrainingConfig()
SOURCE_GRID_LRS = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
SOURCE_GRID_L1_COEFFICIENTS = (1e-2, 3e-2, 1e-1, 3e-1, 1.0)
SOURCE_INSTANCE_SEEDS = (0, 1)


def make_correlated_amplitude_model(
    training: ToyTrainingConfig = FULL_TOY_TRAINING,
    *,
    device: torch.device | str | None = None,
) -> ComposedFeatureTMS:
    """Construct the exact toy-model setup used by the first experiment.

    One feature is selected from each of two sets.  The pair shares one random
    U[0, 1) magnitude because ``set_magnitude_correlation`` is 1.  Thus the two
    non-zero values are equal to each other, but are not fixed to 1.
    """
    torch.manual_seed(training.seed)
    cfg = ToyModelConfig(
        hidden_size=2,
        feat_sets=(2, 2),
        batches=training.batches,
        batch_size=training.batch_size,
        lr=training.lr,
        wd=training.weight_decay,
        correlated_feature_indices=(1, 1),
        correlated_feature_boost=0,
        set_magnitude_correlation=1,
        active_features_per_draw=1,
    )
    return ComposedFeatureTMS(cfg).to(device or default_device())


def train_toy_model(
    model: ComposedFeatureTMS,
    training: ToyTrainingConfig = FULL_TOY_TRAINING,
) -> list[float]:
    """Train the reused TMS without the source module's notebook-only plotting.

    This small orchestration loop is intentionally local: the source ``TMS``
    class overrides ``nn.Module.train`` with an interactive plotting loop,
    making it unsuitable as a reusable library method.  The forward pass,
    sampler, loss, parameters, and optimizer choice remain identical.
    """
    torch.manual_seed(training.seed)
    torch.nn.Module.train(model, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    losses: list[float] = []
    for _ in range(training.batches):
        optimizer.zero_grad(set_to_none=True)
        features = model.get_batch(training.batch_size)
        reconstruction = model(features)
        loss = model.loss(features, reconstruction, model.I.to(features.device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


class TMSHiddenActivationIterator(Iterator[Tensor]):
    """Adapt ``ComposedFeatureTMS`` batches to SAELens's data-provider protocol."""

    def __init__(self, model: ComposedFeatureTMS, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    def __iter__(self) -> TMSHiddenActivationIterator:
        return self

    @torch.no_grad()
    def __next__(self) -> Tensor:
        features = self.model.get_batch(self.batch_size)
        self.model(features)
        if self.model.h is None:
            raise RuntimeError("The toy model did not expose hidden activations.")
        return self.model.h.detach()


def make_sae(
    config: SAETrainingConfig = FULL_SAE_TRAINING,
    *,
    d_in: int = 2,
    device: torch.device | str | None = None,
):
    """Create the standard L1 SAE used as the SAELens analogue of cell 9."""
    from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

    chosen_device = str(device or default_device())
    torch.manual_seed(config.seed)
    sae_cfg = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=config.d_sae,
        device=chosen_device,
        dtype="float32",
        apply_b_dec_to_input=config.apply_b_dec_to_input,
        normalize_activations="none",
        l1_coefficient=config.l1_coefficient,
        lp_norm=1.0,
        l1_warm_up_steps=config.lr_warm_up_steps,
        decoder_init_norm=1.0,
    )
    return StandardTrainingSAE(sae_cfg).to(chosen_device)


def train_sae(
    model: ComposedFeatureTMS,
    config: SAETrainingConfig = FULL_SAE_TRAINING,
    *,
    device: torch.device | str | None = None,
):
    """Train one SAE entirely through SAELens's ``SAETrainer``."""
    from sae_lens.config import LoggingConfig, SAETrainerConfig
    from sae_lens.training.sae_trainer import SAETrainer

    chosen_device = str(device or default_device())
    sae = make_sae(config, d_in=model.cfg.hidden_size, device=chosen_device)
    trainer_cfg = SAETrainerConfig(
        total_training_samples=config.training_samples,
        train_batch_size_samples=config.batch_size,
        lr=config.lr,
        lr_end=config.lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=config.lr_warm_up_steps,
        lr_decay_steps=config.lr_decay_steps,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        device=chosen_device,
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        logger=LoggingConfig(
            log_to_wandb=False,
            eval_every_n_wandb_logs=2**31 - 1,
        ),
        n_batches_for_norm_estimate=1,
    )
    data_provider = TMSHiddenActivationIterator(model, config.batch_size)
    trainer = SAETrainer(cfg=trainer_cfg, sae=sae, data_provider=data_provider)
    return trainer.fit()


def train_sae_seeds(
    model: ComposedFeatureTMS,
    config: SAETrainingConfig,
    seeds: Sequence[int] = SOURCE_INSTANCE_SEEDS,
    *,
    device: torch.device | str | None = None,
) -> list:
    """Replace the legacy five-instances tensor with independent SAELens runs."""
    return [
        train_sae(
            model,
            SAETrainingConfig(**{**config.__dict__, "seed": seed}),
            device=device,
        )
        for seed in seeds
    ]


def train_sae_grid(
    model: ComposedFeatureTMS,
    base_config: SAETrainingConfig = FULL_SAE_TRAINING,
    *,
    learning_rates: Sequence[float] = SOURCE_GRID_LRS,
    l1_coefficients: Sequence[float] = SOURCE_GRID_L1_COEFFICIENTS,
    seeds: Sequence[int] = SOURCE_INSTANCE_SEEDS,
    device: torch.device | str | None = None,
) -> dict[tuple[float, float], list]:
    """Run the source notebook's 5x5 hyperparameter grid with independent seeds."""
    results: dict[tuple[float, float], list] = {}
    for lr in learning_rates:
        for l1_coefficient in l1_coefficients:
            run_cfg = SAETrainingConfig(
                **{
                    **base_config.__dict__,
                    "lr": lr,
                    "l1_coefficient": l1_coefficient,
                }
            )
            results[(lr, l1_coefficient)] = train_sae_seeds(
                model, run_cfg, seeds=seeds, device=device
            )
    return results


def save_toy_model(model: ComposedFeatureTMS, path: Path | str) -> Path:
    """Save toy-model weights with a small, explicit checkpoint schema."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return destination


def load_toy_model(model: ComposedFeatureTMS, path: Path | str) -> None:
    """Load weights saved by :func:`save_toy_model`."""
    state_dict = torch.load(
        path, map_location=next(model.parameters()).device, weights_only=True
    )
    model.load_state_dict(state_dict)


def save_sae(sae, path: Path | str) -> Path:
    """Use SAELens's native training-model serialization."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    sae.save_model(destination)
    return destination


def load_sae(path: Path | str, *, device: torch.device | str | None = None):
    """Load a native SAELens standard-training checkpoint."""
    from sae_lens import StandardTrainingSAE

    return StandardTrainingSAE.load_from_disk(
        path, device=str(device or default_device()), dtype="float32"
    )
