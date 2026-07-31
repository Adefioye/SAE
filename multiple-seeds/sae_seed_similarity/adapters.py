"""Checkpoint adapters exposing a small format-independent SAE interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SAEConfig


class SAEAdapter(ABC):
    """Interface used by activation collection and feature matching."""

    name: str

    @property
    @abstractmethod
    def d_in(self) -> int: ...

    @property
    @abstractmethod
    def d_sae(self) -> int: ...

    @property
    @abstractmethod
    def decoder(self) -> Any:
        """Return decoder directions with shape ``[d_sae, d_in]``."""

    @property
    @abstractmethod
    def encoder(self) -> Any | None:
        """Return encoder directions as ``[d_sae, d_in]`` when available."""

    @abstractmethod
    def encode(self, activations: Any) -> Any:
        """Encode ``[..., d_in]`` activations to post-nonlinearity latents."""

    @abstractmethod
    def decode(self, latents: Any) -> Any:
        """Decode post-nonlinearity ``[..., d_sae]`` latents to activations."""


@dataclass
class SAELensAdapter(SAEAdapter):
    name: str
    sae: Any
    source: str

    @property
    def d_in(self) -> int:
        return int(self.sae.cfg.d_in)

    @property
    def d_sae(self) -> int:
        return int(self.sae.cfg.d_sae)

    @property
    def decoder(self) -> Any:
        return self.sae.W_dec

    @property
    def encoder(self) -> Any:
        return self.sae.W_enc.T

    def encode(self, activations: Any) -> Any:
        return self.sae.encode(activations)

    def decode(self, latents: Any) -> Any:
        return self.sae.decode(latents)


def load_sae(config: SAEConfig, device: str, dtype: str = "float32") -> SAEAdapter:
    """Load one configured SAE without modifying its source checkpoint."""
    if config.format != "sae_lens":
        raise NotImplementedError(
            f"Checkpoint format {config.format!r} is not available in this repository. "
            "Implement SAEAdapter for that format or set format: sae_lens."
        )
    try:
        from sae_lens import SAE
    except ImportError as error:
        raise RuntimeError(
            "sae-lens is required to load configured checkpoints"
        ) from error

    if config.local_path:
        path = Path(config.local_path).expanduser().resolve() / config.checkpoint
        if not path.is_dir():
            raise FileNotFoundError(f"SAE checkpoint directory does not exist: {path}")
        sae = SAE.load_from_disk(path, device=device, dtype=dtype)
        return SAELensAdapter(config.name, sae, str(path))
    if not config.repo_id:
        raise ValueError(f"SAE {config.name!r} requires repo_id or local_path")

    # Download only inference files so arbitrary Hugging Face revisions are honored.
    # The snapshot cache is content-addressed and the remote checkpoint is read-only.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required for remote checkpoints"
        ) from error
    snapshot = Path(
        snapshot_download(
            repo_id=config.repo_id,
            revision=config.revision,
            allow_patterns=[
                f"{config.checkpoint}/cfg.json",
                f"{config.checkpoint}/sae_weights.safetensors",
                f"{config.checkpoint}/sparsity.safetensors",
            ],
        )
    )
    checkpoint_path = snapshot / config.checkpoint
    sae = SAE.load_from_disk(checkpoint_path, device=device, dtype=dtype)
    source = f"hf://{config.repo_id}@{config.revision or 'main'}/{config.checkpoint}"
    return SAELensAdapter(config.name, sae, source)


def load_base_model(config: Any) -> Any:
    """Load a TransformerLens model with checkpoint-compatible processing."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError as error:
        raise RuntimeError(
            "transformer-lens is required to load the base model"
        ) from error
    kwargs = dict(config.model_from_pretrained_kwargs)
    if config.revision is not None:
        kwargs["revision"] = config.revision
    return HookedTransformer.from_pretrained_no_processing(
        config.repo_id,
        device=config.device,
        dtype=config.dtype,
        **kwargs,
    )
