"""Validated YAML configuration for SAE seed evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml


@dataclass
class BaseModelConfig:
    repo_id: str
    hook_point: str
    revision: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SAEConfig:
    name: str
    checkpoint: str
    repo_id: str | None = None
    revision: str | None = None
    format: Literal["sae_lens", "sparsify", "custom"] = "sae_lens"
    local_path: str | None = None


@dataclass
class DatasetConfig:
    repo_id: str
    split: str = "validation"
    revision: str | None = None
    text_column: str = "text"
    token_column: str = "input_ids"
    sequence_length: int = 128
    max_sequences: int = 5000
    random_seed: int = 42
    streaming: bool = True


@dataclass
class ActivationConfig:
    batch_size: int = 16
    encoder_batch_tokens: int = 512
    active_threshold: float = 0.0
    threshold_mode: Literal["fixed", "positive", "quantile"] = "positive"
    threshold_quantile: float = 0.99
    save_format: Literal["sparse_npz", "dense_npy"] = "sparse_npz"
    store_dense: bool = False


@dataclass
class MatchingConfig:
    method: Literal[
        "decoder_cosine", "encoder_cosine", "activation_correlation", "weighted"
    ] = "decoder_cosine"
    decoder_weight: float = 1.0
    encoder_weight: float = 0.0
    activation_correlation_weight: float = 0.0
    minimum_similarity: float = 0.0
    solver: Literal["auto", "exact", "sparse"] = "auto"
    exact_max_features: int = 8192
    candidate_top_k: int = 256
    similarity_batch_size: int = 1024


@dataclass
class CKAConfig:
    max_samples: int = 50000
    center: bool = True
    standardize_features: bool = False


@dataclass
class SVCCAConfig:
    explained_variance: float = 0.99
    max_components: int = 1024
    max_samples: int = 50000
    ridge: float = 1e-6


@dataclass
class AblationConfig:
    enabled: bool = True
    max_feature_pairs: int = 500
    examples_per_pair: int = 50
    minimum_activation: float = 0.0
    selection_mode: Literal[
        "both_active", "either_active", "top_activating", "top_intersection"
    ] = "both_active"
    intervention: Literal["zero"] = "zero"
    intervention_scope: Literal[
        "selected_token", "active_positions", "all_positions"
    ] = "selected_token"
    evaluation_horizon: int = 1
    top_k: int = 10
    include_clean_logits: bool = True
    minimum_effect_norm: float = 1e-8


@dataclass
class ControlsConfig:
    enabled: bool = True
    random_pairs_per_match: int = 1
    frequency_bins: int = 20


@dataclass
class BootstrapConfig:
    samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class EvaluationConfig:
    base_model: BaseModelConfig
    saes: list[SAEConfig]
    dataset: DatasetConfig
    activations: ActivationConfig = field(default_factory=ActivationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    cka: CKAConfig = field(default_factory=CKAConfig)
    svcca: SVCCAConfig = field(default_factory=SVCCAConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    output_dir: str = "results/sae_seed_similarity"
    force: bool = False
    config_path: Path | None = field(default=None, repr=False)

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir).expanduser()
        if not path.is_absolute() and self.config_path is not None:
            path = self.config_path.parent / path
        return path.resolve()

    def validate(self) -> None:
        if len(self.saes) < 2:
            raise ValueError("Configuration must contain at least two SAEs")
        names = [sae.name for sae in self.saes]
        if len(names) != len(set(names)):
            raise ValueError(f"SAE names must be unique; got {names}")
        if self.dataset.sequence_length < 2 or self.dataset.max_sequences < 1:
            raise ValueError(
                "dataset sequence_length must be >=2 and max_sequences >=1"
            )
        if self.activations.batch_size < 1 or self.activations.encoder_batch_tokens < 1:
            raise ValueError("activation batch sizes must be positive")
        if not 0 < self.activations.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be between zero and one")
        if self.matching.candidate_top_k < 1:
            raise ValueError("matching.candidate_top_k must be positive")
        if (
            self.matching.exact_max_features < 1
            or self.matching.similarity_batch_size < 1
        ):
            raise ValueError("matching feature and batch limits must be positive")
        weights = (
            self.matching.decoder_weight,
            self.matching.encoder_weight,
            self.matching.activation_correlation_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("matching weights must be nonnegative and not all zero")
        if not 0 < self.svcca.explained_variance <= 1:
            raise ValueError("svcca.explained_variance must be in (0, 1]")
        if self.svcca.ridge < 0:
            raise ValueError("svcca.ridge cannot be negative")
        if (
            self.cka.max_samples < 2
            or self.svcca.max_samples < 2
            or self.svcca.max_components < 1
        ):
            raise ValueError("representation sample/component limits are too small")
        if (
            self.ablation.max_feature_pairs < 1
            or self.ablation.examples_per_pair < 1
            or self.ablation.evaluation_horizon < 1
            or self.ablation.top_k < 1
        ):
            raise ValueError("ablation limits and top_k must be positive")
        if self.bootstrap.samples < 1:
            raise ValueError("bootstrap.samples must be positive")
        if not 0 < self.bootstrap.confidence_level < 1:
            raise ValueError("bootstrap.confidence_level must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("config_path", None)
        return result


T = TypeVar("T")


def _construct(cls: type[T], value: dict[str, Any] | None) -> T:
    """Recursively construct one known dataclass while rejecting unknown fields."""
    value = {} if value is None else dict(value)
    allowed = {item.name for item in fields(cls)}  # type: ignore[arg-type]
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)  # type: ignore[arg-type]


def load_config(path: str | Path) -> EvaluationConfig:
    """Load and validate an evaluation config from ``path``."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    known = {item.name for item in fields(EvaluationConfig)} - {"config_path"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level configuration fields: {sorted(unknown)}")
    if not isinstance(raw.get("saes"), list):
        raise ValueError("saes must be a list")
    cfg = EvaluationConfig(
        base_model=_construct(BaseModelConfig, raw.get("base_model")),
        saes=[_construct(SAEConfig, item) for item in raw["saes"]],
        dataset=_construct(DatasetConfig, raw.get("dataset")),
        activations=_construct(ActivationConfig, raw.get("activations")),
        matching=_construct(MatchingConfig, raw.get("matching")),
        cka=_construct(CKAConfig, raw.get("cka")),
        svcca=_construct(SVCCAConfig, raw.get("svcca")),
        ablation=_construct(AblationConfig, raw.get("ablation")),
        controls=_construct(ControlsConfig, raw.get("controls")),
        bootstrap=_construct(BootstrapConfig, raw.get("bootstrap")),
        output_dir=raw.get("output_dir", "results/sae_seed_similarity"),
        force=bool(raw.get("force", False)),
        config_path=config_path,
    )
    cfg.validate()
    return cfg
