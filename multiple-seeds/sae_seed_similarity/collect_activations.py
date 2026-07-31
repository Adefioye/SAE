"""Collect one shared token sample and sparse post-nonlinearity SAE latents."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm.auto import tqdm

from .adapters import load_base_model, load_sae
from .config import EvaluationConfig, load_config
from .storage import ArtifactStore
from .utils import (
    configure_logging,
    create_manifest,
    monitored_operation,
    resolve_device,
    seed_everything,
    validate_cache_manifest,
)

LOGGER = logging.getLogger(__name__)


def _resolve_revisions(config: EvaluationConfig) -> dict[str, Any]:
    """Resolve mutable Hugging Face refs to immutable commit hashes for the manifest."""
    from huggingface_hub import HfApi

    api = HfApi()

    def resolve(kind: str, repo_id: str, revision: str | None) -> dict[str, Any]:
        try:
            info = (
                api.dataset_info(repo_id, revision=revision)
                if kind == "dataset"
                else api.model_info(repo_id, revision=revision)
            )
            return {
                "repo_id": repo_id,
                "configured_revision": revision,
                "commit": info.sha,
            }
        except (
            Exception
        ) as error:  # external metadata should not invalidate cached weights
            LOGGER.warning(
                "Could not resolve %s revision for %s: %s", kind, repo_id, error
            )
            return {
                "repo_id": repo_id,
                "configured_revision": revision,
                "commit": None,
                "resolution_error": f"{type(error).__name__}: {error}",
            }

    return {
        "base_model": resolve(
            "model", config.base_model.repo_id, config.base_model.revision
        ),
        "dataset": resolve("dataset", config.dataset.repo_id, config.dataset.revision),
        "saes": {
            item.name: (
                resolve("model", item.repo_id, item.revision)
                if item.repo_id
                else {"local_path": item.local_path, "checkpoint": item.checkpoint}
            )
            for item in config.saes
        },
    }


def _load_examples(
    cfg: EvaluationConfig, tokenizer: Any
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return token IDs and masks with shape ``[sequences, sequence_length]``."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("datasets is required for activation collection") from error

    kwargs: dict[str, Any] = {
        "path": cfg.dataset.repo_id,
        "split": cfg.dataset.split,
        "streaming": cfg.dataset.streaming,
    }
    if cfg.dataset.revision is not None:
        kwargs["revision"] = cfg.dataset.revision
    dataset = load_dataset(**kwargs)
    if cfg.dataset.streaming:
        dataset = dataset.shuffle(seed=cfg.dataset.random_seed, buffer_size=10_000)
    else:
        generator = np.random.default_rng(cfg.dataset.random_seed)
        order = generator.permutation(len(dataset))[: cfg.dataset.max_sequences]
        dataset = dataset.select(order.tolist())

    length = cfg.dataset.sequence_length
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    bos_id = tokenizer.bos_token_id
    rows: list[list[int]] = []
    source_indices: list[int] = []
    progress = tqdm(total=cfg.dataset.max_sequences, desc="load dataset sequences")
    try:
        for dataset_index, example in enumerate(dataset):
            if len(rows) >= cfg.dataset.max_sequences:
                break
            raw_tokens = example.get(cfg.dataset.token_column)
            if raw_tokens is not None:
                tokens = [int(token) for token in raw_tokens]
                if bos_id is not None and (not tokens or tokens[0] != bos_id):
                    tokens = [int(bos_id), *tokens]
            else:
                text = example.get(cfg.dataset.text_column)
                if not isinstance(text, str) or not text:
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < 2:
                continue
            rows.append(tokens[:length])
            source_indices.append(dataset_index)
            progress.update()
    finally:
        progress.close()
    if not rows:
        raise RuntimeError("The configured dataset yielded no usable token sequences")

    token_ids = np.full((len(rows), length), int(pad_id), dtype=np.int64)
    attention_mask = np.zeros_like(token_ids, dtype=np.int8)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = row
        attention_mask[index, : len(row)] = 1
    return token_ids, attention_mask, source_indices


def _to_sparse_latents(values: torch.Tensor, threshold: float) -> sparse.csr_matrix:
    """Convert dense ``[tokens, d_sae]`` post-activation values to CSR."""
    values = values.detach().to(device="cpu", dtype=torch.float32)
    active = values > threshold
    row, column = active.nonzero(as_tuple=True)
    data = values[row, column].numpy()
    return sparse.csr_matrix(
        (data, (row.numpy(), column.numpy())),
        shape=tuple(values.shape),
        dtype=np.float32,
    )


@torch.inference_mode()
def collect(config: EvaluationConfig) -> ArtifactStore:
    """Collect shared rows and latent matrices, reusing complete cached artifacts."""
    store = ArtifactStore(config.output_path).ensure()
    manifest_matches = validate_cache_manifest(
        config.to_dict(), store.root, force=config.force
    )
    complete = (
        manifest_matches
        and store.row_metadata.exists()
        and store.token_ids.exists()
        and store.attention_mask.exists()
        and all(
            store.latent_path(item.name, True).exists()
            or store.latent_path(item.name, False).exists()
            for item in config.saes
        )
    )
    if complete and not config.force:
        LOGGER.info("Activation cache is complete; reusing %s", store.root)
        return store

    seed_everything(config.dataset.random_seed)
    device = resolve_device(config.base_model.device)
    config.base_model.device = device
    LOGGER.info("Loading base model %s on %s", config.base_model.repo_id, device)
    model = load_base_model(config.base_model)
    model.eval()
    token_ids, attention_mask, dataset_indices = _load_examples(config, model.tokenizer)
    np.save(store.token_ids, token_ids)
    np.save(store.attention_mask, attention_mask)

    flat_valid = attention_mask.reshape(-1).astype(bool)
    sequence_ids, positions = np.nonzero(attention_mask)
    valid_token_ids = token_ids.reshape(-1)[flat_valid]
    decoded = [
        model.tokenizer.decode([int(token)])
        for token in tqdm(valid_token_ids, desc="decode token metadata")
    ]
    metadata = pd.DataFrame(
        {
            "activation_row": np.arange(flat_valid.sum(), dtype=np.int64),
            "sequence_id": sequence_ids.astype(np.int64),
            "token_position": positions.astype(np.int64),
            "token_id": valid_token_ids.astype(np.int64),
            "decoded_token": decoded,
            "dataset_index": np.asarray(dataset_indices, dtype=np.int64)[sequence_ids],
        }
    )
    with monitored_operation("save activation row metadata"):
        metadata.to_parquet(store.row_metadata, index=False)

    adapters = [
        load_sae(item, device=device, dtype=config.base_model.dtype)
        for item in config.saes
    ]
    for adapter in adapters:
        if adapter.d_in != model.cfg.d_model:
            raise ValueError(
                f"SAE {adapter.name} d_in={adapter.d_in} does not match model "
                f"activation width {model.cfg.d_model}"
            )

    chunks: dict[str, list[sparse.csr_matrix] | list[np.ndarray]] = {
        adapter.name: [] for adapter in adapters
    }
    batch_size = config.activations.batch_size
    for start in tqdm(
        range(0, len(token_ids), batch_size), desc="model activation batches"
    ):
        end = min(start + batch_size, len(token_ids))
        tokens = torch.as_tensor(token_ids[start:end], device=device)
        mask = torch.as_tensor(attention_mask[start:end], device=device)
        _, cache = model.run_with_cache(
            tokens,
            attention_mask=mask,
            names_filter=[config.base_model.hook_point],
        )
        activation = cache[config.base_model.hook_point]
        valid = mask.bool().reshape(-1)
        activation = activation.reshape(-1, activation.shape[-1])[valid]
        for adapter in adapters:
            encoded_parts = []
            for token_start in range(
                0, len(activation), config.activations.encoder_batch_tokens
            ):
                encoded_parts.append(
                    adapter.encode(
                        activation[
                            token_start : token_start
                            + config.activations.encoder_batch_tokens
                        ]
                    )
                )
            encoded = torch.cat(encoded_parts, dim=0)
            if (
                config.activations.store_dense
                or config.activations.save_format == "dense_npy"
            ):
                chunks[adapter.name].append(encoded.float().cpu().numpy())
            else:
                chunks[adapter.name].append(
                    _to_sparse_latents(encoded, config.activations.active_threshold)
                )
        del cache, activation

    for adapter in adapters:
        adapter_chunks = chunks[adapter.name]
        if (
            config.activations.store_dense
            or config.activations.save_format == "dense_npy"
        ):
            matrix = np.concatenate(adapter_chunks, axis=0)
            store.save_latents(adapter.name, matrix, sparse_matrix=False)
        else:
            matrix = sparse.vstack(adapter_chunks, format="csr")
            store.save_latents(adapter.name, matrix, sparse_matrix=True)
        LOGGER.info("Saved %s latents with shape %s", adapter.name, matrix.shape)

    create_manifest(
        config.to_dict(),
        store.root,
        resolved_revisions=_resolve_revisions(config),
    )
    return store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    collect(load_config(args.config))


if __name__ == "__main__":
    main()
