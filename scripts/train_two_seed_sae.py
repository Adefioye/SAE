#!/usr/bin/env python3
"""Train two TopK SAEs on one shared Pythia activation stream.

This is the command-line version of
``notebooks/research-1/1.2_SAE_two_seed_run.ipynb``. Both SAEs are trained by a
single ``MultiSAETrainingRunner`` so that initialization seed is the only
intended difference between them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from sae_lens import (
    LoggingConfig,
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)


DEFAULT_MODEL = "pythia-160m"
DEFAULT_HOOK = "blocks.6.hook_mlp_out"
DEFAULT_DATASET = (
    "apollo-research/monology-pile-uncopyrighted-tokenizer-"
    "EleutherAI-gpt-neox-20b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two independently initialized TopK SAEs on identical "
            "activation batches using SAELens."
        )
    )
    parser.add_argument(
        "--training-tokens",
        type=int,
        default=500_000_000,
        help="Token budget shared by both SAEs (default: 500000000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pythia-160m-two-seed"),
        help="Root directory for checkpoints and final SAE weights.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=2,
        metavar=("SEED_A", "SEED_B"),
        default=(0, 1),
        help="Exactly two distinct SAE initialization seeds (default: 0 1).",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Seed for the shared data/activation stream (default: 42).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--hook-name", default=DEFAULT_HOOK)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--d-in", type=int, default=768)
    parser.add_argument("--d-sae", type=int, default=32_768)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=2_048)
    parser.add_argument("--batch-tokens", type=int, default=16_384)
    parser.add_argument("--store-batch-size-prompts", type=int, default=2)
    parser.add_argument("--n-batches-in-buffer", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--lr-warm-up-steps", type=int, default=2_000)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="Cosine decay steps; defaults to one fifth of total steps.",
    )
    parser.add_argument("--n-checkpoints", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Multi-SAE checkpoint directory to resume, including its token-count "
            "subdirectory."
        ),
    )
    parser.add_argument("--norm-estimate-batches", type=int, default=1_000)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Training device (default: auto).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file (default: .env).",
    )
    parser.add_argument(
        "--log-to-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable W&B. By default it is enabled when "
            "WANDB_API_KEY is set."
        ),
    )
    parser.add_argument("--wandb-project", default="pythia-160m-seeds")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name; a descriptive name is generated when omitted.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is unavailable")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_args(args: argparse.Namespace) -> None:
    if args.seeds[0] == args.seeds[1]:
        raise ValueError("--seeds must contain two distinct initialization seeds")
    if args.training_tokens < args.batch_tokens:
        raise ValueError("--training-tokens must be at least --batch-tokens")
    if args.training_tokens <= 0 or args.batch_tokens <= 0:
        raise ValueError("token counts must be positive")
    if args.context_size <= 0 or args.batch_tokens % args.context_size != 0:
        raise ValueError("--batch-tokens must be divisible by --context-size")
    if args.d_in <= 0 or args.d_sae <= 0 or args.k <= 0:
        raise ValueError("--d-in, --d-sae, and --k must be positive")
    if args.k > args.d_sae:
        raise ValueError("--k cannot exceed --d-sae")


def token_budget_label(training_tokens: int) -> str:
    if training_tokens % 1_000_000_000 == 0:
        return f"{training_tokens // 1_000_000_000}b"
    if training_tokens % 1_000_000 == 0:
        return f"{training_tokens // 1_000_000}m"
    return str(training_tokens)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.env_file.is_file():
        load_dotenv(args.env_file, override=False)

    device = resolve_device(args.device)
    log_to_wandb = (
        bool(os.getenv("WANDB_API_KEY"))
        if args.log_to_wandb is None
        else args.log_to_wandb
    )
    if log_to_wandb and not os.getenv("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B logging is enabled but WANDB_API_KEY is not set in the "
            "environment or --env-file"
        )

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    trained_sae_dir = output_dir / "trained_saes"
    resume_from_checkpoint = (
        args.resume_from_checkpoint.expanduser().resolve()
        if args.resume_from_checkpoint is not None
        else None
    )
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {resume_from_checkpoint}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trained_sae_dir.mkdir(parents=True, exist_ok=True)

    total_training_steps = args.training_tokens // args.batch_tokens
    lr_decay_steps = (
        args.lr_decay_steps
        if args.lr_decay_steps is not None
        else total_training_steps // 5
    )
    if lr_decay_steps <= 0:
        raise ValueError("--lr-decay-steps must be positive")

    run_name = args.run_name or (
        f"{args.model_name}-topk-{token_budget_label(args.training_tokens)}-"
        f"{len(args.seeds)}-seeds"
    )

    sae_cfgs = {
        f"seed_{seed}": TopKTrainingSAEConfig(
            d_in=args.d_in,
            d_sae=args.d_sae,
            k=args.k,
            apply_b_dec_to_input=False,
            normalize_activations="expected_average_only_in",
        )
        for seed in args.seeds
    }

    initialized_saes = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        name = f"seed_{seed}"
        initialized_saes[name] = TopKTrainingSAE(sae_cfgs[name])

    runner_cfg = MultiSAETrainingRunnerConfig(
        saes=sae_cfgs,
        hook_names=args.hook_name,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        streaming=True,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.batch_tokens,
        store_batch_size_prompts=args.store_batch_size_prompts,
        n_batches_in_buffer=args.n_batches_in_buffer,
        lr=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=args.lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        feature_sampling_window=2_000,
        dead_feature_window=5_000,
        dead_feature_threshold=1e-6,
        device=device,
        seed=args.data_seed,
        dtype="float32",
        autocast=device == "cuda",
        autocast_lm=device == "cuda",
        n_checkpoints=args.n_checkpoints,
        save_final_checkpoint=True,
        checkpoint_path=str(checkpoint_dir),
        resume_from_checkpoint=(
            str(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        ),
        output_path=str(trained_sae_dir),
        n_batches_for_norm_estimate=args.norm_estimate_batches,
        logger=LoggingConfig(
            log_to_wandb=log_to_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=run_name,
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
        ),
    )

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SAE initialization seeds: {list(args.seeds)}")
    print(f"Shared data seed: {args.data_seed}")
    print(f"Training tokens: {args.training_tokens:,}")
    print(f"Training steps (floor): {total_training_steps:,}")
    print(f"Output directory: {output_dir}")
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    print(f"W&B logging: {log_to_wandb}")

    runner = MultiSAETrainingRunner(runner_cfg, override_saes=initialized_saes)
    trained_saes = runner.run()

    print(f"Training complete for: {list(trained_saes)}")
    print(f"Final SAEs: {trained_sae_dir}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
