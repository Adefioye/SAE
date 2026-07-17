# SAE

Experiments on seed stability in sparse autoencoders. The two-seed training
script is the command-line version of
`notebooks/research-1/1.2_SAE_two_seed_run.ipynb` and uses SAELens 6.46.0.

## Run the 500M-token two-seed experiment on RunPod

Use a RunPod PyTorch image with a CUDA GPU and a persistent volume mounted at
`/workspace`. From the repository root, install the notebook's dependencies:

```bash
python -m pip install "sae-lens==6.46.0" python-dotenv
```

Set `WANDB_API_KEY` in the pod environment or in a repository-root `.env` file,
then run:

```bash
python scripts/train_two_seed_sae.py \
  --training-tokens 500000000 \
  --output-dir /workspace/sae-runs/pythia-160m-500m-two-seed \
  --device cuda \
  --log-to-wandb
```

This trains initialization seeds `0` and `1` together on the same activation
batches (shared data seed `42`). It uses Pythia-160M's
`blocks.6.hook_mlp_out`, a 32,768-feature TopK SAE with `k=32`, and the
streaming tokenized Pile dataset from the notebook. Checkpoints and final SAE
weights are written below the supplied `--output-dir`.

If the pod is interrupted, rerun the same command and add
`--resume-from-checkpoint PATH`, where `PATH` is the specific checkpoint
directory containing both `seed_0` and `seed_1` subdirectories.

To run without Weights & Biases, omit `--log-to-wandb` and make sure
`WANDB_API_KEY` is not set, or pass `--no-log-to-wandb`. See all configurable
settings with:

```bash
python scripts/train_two_seed_sae.py --help
```
