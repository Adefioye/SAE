# SAE

Experiments on seed stability in sparse autoencoders. This repository trains
two SAEs on shared Pythia-160M activations and compares their learned features,
activation behavior, and latent-space geometry.

For the definitions, configuration choices, plots, output schemas, resource
requirements, and interpretation guidance, see
[`sae_seed_similarity.md`](sae_seed_similarity.md).

## Setup

On RunPod, use a PyTorch image with a CUDA GPU and a persistent volume mounted
at `/workspace`. From the repository root:

```bash
bash install_miniconda.sh
bash setup_dev_env.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate sae
pip install -e '.[dev]'
pytest
```

The setup script is safe to rerun. Set `SAE_ENV_NAME` or
`SAE_PYTHON_VERSION` before running it to override the default environment name
or Python 3.11. If Miniconda is installed outside `/workspace`, use the
activation command printed by `install_miniconda.sh`.

## Train the two SAE seeds

Set `WANDB_API_KEY` in the pod environment or a repository-root `.env`, then
run the 500M-token experiment:

```bash
python scripts/train_two_seed_sae.py \
  --training-tokens 500000000 \
  --output-dir /workspace/sae-runs/pythia-160m-500m-two-seed \
  --n-checkpoints 4 \
  --device cuda \
  --log-to-wandb
```

The command trains initialization seeds `0` and `1` on the same activation
batches and writes checkpoints and final SAE weights below `--output-dir`. To
disable Weights & Biases, omit `--log-to-wandb` or pass
`--no-log-to-wandb`.

Resume an interrupted run by repeating the command with the checkpoint folder
that contains both `seed_0` and `seed_1`:

```bash
python scripts/train_two_seed_sae.py \
  --training-tokens 500000000 \
  --output-dir /workspace/sae-runs/pythia-160m-500m-two-seed \
  --n-checkpoints 4 \
  --device cuda \
  --log-to-wandb \
  --resume-from-checkpoint PATH
```

See every training option with:

```bash
python scripts/train_two_seed_sae.py --help
```

## Upload checkpoints

Set `HF_REPO_ID` and `HF_TOKEN` in `.env`, then run:

```bash
python scripts/upload_checkpoints_to_hf.py \
  --checkpoint-dir /workspace/sae-runs/pythia-160m-500m-two-seed/checkpoints \
  --path-in-repo pythia-160m-500m-two-seed/checkpoints

python scripts/upload_checkpoints_to_hf.py \
  --checkpoint-dir /workspace/sae-runs/pythia-160m-500m-two-seed/trained_saes \
  --path-in-repo pythia-160m-500m-two-seed/trained_saes
```

Add `--private` when creating a private destination repository. Rerunning an
interrupted upload skips content already committed.

## Run seed-similarity analysis

The default configuration is
[`configs/pythia_160m_two_seed.yaml`](configs/pythia_160m_two_seed.yaml). Run
the complete pipeline with:

```bash
python -m sae_seed_similarity.run_all \
  --config configs/pythia_160m_two_seed.yaml
```

Or run the independently resumable stages:

```bash
python -m sae_seed_similarity.collect_activations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.match_features --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.compare_representations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.make_report --config configs/pythia_160m_two_seed.yaml
```

The companion notebooks are:

- [`notebooks/research-1/1.2_SAE_two_seed_run.ipynb`](notebooks/research-1/1.2_SAE_two_seed_run.ipynb)
  for the original training workflow; and
- [`notebooks/sae_seed_similarity_evaluation.ipynb`](notebooks/sae_seed_similarity_evaluation.ipynb)
  for interactive analysis and cached plot generation.
