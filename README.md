# SAE

Experiments on seed stability in sparse autoencoders. The two-seed training
script is the command-line version of
`notebooks/research-1/1.2_SAE_two_seed_run.ipynb` and uses SAELens 6.46.0.

## Run the 500M-token two-seed experiment on RunPod

Use a RunPod PyTorch image with a CUDA GPU and a persistent volume mounted at
`/workspace`. From the repository root, install Miniconda and create the
project environment:

```bash
bash install_miniconda.sh
bash setup_dev_env.sh
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate sae
```

`setup_dev_env.sh` is safe to rerun: it reuses the `sae` environment and
refreshes the packages in `requirements.txt`. Set `SAE_ENV_NAME` or
`SAE_PYTHON_VERSION` before running it to override the default environment name
or Python 3.11. If Miniconda was installed somewhere other than `/workspace`,
use the `source` path printed by `install_miniconda.sh`.

Set `WANDB_API_KEY` in the pod environment or in a repository-root `.env` file,
then run:

```bash
python scripts/train_two_seed_sae.py \
  --training-tokens 500000000 \
  --output-dir /workspace/sae-runs/pythia-160m-500m-two-seed \
  --n-checkpoints 4 \
  --device cuda \
  --log-to-wandb
```

By default, W&B logs to the
[`pythia-160m-seeds`](https://wandb.ai/abdulhakeemadefioye-personal/pythia-160m-seeds)
project under the `abdulhakeemadefioye-personal` entity.

This trains initialization seeds `0` and `1` together on the same activation
batches (shared data seed `42`). It uses Pythia-160M's
`blocks.6.hook_mlp_out`, a 32,768-feature TopK SAE with `k=32`, and the
streaming tokenized Pile dataset from the notebook. Checkpoints and final SAE
weights are written below the supplied `--output-dir`.

The command saves four scheduled checkpoints containing both SAE and optimizer
state, plus the final checkpoint. To upload the complete checkpoint tree, set
`HF_REPO_ID` and `HF_TOKEN` in `.env`, then run:

```bash
python scripts/upload_checkpoints_to_hf.py \
  --checkpoint-dir /workspace/sae-runs/pythia-160m-500m-two-seed/checkpoints \
  --path-in-repo pythia-160m-500m-two-seed/checkpoints

python scripts/upload_checkpoints_to_hf.py \
  --checkpoint-dir /workspace/sae-runs/pythia-160m-500m-two-seed/trained_saes \
  --path-in-repo pythia-160m-500m-two-seed/trained_saes
```

The destination repository is created as a model repository if it does not
already exist. Add `--private` to make a newly created repository private.
Rerun the same upload command after an interruption; Hugging Face resumes the
folder upload and skips content already committed.

If the pod is interrupted, rerun the same command and add
`--resume-from-checkpoint PATH`, where `PATH` is the specific checkpoint
directory containing both `seed_0` and `seed_1` subdirectories.

To run without Weights & Biases, omit `--log-to-wandb` and make sure
`WANDB_API_KEY` is not set, or pass `--no-log-to-wandb`. See all configurable
settings with:

```bash
python scripts/train_two_seed_sae.py --help
```
