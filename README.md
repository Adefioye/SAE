# SAE

Experiments on seed stability in sparse autoencoders. The two-seed training
script is the command-line version of
`notebooks/research-1/1.2_SAE_two_seed_run.ipynb` and uses SAELens 6.46.0.

## SAE seed evaluation pipeline

The `sae_seed_similarity` package compares independently initialized SAEs at
three distinct levels: individual matched features, the geometry of the full
latent representation, and downstream causal effects. The default configuration
is [configs/pythia_160m_two_seed.yaml](configs/pythia_160m_two_seed.yaml). It
loads the final SAELens 6.46 TopK checkpoints from
`kokolamba/pythia-160m-seeds/pythia-160m-500m-two-seed/trained_saes/{seed_0,seed_1}`.
All repository IDs, revisions, checkpoints, hook points, thresholds, sample
sizes, and devices are configuration values rather than constants in the code.

The interactive front end is
[notebooks/sae_seed_similarity_evaluation.ipynb](notebooks/sae_seed_similarity_evaluation.ipynb).
Every metric is also an importable Python function, so the notebook can load
cached sparse matrices and make new plots without rerunning model inference.

### Install and smoke test

Use the existing environment setup, then install this repository and run the
synthetic test suite:

```bash
pip install -e '.[dev]'
pytest
```

The tests cover identical representations, permuted axes, orthogonal rotations,
shared low-rank signal, disjoint activation support, identical/opposite/zero
ablation effects, configuration validation, and sparse-cache round trips.

### Run the experiment

Each stage is independently resumable and reuses complete artifacts in
`output_dir`. Set `force: true` only when cached outputs should be recomputed.

```bash
python -m sae_seed_similarity.collect_activations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.match_features --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.compare_representations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.run_ablations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.make_report --config configs/pythia_160m_two_seed.yaml
```

Or run the dependency-ordered pipeline:

```bash
python -m sae_seed_similarity.run_all --config configs/pythia_160m_two_seed.yaml
```

For a quick end-to-end experiment, copy the YAML and reduce
`dataset.max_sequences`, `cka.max_samples`, `svcca.max_samples`,
`svcca.max_components`, `ablation.max_feature_pairs`,
`ablation.examples_per_pair`, and `bootstrap.samples`.

### Shared data and matrix shapes

The collector constructs token arrays of shape `[sequences, sequence_length]`.
Padding rows are excluded. Every valid token occurrence receives a stable row in
`activation_rows.parquet` containing `sequence_id`, `token_position`, `token_id`,
`decoded_token`, and `dataset_index`. All SAEs encode the exact same cached base
model activations in the same order.

For SAE seed `s`, its post-nonlinearity latent matrix is
`X_s [N_tokens, d_sae_s]`. TopK matrices are saved as SciPy CSR `.npz`; the
default 32-active-of-32,768 representation therefore remains sparse. Decoder
and encoder dictionaries have shape `[d_sae, d_model]`. Ablation logit vectors
have shape `[vocabulary]` for each selected token and downstream offset.

### What the metrics establish

- Hungarian matching maximizes one-to-one decoder cosine similarity by default;
  it does not assume feature IDs align. Encoder cosine, activation correlation,
  or a weighted combination can be configured. Exact assignment is used for
  smaller dictionaries. The 32k default uses top-k candidate edges followed by
  a global sparse linear assignment, avoiding a 4 GiB similarity matrix.
- Activation overlap measures whether a matched pair fires on the same tokens
  and sequences. It reports Jaccard, overlap coefficients, both conditional
  probabilities, magnitude correlations, weighted Jaccard, frequencies, and
  explicit reasons for empty/degenerate cases.
- Linear CKA measures shared token geometry across the complete latent spaces.
  It supports unequal widths and sparse float64 covariance accumulation. It is
  calculated before and independently of feature matching. Standardized CKA is
  also reported, but standardizing every feature changes the question by giving
  rare low-variance features equal scale.
- SVCCA performs centered PCA/SVD, retains the configured dominant variance,
  then applies ridge-stabilized CCA. It shows dominant shared subspaces, not
  aligned individual features or causal roles.
- Ablation replaces the model activation with each SAE's own full reconstruction.
  For seed A it compares `z_all_A` with `z_-a_A`; seed B is treated separately.
  This controls for unequal reconstruction errors. Logit-delta cosine is the
  primary functional direction metric; base-2 JSD, probability-delta similarity,
  top-1 disagreement, top-k overlap, effect norms, and correlations are also
  saved. Two negligible effects are explicitly labeled inconclusive.

Correlation and geometric similarity do not prove that a representation is
causally used. Conversely, zero ablation can be distribution-shifting and later
layers can self-repair, so causal results should be interpreted with those
limitations in mind.

### Controls and statistics

Random cross-seed features are approximately matched on activation frequency,
mean nonzero activation, decoder norm, layer, and SAE width. Shuffled-token,
identity, and column-permutation controls validate row correspondence and global
metric invariances. Bootstrap intervals are computed over prompt rows and feature
pairs; matched-control summaries include median differences, standardized effect
sizes, and paired permutation tests where applicable. With three or more SAEs,
all seed pairs are evaluated and the resulting pair table can itself be
bootstrapped across seed pairs.

### Outputs

The output directory contains:

```text
run_manifest.json
seed_pair_summary.csv
hungarian_matches.parquet
activation_overlap.parquet
cka_matrix.csv
svcca_summary.csv
svcca_correlations/
ablation_prompt_level.parquet
ablation_feature_level.parquet
controls_summary.csv
plots/*.png
plots/*.svg
```

The manifest captures the full resolved configuration, code commit, platform,
and creation time. Latent CSR matrices, token arrays, row metadata, random-pair
tables, and control overlap tables are retained as reusable intermediates.

### Memory and runtime

Activation collection and ablation require the base model and at least one SAE
on the selected device. Reduce `activations.encoder_batch_tokens` if dense
per-batch TopK encoder outputs exhaust device memory. CKA uses sparse covariance
identities rather than dense centered matrices. SVCCA uses centered sparse
SVD capped by `max_components`; these are the most expensive representation
steps. A 32k exact dense assignment is intentionally avoided in `solver: auto`;
increase `candidate_top_k` to trade memory/runtime for candidate coverage.

### Interpretation table

| Result | Interpretation |
| --- | --- |
| High Hungarian similarity and high activation overlap | Similar individual features and similar firing behavior |
| Low feature matching but high CKA | Different feature axes but similar global token geometry |
| Low feature matching but high SVCCA | Different dictionaries may span similar dominant subspaces |
| High geometric similarity but low ablation similarity | Similar representation geometry but different downstream causal roles |
| High activation overlap and high ablation similarity | Strong evidence that matched features behave similarly |
| Low ablation JSD but negligible individual effects | Inconclusive; both features may simply do nothing |

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
