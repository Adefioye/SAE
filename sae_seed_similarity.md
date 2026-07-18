# SAE Seed Similarity

This document describes the seed-similarity analyses implemented by the
`sae_seed_similarity` package: what each technique measures, how the paper-style
Hungarian analysis defines shared and orphan latents, which plots are produced,
how to configure the experiment, and how to interpret the results.

The default experiment configuration is
[`configs/pythia_160m_two_seed.yaml`](configs/pythia_160m_two_seed.yaml). It
compares the final SAELens 6.46 TopK checkpoints at
`kokolamba/pythia-160m-seeds/pythia-160m-500m-two-seed/trained_saes/{seed_0,seed_1}`.
Repository IDs, revisions, checkpoints, hook points, thresholds, sample sizes,
devices, and output paths are configuration values rather than constants in the
code.

The interactive front end is
[`notebooks/sae_seed_similarity_evaluation.ipynb`](notebooks/sae_seed_similarity_evaluation.ipynb).
Every metric is also an importable Python function, so the notebook can load
cached sparse matrices and create new plots without rerunning model inference.

## Experimental requirement

The SAEs being compared should differ only in the factor whose stability is
being studied. For a seed-stability experiment, train them with different SAE
initialization seeds but the same:

- base model, revision, layer, and hook point;
- activation dataset, ordering, and tokenization;
- SAE architecture, latent width, sparsity target, and optimizer settings; and
- training-token budget and checkpoint selection rule.

The supplied two-seed training script uses initialization seeds `0` and `1` on
the same activation batches with data seed `42`. The evaluation collector also
feeds both trained SAEs the exact same cached base-model activations in the same
row order.

## Run the analysis

Install the repository and run the synthetic tests:

```bash
pip install -e '.[dev]'
pytest
```

Run the complete dependency-ordered pipeline:

```bash
python -m sae_seed_similarity.run_all \
  --config configs/pythia_160m_two_seed.yaml
```

Alternatively, run or resume each stage independently:

```bash
python -m sae_seed_similarity.collect_activations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.match_features --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.compare_representations --config configs/pythia_160m_two_seed.yaml
python -m sae_seed_similarity.make_report --config configs/pythia_160m_two_seed.yaml
```

Complete artifacts in `output_dir` are reused. Set `force: true` only when they
should be recomputed. For a quick end-to-end check, copy the YAML and reduce
`dataset.max_sequences`, `cka.max_samples`, `svcca.max_samples`,
`svcca.max_components`, and `bootstrap.samples`.

The tests cover identical representations, permuted axes, orthogonal rotations,
shared low-rank signal, disjoint activation support, paper-style encoder/decoder
assignment agreement, inclusive shared thresholds, configuration validation,
plot generation, and sparse-cache round trips.

## Shared data and matrix shapes

The collector constructs token arrays with shape
`[sequences, sequence_length]`. Padding rows are excluded. Each valid token
occurrence receives a stable row in `activation_rows.parquet` containing
`sequence_id`, `token_position`, `token_id`, `decoded_token`, and
`dataset_index`.

For SAE seed `s`, the post-nonlinearity latent matrix is
`X_s [N_tokens, d_sae_s]`. TopK matrices are saved as SciPy CSR `.npz` files;
the default 32-active-of-32,768 representation therefore remains sparse. The
decoder and encoder dictionaries both have shape `[d_sae, d_model]`.

## Similarity techniques

The techniques answer different questions. No single score establishes that two
SAEs learned the same representation.

### Hungarian feature matching

Hungarian assignment finds a one-to-one mapping between the latents of two SAEs
that maximizes total similarity. It does **not** assume that feature `i` in seed
A should match feature `i` in seed B. A feature may be paired with any available
counterpart, subject to the global one-to-one constraint.

The general matcher supports these objectives:

- `decoder_cosine`: cosine similarity between normalized decoder directions;
- `encoder_cosine`: cosine similarity between normalized encoder directions;
- `activation_correlation`: correlation between token-level latent activations;
  and
- `weighted`: a configured weighted combination of those quantities.

The default production objective is decoder cosine. `solver: exact` constructs
the full similarity matrix and performs exact linear assignment. `solver:
sparse` uses top-k candidate edges and a global sparse assignment. `solver:
auto` chooses according to SAE width and `exact_max_features`.

This general, single-objective matcher is useful for inspecting aligned feature
pairs, but a weighted encoder/decoder objective is not the same as the paper's
shared/orphan definition.

### Paper shared/orphan protocol

When `paper_matching.enabled: true`, the matching stage runs two independent
Hungarian assignments for every equal-width SAE pair:

1. one assignment on normalized encoder directions; and
2. one assignment on normalized decoder directions.

For latent `i` in SAE A, let the assignments select
`encoder_feature_b[i]` and `decoder_feature_b[i]`. Latent `i` is **shared** when
both of the following conditions hold:

1. `encoder_feature_b[i] == decoder_feature_b[i]`; and
2. both independently matched cosine similarities are greater than or equal to
   `paper_matching.shared_threshold`, which is `0.7` by default.

Every other latent is an **orphan** under this two-seed definition. In
particular, the paper protocol does not require `i == j`; it requires the
encoder and decoder assignments to agree on the same cross-seed counterpart
`j`. The threshold is inclusive: a cosine of exactly `0.7` passes.

For exact paper-style analysis, use:

```yaml
matching:
  solver: exact

paper_matching:
  enabled: true
  shared_threshold: 0.7
  threshold_sweep_points: 101
```

### Maximum-cosine baseline

The maximum-cosine baseline independently chooses the most similar feature in
the other SAE for each feature. Unlike Hungarian matching, it is non-bijective:
several features may choose the same counterpart. Comparing it with the
Hungarian result shows how much the one-to-one constraint changes achievable
similarity and exposes feature splitting or duplication.

### Activation overlap

Activation-overlap metrics ask whether a matched feature pair fires on the same
tokens or sequences. The report includes:

- Jaccard similarity;
- overlap coefficient;
- both directional conditional activation probabilities;
- activation-magnitude correlation;
- weighted Jaccard similarity;
- activation frequencies; and
- explicit reasons for empty or degenerate cases.

High dictionary-vector cosine does not guarantee high activation overlap. These
metrics use the shared row correspondence created by the activation collector.

### Linear CKA

Linear centered kernel alignment (CKA) measures shared token geometry across
the complete latent spaces. It supports unequal SAE widths and uses sparse
float64 covariance accumulation. It is computed before and independently of
feature matching, so it can remain high even when no stable one-to-one feature
alignment exists.

Standardized CKA is also available. Standardizing every latent changes the
question by giving rare, low-variance features equal scale, so report whether
`cka.standardize_features` was enabled when comparing runs.

### SVCCA

SVCCA centers each representation, applies PCA/SVD, retains the configured
dominant variance, and then applies ridge-stabilized canonical correlation
analysis. It tests whether the dominant subspaces are similar. It does not show
that individual features align or that either representation is causally used.

## Figures and tables

With paper matching enabled, the report produces the two-seed paper analyses
and useful companion plots:

- **Figure 1:** encoder matched cosine versus decoder matched cosine, colored by
  whether the two assignments agree.
- **Figure A1:** cosine-only, assignment-agreement-plus-threshold, and
  maximum-cosine overlap fractions while sweeping the threshold from 0 to 1.
- **Figure A2:** Hungarian matched cosine versus non-bijective maximum cosine.
- Shared/orphan counts, fractions, and matched-cosine distributions.
- The encoder counterpart of the Hungarian-versus-maximum-cosine plot.
- Similarity versus firing frequency from the cached two-seed activations.

The paper collected firing counts over 10 million tokens for its frequency
figure. With `sequence_length: 128`, set `dataset.max_sequences` to about
`78125` to target that scale. The default `5000`-sequence configuration is a
smaller but methodologically equivalent diagnostic.

The definitions and plotting ideas follow the authors'
[`EleutherAI/sae_overlap`](https://github.com/EleutherAI/sae_overlap)
notebooks, while applying the paper's formal inclusive `>= 0.7` threshold
consistently.

## Controls and uncertainty

Random cross-seed features are approximately matched on activation frequency,
mean nonzero activation, decoder norm, layer, and SAE width. Shuffled-token,
identity, and column-permutation controls validate row correspondence and global
metric invariances.

Bootstrap intervals are computed over feature pairs. Matched-control summaries
include median differences, standardized effect sizes, and paired permutation
tests where applicable. With three or more SAEs, every seed pair is evaluated
and the resulting pair table can itself be bootstrapped across seed pairs.

Two seeds are sufficient for all pairwise figures and shared/orphan estimates,
but they do not estimate how much those results vary across independently
trained seed pairs. Add more seeds when uncertainty across training runs is part
of the research question.

## Configuration guide

The most important analysis settings are:

| Setting | Purpose |
| --- | --- |
| `saes` | Names, formats, revisions, and local or hosted checkpoints to compare |
| `dataset.max_sequences` | Number of shared sequences used to measure activation behavior |
| `activations.encoder_batch_tokens` | Peak token batch encoded by each SAE |
| `matching.method` | Objective for the general single Hungarian assignment |
| `matching.solver` | Exact full-matrix or sparse candidate assignment |
| `matching.candidate_top_k` | Candidate coverage for sparse matching |
| `paper_matching.shared_threshold` | Inclusive encoder and decoder cosine cutoff |
| `paper_matching.threshold_sweep_points` | Resolution of the threshold-sweep plot |
| `cka.max_samples` | Maximum shared token rows used by CKA |
| `svcca.max_samples` | Maximum shared token rows used by SVCCA |
| `svcca.max_components` | Cap on the retained sparse-SVD components |
| `bootstrap.samples` | Number of bootstrap resamples |
| `output_dir` | Cache, table, and plot destination |
| `force` | Recompute complete cached stages when true |

Use exact matching for a faithful paper replication. A 32,768-by-32,768
float32 score matrix occupies 4 GiB before SciPy solver overhead, and the paper
protocol performs the encoder and decoder assignments sequentially. If host RAM
is insufficient, change `matching.solver` to `sparse` or `auto`, increase
`candidate_top_k` as resources permit, and describe the result as an
approximation rather than an exact replication.

## Output artifacts

The configured output directory contains:

```text
run_manifest.json
seed_pair_summary.csv
hungarian_matches.parquet
paper_hungarian_matches.parquet
paper_seed_pair_summary.csv
paper_threshold_sweep.csv
activation_overlap.parquet
cka_matrix.csv
svcca_summary.csv
svcca_correlations/
controls_summary.csv
plots/*.png
plots/*.svg
```

The manifest records the resolved configuration, code commit, platform, and
creation time. Latent CSR matrices, token arrays, row metadata, random-pair
tables, and control-overlap tables are retained as reusable intermediates.

## Memory and runtime

Activation collection requires the base model and at least one SAE on the
selected device. Reduce `activations.encoder_batch_tokens` if dense per-batch
TopK encoder outputs exhaust device memory. CKA uses sparse covariance
identities rather than dense centered matrices. SVCCA uses centered sparse SVD
capped by `max_components`; these are the most expensive representation-level
steps.

Exact matching is primarily constrained by host RAM. Sparse matching reduces
that cost but depends on candidate coverage. Record the solver and candidate
count with every result so exact and approximate runs are not conflated.

## Interpretation

| Result | Supported interpretation |
| --- | --- |
| High Hungarian similarity and high activation overlap | Similar individual directions also fire similarly on shared data |
| High cosine but low activation overlap | Similar dictionary directions have different activation behavior |
| Low feature matching but high CKA | Feature axes differ while global token geometry remains similar |
| Low feature matching but high SVCCA | Dictionaries may span similar dominant subspaces despite unstable individual features |
| Hungarian below maximum-cosine baseline | The one-to-one constraint prevents reuse of preferred counterparts, consistent with duplication or feature splitting |

These are correlational and geometric comparisons. They do not establish that a
feature or representation is causally used by the base model. Causal claims
require interventions such as activation patching, ablation, or feature
steering followed by behavioral evaluation.
