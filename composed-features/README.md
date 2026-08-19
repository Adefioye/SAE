# Composed-features experiments with SAELens

This directory contains the small composed-feature experiments and their
SAELens comparisons.

## Run

```bash
conda activate sae-composed-features
cd /Users/abdulhakeemadefioye/Desktop/SAE/composed-features
python -m pip install -r requirements.txt
jupyter lab 01_correlated_feature_amplitudes.ipynb
```

Choose the `Python (SAE Composed Features)` kernel in Jupyter.

### Google Colab (Experiment 4/4B)

Open
[`04_fixed_primitive_exposure_budget.ipynb` in Colab](https://colab.research.google.com/github/Adefioye/SAE/blob/main/composed-features/04_fixed_primitive_exposure_budget.ipynb),
select a GPU runtime, and run the notebook's **Environment setup** cell first.
On Colab, that cell clones this repository and installs the complete
`composed-features/requirements.txt` automatically.

The setup cell offers two storage modes:

- `USE_GOOGLE_DRIVE = False` clones into `/content/SAE`; this is convenient for
  a quick pipeline check, but checkpoints disappear when the Colab runtime ends.
- `USE_GOOGLE_DRIVE = True` mounts Drive and clones into
  `MyDrive/SAE`; use this for the long Experiment 4B trajectories so milestone
  checkpoints survive disconnects and can be resumed.

The full Experiment 4B run processes 8.192B samples for each of three SAE
initializations. A normal Colab session may not finish it in one allocation, so
use Drive-backed checkpoints and keep the resumability milestones enabled.

The notebook supports two execution sizes:

- `FULL_RUN = False`: quick model and SAE training.
- `FULL_RUN = True`: 10,000 toy-model batches and 128,000,000 SAE samples.

Set `RUN_GRID = True` to run the 5×5 learning-rate/L1-coefficient grid. Each
grid point trains an independent SAE for every configured seed.

## Experiment 1 files

- `toy_model.py` contains model configuration, the tied-weight network, the
  correlated-pair sampler, toy-model training, and checkpoints.
- `sae.py` contains SAE configuration, the hidden-activation iterator, SAELens
  training, parameter sweeps, and checkpoints.
- `visualization.py` contains evaluation metrics and plots.
- `01_correlated_feature_amplitudes.ipynb` runs training and visualization.
- `test_experiment_1.py` verifies the sampler, hidden activations, and a small
  end-to-end SAELens training run.

## Experiment 3: sparse factorial antipodal geometry

`03_sparse_factorial_antipodal_geometry.ipynb` reuses the trained Experiment 1
dictionary at `N=2`, constructs scalable antipodal dictionaries, and tests
primitive-versus-composition recovery with SAELens `StandardTrainingSAE`.
Long runs are disabled by default. Start with `QUICK_MODE = True`, enable one
`RUN_*` flag at a time, and switch to full mode after the smoke runs succeed.
Checkpoints and CSV results are written beneath
`artifacts/03_sparse_factorial_antipodal/`.

## Experiment 4: fixed primitive exposure at `N=256`

`04_fixed_primitive_exposure_budget.ipynb` tests whether the decline in
large-`N` primitive recovery is caused by the fixed total training budget. It
holds `N=256` fixed and sweeps the expected number of training examples per
primitive, `K = T/N`, along one resumable SAE training trajectory. The
primary `K=16M` milestone matches the successful `N=8`, `T=128M` exposure from
Experiment 3. Review the notebook's compute table and run quick mode before
setting `RUN_EXPERIMENT=True`; the four milestones `K = 8M, 16M, 24M, 32M`
correspond to `T = 2.048B, 4.096B, 6.144B, 8.192B` total samples. The full
trajectory reaches 8M optimizer steps. Checkpoints and results are written beneath
`artifacts/04_fixed_primitive_exposure/`.

## Experiment 5: Standard SAE hyperparameter sweeps

`05_standard_sae_l1_learning_rate_sweeps.ipynb` selects an L1 coefficient and
learning rate for the StandardTraining SAE. Results are written beneath
`artifacts/05_standard_sae_hyperparameter_sweeps/`.

## Experiment 6: TopK SAE learning rate and factorial sparsity

`06_topk_learning_rate_and_sparsity.ipynb` selects a TopK learning rate at
`N=256`, then evaluates primitive recovery across the factorial sparsity sweep.
Results are written beneath `artifacts/06_topk_sae/`.

## Experiment 7: BatchTopK SAE learning rate and factorial sparsity

`07_batchtopk_learning_rate_and_sparsity.ipynb` selects a robust BatchTopK
learning rate across three seeds, then evaluates primitive recovery across the
factorial sparsity sweep. Results are written beneath
`artifacts/07_batchtopk_sae/`.

## Experiment 8: Standard ReLU versus BatchTopK

`08_relu_vs_batchtopk_recovery_comparison.ipynb` compares the Standard ReLU and
BatchTopK recovery results across `N`.

## Correlated feature amplitudes

Each sample has exactly two non-zero features: one of `(x1, x2)` and one of
`(y1, y2)`. The active pair has a shared amplitude sampled uniformly from
`[0, 1)`. The two values are equal to each other but are not generally equal to
1.
