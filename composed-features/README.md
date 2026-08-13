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

## Correlated feature amplitudes

Each sample has exactly two non-zero features: one of `(x1, x2)` and one of
`(y1, y2)`. The active pair has a shared amplitude sampled uniformly from
`[0, 1)`. The two values are equal to each other but are not generally equal to
1.
