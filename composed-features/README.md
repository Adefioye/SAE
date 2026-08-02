# Composed-features experiments with SAELens

This directory currently contains Experiment 1: **Small Composed Model —
Correlated Feature Amplitudes**.

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

## Correlated feature amplitudes

Each sample has exactly two non-zero features: one of `(x1, x2)` and one of
`(y1, y2)`. The active pair has a shared amplitude sampled uniformly from
`[0, 1)`. The two values are equal to each other but are not generally equal to
1.
