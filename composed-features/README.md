# Composed-features experiments with SAELens

This directory reimplements one section at a time from
`../superposition-geometry-toys/experiment_2_hid_dim.ipynb`. It currently
contains only Experiment 1: **Small Composed Model — Correlated Feature
Amplitudes**.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
jupyter lab 01_correlated_feature_amplitudes.ipynb
```

The notebook defaults to a quick, reproducible run. Set `FULL_RUN = True` to
use the source notebook's 10,000 TMS batches and 128,000,000 SAE samples. Set
`RUN_GRID = True` to reproduce the 5×5 learning-rate/L1 sweep. The full grid is
computationally expensive because the source used five independent SAE
instances at each point; this implementation maps those instances to five
independent SAELens seeds.

## Provenance

Reused directly from `superposition-geometry-toys/model_fns.py`:

- `ToyModelConfig`
- `ComposedFeatureTMS`
- the composed-pair probability table and sampler
- the tied-weight TMS forward pass and importance-weighted MSE loss
- AdamW and the original TMS hyperparameters

Reused directly from `superposition-geometry-toys/plot_fns.py`:

- `w_cossim` for data-feature/SAE-decoder cosine similarities

Replaced by SAELens 6.47.0:

- the custom `AutoEncoder` and `AutoEncoderConfig`
- SAE forward/loss implementation
- SAE optimizer/trainer and native serialization

Small local adapters fill API gaps:

- a non-interactive TMS training function, because the source's `train` method
  embeds live notebook plotting and overrides `torch.nn.Module.train`
- an iterator that yields `model.h` batches to `SAETrainer`
- SAELens-compatible analysis plots, because the original plot helpers assume
  the old autoencoder's leading `n_inst` weight dimension
- a seed loop, because SAELens represents five runs as five independent models
  instead of one `n_inst=5` parameter tensor

## Meaning of correlated feature amplitudes

Each sample has exactly two non-zero input features: one of `(x1, x2)` and one
of `(y1, y2)`. Their amplitudes are equal to each other because
`set_magnitude_correlation=1`, but the common value is sampled uniformly from
`[0, 1)`. It is therefore not generally equal to 1.
