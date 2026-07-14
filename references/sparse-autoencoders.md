# Sparse Autoencoders (SAEs)

## Table of Contents
1. [What SAEs Do](#what-saes-do)
2. [Architecture](#architecture)
3. [Architecture Variants](#architecture-variants)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Using SAE Features](#using-sae-features)
7. [Common Issues](#common-issues)
8. [Open Problems](#open-problems)

---

## What SAEs Do

Sparse autoencoders decompose model activations in superposition into sparse, interpretable features. The model encodes many more features than it has dimensions; SAEs learn an overcomplete dictionary that recovers these features.

**Core idea:** Train a one-hidden-layer autoencoder to reconstruct model activations, with a sparsity penalty forcing the hidden layer to use few features per input. Each hidden unit (feature) ideally corresponds to one interpretable concept.

**What you get:**
- **Encoder:** Maps activations to sparse feature coefficients (which features are active and how strongly)
- **Decoder:** A dictionary of feature vectors — each row is a direction in activation space
- **Feature activations:** For any input, a sparse vector saying which features are present

---

## Architecture

### Standard (ReLU) SAE

```
h(z) = ReLU((z - b_dec) @ W_enc + b_enc)    # Sparse feature activations
SAE(z) = h(z) @ W_dec + b_dec                 # Reconstruction
```

**Loss:** `||z - SAE(z)||² + α * ||h(z)||₁`

- First term: reconstruction quality
- Second term: sparsity penalty (L1 on feature activations)
- `α` controls the sparsity-reconstruction tradeoff

---

## Architecture Variants

### Gated SAE
Decouples feature detection (gate) from magnitude estimation. Uses a separate gate mechanism to decide which features are active, then estimates their magnitude independently. This avoids **shrinkage** — the L1 penalty biasing feature magnitudes toward zero.

Achieves similar reconstruction with ~half as many firing features as standard SAEs.

### TopK SAE
Replaces the ReLU + L1 penalty with a hard top-k selection: keep only the k largest features, zero the rest. This:
- Eliminates the need to tune the L1 coefficient
- Gives exact control over sparsity (exactly k features per input)
- Scales well and is competitive with Gated SAEs

**BatchTopK** variant: apply top-k across the flattened batch instead of per-sample, allowing variable sparsity per input.

### JumpReLU SAE
Replaces ReLU with a JumpReLU activation that has a learnable threshold per feature. Features below the threshold are exactly zero; above it, the output equals the input. This gives finer-grained sparsity control.

### Transcoders
Not autoencoders — they map MLP *input* to MLP *output* through a sparse bottleneck. This learns a sparse, interpretable replacement for an MLP layer, making circuit analysis through MLPs much easier. Performance is comparable to SAEs but enables weights-based (not just intervention-based) circuit analysis.

---

## Training

### Key Decisions

- **Where to train:** On residual stream activations (most common), attention layer outputs (useful for head-level analysis), or MLP outputs
- **Width:** The hidden dimension m should be larger than the model dimension d (overcomplete). Common ratios: 4x to 64x.
- **Sparsity target:** Typical L0 of 10-100 active features per input, depending on width and layer
- **Training data:** Use a diverse sample of the model's pretraining distribution

### Training Tips

- **Monitor dead features:** Features that never activate waste capacity. Standard remedy: periodically reinitialize dead features with activations that the SAE currently reconstructs poorly.
- **Decoder normalization:** Keep decoder vectors unit-norm to prevent the model from increasing reconstruction quality by inflating decoder norms rather than learning better features.
- **Learning rate:** SAEs are sensitive to learning rate. Warmup helps.
- **The L1 coefficient α** trades off sparsity vs. reconstruction. Too high → too sparse, poor reconstruction. Too low → dense, uninterpretable features.

---

## Evaluation

### Standard Metrics

| Metric | What It Measures | Good Values |
|--------|-----------------|-------------|
| **L0 norm** | Average active features per input | Task-dependent; lower = sparser |
| **Loss recovered** | % of original cross-entropy loss preserved with SAE reconstructions | >95% is good |
| **Feature density histogram** | Distribution of per-feature firing rates | Should be spread; avoid clumps at 0 or 1 |
| **Dead features** | % of features that never activate | <5% ideally |

### Interpretability Assessment

- **Max-activating examples:** For each feature, find the inputs where it activates most strongly. Look for interpretable patterns.
- **Logit attribution:** Project each feature's decoder vector through the unembedding to see what tokens it promotes/suppresses.
- **Automated scoring:** Use an LLM to generate natural language descriptions of features from their max-activating examples.
- **Feature density vs. interpretability:** Features that fire on ~0.01-1% of tokens tend to be most interpretable; very dense or very sparse features are harder.

### Caution: SAE Quality Is an Open Problem

There is no universally agreed-upon metric for SAE quality. Loss recovered and L0 give a rough picture, but two SAEs with identical metrics can have very different feature quality. Always do qualitative checks.

---

## Using SAE Features

### For Circuit Analysis
- Use SAE features as nodes in circuits instead of raw components
- Attribute edges between features using attribution patching
- Features provide more interpretable circuit descriptions than raw heads/neurons

### For Steering
- SAE features can be used as steering vectors by adding/subtracting decoder directions
- However, simple difference-in-means steering vectors often work just as well — SAEs are not necessarily needed for steering

### For Understanding
- Browse features on Neuronpedia for existing SAEs (GPT-2 Small, etc.)
- Feature dashboards show: max-activating examples, logit effects, density, and auto-descriptions

---

## Common Issues

1. **Dead features:** Features that never activate. Fix with periodic reinitialization or use architecture variants (Gated, TopK) that handle this better.

2. **Feature splitting:** A single concept spread across multiple features at different granularities. Not necessarily a problem — can reflect genuine hierarchical structure.

3. **Reconstruction error is systematic:** SAE errors are not random noise — they systematically shift next-token predictions. This means analyses relying on SAE features may miss systematic effects captured in the error term.

4. **Scaling challenges:** SAE quality may degrade on larger models. The GPT-4 SAE features (Gao et al.) were noted to be less interpretable than GPT-2 features. This is an active area of research.

---

## Open Problems

- **Better evaluation metrics:** How to reliably compare SAE architectures and training procedures
- **Feature completeness:** Do SAEs find all important features? (Sigmoid relationship between concept frequency and probability of learning it)
- **Scalability:** Training high-quality SAEs on frontier models
- **Integration with circuits:** Systematic methods for building circuits from SAE features
- **Real-world applications:** Demonstrating that SAE-based analysis outperforms simpler methods on practical tasks

---

*Adapted from "Towards Monosemanticity" (Bricken et al., Anthropic), "Scaling and Evaluating Sparse Autoencoders" (Gao et al., OpenAI), Rajamanoharan et al. (Gated SAEs), and the ARENA SAE curriculum (2025).*
