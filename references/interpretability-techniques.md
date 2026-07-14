# Interpretability Techniques

## Table of Contents
1. [Direct Logit Attribution](#direct-logit-attribution)
2. [Activation Patching](#activation-patching)
3. [Attribution Patching](#attribution-patching)
4. [Linear Probing](#linear-probing)
5. [Steering Vectors](#steering-vectors)
6. [Automated Circuit Discovery](#automated-circuit-discovery)
7. [Distributed Alignment Search](#distributed-alignment-search)
8. [Choosing the Right Technique](#choosing-the-right-technique)

---

## Direct Logit Attribution

**What:** Decompose the model's output logits into contributions from each component (attention heads, MLPs, embeddings).

**How it works:** Since the residual stream is a sum of component outputs, and logits are a linear function of the residual stream, each component's contribution to any logit is just its output projected through the unembedding matrix.

**When to use:**
- First pass analysis — understand which components matter for a specific prediction
- Quick and exact (no approximation)
- Works for any component that writes to the residual stream

**Limitations:**
- Only captures *direct* effects on the logits
- Misses indirect effects where a component influences later components that then affect logits
- Components close to the final layer have more direct effect; early components often work indirectly

**Best practices:**
- Use the "logit difference" between the correct and incorrect token as the metric (not raw logits)
- Plot per-head logit attribution as a heatmap across layers and heads
- Follow up with patching to confirm causal importance

---

## Activation Patching

**What:** Replace a component's activation on one input with its activation from a different input. Measure the effect on the output. This is the gold standard for causal analysis.

**Also known as:** Causal mediation analysis, interchange interventions, causal tracing, resample ablations.

**How it works:**
1. Choose a **clean** input (where the model behaves correctly) and a **corrupted** input (different in the specific way you care about)
2. Run the model on both inputs and cache all activations
3. Replace one component's activation on the clean input with its value from the corrupted input
4. Measure how much the output changes

**Key insight — use contrast pairs:** Pick inputs that differ *only* in the behavior you're studying. For example, "Michael Jordan plays the sport of" vs. "Babe Ruth plays the sport of" — this controls for "is about sports" while isolating "which sport."

**Variants:**
- **Zero ablation:** Set the component to zero (harsh, can produce out-of-distribution activations)
- **Mean ablation:** Replace with the mean over a dataset (less harsh)
- **Resample ablation:** Replace with the value from a random different input (preserves activation statistics)

**Limitations:**
- Requires one forward pass per component — slow for large models
- Self-repair (hydra effect) can mask the true importance of components
- Results depend on the choice of clean/corrupted inputs

---

## Attribution Patching

**What:** A gradient-based approximation to activation patching that evaluates all components simultaneously.

**How it works:** Approximate the effect of patching component C from input A into input B as:

```
effect ≈ (activation_A - activation_B) · gradient_B
```

This requires only two forward passes and one backward pass, regardless of the number of components.

**When to use:**
- When activation patching is too slow (large models, many components)
- As a first pass to identify promising components, then validate with full patching
- When you need to evaluate all heads and MLPs at once

**Limitations:**
- Approximation quality varies — less reliable at early layers and with saturated attention softmaxes
- The AtP* paper shows it's the best use of a limited compute budget among several gradient-based variants
- Integrated gradients (Marks et al.) is slower but more reliable, especially at early layers

---

## Linear Probing

**What:** Train a linear classifier on model activations to predict whether a feature is present.

**How it works:** Extract activations at a specific layer for inputs with/without a feature. Train a logistic regression or linear classifier to distinguish them. The probe's accuracy indicates how much information about the feature is linearly encoded.

**When to use:**
- Testing the linear representation hypothesis for a specific feature
- Quick check of whether information is present at a layer
- Finding the direction associated with a feature

**Critical limitation — correlation, not causation:** A high-accuracy probe shows information is *encoded* but NOT that the model *uses* it. Always follow up with causal interventions.

**Best practices:**
- Use control tasks (randomized labels) to calibrate probe accuracy
- Sparse probes (at most k non-zero weights) provide evidence about superposition
- The probe direction (normal to the decision hyperplane) is a candidate feature direction — but verify causally

---

## Steering Vectors

**What:** Add or subtract vectors from model activations to control behavior.

**How it works:**
1. Compute a "feature direction" (e.g., mean difference between positive/negative sentiment activations)
2. Add multiples of this direction to the residual stream during inference
3. The model's output shifts accordingly

**Methods for finding directions:**
- **Difference-in-means:** Mean activations with feature minus mean without
- **PCA:** First principal component of feature-related activations
- **Linear probe direction:** Normal vector from a trained probe
- **DAS (Distributed Alignment Search):** Gradient descent to find causal directions

**Notable applications:**
- **Refusal direction:** A single direction mediates whether chat models refuse harmful requests. Ablating it jailbreaks the model with minimal performance damage.
- **Sentiment steering:** Adding/subtracting sentiment directions controls output tone
- **Truthfulness:** ITI (Inference-Time Interventions) found a direction that improves TruthfulQA scores

**Caution:** Steering vectors are powerful but blunt. They don't require SAEs — simple difference-in-means often works. Don't overcomplicate this.

---

## Automated Circuit Discovery

**What:** Automatically find the sparse subgraph of model components responsible for a specific behavior.

**How it works:** Recursively apply patching to identify important nodes and edges. Start from the output and work backward, keeping components that significantly affect the output.

**Key methods:**
- **ACDC (Automated Circuit Discovery):** Edge-level patching, systematic but slow
- **ACDC + attribution patching:** Much faster, comparable quality
- **Sparse Feature Circuits:** Circuit analysis using SAE features as nodes, not raw components

**When to use:**
- Scaling circuit analysis beyond manual feasibility
- Finding circuits in larger models
- Getting a first-pass map of the computation

**Limitations:**
- Results need human validation — automated discovery can miss important edges
- Works best for narrow tasks with clear behavioral metrics
- May not capture the full story if the circuit is distributed

---

## Distributed Alignment Search (DAS)

**What:** Use gradient descent to find a subspace of activations that causally mediates a specific behavior.

**How it works:** Optimize a rotation matrix such that patching along the learned subspace has maximal effect on the output. Unlike probing, this is causal — it finds directions that *matter*, not just directions that *correlate*.

**When to use:**
- When features aren't axis-aligned (most of the time, due to superposition)
- When you need a causal (not just correlational) feature direction
- Best used on the residual stream, not individual layer outputs

**Critical caveat — interpretability illusions:** A direction found by DAS can be the *sum* of a dormant direction (causal but never varies) and a disconnected direction (varies but isn't causal). This sum appears both causal and correlated, passing DAS validation, but doesn't represent a real feature. This illusion is especially common when applying DAS to MLP layer outputs.

---

## Choosing the Right Technique

### First Pass: What Matters?
1. **Direct logit attribution** — fast, identifies candidate components
2. **Activation patching** (or attribution patching for speed) — confirms causal importance

### Deep Dive: How Does It Work?
3. **Attention pattern analysis** — understand QK circuits (what attends to what)
4. **Composition analysis** — identify multi-head circuits
5. **SAE feature analysis** — decompose superposed activations

### Validation: Is This Real?
6. **Contrast pairs** — test on inputs that isolate the behavior of interest
7. **Multiple lines of evidence** — attribution + patching + probing should converge
8. **Check for self-repair** — ablate and verify later components don't compensate

---

*Adapted from Neel Nanda's mechanistic interpretability reading list, "Attribution Patching" (Nanda), and "Open Problems in Mechanistic Interpretability" (Sharkey et al., 2025).*
