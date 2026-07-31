---
name: mech-interp
description: >
  Mechanistic interpretability reference for understanding transformer internals,
  circuits, and features. Use this skill when the user mentions "mechanistic
  interpretability", "mech interp", "transformer circuits", "attention heads",
  "residual stream", "SAE", "sparse autoencoder", "logit attribution", "activation
  patching", "induction heads", "superposition", "polysemanticity", "TransformerLens",
  "NNsight", "SAELens", "circuit analysis", "feature extraction", "probing",
  "steering vectors", or is working on interpretability research, reverse-engineering
  neural networks, or analyzing model internals. Also trigger when editing Python
  code that imports transformerlens, nnsight, sae_lens, circuitsvis, or pyvene,
  or when discussing how language models represent information internally.
---

# Mechanistic Interpretability

A reference for understanding and reverse-engineering the internal mechanisms of transformer language models. Covers core concepts, analysis techniques, tools, and common pitfalls.

**When this skill applies:**
- Analyzing how transformers process information internally
- Finding and interpreting circuits, features, or representations
- Training or evaluating sparse autoencoders
- Using interpretability libraries (TransformerLens, NNsight, SAELens)
- Understanding attention patterns, composition, or superposition
- Doing activation patching, logit attribution, or causal interventions

---

## Core Concepts Quick Reference

### The Residual Stream

The residual stream is the central communication channel in a transformer. Each layer reads from it and writes back to it additively. Think of it as a shared workspace — attention heads and MLPs each contribute terms that accumulate.

- **All components interact through the residual stream** — attention heads and MLPs don't talk to each other directly; they read from and write to this shared vector
- The residual stream at any point is the sum of the token embedding, positional embedding, and every attention/MLP output up to that layer
- Apply the unembedding matrix to the residual stream at any layer to see what the model "thinks" at that point (logit lens)

### Attention Heads: QK and OV Circuits

Each attention head has two independent functions:

| Circuit | Function | Determines |
|---------|----------|-----------|
| **QK circuit** (W_Q, W_K) | What to attend to | Which source positions get high attention weights |
| **OV circuit** (W_V, W_O) | What to output | What information gets moved to the destination |

These are independent — a head can attend to one thing (QK) and output something entirely different (OV). This separation is key to understanding circuits.

### Composition Types

Heads in later layers can read the outputs of earlier heads via the residual stream. Three types:

| Type | What composes | Example |
|------|--------------|---------|
| **Q-composition** | Earlier head output -> later head's queries | "Attend to positions that head 3 marked as important" |
| **K-composition** | Earlier head output -> later head's keys | "Be attended to if head 3 wrote something here" |
| **V-composition** | Earlier head output -> later head's values | "Pass along what head 3 computed" |

K-composition is how **induction heads** work: a previous-token head writes positional info, and an induction head uses this as keys to find where the pattern continues.

### MLPs as Key-Value Memories

MLP layers can be understood as collections of key-value pairs:
- Each neuron has an "input direction" (key) and an "output direction" (value)
- When the residual stream aligns with a neuron's key, it fires and adds its value to the stream
- MLPs often implement factual knowledge retrieval and non-linear feature processing

### Superposition

Models encode more features than they have dimensions. Features are represented as directions in activation space, not aligned with individual neurons. This means:

- **Polysemanticity:** Individual neurons respond to multiple unrelated concepts
- **Distributed representations:** Single features are spread across many neurons
- **Interference:** Features interfere with each other, causing cascading errors
- Superposition increases with feature sparsity — rare features are more compressed

See `references/transformer-circuits.md` for deeper coverage of these concepts.

---

## Technique Selection Guide

| Need | Technique | Strengths | Limitations |
|------|-----------|-----------|-------------|
| What matters for the output? | **Direct logit attribution** | Fast, exact for direct effects | Misses indirect effects through later layers |
| Where is information processed? | **Activation patching** | Causal, gold standard | One forward pass per component (slow) |
| Fast approximate patching | **Attribution patching** | Gradient-based, all components at once | Approximation; less reliable at early layers |
| What features are encoded? | **Linear probing** | Simple, well-understood | Correlational, not causal; may find info model doesn't use |
| Decompose superposition | **Sparse autoencoders** | Scalable, finds interpretable features | Quality hard to evaluate; may miss features |
| Steer model behavior | **Steering vectors** | Cheap to compute, effective | Blunt instrument; hard to control precisely |
| Automated circuit finding | **Automated circuit discovery** | Scales to larger models | May miss important edges; results need validation |

See `references/interpretability-techniques.md` for detailed guidance on each technique.

---

## Sparse Autoencoders Quick Reference

SAEs decompose model activations into sparse, interpretable features. They're a one-hidden-layer autoencoder trained to reconstruct activations with an L1 sparsity penalty.

**Architecture variants:**

| Variant | Key Idea | When to Use |
|---------|----------|-------------|
| **Standard (ReLU)** | L1 penalty on hidden activations | Baseline, well-understood |
| **Gated SAE** | Separate gate and magnitude estimation | Better sparsity-reconstruction tradeoff |
| **TopK SAE** | Keep only k largest features per input | Avoids L1 shrinkage, scales well |
| **JumpReLU SAE** | Learnable threshold per feature | Good sparsity control |
| **Transcoders** | Map MLP input to output (not autoencoder) | Circuit analysis through MLP layers |

**Key evaluation metrics:**
- **L0 norm:** Average number of active features per input (lower = sparser)
- **Loss recovered:** Cross-entropy loss with SAE reconstructions vs. original (higher = better)
- **Feature density histogram:** Distribution of how often each feature fires
- **Dead features:** Features that never activate — indicates training issues

See `references/sparse-autoencoders.md` for training guidance and architectural details.

---

## Tools Quick Reference

| Tool | Best For | Key Feature |
|------|----------|-------------|
| **TransformerLens** | Direct circuit analysis on small models | Hook-based access to all activations; clean API for GPT-2, etc. |
| **NNsight** | Intervention/patching on any HuggingFace model | Works with models too large for TransformerLens; tracing context |
| **SAELens** | Training and analyzing SAEs | Integrates with TransformerLens; provides pretrained SAEs |
| **Pyvene** | Causal interventions and DAS | Systematic intervention framework |
| **CircuitsVis** | Attention pattern visualization | Interactive HTML visualizations |
| **Neuronpedia** | Browse SAE features online | Web viewer for feature dashboards |

**Decision guide:**
- Start with **TransformerLens** for GPT-2 or small model analysis
- Use **NNsight** when you need larger models or HuggingFace compatibility
- Use **SAELens** when training or analyzing sparse autoencoders
- Use **Pyvene** for systematic causal intervention experiments

See `references/tools-and-libraries.md` for setup and usage patterns.

---

## Common Pitfalls

1. **Confirmation bias in circuit discovery** — It's easy to find evidence for a beautiful hypothesis while missing simpler explanations. Actively seek alternative explanations and test them.

2. **Correlation vs. causation in attribution** — Probing shows that information is *encoded*, not that the model *uses* it. Always follow up with causal interventions (patching) to verify.

3. **Ignoring superposition** — Neurons are polysemantic. Don't interpret a neuron as "the X neuron" based on max-activating examples from one dataset — different datasets may show completely different patterns.

4. **Interpretability illusions** — A direction can be both causal and correlated with a feature without actually representing it (the dormant + disconnected direction problem). Be especially cautious with DAS on model layers rather than the residual stream.

5. **Overinterpreting narrow circuits** — A circuit found for one narrow task (e.g., IOI) tells you what components are involved *for that task*. Those same components likely do many other things in other contexts.

6. **Negative results without insight** — "I tried X and it didn't work" is rarely informative. "X didn't work because of Y, which rules out the entire class of Z approaches" is valuable.

7. **Not checking for self-repair** — When you ablate a component, later components may shift to compensate (the hydra effect). This can make components look less important than they are.

---

## Key Concepts Reference

For definitions of specific terms (induction heads, skip trigrams, logit lens, attention patterns, etc.), see `references/glossary.md`.

---

## Additional Resources

| File | Contents |
|------|----------|
| `references/transformer-circuits.md` | Residual stream, attention mechanics, composition, MLPs, superposition |
| `references/interpretability-techniques.md` | Logit attribution, activation patching, probing, steering, circuit discovery |
| `references/sparse-autoencoders.md` | SAE architecture, training, variants, evaluation, and common issues |
| `references/tools-and-libraries.md` | TransformerLens, NNsight, SAELens, Pyvene, visualization tools |
| `references/glossary.md` | Key terms and definitions for mechanistic interpretability |

---

*Adapted from Neel Nanda's mechanistic interpretability reading list, the ARENA curriculum, and "Open Problems in Mechanistic Interpretability" (Sharkey et al., 2025).*