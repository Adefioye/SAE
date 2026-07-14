# Transformer Circuits

## Table of Contents
1. [The Residual Stream](#the-residual-stream)
2. [Attention Heads in Detail](#attention-heads-in-detail)
3. [Composition](#composition)
4. [Key Circuits](#key-circuits)
5. [MLP Layers](#mlp-layers)
6. [Superposition and Polysemanticity](#superposition-and-polysemanticity)

---

## The Residual Stream

The residual stream is the backbone of transformer computation. Every component reads from it and writes back additively.

At position `i` after layer `l`, the residual stream is:

```
x_i^l = embedding(token_i) + pos_encoding(i) + sum(attn_outputs_0..l) + sum(mlp_outputs_0..l)
```

### Why This Matters

- **Linearity of the residual stream** means we can decompose any activation into contributions from individual components (heads, MLPs, embeddings)
- Components can "talk" to each other only through the residual stream — there is no direct head-to-head communication
- The **logit lens** technique exploits this: apply the unembedding matrix to the residual stream at any layer to see what the model predicts at that point
- **Direct logit attribution** decomposes the final logits into per-component contributions by projecting each component's output through the unembedding

### The Unembedding as a Lens

The model's final operation is: `logits = residual_stream @ W_U + bias`. Since the residual stream is a sum of component outputs, each component's contribution to any specific logit can be computed independently. This is the foundation of logit attribution and many circuit analysis techniques.

---

## Attention Heads in Detail

Each attention head performs two independent computations:

### QK Circuit (Where to Attend)

The query-key interaction determines attention patterns:
- **Queries** (W_Q): "What am I looking for?"
- **Keys** (W_K): "What do I contain?"
- Attention score = `query @ key.T / sqrt(d_head)`
- Softmax produces attention weights

The QK circuit is a bilinear form: `attention_score(source, dest) = x_dest @ W_Q @ W_K.T @ x_source.T`

This is a low-rank factorized matrix — the full QK matrix is `d_model x d_model` but has rank `d_head`.

### OV Circuit (What to Output)

The value-output interaction determines what information moves:
- **Values** (W_V): "What information do I have?"
- **Output** (W_O): "How should this be projected back to the residual stream?"
- Output = `attention_weights @ (x @ W_V @ W_O)`

The OV circuit is also a low-rank matrix: `d_model x d_model` with rank `d_head`.

### Key Insight: QK and OV Are Independent

A head can attend to one thing and output something completely different. For example, an induction head attends to the token *after* the previous occurrence of the current token (QK), but outputs the *content* of that next token (OV). Understanding this separation is essential for circuit analysis.

---

## Composition

Later heads can read the outputs of earlier heads through the residual stream. This creates multi-step circuits.

### Q-Composition
An earlier head's output influences a later head's *queries*. The later head attends to positions based on what the earlier head computed.

### K-Composition
An earlier head's output influences a later head's *keys*. Positions become "more attendable" based on what earlier heads wrote there. This is how **induction circuits** work.

### V-Composition
An earlier head's output influences a later head's *values*. The later head passes along information computed by the earlier head.

### Measuring Composition

The composition score between two heads can be computed using the **virtual weights** framework: multiply the OV matrix of the earlier head with the QK or OV matrix of the later head. High-norm virtual weights indicate strong composition.

### Practical Implications

- One-layer models can only implement patterns that depend on the current token and the token being attended to (skip-trigrams)
- Two or more layers are needed for pattern-copying (induction) because it requires K-composition between heads in different layers
- The depth of a circuit limits the complexity of the patterns it can detect

---

## Key Circuits

### Induction Heads

The most well-understood multi-head circuit. Induction heads complete patterns: if "A B ... A" appears, predict "B".

**Mechanism (2-layer version):**
1. **Previous-token head** (layer 0): Attends to the previous position. Writes positional info ("I'm the token after X") to the residual stream.
2. **Induction head** (layer 1): Uses K-composition to attend to tokens where the *previous token matches the current token*, then copies the token that followed.

**Why it matters:**
- Foundational example of a multi-head circuit with clear K-composition
- Important for in-context learning — enables few-shot pattern matching
- Forms at a consistent point in training across model sizes (a phase transition)

### Skip-Trigram Heads

Single-layer attention heads that learn trigram-like statistics: "if token A appears at position i, predict token C is likely at position i+k." These capture simple co-occurrence patterns but cannot do pattern completion (that requires induction).

**Skip-trigram bugs** arise from the QK/OV independence: a head attends to token A (QK) but outputs something appropriate for a *different* context where it would attend to the same token. These are inherent to the attention mechanism.

### Copy Suppression Heads

Heads that detect when earlier layers have decided to predict a token, check if that token appeared earlier in context, and suppress it. This is a calibration mechanism:
- Reduces overconfidence in repeated tokens
- Explains "anti-induction" and "negative name mover" behaviors observed in various circuits
- Part of overall model calibration — loss gets worse without it

### Backup / Self-Repair (The Hydra Effect)

When a component is ablated, later components often shift to partially compensate. This means:
- Ablation studies may underestimate component importance
- The effect is not learned through dropout — it happens in models trained without it
- Partially explained by LayerNorm scaling: removing a component reduces the residual stream norm, so remaining components get relatively amplified

---

## MLP Layers

MLP layers process information non-linearly after each attention layer.

### Conceptual Model: Key-Value Memories

Each neuron in an MLP can be thought of as a key-value pair:
- **Key (input weights):** A direction in the residual stream. When the residual stream aligns with this direction, the neuron activates.
- **Value (output weights):** A direction written to the residual stream when the neuron fires.
- The activation function (ReLU/GELU) acts as a gate.

### What MLPs Do

- **Factual knowledge storage:** MLPs in middle layers often store factual associations
- **Feature processing:** Computing non-linear functions of input features
- **Superposition:** MLP neurons are heavily polysemantic — each participates in encoding many features

### Analyzing MLPs

MLPs are harder to interpret than attention heads because:
- Neurons are polysemantic (superposition)
- Non-linearities make linear decomposition inexact
- Many more parameters per layer than attention

**Transcoders** address this: they're SAE-like models that learn a sparse, interpretable mapping from MLP input to output, enabling circuit analysis through MLP layers.

---

## Superposition and Polysemanticity

### The Core Problem

Models need to represent many more features than they have dimensions. They solve this by encoding features as directions in activation space that aren't aligned with individual neurons.

### Key Findings from Toy Models

- **Sparsity enables superposition:** The sparser a feature (less frequently active), the more the model can compress it with others
- **Importance matters:** More important features get more "space" and less interference
- **Correlated features resist superposition:** Features that co-occur are harder to compress together
- **Phase transitions:** Models undergo sharp transitions between no-superposition and full-superposition regimes

### Implications for Interpretability

- **Max-activating examples are unreliable:** The same neuron shows different patterns on different datasets because different features are most aligned with it in different data distributions
- **Linear probes work** for finding features because features are (approximately) linear directions
- **SAEs are the primary tool** for decomposing superposition into interpretable features
- **Perfect decomposition may be impossible** — superposition causes inherent interference that limits how cleanly features can be separated

---

*Adapted from "A Mathematical Framework for Transformer Circuits" (Elhage et al., Anthropic), "Toy Models of Superposition" (Elhage et al., Anthropic), and Neel Nanda's reading list (2025).*
