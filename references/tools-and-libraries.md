# Interpretability Tools and Libraries

## Table of Contents
1. [TransformerLens](#transformerlens)
2. [NNsight](#nnsight)
3. [SAELens](#saelens)
4. [Pyvene](#pyvene)
5. [Visualization Tools](#visualization-tools)
6. [Choosing the Right Tool](#choosing-the-right-tool)

---

## TransformerLens

**Purpose:** A library for mechanistic interpretability of GPT-style language models. Provides clean, hook-based access to all internal activations.

**Best for:** Direct circuit analysis on small to medium models (GPT-2, GPT-Neo, Pythia, etc.)

**Key features:**
- Load models with `HookedTransformer.from_pretrained("gpt2-small")`
- Access any activation via hooks: residual stream, attention patterns, MLP outputs, individual head outputs
- Built-in cache: `model.run_with_cache(tokens)` returns all activations at once
- Clean API for logit attribution, attention pattern visualization, and ablation studies
- Supports activation patching, mean ablation, and zero ablation out of the box

**Limitations:**
- Only supports models re-implemented in its framework (not arbitrary HuggingFace models)
- Limited to models that fit in memory
- Model list is fixed — new architectures require implementation effort

**Install:** `pip install transformer-lens`

---

## NNsight

**Purpose:** A library for interpreting and intervening on any PyTorch model, especially HuggingFace transformers.

**Best for:** Working with larger models or models not supported by TransformerLens.

**Key features:**
- Works with any HuggingFace model via a tracing context
- **Tracing context:** Define interventions declaratively, then execute them all at once
- Supports activation patching, ablation, and custom interventions
- Can handle models across multiple GPUs
- Compatible with the NNsight remote API for running on models too large for local hardware

**Key patterns:**
```python
with model.trace(input):
    # Read activations
    hidden = model.transformer.h[5].output[0].save()
    # Intervene
    model.transformer.h[5].output[0][:] = patched_value
```

**Limitations:**
- Less ergonomic than TransformerLens for common interpretability workflows
- Requires understanding PyTorch module structure of the target model

**Install:** `pip install nnsight`

---

## SAELens

**Purpose:** A library for training, loading, and analyzing sparse autoencoders on language model activations.

**Best for:** Training SAEs and analyzing their features.

**Key features:**
- Integrates with TransformerLens models
- Provides pretrained SAEs for common models (GPT-2 Small, etc.)
- Supports multiple SAE architectures (standard, gated, topK)
- Feature visualization and analysis utilities
- Integration with Neuronpedia for browsing features

**Typical workflow:**
1. Load a TransformerLens model
2. Train an SAE on activations from a specific layer
3. Analyze features: max-activating examples, logit attribution, density
4. Use features for circuit analysis or steering

**Install:** `pip install sae-lens`

---

## Pyvene

**Purpose:** A systematic framework for causal interventions on neural networks.

**Best for:** Running structured intervention experiments, especially DAS.

**Key features:**
- Supports interchange interventions, activation patching, and DAS
- Works with HuggingFace models
- Provides a clean abstraction for defining and composing interventions
- Built-in support for Distributed Alignment Search (learning causal subspaces)

**When to use:** When you need systematic, reproducible intervention experiments across many components, or when using DAS to find causal feature directions.

**Install:** `pip install pyvene`

---

## Visualization Tools

### CircuitsVis
Interactive HTML visualizations for attention patterns and other interpretability outputs. Generates embeddable visualizations for notebooks and papers.

**Install:** `pip install circuitsvis`

### Neuronpedia
A web-based viewer for SAE features. Browse max-activating examples, logit effects, and feature descriptions for pretrained SAEs. Available at neuronpedia.org.

### BERTViz
Attention visualization specifically for BERT-family models. Shows attention patterns across heads and layers.

**Install:** `pip install bertviz`

### Built-in TransformerLens Visualization
TransformerLens includes plotting utilities for attention patterns, logit attribution heatmaps, and activation distributions. These are often sufficient for quick exploration.

---

## Choosing the Right Tool

| Scenario | Recommended Tool |
|----------|-----------------|
| Circuit analysis on GPT-2 or similar small models | **TransformerLens** |
| Working with Llama, Mistral, or other large HuggingFace models | **NNsight** |
| Training sparse autoencoders | **SAELens** (+ TransformerLens) |
| Browsing existing SAE features | **Neuronpedia** |
| Systematic causal intervention experiments | **Pyvene** |
| Quick attention pattern visualization | **CircuitsVis** or TransformerLens built-in |
| DAS (finding causal subspaces) | **Pyvene** |

### Common Workflow

1. **Start with TransformerLens** for initial exploration on a small model
2. Use **SAELens** to train or load SAEs for feature analysis
3. Browse features on **Neuronpedia** to build intuition
4. Switch to **NNsight** when you need to scale to larger models
5. Use **Pyvene** for systematic DAS experiments

---

*Compiled from the ARENA curriculum, TransformerLens documentation, and Neel Nanda's tooling recommendations (2025).*
