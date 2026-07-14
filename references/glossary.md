# Mechanistic Interpretability Glossary

Key terms and definitions, organized by topic.

---

## Transformer Architecture

**Residual stream** — The central vector that accumulates outputs from all layers. Each attention head and MLP reads from and writes to it additively.

**Attention head** — A component that moves information between token positions. Consists of independent QK (where to attend) and OV (what to output) circuits.

**QK circuit** — The query-key interaction that determines attention patterns. Computes which source positions each destination position attends to.

**OV circuit** — The value-output interaction that determines what information is moved. Independent from QK.

**MLP layer** — A feed-forward network applied after attention. Processes information non-linearly. Can be understood as key-value memories.

**Layer normalization (LayerNorm)** — Normalizes activations before each sublayer. Affects interpretation because it changes the effective scale of residual stream components.

**Embedding / Unembedding** — The embedding matrix maps tokens to vectors; the unembedding maps residual stream vectors to logit scores over the vocabulary.

---

## Circuits and Composition

**Circuit** — A sparse subgraph of model components (heads, MLPs) that implements a specific behavior. The interpretability goal is to find and understand these circuits.

**Composition (Q/K/V)** — When a later head reads the output of an earlier head. Q-composition affects queries, K-composition affects keys, V-composition affects values.

**Induction head** — A head that completes repeated patterns ("A B ... A" -> "B"). Works via K-composition with a previous-token head. Foundational example of a multi-head circuit.

**Skip-trigram** — A pattern captured by a single attention head: "if token A appeared earlier, predict token C." Limited to one-layer computations.

**Previous-token head** — A head that attends uniformly to the previous position. Often the first component in induction circuits.

**Copy suppression head** — A head that suppresses tokens the model is about to predict if they appeared earlier. A calibration mechanism.

**Negative head** — A head whose OV circuit outputs the negative of what it attends to, effectively suppressing or anti-copying.

**Backup heads / self-repair / hydra effect** — When ablating a component, later components partially compensate. Makes ablation studies underestimate component importance.

**Narrow circuit** — A circuit identified for a specific narrow task (e.g., indirect object identification). The same components may do different things in other contexts.

---

## Features and Representations

**Feature** — A human-interpretable property of the input that the model represents as a direction in activation space.

**Superposition** — The phenomenon where models encode more features than they have dimensions, compressing them into overlapping directions.

**Polysemanticity** — When a single neuron responds to multiple unrelated concepts. A consequence of superposition.

**Monosemanticity** — When a neuron or feature corresponds to a single interpretable concept. The ideal state that SAEs try to recover.

**Linear representation hypothesis** — Features are encoded as linear directions in activation space. Supported by extensive evidence (probing, steering, SAEs).

**Privileged basis** — A basis in which individual coordinates are meaningful (e.g., neuron activations after ReLU). The residual stream generally does NOT have a privileged basis; MLP hidden layers do.

**Feature direction** — The vector in activation space corresponding to a feature. Can be found via probes, SAE decoder vectors, or difference-in-means.

---

## Techniques

**Logit lens** — Apply the unembedding matrix to the residual stream at intermediate layers to see what the model "predicts" at that point.

**Direct logit attribution** — Decompose the final logits into per-component contributions using the linearity of the residual stream.

**Activation patching** — Replace a component's activation from one input with its value from another. The gold standard causal technique.

**Attribution patching** — Gradient-based approximation to activation patching. Fast (two forward + one backward pass for all components).

**Zero ablation** — Set a component's output to zero. Harsh; can produce out-of-distribution activations.

**Mean ablation** — Replace a component's output with its mean over a dataset. Less harsh than zero ablation.

**Resample ablation** — Replace with the value from a random different input. Preserves activation statistics.

**Linear probing** — Train a linear classifier on activations to test whether a feature is encoded. Correlational, not causal.

**Steering vector** — A direction added to activations during inference to control model behavior. Found via difference-in-means, PCA, or DAS.

**Distributed Alignment Search (DAS)** — Use gradient descent to find a subspace that causally mediates a behavior. More principled than probing but susceptible to interpretability illusions.

**Causal scrubbing** — A rigorous method for measuring how good a circuit explanation is: do all patches allowed by the explanation and measure damage.

---

## SAE-Specific Terms

**Sparse autoencoder (SAE)** — A one-hidden-layer autoencoder trained to reconstruct model activations with a sparsity penalty. Decomposes superposition into interpretable features.

**SAE feature** — A single unit in the SAE's hidden layer. Ideally corresponds to one interpretable concept.

**Decoder dictionary** — The SAE's decoder weight matrix. Each row is a feature vector in activation space.

**L0 norm** — The average number of active (non-zero) SAE features per input. Measures sparsity.

**Loss recovered** — The percentage of the model's original cross-entropy loss preserved when using SAE reconstructions instead of original activations.

**Dead features** — SAE features that never activate. Waste capacity and indicate training issues.

**Feature density** — The proportion of inputs where a feature activates. Very dense or very sparse features tend to be less interpretable.

**Gated SAE** — Variant that decouples feature detection from magnitude estimation. Reduces shrinkage.

**TopK SAE** — Variant that keeps exactly the k largest features per input. No L1 penalty needed.

**JumpReLU SAE** — Variant with a learnable per-feature activation threshold.

**Transcoder** — Maps MLP input to output through a sparse bottleneck. Enables circuit analysis through MLP layers.

**Feature splitting** — When one concept is represented across multiple SAE features at different granularities.

**Shrinkage** — The L1 penalty biasing feature magnitudes toward zero, reducing reconstruction quality. Addressed by Gated and TopK variants.

---

## Research Concepts

**Interpretability illusion** — When a technique produces results that appear interpretable but are misleading. Examples: max-activating examples varying by dataset, DAS finding dormant+disconnected directions.

**ROME (Rank-One Model Editing)** — A fact insertion technique that revealed an important interpretability illusion: it inserted new facts without editing old ones, creating pathological behavior.

**Indirect Object Identification (IOI)** — The canonical circuit analysis task: "When John and Mary went to the store, John gave the bag to" -> " Mary." Well-studied benchmark for circuit discovery methods.

**Function vectors** — Directions in activation space that encode a task or function. Can be extracted from attention head outputs and used to steer behavior.

**Research debt** — The accumulated burden of unclear explanations and implicit knowledge in a field. Interpretability has significant research debt in its techniques and terminology.

---

*Compiled from Neel Nanda's reading list, "A Mathematical Framework for Transformer Circuits," "Toy Models of Superposition," and "Open Problems in Mechanistic Interpretability" (2025).*
