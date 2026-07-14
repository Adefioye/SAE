### Replicating Seed Stability Results in Sparse Autoencoders
This project replicates results from the SAE multiple-seeds paper and investigates why they differ from later results in another study. Participants will compare features across SAE training seeds, starting with TopK SAEs trained on small models, and extend the comparison methodology to activation overlap, CKA, or SVCCA. If time permits, the project may move on to larger models or more recent SAE architectures.
##### Prep work
- Read the SAE multiple-seeds paper: [https://arxiv.org/abs/2501.16615](https://arxiv.org/abs/2501.16615).
- Read the related comparison paper: [https://arxiv.org/abs/2505.20254](https://arxiv.org/abs/2505.20254).
- Familiarize yourself with [https://github.com/EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) or another SAE training library.
##### Example project tasks
- Train multiple SAE seeds on Pythia-160M.
- Measure overlap between features from different seeds using the methodology from the SAE multiple-seeds paper.
- Compare alternative overlap techniques such as activation overlap, CKA, and SVCCA.
- Extend the experiments to additional models.
- Extend the experiments to different SAE architectures if time permits.
##### Expected deliverables
- Blog post describing the replication, comparison methods, and findings.