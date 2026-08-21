## Motivation
Evan et al. [1] showed that sparse autoencoders(SAEs) were unable to recover true underlying features from toy model of composed features. Even despite the fact that the SAEs were exposed to 75% of the true features, they still were unable to learn the true features of data. There were no explanations provided on why SAEs were unable to recover the true features. This therefore begs the following questions: Can the results be reproduced? Under what conditions will this failure mode happen? and how can the issue be resolved?

## Datasets

![Datasets and composed-feature results from Evans et al.](evans/datasets-pics.png)

*Figure 1. Composed-feature datasets used by Evan et al. [1] and the resulting toy-model and SAE feature geometry.*

Evan et al. [1] use four true features split into two sets, \(\{x_1,x_2\}\) and\(\{y_1,y_2\}\). Each training sample contains one feature from each set. In their first dataset, the two active features share the same random amplitude; in the second,
their amplitudes are sampled independently. A trained two-dimensional ReLU toy model maps these inputs to hidden activations on which the SAE is trained.

Our datasets generalizes this construction to \(n_x\) and
\(n_y\) features and a configurable activation dimension. It samples one
uniformly chosen \(x_i\) and \(y_j\), optionally correlates their amplitudes,
allows singleton examples and noise, and returns both the activation
\(h=zD+\epsilon\) and its known sparse coefficients \(z\). Unlike Evan et al. [1],
we can generate activations directly from a controlled dictionary and scale
\(N\) beyond 2. Thus there are only \(2N\) primitive features but \(N^2\)
possible compositions, allowing us to test whether SAEs recover underlying features as
individual compositions become sparser.

## Research question 1: Can result of Evan et al. be reproduced?
Here, we used the same setup like Evan et al. and then try different four different SAE architectures like TopK, Mytryoshkha BatchTopK, BatchTopK and Matching Pursuit SAEs. We observed that using $N=2$, and performing hyperparameter tuning, all the SAEs were unable to learn the true underlying features.

![Best SAE runs for four architectures at N=2](evans/best-saes.png)

*Figure 2. The five best 128-million-sample runs for each SAE architecture, ranked by primitive margin. Colored arrows show the four true features, while black arrows show learned SAE decoder directions.*
> NOTE: primitive margin is the difference between primitive/true feature metric and composed-feature metric. If primitive margin > 0, then the SAE is more aligned to recovering true features and if negative, it is more aligned to recovering composed features.

## Research question 2: Can increasing the sparsity of composed features help recover the true features?
This question is motivated by classical sparse-coding theory. It has been proven that sparsity is a sufficient condition for guaranteed recovery of true features[2, 3, 4]. Lee Sharkey et al. [5] demonstrated recovery of true sparse features of toy model of superposition using SAE. We therefore asked whether making composed features sparser improves
true feature recovery in toy model of composed features.

We used our modified dataset generator that permits varying the number of true features per feature set, N, in the training samples. 
As \(N\) increases from 2 to 256, each composed feature appears in approximately \(1/N^2\) of samples, decreasing from 25% to about 0.0015%. Each sample gets two correlated features and the sparsity of these composed features exposed to SAE increase as we increase N from 2 to 256. We used standard ReLU and BatchTopK SAE from SAELens library.

We used both Hungarian matching and mean maximum cosine similarity as metrics for determining how well the SAE was able to recover the true features. Let \(D_i\) be a normalized true primitive direction, \(f_k\) a normalized learned SAE decoder direction, and \(S_{ik}=\cos(D_i,f_k)\). The one-to-one Hungarian metric finds the globally optimal bijection \(\pi^*\), then reports the fraction of matched pairs above a cosine threshold \(\tau\):

\[
\pi^*=\underset{\pi\in\mathfrak{S}_{2N}}{\arg\max}
\sum_{i=1}^{2N}S_{i,\pi(i)},
\qquad
H_{\tau}=\frac{1}{2N}\sum_{i=1}^{2N}
\mathbf{1}\!\left[S_{i,\pi^*(i)}\geq\tau\right].
\]

Mean Maximum Cosine Similarity (MMCS) assigns each true primitive its nearest learned decoder without enforcing a one-to-one match:

\[
\operatorname{MMCS}=\frac{1}{2N}\sum_{i=1}^{2N}\max_k S_{ik}.
\]

![Standard ReLU and BatchTopK true-feature recovery across composed-feature sparsity levels](evans/sparsity-hungarian-mmcs.png)

*Figure 3. Mean true-feature recovery for Standard ReLU and BatchTopK SAEs as the number of features per set \(N\) increases. The left panel shows the fraction recovered by one-to-one Hungarian matching at a 0.90 cosine cutoff, and the right panel shows mean maximum cosine similarity (MMCS). Each point is the mean across five SAE seeds trained on 125 million samples.*

Both architectures have zero mean Hungarian recovery at \(N=2\) and only 0.15 at \(N=4\). Recovery then rises sharply as the compositions become sparser: Standard ReLU improves earlier and reaches 0.963 at \(N=16\), while BatchTopK reaches 0.906 at \(N=16\) and slightly exceeds Standard ReLU on Hungarian recovery from \(N=64\) onward. MMCS follows the same overall rise, with Standard ReLU generally remaining higher. These results thereby support the hypothesis that composed-feature sparsity promotes recovery of reusable ground-truth features.

## Conclusion
Without any meaningful change in SAE architectures, it can therefore be asserted that `sparsity of composed features` in toy model of composed features is therefore a necessary and sufficient condition for underlying true feature recovery using SAEs. This simple approach hereby resolves the failure mode reported in Evan et al.


## References

[1] Anders, E., Neo, C., Hoelscher-Obermaier, J., & Howard, J. N. (2024, March 14). *Sparse autoencoders find composed features in small toy models*. LessWrong. https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy

[2] Donoho, D. L., & Elad, M. (2003). Optimally sparse representation in general (nonorthogonal) dictionaries via ℓ1 minimization. *Proceedings of the National Academy of Sciences, 100*(5), 2197–2202. https://doi.org/10.1073/pnas.0437847100

[3] Spielman, D. A., Wang, H., & Wright, J. (2012). Exact recovery of sparsely-used dictionaries. *Proceedings of Machine Learning Research, 23*, 37.1–37.18. https://proceedings.mlr.press/v23/spielman12.html

[4] Agarwal, A., Anandkumar, A., Jain, P., Netrapalli, P., & Tandon, R. (2014). Learning sparsely used overcomplete dictionaries. *Proceedings of Machine Learning Research, 35*, 123–137. https://proceedings.mlr.press/v35/agarwal14a.html

[5] Sharkey, L., Braun, D., & Beren. (2022, December 13). *[Interim research report] Taking features out of superposition with sparse autoencoders*. LessWrong. https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition
