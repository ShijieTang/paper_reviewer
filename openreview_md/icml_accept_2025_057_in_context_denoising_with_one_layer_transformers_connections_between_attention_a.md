# In-Context Denoising With One-Layer Transformers: Connections Between Attention And Associative Memory Retrieval

Matthew Smart 1 Alberto Bietti 2 **Anirvan M. Sengupta** 2 3 4

## Abstract

We introduce in-context denoising, a task that refines the connection between attention-based architectures and dense associative memory (DAM) networks, also known as modern Hopfield networks. Using a Bayesian framework, we show theoretically and empirically that certain restricted denoising problems can be solved optimally even by a single-layer transformer. We demonstrate that a trained attention layer processes each denoising prompt by performing a single gradient descent update on a context-aware DAM energy landscape, where context tokens serve as associative memories and the query token acts as an initial state. This one-step update yields better solutions than exact retrieval of either a context token or a spurious local minimum, providing a concrete example of DAM networks extending beyond the standard retrieval paradigm. Overall, this work solidifies the link between associative memory and attention mechanisms first identified by Ramsauer et al., and demonstrates the relevance of associative memory models in the study of in-context learning.

## 1. Introduction

1 derstanding the mechanisms behind transformer-based networks remains an open challenge. This challenge is exacerbated by the growing scale and complexity of modern large networks. Toward addressing this, researchers studying simplified architectures have identified connections between the attention operation that is central to transformers and associative memory models (Ramsauer et al., 2021), providing not only an avenue for understanding how such architectures encode and retrieve information but also potentially ways to improve them further. The most celebrated model for associative memories in systems neuroscience is the so-called Hopfield model (Amari, 1972; Nakano, 1972; Little, 1974; Hopfield, 1982). This model has a capacity to store "memories" (stable fixed points of a recurrent update rule) proportional to the number of nodes (Hopfield, 1982; Amit et al., 1985). In the last decade, new energy functions (Krotov & Hopfield, 2016; Demircigil et al., 2017) were proposed for dense associative memories with much higher capacities. These energy functions are often referred to as modern Hopfield models. Ramsauer et al. (2021) pointed out the similarity between the one-step update rule of a certain modern Hopfield network (Demircigil et al., 2017) and the softmax attention layer of transformers, generating interest in the statistical physics and systems neuroscience communities (Krotov & Hopfield, 2021; Krotov, 2023; Lucibello & Mezard ´ , 2024; Millidge et al., 2022). Recent work has extended this concept to improve retrieval by incorporating sparsity (Hu et al., 2023; Wu et al., 2024b; Santos et al., 2024; Wu et al., 2024a), while others have leveraged associative memory principles to design new energy-based transformer architectures (Hoover et al., 2023). However, these extensions and the foundational construction in Ramsauer et al. (2021) primarily focus on the specific task of exact retrieval (converging to a fixed point), while in practice transformers may tackle many other tasks. To explore this connection beyond retrieval, we introduce *in-context denoising*, a task that bridges the behavior of trained transformers and associative memory networks through the lens of in-context learning (ICL). In standard ICL, a sequence model is trained to infer an unknown function g from contextual examples, predicting g(XL+1) given a sequence of input-output pairs E =
((X1, g(X1)), ...,(XL, g(XL)),(XL+1, −)). Crucially, g is implied solely through the context and differs across prompts - performant models are therefore said to "learn g(x) in context". While ICL has been extensively studied in supervised settings (Garg et al., 2022; Zhang et al., 2024; Akyurek et al. ¨ , 2023; Reddy, 2024), recent work suggests that transformers may internally emulate gradient descent over a context-specific loss function during inference (Von Oswald et al., 2023; Dai et al., 2023; Ahn et al., 2023). This general perspective aligns with our findings. In this work, we generalize ICL to an unsupervised setting where the prompt consists of L samples from a random distribution and the query is a noise-corrupted sample from the same distribution. This shift allows us to probe how trained transformers internally approximate Bayes optimal inference, while deepening the connection to associative memory models which are prototypical denoisers. By setting up this problem in this way, we also attempt to answer a few questions. One concerns the memorization-generalization dilemma in denoising: a Hopfield model's success is usually measured by successful memory recovery, while in-context learning may have to solve a completely new problem. Another question has to do with the number of iterations of the corresponding Hopfield model: why does the Ramsauer et al. (2021) correspondence involve only one iteration of Hopfield energy minimization and not many? In summary, our contributions are as follows: In Section 2, we introduce in-context denoising as a framework for understanding how transformers perform implicit inference beyond memory retrieval. In Section 3, we establish that single-layer transformers with one attention head are expressive enough to optimally solve certain denoising problems. We then empirically demonstrate that standard training from random weights can recover the Bayes optimal predictors. The trained attention layers are mapped back to dense associative memory networks in Section 4. Our results refine the general connection pointed out in previous work, offer new mechanistic insights into attention, and provide a concrete example of dense associative memory networks extending beyond the standard memory retrieval paradigm to solve a novel in-context learning task.

## 2. Problem Formulation: In-Context Denoising

In this section, we describe our general setup. Recurring common notation is described in Appendix A.1.

## 2.1. Setup

Each task corresponds to a distribution D over the probability distribution of data: pX ∼ D. Let X1, · · · , XL+1 iid∼
pX, define the sampling of the tokens. Let the noise corruption be defined by X˜ ∼ pnoise(·|XL+1). The random sequence E = (X1, X2*, ..., X*L, X˜) are given as "context"
(input) to a sequence model F(·; θ) which outputs an estimate XˆL+1 of the original (L + 1)-th token . The task is to minimize the expected loss E[l(XˆL+1, XL+1)] for some loss function l(·, ·). Namely, our problem is to find

$$\min_{\theta}\mathbb{E}_{p_{X}\sim D,X_{1:L+1}\sim p_{X}^{L+1},\tilde{X}\sim p_{\rm noise}(\cdot|X_{L+1})}[l(F(E,\theta),X_{L+1})].\tag{1}$$

In practice, we choose X˜ = XL+1 + Z, a pure token corrupted by the addition of isotropic Gaussian noise Z ∼
N (0, σ2ZIn), and our objective function to minimize is the mean squared error (MSE) E[||XˆL+1 − XL+1||2].

In the following subsection, we explain the pure token distributions for three specific tasks. These tasks are of course structured so that a one-layer transformer has the expressivity to capture a solution, which, as L → ∞, provides an optimal solution, in some sense. To that end, we derive Bayes optimal estimators for each of the three tasks, under the assumption that we know the original distribution pX of pure tokens. In Section 3, we use these estimators as baselines to evaluate the performance of the denoiser f(*E, θ*)
based on a one-layer transformer trained on finite datasets.

## 2.2. Task-Specific Token Distributions

We consider three elementary in-context denoising tasks, where the data (vectors in R
n) comes from:
1. Linear manifolds (d-dimensional subspaces) 2. Nonlinear manifolds (d-spheres) 3. Small noise Gaussian mixtures (clusters) where the component means have fixed norm Below we describe the task-specific distributions pX and the process for sampling tokens {xt}. The same corruption process applies to all cases: X˜ = XL+1+Z, Z ∼ N (0, σ2Z
In).

## 2.2.1. Case 1 - Linear Manifolds

A given training prompt consists of pure tokens sampled from a random d-dimensional subspace S of R
n.

- Let P be the orthogonal projection operator to a random d-dim subspace S of R
n, sampled according to the uniform measure, induced by the Haar measure on the coset space O(n)/O(n − d) × O(d), on the Grassmanian G(*d, n*), the manifold of all d-dimensional subspaces of R
n.

- Let Y ∼ N (0, σ2 0 In) and define X = P Y ; we use this procedure to construct the starting sequences
(X1*, ..., X*L+1) of L + 1 independent tokens.

(a) (b)
Problem formulation Prompt: Pure tokens from a data distribution and a single corrupted example Query
(prompts are randomly constructed from a pre-specified task distribution)
sample a task from a task distribution target query Prediction Target sample context tokens corruption of final token Case 1:
Linear manifolds Case 2:
Nonlinear manifolds Case 3:
Gaussian mixtures
We thus have pX = N (0, σ2 0P), with the Haar distribution of P characterizing the task ensemble associated with D.

$\mathbf{v}=\mathbf{v}_{0}\mathbf{v}_{0}$
2.2.2. CASE 2 - NONLINEAR MANIFOLDS
We focus on the case of d-dimensional spheres of fixed radius R centered at the origin in R
n.

- Choose a random d+1-dimensional subspace V of R
n, sampled according to the uniform measure, as before, on the Grassmanian G(d + 1, n). The choice of this random subspace generates the distribution of tasks D.

- Inside V , sample uniformly from the radius R sphere
(once more, a Haar induced measure on a coset space O(d + 1)/O(d)). We use this procedure to construct input sequences X1:L+1 = (x1*, ..., x*L+1) of L + 1 independent tokens.

In practice, we uniformly sample points with fixed norm in R 
dand embed them in R
n by concatenating zeros. We then rotate the points by selecting a random orthogonal matrix Q ∈ R
n×n.

2.2.3. CASE 3 - GAUSSIAN MIXTURES (CLUSTERING)
Pure tokens are sampled from a weighted mixture of isotropic Gaussians in n-dimensions, {wa,(µa, σ2a
)}
K
a=1.

The density is

$$p_{X}(x)=\sum_{a=1}^{K}w_{a}C_{a}e^{-\left\|x-\mu_{a}\right\|^{2}/2\sigma_{a}^{2}},$$

where Ca = (2πσ2a
)
−n/2are normalizing constants. The µa are independently chosen from a uniform distribution on the radius R sphere of dimension n − 1, centered around zero. The distribution of tasks D, is decided by the choice of {µa}
K
a=1.

For our ideal case, we will consider the limit that the variances go to zero. In that case, the density is simply

$$p_{X_{0}}(x)=\sum_{a=1}^{K}w_{a}\delta(x-\mu_{a}).$$

## 2.3. Bayes Optimal Denoising Baselines For Each Case

The first L tokens in E are "pure samples" from p that should provide information about the distribution for our denoising task. Our performance is expected to be no better than that of the best method, in the case that the token distribution and also the corrupting process are exactly known. This is where the Bayesian optimal baseline comes in. As is well-known, the Bayes optimal predictor of a quantity is given by the posterior mean. We use that fact to compute the Bayes optimal loss.

In particular, we seek a function f : R
n → R
n such that EX,X˜
h∥X − f(X˜)∥
2iis minimized. Since the perturbation Z is Gaussian, the posterior distribution of X, given X˜
is

$$p_{X|{\tilde{X}}}(x\mid{\tilde{x}})=C({\tilde{x}})p_{X}(x)e^{-\|x-{\tilde{x}}\|^{2}/2\sigma_{x}^{2}}$$

where C(˜x) is a normalizing factor (see Appendix A.2 for more explanation). The following proposition sets up a baseline to which we expect to compare our results as L → ∞. The proof is in Appendix B.1. Proposition 1. For each task, specified by the input distribution pX*, and the noise model* pX˜|X,

$$\mathbb{E}_{X,\bar{X}}\left[\|X-f(\bar{X})\|^{2}\right]\geq\mathbb{E}_{\bar{X}}\left[\operatorname{Tr}\operatorname{Cov}(X\mid\bar{X})\right].\tag{2}$$

This lower bound is met when f(X˜) = E[X | X˜].

Thus, the Bayes optimal denoiser is the posterior expectation for X given X˜. The expected loss is found by computing the posterior sum of variances.

These optimal denoisers can be computed analytically for both the linear and nonlinear manifold cases (given the variances and dimensionalities). In the Gaussian mixture (clustering) case, it depends on the choice of the centroids which then needs to be averaged over. Linear case. For the linear denoising task, pure samples X are drawn from an isotropic Gaussian in a restricted subspace. The following result provides the Bayes optimal predictor in this case, the proof of which is in Appendix C.1.

Proposition 2. For pX corresponding to Subsection 2.2.1, the Bayes optimal answer is

$$f_{o p t}(\tilde{X})=\mathbb{E}[X|\tilde{X}]=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}P\tilde{X},$$
$$e d\;l o s s\;i s$$
P X, ˜ (3)
and the expected loss is
$$\mathbb{E}\left[\left\|P{\bar{X}}-X_{L+1}\right\|^{2}\right]=d\sigma_{0}^{2}\sigma_{Z}^{2}/(\sigma_{0}^{2}+\sigma_{Z}^{2}).$$

Z). (4)
Projection Projection (shrunk)
Manifold case. In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres S (e.g. the circle S
1 ⊂ R
2). For such manifolds, the Bayes optimal answer is given by the following proposition.

Proposition 3. For pX defined as in Subsection *2.2.2, with* P being the orthogonal projection operator to V *, the* d + 1 dimensional linear subspace, with R being the radius of sphere S*, the Bayes optimal answer is*

$$\begin{array}{r l}{{}}&{{}}\\ {f_{o p t}({\bar{X}})=\mathbb{E}[X\mid{\bar{X}}]}\\ {}&{{}={\frac{\int e^{(x,{\bar{X}}_{\parallel})/\sigma_{2}^{2}}\;x\,d S_{x}}{\int e^{(x,{\bar{X}}_{\parallel})/\sigma_{2}^{2}}\;d S_{x}}}}\\ {}&{{}}\\ {}&{{}={\frac{I_{\frac{d+1}{2}}\left(R^{\frac{\|{\bar{X}}_{\parallel}\|}{\sigma_{2}^{2}}}\right)}{I_{\frac{d-1}{2}}\left(R^{\frac{\|{\bar{X}}_{\parallel}\|}{\sigma_{2}^{2}}}\right)}R^{\frac{{\tilde{X}}_{\parallel}}{\|{\bar{X}}_{\parallel}\|}},}}\end{array}$$
$$({\boldsymbol{5}})$$
, (6)
where X˜∥ = P X˜ and Iν *is the modified Bessel function of* the first kind. Clustering case. For clustering with isotropic Gaussian mixtures {wa,(µa, σ2a)}
p a=1, the Bayes optimal predictors for some important special cases are as follows. See Appendix C.3 for the general case. Proposition 4. *For general isotropic Gaussian model with* σa = σ0, ||µa|| = R for all a = 1*, . . . , K*.

$$f_{o p t}({\bar{X}})=\mathbb{E}[X|{\bar{X}}]$$
$$=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}\tilde{X}+\frac{\sigma_{Z}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}\frac{\sum_{a}w_{a}e^{\langle\mu_{a},\tilde{X}\rangle/(\sigma_{0}^{2}+\sigma_{Z}^{2})}}{\sum_{a}w_{a}e^{\langle\mu_{a},\tilde{X}\rangle/(\sigma_{0}^{2}+\sigma_{Z}^{2})}}\frac{\mu_{a}}{\sum_{a}w_{a}e^{\langle\mu_{a},\tilde{X}\rangle/(\sigma_{0}^{2}+\sigma_{Z}^{2})}}.\tag{7}$$
$$\left(7\right)$$
$$I\!f\,\sigma_{0}\to0$$
$$f_{opt}(\tilde{X})=\mathbb{E}[X\mid\tilde{X}]=\frac{\sum_{a}w_{a}e^{(\mu_{a},X)/\sigma_{Z}^{2}}\ \mu_{a}}{\sum_{a}w_{a}e^{(\mu_{a},\tilde{X})/\sigma_{Z}^{2}}}.\tag{8}$$

$$({\mathfrak{I}})$$
$$(4)$$

In all three cases, we notice similarities between the form of the Bayes optimal predictor, and attention operations in transformers, a connection which we explore below.

## 3. In-Context Denoising With One-Layer Transformers - Empirical Results

In this section, we provide simple constructions of one-layer transformers that approximate (and under certain conditions, exactly match) the Bayes optimal predictors above. Input: Let p
(1)
X *, . . . , p*
(N) X
iid∼ D, be distributions sampled for one of the tasks. For each distribution p
(i)
X , we sample E(i):= (X
(i)
1
, . . . , X(i)
L
, X˜(i)) taking value in R
n×(L+1)
be an input to a sequence model. We also retain the true
(L + 1)-th token X
(i)
L+1 for each i.

Objective: Given an input sequence E(i), return the uncorrupted final token X
(i)
L+1. We consider the meansquared error loss over a collection of N training pairs,
{E(i), X(i)
L+1}
N
i=1,

$$C(\theta)=\sum_{i=1}^{N}\|F(E^{(i)},\theta)-x_{L+1}^{(i)}\|^{2},\tag{9}$$

where F(E(i), θ) denotes the parametrized function predicting the target final token based on input sequence E(i).

3.1. One-layer transformer and the attention between the query and pure tokens To motivate our choice of architecture, let us start by discussing the linear case.

There we have fopt(X˜) = σ 2 0 σ 20+σ 2 Z
P X˜. Note that, by the strong law of large numbers, Pˆ =1 σ 20L
PL
t=1 XtXT
tis a random matrix that almost surely converges component-bycomponent to the orthogonal projection P as L → ∞, since, for each t, XtXT
t has the expectation σ 20P and that Xt is a Gaussian random variable with zero mean and a finite covariance matrix. So we could propose

$$f(\tilde{X})=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}\hat{P}\tilde{X}=\frac{1}{(\sigma_{0}^{2}+\sigma_{Z}^{2})L}\sum_{t=1}^{L}X_{t}\langle X_{t},\tilde{X}\rangle.\tag{10}$$

We now consider a simplified one-layer linear transformer (see Appendices D.1 and D.2 for more detailed discussions) which still has sufficient expressive power to capture our finite sample approximation to the Bayes optimal answer. We define

$$\hat{X}=F_{\rm Lin}(E,\theta):=\frac{1}{L}W_{PV}X_{1:L}X_{1:L}^{T}W_{KQ}\tilde{X}\tag{11}$$  taking values in $\mathbb{R}^{n}$, where $X_{1:L}:=[X_{1},\ldots,X_{L}]$ taking 
values in R
n×L, with learnable weights WKQ, WP V ∈
R 
n×n abbreviated by θ. Note that, when WP V =
αIn, WKQ = βIn, and αβ =1 σ 20+σ 2Z
, F(*E, θ*) should approximate the Bayes optimal answer fopt(X˜) as L → ∞.

For a detailed discussion of the convergence rate, see Appendix E, in general, and Proposition 5, in particular. Similarly, we could argue that the second two problems, the d-dimesional spheres and the σ0 → 0 zero limit of the Gaussian mixtures could be addressed by softmax attention

$$\hat{X}=F(E,\theta):=W_{PV}X_{1:L}\mbox{softmax}(X_{1:L}^{T}W_{KQ}\hat{X})\tag{12}$$

taking values in R
n. The function softmax(z) :=
P
1 n i=1 e zi(e z1*, . . . , e*zn )
T ∈ R
n is applied column-wise.

For both problems, namely the spheres and the σ0 → 0 Gaussian mixtures, we could have WP V = αIn, WKQ =
βIn with α = 1, β = 1/σ2Z
providing Bayes optimal answers as L → ∞. In fact, we could make a more general statement about distributions pX where the norm of X is fixed.

Theorem 3.1. If we have a task distribution D so that the support of each pX is the subset of some sphere, centered around the origin, with a pX-dependent radius R, then the function

$$F((\{X_{t}\}_{t=1}^{L},\tilde{x}),\theta^{*})=\frac{\sum_{t=1}^{L}X_{t}e^{\langle X_{t},\tilde{x}\rangle/\sigma_{Z}^{2}}}{\sum_{t=1}^{L}e^{\langle X_{t},\tilde{x}\rangle/\sigma_{Z}^{2}}}\tag{13}$$

converges almost surely to the Bayes optimal answer fopt(˜x)
for all x˜ ∈ R
n, as L → ∞*. The optimal parameter* θ
∗
refers to WP V = In, WKQ =1 σ 2 Z
In.

The proof of the theorem is in Appendix D.3. See Appendix E, particularly Proposition 6, for consideration of convergence rates. Note that the condition of pX being supported on a sphere is not artificial as, in many practical transformers, pre-norm with RMSNorm gives you inputs on the sphere, up to learned diagonal multipliers. Note that the natural form of attention that is suggested by our formulation of in-context denoising would involve Gaussian kernels:

$$\hat{X}=F_{G}(E,\theta):=\frac{\sum_{t}W_{PV}X_{t}e^{-\frac{1}{2}||W_{K}X_{t}-W_{Q}\,\hat{X}||^{2}}}{\sum_{t}e^{-\frac{1}{2}||W_{K}X_{t}-W_{Q}\,\hat{X}||^{2}}}.\tag{14}$$

The relation between softmax attention and the Gaussian kernel has been noted in (Choromanski et al., 2021; Ambrogioni, 2024) and a Gaussian kernel-based attention is implemented in (Chen et al., 2021). A related Hopfield energy, with WK, WQ, and WP V proportional to identity matrices, is proposed in (Hoover et al., 2024a). For the linear case, we use linear attention, but that may not be essential. Informally speaking, the softmax attention model has the capacity to subsume the linear attention model. Proposition 3.2. As ϵ → 0,

  **Proposition 5.2.** As $\epsilon\to0$,  $$F\Big{(}E,\big{(}\frac{1}{\epsilon}W_{PV},\epsilon W_{KQ}\big{)}\Big{)}=\frac{1}{\epsilon}W_{PV}\bar{X}$$ $$+\frac{1}{L}W_{PV}\sum_{t=1}^{L}X_{t}(X_{t}-\bar{X})^{T}W_{KQ}\bar{X}+O(\epsilon),\tag{15}$$  _where $\bar{X}=\frac{1}{L}\sum_{t=1}^{L}X_{t}$ is the empirical mean._
See Appendix F for the details of small WKQ expansion and Appendix F.1 for the proof of Proposition 3.2.

For case 1, note that E[Xt] = 0 and covariance of Xt is finite, E[X¯] = 0, and E[||X¯||2] = O(
1 L
), allowing us to drop X¯ as L → ∞. If, in addition, ϵ is small, only the second term survives. Thus, FE,(
1 ϵWP V *, ϵW*KQ)starts to approximate FLinE,(WP V , WKQ)when L is large and ϵ is small, with ϵ
√L large. We therefore could use the softmax model for all three cases.

## 3.2. Case 1 - Linear Manifolds

The Bayes optimal predictor for the linear denoising task from Section 2.3 suggests that the linear attention weights should be scaled identity matrices with their product satisfying αβ =1 σ 2 0+σ 2 Z
. Fig. 3 shows that a one-layer network of size n = 16 trained on tasks with σ 2Z = 1, σ2 0 =
2, d = 8, L = 500 indeed achieves this bound, training to nearly diagonal weights with the appropriate scale ⟨w
(ii)
KQ⟩⟨w
(ii)
P V ⟩ = 0.327 ≈ 1/3 (similar weights are learned for each seed, up to a sign flip).

(a) Case 1: Linear manifolds Case 2: Nonlinear manifolds Case 3: Gaussian mixtures train softmax test train linear test
(b)
Epoch Epoch Epoch Initial weights Final weights (≈ diagonal) Initial weights Final weights Final weights: linear softmax
Fig. 4(a) displays how this bound is approached as the context length L of training samples is increased. In Fig. 4(b) we study how the performance of a model trained to denoise random subspaces of dimension d = 8 is affected by shifts in the subspace dimension at inference time. We find that when provided sufficient context, such models can adapt with mild performance loss to solve more challenging tasks not present in the training set. It is evident from Fig. 3(a) that the softmax network performs similarly to the linear one for this task. We can understand this through the small argument expansion of the softmax function mentioned above. The learned weights displayed in Fig. 3(b) indicate that β softmax ≈ 0.194 becomes small (note it decreases by a factor ϵ ≈ 0.344 relative to β linear), while the value scale α softmax ≈ 1.607 becomes larger by a similar factor ∼ 1/ϵ to compensate. Thus, although the optimal denoiser for this case is intuitively expressed through linear self-attention, it can also be achieved with softmax self-attention in the appropriate limit. Moreover, we find that when the entire prompt undergoes a global invertible transformation A ̸= I, the optimal attention weights are no longer scaled identity matrices but acquire a structured form determined by A. Both linear and softmax attention layers are able to recover this structure through training; see Appendix H for details and empirical verification.

## 3.3. Case 2 - Nonlinear Manifolds

Fig. 3 (case 2) shows networks of size n = 16 trained to denoise subspheres of dimension d = 8 and radius R = 1, with corruption σ 2 Z = 0.1 and context length L = 500. Once again, the network trains to have scaled identity weights. We note that although the network nearly achieves the optimal MSE on the test set, the weights appear at first glance to deviate slightly from the Bayes optimal predictor of Section 2.3, which indicated WP V = αI, WKQ = βI with α = 1, β = 1/σ2Z
. To better understand this, we consider a coarse-grained MSE loss landscape by scanning over α and β. See Fig. 6(a) in Appendix G. We find that the 2D loss landscape has roughly hyperbolic level sets which is suggestive of the linear attention limit, where the weight scales become constrained by their product αβ. Reflecting the symmetry of the problem, we also note mirrored negative solutions (i.e. one could also identify α = −1, β = −1/σ2Z
from the analysis in Section 2.3). Importantly, the plot shows that the trained network lies in the same valley of the loss landscape as the optimal predictor, in agreement with Fig. 3. Moreover, the shape of the loss landscape suggested that linear attention might also be applicable to this case, which we demonstrate and discuss further in Appendix G.

## 3.4. Case 3 - Gaussian Mixtures

Figure 3 (case 3) shows networks of size n = 16 trained to denoise balanced Gaussian mixtures with p = 8 compo-

(a) Eect of context length L on training Shifting the subspace dimension at inference time subspace dimension can vary
(b)
Predict Mean diagonal weights of trained network we ig ht sc ali ng Performance maintained away from d=8 L=50 L=30 L=500 Train n=16 model: d=8, L=500 Loss of trained network linear projection train test in-context learning subspace provided only via context
nents that have isotropic variance σ 20 = 0.02 and centers randomly placed on the unit sphere in R
n. The corruption magnitude is σ 2Z = 0.1 and context length is L = 500. The baselines show the zero predictor (dashed grey line) as well as the optimum from Proposition (4) (pink) and its σ 2 0 → 0 approximation Eq. (8) (grey). The trained weights qualitatively approach the optimal estimator for the zero-variance limit but with a slightly different scaling: while the scale of WP V is α ≈ 1, the WKQ scale is β ≈ 5.127 < 1/σ2Z. To study this, we provide a corresponding plot of the 2D loss landscape in Fig. 6(a) in Appendix G. While the symmetry of the previous case has been broken
(the context cluster centers {µa} will not satisfy ⟨µ⟩ = 0),
we again find that the trained network lies in the anticipated global valley of the MSE loss landscape.

## 4. Connection To Dense Associative Memory Networks

In each of the denoising problems studied above, we have shown analytically and empirically that the optimal weights of the one-layer transformer are scaled identity matrices WP V ≈ *αI, W*KQ ≈ βI. In the softmax case, the trained denoiser can be concisely expressed as

$$\hat{x}=g(X_{1:L},\bar{x}):=\alpha X_{1:L}\mathrm{softmax}(\beta X_{1:L}^{T}\bar{x}),$$

re-written such that X ∈ R
n×L stores pure context tokens.

We now demonstrate that such denoising corresponds to one-step gradient descent (with specific step sizes) of energy models related to dense associative memory networks, also known as modern Hopfield networks (Ramsauer et al., 2021; Demircigil et al., 2017; Krotov & Hopfield, 2016).

Consider the energy function:

$${\cal E}(X_{1:L},s)=\frac{1}{2\alpha}\|s\|^{2}-\frac{1}{\beta}\log\left(\sum_{t=1}^{L}e^{\beta X_{t}^{T}s}\right),\tag{16}$$

which mirrors the Ramsauer et al. (2021) construction but with a Lagrange multiplier added to the first term. Figure 5 illustrates this energy landscape for the spherical manifold case.

Num. steps: 1 Num. steps: 50 context tokens query target prediction trajectories
Figure 5. Gradient descent denoising for the nonlinear manifold case (spheres) in n = 2 with d = 1. A context-aware dense associative memory network E(X1:L, s) is constructed whose gradient corresponds to the Bayes optimal update (trained attention layer). Note that the density of sampled context tokens sculpts the valleys of the energy landscape. Left: the attention step of a one-layer transformer trained on the denoising task corresponds to a single gradient descent step. Right: Iterating the denoising process—as is conventional for Hopfield networks—can potentially degrade the estimate by causing it to become query-independent (e.g. converging to a distant minimum). Here R *= 1, σ*2Z = 10, L = 20 and α *= 1, β* = 1/σ2Z .

An operation inherent to the associative memory perspective is the recurrent application of a denoising update. Gradient descent iteration s(t + 1) = s(t) − γ ∇sEX1:L, s(t)
yields

$$s(t+1)=\left(1-\frac{\gamma}{\alpha}\right)s(t)+\gamma X_{1:L}\text{softmax}\big{(}\beta X_{1:L}^{T}s(t)\big{)}.\tag{17}$$

It is now clear that initializing the state to the query s(0) = ˜x and taking a single step with size γ = α recovers the behavior of the trained attention model (Fig. 5). The attention mechanism here is thus mechanistically interpretable: the context tokens X1:L induce a context-dependent associative memory landscape, while the query acts as an initial condition for inference-time gradient descent. One could naturally consider alternative step sizes and recurrent iteration. However, Fig. 5 demonstrates that naive iteration of Eq. (17) has the potential to degrade performance. Additional details are provided in Appendix I. In particular, the energy model for linear attention is discussed in Appendix I.1.

## 5. Discussion

Motivated by the connection between attention mechanisms and dense associative memories, here we have introduced incontext denoising, a task that distills their relationship. We first analyze the general problem, deriving Bayes optimal predictors for certain restricted tasks. We identify that onelayer transformers using either softmax or linearized selfattention are expressive enough to describe these predictors. We then empirically demonstrate that standard training of attention layers from random initial weights will readily converge to scaled identity weights with scales that approach the derived optima given sufficient context. Accordingly, the rather minimal transformers studied here can perform optimal denoising of novel tasks provided at inference time via self-contained prompts. This work therefore sheds light on other in-context learning phenomena, a point we return to below. While practical transformers differ in various ways from the minimal models studied here, we note several key connections. Intriguingly, the self-attention heads of trained transformers sometimes exhibit weights WKQ, WP V that resemble scaled identity matrices, i.e. cI +ϵ with small fluctuations ϵij ∼ N (0, σ2), an observation noted in Trockman
& Kolter (2023). This phenomenon motivated their proposal of "mimetic" weight initialization schemes mirroring this learned structure. Relatedly, connections to associative memory concepts have been explored in other architectures
(Smart & Zilman, 2021), which enabled data-dependent weight initialization strategies to be identified and leveraged.

More broadly, our study suggests that trained attention layers can readily adopt structures that facilitate context-aware associative retrieval. We have also noted preliminary connections between our work and other architectural features of modern transformers, namely layer normalization and residual streams, which warrant further study. In-context denoising and generative modeling both involve learning about an underlying distribution, suggesting potential relationships between these two tasks. Recently, Pham et al. (2024) invoked spurious states of the Hopfield model as a way of understanding how one can move away from retrieving individual memorized patterns towards generalization via appropriate mixtures of multiple similar "memories". In our work, one-step updates do not have to land in a spurious minimum, but we often operate under circumstances where there are such states (see, for example, the energy landscape in Fig. 5). More generally, analogies between energy-based associative memory and diffusion models have recently been noted (Ambrogioni, 2024; Hoover et al., 2024b). Lastly, Bayes optimal denoisers play an important role in the analysis (Ghio et al., 2024) of a very related generative model that is based on stochastic interpolants (Albergo & Vanden-Eijnden, 2023). Although this work focuses on the case where it is possible to sample enough tokens from the relevant distributions for certain functions to converge, generative models become important when the distribution is in a prohibitively high-dimensional space making direct sampling difficult. Nonetheless, investigating the precise relationship between our work and different generative modeling approaches would be an interesting direction to pursue. Overall, this work refines the connection between dense associative memories and attention layers first identified in (Ramsauer et al., 2021). While we show that one energy minimization step of a particular DAM (associated with a trained attention layer) is optimal for the denoising tasks studied here, it remains an open question whether multilayer architectures with varying or tied weights could extend these results to more complex tasks by effectively performing multiple iterative steps. This aligns with recent studies on in-context learning, which have considered whether transformers with multiple layers emulate gradient descent updates on a context-specific objective (Von Oswald et al., 2023; Shen et al., 2024; Dai et al., 2023; Ahn et al.,
2023), and may provide a bridge to work on emerging architectures guided by associative memory principles (Hoover et al., 2023). Investigating when and how multilayer attention architectures perform such gradient descent iterations in a manner that is both context-dependent and informed by a large training set represents an exciting direction for future research at the intersection of transformer mechanisms, associative memory retrieval, and in-context learning.

## Software And Data

Python code underlying this work is available at https://github.com/mattsmart/in-context-denoising.

## Acknowledgements

MS acknowledges M. Mezard for very useful feedback on ´ an earlier version of this work. AS thanks D. Krotov and P. Mehta for enlightening discussions on related matters.

Our early work also benefited from AS's participation in the deeplearning23 workshop at the Kavli Institute for Theoretical Physics (KITP), which was supported in part by grants NSF PHY-1748958 and PHY-2309135 to KITP. AS thanks Y. Bahri and C. Pehlevan for their patience and willingness to listen to our early ideas at KITP.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. Transformers learn to implement preconditioned gradient descent for in-context learning. Advances in Neural Information Processing Systems, 36:45614–45650, 2023.

Akyurek, E., Schuurmans, D., Andreas, J., Ma, T., and ¨
Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?

id=0g0X4H8yN4I.

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/ 2209.15571.

Amari, S.-I. Learning patterns and pattern sequences by selforganizing nets of threshold elements. IEEE Transactions on computers, 100(11):1197–1206, 1972.

Ambrogioni, L. In search of dispersed memories: Generative diffusion models are associative memory networks. *Entropy*, 26(5), 2024. ISSN 1099-4300. doi: 10. 3390/e26050381. URL https://www.mdpi.com/ 1099-4300/26/5/381.

Amit, D. J., Gutfreund, H., and Sompolinsky, H. Spinglass models of neural networks. *Physical Review A*, 32
(2):1007–1018, 1985. ISSN 10502947. doi: 10.1103/ PhysRevA.32.1007.

Bolle, D., Nieuwenhuizen, T. M., Castillo, I. P., and Ver- ´
beiren, T. A spherical hopfield model. *Journal of Physics* A: Mathematical and General, 36(41):10269, 2003.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Chen, Y., Zeng, Q., Ji, H., and Yang, Y. Skyformer: Remodel self-attention with gaussian kernel and nystrom¨
method. *Advances in Neural Information Processing* Systems, 34:2122–2135, 2021.

Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=Ua6zuk0WRH.

Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei, F. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023.

URL https://arxiv.org/abs/2212.10559.

Demircigil, M., Heusel, J., Lowe, M., Upgang, S., and ¨
Vermet, F. On a model of associative memory with huge storage capacity. *Journal of Statistical Physics*, 168:288– 299, 2017.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert:
Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://
openreview.net/forum?id=YicbFdNTTy.

Fischer, K. H. and Hertz, J. A. *Spin Glasses*. Cambridge University Press, 1993.

Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. In *Advances in Neural Information* Processing Systems, volume 35, pp. 30583–30598, 2022. URL https://arxiv.org/abs/2208.01066.

Ghio, D., Dandi, Y., Krzakala, F., and Zdeborova, L. Sam- ´
pling with flows, diffusion, and autoregressive neural networks from a spin-glass perspective. Proceedings of the National Academy of Sciences, 121(27):e2311810121, 2024.

Gradshteyn, I. S. and Ryzhik, I. M. Table of Integrals, Series, and Products. Elsevier/Academic Press, Amsterdam, seventh edition, 2007.

Hoeffding, W. Probability inequalities for sums of bounded random variables. The collected works of Wassily Hoeffding, pp. 409–426, 1994.

Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H., Zaki, M. J., and Krotov, D. Energy transformer. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=MbwVNEx9KS.

Hoover, B., Chau, D. H., Strobelt, H., Ram, P., and Krotov, D. Dense associative memory through the lens of random features. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a.

Hoover, B., Strobelt, H., Krotov, D., Hoffman, J., Kira, Z.,
and Chau, D. H. Memory in plain sight: Surveying the uncanny resemblances of associative memories and diffusion models, 2024b. URL https://arxiv.org/ abs/2309.16750.

Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences of the United States of America, 79(8):2554–2558, 1982. ISSN 00278424.

doi: 10.1073/pnas.79.8.2554.

Hu, J. Y.-C., Yang, D., Wu, D., Xu, C., Chen, B.-Y., and Liu, H. On sparse modern hopfield model. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS '23, 2023.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.

Transformers are rnns: fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML'20.

JMLR.org, 2020.

Krotov, D. A new frontier for hopfield networks. Nature Reviews Physics, 5(7):366–367, 2023.

Krotov, D. and Hopfield, J. J. Dense associative memory for pattern recognition. In Advances in Neural Information Processing Systems, volume 29, 2016.

Krotov, D. and Hopfield, J. J. Large associative memory problem in neurobiology and machine learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=X4y_10OX-hX.

Little, W. A. The existence of persistent states in the brain.

Mathematical biosciences, 19(1-2):101–120, 1974.

Loeve, M. Probability theory i. ` Graduate Texts in Mathematics, 1977.

Lucibello, C. and Mezard, M. Exponential capacity of ´
dense associative memories. *Phys. Rev. Lett.*, 132: 077301, Feb 2024. doi: 10.1103/PhysRevLett.132. 077301. URL https://link.aps.org/doi/10. 1103/PhysRevLett.132.077301.

Millidge, B., Salvatori, T., Song, Y., Lukasiewicz, T., and Bogacz, R. Universal hopfield networks: A general framework for single-shot associative memory models. In International Conference on Machine Learning, pp. 15561– 15583. PMLR, 2022.

Nakano, K. Associatron-a model of associative memory.

IEEE Transactions on Systems, Man, and Cybernetics, 2:
380–388, 1972.

Pham, B., Raya, G., Negri, M., Zaki, M. J., Ambrogioni, L., and Krotov, D. Memorization to generalization: The emergence of diffusion models from associative memory. In NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning, 2024.

Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich, M., ¨
Gruber, L., Holzleitner, M., Adler, T., Kreil, D. P., Kopp, M. K., Klambauer, G., Brandstetter, J., and Hochreiter, S. Hopfield networks is all you need. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum? id=tL89RnzIiCd.

Reddy, G. The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=aN4Jf6Cx69.

Rigollet, P. and Hutter, J.-C. High-dimensional statistics. ¨
arXiv preprint arXiv:2310.19244, 2023.

Santos, S. J. R. D., Niculae, V., Mcnamee, D. C., and Martins, A. Sparse and structured hopfield networks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 43368–43388. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/
v235/santos24a.html.

Shen, L., Mishra, A., and Khashabi, D. Position: Do pretrained transformers learn in-context by gradient descent? In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 44712–44740. PMLR, 21– 27 Jul 2024. URL https://proceedings.mlr. press/v235/shen24d.html.

Smart, M. and Zilman, A. On the mapping between hopfield networks and restricted boltzmann machines. International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=RGJbergVIoO.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., ` Azhar, F., et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

Trockman, A. and Kolter, J. Z. Mimetic initialization of selfattention layers. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Łukasz Kaiser, and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, volume 2017-December, 2017.

Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. In *International Conference on Machine Learning*, pp. 35151–35174. PMLR, 2023.

Wu, D., Hu, J. Y.-C., Hsiao, T.-Y., and Liu, H. Uniform memory retrieval with larger capacity for modern hopfield models. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2024a.

Wu, D., Hu, J. Y.-C., Li, W., Chen, B.-Y., and Liu, H.

Stanhop: Sparse tandem hopfield model for memoryenhanced time series prediction. In The Twelfth International Conference on Learning Representations, 2024b. URL https://arxiv.org/abs/2312.17346.

Zhang, R., Frei, S., and Bartlett, P. L. Trained transformers learn linear models in-context. *Journal of Machine* Learning Research, 25(49):1–55, 2024. URL http: //jmlr.org/papers/v25/23-1042.html.

## A. Notation

A.1. Recurring notation
- n - ambient dimension of input tokens.

- xt ∈ R
n - the value of the t-th random input token.

- E = (X1*, ..., X*L, X˜) - the random variable input to the sequence model. The "tilde" indicates that the final token has in some way been corrupted. E takes values (x1*, ..., x*L, x˜) ∈ R
n×(L+1). Note: while capital X or Xi here denotes a random variable, in Section D use X1:L or simply X to refer to the realized matrix of input tokens.

- L - context length = number of uncorrupted tokens.

- d - dimensionality of manifold S that xt are sampled from
- N - number of training pairs

## A.2. Bayes Posterior Notation

- pX(x) is task-dependent (the three scenarios considered here are introduced above).

- pX˜ (˜x) where x˜ = x + z. For a sum of independent random variables, Y = X1 + X2, their pdf is a convolution pY (y) = RpX1
(x)pX2
(y − x)dx. Thus:

$$p_{\tilde{X}}(\tilde{x})=\int p_{Z}(z)p_{X}(\tilde{x}-z)dz$$ $$=C_{Z}\int e^{-\|z\|^{2}/2\sigma_{Z}^{2}}p_{X}(\tilde{x}-z)dz$$

where CZ = (2πσ2Z)
−n/2is a constant.

- pX˜|X(˜x | x): This is simply

$$v_{Z}(\tilde{x}-x)=C_{Z}e^{-\|\tilde{x}-x\|^{2}/2\sigma_{Z}^{2}}.$$

- pX|X˜ (x | x˜): By Bayes' theorem, this is

$$P_{X|\tilde{X}}(x\mid\tilde{x})=\frac{P_{\tilde{X}|X}(\tilde{x}\mid x)p_{X}(x)}{P_{\tilde{X}}(\tilde{x})}$$ $$=\frac{e^{-\|\tilde{x}-x\|^{2}/2\sigma_{\tilde{Z}}^{2}}p_{X}(x)}{\int e^{-\|\tilde{x}-x^{\prime}\|^{2}/2\sigma_{\tilde{Z}}^{2}}p_{X}(x^{\prime})dx^{\prime}}.$$
.
- Posterior mean:

$$\mathbb{E}_{X|\tilde{X}}[X\mid\tilde{X}]=\int x\,p_{X|\tilde{X}}(x\mid\tilde{X})dx$$ $$=\frac{\int x\,e^{-\|\tilde{X}-x\|^{2}/2\sigma_{2}^{2}}p_{X}(x)dx}{\int e^{-\|\tilde{X}-x\|^{2}/2\sigma_{2}^{2}}p_{X}(x)dx}.$$

12

## B. Bayes Optimal Predictors For Square Loss

B.1. Proof of Proposition 1 Proof. Observe that

$$\mathbb{E}\left[\|X-f(\tilde{X})\|^{2}\right]=\mathbb{E}_{\tilde{X}}\left[\mathbb{E}_{X\,|\tilde{X}}\left[\|X-f(\tilde{X})\|^{2}\mid\tilde{X}\right]\right]$$ $$=\mathbb{E}_{\tilde{X}}\left[\mathbb{E}_{X\,|\tilde{X}}\left[\|X-\mathbb{E}[X\mid\tilde{X}]\|^{2}\mid\tilde{X}\right]\right.$$ $$\left.+\,\|\mathbb{E}[X\mid\tilde{X}]-f(\tilde{X})\|^{2}\right]$$ $$\geq\mathbb{E}_{\tilde{X}}\left[\mathbb{E}_{X\,|\tilde{X}}\left[\|X-\mathbb{E}[X\mid\tilde{X}]\|^{2}\mid\tilde{X}\right]\right]$$ $$=\mathbb{E}_{\tilde{X}}\left[\operatorname{Tr}\operatorname{Cov}(X\mid\tilde{X})\right].$$

$\square$
Note the final line is independent of f. This inequality becomes an equality when f(X˜) = E[X | X˜].

## C. Details Of Bayes Optimal Denoising Baselines For Each Case C.1. Proof Of Proposition 2

Proof. The linear denoising task is a special case of the result in Proposition 1. Here, X is an isotropic Gaussian in a restricted subspace,

$$p_{X|{\tilde{X}}}(x\mid{\tilde{x}})=C({\tilde{x}})p_{X}(x)e^{-{\frac{\|x-{\tilde{x}}\|^{2}}{2\sigma_{Z}^{2}}}}$$

where C(˜x) is a normalizing factor. The noise can be decomposed into parallel and perpendicular parts using the projection P onto S, i.e.

$$\tilde{X}=\tilde{X}_{\parallel}+\tilde{X}_{\perp}=P\tilde{X}+(I-P)\tilde{X},$$

so that

$$e^{-{\frac{\|x-{\bar{x}}\|^{2}}{2\sigma_{Z}^{2}}}}=e^{-{\frac{\|x-{\bar{x}}_{\parallel}\|^{2}}{2\sigma_{Z}^{2}}}}\;e^{-{\frac{\|{\bar{x}}_{\perp}\|^{2}}{2\sigma_{Z}^{2}}}}.$$

Only the first factor matters for pX|X˜ (x | x˜) since it depends on x. Then, for x ∈ S, the linear subspace supporting pX,
dropping the x independent x˜⊥ contribution,

$$p_{X}(x)e^{-\frac{\|x-\tilde{x}_{\parallel}\|^{2}}{2\sigma_{Z}^{2}}}\propto e^{-\frac{\|x\|^{2}}{2\sigma_{0}^{2}}-\frac{\|x-\tilde{x}_{\parallel}\|^{2}}{2\sigma_{Z}^{2}}}$$ $$\propto\exp\left(-\frac{\|x-\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}\tilde{x}_{\parallel}\|^{2}}{2\frac{\sigma_{0}^{2}\sigma_{Z}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}}\right).$$
Thus, f(X˜) = σ
$$f(\tilde{X})=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{2}^{2}}\tilde{X}_{\parallel}=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{2}^{2}}P\tilde{X}.$$

Using X˜ = X + Z, X = P X, and the independence of X and Z

$$\mathbb{E}\Big[\|X-\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}P\bar{X}\|^{2}\Big]=\mathbb{E}\Big[\|\frac{\sigma_{Z}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}P X\|^{2}\Big]+\mathbb{E}\Big[\|\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}P Z\|^{2}\Big]=\frac{\sigma_{Z}^{4}d\sigma_{0}^{2}+\sigma_{0}^{4}d\sigma_{Z}^{2}}{(\sigma_{0}^{2}+\sigma_{Z}^{2})^{2}}=\frac{d\sigma_{0}^{2}\sigma_{Z}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}.$$

## C.2. Proof Of Proposition 3

Proof. In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres S (e.g. the circle S
1 ⊂ R
2). For such manifolds, we have

$$\begin{split}\mathbb{E}[X\mid\tilde{X}=\tilde{x}]&=\frac{\int e^{-\frac{\|x-\tilde{x}_{\parallel}\|^{2}}{2\sigma_{Z}^{2}}}x\,p_{X}(x)dx}{\int e^{-\frac{\|x-\tilde{x}_{\parallel}\|^{2}}{2\sigma_{Z}^{2}}}p_{X}(x)dx}\\ &=\frac{\int e^{\langle x,\tilde{x}_{\parallel}\rangle/\sigma_{Z}^{2}}\ x\,d S_{x}}{\int e^{\langle x,\tilde{x}_{\parallel}\rangle/\sigma_{Z}^{2}}\ d S_{x}}.\end{split}$$

We have used the fact that ∥x − x˜∥∥
2 = ∥x∥
2 + ∥x˜∥∥
2 − 2⟨x, x˜∥⟩ and that ∥x∥ is fixed on the sphere.

The integrals can be evaluated directly once the parameters are specified. If S is a d–sphere of radius R, then the optimal predictor is again a shrunk projection of x˜ onto S,

$$\frac{\int_{0}^{\pi}e^{R\|\tilde{x}_{\parallel}\|\cos\theta/\sigma_{Z}^{2}}\,\cos\theta\sin^{(d-1)}\theta\,d\theta}{\int_{0}^{\pi}e^{R\|\tilde{x}_{\parallel}\|\cos\theta/\sigma_{Z}^{2}}\,\sin^{(d-1)}\theta\,d\theta}R\frac{\tilde{x}_{\parallel}}{\|\tilde{x}_{\parallel}\|}$$ $$=\frac{I_{\frac{d+1}{2}}\left(R\frac{\|\tilde{x}_{\parallel}\|}{\sigma_{Z}^{2}}\right)}{I_{\frac{d-1}{2}}\left(R\frac{\|\tilde{x}_{\parallel}\|}{\sigma_{Z}^{2}}\right)}R\frac{\tilde{x}_{\parallel}}{\|\tilde{x}_{\parallel}\|},$$
$\square$
where we used identities involving Iν(y), modified Bessel function of the first kind of order ν (Gradshteyn & Ryzhik, 2007). The vector R
x˜∥
∥x˜∥∥
is the point on S in the direction of x∥.

## C.3. Proof Of Proposition 4

Proof. For the clustering case involving isotropic Gaussian mixtures with parameters {wa,(µa, σ2a)}
p a=1,

$$\mathbb{E}[X\mid{\bar{X}}={\bar{x}}]={\frac{\int e^{-{\frac{\|x-{\bar{x}}\|^{2}}{2\sigma_{\bar{x}}^{2}}}}\sum_{a}\left(w_{a}C_{a}e^{-{\frac{\|x-\mu_{a}\|^{2}}{2\sigma_{a}^{2}}}}\right)x\,d x}{\int e^{-{\frac{\|x-{\bar{x}}\|^{2}}{2\sigma_{\bar{x}}^{2}}}}\sum_{a}\left(w_{a}C_{a}e^{-{\frac{\|x-\mu_{a}\|^{2}}{2\sigma_{a}^{2}}}}\right)\,d x}},$$

where Ca = (2πσ2a)
−
n 2 .

We can simplify this expression by completing the square in the exponent and using the fact that the integral of a Gaussian about its mean is zero. This yields

$$\mathbb{E}[X\mid{\tilde{X}}={\tilde{x}}]={\frac{\sum_{a}w_{a}C_{a}m_{a}\int\exp(-g_{a})\,d x}{\sum_{a}w_{a}C_{a}\int\exp(-g_{a})\,d x}}$$

where we have introduced

$$g_{a}=\frac{1}{2}\Big(\frac{\sigma_{Z}^{2}+\sigma_{a}^{2}}{\sigma_{Z}^{2}\sigma_{a}^{2}}\Big)\left\|x-m_{\alpha}\right\|^{2}\,+\,\frac{1}{2(\sigma_{Z}^{2}+\sigma_{a}^{2})}\|\bar{x}-\mu_{a}\|^{2},$$

with

$$m_{a}=\frac{\sigma_{a}^{2}\,\tilde{x}+\sigma_{Z}^{2}\,\mu_{a}}{\sigma_{a}^{2}+\sigma_{Z}^{2}}.$$

14 Doing the integrals and using the expressions for Ca, ma

$$\mathbb{E}[X\mid\bar{X}=\tilde{x}]=\frac{\sum_{a}w_{a}\big(\frac{\sigma_{\alpha}^{2}+\sigma_{a}^{2}}{\sigma_{a}^{2}}\big)^{n/2}\exp\big(-\frac{\|\tilde{x}-\mu_{a}\|^{2}}{2(\sigma_{Z}^{2}+\sigma_{a}^{2})}\big)\big(\frac{\sigma_{\alpha}^{2}\,\tilde{x}+\sigma_{Z}^{2}\,\mu_{a}}{\sigma_{a}^{2}+\sigma_{Z}^{2}}\big)}{\sum_{a}w_{a}\big(\frac{\sigma_{\alpha}^{2}+\sigma_{a}^{2}}{\sigma_{a}^{2}}\big)^{n/2}\exp\big(-\frac{\|\tilde{x}-\mu_{a}\|^{2}}{2(\sigma_{Z}^{2}+\sigma_{a}^{2})}\big)}$$

In the case that the center norms ∥µa∥ are independent of a and variances σ 2a = σ0, we have

$$\mathbb{E}[X\mid\bar{X}=\tilde{x}]=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}\,\tilde{x}+\frac{\sigma_{Z}^{2}}{\sigma_{0}^{2}+\sigma_{Z}^{2}}\,\frac{\sum_{a}w_{a}\mu_{a}\exp\left(\frac{(\tilde{x},\mu_{a})}{\sigma_{Z}^{2}+\sigma_{0}^{2}}\right)}{\sum_{a}w_{a}\exp\left(\frac{(\tilde{x},\mu_{a})}{\sigma_{Z}^{2}+\sigma_{0}^{2}}\right)}.$$
$$\square$$
$$(\mathbf{A.1})$$
$$(\mathbf{A.2})$$

Note that in the limit that σ0 → 0 , this becomes expressible by one-layer self-attention, since one can simply replace the matrix of cluster centers M = [µ1 *. . . µ*p] implicit in the expression with the context X1:L itself,

$$\mathbb{E}[X\mid\tilde{X}]=\frac{\sum_{a}w_{a}e^{\langle\mu_{\alpha},\tilde{X}\rangle/\sigma_{Z}^{2}}\mu_{a}}{\sum_{a}w_{a}e^{\langle\mu_{\alpha},\tilde{X}\rangle/\sigma_{Z}^{2}}}.$$

## D. Additional Details On Attention Layers And Softmax Expansion D.1. Standard Self-Attention

Given a sequence of Lseq input tokens xi ∈ R
n represented as a matrix X ∈ R
n×Lseq , standard self-attention defines query,

key, and value matrices
K = WKX, Q = WQX, V = WV X (A.1) where WK, WQ ∈ R nattn×n and WV ∈ R nout×n. The softmax self-attention map (Vaswani et al., 2017) is then
$$K=W_{K}X,Q=W_{Q}X,V=W_{V}X$$
$$\operatorname{Attn}(X,W_{V},W_{K}^{T}W_{Q}):=V\mathrm{softmax}(K^{T}Q)\in\mathbb{R}^{n_{\mathrm{out}}\times L_{\mathrm{seq}}}.$$
nout×Lseq. (A.2)
On merging WK, WQ into WKQ = WTKWQ: The simplification WKQ = WTKWQ (made here and elsewhere) is general only when nattn ≥ n; in that case, the product WKQ can have rank n and thus it is reasonable to work with the combined matrix. On the other hand, if nattn < n, then the rank of their product is at most nattn and thus there are matrices in R
n×n that cannot be expressed as WTKWQ. A similar point can be made about WP V . We note that while nattn < n may be used in practical settings, one often also uses multiple heads which when concatenated could be (roughly) viewed as a single higher-rank head. We will also use the simplest version of linear attention (Katharopoulos et al., 2020),

$$\operatorname{Attn}_{\operatorname{Lin}}(X,W_{V},W_{K}^{T}W_{Q}):={\frac{1}{L_{\mathrm{seq}}}}V(K^{T}Q)\in\mathbb{R}^{n_{\mathrm{out}}\times L_{\mathrm{out}}}.$$

## D.2. Minimal Transformer Architecture For Denoising

We now consider a simplified one-layer linear transformer in term of our variable E = (X1:L, X˜) taking values in R
n×(L+1)
and start with the linear transformer which still has sufficient expressive power to capture our finite sample approximation to the Bayes optimal answer in the linear case. Inspired by Zhang et al. (2024), we define

$$\operatorname{Attn}_{\mathrm{Lin}}(E,W_{P V},W_{K Q}):={\frac{1}{L}}W_{P V}E M_{\mathrm{Lin}}E^{T}W_{K Q}E$$

taking values in R
n×(L+1). The additional aspect compared to the last subsection is the masking matrix MLin ∈
R

(L+1)×(L+1) which is of the form
$$M_{\mathrm{Lin}}=\begin{bmatrix}I_{L}&0_{L\times1}\\ 0_{1\times L}&0\end{bmatrix},$$
, (A.5)
$$(\mathbf{A.3})$$
$$(\mathbf{A.4})$$

$$(\mathbf{A}.S)$$
15 preventing WP V X˜ from being added to the output.

Note that this more detailed expression is equivalent to the form used in the main text.

$${\hat{X}}=F_{\mathrm{Lin}}(E,\theta):={\frac{1}{L}}W_{P V}X_{1:L}X_{1:L}^{T}W_{K Q}{\tilde{X}}$$

With learnable weights WKQ, WP V ∈ R
n×n abbreviated by θ, we define

$$F(E,\theta):=[\mathrm{Attn}_{\mathrm{Lin}}(E,W_{P V},W_{K Q})]_{:,L+1}.$$
F(*E, θ*) := [AttnLin(E, WP V , WKQ)]:,L+1. (A.6)

```
Note that, when WP V = αIn, WKQ = βIn, and αβ =1
                                                    σ
                                                     2
                                                     0+σ
                                                        2
                                                        Z
                                                          , F(E, θ) should approximate the Bayes optimal answer

```

fopt(X˜) as L → ∞.

Similarly, we could argue that the second two problems, the d-dimesional spheres and the σ0 → 0 zero limit of the Gaussian mixtures could be addressed by the full softmax attention

$$\mathrm{{Atm}}(E,W_{P V},W_{K Q})=W_{P V}E\mathrm{{softmax}}(E^{T}W_{K Q}E+M)$$
T WKQE + M) (A.7)
taking values in R
n×(L+1) where M ∈ R¯(L+1)×(L+1) is a masking matrix of the form

$$(\mathbf{A.6})$$

$$(\mathbf{A}.7)$$

$$M=\begin{bmatrix}0_{L\times(L+1)}\\ (-\infty)1_{1\times L+1}\end{bmatrix},\tag{1}$$
$$(\mathbf{A.8})$$

$$(\mathbf{A.9})$$
, (A.8)
once more, preventing the contribution of X˜ value to the output. The function softmax(z) := P
1 n i=1 e zi(e z1*, . . . , e*zn )
T ∈ R
n is applied column-wise.

We then define
F(*E, θ*) := [Attn(E, WP V , WKQ)]:,L+1, (A.9)
which is equivalent to the simplified form used in the main text:

$$F(E,\theta):=[\mathrm{Attn}(E,W_{P V},W_{K Q})]_{:,L+1},$$
$$\hat{X}=F(E,\theta):=W_{P V}X_{1:L}\mathrm{softmax}(X_{1:L}^{T}W_{K Q}\bar{X}).$$

## D.3. Proof Of Theorem 3.1

Proof. Let the support of pX be a subset of a sphere, centered around the origin, of radius R. Then the function

$$g(\{X_{t}\}_{t=1}^{L},\tilde{x})=\frac{\sum_{t=1}^{L}X_{t}e^{\langle X_{t},\tilde{x}\rangle/\sigma_{x}^{2}}}{\sum_{t=1}^{L}e^{\langle X_{t},\tilde{x}\rangle/\sigma_{x}^{2}}}=\frac{\frac{1}{L}\sum_{t=1}^{L}X_{t}e^{\langle X_{t},\tilde{x}\rangle/\sigma_{x}^{2}}}{\frac{1}{L}\sum_{t=1}^{L}e^{\langle X_{t},\tilde{x}\rangle/\sigma_{x}^{2}}}.$$ (A.10)
Both the numerator 1L
PL
t=1 Xte
⟨Xt,x˜⟩/σ2Z and the denominator 1L
PL
t=1 e
⟨Xt,x˜⟩/σ2Z are averages of independent and identically distributed bounded random variables. By the strong law of large numbers, as L → ∞, the average vector in the numerator converges to almost surely to Re
⟨x,x˜∥⟩/σ2Z *x dp*X(x) for each component, while the average in the denominator almost surely converges Re
⟨x,x˜∥⟩/σ2Z dpX(x), which is positive. So, as L → ∞, the ratio in Eq. A.10 converges almost surely to

$${\frac{\int e^{\langle x,{\hat{x}}_{\parallel}\rangle/\sigma_{Z}^{2}}\;x\,d p_{X}(x)}{\int e^{\langle x,{\hat{x}}_{\parallel}\rangle/\sigma_{Z}^{2}}\;d p_{X}(x)}},$$

which is the Bayes optimal answer fopt(˜x) for all x˜ ∈ R
n.

## E. Further Discussion Of Convergence Rates As L → ∞ **And The Dependence On Dimensions**

Our analysis primarily focused on the asymptotic behavior as L → ∞ using the strong law of large numbers, which just requires the mean to exist (Loeve ` , 1977). However, in the linear example, our tokens are Gaussian, and in the two nonlinear cases they are bounded. Intuitively, we expect error O( √
1 L
). In fact, we can give precise results of the form that the probability of the difference between the empirical sum for the ideal weights departing from the expectation by less

$$\square$$

than C(˜x)
rfd,ln 1δ Lis greater than 1 − δ. The function C of the query vector and the function f depend on the problem.

Interestingly, these bounds depend on d, the dimension spanned by the tokens, not the ambient dimension n.

As mentioned before, the results of the previous paragraph refer to the convergence of the finite sample attention expressions for ideal weights, namely those corresponding to Bayes optimal answer. There is a second source of error associated with finite sample estimation of weights, which should also get small as L becomes large. Once more the expectation is that the weights are known to error O( √
1 L
) for well-converged training procedures, although this is more difficult to guarantee or quantify analytically. Overall we expect the loss (MSE) to go down inversely with some power of L. Fig. 4(a) provides some empirical evidence for this relationship, showing how performance improves with increasing context length. Notice that the one-layer transformer output is a linear combination of the uncorrupted samples. Hence, if the distribution pX is supported by a d-dimensional linear subspace, the estimate Xˆ is also in that subspace. We can therefore look at convergence restricted to the supporting subspace. Therefore, it is the dimensionality of the supporting subspace that matters.

Let a d-dimensional vector space V be a linear subspace of R
n. We define the maximum norm for V with respect to some orthonormal basis {vi}
d i=1 in V as ||x||∞,V := maxi∈{1,...,d} |⟨vi, x⟩| for any x ∈ V . The conventional maximum norm for R

n, of course, is defined with respect to the standard orthonormal basis {ej}
n j=1. Since |⟨vi, x*⟩| ≤ ||*x||∞,V , for all i,

$$||x||_{2}^{2}=\sum_{i=1}^{d}((v_{i},x))^{2}\leq d||x||_{\infty,V}^{2}\implies||x||_{2}\leq\sqrt{d}||x||_{\infty,V}.$$

Then, for any x ∈ V ⊆ R
n, ||x||∞ ≤
√d||x||∞,V , since |⟨x, ej *⟩| ≤ ||*x||2 ≤
√d||x||∞,V , for all j ∈ {1 *. . . , n*}. Thus, controlling component-wise error in any orthonormal basis in V controls component-wise error in R
n, in an n-independent but d-dependent manner. In the following, we give a flavor of how we can analyze finite sample estimate errors in V . The maximum norm *|| · ||*∞ is to be understood as *|| · ||*∞,V for some orthonormal basis choice. Here is the result relevant to the linear case described Subsubsection 2.2.1.

Proposition 5. Let Xt i.i.d ∼ N (0, σ2 0 Id), t = 1, . . . , L *and let* Π := ˆ 1 σ 2 0L
PL
t=1 XtXT
t
. *Then, for any* δ ∈ (0, 1),

$$P r\left[||\hat{\Pi}\tilde{x}-\tilde{x}||_{\infty}<C||\tilde{x}||_{2}\operatorname*{max}\left\{\sqrt{\frac{d+\ln(\frac{2}{\delta})}{L}},\frac{d+\ln(\frac{2}{\delta})}{L}\right\}\right]>1-\delta$$

for some C > 0. Proof. We start by bounding the maximum norm of the difference,

$||\hat{\Pi}\bar{x}-\bar{x}||_{\infty}\leq||\hat{\Pi}\bar{x}-\bar{x}||_{2}\leq||\hat{\Pi}-I_{d}||_{\infty}$
$\downarrow$ b. 
where *|| · ||*op is the operator norm.

It can be shown that, for any δ ∈ (0, 1)

$$\operatorname*{Pr}\left[||{\hat{\Pi}}-I_{d}||_{\mathrm{op}}<C\operatorname*{max}\left\{{\sqrt{\frac{d+\ln({\frac{2}{\delta}})}{L}}},{\frac{d+\ln({\frac{2}{\delta}})}{L}}\right\}\right]>1-\delta$$

for some C > 0 (Rigollet & Hutter ¨ , 2023). Combining with the first bound, we get our result.

As to the nonlinear cases, the key result of Theorem 3.1 is the convergence of the numerator 1L
PL
t=1 Xte
⟨Xt,x˜∥⟩/σ2Z
to E[Xe⟨X,x˜∥⟩/σ2Z ] = Re
⟨x,x˜∥⟩/σ2Z *x dp*X(x) and the denominator 1L
PL
t=1 e
⟨Xt,x˜∥⟩/σ2Z to E[e
⟨X,x˜∥⟩/σ2Z ] =
Re
⟨x,x˜∥⟩/σ2Z dpX(x).

In the following, we assume that the support of pX is inside a vector space V whose dimension we denote by d (instead of d + 1, as in the sphere problem). In addition, we refer to the projection of the query on V by x˜ ∈ V , instead of x˜∥. As usual, the maximum norm in V is with respect to some orthonormal basis choice

$\square$
Proposition 6. Let Xt i.i.d ∼ pX and ||Xt||2 ≤ R for t = 1*, . . . , L*.

Then, for any δ ∈ (0, 1),

$$P r\left[\left|\frac{1}{L}\sum_{t=1}^{L}e^{(X_{t}\vec{x})/\sigma_{Z}^{2}}-\mathbb{E}[e^{(X_{t}\vec{x})/\sigma_{Z}^{2}}]\right|<\sinh\left(\frac{R||\vec{x}||_{2}}{\sigma_{Z}^{2}}\right)\sqrt{\frac{2}{L}\ln\left(\frac{2}{\delta}\right)}\right]\geq1-\delta\left(\frac{\delta}{\delta}\right).$$

and

$$P r\Biggl[\left|\left|\frac{1}{L}\sum_{t=1}^{L}X_{t}e^{(X_{t},{\vec{x}})/\sigma_{\vec{x}}^{2}}-\mathbb{E}[X e^{(X,{\vec{x}})/\sigma_{\vec{x}}^{2}}]\right|\right|_{\infty}<R e^{\frac{R|\vec{x}|_{12}}{\sigma_{\vec{x}}^{2}}}\sqrt{\frac{2}{L}\ln\left(\frac{2d}{\delta}\right)}\right]\geq1-\delta.$$

Proof. We provide the sketch of our proof here, the key ingredient of which is the Hoeffding inequality (Hoeffding, 1994).

For the average 1L
PL
t=1 e
⟨Xt,x˜⟩/σ2Z , each term in the sum is bounded above and below by e
±
R||x˜||2 σ2Z . So, the Hoeffding inequality leads to

$$\Pr\left[\left|\frac{1}{L}\sum_{i=1}^{L}e^{(X_{i},\lambda_{i})/\sigma_{\lambda}^{2}}-\mathbb{E}[e^{(X_{i},\lambda_{i})/\sigma_{\lambda}^{2}}]\right|\geq t]\leq2\exp\left[-\frac{2L\epsilon^{2}}{\left(\exp\left(\frac{R(\mathbb{E}[t])_{0}}{\sigma_{\lambda}^{2}}\right)-\exp\left(-\frac{R(\mathbb{E}[t])_{0}}{\sigma_{\lambda}^{2}}\right)\right)^{2}}\right]=2\exp\left[-\frac{L\epsilon^{2}}{2\sinh^{2}\left(\frac{R(\mathbb{E}[t])_{0}}{\sigma_{\lambda}^{2}}\right)}\right].$$
Setting $\delta=2\exp\Big[-\frac{L\epsilon^2}{2\sinh^2\big(\frac{R||\vec{x}||2}{\sigma_Z^2}\big)}\Big]$, we get $\epsilon=\sinh\big(\frac{R||\vec{x}||2}{\sigma_Z^2}\big)\sqrt{\frac{2}{L}\ln\big(\frac{2}{\delta}\big)}$, which gives our ...
, which gives our first probabilistic inequality.
For each component of the vector average 1L
 ergae $\frac{1}{L}\sum_{t=1}^{L}X_t e^{\langle X_t,\hat{x}\rangle/\sigma_Z^2}$, the ... 
⟨Xt,x˜⟩/σ2Z , the terms in the sum are bounded above and below by ±R
R||x˜||2 σ2Z . We use similar arguments involving the Hoeffding inequality, combined with the union bound over all d coordinates

$$\mathrm{Pr}\Bigg{[}\Bigg{|}\Bigg{|}\frac{1}{L}\sum_{t=1}^{L}X_{t}e^{(X_{t},\bar{x})/\sigma_{\mathbb{Z}}^{2}}-\mathbb{E}[X e^{(X,\bar{x})/\sigma_{\mathbb{Z}}^{2}}]\Bigg{|}\Bigg{|}_{\infty}\geq\epsilon]\leq2d\exp\Bigg{[}-\frac{L\epsilon^{2}}{2R^{2}\exp\left(\frac{2R||\bar{x}||_{\mathbb{Z}}}{\sigma_{\mathbb{Z}}^{2}}\right)}\Bigg{]}.$$
.
Once more, setting the RHS to δ and solving for ϵ, we get our second probabilistic inequality.

## F. Limiting Behaviors Of The Softmax Function And Softmax Attention

For small argument A Taylor expansion of the softmax function at zero gives

$$\mathrm{softmax}(\beta v)=\frac{1}{Z}\left(1_{L}+\beta v+O(\beta^{2})\right),$$

where Z =Pi 1 + βvi + O(β 2))= L(1 + βv¯ + O(β 2)) is a normalizing factor, with v¯ =
1 L
Pi vi. The notation 1L
stands for an L-dimensional vector of ones.

Thus, we have Lemma F.1 (Small argument expansion of softmax). As β → 0,

$$\operatorname{softmax}(\beta v)={\frac{1}{L\left(1+\beta v+O(\beta^{2})\right)}}\left(\mathbb{1}_{L}+\beta v+O(\beta^{2})\right)={\frac{1}{L}}\left(\mathbb{1}_{L}+\beta(v-{\bar{v}}\mathbb{1})+O(\beta^{2})\right).$$

F.1. Proof of Proposition 3.2 Proof.

$$F\Big(E,\big(\frac{1}{\epsilon}W_{P V},\epsilon W_{K Q}\big)\Big):=\frac{1}{\epsilon}W_{P V}X_{1:L}\mathrm{softmax}(\epsilon X_{1:L}^{T}W_{K Q}\tilde{X}).$$

18 Using Lemma F.1, as ϵ → 0,

$$F\Big{(}E,(\frac{1}{\epsilon}W_{PV},\epsilon W_{KQ})\Big{)}=\frac{1}{\epsilon}W_{PV}X_{1:L}\Bigg{[}\frac{1}{L}\left(1_{L}+\epsilon(X_{L:L}^{T}W_{KQ}\tilde{X}-(\frac{1}{L}\sum_{t}X_{t}^{T}W_{KQ}\tilde{X})1_{L})+O(\epsilon^{2})\right)\Bigg{]}$$ $$=\frac{1}{\epsilon}W_{PV}\tilde{X}+\frac{1}{L}W_{PV}\sum_{t=1}^{L}X_{t}(X_{t}-\tilde{X})^{T}W_{KQ}\tilde{X}+O(\epsilon),\tag{4}$$

where X¯ =
1 L
PL
t=1 Xt is the empirical mean and the notation 1L emphasizes that it is a column vector of ones with dimension L. For large argument As β → ∞, the softmax function simply selects the maximum over its inputs (as long as the the maximum is unique):

$$(\mathbf{A}.11)$$
$\square$
$$\operatorname{softmax}(\beta v)\approx{\begin{cases}1&{\mathrm{if}}\ i=\arg\operatorname*{max}_{j}v_{j},\\ 0&{\mathrm{otherwise}}.\end{cases}}$$

In this case, all attention weight is given to a single element, and the others are effectively ignored.

G. MSE Loss landscape for scaled identity weights

MSE loss landscape for Fig. 3 (Case 2) MSE loss landscape for Fig. 3 (Case 3) (a) (b)
heuristic_KQ heuristic_PV trained model heuristic (theory) 2D scan min
The loss landscapes in Fig. 6 exhibit large, low-cost valleys with a roughly hyperbolic structure that is especially apparent in Case 2. This indicates a multiplicative tradeoff in the scales of WKQ and WP V , which suggests that linear attention might be applicable here as well. For completeness, Figure 7 shows linear attention performance for both cases, demonstrating that it performs quite similarly to softmax for sub-sphere denoising, but less well in the Gaussian mixtures case.

## H. Structured Optimal Weights Under Prompt Transformation

We find that one-layer transformers can learn to undo arbitrary invertible coordinate transformations that warp the denoising tasks. Focusing on the subspace denoising case, suppose each prompt is transformed by a fixed invertible square matrix

(a)
Case 2: Nonlinear manifolds Case 3: Gaussian mixtures
= 0.02 0.1, d = 8, R =
0.08 0.06 x = 0 softmax (train/test)
linear (train/test) 
0.05 0.06 MSE/
x = fopt ( x )
MSE
0.04 0.04 0.03 0.02 50 100 150 200 100 50 150 Epoch Epoch
(b)
Final weights: linear Final weights: softmax Final weights:  Iinear Final weights: softmax W
Wp Wx Wp W
Wp WKO
Wp 2.196 2.105 3.596 1.603 1.748 1.745 5.127 0.998 e 0.0 2.0 3.3 0.0 3.0 1.5 0.0 1. S
4.0 0.0 4.0
A , i.e. E = (X 1:L, X ) → E ′ = (AX 1:L, A X ). If the target remains X L +1 in the untransformed space, then the optimal attention weights are no longer diagonal, but instead take a structured form determined by the transformation matrix:

$$(\mathbf{A}.12)^{\frac{1}{2}}$$

(C)
Linear attention
(d)
Softmax attention
(b)

$$W_{P V}=\alpha A^{-1},\quad W_{K Q}=\beta(A A^{T})^{-1},$$

Theory Wxo = B(AAT)
Learned W XQ
Learned Wro 0.00.00 β ≈ 0.421 β ≈ 0.193 0.50
()
0.25 A (transform)
0.00 1.00 9 0.75
-0.25 0.50 12 12 12
—0.50 0.25
-0.75 15 15 15 0.00 12 12 00 Theory Wpv = aA-1 Learned W P .

Learned Wy 12
−0.50
≈ 0.792 a ≈ 1.61
-0.75 15 0.5
−1.00 12 0.0 s 12 12
-0.5 12 15
Notably, we find that both the linear and softmax attention layers are able to learn these structures; see Fig. 8 for an example.

where αβ =
o i + 2 as before.