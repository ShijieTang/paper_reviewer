# In-Context Denoising with One-Layer Transformers: Connections between Attention and Associative Memory Retrieval

Matthew Smart <sup>1</sup> Alberto Bietti <sup>2</sup> Anirvan M. Sengupta 2 3 4

## Abstract

We introduce in-context denoising, a task that refines the connection between attention-based architectures and dense associative memory (DAM) networks, also known as modern Hopfield networks. Using a Bayesian framework, we show theoretically and empirically that certain restricted denoising problems can be solved optimally even by a single-layer transformer. We demonstrate that a trained attention layer processes each denoising prompt by performing a single gradient descent update on a context-aware DAM energy landscape, where context tokens serve as associative memories and the query token acts as an initial state. This one-step update yields better solutions than exact retrieval of either a context token or a spurious local minimum, providing a concrete example of DAM networks extending beyond the standard retrieval paradigm. Overall, this work solidifies the link between associative memory and attention mechanisms first identified by Ramsauer et al., and demonstrates the relevance of associative memory models in the study of in-context learning.

# 1. Introduction

The transformer architecture [\(Vaswani et al.,](#page-10-0) [2017\)](#page-10-0) has achieved remarkable success across diverse domains, from natural language processing [\(Devlin et al.,](#page-8-0) [2019;](#page-8-0) [Brown](#page-8-1) [et al.,](#page-8-1) [2020;](#page-8-1) [Touvron et al.,](#page-10-1) [2023\)](#page-10-1) to computer vision [\(Doso](#page-8-2)[vitskiy et al.,](#page-8-2) [2021\)](#page-8-2). Despite their practical success, understanding the mechanisms behind transformer-based networks remains an open challenge. This challenge is exacerbated by the growing scale and complexity of modern large networks. Toward addressing this, researchers studying simplified architectures have identified connections between the attention operation that is central to transformers and associative memory models [\(Ramsauer et al.,](#page-9-0) [2021\)](#page-9-0), providing not only an avenue for understanding how such architectures encode and retrieve information but also potentially ways to improve them further.

The most celebrated model for associative memories in systems neuroscience is the so-called Hopfield model [\(Amari,](#page-8-3) [1972;](#page-8-3) [Nakano,](#page-9-1) [1972;](#page-9-1) [Little,](#page-9-2) [1974;](#page-9-2) [Hopfield,](#page-9-3) [1982\)](#page-9-3). This model has a capacity to store "memories" (stable fixed points of a recurrent update rule) proportional to the number of nodes [\(Hopfield,](#page-9-3) [1982;](#page-9-3) [Amit et al.,](#page-8-4) [1985\)](#page-8-4). In the last decade, new energy functions [\(Krotov & Hopfield,](#page-9-4) [2016;](#page-9-4) [Demircigil et al.,](#page-8-5) [2017\)](#page-8-5) were proposed for dense associative memories with much higher capacities. These energy functions are often referred to as modern Hopfield models. [Ram](#page-9-0)[sauer et al.](#page-9-0) [\(2021\)](#page-9-0) pointed out the similarity between the one-step update rule of a certain modern Hopfield network [\(Demircigil et al.,](#page-8-5) [2017\)](#page-8-5) and the softmax attention layer of transformers, generating interest in the statistical physics and systems neuroscience communities [\(Krotov & Hopfield,](#page-9-5) [2021;](#page-9-5) [Krotov,](#page-9-6) [2023;](#page-9-6) [Lucibello & Mezard](#page-9-7) ´ , [2024;](#page-9-7) [Millidge](#page-9-8) [et al.,](#page-9-8) [2022\)](#page-9-8). Recent work has extended this concept to improve retrieval by incorporating sparsity [\(Hu et al.,](#page-9-9) [2023;](#page-9-9) [Wu et al.,](#page-10-2) [2024b;](#page-10-2) [Santos et al.,](#page-9-10) [2024;](#page-9-10) [Wu et al.,](#page-10-3) [2024a\)](#page-10-3), while others have leveraged associative memory principles to design new energy-based transformer architectures [\(Hoover](#page-9-11) [et al.,](#page-9-11) [2023\)](#page-9-11). However, these extensions and the foundational construction in [Ramsauer et al.](#page-9-0) [\(2021\)](#page-9-0) primarily focus on the specific task of exact retrieval (converging to a fixed point), while in practice transformers may tackle many other tasks.

To explore this connection beyond retrieval, we introduce *in-context denoising*, a task that bridges the behavior of trained transformers and associative memory networks through the lens of in-context learning (ICL). In standard ICL, a sequence model is trained to infer an unknown function g from contextual examples, predicting

<sup>1</sup> Center for Computational Biology, Flatiron Institute, New York, NY, USA <sup>2</sup> Center for Computational Mathematics, Flatiron Institute, New York, NY, USA <sup>3</sup> Center for Computational Quantum Physics, Flatiron Institute, New York, NY, USA <sup>4</sup> Department of Physics and Astronomy, Rutgers University, Piscataway, NJ, USA. Correspondence to: Matthew Smart <msmart@flatironinstitute.org>, Anirvan M. Sengupta <anirvans.physics@gmail.com>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

g(XL+1) given a sequence of input-output pairs E = ((X1, g(X1)), ...,(XL, g(XL)),(XL+1, −)). Crucially, g is implied solely through the context and differs across prompts – performant models are therefore said to "learn g(x) in context". While ICL has been extensively studied in supervised settings [\(Garg et al.,](#page-8-6) [2022;](#page-8-6) [Zhang et al.,](#page-10-4) [2024;](#page-10-4) [Akyurek et al.](#page-8-7) ¨ , [2023;](#page-8-7) [Reddy,](#page-9-12) [2024\)](#page-9-12), recent work suggests that transformers may internally emulate gradient descent over a context-specific loss function during inference [\(Von Oswald et al.,](#page-10-5) [2023;](#page-10-5) [Dai et al.,](#page-8-8) [2023;](#page-8-8) [Ahn et al.,](#page-8-9) [2023\)](#page-8-9). This general perspective aligns with our findings.

In this work, we generalize ICL to an unsupervised setting where the prompt consists of L samples from a random distribution and the query is a noise-corrupted sample from the same distribution. This shift allows us to probe how trained transformers internally approximate Bayes optimal inference, while deepening the connection to associative memory models which are prototypical denoisers. By setting up this problem in this way, we also attempt to answer a few questions. One concerns the memorization-generalization dilemma in denoising: a Hopfield model's success is usually measured by successful memory recovery, while in-context learning may have to solve a completely new problem. Another question has to do with the number of iterations of the corresponding Hopfield model: why does the [Ramsauer](#page-9-0) [et al.](#page-9-0) [\(2021\)](#page-9-0) correspondence involve only one iteration of Hopfield energy minimization and not many?

In summary, our contributions are as follows: In Section [2,](#page-1-0) we introduce in-context denoising as a framework for understanding how transformers perform implicit inference beyond memory retrieval. In Section [3,](#page-3-0) we establish that single-layer transformers with one attention head are expressive enough to optimally solve certain denoising problems. We then empirically demonstrate that standard training from random weights can recover the Bayes optimal predictors. The trained attention layers are mapped back to dense associative memory networks in Section [4.](#page-6-0) Our results refine the general connection pointed out in previous work, offer new mechanistic insights into attention, and provide a concrete example of dense associative memory networks extending beyond the standard memory retrieval paradigm to solve a novel in-context learning task.

## 2. Problem formulation: In-context denoising

In this section, we describe our general setup. Recurring common notation is described in Appendix [A.1.](#page-11-0)

## 2.1. Setup

Each task corresponds to a distribution D over the probability distribution of data: p<sup>X</sup> ∼ D. Let X1, · · · , XL+1 iid∼ pX, define the sampling of the tokens. Let the noise corruption be defined by X˜ ∼ pnoise(·|XL+1). The random sequence E = (X1, X2, ..., XL, X˜) are given as "context" (input) to a sequence model F(·; θ) which outputs an estimate XˆL+1 of the original (L + 1)-th token . The task is to minimize the expected loss <sup>E</sup>[l(XˆL+1, XL+1)] for some loss function l(·, ·). Namely, our problem is to find

$$\min_{\theta} \mathbb{E}_{p_X \sim D, X_{1:L+1} \sim p_X^{L+1}, \tilde{X} \sim p_{\text{noise}}(\cdot | X_{L+1})} [l(F(E, \theta), X_{L+1})]. \quad (1)$$

In practice, we choose X˜ = XL+1 + Z, a pure token corrupted by the addition of isotropic Gaussian noise Z ∼ N (0, σ<sup>2</sup> Z In), and our objective function to minimize is the mean squared error (MSE) <sup>E</sup>[||XˆL+1 − XL+1||<sup>2</sup> ].

In the following subsection, we explain the pure token distributions for three specific tasks. These tasks are of course structured so that a one-layer transformer has the expressivity to capture a solution, which, as L → ∞, provides an optimal solution, in some sense. To that end, we derive Bayes optimal estimators for each of the three tasks, under the assumption that we know the original distribution p<sup>X</sup> of pure tokens. In Section [3,](#page-3-0) we use these estimators as baselines to evaluate the performance of the denoiser f(E, θ) based on a one-layer transformer trained on finite datasets.

## 2.2. Task-specific token distributions

We consider three elementary in-context denoising tasks, where the data (vectors in R <sup>n</sup>) comes from:

- 1. Linear manifolds (d-dimensional subspaces)
- 2. Nonlinear manifolds (d-spheres)
- 3. Small noise Gaussian mixtures (clusters) where the component means have fixed norm

Below we describe the task-specific distributions p<sup>X</sup> and the process for sampling tokens {xt}. The same corruption process applies to all cases: X˜ = XL+1+Z, Z ∼ N (0, σ<sup>2</sup> Z In).

## 2.2.1. CASE 1 - LINEAR MANIFOLDS

A given training prompt consists of pure tokens sampled from a random d-dimensional subspace S of R n.

- Let P be the orthogonal projection operator to a random d-dim subspace S of R <sup>n</sup>, sampled according to the uniform measure, induced by the Haar measure on the coset space O(n)/O(n − d) × O(d), on the Grassmanian G(d, n), the manifold of all d-dimensional subspaces of R
  - n.
- Let Y ∼ N (0, σ<sup>2</sup> 0 In) and define X = P Y ; we use this procedure to construct the starting sequences (X1, ..., XL+1) of L + 1 independent tokens.

![](_page_2_Figure_2.jpeg)

Figure 1. (a) Problem formulation for a general in-context denoising task. (b) The three denoising tasks considered here include instances of linear and non-linear manifolds as well as Gaussian mixtures. In each case, the task embedding E (i) consists of a sequence of pure tokens from the data distribution p (i) <sup>X</sup> ∼ D where D denotes the task distribution, along with a single query token that has been corrupted by Gaussian noise. The objective is to predict the target (i.e. *denoise* the query) given information contained only in the prompt.

We thus have p<sup>X</sup> = N (0, σ<sup>2</sup> <sup>0</sup>P), with the Haar distribution of P characterizing the task ensemble associated with D.

## 2.2.2. CASE 2 - NONLINEAR MANIFOLDS

We focus on the case of d-dimensional spheres of fixed radius R centered at the origin in R n.

- Choose a random d+1-dimensional subspace V of R n, sampled according to the uniform measure, as before, on the Grassmanian G(d + 1, n). The choice of this random subspace generates the distribution of tasks D.
- Inside V , sample uniformly from the radius R sphere (once more, a Haar induced measure on a coset space O(d + 1)/O(d)). We use this procedure to construct input sequences X1:L+1 = (x1, ..., xL+1) of L + 1 independent tokens.

In practice, we uniformly sample points with fixed norm in R d and embed them in R <sup>n</sup> by concatenating zeros. We then rotate the points by selecting a random orthogonal matrix Q ∈ R <sup>n</sup>×<sup>n</sup>.

## 2.2.3. CASE 3 - GAUSSIAN MIXTURES (CLUSTERING)

Pure tokens are sampled from a weighted mixture of isotropic Gaussians in n-dimensions, {wa,(µa, σ<sup>2</sup> a )} K <sup>a</sup>=1. The density is

$$p_X(x) = \sum_{a=1}^K w_a C_a e^{-\|x - \mu_a\|^2/2\sigma_a^2},$$

where C<sup>a</sup> = (2πσ<sup>2</sup> a ) −n/2 are normalizing constants. The µ<sup>a</sup> are independently chosen from a uniform distribution on the radius R sphere of dimension n − 1, centered around zero. The distribution of tasks D, is decided by the choice of {µa} K <sup>a</sup>=1.

For our ideal case, we will consider the limit that the variances go to zero. In that case, the density is simply

$$p_{X_0}(x) = \sum_{a=1}^K w_a \delta(x - \mu_a).$$

## 2.3. Bayes optimal denoising baselines for each case

The first L tokens in E are "pure samples" from p that should provide information about the distribution for our denoising task. Our performance is expected to be no better than that of the best method, in the case that the token distribution and also the corrupting process are exactly known. This is where the Bayesian optimal baseline comes in. As is well-known, the Bayes optimal predictor of a quantity is given by the posterior mean. We use that fact to compute the Bayes optimal loss.

In particular, we seek a function f : R <sup>n</sup> → <sup>R</sup> <sup>n</sup> such that <sup>E</sup>X,X˜ h ∥X − f(X˜)∥ 2 i is minimized. Since the perturbation Z is Gaussian, the posterior distribution of X, given X˜ is

$$p_{X|\tilde{X}}(x \mid \tilde{x}) = C(\tilde{x})p_X(x)e^{-\|x-\tilde{x}\|^2/2\sigma_Z^2},$$

where C(˜x) is a normalizing factor (see Appendix [A.2](#page-11-1) for more explanation). The following proposition sets up a baseline to which we expect to compare our results as L → ∞. The proof is in Appendix [B.1.](#page-12-0)

Proposition 1. *For each task, specified by the input distribution* pX*, and the noise model* pX˜|X*,*

$$\mathbb{E}_{X,\tilde{X}} \left[ \|X - f(\tilde{X})\|^2 \right] \geq \mathbb{E}_{\tilde{X}} \left[ \text{Tr Cov}(X \mid \tilde{X}) \right]. \quad (2)$$

*This lower bound is met when* f(X˜) = <sup>E</sup>[X | X˜]*.*

Thus, the Bayes optimal denoiser is the posterior expectation for X given X˜. The expected loss is found by computing the posterior sum of variances.

These optimal denoisers can be computed analytically for both the linear and nonlinear manifold cases (given the variances and dimensionalities). In the Gaussian mixture (clustering) case, it depends on the choice of the centroids which then needs to be averaged over.

Linear case. For the linear denoising task, pure samples X are drawn from an isotropic Gaussian in a restricted subspace. The following result provides the Bayes optimal predictor in this case, the proof of which is in Appendix [C.1.](#page-12-1)

Proposition 2. *For* p<sup>X</sup> *corresponding to Subsection [2.2.1,](#page-1-1) the Bayes optimal answer is*

$$f_{opt}(\tilde{X}) = \mathbb{E}[X|\tilde{X}] = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} P\tilde{X}, \quad (3)$$

*and the expected loss is*

$$\mathbb{E} \left[ \|P\tilde{X} - X_{L+1}\|^2 \right] = d\sigma_0^2 \sigma_Z^2 / (\sigma_0^2 + \sigma_Z^2). \quad (4)$$

![](_page_3_Figure_17.jpeg)

Figure 2. Baseline estimators for the case of random linear manifolds with projection operator P (i) .

Manifold case. In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres S (e.g. the circle S <sup>1</sup> ⊂ <sup>R</sup> 2 ). For such manifolds, the Bayes optimal answer is given by the following proposition.

Proposition 3. *For* p<sup>X</sup> *defined as in Subsection [2.2.2,](#page-2-0) with* P *being the orthogonal projection operator to* V *, the* d + 1 *dimensional linear subspace, with* R *being the radius of sphere* S*, the Bayes optimal answer is*

$$\begin{aligned} f_{opt}(\tilde{X}) &= \mathbb{E}[X \mid \tilde{X}] \\ &= \frac{\int e^{\langle x, \tilde{X}_{\parallel} \rangle} \sigma_z^2 x dS_x}{\int e^{\langle x, \tilde{X}_{\parallel} \rangle} \sigma_z^2 dS_x} \end{aligned} \quad (5)$$

$$= \frac{I_{\frac{d+1}{2}} \left( R \frac{|\tilde{X}_{\parallel}||}{\sigma_Z^2} \right)}{I_{\frac{d-1}{2}} \left( R \frac{|\tilde{X}_{\parallel}||}{\sigma_Z^2} \right)} R \frac{\tilde{X}_{\parallel}}{||\tilde{X}_{\parallel}||}, \quad (6)$$

*where* X˜<sup>∥</sup> = P X˜ *and* I<sup>ν</sup> *is the modified Bessel function of the first kind.*

Clustering case. For clustering with isotropic Gaussian mixtures {wa,(µa, σ<sup>2</sup> a )} p <sup>a</sup>=1, the Bayes optimal predictors for some important special cases are as follows. See Appendix [C.3](#page-13-0) for the general case.

Proposition 4. *For general isotropic Gaussian model with* σ<sup>a</sup> = σ0, ||µa|| = R *for all* a = 1, . . . , K*.*

$$\begin{aligned} f_{opt}(\tilde{X}) &= \mathbb{E}[X|\tilde{X}] \\ &= \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} \tilde{X} + \frac{\sigma_Z^2}{\sigma_0^2 + \sigma_Z^2} \frac{\sum_a w_a e^{\langle \mu_a, \tilde{X} \rangle / (\sigma_0^2 + \sigma_Z^2)}}{\sum_a w_a e^{\langle \mu_a, \tilde{X} \rangle / (\sigma_0^2 + \sigma_Z^2)}} \mu_a. \end{aligned} \quad (7)$$

*If* σ<sup>0</sup> → 0*,*

$$f_{opt}(\tilde{X}) = \mathbb{E}[X \mid \tilde{X}] = \frac{\sum_a w_a e^{\langle \mu_a, \tilde{X} \rangle / \sigma_Z^2} \mu_a}{\sum_a w_a e^{\langle \mu_a, \tilde{X} \rangle / \sigma_Z^2}}. \quad (8)$$

In all three cases, we notice similarities between the form of the Bayes optimal predictor, and attention operations in transformers, a connection which we explore below.

## 3. In-context denoising with one-layer transformers – Empirical results

In this section, we provide simple constructions of one-layer transformers that approximate (and under certain conditions, exactly match) the Bayes optimal predictors above.

Input: Let p (1) <sup>X</sup> , . . . , p (N) X iid∼ <sup>D</sup>, be distributions sampled for one of the tasks. For each distribution p (i) <sup>X</sup> , we sample E(i) := (X (i) 1 , . . . , X(i) L , X˜(i) ) taking value in <sup>R</sup> n×(L+1) be an input to a sequence model. We also retain the true (L + 1)-th token X (i) <sup>L</sup>+1 for each i.

Objective: Given an input sequence E(i) , return the uncorrupted final token X (i) L+1. We consider the meansquared error loss over a collection of N training pairs, {E(i) , X(i) <sup>L</sup>+1} N <sup>i</sup>=1,

$$C(\theta) = \sum_{i=1}^N \|F(E^{(i)}, \theta) - x_{L+1}^{(i)}\|^2, \quad (9)$$

where F(E(i) , θ) denotes the parametrized function predicting the target final token based on input sequence E(i) .

### 3.1. One-layer transformer and the attention between the query and pure tokens

There we have fopt(X˜) = <sup>σ</sup> 0 σ <sup>0</sup>+σ 2 Z P X˜. Note that, by the strong law of large numbers, Pˆ = 1 σ <sup>0</sup>L P<sup>L</sup> <sup>t</sup>=1 <sup>X</sup>tX<sup>T</sup> t is a random matrix that almost surely converges component-bycomponent to the orthogonal projection P as L → ∞, since, for each t, XtX<sup>T</sup> <sup>t</sup> has the expectation σ 2 <sup>0</sup>P and that X<sup>t</sup> is a Gaussian random variable with zero mean and a finite covariance matrix. So we could propose

$$f(\tilde{X}) = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} \hat{P}\tilde{X} = \frac{1}{(\sigma_0^2 + \sigma_Z^2)L} \sum_{t=1}^L X_t \langle X_t, \tilde{X} \rangle. \quad (10)$$

We now consider a simplified one-layer linear transformer (see Appendices [D.1](#page-14-0) and [D.2](#page-14-1) for more detailed discussions) which still has sufficient expressive power to capture our finite sample approximation to the Bayes optimal answer. We define

$$\hat{X} = F_{\text{Lin}}(E, \theta) := \frac{1}{L} W_{PV} X_{1:L} X_{1:L}^T W_{KQ} \tilde{X} \quad (11)$$

taking values in R <sup>n</sup>, where X1:<sup>L</sup> := [X1, . . . , XL] taking values in R <sup>n</sup>×<sup>L</sup>, with learnable weights WKQ, WP V ∈ R <sup>n</sup>×<sup>n</sup> abbreviated by θ. Note that, when WP V = αIn, WKQ = βIn, and αβ = 1 σ 2 <sup>0</sup>+σ 2 Z , F(E, θ) should approximate the Bayes optimal answer fopt(X˜) as L → ∞. For a detailed discussion of the convergence rate, see Appendix [E,](#page-15-0) in general, and Proposition [5,](#page-16-0) in particular.

Similarly, we could argue that the second two problems, the d-dimesional spheres and the σ<sup>0</sup> → 0 zero limit of the Gaussian mixtures could be addressed by softmax attention

$$\hat{X} = F(E, \theta) := W_{PV} X_{1:L} \text{softmax}(X_{1:L}^T W_{KQ} \tilde{X}) \quad (12)$$

taking values in R <sup>n</sup>. The function softmax(z) := P 1 n <sup>i</sup>=1 e zi (e z<sup>1</sup> , . . . , e<sup>z</sup><sup>n</sup> ) <sup>T</sup> ∈ <sup>R</sup> <sup>n</sup> is applied column-wise.

For both problems, namely the spheres and the σ<sup>0</sup> → 0 Gaussian mixtures, we could have WP V = αIn, WKQ = βI<sup>n</sup> with α = 1, β = 1/σ<sup>2</sup> Z providing Bayes optimal answers as L → ∞.

In fact, we could make a more general statement about distributions p<sup>X</sup> where the norm of X is fixed.

Theorem 3.1. *If we have a task distribution* D *so that the support of each* p<sup>X</sup> *is the subset of some sphere, centered around the origin, with a* pX*-dependent radius* R*, then the function*

$$F((\{X_t\}_{t=1}^L, \tilde{x}), \theta^*) = \frac{\sum_{t=1}^L X_t e^{\langle X_t, \tilde{x} \rangle} / \sigma_Z^2}{\sum_{t=1}^L e^{\langle X_t, \tilde{x} \rangle} / \sigma_Z^2} \quad (13)$$

*converges almost surely to the Bayes optimal answer* f*opt*(˜x) *for all* x˜ ∈ R <sup>n</sup>*, as* L → ∞*. The optimal parameter* θ ∗ *refers to* WP V = In, WKQ = 1 σ In*.*

The proof of the theorem is in Appendix [D.3.](#page-15-1) See Appendix [E,](#page-15-0) particularly Proposition [6,](#page-16-1) for consideration of convergence rates. Note that the condition of p<sup>X</sup> being supported on a sphere is not artificial as, in many practical transformers, pre-norm with RMSNorm gives you inputs on the sphere, up to learned diagonal multipliers.

Note that the natural form of attention that is suggested by our formulation of in-context denoising would involve Gaussian kernels:

$$\hat{X} = F_G(E, \theta) := \frac{\sum_t W_{PV} X_t e^{-\frac{1}{2} \|W_K X_t - W_Q \tilde{X}\|^2}}{\sum_t e^{-\frac{1}{2} \|W_K X_t - W_Q \tilde{X}\|^2}}. \quad (14)$$

The relation between softmax attention and the Gaussian kernel has been noted in [\(Choromanski et al.,](#page-8-10) [2021;](#page-8-10) [Am](#page-8-11)[brogioni,](#page-8-11) [2024\)](#page-8-11) and a Gaussian kernel-based attention is implemented in [\(Chen et al.,](#page-8-12) [2021\)](#page-8-12). A related Hopfield energy, with WK, WQ, and WP V proportional to identity matrices, is proposed in [\(Hoover et al.,](#page-9-13) [2024a\)](#page-9-13).

For the linear case, we use linear attention, but that may not be essential. Informally speaking, the softmax attention model has the capacity to subsume the linear attention model.

Proposition 3.2. *As* ϵ → 0*,*

$$\begin{aligned} F\left(E, \left(\frac{1}{\epsilon}W_{PV}, \epsilon W_{KQ}\right)\right) &= \frac{1}{\epsilon}W_{PV}\bar{X} \\ &+ \frac{1}{L}W_{PV} \sum_{t=1}^L X_t(X_t - \bar{X})^T W_{KQ} \tilde{X} + O(\epsilon), \end{aligned} \quad (15)$$

*where* X¯ = 1 L P<sup>L</sup> <sup>t</sup>=1 X<sup>t</sup> *is the empirical mean.*

See Appendix [F](#page-17-0) for the details of small WKQ expansion and Appendix [F.1](#page-17-1) for the proof of Proposition [3.2.](#page-4-0)

For case 1, note that <sup>E</sup>[Xt] = 0 and covariance of X<sup>t</sup> is finite, E[X¯] = 0, and E[||X¯||<sup>2</sup> ] = O( 1 L ), allowing us to drop X¯ as L → ∞. If, in addition, ϵ is small, only the second term survives. Thus, F E,( 1 <sup>ϵ</sup>WP V , ϵWKQ) starts to approximate FLin E,(WP V , WKQ) when L is large and ϵ is small, with ϵ √ L large. We therefore could use the softmax model for all three cases.

#### 3.2. Case 1 – Linear manifolds

The Bayes optimal predictor for the linear denoising task from Section [2.3](#page-2-1) suggests that the linear attention weights should be scaled identity matrices with their product satisfying αβ = σ <sup>0</sup>+σ Z . Fig. [3](#page-5-0) shows that a one-layer network of size n = 16 trained on tasks with σ 2 <sup>Z</sup> = 1, σ<sup>2</sup> <sup>0</sup> = 2, d = 8, L = 500 indeed achieves this bound, training to nearly diagonal weights with the appropriate scale ⟨w (ii) KQ⟩⟨w (ii) P V ⟩ = 0.327 ≈ 1/3 (similar weights are learned for each seed, up to a sign flip).

![](_page_5_Figure_1.jpeg)

Figure 3. (a) Training dynamics for the studied cases using one-layer softmax attention (circles) as well as linear attention (triangles). Solid lines represent the average loss over six seeds, with the shaded area indicating the range for cases 2 and 3. For each case, the grey dashed baseline indicates the 0-predictor, and the pink line indicates the Bayes optimal predictor. All cases use a context length of L = 500, ambient dimension n = 16, and are trained with Adam on a dataset of size 800 with batch size 80 and standard weight initialization wij ∼ U[−1/ √ n, 1/ √ n]. (b) Final attention weights WKQ and WP V are shown. For each, we indicate the mean of the diagonal elements. Representative initial weights are displayed for the second and third case.

Fig. [4\(](#page-6-1)a) displays how this bound is approached as the context length L of training samples is increased. In Fig. [4\(](#page-6-1)b) we study how the performance of a model trained to denoise random subspaces of dimension d = 8 is affected by shifts in the subspace dimension at inference time. We find that when provided sufficient context, such models can adapt with mild performance loss to solve more challenging tasks not present in the training set.

It is evident from Fig. [3\(](#page-5-0)a) that the softmax network performs similarly to the linear one for this task. We can understand this through the small argument expansion of the softmax function mentioned above. The learned weights displayed in Fig. [3\(](#page-5-0)b) indicate that β softmax ≈ 0.194 becomes small (note it decreases by a factor ϵ ≈ 0.344 relative to β linear), while the value scale α softmax ≈ 1.607 becomes larger by a similar factor ∼ 1/ϵ to compensate. Thus, although the optimal denoiser for this case is intuitively expressed through linear self-attention, it can also be achieved with softmax self-attention in the appropriate limit.

Moreover, we find that when the entire prompt undergoes a global invertible transformation A ̸= I, the optimal attention weights are no longer scaled identity matrices but acquire a structured form determined by A. Both linear and softmax attention layers are able to recover this structure through training; see Appendix [H](#page-18-0) for details and empirical verification.

#### 3.3. Case 2 – Nonlinear manifolds

Fig. [3](#page-5-0) (case 2) shows networks of size n = 16 trained to denoise subspheres of dimension d = 8 and radius R = 1, with corruption σ 2 <sup>Z</sup> = 0.1 and context length L = 500. Once again, the network trains to have scaled identity weights.

We note that although the network nearly achieves the optimal MSE on the test set, the weights appear at first glance to deviate slightly from the Bayes optimal predictor of Section [2.3,](#page-2-1) which indicated WP V = αI, WKQ = βI with α = 1, β = 1/σ<sup>2</sup> Z . To better understand this, we consider a coarse-grained MSE loss landscape by scanning over α and β. See Fig. [6\(](#page-18-1)a) in Appendix [G.](#page-18-2) We find that the 2D loss landscape has roughly hyperbolic level sets which is suggestive of the linear attention limit, where the weight scales become constrained by their product αβ. Reflecting the symmetry of the problem, we also note mirrored negative solutions (i.e. one could also identify α = −1, β = −1/σ<sup>2</sup> Z from the analysis in Section [2.3\)](#page-2-1). Importantly, the plot shows that the trained network lies in the same valley of the loss landscape as the optimal predictor, in agreement with Fig. [3.](#page-5-0) Moreover, the shape of the loss landscape suggested that linear attention might also be applicable to this case, which we demonstrate and discuss further in Appendix [G.](#page-18-2)

## 3.4. Case 3 – Gaussian mixtures

Figure [3](#page-5-0) (case 3) shows networks of size n = 16 trained to denoise balanced Gaussian mixtures with p = 8 compo-

![](_page_6_Figure_1.jpeg)

Figure 4. (a) Trained linear attention network converges to Bayes optimal estimator as context length increases (n = 16, d = 8, σ <sup>0</sup> = 2, σ<sup>2</sup> <sup>z</sup> = 1). (b) A network trained to denoise subspaces of dimension d = 8 can accurately denoise subspaces of different dimensions presented at inference time, given sufficient context.

nents that have isotropic variance σ 2 <sup>0</sup> = 0.02 and centers randomly placed on the unit sphere in R <sup>n</sup>. The corruption magnitude is σ <sup>Z</sup> = 0.1 and context length is L = 500. The baselines show the zero predictor (dashed grey line) as well as the optimum from Proposition [\(4\)](#page-3-1) (pink) and its σ 2 <sup>0</sup> → 0 approximation Eq. [\(8\)](#page-3-2) (grey).

The trained weights qualitatively approach the optimal estimator for the zero-variance limit but with a slightly different scaling: while the scale of WP V is α ≈ 1, the WKQ scale is β ≈ 5.127 < 1/σ<sup>2</sup> Z . To study this, we provide a corresponding plot of the 2D loss landscape in Fig. [6\(](#page-18-1)a) in Appendix [G.](#page-18-2) While the symmetry of the previous case has been broken (the context cluster centers {µa} will not satisfy ⟨µ⟩ = 0), we again find that the trained network lies in the anticipated global valley of the MSE loss landscape.

## 4. Connection to dense associative memory networks

In each of the denoising problems studied above, we have shown analytically and empirically that the optimal weights of the one-layer transformer are scaled identity matrices WP V ≈ αI, WKQ ≈ βI. In the softmax case, the trained denoiser can be concisely expressed as

$$\hat{x} = g(X_{1:L}, \tilde{x}) := \alpha X_{1:L} \text{softmax}(\beta X_{1:L}^T \tilde{x}),$$

re-written such that X ∈ R <sup>n</sup>×<sup>L</sup> stores pure context tokens.

We now demonstrate that such denoising corresponds to one-step gradient descent (with specific step sizes) of energy models related to dense associative memory networks, also known as modern Hopfield networks [\(Ramsauer et al.,](#page-9-0) [2021;](#page-9-0) [Demircigil et al.,](#page-8-5) [2017;](#page-8-5) [Krotov & Hopfield,](#page-9-4) [2016\)](#page-9-4).

Consider the energy function:

$$\mathcal{E}(X_{1:L}, s) = \frac{1}{2\alpha} \|s\|^2 - \frac{1}{\beta} \log \left( \sum_{t=1}^L e^{\beta X_t^T s} \right), \quad (16)$$

which mirrors the [Ramsauer et al.](#page-9-0) [\(2021\)](#page-9-0) construction but with a Lagrange multiplier added to the first term. Figure [5](#page-6-2) illustrates this energy landscape for the spherical manifold case.

![](_page_6_Figure_8.jpeg)

Figure 5. Gradient descent denoising for the nonlinear manifold case (spheres) in n = 2 with d = 1. A context-aware dense associative memory network E(X1:L, s) is constructed whose gradient corresponds to the Bayes optimal update (trained attention layer). Note that the density of sampled context tokens sculpts the valleys of the energy landscape. Left: the attention step of a one-layer transformer trained on the denoising task corresponds to a single gradient descent step. Right: Iterating the denoising process—as is conventional for Hopfield networks—can potentially degrade the estimate by causing it to become query-independent (e.g. converging to a distant minimum). Here R = 1, σ<sup>2</sup> <sup>Z</sup> = 10, L = 20 and α = 1, β = 1/σ<sup>2</sup> Z .

An operation inherent to the associative memory perspective is the recurrent application of a denoising update. Gradient descent iteration s(t + 1) = s(t) − γ ∇sE X1:L, s(t) yields

$$s(t+1) = \left(1 - \frac{\gamma}{\alpha}\right) s(t) + \gamma X_{1:L} \text{softmax}(\beta X_{1:L}^T s(t)). \quad (17)$$

It is now clear that initializing the state to the query s(0) = ˜x and taking a single step with size γ = α recovers the behavior of the trained attention model (Fig. [5\)](#page-6-2). The attention mechanism here is thus mechanistically interpretable: the context tokens X1:<sup>L</sup> induce a context-dependent associative memory landscape, while the query acts as an initial condition for inference-time gradient descent. One could naturally consider alternative step sizes and recurrent iteration. However, Fig. [5](#page-6-2) demonstrates that naive iteration of Eq. [\(17\)](#page-7-0) has the potential to degrade performance.

Additional details are provided in Appendix [I.](#page-20-0) In particular, the energy model for linear attention is discussed in Appendix [I.1.](#page-20-1)

## 5. Discussion

Motivated by the connection between attention mechanisms and dense associative memories, here we have introduced incontext denoising, a task that distills their relationship. We first analyze the general problem, deriving Bayes optimal predictors for certain restricted tasks. We identify that onelayer transformers using either softmax or linearized selfattention are expressive enough to describe these predictors. We then empirically demonstrate that standard training of attention layers from random initial weights will readily converge to scaled identity weights with scales that approach the derived optima given sufficient context. Accordingly, the rather minimal transformers studied here can perform optimal denoising of novel tasks provided at inference time via self-contained prompts. This work therefore sheds light on other in-context learning phenomena, a point we return to below.

While practical transformers differ in various ways from the minimal models studied here, we note several key connections. Intriguingly, the self-attention heads of trained transformers sometimes exhibit weights WKQ, WP V that resemble scaled identity matrices, i.e. cI +ϵ with small fluctuations ϵij ∼ N (0, σ<sup>2</sup> ), an observation noted in [Trockman](#page-10-6) [& Kolter](#page-10-6) [\(2023\)](#page-10-6). This phenomenon motivated their proposal of "mimetic" weight initialization schemes mirroring this learned structure. Relatedly, connections to associative memory concepts have been explored in other architectures [\(Smart & Zilman,](#page-10-7) [2021\)](#page-10-7), which enabled data-dependent weight initialization strategies to be identified and leveraged. More broadly, our study suggests that trained attention layers can readily adopt structures that facilitate context-aware associative retrieval. We have also noted preliminary connections between our work and other architectural features of modern transformers, namely layer normalization and residual streams, which warrant further study.

In-context denoising and generative modeling both involve learning about an underlying distribution, suggesting potential relationships between these two tasks. Recently, [Pham](#page-9-14) [et al.](#page-9-14) [\(2024\)](#page-9-14) invoked spurious states of the Hopfield model as a way of understanding how one can move away from retrieving individual memorized patterns towards generalization via appropriate mixtures of multiple similar "memories". In our work, one-step updates do not have to land in a spurious minimum, but we often operate under circumstances where there are such states (see, for example, the energy landscape in Fig. [5\)](#page-6-2). More generally, analogies between energy-based associative memory and diffusion models have recently been noted [\(Ambrogioni,](#page-8-11) [2024;](#page-8-11) [Hoover et al.,](#page-9-15) [2024b\)](#page-9-15). Lastly, Bayes optimal denoisers play an important role in the analysis [\(Ghio et al.,](#page-9-16) [2024\)](#page-9-16) of a very related generative model that is based on stochastic interpolants [\(Albergo & Vanden-Eijnden,](#page-8-13) [2023\)](#page-8-13). Although this work focuses on the case where it is possible to sample enough tokens from the relevant distributions for certain functions to converge, generative models become important when the distribution is in a prohibitively high-dimensional space making direct sampling difficult. Nonetheless, investigating the precise relationship between our work and different generative modeling approaches would be an interesting direction to pursue.

Overall, this work refines the connection between dense associative memories and attention layers first identified in [\(Ramsauer et al.,](#page-9-0) [2021\)](#page-9-0). While we show that one energy minimization step of a particular DAM (associated with a trained attention layer) is optimal for the denoising tasks studied here, it remains an open question whether multilayer architectures with varying or tied weights could extend these results to more complex tasks by effectively performing multiple iterative steps. This aligns with recent studies on in-context learning, which have considered whether transformers with multiple layers emulate gradient descent updates on a context-specific objective [\(Von Oswald](#page-10-5) [et al.,](#page-10-5) [2023;](#page-10-5) [Shen et al.,](#page-10-8) [2024;](#page-10-8) [Dai et al.,](#page-8-8) [2023;](#page-8-8) [Ahn et al.,](#page-8-9) [2023\)](#page-8-9), and may provide a bridge to work on emerging architectures guided by associative memory principles [\(Hoover](#page-9-11) [et al.,](#page-9-11) [2023\)](#page-9-11). Investigating when and how multilayer attention architectures perform such gradient descent iterations in a manner that is both context-dependent and informed by a large training set represents an exciting direction for future research at the intersection of transformer mechanisms, associative memory retrieval, and in-context learning.

- Software and Data Python code underlying this work is available at [https://github.com/mattsmart/in-context-denoising.](https://github.com/mattsmart/in-context-denoising) Acknowledgements MS acknowledges M. Mezard for very useful feedback on ´ an earlier version of this work. AS thanks D. Krotov and
- P. Mehta for enlightening discussions on related matters. Our early work also benefited from AS's participation in the deeplearning23 workshop at the Kavli Institute for Theoretical Physics (KITP), which was supported in part by grants NSF PHY-1748958 and PHY-2309135 to KITP. AS thanks
- Y. Bahri and C. Pehlevan for their patience and willingness to listen to our early ideas at KITP. Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. Transformers learn to implement preconditioned gradient descent for in-context learning. *Advances in Neural Information Processing Systems*, 36:45614–45650, 2023. Akyurek, E., Schuurmans, D., Andreas, J., Ma, T., and ¨ Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=0g0X4H8yN4I) [id=0g0X4H8yN4I](https://openreview.net/forum?id=0g0X4H8yN4I). Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2209.15571) [2209.15571](https://arxiv.org/abs/2209.15571). Amari, S.-I. Learning patterns and pattern sequences by selforganizing nets of threshold elements. *IEEE Transactions on computers*, 100(11):1197–1206, 1972. Ambrogioni, L. In search of dispersed memories: Generative diffusion models are associative memory networks. *Entropy*, 26(5), 2024. ISSN 1099-4300. doi: 10. 3390/e26050381. URL [https://www.mdpi.com/](https://www.mdpi.com/1099-4300/26/5/381) [1099-4300/26/5/381](https://www.mdpi.com/1099-4300/26/5/381). Amit, D. J., Gutfreund, H., and Sompolinsky, H. Spinglass models of neural networks. *Physical Review A*, 32 (2):1007–1018, 1985. ISSN 10502947. doi: 10.1103/ PhysRevA.32.1007. Bolle, D., Nieuwenhuizen, T. M., Castillo, I. P., and Ver- ´ beiren, T. A spherical hopfield model. *Journal of Physics A: Mathematical and General*, 36(41):10269, 2003. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33: 1877–1901, 2020. Chen, Y., Zeng, Q., Ji, H., and Yang, Y. Skyformer: Remodel self-attention with gaussian kernel and nystrom¨ method. *Advances in Neural Information Processing Systems*, 34:2122–2135, 2021. Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In *International Conference on Learning Representations*, 2021. URL [https://openreview.net/forum?](https://openreview.net/forum?id=Ua6zuk0WRH) [id=Ua6zuk0WRH](https://openreview.net/forum?id=Ua6zuk0WRH). Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei,
  - F. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023. URL <https://arxiv.org/abs/2212.10559>. Demircigil, M., Heusel, J., Lowe, M., Upgang, S., and ¨ Vermet, F. On a model of associative memory with huge storage capacity. *Journal of Statistical Physics*, 168:288– 299, 2017. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4171–4186, 2019. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby,
  - N. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*, 2021. URL [https://](https://openreview.net/forum?id=YicbFdNTTy) [openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy). Fischer, K. H. and Hertz, J. A. *Spin Glasses*. Cambridge University Press, 1993. Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. In *Advances in Neural Information*

- *Processing Systems*, volume 35, pp. 30583–30598, 2022. URL <https://arxiv.org/abs/2208.01066>. Ghio, D., Dandi, Y., Krzakala, F., and Zdeborova, L. Sam- ´ pling with flows, diffusion, and autoregressive neural networks from a spin-glass perspective. *Proceedings of the National Academy of Sciences*, 121(27):e2311810121, 2024. Gradshteyn, I. S. and Ryzhik, I. M. *Table of Integrals, Series, and Products*. Elsevier/Academic Press, Amsterdam, seventh edition, 2007. Hoeffding, W. Probability inequalities for sums of bounded random variables. *The collected works of Wassily Hoeffding*, pp. 409–426, 1994. Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H., Zaki, M. J., and Krotov, D. Energy transformer. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL [https:](https://openreview.net/forum?id=MbwVNEx9KS) [//openreview.net/forum?id=MbwVNEx9KS](https://openreview.net/forum?id=MbwVNEx9KS). Hoover, B., Chau, D. H., Strobelt, H., Ram, P., and Krotov,
- D. Dense associative memory through the lens of random features. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024a. Hoover, B., Strobelt, H., Krotov, D., Hoffman, J., Kira, Z., and Chau, D. H. Memory in plain sight: Surveying the uncanny resemblances of associative memories and diffusion models, 2024b. URL [https://arxiv.org/](https://arxiv.org/abs/2309.16750) [abs/2309.16750](https://arxiv.org/abs/2309.16750). Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences of the United States of America*, 79(8):2554–2558, 1982. ISSN 00278424. doi: 10.1073/pnas.79.8.2554. Hu, J. Y.-C., Yang, D., Wu, D., Xu, C., Chen, B.-Y., and Liu,
- H. On sparse modern hopfield model. In *Proceedings of the 37th International Conference on Neural Information Processing Systems*, NIPS '23, 2023. Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: fast autoregressive transformers with linear attention. In *Proceedings of the 37th International Conference on Machine Learning*, ICML'20. JMLR.org, 2020. Krotov, D. A new frontier for hopfield networks. *Nature Reviews Physics*, 5(7):366–367, 2023. Krotov, D. and Hopfield, J. J. Dense associative memory for pattern recognition. In *Advances in Neural Information Processing Systems*, volume 29, 2016. Krotov, D. and Hopfield, J. J. Large associative memory problem in neurobiology and machine learning. In *International Conference on Learning Representations*, 2021. URL [https://openreview.net/forum?](https://openreview.net/forum?id=X4y_10OX-hX) [id=X4y\\_10OX-hX](https://openreview.net/forum?id=X4y_10OX-hX). Little, W. A. The existence of persistent states in the brain. *Mathematical biosciences*, 19(1-2):101–120, 1974. Loeve, M. Probability theory i. ` *Graduate Texts in Mathematics*, 1977. Lucibello, C. and Mezard, M. Exponential capacity of ´ dense associative memories. *Phys. Rev. Lett.*, 132: 077301, Feb 2024. doi: 10.1103/PhysRevLett.132. 077301. URL [https://link.aps.org/doi/10.](https://link.aps.org/doi/10.1103/PhysRevLett.132.077301) [1103/PhysRevLett.132.077301](https://link.aps.org/doi/10.1103/PhysRevLett.132.077301). Millidge, B., Salvatori, T., Song, Y., Lukasiewicz, T., and Bogacz, R. Universal hopfield networks: A general framework for single-shot associative memory models. In *International Conference on Machine Learning*, pp. 15561– 15583. PMLR, 2022. Nakano, K. Associatron-a model of associative memory. *IEEE Transactions on Systems, Man, and Cybernetics*, 2: 380–388, 1972. Pham, B., Raya, G., Negri, M., Zaki, M. J., Ambrogioni, L., and Krotov, D. Memorization to generalization: The emergence of diffusion models from associative memory. In *NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning*, 2024. Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich, M., ¨ Gruber, L., Holzleitner, M., Adler, T., Kreil, D. P., Kopp,
  - M. K., Klambauer, G., Brandstetter, J., and Hochreiter, S. Hopfield networks is all you need. In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021. URL [https://openreview.net/forum?](https://openreview.net/forum?id=tL89RnzIiCd) [id=tL89RnzIiCd](https://openreview.net/forum?id=tL89RnzIiCd). Reddy, G. The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/](https://openreview.net/forum?id=aN4Jf6Cx69) [forum?id=aN4Jf6Cx69](https://openreview.net/forum?id=aN4Jf6Cx69). Rigollet, P. and Hutter, J.-C. High-dimensional statistics. ¨ *arXiv preprint arXiv:2310.19244*, 2023. Santos, S. J. R. D., Niculae, V., Mcnamee, D. C., and Martins, A. Sparse and structured hopfield networks. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 43368–43388. PMLR, 21–27 Jul 2024. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v235/santos24a.html) [v235/santos24a.html](https://proceedings.mlr.press/v235/santos24a.html).

- Shen, L., Mishra, A., and Khashabi, D. Position: Do pretrained transformers learn in-context by gradient descent? In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 44712–44740. PMLR, 21– 27 Jul 2024. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v235/shen24d.html) [press/v235/shen24d.html](https://proceedings.mlr.press/v235/shen24d.html). Smart, M. and Zilman, A. On the mapping between hopfield networks and restricted boltzmann machines. *International Conference on Learning Representations*, 2021. URL [https://openreview.net/forum?](https://openreview.net/forum?id=RGJbergVIoO) [id=RGJbergVIoO](https://openreview.net/forum?id=RGJbergVIoO). Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., ` Azhar, F., et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023. Trockman, A. and Kolter, J. Z. Mimetic initialization of selfattention layers. In *Proceedings of the 40th International Conference on Machine Learning*, ICML'23. JMLR.org, 2023. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Łukasz Kaiser, and Polosukhin, I. Attention is all you need. In *Advances in Neural Information Processing Systems*, volume 2017-December, 2017. Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov,
- M. Transformers learn in-context by gradient descent. In *International Conference on Machine Learning*, pp. 35151–35174. PMLR, 2023. Wu, D., Hu, J. Y.-C., Hsiao, T.-Y., and Liu, H. Uniform memory retrieval with larger capacity for modern hopfield models. In *Proceedings of the 41st International Conference on Machine Learning*, ICML'24. JMLR.org, 2024a. Wu, D., Hu, J. Y.-C., Li, W., Chen, B.-Y., and Liu, H. Stanhop: Sparse tandem hopfield model for memoryenhanced time series prediction. In *The Twelfth International Conference on Learning Representations*, 2024b. URL <https://arxiv.org/abs/2312.17346>. Zhang, R., Frei, S., and Bartlett, P. L. Trained transformers learn linear models in-context. *Journal of Machine Learning Research*, 25(49):1–55, 2024. URL [http:](http://jmlr.org/papers/v25/23-1042.html) [//jmlr.org/papers/v25/23-1042.html](http://jmlr.org/papers/v25/23-1042.html).

## A. Notation

## A.1. Recurring notation

- n ambient dimension of input tokens.
- x<sup>t</sup> ∈ <sup>R</sup> <sup>n</sup> – the value of the t-th random input token.
- E = (X1, ..., XL, X˜) the random variable input to the sequence model. The "tilde" indicates that the final token has in some way been corrupted. E takes values (x1, ..., xL, x˜) ∈ <sup>R</sup> <sup>n</sup>×(L+1). Note: while capital X or X<sup>i</sup> here denotes a random variable, in Section [D](#page-14-2) use X1:<sup>L</sup> or simply X to refer to the realized matrix of input tokens.
- L context length = number of uncorrupted tokens.
- d dimensionality of manifold S that x<sup>t</sup> are sampled from
- N number of training pairs

### A.2. Bayes posterior notation

- pX(x) is task-dependent (the three scenarios considered here are introduced above).
- pX˜ (˜x) where x˜ = x + z. For a sum of independent random variables, Y = X<sup>1</sup> + X2, their pdf is a convolution p<sup>Y</sup> (y) = R p<sup>X</sup><sup>1</sup> (x)p<sup>X</sup><sup>2</sup> (y − x)dx. Thus:

$$\begin{aligned} p_{\tilde{X}}(\tilde{x}) &= \int p_Z(z) p_X(\tilde{x} - z) dz \\ &= C_Z \int e^{-\|z\|^2/2\sigma_Z^2} p_X(\tilde{x} - z) dz \end{aligned}$$

where C<sup>Z</sup> = (2πσ<sup>2</sup> Z ) −n/2 is a constant.

- pX˜|X(˜x | x): This is simply

$$p_Z(\tilde{x} - x) = C_Z e^{-\|\tilde{x} - x\|^2/2\sigma_Z^2}.$$

- pX|X˜ (x | x˜): By Bayes' theorem, this is

$$\begin{aligned} p_{X|\tilde{X}}(x \mid \tilde{x}) &= \frac{p_{\tilde{X}|X}(\tilde{x} \mid x)p_X(x)}{p_{\tilde{X}}(\tilde{x})} \\ &= \frac{e^{-\|\tilde{x}-x\|^2/2\sigma_Z^2}p_X(x)}{\int e^{-\|\tilde{x}-x\|^2/2\sigma_Z^2}p_X(x')dx'}. \end{aligned}$$

- Posterior mean:

$$\begin{aligned}\mathbb{E}_{X|\tilde{X}}[X \mid \tilde{X}] &= \int x p_{X|\tilde{X}}(x \mid \tilde{X}) dx \\ &= \frac{\int x e^{-\|\tilde{X}-x\|^2/2\sigma_z^2} p_X(x) dx}{\int e^{-\|\tilde{X}-x\|^2/2\sigma_z^2} p_X(x) dx}.\end{aligned}$$

## B. Bayes optimal predictors for square loss

## B.1. Proof of Proposition [1](#page-2-2)

*Proof.* Observe that

$$\begin{aligned}\mathbb{E} \left[ \|X - f(\tilde{X})\|^2 \right] &= \mathbb{E}_{\tilde{X}} \left[ \mathbb{E}_{X|\tilde{X}} [\|X - f(\tilde{X})\|^2 \mid \tilde{X}] \right] \\ &= \mathbb{E}_{\tilde{X}} \left[ \mathbb{E}_{X|\tilde{X}} [\|X - \mathbb{E}[X \mid \tilde{X}]\|^2 \mid \tilde{X}] \right. \\ &\quad \left. + \|\mathbb{E}[X \mid \tilde{X}] - f(\tilde{X})\|^2 \right] \\ &\geq \mathbb{E}_{\tilde{X}} \left[ \mathbb{E}_{X|\tilde{X}} [\|X - \mathbb{E}[X \mid \tilde{X}]\|^2 \mid \tilde{X}] \right] \\ &= \mathbb{E}_{\tilde{X}} \left[ \text{Tr Cov}(X \mid \tilde{X}) \right].\end{aligned}$$

Note the final line is independent of f. This inequality becomes an equality when f(X˜) = <sup>E</sup>[X | X˜].

# C. Details of Bayes optimal denoising baselines for each case

## C.1. Proof of Proposition [2](#page-3-3)

*Proof.* The linear denoising task is a special case of the result in Proposition [1.](#page-2-2) Here, X is an isotropic Gaussian in a restricted subspace,

$$p_{X|\tilde{X}}(x \mid \tilde{x}) = C(\tilde{x})p_X(x)e^{-\frac{\|x - \tilde{x}\|^2}{2\sigma_Z^2}}$$

where C(˜x) is a normalizing factor. The noise can be decomposed into parallel and perpendicular parts using the projection P onto S, i.e.

$$\tilde{X} = \tilde{X}_{\parallel} + \tilde{X}_{\perp} = P\tilde{X} + (I - P)\tilde{X},$$

so that

$$e^{-\frac{\|x-\bar{x}\|^2}{2\sigma_Z^2}} = e^{-\frac{\|x-\bar{x}_\perp\|^2}{2\sigma_Z^2}} e^{-\frac{\|\bar{x}_\perp\|^2}{2\sigma_Z^2}}.$$

Only the first factor matters for pX|X˜ (x | x˜) since it depends on x. Then, for x ∈ S, the linear subspace supporting pX, dropping the x independent x˜<sup>⊥</sup> contribution,

$$p_X(x)e^{-\frac{\|x-\tilde{x}\|_2^2}{2\sigma_Z^2}} \propto e^{-\frac{\|x-x_{\parallel}\|_2^2}{2\sigma_Z^2}} - \frac{\|x-\tilde{x}\|_2^2}{2\sigma_Z^2}$$

$$\propto \exp\left(-\frac{\frac{\sigma_0^2}{\sigma_0^2+\sigma_Z^2}\tilde{x}_{\parallel}\|^2}{2\frac{\sigma_0^2\sigma_Z^2}{\sigma_0^2+\sigma_Z^2}}\right)$$

Thus, f(X˜) = <sup>σ</sup> σ <sup>0</sup>+σ Z X˜<sup>∥</sup> = σ σ <sup>0</sup>+σ Z P X˜.

Thus, 
$$f(\tilde{X}) = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} \tilde{X}_{\parallel} = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} P \tilde{X}$$
.

Using X˜ = X + Z, X = P X, and the independence of X and Z

$$\mathbb{E}\left[\|X - \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} P\tilde{X}\|^2\right] = \mathbb{E}\left[\left\|\frac{\sigma_Z^2}{\sigma_0^2 + \sigma_Z^2} PX\right\|^2\right] + \mathbb{E}\left[\left\|\frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} PZ\right\|^2\right] = \frac{\sigma_Z^4 d\sigma_0^2 + \sigma_0^4 d\sigma_Z^2}{(\sigma_0^2 + \sigma_Z^2)^2} = \frac{d\sigma_0^2 \sigma_Z^2}{\sigma_0^2 + \sigma_Z^2}.$$

## C.2. Proof of Proposition [3](#page-3-4)

*Proof.* In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres S (e.g. the circle S <sup>1</sup> ⊂ <sup>R</sup> ). For such manifolds, we have

$$\begin{aligned}\mathbb{E}[X \mid \tilde{X} = \tilde{x}] &= \frac{\int e^{-\frac{\|x-\tilde{x}\|^2}{2\sigma_Z^2}} x p_X(x) dx}{\int e^{-\frac{\|x-\tilde{x}\|^2}{2\sigma_Z^2}} p_X(x) dx} \\ &= \frac{\int e^{\langle x, \tilde{x} \rangle} \frac{\sigma_Z^2}{\sigma_Z^2} x dS_x}{\int e^{\langle x, \tilde{x} \rangle} \frac{\sigma_Z^2}{\sigma_Z^2} dS_x}.\end{aligned}$$

We have used the fact that ∥x − x˜∥∥ <sup>2</sup> = ∥x∥ <sup>2</sup> + ∥x˜∥∥ <sup>2</sup> − 2⟨x, x˜∥⟩ and that ∥x∥ is fixed on the sphere.

The integrals can be evaluated directly once the parameters are specified. If S is a d–sphere of radius R, then the optimal predictor is again a shrunk projection of x˜ onto S,

$$\begin{aligned} & \frac{\int_0^\pi e^{R\|\tilde{x}_{\parallel}\|} \cos \theta / \sigma_z^2 \cos \theta \sin^{(d-1)} \theta \, d\theta}{\int_0^\pi e^{R\|\tilde{x}_{\parallel}\|} \cos \theta / \sigma_z^2 \sin^{(d-1)} \theta \, d\theta} R \frac{\tilde{x}_{\parallel}}{\|\tilde{x}_{\parallel}\|} \\ &= \frac{I_{\frac{d+1}{2}} \left( R \frac{\|\tilde{x}_{\parallel}\|}{\sigma_z^2} \right)}{I_{\frac{d-1}{2}} \left( R \frac{\|\tilde{x}_{\parallel}\|}{\sigma_z^2} \right)} R \frac{\tilde{x}_{\parallel}}{\|\tilde{x}_{\parallel}\|}, \end{aligned}$$

where we used identities involving Iν(y), modified Bessel function of the first kind of order ν [\(Gradshteyn & Ryzhik,](#page-9-17) [2007\)](#page-9-17). The vector R x˜<sup>∥</sup> ∥x˜∥∥ is the point on S in the direction of x∥.

## C.3. Proof of Proposition [4](#page-3-1)

*Proof.* For the clustering case involving isotropic Gaussian mixtures with parameters {wa,(µa, σ<sup>2</sup> a )} p <sup>a</sup>=1,

$$\mathbb{E}[X \mid \tilde{X} = \tilde{x}] = \frac{\int e^{-\frac{\|x-\tilde{x}\|^2}{2\sigma_a^2}} \sum_a \left( w_a C_a e^{-\frac{\|x-\mu_a\|^2}{2\sigma_a^2}} \right) dx dx}{\int e^{-\frac{\|x-\tilde{x}\|^2}{2\sigma_a^2}} \sum_a \left( w_a C_a e^{-\frac{\|x-\mu_a\|^2}{2\sigma_a^2}} \right) dx},$$

where C<sup>a</sup> = (2πσ<sup>2</sup> a ) − n 2 .

We can simplify this expression by completing the square in the exponent and using the fact that the integral of a Gaussian about its mean is zero. This yields

$$\mathbb{E}[X \mid \tilde{X} = \tilde{x}] = \frac{\sum_a w_a C_a m_a \int \exp(-g_a) dx}{\sum_a w_a C_a \int \exp(-g_a) dx}$$

where we have introduced

$$g_a = \frac{1}{2} \left( \frac{\sigma_Z^2 + \sigma_a^2}{\sigma_Z^2 \sigma_a^2} \right) \|x - m_\alpha\|^2 + \frac{1}{2(\sigma_Z^2 + \sigma_a^2)} \|\tilde{x} - \mu_\alpha\|^2,$$

with

$$m_a = \frac{\sigma_a^2 \tilde{x} + \sigma_Z^2 \mu_a}{\sigma_a^2 + \sigma_Z^2}.$$

Doing the integrals and using the expressions for Ca, m<sup>a</sup>

$$\mathbb{E}[X \mid \tilde{X} = \tilde{x}] = \frac{\sum_a w_a \left(\frac{\sigma_Z^2 + \sigma_a^2}{\sigma_a^2}\right)^{n/2} \exp\left(-\frac{\|\tilde{x} - \mu_a\|^2}{2(\sigma_Z^2 + \sigma_a^2)}\right) \left(\frac{\sigma_a^2 + \sigma_Z^2}{\sigma_a^2 + \sigma_a^2} \mu_a\right)}{\sum_a w_a \left(\frac{\sigma_Z^2 + \sigma_a^2}{\sigma_a^2}\right)^{n/2} \exp\left(-\frac{\|\tilde{x} - \mu_a\|^2}{2(\sigma_Z^2 + \sigma_a^2)}\right)}$$

In the case that the center norms ∥µa∥ are independent of a and variances σ 2 <sup>a</sup> = σ0, we have

$$\mathbb{E}[X \mid \tilde{X} = \tilde{x}] = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_Z^2} \tilde{x} + \frac{\sigma_Z^2}{\sigma_0^2 + \sigma_Z^2} \frac{\sum_a w_a \mu_a \exp\left(\frac{\langle \tilde{x}, \mu_a \rangle}{\sigma_Z^2 + \sigma_0^2}\right)}{\sum_a w_a \exp\left(\frac{\langle \tilde{x}, \mu_a \rangle}{\sigma_Z^2 + \sigma_0^2}\right)}.$$

Note that in the limit that σ<sup>0</sup> → 0 , this becomes expressible by one-layer self-attention, since one can simply replace the matrix of cluster centers M = [µ<sup>1</sup> . . . µp] implicit in the expression with the context X1:<sup>L</sup> itself,

$$\mathbb{E}[X \mid \tilde{X}] = \frac{\sum_a w_a e^{\langle \mu_\alpha, \tilde{X} \rangle} / \sigma_z^2 \mu_a}{\sum_a w_a e^{\langle \mu_\alpha, \tilde{X} \rangle} / \sigma_z^2}.$$

## D. Additional details on attention layers and softmax expansion

## D.1. Standard self-attention

Given a sequence of Lseq input tokens x<sup>i</sup> ∈ <sup>R</sup> <sup>n</sup> represented as a matrix X ∈ <sup>R</sup> <sup>n</sup>×Lseq , standard self-attention defines query, key, and value matrices

$$K = W_K X, Q = W_Q X, V = W_V X \quad (\text{A.1})$$

where WK, W<sup>Q</sup> ∈ <sup>R</sup> <sup>n</sup>attn×<sup>n</sup> and W<sup>V</sup> ∈ <sup>R</sup> <sup>n</sup>out×<sup>n</sup>. The softmax self-attention map [\(Vaswani et al.,](#page-10-0) [2017\)](#page-10-0) is then

$$\text{Attn}(X, W_V, W_K^T W_Q) := V \text{softmax}(K^T Q) \in \mathbb{R}^{n_{\text{out}} \times L_{\text{seq}}}. \quad (\text{A.2})$$

On merging WK, W<sup>Q</sup> into WKQ = W<sup>T</sup> <sup>K</sup>WQ: The simplification WKQ = W<sup>T</sup> <sup>K</sup>W<sup>Q</sup> (made here and elsewhere) is general only when nattn ≥ n; in that case, the product WKQ can have rank n and thus it is reasonable to work with the combined matrix. On the other hand, if nattn < n, then the rank of their product is at most nattn and thus there are matrices in <sup>R</sup> n×n that cannot be expressed as W<sup>T</sup> <sup>K</sup>WQ. A similar point can be made about WP V . We note that while nattn < n may be used in practical settings, one often also uses multiple heads which when concatenated could be (roughly) viewed as a single higher-rank head.

We will also use the simplest version of linear attention [\(Katharopoulos et al.,](#page-9-18) [2020\)](#page-9-18),

$$\text{Attn}_{\text{Lin}}(X, W_V, W_K^T W_Q) := \frac{1}{L_{\text{seq}}} V(K^T Q) \in \mathbb{R}^{n_{\text{out}} \times L_{\text{seq}}}. \quad (\text{A.3})$$

## D.2. Minimal transformer architecture for denoising

We now consider a simplified one-layer linear transformer in term of our variable E = (X1:L, X˜) taking values in <sup>R</sup> n×(L+1) and start with the linear transformer which still has sufficient expressive power to capture our finite sample approximation to the Bayes optimal answer in the linear case. Inspired by [Zhang et al.](#page-10-4) [\(2024\)](#page-10-4), we define

$$\text{Attn}_{\text{Lin}}(E, W_{PV}, W_{KQ}) := \frac{1}{L} W_{PV} E M_{\text{Lin}} E^T W_{KQ} E \quad (\text{A.4})$$

taking values in R <sup>n</sup>×(L+1). The additional aspect compared to the last subsection is the masking matrix MLin ∈ R (L+1)×(L+1) which is of the form

$$M_{\text{Lin}} = \begin{bmatrix} I_L & 0_{L \times 1} \\ 0_{1 \times L} & 0 \end{bmatrix}, \quad (\text{A.5})$$

preventing WP V X˜ from being added to the output.

Note that this more detailed expression is equivalent to the form used in the main text.

$$\hat{X} = F_{\text{Lin}}(E, \theta) := \frac{1}{L} W_{PV} X_{1:L} X_{1:L}^T W_{KQ} \tilde{X}$$

With learnable weights WKQ, WP V ∈ <sup>R</sup> <sup>n</sup>×<sup>n</sup> abbreviated by θ, we define

$$F(E, \theta) := [\text{Attn}_{\text{Lin}}(E, W_{PV}, W_{KQ})]_{:, L+1}. \quad (\text{A.6})$$

Note that, when WP V = αIn, WKQ = βIn, and αβ = σ <sup>0</sup>+σ , F(E, θ) should approximate the Bayes optimal answer fopt(X˜) as L → ∞.

Similarly, we could argue that the second two problems, the d-dimesional spheres and the σ<sup>0</sup> → 0 zero limit of the Gaussian mixtures could be addressed by the full softmax attention

$$\text{Attn}(E, W_{PV}, W_{KQ}) = W_{PV} \text{Esoftmax}(E^T W_{KQ} E + M) \quad (\text{A.7})$$

taking values in R <sup>n</sup>×(L+1) where M ∈ <sup>R</sup>¯(L+1)×(L+1) is a masking matrix of the form

$$M = \begin{bmatrix} 0_{L \times (L+1)} \\ (-\infty)1_{1 \times L+1} \end{bmatrix}, \quad (\text{A.8})$$

once more, preventing the contribution of <sup>X</sup>˜ value to the output. The function softmax(z) := <sup>P</sup> 1 n <sup>i</sup>=1 e (e z<sup>1</sup> , . . . , e<sup>z</sup><sup>n</sup> ) <sup>T</sup> ∈ <sup>R</sup> n is applied column-wise.

We then define

$$F(E, \theta) := [\text{Attn}(E, W_{PV}, W_{KQ})]_{\cdot, L+1}, \quad (\text{A.9})$$

which is equivalent to the simplified form used in the main text:

$$\hat{X} = F(E, \theta) := W_{PV} X_{1:L} \text{softmax}(X_{1:L}^T W_{KQ} \tilde{X}).$$

## D.3. Proof of Theorem [3.1](#page-4-1)

*Proof.* Let the support of p<sup>X</sup> be a subset of a sphere, centered around the origin, of radius R. Then the function

$$g(\{X_t\}_{t=1}^L, \tilde{x}) = \frac{\sum_{t=1}^L X_t e^{\langle X_t, \tilde{x} \rangle / \sigma_Z^2}}{\sum_{t=1}^L e^{\langle X_t, \tilde{x} \rangle / \sigma_Z^2}} = \frac{\frac{1}{L} \sum_{t=1}^L X_t e^{\langle X_t, \tilde{x} \rangle / \sigma_Z^2}}{\frac{1}{L} \sum_{t=1}^L e^{\langle X_t, \tilde{x} \rangle / \sigma_Z^2}}. \quad (\text{A.10})$$

Both the numerator <sup>1</sup> L P<sup>L</sup> <sup>t</sup>=1 Xte ⟨Xt,x˜⟩/σ<sup>2</sup> <sup>Z</sup> and the denominator <sup>1</sup> L P<sup>L</sup> <sup>t</sup>=1 e ⟨Xt,x˜⟩/σ<sup>2</sup> <sup>Z</sup> are averages of independent and identically distributed bounded random variables. By the strong law of large numbers, as L → ∞, the average vector in the numerator converges to almost surely to R e ⟨x,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> x dpX(x) for each component, while the average in the denominator almost surely converges R e ⟨x,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> dpX(x), which is positive. So, as L → ∞, the ratio in Eq. [A.10](#page-15-2) converges almost surely to

$$\frac{\int e^{\langle x, \tilde{x}_{\parallel} \rangle} / \sigma_Z^2 x dp_X(x)}{\int e^{\langle x, \tilde{x}_{\parallel} \rangle} / \sigma_Z^2 dp_X(x)},$$

which is the Bayes optimal answer fopt(˜x) for all x˜ ∈ <sup>R</sup> n.

# E. Further discussion of convergence rates as L → ∞ and the dependence on dimensions

Our analysis primarily focused on the asymptotic behavior as L → ∞ using the strong law of large numbers, which just requires the mean to exist [\(Loeve](#page-9-19) ` , [1977\)](#page-9-19). However, in the linear example, our tokens are Gaussian, and in the two nonlinear cases they are bounded. Intuitively, we expect error <sup>O</sup>( √ 1 L ). In fact, we can give precise results of the form that the probability of the difference between the empirical sum for the ideal weights departing from the expectation by less than C(˜x) r f d,ln <sup>1</sup> δ L is greater than 1 − δ. The function C of the query vector and the function f depend on the problem. Interestingly, these bounds depend on d, the dimension spanned by the tokens, not the ambient dimension n.

As mentioned before, the results of the previous paragraph refer to the convergence of the finite sample attention expressions for ideal weights, namely those corresponding to Bayes optimal answer. There is a second source of error associated with finite sample estimation of weights, which should also get small as L becomes large. Once more the expectation is that the weights are known to error <sup>O</sup>( √ L ) for well-converged training procedures, although this is more difficult to guarantee or quantify analytically. Overall we expect the loss (MSE) to go down inversely with some power of L. Fig. [4\(](#page-6-1)a) provides some empirical evidence for this relationship, showing how performance improves with increasing context length.

Notice that the one-layer transformer output is a linear combination of the uncorrupted samples. Hence, if the distribution p<sup>X</sup> is supported by a d-dimensional linear subspace, the estimate Xˆ is also in that subspace. We can therefore look at convergence restricted to the supporting subspace. Therefore, it is the dimensionality of the supporting subspace that matters.

Let a d-dimensional vector space V be a linear subspace of R <sup>n</sup>. We define the maximum norm for V with respect to some orthonormal basis {vi} d <sup>i</sup>=1 in V as ||x||<sup>∞</sup>,V := maxi∈{1,...,d} |⟨v<sup>i</sup> , x⟩| for any x ∈ V . The conventional maximum norm for R <sup>n</sup>, of course, is defined with respect to the standard orthonormal basis {ej} n <sup>j</sup>=1. Since |⟨v<sup>i</sup> , x⟩| ≤ ||x||<sup>∞</sup>,V , for all i,

$$\|x\|_2^2 = \sum_{i=1}^d (\langle v_i, x \rangle)^2 \leq d\|x\|_{\infty,V}^2 \implies \|x\|_2 \leq \sqrt{d}\|x\|_{\infty,V}.$$

Then, for any x ∈ V ⊆ R <sup>n</sup>, ||x||<sup>∞</sup> ≤ √ d||x||<sup>∞</sup>,V , since |⟨x, e<sup>j</sup> ⟩| ≤ ||x||<sup>2</sup> ≤ √ d||x||<sup>∞</sup>,V , for all j ∈ {1 . . . , n}. Thus, controlling component-wise error in any orthonormal basis in V controls component-wise error in R <sup>n</sup>, in an n-independent but d-dependent manner. In the following, we give a flavor of how we can analyze finite sample estimate errors in V . The maximum norm || · ||<sup>∞</sup> is to be understood as || · ||<sup>∞</sup>,V for some orthonormal basis choice. Here is the result relevant to the linear case described Subsubsection [2.2.1.](#page-1-1)

Proposition 5. *Let* X<sup>t</sup> *i.i.d* ∼ N (0, σ<sup>2</sup> 0 Id), t = 1, . . . , L *and let* Π := ˆ <sup>1</sup> σ <sup>0</sup>L P<sup>L</sup> <sup>t</sup>=1 <sup>X</sup>tX<sup>T</sup> t . *Then, for any* δ ∈ (0, 1),

$$Pr \left[ \|\hat{\Pi}\tilde{x} - \tilde{x}\|_{\infty} < C\|\tilde{x}\|_2 \max \left\{ \sqrt{\frac{d + \ln(\frac{2}{\delta})}{L}}, \frac{d + \ln(\frac{2}{\delta})}{L} \right\} \right] > 1 - \delta$$

*for some* C > 0*.*

*Proof.* We start by bounding the maximum norm of the difference,

$$\|\hat{\Pi}\tilde{x} - \tilde{x}\|_\infty \leq \|\hat{\Pi}\tilde{x} - \tilde{x}\|_2 \leq \|\hat{\Pi} - I_d\|_{\text{op}} \|x\|_2,$$

where || · ||op is the operator norm.

It can be shown that, for any δ ∈ (0, 1)

$$\Pr \left[ |\hat{\Pi} - I_d|_{\text{op}} < C \max \left\{ \sqrt{\frac{d + \ln(\frac{2}{\delta})}{L}}, \frac{d + \ln(\frac{2}{\delta})}{L} \right\} \right] > 1 - \delta$$

for some C > 0 [\(Rigollet & Hutter](#page-9-20) ¨ , [2023\)](#page-9-20). Combining with the first bound, we get our result.

As to the nonlinear cases, the key result of Theorem [3.1](#page-4-1) is the convergence of the numerator <sup>1</sup> L P<sup>L</sup> <sup>t</sup>=1 Xte ⟨Xt,x˜∥⟩/σ<sup>2</sup> Z to <sup>E</sup>[Xe⟨X,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> ] = R e ⟨x,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> x dpX(x) and the denominator <sup>1</sup> L P<sup>L</sup> <sup>t</sup>=1 e ⟨Xt,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> to <sup>E</sup>[e ⟨X,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> ] = R e ⟨x,x˜∥⟩/σ<sup>2</sup> <sup>Z</sup> dpX(x).

In the following, we assume that the support of p<sup>X</sup> is inside a vector space V whose dimension we denote by d (instead of d + 1, as in the sphere problem). In addition, we refer to the projection of the query on V by x˜ ∈ V , instead of x˜∥. As usual, the maximum norm in V is with respect to some orthonormal basis choice

Proposition 6. *Let* X<sup>t</sup> *i.i.d* ∼ <sup>p</sup><sup>X</sup> *and* ||Xt||<sup>2</sup> ≤ <sup>R</sup> *for* <sup>t</sup> = 1, . . . , L*.*

*Then, for any* δ ∈ (0, 1),

$$Pr \left[ \left| \frac{1}{L} \sum_{t=1}^L e^{\langle X_t, \tilde{x} \rangle / \sigma_z^2} - \mathbb{E}[e^{\langle X, \tilde{x} \rangle / \sigma_z^2}] \right| < \sinh \left( \frac{R \|\tilde{x}\|_2}{\sigma_z^2} \right) \sqrt{\frac{2}{L} \ln \left( \frac{2}{\delta} \right)} \right] \geq 1 - \delta$$

*and*

$$Pr \left[ \left\| \frac{1}{L} \sum_{t=1}^L X_t e^{\langle X_t, \tilde{x} \rangle / \sigma_z^2} - \mathbb{E}[X e^{\langle X, \tilde{x} \rangle / \sigma_z^2}] \right\|_{\infty} < Re^{\frac{R \|\tilde{x}\|_2}{\sigma_z^2}} \sqrt{\frac{2}{L} \ln \left( \frac{2d}{\delta} \right)} \right] \geq 1 - \delta.$$

*Proof.* We provide the sketch of our proof here, the key ingredient of which is the Hoeffding inequality [\(Hoeffding,](#page-9-21) [1994\)](#page-9-21).

For the average <sup>1</sup> L P<sup>L</sup> <sup>t</sup>=1 e ⟨Xt,x˜⟩/σ<sup>2</sup> <sup>Z</sup> , each term in the sum is bounded above and below by e ± R||x˜||2 σ2 <sup>Z</sup> . So, the Hoeffding inequality leads to

$$\Pr \left[ \left| \frac{1}{L} \sum_{t=1}^L e^{\langle X_t, \tilde{x} \rangle / \sigma_Z^2} - \mathbb{E}[e^{\langle X, \tilde{x} \rangle / \sigma_Z^2}] \right| \geq \epsilon \right] \leq 2 \exp \left[ - \frac{2L\epsilon^2}{\left( \exp \left( \frac{R\|\tilde{x}\|_2}{\sigma_Z^2} \right) - \exp \left( - \frac{R\|\tilde{x}\|_2}{\sigma_Z^2} \right) \right)^2} \right] = 2 \exp \left[ - \frac{L\epsilon^2}{2 \sinh^2 \left( \frac{R\|\tilde{x}\|_2}{\sigma_Z^2} \right)} \right].$$

Z

Z

Setting <sup>δ</sup> = 2 exp h − Lϵ<sup>2</sup> 2 sinh<sup>2</sup> <sup>R</sup>||x˜||<sup>2</sup> σ i , we get <sup>ϵ</sup> = sinh <sup>R</sup>||x˜||<sup>2</sup> σ Z q 2 L ln δ , which gives our first probabilistic inequality.

Z

For each component of the vector average <sup>1</sup> L P<sup>L</sup> <sup>t</sup>=1 Xte ⟨Xt,x˜⟩/σ<sup>2</sup> <sup>Z</sup> , the terms in the sum are bounded above and below by ±R R||x˜||2 σ2 <sup>Z</sup> . We use similar arguments involving the Hoeffding inequality, combined with the union bound over all d coordinates

$$\Pr \left[ \left| \frac{1}{L} \sum_{t=1}^L X_t e^{\langle X_t, \tilde{x} \rangle / \sigma_Z^2} - \mathbb{E}[X e^{\langle X, \tilde{x} \rangle / \sigma_Z^2}] \right| \right|_{\infty} \geq \epsilon ] \leq 2d \exp \left[ - \frac{L\epsilon^2}{2R^2 \exp \left( \frac{2R \|\tilde{x}\|_2}{\sigma_Z^2} \right)} \right].$$

Once more, setting the RHS to δ and solving for ϵ, we get our second probabilistic inequality.

# F. Limiting behaviors of the softmax function and softmax attention

### For small argument

A Taylor expansion of the softmax function at zero gives

$$\text{softmax}(\beta v) = \frac{1}{Z} (\mathbb{1}_L + \beta v + O(\beta^2)),$$

where Z = P i 1 + βv<sup>i</sup> + O(β 2 )) = L(1 + βv¯ + O(β 2 )) is a normalizing factor, with v¯ = 1 L P i vi . The notation <sup>1</sup><sup>L</sup> stands for an L-dimensional vector of ones.

Thus, we have

Lemma F.1 (Small argument expansion of softmax). *As* β → 0*,*

$$\text{softmax}(\beta v) = \frac{1}{L(1 + \beta \bar{v} + O(\beta^2))} (\mathbb{1}_L + \beta v + O(\beta^2)) = \frac{1}{L} (\mathbb{1}_L + \beta(v - \bar{v}\mathbb{1}) + O(\beta^2)).$$

## F.1. Proof of Proposition [3.2](#page-4-0)

*Proof.*

$$F\left(E, \left(\frac{1}{\epsilon}W_{PV}, \epsilon W_{KQ}\right)\right) := \frac{1}{\epsilon}W_{PV}X_{1:L}\text{softmax}(\epsilon X_{1:L}^T W_{KQ} \tilde{X}).$$

Using Lemma [F.1,](#page-17-2) as ϵ → 0,

$$\begin{aligned} F\left(E, \left(\frac{1}{\epsilon}W_{PV}, \epsilon W_{KQ}\right)\right) &= \frac{1}{\epsilon}W_{PV}X_{1:L} \left[ \frac{1}{L} \left( \mathbb{1}_L + \epsilon(X_{1:L}^T W_{KQ} \tilde{X} - \left(\frac{1}{L} \sum_t X_t^T W_{KQ} \tilde{X}\right) \mathbb{1}_L) + O(\epsilon^2) \right) \right] \\ &= \frac{1}{\epsilon}W_{PV} \bar{X} + \frac{1}{L}W_{PV} \sum_{t=1}^L X_t(X_t - \bar{X})^T W_{KQ} \tilde{X} + O(\epsilon), \end{aligned} \quad (\text{A.11})$$

where X¯ = L P<sup>L</sup> <sup>t</sup>=1 X<sup>t</sup> is the empirical mean and the notation <sup>1</sup><sup>L</sup> emphasizes that it is a column vector of ones with dimension L.

### For large argument

As β → ∞, the softmax function simply selects the maximum over its inputs (as long as the the maximum is unique):

$$\text{softmax}(\beta v) \approx \begin{cases} 1 & \text{if } i = \arg \max_j v_j, \\ 0 & \text{otherwise.} \end{cases}$$

In this case, all attention weight is given to a single element, and the others are effectively ignored.

# G. MSE Loss landscape for scaled identity weights

![](_page_18_Figure_9.jpeg)

Figure 6. Loss landscape corresponding to Case 2 and Case 3 of Fig. [3.](#page-5-0) The MSE is numerically evaluated by assuming scaled identity weights WKQ = βI<sup>n</sup> (x-axis) and WP V = αI<sup>n</sup> (y-axis) and scanning over a 50 × 50 grid. The green point corresponds to the heuristic minimizer identified from the posterior mean. In Case 2 it is exact, while in case 3 it is an approximation that neglects the residual term (see Proposition [4\)](#page-3-1). The orange point corresponds to the learned weights displayed in Fig. [3\(](#page-5-0)b), while the white point corresponds to the numerically identified minimum from this 2D scan. These can fluctuate due to the finite context (L = 500) and sampling (N = 800 here). In both panels, it is apparent that the trained weights and the heuristic estimator co-occur in a broad valley (contour) of the loss landscape.

The loss landscapes in Fig. [6](#page-18-1) exhibit large, low-cost valleys with a roughly hyperbolic structure that is especially apparent in Case 2. This indicates a multiplicative tradeoff in the scales of WKQ and WP V , which suggests that linear attention might be applicable here as well. For completeness, Figure [7](#page-19-0) shows linear attention performance for both cases, demonstrating that it performs quite similarly to softmax for sub-sphere denoising, but less well in the Gaussian mixtures case.

# H. Structured optimal weights under prompt transformation

We find that one-layer transformers can learn to undo arbitrary invertible coordinate transformations that warp the denoising tasks. Focusing on the subspace denoising case, suppose each prompt is transformed by a fixed invertible square matrix

![](_page_19_Figure_2.jpeg)

Figure 7. Linear attention performance for Cases 2 and 3. Additional empirical results for the nonlinear manifolds case (left) and the Gaussian mixtures case (right). (a) Loss dynamics for randomly initialized softmax and linear attention layers. Solid lines represent the average loss over six seeds, with shaded area indicating the range. Training details and parameters follow Fig. [3\(](#page-5-0)a). (b) Representative final attention weights for each layer.

A, i.e. E = (X1:L, x˜) → E′ = (AX1:L, Ax˜). If the target remains xL+1 in the untransformed space, then the optimal attention weights are no longer diagonal, but instead take a structured form determined by the transformation matrix:

$$W_{PV} = \alpha A^{-1}, \quad W_{KQ} = \beta (AA^T)^{-1}, \quad (\text{A.12})$$

where αβ = 1 σ <sup>0</sup>+σ Z as before.

![](_page_19_Figure_7.jpeg)

Figure 8. (a) Example transformation A used to globally alter the in-context denoising prompts. (b) Structure of the optimal attention weights for this transformed subspace-denoising task. (c,d) Empirically, we find that both linear attention and softmax attention layers are able to learn these structured targets, but with distinct scalings α, β. Final weights after 500 epochs using Adam, random initializations, and context length L = 500; other parameters follow Fig. [3\(](#page-5-0)a).

We use the same basic training procedure as the limiting case of A = I (no additional coordinate transformation) assumed throughout the main text.

Suppose we still work with transformed coordinates Y<sup>t</sup> = AX<sup>t</sup> and Y˜ = AX˜, but now intend to retrieve YL+1 = AXL+1 in the new coordinate space (rather than XL+1 as above). In this case, we would be dealing with variables with covariance matrices Σ ∝ AA<sup>T</sup> . We would need weight matrices that are not simply proportional to identity to deal with the covariance structure. This is also the case for in-context learning of linear functions when the input has an anisotropic covariance matrix [\(Zhang et al.,](#page-10-4) [2024;](#page-10-4) [Ahn et al.,](#page-8-9) [2023\)](#page-8-9). Recall in the original setting, we had the sample covariance <sup>E</sup>[XX<sup>T</sup> ] ≡ Σ<sup>X</sup> = σ 2 <sup>0</sup>P and noise Σ<sup>Z</sup> ≡ σ 2 Z I, leading to the estimator, Eq. [\(10\)](#page-4-2): Xˆ = (σ <sup>0</sup>+σ )L P<sup>L</sup> <sup>t</sup>=1 <sup>X</sup>t⟨Xt, <sup>X</sup>˜⟩ . Here, the sample covariance is Σ<sup>Y</sup> ≡ σ 2 0AP A<sup>T</sup> , and the noise V ≡ AZ has covariance Σ<sup>V</sup> ≡ σ 2 ZAA<sup>T</sup> . One can show the generalized solution is Yˆ = Σ<sup>Y</sup> (Σ<sup>Y</sup> + Σ<sup>V</sup> ) <sup>−</sup><sup>1</sup>Y˜ . Thus, in the transformed coordinates, the denoising estimate is

$$\hat{Y} = \frac{1}{(\sigma_0^2 + \sigma_Z^2)L} \sum_{t=1}^L Y_t \langle A^{-1} Y_t, A^{-1} \tilde{Y} \rangle. \quad (\text{A.13})$$

For the relationship of this denoising result in Y to energy models, as discussed in Section [4](#page-6-0) and Subsection [I.1,](#page-20-1) we need a modified energy E(Y1:L, s) = <sup>1</sup> 2γ ∥s∥ <sup>2</sup> − 1 2L P<sup>L</sup> t=1⟨A−<sup>1</sup>Yt, A−<sup>1</sup> s⟩ 2 and a preconditioner proportional to AA<sup>T</sup> .

# I. Additional comments on the mapping from attention to associative memory models

## I.1. Linear attention and traditional Hopfield model

We have considered a trained network with linear attention, relating the query X˜ and the estimate of the target Xˆ, of the form

$$\hat{X} = f(\tilde{X}) := \frac{\gamma}{L} \sum_{t=1}^L X_t \langle X_t, \tilde{X} \rangle \quad (\text{A.14})$$

with γ = 1 σ <sup>0</sup>+σ Z .

With

$$\mathcal{E}(X_{1:L}, s) := \frac{1}{2\gamma} \|s\|^2 - \frac{1}{2L} s^T \left( \sum_{t=1}^L X_t X_t^T \right) s \quad (\text{A.15})$$

gradient descent iteration s(t + 1) = s(t) − γ ∇sE X1:L, s(t) gives us

$$s(t+1) = \frac{\gamma}{L} \sum_t X_t \langle X_t, s(t) \rangle$$

making the one-step iteration our denoising operation.

We will call this energy function the Naive Spherical Hopfield model for the following reason. For random memory patterns X1:L, and the query denoting Ising spins s ∈ {−1, 1} <sup>n</sup>, the so-called Hopfield energy is

$$\mathcal{E}_{\text{Hopfield}}(X_{1:L}, s) := -\frac{1}{2L} s^T \left( \sum_{t=1}^L X_t X_t^T \right) s. \quad (\text{A.16})$$

We could relax the Ising nature of the spins by letting s ∈ R <sup>n</sup>, with a constraint ||s||<sup>2</sup> = n. This is the spherical model [\(Fischer & Hertz,](#page-8-14) [1993\)](#page-8-14) since the spin vector s lives on a sphere. If we minimize this energy the optimal s would be aligned with the dominant eigenvector of the matrix <sup>1</sup> L ( P<sup>L</sup> <sup>t</sup>=1 <sup>X</sup>tX<sup>T</sup> t ) [\(Fischer & Hertz,](#page-8-14) [1993\)](#page-8-14), and the model will not have a retrieval phase (see [Bolle et al.](#page-8-15) ´ [\(2003\)](#page-8-15) for a similar model that does). A soft-constrained variant can also be found in Section 3.3, Model C of [Krotov & Hopfield](#page-9-5) [\(2021\)](#page-9-5).

We could reformulate the optimization problem of minimizing the Hopfield energy, subject to ||s||<sup>2</sup> = R<sup>2</sup> , as

$$\arg \min_{s \in \mathbb{R}^n} \left[ \max_{\lambda} \left\{ -\frac{1}{2L} s^T \left( \sum_{t=1}^L X_t X_t^T \right) s + \lambda (s^T s - R^2) \right\} \right].$$

The s-dependent part of the Lagrangian, with λ replaced by <sup>1</sup> 2γ gives us the energy function in Eq. [A.15](#page-20-2) which we have called the Naive Spherical Hopfield model.

$$\mathcal{E}(X_{1:L}, s) := \frac{1}{2\gamma} \|s\|^2 - \frac{1}{2L} s^T \left( \sum_{t=1}^L X_t X_t^T \right) s = \frac{1}{2} s^T \left[ (\sigma_0^2 + \sigma_Z^2) I_n - \frac{1}{L} \left( \sum_{t=1}^L X_t X_t^T \right) \right] s. \quad (\text{A.17})$$

For L much larger than n, L P<sup>L</sup> <sup>t</sup>=1 <sup>X</sup>tX<sup>T</sup> <sup>t</sup> ≈ σ 2 <sup>0</sup>P, so its eigenvalues are either 0 or are very close to σ 2 . Hence, for large L and σ<sup>Z</sup> > 0, this quadratic function is very likely to be positive definite. One-step gradient descent brings s down to the d-dimensional linear subspace S spanned by the patterns, but repeated gradient descent steps would take s towards zero.

### I.2. Remarks on the softmax attention case (mapping to dense associative memory networks)

Regarding the mapping discussed in the main text, we note that there is a symmetry condition on the weights WKQ, WP V that is necessary for the softmax update to be interpreted as a gradient descent (i.e. a conservative flow). In general, a flow ds/dt = f(s) is conservative if it can be written as the gradient of a potential, i.e. f(s) = −∇sV (s) for some V . For this to hold, the Jacobian of the dynamics J<sup>f</sup> (s) = ∇sf must be symmetric.

The softmax layer studied in the main text is f(s) = WP V X softmax(X<sup>T</sup> WKQs). We will denote z(s) = X<sup>T</sup> WKQ s and g(s) = softmax(z(s)), both in <sup>R</sup> <sup>L</sup>. Then the Jacobian is

$$J(s) = W_{PV} X \frac{\partial g}{\partial s} = W_{PV} X (\text{diag}(g) - gg^T) X^T W_{KQ}. \quad (\text{A.18})$$

Observe that Y = X diag(g) − gg<sup>T</sup> X<sup>T</sup> is symmetric (keeping in mind that g(s) depends on WKQ). The Jacobian symmetry requirement J = J T therefore places the following constraint on feasible WKQ, WP V :

$$W_{PV} Y W_{KQ}^T = W_{KQ} Y W_{PV}^T. \quad (\text{A.19})$$

It is clear that this condition holds for the scaled identity attention weights discussed in the main text. Potentially, it could allow for more general weights that might arise from non-isotropic denoising tasks to be cast as gradient descent updates.

The mapping discussed in the main text involves discrete gradient descent steps, Eq. [\(17\)](#page-7-0). In general, this update rule retains a "residual" term in s(t) if we choose a different descent step size γ ̸= α. Thus, taking K recurrent updates could be viewed as the depthwise propagation of query updates through a K-layer architecture if one were to use tied weights. Analogous residual streams are commonly utilized in more elaborate transformer architectures to help propagate information to downstream attention heads.