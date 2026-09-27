# Score Matching with Missing Data

Josh Givens <sup>1</sup> Song Liu <sup>1</sup> Henry W J Reeve <sup>2</sup>

# Abstract

Score matching is a vital tool for learning the distribution of data with applications across many areas including diffusion processes, energy based modelling, and graphical model estimation. Despite all these applications, little work explores its use when data is incomplete. We address this by adapting score matching (and its major extensions) to work with missing data in a flexible setting where data can be partially missing over any subset of the coordinates. We provide two separate score matching variations for general use, an importance weighting (IW) approach, and a variational approach. We provide finite sample bounds for our IW approach in finite domain settings and show it to have especially strong performance in small sample lower dimensional cases. Complementing this, we show our variational approach to be strongest in more complex highdimensional settings which we demonstrate on graphical model estimation tasks on both real and simulated data.

# 1. Introduction

Over the last decade, score matching has established itself as a powerful tool with downstream use in many areas of machine learning. Examples include: energy based modelling [\(Swersky et al., 2011;](#page-10-0) [Bao et al., 2020;](#page-9-0) [Li et al.,](#page-9-1) [2019b\)](#page-9-1), mode-seeking clustering [\(Sasaki et al., 2014\)](#page-10-1), and perhaps most prominently of all Diffusion processes [\(Song](#page-10-2) [& Ermon, 2019;](#page-10-2) [Song et al., 2021b;](#page-10-3) [Tashiro et al., 2021;](#page-10-4) [Song et al., 2021a;](#page-10-5) [Huang et al., 2021\)](#page-9-2). Score matching aims to learn the score of a distribution which is the gradient of the log of the probability density function (PDF) (s(x) = ∇<sup>x</sup> log p(x)). In contrast to modelling the density directly, the score does not need to integrate to one meaning there is no need to calculate a normalising constant. This allows it to be much more more flexibly modelled than the density itself. Furthermore, the validity of the score matching objective itself requires only very mild assumptions of the family of proposed scores further ensuring this flexibility. Alongside the classical method [\(Hyvärinen, 2005\)](#page-9-3), various adaptations of score matching have arisen to improve performance, decrease computational cost, and extend the approach to a wider range of settings [\(Hyvärinen, 2007;](#page-9-4) [Vincent, 2011;](#page-10-6) [Song et al., 2020;](#page-10-7) [Liu et al., 2022\)](#page-10-8).

In this work, we extend the score matching framework to handle missing data at training time. Specifically, we learn the full score function from partially missing multidimensional input data, a paradigm we term *missing score matching*. Crucially, our approach is compatible with any parameterised score model, enabling its application to both explicit score formulations and more general approaches such as neural networks (NNs). We propose two methods to adapt the original score matching method as well as its popular adaptations, truncated, sliced, and denoising score matching [\(Hyvärinen, 2007;](#page-9-4) [Vincent, 2011;](#page-10-6) [Liu & Wang,](#page-10-9) [2017;](#page-10-9) [Song et al., 2020\)](#page-10-7). These two distinct but closely related methods complement each other allowing for a wide range of problems to be tackled. The first method is a simpler importance weighting (IW) approach which we refer to as marginal IW score matching. For this method we obtain finite sample bounds in the bounded domain setting under certain conditions. We also provide experimental results demonstrating its efficacy in lower dimensional settings and where less data is available. Our second approach is a more computationally sophisticated variational approach which we refer to as marginal variational score matching. We demonstrate the efficacy of this approach in more complex, high dimensional settings by applying it to the problem of graphical model estimation with both real and synthetic datasets.

In section [2](#page-0-0) we discuss relevant works for score matching and related fields. In section [3](#page-1-0) we will introduce our problem more formally including score matching and any notation used. Section [4](#page-2-0) will be used to introduce our methods. Section [5](#page-6-0) will present results on some real and simulated datasets. In Section [6](#page-8-0) we give our conclusion.

# 2. Related Works

While there has been some work which utilises score matching with missing data, these approaches mostly do so

School of Mathematics, University of Bristol, Bristol, UK 2 School of Artificial Intelligence, Nanjing University, China. Correspondence to: Josh Givens <josh.givens@bristol.ac.uk>.

exclusively through the lens of diffusion models. Specifically works such as MissDiff [\(Ouyang et al., 2023\)](#page-10-10) and Ambient Diffusion [\(Daras et al., 2023\)](#page-9-5) require the score function itself to take the form of a neural network (NN) which learns the scores of the fully-observed and corrupted scores simultaneously. This prohibits their use in situations where our model for the score is some explicit parameterisation whose parameters we want to learn as is the case in settings such as energy based modelling [Li et al.](#page-9-6) [\(2023\)](#page-9-6); [Bao et al.](#page-9-0) [\(2020\)](#page-9-0); [Salimans & Ho](#page-10-11) [\(2021\)](#page-10-11) and Gaussian graphical models [\(Lin et al., 2016;](#page-9-7) [Yu et al., 2018\)](#page-10-12). Ambient Diffusion also requires the data to be further artificially corrupted in order to create a pseudo-supervised learning paradigm making both Ambient Diffusion and MissDiff subject to various levels of out of sample learning without specific adjustments for this phenomenon.

Looking more generally at distribution estimation with missing data, multiple works in the field of generative modelling have looked to tackle the problem of providing a generative model for a distribution given corrupted samples from it. Prominent among these are MisGAN [\(Li et al.,](#page-9-8) [2019a\)](#page-9-8), which presents a marginalised GAN framework and MCFlow [\(Richardson et al., 2020\)](#page-10-13) , which presents a EM like normalising flow framework. Neither of these approaches allow for flexible specification of a parametric density estimate however with MCFlow requiring the density to be a normalising flow and MisGAN having no model for the density whatsoever.

To our knowledge, the only approach which seems to adapt score matching to missing data in a parameter preserving manner is presented in [\(Uehara et al., 2020\)](#page-10-14) using an iterative EM-like procedure. However they themselves admit that there is little intuitive understanding of when this approach will converge. Additionally, due to the nature of the score matching objective, the expectation step cannot be directly approximated using Monte Carlo estimation and instead requires fractional importance weighting, a method which employs nested Monte Carlo estimates introducing bias into the training objective.

Parallel to this, some papers have looked to extend score matching to the latent variable setting, an area with much commonality to missing data [\(Vértes & Sahani, 2016;](#page-10-15) [Bao](#page-9-0) [et al., 2020;](#page-9-0) [2021\)](#page-9-9). Latent variable modelling differs in two crucial aspects from missing score matching. Firstly the components which are unobserved (the latent variables) remain constant between samples, and secondly there is not necessarily a notion of a ground truth for the unobserved components in when data is corrupted. Additionally each of these works has limitations; [Vértes & Sahani](#page-10-15) [\(2016\)](#page-10-15) only applies to exponential families, [Bao et al.](#page-9-0) [\(2020\)](#page-9-0) requires a gradient unrolling step in its optimisation which is computationally expensive and can lead to errors in the optimisation procedure (as acknowledged in their follow on work), and [Bao et al.](#page-9-9) [\(2021\)](#page-9-9) is only given for denoising score matching, not for classical or sliced score matching.

### 3. Setting

#### 3.1. Notation

For n ∈ N let [n] := {1, . . . , n}. For a random variable Z we use supp(Z) for the support of Z. For f : <sup>R</sup> <sup>d</sup> → <sup>R</sup> we write ∂jf(x) := ∂f ∂x<sup>j</sup> where x = (x1, . . . , xd) ⊤ and ∇xf(x) := (∂jf(x), . . . , ∂df(x))<sup>⊤</sup> , the gradient of f. For f : R <sup>d</sup> → <sup>R</sup> d take f(x)<sup>j</sup> as the j th component of f(x) and write ∇<sup>x</sup> · f(x) := ∂1f(x)<sup>1</sup> + · · · + ∂df(x)d. Finally for a, b ∈ R d , take a ◦ b to be the Hadamard product.

We now introduce some indexing notation which we will be using for RVs and functions throughout. This will prove useful when identifying the missing non-missing components of our data. Let Z be a random variable taking values in R d . We use Z<sup>j</sup> to refer to the j th component Z and for λ ⊆ [d] take Z<sup>λ</sup> = {Zj}j∈λ. We use negation in indexing to mean the complementing coordinates. More precisely we let −j denote [d] \ {j} and let −λ denote [d] \ λ. We typically use Z (i) to denote an independent copy of Z. For a function f : X → Y and x<sup>λ</sup> ∈ Xλ, x ′ <sup>−</sup><sup>λ</sup> ∈ X<sup>−</sup>λ, we take f(xλ, x ′ −λ ) to be f(z) where

$$z_j := \begin{cases} x_j & \text{if } j \in \lambda \\ x'_j & \text{if } j \in -\lambda \end{cases}$$

.

We will take X to be a RV taking values in X ⊆ R d representing our original dataset and X′ to be a RV representing some generative/variational/importance weighting distribution. i.e., the "artificial distributions" we will utilise in our method. Similarly, we take E, E ′ to be expectations with respect to (w.r.t.) X, X′ respectively.

Throughout we take p to be the pdf of the RV, X, and p<sup>θ</sup> to be a model therein. We let q represent an unnormalised density (i.e. N <sup>−</sup><sup>1</sup> · q = p for some normalising constant N > 0.) We will write marginalisations/conditionings for both true and model densities implicitly with p(xλ) := R X p(x)dx−<sup>λ</sup> and p(xλ|x−λ) being the conditional density of Xλ|X−<sup>λ</sup> = x−<sup>λ</sup> for example.

Now that we have introduced our notation we can move onto the key area of focus for our work, score matching.

#### 3.2. Score Matching

First proposed by [\(Hyvärinen, 2005\)](#page-9-3), score matching aims to learn the gradient of the log-density (score). The advantage of this framework over full density approaches such as maximum likelihood estimation (MLE) is that we are not restricted to parametric models which integrate to 1. This allows us to be much more flexible in how we parameterise in turn making high dimensional distribution modelling more feasible. We now introduce the approach.

Let X be a RV over R <sup>d</sup> with PDF p. We say that q is the unnormalised density of X if N <sup>−</sup><sup>1</sup> · q(x) = p(x) where p is the PDF of X and N is the normalising constant of q. Define the score, of X to be

$$s(\mathbf{x}) := \nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log q(\mathbf{x}).$$

The aim of score matching is to learn s from a collection of IID copies of X which we denote D := {X(i)} n i=1. Following [Hyvärinen](#page-9-3) [\(2005\)](#page-9-3), we introduce a generic parameterised proposal score s<sup>θ</sup> for θ ∈ Θ ⊆ <sup>R</sup> p and aim to minimise the *Fisher Divergence* between the true distribution and our proposal distribution which is given by

$$F(\theta) := \mathbb{E}[\|\mathbf{s}(X) - \mathbf{s}_\theta(X)\|^2].$$

The key result from [Hyvärinen](#page-9-3) [\(2005\)](#page-9-3) which enables us to practically implement score matching is that under certain (fairly minimal) regularity conditions, which we provide in Appendix [D.1,](#page-34-0) we have

$$L(\theta) := \mathbb{E} \left[ 2\nabla_X \cdot \mathbf{s}_\theta(X) + \|\mathbf{s}_\theta(X)\|^2 \right] = F(\theta) - C \quad (1)$$

where here and throughout, we take C to represent any constant which does not depend upon θ. Crucially, L(θ) is now an expectation of observable random variables. Hence we can now approximate this with our data and take ˆθ as

$$\hat{\theta} := \operatorname{argmin}_{\theta} \frac{1}{n} \sum_{i=1}^n \left[ 2\nabla_{X^{(i)}} \cdot \mathbf{s}_{\theta}(X^{(i)}) + \|\mathbf{s}_{\theta}(X^{(i)})\|^2 \right].$$

### TRUNCATED SCORE MATCHING

A limitation of standard score matching is that it requires lim<sup>x</sup>i→∞ p(x) = 0 for all x<sup>i</sup> ∈ <sup>R</sup>. Thus it cannot be used for many distributions with compact support if the density does not converge to zero at the (topological) boundary. Initial work to adapt score matching to truncated distributions was presented in [\(Hyvärinen, 2007\)](#page-9-4) for distributions on [0, ∞) then further expanded in [\(Liu et al., 2022;](#page-10-8) [Yu](#page-10-16) [et al., 2022\)](#page-10-16) to general compact spaces X . For our compact space X ⊆ R <sup>d</sup> we use ∂X to denote the (topological) boundary. We now minimise some weighted version of the Fisher divergence whose weights go to zero at the boundary. Specifically let g : X → R be a function satisfying limx→x′ g(x)<sup>j</sup> = 0 for any x ′ ∈ ∂X , j ∈ [d]. Our objective is then

$$F_T(\theta) := \mathbb{E} \left[ \left\| \mathbf{g}^{\frac{1}{2}}(X) \circ (\mathbf{s}_\theta(X) - \mathbf{s}(X)) \right\|^2 \right].$$

integration by parts) giving us that under certain regularity conditions on g, s, and X ,

$$\begin{aligned} L_T(\theta) &:= \mathbb{E} \left[ \sum_{j \in d} \mathbf{g}(X)_j \left( 2\partial_j \mathbf{s}_\theta(X)_j + \mathbf{s}_\theta(X)_j^2 \right) \right] \\ &+ \mathbb{E} \left[ \sum_{j \in d} \partial_j \mathbf{g}(X)_j \mathbf{s}_\theta(X)_j \right] = F_T(\theta) - C. \end{aligned}$$

This can again be approximated via data using standard Monte Carlo approximation. Full details on the conditions required for this approach alongside the proof can be found in [\(Liu et al., 2022\)](#page-10-8). Two other key extensions of score matching are sliced score matching [\(Song et al., 2020\)](#page-10-7) and denoising score matching [\(Vincent, 2011\)](#page-10-6). We introduce these extensions in Appendix [D](#page-34-1) with our corresponding adaptations to missing data given in Appendix [A.1.](#page-11-0) Now, we give our missing data scenario.

### 3.3. Missing Data Scenario

Instead of observing samples from X we assume that we observe samples from the corrupted version of the RV given by X˜. To define X˜ we introduce a mask RV M over {0, 1} d and then define X˜ by

$$\tilde{X}_j = \begin{cases} X_j & \text{if } M_j = 1 \\ \emptyset & \text{if } M_j = 0 \end{cases}$$

where X˜ <sup>j</sup> = <sup>∅</sup> represents that coordinate being missing. We will be focussing on the missing completely at random scenario where M ⊥ X. However, we do provide an extension to missing not at random data in Appendix [A.1.4.](#page-13-0) We introduce the RV Λ on P([d]) defined by Λ := {i ∈ [d]|M<sup>i</sup> = 1} so that Λ gives the non-corrupted coordinates of X˜ and take λ to be a sample of Λ. Crucially given samples from X˜, we also have samples from XΛ.

Our aim is to adapt the score matching objective to estimate the full score s by a parameterised score s<sup>θ</sup> using samples from the corrupted data D˜ := {X˜(i)} n <sup>i</sup>=1 ≡ {X (i) Λ<sup>i</sup> } n <sup>i</sup>=1.

# 4. Marginal Score Matching

To motivate our approach we look at how we might use MLE in the case where the normalising constant and conditional normalising constants were calculable. For p<sup>θ</sup> our parametric model of the density, we would choose ˆθ to be

$$\hat{\theta} := \operatorname{argmax}_{\theta} \sum_{i=1}^n \log \tilde{p}_{\theta}(\tilde{X}^{(i)})$$

this is actually equivalent to maximising

$$\sum_{i=1}^n \log p_{\theta; \Lambda_i}(X_{\Lambda_i}^{(i)}), \text{ where } p_{\theta; \lambda}(\mathbf{x}_\lambda) := \int_{\mathcal{X}_{-\lambda}} p_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda}.$$

For notational simplicity we will thus reframe our problem as working with marginal samples {X (i) Λ<sup>i</sup> } n <sup>i</sup>=1.

#### 4.1. Marginal Score Matching

Our approach is to directly alter the score matching objective similarly. Just as densities have associated *marginal densities* so do scores have associated *marginal scores*.

Definition 4.1 (Marginal Score function). Let s be a score function with s(x) = ∇<sup>x</sup> log q(x) for q an unnormalised PDF. Then the associated *marginal score* function is

$$s_\lambda(\mathbf{x}_\lambda) := \nabla_{\mathbf{x}_\lambda} \log \int_{\mathbb{R}^{d-|\Lambda|}} q(\mathbf{x}) d\mathbf{x}_\lambda. \quad (2)$$

This definition of marginal scores restricts s to a genuine score function. For this reason we will also want s<sup>θ</sup> to always be a genuine score function or at least to have an anti-derivative. The simplest way to achieve this is to work with q<sup>θ</sup> : X → (0, ∞) as our baseline and define sθ(x) := ∇<sup>x</sup> log qθ(x). We will also take pθ(x) := R X qθ(x)dx <sup>−</sup><sup>1</sup> qθ(x) which we assume to be unknown.

With this notion of a marginal score we can define our marginal Fisher divergence to be

$$F_M(\theta) := \mathbb{E}[\|s_\Lambda(X_\Lambda) - s_{\Lambda;\theta}(X_\Lambda)\|^2] \quad (3)$$

where sλ;<sup>θ</sup> is defined analogously to sλ. As with normal score matching can relate this objective to one involving no terms of sλ. We first need the following assumptions.

Assumption 4.2. For any θ > 0, λ ∈ supp(Λ):

- (a) p<sup>θ</sup> is well defined, i.e. R X qθ(x)dx < ∞;
- (b) <sup>E</sup>[∥sλ(Xλ)∥ 2 ], <sup>E</sup>[∥sλ;<sup>θ</sup>(Xλ)∥ 2 ] < ∞;
- (c) pλ(x) is differentiable and qλ;<sup>θ</sup> is twice differentiable;
- (d) pλ(xλ)sλ;<sup>θ</sup>(xλ)−→0 as ∥x∥−→∞;
- (e) pλ;<sup>θ</sup>(Xλ) = pλ(Xλ) almost surely (a.s.) for all λ ∈ supp(Λ), implies that pθ(X) = p(X) a.s..

Assumption (a) ensures that our proposal unnormalised density is always a genuine unnormalised density. Assumptions (b)-(d) are similar to the standard assumptions given for standard score matching. Assumption (e) is an identifiability assumption which is required to be feasibly able to learn the true data distribution from our corrupted data.

Proposition 4.3. *Given Assumptions [4.2\(](#page-3-0)a)-(d) hold*

$$\begin{aligned} L_M(\theta) &:= \mathbb{E}[2\nabla_{X_\Lambda} \cdot \mathbf{s}_{\Lambda;\theta}(X_\Lambda) + \|\mathbf{s}_{\Lambda;\theta}(X_\Lambda)\|^2] \\ &= F_M(\theta) - C. \end{aligned} \quad (4)$$

*If (e) also holds and there exists some* θ ∗ *such that* sθ <sup>∗</sup> (X) = s(X) *a.s.. Then if* ˜θ *is a minimiser of* LM(θ) *we have that* qθ˜(X) = N p(X) *a.s. for some constant* N*, i.e. the minimiser is the true unnormalised density.*

Through this result we have shown, much like with standard score matching, that under certain regularity conditions our objective is uniquely minimised by the true unnormalised density. We then approximate this objective by

$$\hat{L}_{\text{M};n}(\theta) := \frac{1}{n} \sum_{i=1}^n \nabla_{X_{\Lambda_i}^{(i)}} \cdot \boldsymbol{s}_{\Lambda_i}(\theta(X_{\Lambda_i}^{(i)})) + \|\boldsymbol{s}_{\Lambda_i}(\theta(X_{\Lambda_i}^{(i)}))\|^2$$

and choose <sup>ˆ</sup><sup>θ</sup> = argmin<sup>θ</sup> <sup>L</sup>ˆM;n(θ).

Unfortunately this approach in its current state is practically infeasible as the integrals involved in deriving the marginal scores for any non-trivial problem will be intractable. Hence, we must devise a way to estimate the marginal scores without having to compute the integrals. We tackle this issue in Section [4.2,](#page-3-1) but first we provide a similar result for the case of truncated score matching.

#### 4.1.1. TRUNCATED SCORE MATCHING

Truncated score matching can be adapted similarly to standard score matching by simply having marginal weighting functions g<sup>λ</sup> : X<sup>λ</sup> → [0, ∞) for each subset λ ∈ supp(Λ) and taking the marginal truncated Fisher divergence to be

$$F_{\text{TM}}(\theta) := \mathbb{E} \left[ \left\| \mathbf{g}_{\Lambda}(X_{\Lambda})^{\frac{1}{2}} \circ (\mathbf{s}_{\Lambda}(X_{\Lambda}) - \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})) \right\|^2 \right].$$

using integration by parts gives the following equivalence

$$\begin{aligned} L_{\text{TM}}(\theta) &:= \mathbb{E} \left[ \sum_{j \in \Lambda} \mathbf{g}_{\Lambda}(X_{\Lambda})_j (\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})_j^2 + 2\partial_j \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})_j) \right] \\ &\quad + \mathbb{E} \left[ \sum_{j \in \Lambda} 2\partial_j \mathbf{g}_{\Lambda}(X_{\Lambda})_j \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})_j \right] \\ &= F_{\text{TM}}(\theta) - C. \end{aligned} \quad (5)$$

Proof given in Appendix [C.1.1.](#page-21-0) We then take LˆTM;<sup>n</sup> as the Monte-Carlo estimate of LTM. We also construct similar objectives from sliced and denoising score matching as well as a similar result for missing not at random data in Appendix [A.1.](#page-11-0) We now move to the task of estimating the marginal scores in these objectives.

#### 4.2. Importance Weighting

Our first proposal is an importance weighting approach. Let p ′ be a density over R<sup>d</sup>−|λ<sup>|</sup> which we can both evaluate and sample from then

$$\int_{\mathbb{R}^{d-|\lambda|}} q_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} = \mathbb{E}_{X'_\lambda \sim p'} \left[ \frac{q_\theta(\mathbf{x}_\lambda, X'_{-\lambda})}{p'(X'_{-\lambda})} \right]. \quad (6)$$

Algorithm 1 Marginal IW Score Matching

Input: {X (i) Λ<sup>i</sup> }i∈[n] , qθ, p ′ , θ0, r ∈ <sup>N</sup>. Set θ = θ0. repeat for i=1 to n do Sample {X′(i,k)}k∈<sup>r</sup> from p ′ (.|X (i) Λ<sup>i</sup> ). Use X (i) Λ<sup>i</sup> , {X ′(i,k) −Λ<sup>i</sup> }k∈[r] to get Monte-Carlo estimates, sˆΛi,r;<sup>θ</sup>(X (i) Λ<sup>i</sup> ), of the marginal scores by [\(7\)](#page-4-0). end for Use sˆΛi,r;<sup>θ</sup>(X (i) Λ<sup>i</sup> ) to obtain <sup>L</sup>ˆM/TM;n,r(θ) by [\(4\)](#page-3-2). Compute ∇θLˆM/TM;n,r(θ) and update the value of <sup>θ</sup>. until Maximum iteration reached.

This allows us to define our *marginal score estimate*.

Definition 4.4 (Marginal Score Estimate). For a given λ ∈ supp(Λ) , x<sup>λ</sup> ∈ Xλ, score model, sθ, and r ∈ <sup>N</sup> we take our estimate of sθ;λ,r(xλ) to be

$$\hat{s}_{\lambda,r;\theta} := \nabla_{\mathbf{x}_\lambda} \log \left( \frac{1}{r} \sum_{k=1}^r \frac{q_\theta(\mathbf{x}_\lambda, X_{-\lambda}^{(k)})}{p'(X_{-\lambda}^{(k)})} \right) \quad (7)$$

where X ′(1) −λ , . . . , X′(r) −λ are IID copies of X′ <sup>−</sup><sup>λ</sup> ∼ p ′ .

### 4.2.1. IW SAMPLE OBJECTIVE

We can now plug these marginal score estimates into our sample objective for either normal or truncated score matching. We use M/TM to denote analogous definitions and results for both marginal and truncated marginal score matching. Let {X (i) Λ<sup>i</sup> } n <sup>i</sup>=1 be our samples from XΛ. We then take our IW sample objective to be as <sup>L</sup>ˆM/TM;n(θ) but with sˆΛi,r;<sup>θ</sup>(X (i) Λ<sup>i</sup> ) replacing sΛi;<sup>θ</sup>(X (i) Λ<sup>i</sup> ). The full objective is given in Appendix [E.1.1](#page-36-0) We refer to this sample objective as <sup>L</sup>ˆM/TM;n,r(θ) and take our estimate to be

$$\hat{\theta} := \operatorname{argmin}_{\theta} \hat{L}_{\text{M/TM};n,r}(\theta).$$

Algorithm [1](#page-4-1) gives our high level estimation algorithm.

*Remark* 4.5*.* Algorithm [1](#page-4-1) can directly be applied to both sliced and denoised score matching by replacing equation [\(4\)](#page-3-2) by equations [\(13\)](#page-11-1) and [\(15\)](#page-12-0) respectively.

### 4.2.2. FINITE SAMPLE BOUNDS

A benefit of truncated score matching is that it allows us to work on distributions with densities bounded below which enables us to give finite sample bounds for the error of our estimated score w.r.t. our marginal objective. We briefly present these now with more detail given in Appendix [A.2.](#page-14-0)

θn,r ∈ Θ *be the minimiser of* LˆTM;n,r(θ)*. If* Θ ⊆ <sup>R</sup> p *with* diam(Θ) = A *then for sufficiently large* n, r

$$\mathbb{P}\left(F_{\text{TM}}(\theta_{n,r}) \geq \beta_1 \sqrt{\frac{p \log(dnr A/\delta)}{\min\{r, n\}}}\right) < \delta.$$

Note that r is the number of importance weighting samples for each data sample and therefore is something we can choose ourself. This means that with this approach we can achieve approximately √ n convergence rates. A downside however is that to achieve this we need r to be of order at least n which would lead to an O(n 2 ) computational cost. In practice we find relatively strong performance choosing r small. Setting it at r = 10 in our experiments.

*Remark* 4.7*.* The error presented is measured with respect to our Marginal Fisher Divergence, rather than the full Fisher Divergence (which would be the preferred accuracy metric). Relating these two quantities requires connecting the fully observed distribution to its marginals, a task that depends on the specific form of the distribution. Investigating the assumptions and conditions under which this connection can be made offers an interesting and valuable direction for future research.

#### 4.3. Gradient First Approach

A key limitation with an IW approach is that it will struggle in higher dimensional scenarios. Additionally the importance weighting is embedded inside other functions which leads to the same nested expectation issue as the EM approach of [Uehara et al.](#page-10-14) [\(2020\)](#page-10-14), causing bias in our estimator. As an alternative to this we build upon a variational approach initially discussed in the context of latent variable models in [Vértes & Sahani](#page-10-15) [\(2016\)](#page-10-15); [Bao et al.](#page-9-0) [\(2020;](#page-9-0) [2021\)](#page-9-9).

The core idea is to start with L<sup>M</sup> as before and then take gradients w.r.t. our parameters before then writing our objective in terms of expectations over X−λ|λ;<sup>θ</sup>. As we don't then need to take gradients of these expectations w.r.t. θ, we can estimate them with any black-box method we desire, opening the door for variational approximation to be used. This approach has been explored for exponential family distributions [\(Vértes & Sahani, 2016\)](#page-10-15) and for denoising score matching [\(Bao et al., 2021\)](#page-9-9) however we provide the most general version of this result which can be applied to any of the score matching methods and any model class. We first introduce the following key Lemma.

Lemma 4.8. *Fix* λ ⊆ [d], x<sup>λ</sup> ∈ Xλ*. We have that for any function* h<sup>θ</sup> : X → <sup>R</sup>*.*

$$s_{\theta;\lambda}(\mathbf{x}_\lambda) = \mathbb{E}'[s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_\lambda] \quad (8)$$

$$\nabla \mathbb{E}'[h_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] = \mathbb{E}'[\nabla h_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \quad (9)$$

$$+ \text{Cov}'(s_\theta(\mathbf{x}_\lambda, X'_{-\lambda}), h_\theta(\mathbf{x}_\lambda, X'_{-\lambda}))$$

*where* ∇ *represents the gradient w.r.t. either* x<sup>λ</sup> *or* θ *and here* E ′ , Cov′ *are w.r.t.* X′ −λ |X<sup>λ</sup> = x<sup>λ</sup> ∼ pθ(.|xλ)*.*

This results allows us to obtain our alternative objective.

Corollary 4.9. *Let* L<sup>M</sup> *be defined as in* [\(4\)](#page-3-2)*. We have that*

$$\nabla_{\theta} L_M(\theta) = \mathbb{E} \left[ 2 \sum_{j \in \Lambda} \left( \Psi_{\Lambda}(\mathbf{s}_{\theta}(\cdot)_j^2 + \partial_j \mathbf{s}_{\theta}(\cdot)_j) - \mathbb{E}'[\mathbf{s}_{\theta}(X_{\Lambda}, X'_{-\Lambda})_j] \Psi_{\Lambda}(\mathbf{s}_{\theta}(\cdot)_j) \right) \right]. \quad (10)$$

*where for any function* h<sup>θ</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup>*,* λ ⊆ [d]*,*

$$\begin{aligned} \Psi_\Lambda(h_\theta) &= \mathbb{E}'[\nabla_\theta h_\theta(X_\Lambda, X'_{-\Lambda})] \\ &\quad + \text{Cov}'(\nabla_\theta \log q_\theta(X_\Lambda, X'_{-\Lambda}), h_\theta(X_\Lambda, X'_{-\Lambda})) \end{aligned}$$

*and* E ′ , Cov′ *are w.r.t.* X′ −Λ |X<sup>Λ</sup> ∼ pθ(.|XΛ) *with* <sup>E</sup> *being w.r.t.* X<sup>Λ</sup> ∼ p*.*

Proofs for both results are given in Appendix [C.3.](#page-32-0)

Crucially E ′ , Cov′ can be estimated freely. This allows us to use variational inference to approximate pθ(x−λ|xλ) and in turn the expectations and covariances in [\(10\)](#page-5-0).

*Remark* 4.10*.* We provide additional implementation details for computing this gradient estimate in Appendix [A.5.](#page-17-0) We also discuss equivalences between this objective and our marginal IW objective in [A.3.](#page-15-1)

We explore estimation of E ′ , Cov′ in Section [4.3.2](#page-5-1) but first we provide a similar result for truncated score matching.

#### 4.3.1. TRUNCATED SCORE MATCHING

We define a similar objective for truncated score matching.

Corollary 4.11. *With* LTM *defined as in* [\(5\)](#page-3-3) *we have that*

$$\begin{aligned}\nabla_{\theta} L_{\text{TM}}(\theta) &= \mathbb{E} \left[ 2 \sum_{j \in \Lambda} \left( \mathbf{g}_{\Lambda}(X_{\Lambda})_j \left\{ \Psi_{\Lambda}(\mathbf{s}_{\theta}(\cdot)_j^2 + \partial_j \mathbf{s}_{\theta}(\cdot)_j) \right. \right. \right. \\ &\quad \left. \left. - \mathbb{E}'[\mathbf{s}_{\theta}(X_{\Lambda}, X'_{-\Lambda})_j] \Psi_{\Lambda}(\mathbf{s}_{\theta}(\cdot)_j) \right\} \right. \\ &\quad \left. \left. + \partial_j \mathbf{g}_{\Lambda}(X_{\Lambda})_j \Psi_{\Lambda}(\mathbf{s}_{\theta}(\cdot)_j) \right) \right] \quad (11)\end{aligned}$$

*with* Ψ<sup>Λ</sup> *and* <sup>E</sup> ′ *defined as in Corollary [4.9.](#page-5-2)*

Proof given in Appendix [C.1.1.](#page-21-0) Similar results for sliced and denoising score matching are given in Appendix [A.1.](#page-11-0)

# 4.3.2. VARIATIONAL APPROXIMATION

We can now use variational approximation to estimate the expectations and covariances in Corollaries [4.9](#page-5-2) & [4.11.](#page-5-3) Specifically, let p ′ ϕ (x−λ|xλ) be some generative conditional distribution dependent upon parameter ϕ. We want to train p ′ ϕ to approximate pθ. We may write ϕ(θ) to highlight the dependence on our current parameter estimate however we will omit this for brevities sake. The following proposition from [Bao et al.](#page-9-0) [\(2020\)](#page-9-0) shows us how to train ϕ.

Proposition 4.12 [\(Bao et al.](#page-9-0) [\(2020\)](#page-9-0)). *For distributions* p ′ , p *let* F(p ′ |p) *and* KL(p ′ |p) *be the Fisher and KL divergences between* p ′ *and* p*. We have that for any* λ ⊆ [d], x<sup>λ</sup> ∈ X<sup>λ</sup>

$$\begin{aligned} \text{KL}(p'_\phi(\cdot | \mathbf{x}_\lambda) | p_\theta(\cdot | \mathbf{x}_\lambda)) &= \mathbb{E}' \left[ \log \left( \frac{p'_\phi(X'_{-\lambda} | \mathbf{x}_\lambda)}{q_\theta(\mathbf{x}_\lambda, X'_{-\lambda})} \right) \right] + B \\ F(p'_\phi(\cdot | \mathbf{x}_\lambda) | p_\theta(\cdot | \mathbf{x}_\lambda)) &= \mathbb{E}' \left[ \left\| \nabla_{X'_{-\lambda}} \log (p'_\phi(X'_{-\lambda} | \mathbf{x}_\lambda)) \right. \right. \\ &\quad \left. \left. - s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_{-\lambda} \right\|^2 \right] \end{aligned}$$

*where expectations are w.r.t.* X′ <sup>−</sup><sup>Λ</sup> ∼ p ′ ϕ (.|xλ) *and* B *is a constant not depending upon* ϕ *(but will depend on* θ*.) In other words we can fit to the conditional density* pθ(.|xλ) *given only the unconditional unnormalised density* qθ(xλ, .) *or full score* sθ(xλ, .)*.*

This allows us to train p ′ ϕ (.|xλ) to approximate the conditional density, pθ(.|xλ). In our case we won't be learning this variational model for a fixed x<sup>λ</sup> or even fixed observed coordinates λ. Hence we take our objective to be one of

$$J_{KL}(\phi, \theta) := \mathbb{E} \left[ \log \left( \frac{p'_{\phi}(X'_{-\Lambda}|X_{\Lambda})}{q_{\theta}(X_{\Lambda}, X'_{-\Lambda})} \right) \right]$$

$$J_F(\phi, \theta) := \mathbb{E} \left[ \left\| \nabla_{X'_{-\Lambda}} \log \left( \frac{p'_{\phi}(X'_{-\Lambda}|X_{\Lambda})}{q_{\theta}(X_{\Lambda}, X'_{-\Lambda})} \right) \right\| \right]^2$$

with (XΛ, X′ −Λ ) ∼ p ′ ϕ (X′ −Λ |XΛ)p(XΛ). We then take and Jˆ <sup>F</sup> , Jˆ <sup>F</sup> to be the Monte-Carlo approximations with samples (XΛ, X′ −Λ ) from the same distribution.

*Remark* 4.13*.* J<sup>F</sup> has the advantage of not needing to know the normalising constant of q ′ <sup>ϕ</sup> = N<sup>ϕ</sup> · p ′ ϕ either.

*Remark* 4.14*.* As ϕ depends upon θ, we need to update it each time we update θ. In practice we find taking 10 gradient steps of ϕ for each gradient step of θ to work well.

With this, we define <sup>∇</sup>dθLM/TM(θ) to be the Monte-Carlo estimate of [\(10\)](#page-5-0)/[\(11\)](#page-5-4) with samples {(X (i) Λ<sup>i</sup> , X′(i,k) −Λ<sup>i</sup> )}(i,k)∈[n]×[r] where X (i) Λ are our original corrupted data samples from p and X ′(i,k) −Λ are our variational samples from p ′ ϕ (.|X (i) Λ<sup>i</sup> ). We can now state our full variational approach which is given in Algorithm [2.](#page-6-1)

*Remark* 4.15*.* Algorithm [2](#page-6-1) can directly be applied to both sliced and denoised score matching by replacing equation [\(10\)](#page-5-0) by equations [\(14\)](#page-11-3) and [\(16\)](#page-12-1) respectively.

Algorithm 2 Marginal Variational Score Matching

Input: {X (i) Λ<sup>i</sup> }i∈[n] , qθ, p ′ ϕ , θ0, ϕ0, L ∈ <sup>N</sup>, r ∈ <sup>N</sup>. Set θ = θ0, ϕ = ϕ0. repeat for l = 1 to L do For i ∈ [n] sample X′(i) from p ′ ϕ (.|X (i) Λ<sup>i</sup> ). Use {(X (i) Λ<sup>i</sup> , X′(i) −Λ<sup>i</sup> )}i∈[n] to get Monte-Carlo approximates of <sup>J</sup>KL/F (ϕ, θ) given by <sup>J</sup> <sup>ˆ</sup> KL/F (ϕ, θ). Compute ∇ϕ<sup>J</sup> <sup>ˆ</sup> KL/F (ϕ, θ) and update <sup>ϕ</sup>. end for For i ∈ [n] sample {X′(i,k)}k∈<sup>r</sup> from p ′ ϕ (.|X (i) Λ<sup>i</sup> ). Use {(X (i) Λ<sup>i</sup> , X′(i,k) −Λ<sup>i</sup> )}(i,k)∈[n]×[r] to get our Monte-Carlo estimate, <sup>∇</sup>dθLM/TM(θ) using equation [\(10\)](#page-5-0)/[\(11\)](#page-5-4). Use this gradient estimate to update θ. until Maximum iterations reached.

# 5. Results

Here we go through simulated results comparing our IW approach (Marg-IW) in Algorithm [1](#page-4-1) and our variational approach (Marg-Var) in Algorithm [2](#page-6-1) to the EM approach of [Uehara et al.](#page-10-14) [\(2020\)](#page-10-14). We also compare to a naive marginalisation approach involving zeroing out the missing dimensions and only taking the observed output dimensions of the score, which we call Zeroed Score Matching. This approach is the natural adaptation of MissDiff from [Ouyang](#page-10-10) [et al.](#page-10-10) [\(2023\)](#page-10-10) away from NN to explicitly parameterised models. We describe Zeroed Score Matching and its relation to MissDiff in Appendix [D.2.](#page-34-2) In our experiments, we highlight a unique strength of our methods by applying them to explicitly parameterised score models. We could however, equally apply them to more complex, noninterpretable models such as NNs. More implementation details can be found in Appendix [E.3.](#page-37-0) [<sup>1</sup>](#page-6-2)

#### 5.1. Parameter Estimation

# 5.1.1. TRUNCATED GAUSSIAN MODEL

In this experiment a 10-dim normal distribution is set up with fixed mean and random covariance before being truncated on the first 3 dimensions. 1000 samples are taken and corrupted independently on each coordinate with probability 0.2. For each of our methods a Gaussian score is fit and the Fisher divergence between this score and the truth computed. This is repeated 200 times with the mean Fisher divergence alongside 95% C.I.s then presented in figure [1.](#page-6-3) More details in Appendix [E.3.1.](#page-37-1) Marg-IW and EM perform best with Marg-Var approaching asymptotically. We see the effect of Zeroed's naive marginalisation as it does

![](_page_6_Figure_2.jpeg)

Figure 1: Average Fisher Divergence for Gaussian score estimates alongside 95% C.I.s Lower is better.

not converge, a phenomenon we discuss more in Appendix [D.2.](#page-34-2) In Appendix [B.1.1](#page-18-0) we present the average mean and precision estimation error for this experiment. In Appendix [B.1.2](#page-18-1) we present the untruncated results and illustrate how the naive marginalisation poorly models strong relationship between dimensions 1 and 10.

### 5.1.2. NON-GAUSSIAN MODEL

For this experiment we tested our parameter estimation for a an ICA inspired unnormalisable model of the form

$$p(\mathbf{x}) \propto \exp \sum_{i,j} \theta_{i,j}^* x_i^2 x_j^2.$$

Here we parameterise our model identically with the aim of estimating θ ∗ . We vary the dimension of X and plot the estimation error with a sample size of 1,000 and each coordinate missing independently with probability 0.5. The results are presented in Figure [2.](#page-6-4)

![](_page_6_Figure_12.jpeg)

Figure 2: Average Fisher Divergence for Gaussian score estimates alongside 95% C.I.s. Lower is better.

Our variational method (Marg-Var) consistently yields the lowest error. Moreover, as the dimensionality increases, the performance gap between Marg-Var and the other methods widens. This supports the notion that our approach is

<sup>1</sup>All code and data for the experiments presented can also be found at [https://github.com/joshgivens/](https://github.com/joshgivens/ScoreMatchingwithMissingData) [ScoreMatchingwithMissingData](https://github.com/joshgivens/ScoreMatchingwithMissingData)

more accurately able to capture complex marginalisations than the competing approaches which fail as the dimension grows. We note that all other methods perform comparably with the performance of EM and Marg-IW being indistinguishable, a pattern we observe throughout our experiments. This similarity is unsurprising both approaches use self normalised importance weighting to approximate conditional expectations with respect to our current score estimate while being broadly motivated by fitting to the marginal scores. Nevertheless, the precise mechanism for this similarity remains unclear and warrants further exploration. Additional experiments exploring the effect of sample size and missingness probability on estimation accuracy are given in appendix [B.1.3.](#page-19-0)

#### 5.2. Gaussian Graphical Model Estimation

Gaussian graphical models (GGM) are a popular way of modelling dependence between dimensions of data. Let us assume that the underlying data follows a Gaussian distribution with mean µ ∈ R d and precision P ∈ R d×d . In this setting, a Bayesian network (BN) can represent the dependencies between the dimensions of X with the (undirected) edges of the BN exactly being the non-zero off-diagonal entries of the precision, P. Hence estimating the precision matrix P gives the BN. Score matching has been shown to be an effective way of achieving this with L1-regularisation on the off-diagonal of P to push terms to 0 [\(Lin et al., 2016;](#page-9-7) [Yu et al., 2018\)](#page-10-12). Decreasing the level of L1-regularisation then gives a range of classifiers with increasing True and False positive rates (TPR/FPR) as the level of regularisation decreases. Score matching can also be applied to truncated GGMs where we aim to learn the original BN but only observe the samples inside some truncated region.

We apply our methods to learn GGMs and truncated GGMs with missing data as well. We use varying levels of L1 regularisation on our objective via proximal stochastic gradient descent in our optimisation [\(Beck, 2017\)](#page-9-10).

5.2.1. STAR SHAPED TRUNCATED GRAPHICAL MODEL Here we create a star shaped GGM in which one node has a high probability of being connected with each other node independently and all other connections have probability 0. We truncate the data along a random hyperplane such that 20% of the distribution lies outside of the truncation boundary. Each coordinate is then MCAR independently with the same probability. We run multiple experiments with this probability ranging from 0.2 to 0.9 and present the results in figure [3.](#page-7-0) As we can see here Marg-Var performs best with all other approaches performing comparably. For illustrative purposes, we plot individual ROC curves from this experiment in Appendix [B.2.3.](#page-20-0)

![](_page_7_Figure_2.jpeg)

Figure 3: Mean AUC of star graph edge detection with varying missingness alongside 95% C.I.s. Higher is better.

5.2.2. UNSTRUCTURED DENSE GRAPHICAL MODEL Here we create a GGM by making each edge occur independently with probability 0.5. The rest of the experiment was constructed as before. Results are given in Figure [4.](#page-7-1) Again we can see that our variational approach performs

![](_page_7_Figure_7.jpeg)

Figure 4: Mean AUC of dense graph edge detection with varying missingness alongside 95% C.I.s. Higher is better.

best though not as clearly as in the previous example. We believe this to be because for more unstructured problems, naive marginalisation performs moderately well.

### 5.2.3. INCREASING NUMBER OF STARS

To explore this further, we construct and experiment where we vary the number of star centres (high degree nodes) while keeping the edge density constant. We present the results in Figure [5.](#page-8-1) As we increase the number of star centres, Marg-Var no longer noticeably outperforms the other approaches. This is because as the number of stars increases, (i.e. the structure of the graph decreases) naive marginalisation is a better approximation. This is illustrated on the marginal precisions themselves in Appendix [B.2.1.](#page-19-1)

# 5.2.4. S&P 100

Here we took closing price data over 5 years for the 100 stocks in the S&P 100 with each stock being a dimension

![](_page_8_Figure_1.jpeg)

Figure 5: Mean AUC with 95% C.I.s for edge detection as number of star centres in graph varies. Higher is better.

and each day being a sample. Gaussian graphical models with various levels of connectivity were then constructed using standard score matching on the fully observed data. The data was then artificially corrupted and each missing score matching approach applied. The AUC was then calculated for each method taking the GGM from fully observed score matching as the ground truth. More details given in appendix [E.3.3.](#page-37-2) The results are shown in figure [6.](#page-8-2)

![](_page_8_Figure_8.jpeg)

Figure 6: Mean AUC of various methods when compared to non-corrupted score matching with 95% confidence intervals on stocks in S&P 100. Higher is better.

As we can see Marg-Var clearly out performs all the other approaches which appear to perform equivalently.

### 5.2.5. YEAST DATA

Here data first introduced in [Brem & Kruglyak](#page-9-11) [\(2005\)](#page-9-11) is used consisting of readings of expression for 7086 genes/ORFs across 262 yeast segregants. Each gene represents a dimension with each segregant representing a sample. We subset the data to take the 106 genes present in at least 95% of the samples with the aim of learning the relationship between them. The same approach as the previous section is applied with the results shown in figure [7.](#page-8-3) Again Marg-Var clearly outperforms the other approaches which all perform comparably.

![](_page_8_Figure_2.jpeg)

Figure 7: Mean AUC of various methods when compared to non-corrupted score matching with 95% confidence intervals on genetic yeast data. Higher is better.

# 6. Conclusion

To conclude, score matching is a versatile method whose applications at the heart of modern machine learning problems. In this work we have tackled the problem of adapting score matching to partially missing data. We have presented two separate but related approaches to this method, one using importance weighting and another using variational approximation. We have also provided extensions of these methods to truncated score matching, sliced and denoising score matching. For truncated score matching with our IW approach we have provided finite sample bounds on the accuracy of the estimated score in terms of the marginal truncated Fisher divergence.

We have provided several simulated and real world experiments demonstrating our methods' efficacy for both parameter estimation and downstream GGM edge detection. We have shown the benefits and drawbacks of each approach with IW performing best in lower dimensional settings with less data and the variational approach performing best in more complicated higher dimensional settings.

There is, however still much work to be done in this area. From a theoretical perspective, while we have finite sample bound on the error of our loss, marginal nature of the loss makes it unclear exactly how this translates to parameter or general score model accuracy, leaving room for further theoretical exploration. From an implementation perspective, variational inference in the presence of missing data requires accounting for the randomness of "latent" and "observed" variables. The standard variational inference technique can be further refined to accommodate this setting. Finally, since our method is compatible with denoised score matching, it can naturally be extended to diffusionbased model. This paves the way for future work on applying our approach to generative modelling with diffusion processes in the presence of missing data.

- Acknowledgements Josh Givens was supported by a PhD studentship from the EPSRC Centre for Doctoral Training in Computational Statistics and Data Science (COMPASS). Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Bao, F., LI, C., Xu, K., Su, H., Zhu, J., and Zhang,
- B. Bi-level score matching for learning energy-based latent variable models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in neural information processing systems*, volume 33, pp. 18110–18122. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2020/file/d25a34b9c2a87db380ecd7f7115882ec-Paper.pdf) [cc/paper\\_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/d25a34b9c2a87db380ecd7f7115882ec-Paper.pdf) [d25a34b9c2a87db380ecd7f7115882ec-Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/d25a34b9c2a87db380ecd7f7115882ec-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/d25a34b9c2a87db380ecd7f7115882ec-Paper.pdf). Bao, F., Xu, K., Li, C., Hong, L., Zhu, J., and Zhang,
- B. Variational (gradient) estimate of the score function in energy-based latent variable models. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th international conference on machine learning*, volume 139 of *Proceedings of machine learning research*, pp. 651–661. PMLR, July 2021. URL [https://proceedings.](https://proceedings.mlr.press/v139/bao21b.html) [mlr.press/v139/bao21b.html](https://proceedings.mlr.press/v139/bao21b.html). Beck, A. The proximal gradient method. In *Firstorder methods in optimization*, pp. 269–329. Society for Industrial and Applied Mathematics, 2017. doi: 10.1137/1.9781611974997.ch10. URL [https://epubs.siam.org/doi/abs/10.](https://epubs.siam.org/doi/abs/10.1137/1.9781611974997.ch10) [1137/1.9781611974997.ch10](https://epubs.siam.org/doi/abs/10.1137/1.9781611974997.ch10). Brem, R. B. and Kruglyak, L. The landscape of genetic complexity across 5,700 gene expression traits in yeast. *Proceedings of the National Academy of Sciences of the United States of America*, 102(5):1572–1577, February 2005. ISSN 0027-8424 1091-6490. doi: 10.1073/pnas. 0408709102. Place: United States. Burda, Y., Grosse, R. B., and Salakhutdinov, R. Importance weighted autoencoders. In Bengio, Y. and LeCun,
- Y. (eds.), *4th international conference on learning representations, ICLR 2016, san juan, puerto rico, may 2-4, 2016, conference track proceedings*, 2016. URL [http:](http://arxiv.org/abs/1509.00519) [//arxiv.org/abs/1509.00519](http://arxiv.org/abs/1509.00519). tex.bibsource: dblp computer science bibliography, https://dblp.org tex.timestamp: Thu, 25 Jul 2019 14:25:37 +0200. Daras, G., Shah, K., Dagan, Y., Gollakota, A., Dimakis, A., and Klivans, A. Ambient diffusion: Learning clean distributions from corrupted data. In *Thirty-seventh conference on neural information processing systems*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=wBJBLy9kBY) [id=wBJBLy9kBY](https://openreview.net/forum?id=wBJBLy9kBY). Fattahi, S. and Sojoudi, S. Graphical lasso and thresholding: Equivalence and closed-form solutions. *Journal of Machine Learning Research*, 20(10):1–44, 2019. URL [http://jmlr.org/papers/v20/](http://jmlr.org/papers/v20/17-501.html) [17-501.html](http://jmlr.org/papers/v20/17-501.html). Huang, C.-W., Lim, J. H., and Courville, A. C. A variational perspective on diffusion-based generative models and score matching. *Advances in Neural Information Processing Systems*, 34:22863–22876, 2021. Hyvärinen, A. Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research*, 6(24):695–709, 2005. Hyvärinen, A. Some extensions of score matching. *Computational Statistics & Data Analysis*, 51(5):2499–2512, 2007. ISSN 0167-9473. doi: https://doi.org/10.1016/j.csda.2006.09.003. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0167947306003264) [science/article/pii/S0167947306003264](https://www.sciencedirect.com/science/article/pii/S0167947306003264). Li, S. C.-X., Jiang, B., and Marlin, B. MisGAN: Learning from Incomplete Data with Generative Adversarial Networks. In *International Conference on Learning Representations*, 2019a. URL [https://openreview.](https://openreview.net/forum?id=S1lDV3RcKm) [net/forum?id=S1lDV3RcKm](https://openreview.net/forum?id=S1lDV3RcKm). Li, Z., Chen, Y., and Sommer, F. T. Learning energy-based models in high-dimensional spaces with multi-scale denoising score matching, 2019b. arXiv: 1910.07762 [stat.ML]. Li, Z., Chen, Y., and Sommer, F. T. Learning energybased models in high-dimensional spaces with multiscale denoising-score matching. *Entropy. An International and Interdisciplinary Journal of Entropy and Information Studies*, 25(10), 2023. ISSN 1099-4300. doi: 10.3390/e25101367. URL [https://www.mdpi.](https://www.mdpi.com/1099-4300/25/10/1367) [com/1099-4300/25/10/1367](https://www.mdpi.com/1099-4300/25/10/1367). Number: 1367 tex.pubmedid: 37895489. Lin, L., Drton, M., and Shojaie, A. Estimation of highdimensional graphical models using regularized score matching. *Electronic Journal of Statistics*, 10(1):806 – 854, 2016. doi: 10.1214/16-EJS1126. URL [https://](https://doi.org/10.1214/16-EJS1126) [doi.org/10.1214/16-EJS1126](https://doi.org/10.1214/16-EJS1126). Publisher: Institute of Mathematical Statistics and Bernoulli Society.

- Liu, Q. and Wang, D. Learning Deep Energy Models: Contrastive Divergence vs. Amortized MLE, July 2017. URL [http://arxiv.org/abs/1707.](http://arxiv.org/abs/1707.00797) [00797](http://arxiv.org/abs/1707.00797). arXiv:1707.00797 [cs, stat]. Liu, S., Kanamori, T., and Williams, D. J. Estimating density models with truncation boundaries using score matching. *Journal of Machine Learning Research*, 23(186):1–38, 2022. URL [http://jmlr.](http://jmlr.org/papers/v23/21-0218.html) [org/papers/v23/21-0218.html](http://jmlr.org/papers/v23/21-0218.html). Ouyang, Y., Xie, L., Li, C., and Cheng, G. MissDiff: Training diffusion models on tabular data with missing values. In *ICML 2023 workshop on structured probabilistic inference & generative modeling*, 2023. URL [https:](https://openreview.net/forum?id=S435pkeAdT) [//openreview.net/forum?id=S435pkeAdT](https://openreview.net/forum?id=S435pkeAdT). Richardson, T. W., Wu, W., Lin, L., Xu, B., and Bernal,
- E. A. MCFlow: Monte Carlo Flow Models for Data Imputation. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 14193– 14202, June 2020. doi: 10.1109/CVPR42600.2020. 01421. ISSN: 2575-7075. Salimans, T. and Ho, J. Should EBMs model the energy or the score? In *Energy based models workshop*
- *- ICLR 2021*, 2021. URL [https://openreview.](https://openreview.net/forum?id=9AS-TF2jRNb) [net/forum?id=9AS-TF2jRNb](https://openreview.net/forum?id=9AS-TF2jRNb). Sasaki, H., Hyvärinen, A., and Sugiyama, M. Clustering via mode seeking by direct estimation of the gradient of a log-density. In Calders, T., Esposito, F., Hüllermeier, E., and Meo, R. (eds.), *Machine learning and knowledge discovery in databases*, pp. 19–34, Berlin, Heidelberg, 2014. Springer Berlin Heidelberg. ISBN 978-3- 662-44845-8. Song, Y. and Ermon, S. Generative Modeling by Estimating Gradients of the Data Distribution. In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. Song, Y., Garg, S., Shi, J., and Ermon, S. Sliced score matching: A scalable approach to density and score estimation. In Adams, R. P. and Gogate, V. (eds.), *Proceedings of the 35th Uncertainty in Artificial Intelligence Conference*, volume 115 of *Proceedings of Machine Learning Research*, pp. 574–584. PMLR, jul 2020. Song, Y., Durkan, C., Murray, I., and Ermon, S. Maximum likelihood training of score-based diffusion models. *Advances in Neural Information Processing Systems*, 34: 1415–1428, 2021a. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021b. Swersky, K., Ranzato, M., Buchman, D., Freitas, N. D., and Marlin, B. M. On autoencoders and score matching for energy based models. In *Proceedings of the 28th international conference on machine learning (ICML-11)*, pp. 1201–1208, 2011. Tashiro, Y., Song, J., Song, Y., and Ermon, S. CSDI: Conditional score-based diffusion models for probabilistic time series imputation. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems*, 2021. Uehara, M., Matsuda, T., and Kim, J. K. Imputation estimators for unnormalized models with missing data. In Chiappa, S. and Calandra, R. (eds.), *Proceedings of the twenty third international conference on artificial intelligence and statistics*, volume 108 of *Proceedings of machine learning research*, pp. 831–841. PMLR, August 2020. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v108/uehara20b.html) [press/v108/uehara20b.html](https://proceedings.mlr.press/v108/uehara20b.html). Vincent, P. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661– 1674, 2011. doi: 10.1162/NECO\_a\_00142. Vértes, E. and Sahani, M. Learning doubly intractable latent variable models via score matching. *InSymposium on advances in approximate Bayesian inference (AABI)*, 2016. Yang, E. and Lozano, A. C. Robust gaussian graphical modeling with the trimmed graphical lasso. In Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., and Garnett, R. (eds.), *Advances in neural information processing systems*, volume 28. Curran Associates, Inc., 2015. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2015/file/3fb451ca2e89b3a13095b059d8705b15-Paper.pdf) [cc/paper\\_files/paper/2015/file/](https://proceedings.neurips.cc/paper_files/paper/2015/file/3fb451ca2e89b3a13095b059d8705b15-Paper.pdf) [3fb451ca2e89b3a13095b059d8705b15-Paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/3fb451ca2e89b3a13095b059d8705b15-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2015/file/3fb451ca2e89b3a13095b059d8705b15-Paper.pdf). Yoon, J., Jordon, J., and Schaar, M. Gain: Missing data imputation using generative adversarial nets. In *International conference on machine learning*, pp. 5689–5698, 2018. tex.organization: PMLR. Yu, S., Drton, M., and Shojaie, A. Graphical models for non-negative data using generalized score matching. In Storkey, A. and Perez-Cruz, F. (eds.), *Proceedings of the twenty-first international conference on artificial intelligence and statistics*, volume 84 of *Proceedings of machine learning research*, pp. 1781–1790. PMLR, April 2018. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v84/yu18b.html) [press/v84/yu18b.html](https://proceedings.mlr.press/v84/yu18b.html). Yu, S., Drton, M., and Shojaie, A. Generalized score matching for general domains. *Information and Inference: A Journal of the IMA*, 11(2):739–780, 2022.

# A. Additional Theoretical Results

Here we present some interesting results which we feel help further build up the landscape of our method but were unable to fit within the main body of the paper.

# A.1. Additional Methods

Firstly some additional adaptations of score matching. Most of these are relatively immediate adaptations following our framework for missing score matching although there are some important aspects and caveats which make them worth officially documenting. Missing and sliced score matching are introduced in detail in Appendix [D.](#page-34-1)

# A.1.1. TRUNCATED SCORE MATCHING

We have already presented truncated score matching in the paper however we present it in more details alongside its assumptions here.

Assumption A.1. For any λ ∈ supp(Λ), θ ∈ Θ:

- X<sup>λ</sup> is connected, open and Lipschitz;
- pλ, gλ, qλ;<sup>θ</sup> ∈ H<sup>1</sup> (Xλ);
- pλ, g<sup>λ</sup> are continuously differentiable and qθ;<sup>λ</sup> is twice continuously differentiable;
- for any x ′ <sup>λ</sup> ∈ ∂Xλ, and j ∈ λ we have

$$\lim_{x_\lambda \xrightarrow{1} x'_\lambda} s_{\lambda;\theta}(x_\lambda)_j p_\lambda(x_\lambda) g_\lambda(x_\lambda) v_j(x'_\lambda) = 0.$$

where v(x ′ λ ) is the normal vector to the boundary δXλ.

This now leads us to our proposition on the validity of our population objective.

Proposition A.2. *Suppose that assumptions [4.2](#page-3-0) & [A.1](#page-11-2) hold. Then we have*

$$J_{\text{TM}}(\theta) := \mathbb{E} \left\{ g_{\Lambda}(X) \| s_{\Lambda;\theta}(X_{\Lambda}) - s_{\Lambda}(X_{\Lambda}) \|^2 \right\} = L_{\text{TM}}(\theta) - C \quad (12)$$

*where* <sup>C</sup> *is does not depend upon* <sup>θ</sup>*. As a direct result for* ˜<sup>θ</sup> *a minimiser of* <sup>L</sup>TM(θ) *we have that* <sup>s</sup>θ˜(X) = <sup>s</sup>(X) *a.s..*

*Proof.* Proof given in Appendix [C.1.1](#page-21-0)

## A.1.2. MISSING SLICED SCORE MATCHING

For readers who are unfamiliar with sliced score matching we provide a brief introduction in Appendix [D.3.](#page-35-0) For sliced score matching the only adaptations we need to make is to use our marginal scores and now alter our projection vectors to be over the appropriate subspace. Thus our objective becomes

$$\begin{aligned} L_{\text{SM}}(\theta) &:= \mathbb{E}[2 \{ \nabla_{X_\Lambda}(V_\Lambda^\top \mathbf{s}_{\Lambda;\theta}(X_\Lambda)) \}^\top V_\Lambda + V_\Lambda^\top \mathbf{s}_{\Lambda;\theta}(X_\Lambda)] \\ &= F_{\text{SM}}(\theta) - C \end{aligned}$$

where for any λ ∈ supp(Λ), V<sup>λ</sup> is a RV on <sup>R</sup> |λ| satisfying <sup>E</sup>[VλC ⊤ λ ] positive definite and <sup>E</sup>[∥Vλ∥ 2 ] < ∞.

To write this and it's gradient in terms of the full score, s<sup>θ</sup> we can again use Lemma [4.8.](#page-4-2)

This gives the following results

Proposition A.3.

$$\begin{aligned} L_{\text{SM}}(\theta) &= 2\mathbb{E} \left[ \mathbb{E}' \left[ \nabla_{X_\Lambda} (V_\Lambda^\top \mathbf{s}_\theta(X_\Lambda, X'_{-\Lambda})^\top V_\Lambda) \right] + \mathbb{E}'[(V^\top \mathbf{s}_\theta(X_\Lambda, X'_{-\Lambda}))^2] - \mathbb{E}'[(V^\top \mathbf{s}_\theta(X_\Lambda, X'_{-\Lambda}))^2] \right] & (13) \\ \nabla_\theta L_{\text{SM}}(\theta) &= 2\mathbb{E} \left[ \Psi_\Lambda \left( \nabla_{X_\Lambda} (V_\Lambda^\top \mathbf{s}_\theta(\cdot))^\top V_\Lambda \right) + \Psi_\Lambda (V^\top \mathbf{s}_\theta(\cdot))^2 - \mathbb{E}'[V^\top \mathbf{s}_\theta(X_\Lambda, X'_{-\Lambda})] \Psi_\Lambda (V^\top \mathbf{s}_\theta(\cdot)) \right] & (14) \end{aligned}$$

*where for any function* h<sup>θ</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup>*,*

$$\Psi_\Lambda(h_\theta) = \mathbb{E}'[\nabla_\theta h_\theta(X_\Lambda, X'_{-\Lambda})] + \text{Cov}'(\nabla_\theta \log q_\theta(X_\Lambda, X'_{-\Lambda}), h_\theta(X_\Lambda, X'_{-\Lambda}))$$

*and* E ′ , Cov′ *are w.r.t.* X′ −Λ |X<sup>Λ</sup> ∼ pθ(.|XΛ) *with* <sup>E</sup> *being w.r.t.* X<sup>Λ</sup> ∼ p*.*

*Proof.* We first have that

$$\begin{aligned}
L_{\text{SM}}(\theta) &= \mathbb{E} \left[ 2 \left\{ \nabla_{X_\Lambda} (V_\Lambda^\top \mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_\Lambda]) \right\}^\top V_\Lambda + (V_\Lambda^\top \mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_\Lambda])^2 \right] \\
&= \mathbb{E} \left[ \left( 2 \sum_{j \in \Lambda} V_j \nabla_{X_\Lambda} \mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_j]^\top V_\Lambda \right) + (V_\Lambda^\top \mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_\Lambda])^2 \right] \\
&= \mathbb{E} \left[ \left( 2 \sum_{j \in \Lambda} V_j (\mathbb{E}'[\nabla_{X_\Lambda} s_\theta(X_\Lambda, X'_{-\Lambda})_j] + \text{Cov}(s_\theta(X_\Lambda, X'_{-\Lambda}), s_\theta(X_\Lambda, X'_{-\Lambda})_j))^\top V_\Lambda \right) \right. \\
&\quad \left. + (V_\Lambda^\top \mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_\Lambda])^2 \right] \\
&= \mathbb{E} \left[ \left( 2\mathbb{E}' \left[ \nabla_{X_\Lambda} (V_\Lambda^\top s_\theta(X_\Lambda, X'_{-\Lambda})^\top V_\Lambda) \right] + \mathbb{E}'[(V^\top s_\theta(X_\Lambda, X'_{-\Lambda}))^2] - \mathbb{E}'[(V^\top s_\theta(X_\Lambda, X'_{-\Lambda}))^2] \right) \right. \\
&\quad \left. + \mathbb{E}'[V_\Lambda^\top s_\theta(X_\Lambda, X'_{-\Lambda})_\Lambda]^2 \right] \\
&= 2\mathbb{E} \left[ \mathbb{E}' \left[ \nabla_{X_\Lambda} (V_\Lambda^\top s_\theta(X_\Lambda, X'_{-\Lambda})^\top V_\Lambda) \right] + \mathbb{E}'[(V^\top s_\theta(X_\Lambda, X'_{-\Lambda}))^2] - \mathbb{E}'[(V^\top s_\theta(X_\Lambda, X'_{-\Lambda}))^2] \right]
\end{aligned}$$

The second results directly from applying Lemma [4.8](#page-4-2) again alongside the chain rule.

#### A.1.3. MISSING DENOISED SCORE MATCHING

As with sliced score matching the adaptation is relatively immediate however we do first need to make some further restrictions on our noising process. Specifically we require that for any t ∈ [0, 1], and j, j′ ∈ [d] we have X(t)<sup>j</sup> ⊥ X(t)<sup>j</sup> ′ |X(0)<sup>j</sup> .

In most practical implementations each coordinate is independently noised therefore satisfying this condition. We require this to allow us to easily write the marginal transition kernel for any λ ∈ supp(Λ) given by pλ(xλ(t)|xλ(0)). We then make our population objective

$$L_{\text{DM}}(\theta) := \mathbb{E} \left[ \nu(t) \left\{ \| \mathbf{s}_{\Lambda; \theta}(X_{\Lambda}(t), t) \|^2 + \nabla_{X_{\Lambda}(t)} \log p_{\Lambda}(X_{\Lambda}(t) | X_{\Lambda}(0)) \right\} \right]$$

We can again write this in terms of s<sup>θ</sup> as we do in the following proposition

Proposition A.4.

$$L_{\text{DM}}(\theta) = \mathbb{E} \left[ \nu(t) \left\{ \sum_{j \in \Lambda} \mathbb{E}' [s_\theta(X_\Lambda, X'_{-\Lambda})_j]^2 + \nabla_{X_\Lambda(t)} \log p_\Lambda(X_\Lambda(t) | X_\Lambda(0)) \right\} \right] \quad (15)$$

$$\begin{aligned} \nabla_{\theta} L_{\text{DM}}(\theta) &= \mathbb{E} \left[ \nu(t) \left\{ \sum_{j \in \Lambda} \mathbb{E}' [\mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t)_j] \left( \mathbb{E}'[\partial_j \mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda})_j] \right. \right. \right. \\ &\quad \left. \left. + \text{Cov}'(\mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, \mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t)_j)) \right) \right. \\ &\quad \left. \left. + \nabla_{X_{\Lambda}(t)} \log p_{\Lambda}(X_{\Lambda}(t) | X_{\Lambda}(0)) \right\} \right] \end{aligned} \quad (16)$$

*where for any function* h<sup>θ</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup>*,*

$$\Psi_\Lambda(h_\theta) = \mathbb{E}'[\nabla_\theta h_\theta(X_\Lambda(t), X'_{-\Lambda})] + \text{Cov}'(\nabla_\theta \log q_\theta(X_\Lambda(t), X'_{-\Lambda}), h_\theta(X_\Lambda(t), X'_{-\Lambda}))$$

*and* E ′ , Cov′ *are w.r.t.* X′ −Λ |XΛ(t) ∼ pθ(.|XΛ) *with* <sup>E</sup> *being w.r.t.* XΛ(t) ∼ pt*.*

*Proof.* Using Lemma [4.8,](#page-4-2) we have that

$$\begin{aligned} L_{\text{DM}}(\theta) &= \mathbb{E} \left[ \nu(t) \left\{ \sum_{j \in \Lambda} \mathbf{s}_{\Lambda; \theta}(X(t)_{\Lambda}, t)_j^2 + \nabla_{X_{\Lambda}(t)} \log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0)) \right\} \right] \\ &= \mathbb{E} \left[ \nu(t) \left\{ \sum_{j \in \Lambda} \mathbb{E}' [\mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t)_j]^2 + \nabla_{X_{\Lambda}(t)} \log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0)) \right\} \right] \end{aligned}$$

A second application of the lemma then gives,

$$\begin{aligned}\nabla_{\theta} L_{\text{DM}}(\theta) &= \mathbb{E} \left[ \nu(t) \left\{ \sum_{j \in \Lambda} \mathbb{E}' [\mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t)_j] \left( \mathbb{E}'[\partial_j \mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t)_j] \right. \right. \right. \\ &\quad \left. \left. + \text{Cov}'(\mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t), \mathbf{s}_{\theta}(X_{\Lambda}(t), X'_{-\Lambda}, t)_j) \right) \right\} \\ &\quad \left. \left. + \nabla_{X_{\Lambda}(t)} \log p_{\Lambda}(X_{\Lambda}(t) | X_{\Lambda}(0)) \right\} \right].\end{aligned}$$

#### A.1.4. MISSING NOT AT RANDOM DATA

So far we have assumed that our data is missing completely at random so that XΛ|Λ = λ ∼ Xλ. In other words, we could treat corrupted samples as though they were simply marginal samples and still perform valid inference. Often however, such an assumption is unrealistic and the probability of parts of a sample begin missing depends upon the sample itself. Generally this is split into two cases. Missing at Random (MAR) and Missing not at Random (MNAR). MAR data occurs when the probability of a coordinate being missing depends only upon other coordinates of the sample. This means that M<sup>j</sup> ⊥ X<sup>j</sup> |X−<sup>j</sup> . In MNAR data we allow M<sup>j</sup> to depend upon X<sup>j</sup> as well meaning that an observations value determines its own probability of being missing.

Here we will focus in the MNAR scenario and treat the MAR scenario as a special case of this. The core idea of this approach will be to work with a "joint" score rather than a marginal score. Before we do this we need to set-up our MNAR case. Specifically for λ ∈ supp(Λ) define the event

$$E_\lambda := \{X'_\lambda \neq \emptyset, X'_{-\lambda} = \emptyset\}$$

and define φλ(X) := <sup>P</sup>(Eλ|X). Throughout we will assume each φ<sup>λ</sup> to be *known*. This is often an unrealistic assumption however this allows us the flexibility of having a method which is independent of how the φ<sup>λ</sup> are learned.

To work with this MNAR data we need to define some adaptations of densities and score functions.

Definition A.5. X with PDF p and event E define p(x; E) to be the "joint" density satisfying

$$\int_B p(\mathbf{x}; E) dx = \mathbb{P}(\{X \in B\} \cup E)$$

for all B ∈ B<sup>X</sup> .

From this and with our particular events we can redefine the missing score as,

$$s_\lambda(\mathbf{x}_\Lambda) = \nabla_{\mathbf{x}_\lambda} \log p_\lambda(\mathbf{x}_\lambda; E_\lambda) \quad (17)$$

$$= \nabla_{\mathbf{x}_\lambda} \log \left( \int p(\mathbf{x}; E) \mathrm{d}\mathbf{x}_{-\lambda} \right) \quad (18)$$

$$= \nabla_{\mathbf{x}_\lambda} \log \left( \int p(\mathbf{x}) \varphi_\lambda(\mathbf{x}) d\mathbf{x}_{-\lambda} \right)$$

*Remark* A.6*.* this missing score is *not* the same as the marginal score. We slightly abuse notation here using the same notation as we did for the marginal score. This is however reasonable as for the MCAR case the marginal score and the missing score are identical.

With this newly defined score, we can proceed similarly to the MCAR case and use the objective LˆM(θ) defined in [\(4\)](#page-3-2) or [\(5\)](#page-3-3) but with our new defined score. We now show a provide a similar justification for this approach as in the MCAR case but first need to introduce an additional assumption.

Assumption A.7. For each λ ∈ supp(Λ), <sup>P</sup>(Eλ|Xλ) > 0 a.s..

*Remark* A.8*.* We do not require every missingness pattern to have positive probability just that if a missingness pattern does have positive probability, it has positive probability for every possible underlying sample.

This then leads us to our desired result.

Proposition A.9. *Suppose with are in our MNAR set-up and assume that assumptions [4.2](#page-3-0) & [A.7](#page-14-2) hold and that there exists* θ <sup>∗</sup> *with* s<sup>θ</sup> <sup>∗</sup> (X) = sθ(X) *a.s.. Then if* ˜θ *is a minimiser of* LM(θ) *where the missing scores are defined by [\(17](#page-13-1) then* sθ˜(X) = s(X) *a.s..*

The proof for this is similar to the MCAR case and is given in Appendix [C.1.2.](#page-22-0)

Now we have our objective we need to see how we can derive sλ(xλ). Again we can do this similarly to the MCAR case. Let q<sup>θ</sup> be our estimate of the unnormalised density then

$$\begin{aligned}
s_{\lambda;\theta}(\mathbf{x}_\lambda) &= \nabla_{\mathbf{x}_\lambda} \log p_{\lambda;\theta}(\mathbf{x}_\lambda; E_\lambda) \\
&= \nabla_{\mathbf{x}_\lambda} \log \int_{\mathcal{X}_{-\lambda}} p_\theta(\mathbf{x}; E_\lambda) d\mathbf{x}_{-\lambda} \\
&= \nabla_{\mathbf{x}_\lambda} \log \int_{\mathcal{X}_{-\lambda}} p_\theta(\mathbf{x}) \varphi_\lambda(\mathbf{x}) d\mathbf{x}_{-\lambda} \\
&= \nabla_{\mathbf{x}_\lambda} \log \int_{\mathcal{X}_{-\lambda}} q_\theta(\mathbf{x}) \varphi_\lambda(\mathbf{x}) d\mathbf{x}_{-\lambda} \\
&= \nabla_{\mathbf{x}_\lambda} \log \mathbb{E}_{p'_\lambda} \left[ \frac{q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) \varphi_\lambda(\mathbf{x}_\lambda, X'_{-\lambda})}{p'(X'_{-\lambda})} \right] \\
&\approx \nabla_{\mathbf{x}_\lambda} \log \frac{1}{r} \sum_{k=1}^r \left[ \frac{q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)}) \varphi_\lambda(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)})}{p'(X'_{-\lambda}^{(k)})} \right]
\end{aligned}$$

As a result we can approximate our objective analogously to our approach for MCAR data.

# A.2. Finite Sample Bounds for Truncated Importance Weighted Score Matching

To be able to prove finite sample bound results we first need to present some key definitions.

Definition A.10 (Approximate Truncated Marginal Score Matching Objective). For n, r ∈ N, θ ∈ Θ take our sample objective to be

$$\hat{L}_{\text{TM};n,r}(\theta) := \frac{1}{n} \sum_{i=1}^n g_{\Lambda_i}(X_{\Lambda_i}^{(i)}) \|\hat{s}_{\Lambda_i,r;\theta}(X_{\Lambda_i}^{(i)})\|^2 + 2g_{\Lambda_i}(X_{\Lambda_i}^{(i)}) \nabla_{X_{\Lambda_i}^{(i)}} \cdot \hat{s}_{\Lambda_i,r;\theta}(X_{\Lambda_i}^{(i)}) + 2\nabla_{X_{\Lambda_i}^{(i)}} g_{\Lambda_i}(X_{\Lambda_i}^{(i)})^\top \hat{s}_{\Lambda_i,r;\theta}(X_{\Lambda_i}^{(i)})$$

with sˆλ,r;<sup>θ</sup>(xλ) being our estimated marginal score from Definition [4.4.](#page-4-3)

Additionally we define

$$f_{0,\lambda}(\mathbf{x}, \theta) := \frac{q_{\theta}(\mathbf{x})}{p'(\mathbf{x}_{-\lambda})} \quad f_{1,\lambda}(\mathbf{x}, \theta) := \frac{\nabla_{\mathbf{x}} q_{\theta}(\mathbf{x})}{p'(\mathbf{x}_{-\lambda})} \quad f_{2,\lambda}(\mathbf{x}, \theta) := \frac{\nabla_{\mathbf{x}} cdot(\nabla_{\mathbf{x}} q_{\theta}(\mathbf{x}))}{p'(\mathbf{x}_{-\lambda})}.$$

We now set-up the following assumptions

- ∥fk,λ(x, θ)∥, gλ(xλ), ∥∇<sup>x</sup><sup>λ</sup> gλ(xλ)∥< a,
- 1 <sup>a</sup> < f0,λ(x, θ)

*Remark* A.12*.* It is this assumptions which restrict us from obtaining a similar result in the non-truncated case as it is unrealistic to have both <sup>1</sup> <sup>a</sup> < f0,λ(x) and p(xλ) → 0 as ∥xλ∥→ ∞.

Assumption A.13. For each λ ∈ supp(Λ), l ∈ {0, 1, 2} we have that for any θ, θ′ ∈ Θ:

$$\|f_{l,\lambda}(\mathbf{x}, \theta) - f_{0,\lambda}(\mathbf{x}, \theta')\| \leq M_k(\mathbf{x})\rho(\theta, \theta'),$$

where Mk(Xλ, x−λ), Mk(xλ, X′ −λ ) are sub-Gaussian with parameters σl,λ, σ′ l,−λ respectively for all x−<sup>λ</sup> ∈ X<sup>−</sup>λ.

*Remark* A.14*.* This assumption is immediately satisfied if Θ is compact and fl,λ(x, θ) are pointwise Lipschitz w.r.t. θ. Hence this assumption is slightly weaker than a uniformly lipschitz assumption

We can now state our theorem

Theorem A.15. *Assume that assumptions [4.2,](#page-3-0) [A.1,](#page-11-2) [A.11,](#page-14-1) [A.13](#page-15-0) hold and let* θn,r ∈ Θ ⊆ <sup>R</sup> <sup>p</sup> *be the minimisers of* LˆTM;n,r(θ)*. If* Θ ⊆ <sup>R</sup> p *then for sufficiently large* n, r

$$\mathbb{P} \left( F_{\text{TM}}(\theta_{n,r}) \geq \beta_1 \sqrt{\frac{p \log(dnr \operatorname{diam}(\Theta)/\delta)}{r}} + \beta_2 \sqrt{\frac{p \log(n \operatorname{diam}(\Theta)/\delta)}{n}} + \beta_3 \left( \frac{n+r}{nr} \right) \left( C + \sqrt{\frac{\log(n/\delta)}{n}} \right) \right) < \delta.$$

*where* β1, β<sup>2</sup> *depend upon* a*,* β<sup>3</sup> *depends upon* a, {σλ,l, σ′ <sup>−</sup>λ,l}(l,λ)∈{0,1,2}×supp(Λ) *and* C *depends upon* a*,* {E[Mk(Xλ, X′ −λ )]}(l,λ)∈{0,1,2}×supp(Λ)*.*

*Proof.* The proof for this alongside multiple intermediary results can be found in [C.2](#page-24-0)

Here we have shown convergence of our sample/approximate objective to the population objective. This combined with proposition [A.2](#page-11-4) which states that our population objective is minimised by the true score suggests that our approach does give valid inference for learning the score. A key limitation of our result is that to obtain convergence, we require r−→∞. Furthermore, to obtain log(n)/n rate convergence we need r to be of the same order as n. As the computational complexity of our algorithm in O(nr), this suggests that to obtain our desired convergence to the population objective will have O(n 2 ) computational complexity.

*Remark* A.16*.* Our dependency on our Lipschitz constants only enters into the C term with the associate sub-Gaussian parameters entering only into the σ.

*Remark* A.17*.* Dependence upon g simply requires g and ∇g bounded. This is achieved on a compact X by g(x) = minx′∈∂X d(x, x ′ ) and on a non-compact space by g(x) = minx′∈∂X d(x, x ′ ) V 1.

# A.3. Relationship between IW and Variational objectives

Despite being derived quite differently from the marginal score matching objective. We show below that the two objectives are actually identical in some cases. Specifically, when the IW density p ′ doesn't depend upon the observed data xΛ, we can treat the importance weighted approach as an importance weighting approximation of the gradient estimate in [\(10\)](#page-5-0). We demonstrate this through the two results below

Lemma A.18. *For some density* p ′ *which generates IID samples* {X ′(k) −λ }k∈<sup>r</sup> *let*

$$\begin{aligned} w_k &:= \frac{q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)})}{p'(X'_{-\lambda}^{(k)})} \\ \bar{w}_k &:= w_k \left( \sum_{k'=1}^r w_{k'} \right)^{-1} \\ \hat{s}_{\theta, \lambda}(\mathbf{x}_\lambda) &:= \nabla_{\mathbf{x}_\lambda} \log \frac{1}{r} \sum_{k=1}^r w_k \\ \hat{\mathbb{E}}_{iw}[g_\theta(X)] &:= \frac{1}{r} \sum_{k=1}^r \bar{w}_k g_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)}) \\ \hat{\text{Cov}}_{iw}(f(X), g_\theta(X)) &:= \frac{1}{r} \sum_{k=1}^r \bar{w}_k g_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)}) f(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)}) - \left( \frac{1}{r} \sum_{k=1}^r \bar{w}_k g_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)}) \right) \left( \frac{1}{r} \sum_{k=1}^r \bar{w}_k f(\mathbf{x}_\lambda, X'_{-\lambda}^{(k)}) \right). \end{aligned}$$

*Then*

$$\hat{s}_{\theta;\lambda}(\mathbf{x}_\lambda) = \hat{\mathbb{E}}_{iw}[s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \quad (19)$$

$$\nabla \hat{\mathbb{E}}_{iw}[g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] = \hat{\mathbb{E}}_{iw}[\nabla g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] + \hat{\text{Cov}}_{iw}(s_\theta(\mathbf{x}_\lambda, X'_{-\lambda}), g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})). \quad (20)$$

*where* ∇ *represents the gradient w.r.t.* x<sup>λ</sup> *or* θ*. In other words, we can take importance weights first then gradients (LHS) or gradients and then importance weights (RHS).*

*Proof.* Proof given in Section [C.4.](#page-33-0)

Corollary A.19. *We have that*

$$\begin{aligned} \nabla_{\theta} \hat{L}(\theta; \mathbf{x}_{\lambda}, X'_{-\Lambda}) &= -2\hat{\mathbb{E}}_{iw}[\mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i] \left\{ \hat{\mathbb{E}}_{iw}[\nabla_{\theta} \mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i] + \text{Cov}_{iw} (\nabla_{\theta} \log q_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda}), \mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i) \right\} \\ &\quad + 2(\hat{\mathbb{E}}_{iw}[\nabla_{\theta} \mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i^2] + \text{Cov}_{iw} (\nabla_{\theta} \log q_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda}), \mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i^2)) \\ &\quad + 2(\hat{\mathbb{E}}_{iw}[\nabla_{\theta} \partial_i \mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i] + \text{Cov}_{iw} (\nabla_{\theta} \log q_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda}), \partial_i \mathbf{s}_{\theta}(\mathbf{x}_{\lambda}, X'_{-\Lambda})_i)) \end{aligned} \quad (21)$$

*Proof.* Proof given in Section [C.4.](#page-33-0)

# A.4. Exploring the Marginal Fisher Divergence for Normal Distributions

While intuitively the Fisher divergences of the marginal distributions should act as effective proxies for the Fisher divergence for the fully observed distributions, we would like to be able to examine the relationship between the two more explicitly. We do know that marginal Fisher divergences will be zero when then fully observed distributions equivalent however here we give a more detailed examination in the case of normal distributions.

Suppose that X ∼ N(µ, P <sup>−</sup><sup>1</sup> )

$$p(x) = \exp\{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top P(\mathbf{x} - \boldsymbol{\mu})\} + C$$

with C a constant w.r.t. x. We then have that

$$s(x) = -P(x - \mu)$$

If we suppose that our unnormalised density/score model is of the form

$$q_\theta(\mathbf{x}) := \exp \left\{ -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_\theta)^\top P_\theta(\mathbf{x} - \boldsymbol{\mu}_\theta) \right\} \quad \Rightarrow \quad \mathbf{s}_\theta = P_\theta(\mathbf{x} - \boldsymbol{\mu}_\theta)$$

Then with the marginal Fisher taken to be

$$F_M(\theta) = \mathbb{E}_{\Lambda, X_\Lambda} [\|s_\Lambda(X_\Lambda)s_{\theta; \Lambda}\|^2]$$

where here for each λ ∈ supp(Λ), sλ, sθ;<sup>λ</sup> are the true marginal scores. Using properties of the normal distribution and the Schur complement we know that the precision of X<sup>λ</sup> is given by

$$\left\{ (P^{-1})_{\lambda,\lambda} \right\}^{-1} = P_{\lambda,\lambda} - P_{\lambda,-\lambda} P_{-\lambda,-\lambda}^{-1} P_{-\lambda,\lambda}.$$

Plugging this in we get

$$F_M(\theta) = \mathbb{E} \left[ \left\| (P_{\Lambda, \Lambda} - P_{\theta; \Lambda, \Lambda}) X_\Lambda + (P_{\Lambda, -\Lambda} P_{-\Lambda, -\Lambda}^{-1} P_{-\Lambda, \Lambda} - P_{\theta; \Lambda, -\Lambda} P_{\theta; -\Lambda, -\Lambda}^{-1} P_{\theta; -\Lambda, \Lambda}) X_\Lambda \right. \right. \\ \left. \left. - ((P_{\Lambda, \Lambda} - P_{\Lambda, -\Lambda} P_{-\Lambda, -\Lambda}^{-1} P_{-\Lambda, \Lambda}) \boldsymbol{\mu}_\Lambda - (P_{\theta; \Lambda, \Lambda} - P_{\theta; \Lambda, -\Lambda} P_{\theta; -\Lambda, -\Lambda}^{-1} P_{\theta; -\Lambda, \Lambda}) \boldsymbol{\mu}_{\theta; \Lambda}) \right\|^2 \right].$$

This shows why naive marginalisation by zeroing out missing coordinates of our score would not work as in this case the Fisher divergence would be given by

$$F_M(\theta) = \mathbb{E} \left[ \left\| \left( (P_{\Lambda, \Lambda} - P_{\Lambda, -\Lambda} P_{-\Lambda, -\Lambda}^{-1} P_{-\Lambda, \Lambda}) - P_{\theta; \Lambda, \Lambda} \right) X_\Lambda - \left( (P_{\Lambda, \Lambda} - P_{\Lambda, -\Lambda} P_{-\Lambda, \Lambda}^{-1} P_{-\Lambda, \Lambda}) \boldsymbol{\mu}_\Lambda - P_{\theta; \Lambda, \Lambda} \boldsymbol{\mu}_{\theta; \Lambda} \right) \right\|^2 \right]$$

which encourages Pθ;λ,λ to be close to Pλ,λ − Pλ,−λP −1 <sup>−</sup>λ,−λP−λ,λ for all λ ∈ supp(Λ) meaning it will not give us the true density.

### A.5. Variational Pseudo-loss

When using [\(10\)](#page-5-0) it is helpful to be able to view it as the gradient of some pseudo-loss allowing it to plug into a more standard ML framework where we calculate the loss, take the gradient w.r.t. our parameter using auto-differentiation, and update our parameter estimate based on this. The below result show how we can do this by creating a loss with certain instances of our parameter detached from the computational graph.

Proposition A.20. *Let*

$$\begin{aligned}
 J(\theta, \theta', \mathbf{x}_\lambda, X'_{-\lambda}) &:= -2\mathbb{E}'[s_{\theta'}(\mathbf{x}_\lambda, X'_{-\lambda})_i \{ \mathbb{E}'[s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_i] + \text{Cov}'(\log q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}), s_{\theta'}(\mathbf{x}_\lambda, X'_{-\lambda})_i) \}] \\
 &\quad + 2(\mathbb{E}'[s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_i^2] + \text{Cov}'(\log q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}), s_{\theta'}(\mathbf{x}_\lambda, X'_{-\lambda})_i^2)) \\
 &\quad + 2(\mathbb{E}'[\partial_i s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_i] + \text{Cov}'(\log q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}), \partial_i s_{\theta'}(\mathbf{x}_\lambda, X'_{-\lambda})_i))
 \end{aligned}$$

*where* E ′ , Cov′ *are w.r.t.* X′ −λ |X<sup>λ</sup> = xλ; θ ′ *Then*

$$\nabla_{\theta'} L(\theta', \mathbf{x}_\lambda) = \frac{\partial}{\partial \theta} J(\theta, \theta', \mathbf{x}_\lambda) \Big|_{\theta=\theta'}$$

*Proof.* This just follows directly from the exchangeability of expectations and gradients (when the gradient is w.r.t. something independent of the expectation distribution.)

Hence we can use this loss (by replacing all instances of θ ′ with θ and then detaching them from the computation graph) to treat our problem as a standard gradient descent problem.

Note that while we can treat this like a loss for our optimisation, our intent is not actually to minimise it. The estimated form of the loss is given in the proof of Corollary [4.9](#page-5-2) which is given in [C.3](#page-32-0) but we state it again explicitly here for convenience.

$$L_M(\theta) = \mathbb{E} \left[ \sum_{j \in \Lambda} -\mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_j]^2 + 2\mathbb{E}'[s_\theta(X_\Lambda, X'_{-\Lambda})_j^2] + 2\mathbb{E}'[\partial_i s_\theta(X_\Lambda, X'_{-\Lambda})_j] \right]$$

![](_page_18_Figure_1.jpeg)

Figure 8: Average parameter estimation error for truncated Gaussian score estimates alongside 95% confidence intervals under various methods.

# B. Additional Experimental Results

Here we present some additional experimental results not in the main body of the paper.

# B.1. Parameter Estimation

# B.1.1. TRUNCATED GAUSSIAN MODEL

Here we present the accompanying mean and precision error results for Gaussian model estimation experiment presented in Section [5.1.1.](#page-6-5) These results are presented in Figure [8.](#page-18-2)

# B.1.2. UNTRUNCATED GAUSSIAN MODEL

Here we present the untruncated version of the experiment presented in the main paper. Details of the distribution are the same as presented in Appendix [E.3](#page-37-0) but without the truncation.

![](_page_18_Figure_10.jpeg)

Figure 9: Average Fisher Divergence with 95% C.I.s for various approaches. Lower is better.

As we can see we obtains similar results here as in the truncated case.

We also illustrate what the true covariance and precision matrix look like for this example alongside the naive marginalisation in order to highlight where Zeroed Score Matching goes wrong.

In Figure [10](#page-19-2) we can see the covariance and precision of a sample distribution where we can clearly see the strong dependence of dimensions 1 and 10 relative to the others.

![](_page_19_Figure_1.jpeg)

Figure 10: Covariance and precision from a sample distribution from our normal experiment.

been cube-rooted in order to emphasize the difference between zero and non-zero entries. Here we can see that the naive marginalisation wouldn't capture the dependence between dimension 10 and the other dimensions that gets introduced when dimension 1 is removed. This means that a naive marginalisation would assume that dimension 10 must have a direct dependence on dimensions 2-9 even when that is not true. Interestingly, the rest of the marginalisation seems very similar suggesting that in some potentially less structured cases, naive marginalisation can provide a semi-reasonable approximation. This supports the results we see in our GGM estimation where highly structured graphs like star graphs are much more affected naive marginalisation than unstructured graphs.

# B.1.3. NON-GAUSSIAN ESTIMATION

Here we present further experiments exploring the non-Gaussian model presented in Section [5.1.2.](#page-6-6) Here we fix the dimension as 10. In Figure [12a](#page-21-1) we fix the missing probability as 0.5 and vary the sample size. In Figure [12b](#page-21-1) we fix the sample size as 1000 and vary the missing probability.

From Figure [12a](#page-21-1) we see that both EM and Marg-IW have the smallest estimation error for larger sample sizes. Zeroed Score Matching has the largest estimation error due to its inability to appropriately marginalise the distribution. In Figure [12b,](#page-21-1) we observe that Marg-Var has the smallest estimation error with its performance convergence to that of Marg-IW and EM as the missing probability increases.

# B.2. GGM Estimation

# B.2.1. VARYING NUMBER OF STAR CENTRES

Here we present illustrations of the marginalisations for our star-shaped graphs with 1 node and then 5 nodes both with the same edge density.

In Figure [13](#page-22-1) we show the covariance, precision, marginal covariance, and marginal precision for a star graph with 1 centre where the marginal terms are with dimension 1 removed. As we can see clearly the only meaningful structure left in the graph after marginalisation are in the negative precision terms which the model naive marginalisation fails to capture.

In Figure [14](#page-23-0) we show the same thing for the case of a star graph with 5 centres. As we can see in the 5 centre case, the naive marginalisation picks up most of the structure as there are fewer negative terms which it ignores and also lots of additional positive terms which it does successfully pick up.

# B.2.2. VARYING NUMBER OF DIMENSIONS

Here we use our same star-shaped GGM as in the main paper but with a varying number of dimensions. Throughout 1,000 samples are used and each coordinate is missing independently with probability 0.7. Results are presented in Figure [15.](#page-24-1)

![](_page_20_Figure_1.jpeg)

Figure 11: Marginalisation of the precision to remove dimension 1 by the naive approach (i.e. subsetting the precision) and the correct approach. All values cube-rooted for contrastive purposes.

approach catch up and even overtake it. This is because at lower dimensions, IW can effectively model the marginalisation of the score and so the more complicated variational approach is not required.

# B.2.3. INDIVIDUAL ROC CURVES

Here we present individual ROC curves from Section [5.2.1](#page-7-2) with a missing probability of 0.5. Here we specifically present the ROC curves from the first 4 runs out of the 50 performed for the experiment. These ROC curves are displayed in Figure [16.](#page-25-0) We observe that Marg-Var has consistently the best ROC curves of any of the methods.

# C. Additional Proofs

# C.1. Marginal Score Matching Objectives

*proof of Proposition [4.3.](#page-3-4)* For the first claim we have that

$$\begin{aligned} \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_\Lambda) - \mathbf{s}_\Lambda(X_\Lambda)\|^2] &= \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_\Lambda)\|^2] + \mathbb{E}[\|\mathbf{s}_\Lambda(X_\Lambda)\|^2] - 2\mathbb{E}[\mathbf{s}_{\Lambda;\theta}(X_\Lambda)^\top \mathbf{s}_\Lambda(X_\Lambda)] \\ &= C + \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_\Lambda)\|^2] - 2 \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \int_{X_\lambda} p_\lambda(\mathbf{x}_\lambda) \mathbf{s}_{\lambda;\theta}(\mathbf{x}_\lambda)^\top \mathbf{s}_\lambda(\mathbf{x}_\lambda) d\mathbf{x}_\lambda \\ &= C + \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_\Lambda)\|^2] - 2 \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \int_{\mathcal{X}} \nabla_{\mathbf{x}_\lambda} p_\lambda(\mathbf{x}_\lambda)^T \mathbf{s}_{\lambda;\theta}(\mathbf{x}_\lambda) d\mathbf{x}_\lambda \\ &= C + \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_\Lambda)\|^2] - 2 \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \sum_{j=1}^d \int_{\mathcal{X}_\Lambda} \nabla_{\mathbf{x}_\lambda} p_\lambda(\mathbf{x}_\lambda)_j \mathbf{s}_{\lambda;\theta}(\mathbf{x}_\lambda)_j d\mathbf{x}_\lambda \end{aligned}$$

Hence by integration by parts we have

$$\begin{aligned}\mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_{\Lambda}) - \mathbf{s}_{\Lambda}(X_{\Lambda})\|^2] &= C + \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})\|^2] \\ &\quad - 2 \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \sum_{j \in \lambda} \left\{ \lim_{x_j \xrightarrow{\partial \mathcal{X}}} p_{\lambda}(\mathbf{x}_{\lambda}) \mathbf{s}_{\lambda;\theta}(\mathbf{x}_{\lambda}) - \int_{\mathcal{X}_{\lambda}} p_{\lambda}(\mathbf{x}_{\lambda})_j \mathbf{s}_{\lambda;\theta}(\mathbf{x}_{\lambda})_j dx_{\lambda} \right\} \\ &= \mathbb{E}[\|\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})\|^2] + 2\mathbb{E}[\nabla_{X_{\Lambda}} \cdot \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})] + C\end{aligned}$$

justifying our first claim. Hence if ˜θ minimised our objective then it minimises

$$\mathbb{E}[\|s_{\Lambda;\theta}(X_{\Lambda}) - s_{\Lambda}(X_{\Lambda})\|^2].$$

![](_page_21_Figure_1.jpeg)

Figure 12: Average parameter estimation error (L2 norm) for ICA inspired model with 95% C.I.s for various methods. Lower is better

As there exists a "true" θ we know that this objective is minimised at 0 and so we must have sλ;θ˜(Xλ) = sλ(Xλ) a.s. for all λ ∈ supp(Λ). By our assumption this then gives pθ˜(X) = p(X) a.s. .

# C.1.1. TRUNCATED SCORE MATCHING

*Proof of Proposition [A.2.](#page-11-4)* We mostly use [\(Liu et al., 2022\)](#page-10-8). We firstly have that

$$\begin{aligned} & \mathbb{E}[g_{\Lambda}(X_{\Lambda}) \| \mathbf{s}_{\Lambda;\theta}(X_{\Lambda}) - \mathbf{s}_{\Lambda}(X_{\Lambda}) \|^2] \\ &= \mathbb{E}[g_{\Lambda}(X_{\Lambda}) \| \mathbf{s}_{\Lambda}(X_{\Lambda}) \|^2] + \mathbb{E}[g_{\Lambda}(X_{\Lambda}) \| \mathbf{s}_{\Lambda;\theta}(X_{\Lambda}) \|^2] - 2\mathbb{E}[g_{\Lambda}(X_{\Lambda}) \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})^{\top} \mathbf{s}_{\Lambda}(X_{\Lambda})] \end{aligned}$$

Now we have that

$$\begin{aligned} & \mathbb{E}[g_{\Lambda}(X_{\Lambda}) \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})^T \mathbf{s}_{\Lambda}(X_{\Lambda})] \\ &= \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \mathbb{E}[g_{\Lambda}(X_{\Lambda}) \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})^T \mathbf{s}_{\Lambda}(X_{\Lambda}) | \Lambda = \lambda] \\ &= \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \sum_{j \in \lambda} \int_{\mathcal{X}} g_{\lambda}(\mathbf{x}_{\lambda}) \mathbf{s}_{\theta;\lambda}(\mathbf{x}_{\lambda}) \frac{\partial}{\partial \mathbf{x}_j} p_{\lambda}(\mathbf{x}_{\lambda}) d\mathbf{x}_{\lambda} \\ &= \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \sum_{j \in \lambda} \int_{\mathcal{X}} g_{\lambda}(\mathbf{x}_{\lambda}) \mathbf{s}_{\theta;\lambda}(\mathbf{x}_{\lambda})_j \frac{\partial}{\partial \mathbf{x}_j} p_{\lambda}(\mathbf{x}_{\lambda}) d\mathbf{x}_{\lambda} \\ &\stackrel{(a)}{=} \sum_{\lambda \in \text{supp}(\Lambda)} \mathbb{P}(\Lambda = \lambda) \sum_{j \in \lambda} \left\{ \int_{\partial X_{\lambda}} g_{\lambda}(\mathbf{x}_{\lambda}) \mathbf{s}_{\theta;\lambda}(\mathbf{x}_{\lambda})_j p_{\lambda}(\mathbf{x}_{\lambda}) v_j(\mathbf{x}_{\lambda}) ds - \int_{\mathcal{X}} \frac{\partial}{\partial \mathbf{x}_j} [g_{\lambda}(\mathbf{x}_{\lambda}) \mathbf{s}_{\theta;\lambda}(\mathbf{x}_{\lambda})_j] p_{\lambda}(\mathbf{x}_{\lambda}) d\mathbf{x}_{\lambda} \right\} \\ &\stackrel{(b)}{=} - \sum_{\lambda \in \mathcal{P}(d)} \mathbb{P}(\Lambda = \lambda) \sum_{j \in \lambda} \int_{\mathcal{X}} \left[ g_{\lambda}(\mathbf{x}_{\lambda}) \frac{\partial}{\partial \mathbf{x}_j} \mathbf{s}_{\theta;\lambda}(\mathbf{x}_{\lambda})_j + \frac{\partial}{\partial \mathbf{x}_k} g_{\lambda}(\mathbf{x}_{\lambda}) \mathbf{s}_{\lambda;\theta}(\mathbf{x}_{\lambda}) \right] p_{\lambda}(\mathbf{x}_{\lambda}) d\mathbf{x}_{\lambda} \\ &= -\mathbb{E}[\nabla_{X_{\Lambda}} g_{\Lambda}(X_{\Lambda})^T \mathbf{s}_{\Lambda;\theta}(X_{\Lambda}) + g_{\Lambda}(X_{\Lambda}) \mathbf{s}_{\Lambda;\theta}(X_{\Lambda})] \end{aligned}$$

where (a) is given by Green's Theorem and (b) is given by our limiting condition. Plugging this back into our original result gives

$$\begin{aligned}\mathbb{E}[g_\Lambda(X_\Lambda) \| \mathbf{s}_{\Lambda;\theta}(X_\Lambda) - \mathbf{s}_\Lambda(X_\Lambda) \|^2] &= \mathbb{E}[g_\Lambda(X_\Lambda) (\|\mathbf{s}_{\Lambda;\theta}(X_\Lambda)\|^2 + 2\nabla_{X_\Lambda} \cdot \mathbf{s}_{\Lambda;\theta}(X_\Lambda)) + \nabla_{X_\Lambda} g_\Lambda(X_\Lambda)^\top \mathbf{s}_{\Lambda;\theta}(X_\Lambda)] \\ &=: L_{\text{TM}}(\theta)\end{aligned}$$

From our conditions we also know that θ ∗ is a minimiser of J(θ) hence by our conditions it is the unique minimiser. Therefore θ ∗ is the unique minimiser of LTM(θ)

![](_page_22_Figure_1.jpeg)

Figure 13: Covariance, precision, and marginalisations of the precisions to remove dimension 1 by the naive approach (i.e. subsetting the precision) and the correct approach. All values cube-rooted for contrastive purposes.

## C.1.2. MNAR PROOFS

*Proof of Proposition [A.9.](#page-14-3)* Firstly we have that

$$\mathbb{E} \left[ \| \mathbf{s}_{\Lambda, E_{\Lambda}}(X_{\Lambda}) - \mathbf{s}_{\Lambda, E_{\Lambda}; \theta}(X_{\Lambda}) \|^2 \right] = \mathbb{E} \left[ \| \mathbf{s}_{\Lambda, E_{\Lambda}; \theta}(X_{\Lambda}) \|^2 - 2 \mathbf{s}_{\Lambda, E_{\Lambda}}(X_{\Lambda})^{\top} \mathbf{s}_{\Lambda, E_{\Lambda}; \theta}(X_{\Lambda}) \right] + C$$

where C does not depend upon θ. Examining the second term closer we see that

$$\begin{aligned} & \sum_{\lambda \in \text{supp}(\Lambda)} \int_{\mathcal{X}_\lambda} s_\lambda(\mathbf{x}_\lambda)^\top s_{\lambda;\theta}(\mathbf{x}_\lambda) p_\lambda(\mathbf{x}_\lambda; E_\lambda) d\mathbf{x}_\lambda \\ &= \sum_{\lambda \in \text{supp}(\Lambda)} \sum_{j \in \lambda} \int_{\mathcal{X}_\lambda} s_\lambda(\mathbf{x}_\lambda)_j s_{\lambda,\theta}(\mathbf{x}_\lambda)_j p_\lambda(\mathbf{x}_\lambda; E_\lambda) d\mathbf{x}_\lambda \\ &= \sum_{\lambda \in \text{supp}(\Lambda)} \sum_{j \in \lambda} \int_{\mathcal{X}_\lambda} \nabla_{\mathbf{x}_\lambda} p_\lambda(\mathbf{x}_\lambda; E_\lambda)_j s_{\lambda,\theta}(\mathbf{x}_\lambda)_j d\mathbf{x}_\lambda \\ &= \sum_{\lambda \in \text{supp}(\Lambda)} \sum_{j \in \lambda} - \int_{\mathcal{X}_\lambda} p_\lambda(\mathbf{x}_\lambda; E_\lambda) \frac{\partial}{\partial j} s_{\lambda;\theta}(\mathbf{x}_\lambda)_j d\mathbf{x}_\lambda \quad \text{as a result of integration by parts} \\ &= \mathbb{E}_{X,\Lambda} [\nabla_{X_\Lambda} \cdot s_{\Lambda;\theta}(X_\Lambda)] \end{aligned}$$

![](_page_23_Figure_1.jpeg)

Figure 14: Covariance, precision, and marginalisations of the precisions to remove dimension 1 by the naive approach (i.e. subsetting the precision) and the correct approach. All values cube-rooted for contrastive purposes.

Hence we have

$$\mathbb{E} \left[ \| \mathbf{s}_\Lambda(X_\Lambda) - \mathbf{s}_{\Lambda, E_\Lambda; \theta}(X_\Lambda) \|^2 \right] = \mathbb{E} \left[ \| \mathbf{s}_{\Lambda; \theta}(X_\Lambda) \|^2 + 2 \nabla_{X_\Lambda} \cdot \mathbf{s}_{\Lambda; \theta}(X_\Lambda) \right] + C$$

Hence, just as in Proposition [4.3](#page-3-4) this is minimised when

$$s_\lambda(X_\lambda) = s_{\lambda;\theta}(X_\lambda)$$

a.s. for all λ ∈ supp(Λ). We then have that for any λ ∈ supp(Λ)

$$\begin{aligned} s_\lambda(\mathbf{x}_\lambda) &= s_{\lambda;\theta}(\mathbf{x}_\lambda) \quad \text{for all } x_\lambda \in \mathcal{X}_\lambda \\ p_{\lambda;E_\lambda}(\mathbf{x}_\lambda) &= p_{E_\lambda;\theta}(\mathbf{x}_\lambda) \quad \text{for all } x_\lambda \in \mathcal{X}_\lambda \\ \Leftrightarrow \mathbb{P}(E_\lambda|x_\lambda)p_\lambda(\mathbf{x}_\lambda) &= p_{\lambda;\theta}(\mathbf{x}_\lambda)\mathbb{P}(E_\lambda|x_\lambda) \quad \text{for all } x_\lambda \in \mathcal{X}_\lambda \\ \Leftrightarrow p_\lambda(\mathbf{x}_\lambda) &= p_{\lambda;\theta}(\mathbf{x}_\lambda) \quad \text{for all } x_\lambda \in \mathcal{X}_\lambda \end{aligned}$$

![](_page_24_Figure_1.jpeg)

Figure 15: Mean AUC of various methods for edge detection of star-shaped GGM as we increase the dimension presented alongside 95% confidence intervals.

## C.2. Finite Sample Bound Proofs

Here we give some key results alongside their proof to allow us to obtains finite sample bounds for truncated score matching. This first result is what really underpins our approach.

Proposition C.1. *Let* (Θ, d) *be a compact metric space and for any* δ > 0 *denote the* η*-covering number of* Θ *by* N(η, Θ)*. Now define random functions* Z(θ), Zr(θ)Θ → <sup>R</sup> *for all* r ∈ <sup>N</sup> *(with* r *deterministic) such that for any* θ ∈ Θ

$$\mathbb{P} (|Z_r(\theta) - Z(\theta)| > \varepsilon) \leq \delta(\varepsilon, r)$$

*If we additionally assume that* Zr, Z *Lipschitz with constants* Cr, C *respectively then we have then*

$$\mathbb{P} \left( \sup_{\theta \in \Theta} |Z_r(\theta) - Z(\theta)| > \varepsilon + \eta(C_r + C) \right) \leq N(\eta, \Theta) \delta(\varepsilon, r)$$

*Proof.* Let θ1, . . . , θN(η,Θ) be an η cover of Θ then we have that

$$\begin{aligned} & \sup_{\theta \in \Theta} |Z_r(\theta) - Z(\theta)| \\ & \leq \sup_{\theta \in \Theta} \left\{ \min_{j \in [N(\eta, \Theta)]} \{ |Z_r(\theta) - Z_r(\theta_l)| + |(Z(\theta) - Z(\theta_l)| + |Z_r(\theta_l) - Z(\theta_l)| \} \right\} \\ & \leq \sup_{\theta \in \Theta} \left\{ \min_{j \in [N(\eta, \Theta)]} \{ |Z_r(\theta) - Z_r(\theta_l)| + |Z(\theta) - Z(\theta_l)| \} \left\{ \max_{j \in [N(\eta, \Theta)]} |Z_r(\theta_l) - Z(\theta_l)| \right\} \right\} \\ & \leq \sup_{\theta \in \Theta} \left\{ \min_{j \in [N(\eta, \Theta)]} \{ (C_r + C)|\theta - \theta_l| \} \right\} + \max_{j \in [N(\eta, \Theta)]} \{ |Z_r(\theta_l) - Z(\theta_l)| \} \quad \text{by our Lipschitz condition} \\ & \leq (C_r + C)\eta + \max_{j \in [N(\eta, \Theta)]} \{ |Z_r(\theta_l) - Z(\theta_l)| \} \quad \text{by definition of } \theta_1, \dots, \theta_{N(\eta, \Theta)} \end{aligned}$$

Therefore we have the following relationship between events:

$$\left\{ \sup_{\theta \in \Theta} |Z_r(\theta) - Z(\theta)| > \varepsilon + 2M\eta \right\} \subseteq \left\{ \bigcup_{j \in [N(\eta, \Theta)]} |Z_r(\theta_l) - Z(\theta_l)| > \varepsilon \right\}.$$

![](_page_25_Figure_2.jpeg)

First 4 ROC curves for missing probability 0.5

Figure 16: ROC Curves in 4 separate runs for GGM estimation of a model truncated normal distribution with a star-shaped Precision matrix.

This therefore gives

$$\begin{aligned} & \mathbb{P} \left( \sup_{\theta \in \Theta} \left| \frac{1}{n} f(X^{(i)}, \theta) - \mathbb{E}[f(X, \theta)] \right| > \varepsilon + \eta(C_r + C) \right) \\ & \leq \mathbb{P} \left( \bigcup_{j \in [N(\eta, \Theta)]} |Z_r(\theta_l) - Z(\theta_l)| > \varepsilon \right) \\ & \stackrel{N(\eta, \Theta)}{\leq} \sum_{j=1}^{N(\eta, \Theta)} \mathbb{P}(|Z_r(\theta_l) - Z(\theta_l)| > \varepsilon) \\ & \leq N(\eta, \theta) \delta(\varepsilon, n). \end{aligned}$$

*Remark* C.2*.* This result does not require Cr, C too be deterministic. A feature that we will be exploiting later on.

To be able to say meaningful statements about our Lipschitz bounds for our proof, it will also be helpful to make subgaussian statements about nested sums. We give to results to enable this now

Lemma C.3. *Let* X, Y *be independent RVs and define a function* g : X × Y → R*. Now suppose that for any* x ∈ X *,* g(x, Y ) *is sub-Gaussian with parameter* σ*. We then have that* <sup>E</sup>[g(X, Y )|Y ] *is sub-Gaussian with parameter* σ

*Proof.* As g(x, Y ) sub-Gaussian we have that for any λ > 0

$$\mathbb{E}[\exp\{\lambda(g(\mathbf{x}, Y) - \mathbb{E}[g(\mathbf{x}, Y)])\}] \leq \exp\left\{\frac{\lambda^2}{\sigma^2}\right\}.$$

Our aim is to then use this to bound

$$\mathbb{E} [\exp \{ \lambda (\mathbb{E}[g(X, Y)|Y] - \mathbb{E}[g(X, Y)]) \} ] .$$

We first have that <sup>E</sup>[g(X, Y )] = <sup>E</sup>[E[g(X, Y )|X]] = <sup>E</sup>[E[g(X, Y )|X]|Y ] which in turn gives

$$\begin{aligned}\mathbb{E}[\exp\{\lambda(\mathbb{E}[g(X, Y)|Y] - \mathbb{E}[g(X, Y)])\}] &= \mathbb{E}[\exp\{\lambda(\mathbb{E}[g(X, Y) - \mathbb{E}[g(X, Y)|X]|Y])\}] \\ &\leq \mathbb{E}[\mathbb{E}[\exp\{\lambda(g(X, Y) - \mathbb{E}[g(X, Y)|X])\}|Y]] \quad \text{by Jensen's inequality} \\ &= \mathbb{E}[\mathbb{E}[\exp\{\lambda(g(X, Y) - \mathbb{E}[g(X, Y)|X])\}|X]] \\ &\leq \sup_{\mathbf{x} \in \mathcal{X}} \mathbb{E}[\exp\{\lambda(g(\mathbf{x}, Y) - \mathbb{E}[g(\mathbf{x}, Y)])\}] \quad \text{by independence} \\ &\leq \exp\left\{\frac{\lambda^2}{\sigma^2}\right\}.\end{aligned}$$

Lemma C.4. *Let* X, Y *be independent RVs on spaces* X , Y *and* g : X × Y → <sup>R</sup> *s.t. for any* x ∈ X , y ∈ Y g(x, Y ) *and* g(X, y) *are sub-Gaussian with parameters* σ<sup>Y</sup> , σ<sup>X</sup> *respectively. Let* X(i) n <sup>i</sup>=1 *and* Y (k) r <sup>k</sup>=1 *be IID copies of* X, Y *respectively then*

$$\mathbb{P} \left( \left| \frac{1}{nr} \sum_{i=1}^n \sum_{k=1}^r g(X^{(i)}, Y^{(k)}) - \mathbb{E}[g(X, Y)] \right| > \varepsilon \right) \leq n \exp \left\{ -\frac{\varepsilon^2 \sigma_Y^2 m}{4} \right\} + \exp \left\{ -\frac{\varepsilon^2 \sigma_X^2 n}{4} \right\}$$

*Proof.* Again let W(i) := | 1 r P<sup>r</sup> <sup>i</sup>=1 <sup>g</sup>(X(i) , Y (k) ) − <sup>E</sup>[g(X(i) , Y )|X(i) ]|, We aim to bound W(i) as well as 1 n P<sup>n</sup> <sup>i</sup>=1 <sup>E</sup>[g(X(i) , Y )|X(i) ] − <sup>E</sup>[g(X, Y )]

For W(i) we have that

$$\begin{aligned}\mathbb{P}(W^{(i)} > \varepsilon) &= \mathbb{E}[\mathbb{P}(W^{(i)} > \varepsilon | X^{(i)})] \\ &= \mathbb{E} \left[ \mathbb{P} \left( \left| \frac{1}{r} \sum_{i=1}^r g(X^{(i)}, Y^{(k)}) - \mathbb{E}[g(X^{(i)}, Y) | X^{(i)}] \right| < \varepsilon \mid X^{(i)} \right) \right] \\ &= \mathbb{E} \left[ \exp \left\{ -\frac{\varepsilon^2 \sigma_0 r}{2} \right\} \right] = \exp \left\{ -\frac{\varepsilon^2 \sigma_0 r}{2} \right\}\end{aligned}$$

Therefore we have that

$$\begin{aligned} \mathbb{P}\left(\frac{1}{n} \sum_{i=1}^n W^{(i)} > \varepsilon\right) &\leq \mathbb{P}\left(\bigcup_{i=1}^n \{W^{(i)} > \varepsilon\}\right) \\ &\leq \sum_{i=1}^n \mathbb{P}(W^{(i)} > \varepsilon) \\ &\leq n \exp\left\{-\frac{\varepsilon^2 \sigma_Y r}{2}\right\}. \end{aligned}$$

Additionally from Lemma [C.3](#page-26-0) we have that <sup>E</sup>[g(X, Y )|X] is sub-Gaussian with parameter σ<sup>1</sup> giving us that

$$\mathbb{P} \left( \left| \frac{1}{n} \sum_{i=1}^n \mathbb{E}[g(X^{(i)}, Y|X^{(i)}] - \mathbb{E}[g(X, Y)]] \right| > \varepsilon \right) \leq \exp \left\{ -\frac{\varepsilon^2 \sigma_X^2 n}{2} \right\}$$

Hence combining these we get

$$\mathbb{P} \left( \left| \frac{1}{nr} \sum_{i=1}^n \sum_{k=1}^r g(X^{(i)}, Y^{(k)}) - \mathbb{E}[g(X, Y)] \right| > \varepsilon \right) \leq n \exp \left\{ -\frac{\varepsilon^2 \sigma_Y^2 r}{8} \right\} + \exp \left\{ -\frac{\varepsilon^2 \sigma_X^2 n}{8} \right\}$$

To proceed we define the intermediary step between the population objective and the sample objective this is

$$L_{\text{TM};n}(\theta) := \frac{1}{n} \sum_{i=1}^n \nabla_{X_{\Lambda_i}^{(i)}} \cdot \mathbf{s}_{\Lambda_i;\theta}(X^{(i)}) + \frac{1}{2} \|\mathbf{s}_{\Lambda_i;\theta}(X^{(i)})\|^2$$

Proposition C.5. *For* r ∈ N *let* {X (k)} r <sup>k</sup>=1 *be IID copies of* <sup>X</sup>′ ∼ <sup>p</sup> ′ *and assume that* supp(X) ⊆ supp(X′ )*. For* θ ∈ Θ *and* q<sup>θ</sup> : X → [0, ∞)*, with* ∥qθ∥1< ∞ *define RVs* Yθ, Yθ,r *by*

$$\begin{aligned} Z(\theta) &:= L_{\text{TM};1} = g_\Lambda(X_\Lambda) \left( \nabla_{X_\Lambda} \cdot \mathbf{s}_{\Lambda,\theta}(X) + \frac{1}{2} \|\mathbf{s}_{\Lambda,\theta}(X)\|^2 \right) + \nabla_{X_\Lambda} g_\Lambda(X_\Lambda)^\top \mathbf{s}_{\Lambda,\Theta}(X_\Lambda) \\ Z_r(\theta) &:= L_{\text{TM};1,r}(\theta) = g_\Lambda \left( \text{tr}(\nabla_{X_\Lambda} \hat{\mathbf{s}}_{\Lambda,r;\theta}(X)) + \frac{1}{2} \|\hat{\mathbf{s}}_{\Lambda,r;\theta}(X_\Lambda)\|^2 \right) + \frac{1}{2} \nabla_{X_\Lambda} g_\Lambda(X_\Lambda)^\top \mathbf{s}_{\Lambda,\theta}(X_\Lambda) \end{aligned}$$

*Suppose that the following hold for all* x ∈ X , λ ∈ supp(Λ)

- 0 < a<sup>0</sup> < f0,λ(x, θ) < b<sup>0</sup>
- ∥f1,λ(x, θ)∥< b<sup>1</sup>
- |f2,λ(x, θ)|< b<sup>2</sup>
- gλ(xλ) < c<sup>0</sup>

- ∥∇<sup>x</sup><sup>λ</sup> gλ(xλ)∥< c<sup>1</sup>

*Then we have that*

$$\mathbb{P}(|Z_r(\theta) - Z(\theta)| > \varepsilon) \leq (4 + 2d) \exp \left\{ -\frac{\varepsilon^2 m}{\alpha^2} \right\}$$

*with* α *depending upon* a0, b0, b1, b2, c0, c1*.*

*Proof.* First define

$$Y_l(\theta) := \mathbb{E}[f_{l,\Lambda}(X_\Lambda, X'_{-\Lambda}, \theta) | X_\Lambda, \Lambda] \qquad Y_{l,r}(\theta) := \frac{1}{r} \sum_{k=1}^r f_{l,\Lambda}(X_\Lambda, X'_{-\Lambda}^{(k)}).$$

Using the definition of the marginal and estimated marginal score we then have that

$$Z_r(\theta) = g_\Lambda(X_\Lambda) \frac{Y_2(\theta)}{Y_0(\theta)} + \frac{1}{2} \frac{\|Y_1(\theta)\|^2}{Y_0(\theta)^2} + \frac{1}{2} \nabla_{X_\Lambda} g_\Lambda(X_\Lambda)^\top \frac{Y_1(\theta)}{Y_0(\theta)}$$

$$Z_r(\theta) = g_\Lambda(X_\Lambda) \left( \frac{Y_2_r(\theta)}{Y_0_r(\theta)} + \frac{1}{2} \frac{\|Y_{1,r}(\theta)\|^2}{Y_0_{r,r}(\theta)^2} \right) + \frac{1}{2} \nabla_{X_\Lambda} g_\Lambda(X_\Lambda)^\top \frac{Y_1(\theta)}{Y_0(\theta)}.$$

We can therefore write write |Zr(θ) − Z(θ)| as

$$\begin{aligned} |Z_{r,r}(\theta) - Z(\theta)| &= g_{\Lambda}(X_{\Lambda}) \left| \frac{Y_{2,r}(\theta)}{Y_{0,r}(\theta)} + \frac{1}{2} \frac{\|Y_{1,r}(\theta)\|^2}{Y_{0,r}(\theta)^2} - \frac{Y_2(\theta)}{Y_0(\theta)} - \frac{1}{2} \frac{\|Y_3(\theta)\|^2}{Y_2(\theta)^2} \right| + \left| \frac{\nabla_{X_{\Lambda}} g_{\Lambda}(X_{\Lambda}) Y_1(\theta)}{Y_0(\theta)} - \frac{\nabla_{X_{\Lambda}} g_{\Lambda}(X_{\Lambda}) Y_1(\theta)}{Y_{0,r}(\theta)} \right| \\ &\leq \frac{|Y_{2,r}(\theta) - Y_2(\theta)|}{Y_0(\theta)} + \frac{|Y_{0,r}(\theta) - Y_0(\theta)| |Y_2(\theta)|}{Y_0(\theta) Y_{0,r}(\theta)} \\ &\quad + \frac{1}{2} \frac{\|Y_{1,r}(\theta)\|^2 - \|Y_1(\theta)\|^2}{Y_0(\theta)^2} + \frac{1}{2} \frac{|Y_{0,r}(\theta)^2 - Y_0(\theta)^2| \|Y_{1,r}(\theta)\|^2}{Y_0(\theta)^2 Y_{0,r}(\theta)^2} \\ &\quad + \frac{|\nabla_{X_{\Lambda}} g_{\Lambda}(X_{\Lambda})^{\top} (Y_{1,r}(\theta) - Y_1(\theta))|}{Y_0(\theta)} + \frac{|Y_{0,r}(\theta) - Y_0(\theta)| |\nabla_{X_{\Lambda}} g_{\Lambda}(X_{\Lambda})^{\top} Y_1(\theta)|}{Y_0(\theta) Y_{0,r}(\theta)} \\ &\leq \frac{|Y_{2,r}(\theta) - Y_2(\theta)|}{a_0} + \frac{|Y_{0,r}(\theta) - Y_0(\theta)| b_2}{a_0^2} \\ &\quad + \frac{1}{2} \frac{\|Y_{1,r}(\theta)\|^2 - \|Y_1(\theta)\|^2}{a_0^2} + \frac{1}{2} \frac{|Y_{0,r}(\theta)^2 - Y_0(\theta)^2| b_1^2}{a_0^4} \\ &\quad + \frac{\|Y_{1,r}(\theta) - Y_1(\theta)\| c_1}{a_0} + \frac{|Y_{0,r}(\theta) - Y_0(\theta)| c_1 b_1}{a_0^2} \\ &\leq \frac{|Y_{2,r}(\theta) - Y_2(\theta)|}{a_0} + \frac{|Y_{0,r}(\theta) - Y_0(\theta)| b_2}{a_0^2} \\ &\quad + \frac{1}{2} \frac{\|Y_{1,r}(\theta) - Y_1(\theta)\|^2 a_1}{a_0^2} + \frac{1}{2} \frac{|Y_{0,r}(\theta) - Y_0(\theta)| b_0 b_1^2}{a_0^4} \\ &\quad + \frac{\|Y_{1,r}(\theta) - Y_1(\theta)\| c_1}{a_0} + \frac{|Y_{0,r}(\theta) - Y_0(\theta)| c_1 b_1}{a_0^2} \end{aligned}$$

Now if we define the events

$$\begin{aligned} E_0(\theta) &:= \left\{ |Y_{0,r}(\theta) - Y_0(\theta)| > \varepsilon \left( \frac{a_0^2}{6b_2} \wedge \frac{a_0^4}{3b_0 b_1^2} \wedge \frac{a_0^2}{6c_1 b_1} \right) \right\} \\ E_1(\theta) &:= \left\{ \|Y_{1,r}(\theta) - Y_1(\theta)\| > \varepsilon \left( \frac{a_0^2}{3b_1} \wedge \frac{a_0}{6c_1} \right) \right\} \\ E_2(\theta) &:= \left\{ |Y_{2,r}(\theta) - Y_2(\theta)| > \varepsilon \frac{a_0}{6} \right\} \end{aligned}$$

then we have that

$$\{|Z_r(\theta) - Z(\theta)| < \varepsilon\} \subseteq \bigcup_{l=1}^3 E_k(\theta)$$

Using standard Hoeffding bounds for E0(θ), E2(θ) and union bounds in conjunction with Hoeffding bounds for E1(θ) we get that

$$\begin{aligned}\mathbb{P}(E_0(\theta)) &\geq 1 - 2 \exp \left\{ -\varepsilon^2 r \left( \frac{a_0^4}{36b_0^2 b_2^2} \wedge \frac{a_0^4}{9b_4^4 b_1^4} \wedge \frac{a_0^4}{36c_1^2 b_1^2 b_2^2} \right) \right\} \\ \mathbb{P}(E_1(\theta)) &\geq 1 - 2d \exp \left\{ -\varepsilon^2 r \left( \frac{a_0^4}{9b_1^4} \wedge \frac{a_0^2}{36c_1^2 b_1^2} \right) \right\} \\ \mathbb{P}(E_2(\theta)) &\geq 1 - 2 \exp \left\{ -\varepsilon^2 r \frac{a_0^2}{36b_2^2} \right\}.\end{aligned}$$

Hence

$$\mathbb{P}(|Z_r(\theta) - Z(\theta)| > \varepsilon) \leq (4 + 2d) \exp \left\{ -\frac{\varepsilon^2 m}{\alpha^2} \right\} \quad \text{with} \quad \alpha := \max \left\{ \frac{6b_0 b_2}{a_0^2}, \frac{3b_0^2 b_1^2}{a_0^4}, \frac{6c_1 b_1 b_0}{a_0^2}, \frac{3b_1^2}{a_0^2}, \frac{6c_1 b_1}{a_0}, \frac{6b_2}{a_0} \right\}.$$

*proof of Theorem [A.15.](#page-15-2)* Our strategy will be as follows:

- Use our bound on |Zr(θ) − Z(θ)| from Proposition [C.5](#page-27-0) to bound |LTM;n,r(θ) − Ln(θ)|.
- Use a covering number argument alongside Lemma [C.4](#page-26-1) to bound sup<sup>θ</sup> |LTM;n,r(θ) − Ln(θ)|.
- Use a similar approach to bound sup<sup>θ</sup> |Ln(θ) − LTM(θ)|.
- Combine these to bound |LTM;n,r(θ) − LTM(θ)|

For the first step we have that

$$\begin{aligned}\mathbb{P}(|L_{\text{TM};n,r}(\theta) - L_{\text{TM};n}(\theta)| > \varepsilon) &\leq \mathbb{P}\left(\bigcup_{i=1}^n |L_{1,r}^{(i)}(\theta) - L_1^{(i)}(\theta)| > \varepsilon\right) \\ &\leq \sum_{i=1}^n \mathbb{P}(|L_{1,r}(\theta) - L_1(\theta)| > \varepsilon) \\ &= \sum_{i=1}^n \mathbb{P}(|Z_r(\theta) - Z(\theta)| > \varepsilon) \\ &= n(4 + 2d) \exp \left\{ -\frac{\varepsilon^2 m}{\alpha^2} \right\}\end{aligned}$$

where

$$\begin{aligned} L_{1,r}^{(i)}(\theta) &= g_{\Lambda_i}(X_{\Lambda_i}^{(i)}) \left( \frac{\frac{1}{r} \sum_{k=1}^r f_{2,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)}, \theta)}{\frac{1}{r} \sum_{k=1}^r f_{0,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)}, \theta)} + \frac{1}{2} \frac{\left\| \frac{1}{r} \sum_{k=1}^r f_{1,\Lambda}(X_{\Lambda_i}, X_{-\Lambda}^{'(k)}, \theta) \right\|^2}{\left( \frac{1}{r} \sum_{k=1}^r f_{0,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)}, \theta) \right)^2} \right) \\ &\quad + \frac{\frac{1}{n} \sum_{i=1}^n g_{\Lambda_i}(X_{\Lambda}^{(i)})^{\top} f_1(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)})}{\frac{1}{n} \sum_{i=1}^n f_0(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)})} \\ L_1^{(i)}(\theta) &= g_{\Lambda_i}(X_{\Lambda_i}^{(i)}) \left( \frac{\mathbb{E} \left[ f_{2,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'}, \theta) \middle| X_{\Lambda}^{(i)} \right]}{\mathbb{E} \left[ f_{0,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'}, \theta) \middle| X_{\Lambda}^{(i)} \right]} + \frac{1}{2} \frac{\left\| \mathbb{E} \left[ f_{1,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'}, \theta) \middle| X_{\Lambda}^{(i)} \right] \right\|^2}{\mathbb{E} \left[ f_{0,\Lambda}(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'}, \theta) \middle| X_{\Lambda}^{(i)} \right]^2} \right) \\ &\quad + \frac{\mathbb{E}[g_{\Lambda_i}(X_{\Lambda}^{(i)})^{\top} f_1(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)}) | X_{\Lambda}^{(i)}]}{\mathbb{E}[f_0(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)}) | X_{\Lambda}^{(i)}]}. \end{aligned}$$

We now try and derive a Lipschitz constant for both Ln,r and Ln. Define g : X → <sup>R</sup> by

$$g(\mathbf{x}) = \frac{1}{a_0} M_2(\mathbf{x}) + \left( \frac{b_2}{a_0^2} + \frac{b_1^2}{2a_0^4} + \frac{c_1 b_1}{a_0^2} \right) M_0(\mathbf{x}) + \left( \frac{a_1}{2a_0^2} + \frac{c_1}{a_0} \right) M_1(\mathbf{x})$$

Then we have that For Ln,r we have

$$|L_{\text{TM};n,r}(\theta) - L_{n,r}(\theta')| \leq \rho(\theta, \theta') \underbrace{\frac{1}{nr} \sum_{i=1}^n \sum_{k=1}^r g(X_{\Lambda}^{(i)}, X_{-\Lambda}^{'(k)})}_{:=\mathcal{C}_{n,r}}$$

similarly we have

$$|L_{\text{TM};n}(\theta) - L_n(\theta')| \leq \rho(\theta, \theta') \underbrace{\frac{1}{n} \sum_{i=1}^n \mathbb{E}[g(X_{\Lambda}^{(i)}, X'_{-\Lambda}) | X_{\Lambda}^{(i)}]}_{:=C_n}.$$

Using an identical argument to Proposition [C.1](#page-24-2) can get the following bound for any η<sup>1</sup> > 0:

$$\mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM}}(\theta) - L_{\text{TM};n,r}(\theta)| > \varepsilon_1 + \eta_1(C_{n,r} + C_n)) > N(\eta_1, \Theta) n(4 + 2d) \exp \left\{ -\frac{\varepsilon^2 m}{\alpha^2} \right\}$$

Hence we now need to bound C<sup>n</sup> and Cn,r. We know that both of these terms converge to C := <sup>E</sup>[g(XΛ, X′ Λ )]. To obtain rates on this convergence we make sub-Gaussian statements about g. Specifically for x−<sup>λ</sup> ∈ X we have that g(Xλ, x−λ) is sub-Gaussian with parameter

$$\sigma_\lambda := \frac{1}{a_0}\sigma_{2,\lambda} + \left( \frac{b_2}{a_0^2} + \frac{b_1^2}{2a_0^4} + \frac{c_1b_1}{a_0^2} \right) \sigma_{0,\lambda} + \left( \frac{a_1}{2a_0^2} + \frac{c_1}{a_0} \right) \sigma_{1,\lambda}$$

We can therefore immediately bound C<sup>n</sup> − C using Lemma [C.3](#page-26-0) and Hoeffding bounds to get

$$\begin{aligned} \mathbb{P}(C_n - C > \varepsilon) &= \mathbb{E}[\mathbb{P}(C_n - C > \varepsilon | \Lambda)] \\ &\leq \mathbb{E} \left[ \exp \left\{ -\frac{\varepsilon^2 n}{8\sigma_{\Lambda}^2} \right\} \right] \\ &\leq \exp \left\{ -\frac{\varepsilon^2 n}{8\sigma^2} \right\}. \end{aligned}$$

where σ := maxΛ∈supp(Λ) σΛ. To bound Cn,r − C we can use Lemma [C.4](#page-26-1) to get

$$\begin{aligned}\mathbb{E}[\mathbb{P}(C_{n,r} - C > \varepsilon)|\Lambda] &= \mathbb{E}\left[\mathbb{P}\left(\frac{1}{nr} \sum_{i=1}^n \sum_{k=1}^r g(X^{(i)}, Y^{(k)}) - \mathbb{E}[g(X, Y)] > \varepsilon \middle| \Lambda\right)\right] \\ &\leq \mathbb{E}\left[n \exp\left\{-\frac{\varepsilon^2 m}{8\sigma_{-\Lambda}^2}\right\} + \exp\left\{-\frac{\varepsilon^2 n}{8\sigma_{\Lambda}^2}\right\}\right] \\ &= n \exp\left\{-\frac{\varepsilon^2 m}{8\sigma_{'}^2}\right\} + \exp\left\{-\frac{\varepsilon^2 n}{8\sigma^2}\right\}\end{aligned}$$

with σ ′ , σ′ −λ define identically to σ, σ<sup>λ</sup> with σl,λ replaced with σ ′ l,−λ and σ<sup>λ</sup> replaced with σ ′ −λ . As a result we get that

$$\begin{aligned} & \mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM};n,r}(\theta) - L_n(\theta)| > \varepsilon_1 + 2\eta_1(C + \varepsilon)) \\ & < N(\eta_1, \Theta) n(4 + 2d) \exp \left\{ -\frac{\varepsilon^2 m}{\alpha^2} \right\} + n \exp \left\{ -\frac{\varepsilon^2 m}{8\sigma'^2} \right\} + 2 \exp \left\{ -\frac{\varepsilon^2 n}{8\sigma^2} \right\} \end{aligned}$$

Now we have the bound on sup<sup>θ</sup> |LTM;n,r(θ) − LTM(θ)|. We aim to bound sup<sup>θ</sup> |LTM;n(θ) − LTM(θ)|. To that end we have

$$|L_1(\theta)| < \frac{b_2}{a_0} + \frac{1}{2} \frac{b_1^2}{a_0^2} + \frac{c_1 b_1}{a_0} =: \alpha'$$

a.s. and so we can use Hoeffding bounds to get

$$\mathbb{P}(|L_{\text{TM};n}(\theta) - L_{\text{TM}}(\theta)| > \varepsilon) \leq \exp \left\{ -\frac{\varepsilon^2 n}{2\alpha'} \right\}.$$

Again using an argument similar to Proposition [C.1](#page-24-2) we have for any η<sup>2</sup> > 0

$$\mathbb{P}(\sup_{\theta \in \Theta} |L_n(\theta) - L_{\text{TM}}(\theta)| > \varepsilon_2 + \eta_2(2C + \varepsilon_3) \leq N(\eta_2, \Theta) \exp \left\{ -\frac{\varepsilon_2^2 n}{2\alpha'^2} \right\} + \mathbb{P}(C_n - C > \varepsilon_3)$$

Combining these two results gives that for any ε > 0

$$\begin{aligned}
& \mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM};n,r}(\theta) - L_{\text{TM};n,r}(\theta)| \geq \varepsilon_1 + \varepsilon_2 + \eta_1(2C + 2\varepsilon_3) + \eta_2(2C + \varepsilon_3)) \\
& \leq \mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM}}(\theta) - L_n(\theta)| \geq \varepsilon_1 + 2\eta_2(C + \varepsilon_3) \cup \sup_{\theta \in \Theta} |L_{\text{TM};n}(\theta) - L_{\text{TM};n,r}(\theta)| \geq \varepsilon_2 + \eta_2(2C + \varepsilon_3)) \\
& \leq \mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM}}(\theta) - L_n(\theta)| \varepsilon_1 + 2\eta_1(C + \varepsilon_3)) + \mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM};n}(\theta) - L_{\text{TM};n,r}(\theta)| \geq \varepsilon_2 + \eta_2(2C + \varepsilon_3)) \\
& \leq N(\eta_1, \Theta) n(4 + 2d) \exp \{-\varepsilon_1^2 m \alpha^2\} + N(\eta_2, \Theta) \left( \exp \left\{ -\frac{\varepsilon_2^2 n}{2\alpha'} \right\} \right) + \mathbb{P}(C_n - C > \varepsilon_3) + \mathbb{P}(C_{n,r} - C > \varepsilon_3) \\
& \leq N(\eta_1, \Theta) n(4 + 2d) \exp \{-\varepsilon_1^2 m \alpha^2\} + N(\eta_2, \Theta) \left( \exp \left\{ -\frac{\varepsilon_2^2 n}{2\alpha'} \right\} \right) + n \exp \left\{ -\frac{\varepsilon_3^2 m}{8\sigma'^2} \right\} + 2 \exp \left\{ -\frac{\varepsilon_3^2 n}{8\sigma^2} \right\} \quad (22)
\end{aligned}$$

Take η<sup>1</sup> = 1/r and η<sup>2</sup> = 1/n so that for sufficiently large n, r N(η1, Θ) ≤ exp {p log ((3/2) diam(Θ)r)} and N(η2, Θ) = exp {p log ((3/2) diam(Θ)n)}. Plugging this into [\(22\)](#page-31-0) gives

$$\begin{aligned} & \mathbb{P} \left( \sup_{\theta \in \Theta} |L_{\text{TM}}(\theta) - L_{\text{TM};n,r}(\theta)| \geq \varepsilon_1 + \varepsilon_2 + 1/r(2C + 2\varepsilon_3) + 1/n(2C + \varepsilon_3) \right) \\ & \leq n(4 + 2d) \exp \left\{ p \log \frac{3}{2} \text{diam}(\Theta) r - \varepsilon_1^2 n \alpha \right\} + \exp \left\{ p \log \frac{3}{2} \text{diam}(\Theta) n - \frac{\varepsilon_1^2 n}{2\alpha'} \right\} + (2 + n) \exp \left\{ -\frac{\varepsilon_1^2 n}{8(\sigma \vee \sigma')^2} \right\} \end{aligned}$$

Now if we take each of these terms to be equal to δ/3 gives for sufficiently large n, r

$$\mathbb{P}\left(\sup_{\theta \in \Theta} |L_{\text{TM}}(\theta) - L_{\text{TM};n,r}(\theta)| \geq \beta_1 \sqrt{\frac{p \log(dnr \operatorname{diam}(\Theta)/\delta)}{r}} + \beta_2 \sqrt{\frac{p \log(n \operatorname{diam}(\Theta)/\delta)}{n}} + \beta_3 \left(\frac{n+r}{nr}\right) \left(C + \sqrt{\frac{\log(n/\delta)}{n}}\right)\right) < \delta.$$

where β<sup>1</sup> = 27 α , β<sup>2</sup> = 9α ′ , β<sup>3</sup> = 10(σ ∨ σ ′ ). As there exists θ <sup>∗</sup> ∈ Θ a minimiser of LTM(θ) we now have that

$$|L(\theta_n) - L(\theta^*)| \leq |L(\theta_n) - L_n(\theta_n)| + |L_n(\theta^*) - L(\theta^*)|.$$

Finally we know that JTM(θ) = LTM(θ) + C and under a correctly specified model L(θ ∗ ) = 0. Therefore we have that

$$\mathbb{P}(|J(\theta_{n,r})| > 2\varepsilon) \leq \mathbb{P}(\sup_{\theta \in \Theta} |L_{\text{TM};n}(\theta) - L_{\text{TM}}(\theta)| > \varepsilon)$$

and so we have our result simply replacing β<sup>k</sup> with 2βk.

#### C.3. Gradient First Proofs

*Proof of Lemma [4.8.](#page-4-2)* First we have that for any λ, xλ,

$$\begin{aligned} s_{\theta,\lambda}(\mathbf{x}_\lambda) &= \nabla_{\mathbf{x}_\lambda} \log \int p_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} \\ &= \frac{\nabla_{\mathbf{x}_\lambda} \int p_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda}}{\int p_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda}} \\ &= \frac{\int \nabla_{\mathbf{x}_\lambda} p_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda}}{p(\mathbf{x}_{-\lambda})} \\ &= \int \frac{p_\theta(\mathbf{x})}{p_\theta(\mathbf{x}_{-\lambda})} \nabla_{\mathbf{x}_\lambda} \log p_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} \\ &= \mathbb{E}_{\mathbf{x}_{-\lambda} | \mathbf{x}_\lambda; \theta}[s_\theta(\mathbf{x})_\lambda]. \end{aligned}$$

Taking expectations on both side w.r.t. (Λ, XΛ) proves equation [\(8\)](#page-4-4).

For [\(9\)](#page-4-5) we have again for any λ, xλ,

$$\begin{aligned}\nabla \mathbb{E}'_{X'_{-\lambda} \sim p_\theta}[g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] &= \int \nabla(p_\theta(\mathbf{x}_{-\lambda}|\mathbf{x}_\lambda)g_\theta(\mathbf{x})) d\mathbf{x}_{-\lambda} \\ &= \int p_\theta(\mathbf{x}_{-\lambda}|\mathbf{x}_\lambda) \nabla g_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} + \int \nabla p_\theta(\mathbf{x}_{-\lambda}|\mathbf{x}_\lambda) g_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} \\ &= \int p_\theta(\mathbf{x}_{-\lambda}|\mathbf{x}_\lambda) \nabla g_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} + \int p_\theta(\mathbf{x}_{-\lambda}|\mathbf{x}_\lambda) \nabla \log p_\theta(\mathbf{x}_{-\lambda}|\mathbf{x}_\lambda) g_\theta(\mathbf{x}) d\mathbf{x}_{-\lambda} \\ &= \mathbb{E}'[\nabla g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] + \mathbb{E}'[\nabla \log p_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &\quad - \mathbb{E}'[\nabla \log p_\theta(\mathbf{x}_\lambda) g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &= \mathbb{E}'[\nabla g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] + \mathbb{E}'[\nabla \log p_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &\quad - s_{\theta;\lambda}(\mathbf{x}_\lambda) \mathbb{E}'[g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &= \mathbb{E}'[\nabla g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] + \mathbb{E}'[\nabla \log p_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &\quad - \mathbb{E}'[\nabla \log p_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \mathbb{E}'[g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &= \mathbb{E}'[\nabla g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] + \text{Cov}_{p_\theta}(s_\theta(\mathbf{x}_\lambda, X'_{-\lambda}), g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})).\end{aligned}$$

*Proof of Corollary [4.9.](#page-5-2)* First define

$$L_M(\theta; \mathbf{x}_\lambda) := \sum_{j \in \lambda} 2 \partial_j \mathbf{s}_{\lambda; \theta}(\mathbf{x}_\lambda)_j + \mathbf{s}_{\lambda; \theta}(\mathbf{x}_\lambda)_j^2$$

so that <sup>E</sup>[LM(θ; XΛ)] = LM(θ). Then using our two score identities, [\(8\)](#page-4-4) & [\(9\)](#page-4-5) we can re-write L(θ; x) as

$$\begin{aligned} L_M(\theta; \mathbf{x}) &= \sum_{j \in \lambda} \mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]^2 + 2\partial_j \mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j] \quad \text{by (8)} \\ &= \sum_{j \in \lambda} \mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]^2 + \mathbb{E}'[\partial_j \mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j] + \text{Var}(s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j) \quad \text{by (9)} \\ &= \sum_{j \in \lambda} -\mathbb{E}'[s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]^2 + 2\mathbb{E}'[s_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j^2] + 2\mathbb{E}'[\partial_i s_\theta(\mathbf{x}_\lambda, x_{-\lambda})_j] \end{aligned}$$

Where all expectations are w.r.t. X′ −λ |X<sup>λ</sup> = xλ; θ Now using the fact that

$$\nabla_{\theta} \left( \mathbb{E}'[s_{\theta}(\mathbf{x}_{\lambda}, X'_{-\lambda})_j] \right)^2 = 2E[s_{\theta}(\mathbf{x}_{\lambda}, X'_{-\lambda})_j] \nabla_{\theta} \mathbb{E}'[s_{\theta}(\mathbf{x}_{\lambda}, X'_{-\lambda})_j]$$

and using [\(9\)](#page-4-5) again on each term of the above gives followed by taking expectations w.r.t. Λ, X<sup>Λ</sup> gives the result.

# C.3.1. TRUNCATED SCORE MATCHING

*Proof of Corollary [4.11.](#page-5-3)* Now using Lemma [4.8,](#page-4-2) we know that this can be re-written as

$$\begin{aligned} & \sum_{j \in \lambda} g_\lambda(\mathbf{x}_\lambda)_j (\mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]^2 + 2 \operatorname{Var}'(\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j) + 2 \mathbb{E}'[\partial_j \mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]) + 2 \partial_j g(\mathbf{x}_\lambda)_j \mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \\ &= \sum_{j \in \lambda} g_\lambda(\mathbf{x}_\lambda)_j (-\mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]^2 + 2 \mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j^2] + 2 \mathbb{E}'[\partial_j \mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})_j]) + 2 \partial_j g(\mathbf{x}_\lambda)_j \mathbb{E}'[\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \end{aligned}$$

Now taking gradient w.r.t. θ we get

$$\begin{aligned} \nabla_{\theta} L(\theta; \mathbf{x}_{\lambda}) &= \sum_{j \in \lambda} g_{\lambda}(\mathbf{x}_{\lambda})_j \left\{ -2\mathbb{E}'[\mathbf{s}_{\theta}(\mathbf{x})_j] (\mathbb{E}'[\nabla_{\theta} \mathbf{s}_{\theta}(\mathbf{x})_j] + \text{Cov}'(\nabla_{\theta} \log q_{\theta}(\mathbf{x}), \mathbf{s}_{\theta}(\mathbf{x})_j)) \right. \\ &\quad \left. + 2(\mathbb{E}'[\nabla_{\theta} \mathbf{s}_{\theta}(\mathbf{x})_j^2] + \text{Cov}'(\nabla_{\theta} \log q_{\theta}(\mathbf{x}), \mathbf{s}_{\theta}(\mathbf{x})_j^2)) \right. \\ &\quad \left. + 2(\mathbb{E}'[\nabla_{\theta} \partial_j \mathbf{s}_{\theta}(\mathbf{x})_j] + \text{Cov}'(\nabla_{\theta} \log q_{\theta}(\mathbf{x}), \partial_j \mathbf{s}_{\theta}(\mathbf{x})_j)) \right\} \\ &\quad + 2\partial_j g(\mathbf{x}_{\lambda})_j \left\{ \mathbb{E}'[\nabla_{\theta} \mathbf{s}_{\theta}(\mathbf{x})_j] + \text{Cov}'(\nabla_{\theta} \log q_{\theta}(\mathbf{x}), \mathbf{s}_{\theta}(\mathbf{x})_j) \right\}. \end{aligned}$$

# C.4. IW and Gradient First Relationships

*Proof of Lemma [A.18.](#page-16-0)* We first have that,

$$\begin{aligned}\hat{s}_{\theta;\lambda}(\mathbf{x}_\lambda) &= \nabla_{\mathbf{x}_\lambda} \log \frac{1}{r} \sum_{i=1}^r w_i \\ &= \frac{\frac{1}{r} \sum_{i=1}^r \nabla_{\mathbf{x}_\lambda} w_i}{\frac{1}{r} \sum_{i=1}^r w_i} \\ &= \frac{1}{r} \sum_{i=1}^r \bar{w}_i \nabla_{\mathbf{x}_\lambda} \log q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(i)}) \\ &= \hat{\mathbb{E}}_{iw} [\nabla_{\mathbf{x}_\lambda} \log q_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(i)})] = \hat{\mathbb{E}}_{iw} [\mathbf{s}_\theta(\mathbf{x}_\lambda, X'_{-\lambda}^{(i)})_\lambda].\end{aligned}$$

Where the penultimate result uses the fact that ∇<sup>x</sup><sup>λ</sup> w<sup>i</sup> = w<sup>i</sup> log <sup>q</sup>θ(xλ, X′(i) −λ )

Additionally,

$$\nabla_{\mathbf{x}_\lambda} hat E_{iv}[g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] = \frac{\nabla_{\mathbf{x}_\lambda} \sum_{k=1}^n w_k g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})}{\sum_{k=1}^r w_k} - \frac{(\nabla_{\mathbf{x}_\lambda} \sum_{k=1}^r w_k) \left( \sum_{k=1}^n w_k g_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) \right)}{(\sum_{k=1}^r w_k)^2}$$

For the second term we can again use ∇<sup>x</sup><sup>λ</sup> w<sup>k</sup> = wk∇<sup>x</sup><sup>λ</sup> logpθ(xλ, X′(k) −λ ) to write this as

$$\hat{\mathbb{E}}_{iw}[\nabla_{\boldsymbol{x}_\lambda} \log p_\theta(\boldsymbol{x}_\lambda, X'_{-\lambda})] \mathbb{E}_{iw}[g_\theta(\boldsymbol{x}_\lambda, X'_{-\lambda})]$$

The first term we can write as

$$\begin{aligned} & \frac{\sum_{i=1}^n w_k \log p_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})}{\sum_{i=1}^r w_k} + \frac{\sum_{i=1}^n w_k \nabla_{\mathbf{x}_\lambda} g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})}{\sum_{i=1}^r w_k} \\ &= \hat{\mathbb{E}}_{iw} [\log p_\theta(\mathbf{x}_\lambda, X'_{-\lambda}) g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] + \hat{\mathbb{E}}_{iw} [\nabla_{\mathbf{x}_\lambda} g_\theta(\mathbf{x}_\lambda, X'_{-\lambda})] \end{aligned}$$

Combining these 2 gives our desired results.

*Proof of Corollary [A.19.](#page-16-1)* The exact same arguments give us the second result but with the importance weighting identities [\(19\)](#page-16-2) and [\(20\)](#page-16-3) replacing [\(8\)](#page-4-4) & [\(9\)](#page-4-5).

# D. Additional Score Matching Details & Extensions

In this section we give some additional details on score matching and introduce some pre-existing score matching extensions and methods. For some of these approach we can adapt our method to work with them while others act as comparison points for our approach.

# D.1. Classical Score Matching

The assumptions for the classical score matching result presented in [\(1\)](#page-2-1)

Assumption D.1. For

- (a) The pdf p(x) is differentiable w.r.t. x;
- (b) Our score estimating function s<sup>θ</sup> is differentiable w.r.t. x;
- (c) <sup>E</sup>[∥sθ(X)∥ 2 ],<sup>E</sup>[∥s(X)∥ 2 ] < ∞;
- (d) p(x)sθ(x) −→ 0 whenever ∥x∥−→ ∞.

# D.2. Zeroed Score Matching and MissDiff

In this section we take x 0 λ to be the vector in R <sup>d</sup> with (x 0 λ )<sup>i</sup> = x<sup>i</sup> if i ∈ λ and 0 otherwise and then take X<sup>0</sup> λ to be the RV equivalent.

MissDiff is a generative modelling technique that aims to learn generative models from corrupted tabular data using diffusion models with denoising score matching. Throughout their method s<sup>θ</sup> is assumed to be a neural network. The core idea is to replace the standard score matching objective with

$$\mathbb{E}_{t, X_\Lambda(0), X_\Lambda(t)} [\mu(t) \| \mathbf{s}(X_\Lambda^0(t), t)_\Lambda - \nabla_{X_\Lambda(t)} \log p(X_\Lambda(t) | X_\Lambda(0))].$$

Essentially, missing values are zero imputed and then the score is tested only on output dimensions whose input dimensions are non-missing. The idea of this approach is that it trains sθ( 0 λ (t), t)<sup>λ</sup> to approximate the true marginal score sλ;<sup>θ</sup>(xλ(t), t) which in turn encourages sθ(x(t), t) to approximate the full score sθ(x(t), t).

For comparison purposes, we adapt this in two ways to create Zeroed Score Matching. Firstly we change the objective to standard score matching and secondly we no longer require s to be a neural network. Thus our adapted version of the MissDiff objective is

$$\hat{L}_{\text{Zeroed}}(\theta) = \mathbb{E} \left[ \nabla_{(X_{\Lambda}^0)_{\Lambda}} \cdot \mathbf{s}_{\theta}(X_{\Lambda}^0)_{\Lambda} + \|\mathbf{s}_{\theta}(X_{\Lambda}^0)_{\Lambda}\| \right]. \quad (23)$$

The key issue with this approach is that with s<sup>θ</sup> no longer necessarily a neural network it is not reasonable (or indeed even possible) for s<sup>θ</sup> to model both the joint and marginal scores. In other words we cannot expect both sθ(x 0 λ ) = sλ(xλ) and sθ(x) = s(x) making it a naive marginalisation method for the score.

We can give a brief example with a multidimensional normal distribution. Supposed that X ∼ N(0, Σ) with P := Σ−<sup>1</sup> . Then the score is s(x) = −Px which we would model with sθ(x) = −Pθx. Under the Zeroed scheme, we would take the marginal score to be

$$\begin{aligned} s_{\lambda,\theta}(\mathbf{x}) &= (-P\mathbf{x}_\lambda^0)_\lambda \\ &= -P_{\lambda,\lambda;\theta}\mathbf{x}_\lambda. \end{aligned}$$

However we know that if X ∼ N(0, Σθ) with Σ<sup>θ</sup> = P −1 θ then X<sup>λ</sup> ∼ N(0, Σλ,λ;<sup>θ</sup>) and so we should take

$$s_{\lambda,\theta}(\mathbf{x}) = -(\Sigma_{\lambda,\lambda;\theta})^{-1}\mathbf{x}_\lambda = -((P_\theta^{-1})_{\lambda,\lambda})^{-1}\mathbf{x}_\lambda$$

and crucially, Pλ,λ ̸= ((P −1 )λ,λ) <sup>−</sup><sup>1</sup> unless P = I.

We illustrate this for our simulated experimental settings in Appendices [B.1.2](#page-18-1) & [B.2.1.](#page-19-1)

We also explore the implications of this for the marginal Fisher divergence for the normal in Appendix [A.4.](#page-16-4)

## D.3. Sliced Score Matching

One issue with score matching is that ∇xcdotsθ(x) is computationally expensive to compute for large d. A solution to this was proposed in [Song et al.](#page-10-7) [\(2020\)](#page-10-7) where, instead of testing the full score 1-dimensional projections/slices are tested instead. This is done by introducing another RV V on (X , B<sup>X</sup> ) satisfying <sup>E</sup>[V ] = 0 and <sup>E</sup>[V V <sup>⊤</sup>] positive semi-definite.

The original objective is then

$$F_S(\theta) := \mathbb{E} \left[ \left\{ V^\top (s(X) - s_\theta(X)) \right\}^2 \right]$$

Which then leads to the following equivalence

$$\begin{aligned} L_S(\theta) &:= \mathbb{E}[2 \{ \nabla_X (V^\top \mathbf{s}_\theta(X)) \}^\top V + (V^\top \mathbf{s}_\theta(X))^2] \\ &= F_S - C. \end{aligned}$$

which is less computationally expensive w.r.t. d and can be approximated with samples of X and V . For the proof of this results and the precise conditions see [Song et al.](#page-10-7) [\(2020\)](#page-10-7).

#### D.4. Denoised Score matching

Denoised score matching is another adaptation which removes the need to takes derivatives of the score all together [\(Vin](#page-10-6)[cent, 2011\)](#page-10-6). As we will see later on it is also the method used most prominently in diffusion processes.

In denoising score matching we construct a collection of RVs {X(t)} T <sup>t</sup>=0 with X(0) = X and X(t)|X(0) ∼ N(m(t)X(0), σ(t)Ip). We assume that the noise σ(t) grows sufficiently so that X(T) is approximately an Isotropic Gaussian. The aim is now to estimate s(x, t) := ∇<sup>x</sup> log pt(x) where p<sup>t</sup> is the PDF of X(t). The denoising score matching objective is then.

$$\mathbb{E}[\nu(t)\|\mathbf{s}_\theta(X(t), t) - \nabla_{X(t)}\nabla_{X(t)}\log p(X(t)|X(0))\|_2^2] = \mathbb{E}[\nu(t)\|\mathbf{s}_\theta(X(t), t) + p(X(t)|X(0))\|_2^2]$$

where here t is treated as random and uniformly distributed on {1, . . . , T}, p(x(t)|x(0)) is the transition kernel from X(0) to X(t) and λ(t) is a self-specified weighting function over time. Due to our choice of noising process, ∇x(t) log p(x(t)|x(0)) = <sup>1</sup> σ(t) x(t) − m(t)x(0).

*Remark* D.2*.* Convention is to up-weight larger values of t as earlier parts of the reverse diffusion process (hence later parts of the original diffusion process) are seen as the most complex and where most of the data's structure is learned.

Our estimate is thus

$$\hat{\theta} = \operatorname{argmin}_{\theta \in \mathbb{R}^p} \frac{1}{n} \sum_{i \in n} \nu(t_i) \|s_{\theta}(X^{(i)}(t_i), t_i) + \nabla_{X^{(i)}} \log p(X^{(i)}(t) | X^{(i)}(0))\|^2$$

where t1, . . . , t<sup>n</sup> are sampled uniformly from [0, T].

*Remark* D.3*.* Originally denoising score matching was proposed for a single fixed t however for the purpose of generative modelling and annealed Langevin dynamics, multiple noise levels or even a continuous noising processes are used.

# E. Additional Details

#### E.1. Objectives

# E.1.1. MARGINAL IW SCORE MATCHING OBJECTIVE

Let {X ′(i,k) Λ<sup>i</sup> } r <sup>k</sup>=1 be IID copies of <sup>X</sup>′ Λ<sup>i</sup> with known PDF p ′ (xΛ<sup>i</sup> ). Our full sample objective is given by

$$\hat{L}_{\text{M};n,r}(\theta) := \frac{1}{n} \sum_{i=1}^n 2 \nabla_{X_{\Lambda_i}^{(i)}} \cdot \left( \frac{\frac{1}{r} \sum_{k=1}^r \frac{\nabla_{X_{\Lambda_i}^{(i)}} q_{\theta}(X_{\Lambda_i}^{(i)}, X'_{-\Lambda_i})}{p'(X_{\Lambda_i}^{(i,k)})}}{\frac{1}{r} \sum_{k=1}^r \frac{q_{\theta}(X_{\Lambda_i}^{(i)})}{p'(X_{-\Lambda_i}^{(i,k)})}} \right) + \left| \frac{\frac{1}{r} \sum_{k=1}^r \frac{\nabla_{X_{\Lambda_i}^{(i)}} q_{\theta}(X_{\Lambda_i}^{(i)}, x'_{-\Lambda_i})}{p'(X_{\Lambda_i}^{(i,k)})}}{\frac{1}{r} \sum_{k=1}^r \frac{q_{\theta}(X_{\Lambda_i}^{(i)})}{p'(X_{-\Lambda_i}^{(i,k)})}} \right|^2 \quad (24)$$

## E.1.2. MARGINAL TRUNCATED IW SCORE MATCHING

Our full sample objective for truncated IW missing score matching is given by

$$\hat{L}_{M;n,r}(\theta) := \frac{1}{n} \sum_{i=1}^n \sum_{j \in \Lambda_i} \mathbf{g}(X_{\Lambda_i^{(i)}})_j \left\{ 2\partial_j \left( \frac{\frac{1}{r} \sum_{k=1}^r \frac{\partial_j q_\theta(X_{\Lambda_i}^{(i)}, X'_{-\Lambda_i})}{p'(X_{\Lambda_i}^{'(i,k)})}}{\frac{1}{r} \sum_{k=1}^r \frac{q_\theta(X_{\Lambda_i}^{(i)})}{p'(X_{-\Lambda_i}^{'(i,k)})}} \right) + \left( \frac{\frac{1}{r} \sum_{k=1}^r \frac{\partial_j q_\theta(X_{\Lambda_i}^{(i)}, x'_{-\Lambda_i})}{p'(X_{\Lambda_i}^{'(i,k)})}}{\frac{1}{r} \sum_{k=1}^r \frac{q_\theta(X_{\Lambda_i}^{(i)})}{p'(X_{-\Lambda_i}^{'(i,k)})}} \right)^2 \right\} \quad (25)$$

$$+ 2\partial_j \mathbf{g}(X_{\Lambda_i})^{(i)} \left( \frac{\frac{1}{r} \sum_{k=1}^r \frac{\partial_j q_\theta(X_{\Lambda_i}^{(i)}, X'_{-\Lambda_i})}{p'(X_{\Lambda_i}^{'(i,k)})}}{\frac{1}{r} \sum_{k=1}^r \frac{q_\theta(X_{\Lambda_i}^{(i)})}{p'(X_{-\Lambda_i}^{'(i,k)})}} \right).$$

# E.2. Variational Modelling Details

For our purposes our variational model p ′ ϕ has to able able to model pθ(x−λ|xλ) for any value of λ ⊆ d. For our model we take X′ −λ |X<sup>λ</sup> = x<sup>λ</sup> ∼ N(µϕ(xλ), σ<sup>2</sup> ϕ I).

Hence we require µ<sup>λ</sup> to take in any subset of coordinates and output the complementing coordinates We achieve this by creating µ ′ ϕ , a d-dim in to d-dim out Neural Network with 2 hidden layers of 200 nodes as per [Burda et al.](#page-9-12) [\(2016\)](#page-9-12). For the input we replace x<sup>λ</sup> with x 0 λ the zero filled version inline with the approach taken in MissDiff [\(Ouyang et al., 2023\)](#page-10-10). That is (x 0 λ )<sup>j</sup> = 0 if j /∈ λ and is x<sup>j</sup> otherwise. As output we then simply take the appropriate coordinates. We can write this more succinctly as

$$\mu_\phi(\mathbf{x}_\lambda) = \mu'_\phi(\mathbf{x}_\lambda^0) - \lambda$$

We also experimented with making µ ′ ϕ a 2d dim input NN and taking

$$\mu_\phi(\mathbf{x}_\lambda) = \mu'_\phi(\mathbf{x}_\lambda^0, m)_{-\lambda}$$

where m is the d-dimensional binary mask for the corruption similarly to GAIN [\(Yoon et al., 2018\)](#page-10-17) however this did not seem to provide any advantage. We also experimented with making σ<sup>ϕ</sup> depend upon x<sup>λ</sup> however we found this made training much more unstable.

#### E.3. Experiment Implementation Details

#### E.3.1. NORMAL DISTRIBUTION ESTIMATION

The mean is taken fixed at µ = (0.5, . . . , 0.5)<sup>⊤</sup>. We randomly construct the covariance by first sampling eigenvalues uniformly on [0.5, 1.5] and then sampling choosing eigenvectors uniformly on the unit hypersphere for the first 9 dimensions. We then construct the the 10th dimension strong dependence on only the first dimension by taking X<sup>10</sup> = <sup>2</sup>X<sup>1</sup> + 1 2 Z with Var(Z) = Var(X1). The data is then truncated to be above the 10% quantile or each of the first three dimensions.

In each case batches of 100 samples were taken and a learning rate of 0.01 was used with Adam used as the optimisation algorithm. Our score model was parameterised in terms of the Cholesky decomposition of the precision matrix in order to ensure the Precision estimate stayed positive definite. For our Importance weighting and the EM approach of [Uehara et al.](#page-10-14) [\(2020\)](#page-10-14), an isotropic Gaussian with mean 0 and coordinatewise variance of 16 was used.

#### E.3.2. GAUSSIAN GRAPHICAL MODEL ESTIMATION

For Gaussian graphical model estimation we add L1 regularisation thereby modifying our objective to be

$$L_{\text{TM}}(\theta) + \gamma \sum_{1 \geq j < j' \leq d} |P_{j,j'}|$$

where θ = (µ, P) with P being our precision estimate. We minimise the objective by performing proximal gradient descent on LTM(θ). Specifically for a learning rate ν, a current estimate of θ given by θ<sup>t</sup> and an estimate of the gradient given by ηt. We take our estimate to be hγ,ν(θ<sup>t</sup> − νηt) where

$$h(\beta) := \begin{cases} \beta - \gamma\eta & \text{for } \beta > \gamma\eta \\ 0 & \text{for } -\gamma\eta \leq \beta \leq \gamma\eta \\ \beta + \gamma\eta & \text{for } \beta < -\gamma\eta \end{cases}$$

In our experiments we start with a precision estimate being P = I and with a large value of γ and, run 200 iterations, and then decrease the value of γ every 10 subsequent iterations for 100 sequentially smaller values of γ. At the end of each block of 10 iterations the precision matrix is taken and an adjacency matrix is produced by thresholding the entry's values at some small value. The TPR and FPR are then calculated for each of these increasingly dense matrices and then an ROC curves plotted using these values. The AUC of this ROC is then computed which is the statistic reported in the plots.

We took L1 regularisation to ensure that at the highest level the graph had no edges and at the lowest level the graph had all possible edges. For this experiment, this was achieved with γ ∈ (10<sup>−</sup>1.<sup>7</sup> , 10−<sup>4</sup> ). Throughout we took the threshold for edge presence to be 0.002.

#### E.3.3. REAL WORLD DATA

For these experiments we chose the range of L1 regularisation similarly to ensure a full range of edge densities (here this meant γ ∈ (10<sup>1</sup> , 10−<sup>4</sup> ) and then constructed a semi-automated procedure for choosing the threshold for edge presence. There is precedent for choosing the detection threshold after L1 regularisation as per [Fattahi & Sojoudi](#page-9-13) [\(2019\)](#page-9-13). We did this by choosing the non-zero threshold at the smallest value that gave a sufficiently smooth increase in edge density between the snapshots where we sample our estimated adjacency matrices.

Specifically the smoothness we were trying to achieve was avoid sudden decreases in the edge density as our regularisation level decreased. Our rough measure of this was to sum up all the negative jumps between sequential adjacency matrix estimates where the previous jump had been positive. This sum was then taken as our measure of "jumpiness" with larger values representing a larger level. Visually inspecting the change in positive level over time we find that a level of 0.01 for high-dimensional cases and a level 0.05 for low dimensional cases represented a relatively smooth change in edge density. The smallest non-zero threshold that satisfied this was then chosen by iterative shrinking grid search.

To test the performance of our adjacency matrix estimates, we estimated the adjacency matrix using standard score matching on the non-corrupted data. We estimated these adjacency matrices at 5 pre-determined values of edge densities given specifically, 0.05, 0.1, 0.15, 0.20.25. This lead to 5 different "ground truth" adjacency matrices. For each method, the AUC was calculated for each of these "ground truth" adjacency matrices and these 5 AUCs averaged. For each missing probability, 25 random samples of the corruption were produced and this AUC metric calculated. The average of these was then plotted alongside 95% confidence intervals.

The S&P 100 was taken from the S&P 500 data between 2013 and 2018 given in [https://www.kaggle.com/](https://www.kaggle.com/datasets/camnugent/sandp500) [datasets/camnugent/sandp500](https://www.kaggle.com/datasets/camnugent/sandp500) with the 100 stocks that made up the S&P 100 taken from roughly the mid-point of the time period which we obtained from [https://en.wikipedia.org/w/index.php?title=S%26P\\_100&](https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=666413597) [oldid=666413597](https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=666413597).

The yeast data was obtained from [https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1990/](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1990/matrix/) [matrix/](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1990/matrix/) which was accessed via <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1990> and other subsets have previously been studied in the context of GGM estimation in [Yang & Lozano](#page-10-18) [\(2015\)](#page-10-18).

All data can also be found in the GitHub repository at [https://github.com/joshgivens/](https://github.com/joshgivens/ScoreMatchingwithMissingData) [ScoreMatchingwithMissingData](https://github.com/joshgivens/ScoreMatchingwithMissingData).