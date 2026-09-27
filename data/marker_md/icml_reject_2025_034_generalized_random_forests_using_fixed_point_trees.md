## Generalized Random Forests using Fixed-Point Trees

David Fleischer \* 1 David A. Stephens \* 1 Archer Y. Yang \* 1 2

#### Abstract

We propose a computationally efficient alternative to generalized random forests (GRFs) for estimating heterogeneous effects in large dimensions. While GRFs rely on a gradient-based splitting criterion, which in large dimensions is computationally expensive and unstable, our method introduces a fixed-point approximation that eliminates the need for Jacobian estimation. This gradientfree approach preserves GRF's theoretical guarantees of consistency and asymptotic normality while significantly improving computational efficiency. We demonstrate that our method achieves a speedup of multiple times over standard GRFs without compromising statistical accuracy. Experiments on both simulated and real-world data validate our approach. Our findings suggest that the proposed method is a scalable alternative for localized effect estimation in machine learning and causal inference applications.

### 1. Introduction

In many real-world machine learning (ML) applications, practitioners seek to estimate how quantities of interest vary across different feature subgroups rather than assuming uniform effects. For example, medical interventions and policy treatments often have heterogeneous impacts across subpopulations, making localized estimation crucial for improving outcomes [\(Imai & Ratkovic,](#page-10-0) [2013;](#page-10-0) [Knaus et al.,](#page-10-1) [2021;](#page-10-1) [Mur](#page-10-2)[doch et al.,](#page-10-2) [2019;](#page-10-2) [Lee et al.,](#page-10-3) [2020\)](#page-10-3). Similarly, individualized recommendation systems adapt to user-specific features to enhance performance [\(Kohavi et al.,](#page-10-4) [2013\)](#page-10-4).

A key example of localized estimation arises in causal inference, where modern applications prioritize individualized treatment effects over average treatment effects [\(Neyman,](#page-10-5)

[1923;](#page-10-5) [Rubin,](#page-10-6) [1974\)](#page-10-6). The double machine learning framework [\(Chernozhukov et al.,](#page-9-0) [2018\)](#page-9-0) unifies various ML-based causal estimation methods, including lasso [\(Belloni et al.,](#page-9-1) [2017\)](#page-9-1), random forests [\(Athey et al.,](#page-9-2) [2019;](#page-9-2) [Cevid et al.,](#page-9-3) [2022\)](#page-9-3), boosting [\(Powers et al.,](#page-10-7) [2018\)](#page-10-7), deep learning [\(Johans](#page-10-8)[son et al.,](#page-10-8) [2016;](#page-10-8) [Shalit et al.,](#page-11-0) [2017\)](#page-11-0), and general-purpose meta-algorithms [\(Nie & Wager,](#page-10-9) [2021;](#page-10-9) [Kunzel et al.](#page-10-10) ¨ , [2019\)](#page-10-10), all of which focus on capturing variation over feature space.

Generalized random forests (GRFs) [\(Athey et al.,](#page-9-2) [2019;](#page-9-2) [Wa](#page-11-1)[ger & Athey,](#page-11-1) [2018\)](#page-11-1) have emerged as a powerful tool for such tasks, leveraging adaptive partitioning with problemspecific moment conditions instead of standard loss-based splits. GRFs apply broadly to a wide range of important statistical models – local linear regression [\(Friedberg et al.,](#page-10-11) [2020\)](#page-10-11), survival analysis and missing data problems [\(Cui](#page-9-4) [et al.,](#page-9-4) [2023\)](#page-9-4), nonparametric quantile regression, heterogeneous treatment effect estimation, and nonlinear instrumental variables regression [\(Athey & Imbens,](#page-9-5) [2016;](#page-9-5) [Athey et al.,](#page-9-2) [2019\)](#page-9-2). Unlike local linear models [\(Fan et al.,](#page-10-12) [1995;](#page-10-12) [Fan &](#page-9-6) [Gijbels,](#page-9-6) [1996;](#page-9-6) [Friedberg et al.,](#page-10-11) [2020\)](#page-10-11) or kernel-based models [\(Staniswalis,](#page-11-2) [1989;](#page-11-2) [Severini & Staniswalis,](#page-11-3) [1994;](#page-11-3) [Lewbel,](#page-10-13) [2007;](#page-10-13) [Speckman,](#page-11-4) [1988;](#page-11-4) [Robinson,](#page-10-14) [1988\)](#page-10-14) which suffer from the curse of dimensionality [\(Robins & Ritov,](#page-10-15) [1997\)](#page-10-15), the tree-based approach of GRF offers a more scalable solution.

However, GRFs' gradient-based approach [\(Athey et al.,](#page-9-2) [2019\)](#page-9-2) becomes computationally expensive and unstable in large dimensions due to the reliance on Jacobian estimators for tree splitting. To address this, we propose a gradientfree approach based on fixed-point iteration, eliminating the need for Jacobian estimation while retaining GRF's theoretical guarantees of consistency and asymptotic normality. Our method significantly improves computational efficiency while maintaining statistical accuracy, achieving significant speedups in experiments on simulated and real-world datasets.

## 2. Background and Related Work

Given data (X<sup>i</sup> , Oi) ∈ X × O, GRF estimates a target function θ ∗ (x), defined as the solution to an estimating equation of the form

$$0 = \mathbb{E}_{O|X} [\psi_{\theta^*(x),\nu^*(x)}(O) \mid X = x], \quad (1)$$

<sup>1</sup>Department of Mathematics and Statistics, McGill University, Montreal, Canada <sup>2</sup>Mila - Quebec AI Institute, Montreal, Quebec, Canada. Correspondence to: Archer Y. Yang <archer.yang@mcgill.ca>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

for all x ∈ X , where ψ is a score function that identifies the true (θ ∗ (x), ν<sup>∗</sup> (x)) as the root of [\(1\)](#page-0-0), and ν ∗ (x) is an optional nuisance function. GRF can be understood from a nearest-neighbor perspective as approximating θ ∗ (x) through a locally parametric θ <sup>∗</sup> within small neighborhoods of test point x. Suppose L(x) ⊂ {Xi} n <sup>i</sup>=1 is a subset of training observations of the covariates found in a region around x ∈ X over which θ ∗ (x) can be well-approximated by a local parameter. Observations X<sup>i</sup> ∈ L(x) serve as local representatives for x in estimating θ ∗ (x) such that, given sufficiently many training samples in a small enough neighborhood of x, an empirical version of [\(1\)](#page-0-0) over X<sup>i</sup> ∈ L(x) defines an estimator <sup>ˆ</sup>θL(x) that approaches θ ∗ (x),

$$(\hat{\theta}_{L(x)}, \hat{\nu}_{L(x)}) \in \arg \min_{\theta, \nu} \left\| \sum_{i=1}^n \frac{1(X_i \in L(x))}{|L(x)|} \cdot \psi_{\theta, \nu}(O_i) \right\|. \quad (2)$$

In GRF, the set of local representatives L(x) is determined by tree-based partitions which divide the input space into disjoint regions, or leaves. The training samples X<sup>i</sup> that fall in the same leaf as x form the subset L(x). However, single trees are known to have high variance with respect to small changes in the training data [\(Amit & Geman,](#page-9-7) [1997;](#page-9-7) [Breiman,](#page-9-8) [1996;](#page-9-8) [2001;](#page-9-9) [Dietterich,](#page-9-10) [2000\)](#page-9-10), leading to estimates [\(2\)](#page-1-0) that do not generalize well to values of x that are not part of the training set. GRF improves its estimates by leveraging an estimating function that averages many estimating functions of the form [\(2\)](#page-1-0). Specifically, let Lb(x) denote the set of training covariates that fall in the same leaf as x, identified by a tree trained on an independent subsample of the data, indexed by b = 1, . . . , B. The GRF estimator is obtained by aggregating the individual estimating functions [\(2\)](#page-1-0) across a forest of B independently trained trees, i.e. the solution to the following forest-averaged estimating equation:

$$(\hat{\theta}(x), \hat{\nu}(x)) \in \arg\min_{\theta, \nu} \left\| \frac{1}{B} \sum_{b=1}^B \left( \sum_{i=1}^n \alpha_{bi}(x) \psi_{\theta, \nu}(O_i) \right) \right\|_{\mathbb{R}}. \quad (3)$$

where αbi(x) := <sup>1</sup>(Xi∈Lb(x)) |Lb(x)| . Define observational weights αi(x) that measure the relative frequency with which training sample X<sup>i</sup> falls in the same leaf as x, averaged over B trees:

$$\alpha_i(x) := \frac{1}{B} \sum_{b=1}^B \alpha_{bi}(x), \quad (4)$$

for i = 1, . . . , n. Then, the solution ( ˆθ(x), νˆ(x)) to the forest-averaged model [\(3\)](#page-1-1) is equivalent to solving the following locally weighted estimating equation

$$(\hat{\theta}(x), \hat{\nu}(x)) \in \arg \min_{\theta, \nu} \left\| \sum_{i=1}^n \alpha_i(x) \psi_{\theta, \nu}(O_i) \right\|. \quad (5)$$

[Athey et al.](#page-9-2) [\(2019\)](#page-9-2) present [\(5\)](#page-1-2) as the definition of the GRF estimator, motivated in part by the mature analyses of local kernel methods [\(Newey,](#page-10-16) [1994\)](#page-10-16) alongside more recent work on tree-based partitioning and estimating equations [\(Athey & Imbens,](#page-9-5) [2016;](#page-9-5) [Zeileis & Hornik,](#page-11-5) [2007;](#page-11-5) [Zeileis](#page-11-6) [et al.,](#page-11-6) [2008\)](#page-11-6). The GRF algorithm for estimating θ ∗ (x) can be summarized as a two-stage procedure. Stage I: Use trees to calculate weight functions αi(x) for any test observation x ∈ X , measuring the relative importance of the i-th training sample to estimating θ ∗ (·) near x. Stage II: Given a test observation x ∈ X , compute estimate ˆθ(x) of θ ∗ (x) by solving the locally weighted empirical estimating equation [\(5\)](#page-1-2).

Our contribution improves the computational cost of Stage I by introducing a more efficient procedure to train the trees. Training the forest is the most resource-intensive step of GRF, and the cost of each split in the existing approach scales quadratically with the dimension of θ ∗ (x). We adopt a gradient-free splitting mechanism and significantly reduce both the time and memory demands of Stage I. Crucially, solving Stage II with weights αi(x) following our streamlined Stage I produces an estimator ˆθ(x) that preserves the finite-sample performance and asymptotic guarantees of GRF.

## 3. Our Method

In this section we describe the details of our accelerated algorithm for GRF. We closely follow the approach of [Athey](#page-9-2) [et al.](#page-9-2) [\(2019\)](#page-9-2), and define ˆθ(x) as the solution to a locally weighted problem [\(5\)](#page-1-2) with weighting functions αi(x) of the form [\(4\)](#page-1-3). The weight functions are induced by a collection of local subsets {Lb(x)} B <sup>b</sup>=1, such that each subset Lb(x) is determined by the partition rules of a tree trained on a subsample. The construction of each tree, in turn, is determined by recursive splits of the subsample based on a splitting criterion designed to identify regions of X that are homogeneous with respect to θ ∗ (x). Therefore, to fully specify the weight functions αi(x), we must describe a feasible criterion for producing a split of X .

#### 3.1. The target tree-splitting criterion for Stage I

In GRF, the goal of Stage I is to use recursive tree-based splits of the training data to induce a partition over the input space. Each split starts with a parent node P ⊂ X and results in child nodes C1, C<sup>2</sup> ⊂ X , defined by a binary, axisaligned splitting rule of the form C<sup>1</sup> = {X<sup>i</sup> : Xi,ℓ ≤ t} and C<sup>2</sup> = {X<sup>i</sup> : Xi,ℓ > t}, where ℓ denotes a candidate splitting feature/axis and t ∈ R the splitting threshold. For a parent P and any child nodes C1, C<sup>2</sup> of P, let ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) and ( <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> , νˆC<sup>j</sup> ) denote local solutions analogous to [\(2\)](#page-1-0) defined

over the samples in P and C<sup>j</sup> , respectively:

$$(\hat{\theta}_P, \hat{\nu}_P) \in \arg \min_{\theta, \nu} \left\| \sum_{\{i: X_i \in P\}} \psi_{\theta, \nu}(O_i) \right\|, \quad (6)$$

$$(\hat{\theta}_{C_j}, \hat{\nu}_{C_j}) \in \arg \min_{\theta, \nu} \left\| \sum_{\{i: X_i \in C_j\}} \psi_{\theta, \nu}(O_i) \right\|, \quad (7)$$

for j = 1, 2. A strategy to split P into two subsets of greater homogeneity with respect to θ ∗ (·) is as follows: Find child nodes C<sup>1</sup> and C<sup>2</sup> such that the total deviation between the local solutions <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> and the target θ ∗ (X) is minimized, conditional on X ∈ C<sup>j</sup> , j = 1, 2. A natural measure of deviation is the squared-error loss,

$$\begin{aligned} \text{err}(C_1, C_2) &:= \sum_{j=1,2} \mathbb{P}(X \in C_j \mid X \in P) \\ &\times \mathbb{E} \left[ \left\| \theta^*(X) - \hat{\theta}_{C_j} \right\|^2 \mid X \in C_j \right], \end{aligned}$$

such that the resulting split (C1, C2) corresponds to least-squares optimal solutions <sup>ˆ</sup>θ<sup>C</sup><sup>1</sup> and <sup>ˆ</sup>θ<sup>C</sup><sup>2</sup> . However, err(C1, C2) is intractable since θ ∗ (·) is unknown. GRF considers a criterion that measures heterogeneity across a pair of local solutions over a candidate split

$$\Delta(C_1, C_2) := \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \hat{\theta}_{C_1} - \hat{\theta}_{C_2} \right\|^2, \quad (8)$$

where nC<sup>1</sup> , nC<sup>2</sup> , and n<sup>P</sup> denote the number of observations in C1, C2, and P, respectively. In particular, rather than minimizing err(C1, C2), one can seek a split of P such that the cross-split heterogeneity between <sup>ˆ</sup>θ<sup>C</sup><sup>1</sup> and <sup>ˆ</sup>θ<sup>C</sup><sup>2</sup> is maximized. [Athey et al.](#page-9-2) [\(2019\)](#page-9-2) observe that err(C1, C2) and ∆(C1, C2) are coupled according to err(C1, C2) = K(P) − <sup>E</sup> [∆(C1, C2)] + o(r 2 ), where r > 0 is a small radius term tied to the sampling variance, and K(P) does not depend on the split of P. That is, splits that maximize ∆(C1, C2) – which emphasize the heterogeneity of <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> across a split – will asymptotically minimize err(C1, C2), which aims to improve the homogeneity of <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> within a split.

Although the criterion ∆(C1, C2) is computable, evaluating it is very computationally expensive since it requires solving [\(7\)](#page-2-0) to obtain <sup>ˆ</sup>θ<sup>C</sup><sup>1</sup> , <sup>ˆ</sup>θ<sup>C</sup><sup>2</sup> for all possible splits of P, and closedform solutions for <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> are generally not available except in special cases of ψ. Instead, GRF approximates the target ∆-criterion based on a criterion of the form

$$\tilde{\Delta}^{\text{grad}}(C_1, C_2) := \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \tilde{\theta}_{C_1}^{\text{grad}} - \tilde{\theta}_{C_2}^{\text{grad}} \right\|^2, \quad (9)$$

where ˜θ grad C<sup>j</sup> denotes a *gradient-based* approximation of <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> . Specifically, ˜θ grad C<sup>j</sup> is a first-order approximation interpreted

as the result of taking a gradient step away from the parent estimate in the direction towards the true child solution <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> :

$$\tilde{\theta}_{C_j}^{\text{grad}} := \hat{\theta}_P - \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \xi^\top A_P^{-1} \psi_{\hat{\theta}_P, \nu_P}(O_i), \quad (10)$$

where ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) is the local solution over the parent, A<sup>P</sup> is any consistent estimator of the local Jacobian matrix ∇(θ,ν)<sup>E</sup>[ψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) | X<sup>i</sup> ∈ P], and ξ <sup>⊤</sup> can be thought of as a term that selects a θ-subvector from a (θ, ν)-vector, e.g. if θ ∈ R <sup>K</sup> and ν ∈ <sup>R</sup>, then ξ <sup>⊤</sup> such that θ = ξ <sup>⊤</sup>(θ, ν) <sup>⊤</sup> is the rectangular diagonal matrix ξ <sup>⊤</sup> = [<sup>I</sup><sup>K</sup> 0]. When the scoring function ψ is continuously differentiable in (θ, ν), the Jacobian estimator A<sup>P</sup> can be computed as

$$\begin{aligned} A_P &= \nabla_{(\theta, \nu)} \frac{1}{n_P} \sum_{\{i: X_i \in P\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) \\ &= \frac{1}{n_P} \sum_{\{i: X_i \in P\}} \nabla_{(\theta, \nu)} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i). \end{aligned} \quad (11)$$

#### 3.2. Limitations of gradient-based approximation

The use of the Jacobian estimator A<sup>P</sup> in [\(10\)](#page-2-1) introduces considerable computational challenges. First, each parent node P in every tree of the forest requires a distinct A<sup>P</sup> matrix, which imposes a significant computational burden when explicitly calculating A −1 <sup>P</sup> ψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) to determine ˜θ grad C<sup>j</sup> . Second, if the local Jacobian ∇(θ,ν)<sup>E</sup>[ψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) | X<sup>i</sup> ∈ P] is ill-conditioned, then the resulting A<sup>P</sup> estimator may be nearly singular. This instability can lead to highly variable gradient-based approximations ˜θ grad C<sup>j</sup> and highly variable splits of P. For example, consider the following varying-coefficient model for an outcome Y<sup>i</sup> given regressors W<sup>i</sup> = (Wi,1, . . . , Wi,K) <sup>⊤</sup> in the presence of mediating auxiliary covariates X<sup>i</sup> :

$$\mathbb{E}[Y_i \mid X_i = x] = \nu^*(x) + W_i^\top \theta^*(x), \quad (12)$$

where ν ∗ (·) is a nuisance intercept function and θ ∗ (x) = (θ ∗ 1 (x), . . . , θ<sup>∗</sup> <sup>K</sup>(x))<sup>⊤</sup> are the target coefficients. Models of the form [\(12\)](#page-2-2) encompass time- or spatially-varying coefficient frameworks, where (X<sup>i</sup> , Y<sup>i</sup> , Wi) represent the i-th sample associated with spatiotemporal values X<sup>i</sup> . Such models are particularly relevant in applications like heterogeneous treatment effects; see Section [5](#page-6-0) for a more in-depth discussion. The local estimating function ψθ,ν(Y<sup>i</sup> , Wi), identifying (θ ∗ (x), ν<sup>∗</sup> (x)) through moment conditions as in [\(1\)](#page-0-0), is given by:

$$\psi_{\theta, \nu}(Y_i, W_i) := \begin{bmatrix} (Y_i - W_i^\top \theta - \nu) \cdot W_i \\ Y_i - W_i^\top \theta - \nu \end{bmatrix}.$$

![](_page_3_Figure_1.jpeg)

Figure 1: Splits values (top) and split variance (bottom), with 10th and 90th percentile bands, across correlations of Wi,<sup>1</sup> and Wi,2.

Consequently, the corresponding local Jacobian estimator is

$$\begin{aligned} A_P &= \frac{1}{n_P} \sum_{\{i: X_i \in P\}} \nabla_{(\theta, \nu)} \psi_{\theta, \nu}(Y_i, W_i) \\ &= -\frac{1}{n_P} \sum_{\{i: X_i \in P\}} \begin{bmatrix} W_i W_i^\top & W_i^\top \\ W_i & 1 \end{bmatrix}. \end{aligned} \quad (19)$$

When the regressors are highly correlated, the summation over the WiW<sup>⊤</sup> i block of the A<sup>P</sup> matrix leads to nearly singular values of A<sup>P</sup> , resulting in an unstable matrix inverse A −1 P , and therefore unstable values of ˜θ grad C<sup>j</sup> and unstable splits. This issue becomes more pronounced as the number of parent samples n<sup>P</sup> decreases, as is the case at deeper levels of the tree. These challenges highlight the limitations of relying on A<sup>P</sup> as part of an approximation for the child solutions <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> .

As an illustration, consider a simple varying coefficient model with primary regressors Wi,1, Wi,<sup>2</sup> ∼ N (0, 1), auxiliary covariates X<sup>i</sup> ∼ Unif(0, 1), and outcomes Y<sup>i</sup> generated as

$$Y_i = \mathbb{1}(X_i > 0.5)W_{i,1} + W_{i,2} + \epsilon_i, \quad (14)$$

where ϵ<sup>i</sup> ∼ N (0, 1). Figure [1](#page-3-0) illustrates the distribution of 2000 <sup>∆</sup>e grad-optimal binary splits (gradient-based tree stumps) fit over 1000 samples of the varying coefficient model [\(14\)](#page-3-1), repeated over different regressor correlation levels Corr(Wi,1, Wi,2) ∈ {0.80, 0.81, . . . , 0.98, 0.99}. It is clear that splits based on the <sup>∆</sup>e grad-criterion exhibit high variability when the correlation between the regressors is large. In contrast, our proposed method, discussed in the next section, does not suffer from the same problem.

#### 3.3. Fixed-point approximation

To address the limitations of gradient-based approximations, we propose a gradient-free approach based on the form of a single fixed-point iteration. Let ΨC<sup>j</sup> (θ, ν) := 1 nCj P {i:Xi∈C<sup>j</sup> } ψθ,ν(Oi) denote the empirical estimating function for the child solution ( <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> , νˆC<sup>j</sup> ) such that [\(7\)](#page-2-0) is equivalently written as:

$$(\hat{\theta}_{C_j}, \hat{\nu}_{C_j}) \in \arg\min_{\theta, \nu} \|\Psi_{C_j}(\theta, \nu)\|, \quad j = 1, 2. \quad (15)$$

Under mild regularity conditions, ( <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> , νˆC<sup>j</sup> ) is a Zestimator that solves the estimating equation ΨC<sup>j</sup> (θ, ν) = 0. Reformulating this equation as a fixed-point problem, we write:

$$(\theta, \nu) = \underbrace{(\theta, \nu) - \eta \Psi_{C_j}(\theta, \nu)}_{=: f(\theta, \nu)}, \quad \eta > 0. \quad (16)$$

A necessary and sufficient condition for ( <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> , νˆC<sup>j</sup> ) to be a solution of [\(15\)](#page-3-2) is characterized by the fixed-point problem ( <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> , νˆC<sup>j</sup> ) = f( <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> , νˆC<sup>j</sup> ), where f is as defined in [\(16\)](#page-3-3). Iterative fixed-point methods [\(Picard,](#page-10-17) [1890;](#page-10-17) [Lindelof¨](#page-10-18) , [1894;](#page-10-18) [Banach,](#page-9-11) [1922;](#page-9-11) [Ryu & Boyd,](#page-11-7) [2016;](#page-11-7) [Yang et al.,](#page-11-8) [2021\)](#page-11-8) solve such problems by considering an update rule of the form

$$(\theta^+, \nu^+) \leftarrow f(\theta, \nu). \quad (17)$$

The form of [\(17\)](#page-3-4) inspires us to approximate the true child solution <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> using a single fixed-point update taken from the parent solution ˆθ<sup>P</sup> :

$$\begin{aligned} \tilde{\theta}_{C_j}^{\text{FT}} &:= \hat{\theta}_P - \eta \xi^\top \Psi_{C_j}(\hat{\theta}_P, \hat{\nu}_P) \\ &= \hat{\theta}_P - \frac{\eta}{n_{C_j}} \xi^\top \sum_{\{i: X_i \in C_j\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i), \end{aligned} \quad (18)$$

where the product with ξ <sup>⊤</sup> is interpreted similarly to its role in the gradient-based approximation [\(10\)](#page-2-1) and to express the update [\(17\)](#page-3-4) solely in terms of the target θ-quantity. We interpret ˜θ FPT C<sup>j</sup> as an approximation of <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> obtained by taking a step from ˆθ<sup>P</sup> in a direction that reduces the magnitude of the local estimating function ΨC<sup>j</sup> . Notably, the approximation ˜θ C<sup>j</sup> does not involve the A<sup>P</sup> matrix, relying only on the scores ψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) evaluated at the parent solutions. In general, removing the inverse A −1 P provides computational cost savings of O(K<sup>3</sup> ). The corresponding splitting criterion, which uses the fixed-point approximations ˜θ C<sup>j</sup> as substitutes for <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> is given by

$$\tilde{\Delta}^{\text{FPT}}(C_1, C_2) := \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \tilde{\theta}_{C_1}^{\text{FPT}} - \tilde{\theta}_{C_2}^{\text{FPT}} \right\|^2. \quad (19)$$

Revisiting the varying coefficient example from Section [3.2,](#page-2-3) we see that splits based on fixed-point approximations ˜θ C<sup>j</sup>

are significantly more stable than those based on ˜θ grad C<sup>j</sup> . Specifically, Figure [1](#page-3-0) illustrates that splits that maximize <sup>∆</sup>e FPT(C1, C2) are more robust to ill-conditioning in the underlying local Jacobian ∇(θ,ν)<sup>E</sup>[ψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) | X<sup>i</sup> ∈ P], as is the case for highly correlated regressors in the varying coefficient model [\(14\)](#page-3-1), and leading to highly stable splits.

#### 3.4. Pseudo-outcomes

Approximations ˜θ<sup>C</sup><sup>j</sup> of the form [\(10\)](#page-2-1) and [\(18\)](#page-3-5) offer an additional benefit: they enable the <sup>∆</sup>e -criteria of the form [\(9\)](#page-2-4) and [\(19\)](#page-3-6) to be efficiently optimized through a single multivariate CART split. A CART split performed with respect to vectorvalued responses ρ<sup>i</sup> ∈ <sup>R</sup> <sup>K</sup> over a parent node P produces a split (C1, C2) that minimizes the following least-squares criterion:

$$\sum_{\{i: X_i \in C_1\}} \|\rho_i - \bar{\rho}_{C_1}\|^2 + \sum_{\{i: X_i \in C_2\}} \|\rho_i - \bar{\rho}_{C_2}\|^2, \quad (20)$$

where ρ¯C<sup>j</sup> := nCj P {i:Xi∈C<sup>j</sup> } ρi . [<sup>1</sup>](#page-0-1) Equivalently, a CART split that minimizes [\(20\)](#page-4-0) will maximize:

$$n_{C_1} \|\bar{\rho}_{C_1}\|^2 + n_{C_2} \|\bar{\rho}_{C_2}\|^2. \quad (21)$$

The equivalence between the split that minimizes the leastsquares CART criterion [\(20\)](#page-4-0) and the split that maximizes [\(21\)](#page-4-1) is shown in Appendix [B.1.1.](#page-14-0) GRF performs its splits by adopting gradient-based *pseudo-outcomes*, defined as

$$\rho_i^{\text{grad}} := -\xi^\top A_P^{-1} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) \quad (22)$$

such that the gradient-based approximation ˜θ grad C<sup>j</sup> in [\(10\)](#page-2-1) is equivalently written:

$$\tilde{\theta}_{C_j}^{\text{grad}} = \hat{\theta}_P + \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \rho_i^{\text{grad}} = \hat{\theta}_P + \bar{\rho}_{C_j}^{\text{grad}}.$$

In the case of fixed-point approximation, we define fixedpoint pseudo-outcomes:

$$\rho_i^{\text{FPT}} := -\eta \xi^\top \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i), \quad \eta \neq 0, \quad (23)$$

such that the fixed-point approximation ˜θ C<sup>j</sup> in [\(18\)](#page-3-5) is equivalently written as

$$\tilde{\theta}_{C_j}^{\text{FPT}} = \hat{\theta}_P + \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \rho_i^{\text{FPT}} = \hat{\theta}_P + \bar{\rho}_{C_j}^{\text{FPT}}. \quad (24)$$

Substitute the above form of ˜θ FPT C<sup>j</sup> into the <sup>∆</sup>e FPT-criterion [\(19\)](#page-3-6) to equivalently express the criterion in terms of the FPT pseudo-outcomes:

$$\tilde{\Delta}^{\text{FPT}}(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \|\bar{\rho}_{C_1}^{\text{FPT}} - \bar{\rho}_{C_2}^{\text{FPT}}\|^2, \quad (25)$$

where an analogous equivalence holds for <sup>∆</sup>e grad in terms of the gradient-based pseudo-outcomes. We demonstrate in Lemma [B.1](#page-15-0) (in Appendix [B.1.2\)](#page-15-1) that maximizing the fixedpoint criterion <sup>∆</sup>e FPT(C1, C2) is equivalent to maximizing the CART criterion [\(21\)](#page-4-1), and extend this property to any <sup>∆</sup>e -style criterion induced by pseudo-outcomes that can be expressed as a split-independent linear transformation of the parent scores ψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi).

Note that our method does not rely on iterative fixed-point procedures at all. Instead, it uses only a single step of fixed-point approximation to simplify the pseudo-outcomes. These simplified pseudo-outcomes are then passed directly to a standard CART algorithm for splitting. The numerical convergence of our method therefore relies solely on CART's established and well-known stability, not on fixedpoint iteration. CART splits on pseudo-outcomes are computationally efficient. Given a parent node P, the value ρ<sup>i</sup> <sup>=</sup> −Bψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) does not depend on a candidate split (C1, C2) for any matrix B that is fixed with respect to the parent. This allows much of the computation required to maximize <sup>∆</sup>e FPT(C1, C2) to be done at the parent level, and in particular avoids re-calculating the approximations ˜θ C<sup>1</sup> and ˜θ C<sup>2</sup> across the sequence of candidate splits. Once P is fixed and ρ i are computed, the value of <sup>∆</sup>e FPT(C1, C2) for the first candidate split requires O(n<sup>P</sup> ) time, and the value for all other candidate splits of P are queried in O(1) time. While gradient-based pseudo-outcomes share this property, the use of fixed-point pseudo-outcomes eliminates the computational overhead and instability associated with estimating A<sup>P</sup> , as discussed in Section [3.2.](#page-2-3)

We show in Lemma [B.2](#page-16-0) (Appendix [B.1.3\)](#page-16-1) that choosing different values of η does not change the outcome of the fixed-point splitting mechanism. Specifically, the optimal split identified by CART on pseudo-outcomes ρ FPT i of the form [\(23\)](#page-4-2) does not depend on η. This can be heuristically understood by studying how the criterion changes as a function of the candidate splits. To illustrate, we consider a VCM model of the form [\(12\)](#page-2-2) for bivariate regressors W<sup>i</sup> , univariate X<sup>i</sup> ∈ [0, 1], and scalar outcomes Yi . A detailed summary of the settings is found in Appendix [D.1.](#page-29-0) The sequence of valid candidate child nodes obtained by a split over univariate X<sup>i</sup> can be parameterized through scalar t as C1(t) := {X<sup>i</sup> : X<sup>i</sup> ≤ t} and C2(t) := {X<sup>i</sup> : X<sup>i</sup> > t}. Let ∆(t) := ∆(C1(t), C2(t)) denote the parameterized target criterion [\(8\)](#page-2-5), and consider the behavior of ∆(t), <sup>∆</sup>e grad(t), and two fixed-point criteria <sup>∆</sup>e FPT 1 (t) and <sup>∆</sup>e FPT 2 (t) of the form [\(25\)](#page-4-3) based on pseudooutcomes with scale factors η = 1 and η = 1/ √ 2, respectively. Figure [2](#page-5-0) illustrates the different splitting criteria values plotted against the sequence of candidate splits. The visualization clearly shows that the criteria curves for ∆(t), <sup>∆</sup>e grad(t), and <sup>∆</sup>e FPT 1 (t) with η = 1 are all very close to one

<sup>1</sup>The multivariate CART criterion uses a sum of squares impurity measure, as in [De'ath](#page-9-12) [\(2002\)](#page-9-12); [Segal](#page-11-9) [\(1992\)](#page-11-9).

![](_page_5_Figure_1.jpeg)

Figure 2: Criterion values across candidate splits (C1(t), C2(t)) over threshold t ∈ [0, 1]. The location of the optimal split under each criterion is given by the corresponding vertical line.

another. Critically, the fixed-point criterion with η = 1/ √ 2, i.e. <sup>∆</sup>e FPT 2 (t), although scaled differently, still identifies the same maximizing split as <sup>∆</sup>e FPT 1 (t). This is because CART chooses a split based on a rank ordering of the criterion over all candidate splits. The absolute scale of the CART criterion does not matter, and it is only criterion rankings over the candidates that determines the optimal split. Therefore, choosing a different scalar η does not change the outcome of the splitting process.

Based on the scale-invariance of our splitting criterion, we now detail the recursive procedure for growing our fixedpoint trees pseudo-outcomes with η = 1.

The fixed-point tree algorithm. The entire fixed-point tree-growing procedure recursively applies the following two steps on a given parent node P:

- (i) Labeling: Solve [\(6\)](#page-2-6) over P to obtain the parent estimate ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ). Compute the pseudo-outcomes:

$$\rho_i^{\text{FPT}} := -\xi^\top \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i), \quad (26)$$

for all i such that X<sup>i</sup> ∈ P.

- (ii) Regression: Maximize <sup>∆</sup>e FPT(C1, C2) by performing a CART split on the pseudo-outcomes ρ i over P.

### 3.5. Estimates of ˆθ(x) for Stage II

The fixed-point tree algorithm generates a single tree-based partition of X . Repeating this process over subsamples of the training data yields a forest of trees, each specifying local leaf functions Lb(x). These leaf functions define the local weight functions αi(x) via [\(4\)](#page-1-3), completing Stage I of GRF. The full fixed-point tree training algorithm is described in Algorithm [1,](#page-27-0) while Algorithm [2](#page-28-0) provides the pseudocode for the forest-wide Stage I procedure.

To compute the final GRF estimates ˆθ(x) for the target θ ∗ (x), we follow the standard GRF mechanism for Stage II. After the fixed-point trees are trained in Stage I, a test observation x<sup>0</sup> ∈ X is assigned to local leaves Lb(x0), indexed by trees b ∈ {1, . . . , B}. Each leaf Lb(x0) contains the training observations that fall into the same leaf as x<sup>0</sup> in tree b. Using these local leaves, the forest computes training weights αi(x0) as in [\(4\)](#page-1-3). The final estimate ˆθ(x0) is obtained by solving the locally weighted estimating equation [\(5\)](#page-1-2).

Importantly, as discussed in Section [2,](#page-0-2) solving for ˆθ(x0) in Stage II is independent of the specific mechanism used in Stage I. The only requirement is that Stage I produces valid weights. This ensures that Stage II remains a standard weighted estimating equation, enabling the fixed-point tree algorithm to integrate seamlessly into GRF's two-stage framework. We refer to the complete algorithm for estimating θ ∗ (x) using fixed-point trees as GRF-FPT. By preserving Stage II of GRF, the GRF-FPT estimator ˆθ(x) retains GRF's theoretical guarantees of consistency and asymptotic normality while offering a computationally efficient tree-building method. Pseudocode for Stage II of the GRF-FPT algorithm is provided in Algorithm [3,](#page-28-1) located in Appendix [C.3.](#page-27-1)

#### 4. Theoretical Analysis

In this section, we provide a theoretical foundation for the GRF-FPT estimator ˆθ(x). For Stage I, Proposition [4.1](#page-5-1) establishes an asymptotic equivalence between the FPT criterion and a weighted oracle criterion ∆<sup>V</sup> (C1, C2) in [\(27\)](#page-5-2), while Lemma [4.2](#page-6-1) demonstrates that the Specifications [A.2](#page-13-0) are met by a forest based on the ∆<sup>V</sup> -criterion whenever they are met by a forest based on the ∆-criterion. Assumptions [A.1](#page-12-0) and Specifications [A.2](#page-13-0) are the sufficient conditions for the consistency and asymptotic normality of ˆθ(x) in [\(5\)](#page-1-2), and thus are used to formally justify the FPT algorithm as a mechanism for specifying an estimator of θ ∗ (x).

Proposition 4.1. *Suppose Assumptions [A.1](#page-12-0) hold, and assume moreover Neyman orthogonal moment conditions (defined in Appendix [A.4\)](#page-13-1). Denote by* r := sup{i:Xi∈<sup>P</sup> } ∥X<sup>i</sup> − x<sup>P</sup> ∥ *the radius of the parent* P*, where* x<sup>P</sup> *denotes the center of mass over* X<sup>i</sup> ∈ P*. Let* Vθθ(x<sup>P</sup> ) *denote the* θ*-block of* V (x<sup>P</sup> ) *in* [\(37\)](#page-12-1)*. Denote by* ∥·∥<sup>V</sup> *the weighted Euclidean norm* ∥z∥<sup>V</sup> := ∥Vθθ(x<sup>P</sup> )z∥ q <sup>2</sup> = z⊤V ⊤ θθ (x<sup>P</sup> )Vθθ(x<sup>P</sup> )z*. Define the weighted oracle criterion* ∆<sup>V</sup> (C1, C2)*:*

$$\Delta_V(C_1, C_2) := \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \hat{\theta}_{C_1} - \hat{\theta}_{C_2} \right\|_V^2. \quad (27)$$

*Then, treating the split as fixed with* r <sup>−</sup><sup>2</sup> ≪ nC<sup>1</sup> , nC<sup>2</sup> *and* *sufficiently small* r > 0*,*

$$\tilde{\Delta}^{\text{FPT}}(C_1, C_2) = \Delta_V(C_1, C_2) + o_P\left(r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}}\right).$$

Lemma 4.2. *Let* T (∆) *denote a tree whose splitting mechanism seeks splits that maximize* ∆(C1, C2) *defined in* [\(8\)](#page-2-5)*, and let* T (∆<sup>V</sup> ) *denote a tree whose splitting mechanism seeks splits that maximize* ∆<sup>V</sup> (C1, C2) *defined in* [\(27\)](#page-5-2)*. Suppose Assumptions [A.1](#page-12-0) hold and assume moreover that* T (∆) *is a tree that satisfies Specifications [A.2.](#page-13-0) Then,* T (∆<sup>V</sup> ) *satisfies Specifications [A.2.](#page-13-0)*

For Stage II, Theorem [4.3](#page-6-2) establishes the consistency of the GRF-FPT estimator ˆθ(x):

Theorem 4.3. *Suppose that Assumptions [A.1](#page-12-0) hold, and let* ( ˆθ(x), νˆ(x)) *be estimates that solve* [\(5\)](#page-1-2) *based on weights induced by a forest of trees grown under the fixed-point tree algorithm satisfying Specifications [A.2.](#page-13-0) Then,* ( ˆθ(x), νˆ(x)) *converges in probability to* (θ ∗ (x), ν<sup>∗</sup> (x))*.*

The proof of Theorem [4.3](#page-6-2) follows directly from Theorem 3 of [Athey et al.](#page-9-2) [\(2019\)](#page-9-2), which, under Assumptions [A.1,](#page-12-0) establishes consistency for estimates ( ˆθ(x), νˆ(x)) that solve [\(5\)](#page-1-2) with weights from a forest that satisfies Specifications [1-](#page-13-2)[5.](#page-13-3) Thanks to Lemma [4.2,](#page-6-1) these forest specifications must also apply to a forest grown under the FPT mechanism. Specifications [1](#page-13-2)[-3](#page-13-4) collectively impose mild boundary conditions on the splitting procedure. Meanwhile, Specification [4](#page-13-5) requires that trees are trained on subsamples drawn without replacement [\(Biau et al.,](#page-9-13) [2008;](#page-9-13) [Scornet et al.,](#page-11-10) [2015;](#page-11-10) [Wager et al.,](#page-11-11) [2014;](#page-11-11) [Wager & Athey,](#page-11-1) [2018\)](#page-11-1), and Specification [5](#page-13-3) requires that trees must be grown using an additional subsample splitting mechanism known as honesty [\(Athey & Imbens,](#page-9-5) [2016;](#page-9-5) [Biau,](#page-9-14) [2012;](#page-9-14) [Denil et al.,](#page-9-15) [2014\)](#page-9-15). Appendix [C.1](#page-24-0) provides a detailed explanation of the subsampling and honest sample splitting procedure.

Finally, Theorem [4.4](#page-6-3) establishes the asymptotic normality of the GRF-FPT estimator ˆθ(x):

Theorem 4.4. *Under the conditions of Theorem [4.3,](#page-6-2) suppose moreover that Regularity Condition [1](#page-13-6) holds, and that a forest is grown on subsamples of size* s *scaling as* s = n β *, where* β *satisfies Regularity Condition [2.](#page-13-7) Then, there exists a sequence* σn(x) *such that* ( ˆθn(x) − θ ∗ (x))/σn(x) ⇝ N (0, 1) *and* σ 2 n (x) = polylog(n/s) −1 s/n*, where* polylog(n/s) *is a function that is bounded away from 0 and increases at most polynomially with the log of the inverse sampling ratio* log(n/s)*.*

The proof of Theorem [4.4](#page-6-3) is an immediate consequence of Theorem 5 of [Athey et al.](#page-9-2) [\(2019\)](#page-9-2). Theorems [4.3](#page-6-2) and [4.4](#page-6-3) demonstrate that the GRF-FPT estimator is able to meet key statistical guarantees.

#### 5. Applications

In this section, we explore applications of GRF-FPT for two related models: varying coefficient models and heterogeneous treatment effects. We consider an outcome model of the form introduced in Section [3.2.](#page-2-3) For each observation, let Y<sup>i</sup> denote the observed outcome, W<sup>i</sup> = (Wi,1, . . . , Wi,K) ⊤ a K-dimensional regressor, and X<sup>i</sup> a set of mediating auxiliary variables, such that

$$Y_i = \nu^*(X_i) + W_i^\top \theta^*(X_i) + \epsilon_i, \quad (28)$$

where ν ∗ (·) is a nuisance intercept function, θ ∗ (x) = (θ ∗ 1 (x), . . . , θ<sup>∗</sup> <sup>K</sup>(x))<sup>⊤</sup> are the target effect functions local to X<sup>i</sup> = x, under the assumptions <sup>E</sup>[ϵ<sup>i</sup> | X<sup>i</sup> = x] = 0 and <sup>E</sup>[ϵiW<sup>i</sup> | X<sup>i</sup> = x] = 0.

Varying coefficient models (VCM). Given regressors W<sup>i</sup> ∈ R <sup>K</sup>, models of the form [\(28\)](#page-6-4) can be characterized as varying coefficient models [\(Hastie & Tibshirani,](#page-10-19) [1993\)](#page-10-19). As discussed in Section [3.2,](#page-2-3) we must also assume that the regressors W<sup>i</sup> are conditionally exogenous given X<sup>i</sup> = x.

Heterogeneous treatment effects (HTE). A special case of [\(28\)](#page-6-4) arises within the Neyman-Rubin potential outcome framework, which models the causal effect of treatment on an outcome [\(Neyman,](#page-10-5) [1923;](#page-10-5) [Rubin,](#page-10-6) [1974\)](#page-10-6). Here, θ ∗ (x) = (θ ∗ (x), . . . , θ<sup>∗</sup> <sup>K</sup>(x))<sup>⊤</sup> represents heterogeneous treatment effects associated with K discrete treatment levels. Let T<sup>i</sup> ∈ {1, . . . , K} denote the observed treatment level for the i-th observation, and Yi(k) the potential outcome that would have been observed if treatment level k had been applied. The regressors W<sup>i</sup> ∈ {0, 1} <sup>K</sup> in [\(28\)](#page-6-4) are interpreted as a vector of dummy variables indicating the observed treatment level, Wi,k := <sup>1</sup>(T<sup>i</sup> = k). The auxiliary variables X<sup>i</sup> account for potential confounding effects. The conditional average treatment effect of treatment level k ∈ {2, . . . , K} relative to the baseline level k = 1 is then defined as:

$$\theta_k^*(x) := \mathbb{E} [Y_i(k) - Y_i(1) \mid X_i = x] ,$$

where the baseline contrast is set to θ ∗ 1 (x) := 0.

Under exogeneity of the regressors, the target effects θ ∗ (x) in models [\(28\)](#page-6-4) are identified by moment conditions [\(1\)](#page-0-0) for scoring function [\(Angrist & Pischke,](#page-9-16) [2009;](#page-9-16) [Athey et al.,](#page-9-2) [2019\)](#page-9-2)

$$\psi_{\theta,\nu}(Y_i, W_i) := \begin{bmatrix} (Y_i - W_i^\top \theta - \nu) \cdot W_i \\ Y_i - W_i^\top \theta - \nu \end{bmatrix}.$$

The gradient-based pseudo-outcomes [\(22\)](#page-4-4) are computed as

$$\rho_i^{\text{grad}} = -A_P^{-1}(W_i - \overline{W}_P) \left( Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \hat{\theta}_P \right), \quad (29)$$

where W<sup>P</sup> and Y <sup>P</sup> are the local means of W<sup>i</sup> and Y<sup>i</sup> over the observations in P. Centering Y<sup>i</sup> − Y <sup>P</sup> and W<sup>i</sup> − W<sup>P</sup>

removes the baseline effect of the mean νˆ<sup>P</sup> on ρ grad i , and where A<sup>P</sup> is given by [\(13\)](#page-3-7) as:

$$A_P = -\frac{1}{n_P} \sum_{\{i: X_i \in P\}} (W_i - \overline{W}_P)(W_i - \overline{W}_P)^\top. \quad (30)$$

Computing ρ grad i in [\(29\)](#page-6-5) involves the OLS coefficients ˆθ<sup>P</sup> from regressing Yi−Y <sup>P</sup> on Wi−W<sup>P</sup> , over the observations in P:

$$\hat{\theta}_P := -A_P^{-1} \frac{1}{n_P} \sum_{\{i: X_i \in P\}} (W_i - \overline{W}_P)(Y_i - \overline{Y}_P). \quad (31)$$

In comparison, ρ FPT i in [\(26\)](#page-5-3) are computed as:

$$\begin{aligned} \rho_i^{\text{FFT}} &:= -\xi^\top \psi_{\hat{\theta}_P, \hat{\nu}_P}(Y_i, W_i), \\ &= -(W_i - \overline{W}_P) \left( Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \hat{\theta}_P \right), \end{aligned} \quad (32)$$

The relationship ρ grad <sup>i</sup> = A −1 P ρ FPT i reveals a significant benefit of FPT pseudo-outcomes. The form of ρ FPT i eliminates the computational cost associated with the multiplication of A −1 P , leading to O(K<sup>3</sup> ) computational savings. Furthermore, the computation of ˆθ<sup>P</sup> in [\(32\)](#page-7-0) no longer requires solving for A −1 P . Therefore, we can further enhance computational efficiency by using an accelerated form of pseudo-outcome ϕ i instead of ρ i :

$$\phi_i^{\text{FPT}} := -(W_i - \overline{W}_P) \left( Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \tilde{\theta}_P \right), \quad (33)$$

where ˆθ<sup>P</sup> is replaced by ˜θ<sup>P</sup> in [\(32\)](#page-7-0), which is defined as a one-step gradient descent approximation of ˆθ<sup>P</sup> taken from the origin:

$$\tilde{\theta}_P := \gamma \frac{1}{n_P} \sum_{\{i: X_i \in P\}} (W_i - \overline{W}_P)(Y_i - \overline{Y}_P). \quad (34)$$

Here, γ denotes the exact line search step size for the regression of Y<sup>i</sup> − Y <sup>P</sup> on W<sup>i</sup> − W<sup>P</sup> over P:

$$\gamma := \frac{\|(W - \overline{W}_P)^\top (Y - \overline{Y}_P)\|_2^2}{\|(W - \overline{W}_P)(W - \overline{W}_P)^\top (Y - \overline{Y}_P)\|_2^2}, \quad (35)$$

where W = [W<sup>1</sup> · · · W<sup>n</sup><sup>P</sup> ] ⊤ and Y = [Y<sup>1</sup> · · · Y<sup>n</sup><sup>P</sup> ] <sup>⊤</sup> with the notation W −W<sup>P</sup> and Y −Y <sup>P</sup> understood as row-wise centering.

The computational cost associated with ˜θ<sup>P</sup> is comparatively small because many of the products that appear in [\(34\)](#page-7-1) and [\(35\)](#page-7-2) are already computed as part of ρ i in [\(32\)](#page-7-0). Meanwhile, we show in Appendix [B.3](#page-22-0) that the approximation for the FPT child estimator:

$$\bar{\theta}_{C_j}^{\text{FPT}} := \hat{\theta}_P + \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \phi_i^{\text{FPT}},$$

is consistent for the original FPT child estimator ˜θ C<sup>j</sup> as ˜θ <sup>C</sup><sup>j</sup> <sup>−</sup> ¯<sup>θ</sup> C<sup>j</sup> ∥ = o<sup>P</sup> (1), meaning that this approximation does not alter the asymptotic behavior of our estimator. These accelerations are particularly compelling when the dimension of θ ∗ (x) is large and computational efficiency is critical, as in large-scale A/B testing with multiple concurrent treatment arms or observational studies with numerous treatment levels [\(Kohavi et al.,](#page-10-4) [2013;](#page-10-4) [Bakshy et al.,](#page-9-17) [2014\)](#page-9-17).

#### 6. Simulations

In this section, we perform empirical evaluations of the computational efficiency and estimation accuracy of the GRF-FPT method. We let GRF-FPT1 denote the FPT algorithm using the exact form of the FPT VCM/HTE pseudooutcomes [\(32\)](#page-7-0) and we let GRF-FPT2 denote the accelerated FPT algorithm based on the form of the FPT pseudooutcome approximation [\(33\)](#page-7-3) in Section [5.](#page-6-0) We compare both implementations relative to GRF-grad under VCM and HTE designs. Implementation details and links to the reproducible code are found in Appendix [C.4.](#page-27-2)

Settings. We follow the structural model in [\(28\)](#page-6-4). The auxiliary variables X<sup>i</sup> are drawn from the Gaussian copula with latent covariance matrix Σ, where [Σ]j,k = (0.3)<sup>|</sup>j−k<sup>|</sup> . Supporting experiments for multicollinearity in X<sup>i</sup> can be found in Appendix [D.2.](#page-30-0) The outcomes Y<sup>i</sup> follow [\(28\)](#page-6-4) with Gaussian noise ϵ<sup>i</sup> ∼ N (0, 1). For VCM experiments, regressors W<sup>i</sup> ∈ <sup>R</sup> <sup>K</sup> are sampled from NK(0,I). For HTE experiments, W<sup>i</sup> ∈ {0, 1} <sup>K</sup> follows a multinomial distribution, W<sup>i</sup> | X<sup>i</sup> = x ∼ Multinomial(1,(π1(x), . . . , πK(x))), where πk(x) is the probability of treatment level k ∈ {1, . . . , K}, characterizing a variety of different locationspecific dependence structures through the setting of πk(·). We set ν ∗ (x) := 0 and vary the target effect functions θ ∗ k (x) and treatment probabilities πk(x) across different settings, fully detailed in Appendix [C.4.](#page-27-2) Throughout our experiments we use subsampling ratio s/n = 0.5. Supporting experiments under different subsample ratios are found in Appendix [D.2.](#page-30-0)

Results. The relative computational advantage of forests trained under GRF-FPT is displayed in Figure [3,](#page-8-0) while Figure [5](#page-31-0) (in Appendix [D.3\)](#page-31-1) summarizes the absolute fit times across the three methods. These data show that the FPT mechanism is able to consistently offer a relative advantage, observing speedups of up to 3.5× faster than the gradientbased approach at the largest dimension K = 256. Figure [3](#page-8-0) also shows increasing gains with increasing K and provides an empirical measurement of the theoretical scaling benefits discussed in Section [5.](#page-6-0) Moreover, the absolute fit times in Figure [5](#page-31-0) (in Appendix [D.3\)](#page-31-1) illustrate that our method consistently remains faster than GRF-grad, with no clear computational or algorithmic bottleneck as a function of either n or K. Supporting experiments exploring the ef-

![](_page_8_Figure_1.jpeg)

Figure 3: Speedup factor for GRF-FPT in comparison to GRFgrad for VCM timing experiments.

fects of sample sizes up to n = 500, 000 are presented in Appendix [D.2,](#page-30-0) while Figures [7](#page-33-0) and [8](#page-34-0) (in Appendix [D.3\)](#page-31-1) show that even when n is small, GRF-FPT still observes a noticeable gain relative to GRF-grad. Additional timing benchmarks for VCM experiments and all HTE experiments are discussed in Appendix [D.3.](#page-31-1)

To assess estimation accuracy, we evaluate the mean squared error (MSE) of ˆθ(x) across 50 replications of the model and testing on a separate set of 5, 000 observations. Figure [6](#page-32-0) in Appendix [D.3](#page-31-1) confirms that GRF-FPT matches the accuracy of GRF-grad, while significantly reducing computation time. Further comparisons for both VCM and HTE settings are provided in Appendix [D.3.](#page-31-1)

## 7. Real Data Application

Data. In this section we apply GRF-FPT to the analysis of geographically-varying effects θ ∗ (x) on housing prices. The data, first appearing in [Kelley Pace & Barry](#page-10-20) [\(1997\)](#page-10-20), contains 20,640 observations of housing prices taken from the 1990 California census. Each observation corresponds to measurements aggregated over a small geographical census block, and contains measurements of 9 variables: median housing value, longitude, latitude, median housing age, total rooms, total bedrooms, population, households, and median income. We employ a VCM design of the form [\(28\)](#page-6-4) where Y<sup>i</sup> denotes the housing value, X<sup>i</sup> denote the spatial coordinates, and W<sup>i</sup> = (Wi,1, . . . , Wi,6) <sup>⊤</sup> are the remaining six regressors. Details of the model and data transformations used for the California housing analysis is found in Appendix [F.](#page-37-0)

Results. Table [7](#page-38-0) summarizes the computational benefit of GRF-FPT applied to the California housing data. Figure [4](#page-8-1) illustrates the six geographically-varying effect estimates

![](_page_8_Figure_2.jpeg)

Figure 4: Geographically-varying GRF-FPT2 estimates ˆθ(x).

under GRF-FPT2, with qualitatively similar results shown in Figure [16](#page-42-0) for GRF-FPT1 and GRF-grad in Appendix [F.](#page-37-0) Figure [4](#page-8-1) shows clearly the geographically-dependent relationship between different housing features and housing prices. In major urban centers such as LA, San Francisco, and Sacramento, housing prices tend to decrease with an increasing number of households, and may reflect overcrowding in densely populated areas. In contrast, rural regions show the opposite trend: prices rise slightly when rural areas have a larger number of housing units. This suggests that, in sparsely populated rural areas, a modest increase in households makes these places more attractive and livable. Median income, however, consistently shows a positive effect on prices across nearly all of California, while population size tends to show a negative effect, highlighting broader state-wide pressures on housing affordability.

### 8. Conclusion

Our results demonstrate that the FPT algorithm offers a substantial computational advantage over GRF-grad with comparable statistical accuracy, and highlights GRF-FPT as a powerful method for multi-dimensional estimation, particularly when estimates of the target function must be learned from the data rather than observed directly. Future work may explore extensions to larger-scale problems and alternative estimation tasks, as in unsupervised learning and structured prediction. Our findings position GRF-FPT as a scalable and robust alternative for practitioners seeking efficient localized estimation.

- Impact Statement This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. Acknowledgments This work was supported by Natural Sciences and Engineering Research Council (NSERC) Discovery Grant (RGPIN-2024-06780) and FRQNT Team Research Project Grant (FRQ-NT 327788). References Amit, Y. and Geman, D. Shape quantization and recognition with randomized trees. *Neural Computation*, 9(7):1545– 1588, 1997. Angrist, J. D. and Pischke, J.-S. *Mostly harmless econometrics: An empiricist's companion*. Princeton university press, 2009. Athey, S. and Imbens, G. Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27):7353–7360, 2016. Athey, S., Tibshirani, J., and Wager, S. Generalized random forests. *The Annals of Statistics*, 47(2):1148 – 1178, 2019. doi: 10.1214/18-AOS1709. URL [https://](https://doi.org/10.1214/18-AOS1709) [doi.org/10.1214/18-AOS1709](https://doi.org/10.1214/18-AOS1709). Bakshy, E., Eckles, D., and Bernstein, M. S. Designing and deploying online field experiments. In *Proceedings of the 23rd International Conference on World Wide Web*, WWW '14, pp. 283–292, New York, NY, USA, 2014. Association for Computing Machinery. ISBN 9781450327442. doi: 10.1145/ 2566486.2567967. URL [https://doi.org/10.](https://doi.org/10.1145/2566486.2567967) [1145/2566486.2567967](https://doi.org/10.1145/2566486.2567967). Banach, S. Sur les operations dans les ensembles abstraits ´ et leur application aux equations int ´ egrales. ´ *Fundamenta Mathematicae*, 3:133–181, 1922. Belloni, A., Chernozhukov, V., Fernandez-Val, I., and Hansen, C. Program evaluation and causal inference with high-dimensional data. *Econometrica*, 85(1):233–298, 2017. Biau, G. Analysis of a random forests model. *The Journal of Machine Learning Research*, 13(1):1063–1095, 2012. Biau, G., Devroye, L., and Lugosi, G. Consistency of random forests and other averaging classifiers. *Journal of Machine Learning Research*, 9(66):2015–2033, 2008. URL [http://jmlr.org/papers/v9/biau08a.](http://jmlr.org/papers/v9/biau08a.html) [html](http://jmlr.org/papers/v9/biau08a.html). Breiman, L. Bagging predictors. *Machine Learning*, 24: 123–140, 1996. Breiman, L. Random forests. *Machine Learning*, 45:5–32, 2001. Breiman, L., Friedman, J., Olshen, R. A., and Stone, C. J. *Classification and Regression Trees*. CRC, 1984. ISBN 9780412048418. Cevid, D., Michel, L., Naf, J., B ¨ uhlmann, P., and Mein- ¨ shausen, N. Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. *Journal of Machine Learning Research*, 23(333):1– 79, 2022. URL [http://jmlr.org/papers/v23/](http://jmlr.org/papers/v23/21-0585.html) [21-0585.html](http://jmlr.org/papers/v23/21-0585.html). Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1):C1–C68, 01 2018. ISSN 1368-4221. doi: 10.1111/ectj.12097. URL <https://doi.org/10.1111/ectj.12097>. Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., and Zhu,
  - R. Estimating heterogeneous treatment effects with rightcensored data via causal survival forests. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 85(2):179–211, 02 2023. ISSN 1369-7412. doi: 10.1093/jrsssb/qkac001. URL [https://doi.org/](https://doi.org/10.1093/jrsssb/qkac001) [10.1093/jrsssb/qkac001](https://doi.org/10.1093/jrsssb/qkac001). De'ath, G. Multivariate regression trees: a new technique for modeling species–environment relationships. *Ecology*, 83(4):1105–1117, 2002. Denil, M., Matheson, D., and De Freitas, N. Narrowing the gap: Random forests in theory and in practice. In Xing, E. P. and Jebara, T. (eds.), *Proceedings of the 31st International Conference on Machine Learning*, volume 32 of *Proceedings of Machine Learning Research*, pp. 665–673, Bejing, China, 22–24 Jun 2014. PMLR. URL [https://proceedings.mlr.press/v32/](https://proceedings.mlr.press/v32/denil14.html) [denil14.html](https://proceedings.mlr.press/v32/denil14.html). Dietterich, T. G. An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting, and randomization. *Machine Learning*, 40:139–157, 2000. Fan, J. and Gijbels, I. *Local Polynomial Modelling and Its Applications*, volume 66 of *Monographs on Statistics and Applied Probability*. Chapman & Hall/CRC, London, 1996. doi: 10.1201/9780203748725. URL [https://www.taylorfrancis.com/books/](https://www.taylorfrancis.com/books/mono/10.1201/9780203748725) [mono/10.1201/9780203748725](https://www.taylorfrancis.com/books/mono/10.1201/9780203748725).

Fan, J., Heckman, N. E., and Wand, M. P. Local polynomial kernel regression for generalized linear models and quasilikelihood functions. *Journal of the American Statistical Association*, 90(429):141–150, 1995. Friedberg, R., Tibshirani, J., Athey, S., and Wager, S. Local linear forests. *Journal of Computational and Graphical Statistics*, 30(2):503–517, 2020. Friedman, J. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, pp. 1189–1232, 2001. Hastie, T. and Tibshirani, R. Varying-coefficient models. *Journal of the Royal Statistical Society. Series B (Methodological)*, 55(4):757–796, 1993. ISSN 00359246. URL <http://www.jstor.org/stable/2345993>. Imai, K. and Ratkovic, M. Estimating treatment effect heterogeneity in randomized program evaluation. *The Annals of Applied Statistics*, 7(1):443 – 470, 2013. doi: 10.1214/12-AOAS593. URL [https://doi.org/10.](https://doi.org/10.1214/12-AOAS593) [1214/12-AOAS593](https://doi.org/10.1214/12-AOAS593). Johansson, F., Shalit, U., and Sontag, D. Learning representations for counterfactual inference. In *International Conference on Machine Learning*, pp. 3020–3029. PMLR, 2016. Kelley Pace, R. and Barry, R. Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3):291–297, 1997. ISSN 0167-7152. doi: https://doi.org/10.1016/S0167-7152(96)00140-X. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S016771529600140X) [science/article/pii/S016771529600140X](https://www.sciencedirect.com/science/article/pii/S016771529600140X). Knaus, M. C., Lechner, M., and Strittmatter, A. Machine learning estimation of heterogeneous causal effects: Empirical monte carlo evidence. *The Econometrics Journal*, 24(1):134–161, 2021. Kohavi, R., Deng, A., Frasca, B., Walker, T., Xu, Y., and Pohlmann, N. Online controlled experiments at large scale. In *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD '13, pp. 1168–1176, New York, NY, USA, 2013. Association for Computing Machinery. ISBN 9781450321747. doi: 10.1145/ 2487575.2488217. URL [https://doi.org/10.](https://doi.org/10.1145/2487575.2488217) [1145/2487575.2488217](https://doi.org/10.1145/2487575.2488217). Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. Met- ¨ alearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10):4156–4165, 2019. Lee, Y., Veerubhotla, K., Jeong, M. H., and Lee, C. H. Deep learning in personalization of cardiovascular stents. *Journal of Cardiovascular Pharmacology and Therapeutics*, 25(2):110–120, 2020. Lewbel, A. A local generalized method of moments estimator. *Economics Letters*, 94(1):124–128, 2007. Lindelof, E. Sur l'application de la m ¨ ethode des approxima- ´ tions successives aux equations diff ´ erentielles ordinaires ´ du premier ordre. *Comptes Rendus Hebdomadaires des Seances de l'Acad ´ emie des Sciences ´* , 116:454–457, 1894. Murdoch, W. J., Singh, C., Kumbier, K., Abbasi-Asl, R., and Yu, B. Definitions, methods, and applications in interpretable machine learning. *Proceedings of the National Academy of Sciences*, 116(44):22071–22080, 2019. Newey, W. K. Kernel estimation of partial means and a general variance estimator. *Econometric Theory*, 10(2): 1–21, 1994. Neyman, J. Sur les applications de la theorie des prob- ´ abilites aux experiences agricoles: Essai des principes. ´ *Roczniki Nauk Rolniczych*, 10(1):1–51, 1923. Reprinted and translated in Neyman, J. (1990). Statistical Science, 5(4), 463–480. Nie, X. and Wager, S. Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2):299–319, 2021. Picard, E. M´ emoire sur la th ´ eorie des ´ equations aux d ´ eriv ´ ees ´ partielles et la methode des approximations successives. ´ *Journal de Mathematiques Pures et Appliqu ´ ees ´* , 6:145– 210, 1890. Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., and Tibshirani, R. Some methods for heterogeneous treatment effect estimation in high dimensions. *Statistics in Medicine*, 37(11):1767–1787, 2018. Robins, J. M. and Ritov, Y. Toward a curse of dimensionality appropriate (coda) asymptotic theory for semiparametric models. *Statistics in Medicine*, 16(1-3):285– 319, 1997. doi: 10.1002/(SICI)1097-0258(19970215)16: 3⟨285::AID-SIM535⟩3.0.CO;2-\#. URL [https://](https://pubmed.ncbi.nlm.nih.gov/9004398/) [pubmed.ncbi.nlm.nih.gov/9004398/](https://pubmed.ncbi.nlm.nih.gov/9004398/). Robinson, P. M. Root-n-consistent semiparametric regression. *Econometrica: Journal of the Econometric Society*, pp. 931–954, 1988. Rubin, D. B. Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5):688, 1974.

Ryu, E. K. and Boyd, S. A primer on monotone operator methods (survey). *Applied and Computational Mathematics*, 15(1):3–43, 2016. Survey article. Scornet, E., Biau, G., and Vert, J.-P. Consistency of random forests. *The Annals of Statistics*, 43(4):1716 – 1741, 2015. doi: 10.1214/15-AOS1321. URL [https://](https://doi.org/10.1214/15-AOS1321) [doi.org/10.1214/15-AOS1321](https://doi.org/10.1214/15-AOS1321). Segal, M. R. Tree-structured methods for longitudinal data. *Journal of the American Statistical Association*, 87(418): 407–418, 1992. Severini, T. A. and Staniswalis, J. G. Quasi-likelihood estimation in semiparametric models. *Journal of the American Statistical Association*, 89(426):501–511, 1994. Shalit, U., Johansson, F. D., and Sontag, D. Estimating individual treatment effect: generalization bounds and algorithms. In Precup, D. and Teh, Y. W. (eds.), *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pp. 3076–3085. PMLR, 06–11 Aug 2017. URL [https://proceedings.mlr.press/v70/](https://proceedings.mlr.press/v70/shalit17a.html) [shalit17a.html](https://proceedings.mlr.press/v70/shalit17a.html). Speckman, P. Kernel smoothing in partial linear models. *Journal of the Royal Statistical Society. Series B (Methodological)*, 50(3):413–436, 1988. ISSN 00359246. URL <http://www.jstor.org/stable/2345705>. Staniswalis, J. G. The kernel estimate of a regression function in likelihood-based models. *Journal of the American Statistical Association*, 84(405):276–283, 1989. Tibshirani, J., Athey, S., Sverdrup, E., and Wager, S. *grf: Generalized Random Forests*, 2024. URL [https://](https://github.com/grf-labs/grf) [github.com/grf-labs/grf](https://github.com/grf-labs/grf). R package version 2.4.0. Wager, S. and Athey, S. Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523):1228– 1242, 2018. Wager, S. and Walther, G. Adaptive concentration of regression trees, with application to random forests. *arXiv preprint arXiv:1503.06388*, 2015. Wager, S., Hastie, T., and Efron, B. Confidence intervals for random forests: The jackknife and the infinitesimal jackknife. *Journal of Machine Learning Research*, 15 (1):1625–1651, 2014. URL [https://jmlr.org/](https://jmlr.org/papers/volume15/wager14a/wager14a.pdf) [papers/volume15/wager14a/wager14a.pdf](https://jmlr.org/papers/volume15/wager14a/wager14a.pdf). Yang, Y., Gu, Y., Zhao, Y., and Fan, J. Flexible regularized estimating equations: Some new perspectives. 2021. URL <https://arxiv.org/abs/2110.11074>. Zeileis, A. and Hornik, K. Generalized m-fluctuation tests for parameter instability. *Statistica Neerlandica*, 61(4): 488–508, 2007. Zeileis, A., Hothorn, T., and Hornik, K. Model-based recursive partitioning. *Journal of Computational and Graphical Statistics*, 17(2):492–514, 2008.

#### A. Technical Preliminaries

#### A.1. Assumptions

We follow the key assumptions of [Athey et al.](#page-9-2) [\(2019\)](#page-9-2) made for the theoretical analyses of GRF. The predictor and parameter spaces are both subsets of Euclidean space such that x ∈ X = [0, 1]<sup>p</sup> and (θ, ν) ∈ B ⊂ <sup>R</sup> <sup>K</sup>, where B is a compact subset of R <sup>K</sup>. Under the analyses of [Wager & Walther](#page-11-12) [\(2015\)](#page-11-12), we suppose that the features of the auxiliary covariates X<sup>i</sup> = (Xi,1, . . . , Xi,p) <sup>⊤</sup> have density f<sup>X</sup> that is bounded away from 0 and ∞, i.e. c ≤ fX(x) ≤ C < ∞, for some constants c > 0 and C < ∞. GRF does not require that the score function ψ is continuous in (θ, ν), as is the case for quantile estimation, one does require that the expected score/moment function

$$M_{\theta,\nu}(x) := \mathbb{E}_{O|X} [\psi_{\theta,\nu}(O) \mid X = x], \quad (36)$$

is smoothly varying in its parameters (θ, ν).

ASSUMPTION 1. For fixed (θ, ν), the M-function [\(36\)](#page-12-0) is Lipschitz continuous in x.

ASSUMPTION 2. For fixed x, the M-function is twice-differentiable in (θ, ν) with uniformly bounded second derivative,

$$\left\| \nabla_{(\theta, \nu)}^2 M_{\theta, \nu}(x) \right\| < \infty,$$

where · denotes the appropriate tensor norm for the second derivative of Mθ,ν taken with respect to (θ, ν). Let V (x) := ∇(θ,ν)Mθ,ν(x) θ=θ <sup>∗</sup>(x),ν=ν<sup>∗</sup>(x) denote the population Jacobian at the true (θ ∗ (x), ν<sup>∗</sup> (x)), and assume that V (x) is invertible for all x ∈ X . We write V (x) in block form as

$$V(x) = \begin{bmatrix} V_{\theta\theta}(x) & V_{\theta\nu}(x) \\ V_{\nu\theta}(x) & V_{\nu\nu}(x) \end{bmatrix}. \quad (37)$$

ASSUMPTION 3. The score functions ψθ,ν(Oi) have a continuous covariance structure in the following sense: Let γ(·, ·) denote the worst-case variogram:

$$\gamma \left( \begin{bmatrix} \theta_1 \\ \nu_1 \end{bmatrix}, \begin{bmatrix} \theta_2 \\ \nu_2 \end{bmatrix} \right) := \sup_{x \in \mathcal{X}} \{ \| \text{Var}_{O|X}(\psi_{\theta_1, \nu_1}(O_i) - \psi_{\theta_2, \nu_2}(O_i) \mid X_i = x) \|_F \},$$

then, for some L > 0,

$$\gamma \left( \begin{bmatrix} \theta_1 \\ \nu_1 \end{bmatrix}, \begin{bmatrix} \theta_2 \\ \nu_2 \end{bmatrix} \right) \leq L \left\| \begin{bmatrix} \theta_1 \\ \nu_1 \end{bmatrix} - \begin{bmatrix} \theta_2 \\ \nu_2 \end{bmatrix} \right\|_2, \quad \text{for all } (\theta_1, \nu_1), (\theta_2, \nu_2).$$

ASSUMPTION 4. The score function ψθ,ν(Oi) can be written as

$$\psi_{\theta,\nu}(O_i) = \lambda(\theta,\nu;O_i) + \zeta_{\theta,\nu}(g(O_i)),$$

where λ is Lipschitz-continuous in (θ, ν), g : {Oi} → <sup>R</sup> a univariate summary of the observables O<sup>i</sup> , and ζ<sup>θ</sup> : <sup>R</sup> → <sup>R</sup> any family of monotone and bounded functions.

ASSUMPTION 5. For any weights α<sup>i</sup> with Pα<sup>i</sup> = 1, the minimizer ( ˆθ, νˆ) of the weighted empirical estimation problem [\(5\)](#page-1-2) satisfies:

$$\left\| \sum_{i=1}^n \alpha_i \psi_{\theta, \nu}(O_i) \right\|_2 \leq C \max_{1 \leq i \leq n} \{\alpha_i\}, \quad \text{for } C \geq 0.$$

ASSUMPTION 6. The score function ψθ,ν(Oi) is a negative subgradient of a convex function, and the moment function Mθ,ν(Xi) is the negative gradient of a strongly convex function.

#### A.2. Forest specifications

The consistency and asymptotic normality results, Theorems [4.3](#page-6-2) and [4.4,](#page-6-3) require that the forest trained following Algorithm [2](#page-28-0) consists of trees that satisfy a certain set of specifications. These forest specifications are precisely those imposed by [Athey](#page-9-2) [et al.](#page-9-2) [\(2019\)](#page-9-2) for forests of gradient-based trees, and collectively, these specifications describe fairly mild conditions on the tree splitting mechanism, as well as specific requirements for the sampling procedure.

SPECIFICATION 1. (*Symmetric*) Tree estimates are invariant to permutations of the training indices. In other words, the output of a tree does not depend on the order in which the training samples are indexed.

SPECIFICATION 2. (*Balanced/*ω*-regular*) The proportion of parent observations assigned into either child is bound below by some ω > 0, i.e. nC<sup>j</sup> ≥ ωn<sup>P</sup> .

SPECIFICATION 3. (*Randomized/random-split*) The probability of splitting along any feature/dimension of the input space is bound below by some π > 0.

SPECIFICATION 4. (*Subsampling*) Trees are trained on subsample of size s, drawn without replacement from n training samples, where s/n → 0 as s → ∞.

SPECIFICATION 5. (*Honesty*) Trees are trained using the sample splitting procedure described in Appendix [C.1.](#page-24-0)

### A.3. Regularity conditions

REGULARITY CONDITION 1. Let V (x) be as defined in Assumption [2](#page-12-2) and let ρ ∗ i (x) denote the influence function of the i-th observation with respect to the target θ ∗ (x):

$$\rho_i^*(x) := -\xi^\top V(x)^{-1}\psi_{\theta^*(x),\nu^*(x)}(O_i).$$

Then,

$$\text{Var}(\rho_i^*(x) \mid X_i = x) > 0,$$
 for all  $x \in \mathcal{X}$ .

REGULARITY CONDITION 2. Trees are grown on subsamples of size s scaling as s = n β , for some subsample scaling exponent β bound according to βmin < β < 1, such that

$$\beta_{\min} := 1 - \left( 1 + \frac{1}{\pi} \cdot \frac{\log(\omega^{-1})}{\log((1-\omega)^{-1})} \right)^{-1} < \beta < 1,$$

where 0 < π, ω < 1 are constants defined in forest Specifications [2](#page-13-8) and [3.](#page-13-4)

#### A.4. Neyman orthogonality

To identify the underlying local parameters (θ ∗ (x), ν<sup>∗</sup> (x)) ∈ <sup>R</sup> <sup>K</sup> one must have a score ψθ,ν(O) with at least K = K<sup>θ</sup> +K<sup>ν</sup> components, where here we use K<sup>θ</sup> and K<sup>ν</sup> to denote the dimensions of the component subvectors θ ∗ (x) ∈ <sup>R</sup> <sup>K</sup><sup>θ</sup> and ν ∗ (x) ∈ <sup>R</sup> <sup>K</sup><sup>ν</sup> . Conceptually, a score ψθ,ν(O) can be partitioned into the components that identify the θ-coordinates, denoted by ψ1, and those that identify the ν-coordinates, denoted by ψ2, and thus the moment functions Mθ,ν(x) in [\(36\)](#page-12-0) can also be partitioned the same way:

$$\psi_{\theta,\nu}(O) = \begin{bmatrix} \psi_1(\theta, \nu; O) \\ \psi_2(\theta, \nu; O) \end{bmatrix}, \quad M_{\theta,\nu}(x) = \begin{bmatrix} M_1(\theta, \nu; x) \\ M_2(\theta, \nu; x) \end{bmatrix} = \begin{bmatrix} \mathbb{E}[\psi_1(\theta, \nu; O) \mid X = x] \\ \mathbb{E}[\psi_2(\theta, \nu; O) \mid X = x] \end{bmatrix}.$$

The corresponding Jacobian matrix of Mθ,ν(x) taken with respect to (θ, ν) and evaluated at the truth (θ ∗ (x), ν<sup>∗</sup> (x)) is

$$V(x) = \nabla_{(\theta, \nu)} M(\theta, \nu; x)|_{\theta=\theta^*(x), \nu=\nu^*(x)} = \begin{bmatrix} V_{\theta\theta}(x) & V_{\theta\nu}(x) \\ V_{\nu\theta}(x) & V_{\nu\nu}(x) \end{bmatrix},$$

where here the subscripts in the block expressions of V (x) indicate the coordinates with which the gradient is taken, and in all cases are evaluated at the truth (θ ∗ (x), ν<sup>∗</sup> (x)):

$$\begin{aligned} V_{\theta\theta}(x) &= \nabla_{\theta} M_1(\theta, \nu; x)|_{\theta=\theta^*(x), \nu=\nu^*(x)}, \\ V_{\theta\nu}(x) &= \nabla_{\nu} M_1(\theta, \nu; x)|_{\theta=\theta^*(x), \nu=\nu^*(x)}, \\ V_{\nu\theta}(x) &= \nabla_{\theta} M_2(\theta, \nu; x)|_{\theta=\theta^*(x), \nu=\nu^*(x)}, \\ V_{\nu\nu}(x) &= \nabla_{\nu} M_2(\theta, \nu; x)|_{\theta=\theta^*(x), \nu=\nu^*(x)}. \end{aligned}$$

In this context, the assumption of Neyman orthogonal moment conditions is more completely labeled as Neyman orthogonality for the estimation of θ ∗ (x) with respect to the nuisance ν ∗ (x), and can be summarized as an assumption that the moment conditions for θ ∗ (x) are insensitive to first-order changes in ν around the truth ν ∗ (x) whenever θ = θ ∗ (x). For GRF, this means that one assumes [\(1\)](#page-0-0) satisfies M1(θ ∗ (x), ν<sup>∗</sup> (x); x) = 0, and in other words, the partial derivatives of the moment functions for θ ∗ (x) with respect to ν are zero at (θ ∗ (x), ν<sup>∗</sup> (x)):

$$V_{\theta\nu}(x) = \mathbf{0}.$$

## A.5. Example: Neyman orthogonality for VCM and HTE

Consider the VCM/HTE model with data (Y<sup>i</sup> , W<sup>i</sup> , Xi) related according to

$$\mathbb{E}[Y_i \mid X_i = x] = \nu^*(x) + W_i^\top \theta^*(x),$$

such that, as discussed in Section [3.2,](#page-2-3) the score function ψθ,ν that identifies the underlying (θ ∗ (x), ν<sup>∗</sup> (x)) is

$$\psi_{\theta,\nu}(Y_i, W_i) := \begin{bmatrix} (Y_i - W_i^\top \theta - \nu)W_i \\ Y_i - W_i^\top \theta - \nu \end{bmatrix},$$

and the corresponding local Jacobian V (x) has block form

$$V(x) = -\mathbb{E} \left[ \begin{bmatrix} W_i W_i^\top & W_i^\top \\ W_i & 1 \end{bmatrix} \middle| X_i = x \right] = - \begin{bmatrix} \mathbb{E}[W_i W_i^\top \mid X_i = x] & \mathbb{E}[W_i^\top \mid X_i = x] \\ \mathbb{E}[W_i \mid X_i = x] & 1 \end{bmatrix}.$$

Therefore, for Neyman orthogonality to hold one requires that <sup>E</sup>[W<sup>i</sup> | X<sup>i</sup> = x] = 0.

## B. Derivations and Proofs

### B.1. Proofs for Section [3.4](#page-4-5)

### B.1.1. MULTIVARIATE CART CRITERIA

Let ρ<sup>i</sup> ∈ <sup>R</sup> <sup>K</sup> be vector-valued responses associated with covariates X<sup>i</sup> ∈ P. A standard CART split (C1, C2) of P minimizes the conventional least-squares criterion:

$$\sum_{\{i: X_i \in C_1\}} \|\rho_i - \bar{\rho}_{C_1}\|^2 + \sum_{\{i: X_i \in C_2\}} \|\rho_i - \bar{\rho}_{C_2}\|^2, \quad (38)$$

where ρ¯C<sup>j</sup> := n −1 C<sup>j</sup> P {i:Xi∈C<sup>j</sup> } ρi is the local prediction over child node C<sup>j</sup> . We verify that a split (C1, C2) minimizes [\(38\)](#page-14-1) if and only if it maximizes

$$n_{C_1} \|\bar{\rho}_{C_1}\|^2 + n_{C_2} \|\bar{\rho}_{C_2}\|^2. \quad (39)$$

*Proof.* Each sum in [\(38\)](#page-14-1) can be expanded as

$$\begin{aligned} \sum_{\{i: X_i \in C_j\}} \|\rho_i - \bar{\rho}_{C_j}\|^2 &= \sum_{\{i: X_i \in P\}} \|\rho_i - \bar{\rho}_{C_j}\|^2 \cdot \mathbf{1}(X_i \in C_j), \\ &= \sum_{\{i: X_i \in P\}} \left( \|\rho_i\|^2 - 2\rho_i^\top \bar{\rho}_{C_j} + \|\bar{\rho}_{C_j}\|^2 \right) \cdot \mathbf{1}(X_i \in C_j), \\ &= \sum_{\{i: X_i \in P\}} \|\rho_i\|^2 \cdot \mathbf{1}(X_i \in C_j) - n_{C_j} \|\bar{\rho}_{C_j}\|^2. \end{aligned}$$

Therefore, the least-squares criterion CART [\(38\)](#page-14-1) is equivalently written as

$$\begin{aligned} \sum_{j=1,2} \sum_{\{i: X_i \in C_j\}} \|\rho_i - \bar{\rho}_{C_j}\|^2 &= \sum_{j=1,2} \left( \sum_{\{i: X_i \in P\}} \|\rho_i\|^2 \cdot \mathbf{1}(X_i \in C_j) - n_{C_j} \|\bar{\rho}_{C_j}\|^2 \right), \\ &= \sum_{j=1,2} \left( \sum_{\{i: X_i \in P\}} \|\rho_i\|^2 \cdot \mathbf{1}(X_i \in C_j) \right) - \left( n_{C_1} \|\bar{\rho}_{C_1}\|^2 + n_{C_2} \|\bar{\rho}_{C_2}\|^2 \right), \\ &= \sum_{\{i: X_i \in P\}} \|\rho_i\|^2 - \left( n_{C_1} \|\bar{\rho}_{C_1}\|^2 + n_{C_2} \|\bar{\rho}_{C_2}\|^2 \right). \end{aligned}$$

The first term does not depend on the choice of split, and therefore the split that minimizes [\(38\)](#page-14-1) is equivalent to the split that maximizes [\(39\)](#page-14-2). ■

#### B.1.2. SPLITS VIA CART ON PSEUDO-OUTCOMES

The following result is a generalization to the claim made in Section [3.4](#page-4-5) that a CART split on pseudo-outcomes ρ <sup>i</sup> will produce a split that maximizes the <sup>∆</sup>e FPT-criterion, and is sufficiently general to cover gradient-based pseudo-outcomes ρ grad i and the corresponding <sup>∆</sup>e grad-criterion.

Lemma B.1. *Suppose we can write*

$$\tilde{\theta}_{C_j} = a + \frac{1}{n_{C_j}} \sum_{\{x_i \in C_j\}} \rho_i, \quad \rho_i = -B \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i), \quad (40)$$

*where* a *and* B *denote appropriately sized vectors and matrices whose values do not depend on the candidate child node* C<sup>j</sup> *. Under Assumptions [A.1,](#page-12-0) the split* (C1, C2) *that maximizes*

$$\tilde{\Delta}(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \tilde{\theta}_{C_1} - \tilde{\theta}_{C_2} \right\|^2,$$

*is exactly the split chosen by CART for vector-valued responses* ρ<sup>i</sup> *fit over covariates* X<sup>i</sup> ∈ P*.*

Proof of Lemma [B.1.](#page-15-0) The scores ψθ,ν(Oi) satisfy subgradient conditions by Assumption [6,](#page-12-3) and therefore the parent solutions ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) satisfy the first-order conditions

$$\sum_{\{i: X_i \in P\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) = \mathbf{0}.$$

Hence,

$$\begin{aligned} \mathbf{0} &= \sum_{\{i: X_i \in P\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) = \sum_{\{i: X_i \in C_1\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) + \sum_{\{i: X_i \in C_2\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i), \\ &= -B \left( \sum_{\{i: X_i \in C_1\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) + \sum_{\{i: X_i \in C_2\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i) \right), \\ &= \sum_{\{i: X_i \in C_1\}} \rho_i + \sum_{\{i: X_i \in C_2\}} \rho_i. \end{aligned}$$

Each sum in the previous expression is equivalently written as Pρ<sup>i</sup> = nC<sup>j</sup> ( ˜θ<sup>C</sup><sup>j</sup> − <sup>a</sup>). Hence,

$$\begin{aligned} \mathbf{0} &= \sum_{\{i: X_i \in C_1\}} \rho_i + \sum_{\{i: X_i \in C_2\}} \rho_i, \\ &= n_{C_1}(\tilde{\theta}_{C_1} - a) + n_{C_2}(\tilde{\theta}_{C_2} - a), \\ \iff a &= \frac{n_{C_1}}{n_P} \tilde{\theta}_{C_1} + \frac{n_{C_2}}{n_P} \tilde{\theta}_{C_2}. \end{aligned}$$

Writing ρ¯C<sup>j</sup> := nCj P {i:Xi∈C<sup>j</sup> } ρi , one has:

$$\begin{aligned}\bar{\rho}_{C1} &= \tilde{\theta}_{C1} - a, \\ &= \tilde{\theta}_{C1} - \frac{n_{C1}}{n_P} \tilde{\theta}_{C1} - \frac{n_{C2}}{n_P} \tilde{\theta}_{C2}, \\ &= \frac{n_{C2}}{n_P} \left( \tilde{\theta}_{C1} - \tilde{\theta}_{C2} \right),\end{aligned}$$

and

$$\frac{n_{C_1}}{n_P} \|\bar{\rho}_{C_1}\|^2 = \frac{n_{C_1} n_{C_2}^2}{n_P^3} \left\| \tilde{\theta}_{C_1} - \tilde{\theta}_{C_2} \right\|^2.$$

Applying analogous arguments with respect to C2, one has the symmetric result:

$$\frac{n_{C_2}}{n_P} \|\bar{\rho}_{C_2}\|^2 = \frac{n_{C_2} n_{C_1}^2}{n_P^3} \left\| \tilde{\theta}_{C_1} - \tilde{\theta}_{C_2} \right\|^2.$$

Therefore,

$$\begin{aligned} \frac{1}{n_P} \left( n_{C_1} \|\bar{\rho}_{C_1}\|^2 + n_{C_2} \|\bar{\rho}_{C_2}\|^2 \right) &= \frac{n_{C_1} n_{\bar{C}_2}^2}{n_P^3} \left\| \tilde{\theta}_{C_1} - \tilde{\theta}_{C_2} \right\|^2 + \frac{n_{C_2} n_{\bar{C}_1}^2}{n_P^3} \left\| \tilde{\theta}_{C_1} - \tilde{\theta}_{C_2} \right\|^2, \\ &= \frac{n_{C_1} n_{C_1}}{n_P^2} \left\| \tilde{\theta}_{C_1} - \tilde{\theta}_{C_2} \right\|^2, \\ &= \tilde{\Delta}(C_1, C_2). \end{aligned}$$

Based on the arguments in Appendix [B.1.1,](#page-14-0) a split (C1, C2) maximizes nC<sup>1</sup> ∥ρ¯C<sup>1</sup> <sup>2</sup> <sup>+</sup>nC<sup>2</sup> ∥ρ¯C<sup>2</sup> 2 if and only if it is a CART split performed on the <sup>ρ</sup><sup>i</sup> over <sup>P</sup>. That is, ∆( e <sup>C</sup>1, C2) is precisely maximized by a single CART split on <sup>ρ</sup><sup>i</sup> <sup>=</sup> <sup>−</sup>Bψθˆ<sup>P</sup> ,νˆ<sup>P</sup> (Oi) fit over covariates X<sup>i</sup> ∈ P, as desired. ■

#### B.1.3. SCALE INVARIANCE OF CART SPLITS

Lemma B.2 (Argmax equivalence of FPT criteria). *The optimal split identified by CART on pseudo-outcomes* ρ i *of the form* [\(23\)](#page-4-2) *does not depend on the scale factor* η*, for any* η ̸= 0*.*

Proof of Lemma [B.2.](#page-16-0) Denote by ρ (η) <sup>i</sup> FPT pseudo-outcomes based on an arbitrary scale factor η ̸= 0 of the form [\(23\)](#page-4-2):

$$\rho_i^{(\eta)} := -\eta \xi^\top \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i), \quad (41)$$

and let ψ<sup>C</sup><sup>j</sup> denote the child-leaf average score evaluated at the parent solution ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ):

$$\overline{\psi}_{C_j} := \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i),$$

such that the corresponding child-leaf pseudo-outcome averages ρ¯ (η) C<sup>j</sup> := nCj P {i:Xi∈C<sup>j</sup> } ρ (η) i are equivalently written as

$$\bar{\rho}_{C_j}^{(\eta)} = -\eta\xi^\top \bar{\psi}_{C_j}$$

Let <sup>∆</sup>e FPT η (C1, C2) denote the FPT criterion of the form [\(25\)](#page-4-3) based on pseudo-outcomes [\(41\)](#page-16-2):

$$\tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \bar{\rho}_{C_1}^{(n)} - \bar{\rho}_{C_j}^{(n)} \right\|^2 = \frac{n_{C_1} n_{C_2}}{n_P^2} \left\| \eta \xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2}) \right\|^2.$$

$$\|\eta \xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2})\|^2 = \eta^2 \|\xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2})\|^2,$$

and hence the <sup>∆</sup>e FPT η -criteria obey the scaling relation:

$$\tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) = \eta^2 \cdot \tilde{\Delta}_1^{\text{FPT}}(C_1, C_2), \quad (42)$$

where <sup>∆</sup>e FPT <sup>1</sup> denotes the FPT criterion induced by pseudo-outcomes ρ (1) i based on unit scale factor η = 1. The relation [\(42\)](#page-17-0) implies that any nonzero split-independent rescaling ρ (η) <sup>i</sup> = ηρ (1) <sup>i</sup> will induce a splitting criterion <sup>∆</sup>e FPT η (C1, C2) with the same maximizer as <sup>∆</sup>e FPT (C1, C2):

$$\arg \max_{(C_1, C_2)} \left\{ \tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) \right\} = \arg \max_{(C_1, C_2)} \left\{ \eta^2 \cdot \tilde{\Delta}_1^{\text{FPT}}(C_1, C_2) \right\} = \arg \max_{(C_1, C_2)} \left\{ \tilde{\Delta}_1^{\text{FPT}}(C_1, C_2) \right\}.$$

Intuitively, a CART split is chosen by ranking the criterion values among the candidate splits and selecting the maximizing split (C1, C2). Therefore, the FPT splitting mechanism is unaffected by the scale factor η used to specify fixed-point pseudo-outcomes [\(23\)](#page-4-2). The absolute scale of the <sup>∆</sup>e FPT-criterion does not matter when searching for the optimal split, and only the criterion rankings across the candidate splits determine the final partition. ■

#### B.2. Proofs for Section [4](#page-5-4)

## Notation and definitions.

- Let o<sup>P</sup> (a, b, c) := o<sup>P</sup> (max{a, b, c}), with an analogous abbreviation for O<sup>P</sup> (·).
- For a fixed parent node P, denote by x<sup>P</sup> the center of mass of the X<sup>i</sup> ∈ P, and let r := sup{i:Xi∈<sup>P</sup> } ∥X<sup>i</sup> − x<sup>P</sup> ∥ denote the radius of the parent P. Throughout, we consider an asymptotic regime where nC<sup>j</sup> → ∞ and r → 0, corresponding to leaves over X of vanishing radius. Further, r and nC<sup>j</sup> are related under the conditions of GRF Proposition 1, namely, r <sup>−</sup><sup>2</sup> ≪ nC<sup>j</sup> and hence nC<sup>j</sup> r <sup>2</sup> → ∞ and 1/ √<sup>n</sup>C<sup>j</sup> <sup>=</sup> <sup>o</sup>(r).
- Let θ ∗ C<sup>j</sup> denote the true parameter expectation over the child node:

$$\theta_{C_j}^* := \mathbb{E}[\theta^*(X) \mid X \in C_j], \quad j = 1, 2, \quad (43)$$

and let ˜θ ∗ C<sup>j</sup> (x<sup>P</sup> ) denote an oracle version of the gradient-based leaf statistic:

$$\tilde{\theta}_{C_j}^*(x_P) := \theta^*(x_P) - \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \xi^\top V(x_P)^{-1} \psi_{\theta^*(x_P), \nu^*(x_P)}(O_i),$$

where V (x) is the underlying local Jacobian in Assumption [2.](#page-12-2) Equivalently, in terms of the oracle pseudooutcome/influence function ρ ∗ i (·) defined in Regularity Condition [1,](#page-13-6)

$$\tilde{\theta}_{C_j}^*(x_P) := \theta^*(x_P) + \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \rho_i^*(x_P).$$

The following are technical lemmas used for the proof of Proposition [4.1.](#page-5-1)

Lemma B.3. *Suppose Assumptions [A.1](#page-12-0) and Specifications [A.2](#page-13-0) hold. Then,*

$$\Delta(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \|\theta_{C_1}^* - \theta_{C_2}^*\|^2 + o_P\left(r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}}\right).$$

Proof of Lemma [B.3.](#page-17-1) Write the difference <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> − <sup>θ</sup> ∗ C<sup>j</sup> as

$$\hat{\theta}_{C_j} - \theta_{C_j}^* = \underbrace{\left( \hat{\theta}_{C_j} - \tilde{\theta}_{C_j}^*(x_P) \right)}_{T_1} + \underbrace{\left( \tilde{\theta}_{C_j}^*(x_P) - \mathbb{E}[\tilde{\theta}_{C_j}^*(x_P) \mid X \in C_j] \right)}_{T_2} + \underbrace{\left( \mathbb{E}[\tilde{\theta}_{C_j}^*(x_P) \mid X \in C_j] - \theta_{C_j}^* \right)}_{T_3}.$$

Under standard LLN arguments, the second term satisfies T<sup>2</sup> = O<sup>P</sup> (1/ √<sup>n</sup>C<sup>j</sup> ), and in an asymptotic regime with r <sup>−</sup><sup>2</sup> ≪ nC<sup>j</sup> one has T<sup>2</sup> = o<sup>P</sup> (r). Meanwhile, the first and third terms appear in the proofs of Propositions 2 and 1 of [Athey et al.](#page-9-2) [\(2019\)](#page-9-2), respectively, and satisfy T<sup>1</sup> = o<sup>P</sup> (r, 1/ √<sup>n</sup>C<sup>j</sup> ) and T<sup>3</sup> = O(r 2 ) =⇒ T<sup>3</sup> = o(r). It follows

$$\hat{\theta}_{C_j} - \theta_{C_j}^* = o_P\left(r, 1/\sqrt{n_{C_j}}\right),$$

and in particular

$$\hat{\theta}_{C_1} - \hat{\theta}_{C_2} = \theta_{C_1}^* - \theta_{C_2}^* + o_P\left(r, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}}\right).$$

Write A = θ ∗ <sup>C</sup><sup>1</sup> − θ ∗ C<sup>2</sup> and let E be any term satisfying E = o<sup>P</sup> (r, 1/ √<sup>n</sup>C<sup>1</sup> , 1/ √<sup>n</sup>C<sup>2</sup> ) such that ∆(C1, C2) is equivalently written ∆(C1, C2) = (nC<sup>1</sup> nC<sup>2</sup> /n<sup>2</sup> P ) · ∥A + E∥ 2 . Consider the difference

$$\begin{aligned} \Delta(C_1, C_2) - \frac{n_{C_1} n_{C_2}}{n_P^2} \|\theta_{C_1}^* - \theta_{C_2}^*\|^2 &= \frac{n_{C_1} n_{C_2}}{n_P^2} \left( \|A + E\|^2 - \|A\|^2 \right), \\ &= \frac{n_{C_1} n_{C_2}}{n_P^2} \left( 2\langle A, E \rangle + \|E\|^2 \right). \end{aligned}$$

Under Specification [2](#page-13-8) there exists a fixed proportion ω > 0 such that nC<sup>1</sup> , nC<sup>2</sup> ≥ ωn<sup>P</sup> , and hence nC<sup>1</sup> nC<sup>2</sup> /n<sup>2</sup> <sup>P</sup> ≥ ω(1 − ω) and also nC<sup>1</sup> nC<sup>2</sup> /n<sup>2</sup> <sup>P</sup> ≤ 1/4 for all nC<sup>1</sup> + nC<sup>2</sup> = n<sup>P</sup> . Therefore nC<sup>1</sup> nC<sup>2</sup> /n<sup>2</sup> <sup>P</sup> = O(1). Meanwhile, ∥E∥ <sup>2</sup> = o<sup>P</sup> (r 2 , 1/n<sup>C</sup><sup>1</sup> , 1/n<sup>C</sup><sup>2</sup> ) is true by definition of E, and under our assumptions one may follow the arguments of [Athey et al.](#page-9-2) [\(2019\)](#page-9-2) Proposition 1 to see that A = θ ∗ <sup>C</sup><sup>1</sup> − θ ∗ <sup>C</sup><sup>2</sup> = O(r). Thus,

$$\langle A, E \rangle = \mathcal{O}(r) \cdot o_P \left( r, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right) = o_P \left( r^2, \frac{r}{\sqrt{n_{C_1}}}, \frac{r}{\sqrt{n_{C_2}}} \right),$$

and therefore

$$\Delta(C_1, C_2) - \frac{n_{C_1} n_{C_2}}{n_P^2} \|\theta_{C_1}^* - \theta_{C_2}^*\|^2 = o_P\left(r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}}\right),$$

as desired. ■

Lemma B.4. *Suppose the conditions of Lemma [B.3](#page-17-1) hold, and assume moreover Neyman orthogonal moment conditions such that the underlying Jacobian* V (x) *defined in Assumption [2](#page-12-2) with block form* [\(37\)](#page-12-1)*. Then,*

$$\tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \left\| V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*) \right\|^2 + o_P \left( r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}} \right),$$

*where* ∆FPT <sup>η</sup> *defined in Lemma [B.2](#page-16-0) denotes the* FPT *criterion with arbitrary scale factor* η ̸= 0*.*

Proof of Lemma [B.4.](#page-18-0) From the proof of Lemma [B.2](#page-16-0) one finds that ∆FPT η (C1, C2) is equivalently written

$$\Delta_\eta^{\text{FPT}}(C_1, C_2) := \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \|\xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2})\|^2, \quad \bar{\psi}_{C_j} := \frac{1}{n_{C_1}} \sum_{\{i: X_i \in C_j\}} \psi_{\hat{\theta}_P, \hat{\nu}_P}(O_i).$$

Under standard LLN arguments the average scores ψ<sup>C</sup><sup>j</sup> satisfy

$$\overline{\psi}_{C_j} = \mathbb{E}[\psi_{\theta_{P, \dot{\nu}_P}}(O) \mid X \in C_j] + O_P(1/\sqrt{n_{C_j}}). \quad (44)$$

One applies iterated expectation to see

$$\mathbb{E}[\psi_{\hat{\theta}_P, \hat{\nu}_P}(O) \mid X \in C_j] = \mathbb{E} \left[ \mathbb{E} \left[ \psi_{\hat{\theta}_P, \hat{\nu}_P}(O) \mid X \right] \mid X \in C_j \right] = \mathbb{E}[M_{\hat{\theta}_P, \hat{\nu}_P}(X) \mid X \in C_j],$$

$$\bar{\psi}_{C_j} = \mathbb{E}[M_{\theta_{P, \dot{\nu}_P}}(X) \mid X \in C_j] + O_P(1/\sqrt{n_{C_j}}). \quad (45)$$

Expansion of Mθˆ<sup>P</sup> ,νˆ<sup>P</sup> (X). Under Assumption [2](#page-12-2) one considers the Taylor expansion of <sup>M</sup>θˆ<sup>P</sup> ,νˆ<sup>P</sup> (X) about (θ, ν) = (θ ∗ (x<sup>P</sup> ), ν<sup>∗</sup> (x<sup>P</sup> )):

$$\begin{aligned} M_{\hat{\theta}_P, \hat{\nu}_P}(X) &= M_{\theta^*(x_P), \nu^*(x_P)}(X) \\ &+ [\nabla_{(\theta, \nu)} M_{\theta^*(x_P), \nu^*(x_P)}(X)] \left[ \hat{\theta}_P - \theta^*(x_P) \right] + O_P \left( \left\| \left[ \hat{\theta}_P - \theta^*(x_P) \right] \right\|^2 \right). \end{aligned}$$

The consistency of the parent solutions ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) for (θ ∗ (x<sup>P</sup> ), ν<sup>∗</sup> (x<sup>P</sup> )) is established by [Athey et al.](#page-9-2) [\(2019\)](#page-9-2), and in particular ( ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) − (θ ∗ (x<sup>P</sup> ), ν<sup>∗</sup> (x<sup>P</sup> )) = O<sup>P</sup> (r, 1/ √ n<sup>P</sup> ). The asymptotic regime r <sup>−</sup><sup>2</sup> ≪ n<sup>P</sup> implies 1/ √ n<sup>P</sup> = o(r) and therefore the higher order quadratic term is equivalently expressed:

$$M_{\hat{\theta}_P, \hat{\nu}_P}(X) = M_{\theta^*(x_P), \nu^*(x_P)}(X) + [\nabla_{(\theta, \nu)} M_{\theta^*(x_P), \nu^*(x_P)}(X)] \begin{bmatrix} \hat{\theta}_P - \theta^*(x_P) \\ \hat{\nu}_P - \nu^*(x_P) \end{bmatrix} + O_P(r^2),$$

and therefore

$$\begin{aligned} \mathbb{E} \left[ M_{\hat{\theta}_P, \hat{\nu}_P}(X) \mid X \in C_j \right] &= \mathbb{E} \left[ M_{\theta^*(x_P), \nu^*(x_P)}(X) \mid X \in C_j \right] \\ &\quad + \mathbb{E} \left[ \nabla_{(\theta, \nu)} M_{\theta^*(x_P), \nu^*(x_P)}(X) \mid X \in C_j \right] \left[ \hat{\theta}_P - \theta^*(x_P) \right] + O_P(r^2). \end{aligned}$$

One has ∇(θ,ν)M<sup>θ</sup> <sup>∗</sup>(x<sup>P</sup> ),ν<sup>∗</sup>(x<sup>P</sup> )(X) = V (x<sup>P</sup> )+O<sup>P</sup> (r) because Mθ,ν(x) is Lipschitz in x, and the expansion in the previous display becomes:

$$\mathbb{E} \left[ M_{\hat{\theta}_{P, \hat{\nu}_P}}(X) \mid X \in C_j \right] = \mathbb{E} \left[ M_{\theta^*(x_P), \nu^*(x_P)}(X) \mid X \in C_j \right] + V(x_P) \left[ \hat{\theta}_P - \theta^*(x_P) \right]_{\hat{\nu}_P - \nu^*(x_P)} + O_P(r^2). \quad (46)$$

Expansion of M<sup>θ</sup> ∗ (x<sup>P</sup> ),ν<sup>∗</sup> (x<sup>P</sup> )(X). Following similar arguments, the term M<sup>θ</sup> <sup>∗</sup>(x<sup>P</sup> ),ν<sup>∗</sup>(x<sup>P</sup> )(X) is expanded about (θ, ν) = (θ ∗ (X), ν<sup>∗</sup> (X)) as:

$$\begin{aligned} M_{\theta^*(x_P), \nu^*(x_P)}(X) &= M_{\theta^*(X), \nu^*(X)}(X) + V(X) \begin{bmatrix} \theta^*(x_P) - \theta^*(X) \\ \nu^*(x_P) - \nu^*(X) \end{bmatrix} + O_P(r^2), \\ &= V(X) \begin{bmatrix} \theta^*(x_P) - \theta^*(X) \\ \nu^*(x_P) - \nu^*(X) \end{bmatrix} + O_P(r^2), \end{aligned}$$

where M<sup>θ</sup> <sup>∗</sup>(X),ν<sup>∗</sup>(X)(X) = 0 holds because (θ ∗ (X), ν<sup>∗</sup> (X)) are defined as satisfying the GRF moment conditions [\(1\)](#page-0-0) local to X. One takes the conditional expectation of the previous display:

$$\mathbb{E} [M_{\theta^*(x_P), \nu^*(x_P)}(X) \mid X \in C_j] = \mathbb{E} \left[ V(X) \left[ \begin{array}{c} \theta^*(x_P) - \theta^*(X) \\ \nu^*(x_P) - \nu^*(X) \end{array} \right] \mid X \in C_j \right] + O_P(r^2).$$

Whenever X ∈ C<sup>j</sup> one has ∥X − x<sup>P</sup> ∥ = O(r), and the same Lipschitz arguments can be applied to see V (X) = V (x<sup>P</sup> ) + O<sup>P</sup> (r) conditional on X ∈ C<sup>j</sup> , and the previous display simplifies:

$$\mathbb{E} [M_{\theta^*(x_P), \nu^*(x_P)}(X) \mid X \in C_j] = V(x_P) \begin{bmatrix} \theta^*(x_P) - \theta_{C_j}^* \\ \nu^*(x_P) - \nu_{C_j}^* \end{bmatrix} + O_P(r^2), \quad (47)$$

where θ ∗ C<sup>j</sup> := <sup>E</sup>[θ ∗ (X) | X ∈ C<sup>j</sup> ] and ν ∗ C<sup>j</sup> := <sup>E</sup>[ν ∗ (X) | X ∈ C<sup>j</sup> ]. Substitute [\(47\)](#page-19-0) into the conditional expectation [\(46\)](#page-19-1):

$$\begin{aligned}\mathbb{E} \left[ M_{\hat{\theta}_P, \hat{\nu}_P}(X) \mid X \in C_j \right] &= V(x_P) \left[ \begin{array}{c} \theta^*(x_P) - \theta_{C_j}^* \\ \nu^*(x_P) - \nu_{C_j}^* \end{array} \right] + V(x_P) \left[ \begin{array}{c} \hat{\theta}_P - \theta^*(x_P) \\ \hat{\nu}_P - \nu^*(x_P) \end{array} \right] + O_P(r^2), \\ &= V(x_P) \left[ \begin{array}{c} \hat{\theta}_P - \theta_{C_j}^* \\ \hat{\nu}_P - \nu_{C_j}^* \end{array} \right] + O_P(r^2).\end{aligned}$$

Therefore, the child node score averages ψ<sup>C</sup><sup>j</sup> in [\(45\)](#page-18-1) satisfy

$$\overline{\psi}_{C_j} = V(x_P) \begin{bmatrix} \hat{\theta}_P - \theta_{C_j}^* \\ \hat{\nu}_P - \nu_{C_j}^* \end{bmatrix} + O_P(r^2, 1/\sqrt{n_{C_j}}),$$

and the difference ψ<sup>C</sup><sup>1</sup> − ψ<sup>C</sup><sup>2</sup> satisfies

$$\begin{aligned} \bar{\psi}_{C_1} - \bar{\psi}_{C_2} &= V(x_P) \left[ \frac{\theta_P - \theta_{C_1}^*}{\hat{\nu}_P - \nu_{C_1}^*} \right] - V(x_P) \left[ \frac{\theta_P - \theta_{C_2}^*}{\hat{\nu}_P - \nu_{C_2}^*} \right] + O_P \left( r^2, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right), \\ &= -V(x_P) \left[ \frac{\theta_{C_1}^* - \theta_{C_2}^*}{\nu_{C_1}^* - \nu_{C_2}^*} \right] + O_P \left( r^2, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right). \end{aligned}$$

We assume η is a fixed scalar and ξ <sup>⊤</sup> a fixed matrix, it follows:

$$\eta \xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2}) = -\eta \xi^\top V(x_P) \begin{bmatrix} \theta_{C_1}^* - \theta_{C_2}^* \\ \nu_{C_1}^* - \nu_{C_2}^* \end{bmatrix} + O_P \left( r^2, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right). \quad (48)$$

The fixed matrix ξ <sup>⊤</sup> selects the coordinates of the target effect as ξ <sup>⊤</sup>(θ, ν) <sup>⊤</sup> = θ, and hence the product ξ <sup>⊤</sup>V (x<sup>P</sup> ) simplifies:

$$\xi^\top V(x_P) = \xi^\top \begin{bmatrix} V_{\theta\theta}(x_P) & V_{\theta\nu}(x_P) \\ V_{\nu\theta}(x_P) & V_{\nu\nu}(x_P) \end{bmatrix} = [V_{\theta\theta}(x_P) \quad V_{\theta\nu}(x_P)].$$

Under Neyman orthogonality one has Vθν(x<sup>P</sup> ) = 0, implying that ξ <sup>⊤</sup>V (x<sup>P</sup> ) = [Vθθ(x<sup>P</sup> ) 0], and [\(48\)](#page-20-0) becomes

$$\eta \xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2}) = -\eta V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*) + O_P \left( r^2, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right). \quad (49)$$

Asymptotic analysis. Let E be any term satisfying E = O<sup>P</sup> (r 2 , 1/ √<sup>n</sup>C<sup>1</sup> , 1/ √<sup>n</sup>C<sup>2</sup> ). In our asymptotic regime with r <sup>−</sup><sup>2</sup> ≪ nC<sup>j</sup> =⇒ 1/ √<sup>n</sup>C<sup>j</sup> <sup>=</sup> <sup>o</sup>(r), one has

$$E = O_P \left( r^2, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right) \implies E = o_P \left( r, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}} \right)$$

and therefore [\(49\)](#page-20-1) satisfies

$$\eta \xi^\top (\bar{\psi}_{C_1} - \bar{\psi}_{C_2}) = -\eta V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*) + o_P\left(r, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}}\right). \quad (50)$$

Write A = Vθθ(θ ∗ <sup>C</sup><sup>1</sup> − θ ∗ C<sup>2</sup> ) such that <sup>∆</sup>e FPT η (C1, C2) is equivalently written <sup>∆</sup>e FPT η (C1, C2) = (nC<sup>1</sup> nC<sup>2</sup> /n<sup>2</sup> P ) · η <sup>2</sup> ∥A + E∥ . Consider the difference

$$\begin{aligned} \tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) - \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \|V_{\theta\theta}(\theta_{C_1}^* - \theta_{C_2}^*)\|^2 &= \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \left( \|A + E\|^2 - \|A\|^2 \right), \\ &= \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \left( 2\langle A, E \rangle + \|E\|^2 \right). \end{aligned}$$

One repeats the same arguments used in the final asymptotic analysis of Lemma [B.3](#page-17-1) to show

$$2\langle A, E \rangle + \|E\|^2 = o_P \left( r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}} \right),$$

and thus

$$\tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) - \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \|V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*)\|^2 = o_P \left( r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}} \right),$$

Proof of Proposition [4.1.](#page-5-1) First, under Assumptions [A.1](#page-12-0) the θ-block Vθθ(x<sup>P</sup> ) of the local Jacobian V (x<sup>P</sup> ) is strictly positive definite and thus ∥·∥<sup>V</sup> defines a true norm. From the proof of Lemma [B.3:](#page-17-1)

$$\hat{\theta}_{C_1} - \hat{\theta}_{C_2} = \theta_{C_1}^* - \theta_{C_2}^* + o_P\left(r, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}}\right).$$

The matrix V (x<sup>P</sup> ) is non-random and fixed given P and η is a fixed scalar. It follows:

$$\eta V_{\theta\theta}(x_P)(\hat{\theta}_{C_1} - \hat{\theta}_{C_2}) = \eta V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*) + o_P\left(r, \frac{1}{\sqrt{n_{C_1}}}, \frac{1}{\sqrt{n_{C_2}}}\right).$$

Up to a negative factor, the expression on the right is precisely the same as [\(50\)](#page-20-2) in the proof of Lemma [B.4,](#page-18-0) and thus one repeats the arguments to arrive at

$$\left\| \eta V_{\theta\theta}(x_P)(\hat{\theta}_{C_1} - \hat{\theta}_{C_2}) \right\|_2^2 = \left\| \eta V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*) \right\|_2^2 + o_P \left( r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}} \right),$$

and hence

$$\Delta_{\eta V}(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \eta^2 \|V_{\theta\theta}(x_P)(\theta_{C_1}^* - \theta_{C_2}^*)\|_2^2 + o_P \left( r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}} \right).$$

The right hand side is precisely the same as in the statement of Lemma [B.4](#page-18-0) established for <sup>∆</sup>e FPT η (C1, C2), and thus

$$\tilde{\Delta}_\eta^{\text{FPT}}(C_1, C_2) - \Delta_\eta V(C_1, C_2) = o_P\left(r^2, \frac{1}{n_{C_1}}, \frac{1}{n_{C_2}}\right), \quad (51)$$

as desired. ■

Proof of Lemma [4.2.](#page-6-1) Firstly, Specifications [4](#page-13-5) (subsampling) and [5](#page-13-3) (honesty) describe conditions imposed on the sampling mechanism and are not affected by the form of the splitting criterion. It remains to verify whether Specification [1](#page-13-2) (symmetry), Specification [2](#page-13-8) (balanced/ω-regular), and Specification [3](#page-13-4) (randomized/random-split) are satisfied by T (∆<sup>V</sup> ).

- 1. Specification [1:](#page-13-2) Symmetry. A tree is said to be symmetric if its estimates are invariant under permutations of the tree's training samples. Conditional on a sequence of criterion values computed over splits of P, the CART mechanism of selecting the best split by scanning over the collection of candidates does not depend on the parent samples at all. This means that asymmetry could only enter through the criterion values. Therefore, a sufficient condition for symmetry in the tree estimates with respect to permutations of the tree samples is whether the criterion ∆<sup>V</sup> (C1, C2) is symmetric. Conditional on the child solutions <sup>ˆ</sup>θ<sup>C</sup><sup>1</sup> , <sup>ˆ</sup>θ<sup>C</sup><sup>2</sup> , the map ( <sup>ˆ</sup>θ<sup>C</sup><sup>1</sup> , <sup>ˆ</sup>θ<sup>C</sup><sup>2</sup> ) 7→ ∆<sup>V</sup> (C1, C2) does not depend on the parent samples at all, and therefore asymmetry could only enter through child solutions <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> . However, both criteria use precisely the same child solutions <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> in [\(7\)](#page-2-0), and therefore ∆<sup>V</sup> (C1, C2) will be symmetric whenever ∆(C1, C2) is symmetric (specifically, whenever ψθ,ν(Oi) is symmetric with respect to permutations of i).
- 2. Specification [2:](#page-13-8) Balanced/ω-regular. This condition is enforced by by GRF by adding an additional stopping condition to the gradient-based version of Algorithm [1.](#page-27-0) Specifically, GRF stops a recursive splitting path if a proposed ∆-optimal split were to send fewer than ωn<sup>P</sup> of the parent samples into either child. Simply stated, GRF enforces balanced splits by defining the set of valid candidate splits as those with at least ωn<sup>P</sup> parent samples in each child. This is left unchanged by our method.
- 3. Specification [3:](#page-13-4) Randomized/random-split. The asymptotic theory of GRF requires that, at each node, each variable is selected for a split with some lower bound probability π > 0. In order to satisfy the minimum split probability GRF uses the feature sampling device of [Denil et al.](#page-9-15) [\(2014\)](#page-9-15) which, at each step, considers only min{max{Poisson(m), 1}, p} randomly selected features as candidate variables, where p = dim(X ) and m is a GRF tuning parameter. In other words, GRF defines the set of valid candidate splits such that the set of valid splitting dimensions is itself a random variable. This mechanism is left unchanged under our method.

No column of V (x<sup>P</sup> ) is all-zero V·,k(x<sup>P</sup> ) ̸= 0 because V (x<sup>P</sup> ) is strictly positive definite symmetric, and therefore ∆<sup>V</sup> (C1, C2) will not systematically ignore signals along parameter dimensions θ<sup>k</sup> that can be detected by the ∆ criterion. Finally,

$$\hat{\theta}_{C_1} - \hat{\theta}_{C_1} \neq \mathbf{0} \implies V_{\theta\theta}(x_P)(\hat{\theta}_{C_1} - \hat{\theta}_{C_2}) \neq \mathbf{0},$$

because V (x) is strictly positive definite symmetric by Assumption [2.](#page-12-2) Therefore ∆(C1, C2) > 0 =⇒ ∆<sup>V</sup> (C1, C2) > 0 meaning that the ∆<sup>V</sup> -criterion is non-degenerate and will always be able to select at least one feature whenever the ∆-criterion can select a feature.

Therefore, all five specifications are met, and one concludes that T (∆<sup>V</sup> ) must satisfy the forest Specifications [A.2](#page-13-0) whenever they are satisfied by T (∆). ■

#### B.3. Asymptotic equivalence of the pseudo-outcome approximation for VCM/HTE models

In this section we establish the asymptotic equivalence of the further acceleration of the fixed-point method proposed in Section [5](#page-6-0) for VCM/HTE models. The accelerated algorithm is based on FPT pseudo-outcomes that use an approximation ˜θ<sup>P</sup> for the actual parent solutions ˆθ<sup>P</sup> in [\(31\)](#page-7-4). Specifically, the parent-leaf approximations ˜θ<sup>P</sup> are found by a single gradient descent step towards ˆθ<sup>P</sup> taken from the origin [\(54\)](#page-22-1). Let ρ i denote the original FPT pseudo-outcomes for VCM/HTE models:

$$\rho_i^{\text{FPT}} := -(W_i - \overline{W}_P) \left( Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \hat{\theta}_P \right), \quad (52)$$

where the solution ˆθ<sup>P</sup> for the local model over the parent P are precisely the OLS coefficients from the regression the centered outcomes Y<sup>i</sup> − Y <sup>P</sup> ∈ <sup>R</sup> on the centered regressors W<sup>i</sup> − W<sup>P</sup> ∈ <sup>R</sup> <sup>K</sup>. In contrast, let ϕ FPT i denote approximations of ρ FPT i pseudo-outcomes that are of the form

$$\phi_i^{\text{FPT}} := -(W_i - \overline{W}_P) \left( Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \tilde{\theta}_P \right), \quad (53)$$

where ˜θ<sup>P</sup> approximates ˆθ<sup>P</sup> as:

$$\tilde{\theta}_P := \gamma \cdot \frac{1}{n_P} \sum_{\{i: X_i \in P\}} (W_i - \overline{W}_P)(Y_i - \overline{Y}_P) = \gamma \cdot \frac{1}{n_P} W_P^\top Y_P. \quad (54)$$

Here, W<sup>P</sup> ∈ <sup>R</sup> <sup>n</sup><sup>P</sup> <sup>×</sup><sup>K</sup> and Y<sup>P</sup> ∈ <sup>R</sup> <sup>n</sup><sup>P</sup> denote the centered data matrices, W<sup>P</sup> := [W<sup>i</sup> − W<sup>P</sup> ]i:Xi∈<sup>P</sup> and Y<sup>P</sup> := [Y<sup>i</sup> − Y <sup>P</sup> ]i:Xi∈<sup>P</sup> , and the scalar γ > 0 denotes the exact line search step size corresponding to the regression of the centered outcomes on the centered regressors:

$$\gamma := \frac{\|W_P^\top Y_P\|_2^2}{\|W_P W_P^\top Y_P\|_2^2}. \quad (55)$$

Lemma B.5. *Let* ˜θ<sup>C</sup><sup>j</sup> *denote the* FPT *estimator of the form* [\(24\)](#page-4-6) *for the child solution* <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> *for VCM/HTE models. One can express* ˜θ<sup>C</sup><sup>j</sup> *in terms of the corresponding fixed-point pseudo-outcomes:*

$$\tilde{\theta}_{C_j} := \hat{\theta}_P + \frac{1}{n_{C_j}} \sum_{i: X_i \in C_j} \rho_i^{\text{FPT}}.$$

*Similarly, denote by* ¯θ<sup>C</sup><sup>j</sup> *the* FPT *estimator of* <sup>ˆ</sup>θ<sup>C</sup><sup>j</sup> *induced by pseudo-outcomes approximations* ϕ i *:*

$$\bar{\theta}_{C_j} := \hat{\theta}_P + \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} \phi_i^{\text{FPT}}.$$

*Then, under the assumptions of Proposition [4.1,](#page-5-1)* ¯θ<sup>C</sup><sup>j</sup> *is consistent for* ˜θ<sup>C</sup><sup>j</sup> *as:*

$$\left\| \tilde{\theta}_{C_j} - \bar{\theta}_{C_j} \right\| = o_P(1).$$

*Proof.* A direct calculation reveals that the difference between the original FPT pseudo-outcomes ρ i in [\(52\)](#page-22-2) and the approximations ϕ i in [\(53\)](#page-22-3) satisfy

$$\begin{aligned} \rho_i^{\text{FPT}} - \phi_i^{\text{FPT}} &= -(W_i - \overline{W}_P) \left( \left[ Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \hat{\theta}_P \right] - \left[ Y_i - \overline{Y}_P - (W_i - \overline{W}_P)^\top \tilde{\theta}_P \right] \right), \\ &= (W_i - \overline{W}_P)(W_i - \overline{W}_P)^\top (\hat{\theta}_P - \tilde{\theta}_P). \end{aligned} \quad (56)$$

Therefore, the difference between the original FPT child estimator ˜θ<sup>C</sup><sup>j</sup> and the approximation ¯θ<sup>C</sup><sup>j</sup> satisfies

$$\begin{aligned} \tilde{\theta}_{C_j} - \bar{\theta}_{C_j} &= \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} (\rho_i^{\text{FPT}} - \phi_i^{\text{FPT}}), \\ &= \frac{1}{n_{C_j}} \sum_{\{i: X_i \in C_j\}} (W_i - \overline{W}_P)(W_i - \overline{W}_P)^\top (\hat{\theta}_P - \tilde{\theta}_P), \\ &= S_{C_j}(\hat{\theta}_P - \tilde{\theta}_P), \end{aligned}$$

where we denote S<sup>C</sup><sup>j</sup> := nCj P {i:Xi∈C<sup>j</sup> } (W<sup>i</sup> − W<sup>P</sup> )(W<sup>i</sup> − W<sup>P</sup> ) <sup>⊤</sup>. Therefore,

$$\left\| \tilde{\theta}_{C_j} - \tilde{\theta}_{C_j} \right\| = \left\| S_{C_j}(\hat{\theta}_P - \tilde{\theta}_P) \right\| \leq \left\| S_{C_j} \right\|_F \left\| \hat{\theta}_P - \tilde{\theta}_P \right\|. \quad (57)$$

Under GRF's regularity conditions, in a limit where nC<sup>j</sup> → ∞ and the parent radius r := sup{i:Xi∈<sup>P</sup> } X<sup>i</sup> − X<sup>P</sup> goes to zero r → 0, we have S<sup>C</sup><sup>j</sup> <sup>p</sup>→ <sup>Q</sup> for some positive semidefinite symmetric matrix <sup>Q</sup>, and hence S<sup>C</sup><sup>j</sup> F = O<sup>P</sup> (1). Meanwhile, by definition [\(31\)](#page-7-4) for the OLS coefficients ˆθ<sup>P</sup> and definition [\(54\)](#page-22-1) for the one-step approximations ¯θ<sup>P</sup> , we have

$$\begin{aligned} \left\| \hat{\theta}_P - \hat{\theta}_P \right\| &= \left\| (-A_P^{-1} \cdot n_P^{-1} W_P^\top Y_P) - (\gamma \cdot n_P^{-1} W_P^\top Y_P) \right\|, \\ &= \left\| (-A_P^{-1} - \gamma \mathbb{I}) \cdot n_P^{-1} W_P^\top Y_P \right\|, \\ &\leq \left\| -A_P^{-1} - \gamma \mathbb{I} \right\|_F \left\| n_P^{-1} W_P^\top Y_P \right\|, \\ &= \left\| [n_P^{-1} W_P^\top W_P]^{-1} - \gamma \mathbb{I} \right\|_F \left\| n_P^{-1} W_P^\top Y_P \right\|, \end{aligned} \quad (58)$$

where −A<sup>P</sup> = n −1 <sup>P</sup> <sup>W</sup><sup>⊤</sup> <sup>P</sup> W<sup>P</sup> follows from the definition of A<sup>P</sup> as an estimator of the Jacobian ∇ψ, e.g. [\(13\)](#page-3-7) in the context of VCM/HTE models. Under the Lipschitz continuity Assumptions [1](#page-12-4) & [3,](#page-12-5) one has the standard stochastic bound for the cross term n −1 <sup>P</sup> <sup>W</sup><sup>⊤</sup> <sup>P</sup> Y<sup>P</sup> :

$$\|n_P^{-1}W_P^\top Y_P\| = O_P\left(r, \frac{1}{\sqrt{n_P}}\right), \quad (59)$$

while the difference [n −1 <sup>P</sup> <sup>W</sup><sup>⊤</sup> <sup>P</sup> W<sup>P</sup> ] <sup>−</sup><sup>1</sup> − γ<sup>I</sup> is stochastically bound as

$$\left\| \left[ n_P^{-1} W_P^\top W_P \right]^{-1} - \gamma \mathbb{I} \right\|_F = O_P(1),$$

because n −1 <sup>P</sup> <sup>W</sup><sup>⊤</sup> <sup>P</sup> W<sup>P</sup> <sup>p</sup><sup>→</sup> Cov(W<sup>i</sup> | X<sup>i</sup> ∈ P) is non-singular under Assumption [2.](#page-12-2) Coupling these stochastic bounds together according to [\(58\)](#page-23-0) gives

$$\left\| \hat{\theta}_P - \tilde{\theta}_P \right\| = O_P\left(r, \frac{1}{\sqrt{n_P}}\right),$$

and trivially, because nC<sup>j</sup> < n<sup>P</sup> ,

$$\left\| \hat{\theta}_P - \tilde{\theta}_P \right\| = O_P \left( r, \frac{1}{\sqrt{n_{C_j}}} \right). \quad (60)$$

Under Proposition 1 of GRF one assumes r <sup>−</sup><sup>2</sup> ≪ nC<sup>1</sup> , nC<sup>2</sup> and thus, in an asymptotic regime where nC<sup>j</sup> → ∞ and r → 0, one has 1/ √<sup>n</sup>C<sup>j</sup> <sup>=</sup> <sup>o</sup>(r), and hence:

$$\left\| \hat{\theta}_P - \tilde{\theta}_P \right\| = o_P(1). \quad (61)$$

Returning to [\(57\)](#page-23-1), the consistency of the parent approximation ˜θ<sup>P</sup> as [\(61\)](#page-23-2) implies that the approximation ¯θ<sup>C</sup><sup>j</sup> is itself consistent for the original FPT child estimator ˜θ<sup>C</sup><sup>j</sup> :

$$\left\| \tilde{\theta}_{C_j} - \bar{\theta}_{C_j} \right\| = o_P(1), \quad (62)$$

## C. Implementation Details

#### C.1. Honest subsampling

In this section we present the honest subsampling mechanism. Trees are used to form partitions of the input space such as to to specify weight functions αi(x), defined as

$$\alpha_i(x) := \frac{1}{B} \sum_{b=1}^B \alpha_{bi}(x), \quad \text{for} \quad \alpha_{bi}(x) := \frac{\mathbb{1}(X_i \in L_b(x))}{|L_b(x)|}, \quad i = 1, \dots, n, \quad (63)$$

where Lb(x) denotes a subset training samples that fall alongside x according to the partition of tree b. The honesty mechanism ensures that no observation in leaf Lb(x) was used to build the partition rules of tree b. This is achieved by separating an initial subsample into two subsets: One for building the partition rules, and the other allocated as samples to the local leaves Lb(x) according to the trained rules. Below, we give a detailed outline of how subsampling and honest sample splitting is used to train a forest of trees, then show that weight function αi(x) given by honest trees according to [\(63\)](#page-24-1) is conditionally independent of O<sup>i</sup> given X<sup>i</sup> .

## Honest subsampling for GRF

For tree b ∈ {1, . . . , B},

- 1. (*Subsampling*). Draw an initial subsample I
- (b) of size s := |I(b) | from the training set (without replacement).
- 2. (*Honest splitting*). Split I
- (b) into disjoint sets J
- (b) 1 and J
- (b) 2 of size |J (b) 1 | <sup>=</sup> ⌊s/2⌋ and |J (b) 2 | = ⌈s/2⌉.
  - (a) Train tree T(J
- (b) 1 ) based on the first subsample {(X<sup>i</sup> , Oi) : <sup>i</sup> ∈ J (b) 1 }. Let R
- (b) 1 , . . . , R
- (b) <sup>M</sup> denote the partition of X induced by T(J
  - (b) 1 ) such that R(b) <sup>m</sup> := n x ∈ X : x satisfies the partition rules for leaf m of T(J
    - (b) 1 ) o .
- (b) Subset the samples from the second subsample {X<sup>i</sup> : i ∈ J (b) 2 } according to the trained rules of T(J
- (b) 1 ), i.e. the samples of J
  - (b) 2 in the leaves are determined by the rules of T(J
- (b) 1 ).

$$\mathcal{R}_m^{(b)} := \left\{ x \in \mathcal{X} : x \text{ satisfies the partition rules for leaf } m \text{ of } T(\mathcal{J}_1^{(b)}) \right\}.$$

For any x ∈ X , the local leaf Lb(x) that appears in [\(63\)](#page-24-1) is defined as the specific subset of J (b) 2 samples belonging to the same leaf of tree T(J (b) ) as x,

$$L_b(x) = \{X_i \in \mathcal{R}_m^{(b)} : i \in \mathcal{J}_2^{(b)} \text{ and } x \in \mathcal{R}_m^{(b)}\},$$

Conditional independence of αi(x) and O<sup>i</sup> given X<sup>i</sup> . By definition, the partition rules of tree T(J (b) 1 ) depend only on the J (b) 1 subsample. The rules of a tree operate only on covariate values, and therefore the task of subsetting {X<sup>i</sup> : i ∈ J (b) 2 } into leaves according to the rules of T(J (b) 1 ) requires knowledge of the X<sup>i</sup> values from the J (b) 2 subsample but not necessarily the O<sup>i</sup> . Based on this understanding, we will show that αi(x) is conditionally independent of O<sup>i</sup> given X<sup>i</sup> . Based on [\(63\)](#page-24-1), it is sufficient to show

$$\mathbb{E}[\alpha_{bi}(x) \mid O_i, X_i] = \mathbb{E}[\alpha_{bi}(x) \mid X_i].$$

Case 1. Suppose i /∈ J (b) 2 . By definition <sup>L</sup>b(x) ⊂ {X<sup>j</sup> : <sup>j</sup> ∈ J (b) 2 }. It is immediate that <sup>1</sup>({X<sup>i</sup> ∈ Lb(x)}) = 0, and therefore αbi(x) = 0, and trivially

$$\mathbb{E}[\alpha_{bi}(x) \mid O_i, X_i] = \mathbb{E}[\alpha_{bi}(x) \mid X_i] = 0, \quad \text{for all } i \notin \mathcal{J}_2^{(b)}.$$

Case 2. Suppose i ∈ J (b) . We show that each component used to specify αbi(x) in [\(63\)](#page-24-1) is conditionally independent of O<sup>i</sup> given X<sup>i</sup> :

- The rules of tree T(J
- (b) 1 ) operate only on input values. Therefore, conditionally on <sup>X</sup><sup>i</sup> for all <sup>i</sup> ∈ J (b) 2 , the leaves of the J
  - (b) subsample specified by tree T(J
- (b) ) do not depend on the value of O<sup>i</sup> .
- Leaf Lb(x) is the specific subset of the J
- (b) 2 samples satisfying the same partition rules of T(J
- (b) 1 ) as x. Given the leaves have been specified by the previous step, this depends only on x.

Therefore, the individual component functions αbi(x) = <sup>1</sup>({X<sup>i</sup> ∈ Lb(x)})/|Lb(x)| are conditionally independent of O<sup>i</sup> given X<sup>i</sup> ,

$$\mathbb{E}[\alpha_{bi}(x) \mid O_i, X_i] = \mathbb{E}[\alpha_{bi}(x) \mid X_i], \quad \text{for all } i \in \mathcal{J}_2^{(b)}.$$

Demonstration of honest subsampling. Let {(X<sup>i</sup> , Oi)} n <sup>i</sup>=1 denote a training set of n = 20 observations, where each X<sup>i</sup> = (Xi,1, Xi,2) is over X ≡ <sup>R</sup> 2 . We will use a forest of a single tree (B = 1) to specify the functional form of weights αi(x).

- 1. (Subsampling). Draw an initial subsample I of size s = 10.
- 2. (Honest splitting). Split I into two disjoint sets J<sup>1</sup> and J<sup>2</sup> , each with s/2 = 5 samples.

| i ∈ J 1 2 i ∈ I X i, 1 X i, 2 O i 3 2 8 3 J 1 15 5 i X i, 1 X i, 2 O i 20 8 1 I 10 11 i ∈ J 2 20 14 | X i, 1 X i, 1 | X i, 2 O i X i, 2 O i |
|-----------------------------------------------------------------------------------------------------|---------------|-----------------------|
| J 2                                                                                                 |               |                       |
| 5                                                                                                   | 1             | 0                     |
| 10                                                                                                  | 2             | -2                    |
| 11                                                                                                  | 0             | 1                     |
| 14                                                                                                  | 1             | -2                    |
| 16                                                                                                  | 2             | 2                     |

- (a) Train a tree using the data from the first subsample J<sup>1</sup> , inducing a partition of X ≡ <sup>R</sup> 2 . Suppose the fitted tree has the following structure:

![](_page_25_Figure_8.jpeg)

- (b) Use the trained partition rules to subset the J<sup>2</sup> subsample into separate leaves.

![](_page_26_Figure_1.jpeg)

The tree trained on the J<sup>1</sup> subsample will subset the J<sup>2</sup> subsample as

$$\{X_i : i \in \mathcal{J}_2\} = \{X_{10}, X_{14}\} \cup \{X_5, X_{11}, X_{16}\} \cup \emptyset,$$

where we include the trivial union with ∅ to note that the tree assigns none of the J<sup>2</sup> samples to the partition of R <sup>2</sup> where Xi,<sup>1</sup> ≥ 3.

The leaf Lb(x) is the specific subset of the J<sup>2</sup> subsample such that X<sup>i</sup> ∈ J<sup>2</sup> satisfy the same partition rules as x. Given a test point x = x0, there are three possible scenarios for Lb(x0) that correspond to the three regions R1, R2, R<sup>3</sup> ⊂ <sup>R</sup> in which the test point x<sup>0</sup> can appear.

Region 1. If x<sup>0</sup> ∈ R<sup>1</sup> then Lb(x0) = {X10, X14} and

$$\alpha_{bi}(x_0) = \frac{\mathbf{1}(\{X_i \in L_b(x_0)\})}{|L_b(x_0)|} = \begin{cases} \frac{1}{2} & \text{if } i \in \{10, 14\}, \\ 0 & \text{otherwise.} \end{cases}$$

Therefore, αi(x0) = <sup>1</sup> 2 for i = 10, 14 and zero for i ∈ {1, . . . , 20} \ {10, 14}.

Region 2. If x<sup>0</sup> ∈ R<sup>2</sup> then Lb(x0) = {X5, X11, X16} and

$$\alpha_{bi}(x_0) = \frac{1(\{X_i \in L_b(x_0)\})}{|L_b(x_0)|} = \begin{cases} \frac{1}{3} & \text{if } i \in \{5, 11, 16\}, \\ 0 & \text{otherwise.} \end{cases}$$

Therefore, αi(x0) = <sup>1</sup> 3 for i = 5, 11, 16 and zero i ∈ {1, . . . , 20} \ {5, 11, 16}.

Region 3. If x<sup>0</sup> ∈ R<sup>3</sup> then Lb(x0) = ∅. This is a degenerate case such that

$$\alpha_{bi}(x_0) = \frac{\mathbb{1}(\{X_i \in L_b(x_0)\})}{|L_b(x_0)|}$$

is undefined, leading to a non-identifiability problem whenever x<sup>0</sup> ∈ R3. When this occurs, [Tibshirani et al.](#page-11-13) [\(2024\)](#page-11-13) recommends calculating αi(x0) based on only the trees with non-empty Lb(x0). Let B := {b ∈ {1, . . . , B} : |Lb(x0)| > 0} denote the indices of non-empty leaves associated with x0. Then, the GRF weight functions based on this recommendation can be written as

$$\alpha_i(x_0) = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \alpha_{bi}(x_0).$$

Algorithm 1 The fixed-point tree algorithm function TRAINFIXEDPOINTTREE

Input: node N

node P<sup>0</sup> ← GETSAMPLES(N ) queue Q ← INITIALIZEQUEUE(P0) while NOTNULL(node P ← POP(Q)) do

(

ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) ← SOLVEESTIMATINGEQUATION(P) ▷ Computes [\(6\)](#page-2-6).

ρ

FPT ← FPTPSEUDOOUTCOMES(

ˆθ<sup>P</sup> , νˆ<sup>P</sup> ) ▷ Applies [\(26\)](#page-5-3) over P.

split Σ ← CARTSPLIT(P, ρ

FPT) ▷ Optimizes [\(19\)](#page-3-6).

if SPLITSUCCEEDED(Σ) then

SETCHILDREN(P, GETLEFTCHILD(Σ), GETRIGHTCHILD(Σ))

ADDTOQUEUE(Q, GETLEFTCHILD(Σ)) ADDTOQUEUE(Q, GETRIGHTCHILD(Σ))

end if end while

Output: tree with root node P<sup>0</sup>

end function

POP returns and removes the oldest element of queue a Q, unless Q is empty, in which case it returns NULL. CARTSPLIT runs a multivariate CART split on the pseudo-outcomes ρ FPT := {ρ i }i∈<sup>P</sup> , and either returns a pair of child nodes or indicates that no split of P is possible.

#### C.2. In-sample predictions

There is additional bias associated with making predictions based on in-sample observations X<sup>i</sup> that may have been used either to train the tree structure or to populate the local leaves Lb(x). The recommendation of [Tibshirani et al.](#page-11-13) [\(2024\)](#page-11-13) is along the lines of the out-of-bag mechanism used by [Breiman](#page-9-9) [\(2001\)](#page-9-9). For an in-sample observation x ′ ∈ {Xi} n <sup>i</sup>=1, calculate weights α oob i (x ′ ) based only on those trees whose initial subsample I (b) does not contain x ′ . Then the out-of-bag weight is defined as:

$$\alpha_i^{\text{oob}}(x') := \frac{1}{|\{b : x' \notin \mathcal{I}^{(b)}\}|} \sum_{\{b:x' \notin \mathcal{I}^{(b)}\}} \alpha_{bi}^{\text{oob}}(x') \quad \text{for} \quad \alpha_{bi}^{\text{oob}}(x') := \frac{1}{|L_b(x')|} L_b(x').$$

The in-sample prediction ˆθ oob(x ′ ) for x ′ is made by GRF by solving a version of the locally weighted estimating equation [\(5\)](#page-1-2) using out of bag weights α oob i (x ′ )

$$\left(\hat{\theta}^{\text{oob}}(x'), \hat{\nu}^{\text{oob}}(x')\right) \in \arg\min_{\theta, \nu} \left\| \sum_{i=1}^n \alpha_i^{\text{oob}}(x') \psi_{\theta, \nu}(O_i) \right\|,$$

which preserve the consistency and asymptotic normality of the GRF estimator at in-sample observations.

## C.3. Algorithms and Pseudocode

## C.4. Simulation Details

Implementation details. We implement the GRF-FPT algorithm in a fork of grf [\(Tibshirani et al.,](#page-11-13) [2024\)](#page-11-13) available at <https://github.com/dfleis/grf>. The functions grf::lm forest and grf::multi arm causal forest provide an easy to use interface for VCM and HTE estimation, respectively, and we allow the choice GRF-FPT1, GRF-FPT2, or GRF-grad to be controlled via the method argument. Code and data for reproducing all experiments and figures are available at <https://github.com/dfleis/grf-experiments>.

Data-generating settings. The different setting for the target effects θ ∗ k (x) include a sparse linear setting, a sparse logistic setting with interaction, a dense logistic setting, and a random function generator setting. Tables [2](#page-29-1) and [3](#page-29-2) provide the details of each regime for VCM and HTE experiments, respectively, for the data-generating model [\(28\)](#page-6-4). These tables also summarize

Algorithm 2 Stage I GRF-FPT: Training a generalized random forest using fixed-point trees

function TRAINGENERALIZEDRANDOMFORESTFPT

Input: samples S, number of trees B

for b = 1, . . . , B do

set of samples I ← SUBSAMPLE(S)

sets of samples JBUILD,JPOPULATE ← HONESTSPLIT(I) ▷ See honesty: Appendix [C.1.](#page-24-0)

tree T<sup>b</sup> ← TRAINFIXEDPOINTTREE(JBUILD) ▷ See Algorithm [1.](#page-27-0)

leaves L<sup>b</sup> ← POPULATELEAVES(Tb,JPOPULATE) ▷ See honesty: Appendix [C.1.](#page-24-0)

end for

Output: forest F ← {L1, . . . ,LB}

end function

POPULATELEAVES creates a collection of subsets (leaves) of the JPOPULATE samples based on the partition rules of tree Tb. For weight functions αi(x), see GETWEIGHTS in Algorithm [3.](#page-28-1) For Stage II, see ESTIMATE in Algorithm [3,](#page-28-1) where estimates ˆθ(x) are made given a forest F.

Algorithm 3 GRF-FPT: Estimates of θ (x) function ESTIMATE Input: forest F, test observation x ∈ X weights α ← GETWEIGHTS(F, x) Output: ˆθ(x), the solution to the weighted estimating equation [\(5\)](#page-1-2) using weights α end function function GETWEIGHTS Input: forest F, test observation x vector of weights α ← ZEROS(n) ▷ Initialize weights; n = |S| used to train F. for indices i : X<sup>i</sup> ∈ Lb(x) do α[i] += 1/|Lb(x)| end for Output: local weights α ← α/|F| ▷ Weights [\(4\)](#page-1-3). end function

#### ∗

Stage II of the GRF-FPT algorithm. The procedure ESTIMATE returns an estimate of θ ∗ (x) given a forest F trained under Stage I and a test observation x; see Algorithm [2.](#page-28-0)

the different settings used to generate the K-dimensional regressors W<sup>i</sup> = (Wi,1, . . . , Wi,K) <sup>⊤</sup>. For VCM experiments, Wi,k ∼ N (0, 1) for all k = 1, . . . , K. For HTE experiments, W<sup>i</sup> | X<sup>i</sup> = x ∼ Multinomial(1,(π1(x), . . . , πK(x))), where πk(x) denotes the underlying probability the sample is observed as having treatment level k ∈ {1, . . . , K}.

Random function generator. The effect functions θ ∗ k (x) = RFG(x) under VCM Setting 4 (in Table [2\)](#page-29-1) and HTE Setting 5 (in Table [3\)](#page-29-2) follow the random function generator design of [Friedman](#page-10-21) [\(2001\)](#page-10-21). The idea is to measure the performance of the estimator under a variety of randomly generated targets. Each θ ∗ k (·) is randomly generated as a linear combination of functions {gℓ(·)} 20 ℓ of the form

$$\theta_k^*(x) = \sum_{\ell=1}^{20} a_\ell g_\ell(z_\ell),$$

where the coefficients {aℓ} 20 <sup>ℓ</sup>=1 are randomly generated from a uniform distribution a<sup>ℓ</sup> ∼ U([−1, 1]). Each gl(zl) is a function of a randomly selected pℓ-size subset of the p-dimensional variable x, where the size of each subset p<sup>ℓ</sup> is randomly chosen by p<sup>ℓ</sup> = min(⌊1.5 + rℓ⌋ , p), and r<sup>ℓ</sup> is generated from an exponential distribution with mean 2, r<sup>ℓ</sup> ∼ Exp(0.5). Each g(zℓ) uses a pℓ-sized random subset z<sup>ℓ</sup> ∈ <sup>R</sup> <sup>p</sup><sup>ℓ</sup> of the p-dimensional input x ∈ <sup>R</sup> p :

$$z_\ell := (x_{\phi_\ell(1)}, \dots, x_{\phi_\ell(p_\ell)}) \in \mathbb{R}^{p_\ell},$$

| Parameter | Values                  |           |            |
|-----------|-------------------------|-----------|------------|
| K         | 4; 16; 64; 256          |           |            |
| n         | 10,000; 20,000; 100,000 |           |            |
| dim( X )  | 5                       |           |            |
| nTrees    | 100                     |           |            |
|           |                         | Parameter | Values     |
|           |                         | K         | 4; 16;     |
|           |                         | n         | 1000; 4000 |
|           |                         | dim( X )  | 2          |
|           |                         | nTrees    | 100; 500   |

Table 1: Parameter values for VCM and HTE experiments in Section [6.](#page-7-5) Target/regressor dimension K, number of observations n, dimension of the auxiliary variables dim(X ), and number of trees nTrees. Experiments include a large-n setting (left table) and a small-n setting (right table).

|             | ∗                                                                                   |
|-------------|-------------------------------------------------------------------------------------|
| VCM Setting | Effect function θ                                                                   |
|             | ( x ) W i,k k ∗                                                                     |
| 1           | θ                                                                                   |
|             | ( x ) = β k 1 x 1 , β k 1 ∼ N (0 , 1) N (0 , 1) k ∗                                 |
| 2           | θ                                                                                   |
|             | ( x ) = ς ( β k 1 x 1 ) ς ( β k 2 x 2 ) , β k 1 , β k 2 ∼ N (0 , 1) N (0 , 1) k ∗ ⊤ |
| 3           | θ                                                                                   |
|             | ( x ) = ς ( β                                                                       |
|             | k x ) , for β k ∼ N p ( 0 , I ) N (0 , 1) k ∗                                       |
| 4           | θ                                                                                   |
|             | ( x ) = RFG ( x ) N (0 , 1) k                                                       |

Table 2: Settings for the true effects θ ∗ <sup>k</sup>(·) and the regressors Wi,k for VCM experiments in Section [6.](#page-7-5) The function ς(u) := 1 + (1 + e <sup>−</sup>20(u−1/3)) −1 is a logistic-type function in [\(Athey et al.,](#page-9-2) [2019\)](#page-9-2). The random function generator RFG(x) is described in Appendix [C.4.](#page-27-2)

|             | ∗                                                                    |
|-------------|----------------------------------------------------------------------|
| HTE Setting | Treatment effect θ                                                   |
|             | ( x ) Treatment probability π k ( x ) for W i,k k ∗                  |
| 1           | θ                                                                    |
|             | ( x ) = β k 1 x 1 , β k 1 ∼ N (0 , 1) π k ( x ) = 1 /K for all k k   |
| 2           | θ                                                                    |
|             | ( x ) = β k 1 x 1 , β k 1 ∼ N (0 , 1) π k ( x ) = (                  |
|             | x 1 k = 1 , ∗ k                                                      |
|             | K − 1                                                                |
|             | (1 − x 1 ) k = 2 , , K ∗                                             |
| 3           | θ                                                                    |
|             | ( x ) = ς ( β k 1 x 1 ) ς ( β k 2 x 2 ) π k ( x ) = 1 /K for all k k |
|             | for β k 1 , β k 2 ∼ N (0 , 1)                                        |
| 4           | θ                                                                    |
|             | ( x ) = ς ( β                                                        |
|             | k x ) , for β k ∼ N p ( 0 , I ) π k ( x ) = (                        |
|             | x 1 k = 1 , ∗ ⊤ k                                                    |
|             | K − 1                                                                |
|             | (1 − x 1 ) k = 2 , , K.                                              |
| 5           | θ                                                                    |
|             | ( x ) = RFG ( x ) π k ( x ) = exp { γ                                |
|             | k x } ⊤ ∗ k                                                          |
|             | P K                                                                  |
|             | j =1 exp { γ ⊤                                                       |
|             | x } j                                                                |
|             | for γ k ∼ N p ( 0 , I )                                              |

Table 3: Settings for the underlying treatment effects θ ∗ <sup>k</sup>(·) and treatment probabilities πk(x) for HTE experiments in Section [6.](#page-7-5) The function ς(u) := 1 + (1 +e <sup>−</sup>20(u−1/3)) −1 is a logistic-type function used in [\(Athey et al.,](#page-9-2) [2019\)](#page-9-2). The random function generator RFG(x) is described in Appendix [C.4.](#page-27-2)

such that {ϕℓ(1), . . . , ϕℓ(pℓ)} is a length-p<sup>ℓ</sup> permutation of indices drawn from {1, . . . , p}, without replacement. The functions gℓ(·) are Gaussian functions of the p<sup>ℓ</sup> sampled variables:

$$g_\ell(z_\ell) := \exp \left\{ -\frac{1}{2}(z_\ell - \mu_\ell)^\top \mathbf{V}_\ell(z_\ell - \mu_\ell) \right\},$$

where the mean vector µ<sup>ℓ</sup> ∈ <sup>R</sup> p<sup>ℓ</sup> is randomly generated from a standard multivariate Gaussian, µ<sup>ℓ</sup> ∼ N<sup>p</sup><sup>ℓ</sup> (0,I). The p<sup>l</sup> × p<sup>l</sup> covariance matrix V<sup>l</sup> are formed through the spectral decomposition:

$$\mathbf{V}_\ell = \mathbf{U}_\ell \mathbf{D}_\ell \mathbf{U}_\ell^\top,$$

where U<sup>ℓ</sup> is a random p<sup>ℓ</sup> × p<sup>ℓ</sup> orthonormal matrix and D<sup>ℓ</sup> := diag(d1,ℓ, . . . , d<sup>p</sup>ℓ,ℓ) with diagonal entries dj,ℓ generated from a uniform distribution according to p dj,ℓ ∼ U(0.1, 2.0).

## D. Additional Simulations

## D.1. Settings for the criterion value experiment in Section [3.4](#page-4-5)

The criterion value experiment in Section [3.4](#page-4-5) was run under a varying coefficient model of the form

$$Y_i := W_i^\top \theta^*(X_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 0.5^2), \quad (64)$$

where the regressors W<sup>i</sup> were generated as bivariate standard Gaussian samples W<sup>i</sup> ∼ N2(0,I) and the auxiliary covariates were generated as standard uniform samples X<sup>i</sup> ∼ U(0, 1). The data-generating coefficient functions were θ ∗ (x) := (sin(2πx), x) and the criterion values were computed based on n = 1000 samples following [\(64\)](#page-29-3).

### D.2. Supporting experiments for Section [6](#page-7-5)

Multicollinearity in auxiliary covariates. We conducted a VCM experiment with highly correlated auxiliary covariate features. We ran a modified version of VCM Setting 3 by generating auxiliary covariates as X<sup>i</sup> ∼ N (0, Σ), where [Σ]j,k = ω |j−k| for ω ∈ {0, 0.5, 0.9}. Table [4](#page-30-1) provides a clear summary of the computational performance of GRF-FPT relative to GRF-grad and statistical accuracy (MSE – multiplied by 100 for readability). All experiments were run over a forest of 10 trees and MSE estimates were computed over 50 replications of the model and evaluated on a separate set of n = 5, 000 samples, carried out using GRF-FPT2 and GRF-grad. These results demonstrate clearly that GRF-FPT remains robust, stable, and computationally efficient, even under high multicollinearity in X<sup>i</sup> .

| dim( | X ) n  | K  | ω    | Speedup | 100 × MSE grad | 100 × MSE FPT2 |
|------|--------|----|------|---------|----------------|----------------|
| 5    | 10,000 | 64 | 0.00 | 2.55    | 16.60          | 16.83          |
| 5    | 10,000 | 64 | 0.50 | 2.53    | 15.48          | 15.47          |
| 5    | 10,000 | 64 | 0.90 | 2.35    | 10.95          | 11.09          |

Table 4: Effect of multicollinearity in the auxiliary covariates X<sup>i</sup> on the relative computational gain of GRF-FPT2, as well as the statistical accuracy of both GRF-FPT and GRF-grad estimators.

Subsampling ratio. We carried out an experiment to show that the subsample proportion does not affect the computational advantage or statistical accuracy of GRF-FPT relative to GRF-grad. We varied the subsampling ratio s/n ∈ {0.25, 0.50, 0.75} under VCM Setting 3 over a forest of 10 trees carried out using GRF-FPT2 and GRF-grad. Table [5](#page-30-2) summarizes our results, averaged over 50 replications of the model, with a test set of 5,000 samples. These results show clearly that the statistical accuracy (MSE) of GRF-FPT2 relative to GRF-grad does not depend strongly on the subsample ratio.

| dim( | X ) n  | K  | s/n  | Speedup | 100 × MSE grad | 100 × MSE FPT2 |
|------|--------|----|------|---------|----------------|----------------|
| 2    | 10,000 | 64 | 0.25 | 2.77    | 2.86           | 2.90           |
| 2    | 10,000 | 64 | 0.50 | 3.10    | 2.91           | 2.90           |
| 2    | 10,000 | 64 | 0.75 | 2.98    | 3.21           | 3.19           |

Table 5: Effect of the subsampling ratio s/n on the relative computational gain of GRF-FPT2, as well as the statistical accuracy of both GRF-FPT and GRF-grad estimators.

Large sample size. We ran additional experiments to clearly show how our method scales for very large datasets. Using a forest of 10 trees, we tested our method on VCM Setting 3 with sample sizes up to n = 500, 000, carried out using GRF-FPT2 and GRF-grad. The results are summarized in Table [6](#page-30-3) and demonstrate that, even as the dataset grows very large, our method consistently remains faster than GRF-grad. While the relative speedup slightly decreases at first, it stabilizes towards a consistent advantage as grows n grows sufficiently large, suggesting that the advantage is not bottlenecked by n and maintains a robust advantage at scale.

| dim( | X ) K | n       | Speedup |
|------|-------|---------|---------|
| 2    | 256   | 10,000  | 4.54    |
| 2    | 256   | 20,000  | 3.59    |
| 2    | 256   | 50,000  | 3.49    |
| 2    | 256   | 100,000 | 3.11    |
| 2    | 256   | 200,000 | 3.04    |
| 2    | 256   | 500,000 | 3.08    |

Table 6: Effect of increasing sample sizes n on the relative computational gain of GRF-FPT2.

![](_page_31_Figure_1.jpeg)

Figure 5: Absolute fit times for VCM timing experiments under the settings in Table [2](#page-29-1) and large-n settings in Table [1.](#page-29-4)

### D.3. Supporting figures for Section [6](#page-7-5)

#### D.3.1. VCM EXPERIMENTS

Large n VCM simulations. Figure [5](#page-31-0) illustrates the absolute fit times for the GRF-FPT algorithms under the four VCM settings for θ ∗ k (x) described in Table [2](#page-29-1) over the large-n settings in Table [1.](#page-29-4) Across all settings and all dimensions, GRF-FPT is consistently several factors faster than GRF-grad. The speedup factor is summarized in Figure [3,](#page-8-0) which illustrates the relative speedup of GRF-FPT, calculated as the ratio of GRF-grad fit times over GRF-FPT fit times. Consistent with the observations in Section [5,](#page-6-0) we find that the speed advantage of GRF-FPT increases as the dimension of the target increases.

Figure [6](#page-32-0) shows that this speed advantage comes while performing comparably to GRF-grad in terms of statistical accuracy. Across all settings for VCMs with K = 4 dimensional targets, the MSE estimates from GRF-FPT is highly similar to the MSE estimates of GRF-grad, while for K = 256 dimensional targets one sees more variation in MSE estimates across the methods. This effect likely reflects the increased variance associated with high-dimensional estimation. In some cases we see GRF-FPT1 slightly outperform both GRF-FPT2 and GRF-grad, in other cases we see GRF-grad slightly outperform both GRF-FPT methods, and in others GRF-FPT2 yields the lowest MSE. One sees that these differences are typically small. The key benefit we emphasize is that GRF-FPT is able to achieve nearly identical statistical accuracy with a substantial improvement in computational speed.

Small n VCM simulations. Figures [8](#page-34-0) and [7](#page-33-0) illustrate the absolute fit times and relative speed advantage, respectively, of GRF-FPT under the VCM design of θ ∗ k (x) over the small-n settings. One sees that even when n is more modest, GRF-FPT consistently offers a computational advantage over GRF-grad, with possible outliers under VCM Setting 2 at K = 4. We believe this negative relative advantage to be caused by random fluctuations in computation and are not representative of the FPT algorithm itself, particularly in light of the fact that the negative effect vanishes when the number of trees increases from 100 to 500. As one would expect based on the large-n results, the relative advantage tends to increase with increasing K, and generally stabilizes with increasing n. Figure [9](#page-35-0) shows that the GRF-FPT speed advantage does not come at any material cost in statistical accuracy, with similar performance to GRF-grad across all settings.

![](_page_32_Figure_2.jpeg)

50 model replications, 5000 test observations

## MSE estimates: Varying coefficient model (VCM)

![](_page_32_Figure_4.jpeg)

50 model replications, 5000 test observations

## MSE estimates: Varying coefficient model (VCM)

Figure 6: Estimates of MSE <sup>E</sup>[∥θ ∗ (X) − ˆθ(X)/K∥ <sup>2</sup>] for VCM for K = 256 dimensional (top) and K = 4 dimensional targets (bottom) under the large-n settings in Table [1.](#page-29-4)

![](_page_33_Figure_1.jpeg)

Figure 7: Speedup factor for GRF-FPT in comparison to GRF-grad for VCM timing experiments under the small-n settings in Table [1.](#page-29-4)

### D.4. HTE experiments

Large n HTE simulations. Figure [11](#page-37-1) illustrates the absolute fit times for the GRF-FPT algorithm under the five HTE settings of θ ∗ k (x) and πk(x) described in Table [3](#page-29-2) over the large-n settings in Table [1.](#page-29-4) We find that GRF-FPT is consistently faster than GRF-grad. The speedup factor of GRF-FPT relative to GRF-grad is summarized in Figure [10,](#page-36-0) calculated as the ratio of GRF-grad fit times over GRF-FPT fit times. As was seen for VCM experiments, the speed advantage of GRF-FPT scales with the dimensionality K of the target. One sees from both Figures [10](#page-36-0) and [11](#page-37-1) that GRF-FPT's computational advantage is less dramatic than under the VCM experiments. This can be understood based on the fact that the VCM regressors W<sup>i</sup> are continuous while the HTE regressors represent binary indicators. Continuous regressors provide more granularity when fitting the child statistics ˜θ<sup>C</sup><sup>j</sup> , and as a result provide a larger set of candidate splits over the covariates. Nevertheless, one sees in Figure [10](#page-36-0) that the FPT splitting mechanism is still up to 1.5× faster under the largest regressor setting K = 256, with a more modest, but persistent savings across all settings.

The statistical benchmarks for our HTE experiments are shown in Figure [12.](#page-38-1) Consistent with the VCM experiments, one sees that the computational advantage of GRF-FPT does not come at the cost of in terms of its statistical accuracy.

Small n HTE simulations. Figures [13](#page-39-0) and [14](#page-40-0) summarize the relative speed advantage and absolute fit times for the GRF-FPT algorithm under the small-n HTE design. Consistent with the large-n HTE experiments the FPT2 mechanism sees a stable computational advantage across all settings, with an increasing effect in increasing K, while the FPT1 mechanism displays a persistent advantage for K = 16 and comparable computational performance for K = 4. The more modest relative advantage for the small-n experiments is itself consistent with the VCM small-n experiments, owing in large part due to the smaller values of K. Figure [15](#page-41-0) compares the statistical performance of GRF-FPT to GRF-grad, with no material difference between either GRF-FPT1, GRF-FPT2, or GRF-grad's estimation accuracy.

![](_page_34_Figure_1.jpeg)

Figure 8: Absolute fit times for VCM timing experiments under the settings in Table [2](#page-29-1) and small-n settings in Table [1.](#page-29-4)

![](_page_35_Figure_2.jpeg)

50 model replications, 5000 test observations

## MSE estimates: Varying coefficient model (VCM)

![](_page_35_Figure_4.jpeg)

50 model replications, 5000 test observations

## MSE estimates: Varying coefficient model (VCM)

Figure 9: Estimates of MSE <sup>E</sup>[∥θ ∗ (X) − ˆθ(X)/K∥ 2 <sup>2</sup>] for VCM for K = 16 dimensional (top) and K = 4 dimensional targets (bottom) under the small-n settings in Table [1.](#page-29-4)

![](_page_36_Figure_2.jpeg)

## Heterogeneous treatment effects (HTE) Fit time speedup factor: GRF−grad/GRF−FPT (forests)

Figure 10: Speedup factor for GRF-FPT in comparison to GRF-grad for HTE timing experiments under the large-n setting in Table [1.](#page-29-4)

## E. Additional Examples

### E.1. Pseudo-outcomes for nonparametric regression

Consider the task of estimating the conditional mean function θ ∗ (x) := <sup>E</sup>[Y |X = x]. The target θ ∗ (x) is identified by a moment condition of the form [\(1\)](#page-0-0) with scoring function ψθ(Yi) := Y<sup>i</sup> − θ, the residual associated with using θ as the local estimate with respect to the i-th sample. The local solution ˆθ<sup>P</sup> over P is the mean observed response over the parent,

$$\hat{\theta}_P = \overline{Y}_P := \frac{1}{n_P} \sum_{\{i: X_i \in P\}} Y_i.$$

The fixed-point pseudo-outcomes are simple the (negative) residuals that result from fitting [\(6\)](#page-2-6) over P

$$\rho_i^{\text{FPT}} = -(Y_i - \hat{\theta}_P) = -(Y_i - \overline{Y}_P).$$

The gradient of the score function is ∇θψθ(y) = −1, and hence A<sup>P</sup> = −1. Therefore, up to a constant factor, the gradient-based pseudo-outcomes ρ grad i for conditional mean estimation reduce to their fixed-point counterparts ρ i ,

$$\rho_i^{\text{grad}} = -A_P^{-1}\psi_{\hat{\theta}_P}(Y_i) = Y_i - \overline{Y}_P = -\rho_i^{\text{FPT}}.$$

In this special case, we recover the conventional splitting algorithm used for univariate responses [\(Breiman et al.,](#page-9-18) [1984;](#page-9-18) [Breiman,](#page-9-9) [2001\)](#page-9-9) or for multivariate responses [\(De'ath,](#page-9-12) [2002;](#page-9-12) [Segal,](#page-11-9) [1992\)](#page-11-9). Trees grown using ρ i , ρ grad i , or Y<sup>i</sup> will be identical to one another because CART splits are scale and translation invariant with respect to the response.

More generally, for targets θ ∗ (x) beyond the conditional mean, the form of ρ grad <sup>i</sup> will be equivalent to ρ <sup>i</sup> whenever the target function θ ∗ : X → Θ is a map from the input space X a one-dimensional parameter space Θ.

![](_page_37_Figure_1.jpeg)

Figure 11: Absolute fit times for HTE timing experiments under the settings in Table [3](#page-29-2) and large-n settings in Table [1.](#page-29-4)

## F. Real Data Comparison: California Housing

Data. The California housing data appeared in [Kelley Pace & Barry](#page-10-20) [\(1997\)](#page-10-20) and can be directly obtained from the Carnegie Mellon StatLib repository (<https://lib.stat.cmu.edu/datasets/houses.zip>). The data includes 20640 observations, where each observation corresponds to measurements over an individual census block group in California taken from the 1990 census. A census block is the smallest geographical area for which the U.S. Census Bureau publishes sample data, typically with a population between 600 and 3000 people per block. Each observation from the California housing data set contains measurements of 9 variables: median housing value (dollars), longitude, latitude, median housing age (years), total rooms (count, aggregated over the census block), total bedrooms (count, aggregated over the census block), population (count), households (count), median income (dollars).

Model. We consider a varying coefficient model of the form

$$Y_i = \nu^*(X_i) + \theta_1^*(X_i)W_{i,1} + \cdots + \theta_6^*(X_i)W_{i,6} + \epsilon_i \quad (65)$$

where we suppose that our effects are local to spatial coordinates x := (latitude<sup>i</sup> , longitude<sup>i</sup> ), Y<sup>i</sup> denotes the log median housing value of the census block, and the primary regressors W<sup>i</sup> = (Wi,1, . . . , Wi,6) passed to the model were median housing age, log(total rooms), log(total bedrooms), log(population), log(households), and log(median income). Here, each θ ∗ k (x) denotes the geographically-varying effect of the corresponding regressor Wi,k, for k = 1, . . . , 6. The empirical distribution of the transformed regressors passed to each of the GRF models is seen in Figure [17.](#page-43-0)

![](_page_38_Figure_2.jpeg)

Figure 12: Estimates of MSE <sup>E</sup>[∥θ ∗ (X)− ˆθ(X)/K∥ 2 <sup>2</sup>] for HTE for K = 256 dimensional (top) and K = 4 dimensional targets (bottom) under the large-n settings in Table [1.](#page-29-4)

Algorithms. We target GRF estimates ˆθ(x) = (ˆθ1(x), . . . , ˆθ6(x))<sup>⊤</sup> of θ ∗ (x) = (θ ∗ 1 (x), . . . , θ<sup>∗</sup> 6 (x))<sup>⊤</sup> based on the GRF-FPT1 and GRF-FPT2 algorithms described in Section [6,](#page-7-5) and compare those to GRF-grad. All forests were fit using the grf::lm forest function, which trains the Stage I forest and optionally solves for the Stage II estimates ˆθ(x) for varying coefficient models [\(65\)](#page-37-2). All versions fit a forest of 2000 trees, the default settings of the original R implementation [\(Tibshirani et al.,](#page-11-13) [2024\)](#page-11-13), a subsample ratio of 0.5, and a target minimum node size of 5 observations.

| Algorithm | Training time (sec.) | Speedup factor |
|-----------|----------------------|----------------|
| GRF- grad | 19.1                 |                |
| GRF- FPT1 | 15.4                 | 1.24           |
| GRF- FPT2 | 12.6                 | 1.52           |

Table 7: Fit times to train a forest of 2000 trees on the California housing data.

![](_page_39_Figure_1.jpeg)

Fit time speedup factor: GRF−grad/GRF−FPT (forests)

Figure 13: Speedup factor for GRF-FPT in comparison to GRF-grad for HTE timing experiments under the small-n settings in Table [1.](#page-29-4)

Results. Table [7](#page-38-0) summarizes the computational benefit of GRF-FPT applied to the California housing data. Figures [4](#page-8-1) illustrates the local estimates ˆθ(x) made by GRF-FPT2, while Figure [16](#page-42-0) illustrates the fits under GRF-FPT1 and GRFgrad.

#### Generalized Random Forests using Fixed-Point Trees

![](_page_40_Figure_2.jpeg)

Figure 14: Absolute fit times for HTE timing experiments under the settings in Table [3](#page-29-2) and small-n settings in Table [1.](#page-29-4)

![](_page_41_Figure_2.jpeg)

### MSE estimates: Heterogeneous treatment effects (HTE)

![](_page_41_Figure_3.jpeg)

50 model replications, 5000 test observations

Figure 15: Estimates of MSE <sup>E</sup>[∥θ ∗ (X) − ˆθ(X)/K∥ ] for HTE for K = 16 dimensional (top) and K = 4 dimensional targets (bottom) under the small-n settings in Table [1.](#page-29-4)

![](_page_42_Figure_1.jpeg)

Figure 16: Geographically-varying local estimates ˆθ(x) = (ˆθ1(x), . . . , ˆθ6(x)), fit under GRF-FPT1 (top) and GRF-grad (bottom). Results for GRF-FPT2 are presented in Figure [4](#page-8-1) found in Section [7.](#page-8-2)

![](_page_43_Figure_2.jpeg)

# California housing data: Regressor distribution

Figure 17: Empirical distribution of the regressors from the California housing data passed to GRF.