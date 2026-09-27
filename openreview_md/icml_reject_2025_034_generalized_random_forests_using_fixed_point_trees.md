# 

David Fleischer * 1 **David A. Stephens** * 1 **Archer Y. Yang** * 1 2

## Abstract

We propose a computationally efficient alternative to generalized random forests (GRFs) for estimating heterogeneous effects in large dimensions. While GRFs rely on a gradient-based splitting criterion, which in large dimensions is computationally expensive and unstable, our method introduces a fixed-point approximation that eliminates the need for Jacobian estimation. This gradientfree approach preserves GRF's theoretical guarantees of consistency and asymptotic normality while significantly improving computational efficiency. We demonstrate that our method achieves a speedup of multiple times over standard GRFs without compromising statistical accuracy. Experiments on both simulated and real-world data validate our approach. Our findings suggest that the proposed method is a scalable alternative for localized effect estimation in machine learning and causal inference applications.

## 1. Introduction

In many real-world machine learning (ML) applications, practitioners seek to estimate how quantities of interest vary across different feature subgroups rather than assuming uniform effects. For example, medical interventions and policy treatments often have heterogeneous impacts across subpopulations, making localized estimation crucial for improving outcomes (Imai & Ratkovic, 2013; Knaus et al., 2021; Murdoch et al., 2019; Lee et al., 2020). Similarly, individualized recommendation systems adapt to user-specific features to enhance performance (Kohavi et al., 2013).

1923; Rubin, 1974). The double machine learning framework (Chernozhukov et al., 2018) unifies various ML-based causal estimation methods, including lasso (Belloni et al., 2017), random forests (Athey et al., 2019; Cevid et al.,
2022), boosting (Powers et al., 2018), deep learning (Johansson et al., 2016; Shalit et al., 2017), and general-purpose meta-algorithms (Nie & Wager, 2021; Kunzel et al. ¨ , 2019), all of which focus on capturing variation over feature space. Generalized random forests (GRFs) (Athey et al., 2019; Wager & Athey, 2018) have emerged as a powerful tool for such tasks, leveraging adaptive partitioning with problemspecific moment conditions instead of standard loss-based splits. GRFs apply broadly to a wide range of important statistical models - local linear regression (Friedberg et al., 2020), survival analysis and missing data problems (Cui et al., 2023), nonparametric quantile regression, heterogeneous treatment effect estimation, and nonlinear instrumental variables regression (Athey & Imbens, 2016; Athey et al., 2019). Unlike local linear models (Fan et al., 1995; Fan & Gijbels, 1996; Friedberg et al., 2020) or kernel-based models (Staniswalis, 1989; Severini & Staniswalis, 1994; Lewbel, 2007; Speckman, 1988; Robinson, 1988) which suffer from the curse of dimensionality (Robins & Ritov, 1997), the tree-based approach of GRF offers a more scalable solution.

However, GRFs' gradient-based approach (Athey et al.,
2019) becomes computationally expensive and unstable in large dimensions due to the reliance on Jacobian estimators for tree splitting. To address this, we propose a gradientfree approach based on fixed-point iteration, eliminating the need for Jacobian estimation while retaining GRF's theoretical guarantees of consistency and asymptotic normality. Our method significantly improves computational efficiency while maintaining statistical accuracy, achieving significant speedups in experiments on simulated and real-world datasets.

## 2. Background And Related Work

Given data (Xi, Oi) *∈ X × O*, GRF estimates a target function θ
∗(x), defined as the solution to an estimating equation of the form

$${\mathrm{D)}}\mid X=x$$

1 for all x ∈ X , where ψ is a score function that identifies the true (θ
∗(x), ν∗(x)) as the root of (1), and ν
∗(x)
is an optional nuisance function. GRF can be understood from a nearest-neighbor perspective as approximating θ
∗(x)
through a locally parametric θ
∗ within small neighborhoods of test point x. Suppose L(x) ⊂ {Xi}
n i=1 is a subset of training observations of the covariates found in a region around x ∈ X over which θ
∗(x) can be well-approximated by a local parameter. Observations Xi ∈ L(x) serve as local representatives for x in estimating θ
∗(x) such that, given sufficiently many training samples in a small enough neighborhood of x, an empirical version of (1) over Xi ∈ L(x)
defines an estimator ˆθL(x)that approaches θ
∗(x),

$$\left(\hat{\theta}_{L(x)},\hat{\nu}_{L(x)}\right)\in\operatorname*{arg\,min}_{\theta,\nu}\left\|\sum_{i=1}^{n}\frac{\mathds{1}(X_{i}\in L(x))}{|L(x)|}\cdot\psi_{\theta,\nu}(O_{i})\right\|.\tag{2}$$

In GRF, the set of local representatives L(x) is determined by tree-based partitions which divide the input space into disjoint regions, or leaves. The training samples Xithat fall in the same leaf as x form the subset L(x). However, single trees are known to have high variance with respect to small changes in the training data (Amit & Geman, 1997; Breiman, 1996; 2001; Dietterich, 2000), leading to estimates (2) that do not generalize well to values of x that are not part of the training set. GRF improves its estimates by leveraging an estimating function that averages many estimating functions of the form (2). Specifically, let Lb(x) denote the set of training covariates that fall in the same leaf as x, identified by a tree trained on an independent subsample of the data, indexed by b = 1*, . . . , B*. The GRF estimator is obtained by aggregating the individual estimating functions (2) across a forest of B independently trained trees, i.e. the solution to the following forest-averaged estimating equation:

$$(\hat{\theta}(x),\hat{\nu}(x))\in\operatorname*{arg\,min}_{\theta,\nu}\left\|\frac{1}{B}\sum_{b=1}^{B}\left(\sum_{i=1}^{n}\alpha_{bi}(x)\psi_{\theta,\nu}(O_{i})\right)\right\|_{\cdot}.\tag{3}$$

αi(x) that measure the relative frequency with which trainwhere $\alpha_{bi}(x):=\frac{1(X_{i}\in L_{b}(x))}{|L_{b}(x)|}$. Define observational weights.  
ing sample Xi falls in the same leaf as x, averaged over B trees:
$$\alpha_{i}(x):={\frac{1}{B}}\sum_{b=1}^{B}\alpha_{b i}(x),\qquad\qquad(4)$$
for i = 1*, . . . , n*. Then, the solution (
ˆθ(x), νˆ(x)) to the forest-averaged model (3) is equivalent to solving the following locally weighted estimating equation

$$(\hat{\theta}(x),\hat{\nu}(x))\in\operatorname*{arg\,min}_{\theta,\nu}\left\|\sum_{i=1}^{n}\alpha_{i}(x)\psi_{\theta,\nu}(O_{i})\right\|.\tag{5}$$

Athey et al. (2019) present (5) as the definition of the GRF estimator, motivated in part by the mature analyses of local kernel methods (Newey, 1994) alongside more recent work on tree-based partitioning and estimating equations (Athey & Imbens, 2016; Zeileis & Hornik, 2007; Zeileis et al., 2008). The GRF algorithm for estimating θ
∗(x) can be summarized as a two-stage procedure. **Stage I:** Use trees to calculate weight functions αi(x) for any test observation x ∈ X , measuring the relative importance of the i-th training sample to estimating θ
∗(·) near x. **Stage II:** Given a test observation x ∈ X , compute estimate ˆθ(x) of θ
∗(x) by solving the locally weighted empirical estimating equation (5). Our contribution improves the computational cost of Stage I by introducing a more efficient procedure to train the trees. Training the forest is the most resource-intensive step of GRF, and the cost of each split in the existing approach scales quadratically with the dimension of θ
∗(x). We adopt a gradient-free splitting mechanism and significantly reduce both the time and memory demands of Stage I. Crucially, solving Stage II with weights αi(x) following our streamlined Stage I produces an estimator ˆθ(x) that preserves the finite-sample performance and asymptotic guarantees of GRF.

## 3. Our Method

In this section we describe the details of our accelerated algorithm for GRF. We closely follow the approach of Athey et al. (2019), and define ˆθ(x) as the solution to a locally weighted problem (5) with weighting functions αi(x) of the form (4). The weight functions are induced by a collection of local subsets {Lb(x)}
B
b=1, such that each subset Lb(x)
is determined by the partition rules of a tree trained on a subsample. The construction of each tree, in turn, is determined by recursive splits of the subsample based on a splitting criterion designed to identify regions of X that are homogeneous with respect to θ
∗(x). Therefore, to fully specify the weight functions αi(x), we must describe a feasible criterion for producing a split of X .

## 3.1. The Target Tree-Splitting Criterion For Stage I

In GRF, the goal of Stage I is to use recursive tree-based splits of the training data to induce a partition over the input space. Each split starts with a parent node P ⊂ X and results in child nodes C1, C2 ⊂ X , defined by a binary, axisaligned splitting rule of the form C1 = {Xi: Xi,ℓ ≤ t} and C2 = {Xi: Xi,ℓ > t}, where ℓ denotes a candidate splitting feature/axis and t ∈ R the splitting threshold. For a parent P and any child nodes C1, C2 of P, let (ˆθP , νˆP ) and
(ˆθCj
, νˆCj
) denote local solutions analogous to (2) defined 2 over the samples in P and Cj , respectively:

$$(\hat{\theta}_{P},\hat{\nu}_{P})\in\operatorname*{arg\,min}_{\theta,\nu}\left\|\sum_{\{i:X_{i}\in P\}}\psi_{\theta,\nu}(O_{i})\right\|,\tag{6}$$  $$(\hat{\theta}_{C_{j}},\hat{\nu}_{C_{j}})\in\operatorname*{arg\,min}_{\theta,\nu}\left\|\sum_{\{i:X_{i}\in C_{j}\}}\psi_{\theta,\nu}(O_{i})\right\|,\tag{7}$$  for $j=1,2$. A strategy to split $P$ into two subsets of greater homogeneity with respect to $\theta^{*}(\cdot)$ is as follows:
Find child nodes C1 and C2 such that the total deviation between the local solutions ˆθCjand the target θ
∗(X) is minimized, conditional on X ∈ Cj , j = 1, 2. A natural measure of deviation is the squared-error loss,

$$\operatorname{err}(C_{1},C_{2}):=\sum_{j=1,2}\mathbb{P}\left(X\in C_{j}\mid X\in P\right)$$ $$\times\ \mathbb{E}\left[\left\|\theta^{*}(X)-{\hat{\theta}}_{C_{j}}\right\|^{2}\,\right]X\in C_{j}\right],$$

such that the resulting split (C1, C2) corresponds to least-squares optimal solutions ˆθC1and ˆθC2. However, err(C1, C2) is intractable since θ
∗(·) is unknown. GRF
considers a criterion that measures heterogeneity across a pair of local solutions over a candidate split

$$\Delta(C_{1},C_{2}):=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\hat{\theta}_{C_{1}}-\hat{\theta}_{C_{2}}\right\|^{2},\tag{8}$$

where nC1, nC2, and nP denote the number of observations in C1, C2, and P, respectively. In particular, rather than minimizing err(C1, C2), one can seek a split of P such that the cross-split heterogeneity between ˆθC1and ˆθC2is maximized. Athey et al. (2019) observe that err(C1, C2) and ∆(C1, C2) are coupled according to err(C1, C2) =
K(P) − E [∆(C1, C2)] + o(r 2), where r > 0 is a small radius term tied to the sampling variance, and K(P) does not depend on the split of P. That is, splits that maximize
∆(C1, C2) - which emphasize the heterogeneity of ˆθCj across a split - will asymptotically minimize err(C1, C2),
which aims to improve the homogeneity of ˆθCj within a split. Although the criterion ∆(C1, C2) is computable, evaluating it is very computationally expensive since it requires solving
(7) to obtain ˆθC1
,ˆθC2for all possible splits of P, and closedform solutions for ˆθCjare generally not available except in special cases of ψ. Instead, GRF approximates the target ∆-criterion based on a criterion of the form

$$\tilde{\Delta}^{\mathrm{grad}}(C_{1},C_{2}):=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\tilde{\theta}_{C_{1}}^{\mathrm{xrad}}-\tilde{\theta}_{C_{2}}^{\mathrm{xrad}}\right\|^{2},\tag{9}$$

where ˜θ grad Cjdenotes a *gradient-based* approximation of ˆθCj.

Specifically, ˜θ grad Cjis a first-order approximation interpreted as the result of taking a gradient step away from the parent estimate in the direction towards the true child solution ˆθCj:

$$\bar{\theta}_{C_{j}}^{\mathrm{\tiny{Trad}}}:=\hat{\theta}_{P}-\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\xi^{\top}A_{P}^{-1}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i}),\tag{10}$$

where (ˆθP , νˆP ) is the local solution over the parent, AP
is any consistent estimator of the local Jacobian matrix
∇(θ,ν)E[ψθˆP ,νˆP
(Oi) | Xi ∈ P], and ξ
⊤ can be thought of as a term that selects a θ-subvector from a (*θ, ν*)-vector, e.g.

if θ ∈ R
K and ν ∈ R, then ξ
⊤ such that θ = ξ
⊤(*θ, ν*)
⊤ is the rectangular diagonal matrix ξ
⊤ = [IK 0]. When the scoring function ψ is continuously differentiable in (*θ, ν*),
the Jacobian estimator AP can be computed as

$$A_{P}=\nabla_{(\theta,\nu)}\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i})$$ $$=\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}\nabla_{(\theta,\nu)}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i}).\tag{11}$$

## 3.2. Limitations Of Gradient-Based Approximation

The use of the Jacobian estimator AP in (10) introduces considerable computational challenges. First, each parent node P in every tree of the forest requires a distinct AP
matrix, which imposes a significant computational burden when explicitly calculating A
−1 P ψθˆP ,νˆP
(Oi) to determine
˜θ grad Cj. Second, if the local Jacobian ∇(θ,ν)E[ψθˆP ,νˆP
(Oi) | Xi ∈ P] is ill-conditioned, then the resulting AP estimator may be nearly singular. This instability can lead to highly variable gradient-based approximations ˜θ grad Cjand highly variable splits of P. For example, consider the following varying-coefficient model for an outcome Yi given regressors Wi = (Wi,1, . . . , Wi,K)
⊤ in the presence of mediating auxiliary covariates Xi:

$$\mathbb{E}[Y_{i}\mid X_{i}=x]=\nu^{*}(x)+W_{i}^{\top}\theta^{*}(x),\tag{12}$$

where ν
∗(·) is a nuisance intercept function and θ
∗(x) =
(θ
∗1(x)*, . . . , θ*∗K(x))⊤ are the target coefficients. Models of the form (12) encompass time- or spatially-varying coefficient frameworks, where (Xi, Yi, Wi) represent the i-th sample associated with spatiotemporal values Xi. Such models are particularly relevant in applications like heterogeneous treatment effects; see Section 5 for a more in-depth discussion. The local estimating function ψθ,ν(Yi, Wi),
identifying (θ
∗(x), ν∗(x)) through moment conditions as in (1), is given by:

$$\psi_{\theta,\nu}(Y_{i},W_{i}):=\left[\begin{array}{c}{{(Y_{i}-W_{i}^{\top}\theta-\nu)\cdot W_{i}}}\\ {{Y_{i}-W_{i}^{\top}\theta-\nu}}\end{array}\right].$$

Boxplots: Split values over regressor correlations 0.00 0.25 0.50 0.75 1.00 Spli t Val ueMethod grad FPT
0.00 0.25 0.50 0.75 1.00 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 Regressor Correlation Median split variance over 250 replications 1e−4 1e−3 1e−2 1e−1 0.80 0.85 0.90 0.95 Regressor Correlation Split Var ianceMethod grad FPT
Consequently, the corresponding local Jacobian estimator is

$$A_{P}=\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}\nabla_{(\theta,\nu)}\psi_{\theta,\nu}(Y_{i},W_{i})$$ $$=-\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}\left[\begin{matrix}W_{i}W_{i}^{\top}&W_{i}^{\top}\\ W_{i}&1\end{matrix}\right].\tag{13}$$

When the regressors are highly correlated, the summation over the WiW⊤
iblock of the AP matrix leads to nearly singular values of AP , resulting in an unstable matrix inverse A
−1 P
, and therefore unstable values of ˜θ grad Cjand unstable splits. This issue becomes more pronounced as the number of parent samples nP decreases, as is the case at deeper levels of the tree. These challenges highlight the limitations of relying on AP as part of an approximation for the child solutions ˆθCj.

As an illustration, consider a simple varying coefficient model with primary regressors Wi,1, Wi,2 ∼ N (0, 1), auxiliary covariates Xi ∼ Unif(0, 1), and outcomes Yi generated as

$$Y_{i}=1(X_{i}>0.5)W_{i,1}+W_{i,2}+\epsilon_{i},\tag{14}$$

where ϵi ∼ N (0, 1). Figure 1 illustrates the distribution of 2000 ∆e grad-optimal binary splits (gradient-based tree stumps) fit over 1000 samples of the varying coefficient model (14), repeated over different regressor correlation levels Corr(Wi,1, Wi,2) ∈ {0.80, 0.81*, . . . ,* 0.98, 0.99}. It is clear that splits based on the ∆e grad-criterion exhibit high variability when the correlation between the regressors is large. In contrast, our proposed method, discussed in the next section, does not suffer from the same problem.

## 3.3. Fixed-Point Approximation

To address the limitations of gradient-based approximations, we propose a gradient-free approach based on the form of a single fixed-point iteration. Let ΨCj(*θ, ν*) :=
1 nCj P{i:Xi∈Cj } ψθ,ν(Oi) denote the empirical estimating function for the child solution (ˆθCj
, νˆCj
) such that (7) is equivalently written as:

$$(\hat{\theta}_{C_{j}},\hat{\nu}_{C_{j}})\in\mathop{\arg\min}_{\theta,\nu}\left\|\Psi_{C_{j}}(\theta,\nu)\right\|,\quad j=1,2.\tag{15}$$

Under mild regularity conditions, (ˆθCj, νˆCj) is a Z-
estimator that solves the estimating equation ΨCj(*θ, ν*) = 0.

Reformulating this equation as a fixed-point problem, we write:

$$(\theta,\nu)=\underbrace{(\theta,\nu)-\eta\Psi_{C_{j}}(\theta,\nu)}_{=:f(\theta,\nu)},\quad\eta>0.\tag{16}$$
$$(17)$$

A necessary and sufficient condition for (ˆθCj
, νˆCj
) to be a solution of (15) is characterized by the fixed-point problem
(ˆθCj
, νˆCj
) = f(ˆθCj
, νˆCj
), where f is as defined in (16).

Iterative fixed-point methods (Picard, 1890; Lindelof¨ , 1894; Banach, 1922; Ryu & Boyd, 2016; Yang et al., 2021) solve such problems by considering an update rule of the form

$$(\theta^{+},\nu^{+})\gets f(\theta,\nu).$$

The form of (17) inspires us to approximate the true child solution ˆθCjusing a single fixed-point update taken from the parent solution ˆθP :

$$\hat{\theta}_{C_{j}}^{\mathbb{P}\mathbb{P}}:=\hat{\theta}_{P}-\eta\xi^{\top}\Psi_{C_{j}}(\hat{\theta}_{P},\hat{\nu}_{P})$$ $$=\hat{\theta}_{P}-\frac{\eta}{n_{C_{j}}}\xi^{\top}\sum_{\{i:X_{i}\in C_{j}\}}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i}),\tag{18}$$

where the product with ξ
⊤ is interpreted similarly to its role in the gradient-based approximation (10) and to express the update (17) solely in terms of the target θ-quantity. We interpret ˜θ FPT
Cjas an approximation of ˆθCj obtained by taking a step from ˆθP in a direction that reduces the magnitude of the local estimating function ΨCj. Notably, the approximation ˜θ FPT
Cjdoes not involve the AP matrix, relying only on the scores ψθˆP ,νˆP
(Oi) evaluated at the parent solutions.

In general, removing the inverse A
−1 Pprovides computational cost savings of O(K3). The corresponding splitting criterion, which uses the fixed-point approximations ˜θ FPT Cj as substitutes for ˆθCjis given by

$$\widetilde{\Lambda}^{\mathbb{F}\mathbb{P}\mathbb{T}}(C_{1},C_{2}):=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\widetilde{\theta}_{C_{1}}^{\mathbb{F}\mathbb{T}}-\widetilde{\theta}_{C_{2}}^{\mathbb{F}\mathbb{T}}\right\|^{2}.\tag{19}$$

Revisiting the varying coefficient example from Section 3.2, we see that splits based on fixed-point approximations ˜θ FPT
Cj are significantly more stable than those based on ˜θ grad Cj.

Specifically, Figure 1 illustrates that splits that maximize
∆e FPT(C1, C2) are more robust to ill-conditioning in the underlying local Jacobian ∇(θ,ν)E[ψθˆP ,νˆP
(Oi) | Xi ∈ P],
as is the case for highly correlated regressors in the varying coefficient model (14), and leading to highly stable splits.

## 3.4. Pseudo-Outcomes

Approximations ˜θCj of the form (10) and (18) offer an additional benefit: they enable the ∆e -criteria of the form (9) and
(19) to be efficiently optimized through a single multivariate CART split. A CART split performed with respect to vectorvalued responses ρi ∈ R
K over a parent node P produces a split (C1, C2) that minimizes the following least-squares criterion:

$$\sum_{\{i:X_{i}\in C_{1}\}}\left\|\rho_{i}-\bar{\rho}_{C_{1}}\right\|^{2}+\sum_{\{i:X_{i}\in C_{2}\}}\left\|\rho_{i}-\bar{\rho}_{C_{2}}\right\|^{2},\tag{20}$$  where $\bar{\rho}_{C_{i}}:=\frac{1}{n\alpha}\sum_{\{i:X_{i}\in C_{i}\}}\rho_{i}$. Equivalently, a CART
nCj
split that minimizes (20) will maximize:
$$n_{C_{1}}\left\|{\bar{\rho}}_{C_{1}}\right\|^{2}+n_{C_{2}}\left\|{\bar{\rho}}_{C_{2}}\right\|^{2}.$$

The equivalence between the split that minimizes the leastsquares CART criterion (20) and the split that maximizes (21) is shown in Appendix B.1.1. GRF performs its splits by adopting gradient-based *pseudo-outcomes*, defined as

$$\rho_{i}^{\mathrm{grad}}:=-\xi^{\top}A_{P}^{-1}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i})$$

such that the gradient-based approximation ˜θ grad Cjin (10) is equivalently written:

$$\bar{\theta}_{C_{j}}^{\mathrm{\scriptsize{grad}}}=\hat{\theta}_{P}+\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\rho_{i}^{\mathrm{\scriptsize{grad}}}=\hat{\theta}_{P}+\bar{\rho}_{C_{j}}^{\mathrm{\scriptsize{grad}}}.$$

In the case of fixed-point approximation, we define fixedpoint pseudo-outcomes:

$$\rho_{i}^{\mathbb{P}\mathbb{P}^{\top}}:=-\eta\xi^{\top}\psi_{\theta_{P},\hat{\nu}_{P}}(O_{i}),\quad\eta\neq0,\tag{23}$$

such that the fixed-point approximation ˜θ FPT
Cjin (18) is equivalently written as

$$\hat{\theta}_{C_{j}}^{\mathbb{P}^{\mathbb{P}^{\mathbb{T}}}}=\hat{\theta}_{P}+\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\rho_{i}^{\mathbb{P}^{\mathbb{T}}}=\hat{\theta}_{P}+\bar{\rho}_{C_{j}}^{\mathbb{P}^{\mathbb{T}}}.\tag{24}$$

Substitute the above form of ˜θ FPT
Cjinto the ∆e FPT-criterion
(19) to equivalently express the criterion in terms of the FPT
pseudo-outcomes:

$$\widetilde{\Delta}^{\mathbb{FPT}}(C_{1},C_{2})=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\bar{\rho}_{C_{1}}^{\mathbb{FPT}}-\bar{\rho}_{C_{2}}^{\mathbb{FPT}}\right\|^{2},\tag{25}$$

where an analogous equivalence holds for ∆e grad in terms of the gradient-based pseudo-outcomes. We demonstrate in Lemma B.1 (in Appendix B.1.2) that maximizing the fixedpoint criterion ∆e FPT(C1, C2) is equivalent to maximizing the CART criterion (21), and extend this property to any
∆e -style criterion induced by pseudo-outcomes that can be expressed as a split-independent linear transformation of the parent scores ψθˆP ,νˆP
(Oi).

Note that our method does not rely on iterative fixed-point procedures at all. Instead, it uses only a single step of fixed-point approximation to simplify the pseudo-outcomes. These simplified pseudo-outcomes are then passed directly to a standard CART algorithm for splitting. The numerical convergence of our method therefore relies solely on CART's established and well-known stability, not on fixedpoint iteration. CART splits on pseudo-outcomes are computationally efficient. Given a parent node P, the value ρi = −BψθˆP ,νˆP
(Oi) does not depend on a candidate split
(C1, C2) for any matrix B that is fixed with respect to the parent. This allows much of the computation required to maximize ∆e FPT(C1, C2) to be done at the parent level, and in particular avoids re-calculating the approximations ˜θ FPT
C1 and ˜θ FPT
C2across the sequence of candidate splits. Once P
is fixed and ρ FPT
iare computed, the value of ∆e FPT(C1, C2)
for the first candidate split requires O(nP ) time, and the value for all other candidate splits of P are queried in O(1) time. While gradient-based pseudo-outcomes share this property, the use of fixed-point pseudo-outcomes eliminates the computational overhead and instability associated with estimating AP , as discussed in Section 3.2.

$$(21)$$
$$(22)$$

We show in Lemma B.2 (Appendix B.1.3) that choosing different values of η does not change the outcome of the fixed-point splitting mechanism. Specifically, the optimal split identified by CART on pseudo-outcomes ρ FPT
iof the form (23) does not depend on η. This can be heuristically understood by studying how the criterion changes as a function of the candidate splits. To illustrate, we consider a VCM model of the form (12) for bivariate regressors Wi, univariate Xi ∈ [0, 1], and scalar outcomes Yi. A detailed summary of the settings is found in Appendix D.1. The sequence of valid candidate child nodes obtained by a split over univariate Xi can be parameterized through scalar t as C1(t) := {Xi: Xi ≤ t} and C2(t) := {Xi: Xi > t}. Let ∆(t) := ∆(C1(t), C2(t))
denote the parameterized target criterion (8), and consider the behavior of ∆(t), ∆e grad(t), and two fixed-point criteria ∆e FPT
1(t) and ∆e FPT
2(t) of the form (25) based on pseudooutcomes with scale factors η = 1 and η = 1/
√2, respectively. Figure 2 illustrates the different splitting criteria values plotted against the sequence of candidate splits. The visualization clearly shows that the criteria curves for ∆(t),
∆e grad(t), and ∆e FPT
1(t) with η = 1 are all very close to one

C1 = {Xi : Xi ≤ t} and C2 = {Xi : Xi > t}
Criterion values ∆(C1,C2) over candidate splits {C1,C2}
∆ ∆ ~grad ∆ ~
1 FPTη = 1
∆
~
2 FPTη = 1 2 0.0 0.2 0.4 Cri ter io n val ue
∆

0.00 0.25 0.50 0.75 1.00 Threshold t
another. Critically, the fixed-point criterion with η = 1/
√2, i.e. ∆e FPT
2(t), although scaled differently, still identifies the same maximizing split as ∆e FPT
1(t). This is because CART
chooses a split based on a rank ordering of the criterion over all candidate splits. The absolute scale of the CART criterion does not matter, and it is only criterion rankings over the candidates that determines the optimal split. Therefore, choosing a different scalar η does not change the outcome of the splitting process. Based on the scale-invariance of our splitting criterion, we now detail the recursive procedure for growing our fixedpoint trees pseudo-outcomes with η = 1. The fixed-point tree algorithm. The entire fixed-point tree-growing procedure recursively applies the following two steps on a given parent node P:
(i) **Labeling:** Solve (6) over P to obtain the parent estimate (ˆθP , νˆP ). Compute the pseudo-outcomes:

$$\rho_{i}^{\mathbb{F P T}}:=-\xi^{\top}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i}),$$  that $X_{i}\in P$. 
(Oi), (26)
for all i such that Xi ∈ P.

(ii) **Regression:** Maximize ∆e FPT(C1, C2) by performing a CART split on the pseudo-outcomes ρ FPT
iover P.

## 3.5. Estimates Of ˆΘ(X) **For Stage Ii**

The fixed-point tree algorithm generates a single tree-based partition of X . Repeating this process over subsamples of the training data yields a forest of trees, each specifying local leaf functions Lb(x). These leaf functions define the local weight functions αi(x) via (4), completing Stage I of GRF. The full fixed-point tree training algorithm is described in Algorithm 1, while Algorithm 2 provides the pseudocode for the forest-wide Stage I procedure.

To compute the final GRF estimates ˆθ(x) for the target θ
∗(x), we follow the standard GRF mechanism for Stage II. After the fixed-point trees are trained in Stage I, a test observation x0 ∈ X is assigned to local leaves Lb(x0), indexed by trees b ∈ {1*, . . . , B*}. Each leaf Lb(x0) contains the training observations that fall into the same leaf as x0 in tree b. Using these local leaves, the forest computes training weights αi(x0) as in (4). The final estimate ˆθ(x0) is obtained by solving the locally weighted estimating equation (5).

Importantly, as discussed in Section 2, solving for ˆθ(x0)
in Stage II is independent of the specific mechanism used in Stage I. The only requirement is that Stage I produces valid weights. This ensures that Stage II remains a standard weighted estimating equation, enabling the fixed-point tree algorithm to integrate seamlessly into GRF's two-stage framework. We refer to the complete algorithm for estimating θ
∗(x) using fixed-point trees as GRF-FPT. By preserving Stage II of GRF, the GRF-FPT estimator ˆθ(x)
retains GRF's theoretical guarantees of consistency and asymptotic normality while offering a computationally efficient tree-building method. Pseudocode for Stage II of the GRF-FPT algorithm is provided in Algorithm 3, located in Appendix C.3.

## 4. Theoretical Analysis

In this section, we provide a theoretical foundation for the GRF-FPT estimator ˆθ(x). For Stage I, Proposition 4.1 establishes an asymptotic equivalence between the FPT criterion and a weighted oracle criterion ∆V (C1, C2) in (27), while Lemma 4.2 demonstrates that the Specifications A.2 are met by a forest based on the ∆V -criterion whenever they are met by a forest based on the ∆-criterion. Assumptions A.1 and Specifications A.2 are the sufficient conditions for the consistency and asymptotic normality of ˆθ(x) in (5), and thus are used to formally justify the FPT algorithm as a mechanism for specifying an estimator of θ
∗(x).

$$(26)$$

Proposition 4.1. Suppose Assumptions A.1 *hold, and* assume moreover Neyman orthogonal moment conditions (defined in Appendix *A.4). Denote by* r :=
sup{i:Xi∈P } ∥Xi − xP ∥ the radius of the parent P*, where* xP denotes the center of mass over Xi ∈ P*. Let* Vθθ(xP )
denote the θ*-block of* V (xP ) in (37). Denote by ∥·∥V the weighted Euclidean norm ∥z∥V:= ∥Vθθ(xP )z∥
q 2 
=
z⊤V
⊤
θθ (xP )Vθθ(xP )z. Define the weighted oracle criterion ∆V (C1, C2):

$$\Delta_{V}(C_{1},C_{2}):=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\hat{\theta}_{C_{1}}-\hat{\theta}_{C_{2}}\right\|_{V}^{2}.\tag{27}$$

Then, treating the split as fixed with r
−2 ≪ nC1
, nC2 and sufficiently small r > 0,

$$\widetilde{\Delta}^{^{\mathrm{{\tiny{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$\mathrm{$}}}}}}}}}}}}}}}}}}}(C_{1},C_{2})=\Delta_{V}(C_{1},C_{2})+o_{P}\left(r^{2},\;\frac{1}{n_{C_{1}}},\;\frac{1}{n_{C_{2}}}\right).$$

Lemma 4.2. Let T (∆) denote a tree whose splitting mechanism seeks splits that maximize ∆(C1, C2) defined in (8), and let T (∆V ) denote a tree whose splitting mechanism seeks splits that maximize ∆V (C1, C2) defined in (27). Suppose Assumptions A.1 *hold and assume moreover that* T (∆)
is a tree that satisfies Specifications *A.2. Then,* T (∆V )
satisfies Specifications *A.2.* For Stage II, Theorem 4.3 establishes the consistency of the GRF-FPT estimator ˆθ(x):
Theorem 4.3. Suppose that Assumptions A.1 *hold, and let*
(
ˆθ(x), νˆ(x)) *be estimates that solve* (5) based on weights induced by a forest of trees grown under the fixed-point tree algorithm satisfying Specifications *A.2. Then,* (
ˆθ(x), νˆ(x))
converges in probability to (θ
∗(x), ν∗(x)).

The proof of Theorem 4.3 follows directly from Theorem 3 of Athey et al. (2019), which, under Assumptions A.1, establishes consistency for estimates (
ˆθ(x), νˆ(x)) that solve (5)
with weights from a forest that satisfies Specifications 1-5. Thanks to Lemma 4.2, these forest specifications must also apply to a forest grown under the FPT mechanism. Specifications 1-3 collectively impose mild boundary conditions on the splitting procedure. Meanwhile, Specification 4 requires that trees are trained on subsamples drawn without replacement (Biau et al., 2008; Scornet et al., 2015; Wager et al., 2014; Wager & Athey, 2018), and Specification 5 requires that trees must be grown using an additional subsample splitting mechanism known as honesty (Athey & Imbens, 2016; Biau, 2012; Denil et al., 2014). Appendix C.1 provides a detailed explanation of the subsampling and honest sample splitting procedure. Finally, Theorem 4.4 establishes the asymptotic normality of the GRF-FPT estimator ˆθ(x):
Theorem 4.4. Under the conditions of Theorem 4.3, suppose moreover that Regularity Condition 1 *holds, and* that a forest is grown on subsamples of size s scaling as s = n β, where β satisfies Regularity Condition 2. Then, there exists a sequence σn(x) *such* that (ˆθn(x) − θ
∗(x))/σn(x) ⇝ N (0, 1) and σ 2n
(x) =
polylog(n/s)
−1s/n*, where* polylog(n/s) is a function that is bounded away from 0 and increases at most polynomially with the log of the inverse sampling ratio log(n/s). The proof of Theorem 4.4 is an immediate consequence of Theorem 5 of Athey et al. (2019). Theorems 4.3 and 4.4 demonstrate that the GRF-FPT estimator is able to meet key statistical guarantees.

## 5. Applications

In this section, we explore applications of GRF-FPT for two related models: varying coefficient models and heterogeneous treatment effects. We consider an outcome model of the form introduced in Section 3.2. For each observation, let Yi denote the observed outcome, Wi = (Wi,1, . . . , Wi,K)
⊤
a K-dimensional regressor, and Xi a set of mediating auxiliary variables, such that

$$Y_{i}=\nu^{*}(X_{i})+W_{i}^{\top}\theta^{*}(X_{i})+\epsilon_{i},\tag{28}$$

where ν
∗(·) is a nuisance intercept function, θ
∗(x) =
(θ
∗1
(x)*, . . . , θ*∗K(x))⊤ are the target effect functions local to Xi = x, under the assumptions E[ϵi| Xi = x] = 0 and E[ϵiWi| Xi = x] = 0.

Varying coefficient models (VCM). Given regressors Wi ∈
R 
K, models of the form (28) can be characterized as varying coefficient models (Hastie & Tibshirani, 1993). As discussed in Section 3.2, we must also assume that the regressors Wi are conditionally exogenous given Xi = x.

Heterogeneous treatment effects (HTE). A special case of (28) arises within the Neyman-Rubin potential outcome framework, which models the causal effect of treatment on an outcome (Neyman, 1923; Rubin, 1974). Here, θ
∗(x) =
(θ
∗1(x)*, . . . , θ*∗K(x))⊤ represents heterogeneous treatment effects associated with K discrete treatment levels. Let Ti ∈ {1*, . . . , K*} denote the observed treatment level for the i-th observation, and Yi(k) the potential outcome that would have been observed if treatment level k had been applied. The regressors Wi ∈ {0, 1}
K in (28) are interpreted as a vector of dummy variables indicating the observed treatment level, Wi,k := 1(Ti = k). The auxiliary variables Xi account for potential confounding effects. The conditional average treatment effect of treatment level k ∈ {2*, . . . , K*} relative to the baseline level k = 1 is then defined as:

$$\theta_{k}^{*}(x):=\mathbb{E}\left[Y_{i}(k)-Y_{i}(1)\mid X_{i}=x\right],$$

where the baseline contrast is set to θ
∗
1(x) := 0.

Under exogeneity of the regressors, the target effects θ
∗(x)
in models (28) are identified by moment conditions (1) for scoring function (Angrist & Pischke, 2009; Athey et al., 2019)

$$\psi_{\theta,\nu}(Y_{i},W_{i}):=\left[\begin{array}{c}{{(Y_{i}-W_{i}^{\top}\theta-\nu)\cdot W_{i}}}\\ {{Y_{i}-W_{i}^{\top}\theta-\nu}}\end{array}\right].$$

The gradient-based pseudo-outcomes (22) are computed as

$$\rho_{i}^{\text{grad}}=-A_{P}^{-1}(W_{i}-\overline{W}_{P})\left(Y_{i}-\overline{Y}_{P}-(W_{i}-\overline{W}_{P})^{\top}\hat{\theta}_{P}\right)\tag{29}$$  where $\overline{W}_{P}$ and $\overline{Y}_{P}$ are the local means of $W_{i}$ and $Y_{i}$ over 
,
where WP and Y P are the local means of Wi and Yi over
the observations in P. Centering Yi − Y P and Wi − WP
removes the baseline effect of the mean νˆP on ρ grad i, and where AP is given by (13) as:

$$A_{P}=-\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}(W_{i}-\overline{W}_{P})(W_{i}-\overline{W}_{P})^{\top}.\tag{30}$$

Computing ρ grad iin (29) involves the OLS coefficients ˆθP
from regressing Yi−Y P on Wi−WP , over the observations in P:

$$\hat{\theta}_{P}:=-A_{P}^{-1}\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}(W_{i}-\overline{W}_{P})(Y_{i}-\overline{Y}_{P}).\tag{31}$$
In comparison, $\rho_i^{\tilde{\mathbb{PT}}}$ in (26) are computed as:
In comparison, $\rho_{i}^{\mathbb{P}^{\top}\mathbb{P}^{\top}}$ in (26) are computed as:  $$\rho_{i}^{\mathbb{P}^{\top}\mathbb{P}^{\top}}:=-\xi^{\top}\psi_{\hat{\theta}_{P,\hat{\theta}_{P}}}(Y_{i},W_{i}),$$ $$=-(W_{i}-\overline{W}_{P})\left(Y_{i}-\overline{Y}_{P}-(W_{i}-\overline{W}_{P})^{\top}\hat{\theta}_{P}\right),\tag{32}$$

The relationship ρ grad i = A
−1 P
ρ FPT
ireveals a significant benefit of FPT pseudo-outcomes. The form of ρ FPT
ieliminates the computational cost associated with the multiplication of A
−1 P
, leading to O(K3) computational savings.

Furthermore, the computation of ˆθP in (32) no longer requires solving for A
−1 P. Therefore, we can further enhance computational efficiency by using an accelerated form of pseudo-outcome ϕ FPT
iinstead of ρ FPT
i:

$$\phi_{i}^{\mathbb{F}\mathbb{P}^{\mathbb{T}}}:=-(W_{i}-\overline{W}_{P})\left(Y_{i}-\overline{Y}_{P}-(W_{i}-\overline{W}_{P})^{\top}\tilde{\theta}_{P}\right),\tag{33}$$

where ˆθP is replaced by ˜θP in (32), which is defined as a one-step gradient descent approximation of ˆθP taken from the origin:

$$\tilde{\theta}_{P}:=\gamma\frac{1}{n_{P}}\sum_{\{i:X_{i}\in P\}}(W_{i}-\overline{W}_{P})(Y_{i}-\overline{Y}_{P}).\tag{34}$$

Here, γ denotes the exact line search step size for the regression of Yi − Y P on Wi − WP over P:

$$\gamma:=\frac{\left\|(W-\overline{W}_{P})^{\top}(Y-\overline{Y}_{P})\right\|_{2}^{2}}{\left\|(W-\overline{W}_{P})(W-\overline{W}_{P})^{\top}(Y-\overline{Y}_{P})\right\|_{2}^{2}},\tag{35}$$

where W = [W1 *· · ·* WnP ]
⊤and Y = [Y1 *· · ·* YnP ]
⊤ with the notation W −WP and Y −Y P understood as row-wise centering.

The computational cost associated with ˜θP is comparatively small because many of the products that appear in (34) and
(35) are already computed as part of ρ FPT
iin (32). Meanwhile, we show in Appendix B.3 that the approximation for the FPT child estimator:

$$\hat{\theta}_{C_{j}}^{\mathbb{P}\mathbb{P}\mathbb{T}}:=\hat{\theta}_{P}+\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\phi_{i}^{\mathbb{P}\mathbb{P}\mathbb{T}},$$

is consistent for the original FPT child estimator ˜θ FPT
Cjas
∥˜θ FPT
Cj − ¯θ FPT
Cj
∥ = oP (1), meaning that this approximation does not alter the asymptotic behavior of our estimator. These accelerations are particularly compelling when the dimension of θ
∗(x) is large and computational efficiency is critical, as in large-scale A/B testing with multiple concurrent treatment arms or observational studies with numerous treatment levels (Kohavi et al., 2013; Bakshy et al., 2014).

## 6. Simulations

In this section, we perform empirical evaluations of the computational efficiency and estimation accuracy of the GRF-FPT method. We let GRF-FPT1 denote the FPT algorithm using the exact form of the FPT VCM/HTE pseudooutcomes (32) and we let GRF-FPT2 denote the accelerated FPT algorithm based on the form of the FPT pseudooutcome approximation (33) in Section 5. We compare both implementations relative to GRF-grad under VCM
and HTE designs. Implementation details and links to the reproducible code are found in Appendix C.4.

Settings. We follow the structural model in (28). The auxiliary variables Xi are drawn from the Gaussian copula with latent covariance matrix Σ, where [Σ]j,k = (0.3)|j−k|.

Supporting experiments for multicollinearity in Xi can be found in Appendix D.2. The outcomes Yi follow (28) with Gaussian noise ϵi ∼ N (0, 1). For VCM experiments, regressors Wi ∈ R
K are sampled from NK(0,I). For HTE
experiments, Wi ∈ {0, 1}
K follows a multinomial distribution, Wi| Xi = x ∼ Multinomial(1,(π1(x)*, . . . , π*K(x))), where πk(x) is the probability of treatment level k ∈
{1*, . . . , K*}, characterizing a variety of different locationspecific dependence structures through the setting of πk(·).

We set ν
∗(x) := 0 and vary the target effect functions θ
∗
k(x)
and treatment probabilities πk(x) across different settings, fully detailed in Appendix C.4. Throughout our experiments we use subsampling ratio s/n = 0.5. Supporting experiments under different subsample ratios are found in Appendix D.2. Results. The relative computational advantage of forests trained under GRF-FPT is displayed in Figure 3, while Figure 5 (in Appendix D.3) summarizes the absolute fit times across the three methods. These data show that the FPT
mechanism is able to consistently offer a relative advantage, observing speedups of up to 3.5× faster than the gradientbased approach at the largest dimension K = 256. Figure 3 also shows increasing gains with increasing K and provides an empirical measurement of the theoretical scaling benefits discussed in Section 5. Moreover, the absolute fit times in Figure 5 (in Appendix D.3) illustrate that our method consistently remains faster than GRF-grad, with no clear computational or algorithmic bottleneck as a function of either n or K. Supporting experiments exploring the ef-

Varying coefficient model (VCM) Fit time speedup factor: GRF−grad/GRF−FPT (forests)
FPT1 **FPT2**
1.0 1.5 2.0 2.5 3.0 3.5 dim(X) = 5 dim(X) = 5 n = 
100 00 Spee du p fa ctor VCM Setting 1 2 3 4 1.0 1.5 2.0 2.5 3.0 3.5 dim(X) = 5 dim(X) = 5 nTre es = 100 n = 
200 00 1.0 1.5 2.0 2.5 3.0 3.5 dim(X) = 5 dim(X) = 5 n = 1 000 00 4 16 64 256 4 16 64 256 Regressor dimension (K)
Spatially−varying effects on log median house values California housing data: GRF−FPT2 log Households Median Housing Age **log Median Income**
Sacramento Sacramento Sacramento San Francisco San Francisco San Francisco Effect on log value
−3
−2

−1 0 1 2 3 Los Angeles Los Angeles Los Angeles San Diego San Diego San Diego log Population log Census Block Bedrooms **log Census Block Rooms**
Sacramento Sacramento Sacramento San Francisco San Francisco San Francisco Los Angeles Los Angeles Los Angeles San Diego San Diego San Diego
fects of sample sizes up to n = 500, 000 are presented in Appendix D.2, while Figures 7 and 8 (in Appendix D.3)
show that even when n is small, GRF-FPT still observes a noticeable gain relative to GRF-grad. Additional timing benchmarks for VCM experiments and all HTE experiments are discussed in Appendix D.3. To assess estimation accuracy, we evaluate the mean squared error (MSE) of ˆθ(x) across 50 replications of the model and testing on a separate set of 5, 000 observations. Figure 6 in Appendix D.3 confirms that GRF-FPT matches the accuracy of GRF-grad, while significantly reducing computation time. Further comparisons for both VCM and HTE settings are provided in Appendix D.3.

## 7. Real Data Application

Data. In this section we apply GRF-FPT to the analysis of geographically-varying effects θ
∗(x) on housing prices.

The data, first appearing in Kelley Pace & Barry (1997), contains 20,640 observations of housing prices taken from the 1990 California census. Each observation corresponds to measurements aggregated over a small geographical census block, and contains measurements of 9 variables: median housing value, longitude, latitude, median housing age, total rooms, total bedrooms, population, households, and median income. We employ a VCM design of the form (28) where Yi denotes the housing value, Xi denote the spatial coordinates, and Wi = (Wi,1*, . . . , W*i,6)
⊤ are the remaining six regressors. Details of the model and data transformations used for the California housing analysis is found in Appendix F.

Results. Table 7 summarizes the computational benefit of GRF-FPT applied to the California housing data. Figure 4 illustrates the six geographically-varying effect estimates under GRF-FPT2, with qualitatively similar results shown in Figure 16 for GRF-FPT1 and GRF-grad in Appendix F. Figure 4 shows clearly the geographically-dependent relationship between different housing features and housing prices. In major urban centers such as LA, San Francisco, and Sacramento, housing prices tend to decrease with an increasing number of households, and may reflect overcrowding in densely populated areas. In contrast, rural regions show the opposite trend: prices rise slightly when rural areas have a larger number of housing units. This suggests that, in sparsely populated rural areas, a modest increase in households makes these places more attractive and livable. Median income, however, consistently shows a positive effect on prices across nearly all of California, while population size tends to show a negative effect, highlighting broader state-wide pressures on housing affordability.

## 8. Conclusion

Our results demonstrate that the FPT algorithm offers a substantial computational advantage over GRF-grad with comparable statistical accuracy, and highlights GRF-FPT as a powerful method for multi-dimensional estimation, particularly when estimates of the target function must be learned from the data rather than observed directly. Future work may explore extensions to larger-scale problems and alternative estimation tasks, as in unsupervised learning and structured prediction. Our findings position GRF-FPT as a scalable and robust alternative for practitioners seeking efficient localized estimation.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgments

This work was supported by Natural Sciences and Engineering Research Council (NSERC) Discovery Grant (RGPIN-
2024-06780) and FRQNT Team Research Project Grant
(FRQ-NT 327788).

## References

Amit, Y. and Geman, D. Shape quantization and recognition with randomized trees. *Neural Computation*, 9(7):1545– 1588, 1997.

Angrist, J. D. and Pischke, J.-S. Mostly harmless econometrics: An empiricist's companion. Princeton university press, 2009.

Athey, S. and Imbens, G. Recursive partitioning for heterogeneous causal effects. *Proceedings of the National* Academy of Sciences, 113(27):7353–7360, 2016.

Athey, S., Tibshirani, J., and Wager, S. Generalized random forests. *The Annals of Statistics*, 47(2):1148 - 1178, 2019. doi: 10.1214/18-AOS1709. URL https:// doi.org/10.1214/18-AOS1709.

Bakshy, E., Eckles, D., and Bernstein, M. S. Designing and deploying online field experiments. In Proceedings of the 23rd International Conference on World Wide Web, WWW '14, pp. 283–292, New York, NY, USA, 2014. Association for Computing Machinery. ISBN 9781450327442. doi: 10.1145/ 2566486.2567967. URL https://doi.org/10. 1145/2566486.2567967.

Banach, S. Sur les operations dans les ensembles abstraits ´
et leur application aux equations int ´ egrales. ´ Fundamenta Mathematicae, 3:133–181, 1922.

Belloni, A., Chernozhukov, V., Fernandez-Val, I., and Hansen, C. Program evaluation and causal inference with high-dimensional data. *Econometrica*, 85(1):233–298, 2017.

Biau, G. Analysis of a random forests model. *The Journal* of Machine Learning Research, 13(1):1063–1095, 2012.

Biau, G., Devroye, L., and Lugosi, G. Consistency of random forests and other averaging classifiers. Journal of Machine Learning Research, 9(66):2015–2033, 2008.

URL http://jmlr.org/papers/v9/biau08a. html.

Breiman, L. Bagging predictors. *Machine Learning*, 24:
123–140, 1996.

Breiman, L. Random forests. *Machine Learning*, 45:5–32, 2001.

Breiman, L., Friedman, J., Olshen, R. A., and Stone, C. J.

Classification and Regression Trees. CRC, 1984. ISBN 9780412048418.

Cevid, D., Michel, L., Naf, J., B ¨ uhlmann, P., and Mein- ¨
shausen, N. Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. *Journal of Machine Learning Research*, 23(333):1– 79, 2022. URL http://jmlr.org/papers/v23/
21-0585.html.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
Hansen, C., Newey, W., and Robins, J. Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1):C1–C68, 01 2018. ISSN 1368-4221. doi: 10.1111/ectj.12097. URL
https://doi.org/10.1111/ectj.12097.

Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., and Zhu, R. Estimating heterogeneous treatment effects with rightcensored data via causal survival forests. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(2):179–211, 02 2023. ISSN 1369-7412. doi:
10.1093/jrsssb/qkac001. URL https://doi.org/ 10.1093/jrsssb/qkac001.

De'ath, G. Multivariate regression trees: a new technique for modeling species–environment relationships. *Ecology*, 83(4):1105–1117, 2002.

Denil, M., Matheson, D., and De Freitas, N. Narrowing the gap: Random forests in theory and in practice. In Xing, E. P. and Jebara, T. (eds.), *Proceedings of the* 31st International Conference on Machine Learning, volume 32 of *Proceedings of Machine Learning Research*, pp. 665–673, Bejing, China, 22–24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/ denil14.html.

Dietterich, T. G. An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting, and randomization. *Machine Learning*, 40:139–157, 2000.

Fan, J. and Gijbels, I. Local Polynomial Modelling and Its Applications, volume 66 of Monographs on Statistics and Applied Probability. Chapman & Hall/CRC, London, 1996. doi: 10.1201/9780203748725. URL
https://www.taylorfrancis.com/books/ mono/10.1201/9780203748725.

Fan, J., Heckman, N. E., and Wand, M. P. Local polynomial kernel regression for generalized linear models and quasilikelihood functions. Journal of the American Statistical Association, 90(429):141–150, 1995.

Friedberg, R., Tibshirani, J., Athey, S., and Wager, S. Local linear forests. Journal of Computational and Graphical Statistics, 30(2):503–517, 2020.

Friedman, J. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, pp. 1189–1232, 2001.

Hastie, T. and Tibshirani, R. Varying-coefficient models.

Journal of the Royal Statistical Society. Series B (Methodological), 55(4):757–796, 1993. ISSN 00359246. URL http://www.jstor.org/stable/2345993.

Imai, K. and Ratkovic, M. Estimating treatment effect heterogeneity in randomized program evaluation. The Annals of Applied Statistics, 7(1):443 - 470, 2013. doi: 10.1214/12-AOAS593. URL https://doi.org/10. 1214/12-AOAS593.

Johansson, F., Shalit, U., and Sontag, D. Learning representations for counterfactual inference. In International Conference on Machine Learning, pp. 3020–3029. PMLR, 2016.

Kelley Pace, R. and Barry, R. Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3):291–297, 1997. ISSN 0167-7152. doi: https://doi.org/10.1016/S0167-7152(96)00140-X.

URL https://www.sciencedirect.com/ science/article/pii/S016771529600140X.

Knaus, M. C., Lechner, M., and Strittmatter, A. Machine learning estimation of heterogeneous causal effects: Empirical monte carlo evidence. *The Econometrics Journal*, 24(1):134–161, 2021.

Kohavi, R., Deng, A., Frasca, B., Walker, T., Xu, Y.,
and Pohlmann, N. Online controlled experiments at large scale. In *Proceedings of the 19th ACM*
SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '13, pp. 1168–1176, New York, NY, USA, 2013. Association for Computing Machinery. ISBN 9781450321747. doi: 10.1145/
2487575.2488217. URL https://doi.org/10. 1145/2487575.2488217.

Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. Met- ¨
alearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10):4156–4165, 2019.

Lee, Y., Veerubhotla, K., Jeong, M. H., and Lee, C. H. Deep learning in personalization of cardiovascular stents. Journal of Cardiovascular Pharmacology and Therapeutics, 25(2):110–120, 2020.

Lewbel, A. A local generalized method of moments estimator. *Economics Letters*, 94(1):124–128, 2007.

Lindelof, E. Sur l'application de la m ¨ ethode des approxima- ´
tions successives aux equations diff ´ erentielles ordinaires ´ du premier ordre. Comptes Rendus Hebdomadaires des Seances de l'Acad ´ *emie des Sciences* ´ , 116:454–457, 1894.

Murdoch, W. J., Singh, C., Kumbier, K., Abbasi-Asl, R.,
and Yu, B. Definitions, methods, and applications in interpretable machine learning. Proceedings of the National Academy of Sciences, 116(44):22071–22080, 2019.

Newey, W. K. Kernel estimation of partial means and a general variance estimator. *Econometric Theory*, 10(2):
1–21, 1994.

Neyman, J. Sur les applications de la theorie des prob- ´
abilites aux experiences agricoles: Essai des principes. ´ Roczniki Nauk Rolniczych, 10(1):1–51, 1923. Reprinted and translated in Neyman, J. (1990). Statistical Science, 5(4), 463–480.

Nie, X. and Wager, S. Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2):299–319, 2021.

Picard, E. M´ emoire sur la th ´ eorie des ´ equations aux d ´ eriv ´ ees ´
partielles et la methode des approximations successives. ´ Journal de Mathematiques Pures et Appliqu ´ ees ´ , 6:145–
210, 1890.

Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H.,
Hastie, T., and Tibshirani, R. Some methods for heterogeneous treatment effect estimation in high dimensions.

Statistics in Medicine, 37(11):1767–1787, 2018.

Robins, J. M. and Ritov, Y. Toward a curse of dimensionality appropriate (coda) asymptotic theory for semiparametric models. *Statistics in Medicine*, 16(1-3):285–
319, 1997. doi: 10.1002/(SICI)1097-0258(19970215)16: 3⟨285::AID-SIM535⟩3.0.CO;2-\\#. URL https:// pubmed.ncbi.nlm.nih.gov/9004398/.

Robinson, P. M. Root-n-consistent semiparametric regression. *Econometrica: Journal of the Econometric Society*, pp. 931–954, 1988.

Rubin, D. B. Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66(5):688, 1974.

Ryu, E. K. and Boyd, S. A primer on monotone operator methods (survey). Applied and Computational Mathematics, 15(1):3–43, 2016. Survey article.

Scornet, E., Biau, G., and Vert, J.-P. Consistency of random forests. *The Annals of Statistics*, 43(4):1716 - 1741, 2015. doi: 10.1214/15-AOS1321. URL https:// doi.org/10.1214/15-AOS1321.

Segal, M. R. Tree-structured methods for longitudinal data.

Journal of the American Statistical Association, 87(418): 407–418, 1992.

Severini, T. A. and Staniswalis, J. G. Quasi-likelihood estimation in semiparametric models. Journal of the American Statistical Association, 89(426):501–511, 1994.

Shalit, U., Johansson, F. D., and Sontag, D. Estimating individual treatment effect: generalization bounds and algorithms. In Precup, D. and Teh, Y. W. (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 3076–3085. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/ shalit17a.html.

Speckman, P. Kernel smoothing in partial linear models.

Journal of the Royal Statistical Society. Series B (Methodological), 50(3):413–436, 1988. ISSN 00359246. URL http://www.jstor.org/stable/2345705.

Staniswalis, J. G. The kernel estimate of a regression function in likelihood-based models. Journal of the American Statistical Association, 84(405):276–283, 1989.

Tibshirani, J., Athey, S., Sverdrup, E., and Wager, S. *grf:*
Generalized Random Forests, 2024. URL https:// github.com/grf-labs/grf. R package version 2.4.0.

Wager, S. and Athey, S. Estimation and inference of heterogeneous treatment effects using random forests. *Journal* of the American Statistical Association, 113(523):1228– 1242, 2018.

Wager, S. and Walther, G. Adaptive concentration of regression trees, with application to random forests. arXiv preprint arXiv:1503.06388, 2015.

Wager, S., Hastie, T., and Efron, B. Confidence intervals for random forests: The jackknife and the infinitesimal jackknife. *Journal of Machine Learning Research*, 15 (1):1625–1651, 2014. URL https://jmlr.org/ papers/volume15/wager14a/wager14a.pdf.

Yang, Y., Gu, Y., Zhao, Y., and Fan, J. Flexible regularized estimating equations: Some new perspectives. 2021. URL https://arxiv.org/abs/2110.11074.

Zeileis, A. and Hornik, K. Generalized m-fluctuation tests for parameter instability. *Statistica Neerlandica*, 61(4): 488–508, 2007.

Zeileis, A., Hothorn, T., and Hornik, K. Model-based recursive partitioning. Journal of Computational and Graphical Statistics, 17(2):492–514, 2008.

## A. Technical Preliminaries A.1. Assumptions

We follow the key assumptions of Athey et al. (2019) made for the theoretical analyses of GRF. The predictor and parameter spaces are both subsets of Euclidean space such that x ∈ X = [0, 1]pand (θ, ν) *∈ B ⊂* R
K, where B is a compact subset of R
K. Under the analyses of Wager & Walther (2015), we suppose that the features of the auxiliary covariates Xi = (Xi,1, . . . , Xi,p)
⊤ have density fX that is bounded away from 0 and ∞, i.e. c ≤ fX(x) ≤ C < ∞, for some constants c > 0 and C < ∞. GRF does not require that the score function ψ is continuous in (*θ, ν*), as is the case for quantile estimation, one does require that the expected score/moment function

$$M_{\theta,\nu}(x):=\mathbb{E}_{O|X}\left[\psi_{\theta,\nu}(O)\mid X=x\right],$$
Mθ,ν(x) := EO|X [ψθ,ν(O) | X = x] , (36)
is smoothly varying in its parameters (*θ, ν*).

ASSUMPTION 1. For fixed (θ, ν), the M-function (36) is Lipschitz continuous in x. ASSUMPTION 2. For fixed x, the M-function is twice-differentiable in (*θ, ν*) with uniformly bounded second derivative,

$$(36)$$
$$\left\|\nabla_{(\theta,\nu)}^{2}M_{\theta,\nu}(x)\right\|<\infty,$$

where · denotes the appropriate tensor norm for the second derivative of Mθ,ν taken with respect to (θ, ν). Let V (x) := ∇(θ,ν)Mθ,ν(x)θ=θ
∗(x),ν=ν∗(x)
denote the population Jacobian at the true (θ
∗(x), ν∗(x)), and assume that V (x)
is invertible for all x ∈ X . We write V (x) in block form as

$$V(x)=\begin{bmatrix}V_{\theta\theta}(x)&V_{\theta\nu}(x)\\ V_{\nu\theta}(x)&V_{\nu\nu}(x)\end{bmatrix}.\tag{1}$$
$$(37)$$

ASSUMPTION 3. The score functions ψθ,ν(Oi) have a continuous covariance structure in the following sense: Let γ(·, ·)
denote the worst-case variogram:

$$\gamma\left(\begin{bmatrix}\theta_{1}\\ \nu_{1}\end{bmatrix},\begin{bmatrix}\theta_{2}\\ \nu_{2}\end{bmatrix}\right):=\operatorname*{sup}_{x\in\mathcal{X}}\left\{\left|\mathrm{Var}_{O|X}\left(\psi_{\theta_{1},\nu_{1}}(O_{i})-\psi_{\theta_{2},\nu_{2}}(O_{i})\mid X_{i}=x\right)\right|\right|_{F}\right\},$$

then, for some L > 0,

$$\gamma\left(\begin{bmatrix}\theta_{1}\\ \nu_{1}\end{bmatrix},\begin{bmatrix}\theta_{2}\\ \nu_{2}\end{bmatrix}\right)\leq L\left\|\begin{bmatrix}\theta_{1}\\ \nu_{1}\end{bmatrix}-\begin{bmatrix}\theta_{2}\\ \nu_{2}\end{bmatrix}\right\|_{2},\quad\mathrm{for~all~}(\theta_{1},\nu_{1}),\ (\theta_{2},\nu_{2}).$$

ASSUMPTION 4. The score function ψθ,ν(Oi) can be written as

$$\mathbf{u}(O_{i})\mathbf{)},$$

ψθ,ν(Oi) = λ(*θ, ν*; Oi) + ζθ,ν(g(Oi)),
where λ is Lipschitz-continuous in (θ, ν), g : {Oi} → R a univariate summary of the observables Oi, and ζθ : R → R any family of monotone and bounded functions.

ASSUMPTION 5. For any weights αi with Pαi = 1, the minimizer (
ˆθ, νˆ) of the weighted empirical estimation problem
(5) satisfies:

$$\left\|\sum_{i=1}^{n}\alpha_{i}\psi_{\hat{\theta},\hat{\nu}}(O_{i})\right\|_{2}\leq C\operatorname*{max}_{1\leq i\leq n}\{\alpha_{i}\},\quad{\mathrm{for~}}C\geq0.$$

ASSUMPTION 6. The score function ψθ,ν(Oi) is a negative subgradient of a convex function, and the moment function Mθ,ν(Xi) is the negative gradient of a strongly convex function.

## A.2. Forest Specifications

The consistency and asymptotic normality results, Theorems 4.3 and 4.4, require that the forest trained following Algorithm 2 consists of trees that satisfy a certain set of specifications. These forest specifications are precisely those imposed by Athey et al. (2019) for forests of gradient-based trees, and collectively, these specifications describe fairly mild conditions on the tree splitting mechanism, as well as specific requirements for the sampling procedure.

SPECIFICATION 1. (*Symmetric*) Tree estimates are invariant to permutations of the training indices. In other words, the output of a tree does not depend on the order in which the training samples are indexed.

SPECIFICATION 2. (Balanced/ω*-regular*) The proportion of parent observations assigned into either child is bound below by some ω > 0, i.e. nCj ≥ ωnP .

SPECIFICATION 3. (*Randomized/random-split*) The probability of splitting along any feature/dimension of the input space is bound below by some π > 0.

SPECIFICATION 4. (*Subsampling*) Trees are trained on subsample of size s, drawn without replacement from n training samples, where s/n → 0 as s → ∞.

SPECIFICATION 5. (*Honesty*) Trees are trained using the sample splitting procedure described in Appendix C.1.

## A.3. Regularity Conditions

REGULARITY CONDITION 1. Let V (x) be as defined in Assumption 2 and let ρ
∗
i(x) denote the influence function of the i-th observation with respect to the target θ
∗(x):

$$\rho_{i}^{*}(x):=-\xi^{\top}V(x)^{-1}\psi_{\theta^{*}(x),\nu^{*}(x)}(O_{i}).$$

Then,

$\mbox{Var}(\rho_{i}^{*}(x)\mid X_{i}=x)>0$, for all $x\in\mathcal{X}$.  
REGULARITY CONDITION 2. Trees are grown on subsamples of size s scaling as s = n β, for some subsample scaling exponent β bound according to βmin *< β <* 1, such that

$$\beta_{\mathrm{min}}:=1-\left(1+{\frac{1}{\pi}}\cdot{\frac{\log\left(\omega^{-1}\right)}{\log\left((1-\omega)^{-1}\right)}}\right)^{-1}<\beta<1,$$

where 0 *< π, ω <* 1 are constants defined in forest Specifications 2 and 3.

## A.4. Neyman Orthogonality

To identify the underlying local parameters (θ
∗(x), ν∗(x)) ∈ R
K one must have a score ψθ,ν(O) with at least K = Kθ +Kν components, where here we use Kθ and Kν to denote the dimensions of the component subvectors θ
∗(x) ∈ R
Kθ and ν
∗(x) ∈ R
Kν . Conceptually, a score ψθ,ν(O) can be partitioned into the components that identify the θ-coordinates, denoted by ψ1, and those that identify the ν-coordinates, denoted by ψ2, and thus the moment functions Mθ,ν(x) in (36) can also be partitioned the same way:

$$\psi_{\theta,\nu}(O)=\begin{bmatrix}\psi_{1}(\theta,\nu;O)\\ \psi_{2}(\theta,\nu;O)\end{bmatrix},\qquad M_{\theta,\nu}(x)=\begin{bmatrix}M_{1}(\theta,\nu;x)\\ M_{2}(\theta,\nu;x)\end{bmatrix}=\begin{bmatrix}\mathbb{E}[\psi_{1}(\theta,\nu;O)\mid X=x]\\ \mathbb{E}[\psi_{2}(\theta,\nu;O)\mid X=x]\end{bmatrix}.$$

The corresponding Jacobian matrix of Mθ,ν(x) taken with respect to (*θ, ν*) and evaluated at the truth (θ
∗(x), ν∗(x)) is

$$V(x)=\nabla_{(\theta,\nu)}\;M(\theta,\nu;x)|_{\theta=\theta^{*}(x),\nu=\nu^{*}(x)}=\begin{bmatrix}V_{\theta\theta}(x)&V_{\theta\nu}(x)\\ V_{\nu\theta}(x)&V_{\nu\nu}(x)\end{bmatrix},$$

where here the subscripts in the block expressions of V (x) indicate the coordinates with which the gradient is taken, and in all cases are evaluated at the truth (θ
∗(x), ν∗(x)):

$$V_{\theta\theta}(x)=\nabla_{\theta}\ M_{1}(\theta,\nu;x)|_{\theta=\theta^{\star}(x),\nu=\nu^{\star}(x)}\ ,$$
$\left[x\right]$ . 
Vθν(x) = 0.
$$\left.\partial_{\nu}(x)=\nabla_{\nu}\ M_{1}(\theta,\nu;x)\right|_{\theta=\theta^{*}(x),\nu=\nu^{*}(x)},$$
$\sigma(x)=\nabla_{\theta}\ M_{2}(\theta,\nu;x)|_{\theta=\theta^{*}}(x),\nu=\nu^{*}(x)$
$$\left.\nabla\nu(x)=\nabla_{\nu}\ M_{2}(\theta,\nu;x)\right|_{\theta=\theta}$$
∗(x),ν=ν∗(x)
.
In this context, the assumption of Neyman orthogonal moment conditions is more completely labeled as Neyman orthogonality for the estimation of θ
∗(x) with respect to the nuisance ν
∗(x), and can be summarized as an assumption that the moment conditions for θ
∗(x) are insensitive to first-order changes in ν around the truth ν
∗(x) whenever θ = θ
∗(x). For GRF, this means that one assumes (1) satisfies M1(θ
∗(x), ν∗(x); x) = 0, and in other words, the partial derivatives of the moment functions for θ
∗(x) with respect to ν are zero at (θ
∗(x), ν∗(x)):

## A.5. Example: Neyman Orthogonality For Vcm And Hte

Consider the VCM/HTE model with data (Yi, Wi, Xi) related according to

$$\mathbb{E}[Y_{i}\mid X_{i}=1]$$
E[Yi| Xi = x] = ν
∗(x) + W⊤
i
θ
∗(x),
such that, as discussed in Section 3.2, the score function ψθ,ν that identifies the underlying (θ
∗(x), ν∗(x)) is

$$\psi_{\theta,\nu}(Y_{i},W_{i}):=\left[\begin{array}{c}{{(Y_{i}-W_{i}^{\top}\theta-\nu)W_{i}}}\\ {{Y_{i}-W_{i}^{\top}\theta-\nu}}\end{array}\right],$$

and the corresponding local Jacobian V (x) has block form

$$V(x)=-\mathbb{E}\left[\begin{bmatrix}W_{i}W_{i}^{\top}&W_{i}^{\top}\\ W_{i}&1\end{bmatrix}\mid X_{i}=x\right]=-\begin{bmatrix}\mathbb{E}[W_{i}W_{i}^{\top}\mid X_{i}=x]&\mathbb{E}[W_{i}^{\top}\mid X_{i}=x]\\ \mathbb{E}[W_{i}\mid X_{i}=x]&1\end{bmatrix}.$$

Therefore, for Neyman orthogonality to hold one requires that E[Wi| Xi = x] = 0.

## B. Derivations And Proofs

B.1. Proofs for Section 3.4 B.1.1. MULTIVARIATE CART CRITERIA
Let ρi ∈ R
K be vector-valued responses associated with covariates Xi ∈ P. A standard CART split (C1, C2) of P
minimizes the conventional least-squares criterion:

$$\sum_{\{i:X_{i}\in C_{1}\}}\|\rho_{i}-\bar{\rho}_{C_{1}}\|^{2}+\sum_{\{i:X_{i}\in C_{2}\}}\|\rho_{i}-\bar{\rho}_{C_{2}}\|^{2}\,,$$
2, (38)
where ρ¯Cj:= n
−1
Cj
P{i:Xi∈Cj }
ρiis the local prediction over child node Cj . We verify that a split (C1, C2) minimizes (38)
if and only if it maximizes
$$n_{C_{1}}\left\|{\bar{\rho}}_{C_{1}}\right\|^{2}+n_{C_{2}}\left\|{\bar{\rho}}_{C_{2}}\right\|^{2}.$$
2. (39)
Proof. Each sum in (38) can be expanded as

$$\sum_{\{i:X_{i}\in C_{j}\}}\left\|\rho_{i}-\bar{\rho}_{C_{j}}\right\|^{2}=\sum_{\{i:X_{i}\in P\}}\left\|\rho_{i}-\bar{\rho}_{C_{j}}\right\|^{2}\cdot\mathds{1}(X_{i}\in C_{j}),$$ $$=\sum_{\{i:X_{i}\in P\}}\left(\left\|\rho_{i}\right\|^{2}-2\rho_{i}^{\top}\bar{\rho}_{C_{j}}+\left\|\bar{\rho}_{C_{j}}\right\|^{2}\right)\cdot\mathds{1}(X_{i}\in C_{j}),$$ $$=\sum_{\{i:X_{i}\in P\}}\left\|\rho_{i}\right\|^{2}\cdot\mathds{1}(X_{i}\in C_{j})-n_{C_{j}}\left\|\bar{\rho}_{C_{j}}\right\|^{2}.$$
$$(38)$$
$$(39)^{\frac{1}{2}}$$

Therefore, the least-squares criterion CART (38) is equivalently written as

j=1,2 X {i : Xi∈Cj } ρi − ρ¯Cj  2= X j=1,2   X {i:Xi∈P } ∥ρi∥ 2· 1(Xi ∈ Cj ) − nCj ρ¯Cj  2  X  , =X j=1,2   X {i:Xi∈P } ∥ρi∥ 2· 1(Xi ∈ Cj )   − nC1 ∥ρ¯C1 ∥ 2 + nC2 ∥ρ¯C2 ∥ 2, {i:Xi∈P } ∥ρi∥ 2 − nC1 ∥ρ¯C1 ∥ 2 + nC2 ∥ρ¯C2 ∥ 2. =X
The first term does not depend on the choice of split, and therefore the split that minimizes (38) is equivalent to the split that maximizes (39). ■
B.1.2. SPLITS VIA CART ON PSEUDO-OUTCOMES
The following result is a generalization to the claim made in Section 3.4 that a CART split on pseudo-outcomes ρ FPT
i will produce a split that maximizes the ∆e FPT-criterion, and is sufficiently general to cover gradient-based pseudo-outcomes ρ grad iand the corresponding ∆e grad-criterion.

Lemma B.1. *Suppose we can write*

$$\tilde{\theta}_{C_{j}}=a+\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\rho_{i},\qquad\rho_{i}=-B\psi_{\theta_{P},\tilde{\varepsilon}_{P}}(O_{i}),\tag{40}$$

where a and B *denote appropriately sized vectors and matrices whose values do not depend on the candidate child node* Cj . Under Assumptions *A.1, the split* (C1, C2) *that maximizes*

$$\widetilde{\Delta}(C_{1},C_{2})=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\widetilde{\theta}_{C_{1}}-\widetilde{\theta}_{C_{2}}\right\|^{2},$$

is exactly the split chosen by CART for vector-valued responses ρi *fit over covariates* Xi ∈ P. Proof of Lemma **B.1.** The scores ψθ,ν(Oi) satisfy subgradient conditions by Assumption 6, and therefore the parent solutions (
ˆθP , νˆP ) satisfy the first-order conditions

$$\sum_{\{i:X_{i}\in P\}}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i})=\mathbf{0}.$$

Hence,

0 =X {i:Xi∈P } ψθˆP ,νˆP (Oi) = X {i:Xi∈C1} ψθˆP ,νˆP (Oi) + X {i:Xi∈C2} ψθˆP ,νˆP (Oi), = −B  X {i:Xi∈C1} ψθˆP ,νˆP (Oi) + X {i:Xi∈C2} ψθˆP ,νˆP (Oi)   , =X {i:Xi∈C1} ρi +X {i:Xi∈C2} ρi.
Each sum in the previous expression is equivalently written as Pρi = nCj(
˜θCj − a). Hence,

$$\begin{array}{c c c}{{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{0=\sum_{\{i:X_{i}\in C_{1}\}}\rho_{i}+\sum_{\{i:X_{i}\in C_{2}\}}\rho_{i},}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{=n_{C_{1}}(\tilde{\theta}_{C_{1}}-a)+n_{C_{2}}(\tilde{\theta}_{C_{2}}-a),}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{a=\frac{n_{C_{1}}}{n_{P}}\tilde{\theta}_{C_{1}}+\frac{n_{C_{2}}}{n_{P}}\tilde{\theta}_{C_{2}}.}}\end{array}$$

Writing ρ¯Cj
:=1 nCj P{i:Xi∈Cj }
ρi, one has:

$$\begin{array}{r l}{{\bar{\rho}_{C_{1}}=\bar{\theta}_{C_{1}}-a,}}\\ {{}}&{{=\bar{\theta}_{C_{1}}-\frac{n_{C_{1}}}{n_{P}}\bar{\theta}_{C_{1}}-\frac{n_{C_{2}}}{n_{P}}\bar{\theta}_{C_{2}},}}\\ {{}}&{{=\frac{n_{C_{2}}}{n_{P}}\left(\bar{\theta}_{C_{1}}-\bar{\theta}_{C_{2}}\right),}}\end{array}$$
and
$$\frac{n_{C_{1}}}{n_{P}}\left\|\bar{\rho}_{C_{1}}\right\|^{2}=\frac{n_{C_{1}}n_{C_{2}}^{2}}{n_{P}^{3}}\left\|\tilde{\theta}_{C_{1}}-\tilde{\theta}_{C_{2}}\right\|^{2}.$$  Next to $C_{2}$, one has the symmetric result:
Applying analogous arguments with respect to C2, one has the symmetric result:

$$\frac{n_{C_{2}}}{n_{P}}\left\|\bar{\rho}_{C_{2}}\right\|^{2}=\frac{n_{C_{2}}n_{C_{1}}^{2}}{n_{P}^{3}}\left\|\bar{\theta}_{C_{1}}-\bar{\theta}_{C_{2}}\right\|^{2}.$$

Therefore,

$$\frac{1}{n_{P}}\left(n_{C_{1}}\left\|\tilde{\rho}_{C_{1}}\right\|^{2}+n_{C_{2}}\left\|\tilde{\rho}_{C_{2}}\right\|^{2}\right)=\frac{n_{C_{1}}n_{C_{2}}^{2}}{n_{P}^{3}}\left\|\tilde{\theta}_{C_{1}}-\tilde{\theta}_{C_{2}}\right\|^{2}+\frac{n_{C_{2}}n_{C_{1}}^{2}}{n_{P}^{3}}\left\|\tilde{\theta}_{C_{1}}-\tilde{\theta}_{C_{2}}\right\|^{2},$$ $$=\frac{n_{C_{1}}n_{C_{1}}}{n_{P}^{2}}\left\|\tilde{\theta}_{C_{1}}-\tilde{\theta}_{C_{2}}\right\|^{2},$$ $$=\tilde{\Delta}(C_{1},C_{2}).$$

Based on the arguments in Appendix B.1.1, a split (C1, C2) maximizes nC1
∥ρ¯C1
∥
2 +nC2
∥ρ¯C2
∥
2if and only if it is a CART
split performed on the ρi over P. That is, ∆( e C1, C2) is precisely maximized by a single CART split on ρi = −BψθˆP ,νˆP
(Oi)
fit over covariates Xi ∈ P, as desired. ■
B.1.3. SCALE INVARIANCE OF CART SPLITS
Lemma B.2 (Argmax equivalence of FPT criteria). *The optimal split identified by CART on pseudo-outcomes* ρ FPT
iof the form (23) does not depend on the scale factor η*, for any* η ̸= 0.

Proof of Lemma **B.2.** Denote by ρ
(η)
i FPT pseudo-outcomes based on an arbitrary scale factor η ̸= 0 of the form (23):

$$\rho_{i}^{(\eta)}:=-\eta\xi^{\top}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i}),$$
$$(41)^{\frac{1}{2}}$$
(Oi), (41)
and let ψCj denote the child-leaf average score evaluated at the parent solution (ˆθP , νˆP ):

$$\overline{{{\psi}}}_{C_{j}}:=\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O_{i}),$$

such that the corresponding child-leaf pseudo-outcome averages ρ¯
(η)
 $\mathbf{)}:=\frac{1}{nC_j}\sum_{\{i:X_i\in C_j\}}\rho_i^{(\eta)}$ are equiv. 
iare equivalently written as
$$\bar{\rho}_{C_{j}}^{(\eta)}=-\eta\xi^{\top}\,\overline{{{\psi}}}_{C_{j}}$$
Let ∆e FPT
η(C1, C2) denote the FPT criterion of the form (25) based on pseudo-outcomes (41):

$$\widetilde{\Delta}_{\eta}^{\mathbb{P}\mathbb{T}}(C_{1},C_{2})=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\bar{\rho}_{C_{1}}^{(\eta)}-\bar{\rho}_{C_{j}}^{(\eta)}\right\|^{2}=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\eta\xi^{\top}(\overline{{{\psi}}}_{C_{1}}-\overline{{{\psi}}}_{C_{2}})\right\|^{2}.$$
One has:
$$\left\|\eta\xi^{\top}(\overline{{{\psi}}}_{C_{1}}-\overline{{{\psi}}}_{C_{2}})\right\|^{2}=\eta^{2}\left\|\xi^{\top}(\overline{{{\psi}}}_{C_{1}}-\overline{{{\psi}}}_{C_{2}})\right\|^{2},$$

17 and hence the ∆e FPT
η-criteria obey the scaling relation:

$$\widetilde{\Delta}_{\eta}^{\mathbb{P}\mathbb{T}}(C_{1},C_{2})=\eta^{2}\cdot\widetilde{\Delta}_{1}^{\mathbb{P}\mathbb{P}\mathbb{T}}(C_{1},C_{2}),$$
$\eqref{eq:walpha}$
1(C1, C2), (42)
where ∆e FPT
1 denotes the FPT criterion induced by pseudo-outcomes ρ
(1)
ibased on unit scale factor η = 1. The relation (42)
implies that any nonzero split-independent rescaling ρ
(η)
i = ηρ
(1)
i will induce a splitting criterion ∆e FPT
η(C1, C2) with the same maximizer as ∆e FPT
1(C1, C2):

$$\operatorname*{arg\,max}_{(C_{1},C_{2})}\,\left\{\widetilde{\Delta}_{\eta}^{\mathbb{P}^{\top}\mathbb{T}}(C_{1},C_{2})\right\}=\operatorname*{arg\,max}_{(C_{1},C_{2})}\,\left\{\eta^{2}\cdot\widetilde{\Delta}_{1}^{\mathbb{P}^{\top}\mathbb{T}}(C_{1},C_{2})\right\}=\operatorname*{arg\,max}_{(C_{1},C_{2})}\,\left\{\widetilde{\Delta}_{1}^{\mathbb{P}^{\top}\mathbb{T}}(C_{1},C_{2})\right\}.$$

Intuitively, a CART split is chosen by ranking the criterion values among the candidate splits and selecting the maximizing split (C1, C2). Therefore, the FPT splitting mechanism is unaffected by the scale factor η used to specify fixed-point pseudo-outcomes (23). The absolute scale of the ∆e FPT-criterion does not matter when searching for the optimal split, and only the criterion rankings across the candidate splits determine the final partition. ■

## B.2. Proofs For Section 4

Notation and definitions.

- Let oP (*a, b, c*) := oP (max{*a, b, c*}), with an analogous abbreviation for OP (·).

- For a fixed parent node P, denote by xP the center of mass of the Xi ∈ P, and let r := sup{i:Xi∈P } ∥Xi − xP ∥ denote the radius of the parent P. Throughout, we consider an asymptotic regime where nCj → ∞ and r → 0, corresponding to leaves over X of vanishing radius. Further, r and nCjare related under the conditions of GRF Proposition 1, namely, r
−2 ≪ nCjand hence nCjr 2 → ∞ and 1/
√nCj = o(r).

- Let θ
∗Cj denote the true parameter expectation over the child node:

$$\theta_{C_{j}}^{*}:=\mathbb{E}[\theta^{*}(X)\mid X\in C_{j}],\qquad j=1,2,$$
∗(X) | X ∈ Cj ], j = 1, 2, (43)
and let ˜θ
∗
Cj
(xP ) denote an oracle version of the gradient-based leaf statistic:

$$\tilde{\theta}_{C_{j}}^{*}(x_{P}):=\theta^{*}(x_{P})-\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\xi^{\top}V(x_{P})^{-1}\psi_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(O_{i}),$$

where V (x) is the underlying local Jacobian in Assumption 2. Equivalently, in terms of the oracle pseudooutcome/influence function ρ
∗
i(·) defined in Regularity Condition 1,

$$\tilde{\theta}_{C_{j}}^{*}(x_{P}):=\theta^{*}(x_{P})+\frac{1}{n_{C_{j}}}\sum_{\{i:X_{i}\in C_{j}\}}\rho_{i}^{*}(x_{P}).$$

The following are technical lemmas used for the proof of Proposition 4.1. Lemma B.3. Suppose Assumptions A.1 and Specifications A.2 *hold. Then,*

$$\Delta(C_{1},C_{2})=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\theta_{C_{1}}^{*}-\theta_{C_{2}}^{*}\right\|^{2}+o_{P}\left(r^{2},\frac{1}{n_{C_{1}}},\;\frac{1}{n_{C_{2}}}\right).$$

Proof of Lemma **B.3.** Write the difference ˆθCj − θ
∗Cj as

$$\hat{\theta}_{C_{j}}-\theta_{C_{j}}^{*}=\underbrace{\left(\hat{\theta}_{C_{j}}-\hat{\theta}_{C_{j}}^{*}(x_{P})\right)}_{\hat{Y}_{1}}+\underbrace{\left(\hat{\theta}_{C_{j}}^{*}(x_{P})-\mathbb{E}[\hat{\theta}_{C_{j}}^{*}(x_{P})\mid X\in C_{j}]\right)}_{\hat{Y}_{2}}+\underbrace{\left(\mathbb{E}[\hat{\theta}_{C_{j}}^{*}(x_{P})\mid X\in C_{j}]-\theta_{C_{j}}^{*}\right)}_{\hat{Y}_{3}}.$$
$$(43)$$

Under standard LLN arguments, the second term satisfies T2 = OP (1/
√nCj
), and in an asymptotic regime with r
−2 ≪ nCj one has T2 = oP (r). Meanwhile, the first and third terms appear in the proofs of Propositions 2 and 1 of Athey et al. (2019), respectively, and satisfy T1 = oP (r, 1/
√nCj) and T3 = O(r 2) =⇒ T3 = o(r). It follows

$$\hat{\theta}_{C_{j}}-\theta_{C_{j}}^{*}=o_{P}\left(r,\,1/\sqrt{n_{C_{j}}}\right),$$

and in particular

$$\hat{\theta}_{C_{1}}-\hat{\theta}_{C_{2}}=\theta_{C_{1}}^{*}-\theta_{C_{2}}^{*}+o_{P}\left(r,\;\frac{1}{\sqrt{n_{C_{1}}}},\;\frac{1}{\sqrt{n_{C_{2}}}}\right).$$

Write A = θ
∗C1 − θ
∗C2 and let E be any term satisfying E = oP (r, 1/
√nC1, 1/
√nC2) such that ∆(C1, C2) is equivalently written ∆(C1, C2) = (nC1 nC2 /n2P
) · ∥A + E∥
2. Consider the difference

$$\begin{split}\Delta(C_{1},C_{2})-\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\theta_{C_{1}}^{*}-\theta_{C_{2}}^{*}\right\|^{2}&=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left(\left\|A+E\right\|^{2}-\left\|A\right\|^{2}\right),\\ &=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left(2\langle A,E\rangle+\left\|E\right\|^{2}\right).\end{split}$$

Under Specification 2 there exists a fixed proportion ω > 0 such that nC1, nC2 ≥ ωnP , and hence nC1 nC2/n2P ≥
ω(1 − ω) and also nC1 nC2 /n2P ≤ 1/4 for all nC1 + nC2 = nP . Therefore nC1 nC2 /n2P = O(1). Meanwhile, ∥E∥
2 =
oP (r 2, 1/nC1, 1/nC2) is true by definition of E, and under our assumptions one may follow the arguments of Athey et al.

(2019) Proposition 1 to see that A = θ
∗
C1 − θ
∗
C2 = O(r). Thus,

$$\langle A,E\rangle={\mathcal{O}}(r)\cdot o_{P}\left(r,\ {\frac{1}{\sqrt{n_{C_{1}}}}},\ {\frac{1}{\sqrt{n_{C_{2}}}}}\right)=o_{P}\left(r^{2},\ {\frac{r}{\sqrt{n_{C_{1}}}}},\ {\frac{r}{\sqrt{n_{C_{2}}}}}\right),$$

and therefore

$$\Delta(C_{1},C_{2})-\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\left\|\theta_{C_{1}}^{*}-\theta_{C_{2}}^{*}\right\|^{2}=o_{P}\left(r^{2},\;\frac{1}{n_{C_{1}}},\;\frac{1}{n_{C_{2}}}\right),$$

as desired. ■
Lemma B.4. Suppose the conditions of Lemma B.3 hold, and assume moreover Neyman orthogonal moment conditions such that the underlying Jacobian V (x) defined in Assumption 2 with block form (37)*. Then,*

$$\widetilde{\Delta}_{\eta}^{\mp\mp\mp}(C_{1},C_{2})=\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\eta^{2}\left\|V_{\theta\theta}(x_{P})(\theta_{C_{1}}^{*}-\theta_{C_{2}}^{*})\right\|^{2}+o_{P}\left(r^{2},\,\frac{1}{n_{C_{1}}},\,\frac{1}{n_{C_{2}}}\right),$$

where ∆FPT
η defined in Lemma B.2 *denotes the* FPT *criterion with arbitrary scale factor* η ̸= 0.

Proof of Lemma **B.4.** From the proof of Lemma B.2 one finds that ∆FPT
η(C1, C2) is equivalently written

$$\Delta_{\eta}^{\mp\mp\mp}(C_{1},C_{2})\coloneqq\frac{n_{C_{1}}n_{C_{2}}}{n_{P}^{2}}\eta^{2}\left\|\xi^{\top}\big(\overline{{{\psi}}}_{C_{1}}-\overline{{{\psi}}}_{C_{2}}\big)\right\|^{2},\qquad\overline{{{\psi}}}_{C_{j}}\coloneqq\frac{1}{n_{C_{1}}}\sum_{\{i:X_{i}\in C_{j}\}}\psi_{\theta_{P},\delta_{P}}(O_{i}).$$

Under standard LLN arguments the average scores ψCj satisfy

$${\overline{{\psi}}}_{C_{j}}=\mathbb{E}[\psi_{\hat{\theta}_{P},\hat{\nu}_{P}}(O)\mid X\in C_{j}]+O_{P}(1/{\sqrt{n_{C_{j}}}}).$$
√nCj). (44)
One applies iterated expectation to see

$\mathbb{E}[\psi_{\hat{\theta}_{P},\hat{\iota}_{P}}(O)\mid X\in C_{j}]=\mathbb{E}\left[\mathbb{E}\left[\psi_{\hat{\theta}_{P},\hat{\rho}_{P}}(O)\mid X\right]\mid X\in C_{j}\right]=\mathbb{E}[M_{\hat{\theta}_{P},\hat{\iota}_{P}}(X)\mid X\in C_{j}],$

and hence
$$\overline{{{\psi}}}_{C_{j}}=\mathbb{E}[M_{\hat{\theta}_{P},\hat{\nu}_{P}}(X)\mid X\in C_{j}]+O_{P}(1/\sqrt{n_{C_{j}}}).$$
). (45)
$$(444)$$
$$(45)^{\frac{1}{2}}$$
19 Expansion of MθˆP ,νˆP
(X). Under Assumption 2 one considers the Taylor expansion of MθˆP ,νˆP
(X) about (*θ, ν*) =
(θ
∗(xP ), ν∗(xP )):
MθˆP ,νˆP
(X) = Mθ
∗(xP ),ν∗(xP )(X)

$$+\left[\nabla_{(\theta,\nu)}M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)\right]\left[\begin{matrix}\hat{\theta}_{P}-\theta^{*}(x_{P})\\ \hat{\nu}_{P}-\nu^{*}(x_{P})\end{matrix}\right]+O_{P}\left(\left\|\begin{matrix}\hat{\theta}_{P}-\theta^{*}(x_{P})\\ \hat{\nu}_{P}-\nu^{*}(x_{P})\end{matrix}\right\|^{2}\right).$$

The consistency of the parent solutions (ˆθP , νˆP ) for (θ
∗(xP ), ν∗(xP )) is established by Athey et al. (2019), and in particular
(
ˆθP , νˆP ) − (θ
∗(xP ), ν∗(xP )) = OP (r, 1/
√nP ). The asymptotic regime r
−2 ≪ nP implies 1/
√nP = o(r) and therefore the higher order quadratic term is equivalently expressed:

$$M_{\theta_{P},\hat{\mu}_{P}}(X)=M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)+\left[\nabla_{(\theta,\nu)}M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)\right]\left[\begin{matrix}\hat{\theta}_{P}-\theta^{*}(x_{P})\\ \hat{\rho}_{P}-\nu^{*}(x_{P})\end{matrix}\right]+O_{P}(r^{2}),$$

and therefore

$$\mathbb{E}\left[M_{\theta_{P},\rho_{P}}(X)\mid X\in C_{j}\right]=\mathbb{E}\left[M_{\theta^{*}(x_{P}),\rho^{*}(x_{P})}(X)\mid X\in C_{j}\right]$$ $$+\mathbb{E}\left[\nabla_{(\theta,\nu)}M_{\theta^{*}(x_{P}),\rho^{*}(x_{P})}(X)\mid X\in C_{j}\right]\begin{bmatrix}\hat{\theta}_{P}-\theta^{*}(x_{P})\\ \hat{\rho}_{P}-\nu^{*}(x_{P})\end{bmatrix}+O_{P}(r^{2}).$$

One has ∇(θ,ν)Mθ
∗(xP ),ν∗(xP )(X) = V (xP )+OP (r) because Mθ,ν(x) is Lipschitz in x, and the expansion in the previous display becomes:

$$\mathbb{E}\left[M_{\theta_{P},\varepsilon_{P}}(X)\mid X\in C_{j}\right]=\mathbb{E}\left[M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)\mid X\in C_{j}\right]+V(x_{P})\left[\begin{matrix}\theta_{P}-\theta^{*}(x_{P})\\ \theta_{P}-\nu^{*}(x_{P})\end{matrix}\right]+O_{P}(r^{2}).\tag{46}$$

Expansion of Mθ
∗(xP ),ν∗(xP )(X). Following similar arguments, the term Mθ
∗(xP ),ν∗(xP )(X) is expanded about
(*θ, ν*) = (θ
∗(X), ν∗(X)) as:

$$M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)=M_{\theta^{*}(X),\nu^{*}(X)}(X)+V(X)\begin{bmatrix}\theta^{*}(x_{P})-\theta^{*}(X)\\ \nu^{*}(x_{P})-\nu^{*}(X)\end{bmatrix}+O_{P}(r^{2}),$$ $$=V(X)\begin{bmatrix}\theta^{*}(x_{P})-\theta^{*}(X)\\ \nu^{*}(x_{P})-\nu^{*}(X)\end{bmatrix}+O_{P}(r^{2}),$$

where Mθ
∗(X),ν∗(X)(X) = 0 holds because (θ
∗(X), ν∗(X)) are defined as satisfying the GRF moment conditions (1) local to X. One takes the conditional expectation of the previous display:

$$\mathbb{E}\left[M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)\mid X\in C_{j}\right]=\mathbb{E}\left[V(X)\left[\begin{matrix}\theta^{*}(x_{P})-\theta^{*}(X)\\ \nu^{*}(x_{P})-\nu^{*}(X)\end{matrix}\right]\mid X\in C_{j}\right]+O_{P}(r^{2}).$$

Whenever X ∈ Cj one has ∥X − xP ∥ = O(r), and the same Lipschitz arguments can be applied to see V (X) = V (xP ) + OP (r) conditional on X ∈ Cj , and the previous display simplifies:

$$\mathbb{E}\left[M_{\theta^{*}(x_{P}),\nu^{*}(x_{P})}(X)\mid X\in C_{j}\right]=V(x_{P})\left[\begin{matrix}\theta^{*}(x_{P})-\theta^{*}_{C_{j}}\\ \nu^{*}(x_{P})-\nu^{*}_{C_{j}}\end{matrix}\right]+O_{P}(r^{2}),\tag{47}$$

where θ
∗Cj
:= E[θ
∗(X) | X ∈ Cj ] and ν
∗Cj
:= E[ν
∗(X) | X ∈ Cj ]. Substitute (47) into the conditional expectation (46):

$$\mathbb{E}\left[M_{\theta_{P},\hat{\nu}_{P}}(X)\mid X\in C_{j}\right]=V(x_{P})\begin{bmatrix}\theta^{*}(x_{P})-\theta^{*}_{C_{j}}\\ \nu^{*}(x_{P})-\nu^{*}_{C_{j}}\end{bmatrix}+V(x_{P})\begin{bmatrix}\hat{\theta}_{P}-\theta^{*}(x_{P})\\ \hat{\nu}_{P}-\nu^{*}(x_{P})\end{bmatrix}+O_{P}(r^{2}),$$ $$=V(x_{P})\begin{bmatrix}\hat{\theta}_{P}-\theta^{*}_{C_{j}}\\ \hat{\nu}_{P}-\nu^{*}_{C_{j}}\end{bmatrix}+O_{P}(r^{2}).$$