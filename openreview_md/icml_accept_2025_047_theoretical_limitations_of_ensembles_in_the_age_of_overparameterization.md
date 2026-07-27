# 

Niclas Dern* 1 John P. Cunningham 2 **Geoff Pleiss** 3 4

## Abstract 1. Introduction

Classic ensembles generalize better than any single component model. In contrast, recent empirical studies find that modern ensembles of (overparameterized) neural networks may not provide any inherent generalization advantage over single but larger neural networks. This paper clarifies how modern overparameterized ensembles differ from their classic underparameterized counterparts, using ensembles of random feature (RF) regressors as a basis for developing theory. In contrast to the underparameterized regime, where ensembling typically induces regularization and increases generalization, we prove with minimal assumptions that infinite ensembles of overparameterized RF regressors become pointwise equivalent to (single) infinite-width RF regressors, and finite width ensembles rapidly converge to single models with the same parameter budget. These results, which are exact for ridgeless models and approximate for small ridge penalties, imply that overparameterized ensembles and single large models exhibit nearly identical generalization. We further characterize the predictive variance amongst ensemble members, demonstrating that it quantifies the expected effects of increasing capacity rather than capturing any conventional notion of uncertainty. Our results challenge common assumptions about the advantages of ensembles in overparameterized settings, prompting a reconsideration of how well intuitions from underparameterized ensembles transfer to deep ensembles and the overparameterized regime.

Historically, most machine learning ensembles aggregated component models that are simple by today's standards (e.g. Hansen & Salamon, 1990; Opitz & Maclin, 1999; Dietterich, 2000). Common techniques like bagging (Breiman, 1996), feature selection (Breiman, 2001), random projections (Kaban´ , 2014; Thanei et al., 2017), and boosting (Freund, 1995; Chen & Guestrin, 2016) were developed and analyzed assuming decision trees, least-squares regressors, and other *underparameterized* component models incapable of achieving zero training error.

Researchers and practitioners have now turned to ensembles of *overparameterized* models, such as neural networks, which have capacity to memorize entire training datasets.

Motivated by heuristics from classic ensembles (Mentch & Hooker, 2016), some have argued that ensembles provide robustness to dataset shift (Lee et al., 2015; Fort et al., 2019) and that the predictive variance amongst component models in these so-called *deep ensembles* is a notion of uncertainty that can be used on downstream decision-making tasks (Lakshminarayanan et al., 2017; Gal et al., 2017; Gustafsson et al., 2020; Ovadia et al., 2019; Yu et al., 2020). While few theoretical works analyze modern overparameterized ensembles, recent empirical evidence suggests that intuitions from their underparameterized counterparts do not hold in this new regime. For example, classic methods to increase diversity amongst component models, such as bagging, can be harmful for deep ensembles (Nixon et al., 2020; Jeffares et al., 2024; Abe et al., 2022a; 2024; Webb et al., 2021) despite being nearly universally beneficial for underparameterized ensembles. Moreover, while established underparameterized ensembling techniques offer well-founded quantifications of uncertainty (e.g. Mentch & Hooker, 2016; Wager et al., 2014), several recent studies question the reliability of the uncertainty estimates from deep ensembles (Abe et al., 2022b; Theisen et al., 2024; Chen et al., 2024). To address this divergence and verify recent empirical findings, we develop a theoretical characterization of ensembles in the overparameterized regime, with the goal of contrasting against (traditional) underparameterized ensembles. We answer the following questions:
1 1. Do ensembles of overparameterized models provide generalization or robustness benefits over a single (very large) model trained on the same data? Does the capacity of the component models affect this difference?

2. What does the predictive variance of overparameterized ensembles measure, and does it relate to classic frequentist or Bayesian notions of uncertainty?

To answer these questions, we analyze ensembles of overparameterized random feature (RF) linear regressors, a theoretically-tractable approximation of neural networks. Unlike prior work on RF models, our analysis makes very few assumptions about the distribution of random features, which—as we will show—is crucial for highlighting the differences between ensemble variance versus more established notions of uncertainty. Our analysis focuses on the practically relevant regime where RF models are trained with little to no regularization. We verify and contextualize our theory with experiments on RF and neural networks ensembles.

## 1.1. Related Work

Deep ensembles. A primary motivation of this paper is to understand recent empirical findings about uncertainty quantification afforded by deep ensembles (Lakshminarayanan et al., 2017). Historically, variance amongst deep ensemble members has been a proxy for *epistemic uncertainty* (e.g. Kendall & Gal, 2017; Gustafsson et al., 2020), i.e., the uncertainty that can be reduced by collecting more data. This view reflects a classical intuition of ensembles: ignoring effects of overparameterization and inductive bias, all ensemble members should converge to the same prediction in the infinite data limit, and thus differing predictions suggest a region of the input space with insufficient data. However, recent empirical findings challenge this interpretation of ensemble variance (Abe et al., 2022b; Theisen et al., 2024). Most relevant to our work, Abe et al. (2022b) demonstrate a strong correlation between ensemble variance and the expected improvement that results from increasing model capacity.

Specifically, across numerous architectures and datasets, they demonstrate a strong point-wise correlation between the predictions of an ensemble (e.g., 4 ResNet-18s) and a single larger model (e.g., a WideResNet-18 with 4× the width)
on both in-distribution and out-of-distribution data. The authors conclude that ensemble variance is more reflective of sensitivity to model capacity rather than data availability, a finding with significant implications for decision-making and robustness. We theoretically verify these findings in ensembles of overparameterized random feature models.

Random feature models. The connection between infinitely wide neural networks and kernel methods, particularly Gaussian processes, was pioneered by Neal & Neal
(1996) and Williams (1996). Building on these ideas, random feature (RF) models were later introduced as a scalable approximation to kernel machines (Rahimi & Recht, 2007; 2008a;b). RF regressors have seen growing theoretical interest as simplified models of neural networks (e.g. Belkin et al., 2018; 2019; Jacot et al., 2018; Bartlett et al., 2020; Mei & Montanari, 2022; Simon et al., 2024). Random feature models can be interpreted as neural networks where only the last layer is trained (e.g. Rudi & Rosasco, 2017; Belkin et al., 2019) or as first-order Taylor approximations of neural networks (e.g. Jacot et al., 2018). Underparameterized random feature models and ensembles. In this paragraph, we restrict our discussion to analyses of (ensembles of) underparameterized RF regressors, where the number of random features (i.e., the width) is assumed to be far fewer than the number of data points. In the fixed design setting, infinite ensembles of unregularized RF regressors achieve the same generalization error as ridge regression on the original (unprojected) inputs (Kaban´ , 2014; Thanei et al., 2017; Bach, 2024b). We provide theoretical analysis in Appx. E that further demonstrates ridge-like behaviour of underpameterized RF ensembles. Overparameterized random feature models. Recent works on RF models have focused on the overparameterized regime, often using high-dimensional asymptotics to characterize generalization error (Adlam & Pennington, 2020; Bach, 2024b; Hastie et al., 2022; Loureiro et al., 2022; Mei
& Montanari, 2022; Ruben et al., 2024). Many works rely on results derived assuming that the the marginal distributions over the random features can be replaced by momentmatched Gaussians. While such approximations are wellfounded for asymptotic results (e.g. Goldt et al., 2022; Hu & Lu, 2022; Montanari & Saeed, 2022; Tao, 2012), we argue that they may be harmful specifically for an analysis which aims to characterize the uncertainty properties of ensemble variance. Assuming Gaussianity results in an ensemble variance that is proportional to the predictive variance of Gaussian process regression, often held as a gold standard for uncertainty quantification (Rasmussen & Williams, 2006; Lee et al., 2018; 2020; Ovadia et al., 2019). In contrast, our non-Gaussian analysis yields a characterization of ensemble variance that differs from this conventional notion of uncertainty, closely matching recent empirical studies of ensemble variance (Abe et al., 2022b; Theisen et al., 2024). The benefits of overparameterization and ensembling for out-of-distribution generalization in random feature models have been analyzed by Hao et al. (2024), who provide lower bounds on OOD risk improvements when increasing capacity or using ensembles. Their work focuses on nonasymptotic guarantees under specific distributional shifts, while ours examines the equivalence of ensembles and sin-

4 2 0 2 4 1.0 0.5 0.0 0.5 1.0 1.5 RF Models Kernel Model Training Data 4 2 0 2 4 0.5 0.0 0.5 1.0 1.5 Ensemble Model Kernel Model Training Data
gle large models under minimal assumptions. Concurrent work by Ruben et al. (2024) also finds RF ensembles offer little advantage over larger single models, though their analysis uses optimal ridge tuning and Gaussian universality assumptions. Most related to our work is Jacot et al. (2020), who analyze the pointwise expectation and variance of ridgeregularized RF models with Gaussian process (GP) features, leveraging Gaussianity to simplify their analysis. We go beyond this prior work by significantly weakening the assumptions on the distribution of random features, enabling us to characterize differences between ensembles versus Gaussian models with respect to uncertainty and robustness properties. Moreover, we provide a finite-sample analysis as well as a characterization of the transition from the ridgeless to ridge-regularized regimes, which—to the best of our knowledge—are novel results for overparameterized RF ensembles.

## 1.2. Contributions

We consider ensembles of *overparameterized* RF regressors in both the ridgeless and small ridge regimes. Unlike prior work, we make minimal assumptions about the distribution of the random features and so our results are not restricted to high-dimensional asymptotics where Gaussian universality might typically apply. Our results thus distinguish differences between RF ensembles and more traditional uncertainty-aware models like Gaussian processes. Concretely, we make the following contributions:
To answer Question 1: we show that the average ridgeless RF regressor is pointwise equivalent to its corresponding ridgeless kernel regressor (Theorem 3.2), implying that an infinite ensemble of overparameterized RF models is *exactly* equivalent to a single infinite-width RF model (cf. Fig. 1). We further show that this equivalence approximately holds in the small ridge regime (Theorem 3.5). Moreover, we extend these results to a finite parameter budget, showing that the functional difference between the parameters of a larger single model and a finite ensemble, each with the same total number of parameters, is small with high probability (see Sec. 3.2). We validate these theoretical results with supporting experiments on RF and neural network ensembles, using synthetic data and the California Housing dataset (Kelley Pace & Barry, 1997) with various activation functions (detailed in Appx. A.1 and Appx. B). To answer Question 2: we show that the predictive variance in an overparameterized ensemble generally does not have a frequentist or Bayesian interpretation, unlike uncertainty quantifications obtained from Gaussian processes. Instead, we find that the variance measures the expected squared difference between the predictions from a (finite-width) RF regressor and its corresponding kernel regressor (i.e., the infinite-width model) (see Sec. 3.3). Crucially, this finding relies on our non-Gaussian analysis of RF models. Altogether, these results support recent empirical findings that deep ensembles offer few generalization and uncertainty quantification benefits over larger single models (Abe et al., 2022b; Theisen et al., 2024). Our theory and experiments demonstrate that these phenomena are not specific to neural networks or Gaussian models but are more general properties of ensembles in the overparameterized regime.

## 2. Setup

We work in a regression setting. The training dataset D = {(xi, yi)}
N
i=1 ∈ (X × R)
N is a *fixed* set of size N. The vector y ∈ R
N is the concatenation of all training responses.

We consider *RF models* adhering to the form hW(x) =
√
1 D
PD
i=1 ϕ(ωi, x)θi, where θi are learned parameters, W = {ωi}
D
i=1 ∈ Ω
D are i.i.d. draws from some distribution π(·), and ϕ : Ω*×X →* R is a *feature extraction function*. In the case of a ReLU-based RF model with p-dimensional inputs, we have X = Ω = R
pand ϕ(ωi, x) = max(0, ω⊤
i x).

Though RF models cannot fully explain the behaviour of neural networks (e.g. Ghorbani et al., 2019; Li et al., 2021; Pleiss & Cunningham, 2021), they can be a useful proxy for understanding the effects of overparameterization and capacity on generalization (e.g. Belkin et al., 2019; Adlam & Pennington, 2020; Mallinar et al., 2022).

Notation. For any x, x′ ∈ X , we denote the second moment of the feature extraction function ϕ(ω, ·) as k(*x, x*′) =
Eω[ϕ(ω, x)ϕ(*ω, x*′)], which is a positive definite kernel function. We use the matrix K := [k(xi, xj )]ij ∈ R
N×N
for the kernel function applied to all training data pairs and the matrix ΦW := [ϕ(ωj , xi)]ij ∈ R
N×D for the feature extraction function applied to all data/feature combinations.

In this notation, [·]ij refers to the entry in the i-th row and j-th column; if one index is omitted (e.g., [v]j ), it refers to the j-th element of a row- or column-vector, depending on the context. We drop the subscript W when the set of random features is clear from context. Furthermore, we assume that K is invertible. Throughout our analysis, it will be useful to consider the
"whitened" features W = R−⊤Φ ∈ R
N×D where R⊤R =
K is the Cholesky decomposition of the kernel matrix K. When considering a test point x
∗ ∈ X (or equivalently a set of test points), we extend the K, R, Φ, W notation by

$$\begin{bmatrix}K&[k(x_{i},x^{*})]_{i}\\ [k(x^{*},x_{j})]_{j}&k(x^{*},x^{*})\end{bmatrix}=\begin{bmatrix}R&c\\ 0&r_{\perp}\end{bmatrix}^{\top}\begin{bmatrix}R&c\\ 0&r_{\perp}\end{bmatrix},$$ $$\begin{bmatrix}W\\ w_{\perp}^{\top}\end{bmatrix}=\begin{bmatrix}R&c\\ 0&r_{\perp}\end{bmatrix}^{-\top}\begin{bmatrix}\Phi\\ [\phi(\omega_{i},x^{*})]_{i}\end{bmatrix}.\tag{1}$$

For fixed training/test points, EW [WW⊤] = D · I,
Ew⊥ [w
⊤
⊥w⊥] = D and EW,w⊥ [w
⊤⊥W⊤] = 0 which can be directly derived from EΦ[ΦΦ⊤] = D · K (and similar properties for ϕ
∗, the vector of feature evaluations at x
∗, i.e.,
[ϕ(ωj , x∗)]j ). Moreover, the columns [wi; w⊥i] of [W; w⊥]
are i.i.d. since they are affine transformations of the i.i.d. columns of Φ.

Overparameterized ridge/ridgeless regressors and ensembles. As our focus is the overparameterized regime, we assume a computational budget of *D > N* features
(W = {ω1, . . . , ωD} ∼ π D) to construct an RF regressor hW(x) = √
1 D
ϕW(x)
⊤θ. We train the regressor parameters θ to minimize the loss ∥ √
1 D
ΦWθ − y∥
22 + λ∥θ∥
22 for some ridge parameter λ ≥ 0. When λ > 0 this optimization problem admits the closed-form solution θ
(RR)
W,λ =
√
1 D
Φ
⊤ W
1D· ΦWΦ
⊤
W + λI−1y. Although the learning problem is underspecified when λ = 0 (i.e. in the ridgeless case), the implicit bias of (stochastic) gradient descent initialized at zero leads to the minimum norm interpolating solution θ
(LN)
W = √
1 D
(Φ)⊤1D· ΦΦ⊤−1y.

We denote the resulting ridge(less) regressors as h
(LN)
W (·) := √
1 D
-ϕ(ωj , ·)j θ
(LN)
W , and h
(RR)
W,λ (·) :=
√
1 D
-ϕ(ωj , ·)j θ
(RR)
W,λ .

We also consider ensembles of M ridge(less) regressors. We assume that each is trained on a different set of i.i.d. *D > N*
random features W1*, . . . ,* WM ∼ π D but trained on the same training set. Thus, the only source of randomness in these ensembles comes from the random selection of features Wi, analogous to the standard training procedure of deep ensembles (Lakshminarayanan et al., 2017). The ensemble prediction is given by the arithmetic average of the individual models h¯W1:M (·) = 1M
PMm=1 h
(LN)
Wm
(·).

Assumptions. A key difference between this paper and prior literature is the set of assumptions about the random feature distribution π(·). Most prior works assume that entries in the extended whitened feature matrix [W; w⊥] are i.i.d. draws from standard normal distribution (e.g. Adlam & Pennington, 2020; Jacot et al., 2020; Mei & Montanari, 2022; Simon et al., 2024) implying that ϕ(ωi, ·) are draws from a Gaussian process with covariance k.

1 While Gaussianity is appropriate in high-dimensional asymptotics, it essentially reduces analysis about the ensemble distribution to a statement about Gaussian processes. A major focus of this work is to differentiate ensembles from Gaussian processes with regards to uncertainty quantification. Even if we were to relax the Gaussian assumption to a sub-Gaussian assumption, (as done by Bartlett et al., 2020; Bach, 2024a), the distribution of random features will still not accurately reflect common neural network features if the entries of [W; w⊥] are assumed to be i.i.d. For instance, consider ReLU features. If X ⊆ R
p with *p < N*, the function max(ω
⊤x, 0) can be fully specified by a p-dimensional random variable. Thus, knowing N evaluations of ω
⊤
j xi allows one to infer ωj , making w⊥ deterministic given W. We instead consider the following less restrictive assumptions on the distribution of random feature functions π(·):
1If the entries of *W, w*⊥ are i.i.d. Gaussian, then the i th feature applied to train/test inputs ([R
⊤wi; c
⊤wi+r⊥w⊥i]) is multivariate Gaussian. This fact holds for any train/test data; thus the i th feature is a GP by definition (e.g. Rasmussen & Williams, 2006, Ch. 2).

0 10 20 30 40 50 Number of features per model 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 h

(LN) 
h

(LN) 
L

1 N = Total Parameters 0 5000 10000 15000 20000 25000 30000 35000 40000 Number of parameters per model 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 h 1 : M
h Wsingl e L

1 N = Total Parameters
Assumption 2.1 (Assumption of subexponentiality).

1. wiw⊥i (where wiis the i th column of W) is subexponential ∀i ∈ {1*, ..., D*} and

$\sum_{i=1}^{n+1-n}$
2. PD
i=1 wiw
⊤
iis a.s. positive definite for any D ≥ N.

The first condition of Assumption 2.1 ensures that the whitened random features do not have excessively "heavy tails", meaning their values are well-concentrated. This is a mild condition, satisfied if the individual feature components wi and w⊥i are sub-Gaussian (but potentially dependent), which is true if the features come from activation functions with bounded derivatives and sub-Gaussian weights. The second condition is equivalent to Φ having almost surely full rank, which is not true for ReLUs and leaky-ReLUs features but which is true for arbitrarily precise approximations thereof.2 Note we make no assumptions about the mean or independence of the entries in a given column of [W; w⊥].

## 3. Main Results 3.1. Equivalence Of Infinite Ensembles And The Infinite-Width Single Models

We at first assume an infinite computational budget and consider the following two limiting predictors, for which we will show pointwise equivalence in predictions:
1. An infinite-width least norm predictor, h
(LN)
∞ , the a.s.

limit of h
(LN)
W as |W| = D → ∞
2. An infinite ensemble of finite-width least norm predictors, h¯
(LN)
∞ , which is the almost sure limit of h¯
(LN) W1:M
as M → ∞, with *N < D <* ∞ remaining constant.

These limiting predictors not only approximate large ensembles and very large single models but also help characterize the variance and generalization error of finite overparameterized ensembles, as discussed in Sec. 3.3.

Define kN (·) : X → R
N as the vector of kernel evaluations with the training data kN (·) = -k(x1, ·) *· · ·* k(xN , ·)⊤.

As D → ∞, the minimum norm interpolating model converges pointwise almost surely to the ridgeless kernel regressor by the Strong Law of Large Numbers:

$$h_{\mathcal{W}}^{(\mathrm{LN})}(\cdot)\stackrel{\mathrm{a.s.}}{\longrightarrow}h_{\infty}^{(\mathrm{LN})}(\cdot),\qquad h_{\infty}^{(\mathrm{LN})}(\cdot):=k_{N}(\cdot)^{\top}K^{-1}y.$$

On the other hand, using W and w⊥ as introduced in Sec. 2 we can rewrite the infinite ensemble prediction h¯
(LN)
∞ (x
∗)
as (for a derivation of this, see Appx. C.1)

$$\begin{array}{l}{{\bar{h}_{\infty}^{(L N)}(x^{*})=h_{\infty}^{(\mathrm{LN})}(x^{*})}}\\ {{\quad+\ \ r_{\perp}\mathbb{E}_{W,w_{\perp}}\left[w\!\perp^{\top}W^{\top}\left(W W^{\top}\right)^{-1}\right]R^{-\top}y}}\end{array}\quad\mathrm{(2)}$$

To prove the pointwise equivalence of the infinite ensemble and infinite-width single model, we need to show that EW,w⊥ [w
⊤⊥W⊤(WW⊤)
−1] term in Eq. (2) is zero. Note that this result trivially holds when the entries of W and w⊥
are i.i.d. and zero mean, as assumed in prior work (e.g. Jacot et al., 2020). In the following lemma, we show that this term is zero even when w⊥ and W are dependent, whichas described in Sec. 2—is a more realistic assumption for neural network features: Lemma 3.1. Under Assumption *2.1, it holds that* EW,w⊥ [w
⊤
⊥W⊤(WW⊤)
−1] = 0.

(Proof: see Appx. C.1.) Combining Lemma 3.1 and Eq. (2)
yields the pointwise equivalence of h¯
(LN)
∞ and h
(LN)
∞ :
Theorem 3.2 (Equivalence of infinite-width single model and infinite ensembles). Under Assumption 2.1, the infinite ensemble of finite-width (but overparameterized) RF
regressors h¯
(LN)
∞ *is pointwise almost surely equivalent to* the (single) infinite-width RF regressor h
(LN)
∞ .

Theorem 3.2 implies that ensembling overparameterized RF models yields *exactly* the same predictions as simply increasing the capacity of a single RF model, regardless of the RF distribution (see Fig. 1 for a visualization). Note that this result significantly generalizes prior characterizations of overparameterized RF models that have relied on Gaussianity assumptions or asymptotic analyses (e.g. Adlam & Pennington, 2020; Jacot et al., 2020), demonstrating that the ensemble/infinite-single model equivalence is a fundamental property of overparameterization. Consequently, we should not expect substantial differences in generalization between large single models and overparameterized ensembles, consistent with recent empirical findings by (Abe et al., 2022b; 2024; Theisen et al., 2024). We emphasize a contrast with the underparameterized regime, where RF ensembles match the generalization error of kernel ridge regression (see Appx. E or Bach, 2024b, Sec. 10.2.2). Width controls the implicit ridge parameter in the underparameterized regime (see Sec. 1.1), whereas width does not affect the ensemble predictor in the overparameterized regime. We confirm this difference in Fig. 2 which shows that RF ensembles are close to the ridgeless kernel regressor when *D > N* but not when *D < N*. The figure also illustrates similar behaviours when comparing deep ensembles versus single large neural networks.

## 3.2. Ensembles Versus Larger Single Models Under A Finite Parameter Budget

For modest parameter budgets (where our asymptotic results are not applicable), we compare whether ensembles are more parameter efficient than larger single models. Specifically, given access to MD random features, we compare ensembles of M models each of which use D of the features (h¯
(LN) W1:M
=1M
PMm=1 h
(LN) Wm
) against a single model that uses all MD features h
(LN)
W∗ (·) (i.e., here |Wm| = D
for all m and |W∗| = MD).

First, we provide a non-asymptotic theorem showing that h¯
(LN)
W1:M
and h
(LN)
W∗ behave similarly, with their difference becoming negligible as the number of features per ensemble member increases (for a formal version, see Appx. C.2). Theorem 3.3 (Non-asymptotic difference between ensembles and single models (informal version)). *Under slightly* stronger assumptions than Assumption 2.1, the L2 difference between a single neural network with MD features and an ensemble of M neural networks each with D *features is,*
with probability 1 − δ*, upper bounded by:*

$$\left\|h_{\mathcal{W}^{*}}^{\mathrm{(LN)}}(\cdot)-\bar{h}_{\mathcal{W}_{1:M}}^{\mathrm{(LN)}}(\cdot)\right\|_{2}^{2}\leq O(\sqrt{\log(1/\delta)})+O(1/D)$$

Theorem 3.3 is supported through a standard bias-variance decomposition of risk:

$$\mathbb{E}_{h}\left[L\left(h\right)\right]:=\mathbb{E}_{h}\left[\ \mathbb{E}_{x}\left[\left(h(x)-\mathbb{E}[y\ |\ x]\right)^{2}\right]\right.$$ $$=L\left(\mathbb{E}_{h}\left[h\right]\right)+\mathbb{E}_{x}\left[\mathbb{V}_{h}\left(h(x)\right)\right].\tag{3}$$

Since h
(LN)
W∗ and h
(LN) W1
, . . . , h(LN)
WM
share the same expected predictor (as established in Theorem 3.2), the only difference in the generalization of h
(LN)
W∗ and h¯
(LN)
W1:M
arises from their variances. Due to the independence between ensemble members, we have that VW1:M [h¯
(LN) W1:M
(x)] =
1 M 
VWm[h
(LN)
Wm
(x)]. Moreover, prior works such as (Adlam
& Pennington, 2020) and empirical results (see Appx. A.3) suggest the variance of a single RF model is inversely proportional to the number of features.3 As a consequence, we have that VW∗ [h
(LN)
W∗ (x)] : VWm[h
(LN)
Wm
] ≍ 1/M, further suggesting that the generalization of ensembles and single models should be similar under the same parameter budget. Fig. 3 (left) and Appx. A.3 empirically confirm that RF ensembles versus single RF models obtain similar generalization under fixed feature budgets. Moreover, Fig. 3 (right) depicts a similar trend for neural networks: deep ensembles perform roughly the same as larger single models under a fixed parameter budget. These results show that ensembles offer no meaningful generalization advantage over (large) single models and, since the arguments hold for any test distribution, align with empirical findings (Abe et al., 2022b) that ensembles provide no additional robustness benefits.

## 3.3. Implications For Uncertainty Quantification

We now analyze the predictive variance amongst component models in an overparameterized RF ensemble, a quantity often used to quantify predictive uncertainty in safety-critical applications (Lakshminarayanan et al., 2017). Before diving in to a mathematical characterization, it is worth reflecting on the qualitative characterization based on our existing results. Because the expected overparameterized RF model 3This rate is exact for Gaussian features and approximate for the general case.

Single model Kernel model Ensemble 0 1000 2000 3000 4000 5000 6000 7000 Total number of features used 1.6 1.7 1.8 1.9 2.0 2.1 2.2 G
en er ali za tio n e rro r Single Large Model Ensemble 0 1 2 3 4 5 6 7 Total number of parameters 1e6 0.220 0.225 0.230 0.235 0.240 0.245 0.250 G
en era liz ati on er ror
$\bar{\mathfrak{u}}\mathfrak{n}\mathfrak{f}\mathfrak{h}\mathfrak{r}$
is the infinite-width model, predictive variance is equal to

$$\begin{array}{c}{{\mathbb{E}_{\mathcal{W}}[h_{\mathcal{W}}^{(\mathrm{LN})}(x^{*})]=\mathbb{E}_{\mathcal{W}}\Big[\big(h_{\mathcal{W}}^{(\mathrm{LN})}(x^{*})-\frac{\mathbb{E}_{\mathcal{W}}[h_{\mathcal{W}}^{(\mathrm{LN})}(x^{*})]}{\bar{h}_{\infty}^{(L N)}(x^{*})}\big)^{2}\Big]}}\\ {{=\mathbb{E}_{\mathcal{W}}\Big[\big(h_{\mathcal{W}}^{(\mathrm{LN})}(x^{*})-h_{\infty}^{(\mathrm{LN})}(x^{*})\big)^{2}\Big],}}\end{array}$$

i.e. the expected difference between finite- versus infinitewidth RF model predictions. In other words, *ensemble* variance quantifies how predictions change if we increase model capacity. This characterization, which holds for all random feature distributions satisfying Assumption 2.1, is not a standard frequentist or Bayesian notion of uncertainty except under specific distributional assumptions. Uncertainty quantification under Gaussian features. Using Theorem 3.2, the variance of the predictions of a single RF model with respect to its random features can be expressed as (see Appx. C.3 for a derivation)

$$\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(\mathrm{LN})}(x^{*})]=r_{\perp}^{2}\Big{(}y^{\top}R^{-1}\,\mathbb{E}_{W,w_{\perp}}\big{[}(WW^{\top})^{-T}Ww_{\perp}$$ $$\cdot w_{\perp}^{\top}W^{\top}(WW^{\top})^{-1}\big{]}\,R^{-\top}y\Big{)}.\tag{4}$$

In the special case where W and w⊥ are i.i.d. standard normal, this expression simplifies to

$$\forall_{\mathcal{W}}[h_{\mathcal{W}}^{(\mathrm{LN})}(x^{*})]=r_{\perp}^{2}\left(\frac{\|h_{\infty}^{(\mathrm{LN})}\|_{k}^{2}}{D-N-1}\right),$$

where ∥h
(LN)
∞ ∥
2K represents the squared norm of h
(LN)
∞ in the RKHS defined by the kernel k(·, ·). From this equation, we see that VW[h
(LN)
W (x
∗)] only depends on x
∗through the quantity r 2⊥, which by Eq. (1) is equal to

$$r_{\perp}^{2}=k(x^{*},x^{*})-k_{N}(x^{*})^{\top}K^{-1}k_{N}(x^{*}).$$
$$(5)$$

We recognize this quantity as the Gaussian process posterior variance with prior covariance k(·, ·) (e.g. Rasmussen &
Williams, 2006). Thus, with Gaussian features, ensemble variance admits a Bayesian interpretation in addition to the model capacity interpretation. In other words, Gaussianity assumptions justify the use of overparameterized ensemble variance in uncertainty quantification tasks. Uncertainty quantification under general features. Unfortunately, this Bayesian interpretation explicitly does not carry over to the general Assumption 2.1 case. Although Eq. (4) still holds for general feature distributions, it does not have a simple expression unless W and w⊥ are independent. The variance depends on x
∗through both r 2⊥ as well as through a complicated expectation involving W and w⊥. In Appx. C.3 we demonstrate with a simple example that this expectation can indeed depend on x
∗, implying that ensemble variance does not correspond to a scalar multiple of r 2
⊥ (i.e. the Gaussian process posterior variance).

In numerical experiments using ReLU random features, (Fig. 4 and Appx. A.3), we observe significant deviations between the ensemble variance and the Gaussian process posterior variance, further suggesting that one cannot view ensembles through a classic framework of uncertainty. These discrepancies are particularly important for uncertainty estimation in safety-critical applications or active learning (e.g. Gal et al., 2017; Beluch et al., 2018), as the only meaningful interpretation of ensemble variance (the expected change in prediction from increasing capacity) may not yield reliability guarantees or useful exploration-exploitation tradeoffs.

4 2 0 2 4 0.000 0.002 0.004 0.006 0.008 0.010 0.012 0.014 0.016 VarπD[h
(LN)
S(·)]
Training Data Points 4 2 0 2 4 0.000 0.002 0.004 0.006 0.008 0.010 0.012 r 2 ⊥
Training Data Points

## 3.4. **Equivalence Of The Limiting Predictors In The Small** Ridge Regime

Having established the equivalence between infinite ensembles and infinite-width single models in the ridgeless regime, we now investigate whether this equivalence approximately persists in the practically relevant setting when a small ridge regularization parameter λ > 0 is introduced. More generally, we aim to determine whether the transition from the ridgeless case to the small ridge regime is smooth. While h
(RR)
∞,λ , the infinite-width limit of h
(RR)
W,λ as |W| = D → ∞,
almost surely converges to the kernel ridge regressor with ridge λ, the infinite ensemble h¯
(RR)
∞,λ := EW[h
(RR)
W,λ (x)] does not generally maintain pointwise equivalence with h
(RR)
∞,λ .

This divergence occurs even under Gaussianity assumptions (Jacot et al., 2020). However, we hypothesize that the difference between these limiting predictors is small when λ is close to zero, which is common in practical applications. To analyze this regime, we introduce a minor additional assumption, which is weaker than Gaussianity:
Assumption 3.4. We assume that EW[ΦWΦ
⊤W
−1] is finite for all |W| = *D > N*.

Under Assumptions 2.1 and 3.4, we show that the difference between ridge-regularized ensembles and single models is Lipschitz-continuous with respect to λ (proof in Appx. D.1). Theorem 3.5 (The difference between ensembles and large single models is smooth with respect to λ.). Under Assumptions 2.1 and *3.4, the difference* |h¯
(RR)
∞,λ (x
∗) − h
(RR)
∞,λ (x
∗)| between the infinite ensemble and the single infinite-width model trained with ridge λ ≥ 0 *is Lipschitz-continuous in* λ.

The Lipschitz constant is independent of x
∗*for compact* X .

This result is illustrated in Fig. 5, where the terms bounding the difference evolve smoothly with λ. To the best of our knowledge, this Lipschitz-continuity has not been established even under Gaussianity assumptions. We note that the bound by Jacot et al. (2020, Thm. 4.1), which characterizes the difference between ridge ensembles and infinite models with Gaussian random features, becomes vacuous as λ → 0. Since Theorem 3.2 ensures that |h¯
(RR)
∞,λ (x
∗) − h
(RR)
∞,λ (x
∗)| = 0 for λ = 0, we can conclude that the pointwise difference grows at most linearly with λ.

Specifically, we have that

$$\left|\bar{h}_{\infty,\lambda}^{(R R)}-h_{\infty,\lambda}^{(R R)}(x)\right|\leq C\cdot\lambda,$$

for some constant C independent of x
∗, provided that X
is compact. In practical terms, this result indicates that for sufficiently small values of λ, the predictions of large ensembles and large single models remain nearly indistinguishable, reinforcing our findings from the ridgeless regime.

## 4. Conclusion

This work characterized overparameterized RF ensembles and contextualized theoretical findings with neural network experiments. We used weaker distributional assumptions than prior work to (a) more faithfully approximate realworld models and (b) highlight differences between ensembles versus models with Gaussian behaviour. For *Question 1*, we demonstrated under weak conditions that infinite ensembles and single infinite-width models are pointwise equivalent in the ridgeless regime (Theorem 3.2) and nearly identical with a small ridge (Theorem 3.5), significantly expanding on prior results. We further provide a non-asymptotic characterization, showing that ensembles and large single models with the same parameter budget are nearly equivalent (Theorem 3.3). These results verify recent empirical findings (e.g. Abe et al., 2022b) that much

0.0000 0.0002 0.0004 0.0006 0.0008 0.0010 2.0 1.5 1.0 0.5 0.0 0.5 1.0 1.5 0.0000 0.0002 0.0004 0.0006 0.0008 0.0010 2 1 0 1 2 |h
(R R)
, 
( ) 
h

(LS)
( 
)| |h
(R
R)
, 
( ) 
h

(LS
)( 
)|
Figure 5: **Lipschitz continuity of predictions for an infinite ensemble and kernel regressor with respect to the ridge**
parameter. (Left) We plot |h¯
(RR)
∞,λ (x
∗) − h¯
(LS)
∞ (x
∗)| as a function of λ for 500 test points. (Right) We show the evolution of |h
(RR)
∞,λ (x
∗) − h
(LS)
∞ (x
∗)| for the same test points. Both plots use ReLU activation functions and the California Housing Dataset with N = 12 and D = 200. While the direct difference |h¯
(RR)
∞,λ (x
∗)−h
(RR)
∞,λ (x
∗)| is not shown (for reasons outlined in Appx. A.4), it is bounded by the sum of the plotted quantities (see Appx. D.1). The evolution of these plotted bounding terms thus illustrates that this direct difference is Lipschitz continuous in λ (proven in Theorem 3.5) and converges to zero as λ → 0 (a consequence of Theorem 3.5 and the ridgeless equivalence established in Theorem 3.2).

of the benefit attributed to overparameterized ensembles, such as improved predictive performance and robustness, can be explained by their similarity to larger single models. Notably, our analysis does not rely on Gaussianity, emphasizing that these phenomena are fundamental properties of overparameterized models and not artifacts of specific feature assumptions. In contrast, for *Question 2*, we found that uncertainty interpretations of ensemble variance are contingent on Gaussianity assumptions and fall apart under more general feature distributions. We characterize ensemble variance as the expected difference to a single larger model, which only corresponds to a (scaled) Bayesian notion of uncertainty under strong independence assumptions. With more realistic feature distributions, the ensemble variance does not correspond to any conventional notion of uncertainty, reinforcing recent empirical findings on the limitations of ensemble uncertainty quantification (Abe et al., 2022b). This deviation supplies further evidence that caution is needed when using ensembles in safety-critical settings. Overall, while our results do not contradict the utility of overparameterized ensembles, they suggest that their benefits may often be explained by their similarity to larger models and that further research is needed to improve uncertainty quantification methods.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

GP is supported by Canada CIFAR AI Chairs. JPC is supported by the Gatsby Charitable Foundation (GAT3708), the Simons Foundation (542963), the NSF AI Institute for Artificial and Natural Intelligence (ARNI: NSF DBI 2229929) and the Kavli Foundation. We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC: RGPIN-2024-06405). Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

## References

Abe, T., Buchanan, E. K., Pleiss, G., and Cunningham, J. P.

The best deep ensembles sacrifice predictive diversity. In I Can't Believe It's Not Better Workshop: Understanding Deep Learning Through Empirical Falsification, 2022a.

Abe, T., Buchanan, E. K., Pleiss, G., Zemel, R., and Cunningham, J. P. Deep ensembles work, but are they necessary? In Advances in Neural Information Processing Systems, 2022b.

Abe, T., Buchanan, E. K., Pleiss, G., and Cunningham, J. P.

Pathologies of predictive diversity in deep ensembles. Transactions on Machine Learning Research, 2024.

Adlam, B. and Pennington, J. Understanding double descent requires a fine-grained bias-variance decomposition. In *Advances in Neural Information Processing Systems*, 2020.

Angelova, J. A. On moments of sample mean and variance.

Int. J. Pure Appl. Math, 79(1):67–85, 2012.

Bach, F. High-dimensional analysis of double descent for linear regression with random projections. SIAM Journal on Mathematics of Data Science, 6(1):26–50, 2024a.

Bach, F. *Learning Theory from First Principles*. MIT Press, 2024b.

Bartlett, P. L., Long, P. M., Lugosi, G., and Tsigler, A.

Benign overfitting in linear regression. Proceedings of the National Academy of Sciences, 117(48):30063–30070, 2020.

Belkin, M., Ma, S., and Mandal, S. To understand deep learning we need to understand kernel learning. In International Conference on Machine Learning, pp. 541–549, 2018.

Belkin, M., Hsu, D., Ma, S., and Mandal, S. Reconciling modern machine-learning practice and the classical bias– variance trade-off. *Proceedings of the National Academy* of Sciences, 116(32):15849–15854, 2019.

Beluch, W. H., Genewein, T., Nurnberger, A., and Kohler, J. M. The power of ensembles for active learning in image classification. In Computer Vision and Pattern Recognition, 2018.

Breiman, L. Bagging predictors. *Machine Learning*, 24(2):
123–140, 1996.

Breiman, L. Random forests. *Machine Learning*, 45(1):
5–32, 2001.

Chen, L., Lukasik, M., Jitkrittum, W., You, C., and Kumar, S. On bias-variance alignment in deep models. In International Conference on Learning Representations, 2024.

Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. In International Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016.

Cho, Y. and Saul, L. Kernel methods for deep learning.

In *Advances in Neural Information Processing Systems*,
2009.

Dietterich, T. G. Ensemble methods in machine learning. In International Workshop on Multiple Classifier Systems, pp. 1–15. Springer, 2000.

Fort, S., Hu, H., and Lakshminarayanan, B. Deep ensembles: A loss landscape perspective. arXiv preprint arXiv:1912.02757, 2019.

Freund, Y. Boosting a weak learning algorithm by majority.

Information and Computation, 121(2):256–285, 1995.

Gal, Y., Islam, R., and Ghahramani, Z. Deep Bayesian active learning with image data. In *International Conference on* Machine Learning, pp. 1183–1192, 2017.

Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A.

Limitations of lazy training of two-layers neural network. In *Advances in Neural Information Processing Systems*, 2019.

Goldt, S., Loureiro, B., Reeves, G., Krzakala, F., Mezard, ´
M., and Zdeborova, L. The gaussian equivalence of gen- ´ erative models for learning with shallow neural networks.

In *Mathematical and Scientific Machine Learning*, pp.

426–471, 2022.

Gustafsson, F. K., Danelljan, M., and Schon, T. B. Evaluating scalable bayesian deep learning methods for robust computer vision. In Computer Vision and Pattern Recognition Workshops, pp. 318–319, 2020.

Hansen, L. and Salamon, P. Neural network ensembles.

Pattern Analysis and Machine Intelligence, 12(10):993–
1001, 1990.

Hao, Y., Lin, Y., Zou, D., and Zhang, T. On the benefits of over-parameterization for out-of-distribution generalization. *arXiv preprint arXiv:2403.17592*, 2024.

Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J.

Surprises in high-dimensional ridgeless least squares interpolation. *Annals of Statistics*, 50(2):949, 2022.

Hu, H. and Lu, Y. M. Universality laws for high-dimensional learning with random features. IEEE Transactions on Information Theory, 69(3):1932–1964, 2022.

Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel:
Convergence and generalization in neural networks. Advances in Neural Information Processing Systems, 2018.

Jacot, A., Simsek, B., Spadaro, F., Hongler, C., and Gabriel, F. Implicit regularization of random feature models. In International Conference on Machine Learning, 2020.

Jeffares, A., Liu, T., Crabbe, J., and van der Schaar, M. Joint ´
training of deep ensembles fails due to learner collusion.

In *Advances in Neural Information Processing Systems*,
2024.

Kaban, A. New bounds on compressive linear least squares ´
regression. In *Artificial Intelligence and Statistics*, pp. 448–456, 2014.

Kelley Pace, R. and Barry, R. Sparse spatial autoregressions.

Statistics & Probability Letters, 33(3):291–297, 1997.

Kendall, A. and Gal, Y. What uncertainties do we need in Bayesian deep learning for computer vision? In *Advances* in Neural Information Processing Systems, 2017.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, 2017.

Lee, J., Bahri, Y., Novak, R., Schoenholz, S., Pennington, J., and Sohl-dickstein, J. Deep neural networks as gaussian processes. In *International Conference on Learning* Representations, 2018.

Lee, J., Schoenholz, S., Pennington, J., Adlam, B., Xiao, L.,
Novak, R., and Sohl-Dickstein, J. Finite versus infinite neural networks: an empirical study. In *Advances in* Neural Information Processing Systems, volume 33, pp. 15156–15172, 2020.

Lee, S., Purushwalkam, S., Cogswell, M., Crandall, D., and Batra, D. Why m heads are better than one: Training a diverse ensemble of deep networks. arXiv preprint arXiv:1511.06314, 2015.

Li, M., Nica, M., and Roy, D. The future is log-Gaussian:
ResNets and their infinite-depth-and-width limit at initialization. *Advances in Neural Information Processing* Systems, 34:7852–7864, 2021.

Loureiro, B., Gerbelot, C., Refinetti, M., Sicuro, G., and Krzakala, F. Fluctuations, bias, variance & ensemble of learners: Exact asymptotics for convex losses in highdimension. In *International Conference on Machine* Learning, pp. 14283–14314. PMLR, 2022.

Mallinar, N., Simon, J. B., Abedsoltan, A., Pandit, P., Belkin, M., and Nakkiran, P. Benign, tempered, or catastrophic: A taxonomy of overfitting. In Advances in Neural Information Processing Systems, 2022.

Mei, S. and Montanari, A. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75(4):667–766, 2022.

Mentch, L. and Hooker, G. Quantifying uncertainty in random forests via confidence intervals and hypothesis tests. *Journal of Machine Learning Research*, 17(26):
1–41, 2016.

Montanari, A. and Saeed, B. N. Universality of empirical risk minimization. In *Conference on Learning Theory*,
pp. 4310–4312, 2022.

Neal, R. M. and Neal, R. M. Priors for infinite networks.

Bayesian learning for neural networks, pp. 29–53, 1996.

Nixon, J., Lakshminarayanan, B., and Tran, D. Why are bootstrapped deep ensembles not better? In NeurIPS "I Can't Believe It's Not Better!" Workshop, 2020.

Opitz, D. and Maclin, R. Popular ensemble methods: An empirical study. Journal of Artificial Intelligence Research, 11:169–198, 1999.

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D.,
Nowozin, S., Dillon, J., Lakshminarayanan, B., and Snoek, J. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, 2019.

Pleiss, G. and Cunningham, J. P. The limitations of large width in neural networks: A deep Gaussian process perspective. Advances in Neural Information Processing Systems, 34:3349–3363, 2021.

Rahimi, A. and Recht, B. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, 2007.

Rahimi, A. and Recht, B. Uniform approximation of functions with random bases. In 46th Annual Allerton Conference on Communication, Control, and Computing, pp. 555–561, 2008a.

Rahimi, A. and Recht, B. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. In *Advances in Neural Information Processing* Systems, 2008b.

Rasmussen, C. E. and Williams, C. K. Gaussian Processes for Machine Learning. MIT Press, 2006.

Ruben, B. S., Tong, W. L., Chaudhry, H. T., and Pehlevan, C. No free lunch from random feature ensembles. *arXiv* preprint arXiv:2412.05418, 2024.

Rudi, A. and Rosasco, L. Generalization properties of learning with random features. Advances in Neural Information Processing Systems, 2017.

Simon, J. B., Karkada, D., Ghosh, N., and Belkin, M. More is better in modern machine learning: when infinite overparameterization is optimal and overfitting is obligatory. In *International Conference on Learning Representations*, 2024.

Tao, T. *Topics in Random Matrix Theory*. American Mathematical Soc., 2012.

Thanei, G.-A., Heinze, C., and Meinshausen, N. Random projections for large-scale regression. Big and Complex Data Analysis: Methodologies and Applications, pp. 51–
68, 2017.

Theisen, R., Kim, H., Yang, Y., Hodgkinson, L., and Mahoney, M. W. When are ensembles really effective? Advances in Neural Information Processing Systems, 36, 2024.

Wager, S., Hastie, T., and Efron, B. Confidence intervals for random forests: The jackknife and the infinitesimal jackknife. *Journal of Machine Learning Research*, 15(1): 1625–1651, 2014.

Wainwright, M. J. High-Dimensional Statistics: A Non-
Asymptotic Viewpoint. Cambridge University Press, 2019.

Webb, A., Reynolds, C., Chen, W., Reeve, H., Iliescu, D.,
Lujan, M., and Brown, G. To ensemble or not ensemble: When does end-to-end training fail? In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part III, pp. 109–
123. Springer, 2021.

Williams, C. K. I. Computing with infinite networks. In Advances in Neural Information Processing Systems, 1996.

Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J. Y., Levine, S., Finn, C., and Ma, T. MOPO: Model-based offline policy optimization. In Advances in Neural Information Processing Systems, 2020.

4 2 0 2 4 1.75 1.50 1.25 1.00 0.75 0.50 0.25 0.00 0.25 Data Generating Function Training Data 4 2 0 2 4 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00 Data Generating Function Training Data
In the appendix, we will provide the following additional results:
1. In Appx. A, we will describe our experimental setup for RF models in more detail, difficulties we encountered when developing the experiments, and provide the results of additional experiments.

2. In Appx. B, we will describe our experimental setup for neural network models in more detail, and provide the results of additional experiments.

3. In Appx. C we will give the proofs for Secs. 3.1 and 3.3 in the main paper. 4. In Appx. D we will give the proofs for Sec. 3.4 in the main paper. 5. Finally, in Appx. E, we prove (under mild assumptions) that infinite underparameterized RF ensembles are equivalent to kernel ridge regression under some transformed kernel.

The code to run all our experiments can be found on GitHub: https://github.com/nic-dern/ theoretical-limitations-overparameterized-ensembles. It contains a README.md file that explains how to set up and run the experiments.

## A. Experimental Setup And Additional Results For Rf Models A.1. Experimental Setup

We had two setups using which we performed most of our experiments:
1. We generate training and test points uniformly at random from [−5, 5]d using the function f(x) = sin(5 · b
⊤x), where b is a vector (depending on the random seed) and the noise parameter is σ = 0.05 (we assume Gaussian noise with mean 0). In this setting, we use N = 6, D = 200, and data from R (i.e., d = 1) if not specified otherwise. You can find a plot of an example true function in Fig. 6.

2. We use the California Housing (Kelley Pace & Barry, 1997) dataset and sample distinct training and test points from it (randomly permutating the dataset initially). In this setting, we use N = 12, D = 200 if not differently specified.

The data dimension is R
8 here. In contrast to the first setting, we employ a data normalization using a max-min normalization *on the entire dataset* since we experimentally found this makes our methods more stable.

We calculate the generalization error using N = 1000 test points in both settings. In the first setting, we calculate the variance of the predictions of a single model using M = 20, 000 models, while in the second setting, we use M = 4, 000 models. Apart from Fig. 12 where we use 100, 000 samples, "infinite" ensembles consist of M *= 10*, 000 models.

x1 Training Points after transformation Line x0 = 1 Hyperplane of ω1 Hyperplane of ω2 x0
As distribution τ (·) of the elements ωi ∈ W we always use N (0, I). As activation functions, we use ReLU, the Gaussian error function, and the softplus function 1β· log(1 + exp(β · ω
⊤x)) with β = 1. For the first two activation functions, there exist analytically calculatable limiting kernels, the arc-cosine kernel (Cho & Saul, 2009) and the erf-kernel (Williams, 1996). The closed forms for these are

$$k_{\mathrm{arc-cosine}}(x,x^{\prime})={\frac{1}{2\pi}}\|x\|\|x^{\prime}\|\left(\sin\theta+(\pi-\theta)\cos\theta\right),$$
where $\theta=\cos^{-1}\left(\frac{x^{\top}x^{\prime}}{\|x\|\|x^{\prime}\|}\right)$ and 
$$k_{\mathrm{{erf}}}(x,x^{\prime})={\frac{2}{\pi}}\sin^{-1}\left({\frac{2x^{\top}x^{\prime}}{\sqrt{(1+2\|x\|^{2})(1+2\|x^{\prime}\|^{2})}}}\right)$$
.
For the softplus function, we approximate the kernel by estimating the second moment k(*x, x*′) = E[ϕ(ω, x)ϕ(ω, x′) | *x, x*′]
of the feature extraction using 107samples from τ (·). For sampling Gaussian features, we use the same approach as described by Jacot et al. (2020). Before training on data, we always append a 1 in the zeroeth-dimension of the data before calculating the dot product with ω
(correspondingly, the dimension of ω is d + 1) and applying the activation function. In the ridgeless case, we use λ = 10−8 to avoid numerical issues.

## A.2. Notes On Stability

During our experiments, we encountered challenges related to both mathematical stability (i.e., matrices being truly singular rather than nearly singular) and numerical stability. This section outlines these issues and describes the steps we took to mitigate them.

Most importantly, the matrix ΦWΦ
⊤
W is not almost surely invertible when using the ReLU activation function, meaning that technically, the second condition of our Assumption 2.1 is not fulfilled. In numerical experiments, this results in cases where (ΦWΦ
⊤
W)
−1is nearly singular (though stabilized with λ = 10−8).

On the other hand, when D is sufficiently large relative to N, ΦW is full rank with high probability, which implies that ΦWΦ
⊤
W is invertible with high probability. Given our data transformation of appending a 1 in the zeroeth dimension, one can see this as there exists a series of (non-zero probability sets of) hyperplanes separating an increasing subset of the training points, leading to a subset of ΦW's columns that form a triangular, invertible matrix (see Fig. 7 for a visualization).

Intuitively, higher data dimensionality and better separability of the points increase the probability of ΦW having full rank. As an example of the discussed instabilities, see the adversarial scenario shown in Fig. 8, where N = 15 and many training points are placed very close to each other. In this case, individual RF regressors exhibit relatively high variance output values
(due to numerical instabilities), which are not averaged out in the "infinite" ensemble. Similar issues were also observed when using the Gaussian error function as the activation function, although they were generally less pronounced.

4 2 0 2 4 14 12 10 8 6 4 2 0 2 RF Models Kernel Model Training Data 4 2 0 2 4 4 3 2 1 0 Ensemble Model Kernel Model Training Data 0 10 20 30 40 50 Number of features per model 0.3 0.4 0.5 0.6 0.7 0.8 0.9 N = Total Parameters h

(LN) 
h

(LN) 
L

1 h

(LN) 
h

(LN) 
L

1 0 10 20 30 40 50 Number of features per model 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 N = Total Parameters
To alleviate these issues, we used the following approaches:
- We used a relatively low number of samples, N = 6 or N = 12, compared to D = 200. As shown in Fig. 1, even with D = 200, there is still a considerable amount of variance in the RF regressors (i.e., the individual RF regressors are not yet closely approximating the limiting kernel ridge regressor).

- We appended a 1 in the zeroeth dimension of the data before calculating the dot product with ω.

- We performed additional experiments using the softplus function with β = 1 as a smooth approximation of the ReLU
activation function. This often helped stabilize the numerical computations, as seen in Fig. 9, where we repeated a part of the experiment from Fig. 2 using the ReLU function as activation function which increased the numerical instability for low D values.

- We used a ridge term λ = 10−8in the ridgeless case to stabilize the inversion of ΦWΦ
⊤
W.

- We used *double precision* for all computations and used the torch.linalg.lstsq function with the driver gelsd
(for not-well-conditioned matrices) to solve linear systems.

- We applied max-min normalization to the entire California Housing dataset to improve stability.

## A.3. Additional Experiments For The Ridgeless Case

To address the question of whether our findings are specific to normally distributed weights ωi for the feature generating function, we supplement Fig. 1. Fig. 10 replicates that visualization using weights ωi drawn from a Uniform(-10, 10)
distribution and the softplus activation function. As can be seen, the equivalence between the infinite ensemble of overparameterized RF models and the single infinite-width RF model remains apparent, and no perceptible difference is observed.

4 2 0 2 4 1.0 0.5 0.0 0.5 1.0 1.5 RF Models Kernel Model Training Data 4 2 0 2 4 0.5 0.0 0.5 1.0 1.5 Ensemble Model Kernel Model Training Data 
Figure 10: Replication of Fig. 1 **with Uniformly Distributed Weights** ωi. Similar to Fig. 1, we demonstrate the equivalence of an infinite ensemble of overparameterized RF models to a single infinite-width RF model. Here, the weights ωi for the Softplus activation functions are drawn from a Uniform(-10, 10) distribution. (Left) A sample of 100 finite-width RF models (blue) trained on N = 6 data points, with the single infinite-width RF model (pink). (Right) The infinite-width RF model (pink) and the "infinite" ensemble of M *= 10*, 000 RF models (blue). No perceptible difference is observed, mirroring the findings with normally distributed weights.

Furthermore, to illustrate the convergence of finite ensembles to the infinite-width model prediction as the number of ensemble members M increases, Fig. 11 expands on the setting of Fig. 1. It shows that even with a small number of ensemble members, the average prediction begins to concentrate around the infinite-width model, and this concentration improves as M grows.

4 2 0 2 4 0.5 0.0 0.5 1.0 1.5 Ensemble Model Kernel Model Training Data 4 2 0 2 4 0.5 0.0 0.5 1.0 1.5 Ensemble Model Kernel Model Training Data
(a) Ensembles of M = 5 members
(b) Ensembles of M = 10 members 4 2 0 2 4 0.5 0.0 0.5 1.0 1.5 Ensemble Model Kernel Model Training Data 4 2 0 2 4 0.5 0.0 0.5 1.0 1.5 Ensemble Model Kernel Model Training Data
(c) Ensembles of M = 20 members
(d) Ensembles of M = 40 members
Additional experiments on the identity of infinite-width single model and infinite ensembles. In Fig. 12, we show that the term E[w
⊤⊥W⊤(WW⊤)
−1] is consistently zero for both ReLU and the Gaussian error function activations consistetly with Lemma 3.1. To further demonstrate that this result is not dependent on Gaussian-like weight distributions, Fig. 13 shows this for a softplus activation function, with weights ωi drawn from a Uniform(-10, 10) distribution and a Laplace(0, 1)
distribution. The expectation of the term remains centered at zero, supporting the generality of our theoretical findings. Additional experiments on the ensemble variance. We observed a different behavior of the RF regressor variance and r 2
⊥ as shown in Fig. 4 consistently across different random seeds and dimensions for both ReLU and the Gaussian error function activations as activation functions. In Fig. 14, we present additional examples for the Gaussian error function in one dimension and the ReLU activation in two dimensions. Additional experiment on generalization error and variance scaling. In Fig. 3, the generalization error decay for the ReLU activation function. To verify the consistency of this trend, we repeated the experiment using the Gaussian error function and the corresponding erf-kernel. The result is very similar, shown in Fig. 15. Furthermore, this figure shows that the variance of a single model with MD features decays as ∼1 MD , matching the ensemble's behavior.

0.08 0.06 0.04 0.02 0.00 0.02 0.04 0.06 0.08
[w W (WW ) 1]1 (whitened finite model residual)
0 2000 4000 6000 8000 10000 12000 0.20 0.15 0.10 0.05 0.00 0.05 0.10 0.15
[w W (WW ) 1]1 (whitened finite model residual)
0 50 100 150 200 250 300 Mean: 0.00 Mean: 0.00 Frequen cy Frequen cy Mean: 0.00 Mean: 0.00 0.15 0.10 0.05 0.00 0.05 0.10 0.15
[w W (WW ) 1]1 (whitened finite model residual)
0 5 10 15 20 25 30 35 40 0.20 0.15 0.10 0.05 0.00 0.05 0.10 0.15
[w W (WW ) 1]1 (whitened finite model residual)
0 5 10 15 20 25 30 35 40 Frequenc y Frequenc y

(a) Softplus activation, Uniform(-10, 10) weights ωi
(b) Softplus activation, Laplace(0, 1) weights ωi 4 2 0 2 4 0.000 0.002 0.004 0.006 0.008 0.010 0.012 0.014 0.016 VarπD[h
(LN)
S(·)]
Training Data Points 4 2 0 2 4 0.000 0.005 0.010 0.015 0.020 0.025 0.030 r 2 ⊥
Training Data Points r 2
⊥
Training Data Points VarπD
[h
(LN)
S(·)]
Training Data Points 4 2 0 2 44 2 0 2 4 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 4 2 0 2 44 2 0 2 4 0.0 0.1 0.2 0.3 0.4 Single model Kernel model Ensemble 10 3 Number of features of random feature models 10 4 10 3 10 2 10 1 10 0 3.2 3.4 3.6 3.8 G
en er ali za tio n e rr or V

ar D [
h

(LN
)

S

( 
)]
Point 1 Point 2 Point 3 Point 4 Point 5 0 1000 2000 3000 4000 5000 6000 7000 Total number of features used Mean: 0.00 Mean: -0.00 0.20 0.15 0.10 0.05 0.00
[w W (WW + D R R
1) 1]3 0 200 400 600 800 1000 0.010 0.005 0.000 0.005 0.010
[w W (WW + D R R
1) 1]3 0 50 100 150 200 250 Freq uen cy Freq uen cy
Figure 16: **Empirically, the term** EW,w⊥
hw
⊤
⊥W⊤WW⊤ + D · λ · R−⊤R−1−1i**is consistently zero.** We show the empirical distribution of an index of w
⊤
⊥W⊤WW⊤ + D · λ · R−⊤R−1−1∈ R
N , which captures the difference in predictions between c
⊤EW,w⊥
hWW⊤WW⊤ + D · λ · R−⊤R−1−1iR−⊤y and a finite-sized overparameterized RF
model (see Eq. (7)). We use λ = 1.0 in both plots. (Left) We use a ReLU activation function, xi ∈ R, and N = 6, D = 200.

(Right) We use the Gaussian Error Function as activation function, the California Housing dataset, and N = 12, D = 200.

## A.4. More Experiments For The Ridge Case

Additional experiments for the convergence of the expected value term. In Appx. D, we show that a variant of Lemma 3.1 also holds in the ridge case. More precisely, we show that

$$\mathbb{E}_{W,w_{\perp}}\left[w_{\perp}^{\top}W^{\top}\left(W W^{\top}+D\cdot\lambda\cdot R^{-\top}R^{-1}\right)^{-1}\right]=0$$

under Assumption 2.1. We repeated the experiment from Fig. 12 for the ridge case to verify this experimentally. The results are shown in Fig. 16.

Additional notes. In Fig. 5, we illustrate the Lipschitz continuity of the predictions for an infinite ensemble and a kernel regressor with respect to the ridge parameter. Rather than directly presenting the difference
h¯
(RR)
∞,λ (x
∗) − h
(RR)
∞,λ (x
∗)
, we show the evolution of
h¯
(RR)
∞,λ (x
∗) − h¯
(LS)
∞ (x
∗)
and
h
(RR)
∞,λ (x
∗) − h
(LS)
∞ (x
∗)
. This choice was made because the upper bound we obtained was not consistently tight for settings with large D. In particular, the pointwise predictions of the infinite ensemble h¯
(RR)
∞,λ and the single infinite-width model h
(RR)
∞,λ trained with ridge λ were already very close for non-zero λ. We opted to display the upper bounds rather than the direct difference to avoid cherry-picking favorable settings. Our best explanation for this phenomenon is that infinite ensembles under Assumption 2.1 in the ridge regime often behave similarly to the single infinite-width model h
(RR)
∞,λ˜ with an *implicit ridge* parameter λ˜, which solves the equation

$$\tilde{\lambda}=\lambda+\frac{\tilde{\lambda}}{D}\sum_{i=1}^{N}\frac{d_{i}}{\bar{\lambda}+d_{i}}$$

where di are the eigenvalues of the kernel matrix K, as shown by Jacot et al. (2020) under Gaussianity. Intuitively and empirically, for large D, the implicit ridge λ˜ tends to be very close to the true ridge λ. Using Lemma D.1, this suggests that for small values of λ, the difference between the infinite ensemble and the infinite-width single model h
(RR)
∞,λ with ridge λ is already minimal before λ approaches zero.

Interestingly, our findings (see Fig. 2) suggest that in the ridgeless case, the similarity to the ridge regressor with the implicit ridge only holds in the overparameterized regime. Note that this does not violate the results from Jacot et al. (2020) since the constants in their bounds blow up as λ → 0 in both the underparameterized and overparameterized regimes.