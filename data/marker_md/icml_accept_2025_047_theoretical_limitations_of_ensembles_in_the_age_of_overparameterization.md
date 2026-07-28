# Theoretical Limitations of Ensembles in the Age of Overparameterization

Niclas Dern\* <sup>1</sup> John P. Cunningham <sup>2</sup> Geoff Pleiss 3 4

## Abstract

Classic ensembles generalize better than any single component model. In contrast, recent empirical studies find that modern ensembles of (overparameterized) neural networks may not provide any inherent generalization advantage over single but larger neural networks. This paper clarifies how modern overparameterized ensembles differ from their classic underparameterized counterparts, using ensembles of random feature (RF) regressors as a basis for developing theory. In contrast to the underparameterized regime, where ensembling typically induces regularization and increases generalization, we prove with minimal assumptions that infinite ensembles of overparameterized RF regressors become pointwise equivalent to (single) infinite-width RF regressors, and finite width ensembles rapidly converge to single models with the same parameter budget. These results, which are exact for ridgeless models and approximate for small ridge penalties, imply that overparameterized ensembles and single large models exhibit nearly identical generalization. We further characterize the predictive variance amongst ensemble members, demonstrating that it quantifies the expected effects of increasing capacity rather than capturing any conventional notion of uncertainty. Our results challenge common assumptions about the advantages of ensembles in overparameterized settings, prompting a reconsideration of how well intuitions from underparameterized ensembles transfer to deep ensembles and the overparameterized regime.

## 1. Introduction

Historically, most machine learning ensembles aggregated component models that are simple by today's standards (e.g. [Hansen & Salamon,](#page-9-0) [1990;](#page-9-0) [Opitz & Maclin,](#page-10-0) [1999;](#page-10-0) [Diet](#page-9-1)[terich,](#page-9-1) [2000\)](#page-9-1). Common techniques like bagging [\(Breiman,](#page-9-2) [1996\)](#page-9-2), feature selection [\(Breiman,](#page-9-3) [2001\)](#page-9-3), random projections [\(Kaban´](#page-9-4) , [2014;](#page-9-4) [Thanei et al.,](#page-10-1) [2017\)](#page-10-1), and boosting [\(Fre](#page-9-5)[und,](#page-9-5) [1995;](#page-9-5) [Chen & Guestrin,](#page-9-6) [2016\)](#page-9-6) were developed and analyzed assuming decision trees, least-squares regressors, and other *underparameterized* component models incapable of achieving zero training error.

Researchers and practitioners have now turned to ensembles of *overparameterized* models, such as neural networks, which have capacity to memorize entire training datasets. Motivated by heuristics from classic ensembles [\(Mentch &](#page-10-2) [Hooker,](#page-10-2) [2016\)](#page-10-2), some have argued that ensembles provide robustness to dataset shift [\(Lee et al.,](#page-10-3) [2015;](#page-10-3) [Fort et al.,](#page-9-7) [2019\)](#page-9-7) and that the predictive variance amongst component models in these so-called *deep ensembles* is a notion of uncertainty that can be used on downstream decision-making tasks [\(Lak](#page-10-4)[shminarayanan et al.,](#page-10-4) [2017;](#page-10-4) [Gal et al.,](#page-9-8) [2017;](#page-9-8) [Gustafsson](#page-9-9) [et al.,](#page-9-9) [2020;](#page-9-9) [Ovadia et al.,](#page-10-5) [2019;](#page-10-5) [Yu et al.,](#page-11-0) [2020\)](#page-11-0).

While few theoretical works analyze modern overparameterized ensembles, recent empirical evidence suggests that intuitions from their underparameterized counterparts do not hold in this new regime. For example, classic methods to increase diversity amongst component models, such as bagging, can be harmful for deep ensembles [\(Nixon](#page-10-6) [et al.,](#page-10-6) [2020;](#page-10-6) [Jeffares et al.,](#page-9-10) [2024;](#page-9-10) [Abe et al.,](#page-8-0) [2022a;](#page-8-0) [2024;](#page-8-1) [Webb et al.,](#page-11-1) [2021\)](#page-11-1) despite being nearly universally beneficial for underparameterized ensembles. Moreover, while established underparameterized ensembling techniques offer well-founded quantifications of uncertainty (e.g. [Mentch](#page-10-2) [& Hooker,](#page-10-2) [2016;](#page-10-2) [Wager et al.,](#page-11-2) [2014\)](#page-11-2), several recent studies question the reliability of the uncertainty estimates from deep ensembles [\(Abe et al.,](#page-8-2) [2022b;](#page-8-2) [Theisen et al.,](#page-11-3) [2024;](#page-11-3) [Chen et al.,](#page-9-11) [2024\)](#page-9-11).

To address this divergence and verify recent empirical findings, we develop a theoretical characterization of ensembles in the overparameterized regime, with the goal of contrasting against (traditional) underparameterized ensembles. We answer the following questions:

<sup>\*</sup>Work done while at the Vector Institute. <sup>1</sup> School of Computation, Information and Technology, Technical University of Munich, Munich, Germany <sup>2</sup>Department of Statistics, Columbia University, Zuckerman Institute, New York, USA <sup>3</sup>Department of Statistics, University of British Columbia, Vancouver, Canada <sup>4</sup>Vector Institute, Toronto, Canada. Correspondence to: Niclas Dern <niclas.dern@gmail.com>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

- 1. Do ensembles of overparameterized models provide generalization or robustness benefits over a single (very large) model trained on the same data? Does the capacity of the component models affect this difference?
- 2. What does the predictive variance of overparameterized ensembles measure, and does it relate to classic frequentist or Bayesian notions of uncertainty?

To answer these questions, we analyze ensembles of overparameterized random feature (RF) linear regressors, a theoretically-tractable approximation of neural networks. Unlike prior work on RF models, our analysis makes very few assumptions about the distribution of random features, which—as we will show—is crucial for highlighting the differences between ensemble variance versus more established notions of uncertainty. Our analysis focuses on the practically relevant regime where RF models are trained with little to no regularization. We verify and contextualize our theory with experiments on RF and neural networks ensembles.

### 1.1. Related Work

Deep ensembles. A primary motivation of this paper is to understand recent empirical findings about uncertainty quantification afforded by deep ensembles [\(Lakshminarayanan](#page-10-4) [et al.,](#page-10-4) [2017\)](#page-10-4). Historically, variance amongst deep ensemble members has been a proxy for *epistemic uncertainty* (e.g. [Kendall & Gal,](#page-10-7) [2017;](#page-10-7) [Gustafsson et al.,](#page-9-9) [2020\)](#page-9-9), i.e., the uncertainty that can be reduced by collecting more data. This view reflects a classical intuition of ensembles: ignoring effects of overparameterization and inductive bias, all ensemble members should converge to the same prediction in the infinite data limit, and thus differing predictions suggest a region of the input space with insufficient data. However, recent empirical findings challenge this interpretation of ensemble variance [\(Abe et al.,](#page-8-2) [2022b;](#page-8-2) [Theisen et al.,](#page-11-3) [2024\)](#page-11-3). Most relevant to our work, [Abe et al.](#page-8-2) [\(2022b\)](#page-8-2) demonstrate a strong correlation between ensemble variance and the expected improvement that results from increasing model capacity. Specifically, across numerous architectures and datasets, they demonstrate a strong point-wise correlation between the predictions of an ensemble (e.g., 4 ResNet-18s) and a single larger model (e.g., a WideResNet-18 with <sup>4</sup>× the width) on both in-distribution and out-of-distribution data. The authors conclude that ensemble variance is more reflective of sensitivity to model capacity rather than data availability, a finding with significant implications for decision-making and robustness. We theoretically verify these findings in ensembles of overparameterized random feature models.

Random feature models. The connection between infinitely wide neural networks and kernel methods, particularly Gaussian processes, was pioneered by [Neal & Neal](#page-10-8)

[\(1996\)](#page-10-8) and [Williams](#page-11-4) [\(1996\)](#page-11-4). Building on these ideas, random feature (RF) models were later introduced as a scalable approximation to kernel machines [\(Rahimi & Recht,](#page-10-9) [2007;](#page-10-9) [2008a](#page-10-10)[;b\)](#page-10-11). RF regressors have seen growing theoretical interest as simplified models of neural networks (e.g. [Belkin](#page-9-12) [et al.,](#page-9-12) [2018;](#page-9-12) [2019;](#page-9-13) [Jacot et al.,](#page-9-14) [2018;](#page-9-14) [Bartlett et al.,](#page-9-15) [2020;](#page-9-15) [Mei & Montanari,](#page-10-12) [2022;](#page-10-12) [Simon et al.,](#page-10-13) [2024\)](#page-10-13). Random feature models can be interpreted as neural networks where only the last layer is trained (e.g. [Rudi & Rosasco,](#page-10-14) [2017;](#page-10-14) [Belkin et al.,](#page-9-13) [2019\)](#page-9-13) or as first-order Taylor approximations of neural networks (e.g. [Jacot et al.,](#page-9-14) [2018\)](#page-9-14).

Underparameterized random feature models and ensembles. In this paragraph, we restrict our discussion to analyses of (ensembles of) underparameterized RF regressors, where the number of random features (i.e., the width) is assumed to be far fewer than the number of data points. In the fixed design setting, infinite ensembles of unregularized RF regressors achieve the same generalization error as ridge regression on the original (unprojected) inputs [\(Kaban´](#page-9-4) , [2014;](#page-9-4) [Thanei et al.,](#page-10-1) [2017;](#page-10-1) [Bach,](#page-9-16) [2024b\)](#page-9-16). We provide theoretical analysis in Appx. [E](#page-31-0) that further demonstrates ridge-like behaviour of underpameterized RF ensembles.

Overparameterized random feature models. Recent works on RF models have focused on the *overparameterized regime*, often using high-dimensional asymptotics to characterize generalization error [\(Adlam & Pennington,](#page-8-3) [2020;](#page-8-3) [Bach,](#page-9-16) [2024b;](#page-9-16) [Hastie et al.,](#page-9-17) [2022;](#page-9-17) [Loureiro et al.,](#page-10-15) [2022;](#page-10-15) [Mei](#page-10-12) [& Montanari,](#page-10-12) [2022;](#page-10-12) [Ruben et al.,](#page-10-16) [2024\)](#page-10-16). Many works rely on results derived assuming that the the marginal distributions over the random features can be replaced by momentmatched Gaussians. While such approximations are wellfounded for asymptotic results (e.g. [Goldt et al.,](#page-9-18) [2022;](#page-9-18) [Hu &](#page-9-19) [Lu,](#page-9-19) [2022;](#page-9-19) [Montanari & Saeed,](#page-10-17) [2022;](#page-10-17) [Tao,](#page-10-18) [2012\)](#page-10-18), we argue that they may be harmful specifically for an analysis which aims to characterize the uncertainty properties of ensemble variance. Assuming Gaussianity results in an ensemble variance that is proportional to the predictive variance of Gaussian process regression, often held as a gold standard for uncertainty quantification [\(Rasmussen & Williams,](#page-10-19) [2006;](#page-10-19) [Lee et al.,](#page-10-20) [2018;](#page-10-20) [2020;](#page-10-21) [Ovadia et al.,](#page-10-5) [2019\)](#page-10-5). In contrast, our non-Gaussian analysis yields a characterization of ensemble variance that differs from this conventional notion of uncertainty, closely matching recent empirical studies of ensemble variance [\(Abe et al.,](#page-8-2) [2022b;](#page-8-2) [Theisen et al.,](#page-11-3) [2024\)](#page-11-3).

The benefits of overparameterization and ensembling for out-of-distribution generalization in random feature models have been analyzed by [Hao et al.](#page-9-20) [\(2024\)](#page-9-20), who provide lower bounds on OOD risk improvements when increasing capacity or using ensembles. Their work focuses on nonasymptotic guarantees under specific distributional shifts, while ours examines the equivalence of ensembles and sin-

![](_page_2_Figure_1.jpeg)

Figure 1: An infinite ensemble of overparameterized RF models is equivalent to a single infinite-width RF model. (Left) We show a sample of 100 finite-width RF models (blue) with ReLU activations trained on the same N = 6 data points. Additionally, we show the single infinite-width RF model (pink). The finite-width predictions concentrate around the infinite-width model. (Right) We again show the single infinite-width RF model (pink) and the "infinite" ensemble of M = 10, 000 RF models (blue). We note no perceptible difference between the two in this setting, though extreme numerical conditions can break this equivalence (cf. Fig. [8\)](#page-14-0).

gle large models under minimal assumptions. Concurrent work by [Ruben et al.](#page-10-16) [\(2024\)](#page-10-16) also finds RF ensembles offer little advantage over larger single models, though their analysis uses optimal ridge tuning and Gaussian universality assumptions. Most related to our work is [Jacot et al.](#page-9-21) [\(2020\)](#page-9-21), who analyze the pointwise expectation and variance of ridgeregularized RF models with Gaussian process (GP) features, leveraging Gaussianity to simplify their analysis. We go beyond this prior work by significantly weakening the assumptions on the distribution of random features, enabling us to characterize differences between ensembles versus Gaussian models with respect to uncertainty and robustness properties. Moreover, we provide a finite-sample analysis as well as a characterization of the transition from the ridgeless to ridge-regularized regimes, which—to the best of our knowledge—are novel results for overparameterized RF ensembles.

### 1.2. Contributions

We consider ensembles of *overparameterized* RF regressors in both the ridgeless and small ridge regimes. Unlike prior work, we make minimal assumptions about the distribution of the random features and so our results are not restricted to high-dimensional asymptotics where Gaussian universality might typically apply. Our results thus distinguish differences between RF ensembles and more traditional uncertainty-aware models like Gaussian processes. Concretely, we make the following contributions:

To answer Question 1: we show that the average ridgeless RF regressor is pointwise equivalent to its corresponding ridgeless kernel regressor (Theorem [3.2\)](#page-5-0), implying that an infinite ensemble of overparameterized RF models is *exactly*

equivalent to a single infinite-width RF model (cf. Fig. [1\)](#page-2-0). We further show that this equivalence approximately holds in the small ridge regime (Theorem [3.5\)](#page-7-0). Moreover, we extend these results to a finite parameter budget, showing that the functional difference between the parameters of a larger single model and a finite ensemble, each with the same total number of parameters, is small with high probability (see Sec. [3.2\)](#page-5-1). We validate these theoretical results with supporting experiments on RF and neural network ensembles, using synthetic data and the California Housing dataset [\(Kelley Pace & Barry,](#page-9-22) [1997\)](#page-9-22) with various activation functions (detailed in Appx. [A.1](#page-12-0) and Appx. [B\)](#page-20-0).

To answer Question 2: we show that the predictive variance in an overparameterized ensemble generally does not have a frequentist or Bayesian interpretation, unlike uncertainty quantifications obtained from Gaussian processes. Instead, we find that the variance measures the expected squared difference between the predictions from a (finite-width) RF regressor and its corresponding kernel regressor (i.e., the infinite-width model) (see Sec. [3.3\)](#page-5-2). Crucially, this finding relies on our non-Gaussian analysis of RF models.

Altogether, these results support recent empirical findings that deep ensembles offer few generalization and uncertainty quantification benefits over larger single models [\(Abe](#page-8-2) [et al.,](#page-8-2) [2022b;](#page-8-2) [Theisen et al.,](#page-11-3) [2024\)](#page-11-3). Our theory and experiments demonstrate that these phenomena are not specific to neural networks or Gaussian models but are more general properties of ensembles in the overparameterized regime.

#### 2. Setup

We work in a regression setting. The training dataset D <sup>=</sup> {(x<sup>i</sup> , yi)} N <sup>i</sup>=1 ∈ (X × <sup>R</sup>) <sup>N</sup> is a *fixed* set of size N. The vector <sup>y</sup> ∈ <sup>R</sup> <sup>N</sup> is the concatenation of all training responses.

We consider *RF models* adhering to the form hW(x) = √ D P<sup>D</sup> <sup>i</sup>=1 ϕ(ω<sup>i</sup> , x)θ<sup>i</sup> , where θ<sup>i</sup> are learned parameters, W <sup>=</sup> {<sup>ω</sup>i} D <sup>i</sup>=1 ∈ <sup>Ω</sup> <sup>D</sup> are i.i.d. draws from some distribution <sup>π</sup>(·), and <sup>ϕ</sup> : Ω×X → <sup>R</sup> is a *feature extraction function*. In the case of a ReLU-based RF model with p-dimensional inputs, we have X = Ω = <sup>R</sup> p and ϕ(ω<sup>i</sup> , x) = max(0, ω<sup>⊤</sup> <sup>i</sup> x). Though RF models cannot fully explain the behaviour of neural networks (e.g. [Ghorbani et al.,](#page-9-23) [2019;](#page-9-23) [Li et al.,](#page-10-22) [2021;](#page-10-22) [Pleiss & Cunningham,](#page-10-23) [2021\)](#page-10-23), they can be a useful proxy for understanding the effects of overparameterization and capacity on generalization (e.g. [Belkin et al.,](#page-9-13) [2019;](#page-9-13) [Adlam](#page-8-3) [& Pennington,](#page-8-3) [2020;](#page-8-3) [Mallinar et al.,](#page-10-24) [2022\)](#page-10-24).

Notation. For any x, x′ ∈ X , we denote the second moment of the feature extraction function <sup>ϕ</sup>(ω, ·) as <sup>k</sup>(x, x′ ) = <sup>E</sup>ω[ϕ(ω, x)ϕ(ω, x′ )], which is a positive definite kernel function. We use the matrix K := [k(x<sup>i</sup> , x<sup>j</sup> )]ij ∈ <sup>R</sup> N×N for the kernel function applied to all training data pairs and the matrix <sup>Φ</sup><sup>W</sup> := [ϕ(ω<sup>j</sup> , xi)]ij ∈ <sup>R</sup> <sup>N</sup>×<sup>D</sup> for the feature extraction function applied to all data/feature combinations. In this notation, [·]ij refers to the entry in the <sup>i</sup>-th row and j-th column; if one index is omitted (e.g., [v]<sup>j</sup> ), it refers to the j-th element of a row- or column-vector, depending on the context. We drop the subscript W when the set of random features is clear from context. Furthermore, we assume that K is invertible.

Throughout our analysis, it will be useful to consider the "whitened" features <sup>W</sup> <sup>=</sup> <sup>R</sup>−⊤<sup>Φ</sup> ∈ <sup>R</sup> <sup>N</sup>×<sup>D</sup> where R⊤R = K is the Cholesky decomposition of the kernel matrix K. When considering a test point x <sup>∗</sup> ∈ X (or equivalently a set of test points), we extend the K, R, Φ, W notation by

$$\begin{bmatrix} K & [k(x_i, x^*)_i]_i \\ [k(x^*, x_j)_j] & k(x^*, x^*) \end{bmatrix} = \begin{bmatrix} R & c \\ 0 & r_\perp \end{bmatrix}^\top \begin{bmatrix} R & c \\ 0 & r_\perp \end{bmatrix}, \quad (1)$$

$$\begin{bmatrix} W \\ w_\perp \end{bmatrix} = \begin{bmatrix} R & c \\ 0 & r_\perp \end{bmatrix}^{-\top} \begin{bmatrix} \Phi \\ [\phi(\omega_i, x^*)_i] \end{bmatrix}.$$

For fixed training/test points, <sup>E</sup><sup>W</sup> [WW<sup>⊤</sup>] = <sup>D</sup> · <sup>I</sup>, <sup>E</sup>w<sup>⊥</sup> [w ⊤ <sup>⊥</sup>w⊥] = D and <sup>E</sup>W,w<sup>⊥</sup> [w ⊤ <sup>⊥</sup>W<sup>⊤</sup>] = 0 which can be directly derived from <sup>E</sup>Φ[ΦΦ<sup>⊤</sup>] = <sup>D</sup> · <sup>K</sup> (and similar properties for ϕ ∗ , the vector of feature evaluations at x ∗ , i.e., [ϕ(ω<sup>j</sup> , x<sup>∗</sup> )]<sup>j</sup> ). Moreover, the columns [w<sup>i</sup> ; w⊥<sup>i</sup> ] of [W; w⊥] are i.i.d. since they are affine transformations of the i.i.d. columns of Φ.

Overparameterized ridge/ridgeless regressors and ensembles. As our focus is the overparameterized regime, we assume a computational budget of D > N features

(W <sup>=</sup> {<sup>ω</sup>1, . . . , ωD} ∼ <sup>π</sup> <sup>D</sup>) to construct an RF regressor <sup>h</sup>W(x) = √ D ϕW(x) <sup>⊤</sup>θ. We train the regressor parameters <sup>θ</sup> to minimize the loss ∥ <sup>√</sup> D <sup>Φ</sup>W<sup>θ</sup> − <sup>y</sup>∥ <sup>2</sup> <sup>+</sup> <sup>λ</sup>∥θ∥ 2 for some ridge parameter <sup>λ</sup> ≥ <sup>0</sup>. When λ > <sup>0</sup> this optimization problem admits the closed-form solution θ (RR) <sup>W</sup>,λ = √ D Φ ⊤ W D · <sup>Φ</sup>W<sup>Φ</sup> ⊤ <sup>W</sup> <sup>+</sup> λI<sup>−</sup><sup>1</sup> y. Although the learning problem is underspecified when λ = 0 (i.e. in the ridgeless case), the implicit bias of (stochastic) gradient descent initialized at zero leads to the minimum norm interpolating solution θ (LN) <sup>W</sup> = √ 1 D (Φ)<sup>⊤</sup> D · ΦΦ<sup>⊤</sup> <sup>−</sup><sup>1</sup> y. We denote the resulting ridge(less) regressors as h (LN) <sup>W</sup> (·) := <sup>√</sup> 1 D <sup>ϕ</sup>(ω<sup>j</sup> , ·) j θ (LN) <sup>W</sup> , and h (RR) <sup>W</sup>,λ (·) := √ D <sup>ϕ</sup>(ω<sup>j</sup> , ·) j θ (RR) <sup>W</sup>,λ .

We also consider ensembles of M ridge(less) regressors. We assume that each is trained on a different set of i.i.d. D > N random features W1, . . . , W<sup>M</sup> ∼ <sup>π</sup> <sup>D</sup> but trained on the same training set. Thus, the only source of randomness in these ensembles comes from the random selection of features W<sup>i</sup> , analogous to the standard training procedure of deep ensembles [\(Lakshminarayanan et al.,](#page-10-4) [2017\)](#page-10-4). The ensemble prediction is given by the arithmetic average of the individual models <sup>h</sup>¯W1:<sup>M</sup> (·) = <sup>1</sup> M P<sup>M</sup> <sup>m</sup>=1 h (LN) W<sup>m</sup> (·).

Assumptions. A key difference between this paper and prior literature is the set of assumptions about the random feature distribution <sup>π</sup>(·). Most prior works assume that entries in the extended whitened feature matrix [W; w⊥] are i.i.d. draws from standard normal distribution (e.g. [Adlam](#page-8-3) [& Pennington,](#page-8-3) [2020;](#page-8-3) [Jacot et al.,](#page-9-21) [2020;](#page-9-21) [Mei & Montanari,](#page-10-12) [2022;](#page-10-12) [Simon et al.,](#page-10-13) [2024\)](#page-10-13) implying that ϕ(ω<sup>i</sup> , ·) are draws from a Gaussian process with covariance k. [<sup>1</sup>](#page-3-0) While Gaussianity is appropriate in high-dimensional asymptotics, it essentially reduces analysis about the ensemble distribution to a statement about Gaussian processes. A major focus of this work is to differentiate ensembles from Gaussian processes with regards to uncertainty quantification.

Even if we were to relax the Gaussian assumption to a sub-Gaussian assumption, (as done by [Bartlett et al.,](#page-9-15) [2020;](#page-9-15) [Bach,](#page-9-24) [2024a\)](#page-9-24), the distribution of random features will still not accurately reflect common neural network features if the entries of [W; w⊥] are assumed to be i.i.d. For instance, consider ReLU features. If X ⊆ <sup>R</sup> <sup>p</sup> with p < N, the function max(ω <sup>⊤</sup>x, 0) can be fully specified by a p-dimensional random variable. Thus, knowing N evaluations of ω ⊤ <sup>j</sup> x<sup>i</sup> allows one to infer ω<sup>j</sup> , making w<sup>⊥</sup> deterministic given W. We instead consider the following less restrictive assumptions on the distribution of random feature functions <sup>π</sup>(·):

<sup>1</sup> If the entries of W, w<sup>⊥</sup> are i.i.d. Gaussian, then the i th feature applied to train/test inputs ([R <sup>⊤</sup>wi; c <sup>⊤</sup>wi+r⊥w⊥i]) is multivariate Gaussian. This fact holds for any train/test data; thus the i th feature is a GP by definition (e.g. [Rasmussen & Williams,](#page-10-19) [2006,](#page-10-19) Ch. 2).

![](_page_4_Figure_1.jpeg)

Figure 2: Overparameterized ensembles are equivalent to a single infinite-width model regardless of feature distribution, while underparameterized ensembles behave differently. We present the average absolute difference between large ensembles of models with D features versus a single large (or infinite) width model. (Left) RF ensembles with softplus activations, N = 12, using the California Housing dataset [\(Kelley Pace & Barry,](#page-9-22) [1997\)](#page-9-22). (Right) Neural network ensembles with ReLU activations, N = 12, 000, on the same dataset. The shaded region shows the standard deviation. Both exhibit a "hockey stick"-like pattern, less pronounced for neural networks, where the difference between underparameterized ensembles and the large model is substantial, but diminishes for D > N.

Assumption 2.1 (Assumption of subexponentiality).

- 1. wiw⊥<sup>i</sup> (where w<sup>i</sup> is the i th column of W) is subexponential ∀<sup>i</sup> ∈ {1, ..., D} and
- 2. P<sup>D</sup> <sup>i</sup>=1 wiw ⊤ i is a.s. positive definite for any <sup>D</sup> ≥ <sup>N</sup>.

The first condition of Assumption [2.1](#page-3-1) ensures that the whitened random features do not have excessively "heavy tails", meaning their values are well-concentrated. This is a mild condition, satisfied if the individual feature components w<sup>i</sup> and w⊥<sup>i</sup> are sub-Gaussian (but potentially dependent), which is true if the features come from activation functions with bounded derivatives and sub-Gaussian weights. The second condition is equivalent to Φ having almost surely full rank, which is not true for ReLUs and leaky-ReLUs features but which is true for arbitrarily precise approximations thereof.[<sup>2</sup>](#page-4-0) Note we make no assumptions about the mean or independence of the entries in a given column of [W; w⊥].

## 3. Main results

#### 3.1. Equivalence of Infinite Ensembles and the Infinite-Width Single Models

We at first assume an infinite computational budget and consider the following two limiting predictors, for which we will show pointwise equivalence in predictions:

- 1. An infinite-width least norm predictor, h (LN) ∞ , the a.s.

limit of h (LN) <sup>W</sup> as |W| <sup>=</sup> <sup>D</sup> → ∞

limit of 
$$h_{\mathcal{W}}^{(\text{LN})}$$
 as  $|\mathcal{W}| = D \rightarrow \infty$ 

- 2. An infinite ensemble of finite-width least norm predictors, h¯ (LN) <sup>∞</sup> , which is the almost sure limit of h¯ (LN) W1:<sup>M</sup> as <sup>M</sup> → ∞, with N < D < ∞ remaining constant.

These limiting predictors not only approximate large ensembles and very large single models but also help characterize the variance and generalization error of finite overparameterized ensembles, as discussed in Sec. [3.3.](#page-5-2)

Define <sup>k</sup><sup>N</sup> (·) : X → <sup>R</sup> <sup>N</sup> as the vector of kernel evaluations with the training data <sup>k</sup><sup>N</sup> (·) = <sup>k</sup>(x1, ·) · · · <sup>k</sup>(x<sup>N</sup> , ·) ⊤ . As <sup>D</sup> → ∞, the minimum norm interpolating model converges pointwise almost surely to the ridgeless kernel regressor by the Strong Law of Large Numbers:

$$h_{\mathcal{W}}^{(\text{LN})}(\cdot) \xrightarrow{\text{a.s.}} h_{\infty}^{(\text{LN})}(\cdot), \quad h_{\infty}^{(\text{LN})}(\cdot) := k_N(\cdot)^{\top} K^{-1} y.$$

On the other hand, using W and w<sup>⊥</sup> as introduced in Sec. [2](#page-3-2) we can rewrite the infinite ensemble prediction h¯ (LN) <sup>∞</sup> (x ∗ ) as (for a derivation of this, see Appx. [C.1\)](#page-20-1)

$$\begin{aligned} \bar{h}_\infty^{(LN)}(x^*) &= h_\infty^{(LN)}(x^*) \\ &+ r_\perp \mathbb{E}_{W, w_\perp} \left[ w \perp^\top W^\top (WW^\top)^{-1} \right] R^{-\top} y \quad (2) \end{aligned}$$

To prove the pointwise equivalence of the infinite ensemble and infinite-width single model, we need to show that <sup>E</sup>W,w<sup>⊥</sup> [w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 ] term in Eq. [\(2\)](#page-4-1) is zero. Note that this result trivially holds when the entries of W and w<sup>⊥</sup> are i.i.d. and zero mean, as assumed in prior work (e.g. [Jacot](#page-9-21) [et al.,](#page-9-21) [2020\)](#page-9-21). In the following lemma, we show that this

<sup>2</sup>E.g., ϕα(ω, x) = <sup>1</sup> α log(1 + e αω⊤x ), α > 0 yields an a.s. full-rank Φ, and ϕα(ω, x) <sup>α</sup>→∞→ ReLU(<sup>ω</sup> <sup>⊤</sup>x).

term is zero even when w<sup>⊥</sup> and W are dependent, which as described in Sec. [2—](#page-3-3)is a more realistic assumption for neural network features:

Lemma 3.1. *Under Assumption [2.1,](#page-3-1) it holds that* <sup>E</sup>W,w<sup>⊥</sup> [w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 ] = 0*.*

(Proof: see Appx. [C.1.](#page-22-0)) Combining Lemma [3.1](#page-5-3) and Eq. [\(2\)](#page-4-1) yields the pointwise equivalence of h¯ (LN) <sup>∞</sup> and h (LN) ∞ :

Theorem 3.2 (Equivalence of infinite-width single model and infinite ensembles). *Under Assumption [2.1,](#page-3-1) the infinite ensemble of finite-width (but overparameterized) RF regressors* h¯ (LN) ∞ *is pointwise almost surely equivalent to the (single) infinite-width RF regressor* h (LN) ∞ *.*

Theorem [3.2](#page-5-0) implies that ensembling overparameterized RF models yields *exactly* the same predictions as simply increasing the capacity of a single RF model, regardless of the RF distribution (see Fig. [1](#page-2-0) for a visualization). Note that this result significantly generalizes prior characterizations of overparameterized RF models that have relied on Gaussianity assumptions or asymptotic analyses (e.g. [Adlam &](#page-8-3) [Pennington,](#page-8-3) [2020;](#page-8-3) [Jacot et al.,](#page-9-21) [2020\)](#page-9-21), demonstrating that the ensemble/infinite-single model equivalence is a fundamental property of overparameterization. Consequently, we should not expect substantial differences in generalization between large single models and overparameterized ensembles, consistent with recent empirical findings by [\(Abe et al.,](#page-8-2) [2022b;](#page-8-2) [2024;](#page-8-1) [Theisen et al.,](#page-11-3) [2024\)](#page-11-3).

We emphasize a contrast with the underparameterized regime, where RF ensembles match the generalization error of kernel ridge regression (see Appx. [E](#page-31-0) or [Bach,](#page-9-16) [2024b,](#page-9-16) Sec. 10.2.2). Width controls the implicit ridge parameter in the underparameterized regime (see Sec. [1.1\)](#page-1-0), whereas width does not affect the ensemble predictor in the overparameterized regime. We confirm this difference in Fig. [2](#page-4-2) which shows that RF ensembles are close to the ridgeless kernel regressor when D > N but not when D < N. The figure also illustrates similar behaviours when comparing deep ensembles versus single large neural networks.

#### 3.2. Ensembles versus Larger Single Models under a Finite Parameter Budget

For modest parameter budgets (where our asymptotic results are not applicable), we compare whether ensembles are more parameter efficient than larger single models. Specifically, given access to MD random features, we compare ensembles of M models each of which use D of the features (h¯ (LN) W1:<sup>M</sup> = M P<sup>M</sup> <sup>m</sup>=1 h (LN) W<sup>m</sup> ) against a single model that uses all MD features h (LN) <sup>W</sup><sup>∗</sup> (·) (i.e., here |Wm| <sup>=</sup> <sup>D</sup> for all <sup>m</sup> and |W<sup>∗</sup> | <sup>=</sup> MD).

First, we provide a non-asymptotic theorem showing that h¯ (LN) W1:<sup>M</sup> and h (LN) <sup>W</sup><sup>∗</sup> behave similarly, with their difference becoming negligible as the number of features per ensemble member increases (for a formal version, see Appx. [C.2\)](#page-23-0).

Theorem 3.3 (Non-asymptotic difference between ensembles and single models (informal version)). *Under slightly stronger assumptions than Assumption [2.1,](#page-3-1) the* L<sup>2</sup> *difference between a single neural network with* MD *features and an ensemble of* M *neural networks each with* D *features is, with probability* <sup>1</sup> − <sup>δ</sup>*, upper bounded by:*

$$\left\| h_{\mathcal{W}^*}^{(\text{LN})}(\cdot) - \bar{h}_{\mathcal{W}_{1:M}}^{(\text{LN})}(\cdot) \right\|_2^2 \leq O(\sqrt{\log(1/\delta)}) + O(1/D)$$

Theorem [3.3](#page-5-4) is supported through a standard bias-variance decomposition of risk:

$$\begin{aligned}\mathbb{E}_h [L(h)] &:= \mathbb{E}_h [\mathbb{E}_x [(h(x) - \mathbb{E}[y \mid x])^2] \\ &= L(\mathbb{E}_h [h]) + \mathbb{E}_x [\mathbb{V}_h(h(x))].\end{aligned}\tag{3}$$

Since h (LN) <sup>W</sup><sup>∗</sup> and h (LN) W<sup>1</sup> , . . . , h(LN) W<sup>M</sup> share the same expected predictor (as established in Theorem [3.2\)](#page-5-0), the only difference in the generalization of h (LN) <sup>W</sup><sup>∗</sup> and <sup>h</sup>¯ (LN) W1:<sup>M</sup> arises from their variances. Due to the independence between ensemble members, we have that <sup>V</sup>W1:<sup>M</sup> [h¯ (LN) W1:<sup>M</sup> (x)] = 1 <sup>M</sup> <sup>V</sup>Wm[h (LN) W<sup>m</sup> (x)]. Moreover, prior works such as [\(Adlam](#page-8-3) [& Pennington,](#page-8-3) [2020\)](#page-8-3) and empirical results (see Appx. [A.3\)](#page-15-0) suggest the variance of a single RF model is inversely proportional to the number of features.[<sup>3</sup>](#page-5-5) As a consequence, we have that <sup>V</sup>W<sup>∗</sup> [h (LN) <sup>W</sup><sup>∗</sup> (x)] : <sup>V</sup>Wm[h (LN) W<sup>m</sup> ] ≍ <sup>1</sup>/M, further suggesting that the generalization of ensembles and single models should be similar under the same parameter budget. Fig. [3](#page-6-0) (left) and Appx. [A.3](#page-15-0) empirically confirm that RF ensembles versus single RF models obtain similar generalization under fixed feature budgets. Moreover, Fig. [3](#page-6-0) (right) depicts a similar trend for neural networks: deep ensembles perform roughly the same as larger single models under a fixed parameter budget. These results show that ensembles offer no meaningful generalization advantage over (large) single models and, since the arguments hold for any test distribution, align with empirical findings [\(Abe et al.,](#page-8-2) [2022b\)](#page-8-2) that ensembles provide no additional robustness benefits.

#### 3.3. Implications for Uncertainty Quantification

We now analyze the predictive variance amongst component models in an overparameterized RF ensemble, a quantity often used to quantify predictive uncertainty in safety-critical applications [\(Lakshminarayanan et al.,](#page-10-4) [2017\)](#page-10-4). Before diving in to a mathematical characterization, it is worth reflecting on the qualitative characterization based on our existing results. Because the expected overparameterized RF model

<sup>3</sup>This rate is exact for Gaussian features and approximate for the general case.

![](_page_6_Figure_1.jpeg)

Figure 3: The generalization error of overparameterized ensembles and single large models scales similarly with the total number of features. We present the generalization error of ensembles compared to single large models of the same type with equivalent total parameter budgets. Both exhibit nearly identical dependence on the total feature budget. Left: RF ensembles/models, N = 12, ReLU activations, ensembles of models with D = 200 features. Right: neural networks ensembles/models, N = 12, 000, ensembles of three-layer MLPs with width 256 in each layer. The shaded region shows the standard deviation.

is the infinite-width model, predictive variance is equal to

$$\begin{aligned}\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)] &= \mathbb{E}_{\mathcal{W}} \left[ (h_{\mathcal{W}}^{(\text{LN})}(x^*) - \underbrace{\mathbb{E}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)]}_{\bar{h}_{\infty}}(x^*))^2 \right] \\ &= \mathbb{E}_{\mathcal{W}} \left[ (h_{\mathcal{W}}^{(\text{LN})}(x^*) - h_{\infty}^{(\text{LN})}(x^*))^2 \right],\end{aligned}$$

i.e. the expected difference between finite- versus infinitewidth RF model predictions. In other words, *ensemble variance quantifies how predictions change if we increase model capacity.* This characterization, which holds for all random feature distributions satisfying Assumption [2.1,](#page-3-1) is not a standard frequentist or Bayesian notion of uncertainty except under specific distributional assumptions.

Uncertainty quantification under Gaussian features. Using Theorem [3.2,](#page-5-0) the variance of the predictions of a single RF model with respect to its random features can be expressed as (see Appx. [C.3](#page-27-0) for a derivation)

$$\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)] = r_{\perp}^2 \left( y^{\top} R^{-1} \mathbb{E}_{W, w_{\perp}} [(WW^{\top})^{-T} W w_{\perp}] \right. \\ \left. \cdot w_{\perp}^{\top} W^{\top} (WW^{\top})^{-1} \right] R^{-\top} y \Big). \quad (4)$$

In the special case where W and w<sup>⊥</sup> are i.i.d. standard normal, this expression simplifies to

$$\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)] = r_{\perp}^2 \left( \frac{\|h_{\infty}^{(\text{LN})}\|_k^2}{D-N-1} \right), \quad (5)$$

where ∥<sup>h</sup> (LN) <sup>∞</sup> ∥ 2 <sup>K</sup> represents the squared norm of h (LN) ∞ in the RKHS defined by the kernel <sup>k</sup>(·, ·). From this equation, we see that <sup>V</sup>W[h (LN) <sup>W</sup> (x ∗ )] only depends on x ∗ through the quantity r 2 <sup>⊥</sup>, which by Eq. [\(1\)](#page-3-2) is equal to

$$r_{\perp}^2 = k(x^*, x^*) - k_N(x^*)^{\top} K^{-1} k_N(x^*).$$

We recognize this quantity as the Gaussian process posterior variance with prior covariance <sup>k</sup>(·, ·) (e.g. [Rasmussen &](#page-10-19) [Williams,](#page-10-19) [2006\)](#page-10-19). Thus, with Gaussian features, ensemble variance admits a Bayesian interpretation in addition to the model capacity interpretation. In other words, Gaussianity assumptions justify the use of overparameterized ensemble variance in uncertainty quantification tasks.

Uncertainty quantification under general features. Unfortunately, this Bayesian interpretation explicitly does not carry over to the general Assumption [2.1](#page-3-1) case. Although Eq. [\(4\)](#page-6-1) still holds for general feature distributions, it does not have a simple expression unless W and w<sup>⊥</sup> are independent. The variance depends on x ∗ through both r 2 <sup>⊥</sup> as well as through a complicated expectation involving W and w⊥. In Appx. [C.3](#page-27-1) we demonstrate with a simple example that this expectation can indeed depend on x ∗ , implying that ensemble variance does not correspond to a scalar multiple of r 2 <sup>⊥</sup> (i.e. the Gaussian process posterior variance).

In numerical experiments using ReLU random features, (Fig. [4](#page-7-1) and Appx. [A.3\)](#page-15-0), we observe significant deviations between the ensemble variance and the Gaussian process posterior variance, further suggesting that one cannot view ensembles through a classic framework of uncertainty. These discrepancies are particularly important for uncertainty estimation in safety-critical applications or active learning (e.g. [Gal et al.,](#page-9-8) [2017;](#page-9-8) [Beluch et al.,](#page-9-25) [2018\)](#page-9-25), as the only meaningful interpretation of ensemble variance (the expected change in prediction from increasing capacity) may not yield reliability guarantees or useful exploration-exploitation tradeoffs.

![](_page_7_Figure_1.jpeg)

Figure 4: RF ensemble variance (left) and Bayesian notions of uncertainty (right) can differ significantly. For N = 6 and D = 200 with ReLU activations, the overparameterized ensemble variance (left) and the posterior variance of a Gaussian process with prior covariance <sup>k</sup>(·, ·) (right) differ substantially across the input range.

#### 3.4. Equivalence of the Limiting Predictors in the Small Ridge Regime

Having established the equivalence between infinite ensembles and infinite-width single models in the ridgeless regime, we now investigate whether this equivalence approximately persists in the practically relevant setting when a small ridge regularization parameter λ > 0 is introduced. More generally, we aim to determine whether the transition from the ridgeless case to the small ridge regime is smooth. While h (RR) <sup>∞</sup>,λ , the infinite-width limit of h (RR) <sup>W</sup>,λ as |W| <sup>=</sup> <sup>D</sup> → ∞, almost surely converges to the kernel ridge regressor with ridge λ, the infinite ensemble h¯ (RR) <sup>∞</sup>,λ := <sup>E</sup>W[h (RR) <sup>W</sup>,λ (x)] does not generally maintain pointwise equivalence with h (RR) <sup>∞</sup>,λ . This divergence occurs even under Gaussianity assumptions [\(Jacot et al.,](#page-9-21) [2020\)](#page-9-21). However, we hypothesize that the difference between these limiting predictors is small when λ is close to zero, which is common in practical applications. To analyze this regime, we introduce a minor additional assumption, which is weaker than Gaussianity:

Assumption 3.4. We assume that <sup>E</sup>W[ ΦWΦ ⊤ W <sup>−</sup><sup>1</sup> ] is finite for all |W| <sup>=</sup> D > N.

Under Assumptions [2.1](#page-3-1) and [3.4,](#page-7-2) we show that the difference between ridge-regularized ensembles and single models is Lipschitz-continuous with respect to λ (proof in Appx. [D.1\)](#page-29-0).

Theorem 3.5 (The difference between ensembles and large single models is smooth with respect to λ.). *Under Assumptions [2.1](#page-3-1) and [3.4,](#page-7-2) the difference* |h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ )| *between the infinite ensemble and the single infinite-width model trained with ridge* <sup>λ</sup> ≥ <sup>0</sup> *is Lipschitz-continuous in* <sup>λ</sup>*. The Lipschitz constant is independent of* x ∗ *for compact* X *.*

This result is illustrated in Fig. [5,](#page-8-4) where the terms bounding the difference evolve smoothly with λ. To the best of

our knowledge, this Lipschitz-continuity has not been established even under Gaussianity assumptions. We note that the bound by [Jacot et al.](#page-9-21) [\(2020,](#page-9-21) Thm. 4.1), which characterizes the difference between ridge ensembles and infinite models with Gaussian random features, becomes vacuous as <sup>λ</sup> → <sup>0</sup>. Since Theorem [3.2](#page-5-0) ensures that |h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ )| = 0 for <sup>λ</sup> = 0, we can conclude that the pointwise difference grows at most linearly with λ. Specifically, we have that

$$\left| \bar{h}_{\infty,\lambda}^{(RR)} - h_{\infty,\lambda}^{(RR)}(x) \right| \leq C \cdot \lambda,$$

for some constant C independent of x ∗ , provided that X is compact. In practical terms, this result indicates that for sufficiently small values of λ, the predictions of large ensembles and large single models remain nearly indistinguishable, reinforcing our findings from the ridgeless regime.

## 4. Conclusion

This work characterized overparameterized RF ensembles and contextualized theoretical findings with neural network experiments. We used weaker distributional assumptions than prior work to (a) more faithfully approximate realworld models and (b) highlight differences between ensembles versus models with Gaussian behaviour.

For *Question 1*, we demonstrated under weak conditions that infinite ensembles and single infinite-width models are pointwise equivalent in the ridgeless regime (Theorem [3.2\)](#page-5-0) and nearly identical with a small ridge (Theorem [3.5\)](#page-7-0), significantly expanding on prior results. We further provide a non-asymptotic characterization, showing that ensembles and large single models with the same parameter budget are nearly equivalent (Theorem [3.3\)](#page-5-4). These results verify recent empirical findings (e.g. [Abe et al.,](#page-8-2) [2022b\)](#page-8-2) that much

![](_page_8_Figure_1.jpeg)

Figure 5: Lipschitz continuity of predictions for an infinite ensemble and kernel regressor with respect to the ridge parameter. (Left) We plot |h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup>¯ (LS) <sup>∞</sup> (x ∗ )| as a function of <sup>λ</sup> for 500 test points. (Right) We show the evolution of |<sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (LS) <sup>∞</sup> (x ∗ )| for the same test points. Both plots use ReLU activation functions and the California Housing Dataset with <sup>N</sup> = 12 and <sup>D</sup> = 200. While the direct difference |h¯ (RR) <sup>∞</sup>,λ (x ∗ )−<sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ )| is not shown (for reasons outlined in Appx. [A.4\)](#page-19-0), it is bounded by the sum of the plotted quantities (see Appx. [D.1\)](#page-28-0). The evolution of these plotted bounding terms thus illustrates that this direct difference is Lipschitz continuous in λ (proven in Theorem [3.5\)](#page-7-0) and converges to zero as <sup>λ</sup> → <sup>0</sup> (a consequence of Theorem [3.5](#page-7-0) and the ridgeless equivalence established in Theorem [3.2\)](#page-5-0).

of the benefit attributed to overparameterized ensembles, such as improved predictive performance and robustness, can be explained by their similarity to larger single models. Notably, our analysis does not rely on Gaussianity, emphasizing that these phenomena are fundamental properties of overparameterized models and not artifacts of specific feature assumptions.

In contrast, for *Question 2*, we found that uncertainty interpretations of ensemble variance are contingent on Gaussianity assumptions and fall apart under more general feature distributions. We characterize ensemble variance as the expected difference to a single larger model, which only corresponds to a (scaled) Bayesian notion of uncertainty under strong independence assumptions. With more realistic feature distributions, the ensemble variance does not correspond to any conventional notion of uncertainty, reinforcing recent empirical findings on the limitations of ensemble uncertainty quantification [\(Abe et al.,](#page-8-2) [2022b\)](#page-8-2). This deviation supplies further evidence that caution is needed when using ensembles in safety-critical settings.

Overall, while our results do not contradict the utility of overparameterized ensembles, they suggest that their benefits may often be explained by their similarity to larger models and that further research is needed to improve uncertainty quantification methods.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### Acknowledgements

GP is supported by Canada CIFAR AI Chairs. JPC is supported by the Gatsby Charitable Foundation (GAT3708), the Simons Foundation (542963), the NSF AI Institute for Artificial and Natural Intelligence (ARNI: NSF DBI 2229929) and the Kavli Foundation. We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC: RGPIN-2024-06405). Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

## References


[1] Abe, T., Buchanan, E. K., Pleiss, G., and Cunningham, J. P. The best deep ensembles sacrifice predictive diversity. In *I Can't Believe It's Not Better Workshop: Understanding Deep Learning Through Empirical Falsification*, 2022a. Abe, T., Buchanan, E. K., Pleiss, G., Zemel, R., and Cunningham, J. P. Deep ensembles work, but are they necessary? In *Advances in Neural Information Processing Systems*, 2022b. Abe, T., Buchanan, E. K., Pleiss, G., and Cunningham, J. P. Pathologies of predictive diversity in deep ensembles. *Transactions on Machine Learning Research*, 2024. Adlam, B. and Pennington, J. Understanding double de-

[2] scent requires a fine-grained bias-variance decomposition. In *Advances in Neural Information Processing Systems*, 2020. Angelova, J. A. On moments of sample mean and variance. *Int. J. Pure Appl. Math*, 79(1):67–85, 2012. Bach, F. High-dimensional analysis of double descent for linear regression with random projections. *SIAM Journal on Mathematics of Data Science*, 6(1):26–50, 2024a. Bach, F. *Learning Theory from First Principles*. MIT Press, 2024b. Bartlett, P. L., Long, P. M., Lugosi, G., and Tsigler, A. Benign overfitting in linear regression. *Proceedings of the National Academy of Sciences*, 117(48):30063–30070, 2020. Belkin, M., Ma, S., and Mandal, S. To understand deep learning we need to understand kernel learning. In *International Conference on Machine Learning*, pp. 541–549, 2018. Belkin, M., Hsu, D., Ma, S., and Mandal, S. Reconciling modern machine-learning practice and the classical bias– variance trade-off. *Proceedings of the National Academy of Sciences*, 116(32):15849–15854, 2019. Beluch, W. H., Genewein, T., Nurnberger, A., and Kohler,

[3] J. M. The power of ensembles for active learning in image classification. In *Computer Vision and Pattern Recognition*, 2018. Breiman, L. Bagging predictors. *Machine Learning*, 24(2): 123–140, 1996. Breiman, L. Random forests. *Machine Learning*, 45(1): 5–32, 2001. Chen, L., Lukasik, M., Jitkrittum, W., You, C., and Kumar, S. On bias-variance alignment in deep models. In *International Conference on Learning Representations*, 2024. Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. In *International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016. Cho, Y. and Saul, L. Kernel methods for deep learning. In *Advances in Neural Information Processing Systems*, 2009. Dietterich, T. G. Ensemble methods in machine learning. In *International Workshop on Multiple Classifier Systems*, pp. 1–15. Springer, 2000. Fort, S., Hu, H., and Lakshminarayanan, B. Deep ensembles: A loss landscape perspective. *arXiv preprint arXiv:1912.02757*, 2019. Freund, Y. Boosting a weak learning algorithm by majority. *Information and Computation*, 121(2):256–285, 1995. Gal, Y., Islam, R., and Ghahramani, Z. Deep Bayesian active learning with image data. In *International Conference on Machine Learning*, pp. 1183–1192, 2017. Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A. Limitations of lazy training of two-layers neural network. In *Advances in Neural Information Processing Systems*, 2019. Goldt, S., Loureiro, B., Reeves, G., Krzakala, F., Mezard, ´ M., and Zdeborova, L. The gaussian equivalence of gen- ´ erative models for learning with shallow neural networks. In *Mathematical and Scientific Machine Learning*, pp. 426–471, 2022. Gustafsson, F. K., Danelljan, M., and Schon, T. B. Evaluating scalable bayesian deep learning methods for robust computer vision. In *Computer Vision and Pattern Recognition Workshops*, pp. 318–319, 2020. Hansen, L. and Salamon, P. Neural network ensembles. *Pattern Analysis and Machine Intelligence*, 12(10):993– 1001, 1990. Hao, Y., Lin, Y., Zou, D., and Zhang, T. On the benefits of over-parameterization for out-of-distribution generalization. *arXiv preprint arXiv:2403.17592*, 2024. Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J. Surprises in high-dimensional ridgeless least squares interpolation. *Annals of Statistics*, 50(2):949, 2022. Hu, H. and Lu, Y. M. Universality laws for high-dimensional learning with random features. *IEEE Transactions on Information Theory*, 69(3):1932–1964, 2022. Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel: Convergence and generalization in neural networks. *Advances in Neural Information Processing Systems*, 2018. Jacot, A., Simsek, B., Spadaro, F., Hongler, C., and Gabriel,
  - F. Implicit regularization of random feature models. In *International Conference on Machine Learning*, 2020. Jeffares, A., Liu, T., Crabbe, J., and van der Schaar, M. Joint ´ training of deep ensembles fails due to learner collusion. In *Advances in Neural Information Processing Systems*, 2024. Kaban, A. New bounds on compressive linear least squares ´ regression. In *Artificial Intelligence and Statistics*, pp. 448–456, 2014. Kelley Pace, R. and Barry, R. Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3):291–297, 1997.

[4] Kendall, A. and Gal, Y. What uncertainties do we need in Bayesian deep learning for computer vision? In *Advances in Neural Information Processing Systems*, 2017. Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems*, 2017. Lee, J., Bahri, Y., Novak, R., Schoenholz, S., Pennington, J., and Sohl-dickstein, J. Deep neural networks as gaussian processes. In *International Conference on Learning Representations*, 2018. Lee, J., Schoenholz, S., Pennington, J., Adlam, B., Xiao, L., Novak, R., and Sohl-Dickstein, J. Finite versus infinite neural networks: an empirical study. In *Advances in Neural Information Processing Systems*, volume 33, pp. 15156–15172, 2020. Lee, S., Purushwalkam, S., Cogswell, M., Crandall, D., and Batra, D. Why m heads are better than one: Training a diverse ensemble of deep networks. *arXiv preprint arXiv:1511.06314*, 2015. Li, M., Nica, M., and Roy, D. The future is log-Gaussian: ResNets and their infinite-depth-and-width limit at initialization. *Advances in Neural Information Processing Systems*, 34:7852–7864, 2021. Loureiro, B., Gerbelot, C., Refinetti, M., Sicuro, G., and Krzakala, F. Fluctuations, bias, variance & ensemble of learners: Exact asymptotics for convex losses in highdimension. In *International Conference on Machine Learning*, pp. 14283–14314. PMLR, 2022. Mallinar, N., Simon, J. B., Abedsoltan, A., Pandit, P., Belkin, M., and Nakkiran, P. Benign, tempered, or catastrophic: A taxonomy of overfitting. In *Advances in Neural Information Processing Systems*, 2022. Mei, S. and Montanari, A. The generalization error of random features regression: Precise asymptotics and the double descent curve. *Communications on Pure and Applied Mathematics*, 75(4):667–766, 2022. Mentch, L. and Hooker, G. Quantifying uncertainty in random forests via confidence intervals and hypothesis tests. *Journal of Machine Learning Research*, 17(26): 1–41, 2016. Montanari, A. and Saeed, B. N. Universality of empirical risk minimization. In *Conference on Learning Theory*, pp. 4310–4312, 2022. Neal, R. M. and Neal, R. M. Priors for infinite networks. *Bayesian learning for neural networks*, pp. 29–53, 1996. Nixon, J., Lakshminarayanan, B., and Tran, D. Why are bootstrapped deep ensembles not better? In *NeurIPS "I Can't Believe It's Not Better!" Workshop*, 2020. Opitz, D. and Maclin, R. Popular ensemble methods: An empirical study. *Journal of Artificial Intelligence Research*, 11:169–198, 1999. Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J., Lakshminarayanan, B., and Snoek, J. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In *Advances in Neural Information Processing Systems*, 2019. Pleiss, G. and Cunningham, J. P. The limitations of large width in neural networks: A deep Gaussian process perspective. *Advances in Neural Information Processing Systems*, 34:3349–3363, 2021. Rahimi, A. and Recht, B. Random features for large-scale kernel machines. In *Advances in Neural Information Processing Systems*, 2007. Rahimi, A. and Recht, B. Uniform approximation of functions with random bases. In *46th Annual Allerton Conference on Communication, Control, and Computing*, pp. 555–561, 2008a. Rahimi, A. and Recht, B. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. In *Advances in Neural Information Processing Systems*, 2008b. Rasmussen, C. E. and Williams, C. K. *Gaussian Processes for Machine Learning*. MIT Press, 2006. Ruben, B. S., Tong, W. L., Chaudhry, H. T., and Pehlevan,
  - C. No free lunch from random feature ensembles. *arXiv preprint arXiv:2412.05418*, 2024. Rudi, A. and Rosasco, L. Generalization properties of learning with random features. *Advances in Neural Information Processing Systems*, 2017. Simon, J. B., Karkada, D., Ghosh, N., and Belkin, M. More is better in modern machine learning: when infinite overparameterization is optimal and overfitting is obligatory. In *International Conference on Learning Representations*, 2024. Tao, T. *Topics in Random Matrix Theory*. American Mathematical Soc., 2012. Thanei, G.-A., Heinze, C., and Meinshausen, N. Random projections for large-scale regression. *Big and Complex Data Analysis: Methodologies and Applications*, pp. 51– 68, 2017.

[5] Theisen, R., Kim, H., Yang, Y., Hodgkinson, L., and Mahoney, M. W. When are ensembles really effective? *Advances in Neural Information Processing Systems*, 36, 2024. Wager, S., Hastie, T., and Efron, B. Confidence intervals for random forests: The jackknife and the infinitesimal jackknife. *Journal of Machine Learning Research*, 15(1): 1625–1651, 2014. Wainwright, M. J. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019. Webb, A., Reynolds, C., Chen, W., Reeve, H., Iliescu, D., Lujan, M., and Brown, G. To ensemble or not ensemble: When does end-to-end training fail? In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part III*, pp. 109–

123. Springer, 2021. Williams, C. K. I. Computing with infinite networks. In *Advances in Neural Information Processing Systems*, 1996. Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J. Y., Levine, S., Finn, C., and Ma, T. MOPO: Model-based offline policy optimization. In *Advances in Neural Information Processing Systems*, 2020.

[7] ![](_page_12_Figure_1.jpeg)

[8] Figure 6: True function <sup>f</sup>(x) = sin(5 · <sup>b</sup> <sup>⊤</sup>x) with different random seeds. The blue line shows the true function, while red dots represent training samples for two distinct random seeds.

[9] In the appendix, we will provide the following additional results:

1. In Appx. [A,](#page-12-1) we will describe our experimental setup for RF models in more detail, difficulties we encountered when developing the experiments, and provide the results of additional experiments.

2. In Appx. [B,](#page-20-0) we will describe our experimental setup for neural network models in more detail, and provide the results of additional experiments.

3. In Appx. [C](#page-20-2) we will give the proofs for Secs. [3.1](#page-4-3) and [3.3](#page-5-2) in the main paper.

4. In Appx. [D](#page-28-1) we will give the proofs for Sec. [3.4](#page-7-3) in the main paper.

5. Finally, in Appx. [E,](#page-31-0) we prove (under mild assumptions) that infinite underparameterized RF ensembles are equivalent to kernel ridge regression under some transformed kernel.

[15] The code to run all our experiments can be found on GitHub: [https://github.com/nic-dern/](https://github.com/nic-dern/theoretical-limitations-overparameterized-ensembles) [theoretical-limitations-overparameterized-ensembles](https://github.com/nic-dern/theoretical-limitations-overparameterized-ensembles). It contains a README.md file that explains how to set up and run the experiments.
## A. Experimental Setup and Additional Results for RF Models

### A.1. Experimental Setup

We had two setups using which we performed most of our experiments:

- 1. We generate training and test points uniformly at random from [−5, 5]<sup>d</sup> using the function <sup>f</sup>(x) = sin(5 · <sup>b</sup> <sup>⊤</sup>x), where b is a vector (depending on the random seed) and the noise parameter is σ = 0.05 (we assume Gaussian noise with mean 0). In this setting, we use N = 6, D = 200, and data from R (i.e., d = 1) if not specified otherwise. You can find a plot of an example true function in Fig. [6.](#page-12-2)
- 2. We use the California Housing [\(Kelley Pace & Barry,](#page-9-22) [1997\)](#page-9-22) dataset and sample distinct training and test points from it (randomly permutating the dataset initially). In this setting, we use N = 12, D = 200 if not differently specified. The data dimension is R <sup>8</sup> here. In contrast to the first setting, we employ a data normalization using a max-min normalization *on the entire dataset* since we experimentally found this makes our methods more stable.

We calculate the generalization error using N = 1000 test points in both settings. In the first setting, we calculate the variance of the predictions of a single model using M = 20, 000 models, while in the second setting, we use M = 4, 000 models. Apart from Fig. [12](#page-17-0) where we use 100, 000 samples, "infinite" ensembles consist of M = 10, 000 models.

![](_page_13_Figure_1.jpeg)

Figure 7: Visualization of hyperplanes separating training points. We illustrate how a series of hyperplanes can separate a growing subset of the training points, leading to a triangular, invertible matrix structure as a subset of Φ.

As distribution <sup>τ</sup> (·) of the elements <sup>ω</sup><sup>i</sup> ∈ W we always use N (0, I). As activation functions, we use ReLU, the Gaussian error function, and the softplus function <sup>1</sup> β · log(1 + exp(<sup>β</sup> · <sup>ω</sup> <sup>⊤</sup>x)) with β = 1. For the first two activation functions, there exist analytically calculatable limiting kernels, the arc-cosine kernel [\(Cho & Saul,](#page-9-26) [2009\)](#page-9-26) and the erf-kernel [\(Williams,](#page-11-4) [1996\)](#page-11-4). The closed forms for these are

$$k_{\text{arc-cosine}}(x, x') = \frac{1}{2\pi} \|x\| \|x'\| (\sin \theta + (\pi - \theta) \cos \theta),$$

where θ = cos−<sup>1</sup> x <sup>⊤</sup>x ∥x∥∥x′∥ and

$$k_{\text{eff}}(x, x') = \frac{2}{\pi} \sin^{-1} \left( \frac{2x^\top x'}{\sqrt{(1+2\|x\|^2)(1+2\|x'\|^2)}} \right)$$

.

For the softplus function, we approximate the kernel by estimating the second moment k(x, x′ ) = <sup>E</sup>[ϕ(ω, x)ϕ(ω, x′ ) | x, x′ of the feature extraction using 10<sup>7</sup> samples from <sup>τ</sup> (·). For sampling Gaussian features, we use the same approach as described by [Jacot et al.](#page-9-21) [\(2020\)](#page-9-21).

Before training on data, we always append a 1 in the zeroeth-dimension of the data before calculating the dot product with ω (correspondingly, the dimension of ω is d + 1) and applying the activation function. In the ridgeless case, we use λ = 10−<sup>8</sup> to avoid numerical issues.

### A.2. Notes on Stability

During our experiments, we encountered challenges related to both mathematical stability (i.e., matrices being truly singular rather than nearly singular) and numerical stability. This section outlines these issues and describes the steps we took to mitigate them.

Most importantly, the matrix ΦWΦ ⊤ <sup>W</sup> is not almost surely invertible when using the ReLU activation function, meaning that technically, the second condition of our Assumption [2.1](#page-3-1) is not fulfilled. In numerical experiments, this results in cases where (ΦWΦ ⊤ <sup>W</sup>) −1 is nearly singular (though stabilized with λ = 10−<sup>8</sup> ).

On the other hand, when D is sufficiently large relative to N, Φ<sup>W</sup> is full rank with high probability, which implies that ΦWΦ ⊤ <sup>W</sup> is invertible with high probability. Given our data transformation of appending a 1 in the zeroeth dimension, one can see this as there exists a series of (non-zero probability sets of) hyperplanes separating an increasing subset of the training points, leading to a subset of ΦW's columns that form a triangular, invertible matrix (see Fig. [7](#page-13-0) for a visualization). Intuitively, higher data dimensionality and better separability of the points increase the probability of Φ<sup>W</sup> having full rank.

As an example of the discussed instabilities, see the adversarial scenario shown in Fig. [8,](#page-14-0) where N = 15 and many training points are placed very close to each other. In this case, individual RF regressors exhibit relatively high variance output values (due to numerical instabilities), which are not averaged out in the "infinite" ensemble. Similar issues were also observed when using the Gaussian error function as the activation function, although they were generally less pronounced.

![](_page_14_Figure_1.jpeg)

Figure 8: An adversarial example where the infinite ensemble of overparameterized RF models is numerically not equivalent to a single infinite-width RF model. (Left) We show a sample of 100 RF models (blue) with ReLU activations trained on the same N = 15 densely clustered data points. Additionally, we show the single infinite-width RF model (pink). (Right) We again show the single infinite-width RF model (blue) and the "infinite" ensemble of M = 10, 000 RF models (pink). A significant difference between the two models is observed in this adversarial case, indicating instability.

![](_page_14_Figure_3.jpeg)

Figure 9: Using softplus activations instead of ReLU activations reduces instabilities in overparameterized RF ensembles. The plots show the average absolute difference between the predictions of an infinite ensemble and a single infinite-width model for varying feature counts D, using N = 12 training samples from the California Housing dataset. (Left) ReLU activations exhibit significant instability, especially for D > N, D ≈ <sup>N</sup>, and do not consistently show the expected pointwise equivalence between the infinite ensemble and the single infinite-width model. (Right) Softplus activations — as equivalently shown in Fig. [2](#page-4-2) — smooth out these instabilities and more consistently show the expected pointwise equivalence.

To alleviate these issues, we used the following approaches:

- We used a relatively low number of samples, N = 6 or N = 12, compared to D = 200. As shown in Fig. [1,](#page-2-0) even with D = 200, there is still a considerable amount of variance in the RF regressors (i.e., the individual RF regressors are not yet closely approximating the limiting kernel ridge regressor).
- We appended a 1 in the zeroeth dimension of the data before calculating the dot product with ω.
- We performed additional experiments using the softplus function with β = 1 as a smooth approximation of the ReLU activation function. This often helped stabilize the numerical computations, as seen in Fig. [9,](#page-14-1) where we repeated a part of the experiment from Fig. [2](#page-4-2) using the ReLU function as activation function which increased the numerical instability for low D values.
- We used a ridge term λ = 10−<sup>8</sup> in the ridgeless case to stabilize the inversion of ΦWΦ ⊤
  - <sup>W</sup>.
- We used *double precision* for all computations and used the torch.linalg.lstsq function with the driver gelsd (for not-well-conditioned matrices) to solve linear systems.
- We applied max-min normalization to the entire California Housing dataset to improve stability.

### A.3. Additional Experiments for the Ridgeless Case

To address the question of whether our findings are specific to normally distributed weights ω<sup>i</sup> for the feature generating function, we supplement Fig. [1.](#page-2-0) Fig. [10](#page-15-1) replicates that visualization using weights ω<sup>i</sup> drawn from a Uniform(-10, 10) distribution and the softplus activation function. As can be seen, the equivalence between the infinite ensemble of overparameterized RF models and the single infinite-width RF model remains apparent, and no perceptible difference is observed.

![](_page_15_Figure_5.jpeg)

(a) Sample of 100 finite-width RF models (Uniform ωi)

![](_page_15_Figure_6.jpeg)

(b) Infinite ensemble (Uniform ωi) vs. infinite-width model

Figure 10: Replication of Fig. [1](#page-2-0) with Uniformly Distributed Weights ω<sup>i</sup> . Similar to Fig. [1,](#page-2-0) we demonstrate the equivalence of an infinite ensemble of overparameterized RF models to a single infinite-width RF model. Here, the weights ω<sup>i</sup> for the Softplus activation functions are drawn from a Uniform(-10, 10) distribution. (Left) A sample of 100 finite-width RF models (blue) trained on N = 6 data points, with the single infinite-width RF model (pink). (Right) The infinite-width RF model (pink) and the "infinite" ensemble of M = 10, 000 RF models (blue). No perceptible difference is observed, mirroring the findings with normally distributed weights.

Furthermore, to illustrate the convergence of finite ensembles to the infinite-width model prediction as the number of ensemble members M increases, Fig. [11](#page-16-0) expands on the setting of Fig. [1.](#page-2-0) It shows that even with a small number of ensemble members, the average prediction begins to concentrate around the infinite-width model, and this concentration improves as M grows.

![](_page_16_Figure_1.jpeg)

Figure 11: Evolution of Ensemble Predictions with Increasing Number of Members (M). Following the setup of Fig. [1](#page-2-0) (ReLU activations, normally distributed ω<sup>i</sup> , N = 6 data points), these plots show 10 sample ensemble predictions (blue lines) for varying ensemble sizes M. The single infinite-width RF model is shown in pink. As M increases, the ensemble predictions become more concentrated around the infinite-width model.

Additional experiments on the identity of infinite-width single model and infinite ensembles. In Fig. [12,](#page-17-0) we show that the term <sup>E</sup>[w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 ] is consistently zero for both ReLU and the Gaussian error function activations consistetly with Lemma [3.1.](#page-5-3) To further demonstrate that this result is not dependent on Gaussian-like weight distributions, Fig. [13](#page-17-1) shows this for a softplus activation function, with weights ω<sup>i</sup> drawn from a Uniform(-10, 10) distribution and a Laplace(0, 1) distribution. The expectation of the term remains centered at zero, supporting the generality of our theoretical findings.

Additional experiments on the ensemble variance. We observed a different behavior of the RF regressor variance and r 2 <sup>⊥</sup> as shown in Fig. [4](#page-7-1) consistently across different random seeds and dimensions for both ReLU and the Gaussian error function activations as activation functions. In Fig. [14,](#page-18-0) we present additional examples for the Gaussian error function in one dimension and the ReLU activation in two dimensions.

Additional experiment on generalization error and variance scaling. In Fig. [3,](#page-6-0) the generalization error decay for the ReLU activation function. To verify the consistency of this trend, we repeated the experiment using the Gaussian error function and the corresponding erf-kernel. The result is very similar, shown in Fig. [15.](#page-18-1) Furthermore, this figure shows that the variance of a single model with MD features decays as ∼ MD , matching the ensemble's behavior.

![](_page_17_Figure_1.jpeg)

Figure 12: Empirically, the term <sup>E</sup>[w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 ] is consistently zero. We plot the distribution of the first index of w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 , which captures the difference between the infinite-width single model and a smaller overparameterized RF model (see Eq. [\(2\)](#page-4-1)). (Left) We use ReLU as activation function, <sup>x</sup><sup>i</sup> ∈ <sup>R</sup>, and <sup>N</sup> = 6, D = 200. (Right) We use the Gaussian Error activation function, the California Housing dataset and N = 12, D = 200.

![](_page_17_Figure_3.jpeg)

Figure 13: Empirical validation of <sup>E</sup>[w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 ] ≈ <sup>0</sup> for Softplus activation and non-Gaussian weights. Similar to Fig. [12,](#page-17-0) we plot the distribution of the first index of w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 . Both plots use a softplus activation function, the California Housing dataset, and N = 12, D = 200. (Left) Weights ω<sup>i</sup> are drawn from a Uniform(-10, 10) distribution. (Right) Weights ω<sup>i</sup> are drawn from a Laplace distribution with location 0 and scale 1. In both cases, the distribution is centered at zero.

![](_page_18_Figure_1.jpeg)

Figure 14: Variance and r <sup>⊥</sup> for different activations and dimensions. (Top left) Variance of RF model predictions across the input range for D = 200 and N = 6, using the erf activation function. (Top right) Corresponding r 2 <sup>⊥</sup> values across the input range using the erf kernel. (Bottom left) Variance of RF model predictions across the input range for D = 200, p = 2, and N = 12, using the ReLU activation function. (Bottom right) Corresponding r 2 <sup>⊥</sup> values across the input range using the arc-cosine kernel.

![](_page_18_Figure_3.jpeg)

Figure 15: Variance and generalization error scale similarly with the number of features, consistent with Fig. [3.](#page-6-0) In *(a)*, the variance of a single model with MD features decays as ∼ 1 MD , matching the ensemble's behavior. In *(b)*, the generalization error of an ensemble with M models and D = 200 features shows a similar decay to that of a single model with MD features. Results use the Gaussian error function, California Housing dataset, and N = 12.

![](_page_19_Figure_1.jpeg)

Figure 16: Empirically, the term <sup>E</sup>W,w<sup>⊥</sup> h w ⊤ ⊥W<sup>⊤</sup> WW<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · <sup>λ</sup> · <sup>R</sup>−⊤R−<sup>1</sup> <sup>−</sup><sup>1</sup> i is consistently zero. We show the empirical distribution of an index of w ⊤ ⊥W<sup>⊤</sup> WW<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · <sup>λ</sup> · <sup>R</sup>−⊤R−<sup>1</sup> <sup>−</sup><sup>1</sup> ∈ <sup>R</sup> <sup>N</sup> , which captures the difference in predictions between c <sup>⊤</sup>EW,w<sup>⊥</sup> h WW<sup>⊤</sup> WW<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · <sup>λ</sup> · <sup>R</sup>−⊤R−<sup>1</sup> <sup>−</sup><sup>1</sup> i R−⊤y and a finite-sized overparameterized RF model (see Eq. [\(7\)](#page-22-0)). We use <sup>λ</sup> = 1.<sup>0</sup> in both plots. (Left) We use a ReLU activation function, <sup>x</sup><sup>i</sup> ∈ <sup>R</sup>, and <sup>N</sup> = 6, D = 200. (Right) We use the Gaussian Error Function as activation function, the California Housing dataset, and N = 12, D = 200.

#### A.4. More Experiments for the Ridge Case

Additional experiments for the convergence of the expected value term. In Appx. [D,](#page-28-1) we show that a variant of Lemma [3.1](#page-5-3) also holds in the ridge case. More precisely, we show that

$$\mathbb{E}_{W, w_\perp} \left[ w_\perp^\top W^\top (WW^\top + D \cdot \lambda \cdot R^{-\top} R^{-1})^{-1} \right] = 0$$

under Assumption [2.1.](#page-3-1) We repeated the experiment from Fig. [12](#page-17-0) for the ridge case to verify this experimentally. The results are shown in Fig. [16.](#page-19-1)

Additional notes. In Fig. [5,](#page-8-4) we illustrate the Lipschitz continuity of the predictions for an infinite ensemble and a kernel regressor with respect to the ridge parameter. Rather than directly presenting the difference h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ ) , we show the evolution of h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup>¯ (LS) <sup>∞</sup> (x ∗ ) and h (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (LS) <sup>∞</sup> (x ∗ ) . This choice was made because the upper bound we obtained was not consistently tight for settings with large D. In particular, the pointwise predictions of the infinite ensemble h¯ (RR) <sup>∞</sup>,λ and the single infinite-width model h (RR) <sup>∞</sup>,λ trained with ridge λ were already very close for non-zero λ. We opted to display the upper bounds rather than the direct difference to avoid cherry-picking favorable settings.

Our best explanation for this phenomenon is that infinite ensembles under Assumption [2.1](#page-3-1) in the ridge regime often behave similarly to the single infinite-width model h (RR) <sup>∞</sup>,λ˜ with an *implicit ridge* parameter <sup>λ</sup>˜, which solves the equation

$$\tilde{\lambda} = \lambda + \frac{\tilde{\lambda}}{D} \sum_{i=1}^N \frac{d_i}{\tilde{\lambda} + d_i}$$

where d<sup>i</sup> are the eigenvalues of the kernel matrix K, as shown by [Jacot et al.](#page-9-21) [\(2020\)](#page-9-21) under Gaussianity. Intuitively and empirically, for large D, the implicit ridge λ˜ tends to be very close to the true ridge λ. Using Lemma [D.1,](#page-28-0) this suggests that for small values of λ, the difference between the infinite ensemble and the infinite-width single model h (RR) <sup>∞</sup>,λ with ridge λ is already minimal before λ approaches zero.

Interestingly, our findings (see Fig. [2\)](#page-4-2) suggest that in the ridgeless case, the similarity to the ridge regressor with the implicit ridge only holds in the overparameterized regime. Note that this does not violate the results from [Jacot et al.](#page-9-21) [\(2020\)](#page-9-21) since the constants in their bounds blow up as <sup>λ</sup> → <sup>0</sup> in both the underparameterized and overparameterized regimes.

## B. Experimental Setup and Additional Results for Neural Network Models

#### B.1. Experimental Setup

For all our experiments with neural networks, we used a three-layer MLP with hidden layers of equal width and ReLU activations. Models were trained for 1000 epochs using SGD with momentum, a learning rate of 0.01, and a momentum decay of 0.9.

Training was performed on the same set of 12,000 samples from the California Housing dataset, with a validation set of 3,000 samples and a test set of 5,000 samples. Since the number of parameters scales quadratically with the hidden layer width, the overparameterized regime is reached at a width of approximately 80.

All reported results are based on the best checkpoint selected using validation performance over the 1000 training epochs.

### B.2. Additional Results and Limitations of Our Experiments

The results in Fig. [2,](#page-4-2) which show the average absolute difference in predictions between large ensembles with increasing number of parameters (D) in the component models and a single large model (with MD parameters), are further supported by the increasing correlation of residuals between the ensemble and the single model, as shown in Fig. [17.](#page-21-0) When the component models of the ensemble become overparameterized, ensemble residuals align better with those of a single large model, indicating that overparameterized ensembles make more similar errors to a large single model than their underparameterized counterparts.

Furthermore, the correlations of the residuals of large ensembles of overparameterized models and two large single models trained with different initializations were comparably high, with the average correlation between an ensemble and a single model even slightly higher. At the same time, the residual correlation between two large ensembles trained with different initializations was significantly higher than both of these correlations. The lower variance in predictions across multiple ensembles compared to a single large model does not align with our theoretical expectations and experiments with random feature models. We hypothesize that this discrepancy arises from large single models being more unstable to train but did not investigate this limitation of our experiments in more detail.

## C. Proofs for Overparameterized Ridgeless Regression

### C.1. Equivalence of Infinite Ensemble and Infinite Single Model

We start by proving the equivalent formulation of the infinite ensemble prediction stated in Eq. [\(2\)](#page-4-1) using the terms W and w<sup>⊥</sup> as introduced in Sec. [2:](#page-3-2)

*Proof.* Defining ϕ ∗ <sup>W</sup> = [ϕ(ω<sup>i</sup> , x<sup>∗</sup> )]<sup>i</sup> ∈ <sup>R</sup> <sup>D</sup>, we have

$$\begin{aligned}
\bar{h}_\infty(x^*) &= \mathbb{E}_W \left[ \frac{1}{D} \phi_W^* \Phi_W^\top \left( \frac{1}{D} \cdot \Phi_W \Phi_W^\top \right)^{-1} \right] y \\
&= \mathbb{E}_{W, w_\perp} \left[ (c^\top W + r_\perp w_\perp^\top) W^\top R (R^\top W W^\top R)^{-1} \right] y \\
&= \mathbb{E}_{W, w_\perp} \left[ (c^\top W + r_\perp w_\perp^\top) W^\top (W W^\top)^{-1} \right] R^{-\top} y \\
&= c^\top R^{-\top} y + r_\perp \mathbb{E}_{W, w_\perp} \left[ w_\perp^\top W^\top (W W^\top)^{-1} \right] R^{-\top} y, \tag{6}
\end{aligned}$$

where c, R, r<sup>⊥</sup> are as defined in Eq. [\(1\)](#page-3-2). The left term in Eq. [\(6\)](#page-20-3) is equal to h (LN) <sup>∞</sup> (x):

$$h_\infty^{(\text{LN})}(x^*) = [k(x_i, x^*)]_{i=1}^N K^{-1} y = c^\top R R^{-1} R^{-\top} y = c^\top R^{-\top} y.$$

![](_page_21_Figure_1.jpeg)

Figure 17: Correlation of residuals between ensembles and a single large model. Scatter plots comparing the residuals of a single large model with MD parameters to those of ensembles with increasing component parameters (shown by the the width of the component models): (Top left) 20, (Top right) 50, (Bottom left) 80, and (Bottom right) 110. The correlation of residuals increases as the component parameter count grows. This suggests that overparameterized ensembles make more similar errors to a single large model than (strongly) underparameterized ensembles.

In the case of λ > 0, we can similarly see that

$$\begin{aligned}
\bar{h}_{\infty,\lambda}^{(RR)}(x^*) &= \mathbb{E}_W \left[ \frac{1}{D} \phi_W^* \Phi_W^\top \left( \frac{1}{D} \cdot \Phi_W \Phi_W^\top + \lambda I \right)^{-1} \right] y \\
&= \mathbb{E}_{W,w_\perp} \left[ (c^\top W + r_\perp w_\perp^\top) W^\top R (R^\top W W^\top R + D \cdot \lambda \cdot R^\top R^{-\top} R^{-1} R)^{-1} \right] y \\
&= \mathbb{E}_{W,w_\perp} \left[ (c^\top W + r_\perp w_\perp^\top) W^\top (W W^\top + D \cdot \lambda \cdot R^{-\top} R^{-1})^{-1} \right] R^{-\top} y \\
&= c^\top \mathbb{E}_{W,w_\perp} \left[ W W^\top (W W^\top + D \cdot \lambda \cdot R^{-\top} R^{-1})^{-1} \right] R^{-\top} y \\
&+ r_\perp \mathbb{E}_{W,w_\perp} \left[ w_\perp^\top W^\top (W W^\top + D \cdot \lambda \cdot R^{-\top} R^{-1})^{-1} \right] R^{-\top} y.
\end{aligned} \tag{7}$$

Note that the simplification demonstrated in Eq. [\(2\)](#page-4-1) does not work as nicely in the underparameterized case (<sup>D</sup> ≤ <sup>N</sup>). This is because the weights, in this case, are given by θ = (Φ<sup>⊤</sup> <sup>W</sup>ΦW) <sup>−</sup><sup>1</sup>Φ ⊤ <sup>W</sup>y, and thus the infinite ensemble prediction expands as:

$$\begin{aligned}\bar{h}_\infty(x^*) &= \mathbb{E}_W \left[ \phi_W^* (\Phi_W^\top \Phi_W)^{-1} \Phi_W^\top \right] y \\ &= \mathbb{E}_{W, w_\perp} \left[ (c^\top W + r_\perp w_\perp^\top) (W^\top R R^\top W)^{-1} W^\top R \right] y.\end{aligned}$$

Here, RR<sup>⊤</sup> lies inside the inverse, preventing the simplifications available in the overparameterized regime.

Next up, we show that the expected value <sup>E</sup>W,w<sup>⊥</sup> h w ⊤ ⊥W<sup>⊤</sup> WW<sup>⊤</sup> <sup>−</sup><sup>1</sup> i is zero under Assumption [2.1.](#page-3-1) This directly implies the pointwise equivalence of the infinite ensemble and the single infinite-width model (see Theorem [3.2\)](#page-5-0).

*Lemma [3.1](#page-5-3) (Restated).* Under Assumption [2.1,](#page-3-1) it holds that <sup>E</sup>W,w<sup>⊥</sup> [w ⊤ <sup>⊥</sup>W<sup>⊤</sup>(WW<sup>⊤</sup>) −1 ] = 0.

*Proof.* Define <sup>A</sup>−<sup>i</sup> = (WW<sup>⊤</sup> − <sup>w</sup>i<sup>w</sup> ⊤ i ). Note that A−<sup>1</sup> is almost surely invertible and positive definite by assumption Assumption [2.1.](#page-3-1)

By the Woodbury formula, for almost every WW<sup>⊤</sup> we have that

$$(WW^\top)^{-1} = (A_{-i} + w_i w_i^\top)^{-1} = A_{-i}^{-1} - \frac{A_{-i}^{-1} w_i w_i^\top A_{-i}^{-1}}{1+w_i^\top A_{-i}^{-1} w_i},$$

which implies that

$$\begin{aligned} w_{\perp}^{\top} W^{\top} (WW^{\top})^{-1} &= \sum_{i=1}^D w_{\perp i} w_i^{\top} \left( A_{-i}^{-1} - \frac{A_{-i}^{-1} w_i w_i^{\top} A_{-i}^{-1}}{1+w_i^{\top} A_{-i}^{-1} w_i} \right) \\ &= \sum_{i=1}^D w_{\perp i} \left( w_i^{\top} \frac{A_{-i}^{-1} w_i A_{-i}^{-1} w_i^{\top} A_{-i}^{-1} w_i}{1+w_i^{\top} A_{-i}^{-1} w_i} - \frac{w_i^{\top} A_{-i}^{-1} w_i w_i^{\top} A_{-i}^{-1}}{1+w_i^{\top} A_{-i}^{-1} w_i} \right) \\ &= \sum_{i=1}^D \frac{w_{\perp i} w_i^{\top}}{1+w_i^{\top} A_{-i}^{-1} w_i} A_{-i}^{-1}. \end{aligned}$$

For any positive definite matrix <sup>B</sup> ∈ <sup>R</sup> <sup>N</sup>×<sup>N</sup> and any vector <sup>v</sup> ∈ <sup>R</sup> <sup>N</sup> ; ∥v∥ = 1 and any <sup>i</sup> ∈ {1, ..., D}, we have

$$\begin{aligned} \left| \mathbb{E}_{w_{\perp i}, w_i} \left[ \frac{w_{\perp i} w_i^\top}{1+w_i^\top B w_i} \right] v \right| &\leq \mathbb{E}_{w_{\perp i}, w_i} \left[ \left\| \frac{w_{\perp i} w_i^\top v}{1+w_i^\top B w_i} \right\| \right] \\ &= \int_0^\infty \mathbb{P} \left[ \left\| \frac{w_{\perp i} w_i^\top v}{1+w_i^\top B w_i} \right\| \geq t \right] dt \\ &= \int_0^\infty \mathbb{P} \left[ |w_{\perp i} w_i^\top v| \geq (1+w_i^\top B w_i) t \right] dt \\ &\leq \int_0^\infty \mathbb{P} \left[ |w_{\perp i} w_i^\top| > t \right] dt \\ &\leq \int_0^{\nu^2/\alpha} 2 \exp \left( -\frac{t^2}{2\nu} \right) dt + \int_{\nu^2/\alpha}^\infty 2 \exp \left( -\frac{t}{2\alpha} \right) dt, \end{aligned} \quad (8)$$

where the last inequality is a standard sub-exponential bound applied to w⊥iw<sup>i</sup> . Note that we here use the fact that <sup>E</sup>[w⊥iw ⊤ i ] = 0 and the (ν 2 , α)-sub-exponentiality of w⊥iw ⊤ i .

Since the last two integrals in Eq. [\(8\)](#page-22-1) are finite, the expectation <sup>E</sup>W,w<sup>⊥</sup> (w⊥iw ⊤ i )/(1 + w ⊤ <sup>i</sup> Bwi) v is finite. By the weak law of large numbers, for i.i.d. random variables w (j) i and w (j) ⊥i across different j's, we have

$$\mathbb{P} \left[ \left| \frac{1}{M} \sum_{j=1}^M \frac{w_{\perp i}^{(j)} (w_i^{(j)})^\top v}{1 + (w_i^{(j)})^\top B w_i^{(j)}} - \mathbb{E} W_{, w_{\perp}} \left[ \frac{w_{\perp i} w_i^\top}{1 + w_i^\top B w_i} \right] v \right| > t \right] \rightarrow 0,$$

for any t > <sup>0</sup> and <sup>v</sup> ∈ <sup>R</sup> <sup>N</sup> such that ∥v∥ = 1 as <sup>M</sup> → ∞. At the same time, repeating the sub-exponential argument above, we have that

$$\begin{aligned} \mathbb{P} \left[ \left| \frac{1}{M} \sum_{j=1}^M \frac{w_{\perp i}^{(j)}(w_i^{(j)})^\top v}{1 + (w_i^{(j)})^\top B w_i^{(j)}} \right| > t \right] &\leq \mathbb{P} \left[ \left| \frac{1}{M} \sum_{i=1}^M w_{\perp i}^{(j)} (w_i^{(j)})^\top v \right| > t \right] \\ &\leq \begin{cases} 2 \exp \left( -\frac{Mt^2}{2\nu} \right) & 0 < t \leq \nu^2/\alpha \\ 2 \exp \left( -\frac{Mt}{2\alpha} \right) & t > \nu^2/\alpha \end{cases} \\ &\rightarrow 0 \end{aligned}$$

as <sup>M</sup> → ∞. Here we use the property that the sum of <sup>M</sup> (<sup>ν</sup> , α)-sub-exponential random variables is (Mν<sup>2</sup> , α)-subexponential.

Together, these results imply that <sup>E</sup>W,w<sup>⊥</sup> -(w⊥iw ⊤ i )/(1 + w ⊤ <sup>i</sup> Bwi) = 0 for every positive definite B. Since the random matrix A−<sup>i</sup> is positive semidefinite, almost surely invertible (by the second half of Assumption [2.1\)](#page-3-1), and independent of w<sup>i</sup> , w⊥<sup>i</sup> , we have that

$$\begin{aligned}\mathbb{E}_{w_\perp, W} \left[ w_\perp W^\top (WW^\top)^{-1} \right] &= \sum_{i=1}^D \mathbb{E}_{w_\perp, w_i, A_{-i}} \left[ \frac{w_{\perp i} w_i^\top}{1+w_i^\top A_{-i}^{-1} w_i} A_{-i}^{-1} \right] \\ &= \sum_{i=1}^D \mathbb{E}_{A_{-i}} \left[ \mathbb{E}_{w_\perp, w_i} \left[ \frac{w_{\perp i} w_i^\top}{1+w_i^\top A_{-i}^{-1} w_i} \right] A_{-i}^{-1} \right] = 0.\end{aligned}$$

We remark that this proof equivalently holds for the ridge-regression case, i.e., <sup>E</sup>W,w<sup>⊥</sup> h w ⊤ ⊥W<sup>⊤</sup> WW<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · <sup>λ</sup> · <sup>R</sup>−⊤R−<sup>1</sup> <sup>−</sup><sup>1</sup> i = 0 since the proof does not rely on the specific form of the matrix A−<sup>i</sup> other than it being positive definite. Thus by Eq. [\(7\)](#page-22-0) we directly get that under Assumption [2.1](#page-3-1) it holds that

$$\bar{h}_{\infty,\lambda}^{(RR)}(x^*) = c^\top \mathbb{E}_{W,w_\perp} \left[ WW^\top (WW^\top + D \cdot \lambda \cdot R^{-\top} R^{-1})^{-1} \right] R^{-\top} y. \quad (9)$$

### C.2. Ensembles versus Larger Single Models under a Finite Feature Budget

We now prove the formal version of Theorem [3.3.](#page-5-4)

Let's first restate the informal version of the theorem:

*Theorem [3.3](#page-5-4) (Restated).* Under slightly stronger assumptions than Assumption [2.1,](#page-3-1) the L<sup>2</sup> difference between a single neural network with MD features and an ensemble of <sup>M</sup> neural networks each with <sup>D</sup> features is, with probability <sup>1</sup> − <sup>δ</sup>, upper bounded by:

$$\left\| h_{\mathcal{W}^*}^{(\text{LN})}(\cdot) - \bar{h}_{\mathcal{W}_{1:M}}^{(\text{LN})}(\cdot) \right\|_2^2 \leq O(\sqrt{\log(1/\delta)}) + O(1/D)$$

We now provide the formal version of this theorem, which uses the following definitions:

Definition C.1. Define Σ : L 2 (X ) → L<sup>2</sup> (X ) as <sup>Σ</sup><sup>f</sup> <sup>=</sup> R X <sup>k</sup>(x, ·)f(x)dµ(x).

Definition C.2. For any fixed set of random features W <sup>=</sup> {<sup>ω</sup>1, . . . , ωD} of size <sup>D</sup> define <sup>ϕ</sup>W(x) = √ D [ϕ(ω1, x), . . . , ϕ(ωD, x)]<sup>⊤</sup>, and the approximated kernel function <sup>ˆ</sup>kW(x, ·) = <sup>ϕ</sup>W(x) <sup>⊤</sup>ϕW(·). Using this, define Σˆ<sup>W</sup> : L (X ) → L<sup>2</sup> (X ) as ΣˆW<sup>f</sup> <sup>=</sup> R X <sup>ˆ</sup>kW(x, ·)f(x)dµ(x).

(We will drop the W subscript from <sup>ϕ</sup>W, ˆkW, and Σˆ<sup>W</sup> when the set of random features is clear from context.) Now, we state the assumptions that we need for the proof which are stronger than the assumptions in Assumption [2.1:](#page-3-1)

Assumption C.3. We make the following assumptions:

- The columns of <sup>Φ</sup> are subgaussian with constant <sup>L</sup>, i.e. P(|Xv| ≥ <sup>t</sup>) ≤ 2 exp −t <sup>2</sup>/L<sup>2</sup> for all <sup>v</sup> ∈ <sup>R</sup> <sup>D</sup> with |v| ≤ <sup>1</sup>.
- For any <sup>δ</sup><sup>1</sup> ∈ (0, 1) there exists a C < ∞ and <sup>D</sup>0(δ1) such that for all <sup>D</sup> ≥ <sup>D</sup>0(δ1) it holds that Σˆ − <sup>Σ</sup> ≤ <sup>C</sup> with probability ≥ <sup>1</sup> − <sup>δ</sup>1.
- The feature extraction <sup>ϕ</sup>(ω, ·) is almost surely square integrable over the data probability measure (i.e. <sup>E</sup>x[ϕ(ω, ·) 2 ] < ∞).

Theorem C.4 (Non-asymptotic bound on the L<sup>2</sup> difference between ensembles and single models (informal version)). *Under Assumption [C.3,](#page-24-0) there exist constants* <sup>c</sup>1, c2, c<sup>3</sup> *such that for any* <sup>δ</sup><sup>1</sup> ∈ (0, 1) *and all* M, N, D *with* <sup>M</sup> · <sup>D</sup> ≥ <sup>D</sup>0(δ1) *and defining* λmin := min(1, λmin(K)) *it holds:*

*If* <sup>λ</sup>min <sup>2</sup>·L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>D</sup> + N D o <sup>&</sup>gt; <sup>0</sup> *and* <sup>δ</sup><sup>2</sup> <sup>=</sup> <sup>M</sup> · <sup>c</sup>2<sup>e</sup> <sup>−</sup>c3<sup>D</sup> min(<sup>κ</sup>2<sup>j</sup> ,κ<sup>2</sup> <sup>2</sup><sup>j</sup> ) <sup>+</sup> <sup>c</sup>2<sup>e</sup> <sup>−</sup>c3MD min(<sup>κ</sup>1,κ<sup>2</sup> <sup>1</sup>) <sup>&</sup>lt; <sup>1</sup> − <sup>δ</sup>1*, where* <sup>κ</sup><sup>1</sup> <sup>=</sup> max(0, λmin <sup>2</sup>·L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>M</sup>·<sup>D</sup> + N M·D o ) *and* κ2<sup>j</sup> = max(0, λmin <sup>2</sup>·L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>D</sup> + N D o )*, then for any* <sup>δ</sup><sup>3</sup> ∈ (0, <sup>1</sup> − <sup>δ</sup><sup>1</sup> − <sup>δ</sup>2) *it holds with probability at least* <sup>1</sup> − <sup>δ</sup><sup>1</sup> − <sup>δ</sup><sup>2</sup> − <sup>δ</sup><sup>3</sup> *that the* <sup>L</sup>2*-norm of the difference between the larger, but finite-width single model and the finite ensemble with the same features is bounded by*

$$\left\| h_{\mathcal{W}}^{(\text{LN})}(\cdot) - \bar{h}_{W_{1,M}}(\cdot) \right\|_2^2 \leq \epsilon + O(1/D)$$

*where* ϵ = r λmin log δ<sup>3</sup> *.*

$$\text{where } \epsilon = \sqrt{\frac{1}{\lambda_{min}} \log\left(\frac{2}{\delta_3}\right)}.$$

#### *Proof.*

*First step: Expressing as the difference in their parameter norms.*

In the following, we define <sup>ϕ</sup>(x) = √ MD [ϕ(ω1, x), . . . , ϕ(ωMD, x)]<sup>⊤</sup>. Using this definition we get that with θ (ENS) <sup>=</sup> √ 1 M [θ (ENS) 1 , . . . , θ(ENS) <sup>M</sup> ] <sup>⊤</sup>, where θ (ENS) j are the parameters of the j-th component model, we get ϕ(x) <sup>⊤</sup>θ (ENS) = M P<sup>M</sup> <sup>j</sup>=1 √ D [ϕ(ω(j−1)D, x), . . . , ϕ(ωjD, x)]<sup>⊤</sup>θ (ENS) j . At the same time, we can write ϕ(x) <sup>⊤</sup>θ (Single) = √ 1 MD [ϕ(ω1, x), . . . , ϕ(ωMD, x)]<sup>⊤</sup><sup>θ</sup> (Single) .

Using an equivalence of norm argument, we get E<sup>x</sup> h ϕ(x) <sup>⊤</sup>θ (ENS) − <sup>ϕ</sup>(x) <sup>⊤</sup>θ (Single ) 2 i ≤ <sup>C</sup>upper θ (ENS) − <sup>θ</sup> (Single ) 2 2 .

More precisely, for this argument, Cupper will be the operator norm of the matrix <sup>E</sup>x[ϕ(x)ϕ(x) <sup>⊤</sup>]. Since this is a positive semi-definite matrix, the operator norm is equal to its largest eigenvalue.

We can now define the operator T : L 2 (X ) → <sup>R</sup> <sup>D</sup>, T(f) = R X ϕ(x)f(x)dµ(x). The adjoint operator is T ∗ : R <sup>D</sup> → L 2 (X ), T <sup>∗</sup> (y) = P<sup>D</sup> <sup>i</sup>=1 yiϕ(ω<sup>i</sup> , ·).

Let's see how T T <sup>∗</sup> acts on a vector <sup>v</sup> ∈ <sup>R</sup> D:

$$\begin{aligned} TT^*v &= T(T^*v) \\ &= T(\phi(\cdot)^\top v) \\ &= \int_{\mathcal{X}} \phi(x)\phi(\cdot)^\top v d\mu(x) \\ &= \int_{\mathcal{X}} \phi(x)\phi(\cdot)^\top d\mu(x)v \\ &= E_{\mathbf{x}}[\phi(\mathbf{x})\phi(\mathbf{x})^\top]v \end{aligned}$$

Thus, we have that T T <sup>∗</sup> = Ex[ϕ(x)ϕ(x) <sup>⊤</sup>]. We know from linear algebra that T T <sup>∗</sup> has the same eigenvalues as T <sup>∗</sup>T. Thus, it is enough to bound the eigenvalues of Σ =ˆ T <sup>∗</sup>T.

To bound the eigenvalues of Σˆ, we bound its difference in operator norm to the equivalent operator for the true kernel K, i.e. Σf = R X <sup>k</sup>(x, ·)f(x)dµ(x). Since we assume that for MD > D0(δ1) we have that Σˆ − <sup>Σ</sup> ≤ <sup>C</sup> with probability ≥ <sup>1</sup> − <sup>δ</sup>1, we get that Σˆ ≤ ∥Σ∥ <sup>+</sup> <sup>C</sup> with probability ≥ <sup>1</sup> − <sup>δ</sup>1. This implies that with probability ≥ <sup>1</sup> − <sup>δ</sup><sup>1</sup> <sup>C</sup>upper is bounded by a constant independent of M and D.

## *Second step: Using least norm geometry.*

By least norm geometry, we get that θ (ENS) − <sup>θ</sup> (Single ) 2 2 = θ (ENS) 2 2 − θ (Single ) 2 2 . Furthermore, we directly get that the norm of the ensemble as the sum of the norms of the component models, i.e. θ (ENS) 2 = M P<sup>M</sup> j=1 θ (ENS) j 2 2 .

*Third step: bound the probability that all empirical kernel inverses admit a Taylor expansion and have bounded lower eigenvalues.*

We now want to bound probability that the eigenvalues of <sup>1</sup> <sup>M</sup>∗<sup>D</sup> ΦΦ<sup>T</sup> − <sup>K</sup>—i.e. the difference between the empirical kernel matrix and the true kernel matrix—are bigger than <sup>λ</sup>min 2 . If the eigenvalues are less than <sup>λ</sup>min 2 , then

- 1. the inverse of the matrix K−1/<sup>2</sup> <sup>1</sup> <sup>M</sup>∗<sup>D</sup> ΦΦ<sup>T</sup> <sup>K</sup>−1/<sup>2</sup> admits a Taylor expansion and
- 2. the minimum eigenvalue of ( 1 MD ΦΦ<sup>T</sup> ) <sup>−</sup><sup>1</sup> will be lower bounded.

We use the following concentration inequality to bound this probability:

Lemma C.5 [\(Wainwright](#page-11-5) [\(2019\)](#page-11-5), Thm. 6.5). *Let* <sup>x</sup>1, . . . , x<sup>n</sup> ∈ <sup>R</sup> <sup>d</sup> *be i.i.d.* L*-subgaussian random variables with* A = <sup>E</sup>[xix ⊤ i ] ∈ <sup>R</sup> d×d *. Then for any* <sup>δ</sup> ≥ <sup>0</sup>*, there exists some* <sup>c</sup>1, c2, c<sup>3</sup> <sup>&</sup>gt; <sup>0</sup> *so that*

$$\mathbb{P} \left[ \frac{\|\frac{1}{n} \sum_{i=1}^n x_i x_i^T - A\|_2}{L^2} \geq c_1 \left\{ \sqrt{\frac{d}{n}} + \frac{d}{n} \right\} + \delta \right] \leq c_2 e^{-c_3 n \min(\delta, \delta^2)}$$

Applying Lemma [C.5](#page-25-0) to the case of the random matrix <sup>1</sup> <sup>M</sup>·<sup>D</sup> ΦΦ<sup>T</sup> − <sup>K</sup>, we have that <sup>n</sup> <sup>=</sup> <sup>M</sup> ∗ <sup>D</sup>, <sup>d</sup> <sup>=</sup> <sup>N</sup>, <sup>A</sup> <sup>=</sup> <sup>K</sup> and <sup>x</sup><sup>i</sup> is the <sup>i</sup>-th column of <sup>Φ</sup>. Furthermore, we want to bound the probability that ∥ 1 <sup>M</sup>·<sup>D</sup> ΦΦ<sup>T</sup> <sup>−</sup>K∥<sup>2</sup> <sup>L</sup><sup>2</sup> is bigger than <sup>λ</sup>min <sup>2</sup>·L<sup>2</sup> . Thus, we set κ<sup>1</sup> = max(0, λmin <sup>2</sup>·L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>M</sup>·<sup>D</sup> + N M·D o ) and get that if <sup>λ</sup>min <sup>2</sup>·L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>M</sup>·<sup>D</sup> + N M·D o > 0, then the probability that the eigenvalues of the difference are bigger than <sup>λ</sup>min 2 is at most c2e <sup>−</sup>c3M·<sup>D</sup> min(<sup>κ</sup>1,κ<sup>2</sup> 1) .

Similarly, we get that the probability that the eigenvalues of the difference for a single component model ( <sup>D</sup> ΦjΦ T <sup>j</sup> − <sup>K</sup>) are bigger than <sup>λ</sup>min 2 is bounded by c2e <sup>−</sup>c3<sup>D</sup> min(<sup>κ</sup>2<sup>j</sup> ,κ<sup>2</sup> <sup>2</sup><sup>j</sup> ) , where we define κ2<sup>j</sup> = max(0, λmin <sup>2</sup>∗L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>D</sup> + N D o ). The probability that the eigenvalues of any of the component models are bigger than <sup>λ</sup>min 2 is then by a union bound bounded by <sup>M</sup> · <sup>c</sup>2<sup>e</sup> <sup>−</sup>c3<sup>D</sup> min(<sup>κ</sup>2<sup>j</sup> ,κ<sup>2</sup> <sup>2</sup><sup>j</sup> ) (again if <sup>λ</sup>min <sup>2</sup>·L<sup>2</sup> − c<sup>1</sup> L<sup>2</sup> nq N <sup>D</sup> + N D o > 0).

We now define <sup>δ</sup><sup>2</sup> <sup>=</sup> <sup>M</sup> · <sup>c</sup>2<sup>e</sup> <sup>−</sup>c3<sup>D</sup> min(<sup>κ</sup>2<sup>j</sup> ,κ<sup>2</sup> <sup>2</sup><sup>j</sup> ) <sup>+</sup> <sup>c</sup>2<sup>e</sup> <sup>−</sup>c3<sup>D</sup> min(<sup>κ</sup>1,κ<sup>2</sup> 1) .

*Fourth step: assume that all empirical kernel matrices come from a truncated distribution.*

Let π˜<sup>D</sup> and π˜MD be the distributions over <sup>1</sup> <sup>D</sup> ΦiΦ ⊤ i and <sup>1</sup> MD ΦΦ<sup>⊤</sup> matrices *conditioned* on the fact that their inverse matrices admit a Taylor expansion. With probability <sup>1</sup> − <sup>δ</sup>2, the <sup>1</sup> <sup>D</sup> ΦiΦ ⊤ <sup>i</sup> matrices that form our ensemble and single model admit Taylor expansions. In other words, with probability <sup>1</sup> − <sup>δ</sup><sup>2</sup> we can view the <sup>1</sup> <sup>D</sup> ΦiΦ ⊤ <sup>i</sup> matrices as i.i.d. draws from π˜<sup>D</sup> and we can view <sup>1</sup> MD ΦΦ<sup>⊤</sup> as a draw from <sup>π</sup>˜D.

Under π˜D, we have that:

$$\begin{aligned}\mathbb{E}_{\tilde{\pi}_D} \left[ K^{1/2} \left( \frac{1}{D} \Phi_i \Phi_i^\top \right)^{-1} K^{1/2} \right] &= \mathbb{E}_{\tilde{\pi}_D} \left[ \left( I - \left( I - K^{-1/2} \left( \frac{1}{D} \Phi_i \Phi_i^\top \right) K^{-1/2} \right) \right)^{-1} \right] \\ &= \sum_{i=0}^{\infty} \mathbb{E}_{\tilde{\pi}_D} \left[ \left( I - K^{-1/2} \left( \frac{1}{D} \Phi_i \Phi_i^\top \right) K^{-1/2} \right)^i \right]\end{aligned}$$

The zero-th term in the Taylor expansion is I. Recognizing that ( <sup>D</sup> ΦiΦ ⊤ i ) is a sample mean random feature outer products with expectation K, the first term in the Taylor expansion is 0. Using formula (15) from [\(Angelova,](#page-9-27) [2012\)](#page-9-27) we find that the second term is equal to <sup>1</sup> <sup>D</sup>M<sup>2</sup> for some constant M<sup>2</sup> an all other terms are O( <sup>D</sup><sup>2</sup> ). Thus:

$$\mathbb{E}_{\pi_D} \left[ K^{1/2} \left( \frac{1}{D} \Phi_i \Phi_i^\top \right)^{-1} K^{1/2} \right] = I + \frac{1}{D} M_2 + O \left( \frac{1}{D^2} \right).$$

Following the same argument we have that

$$\mathbb{E}_{\tilde{\pi}_{MD}} \left[ K^{1/2} \left( \frac{1}{MD} \Phi \Phi^\top \right)^{-1} K^{1/2} \right] = I + \frac{1}{MD} M_2 + O \left( \frac{1}{(MD)^2} \right).$$

Thus,

$$\begin{aligned} & \frac{1}{M} \sum_{i=1}^M \mathbb{E}_{\tilde{\pi}_D} \left[ y^\top \left( \frac{1}{D} \Phi_i \Phi_i^\top \right)^{-1} y \right] - \mathbb{E}_{\tilde{\pi}_{MD}} \left[ y^\top \left( \frac{1}{MD} \Phi \Phi^\top \right)^{-1} y \right] \\ &= y^\top K^{-1/2} \left( \left[ I + \frac{1}{D} M_2 + O \left( \frac{1}{D^2} \right) \right] - \left[ I + \frac{1}{MD} M_2 + O \left( \frac{1}{(MD)^2} \right) \right] \right) K^{-1/2} y \\ &= (y^\top K^{-1} y) O \left( \frac{1}{D} \right) \end{aligned}$$

#### *Sixth step: Using a Hoeffding bound.*

Lastly, we bound the difference between y( 1 MD ΦΦ<sup>⊤</sup>) <sup>−</sup><sup>1</sup>y and its expected value (over the truncated distribution). Equivalently, we have to do this for the ensemble terms <sup>1</sup> M P<sup>M</sup> <sup>i</sup>=1 y ⊤( 1 <sup>D</sup> ΦiΦ ⊤ i ) <sup>−</sup><sup>1</sup>y.

We first employ that we now that the operator norm of the difference <sup>1</sup> MD ΦΦ<sup>⊤</sup> − <sup>K</sup> is bounded by <sup>λ</sup>min 2 , implying that the eigenvalues of <sup>1</sup> MD ΦΦ<sup>⊤</sup> are bounded by <sup>λ</sup>min 2 from below.

Thus, we have that <sup>0</sup> ≤ <sup>y</sup> ⊤( 1 MD ΦΦ<sup>⊤</sup>) <sup>−</sup><sup>1</sup><sup>y</sup> ≤ 2 λmin · ∥y∥ 2 is bounded a.s. under the truncated distribution. Equivalently, this holds for the ensemble terms.

To bound the difference between the ensemble and single model, we can now employ a Hoeffding bound. This gives us that

$$\mathbb{P} \left[ \left| \frac{1}{M} \sum_{i=1}^M y^\top \left( \frac{1}{D} \Phi_i \Phi_i^\top \right)^{-1} y - \mathbb{E}_{\pi_D} \left[ y^\top \left( \frac{1}{D} \Phi_i \Phi_i^\top \right)^{-1} y \right] \right| \geq \epsilon \right] \leq \exp \left( -M\epsilon^2 \lambda_{min} \right)$$

and for the single model term:

$$\mathbb{P} \left[ \left| y^\top \left( \frac{1}{MD} \Phi \Phi^\top \right)^{-1} y - \mathbb{E}_{\tilde{\pi}_{MD}} \left[ y^\top \left( \frac{1}{MD} \Phi \Phi^\top \right)^{-1} y \right] \right| \geq \epsilon \right] \leq \exp \left( -\epsilon^2 \lambda_{min} \right)$$

Setting exp −M ϵ<sup>2</sup>λmin + exp −ϵ <sup>2</sup>λmin ≤ <sup>δ</sup><sup>3</sup> := 2 · exp −ϵ <sup>2</sup>λmin and solving for ϵ gives us that:

$$\epsilon = \sqrt{\frac{1}{\lambda_{min}} \log\left(\frac{2}{\delta_3}\right)}$$

#### *Seventh step: Taking everything together.*

We can now employ a union bound to get that the probability that all three conditions are satisfied is at least <sup>1</sup> − <sup>δ</sup><sup>1</sup> − <sup>δ</sup><sup>2</sup> − <sup>δ</sup><sup>3</sup> and thus we get the bound on the difference between the ensemble and single model in L<sup>2</sup> norm.

Note that the fact that we worked under a truncated distribution in the previous step is not a problem, since the corresponding events have a lower probability under the the non-truncated distribution as long as we have already assumed the exclusion of all events where the difference between the empirical kernel matrix and the true kernel matrix is bigger than <sup>λ</sup>min 2 .

#### C.3. Variance of Ensemble Predictions

In the next step, we show the formula for the variance of a single model prediction under Gaussianity. Note that one could also get this result by slightly extending proofs by [\(Jacot et al.,](#page-9-21) [2020\)](#page-9-21).

Lemma C.6 (Variance of single model predictions). *Under Gaussianity and assuming* D > N + 1*, the variance of single model prediction at a test point* x ∗ *is given by*

$$\mathbb{V}_W[h_W^{(\text{LN})}(x^*)] = r_\perp^2 \frac{\|h_\infty^{(\text{LN})}\|_{\mathcal{H}}^2}{D - N - 1}, \quad (10)$$

*where* ∥ · ∥<sup>H</sup> *is norm defined by the RKHS associated with kernel* <sup>k</sup>(·, ·)*.*

*Proof.* We start by writing down the variance of the prediction of a single model:

$$\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)] = \mathbb{E}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)^2] - \mathbb{E}_{\mathcal{W}}[h_{\mathcal{W}}^{(\text{LN})}(x^*)]^2$$

Using Theorem [3.2,](#page-5-0) the definition of the prediction of a single model and the definition of W and w⊥, we can expand this expression:

$$\begin{aligned}
&= \mathbb{E}_{\mathcal{W}}[\phi_{\mathcal{W}}^* \Phi_{\mathcal{W}}^{\top} (\Phi_{\mathcal{W}} \Phi_{\mathcal{W}}^{\top})^{-1} y y^{\top} (\Phi_{\mathcal{W}} \Phi_{\mathcal{W}}^{\top})^{-\top} \Phi_{\mathcal{W}} \phi_{\mathcal{W}}^{*\top}] - (h_{\infty}^{(\text{LN})}(x^*))^2 \\
&= \mathbb{E}_{W_i w_{\perp}} [(r_{\perp} w_{\perp}^{\top} + c^{\top} W) W^{\top} R (R^{\top} W W^{\top} R)^{-1} y y^{\top} (R^{\top} W W^{\top} R)^{-\top} R^{\top} W (r_{\perp} w_{\perp}^{\top} + c^{\top} W)^{\top}] \\
&\quad - (h_{\infty}^{(\text{LN})}(x))^2 \\
&= \mathbb{E}_{W_i w_{\perp}} [(r_{\perp} w_{\perp}^{\top} + c^{\top} W) W^{\top} (W W^{\top})^{-1} R^{-\top} y y^{\top} R^{-1} (W W^{\top})^{-\top} W (r_{\perp} w_{\perp}^{\top} + c^{\top} W)^{\top}] \\
&\quad - (h_{\infty}^{(\text{LN})}(x))^2 \\
&= (c^{\top} R^{-\top} y)^2 - (h_{\infty}^{(\text{LN})}(x))^2 \\
&+ 2 \cdot r_{\perp}^{\top} \mathbb{E}_{W_i w_{\perp}} [w_{\perp}^{\top} W^{\top} (W W^{\top})^{-1}] R^{-\top} y y^{\top} R^{-1} c \\
&+ r_{\perp}^2 \mathbb{E}_{W_i w_{\perp}} [w_{\perp}^{\top} W^{\top} (W W^{\top})^{-1} R^{-\top} y y^{\top} R^{-1} (W W^{\top})^{-\top} W w_{\perp}]
\end{aligned}$$

Now we can see that the first two terms cancel out (since h (LN) <sup>∞</sup> (x) = c <sup>⊤</sup>R−⊤y) and the third term is zero by Lemma [3.1.](#page-5-3) We are left with the fourth term, which we can slightly rewrite:

$$\begin{aligned} \mathbb{V}_W[h_W^{(\text{LN})}(x^*)] &= r_\perp^2 \mathbb{E}_{W, w_\perp} [w_\perp^\top W^\top (WW^\top)^{-1} R^{-\top} y w^\top R^{-1} (WW^\top)^{-T} W w_\perp] \\ &= r_\perp^2 y^\top R^{-1} \mathbb{E}_{W, w_\perp} [(WW^\top)^{-T} W w_\perp w_\perp^\top W^\top (WW^\top)^{-1}] R^{-\top} y \end{aligned} \quad (11)$$

Using the tower rule for conditional expectations, we have:

$$\begin{aligned} \mathbb{V}_W[h_W^{(\text{LN})}(x)] &= r_\perp^2 y^\top R^{-1} \mathbb{E}_{W, w_\perp} [(WW^\top)^{-T} W w_\perp w_\perp^\top W^\top (WW^\top)^{-1}] R^{-\top} y \\ &= r_\perp^2 y^\top R^{-1} \mathbb{E}_W [(WW^\top)^{-T} W \mathbb{E}_{w_\perp | W} [w_\perp w_\perp^\top | W] W^\top (WW^\top)^{-1}] R^{-\top} y \end{aligned}$$

Since the Gaussianity assumption implies W and w<sup>⊥</sup> are independent, we get:

$$\mathbb{V}_W[h_W^{(\text{LN})}(x)] = r_\perp^2 y^\top R^{-1} \mathbb{E}_W[(WW^\top)^{-T} W \mathbb{E}_{w_\perp}[w_\perp w_\perp^\top] W^\top (WW^\top)^{-1}] R^{-\top} y$$

Moreover, since by Gaussianity w<sup>⊥</sup> and W are multivariate Gaussians with the identity matrix as covariance, we get (via the expected value of a Wishart and an inverse Wishart distribution; note that for getting this expected value, we need to assume that D > N + 1):

$$\begin{aligned} \mathbb{V}_W[h_W^{(\text{LN})}(x)] &= r_{\perp}^2 y^\top R^{-1} \mathbb{E}_W[(WW^\top)^{-T}(WW^\top)(WW^\top)^{-1}] R^{-\top} y \\ &= r_{\perp}^2 y^\top R^{-1} \mathbb{E}_W[(WW^\top)^{-T}] R^{-\top} y \\ &= r_{\perp}^2 \frac{y^\top R^{-1} R^{-\top} y}{D - N - 1} \\ &= r_{\perp}^2 \frac{y^\top K^{-1} y}{D - N - 1}. \end{aligned}$$

Recognizing that y <sup>⊤</sup>K−<sup>1</sup><sup>y</sup> <sup>=</sup> ∥<sup>h</sup> (LN) <sup>∞</sup> ∥ <sup>H</sup> (e.g. [Wainwright,](#page-11-5) [2019,](#page-11-5) Ch. 12) completes the proof.

An equivalent argument does not work under the more general Assumption [2.1](#page-3-1) since w<sup>⊥</sup> and W are not necessarily independent. Even in the case of independence, <sup>E</sup><sup>W</sup> [(WW<sup>⊤</sup>) −1 ] might not be known.

Counterexample for subexponential case. We now give an explicit counterexample showing that when only assuming uncorrelatedness between W and w<sup>⊥</sup> the term

$$E := \mathbb{E}_{W, w_\perp} [(WW^\top)^{-T} W w_\perp w_\perp^\top W^\top (WW^\top)^{-1}]$$

from Eq. [\(11\)](#page-27-1) depends on x ∗ implying that the variance does not only depend on x <sup>∗</sup> via r 2 ⊥.

Let us assume <sup>N</sup> <sup>=</sup> <sup>D</sup> = 1 and let <sup>W</sup> be uniformly distributed across the set n − √ 4 12.5 , − <sup>√</sup> 12.5 , √ 12.5 , √ 4 12.5 o . Then we have <sup>E</sup>[W] = 0 and <sup>E</sup>[W<sup>2</sup> ] = <sup>1</sup> 2 · <sup>12</sup>.<sup>5</sup> + 2 · 9 <sup>12</sup>.<sup>5</sup> = 1.

Now consider an x ∗ that produces a w<sup>⊥</sup> so that w<sup>⊥</sup> = √ 2 when W = n − √ 3 12.5 , √ 12.5 o and w<sup>⊥</sup> = 0 otherwise. Then we have <sup>E</sup>[w ⊤ <sup>⊥</sup>W] = 0 and <sup>E</sup>[w 2 <sup>⊥</sup>] = 1. The value of <sup>E</sup> is now <sup>12</sup>.<sup>5</sup> 9 .

Furthermore, consider an x ∗ that produces a w<sup>⊥</sup> so that w<sup>⊥</sup> = √ 2 when W = n − √ 4 12.5 , √ 4 12.5 o and w<sup>⊥</sup> = 0 otherwise. Then we have <sup>E</sup>[w ⊤ <sup>⊥</sup>W] = 0 and <sup>E</sup>[w 2 <sup>⊥</sup>] = 1. The value of <sup>E</sup> is now <sup>12</sup>.<sup>5</sup> <sup>16</sup> .

## D. Proofs for Overparameterized Ridge Regression

### D.1. Difference between the Infinite Ensemble and Infinite Single Model

We begin with a lemma, which shows that the prediction of kernel regressors is Lipschitz-continuous in λ for any x ∗ and <sup>λ</sup> ≥ <sup>0</sup>. We will denote the kernel ridge regressor with regularization parameter <sup>λ</sup> as <sup>h</sup> (RR) <sup>∞</sup>,λ , as introduced in Sec. [3.4.](#page-7-3)

Lemma D.1 (Bound on the difference between the kernel ridge regressors). *Let* λ, λ′ ≥ <sup>0</sup> *be two regularization parameters. Then, for any* x <sup>∗</sup> ∈ X *it holds that:*

$$|h_{\infty,\lambda'}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*)| \leq \sqrt{n} \cdot C_1 \cdot |\lambda' - \lambda| \cdot \sqrt{y^T K^{-4}} y$$

*where we assume* k(x<sup>i</sup> , x<sup>∗</sup> ) ≤ <sup>C</sup><sup>1</sup> *for all* <sup>i</sup> ∈ [N]*.*

*Proof.* We can write the kernel ridge regressors as h (RR) <sup>∞</sup>,λ (x ∗ ) = P<sup>n</sup> <sup>i</sup>=1 α1,ik(x<sup>i</sup> , x<sup>∗</sup> ) and h (RR) <sup>∞</sup>,λ′ (x ∗ ) = P<sup>n</sup> <sup>i</sup>=1 α2,ik(x<sup>i</sup> , x<sup>∗</sup> ) with coefficients α<sup>1</sup> and α<sup>2</sup> given by:

$$\begin{aligned}\alpha_1 &= (K + \lambda I)^{-1}y \\ \alpha_2 &= (K + \lambda' I)^{-1}y\end{aligned}$$

We now write y in the orthonormal basis of the eigenvectors of K, i.e. y = P<sup>n</sup> <sup>i</sup>=1 aiv<sup>i</sup> . We call the corresponding eigenvalues of K d1, . . . , d<sup>n</sup> > 0.

The matrix (K + λI) <sup>−</sup><sup>1</sup> has the same eigenvectors as K and the eigenvalues are 0 < ˜d<sup>i</sup> = 1 <sup>d</sup>i+<sup>λ</sup> ≤ 1 λ . Thus, we can write α<sup>1</sup> = P<sup>n</sup> <sup>i</sup>=1 a<sup>i</sup> di+λ v<sup>i</sup> and α<sup>2</sup> = P<sup>n</sup> <sup>i</sup>=1 a<sup>i</sup> 1 <sup>d</sup>i+λ′ v<sup>i</sup> .

In the next step, we bound ∥<sup>α</sup><sup>1</sup> − <sup>α</sup>2∥ 2 2 : Using the orthonormality of the eigenvectors, we get:

$$\|\alpha_1 - \alpha_2\|_2^2 = \sum_{i=1}^n \left( a_i \left( \frac{1}{d_i + \lambda} - \frac{1}{d_i + \lambda'} \right) \right)^2$$

Now we bound 1 λ+d<sup>i</sup> − λ′+d<sup>i</sup>  ≤ λ ′−λ λλ′+(λ+λ′)di+d  ≤ |λ ′−λ| d which gives us:

$$\|\alpha_1 - \alpha_2\|_2^2 \leq \sum_{i=1}^n \left( \frac{a_i |\lambda' - \lambda|}{d_i^2} \right)^2 \leq |\lambda' - \lambda|^2 y^T K^{-4} y$$

Using this result, we can bound the difference between the predictions of the two kernel regressors at a single point x ∗ :

$$|h_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda'}^{(RR)}(x^*)| = |\sum_{i=1}^n (\alpha_{1,i} - \alpha_{2,i})k(x_i, x^*)| \leq \sum_{i=1}^n |\alpha_{1,i} - \alpha_{2,i}|k(x_i, x^*)$$

Since k(x<sup>i</sup> , x<sup>∗</sup> ) ≤ <sup>C</sup>1, we get (using the relation between the 1-norm and the 2-norm):

$$|f_\lambda(x^*) - f_{\lambda'}(x^*)| \leq C_1 \sum_{i=1}^n |\alpha_{1,i} - \alpha_{2,i}| \leq C_1 \|\alpha_1 - \alpha_2\|_2 \sqrt{n} \leq \sqrt{n} \cdot C_1 \cdot |\lambda' - \lambda| \cdot \sqrt{y^\top K^{-4}} y$$

Using similar arguments, we now show that the expected prediction of RF regressors, i.e., the prediction of the infinite ensemble of RF regressors, is Lipschitz-continuous for any x ∗ and <sup>λ</sup> ≥ <sup>0</sup>:

Lemma D.2 (Bound on the difference between expected RF Regressors). *Under Assumption [2.1](#page-3-1) and Assumption [3.4,](#page-7-2) the expected value of the prediction of RF regressors is Lipschitz-continuous in* λ *for any* x <sup>∗</sup> *and* <sup>λ</sup> ≥ <sup>0</sup>*, i.e., for any* <sup>x</sup> ∗ *it holds that:*

$$|\bar{h}_{\infty,\lambda'}^{(RR)}(x^*) - \bar{h}_{\infty,\lambda}^{(RR)}(x^*)| \leq \|c^\top R^{-\top}\| \|y\| DC_2 \|\lambda' - \lambda\|$$

*where* C<sup>2</sup> *is a constant depending on the distribution of* Φ*.*

*Proof.* We use the characterization of h¯ (RR) <sup>∞</sup>,λ (x ∗ ) from Eq. [\(9\)](#page-23-1), which gives us the difference as

$$\left| c^\top \mathbb{E}_{W, w_\perp} \left[ WW^\top \left( (WW^\top + D \cdot \lambda' \cdot R^{-\top} R^{-1})^{-1} - (WW^\top + D \cdot \lambda' \cdot R^{-\top} R^{-1})^{-1} \right) \right] R^{-\top} y \right|.$$

We can now reverse some steps we made to get this characterization and write it in terms of Φ again:

$$\left| c^\top R^{-\top} \mathbb{E}_W \left[ \Phi_W \Phi_W^\top \left( (\Phi_W \Phi_W^\top + D \cdot \lambda' \cdot I)^{-1} - (\Phi_W \Phi_W^\top + D \cdot \lambda \cdot I)^{-1} \right) \right] y \right|.$$

And now, using Jensen's inequality and the convexity of the two-norm, we can pull out the expected value to the outside of the difference:

$$\| c^\top R^{-\top} \| \cdot \mathbb{E}_W \left[ \| \Phi_W \Phi_W^\top \left( (\Phi_W \Phi_W^\top + D \cdot \lambda' \cdot I)^{-1} - (\Phi_W \Phi_W^\top + D \cdot \lambda \cdot I)^{-1} \right) y \| \right].$$

Similarly to the proof of Lemma [D.1,](#page-28-0) we can write y in the orthonormal basis of the eigenvectors of ΦΦ<sup>⊤</sup> (note that we drop the subscript W for notational simplicity), i.e. <sup>y</sup> <sup>=</sup> P<sup>n</sup> <sup>i</sup>=1 aiv<sup>i</sup> . Furthermore we define the eigenvalues of ΦΦ<sup>⊤</sup> as <sup>d</sup>1, . . . , d<sup>n</sup> <sup>&</sup>gt; <sup>0</sup>. The matrix (ΦΦ<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · λI) −1 again has the same eigenvectors as ΦΦ<sup>⊤</sup> and the eigenvalues are 0 < 1 <sup>d</sup>i+D·<sup>λ</sup> ≤ D·λ .

Multiplying <sup>y</sup> with ΦΦ<sup>⊤</sup>(ΦΦ<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · λI) −1 and ΦΦ<sup>⊤</sup>(ΦΦ<sup>⊤</sup> <sup>+</sup> <sup>D</sup> · <sup>λ</sup> ′ I) −1 then gives us:

$$\Phi \Phi^\top (\Phi \Phi^\top + D \cdot \lambda I)^{-1} y = \sum_{i=1}^n a_i \frac{d_i}{d_i + D \cdot \lambda} v_i$$

$$\Phi \Phi^\top (\Phi \Phi^\top + D \cdot \lambda' I)^{-1} y = \sum_{i=1}^n a_i \frac{d_i}{d_i + D \cdot \lambda'} v_i$$

We can now calculate the difference of these two vectors using the orthonormality of the eigenvectors:

$$\|\Phi\Phi^\top(\Phi\Phi^\top + D \cdot \lambda' I)^{-1}y - \Phi\Phi^\top(\Phi\Phi^\top + D \cdot \lambda I)^{-1}y\|_2^2 = \sum_{i=1}^n \left( a_i \left( \frac{d_i}{d_i + D \cdot \lambda} - \frac{d_i}{d_i + D \cdot \lambda'} \right) \right)^2$$

Now we look at the difference between the two coefficients and see that for each i, we have:

$$\left| \frac{d_i}{d_i + D \cdot \lambda} - \frac{d_i}{d_i + D \cdot \lambda'} \right| \leq \frac{D \cdot |\lambda' - \lambda|}{d_i}$$

Thus, we have that the difference is bounded by:

$$\|\Phi\Phi^\top(\Phi\Phi^\top + D \cdot \lambda' I)^{-1}y - \Phi\Phi^\top(\Phi\Phi^\top + D \cdot \lambda I)^{-1}y\|_2^2 \leq \frac{D^2 \cdot |\lambda - \lambda'|^2}{d_N^2} \|y\|_2^2.$$

All together, we can now bound the difference of the expected values of the predictions of RF regressors via:

$$|\bar{h}_{\infty,\lambda'}^{(RR)}(x^*) - \bar{h}_{\infty,\lambda}^{(RR)}(x^*)| \leq \|c^\top R^{-\top} \| \|y\| D \|\lambda' - \lambda\| \mathbb{E}_{d_N} \left[ \frac{1}{d_N} \right]$$

Since tr((ΦΦ<sup>⊤</sup>) −1 ) = P<sup>n</sup> i=1 d<sup>i</sup> , and the trace is a linear operator, we can write:

$$\mathbb{E}_{d_N} \left[ \frac{1}{d_N} \right] \leq \mathbb{E}_W [(\text{tr}(\Phi_W \Phi_W^\top)^{-1})] = \text{tr}(\mathbb{E}_W [(\Phi_W \Phi_W^\top)^{-1}]) =: C_2$$

which is finite whenever E<sup>W</sup> (ΦWΦ ⊤ <sup>W</sup>) −1 is finite, i.e. Assumption [3.4](#page-7-2) holds.

Using Lemma [D.1](#page-28-0) and Lemma [D.2](#page-29-0) we can now show that the difference between the infinite ensemble where each model has ridge <sup>λ</sup> and the infinite single model with ridge <sup>λ</sup> is Lipschtiz-continuous in <sup>λ</sup> for <sup>λ</sup> ≥ <sup>0</sup>:

*Theorem [3.5](#page-7-0) (Restated).* Under Assumptions [2.1](#page-3-1) and [3.4,](#page-7-2) the difference |h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ )| between the infinite ensemble and the single infinite-width model trained with ridge <sup>λ</sup> ≥ <sup>0</sup> is Lipschitz-continuous in <sup>λ</sup>. The Lipschitz constant is independent of x ∗ for compact X .

*Proof.* We bound difference |h¯ (RR) <sup>∞</sup>,λ′ (x ∗ ) − <sup>h</sup> (RR) <sup>∞</sup>,λ′ (x ∗ )| − |h¯ (RR) <sup>∞</sup>,λ (x ∗ ) − <sup>h</sup> (RR) <sup>∞</sup>,λ (x ∗ )|  by using first the inverse, then the normal triangle inequality:

$$\begin{aligned} & |\bar{h}_{\infty,\lambda'}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*)| - |\bar{h}_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*)| \\ & \leq |\bar{h}_{\infty,\lambda'}^{(RR)}(x^*) - \bar{h}_{\infty,\lambda}^{(RR)}(x^*)| + h_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*) \\ & \leq |\bar{h}_{\infty,\lambda'}^{(RR)}(x^*) - \bar{h}_{\infty,\lambda}^{(RR)}(x^*)| + |h_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda'}^{(RR)}(x^*)| \end{aligned}$$

Using the bound from Lemma [D.1](#page-28-0) and Lemma [D.2](#page-29-0) (and summarizing the the corresponding constants as c<sup>1</sup> and c2) we can bound this by:

$$|\bar{h}_{\infty,\lambda'}^{(RR)}(x^*) - h_{\infty,\lambda'}^{(RR)}(x^*)| - |\bar{h}_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*)| \leq c_1|\lambda' - \lambda| + c_2|\lambda' - \lambda|$$

Thus we have Lipschitz-continuity in <sup>λ</sup> for <sup>λ</sup> ≥ <sup>0</sup>.

The Lipschitz constant is independent of x ∗ for X compact since the Lipschitz constants from Lemma [D.1](#page-28-0) and Lemma [D.2](#page-29-0) depend on x ∗ in a continuous fashion.

Note that an equivalent argument in combination with [\(Jacot et al.,](#page-9-21) [2020\)](#page-9-21)[Proposition 4.2], i.e. <sup>λ</sup>˜ ≤ γ γ−1 λ, directly gives the Lipschitz-continuity in <sup>λ</sup> for <sup>λ</sup> ≥ <sup>0</sup> for the difference between the infinite ensemble and the infinite-width single model with effective ridge in the overparameterized regime.

## E. Underparameterized Ensembles

Here, we offer a proof that infinite, unregularized, underparameterized RF ensembles are equivalent to kernel ridge regression under a transformed kernel function. We emphasize the difference from the overparameterized case—the central focus of our paper—in which the infinite ensemble is equivalent to a ridgeless kernel regressor. Thus, underparameterized ensembles induce regularization, while overparameterized ensembles do not.

Other works have explored the ridge behavior of underparameterized RF ensembles [\(Kaban´](#page-9-4) , [2014;](#page-9-4) [Thanei et al.,](#page-10-1) [2017;](#page-10-1) [Bach,](#page-9-16) [2024b\)](#page-9-16); however, these works often focus on an equivalence in generalization error whereas we establish a pointwise equivalence. To the best of our knowledge, the following result is novel:

Lemma E.1. *If the expected orthogonal projection matrix* <sup>E</sup>W˜ h R⊤W˜ W⊤RR⊤W <sup>−</sup><sup>1</sup> <sup>W</sup>˜ <sup>⊤</sup><sup>R</sup> i *is well defined, and a contraction (i.e., singular values strictly less than 1), then the infinite underparameterized RF ensemble* h¯ (LN) <sup>∞</sup> (x ∗ ) *is equivalent to kernel ridge regression under some kernel function* ˜k(·, ·)*.*

*Proof.* When D < N, the infinite ridgeless RF ensemble is given by

$$\begin{aligned}\bar{h}_\infty^{(LN)}(x^*) &= \mathbb{E}_W \left[ \frac{1}{D} \sum_{j=1}^D \phi(\omega_j, x^*) \left( \frac{1}{D} \Phi_W^\top \Phi_W \right)^{-1} \Phi_W^\top \right] y \\ &= \mathbb{E}_{W, w_\perp} \left[ (r_\perp w_\perp^\top + c^\top W) (W^\top R R^\top W)^{-1} W^\top \right] R y,\end{aligned}\tag{12}$$

where W, w⊥, r⊥, c, R are as defined in Sec. [2.](#page-3-3) Defining the following block matrices:

$$\tilde{W} = \begin{bmatrix} W_{\perp} \\ W_{\perp} \end{bmatrix} \in \mathbb{R}^{(N+1) \times D}, \quad \tilde{R} = \begin{bmatrix} R \\ 0 \end{bmatrix} \in \mathbb{R}^{(N+1) \times N}, \quad \tilde{c} = \begin{bmatrix} c \\ r_{\perp} \end{bmatrix} \in \mathbb{R}^{(N+1)},$$

we can rewrite Eq. [\(12\)](#page-31-1) as

$$\bar{h}_\infty^{(LN)}(x^*) = \tilde{c}^\top \left( \mathbb{E}_{\tilde{W}} \left[ \tilde{W} \left( W^\top R R^\top W \right)^{-1} \tilde{W}^\top \right] \right) \tilde{R}_y.$$

By adding and subtracting R˜R˜<sup>⊤</sup> inside the outer parenthesis, we can massage this expression into kernel ridge regression in a transformed coordinate system:

$$\begin{aligned}\bar{h}_{\infty}^{(LN)}(x^*) &= \tilde{c}^{\top} \left( \tilde{R}\tilde{R}^{\top} + \left[ \mathbb{E}_{\tilde{W}} \left[ \tilde{W} (W^{\top} R R^{\top} W)^{-1} \tilde{W}^{\top} \right] \right)^{-1} - \tilde{R}\tilde{R}^{\top} \right)^{-1} \tilde{R} y. \\ &= \tilde{c}^{\top} \tilde{A}^{-1} \tilde{R} \left( \tilde{R}^{\top} \tilde{A}^{-1} \tilde{R} + I \right)^{-1} y.\end{aligned}\tag{13}$$

Applying the Woodbury inversion lemma to A˜−<sup>1</sup> , we have:

$$\begin{aligned} \tilde{A}^{-1} &= \mathbb{E}_{\tilde{W}} \left[ \tilde{W} (W^\top R R^\top W)^{-1} \tilde{W}^\top \right] \\ &+ \mathbb{E}_{\tilde{W}} \left[ \tilde{W} (W^\top R R^\top W)^{-1} W^\top R \right] (I - \mathbb{E}_W[P_W])^{-1} \mathbb{E}_{\tilde{W}} \left[ R^\top W (W^\top R R^\top W)^{-1} \tilde{W}^\top \right], \end{aligned} \quad (14)$$

where P<sup>W</sup> is the (random) orthogonal projection matrix onto the span of the columns of R⊤W:

$$P_W = R^\top W (W^\top R R^\top W)^{-1} W^\top R.$$

Because <sup>P</sup><sup>W</sup> is an orthogonal projection matrix, we have that ∥<sup>P</sup><sup>W</sup> ∥<sup>2</sup> = 1, and thus (by Jensen's inequality) ∥<sup>E</sup><sup>W</sup> [P<sup>W</sup> ]∥<sup>2</sup> ≤ <sup>1</sup>. If this inequality is strict so that <sup>I</sup> − <sup>E</sup><sup>W</sup> [P<sup>W</sup> ] is invertible, we have by inspection of Eq. [\(14\)](#page-31-2) that <sup>A</sup>˜ is positive definite. Therefore, the block matrix

$$\begin{bmatrix} \tilde{R}^\top \\ \tilde{c}^\top \end{bmatrix} \tilde{A}^{-1} \begin{bmatrix} \tilde{R} & \tilde{c} \end{bmatrix} = \begin{bmatrix} \tilde{R}^\top \tilde{A}^{-1} \tilde{R} & \tilde{R}^\top \tilde{A}^{-1} \tilde{c} \\ \tilde{c}^\top \tilde{A}^{-1} \tilde{R} & \tilde{c}^\top \tilde{A}^{-1} \tilde{c} \end{bmatrix} \quad (15)$$

is also positive definite and thus the realization of some kernel function ˜k(·, ·); i.e.

$$\begin{bmatrix} \tilde{R}^\top \tilde{A}^{-1} \tilde{R} & \tilde{R}^\top \tilde{A}^{-1} \tilde{c} \\ \tilde{c}^\top \tilde{A}^{-1} \tilde{R} & \tilde{c}^\top \tilde{A}^{-1} \tilde{c} \end{bmatrix} = \begin{bmatrix} \tilde{k}(x_1, x_1) & \cdots & \tilde{k}(x_1, x_N) & \tilde{k}(x_1, x^*) \\ \vdots & \ddots & \vdots & \vdots \\ \tilde{k}(x_N, x_1) & \cdots & \tilde{k}(x_N, x_N) & \tilde{k}(x_N, x^*) \\ \tilde{k}(x^*, x_1) & \cdots & \tilde{k}(x^*, x_N) & \tilde{k}(x^*, x^*) \end{bmatrix}.$$

Note that if A˜ = I then by Eq. [\(1\)](#page-3-2) we recover the original kernel matrix

$$\begin{bmatrix} \tilde{R}^\top \tilde{R} & \tilde{R}^\top \tilde{\epsilon} \\ \tilde{\epsilon}^\top \tilde{R} & \tilde{\epsilon}^\top \tilde{\epsilon} \end{bmatrix} = \begin{bmatrix} k(x_1, x_1) & \cdots & k(x_1, x_N) & k(x_1, x^*) \\ & \ddots & \vdots & \vdots \\ k(x_N, x_1) & \cdots & k(x_N, x_N) & k(x_N, x^*) \\ k(x^*, x_1) & \cdots & k(x^*, x_N) & k(x^*, x^*) \end{bmatrix}.$$

Thus, the underparameterized ensemble in Eq. [\(13\)](#page-31-3) simplifies to

$$\bar{h}_\infty^{(LN)}(x^*) = [\tilde{k}(x^*, x_1) \quad \cdots \quad \tilde{k}(x^*, x_N)] \begin{pmatrix} [\tilde{k}(x_1, x_1) & \cdots & \tilde{k}(x_1, x_N)] \\ \vdots & \ddots & \vdots \\ \tilde{k}(x_N, x_1) & \cdots & \tilde{k}(x_N, x_N) \end{pmatrix} + I \quad y,$$

which is kernel ridge regression with respect to the kernel ˜k(·, ·).