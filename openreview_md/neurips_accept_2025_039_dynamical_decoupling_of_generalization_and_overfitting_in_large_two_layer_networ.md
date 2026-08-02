# Dynamical Decoupling Of Generalization And Overfitting In Large Two-Layer Networks

Andrea Montanari Department of Statistics and Department of Mathematics, Stanford University Pierfrancesco Urbani Université Paris-Saclay, CNRS, CEA,
Institut de Physique Théorique, 91191, Gif-Sur-Yvette, France

## Abstract

Understanding the inductive bias and generalization properties of large overparametrized machine learning models requires to characterize the dynamics of the training algorithm. We study the learning dynamics of large two-layer neural networks via dynamical mean field theory, a well established technique of nonequilibrium statistical physics. We show that, for large network width m, and large number of samples per input dimension n/d, the training dynamics exhibits a separation of timescales which implies: (i) The emergence of a slow time scale associated with the growth in Gaussian/Rademacher complexity of the network;
(ii) Inductive bias towards small complexity if the initialization has small enough complexity; (iii) A dynamical decoupling between feature learning and overfitting regimes; (iv) A non-monotone behavior of the test error, associated 'feature unlearning' regime at large times.

## 1 Introduction

Machine learning (ML) models are trained using stochastic gradient descent (SGD), or one of its variants to minimize the error on training data (empirical risk function). Classically, their good behavior on unseen test data is explained by the fact that model complexity is kept small by regularization techniques: these models do not 'overfit.' Traditional ML theory decouples the analysis of the model from the optimization algorithm, which is assumed to converge to an approximate global minimizer [47]. In contrast, in modern ML, the empirical risk is highly non-convex, the number of parameters is comparable with the number of training samples, and the model complexity is only weakly controlled. As a consequence, there can be many assignments of the model parameters (many global empirical risk minimizers) that perfectly interpolate the data —even when these are noisy. While all of these interpolators are indistinguishable on the training data, they behave very differently (and some of them very poorly) on test data. It has been hypothesized that models trained by SGD generalize well to test data because the algorithm selects a near global minimizer with low complexity, although a mechanistic understanding of this process is lacking. For this reason, the generalization properties cannot be decoupled from the training dynamics. Several striking consequences of this lack of decoupling are documented in the literature (and have long been familiar to practitioners): (i) Test error after training is observed to depend strongly on the initial weights distribution [28]; (ii) Test error depends strongly on the optimization algorithm (SGD, RMSProp, ADAM, to name a few), even when these algorithms achieve the same train error
[55]; (iii) Careful choice of the hyperparameters in the optimization algorithm is crucial [34, 59],
and the optimal choice is often different from the one that minimizes train error; (iv) Models learned by training for a shorter time have smaller complexity and can generalize better [44, 11].

0.7 MF Feat. learning Feat. learning Overfit/Unlearn 5 0.6 kW2ndk1 
√m 2 n d lay e r

`
1
-
n o r m 4 Tr a i n
/
Te st e rr o r 0.5 0.4 3 0.3 2 0.2 generaliz. error kW2ndk1  1 1 0.1 tmf(m)  1 tof(m)  m 100 101 102 t 0.0 0
These observations have motivated a broad effort to encapsulate the effect of the dynamics as 'implicit regularization' [48, 3, 15, 56]: the algorithm selects an empirical risk minimizer that also minimizes a specific notion of model complexity. While this *implicit regularization hypothesis* has been fruitful, it can only be validated if we can precisely understand the training dynamics. In this work we leverage tools from theoretical physics to directly analyze the training dynamics and derive quantitative predictions on the implicit bias of neural network training, in a simple setting.

This allows us to capture feature learning and lazy/overfitting regimes within the same unified picture. We discover a time-scale separation in the training dynamics, between an early stage in which the model learns the relevant features representation of the data, and a late stage of training that is characterized by overfitting, feature 'unlearning,' and hence test error that increases with training. While the regularizing effect of early stopping has been an important object of study (for simpler models) in the past [44, 11, 61, 57], our work is the first to point out a time-scale separation between feature learning (on a faster timescale) and overfitting (on a slower time scale), thus reconciling the feature learning and neural tangent theories of learning.

We study two-layer fully connected neural networks f(· ; θ) : R
d → R, i.e.

$$f(\mathbf{x};\mathbf{\theta})=\frac{1}{m}\sum_{i=1}^{m}a_{i}\,\sigma(\langle\mathbf{w}_{i},\mathbf{x}\rangle)\,,\tag{1.1}$$  $\mathbf{\theta}=(\mathbf{w}_{1},\ldots,\mathbf{w}_{m})\in\mathbb{R}^{d\times m}$ and $\mathbf{a}=(a_{1},\ldots,a_{m})\in\mathbb{R}^{m}$ are 
where θ = (a,W), where W = (w1*, . . . ,* wm) ∈ R
d×m and a = (a1*, . . . , a*m) ∈ R
m are, respectively, first- and second-layer weights. For convenience, we fix the normalization ∥wi∥ = 1, and assume that σ does not depend on m. We apply model (1.1) to a supervised learning task. We are given i.i.d. data (yi, xi), i ≤ n, with yi ∈ R a response variable and xi ∈ R
da feature vector, and try to learn a model f(· ; θ) to predict the response ynew corresponding to a new input xnew. We use gradient flow (GF) to minimize the empirical risk under square loss, namely

$$\hat{\mathbf{\theta}}(t)=-\frac{n}{d}\mathbf{P}_{\mathbf{\theta}}\nabla\hat{\mathscr{R}}_{n}(\mathbf{\theta}(t))\,,\qquad\hat{\mathscr{R}}_{n}(\mathbf{\theta}):=\frac{1}{2n}\sum_{i=1}^{n}\left(y_{i}-f(\mathbf{x}_{i};\mathbf{\theta})\right)^{2}.\tag{1.2}$$

Here P θ is a projection matrix that guarantees that wi(t) ∈ S
d−1at all times. The factor n/d is introduced for convenience and simply amounts to a rescaling of time. We will typically initialize the training by setting (wi)i≤m ∼iid Unif(S
d−1), and ai = a0 for all i ≤ m, and study the dependence of the training dynamics on three key parameters:
Network width: m, Overparametrization ratio: α :=n md, Initialization scale: a0 .

Alongside the train error, we will be interested in the test error at time t, i.e. R(θ(t)) := E{(ynew −
f(xnew; θ(t)))2}/2, and the generalization error R(θ(t)) − Rbn(θ(t)).

Model (1.1) is much simpler than state-of-the-art architectures [52], but is rich enough to investigate several general questions, which we summarize below:
When the network is sufficiently overparametrized (α small) and a0 is large, neural tangent kernel
(NTK) theory predicts that GF converges to an interpolator [30, 22, 16] .

Q1. For which region of *α, a*0 does convergence take place, beyond NTK theory?

Q2. Does the selected model provide good generalization or not [27, 37]?

In contrast, when a0 is small, gradient-based algorithms can learn non-linear low-dimensional representation of the data [5, 21, 1, 6]. In these results, the difference between train and test error (generalization error) is negligible: the model does not overfit.

Q3. Can we reconcile this feature-learning/no-overfitting behavior with the lazytraining/overfitting regime described previously?

In the early phase of training, the generalization error vanishes. However, training longer times can be beneficial, despite leading to overfitting.

Q4. When does the test error start increasing with training time? When should we stop training?

Finally, scaling with the network size is crucial:
Q5. How does the generalization error depend on network size and number of iterations? Q6. Does overfitting start earlier for larger networks or later?

In Section 2, we will present our analysis using theoretical physics techniques. Section 3 presents rigorous results confirming the picture emerging from this analysis. Finally, in Section 4 we discuss how our results address the above questions.

## 2 Main Results: Dynamical Mean Field Theory

We study the dynamics of model (1.1) under the simplest data distribution in which genuine non-linear learning is required to efficiently learn a good prediction rule, the so called k*-index model*. Namely, we assume xi ∼ N(0, Id) and yithat depends on a low-dimensional projection U
Txi:

$$y_{i}=\varphi(\mathbf{U}^{\mathsf{T}}\mathbf{x}_{i})+\varepsilon_{i}\,,\quad\varepsilon_{i}\sim\mathsf{N}(0,\tau^{2})\,,$$
$\eqref{eq:walpha}$
Txi) + εi, εi ∼ N(0, τ 2), (2.1)
where the noise εiis independent of xi, U ∈ R
d×kis an orthogonal matrix (U
TU = Ik) and φ : R
k → R is a nonlinear function, E{φ(g)
2} < ∞ for g standard Gaussian.

An important aspect of this data distribution is that (for large d) it presents the largest possible gap between linear/kernel learning, which requires sample size to be superpolynomial in d [27, 58], and nonlinear/neural network learning which only requires n = O(d) (generically, for constant k).

When the dimension d becomes large, discovering the latent features U
Tx is crucial for learning and requires nonlinear processing of the labels yi [5, 21, 1, 6].

Our main focus will be on the simplest case, namely k = 1, with φ a generic function (in particular E{φ(G)G} ̸= 0 for G ∼ N(0, 1), which corresponds *information exponent* equal to one according to the classification of [4].). Some of our results apply to k-index models for general fixed k (in particular, the rigorous results of Section 3). We defer to future work a more complete analysis of the DMFT for k ≥ 2.

We discover a separation of time scales at large m (or large n/d), for sufficiently small initialization a0: feature learning takes place on a fast time scale, followed by overfitting/reversal to kernel learning.

This scenario is summarized in Figure 1, which plots numerical evaluations of our theoretical predictions at k = 1, τ > 0 data distribution, in the limit n, d, m → ∞ at overparametrization ratio α = 0.3.

More precisely, we observe three regimes (below W2nd := a/m is the vector of second-layer weights in model (1.1)):
(i) *Mean field feature learning.* t = O(1). The network learns the low-dimensional features U
Tx; the train error and test error decrease while their difference (generalization error) is negligible; the second layer weights remain small ∥W2nd∥1 = O(1).

(ii) *Extended feature learning.* 1 ≪ t ≪ m. The train error decreases slowly; the generalization error increases is small, i.e. R(θ(t)) − Rbn(θ(t)) = o(1); the test error can evolve non-monotonically, but remains approximately constant. Second-layer weights become large 1 ≪ ∥W2nd∥1 ≪
√m.

(iii) *Overfitting and feature unlearning.* t ≳ m. Train error and test error diverge significantly, i.e.

R(θ(t)) − Rbn(θ(t)) becomes of order one. At the end of this regime, the train error converges to 0, i.e. the neural network interpolates the noisy data. The test error instead grows, and its limit value is the one of a (data independent) kernel method: in other words, the model unlearns the low-dimensional structure. Finally, the second weights grow to ∥W2nd∥1 ≍
√m, which indeed is the scale required for interpolation. In this section we outline our results based on 'dynamical mean field theory' (DMFT). The next section will present rigorous results that are proven independently.

## 2.1 Technique

Our DMFT analysis is based on the following two steps: Step 1: We leverage techniques from theoretical physics to derive an approximate asymptotic characterization of the gradient flow dynamics (1.2) in the limit n, d → ∞, with n/d → α. This characterization consists of a set of integral-differential equations for the following asymptotic quantities (here p-lim denotes limit in probability, and we use the superscripts n to emphasize the dependence of the right-hand side on n, d)

$$\begin{array}{l}{{C_{i j}(t_{1},t_{2}):=\operatorname*{p-lim}_{n,d\to\infty}\langle\mathbf{w}_{i}^{n}(t_{1}),\mathbf{w}_{j}^{n}(t_{2})\rangle\,,}}\\ {{\mathbf{v}_{i}(t):=\operatorname*{p-lim}_{n,d\to\infty}\mathbf{U}^{\mathsf{T}}\mathbf{w}_{i}^{n}(t)\,,\quad a_{i}(t):=\operatorname*{p-lim}_{n,d\to\infty}a_{i}^{n}(t)\,.}}\end{array}$$
$$(2.2)$$

A rigorous derivation of the DMFT in a setting that includes two-layer networks is given in [13].

However, the asymptotically exact DMFT characterization of [13] is rather complex to integrate numerically or to study analytically. In order to circumvent this problem, we use a DMFT that is is asymptotically exact for a well-defined Gaussian version of the original model. Namely, we observe that the empirical risk of Eq. (1.2) takes the form

$$\widehat{\mathcal{R}}_{n}(\mathbf{\theta})=\frac{1}{2n}\big\|\mathbf{F}(\mathbf{\theta})\big\|^{2}\,,$$
$$(2.3)$$
2, (2.3)
where F : (S
d−1)
m×R
m → R
n is s stochastic process with i.i.d. components Fi(θ) = yi−f(xi; θ).

We replace these by Gaussian processes with matching mean and covariance, and study the DMFT
for gradient flow with respect to the associated risk Rbgn(θ).

The Gaussian approximation comes with an error which we show analytically is vanishing on time scales of order one ( indeed on these time scales we correctly recover the mean field theory of [38, 14])
and we demonstrate empirically to be small on larger time scales ( see for instance example Fig. 4.) The curves in Fig. 1 were obtained by solving numerically the DMFT equations, see Appendix C for details.

Step 2: We study this DMFT, with special attention to the large network limit m → ∞, and large sample size α → ∞, with α = α/m fixed, for a generic single index model (k = 1). We obtain a separation of time scales in the dynamics, corresponding to distinct learning regimes.

10−1 100 101 102 103 t 0 1 2 3 4 5 6 10−1 100 101 102 103 t 0.00 0.05 0.10 0.15 0.20 0.25 m = 24 m = 25 m = 26 m = 27 m = 28 m = 29 Tr ai n error a m = 24 m = 25 m = 26 m = 27 m = 28 m = 29
The analysis of the DMFT equations in the double limit m, t → ∞ is an example of singular perturbation theory [9, 29]. Making this type of analysis rigorous is notoriously challenging and we proceed by a combination of numerical solutions and analytical derivations. In the following, we will first consider the simplest possible setting, pure noise data, and subsequently consider the single-index model. The structure of the activation function and target nonlinearity will be encoded in the functions h(q) := E{σ(G1)σ(Gq)}, φb(q) := E{φ(G1)σ(Gq)} ,
where G1, Gq are standard jointly Gaussian with E{G1Gq} = q. The relation between *σ, φ* P
and h, φb is conveniently expressed in terms of the expansions in Hermite polynomials σ(x) =
k≥0skHek(x), φ(x) = Pk≥0fkHek(x), which corresponds to the analytic expansion h(q) =
Pk≥0s 2 kq k, φb(q) = Pk≥0skfkq k.

As mentioned above, we assume throughout n, d → ∞, with n/d → α ∈ (0, ∞), with the limit m, α → ∞ taken afterwards. To further simplify our analysis, we assume a symmetric initialization whereby ai(0) = a0 is independent of i ≤ m and (wi(0) : i ≤ m) ∼iid Unif(S
d−1). Throughout, we use 'with high probability' for 'with probability converging to one as n, d → ∞.'
In Section 3 we present rigorous results that do not require either of these simplifying assumptions.

## 2.2 Training On Pure Noise

We begin by the case in which the data is pure noise: yi = εi ∼ N(0, τ 2). A by-now-classic experiment [60] showed that deep learning models have sufficient capacity to achieve vanishing training error even when actual labels are replaced by random ones: they 'interpolate pure noise.'
The ability of a model FΘ = (f(· ; θ) : θ ∈ Θ) to interpolate pure noise is intimately connected to its Gaussian complexity G(FΘ; n) := E supθ∈Θ⟨g, f(X; θ)⟩/n [53] (where g ∼ N(0, In)
is independent of f(X, ; θ) = (f(xi; θ) : i ≤ n). Indeed, interpolation is impossible unless G(FΘ; n) ≥ τ . Viceversa, G(FΘ; n) ≪ τ ensures good generalization.

By a theorem of [7] for the network (1.1), G(FΘ; n) ≤ Lσ∥a/m∥1 pd/n (with Lσ depending uniquely on σ). This means that, in order to interpolate noise, the average magnitude of second layer weights must be ∥a/m∥1 ≥ L
−1 στpn/d = (L
−1 σ α 1/2)τ
√m.

However, complexity bounds do not have implications on the convergence of GF to an interpolator.

Figure 2 compares the DMFT predictions to simulations using SGD to train an actual two layer networks. In this figure we initialize a(0) = 1, and let a(t) evolve with GF alongside the first layer weigths. We observe that the theory describes well the empirical results, despite the Gaussian

0 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 0 0.2 0.4 0.6 0.8 1 1.2 1.4 m = 23 m = 24 m = 25 m = 26 m = 27 m = 28 m = 29 m = 210 m = 211 e lz1 tr(tm, 1)
Trai n/
Tes t er ror Trai n/
Tes t er ror m = 23 m = 24 m = 25 m = 26 m = 27 m = 28 m = 29 m = 210 m = 211 e lz2 tr(t, 1)
1 10 100 1000 10−4 10−3 10−2 10−1 1 10 102 103 t tm
approximation in our DMFT and the difference between SGD and GF. We also observe that secondlayer weights remain roughly constant until a large time t\#(m), which appears to increase with m.

Roughly at the same time, train error starts to decrease and converges to zero. In Section G.1 of the appendix, we will make precise the above picture of the evolution of a(t). Here, we consider a simplified setting in which a(t) = γ
√m with γ independent of m, not evolving with training. Note that G(FΘ; n) ≍ γ/√α and hence such a network can interpolate pure noise if γ is larger than threshold depending on α. Our DMFT predicts a sharp phase transition. For α ∈ (0, 1),
GF converges to vanishing train error with high probability if γ > γGF(*α, m*)τ , and converges to a strictly positive training error if *γ < γ*GF(α, m)τ . The threshold γGF(*α, m*) converges to a limit γ
∗
GF(α) ∈ (0, 1) as m → ∞.

A rephrasing of the same phenomenon states that limn,d→∞ Rbgn(θ(t)) = etr(t; *m, γ*), and

$$\lim_{t\to\infty}\lim_{m\to\infty}e_{\text{tr}}(t;m,\gamma_{0})=\begin{cases}e_{*}(\gamma)>0&\text{for}\gamma<\gamma_{\text{GF}}^{*}(\alpha)\tau,\\ 0&\text{for}\gamma\geq\gamma_{\text{GF}}^{*}(\alpha)\tau.\end{cases}\tag{2.4}$$

Informally γ
∗
GF(α) is the minimum complexity γ for a very large network to interpolate noise via gradient flow. The functions γ
∗
GF(α), e∗(γ) will play an important role below.

We will next consider training on data from a single-index model. The initial scale of secondlayer weights ∥a(0)/m∥1 plays a crucial role and we will separately analyze lazy and mean field initializations.

## 2.3 Training On Data With Latent Structure: Lazy Initialization

We initialize a(0) = γ0
√m, and let a(t) evolve according to GF alongside first-layer weights. DMFT
predicts the emergence of three dynamical regimes for large m and large α (with n/d → α). For an illustration, we refer to Fig. 3.

First dynamical regime: t = O(1/m). Second layer weights do not change significantly γ(t) =
γ0 + om(1), while first layer-weights move by ∥wi(t) − wi(0)∥ = Θ(1/
√m). Because the weights ai(t) are of order 
√m, even an O(1/
√m) change in the wileads to a significant decrease in test error and train error. Train and test error are close to each other. Namely, the following limits are well defined

$$\lim_{n,d\to\infty}\widehat{\mathscr{R}}^{g}_{n}(\mathbf{\theta}(t))=e_{\mbox{tr}}(t;\varphi,\gamma_{0},m,\alpha)\,,\qquad\lim_{n,d\to\infty}\widehat{\mathscr{R}}^{g}(\mathbf{\theta}(t))=e_{\mbox{ts}}(t;\varphi,\gamma_{0},m,\alpha)\,.\tag{2.5}$$

with limm→∞ etr(t/mˆ ; φ, γ0*, m, α*) = limm→∞ ets(t/mˆ ; φ, γ0, m, α*) =:* e

$\epsilon=:e^{\mathbb{i}t1}(\hat{t};\varphi,\gamma_0,\alpha)$. 
For large scaled time tˆ, the error e lz1(tˆ; φ, γ0, α) converges to the error of the best linear approximation to f∗. This dynamical regime follows the qualitative predictions of NTK theory, and is essentially linear in the weights wi, but the time is too short for the model to overfit the data.

Second dynamical regime: t = Θ(1). Second layer weights do not change significantly: γ(t) =
γ0 + om(1), while first layer weights change significantly ∥wi(t) − wi(0)∥ = Θ(1). However they
change orthogonally to the latent subspace U and hence the test error does not change: no actual learning takes place in this regime, but the model starts to overfit the data.
More formally, train and test error have well defined limits as the network width diverges:
e
lz2
$${}^{2}(t;\varphi,\gamma_{0},\alpha):=\operatorname*{lim}_{m\to\infty}$$
etr(t; φ, γ0, m, α), elz2
ts (t; φ, γ0, α) := lim 
m→∞
ets(t; φ, γ0*, m, α*). (2.6)
$\mathfrak{su}(t;\varphi,\gamma_0,m,\alpha)$
However, the scaling function e lz2 ts (t; φ, γ0, α) for the test error is constant in time and equal to the value achieved at the end of the first dynamical regime. Namely

value achieved at the end of the first dynamical regime. Namely  $$e_{\rm cg}^{\rm l/2}(t;\varphi,\gamma_{0},\alpha)=\lim_{t\to\infty}e^{\rm l/t}(\hat{t};\varphi,\gamma_{0},\alpha)=\frac{1}{2}\left(\tau^{2}+\|\varphi\|^{2}-\frac{\|\nabla\widehat{\varphi}({\bf0})\|^{2}}{h^{\prime}(0)}+\gamma_{0}^{2}(h(1)-h^{\prime}(0))\right)\,.\tag{2.7}$$
$\sqrt[4]{24}$ 3. 
$$m,\alpha)\,.\quad(2.6)$$
Since the wi's move orthogonally to the latent space, their dynamics is equivalent (for large m) to the one in the pure noise setting, modulo a redefinition of h. The right plot in Fig. 3 illustrates this.

Third dynamical regime: t = Θ(m). The qualitative properties of this regime depend whether or not γ0 is larger than an interpolation threshold γ
∗
GF(*α, φ, τ* ), which generalizes the threshold γ
∗
GF(α) = γ
∗
GF(α, 0, 1) introduced in the pure noise case. Because the dynamics of weights wiin the subspace orthogonal to U is equivalent to dynamics in pure noise, we expect the interpolation threshold γ
∗
GF(*α, φ, τ* ) to be given in terms of pure noise threshold γ
∗
GF(α) as follows:

$$\gamma^{*}_{\rm GF}(\alpha,\varphi,\tau)=\left(\tau^{2}+\|\varphi\|^{2}-\frac{\|\nabla\hat{\varphi}({\bf0})\|^{2}}{h^{\prime}(0)}\right)^{1/2}\gamma^{*}_{\rm GF}(\alpha)\,.\tag{2.8}$$

For γ0 > γ∗GF(*α, φ, τ* ), interpolation is achieved during the second dynamical regime, no further evolution takes place.

For γ0 < γ∗GF(*α, φ, τ* ), a non-trivial evolution takes place for t = Θ(m). Introducing the rescaled time z ∈ (0, ∞), we obtain, as m → ∞,
γ(mz) = γ lz3(z) + om(1), etr(mz) = e lz3 tr (z) + om(1), ets(mz) = e lz3 ts (z) + om(1). (2.9)
Further, for large values of the rescaled time z → ∞, γ lz3(z) grows to γ
∗
GF(*α, φ, τ* ) ≈ γ
∗
GF(*α, φ, τ* ),
while e lz3 tr (z) decreases to 0. In other words, interpolation is achieved on this third regime.

Further the test error e lz3 ts (z) increases from e lz2 ts (t; φ, γ0, α) to e lz2 ts (t; φ, γ∗GF, α), with γ
∗
GF =
γ
∗
GF(*α, φ, τ* ) whereby e lz2 ts (· · ·) is given by Eq. (2.7).

## 2.4 Training On Data With Latent Structure: Mean Field Initialization

We initialize a(0) = a0, independent of m and let second layer weights evolve. Note that at initialization the network's Rademacher complexity is small, namely of order a0 pd/n = a0/
√αm.

Our DMFT analyisis predicts two dynamical regimes for large m. We will refer to them as 'first' and
'third regime' for consistency with other settings ( see Sec.G.2 of the appendix). For an illustration, we refer to Figs. 4 and 5.

First dynamical regime: t = O(1). Both first and second layer weights change by order one:
a(t) = a0 + Θ(1) and ∥wi(t) − wi(0)∥ = Θ(1). and as a consequence test and train error decrease significantly. In this regime, the two errors remain close to each other and their evolution is well captured by the mean field theory of [38, 14], as specialized to the case of spherically invariant distributions [10, 2].

Namely, limm→∞ a(t) = a mf1(t), limm→∞ v(t) = v mf1(t), and DMFT reduces to a system of k + 1 ordinary differential equations for the k + 1 scalar variables (a mf1(t), v mf1(t))

$$\partial_{t}\mathbf{v}^{\rm{\it{m}t}}(t)=\alpha a^{\rm{\it{m}t}}(t)\mathbf{Q}_{\mathbf{v}^{\rm{\it{m}t}}(t)}\Big{(}\nabla\hat{\varphi}(\mathbf{v}^{\rm{\it{m}t}}(t))-a^{\rm{\it{m}t}}(t)h^{\prime}(\|\mathbf{v}^{\rm{\it{m}t}}(t)\|^{2})\mathbf{v}^{\rm{\it{m}t}}(t)\Big{)}\,,\tag{2.10}$$ $$\partial_{t}a^{\rm{\it{m}t}}(t)=\alpha\hat{\varphi}(\mathbf{v}^{\rm{\it{m}t}}(t))-\alpha a^{\rm{\it{m}t}}(t)h(\|\mathbf{v}^{\rm{\it{m}t}}(t)\|^{2})\,,$$

where Qv:= Ik − vvT. As mentioned above, train and test error coincide in the large width limit
$$\operatorname*{lim}_{m\to\infty}e_{\mathrm{tr}}(t)=\operatorname*{lim}_{m\to\infty}$$
m→∞
ets(t) = e
mf1(t).

101 0.0 0.2 0.4 0.6 0.8 m = 25 m = 26 m = 27 m = 28 m = 29 Train
/Te st a m = 25 m = 26 m = 27 m = 28 m = 29 100 101 102 103 t 100 100 101 102 103 t
An explicit formula for e mf1(t) is given in Appendix G.2.1. In the case k = 1 and φb(z) = h(z),
we have that a mf1 = 1, v mf1 = 1 is a fixed point of Eq. (2.10), and indeed the only fixed point with v mf1 > 0. If h
′(0) > 0, then, we have (a mf1(t), vmf1(t)) → (1, 1) as t → ∞, and therefore test and train error converge to the Bayes error e mf1(t) → τ 2/2. This is significantly smaller than the test error achieved with lazy initialization. The separation between lazy and mean-field initialization is expected because feature learning takes place in the mean field regime.

Third dynamical regime: t = Ω(m). Computing the local stability of DMFT solutions around the mean field asymptotics (see Appendix G.2.2) suggests that the latter breaks down for t *= Θ(*m). For t ≳ m, we observe that the second layer weights grow to achieve a(t) ≍
√m, the projection onto the latent space decreases to v(t) ≍ 1/
√m, and train and test error diverge, eventually achieving etr(t) ≈ 0 and test error significantly larger than the Bayes error achieved earlier. We refer to this phenomenon as 'feature unlearning.'
Denoting by t0(m; c) the time at which a(t) = c
√m (for c a small constant), we expect the existence of a window size w(m) such that

$$\lim_{m\to\infty}\frac{a\big{(}t_{0}(m;c)+z\,w(m)\big{)}}{\sqrt{m}}=\gamma^{m0}(z)\,,\qquad\lim_{m\to\infty}e_{\mathbf{U}/\mathbf{S}}\big{(}t_{0}(m;c)+z\,w(m)\big{)}=e_{\mathbf{U}/\mathbf{S}}^{m0}(z)\,,\tag{2.11}$$

where γ mf3(z), e mf3 tr (z), e mf3 ts (z) are scaling functions describing the dynamics on this timescale. We expect t0(m; c) = t∗(c)m + o(m), and w(m) ≲ t0(m; c), but our numerical solutions are not sufficient to determine the precise scaling. On the other hand, it appears that at large times, the complexity converges close the interpolation threshold:
GF(*α, φ, τ* ). (2.12)

lim z→∞ γ
$$(z)=\overline{{{\gamma}}}_{\mathrm{GF}}^{*}(\alpha,\varphi,\tau)\approx\gamma_{\mathrm{GF}}^{*}(\alpha,\varphi,\tau)\,.$$
Finally, the evolution of train and test error for a(t) ≍
√m appears to match the behavior at fixed second-layer weights. Namely, we define two functions ε mf tr/ts(γ) := lim m→∞
etr/ts(t0(m; γ), m). (2.13)
We observe that the limit curves (*γ, ε*mf tr(γ)), (*γ, ε*mf ts(γ)), match closely asymptotic train and test error obtained by fixing a(t) = γ
√m, and not letting second-layer weight evolve. This confirms the hypothesis that γ(t) is a slow variable, while others converge as if γ was fixed.

## 3 Lower Bounding The Overfitting Timescale

In this section we rigorously establish two results that confirm elements of the scenario outlined in the previous sections. We emphasize that the result presented here are non-asymptotic, i.e. hold at finite

$$(2.12)$$

$$(2.13)$$
$\frac{1}{4}$ . 
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 m = 23 m = 24 m = 25 m = 26 m = 27 m = 28 m = 29 m = 210 m = 211 m = 212 m = 213 m = 214 m = 23 m = 24 m = 25 m = 26 m = 27 m = 28 m = 29 m = 210 m = 211 m = 212 m = 213 m = 214 m = 23 m = 24 m = 25 m = 26 m = 27 m = 28 m = 29 m = 210 m = 211 m = 212 m = 213 m = 214 Lazy a(t)
/√
m ets
− et r v(t
)

10−3 10−2 10−1 1 10 102 0.001 0.01 0.1 1 10 100 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

$$(3.1)$$

a/√m t/m t/m
n, m, d modulo unspecified absolute constants. Further, we do not assume a symmetric initialization of the weights. Throughout this section setting, it is more convenient to rescale time defining tˆ= tα.

Hence. instead of the flow (1.2), we study

$$\dot{\theta}(\hat{t})=-m{\cal P}_{\theta}\nabla\hat{\mathcal{R}}_{n}(\theta(\hat{t}))\,.$$

θ˙(tˆ) = −mP θ∇Rbn(θ(tˆ)). (3.1)
For α = Θ(1) the parametrizations t and tˆare equivalent.

The first result of this section implies that (under mean field initialization) overfitting cannot take place on times of order one.

Theorem 1. *Under the GF dynamics* (1.2), and the data distribution in the introduction (with k arbitrary), further assume ∥σ∥Lip, ∥σ∥∞ ≤ L, |φ(0)|, ∥φ∥Lip ≤ L, ∥a(0)∥∞ ≤ a0*, for some* a0 ≥ 1 and that the wi(0), i ≤ m *are independent of the data* {(yi, xi) : i ≤ n}*. Finally assume* n ≥ d∨ m.

Then, there exist universal constants C0, C1*, and the following holds for all* tˆ≥ 0, Under mean field initialization, a0 is a fixed constant and hence a1 is also bounded, whence the generalization error in Eq. (3.3) is small as long as tˆ = o((n/d)
1/4) (equivalently, for α fixed, tˆ= o(m1/4)).

By itself, this result implies a separation of timescales between learning and overfitting, thus confirming the picture developed within DMFT, but falls short of characterizing the overfitting timescale. The second result implies that, up to time-scale of order one, the dynamics is closely tracked by the mean field equations (2.10). Since the ai(0) at initialization are not necessarily all equal, these are generalized as

∂tˆv mf1 i(tˆ) = a mf1 i(tˆ)Qv mf1 i(tˆ) ∇φˆ(v mf1 i(tˆ)) − 1 m Xm j=1 a mf1 j(tˆ)h ′(⟨v mf1 i(tˆ), v mf1 j(tˆ)⟩)v mf1 j(tˆ) , ∂tˆa mf1 i(tˆ) = ˆφ(v mf1 i(tˆ)) − 1 m Xm j=1 a mf1 j(tˆ)h(⟨v mf1 i(tˆ), v mf1 j(tˆ)⟩). (3.4)
$$(3.2)$$
$$(3.3)$$
The mean field prediction for test error is the same as for training error and given by

$$e_{\mathrm{ts}}(\hat{t})=\frac{1}{2}\|\varphi\|_{L^{2}}^{2}-\frac{1}{m}\sum_{j=1}^{m}a_{j}^{\mathrm{an}}(\hat{t})\phi(\mathbf{v}_{j}^{\mathrm{an}}(\hat{t}))+\frac{1}{2m^{2}}\sum_{j=1}^{m}a_{i}^{\mathrm{an}}(\hat{t})a_{j}^{\mathrm{an}}(\hat{t})\,h(\langle\mathbf{v}_{i}^{\mathrm{an}}(\hat{t}),\mathbf{v}_{j}^{\mathrm{an}}(\hat{t})\rangle)$$  **Lemma 2**: _It has the CF between (1) and (2) and the Little distribution in the interval $(\alpha,\beta)$._
Theorem 2. *Under the the GF dynamics* (1.2), and the data distribution in the introduction (with k arbitrary), further assume that ∥φ∥∞, ∥φ
′∥∞, ∥φ
′∥Lip ≤ L, ∥σ∥∞, ∥σ
′∥∞, ∥σ
′∥Lip ≤ L. Further

∥a(tˆ)∥∞ ≤ a0 + a1 t , a ˆ 1 := C0L(τ + √k + a0L), (3.2) R(a(tˆ),W(tˆ)) − Rbn(a(tˆ),W(tˆ))  ≤ C1(L 2(a0 + a1tˆ) 2 + τ 2) · rd n
. (3.3)
assume |ai(0)| ≤ L for all i ≤ m, (wi(0))i≤m ∼iid Unif(S
d−1). Then for any δ > 0 *there exist* constants c0 c1, C depending on L, τ, δ, k *such that, letting* Tlb = c0(log m)
1/3 ∧ (log n/d)
1/3*, the* following happens with probability at least 1 − 2 exp(−c1d),

$$\sup_{t\leq T_{\rm fb}}\frac{1}{m}\sum_{i=1}^{m}\left(|a_{i}(\hat{t})-a_{i}^{\rm mI}(\hat{t})|+\|\mathbf{v}_{i}(\hat{t})-\mathbf{v}_{i}^{\rm mI}(\hat{t})\|\right)\leq C\left(\frac{1}{m}\vee\frac{1}{d}\vee\frac{d}{n}\right)^{1/2-\delta},\tag{3.5}$$  $$\sup_{t\leq T_{\rm fb}}\left|\mathscr{R}(\mathbf{a}(\hat{t}),\mathbf{W}(\hat{t}))-e_{\rm tS}(\hat{t})\right|\leq C\left(\frac{1}{m}\vee\frac{1}{d}\vee\frac{d}{n}\right)^{1/2-\delta}.\tag{3.6}$$

Remark 3.1. While the analysis in the previous section requires m → ∞ after n, d → ∞, neither Theorem 3.1 nor Theorem 3.2 make the assumption. In particular, Eq. (3.3) implies that the generalization error is small for tˆ= o((n/d)
1/4) *irrespective of* m.

Similarly, Eqs. (3.5), (3.6) imply that the mean field theory of [38, 14, 45] captures well the evolution of the system for times t = o((log m)
1/3 ∧ (log n/d)
1/3).

## 4 Discussion

We conclude by highlighting a few qualitative conclusions of our work, and how they address questions raised in Section 1. In the following remarks, we consider α = *n/md* as constant.

Interpolation mechanism. In the current setting, the neural model complexity is proportional to
∥a(t)∥1/
√m = γ(t)+on(1). We observe two alternative scenarios. If the complexity at initialization is large enough γ0 > γ∗
GF(α)τ , then the gradient flow rapidly converges to a near interpolator without significant change in γ(t). If instead, γ0 < γ∗GF(α)τ , then γ(t) grows to reach the interpolation threshold at which point the training error converges to 0. Adiabatic evolution of model complexity. In the latter case, the complexity γ(t) evolves on a slower time scale than other degrees of freedom. The dynamics on shorter timescales is well approximated by the one at fixed γ (given by the current value γ(t)). The generalization error becomes of order one only when γ(t) is of order one.

Decoupling of learning and overfitting. When γ0 = om(1), the fact that γ(t) acts as a slow variable implies a large-m decoupling between learning (which takes place on faster timescales, as long as γ(t) = om(1)), and overfitting (which takes place on slower timescales, when γ(t) = Ωm(1)). This has several implications for the questions outlined in the introduction.

Q3: Lazy initialization a(0) ≍
√m leads to poor generalization because the feature-learning phase is skipped either partially or altogether. Q2: Training until interpolation is generally suboptimal. Q4: The optimal tradeoff is obtained at the end of the first phase. Q5, Q6: Further, at fixed overerparametrization *n/md* = α, overfitting starts later for larger models.

Overfitting and feature unlearning. The above description points at a non-monotonicity of the model quality, which improves on short time scales, and deteriorates at larger time scales. Reciprocally, early stopping acts as a regularization. While this phenomenon is well understood for linear models [24, 57], our analysis provides an analogous (quantitative) scenario for training neural network models. In particular, it clarifies the underlying mechanism: in the same dynamical regime in which network complexity grows (γ(t) becomes of order one), and training error becomes negligible, the low-dimensional latent features are 'unlearned' (v(t) becomes of order 1/
√m). We expect that these findings also allow to understand the beneficial effect of regularization on the second layer.

## Acknowledgments

This work was supported by the NSF through award DMS-2031883, the Simons Foundation through Award 814639 for the Collaboration on the Theoretical Foundations of Deep Learning, and the ONR
grant N00014-18-1-2729. This work was supported by the French government under the France 2030 program (PhOM - Graduate School of Physics) with reference ANR-11-IDEX-0003.

## References

[1] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. The merged-staircase property:
a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In *Conference on Learning Theory*, pages 4782–4887. PMLR, 2022.

[2] Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro. From highdimensional and mean-field dynamics to dimensionless odes: A unifying approach to sgd in two-layers networks. In *The Thirty Sixth Annual Conference on Learning Theory*, pages 1199–1227. PMLR, 2023.

[3] Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. *Advances in Neural Information Processing Systems*, 32, 2019.

[4] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. *Journal of Machine Learning Research*, 22(106):1–51, 2021.

[5] Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. Highdimensional asymptotics of feature learning: How one gradient step improves the representation.

Advances in Neural Information Processing Systems, 35:37932–37946, 2022.

[6] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang.

Hidden progress in deep learning: SGD learns parities near the computational limit. Advances in Neural Information Processing Systems, 35:21750–21764, 2022.

[7] Peter Bartlett. For valid generalization the size of the weights is more important than the size of the network. *Advances in neural information processing systems*, 9, 1996.

[8] Gérard Ben Arous, Amir Dembo, and Alice Guionnet. Cugliandolo-kurchan equations for dynamics of spin-glasses. *Probability theory and related fields*, 136(4):619–660, 2006.

[9] Nils Berglund. Perturbation theory of dynamical systems. *arXiv preprint math/0111178*, 2001.

[10] Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. *Foundations of Computational Mathematics*, pages 1–84, 2024.

[11] Christopher M Bishop. Regularization and complexity control in feed-forward networks. In Proceedings International Conference on Artificial Neural Networks ICANN'95, pages 141–148, 1995.

[12] Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. *Advances in Neural Information Processing Systems*, 35:32240–32256, 2022.

[13] Michael Celentano, Chen Cheng, and Andrea Montanari. The high-dimensional asymptotics of first order methods with random data. *arXiv:2112.07572*, 2021.

[14] Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for overparameterized models using optimal transport. Advances in neural information processing systems, 31, 2018.

[15] Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In *Conference on learning theory*, pages 1305–1338. PMLR, 2020.

[16] Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. *Advances in neural information processing systems*, 32, 2019.

[17] Andrea Crisanti, Heinz Horner, and H J Sommers. The spherical p-spin interaction spin-glass model: the dynamics. *Zeitschrift für Physik B Condensed Matter*, 92:257–271, 1993.

[18] Leticia F Cugliandolo. Recent applications of dynamical mean-field methods. Annual Review of Condensed Matter Physics, 15, 2023.

[19] Leticia F Cugliandolo and David S Dean. Full dynamical solution for a spherical spin-glass model. *Journal of Physics A: Mathematical and General*, 28(15):4213, 1995.

[20] Leticia F Cugliandolo and Jorge Kurchan. Analytical solution of the off-equilibrium dynamics of a long-range spin-glass model. *Physical Review Letters*, 71(1):173, 1993.

[21] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In *Conference on Learning Theory*, pages 5413–5452. PMLR,
2022.

[22] Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In *International conference on machine learning*, pages 1675–1685. PMLR, 2019.

[23] Giampaolo Folena, Silvio Franz, and Federico Ricci-Tersenghi. Rethinking mean-field glassy dynamics and its relation with the energy landscape: The surprising case of the spherical mixed p-spin model. *Physical Review X*, 10(3):031045, 2020.

[24] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Additive logistic regression: a statistical view of boosting (with discussion and a rejoinder by the authors). *The annals of statistics*,
28(2):337–407, 2000.

[25] Yan V Fyodorov. A spin glass model for reconstructing nonlinearly encrypted signals corrupted by noise. *Journal of Statistical Physics*, 175:789–818, 2019.

[26] Yan V Fyodorov and Rashel Tublin. Optimization landscape in the simplest constrained random least-square problem. *Journal of Physics A: Mathematical and Theoretical*, 55(24):244008, 2022.

[27] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. *The Annals of Statistics*, 49(2), 2021.

[28] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249–256. JMLR Workshop and Conference Proceedings, 2010.

[29] Mark Holmes. *Introduction to Perturbation Methods*. Springer, 2013. [30] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. *Advances in neural information processing systems*, 31, 2018.

[31] Persia Jana Kamali and Pierfrancesco Urbani. Dynamical mean field theory for models of confluent tissues and beyond. *SciPost Physics*, 15(5):219, 2023.

[32] Persia Jana Kamali and Pierfrancesco Urbani. Stochastic gradient descent outperforms gradient descent in recovering a high-dimensional signal in a glassy energy landscape. *arXiv preprint* arXiv:2309.04788, 2023.

[33] Jaron Kent-Dobias. On the topology of solutions to random continuous constraint satisfaction problems. *arXiv preprint arXiv:2409.12781*, 2024.

[34] Yuanzhi Li, Colin Wei, and Tengyu Ma. Towards explaining the regularization effect of initial large learning rate in training neural networks. *Advances in neural information processing* systems, 32, 2019.

[35] Stefano Sarao Mannelli, Florent Krzakala, Pierfrancesco Urbani, and Lenka Zdeborova. Passed
& spurious: Descent algorithms and local minima in spiked matrix-tensor models. In international conference on machine learning, pages 4333–4342. PMLR, 2019.

[36] Andreas Maurer. A vector-contraction inequality for Rademacher complexities. In Algorithmic Learning Theory: 27th International Conference, pages 3–17. Springer, 2016.

[37] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration. Applied and Computational Harmonic Analysis, 59:3–84, 2022.

[38] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. *Proceedings of the National Academy of Sciences*, 115(33):E7665– E7671, 2018.

[39] Marc Mézard, Giorgio Parisi, and Miguel Angel Virasoro. *Spin glass theory and beyond*,
volume 9. World Scientific, 1987.

[40] Francesca Mignacco, Florent Krzakala, Pierfrancesco Urbani, and Lenka Zdeborová. Dynamical mean-field theory for stochastic gradient descent in gaussian mixture classification. Advances in Neural Information Processing Systems, 33:9540–9550, 2020.

[41] Francesca Mignacco and Pierfrancesco Urbani. The effective noise of stochastic gradient descent. *Journal of Statistical Mechanics: Theory and Experiment*, 2022(8):083405, 2022.

[42] Andrea Montanari and Eliran Subag. Solving overparametrized systems of random equations: I.

model and algorithms for approximate solutions. *arXiv:2306.13326*, 2023.

[43] Andrea Montanari and Eliran Subag. On Smale's 17th problem over the reals. *arXiv:2405.01735*,
2024.

[44] Nelson Morgan and Hervé Bourlard. Generalization and parameter estimation in feedforward nets: Some experiments. *Advances in neural information processing systems*, 2, 1989.

[45] Grant Rotskoff and Eric Vanden-Eijnden. Trainability and accuracy of artificial neural networks:
An interacting particle system approach. *Communications on Pure and Applied Mathematics*, 75(9):1889–1935, 2022.

[46] Mark Sellke. The threshold energy of low temperature Langevin dynamics for pure spherical spin glasses. *Communications on Pure and Applied Mathematics*, 77(11):4065–4099, 2024.

[47] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge University Press, 2014.

[48] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. *The Journal of Machine Learning Research*,
19(1):2822–2878, 2018.

[49] Eliran Subag. Concentration for the zero set of random polynomial systems. arXiv preprint arXiv:2303.11924, 2023.

[50] Michel Talagrand. *Mean field models for spin glasses: Volume I: Basic examples*, volume 54.

Springer Science & Business Media, 2010.

[51] Pierfrancesco Urbani. A continuous constraint satisfaction problem for the rigidity transition in confluent tissues. *Journal of Physics A: Mathematical and Theoretical*, 56(11):115003, 2023.

[52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, pages 5998–6008. Curran Associates, Inc.,
2017.

[53] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

[54] Nikhil Vyas, Yamini Bansal, and Preetum Nakkiran. Limitations of the ntk for understanding generalization in deep learning. *arXiv preprint arXiv:2206.10012*, 2022.

[55] Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. Advances in neural information processing systems, 30, 2017.

[56] Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In *Conference on Learning Theory*, pages 3635–3673. PMLR, 2020.

[57] Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning. *Constructive Approximation*, 26(2):289–315, 2007.

[58] Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding neural networks. *Advances in neural information processing systems*, 32, 2019.

[59] Kaichao You, Mingsheng Long, Jianmin Wang, and Michael I Jordan. How does learning rate decay help modern neural networks? *arXiv preprint arXiv:1908.01878*, 2019.

[60] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3):107–
115, 2021.

[61] Tong Zhang and Bin Yu. Boosting with early stopping: Convergence and consistency. Annals of Statistics, pages 1538–1579, 2005.

[62] Jean Zinn-Justin. *Quantum field theory and critical phenomena*. Oxford University Press, 2021.

## Neurips Paper Checklist 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] , Justification: We conduct a theoretical analysis that is described by the abstract and that answers the questions detailed in the introduction. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: In the introduction section we discuss how the contribution compares to previous literature and limitations related to the use of non-rigorous mathematical techniques. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: We conduct a theoretical analysis of training dynamics. The method that we use is non-rigorous but well established in theoretical physics. We show that the method correctly reproduces observations and it is checked against simulations. We prove two theorems that support our analysis. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We detail the numerical simulations in the appendix Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: Our paper is theoretical in nature and simulations are fairly standard and only play a support role. Guidelines:
- The answer NA means that paper does not include experiments requiring code. - While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The theoretical results and figures are detailed with the corresponding settings that we used to produce them. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] . Justification: The paper contains all the details about the numerical simulations we used. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA] .

Justification: There are no extensive or complex experiments we have performed. The paper is theoretical in nature and aims at understanding simple yet paradigmatic models. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Answer: [Yes] Justification: We conform with the NeurIPS Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] . Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] . Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications. Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] . Justification: We do not use existing datasets or codes. Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

13. **New assets**
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] . Justification: We do not produce any new asset. Our study is purely theoretical in nature. Guidelines:
- The answer NA means that the paper does not release new assets. - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications. We do not perform experiments with humans. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human** Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: We do not conduct experiments with humans. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

16. **Declaration of LLM usage**