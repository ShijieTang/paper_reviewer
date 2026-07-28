# Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks

Andrea Montanari Department of Statistics and Department of Mathematics, Stanford University

Pierfrancesco Urbani Université Paris-Saclay, CNRS, CEA, Institut de Physique Théorique, 91191, Gif-Sur-Yvette, France

## Abstract

Understanding the inductive bias and generalization properties of large overparametrized machine learning models requires to characterize the dynamics of the training algorithm. We study the learning dynamics of large two-layer neural networks via dynamical mean field theory, a well established technique of nonequilibrium statistical physics. We show that, for large network width m, and large number of samples per input dimension n/d, the training dynamics exhibits a separation of timescales which implies: (i) The emergence of a slow time scale associated with the growth in Gaussian/Rademacher complexity of the network; (ii) Inductive bias towards small complexity if the initialization has small enough complexity; (iii) A dynamical decoupling between feature learning and overfitting regimes; (iv) A non-monotone behavior of the test error, associated 'feature unlearning' regime at large times.

## 1 Introduction

Machine learning (ML) models are trained using stochastic gradient descent (SGD), or one of its variants to minimize the error on training data (empirical risk function). Classically, their good behavior on unseen test data is explained by the fact that model complexity is kept small by regularization techniques: these models do not 'overfit.' Traditional ML theory decouples the analysis of the model from the optimization algorithm, which is assumed to converge to an approximate global minimizer [\[47\]](#page-12-0).

In contrast, in modern ML, the empirical risk is highly non-convex, the number of parameters is comparable with the number of training samples, and the model complexity is only weakly controlled. As a consequence, there can be many assignments of the model parameters (many global empirical risk minimizers) that perfectly interpolate the data —even when these are noisy. While all of these *interpolators* are indistinguishable on the training data, they behave very differently (and some of them very poorly) on test data. It has been hypothesized that models trained by SGD generalize well to test data because the algorithm selects a near global minimizer with low complexity, although a mechanistic understanding of this process is lacking. For this reason, the generalization properties cannot be decoupled from the training dynamics.

Several striking consequences of this lack of decoupling are documented in the literature (and have long been familiar to practitioners): (i) Test error after training is observed to depend strongly on the initial weights distribution [\[28\]](#page-11-0); (ii) Test error depends strongly on the optimization algorithm (SGD, RMSProp, ADAM, to name a few), even when these algorithms achieve the same train error [\[55\]](#page-12-1); (iii) Careful choice of the hyperparameters in the optimization algorithm is crucial [\[34,](#page-11-1) [59\]](#page-13-0), and the optimal choice is often different from the one that minimizes train error; (iv) Models learned by training for a shorter time have smaller complexity and can generalize better [\[44,](#page-12-2) [11\]](#page-10-0).

![](_page_1_Figure_0.jpeg)

Figure 1: Three dynamical regimes of learning in a two-layer neural networks, with m hidden neurons. Training data comprises n points in d dimensions distributed according to a single index model. We assume n, m, d all large with n/md = α (here α = 0.3). Blue: test error. Purple: train error. Red: ℓ<sup>1</sup> norm of second-layer weights (a proxy for model complexity).

These observations have motivated a broad effort to encapsulate the effect of the dynamics as 'implicit regularization' [\[48,](#page-12-3) [3,](#page-10-1) [15,](#page-10-2) [56\]](#page-13-1): the algorithm selects an empirical risk minimizer that also minimizes a specific notion of model complexity. While this *implicit regularization hypothesis* has been fruitful, it can only be validated if we can precisely understand the training dynamics.

In this work we leverage tools from theoretical physics to directly analyze the training dynamics and derive quantitative predictions on the implicit bias of neural network training, in a simple setting. This allows us to capture feature learning and lazy/overfitting regimes within the same unified picture. We discover a time-scale separation in the training dynamics, between an early stage in which the model learns the relevant features representation of the data, and a late stage of training that is characterized by overfitting, feature 'unlearning,' and hence test error that increases with training. While the regularizing effect of early stopping has been an important object of study (for simpler models) in the past [\[44,](#page-12-2) [11,](#page-10-0) [61,](#page-13-2) [57\]](#page-13-3), our work is the first to point out a time-scale separation between feature learning (on a faster timescale) and overfitting (on a slower time scale), thus reconciling the feature learning and neural tangent theories of learning.

We study two-layer fully connected neural networks <sup>f</sup>(· ; <sup>θ</sup>) : <sup>R</sup> <sup>d</sup> → <sup>R</sup>, i.e.

$$f(\mathbf{x}; \boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^m a_i \sigma(\langle \mathbf{w}_i, \mathbf{x} \rangle), \quad (1.1)$$

where <sup>θ</sup> = (a,W), where <sup>W</sup> = (w1, . . . , <sup>w</sup>m) ∈ <sup>R</sup> <sup>d</sup>×<sup>m</sup> and <sup>a</sup> = (a1, . . . , am) ∈ <sup>R</sup> <sup>m</sup> are, respectively, first- and second-layer weights. For convenience, we fix the normalization ∥<sup>w</sup>i∥ = 1, and assume that σ does not depend on m. We apply model [\(1.1\)](#page-1-0) to a supervised learning task. We are given i.i.d. data (y<sup>i</sup> , <sup>x</sup>i), <sup>i</sup> ≤ <sup>n</sup>, with <sup>y</sup><sup>i</sup> ∈ <sup>R</sup> a response variable and <sup>x</sup><sup>i</sup> ∈ <sup>R</sup> d a feature vector, and try to learn a model <sup>f</sup>(· ; <sup>θ</sup>) to predict the response <sup>y</sup>new corresponding to a new input <sup>x</sup>new. We use gradient flow (GF) to minimize the empirical risk under square loss, namely

$$\dot{\boldsymbol{\theta}}(t) = -\frac{n}{d} \mathbf{P}_{\boldsymbol{\theta}} \nabla \widehat{\mathcal{R}}_n(\boldsymbol{\theta}(t)), \quad \widehat{\mathcal{R}}_n(\boldsymbol{\theta}) := \frac{1}{2n} \sum_{i=1}^n (y_i - f(\mathbf{x}_i; \boldsymbol{\theta}))^2. \quad (1.2)$$

Here <sup>P</sup> <sup>θ</sup> is a projection matrix that guarantees that <sup>w</sup>i(t) ∈ <sup>S</sup> d−1 at all times. The factor n/d is introduced for convenience and simply amounts to a rescaling of time. We will typically initialize the training by setting (wi)i≤<sup>m</sup> ∼iid Unif(<sup>S</sup> d−1 ), and <sup>a</sup><sup>i</sup> <sup>=</sup> <sup>a</sup><sup>0</sup> for all <sup>i</sup> ≤ <sup>m</sup>, and study the dependence of the training dynamics on three key parameters:

Network width: m, Overparametrization ratio: α := n md, Initialization scale: <sup>a</sup><sup>0</sup> .

Network width: 
$$m$$
, Overparametrization ratio:  $\alpha := \frac{n}{md}$ , Initialization scale:  $a_0$ .

Alongside the train error, we will be interested in the test error at time <sup>t</sup>, i.e. <sup>R</sup>(θ(t)) := <sup>E</sup>{(ynew − <sup>n</sup>(θ(t)).

<sup>f</sup>(xnew; <sup>θ</sup>(t)))<sup>2</sup>}/2, and the generalization error <sup>R</sup>(θ(t)) <sup>−</sup> <sup>R</sup>b Model [\(1.1\)](#page-1-0) is much simpler than state-of-the-art architectures [\[52\]](#page-12-4), but is rich enough to investigate several general questions, which we summarize below:

When the network is sufficiently overparametrized (α small) and a<sup>0</sup> is large, neural tangent kernel (NTK) theory predicts that GF converges to an interpolator [\[30,](#page-11-2) [22,](#page-11-3) [16\]](#page-10-3) .

Q1. For which region of α, a<sup>0</sup> does convergence take place, beyond NTK theory? Q2. Does the selected model provide good generalization or not [\[27,](#page-11-4) [37\]](#page-12-5)?

In contrast, when a<sup>0</sup> is small, gradient-based algorithms can learn non-linear low-dimensional representation of the data [\[5,](#page-10-4) [21,](#page-11-5) [1,](#page-10-5) [6\]](#page-10-6). In these results, the difference between train and test error (generalization error) is negligible: the model does not overfit.

Q3. Can we reconcile this feature-learning/no-overfitting behavior with the lazytraining/overfitting regime described previously?

In the early phase of training, the generalization error vanishes. However, training longer times can be beneficial, despite leading to overfitting.

Q4. When does the test error start increasing with training time? When should we stop training? Finally, scaling with the network size is crucial:

Q5. How does the generalization error depend on network size and number of iterations? Q6. Does overfitting start earlier for larger networks or later?

In Section [2,](#page-2-0) we will present our analysis using theoretical physics techniques. Section [3](#page-7-0) presents rigorous results confirming the picture emerging from this analysis. Finally, in Section [4](#page-9-0) we discuss how our results address the above questions.

## 2 Main results: Dynamical mean field theory

We study the dynamics of model [\(1.1\)](#page-1-0) under the simplest data distribution in which genuine non-linear learning is required to efficiently learn a good prediction rule, the so called k*-index model*. Namely, we assume <sup>x</sup><sup>i</sup> ∼ <sup>N</sup>(0, <sup>I</sup>d) and <sup>y</sup><sup>i</sup> that depends on a low-dimensional projection U <sup>T</sup>x<sup>i</sup> :

$$y_i = \varphi(\mathbf{U}^T \mathbf{x}_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \tau^2), \quad (2.1)$$

where the noise ε<sup>i</sup> is independent of x<sup>i</sup> , <sup>U</sup> ∈ <sup>R</sup> d×k is an orthogonal matrix (U <sup>T</sup>U = Ik) and φ : <sup>R</sup> <sup>k</sup> → <sup>R</sup> is a nonlinear function, <sup>E</sup>{φ(g) <sup>2</sup>} <sup>&</sup>lt; ∞ for <sup>g</sup> standard Gaussian.

An important aspect of this data distribution is that (for large d) it presents the largest possible gap between linear/kernel learning, which requires sample size to be superpolynomial in d [\[27,](#page-11-4) [58\]](#page-13-4), and nonlinear/neural network learning which only requires n = O(d) (generically, for constant k). When the dimension d becomes large, discovering the latent features U <sup>T</sup>x is crucial for learning and requires nonlinear processing of the labels y<sup>i</sup> [\[5,](#page-10-4) [21,](#page-11-5) [1,](#page-10-5) [6\]](#page-10-6).

Our main focus will be on the simplest case, namely k = 1, with φ a generic function (in particular <sup>E</sup>{φ(G)G} ̸= 0 for <sup>G</sup> ∼ <sup>N</sup>(0, 1), which corresponds *information exponent* equal to one according to the classification of [\[4\]](#page-10-7).). Some of our results apply to k-index models for general fixed k (in particular, the rigorous results of Section [3\)](#page-7-0). We defer to future work a more complete analysis of the DMFT for <sup>k</sup> ≥ <sup>2</sup>.

We discover a separation of time scales at large m (or large n/d), for sufficiently small initialization a0: feature learning takes place on a fast time scale, followed by overfitting/reversal to kernel learning. This scenario is summarized in Figure [1,](#page-1-1) which plots numerical evaluations of our theoretical predictions at <sup>k</sup> = 1, τ > <sup>0</sup> data distribution, in the limit n, d, m → ∞ at overparametrization ratio α = 0.3.

More precisely, we observe three regimes (below W2nd := a/m is the vector of second-layer weights in model [\(1.1\)](#page-1-0)):

(i) *Mean field feature learning.* t = O(1). The network learns the low-dimensional features U <sup>T</sup>x; the train error and test error decrease while their difference (generalization error) is negligible; the second layer weights remain small ∥<sup>W</sup>2nd∥<sup>1</sup> <sup>=</sup> <sup>O</sup>(1).

(ii) *Extended feature learning.* <sup>1</sup> ≪ <sup>t</sup> ≪ <sup>m</sup>. The train error decreases slowly; the generalization error increases is small, i.e. <sup>R</sup>(θ(t)) <sup>−</sup> <sup>R</sup>b <sup>n</sup>(θ(t)) = o(1); the test error can evolve non-monotonically, but remains approximately constant. Second-layer weights become large <sup>1</sup> ≪ ∥<sup>W</sup>2nd∥<sup>1</sup> ≪ √ m.

(iii) *Overfitting and feature unlearning.* t ≳ m. Train error and test error diverge significantly, i.e. <sup>R</sup>(θ(t)) <sup>−</sup> <sup>R</sup>b <sup>n</sup>(θ(t)) becomes of order one. At the end of this regime, the train error converges to 0, i.e. the neural network interpolates the noisy data. The test error instead grows, and its limit value is the one of a (data independent) kernel method: in other words, the model unlearns the low-dimensional structure. Finally, the second weights grow to ∥<sup>W</sup>2nd∥<sup>1</sup> ≍ √ m, which indeed is the scale required for interpolation.

In this section we outline our results based on 'dynamical mean field theory' (DMFT). The next section will present rigorous results that are proven independently.

#### 2.1 Technique

Our DMFT analysis is based on the following two steps:

*Step 1:* We leverage techniques from theoretical physics to derive an approximate asymptotic characterization of the gradient flow dynamics [\(1.2\)](#page-1-2) in the limit n, d → ∞, with n/d → <sup>α</sup>. This characterization consists of a set of integral-differential equations for the following asymptotic quantities (here p-lim denotes limit in probability, and we use the superscripts n to emphasize the dependence of the right-hand side on n, d)

$$\begin{aligned} C_{ij}(t_1, t_2) &:= \text{p-lim}_{n,d \rightarrow \infty} \langle \mathbf{w}_i^n(t_1), \mathbf{w}_j^n(t_2) \rangle, \\ \mathbf{v}_i(t) &:= \text{p-lim}_{n,d \rightarrow \infty} \mathbf{U}^T \mathbf{w}_i^n(t), \quad a_i(t) := \text{p-lim}_{n,d \rightarrow \infty} a_i^n(t). \end{aligned} \quad (2.2)$$

A rigorous derivation of the DMFT in a setting that includes two-layer networks is given in [\[13\]](#page-10-8).

However, the asymptotically exact DMFT characterization of [\[13\]](#page-10-8) is rather complex to integrate numerically or to study analytically. In order to circumvent this problem, we use a DMFT that is is asymptotically exact for a well-defined Gaussian version of the original model. Namely, we observe that the empirical risk of Eq. [\(1.2\)](#page-1-2) takes the form

$$\widehat{\mathcal{R}}_n(\boldsymbol{\theta}) = \frac{1}{2n} \|\mathbf{F}(\boldsymbol{\theta})\|^2, \quad (2.3)$$

where F : (S d−1 ) <sup>m</sup>×<sup>R</sup> <sup>m</sup> → <sup>R</sup> <sup>n</sup> is s stochastic process with i.i.d. components <sup>F</sup>i(θ) = <sup>y</sup>i−f(x<sup>i</sup> ; θ). We replace these by Gaussian processes with matching mean and covariance, and study the DMFT n (θ).

for gradient flow with respect to the associated risk <sup>R</sup>b<sup>g</sup> The Gaussian approximation comes with an error which we show analytically is vanishing on time scales of order one ( indeed on these time scales we correctly recover the mean field theory of [\[38,](#page-12-6) [14\]](#page-10-9)) and we demonstrate empirically to be small on larger time scales ( see for instance example Fig. [4.](#page-7-1)) The curves in Fig. [1](#page-1-1) were obtained by solving numerically the DMFT equations, see Appendix [C](#page-23-0) for details.

*Step 2:* We study this DMFT, with special attention to the large network limit <sup>m</sup> → ∞, and large sample size <sup>α</sup> → ∞, with <sup>α</sup> <sup>=</sup> α/m fixed, for a generic single index model (<sup>k</sup> = 1). We obtain a separation of time scales in the dynamics, corresponding to distinct learning regimes.

![](_page_4_Figure_0.jpeg)

Figure 2: Evolution of second-layer weights (left) and train error (right) when fitting pure noise data. Here we use mean field initialization, h(z) = (9/10)z + (1/6)z 3 , α = 0.4 and τ = 0.6. Symbols: SGD results on actual 2-layer networks with d = 200, n = αmd (averaged over 10 simulations). Continuous viridis lines: Numerical solution of the DMFT equations. Note that the second layer weights are given in terms of a scalar quantity as the result of the statistically symmetric initialization.

The analysis of the DMFT equations in the double limit m, t → ∞ is an example of singular perturbation theory [\[9,](#page-10-10) [29\]](#page-11-6). Making this type of analysis rigorous is notoriously challenging and we proceed by a combination of numerical solutions and analytical derivations.

In the following, we will first consider the simplest possible setting, pure noise data, and subsequently consider the single-index model. The structure of the activation function and target nonlinearity will be encoded in the functions

$$h(q) := \mathbb{E}\{\sigma(G_1)\sigma(G_q)\}, \quad \hat{\varphi}(q) := \mathbb{E}\{\varphi(G_1)\sigma(G_q)\} ,$$

<sup>h</sup>(q) := <sup>E</sup>{σ(G1)σ(Gq)}, <sup>φ</sup>b(q) := <sup>E</sup>{φ(G1)σ(Gq)} , where <sup>G</sup>1, G<sup>q</sup> are standard jointly Gaussian with <sup>E</sup>{<sup>G</sup>1Gq} <sup>=</sup> <sup>q</sup>. The relation between σ, φ P and h, <sup>φ</sup>b is conveniently expressed in terms of the expansions in Hermite polynomials <sup>σ</sup>(x) = k≥0 <sup>s</sup>kHek(x), <sup>φ</sup>(x) = P k≥0 fkHek(x), which corresponds to the analytic expansion h(q) = P k≥0 s 2 k q k k≥0 skfkq k .

, <sup>φ</sup>b(q) = P As mentioned above, we assume throughout n, d → ∞, with n/d → <sup>α</sup> ∈ (0, ∞), with the limit m, <sup>α</sup> → ∞ taken afterwards. To further simplify our analysis, we assume a symmetric initialization whereby <sup>a</sup>i(0) = <sup>a</sup><sup>0</sup> is independent of <sup>i</sup> ≤ <sup>m</sup> and (wi(0) : <sup>i</sup> ≤ <sup>m</sup>) ∼iid Unif(<sup>S</sup> d−1 ). Throughout, we use 'with high probability' for 'with probability converging to one as n, d → ∞.'

In Section [3](#page-7-0) we present rigorous results that do not require either of these simplifying assumptions.

#### 2.2 Training on pure noise

We begin by the case in which the data is pure noise: <sup>y</sup><sup>i</sup> <sup>=</sup> <sup>ε</sup><sup>i</sup> ∼ <sup>N</sup>(0, τ <sup>2</sup> ). A by-now-classic experiment [\[60\]](#page-13-5) showed that deep learning models have sufficient capacity to achieve vanishing training error even when actual labels are replaced by random ones: they 'interpolate pure noise.'

The ability of a model F<sup>Θ</sup> = (f(· ; <sup>θ</sup>) : <sup>θ</sup> ∈ Θ) to interpolate pure noise is intimately connected to its Gaussian complexity G(FΘ; <sup>n</sup>) := <sup>E</sup> supθ∈Θ⟨g, f(X; <sup>θ</sup>)⟩/n [\[53\]](#page-12-7) (where <sup>g</sup> ∼ <sup>N</sup>(0, <sup>I</sup>n) is independent of f(X, ; θ) = (f(x<sup>i</sup> ; <sup>θ</sup>) : <sup>i</sup> ≤ <sup>n</sup>). Indeed, interpolation is impossible unless G(FΘ; <sup>n</sup>) ≥ <sup>τ</sup> . Viceversa, G(FΘ; <sup>n</sup>) ≪ <sup>τ</sup> ensures good generalization.

By a theorem of [\[7\]](#page-10-11) for the network [\(1.1\)](#page-1-0), G(FΘ; <sup>n</sup>) ≤ <sup>L</sup>σ∥a/m∥<sup>1</sup> p d/n (with L<sup>σ</sup> depending uniquely on σ). This means that, in order to interpolate noise, the average magnitude of second layer weights must be ∥a/m∥<sup>1</sup> ≥ <sup>L</sup> −1 σ τ p n/d = (L −1 <sup>σ</sup> α 1/2 )τ √ m.

However, complexity bounds do not have implications on the convergence of GF to an interpolator.

Figure [2](#page-4-0) compares the DMFT predictions to simulations using SGD to train an actual two layer networks. In this figure we initialize a(0) = 1, and let a(t) evolve with GF alongside the first layer weigths. We observe that the theory describes well the empirical results, despite the Gaussian

![](_page_5_Figure_0.jpeg)

Figure 3: Train/test error (right) when fitting data from a single index model. We set h(z) = <sup>φ</sup>b(z) = (9/10)<sup>z</sup> <sup>+</sup> <sup>z</sup> <sup>2</sup>/2, τ = 0.3 and α = 0.3. Lines correspond to predictions from the DMFT (continuous: train error; dashed: test error). Black continuous line is the <sup>m</sup> → ∞ value. Right: Same data plotted versus t.

approximation in our DMFT and the difference between SGD and GF. We also observe that secondlayer weights remain roughly constant until a large time t#(m), which appears to increase with m. Roughly at the same time, train error starts to decrease and converges to zero.

In Section [G.1](#page-48-0) of the appendix, we will make precise the above picture of the evolution of a(t). Here, we consider a simplified setting in which a(t) = γ √ m with γ independent of m, not evolving with training. Note that G(FΘ; <sup>n</sup>) ≍ γ/√ α and hence such a network can interpolate pure noise if γ is larger than threshold depending on <sup>α</sup>. Our DMFT predicts a sharp phase transition. For <sup>α</sup> ∈ (0, 1), GF converges to vanishing train error with high probability if γ > γGF(α, m)τ , and converges to a strictly positive training error if γ < γGF(α, m)τ . The threshold γGF(α, m) converges to a limit γ ∗ GF(α) ∈ (0, 1) as <sup>m</sup> → ∞.

A rephrasing of the same phenomenon states that 
$$\lim_{n,d \rightarrow \infty} \widehat{\mathcal{R}}_n^g(\boldsymbol{\theta}(t)) = e_{\text{tr}}(t; m, \gamma)$$
, and
$$\lim_{t \rightarrow \infty} \lim_{m \rightarrow \infty} e_{\text{tr}}(t; m, \gamma_0) = \begin{cases} e_*(\gamma) > 0 & \text{for } \gamma < \gamma_{\text{GF}}^*(\alpha)\tau, \\ 0 & \text{for } \gamma \geq \gamma_{\text{GF}}^*(\alpha)\tau. \end{cases} \quad (2.4)$$

n (θ(t)) = etr(t; m, γ), and

Informally γ ∗ GF(α) is the minimum complexity γ for a very large network to interpolate noise via gradient flow. The functions γ ∗ GF(α), e∗(γ) will play an important role below.

We will next consider training on data from a single-index model. The initial scale of secondlayer weights ∥a(0)/m∥<sup>1</sup> plays a crucial role and we will separately analyze lazy and mean field initializations.

#### 2.3 Training on data with latent structure: lazy initialization

We initialize a(0) = γ<sup>0</sup> √ m, and let a(t) evolve according to GF alongside first-layer weights. DMFT predicts the emergence of three dynamical regimes for large <sup>m</sup> and large <sup>α</sup> (with n/d → <sup>α</sup>). For an illustration, we refer to Fig. [3.](#page-5-0)

*First dynamical regime:* t = O(1/m)*.* Second layer weights do not change significantly γ(t) = <sup>γ</sup><sup>0</sup> <sup>+</sup> <sup>o</sup>m(1), while first layer-weights move by ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1/ √ m). Because the weights <sup>a</sup>i(t) are of order √ m, even an O(1/ √ m) change in the w<sup>i</sup> leads to a significant decrease in test error and train error.

Train and test error are close to each other. Namely, the following limits are well defined

$$\lim_{n, d \rightarrow \infty} \widehat{\mathcal{R}}_n^g(\boldsymbol{\theta}(t)) = e_{\text{tr}}(t; \varphi, \gamma_0, m, \alpha), \quad \lim_{n, d \rightarrow \infty} \mathcal{R}^g(\boldsymbol{\theta}(t)) = e_{\text{ts}}(t; \varphi, \gamma_0, m, \alpha). \quad (2.5)$$
a lim  $\rightarrow \infty$   $e_{\text{tr}}(\hat{t}; m; \varphi, \gamma_0, m, \alpha) = \lim_{m \rightarrow \infty} e_{\text{ts}}(\hat{t}; m; \varphi, \gamma_0, m, \alpha) =: e^{\text{lzl}}(\hat{t}; \varphi, \gamma_0, \alpha)$ .

with limm→∞ etr(t/mˆ ; φ, γ0, m, α) = limm→∞ ets(t/mˆ ; φ, γ0, m, α) =: e lz1(tˆ; φ, γ0, α).

with 
$$\lim_{m \rightarrow \infty} e_{\text{tr}}(\hat{t}; m; \varphi, \gamma_0, m, \alpha) = \lim_{m \rightarrow \infty} e_{\text{ts}}(\hat{t}; m; \varphi, \gamma_0, m, \alpha) =: e^{i t}(\hat{t}; \varphi, \gamma_0, \alpha)$$
.

For large scaled time tˆ, the error e lz1(tˆ; φ, γ0, α) converges to the error of the best linear approximation to f∗. This dynamical regime follows the qualitative predictions of NTK theory, and is essentially linear in the weights w<sup>i</sup> , but the time is too short for the model to overfit the data.

*Second dynamical regime:* t = Θ(1)*.* Second layer weights do not change significantly: γ(t) = <sup>γ</sup><sup>0</sup> <sup>+</sup> <sup>o</sup>m(1), while first layer weights change significantly ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1). However they change orthogonally to the latent subspace U and hence the test error does not change: no actual learning takes place in this regime, but the model starts to overfit the data.

More formally, train and test error have well defined limits as the network width diverges:

$$e_{\text{tr}}^{l2}(t; \varphi, \gamma_0, \alpha) := \lim_{m \rightarrow \infty} e_{\text{tr}}(t; \varphi, \gamma_0, m, \alpha), \quad e_{\text{ts}}^{l2}(t; \varphi, \gamma_0, \alpha) := \lim_{m \rightarrow \infty} e_{\text{ts}}(t; \varphi, \gamma_0, m, \alpha). \quad (2.6)$$

However, the scaling function e ts (t; φ, γ0, α) for the test error is constant in time and equal to the value achieved at the end of the first dynamical regime. Namely

$$e_{\text{ts}}^{l2}(t; \varphi, \gamma_0, \alpha) = \lim_{t \rightarrow \infty} e^{l_1 t} (\hat{t}; \varphi, \gamma_0, \alpha) = \frac{1}{2} \left( \tau^2 + \|\varphi\|^2 - \frac{\|\nabla \hat{\varphi}(\mathbf{0})\|^2}{h'(0)} + \gamma_0^2(h(1) - h'(0)) \right). \quad (2.7)$$

Since the wi's move orthogonally to the latent space, their dynamics is equivalent (for large m) to the one in the pure noise setting, modulo a redefinition of h. The right plot in Fig. [3](#page-5-0) illustrates this.

*Third dynamical regime:* t = Θ(m)*.* The qualitative properties of this regime depend whether or not γ<sup>0</sup> is larger than an interpolation threshold γ ∗ GF(α, φ, τ ), which generalizes the threshold γ ∗ GF(α) = γ ∗ GF(α, 0, 1) introduced in the pure noise case. Because the dynamics of weights w<sup>i</sup> in the subspace orthogonal to U is equivalent to dynamics in pure noise, we expect the interpolation threshold γ ∗ GF(α, φ, τ ) to be given in terms of pure noise threshold γ ∗ GF(α) as follows:

$$\gamma_{\text{GF}}^*(\alpha, \varphi, \tau) = \left( \tau^2 + \|\varphi\|^2 - \frac{\|\nabla \hat{\varphi}(\mathbf{0})\|^2}{h'(0)} \right)^{1/2} \gamma_{\text{GF}}^*(\alpha). \quad (2.8)$$

For γ<sup>0</sup> > γ<sup>∗</sup> GF(α, φ, τ ), interpolation is achieved during the second dynamical regime, no further evolution takes place.

For γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ), a non-trivial evolution takes place for t = Θ(m). Introducing the rescaled time <sup>z</sup> ∈ (0, ∞), we obtain, as <sup>m</sup> → ∞,

$$\gamma(mz) = \gamma^{l3}(z) + o_m(1), \quad e_{\text{tr}}(mz) = e_{\text{tr}}^{l3}(z) + o_m(1), \quad e_{\text{ts}}(mz) = e_{\text{ts}}^{l3}(z) + o_m(1). \quad (2.9)$$

Further, for large values of the rescaled time <sup>z</sup> → ∞, <sup>γ</sup> lz3(z) grows to γ ∗ GF(α, φ, τ ) ≈ <sup>γ</sup> ∗ GF(α, φ, τ ), while e lz3 tr (z) decreases to 0. In other words, interpolation is achieved on this third regime.

Further the test error e lz3 ts (z) increases from e lz2 ts (t; φ, γ0, α) to e lz2 ts (t; φ, γ<sup>∗</sup> GF, α), with γ ∗ GF = γ ∗ GF(α, φ, τ ) whereby e lz2 ts (· · ·) is given by Eq. [\(2.7\)](#page-6-0).

#### 2.4 Training on data with latent structure: mean field initialization

We initialize a(0) = a0, independent of m and let second layer weights evolve. Note that at initialization the network's Rademacher complexity is small, namely of order a<sup>0</sup> p d/n = a0/ √ αm. Our DMFT analyisis predicts two dynamical regimes for large m. We will refer to them as 'first' and 'third regime' for consistency with other settings ( see Sec[.G.2](#page-54-0) of the appendix). For an illustration, we refer to Figs. [4](#page-7-1) and [5.](#page-8-0)

*First dynamical regime:* t = O(1)*.* Both first and second layer weights change by order one: <sup>a</sup>(t) = <sup>a</sup><sup>0</sup> + Θ(1) and ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1). and as a consequence test and train error decrease significantly. In this regime, the two errors remain close to each other and their evolution is well captured by the mean field theory of [\[38,](#page-12-6) [14\]](#page-10-9), as specialized to the case of spherically invariant distributions [\[10,](#page-10-12) [2\]](#page-10-13).

Namely, limm→∞ a(t) = a mf1(t), limm→∞ v(t) = v mf1(t), and DMFT reduces to a system of k + 1 ordinary differential equations for the k + 1 scalar variables (a mf1(t), v mf1(t))

$$\begin{aligned} \partial_t \mathbf{v}^{\text{mfl}}(t) &= \alpha a^{\text{mfl}}(t) \mathbf{Q}_{\mathbf{v}^{\text{mfl}}(t)} \left( \nabla \hat{\varphi}(\mathbf{v}^{\text{mfl}}(t)) - a^{\text{mfl}}(t) h'(\|\mathbf{v}^{\text{mfl}}(t)\|^2) \mathbf{v}^{\text{mfl}}(t) \right), \\ \partial_t a^{\text{mfl}}(t) &= \alpha \hat{\varphi}(\mathbf{v}^{\text{mfl}}(t)) - \alpha a^{\text{mfl}}(t) h(\|\mathbf{v}^{\text{mfl}}(t)\|^2), \end{aligned} \quad (2.10)$$

where Q<sup>v</sup> := <sup>I</sup><sup>k</sup> − vv<sup>T</sup>. As mentioned above, train and test error coincide in the large width limit

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t) = \lim_{m \rightarrow \infty} e_{\text{ts}}(t) = e^{\text{mfl}}(t).$$

![](_page_7_Figure_0.jpeg)

Figure 4: Training dynamics under a single-index model. We set <sup>h</sup>(q) = <sup>φ</sup>b(q) = (9/10)<sup>q</sup> <sup>+</sup> <sup>q</sup> <sup>3</sup>/6, τ = 0.3 and α = 0.3, under mean field initialization. Left: second-layer weights. Right: train and test error. Symbols are empirical results for SGD with actual two-layer neural networks with d = 200, n = αmd (averaged over 10 simulations). Lines correspond to predictions from the DMFT (on the right, continuous: train error; dashed: test error).

An explicit formula for e mf1(t) is given in Appendix [G.2.1.](#page-54-1) In the case <sup>k</sup> = 1 and <sup>φ</sup>b(z) = <sup>h</sup>(z), we have that a mf1 = 1, v mf1 = 1 is a fixed point of Eq. [\(2.10\)](#page-6-1), and indeed the only fixed point with v mf1 > 0. If h ′ (0) > 0, then, we have (a mf1(t), vmf1(t)) → (1, 1) as <sup>t</sup> → ∞, and therefore test and train error converge to the Bayes error e mf1(t) → <sup>τ</sup> <sup>2</sup>/2. This is significantly smaller than the test error achieved with lazy initialization. The separation between lazy and mean-field initialization is expected because feature learning takes place in the mean field regime.

*Third dynamical regime:* t = Ω(m)*.* Computing the local stability of DMFT solutions around the mean field asymptotics (see Appendix [G.2.2\)](#page-56-0) suggests that the latter breaks down for t = Θ(m). For <sup>t</sup> <sup>≳</sup> <sup>m</sup>, we observe that the second layer weights grow to achieve <sup>a</sup>(t) ≍ √ m, the projection onto the latent space decreases to <sup>v</sup>(t) ≍ <sup>1</sup>/ √ m, and train and test error diverge, eventually achieving <sup>e</sup>tr(t) ≈ <sup>0</sup> and test error significantly larger than the Bayes error achieved earlier. We refer to this phenomenon as 'feature unlearning.'

Denoting by t0(m; c) the time at which a(t) = c √ m (for c a small constant), we expect the existence of a window size w(m) such that

$$\lim_{m \rightarrow \infty} \frac{a(t_0(m; c) + z w(m))}{\sqrt{m}} = \gamma^{\text{mf3}}(z), \quad \lim_{m \rightarrow \infty} e_{\text{tr/ts}}(t_0(m; c) + z w(m)) = e_{\text{tr/ts}}^{\text{mf3}}(z), \quad (2.11)$$

where γ mf3(z), e mf3 tr (z), e mf3 ts (z) are scaling functions describing the dynamics on this timescale. We expect t0(m; c) = t∗(c)m + o(m), and w(m) ≲ t0(m; c), but our numerical solutions are not sufficient to determine the precise scaling. On the other hand, it appears that at large times, the complexity converges close the interpolation threshold:

$$\lim_{z \rightarrow \infty} \gamma^{\text{mf3}}(z) = \bar{\gamma}_{\text{GF}}^*(\alpha, \varphi, \tau) \approx \gamma_{\text{GF}}^*(\alpha, \varphi, \tau). \quad (2.12)$$

Finally, the evolution of train and test error for <sup>a</sup>(t) ≍ √ m appears to match the behavior at fixed second-layer weights. Namely, we define two functions

$$\varepsilon_{\text{tr}/\text{ts}}^{\text{mf}}(\gamma) := \lim_{m \rightarrow \infty} e_{\text{tr}/\text{ts}}(t_0(m; \gamma), m). \quad (2.13)$$

We observe that the limit curves (γ, εmf tr(γ)), (γ, εmf ts(γ)), match closely asymptotic train and test error obtained by fixing a(t) = γ √ m, and not letting second-layer weight evolve. This confirms the hypothesis that γ(t) is a slow variable, while others converge as if γ was fixed.

## 3 Lower bounding the overfitting timescale

In this section we rigorously establish two results that confirm elements of the scenario outlined in the previous sections. We emphasize that the result presented here are non-asymptotic, i.e. hold at finite

![](_page_8_Figure_0.jpeg)

Figure 5: Left: second layer weights on the scale √ m as a function of t/m. Curves appear to collapse on a master curve. The red arrow denotes γ ∗ GF and the curves appear to converge to that limit. Center: the projection of the first layer weights on the latent space in the single index model as a function of time on timescales of order m. Right: difference between test and train error as a function of the second layer weights on the scale √ m. The finite m curve are approaching a scaling curve which coincides with the one obtained by evaluating the same quantity but with a lazy initialization and fixed second layer weights.

n, m, d modulo unspecified absolute constants. Further, we do not assume a symmetric initialization of the weights. Throughout this section setting, it is more convenient to rescale time defining tˆ= tα. Hence. instead of the flow [\(1.2\)](#page-1-2), we study

<sup>θ</sup>˙(tˆ) = <sup>−</sup><sup>m</sup><sup>P</sup> <sup>θ</sup>∇Rb For α = Θ(1) the parametrizations t and tˆare equivalent.

$$\dot{\theta}(\hat{t}) = -mP_{\theta}\nabla\hat{\mathcal{R}}_n(\theta(\hat{t})). \quad (3.1)$$

The first result of this section implies that (under mean field initialization) overfitting cannot take place on times of order one.

Theorem 1. *Under the GF dynamics* [\(1.2\)](#page-1-2)*, and the data distribution in the introduction (with* k *arbitrary), further assume* ∥σ∥Lip, ∥σ∥<sup>∞</sup> ≤ <sup>L</sup>*,* |φ(0)|, ∥φ∥Lip ≤ <sup>L</sup>*,* ∥a(0)∥<sup>∞</sup> ≤ <sup>a</sup>0*, for some* <sup>a</sup><sup>0</sup> ≥ <sup>1</sup> *and that the* <sup>w</sup>i(0)*,* <sup>i</sup> ≤ <sup>m</sup> *are independent of the data* {(y<sup>i</sup> , <sup>x</sup>i) : <sup>i</sup> ≤ <sup>n</sup>}*. Finally assume* <sup>n</sup> ≥ <sup>d</sup>∨ <sup>m</sup>*. Then, there exist universal constants* <sup>C</sup>0, C1*, and the following holds for all* <sup>t</sup><sup>ˆ</sup>≥ <sup>0</sup>*,*

$$\|\mathbf{a}(\hat{t})\|_\infty \leq a_0 + a_1 \hat{t}, \quad a_1 := C_0 L(\tau + \sqrt{k} + a_0 L), \quad (3.2)$$

$$|\mathcal{R}(\mathbf{a}(t), \mathbf{W}(t)) - \widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t))| \leq C_1(L^2(a_0 + a_1 t)^2 + \tau^2) \cdot \sqrt{\frac{d}{n}}. \quad (3.3)$$

Under mean field initialization, a<sup>0</sup> is a fixed constant and hence a<sup>1</sup> is also bounded, whence the generalization error in Eq. [\(3.3\)](#page-8-1) is small as long as tˆ = o((n/d) 1/4 ) (equivalently, for α fixed, tˆ= o(m<sup>1</sup>/<sup>4</sup> )).

By itself, this result implies a separation of timescales between learning and overfitting, thus confirming the picture developed within DMFT, but falls short of characterizing the overfitting timescale.

The second result implies that, up to time-scale of order one, the dynamics is closely tracked by the mean field equations [\(2.10\)](#page-6-1). Since the ai(0) at initialization are not necessarily all equal, these are generalized as

$$\begin{aligned} \partial_t \mathbf{v}_i^{\text{mfl}}(\hat{t}) &= a_i^{\text{mfl}}(\hat{t}) \mathbf{Q}_{\mathbf{v}_i^{\text{mfl}}(\hat{t})} \left( \nabla \hat{\varphi}(\mathbf{v}_i^{\text{mfl}}(\hat{t})) - \frac{1}{m} \sum_{j=1}^m a_j^{\text{mfl}}(\hat{t}) h'(\langle \mathbf{v}_i^{\text{mfl}}(\hat{t}), \mathbf{v}_j^{\text{mfl}}(\hat{t}) \rangle) \mathbf{v}_j^{\text{mfl}}(\hat{t}) \right), \\ \partial_t a_i^{\text{mfl}}(\hat{t}) &= \hat{\varphi}(\mathbf{v}_i^{\text{mfl}}(\hat{t})) - \frac{1}{m} \sum_{j=1}^m a_j^{\text{mfl}}(\hat{t}) h(\langle \mathbf{v}_i^{\text{mfl}}(\hat{t}), \mathbf{v}_j^{\text{mfl}}(\hat{t}) \rangle). \end{aligned} \quad (3.4)$$

The mean field prediction for test error is the same as for training error and given by

$$e_{\text{ts}}(\hat{t}) = \frac{1}{2}\|\varphi\|_{L^2}^2 - \frac{1}{m} \sum_{j=1}^m a_j^{\text{mfl}}(\hat{t}) \hat{\varphi}(\mathbf{v}_j^{\text{mfl}}(\hat{t})) + \frac{1}{2m^2} \sum_{j=1}^m a_i^{\text{mfl}}(\hat{t}) a_j^{\text{mfl}}(\hat{t}) h(\langle \mathbf{v}_i^{\text{mfl}}(\hat{t}), \mathbf{v}_j^{\text{mfl}}(\hat{t}) \rangle)$$

Theorem 2. *Under the the GF dynamics* [\(1.2\)](#page-1-2)*, and the data distribution in the introduction (with* <sup>k</sup> *arbitrary), further assume that* ∥φ∥∞*,* ∥<sup>φ</sup> ′∥∞*,* ∥<sup>φ</sup> ′∥Lip ≤ <sup>L</sup>*,* ∥σ∥∞*,* ∥<sup>σ</sup> ′∥∞*,* ∥<sup>σ</sup> ′∥Lip ≤ <sup>L</sup>*. Further*

*assume* |<sup>a</sup>i(0)| ≤ <sup>L</sup> *for all* <sup>i</sup> ≤ <sup>m</sup>*,* (wi(0))i≤<sup>m</sup> ∼iid Unif(<sup>S</sup> d−1 )*. Then for any* δ > 0 *there exist constants* c<sup>0</sup> c1*,* C *depending on* L, τ, δ, k *such that, letting* Tlb = c0(log m) <sup>1</sup>/<sup>3</sup> ∧ (log n/d) 1/3 *, the following happens with probability at least* <sup>1</sup> − 2 exp(−<sup>c</sup>1d)*,*

$$\sup_{t \leq T_{\text{b}}} \frac{1}{m} \sum_{i=1}^m \left( |a_i(\hat{t}) - a_i^{\text{mfl}}(\hat{t})| + \|\mathbf{v}_i(\hat{t}) - \mathbf{v}_i^{\text{mfl}}(\hat{t})\| \right) \leq C \left( \frac{1}{m} \vee \frac{1}{d} \vee \frac{d}{n} \right)^{1/2-\delta}, \quad (3.5)$$

$$\sup_{t \leq T_{\text{b}}} \left| \mathcal{R}(\mathbf{a}(\hat{t}), \mathbf{W}(\hat{t})) - e_{\text{ts}}(\hat{t}) \right| \leq C \left( \frac{1}{m} \vee \frac{1}{d} \vee \frac{d}{n} \right)^{1/2-\delta}. \quad (3.6)$$

Remark 3.1. While the analysis in the previous section requires <sup>m</sup> → ∞ *after* n, d → ∞, neither Theorem 3.1 nor Theorem 3.2 make the assumption. In particular, Eq. [\(3.3\)](#page-8-1) implies that the generalization error is small for tˆ= o((n/d) 1/4 ) *irrespective of* m.

Similarly, Eqs. [\(3.5\)](#page-9-1), [\(3.6\)](#page-9-2) imply that the mean field theory of [\[38,](#page-12-6) [14,](#page-10-9) [45\]](#page-12-8) captures well the evolution of the system for times t = o((log m) <sup>1</sup>/<sup>3</sup> ∧ (log n/d) 1/3 ).

## 4 Discussion

We conclude by highlighting a few qualitative conclusions of our work, and how they address questions raised in Section [1.](#page-0-0) In the following remarks, we consider α = n/md as constant.

Interpolation mechanism. In the current setting, the neural model complexity is proportional to ∥a(t)∥1/ √ m = γ(t)+on(1). We observe two alternative scenarios. If the complexity at initialization is large enough γ<sup>0</sup> > γ<sup>∗</sup> GF(α)τ , then the gradient flow rapidly converges to a near interpolator without significant change in γ(t). If instead, γ<sup>0</sup> < γ<sup>∗</sup> GF(α)τ , then γ(t) grows to reach the interpolation threshold at which point the training error converges to 0.

Adiabatic evolution of model complexity. In the latter case, the complexity γ(t) evolves on a slower time scale than other degrees of freedom. The dynamics on shorter timescales is well approximated by the one at fixed γ (given by the current value γ(t)). The generalization error becomes of order one only when γ(t) is of order one.

Decoupling of learning and overfitting. When γ<sup>0</sup> = om(1), the fact that γ(t) acts as a slow variable implies a large-m decoupling between learning (which takes place on faster timescales, as long as γ(t) = om(1)), and overfitting (which takes place on slower timescales, when γ(t) = Ωm(1)). This has several implications for the questions outlined in the introduction.

Q3: Lazy initialization <sup>a</sup>(0) ≍ √ m leads to poor generalization because the feature-learning phase

is skipped either partially or altogether. Q2: Training until interpolation is generally suboptimal. Q4: The optimal tradeoff is obtained at the end of the first phase.

Q5, Q6: Further, at fixed overerparametrization n/md = α, overfitting starts later for larger models.

Overfitting and feature unlearning. The above description points at a non-monotonicity of the model quality, which improves on short time scales, and deteriorates at larger time scales. Reciprocally, early stopping acts as a regularization. While this phenomenon is well understood for linear models [\[24,](#page-11-7) [57\]](#page-13-3), our analysis provides an analogous (quantitative) scenario for training neural network models. In particular, it clarifies the underlying mechanism: in the same dynamical regime in which network complexity grows (γ(t) becomes of order one), and training error becomes negligible, the low-dimensional latent features are 'unlearned' (v(t) becomes of order 1/ √ m). We expect that these findings also allow to understand the beneficial effect of regularization on the second layer.

## Acknowledgments

This work was supported by the NSF through award DMS-2031883, the Simons Foundation through Award 814639 for the Collaboration on the Theoretical Foundations of Deep Learning, and the ONR grant N00014-18-1-2729. This work was supported by the French government under the France 2030 program (PhOM - Graduate School of Physics) with reference ANR-11-IDEX-0003.

## References


[1] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In *Conference on Learning Theory*, pages 4782–4887. PMLR, 2022. [2] Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro. From highdimensional and mean-field dynamics to dimensionless odes: A unifying approach to sgd in two-layers networks. In *The Thirty Sixth Annual Conference on Learning Theory*, pages 1199–1227. PMLR, 2023. [3] Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. *Advances in Neural Information Processing Systems*, 32, 2019. [4] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. *Journal of Machine Learning Research*, 22(106):1–51, 2021. [5] Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. Highdimensional asymptotics of feature learning: How one gradient step improves the representation. *Advances in Neural Information Processing Systems*, 35:37932–37946, 2022. [6] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: SGD learns parities near the computational limit. *Advances in Neural Information Processing Systems*, 35:21750–21764, 2022. [7] Peter Bartlett. For valid generalization the size of the weights is more important than the size of the network. *Advances in neural information processing systems*, 9, 1996. [8] Gérard Ben Arous, Amir Dembo, and Alice Guionnet. Cugliandolo-kurchan equations for dynamics of spin-glasses. *Probability theory and related fields*, 136(4):619–660, 2006. [9] Nils Berglund. Perturbation theory of dynamical systems. *arXiv preprint math/0111178*, 2001. [10] Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. *Foundations of Computational Mathematics*, pages 1–84, 2024. [11] Christopher M Bishop. Regularization and complexity control in feed-forward networks. In *Proceedings International Conference on Artificial Neural Networks ICANN'95*, pages 141–148, 1995. [12] Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. *Advances in Neural Information Processing Systems*, 35:32240–32256, 2022. [13] Michael Celentano, Chen Cheng, and Andrea Montanari. The high-dimensional asymptotics of first order methods with random data. *arXiv:2112.07572*, 2021. [14] Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for overparameterized models using optimal transport. *Advances in neural information processing systems*, 31, 2018. [15] Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In *Conference on learning theory*, pages 1305–1338. PMLR, 2020. [16] Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. *Advances in neural information processing systems*, 32, 2019. [17] Andrea Crisanti, Heinz Horner, and H J Sommers. The spherical p-spin interaction spin-glass model: the dynamics. *Zeitschrift für Physik B Condensed Matter*, 92:257–271, 1993. [18] Leticia F Cugliandolo. Recent applications of dynamical mean-field methods. *Annual Review of Condensed Matter Physics*, 15, 2023.

[19] Leticia F Cugliandolo and David S Dean. Full dynamical solution for a spherical spin-glass model. *Journal of Physics A: Mathematical and General*, 28(15):4213, 1995. [20] Leticia F Cugliandolo and Jorge Kurchan. Analytical solution of the off-equilibrium dynamics of a long-range spin-glass model. *Physical Review Letters*, 71(1):173, 1993. [21] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In *Conference on Learning Theory*, pages 5413–5452. PMLR, 2022. [22] Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In *International conference on machine learning*, pages 1675–1685. PMLR, 2019. [23] Giampaolo Folena, Silvio Franz, and Federico Ricci-Tersenghi. Rethinking mean-field glassy dynamics and its relation with the energy landscape: The surprising case of the spherical mixed p-spin model. *Physical Review X*, 10(3):031045, 2020. [24] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Additive logistic regression: a statistical view of boosting (with discussion and a rejoinder by the authors). *The annals of statistics*, 28(2):337–407, 2000. [25] Yan V Fyodorov. A spin glass model for reconstructing nonlinearly encrypted signals corrupted by noise. *Journal of Statistical Physics*, 175:789–818, 2019. [26] Yan V Fyodorov and Rashel Tublin. Optimization landscape in the simplest constrained random least-square problem. *Journal of Physics A: Mathematical and Theoretical*, 55(24):244008, 2022. [27] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. *The Annals of Statistics*, 49(2), 2021. [28] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, pages 249–256. JMLR Workshop and Conference Proceedings, 2010. [29] Mark Holmes. *Introduction to Perturbation Methods*. Springer, 2013. [30] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. *Advances in neural information processing systems*, 31, 2018. [31] Persia Jana Kamali and Pierfrancesco Urbani. Dynamical mean field theory for models of confluent tissues and beyond. *SciPost Physics*, 15(5):219, 2023. [32] Persia Jana Kamali and Pierfrancesco Urbani. Stochastic gradient descent outperforms gradient descent in recovering a high-dimensional signal in a glassy energy landscape. *arXiv preprint arXiv:2309.04788*, 2023. [33] Jaron Kent-Dobias. On the topology of solutions to random continuous constraint satisfaction problems. *arXiv preprint arXiv:2409.12781*, 2024. [34] Yuanzhi Li, Colin Wei, and Tengyu Ma. Towards explaining the regularization effect of initial large learning rate in training neural networks. *Advances in neural information processing systems*, 32, 2019. [35] Stefano Sarao Mannelli, Florent Krzakala, Pierfrancesco Urbani, and Lenka Zdeborova. Passed & spurious: Descent algorithms and local minima in spiked matrix-tensor models. In *international conference on machine learning*, pages 4333–4342. PMLR, 2019. [36] Andreas Maurer. A vector-contraction inequality for Rademacher complexities. In *Algorithmic Learning Theory: 27th International Conference*, pages 3–17. Springer, 2016.

[37] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration. *Applied and Computational Harmonic Analysis*, 59:3–84, 2022. [38] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. *Proceedings of the National Academy of Sciences*, 115(33):E7665– E7671, 2018. [39] Marc Mézard, Giorgio Parisi, and Miguel Angel Virasoro. *Spin glass theory and beyond*, volume 9. World Scientific, 1987. [40] Francesca Mignacco, Florent Krzakala, Pierfrancesco Urbani, and Lenka Zdeborová. Dynamical mean-field theory for stochastic gradient descent in gaussian mixture classification. *Advances in Neural Information Processing Systems*, 33:9540–9550, 2020. [41] Francesca Mignacco and Pierfrancesco Urbani. The effective noise of stochastic gradient descent. *Journal of Statistical Mechanics: Theory and Experiment*, 2022(8):083405, 2022. [42] Andrea Montanari and Eliran Subag. Solving overparametrized systems of random equations: I. model and algorithms for approximate solutions. *arXiv:2306.13326*, 2023. [43] Andrea Montanari and Eliran Subag. On Smale's 17th problem over the reals. *arXiv:2405.01735*, 2024. [44] Nelson Morgan and Hervé Bourlard. Generalization and parameter estimation in feedforward nets: Some experiments. *Advances in neural information processing systems*, 2, 1989. [45] Grant Rotskoff and Eric Vanden-Eijnden. Trainability and accuracy of artificial neural networks: An interacting particle system approach. *Communications on Pure and Applied Mathematics*, 75(9):1889–1935, 2022. [46] Mark Sellke. The threshold energy of low temperature Langevin dynamics for pure spherical spin glasses. *Communications on Pure and Applied Mathematics*, 77(11):4065–4099, 2024. [47] Shai Shalev-Shwartz and Shai Ben-David. *Understanding machine learning: From theory to algorithms*. Cambridge University Press, 2014. [48] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. *The Journal of Machine Learning Research*, 19(1):2822–2878, 2018. [49] Eliran Subag. Concentration for the zero set of random polynomial systems. *arXiv preprint arXiv:2303.11924*, 2023. [50] Michel Talagrand. *Mean field models for spin glasses: Volume I: Basic examples*, volume 54. Springer Science & Business Media, 2010. [51] Pierfrancesco Urbani. A continuous constraint satisfaction problem for the rigidity transition in confluent tissues. *Journal of Physics A: Mathematical and Theoretical*, 56(11):115003, 2023. [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 30, pages 5998–6008. Curran Associates, Inc., 2017. [53] Roman Vershynin. *High-dimensional probability: An introduction with applications in data science*, volume 47. Cambridge university press, 2018. [54] Nikhil Vyas, Yamini Bansal, and Preetum Nakkiran. Limitations of the ntk for understanding generalization in deep learning. *arXiv preprint arXiv:2206.10012*, 2022. [55] Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. *Advances in neural information processing systems*, 30, 2017.

[56] Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In *Conference on Learning Theory*, pages 3635–3673. PMLR, 2020. [57] Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning. *Constructive Approximation*, 26(2):289–315, 2007. [58] Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding neural networks. *Advances in neural information processing systems*, 32, 2019. [59] Kaichao You, Mingsheng Long, Jianmin Wang, and Michael I Jordan. How does learning rate decay help modern neural networks? *arXiv preprint arXiv:1908.01878*, 2019. [60] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3):107– 115, 2021. [61] Tong Zhang and Bin Yu. Boosting with early stopping: Convergence and consistency. *Annals of Statistics*, pages 1538–1579, 2005. [62] Jean Zinn-Justin. *Quantum field theory and critical phenomena*. Oxford University Press, 2021.
## NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] ,

Justification: We conduct a theoretical analysis that is described by the abstract and that answers the questions detailed in the introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the introduction section we discuss how the contribution compares to previous literature and limitations related to the use of non-rigorous mathematical techniques.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

## Answer: [Yes]

Justification: We conduct a theoretical analysis of training dynamics. The method that we use is non-rigorous but well established in theoretical physics. We show that the method correctly reproduces observations and it is checked against simulations. We prove two theorems that support our analysis.

## Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: We detail the numerical simulations in the appendix

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
  - (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
  - (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
  - (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
  - (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: Our paper is theoretical in nature and simulations are fairly standard and only play a support role.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The theoretical results and figures are detailed with the corresponding settings that we used to produce them.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes] .

Justification: The paper contains all the details about the numerical simulations we used.

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA] .

Justification: There are no extensive or complex experiments we have performed. The paper is theoretical in nature and aims at understanding simple yet paradigmatic models.

#### Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

Answer: [Yes]

Justification: We conform with the NeurIPS Code of Ethics.

## Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications.

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA] .

Justification: We do not use existing datasets or codes.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA] .

Justification: We do not produce any new asset. Our study is purely theoretical in nature.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications. We do not perform experiments with humans.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: We do not conduct experiments with humans.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] .

Justification: We do not use LLM.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

## A Setting

We recall for reference some basic definitions and notations. We consider the 2-layer network defined by

$$f(\mathbf{x}; \mathbf{a}, \mathbf{W}) = \frac{1}{m} \sum_{i=1}^m a_i \sigma(\langle \mathbf{w}_i, \mathbf{x} \rangle). \quad (\text{A.1})$$

Throughout, we assume an offset to be subtracted so that <sup>E</sup>σ(G) = 0, for <sup>G</sup> ∼ <sup>N</sup>(0, 1). The network input x is a d-dimensional real vector and the output is a scalar variable. The parameters of the network are the weights of the first layer collected in the matrix W defined as

$$W = \begin{pmatrix} w_1 \\ w_2 \\ \vdots \\ w_m \end{pmatrix} \in \mathbb{R}^{m \times d}, \quad w_i \in \mathbb{R}^d. \quad (\text{A.2})$$

We will assume that ∥<sup>w</sup>i∥ <sup>2</sup> = 1. The weights of the second layer are instead (a1, . . . , am) and are real, possibly unbounded, variables.

We consider a dataset of n points independent and identically distributed (y<sup>i</sup> , <sup>x</sup>i)i≤<sup>n</sup> where <sup>x</sup><sup>i</sup> ∼ N(0, Id), and the labels y<sup>i</sup> are generated according to the following k-index models:

$$y_i = \varphi(\mathbf{U}^\top \mathbf{x}_i) + \varepsilon_i . \quad (\text{A.3})$$

Therefore, labels depend on the projection of the covariates on a fixed subspace <sup>U</sup> ∈ <sup>R</sup> d×k , with U <sup>T</sup>U = I<sup>k</sup> (there is no loss of generality in assuming U orthogonal). Efficient learning requires to estimate this subspace. Since we consider learning with square loss, we assume

$$\|\varphi\|_2^2 := \mathbb{E}\{\varphi(\mathbf{U}^\top \mathbf{x}_i)^2\} = \mathbb{E}\{\varphi(\mathbf{g})^2\},$$

where <sup>g</sup> ∼ <sup>N</sup>(0, <sup>I</sup>k). We refer to the case <sup>φ</sup> = 0 as the 'pure noise case' or 'pure noise data'.

We now discuss the covariance structure of the network given by Eq. [\(A.1\)](#page-21-0). For two sets of weights (a1,W1) and (a2,W2) we have

$$\mathbb{E}\{f(x; \mathbf{a}_1, \mathbf{W}_1) f(x; \mathbf{a}_2, \mathbf{W}_2)\} = \frac{1}{m^2} \sum_{i,j=1}^m a_{1,i} a_{2,j} h(\langle \mathbf{w}_{1,i}, \mathbf{w}_{2,j} \rangle). \quad (\text{A.4})$$

The average in the rhs of Eq. [\(A.4\)](#page-21-1) is over the data distribution while the function h(q) is defined as

$$h(q) = \mathbb{E}\{\sigma(G_1)\sigma(G_2)\} \quad (\text{A.5})$$

for (G1, G2) centered jointly Gaussian with <sup>E</sup>{G<sup>2</sup> i } = 1, <sup>E</sup>{<sup>G</sup>1G2} <sup>=</sup> <sup>q</sup>.

Furthermore we have that:

$$\mathbb{E}\{f(\mathbf{x}; \mathbf{a}, \mathbf{W}) \varphi(\mathbf{U}^\top \mathbf{x})\} = \frac{1}{m} \sum_{i=1}^m a_i \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_i). \quad (\text{A.6})$$

where <sup>φ</sup>b is given by

<sup>φ</sup>b(v) := <sup>E</sup> for <sup>G</sup> ∼ <sup>N</sup>(0, <sup>I</sup>k) independent of <sup>G</sup><sup>0</sup> ∼ <sup>N</sup>(0, 1).

$$\hat{\varphi}(\mathbf{v}) := \mathbb{E}\left\{\sigma(\langle \mathbf{v}, \mathbf{G} \rangle + \sqrt{1 - \|\mathbf{v}\|^2} G_0) \varphi(\mathbf{G})\right\}, \quad (\text{A.7})$$

We consider Gaussian process f g (a,W), φ <sup>g</sup> with the same covariance function defined above and define the empirical risk under Gaussian approximation as

$$\begin{aligned} \hat{\mathcal{R}}_n^g(\mathbf{a}, \mathbf{W}) &= \frac{1}{2n} \sum_{i=1}^n (f_i^g(\mathbf{a}, \mathbf{W}) - \varphi_i^g - \varepsilon_i)^2 \\ &= \frac{1}{2n} \|\mathbf{f}^g(\mathbf{a}, \mathbf{W}) - \varphi^g - \varepsilon\|^2, \end{aligned} \quad (\text{A.8})$$

where f g (· · ·) = (<sup>f</sup> g i (· · ·) : <sup>i</sup> ≤ <sup>n</sup>), <sup>φ</sup><sup>g</sup> = (<sup>φ</sup> g i : <sup>i</sup> ≤ <sup>n</sup>), <sup>ε</sup> = (ε<sup>i</sup> : <sup>i</sup> ≤ <sup>n</sup>) are vectors containing <sup>n</sup> i.i.d. copies of the above processes. We will also write y <sup>g</sup> = φ<sup>g</sup> + ε.

Given a model with estimated parameters 
$$\hat{\alpha}$$
,  $\widehat{W}$ , the test error is given by
$$\begin{aligned} \mathcal{R}(\hat{\alpha}, \widehat{W}) &= \frac{1}{2} \mathbb{E}\{(f^g(\hat{\alpha}, \widehat{W}) - \varphi^g - \varepsilon)^2\} \\ &= \frac{1}{2} \mathbb{E}\{(f(x, \hat{\alpha}, \widehat{W}) - \varphi(\mathbf{U}^T x) - \varepsilon)^2\}, \end{aligned} \quad (\text{A.9})$$

where the expectation in the first line is over a triple (f g , φ<sup>g</sup> , ε) independent of the data, and in the second line with respect to x. The two expectations coincide because they depend uniquely on the second moments of these processes.

We are interested in studying the gradient flow dynamics in the random landscape 
$$\widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W})$$

$$\begin{aligned}\dot{\mathbf{a}}(t) &= -\frac{n}{d} \nabla_{\mathbf{a}} \widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)), \\ \dot{\mathbf{w}}_i(t) &= -\frac{n}{d} \nabla_{\mathbf{w}_i} \widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)) - \nu_i(t) \mathbf{w}_i(t) \quad \forall i = 1, \dots, m.\end{aligned}\tag{A.10}$$
The Lagrange multipliers  $\nu_i$  are added to enforce the spherical constraint  $\|\mathbf{w}_i(t)\|^2 = 1$ . While we consider the scale of normalized first lagrange rights, our proposal can be expanded to constrainting all

<sup>n</sup>(a,W)

The Lagrange multipliers <sup>ν</sup><sup>i</sup> are added to enforce the spherical constraint ∥<sup>w</sup>i(t)∥ <sup>2</sup> = 1. While we consider the case of normalized first-layer weights, our approach can be generalized to unconstrained weights or to include weight decay (ridge regularization). As explained in the main text, we will replace this by gradient flow in the Gaussian model <sup>R</sup>b<sup>g</sup> n (a,W). We refer to Section [K](#page-73-0) for a discussion of DMFT in the original non-Gaussian model.

In our analysis we will always consider the proportional asymptotics

$$n, d \rightarrow \infty, \quad \frac{n}{d} \rightarrow \bar{\alpha} \in (0, \infty). \quad (\text{A.11})$$

We typically index sequences and limits by <sup>n</sup>, but it is understood that <sup>d</sup> <sup>=</sup> <sup>d</sup>(n) → ∞ as well. After n, d → ∞ proportionally, we will consider the large network asymptotics <sup>m</sup> → ∞ at fixed α = α/m.

In the following we will drop the superscript <sup>g</sup> and write, for instance <sup>R</sup>b <sup>n</sup>(a,W) instead of Rbg n (a,W) whenever clear from the context. All of our analytical predictions (except for Section [3\)](#page-7-0) are obtained within the Gaussian model.

## B Technique

Notice that each fitting error <sup>F</sup>i(θ) = <sup>y</sup>i−f(x<sup>i</sup> ; <sup>θ</sup>), <sup>i</sup> ∈ {1, . . . , n} is a random function of the model parameters θ. The randomness is due to the randomness in x<sup>i</sup> and in the noise ε<sup>i</sup> . The empirical risk in Eq. [\(1.2\)](#page-1-2) can be rewritten as

$$\widehat{\mathcal{R}}_n(\boldsymbol{\theta}) = \frac{1}{2n} \|\mathbf{F}(\boldsymbol{\theta})\|^2, \quad \mathbf{F}(\boldsymbol{\theta}) = (F_1(\boldsymbol{\theta}), \dots, F_n(\boldsymbol{\theta})). \quad (\text{B.1})$$

Our key approximation consists in replacing the i.i.d. random functions (Fi)i≤<sup>n</sup> by i.i.d. Gaussian processes (F g i )i≤<sup>n</sup> with matching mean and covariance. While DMFT equations have been recently proven without recurring to this approximation (see [\[13\]](#page-10-8) and appendices), their structure is simpler in the Gaussian case, which allows us to carry out the large-m analysis.

Computing the covariance of <sup>F</sup>(·) is a straightforward exercise. We assume for simplicity that an intercept is subtracted so that <sup>E</sup>[σ(G)] = 0, <sup>E</sup>[φ(G)] = 0 and otherwise these functions are generic (G, G1, G and so on will denote standard Gaussian vectors). We then have

$$\mathbb{E}\{f(x; \theta_1) f(x; \theta_2)\} = \frac{1}{m_2} \langle a_1, h(W_1^\top W_2) a_2 \rangle, \quad (\text{B.2})$$

$$\mathbb{E}\{f(x; \theta_1) v\} = \frac{1}{m_1} \langle a_1, \hat{\otimes}(W_1^\top I v) \rangle. \quad (\text{B.3})$$

$$\mathbb{E}\{f(x; \boldsymbol{\theta})y\} = \frac{1}{m^2}\langle a, \hat{\varphi}(\mathbf{W}^T \mathbf{U}) \rangle. \quad (\text{B.3})$$

⟨a, <sup>φ</sup>b(W<sup>T</sup>U)⟩. (B.3) Recall that <sup>θ</sup> = (a,W) where <sup>a</sup> ∈ <sup>R</sup> <sup>m</sup>, <sup>W</sup> = (w1, . . . , <sup>w</sup>m) ∈ <sup>R</sup> <sup>d</sup>×<sup>m</sup> are the first layer weights Finally, <sup>h</sup> : <sup>R</sup> <sup>→</sup> <sup>R</sup>, <sup>φ</sup>b : <sup>R</sup> <sup>k</sup> → <sup>R</sup> encode the activations <sup>σ</sup> and the target function <sup>φ</sup>, with <sup>h</sup> applied entrywise to the matrix W<sup>T</sup> <sup>1</sup>W2.

The covariance of <sup>F</sup>i(θ) = <sup>y</sup><sup>i</sup> − <sup>f</sup>i(x; <sup>θ</sup>) is easily computed from the above, and this defines completely the corresponding Gaussian process (F g i )i≤n. We denote the associated risk function n (θ) := ∥<sup>F</sup> g (θ)∥ <sup>2</sup>/2n.

Rbg Let us emphasize that the cost function <sup>R</sup>b<sup>g</sup> n (θ) remains highly non-trivial despite the fact that the functions F<sup>i</sup> are replaced by Gaussian processes. Near-minima of high-dimensional Gaussian processes have a very rich structure, which is a central theme in spin glass theory [\[39,](#page-12-9) [50\]](#page-12-10). Additional layers of complexity arise here for two reasons. First, <sup>R</sup>b<sup>g</sup> n (θ) is a sum of *squares of Gaussians* and, second, the underlying Gaussian process has a significantly more intricate covariance than in standard spin glasses (where typically depends only on the inner product ⟨<sup>θ</sup>1, <sup>θ</sup>2⟩). Recent work explored the simpler case in which F g i (·) is a Gaussian process with covariance <sup>E</sup>{<sup>F</sup> g i (θ1)F g i (θ2)} <sup>=</sup> <sup>ξ</sup>(⟨<sup>θ</sup>1, <sup>θ</sup>2⟩) depending uniquely on the inner product [\[25,](#page-11-8) [26,](#page-11-9) [51,](#page-12-11) [49,](#page-12-12) [42,](#page-12-13) [43,](#page-12-14) [33\]](#page-11-10). Gradient descent dynamics on these models has been recently studied via DMFT in [\[31,](#page-11-11) [32\]](#page-11-12): our work builds on these advances. DMFT was leveraged before to address other questions in high-dimensional statistics and ML [\[35,](#page-11-13) [12\]](#page-10-14). We refer to [\[8,](#page-10-15) [13\]](#page-10-8) for mathematical results on the DMFT approach.

While <sup>R</sup>b<sup>g</sup> n (θ) has a non-trivial structure, methods from statistical physics can be brought to bear to derive an asymptotic characterization. Namely, define the functions

$$C_{ij}^n(t_1, t_2) = \langle \mathbf{w}_i(t_1), \mathbf{w}_j(t_2) \rangle, \quad \mathbf{v}_i^n(t) := \mathbf{U}^\top \mathbf{w}_i(t), \quad \mathbf{a}_i^n(t). \quad (\text{B.4})$$

These functions are random (because of the random initialization and the randomness in F g ) and depend on n, d. However, as n, d → ∞ with n/d → <sup>α</sup>, they converge to non-random limits (Cij (t1, t2))i<j≤m, (vi(t))i≤m, (ai(t))i<j≤<sup>m</sup> that are the unique solution of a set of coupled integrodifferential equations, see the appendices. We refer to these as to the DMFT equations.

Our main focus is on the behavior of the solutions of these equations for large m and, at first sight, the complexity of the DMFT increases with m. An important simplification arises when choosing a symmetric initial condition <sup>a</sup>i(0) = <sup>a</sup><sup>0</sup> for all <sup>i</sup> ≤ <sup>m</sup>, and (wi(0))i≤<sup>m</sup> ∼iid Unif(<sup>S</sup> d−1 ). Namely, the solution of the DMFT equations is symmetric under permutations of the neurons: <sup>C</sup>ii(t1, t2) = <sup>C</sup>d(t1, t2) for <sup>i</sup> ≤ <sup>m</sup> and <sup>C</sup>ij (t1, t2) = <sup>C</sup>o(t1, t2) for <sup>i</sup> ̸<sup>=</sup> <sup>j</sup> ≤ <sup>m</sup>, while <sup>v</sup>i(t) = <sup>v</sup>(t), <sup>a</sup>i(t) = <sup>a</sup>(t) for <sup>i</sup> ≤ <sup>m</sup>. We then have a reduction to a set of integro-differential equations on <sup>k</sup> + 3 functions, that depend parametrically on m.

We use two approaches to study these equations (see appendix):

- (a) Numerical integration for increasing values of m under different initial conditions.
- (b) Asymptotics as <sup>m</sup> → ∞ (at fixed <sup>α</sup> <sup>=</sup> α/m) via singular perturbation theory [\[9,](#page-10-10) [29\]](#page-11-6).

For (b), a specific dynamical regime is identified by a scaling of the time variable, which in our case will take the form <sup>t</sup> <sup>=</sup> <sup>t</sup>#(m) · <sup>t</sup>ˆfor a certain fixed function <sup>t</sup>#(m) and <sup>t</sup>ˆ= <sup>O</sup>(1) a scaled time. The asymptotics of DMFT quantities in that regime takes the form

$$\lim_{m \rightarrow \infty} \mathbf{v}\left(t_{\#}(m) \cdot \hat{t}; m, \alpha = \frac{\bar{\alpha}}{m}\right) = \mathbf{v}_*(\hat{t}; \alpha). \quad (\text{B.5})$$

## C Dynamical Mean Field Theory (DMFT)

In this section we state the results of Dynamical Mean Field Theory (DMFT). We will outline a heuristic derivation in Section [L.](#page-74-0) We first introduce the general DMFT equations in Section [C.1](#page-23-1) and the corresponding predictions for certain observable of interest in Section [C.2.](#page-26-0) These are a set of Θ(m<sup>2</sup> ) integro-differential equations in as many unknown functions.

We then specialize these equations to the case of a symmetric initialization, in which <sup>w</sup>i(0) ∼ Unif(S d−1 ) and <sup>a</sup>i(0) = <sup>a</sup><sup>0</sup> for all <sup>i</sup> ≤ <sup>m</sup>, see Section [C.3](#page-26-1) In this case, the dynamics is characterized by a set of k + 3 equations which are stated in Sections [C.4](#page-27-0) and [C.5.](#page-28-0)

### C.1 General DMFT equations

Let a n i (t), w<sup>n</sup> i (t), ν n i (t) the the solution of Eq [\(A.10\)](#page-22-0) when the dynamics is initialized at non-random a n i (0) = <sup>a</sup><sup>0</sup>,i, <sup>i</sup> ≤ <sup>n</sup> and possibly random, <sup>w</sup><sup>n</sup> i (0) such that ⟨w<sup>n</sup> i (0), w<sup>n</sup> j (0)⟩ → <sup>C</sup> 0 ij for i, j ≤ <sup>n</sup>, U <sup>T</sup>w<sup>n</sup> i (0) → <sup>v</sup> 0 i for <sup>i</sup> ≤ <sup>n</sup>. While random, the <sup>w</sup><sup>n</sup> i (0) are assumed here to be independent of the random processes f g , φ<sup>g</sup> , ε.

For t, s ≥ <sup>0</sup> consider the quantities

$$C_{ij}^m(t, s) := \langle w_i^n(t), w_j^n(s) \rangle, \quad v_i^n(t) := U^\top w_i^n(t). \quad (\text{C.1})$$

Then DMFT predicts that these quantities have a well defined non-random limit as n, d → ∞,

$$C_{ij}(t, s) = \lim_{n, d \rightarrow \infty} C_{ij}^n(t, s), \quad \mathbf{v}_i(t) = \lim_{n, d \rightarrow \infty} \mathbf{v}_i^n(t), \quad a_i(t) = \lim_{n, d \rightarrow \infty} a_i^n(t), \quad (\text{C.2})$$

where the limits are understood to hold in almost sure sense. These limits are the unique solution of a set of integro-differential equations in the unknowns {<sup>C</sup>ij (t, s), Rij (t, s), <sup>v</sup>i(t), ai(t) : i, j ≤ <sup>m</sup>}, which we next state as three sets: (1) Dynamical equations; (2) Equations for auxiliary functions; (3) Boundary conditions. Before that, we mention some constraints that need to be satisfied by the solution of these equations.

(0) Constraints. The functions Cij (t, s), Rij (t, s) satisfy:

$$\begin{aligned} C_{ii}(t, t) &= 1 \quad \forall 0 \leq t, \\ C_{-i}(t, e) &= C_{-i}(e, t) \quad \forall 0 \leq t, e \end{aligned} \quad (\text{C.3}) \quad (\text{C.4})$$

$$C_{ij}(t, s) = C_{ji}(s, t) \quad \forall 0 \leq t, s, \quad (\text{C.4})$$

$$R_{-}(t, s) = 0, \quad \forall 0 \leq t \leq s. \quad (\text{C.5})$$

$$R_{ij}(t, s) = 0 \quad \forall 0 \leq t < s. \quad (\text{C.5})$$

The first condition in particular implies the following useful relation:

$$\frac{dC_{ij}(t, t)}{dt} = \lim_{t' \rightarrow t} \left[ \frac{\partial C_{ij}(t, t')}{\partial t} + \frac{\partial C_{ji}(t, t')}{\partial t} \right]. \quad (\text{C.6})$$

We refer to the property [\(C.5\)](#page-24-0) (and similar ones for R functions appearing below) as 'causality constraint.'

(1) Dynamical equations. These equations determine the dynamics of {<sup>C</sup>ij (t, s), Rij (t, s), <sup>v</sup>i(t), ai(t) : i, j ≤ <sup>m</sup>}, and involve the auxiliary functions (memory kernels) M<sup>C</sup> ij (t, s), M<sup>R</sup> ij (t, s) and (Lagrange multipliers) νi(t) (the last equations assume implicitly t<sup>a</sup> > tb):

$$\begin{aligned} \frac{da_i(t)}{dt} &= -\frac{\bar{\alpha}}{m} \int_0^t R_A(t, s) \left[ \frac{1}{m} \sum_{l=1}^m a_l(s) h(C_{li}(s, t)) - \hat{\varphi}(\mathbf{v}_i(t)) \right] ds \\ &\quad - \frac{\bar{\alpha}}{m} \int_0^t C_A(t, s) \frac{1}{m} \sum_{l=1}^m a_l(s) h'(C_{li}(s, t)) R_{il}(t, s) ds, \end{aligned} \quad (\text{C.7})$$

$$\frac{d\mathbf{v}_i(t)}{dt} = -\nu_i(t)\mathbf{v}_i(t) + \frac{\bar{\alpha}}{m}a_i(t)\nabla\hat{\varphi}(\mathbf{v}_i(t)) \int_0^t R_A(t,s) ds - \frac{1}{m} \sum_{j=1}^m \int_0^t M_{ij}^R(t,s) \mathbf{v}_j(s) ds, \quad (\text{C.8})$$

$$\frac{\partial C_{ij}(t_a, t_b)}{\partial t_a} = -\nu_i(t_a)C_{ij}(t_a, t_b) + \frac{\bar{\alpha}}{m}a_i(t_a)\langle\nabla\hat{\varphi}(\mathbf{v}_i(t_a)), \mathbf{v}_j(t_b)\rangle \int_0^{t_a} R_A(t_a, s) \, ds \quad (\text{C.9})$$

$$-\frac{1}{m} \sum_{l=1}^m \int_0^{t_a} M_{il}^R(t_a, s) C_{lj}(s, t_b) ds - \frac{1}{m} \sum_{l=1}^m \int_0^{t_b} M_{il}^C(t_a, s) R_{jl}(t_b, s) ds,$$

$$\frac{\partial R_{ij}(t_a, t_b)}{\partial t_a} = -\nu_i(t_a)R_{ij}(t_a, t_b) + \delta_{ij}\delta(t_a - t_b) - \frac{1}{m} \sum_{l=1}^m \int_{t_b}^{t_a} M_{il}^R(t_a, s) R_{lj}(s, t_b) \, ds. \quad (\text{C.10})$$

We point out that the <sup>δ</sup>(t<sup>a</sup> − <sup>t</sup>b) in the last equation (together with Eq. [\(C.5\)](#page-24-0)) has to be interpreted as follows: Rij (t, t′ ) = 0 for t < t′ while, for ε > 0, Rij (t + ε, t) = δij + oε(1).

Equations [\(C.9\)](#page-24-1) and [\(C.10\)](#page-24-2) can also be written in terms of an effective stochastic process in R m: w<sup>e</sup> (t) = (w e i (t) : <sup>i</sup> ≤ <sup>m</sup>). This is defined as the solution of the following set of ODEs (for

<sup>i</sup> ∈ {1, . . . , m}):

$$\frac{dw_i^e(t)}{dt} = -\nu_i(t)w_i^e(t) + \alpha a_i(t)\langle \nabla \hat{\varphi}(\mathbf{v}(t)), \mathbf{v}(t') \rangle \int_0^t R_A(t, s) ds \quad (\text{C.11})$$

$$(i) (t) w_i^e(t) + \alpha a_i(t) \langle \nabla \varphi(\mathbf{v}(t)), \mathbf{v}(t') \rangle \int_0^t R_A(t, s) ds \quad (\text{C.11})$$

$$- \frac{1}{m} \sum_{l=1}^m \int_0^t M_{il}^R(t, s) w_l^e(s) ds + \eta_i(t) + b_i(t), \quad (\text{C.12})$$

$$(ii) (t) w_i^e(t) + \alpha a_i(t) \langle \nabla \varphi(\mathbf{v}(t)), \mathbf{v}(t') \rangle \int_0^t R_A(t, s) ds \quad (\text{C.13})$$

$$w_i^\epsilon(0) \sim \mathbf{N}(0, 1), \quad (\text{C.13})$$

where (ηi(t) : <sup>i</sup> ≤ <sup>m</sup>) is a centered Gaussian process with covariance

$$\mathbb{E}[\eta_i(t)\eta_j(t')] = -\frac{1}{m}M_{ij}^C(t,t'). \quad (\text{C.14})$$

Define <sup>b</sup>(t) = (bi(t) : <sup>i</sup> ≤ <sup>m</sup>). The solution of Eqs. [\(C.9\)](#page-24-1) and [\(C.10\)](#page-24-2) can be written as

$$C_{ij}(t, t') = \lim_{b \rightarrow 0} \mathbb{E} [w_i(t)w_j(t')] , \quad (\text{C.15})$$

$$R_{ij}(t, t') = \lim_{\underline{b} \rightarrow 0} \frac{\delta \mathbb{E}[w_i(t)]}{\delta b_j(t')}. \quad (\text{C.16})$$

In fact the stochastic process of Eq. [\(C.11\)](#page-25-0) is expected to describe the limit distribution of the secondlayer weights <sup>W</sup>(t). Namely, for <sup>i</sup> ≤ <sup>d</sup>, define <sup>w</sup>˜ <sup>i</sup>(t) = <sup>W</sup>(t)e<sup>i</sup> ∈ <sup>R</sup> <sup>m</sup> be a vector containing the i-th coordinate of each neuron. Then, for any fixed i and any T,

$$(\tilde{w}_i(t) : 0 \leq t \leq T) \xrightarrow{d} (w^e(t) : 0 \leq t \leq T). \quad (\text{C.17})$$

Here <sup>d</sup> ⇒ denotes convergence in distribution as n, d → ∞, in <sup>C</sup>([0, T], <sup>R</sup> <sup>m</sup>).

(2) Equations for auxiliary functions. The memory kernels M<sup>R</sup> and M<sup>C</sup> are defined by

$$\begin{aligned} M_{ij}^R(t, s) &= \frac{\bar{\alpha}}{m} [R_A(t, s)h'(C_{ij}(t, s)) + C_A(t, s)h''(C_{ij}(t, s))R_{ij}(t, s)] a_i(t)a_j(s), \\ M_{ij}^C(t, s) &= \frac{\bar{\alpha}}{m} C_A(t, s)h'(C_{ij}(t, s))a_i(t)a_j(s). \end{aligned} \quad (\text{C.18})$$

where the functions R<sup>A</sup> and C<sup>A</sup> satisfy the symmetry properties CA(t, s) = CA(s, t) and RA(t, s) = 0 for t < s, and are the unique solution

$$\begin{aligned} \int_{t'}^t [\delta(t-s) + \Sigma_R(t,s)] R_A(s,t') ds &= \delta(t-t'), \\ \int_0^t [\delta(t-s) + \Sigma_R(t,s)] C_A(s,t') ds + \int_0^{t'} \Sigma_C(t,s) R_A(t',s) ds &= 0, \end{aligned} \quad (\text{C.19})$$

where

$$\begin{aligned} \Sigma_C(t, s) &:= \tau^2 + \|\varphi\|^2 + \frac{1}{m^2} \sum_{i,j=1}^m a_i(t) a_j(s) h(C_{ij}(t, s)) \\ &\quad - \frac{1}{m} \sum_{l=1}^m a_l(t) \hat{\varphi}(\mathbf{v}_l(t)) - \frac{1}{m} \sum_{l=1}^m a_l(s) \hat{\varphi}(\mathbf{v}_l(s)), \end{aligned} \quad (\text{C.20})$$

$$\Sigma_R(t, s) := \frac{1}{m^2} \sum_{i,j=1}^m a_i(t) a_j(s) h'(C_{ij}(t, s)) R_{ij}(t, s).$$

The Lagrange multipliers νi(t) have to be fixed to enforce the constraint Cii(t, t) = 1 which follows from <sup>w</sup><sup>α</sup> ∈ <sup>S</sup> d−1 . The corresponding equations are

$$\begin{aligned} \nu_i(t_a) &= \frac{\bar{\alpha}}{km} a_i(t_a) \langle \mathbf{v}_i(t_a), \nabla \hat{\varphi}(\mathbf{v}_i(t_a)) \rangle \int_0^{t_a} R_A(t_a, s) \, ds \\ &- \frac{1}{m} \sum_{j=1}^m \int_0^{t_a} M_{ij}^R(t_a, s) C_{ij}(s, t_a) \, ds - \frac{1}{m} \sum_{j=1}^m \int_0^{t_a} M_{ij}^C(t_a, s) R_{ji}(t_a, s) \, ds. \end{aligned} \quad (\text{C.21})$$

(3) Boundary conditions. The dynamical equations [\(C.7\)](#page-24-3) to [\(C.10\)](#page-24-2) can be integrated from a set of initial conditions that reflect initial conditions of the GF dynamics:

$$\begin{aligned} \mathbf{v}_i(0) &= \mathbf{v}_i^0, & \mathbf{a}_i(0) &= \mathbf{a}_i^0, & \forall i \in \{1, \dots, m\}, \\ C_{ij}(0, 0) &= C_{ij}^0, & \forall i, j \in \{1, \dots, m\}, & & (\text{C.22}) \\ R_{ij}(0, 0) &= 0, & \forall i, j \in \{1, \dots, m\}. & & \end{aligned}$$

## C.2 Expressions for train and test error

The asymptotics of many quantities of interest can be expressed in terms of the solutions of the DMFT equations stated in the last section. In particular, the train error <sup>R</sup>b <sup>n</sup>(W(t), a(t)) and test error R(W(t), a(t)) at time t have well defined limits under the proportional asymptotics:

$$\lim_{n \rightarrow \infty} \mathcal{R}_n^g(\mathbf{W}(t), \mathbf{a}(t)) = e_{\text{tr}}(t), \quad \lim_{n \rightarrow \infty} \mathcal{R}^g(\mathbf{W}(t), \mathbf{a}(t)) = e_{\text{ts}}(t). \quad (\text{C.23})$$

$$e_{\text{tr}}(t) \ e_{\text{ts}}(t) \text{ are given by}$$

The functions etr(t) ets(t) are given by

$$e_{\text{tr}}(t) = -\frac{1}{2}C_A(t, t), \quad (\text{C.24})$$

$$e_{\text{ts}}(t) = \frac{1}{2} \left\{ \tau^2 + \frac{1}{k} \|\varphi\|^2 + \frac{1}{m^2} \sum_{i,j=1}^m h(C_{ij}(t,t)) - \frac{2}{m} \sum_{i=1}^m \hat{\varphi}(\mathbf{v}_i(t)) \right\} \quad (\text{C.25})$$

More generally, CA(t, s) gives the asymptotics of the correlation of residuals:

$$\lim_{n \rightarrow \infty} \frac{1}{n} \langle \Delta(t), \Delta(s) \rangle = -C_A(t, s), \quad (\text{C.26})$$

$$\Delta(t) := \mathbf{y}^g - \mathbf{f}^g(\mathbf{a}(t), \mathbf{W}(t)). \quad (\text{C.27})$$

$$\Delta(t) := y^g - f^g(a(t), W(t)). \quad (\text{C.27})$$

where we recall that y <sup>g</sup> = φ<sup>g</sup> + ε.

## C.3 Symmetric initialization and solutions

As anticipated, we consider the uninformative initialization w<sup>n</sup> i (0) ∼ Unif(<sup>S</sup> d−1 ) and a n i (0) = a<sup>0</sup> for all <sup>i</sup> ≤ <sup>m</sup>. This results in the following initialization for the DMFT equations of

$$\begin{aligned} \mathbf{v}_i(0) &= \mathbf{v}_i^0 = \mathbf{v}_i \in \{1, \dots, m\}, \\ C_{i \neq j}(0, 0) &= C_{i \neq j}^0 = \mathbf{v}_i \neq \mathbf{v}_j, i, j \in \{1, \dots, m\}, \\ C_{ii}(0, 0) &= C_{ii}^0 = \mathbf{v}_i \in \{1, \dots, m\}. \end{aligned} \tag{C.28}$$

This initialization is invariant under permutations of the m neurons. Since the DMFT equations of Section [C.1](#page-23-1) are equivariant under such permutations, their solution is also invariant under permutations. This means that it takes the form:

$$C_{ij}(t, t') = \begin{cases} C_d(t, t') & \text{if } i = j, \\ C_o(t, t') & \text{if } i \neq j, \end{cases} \quad R_{ij}(t, t') = \begin{cases} R_d(t, t') & \text{if } i = j, \\ R_o(t, t') & \text{if } i \neq j, \end{cases} \quad (\text{C.29})$$

$$\mathbf{v}_i(t) = \mathbf{v}(t) , \quad \nu_i(t) = \nu(t) , \quad a_i(t) = a(t) \quad \forall i . \quad (\text{C.30})$$

As a consequence, the memory kernels in Eq. [\(C.18\)](#page-25-1) take the form

$$M_{ij}^C(t, t') = \begin{cases} M_d^C(t, t') & \text{if } i = j, \\ M_o^C(t, t') & \text{if } i \neq j, \end{cases} \quad M_{ij}^R(t, t') = \begin{cases} M_d^R(t, t') & \text{if } i = j, \\ M_o^R(t, t') & \text{if } i \neq j. \end{cases} \quad (\text{C.31})$$

## C.4 DMFT equations for symmetric initialization (SymmDMFT )

(1) Dynamical equations. Substituting the ansats of the previous section in the equations of Section [C.1,](#page-23-1) we obtain the following equations for the functions a(t), v(t), Cd(t, s), Co(t, s), Rd(t, s), Ro(t, s):

$$\begin{aligned} \frac{da}{dt}(t) &= \frac{\bar{\alpha}}{m} \hat{\varphi}(\mathbf{v}(t)) \int_0^t R_A(t, s) ds & (C.32) \\ &- \frac{\bar{\alpha}}{m} \int_0^t R_A(t, s) a(s) \left[ \frac{1}{m} h(C_d(t, s)) + \frac{m-1}{m} h(C_o(t, s)) \right] ds \\ &- \frac{\bar{\alpha}}{m} \int_0^t C_A(t, s) a(s) \left[ \frac{1}{m} h'(C_d(t, s)) R_d(t, s) + \frac{m-1}{m} h'(C_o(t, s)) R_o(t, s) \right] ds, \\ \frac{d\mathbf{v}}{dt}(t) &= -\nu(t) \mathbf{v}(t) + \frac{\bar{\alpha}}{m} \nabla \hat{\varphi}(\mathbf{v}(t)) a(t) \int_0^t R_A(t, s) ds & (C.33) \\ &- \frac{1}{m} \int_0^t \left[ M_R^{(d)}(t, s) + (m-1) M_R^{(o)}(t, s) \right] \mathbf{v}(s) ds, \end{aligned}$$

$$\partial_t C_d(t, t') = -\nu(t) C_d(t, t') + \frac{\bar{\alpha}}{m} \langle \nabla \dot{\varphi}'(\mathbf{v}(t)), \mathbf{v}(t') \rangle a(t) \int_0^t R_A(t, s) \, \mathrm{d}s \quad (\text{C.34})$$

$$\begin{aligned} \partial_t C_d(t, t') = & -\nu(t)C_d(t, t') + \frac{\bar{\alpha}}{m} \langle \nabla \dot{\varphi}'(\mathbf{v}(t)), \mathbf{v}(t') \rangle a(t) \int_0^t R_A(t, s) \, ds \\ & - \frac{1}{m} \int_0^t \left[ M_R^{(d)}(t, s) C_d(t', s) + (m-1) M_R^{(o)}(t, s) C_o(t', s) \right] \, ds \\ & - \frac{1}{m} \int_0^{t'} \left[ M_C^{(d)}(t, s) R_d(t', s) + (m-1) M_C^{(o)}(t, s) R_o(t', s) \right] \, ds, \end{aligned} \quad (\text{C.34})$$

$$\begin{aligned} \partial_t C_o(t, t') &= -\nu(t) C_o(t, t') + \frac{\bar{\alpha}}{m} \langle \nabla \hat{\varphi}(\mathbf{v}(t)), \mathbf{v}(t') \rangle a(t) \int_0^t R_A(t, s) \, ds \\ &\quad - \frac{1}{m} \int_0^t \left[ M_R^{(d)}(t, s) C_o(t', s) + M_R^{(o)}(t, s) C_d(t', s) + (m-2) M_R^{(o)}(t, s) C_o(t', s) \right] \, ds \\ &\quad - \frac{1}{m} \int_0^{t'} \left[ M_C^{(d)}(t, s) R_o(t', s) + M_C^{(o)}(t, s) R_d(t', s) + (m-2) M_C^{(o)}(t, s) R_o(t', s) \right] \, ds, \\ \partial_t R_d(t, t') &= -\nu(t) R_d(t, t') + \delta(t - t') \end{aligned} \quad (\text{C.36})$$

− 1 m

$$\begin{aligned} & -\frac{1}{m} \int_{t'}^t \left[ M_R^{(d)}(t,s)R_d(s,t') + (m-1)M_R^{(o)}(t,s)R_o(s,t') \right] ds, \\ \partial_t R_o(t,t') &= -\nu(t)R_o(t,t') - \frac{1}{m} \int_{t'}^t \left[ M_R^{(d)}(t,s)R_o(s,t') + M_R^{(o)}(t,s)R_d(s,t') \right. \\ &\quad \left. + (m-2)M_R^{(o)}(t,s)R_o(s,t') \right] ds. \end{aligned} \quad (\text{C.37})$$

M (d)

<sup>R</sup> (t, s)Rd(s, t′

) + (<sup>m</sup> − 1)<sup>M</sup>

(o)

<sup>R</sup> (t, s)Ro(s, t′

)

ds ,

(2) Equations for auxiliary functions. The memory kernels M (s) <sup>R</sup> (t, s), M (o) <sup>R</sup> (t, s) and M (s) C (t, s), M (o) C (t, s) are given by:

$$M_R^{(d)}(t, s) = \frac{\bar{\alpha}}{m} a(t) a(s) [R_A(t, s) h'(C_d(t, s)) + C_A(t, s) h''(C_d(t, s)) R_d(t, s)] , \quad (\text{C.38})$$

$$M_R^{(o)}(t, s) = \frac{m}{\alpha} a(t) a(s) [R_A(t, s) h'(C_o(t, s)) + C_A(t, s) h''(C_o(t, s)) R_o(t, s)] , \quad (\text{C.39})$$

$$M_C^{(d)}(t, s) = \frac{\bar{\alpha}}{m} a(t) a(s) C_A(t, s) h'(C_d(t, s)), \quad (\text{C.40})$$

$$M_C^{(o)}(t, s) = \frac{\bar{\alpha}}{m} a(t) a(s) C_A(t, s) h'(C_o(t, s)). \quad (\text{C.41})$$

Further, CA(t, s), RA(t, s) are given by the same equations [\(C.19\)](#page-25-2), where Σ<sup>C</sup> , Σ<sup>R</sup> are simplified as follows:

$$\begin{aligned} \Sigma_C(t, s) &= \tau^2 + \|\varphi\|^2 - a(t)\hat{\varphi}(\mathbf{v}(t)) - a(s)\hat{\varphi}(\mathbf{v}(s)) + \frac{a(t)a(s)}{m}h(C_d(t, s)) \\ &\quad + \frac{m-1}{m}a(t)a(s)h(C_o(t, s)) \\ \Sigma_R(t, s) &= \frac{a(t)a(s)}{m}h'(C_d(t, s))R_d(t, s) + \frac{m-1}{m}a(t)a(s)h'(C_o(t, s))R_o(t, s) \end{aligned} \quad (\text{C.42})$$

Finally, the Lagrange multipliers are determined by

$$\begin{aligned} \nu(t) &= \frac{\bar{\alpha}}{m} \langle \nabla \hat{\varphi}(\mathbf{v}(t)), \mathbf{v}(t) \rangle a(t) \int_0^t R_A(t, s) \, ds \\ &\quad - \frac{1}{m} \int_0^t \left[ M_R^{(s)}(t, s) C_d(t, s) + (m-1) M_R^{(o)}(t, s) C_o(t, s) \right] \, ds \\ &\quad - \frac{1}{m} \int_0^t \left[ M_C^{(s)}(t, s) R_d(t, s) + (m-1) M_C^{(o)}(t, s) R_o(t, s) \right] \, ds . \end{aligned} \quad (\text{C.43})$$

(3) Boundary conditions. As anticipated the SymmDMFT is initialized as

$$\mathbf{v}(0) = \mathbf{0}, \quad C_d(0,0) = 1 \quad C_o(0,0) = 0. \quad (\text{C.44})$$

## C.5 Expressions for train and test error under symmetric initialization

The general expression for train and test error given in Section [C.2](#page-26-0) specialize to:

$$e_{\text{tr}}(t) = -\frac{1}{2}C_A(t, t), \quad (\text{C.45})$$

$$e_{\text{ts}}(t) = \frac{1}{2} \left[ \tau^2 + \|\varphi\|^2 - 2a(t)\hat{\varphi}(\mathbf{v}(t)) + \frac{1}{m}a^2(t)h(1) + \frac{m-1}{m}a^2(t)h(C_o(t, t)) \right]. \quad (\text{C.46})$$

## D Numerical integration of the DMFT equations

## D.1 Integration technique

We integrate the SymmDMFT equations [\(C.32\)](#page-27-1) to [\(C.37\)](#page-27-2) using a standard Euler discretization. Namely, we discretize time on an equi-spaced grid <sup>t</sup> ∈ <sup>T</sup> := {0, η, <sup>2</sup>η, . . . } and approximate derivatives by differences and integrals by sums on this grid. As an example, Eq. [\(C.32\)](#page-27-1) is replaced by

$$\begin{aligned} \frac{a(t+\eta) - a(t)}{\eta} &= \frac{\bar{\alpha}}{m} \hat{\varphi}(\mathbf{v}(t)) \sum_{s \in \mathbb{T}, s \leq t} R_A(t, s) \eta \\ &\quad - \frac{\bar{\alpha}}{m} \sum_{s \in \mathbb{T}, s \leq t} R_A(t, s) a(s) \left[ \frac{1}{m} h(C_d(t, s)) + \frac{m-1}{m} h(C_o(t, s)) \right] \eta \\ &\quad - \frac{\bar{\alpha}}{m} \sum_{s \in \mathbb{T}, s \leq t} C_A(t, s) a(s) \left[ \frac{1}{m} h'(C_d(t, s)) R_d(t, s) + \frac{m-1}{m} h'(C_o(t, s)) R_o(t, s) \right] \eta. \end{aligned} \quad (\text{D.1})$$

The discretization of Eq. [\(C.10\)](#page-24-2) deserves an additional clarification because of the delta-function. For <sup>t</sup><sup>a</sup> ≥ <sup>t</sup>b, <sup>t</sup>a, t<sup>b</sup> ∈ <sup>N</sup>η, we compute

$$\frac{R_{ij}(t_a + \eta, t_b) - R_{ij}(t_a, t_b)}{\eta} = -\nu_i(t_a) R_{ij}(t_a, t_b) - \frac{1}{\eta} \delta_{ij} \mathbf{1}_{t_a=t_b}$$

$$-\frac{1}{m} \sum_{l=1}^m \sum_{s \in [t_a, t_b] \cap \mathbb{N}_\eta} M_{il}^R(t_a, s) R_{lj}(s, t_b) \eta,$$

with boundary condition

$$R_{ij}(t_b, t_b) = 0 \quad \forall i, j \leq m.$$

Of course, the solution of this system of difference equation does not coincide with the solution of the original equations [\(C.32\)](#page-27-1) to [\(C.37\)](#page-27-2), and in this section we will write a(t; η), Co(t, s; η) and so on to emphasize the distinction.

Equations [\(C.42\)](#page-28-1) can be directly interpreted as determining <sup>Σ</sup><sup>C</sup> (t, s) and <sup>Σ</sup>R(t, s) on the grid t, s ∈ <sup>T</sup>. Finally, we discretize Eq. [\(C.19\)](#page-25-2) as

$$\begin{aligned} \sum_{s \in \mathbb{T}} [\mathbf{1}_{t=s} + \Sigma_R(t, s)\eta] R_A(s, t') &= \frac{1}{\eta} \mathbf{1}_{t=t'} , \\ \sum_{s \in \mathbb{T}} [\mathbf{1}_{t=s} + \Sigma_R(t, s)\eta] C_A(s, t') + \sum_{s \in \mathbb{T}} \Sigma_C(t, s) R_A(t', s) \eta &= 0 . \end{aligned} \quad (\text{D.2})$$

Note that we dropped the integration limits here, since they are enforced by the causality constraints implying <sup>Σ</sup>R(t, s) = 0, <sup>R</sup>A(t, s) = 0 for t < s. Defining the matrices <sup>Σ</sup><sup>R</sup> = (ΣR(t, s) : t, s ∈ <sup>T</sup>), and similarly for Σ<sup>C</sup> , CA, RA, we can rewrite [\(D.2\)](#page-29-0) as

$$[I + \eta \Sigma_R] \mathbf{R}_A = \frac{1}{\eta} I, \quad (\text{D.3})$$

$$[\mathbf{I} + \eta \boldsymbol{\Sigma}_R] \mathbf{C}_A + \eta \boldsymbol{\Sigma}_C \mathbf{R}_A = \mathbf{0}. \quad (\text{D.4})$$

We truncate these matrices (which are infinite) to a maximum time T (e.g., redefine Σ<sup>R</sup> = (ΣR(t, s) : t, s ∈ <sup>T</sup>, s, t ≤ <sup>T</sup>)) and solve these equations by matrix inversion:

$$R_A = \frac{1}{\eta} (I + \eta \Sigma_R)^{-1}, \quad (\text{D.5})$$

$$C_A = -(I + \eta \Sigma_R)^{-1} \Sigma_C (I + \eta \Sigma_R)^{-1}. \quad (\text{D.6})$$

$$\mathbf{C}_A = -(\mathbf{I} + \eta \mathbf{\Sigma}_R)^{-1} \mathbf{\Sigma}_C (\mathbf{I} + \eta \mathbf{\Sigma}_R)^{-1}. \quad (\text{D.6})$$

We denote by a(t; η), v(t; η), Co(t, s; η), Cd(t, s; η), Ro(t, s; η), Rd(t, s; η), the functions obtained via the Euler integration scheme. We will assume that this solution is interpolated continuously for t, s ̸∈ <sup>T</sup>. For instance, for i, j ∈ <sup>N</sup> a, b ∈ [0, 1), we let

$$C_d((i+a)\eta, (j+b)\eta; \eta) = (1-a)(1-b) C_d(i\eta, j\eta; \eta) + a(1-b) C_d((i+1)\eta, j\eta; \eta) \quad (\text{D.7}) \\ + (1-a)b C_d((i+1)\eta, j\eta; \eta) + ab C_d((i+1)\eta, (j+1)\eta; \eta).$$

Finally, while we described the discretization procedure for the SymmDMFT , the discussion above applies verbatimly for the full DMFT of Section [C.1.](#page-23-1)

The DMFT equations and their symmetric specialization have a causal structure which means that they can be integrated by progressively by increasing T. Furthermore there is no self-consistency condition in the integration scheme at variance with the non-Gaussian settings, see for example [\[40\]](#page-12-15). This simplification allows to investigate the long time behavior of the dynamics in a numerical, rather efficient, way.

#### D.2 Accuracy of the numerical integration scheme

The discretization of DMFT is expected to converge to the actual solution with errors of order η. Namely, we expect

$$C_d(t, t'; \eta) = C_d(t, t') + O(\eta), \quad C_o(t, t'; \eta) = C_o(t, t') + O(\eta), \quad (\text{D.8})$$

and similarly for the other functions. We refer to [\[13\]](#page-10-8) for related examples in which the convergence was proved rigorously, and to [\[31\]](#page-11-11) for an empirical study in a closely related model.

In order to test the accuracy of our approach, and the correctness of the DMFT equations, we simulated the gradient descent (GD) dynamics for the Gaussian model. Namely, we generate realizations of the process f g (a,W) = (f g i (a,W) : <sup>i</sup> ≤ <sup>n</sup>) with the prescribed covariance [\(A.4\)](#page-21-1), and the vector φ<sup>g</sup> = (φ g i : <sup>i</sup> <sup>≤</sup> <sup>n</sup>) with same covariance as in Eq. [\(A.6\)](#page-21-2) (see Section [D.4.](#page-32-0)) We define <sup>R</sup>b <sup>n</sup>(a,W) via Eq. [\(A.8\)](#page-21-3) and implement the following GD iteration

$$\begin{aligned} \mathbf{a}^n(t + \eta_{\text{GD}}) &= \mathbf{a}^n(t) - \frac{\eta_{\text{GD}}n}{d} \nabla_{\mathbf{a}} \widehat{\mathcal{R}}_n(\mathbf{a}^n(t), \mathbf{W}^n(t)), \\ \mathbf{w}_i^n(t + \eta_{\text{GD}}) &= \mathbf{P}_{\mathbb{S}^{d-1}} \left( \mathbf{w}_i^n(t) - \frac{\eta_{\text{GD}}n}{d} \nabla_{\mathbf{w}_i} \widehat{\mathcal{R}}_n(\mathbf{a}^n(t), \mathbf{W}^n(t)) \right), \end{aligned} \quad (\text{D.9})$$

where P <sup>S</sup> <sup>d</sup>−<sup>1</sup> is the projector to the unit sphere, i.e. P <sup>S</sup> <sup>d</sup>−<sup>1</sup> (x) = <sup>x</sup>/∥x∥ if <sup>x</sup> ̸<sup>=</sup> <sup>0</sup> and <sup>P</sup> <sup>S</sup> <sup>d</sup>−<sup>1</sup> (0) = 0. Note that the trajectories of Eq. [\(D.9\)](#page-29-1) depend on the sample size n (and hence the dimension d = dn) and the stepsize ηGD. To emphasize this dependence, we also use the notation a <sup>n</sup>(t; ηGD) W<sup>n</sup> (t; ηGD). We expect the GD trajectories defined by Eq. [\(D.9\)](#page-29-1) approach the GF trajectories defined by Eq. [\(A.10\)](#page-22-0) as <sup>η</sup>GD → <sup>0</sup> uniformly in n, d. Namely,

$$\lim_{\eta_{\text{GD}} \rightarrow 0} \limsup_{n, d \rightarrow \infty} \|\mathbf{W}^n(t; \eta_{\text{GD}}) - \mathbf{W}^n(t)\|_F = 0, \quad (\text{D.10})$$

$$\lim_{\eta_{\text{GD}} \rightarrow 0} \limsup_{n, d \rightarrow \infty} \|\mathbf{a}^n(t; \eta_{\text{GD}}) - \mathbf{a}^n(t)\|_2 = 0, \quad (\text{D.11})$$

where the limits are understood to hold in probability for any fixed t. Informally, for fixed small ηGD, GD dynamics is a good approximation to GF dynamics, irrespective of the dimension.

We generate several realizations of the processes f g , φ<sup>g</sup> , and of the gradient descent trajectories [\(D.9\)](#page-29-1). We average observables of interest over these realizations and compare these with the Euler discretization of the DMFT equations. For instance, consider the correlation functions Cij (t, s). Then we can compare:

- C n ij (t, s; <sup>η</sup>GD) = <sup>E</sup>⟨w<sup>n</sup> i (t; ηGD), w<sup>n</sup> j (s; <sup>η</sup>GD)⟩ where the expectation is taken with respect to the GD process [\(D.9\)](#page-29-1).
- Cij (t, s; η), the solution of the Euler discretization of the DMFT, described in the previous section.

Some results of this comparison are presented in the next subsection. This comparison allows us to gauge two types of systematic effects:

- 1. The effect of finite n, d. Indeed, the DMFT equations characterize the n, d → ∞ limit of the GD dynamics [\(D.9\)](#page-29-1).
- 2. The non-zero stepsize η. Note that the effect of discretization introduced in the DMFT equations are different from the ones in the gradient descent [\(D.9\)](#page-29-1). Therefore the disagreement between the two is a measure of the nonzero-η effects.

To clarify further the last point, we emphasize that, despite the notation, Cij (t, s; η) is not the n, d → ∞ limit of <sup>C</sup> n ij (t, s; η).

We note in passing that it is possible to derive DMFT equations for GD, hence characterizing limn→∞ C n ij (t, s; ηGD). Similar characterizations were obtained for related (simpler) models in [\[40,](#page-12-15) [13,](#page-10-8) [41,](#page-12-16) [31,](#page-11-11) [32\]](#page-11-12). We defer the analysis of GD with large stepsizes to future work.

## D.3 Testing the numerical accuracy

Figures [6](#page-31-0) and [7](#page-31-1) we present examples of the numerical comparison described in the previous section, under two different settings, as described below.

Setting 1. We assume pure noise data with τ = 1 and train a network with m = 4 neurons and covariance structure given by h(z) = z/10 + z <sup>2</sup>/2. We simulate GD trajectories, according to Eq. [\(D.9\)](#page-29-1) with d = 100, n = 150, and correspondingly evaluate the Euler discretization of DMFT, cf. Section [D.1](#page-28-2) for α = n/d = 1.5.

We choose an initialization that is not symmetric and therefore we have to use the full DMFT equations of Section [C.1.](#page-23-1) More precisely, we initialize second layer weights as follows:

$$a_1(0) = a_2(0) = 1 \quad a_3(0) = a_4(0) = -1 \quad (\text{D.12})$$

The weights of the first layer are instead initialized by generating two random vectors y<sup>1</sup> , <sup>y</sup><sup>2</sup> ∼ Unif(S d−1 ), and setting

$$w_1(0) = w_3(0) = y_1 \quad w_2(0) = w_4(0) = y_2 \quad (\text{D.13})$$

This initialization results in initializing the DMFT equations with

$$\begin{aligned} C_{11}(0, 0) &= C_{22}(0, 0) = C_{33}(0, 0) = C_{44}(0, 0) = 1, \\ C_{13}(0, 0) &= C_{24}(0, 0) = 1, \\ C_{12}(0, 0) &= C_{14}(0, 0) = C_{23}(0, 0) = C_{34}(0, 0) = 0. \end{aligned} \tag{D.14}$$

![](_page_31_Figure_0.jpeg)

Figure 6: Comparison between discretized DMFT and GD dynamics for the Gaussian model (labeled as 'Simulations'). GD results are averaged over N = 10<sup>4</sup> realizations of the Gaussian process, under Setting 1 described in the main text. Left frame: Second layer for DMFT and GD with ηGD = η = 0.1. Right frame: Train error and correlation function for DMFT with a few values of η, and for GD.

![](_page_31_Figure_2.jpeg)

Figure 7: Comparison between discretized DMFT and GD dynamics (labeled as 'Simulations'). GD results are averaged over N = 10<sup>4</sup> realizations of the Gaussian process, under Setting 2 described in the main text. Results for GD are averaged over N = 10<sup>4</sup> samples.

Both for the discretized DMFT and for GD for several values of the stepsize. The results of this analysis are plotted in Fig. [6.](#page-31-0)

Setting 2. We consider again pure noise with τ = 1, a network with m = 4, input dimension d = 100 and sample size n = 150. We use hidden neurons with the same covariance structure as in the Setting 1.

However, we change the initialization with respect to Setting 1. First layer are initialized independently and uniformly at random. It follows that

$$C_{ij}(0,0) = \delta_{ij} \quad \forall i, j = 1, \dots, 4 \quad (\text{D.15})$$

Second layer weights are initialized according to

$$a_1(0) = -1, \quad a_2(0) = -\frac{1}{2}, \quad a_3(0) = \frac{1}{3}, \quad a_4(0) = \frac{2}{3}. \quad (\text{D.16})$$

#### D.4 Construction of the Gaussian process f g (·)

The Gaussian process f g can be constructed as follows. Define a sequence of independent Gaussian tensors J (k) ∈ (<sup>R</sup> d ) ⊗k , <sup>k</sup> ≥ <sup>1</sup>, with entries (<sup>J</sup> (k) i1,...,i<sup>k</sup> : <sup>i</sup><sup>j</sup> ≤ <sup>d</sup>) ∼iid <sup>N</sup>(0, 1). We then let

$$f^g(\mathbf{a}, \mathbf{W}) = \frac{1}{m} \sum_{i=1}^m a_i \sum_{k=0}^{\infty} c_k \sum_{i_1, \dots, i_k=1}^d J_{i_1, \dots, i_k}^{(k)} w_{i, i_1} \dots w_{i, i_k} \quad (\text{D.17})$$

It is easy to check that this stochastic process has the prescribed covariance, with

$$h(z) = \sum_{k=0}^{\infty} c_k^2 z^k, \quad (\text{D.18})$$

has long as the series above has radius of convergence larger than 1. An analogous construction holds for φ g .

## E Dynamical regimes: General preliminaries

In the next two sections, we will study the SymmDMFT equations of Section [C.4](#page-27-0) and characterize different dynamical regimes in the large network limit. From a technical viewpoint, we develop a *singular perturbation theory* of the DMFT equations as <sup>m</sup> → ∞ for fixed overparametrization ratio α = α/m.

While singular perturbation theory is a classical domain of mathematics [\[9,](#page-10-10) [29\]](#page-11-6), making this type of analysis rigorous is notoriously challenging. We will proceed heuristically as follows: (i) Hypothesize a certain asymptotic behavior of the DMFT solution in a specific time-scale; (ii) Check consistency with the DMFT equations; (iii) Check that this behavior is observed in the numerical solution of the DMFT equations.

More precisely, a specific dynamical regime is identified by a scaling of the time variable, which in our case will take the form <sup>t</sup> <sup>=</sup> <sup>t</sup>#(m) · <sup>t</sup>ˆfor a certain fixed function <sup>t</sup>#(m) and <sup>t</sup>ˆa scaled time of order one. The asymptotics of DMFT quantities in that regime takes the form (for instance)

$$\lim_{m \rightarrow \infty} \frac{1}{c_{\#}(m)} C_o\left(t_{\#}(m) \cdot \hat{t}, t_{\#}(m) \cdot \hat{s}; m, \bar{\alpha} = \frac{\alpha}{m}\right) = c_o(\hat{t}, \hat{s}; \alpha), \quad (\text{E.1})$$

where <sup>c</sup>#(m), <sup>c</sup>o(t,<sup>ˆ</sup> <sup>s</sup>ˆ; <sup>α</sup>) are two fixed functions, the limit is understood to hold at fixed t,<sup>ˆ</sup> s, α <sup>ˆ</sup> ∈ (0, ∞), and we made explicit the dependence of <sup>C</sup><sup>o</sup> on <sup>m</sup>, <sup>α</sup>. More concisely, we will often write the above formula as

$$C_o\left(t_{\#}(m) \cdot \hat{t}, t_{\#}(m) \cdot \hat{s}; m, \bar{\alpha} = \frac{\alpha}{m}\right) = c_{\#}(m)c_o(\hat{t}, \hat{s}; \alpha) + o(c_{\#}(m)), \quad (\text{E.2})$$

and we will typically use t, s instead of tˆ, sˆ for the dummy variables.

The behavior of the DMFT equations depends in a crucial way in the initialization of the second layer weights:

- In Section [F,](#page-32-1) we will consider the case of a 'lazy initialization,' i.e. we will assume a(0) = γ<sup>0</sup> √ <sup>m</sup> for some constant <sup>γ</sup><sup>0</sup> ∈ (0, ∞) independent of <sup>m</sup>.
- In Section [G,](#page-48-1) we will consider the 'mean field initialization' i.e. assume a(0) = a<sup>0</sup> to be constant and independent of m.

## F Dynamical regimes: Lazy initialization

As anticipated, in this section we study dynamical regimes under lazy initialization. In subsection [F.1,](#page-33-0) we will consider the case of pure noise data and in subsection [F.2](#page-40-0) the k-index model.

Throughout this section, we let γ(t) = a(t)/ √ m (in particular, γ(0) = γ0).

![](_page_33_Figure_0.jpeg)

Figure 8: Training of pure noise data: first dynamical regime. Rescaled correlation function <sup>m</sup>(Cd(t, s)−1) in the first dynamical regime as a function of the scaled time tm for a model initialized with a lazy scaling and fixed second layer weights. Different curves correspond to the numerical integration of the SymmDMFT equations at various values of m. They appear to converge to the scaling solution in the large m limit described by Eqs. [\(F.6\)](#page-34-0). Here α = 0.5, h˜(z) = (3/10)z + z <sup>2</sup>/2 and τ = 1.

#### F.1 Pure noise model

Under the pure noise model, we have φ = ˆφ = 0. Further, the variable v(t) is not defined and can be dropped (equivalently, we can set v(t) = 0).

We identify three dynamical regimes:

- 1. t = O(1/m): γ(t) = γ<sup>0</sup> + om(1), train error decreases, and the network approximates the null function (Section [F.1.1\)](#page-33-1).
- 2. t = Θ(1): γ(t) = γ<sup>0</sup> + om(1), first-layer weights move significantly and train error converges to a limit e∗(γ0) (Section [F.1.2\)](#page-35-0). If γ<sup>0</sup> is larger than the interpolation threshold, then train error vanishes in this regime.
- 3. t = Θ(m): This regime emerges only if γ<sup>0</sup> is smaller than the interpolation threshold. (We discuss the identification of the interpolation transition of gradient flow in Section [F.1.3.](#page-37-0)) If this is the case, γ(t) grows on the time scale t = Θ(m) until it crosses the interpolation threshold. At that point the train error vanishes (Section [F.1.4\)](#page-39-0).

Since in the first two regimes γ(t) does not change appreciably, the dynamics in these time scales is essentially equivalent to the one of a network in which second-layer weights are fixed and do not evolve by GF. In Section [F.1.1](#page-33-1) and [F.1.2](#page-35-0) we first consider this case.

We note that the pure noise model is unchanged if we rescale <sup>τ</sup> → cτ , <sup>γ</sup><sup>0</sup> → cγ0. More precisely, this results in a rescaling of the risk by c 2 and hence of time by the same factor. As a consequence quantities of interest often depend on γ, τ uniquely through their ratio γ/τ .

## F.1.1 First dynamical regime: t = O(1/m)

We first consider the case in which the (scaled) second layer weights are not updated and fixed to their initialization, i.e. γ(t) = γ0.

It is possible to check that, up to higher-order terms, the SymmDMFT equations are solved by functions of the form (the first equation holds in weak sense, i.e. after integrating against a test function)

$$R_A(t/m, s/m) = m \delta(t - s) + o_m(m) \quad C_A(t/m, s/m) = C_A^{la1}(t, s) + o_m(1) \quad (\text{F.1})$$

$$\begin{aligned}
R_o(t/m, s/m) &= \frac{1}{m} R_o^{\text{lxl}}(t, s) + o_m(1/m) & C_o(t/m, s/m) &= \frac{1}{m} C_o^{\text{lxl}}(t, s) + o_m(1/m) & (\text{F.2}) \\
R_d(t/m, s/m) &= \vartheta(t-s) + o_m(1) & C_d(t/m, s/m) &= 1 + \frac{1}{m} C_d^{\text{lxl}}(t, s) + o_m(1/m) \\
& & & & (\text{F.3}) \\
\nu(t/m) &= \nu^{\text{lxl}}(t) + o_m(1) & & & (\text{F.4})
\end{aligned}$$

$$\nu(t/m) = \nu^{\text{lzl}}(t) + o_m(1). \quad (\text{F.4})$$

where C lz1 <sup>A</sup> , C lz1 d , C lz1 o , ν lz1 and Rlz1 o are suitable functions independent of m. Here and below, we use the notation ϑ(t) = 1(t > 0).

Note that Eq. [\(F.3\)](#page-34-1) implies that on this dynamical regime the weights of the first layer change by order 1/m.

Plugging the asymptotic form in Eqs. [\(F.1\)](#page-33-2) to [\(F.4\)](#page-34-2) into the SymmDMFT equations and matching the leading orders for large m, we obtain that the functions C lz1 <sup>A</sup> , C lz1 d , C lz1 o , ν lz1 and Rlz1 <sup>o</sup> must satisfy

$$\begin{aligned}\nu^{lz1}(t) &= -\alpha\gamma_0^2 h'(1) - \alpha\gamma_0^2 h'(0)C_o^{lz1}(t, t) , \\ \partial_t R_o^{lz1}(t, t') &= -\alpha\gamma_0^2 h'(0) \left(1 + R_o^{lz1}(t, t')\right) , \\ \partial_t C_o^{lz1}(t, t') &= -\alpha\gamma_0^2 h'(0) \left(1 + C_o^{lz1}(t, t')\right) , \\ \partial_t C_d^{lz1}(t, t') &= \alpha\gamma_0^2 h'(0) \left(C_o^{lz1}(t, t) - C_o^{lz1}(t, t')\right) , \\ C_A^{lz1}(t, s) &= -\left[\tau^2 + \gamma_0^2 h(1) - \gamma_0^2 h(0)C_o^{lz1}(t, s)\right] .\end{aligned}\tag{F.5}$$

These are a set of ordinary differential equations that can be solved explicitly. We get

$$\begin{aligned} R_o^{\text{lz1}}(t, t') &= \vartheta(t - t') \left[ e^{-\alpha\gamma_0^2 h'(0)(t-t')} - 1 \right], \\ C_o^{\text{lz1}}(t, t) &= e^{-2\alpha\gamma_0^2 h'(0)t} - 1, \\ C_o^{\text{lz1}}(t, t') &= -1 + e^{-\alpha\gamma_0^2 h'(0)(t-t')} (C_o^{\text{lz1}}(t', t') + 1) \quad \text{for } t > t', \\ C_d^{\text{lz1}}(t, t') &= 1 + e^{-\alpha\gamma_0^2 h'(0)(t+t')} - \frac{1}{2} \left( e^{-2\alpha\gamma_0^2 h'(0)t} + e^{-2\alpha\gamma_0^2 h'(0)t'} \right), \quad \text{for } t > t'. \end{aligned} \quad (\text{F.6})$$

In particular, Eqs. [\(F.6\)](#page-34-0) imply

$$\begin{aligned} \lim_{t \rightarrow \infty} C_o^{lz1}(t, t) &= -1, \\ \lim_{t, t' \rightarrow \infty, t-t' \rightarrow \infty} R_o^{lz1}(t, t') &= -1. \end{aligned} \quad (\text{F.7})$$

Recalling Eq. [\(F.2\)](#page-34-3) we conclude that

$$\lim_{t \rightarrow \infty} \lim_{m \rightarrow \infty} m C_o(t/m, t/m) = -1, \quad (\text{F.8})$$

or, using the interpretation of Co,

$$\lim_{t \rightarrow \infty} \lim_{m \rightarrow \infty} \lim_{n \rightarrow \infty} m \cdot \langle \mathbf{w}_i(t/m), \mathbf{w}_j(t/m) \rangle = -1 \quad \forall i \neq j. \quad (\text{F.9})$$

In other words, at the end of this dynamical regime, the first-layer weights form a regular simplex, with center <sup>w</sup>(t/m) := <sup>m</sup><sup>−</sup><sup>1</sup> P<sup>m</sup> <sup>i</sup>=1 <sup>w</sup>i(t/m) satisfying ∥w(t/m)∥ <sup>2</sup> = om(1).

Hence, at the end of the first dynamical regime, the first-layer weights are such that the linear component of the activation function σ is removed. In other words, for t a large constant, we have

$$f^g(\cdot; \mathbf{a}(t/m), \mathbf{W}(t/m)) = \frac{\gamma_0}{\sqrt{m}} \sum_{i=1}^m \sigma_G^{\text{nl}}(\mathbf{w}_i(t/m)) + \text{err}, \quad (\text{F.10})$$

where σ nl <sup>G</sup> (w) is a Gaussian process with covariance structure given by <sup>h</sup>(z) − zh′ (0), and err is small in mean square.

Notice also that this is achieved by a O(1/ √ m) change in each of the first layer weights. Indeed, by Eq. [\(F.3\)](#page-34-1), we have

$$\lim_{n \rightarrow \infty} \|\mathbf{w}_i(0) - \mathbf{w}_i(t/m)\|^2 = 2 - 2C_d(0, t/m) = -\frac{2}{m} C_d^{n!}(0, t) + o_m(1/m). \quad (\text{F.11})$$

![](_page_35_Figure_0.jpeg)

Figure 9: Training with pure noise data under lazy initialization: second dynamical regime t = Θ(1). Left panel: First-layer weights correlation function Cd(t, 0) measuring the inner product between neurons at time 0 and time t, plotted versus t for several values of m, and compared with the large m-asymptotics C d . Right panel: training error a etr(t, γ0, m) plotted versus t for several values of m, and compared with the large-m asymptotics in this regime e lz2 tr (t, 1). Notice the two-steps decrease of the training error, corresponding to the two regimes t = O(1/m) and t = Θ(1). Inset: Same curves plotted versus tm, and compared with the asymptotic prediction e lz1 tr (· , 1) in the first dynamical regime. For both panels we use α = 0.5, h˜(z) = (3/10)z + z <sup>2</sup>/2, γ<sup>0</sup> = 1 and τ = 1.

Equations [\(F.1\)](#page-33-2) to [\(F.4\)](#page-34-2) can be used to compute the behavior of the train error in this dynamical regime:

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t/m) = e_{\text{tr}}^{lz1}(t). \quad (\text{F.12})$$

Using Eqs. [\(F.6\)](#page-34-0), we get the expression:

$$\epsilon_{\text{tr}}^{\text{lzl}}(t) = \frac{1}{2} \left[ \tau^2 + \gamma_0^2 h(1) + \gamma_0^2 h'(0) C_o^{\text{lzl}}(t, t) \right]. \quad (\text{F.13})$$

In particular, the train error at the end of this dynamical regime is

$$\lim_{t \rightarrow \infty} \lim_{m \rightarrow \infty} e_{\text{tr}}(t/m) = \lim_{t \rightarrow \infty} e_{\text{tr}}^{l_{\text{tr}}}(t) = \frac{1}{2} [\tau^2 + \gamma_0^2 h(1) - \gamma_0^2 h'(0)] . \quad (\text{F.14})$$

This is in agreement with [\(F.10\)](#page-34-4). Indeed, note that

$$\widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) = \frac{1}{2n} \|\varepsilon\|^2 - \frac{1}{n} \langle \varepsilon, \mathbf{f}^g(\mathbf{a}, \mathbf{W}) \rangle + \frac{1}{2n} \|\mathbf{f}^g(\mathbf{a}, \mathbf{W})\|^2. \quad (\text{F.15})$$

Training in this timescale attempts to minimize ∥<sup>f</sup> g (a,W)∥ <sup>2</sup> without fitting the noise.

This picture is confirmed by the fact that Eqs. [\(F.6\)](#page-34-0) depend on h only through h ′ (0). This means that the dynamics on timescales of order 1/m is controlled by the linear part of the covariance structure of the hidden layer.

In Fig[.8](#page-33-3) we test the correctness of the asymtotic ansatz of Eqs. [\(F.1\)](#page-33-2) to [\(F.4\)](#page-34-2). Namely, we compare the results of numerical integration of the SymmDMFT equations for various values of m, with the prediction of Eqs. [\(F.6\)](#page-34-0). The match is excellent.

So far we assumed that second-layer weights are not optimized and γ(t) = γ0. What happens if drop this constraint? It can be checked that the form given in Eqs. [\(F.1\)](#page-33-2)-[\(F.4\)](#page-34-2) still solves the SymmDMFT equations when a(t) is allowed to evolve, and γ(t/m) = γ<sup>0</sup> + om(1) for all fixed <sup>t</sup> ∈ (0, ∞). In other words, second layer weights do not change significantly during this dynamical regime.

## F.1.2 Second dynamical regime: t = Θ(1)

The second dynamical regime arises when t = Θ(1). Recall from the previous subsection that, for t = om(1), the train error remains close (for large m) to the plateau characterized at the end of the first dynamical regime, see Eq. [\(F.14\)](#page-35-1). When t is of order one, the first layer weights start changing by an amount of order one as well, and the model starts to fit the noise.

As before, we begin by considering the simplified setting in which γ(t) = γ<sup>0</sup> is fixed and not optimized by GF.

We claim that the SymmDMFT equations are solved by the following ansatz, up to lower order terms as <sup>m</sup> → ∞:

$$\nu(t) = \nu^{l2}(t) + o_m(1), \quad (\text{F.16})$$

$$C_d(t, t') = C_d^{ly2}(t, t') + o_m(1), \quad (\text{F.17})$$

$$R_d(t, t') = R_d^{l2}(t, t') + o_m(1), \quad (\text{F.18})$$

$$C_o(t, t') = \frac{1}{m} C_o^{lo2}(t, t') + o_m(1/m) = -\frac{1}{m} C_d^{lo2}(t, t') + o_m(1/m), \quad (\text{F.19})$$

$$R_o(t, t') = \frac{1}{m} R_o^{l2}(t, t') + o_m(1/m) = -\frac{1}{m} R_d^{l2}(t, t') + o_m(1/m). \quad (\text{F.20})$$

Here C lz2 d , Rlz2 d , C lz2 o , Rlz2 o and ν lz2 are certain functions independent of m. Equations [\(F.19\)](#page-36-0), [\(F.20\)](#page-36-1) state in particular that C lz2 o (t, t′ ) = −<sup>C</sup> lz2 d (t, t′ ) and Rlz2 o (t, t′ ) = −Rlz2 d (t, t′ ), and the therefore we are left with the task of determining C lz2 d (t, t′ ), Rlz2 d (t, t′ ). By substituting Eqs. [\(F.16\)](#page-36-2) to [\(F.20\)](#page-36-1) into the SymmDMFT equations and matching leading order terms, we get a set of two integral-differential equations for C lz2 d (t, t′ ), Rlz2 d (t, t′ ), which we next state.

We first define

$$\begin{aligned}\Sigma_R^{lx2}(t, s) &:= \gamma_0^2 (h'(C_d^{lx2}(t, s)) - h'(0)) R_d^{lx2}(t, s), \\\Sigma_C^{lx2}(t, s) &:= \tau^2 + \gamma_0^2 h(C_d^{lx2}(t, s)) - \gamma_0^2 h'(0) C_d^{lx2}(t, s),\end{aligned}\tag{F21}$$

then we define Rlz2 <sup>A</sup> and C lz2 <sup>A</sup> as the solution of

$$\begin{aligned} \delta(t - t') &= \int_{t'}^t [\delta(t - s) + \Sigma_R^{lx2}(t, s)] R_A^{lx2}(s, t') ds , \\ 0 &= \int_0^t [\delta(t - s) + \Sigma_R^{lx2}(t, s)] C_A^{lx2}(s, t') ds + \int_0^{t'} \Sigma_C^{lx2}(t, s) R_A^{lx2}(t', s) ds . \end{aligned} \quad (\text{F.22})$$

We next define the asymptotic form for the memory kernels

$$\begin{aligned} M_R^{lz2}(t, s) &:= \alpha \left[ R_A^{lz2}(t, s) \tilde{h}'(C_d^{lz2}(t, s)) + C_A^{lz2}(t, s) \tilde{h}''(C_d^{lz2}(t, s)) R_d^{lz2}(t, s) \right], \\ M_C^{lz}(t, s) &:= \alpha \tilde{h}'(C_d^{lz2}(t, s)) C_A^{lz2}(t, s). \end{aligned} \quad (\text{F.23})$$

and we have defined

$$\tilde{h}(z) := h(z) - h'(0)z . \quad (\text{F.24})$$

The equations for ν lz2 , C lz2 d and Rlz2 d are then given by

$$v^{l2}(t) = - \int_0^t [M_R^{l2}(t,s)C_d^{l2}(t,s) + M_C^{l2}(t,s)R_d^{l2}(t,s)] \, ds, \quad (\text{F.25})$$

$$\partial_t R_d^{l2}(t, t') = \delta(t - t') - \nu^{l2}(t) R_d^{l2}(t, t') - \int_{t'}^t M_R^{l2}(t, s) R_d^{l2}(s, t') ds, \quad (\text{F.26})$$

$$\partial_t C_d^{lx2}(t, t') = -\nu^{lx2}(t) C_d^{lx2}(t, t') - \int_0^t M_R^{lx2}(t, s) C_d^{lx2}(t', s) \, ds - \int_0^{t'} M_C^{lx2}(t, s) R_d^{lx2}(t', s) \, ds . \quad (\text{F.27})$$

(As before, in the second and last equation, it is understood that <sup>t</sup> ≥ <sup>t</sup> ′ , and the last equation is understood to hold in weak sense.)

Given the constraints on Cd, Rd, we have the following constraints on C lz2 d , Rlz2 d ,

$$C_d^{lz2}(t, t) = 1, \quad (\text{F.28})$$

$$C_d^{lz2}(t, s) = C_d^{lz2}(s, t), \quad (\text{F.29})$$

$$R_d^{l2}(t, s) = 0 \quad \forall t \leq s. \quad (\text{F.30})$$

![](_page_37_Figure_0.jpeg)

Figure 10: Training with pure noise data under lazy initialization: algorithmic interpolation threshold. Left Panel. We plot the train error as a function of time for different values of α at γ<sup>0</sup> = 10/7. The train error has an exponential decay to zero for α below the interpolation threshold. Right Panel. We plot the time for GF to converge to near-zero training error as a function of α, for various values of γ0, as computed using the theory of Section [F.1.2.](#page-35-0) The divergence of trel signals the phase transition for GF interpolation α ∗ GF. Inset: τrel versus α in linear scale. Main panel: trel versus α ∗ GF − <sup>α</sup> (with the fitted value of <sup>α</sup> ∗ GF). Here we use h(z) = (3/10)z + z <sup>2</sup>/2.

The evolution of the the train error in this second dynamical regime is given by

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t) = e_{\text{tr}}^{l2}(t, \gamma_0), \quad (\text{F.31})$$

$$e_{\text{tr}}^{\text{lz2}}(t, \gamma_0) = -\frac{1}{2}C_A^{\text{lz2}}(t, t), \quad (\text{F.32})$$

where we have made explicit the dependence on the initialization of second-layer weights γ0.

Note that Eq. [\(F.22\)](#page-36-4) implies C lz2 <sup>A</sup> (0, 0) = −<sup>Σ</sup> lz2 <sup>C</sup> (0, 0), and Eq. [\(F.21\)](#page-36-5) yields Σ lz2 <sup>C</sup> (0, 0) = τ <sup>2</sup> + γ 2 <sup>0</sup>h(1) − <sup>γ</sup> 2 0h ′ (0). Therefore

$$\lim_{t \rightarrow 0} e_{\text{tr}}^{l2}(t, \gamma_0) = \frac{1}{2} \left[ \tau^2 + \gamma_0^2 \tilde{h}(1) \right] = \lim_{t \rightarrow \infty} e_{\text{tr}}^{l1}(t, \gamma_0). \quad (\text{F.33})$$

In other words, this second dynamical regime captures the decrease of the training error which starts at the plateau reached in the first regime, cf. Eq. [\(F.14\)](#page-35-1). which coincides with the long time extrapolation of the first dynamical regime.

This second dynamical regime is fully non-linear and depends on the entire covariance function h˜. Further, the first order weights move by an amount ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1), as follows from the fact that C lz2 d (t, 0) < 1 strictly.

In order to confirm the ansatz [\(F.16\)](#page-36-2) to [\(F.20\)](#page-36-1), we compared the solution of the full SymmDMFT equations, with the solution of the asymptotic equations [\(F.25\)](#page-36-6), [\(F.27\)](#page-36-7). An example of such a comparison is presented in Fig. [9:](#page-35-2) the agreement is excellent.

The treatment above assumed the constraint γ(t) = γ0. However, as in the first dynamical regime, if we let second layer weights evolve, they do not change appreciably. Namely, the asymptotic form given in Eqs. [\(F.16\)](#page-36-2) to [\(F.20\)](#page-36-1) still solves the SymmDMFT equations when a(t) is allowed to evolve. We have γ(t) = γ<sup>0</sup> + om(1) on this timescale.

## F.1.3 The algorithmic interpolation transition

For the discussion in this section, we denote by etr(t, γ0, m, α) the train error as a function of t, where we emphasized the dependence on the initial condition γ0, on the number of neurons m, and on the overparametrization ratio α. We further assume that second layer weigths are not evolved and therefore γ(t) = γ<sup>0</sup> for all t. We define the asymptotic train error achieved by GF as

$$e_{\text{tr},\infty}(\gamma_0, m, \alpha) := \lim_{t \rightarrow \infty} e_{\text{tr}}(t, \gamma_0, m, \alpha) \quad (\text{F.34})$$

$$= \lim_{t \rightarrow \infty} \lim_{n \rightarrow \infty} \hat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}(t)). \quad (\text{F.35})$$

Again, in this definition a<sup>i</sup> = γ<sup>0</sup> √ m is kept fixed and does not evolve with time.

Notice that it is in principle possible that limn→∞ <sup>R</sup>b <sup>n</sup>(a,W(tn)) is strictly smaller than etr,<sup>∞</sup>(γ0, m, α) if we let t<sup>n</sup> diverge with n at sufficiently fast rate. However, based on results on related models in spin-glass theory we expect this not to be the case as long as t<sup>n</sup> is polynomial in <sup>n</sup>. Explicitly, we expect that, for any sequence <sup>t</sup><sup>n</sup> → ∞

$$t_n \leq n^C \Rightarrow \lim_{n \rightarrow \infty} \widehat{\mathcal{H}}_n(\mathbf{a}, \mathbf{W}(t_n)) = e_{\text{tr}, \infty}(\gamma_0, m, \alpha). \quad (\text{F.36})$$

Using the reduced equations for t = Θ(1) timescale, i.e. Eqs. [\(F.25\)](#page-36-6) to [\(F.27\)](#page-36-7), we can also define

$$\begin{aligned} e_{\text{tr},\infty}^{l2}(\gamma_0, \alpha) &:= \lim_{t \rightarrow \infty} e_{\text{tr}}^{l2}(t, \gamma_0, \alpha) \\ &= \lim_{t \rightarrow \infty} \lim_{m \rightarrow \infty} \lim_{n \rightarrow \infty} \widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}(t)). \end{aligned} \quad (\text{F.37})$$

A natural question is whether the large m limit of etr,<sup>∞</sup>(γ0, m, α) coincides with e lz2 tr,<sup>∞</sup>(γ0, α). This amounts to asking whether there exists dynamical regime with timescale t(m) diverging with m at which etr(t(m), γ0, m, α) starts diverging significantly from the value at the end of the second dynamical regime namely e lz2 tr,<sup>∞</sup>(γ0, α). If e lz2 tr,<sup>∞</sup>(γ0, α) = 0 then of course limm→∞ etr(t(m), γ0, m, α) = 0 as well.

If however e lz2 tr,<sup>∞</sup>(γ0, α) > 0, then the answer depends upon whether the second layer weights are evolved with GF:

- In the constrained setting in which second-layer weights do not evolve, we observe (from numerical solutions of SymmDMFT) that

$$\lim_{m \rightarrow \infty} e_{\text{tr}, \infty}(\gamma_0, m, \alpha) = e_{\text{tr}, \infty}^{|z_2|}(\gamma_0, \alpha). \quad (\text{F.38})$$

- In the next section we will see that if γ(t) evolves with GF then the train error achieved on a diverging timescale t = Θ(m) is strictly smaller than e lz2 tr,<sup>∞</sup>(γ0, α) and vanishes for large enough t.

Note that etr,<sup>∞</sup>(γ0, m, α) and e lz2 tr,<sup>∞</sup>(γ0, α) also depend on the noise variance τ 2 . However, because of the invariance under rescaling discussed at the beginning of this section (adding τ as an argument):

$$e_{\text{tr},\infty}^{lz_2}(\gamma_0, \alpha, \tau^2) = \tau^2 \cdot e_{\text{tr},\infty}^{lz_2}(\gamma_0/\tau, \alpha, \tau^2 = 1), \quad (\text{F.39})$$

and similarly for etr,<sup>∞</sup>(γ0, m, α). Because of this relation, we can think that τ 2 is fixed throughout, e.g. τ <sup>2</sup> = 1.

We expect etr,<sup>∞</sup>(γ0, m, α), e lz2 tr,<sup>∞</sup>(γ0, α) to be non-increasing in γ0, and define the thresholds

$$\begin{aligned}\gamma_{\text{GF}}(\alpha, m) &:= \inf \{\gamma_0 : e_{\text{tr}, \infty}(\gamma_0, m, \alpha) = 0\}, & (\text{F.40}) \\ \gamma_{\text{GF}}^*(\alpha) &:= \inf \{\gamma_0 : e_{\text{tr}, \infty}^{\text{lz}}(\gamma_0, \alpha) = 0\}. & (\text{F.41})\end{aligned}$$

$$\begin{aligned} \text{gr}_{\text{F}}(\alpha, m) &:= \inf \{\gamma_0 : e_{\text{tr}, \infty}(\gamma_0, m, \alpha) = 0\}, & (\text{F.40}) \\ \gamma_{\text{Gr}}^*(\alpha) &:= \inf \{\gamma_0 : e_{\text{tr}, \infty}^{\text{l2}}(\gamma_0, \alpha) = 0\}. & (\text{F.41}) \end{aligned}$$

(These definitions need to be modified if <sup>γ</sup><sup>0</sup> 7→ <sup>e</sup>tr,<sup>∞</sup>(γ0, m, α) is non-monotone.)

Of course, Eq. [\(F.38\)](#page-38-0) implies

$$\lim_{m \rightarrow \infty} \gamma_{\text{GF}}(\alpha, m) = \gamma_{\text{GF}}^*(\alpha). \quad (\text{F.42})$$

The numerical solution of the SymmDMFT equations imply that the curve γ ∗ GF(α) is monotone increasing with α, as also suggested by the Gaussian complexity bound (see Section 2.2 in the main text). Hence we can invert it to get a threshold α ∗ GF(γ0): the two descriptions are equivalent.

In order to determine α ∗ GF(γ0), we adopt a procedure already implemented in [\[31\]](#page-11-11) for a simpler model. The procedure is based on the observation (from numerical solutions) that when e lz2 tr,<sup>∞</sup>(γ0, α) = 0, e lz2 tr (t, γ0, α) = exp(−t/trel(α; <sup>ε</sup>) + <sup>o</sup>(t)) for some <sup>t</sup>rel(α) <sup>&</sup>gt; <sup>0</sup> which diverges as <sup>α</sup> ↑ <sup>α</sup>GF.

- 1. Define a grid of values of <sup>α</sup>, <sup>A</sup><sup>0</sup> <sup>=</sup> {<sup>α</sup>1, α2, . . . , αK}, which are expected to be smaller than α ∗ GF(γ0).

![](_page_39_Figure_0.jpeg)

Figure 11: Training with pure noise data under lazy initialization: second layer weights in the third dynamical regime. Evolution of the (rescaled) weights of the second layer as a function of t/m. Here τ = 2.5 and γ<sup>0</sup> = 1, α = 0.3, and covariance structure for the neurons given by h(z) = (9/10)z + z <sup>2</sup>/2.

- 2. For each value <sup>α</sup> ∈ <sup>A</sup>0, integrate numerically the reduced equations [\(F.25\)](#page-36-6) to [\(F.27\)](#page-36-7). Verify that e lz2 tr (t, γ0, α) appear to converge to <sup>0</sup> with <sup>t</sup> → ∞. Let <sup>A</sup> ⊆ <sup>A</sup><sup>0</sup> be the subset of values for which this happens.
- 3. For each <sup>α</sup> ∈ <sup>A</sup>, define <sup>t</sup>rel(α<sup>i</sup> ; <sup>ε</sup>) := inf{<sup>t</sup> : <sup>e</sup> tr (t, γ0, αi) < ε · <sup>τ</sup> <sup>2</sup>} where <sup>ε</sup> is a small threshold value (we use ε = 10<sup>−</sup><sup>7</sup> ).
- 4. Estimate parameters α ∗ GF(γ0), c, ν by fitting the relation trel(α<sup>i</sup> ; <sup>ε</sup>) ∼ <sup>c</sup>(<sup>α</sup> ∗ GF − <sup>α</sup>i) −ν .

Figure [10](#page-37-1) illustrates the calculation of α ∗ GF(γ0) for three values of γ0. In the inset we plot trel for three values of γ<sup>0</sup> as a function of α. In the main panel, we demonstrate the divergence of trel when (α ∗ GF − <sup>α</sup>) vanishes. In practice, we observe <sup>ν</sup> = 2 fit well the data across a variety of settings, suggesting this is the universal exponent for the divergence of trel.

#### F.1.4 Third dynamical regime: t = Θ(m)

In the first two dynamical regimes, the large-m behavior did not depend on whether we would let second layer evolve with GF or we kept them fixed, i.e. γ(t) = γ0.

In contrast, the behavior on timescales diverging with m depends significantly on the dynamics of second-layer weights.

- If second layer weights are fixed, no significant further evolution takes place. In particular, the training error does not decrease significantly below the value reached at the end of the second dynamical regime, i.e. e ℓ tr,<sup>∞</sup>(γ0, α). This is stated formally in Eq. [\(F.38\)](#page-38-0).
- If second layer weights evolve according to GF, then the dynamics on time-scales diverging with m can be non-trivial and depends on the second-layer weights initialization γ0. If γ<sup>0</sup> > γ<sup>∗</sup> GF(α), then GR reaches vanishing training error during the second dynamical regime, and no further evolution takes place. However, if γ<sup>0</sup> < γ<sup>∗</sup> GF(α), second layer weights start evolving when t = Θ(m), thus giving rise to a third dynamical regime. This is the object of the present subsection.

In Fig. [11,](#page-39-1) left frame, we plot the rescaled second layer weights γ(t) (as predicted by numerical integration of the SymmDMFT equation) as a function of time for several values of m. Here, obviously, we do not constrain γ(t) = γ(0).

We observe that γ(t) changes only when t = Θ(m). Indeed, when plotted against t/m, curves obtained for different values of m collapse onto each other. This suggests that, for t = o(m) γ(t) = γ(0) + om(1) (recall that γ(0) = γ<sup>0</sup> by definition). Further, the curve collapse suggests that, for any fixed <sup>t</sup><sup>ˆ</sup>∈ (0, ∞):

$$\lim_{m \rightarrow \infty} \gamma(\hat{t} m, \gamma_0) = \gamma^{|\lambda^3|}(\hat{t}, \gamma_0), \quad (\text{F.43})$$

where we have made explicit the dependence on γ0. Of course, the case γ<sup>0</sup> > γ<sup>∗</sup> GF(α) fits in this framework with γ lz3(z, γ0) = γ<sup>0</sup> identically.

We next consider the evolution of the train error. In Fig. [12,](#page-41-0) left frame, we plot the train error (again, as predicted by numerical integration of the SymmDMFT equation) as a function of time for several values of m.

Again, when plotted as a function of t/m, curves for different values of m reach a plateau, and collapse below the plateau. This suggests the following limit behavior, which is consistent with Eq. [\(F.43\)](#page-40-1)

$$\lim_{m \rightarrow \infty} \tilde{e}_{\text{tr}}(\hat{t} m, \gamma_0, m) = e_{\text{tr}}^{lz3}(\hat{t}, \gamma_0) . \quad (\text{F.44})$$

(Here we use e˜tr(t m, γ ˆ <sup>0</sup>) to denote the train error when second-layer weights evolve, in contrast with etr(t m, γ ˆ <sup>0</sup>) which we used for the setting in which second-layer weights are constrained.)

Matching the present dynamical regime (t = Θ(m)) with previous one (t = Θ(1), cf. Section [F.1.2\)](#page-35-0), implies that

$$\lim_{\hat{t} \rightarrow 0^+} e_{\text{tr}}^{lz3}(\hat{t}, \gamma_0) = \lim_{t \rightarrow \infty} e_{\text{tr}}^{lz2}(t, \gamma_0) = e_{\text{tr}, \infty}^{lz2}(\gamma_0) \quad (\text{F.45})$$

In other words, the function e lz3 tr describes the decrease of the train error below the level e lz2 tr,<sup>∞</sup>(γ0) achieved during the second dynamical regime.

In order to characterize the scaling function e lz3 tr , in Fig. [12,](#page-41-0) right frame, we plot parametrically the the train error for different values of m as a function of the second layer weights γ(t). We also plot the curve (γ, elz2 tr,<sup>∞</sup>(γ)). This plot is consistent with the following behavior as <sup>m</sup> → ∞. In a first regimes (corresponding to <sup>t</sup> <sup>=</sup> <sup>o</sup>(m)) the train error has a drop that becomes vertical in the <sup>m</sup> → ∞ limit, implying that γ(t) does not evolve while the train error decreases until it reaches e ℓ tr,<sup>∞</sup>(γ0). In the last regime (corresponding to t = Θ(m)), γ(t) increases together with the decrease of the train error e ℓ tr(t, γ0). Remarkably, they follow the curve (γ, elz2 tr,<sup>∞</sup>(γ)).

In order to describe the last regime, we point out that <sup>t</sup> 7→ <sup>γ</sup> lz3(t) is monotone increasing. Therefore we can re-parametrize time by the value of the second layer weights. Namely, define γ˜ −1 the inverse function, so that

$$\hat{t} = \tilde{\gamma}^{-1}(\gamma^{\text{lz}}(\hat{t}, \gamma_0), \gamma_0). \quad (\text{F.46})$$

Using this reparametrization of time, the behavior in Fig. [12](#page-41-0) can be formalized as

$$\lim_{t, m \rightarrow \infty: \gamma(t, \gamma_0, m) = \tilde{\gamma}} e_{\text{tr}}^{l3}(t, \gamma_0, m) = e_{\text{tr}}^{l3}(\tilde{\gamma}^{-1}(\tilde{\gamma}, \gamma_0), \gamma_0) =: \varepsilon(\tilde{\gamma}, \gamma_0) . \quad (\text{F.47})$$

The collapse on finite m curves in Fig. [12,](#page-41-0) right frame, onto the curve (γ, elz2 tr,<sup>∞</sup>(γ)) suggests that

$$\gamma > \gamma_0 \quad \Rightarrow \quad \varepsilon(\tilde{\gamma}, \gamma_0) = e_{\text{tr},\infty}^{|\gamma_0|^2}(\gamma). \quad (\text{F.48})$$

In other words, the dynamics on timescales of order m is *adiabatic*: at each increase of γ(t) on timescales of order m, the train error relaxes to the the value it would have had if the second layer weights would have been fixed in time at the corresponding value of γ.

A remarkable consequence of Eq. [\(F.48\)](#page-40-2) is that that

$$\lim_{\hat{t} \rightarrow \infty} \gamma^{lz}(\hat{t}) = \lim_{m \rightarrow \infty} \gamma_{\text{GF}}(\alpha, m) = \gamma_{\text{GF}}^*(\alpha). \quad (\text{F.49})$$

In words, in the large network limit, the norm of second-layer weights at the end of training is asymptotically the minimum norm that allows for interpolation.

## F.2 Multi-index model

In this section we generalize the computations of Section [F.1](#page-33-0) to the case in which the dataset has a structure produced via a k-index model. The weights of the second layer are set to a(t) = γ(t) √ m and evolve with GF. The initialization scale γ(0) = γ<sup>0</sup> is fixed and independent of m.

![](_page_41_Figure_0.jpeg)

Figure 12: Training with pure noise data under lazy initialization: third dynamical regime. Left frame: Train error on timescales of order m. Right frame: GF trajectories in the plane γ (second layer weights) — etr (train error). Black dots represent pairs (γ, elz2 tr,<sup>∞</sup>(γ)), where e lz2 tr,<sup>∞</sup>(γ) is the train error achieved at the end of the first dynamical regime, cf. Section [F.1.3.](#page-37-0) The data has been produced from the same model as in Fig. [11.](#page-39-1)

![](_page_41_Figure_2.jpeg)

Figure 13: SymmDMFT predictions and large network scaling for lazy training in a single index model. Left: Projection v(t) of the first layer weights onto the latent direction on timescales of the order <sup>1</sup>/m. The result for <sup>m</sup> → ∞, <sup>v</sup> lz1, has been obtained by integrating analytically Eq. [\(F.55\)](#page-42-0). Right: The behavior of Cd(t, 0) on timescales t = Θ(1), compared with the scaling theory for <sup>m</sup> → ∞, namely <sup>C</sup> d . In both cases with h(z) = ˆφ(z) = (9/10)z + z <sup>2</sup>/2, τ = 0.3 and α = 0.3, γ<sup>0</sup> = 1.

- 1. <sup>t</sup> <sup>=</sup> <sup>O</sup>(1/m): <sup>γ</sup>(t) = <sup>γ</sup><sup>0</sup> <sup>+</sup> <sup>o</sup>m(1), ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1/ √ m). On this scale the network only learns a linear approximation of the target. Test and train error remain close to each other (Section [F.2.1\)](#page-41-1).
- 2. <sup>t</sup> = Θ(1): <sup>γ</sup>(t) = <sup>γ</sup><sup>0</sup> <sup>+</sup> <sup>o</sup>m(1), ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1). Test error does not change but train error decreases significantly (Section [F.2.2\)](#page-42-1).
- 3. t = Θ(m): This regime only emerges if γ<sup>0</sup> is below a certain interpolation threshold, i.e. γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ). In this regime γ(t) grows until the threshold, and train error decreases to 0 while test error decreases to 0 (Section [F.2.5\)](#page-46-0).

## F.2.1 First dynamical regime: t = O(1/m)

On this timescale, the SymmDMFT equations are solved, up to higher order terms, by the following ansatz:

$$C_d(t/m, s/m) = 1/o_m(1), \quad R_d(t/m, s/m) = \vartheta(t-s) + o_m(1), \quad (\text{F.50})$$

$$C_o(t/m, s/m) = \frac{1}{m} C_o^{\text{tl1}}(t, s) + o_m(m^{-1}), \quad R_o(t/m, s/m) = \frac{1}{m} R_o^{\text{tl1}}(t, s) + o_m(m^{-1}), \quad (\text{F.51})$$

$$\begin{aligned} \frac{1}{m} R_A(t/m, s/m) &= \delta(t - s) + o_m(1) , & C_A(t/m, s/m) &= -\Sigma_C^{la1}(t, s) + o_m(1) , \quad (\text{F.52}) \\ a(t/m)\sqrt{m} &= \gamma_0 + o_m(1) , & v(t/m) &= \frac{1}{\sqrt{m}} \mathbf{v}^{la1}(t) + o_m(m^{-1/2}) , \\ & & & (\text{F.53}) \end{aligned}$$

with

$$\Sigma_C^{\text{lzl}}(t, s) = \tau^2 + \|\varphi\|^2 - \gamma_0 \langle \nabla \hat{\varphi}(\mathbf{0}), \mathbf{v}^{\text{lzl}}(t) \rangle - \gamma_0 \langle \nabla \hat{\varphi}(\mathbf{0}), \mathbf{v}^{\text{lzl}}(s) \rangle + \gamma_0^2 (h(1) + h'(0)C_o^{\text{lzl}}(t, s)). \quad (\text{F.54})$$
In particular, Eq. (F.50) implies  $\|\mathbf{w}_i(0) - \mathbf{w}_i(t/m)\| = o_m(1)$ : weights of the first layer change by

In particular, Eq. [\(F.50\)](#page-41-2) implies ∥<sup>w</sup>i(0) − <sup>w</sup>i(t/m)∥ <sup>=</sup> <sup>o</sup>m(1): weights of the first layer change by small amount.

The scaling functions defined in Eqs. [\(F.50\)](#page-41-2)-[\(F.53\)](#page-42-2) satisfy a set of equations that can be derived directly from the SymmDMFT equations:

$$\begin{aligned} \partial_t \mathbf{v}^{lz1}(t) &= \alpha\gamma_0 \nabla \hat{\varphi}(\mathbf{0}) - \alpha\gamma_0^2 h'(0) \mathbf{v}^{lz1}(t), \\ \partial_t C_o^{lz1}(t, t') &= \alpha\gamma_0 \langle \nabla \hat{\varphi}'(\mathbf{0}), \mathbf{v}^{lz1}(t') \rangle - \alpha\gamma_0^2 h'(0) (1 + C_o^{lz1}(t, t')) \\ \partial_t R_o^{lz1}(t, s) &= -\alpha\hat{\gamma}_0^2 h'(0) (1 + R_o^{lz1}(t, s)). \end{aligned} \quad (\text{F.55})$$

Note that

$$\frac{dC_o^{\text{lzl}}(t, t)}{dt} = 2 \lim_{t' \rightarrow t^-} \partial_t C_o^{\text{lzl}}(t, t'). \quad (\text{F.56})$$

The solution of Eqs. [\(F.55\)](#page-42-0) implies that

$$\mathbf{v}_\infty^{\text{lxl}} := \lim_{t \rightarrow \infty} \mathbf{v}^{\text{lxl}}(t) = \frac{\nabla \hat{\varphi}(\mathbf{0})}{\gamma_0 h'(0)}, \quad (\text{F.57})$$

$$\lim_{t \rightarrow \infty} C_o^{\text{lxl}}(t, t) = -(1 - \|\mathbf{v}_\infty^{\text{lxl}}\|^2).$$

Furthermore, on this timescale, the train and test error coincide and are given by

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t/m) = \lim_{m \rightarrow \infty} e_{\text{ts}}(t/m) = \frac{1}{2} \Sigma_C^{lz1}(t, t) . \quad (\text{F.58})$$

The corresponding asymptotic value is given by

$$\lim_{t \rightarrow \infty} \lim_{m \rightarrow \infty} e_{\text{ts}}(t/m) = e_{\text{ts}, \infty}^{l_1} = \frac{1}{2} \left( \tau^2 + \|\varphi\|^2 - \frac{1}{h'_s(0)} \|\nabla \hat{\varphi}(\mathbf{0})\|^2 + \gamma_0^2 \tilde{h}(1) \right) \quad (\text{F.59})$$

where

$$\tilde{h}(z) = h(z) - h'(0) . \quad (\text{F.60})$$

The interpretation of this dynamical regime is analogous to the one of the same regime in the purenoise setting, as confirmed by Eq. [\(F.59\)](#page-42-3) : the network learns the linear component of the data distribution.

In the left panel of Fig. [13](#page-41-3) we test the scaling theory in this dynamical regime, as given by Eqs. [\(F.50\)](#page-41-2) to [\(F.53\)](#page-42-2). We plot the solution of the SymmDMFT equations, versus tm, for increasing values of m: the curve collapse well on their conjectured <sup>m</sup> → ∞ limit.

## F.2.2 Second dynamical regime: t = Θ(1)

We next consider t = Θ(1). One can show that the SymmDMFT equations are solved, up to higher order terms as <sup>m</sup> → ∞, by the following ansatz

$$\begin{aligned} C_d(t, s) &= C_d^{\text{tl2}}(t, s) + o_m(1), & R_d(t, s) &= R_d^{\text{tl2}}(t, s) + o_m(1), \\ C_o(t, s) &= \frac{1}{m} C_o^{\text{tl2}}(t, s) + o_m(m^{-1}), & R_o(t, s) &= \frac{1}{m} R_o^{\text{tl2}}(t, s) + o_m(m^{-1}), & (\text{F.61}) \\ \mathbf{v}(t) &= \frac{1}{\sqrt{m}} \mathbf{v}_\infty^{\text{tl1}} + o_m(m^{-1/2}), & \nu(t) &= \nu^{\text{tl2}}(t) + o_m(1), \end{aligned}$$

$$C_o^{lx2}(t, s) = -C_d^{lx2}(t, s) + \|\mathbf{v}_\infty^{lx2}\|^2, \quad R_o^{lx2}(t, s) = -R_d^{lx2}(t, s). \quad (\text{F.62})$$

In other words, on this time scale first layer weights move by order one ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ = Θ(1), but in a linear subspace that is orthogonal to the latent space. Second layer weights do not move appreciably. As a consequence, no additional learning takes place in this regime, but the model begins to overfit the data.

Note that the above scaling form is compatible with the long time limit of the previous dynamical regime.

In order to define the equations for the functions on the right-hand side of Eq. [\(F.61\)](#page-42-4) we define Rlz2 A and C lz2 <sup>A</sup> to be the solution of

$$\begin{aligned} \delta(t - t') &= \int_{t'}^t [\delta(t - s) + \Sigma_R^{lx2}(t, s)] R_A^{lx2}(s, t') ds, \\ 0 &= \int_0^t [\delta(t - s) + \Sigma_R^{lx2}(t, s)] C_A^{lx2}(s, t') ds + \int_0^{t'} \Sigma_C^{lx2}(t, s) R_A^{lx2}(t', s) ds, \end{aligned} \quad (\text{F.63})$$

where

$$\begin{aligned}\Sigma_R^{lx2}(t, s) &= \gamma_0^2 \left( h'(C_d^{lx2}(t, s)) - h'(0) \right) R_d^{lx2}(t, s) , \\ \Sigma_{C'}^{lx2}(t, s) &= \tau^2 + \|\varphi\|^2 - 2\gamma_0 \langle \nabla \hat{\varphi}(\mathbf{0}), \mathbf{v}_{\infty}^{lx} \rangle + \gamma_0^2 \left( h(C_d^{lx2}(t, s)) + h'(0)C_o^{lx2}(t, s) \right) .\end{aligned}\tag{F.64}$$

Define the following memory kernels

$$\begin{aligned} M_{R,d}^{l2}(t, s) &= \alpha\gamma_0^2 [R_A^{l2}(t, s)h'(C_d^{l2}(t, s)) + C_A^{l2}(t, s)h''(C_d^{l2}(t, s))R_d^{l2}(t, s)] , \\ M_{R,o}^{l2}(t, s) &= \alpha\gamma_0^2 h'(0)R_A^{l2}(t, s) , \\ M_{C,d}^{l2}(t, s) &= \alpha\gamma_0^2 h'(C_d^{l2}(t, s))C_A^{l2}(t, s) , \\ M_{C,o}^{l2}(t, s) &= \alpha\gamma_0^2 h'(0)C_A^{l2}(t, s) . \end{aligned} \tag{F.65}$$

Substituting the ansatz [\(F.61\)](#page-42-4) into the SymmDMFT equations, and using Eqs. [\(F.62\)](#page-42-5), we obtain the following equations for C lz2 d (t, t′ ), Rlz2 d (t, t′ ), ν lz2(t)

$$\begin{aligned} \partial_t C_d^{lz2}(t, t') &= -\nu^{lz2}(t) C_d^{lz2}(t, t') + \alpha \gamma_0 \langle \nabla \dot{\varphi}'(\mathbf{0}), \mathbf{v}_\infty^{lz1} \rangle \int_0^t R_A^{lz2}(t, s) ds \\ &\quad - \int_0^t ds [M_{R,d}^{lz2}(t, s) C_d^{lz2}(t', s) + M_{R,o}^{lz2}(t, s) C_o^{lz2}(t', s)] ds \\ &\quad - \int_0^{t'} [M_{C,d}^{lz2}(t, s) R_d^{lz2}(t', s) + M_{C,o}^{lz2}(t, s) R_o^{lz2}(t', s)] ds, \\ \partial_t R_d^{lz2}(t, t') &= -\nu^{lz2}(t) R_d^{lz2}(t, t') + \delta(t - t') \\ &\quad - \int_{t'}^t [M_{R,d}^{lz2}(t, s) R_d^{lz2}(s, t') + M_{R,o}^{lz2}(t, s) R_o^{lz2}(s, t')] ds, \\ \nu^{lz2}(t) &= \alpha \gamma_0 \langle \nabla \dot{\varphi}'(\mathbf{0}), \mathbf{v}_\infty^{lz1} \rangle \int_0^t R_A^{lz2}(t, s) ds - \int_0^t [M_{R,d}^{lz2}(t, s) C_d^{lz2}(t, s) + M_{R,o}^{lz2}(t, s) C_o^{lz2}(t, s)] ds \\ &\quad - \int_0^t [M_{C,d}^{lz2}(t, s) R_d^{lz2}(t, s) + M_{C,o}^{lz2}(t, s) R_o^{lz2}(t, s)] ds. \end{aligned} \quad (\text{F.66})$$

Finally, the train and test errors converge to well defined limits for <sup>t</sup> fixed and <sup>m</sup> → ∞:

$$e_{\text{tr}}(t, \gamma_0) = e_{\text{tr}}^{72}(t, \gamma_0) + o_m(1), \quad e_{\text{ts}}(t, \gamma_0) = e_{\text{ts}}^{72}(t, \gamma_0) + o_m(1) ., \quad (\text{F.69})$$

where

$$e_{\text{tr}}^{\text{lz}}(t, \gamma_0) = -\frac{1}{2}C_A^{\text{lz}}(t, t), \quad e_{\text{ts}}^{\text{lz}}(t, \gamma_0) = \frac{1}{2}\Sigma_C^{\text{lz}}(t, t). \quad (\text{F.70})$$

Note that, using Eqs. [\(F.62\)](#page-42-5), [\(F.64\)](#page-43-0), and the fact that C lz2 d (t, t) = 1 (because of the unit norm constraint on the first layer weights), we get

$$e_{\text{ts}}^{k2}(t, \gamma_0) = \frac{1}{2} \left\{ \tau^2 + \|\varphi\|^2 - 2\gamma_0 \langle \nabla \hat{\varphi}(\mathbf{0}), \mathbf{v}_\infty^{k1} \rangle + \gamma_0^2 (h(1) - h'(0) + h'(0) \|\mathbf{v}_\infty^{k1}\|^2) \right\}. \quad (\text{F.71})$$

![](_page_44_Figure_0.jpeg)

Figure 14: SymmDMFT predictions and large network scaling for lazy training in a single index model: train and test error. Left frame: train and test error on the time scale t = Θ(1/m) for several values of <sup>m</sup>, together with the asymptotic prediction as <sup>m</sup> → ∞ on this time scale e lz1 tr (t, γ <sup>ˆ</sup> <sup>0</sup>) = <sup>e</sup> lz1 ts (t, γ <sup>ˆ</sup> <sup>0</sup>). Right: train and test error on the time scale <sup>t</sup> = Θ(1) for several values of <sup>m</sup>, together with the asymptotic prediction as <sup>m</sup> → ∞ on this time scale <sup>e</sup> lz2 tr (t, γ0). Here γ<sup>0</sup> = 1, h(z) = ˆφ(z) = (9/10)z + z <sup>2</sup>/2, τ = 0.3, α = 0.3.

Using Eq. [\(F.57\)](#page-42-6), we obtain that the asymptotic test error in this dynamical regime is constant and equal to the test error achieved at the end of the previous regime, namely e lz2 ts (t, γ0) = e lz1 ts,∞, cf. Eq. [\(F.59\)](#page-42-3). As anticipated, no learning takes place on this timescale.

The predictions of Eqs. [\(F.61\)](#page-42-4) are tested in the right panel of Fig. [13.](#page-41-3) We plot the correlation function Cd(t, 0) for several values of m, as obtained by solving the SymmDMFT equations. We compare these results with the <sup>m</sup> → ∞ prediction <sup>C</sup> lz2 d (t, 0) obtained by solving Eqs. [\(F.66\)](#page-43-1) to [\(F.68\)](#page-43-2). We observe collapse of finite m curves on the large m asymptotics supporting our conclusions.

In Fig. [14](#page-44-0) we plot the behavior of the train and test error both on timescales t = Θ(1/m) (left frame, plotting etr(t, γ0), ets(t, γ0) versus tm) and t = Θ(1) (right frame, plotting etr(t, γ0), ets(t, γ0) versus tm). We the solutions of SymmDMFT equations at increasing values of m with the theory scaling theory presented in the previous section (for t = Θ(1/m), left frame) and in this section (for t = Θ(1), right frame). As anticipated, we observe the following:

- On the time scale <sup>t</sup> = Θ(1/m) (left panel), test and train error collapse (as <sup>m</sup> → ∞) on a common limiting curve e lz1 tr (t, γ <sup>ˆ</sup> <sup>0</sup>) = <sup>e</sup> lz1 ts (t, γ <sup>ˆ</sup> <sup>0</sup>) which converges, for large <sup>t</sup>ˆ, to the positive limiting value e lz1 ts,<sup>∞</sup> characterized in the previous section.
- On the time scale <sup>t</sup> = Θ(1) (right panel), test and train error collapse (as <sup>m</sup> → ∞) on two distinct limiting curves. The first one is constant and equal to e lz1 ts,∞. The second one decreases from e lz1 ts,<sup>∞</sup> to 0 and is predicted by the asymptotic theory in this section, cf. Eq. [\(F.70\)](#page-43-3).

Note that, in the example of Fig. [14,](#page-44-0) the initialization γ<sup>0</sup> is sufficiently large that the train error decreases to zero on the time scale Θ(1), namely γ<sup>0</sup> > γ<sup>∗</sup> GF(α, φ, τ ), for a suitable threshold γ ∗ GF(α, φ, τ ). As we will see in the next section, a third dynamical regime emerges when γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ).

## F.2.3 The algorithmic interpolation threshold

The asymptotic theory within the second dynamical regime, described in Section [F.2.2,](#page-42-1) turns out to be equivalent to the one in the pure-noise model, Section [F.1.2,](#page-35-0) up to a change of variables. Namely, defining

$$\tilde{C}_o(t, s) = C_o^{lz2}(t, s) + \|\mathbf{v}_\infty^{lz1}\|^2, \quad (\text{F.72})$$

with initial condition C˜ <sup>o</sup>(0, 0) = −<sup>1</sup> , reduce the equations of Section [F.2.2](#page-42-1) to the ones of Section [F.1.2](#page-35-0) with noise level τ replaced by

$$\tau'^2 = \tau^2 + \|\varphi\|^2 - \frac{\|\nabla \hat{\varphi}(\mathbf{0})\|^2}{h'(0)}. \quad (\text{F.73})$$

![](_page_45_Figure_0.jpeg)

Figure 15: The asymptotic behavior of the test error as a function of m for different h(z) = ˆφ(z). We observe that soon as h(z) contains a z 2 term, the NTK limit for <sup>m</sup> → ∞ is approached from below (left panel). Furthermore the speed of the convergence to the limiting value depends crucially on whether a z <sup>2</sup> monomial is present in the Taylor expansion of h(z) (right panel). The data has been produced with α = 0.3 and τ = 0.6.

The interpretation of this reduction is simple. On the time scale t = Θ(1), the first layer weights move orthogonally to the latent subspace spanned by U. Hence, the dynamics on this timescale is not affected by the signal and only attempts to fit the labels noise. The noise is inflated as per Eq. [\(F.73\)](#page-44-1), because the network is not able to fit beyond the linear part of the target distribution.

As a corollary of the above equivalence, the interpolation threshold of the k-index model coincides with with the interpolation threshold on pure noise data with noise level given by Eq. [\(F.73\)](#page-44-1). Using the extended notation γ ∗ GF(α, φ, τ ) to indicate the dependence on the underlying data distribution (which is parametrized by φ, τ ), we can write the stated relation as

$$\gamma_{\text{GF}}^*(\alpha, \varphi, \tau) = \left( \tau^2 + \|\varphi\|^2 - \frac{\|\nabla \hat{\varphi}(\mathbf{0})\|^2}{h'(0)} \right)^{1/2} \gamma_{\text{GF}}^*(\alpha, 0, 1). \quad (\text{F.74})$$

(Here we used the invariance under rescaling in the pure noise model, which implies γ ∗ GF(α, 0, τ <sup>2</sup> ) = τ γ<sup>∗</sup> GF(α, 0, 1).)

#### F.2.4 Dependence on m

Within NTK theory, it is normally assumed that optimal models are achieved at very large network sizes <sup>m</sup> → ∞. Empirical results contradicting this expectation have been put forward in [\[54\]](#page-12-17), but no theoretical analysis was provided either in [\[54\]](#page-12-17) or in subsequent work. We can use the SymmDMFT theory to fill this gap and study the dependence of test error on the number of neurons m under lazy initialization. We choose γ<sup>0</sup> > γ<sup>∗</sup> GF(α, φ, τ ), and therefore vanishing training error is reached during the second dynamical regime, i.e. for t = Θ(1): this is therefore the last dynamical regime. Throughout this regime, we have γ(t) = γ<sup>0</sup> + om(1).

Recalling that ets(t, γ0, m, α) is the test error at time t in this setting, as predicted by SymmDMFT we consider the limit

$$e_\infty^\ell(\gamma_0, m, \alpha) = \lim_{t \rightarrow \infty} e_{\text{ts}}(t, \gamma_0, m, \alpha). \quad (\text{F.75})$$

We note that, for γ<sup>0</sup> > γ<sup>∗</sup> GF(α, φ, τ ), we expect

$$\lim_{m \rightarrow \infty} e_{\infty}^{\ell}(\gamma_0, m, \alpha) = e_{\text{ts}, \infty}^{\text{lzl}}, \quad (\text{F.76})$$

to be given by Eq. [\(F.59\)](#page-42-3).

In Fig. [15](#page-45-0) we plot the SymmDMFT prediction for e ℓ <sup>∞</sup>(γ0, m, α) as a function of m for several choices of <sup>h</sup> (we use <sup>h</sup> = ˆ<sup>φ</sup> here). The limit <sup>m</sup> → ∞ of these curves matches <sup>e</sup> lz1 ts,<sup>∞</sup> as expected. However we empirically observe that e ℓ <sup>∞</sup>(γ0, m, α) approaches e lz1 ts,<sup>∞</sup> in two qualitatively different ways:

![](_page_46_Figure_0.jpeg)

Figure 16: Train and test error on different timescales when training on single index data and lazy initialization. Train error (solid curves) and test error (dashed curves) for a model trained on a single index data with h(z) = (9/10)z + z <sup>2</sup>/2 = ˆφ(z). The noise level is τ = 2.5 and initialization a(0) = γ<sup>0</sup> √ m, γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ). Left panel: timescales of order one. The grey dashed line corresponds to the scaling solution for <sup>m</sup> → ∞ when the second layer does not evolve with GF. Right panel: same data plotted versus t/m, to explore timescales of order m. The arrows show scaling appearing and curves collapsing on a master curve.

- In the cases we consider that have h ′′(0) ̸= 0, <sup>e</sup> lz1 ts,<sup>∞</sup> is approached from below as <sup>m</sup> → ∞, and e ℓ <sup>∞</sup>(γ0, m, α) is non-monotone. We also observe that, for the values of m we consider, the approach to the asymptotic value is compatible with a rate m<sup>−</sup>1/<sup>2</sup> : e ℓ <sup>∞</sup>(γ0, m, α) = e lz1 ts,<sup>∞</sup> − Θ(m<sup>−</sup>1/<sup>2</sup> ).
- In the cases we consider that have h ′′(0) ̸= 0, then <sup>e</sup> lz1 ts,<sup>∞</sup> is approached from above as <sup>m</sup> → ∞, and <sup>e</sup> ℓ <sup>∞</sup>(γ0, m, α) is typically monotone. In this case the approach to the limiting behavior is compatible with a rate m<sup>−</sup><sup>1</sup> : e ℓ <sup>∞</sup>(γ0, m, α) = e lz1 ts,<sup>∞</sup> + Θ(m<sup>−</sup><sup>1</sup> ).

The first scenario is the generic one, and similar to what is observed in [\[54\]](#page-12-17) for actual neural networks. An intuitive explanation is that –at finite m– the projection of neurons onto the latent space ∥v <sup>∞</sup>∥ = Θ(1/ √ m) is sufficient for the network to partially learn the quadratic component of the target function. In order to establish on more solid grounds these empirical observations one should study the 1/m corrections to the scaling theory developed here. This is left for future work.

#### F.2.5 Third dynamical regime: t = Θ(m)

As for the pure noise case, beyond the time scale t = Θ(1), we distinguish two situations. If γ<sup>0</sup> > γ<sup>∗</sup> GF(α, φ, τ ), then vanishing training error is reached within the second dynamical regime t = Θ(1). If γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ), GF dynamics develops an additional regime for t = Θ(m). In this section, we study this third regime.

In Figure [16,](#page-46-1) we plot the SymmDMFT predictions for train and test errors as a function of time for several values of m, for a setting with γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ). In particular, in Fig[.16-](#page-46-1)left we plot train and test error as a function of t. The curves for the train error for increasing value of m collapse on limit curve given by e lz2 tr (t, γ0) characterized in Section [F.2.2.](#page-42-1) In other words, the dynamics on this timescales follows the scaling theory of Section [F.2.2.](#page-42-1) However in this case γ<sup>0</sup> < γ<sup>∗</sup> GF(α, φ, τ ), whence by definition e lz2 tr,<sup>∞</sup> > 0. This correspond to the limit curve in Fig. [16-](#page-46-1)left having a strictly positive asymptote.

Figure [16-](#page-46-1)right shows train and test error plotted against t/m. We observe that curves training error curves collapse on a common limit, that decreases from e lz2 tr,<sup>∞</sup> to 0, while test error curves increase above the plateau e lz1 ts,∞. This suggests the following limit behavior

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(m\hat{t}, \gamma_0, m) = e_{\text{tr}}^{lx3}(\hat{t}, \gamma_0) \\ \lim_{m \rightarrow \infty} e_{\text{ts}}(m\hat{t}, \gamma_0, m) = e_{\text{ts}}^{lx3}(\hat{t}, \gamma_0) . \quad (\text{F.77})$$

![](_page_47_Figure_0.jpeg)

Figure 17: Training a two layer network in the same setting of Figure [16.](#page-46-1) Left panel: second layer weights on the timescale of order m. The black arrow corresponds to the interpolation threshold for a model, γ ∗ GF (α, τ ) obtained by fitting the relaxation time as a function of the weights of an lazy initialized model for γ<sup>0</sup> > γ<sup>∗</sup> GF(α, τ ). The second layer weights, at finite m develop a plateau at long time. In the inset we show the approach of this plateaus to the limiting value given by γ ∗ GF(α, φ, τ ). Right panel: parametric plot of the train error as a function of the scaled weights of the second layer. The dashed gray dashed line corresponds to the extrapolated train error for an network with second layer weights fixed to the corresponding value in the <sup>m</sup> → ∞ (as extracted from the numerical integration of the scaling theory).

In order to further explore the GF dynamics in this regime, in Fig. [17-](#page-47-0)left we plot the evolution of the second layer rescaled weights against t/m. The curves for increasing values of m collapse on a master curve, suggesting the existence of a limit

$$\lim_{m \rightarrow \infty} \gamma(m\hat{t}, \gamma_0) = \gamma^{lz}(\hat{t}, \gamma_0). \quad (\text{F.78})$$

The limit curve γ lz3(t, γ ˆ <sup>0</sup>) increases from γ<sup>0</sup> to a limit value:

$$\lim_{\hat{t} \rightarrow \infty} \gamma^{lz3}(\hat{t}, \gamma_0) = \gamma_\infty^{lz3}(\gamma_0). \quad (\text{F.79})$$

As in Section [F.2.5](#page-46-0) we consider the inverse function of <sup>t</sup> 7→ <sup>γ</sup> lz3(t, γ0), denoted by <sup>γ</sup> 7→ <sup>γ</sup>˜ −1 (γ, γ0). In Fig. [17-](#page-47-0)right we plot the train error as a function of the second layer weights γ(t). Again, for increasing values of m the curves collapse on a master curve which is given by

$$\varepsilon(\gamma, \gamma_0) = e_{\text{tr}}^{\text{lz}}(\tilde{\gamma}^{-1}(\gamma, \gamma_0), \gamma_0) \quad (\text{F.80})$$

We then also plot in Fig[.17-](#page-47-0)right the asymptotic value of the train error for a network initialized with second layer weights blocked at an initialization scale γ, call it e lz2 tr,<sup>∞</sup>(γ).

The curves ε(γ, γ0) appear to have a vertical segment (corresponding to t = o(m)) in which the training error decreases, while γ(t) = γ0+om(1) is nearly unchanged, and a continuously decreasing segment in which γ(t) increases while etr(t, γ0) decreases to 0 (corresponding to t = Θ(m)). In the second phase, the curves appear to converge to e lz2 tr,<sup>∞</sup>(γ) as <sup>m</sup> → ∞. This suggests

$$\varepsilon(\gamma, \gamma_0) = e_{\text{tr}, \infty}^{l2^2}(\gamma) \quad \forall \gamma \geq \gamma_0 . \quad (\text{F.81})$$

In other words the dynamics on timescales of order m is adiabatic also in the multi index case. For a small change of the second layer weights on a scale of order √ m, the train error relaxes to its asymptotic value on timescales of order one. This graph suggests that the limit value of γ(t) coincides with the critical value for interpolation. Namely recalling the definition [\(F.79\)](#page-47-1) for the asymptotic value of γ(t), we have

$$\gamma_\infty^{lz3}(\gamma_0) = \gamma_{\text{GF}}^*(\alpha, \varphi, \tau) \quad (\text{F.82})$$

where the interpolation threshold in the multi-index model γ ∗ GF(α, φ, τ ) is related to the interpolation threshold in the pure noise model via Eq. [\(F.74\)](#page-45-1).

## G Dynamical regimes: Mean field initialization

In this section we assume the initialization of the weights of the second layer is kept of order one. To be definite, we set a(0) = a0, independent of m. This corresponds to the mean field initialization studied in [\[38,](#page-12-6) [14,](#page-10-9) [45\]](#page-12-8).

Specializing to the data distribution considered here, earlier work characterized the dynamics up to time T, under a few settings (which prove equivalent in this regime):

- One-pass SGD, with stepsize <sup>ε</sup> ≪ <sup>1</sup>/d and therefore time horizons such that <sup>T</sup> ≪ d/n (the latter inequality follows from <sup>T</sup> ≤ nε for one-pass SGD). In this case, the dynamics is characterized by a set of ODEs for for the projections of the weights on the latent space and inner products between weights.
- Gradient flow in the population risk, which admits the same characterization and corresponds to the limit <sup>n</sup> → ∞ of the above.
- The limit of the above regimes for large width <sup>m</sup> → ∞. This is characterized by a partial differential equation for the distribution of projections of first layer weights onto the latent space, provided <sup>T</sup> ≤ <sup>c</sup><sup>0</sup> log <sup>m</sup>, for <sup>c</sup><sup>0</sup> a sufficiently small constant.

We refer to [\[5,](#page-10-4) [21,](#page-11-5) [1,](#page-10-5) [6,](#page-10-6) [2,](#page-10-13) [10\]](#page-10-12) for a few pointers to this literature. In all of these settings, the train error remains close to the test error. In contrast, the analysis presented here allow us to explore the overfitting regime.

Section [G.1,](#page-48-0) we will focus on a pure noise data distribution, while Section [G.2,](#page-54-0) considers a multiindex model. As in the case of lazy initializations, we consider first the limit n, d → ∞ at n/md <sup>=</sup> <sup>α</sup> and m fixed (hence characterized by SymmDMFT ) and subsequently study dynamical regimes emerging as <sup>m</sup> → ∞ at n/md <sup>=</sup> <sup>α</sup> fixed.

#### G.1 Pure noise model

Under the pure noise model, we have φ = ˆφ = 0. We identify three distinct dynamical regimes:

- t = O(1): a(t) = a<sup>0</sup> + om(1), etr(t) = τ <sup>2</sup>/2 + <sup>o</sup>m(1), and ∥<sup>w</sup>i(t) − <sup>w</sup>i(0)∥ <sup>=</sup> <sup>o</sup>m(1). In words, the weights change minimally and the train error remains close to the one of the null network <sup>f</sup>(x; <sup>θ</sup>) ≈ <sup>0</sup> (Section [G.1.1\)](#page-48-2).
- <sup>t</sup> = Θ(√ m): a(t) = Θ(1), etr(t) = τ <sup>2</sup>/2+om(1), and ∥<sup>w</sup>i(t)−<sup>w</sup>i(0)∥ = Θ(1). Namely, weights change but the train error does not change significantly. (Section [G.1.2\)](#page-49-0).
- <sup>t</sup> = Θ(m). In this regime <sup>a</sup>(t) = √ mγ(t/m) + om(1), and therefore the network complexity becomes large enough for it to fit the noise. The dynamics on this timescale is closely related to the one under lazy initialization, studied in Section [F.1.4.](#page-39-0) In particular, γ(tˆ) converge to the interpolation threshold γ ∗ GF(α, τ ) if <sup>t</sup><sup>ˆ</sup>→ ∞ (after <sup>m</sup> → ∞). (Section [G.1.3\)](#page-52-0).

## G.1.1 First dynamical regime: t = O(1)

In this dynamical regime, the SymmDMFT equations are solved by the following scaling ansatz

$$C_d(t, \omega) = \text{Comf}(t, \omega) + \omega(1 - \omega) - R_d(t, \omega) - \text{Pom}(t, \omega) + \omega(1 - \omega) - G(1). \quad (\text{G.1})$$

$$mC_o(t, s) = C_o^{\text{mfl}}(t, s) + o_m(1) \quad mR_o(t, s) = R_o^{\text{mfl}}(t, s) + o_m(1), \quad (\text{G.2})$$

$$\begin{aligned} m_{\mathcal{O}}(t, s) &= \mathcal{O}_{\mathcal{O}} \cdot (t, s) + o_m(1) & m_{\mathcal{A}}(t, s) &= \mathcal{H}_{\mathcal{O}} \cdot (t, s) + o_m(1), & (G.2) \\ a(t) &= a_0 + o_m(1) & \nu(t) &= o_m(1). & (G.3) \end{aligned}$$

Furthermore we have

$$R_A^{\text{mfl}}(t, s) = \delta(t - s) + o_m(1) \quad C_A^{\text{mfl}}(t, s) = -\tau^2 + o_m(1), \quad (\text{G.4})$$

Plugging the scaling ansatz in the SymmDMFT , we obtain equations determining the scaling functions C mf1 o , Rmf1 o . Defining

$$\rho_0 := \alpha a_0^2 h'(0) \quad (\text{G.5})$$

![](_page_49_Figure_0.jpeg)

Figure 18: Training on pure noise data under mean-field initialization: t = Θ(1) regime. We plot Co(t, 0) and Co(t, t) as given by solving the SymmDMFT equations for different values of m and compare them with the asymptotic solution of Section [G.1.1.](#page-48-2) Here we use τ = 0.6, α = 0.3 and h(z) = (9/10)z + z <sup>3</sup>/6. Note that the vertical axis is multiplied by a factor m, in agreement with the prediction of Eq. [\(G.2\)](#page-48-3).

we have

$$\begin{aligned} R_o^{\text{mfl}}(t, s) &= \left[ e^{-\rho_0(t-s)} - 1 \right] \vartheta(t-s), \\ C_o^{\text{mfl}}(t, t') &= \left[ \left[ \frac{2\tau^2}{\rho_0} - \frac{1}{\rho_0} (\tau^2 - \rho_0) \right] e^{-2\rho_0 t'} - \frac{\tau^2}{\rho_0} e^{-\rho_0 t'} \right] e^{-\rho_0(t-t')} \\ &\quad + \frac{\tau^2 - \rho_0}{\rho_0} - \frac{\tau^2}{\rho_0} e^{-\rho_0 t'}. \end{aligned} \quad (\text{G.6})$$

In particular

$$\begin{aligned} \lim_{t \rightarrow \infty} C_o^{\text{mfl}}(t, t) &= \frac{\tau^2 - \rho_0}{\rho_0}, \\ \lim_{t \rightarrow \infty} C_o^{\text{mfl}}(t, t') &= \frac{\tau^2 - \rho_0}{\rho_0} - \frac{\tau^2}{\rho_0} e^{-\rho_0 t'}, \\ \lim_{t \rightarrow \infty, t' \rightarrow \infty, t-t' \geq 0} C_o^{\text{mfl}}(t, t') &= \frac{\tau^2 - \rho_0}{\rho_0}. \end{aligned} \quad (\text{G.7})$$

The equations [\(G.4\)](#page-48-4) imply that the train error is constant in this regime and equal to

$$e_{\text{tr}}(t) = \frac{\tau^2}{2} + o_m(1). \quad (\text{G.8})$$

In other words, in this regime both first and second layer weights change minimally and the resulting error remains close to the one to the null function <sup>f</sup>(x; <sup>θ</sup>) ≈ <sup>0</sup>. We will see that this regime is significantly more interesting for the case of data with a signal, see Section [G.2.](#page-54-0) We note in passing that the limit value ⟨<sup>w</sup><sup>j</sup> , <sup>w</sup><sup>j</sup> ⟩ ≈ <sup>τ</sup> <sup>2</sup>−ρ<sup>0</sup> mρ<sup>0</sup> for <sup>i</sup> ̸<sup>=</sup> <sup>j</sup> corresponds to minimizing the empirical risk under the linear approximation in which <sup>σ</sup>(z) is replaced by p h ′(0)z.

The above predictions are tested in Fig. [18](#page-49-1) where we plot Co(t, t) and Co(t, 0) for different values of m and check their approach to the scaling functions C mf1 o (t, 0) and C mf1 o (t, t).

#### G.1.2 Second dynamical regime: <sup>t</sup> = Θ(√ m)

We now consider the case in which time scales as √ m. The following asymptotic forms can be checked to solve the SymmDMFT equations, up to higher order terms, for suitable choices of the scaling functions on the right-hand side:

$$C_d(t\sqrt{m}, s\sqrt{m}) = C_d^{\text{mf2}}(t, s) + o_m(1) \qquad R_d(t\sqrt{m}, s\sqrt{m}) = R_d^{\text{mf2}}(t, s) + o_m(1),$$

![](_page_50_Figure_0.jpeg)

Figure 19: Training on pure noise data under mean-field initialization: <sup>t</sup> = Θ(√ m) regime, under the same setting as in Fig. [18.](#page-49-1) We plot the solutions of the SymmDMFT equations for several values of <sup>m</sup> as a function of t/√ <sup>m</sup>. We compare these to the <sup>m</sup> → ∞ scaling theory of Section [G.1.2,](#page-49-0) i.e. to numerical solutions of Eqs. [\(G.14\)](#page-50-0) to [\(G.17\)](#page-51-0).

$$C_o(t\sqrt{m}, s\sqrt{m}) = \frac{1}{m} C_o^{\text{mf2}}(t, s) + o_m(m^{-1}) \quad R_o(t\sqrt{m}, s\sqrt{m}) = \frac{1}{m} R_o^{\text{mf2}}(t, s) + o_m(m^{-1}), \quad (\text{G.10})$$

$$\begin{aligned} \sqrt{m}R_A(t\sqrt{m}, s\sqrt{m}) &= \delta(t-s) + o_m(1) & C_A(t\sqrt{m}, s\sqrt{m}) &= -\tau^2 + o_m(1), & (G.11) \\ \sqrt{m}\nu(t\sqrt{m}) &= \nu^{\text{mf2}}(t) + o_m(1) & a(t\sqrt{m}) &= a^{\text{mf2}}(t) + o_m(1). & (G.12) \end{aligned}$$

Plugging this scaling ansatz into the SymmDMFT equations we get the constraints

$$R_o^{\text{mf2}}(t, s) = -R_d^{\text{mf2}}(t, s), \quad (\text{G.13})$$

$$C_o^{\text{mf2}}(t, s) = -C_d^{\text{mf2}}(t, s) + \frac{\tau^2}{\alpha h'(0)(a^{\text{mf2}}(t))^2}.$$

We also obtain that the following equations must be satisfied by C mf2 d (t, t′ ), Rmf2 d (t, t′ ), a mf2(t), ν mf2(t),

$$\partial_t C_d^{\text{mf2}}(t, t') = -\nu^{\text{mf2}}(t) C_d^{\text{mf2}}(t, t') + \alpha \tau^2 a^{\text{mf2}}(t) \int_0^t a^{\text{mf2}}(s) h''(C_d^{\text{mf2}}(t, s)) R_d^{\text{mf2}}(t, s) C_d^{\text{mf2}}(t', s) \, \text{d}s \quad (\text{G.14})$$

+ ατ <sup>2</sup> a mf2(t) Z t ′

0 a mf2(s) h ′ (C mf2 d

(t, s)) − <sup>h</sup>

′ (0) R mf2 d (t ′ , s) ds ,

∂tR mf2 d (t, t′

) = <sup>δ</sup>(<sup>t</sup> − <sup>t</sup>

′ ) − <sup>ν</sup>

mf2(t)R mf2 d (t, t′

) (G.15)

+ ατ <sup>2</sup> a mf2(t)

$$\begin{aligned} &+ \alpha \tau^2 a^{\text{mf2}}(t) \int_{t'}^t a^{\text{mf2}}(s) h''(C_d^{\text{mf2}}(t, s)) R_d^{\text{mf2}}(t, s) R_d^{\text{mf2}}(s, t') \, ds , \\ \nu^{\text{mf2}}(t) &= \alpha \tau^2 a^{\text{mf2}}(t) \int_0^t [a^{\text{mf2}}(s) h''(C_d^{\text{mf2}}(t, s)) R_d^{\text{mf2}}(t, s) C_d^{\text{mf2}}(t, s)] \, ds \end{aligned} \quad (\text{G.16})$$

a mf2(s)h

′′(C mf2 d

(t, s))R

mf2 d

(t, s)R mf2 d (s, t′

) ds ,

$$\begin{aligned} &+ \alpha \tau^2 a^{\text{mf2}}(t) \int_0^t a^{\text{mf2}}(s) [h'(C_d^{\text{mf2}}(t, s)) - h'(0)] R_d^{\text{mf2}}(t, s) ds, \\ \frac{da^{\text{mf2}}(t)}{dt} &= \alpha \tau^2 \int_0^t a^{\text{mf2}}(s) [h'(C_d^{\text{mf2}}(t, s)) - h'(0)] R_d^{\text{mf2}}(t, s) ds, \end{aligned} \quad (\text{G.17})$$
initial conditions given by

with initial conditions given by

$$C_d^{\text{mf2}}(0, 0) = 1 \quad R_d^{\text{mf2}}(0+, 0) = 1 \quad a^{\text{mf2}}(0) = a_0 . \quad (\text{G.18})$$

We test these predictions in Fig. [19.](#page-50-1) We plot several quantities in the solution of the SymmDMFT equations for increasing values of m and compare them with the solution of the asymptotic equations [\(G.14\)](#page-50-0) to [\(G.17\)](#page-51-0). We observe convergence to the predicted asymptotic behavior.

Equations [\(G.14\)](#page-50-0) to [\(G.17\)](#page-51-0) can be further simplified. The right-hand side of Eq. [\(G.17\)](#page-51-0) is a positive. Therefore a mf2(t) is a monotone increasing function. Define the time change

$$\tilde{t}(t) = \tau \sqrt{\alpha} \int_0^t a^{\text{mf2}}(s) \, ds, \quad (\text{G.19})$$

and the corresponding time-changed scaling functions

$$\begin{aligned}\tilde{\nu}(\tilde{t}(t)) &= \frac{\nu^{\text{mf2}}(t)}{a^{\text{mf2}}(t)\tau\sqrt{\alpha}}, \\ \tilde{C}_d^{\text{mf}}(\tilde{t}(t), \tilde{t}(t')) &= C_d^{\text{mf2}}(t, t'), \\ \tilde{R}_d^{\text{mf}}(\tilde{t}(t), \tilde{t}(t')) &= R_d^{\text{mf2}}(t, t').\end{aligned}\tag{G.20}$$

Equations [\(G.14\)](#page-50-0) to [\(G.17\)](#page-51-0) imply that these time-changed function functions satisfy

$$\begin{aligned} \partial_t \tilde{C}_d^{\text{mf}}(t, t') &= -\tilde{\nu}^{\text{mf}}(t) \tilde{C}_d^{\text{mf}}(t, t') + \int_0^t \tilde{h}''(\tilde{C}_d^{\text{mf}}(t, s)) \tilde{R}_d^{\text{mf}}(t, s) \tilde{C}_d^{\text{mf}}(t', s) \, ds \\ &+ \int_0^{t'} \tilde{h}'(\tilde{C}_d^{\text{mf}}(t, s)) \tilde{R}_d^{\text{mf}}(t', s) \, ds, \end{aligned} \quad (\text{G.21})$$

$$\partial_t \tilde{R}_d^{\text{mf}}(t, t') = \delta(t - t') - \tilde{\nu}^{\text{mf}}(t) \tilde{R}_d^{\text{mf}}(t, t') + \int_{t'}^t \tilde{h}''(\tilde{C}_d^{\text{mf}}(t, s)) \tilde{R}_d^{\text{mf}}(t, s) \tilde{R}_d^{\text{mf}}(s, t') \, \text{d}s \quad (\text{G.22})$$

d

d

$$\tilde{\nu}^{\text{mf}}(t) = \int_0^t \tilde{h}''(\tilde{C}_d^{\text{mf}}(t, s)) \tilde{R}_d^{\text{mf}}(t, s) \tilde{C}_d^{\text{mf}}(t, s) \, ds + \int_0^t \tilde{h}'(\tilde{C}_d^{\text{mf}}(t, s)) \tilde{R}_d^{\text{mf}}(t, s) \, ds, \quad (\text{G.23})$$

where again <sup>h</sup>˜(z) = <sup>h</sup>(z) − <sup>h</sup> ′ (0)z.

Equations [\(G.21\)](#page-51-1), [\(G.23\)](#page-51-2) are independent of the dynamics of the second layer weights. These equations are nothing but the DMFT equations describing gradient descent dynamics of the celebrated spherical mixed p-spin glass model [\[17,](#page-10-16) [20,](#page-11-14) [8,](#page-10-15) [23\]](#page-11-15), whose definition we recall next. Consider a random cost function <sup>H</sup>(x) indexed <sup>x</sup> ∈ <sup>S</sup> d−1 , which is a centered Gaussian process with covariance structure given by

$$\mathbb{E} (H(x)H(y)) = d\tilde{h}(\langle x, y \rangle). \quad (\text{G.24})$$

Define the gradient flow dynamics

$$\dot{\mathbf{x}}(t) = -\mathbf{P}_{\mathbf{x}(t)}^\perp \nabla H(\mathbf{x}(t)), \quad (\text{G.25})$$

where P ⊥ x(t) is the projector orthogonal to x(t). Then the high-dimensional asymptotics of this dynamics is characterized by Eqs. [\(G.21\)](#page-51-1), [\(G.23\)](#page-51-2). In particular limd→∞⟨x(t), <sup>x</sup>(s)⟩ <sup>=</sup> <sup>C</sup>˜mf d (t, t′ ) almost surely.

A particularly interesting quantity is the asymptotic energy value in the mixed p-spin model:

$$\mathcal{E} = \lim_{t \rightarrow \infty} \lim_{d \rightarrow \infty} \frac{1}{d} H(\mathbf{x}(t)). \quad (\text{G.26})$$

The DMFT analysis for this problem implies that

$$\mathcal{E} = - \lim_{t \rightarrow \infty} \int_0^t \tilde{h}'(\tilde{C}_d^{\text{mf}}(t, s)) \hat{R}_d^{\text{mf}}(t, s) \, ds . \quad (\text{G.27})$$

![](_page_52_Figure_0.jpeg)

Figure 20: Evolution of second layer weights, as predicted by the numerical solution of Eq. [\(G.17\)](#page-51-0). Here we use h(z) = (9/10)z + z <sup>3</sup>/6, α = 0.3 and τ = 0.6. The straight line is just a guide to the eyes to test the prediction of Eq. [\(G.29\)](#page-52-1).

For h˜(z) = c 2 k z k , <sup>k</sup> ≥ <sup>2</sup>, we have the explicit expression [\[20,](#page-11-14) [19,](#page-11-16) [46\]](#page-12-18)

$$\mathcal{E} = -2c_k\sqrt{\frac{k-1}{k}} . \quad (\text{G.28})$$

An explicit expression for E for general covariance structure is an unknown [\[23\]](#page-11-15).

The asymptotic energy E has an interesting interpretation for the dynamics of two-layer networks –within the SymmDMFT theory. Eq. [\(G.28\)](#page-52-2) implies that

$$\lim_{t \rightarrow \infty} \frac{a^{\text{mf2}}(t)}{t} = -\tau\sqrt{\alpha \mathcal{E}} =: A_\infty. \quad (\text{G.29})$$

In Fig. [20](#page-52-3) we test the prediction of Eq. [\(G.29\)](#page-52-1) by integrating numerically Eqs. [\(G.14\)](#page-50-0) to [\(G.17\)](#page-51-0) and plotting the prediction for the second-layer weigths a mf(t). We observe that at large t, a mf2(t) ≈ <sup>A</sup>∞t, with A<sup>∞</sup> given by Eq. [\(G.29\)](#page-52-1)as predicted.

We also note that <sup>C</sup>A(t, t) = −<sup>τ</sup> 2 also in this timescale, and hence the train error does not change significantly. Namely , for any constant t, we have

$$e_{\text{tr}}(t\sqrt{m}) = \frac{1}{2}\tau^2 + o_m(1). \quad (\text{G.30})$$

If we use heuristically Eq. [\(G.28\)](#page-52-2) and Eq. [\(G.12\)](#page-50-2) beyond the √ t time scale, we obtain

$$a(t) \approx a^{\text{mf2}}(t/\sqrt{m}) \approx A_{\infty} \frac{t}{\sqrt{m}}. \quad (\text{G.31})$$

This suggests that <sup>a</sup>(t) becomes of order √ m on timescale of order m. When this happens, the network complexity is large enough to allow for interpolation, and hence we expect the dynamics to change. Indeed a new dynamical regime emerges for t = Θ(m), as we will study next.

## G.1.3 Third dynamical regime: t = Θ(m)

As anticipated, an additional regime arises on timescales of order m. In Figure [21](#page-53-0) we plot the evolution of the weights of the second layer as a function of t/m for increasing values of the width m. The different curves collapse suggesting the following limit to exist

$$\lim_{m \rightarrow \infty} \frac{a(tm)}{\sqrt{m}} = \gamma^{\text{mf3}}(t) . \quad (\text{G.32})$$

The limit curve appears to grows linearly at small t, γ mf3(t) = A∞t + o(t), where A<sup>∞</sup> is the coefficient computed in the previous section, cf. Eq. [\(G.29\)](#page-52-1). Hence, this third dynamical regime matches directly with the previous one. As can be seen from the right plot, there appear to be a finite limit limt→∞ γ mf3(t) <sup>&</sup>lt; ∞.

![](_page_53_Figure_0.jpeg)

Figure 21: Evolution of the second layer weigths when training on pure noise data under mean field initialization for t = Θ(m). Rescaled second layer weights a(t)/ √ m as a function of t/m. We plot solutions of the SymmDMFT equations for the setting of Fig. [18.](#page-49-1)

![](_page_53_Figure_2.jpeg)

Figure 22: Train error and Lagrange multiplier ν(t) on timescales of order m under mean field initialization for pure noise data. Solutions of the SymmDMFT equations for the setting of Fig. [18.](#page-49-1) Finite m curves accumulate on master curves suggesting the existence of scaling functions.

We now turn to the analysis of the train error. Recall that on the previous timescales, the train error stays approximately constant, and equal to the train error of the null network, namely etr(t √ m) = τ <sup>2</sup>/2 + om(1) for any fixed t. In Fig. [22](#page-53-1) we plot both the train error and the Lagrange multiplier ν as a function of t/m. Again, as m grows, these curve converge to limit curves. This suggests the existence of the following limits

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(tm) = e_{\text{tr}}^{\text{mf3}}(t), \quad (\text{G.33})$$

$$\lim_{m \rightarrow \infty} \nu(tm) = \nu^{\text{mf3}}(t). \quad (\text{G.34})$$

Note that in this case, differently from the lazy initialization setting, the corresponding scaling function do not depend on the initialization parameter a0.

In order to characterize the limits in Eqs. [\(G.33\)](#page-53-2)-[\(G.34\)](#page-53-3), we proceed as in Sec. [F.1.4.](#page-39-0) Namely, in Fig. [23](#page-54-2) we plot the train error and the Lagrange multiplier ν as a function of the rescaled second layer weights γ = a(t)/ √ m. We also plot the asymptotic value of train error and Lagrange multiplier under the constrained GF dynamics in which second layer weigths are fixed to a(t) = γ √ m and do not evolve with time: e lz2 tr,<sup>∞</sup>(γ) := limt→∞ e tr (t, γ) and ν <sup>∞</sup>(γ) := limt→∞ ν lz2(t, γ). These are computed by integration of the scaling theory in Section [F.1.2.](#page-35-0)

The good collapse on these curves suggests to consider the the following construction, analogous to Sec. [F.1.4.](#page-39-0) Define the inverse function of <sup>t</sup> 7→ <sup>γ</sup> mf3(t), denoted by (γ mf3) −1 . Then, define

$$\begin{aligned} \varepsilon^{\text{mf3}}(\gamma) &= \varepsilon_{\text{tr}}^{\text{mf3}}((\gamma^{\text{mf3}})^{-1}(\gamma)), \\ \nu_*^{\text{mf3}}(\gamma) &= \nu^{\text{mf3}}((\gamma^{\text{mf3}})^{-1}(\gamma)). \end{aligned} \quad (\text{G.35})$$

![](_page_54_Figure_0.jpeg)

Figure 23: Train error, rescaled second-layer weights and the Lagrange multiplier ν(t) on timescales of order m under mean field initialization. Left Panel: parametric plot of the train error as a function of the weights of the second layer on the scale √ m, namely γ = a(t)/ √ m. The inset shows the same data on a logarithmic scale. Right Panel: same plot for the Lagrange multiplier ν. Data is the same as in Fig. [18.](#page-49-1)

Figure [23](#page-54-2) suggests that

$$\varepsilon^{\text{mf3}}(\gamma) \approx e_{\text{tr},\infty}^{\text{l2}}(\gamma), \quad (\text{G.36})$$

$$\text{mf3}(\gamma) \approx e_{\text{tr},\infty}^{\text{l2}}(\gamma)$$

$$\nu_*^{\text{mf3}}(\gamma) \approx \nu_{\text{tr},\infty}^{\text{lz2}}(\gamma). \quad (\text{G.37})$$

Equations [\(G.36\)](#page-54-3), [\(G.36\)](#page-54-3) imply that on timescales of order m the dynamics is adiabatic. For each incremental change of <sup>a</sup> on a the scale √ m, all one-time quantities relax to the asymptotic value which turns out to be the same as a constrained model with a(t)/ √ m = γ fixed.

The consequence of Eqs. [\(G.36\)](#page-54-3)-[\(G.36\)](#page-54-3) is that

$$\lim_{t \rightarrow \infty} \gamma^{\text{mf3}}(t) \approx \gamma_{\text{GF}}^*(\alpha, \tau). \quad (\text{G.38})$$

where γ ∗ GF(α, τ ) corresponds to the interpolation value of the initialization scale of a lazy model.

## G.2 Multi-index model

In this section we consider the case in which the dataset is distributed according to a multi-index model. For time scales beyond t = O(1), we will assume that h(z) = ˆφ(z). This simplifies the asymptotics for t large but of order one.

We identify two dynamical regimes emerging as <sup>m</sup> → ∞:

- t = O(1): a(t) = O(1) but is not constant. Also, the projection v(t) of first layer weights onto the latent space evolve as well as do train and test error. We further have etr(t) = ets(t) + om(1): there is no overfitting. This evolution is captured the mean field theory of [\[38,](#page-12-6) [14\]](#page-10-9) which we recover as <sup>m</sup> → ∞ limit of SymmDMFT .
- <sup>t</sup> = Θ(m): <sup>a</sup>(t) = Θ(√ m), v(t) decreases towards 0 and train and test error diverge. In this dynamical regime the network unlearns to a large extent the latent structure of the data and overfit it.

## G.2.1 First dynamical regime: t = Θ(1)

For timescales of order one, the SymmDMFT equations are solved, up to subleading terms as <sup>m</sup> → ∞, by the following ansatz

$$C_d(t, s) = C_d^{\text{mff}}(t, s) + o_m(1), \quad C_o(t, s) = C_o^{\text{mff}}(t, s) + o_m(1), \quad (\text{G.39})$$

$$R_d(t, s) = R_d^{\text{mfl}}(t, s) + o_m(1), \quad mR_o(t, s) = R_o^{\text{mfl}}(t, s) + o_m(1) \quad (\text{G.40})$$

$$v(t) = v^{\text{mfl}}(t) + o_m(1), \quad a(t) = a^{\text{mfl}}(t). \quad (\text{G.41})$$

![](_page_55_Figure_0.jpeg)

Figure 24: Gradient flow dynamics under mean field initialization in the first dynamical regime t = O(1). for data distributed according to a single index model. Curves are numerical solutions of the SymmDMFT equations: we plot v(t) and a(t) for different values of m and compare them to the mean field predictions. Data is distributed according to a single index model with ht(z) = ˆφ(z) = h(z) = (9/10)z + z <sup>3</sup>/6 with τ = 0.6 and α = 0.3.

![](_page_55_Figure_2.jpeg)

Figure 25: Evolution of the train and test error on different timescales under mean field initialization a(0) = 1. The train (solid curves) and test (dashed curves) errors as a function of time t (left panel) and scaled time t/m (right panel). Curves are numerical solutions of the SymmDMFT equations for h(z) = (9/10)z + z <sup>3</sup>/6, φˆ(z) = h(z), τ = 0.6 and α = 0.3. The arrow on the right panel corresponds to the asymptotic test error for a model with second layer weights fixed to the corresponding interpolation threshold.

The corresponding scaling equations are then given by

$$\begin{aligned} \partial_t R_o^{\text{mffl}}(t, t') &= -\nu^{\text{mffl}}(t) R_o^{\text{mffl}}(t, t') - \alpha a^{\text{mffl}}(t)^2 h'(C_o^{\text{mffl}}(t, t)) (R_d^{\text{mffl}}(t, t') + R_o^{\text{mffl}}(t, t')) , \\ \partial_t C_o^{\text{mffl}}(t, t') &= -\nu^{\text{mffl}}(t) C_o^{\text{mffl}}(t, t') + \alpha \langle \nabla \hat{\varphi}(\mathbf{v}^{\text{mffl}}(t)), \mathbf{v}^{\text{mffl}}(t') \rangle a^{\text{mffl}}(t) - \alpha a^{\text{mffl}}(t)^2 h'(C_o^{\text{mffl}}(t, t)) C_o^{\text{mffl}}(t, t') , \\ \nu^{\text{mffl}}(t) &= \alpha \langle \nabla \hat{\varphi}(\mathbf{v}^{\text{mffl}}(t)), \mathbf{v}^{\text{mffl}}(t) \rangle a^{\text{mffl}}(t) - \alpha a^{\text{mffl}}(t)^2 h'(C_o^{\text{mffl}}(t, t)) C_o^{\text{mffl}}(t, t) \\ \partial_t C_d^{\text{mffl}}(t, t') &= -\nu^{\text{mffl}}(t) C_d^{\text{mffl}}(t, t') + \alpha \langle \nabla \hat{\varphi}(\mathbf{v}^{\text{mffl}}(t)), \mathbf{v}^{\text{mffl}}(t') \rangle a^{\text{mffl}}(t) - \alpha a^{\text{mffl}}(t)^2 h'(C_o^{\text{mffl}}(t, t)) C_o^{\text{mffl}}(t, t') , \\ \partial_t R_d^{\text{mffl}}(t, t') &= -\nu^{\text{mffl}}(t) R_d^{\text{mffl}}(t, t') + \delta(t - t') , \\ \partial_t \mathbf{v}^{\text{mffl}}(t) &= -\nu^{\text{mffl}}(t) \mathbf{v}^{\text{mffl}}(t) + \alpha \nabla \hat{\varphi}(\mathbf{v}^{\text{mffl}}(t)) a^{\text{mffl}}(t) - \alpha a^{\text{mffl}}(t)^2 h'(C_o^{\text{mffl}}(t, t)) \mathbf{v}^{\text{mffl}}(t) , \\ \partial_t a^{\text{mffl}}(t) &= \alpha (\hat{\varphi}(\mathbf{v}^{\text{mffl}}(t)) - a^{\text{mffl}}(t) h(C_o^{\text{mffl}}(t, t))) . \end{aligned} \tag{G.42}$$

These equations are solved by setting:

$$C_o^{\text{mfl}}(t, t') = \langle \mathbf{v}^{\text{mfl}}(t), \mathbf{v}^{\text{mfl}}(t') \rangle \quad (\text{G.43})$$

![](_page_56_Figure_0.jpeg)

Figure 26: Finite width corrections. The 1/m corrections to the second layer weights and the projection on the latent space of the single index model on timescales of order 1. Dashed lines are obtained by integrating numerically Eqs. [\(G.56\)](#page-57-0) to [\(G.59\)](#page-57-1) determining the limits <sup>m</sup> → ∞. Here, φˆ(z) = h(z) = (9/10)z + z <sup>3</sup>/6 with τ = 0.6 and α = 0.3.

with v mf1(t), a mf1(t) the solution of

$$\begin{aligned} \partial_t \mathbf{v}^{\text{mfl}}(t) &= \alpha a^{\text{mfl}}(t) (\mathbf{I}_k - \mathbf{v}^{\text{mfl}}(t) \mathbf{v}^{\text{mfl}}(t)^\top) (\nabla \hat{\varphi}(\mathbf{v}^{\text{mfl}}(t)) - a^{\text{mfl}}(t) h'(\|\mathbf{v}^{\text{mfl}}(t)\|^2) \mathbf{v}^{\text{mfl}}(t)), \\ \partial_t a^{\text{mfl}}(t) &= \alpha (\hat{\varphi}(\mathbf{v}^{\text{mfl}}(t)) - a^{\text{mfl}}(t) h(\|\mathbf{v}^{\text{mfl}}(t)\|^2)), \end{aligned} \tag{G.44}$$

with initial conditions given by v mf1(0) = 0 and a mf1(0) = a0.

Equations [\(G.44\)](#page-56-1) coincide with the mean field theory of [\[38,](#page-12-6) [14,](#page-10-9) [45\]](#page-12-8), when the latter are specialized to the multi-index model studied here, under symmetric initializations [\[10\]](#page-10-12). (See also [\[2\]](#page-10-13).) Using the ansatz of Eqs. [\(G.39\)](#page-54-4) to [\(G.41\)](#page-54-5) in the formulas for training and test error [\(C.45\)](#page-28-3), [\(C.46\)](#page-28-4), we get

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t) = \lim_{m \rightarrow \infty} e_{\text{ts}}(t) = e^{\text{mfl}}(t), \quad (\text{G.45})$$

with

$$e^{\text{mfl}}(t) = \frac{1}{2} \left[ \tau^2 + \|\varphi\|^2 - 2a^{\text{mfl}}(t)\hat{\varphi}(\mathbf{v}^{\text{mfl}}(t)) + a^{\text{mfl}}(t)^2 h(\|\mathbf{v}^{\text{mfl}}(t)\|^2) \right]. \quad (\text{G.46})$$

A particularly simple case is the one in which k = 1 (single index model) and φ = σ (whence φˆ = h). For a class of such activations with h ′ (0) > 0, we have a mf1(t), vmf1(t) → <sup>1</sup> as <sup>t</sup> → ∞ and therefore

$$\lim_{t \rightarrow \infty} e^{\text{mfl}}(t) = \frac{\tau^2}{2}. \quad (\text{G.47})$$

In other words, neurons align perfectly with latent direction, the generalization error vanishes, and and train and test error converge for large constant t to the Bayes error τ <sup>2</sup>/2.

In Fig. [24](#page-55-0) we compare the solution of Eqs. [\(G.44\)](#page-56-1) with the numerical integrations of the SymmDMFT equations for a range of values of m. As m increases, the SymmDMFT solutions converge to the asymptotic predictions v mf1(t), a mf1(t), confirming the above ansatz.

Similarly, in Fig. [25-](#page-55-1)left panel we compute the train and test error by solving the SymmDMFT equations and compare the results to the asymptotic prediction provided by Eq. [\(G.46\)](#page-56-2). We observe that –as predicted– train and test error match on an increasingly long time interval. At a certain point, they diverge: we will next characterize the timescale on which this happens.

#### G.2.2 Escape from the mean field dynamical regime

In order to understand on which time scale the dynamics diverges from mean field theory described above, we will study small deviations from this theory. We expect that these deviations will diverge with time. Characterizing this divergence will allow to determine time scale on which we exit the mean field regime.

We focus on the case of a single index model k = 1, with φˆ = h, and set a(0) = 1. We believe that the qualitative conclusions obtained in this case apply more generally. We also assume h to be such that the long time asymptotics of mean field dynamical solutions is

$$\lim_{t \rightarrow \infty} a^{\text{mfl}}(t) = 1, \quad \lim_{t \rightarrow \infty} v^{\text{mfl}}(t) = 1. \quad (\text{G.48})$$

As mentioned in the previous section, this holds for a broad class of activations. In other words, for time t large and yet of order one, the neurons are very well aligned.

We next study the corrections to the mean field solution. We claim that such corrections are of order 1/m and define the functions a˜(t), v˜(t), dots ,R˜ o(t, t′ ), via

$$m(a(t) - a^{\text{mfl}}(t)) = \tilde{a}(t) + o_m(1), \quad (\text{G.49})$$
(G.50)

$$m(v(t) - v^{\text{mfl}}(t)) = \tilde{v}(t) + o_m(1), \quad (\text{G.50})$$

$$m(C_d(t, t') - C_d^{\text{mfl}}(t, t')) = \tilde{C}_d(t, t') + o_m(1), \quad (\text{G.51})$$

$$- (C_d(t, t') - C_d^{\text{mfl}}(t, t')) - \tilde{C}_d(t, t') - o_m(1), \quad (\text{G.52})$$

$$m(C_o(t, t') - C_o^{\text{mfl}}(t, t')) = \tilde{C}_o(t, t') + o_m(1), \quad (\text{G.52})$$

$$m(R_d(t, t') - R_d^{\text{mff}}(t, t')) = \tilde{R}_d(t, t') + o_m(1), \quad (\text{G.53})$$

$$m(R_o(t, t') - R_o^{\text{mfl}}(t, t')) = \tilde{R}_o(t, t') + o_m(1), \quad (\text{G.54})$$

$$((())) \quad \text{mfl}((())) \quad \tilde{z}(()) \quad ((())) \quad (\text{G.55})$$

$$m(\nu(t) - \nu^{\text{mfl}}(t)) = \tilde{\nu}(t) + o_m(1). \quad (\text{G.55})$$

Substituting the above form into the SymmDMFT equations and matching the next-to-leading order in m we can obtain the equations for the 1/m corrections. It turns out that equations for a, ˜ v, ˜ C˜ o and ν˜ decouple from the equations for C˜ <sup>d</sup>, R˜ <sup>d</sup> and R˜ <sup>o</sup>. Given that we are interested in the former quantities we only report the corresponding equations:

$$\begin{aligned}
 \frac{d\tilde{a}(t)}{dt} &= \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) \tilde{v}(t) - \alpha \hat{\varphi}(v^{\text{mfl}}(t)) \int_0^t \Sigma_R^{(1)}(t, s) ds - \alpha a^{\text{mfl}}(t) [h(1) - h(C_o^{\text{mfl}}(t, t))] \\
 &\quad + \alpha \int_0^t \Sigma_R^{(1)}(t, s) a^{\text{mfl}}(s) h(C_o^{\text{mfl}}(t, s)) ds - \alpha \tilde{a}(t) h(C_o^{\text{mfl}}(t, t)) - \alpha a^{\text{mfl}}(t) h'(C_o^{\text{mfl}}(t, t)) \tilde{C}_o(t, t) \\
 &\quad - \alpha \int_0^t C_A^{\text{mfl}}(t, s) a^{\text{mfl}}(s) [h'(C_d^{\text{mfl}}(t, s)) R_d^{\text{mfl}}(t, s) + h'(C_o^{\text{mfl}}(t, s)) R_o^{\text{mfl}}(t, s)] ds , \\
 \frac{d\tilde{v}(t)}{dt} &= -\nu^{\text{mfl}}(t) \tilde{v}(t) - \tilde{\nu}(t) v^{\text{mfl}}(t) + \alpha \hat{\varphi}(v^{\text{mfl}}(t)) \tilde{a}(t) + \alpha \hat{\varphi}''(v^{\text{mfl}}(t)) \tilde{v}(t) a^{\text{mfl}}(t) \\
 &\quad - \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) a^{\text{mfl}}(t) \int_0^t \Sigma_R^{(1)}(t, s) ds - \int_0^t [\tilde{M}_R^{(d)}(t, s) - M_{R,o}^{(0)}(t, s)] v^{\text{mfl}}(s) ds \\
 &\quad - \int_0^t [M_{R,o}^{(1)}(t, s) v^{\text{mfl}}(s) + M_{R,o}^{(0)}(t, s) \tilde{v}(s)] ds , \\
 \tilde{\nu}(t) &= \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) \tilde{v}(t) a^{\text{mfl}}(t) + \alpha \hat{\varphi}''(v^{\text{mfl}}(t)) v^{\text{mfl}}(t) \tilde{v}(t) a^{\text{mfl}}(t) \\
 &\quad + \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) v^{\text{mfl}}(t) \tilde{a}(t) - \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) v^{\text{mfl}}(t) \int_0^t \Sigma_R^{(1)}(t, s) ds \\
 &\quad - \int_0^t [\tilde{M}_R^{(d)}(t, s) C_d^{\text{mfl}}(t, s) - M_{R,o}^{(0)}(t, s) C_o^{\text{mfl}}(t, s)] ds \\
 &\quad - \int_0^t [M_{R,o}^{(1)}(t, s) C_o^{\text{mfl}}(t, s) + M_{R,o}^{(0)}(t, s) \tilde{C}_o(t, s)] ds \\
 &\quad - \int_0^t [\tilde{M}_C^{(d)}(t, s) R_d^{\text{mfl}}(t, s) + M_{C,o}^{(0)}(t, s) R_o^{\text{mfl}}(t, s)] ds , \\
 \frac{\partial \tilde{C}_o(t, t')}{\partial t} &= -\nu^{\text{mfl}}(t) \tilde{C}_o(t, t') - \tilde{\nu}(t) C_o^{\text{mfl}}(t, t') + \alpha \hat{\varphi}''(v^{\text{mfl}}(t)) \tilde{v}(t) v^{\text{mfl}}(t') a^{\text{mfl}}(t) \\
 &\quad + \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) \tilde{v}(t') a^{\text{mfl}}(t) + \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) v^{\text{mfl}}(t') \tilde{a}(t) \\
 &\quad - \alpha \hat{\varphi}'(v^{\text{mfl}}(t)) v^{\text{mfl}}(t') a^{\text{mfl}}(t) \int_0^t \Sigma_R^{(1)}(t, s) ds \\
 &\quad - \int_0^t [\tilde{M}_R^{(d)}(t, s) C_o^{\text{mfl}}(t', s) + M_{R,o}^{(0)}(t, s) C_d^{\text{mfl}}(t', s) - 2M_{R,o}^{(0)}(t, s) C_o^{\text{mfl}}(t', s)] ds
 \end{aligned}
 \tag{G.56}
 \tag{G.59}$$

$$\begin{aligned} & - \int_0^t \left[ M_{R,o}^{(0)}(t,s) \tilde{C}_o(t',s) + M_{R,o}^{(1)}(t,s) C_o^{\text{mfl}}(t',s) \right] ds \\ & - \int_0^{t'} \left[ \tilde{M}_C^{(d)}(t,s) R_d^{\text{mfl}}(t',s) + M_{C,o}^{(0)}(t,s) R_o^{\text{mfl}}(t',s) \right] ds. \end{aligned}$$

Here, we used the following auxiliary functions

$$\Sigma_R^{(1)}(t, s) = a^{\text{mfl}}(t)a^{\text{mfl}}(s) \left[ h'(C_d^{\text{mfl}}(t, s))R_d^{\text{mfl}}(t, s) + h'(C_o^{\text{mfl}}(t, s))R_o^{\text{mfl}}(t, s) \right], \quad (\text{G.60})$$

$$\begin{aligned}\Sigma_R^{(1)}(t, s) &= a^{\text{mfl}}(t)a^{\text{mfl}}(s) [h'(C_d^{\text{mfl}}(t, s))R_d^{\text{mfl}}(t, s) + h'(C_o^{\text{mfl}}(t, s))R_o^{\text{mfl}}(t, s)], & (\text{G.60}) \\ C_A^{\text{mfl}}(t, s) &= - [\tau^2 + h_t(1) - a^{\text{mfl}}(t)\varphi(v^{\text{mfl}}(t)) - a^{\text{mfl}}(s)\varphi(v^{\text{mfl}}(s)) + a^{\text{mfl}}(t)a^{\text{mfl}}(s)h(C_o^{\text{mfl}}(t, s))], \\ & \quad (\text{G.61})\end{aligned}$$

$$\begin{aligned} \tilde{M}_R^{(d)}(t, s) &= \alpha a^{\text{mfl}}(t) a^{\text{mfl}}(s) [h'(1)\delta(t-s) + C_A^{\text{mfl}}(t, s)h''(C_d^{\text{mfl}}(t, s))R_d^{\text{mfl}}(t, s)] , & (\text{G.62}) \\ M_{R,o}^{(0)}(t, s) &= \alpha(a^{\text{mfl}}(t))^2 h'(C_o^{\text{mfl}}(t, s))\delta(t-s) , & (\text{G.63}) \end{aligned}$$

$$M_{R_o^{(0)}}^{(0)}(t, s) = \alpha(a^{\text{mfl}}(t))^2 h'(C_o^{\text{mfl}}(t, s)) \delta(t - s) , \quad (\text{G.63})$$

$$M_{R,o}^{(1)}(t, s) = \alpha \left[ 2a^{\text{mfl}}(t)\tilde{a}(t)h'(C_o^{\text{mfl}}(t, t)) + (a^{\text{mfl}}(t))^2h''(C_o^{\text{mfl}}(t, t))\tilde{C}(t, t) \right] \delta(t - s) \quad (\text{G.64})$$

$$- \alpha a^{\text{mfl}}(t)a^{\text{mfl}}(s) \sum_{i=1}^{(1)} (t, s) h'(C^{\text{mfl}}(t, s)) \quad (\text{G.65})$$

$$\begin{aligned} & -\alpha a^{\text{mfl}}(t)a^{\text{mfl}}(s)\Sigma_R^{(1)}(t,s)h'(C_o^{\text{mfl}}(t,s)) \\ & + \alpha a^{\text{mfl}}(t)a^{\text{mfl}}(s)C_A^{\text{mfl}}(t,s)h''(C_o^{\text{mfl}}(t,s))R_o^{\text{mfl}}(t,s), \end{aligned} \quad (\text{G.65})$$

$$-\alpha a^{\text{mfl}}(t)a^{\text{mfl}}(s)\Sigma_R^{(1)}(t,s)h'(C_o^{\text{mfl}}(t,s)) \quad (\text{G.65})$$

$$\tilde{M}_C^{(d)}(t, s) = \alpha a^{\text{mfl}}(t) a^{\text{mfl}}(s) C_A^{\text{mfl}}(t, s) h'(C_d^{\text{mfl}}(t, s)), \quad (\text{G.67})$$

$$M_{C_o^0}^{(0)}(t, s) = \alpha a^{\text{mfl}}(t) a^{\text{mfl}}(s) C_A^{\text{mfl}}(t, s) h'(C_o^{\text{mfl}}(t, s)). \quad (\text{G.68})$$

Note that Eqs. [\(G.56\)](#page-57-0) to [\(G.59\)](#page-57-1) are a set of four integral-differential equations for the four functions a˜(t), v˜(t), ν˜(t), C˜ o(t, t′ ). The original SymmDMFT equations involve three other functions: C˜ d(t, t′ ), R˜ d(t, t′ ), R˜ o(t, t′ )? We also remark that: (i) These equations are linear in the unknowns a˜(t), v˜(t), ν˜(t), C˜ o(t, t′ ); (ii) They can be integrated numerically with the same strategy used to integrate the SymmDMFT equations.

In Fig. [26](#page-56-3) we plot the deviations from the mean field limit <sup>m</sup>(a(t) − <sup>a</sup> mf1(t)) and <sup>m</sup>(v(t) − <sup>v</sup> mf1(t)) as a function of time t, as obtained by solving the SymmDMFT equations[<sup>1</sup>](#page-61-0) , for several values of m. We also plot the predicted limits a˜(t), v˜(t), which are obtained by integrating Eqs. [\(G.56\)](#page-57-0) to [\(G.59\)](#page-57-1) As m gets large, the finite-m curves appear to converge to the predictions a˜(t), v˜(t).

In Figure [27](#page-59-0) we plot the result of integrating Eqs. [\(G.56\)](#page-57-0) to [\(G.59\)](#page-57-1) over a wider time window. We observe that v˜, a˜, ν˜ and C˜ <sup>o</sup>(t, t) diverge linearly with t.

This suggests the following asymptotics for these corrections

$$\lim_{t \rightarrow \infty} \frac{\tilde{a}(t)}{t} = a_*, \quad \lim_{t \rightarrow \infty} \frac{\tilde{v}(t)}{t} = v_*, \quad (\text{G.69})$$

$$\lim_{t \rightarrow \infty} \frac{\tilde{\nu}(t)}{t} = \nu_*, \quad \lim_{t \rightarrow \infty} \frac{\tilde{C}_0(t, t)}{t} = c_*. \quad (\text{G.70})$$

The values of the constant a∗, v∗, ν<sup>∗</sup> and c<sup>∗</sup> can be obtained analytically by using the above ansatz in Eqs. [\(G.56\)](#page-57-0) to [\(G.59\)](#page-57-1). We obtain that they solve the following linear equations

$$0 = \varphi'(1)v_* + \varphi(1)a_*, \quad (\text{G.71})$$

$$0 = \hat{\phi}'(1)c_* + 2\hat{\phi}(1)a_*, \quad (\text{G.72})$$

$$0 = -\phi'(1)\nu_* - \phi'(1)(\alpha\phi''(1) - \alpha\phi'(1) - \alpha(\phi'(1))^2)a_* + 2\alpha\phi(1)\phi''(1), \quad (\text{G.73})$$

$$0 = -\frac{1}{2}c_* - \nu_1 c_* - 2\nu_* v_1 + 4v_1 \alpha \tau^2, \quad (\text{G.74})$$

$$0 = -\frac{1}{2}c_* - \nu_1 c_* - 2\nu_* v_1 + 4v_1 \alpha \tau^2, \quad (\text{G.74})$$

where

$$v_1 := \lim_{t \rightarrow \infty} (v(t) - 1)t, \quad (\text{G.75})$$

$$\nu_1 := \lim_{t \rightarrow \infty} \tilde{\nu}(t)t. \quad (\text{G.76})$$

The asymptotic linear behavior predicted by Eqs. [\(G.69\)](#page-58-0), [\(G.70\)](#page-58-1), with the coefficients determined by Eqs. [\(G.71\)](#page-58-2)-[\(G.74\)](#page-58-3) is plotted in Fig. [27.](#page-59-0) We observe good agreement with the numerical integration of Eqs. [\(G.56\)](#page-57-0) to [\(G.59\)](#page-57-1).

![](_page_59_Figure_0.jpeg)

Figure 27: Finite width corrections to dynamical observables under mean field initialization. The 1/m corrections to v(t), a(t), Co(t, t) and ν˜(t) as a function of time as extracted from the numerical integration of the corresponding equations. The dashed lines are the asymptotic predictions for <sup>t</sup> → ∞ which show that the divergence of all quantities is linear with time. Here, <sup>φ</sup>ˆ(z) = <sup>h</sup>(z) = (9/10)z + z <sup>3</sup>/6 with τ = 0.6 and α = 0.3.

![](_page_59_Figure_2.jpeg)

Figure 28: econd layer weights and projection on of the first layer weigths onto the latent structure of the data for gradient flow under mean field initialization on timescales of order m. Left: rescaled second layer weights a(t)/m as a function of the rescaled time t/m. The arrow on the right points at the threshold γ ∗ GF(α, φ, τ ) for interpolation under gradient flow, see Section [F.2.3.](#page-44-2) Right: projection of the first layer weights on the latent space in the single index model as a function of rescaled time t/m. Here, φˆ(z) = h(z) = (9/10)z + z <sup>3</sup>/6 with τ = 0.6 and α = 0.3. v = 1/γ in [\(F.57\)](#page-42-6).

![](_page_60_Figure_0.jpeg)

Figure 29: Parametric plot of the rescaled projection onto the latent direction v √ m against rescaled second layer weights <sup>γ</sup> <sup>=</sup> a/√ m. Same data as in Fig. [28.](#page-59-1) Dashed line is v √ m = 1/γ.

![](_page_60_Figure_2.jpeg)

Figure 30: Train and test error of gradient flow under mean field initialization, for increasing values of m. Left: train error as a function of rescaled weights a(t)/ √ m. Dashed line is the Bayes error τ <sup>2</sup>/2. Curves are traversed in time from top to bottom. Right: test error versus train error. Curves are traversed in time from right to left. Here φˆ(z) = h(z) = (9/10)z + z <sup>3</sup>/6 with τ = 0.6 and α = 0.3.

![](_page_60_Figure_4.jpeg)

Figure 31: The difference between test and train error for the single index data. Left panel: the difference between test and train error plotted as a function of a/√ m and compared to what is obtained from a model with fixed second layer weights initialized with Lazy scaling. Right panel: the difference between test and train error on timescales of order m. Here, φˆ(z) = h(z) = (9/10)z + z <sup>3</sup>/6 with τ = 0.6 and α = 0.3.

The above analysis implies that (considering to be definite second layer weights, and projection of first layer weigths onto the latent direction), for <sup>m</sup> ≫ <sup>1</sup>, <sup>t</sup> ≫ <sup>1</sup>,

$$a(t) = a^{\text{mfl}}(t) + \frac{1}{m}(a_*t + o(t)) + \frac{1}{m}\Delta_a(t, m), \quad (\text{G.77})$$

$$v(t) = v^{\text{mfl}}(t) + \frac{1}{m}(v_*t + o(t)) + \frac{1}{m}\Delta_v(t, m) \quad (\text{G.78})$$

$$v(t) = v^{\text{ffl}}(t) + \frac{1}{m}(v_*t + o(t)) + \frac{1}{m}\Delta_v(t, m), \quad (\text{G.78})$$

where limm→∞ ∆a/v(t, m) = 0. If we neglect the error terms, and assume that this expression holds for t larger than O(1) in m, then it indicates that a(t), v(t) differ significantly from the mean field prediction when t/m becomes of order one. We expect therefore a third dynamical regime for t = Θ(m), which will be the object of the next section.

## G.2.3 Second dynamical regime: t = Θ(m) and beyond

As pointed out at the end of the previous section, we expect a third dynamical regime when t = Θ(m). By this time, the stability calculation in the previous section indicates that second layer weights become of order √ m. Figure [28](#page-59-1) confirms this, and shows that, in the same regime v(t) becomes small. In fact, numerical solution of the SymmDMFT equations are consistent with <sup>a</sup>(t) = Θ(√ m), v(t) = Θ(1/ √ <sup>m</sup>), and <sup>a</sup>(t)v(t) ≈ <sup>1</sup> for <sup>t</sup> = Θ(m).

For a small constant c denote by t0(m; c) the time at which a(t0(m; c)) = c √ m. We then expect that the following exists

$$\lim_{m \rightarrow \infty} \frac{a(t_0(m; c) + \theta w(m))}{\sqrt{m}} = \gamma^{\text{mf3}}(\theta), \quad (\text{G.79})$$

$$\lim_{m \rightarrow \infty} \mathbf{v}(t_0(m; c) + \theta w(m)) \sqrt{m} = \mathbf{v}^+(\theta), \quad (\text{G.80})$$

provided w(m) is a suitable function (with w(m) = O(t0(m; c))). The stability analysis in the previous section suggests that <sup>t</sup>0(m; <sup>c</sup>) ≤ <sup>t</sup>∗(c)<sup>m</sup> <sup>+</sup> <sup>o</sup>(m). Our numerical solutions do not cover a large enough range of values of m to verify this ansatz, and determine the scaling of w(m) with m. On the other hand, they indicate that indeed t0(m; c) = Θ(m).

Since the second layer weights become of order √ m in this dynamical regime, train and test error start to differ significantly. We expect

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t_0(m; c) + \theta w(m)) = e_{\text{tr}}^{\text{mf3}}(\theta), \quad (\text{G.81})$$

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t_0(m; c) + \theta w(m)) = e_{\text{ts}}^{\text{mf3}}(\theta) . \quad (\text{G.82})$$

This picture is confirmed by Fig. [30,](#page-60-0) which reports train and test error as predicted by numerical solutions of the SymmDMFT equations for increasing values of m. On the left, we plot the train error as a function of the rescaled second layer weights <sup>γ</sup> <sup>=</sup> a/√ m. We observe that curves for different values of m decrease until they reach the Bayes error τ 2 . On this phase however different curves do not collapse corresponding to the fact that γ vanishes. In the second phase, γ grows to be of order one and correspondingly the train error decreases below the Bayes error: this is the third dynamical regime. Overfitting takes place at this point.

In the right frame of Fig. [30,](#page-60-0) we plot test error versus train error. We observe, again, the two phases emerging for large m. In the first phase train error and test error are closely matched. In the second phase, train error decreases and test error correspondingly increases. Again, this takes place when t = Θ(m).

Finally, in Fig. [31,](#page-60-1) we repeat similar plots for the generalization error (difference between test and train error).

When t/m is large, the train error vanishes. We observe from Figure [28,](#page-59-1) left frame that, as <sup>t</sup> → ∞, rescaled second layer weights reach a finite limit that is close to the interpolation threshold characterized in Section [F.2.3.](#page-44-2) Namely

$$\lim_{\tau \rightarrow \infty} \gamma^+(\theta) \approx \gamma_{\text{GF}}^*(\alpha, \varphi, \tau). \quad (\text{G.83})$$

<sup>1</sup>We note that solving the SymmDMFT equations accurately enough to capture these corrections requires either to use very fine discretization, or a higher-order integration method.

![](_page_62_Figure_0.jpeg)

Figure 32: The interpolation transition for pure noise data and a network with second layer weights that do not evolve with time, fixed at a = 1, see Section [H.](#page-62-0) The noise level is fixed to τ = 1 and we considered h(z) = (3/10)z + z <sup>2</sup>/2. Top left panel: relaxation time ((rate for convergence to vanishing error) for different values of m. Top right panel: logarithmic plot of the relaxation time. The value of the algorithmic threshold for different values of m is a fitting parameter. Bottom left panel: values of the algorithmic thresholds as a function of m. Bottom right panel: the relaxation time as extracted from the scaling limit of the SymmDMFT equations in the <sup>m</sup> → ∞ limit. The algorithmic threshold is in this case <sup>α</sup>GF(∞) ≈ <sup>1</sup>.<sup>18</sup> which fits well the behavior plotted in the left bottom plot.

## H Dynamics under mean field initialization for n/d = α fixed

## H.1 Interpolation threshold at fixed a(t) = a<sup>0</sup>

In this section, we consider an alternative scaling in the large width limit. As before, we use the SymmDMFT equations, and therefore study the limit n, d → ∞ with n/d → <sup>α</sup>. In the previous sections we studied the large width limit <sup>m</sup> → ∞ with <sup>α</sup> <sup>=</sup> α/m fixed. In that setting interpolation is only possible when the network complexity scales, i.e. second-layer weights are <sup>a</sup> = Θ(√ m)

Here instead we keep a(t) = 1 and do not let evolve second-layer weights with GF. We consider pure noise data, and show that interpolation takes place if α < αGF(m), while the train error remains bounded away from zero for α > αGF(m). As expected from Gaussian complexity considerations, the threshold <sup>α</sup>GF(m) has a finite limit as <sup>m</sup> → ∞. In particular, for any α > <sup>0</sup>, a network with <sup>a</sup> bounded cannot interpolate pure noise data.

As thorough in Sec[.F](#page-32-1) we fix α and integrate numerically the SymmDMFT equations for finite but increasing values of m. We fix the initialization scale a<sup>0</sup> and the noise level τ and change only α.

We observe that for α small enough the train error decreases exponentially fast to zero. Namely, recalling that <sup>e</sup>tr(t; <sup>α</sup>) := limn,d→∞ <sup>R</sup>b <sup>n</sup>(a(t),W(t)), we have that

$$\bar{\alpha} < \bar{\alpha}_{\text{GF}}(m) \quad \Rightarrow \quad e_{\text{tr}}(t; \bar{\alpha}) = \exp\{-t/t_{\text{rel}}^*(\bar{\alpha}, m) + o(t)\}. \quad (\text{H.1})$$

However, the relaxation time time t ∗ rel(α, m) increases as <sup>α</sup> ↑ <sup>α</sup>GF(m). Concretely, we define <sup>t</sup>rel(α, m, c) as the infimum time such that <sup>e</sup>tr(t; <sup>α</sup>) ≤ <sup>c</sup>, where <sup>c</sup> is some small constant. In practice, we set c = 10<sup>−</sup><sup>7</sup> . The results are plotted as a function of α for several values of m in Fig. [32,](#page-62-1) top left plot.

![](_page_63_Figure_0.jpeg)

Figure 33: heck of the convergence of the numerical solution of the SymmDMFT for α fixed to the scaling solution for <sup>m</sup> → ∞. The left panel shows the behavior of the train error while the right panel shows the behavior of the correlation Cd(t, 0). Both panels refer to a model where the teacher is pure noise with τ = 1 and the student is made of of neurons whose covariance structure is given by h(z) = (3/10)z + z <sup>2</sup>/2.

For each value of m the relaxation time appears to diverge at the critical point αGF(m) as an inverse power of <sup>α</sup>GF(m) − <sup>α</sup>, namely:

$$\bar{\alpha} \uparrow \bar{\alpha}_{\text{GF}}(m) \quad \Rightarrow \quad t_{\text{rel}}(\bar{\alpha}, m, c) = \frac{L(m, c)}{(\bar{\alpha}_{\text{GF}}(m) - \bar{\alpha})^\nu} (1 + o(1)). \quad (\text{H.2})$$

The exponent ν appears to be independent of m. We fit this form to our data and extract the interpolation thresholds αrel(m). In Fig. [32,](#page-62-1) top right, we plot trel(α, m, c)/m as a function of the gap to this threshold. This plot confirms the form [\(H.2\)](#page-63-0), with exponent <sup>ν</sup> ≈ <sup>2</sup>. Also, the fact that different curves superimpose indicate that <sup>L</sup>(m, c) ≈ <sup>L</sup>∗(c)m.

The estimated interpolation thresholds αGF(m) are plotted as a function of m in the bottom left of Fig. [32.](#page-62-1) These data are consistent with the existence of a finite limit

$$\bar{\alpha}_{\text{GF}}(\infty) = \lim_{m \rightarrow \infty} \bar{\alpha}_{\text{GF}}(m), \quad (\text{H.3})$$

and numerically <sup>α</sup>GF(∞) ≈ <sup>1</sup>.18.

In the next subsection, we derive equations describing the <sup>m</sup> → ∞ limit for <sup>α</sup> <sup>=</sup> <sup>O</sup>(1), <sup>a</sup> <sup>=</sup> <sup>O</sup>(1) fixed. Studying these equations yields further support to Eq. [\(H.3\)](#page-63-1).

## H.2 Infinite width limit at fixed α

In order to study the limit <sup>m</sup> → ∞ at fixed <sup>α</sup>, we discuss the limit of the SymmDMFT equations when <sup>m</sup> → ∞. As we have seen previously, the relaxation time of the train error is proportional to <sup>m</sup>. This is clearly visible in Fig. [32-](#page-62-1)top/left. This suggests that for <sup>m</sup> → ∞, dynamics takes place on timescales of order m. Therefore we propose the following asymptotic ansatz

$$mR_o(tm, sm) = \tilde{R}_o^{\bar{\alpha}}(t, s) + o_m(1), \quad C_o(tm, sm) = \tilde{C}_o^{\bar{\alpha}}(t, s) + o_m(1), \quad (\text{H.4})$$

$$R_d(tm, sm) = \tilde{R}_d^{\bar{c}}(t, s) + o_m(1), \quad C_d(tm, sm) = \tilde{C}_d^{\bar{c}}(t, s) + o_m(1), \quad (\text{H.5})$$

$$m\nu(tm) = \tilde{\nu}^{\bar{\alpha}}(t) + o_m(1), \quad (\text{H.6})$$

which defines a set of functions, R˜<sup>α</sup> d , C˜<sup>α</sup> d , R˜<sup>α</sup> o , C˜<sup>α</sup> o and ν˜ <sup>α</sup>. We now describe the equations that these scaling functions satisfy satisfy. First we define C˜<sup>α</sup> <sup>A</sup> and R˜<sup>α</sup> <sup>A</sup> as the solution of

$$\begin{aligned}\delta(t - t') &= \int_{t'}^t \left[ \delta(t - s) + \tilde{\Sigma}_R(t, s) \right] \tilde{R}_A^{\bar{\alpha}}(s, t' ds, ) \\ 0 &= \int_0^t \left[ \delta(t - s) + \tilde{\Sigma}_R(t, s) \right] \tilde{C}_A^{\bar{\alpha}}(t', s) ds + \int_0^{t'} ds \tilde{\Sigma}_C(t, s) \tilde{R}_A^{\bar{\alpha}}(t', s) ds,\end{aligned}\tag{H.7}$$

where

$$\begin{aligned}\tilde{\Sigma}_R(t, s) &= h'(\tilde{C}_d^{\bar{\alpha}}(t, s))\tilde{R}_d^{\bar{\alpha}}(t, s) + h'(\tilde{C}_o^{\bar{\alpha}}(t, s))\tilde{R}_o^{\bar{\alpha}}(t, s) \\ \tilde{\Sigma}_C(t, s) &= \tau^2 + h(\tilde{C}_o^{\bar{\alpha}}(t, s)).\end{aligned}\tag{H.8}$$

Then we define the limit memory kernels:

$$\begin{aligned}\tilde{M}_R^{(d)}(t, s) &= \bar{\alpha}\tilde{C}_A(t, s)h''(\tilde{C}_d^{\bar{\alpha}}(t, s))\tilde{R}_d^{\bar{\alpha}}(t, s), \\ \tilde{M}_C^{(d)}(t, s) &= \bar{\alpha}\tilde{C}_A^{\bar{\alpha}}(t, s)h'(\tilde{C}_d^{\bar{\alpha}}(t, s)), \\ \tilde{M}_R^{(o)}(t, s) &= \bar{\alpha} \left[ \tilde{C}_A(t, s)h''(\tilde{C}_o^{\bar{\alpha}}(t, s))\tilde{R}_o^{\bar{\alpha}}(t, s) + \tilde{R}_A^{\bar{\alpha}}(t, s)h'(\tilde{C}_o^{\bar{\alpha}}(t, s)) \right], \\ \tilde{M}_C^{(o)}(t, s) &= \bar{\alpha}\tilde{C}_A^{\bar{\alpha}}(t, s)h'(\tilde{C}_o^{\bar{\alpha}}(t, s)).\end{aligned}\tag{H.9}$$

Substituting the above ansatz in the SymmDMFT equations and matching the leading order terms, we get the following equations that determine R˜<sup>α</sup> d , C˜<sup>α</sup> d , R˜<sup>α</sup> o , C˜<sup>α</sup> o and ν˜ α:

$$\begin{aligned} \partial_t \tilde{C}_d^{\bar{\alpha}}(t, t') &= -\tilde{\nu}^{\bar{\alpha}}(t) \tilde{C}_d^{\bar{\alpha}}(t, t') - \int_0^t \left[ \tilde{M}_R^{(d)}(t, s) \tilde{C}_d^{\bar{\alpha}}(t', s) + \tilde{M}_R^{(o)}(t, s) \tilde{C}_o^{\bar{\alpha}}(t', s) \right] ds \\ &\quad - \int_0^{t'} \left[ \tilde{M}_C^{(d)}(t, s) \tilde{R}_d^{\bar{\alpha}}(t', s) + \tilde{M}_C^{(o)}(t, s) \tilde{R}_o^{\bar{\alpha}}(t', s) \right] ds, \end{aligned} \quad (\text{H.10})$$

$$\begin{aligned} \partial_t \tilde{C}_o^{\bar{\alpha}}(t, t') &= -\tilde{\nu}^{\bar{\alpha}}(t) \tilde{C}_o^{\bar{\alpha}}(t, t') - \int_0^t \left[ \tilde{M}_R^{(d)}(t, s) + \tilde{M}_R^{(o)}(t, s) \right] \tilde{C}_o^{\bar{\alpha}}(t', s) \, \mathrm{d}s \\ &\quad - \int_0^{t'} \left[ \tilde{R}_o^{\bar{\alpha}}(t', s) + \tilde{R}_d^{\bar{\alpha}}(t', s) \right] \tilde{M}_C^{(o)}(t, s) \, \mathrm{d}s, \end{aligned} \quad (\text{H.11})$$

$$\partial_t \tilde{R}_d^{\bar{\alpha}}(t, t') = -\tilde{\nu}^{\bar{\alpha}}(t) \tilde{R}_d^{\bar{\alpha}}(t, t') + \delta(t - t') - \int_{t'}^t ds \tilde{M}_R^{(d)}(t, s) \tilde{R}_d^{\bar{\alpha}}(s, t'), \quad (\text{H.12})$$

$$\begin{aligned} \partial_t \tilde{R}_o^{\bar{\alpha}}(t, t') = & -\tilde{\nu}^{\bar{\alpha}}(t) \tilde{R}_o^{\bar{\alpha}}(t, t') - \int_{t'}^t \left[ \tilde{M}_R^{(d)}(t, s) \tilde{R}_o^{\bar{\alpha}}(s, t') + \tilde{M}_R^{(o)}(t, s) \tilde{R}_d^{\bar{\alpha}}(s, t') \right. \\ & \left. + \tilde{M}_R^{(o)}(t, s) \tilde{R}_o^{\bar{\alpha}}(s, t') \right] ds, \end{aligned} \quad (\text{H.13})$$

$$\begin{aligned} \tilde{\nu}^{\bar{\alpha}}(t) &= - \int_0^t \mathrm{d}s \left[ \tilde{M}_R^{(d)}(t, s) \tilde{C}_d(t, s) + \tilde{M}_R^{(o)}(t, s) \tilde{C}_o^{\bar{\alpha}}(t, s) \right] \mathrm{d}s \\ &\quad - \int_0^t \left[ \tilde{M}_C^{(d)}(t, s) \tilde{R}_d^{\bar{\alpha}}(t, s) + \tilde{M}_C^{(o)}(t, s) \tilde{R}_o^{\bar{\alpha}}(t, s) \right] \mathrm{d}s. \end{aligned} \quad (\text{H.14})$$

These are to be solved with boundary condition

$$\tilde{C}_o^{\bar{\alpha}}(0,0) = 0, \quad \tilde{R}_o^{\bar{\alpha}}(0,0) = 0, \quad (\text{H.15})$$

$$\tilde{C}_d^{\bar{\alpha}}(0,0) = 1, \quad \tilde{R}_d^{\bar{\alpha}}(0^+,0) = 1. \quad (\text{H.16})$$

The scaling behavior of the train error is then given by

$$\lim_{m \rightarrow \infty} e_{\text{tr}}(t) = -\frac{1}{2} \tilde{C}_A^{\bar{\alpha}}(t, t) =: e_{\text{tr}}^{\bar{\alpha}}(t). \quad (\text{H.17})$$

In order to test the accuracy of the asymptotic analysis developed in this sections, we solved numerically the SymmDMFT equations for increasing values of m and compare the results to the numerical integration of Eqs. [\(H.10\)](#page-64-0), [\(H.14\)](#page-64-1) presented in this section. Some results of this comparison are presented in Fig. [33,](#page-63-2) which shows good agreement between finite-<sup>m</sup> curves and <sup>m</sup> → ∞ limit.

The solution of Eqs. [\(H.10\)](#page-64-0), [\(H.14\)](#page-64-1) provides another route to estimate the large-m interpolation threshold <sup>α</sup>GF(∞) at fixed <sup>a</sup>(t) = 1. Namely, we solve the equations numerically and extract the <sup>t</sup>rel(α, ∞, c), which is defined analogously to above. We then fit the divergence of <sup>t</sup>rel(α, ∞, c) at <sup>α</sup>GF(∞) according to Eq. [\(H.2\)](#page-63-0). We obtain <sup>α</sup>GF(∞) ≈ <sup>1</sup>.18, in agreement with the threshold obtained by extrapolating the finite-m thresholds αGF(m). In the bottom right plot of Fig. [32](#page-62-1) we plot <sup>t</sup>rel(α, ∞, c) as function of <sup>α</sup>GF(∞) − <sup>α</sup>. This confirms the behavior of Eq. [\(H.2\)](#page-63-0) with <sup>ν</sup> ≈ <sup>2</sup>.

We conclude by emphasizing that, throughout this section α(t) = 1 and τ = 1 were fixed. If we generalize to arbitrary α(t) = a<sup>0</sup> and arbitrary τ > 0, the threshold αGF(m) will of course on these quantities through the ratio a0/τ .

## I Details about SGD simulations

In this appendix we provide some details about the numerical simulations with stochastic gradient descent (SGD) presented in Figures [2,](#page-4-0) [4.](#page-7-1)

We generate data according to the pure noise model y<sup>i</sup> = ε<sup>i</sup> (Fig. [2\)](#page-4-0), y<sup>i</sup> = φ(w<sup>T</sup> <sup>∗</sup>xi) + ε<sup>i</sup> (Fig. [4\)](#page-7-1), <sup>i</sup> ≤ <sup>n</sup>. We learn the two-layer network of Eq. [\(1.1\)](#page-1-0), see below for the class definition.

class Net ( nn . Module ) : def \_\_init\_\_ ( self , a , m , d ) : super () . \_\_init\_\_ () self . m = m self . lin1 = nn . Linear (d ,m , bias = False ) self . lin1 . weight . data = (1/ np . sqrt ( d )) \* torch . randn (( m , d ) ) self . lin2 = nn . Linear (m ,1 , bias = False ) self . lin2 . weight . data [0 ,:] = a self . act = Myact () self . project () def forward ( self , x ) : x1 = self . act ( self . lin1 ( x ) ) return self . lin2 ( x1 ) / self . m def project ( self , epsilon ) : row\_norms = torch . norm ( self . lin1 . weight . data , dim =1 , keepdim = True ) row\_norms = torch . clamp ( row\_norms , min= epsilon ) self . lin1 . weight . data = self . lin1 . weight . data / row\_norms

As shown in this code, we use the initialization

$$(\mathbf{a}_0, \mathbf{W}_0) = \mathbf{P}_B(\bar{\mathbf{a}}_0, \bar{\mathbf{W}}_0), \quad (1.1)$$

$$\bar{\mathbf{a}}_0 = (a_0, \dots, a_0), \quad (W_{0,ij})_{i \leq m, j \leq d} \sim \mathcal{N}(0, 1/d). \quad (1.2)$$

$$\bar{\mathbf{a}}_0 = (a_0, \dots, a_0), \quad (W_{0,ij})_{i \leq m, j \leq d} \sim \text{N}(0, 1/d). \quad (\text{I.2})$$

where P <sup>B</sup> projects first layer weights to the unit ball:

$$P_{\text{B}}\left(a, (w_1, \dots, w_m)\right) = \left(a, \left(\frac{w_1}{\|w_1\| \wedge 1}, \dots, \frac{w_m}{\|w_m\| \wedge 1}\right)\right). \quad (1.3)$$

We use the standard SGD iteration without weight decay and constant stepsize η, and batch size b:

$$\bar{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla \hat{\mathcal{R}}_{S(k)}(\boldsymbol{\theta}_k), \quad \hat{\mathcal{R}}_S(\boldsymbol{\theta}) = \frac{1}{2|S|} \sum_{i \in S} (y_i - f(\mathbf{x}_i; \boldsymbol{\theta}))^2, \quad (1.4)$$

$$\boldsymbol{\theta}_{k+1} = \mathbf{P}_{\text{B}}(\bar{\boldsymbol{\theta}}_{k+1}). \quad (1.5)$$

$$\theta_{k+1} = P_{\text{B}}(\bar{\theta}_{k+1}). \quad (1.5)$$

The optimizer is defined in the code below

optimizer = optim . SGD ( net . parameters () , lr = lr , momentum =0. , weight\_decay =0.) lambda\_step = lambda epoch : 1 scheduler = torch . optim . lr\_scheduler . LambdaLR ( optimizer , lr\_lambda = lambda\_step )

In the simulations of Figures [2,](#page-4-0) and [4](#page-7-1) we use batch size b = 100 and step size η = 0.1. Each symbol reports the average of Nsim = 10 simulations.

## J Lower bounding the overfitting timescale

Throughout this appendix we use t to denote the rescaled time tˆintroduced in Section [3.](#page-7-0)

## J.1 Proof of Theorem 3.1

$$\text{By computing the derivative } \partial_{a_i} \widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)), \text{ we get} \\ \frac{d}{dt} |a_\ell(t)| \leq \left| \frac{1}{n} \sum_{i=1}^n (y_i - f(\mathbf{x}_i; \mathbf{a}(t), \mathbf{W}(t)) \sigma(\mathbf{w}_\ell(t)^\top \mathbf{x}_i) \right|$$

<sup>n</sup>(a(t),W(t)), we get

$$\begin{aligned} &\leq \sqrt{2\hat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t))} \cdot \sqrt{\frac{1}{n} \sum_{i=1}^n \sigma(\mathbf{w}_\ell^\top \mathbf{x}_i)^2} \\ &\leq 4L\sqrt{2\hat{\mathcal{R}}_n(\mathbf{a}(0), \mathbf{W}(0))} \cdot \sqrt{\frac{1}{n} \sum_{i=1}^n (1 + (\mathbf{w}_\ell(t)^\top \mathbf{x}_i)^2)} \\ &\leq 10L\sqrt{2\hat{\mathcal{R}}_n(\mathbf{a}(0), \mathbf{W}(0))}, \end{aligned}$$

2Rb where, for <sup>n</sup> ≥ <sup>d</sup>, the last inequality holds with probability at least <sup>1</sup> − 2 exp(−cn) (for some universal c > 0) by standard upper bounds on the norm of random matrices [\[53\]](#page-12-7). Further

$$\begin{aligned}\sqrt{2n\hat{\mathcal{H}}_n(\mathbf{a}(0), \mathbf{W}(0))} &= \left\| \mathbf{y} - \frac{1}{m} \sum_{i=1}^m a_i \sigma(\mathbf{X} \mathbf{w}_i) \right\| \\ &\stackrel{(a)}{\leq} \|\mathbf{y}\| + a_0 \max_{\ell \leq m} \|\sigma(\mathbf{X} \mathbf{w}_\ell(0))\| \\ &\stackrel{(b)}{\leq} \tau \|\mathbf{g}\| + \|\varphi(\mathbf{X} \mathbf{U})\| + a_0 \max_{\ell \leq m} \|\sigma(\mathbf{X} \mathbf{w}_\ell(0))\| \\ &\leq \tau \|\mathbf{g}\| + L \|\mathbf{X} \mathbf{U}\| + \sqrt{n} |\varphi(0)| + a_0 L \|\mathbf{X}\|_{\text{op}} + a_0 \sqrt{n} |\sigma(0)|,\end{aligned}$$

 X where in (a) it is understood that <sup>σ</sup> is applied entrywise to Xw<sup>i</sup> ∈ <sup>R</sup> <sup>n</sup> and in (b) we have <sup>g</sup> ∼ <sup>N</sup>(0, <sup>I</sup>n), and <sup>φ</sup> is applied row-wise to XU ∈ <sup>R</sup> n×k . By using standard concentration on the norm of random matrices, also with probability <sup>1</sup> − exp(−cn), we have (for <sup>m</sup> ≤ <sup>n</sup>)

$$\sqrt{2\hat{\mathcal{R}}_n(\mathbf{a}(0), \mathbf{W}(0))} = C(\tau + \sqrt{k} + a_0L).$$

Summarizing the above bounds, we have

$$\frac{d}{dt} |a_\ell(t)| \leq a_1,$$

which implies the first claim by integration.

To prove the second claim, we consider the following sets of parameters W<sup>∞</sup> m,d(a) ⊆ Wm,d(a) (which will also prove useful in the next section)

$$W_{m,d}(\bar{a}) := \left\{ (\mathbf{a}, \mathbf{W}) \in \mathbb{R}^m \times \mathbb{R}^{m \times d} : \frac{\|\mathbf{a}\|_1}{m} \leq \bar{a}, \|\mathbf{w}_i\|_2 = 1 \ \forall i \leq m \right\}, \quad (\text{J.1})$$

$$W_{\infty}(\bar{a}) := \left\{ (\mathbf{a}, \mathbf{W}) \in \mathbb{P}^m \times \mathbb{P}^{m \times d} : \|\mathbf{a}\| \leq \bar{a}, \|\mathbf{w}_i\|_2 = 1 \ \forall i \leq m \right\}, \quad (\text{J.2})$$

$$\mathcal{W}_{m,d}^\infty(\bar{a}) := \left\{ (\mathbf{a}, \mathbf{W}) \in \mathbb{R}^m \times \mathbb{R}^{m \times d} : \|\mathbf{a}\|_\infty \leq \bar{a}, \|\mathbf{w}_i\|_2 = 1 \quad \forall i \leq m \right\}. \quad (\text{J.2})$$

The second claim follows in turn if we prove that there exists a universal constant C such that

$$\sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} |\widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \mathcal{R}(\mathbf{a}, \mathbf{W})| \leq C(L^2 \bar{\mathbf{a}}^2 + \tau^2) \sqrt{\frac{d}{n}}. \quad (\text{J.3})$$

This is a standard estimate, that we reproduce for the readers' convenience.

We begin by bounding the expectation of the supremum by symmetrization and contraction inequalities. Letting (ξi)i≤<sup>n</sup> ∼iid Unif({+1, −1}), we have

$$\begin{aligned} \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} |\widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \mathcal{R}(\mathbf{a}, \mathbf{W})| &\leq \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \frac{1}{n} \sum_{i=1}^n \xi_i (y_i - f(\mathbf{x}_i; \mathbf{a}, \mathbf{W}))^2 \\ &\leq 2\mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \frac{1}{n} \sum_{i=1}^n \xi_i y_i f(\mathbf{x}_i; \mathbf{a}, \mathbf{W}) + \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \frac{1}{n} \sum_{i=1}^n \xi_i f(\mathbf{x}_i; \mathbf{a}, \mathbf{W})^2 \\ &=: 2E_1 + E_2 . \end{aligned}$$

We begin by bounding E1:

$$\begin{aligned} E_1 &= \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \sum_{j=1}^m \frac{a_j}{m} \frac{1}{n} \sum_{i=1}^n \xi_i y_i \sigma(\mathbf{w}_{\mathbf{j}}^{\top} \mathbf{x}_i) \\ &\leq \bar{\mathbf{a}} \mathbb{E} \sup_{\|\mathbf{w}\|=1} \frac{1}{n} \sum_{i=1}^n \xi_i y_i \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \\ &\stackrel{(a)}{\leq} \bar{\mathbf{a}} L \mathbb{E} \sup_{\|\mathbf{w}\|=1} \frac{1}{n} \sum_{i=1}^n \xi_i (1 + |y_i|) \mathbf{w}^{\top} \mathbf{x}_i \\ &\leq \bar{\mathbf{a}} L \mathbb{E} \left\{ \left\| \frac{1}{n} \sum_{i=1}^n \xi_i (1 + |y_i|) \mathbf{x}_i \right\| \right\} \\ &\leq C \bar{\mathbf{a}} L (L + \tau) \sqrt{\frac{d}{n}}, \end{aligned}$$

where in (a) we applied the contraction inequality of [\[36\]](#page-11-17) to the function <sup>ψ</sup>i(t) = <sup>y</sup>iσ(t/(|<sup>y</sup><sup>i</sup> | + 1)). We next bound term E2:

$$\begin{aligned} E_2 &= \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \sum_{j, l=1}^m \frac{a_j a_l}{m^2} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma(\mathbf{w}_j^\top \mathbf{x}_i) \sigma(\mathbf{w}_l^\top \mathbf{x}_i) \\ &\leq \bar{\mathbf{a}}^2 \mathbb{E} \sup_{\mathbf{w}, \tilde{\mathbf{w}} \in \mathbb{S}^{d-1}} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma(\mathbf{w}^\top \mathbf{x}_i) \sigma(\tilde{\mathbf{w}}^\top \mathbf{x}_i) \\ &\stackrel{(b)}{\leq} CL^2 \bar{\mathbf{a}}^2 \mathbb{E} \sup_{\mathbf{w} \in \mathbb{S}^{d-1}} \frac{1}{n} \sum_{i=1}^n \xi_i \mathbf{w}^\top \mathbf{x}_i \\ &\leq CL^2 \bar{\mathbf{a}}^2 \sqrt{\frac{d}{n}}, \end{aligned}$$

where inequality (b) follows by applying the contraction inequality of [\[36\]](#page-11-17) to ψ(t1, t2) = σ(t1)σ(t2) which is CL<sup>2</sup> -Lipschitz because ∥σ∥Lip, ∥σ∥<sup>∞</sup> ≤ <sup>L</sup>.

Summarizing, we proved that

$$\mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} |\widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \mathcal{R}(\mathbf{a}, \mathbf{W})| \leq C(L^2 \bar{\mathbf{a}}^2 + \tau^2) \sqrt{\frac{d}{n}}. \quad (\text{J.4})$$

In order to complete the proof of Eq. [\(J.3\)](#page-66-0), we will show that the supremum concentrates around its expectation. For fixed (a,W) ∈ Wm,d(a), we have

$$|f(\mathbf{x}; \mathbf{a}, \mathbf{W}) - f(\mathbf{x}'; \mathbf{a}, \mathbf{W})| \leq L \bar{a} \|\mathbf{x} - \mathbf{x}'\|_2,$$

$$\|\varphi(\mathbf{U}^\top \mathbf{x}) - \varphi(\mathbf{U}^\top \mathbf{x}')\| \leq L \|\mathbf{x} - \mathbf{x}'\|_2.$$
to emphasize the dependence of the risk on  $\mathbf{x}$ 

φ(U <sup>≤</sup> <sup>L</sup>∥<sup>x</sup> <sup>−</sup> <sup>x</sup> We write <sup>R</sup>b <sup>n</sup>(X; a,W) to emphasize the dependence of the risk on X Letting r(x<sup>i</sup> ; a,W) = φ(U <sup>T</sup>x) − <sup>f</sup>(x; <sup>a</sup>,W), we have

$$\begin{aligned}\nabla_{\mathbf{x}_i} \widehat{\mathcal{R}}_n(\mathbf{X}; \mathbf{a}, \mathbf{W}) &= \frac{1}{n} (\varepsilon_i + r(\mathbf{x}_i; \mathbf{a}, \mathbf{W})) \nabla_{\mathbf{x}_i} r(\mathbf{x}_i; \mathbf{a}, \mathbf{W}), \\ \Rightarrow \|\nabla_{\mathbf{x}_i} \widehat{\mathcal{R}}_n(\mathbf{X}; \mathbf{a}, \mathbf{W})\| &\leq \frac{C}{n} (|\varepsilon_i| + L \bar{a}) L \bar{a}.\end{aligned}$$

Hence

$$\begin{aligned} \|\nabla_{\mathbf{X}} \widehat{\mathcal{R}}_n(\mathbf{X}; \mathbf{a}, \mathbf{W})\| &\leq \frac{C}{\sqrt{n}} L \bar{a} \left( L \bar{a} + \frac{\|\boldsymbol{\varepsilon}\|}{\sqrt{n}} \right) \\ &\leq \frac{C'}{\sqrt{n}} L \bar{a} (L \bar{a} + \tau), \end{aligned}$$

where the last inequality holds on an event that has probability at least <sup>1</sup>−<sup>e</sup> <sup>−</sup><sup>n</sup>. Defining Zn,d,m(a) := sup(a,W)∈Wm,d(a)  Rb <sup>n</sup>(a,W) − <sup>R</sup>(a,W)  , Borell inequality yields

$$|\widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \mathcal{R}(\mathbf{a}, \mathbf{W})|, \text{ Borell inequality yields } \mathbb{P}\left\{\left|Z_{n,d,m}(\bar{\mathbf{a}}) - \mathbb{E}Z_{n,d,m}(\bar{\mathbf{a}})\right| \geq B t\right\} \leq 2 e^{-nt^2} + e^{-n},$$

$$B := C'' L \bar{\mathbf{a}} (L \bar{\mathbf{a}} + \tau).$$

Together with Eq. [\(J.4\)](#page-67-0), we thus obtain that the following holds with probability <sup>1</sup> − <sup>2</sup><sup>e</sup> <sup>−</sup><sup>t</sup> − <sup>e</sup> −n

$$\mathbb{E}_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d(\bar{\mathbf{a}})}} |\widehat{\mathcal{H}}_n(\mathbf{a}, \mathbf{W}) - \mathcal{R}(\mathbf{a}, \mathbf{W})| \leq C(L^2 \bar{\mathbf{a}}^2 + \tau^2) \sqrt{\frac{d}{n}} + C(L^2 \bar{\mathbf{a}}^2 + \tau^2) \sqrt{\frac{t}{n}}$$

This yields the desired claim.

## J.2 Proof of Theorem 3.2

We introduce the notations:

$$g_{n,\ell}^w(\mathbf{a}, \mathbf{W}) := \frac{m}{|a_\ell|} [\nabla_{w_\ell} \widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \nabla_{w_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})], \quad (\text{J.5})$$

$$g_{n,\ell}^a(\mathbf{a}, \mathbf{W}) := m [\nabla_{a_\ell} \widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \nabla_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})]. \quad (\text{J.6})$$

$$g_{n,\ell}^a(\mathbf{a}, \mathbf{W}) := m[\nabla_{a_\ell} \widehat{\mathcal{R}}_n(\mathbf{a}, \mathbf{W}) - \nabla_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})]. \quad (\text{J.6})$$

We begin by establishing a uniform convergence lemma.

Lemma J.1. *Under the data distribution of Section [A,](#page-21-4) assume* ∥φ∥<sup>∞</sup> ≤ <sup>L</sup> *and the activation function to be bounded differentiable with Lipschitz continuous first derivative* ∥σ∥∞, ∥<sup>σ</sup> ′∥∞, ∥<sup>σ</sup> ′∥Lip ≤ <sup>L</sup>*. Then there exists a universal constant* C1*, and a constant* c<sup>0</sup> > 0 *dependent on* L, τ, α *such that, with probability at least* <sup>1</sup> − 2 exp(−nc0)*,*

$$\sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \max_{\ell \leq m} \|\mathbf{g}_{n, \ell}^{\mathbf{w}}(\mathbf{a}, \mathbf{W})\| \leq C(L^2\bar{\mathbf{a}} + \tau^2) \sqrt{\frac{d}{n} \log(ne/d)}, \quad (\text{G.7})$$

$$\sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \max_{\ell \leq m} \|\mathbf{g}_{n, \ell}^{\mathbf{a}}(\mathbf{a}, \mathbf{W})\| \leq C(L^2\bar{\mathbf{a}} + \tau^2) \sqrt{d} \quad (\text{G.8})$$

$$\sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, \bar{a}, (\bar{a})}} \max_{\ell \leq m} |g_{n, \ell}^a(\mathbf{a}, \mathbf{W})| \leq C(L^2 \bar{a} + \tau^2) \sqrt{\frac{d}{n}}. \quad (\text{J.8})$$

*Proof.* Gradient with respect to wℓ. By a concentration argument, it is sufficient to consider the expected supremum. Writing the formula for <sup>∇</sup><sup>w</sup>ℓ<sup>R</sup>b <sup>n</sup> and using a standard symmetrization argument, we get

$$\begin{aligned} \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} \|\mathbf{g}_{n, \ell}^w(\mathbf{a}, \mathbf{W})\| &= \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}}), \|\mathbf{u}\| \leq 1} \langle \mathbf{u}, \mathbf{g}_{n, \ell}^w(\mathbf{a}, \mathbf{W}) \rangle \\ &\leq 2 \mathbb{E} \sup_{\mathbf{w}, \mathbf{u}} \frac{1}{n} \sum_{i=1}^n \xi_i y_i \sigma'(\mathbf{w}^\top \mathbf{x}_i) \mathbf{u}^\top \mathbf{x}_i + 2 \bar{\mathbf{a}} \mathbb{E} \sup_{\mathbf{w}, \bar{\mathbf{w}}, \mathbf{u}} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma'(\mathbf{w}^\top \mathbf{x}_i) \sigma(\bar{\mathbf{w}}^\top \mathbf{x}_i) \mathbf{u}^\top \mathbf{x}_i \\ &=: B_1 + B_2 , \end{aligned}$$

where the ξ<sup>i</sup> are i.i.d. Radamacher random variables and in the last two lines it is understood that the supremum is over ∥w∥, ∥w∥, ∥u∥ ≤ <sup>1</sup>. Consider the second term in the last expression. Defining <sup>η</sup>(x) = <sup>x</sup>1|x|≤<sup>M</sup> <sup>+</sup> <sup>M</sup>(1x>M − <sup>1</sup>x<−M), and <sup>η</sup>(x) = <sup>x</sup> − <sup>η</sup>(x), we have

$$\begin{aligned} B_2 &= 2\bar{a} \mathbb{E} \sup_{\mathbf{w}, \bar{\mathbf{w}}, \mathbf{u} \in \mathcal{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma'(\mathbf{w}^\top \mathbf{x}_i) \sigma(\bar{\mathbf{w}}^\top \mathbf{x}_i) \mathbf{u}^\top \mathbf{x}_i \\ &\leq 2\bar{a} \mathbb{E} \sup_{\mathbf{w}, \bar{\mathbf{w}}, \mathbf{u} \in \mathcal{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma'(\mathbf{w}^\top \mathbf{x}_i) \sigma(\bar{\mathbf{w}}^\top \mathbf{x}_i) \eta(\mathbf{u}^\top \mathbf{x}_i) \\ &\quad + 2\bar{a} \mathbb{E} \sup_{\mathbf{w}, \bar{\mathbf{w}}, \mathbf{u} \in \mathcal{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma'(\mathbf{w}^\top \mathbf{x}_i) \sigma(\bar{\mathbf{w}}^\top \mathbf{x}_i) \bar{\eta}(\mathbf{u}^\top \mathbf{x}_i) \\ &=: B_{2,1} + B_{2,2}. \end{aligned}$$

Further defining ϕ(t1, t2, t3) := σ(t1)σ ′ (t2)η(t3) (which is CL2M-Lipschitz for <sup>M</sup> ≥ <sup>1</sup>), we have

$$B_{2,1} = \bar{a} \mathbb{E} \sup_{w, \bar{w}, u \in \mathbb{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \phi(w^\top x_i, \bar{w}^\top x_i, u^\top x_i). \quad (\text{J.9})$$

Using the contraction inequality of [\[36\]](#page-11-17), we get

$$\begin{aligned} B_{2,1} &\leq CL^2 M \bar{a} \left\{ \mathbb{E} \sup_{\mathbf{w} \in \mathbf{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \mathbf{w}^\top \mathbf{x}_i + \mathbb{E} \sup_{\bar{\mathbf{w}} \in \mathbf{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \bar{\mathbf{w}}^\top \mathbf{x}_i + \mathbb{E} \sup_{\mathbf{u} \in \mathbf{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \mathbf{u}^\top \mathbf{x}_i \right\} \\ &\leq CL^2 M \bar{a} \sqrt{\frac{d}{n}}. \end{aligned}$$

Next consider B2,2:

$$\begin{aligned} B_{2,2} &\leq 2\bar{a}L^2\mathbb{E} \sup_{\mathbf{u} \in \mathbf{B}^d(1)} \frac{1}{n} \sum_{i=1}^n |\bar{\eta}(\mathbf{u}^\top \mathbf{x}_i)| \\ &\leq 2\bar{a}L^2 \sup_{\mathbf{u} \in \mathbf{B}^d(1)} \mathbb{E}|\bar{\eta}(\mathbf{u}^\top \mathbf{x}_i)| + 2\bar{a}L^2\mathbb{E} \sup_{\mathbf{u} \in \mathbf{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i |\bar{\eta}(\mathbf{u}^\top \mathbf{x}_i)| \\ &\leq CL^2 \bar{a} e^{-M^2/4} + CL^2 \bar{a} \sqrt{\frac{d}{n}}, \end{aligned}$$

where the last inequality holds because u <sup>T</sup>x<sup>i</sup> is Gaussian with variance ∥u∥ 2 , and using again the contraction inequality. Collecting various terms and optimizing over <sup>M</sup> ≥ <sup>1</sup>, we obtain

$$\begin{aligned} B_2 &\leq CL^2\bar{a}\left\{M\sqrt{\frac{d}{n}} + e^{-M^2/4}\right\} \\ &\leq CL^2\bar{a}\sqrt{\frac{d}{n}\log(n/d)}. \end{aligned}$$

The proof of Eq. [\(J.7\)](#page-68-0) is completed by bounding B<sup>1</sup> along the same lines.

$$\begin{aligned} & \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} |g_{n, \ell}^a(\mathbf{a}, \mathbf{W})| = \mathbb{E} \sup_{(\mathbf{a}, \mathbf{W}) \in \mathcal{W}_{m, d}(\bar{\mathbf{a}})} g_{n, \ell}^a(\mathbf{a}, \mathbf{W}) \\ & \leq 2 \mathbb{E} \sup_{\mathbf{W}, \mathbf{u}} \frac{1}{n} \sum_{i=1}^n \xi_i y_i \sigma(\mathbf{w}_{\ell}^{\top} \mathbf{x}_i) + 2 \mathbb{E} \sup_{\mathbf{a}, \mathbf{W}, \mathbf{u}} \frac{1}{n} \sum_{i=1}^n \xi_i \sum_{j=1}^m \frac{a_j}{m} \sigma(\mathbf{w}_j^{\top} \mathbf{x}_i) \sigma(\mathbf{w}_{\ell}^{\top} \mathbf{x}_i) \\ & =: D_1 + D_2. \end{aligned}$$

<sup>n</sup> and using symmetrization, we get

Consider term D2, and define the L 2 -Lipschitz function ψ(t1, t2) := σ(t1)σ(t2),

$$\begin{aligned} D_2 &\leq 2\bar{a}\mathbb{E} \sup_{\mathbf{W}, \mathbf{u}} \max_{1 \leq j \leq m} \frac{1}{n} \sum_{i=1}^n \xi_i \sigma(\mathbf{w}_j^\top \mathbf{x}_i) \sigma(\mathbf{w}_\ell^\top \mathbf{x}_i) \\ &\leq 2\bar{a}\mathbb{E} \sup_{\mathbf{w}, \bar{\mathbf{w}} \in \mathbf{B}^d(1)} \frac{1}{n} \sum_{i=1}^n \xi_i \psi(\mathbf{w}^\top \mathbf{x}_i, \bar{\mathbf{w}}^\top \mathbf{x}_i) \\ &\leq CL^2 \bar{a} \sqrt{\frac{d}{n}}. \end{aligned}$$

Term D<sup>1</sup> is controlled analogously, yielding the proof of Eq. [\(J.8\)](#page-68-1).

We next prove some continuity properties of the population risk R. It is useful to recall the form:

$$\mathcal{R}(\mathbf{a}, \mathbf{W}) = \frac{1}{2}(\tau^2 + \|\varphi\|^2) - \frac{1}{m} \sum_{i=1}^m a_i \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_i) + \frac{1}{2m^2} \sum_{i,j=1}^m a_i a_j h(\mathbf{w}_i^\top \mathbf{w}_j). \quad (\text{J.10})$$

Lemma J.2. *Under the data distribution of Section [A,](#page-21-4) assume* ∥φ∥<sup>∞</sup> ≤ <sup>L</sup> *that* <sup>φ</sup> *and* <sup>σ</sup> *are bounded differentiable with Lipschitz continuous first derivative,* ∥σ∥∞, ∥<sup>σ</sup> ′∥∞, ∥<sup>σ</sup> ′∥<sup>∞</sup> ≤ <sup>L</sup>*,* ∥φ∥∞, ∥∇φ∥∞, ∥∇φ∥Lip ≤ <sup>L</sup>*,* <sup>L</sup> ≥ <sup>1</sup>*. Then, there exists an absolute constant* <sup>C</sup> *such that for any* (a,W),(a,W˜ ) ∈ Wm,d(a)*:*

$$\|\nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \tilde{\mathbf{W}}) - \nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})\| \leq CL^2 \frac{|a_\ell|}{m} (1 + \bar{a}) \max_{j \leq m} \|\tilde{\mathbf{w}}_j - \mathbf{w}_j\|, \quad (\text{J.11})$$

$$|\partial_{a_\ell} \mathcal{R}(\mathbf{a}, \tilde{\mathbf{W}}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| < \frac{CL^2}{m} (1 + \bar{a}) \max \|\tilde{\mathbf{w}}_i - \mathbf{w}_i\|, \quad (\text{J.12})$$

$$|\partial_{a_\ell} \mathcal{R}(\mathbf{a}, \tilde{\mathbf{W}}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| \leq \frac{CL^2}{m} (1 + \bar{a}) \max_{j \leq m} \|\tilde{\mathbf{w}}_j - \mathbf{w}_j\|, \quad (\text{J.12})$$

*and*

$$\|\nabla_{\mathbf{w}_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \mathbf{W}) - \nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})\| \leq \frac{CL^2}{m^2} (1 + \bar{\mathbf{a}}) |\tilde{a}_\ell - a_\ell| + CL^2 \frac{|a_\ell|}{m^2} \|\tilde{\mathbf{a}} - \mathbf{a}\|_1, \quad (\text{J.13})$$

$$|\partial_{a_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \mathbf{W}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| \leq \frac{CL^2}{2} \|\tilde{\mathbf{a}} - \mathbf{a}\|_1. \quad (\text{J.14})$$

$$\|\nabla_{\mathbf{w}_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \mathbf{W}) - \nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})\| \leq \frac{CL^2}{m} (1 + \tilde{a}) |\tilde{a}_\ell - a_\ell| + CL^2 \frac{|a_\ell|}{m^2} \|\tilde{\mathbf{a}} - \mathbf{a}\|_1, \quad (\text{J.13})$$

$$|\partial_{a_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \mathbf{W}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| \leq \frac{CL^2}{m^2} \|\tilde{\mathbf{a}} - \mathbf{a}\|_1. \quad (\text{J.14})$$

*Proof.* As a preliminary remark, the assumptions on φ, σ imply similar smoothness properties of <sup>φ</sup>b, <sup>h</sup>. In particular, recall that <sup>h</sup>(q) = <sup>E</sup>[σ(G1)σ(Gq)] for (G1, Gq) jointly Gaussian, centered with unit variance and covariance <sup>E</sup>[G1, Gq] = 1, whence its k-th derivative is h (k) (q) = <sup>E</sup>[σ (k) (G1)σ (k) (Gq)] (whenever <sup>σ</sup> ∈ <sup>C</sup> (k) (R)). Therefore, the assumptions on <sup>σ</sup> imply that ∥<sup>h</sup> ′∥∞, ∥h ′∥Lip ≤ <sup>L</sup> 2 . Similarly, ∥∇φb<sup>∥</sup>∞, ∥∇φb<sup>∥</sup>Lip <sup>≤</sup> CL<sup>2</sup> .

Proof of Eq. [\(J.11\)](#page-70-0). Differentiating Eq. [\(J.10\)](#page-69-0)

$$\frac{m}{a_\ell} \nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W}) = -\mathbf{U} \nabla \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_\ell) + \sum_{j=1}^m \frac{a_j}{m} h'(\mathbf{w}_\ell^\top \mathbf{w}_j) \mathbf{w}_j . \quad (\text{J.15})$$

Therefore

$$\begin{aligned} \frac{m}{|a_\ell|} \|\nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \tilde{\mathbf{W}}) - \nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})\| &\leq \|\nabla \hat{\varphi}(\mathbf{U}^\top \tilde{\mathbf{w}}_\ell) - \nabla \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_\ell)\| \\ &\quad + \sum_{j=1}^m \frac{|a_j|}{m} \|h'(\tilde{\mathbf{w}}_\ell^\top \tilde{\mathbf{w}}_j) \tilde{\mathbf{w}}_j - h'(\mathbf{w}_\ell^\top \mathbf{w}_j) \mathbf{w}_j\| \\ &\leq CL^2 \|\tilde{\mathbf{w}}_\ell - \mathbf{w}_\ell\| + \bar{a} \max_{j \leq m} \|h'(\tilde{\mathbf{w}}_\ell^\top \tilde{\mathbf{w}}_j) \tilde{\mathbf{w}}_j - h'(\mathbf{w}_\ell^\top \mathbf{w}_j) \mathbf{w}_j\|. \end{aligned}$$

Further, by the above smoothness properties of h,

$$\|h'(\tilde{w}_\ell^\top \tilde{w}_j) \tilde{w}_j - h'(w_\ell^\top w_j) w_j\| \leq CL^2 \|\tilde{w}_j - w_j\| + CL^2 \|\tilde{w}_\ell - w_\ell\|.$$

Substituting above, this yields the claim [\(J.11\)](#page-70-0).

Proof of Eq. [\(J.12\)](#page-70-1). We proceed analogously to the previous point. Namely

$$m\partial_{a_\ell}\mathcal{R}(\mathbf{a}, \mathbf{W}) = -\hat{\varphi}(\mathbf{U}^\top \mathbf{w}_\ell) + \sum_{j=1}^m \frac{a_j}{m} h(\mathbf{w}_\ell^\top \mathbf{w}_j), \quad (\text{J.16})$$

whence

$$\begin{aligned} m |\partial_{a_\ell} \mathcal{R}(\mathbf{a}, \tilde{\mathbf{W}}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| &\leq |\hat{\varphi}(\mathbf{U}^\top \tilde{\mathbf{w}}_\ell) - \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_\ell)| + \sum_{j=1}^m \frac{|a_j|}{m} |h(\tilde{\mathbf{w}}_\ell^\top \tilde{\mathbf{w}}_j) - h(\mathbf{w}_\ell^\top \mathbf{w}_j)| \\ &\leq CL^2 \|\tilde{\mathbf{w}}_\ell - \mathbf{w}_\ell\| + C\bar{a}L^2 (\|\tilde{\mathbf{w}}_\ell - \mathbf{w}_\ell\| + \|\tilde{\mathbf{w}}_j - \mathbf{w}_j\|), \end{aligned}$$
which implies immediately Eq. (J.12)

which implies immediately Eq. [\(J.12\)](#page-70-1)

Proof of Eq. [\(J.13\)](#page-70-2). Recalling Eq. [\(J.15\)](#page-70-3), we have

$$m\|\nabla_{\mathbf{w}_\ell}\mathcal{R}(\tilde{\mathbf{a}}, \mathbf{W}) - \nabla_{\mathbf{w}_\ell}\mathcal{R}(\mathbf{a}, \mathbf{W})\| \leq \|\nabla \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_\ell)\| |\tilde{a}_\ell - a_\ell| + \sum_{j=1}^m \frac{1}{m} \|h'_s(\mathbf{w}_\ell^\top \mathbf{w}_j) \mathbf{w}_j\| |\tilde{a}_\ell \tilde{a}_j - a_\ell a_j|$$

$$\leq CL|\tilde{a}_\ell - a_\ell| + CL^2\frac{|a_\ell|}{m}\|\tilde{\mathbf{a}} - \mathbf{a}\|_1 + CL^2\bar{a}\|\tilde{a}_\ell - a_\ell\|,$$

which proves the desired claim.

Proof of Eq. [\(J.13\)](#page-70-2). Recalling Eq. [\(J.16\)](#page-70-4), we have

$$\begin{aligned} m|\partial_{a_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \mathbf{W}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| &\leq \sum_{j=1}^m \frac{1}{m} |h(\mathbf{w}_\ell^\top \mathbf{w}_j)| \cdot |\tilde{a}_j - a_j| \\ &\leq \frac{CL^2}{m} \|\tilde{\mathbf{a}} - \mathbf{a}\|_1. \end{aligned}$$

Using the last lemma and triangle inequality we get the following.

Corollary J.3. *Under the assumptions of Lemma [J.2,](#page-70-5) there exists an absolute constant* C *such that, for all* (a,W),(a,W˜ ) ∈ W<sup>∞</sup> m,d(a)*:*

$$\begin{aligned} \max_{\ell \leq m} \|\nabla_{\mathbf{w}_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \tilde{\mathbf{W}}) - \nabla_{\mathbf{w}_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})\| &\leq \frac{CL^2 \bar{a}}{m} (1 + \bar{a}) \max_{j \leq m} \|\tilde{\mathbf{w}}_j - \mathbf{w}_j\| + \frac{CL^2}{m} (1 + \bar{a}) \|\tilde{\mathbf{a}} - \mathbf{a}\|_\infty, \\ \max_{\ell \leq m} |\partial_{a_\ell} \mathcal{R}(\tilde{\mathbf{a}}, \tilde{\mathbf{W}}) - \partial_{a_\ell} \mathcal{R}(\mathbf{a}, \mathbf{W})| &\leq \frac{CL^2}{m} (1 + \bar{a}) \max_{j \leq m} \|\tilde{\mathbf{w}}_j - \mathbf{w}_j\| + \frac{CL^2}{m} \|\tilde{\mathbf{a}} - \mathbf{a}\|_\infty. \end{aligned}$$

We next consider a(t),W(t) that follows GF with respect to the empirical risk, as per Eq. [\(A.10\)](#page-22-0), which we rewrite as

$$\begin{aligned} \dot{\mathbf{a}}(t) &= -m\nabla_{\mathbf{a}}\widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)), \\ \dot{\mathbf{w}}_i(t) &= -m\mathbf{P}_{\mathbf{w}_i}^\perp \nabla_{\mathbf{w}_i}\widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)) \quad \forall i = 1, \dots, m, \\ \text{), } \mathbf{W}_0(t) \text{ the GF with respect to population risk:} \end{aligned} \tag{J.17}$$

and denote by a0(t),W0(t) the GF with respect to population risk:

$$\begin{aligned} \dot{\mathbf{a}}_0(t) &= -m\nabla_{\mathbf{a}}\mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t)), \\ \dot{\mathbf{w}}_{0,i}(t) &= -m\mathbf{P}_{\mathbf{w}_i}^\perp \nabla_{\mathbf{w}_i}\mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t)) \quad \forall i = 1, \dots, m. \end{aligned} \tag{J.18}$$

Lemma J.4. *Under the data distribution of Section [A,](#page-21-4) there exists constant* c<sup>∗</sup> = c∗(δ)*,* c<sup>0</sup> = c0(δ) *depending uniquely on* δ > 0*, and an absolute constant* C *such that the following holds. Assume* φ, σ *to be bounded, differentiable with Lipschitz continuous first derivative* ∥φ∥∞, ∥<sup>φ</sup> ′∥∞, ∥<sup>φ</sup> ′∥Lip ≤ <sup>L</sup>*.* ∥σ∥∞, ∥<sup>σ</sup> ′∥∞, ∥<sup>σ</sup> ′∥Lip ≤ <sup>L</sup>*, Further assume* n/d ≥ exp(c0<sup>L</sup> 2 )*,* <sup>L</sup> ≥ <sup>1</sup>*. Let* (a(t),W(t))*,* (a0(t),W0(t))*, be defined as above, with* <sup>W</sup>(0) = <sup>W</sup>0(0) *and* <sup>a</sup>(0) = <sup>a</sup>0(0) *such that* ∥a(0)∥<sup>∞</sup> <sup>=</sup> ∥<sup>a</sup>0(0)∥<sup>∞</sup> ≤ <sup>a</sup>0*. Define*

$$T_*(m; c) := \inf \left\{ t : (\|\mathbf{a}(t)\|_\infty \vee \|\mathbf{a}_0(t)\|_\infty) \geq \left( c_* L^{-2} \log \frac{ne}{d} \right)^{1/3} \right\} \wedge \left( c_* L^{-2} \log \frac{ne}{d} \right)^{1/3}. \quad (\text{J.19})$$

*Then*

$$\sup_{t \leq T_*(m;c)} \Delta(t) \leq C(L^2 + \tau^2) \left( \frac{d}{n} \right)^{1/2-\delta}, \quad \Delta(t) := \max_{\ell \leq m} \|\tilde{\mathbf{w}}_\ell(t) - \mathbf{w}_\ell(t)\| + \|\tilde{\mathbf{a}}(t) - \mathbf{a}(t)\|_\infty. \quad (\text{J.20})$$

*Proof.* We will prove that the desired bound holds on the high-probability event of Lemma [J.1,](#page-68-2) where by we set a = (c1L −2 log ne/d) 1/3 . Throughout the proof, we use c0, c1, C to denote constants that might change from line to line, with dependence on the parameters of the problem as per the statement of the lemma.

We start by noting that, letting <sup>v</sup><sup>i</sup> <sup>=</sup> <sup>−</sup>m∇<sup>w</sup>i<sup>R</sup>b <sup>n</sup>(a,W) and <sup>v</sup>0,i <sup>=</sup> −(n/d)∇<sup>w</sup>0,iR(a0,W0, and P ⊥ <sup>w</sup> := <sup>I</sup> − ww<sup>T</sup> the projector orthogonal to <sup>w</sup>.

$$\left\| P_{w_i}^\perp v_i - P_{w_{0,i}}^\perp v_{0,i} \right\| \leq \left\| P_{w_i}^\perp (v_i - v_{0,i}) \right\| + \left\| (P_{w_i}^\perp - P_{w_{0,i}}^\perp) v_{0,i} \right\|$$

$$\begin{aligned} &\leq \|\mathbf{v}_i - \mathbf{v}_{0,i}\| + \|\mathbf{w}_i \mathbf{w}_i^\top - \mathbf{w}_{0,i} \mathbf{w}_{0,i}^\top\|_{\text{op}} \|\mathbf{v}_{0,i}\| \\ &\leq \|\mathbf{v}_i - \mathbf{v}_{0,i}\| + 2\|\mathbf{w}_i - \mathbf{w}_{0,i}\|_{\text{op}} \|\mathbf{v}_{0,i}\|. \end{aligned}$$

Hence, comparing the evolution of wi(t) and w0,i(t), we get

$$\begin{aligned} \frac{d}{dt} \|\mathbf{w}_i(t) - \mathbf{w}_{0,i}(t)\| &\leq m \|\nabla_{\mathbf{w}_i} \widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)) - \nabla_{\mathbf{w}_i} \mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t))\| \\ &\quad + m \|\nabla_{\mathbf{w}_i} \mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t))\| \cdot \|\mathbf{w}_i(t) - \mathbf{w}_{i,0}(t)\| \\ &=: D_1 + D_2 \cdot \|\mathbf{w}_i(t) - \mathbf{w}_{i,0}(t)\|. \end{aligned}$$

Since we are working on the event of Lemma [J.1,](#page-68-2) and using Corollary [J.3,](#page-71-0) we get, for <sup>t</sup> ≤ <sup>T</sup>∗(m; <sup>c</sup>).

$$\begin{aligned} D_1 &\leq m \|\nabla_{\mathbf{w}_i} \widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)) - \nabla_{\mathbf{w}_i} \mathcal{R}(\mathbf{a}(t), \mathbf{W}(t))\| \\ &\quad + m \|\nabla_{\mathbf{w}_i} \mathcal{R}(\mathbf{a}(t), \mathbf{W}(t)) - \nabla_{\mathbf{w}_i} \mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t))\| \\ &\leq C(L^2 \bar{\mathbf{a}} + \tau^2) \sqrt{\frac{d}{n} \log(ne/d)} + CL^2(1 + \bar{\mathbf{a}}^2) \max_{j \leq m} \|\mathbf{w}_j(t) - \mathbf{w}_{0,j}(t)\| \\ &\quad + CL^2(1 + \bar{\mathbf{a}}) \|\mathbf{a}(t) - \mathbf{a}_0(t)\|_{\infty}. \end{aligned}$$

Further

$$D_2 = |a_i| \left\| \mathbf{U} \nabla \hat{\varphi}(\mathbf{U}^\top \mathbf{w}_i) - \frac{1}{m} \sum_{j=1}^m a_j h'(\mathbf{w}_i^\top \mathbf{w}_j) \mathbf{w}_j \right\| \leq C \bar{a} (L^2 + \bar{a} L^2).$$

Collecting all the terms, and using <sup>a</sup> ≥ <sup>1</sup>, we get

$$\frac{d}{dt} \|\mathbf{w}_i(t) - \mathbf{w}_{0,i}(t)\| \leq C \bar{a} (L^2 \bar{a} + \tau^2) \sqrt{\frac{d}{n} \log(ne/d)} + C L^2 (1 + \bar{a}^2) \Delta(t). \quad (\text{J.21})$$

We next consider the evolution of second-layer weights:

$$\begin{aligned}
\frac{d}{dt}|a_i(t) - a_{0,i}(t)| &\leq m\|\partial_{a_i}\widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)) - \partial_{a_i}\mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t))\| \\
&\leq m\|\partial_{a_i}\widehat{\mathcal{R}}_n(\mathbf{a}(t), \mathbf{W}(t)) - \partial_{a_i}\mathcal{R}(\mathbf{a}(t), \mathbf{W}(t))\| \\
&\quad + m\|\partial_{a_i}\mathcal{R}_n(\mathbf{a}(t), \mathbf{W}(t)) - \partial_{a_i}\mathcal{R}(\mathbf{a}_0(t), \mathbf{W}_0(t))\| \\
&\leq C(L^2\bar{a} + \tau^2)\sqrt{\frac{d}{n}} + CL^2(1 + \bar{a})\max_{j \leq m}\|\mathbf{w}_j(t) - \mathbf{w}_{0,j}(t)\| + CL^2\|\mathbf{a}(t) - \mathbf{a}_0(t)\|_\infty \\
&\leq C(L^2\bar{a} + \tau^2)\sqrt{\frac{d}{n}} + CL^2(1 + \bar{a})\Delta(t).
\end{aligned}$$

Using the last bound together with Eq. [\(J.21\)](#page-72-0), we get

$$\frac{d}{dt} \Delta(t) \leq C \bar{a} (L^2 \bar{a} + \tau^2) \sqrt{\frac{d}{n} \log(ne/d)} + C L^2 (1 + \bar{a}^2) \Delta(t)$$

whence the claim follows by Gromwall inequality for sufficiently small c1.

We finally need a lemma from [\[10\]](#page-10-12) approximating GF in the population risk by the mean field dynamics.

Lemma J.5 (Corollary 1 and Proposition 3 [\[10\]](#page-10-12)). *Let* a0(t)*,* W0(t) *be GF with respect to the population risk* [\(J.18\)](#page-71-1) *with initialization* |<sup>a</sup><sup>0</sup>,i(0)| ≤ <sup>a</sup><sup>0</sup> *and* (w0,i(0))i≤<sup>m</sup> ∼ Unif(<sup>S</sup> d−1 )*. Recall that* a mf1 i (t)*,* v mf1(t) *is the solution of the ODEs* [\(3.4\)](#page-8-2) *with initialization* a mf1 i (0) = a<sup>0</sup>,i(0)*,* v mf1 i (t) = 0*. Under the assumptions of Theorem 3.2, for any* ε > 0 *there exists constants* c0, c<sup>1</sup> *depending uniquely on* L*, and an absolute constant* C *such that letting* Tlb(m) = ((c0/ε) log m) 1/3 *, the following happens with probability at least* <sup>1</sup> − 2 exp(−<sup>c</sup>1d)*,*

$$\sup_{t \leq T_{\text{lb}}(m)} \frac{1}{m} \sum_{i=1}^m \left( |a_i(t) - a_i^{\text{mfl}}(t)| + \|\mathbf{v}_i(t) - \mathbf{v}_i^{\text{mfl}}(t)\| \right) \leq C m^\varepsilon \left\{ \frac{1}{\sqrt{m}} + \frac{1}{\sqrt{d}} \right\}, \quad (\text{J.22})$$

$$\sup_{t \leq T_{\text{lb}}(m)} \left( \mathcal{R}(\mathbf{a}(t), \mathbf{W}(t)) - e_{\text{ts}}(t) \right) \leq C m^\varepsilon \left\{ \frac{1}{\sqrt{m}} + \frac{1}{\sqrt{d}} \right\}. \quad (\text{J.23})$$

*Proof of Theorem 3.2.* Throughout the proof L, τ, α are assumed to be fixed, and constants C, c0, . . . depend on them and can change from line to line. We will further work on the high probability events of Theorem 3.1, Lemma [J.4,](#page-71-2) and Lemma [J.5.](#page-72-1) By Theorem 3.1, for all <sup>t</sup> ≤ <sup>T</sup>lb(m) we have ∥a(t)∥<sup>∞</sup> ≤ c2(log 2m) 1/3 (where the constant c<sup>2</sup> can be made sufficiently small, by eventually reducing c1). An analogous of of Theorem 3.1 for the population risk implies ∥<sup>a</sup>0(t)∥<sup>∞</sup> ≤ <sup>c</sup>2(log 2m) 1/3 as well for all <sup>t</sup> ≤ <sup>T</sup>lb(m). Hence we can apply Lemma [J.4](#page-71-2) and Lemma [J.5,](#page-72-1) which yields the claim.

## K Dynamical mean field theory for non-Gaussian model

The DMFT equations for GF in the original non-Gaussian model can be derived from the general theory of [\[13\]](#page-10-8).

Given a (positive semi-definite) kernel <sup>Q</sup> : <sup>R</sup>≥<sup>0</sup> × <sup>R</sup>≥<sup>0</sup> → <sup>R</sup> <sup>m</sup>×<sup>m</sup>, (t, z) 7→ <sup>Q</sup>(t, s), we write <sup>z</sup> ∼ GP(0, <sup>Q</sup>) if <sup>z</sup> is a centered Gaussian process with values in <sup>R</sup> <sup>m</sup> and covariance <sup>E</sup>[z(t)z(s) <sup>T</sup>] = Q(t, s).

The DMFT equations can be interpreted as a set of fixed point equations for the functions Cij , Rij , a<sup>i</sup> . We define the deterministic processes a(t), νi(t) and stochastic processes w<sup>e</sup> (t) = (w e i (t) : <sup>i</sup> ≤ <sup>m</sup>), <sup>r</sup>(t) = (ri(t) : <sup>i</sup> ≤ <sup>m</sup>), as the solution of

$$\frac{da_i(t)}{dt} = \frac{\bar{\alpha}}{m} \mathbb{E}\{E(t) \sigma(r_i(t))\}, \quad (\text{K.1})$$

$$\nu_i(t) = \frac{\bar{\alpha}}{m} a_i(t) \mathbb{E}\{E(t) \sigma'(r_i(t)) r_i(t)\} \quad (\text{K.2})$$

$$\nu_i(t) = \frac{\bar{\alpha}}{m} a_i(t) \mathbb{E}\{E(t) \sigma'(r_i(t)) r_i(t)\}, \quad (\text{K.2})$$

$$w_i^e(t) = \frac{e_i(t)}{e_i(t)} - 1 \sum_{j=1}^m \int_0^t M_{ij}(s) e_j(s) ds, \quad (\text{K.3})$$

$$\frac{dw_i^e(t)}{dt} = -\nu_i(t)w_i^e(t) - \frac{1}{m} \sum_{l=1}^m \int_0^t M_{i,l}(t,s)w_l^e(s) ds \quad (\text{K.3})$$

$$- \sum_{j=1}^k M_{i,j}(t, *) u_j + \eta_i(t), \quad \boldsymbol{\eta} \sim \text{GP}(0, \mathbf{C}^E),$$

$$r_i(t) = \frac{1}{m} \sum_{l=1}^m \int_0^t R_{il}(t, s) a_l(s) E(s) \sigma'(r_l(s)) ds + \xi_i(t), \quad \xi \sim \text{GP}(0, \mathbf{C}), \quad (\text{K.4})$$

$$E(t) := y - \frac{1}{m} \sum_{l=1}^m a_l(t) \sigma(r_l(t)). \quad (\text{K.5})$$

Here, in the first equation, (w<sup>e</sup> (0),u) ∼ <sup>N</sup>(0, <sup>I</sup>m) ⊗ <sup>N</sup>(0, <sup>I</sup>k) are independent of <sup>η</sup>. In the second equation, <sup>y</sup> <sup>=</sup> <sup>φ</sup>(r0) + <sup>ε</sup> with (r0, ε) ∼ <sup>N</sup>(0, <sup>I</sup>k) ⊗ <sup>N</sup>(0, τ <sup>2</sup> ) independent of ξ.

$$M_{ij}(t, s) = \bar{\alpha} \mathbb{E}\{S_{ij}(t)\} \delta(t - s) + \bar{\alpha} \sum_{l=1}^m \mathbb{E}\{S_{il}(t) \frac{\partial r_l(t)}{\partial \xi_j(s)}\}, \quad (\text{K.6})$$

$$M_{ij}(t, *) = -\bar{\alpha} \frac{a_i(t)}{m} \mathbb{E}\{\sigma'(r_i(t)) \nabla_j \varphi(\mathbf{r}_0)\} + \frac{\bar{\alpha}}{m} \sum_{l=1}^m \mathbb{E}\{S_{il}(t) \frac{\partial r_l(t)}{\partial r_{0,j}}\}, \quad (\text{K.7})$$

$$C_{i,j}^E(t, s) = \frac{a_i(t)a_j(s)}{m^2} \mathbb{E}\{E(t)E(s)\sigma'(r_i(t))\sigma'(r_j(s))\} , \quad (\text{K.8})$$

$$S_{ij}(t) := -a_i(t) E(t)\sigma''(r_i(t))\delta_{ij} + \frac{a_i(t)a_j(t)}{m}\sigma'(r_i(t))\sigma'(r_j(t)), \quad (\text{K.9})$$

and

$$C_{ij}(t, s) = \mathbb{E}\left\{w_i^e(t)w_j^e(s)\right\}, \quad (\text{K.10})$$

$$\mathbb{P}_{-}(t, s) = \mathbb{E}\left[\partial w_i^e(t)\right], \quad (\text{K.11})$$

$$R_{ij}(t, s) = \mathbb{E}\left\{\frac{\partial w_i^e(t)}{\partial \eta_j(s)}\right\}. \quad (\text{K.11})$$

In solving the above, the random functions ∂w<sup>e</sup> i (t) ∂η<sup>j</sup> (s) and ∂ri(t) ∂ξ<sup>j</sup> (s) (for t > s) are defined to be solutions of the following linear ODEs:

$$\frac{d}{dt} \frac{\partial w_i^e(t)}{\partial \eta_j(s)} = -\nu_i(t) \frac{\partial w_i^e(t)}{\partial \eta_j(s)} - \frac{1}{m} \sum_{l=1}^m \int_s^t M_{i,l}(t, t') \frac{\partial w_l^e(t')}{\partial \eta_j(s)} dt , \quad (\text{K.12})$$

$$\frac{\partial r_i(t)}{\partial \xi_j(s)} = -\frac{1}{m} \sum_{l,q=1}^m \int_s^t R_{il}(t, t') S_{lq}(t') \left[ \frac{\partial r_q(t')}{\partial \xi_j(s)} + \delta_{qj} \delta(t' - s) \right] dt', \quad (\text{K.13})$$

$$\frac{\partial r_i(t)}{\partial r_{0,j}} = \frac{1}{m} \sum_{l=1}^m \int_0^t R_{il}(t, s) a_l(s) \sigma'(r_l(s)) \nabla_j \varphi(\mathbf{r}_0) ds - \frac{1}{m} \sum_{l,q=1}^m \int_0^t R_{il}(t, s) S_{lq}(s) \frac{\partial r_q(s)}{\partial r_{0,j}} ds, \quad (\text{K.14})$$

with boundary condition ∂w<sup>e</sup> i (t) ∂η<sup>j</sup> (t) = δij for the first equation.

## L Derivation of the dynamical mean field theory equations

The study of the dynamics in such high-dimensional limit can be done via dynamical mean field theory (DMFT) [\[18\]](#page-10-17). The theoretical technology that we will employ is an evolution of the one first derived in [\[31,](#page-11-11) [32\]](#page-11-12) to study gradient flow and stochastic gradient descent on models that are very much related to the Gaussian process we are discussing here [\[51,](#page-12-11) [42,](#page-12-13) [33\]](#page-11-10). We remark that the formalism considered here can be used to study both the single index model and the pure noise case. To obtain the pure noise model, one can set h<sup>t</sup> = ˆφ = 0. Furthermore, the extension to multi-index models can be also done easily on the same lines.

The analysis of Eqs. [\(A.10\)](#page-22-0) can be done by recasting them into a path integral representation. We follow the same procedure presented in [\[31\]](#page-11-11). Eqs.[\(A.10\)](#page-22-0) can be packed into a dynamical partition function

$$1 = Z_{dyn} = \int Da D\tilde{a} \int DW D\hat{W} \exp \left[ A[a, \tilde{a}, W, \hat{W}] \right] \quad (L.1)$$

where the path measure Da(t)Da˜(t)DWDWˆ is implicitly defined. The action A reads

$$A = i \sum_{l=1}^m \int \tilde{a}_l(t) \left[ d \frac{da_l(t)}{dt} + n \frac{\partial \widehat{\mathcal{R}}_n}{\partial a_l(t)} \right] dt + i \sum_{l=1}^m \int \langle \hat{\mathbf{w}}_l(t), d \frac{\mathbf{w}_l(t)}{dt} + d\nu_i(t) \mathbf{w}_i(t) + n \frac{\partial \widehat{\mathcal{R}}_n}{\partial \mathbf{w}_l(t)} \rangle dt. \quad (\text{L.2})$$

Eq. [\(L.2\)](#page-74-1) can be rewritten by introducing Grassmann variables [\[62\]](#page-13-6). Call aˆ = (ta, θa) a supertime coordinate, with θ<sup>a</sup> a Grassmann variable. Define, with a slight abuse of notation

$$\begin{aligned} \mathbf{w}_l(\hat{a}) &= \mathbf{w}_l(t_a) + i\theta_a\hat{\mathbf{w}}_l \\ a_l(\hat{a}) &= a_l(t_a) + i\theta_a\tilde{a}_l(t_a) \quad l \leq m. \end{aligned} \tag{L.3}$$

Eq. [\(L.2\)](#page-74-1) can be written as

$$A = \frac{d}{2} \sum_{i,j=1}^m \int_{\hat{a}, \hat{b}} \mathcal{K}_{ij}(\hat{a}, \hat{b}) \langle \mathbf{w}_i(\hat{a}), \mathbf{w}_j(\hat{b}) \rangle + \frac{d}{2} \sum_{i,j=1}^m \int_{\hat{a}, \hat{b}} \tilde{\mathcal{K}}_{ij}(\hat{a}, \hat{b}) a_i(\hat{a}) a_j(\hat{b}) - n \int_{\hat{a}} \hat{\mathcal{R}}_n(\boldsymbol{\theta}(\hat{a})) . \quad (\text{L.4})$$

The first two terms of the sum describe the kinetic terms of the dynamical equations of motion. The last term instead contains the interaction between the weights of the network. The empirical risk <sup>R</sup>b n depends on the training dataset. We are interested in understanding the behavior of the dynamics of gradient flow when we average over its realizations. Since the dynamical partition function is identically one we can average it directly over the dataset [<sup>2</sup>](#page-74-2) . In this way we have

$$1 = Z_{dyn} = \int D\mathbf{a}(\hat{a}) D\mathbf{W}(\hat{a}) \exp \left[ \frac{d}{2} \sum_{i,j=1}^m \int_{\hat{a}, \hat{b}} \mathcal{K}_{ij}(\hat{a}, \hat{b}) \langle \mathbf{w}_i(\hat{a}), \mathbf{w}_j(\hat{b}) \rangle \right. \\ \left. + \frac{d}{2} \sum_{l,l'=1}^m \int_{\hat{a}, \hat{b}} \tilde{\mathcal{K}}_{ll'}(\hat{a}, \hat{b}) a_l(\hat{a}) a_{l'}(\hat{b}) \right] \mathbb{E} \left[ \exp \left( -n \int_{\hat{a}} \widehat{\mathcal{R}}_n(\boldsymbol{\theta}(\hat{a})) \right) \right]. \quad (\text{L.5})$$

<sup>2</sup>We emphasize anyway that the average over the dataset is not mandatory: the resulting DMFT equations are self-averaging.

Performing standard manipulation, see [\[31\]](#page-11-11), the dynamical partition function, for <sup>d</sup> → ∞, can be written as

$$Z_{dyn} = \int D(\underline{a}, \tilde{Q}, R) \exp \left[ S_{dyn}(\underline{a}, \tilde{Q}, R) \right] . \quad (\text{L.6})$$

The dynamical action Sdyn is given by

$$\begin{aligned}
S_{dyn} &= \frac{d}{2} \sum_{ll'=1}^m \int_{\hat{a}\hat{b}} \mathcal{K}_{ll'}(\hat{a}, \hat{b}) \left( \tilde{\mathcal{Q}}_{ll'}(\hat{a}, \hat{b}) + r_l(\hat{a})r_{l'}(\hat{b}) \right) + \frac{d}{2} \ln \det(\tilde{\mathcal{Q}}) + \frac{\bar{\alpha}d}{2} \ln \det(\mathbf{I} + \Sigma_+) \\
&+ \frac{d}{2} \int_{\hat{a}\hat{b}} \sum_{ll'} \tilde{\mathcal{K}}_{ll'}(\hat{a}, \hat{b}) a_l(\hat{a}) a_{l'}(\hat{b})
\end{aligned} \tag{L.7}$$

where α = n/d and

$$\begin{aligned} \Sigma_+(\hat{a}, \hat{b}) &= \tau^2 + h_t(1) + \frac{1}{m^2} \sum_{l,l'=1}^m a_l(\hat{a})a_{l'}(\hat{b})h\left(\tilde{Q}_{ll'}(\hat{a}, \hat{b}) + r_l(\hat{a})r_{l'}(\hat{b})\right) \\ &\quad - \frac{1}{m} \sum_{l=1}^m a_l(\hat{a})\hat{\varphi}(r_l(\hat{a})) - \frac{1}{m} \sum_{l=1}^m a_l(\hat{b})\hat{\varphi}(r_l(\hat{b})) . \end{aligned} \quad (\text{L.8})$$

The kinetic kernels K and K˜ are implicitly defined in such a way that they reproduce the time derivative part of the dynamical equations [\(A.10\)](#page-22-0).

In the large d limit, fixing m and α, the path integral in Eq. [\(L.5\)](#page-74-3) concentrates on its saddle point. The corresponding equations are

$$0 = \sum_{\gamma=1}^m \int_{\hat{c}} \mathcal{K}_{l\gamma}(\hat{a}, \hat{c}) Q_{\gamma l'}(\hat{c}, \hat{b}) + \frac{\bar{\alpha}}{m} a_l(\hat{a}) \hat{\varphi}'(r_l(\hat{a})) r_{l'}(\hat{b}) \int_{\hat{d}} (\mathbf{I} + \Sigma)^{-1}(\hat{a}, \hat{d}) - \frac{\bar{\alpha}}{m^2} \sum_{\gamma=1}^m \int_{\hat{c}} (\mathbf{I} + \Sigma)^{-1}(\hat{a}, \hat{c}) a_l(\hat{a}) a_\gamma(\hat{c}) h'(Q_{l\gamma}(\hat{a}, \hat{c})) Q_{\gamma l'}(\hat{c}, \hat{b}) + \delta_{ll'}(\hat{a}, \hat{b}) \quad (\text{L.9})$$

and

$$0 = \sum_{\gamma=1}^m \int_{\hat{c}} \mathcal{K}_{l\gamma}(\hat{a}, \hat{c}) r_\gamma(\hat{c}) + \frac{\bar{\alpha}}{m} a_l(\hat{a}) \varphi'(r_l(\hat{a})) \int_{\hat{d}} (\mathbf{I} + \Sigma)^{-1}(\hat{a}, \hat{d}) - \frac{\bar{\alpha}}{m^2} \sum_{\gamma=1}^m \int_{\hat{c}} (\mathbf{I} + \Sigma)^{-1}(\hat{a}, \hat{c}) a_l(\hat{a}) a_\gamma(\hat{c}) h'(Q_{l\gamma}(\hat{c})) r_\gamma(\hat{c}) \quad (\text{L.10})$$

where

$$\begin{aligned}Q_{ll'}(\hat{a}, \hat{b}) &= \tilde{Q}_{ll'}(\hat{a}, \hat{b}) + r_l(\hat{a})r_{l'}(\hat{b}) \\ \Sigma(\hat{a}, \hat{b}) &= \tau^2 + h_t(1) + \frac{1}{m^2} \sum_{ll'}^m a_l(\hat{a})a_{l'}(\hat{b})h\left(Q_{ll'}(\hat{a}, \hat{b})\right) \\ &\quad - \frac{1}{m} \sum_{l=1}^m a_l(\hat{a})\hat{\varphi}(r_l(\hat{a})) - \frac{1}{m} \sum_{l=1}^m a_l(\hat{b})\hat{\varphi}(r_l(\hat{b})).\end{aligned}\tag{L.11}$$

If Lagrange multipliers are added to constrain the norm of the the weights of the first layer, one should provide additional equations for them. Finally the equations for the dynamics of the second layer weights are given by

$$\sum_{\gamma=1} \int_{\hat{\epsilon}} \tilde{\mathcal{K}}_{l\gamma}(\hat{a}, \hat{\epsilon}) a_{\gamma}(\hat{\epsilon}) = -\bar{\alpha} \int_{\hat{\epsilon}} (\mathbf{I} + \Sigma)^{-1} (\hat{c}, \hat{a}) \left[ \frac{1}{m^2} \sum_{\gamma=1}^m a_{\gamma}(\hat{\epsilon}) h[Q_{\gamma l}(\hat{c}, \hat{a})] - \frac{1}{m} \hat{\varphi}(r_l(\hat{a})) \right] \quad (\text{L.12})$$

Eqs. [\(L.9\)](#page-75-0)-[\(L.12\)](#page-75-1) contain all the information about the dynamics. In order to fully specify the behavior of physical quantities such has the train and test error, it is useful to unfold the Grassmann structure of Eqs. [\(L.9\)](#page-75-0)-[\(L.12\)](#page-75-1).

#### L.1 Unfolding the Grassmann structure

Causality of the dynamics implies that the following parametrization is the most general solution of the saddle point equations

$$\begin{aligned} r_\alpha(\hat{a}) &= r_\alpha(t_a) \\ a_\alpha(\hat{a}) &= a_\alpha(t_a) \\ Q_{\alpha\beta}(\hat{a}, \hat{b}) &= C_{\alpha\beta}(t_a, t_b) + \theta_a R_{\beta\alpha}(t_b, t_a) + \theta_b R_{\alpha\beta}(t_a, t_b) \\ (\mathbf{I} + \Sigma)^{-1}(\hat{a}, \hat{b}) &= C_A(t_a, t_b) + \theta_b R_A(t_a, t_b) + \theta_a R_A(t_b, t_a) . \end{aligned} \tag{L.13}$$

Plugging this parametrization into the saddle point equations we get that the correlators in Eqs. [\(L.13\)](#page-76-0) satisfy the following DMFT equations

$$\begin{aligned} \frac{da_\alpha(t)}{dt} &= -\frac{\bar{\alpha}}{m} \int_0^t R_A(t, s) \left[ \frac{1}{m} \sum_{l=1}^m a_l(s) h [C_{l\alpha}(s, t)] - \hat{\varphi}(r_\alpha(t)) \right] ds \\ &\quad - \frac{\bar{\alpha}}{m} \int_0^t C_A(t, s) \frac{1}{m} \sum_{l=1}^m a_l(s) h' [C_{l\alpha}(s, t)] R_{\alpha l}(t, s) ds \end{aligned} \quad (\text{L.14})$$

$$\begin{aligned} \frac{dr_\alpha(t)}{dt} &= -\nu_\alpha(t)r_\alpha(t) + \frac{\bar{\alpha}}{m}a_\alpha(t)\hat{\varphi}'(r_\alpha(t)) \int_0^t R_A(t, s)ds \\ &\quad - \frac{a_\alpha(t)}{m} \sum_{\gamma=1}^m \int_0^t M_{\alpha\gamma}^R(t, s)a_\gamma(s)r_\gamma(s)ds \end{aligned} \quad (\text{L.15})$$

$$\begin{aligned} \frac{\partial C_{\alpha\beta}(t_a, t_b)}{\partial t_a} &= -\nu_\alpha(t_a)C_{\alpha\beta}(t_a, t_b) + \frac{\bar{\alpha}}{m}a_\alpha(t_a)\dot{\varphi}'(r_\alpha(t_a))r_\beta(t_b) \int_0^{t_a} R_A(t_a, s) ds \\ &\quad - \frac{a_\alpha(t_a)}{m} \sum_{\gamma=1}^m \int_0^{t_a} M_{\alpha\gamma}^R(t_a, s) a_\gamma(s) C_{\gamma\beta}(s, t_b) ds \\ &\quad - \frac{a_\alpha(t_a)}{m} \sum_{\gamma=1}^m \int_0^{t_b} M_{\alpha\gamma}^C(t_a, s) a_\gamma(s) R_{\beta\gamma}(t_b, s) ds \end{aligned} \quad (\text{L.16})$$

$$\begin{aligned} \frac{\partial R_{\alpha\beta}(t_a, t_b)}{\partial t_a} &= -\nu_\alpha(t_a) R_{\alpha\beta}(t_a, t_b) + \delta_{\alpha\beta}(t_a - t_b) \\ &\quad - \frac{a_\alpha(t_a)}{m} \sum_{\gamma=1}^m \int_{t_b}^{t_a} M_{\alpha\gamma}^R(t_a, s) a_\gamma(s) R_{\gamma\beta}(s, t_b) ds . \end{aligned} \quad (\text{L.17})$$

Note that we used the notation according to which the prime sign denotes the derivatives of the functions with respect to their argument. The memory kernels M<sup>R</sup> and M<sup>C</sup> are defined by

$$M_{\alpha\gamma}^R(t, s) = \frac{\bar{\alpha}}{m} [R_A(t, s)h'(C_{\alpha\gamma}(t, s)) + C_A(t, s)h''(C_{\alpha\gamma}(t, s))R_{\alpha\gamma}(t, s)] \quad (\text{L.18})$$

$$M_{\alpha\gamma}^C(t, s) = \frac{\bar{\alpha}}{m} C_A(t, s)h'(C_{\alpha\gamma}(t, s)) .$$

The kernels in Eq. [\(L.18\)](#page-76-1) depend on R<sup>A</sup> and C<sup>A</sup> that are defined in Eqs. [\(L.13\)](#page-76-0). The corresponding equations are

$$\begin{aligned} \int_{t'}^t [\delta(t-s) + \Sigma_R(t,s)] R_A(s,t') ds &= \delta(t-t') \\ \int_0^t [\delta(t-s) + \Sigma_R(t,s)] C_A(s,t') ds + \int_0^{t'} \Sigma_C(t,s) R_A(t',s) ds &= 0 \end{aligned} \quad (\text{L.19})$$

where

$$\begin{aligned} \Sigma_C(t, s) &= \tau^2 + h_t(1) + \frac{1}{m^2} \sum_{l'=1}^m a_l(t) a_{l'}(s) h[C_{ll'}(t, s)] \\ &\quad - \frac{1}{m} \sum_{l=1}^m a_l(t) \hat{\varphi}(r_l(t)) - \frac{1}{m} \sum_{l=1}^m a_l(s) \hat{\varphi}(r_l(s)) \end{aligned} \quad (\text{L.20})$$

$$\Sigma_R(t, s) = \frac{1}{m^2} \sum_{l'=1}^m a_l(t) a_{l'}(s) h'[C_{ll'}(t, s)] R_{ll'}(t, s).$$

The Lagrange multipliers να(t) have to be fixed self-consistently to enforce that Cα,α(t, t) = 1 given that <sup>w</sup><sup>α</sup> ∈ <sup>S</sup> d−1 . The corresponding equations are

$$\begin{aligned} \nu_\alpha(t_a) &= \frac{\bar{\alpha}}{km} \sum_{\tau=1}^k a_\alpha(t_a) \hat{\varphi}'(r_{\tau\alpha}(t_a)) r_{\tau\alpha}(t_a) \int_0^{t_a} R_A(t_a, s) ds \\ &\quad - \frac{a_\alpha(t_a)}{m} \sum_{\gamma=1}^m \int_0^{t_a} M_{\alpha\gamma}^R(t_a, s) a_\gamma(s) C_{\gamma\alpha}(s, t_a) ds \\ &\quad - \frac{a_\alpha(t_a)}{m} \sum_{\gamma=1}^m \int_0^{t_a} M_{\alpha\gamma}^C(t_a, s) a_\gamma(s) R_{\gamma\alpha}(t_a, s) ds \end{aligned} \quad (\text{L.21})$$

Finally we need to add a set of equation to propagate the diagonal elements of the correlation matrix:

$$\frac{dC_{\alpha\beta}(t_a, t_a)}{dt_a} = \lim_{t' \rightarrow t_a} \left[ \frac{\partial C_{\alpha\beta}(t_a, t')}{\partial t_a} + \frac{\partial C_{\beta\alpha}(t_a, t')}{\partial t_a} \right]. \quad (\text{L.22})$$

These dynamical equations can be integrated from a set of initial conditions that fully specify the initial status of the neurons. We will consider a random initial condition for the weights of the first layer so that

$$\begin{aligned} & r_\alpha(0) = \forall \alpha = 1, \dots, m \\ & C_{\alpha \neq \beta}(0, 0) = \forall \alpha \neq \beta = 1, \dots, m \\ & C_{\alpha\alpha}(0, 0) = \forall \alpha = 1, \dots, m \\ & R_{\alpha\beta}(0, 0) = \forall \alpha, \beta = 1, \dots, m. \end{aligned} \tag{L.23}$$

Finally, the initial conditions for the weights of the last layer aα(0) are completely arbitrary. The solution of the DMFT equations gives access to the dynamics of the train and test error. The train error as a function of time is defined as

$$e_{\text{tr}}(t) = \lim_{d \rightarrow \infty} \widehat{\mathcal{H}}_n(t). \quad (\text{L.24})$$

A simple way to derive the expression of etr as a function of the solution of the DMFT equations in the <sup>d</sup> → ∞ limit is to consider a deformation of Eq. [\(L.5\)](#page-74-3) which consists in replacing

$$\exp \left( -n \int_{\hat{a}} \widehat{\mathcal{H}}_n(\hat{a}) \right) \rightarrow \exp \left( -n \int_{\hat{a}} P(\hat{a}) \widehat{\mathcal{H}}_n(\hat{a}) \right). \quad (\text{L.25})$$

For P(ˆa) = 1 we get back the original expression. The main idea of the derivation is to use P(ˆa) as a source field. In particular we have that

$$e_{\text{tr}}(t) = - \int d\theta_a \frac{\delta}{\delta P(\hat{a})} \ln Z_{dyn}[P] \Big|_{P=1}. \quad (\text{L.26})$$

 Note that the deformed dynamical partition function Zdyn[P] does not equal 1 for generic P so that the formula above makes perfectly sense. The deformation of the partition function produces a deformation of Sdyn in Eq. [\(L.7\)](#page-75-2) which consist in replacing

$$\frac{\bar{\alpha}d}{2} \ln \det(\mathbf{I} + \Sigma_+) \rightarrow \frac{\bar{\alpha}d}{2} \ln \det(\mathbf{I} + \Sigma_*) \quad (\text{L.27})$$

$$\Sigma_*(\hat{a}, \hat{b}) = P(\hat{a})\Sigma_+(\hat{a}, \hat{b}) .$$

Performing explicitly the derivatives with respect to P one gets

$$e_{\text{tr}}(t) = \frac{1}{2} \int_0^t [R_A(t, s) \Sigma_C(t, s) + C_A(t, s) \Sigma_R(t, s)] ds . \quad (\text{L.28})$$

The computation of the test error can be done in analogous way

$$\begin{aligned} e_{\text{ts}}(t) &= \lim_{d \rightarrow \infty} \frac{1}{2} \mathbb{E} \left[ \left( y_{\text{new}} - y_{\text{new}}^{(s)} \right)^2 \right] \\ &= \frac{1}{2} \left[ \tau^2 + \frac{1}{k} h_t(1) + \frac{1}{m^2} \sum_{l'}^m h[C_{ll'}(t, t)] - 2 \frac{1}{m} \sum_l^m \hat{\varphi}(r_l(t)) \right]. \end{aligned} \quad (\text{L.29})$$

The average in Eq. [\(L.29\)](#page-78-0) is performed over the training set and an additional datapoint, not presented in the training set and having the same statistical structure.

In summary, the solution of the DMFT equations gives access to the train and test error dynamics in the large dimensional limit. These equations can be integrated numerically very efficiently. Our goal is to understand their behavior for infinite number of neurons, <sup>m</sup> → ∞ at fixed sample complexity <sup>α</sup>. We will be mostly interested in two types of questions: first, given a dataset that is pure noise, what are the sample complexities at which the network is able to interpolate the dataset. Second: given a dataset built out of a single index process what is the dynamics of the test and train error.