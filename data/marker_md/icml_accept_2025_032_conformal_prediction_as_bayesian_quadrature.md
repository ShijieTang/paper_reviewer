# Conformal Prediction as Bayesian Quadrature

Jake C. Snell <sup>1</sup> Thomas L. Griffiths 1 2

# Abstract

As machine learning-based prediction systems are increasingly used in high-stakes situations, it is important to understand how such predictive models will perform upon deployment. Distributionfree uncertainty quantification techniques such as conformal prediction provide guarantees about the loss black-box models will incur even when the details of the models are hidden. However, such methods are based on frequentist probability, which unduly limits their applicability. We revisit the central aspects of conformal prediction from a Bayesian perspective and thereby illuminate the shortcomings of frequentist guarantees. We propose a practical alternative based on Bayesian quadrature that provides interpretable guarantees and offers a richer representation of the likely range of losses to be observed at test time.

# 1. Introduction

Machine learning systems based on deep learning are increasingly used in high-stakes settings, such as medical diagnosis or financial applications. These settings impose unique constraints on the performance of these systems: we want them to produce good outcomes in the aggregate, but also do so fairly and with a guarantee of a low probability of harm. However, predictive models based on deep learning can be difficult to interpret, and commercial models increasingly tend to offer little information about the techniques used in training. This creates a new challenge: How can we flexibly and reliably quantify the suitability of a model for deployment without making too many assumptions about how the model was trained or in which settings it will be used?

Recent research on quantifying uncertainty has employed methods based on conformal prediction [\(Vovk et al.,](#page-10-0) [2005\)](#page-10-0), which aim to provide guarantees for model performance in a distribution-free way. However, these techniques are based on ideas from frequentist statistics, making it difficult to incorporate prior knowledge that might be available about specific models. For example, in a particular setting we might have access to some information about the distribution of the data that is likely to be encountered, and can construct tighter guarantees on the performance of models by making use of this information. Moreover, they focus on controlling the expected loss averaged over many unobserved datasets rather than focusing on the actual set of observations.

In this paper, we show how methods for guaranteeing model performance can be understood and extended by viewing them from a Bayesian perspective. We develop a framework in which we explicitly model uncertainty in the quantile values associated with particular observations, providing a nonparametric tool for characterizing possible distributions where the model might be deployed that is appropriately constrained by observed data. This framework allows us to draw upon methods from the fields of statistical prediction analysis [\(Aitchison & Dunsmore,](#page-9-0) [1975\)](#page-9-0) and probabilistic numerics [\(Cockayne et al.,](#page-9-1) [2019;](#page-9-1) [Hennig et al.,](#page-9-2) [2022\)](#page-9-2) to develop guarantees that are interpretable and make adaptive use of available information.

We show that two popular uncertainty quantification methods, split conformal prediction [\(Vovk et al.,](#page-10-0) [2005;](#page-10-0) [Pa](#page-9-3)[padopoulos et al.,](#page-9-3) [2002\)](#page-9-3) and conformal risk control [\(An](#page-9-4)[gelopoulos et al.,](#page-9-4) [2024\)](#page-9-4), can both be recovered as special cases of our framework. Our approach gives a more complete characterization of the performance of these approaches, as we are able to determine the full distribution of possible outcomes rather than a single point estimate. Since our approach is grounded in Bayesian probability, we can easily incorporate knowledge relevant to evaluating the performance of these models when it is present, such as monotonicity or distributional assumptions, while defaulting to existing methods when absent. Our results show that Bayesian probability, while it is often discarded due to the apparent need to specify prior distributions, is actually well-suited for distribution-free uncertainty quantification.

<sup>1</sup>Department of Computer Science, Princeton University <sup>2</sup>Department of Psychology, Princeton University. Correspondence to: Jake C. Snell <jsnell@princeton.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

### 2. Background

Conformal prediction methods apply a wrapper on top of black-box predictive models to be able to subject them to statistical analysis. In order to generate meaningful predictions about future performance, it is assumed that we have access to a small calibration dataset that is representative of the deployment conditions. Performance on this dataset then provides the foundation for generating predictions about future performance. We begin by reviewing existing current distribution-free uncertainty quantification techniques and Bayesian quadrature methods.

#### 2.1. Distribution-free Uncertainty Quantification Techniques

Uncertainty quantification techniques provide guarantees on the future performance of a black-box predictive model mapping inputs X to outputs Y based on a calibration set consisting of X1, . . . , X<sup>n</sup> and Y1, . . . , Yn. Different approaches do so in different ways. For more information on these techniques, refer to [Shafer & Vovk](#page-9-5) [\(2008\)](#page-9-5) or [An](#page-9-6)[gelopoulos & Bates](#page-9-6) [\(2023\)](#page-9-6).

Split Conformal Prediction The goal of Split Conformal Prediction [\(Vovk et al.,](#page-10-0) [2005;](#page-10-0) [Papadopoulos et al.,](#page-9-3) [2002\)](#page-9-3) is to generate a prediction set or interval that contains the ground-truth output with high probability. This is often expressed in terms of the coverage level 1 − α. It relies on a score function s(x, y) which measures the disagreement between a predictor's output and the ground truth.

The conformal guarantee is

$$\Pr \left( Y_{n+1} \notin \mathcal{C}(X_{n+1}) \right) \leq \alpha, \quad (1)$$

where

$$\mathcal{C}(X_{n+1}) = \{y : s(X_{n+1}, y) \leq \hat{q}\} \quad (2)$$

and qˆ is the ⌈(n+1)(1−α)⌉ n quantile of s<sup>1</sup> = s(X1, Y1), . . . , s<sup>n</sup> = s(Xn, Yn). Here, C(Xn+1) is a prediction set or interval which aims to include the ground-truth output.

Conformal Risk Control In Conformal Risk Control [\(An](#page-9-4)[gelopoulos et al.,](#page-9-4) [2024\)](#page-9-4), the goal is to generalize conformal prediction to more general loss functions that are monotonic functions of a single parameter λ. Conformal Risk Control (CRC) proceeds by viewing the coverage guarantee [\(1\)](#page-1-0) as the expected value of a 0-1 loss. It is assumed that the maximum possible value of the loss is B and that the problem is "achievable" by design in that there exists some setting λmax that satisfies the conformal guarantee. Additionally, each loss function Li(λ) is assumed to be a monotonic non-increasing function of λ. The guarantee

offered by Conformal Risk Control is of the form

$$E\left(\ell(\mathcal{C}_{\hat{\lambda}}(X_{n+1}), Y_{n+1})\right) \leq \alpha, \quad (3)$$

where

$$\hat{\lambda} = \inf \left\{ \lambda : \frac{n}{n+1} \hat{R}_n(\lambda) + \frac{B}{n+1} \leq \alpha \right\} \quad (4)$$

and Rˆ <sup>n</sup>(λ) = <sup>1</sup> n P<sup>n</sup> <sup>i</sup>=1 Li(λ) is the empirical risk.

### 2.2. Bayesian Quadrature

Bayesian quadrature [\(Diaconis,](#page-9-7) [1988;](#page-9-7) [O'Hagan,](#page-9-8) [1991\)](#page-9-8) is a general technique for evaluating integrals that allows for uncertainty in the integrand. It estimates the value of an integral R <sup>b</sup> a f(x) dx by the following four steps: (1) place a prior p(f) on functions, (2) evaluate f at x1, x2, . . . , xn, (3) compute a posterior given the observed values of f by Bayes' rule, and (4) estimate R <sup>b</sup> a f(x) dx. Suppose that f(xi) = y<sup>i</sup> for i = 1, 2, . . . , n. The posterior over f is

$$p(f \mid x_{1:n}, y_{1:n}) \propto p(f) \prod_{i=1}^n \delta(y_i - f(x_i)), \quad (5)$$

where δ(·) is the Dirac delta function. The posterior mean then provides an estimate for the integral:

$$\int_a^b f(x) dx \approx \int_a^b f_n(x) dx, \text{ where } (6)$$

$$f_n(t) = E(f(t) \mid x_{1:n}, y_{1:n}). \quad (7)$$

It has been demonstrated that many classical quadrature procedures such as the trapezoid rule can be recovered by placing a Gaussian process prior on functions [\(Karvonen &](#page-9-9) [Sarkk](#page-9-9) ¨ a¨, [2017\)](#page-9-9).

#### 2.3. Summary and Prospectus

Bayesian quadrature provides an illustration of how a primarily numerical method can be connected to Bayesian inference, and in doing so potentially admit additional information about the underlying function that can be incorporated via a prior distribution. In next section, we will see how a similar approach can be applied to conformal prediction, identifying a Bayesian framework that reproduces existing distribution-free uncertainty quantification techniques. The challenge in doing so is that we want guarantees of the style obtained from Bayesian models, but we want to make the approach as general as possible in its assumptions about the underlying distribution. We solve this problem via an approach inspired by probabilistic numerics to construct a nonparametric characterization of the underlying distribution based on the calibration set.

### 3. Decision-theoretic Formulation

In this section we show how split conformal prediction and conformal risk control can be formulated as instances of a general decision problem.

Let z = (z1, . . . , zn) be a set of calibration data where each observation z<sup>i</sup> = (x<sup>i</sup> , yi) consists of an input and a ground truth label. Let θ denote the true state of nature that defines a shared density f(z<sup>i</sup> | θ) for the data.[<sup>1</sup>](#page-2-0) A new test point znew is assumed to have the same distribution. Let λ be a control parameter (e.g. threshold) that must be chosen based on the calibration data. We assume the presence of a loss function L(θ, λ) which quantifies the loss incurred by selecting λ when the true state of nature is θ.

The decision-theoretic goal is to choose a decision rule λ(z) that controls the *risk*:

$$R(\theta, \lambda) = \int L(\theta, \lambda(z)) f(z \mid \theta) dz. \quad (8)$$

It is often desirable to choose λ so that it is robust to any possible state of nature θ. The *maximum risk* is defined as

$$\bar{R}(\lambda) = \sup_{\theta} R(\theta, \lambda). \quad (9)$$

In distribution-free uncertainty quantification applications, it is often trivial to achieve arbitrarily low risk (for example by forming prediction sets covering the entire output space). We thus want to find decision rules whose risk is upper bounded by a constant α:

$$\bar{R}(\lambda) \leq \alpha, \quad (10)$$

and use another criterion (such as expected prediction set size) to select among these. We call a rule that satisfies [\(10\)](#page-2-1) an α*-acceptable decision rule*.

## 3.1. Recovering Split Conformal Prediction

We now show how split conformal prediction is a special case of this decision-theoretic problem. Let Lscp(θ, λ) be the *miscoverage loss*:

$$\begin{aligned} L_{\text{scp}}(\theta, \lambda) &= \Pr\{s(z_{\text{new}}) > \lambda\} \\ &= 1 - \Pr\{s(z_{\text{new}}) \leq \lambda\} \\ &= 1 - \int \mathbb{1}\{s(z_{\text{new}}) \leq \lambda\} f(z_{\text{new}} \mid \theta) dz_{\text{new}}, \end{aligned} \tag{11}$$

where s is an arbitrary nonconformity function.

Proposition 3.1. *Define* s<sup>i</sup> ≜ s(zi) *for* i = 1, . . . , n *and let* s(1) ≤ s(2) ≤ . . . ≤ s(n) *be the corresponding order* *statistics. Let* λscp *be the following decision rule:*

$$\lambda_{\text{scp}} = \begin{cases} s(\lceil (n+1)(1-\alpha) \rceil), & \text{if } \lceil (n+1)(1-\alpha) \rceil \leq n \\ \infty, & \text{otherwise.} \end{cases} \quad (12)$$

*Then* λscp *is an* α*-acceptable decision rule for the miscoverage loss* Lscp *defined in* [\(11\)](#page-2-2)*.*

*Proof.* Proofs for all theoretical results may be found in Appendix [B.](#page-11-0)

Therefore the prediction set can be constructed as in [\(2\)](#page-1-1):

$$\mathcal{C}_{\text{scp}}(x_{\text{new}}) = \{y \in \mathcal{Y} : s(x_{\text{new}}, y) \leq \lambda_{\text{scp}}\}, \quad (13)$$

and by Proposition [3.1,](#page-2-3) Cscp satisfies the conformal guarantee from [\(1\)](#page-1-0).

# 3.2. Recovering Conformal Risk Control

Conformal risk control generalizes split conformal prediction by considering losses that are monotonic non-increasing functions of a single parameter λ.

$$L_{\text{crc}}(\theta, \lambda) = \int \ell(z_{\text{new}}, \lambda) f(z_{\text{new}} \mid \theta) dz_{\text{new}}, \quad (14)$$

where ℓ(znew, λ) is an individual loss function that is monotonically non-increasing in λ.

Proposition 3.2. *Let* λcrc *be the following decision rule:*

$$\lambda_{\text{crc}} = \inf \left\{ \lambda : \frac{1}{n+1} \left( \sum_{i=1}^n \ell(z_i, \lambda) + B \right) \leq \alpha \right\}. \quad (15)$$

*Then* λcrc *is an* α*-acceptable decision rule for* Lcrc *defined in* [\(14\)](#page-2-4)*.*

Note in particular that when ℓ(z, λ) can be expressed in the form ℓ(Cλ(xn+1), yn+1), this recovers the conformal risk control guarantee from [\(3\)](#page-1-2).

# 4. Our Approach

We introduce our approach by reinterpreting split conformal prediction and conformal risk control as special cases of a more general Bayesian procedure. In order to do so, we borrow ideas from both Bayesian quadrature [\(Diaconis,](#page-9-7) [1988;](#page-9-7) [O'Hagan,](#page-9-8) [1991\)](#page-9-8) and distribution-free tolerance regions [\(Guttman,](#page-9-10) [1970\)](#page-9-10). Bayesian quadrature (Section [2.2\)](#page-1-3) solves a numerical integration problem by placing a prior on functions and using Bayesian inference to compute a distribution over the value of the integral. Distributionfree tolerance regions provide a distribution over quantile spacings that holds regardless of the original underlying distribution. Putting these ideas together allows us to extend

In the interest of notational convenience, we assume densities and integrals over z<sup>i</sup> but these may be replaced by probability mass functions and summations as appropriate.

conformal prediction by producing bounds on expected loss tailored to the actual losses observed in the calibration set.

The remainder of this section is structured as follows. In Section [4.1,](#page-3-0) we discuss the relationship between risk control and Bayes risk. In Section [4.2,](#page-3-1) we describe a general approach for using Bayesian quadrature to bound the posterior risk. In Section [4.3,](#page-3-2) we make the quadrature "distributionfree" by removing the dependence on a prior over functions. In Section [4.4](#page-4-0) we handle uncertainty in the evaluation locations of the function by applying results that characterize the spacing between consecutive quantiles. In Section [4.5,](#page-4-1) we show how to use these results to produce an upper bound on the expected loss. Finally, in Section [4.6,](#page-4-2) we show how previous conformal prediction techniques can be viewed as a special case of our procedure that only considers the expectation of the posterior loss.

# 4.1. Bayes Risk

The risk R(θ, λ) measures the expected loss for one who already knows the true state of nature θ but not the particular data observed. However, in practical applications the situation is reversed: we *do* know the observed data but there is uncertainty about the state of nature. Therefore, we want a decision rule that protects against high loss for a range of possible θ. This idea is expressed as the *integrated risk*:

$$r(\pi, \lambda) = \int R(\theta, \lambda) \pi(\theta) d\theta, \quad (16)$$

where the prior π(θ) ≥ 0 measures the relative importance of the different possible states of nature. It is well-known that the minimizer of the integrated risk is the so-called *Bayes decision rule*:

$$\lambda^\pi \triangleq \operatorname{argmin}_{\lambda} r(\lambda \mid z), \quad (17)$$

where r(λ | z) is the *posterior risk*

$$r(\lambda \mid z) = E(L_\lambda \mid z) = \int L(\theta, \lambda(z)) \pi(\theta \mid z) d\theta, \quad (18)$$

and π(θ | z) ∝ π(θ)f(z | θ). Interestingly, the worst-case integrated risk of a decision rule is identical to its maximum risk [\(9\)](#page-2-5)

$$\bar{r}(\lambda) \triangleq \sup_{\pi} r(\pi, \lambda) = \sup_{\theta} R(\theta, \lambda) = \bar{R}(\lambda). \quad (19)$$

We can therefore focus on bounding the worst-case integrated risk r¯(λ), since this will also bound the maximum risk R¯(λ).

# 4.2. Reformulation as Bayesian Quadrature

expectation over individual losses:

$$L(\theta, \lambda) = \int \ell(z_{\text{new}}, \lambda) f(z_{\text{new}} \mid \theta) dz_{\text{new}}. \quad (20)$$

It is well-known that the expectation of a random variable is equal to the definite integral of its quantile function over its domain [\(Shorack,](#page-9-11) [2000,](#page-9-11) p. 116). Consider the distribution function of individual losses induced by λ for a particular value of θ:

$$F(\ell) \triangleq \Pr\{\ell(z_{\text{new}}, \lambda) \leq \ell \mid \theta\} \quad (21)$$

The corresponding quantile function is:

$$K(t) \equiv F^{-1}(t) = \inf\{\ell : F(\ell) \geq t\}, \quad (22)$$

and the expected loss given <sup>K</sup> is simply R <sup>1</sup> <sup>0</sup> K(t) dt.

Instead of performing posterior inference over θ, we propose to take an approach inspired by Bayesian quadrature that places a corresponding prior over K. Figure [1](#page-4-3) shows a schematic overview of Bayesian quadrature in this setting and how our proposed approach differs. The posterior risk given the observed individual losses ℓ<sup>i</sup> ≜ ℓ(z<sup>i</sup> , λ) for i = 1, . . . , n becomes:

$$E(L \mid \ell_{1:n}) = \int J[K]p(K \mid \ell_{1:n}) dK, \quad (23)$$

where J[K] ≜ R 1 <sup>0</sup> K(t) dt and we have suppressed the dependence on λ for notational convenience. The posterior over quantile functions can be expressed as:

$$p(K \mid \ell_{1:n}) = \int p(K \mid t_{1:n}, \ell_{1:n}) p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \quad (24)$$

$$p(K \mid t_{1:n}, \ell_{1:n}) \propto \pi(K) \prod_{i=1}^n \delta(\ell_i - K(t_i)). \quad (25)$$

This resembles the Bayesian quadrature problem from Section [2.2,](#page-1-3) except the evaluation sites t1, . . . , t<sup>n</sup> are unknown. Fortunately, the distribution of t1, . . . , t<sup>n</sup> is independent of the true distribution of the losses, as we shall now show.

## 4.3. Elimination of the Prior Distribution

In order to address the dependence of the posterior risk on the prior π(K), we derive an upper bound on the posterior expected loss. The bound takes the form of a weighted sum of the observed losses, where the weights are determined by the spacing between consecutive quantiles.

Theorem 4.1. *Let* t(0) = 0*,* t(n+1) = 1*, and* ℓ(n+1) = B*. Then*

$$\sup_{\pi} E(L \mid t_{1:n}, \ell_{1:n}) \leq \sum_{i=1}^{n+1} u_i \ell_i(), \quad (26)$$

![](_page_4_Figure_1.jpeg)

Figure 1. Overview of our approach. Left: Standard Bayesian quadrature places a prior over the quantile function of the loss distribution. The posterior is formed via Bayes' rule after observing a set of loss values and quantile levels. However, in practice quantile levels are not directly observed. Middle: Our approach combines properties of quantile spacings with a right rectangular integration rule to construct an upper bound on the posterior distribution of the expected loss. Randomly sampled spacings and corresponding quantile functions are shown in blue along with a 95% credible interval for each quantile level in black. Right: The posterior distribution for a random variable L <sup>+</sup> that upper bounds the expected loss is constructed by integrating over the unknown quantile levels.

Theorem [4.1](#page-3-4) is based on the definite integral of the "worstcase" quantile function that is consistent with the observations. This strategy eliminates the need to specify a prior or evaluate an integral over functions K. We now turn our attention to handling the uncertainty over the quantiles t1:n.

#### 4.4. Random Quantile Spacings

We now appeal to a result about distribution-free tolerance regions that characterizes the distribution of spacings between consecutive ordered quantiles. Knowledge of this distribution will allow us to handle the input noise in the quadrature problem.

Lemma 4.2 (Distribution of Quantile Spacings [\(Aitchison](#page-9-0) [& Dunsmore,](#page-9-0) [1975,](#page-9-0) p. 140)). *Suppose that* ℓ1, . . . , ℓ<sup>n</sup> *are drawn i.i.d. with continuous*[<sup>2</sup>](#page-4-4) *distribution function* F*. Let* t<sup>i</sup> = F(ℓi) *and* u<sup>i</sup> = t(i) − t(i−1)*, where by convention* <sup>t</sup>(0) = 0 *and* <sup>t</sup>(n+1) = 1*. Then* (u1, u2, . . . , un+1) <sup>∼</sup><sup>=</sup> Dir(1, . . . , 1)*.*

We are now ready to present our algorithm for bounding the expected loss E(L | ℓ1:n).

## 4.5. Bound on Maximum Posterior Risk

Putting together Lemma [4.2](#page-4-5) and Theorem [4.1](#page-3-4) allows us to bound the maximum posterior risk.

Theorem 4.3. *Define* ℓ(i) *to be the order statistics of* ℓ1, . . . , ℓ<sup>n</sup> *for* i = 1, . . . , n *and* ℓ(n+1) ≜ B*. Let* L <sup>+</sup> *be*

*the random variable defined as follows:*

$$U_1, \dots, U_{n+1} \sim \text{Dir}(1, \dots, 1), \quad L^+ = \sum_{i=1}^{n+1} U_i \ell_{(i)}. \quad (27)$$

*Then for any* b ∈ (−∞, B]*,*

$$\inf_{\pi} \Pr(L \leq b \mid \ell_{1:n}) \geq \Pr(L^+ \leq b). \quad (28)$$

Theorem [4.3](#page-4-6) states that L <sup>+</sup> stochastically dominates the posterior risk[<sup>3</sup>](#page-4-7) , which allows us to directly form upper confidence bounds as follows.

Corollary 4.4. *For any desired confidence level* β ∈ (0, 1)*, define*

$$b_\beta^* = \inf_b \{b : \Pr(L^+ \leq b \mid \ell_{1:n}) \geq \beta\}. \quad (29)$$

*Then* inf<sup>π</sup> Pr(L ≤ b | ℓ1:n) ≥ β *for any* b ≥ b ∗ β *.*

The critical value b ∗ β can be calculated by applying techniques for bounding linear combinations of Dirichlet random variables [\(Ng et al.,](#page-9-12) [2011,](#page-9-12) p. 63). Alternatively, straightforward Monte Carlo simulation of L <sup>+</sup> is often sufficient, and is the approach we take in our experiments. An illustration is shown in Figure [2.](#page-5-0)

### 4.6. Recovering Conformal Methods

This perspective puts the previous distribution-free uncertainty techniques in a new light. Taking the expected value

<sup>2</sup>The correspondence to a Dirichlet distribution holds exactly for continuous distributions. Weighted sums of Dirichlet random variates stochastically dominate weighted sums of discrete quantile spacings, and thus due to space constraints we only consider continuous distributions here.

<sup>3</sup>We note that the distribution of the U<sup>i</sup> conditions just on the order of the losses and not their specific values. This assumption is consistent with the frequentist treatment of conformal prediction, which assumes that a new observation is equally likely to fall into each of the intervals formed by the loss values and the endpoints.

![](_page_5_Figure_1.jpeg)

Figure 2. Our Bayesian approach to conformal prediction accounts for the variability in quantile levels better than previous approaches. Left: Conformal Risk Control [\(Angelopoulos et al.,](#page-9-4) [2024\)](#page-9-4) considers only the expectation over the unobserved quantile values t1, . . . , tn. This can underestimate the true expected loss (shown here: estimated expected loss 0.45 vs. true expected loss 0.50). Right: Our approach makes use of the fact that the quantile spacings are drawn from a Dirichlet distribution. By considering the full distribution over quantiles, we gain a more complete view of the expected loss. Shown here is one sample drawn from this distribution, which estimates the expected loss as 0.58.

of L <sup>+</sup>, we find

$$E(L^+) = \sum_{i=1}^{n+1} E(U_i) \ell_{(i)} = \frac{1}{n+1} \left( \sum_{i=1}^n \ell_i + B \right). \quad (30)$$

The Conformal Risk Control decision rule [\(15\)](#page-2-6) then is simply the infimum over λ for which E(L <sup>+</sup>) ≤ α.

For, split conformal prediction, the individual loss is defined as ℓ<sup>i</sup> = 1 − <sup>1</sup>{s<sup>i</sup> ≤ λ}. Therefore, suppose that λ = s(k) . The expected value of L <sup>+</sup> then becomes:

$$E(L^+) = \frac{1}{n+1} \left( n+1 - \sum_{i=1}^n \mathbb{1}\{s_i \leq s_{(k)}\} \right) \quad (31)$$

$$= 1 - \frac{k}{n+1} \quad (32)$$

Therefore, E(L <sup>+</sup>) ≤ α is satisfied whenever k ≥ (n + 1)(1−α), and in particular by k <sup>∗</sup> = ⌈(n+ 1)(1−α)⌉. This recovers [\(12\)](#page-2-7) when ⌈(n + 1)(1 − α)⌉ ≤ n.

Putting these results together, we have recovered standard conformal prediction techniques but have the additional flexibility of considering the distribution of L <sup>+</sup> rather than the expected value alone. Our experiments explore the value of this approach.

# 5. Experiments

The primary goal of our experiments is to demonstrate the utility of producing a posterior distribution over the expected loss. We conduct experiments on both synthetic data and calibration data collected from MS-COCO [\(Lin et al.,](#page-9-13) [2014\)](#page-9-13). For each data setting, we randomly generate M = 10,000 data splits. Each method is used to select λ with the goal

of controlling the risk such that R(θ, λ) ≤ α for unknown θ. We compare algorithms on the basis of both the relative frequency of incurring risk greater than α and the prediction set size of the chosen λ. The ideal algorithm would select λ such that the relative frequency of exceeding the target risk is at most a target failure rate of 1 − β = 0.05 while minimizing prediction set size.

As demonstrated in Section [4.6,](#page-4-2) our method recovers conformal risk control by taking the expected value of L +. Therefore, in order to demonstrate the effect of targeting a conditional guarantee (as opposed to a marginal one as in conformal risk control), we use our Bayesian quadraturebased method to compute the decision rule based on the one-sided highest posterior density (HPD) interval:

$$\lambda_{\text{hpd}}^\beta \triangleq \inf_\lambda \{\lambda : \Pr(L^+ \leq \alpha \mid \ell_{1:n}) \geq \beta\}, \quad (33)$$

by finding the corresponding critical values b ∗ β according to [\(29\)](#page-4-8) via Monte Carlo simulation of Dirichlet random variates with 1000 samples. We include Risk-controlling Prediction Sets (RCPS) [\(Bates et al.,](#page-9-14) [2021\)](#page-9-14) with Hoeffding upper confidence bound as an additional baseline. Code for our experiments is publicly available on Github.[<sup>4</sup>](#page-5-1)

## 5.1. Synthetic Binomial Data

We first sample directly from a known loss distribution so that we can directly compute the frequency of excessively large risk. Here the loss distribution is chosen to be a scaled binomial distribution, normalized to have a maximum loss of B = 1 and probability of failure set to 1 − λ. This was

<sup>4</sup>[https://github.com/jakesnell/](https://github.com/jakesnell/conformal-as-bayes-quad) [conformal-as-bayes-quad](https://github.com/jakesnell/conformal-as-bayes-quad)

Table 1. Relative frequency of trials (out of 10,000) for which the resulting decision rule λ exceeded the target risk threshold α.

| Decision Rule     | Relative Freq. | 95% CI           |
|-------------------|----------------|------------------|
| CRC               | 21.20%         | [20.40%, 22.01%] |
| RCPS              | 0.00%          | [0.00%, 0.04%]   |
| Ours ( β = 0 95 ) | 0.03%          | [0.01%, 0.09%]   |

Note: Error bars are computed as 95% Clopper-Pearson confidence intervals for binomial proportions.

simulated by computing

$$\ell(z_i, \lambda) = \frac{1}{K} \sum_{k=1}^K \mathbb{1}\{V_{ik} > \lambda\}, \quad (34)$$

where Vik ∼ Uniform(0, 1) for i = 1, . . . , n and k = 1, . . . , K. This loss is therefore monotonically nonincreasing in λ and achieves zero loss at λmax = 1. We set n = 10, K = 4, and α = 0.4.

Since the expectation of the loss [\(34\)](#page-6-0) is 1 − λ, any trial for which λ < 0.6 constitutes a risk exceeding the α threshold. The relative frequency of trials exceeding this risk threshold are tabulated in Table [5.1.](#page-6-1) A histogram of the chosen λ for each of the methods across all 10,000 trials is shown in Figure [3.](#page-7-0) For conformal risk control, the mean risk across all trials was 0.3363 ± 0.0007 and for our approach λ 0.95 hpd the mean risk was 0.1758 ± 0.0006. In order to visualize the distribution of L <sup>+</sup>, we plot a histogram of L <sup>+</sup> according to [\(27\)](#page-4-9) estimated with 100,000 Dirichlet samples for three settings of λ ∈ {0.7, 0.8, 0.9}. The results are shown in Figure [4.](#page-8-0)

#### 5.2. Synthetic Heteroskedastic Data

In this experiment we also use 10,000 random trials. We use n = 200 calibration samples each. To achieve heteroskedasticity, we let X ∼ U[0, 4] and Y | X ∼ N (0, X<sup>2</sup> ). The prediction intervals are then formed as [−λ, ˆ λˆ] where λˆ is selected by each method. The loss is the miscoverage loss and the target loss is set to α = 0.1 (i.e. 90% coverage). The maximum allowable risk failure rate is set to 5% (i.e. β = 0.95). The results are show in Table [5.2.](#page-7-1)

#### 5.3. False Negative Rate on MS-COCO

We also compare methods on controlling the false negative rate of multilabel classification on the MS-COCO dataset [\(Lin et al.,](#page-9-13) [2014\)](#page-9-13). The experimental setup mirrors that used by [Angelopoulos & Bates](#page-9-6) [\(2023,](#page-9-6) Section 5.1). Each random split contains 1000 calibration examples and 3952 test examples. The results of this experiment are summarized in Table [5.3.](#page-7-2)

#### 6. Discussion

Our results in Table [5.1](#page-6-1) demonstrate that even though the Conformal Risk Control marginal guarantee holds, a significant number of individual trials (21.20%) may incur risk exceeding the target threshold. In contrast, by using the more conservative HPD criterion, very few of the trials (0.03%) exceeded the target risk. In Table [5.2,](#page-7-1) both RCPS and our method achieve failure rate below the target of 5% but our method achieves significantly smaller prediction intervals.

These results point to the qualitative difference in a marginal guarantee, which averages over many possible yet unobserved data sets vs. a conditional guarantee which focuses on knowledge about the state of nature conditioned on the calibration data actually observed. Previous work on conditional guarantees [\(Barber et al.,](#page-9-15) [2021;](#page-9-15) [Gibbs et al.,](#page-9-16) [2024\)](#page-9-16) has focused on input-conditional guarantees, where the guarantee is conditioned on for all in the input domain. Guarantees of this nature have been shown to be generally impossible without stronger distribution assumptions. Our guarantees are perhaps better characterized by the term "data-conditional guarantee", where we condition on the set of observed loss values. Our experiments demonstrate the practical benefits of this by achieving decisions that produce smaller prediction sets and intervals while not violating the constraint on maximum allowable failure rate. Our guarantees, in contrast, do not rely on strong distribution assumptions that would be necessary to produce an input-conditional guarantee.

The results are again confirmed in Table [5.3](#page-7-2) on MS-COCO, which show that the marginal guarantees of Conformal Risk Control lead to an even greater percentage of trials exceeding the risk threshold. On the other hand, RCPS is able to control the risk but this comes at the cost of larger prediction sets. Our approach successfully balances these two concerns, producing prediction intervals that are shorter than baselines while not exceeding the maximum acceptable failure rate. It is also clear that the distribution of the expected loss upper bound L <sup>+</sup> in Figure [4](#page-8-0) provides a more complete view of the range of possible losses and its dependence on λ, a perspective that is not offered by previous methods.

Our goal in this work is to show that the Bayesian viewpoint unlocks a richer interpretation compared to previous works, which focus on marginal guarantees that as we have shown in the paper correspond to the posterior mean. In order to draw an explicit correspondence between our work and previous approaches, the dependence on the prior was removed in Section [4.3.](#page-3-2) The intuition is that that any rational decision maker operating according to the rules of probability, regardless of prior (sufficiently expressive), would agree with the upper-bounding distribution of we derive. Naturally, commitment to a specific choice of prior would

![](_page_7_Figure_1.jpeg)

Figure 3. Comparison of risk incurred by each procedure across multiple trials. Left: Histogram of the decision rule λcrc chosen by Conformal Risk Control across M = 10,000 randomly sampled calibration sets. The region where per-trial risk exceeds α is highlighted in red. Right: Histogram of the λ 0.95 hpd chosen according to our 95% Bayesian posterior interval.

Table 2. Relative frequency of trials (out of 10,000) for which the resulting decision rule λ exceeded the target risk threshold α in the synthetic heteroskedastic experiment.

| Decision Rule                    | Relative Freq. | 95% CI           | Mean Prediction Interval Length |
|----------------------------------|----------------|------------------|---------------------------------|
| Split Conformal Prediction / CRC | 46.19%         | [45.21%, 47.17%] | 7.99                            |
| RCPS                             | 0.0%           | [0.0%, 0.04%]    | 14.29                           |
| Ours ( β = 0 95 )                | 3.42%          | [3.07%, 3.80%]   | 9.50                            |

Note: Error bars are computed as 95% Clopper-Pearson confidence intervals for binomial proportions.

Table 3. Results on MS-COCO comparing relative frequency of trials for which the resulting decision rule λ exceeded the target risk threshold α and average prediction set size.

| Method   |          | Relative Freq. | Pred. Set Size |
|----------|----------|----------------|----------------|
| CRC      |          | 45.05%         | 2.92           |
| RCPS     |          | 0.0%           | 3.57           |
| Ours ( β | = 0 95 ) | 5.43%          | 3.04           |

lead to tighter distributions over the posterior risk, and in future work we seek to bridge these fields even further by exploring specific choices of priors over quantile functions.

The limitations of our method lie primarily in the two main assumptions it makes. First, it assumes that the data at deployment time are independent and identically distributed to the calibration data. Second, it assumes an upper bound B on the losses. If either of these assumptions do not hold, then the guarantees produced by our method are no longer valid. Additionally, the bounds produced by our method are conservative in the sense that they hold for any choice of prior for the loss distribution (provided that the prior is consistent with the calibration data). Therefore, if the two

aforementioned assumptions do hold, the actual loss values may be significantly less than indicated by our method.

Overall, our approach demonstrates how conformal prediction techniques can be recovered and extended using Bayesian probability, all without having to specify a prior distribution. This Bayesian formulation is highly flexible due to its nonparametric nature, yet is amenable to incorporating specific information about the distribution of losses likely to be encountered. In practical applications, maximizing the risk with respect to all possible priors may be too conservative, and thus future work may explore the effect of specific priors on the risk estimate.

# 7. Related Work

Statistical Prediction Analysis. Statistical prediction analysis [\(Aitchison & Dunsmore,](#page-9-0) [1975\)](#page-9-0) deals with the use of statistical inference to reason about the likely outcomes of future prediction tasks given past ones. Within statistical prediction analysis, the area of distribution-free prediction assumes that the parameters or the form of the distributions involved cannot be identified. This idea can be traced back to [Wilks](#page-10-1) [\(1941\)](#page-10-1), who constructed a method to form

![](_page_8_Figure_1.jpeg)

Figure 4. Probability density for L <sup>+</sup> with λ ∈ {0.7, 0.8, 0.9} estimated using 100,000 Dirichlet samples.

distribution-free tolerance regions. [Tukey](#page-10-2) [\(1947;](#page-10-2) [1948\)](#page-10-3) generalized distribution-free tolerance regions and introduced the concept of statistically equivalent blocks, which are analogous to the intervals between consecutive order statistics of the losses. Much of the relevant theory is summarized by [Guttman](#page-9-10) [\(1970\)](#page-9-10), and the Dirichlet distribution of quantile spacing is discussed by [Aitchison & Dunsmore](#page-9-0) [\(1975\)](#page-9-0). We build upon these works by connecting them to Bayesian quadrature and applying them in the more modern context of distribution-free uncertainty quantification.

Bayesian Quadrature. The use of Bayesian probability to represent the outcome of a arbitrary computation is termed *probabilistic numerics* [\(Cockayne et al.,](#page-9-1) [2019;](#page-9-1) [Hennig et al.,](#page-9-2) [2022\)](#page-9-2). Since our approach is fundamentally based on integration, we focus primarily on the relationship with the more narrow approach of Bayesian quadrature, which employs Bayes rule to estimate the value of an integral. A lucid overview of this approach is discussed under the term *Bayesian numerical analysis* by [Diaconis](#page-9-7) [\(1988\)](#page-9-7), who traces it back to the late nineteenth century [\(Poincare´,](#page-9-17) [1896\)](#page-9-17). The use of Gaussian processes in performing Bayesian quadrature is discussed in detail by [O'Hagan](#page-9-8) [\(1991\)](#page-9-8). Our approach is formulated similarly but differs in two main ways: (a) we use a conservative bound instead of an explicit prior, and (b) we have input noise induced by the random quantile spacings.

Distribution-Free Uncertainty Quantification. Relevant background on distribution-free uncertainty quantification techniques is discussed in Section [2.1.](#page-1-4) A recent and comprehensive introduction to conformal prediction and related techniques may be found in [\(Angelopoulos & Bates,](#page-9-6) [2023\)](#page-9-6). Some recent works, like ours, also make use of quantile functions [\(Snell et al.,](#page-10-4) [2023;](#page-10-4) [Farzaneh et al.,](#page-9-18) [2024\)](#page-9-18) but remain grounded in frequentist probability. Another class of related work seeks to endow conformal prediction with properties naturally enjoyed by Bayesian methods, including the ability to incorporate prior information [\(Hoff,](#page-9-19) [2023\)](#page-9-19), produce predictive distributions [\(Vovk et al.,](#page-10-5) [2017\)](#page-10-5), or adapt to time-varying data distributions [\(Prinster et al.,](#page-9-20) [2024\)](#page-9-20). [Stan](#page-10-6)[ton et al.](#page-10-6) [\(2023\)](#page-10-6) show how to apply conformal prediction to Bayesian optimization, another setting where maintaining a posterior over functions is important. Separately, Bayesian approaches to predictive uncertainty are popular [\(Hobb](#page-9-21)[hahn et al.,](#page-9-21) [2022\)](#page-9-21) but make extensive assumptions about the form of the underlying predictive model. To our knowledge, we are the first to apply statistical prediction analysis and Bayesian quadrature in order to analyze the performance of black-box predictive models in a distribution-free way.

# 8. Conclusion

Safely deploying black-box predictive models, such as those based on deep neural networks, requires developing methods that provide guarantees of their performance. Existing techniques for solving this problem are based on frequentist statistics, and are thus difficult to extend to incorporate knowledge about the situation in which models may be deployed. In this work we provided a Bayesian alternative to distribution-free uncertainty quantification, showing that two popular existing methods are special cases of this approach. Our results show that Bayesian probability can be used to extend uncertainty quantification techniques, making their underlying assumptions more explicit, allowing incorporation of additional knowledge, and providing a more intuitive foundation for constructing performance guarantees that avoid overly-optimistic guarantees that can be produced by existing methods.

# Impact Statement

This paper introduces a practical algorithm for computing a posterior distribution for the expected loss based on the observed losses from a set of calibration data. The intended purpose of this algorithm is for the posterior distribution to inform deployment decisions of black-box predictive systems (e.g. deep neural networks) in safety-critical applications. Our method makes use of certain assumptions, discussed in Section [6,](#page-6-2) which if violated will lead to guarantees that may no longer hold. In particular, it is important to ensure proper monitoring to detect distribution shift between calibration and deployment.

# Acknowledgements

The authors would like to thank the anonymous reviewers for helpful comments and Hamed Hassani for helpful discussions about the posterior quantile distribution. We also

- would like to thank Drew Prinster for bringing several related works to our attention. This work was supported by grant N00014-23-1-2510 from the Office of Naval Research. References Aitchison, J. and Dunsmore, I. R. *Statistical Prediction Analysis*. New York: Cambridge University Press, 1975. Angelopoulos, A. N. and Bates, S. Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4):494–591, 2023. Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal risk control. In *The Twelfth International Conference on Learning Representations*, 2024. Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, `
- R. J. The limits of distribution-free conditional predictive inference. *Information and Inference: A Journal of the IMA*, 10(2):455–482, June 2021. Bates, S., Angelopoulos, A., Lei, L., Malik, J., and Jordan, M. Distribution-free, risk-controlling prediction sets. *Journal of the ACM*, 68(6), 2021. Cockayne, J., Oates, C. J., Sullivan, T. J., and Girolami, M. Bayesian probabilistic numerical methods. *SIAM Review*, 61(3):756–789, 2019. Diaconis, P. Bayesian numerical analysis. In Berger, J. and Gupta, S. (eds.), *Statistical Decision Theory and Related Topics IV*, volume 1, pp. 163–175. Springer-Verlag, 1988. Farzaneh, A., Park, S., and Simeone, O. Quantile learnthen-test: Quantile-based risk control for hyperparameter optimization. *IEEE Signal Processing Letters*, 2024. Gibbs, I., Cherian, J. J., and Candes, E. J. Conformal Pre- ` diction With Conditional Guarantees, September 2024. Guttman, I. *Statistical Tolerance Regions: Classical and Bayesian*. Griffin's Statistical Monographs and Courses, No. 26. Griffin, 1970. Hennig, P., Osborne, M. A., and Kersting, H. P. *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press, 2022. Hill, B. M. Posterior distribution of percentiles: Bayes' theorem for sampling from a population. *Journal of the American Statistical Association*, 63(322):677–691, June 1968. Hill, B. M. de Finetti's theorem, induction, and A<sup>n</sup> or Bayesian nonparametric predictive inference (with discussion). *Bayesian statistics*, 3:211–241, 1988. Hobbhahn, M., Kristiadi, A., and Hennig, P. Fast predictive uncertainty for classification with Bayesian deep networks. In *Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence*, 2022. Hoff, P. Bayes-optimal prediction with frequentist coverage control. *Bernoulli*, 29(2), May 2023. Karvonen, T. and Sarkk ¨ a, S. Classical quadrature rules via ¨ Gaussian processes. In *2017 IEEE 27th International Workshop on Machine Learning for Signal Processing (MLSP)*, 2017. Kot, M. *A First Course in the Calculus of Variations*. American Mathematical Society, 2014. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523):1094–1111, July 2018. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft ´ COCO: Common objects in context. In *13th European Conference on Computer Vision*, pp. 740–755. Springer, 2014. Ng, K. W., Tian, G.-L., and Tang, M.-L. *Dirichlet and Related Distributions: Theory, Methods and Applications*. Wiley, 2011. O'Hagan, A. Bayes–Hermite quadrature. *Journal of Statistical Planning and Inference*, 29(3):245–260, November 1991. Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. Inductive confidence machines for regression. In Elomaa, T., Mannila, H., and Toivonen, H. (eds.), *ECML 2002*, volume 2430, pp. 345–356. Springer Berlin Heidelberg, 2002. Poincare, H. ´ *Calcul des Probabilites´* . Georges Carre, 1896. ´ Prinster, D., Stanton, S., Liu, A., and Saria, S. Conformal validity guarantees exist for any data distribution (and how to find them). In *Proceedings of the 41st International Conference on Machine Learning*, 2024. Shafer, G. and Vovk, V. A tutorial on conformal prediction. *Journal of Machine Learning Research*, 9(12):371–421, 2008. Shao, J. *Mathematical Statistics*. Springer, 2003. Shorack, G. *Probability for Statisticians*. Springer-Verlag, 2000. Shorack, G. R. and Wellner, J. A. *Empirical Processes with Applications to Statistics*. Society for Industrial and Applied Mathematics, 2009.

Snell, J. C., Zollo, T. P., Deng, Z., Pitassi, T., and Zemel, R. Quantile risk control: A flexible framework for bounding the probability of high-loss predictions. In *The Eleventh International Conference on Learning Representations*, 2023. Stanton, S., Maddox, W., and Wilson, A. G. Bayesian Optimization with Conformal Prediction Sets. In *Proceedings of The 26th International Conference on Artificial Intelligence and Statistics*, pp. 959–986, April 2023. Tukey, J. W. Nonparametric estimation II. Statistically equivalent blocks and tolerance regions–the continuous case. *The Annals of Mathematical Statistics*, pp. 529–539, 1947. Tukey, J. W. Nonparametric estimation, III. Statistically equivalent blocks and multivariate tolerance regions–the discontinuous case. *The Annals of Mathematical Statistics*, pp. 30–39, 1948. Vovk, V., Gammerman, A., and Shafer, G. *Algorithmic Learning in a Random World*. Springer Science & Business Media, 2005. Vovk, V., Shen, J., Manokhin, V., and Xie, M. Nonparametric predictive distributions based on conformal prediction. In *Proceedings of the Sixth Workshop on Conformal and Probabilistic Prediction and Applications*, pp. 82–102. PMLR, May 2017. Wilks, S. S. Determination of sample sizes for setting tolerance limits. *The Annals of Mathematical Statistics*, 12(1):91–96, 1941.

# A. Theoretical Preliminaries

### A.1. Review of Problem Setup

We first review some relevant aspects of our problem setup.

Loss Function. We assume an upper bound on the losses: ℓ<sup>i</sup> ∈ (−∞, B] for i = 1, . . . , n. We assume the same upper bound for ℓnew.

Bayesian Quadrature of Quantile Functions. Recall Bayes rule for quantile functions:

$$p(K \mid t_{1:n}, \ell_{1:n}) \propto \pi(K) \prod_{i=1}^n \delta(\ell_i - K(t_i)), \quad (35)$$

where δ is the Dirac delta function. The prior π(K) is assumed to be sufficiently expressive to have nonzero measure for the set K<sup>n</sup> of quantile functions such that K(ti) = ℓ<sup>i</sup> for i = 1, . . . , n and K ∈ Kn. This is necessary to prevent the posterior distribution in [\(35\)](#page-11-1) from becoming degenerate.

## A.2. Background

We begin by recalling some basic properties of distribution functions and quantile functions.

Proposition A.1 (Properties of Distribution Functions [\(Shao,](#page-9-22) [2003,](#page-9-22) p. 4)). *Let* F(x) = Pr(X ≤ x) *be a distribution function. Then* F(−∞) = limx→−∞ F(x) = 0*,* F(∞) = limx→∞ F(x) = 1*,* F *is nondecreasing (i.e.,* F(x) ≤ F(y) *if* x ≤ y*), and* F *is right continuous (i.e.,* limy→x,y>x F(y) = F(x)*).*

Let F be a distribution function and K(t) ≡ F −1 (t) = inf{x : F(x) ≥ t} be the corresponding quantile function.

Proposition A.2 (Quantile Functions are Nondecreasing). *If* t ≤ u*, then* K(t) ≤ K(u)*.*

*Proof.* Since u ≥ t, it follows that {x : F(x) ≥ u} ⊆ {x : F(x) ≥ t}. Taking the infimum of both sides yields

$$\inf\{x : F(x) \geq u\} \geq \inf\{x : F(x) \geq t\} \Rightarrow K(u) \geq K(t). \quad (36)$$

We also will make use of the probability integral transformation, which we state here for convenience.

Proposition A.3 (Probability Integral Transformation [\(Shorack & Wellner,](#page-9-23) [2009,](#page-9-23) p. 5)). *If* X *has distribution function* F*, then*

$$\Pr(F(X) \leq t) \leq t \quad \text{for all } 0 \leq t \leq 1, \quad (37)$$

*with equality failing if and only if* t *is not in the closure of the range of* F*. Thus if* F *is continuous, then* T = F(X) *is* Uniform(0, 1)*.*

# B. Proof of Results from the Main Paper

# B.1. Proof of Proposition [3.1](#page-2-3)

Recall that Lscp(θ, λ) is the *miscoverage loss*:

$$\begin{aligned} L_{\text{scp}}(\theta, \lambda) &= \Pr\{s(z_{\text{new}}) > \lambda\} \\ &= 1 - \Pr\{s(z_{\text{new}}) \leq \lambda\} \\ &= 1 - \int \mathbb{1}\{s(z_{\text{new}}) \leq \lambda\} f(z_{\text{new}} \mid \theta) dz_{\text{new}}, \end{aligned} \tag{38}$$

Proposition 3.1. *Define* s<sup>i</sup> ≜ s(zi) *for* i = 1, . . . , n *and let* s(1) ≤ s(2) ≤ . . . ≤ s(n) *be the corresponding order statistics. Let* λscp *be the following decision rule:*

$$\lambda_{\text{scp}} = \begin{cases} s_{(\lceil (n+1)(1-\alpha) \rceil)}, & \text{if } \lceil (n+1)(1-\alpha) \rceil \leq n \\ \infty, & \text{otherwise.} \end{cases} \quad (12)$$

*Then* λscp *is an* α*-acceptable decision rule for the miscoverage loss* Lscp *defined in* [\(11\)](#page-2-2)*.*

*Proof.* By [Lei et al.](#page-9-24) [\(2018,](#page-9-24) Section 2),

$$\Pr(s_{\text{new}} \leq \hat{q}_{1-\alpha}) \geq 1 - \alpha, \quad (39)$$

where

$$\hat{q}_{1-\alpha} = \begin{cases} s_{\lceil ((n+1)(1-\alpha)) \rceil} & \text{if } \lceil ((n+1)(1-\alpha)) \rceil \leq n \\ \infty, & \text{otherwise.} \end{cases} \quad (40)$$

But Lscp(θ, λ) = 1 − Pr(snew ≤ λ | θ), so for λ = ˆq1−α, R(θ, λscp) ≤ α. This statement not depend on θ, and so R¯(λscp) ≤ α.

# B.2. Proof of Proposition [3.2](#page-2-8)

Recall that the Lcrc is defined as:

$$L_{\text{crc}}(\theta, \lambda) = \int \ell(z_{\text{new}}, \lambda) f(z_{\text{new}} \mid \theta) dz_{\text{new}}, \quad (41)$$

where ℓ(znew, λ) is an individual loss function that is monotonically non-increasing in λ.

Proposition 3.2. *Let* λcrc *be the following decision rule:*

$$\lambda_{\text{crc}} = \inf \left\{ \lambda : \frac{1}{n+1} \left( \sum_{i=1}^n \ell(z_i, \lambda) + B \right) \leq \alpha \right\}. \quad (15)$$

*Then* λcrc *is an* α*-acceptable decision rule for* Lcrc *defined in* [\(14\)](#page-2-4)*.*

*Proof.* Let L1, . . . , Ln, Ln+1 be an exchangeable collection of non-increasing random functions L<sup>i</sup> : Λ → (−∞, B]. By [Angelopoulos et al.](#page-9-4) [\(2024,](#page-9-4) Theorem 1),

$$\mathbb{E}[L_{n+1}(\hat{\lambda})] \leq \alpha, \quad (42)$$

where

$$\hat{\lambda} = \inf \left\{ \lambda : \frac{n}{n+1} \hat{R}_n(\lambda) + \frac{B}{n+1} \leq \alpha \right\} \quad (43)$$

and Rˆ <sup>n</sup>(λ) = (L1(λ) + . . . + Ln(λ))/n.

Interpreting these results using the notation from Section 3 of the main paper, we identify:

- Li(λ) = ℓ(z<sup>i</sup> , λ) for i = 1, . . . , n and Ln+1(λ) = ℓ(znew, λ),
- λcrc is identical to λˆ from [\(43\)](#page-12-0), and
- [\(42\)](#page-12-1) states that R(θ, λcrc) ≤ α for any θ.

#### B.3. Proof of Theorem [4.1](#page-3-4)

In order to prove Theorem [4.1,](#page-3-4) we will need to make use of two auxiliary propositions (Proposition [B.1](#page-13-0) and Proposition [B.2\)](#page-13-1). We state and prove these first, and then proceed to prove Theorem [4.1.](#page-3-4)

Proposition B.1. *Consider the following variational maximization problem:*

$$I[f] = \int_a^b f(x) \, dx \quad (44)$$

*subject to* f(a) = fa*,* f(b) = fb*, and* f<sup>a</sup> ≤ f(x) ≤ f<sup>b</sup> *for all* x ∈ [a, b]*, where* f<sup>a</sup> ≤ fb*. Then* I[f] *is maximized by*

$$f^*(x) = \begin{cases} f_a & \text{if } x = a, \\ f_b & \text{otherwise,} \end{cases} \quad (45)$$

*and* I[f ∗ ] = (b − a)fb*.*

*Proof.* We apply Euler's method [\(Kot,](#page-9-25) [2014,](#page-9-25) Section 2.2), which approximates the variational problem as an m-dimensional problem and takes the limit as m → ∞. Let the interval [a, b] be divided into m+1 subintervals of equal width ∆x = b − a m + 1 . The objective functional can then be approximated as

$$I(f_1, \dots, f_m) \equiv \sum_{j=0}^m f_j \Delta x, \quad (46)$$

where f<sup>0</sup> = f<sup>a</sup> and fm+1 = f<sup>b</sup> due to the boundary conditions. In order to handle the f<sup>a</sup> ≤ f(x) ≤ f<sup>b</sup> constraint, we first impose f(x) ≤ f<sup>b</sup> and check if the solution also satisfies f(x) ≥ fa. To that end, we substitute f<sup>j</sup> = f<sup>b</sup> − ξ j :

$$I(\xi_1, \dots, \xi_m) = \sum_{j=0}^m (f_b - \xi_j^2) \Delta x. \quad (47)$$

We then take partial derivatives with respect to ξk:

$$\frac{\partial I}{\partial \xi_k} = -2\xi_k \Delta x \Rightarrow \frac{1}{\Delta x} \frac{\partial I}{\partial \xi_k} = -2\xi_k. \quad (48)$$

Taking the limit as m → ∞ and ∆x → 0, the variational derivative becomes:

$$\frac{\delta I}{\delta \xi} = -2\xi. \quad (49)$$

Setting δI δξ = 0 yields <sup>ξ</sup>(x) = 0, which recovers <sup>f</sup>(x) = <sup>f</sup>b, except at <sup>x</sup> <sup>=</sup> <sup>a</sup>, where <sup>f</sup>(a) = <sup>f</sup><sup>a</sup> by the boundary conditions. This recovers f ∗ (x) from [\(45\)](#page-13-2), which indeed satisfies f(x) ≥ fa. For f ∗ , it is evident that the value of the functional is I[f ∗ ] = (b − a)fb.

Proposition B.2. *Let* K<sup>n</sup> *be the set of quantile functions for which* K(ti) = ℓ<sup>i</sup> *for* i = 1, . . . , n*. Then*

$$\sup_{K \in \mathcal{K}_n} J[K] = \sum_{i=1}^{n+1} (t_{(i)} - t_{(i-1)}) \ell_{(i)}, \quad (50)$$

*Proof.* By Proposition [A.2,](#page-11-2) quantile functions preserve orderings and therefore K(t(i)) = ℓ(i) . We divide J[K] into intervals with endpoints (0, t(1)),(t(1), t(2)), . . . ,(t(n) , 1):

$$\sup_{K \in \mathcal{K}_n} J[K] = \sup_{K \in \mathcal{K}_n} \int_0^1 K(t) dt \quad (51)$$

$$= \sup_{K \in \mathcal{K}_n} \sum_{i=1}^{n+1} \int_{t_{(i-1)}}^{t_{(i)}} K(t) dt \quad (52)$$

$$\leq \sum_{i=1}^{n+1} \sup_{K \in \mathcal{K}_n} \int_{t_{(i-1)}}^{t_{(i)}} K(t) dt \quad (53)$$

By Proposition [A.2,](#page-11-2) K(t(i−1)) ≤ K(t) ≤ K(t(i)) for any t ∈ [t(i−1), t(i) ]. We view each term as a variational subproblem where J<sup>i</sup> [K<sup>i</sup> ] ≜ Z t(i) t(i−1) Ki(t) dt with boundary conditions Ki(t(i−1)) = ℓ(i−1) and Ki(t(i)) = ℓ(i) . We therefore appeal to Proposition [B.1](#page-13-0) to conclude that

$$K_i^*(t) = \begin{cases} \ell_{(i-1)} & \text{if } t = t_{(i-1)}, \\ \ell_{(i)} & \text{otherwise,} \end{cases} \quad (54)$$

and J[K<sup>∗</sup> i ] = (t(i) − t(i−1))ℓ(i) . We therefore have

$$\sup_{K \in \mathcal{K}_n} J[K] \leq \sum_{i=1}^{n+1} (t_{(i)} - t_{(i-1)}) \ell_{(i)}. \quad (55)$$

By composing K<sup>∗</sup> i from each subinterval, it is straightforward to see that the bound is tight for

$$K_{t_{1:n}, \ell_{1:n}}^*(t) = \begin{cases} \ell_{(1)} & \text{if } t \leq t_{(1)} \\ \ell_{(2)} & \text{if } t_{(1)} < t \leq t_{(2)} \\ \dots & \\ \ell_{(n)} & \text{if } t_{(n-1)} < t \leq t_{(n)} \\ B & \text{if } t > t_{(n)}. \end{cases} \quad (56)$$

K<sup>∗</sup> t1:n,ℓ1:<sup>n</sup> is therefore the "worst-case" quantile function that is consistent with the observations, and J[K<sup>∗</sup> t1:n,ℓ1:<sup>n</sup> ] = P<sup>n</sup>+1 <sup>i</sup>=1 (t(i) − t(i−1))ℓ(i) .

We are now ready to prove Theorem [4.1.](#page-3-4)

Theorem 4.1. *Let* t(0) = 0*,* t(n+1) = 1*, and* ℓ(n+1) = B*. Then*

$$\sup_{\pi} E(L \mid t_{1:n}, \ell_{1:n}) \leq \sum_{i=1}^{n+1} u_i \ell_{(i)}, \quad (26)$$

*where* u<sup>i</sup> = t(i) − t(i−1)*.*

*Proof.* Let <sup>J</sup>[K] = R <sup>1</sup> <sup>0</sup> K(t) dt. The conditional expected loss can be expressed as:

$$E(L \mid t_{1:n}, \ell_{1:n}) = \int J[K]p(K \mid t_{1:n}, \ell_{1:n}) dK \quad (57)$$

$$\leq \sup_{K \in \mathcal{K}_n} J[K], \quad (58)$$

where K<sup>n</sup> is the set of quantile functions for which K(ti) = ℓ<sup>i</sup> for i = 1, . . . , n. By Proposition [B.2,](#page-13-1) it follows that

$$E(L \mid t_{1:n}, \ell_{1:n}) \leq \sum_{i=1}^{n+1} (t_{(i)} - t_{(i-1)}) \ell_{(i)} = \sum_{i=1}^{n+1} u_i \ell_{(i)} \quad (59)$$

#### B.4. Proof of Lemma [4.2](#page-4-5)

Lemma 4.2 (Distribution of Quantile Spacings [\(Aitchison & Dunsmore,](#page-9-0) [1975,](#page-9-0) p. 140)). *Suppose that* ℓ1, . . . , ℓ<sup>n</sup> *are drawn i.i.d. with continuous*[<sup>5</sup>](#page-15-0) *distribution function* F*. Let* t<sup>i</sup> = F(ℓi) *and* u<sup>i</sup> = t(i) − t(i−1)*, where by convention* t(0) = 0 *and* <sup>t</sup>(n+1) = 1*. Then* (u1, u2, . . . , un+1) <sup>∼</sup><sup>=</sup> Dir(1, . . . , 1)*.*

*Proof.* By the probability integral transformation (Proposition [A.3\)](#page-11-3), T<sup>i</sup> is Uniform(0, 1) for i = 1, . . . , n. Since the transformation from (t1, . . . , tn) → (t(1), . . . , t(n)) is a sorting operation where n! permutations map to the same vector of order statistics, the probability density for t(1), . . . , t(n) is therefore

$$f_{(t_{(1)}, \dots, t_{(n)})}(t_1, \dots, t_n) = n!, \quad 0 \leq t_1 \leq t_2 \leq \dots \leq t_n \leq 1. \quad (60)$$

If u1:<sup>n</sup> = G(t(1:n)) where G is differentiable and invertible, then by change of variables the density for u1:<sup>n</sup> can be expressed as

$$f_{u_{1:n}}(u_{1:n}) = f_{t_{(1:n)}}(G^{-1}(u_{1:n})) \left| \det \left( \frac{\partial}{\partial u_{1:n}} G^{-1}(u_{1:n}) \right) \right|. \quad (61)$$

Observe that the inverse transformation t(1:n) = G−<sup>1</sup> (u1:n) can be expressed as

$$\begin{bmatrix} t_{(1)} \\ t_{(2)} \\ t_{(3)} \\ \vdots \\ t_{(n-1)} \\ t_{(n)} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & \dots & 0 & 0 \\ 1 & 1 & 0 & \dots & 0 & 0 \\ 1 & 1 & 1 & \dots & 0 & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 1 & 1 & 1 & \dots & 1 & 0 \\ 1 & 1 & 1 & \dots & 1 & 0 \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ \vdots \\ u_{n-1} \\ u_n \end{bmatrix}. \quad (62)$$

Hence the absolute Jacobian of inverse transformation t(1:n) = G−<sup>1</sup> (u1:n) is 1. The density of u1:<sup>n</sup> is therefore

$$f_{u_{1:n}}(u_{1:n}) = f_{t_{(1:n)}}(G^{-1}(u_{1:n})) = n!, \quad \text{where } u_i \geq 0 \text{ for } i = 1, \dots, n \text{ and } \sum_{i=1}^n u_i \leq 1. \quad (63)$$

Recall that the Dirichlet density with parameter α1, . . . , αn+1 is:

$$\text{Dir}(u_{1:n+1} \mid \alpha_{1:n+1}) = \frac{\Gamma(\sum_{i=1}^{n+1} \alpha_i)}{\Gamma(\alpha_1) \dots \Gamma(\alpha_{n+1})} \prod_{i=1}^{n+1} u_i^{\alpha_i-1}, \quad \text{where } u_i \geq 0 \text{ and } \sum_{i=1}^{n+1} u_i = 1. \quad (64)$$

In particular, if α<sup>1</sup> = α<sup>2</sup> = . . . = αn+1 = 1,

$$\text{Dir}(u_{1:n+1} \mid 1, \dots, 1) = \Gamma(n+1) = n!, \quad (65)$$

which is identical to [\(63\)](#page-15-1) with <sup>u</sup>n+1 = 1 − <sup>u</sup><sup>1</sup> − . . . − <sup>u</sup>n. Therefore, (u1, u2, . . . , un+1) ∼<sup>=</sup> Dir(1, . . . , 1).

#### B.5. Proof of Theorem [4.3](#page-4-6)

Theorem 4.3. *Define* ℓ(i) *to be the order statistics of* ℓ1, . . . , ℓ<sup>n</sup> *for* i = 1, . . . , n *and* ℓ(n+1) ≜ B*. Let* L <sup>+</sup> *be the random variable defined as follows:*

$$U_1, \dots, U_{n+1} \sim \text{Dir}(1, \dots, 1), \quad L^+ = \sum_{i=1}^{n+1} U_i \ell_{(i)}. \quad (27)$$

*Then for any* b ∈ (−∞, B]*,*

$$\inf_{\pi} \Pr(L \leq b \mid \ell_{1:n}) \geq \Pr(L^+ \leq b). \quad (28)$$

<sup>5</sup>The correspondence to a Dirichlet distribution holds exactly for continuous distributions. Weighted sums of Dirichlet random variates stochastically dominate weighted sums of discrete quantile spacings, and thus due to space constraints we only consider continuous distributions here.

*Proof.*

$$\inf_{\pi} \Pr(L \leq b \mid \ell_{1:n}) = \inf_{\pi} \int \mathbb{1}\{J[K] \leq b\} p(K \mid \ell_{1:n}) dK \quad (66)$$

$$= \inf_{\pi} \int \mathbb{1}\{J[K] \leq b\} \left( \int p(K \mid t_{1:n}, \ell_{1:n}) p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \right) dK \quad (67)$$

$$= \inf_{\pi} \int \left( \int \mathbb{1}\{J[K] \leq b\} p(K \mid t_{1:n}, \ell_{1:n}) dK \right) p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \quad (68)$$

$$\geq \int \left( \inf_{\pi} \int \mathbb{1}\{J[K] \leq b\} p(K \mid t_{1:n}, \ell_{1:n}) dK \right) p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \quad (69)$$

$$\geq \int \left( \inf_{K \in \mathcal{K}_n} \mathbb{1}\{J[K] \leq b\} \right) p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \quad (70)$$

$$\geq \int \mathbb{1}\{J[K_{t_{1:n}, \ell_{1:n}}^*] \leq b\} p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \quad (71)$$

$$= \int \mathbb{1} \left\{ \sum_{i=1}^{n+1} (t_{(i)} - t_{(i-1)}) \ell_{(i)} \leq b \right\} p(t_{1:n} \mid \ell_{1:n}) dt_{1:n} \quad (72)$$

$$= \int \mathbb{1} \left\{ \sum_{i=1}^{n+1} u_i \ell_{(i)} \leq b \right\} p(u_{1:n+1} \mid \ell_{1:n}) du_{1:n+1} \quad (73)$$

$$= \Pr(L^+ \leq b) \quad (74)$$

Note that here the quantile distribution p(t1:<sup>n</sup> | ℓ1:n) takes into account the ordering of the losses but not the values themselves. This is consistent with the implicit assumption, made in the frequentist treatment of conformal prediction, that a new observation is equally likely to fall into each of the intervals formed by the loss values and the endpoints. The results for characterizing this distribution can be traced back to [Tukey](#page-10-2) [\(1947\)](#page-10-2), by way of [Guttman](#page-9-10) [\(1970\)](#page-9-10) and [Aitchison & Dunsmore](#page-9-0) [\(1975\)](#page-9-0), and is also known as A<sup>n</sup> in the literature [\(Hill,](#page-9-26) [1968;](#page-9-26) [1988\)](#page-9-27). These results allow us to make minimal assumptions about the form of the CDF.

#### B.6. Proof of Corollary [4.4](#page-4-10)

Corollary 4.4. *For any desired confidence level* β ∈ (0, 1)*, define*

$$b_\beta^* = \inf_b \{b : \Pr(L^+ \leq b \mid \ell_{1:n}) \geq \beta\}. \quad (29)$$

*Then* inf<sup>π</sup> Pr(L ≤ b | ℓ1:n) ≥ β *for any* b ≥ b ∗ β

*Proof.* For any b ≥ b ∗ β , Pr(L <sup>+</sup> ≤ b | ℓ1:n) ≥ β. Substitution into [\(28\)](#page-4-11) provides the desired result.