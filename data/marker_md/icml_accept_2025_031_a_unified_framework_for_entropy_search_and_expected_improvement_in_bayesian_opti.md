# A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization

Nuojin Cheng\* <sup>1</sup> Leonard Papenmeier\* <sup>2</sup> Stephen Becker <sup>1</sup> Luigi Nardi 2 3

# Abstract

Bayesian optimization is a widely used method for optimizing expensive black-box functions, with Expected Improvement being one of the most commonly used acquisition functions. In contrast, information-theoretic acquisition functions aim to reduce uncertainty about the function's optimum and are often considered fundamentally distinct from EI. In this work, we challenge this prevailing perspective by introducing a unified theoretical framework, Variational Entropy Search, which reveals that EI and information-theoretic acquisition functions are more closely related than previously recognized. We demonstrate that EI can be interpreted as a variational inference approximation of the popular information-theoretic acquisition function, named Max-value Entropy Search. Building on this insight, we propose VES-Gamma, a novel acquisition function that balances the strengths of EI and MES. Extensive empirical evaluations across both low- and high-dimensional synthetic and real-world benchmarks demonstrate that VES-Gamma is competitive with state-ofthe-art acquisition functions and in many cases outperforms EI and MES.

# 1. Introduction

Bayesian optimization (BO) is a widely used technique for maximizing black-box functions. Given a function f : X → R, BO iteratively refines a probabilistic surrogate of f, typically a Gaussian process (GP), and selects the next evaluation point accordingly. At each iteration, the next sampling point is determined by maximizing an acquisition function (AF) α : X → R. An effective AF must balance the exploration-exploitation trade-off, where exploitation prioritizes sampling points predicted by the surrogate to yield high objective values, while exploration targets regions with the potential to uncover even better values.

Expected Improvement (EI) [\(Mockus, 1998\)](#page-9-0) is one of the most widely used AFs, valued for its simple formulation, computational efficiency, and strong empirical performance. The core idea behind EI is to maximize the expected improvement over the current best observed value, which typically requires a noise-free assumption. More recently, [\(Villemonteix et al., 2009;](#page-10-0) [Hennig & Schuler, 2012\)](#page-8-0) have introduced the concepts of information-theoretic AFs, which represents a paradigm shift in Bayesian optimization. Unlike EI, which focuses on directly maximizing potential improvement, information-theoretic AFs aim to reduce uncertainty about the function f's optimal position and/or value, often through entropy-based measures. Due to their fundamentally different underlying philosophies and selection criteria, EI and information-theoretic AFs are widely regarded as distinct methodologies within the BO community [\(Hennig et al., 2022\)](#page-8-1).

Despite their apparent differences, we argue that EI and information-theoretic AFs share deeper theoretical connections than previously recognized. Understanding this relationship is crucial, as it provides novel insights into designing new acquisition functions. By unifying the perspectives of both sides, we introduce VES-Gamma, a new AF that effectively balances their strengths, resulting in a robust AF that adapts well to diverse optimization problems. VES-Gamma inherits the performance of EI while incorporating information-theoretic considerations.

In summary, we make the following key contributions:

- 1. We introduce the Variational Entropy Search (VES) framework which shows that EI can be interpreted as a special case of the popular information-theoretic acquisition function Max-value Entropy Search (MES). This unified theoretical perspective reveals that these two types of AFs are more closely related than previously recognized.

<sup>\*</sup>Equal contribution <sup>1</sup>Department of Applied Mathematics, University of Colorado Boulder <sup>2</sup>Department of Computer Science, Lund University <sup>3</sup>DBtune. Correspondence to: Nuojin Cheng <Nuojin.Cheng@colorado.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

- 2. We propose VES-Gamma as an intermediary between EI and MES, incorporating information-theoretic principles while maintaining EI's strength in performance.
- 3. We provide an extensive evaluation across a diverse set of low- and high-dimensional synthetic, GP samples, and real-world benchmarks, demonstrating that VES-Gamma consistently performs competitively and, in many cases, outperforms both EI and MES.

# 2. Background and Related Work

#### 2.1. Gaussian Processes

A Gaussian process is a stochastic process that models an unknown function. It is characterized by the property that any finite set of function evaluations follows a multivariate Gaussian distribution. Assuming that f has a zero mean, a Gaussian process is uniquely determined by the current observations D<sup>t</sup> := {(x<sup>i</sup> , y<sup>x</sup><sup>i</sup> )} t <sup>i</sup>=1 and the kernel function κ(x, x ′ ). Given these, at stage t, the predicted mean of y<sup>x</sup> at a new point x is µt(x) = κt(x) T (Kt) <sup>−</sup><sup>1</sup>yt, and the predicted covariance between points x and x ′ is Covt(x, x ′ ) = κ(x, x ′ ) − κt(x) T (Kt) <sup>−</sup><sup>1</sup>κt(x ′ ), where [κt(x)]<sup>i</sup> = κ(x<sup>i</sup> , x), [yt]<sup>i</sup> = y<sup>x</sup><sup>i</sup> , and [Kt]i,j = κ(x<sup>i</sup> , x<sup>j</sup> ); see [Rasmussen et al.](#page-9-1) [\(2006\)](#page-9-1) for more details.

#### 2.2. Acquisition Functions

Various acquisition functions (AFs) have been proposed to balance exploration and exploitation in optimization tasks, each tailored to different problem characteristics and assumptions. These include Probability of Improvement (PI), Expected Improvement (EI) [\(Mockus, 1998;](#page-9-0) [Jones et al.,](#page-9-2) [1998\)](#page-9-2), Upper Confidence Bound (UCB) [\(Srinivas et al.,](#page-9-3) [2010\)](#page-9-3), Knowledge Gradient (KG) [\(Frazier et al., 2008\)](#page-8-2), and information-theoretic AFs [\(Villemonteix et al., 2009;](#page-10-0) [Hen](#page-8-0)[nig & Schuler, 2012;](#page-8-0) [Hernandez-Lobato et al., 2014;](#page-8-3) [Wang](#page-10-1) ´ [& Jegelka, 2017;](#page-10-1) [Hvarfner et al., 2022;](#page-9-4) [Tu et al., 2022\)](#page-10-2). Below, we discuss two types of acquisition functions relevant to this study.

Expected Improvement. Expected Improvement (EI) is one of the most commonly used acquisition functions and is formulated as follows:

$$\alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}_{p(y_{\mathbf{x}}|\mathcal{D}_t)} [\max\{y_{\mathbf{x}}, y_t^*\}] - y_t^*, \quad (1)$$

where y ∗ t is the maximum observed value in Dt, and <sup>E</sup>p(·) denotes the expectation with respect to the predictive density p(·). The −y ∗ t term at the end can be dropped since it is constant with respect to x.

that its evaluation reduces uncertainty regarding the optimal points of the objective function. This uncertainty is quantified using differential entropy, defined as <sup>H</sup>[y] := <sup>E</sup>p(y) [− log p(y)]. Similarly, the conditional entropy is expressed as <sup>H</sup>[y|x] := <sup>H</sup>[x, y] − <sup>H</sup>[x].

The first information-theoretic AF for BO is Entropy Search (ES) [\(Hennig & Schuler, 2012\)](#page-8-0), which is formulated as:

$$\alpha_{\text{ES}}(\mathbf{x}) = \mathbb{H}[\mathbf{x}^* \mid \mathcal{D}_t] - \mathbb{E}_{p(y_{\mathbf{x}} \mid \mathcal{D}_t)} [\mathbb{H}[\mathbf{x}^* \mid \mathcal{D}_t, y_{\mathbf{x}}]]. \quad (2)$$

Here, the random variable x ∗ represents the location of the maximum.

Predictive Entropy Search (PES) [\(Hernandez-Lobato et al.,](#page-8-3) ´ [2014\)](#page-8-3) offers a reformulation of ES that is computationally more efficient:

$$\alpha_{\text{PES}}(\mathbf{x}) = \mathbb{H}[y_{\mathbf{x}} \mid \mathcal{D}_t] - \mathbb{E}_{p(\mathbf{x}^* \mid \mathcal{D}_t)} [\mathbb{H}[y_{\mathbf{x}} \mid \mathcal{D}_t, \mathbf{x}^*]]. \quad (3)$$

Since directly estimating the entropy with x ∗ is expensive, following the PES format, Max-value Entropy Search (MES) [\(Wang & Jegelka, 2017\)](#page-10-1) introduced an alternative approach that focuses on reducing the differential entropy of the 1D maximum value y ∗ :

$$\begin{aligned} \alpha_{\text{MES}}(\mathbf{x}) &= \underbrace{\mathbb{H}[[y^* \mid \mathcal{D}_t] - \mathbb{E}_{p(y_{\mathbf{x}}|\mathcal{D}_t)} [\mathbb{H}[[y^* \mid \mathcal{D}_t, y_{\mathbf{x}}]]]}_{\text{close-form}} \\ &= \underbrace{\mathbb{H}[[y_{\mathbf{x}} \mid \mathcal{D}_t] - \mathbb{E}_{p(y^*|\mathcal{D}_t)} [\mathbb{H}[[y_{\mathbf{x}} \mid \mathcal{D}_t, y^*]]]}_{\text{non-closed-form}}. \end{aligned} \quad (4)$$

Unlike MES and its subsequent extensions [\(Hvarfner et al.,](#page-9-4) [2022;](#page-9-4) [Takeno et al., 2022\)](#page-10-3) which approximate p(y<sup>x</sup> | Dt, y<sup>∗</sup> ) using a truncated Gaussian, we focus on directly estimating p(y ∗ | Dt, yx) via variational inference.

#### 2.3. Related Work

Variational Inference and Evidence Lower Bound. Variational Inference (VI) is a widely used technique in Bayesian modeling to approximate intractable posterior distributions [\(Paisley et al., 2012;](#page-9-5) [Hoffman et al., 2013;](#page-9-6) [Kingma & Welling, 2014\)](#page-9-7). It relies on maximizing the Evidence Lower Bound (ELBO) to approximate the loglikelihood log p(x˜) in the presence of latent variables z. The log-likelihood can be decomposed as follows:

$$\log p(\tilde{\mathbf{x}}) \geq \mathbb{E}_{q(\mathbf{z})} \left[ \log \left( \frac{p(\tilde{\mathbf{x}} \mid \mathbf{z})p(\mathbf{z})}{q(\mathbf{z})} \right) \right], \quad (5)$$

where p(z) is a fixed prior distribution, and q(z) is a variational approximation to the true posterior p(z | x˜). The ELBO is formally defined as:

$$\text{ELBO}(p(\tilde{x} \mid \mathbf{z}); q(\mathbf{z})) := \mathbb{E}_{q(\mathbf{z})} \left[ \log \left( \frac{p(\tilde{x} \mid \mathbf{z})p(\mathbf{z})}{q(\mathbf{z})} \right) \right]. \quad (6)$$

![](_page_2_Figure_1.jpeg)

Figure 1. MES aims to optimize x such that the entropy (averaged over all yx) of the maximum values p(y ∗ | Dt, yx) is reduced. The left figure illustrates a noiseless Gaussian process conditioned on the observations D<sup>t</sup> with three points (black crosses) and a sample y<sup>x</sup> at x = 1 drawn from p(y<sup>x</sup> | Dt) (red star). The mid and right panels illustrate the density p(y ∗ | Dt, yx) (blue curves). When p(y ∗ | Dt, yx) is approximated using an exponential distribution (green dashed curve), this leads to the VES-Exp AF that is equivalent to EI. Furthermore, VES-Gamma, which approximates p(y ∗ | Dt, yx) using a Gamma distribution (red dash-dot curve), leads to a more accurate approximation and a generalized version of EI.

By maximizing the ELBO, VI indirectly maximizes the log-likelihood log p(x˜), thereby improving the quality of the posterior approximation. In many applications, such as variational autoencoders (VAEs) [\(Kingma & Welling,](#page-9-7) [2014\)](#page-9-7) and variational diffusion [\(Kingma et al., 2021\)](#page-9-8), both the conditional likelihood p(x˜ | z) and the variational distribution q(z) are parameterized using neural networks. Since both the expectation reference probability and the term inside the ELBO are parameterized, one common strategy is to estimate the gradient using finite Monte Carlo samples and the *reparameterization trick* to optimize the parameters. We adopt this approach, which enables efficient gradient-based optimization and has been widely applied in the BO community [\(Wilson et al., 2017\)](#page-10-4).

Improving the Expected Improvement. It is widely recognized that EI can be prone to over-exploitation [\(Qin](#page-9-9) [et al., 2017;](#page-9-9) [Berk et al., 2019;](#page-8-4) [De Ath et al., 2021\)](#page-8-5). To mitigate this issue, [Hoffman et al.](#page-9-10) [\(2011\)](#page-9-10) and [Kan](#page-9-11)[dasamy et al.](#page-9-11) [\(2020\)](#page-9-11) propose to use a portfolio of AFs, which assigns probabilities to different AFs at each step. [Snoek et al.](#page-9-12) [\(2012\)](#page-9-12) proposed a fully-Bayesian treatment on EI to improve empirical performance. Another approach is Weighted EI (WEI), which adaptively adjusts the weights of the components within the EI acquisition function [\(Sobester et al., 2005;](#page-9-13) [Benjamins et al., 2023\)](#page-8-6). Simi- ´ larly, [Qin et al.](#page-9-9) [\(2017\)](#page-9-9) suggest "weakening" EI using suboptimal points suggested by the AF to mitigate its overexploitative behavior. However, these methods are primarily based on heuristics. Furthermore, information-theoretic acquisition functions are often excluded from these design enhancements, as they are generally considered distinct from heuristic AFs such as PI, EI, UCB, or KG.

Entropy Approximation in Information-theoretic AFs. Estimating entropy in information-theoretic acquisition functions is computationally expensive and typically requires approximation techniques. Methods such as ES and PES employ sampling-based approaches, including Markov chain Monte Carlo and expectation propagation. In contrast, MES derives an explicit approximation [\(Wang &](#page-10-1) [Jegelka, 2017,](#page-10-1) Eq. 6), which was later interpreted as a variational inference formulation by [Takeno et al.](#page-9-14) [\(2020\)](#page-9-14). This variational perspective has since been extended to multiobjective optimization [\(Qing et al., 2023\)](#page-9-15). However, this approximation scheme lacks flexibility in tuning the variational distributions. Furthermore, to the best of our knowledge, most MES-based methods focus on approximating p(y<sup>x</sup> | y ∗ , Dt). An exception is [Ma et al.](#page-9-16) [\(2023\)](#page-9-16), which approximates p(y ∗ | Dt, yx) using a Gaussian distribution. While this approach provides computational advantages, the inherent symmetry of the Gaussian distribution does not align with the properties of y ∗ .

# 3. Variational Entropy Search

#### 3.1. Entropy Search Lower Bound

The idea behind our Variational Entropy Search (VES) framework is to maximize a variational lower bound of MES with a predetermined family of densities to approximate p(y ∗ | Dt, yx). Since we assume noiseless observations, the support is [max{yx, y<sup>∗</sup> <sup>t</sup> }, +∞). VES is illustrated in Figure [1.](#page-2-0) The lower bound is formalized in Theorem [3.1](#page-2-1) and proven in Appendix [A.1.](#page-11-0)

Theorem 3.1. *The MES acquisition function in Eq.* [\(4\)](#page-1-0) *adheres to the Barber-Agakov (BA) bound [\(Barber & Agakov,](#page-8-7) [2004;](#page-8-7) [Poole et al., 2019\)](#page-9-17) and can be bounded from below as follows:*

$$\begin{aligned} \alpha_{MES}(\mathbf{x}) &= \mathbb{H}[y^* \mid \mathcal{D}_t] - \mathbb{E}_{p(y_{\mathbf{x}}|\mathcal{D}_t)} [\mathbb{H}[y^* \mid \mathcal{D}_t, y_{\mathbf{x}}]] \\ &\geq \mathbb{H}[y^* \mid \mathcal{D}_t] + \mathbb{E}_{p(y^*, y_{\mathbf{x}}|\mathcal{D}_t)} [\log q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}})], \end{aligned} \tag{7}$$

*where* q(y ∗ | Dt, yx) *is any chosen density function that is absolutely continuous with respect to* p(y ∗ | Dt, yx)*.*

Since the first term on the right-hand side of Eq. [\(7\)](#page-2-2), <sup>H</sup>[y ∗ Dt], is independent of both q and x, we can omit it. This leads us to define the remaining term as the Entropy Search Lower Bound (ESLBO):

$$\text{ESLBO}(\mathbf{x}; q) := \mathbb{E}_{p(y^*, y_{\mathbf{x}} | \mathcal{D}_t)} [\log q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}})], \quad (26)$$

(8) where p(y ∗ , y<sup>x</sup> | Dt) represents a joint density, which can be sampled using Gaussian process path sampling [\(Hernandez-Lobato et al., 2014;](#page-8-3) [Wang & Jegelka,](#page-10-1) ´ [2017\)](#page-10-1).

To optimize αMES(x), we adopt the VI approach [\(Paisley](#page-9-5) [et al., 2012\)](#page-9-5), indirectly maximizing αMES(x) by instead maximizing ESLBO. To ensure computational feasibility, the VI method constrains the density q to a predefined family Q. When parameterizing q within Q, the problem becomes tractable by solving for q and x iteratively, as detailed in Algorithm [1.](#page-3-0)

Notably, this procedure, known as expectation maximization (EM), is analogous to maximizing the ELBO in Eq. [\(5\)](#page-1-1). We conclude our discussion by summarizing the correspondence between ESLBO and ELBO in Table [1.](#page-4-0)

Algorithm 1 VES Framework

Input: Observations Dt, variational family Q, number of inner iteration N

Output: Next sampling location xt+1

- 1: initialize x
- (0) t+1 2: for n = 1 : N do 3: q
- (n) (y ∗ ) ← arg maxq∈Q ESLBO(x (n−1) <sup>t</sup>+1 ; q) 4: x
  - (n) <sup>t</sup>+1 ← arg max<sup>x</sup>t+1 ESLBO(xt+1; q
- (n) ) 5: end for 6: return x
  - (N) t+1

#### 3.2. EI Through the Lens of the VES Framework

In this section, we aim to establish an explicit connection between the VES and EI acquisition functions, allowing us to see EI through the lens of a VI approximation of the information-theoretical MES AF. We define Q as the set of all exponential density functions, Qexp, parameterized by the λ > 0 exponential density parameter and with support bounded from below by max{yx, y<sup>∗</sup> <sup>t</sup> }. The variational density function q is given by

$$q(y^* | \mathcal{D}_t, y_{\mathbf{x}}; \lambda) = \lambda e^{-\lambda(y^* - \max\{y_{\mathbf{x}}, y_t^*\})} \mathbf{1}_{y^* \geq \max\{y_{\mathbf{x}}, y_t^*\}}. \quad (9)$$

For noiseless observations, the indicator function 1y∗≥max{yx,y<sup>∗</sup> <sup>t</sup> } always equals one and can be omitted. Plugging in q from Eq. [\(9\)](#page-3-1) into the ESLBO (Eq. [\(8\)](#page-3-2)) yields a new λ-parameterized AF. Since this AF stems from the exponential distribution, we name it VES-Exp. Theorem [3.2](#page-3-3) shows that the next sampling point generated from VES-Exp within Algorithm [1](#page-3-0) will be the same as for the EI AF; the theorem is proven in Appendix [A.2.](#page-11-1)

Theorem 3.2. *When the family* Qexp *is selected as in Eq.* [\(9\)](#page-3-1) *and the function is noiseless, ESLBO in Eq.* [\(8\)](#page-3-2) *turns into*

$$\begin{aligned} ESLBO(\mathbf{x}; \lambda) &= \log \lambda - \lambda \underbrace{\mathbb{E}_{P(y^* | \mathcal{D}_t)} [y^*]}_{\text{constant}} \\ &+ \lambda \underbrace{\mathbb{E}_{P(y_{\mathbf{x}} | \mathcal{D}_t)} [\max\{y_{\mathbf{x}}, y_t^*\}]}_{\text{EI}}. \end{aligned} \quad (10)$$

*Maximizing ESLBO*(x; λ) *in Eq.* [\(10\)](#page-3-4) *with respect to* x *and* λ *yields the same* x *solution as the maximization of EI in Eq.* [\(1\)](#page-1-2)*.*

The key idea behind the proof is that, following Algorithm [1,](#page-3-0) the ESLBO in Eq. [\(10\)](#page-3-4) always converges within two iterations. Regardless of the positive value of λ, the value of x that maximizes ESLBO(x; λ) remains the same. Consequently, starting from an arbitrary initial point x (0) , a positive λ (1) is derived, ensuring that ESLBO reaches its maximum value in the next iteration.

Theorem [3.2](#page-3-3) reveals that EI can be viewed as a special case of MES, giving a new information-theoretic interpretation of the most popular acquisition function in use today. However, the exponential distribution has a fairly rigid parametric form that does not capture the characteristics of p(y ∗ | Dt, yx). Figure [1](#page-2-0) (right) shows an example of the structural limitations of the exponential density in green. We generate 1000 samples from an example distribution p(y ∗ | Dt, yx), and observe that it significantly deviates from an exponential distribution. Specifically, the density of p(y ∗ | Dt, yx) is non-monotonic, exhibiting a peak before decreasing near max{yx, y<sup>∗</sup> <sup>t</sup> } (approximately 1.55), while exponential distributions are necessarily monotonic.

This observation motivates the need to enrich the variational distributions Q to allow more flexibility. A natural extension is to use a Gamma distribution, which is a generalization of the exponential distribution. The Gamma density approximation in the previous example is shown in red in Figure [1](#page-2-0) (right). The next section introduces VES-Gamma, which is a more general AF that extends VES-Exp and its equivalent EI acquisition function.

#### 3.3. VES-Gamma: A Generalization of EI

VES-Gamma defines Q as the Gamma distribution parameterized by k, β > 0 with its support bounded from below by max{yx, y<sup>∗</sup> <sup>t</sup> }. The variational density is

$$q(y^* \mid \mathcal{D}_t, y_x; k, \beta) = \frac{\beta^k}{\Gamma(k)} (y^* - \max\{y_x, y_t^*\})^{k-1} \times e^{-\beta(y^* - \max\{y_x, y_t^*\})} \mathbf{1}_{y^* \geq \max\{y_x, y_t^*\}},$$

(11) where Γ(·) denotes the Gamma function. The noise-free assumption allows us to omit the indicator function, and

Table 1. Comparison of key aspects between the ELBO and ESLBO approaches.

| Property                | ELBO Approach                  | ESLBO Approach  |
|-------------------------|--------------------------------|-----------------|
| Primary Variable        | p ( x ˜   z )                  | x               |
| Variational Variable    | q ( z )                        | q ( y           |
|                         |                                | y x , D t )     |
| Lower Bound Formulation | ELBO ( q ( z ); p ( x ˜   z )) | ESLBO ( q ; x ) |

the ESLBO is reformulated as

$$\begin{aligned} \text{ESLBO}(\mathbf{x}; k, \beta) &= k \log \beta - \log \Gamma(k) \\ &+ (k-1)\mathbb{E}_{p(y^*, y_{\mathbf{x}} | \mathcal{D}_t)} [\log(y^* - \max\{y_{\mathbf{x}}, y_t^*\})] \\ &- \beta \mathbb{E}_{p(y^* | \mathcal{D}_t)} [y^*] + \beta \underbrace{\mathbb{E}_{p(y_{\mathbf{x}} | \mathcal{D}_t)} [\max\{y_{\mathbf{x}}, y_t^*\}]}_{\text{EI}}. \end{aligned} \quad (12)$$

The ESLBO in Eq. [\(12\)](#page-4-1) serves as the primary objective in the VES-Gamma algorithm. Eq. [\(12\)](#page-4-1) consists of five terms, with the last term being the EI AF in Eq. [\(1\)](#page-1-2) scaled by a multiplicative factor. The two hyperparameters, k and β, originally part of the Gamma distribution, dynamically balance different components of the objective. In particular, when k = 1, the Gamma distribution reduces to an exponential distribution, making the ESLBO in Eq. [\(12\)](#page-4-1) equivalent to Eq. [\(10\)](#page-3-4). In the following section, we discuss the approach for determining values for k and β.

Auto-determination of Tradeoff Hyperparameters. For any fixed x, the global maximum of the ESLBO in Eq. [\(12\)](#page-4-1) with respect to k and β uniquely exists, as can be demonstrated through derivative analysis. Taking the partial derivatives of ESLBO in Eq. [\(12\)](#page-4-1) and setting them to zero, we obtain:

$$\log \beta - \frac{\partial \log \Gamma(k)}{\partial k} + \mathbb{E} [\log z_{\mathbf{x}}^*] = 0, \quad \frac{k}{\beta} - \mathbb{E} [z_{\mathbf{x}}^*] = 0,$$

where the random variable z ∗ x := y <sup>∗</sup> − max{yx, y<sup>∗</sup> <sup>t</sup> }.

Substituting the second equation into the first yields:

$$\log k - \psi(k) = \log \mathbb{E}[z_{\mathbf{x}}^*] - \mathbb{E}[\log z_{\mathbf{x}}^*], \quad (13)$$

where ψ(k) := ∂ log Γ(k)/∂k is the digamma function [\(Abramowitz et al., 1988\)](#page-8-8), which can be efficiently approximated as a series. By Jensen's inequality, log <sup>E</sup>[z ∗ <sup>x</sup>] − <sup>E</sup>[log z ∗ <sup>x</sup>] ≥ 0. Since log k − ψ(k) is strictly decreasing and approaches zero asymptotically (see Figure [2\)](#page-4-2), the root of Eq. [\(13\)](#page-4-3), k ∗ x, exists uniquely—except in the degenerate case where z ∗ <sup>x</sup> is deterministic. In the practical implementation, we apply a clamping function to ensure that the term log k − ψ(k) does not become zero, and we employ a regularization mechanism to keep the resulting root k ∗ <sup>x</sup> close to 1. Specifically, we use L2 regularization when solving log k − ψ(k) = log <sup>E</sup>[z ∗ <sup>x</sup>] − <sup>E</sup>[log z ∗ x] since the unregularized version is unstable, presumably due to a widely flat landscape. In particular, for ξ(k) :=

![](_page_4_Figure_3.jpeg)

Figure 2. Plot of log k − ψ(k) for k ∈ [0, 500]. The function is strictly decreasing and asymptotically approaches zero.

log k − ψ(k) − log <sup>E</sup>[z ∗ <sup>x</sup>] + <sup>E</sup>[log z ∗ x], we solve the following optimization problem:

$$\min_k \xi(k)^2 + \lambda (k - 1)^2, \quad (14)$$

where λ is a regularization parameter which is set to 1 in our experiments.

With this analysis, the value k ∗ <sup>x</sup> is determined by minimizing Eq. [\(14\)](#page-4-4) using Brent's method [\(Brent, 2013\)](#page-8-9), where expectations of z ∗ <sup>x</sup> are estimated via Monte Carlo sampling from p(y ∗ , y<sup>x</sup> | Dt). Once k ∗ <sup>x</sup> is obtained, the corresponding β ∗ <sup>x</sup> follows as:

$$\beta_{\mathbf{x}}^* \leftarrow \frac{k_{\mathbf{x}}^*}{\mathbb{E}[z_{\mathbf{x}}^*]}. \quad (15)$$

Notably, the weighting parameters k ∗ <sup>x</sup> and β ∗ <sup>x</sup> are locationdependent, as z ∗ <sup>x</sup> itself varies with x. The VES-Gamma algorithm, which incorporates these principles, is detailed in Algorithm [2.](#page-5-0)

Although we provide both a theoretical justification and a practical implementation for the VES-Gamma AF, a deeper interpretation of the ESLBO in Eq. [\(12\)](#page-4-1) remains an open research question. Due to the complex non-linear structure of Eq. [\(12\)](#page-4-1), it is currently uncertain if there is a clear and straightforward interpretation of the various terms and the overall expression. As an example, we hypothesize that the third term acts as an "anti-EI" component, steering the VES-Gamma solution away from the EI recommendation to promote diversity, with the values of β ∗ <sup>x</sup> and k ∗ <sup>x</sup> dynamically balancing its influence. Investigating this hypothesis

Algorithm 2 VES-Gamma

Input: Sample set Dt, number of inner iterations N

Output: Next sampling location xt+1

- 1: initialize x
- (0) t+1 2: for n = 1 : N do 3: Evaluate values of <sup>E</sup> [z ∗ <sup>x</sup>] and <sup>E</sup> [log (z ∗ <sup>x</sup>)] by sampling p(y ∗ , y<sup>x</sup> | Dt) given x = x (n−1) t+1 4: Solve k
- (n) from Eq. [\(13\)](#page-4-3) 5: Solve β
- (n) from Eq. [\(15\)](#page-4-5) 6: Update x
  - (n) <sup>t</sup>+1 ← arg max<sup>x</sup> ESLBO(x; k
- (n) , β(n) ) defined in Eq. [\(12\)](#page-4-1) 7: end for 8: return x
  - (N) t+1

Table 2. Average duration of a BO loop for each AF. We measure the runtime on the Branin, Levy, and Hartmann benchmarks and average over benchmarks, BO iterations, and 10 random restarts. For N = 5 outer repetitions, VES has a higher runtime than the other acquisition functions.

| AF  | average time per BO iteration |
|-----|-------------------------------|
| EI  | 1 627 s ( ± 0 916 s )         |
| MES | 1 120 s ( ± 0 472 s )         |
| VES | 10 910 s ( ± 12 323)          |

and further elucidating the role of each term within ESLBO will be the focus of future research.

Computational Cost of VES-Gamma. Implementing VES-Gamma in Algorithm [2](#page-5-0) is computationally intensive. The number of inner iterations, N, must be sufficiently large for convergence, and each inner iteration requires estimating <sup>E</sup>[z ∗ <sup>x</sup>] by sampling a large number of y ∗ . Consequently, the overall BO loop takes significantly more time than EI and MES, as shown in Table [2](#page-5-1) for N = 5. However, since black-box function evaluations are often expensive, the additional computational cost of VES-Gamma is not a major bottleneck in many real-world applications.

# 4. Results

#### 4.1. Experimental Setup

We employ a consistent Gaussian Process (GP) hyperparameter and prior setting across all benchmarks and acquisition functions, evaluating Bayesian optimization (BO) performance using the simple regret r(t) := f <sup>∗</sup> − max(xi,yx<sup>i</sup> )∈D<sup>t</sup> y<sup>x</sup><sup>i</sup> , where f ∗ := maxx∈X f(x). When f ∗ is unknown, we instead report the negative best function value, − max(xi,yx<sup>i</sup> )∈D<sup>t</sup> y<sup>x</sup><sup>i</sup> .

To warm-start the optimization process, we initialize with 20 random samples drawn uniformly from X and model the GP using a <sup>5</sup>/2-Matern kernel with automatic relevance ´ determination (ARD) and a dimensionality-scaled lengthscale prior [\(Hvarfner et al., 2024\)](#page-9-18). Following the theoretical assumption in the VES framework, we only focus on experiments with noise-free observations. Although all benchmarks are noiseless, we allow the GP to accommodate potential non-stationarity or discontinuities in the underlying function.

Each experiment is repeated 10 times to estimate average performance, with results reported as mean ± one standard deviation. For problems with dimension less than 50, we run 100 iterations, otherwise 1000 iterations are computed. For numerical stability in VES-Gamma, we apply clamping: z ∗ <sup>x</sup> = max{10−<sup>10</sup>, y<sup>∗</sup> − max {yx, y<sup>∗</sup> <sup>t</sup> }}. The expectation in Eq. [\(12\)](#page-4-1) is estimated via pathwise conditioning [\(Wilson et al., 2021\)](#page-10-5) using 128 posterior samples. Additionally, the number of inner iterations N in Algorithm [2](#page-5-0) is set to 5, with early stopping applied if ∥x (n−1) − x (n)∥ < d · 10−<sup>5</sup> , where d denotes the problem dimension. We implement VES-Gamma and our other experiments using BoTorch [\(Balandat et al., 2020\)](#page-8-10). We always compare against LogEI [\(Ament et al., 2023\)](#page-8-11) and use EI and LogEI interchangeably. The code is available in [https://github.com/NUOJIN/variational-entropy-search.](https://github.com/NUOJIN/variational-entropy-search)

Benchmarks. To evaluate VES, we consider three distinct categories of benchmark problems: synthetic benchmarks, GP samples, and real-world optimization tasks.

For synthetic benchmarks, we examine commonly used functions that are diverse in dimensionality and landscape complexity. Specifically, we evaluate the 2-dimensional Branin, the 4-dimensional Levy, the 6-dimensional Hartmann, and the 8-dimensional Griewank functions. These benchmarks are widely utilized in optimization studies and provide controlled testbeds for algorithmic comparisons [\(Surjanovic & Bingham\)](#page-9-19).

For GP sample benchmarks, we draw from a GP prior with a ν = <sup>5</sup>/<sup>2</sup> Matern kernel. These experiments examine the ´ impact of varying length scales (ℓ = {0.5, 1, 2}) and dimensionalities (d = {2, 50, 100}) on algorithmic performance.

For real-world scenarios, we utilize a set of benchmarks reflecting practical high-dimensional problems. These include the 60-dimensional Rover problem [\(Wang](#page-10-6) [et al., 2018\)](#page-10-6), the 124-dimensional soft-constrained Mopta08 [\(Jones, 2008\)](#page-9-20) benchmark introduced in [Eriksson](#page-8-12) [& Jankowiak](#page-8-12) [\(2021\)](#page-8-12), the 180-dimensional Lasso-DNA problem from LassoBench (Sehi ˇ [c et al., 2022\)](#page-9-21), and the ´ 388-dimensional SVM benchmark, also introduced in [Eriks](#page-8-12)[son & Jankowiak](#page-8-12) [\(2021\)](#page-8-12). These tasks represent optimization challenges in engineering design, machine learning, and computational biology.

Due to space constraints, additional experiments are provided in Appendix [B.](#page-11-2)

#### 4.2. Comparing VES-Exp and EI

Kolmogorov-Smirnov Test. After establishing the theoretical equivalence of VES-Exp and EI in Section [3.2,](#page-3-5) we aim to validate this equivalence in our practical implementation. To this end, we employ the Kolmogorov-Smirnov (KS) two-sample test with a significance level of α = 5% to assess statistical similarity. The two samples consist of function values evaluated by each acquisition function (AF) across 10 repeated trials, i.e., YEI(t) := {y i t} 10 <sup>i</sup>=1, where y i t denotes the function evaluation at step t in the i-th trial. The null hypothesis states that the function evaluations from VES-Exp and EI originate from the same distribution.

We collect function values for all 500 iterations and consider a test successful (pass) for each iteration t if the null hypothesis is not rejected. We include six different benchmarks spanning low-dimensional synthetic problems to high-dimensional real-world scenarios. Additional implementation details on KS test are presented in Appendix [C.](#page-12-0)

Empirical Equivalence Results Figure [3](#page-7-0) illustrates the function values obtained by VES-Exp and EI, while Table [3](#page-6-0) reports the passing rates of the KS test across six benchmarks. The results show that all passing rates exceed 90%, with the Hartmann benchmark achieving the highest proportion of accepted tests.

Several factors explain the remaining discrepancies between VES-Exp and EI. First, since both acquisition functions are non-convex, their optimization may yield different next sampling points xt+1 due to variations in initialization. Second, VES methods employ a clamping mechanism to ensure that z ∗ <sup>x</sup> remains numerically positive, which introduces a dependency between y ∗ and x. In practice, this violates the assumptions used in the proof in Appendix [A.2.](#page-11-1) We also employed Log-EI [\(Ament et al., 2023\)](#page-8-11) instead of EI in our experiment, which may also explain the difference. Finally, while EI has a closed-form expression, VES-Exp relies on Monte Carlo estimation, introducing numerical inexactness and potential discrepancies.

## 4.3. Performance of VES-Gamma

Synthetic Test Functions. Figure [4](#page-7-1) illustrates the performance of various methods, including MES, EI, and VES-Gamma, across four synthetic benchmark functions: Branin (d = 2), Levy (d = 4), Hartmann (d = 6), and Griewank (d = 8). The metric shown is the logarithm of the best value (or simple regret), averaged over 10 independent runs.

Table 3. Kolmogorov-Smirnov two-sample test passing rate between VES-Exp and EI for various benchmarks. More details about p-values are available in Figure [8](#page-13-0) in the appendix.

|          |     |   |     | Passing | Rate (%) |
|----------|-----|---|-----|---------|----------|
| Branin   | (   | d | =   | 2)      | 94.00    |
| Hartmann |     |   | (   | d = 6)  | 99.80    |
| Rover    | (   | d | =   | 60)     | 92.60    |
| Mopta08  |     |   | ( d | = 124)  | 93.20    |
| Prior    | ( d | = | 2)  |         | 93.60    |
| Prior    | ( d | = | 50) |         | 94.60    |

On Branin, VES-Gamma achieves the best performance, with MES and EI lagging behind. For Levy, VES-Gamma and EI are effectively tied for the best results, with MES showing slightly worse performance. On the Hartmann function, VES-Gamma outperforms all other methods. Finally, for the Griewank function, VES-Gamma and EI once again demonstrate similar performance, significantly outperforming MES.

Overall, these results highlight the robustness of VES-Gamma across diverse synthetic benchmarks, consistently ranking among the top-performing methods.

GP Samples. Here, we study problem instances where the GP can be fitted without model mismatch. To this end, we sample realizations from an isotropic 100-dimensional GP prior with varying length scale ℓ = 0.05, 0.1, 0.25, 0.5, using the same <sup>5</sup>/2-Matern covariance function for the GP ´ prior and the GP we fit to the observations.

Figure [5](#page-8-13) shows the optimization performance on the 100 dimensional GP prior samples. For ℓ = 0.05, 0.1, 0.25, VES-Gamma outperforms EI and MES by a wide margin. EI and MES converge to a suboptimal solution. Only for ℓ = 0.5 does EI reach the same quality as VES-Gamma, outperforming MES.

Real-World Benchmarks. Figure [6](#page-8-14) presents the performance of VES-Gamma, EI, and MES across four realworld optimization problems: the 60-dimensional Rover trajectory optimization, the 124-dimensional Mopta08 vehicle optimization, the 180-dimensional weighted Lasso-DNA regression, and the 388-dimensional SVM hyperparameter tuning benchmarks.

Consistent with previous observations, VES-Gamma delivers strong performance, significantly outperforming all other acquisition functions on the SVM benchmark. It also ranks among the top-performing methods, alongside EI, on the Mopta08 and Lasso-DNA benchmarks. On the Rover problem, VES-Gamma performs comparably to EI, while MES achieves the best results in this scenario.

![](_page_7_Figure_1.jpeg)

Figure 3. Function values observed at each BO iteration for the EI and VES-Exp acquisition functions.

MES exhibits mixed performance across the benchmarks, achieving the best results on Rover but falling behind on the Mopta08 and SVM problems.

Overall, VES-Gamma demonstrates robust and consistent performance across all benchmarks, establishing itself as a versatile and reliable acquisition function for highdimensional real-world optimization problems.

# 5. Conclusion

In this work, we introduce Variational Entropy Search (VES), a unified framework that bridges Expected Improvement (EI) and information-theoretic acquisition functions through a variational inference approach. We demonstrate that EI can be interpreted as a special case of Maxvalue Entropy Search (MES), revealing a deeper theoretical connection between these two widely used methodologies in Bayesian optimization. Building on this insight, we propose VES-Gamma, a novel acquisition function that dynamically balances the strengths of EI and MES. Comprehensive benchmark evaluations across a diverse set of lowand high-dimensional optimization problems highlight the robust and consistently high performance of VES-Gamma. These results underscore the potential of the VES framework as a promising foundation for developing more adaptive and efficient acquisition functions in Bayesian optimization.

![](_page_7_Figure_2.jpeg)

Figure 4. VES-Gamma, EI, and MES on the synthetic Branin (d = 2), Levy (d = 4), Hartmann (d = 6), and Griewank (d = 8) benchmark functions. Average log simple regret: VES-Gamma performs best on Branin and Hartmann, and it is competitive on Levy and Griewank.

Limitations and future work. While the Gamma distribution offers flexibility, future work will explore alternative variational distributions to enhance the adaptability of VES-Gamma. Another key direction is improving computational efficiency. Additionally, extending the theoretical framework to noisy settings remains an open challenge, requiring adaptations in variational inference to account for stochastic density supports.

# Acknowledgements

This project was partly supported by the Wallenberg AI, Autonomous Systems, and Software program (WASP) funded by the Knut and Alice Wallenberg Foundation, the AFOSR awards FA9550-20-1-0138, with Dr. Fariba Fahroo as the program manager, DOE award DE-SC0023346, and by the US Department of Energy's Wind Energy Technologies Office. The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725

# Impact Statement

This paper presents work that aims to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

![](_page_8_Figure_1.jpeg)

Figure 5. Performance curves (best values up to each iteration). VES-Gamma shows superior performance on all but one problem where it performs as good as EI.

- References Abramowitz, M., Stegun, I. A., and Romer, R. H. Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables, 1988. Ament, S., Daulton, S., Eriksson, D., Balandat, M., and Bakshy, E. Unexpected Improvements to Expected Improvement for Bayesian Optimization. *Advances in Neural Information Processing Systems*, 36:20577–20612, 2023. Balandat, M., Karrer, B., Jiang, D., Daulton, S., Letham, B., Wilson, A. G., and Bakshy, E. BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. *Advances in Neural Information Processing Systems*, 33: 21524–21538, 2020. Barber, D. and Agakov, F. The IM Algorithm: A variational approach to Information Maximization. *Advances in Neural Information Processing Systems*, 16(320):201, 2004. Benjamins, C., Raponi, E., Jankovic, A., Doerr, C., and Lindauer, M. Self-Adjusting Weighted Expected Improvement for Bayesian Optimization. In *International Conference on Automated Machine Learning*, pp. 6–1. PMLR, 2023. Berk, J., Nguyen, V., Gupta, S., Rana, S., and Venkatesh,
- S. Exploration Enhanced Expected Improvement for Bayesian Optimization. In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2018, Dublin, Ireland, September 10–14, 2018, Proceedings, Part II 18*, pp. 621–637. Springer, 2019. Brent, R. P. *Algorithms for minimization without derivatives*. Courier Corporation, 2013. De Ath, G., Everson, R. M., Rahat, A. A., and Fieldsend, J. E. Greed is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation. *ACM Transactions on Evolutionary Learning and Optimization*, 1(1):1–22, 2021. Eriksson, D. and Jankowiak, M. High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces. In *Uncertainty in Artificial Intelligence*, pp. 493–
  - 503. PMLR, 2021. Frazier, P. I., Powell, W. B., and Dayanik, S. A Knowledge-Gradient Policy for Sequential Information Collection. *SIAM Journal on Control and Optimization*, 47(5): 2410–2439, 2008. Golub, G. H. and Pereyra, V. The differentiation of pseudoinverses and nonlinear least squares problems whose variables separate. *SIAM Journal on Numerical Analysis*, 10(2):413–432, 1973. Hennig, P. and Schuler, C. J. Entropy Search for Information-Efficient Global Optimization. *Journal of Machine Learning Research*, 13(6), 2012. Hennig, P., Osborne, M. A., and Kersting, H. P. *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press, 2022. Hernandez-Lobato, J. M., Hoffman, M. W., and Ghahra- ´ mani, Z. Predictive Entropy Search for Efficient Global Optimization of Black-box Functions. *Advances in Neural Information Processing Systems*, 27, 2014.

![](_page_8_Figure_2.jpeg)

Figure 6. Performance curves (best function value up to each iteration). VES-Gamma outperforms all other AFs on SVM and performs well on the other problems.

- Hoffman, M., Brochu, E., and de Freitas, N. Portfolio Allocation for Bayesian Optimization. In *Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence*, pp. 327–336, 2011. Hoffman, M. D., Blei, D. M., Wang, C., and Paisley, J. Stochastic Variational Inference. *Journal of Machine Learning Research*, 2013. Hvarfner, C., Hutter, F., and Nardi, L. Joint Entropy Search for Maximally-Informed Bayesian Optimization. *Advances in Neural Information Processing Systems*, 35: 11494–11506, 2022. Hvarfner, C., Hellsten, E. O., and Nardi, L. Vanilla Bayesian Optimization Performs Great in High Dimensions. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 20793–20817. PMLR, 21–27 Jul 2024. Jones, D. R. Large-Scale Multi-Disciplinary Mass Optimization in the Auto Industry. In *MOPTA 2008 Conference (20 August 2008)*, 2008. Jones, D. R., Schonlau, M., and Welch, W. J. Efficient global optimization of expensive black-box functions. *Journal of Global optimization*, 13:455–492, 1998. Kandasamy, K., Vysyaraju, K. R., Neiswanger, W., Paria, B., Collins, C. R., Schneider, J., Poczos, B., and Xing,
- E. P. Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly. *Journal of Machine Learning Research*, 21(81): 1–27, 2020. Kingma, D., Salimans, T., Poole, B., and Ho, J. Variational diffusion models. *Advances in Neural Information Processing Systems*, 34:21696–21707, 2021. Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes. In Bengio, Y. and LeCun, Y. (eds.), *2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings*, 2014. Ma, H., Zhang, T., Wu, Y., Calmon, F. P., and Li, N. Gaussian Max-Value Entropy Search for Multi-Agent Bayesian Optimization. In *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 10028–10035. IEEE, 2023. Mockus, J. The application of Bayesian methods for seeking the extremum. *Towards global optimization*, 2:117, 1998. Paisley, J., Blei, D., and Jordan, M. Variational Bayesian Inference with Stochastic Search. In *Proceedings of the International Conference on Machine Learning*, 2012. Poole, B., Ozair, S., Van Den Oord, A., Alemi, A., and Tucker, G. On Variational Bounds of Mutual Information. In *International Conference on Machine Learning*, pp. 5171–5180. PMLR, 2019. Poon, C. and Peyre, G. Smooth over-parameterized solvers ´ for non-smooth structured optimization. *Mathematical Programming*, 201(1):897–952, 2023. Qin, C., Klabjan, D., and Russo, D. Improving the Expected Improvement Algorithm. *Advances in Neural Information Processing Systems*, 30, 2017. Qing, J., Moss, H. B., Dhaene, T., and Couckuyt, I. {PF} <sup>2</sup>ES: Parallel Feasible Pareto Frontier Entropy Search for Multi-Objective Bayesian Optimization. In *26th International Conference on Artificial Intelligence and Statistcs (AISTATS) 2023*, volume 206, pp. 2565– 2588, 2023. Rasmussen, C. E., Williams, C. K., et al. *Gaussian Processes for Machine Learning*, volume 1. Springer, 2006. Sehi ˇ c, K., Gramfort, A., Salmon, J., and Nardi, L. Las- ´ soBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso. In *International Conference on Automated Machine Learning*, pp. 2–1. PMLR, 2022. Snoek, J., Larochelle, H., and Adams, R. P. Practical Bayesian Optimization of Machine Learning Algorithms. In Pereira, F., Burges, C., Bottou, L., and Weinberger, K. (eds.), *Advances in Neural Information Processing Systems*, volume 25. Curran Associates, Inc., 2012. Sobester, A., Leary, S. J., and Keane, A. J. On the Design of ´ Optimization Strategies Based on Global Response Surface Approximation Models. *Journal of Global Optimization*, 33:31–59, 2005. Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. In *Proceedings of the International Conference on Machine Learning*, 2010. Surjanovic, S. and Bingham, D. Virtual library of simulation experiments: Test functions and datasets, optimization test problems. [https://www.sfu.ca/](https://www.sfu.ca/~ssurjano/optimization.html) [˜ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html). Accessed: 2024-09-01. Takeno, S., Fukuoka, H., Tsukada, Y., Koyama, T., Shiga, M., Takeuchi, I., and Karasuyama, M. Multi-fidelity

- Bayesian Optimization with Max-value Entropy Search and its Parallelization. In *International Conference on Machine Learning*, pp. 9334–9345. PMLR, 2020. Takeno, S., Tamura, T., Shitara, K., and Karasuyama, M. Sequential and Parallel Constrained Max-value Entropy Search via Information Lower Bound. In *Proceedings of the 39th International Conference on Machine Learning (ICML)*, volume 162 of *Proceedings of Machine Learning Research*, pp. 20960–20986. PMLR, June 2022. Tu, B., Gandy, A., Kantas, N., and Shafei, B. Joint Entropy Search for Multi-objective Bayesian Optimization. *Advances in Neural Information Processing Systems*, 35: 9922–9938, 2022. Villemonteix, J., Vazquez, E., and Walter, E. An informational approach to the global optimization of expensiveto-evaluate functions. *Journal of Global Optimization*, 44:509–534, 2009. Wang, Z. and Jegelka, S. Max-value Entropy Search for Efficient Bayesian Optimization. In *International Conference on Machine Learning*, pp. 3627–3635. PMLR, 2017. Wang, Z., Gehring, C., Kohli, P., and Jegelka, S. Batched Large-scale Bayesian Optimization in High-dimensional Spaces. In *International Conference on Artificial Intelligence and Statistics*, pp. 745–754. PMLR, 2018. Wilson, J. T., Moriconi, R., Hutter, F., and Deisenroth,
- M. P. The Reparameterization Trick for Acquisition Functions. In *NIPS Workshop on Bayesian Optimization*, 2017. URL [https://bayesopt.github.](https://bayesopt.github.io/papers/2017/32.pdf) [io/papers/2017/32.pdf](https://bayesopt.github.io/papers/2017/32.pdf). Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P., and Deisenroth, M. P. Pathwise Conditioning of Gaussian Processes. *Journal of Machine Learning Research*, 22(105):1–47, 2021.

### A. Proofs

#### A.1. ESLB Proof

The MES acquisition function in Eq. [\(4\)](#page-1-0) can be lower bounded as follows,

*Proof.*

$$\begin{aligned} \alpha_{\text{MES}}(\mathbf{x}) &= \mathbb{H}[y^* \mid \mathcal{D}_t] - \mathbb{E}_{p(y_{\mathbf{x}} \mid \mathcal{D}_t)} \mathbb{H}[y^* \mid \mathcal{D}_t, y_{\mathbf{x}}] \\ &= \mathbb{H}[y^* \mid \mathcal{D}_t] + \mathbb{E}_{p(y^*, y_{\mathbf{x}} \mid \mathcal{D}_t)} [\log(p(y^* \mid \mathcal{D}_t, y_{\mathbf{x}}))] \\ &= \mathbb{H}[y^* \mid \mathcal{D}_t] + \mathbb{E}_{p(y^*, y_{\mathbf{x}} \mid \mathcal{D}_t)} \left[ \log \left( \frac{p(y^* \mid \mathcal{D}_t, y_{\mathbf{x}}) q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}})}{q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}})} \right) \right] \\ &= \mathbb{H}[y^* \mid \mathcal{D}_t] + \mathbb{E}_{p(y^*, y_{\mathbf{x}} \mid \mathcal{D}_t)} [\log(q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}}))] + \mathbb{E}_{p(y_{\mathbf{x}} \mid \mathcal{D}_t)} [D_{\text{KL}}(p(y^* \mid \mathcal{D}_t, y_{\mathbf{x}}) \| q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}}))] \\ &\geq \mathbb{H}[y^* \mid \mathcal{D}_t] + \mathbb{E}_{p(y^*, y_{\mathbf{x}} \mid \mathcal{D}_t)} [\log(q(y^* \mid \mathcal{D}_t, y_{\mathbf{x}}))], \end{aligned}$$

where the KL divergence DKL(p(x)∥q(x)) := <sup>E</sup>p(x) [log(p(x)/q(x))]. The inequality is tight if and only if <sup>E</sup>p(yx|Dt) [DKL p(y ∗ | Dt, yx)∥q(y ∗ | Dt, yx) ] = 0, which implies p(y ∗ | Dt, yx) = q(y ∗ | Dt, yx) for all y<sup>x</sup> | Dt.

# A.2. VES-Exp and EI Algorithmic Equivalence

Theorem [3.2](#page-3-3) is proved as follows:

*Proof.* By restricting the variational distributions to exponential distributions, we slightly abuse the input notations of ESLBO in [\(8\)](#page-3-2) and define:

$$\begin{aligned} \text{ESLBO}(\lambda, \mathbf{x}) &= \mathbb{E}_{p(y^*, y_{\mathbf{x}}|\mathcal{D}_t)} [\log(\lambda \exp(-\lambda(y^* - \max\{y_{\mathbf{x}}, y_t^*\})))] \\ &= \log \lambda - \lambda \mathbb{E}_{p(y^*, y_{\mathbf{x}}|\mathcal{D}_t)} [(y^* - \max\{y_{\mathbf{x}}, y_t^*\})] \\ &= \log \lambda - \lambda \underbrace{\mathbb{E}_{p(y^*|\mathcal{D}_t)} [y^*]}_{\text{constant}} + \lambda \underbrace{\mathbb{E}_{p(y_{\mathbf{x}}|\mathcal{D}_t)} [\max\{y_{\mathbf{x}}, y_t^*\}]}_{\text{EI AF}}. \end{aligned} \quad (16)$$

Beginning with an arbitrary initial value x (0), we determine the corresponding parameter

$$\lambda^{(1)} = \frac{1}{\mathbb{E}_{p(y^*, y_{\mathbf{x}(0)} | \mathcal{D}_t)} [(y^* - \max\{y_{\mathbf{x}(0)}, y_t^*\})]}, \quad (17)$$

which is derived by taking the derivative of [\(16\)](#page-11-3) and letting it equal zero. With λ fixed, ESLBO(λ (1) , x) produces the same result as the EI acquisition function in [\(1\)](#page-1-2). We then compute λ (2) based on x (1) following [\(17\)](#page-11-4). Regardless of the specific value of λ (2), the ESLBO function consistently yields the same result, x (1). This consistency ensures that the VES iteration process converges in a single step. The final outcome, represented as (x (1), λ(2)), indicates that the corresponding q(y ∗ | yx, Dt) is the closest approximation to p(y ∗ | yx, Dt) within Qexp (in the sense that minimizes their KL divergence).

# B. Additional Experimental Results

#### B.1. Synthetic Test Functions

![](_page_12_Figure_2.jpeg)

Figure 7. Performance plots for EI, MES, and VES-Gamma on additional synthetic benchmark functions. VES-Gamma shows robust performance throughout the bank.

Figure [7](#page-12-1) shows the performance of the different acquisition functions, EI, MES, and VES-Gamma, on additional synthetic benchmark functions: the Ackley and Michalewicz test functions[<sup>1</sup>](#page-12-2) and the Lasso-High and Lasso-Hard benchmarks (Sehi ˇ [c et al., 2022\)](#page-9-21). On the 1000-dimensional ´ Lasso-Hard problem, VES-Gamma ran into a timeout after 48 hours. Therefore, we plot the mean up to the minimum number of iterations performed across all repetitions. VES-Gamma demonstrates robust performance across the benchmarks, outperforming all other acquisition functions on Ackley, MES on Michalewicz, and performing similarly to the other acquisition functions on the Lasso benchmarks. VES-Gamma and MES perform considerably worse than VES-Gamma, especially on the more high-dimensional problems.

# C. Kolmogorov-Smirnov Test Statistic

The Kolmogorov-Smirnov (KS) two-sample test is a non-parametric statistical method used to determine whether two samples are drawn from the same continuous distribution. It compares their empirical cumulative distribution functions (ECDFs) and calculates a test statistic that quantifies their maximum difference. Given two independent samples as function evaluations from VES-Exp {X1, X2, . . . , X<sup>n</sup><sup>1</sup> } and from EI {Y1, Y2, . . . , Y<sup>n</sup><sup>2</sup> }, their ECDFs are defined as:

$$F_X(x) = \frac{1}{n_1} \sum_{i=1}^{n_1} \mathbb{I}(X_i \leq x), \quad F_Y(x) = \frac{1}{n_2} \sum_{j=1}^{n_2} \mathbb{I}(Y_j \leq x),$$

where <sup>I</sup>(·) is the indicator function, equal to 1 if the condition is true and 0 otherwise. The KS test statistic is given by:

$$D = \sup_x |F_X(x) - F_Y(x)|,$$

where sup<sup>x</sup> denotes the supremum over all possible values of x. This statistic measures the maximum absolute difference between the ECDFs of the two samples.

Statistical Hypotheses. The hypotheses for the KS test are defined as:

- Null hypothesis (H0): FX(x) = F<sup>Y</sup> (x) for all x (the two samples come from the same distribution).
- Alternative hypothesis (Ha): FX(x) ̸= F<sup>Y</sup> (x) for at least one x (the two samples come from different distributions).

To test these hypotheses, the test p-value is solved using the Kolmogorov-Smirnov survival function:

$$p_{\text{test}} = Q_{\text{KS}} \left( \sqrt{\frac{n_1 n_2}{n_1 + n_2}} D \right),$$

<sup>1</sup>[https://www.sfu.ca/˜ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)

where QKS(·) represents the survival function of the Kolmogorov distribution:

$$Q_{\text{KS}}(z) = 2 \sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2 z^2}.$$

Alternatively, the significance level α = 0.05 can be tested using the critical value:

$$D_{0.05} \approx \sqrt{-\frac{1}{2} \ln(0.025)} \cdot \sqrt{\frac{n_1 + n_2}{n_1 n_2}}.$$

If D > D0.05, we reject the null hypothesis and consider it as failure (not pass).

Detailed p-values for VES-Exp and EI Comparison. We present the p-values obtained from the experiments detailed in Section [4.2.](#page-6-1) These results are illustrated in Figure [8.](#page-13-0) It is observed that for the majority of the sample pairs, the calculated p-values are substantially above the 5% significance level.

![](_page_13_Figure_8.jpeg)

Figure 8. Distribution of p-values for 500 sample pairs generated using the EI and VES-Exp acquisition functions.

# D. VES-Gamma Computational Acceleration

Table [2](#page-5-1) highlights the higher computational cost of VES methods compared to EI and MES. However, we observe that a technique known as Variable Projection (VarPro) [\(Golub & Pereyra, 1973;](#page-8-15) [Poon & Peyre, 2023\)](#page-9-22) can be leveraged to ´ accelerate the computation of VES under certain conditions, which VES-Gamma satisfies.

The key idea behind VarPro is that when the function ESLBO has a specific structure,

$$\max_{\mathbf{x}; k, \beta} \text{ESLBO}(\mathbf{x}; k, \beta) = \max_{\mathbf{x}} \left( \underbrace{\max_{k, \beta} \text{ESLBO}(\mathbf{x}; k, \beta)}_{\varphi(\mathbf{x})} \right), \quad (18)$$

and the solution to maxk,β ESLBO(x; k, β) is unique, then φ(x) is differentiable, with

$$\frac{d}{dx}\varphi(x) = \frac{\partial}{\partial x}\text{ESLBO}(x, k_x^*, \beta_x^*), \quad (19)$$

where k ∗ and β ∗ are the unique values that maximize ESLBO.

Following the proof in Eq. [\(13\)](#page-4-3), we establish that the solutions k ∗ <sup>x</sup> and β ∗ <sup>x</sup> are unique. This confirms that it is feasible to implement the VarPro strategy to accelerate the computation of VES-Gamma, eliminating the need for the iterative scheme in Algorithm [1.](#page-3-0) This ongoing work aims to reduce the computational cost of VES-Gamma to a level comparable to EI and MES.