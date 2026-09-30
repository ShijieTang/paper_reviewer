Jesse Farebrother 1 2 † Matteo Pirotta <sup>3</sup> Andrea Tirinzoni <sup>3</sup> Remi Munos ´ 3 Alessandro Lazaric <sup>3</sup> Ahmed Touati <sup>3</sup>

# Abstract

Predictive models of the future are fundamental for an agent's ability to reason and plan. A common strategy learns a world model and unrolls it step-by-step at inference, where small errors can rapidly compound. Geometric Horizon Models (GHMs) offer a compelling alternative by directly making predictions of future states, avoiding cumulative inference errors. While GHMs can be conveniently learned by a generative analog to temporal difference (TD) learning, existing methods are negatively affected by bootstrapping predictions at train time and struggle to generate high-quality predictions at long horizons. This paper introduces Temporal Difference Flows (TD-Flow), which leverages the structure of a novel Bellman equation on probability paths alongside flow-matching techniques to learn accurate GHMs at over 5× the horizon length of prior methods. Theoretically, we establish a new convergence result and primarily attribute TD-Flow's efficacy to reduced gradient variance during training. We further show that similar arguments can be extended to diffusion-based methods. Empirically, we validate TD-Flow across a diverse set of domains on both generative metrics and downstream tasks, including policy evaluation. Moreover, integrating TD-Flow with recent behavior foundation models for planning over policies demonstrates substantial performance gains, underscoring its promise for long-horizon decision-making.

# 1. Introduction

Predictive modeling lies at the heart of intelligent decisionmaking, enabling agents to reason and plan in complex environments. In Reinforcement Learning (RL), this predictive capability has traditionally been achieved through world models that capture the transition structure of the environment. These models have enabled significant advances across numerous domains — from robotics manipulation employing model-predictive control [\(Sikchi et al.,](#page-11-0) [2021;](#page-11-0) [Hafner et al.,](#page-9-0) [2023;](#page-9-0) [Hansen et al.,](#page-9-1) [2022;](#page-9-1) [2024\)](#page-9-2), to sampleefficient exploration strategies [\(Schmidhuber,](#page-11-1) [1991;](#page-11-1) [Stadie](#page-11-2) [et al.,](#page-11-2) [2016;](#page-11-2) [Pathak et al.,](#page-10-0) [2017\)](#page-10-0), and sophisticated planning algorithms [\(Silver et al.,](#page-11-3) [2016;](#page-11-3) [2017;](#page-11-4) [Schrittwieser](#page-11-5) [et al.,](#page-11-5) [2020\)](#page-11-5). However, while world models have demonstrated impressive results, they face fundamental limitations when deployed for long-horizon reasoning. The standard approach of unrolling predictions step-by-step leads to compounding errors, as small inaccuracies in each prediction accumulate and propagate forward in time [\(Talvitie,](#page-11-6) [2014;](#page-11-6) [Jafferjee et al.,](#page-9-3) [2020;](#page-9-3) [Lambert et al.,](#page-10-1) [2022\)](#page-10-1). This "curse of horizon" presents a significant challenge for applications requiring reliable long-range predictions.

An alternative approach is to learn a generative model of future states directly, avoiding compounding errors during inference. These models, usually referred to as Geometric Horizon Models (GHM; [Thakoor et al.,](#page-11-7) [2022\)](#page-11-7) or γ-models [\(Janner et al.,](#page-9-4) [2020\)](#page-9-4), are learned by leveraging the temporal difference structure of the successor measure [\(Blier et al.,](#page-8-0) [2021\)](#page-8-0). However, their reliance on bootstrapped predictions during training can lead to instability and growing inaccuracy over long horizons. As a result, current methods struggle to make accurate predictions beyond 20- 50 steps, also limiting their utility for long-term decisionmaking. In this paper, we show that while state-of-the-art generative methods like flow matching [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2) and denoising diffusion [\(Ho et al.,](#page-9-5) [2020\)](#page-9-5) cannot be directly applied to learn long-horizon GHMs, their iterative nature can be leveraged to better exploit the temporal difference structure of the problem. This insight yields a new class of methods that provably converges to the successor measure while reducing the variance of their sample-based gradient estimates, enabling stable long-horizon predictions. Empirically, our approach produces significantly more accurate GHMs at all horizons, consistently outperforming state-ofthe-art algorithms across domains and metrics, including prediction accuracy, value function estimation, and generalized policy improvement.

<sup>†</sup>Work done at Meta <sup>1</sup>McGill University <sup>2</sup>Mila - Quebec AI ´ Institute <sup>3</sup> FAIR at Meta. Correspondence to: Jesse Farebrother <jfarebro@cs.mcgill.ca>, Ahmed Touati <atouati@meta.com>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

#### 2. Background

In the following, we use capital letters to denote random variables, sans-serif fonts for sets, and P(A) to denote the space of probability measures over a measurable set A.

Markov Decision Process We consider a reward-free discounted Markov decision process M = (S, A, P, γ), which characterizes the dynamics of a sequential decision-making problem. At each step, the agent selects an action a ∈ A in state s ∈ S according to its policy π : S → A. This action influences the transition to the next state s ′ ∈ S, governed by the transition kernel P : S × A → P(S), which defines a probability measure over successor states. The discount factor γ ∈ [0, 1) can be interpreted as implying a process that either continues with probability γ or terminates with probability 1 − γ. This interpretation naturally defines a geometric distribution of future states the agent will occupy, where states reached after k steps are discounted by γ k .

Successor Measure The normalized *successor measure* [\(Dayan,](#page-9-6) [1993;](#page-9-6) [Blier et al.,](#page-8-0) [2021\)](#page-8-0) of a policy π describes the discounted distribution of future states visited by π starting from an initial state-action pair (s, a). For the measurable subset X ⊆ S the successor measure m<sup>π</sup> (X | s, a) represents the probability that future states fall within X, geometrically discounted by γ according to the time of visitation. Formally, it is defined as:

$$m^\pi(\mathbf{X} \mid s, a) = (1)$$

$$(1 - \gamma) \sum_{k=0}^{\infty} \gamma^k \Pr(S_{k+1} \in \mathbf{X} \mid S_0 = s, A_0 = a, \pi),$$

where Pr(· | S0, A0, π) denotes the probability of stateaction sequences (Sk, Ak)k≥<sup>0</sup> generated from (S0, A0) following S<sup>k</sup> ∼ P(· | Sk−1, Ak−1) and A<sup>k</sup> = π(Sk). The successor measure encapsulates the long-term dynamics of π, enabling value estimation for any reward function r : S → R. Specifically, the value of taking action a ∈ A in state s ∈ S is the expected reward under states visited by π amplified by the effective horizon (1 − γ) −1 :

$$Q^\pi(s, a) = (1 - \gamma)^{-1} \mathbb{E}_{X \sim m^\pi(\cdot | s, a)} [r(X)]. \quad (2)$$

Moreover, m<sup>π</sup> is the fixed point of the Bellman operator T π : P(S) <sup>S</sup>×<sup>A</sup> → P(S) <sup>S</sup>×<sup>A</sup> [\(Thakoor et al.,](#page-11-7) [2022\)](#page-11-7):

$$\begin{aligned} m^\pi(\cdot | s, a) &= (\mathcal{T}^\pi m^\pi)(\cdot | s, a) \\ &:= (1 - \gamma)P(\cdot | s, a) + \gamma (P^\pi m^\pi)(\cdot | s, a). \end{aligned} \quad (3)$$

The operator P π applied to m mixes the one-step kernel with the successor measure, accounting for transitioning from (s, a) to a new state-action pair (s ′ , π(s ′ )) and querying the successor measure m(· | s,′ π(s ′ )) thereafter:

$$(P^\pi m)(dx \mid s, a) = \int_{s'} P(ds' \mid s, a) m(dx \mid s', \pi(s')).$$

Geometric Horizon Model A *Geometric Horizon Model* (GHM; [Thakoor et al.,](#page-11-7) [2022\)](#page-11-7) or γ-model [\(Janner et al.,](#page-9-4) [2020\)](#page-9-4) is a generative model of the *normalized* successor measure. To learn the parametric model <sup>m</sup>e (· · · ; <sup>θ</sup>) <sup>≈</sup> <sup>m</sup><sup>π</sup> we can minimize a Monte-Carlo cross-entropy objective over source states from the empirical distribution ρ as,

$$\mathbb{E}_{s \sim \rho, X \sim m^\pi(\cdot | S, \pi(A))} \left[ -\log \tilde{m}(X | S, A; \theta) \right].$$

In order to sample from m<sup>π</sup> we deploy policy π for t ∼ Geom(1 − γ) steps resulting in state X = St. Similar to other Monte Carlo methods in RL, this approach is problematic when learning from off-policy data, often resulting in high-variance estimators that rely on importance sampling.

Alternatively, we can leverage the Bellman equation [\(3\)](#page-1-0) to construct an off-policy iterative method for estimating m<sup>π</sup> . Given initial weights θ (0), each iteration updates θ by minimizing the following temporal-difference cross-entropy objective over transitions that need not come from policy π,

$$\mathbb{E}_{(S,A) \sim \rho, X \sim (\mathcal{T}^\pi \tilde{m}^{(n)})(\cdot | S,A)} [-\log \tilde{m}(X \mid S, A; \theta)]. \quad (4)$$

In the equation above and throughout the paper, we adopt the shorthand <sup>m</sup>e (n) <sup>=</sup> <sup>m</sup>e (· · · ; <sup>θ</sup> (n) ). To generate samples X ∼ T πme (n) (· | S, A) we first draw a successor state S ′ ∼ P(· | S, A); then with probability 1 − γ, we return S ′ ; otherwise, with probability γ, we return a *bootstrapped sample* drawn from <sup>m</sup>e (n) (· | S ′ , π(S ′ )).

Several probabilistic models have been applied to this problem, including generative adversarial networks (e.g., [Janner](#page-9-4) [et al.,](#page-9-4) [2020;](#page-9-4) [Wiltzer et al.,](#page-12-0) [2024b\)](#page-12-0), normalizing flows (e.g., [Janner et al.,](#page-9-4) [2020\)](#page-9-4), and variational auto-encoders (e.g., [Thakoor et al.,](#page-11-7) [2022;](#page-11-7) [Tomar et al.,](#page-11-8) [2024\)](#page-11-8). We now turn our attention to a class of generative models based on the flowmatching framework specifically designed to leverage the underlying structure of the Bellman equation [\(3\)](#page-1-0), enabling more effective generative models of the successor measure.

# 3. Temporal Difference Flows

Flow Matching (FM; [Lipman et al.,](#page-10-2) [2023;](#page-10-2) [2024;](#page-10-3) [Liu et al.,](#page-10-4) [2023;](#page-10-4) [Albergo & Vanden-Eijnden,](#page-8-1) [2023\)](#page-8-1) constructs a timedependent probability path m<sup>t</sup> : S × A → P(S) for t ∈ [0, 1] that evolves smoothly from the source distribution m<sup>0</sup> = p<sup>0</sup> ∈ P(S) to the target distribution m<sup>1</sup> ≈ m<sup>π</sup> . This evolution is governed by a vector field v<sup>t</sup> : S × S × A → S, which dictates the instantaneous movement of samples along mt. The relationship between v<sup>t</sup> and the resulting probability path m<sup>t</sup> is established through a time-dependent flow ψ<sup>t</sup> : S × S × A → S, defined by the following ODE:

$$\begin{aligned} \frac{d}{dt}\psi_t(x \mid s, a) &= v_t(\psi_t(x \mid s, a) \mid s, a), \quad \psi_0(x \mid s, a) = x \\ \iff \psi_t(x \mid s, a) &= x + \int_0^t v_\tau(\psi_\tau(x \mid s, a) \mid s, a) d\tau. \end{aligned}$$

![](_page_2_Figure_2.jpeg)

Figure 1. Visual depiction of TD-Flow variants. Samples are mapped from m<sup>0</sup> to the target distribution m (n) 1 through the neural ODE ψ (n) t . Dashed lines depict the neural ODE trajectory; solid lines show the conditional probability path ut. (Left) TD-CFM maps X<sup>0</sup> to X<sup>1</sup> before creating a separate conditional path between X ′ <sup>0</sup> and X1, resulting in crossing paths. (Middle) TD-CFM(C) directly couples X<sup>0</sup> used to generate X<sup>1</sup> when constructing the conditional probability path. (Right) TD<sup>2</sup> -CFM solves the neural ODE up to time t to directly obtain the target velocity v˜t.

We say that v<sup>t</sup> generates m<sup>t</sup> if its flow ψ<sup>t</sup> satisfies X<sup>t</sup> := ψt(X<sup>0</sup> | S, A) ∼ mt(· | S, A) for X<sup>0</sup> ∼ m0. In words, the flow ψ<sup>t</sup> pushes samples forward through time, ensuring they are distributed according to m<sup>t</sup> at time t. To learn this transformation, we can minimize the squared L <sup>2</sup> distance between a parameterized vector field v˜t(· · · ; θ) and the true vector field v<sup>t</sup> over t ∼ U([0, 1]), yielding the Monte-Carlo Flow Matching (MC-FM) loss ℓMC-FM(θ):

$$\mathbb{E}_{\rho, t, X_t} \left[ \|\tilde{v}_t(X_t \mid S, A; \theta) - v_t(X_t \mid S, A)\|^2 \right],$$
where  $X_t \sim m_t(\cdot \mid S, A)$ . (MC-FM; 5)

Despite its conceptual simplicity, direct optimization of the flow matching objective above proves challenging due to the inaccessibility of the true probability path m<sup>t</sup> and its associated vector field vt.

Alternatively, [Lipman et al.](#page-10-2) [\(2023\)](#page-10-2) shows that we can sidestep this problem entirely by introducing additional conditioning information. Instead of directly modeling the probability path m<sup>t</sup> we can introduce a random variable Z and define a *conditional path* on Z as pt|<sup>Z</sup> : S×Z → P(S) [\(Lipman et al.,](#page-10-3) [2024;](#page-10-3) [Tong et al.,](#page-11-9) [2024\)](#page-11-9). The conditional velocity field ut|<sup>Z</sup> : S×Z → S that generates pt|<sup>Z</sup> can now be computed in closed form for many simple choices of Z and pt|Z. One such choice is taking Z = X<sup>1</sup> and performing a linear Gaussian interpolation from X<sup>0</sup> → X<sup>1</sup> resulting in pt|1(· | X1) = N (· | tX1, (1 − t) 2 I) with the corresponding vector field given by ut|1(x | X1) = (X<sup>1</sup> − x)/(1 − t). Armed with the ability to sample from pt|<sup>1</sup> and to compute ut|1, we can directly learn v˜<sup>t</sup> by optimizing the Monte-Carlo *Conditional* Flow Matching (MC-CFM) objective ℓMC-CFM(θ):

$$\mathbb{E}_{\rho, t, Z, X_t} \left[ \left\| \tilde{v}_t(X_t \mid S, A; \theta) - u_{t|Z}(X_t \mid Z) \right\|^2 \right],$$
where  $Z = X_1 \sim m^\pi(\cdot \mid S, A)$ ,  $X_t \sim p_{t|Z}(\cdot \mid Z)$ .  
(MC-CFM; 6)

Remarkably, both (MC-FM[; 5\)](#page-2-0) and (MC-[CFM](#page-2-1); 6) share the same gradient and converge to the same solution.

Proposition 1 [\(Lipman et al.](#page-10-3) [2024\)](#page-10-3). *Given a conditional probability path* pt|<sup>Z</sup> *and vector field* ut|<sup>Z</sup> *with their associated marginal counterparts* pt(x) *and* vt(x)*, we have*

$$\nabla_{\theta} \ell_{\text{MC-FM}}(\theta) = \nabla_{\theta} \ell_{\text{MC-CFM}}(\theta).$$

TD-CFM While (MC-[CFM](#page-2-1); 6) requires direct access to samples from the target distribution m<sup>π</sup> , we can instead learn from an offline dataset ρ containing only one-step transitions (S, A, S′ ) through an iterative process similar to [\(4\)](#page-1-1). Starting with initial parameters θ (0), at each iteration, we minimize the TD-Conditional Flow Matching (TD-CFM) loss ℓTD-CFM — an extension of (MC-[CFM](#page-2-1); 6) that differs only in its sampling procedure:

$$Z = X_1 \sim (1-\gamma) \delta_{S'} + \gamma \delta_{\tilde{\psi}_1^{(n)}(X_0 | S', \pi(S'))} \cdot (\text{TD-CFM; } 7)$$

In this procedure, with probability 1 − γ, we return the successor state S ′ . Otherwise, with probability γ we sample from the neural ordinary differential equation [\(Chen et al.,](#page-9-7) [2018\)](#page-9-7) <sup>ψ</sup>e(n) <sup>t</sup> with corresponding vector field v˜ (n) t (X<sup>t</sup> | S ′ , π(S ′ )) from X<sup>0</sup> ∼ p<sup>0</sup> to produce a sample <sup>X</sup><sup>1</sup> <sup>∼</sup> <sup>m</sup>e (n) (· | S ′ , π(S ′ )).

Coupled TD-CFM Although (TD-[CFM](#page-2-2); 7) offers a principled way of learning the flow from noise to data, an increasingly popular strategy to improve flow matching methods is to correlate noise and data whenever a "natural" coupling is available (e.g., [Liu et al.,](#page-10-4) [2023;](#page-10-4) [Shi et al.,](#page-11-10) [2023;](#page-11-10) [Pooladian et al.,](#page-10-5) [2023;](#page-10-5) [Tong et al.,](#page-11-9) [2024;](#page-11-9) [De Bortoli et al.,](#page-9-8) [2024\)](#page-9-8). Motivated by this idea, we observe that the process used to generate X<sup>1</sup> described above already provides a direct coupling between X<sup>0</sup> and X1. We can leverage this coupling by conditioning the probability path pt|<sup>Z</sup> on both endpoints, i.e., Z = (X0, X1), rather than just conditioning on Z = X<sup>1</sup> as in TD-CFM. As illustrated in [Figure 1,](#page-2-3) this coupling helps align X<sup>t</sup> with the path generated by

<sup>ψ</sup>e(n) t , potentially simplifying the regression problem. This procedure gives rise to the Coupled TD-Conditional Flow Matching (TD-CFM(C)) loss ℓTD-CFM(C) which now extends ℓTD-CFM, again, differing only in its sampling procedure:

$$\begin{aligned} X_0 &\sim p_0 \\ X_1 &\sim (1-\gamma)\delta_{S'} + \gamma\delta_{\widetilde{\psi}_1^{(n)}(X_0|_{S'}, \pi(S'))} \\ Z &= (X_0, X_1). \end{aligned} \quad (\text{TD-CFM(C); } 8)$$

A convenient approach to specifying the conditional path pt|<sup>Z</sup> is to define X<sup>t</sup> = ϕt(X0, X1) = αtX<sup>1</sup> + βtX<sup>0</sup> as the affine interpolant between X<sup>0</sup> and X1, with the interpolation coefficients satisfying the boundary conditions α<sup>0</sup> = β<sup>1</sup> = 0, α<sup>1</sup> = β<sup>0</sup> = 1, and monotonicity constraints α˙ <sup>t</sup> > 0, −β˙ <sup>t</sup> > 0, where the over-dot denotes the time derivative. From this definition, the conditional vector field arises as the time derivative of this interpolant defined as <sup>u</sup>t|0,1(X<sup>t</sup> | <sup>X</sup>0, X1) = <sup>ϕ</sup>˙ <sup>t</sup>(X0, X1) = ˙αtX<sup>1</sup> + β˙ <sup>t</sup>X<sup>0</sup> [\(Al](#page-8-2)[bergo et al.,](#page-8-2) [2023\)](#page-8-2). A simple choice of the interpolation coefficients that yields a linear (straight-line) conditional path is given by β<sup>t</sup> = 1 − α<sup>t</sup> = 1 − t.

TD<sup>2</sup> -CFM While (TD-CFM(C[\); 8\)](#page-3-0) improves upon (TD-[CFM](#page-2-2); 7) by accounting for the coupling between bootstrapped samples and their generating noise, both methods rely upon fitting an ad-hoc conditional vector field ut|<sup>Z</sup> that generates the surrogate conditional path pt|Z. To formulate a more structured approach, we exploit the linearity of the Bellman equation, as detailed in the following result.

Lemma 1. *Let* <sup>→</sup> p<sup>t</sup> *be a probability path for* P *generated by vector field* <sup>→</sup> v<sup>t</sup> *and* ↷ p (n) <sup>t</sup> *be a probability path for* P <sup>π</sup>m (n) 1 *generated by* ↷ v (n) t *such that* <sup>→</sup> p<sup>0</sup> = p (n) <sup>0</sup> = m0*. For any* t ∈ [0, 1] *and* (s, a) *let* v (n+1) t (· | s, a) *be the solution of* [<sup>1</sup>](#page-3-1)

$$\begin{aligned} \arg \min_{v: \mathbb{R}^d \rightarrow \mathbb{R}^d} (1 - \gamma) \mathbb{E}_{\vec{X}_t \sim \vec{p}_t(\cdot | s, a)} \left[ \|v(\vec{X}_t) - \vec{v}_t(\vec{X}_t \mid s, a)\|^2 \right] \\ + \gamma \mathbb{E}_{\widehat{X}_t \sim \widehat{p}_t^{(n)}(\cdot | s, a)} \left[ \|v(\widehat{X}_t) - \widehat{v}_t^{(n)}(\widehat{X}_t \mid s, a)\|^2 \right]. \end{aligned}$$

*Then* v (n+1) t *induces a probability path* m (n+1) t *such that* m (n+1) <sup>0</sup> = m<sup>0</sup> *and* m (n+1) <sup>1</sup> = T <sup>π</sup>m (n) 1 *.*

This result shows that it is possible to use two independent probability paths for the two terms in the sampling process induced by the Bellman operator. For the first term, we can use a standard CFM approach for Z = X<sup>1</sup> with conditional path <sup>→</sup> <sup>p</sup>t|<sup>1</sup> and vector field <sup>→</sup> ut|1, which induces the marginal,

$$\vec{v}_t(x|s, a) = \int \vec{u}_{t|1}(x \mid x_1) \frac{\vec{p}_{t|1}(x \mid x_1) P(dx_1|s, a)}{\vec{p}_t(x|s, a)},$$

where <sup>→</sup> pt(x|s, a) = R <sup>→</sup> pt|1(x|s ′ )P(ds ′ |s, a). For the second term, we can leverage the GHM m (n) t learned at the previous iteration to construct the marginal,

$$\widehat{v}_t^{(n)}(x|s, a) = \int v_t^{(n)}(x|s', a') \frac{m_t^{(n)}(x|s', a') P(ds'|s, a)}{\widehat{p}_t^{(n)}(x|s, a)},$$

where ↷ p (n) t (x | s, a) = R m (n) t (x | s ′ , a′ )P(ds ′ | s, a), and a ′ = π(s ′ ). This shows that m (n) <sup>t</sup> plays the role of a conditional probability path for the bootstrapped term and v (n) t is its associated conditional vector field. We can then use the equivalence between FM and CFM in [Proposition 1](#page-2-4) to replace the marginal probability paths and vector fields in [Lemma 1](#page-3-2) with their conditional counterparts to obtain the loss:

$$\begin{aligned} \vec{\ell}(\theta) &= \mathbb{E}_{\rho, t, Z, \vec{X}_t} \left[ \left\| \tilde{v}_t(\vec{X}_t \mid S, A; \theta) - \vec{u}_t|_Z(\vec{X}_t \mid Z) \right\|^2 \right], \\ \text{where } Z &= X_1 \sim P(\cdot \mid S, A), \quad \vec{X}_t \sim \vec{p}_{t|Z}(\cdot \mid Z), \\ \widehat{\ell}(\theta) &= \mathbb{E}_{\rho, t, \widehat{X}_t} \left[ \left\| \tilde{v}_t(\widehat{X}_t \mid S, A; \theta) - \tilde{v}_t^{(n)}(\widehat{X}_t \mid S', \pi(S')) \right\|^2 \right], \\ \text{where } X_0 &\sim p_0, \quad S' \sim P(\cdot \mid S, A), \\ \widehat{X}_t &= \tilde{\psi}_t^{(n)}(X_0 \mid S', \pi(S')), \\ \ell_{\text{TD}^2\text{-CFM}}(\theta) &= (1 - \gamma) \vec{\ell}(\theta) + \gamma \widehat{\ell}(\theta). \quad (\text{TD}^2\text{-CFM}; 9) \end{aligned}$$

$$\ell_{\text{TD}^2\text{-CFM}}(\theta) = (1 - \gamma)\vec{\ell}(\theta) + \gamma\vec{\ell}(\theta). \quad (\text{TD}^2\text{-CFM}; 9)$$

Since we now bootstrap the previous estimate not only in the sampling process but also in the objective function, we refer to this method as TD<sup>2</sup> -Conditional Flow Matching (TD<sup>2</sup> - CFM). The right panel of [Figure 1](#page-2-3) depicts the process of obtaining the bootstrapped vector field v˜ (n) t for TD<sup>2</sup> -CFM. We provide further implementation details and pseudo-code for all TD-Flow methods in [Appendix C.3.1.](#page-19-0) Next, we extend our TD<sup>2</sup> result to the class of denoising diffusion models.

# 3.1. Extension to Diffusion Models

Denoising Diffusion models [\(Sohl-Dickstein et al.,](#page-11-11) [2015;](#page-11-11) [Ho et al.,](#page-9-5) [2020\)](#page-9-5) build a diffusion process starting from a data sample X<sup>0</sup> ∼ q<sup>0</sup> = m<sup>π</sup> (· | S, A) [2](#page-3-3) and corrupting it via a stochastic differential equation (SDE),

$$dX_t = f(t) X_t dt + g(t) dW_t, \quad (10)$$

where t ∈ [0, T] for some time horizon T, f, g : [0, T] → <sup>R</sup> is drift and diffusion term, and W<sup>t</sup> ∈ <sup>R</sup> d is a standard Brownian motion. The forward process of the linear SDE [\(10\)](#page-3-4) has an analytic Gaussian kernel qt|0(·|X0) = N (·|αtX0, σ<sup>2</sup> t I), where α<sup>t</sup> and σ<sup>t</sup> can be computed in closed form. To sample from the target data distribution q0, we can solve the reverse SDE [\(Song & Ermon,](#page-11-12) [2019\)](#page-11-12) from time T to 0:

$$dX_t = \left( f(t) X_t - g(t) \nabla_{X_t} \log q_t(X_t | S, A) \right) dt + g(t) d\bar{W}_t \quad (11)$$

<sup>1</sup>Notice here that the minimization is over the space of all functions and not the parameterized vector fields v˜t(· · · ; θ).

<sup>2</sup>Different to flow matching, time is inverted in diffusion models and ranges from 0 to T.

where W<sup>t</sup> is the reverse-time Brownian motion and q<sup>t</sup> is the marginal distribution of both the forward [\(16\)](#page-13-0) and reverse [\(17\)](#page-13-1) process. To simulate [\(11\)](#page-3-5), we can train a parametrized score function s˜t(x | s, a; θ) to approximate ∇<sup>x</sup><sup>t</sup> log qt(x<sup>t</sup> | s, a) using the denoising diffusion / score matching objective [\(Vincent,](#page-11-13) [2011\)](#page-11-13) ℓDD(θ):

$$\mathbb{E}_{\rho,t,X_0,X_t} \left[ \|\tilde{s}_t(X_t \mid S, A; \theta) - \nabla_{X_t} \log q_{t|0}(X_t \mid X_0)\|^2 \right],$$

$$\text{where } X_0 \sim m^\pi(\cdot \mid S, A), \quad X_t \sim q_{t|0}(\cdot \mid X_0). \quad (\text{DD; 12})$$

Temporal Difference Diffusion Following the blueprint in [§3,](#page-1-2) we define an iterative process starting from s˜ (0) = s˜(· · · ; θ (0)) and minimize at each iteration the Temporal-Difference Denoising Diffusion (TD-DD) loss ℓTD-DD(θ):

$$\mathbb{E}_{\rho, t, X_0, X_t} \left[ \left\| \tilde{s}(X_t \mid S, A; \theta) - \nabla_x \log q_{t|0}(X_t \mid X_0) \right\|^2 \right],$$

$$\text{where } X_0 \sim \left( \mathcal{T}^\pi \tilde{m}_{|_T}^{(n)} \right) (\cdot \mid S, A), X_t \sim q_{t|0}(\cdot \mid X_0).$$
(TD-DD; 13)

Once again, to sample X<sup>0</sup> ∼ T πme (n) 0|T (· | S, A), we proceed as follows: with probability 1−γ, we draw a successor state S ′ ∼ P(· | S, A); conversely, with probability γ, we sample from the bootstrapped model by solving the reverse SDE with score function s˜ (n) , initiated from X<sup>T</sup> . Following an approach analogous to [Lemma 1,](#page-3-2) we demonstrate in [Appendix B](#page-13-2) that we can employ two distinct diffusion processes for the two terms involved in the Bellman operator, which consequently leads to the TD<sup>2</sup> -DD objective:

$$\vec{\ell}(\theta) = \mathbb{E}_{\rho, t, \vec{X}_t} \left[ \left\| \tilde{s}_t(\vec{X}_t | S, A; \theta) - \nabla_{\vec{X}_t} q_{t|0}(\vec{X}_t | S') \right\|^2 \right],$$

$$\text{where } \vec{X}_t \sim q_{t|0}(\cdot \mid S'),$$

$$\begin{aligned} \widehat{\ell}(\theta) &= \mathbb{E}_{\rho, t, \widehat{X}_t} \left[ \left\| \tilde{s}_t(\widehat{X}_t | S, A; \theta) - \tilde{s}_t^{(n)}(\widehat{X}_t | S', \pi(S')) \right\|^2 \right], \\ &\text{where } X_T \sim q_T, \quad \widehat{X}_t \sim q_{t|T}^{(n)}(\cdot | S', \pi(S')), \\ \ell_{\text{TD}^2\text{-DD}}(\theta) &= (1 - \gamma) \vec{\ell}(\theta) + \gamma \widehat{\ell}(\theta). \quad (\text{TD}^2\text{-DD}; 14) \end{aligned}$$

# 4. Theoretical Analysis

We now study the learning dynamics of an idealized version of the TD-Flow methods, assuming that the flow-matching loss is minimized exactly at each iteration. Under this assumption, at each iteration we compute a probability path m (n) t such that m (n) <sup>1</sup> = T <sup>π</sup>m (n−1) 1 , which implies that m (n) <sup>1</sup> → m<sup>π</sup> by the contraction property of T π . The following result shows that the overall probability paths m (n) t follow a similar process. Proofs are deferred to [Appendix E.](#page-31-0)

Theorem 1. *For any* n ≥ 1*, the probability paths generated by* TD-CFM*,* TD-CFM(C)*, or* TD<sup>2</sup> -CFM *satisfy*

$$m_t^{(n+1)}(x \mid s, a) = \left( \mathcal{B}_t^\pi m_t^{(n)} \right)(x \mid s, a), \quad \forall t \in [0, 1]$$

*where* B π <sup>t</sup> m := (1 − γ)P<sup>t</sup> + γP <sup>π</sup>m *and* Pt(x|s, a) := R pt|1(x | x1)P(x1|s, a)dx1*. For any* t ∈ [0, 1]*, the operator* B π t *is a* γ*-contraction in 1-Wasserstein distance, that is, for any couple of probability paths* pt, qt*,*

$$\begin{aligned} \sup_{s,a} W_1 ((\mathcal{B}_t^\pi p_t)(\cdot | s, a), (\mathcal{B}_t^\pi q_t)(\cdot | s, a)) \\ \leq \gamma \sup_{s,a} W_1 (p_t(\cdot | s, a), q_t(\cdot | s, a)). \end{aligned}$$

[Theorem 1](#page-4-0) shows that all TD-flow methods fundamentally implement the same update where the probability path at t ∈ [0, 1] is obtained by applying a Bellman-like operator B<sup>t</sup> to the previous iteration. This operator is a γ-contraction as T π , directly implying the following result.

Corollary 1. *Let* {m (n) <sup>t</sup> }n≥<sup>0</sup> *be the sequence of probability paths produced by* TD-CFM*,* TD-CFM(C)*, or* TD<sup>2</sup> -CFM *starting from an arbitrary vector field* v (0) t *. Then,*

$$\lim_{n \rightarrow \infty} m_t^{(n)} = \bar{m}_t = \mathcal{B}_t \bar{m}_t,$$

*where* m<sup>t</sup> *is the unique fixed point of* Bt*, and* m<sup>t</sup> = mMC t *, where* mMC t (· | s, a) = R pt|1(· | x1) m<sup>π</sup> (x<sup>1</sup> | s, a) *is the probability path of the Monte-Carlo approach* (MC-[CFM](#page-2-1); 6)*.*

This corollary shows that the fixed point of B<sup>t</sup> coincides with the probability path generated in Monte-Carlo Conditional Flow Matching (MC-[CFM](#page-2-1); 6), which assumes direct access to samples of m<sup>π</sup> . An important subtlety in [Theorem 1](#page-4-0) is that all algorithms apply the same operator for n ≥ 1, but the result holds for n = 0 only for TD<sup>2</sup> -CFM. This means that even starting from the same θ (0), the three algorithms may generate different sequences {m (n) <sup>t</sup> }n≥0, while still converging to mt. In Theorems [5](#page-35-0) and [6](#page-37-0) , we show we can reconcile TD-CFM(C) and TD-CFM with TD<sup>2</sup> -CFM under a mild assumption on the form of the initial vector field.

While [Theorem 1](#page-4-0) analyzes an idealized version of the algorithms, in practice gradients are estimated from samples and the following analysis reveals important differences in their variance. We introduce the (unbiased) sample-based gradients for each of the algorithms,

$$\begin{aligned} \mathbb{E}[g_{\text{TD-CFM}}(Y_{\text{TD-CFM}})] &= \nabla_{\theta} \ell_{\text{TD-CFM}}(\theta), \\ \mathbb{E}[g_{\text{TD-CFM}(C)}(Y_{\text{TD-CFM}(C)})] &= \nabla_{\theta} \ell_{\text{TD-CFM}(C)}(\theta) \\ \mathbb{E}[g_{\text{TD}^2\text{-CFM}}(Y_{\text{TD}^2\text{-CFM}})] &= \nabla_{\theta} \ell_{\text{TD}^2\text{-CFM}}(\theta), \end{aligned}$$

where Y summarizes the random variables involved in the loss definitions in (TD-[CFM](#page-2-2); 7), (TD-CFM(C[\); 8\)](#page-3-0), and (TD<sup>2</sup> -[CFM](#page-3-6); 9) (see [Appendix E.6](#page-38-0) for a formal definition of the gradients). We want to compare the total variance of the gradient estimates σ <sup>2</sup> = Tr Cov<sup>Y</sup> [ g(Y ) ] , where Tr denotes the trace.

Theorem 2. *For any* n ≥ 1 *and* t ∈ [0, 1]*, assume that* m (n) t (x | s, a) = R pt|1(x | x1)m (n) 1 (x<sup>1</sup> | s, a)dx1*, then*

$$\sigma_{\text{TD-CFM}}^2 = \sigma_{\text{TD}^2\text{-CFM}}^2 + \gamma^2 \mathbb{E} \left[ \text{Tr} \left( \text{Cov}_{X_1|s,a,X_t} [\nabla_\theta v_t(X_t|s,a;\theta)^\top u_{t|1}(X_t|X_1)] \right) \right].$$

Theorem 3. *For any* n ≥ 1 *and* t ∈ [0, 1]*, assume that* m (n) t (x | s, a) = R pt|0,1(x | x0, x1)m (n) 0,1 (x0, x<sup>1</sup> | s, a)dx0dx<sup>1</sup> [3](#page-5-0) *, then we obtain*

$$\sigma_{\text{TD-CFM}(\text{C})}^2 = \sigma_{\text{TD}^2\text{-CFM}}^2 + \gamma^2 \mathbb{E} \left[ \text{Tr}(\text{Cov}_{\mathbb{Z}|S,A,X_t} [\nabla_{\theta} v_t(X_t|S, A; \theta)^\top u_{t|\mathbb{Z}}(X_t|\mathbb{Z})]) \right],$$

*where* Z = (X0, X1)*. Furthermore, if we use straight conditional paths, i.e.,* X<sup>t</sup> = tX<sup>1</sup> + (1 − t)X0*, and the linear interpolant* X<sup>t</sup> *does not intersect for any* s, a, s′ *, then* σ 2 TD-CFM(C) = σ 2 TD<sup>2</sup> -CFM *.*

In both results, the probability path m (n) t from the previous iteration must be identical for the algorithms being compared. The analysis reveals that TD-CFM and TD-CFM(C) suffer from a larger variance compared to TD<sup>2</sup> -CFM, which uses the vector field v (n) both to sample X<sup>t</sup> and as a target for the regression problem. This variance gap is "discounted" by γ 2 , which suggests that the performance of these algorithms would be similar for problems with small horizons but would increase as γ → 1. The extra variance in both cases stems from samples generated by the algorithm (i.e., they do not depend on the transitions available in the dataset). In this sense, we can refer to it as *computational variance*, and in principle, it could be reduced by increasing the number of samples X0, X1, and X<sup>t</sup> used in gradient computation. While the variance of TD-CFM and TD-CFM(C) cannot be directly compared, we expect that constructing X<sup>t</sup> from X<sup>0</sup> and X<sup>1</sup> (instead of X<sup>1</sup> only) will tend to reduce its variance. Specifically, when X<sup>t</sup> is obtained by linear interpolation between X<sup>0</sup> and X1, and it does not generate crossing paths, the variance of TD-CFM(C) reduces to the one of TD<sup>2</sup> -CFM.

# 5. Experiments

We now present a series of experiments to assess the efficacy of our TD-based flow and diffusion approaches with baselines employing Generative Adversarial Networks [\(Goodfel](#page-9-9)[low et al.,](#page-9-9) [2014\)](#page-9-9) and β-Variational Auto-Encoders [\(Higgins](#page-9-10) [et al.,](#page-9-10) [2017\)](#page-9-10). Following the methodology from [Touati et al.](#page-11-14) [\(2023\)](#page-11-14); [Pirotta et al.](#page-10-6) [\(2024\)](#page-10-6), we benchmark 22 tasks spanning 4 domains (Maze, Walker, Cheetah, Quadruped) from the DeepMind Control Suite [\(Tunyasuvunakool et al.,](#page-11-15) [2020\)](#page-11-15). For a single policy, we evaluate how well each method models its i) successor measure and ii) value function. While lower errors in estimating the successor measure are expected to lead to better value estimation, this is not always the case since modeling errors may disproportionally affect states with negligible rewards. Additionally, motivated by our theoretical results, we explore how the probability path's design affects our proposed methods' relative performance.

Finally, we examine the scalability of our approach by learning a generative model of the successor measure across a class of parameterized policies derived from the Forward-Backward (FB) representation [\(Touati & Ollivier,](#page-11-16) [2021;](#page-11-16) [Touati et al.,](#page-11-14) [2023\)](#page-11-14), a non-generative model of the successor measure. We conclude by demonstrating how TD<sup>2</sup> enables more effective planning for task-relevant policies when performing Generalized Policy Improvement (GPI; [Barreto](#page-8-3) [et al.,](#page-8-3) [2017\)](#page-8-3), far surpassing the capabilities of FB alone.

# 5.1. Empirical Evaluation of Geometric Horizon Models

Before benchmarking, we must first obtain a policy to evaluate. We follow the approach taken in [Thakoor et al.](#page-11-7) [\(2022\)](#page-11-7) and pre-train a set of deterministic policies – one for each task – using TD3 [\(Fujimoto et al.,](#page-9-11) [2018\)](#page-9-11). The final policy obtained from this pre-training phase is now fixed for the remainder of our experiments. GHM training proceeds in an off-policy manner where we learn the successor measure of a TD3 policy using transition data from the ExoRL dataset [\(Yarats et al.,](#page-12-1) [2022\)](#page-12-1); specifically, we use a dataset of 10M transitions collected by a random network distillation policy [\(Burda et al.,](#page-8-4) [2019\)](#page-8-4). All GHM methods are trained for 3M gradient steps using the AdamW optimizer [\(Loshchilov &](#page-10-7) [Hutter,](#page-10-7) [2019\)](#page-10-7) with a batch size of 1024 and weight decay of 0.001. We maintain a target network using an exponential moving average of the training parameters with a step size of 0.001. Special care was taken to match the capacity of the neural networks between methods with a UNet-style architecture employed for all flow and diffusion methods, while the GAN and VAE baselines use an MLP with residual connections for all their respective networks. Full details for the training methodology, network architecture, and hyperparameters can be found in [Appendix C.](#page-17-0)

We implement all conditional flow matching methods (TD-CFM, TD-CFM(C), TD<sup>2</sup> -CFM) with the Optimal Transport Gaussian conditional path from [Lipman et al.](#page-10-2) [\(2023\)](#page-10-2). When constructing our bootstrap targets, we sample from the neural ODE using the Midpoint solver with a constant step size of t/10 for a maximum of 10 steps. For TD<sup>2</sup> -CFM, we sample t ∼ U([0, 1]); otherwise, we integrate to t = 1 and construct X<sup>t</sup> using the conditional path. For Denoising Diffusion methods (TD-DD, TD<sup>2</sup> -DD), we train a DDPM [\(Ho](#page-9-5) [et al.,](#page-9-5) [2020\)](#page-9-5) by discretizing β ∈ (0.1, 20) using T = 1, 000 steps. We construct diffusion bootstrapped targets using

<sup>3</sup>m (n) 0,1 (x0, x1|s, a) = m0(x0)δ ψ (n) (x0|s,a) (x1) is the joint distribution of (X0, X1), *i.e* the endpoints of the ODE.

![](_page_6_Figure_3.jpeg)

Figure 2. Value-Function prediction error as a function of the effective horizon (1 − γ) −1 for γ ∈ {0.8, 0.9, 0.95, 0.98, 0.99} on the POINTMASS loop task. TD<sup>2</sup> methods show impressive robustness to increasingly long-horizon predictions.

20 steps of the DDIM [\(Song et al.,](#page-11-17) [2021a\)](#page-11-17) sampler. For TD-DD, we solve to t = 0 and regress towards the noise that re-corrupted our sample. Alternatively, TD<sup>2</sup> -DD directly regresses towards the noise prediction from the target network at a randomly selected noise level. The first baseline we consider is a GHM instantiated as a Generative Adversarial Network [\(Goodfellow et al.,](#page-9-9) [2014\)](#page-9-9) similar to the one found in [Janner et al.](#page-9-4) [\(2020\)](#page-9-4). We follow the best practices from [Huang et al.](#page-9-12) [\(2024\)](#page-9-12) with the primary modification being a relativistic discriminator [\(Jolicoeur-Martineau,](#page-9-13) [2019\)](#page-9-13) equipped with a zero-centered gradient penalty on both real and fake samples. For our second baseline, we implement a β-VAE [\(Higgins et al.,](#page-9-10) [2017\)](#page-9-10) following the practices outlined in [Thakoor et al.](#page-11-7) [\(2022\)](#page-11-7).

To evaluate the quality of our models, we first generate samples from the ground truth successor measure m<sup>π</sup> according to the following procedure. We first randomly sample 64 source states S<sup>0</sup> from the initial state distribution and execute policy π for 1, 000 steps. Along each trajectory, we resample 2048 states with replacement according to the stopping time t ∼ Geometric(1 − γ). For the same 64 source states, we generate a matching set of 2048 samples from each GHM. Now in possession of these two sets of samples, we evaluate the: 1) log-likelihood of the true samples for models with tractable densities (i.e., diffusion and flow methods); 2) Earth Mover's Distance (EMD; [Rubner et al.,](#page-10-8) [2000\)](#page-10-8), which quantifies the minimal transport cost between the two empirical distributions; and 3) mean-squared error of a Monte-Carlo estimate of the true value function Q<sup>π</sup> and the value function derived from GHM samples using [\(2\)](#page-1-3). Full details can be found in [Appendix C.1.](#page-17-1)

Having established our training framework, baselines, and evaluation protocol, we proceed to investigate a key prediction from our theoretical analysis. Our variance analysis

|            | Method EMD ↓ Norm NLL ↓ MSE(V) ↓                           |
|------------|------------------------------------------------------------|
| C HEETAH   |                                                            |
| TD         | DD 20 22 ( 0 26 ) 2 824 ( 0 195 ) 454 49 ( 131 97 )        |
| TD         | 2                                                          |
|            | DD 14 14 ( 1 08 ) 0 806 ( 0 016 ) 189 15 ( 23 63 )         |
| TD         | CFM 12 26 ( 0 02 ) 0 886 ( 0 024 ) 228 77 ( 2 20 )         |
| TD         | CFM ( C ) 10 51 ( 0 06 ) 0 447 ( 0 020 ) 140 78 ( 18 72 )  |
| TD         | 2                                                          |
|            | CFM 10 57 ( 0 07 ) 0 422 ( 0 014 ) 135 22 ( 19 79 )        |
|            | GAN 23 97 ( 0 46 ) — 2463 22 ( 628 05 )                    |
|            | VAE 83 77 ( 0 41 ) — 1284 27 ( 37 62 )                     |
| P OINTMASS |                                                            |
| TD         | DD 0 149 ( 0 001 ) 2 974 ( 0 100 ) 1245 20 ( 29 27 )       |
| TD         | 2                                                          |
|            | DD 0 027 ( 0 001 ) 0 761 ( 0 082 ) 11 13 ( 3 09 )          |
| TD         | CFM 0 062 ( 0 003 ) 0 554 ( 0 033 ) 355 56 ( 82 83 )       |
| TD         | CFM ( C ) 0 022 ( 0 002 ) − 0 696 ( 0 094 ) 11 89 ( 3 16 ) |
| TD         | 2                                                          |
|            | CFM 0 021 ( 0 000 ) − 0 843 ( 0 027 ) 8 74 ( 2 09 )        |
|            | GAN 0 203 ( 0 037 ) — 1257 26 ( 112 86 )                   |
|            | VAE 0 410 ( 0 036 ) — 1821 89 ( 69 78 )                    |
| Q UADRUPED |                                                            |
| TD         | DD 28 33 ( 0 33 ) 1 908 ( 0 041 ) 1490 75 ( 444 49 )       |
| TD         | 2                                                          |
|            | DD 22 64 ( 2 47 ) 0 861 ( 0 028 ) 159 03 ( 14 64 )         |
| TD         | CFM 15 73 ( 0 06 ) 1 056 ( 0 002 ) 525 06 ( 28 90 )        |
| TD         | CFM ( C ) 14 38 ( 0 03 ) 0 488 ( 0 003 ) 155 25 ( 5 58 )   |
| TD         | 2                                                          |
|            | CFM 14 51 ( 0 05 ) 0 379 ( 0 011 ) 141 77 ( 3 10 )         |
|            | GAN 36772 12 ( 13898 25 ) — 2634 69 ( 798 38 )             |
|            | VAE 60 27 ( 0 28 ) — 1156 33 ( 36 52 )                     |
| W ALKER    |                                                            |
| TD         | DD 20 58 ( 0 24 ) 2 649 ( 0 137 ) 382 40 ( 458 63 )        |
| TD         | 2                                                          |
|            | DD 12 09 ( 0 12 ) 0 537 ( 0 060 ) 39 04 ( 6 08 )           |
| TD         | CFM 13 53 ( 0 11 ) 0 713 ( 0 028 ) 225 27 ( 42 43 )        |
| TD         | CFM ( C ) 11 91 ( 0 02 ) 0 219 ( 0 016 ) 30 71 ( 3 44 )    |
| TD         | 2                                                          |
|            | CFM 11 92 ( 0 10 ) 0 104 ( 0 001 ) 28 35 ( 6 10 )          |
|            | GAN 24 51 ( 0 89 ) — 3690 65 ( 1117 94 )                   |
|            | VAE 111 73 ( 2 53 ) — 2457 61 ( 16 25 )                    |

Table 1. Evaluation results comparing our TD-based methods along with GAN and VAE baselines for a single-policy. Results are computed over 19 tasks from 4 domains and further averaged across 3 seeds. For each metric we highlight the best performing methods.

| Method               | EMD ↓         | Norm NLL ↓   | MSE(V) ↓       |
|----------------------|---------------|--------------|----------------|
| TD-CFM(Q)            | 14.08 (12.42) | 7.19 (1.98)  | 31.04 (258.94) |
| TD <sup>2</sup> -CFM | 0.09 (0.09)   | −0.01 (0.04) | −3.36 (7.76)   |

Table 2. Performance difference for TD-CFM(C) and TD<sup>2</sup> -CFM when employing a curved instead of straight conditional path. Lower is better with negative values indicating a net improvement for using a curved path.

suggests that our TD-Flow framework should enable more stable training across extended temporal horizons. To validate this hypothesis, we train each GHM for 3 seeds on the loop task in the Maze domain while varying the effective horizon (1 − γ) −1 across five values: {5, 10, 20, 50, 100}. [Figure 2](#page-6-0) illustrates the relationship between value function MSE and the effective horizon. The results demonstrate that TD<sup>2</sup> -based methods maintain consistent performance even as the effective horizon increases, while alternative approaches show significant performance degradation. Notably, at an effective horizon of 100, TD<sup>2</sup> -based methods maintain their accuracy and achieve performance improvements of nearly four orders of magnitude compared to their naive implementations. These results empirically support for our initial hypothesis, with the stability of TD<sup>2</sup> methods aligning with our predictions.

In the following, we shift our attention to a more in-depth analysis of the largest horizon of 100 (γ = 0.99). For each

![](_page_7_Figure_1.jpeg)

Figure 3. Performance improvement over the zero-shot Forward Backward (FB; [Touati & Ollivier,](#page-11-16) [2021\)](#page-11-16) policies when planning with Generalized Policy Improvement (GPI; [Barreto et al.,](#page-8-3) [2017\)](#page-8-3). FB-GPI performs GPI over the FB value-function Q <sup>π</sup><sup>w</sup> . DD-GPI and FM-GPI perform GPI with the value function implied by the GHM m<sup>π</sup><sup>w</sup> for our diffusion-based and flow-based methods, respectively. Results are averaged over 22 tasks across 4 domains.

algorithm, we train a GHM for 3 independent seeds for all domains and tasks. [Table 1](#page-6-1) reports aggregate performance across our full suite of metrics. For each domain and metric, we highlight results in a 1% range with respect to the bestperforming method. The results demonstrate a clear pattern of superior performance for TD<sup>2</sup> -based algorithms: TD<sup>2</sup> - CFM achieves significant improvements over TD-CFM with a 10× reduction in value-function MSE, 1.5× reduction in EMD, and 3× reduction in log-likelihood, averaged across all four domains. In line with our theoretical predictions, the coupled variant of TD-CFM performs comparably to TD<sup>2</sup> - CFM, given straight conditional paths. While a comparison between flow matching and diffusion is not at the core of this paper, in our experiments, flow matching generally outperforms diffusion across all metrics. We posit this is primarily due to noise in the diffusion process adversely impacting an already noisy prediction problem for large horizons.

Given the comparable performance between TD-CFM(C) and TD<sup>2</sup> -CFM with straight conditional paths, we next examine how these methods behave with alternative path geometries. Our theoretical analysis suggests an important distinction: TD<sup>2</sup> -CFM should maintain its effectiveness with non-straight paths, while the performance of TD-CFM(C) should degrade. To test this prediction, we maintain the methodology above while replacing conditional path in (TD<sup>2</sup> -[CFM](#page-3-6); 9) with the following curved path pt|1(· | X1) = N (· | αtX1, β<sup>2</sup> t ) with coefficients α<sup>t</sup> = sin π 2 t and β<sup>t</sup> = cos π 2 t . The corresponding conditional vector field is now given by ut|1(Xt|X1) = α˙ <sup>t</sup> − α<sup>t</sup> β<sup>t</sup> X<sup>1</sup> + β˙ β<sup>t</sup> Xt. Additionally, for TD-CFM(C) we condition the curved path above on X<sup>0</sup> and X<sup>1</sup> resulting in the conditional vector field

ut|0,1(X<sup>t</sup> | X0, X1) = <sup>π</sup> 2 βtX<sup>1</sup> − αtX<sup>0</sup> . [Table 2](#page-6-2) illustrates the performance difference relative to the straight path results [\(Table 1\)](#page-6-1) averaged across all domains and tasks. The results strongly support our theoretical prediction: TD<sup>2</sup> -CFM not only maintained but surprisingly improved performance compared to the linear path. In contrast, TD-CFM(C) showed significant performance degradation, confirming our hypothesis about its limitations with non-straight paths.

#### 5.2. Planning via Generalized Policy Improvement

We now turn our attention towards training policyconditioned GHMs which can be utilized for test-time planning. To accomplish this, we first pre-train a Forward Backward (FB; [Touati & Ollivier,](#page-11-16) [2021;](#page-11-16) [Touati et al.,](#page-11-14) [2023\)](#page-11-14) representation using the same dataset of 10M transitions as described in [§5.1.](#page-5-1) This pre-training yields a class of wconditioned policies πw, where each w ∈ W = S d−1 ( √ d) represents an embedding of a reward function situated on a <sup>d</sup>-dimensional hypersphere with radius √ d. We then train the GHM m<sup>π</sup><sup>w</sup> conditioned on the policy by incorporating the embedding w directly into the model's input. All GHM methods are trained for 8M gradient steps, maintaining the same parameters used in [§5.1,](#page-5-1) with the exception of a higher weight decay coefficient of 0.01. For additional insights into the accuracy of the policy-conditioned GHMs, we direct the reader to [Appendix D.](#page-24-0) Overall, we observed similar trends to those seen in our single-policy experiments.

Given that both FB and w-conditioned GHM models enable estimation of a policy's value function Q<sup>π</sup><sup>w</sup> , we can utilize this information to perform Generalized Policy Improvement (GPI; [Barreto et al.,](#page-8-3) [2017\)](#page-8-3) during evaluation. Specifically, at each time step t, we choose an action a<sup>t</sup> = π<sup>w</sup><sup>t</sup> (st), where w<sup>t</sup> is derived as follows:

$$w_t \in \arg\max_{w \sim D(W)} \underbrace{(1 - \gamma)^{-1} \mathbb{E}_{X \sim m^{\pi w}(\cdot | s_t, \pi_w(s_t))} [r(X)]}_{Q^{\pi w}(s_t, \pi_w(s_t))}. \quad (15)$$

Here D(W) is a sampling distribution over W. We consider three such distributions: *i) Random*: uniform distribution over W; *ii) Local Perturbation*: we perturb the embedding w<sup>r</sup> of the task reward r by the uniform distribution; *iii) Train Distribution*: we sample w from the training distribution used by FB. To approximate [\(15\)](#page-8-5), we sample 255 embeddings from D(W) and explicitly include the task embedding wr, resulting in a maximization over 256 policies. To estimate Q<sup>π</sup><sup>w</sup> , we average the reward over 128 states sampled from m<sup>π</sup><sup>w</sup> . Performance is measured by averaging returns over 100 episodes, each lasting 1000 steps.

[Figure 3](#page-7-0) illustrates the average percentage of improvement for each algorithm and w-sampling strategy relative to the performance of the FB policy π<sup>w</sup><sup>r</sup> for the task reward r. We refer to [Appendix D](#page-24-0) for a more detailed view of these results. All TD-based GHM approaches lead to a significant improvement over the base FB policy, with TD-CFM(C) and TD<sup>2</sup> -CFM providing ≈ 30%+ improvement with all sampling approaches. TD<sup>2</sup> -DD also leads to significant performance gains but is still dominated by the flow matching methods. Notably, FB-based GPI not only fails to improve performance but actually deteriorates it on average with significant degradation observed in three out of four domains (detailed results available in [Appendix D\)](#page-24-0). When comparing different distributions D(W), we observe that while FB-GPI's performance fluctuates considerably, GHM methods maintain their robustness across distributions, showing only minor variation. These results underscore the ability of our improved GHMs to make long-term predictions enabling powerful planning capabilities.

# 6. Discussion

In this paper, we introduced temporal difference flows, a novel generative modeling approach that significantly advances long-horizon predictive models of state. By leveraging the successor measure's temporal difference structure both in its sampling procedure and learning objective, TD<sup>2</sup> - CFM and TD<sup>2</sup> -DD effectively address challenges associated with modeling long-range state dynamics. The methods developed in this paper provide a robust theoretical and empirical foundation that demonstrates the advantages of our framework across a range of tasks, metrics, and domains. We envision numerous exciting applications emerging from this work, particularly around imitation learning [\(Wu et al.,](#page-12-2) [2025;](#page-12-2) [Jain et al.,](#page-9-14) [2025\)](#page-9-14), planning [\(Sutton,](#page-11-18) [1991;](#page-11-18) [Thakoor](#page-11-7) [et al.,](#page-11-7) [2022;](#page-11-7) [Zhu et al.,](#page-12-3) [2024\)](#page-12-3), and off-policy evaluation [\(Pre](#page-10-9)[cup et al.,](#page-10-9) [2000;](#page-10-9) [2001;](#page-10-10) [Nachum et al.,](#page-10-11) [2019;](#page-10-11) [Fujimoto et al.,](#page-9-15)

[2021\)](#page-9-15). Furthermore, recent work on consistency models [\(Song et al.,](#page-11-19) [2023;](#page-11-19) [Yang et al.,](#page-12-4) [2024\)](#page-12-4) and self-distillation [\(Frans et al.,](#page-9-16) [2025\)](#page-9-16) suggests promising avenues for tackling the computational burden of sampling — a limitation common to the family of iterative generative models that our approach builds upon.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References


[1] Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In *International Conference on Learning Representations, (ICLR)*, 2023. Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E. Stochastic interpolants: A unifying framework for flows and diffusions. *CoRR*, abs/2303.08797, 2023. Anderson, B. D. Reverse-time diffusion equation models. *Stochastic Processes and their Applications*, 12(3):313– 326, 1982. Ba, J., Kiros, J., and Hinton, G. E. Layer normalization. *CoRR*, abs/1607.06450, 2016. Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T., Silver, D., and van Hasselt, H. Successor features for transfer in reinforcement learning. In *Neural Information Processing Systems (NeurIPS)*, 2017. Barreto, A., Hou, S., Borsa, D., Silver, D., and Precup, D. Fast reinforcement learning with generalized policy updates. *Proceedings of the National Academy of Sciences (PNAS)*, 117(48):30079–30087, 2020. Blier, L., Tallec, C., and Ollivier, Y. Learning successor states and goal-dependent values: A mathematical viewpoint. *CoRR*, abs/2101.07123, 2021. Borsa, D., Barreto, A., Quan, J., Mankowitz, D. J., van Hasselt, H., Munos, R., Silver, D., and Schaul, T. Universal successor features approximators. In *International Conference on Learning Representations (ICLR)*, 2019. Burda, Y., Edwards, H., Storkey, A. J., and Klimov, O. Exploration by random network distillation. In *International Conference on Learning Representations (ICLR)*, 2019. Cetin, E., Touati, A., and Ollivier, Y. Finer behavioral foundation models via auto-regressive features and advantage weighting. *CoRR*, abs/2412.04368, 2024.

[2] Chen, T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud,

[3] D. Neural ordinary differential equations. In *Neural Information Processing Systems (NeurIPS)*, 2018. Dayan, P. Improving generalization for temporal difference learning: The successor representation. *Neural Computation*, 1993. De Bortoli, V., Korshunova, I., Mnih, A., and Doucet, A. Schrodinger bridge flow for unpaired data translation. In ¨ *Neural Information Processing Systems (NeurIPS)*, 2024. Dinh, L., Krueger, D., and Bengio, Y. NICE: Non-linear independent components estimation. In *International Conference on Learning Representations (ICLR), Workshop Track Proceedings*, 2015. Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real nvp. In *International Conference on Learning Representations (ICLR)*, 2017. Farebrother, J., Greaves, J., Agarwal, R., Le Lan, C., Goroshin, R., Castro, P. S., and Bellemare, M. G. Protovalue networks: Scaling representation learning with auxiliary tasks. In *International Conference on Learning Representations (ICLR)*, 2023. Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., Gautheron, L., Gayraud, N. T., Janati, H., Rakotomamonjy, A., Redko, I., Rolet, A., Schutz, A., Seguy, V., Sutherland, D. J., Tavenard, R., Tong, A., and Vayer, T. Pot: Python optimal transport. *Journal of Machine Learning Research*, 22(78):1–8, 2021. Frans, K., Hafner, D., Levine, S., and Abbeel, P. One step diffusion via shortcut models. In *International Conference on Learning Representations (ICLR)*, 2025. Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. In *International Conference on Machine Learning (ICML)*, 2018. Fujimoto, S., Meger, D., and Precup, D. A deep reinforcement learning approach to marginalized importance sampling with the successor representation. In *International Conference on Machine Learning (ICML)*, 2021. Ghosh, D., Bhateja, C. A., and Levine, S. Reinforcement learning from passive data via latent intentions. In *International Conference on Machine Learning (ICML)*, 2023. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio,

[4] Y. Generative adversarial nets. In *Neural Information Processing Systems (NeurIPS)*, 2014. Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., and Duvenaud, D. FFJORD: free-form continuous dynamics for scalable reversible generative models. In *International Conference on Learning Representations (ICLR)*, 2019. Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. P. Mastering diverse domains through world models. *CoRR*, abs/2301.04104, 2023. Hansen, N., Su, H., and Wang, X. Temporal difference learning for model predictive control. In *International Conference on Machine Learning (ICML)*, 2022. Hansen, N., Su, H., and Wang, X. TD-MPC2: Scalable, robust world models for continuous control. In *International Conference on Learning Representations (ICLR)*, 2024. Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. betavae: Learning basic visual concepts with a constrained variational framework. In *International Conference on Learning Representations (ICLR)*, 2017. Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In *Neural Information Processing Systems (NeurIPS)*, 2020. Huang, N., Gokaslan, A., Kuleshov, V., and Tompkin, J. The gan is dead; long live the gan! a modern gan baseline. In *Neural Information Processing Systems (NeurIPS)*, 2024. Jafferjee, T., Imani, E., Talvitie, E., White, M., and Bowling, M. Hallucinating value: A pitfall of dyna-style planning with imperfect environment models. *CoRR*, abs/2006.04363, 2020. Jain, A. K., Lehnert, L., Rish, I., and Berseth, G. Maximum state entropy exploration using predecessor and successor representations. In *Neural Information Processing Systems (NeurIPS)*, 2023. Jain, A. K., Wiltzer, H., Farebrother, J., Rish, I., Berseth, G., and Choudhury, S. Non-adversarial inverse reinforcement learning via successor feature matching. In *International Conference on Learning Representations (ICLR)*, 2025. Janner, M., Mordatch, I., and Levine, S. Gamma-models: Generative temporal difference learning for infinitehorizon prediction. In *Neural Information Processing Systems (NeurIPS)*, 2020. Jolicoeur-Martineau, A. The relativistic discriminator: a key element missing from standard gan. In *International Conference on Learning Representations (ICLR)*, 2019.

[5] Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*, 2015. Kingma, D. P. and Welling, M. Auto-encoding variational bayes. In *International Conference on Learning Representations (ICLR)*, 2014. Lambert, N., Pister, K., and Calandra, R. Investigating compounding prediction errors in learned dynamics models. *CoRR*, abs/2203.09637, 2022. Le Lan, C., Tu, S., Oberman, A., Agarwal, R., and Bellemare, M. G. On the generalization of representations in reinforcement learning. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2022. Le Lan, C., Greaves, J., Farebrother, J., Rowland, M., Pedregosa, F., Agarwal, R., and Bellemare, M. G. A novel stochastic gradient descent algorithm for learning principal subspaces. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2023a. Le Lan, C., Tu, S., Rowland, M., Harutyunyan, A., Agarwal, R., Bellemare, M. G., and Dabney, W. Bootstrapped representations in reinforcement learning. In *International Conference on Machine Learning (ICML)*, 2023b. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In *International Conference on Learning Representations (ICLR)*, 2023. Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R. T. Q., Lopez-Paz, D., Ben-Hamu, H., and Gat, I. Flow matching guide and code. *CoRR*, abs/2412.06264, 2024. Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. In *International Conference on Learning Representations (ICLR)*, 2023. Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*, 2019. Machado, M. C., Rosenbaum, C., Guo, X., Liu, M., Tesauro, G., and Campbell, M. Eigenoption discovery through the deep successor representation. In *International Conference on Learning Representations (ICLR)*, 2018. Machado, M. C., Bellemare, M. G., and Bowling, M. Countbased exploration with the successor representation. In *AAAI Conference on Artificial Intelligence*, 2020. Machado, M. C., Barreto, A., Precup, D., and Bowling, M. Temporal abstraction in reinforcement learning with the successor representation. *Journal of Machine Learning Research (JMLR)*, 24:80:1–80:69, 2023. Misra, D. Mish: A self regularized non-monotonic neural activation function. *CoRR*, abs/1908.08681, 2019. Nachum, O., Chow, Y., Dai, B., and Li, L. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. In *Neural Information Processing Systems (NeurIPS)*, 2019. Park, S., Kreiman, T., and Levine, S. Foundation policies with hilbert representations. In *International Conference on Machine Learning (ICML)*, 2024. Pathak, D., Agrawal, P., Efros, A. A., and Darrell, T. Curiosity-driven exploration by self-supervised prediction. In *International Conference on Machine Learning (ICML)*, 2017. Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. C. Film: Visual reasoning with a general conditioning layer. In *AAAI Conference on Artificial Intelligence*, 2018. Pirotta, M., Tirinzoni, A., Touati, A., Lazaric, A., and Ollivier, Y. Fast imitation via behavior foundation models. In *International Conference on Learning Representations (ICLR)*, 2024. Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., and Chen, R. T. Q. Multisample flow matching: Straightening flows with minibatch couplings. In *International Conference on Machine Learning (ICML)*, 2023. Precup, D., Sutton, R. S., and Singh, S. Eligibility traces for off-policy policy evaluation. In *International Conference on Machine Learning (ICML)*, 2000. Precup, D., Sutton, R. S., and Dasgupta, S. Off-policy temporal difference learning with function approximation. In *International Conference on Machine Learning (ICML)*, 2001. Rezende, D. and Mohamed, S. Variational inference with normalizing flows. In *International Conference on Machine Learning (ICML)*, 2015. Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, volume 9351, pp. 234–241, 2015. Rubner, Y., Tomasi, C., and Guibas, L. J. The earth mover's distance as a metric for image retrieval. *International Journal of Computer Vision*, 40(2):99–121, 2000.

[6] Schmidhuber, J. A possibility for implementing curiosity and boredom in model-building neural controllers. In *International Conference on Simulation of Adaptive Behavior*, 1991. Schramm, L. and Boularias, A. Bellman diffusion models. *CoRR*, abs/2407.12163, 2024. Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., and Silver, D. Mastering atari, go, chess and shogi by planning with a learned model. *Nature*, 588(7839):604–609, 2020. Shi, Y., De Bortoli, V., Campbell, A., and Doucet, A. Diffusion schrodinger bridge matching. In ¨ *Neural Information Processing Systems (NeurIPS)*, 2023. Sikchi, H., Zhou, W., and Held, D. Learning off-policy with online planning. In *Conference on Robot Learning (CoRL)*, 2021. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap,

[7] T. P., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. Mastering the game of go with deep neural networks and tree search. *Nature*, 529(7587):484–489, 2016. Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., and Hassabis, D. Mastering the game of go without human knowledge. *Nature*, 550(7676):354–359, 2017. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In *International Conference on Machine Learning (ICML)*, 2015. Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In *International Conference on Learning Representations (ICLR)*, 2021a. Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. In *Neural Information Processing Systems (NeurIPS)*, 2019. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations (ICLR)*, 2021b. Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In *International Conference on Machine Learning (ICML)*, 2023. Stadie, B. C., Levine, S., and Abbeel, P. Incentivizing exploration in reinforcement learning with deep predictive models. In *International Conference on Learning Representations (ICLR)*, 2016. Sutton, R. S. Dyna, an integrated architecture for learning, planning, and reacting. *ACM SIGART*, 2(4):160–163, 1991. Talvitie, E. Model regularization for stable sample rollouts. In *Conference on Uncertainty in Artificial Intelligence (UAI)*, 2014. Thakoor, S., Rowland, M., Borsa, D., Dabney, W., Munos, R., and Barreto, A. Generalised policy improvement with geometric policy composition. In *International Conference on Machine Learning (ICML)*, 2022. Tirinzoni, A., Touati, A., Farebrother, J., Guzek, M., Kanervisto, A., Xu, Y., Lazaric, A., and Pirotta, M. Zero-shot whole-body humanoid control via behavioral foundation models. In *International Conference on Learning Representations (ICLR)*, 2025. Tomar, M., Hansen-Estruch, P., Bachman, P., Lamb, A., Langford, J., Taylor, M. E., and Levine, S. Video occupancy models. *CoRR*, abs/2407.09533, 2024. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., and Bengio, Y. Improving and generalizing flow-based generative models with minibatch optimal transport. In *Transactions on Machine Learning Research (TMLR)*, 2024. Touati, A. and Ollivier, Y. Learning one representation to optimize all rewards. In *Neural Information Processing Systems (NeurIPS)*, 2021. Touati, A., Rapin, J., and Ollivier, Y. Does zero-shot reinforcement learning exist? In *International Conference on Learning Representations (ICLR)*, 2023. Tunyasuvunakool, S., Muldal, A., Doron, Y., Liu, S., Bohez, S., Merel, J., Erez, T., Lillicrap, T., Heess, N., and Tassa,
  - Y. dm control: Software and tasks for continuous control. *Software Impacts*, 6:100022, 2020. van den Oord, A., Vinyals, O., and Kavukcuoglu, K. Neural discrete representation learning. In *Neural Information Processing Systems (NeurIPS)*, 2017. Vincent, P. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661– 1674, 2011. Wiltzer, H., Farebrother, J., Gretton, A., and Rowland, M. Foundations of multivariate distributional reinforcement learning. In *Neural Information Processing Systems (NeurIPS)*, 2024a.

[8] Wiltzer, H., Farebrother, J., Gretton, A., Tang, Y., Barreto, A., Dabney, W., Bellemare, M. G., and Rowland, M. A distributional analogue to the successor representation. In *International Conference on Machine Learning (ICML)*, 2024b. Wu, R., Chen, Y., Swamy, G., Brantley, K., and Sun, W. Diffusing states and matching scores: A new framework for imitation learning. In *International Conference on Learning Representations (ICLR)*, 2025. Yang, L., Zhang, Z., Zhang, Z., Liu, X., Xu, M., Zhang, W., Meng, C., Ermon, S., and Cui, B. Consistency flow matching: Defining straight flows with velocity consistency. *CoRR*, abs/2407.02398, 2024. Yarats, D., Brandfonbrener, D., Liu, H., Laskin, M., Abbeel, P., Lazaric, A., and Pinto, L. Don't change the algorithm, change the data: Exploratory data for offline reinforcement learning. *CoRR*, abs/2201.13425, 2022. Zhang, P., Chen, X., Zhao, L., Xiong, W., Qin, T., and Liu, T.-Y. Distributional reinforcement learning for multidimensional reward functions. In *Neural Information Processing Systems (NeurIPS)*, 2021. Zhu, C., Wang, X., Han, T., Du, S. S., and Gupta, A. Distributional successor features enable zero-shot policy optimization. In *Neural Information Processing Systems (NeurIPS)*, 2024.
# Appendices

# A. Related Work

The Successor Representation [\(Dayan,](#page-9-6) [1993\)](#page-9-6) was originally proposed for tabular MDPs and was later generalized to continuous state spaces with the Successor Measure [\(Blier et al.,](#page-8-0) [2021\)](#page-8-0). Successor Features [\(Barreto et al.,](#page-8-3) [2017;](#page-8-3) [2020\)](#page-8-6) extends these ideas by instead modeling the evolution of multi-dimensional features assuming rewards decompose linearly over these features. Prior works have leveraged these methods for zero-shot policy evaluation [\(Dayan,](#page-9-6) [1993;](#page-9-6) [Barreto et al.,](#page-8-3) [2017;](#page-8-3) [Wiltzer et al.,](#page-12-0) [2024b\)](#page-12-0), zero-shot policy optimization [\(Borsa et al.,](#page-8-7) [2019;](#page-8-7) [Touati & Ollivier,](#page-11-16) [2021;](#page-11-16) [Touati et al.,](#page-11-14) [2023;](#page-11-14) [Park et al.,](#page-10-12) [2024;](#page-10-12) [Zhu et al.,](#page-12-3) [2024;](#page-12-3) [Cetin et al.,](#page-8-8) [2024;](#page-8-8) [Tirinzoni et al.,](#page-11-20) [2025\)](#page-11-20), imitation learning [\(Pirotta et al.,](#page-10-6) [2024;](#page-10-6) [Jain](#page-9-14) [et al.,](#page-9-14) [2025\)](#page-9-14), exploration [\(Machado et al.,](#page-10-13) [2020;](#page-10-13) [Jain et al.,](#page-9-17) [2023\)](#page-9-17), representation learning [\(Le Lan et al.,](#page-10-14) [2022;](#page-10-14) [2023a;](#page-10-15)[b;](#page-10-16) [Farebrother et al.,](#page-9-18) [2023;](#page-9-18) [Ghosh et al.,](#page-9-19) [2023\)](#page-9-19), and building temporal abstractions [\(Machado et al.,](#page-10-17) [2018;](#page-10-17) [2023\)](#page-10-18).

[Janner et al.](#page-9-4) [\(2020\)](#page-9-4) originally proposed a method to learn a generative model of the successor measure with modeling techniques spanning from Generative Adversarial Networks [\(Goodfellow et al.,](#page-9-9) [2014\)](#page-9-9) to Normalizing Flows [\(Dinh et al.,](#page-9-20) [2015;](#page-9-20) [Rezende & Mohamed,](#page-10-19) [2015\)](#page-10-19) like RealNVP [\(Dinh et al.,](#page-9-21) [2017\)](#page-9-21). Followup work (e.g., [Thakoor et al.,](#page-11-7) [2022;](#page-11-7) [Tomar](#page-11-8) [et al.,](#page-11-8) [2024\)](#page-11-8) explored other generative modeling techniques including various types of auto-encoders (e.g., [Higgins et al.,](#page-9-10) [2017;](#page-9-10) [van den Oord et al.,](#page-11-21) [2017\)](#page-11-21). Also of note is recent work learning generative models of multi-dimensional cumulants including features [\(Wiltzer et al.,](#page-11-22) [2024a;](#page-11-22) [Zhu et al.,](#page-12-3) [2024\)](#page-12-3) and multi-variate reward functions [\(Zhang et al.,](#page-12-5) [2021\)](#page-12-5). Prior work by [Wiltzer et al.](#page-12-0) [\(2024b\)](#page-12-0) sought to deal with the instability of long-horizon predictions in GHMs by employing an n-step mixture distribution where they sample t ∼ Geometric(1 − γ) and bootstrap if t > n; otherwise returning the state at time t along the trajectory. Without resorting to importance sampling this approach is limited to the on-policy setting. Finally, most closely related to our work is that of [Schramm & Boularias](#page-11-23) [\(2024\)](#page-11-23) who provide a preliminary and limited derivation of what we term TD<sup>2</sup> -DD. In contrast, our work not only rigorously formalizes and significantly extends these ideas but also integrates them into the more general flow-matching framework [\(Lipman et al.,](#page-10-2) [2023;](#page-10-2) [2024\)](#page-10-3), additionally incorporating extensions to score-matching [\(Song et al.,](#page-11-24) [2021b;b\)](#page-11-24) and diffusion [\(Sohl-Dickstein et al.,](#page-11-11) [2015;](#page-11-11) [Ho et al.,](#page-9-5) [2020\)](#page-9-5). Moreover, we conduct an extensive empirical analysis, demonstrating the efficacy of our approach — an aspect notably absent from [Schramm & Boularias](#page-11-23) [\(2024\)](#page-11-23).

# B. Extension to Score Matching and Diffusion Models

This section extends our framework to score matching and denoising diffusion models. We leverage the unification of these methods under stochastic differential equations [\(Song et al.,](#page-11-24) [2021b\)](#page-11-24) introducing an analogous class of Temporal Difference Diffusion methods.

# B.1. Background

Both score-based generative modeling [\(Song & Ermon,](#page-11-12) [2019\)](#page-11-12) and diffusion probabilistic modeling [\(Sohl-Dickstein et al.,](#page-11-11) [2015;](#page-11-11) [Ho et al.,](#page-9-5) [2020\)](#page-9-5) can be unified under the framework of stochastic differential equations (SDE) introduced in [Song](#page-11-24) [et al.](#page-11-24) [\(2021b\)](#page-11-24). Unlike in flow-matching, time is inverted in diffusion models and ranges from time 0 to T. Given the data distribution q<sup>0</sup> and prior simple distribution q<sup>T</sup> (the "noise" distribution), we construct a diffusion process {Xt}t∈[0,T] such that X<sup>0</sup> ∼ q<sup>0</sup> and X<sup>T</sup> ∼ q<sup>T</sup> . This diffusion can be modeled as the solution to an Ito SDE:

$$dX_t = f(t) X_t dt + g(t) dW_t \quad | \quad X_0 \sim q_0, \quad (16)$$

where W<sup>t</sup> is a standard Brownian motion and f : [0, T] → <sup>R</sup> d is scalar function called the drift coefficient, and g : [0, T] → R is scalar function known as diffusion coefficient.

Generating samples from X<sup>0</sup> ∼ q<sup>0</sup> consists in sampling X<sup>T</sup> ∼ q<sup>T</sup> and reversing the forward-SDE process in [\(16\)](#page-13-0). A known result from [Anderson](#page-8-9) [\(1982\)](#page-8-9) states that the reverse of a diffusion process is also a diffusion process, running backward in time and given by the reverse-time SDE:

$$dX_t = \left( f(t) X_t - g(t)^2 \nabla_{X_t} \log q_t(X_t) \right) dt + g(t) d\bar{W}_t \mid X_T \sim q_T, \quad (17)$$

the marginal distribution of Xt. Therefore, once we learn the score of the marginal distribution ∇<sup>x</sup> log qt(x), we can sample from q<sup>0</sup> by simulating the reverse diffusion process [\(17\)](#page-13-1).

To estimate ∇<sup>x</sup> log qt(x), we can train a time-dependent score-based model s˜(· · · ; θ) : [0, T] × <sup>R</sup> <sup>d</sup> → <sup>R</sup> <sup>d</sup> via the denoising diffusion / score matching objective [\(Vincent,](#page-11-13) [2011;](#page-11-13) [Song & Ermon,](#page-11-12) [2019\)](#page-11-12):

$$\ell_{\text{DD}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}([0,1]), X_0 \sim q_0} \mathbb{E}_{X_t \sim q_{t|0}(\cdot | X_0)} \left[ \left\| \tilde{s}_t(X_t; \theta) - \nabla_{X_t} \log q_{t|0}(X_t \mid X_0) \right\|^2 \right]. \quad (18)$$

For ℓDD to be tractable, we need to know the conditional probability qt|0. Usually, specific choices of the drift and diffusion coefficients f<sup>t</sup> and g<sup>t</sup> are used such that qt|<sup>0</sup> is always a Gaussian distribution N (· | αtx0, σ<sup>2</sup> t ), where the mean α<sup>t</sup> and variance σ 2 t can be computed in closed-form. The global minimizer of ℓDD(θ) denoted by s ⋆ t (x) is equal to the score function ∇<sup>x</sup> log qt(x), thanks to the following proposition:

Proposition 2 [\(Vincent](#page-11-13) [2011\)](#page-11-13). *Let* qt(x) = R q0(x0)qt|0(x|x0) dx0*, then we have:*

$$\nabla \theta \ell_{\text{DD}}(\theta) = \nabla_{\theta} \mathbb{E}_{t, X_t \sim q_{\theta}} \left[ \|\tilde{s}_t(X_t; \theta) - \nabla_{X_t} \log q_t(X_t)\|^2 \right]. \quad (19)$$

# B.2. Temporal Difference Diffusion

To learn a predictive model of m<sup>π</sup> using diffusion from an offline dataset, we follow a similar approach to what we presented in [§3](#page-1-2) and we define an iterative process starting from initial weights θ (0) and at each iteration minimizing the Temporal-Difference Denoising Diffusion (TD-DD) loss:

$$\begin{aligned} \ell_{\text{TD-DD}}(\theta) &= \mathbb{E}_{\rho, t, X_0, X_t} \left[ \|\tilde{s}_t(X_t \mid S, A; \theta) - \nabla_x \log q_{t|0}(X_t \mid X_1)\|^2 \right], \\ \text{where, } X_0 &\sim \left( \mathcal{T}^\pi \tilde{m}_{|0|T}^{(n)} \right) (\cdot \mid S, A), \quad X_t \sim q_{t|0}(\cdot \mid X_0). \end{aligned} \quad (\text{TD-DD; } 20)$$

In order to sample X<sup>0</sup> ∼ T πme (n) 0|T (· | s, a), with probability 1 − γ, we return the successor state S ′ ∼ P(· | S, A). Otherwise, with probability γ we solve the following reverse-time SDE from X<sup>T</sup> using the score s˜ (n) t ,

$$dX_t = \left( f(t) \, X_t - g(t)^2 \tilde{s}_t^{(n)}(X_t \mid S, A) \right) dt + g(t) d\overline{W}_t. \quad (21)$$

Minimizing ℓTD-DD(θ) leads to score function s˜ (n+1) t (s | s, a) generating a marginal probability q (n+1) t that approximates T π q (n) 0 at t = 0.

Following the TD<sup>2</sup> -CFM blueprint, we can further exploit the structure of the target bootstrapped distribution to design an improved diffusion process that converts Gaussian noise to T π q (n) 0 . First, we show below that the mixture of a diffusion process is also a diffusion process with modified drift and diffusion functions.

Lemma 2. *Consider two diffusion processes with drift functions* → f *and* ↷ f*, sharing the same diffusion coefficient* g*:*

$$\begin{aligned} dX_t &= \vec{f}_t(X_t) \, dt + g(t) \, dW \\ dX_t &= \hat{f}_t(X_t) \, dt + g(t) \, dW . \end{aligned}$$

*Let* <sup>→</sup> q<sup>t</sup> *and* ↷ q<sup>t</sup> *be their marginal distribution, then the diffusion process corresponding to the mixture marginal distribution* q<sup>t</sup> = (1 − γ) → q<sup>t</sup> + γ q<sup>t</sup> *is:*

$$dX_t = \frac{(1-\gamma)\vec{q}_t\vec{f}_t + \gamma\widehat{q}_t\widehat{f}_t}{(1-\gamma)\vec{q}_t + \gamma\widehat{q}_t}(X_t) dt + g(t) dW .$$

*Proof.* The marginal probabilities <sup>→</sup> p and ↷ p are characterized by the Fokker-Planck equations:

$$\begin{aligned}\frac{\partial \vec{p}_t}{\partial t} &= -\operatorname{div}(\vec{p}_t \vec{f}_t) + \frac{g_t^2}{2} \Delta \vec{p}_t \\ \frac{\partial \vec{p}_t}{\partial t} &= -\operatorname{div}(\hat{p}_t \hat{f}_t) + \frac{g_t^2}{2} \Delta \hat{p}_t\end{aligned}$$

where div is the divergence operator and ∆ = div∇ is the Laplace operator. Therefore,

$$\begin{aligned} \frac{\partial p_t}{\partial t} &= (1 - \gamma) \frac{\partial \vec{p}_t}{\partial t} + \gamma \frac{\partial \hat{\vec{p}}_t}{\partial t} \\ &= -\operatorname{div}(\vec{p}_t \vec{f}_t) + \frac{g_t^2}{2} \Delta \vec{p}_t - \operatorname{div}(\hat{\vec{p}}_t \hat{f}_t) + \frac{g_t^2}{2} \Delta \hat{\vec{p}}_t \\ &= -\operatorname{div} \left( (1 - \gamma) \vec{p}_t \vec{f}_t + \gamma \hat{\vec{p}}_t \hat{f}_t \right) + \frac{g_t^2}{2} \Delta ((1 - \gamma) \vec{p}_t + \gamma \hat{\vec{p}}_t) \\ &= \operatorname{div} \left( p_t \frac{(1 - \gamma) \vec{p}_t \vec{f}_t + \gamma \hat{\vec{p}}_t \hat{f}_t}{(1 - \gamma) \vec{p}_t + \gamma \hat{\vec{p}}_t} \right) + \frac{g_t^2}{2} \Delta p_t. \end{aligned}$$

The drift (1−γ) →pt → ft+γ ↷pt ft (1−γ) <sup>→</sup>pt+<sup>γ</sup> ↷pt and the diffusion coefficient g<sup>t</sup> satisfy the Fokker-Planck equation with the probability path pt, and therefore their associated diffusion process generate pt.

[Lemma 2](#page-14-0) can be easily extended to the case of a continuous mixture of diffusion processes.

This result shows that it is possible to use two independent diffusion processes for the two terms in the sampling process induced by the Bellman operator. For the first, we can use the standard noising diffusion process:

$$\vec{q}_t(x \mid s, a) = \int q_{t|0}(x \mid s') P(ds' \mid s, a),$$

where we sample X<sup>t</sup> ∼ qt|0(· | s ′ ) by simulating a simple forward diffusion process [\(16\)](#page-13-0). For the second term, we can leverage the GHM m (n) t at the previous iteration to construct the process,

$$\widehat{q}_t^{(n)}(x \mid s, a) = \int m_t^{(n)}(x \mid s', \pi(s')) P(\mathrm{d}s' \mid s, a),$$

where m (n) t (x | s ′ , a′ ) is the marginal probability of the reverse SDE induced by the score s (n) ,

$$dX_t = \left( f(t) X_t - g(t)^2 s_t^{(n)}(X_t \mid s, a) \right) dt + g(t) d\overline{W}_t.$$

Additionally, ↷ q (n) t (x | s, a), as continuous mixture of diffusion's marginals m (n) t (x | s ′ , π(s ′ )) weighted by P(s ′ | s, a), can be generated by the diffusion process,

$$dX_t = (f(t) X_t - g(t)^2 \widehat{s}_t(X_t \mid s, a)) dt + g(t) d\bar{W}_t, \text{ where } \widehat{s}_t(x_t \mid s, a) = \frac{\int P(ds' \mid s, a) q_t^{(n)}(x \mid s', \pi(s')) s_t^{(n)}(x_t \mid s', \pi(s'))}{\int P(ds' \mid s, a) q_t^{(n)}(x \mid s', \pi(s'))}.$$

$$dX_t = (f(t) X_t - g(t)^2 \widehat{s}_t(X_t \mid s, a)) dt + g(t) d\overline{W}_t, \text{ where}$$

Given these two diffusion processes, the target probability q (n+1) <sup>t</sup> = (1 − γ) → q<sup>t</sup> + γ q (n) t can be generated by the following reverse SDE,

$$dX_t = \left( f(t)X_t - g(t)^2 s_t^{(n+1)}(X_t \mid s, a) \right) dt + g(t) d\overline{W}_t,$$

where s (n+1) t (<sup>x</sup> | s, a) = (1−γ) <sup>→</sup>qt∇<sup>x</sup> log→qt+<sup>γ</sup> ↷q (n) t ↷s (n) t (1−γ) <sup>→</sup>qt+<sup>γ</sup> ↷q (n) (x | s, a). Therefore, we can learn s˜t(· · · ; θ) to approximate s (n+1) <sup>t</sup> by minimizing the loss,

$$\begin{aligned} \ell(\theta) &= (1 - \gamma) \mathbb{E}_{\rho, t, X_t \sim \vec{q}_t(\cdot | S, A)} \left[ \| \tilde{s}(X_t | S, A; \theta) - \nabla_{X_t} \log \vec{q}_t(X_t | S, A) \|^2 \right] \\ &\quad + \gamma \mathbb{E}_{\rho, t, X_t \sim \vec{q}_t^{(n)}(\cdot | S, A)} \left[ \| \tilde{s}(X_t | S, A; \theta) - \hat{s}_t^{(n)}(X_t | S, A) \|^2 \right]. \end{aligned} \quad (22)$$

We can simplify the first term via [Proposition 2](#page-14-1) (since <sup>→</sup> qt(x|s, a) = R qt|0(x|s ′ )P(ds ′ |s, a)), hence we have

$$\begin{aligned}\nabla_{\theta} \mathbb{E}_{\rho, t, X_t \sim \vec{q}_t(\cdot | s, a)} \left[ \left\| \tilde{s}(X_t \mid s, a; \theta) - \nabla_{X_t} \log \vec{q}_t(X_t \mid S, A) \right\|^2 \right] &= \\ &\nabla_{\theta} \mathbb{E}_{\rho, t, X_t \sim q_{t|0}(\cdot | S')} \left[ \left\| \tilde{s}(X_t \mid S, A; \theta) - \nabla_{X_t} \log q_{t|0}(X_t \mid S') \right\|^2 \right].\end{aligned}$$

Moreover, using a similar argument for equivalence between the gradient of marginal and conditional flow-matching objectives, we can show that

$$\begin{aligned}\nabla_{\theta} \mathbb{E}_{\rho, t, X_t \sim \widehat{q}_t^{(n)}(\cdot | S, A)} \left[ \left\| \tilde{s}(X_t | S, A; \theta) - \widehat{s}_t^{(n)}(X_t | S, A) \right\|^2 \right] &= \\ \nabla_{\theta} \mathbb{E}_{\rho, t, X_T \sim q_T, X_t \sim q_{t|T}^n(\cdot | s, a)} \left[ \left\| \tilde{s}(X_t | S, A; \theta) - s_t^{(n)}(X_t | S, A) \right\|^2 \right].\end{aligned}$$

This leads us to the final TD<sup>2</sup> -DD loss function,

$$\begin{aligned} \ell_{\text{TD}^2\text{-DD}}(\theta) &= (1 - \gamma) \mathbb{E}_{\rho, t, X_t \sim q_{t|0}(\cdot|S')} \left[ \left\| \tilde{s}_t(X_t \mid S, A; \theta) - \nabla_x \log p_{t|0}(X_t \mid S') \right\|^2 \right] \\ &\quad + \gamma \mathbb{E}_{\rho, t, X_t \sim q_{t|T}^{(n)}(\cdot|S', \pi(S'))} \left[ \left\| \tilde{s}(X_t \mid S, A; \theta) - \tilde{s}_t^{(n)}(X_t \mid S', \pi(S')) \right\|^2 \right]. \end{aligned} \quad (23)$$

# C. Experimental Details

#### C.1. Evaluation

Table 3. Evaluation hyper-parameters for both single and multi-policy experiments.

| Evaluation | Hyperparameter |          |             |           | Value      |
|------------|----------------|----------|-------------|-----------|------------|
| Number     | of             | states   | s 0         |           | 64         |
| Number     | of             | m        | -samples    | per state | 2048       |
| Number     | of             | episodes | per         | state     | 1          |
| Episode    |                | length   |             |           | 1000       |
| Number     | of             | state    | s 0         |           | 64         |
| Number     | of             |          | GHM-samples | per       | state 2048 |
| Number     | of             | episodes | per         | state     | 1          |
| Episode    |                | length   |             |           | 1000       |
| Number     | of             | z        | samples     |           | 256        |
| Number     | of             | GHM      | samples     |           | 128        |
| Number     | of             | FB       | inference   | samples   | 250 , 000  |

Evaluating a GHM can be challenging, TD-based losses employing bootstrapping do not provide a good signal as to the quality of the learned model. Instead, we opt to measure 1) the likelihood of a trajectory coming from the true discounted occupancy of a given policy, 2) the Earth Mover's Distance (EMD; [Rubner et al.,](#page-10-8) [2000\)](#page-10-8) between samples from the true occupancy and our GHM which provides an estimate of the distance between these two probability distributions, and 3) the value-function approximation error. In all cases, to obtain samples from the true discounted occupancy, we collect trajectories {(s0, s1, . . . , s<sup>T</sup> )} N <sup>i</sup>=1 from policy π and subsequently resample states according to t ∼ Geometic(1 − γ) for a particular discount factor γ ∈ [0, 1). Armed with samples from m<sup>π</sup> we compute the aforementioned metrics following the procedures stated below along with the parameter values outlined in [Table 3.](#page-17-2)

Normalized Negative Log-Likelihood. To compute the log-likelihood of our flow matching and diffusion methods, we take advantage of the following change in variables formula [\(Dinh et al.,](#page-9-20) [2015;](#page-9-20) [Rezende & Mohamed,](#page-10-19) [2015;](#page-10-19) [Chen et al.,](#page-9-7) [2018\)](#page-9-7),

$$\log(\tilde{m}(x_1 \mid s, a; \theta)) = \log \varphi(x_0) + \int_0^1 \frac{\partial \log(\tilde{m}(x_t \mid s, a; \theta))}{\partial x_t} dt,$$

where φ is the probability density function of a standard Gaussian distribution, which acts as the prior on x0. The change in log density over time can be written as the following differential equation called the instantaneous change of variables formula [\(Chen et al.,](#page-9-7) [2018,](#page-9-7) Theorem 1),

$$\frac{\partial \log(\tilde{m}(x_t | s, a; \theta))}{\partial x_t} = -\text{Tr}\left(\frac{\partial \tilde{v}_t(x_t | s, a; \theta)}{\partial x_t}\right).$$

We can now compute the log-likelihood for a sample X ∼ m<sup>π</sup> (· | s, a) by integrating the total change in log-density backward in time from x<sup>1</sup> = X to obtain x<sup>0</sup> which has tractable likelihood. In practice, we solve the following coupled initial value problem using numerical integration [\(Grathwohl et al.,](#page-9-22) [2019\)](#page-9-22),

$$\begin{bmatrix} \log \tilde{m}(x_1 \mid s, a; \theta) - \log \varphi(x_0) \\ \log \tilde{m}(x \mid s, a; \theta) - \log \tilde{m}(x_1 \mid s, a; \theta) \end{bmatrix} = \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = \begin{bmatrix} -\tilde{v}_t(x_t \mid s, a; \theta) \\ \left( \frac{\partial \tilde{v}_t(x_t \mid s, a; \theta)}{\partial x_t} \right) \end{bmatrix} dt, \quad (24)$$

For all experiments we report the negative log-likelihood *normalized by the dimension of the observation space*.

Earth Mover's Distance We compute the Earth Mover's Distance (EMD; [Rubner et al.,](#page-10-8) [2000\)](#page-10-8), also known as the Wasserstein-1 distance, between m = 2048 samples from the ground truth distribution X ∼ m<sup>π</sup> (· | Sk, Ak) and our learned GHM <sup>X</sup>e <sup>∼</sup> <sup>m</sup>e (· | <sup>S</sup>k, Ak; <sup>θ</sup>) for a set of randomly sampled state-action pairs {(Sk, Ak)} n <sup>k</sup>=1. Intuitively, the EMD quantifies the minimum cost required to transform one distribution into another, where the cost is defined in terms of the Euclidean distance between states X(i) , X(j) . Formally, we have,

$$\text{EMD}(\{X^{(1)}, \dots, X^{(m)}\}, \{\tilde{X}^{(1)}, \dots, \tilde{X}^{(m)}\}) = \min_{\xi \in \Xi} \sum_{i,j} \xi_{ij} \sum_{k=1}^d \left( X_k^{(i)} - \tilde{X}_k^{(j)} \right)^2,$$

where ξ is a transport plan such that ξij specifies the proportion of mass moved from X<sup>i</sup> to <sup>X</sup>e<sup>j</sup> . We report the average EMD across n = 64 source states using the Python Optimal Transport [\(Flamary et al.,](#page-9-23) [2021\)](#page-9-23) library.

Value Function Mean Square Error (MSE(V)). We compute the mean square error between a Monte-Carlo estimation Ve π MC of the value function V π (s) and the estimation <sup>V</sup>eGHM obtained using the learned model. We obtain <sup>V</sup>e <sup>π</sup> MC by collecting a trajectory {(s0, s1, . . . , s<sup>T</sup> )} from policy π and computing the discounted sum of rewards. We generate a single trajectory since both the policy and the environment are deterministic. The GHM estimate is given by [\(2\)](#page-1-3), i.e.,

$$\widetilde{V}_{\text{GHM}}^{\pi}(s) = (1 - \gamma)^{-1}\mathbb{E}_{\widetilde{X} \sim \widetilde{m}(\cdot | s, \pi(s))} \left[ r(\widetilde{X}) \right].$$

Then, MSE(Ve <sup>π</sup> MC, <sup>V</sup>e <sup>π</sup> GHM) = <sup>E</sup>S0∼<sup>ν</sup> h (Ve <sup>π</sup> GHM(S0) − <sup>V</sup>e <sup>π</sup> MC(S0))<sup>2</sup> i . We average our results over 64 initial states S<sup>0</sup> sampled from the initial state distribution ν.

Planning with GPI. We evaluate planning performance by computing the average return over 100 episodes, each lasting 1, 000 steps, for every task. For the Forward-Backward representation [\(Touati & Ollivier,](#page-11-16) [2021\)](#page-11-16), we directly follow the policy π<sup>w</sup><sup>r</sup> (thus a<sup>t</sup> = π<sup>w</sup><sup>r</sup> (st)) where w<sup>r</sup> = <sup>E</sup>(S,R)∼ρ[B(s) · R] is the zero-shot policy embedding inferred using 250, 000 transitions labeled with the task reward function r. Given that FB provides a direct way of estimating the value function of a policy (i.e., Q<sup>π</sup><sup>w</sup> r (s, a) = F(s, a, w) T zr), we can do planning in the policy embedding space by solving the following problem:

$$w_t^{\text{FB-GPI}} \in \arg \max_{w \sim D(W)} F(s_t, \pi_w(s_t), w)^T w_r.$$

This optimization problem requires no generation except sampling from D(W). We approximate the max using 255 samples from D(W) and additionally incorporating w<sup>r</sup> to ultimately maximize over 256 policies. On the other hand, for GHM-GPI, we solve the following optimization problem,

$$w_t^{\text{GHM-GPI}} \in \arg \max_{w \sim D(W)} \underbrace{(1 - \gamma)^{-1} \mathbb{E}_{X \sim m^{\pi w}(\cdot | s_t, \pi_w(s_t))} [r(X)]}_{Q^{\pi w}(s_t, \pi_w(s_t))},$$

which requires generating samples from m<sup>π</sup><sup>w</sup> . In our experiments we generate 128 samples from m<sup>π</sup><sup>w</sup> .

# C.2. Environments

Experiments in this paper were conducted with a subset of domains from the DeepMind Control Suite [\(Tunyasuvunakool](#page-11-15) [et al.,](#page-11-15) [2020\)](#page-11-15) highlighted in [Figure 4.](#page-18-0)

![](_page_18_Picture_12.jpeg)

Figure 4. A visual depiction of each domain used in our experiments from the DeepMind Control Suite [\(Tunyasuvunakool et al.,](#page-11-15) [2020\)](#page-11-15). From left to right: MAZE, CHEETAH, QUADRUPED, WALKER.

#### C.3. Geometric Horizon Models

This section describes each class of generative model used for our empirical experiments.

# C.3.1. FLOW MATCHING

|     | Algorithm 1 Template for TD-Flow algorithms                                           |
|-----|---------------------------------------------------------------------------------------|
| 1:  | Inputs : offline dataset D , policy π , batch size n , Polyak coefficient ζ ,         |
|     | weight decay λ , randomly initialized weights θ , discount factor γ , learning        |
|     | rate η , one-step conditional path → p t   1 and conditional vector-field → u t   1 , |
|     | bootstrap path ↷ p t and vector-field ↷ v t                                           |
| 2:  | for n = 1 , do                                                                        |
| 3:  | Sample mini-batch { ( S k , A k , S ′                                                 |
|     | k ) }                                                                                 |
|     | k =1 from D                                                                           |
| 4:  | for k = 1 , , K do                                                                    |
| 5:  | Sample t k ∼ U ([0 , 1])                                                              |
| 6:  | Sample                                                                                |
|     | X k ∼                                                                                 |
|     | → p t k   1 (   S                                                                     |
|     | k )                                                                                   |
|     | ℓ k ( θ ) =                                                                           |
|     | v t k                                                                                 |
|     | X k   S k , A k ; θ ) −                                                               |
|     | → u t k   1 (                                                                         |
|     | X k   S                                                                               |
|     | k )                                                                                   |
| 8:  | Sample                                                                                |
|     | X k ∼                                                                                 |
|     | ↷ p t k                                                                               |
|     | (   S                                                                                 |
|     | k , π ( S                                                                             |
|     | k ); ¯ θ )                                                                            |
|     | ℓ k ( θ ) =                                                                           |
|     | v t k                                                                                 |
|     | X k   S k , A k ; θ ) −                                                               |
|     | ↷ v t k                                                                               |
|     | X k   S                                                                               |
|     | k , π ( S                                                                             |
|     | k ); ¯ θ )                                                                            |
| 10: | end for                                                                               |
| 11: | # Compute loss                                                                        |
| 12: | ℓ ( θ ) = 1                                                                           |
|     | P K                                                                                   |
|     | k =1 (1 − γ )                                                                         |
|     | ℓ k ( θ ) + γ                                                                         |
|     | ℓ k ( θ )                                                                             |
| 13: | # Perform gradient step                                                               |
| 14: | θ ← θ − η ∇ θ                                                                         |
|     | ℓ ( θ ) + λ ∥ θ ∥                                                                     |
| 15: | # Update parameters of target vector field                                            |
| 16: | ¯ θ ← ζ                                                                               |
|     | ¯ θ + (1 − ζ ) θ                                                                      |
| 17: | end for                                                                               |

Table 4. Summary of how different TD-flow algorithms generate the target probability path and vector field. The neural ode ψ<sup>t</sup> is defined by the vector field ↷ v <sup>t</sup> computed at iteration n.

|               |     | ↷     |   | ↷                               |
|---------------|-----|-------|---|---------------------------------|
|               |     | p     | t |                                 |
|               |     |       |   | v t                             |
| TD CFM        |     |       |   |                                 |
| X             | 0   | ∼     | m | 0                               |
| X             |     |       |   | u t   1 ( X t   X 1 )           |
| 1 = ψ 1       | ( X | 0     |   | S                               |
|               |     |       |   | , A ′                           |
|               |     |       |   | ; ¯ θ ) ′                       |
| X t ∼         | p   | t   1 | ( | X 1 )                           |
| TD CFM ( C )  |     |       |   |                                 |
| X             | 0   | ∼     | m | 0                               |
| X             |     |       |   | u t   0 , 1 ( X t   X 0 , X 1 ) |
| 1 = ψ 1       | ( X | 0     |   | S                               |
|               |     |       |   | , A ′                           |
|               |     |       |   | ; ¯ θ ) ′                       |
| X t ∼ p t CFM | 0 , | 1 (   |   | X 0 , X 1 )                     |
| X             | 0   | ∼     | m | 0                               |
|               |     |       |   | v t ( X t   S                   |
|               |     |       |   | , A ′                           |
|               |     |       |   | ; ¯ θ ) ′                       |
| X t = ψ t     | ( X | 0     |   | S                               |
|               |     |       |   | , A ′                           |
| TD            |     |       |   | ; ¯ θ ) ′                       |

To discuss the TD-Flow methods introduced herein, we first unify the loss function through defining a general template for the loss as,

$$\begin{aligned} \ell(\theta) &= (1 - \gamma)\mathbb{E}_{\rho, t, X_t \sim \vec{p}_{t|1}(\cdot | S')} \left[ \|v_t(X_t | S, A; \theta) - \vec{u}_{t|1}(X_t | S')\|^2 \right] \\ &\quad + \gamma\mathbb{E}_{\rho, t, X_t \sim \widehat{p}_t^{(n)}(\cdot | Z)} \left[ \|v_t(X_t | S, A; \theta) - \widehat{v}_t^{(n)}(X_t | Z)\|^2 \right]. \end{aligned}$$

We can now recover each algorithm by a specific choice of the target probability path ↷ p (n) t and vector field ↷ v (n) t as illustrated in [Figure 4.](#page-19-1) Based on this unified structure, we present pseudo-code for the TD flow methods in [Figure 1.](#page-19-1) In practice, instead of proceeding through full iterations, we use standard mini-batch gradient updates with a target network ¯θ updated as a moving average of θ.

When employing the conditional probability path <sup>→</sup> <sup>p</sup>t|<sup>1</sup> and vector field <sup>→</sup> ut|<sup>1</sup> we use the standard Gaussian linear interpolation defined as <sup>→</sup> pt|1(· | X1) = N (· | tX1,(1 − t) 2 <sup>I</sup>), hence <sup>X</sup><sup>t</sup> <sup>=</sup> tX<sup>1</sup> + (1 − <sup>t</sup>)X<sup>0</sup> ∼ <sup>p</sup>t|1, resulting in <sup>→</sup> ut|1(X<sup>t</sup> | X1) = (X<sup>1</sup> − Xt)/(1 − t) [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2). The source distribution for all experiments is m0(·) = N (· | 0, I). To sample from the Neural ODE we use the Midpoint method with a constant step size of dt = t/10 for a total of 10 steps. We found both coupled and TD<sup>2</sup> methods do not require many solver steps and hypothesize this is due to the reduction in transport cost as analyzed in [Appendix E.7.](#page-41-0)

For all flow and diffusion-based methods, we employ a U-Net-style architecture [\(Ronneberger et al.,](#page-10-20) [2015\)](#page-10-20) that has hierarchical skip connections throughout an MLP. We embed the timestep t by first increasing its dimensionality with a sinusoidal embedding before transforming it through a two-layer MLP with mish activations [\(Misra,](#page-10-21) [2019\)](#page-10-21). We further process additional conditioning information, such as the state-action pair and Forward-Backward embedding z through an additional two-layer MLP, whose result then gets concatenated with our time embedding. Finally, the network integrates all prior conditioning information through FiLM modulation [\(Perez et al.,](#page-10-22) [2018\)](#page-10-22) that replaces the learned affine transformation for layer normalization [\(Ba et al.,](#page-8-10) [2016\)](#page-8-10).

#### C.3.2. DENOISING DIFFUSION

We train a Denoising Diffusion Probabilistic Model (DDPM; [Ho et al.,](#page-9-5) [2020\)](#page-9-5) using the same architecture as our flow matching model above, with the output now being interpreted as a prediction of the noise seed ϵ<sup>0</sup> that began the diffusion process. We discretize the diffusion process using 1, 000 steps with βmin = 0.1 and βmax = 20. We employ the DDIM sampler[\(Song et al.,](#page-11-17) [2021a\)](#page-11-17) with 50 sampling steps for both training and evaluation.

For evaluating our DDPM model, we compute exact log-likelihoods using the instantaneous change of variables formula [\(Chen et al.,](#page-9-7) [2018\)](#page-9-7) along with the probability flow ODE from [Song et al.](#page-11-24) [\(2021b\)](#page-11-24). That is, we solve the initial value problem in [\(24\)](#page-17-3) using the vector field,

$$v_t(x_t \mid s, a; \theta) = -\frac{1}{2} (\beta_{\min} + t (\beta_{\max} - \beta_{\min})) \left( x_t - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t(x_t \mid s, a; \theta) \right).$$

We now outline the losses for each of the TD-DPM experiments in the paper:

TD-DD To train our vanilla Diffusion GHM we employ the standard DDPM-style objective, that is, we optimize the following loss:

$$\mathbb{E}_{\substack{\rho, t, \epsilon \sim \mathcal{N}(\cdot \mid 0, I) \\ X_0 \sim (T^\pi \hat{m}^{(n)})(\cdot \mid S, A)}} \left[ \left\| \epsilon - \epsilon_t(\sqrt{\bar{\alpha}} X_0 + \sqrt{1 - \bar{\alpha}} \epsilon \mid S, A; \theta) \right\|^2 \right], \quad (25)$$

where ¯θ are the target parameters and α¯ are the standard diffusion coefficients as seen in [Ho et al.](#page-9-5) [\(2020\)](#page-9-5).

TD<sup>2</sup> -DD As outlined in [§3.1](#page-3-7) we can split our DDPM loss into two terms, one that will use standard DDPM training on one-step transitions and the second term that will regress to our target networks noise prediction. This materializes as,

$$\begin{aligned} \vec{\ell}(\theta) &= \mathbb{E}_{\rho, t, \epsilon, X_0} \left[ \left| |\epsilon_t(\sqrt{\alpha_t} X_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon | S, A; \theta) - \epsilon | \right|^2 \right] \\ &\quad \text{where } X_0 \sim P(\cdot | S, A), \\ \widehat{\ell}(\theta) &= \mathbb{E}_{\rho, t, \epsilon, \widehat{X}_t} \left[ \left| |\epsilon_t(\widehat{X}_t | S, A; \theta) - \epsilon_t^{(n)}(\widehat{X}_t | S', \pi(S'))| \right|^2 \right] \\ &\quad \text{where } \widehat{X}_t \sim q_{t|T}^{(n)}(\cdot | S', \pi(S')) \\ \ell_{\text{TD}^2\text{-DD}}(\theta) &= (1 - \gamma) \vec{\ell}(\theta) + \gamma \widehat{\ell}(\theta) \end{aligned} \tag{26}$$

### C.3.3. GENERATIVE ADVERSARIAL NETWORK

We implement a modern Generative Adversarial Network (GAN; [Goodfellow et al.,](#page-9-9) [2014\)](#page-9-9) baseline based on the recommendations in [Huang et al.](#page-9-12) [\(2024\)](#page-9-12). Specifically, we train a relativistic GAN [\(Jolicoeur-Martineau,](#page-9-13) [2019\)](#page-9-13) resulting in the following loss,

$$\ell_{\text{GAN}}(\theta_G, \theta_D) = \mathbb{E}_{\rho, X_0, X_1} \left[ f(D(G(X_0 | S, A; \theta_G); \theta_D) - D(X_1 | S, A; \theta_D)) \right],$$

$$\text{where } X_0 \sim \mathcal{N}(\cdot | 0, I), X_1 \sim \left( \mathcal{T}^\pi \tilde{m}^{(n)} \right) (\cdot | S, A),$$

We take f(x) = − log (1 + exp (−x)) to be the log-sigmoid function [\(Jolicoeur-Martineau,](#page-9-13) [2019\)](#page-9-13) and further add the following zero-centered gradient penalties on the discriminator,

$$\begin{aligned} R_1(\theta_D) &= \mathbb{E}_{\rho, X \sim (\mathcal{T}^\pi \tilde{m}(n))(\cdot | S, A)} [\|\nabla_X D(X | S, A)\|^2], \\ R_2(\theta_G, \theta_D) &= \mathbb{E}_{\rho, X \sim (\mathcal{T}^\pi \tilde{m}(\cdot | S, A; \theta_G))} [\|\nabla_X D(X | S, A)\|^2]. \end{aligned}$$

The penalty <sup>R</sup><sup>1</sup> penalizes the gradient norm of the discriminator <sup>D</sup> on "real data" sampled from our current iterate <sup>m</sup>e (n) , whereas R<sup>2</sup> penalizes the gradient norm on "fake data" generated directly from the current generator. We experimented with different coefficients and schedules for these gradient penalties and settled on a linear decay schedule from 0.05 → 0.005

throughout training. Furthermore, as is common practice, we impose a schedule on the second moment EMA coefficient β<sup>2</sup> of Adam [\(Kingma & Ba,](#page-10-23) [2015\)](#page-10-23) to increase from 0.9 → 0.99 throughout training.

The generator and discriminator architecture in our GAN is implemented as a Residual MLP with leaky ReLU activations with the same FiLM-style conditioning [\(Perez et al.,](#page-10-22) [2018\)](#page-10-22) as our flow and diffusion models. The input to our generator is random noise sampled from an isotropic Gaussian with dimensionality equal to the number of state dimensions in the environment.

# C.3.4. VARIATIONAL AUTO-ENCODER

We implement a β-Variational Auto-Encoder [\(Kingma & Welling,](#page-10-24) [2014;](#page-10-24) [Higgins et al.,](#page-9-10) [2017\)](#page-9-10) following the best practices outlined in [Thakoor et al.](#page-11-7) [\(2022\)](#page-11-7). That is, we train our VAE to minimize the following loss,

$$\ell_{\text{VAE}}(\theta_{\text{E}}, \theta_{\text{D}}) = \mathbb{E}_{\rho, X_1} \left[ \mathbb{E}_{X_0 \sim q_{\theta_{\text{E}}}(\cdot | S, A, X_1)} \left[ \log p_{\theta_{\text{D}}}(X_1 \mid S, A, X_0) \right] - \beta D_{\text{KL}}(q_{\theta_{\text{E}}} || p_0) \right],$$

$$\text{where } X_1 \sim \left( \mathcal{T}^\pi \widetilde{m}^{(n)} \right) (\cdot \mid S, A).$$

We employ a similar architecture to our GAN-GHM and use a residual MLP for the encoder and decoder. We use an isotropic Gaussian latent space with the number of latents equal to the number of state dimensions in the environment. We also swept over β ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} on the MAZE domain and chose β = 0.5 for the rest of our experiments. Overall, we found the β-VAE-based GHM to be very unstable and likely requires very careful fine-tuning of β to get adequate performance at long-horizons.

### C.4. Hyperparameters

We report the hyper-parameters for training the GHM models used in the single and multi-policy experiments. [Table 5](#page-22-0) shows the parameters for Flow Matching and Denoising Diffusion. We also report the hyper-parameters for pre-training the Forward-Backward representation [\(Touati & Ollivier,](#page-11-16) [2021\)](#page-11-16) utilized in the multi-policy GHM experiments in [Table 8.](#page-23-0)

Table 5. Flow Matching and Denoising Diffusion hyper-parameters used for the single and multi-policy experiments across tasks and domains. We highlight any differences depending on the training context.

| ODE Flow Matching ODE (Lipman et al., 2023) ODE | Hyperparameter Solver d t (train) d t (eval) | 0 0      | 1 1   | Single Policy Midpoint | 0 0         | 1 05  | Multi-Policy Midpoint ( 0 1 for GPI) |
|-------------------------------------------------|----------------------------------------------|----------|-------|------------------------|-------------|-------|--------------------------------------|
| Diffusion (DDPM)                                |                                              |          |       |                        |             |       |                                      |
| (Ho et al., 2020)                               |                                              |          |       |                        |             |       |                                      |
| β min                                           |                                              | 0        | 1     |                        | 0           | 1     |                                      |
| β max                                           |                                              | 20       |       |                        | 20          |       |                                      |
| Discretization                                  | Steps                                        | 1        | , 000 |                        | 1           | , 000 |                                      |
| SDE                                             | Solver                                       | DDIM     |       | (Song et al.,          | 2021a) DDIM |       | (Song et al., 2021a)                 |
| SDE                                             | Solver Steps (train)                         | 20       |       |                        | 20          |       |                                      |
| SDE                                             | Solver Steps (eval)                          | 20       |       |                        | 20          |       |                                      |
| Network (U-Net)                                 |                                              |          |       |                        |             |       |                                      |
| (Ronneberger et al., 2015)                      |                                              |          |       |                        |             |       |                                      |
| t -Positional                                   | Embedding                                    | Dim. 256 |       |                        | 256         |       |                                      |
| t -Positional                                   | Embedding                                    | MLP (256 |       | , 256)                 | (256        |       | , 256)                               |
| Hidden                                          | Activation                                   | mish     |       | (Misra, 2019)          | mish        |       | (Misra, 2019)                        |
| Blocks                                          | per Stage                                    | 1        |       |                        | 1           |       |                                      |
| Block                                           | Dimensions                                   | (512     |       | , 512 , 512)           | (1024       |       | , 1024 , 1024)                       |
| Conditional Encoder                             |                                              |          |       |                        |             |       |                                      |
| Encoder                                         | Input                                        | s,       | a     |                        | s,          | a,    | z                                    |
| Encoder                                         | MLP                                          | (512     |       | , 512 , 512)           | (1024       |       | , 1024 , 1024)                       |
| Encoder                                         | Activation                                   | mish     |       | (Misra, 2019)          | mish        |       | (Misra, 2019)                        |
| Optimizer (AdamW)                               |                                              |          |       |                        |             |       |                                      |
| (Loshchilov & Hutter, 2019)                     |                                              |          |       |                        |             |       |                                      |
| AdamW                                           | β 1                                          | 0        | 9     |                        | 0           | 9     |                                      |
| AdamW                                           | β 2                                          | 0        | 999   |                        | 0           | 999   |                                      |
| AdamW                                           | ϵ                                            | 10       | −     | 4                      | 10          | −     | 4                                    |
| Learning                                        | Rate                                         | 10       | −     | 4                      | 10          | −     | 4                                    |
| Weight                                          | Decay                                        | 10       | −     | 3                      | 10          | −     | 2                                    |
| Gradient Common                                 | Steps                                        | 3        | M     |                        | 8           | M     |                                      |
| Batch                                           | Size                                         | 1024     |       |                        | 1024        |       |                                      |
| Target                                          | Network EMA                                  | 10       | −     | 3                      | 10          | −     | 4                                    |

Table 6. β-VAE [\(Higgins et al.,](#page-9-10) [2017\)](#page-9-10) hyper-parameters for single policy experiments across tasks and domains.

| β β -VAE Latent (Higgins et al., 2017) Latent Encoder Decoder Network Hidden Blocks Block Encoder Encoder Conditional Encoder Encoder | Prior Dimension Activation per Stage Dimensions Input MLP Activation | N   1 s, | 10 S   mish (512 a (512 mish | Value (0 , I ) Residual MLP Residual MLP (Misra, 2019) , 512 , 512) , 512 , 512) (Misra, 2019) |
|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|----------|------------------------------|------------------------------------------------------------------------------------------------|
| Optimizer (AdamW)                                                                                                                     |                                                                      |          |                              |                                                                                                |
| (Loshchilov & Hutter, 2019)                                                                                                           |                                                                      |          |                              |                                                                                                |
| AdamW                                                                                                                                 | β 1                                                                  | 0        | 9                            |                                                                                                |
| AdamW                                                                                                                                 | β 2                                                                  | 0        | 999                          |                                                                                                |
| AdamW                                                                                                                                 | ϵ                                                                    | 10       | −                            | 4                                                                                              |
| Learning                                                                                                                              | Rate                                                                 | 10       | −                            | 4                                                                                              |
| Weight                                                                                                                                | Decay                                                                | 10       | −                            | 3                                                                                              |
| Gradient Common                                                                                                                       | Steps                                                                | 3        | M                            |                                                                                                |
| Batch                                                                                                                                 | Size                                                                 | 1024     |                              |                                                                                                |
| Target                                                                                                                                | Network EMA                                                          | 10       | −                            | 3                                                                                              |

Table 7. GAN hyper-parameters for single policy experiments across tasks and domains.

| Grad. RGAN Latent 2019) Latent Network Hidden Blocks Block Encoder Encoder Conditional Encoder Encoder | Penalty Prior Dimension Generator Discriminator Activation per Stage Dimensions Input MLP Activation | Coef N   1 s, | S   Leaky (512 a (512 Leaky | Value Linear( 0 05 → 0 005 ) (0 , I ) Residual MLP Residual MLP ReLU , 512 , 512) , 512 , 512) ReLU |
|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|---------------|-----------------------------|-----------------------------------------------------------------------------------------------------|
| Optimizer (AdamW)                                                                                      |                                                                                                      |               |                             |                                                                                                     |
| (Loshchilov & Hutter, 2019)                                                                            |                                                                                                      |               |                             |                                                                                                     |
| AdamW                                                                                                  | β 1                                                                                                  | 0             | 9                           |                                                                                                     |
| AdamW                                                                                                  | β 2                                                                                                  | Linear(       |                             | 0 9 → 0 99 )                                                                                        |
| AdamW                                                                                                  | ϵ                                                                                                    | 10            | −                           | 4                                                                                                   |
| Learning                                                                                               | Rate                                                                                                 | 10            | −                           | 4                                                                                                   |
| Weight                                                                                                 | Decay                                                                                                | 10            | −                           | 3                                                                                                   |
| Gradient Common                                                                                        | Steps                                                                                                | 3             | M                           |                                                                                                     |
| Batch                                                                                                  | Size                                                                                                 | 1024          |                             |                                                                                                     |
| Target                                                                                                 | Network EMA                                                                                          | 10            | −                           | 3                                                                                                   |

Table 8. Forward Backward Representation hyper-parameters. We largely reuse the hyper-parameters from [Pirotta et al.](#page-10-6) [\(2024\)](#page-10-6) and highlight any deviations.

|                           | Hyperparameter             | Walker      | Cheetah     | Quadruped   | Maze        |
|---------------------------|----------------------------|-------------|-------------|-------------|-------------|
| Forward Backward          |                            |             |             |             |             |
| (Touati & Ollivier, 2021) |                            |             |             |             |             |
|                           | Embedding Dimension d      | 100         | 50          | 50          | 100         |
|                           | Embedding Prior            | S           |             |             |             |
|                           |                            | d           | S           |             |             |
|                           |                            |             | d           | S           |             |
|                           |                            |             |             | d           | S d         |
|                           | Embedding Prior Goal Prob. | 0           | 0           | 0           | 1 / 2       |
|                           | B Normalization            | ℓ 2         | ℓ 2         | ℓ 2         | ℓ 2         |
|                           | Orthonormal Loss Coeff.    | 1           | 1           | 1           | 1           |
| Policy (TD3)              |                            |             |             |             |             |
| (Fujimoto et al., 2018)   |                            |             |             |             |             |
|                           | Target Policy Noise        | N (0 , 0 2) | N (0 , 0 2) | N (0 , 0 2) | N (0 , 0 2) |
|                           | Target Policy Clipping     | 0 3         | 0 3         | 0 3         | 0 3         |
|                           | Policy Update Frequency    | 1           | 1           | 1           | 1           |
| Optimizer (Adam)          |                            |             |             |             |             |
| (Kingma & Ba, 2015)       |                            |             |             |             |             |
|                           | Learning Rate (F, B)       | ( 10 − 4    |             |             |             |
|                           |                            | , 10 − 4    |             |             |             |
|                           |                            | )           | ( 10 − 4    |             |             |
|                           |                            |             | , 10 − 4    |             |             |
|                           |                            |             | )           | ( 10 − 4    |             |
|                           |                            |             |             | , 10 − 4    |             |
|                           |                            |             |             | )           | ( 10 − 4    |
|                           |                            |             |             |             | , 10 − 6    |
|                           | Learning Rate ( π )        | 10 − 4      | 10 − 4      | 10 − 4      | 10 − 6      |
|                           | Adam β 1                   | 0 9         | 0 9         | 0 9         | 0 9         |
|                           | Adam β 2                   | 0 999       | 0 999       | 0 999       | 0 999       |
|                           | Adam ϵ                     | 10 − 8      | 10 − 8      | 10 − 8      | 10 − 8      |
|                           | Batch Size                 | 2048        | 1024        | 2048        | 1024        |
| Common                    | Gradient Steps             | 3 M         | 3 M         | 3 M         | 5 M         |
|                           | Discount Factor γ          | 0 98        | 0 98        | 0 98        | 0 99        |
|                           | Target Network EMA         | 0 99        | 0 99        | 0 99        | 0 99        |
|                           | Reward Inference Samples   | 250 , 000   | 250 , 000   | 250 , 000   | 250 , 000   |

)

# D. Additional Experimental Results

In this section, we report additional results about the experiments.

Single policy. We report metrics averaged over tasks using a curved conditional path in [Table 12.](#page-26-0) We also report the performance per task in [Table 13.](#page-26-1) [Table 11](#page-25-0) shows the performance of the single-policy experiments [\(§5.1](#page-5-1) in the main paper) expanded for each task. While the performance of TD-based methods is reasonably stable across tasks, VAE and GAN have a large variance across tasks. For example, the EMD of GAN diverges in 2 tasks out of 4 in QUADRUPED.

Multiple policies and planning. We report aggregate performance across our full suite of evaluation metrics for the multi-policy experiments in [Table 14.](#page-27-0) We also report per-task metrics in [Table 16.](#page-28-0) We can notice that TD<sup>2</sup> -DD achieves quite a high EMD compared to TD-DD while achieving a better MSE(V). By further inspecting the generated samples (see [Figure 5\)](#page-30-0), we found that TD-DD tends to generate highly concentrated samples, while TD<sup>2</sup> -DD is more diffuse. However, the samples generated by TD-DD appear to be better at a visual inspection. This may explain the discrepancy between the two metrics. Finally, we report aggregate planning performance in [Table 15](#page-27-1) and per-task results in [Table 17.](#page-29-0)

Comparison with planning with one-step world model we include in [Table 9](#page-24-1) results for a Model Predictive Path Integral (MPPI) controller with a learned dynamics model. We train a similar capacity dynamics model to that of TD<sup>2</sup> -CFM before evaluating MPPI with a finite horizon of 32 for locomotion tasks and 128 for maze, where at each step we sample 256 action candidates and perform 10 optimization rounds with 64 elites (top-k actions) per round. The results show that GPI with TD<sup>2</sup> -CFM significantly outperforms MPPI in 3/4 domains with comparable results in Walker. MPPI notably displayed instability related to compounding errors in environments with difficult to model dynamics.

Impact of number of ODE integration steps we report in [Table 10](#page-24-2) an empirical analysis showing how prediction quality degrades as we reduce the number of integration steps on the Loop task in Pointmass Maze. The results show that TD<sup>2</sup> -CFM remains robust even at coarse discretizations of the ODE with as little as 5 integration steps, while we observe with a predictable degradation when the number of steps is too small,

Table 9. Comparison with planning with one-step world models Table 10. Ablation of the number of ODE integration steps

| Domain    |     |    | FB   |    |       |    |     |    |       |    | MPPI  |     |    |   |
|-----------|-----|----|------|----|-------|----|-----|----|-------|----|-------|-----|----|---|
| Cheetah   | 479 | 35 | ( 14 | 56 | ) 693 | 63 | ( 5 | 50 | ) 541 | 22 |       | ( 5 | 28 | ) |
| Pointmass | 472 | 45 | ( 14 | 40 | ) 800 | 99 | ( 8 | 56 | ) 286 | 43 | (     | 54  | 95 | ) |
| Quadruped | 627 | 28 | ( 1  | 98 | ) 695 | 73 | ( 2 | 07 | ) 156 | 80 | ( 122 |     | 89 | ) |
| Walker    | 526 | 66 | ( 5  | 94 | ) 627 | 63 | ( 7 | 97 | ) 658 | 15 | (     | 21  | 46 | ) |

| ODE Steps | NLL   | ↓       | EMD    | ↓        | MSE(VF) | ↓        |
|-----------|-------|---------|--------|----------|---------|----------|
| 2         | -0.48 | ( 0.21) | 0.076  | ( 0.003) | 379.52  | ( 81.75) |
| 5         | -2.75 | ( 0.15) | 0.036  | ( 0.000) | 23.82   | ( 2.05)  |
| 10        | -2.85 | ( 0.17) | 0.025  | ( 0.001) | 7.71    | ( 2.75)  |
| 20        | -2.99 | ( 0.04) | 0.0218 | ( 0.001) | 4.40    | ( 0.82)  |

Table 11. Per task results for the single policy experiments.

| Task            | EMD       | ↓      |    |   |     |       |     | NLL | ↓     |     |     |     |      |        |        |       |      | ↓    | Task EMD ↓ NLL ↓ ↓                                                |
|-----------------|-----------|--------|----|---|-----|-------|-----|-----|-------|-----|-----|-----|------|--------|--------|-------|------|------|-------------------------------------------------------------------|
| TD DD 20        | 06 (      | 0 27   | )  |   |     | 2 713 |     | (   | 0 189 |     | )   |     | 120  | 21     | (      | 52    | 91   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD 11           | 05 (      | 0 01   | )  |   |     | 0 543 |     | (   | 0 164 |     | )   |     | 24   | 02     | ( 25   |       | 54   | )    |                                                                   |
| TD CFM 12       | 46 (      | 0 35   | )  |   |     | 0 608 |     | (   | 0 026 |     | )   |     | 148  | 56     | (      | 29    | 24   | )    |                                                                   |
| TD CFM ( C ) 10 | 90 (      | 0 05   | )  |   |     | 0 112 |     | (   | 0 018 |     | )   |     |      | 9 53   | (      | 1     | 37   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM 10          | 59        | ( 0 13 | )  |   |     | − 0   | 026 |     | ( 0   | 005 |     | )   |      | 11 90  | (      | 7     | 59   | )    |                                                                   |
| GAN 23          | 99 (      | 1 15   | )  |   |     |       |     | —   |       |     |     |     | 827  | 79     | ( 130  |       |      | 38   | )                                                                 |
| VAE 114         | 95        | ( 2    | 51 | ) |     |       |     | —   |       |     |     |     | 646  | 96     | (      | 21    | 57   | )    |                                                                   |
| TD DD 21        | 55 (      | 0 17   | )  |   |     | 2 754 |     | (   | 0 062 |     | )   |     | 1812 | 90     | ( 2016 |       |      | 68   | )                                                                 |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD 13           | 57 (      | 0 02   | )  |   |     | 0 561 |     | (   | 0 014 |     | )   |     |      | 21 81  | (      | 9     | 55   | )    |                                                                   |
| TD CFM 15       | 46 (      | 0 26   | )  |   |     | 0 838 |     | (   | 0 021 |     | )   |     | 379  | 13     | ( 180  |       |      | 63   | )                                                                 |
| TD CFM ( C ) 12 | 94        | ( 0 08 | )  |   |     | 0 321 |     | (   | 0 009 |     | )   |     |      | 22 36  | (      | 4     | 99   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM 13          | 27 (      | 0 15   | )  |   |     | 0     | 200 | (   | 0 007 |     | )   |     |      | 7 14   | (      | 1     | 72   | )    |                                                                   |
| GAN 26          | 85 (      | 1 98   | )  |   |     |       |     | —   |       |     |     |     | 2948 | 74     | ( 4541 |       |      | 66   | )                                                                 |
| VAE 103         | 73        | ( 6    | 86 | ) |     |       |     | —   |       |     |     |     | 431  | 70     | (      | 87    | 04   | )    |                                                                   |
| TD DD 19        | 82 (      | 0 07   | )  |   |     | 2 579 |     | (   | 0 180 |     | )   |     | 56   | 11     | ( 18   |       | 32   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD 12           | 15        | ( 0 20 | )  |   |     | 0 487 |     | (   | 0 040 |     | )   |     |      | 21 65  | (      | 4     | 76   | )    |                                                                   |
| TD CFM 13       | 30 (      | 0 17   | )  |   |     | 0 669 |     | (   | 0 041 |     | )   |     |      | 32 95  | (      | 8     | 33   | )    |                                                                   |
| TD CFM ( C ) 12 | 25        | ( 0 11 | )  |   |     | 0 218 |     | (   | 0 002 |     | )   |     |      | 12 76  | (      | 3     | 17   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM 12          | 27        | ( 0 14 | )  |   |     | 0     | 126 | (   | 0 019 |     | )   |     | 14   | 96     | ( 10   |       | 23   | )    |                                                                   |
| GAN 22          | 98 (      | 1 31   | )  |   |     |       |     | —   |       |     |     |     | 5041 | 85     | (      | 654   |      | 87   | )                                                                 |
| VAE 114         | 46        | ( 0    | 28 | ) |     |       |     | —   |       |     |     |     | 3863 | 70     |        | ( 38  |      | 24   | )                                                                 |
| TD DD 21        | 29 (      | 0 46   | )  |   |     | 2 635 |     | (   | 0 072 |     | )   |     | 121  | 50     | (      | 34    | 67   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD 11           | 59        | ( 0 34 | )  |   |     | 0 558 |     | (   | 0 080 |     | )   |     | 88   | 67     | ( 28   |       | 94   | )    |                                                                   |
| TD CFM 12       | 91 (      | 0 16   | )  |   |     | 0 738 |     | (   | 0 084 |     | )   |     | 340  | 43     | (      | 63    | 65   | )    |                                                                   |
| TD CFM ( C ) 11 | 55        | ( 0 02 | )  |   |     | 0 225 |     | (   | 0 036 |     | )   |     | 78   | 20     | (      | 14    | 20   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM 11          | 54        | ( 0 27 | )  |   |     | 0     | 118 | (   | 0 022 |     | )   |     | 79   | 41     | ( 16   |       | 24   | )    |                                                                   |
| GAN 24          | 21 (      | 1 43   | )  |   |     |       |     | —   |       |     |     |     | 5944 | 23     | (      | 302   |      | 73   | )                                                                 |
| VAE 113         | 79        | ( 1    | 65 | ) |     |       |     | —   |       |     |     |     | 4888 | 07     |        | ( 78  |      | 85   | )                                                                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 27 89 ( 0 67 ) 1 890 ( 0 025 ) 1778 78 ( 611 15 )           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 25 62 ( 3 75 ) 0 906 ( 0 013 ) 12 88 ( 2 07 )                  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 15 68 ( 0 15 ) 1 068 ( 0 006 ) 523 10 ( 42 47 )            |
| RUN             |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 14 12 ( 0 00 ) 0 518 ( 0 002 ) 10 10 ( 1 32 ) JUMP   |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 14 27 ( 0 06 ) 0 426 ( 0 005 ) 12 89 ( 2 86 )                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 18 23 ( 0 34 ) — 3546 34 ( 984 61 )                           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 60 54 ( 0 29 ) — 1939 62 ( 22 15 )                            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 28 01 ( 1 02 ) 1 975 ( 0 061 ) 438 92 ( 310 44 )            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 22 79 ( 3 08 ) 0 856 ( 0 033 ) 32 38 ( 4 36 )                  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 15 74 ( 0 05 ) 1 051 ( 0 026 ) 170 86 ( 19 61 )            |
| SPIN            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 14 62 ( 0 11 ) 0 457 ( 0 006 ) 26 01 ( 4 44 ) RUN    |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 14 75 ( 0 05 ) 0 338 ( 0 004 ) 18 36 ( 2 62 )                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 19 21 ( 0 13 ) — 195 11 ( 144 29 )                            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 60 56 ( 0 21 ) — 428 69 ( 10 48 )                             |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 28 57 ( 0 50 ) 1 832 ( 0 034 ) 2083 77 ( 1767 03 )          |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 20 81 ( 1 81 ) 0 867 ( 0 040 ) 20 09 ( 19 08 )                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 15 03 ( 0 18 ) 1 003 ( 0 026 ) 505 51 ( 88 47 )            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 13 91 ( 0 02 ) 0 483 ( 0 005 ) 12 86 ( 4 65 ) STAND  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 14 07 ( 0 12 ) 0 393 ( 0 021 ) 7 77 ( 0 91 )                  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 91273 39 ( 81559 61 ) — 3631 15 ( 2289 14 )                   |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 59 42 ( 0 49 ) — 859 51 ( 101 82 )                            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 28 83 ( 0 41 ) 1 934 ( 0 075 ) 1661 52 ( 402 07 )           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 21 36 ( 1 70 ) 0 815 ( 0 040 ) 570 75 ( 35 38 )                |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 16 48 ( 0 09 ) 1 103 ( 0 006 ) 900 78 ( 85 36 )            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 14 89 ( 0 01 ) 0 494 ( 0 006 ) 572 02 ( 24 55 ) WALK |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
| WALK            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 14 96 ( 0 13 ) 0 361 ( 0 022 ) 528 06 ( 11 32 )               |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 55777 67 ( 28193 15 ) — 3166 15 ( 54 62 )                     |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 60 57 ( 0 54 ) — 1397 52 ( 100 28 )                           |
|                 | Pointmass |        |    |   |     | Maze  |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| Task Method     |           | EMD    |    |   | ↓   |       |     |     | NLL   |     | ↓   |     |      | MSE(V) |        |       |      | ↓    |                                                                   |
| TD DD           | 0         | 189    | (  | 0 | 003 | )     | 3   | 462 |       | ( 0 | 232 |     | )    | 4717   | 87     | (     | 83   | 53   | )                                                                 |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD              | 0         | 031    | (  | 0 | 003 | )     | 0   | 577 |       | ( 0 | 027 |     | )    | 4      | 27     | ( 1   | 36   | )    |                                                                   |
| TD CFM          | 0         | 071    | (  | 0 | 007 | )     | 0   | 748 |       | ( 0 | 070 |     | )    | 677    | 48     | ( 154 |      | 81   | )                                                                 |
| TD CFM ( C      | ) 0       | 025    | (  | 0 | 002 | )     | −   | 0   | 703   | (   | 0   | 032 | )    | 10     | 91     | (     | 2 35 | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM             | 0         | 020    | (  | 0 | 001 | )     | −   | 0   | 674   | (   | 0   | 072 | )    | 1      | 75     | ( 0   | 13   | )    |                                                                   |
| GAN             | 0         | 225    | (  | 0 | 014 | )     |     |     | —     |     |     |     |      | 2276   | 26     | (     | 361  | 04   | )                                                                 |
| VAE             | 0         | 456    | (  | 0 | 045 | )     |     |     | —     |     |     |     |      | 4011   | 19     | (     | 85   | 44   | )                                                                 |
| BOTTOM LEFT     |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| TD DD           | 0         | 139    | (  | 0 | 002 | )     | 2   | 808 |       | ( 0 | 058 |     | )    | 320    | 80     | (     | 27   | 06   | )                                                                 |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD              | 0         | 025    | (  | 0 | 001 | )     | 0   | 980 |       | ( 0 | 174 |     | )    | 5      | 76     | ( 3   | 15   | )    |                                                                   |
| TD CFM          | 0         | 059    | (  | 0 | 001 | )     | 0   | 520 |       | ( 0 | 031 |     | )    | 224    | 13     | (     | 33   | 19   | )                                                                 |
| TD CFM ( C      | ) 0       | 024    | (  | 0 | 002 | )     | −   | 0   | 729   | (   | 0   | 167 | )    | 16     | 58     | ( 12  |      | 10 ) |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM             | 0         | 020    | (  | 0 | 002 | )     | −   | 0   | 984   | (   | 0   | 053 | )    | 10     | 44     | (     | 7 08 | )    |                                                                   |
| GAN             | 0         | 269    | (  | 0 | 150 | )     |     |     | —     |     |     |     |      | 1199   | 80     | (     | 212  | 47   | )                                                                 |
| VAE             | 0         | 313    | (  | 0 | 029 | )     |     |     | —     |     |     |     |      | 981    | 22     | ( 195 |      | 70   | )                                                                 |
| BOTTOM RIGHT    |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| TD DD           | 0         | 174    | (  | 0 | 004 | )     | 3   | 270 |       | ( 0 | 257 |     | )    | 230    | 79     | (     | 18   | 24   | )                                                                 |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD              | 0         | 025    | (  | 0 | 001 | )     | 0   | 640 |       | ( 0 | 283 |     | )    | 4      | 82     | ( 2   | 61   | )    |                                                                   |
| TD CFM          | 0         | 066    | (  | 0 | 004 | )     | 0   | 549 |       | ( 0 | 040 |     | )    | 166    | 07     | (     | 35   | 75   | )                                                                 |
| TD CFM ( C      | ) 0       | 023    | (  | 0 | 001 | )     | −   | 0   | 759   | (   | 0   | 034 | )    | 10     | 95     | (     | 2 63 | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM             | 0         | 020    | (  | 0 | 002 | )     | −   | 0   | 855   | (   | 0   | 022 | )    | 4      | 84     | ( 3   | 08   | )    |                                                                   |
| GAN             | 0         | 170    | (  | 0 | 018 | )     |     |     | —     |     |     |     |      | 416    | 75     | (     | 54   | 72   | )                                                                 |
| VAE             | 0         | 505    | (  | 0 | 051 | )     |     |     | —     |     |     |     |      | 489    | 06     | (     | 6    | 44 ) |                                                                   |
| TOP LEFT        |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| TD DD           | 0         | 102    | (  | 0 | 001 | )     | 2   | 407 |       | ( 0 | 059 |     | )    | 593    | 98     | (     | 72   | 33   | )                                                                 |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD              | 0         | 033    | (  | 0 | 003 | )     | 0   | 863 |       | ( 0 | 255 |     | )    | 34     | 43     | ( 10  |      | 96 ) |                                                                   |
| TD CFM          | 0         | 055    | (  | 0 | 006 | )     | 0   | 454 |       | ( 0 | 167 |     | )    | 472    | 54     | ( 308 |      | 65   | )                                                                 |
| TD CFM ( C      | ) 0       | 021    | (  | 0 | 003 | )     | −   | 0   | 517   | (   | 0   | 445 | )    | 14     | 85     | (     | 3    | 28 ) |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM             | 0         | 025    | (  | 0 | 002 | )     | −   | 0   | 797   | (   | 0   | 057 | )    | 23     | 48     | (     | 5 46 | )    |                                                                   |
| GAN             | 0         | 132    | (  | 0 | 022 | )     |     |     | —     |     |     |     |      | 1350   | 49     | (     | 716  | 52   | )                                                                 |
| VAE             | 0         | 321    | (  | 0 | 029 | )     |     |     | —     |     |     |     |      | 2404   | 42     | (     | 498  | 13   | )                                                                 |
| TOP RIGHT       |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| TD DD           | 0         | 141    | (  | 0 | 002 | )     | 2   | 924 |       | ( 0 | 243 |     | )    | 362    | 56     | (     | 8    | 06 ) |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| DD              | 0         | 023    | (  | 0 | 003 | )     | 0   | 743 |       | ( 0 | 259 |     | )    | 6      | 38     | ( 1   | 55   | )    |                                                                   |
| TD CFM          | 0         | 059    | (  | 0 | 002 | )     | 0   | 501 |       | ( 0 | 018 |     | )    | 237    | 57     | (     | 47   | 18   | )                                                                 |
| TD CFM ( C      | ) 0       | 020    | (  | 0 | 001 | )     | −   | 0   | 771   | (   | 0   | 090 | )    | 6      | 18     | ( 3   | 37   | )    |                                                                   |
| TD 2            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      |                                                                   |
| CFM             | 0         | 018    | (  | 0 | 001 | )     | −   | 0   | 903   | (   | 0   | 074 | )    | 3      | 21     | ( 2   | 22   | )    |                                                                   |
| GAN             | 0         | 218    | (  | 0 | 044 | )     |     |     | —     |     |     |     |      | 1043   | 01     | (     | 337  | 10   | )                                                                 |
| VAE             | 0         | 453    | (  | 0 | 106 | )     |     |     | —     |     |     |     |      | 1223   | 57     | (     | 80   | 69   | )                                                                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | Task Method EMD ↓ NLL ↓ MSE(V) ↓                                  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 20 31 ( 0 31 ) 2 669 ( 0 086 ) 601 62 ( 314 84 )            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 14 44 ( 1 79 ) 0 758 ( 0 028 ) 172 03 ( 35 51 )                |
| LOOP            |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 11 90 ( 0 03 ) 0 868 ( 0 008 ) 211 92 ( 26 25 )            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 10 55 ( 0 03 ) 0 485 ( 0 024 ) 124 08 ( 17 89 ) FLIP |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 10 67 ( 0 04 ) 0 447 ( 0 021 ) 67 76 ( 21 99 )                |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 23 55 ( 2 52 ) — 3608 55 ( 1948 65 )                          |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 83 00 ( 1 02 ) — 3339 01 ( 44 80 )                            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 16 67 ( 0 02 ) 2 647 ( 0 186 ) 1043 27 ( 369 92 )           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
| REACH           |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 12 99 ( 2 64 ) 0 894 ( 0 025 ) 463 04 ( 89 08 )                |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 10 91 ( 0 12 ) 0 927 ( 0 047 ) 398 66 ( 59 04 ) FLIP       |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 9 90 ( 0 07 ) 0 542 ( 0 023 ) 410 49 ( 77 16 )       |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 10 11 ( 0 14 ) 0 542 ( 0 006 ) 370 69 ( 112 59 )              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 20 80 ( 1 56 ) — 3761 79 ( 785 37 )                           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 84 65 ( 0 31 ) — 918 32 ( 25 62 )                             |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 20 26 ( 0 06 ) 2 907 ( 0 336 ) 46 48 ( 13 06 )              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
| REACH           |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 16 91 ( 4 04 ) 0 813 ( 0 028 ) 86 53 ( 55 44 )                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 12 21 ( 0 05 ) 0 872 ( 0 032 ) 54 98 ( 11 01 )             |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 10 44 ( 0 08 ) 0 434 ( 0 018 ) 24 52 ( 5 89 ) RUN    |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 10 53 ( 0 08 ) 0 412 ( 0 020 ) 27 69 ( 5 44 )                 |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 25 48 ( 2 01 ) — 183 47 ( 72 39 )                             |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 83 91 ( 0 57 ) — 109 45 ( 9 86 )                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 21 47 ( 0 32 ) 3 074 ( 0 376 ) 20 28 ( 5 95 )               |
| REACH           |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 13 04 ( 1 22 ) 0 818 ( 0 016 ) 14 87 ( 2 34 )                  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 13 38 ( 0 20 ) 0 989 ( 0 056 ) 37 90 ( 2 98 ) RUN          |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 11 02 ( 0 05 ) 0 452 ( 0 023 ) 8 71 ( 1 05 )         |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 11 06 ( 0 08 ) 0 414 ( 0 016 ) 8 33 ( 1 89 )                  |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 24 77 ( 0 43 ) — 270 21 ( 4 08 )                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 82 91 ( 0 36 ) — 734 77 ( 22 94 )                             |
| REACH           |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 21 57 ( 0 84 ) 2 790 ( 0 151 ) 546 05 ( 86 30 )             |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 12 85 ( 1 67 ) 0 780 ( 0 047 ) 238 01 ( 11 17 )                |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 12 27 ( 0 12 ) 0 802 ( 0 034 ) 377 45 ( 101 61 )           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 10 24 ( 0 17 ) 0 354 ( 0 021 ) 176 99 ( 28 54 ) WALK |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 10 18 ( 0 08 ) 0 336 ( 0 021 ) 229 89 ( 21 93 )               |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 24 39 ( 1 11 ) — 3520 88 ( 1050 76 )                          |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 84 39 ( 0 41 ) — 2138 32 ( 233 01 )                           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD DD 21 05 ( 0 30 ) 2 854 ( 0 094 ) 469 23 ( 133 50 )            |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | DD 14 64 ( 2 48 ) 0 771 ( 0 019 ) 160 42 ( 42 25 )                |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM 12 89 ( 0 14 ) 0 857 ( 0 033 ) 291 71 ( 66 89 ) WALK       |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD CFM ( C ) 10 88 ( 0 01 ) 0 412 ( 0 023 ) 99 90 ( 4 20 )        |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | TD 2                                                              |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | CFM 10 86 ( 0 12 ) 0 381 ( 0 014 ) 106 97 ( 10 45 )               |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | GAN 24 86 ( 0 34 ) — 3434 43 ( 189 45 )                           |
|                 |           |        |    |   |     |       |     |     |       |     |     |     |      |        |        |       |      |      | VAE 83 73 ( 0 63 ) — 465 72 ( 16 06 )                             |

Table 12. Results averaged over tasks for the single policy experiments with a curved conditional path.

| Domain         | Method               | EMD $\downarrow$ | NLD $\downarrow$ | MSE(V) $\downarrow$ |
|----------------|----------------------|------------------|------------------|---------------------|
| CHEETAH        | TD-CFM               | 13.91 (0.73)     | 1.354 (0.017)    | 477.89 (16.07)      |
|                | TD-CFM(C)            | 25.86 (18.91)    | 1.295 (0.067)    | 189.21 (17.69)      |
|                | TD <sup>2</sup> -CFM | 10.79 (0.03)     | 0.412 (0.014)    | 121.67 (5.68)       |
| POINTMASS MAZE | TD-CFM               | 0.901 (0.003)    | 1.156 (0.081)    | 578.16 (103.54)     |
|                | TD-CFM(C)            | 0.808 (0.006)    | 4.340 (0.456)    | 679.29 (20.11)      |
|                | TD <sup>2</sup> -CFM | 0.021 (0.000)    | -0.806 (0.017)   | 9.22 (1.42)         |
| QUADRUPED      | TD-CFM               | 15.63 (0.09)     | 1.478 (0.088)    | 273.68 (34.07)      |
|                | TD-CFM(C)            | 34.00 (6.96)     | 0.930 (0.036)    | 522.28 (155.42)     |
|                | TD <sup>2</sup> -CFM | 14.56 (0.02)     | 0.327 (0.014)    | 142.18 (9.38)       |
| WALKER         | TD-CFM               | 13.00 (0.11)     | 1.147 (0.042)    | 608.47 (124.69)     |
|                | TD-CFM(C)            | 33.20 (6.04)     | 1.039 (0.057)    | 189.66 (12.02)      |
|                | TD <sup>2</sup> -CFM | 12.00 (0.05)     | 0.099 (0.005)    | 27.56 (0.53)        |

Table 13. Per task results for the single policy experiments with a curved conditional path.

| Task     |         |      |    | EMD  |   | ↓  |     |       | NLL | ↓     |       |    |       |      | ↓    |   |
|----------|---------|------|----|------|---|----|-----|-------|-----|-------|-------|----|-------|------|------|---|
| TD       | CFM     | 12   | 39 | (    | 0 | 17 | ) 1 | 218   | ( 0 | 107   | ) 326 | 14 | (     | 56   | 12   | ) |
| TD RUN   | CFM ( C | ) 23 | 80 | (    | 4 | 91 | ) 0 | 923   | ( 0 | 191   | ) 69  | 39 | (     | 8    | 84 ) |   |
| TD       | 2       |      |    |      |   |    |     |       |     |       |       |    |       |      |      |   |
|          | CFM     | 10   | 69 | (    | 0 | 06 | ) − | 0 040 | (   | 0 008 | ) 11  | 69 | (     | 4    | 01   | ) |
| TD       | CFM     | 14   | 08 | (    | 0 | 12 | ) 1 | 410   | ( 0 | 189   | ) 896 | 83 | ( 278 |      | 52   | ) |
| TD SPIN  | CFM ( C | ) 47 | 39 | (    | 3 | 14 | ) 1 | 801   | ( 0 | 186   | ) 401 | 22 | ( 321 |      | 52   | ) |
| TD       | 2       |      |    |      |   |    |     |       |     |       |       |    |       |      |      |   |
|          | CFM     | 13   | 37 | (    | 0 | 11 | ) 0 | 198   | ( 0 | 008   | ) 7   | 65 | (     | 2 29 | )    |   |
| TD       | CFM     | 13   | 24 | (    | 0 | 23 | ) 0 | 896   | ( 0 | 053   | ) 274 | 60 | ( 121 |      | 20   | ) |
| TD STAND | CFM ( C | ) 36 | 32 | ( 13 |   | 80 | ) 0 | 625   | ( 0 | 053   | ) 159 | 74 | (     | 31   | 53   | ) |
| TD       | 2       |      |    |      |   |    |     |       |     |       |       |    |       |      |      |   |
|          | CFM     | 12   | 50 | (    | 0 | 17 | ) 0 | 119   | ( 0 | 008   | ) 9   | 42 | (     | 1 95 | )    |   |
| TD       | CFM     | 12   | 69 | (    | 0 | 20 | ) 1 | 067   | ( 0 | 015   | ) 936 | 30 | (     | 86   | 71   | ) |
| TD WALK  | CFM ( C | ) 25 | 29 | (    | 7 | 62 | ) 0 | 808   | ( 0 | 049   | ) 128 | 28 | (     | 65   | 70   | ) |
| TD       | 2       |      |    |      |   |    |     |       |     |       |       |    |       |      |      |   |
|          | CFM     | 11   | 42 | (    | 0 | 20 | ) 0 | 119   | ( 0 | 026   | ) 81  | 47 | (     | 3    | 53   | ) |

| Task     |         |      |    | EMD | ↓  |     |     | NLL | ↓   |       |    |       |      | ↓  |   |
|----------|---------|------|----|-----|----|-----|-----|-----|-----|-------|----|-------|------|----|---|
| TD       | CFM     | 15   | 31 | ( 0 | 17 | ) 1 | 460 | ( 0 | 188 | ) 115 | 99 | ( 138 |      | 59 | ) |
| TD JUMP  | CFM ( C | ) 39 | 28 | ( 8 | 90 | ) 0 | 980 | ( 0 | 062 | ) 686 | 51 | ( 314 |      | 49 | ) |
| TD       | 2       |      |    |     |    |     |     |     |     |       |    |       |      |    |   |
|          | CFM     | 14   | 36 | ( 0 | 07 | ) 0 | 358 | ( 0 | 010 | ) 10  | 84 | (     | 3    | 05 | ) |
| TD       | CFM     | 15   | 61 | ( 0 | 16 | ) 1 | 450 | ( 0 | 060 | ) 104 | 52 | (     | 33   | 53 | ) |
| TD RUN   | CFM ( C | ) 40 | 27 | ( 7 | 59 | ) 0 | 898 | ( 0 | 040 | ) 240 | 50 | (     | 58   | 83 | ) |
| TD       | 2       |      |    |     |    |     |     |     |     |       |    |       |      |    |   |
|          | CFM     | 14   | 73 | ( 0 | 06 | ) 0 | 288 | ( 0 | 015 | ) 21  | 13 | (     | 3    | 52 | ) |
| TD       | CFM     | 15   | 24 | ( 0 | 11 | ) 1 | 515 | ( 0 | 215 | ) 173 | 07 | (     | 34   | 09 | ) |
| TD STAND | CFM ( C | ) 22 | 77 | ( 6 | 86 | ) 0 | 924 | ( 0 | 053 | ) 275 | 03 | ( 249 |      | 91 | ) |
| TD       | 2       |      |    |     |    |     |     |     |     |       |    |       |      |    |   |
|          | CFM     | 14   | 17 | ( 0 | 09 | ) 0 | 342 | ( 0 | 019 | ) 7   | 05 | (     | 1 80 | )  |   |
| TD       | CFM     | 16   | 37 | ( 0 | 10 | ) 1 | 486 | ( 0 | 022 | ) 701 | 13 | (     | 83   | 58 | ) |
| TD WALK  | CFM ( C | ) 33 | 68 | ( 4 | 69 | ) 0 | 917 | ( 0 | 036 | ) 887 | 11 | ( 120 |      | 92 | ) |
| TD       | 2       |      |    |     |    |     |     |     |     |       |    |       |      |    |   |
|          | CFM     | 14   | 99 | ( 0 | 08 | ) 0 | 318 | ( 0 | 016 | ) 529 | 71 | (     | 35   | 40 | ) |

| Task         |       |   | EMD |     | ↓     | Maze |     | NLL |   | ↓   |       |      |      |    |     |     | ↓    |
|--------------|-------|---|-----|-----|-------|------|-----|-----|---|-----|-------|------|------|----|-----|-----|------|
| TD           | CFM   | 0 | 112 | ( 0 | 015 ) | 1    | 465 | (   | 0 | 171 | )     | 1888 |      | 54 | (   | 444 | 66 ) |
| TD CFM LOOP  | ( C ) | 0 | 132 | ( 0 | 031 ) | 5    | 191 | (   | 1 | 328 | )     | 1354 |      | 09 | (   | 102 | 55 ) |
| TD 2         |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
|              | CFM   | 0 | 020 | ( 0 | 000   | ) −  | 0   | 708 | ( | 0   | 013 ) |      | 2 31 |    | ( 0 | 59  | )    |
| BOTTOM LEFT  |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
| TD REACH     | CFM   | 0 | 096 | ( 0 | 012 ) | 1    | 091 | (   | 0 | 142 | )     | 628  | 74   | (  |     | 118 | 04 ) |
| TD CFM       | ( C ) | 0 | 078 | ( 0 | 001 ) | 3    | 942 | (   | 0 | 576 | )     | 820  |      | 02 | (   | 52  | 88 ) |
| TD 2         |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
|              | CFM   | 0 | 022 | ( 0 | 001   | ) −  | 0   | 883 | ( | 0   | 057 ) |      | 10   | 55 | ( 9 | 13  | )    |
| BOTTOM RIGHT |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
| TD REACH     | CFM   | 0 | 097 | ( 0 | 001 ) | 1    | 296 | (   | 0 | 220 | )     | 290  |      | 21 | (   | 29  | 94 ) |
| TD CFM       | ( C ) | 0 | 109 | ( 0 | 009 ) | 5    | 310 | (   | 0 | 552 | )     | 409  |      | 28 | (   | 10  | 79 ) |
| TD 2         |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
|              | CFM   | 0 | 019 | ( 0 | 001   | ) −  | 0   | 833 | ( | 0   | 049 ) |      | 2 64 |    | ( 0 | 30  | )    |
| TOP LEFT     |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
| TD REACH     | CFM   | 0 | 070 | ( 0 | 003 ) | 0    | 894 | (   | 0 | 139 | )     | 500  | 63   | (  |     | 142 | 18 ) |
| TD CFM       | ( C ) | 0 | 048 | ( 0 | 002 ) | 2    | 821 | (   | 0 | 268 | )     | 75   | 79   | (  | 20  | 06  | )    |
| TD 2         |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
|              | CFM   | 0 | 025 | ( 0 | 002   | ) −  | 0   | 738 | ( | 0   | 011 ) |      | 26   | 56 | ( 9 | 99  | )    |
| TOP RIGHT    |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
| TD REACH     | CFM   | 0 | 083 | ( 0 | 004 ) | 1    | 035 | (   | 0 | 138 | )     | 482  | 68   | (  |     | 128 | 45 ) |
| TD CFM       | ( C ) | 0 | 080 | ( 0 | 001 ) | 4    | 436 | (   | 0 | 305 | )     | 737  |      | 30 | (   | 23  | 75 ) |
| TD 2         |       |   |     |     |       |      |     |     |   |     |       |      |      |    |     |     |      |
|              | CFM   | 0 | 019 | ( 0 | 001   | ) −  | 0   | 866 | ( | 0   | 026 ) |      | 4 02 |    | ( 1 | 75  | )    |

| Task    |         |      |    | EMD  |   | ↓  |     |     | NLL | ↓   |        |    |       |     | ↓    |   |
|---------|---------|------|----|------|---|----|-----|-----|-----|-----|--------|----|-------|-----|------|---|
| TD      | CFM     | 12   | 92 | (    | 1 | 25 | ) 1 | 324 | ( 0 | 042 | ) 342  | 71 | ( 129 |     | 09   | ) |
| TD FLIP | CFM ( C | ) 22 | 90 | ( 15 |   | 00 | ) 1 | 364 | ( 0 | 108 | ) 140  | 32 | (     | 42  | 14   | ) |
| TD      | 2       |      |    |      |   |    |     |     |     |     |        |    |       |     |      |   |
|         | CFM     | 10   | 89 | (    | 0 | 08 | ) 0 | 433 | ( 0 | 012 | ) 74   | 34 | (     | 6   | 50 ) |   |
| TD FLIP | CFM     | 14   | 52 | (    | 4 | 08 | ) 1 | 346 | ( 0 | 190 | ) 576  | 31 | ( 169 |     | 57   | ) |
| TD      | CFM ( C | ) 25 | 46 | ( 25 |   | 58 | ) 1 | 427 | ( 0 | 027 | ) 388  | 45 | (     | 87  | 18   | ) |
| TD      | 2       |      |    |      |   |    |     |     |     |     |        |    |       |     |      |   |
|         | CFM     | 10   | 48 | (    | 0 | 23 | ) 0 | 538 | ( 0 | 034 | ) 283  | 84 | (     | 40  | 81   | ) |
| TD      | CFM     | 14   | 00 | (    | 0 | 77 | ) 1 | 390 | ( 0 | 043 | ) 114  | 51 |       | ( 3 | 11   | ) |
| TD RUN  | CFM ( C | ) 17 | 42 | (    | 5 | 78 | ) 1 | 423 | ( 0 | 091 | ) 37   | 23 | (     | 8   | 74 ) |   |
| TD      | 2       |      |    |      |   |    |     |     |     |     |        |    |       |     |      |   |
|         | CFM     | 10   | 85 | (    | 0 | 08 | ) 0 | 405 | ( 0 | 010 | ) 32   | 58 | (     | 8   | 42 ) |   |
| TD RUN  | CFM     | 14   | 50 | (    | 0 | 31 | ) 1 | 439 | ( 0 | 102 | ) 109  | 32 |       | ( 5 | 35   | ) |
| TD      | CFM ( C | ) 38 | 06 | ( 28 |   | 90 | ) 1 | 283 | ( 0 | 110 | ) 101  | 24 | ( 149 |     | 88   | ) |
| TD      | 2       |      |    |      |   |    |     |     |     |     |        |    |       |     |      |   |
|         | CFM     | 11   | 06 | (    | 0 | 05 | ) 0 | 399 | ( 0 | 007 | ) 12   | 32 | (     | 2   | 34 ) |   |
| TD      | CFM     | 13   | 66 | (    | 0 | 71 | ) 1 | 290 | ( 0 | 041 | ) 1040 | 43 | (     | 147 | 86   | ) |
| TD WALK | CFM ( C | ) 21 | 01 | ( 16 |   | 43 | ) 1 | 096 | ( 0 | 028 | ) 343  | 71 | (     | 66  | 91   | ) |
| TD      | 2       |      |    |      |   |    |     |     |     |     |        |    |       |     |      |   |
|         | CFM     | 10   | 45 | (    | 0 | 04 | ) 0 | 323 | ( 0 | 010 | ) 213  | 87 | (     | 23  | 09   | ) |
| TD WALK | CFM     | 13   | 83 | (    | 0 | 89 | ) 1 | 336 | ( 0 | 033 | ) 684  | 05 | (     | 21  | 31   | ) |
| TD      | CFM ( C | ) 30 | 29 | ( 22 |   | 58 | ) 1 | 178 | ( 0 | 206 | ) 124  | 29 | (     | 17  | 61   | ) |
| TD      | 2       |      |    |      |   |    |     |     |     |     |        |    |       |     |      |   |
|         | CFM     | 11   | 00 | (    | 0 | 11 | ) 0 | 372 | ( 0 | 015 | ) 113  | 09 | (     | 22  | 45   | ) |

Table 14. Per domain results for the quantitative multipolicy experiments.

| Domain     | Method EMD ↓ NLL ↓ MSE(V) ↓                                 |
|------------|-------------------------------------------------------------|
| C HEETAH   |                                                             |
| TD         | DD 17 79 ( 0 40 ) 1 442 ( 0 042 ) 534 82 ( 107 81 )         |
| TD         | 2                                                           |
|            | DD 74 35 ( 7 49 ) 0 771 ( 0 020 ) 253 89 ( 21 42 )          |
| TD         | CFM 12 54 ( 0 04 ) 1 044 ( 0 044 ) 826 54 ( 58 01 )         |
| TD         | CFM ( C ) 11 19 ( 0 11 ) 0 581 ( 0 011 ) 249 02 ( 19 81 )   |
| TD         | 2                                                           |
|            | CFM 11 06 ( 0 08 ) 0 481 ( 0 008 ) 230 34 ( 44 81 )         |
| P OINTMASS |                                                             |
| TD         | DD 0 152 ( 0 006 ) 2 048 ( 0 093 ) 662 96 ( 76 86 )         |
| TD         | 2                                                           |
|            | DD 0 349 ( 0 037 ) 0 666 ( 0 027 ) 312 98 ( 66 46 )         |
| TD         | CFM 0 087 ( 0 003 ) 0 771 ( 0 025 ) 580 94 ( 41 28 )        |
| TD         | CFM ( C ) 0 063 ( 0 000 ) 0 174 ( 0 021 ) 220 11 ( 100 36 ) |
| TD         | 2                                                           |
|            | CFM 0 060 ( 0 002 ) 0 043 ( 0 022 ) 169 74 ( 85 76 )        |
| Q UADRUPED |                                                             |
| TD         | DD 20 21 ( 1 76 ) 1 403 ( 0 022 ) 499 88 ( 292 17 )         |
| TD         | 2                                                           |
|            | DD 135 79 ( 9 24 ) 0 901 ( 0 051 ) 415 29 ( 101 86 )        |
| TD         | CFM 15 06 ( 0 08 ) 0 950 ( 0 024 ) 391 12 ( 141 00 )        |
| TD         | CFM ( C ) 14 98 ( 0 15 ) 0 528 ( 0 016 ) 176 62 ( 13 73 )   |
| TD         | 2                                                           |
|            | CFM 14 74 ( 0 12 ) 0 340 ( 0 010 ) 178 95 ( 30 43 )         |
| W ALKER    |                                                             |
| TD         | DD 21 49 ( 0 64 ) 1 441 ( 0 009 ) 571 72 ( 196 76 )         |
| TD         | 2                                                           |
|            | DD 104 44 ( 2 84 ) 0 688 ( 0 009 ) 180 45 ( 47 82 )         |
| TD         | CFM 15 08 ( 0 28 ) 0 920 ( 0 023 ) 768 13 ( 66 48 )         |
| TD         | CFM ( C ) 13 57 ( 0 09 ) 0 414 ( 0 019 ) 179 39 ( 24 52 )   |
| TD         | 2                                                           |
|            | CFM 13 70 ( 0 33 ) 0 307 ( 0 008 ) 154 75 ( 8 70 )          |

Table 15. Per domain results for the multi-policy experiments evaluating planning performance with generalized policy improvement.

| Domain     | Method Planner Z-Distribution D ( Z ) Random Local Perturbation Train Distribution FB — 479 35 ( 14 56 ) |
|------------|----------------------------------------------------------------------------------------------------------|
| C HEETAH   |                                                                                                          |
|            | FB GPI 275 32 ( 2 50 ) 401 08 ( 5 92 ) 269 59 ( 8 18 )                                                   |
| TD         | DD GPI 574 05 ( 3 88 ) 604 53 ( 11 87 ) 620 72 ( 14 29 )                                                 |
| TD         | 2                                                                                                        |
|            | DD GPI 662 17 ( 0 94 ) 680 22 ( 5 98 ) 678 98 ( 3 67 )                                                   |
| TD         | CFM GPI 403 54 ( 81 24 ) 426 46 ( 81 69 ) 372 40 ( 99 68 )                                               |
| TD         | CFM ( C ) GPI 681 52 ( 6 49 ) 700 97 ( 6 57 ) 697 81 ( 3 16 )                                            |
| TD         | 2                                                                                                        |
|            | CFM GPI 682 21 ( 5 41 ) 692 72 ( 7 96 ) 693 63 ( 5 50 )                                                  |
| P OINTMASS |                                                                                                          |
|            | FB — 472 45 ( 14 40 )                                                                                    |
|            | FB GPI − 0 64 ( 7 70 ) 240 54 ( 23 69 ) − 17 74 ( 4 34 )                                                 |
| TD         | DD GPI 569 05 ( 37 58 ) 599 92 ( 37 26 ) 537 69 ( 47 54 )                                                |
| TD         | 2                                                                                                        |
|            | DD GPI 763 95 ( 38 02 ) 805 72 ( 2 23 ) 788 87 ( 17 13 )                                                 |
| TD         | CFM GPI 625 44 ( 23 12 ) 671 53 ( 52 75 ) 695 70 ( 27 88 )                                               |
| TD         | CFM ( C ) GPI 800 87 ( 3 46 ) 812 44 ( 1 58 ) 808 03 ( 2 77 )                                            |
| TD         | 2                                                                                                        |
|            | CFM GPI 790 34 ( 14 16 ) 813 90 ( 1 62 ) 800 99 ( 8 56 )                                                 |
| Q UADRUPED |                                                                                                          |
|            | FB — 627 28 ( 1 98 )                                                                                     |
|            | FB GPI 671 95 ( 0 58 ) 674 09 ( 0 53 ) 646 05 ( 2 28 )                                                   |
| TD         | DD GPI 657 98 ( 1 87 ) 662 29 ( 1 46 ) 657 44 ( 4 71 )                                                   |
| TD         | 2                                                                                                        |
|            | DD GPI 667 24 ( 6 32 ) 671 54 ( 1 40 ) 665 52 ( 5 12 )                                                   |
| TD         | CFM GPI 669 35 ( 5 82 ) 672 46 ( 4 96 ) 668 61 ( 5 74 )                                                  |
| TD         | CFM ( C ) GPI 695 52 ( 4 51 ) 697 65 ( 5 21 ) 696 18 ( 3 29 )                                            |
| TD         | 2                                                                                                        |
|            | CFM GPI 696 58 ( 4 10 ) 696 57 ( 2 36 ) 695 73 ( 2 07 )                                                  |
| W ALKER    |                                                                                                          |
|            | FB — 526 66 ( 5 94 )                                                                                     |
|            | FB GPI 35 23 ( 0 98 ) 37 51 ( 1 20 ) 39 04 ( 1 48 )                                                      |
| TD         | DD GPI 512 65 ( 19 19 ) 553 35 ( 14 28 ) 533 37 ( 27 24 )                                                |
| TD         | 2                                                                                                        |
|            | DD GPI 509 39 ( 10 26 ) 598 40 ( 6 44 ) 609 28 ( 5 87 )                                                  |
| TD         | CFM GPI 506 62 ( 15 84 ) 524 34 ( 4 75 ) 537 24 ( 17 20 )                                                |
| TD         | CFM ( C ) GPI 513 24 ( 17 77 ) 608 80 ( 16 14 ) 624 19 ( 19 45 )                                         |
| TD         | 2                                                                                                        |
|            | CFM GPI 518 07 ( 20 74 ) 617 08 ( 6 55 ) 627 63 ( 7 97 )                                                 |

Table 16. Per task results for the quantitative multi-policy experiments.

| Walker               |                      |               |               |                  |  |
|----------------------|----------------------|---------------|---------------|------------------|--|
| Task                 | Method               | EMD ↓         | NLL ↓         | MSE(V) ↓         |  |
| FLIP                 | TD-DD                | 24.22 (0.37)  | 1.595 (0.021) | 494.85 (221.39)  |  |
|                      | TD <sup>2</sup> -DD  | 108.16 (1.64) | 0.893 (0.065) | 103.71 (34.77)   |  |
| TD-CFM               | TD-CFM               | 16.01 (0.33)  | 1.120 (0.037) | 431.62 (64.40)   |  |
|                      | TD-CFM(C)            | 14.77 (0.38)  | 0.704 (0.083) | 74.42 (13.13)    |  |
| TD <sup>2</sup> -CFM | TD <sup>2</sup> -CFM | 14.81 (0.56)  | 0.546 (0.012) | 73.86 (26.41)    |  |
|                      | TD-DD                | 21.28 (0.97)  | 1.389 (0.055) | 53.28 (20.52)    |  |
| RUN                  | TD <sup>2</sup> -DD  | 102.69 (3.60) | 0.546 (0.070) | 6.35 (0.88)      |  |
|                      | TD-CFM               | 14.99 (0.65)  | 0.845 (0.085) | 209.80 (54.21)   |  |
| TD-CFM(C)            | TD-CFM(C)            | 13.01 (0.35)  | 0.260 (0.089) | 32.84 (8.26)     |  |
|                      | TD <sup>2</sup> -CFM | 13.20 (0.36)  | 0.180 (0.076) | 34.61 (21.58)    |  |
| SPIN                 | TD-DD                | 21.31 (0.65)  | 1.482 (0.015) | 1093.50 (700.34) |  |
|                      | TD <sup>2</sup> -DD  | 103.72 (1.69) | 0.903 (0.067) | 115.58 (28.18)   |  |
| TD-CFM               | TD-CFM               | 15.16 (0.53)  | 1.020 (0.036) | 482.78 (24.82)   |  |
|                      | TD-CFM(C)            | 14.22 (0.06)  | 0.605 (0.076) | 170.20 (48.23)   |  |
| STAND                | TD-CFM               | 14.34 (0.20)  | 0.449 (0.056) | 197.13 (26.98)   |  |
|                      | TD-DD                | 21.34 (0.66)  | 1.459 (0.029) | 594.94 (219.72)  |  |
| TD-CFM               | TD <sup>2</sup> -DD  | 103.86 (4.22) | 0.630 (0.030) | 250.96 (79.14)   |  |
|                      | TD-CFM               | 14.28 (0.32)  | 0.829 (0.107) | 1371.68 (326.61) |  |
| TD-CFM(C)            | TD-CFM(C)            | 13.43 (0.34)  | 0.335 (0.067) | 265.09 (12.84)   |  |
|                      | TD <sup>2</sup> -CFM | 13.52 (0.61)  | 0.284 (0.062) | 166.16 (17.51)   |  |
| WALK                 | TD-DD                | 19.30 (0.80)  | 1.282 (0.033) | 622.04 (186.99)  |  |
|                      | TD <sup>2</sup> -DD  | 103.79 (3.80) | 0.471 (0.055) | 425.65 (131.20)  |  |
| TD-CFM               | TD-CFM               | 14.97 (0.14)  | 0.787 (0.070) | 1344.77 (149.38) |  |
|                      | TD-CFM(C)            | 12.39 (0.25)  | 0.165 (0.042) | 354.40 (114.40)  |  |
| TD <sup>2</sup> -CFM | TD <sup>2</sup> -CFM | 12.63 (0.46)  | 0.078 (0.072) | 301.97 (21.93)   |  |
|                      | TD-CFM               |               |               |                  |  |
| Pointmass Maze       |                      |               |               |                  |  |
| Task                 | Method               | EMD ↓         | NLL ↓         | MSE(V) ↓         |  |
| FAST SLOW            | TD-DD                | 0.164 (0.013) | 2.012 (0.089) | 1642.91 (26.55)  |  |
|                      | TD <sup>2</sup> -DD  | 0.350 (0.038) | 0.637 (0.046) | 236.52 (58.27)   |  |
|                      | TD-CFM               | 0.082 (0.004) | 0.772 (0.065) | 575.00 (75.51)   |  |
|                      | TD-CFM(C)            | 0.061 (0.002) | 0.083 (0.013) | 93.04 (5.55)     |  |
| REACH BOTTOM LEFT    | TD <sup>2</sup> -CFM | 0.060 (0.003) | 0.010 (0.059) | 61.08 (20.86)    |  |
|                      | TD-DD                | 0.151 (0.007) | 2.094 (0.119) | 537.80 (22.89)   |  |
|                      |                      |               |               |                  |  |

Table 17. Per task results for planning with GPI.

| Domain Method FB | Planner — |           |        |       |    |      | Local | 326            | 94           | (     | 7  | 00   | D ( Z ) ) | Train  |              |       |    |    | Domain Method Planner D ( Z ) Local Train FB — 683 96 ( 2 09 )      |
|------------------|-----------|-----------|--------|-------|----|------|-------|----------------|--------------|-------|----|------|-----------|--------|--------------|-------|----|----|---------------------------------------------------------------------|
| FB GPI           |           | 14 13 ( 0 | 51     | )     |    |      |       | 14 06          | ( 0 43       |       | )  |      |           | 14 57  | (            | 0     | 33 | )  |                                                                     |
| TD DD GPI        | 328       | 61        | ( 2 66 |       | )  |      | 303   | 45             | ( 44         | 57    |    | )    |           | 292 96 | (            | 52    |    | 91 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD GPI           | 316       | 41 (      | 4 59   | )     |    |      | 338   | 14             | ( 14         | 23    |    | )    |           | 349 77 | (            | 12    |    | 78 | )                                                                   |
| TD CFM GPI       | 301       | 88 (      | 17 94  |       | )  |      |       | 221 07         | ( 3          | 80    | )  |      |           | 199    | 50           | ( 4   | 09 | )  |                                                                     |
| TD CFM ( C ) GPI | 325       | 92 (      | 14     | 31    | )  |      |       | 368 97         | ( 16         | 10    |    | )    |           | 367 38 | (            | 11    |    | 48 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM GPI          | 325       | 76 (      | 12     | 88    | )  |      | 362   | 79             | ( 19         | 09    |    | )    |           | 358 73 | (            | 24    |    | 84 | )                                                                   |
| FB               | —         |           |        |       |    |      |       | 338            | 41           | (     | 2  | 98   | )         |        |              |       |    |    |                                                                     |
| FB GPI           |           | 29 32 ( 2 | 15     | )     |    |      |       | 36 99          | ( 4 28       |       | )  |      |           | 39 09  | (            | 5     | 90 | )  |                                                                     |
| TD DD GPI        | 281       | 76 (      | 74 27  |       | )  |      | 304   | 28             | ( 65         | 49    |    | )    |           | 299 92 | (            | 72    |    | 32 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD GPI           | 287       | 70 (      | 50 25  |       | )  |      | 298   | 07             | ( 51         | 86    |    | )    |           | 298 78 | (            | 34    |    | 52 | )                                                                   |
| TD CFM GPI       | 323       | 56 (      | 32 38  |       | )  |      |       | 323 90         | ( 34         | 69    |    | )    |           | 328 47 | (            | 15    |    | 90 | )                                                                   |
| TD CFM ( C ) GPI | 266       | 80 (      | 80 22  |       | )  |      | 251   | 21             | ( 64         | 39    |    | )    |           | 284 37 | (            | 73    |    | 57 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM GPI          | 287       | 26 (      | 84 37  |       | )  |      | 313   | 07             | ( 33         | 89    |    | )    |           | 320 22 | (            | 96    |    | 14 | )                                                                   |
| FB               | —         |           |        |       |    |      |       | 852            | 55           | ( 19  |    | 44   | )         |        |              |       |    |    |                                                                     |
| FB GPI           |           | 79 24 ( 2 | 32     | )     |    |      |       | 80 60          | ( 3 28       |       | )  |      |           | 82 58  | (            | 3     | 06 | )  |                                                                     |
| TD DD GPI        | 852       | 49 (      | 26 17  |       | )  |      |       | 806 55         | ( 7          | 62    | )  |      |           | 872 85 | (            | 19    |    | 73 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD GPI           | 839       | 00 (      | 22 97  |       | )  |      |       | 914 47         | ( 6          | 14    | )  |      |           | 936 41 | (            | 11    |    | 77 | )                                                                   |
| TD CFM GPI       | 823       | 10 (      | 15 06  |       | )  |      | 758   | 91             | ( 19         | 84    |    | )    |           | 846    | 08           | ( 7   | 85 | )  |                                                                     |
| TD CFM ( C ) GPI | 858       | 42 (      | 5 92   | )     |    |      |       | 931 74         | ( 12         | 01    |    | )    |           | 947    | 69           | ( 6   | 45 | )  |                                                                     |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM GPI          | 863       | 16 (      | 4 15   | )     |    |      |       | 923 77         | ( 9          | 88    |    | )    |           | 963    | 10           | ( 6   | 69 |    | )                                                                   |
| FB               | —         |           |        |       |    |      |       | 588            | 74           | (     | 5  | 30   | )         |        |              |       |    |    |                                                                     |
| FB GPI           |           | 18 23 ( 0 | 68     | )     |    |      |       | 18 36          | ( 1 15       |       | )  |      |           | 19 94  | (            | 0     | 87 | )  |                                                                     |
| TD DD GPI        | 587       | 76 (      | 6 46   | )     |    |      | 799   | 12             | ( 11         | 85    |    | )    |           | 667 74 | (            | 81    |    | 98 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD GPI           | 594       | 48 (      | 3 74   | )     |    |      | 842   | 94             | ( 28         | 16    |    | )    |           | 852 17 | (            | 12    |    | 09 | )                                                                   |
| TD CFM GPI       | 577       | 95 (      | 2 89   | )     |    |      | 793   | 47             | ( 21         | 27    |    | )    |           | 774 91 | (            | 55    |    | 37 | )                                                                   |
| TD CFM ( C ) GPI | 601       | 81 (      | 8 58   | )     |    |      |       | 883 26         | ( 4          | 06    |    | )    |           | 897 33 | (            | 10    |    | 88 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM GPI          | 596       | 08 (      | 3 41   | )     |    |      | 868   | 69             | ( 14         | 42    |    | )    |           | 868 46 | (            | 44    |    | 44 | )                                                                   |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 742 71 ( 1 01 ) 746 48 ( 1 63 ) 718 52 ( 2 65 )              |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 673 33 ( 6 07 ) 690 13 ( 6 34 ) 677 58 ( 5 71 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
| RUN              |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 744 92 ( 0 69 ) 750 42 ( 2 30 ) 745 29 ( 1 12 ) JUMP         |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 748 19 ( 10 47 ) 753 72 ( 0 58 ) 745 93 ( 12 60 )        |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 790 56 ( 14 06 ) 795 84 ( 16 14 ) 785 20 ( 13 69 ) |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 796 39 ( 13 27 ) 800 34 ( 9 63 ) 791 43 ( 11 66 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 452 38 ( 3 25 )                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 486 71 ( 0 64 ) 488 23 ( 0 48 ) 469 03 ( 2 35 )              |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 484 45 ( 1 07 ) 482 81 ( 2 55 ) 482 53 ( 2 38 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
| SPIN             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 485 26 ( 1 63 ) 486 35 ( 0 93 ) 484 89 ( 2 43 ) RUN          |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 488 93 ( 1 08 ) 488 45 ( 0 62 ) 488 98 ( 0 28 )          |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 491 66 ( 2 75 ) 490 89 ( 2 05 ) 491 81 ( 2 14 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 488 89 ( 1 35 ) 488 65 ( 1 19 ) 489 31 ( 1 03 )             |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 896 43 ( 5 80 )                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 975 01 ( 1 40 ) 977 94 ( 0 76 ) 938 44 ( 7 04 )              |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 976 59 ( 2 78 ) 976 75 ( 0 86 ) 975 25 ( 2 49 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
| STAND            |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 981 26 ( 1 56 ) 981 59 ( 1 45 ) 979 46 ( 0 93 ) STAND        |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 982 08 ( 1 27 ) 981 06 ( 0 26 ) 981 29 ( 1 34 )          |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 984 03 ( 1 20 ) 984 50 ( 1 49 ) 983 33 ( 1 20 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 984 36 ( 0 25 ) 985 52 ( 0 89 ) 984 36 ( 1 21 )             |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 476 34 ( 4 71 )                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 483 37 ( 1 05 ) 483 73 ( 3 02 ) 458 20 ( 6 62 )              |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 497 55 ( 10 40 ) 499 45 ( 11 65 ) 494 38 ( 19 75 )        |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
| WALK             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 457 54 ( 23 37 ) 467 78 ( 4 58 ) 452 44 ( 17 14 ) WALK       |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 458 20 ( 29 01 ) 466 62 ( 19 30 ) 458 24 ( 30 28 )       |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 515 84 ( 5 84 ) 519 36 ( 14 37 ) 524 37 ( 1 56 )   |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 516 67 ( 3 49 ) 511 77 ( 3 70 ) 517 82 ( 2 58 )             |
|                  |           | Pointmass |        |       |    | Maze |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| Domain Method    | Planner   |           |        |       |    |      |       | Z-Distribution |              |       |    |      | D ( Z     | )      |              |       |    |    |                                                                     |
|                  |           | Random    |        |       |    |      | Local |                | Perturbation |       |    |      |           | Train  | Distribution |       |    |    |                                                                     |
| FAST SLOW        |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| FB               | —         |           |        |       |    |      |       | 223            |              | 85    |    | ( 23 | 81 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 1         | 67 (   | 0     | 30 | )    |       | 74             | 52           | ( 2   | 24 | )    |           | 1      | 24 (         | 0     | 28 | )  |                                                                     |
| TD DD            | GPI       | 169       | 55     | ( 74  |    | 06   | )     | 363            | 47           | ( 23  |    | 78   | )         | 148    | 59           | ( 43  |    | 53 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 781       | 84     | (     | 1  | 20   | )     | 769            | 02           | (     | 5  | 03   | )         | 768    | 67           | ( 11  |    | 19 | )                                                                   |
| TD CFM           | GPI       | 254       | 07     | ( 85  |    | 86   | )     | 546            | 75           | ( 191 |    | 17   | )         | 359    | 50           | ( 144 |    | 41 | )                                                                   |
| TD CFM ( C )     | GPI       | 763       | 24     | ( 15  |    | 57   | )     | 776            | 51           | ( 12  |    | 37   | )         | 769    | 87           | ( 13  |    | 18 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 773       | 51     | (     | 2  | 71   | )     | 773            | 81           | (     | 4  | 71   | )         | 772    | 22           | (     | 3  | 11 | )                                                                   |
| FB               | —         |           |        |       |    |      |       |                | 317          | 59    |    | ( 8  | 55 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 81        | 99     | ( 5   | 11 | )    |       | 315            | 10           | (     | 1  | 95   | )         | 61     | 41           | ( 3   | 58 | )  |                                                                     |
| TD DD            | GPI       | 462       | 86     | (     | 5  | 90   | )     | 430            | 51           | ( 72  |    | 79   | )         | 593    | 64           | ( 56  |    | 15 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 876       | 91     | (     | 9  | 21   | )     | 889            | 03           | (     | 2  | 40   | )         | 878    | 78           | (     | 2  | 43 | )                                                                   |
| TD CFM           | GPI       | 832       | 91     | ( 27  |    | 77   | )     | 797            | 10           | ( 57  |    | 17   | )         | 852    | 81           | ( 16  |    | 74 | )                                                                   |
| TD CFM ( C )     | GPI       | 873       | 85     | ( 21  |    | 16   | )     | 885            | 90           | (     | 4  | 21   | )         | 875    | 45           | (     | 3  | 43 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 885       | 07     | (     | 2  | 79   | )     | 887            | 18           | (     | 5  | 27   | )         | 878    | 26           | (     | 0  | 64 | )                                                                   |
| BOTTOM LEFT      |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| FB               | —         |           |        |       |    |      |       |                | 830          | 60    |    | ( 0  | 63 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 0         | 18 (   | 0     | 17 | )    |       | 127            | 90           | ( 20  |    | 14   | )         | 0      | 11 (         | 0     | 10 | )  |                                                                     |
| TD DD            | GPI       | 781       | 69     | (     | 8  | 09   | )     | 797            | 98           | (     | 3  | 52   | )         | 795    | 12           | (     | 3  | 88 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 823       | 28     | (     | 2  | 76   | )     | 820            | 15           | (     | 1  | 89   | )         | 824    | 00           | (     | 1  | 40 | )                                                                   |
| TD CFM           | GPI       | 808       | 61     | (     | 7  | 06   | )     | 801            | 97           | (     | 2  | 97   | )         | 813    | 36           | (     | 6  | 35 | )                                                                   |
| TD CFM ( C )     | GPI       | 824       | 02     | (     | 0  | 73   | )     | 824            | 17           | (     | 1  | 77   | )         | 824    | 18           | (     | 3  | 84 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 827       | 85     | (     | 1  | 45   | )     | 820            | 98           | (     | 3  | 63   | )         | 828    | 45           | (     | 3  | 10 | )                                                                   |
| REACH BOTTOM     |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| LEFT LONG        |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| FB               | —         |           |        |       |    |      |       |                | 49           | 31    |    | ( 0  | 09 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | − 464     | 55     | (     | 19 | 21   | )     | 0              | 58 (         | 1     | 79 | )    |           | − 401  | 26           | (     | 28 | 43 | )                                                                   |
| TD DD            | GPI       | 461       | 30     | (     | 7  | 43   | )     | 468            | 73           | ( 26  |    | 94   | )         | 252    | 28           | ( 241 |    | 97 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 609       | 10     | ( 11  |    | 64   | )     | 597            | 03           | (     | 6  | 46   | )         | 668    | 76           | (     | 4  | 02 | )                                                                   |
| TD CFM           | GPI       | 180       | 27     | ( 35  |    | 66   | )     | 311            | 59           | ( 152 |    | 06   | )         | 439    | 47           | ( 230 |    | 80 | )                                                                   |
| TD CFM ( C )     | GPI       | 631       | 52     | ( 11  |    | 58   | )     | 614            | 90           | (     | 8  | 82   | )         | 688    | 44           | (     | 4  | 05 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 646       | 67     | (     | 9  | 38   | )     | 639            | 90           | ( 13  |    | 22   | )         | 691    | 68           | (     | 2  | 99 | )                                                                   |
| BOTTOM RIGHT     |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| FB               | —         |           |        |       |    |      |       | 366            |              | 39    | (  | 27   | 01 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 0         | 00 (   | 0     | 00 | )    |       | 0              | 00 (         | 0     | 00 | )    |           | 0      | 00 (         | 0     | 00 | )  |                                                                     |
| TD DD            | GPI       | 343       | 62     | ( 112 |    | 70   | )     | 470            | 97           | ( 42  |    | 25   | )         | 398    | 94           | ( 81  |    | 80 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 360       | 71     | ( 312 |    | 31   | )     | 674            | 54           | (     | 7  | 35   | )         | 529    | 97           | ( 137 |    | 98 | )                                                                   |
| TD CFM           | GPI       | 394       | 78     | ( 159 |    | 98   | )     | 356            | 58           | ( 69  |    | 31   | )         | 548    | 65           | ( 73  |    | 62 | )                                                                   |
| TD CFM ( C )     | GPI       | 642       | 67     | (     | 6  | 59   | )     | 686            | 08           | (     | 4  | 46   | )         | 679    | 75           | (     | 2  | 98 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 534       | 62     | ( 57  |    | 49   | )     | 687            | 66           | (     | 1  | 75   | )         | 641    | 45           | (     | 2  | 00 | )                                                                   |
| TOP LEFT         |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| FB               | —         |           |        |       |    |      |       |                | 895          | 88    |    | ( 1  | 26 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 351       | 72     | ( 17  |    | 68   | )     | 837            | 14           | (     | 2  | 07   | )         | 185    | 50           | ( 13  |    | 00 | )                                                                   |
| TD DD            | GPI       | 941       | 32     | ( 16  |    | 86   | )     | 812            | 40           | ( 152 |    | 88   | )         | 920    | 44           | ( 28  |    | 63 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 940       | 90     | (     | 5  | 41   | )     | 967            | 44           | (     | 3  | 52   | )         | 939    | 49           | ( 10  |    | 02 | )                                                                   |
| TD CFM           | GPI       | 964       | 27     | (     | 0  | 34   | )     | 948            | 83           | ( 11  |    | 63   | )         | 955    | 82           | (     | 9  | 32 | )                                                                   |
| TD CFM ( C )     | GPI       | 940       | 00     | ( 29  |    | 18   | )     | 967            | 03           | (     | 3  | 38   | )         | 931    | 02           | ( 18  |    | 53 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 943       | 43     | ( 19  |    | 57   | )     | 967            | 06           | (     | 1  | 42   | )         | 940    | 05           | ( 18  |    | 13 | )                                                                   |
| TOP RIGHT        |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| FB               | —         |           |        |       |    |      |       |                | 715          | 25    |    | ( 4  | 47 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 0         | 72 (   | 0     | 96 | )    |       | 358            | 22           | ( 20  |    | 05   | )         | 1      | 35 (         | 0     | 85 | )  |                                                                     |
| TD DD            | GPI       | 766       | 59     | (     | 6  | 78   | )     | 771            | 64           | (     | 9  | 55   | )         | 733    | 83           | ( 44  |    | 76 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 822       | 44     | (     | 1  | 74   | )     | 818            | 06           | (     | 6  | 60   | )         | 823    | 09           | (     | 1  | 76 | )                                                                   |
| TD CFM           | GPI       | 777       | 94     | ( 46  |    | 86   | )     | 765            | 68           | ( 41  |    | 55   | )         | 754    | 73           | ( 45  |    | 71 | )                                                                   |
| TD CFM ( C )     | GPI       | 826       | 30     | (     | 1  | 36   | )     | 824            | 87           | (     | 2  | 61   | )         | 821    | 51           | (     | 5  | 64 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 809       | 75     | ( 28  |    | 90   | )     | 824            | 23           | (     | 1  | 88   | )         | 788    | 98           | ( 45  |    | 69 | )                                                                   |
| FB               | —         |           |        |       |    |      |       |                | 337          | 33    |    | ( 9  | 46 )      |        |              |       |    |    |                                                                     |
| FB               | GPI       | 4         | 89 (   | 1     | 03 | )    |       | 148            | 93           | (     | 0  | 90   | )         | 2      | 97 (         | 0     | 82 | )  |                                                                     |
| TD DD            | GPI       | 585       | 01     | ( 39  |    | 45   | )     | 587            | 71           | ( 46  |    | 77   | )         | 451    | 92           | (     | 5  | 35 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| DD               | GPI       | 896       | 45     | (     | 7  | 94   | )     | 910            | 52           | (     | 2  | 63   | )         | 878    | 19           | ( 11  |    | 18 | )                                                                   |
| TD CFM           | GPI       | 790       | 65     | (     | 3  | 06   | )     | 843            | 76           | (     | 8  | 91   | )         | 841    | 25           | ( 22  |    | 73 | )                                                                   |
| TD CFM ( C )     | GPI       | 905       | 41     | (     | 1  | 13   | )     | 920            | 06           | (     | 2  | 51   | )         | 874    | 00           | ( 10  |    | 23 | )                                                                   |
| TD 2             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    |                                                                     |
| CFM              | GPI       | 901       | 82     | (     | 1  | 29   | )     | 910            | 41           | (     | 1  | 66   | )         | 866    | 85           | (     | 7  | 95 | )                                                                   |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | Domain Method Planner Z-Distribution D ( Z )                        |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | Random Local Perturbation Train Distribution                        |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 221 55 ( 44 79 )                                               |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 355 27 ( 5 95 ) 356 52 ( 9 99 ) 355 94 ( 5 10 )              |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 451 93 ( 81 15 ) 445 10 ( 100 81 ) 424 78 ( 100 74 )      |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 702 98 ( 27 77 ) 712 72 ( 16 66 ) 683 62 ( 35 04 ) FLIP      |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 355 69 ( 110 25 ) 420 53 ( 184 00 ) 341 40 ( 124 16 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 724 85 ( 8 19 ) 710 02 ( 4 51 ) 711 16 ( 13 29 )   |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 722 08 ( 7 50 ) 718 74 ( 14 51 ) 713 66 ( 14 14 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 463 12 ( 5 73 )                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 238 33 ( 9 74 ) 388 33 ( 25 98 ) 249 60 ( 5 64 )             |
| LOOP             |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 620 00 ( 69 42 ) 596 45 ( 38 20 ) 595 59 ( 34 96 ) FLIP   |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 706 99 ( 8 08 ) 690 83 ( 3 20 ) 706 75 ( 8 34 )              |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 545 12 ( 184 05 ) 540 36 ( 186 74 ) 492 55 ( 173 13 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 727 23 ( 25 25 ) 716 22 ( 29 49 ) 711 11 ( 20 97 ) |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 709 19 ( 16 76 ) 684 33 ( 37 92 ) 694 16 ( 15 24 )          |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 310 39 ( 35 44 )                                               |
| REACH            |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 200 65 ( 4 44 ) 301 34 ( 11 26 ) 191 10 ( 6 56 )             |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 436 74 ( 3 52 ) 438 90 ( 4 92 ) 434 94 ( 3 02 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 427 15 ( 16 50 ) 429 98 ( 13 04 ) 421 92 ( 14 83 ) RUN       |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 206 96 ( 45 56 ) 243 53 ( 60 37 ) 238 96 ( 66 97 )       |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 465 08 ( 2 50 ) 470 44 ( 5 05 ) 462 89 ( 3 15 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 462 71 ( 9 73 ) 467 25 ( 14 78 ) 454 90 ( 10 61 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 201 07 ( 10 72 )                                               |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 5 31 ( 2 02 ) 102 20 ( 5 73 ) 19 11 ( 2 52 )                 |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 165 02 ( 4 50 ) 246 72 ( 12 09 ) 325 40 ( 0 86 ) RUN      |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 224 90 ( 21 33 ) 310 10 ( 22 82 ) 322 33 ( 4 05 )            |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 90 83 ( 28 26 ) 92 46 ( 15 59 ) 49 88 ( 29 15 )          |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 222 14 ( 36 05 ) 342 15 ( 2 02 ) 333 90 ( 3 00 )   |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 252 70 ( 10 86 ) 319 46 ( 35 05 ) 332 21 ( 0 77 )           |
| REACH            |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 792 89 ( 52 74 )                                               |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 830 00 ( 15 20 ) 889 84 ( 5 00 ) 733 11 ( 34 27 )            |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 977 30 ( 3 13 ) 978 74 ( 2 47 ) 979 48 ( 3 47 )           |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 959 18 ( 30 39 ) 955 97 ( 25 64 ) 956 79 ( 29 06 ) WALK      |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 767 47 ( 96 47 ) 805 68 ( 104 96 ) 853 73 ( 117 82 )     |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 985 04 ( 0 10 ) 985 06 ( 0 29 ) 984 90 ( 0 18 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 984 21 ( 0 03 ) 984 46 ( 0 09 ) 984 23 ( 0 07 )             |
| REACH            |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB — 897 16 ( 32 19 )                                               |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | FB GPI 22 40 ( 10 18 ) 373 19 ( 13 71 ) 78 60 ( 18 72 )             |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD DD GPI 793 32 ( 52 67 ) 946 70 ( 12 57 ) 982 37 ( 0 21 ) WALK    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | DD GPI 951 82 ( 11 09 ) 981 74 ( 0 27 ) 982 45 ( 0 07 )             |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM GPI 455 18 ( 190 16 ) 456 19 ( 140 79 ) 257 85 ( 173 06 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD CFM ( C ) GPI 964 75 ( 4 54 ) 981 93 ( 0 26 ) 982 89 ( 0 25 )    |
|                  |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | TD 2                                                                |
| REACH SQUARE     |           |           |        |       |    |      |       |                |              |       |    |      |           |        |              |       |    |    | CFM GPI 962 41 ( 4 32 ) 982 08 ( 0 16 ) 982 64 ( 0 05 )             |

![](_page_30_Figure_1.jpeg)

Figure 5. Qualitative samples generated with TD-CFM, TD-DD, VAE, and GAN methods for various discount factors γ on the LOOP task in the POINTMASS MAZE domain. The last row depicts ground truth discounted occupancies.

### E. Theoretical Results

#### E.1. Proofs of Main Results

Lemma 1. *Let* <sup>→</sup> p<sup>t</sup> *be a probability path for* P *generated by vector field* <sup>→</sup> v<sup>t</sup> *and* ↷ p (n) <sup>t</sup> *be a probability path for* P <sup>π</sup>m (n) 1 *generated by* ↷ v (n) t *such that* <sup>→</sup> p<sup>0</sup> = p (n) <sup>0</sup> = m0*. For any* t ∈ [0, 1] *and* (s, a) *let* v (n+1) t (· | s, a) *be the solution of* [<sup>4</sup>](#page-0-0)

$$\begin{aligned} & \arg \min_{v: \mathbb{R}^d \rightarrow \mathbb{R}^d} (1 - \gamma) \mathbb{E}_{\vec{X}_t \sim \vec{p}_t(\cdot | s, a)} \left[ \|v(\vec{X}_t) - \vec{v}_t(\vec{X}_t \mid s, a)\|^2 \right] \\ & + \gamma \mathbb{E}_{\widehat{X}_t \sim \widehat{p}_t^{(n)}(\cdot | s, a)} \left[ \|v(\widehat{X}_t) - \widehat{v}_t^{(n)}(\widehat{X}_t \mid s, a)\|^2 \right]. \end{aligned}$$

*Then* v (n+1) t *induces a probability path* m (n+1) t *such that* m (n+1) <sup>0</sup> = m<sup>0</sup> *and* m (n+1) <sup>1</sup> = T <sup>π</sup>m (n) *.*

*Proof.* By [Lemma 4,](#page-32-0) we have that

$$v_t^{(n+1)}(x \mid s, a) = \frac{(1 - \gamma) \vec{p}_t(x \mid s, a) \vec{v}_t(x \mid s, a) + \gamma \hat{\vec{p}}_t^{(n)}(x \mid s, a) \hat{\vec{v}}_t^{(n)}(x \mid s, a)}{m_t^{(n+1)}(x \mid s, a)},$$

where m (n+1) t (x|s, a) = (1 − γ) → pt(x|s, a) + γ p (n) t (x|s, a). [Lemma 3](#page-32-1) implies that m (n+1) t is the probability path generated by v (n+1) t . It is easy to see that m (n+1) <sup>0</sup> <sup>=</sup> <sup>m</sup><sup>0</sup> since <sup>→</sup> p<sup>0</sup> = p (n) <sup>0</sup> <sup>=</sup> <sup>m</sup>0. Moreover, since <sup>→</sup> p<sup>1</sup> = P and ↷ p (n) <sup>1</sup> = P <sup>π</sup>m (n) 1 by assumption, m (n+1) <sup>1</sup> = (1 − γ)P + γP <sup>π</sup>m (n) <sup>1</sup> = T <sup>π</sup>m (n) 1 , which proves the result.

Theorem 1. *For any* n ≥ 1*, the probability paths generated by* TD-CFM*,* TD-CFM(C)*, or* TD<sup>2</sup> -CFM *satisfy*

$$m_t^{(n+1)}(x \mid s, a) = \left( \mathcal{B}_t^\pi m_t^{(n)} \right) (x \mid s, a), \quad \forall t \in [0, 1]$$

*where* B π <sup>t</sup> m := (1 − γ)P<sup>t</sup> + γP <sup>π</sup>m *and* Pt(x|s, a) := R pt|1(x | x1)P(x1|s, a)dx1*. For any* t ∈ [0, 1]*, the operator* B π t *is a* γ*-contraction in 1-Wasserstein distance, that is, for any couple of probability paths* pt, qt*,*

$$\begin{aligned} \sup_{s,a} W_1 ((\mathcal{B}_t^\pi p_t)(\cdot | s, a), (\mathcal{B}_t^\pi q_t)(\cdot | s, a)) \\ \leq \gamma \sup_{s,a} W_1 (p_t(\cdot | s, a), q_t(\cdot | s, a)). \end{aligned}$$

*Proof.* To prove that the iterates of the three algorithms satisfy a Bellman-like update through the operator B π <sup>t</sup> we only need to apply [Proposition 3](#page-33-0) for TD<sup>2</sup> -CFM, [Theorem 5](#page-35-0) for TD-CFM, and [Theorem 6](#page-37-0) for TD-CFM(C). That B<sup>t</sup> is a γ-contraction in 1-Wasserstein distance can be seen by applying [Theorem 4](#page-33-1) with k = 1.

Corollary 1. *Let* {m (n) <sup>t</sup> }n≥<sup>0</sup> *be the sequence of probability paths produced by* TD-CFM*,* TD-CFM(C)*, or* TD<sup>2</sup> -CFM *starting from an arbitrary vector field* v (0) t *. Then,*

$$\lim_{n \rightarrow \infty} m_t^{(n)} = \overline{m}_t = \mathcal{B}_t \overline{m}_t,$$

*where* m<sup>t</sup> *is the unique fixed point of* Bt*, and* m<sup>t</sup> = mMC t *, where* mMC t (· | s, a) = R pt|1(· | x1) m<sup>π</sup> (x<sup>1</sup> | s, a) *is the probability path of the Monte-Carlo approach* (MC-[CFM](#page-2-1); 6)*.*

*Proof.* That B π <sup>t</sup> has a unique fixed point m¯ <sup>t</sup> to which every sequence m (n) t converges to is a consequence of the Banach fixed point theorem applied on the space of all probability paths m<sup>t</sup> : S × A → P(<sup>R</sup> d ) equipped with the sup-1- Wasserstein metric. By inspecting the definition of B π t , it is easy to see that m¯ <sup>t</sup> = (I − γP <sup>π</sup> ) <sup>−</sup><sup>1</sup>Pt. Since Pt(x|s, a) = R pt|1(x|x1)P(x1|s, a)dx1,

$$\bar{m}_t(x|s, a) = [(I - \gamma P^\pi)^{-1} P_t](x|s, a) = \int p_{t|1}(x|x_1) \underbrace{[(I - \gamma P^\pi)^{-1} P](x_1|s, a)}_{=m^\pi(x_1|s, a)} dx_1 = m_t^{\text{MC}}(x|s, a).$$

Theorem 2. *For any* n ≥ 1 *and* t ∈ [0, 1]*, assume that* m (n) t (x | s, a) = R pt|1(x | x1)m (n) 1 (x<sup>1</sup> | s, a)dx1*, then*

$$\sigma_{\text{TD-CFM}}^2 = \sigma_{\text{TD}^2\text{-CFM}}^2 + \gamma^2 \mathbb{E} [\text{Tr}(\text{Cov}_{X_1|s,a,X_t} [\nabla_\theta v_t(X_t|s, a; \theta)^\top u_{t|1}(X_t|X_1)])].$$

*Proof.* See [Theorem 7.](#page-38-1)

Theorem 3. *For any* n ≥ 1 *and* t ∈ [0, 1]*, assume that* m (n) t (x | s, a) = R pt|0,1(x | x0, x1)m (n) 0,1 (x0, x<sup>1</sup> | s, a)dx0dx<sup>1</sup> [5](#page-0-0) *, then we obtain*

$$\sigma_{\text{TD-CFM(C)}}^2 = \sigma_{\text{TD}^2\text{-CFM}}^2 + \gamma^2 \mathbb{E} \left[ \text{Tr}(\text{Cov}_{\mathbb{Z}|S,A,X_t} [\nabla_{\theta} v_t(X_t|S,A;\theta)^\top u_{t|Z}(X_t|Z)]) \right],$$

*where* Z = (X0, X1)*. Furthermore, if we use straight conditional paths, i.e.,* X<sup>t</sup> = tX<sup>1</sup> + (1 − t)X0*, and the linear interpolant* X<sup>t</sup> *does not intersect for any* s, a, s′ *, then* σ 2 TD-CFM(C) = σ 2 TD<sup>2</sup> -CFM *.*

*Proof.* See [Theorem 8.](#page-39-0)

# E.2. General Results

Lemma 3. *Let* v 1 <sup>t</sup> *and* v 2 <sup>t</sup> *be vector fields that generate the probability paths* p 1 <sup>t</sup> *and* p 2 t *, respectively. Then, for any* γ ∈ [0, 1]*, the mixture probability path* p<sup>t</sup> = (1 − γ)p 1 <sup>t</sup> + γp<sup>2</sup> t *is generated by the vector field*

$$v_t := \frac{(1 - \gamma)p_t^1 v_t^1 + \gamma p_t^2 v_t^2}{(1 - \gamma)p_t^1 + \gamma p_t^2}. \quad (27)$$

*Proof.* Since v t 1 (resp. v t 2 ) generates p 1 t (resp. p 2 t ), we know from the continuity equation that:

$$\frac{\partial p_t^1}{\partial t} = \operatorname{div}(p_t^1 v_t^1), \quad \frac{\partial p_t^2}{\partial t} = \operatorname{div}(p_t^2 v_t^2),$$

where div denotes the divergence operator. Then, by linearity of div,

$$\begin{aligned} \frac{\partial p_t}{\partial t} &= \frac{\partial ((1-\gamma)p_t^1 + \gamma p_t^2)}{\partial t} \\ &= (1-\gamma)\operatorname{div}(p_t^1 v_t^1) + \gamma \operatorname{div}(p_t^2 v_t^2) \\ &= \operatorname{div} \left( (1-\gamma)p_t^1 v_t^1 + \gamma p_t^2 v_t^2 \right) \\ &= \operatorname{div} \left( \frac{(1-\gamma)p_t^1 v_t^1 + \gamma p_t^2 v_t^2}{(1-\gamma)p_t^1 + \gamma p_t^2} ((1-\gamma)p_t^1 + \gamma p_t^2) \right) \\ &= \operatorname{div} \left( \frac{(1-\gamma)p_t^1 v_t^1 + \gamma p_t^2 v_t^2}{(1-\gamma)p_t^1 + \gamma p_t^2} p_t \right) \\ &= \operatorname{div}(v_t p_t). \end{aligned}$$

Hence, (vt, pt) satisfies the continuity equation, which implies that v<sup>t</sup> generates pt.

Lemma 4. *Let* v 1 <sup>t</sup> *and* v 2 <sup>t</sup> *be vector fields that generate the probability paths* p 1 <sup>t</sup> *and* p 2 t *, respectively. For* γ ∈ [0, 1]*, the vector field* v<sup>t</sup> = (1−γ)p v <sup>t</sup> <sup>+</sup>γp<sup>2</sup> v (1−γ)p 1 <sup>t</sup> +γp<sup>2</sup> *satisfies*

$$v_t = \arg \min_{v: \mathbb{R}^d \rightarrow \mathbb{R}^d} \left\{ (1 - \gamma) \mathbb{E}_{x_t \sim p_t^1} \left[ \|v_t(x_t) - v_t^1(x_t)\|^2 \right] + \gamma \mathbb{E}_{x_t \sim p_t^2} \left[ \|v_t(x_t) - v_t^2(x_t)\|^2 \right] \right\}.$$

*Proof.* Let ℓt(v) := (1 − γ) <sup>E</sup>xt∼<sup>p</sup> -∥vt(xt) − v 1 t (xt)∥ 2 + γ <sup>E</sup>xt∼<sup>p</sup> -∥vt(xt) − v 2 t (xt)∥ 2 . The functional derivative of this quantity wrt v evaluated at some point x is

$$\nabla_v \ell_t(v)(x) = (1 - \gamma)p_1^t(x)(v_t(x) - v_t^1(x)) + \gamma p_2^t(x)(v_t(x) - v_t^2(x)).$$

#### E.3. Analysis of TD<sup>2</sup> -CFM

We study the learning dynamics of an idealized variant of TD<sup>2</sup> -CFM which minimizes the flow-matching loss exactly. Starting from an arbitrary vector field v (0) t , at each iteration n ≥ 0 we compute

$$v_t^{(n+1)}(\cdot | s, a) \in \arg \min_{v: \mathbb{R}^d \rightarrow \mathbb{R}^d} \ell_{\text{TD}-\text{CFM}}^{(n)}(t, s, a), \quad (28)$$

where

$$\begin{aligned} \ell_{\text{TD}-\text{CFM}}^{(n)}(t, s, a) &:= (1 - \gamma) \vec{\ell}(t, s, a) + \gamma \widehat{\ell}(t, s, a) \\ \vec{\ell}(t, s, a) &:= \mathbb{E}_{S' \sim P(\cdot|s, a), X_t \sim p_{t|1}(\cdot|S')} \left[ \left\| v(X_t|s, a) - u_t(X_t|S') \right\|^2 \right] \\ \widehat{\ell}(t, s, a) &:= \mathbb{E}_{S' \sim P(\cdot|s, a), X_t \sim m_t^{(n)}(\cdot|s', \pi(s'))} \left[ \left\| v(X_t|s, a) - v_t^{(n)}(X_t|S', \pi(S')) \right\|^2 \right], \end{aligned}$$

and m (n) t (x|s, a) is the probability path generated by v (n) t (x|s, a).

Lemma 5. *For any* n ≥ 0*, the vector field minimizing* [\(28\)](#page-33-2) *is*

$$v_t^{(n+1)}(x \mid s, a) = \frac{(1 - \gamma) \int u_{t|1}(x \mid x_1) p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 + \gamma \mathbb{E}_{S' \sim P(\cdot \mid s, a)} [m_t^{(n)}(x \mid S', \pi(S')) v_t^{(n)}(x \mid S', \pi(S'))]}{m_t^{(n+1)}(x \mid s, a)}$$

*where we define* m (n+1) t (x|s, a) := (1 − γ)Pt(x|s, a) + γES′∼<sup>P</sup> (·|s,a) [m (n) t (x|S ′ , π(S ′ ))] *and* Pt(x|s, a) := R pt|1(x | x1)P(x1|s, a)dx1*. Moreover* v (n+1) <sup>t</sup> *generates* m (n+1) t *.*

*Proof.* By Theorem 2 of [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2), we have for the first term in ℓTD<sup>2</sup> -CFM

$$\nabla_{\theta} \vec{\ell}(t, s, a) = \nabla_{\theta} \mathbb{E}_{X_t \sim P_t(\cdot | s, a)} \left[ \|v_t(X_t | s, a) - \vec{v}_t(X_t | s, a)\|^2 \right],$$

where Pt(x|s, a) := R pt|1(x | x1)P(x1|s, a)dx1, → vt(x|s, a) = R ut|1(x|x1)pt|1(x|x1)P (x1|s,a)dx<sup>1</sup> Pt(x|s,a) . Similarly, we have for the second term:

$$\nabla_{\theta} \widetilde{\ell}(t, s, a) = \nabla_{\theta} \mathbb{E}_{X_t \sim \widehat{p}_t^{(n)}(\cdot | s, a)} \left[ \| v_t(X_t | s, a) - \widehat{v}_t(X_t | s, a) \|^2 \right],$$

where ↷ p (n) <sup>t</sup> = P <sup>π</sup>m (n) t and ↷ v<sup>t</sup> = P <sup>π</sup>(m (n) t v (n) t ) P <sup>π</sup>m (n) .

Therefore, ℓ (n) TD-CFM(t, s, a) is equivalent, in term of gradient, to a mixture of two marginal flow-matching losses, which implies that v (n+1) <sup>t</sup> has the stated expression by [Lemma 4.](#page-32-0) The fact that it generates m (n+1) t is a consequence of [Lemma 3.](#page-32-1)

We then define the following operator to characterize the iterates of TD<sup>2</sup> -CFM.

Definition 1 (Bellman operator for probability paths). *For any* t ∈ [0, 1]*, we define the operator* B π <sup>t</sup> m := (1−γ)Pt+γP <sup>π</sup>m*, where* Pt(x|s, a) := R pt|1(x | x1)P(x1|s, a)dx1*.*

The following observation is then immediate from [Lemma 5.](#page-33-3)

Proposition 3. *For any* n ≥ 0*, the probability path generated by* TD<sup>2</sup> -CFM *satisfies* m (n+1) t (x|s, a) = B π <sup>t</sup> m (n) t (x | s, a)*, where* B π t *is the operator of [Definition 1.](#page-33-4) Moreover,* m (n+1) 1 (x|s, a) = T <sup>π</sup>m (n) 1 (x | s, a)*.*

Theorem 4. *For any* t ∈ [0, 1]*, the operator* B π <sup>t</sup> *of [Definition 1](#page-33-4) is a* γ <sup>1</sup>/k*-contraction in Wasserstein k-distance, i.e., for any couple of probability paths* pt, q<sup>t</sup> *and* k ∈ [1, ∞)*,*

$$\sup_{s,a} W_k \left( (\mathcal{B}_t^\pi p_t)(\cdot \mid s, a), (\mathcal{B}_t^\pi q_t)(\cdot \mid s, a) \right) \leq \gamma^{1/k} \sup_{s,a} W_k (p_t(\cdot \mid s, a), q_t(\cdot \mid s, a)).$$

*Proof.* Recall that the Wasserstein k-distance between p<sup>t</sup> and q<sup>t</sup> induced by a metric d is defined as

$$W_k(p_t(\cdot|s, a), q_t(\cdot|s, a)) := \inf_{\Gamma(\cdot|s, a) \in \mathcal{C}(p_t(\cdot|s, a), q_t(\cdot|s, a))} \mathbb{E}_{(X, Y) \sim \Gamma(\cdot|s, a)} [d(X, Y)^k]^{1/k},$$

where C(pt(·|s, a), qt(·|s, a)) is the set of all couplings between the two measures. Now take any coupling Γ( ˜ ·|s, a) ∈ C(pt(·|s, a), qt(·|s, a)) for any s, a. Then, the following quantity

$$\Theta(x, y|s, a) = (1 - \gamma)P(x|s, a)\delta(x - y) + \gamma(P^\pi \tilde{\Gamma})(x, y|s, a)$$

is a valid coupling between B π <sup>t</sup> p<sup>t</sup> (· | s, a) and B π t qt (· | s, a). In fact,

$$\begin{aligned} \int \Theta(x, y|s, a) dx &= (1 - \gamma) \int P(x|s, a) \delta(x - y) dx + \gamma \int (P^\pi \tilde{\Gamma})(x, y|s, a) dx \\ &= (1 - \gamma) P(y|s, a) + \gamma \int \mathbb{E}_{s' \sim P(\cdot|s, a)} \left[ \tilde{\Gamma}(x, y|s', \pi(s')) \right] dx \\ &= (1 - \gamma) P(y|s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s, a)} \left[ \int \tilde{\Gamma}(x, y|s', \pi(s')) dx \right] \\ &= (1 - \gamma) P(y|s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s, a)} [q_t(y|s', \pi(s'))] \\ &= (\mathcal{T}^\pi q_t)(y|s, a). \end{aligned}$$

Analogously, we can prove that R Θ(x, y|s, a)dy = B <sup>π</sup>p<sup>t</sup> (x|s, a). Then,

$$\begin{aligned} W_k \left( (\mathcal{B}_t^\pi p_t)(\cdot \mid s, a), (\mathcal{B}_t^\pi q_t)(\cdot \mid s, a) \right) &= \inf_{\Gamma(\cdot \mid s, a) \in \mathcal{C}([\mathcal{L}_t^\pi p_t](\cdot \mid s, a), [\mathcal{L}_t^\pi q_t](\cdot \mid s, a))} \mathbb{E}_{(X, Y) \sim \Gamma(\cdot \mid s, a)} [d(X, Y)^k]^{1/k} \\ &\leq \mathbb{E}_{(X, Y) \sim \Theta(\cdot \mid s, a)} [d(X, Y)^k]^{1/k} \\ &= \left( (1 - \gamma) \mathbb{E}_{(X \sim P(\cdot \mid s, a), Y \sim \delta_X)} [d(X, Y)^k] + \gamma \mathbb{E}_{(X, Y) \sim [P^\pi \tilde{\Gamma}](\cdot \mid s, a)} [d(X, Y)^k] \right)^{1/k} \\ &= \gamma^{1/k} \mathbb{E}_{s' \sim P(\cdot \mid s, a), (X, Y) \sim \tilde{\Gamma}(\cdot \mid s', \pi(s'))} [d(X, Y)^k]^{1/k}. \end{aligned}$$

Since this holds for any coupling Γ( ˜ ·|s, a) ∈ C(pt(·|s, a), qt(·|s, a)), we can take the infimum over all such couplings on the right-hand side, so that

$$\begin{aligned} W_k \left( (\mathcal{B}_t^\pi p_t)(\cdot \mid s, a), (\mathcal{B}_t^\pi q_t)(\cdot \mid s, a) \right) &\leq \gamma^{1/k} \left( \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ \Gamma \in \mathcal{C}(p_t(\cdot \mid s', \pi(s')), q_t(\cdot \mid s', \pi(s'))) \mathbb{E}_{(X,Y) \sim \Gamma}[d(X, Y)^k] \right] \right)^{1/k} \\ &= \gamma^{1/k} \left( \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ W_k(p_t(\cdot \mid s', \pi(s')), q_t(\cdot \mid s', \pi(s')))^k \right] \right)^{1/k} \\ &\leq \gamma^{1/k} \sup_{s, a} W_k(p_t(\cdot \mid s, a), q_t(\cdot \mid s, a)). \end{aligned}$$

Taking the supremum over (s, a) of the left-hand side concludes the proof.

### E.4. Analysis of TD-CFM

We study the learning dynamics of an idealized variant of TD-CFM which minimizes the flow-matching loss exactly. Starting from an arbitrary vector field v (0) t , at each iteration n ≥ 0 we compute

$$v_t^{(n+1)}(\cdot|s, a) \in \arg \min_{v_t(\cdot): \mathbb{R} d \rightarrow \mathbb{R}^d} \ell_{\text{TD-CFM}}^{(n)}(t, s, a) := \mathbb{E}_{X_1 \sim (\mathcal{T}^\pi m_1^{(n)})(s, a), X_t \sim p_{t|1}(\cdot|X_1)} \left[ \|v_t(X_t) - u_{t|1}(X_t|X_1)\|^2 \right], \quad (29)$$

where m (n) t (x|s, a) is the probability path generated by v (n) t (x|s, a).

Lemma 6. *For any* n ≥ 0*, the vector field minimizing* [\(29\)](#page-34-0) *is*

$$v_t^{(n+1)}(x \mid s, a) = \int u_{t|1}(x|x_1) \frac{p_{t|1}(x \mid x_1)(\mathcal{T}^\pi m_1^{(n)})(x_1 \mid s, a)}{m_t^{(n+1)}(x|s, a)} dx_1,$$

*Proof.* Note that [\(29\)](#page-34-0) is a standard flow matching loss for the target distribution T <sup>π</sup>m (n) 1 . The expression of v (n+1) t (x | s, a) given in the statement is exactly the vector field obtained by marginalization of the conditional vector field ut|1, which we know to be the minimizer of the loss from Theorem 2 of [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2). The fact that v (n+1) <sup>t</sup> generates m (n+1) t is a consequence of Theorem 1 of [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2).

Lemma 7. *For any* n ≥ 0*, the probability path generated by* [\(29\)](#page-34-0) *satisfies* m (n+1) 1 (x|s, a) = T <sup>π</sup>m (n) 1 (x|s, a)*.*

*Proof.* This is immediate from the definition of conditional probability path, as we set p1|1(x | x1) = δ(x − x1) by construction, where δ(·) is the Dirac's delta function.

Theorem 5. *For any* n ≥ 1*, the probability path generated by* [\(29\)](#page-34-0) *satisfies*

$$m_t^{(n+1)}(x|s, a) = (\mathcal{B}_t^\pi m_t^{(n)})(x|s, a),$$

*where* B π t *is the operator of [Definition 1.](#page-33-4) Moreover, if the initial vector field* v (0) t *satisfies*

$$v_t^{(0)}(x \mid s, a) = \int u_{t|1}(x|x_1) \frac{p_{t|1}(x \mid x_1) m_t^{(0)}(x_1 \mid s, a)}{m_t^{(0)}(x|s, a)} dx_1,$$

*with* m (0) <sup>t</sup> *being its generated proability path, then this result is valid at all* n ≥ 0*.*

*Proof.* We know that, for all n ≥ 0, v n+1 <sup>t</sup> generates m (n+1) t [\(Lemma 6\)](#page-34-1) and that m (n+1) <sup>1</sup> = T <sup>π</sup>m (n) 1 [\(Lemma 7\)](#page-35-1). Note that m (n+1) t is written as a function of m (n) 1 only, i.e., at each iteration we keep only the distribution generated at time t = 1 (m (n) 1 ) and discard the associated probability path (m (n) t for t < 1). We can however express m (n+1) t as a function of m (n) t thanks to the linearity of the Bellman operator and the definition of marginal paths. For any n ≥ 1,

$$\begin{aligned}
m_t^{(n+1)}(x \mid s, a) &:= \int p_{t|1}(x \mid x_1) (\mathcal{T}^\pi m_1^{(n)})(x_1 \mid s, a) dx_1 \\
&= \int p_{t|1}(x \mid x_1) \left( (1 - \gamma) P(x_1 \mid s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ m_1^{(n)}(x_1 \mid s', \pi(s')) \right] \right) dx_1 \\
&= (1 - \gamma) \int p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ \int p_{t|1}(x \mid x_1) m_1^{(n)}(x_1 \mid s', \pi(s')) dx_1 \right] \\
&= (1 - \gamma) \int p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ \int p_{t|1}(x \mid x_1) (\mathcal{T}^\pi m_1^{(n-1)})(x_1 \mid s', \pi(s')) dx_1 \right] \\
&= (1 - \gamma) \int p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ m_t^{(n)}(x \mid s', \pi(s')) \right] \\
&= (1 - \gamma) P_t(x \mid s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ m_t^{(n)}(x \mid s', \pi(s')) \right] = (\mathcal{B}_t^\pi m_t^{(n)})(x \mid s, a).
\end{aligned}$$

This proves the first part of the statement. For the second part, we only need to prove that the result also holds at n = 0. Note that the assumption on v (0) t implies that m (0) t (x | s, a) := R pt|1(x | x1)m (0) 1 (x<sup>1</sup> | s, a)dx1. Thus,

$$\begin{aligned}
m_t^{(1)}(x \mid s, a) &:= \int p_{t|1}(x \mid x_1) (\mathcal{T}^\pi m_1^{(0)})(x_1 \mid s, a) dx_1 \\
&= \int p_{t|1}(x \mid x_1) \left( (1 - \gamma) P(x_1 \mid s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ m_1^{(0)}(x_1 \mid s', \pi(s')) \right] \right) dx_1 \\
&= (1 - \gamma) \int p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ \int p_{t|1}(x \mid x_1) m_1^{(0)}(x_1 \mid s', \pi(s')) dx_1 \right] \\
&= (1 - \gamma) \int p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ m_t^{(0)}(x \mid s', \pi(s')) \right] = (\mathcal{B}_t^\pi m_t^{(0)})(x \mid s, a).
\end{aligned}$$

#### E.5. Analysis of TD-CFM(C)

The idealized update of TD-CFM(C) is, for any n ≥ 0,

$$v_t^{(n+1)}(\cdot|s, a) \in \underset{v_t(\cdot):\mathbb{R}^d \rightarrow \mathbb{R}^d}{\text{arg min}} \ell_{\text{TD-CFM}(C)}^{(n)}(t, s, a), \text{ where } (30)$$

$$\ell_{\text{TD-CFM}(C)}^{(n)}(t, s, a) := \mathbb{E}_{(X_0, X_1) \sim \Gamma_{0,1}^{(n)}(\cdot|s, a), X_t \sim p_{t|0,1}(\cdot|X_0, X_1)} \left[ \|v_t(X_t) - u_{t|0,1}(X_t \mid X_0, X_1)\|^2 \right],$$

and Γ (n) 0,1 (· | s, a) is the coupling between m<sup>0</sup> and T <sup>π</sup>m (n) 1 , while pt|0,1, ut|0,<sup>1</sup> are such that ut|0,1(x | x0, x1) generates pt|0,1(x | x0, x1), p1|0,1(x | x0, x1) = δ<sup>x</sup><sup>1</sup> (x), and

$$p_{t|1}(x \mid x_1) = \int p_{t|0,1}(x \mid x_0, x_1) m_0(x_0) dx_0. \quad (31)$$

Lemma 8. *The coupling* Γ (n) 0,1 (· | s, a) *satisfies*

$$\Gamma_{0,1}^{(n)}(x_0, x_1 \mid s, a) = (1 - \gamma)P(x_1 \mid s, a)m_0(x_0) + \gamma \mathbb{E}_{S' \sim P(\cdot \mid s, a)} \left[ m_{0,1}^{(n)}(x_0, x_1 \mid S', \pi(S')) \right],$$

*where* m (n) 0,1 (x0, x<sup>1</sup> | s, a) = <sup>m</sup>0(x0)δ<sup>ψ</sup> (n) (x0|s,a) (x1) *is the joint distribution of* (X0, X1)*, i.e the endpoints of the ODE.*

*Proof.* For any x0, x1, we can write Γ (n) 0,1 (x0, x<sup>1</sup> | s, a) = Γ(n) 1 (x<sup>1</sup> | s, a, x0)m0(x0), where Γ (n) is the corresponding conditional distribution. By definition, we have

$$\Gamma_1^{(n)}(x_1 \mid s, a, x_0) = (1 - \gamma)P(x_1 \mid s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ \delta_{\psi_1^{(n)}(x_0 \mid s', \pi(s'))}(x_1) \right]$$

where ψ (n) 1 is the flow that generates m (n) 1 . Multiplying both sides by m0(x0) and using that m (n) 0,1 (x0, x<sup>1</sup> | s, a) = <sup>m</sup>0(x0)δ<sup>ψ</sup> (n) (x0|s,a) (x1) concludes the proof.

Lemma 9. *For any* n ≥ 0*, the vector field minimizing* [\(30\)](#page-36-0) *is*

$$v_t^{(n+1)}(x \mid s, a) = \int \int u_{t|0,1}(x \mid x_0, x_1) \frac{p_{t|0,1}(x \mid x_0, x_1) \Gamma_{0,1}^{(n)}(x_0, x_1 \mid s, a)}{m_t^{(n+1)}(x \mid s, a)} dx_0 dx_1,$$

*where* m (n+1) t (<sup>x</sup> | s, a) := R R <sup>p</sup>t|0,1(<sup>x</sup> | <sup>x</sup>0, x1)Γ(n) 0,1 (x0, x<sup>1</sup> | s, a)dx0dx1*. Moreover* v (n+1) <sup>t</sup> *generates* m (n+1) t *.*

*Proof.* Note that [\(30\)](#page-36-0) is a standard conditional flow matching loss since ut|0,1(x | x0, x1) generates pt|0,1(x | x0, x1) and p1|0,1(x | x0, x1) = δ<sup>x</sup><sup>1</sup> (x). The expression of v (n+1) t (x | s, a) given in the statement is exactly the vector field obtained by marginalization of the conditional vector field ut|0,1, which we know to be the minimizer of the loss from Theorem 2 of [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2). The fact that v (n+1) <sup>t</sup> generates m (n+1) t is a consequence of Theorem 1 of [\(Lipman et al.,](#page-10-2) [2023\)](#page-10-2).

Lemma 10. *For any* n ≥ 0*, the probability path generated by* [\(29\)](#page-34-0) *satisfies* m (n+1) (x | s, a) = T <sup>π</sup>m (n) 1 (x | s, a)*.*

*Proof.* By [Lemma 9](#page-36-1) and the fact that p1|0,1(x | x0, x1) = δ<sup>x</sup><sup>1</sup> (x),

$$\begin{aligned} m_1^{(n+1)}(x \mid s, a) &:= \int \int p_{1|0,1}(x \mid x_0, x_1) \Gamma_{0,1}^{(n)}(x_0, x_1 \mid s, a) dx_0 dx_1 \\ &= \int \Gamma_{0,1}^{(n)}(x_0, x \mid s, a) dx_0 \\ &= (\mathcal{T}^\pi m_1^{(n)})(x \mid s, a). \end{aligned}$$

Theorem 6. *For any* n ≥ 1*, the probability path generated by* [\(29\)](#page-34-0) *satisfies*

$$m_t^{(n+1)}(x \mid s, a) = (\mathcal{B}_t^\pi m_t^{(n)})(x \mid s, a),$$

*where* B π t *is the operator of [Definition 1.](#page-33-4) Moreover, if the initial vector field* v (0) t *satisfies*

$$u_t^{(0)}(x \mid s, a) = \int \int u_{t|0,1}(x|x_0, x_1) \frac{p_{t|0,1}(x \mid x_0, x_1) m_{0,1}^{(0)}(x_0, x_1 \mid s, a)}{m_t^{(0)}(x \mid s, a)} dx_0 dx_1,$$

*with* m (0) <sup>t</sup> *being its generated probability path, then this result is valid at all* n ≥ 0*.*

*Proof.* We know that, for all n ≥ 0, v n+1 <sup>t</sup> generates m (n+1) t [\(Lemma 9\)](#page-36-1) and that m (n+1) <sup>1</sup> = T <sup>π</sup>m (n) 1 [\(Lemma 10\)](#page-36-2). While m (n+1) t is written as a function of Γ (n) 0,1 only, we can rewrite it as a function of m (n) t thanks to the linearity of the Bellman operator and the definition of marginal paths. For any n ≥ 1, By [Lemma 8,](#page-36-3)

$$\begin{aligned} m_t^{(n+1)}(x \mid s, a) &:= \int \int p_{t|0,1}(x \mid x_0, x_1) \Gamma_{0,1}^{(n)}(x_0, x_1 \mid s, a) dx_0 dx_1 \\ &= \int \int p_{t|0,1}(x \mid x_0, x_1) \left( (1 - \gamma) P(x_1 \mid s, a) m_0(x_0) + \gamma \mathbb{E}_{S' \sim P(\cdot \mid s, a)} \left[ m_{0,1}^{(n)}(x_0, x_1 \mid S', \pi(S')) \right] \right) dx_0 dx_1 \\ &= (1 - \gamma) \underbrace{\int \int p_{t|0,1}(x \mid x_0, x_1) P(x_1 \mid s, a) m_0(x_0) dx_0 dx_1}_{(i)} \\ &\quad + \underbrace{\gamma \mathbb{E}_{S' \sim P(\cdot \mid s, a)} \left[ \int \int p_{t|0,1}(x \mid x_0, x_1) m_{0,1}^{(n)}(x_0, x_1 \mid S', \pi(S')) dx_0 dx_1 \right]}_{(ii)}. \end{aligned}$$

By [\(31\)](#page-36-4),

$$(i) = \int p_{t|1}(x \mid x_1) P(x_1 \mid s, a) dx_1 = P_t(x \mid s, a).$$

For (ii), by Lemma [9,](#page-36-1) we have m (n) t (<sup>x</sup> | s, a) = R R <sup>p</sup>t|0,1(<sup>x</sup> | <sup>x</sup>0, x1)Γ(n−1) 0,1 (x0, x<sup>1</sup> | s, a)dx0dx1, ∀n ≥ 0, which implies

$$m_{0,1}^{(n)}(x_0, x_1 \mid s', \pi(s')) = \Gamma_{0,1}^{(n-1)}(x_0, x_1 \mid s', \pi(s')).$$

Therefore, again by definition of m (n) t [\(Lemma 9\)](#page-36-1),

$$\begin{aligned} (ii) &= \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ \int \int p_{t|0,1}(x \mid x_0, x_1) \Gamma_{0,1}^{(n-1)}(x_0, x_1 \mid s', \pi(s')) dx_0 dx_1 \right] \\ &= \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ m_t^{(n)}(x \mid s', \pi(s')) \right]. \end{aligned}$$

Plugging the expressions of (i) and (ii) into the one of m (n+1) t (x | s, a) yields the first part of the statement.

For the second part, we only need to prove that the result also holds at n = 0. Note that the assumption on v (0) t implies that m (0) t (x | s, a) = R R pt|0,1(x | x0, x1)m (0) 0,1 (x0, x<sup>1</sup> | s ′ , π(s ′ ))dx0dx1. Thus, using the same decomposition above, we have

$$\begin{aligned} m_t^{(1)}(x \mid s, a) &= (1 - \gamma)P_t(x \mid s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ \int \int p_{t|0,1}(x \mid x_0, x_1) m_{0,1}^{(0)}(x_0, x_1 \mid s', \pi(s')) \mathrm{d}x_0 \mathrm{d}x_1 \right] \\ &= (1 - \gamma)P_t(x \mid s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ m_t^{(0)}(x \mid s', \pi(s')) \right], \end{aligned}$$

#### E.6. Variance Analysis

Theorem 7. *Let us define the random variables*

$$\begin{aligned} g_{\text{TD}^2\text{-CFM}}(t, s, a, s', \vec{X}_t, X_t^{(n)}) &:= (1 - \gamma) \nabla_{\theta} v_t(\vec{X}_t | s, a; \theta)^{\top} (v_t(\vec{X}_t | s, a; \theta) - u_{t|1}(\vec{X}_t | s')) \\ &\quad + \gamma \nabla_{\theta} v_t(X_t^{(n)} | s, a; \theta)^{\top} (v_t(X_t^{(n)} | s, a; \theta) - v_t^{(n)}(X_t^{(n)} | s', \pi(s'))) \\ g_{\text{TD-CFM}}(t, s, a, s', \vec{X}_t, X_1, X_t) &:= (1 - \gamma) \nabla_{\theta} v_t(\vec{X}_t | s, a; \theta)^{\top} (v_t(\vec{X}_t | s, a; \theta) - u_{t|1}(\vec{X}_t | s')) \\ &\quad + \gamma \nabla_{\theta} v_t(X_t | s, a; \theta)^{\top} (v_t(X_t | s, a; \theta) - u_{t|1}(X_t | X_1)) \end{aligned}$$

*where* t ∼ U([0, 1]),(s, a) ∼ ρ, s′ ∼ P(·|s, a)*,* → X<sup>t</sup> ∼ pt|1(·|s ′ ), X(n) <sup>t</sup> ∼ m (n) t (· | s ′ , π(s ′ )), X<sup>1</sup> ∼ m (n) 1 (· | s ′ , π(s ′ ))*, and* X<sup>t</sup> ∼ pt|1(·|X1)*. Then,* gTD<sup>2</sup> -CFM *and* gTD-CFM *are respectively unbiased estimates of the gradients* ∇θℓTD<sup>2</sup> -CFM(θ) *and* ∇θℓTD-CFM(θ)*.*

*Moreover, if we consider their respective total variations defined as:*

$$\begin{aligned}\sigma_{\text{TD-CFM}}^2 &= \text{Trace} \left( \text{Cov}_{t,s,a,s',\vec{X}_t, X_t^{(n)}} \left[ g_{\text{TD}^2\text{-CFM}}(t, s, a, s', \vec{X}_t, X_t^{(n)}) \right] \right) \\ \sigma_{\text{TD-CFM}}^2 &= \text{Trace} \left( \text{Cov}_{t,s,a,s',\vec{X}_t, X_1, X_t} \left[ g_{\text{TD-CFM}}(t, s, a, s', \vec{X}_t, X_1, X_t) \right] \right)\end{aligned}$$

*and we assume that* m (n) t (x | s, a) = R pt|1(x | x1)m (n) 1 (x<sup>1</sup> | s, a)dx1*, then we obtain*

$$\sigma_{\text{TD-CFM}}^2 = \sigma_{\text{TD}^2\text{-CFM}}^2 + \gamma^2 \mathbb{E}_{t,s,a,X_t} \left[ \text{Trace} \left( \text{Cov}_{X_1|s,a,X_t} \left[ \nabla_\theta v_t(X_t|s,a;\theta)^\top u_{t|1}(X_t \mid X_1) \right] \right) \right) \right].$$

*Proof.* Recall the TD<sup>2</sup> -CFM and TD-CFM objectives:

$$\begin{aligned} \ell_{\text{TD}^2\text{-CFM}}(\theta) &= (1 - \gamma) \mathbb{E}_{t,s,a,s',X_t \sim p_{t|1}(\cdot|s')} \left[ \left\| v_t(X_t|s, a; \theta) - u_{t|1}(X_t|s') \right\|^2 \right] \\ &\quad + \gamma \mathbb{E}_{t,s,a,s',X_t \sim m_t^{(n)}(\cdot|s', \pi(s'))} \left[ \left\| v_t(X_t|s, a; \theta) - v_t^{(n)}(X_t|s', \pi(s')) \right\|^2 \right], \\ \ell_{\text{TD-CFM}}(\theta) &= (1 - \gamma) \mathbb{E}_{t,s,a,s',X_t \sim p_{t|1}(\cdot|s')} \left[ \left\| v_t(X_t|s, a; \theta) - u_{t|1}(X_t|s') \right\|^2 \right] \\ &\quad + \gamma \mathbb{E}_{t,s,a,s',X_1 \sim m_1^{(n)}(\cdot|s', \pi(s')), X_t \sim p_{t|1}(\cdot|X_1)} \left[ \left\| v_t(X_t|s, a; \theta) - u_{t|1}(X_t|X_1) \right\|^2 \right]. \end{aligned}$$

Computing the gradients of these quantities w.r.t. θ, it is easy to check that gTD<sup>2</sup> -CFM and gTD-CFM are their unbiased estimates.

Let us now analyze the total variation of these estimators. By assumption, we have m (n) t (x | s, a) = R pt|1(x | x1)m (n) 1 (x<sup>1</sup> | s, a)dx1, which implies that X (n) t and X<sup>t</sup> follow the same law. Moreover, we obtain the following identities:

$$\begin{aligned} v_t^{(n)}(x \mid s', \pi(s')) &= \mathbb{E}_{X_1|x, s'} [u_{t|1}(x \mid X_1)], \\ g_{\text{TD}^2\text{-CFM}}(t, s, a, s', \vec{X}_t, X_t) &= \mathbb{E}_{X_1|X_1, s'} [g_{\text{TD-CFM}}(t, s, a, s', X_t^o, X_1, X_t)], \\ \mathbb{E}_{X_t \sim m_t^{(n)}(\cdot|s', \pi(s'))} [g_{\text{TD}^2\text{-CFM}}(t, s, a, s', \vec{X}_t, X_t)] &= \mathbb{E}_{X_1 \sim m_{t|1}^{(n)}(\cdot|s', \pi(s'))} [g_{\text{TD-CFM}}(t, s, a, s', \vec{X}_t, X_1, X_t)], \end{aligned}$$

where X<sup>1</sup> | x, s′ ∼ pt|1(x|X1)m (n) 1 (X1|s ′ ,π(s ′ )) m (n) (x|s,a) is the posterior distribution of X<sup>1</sup> given x and s ′ .

conditional variance, Var(X) = <sup>E</sup>[Var(X|Y )]) + Var(E[X|Y]), we conclude that

$$\begin{aligned}\sigma_{\text{TD-CFM}} &= \text{Trace} (\text{Cov}_{Y, X_1, X_t} [g_{\text{TD-CFM}}(Y, X_1, X_t)]) \\ &= \mathbb{E}_{Y, X_1, X_t} \left[ \left\| g_{\text{TD-CFM}}(Y, X_1, X_t) - \mathbb{E}_{Y, X_1, X_t} [g_{\text{TD-CFM}}(Y, X_1, X_t)] \right\|^2 \right] \\ &= \mathbb{E}_{Y, X_t} \left[ \left\| \mathbb{E}_{X_1|Y, X_t} [g_{\text{TD-CFM}}(Y, X_1, X_t)] - \mathbb{E}_{Y, X_1, X_t} [g_{\text{TD-CFM}}(Y, X_1, X_t)] \right\|^2 \right] \\ &\quad + \mathbb{E}_{Y, X_t} \left[ \mathbb{E}_{X_1|Y, X_t} \left[ \left\| g_{\text{TD-CFM}}(Y, X_1, X_t) - \mathbb{E}_{X_1|Y, X_t} [g_{\text{TD-CFM}}(Y, X_1, X_t)] \right\|^2 \right] \right] \\ &= \mathbb{E}_{Y, X_t} \left[ \left\| g_{\text{TD}^2\text{-CFM}}(Y, X_t) - \mathbb{E}_{Y, X_t} [g_{\text{TD}^2\text{-CFM}}(Y, X_t)] \right\|^2 \right] \\ &\quad + \gamma^2 \mathbb{E}_{Y, X_t} \left[ \mathbb{E}_{X_1|Y, X_t} \left[ \left\| \nabla_{\theta} v_t(X_t|s, a; \theta)^{\top} u_{t|1}(X_t \mid X_1) - \mathbb{E}_{X_1|Y, X_t} [\nabla_{\theta} v_t(X_t|s, a; \theta)^{\top} u_{t|1}(X_t \mid X_1)] \right\|^2 \right] \right] \\ &= \sigma_{\text{TD}^2\text{-CFM}} + \gamma^2 \mathbb{E}_{Y, X_t} [\text{Trace} (\text{Cov}_{X_1|Y, X_t} [\nabla_{\theta} v_t(X_t|s, a; \theta)^{\top} u_{t|1}(X_t \mid X_1)])] \\ &= \sigma_{\text{TD}^2\text{-CFM}} + \gamma^2 \mathbb{E}_{t, s, a, X_t} [\text{Trace} (\text{Cov}_{X_1|s, a, X_t} [\nabla_{\theta} v_t(X_t|s, a; \theta)^{\top} u_{t|1}(X_t \mid X_1)])].\end{aligned}$$

Theorem 8. *Let us define the random variable*

$$g_{\text{TD-CFM(C)}}(t, s, a, s', \vec{X}_t, X_0, X_1, X_t) := (1 - \gamma) \nabla_{\theta} v_t(\vec{X}_t | s, a; \theta)^{\top} (v_t(\vec{X}_t | s, a; \theta) - u_{t|0,1}(\vec{X}_t | X_0, s')) \\ + \gamma \nabla_{\theta} v_t(X_t | s, a; \theta)^{\top} (v_t(X_t | s, a; \theta) - u_{t|0,1}(X_t | X_0, X_1))$$

*where* t ∼ U([0, 1]),(s, a) ∼ ρ, s′ ∼ P(·|s, a), → X<sup>t</sup> ∼ pt|1(·|s ′ ),(X0, X1) ∼ m (n) 0,1 (· | s ′ , π(s ′ )) *and* X<sup>t</sup> ∼ pt|0,1(·|X0, X1,)*. Then* gTD-CFM(C) *is an unbiased estimate of the gradient* ∇θℓTD-CFM(C)(θ)*.*

*Moreover, if we consider its total variation defined as:*

$$\sigma_{\text{TD-CFM}(\text{C})} = \text{Trace} \left( \text{Cov}_{t,s,a,s',\vec{X}_t,X_0,X_1,X_t} \left[ g_{\text{TD-CFM}(\text{C})}(t, s, a, s', \vec{X}_t, X_0, X_1, X_t) \right] \right)$$

*and we assume that* m (n) t (x | s, a) = R R pt|0,1(x | x0, x1)m (n) 0,1 (x0, x<sup>1</sup> | s, a)dx0dx1*, then we obtain*

$$\sigma_{\text{TD-CFM}(\text{C})} = \sigma_{\text{TD}^2\text{-CFM}} + \gamma^2 \mathbb{E}_{t,s,a,X_t} \left[ \text{Trace} \left( \text{Cov}_{(X_0, X_1)|s,a,X_t} \left[ \nabla_{\theta} v_t(X_t|s, a; \theta)^\top u_{t|0,1}(X_t \mid X_0, X_1) \right] \right) \right].$$

*Furthermore, if we use straight conditional paths, i.e.,* pt|0,1(x|x0, x1) = δ(tx<sup>1</sup> + (1 − t)x<sup>0</sup> − x)*, then*

$$\begin{aligned} \sigma_{\text{TD-CFM}(\text{C})} &\leq \sigma_{\text{TD}^2\text{-CFM}} \\ &+ \gamma^2 \sup_{t,s,a,x} \left\| \nabla_\theta v_t(x|s, a; \theta) \right\|^2 \mathbb{E}_{t,s,a,s',X_0,X_1,X_t} [\|X_1 - X_0 - \mathbb{E}_{(X_1, X_0)|s, a, s', X_t} [X_1 - X_0]\|^2]. \end{aligned}$$

*In particular, when the paths of the linear interpolation* X<sup>t</sup> *do not intersect for any* s, a, s′ *, we have* Et,s,a,s′ ,X0,X1,X<sup>t</sup> -∥X<sup>1</sup> − X<sup>0</sup> − <sup>E</sup>(X1,X0)|s,a,s′ ,X<sup>t</sup> [X<sup>1</sup> − X0] ∥ = 0 *and* σTD-CFM(C) = σTD<sup>2</sup> -CFM*.*

*Proof.* The first two statements can be checked by repeating the proof of [Theorem 7](#page-38-1) with conditional paths pt|0,<sup>1</sup> and vector fields ut|0,1. Let us thus prove the second part. We know that the flow ϕt(x0, x1) that generates the the conditonal path pt|0,1(x|x0, x1) = δtx1+(1−t)x<sup>0</sup> (x) is ϕt(x0, x1) = tx<sup>1</sup> + (1 − t)x0. Its associated vector field ut|0,<sup>1</sup> is thus

$$u_{t|0,1}(\phi_t(x_0, x_1)|x_0, x_1) = \frac{d}{dt}\phi_t(x_0, x_1) = x_1 - x_0.$$

Theorefore, denoting Y = (t, s, a), we can bound the second term in the decomposition of σTD-CFM(C) as

$$\begin{aligned} & \mathbb{E}_{Y, X_t} \left[ \text{Trace} \left( \text{Cov}_{(X_0, X_1)|Y, X_t} [\nabla_{\theta} v_t(X_t|s, a; \theta)^T u_{t|1}(X_t \mid X_0, X_1)] \right) \right] \\ &= \mathbb{E}_{Y, X_t} \left[ \mathbb{E}_{X_0, X_1|Y, X_t} \left[ \left\| \nabla_{\theta} v_t(X_t|s, a; \theta)^T u_{t|0,1}(X_t \mid X_0, X_1) - \mathbb{E}_{X_0, X_1|Y, X_t} [\nabla_{\theta} v_t(X_t|s, a; \theta)^T u_{t|0,1}(X_t \mid X_0, X_1)] \right\|^2 \right] \right] \\ &\leq \mathbb{E}_{Y, X_t} \left[ \|\nabla_{\theta} v_t(X_t|s, a; \theta)\|^2 \mathbb{E}_{X_0, X_1|Y, X_t} \left[ \left\| u_{t|0,1}(X_t \mid X_0, X_1) - \mathbb{E}_{X_0, X_1|Y, X_t} [u_{t|0,1}(X_t \mid X_0, X_1)] \right\|^2 \right] \right] \\ &= \mathbb{E}_{Y, X_t} \left[ \|\nabla_{\theta} v_t(X_t|s, a; \theta)\|^2 \mathbb{E}_{X_0, X_1|Y, X_t} \left[ \left\| X_0 - X_1 - \mathbb{E}_{X_0, X_1|Y, X_t} [X_1 - X_0] \right\|^2 \right] \right] \\ &\leq \sup_{t, s, a, x} \|\nabla_{\theta} v_t(x|s, a; \theta)\|^2 \mathbb{E}_{Y, X_t} \left[ \mathbb{E}_{X_0, X_1|Y, X_t} \left[ \left\| X_0 - X_1 - \mathbb{E}_{X_0, X_1|Y, X_t} [X_1 - X_0] \right\|^2 \right] \right]. \end{aligned}$$

This proves the third statement. For the last point, simply note that if the paths generating X<sup>t</sup> do not cross, then the distribution of X0, X1|Y, X<sup>t</sup> is supported over a single couple (X0, X1), which means that its variance is zero.

#### E.7. Transport Cost Analysis

Theorem 9. *Assume that* m (n) t (x | s, a) = R pt|1(x | x1)m (n) 1 (x<sup>1</sup> | s, a)dx1*, where* pt|1(· | x1) = N (tx1,(1 − t) 2 I) *is a Gaussian path. Then, the conditional paths* [<sup>6</sup>](#page-41-1)*built by* TD-CFM(C) *and* TD<sup>2</sup> -CFM *to generate* m (n+1) <sup>1</sup> = T <sup>π</sup>m (n) 1 *induce a smaller transport cost than those built by* TD-CFM*. Formally, for every* t, s, a*,*

$$\mathbb{E}_{t,s,a,s',X_0 \sim m_0, X_1 \sim (1-\gamma)\delta_{s'} + \gamma\delta_{\psi_1^{(n)}(X_0)_{s'}, \pi(s')}} [\|X_1 - X_0\|^2] \leq \mathbb{E}_{t,s,a,s',X_0 \sim m_0, X_1 \sim [T\pi m_1^{(n)}](\cdot | s, a)} [\|X_1 - X_0\|^2].$$

*Proof.* The paths generated by TD-CFM(C) and TD<sup>2</sup> -CFM induce the same transport cost since both algorithms connect the endpoints of the ODE path m (n) t in the bootstrapped term. Hence,

$$\begin{aligned} & \mathbb{E}_{t,s,a,s',X_0 \sim m_0, X_1 \sim (1-\gamma)\delta_{s'} + \gamma \delta_{\psi_1^{(n)}(X_0|s', \pi(s'))} [\|X_1 - X_0\|^2] \\ &= (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \mathbb{E}_{t,s,a,s',X_0} [\|\psi_1^{(n)}(X_0 \mid s', \pi(s')) - X_0\|^2] \\ &\stackrel{(a)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \mathbb{E}_{t,s,a,s',X_0} \left[ \left\| \int v_t^{(n)}(\psi_t^{(n)}(X_0 \mid s', \pi(s'))) dt \right\|^2 \right] \\ &\stackrel{(b)}{\leq} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_0} \left[ \left\| v_t^{(n)}(\psi_t^{(n)}(X_0 \mid s', \pi(s'))) \right\|^2 \right] dt \\ &\stackrel{(c)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_t \sim m_t^{(n)}(\cdot \mid s', \pi(s'))} \left[ \left\| v_t^{(n)}(X_t \mid s', \pi(s')) \right\|^2 \right] dt \\ &\stackrel{(d)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_t \sim m_t^{(n)}(\cdot \mid s', \pi(s'))} \left[ \left\| \mathbb{E}_{X_1 \mid s', X_t} [u_{t|1}(X_t \mid X_1)] \right\|^2 \right] dt \\ &\stackrel{(e)}{\leq} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_t \sim m_t^{(n)}(\cdot \mid s', \pi(s'))} \left[ \mathbb{E}_{X_1 \mid s', X_t} \left[ \left\| u_{t|1}(X_t \mid X_1) \right\|^2 \right] \right] dt \\ &\stackrel{(f)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_1 \sim m_1^{(n)}(\cdot \mid s', \pi(s')), X_t \sim p_{t|1}(\cdot \mid X_1)} \left[ \left\| u_{t|1}(X_t \mid X_1) \right\|^2 \right] dt \\ &\stackrel{(g)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_1 \sim m_1^{(n)}(\cdot \mid s', \pi(s')), X_0} \left[ \left\| u_{t|1}(tX_1 + (1-t)X_0 \mid X_1) \right\|^2 \right] dt \\ &\stackrel{(h)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \int \mathbb{E}_{t,s,a,s',X_1 \sim m_1^{(n)}(\cdot \mid s', \pi(s')), X_0} \left[ \left\| X_1 - X_0 \right\|^2 \right] dt \\ &\stackrel{(i)}{=} (1-\gamma) \mathbb{E}_{t,s,a,s',X_0} [\|s' - X_0\|^2] + \gamma \mathbb{E}_{t,s,a,s',X_1 \sim m_1^{(n)}(\cdot \mid s', \pi(s')), X_0} \left[ \left\| X_1 - X_0 \right\|^2 \right] \\ &\stackrel{(j)}{=} \mathbb{E}_{t,s,a,s',X_0 \sim m_0, X_1 \sim [\mathcal{T}^\pi m_1^{(n)}](\cdot \mid s, a)} [\|X_1 - X_0\|^2], \end{aligned}$$

where (a) uses the definition of flow as integration of a vector field, (b) uses Cauchy-Schwarz inequality, (c) uses that m<sup>0</sup> ∗ ψ (n) t is the pushforward measure generating m (n) t , (d) defines X<sup>1</sup> | x, s′ ∼ pt|1(x|X1)m (n) (X1|s ,π(s )) m (n) (x|s,a) as the posterior distribution of X<sup>1</sup> given x, s′ and uses that v (n) t is in marginal form by assumption, (e) uses Jensen's inequality, (f) uses the Tower property of expectations, (g) uses the definition of pt|<sup>1</sup> and the corresponding linear-interpolation flow, (h) uses the definition of ut|1, (i) is trivial, and (j) simply combines the two terms using the definition of Bellman operator T π .

<sup>6</sup>Recall that, given a marginal probability path m (n) t (x | s, a), the conditional probability path built by TD-CFM(C) and TD<sup>2</sup> -CFM to generate T <sup>π</sup>m (n) 1 is a linear interpolation between noise <sup>X</sup><sup>0</sup> ∼ <sup>m</sup><sup>0</sup> and <sup>X</sup><sup>1</sup> ∼ (1 − γ)δs′ <sup>+</sup> γψ(n) 1 (X0|s ′ , π(s ′ )), while the one built by TD-CFM is a linear interpolation between noise X<sup>0</sup> ∼ m<sup>0</sup> and a sample X<sup>1</sup> ∼ [T <sup>π</sup>m (n) 1 ](· | s, a) from the target distribution.