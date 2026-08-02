# 

Jesse Farebrother 1 2 † Matteo Pirotta 3 Andrea Tirinzoni 3 **Remi Munos** ´
3 Alessandro Lazaric 3 **Ahmed Touati** 3

## Abstract

Predictive models of the future are fundamental for an agent's ability to reason and plan. A common strategy learns a world model and unrolls it step-by-step at inference, where small errors can rapidly compound. Geometric Horizon Models (GHMs) offer a compelling alternative by directly making predictions of future states, avoiding cumulative inference errors. While GHMs can be conveniently learned by a generative analog to temporal difference (TD) learning, existing methods are negatively affected by bootstrapping predictions at train time and struggle to generate high-quality predictions at long horizons. This paper introduces Temporal Difference Flows (TD- Flow), which leverages the structure of a novel Bellman equation on probability paths alongside flow-matching techniques to learn accurate GHMs at over 5× the horizon length of prior methods.

Theoretically, we establish a new convergence result and primarily attribute TD-Flow's efficacy to reduced gradient variance during training. We further show that similar arguments can be extended to diffusion-based methods. Empirically, we validate TD-Flow across a diverse set of domains on both generative metrics and downstream tasks, including policy evaluation. Moreover, integrating TD-Flow with recent behavior foundation models for planning over policies demonstrates substantial performance gains, underscoring its promise for long-horizon decision-making.

## 1. Introduction

Predictive modeling lies at the heart of intelligent decisionmaking, enabling agents to reason and plan in complex environments. In Reinforcement Learning (RL), this pre-
†Work done at Meta 1McGill University 2Mila - Quebec AI ´
Institute 3FAIR at Meta. Correspondence to: Jesse Farebrother
<jfarebro@cs.mcgill.ca>, Ahmed Touati <atouati@meta.com>.

dictive capability has traditionally been achieved through world models that capture the transition structure of the environment. These models have enabled significant advances across numerous domains - from robotics manipulation employing model-predictive control (Sikchi et al., 2021; Hafner et al., 2023; Hansen et al., 2022; 2024), to sampleefficient exploration strategies (Schmidhuber, 1991; Stadie et al., 2016; Pathak et al., 2017), and sophisticated planning algorithms (Silver et al., 2016; 2017; Schrittwieser et al., 2020). However, while world models have demonstrated impressive results, they face fundamental limitations when deployed for long-horizon reasoning. The standard approach of unrolling predictions step-by-step leads to compounding errors, as small inaccuracies in each prediction accumulate and propagate forward in time (Talvitie, 2014; Jafferjee et al., 2020; Lambert et al., 2022). This "curse of horizon" presents a significant challenge for applications requiring reliable long-range predictions. An alternative approach is to learn a generative model of future states directly, avoiding compounding errors during inference. These models, usually referred to as Geometric Horizon Models (GHM; Thakoor et al., 2022) or γ-models (Janner et al., 2020), are learned by leveraging the temporal difference structure of the successor measure (Blier et al., 2021). However, their reliance on bootstrapped predictions during training can lead to instability and growing inaccuracy over long horizons. As a result, current methods struggle to make accurate predictions beyond 2050 steps, also limiting their utility for long-term decisionmaking. In this paper, we show that while state-of-the-art generative methods like flow matching (Lipman et al., 2023) and denoising diffusion (Ho et al., 2020) cannot be directly applied to learn long-horizon GHMs, their iterative nature can be leveraged to better exploit the temporal difference structure of the problem. This insight yields a new class of methods that provably converges to the successor measure while reducing the variance of their sample-based gradient estimates, enabling stable long-horizon predictions. Empirically, our approach produces significantly more accurate GHMs at all horizons, consistently outperforming state-ofthe-art algorithms across domains and metrics, including prediction accuracy, value function estimation, and generalized policy improvement.

1

## 2. Background

In the following, we use capital letters to denote random variables, sans-serif fonts for sets, and P(A) to denote the space of probability measures over a measurable set A. Markov Decision Process We consider a reward-free discounted Markov decision process M = (S, A*, P, γ*), which characterizes the dynamics of a sequential decision-making problem. At each step, the agent selects an action a ∈ A in state s ∈ S according to its policy π : S → A. This action influences the transition to the next state s
′ ∈ S, governed by the transition kernel P : S × A → P(S), which defines a probability measure over successor states. The discount factor γ ∈ [0, 1) can be interpreted as implying a process that either continues with probability γ or terminates with probability 1 − γ. This interpretation naturally defines a geometric distribution of future states the agent will occupy, where states reached after k steps are discounted by γ k.

Successor Measure The normalized successor measure (Dayan, 1993; Blier et al., 2021) of a policy π describes the discounted distribution of future states visited by π starting from an initial state-action pair (s, a). For the measurable subset X ⊆ S the successor measure mπ(X | *s, a*)
represents the probability that future states fall within X, geometrically discounted by γ according to the time of visitation. Formally, it is defined as:

$$\begin{array}{c}{{m^{\pi}(\mathsf{X}\mid s,a\,)=}}\\ {{(1-\gamma)\sum_{k=0}^{\infty}\gamma^{k}\Pr(S_{k+1}\in\mathsf{X}\mid S_{0}=s,\,A_{0}=a,\,\pi),}}\end{array}$$

where Pr(· | S0, A0, π) denotes the probability of stateaction sequences (Sk, Ak)k≥0 generated from (S0, A0) following Sk ∼ P(· | Sk−1, Ak−1) and Ak = π(Sk). The successor measure encapsulates the long-term dynamics of π, enabling value estimation for any reward function r : S → R. Specifically, the value of taking action a ∈ A in state s ∈ S is the expected reward under states visited by π amplified by the effective horizon (1 − γ)
−1:

$$Q^{\pi}(s,a)=(1-\gamma)^{-1}\,\mathbb{E}_{X\sim m^{\pi}}(\cdot|s,a)\big{[}r(X)\big{]}\,.\tag{2}$$

Moreover, mπis the fixed point of the Bellman operator T
π: P(S)
S×A → P(S)
S×A (Thakoor et al., 2022):

$$m^{\pi}(\cdot\mid s,a)=({\cal T}^{\pi}m^{\pi})\,(\cdot\mid s,a)\tag{3}$$ $$:=(1-\gamma)P(\cdot\mid s,a)+\gamma\,(P^{\pi}m^{\pi})\,(\cdot\mid s,a)\,.$$

The operator P
πapplied to m mixes the one-step kernel with the successor measure, accounting for transitioning from (*s, a*) to a new state-action pair (s
′, π(s
′)) and querying the successor measure m(· | s,′ π(s
′)) thereafter:

$$(P^{\pi}m)\,(\mathrm{d}x\mid s,a)=\int_{s^{\prime}}P(\mathrm{d}s^{\prime}\mid s,a)\,m(\mathrm{d}x\mid s^{\prime},\pi(s^{\prime}))\,.$$

Geometric Horizon Model A *Geometric Horizon Model* (GHM; Thakoor et al., 2022) or γ-model (Janner et al., 2020) is a generative model of the *normalized* successor
measure. To learn the parametric model me (*· · ·* ; θ) ≈ mπ
we can minimize a Monte-Carlo cross-entropy objective over source states from the empirical distribution ρ as,
$$\mathbb{E}_{s\sim\rho,\,X\sim m^{\pi}(\cdot|S,\pi(A))}\big[\,-\log\widetilde{m}(X\mid S,A;\theta))\,\big]\,.$$
In order to sample from mπ we deploy policy π for t ∼ Geom(1 − γ) steps resulting in state X = St. Similar to
other Monte Carlo methods in RL, this approach is problematic when learning from off-policy data, often resulting in high-variance estimators that rely on importance sampling. Alternatively, we can leverage the Bellman equation (3) to construct an off-policy iterative method for estimating
mπ. Given initial weights θ
(0), each iteration updates θ by
minimizing the following temporal-difference cross-entropy
objective over transitions that need not come from policy π,
$$\mathbb{E}_{(S,A)\sim\rho,X\sim\left(\mathcal{T}^{\pi}\,\widetilde{m}^{(n)}\right)(\cdot|S,A)}[-\log\widetilde{m}(X\mid S,A;\theta)].\tag{4}$$
In the equation above and throughout the paper, we adopt
the shorthand me
(n) = me (*· · ·* ; θ
(n)). To generate samples
X ∼T
πme
(n)(· | *S, A*) we first draw a successor state
S
′ ∼ P(· | *S, A*); then with probability 1 − γ, we return
S
′; otherwise, with probability γ, we return a *bootstrapped*
sample drawn from me
(n)(· | S
′, π(S
′)).

Several probabilistic models have been applied to this problem, including generative adversarial networks (e.g., Janner et al., 2020; Wiltzer et al., 2024b), normalizing flows (e.g., Janner et al., 2020), and variational auto-encoders (e.g., Thakoor et al., 2022; Tomar et al., 2024). We now turn our attention to a class of generative models based on the flowmatching framework specifically designed to leverage the underlying structure of the Bellman equation (3), enabling more effective generative models of the successor measure.

## 3. Temporal Difference Flows

Flow Matching (FM; Lipman et al., 2023; 2024; Liu et al., 2023; Albergo & Vanden-Eijnden, 2023) constructs a timedependent probability path mt : S × A → P(S) for t ∈
[0, 1] that evolves smoothly from the source distribution m0 = p0 ∈ P(S) to the target distribution m1 ≈ mπ. This evolution is governed by a vector field vt : S × S × A →
S, which dictates the instantaneous movement of samples along mt. The relationship between vt and the resulting probability path mt is established through a time-dependent flow ψt : S × S × A → S, defined by the following ODE:

$${\frac{\mathrm{d}}{\mathrm{d}t}}\psi_{t}(x\mid s,a)=v_{t}\big(\psi_{t}(x\mid s,a)\mid s,a\big),\ \psi_{0}(x\mid s,a)=x$$ $$\iff\ \psi_{t}(x\mid s,a)=x+\int_{0}^{t}v_{\tau}\big(\psi_{\tau}(x|s,a)\,|\,s,a\big)\,\mathrm{d}\tau\,.$$

TDCFM Coupled TDCFM TD²CFM
We say that vt generates mt if its flow ψt satisfies Xt := ψt(X0 | S, A) ∼ mt(· | *S, A*) for X0 ∼ m0. In words, the flow ψt pushes samples forward through time, ensuring they are distributed according to mt at time t. To learn this transformation, we can minimize the squared L
2 distance between a parameterized vector field v˜t(*· · ·* ; θ) and the true vector field vt over t ∼ U([0, 1]), yielding the Monte-Carlo Flow Matching (MC-FM) loss ℓMC-FM(θ):

$\mathbb{E}_{\rho,t,X_{t}}\left[\left|\left|\tilde{v}_{t}(X_{t}\mid S,A;\theta)-v_{t}(X_{t}\mid S,A)\right|\right|^{2}\right],$  where $X_{t}\sim m_{t}(\cdot\mid S,A)$. (MC-FM; 5)
Despite its conceptual simplicity, direct optimization of the flow matching objective above proves challenging due to the inaccessibility of the true probability path mt and its associated vector field vt. Alternatively, Lipman et al. (2023) shows that we can sidestep this problem entirely by introducing additional conditioning information. Instead of directly modeling the probability path mt we can introduce a random variable Z
and define a *conditional path* on Z as pt|Z : S×Z → P(S)
(Lipman et al., 2024; Tong et al., 2024). The conditional velocity field ut|Z : S×Z → S that generates pt|Z can now be computed in closed form for many simple choices of Z and pt|Z. One such choice is taking Z = X1 and performing a linear Gaussian interpolation from X0 → X1 resulting in pt|1(· | X1) = N (· | tX1, (1 − t)
2I) with the corresponding vector field given by ut|1(x | X1) = (X1 − x)/(1 − t). Armed with the ability to sample from pt|1 and to compute ut|1, we can directly learn v˜t by optimizing the Monte-Carlo Conditional Flow Matching (MC-CFM) objective ℓMC-CFM(θ):

$$\mathbb{E}_{\rho,t,Z,X_{t}}\left[\left|\left|\tilde{v}_{t}(X_{t}\mid S,A;\theta)-u_{t|Z}(X_{t}\mid Z)\right|\right|^{2}\right],$$  where $Z=X_{1}\sim m^{\pi}(\cdot\mid S,A)\,,X_{t}\sim p_{t|Z}(\cdot\mid Z)\,.$  (MC-CFM; 6)
Remarkably, both (MC-FM; 5) and (MC-CFM; 6) share the
same gradient and converge to the same solution.
Proposition 1 (Lipman et al. 2024). Given a conditional probability path pt|Z and vector field ut|Z with their associated marginal counterparts pt(x) and vt(x)*, we have*

$$\nabla_{\theta}\,\ell_{\mathrm{MC-FM}}(\theta)=\nabla_{\theta}\,\ell_{\mathrm{MC-CFM}}(\theta).$$

TD-CFM While (MC-CFM; 6) requires direct access to samples from the target distribution mπ, we can instead learn from an offline dataset ρ containing only one-step transitions (*S, A, S*′) through an iterative process similar to (4).

Starting with initial parameters θ
(0), at each iteration, we minimize the TD-Conditional Flow Matching (TD-CFM)
loss ℓTD-CFM - an extension of (MC-CFM; 6) that differs only in its sampling procedure:

X0 ∼ p0 Z = X1 ∼ (1 − γ) δS′ + γ δψe(n) 1(X0 | S′,π(S′))  . (TD-CFM; 7)
In this procedure, with probability 1 − γ, we return the successor state S
′. Otherwise, with probability γ we sample from the neural ordinary differential equation
(Chen et al., 2018) ψe
(n)
t with corresponding vector field v˜
(n)
t(Xt | S
′, π(S
′)) from X0 ∼ p0 to produce a sample X1 ∼ me
(n)(· | S
′, π(S
′)).

Coupled TD-CFM Although (TD-CFM; 7) offers a principled way of learning the flow from noise to data, an increasingly popular strategy to improve flow matching methods is to correlate noise and data whenever a "natural" coupling is available (e.g., Liu et al., 2023; Shi et al., 2023; Pooladian et al., 2023; Tong et al., 2024; De Bortoli et al., 2024). Motivated by this idea, we observe that the process used to generate X1 described above already provides a direct coupling between X0 and X1. We can leverage this coupling by conditioning the probability path pt|Z on both endpoints, i.e., Z = (X0, X1), rather than just conditioning on Z = X1 as in TD-CFM. As illustrated in Figure 1, this coupling helps align Xt with the path generated by ψe(n)
t, potentially simplifying the regression problem. This procedure gives rise to the Coupled TD-Conditional Flow Matching (TD-CFM(C)) loss ℓTD-CFM(C) which now extends ℓTD-CFM, again, differing only in its sampling procedure:

$X_{0}\sim p_{0}$  $X_{1}\sim(1-\gamma)\,\delta_{S^{\prime}}+\gamma\,\delta_{\widetilde{\psi}_{1}^{(n)}}(X_{0}|S^{\prime},\pi(S^{\prime}))$  $Z=(X_{0},X_{1})\,.$  (TD-CFM(C); 8)
A convenient approach to specifying the conditional path pt|Z is to define Xt = ϕt(X0, X1) = αtX1 + βtX0 as the affine interpolant between X0 and X1, with the interpolation coefficients satisfying the boundary conditions α0 = β1 = 0, α1 = β0 = 1, and monotonicity constraints α˙ t > 0, −β˙t > 0, where the over-dot denotes the time derivative. From this definition, the conditional vector field arises as the time derivative of this interpolant defined as ut|0,1(Xt | X0, X1) = ϕ˙t(X0, X1) = ˙αtX1 + β˙tX0 (Albergo et al., 2023). A simple choice of the interpolation coefficients that yields a linear (straight-line) conditional path is given by βt = 1 − αt = 1 − t.

TD2**-CFM** While (TD-CFM(C); 8) improves upon
(TD-CFM; 7) by accounting for the coupling between bootstrapped samples and their generating noise, both methods rely upon fitting an ad-hoc conditional vector field ut|Z that generates the surrogate conditional path pt|Z. To formulate a more structured approach, we exploit the linearity of the Bellman equation, as detailed in the following result.

Lemma 1. Let →
pt be a probability path for P generated by vector field →
vt and 
↷

p
(n)
t be a probability path for P
πm
(n) 1 generated by ↷
v
(n)
t*such that* →
p0 =
↷

p
(n)
0 = m0*. For any* t ∈ [0, 1] and (s, a) let v
(n+1)
t(· | s, a) *be the solution of* 1

$$\begin{array}{c}{{\arg\min\left(1-\gamma\right)\mathbb{E}_{\vec{X}_{t}\sim\vec{p_{t}}\left(\cdot\mid s,a\right)}\left[\left\|v(\vec{X}_{t})-\vec{v}_{t}(\vec{X}_{t}\mid s,a)\right\|^{2}\right]}}\\ {{v:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}}}\\ {{\qquad+\gamma\mathbb{E}_{\vec{X}_{t}\sim\vec{p_{t}}\left(\cdot\mid s,a\right)}\left[\left\|v(\vec{X}_{t})-\hat{v}_{t}^{(n)}(\hat{X}_{t}\mid s,a)\right\|^{2}\right].}}\end{array}$$

Then v
(n+1)
t*induces a probability path* m
(n+1)
t*such that* m
(n+1)
0 = m0 and m
(n+1)
1 = T
πm
(n)
1.

This result shows that it is possible to use two independent probability paths for the two terms in the sampling process induced by the Bellman operator. For the first term, we can use a standard CFM approach for Z = X1 with conditional path →
pt|1 and vector field →
ut|1, which induces the marginal,

$${\vec{v}}_{t}(x|s,a)=\int{\vec{u}}_{t|1}(x\mid x_{1}){\frac{{\vec{p}}_{t|1}(x\mid x_{1})P(\mathrm{d}x_{1}|s,a)}{{\vec{p}}_{t}(x|s,a)}},$$

where →
pt(x|*s, a*) = R →
pt|1(x|s
′)P(ds
′|*s, a*). For the second term, we can leverage the GHM m
(n)
tlearned at the previous iteration to construct the marginal,

$$\widetilde{v}_{t}^{(n)}(x|s,a){=}{\int}\,v_{t}^{(n)}(x|s^{\prime},a^{\prime})\frac{m_{t}^{(n)}(x|s^{\prime},a^{\prime})P(\mathrm{d}s^{\prime}|s,a)}{\widetilde{p}_{t}^{(n)}(x|s,a)},$$

$$P_{t}\quad(x^{\prime}|s,a)$$  where $\widehat{P}_{t}^{(n)}(x\mid s,a)=\int m_{t}^{(n)}(x\mid s^{\prime},a^{\prime})P(\mathrm{d}s^{\prime}\mid s,a)$, and $a^{\prime}=\pi(s^{\prime})$. This shows that $m_{t}^{(n)}$ plays the role of a conditional probability path for the bootstrapped term and $v_{t}^{(n)}$ is its associated conditional vector field. We can then use the equivalence between FM and cfFM in Proposition 1 to replace the marginal probability paths and vector fields in Lemma 1.  

with their conditional counterparts to obtain the loss:
$$\begin{array}{c}{{\vec{\ell}(\theta)=\mathbb{E}_{\rho,t,Z,\vec{X}_{t}}\left[\left|\left|\vec{v}_{t}(\vec{X}_{t}\mid S,A;\theta)-\vec{u}_{t\mid Z}(\vec{X}_{t}\mid Z)\right|\right|^{2}\right],}}\\ {{\mathrm{~where~}Z=X_{1}\sim P(\cdot\mid S,A),\,\vec{X}_{t}\sim\vec{p}_{t\mid Z}(\cdot\mid Z)\,,}}\end{array}$$
$$\widehat{\ell}(\theta)=\mathbb{E}_{\rho,t,\widehat{X}_{t}}\left[\left\|\,\bar{v}_{t}(\widehat{X}_{t}\,|\,S,A;\theta)-\bar{v}_{t}^{(n)}(\widehat{X}_{t}\,|\,S^{\prime},\pi(S^{\prime})\|^{2}\right],\right.$$ $$\left.\text{where}X_{0}\sim p_{0},\,S^{\prime}\sim P(\cdot\mid S,A),\right.$$ $$\left.\widehat{X}_{t}=\widehat{\psi}_{t}^{(n)}(X_{0}\mid S^{\prime},\pi(S^{\prime}))\,,\right.$$

$$\ell_{_\mathrm{TD^{2}-CFM}}(\theta)=(1-\gamma)\vec{\ell}(\theta)+\gamma\widehat{\ell}(\theta)\;.\qquad(\mathrm{TD^{2}-CFM};\,9)$$

Since we now bootstrap the previous estimate not only in the sampling process but also in the objective function, we refer to this method as TD2-Conditional Flow Matching (TD2-
CFM). The right panel of Figure 1 depicts the process of obtaining the bootstrapped vector field v˜
(n)
tfor TD2-CFM. We provide further implementation details and pseudo-code for all TD-Flow methods in Appendix C.3.1. Next, we extend our TD2result to the class of denoising diffusion models.

## 3.1. Extension To Diffusion Models

Denoising Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020) build a diffusion process starting from a
data sample X0 ∼ q0 = mπ(· | *S, A*)
2and corrupting it via
a stochastic differential equation (SDE),
$$\mathrm{d}X_{t}=f(t)\,X_{t}\,\mathrm{d}t+g(t)\,\mathrm{d}W_{t}\,,\tag{10}$$
where t ∈ [0, T] for some time horizon T, f, g : [0, T] → R is drift and diffusion term, and Wt ∈ R
dis a standard Brownian motion. The forward process of the linear SDE (10) has
an analytic Gaussian kernel qt|0(·|X0) = N (·|αtX0, σ2
t
I),
where αt and σt can be computed in closed form. To sample from the target data distribution q0, we can solve the reverse SDE (Song & Ermon, 2019) from time T to 0:
$$\mathrm{d}X_{t}=\left(f(t)\,X_{t}-g(t)\,\nabla_{X_{t}}\log q_{t}(X_{t}\,|\,S,A)\right)\mathrm{d}t+g(t)\,\mathrm{d}\overline{W}_{t},\tag{11}$$
2Different to flow matching, time is inverted in diffusion models and ranges from 0 to T.
where Wt is the reverse-time Brownian motion and qt is the marginal distribution of both the forward (16) and reverse (17) process. To simulate (11), we can train a parametrized score function s˜t(x | *s, a*; θ) to approximate ∇xtlog qt(xt | *s, a*) using the denoising diffusion / score matching objective (Vincent, 2011) ℓDD(θ):

$\mathbb{E}_{\rho,t,X_{0},X_{t}}\left[\left|\left|\tilde{s}_{t}(X_{t}\mid S,A;\theta)-\nabla_{X_{t}}\log q_{t|0}(X_{t}\mid X_{0})\right|\right|^{2}\right]$, where $X_{0}\sim m^{\pi}(\cdot\mid S,A)$, $X_{t}\sim q_{t|0}(\cdot\mid X_{0})$. (DD; 12)
Temporal Difference Diffusion Following the blueprint in §3, we define an iterative process starting from s˜
(0) =
s˜(*· · ·* ; θ
(0)) and minimize at each iteration the Temporal-
Difference Denoising Diffusion (TD-DD) loss ℓTD-DD(θ):

$$\mathbb{E}_{\rho,t,X_{0},X_{t}}\left[\left\|\tilde{s}(X_{t}\mid S,A;\theta)-\nabla_{x}\log q_{t|0}(X_{t}\mid X_{0})\right\|^{2}\right],$$  where $X_{0}\sim\left(\mathcal{T}^{\pi}\widetilde{m}_{0|T}^{(n)}\right)(\cdot\mid S,A),X_{t}\sim q_{t|0}(\cdot\mid X_{0})\,.$  (TD-DD; 13)  Once again, to sample $X_{0}\sim\left(\mathcal{T}^{\pi}\widetilde{m}_{0|T}^{(n)}\right)(\cdot\mid S,A)$, we prove the following with probability $1$:
ceed as follows: with probability 1−γ, we draw a successor state S
′ ∼ P(· | *S, A*); conversely, with probability γ, we sample from the bootstrapped model by solving the reverse SDE with score function s˜
(n), initiated from XT . Following an approach analogous to Lemma 1, we demonstrate in Appendix B that we can employ two distinct diffusion processes for the two terms involved in the Bellman operator, which consequently leads to the TD2-DD objective:

$$\vec{\ell}(\theta)=\mathbb{E}_{\rho,t,\vec{X}_{t}}\Big[\big|\vec{s}_{t}(\vec{X}_{t}\,|\,S,A;\theta)-\nabla_{\vec{X}_{t}}q_{t|0}(\vec{X}_{t}\,|\,S^{\prime})\big|^{2}\Big],$$ $$\text{where}\vec{X}_{t}\sim q_{t|0}(\cdot\mid S^{\prime})\,,$$
$$\begin{array}{c}{{\widehat{\ell}(\theta)=\mathbb{E}_{p,t,\widehat{X}_{t}}\left[\left|\left|\tilde{s}_{t}(\widehat{X}_{t}\,|\,S,A;\theta)-\tilde{s}_{t}^{(n)}(\widehat{X}_{t}\,|\,S^{\prime},\pi(S^{\prime})\right|\right|^{2}\right],}}\\ {{\mathrm{~where~}X_{T}\sim q_{T},\;\widehat{X}_{t}\sim q_{t|T}^{(n)}(\cdot\mid S^{\prime},\pi(S^{\prime}))\,,}}\end{array}$$

$$\ell_{\mathrm{TD^{2}-DD}}(\theta)=(1-\gamma)\vec{\ell}(\theta)+\gamma\widetilde{\ell}(\theta)\,.\qquad(\mathrm{TD^{2}-DD};\,14)$$

## 4. Theoretical Analysis

We now study the learning dynamics of an idealized version of the TD-Flow methods, assuming that the flow-matching loss is minimized exactly at each iteration. Under this assumption, at each iteration we compute a probability path m
(n)
tsuch that m
(n)
1 = T
πm
(n−1)
1, which implies that m
(n)
1 → mπ by the contraction property of T
π. The following result shows that the overall probability paths m
(n) t follow a similar process. Proofs are deferred to Appendix E.

Theorem 1. For any n ≥ 1, the probability paths generated by TD-CFM, TD-CFM(C)*, or* TD2-CFM *satisfy*

$$m_{t}^{(n+1)}(x\mid s,a)=\left({\mathcal{B}}_{t}^{\pi}m_{t}^{(n)}\right)(x\mid s,a),\;\;\forall\,t\in[0,1]$$

where B
π t m := (1 − γ)Pt + γP πm and Pt(x|*s, a*) :=
Rpt|1(x | x1)P(x1|s, a)dx1. For any t ∈ [0, 1], the operator B
π tis a γ-contraction in 1-Wasserstein distance, that is, for any couple of probability paths pt, qt,

$$\begin{array}{c}{{\operatorname*{sup}_{s,a}W_{1}\left((\mathcal{B}_{t}^{\pi}p_{t})\,(\cdot\mid s,a),(\mathcal{B}_{t}^{\pi}q_{t})\,(\cdot\mid s,a)\right)}}\\ {{\qquad\qquad\leq\gamma\operatorname*{sup}_{s,a}W_{1}\left(p_{t}(\cdot\mid s,a),q_{t}(\cdot\mid s,a)\right).}}\end{array}$$

Theorem 1 shows that all TD-flow methods fundamentally implement the same update where the probability path at t ∈ [0, 1] is obtained by applying a Bellman-like operator Bt to the previous iteration. This operator is a γ-contraction as T
π, directly implying the following result.

Corollary 1. Let {m
(n)
t }n≥0 be the sequence of probability paths produced by TD-CFM, TD-CFM(C)*, or* TD2-CFM
starting from an arbitrary vector field v
(0)
t*. Then,*

$$\operatorname*{lim}_{n\to\infty}m_{t}^{(n)}=\overline{{{m}}}_{t}=\mathcal{B}_{t}\overline{{{m}}}_{t},$$

where mt is the unique fixed point of Bt*, and* mt = mMC
t, where mMC
t(· | *s, a*) = Rpt|1(· | x1) mπ(x1 | s, a) *is the* probability path of the Monte-Carlo approach (MC-CFM; 6).

This corollary shows that the fixed point of Bt coincides with the probability path generated in Monte-Carlo Conditional Flow Matching (MC-CFM; 6), which assumes direct access to samples of mπ. An important subtlety in Theorem 1 is that all algorithms apply the same operator for n ≥ 1, but the result holds for n = 0 only for TD2-CFM. This means that even starting from the same θ
(0), the three algorithms may generate different sequences {m
(n)
t }n≥0, while still converging to mt. In Theorems 5 and 6 , we show we can reconcile TD-CFM(C) and TD-CFM with TD2-CFM under a mild assumption on the form of the initial vector field.

While Theorem 1 analyzes an idealized version of the algorithms, in practice gradients are estimated from samples and the following analysis reveals important differences in their variance. We introduce the (unbiased) sample-based gradients for each of the algorithms,

$$\begin{array}{c}{{\mathbb{E}\left[g_{\mathrm{TD-CFM}}(Y_{\mathrm{TD-CFM}})\right]=\nabla_{\theta}\,\ell_{\mathrm{TD-CFM}}(\theta),}}\\ {{\mathbb{E}\left[g_{\mathrm{TD-CFM(C)}}(Y_{\mathrm{TD-CFM(C)}})\right]=\nabla_{\theta}\,\ell_{\mathrm{TD-CFM(C)}}(\theta),}}\\ {{\mathbb{E}\left[g_{\mathrm{TD^{2}-CFM}}(Y_{\mathrm{TD^{2}-CFM}})\right]=\nabla_{\theta}\,\ell_{\mathrm{TD^{2}-CFM}}(\theta),}}\end{array}$$

where Y summarizes the random variables involved in the loss definitions in (TD-CFM; 7), (TD-CFM(C); 8),
and (TD2-CFM; 9) (see Appendix E.6 for a formal definition of the gradients). We want to compare the total variance of the gradient estimates σ 2 = TrCovY [ g(Y ) ] , where Tr denotes the trace.

Theorem 2. For any n ≥ 1 and t ∈ [0, 1]*, assume that* m
(n)
t(x | *s, a*) = Rpt|1(x | x1)m
(n)
1(x1 | s, a)dx1*, then*

σ 2 TD-CFM = σ 2 TD2-CFM + γ 2 E -TrCovX1|s,a,Xt -∇θvt(Xt|s, a; θ) ⊤ut|1(Xt|X1).

Theorem 3. For any n ≥ 1 and t ∈ [0, 1]*, assume* that m
(n)
t(x | *s, a*) = Rpt|0,1(x | x0, x1)m
(n)
0,1(x0, x1 | s, a)dx0dx1 3*, then we obtain*

$$\begin{array}{l}{{\sigma_{\mathrm{{T\!D-CFM}(C)}}^{2}=\sigma_{\mathrm{{T\!D}}^{2}\leftarrow\mathrm{{FM}}}^{2}\ +}}\\ {{\ \gamma^{2}\mathbb{E}\left[\mathrm{Tr}\big(\mathrm{Cov}_{Z}|S,A,X_{t}\left[\nabla_{\theta}v_{t}(X_{t}|S,A;\theta)^{\top}u_{t|Z}(X_{t}|Z)\right]\big)\right],}}\end{array}$$

where Z = (X0, X1). Furthermore, if we use straight conditional paths, i.e., Xt = tX1 + (1 − t)X0, and the linear interpolant Xt does not intersect for any s, a, s′*, then* σ 2 TD-CFM(C) 
= σ 2 TD2-CFM
.

In both results, the probability path m
(n)
tfrom the previous iteration must be identical for the algorithms being compared. The analysis reveals that TD-CFM and TD-CFM(C) suffer from a larger variance compared to TD2-CFM, which uses the vector field v
(n) both to sample Xt and as a target for the regression problem. This variance gap is "discounted" by γ 2, which suggests that the performance of these algorithms would be similar for problems with small horizons but would increase as γ → 1. The extra variance in both cases stems from samples generated by the algorithm (i.e., they do not depend on the transitions available in the dataset). In this sense, we can refer to it as computational variance, and in principle, it could be reduced by increasing the number of samples X0, X1, and Xt used in gradient computation. While the variance of TD-CFM and TD-CFM(C) cannot be directly compared, we expect that constructing Xt from X0 and X1 (instead of X1 only) will tend to reduce its variance. Specifically, when Xt is obtained by linear interpolation between X0 and X1, and it does not generate crossing paths, the variance of TD-CFM(C)
reduces to the one of TD2-CFM.

## 5. Experiments

We now present a series of experiments to assess the efficacy of our TD-based flow and diffusion approaches with baselines employing Generative Adversarial Networks (Goodfellow et al., 2014) and β-Variational Auto-Encoders (Higgins et al., 2017). Following the methodology from Touati et al. (2023); Pirotta et al. (2024), we benchmark 22 tasks spanning 4 domains (Maze, Walker, Cheetah, Quadruped) from the DeepMind Control Suite (Tunyasuvunakool et al., 2020).

For a single policy, we evaluate how well each method models its i) successor measure and ii) value function. While lower errors in estimating the successor measure are expected to lead to better value estimation, this is not always the case since modeling errors may disproportionally affect states with negligible rewards. Additionally, motivated by our theoretical results, we explore how the probability path's design affects our proposed methods' relative performance. Finally, we examine the scalability of our approach by learning a generative model of the successor measure across a class of parameterized policies derived from the Forward- Backward (FB) representation (Touati & Ollivier, 2021; Touati et al., 2023), a non-generative model of the successor measure. We conclude by demonstrating how TD2enables more effective planning for task-relevant policies when performing Generalized Policy Improvement (GPI; Barreto et al., 2017), far surpassing the capabilities of FB alone.

## 5.1. **Empirical Evaluation Of Geometric Horizon Models**

Before benchmarking, we must first obtain a policy to evaluate. We follow the approach taken in Thakoor et al. (2022) and pre-train a set of deterministic policies - one for each task - using TD3 (Fujimoto et al., 2018). The final policy obtained from this pre-training phase is now fixed for the remainder of our experiments. GHM training proceeds in an off-policy manner where we learn the successor measure of a TD3 policy using transition data from the ExoRL dataset (Yarats et al., 2022); specifically, we use a dataset of 10M transitions collected by a random network distillation policy (Burda et al., 2019). All GHM methods are trained for 3M gradient steps using the AdamW optimizer (Loshchilov & Hutter, 2019) with a batch size of 1024 and weight decay of 0.001. We maintain a target network using an exponential moving average of the training parameters with a step size of 0.001. Special care was taken to match the capacity of the neural networks between methods with a UNet-style architecture employed for all flow and diffusion methods, while the GAN and VAE baselines use an MLP with residual connections for all their respective networks. Full details for the training methodology, network architecture, and hyperparameters can be found in Appendix C. We implement all conditional flow matching methods (TD-
CFM, TD-CFM(C), TD2-CFM) with the Optimal Transport Gaussian conditional path from Lipman et al. (2023). When constructing our bootstrap targets, we sample from the neural ODE using the Midpoint solver with a constant step size of t/10 for a maximum of 10 steps. For TD2-CFM, we sample t ∼ U([0, 1]); otherwise, we integrate to t = 1 and construct Xt using the conditional path. For Denoising Diffusion methods (TD-DD, TD2-DD), we train a DDPM (Ho et al., 2020) by discretizing β ∈ (0.1, 20) using T = 1, 000 steps. We construct diffusion bootstrapped targets using

TD-GAN TD-VAE
Scaling Effective Horizon Effective Horizon 10 4 10 2 10 0 10 2 10 4 TD-DD
Va lu e F
u n ct io n M
S
E

TD-CFM TD²-CFM TD-CFM(C) TD²-DD
5 10 20 50 100
20 steps of the DDIM (Song et al., 2021a) sampler. For TD-DD, we solve to t = 0 and regress towards the noise that re-corrupted our sample. Alternatively, TD2-DD directly regresses towards the noise prediction from the target network at a randomly selected noise level. The first baseline we consider is a GHM instantiated as a Generative Adversarial Network (Goodfellow et al., 2014) similar to the one found in Janner et al. (2020). We follow the best practices from Huang et al. (2024) with the primary modification being a relativistic discriminator (Jolicoeur-Martineau, 2019) equipped with a zero-centered gradient penalty on both real and fake samples. For our second baseline, we implement a β-VAE (Higgins et al., 2017) following the practices outlined in Thakoor et al. (2022). To evaluate the quality of our models, we first generate samples from the ground truth successor measure mπaccording to the following procedure. We first randomly sample 64 source states S0 from the initial state distribution and execute policy π for 1, 000 steps. Along each trajectory, we resample 2048 states with replacement according to the stopping time t ∼ Geometric(1 − γ). For the same 64 source states, we generate a matching set of 2048 samples from each GHM. Now in possession of these two sets of samples, we evaluate the: 1) log-likelihood of the true samples for models with tractable densities (i.e., diffusion and flow methods); 2) Earth Mover's Distance (EMD; Rubner et al., 2000), which quantifies the minimal transport cost between the two empirical distributions; and 3) mean-squared error of a Monte-Carlo estimate of the true value function Qπ and the value function derived from GHM samples using (2). Full details can be found in Appendix C.1. Having established our training framework, baselines, and evaluation protocol, we proceed to investigate a key prediction from our theoretical analysis. Our variance analysis

| Method    | EMD ↓               | Norm NLL ↓     | MSE(V) ↓          |
|-----------|---------------------|----------------|-------------------|
| TD-DD     | 20.22 (0.26)        | 2.824 (0.195)  | 454.49 (131.97)   |
| TD2 -DD   | 14.14 (1.08)        | 0.806 (0.016)  | 189.15 (23.63)    |
| TD-CFM    | 12.26 (0.02)        | 0.886 (0.024)  | 228.77 (2.20)     |
| TD-CFM(C) | 10.51 (0.06)        | 0.447 (0.020)  | 140.78 (18.72)    |
| TD2 -CFM  | 10.57 (0.07)        | 0.422 (0.014)  | 135.22 (19.79)    |
| GAN       | 23.97 (0.46)        | -              | 2463.22 (628.05)  |
| VAE       | 83.77 (0.41)        | -              | 1284.27 (37.62)   |
| TD-DD     | 0.149 (0.001)       | 2.974 (0.100)  | 1245.20 (29.27)   |
| TD2 -DD   | 0.027 (0.001)       | 0.761 (0.082)  | 11.13 (3.09)      |
| TD-CFM    | 0.062 (0.003)       | 0.554 (0.033)  | 355.56 (82.83)    |
| TD-CFM(C) | 0.022 (0.002)       | −0.696 (0.094) | 11.89 (3.16)      |
| TD2 -CFM  | 0.021 (0.000)       | −0.843 (0.027) | 8.74 (2.09)       |
| GAN       | 0.203 (0.037)       | -              | 1257.26 (112.86)  |
| VAE       | 0.410 (0.036)       | -              | 1821.89 (69.78)   |
| TD-DD     | 28.33 (0.33)        | 1.908 (0.041)  | 1490.75 (444.49)  |
| TD2 -DD   | 22.64 (2.47)        | 0.861 (0.028)  | 159.03 (14.64)    |
| TD-CFM    | 15.73 (0.06)        | 1.056 (0.002)  | 525.06 (28.90)    |
| TD-CFM(C) | 14.38 (0.03)        | 0.488 (0.003)  | 155.25 (5.58)     |
| TD2 -CFM  | 14.51 (0.05)        | 0.379 (0.011)  | 141.77 (3.10)     |
| GAN       | 36772.12 (13898.25) | -              | 2634.69 (798.38)  |
| VAE       | 60.27 (0.28)        | -              | 1156.33 (36.52)   |
| TD-DD     | 20.58 (0.24)        | 2.649 (0.137)  | 382.40 (458.63)   |
| TD2 -DD   | 12.09 (0.12)        | 0.537 (0.060)  | 39.04 (6.08)      |
| TD-CFM    | 13.53 (0.11)        | 0.713 (0.028)  | 225.27 (42.43)    |
| TD-CFM(C) | 11.91 (0.02)        | 0.219 (0.016)  | 30.71 (3.44)      |
| TD2 -CFM  | 11.92 (0.10)        | 0.104 (0.001)  | 28.35 (6.10)      |
| GAN       | 24.51 (0.89)        | -              | 3690.65 (1117.94) |
| VAE       | 111.73 (2.53)       | -              | 2457.61 (16.25)   |

| Method    | EMD ↓         | Norm NLL ↓   | MSE(V) ↓        |
|-----------|---------------|--------------|-----------------|
| TD-CFM(C) | 14.08 (12.42) | 1.79 (1.98)  | 310.45 (258.94) |
| TD2 -CFM  | 0.09 (0.09)   | −0.01 (0.04) | −3.36 (7.76)    |

suggests that our TD-Flow framework should enable more stable training across extended temporal horizons. To validate this hypothesis, we train each GHM for 3 seeds on the loop task in the Maze domain while varying the effective horizon (1 − γ)
−1across five values: {5, 10, 20, 50, 100}.

Figure 2 illustrates the relationship between value function MSE and the effective horizon. The results demonstrate that TD2-based methods maintain consistent performance even as the effective horizon increases, while alternative approaches show significant performance degradation. Notably, at an effective horizon of 100, TD2-based methods maintain their accuracy and achieve performance improvements of nearly four orders of magnitude compared to their naive implementations. These results empirically support for our initial hypothesis, with the stability of TD2 methods aligning with our predictions.

In the following, we shift our attention to a more in-depth analysis of the largest horizon of 100 (γ = 0.99). For each

Planning via Generalized Policy Improvement FB-GPI DD-GPI FM-GPI Coupled TD²
+8
+36 +36
+8
+36 +36
+9
+36 +36
+9
+36 +36
-38
+16
+33
-38
+16
+33
-59
+12
+32
-59
+12
+32
%
 I
m pr o ve m e nt O
ve r F
B

+5
+30 +29
+5
+30 +29 Random Train Distribution Local Perturbation
−10 0 10 20 30
-57
+10
+25
-57
+10
+25
algorithm, we train a GHM for 3 independent seeds for all domains and tasks. Table 1 reports aggregate performance across our full suite of metrics. For each domain and metric, we highlight results in a 1% range with respect to the bestperforming method. The results demonstrate a clear pattern of superior performance for TD2-based algorithms: TD2-
CFM achieves significant improvements over TD-CFM with a 10× reduction in value-function MSE, 1.5× reduction in EMD, and 3× reduction in log-likelihood, averaged across all four domains. In line with our theoretical predictions, the coupled variant of TD-CFM performs comparably to TD2-
CFM, given straight conditional paths. While a comparison between flow matching and diffusion is not at the core of this paper, in our experiments, flow matching generally outperforms diffusion across all metrics. We posit this is primarily due to noise in the diffusion process adversely impacting an already noisy prediction problem for large horizons. Given the comparable performance between TD-CFM(C)
and TD2-CFM with straight conditional paths, we next examine how these methods behave with alternative path geometries. Our theoretical analysis suggests an important distinction: TD2-CFM should maintain its effectiveness with non-straight paths, while the performance of TD-CFM(C) should degrade. To test this prediction, we maintain the methodology above while replacing conditional path in
(TD2-CFM; 9) with the following curved path pt|1(· | X1) =
N (· | αtX1, β2 t) with coefficients αt = sin π2 tand βt =
cos π2 t. The corresponding conditional vector field is now given by ut|1(Xt|X1) = α˙ t −
αt βt X1 +
β˙t βt Xt. Additionally, for TD-CFM(C) we condition the curved path above on X0 and X1 resulting in the conditional vector field ut|0,1(Xt | X0, X1) = π2 βtX1 − αtX0. Table 2 illustrates the performance difference relative to the straight path results (Table 1) averaged across all domains and tasks. The results strongly support our theoretical prediction: TD2-CFM
not only maintained but surprisingly improved performance compared to the linear path. In contrast, TD-CFM(C) showed significant performance degradation, confirming our hypothesis about its limitations with non-straight paths.

## 5.2. Planning Via Generalized Policy Improvement

We now turn our attention towards training policyconditioned GHMs which can be utilized for test-time planning. To accomplish this, we first pre-train a Forward Backward (FB; Touati & Ollivier, 2021; Touati et al., 2023) representation using the same dataset of 10M transitions as described in §5.1. This pre-training yields a class of wconditioned policies πw, where each w ∈ W = S
d−1(
√d)
represents an embedding of a reward function situated on a d-dimensional hypersphere with radius 
√d. We then train the GHM mπw conditioned on the policy by incorporating the embedding w directly into the model's input. All GHM methods are trained for 8M gradient steps, maintaining the same parameters used in §5.1, with the exception of a higher weight decay coefficient of 0.01. For additional insights into the accuracy of the policy-conditioned GHMs, we direct the reader to Appendix D. Overall, we observed similar trends to those seen in our single-policy experiments.

Given that both FB and w-conditioned GHM models enable estimation of a policy's value function Qπw , we can utilize this information to perform Generalized Policy Improvement (GPI; Barreto et al., 2017) during evaluation. Specifically, at each time step t, we choose an action at = πwt
(st),
where wt is derived as follows:

$$w_{t}\in\operatorname*{arg\,max}_{w\sim D(\mathsf{W})}\ \underbrace{\left(1-\gamma\right)^{-1}\mathbb{E}_{X\sim m^{\pi_{w}}\left(\cdot|s_{t},\pi_{w}(s_{t}))\right)}\left[\,r(X)\,\right]}_{Q^{\pi_{w}}\left(s_{t},\pi_{w}(s_{t})\right)}\tag{15}$$

(15)
Here D(W) is a sampling distribution over W. We consider three such distributions: *i) Random*: uniform distribution over W; *ii) Local Perturbation*: we perturb the embedding wr of the task reward r by the uniform distribution; *iii)*
Train Distribution: we sample w from the training distribution used by FB. To approximate (15), we sample 255 embeddings from D(W) and explicitly include the task embedding wr, resulting in a maximization over 256 policies. To estimate Qπw , we average the reward over 128 states sampled from mπw . Performance is measured by averaging returns over 100 episodes, each lasting 1000 steps. Figure 3 illustrates the average percentage of improvement for each algorithm and w-sampling strategy relative to the performance of the FB policy πwrfor the task reward r.

We refer to Appendix D for a more detailed view of these results. All TD-based GHM approaches lead to a significant improvement over the base FB policy, with TD-CFM(C)
and TD2-CFM providing ≈ 30%+ improvement with all sampling approaches. TD2-DD also leads to significant performance gains but is still dominated by the flow matching methods. Notably, FB-based GPI not only fails to improve performance but actually deteriorates it on average with significant degradation observed in three out of four domains (detailed results available in Appendix D). When comparing different distributions D(W), we observe that while FB-GPI's performance fluctuates considerably, GHM methods maintain their robustness across distributions, showing only minor variation. These results underscore the ability of our improved GHMs to make long-term predictions enabling powerful planning capabilities.

## 6. Discussion

In this paper, we introduced temporal difference flows, a novel generative modeling approach that significantly advances long-horizon predictive models of state. By leveraging the successor measure's temporal difference structure both in its sampling procedure and learning objective, TD2-
CFM and TD2-DD effectively address challenges associated with modeling long-range state dynamics. The methods developed in this paper provide a robust theoretical and empirical foundation that demonstrates the advantages of our framework across a range of tasks, metrics, and domains. We envision numerous exciting applications emerging from this work, particularly around imitation learning (Wu et al.,
2025; Jain et al., 2025), planning (Sutton, 1991; Thakoor et al., 2022; Zhu et al., 2024), and off-policy evaluation (Precup et al., 2000; 2001; Nachum et al., 2019; Fujimoto et al.,
2021). Furthermore, recent work on consistency models (Song et al., 2023; Yang et al., 2024) and self-distillation (Frans et al., 2025) suggests promising avenues for tackling the computational burden of sampling - a limitation common to the family of iterative generative models that our approach builds upon.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations, (ICLR), 2023.

Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E.

Stochastic interpolants: A unifying framework for flows and diffusions. *CoRR*, abs/2303.08797, 2023.

Anderson, B. D. Reverse-time diffusion equation models.

Stochastic Processes and their Applications, 12(3):313– 326, 1982.

Ba, J., Kiros, J., and Hinton, G. E. Layer normalization.

CoRR, abs/1607.06450, 2016.

Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T.,
Silver, D., and van Hasselt, H. Successor features for transfer in reinforcement learning. In Neural Information Processing Systems (NeurIPS), 2017.

Barreto, A., Hou, S., Borsa, D., Silver, D., and Precup, D.

Fast reinforcement learning with generalized policy updates. Proceedings of the National Academy of Sciences (PNAS), 117(48):30079–30087, 2020.

Blier, L., Tallec, C., and Ollivier, Y. Learning successor states and goal-dependent values: A mathematical viewpoint. *CoRR*, abs/2101.07123, 2021.

Borsa, D., Barreto, A., Quan, J., Mankowitz, D. J., van Hasselt, H., Munos, R., Silver, D., and Schaul, T. Universal successor features approximators. In *International* Conference on Learning Representations (ICLR), 2019.

Burda, Y., Edwards, H., Storkey, A. J., and Klimov, O. Exploration by random network distillation. In International Conference on Learning Representations (ICLR), 2019.

Cetin, E., Touati, A., and Ollivier, Y. Finer behavioral foundation models via auto-regressive features and advantage weighting. *CoRR*, abs/2412.04368, 2024.

Chen, T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. Neural ordinary differential equations. In *Neural* Information Processing Systems (NeurIPS), 2018.

Dayan, P. Improving generalization for temporal difference learning: The successor representation. Neural Computation, 1993.

De Bortoli, V., Korshunova, I., Mnih, A., and Doucet, A.

Schrodinger bridge flow for unpaired data translation. In ¨ Neural Information Processing Systems (NeurIPS), 2024.

Dinh, L., Krueger, D., and Bengio, Y. NICE: Non-linear independent components estimation. In *International* Conference on Learning Representations (ICLR), Workshop Track Proceedings, 2015.

Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real nvp. In International Conference on Learning Representations (ICLR), 2017.

Farebrother, J., Greaves, J., Agarwal, R., Le Lan, C.,
Goroshin, R., Castro, P. S., and Bellemare, M. G. Protovalue networks: Scaling representation learning with auxiliary tasks. In International Conference on Learning Representations (ICLR), 2023.

Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., Gautheron, L., Gayraud, N. T., Janati, H., Rakotomamonjy, A., Redko, I., Rolet, A., Schutz, A., Seguy, V., Sutherland, D. J., Tavenard, R., Tong, A., and Vayer, T. Pot: Python optimal transport. Journal of Machine Learning Research, 22(78):1–8, 2021.

Frans, K., Hafner, D., Levine, S., and Abbeel, P. One step diffusion via shortcut models. In International Conference on Learning Representations (ICLR), 2025.

Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning (ICML), 2018.

Fujimoto, S., Meger, D., and Precup, D. A deep reinforcement learning approach to marginalized importance sampling with the successor representation. In International Conference on Machine Learning (ICML), 2021.

Ghosh, D., Bhateja, C. A., and Levine, S. Reinforcement learning from passive data via latent intentions. In International Conference on Machine Learning (ICML),
2023.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B.,
Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In Neural Information Processing Systems (NeurIPS), 2014.

Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., and Duvenaud, D. FFJORD: free-form continuous dynamics for scalable reversible generative models. In International Conference on Learning Representations (ICLR), 2019.

Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. P. Mastering diverse domains through world models. *CoRR*, abs/2301.04104, 2023.

Hansen, N., Su, H., and Wang, X. Temporal difference learning for model predictive control. In *International* Conference on Machine Learning (ICML), 2022.

Hansen, N., Su, H., and Wang, X. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations (ICLR), 2024.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X.,
Botvinick, M., Mohamed, S., and Lerchner, A. betavae: Learning basic visual concepts with a constrained variational framework. In *International Conference on* Learning Representations (ICLR), 2017.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Neural Information Processing Systems (NeurIPS), 2020.

Huang, N., Gokaslan, A., Kuleshov, V., and Tompkin, J. The gan is dead; long live the gan! a modern gan baseline. In Neural Information Processing Systems (NeurIPS), 2024.

Jafferjee, T., Imani, E., Talvitie, E., White, M., and Bowling, M. Hallucinating value: A pitfall of dyna-style planning with imperfect environment models. *CoRR*,
abs/2006.04363, 2020.

Jain, A. K., Lehnert, L., Rish, I., and Berseth, G. Maximum state entropy exploration using predecessor and successor representations. In Neural Information Processing Systems (NeurIPS), 2023.

Jain, A. K., Wiltzer, H., Farebrother, J., Rish, I., Berseth, G.,
and Choudhury, S. Non-adversarial inverse reinforcement learning via successor feature matching. In International Conference on Learning Representations (ICLR), 2025.

Janner, M., Mordatch, I., and Levine, S. Gamma-models:
Generative temporal difference learning for infinitehorizon prediction. In Neural Information Processing Systems (NeurIPS), 2020.

Jolicoeur-Martineau, A. The relativistic discriminator: a key element missing from standard gan. In *International* Conference on Learning Representations (ICLR), 2019.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

successor representation. Journal of Machine Learning Research (JMLR), 24:80:1–80:69, 2023.

Misra, D. Mish: A self regularized non-monotonic neural activation function. *CoRR*, abs/1908.08681, 2019.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. In International Conference on Learning Representations (ICLR), 2014.

Nachum, O., Chow, Y., Dai, B., and Li, L. Dualdice:
Behavior-agnostic estimation of discounted stationary distribution corrections. In Neural Information Processing Systems (NeurIPS), 2019.

Lambert, N., Pister, K., and Calandra, R. Investigating compounding prediction errors in learned dynamics models. CoRR, abs/2203.09637, 2022.

Park, S., Kreiman, T., and Levine, S. Foundation policies with hilbert representations. In *International Conference* on Machine Learning (ICML), 2024.

Le Lan, C., Tu, S., Oberman, A., Agarwal, R., and Bellemare, M. G. On the generalization of representations in reinforcement learning. In *International Conference on* Artificial Intelligence and Statistics (AISTATS), 2022.

Pathak, D., Agrawal, P., Efros, A. A., and Darrell, T.

Curiosity-driven exploration by self-supervised prediction. In International Conference on Machine Learning (ICML), 2017.

Le Lan, C., Greaves, J., Farebrother, J., Rowland, M., Pedregosa, F., Agarwal, R., and Bellemare, M. G. A novel stochastic gradient descent algorithm for learning principal subspaces. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2023a.

Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. C. Film: Visual reasoning with a general conditioning layer. In AAAI Conference on Artificial Intelligence, 2018.

Le Lan, C., Tu, S., Rowland, M., Harutyunyan, A., Agarwal, R., Bellemare, M. G., and Dabney, W. Bootstrapped representations in reinforcement learning. In International Conference on Machine Learning (ICML), 2023b.

Pirotta, M., Tirinzoni, A., Touati, A., Lazaric, A., and Ollivier, Y. Fast imitation via behavior foundation models.

In International Conference on Learning Representations (ICLR), 2024.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M.,
and Le, M. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C.,
Amos, B., Lipman, Y., and Chen, R. T. Q. Multisample flow matching: Straightening flows with minibatch couplings. In International Conference on Machine Learning (ICML), 2023.

Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M.,
Karrer, B., Chen, R. T. Q., Lopez-Paz, D., Ben-Hamu, H., and Gat, I. Flow matching guide and code. *CoRR*,
abs/2412.06264, 2024.

Precup, D., Sutton, R. S., and Singh, S. Eligibility traces for off-policy policy evaluation. In International Conference on Machine Learning (ICML), 2000.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast:
Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR), 2023.

Precup, D., Sutton, R. S., and Dasgupta, S. Off-policy temporal difference learning with function approximation. In International Conference on Machine Learning (ICML), 2001.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

Rezende, D. and Mohamed, S. Variational inference with normalizing flows. In International Conference on Machine Learning (ICML), 2015.

Machado, M. C., Rosenbaum, C., Guo, X., Liu, M., Tesauro, G., and Campbell, M. Eigenoption discovery through the deep successor representation. In International Conference on Learning Representations (ICLR), 2018.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), volume 9351, pp. 234–241, 2015.

Machado, M. C., Bellemare, M. G., and Bowling, M. Countbased exploration with the successor representation. In AAAI Conference on Artificial Intelligence, 2020.

Rubner, Y., Tomasi, C., and Guibas, L. J. The earth mover's distance as a metric for image retrieval. *International* Journal of Computer Vision, 40(2):99–121, 2000.

Machado, M. C., Barreto, A., Precup, D., and Bowling, M.

Temporal abstraction in reinforcement learning with the Schmidhuber, J. A possibility for implementing curiosity and boredom in model-building neural controllers. In International Conference on Simulation of Adaptive Behavior, 1991.

Schramm, L. and Boularias, A. Bellman diffusion models.

CoRR, abs/2407.12163, 2024.

Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., and Silver, D. Mastering atari, go, chess and shogi by planning with a learned model. *Nature*, 588(7839):604–609, 2020.

Shi, Y., De Bortoli, V., Campbell, A., and Doucet, A. Diffusion schrodinger bridge matching. In ¨ Neural Information Processing Systems (NeurIPS), 2023.

Sikchi, H., Zhou, W., and Held, D. Learning off-policy with online planning. In *Conference on Robot Learning* (CoRL), 2021.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L.,
van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T. P., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. Mastering the game of go with deep neural networks and tree search. *Nature*, 529(7587):484–489, 2016.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., and Hassabis, D. Mastering the game of go without human knowledge.

Nature, 550(7676):354–359, 2017.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In *International Conference on* Machine Learning (ICML), 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In *International Conference on Learning* Representations (ICLR), 2021a.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. In *Neural Information* Processing Systems (NeurIPS), 2019.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021b.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In International Conference on Machine Learning (ICML), 2023.

Stadie, B. C., Levine, S., and Abbeel, P. Incentivizing exploration in reinforcement learning with deep predictive models. In International Conference on Learning Representations (ICLR), 2016.

Sutton, R. S. Dyna, an integrated architecture for learning, planning, and reacting. *ACM SIGART*, 2(4):160–163, 1991.

Talvitie, E. Model regularization for stable sample rollouts.

In Conference on Uncertainty in Artificial Intelligence
(UAI), 2014.

Thakoor, S., Rowland, M., Borsa, D., Dabney, W., Munos, R., and Barreto, A. Generalised policy improvement with geometric policy composition. In International Conference on Machine Learning (ICML), 2022.

Tirinzoni, A., Touati, A., Farebrother, J., Guzek, M., Kanervisto, A., Xu, Y., Lazaric, A., and Pirotta, M. Zero-shot whole-body humanoid control via behavioral foundation models. In International Conference on Learning Representations (ICLR), 2025.

Tomar, M., Hansen-Estruch, P., Bachman, P., Lamb, A.,
Langford, J., Taylor, M. E., and Levine, S. Video occupancy models. *CoRR*, abs/2407.09533, 2024.

Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y.,
Rector-Brooks, J., Wolf, G., and Bengio, Y. Improving and generalizing flow-based generative models with minibatch optimal transport. In Transactions on Machine Learning Research (TMLR), 2024.

Touati, A. and Ollivier, Y. Learning one representation to optimize all rewards. In *Neural Information Processing* Systems (NeurIPS), 2021.

Touati, A., Rapin, J., and Ollivier, Y. Does zero-shot reinforcement learning exist? In *International Conference on* Learning Representations (ICLR), 2023.

Tunyasuvunakool, S., Muldal, A., Doron, Y., Liu, S., Bohez, S., Merel, J., Erez, T., Lillicrap, T., Heess, N., and Tassa, Y. dm control: Software and tasks for continuous control. Software Impacts, 6:100022, 2020.

van den Oord, A., Vinyals, O., and Kavukcuoglu, K. Neural discrete representation learning. In Neural Information Processing Systems (NeurIPS), 2017.

Vincent, P. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661– 1674, 2011.

Wiltzer, H., Farebrother, J., Gretton, A., and Rowland, M.

Foundations of multivariate distributional reinforcement learning. In Neural Information Processing Systems
(NeurIPS), 2024a.

Wiltzer, H., Farebrother, J., Gretton, A., Tang, Y., Barreto, A., Dabney, W., Bellemare, M. G., and Rowland, M. A distributional analogue to the successor representation. In International Conference on Machine Learning (ICML), 2024b.

Wu, R., Chen, Y., Swamy, G., Brantley, K., and Sun, W.

Diffusing states and matching scores: A new framework for imitation learning. In *International Conference on* Learning Representations (ICLR), 2025.

Yang, L., Zhang, Z., Zhang, Z., Liu, X., Xu, M., Zhang, W., Meng, C., Ermon, S., and Cui, B. Consistency flow matching: Defining straight flows with velocity consistency. *CoRR*, abs/2407.02398, 2024.

Yarats, D., Brandfonbrener, D., Liu, H., Laskin, M., Abbeel, P., Lazaric, A., and Pinto, L. Don't change the algorithm, change the data: Exploratory data for offline reinforcement learning. *CoRR*, abs/2201.13425, 2022.

Zhang, P., Chen, X., Zhao, L., Xiong, W., Qin, T., and Liu, T.-Y. Distributional reinforcement learning for multidimensional reward functions. In Neural Information Processing Systems (NeurIPS), 2021.

Zhu, C., Wang, X., Han, T., Du, S. S., and Gupta, A. Distributional successor features enable zero-shot policy optimization. In Neural Information Processing Systems (NeurIPS), 2024.

# Appendices

## A. Related Work

The Successor Representation (Dayan, 1993) was originally proposed for tabular MDPs and was later generalized to continuous state spaces with the Successor Measure (Blier et al., 2021). Successor Features (Barreto et al., 2017; 2020) extends these ideas by instead modeling the evolution of multi-dimensional features assuming rewards decompose linearly over these features. Prior works have leveraged these methods for zero-shot policy evaluation (Dayan, 1993; Barreto et al., 2017; Wiltzer et al., 2024b), zero-shot policy optimization (Borsa et al., 2019; Touati & Ollivier, 2021; Touati et al., 2023; Park et al., 2024; Zhu et al., 2024; Cetin et al., 2024; Tirinzoni et al., 2025), imitation learning (Pirotta et al., 2024; Jain et al., 2025), exploration (Machado et al., 2020; Jain et al., 2023), representation learning (Le Lan et al., 2022; 2023a;b; Farebrother et al., 2023; Ghosh et al., 2023), and building temporal abstractions (Machado et al., 2018; 2023). Janner et al. (2020) originally proposed a method to learn a generative model of the successor measure with modeling techniques spanning from Generative Adversarial Networks (Goodfellow et al., 2014) to Normalizing Flows (Dinh et al., 2015; Rezende & Mohamed, 2015) like RealNVP (Dinh et al., 2017). Followup work (e.g., Thakoor et al., 2022; Tomar et al., 2024) explored other generative modeling techniques including various types of auto-encoders (e.g., Higgins et al., 2017; van den Oord et al., 2017). Also of note is recent work learning generative models of multi-dimensional cumulants including features (Wiltzer et al., 2024a; Zhu et al., 2024) and multi-variate reward functions (Zhang et al., 2021). Prior work by Wiltzer et al. (2024b) sought to deal with the instability of long-horizon predictions in GHMs by employing an n-step mixture distribution where they sample t ∼ Geometric(1 − γ) and bootstrap if *t > n*; otherwise returning the state at time t along the trajectory. Without resorting to importance sampling this approach is limited to the on-policy setting. Finally, most closely related to our work is that of Schramm & Boularias (2024) who provide a preliminary and limited derivation of what we term TD2-DD. In contrast, our work not only rigorously formalizes and significantly extends these ideas but also integrates them into the more general flow-matching framework (Lipman et al., 2023; 2024), additionally incorporating extensions to score-matching (Song et al., 2021b;b) and diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020). Moreover, we conduct an extensive empirical analysis, demonstrating the efficacy of our approach - an aspect notably absent from Schramm & Boularias (2024).

## B. Extension To Score Matching And Diffusion Models

This section extends our framework to score matching and denoising diffusion models. We leverage the unification of these methods under stochastic differential equations (Song et al., 2021b) introducing an analogous class of Temporal Difference Diffusion methods.

## B.1. Background

Both score-based generative modeling (Song & Ermon, 2019) and diffusion probabilistic modeling (Sohl-Dickstein et al., 2015; Ho et al., 2020) can be unified under the framework of stochastic differential equations (SDE) introduced in Song et al. (2021b). Unlike in flow-matching, time is inverted in diffusion models and ranges from time 0 to T. Given the data distribution q0 and prior simple distribution qT (the "noise" distribution), we construct a diffusion process {Xt}t∈[0,T] such that X0 ∼ q0 and XT ∼ qT . This diffusion can be modeled as the solution to an Ito SDE:
dXt = f(t) Xt dt + g(t) dWt | X0 ∼ q0 , (16)
where Wt is a standard Brownian motion and f : [0, T] → R
dis scalar function called the drift coefficient, and g : [0, T] →
R
 is scalar function known as diffusion coefficient.

Generating samples from X0 ∼ q0 consists in sampling XT ∼ qT and reversing the forward-SDE process in (16). A known result from Anderson (1982) states that the reverse of a diffusion process is also a diffusion process, running backward in time and given by the reverse-time SDE:

$$\mathrm{d}X_{t}=\Big(f(t)\,X_{t}-g(t)^{2}\,\nabla_{X_{t}}\log q_{t}(X_{t})\Big)\mathrm{d}t+g(t)\,\mathrm{d}\overline{{{W}}}_{t}\;\mid\;X_{T}\sim q_{T}\,,$$
dt + g(t) dWt | XT ∼ qT , (17)
where Wt is a Brownian motion when time flows backwards from T to 0, dt is an infinitesimal negative timestep and qt is

$$(16)^{\frac{1}{2}}$$

$$(17)^{\frac{1}{2}}$$
$$(18)$$
$$(19)$$

the marginal distribution of Xt. Therefore, once we learn the score of the marginal distribution ∇x log qt(x), we can sample from q0 by simulating the reverse diffusion process (17).

To estimate ∇x log qt(x), we can train a time-dependent score-based model s˜(*· · ·* ; θ) : [0, T] × R
d → R
d via the denoising diffusion / score matching objective (Vincent, 2011; Song & Ermon, 2019):

$$\ell_{\mathrm{DD}}(\theta)=\mathbb{E}_{t\sim\mathcal{U}([0,1]),X_{0}\sim q_{0}}\mathbb{E}_{X_{t}\sim q_{t|0}(\,\cdot\,|X_{0})}\Big[\|\tilde{s}_{t}(X_{t};\theta)-\nabla_{X_{t}}\log q_{t|0}(X_{t}\mid X_{0})\|^{2}\Big]\,.$$
2i. (18)
qt = (1 − γ)
→
qt + γ
↷

qt is:
For ℓDD to be tractable, we need to know the conditional probability qt|0. Usually, specific choices of the drift and diffusion coefficients ft and gt are used such that qt|0 is always a Gaussian distribution N (· | αtx0, σ2 t
), where the mean αt and variance σ 2 t can be computed in closed-form. The global minimizer of ℓDD(θ) denoted by s
⋆ t
(x) is equal to the score function
∇x log qt(x), thanks to the following proposition:
Proposition 2 (Vincent 2011). Let qt(x) = Rq0(x0)qt|0(x|x0) dx0*, then we have:*

$$\nabla_{\theta}\,\ell_{\mathrm{DD}}(\theta)=\nabla_{\theta}\,\mathbb{E}_{t,X_{t}\sim q_{t}}\left[\left\|{\tilde{s}}_{t}(X_{t};\theta)-\nabla_{X_{t}}\log q_{t}(X_{t})\right\|^{2}\right].\tag{1}$$

## B.2. Temporal Difference Diffusion

To learn a predictive model of mπ using diffusion from an offline dataset, we follow a similar approach to what we presented in §3 and we define an iterative process starting from initial weights θ
(0) and at each iteration minimizing the Temporal-Difference Denoising Diffusion (TD-DD) loss:

$$\ell_{\text{TD-DD}}(\theta)=\mathbb{E}_{p,t,X_{0},X_{t}}\left[\left\|\tilde{s}_{t}(X_{t}\mid S,A;\theta)-\nabla_{x}\log q_{t|0}(X_{t}\mid X_{1})\right\|^{2}\right],$$ (TD-DD; 20) $$\text{where},\;X_{0}\sim\left(\mathcal{T}^{\pi}\widehat{m}_{0|T}^{(n)}\right)(\cdot\mid S,A),\;X_{t}\sim q_{t|0}(\cdot\mid X_{0})\;.$$

In order to sample X0 ∼
$\sim\left(T^{\pi}\tilde{m}_{\mu\nu}^{(n)}\right)(\cdot\mid s,a)$. 
πme
0|T
(· | *s, a*), with probability 1 − γ, we return the successor state S
′ ∼ P(· | *S, A*).

Otherwise, with probability γ we solve the following reverse-time SDE from XT using the score s˜
(n)
t,

$$\mathrm{d}X_{t}=\left(f(t)\,X_{t}-g(t)^{2}\bar{s}_{t}^{(n)}(X_{t}\mid S,A)\right)\mathrm{d}t+g(t)\mathrm{d}\overline{{{W}}}_{t}\,.$$
dt + g(t)dWt . (21)
Minimizing ℓTD-DD(θ) leads to score function s˜
(n+1)
t(s | *s, a*) generating a marginal probability q
(n+1)
tthat approximates T
πq
(n)
0at t = 0.

Following the TD2-CFM blueprint, we can further exploit the structure of the target bootstrapped distribution to design an improved diffusion process that converts Gaussian noise to T
πq
(n)
0. First, we show below that the mixture of a diffusion process is also a diffusion process with modified drift and diffusion functions. Lemma 2. *Consider two diffusion processes with drift functions*
→
f and
↷

f*, sharing the same diffusion coefficient* g:

$$(21)$$
$$\begin{array}{r}{\mathrm{d}X_{t}={\vec{f}}_{t}(X_{t})\,\mathrm{d}t+g(t)\,\mathrm{d}W}\\ {\mathrm{d}X_{t}={\vec{f}}_{t}(X_{t})\,\mathrm{d}t+g(t)\,\mathrm{d}W\,.}\end{array}$$

```
Let →
   qt and 
         ↷
           
         qt be their marginal distribution, then the diffusion process corresponding to the mixture marginal distribution

```

$$\mathrm{d}X_{t}={\frac{(1-\gamma){\vec{q}}_{t}{\vec{f}}_{t}+\gamma{\widehat{q}}_{t}{\widehat{f}}_{t}}{(1-\gamma){\vec{q}}_{t}+\gamma{\widehat{q}}_{t}}}(X_{t})\,\mathrm{d}t+g(t)\,\mathrm{d}W\,.$$

Proof. The marginal probabilities →
p and ↷
p are characterized by the Fokker-Planck equations:

$$\begin{array}{l}{{\frac{\partial\vec{p}_{t}}{\partial t}=-\mathrm{div}(\vec{p}_{t}\vec{f}_{t})+\frac{g_{t}^{2}}{2}\Delta\vec{p}_{t}}}\\ {{\frac{\partial\vec{p}_{t}}{\partial t}=-\mathrm{div}(\widehat{p}_{t}\widehat{f}_{t})+\frac{g_{t}^{2}}{2}\Delta\widehat{p}_{t}}}\end{array}$$

where div is the divergence operator and ∆ = div∇ is the Laplace operator. Therefore,

∂pt ∂t = (1 − γ) ∂ → pt ∂t  + γ ∂ ↷ pt ∂t = −div(→ pt → ft) +  g 2 t 2 ∆ → pt − div(↷ pt ↷ ft) +  g 2 t 2 ∆ ↷ pt = −div (1 − γ) → pt → ft + γ ↷ pt ↷ ft + g 2 t 2 ∆ ((1 − γ) → pt + γ ↷ pt) = div pt (1 − γ) → pt → ft + γ ↷ pt ↷ ft) (1 − γ) → pt + γ ↷ pt !+ g 2 t 2 ∆pt .

The drift (1−γ)
→pt
→
ft+γ
↷pt
↷

ft
(1−γ)
→pt+γ
↷ptand the diffusion coefficient gt satisfy the Fokker-Planck equation with the probability path pt, and therefore their associated diffusion process generate pt.

Lemma 2 can be easily extended to the case of a continuous mixture of diffusion processes. This result shows that it is possible to use two independent diffusion processes for the two terms in the sampling process induced by the Bellman operator. For the first, we can use the standard noising diffusion process:

$$\vec{q}_{t}(x\mid s,a)=\int q_{t\mid0}(x\mid s^{\prime})P(\mathrm{d}s^{\prime}\mid s,a)\,,$$

where we sample Xt ∼ qt|0(· | s
′) by simulating a simple forward diffusion process (16). For the second term, we can leverage the GHM m
(n)
tat the previous iteration to construct the process,

$$\widetilde{q}_{t}^{(n)}(x\mid s,a)=\int m_{t}^{(n)}(x\mid s^{\prime},\pi(s^{\prime}))\,P(\mathrm{d}s^{\prime}\mid s,a)\,,$$

where m
(n)
t(x | s
′, a′) is the marginal probability of the reverse SDE induced by the score s
(n),

$$\mathrm{d}X_{t}=\left(f(t)\,X_{t}-g(t)^{2}\,s_{t}^{(n)}(X_{t}\mid s,a)\right)\,\mathrm{d}t+g(t)\,\mathrm{d}\overline{{{W}}}_{t}\,.$$

Additionally, 
↷

q
(n)
t(x | *s, a*), as continuous mixture of diffusion's marginals m
(n)
t(x | s
′, π(s
′)) weighted by P(s
′| *s, a*),
can be generated by the diffusion process,

$$\mathrm{d}X_{t}=\left(f(t)\,X_{t}-g(t)^{2}\,\widehat{s_{t}}(X_{t}\mid s,a)\right)\mathrm{d}t+g(t)\,\mathrm{d}\overline{{{W}}}_{t},{\mathrm{~where~}}$$
$$\widetilde{s}_{t}(x_{t}\mid s,a)=\frac{\int P(\mathrm{d}s^{\prime}\mid s,a)\,q_{t}^{(n)}(x\mid s^{\prime},\pi(s^{\prime}))\,s_{t}^{(n)}(x_{t}\mid s^{\prime},\pi(s^{\prime}))}{\int P(\mathrm{d}s^{\prime}\mid s,a)\,q_{t}^{(n)}(x\mid s^{\prime},\pi(s^{\prime}))}\,.$$

.
Given these two diffusion processes, the target probability q
(n+1)
t = (1 − γ)
→
qt + γ
↷

q
(n)
tcan be generated by the following
reverse SDE,
$$\mathrm{d}X_{t}=\left(f(t)X_{t}-g(t)\right)$$
2s
(n+1)
t(Xt | *s, a*)
dt + g(t) dWt, where s
(n+1)
t(x | *s, a*) = 
(1−γ)
→qt∇x log→qt+γ
↷q
(n)
t
↷s
(n)
t
(1−γ)
→qt+γ
↷q
(n)
t
(x | *s, a*). Therefore, we can learn s˜t(*· · ·* ; θ) to approximate s
(n+1)
t by minimizing the loss,

$$\ell(\theta)=(1-\gamma)\mathbb{E}_{\rho,t,X_{t}\sim\widetilde{q}_{t}^{(1:S,A)}}\Big{[}\|\tilde{s}(X_{t}\mid S,A;\theta)-\nabla_{X_{t}}\log\widetilde{q}_{t}(X_{t}\mid S,A)\|^{2}\Big{]}\tag{22}$$ $$+\gamma\mathbb{E}_{\rho,t,X_{t}\sim\widetilde{q}_{t}^{(1:S,A)}}\Big{[}\|\tilde{s}(X_{t}\mid S,A;\theta)-\widetilde{s}_{t}^{(n)}(X_{t}\mid S,A)\|^{2}\Big{]}.$$

We can simplify the first term via Proposition 2 (since →
qt(x|*s, a*) = Rqt|0(x|s
′)P(ds
′|*s, a*)), hence we have

$$\nabla_{\theta}\,\mathbb{E}_{\rho,t,X_{t}\sim\widehat{\mathcal{W}}(\mid s,a)}\left[\left\|\tilde{s}(X_{t}\mid s,a;\theta)-\nabla_{X_{t}}\log\widetilde{q}_{t}(X_{t}\mid S,A)\right\|^{2}\right]=$$ $$\nabla_{\theta}\,\mathbb{E}_{\rho,t,X_{t}\sim q_{t0}(\mid S^{\prime})}\left[\left\|\tilde{s}(X_{t}\mid S,A;\theta)-\nabla_{X_{t}}\log q_{t\mid0}(X_{t}\mid S^{\prime})\right\|^{2}\right].$$

Moreover, using a similar argument for equivalence between the gradient of marginal and conditional flow-matching objectives, we can show that

$$\nabla_{\theta}\,\mathbb{E}_{\rho,t_{i},X_{t}\sim\widehat{q}_{t_{i}}^{(n)}(\cdot\mid S,A)}\left(\left\|\widehat{s}(X_{t}\mid S,A;\theta)-\widehat{q}_{t}^{(n)}(X_{t}\mid S,A)\right\|^{2}\right)=$$ $$\nabla_{\theta}\,\mathbb{E}_{\rho,t_{i},X_{t}\sim q_{T},X_{t}\sim q_{T}^{(n)}(\cdot\mid s,A)}\left(\left\|\widehat{s}(X_{t}\mid S,A;\theta)-s_{t}^{(n)}(X_{t}\mid S,A)\right\|^{2}\right).$$

$$(23)$$

This leads us to the final TD2-DD loss function,

$$\begin{array}{l}{{\ell_{\mathrm{TP}^{2}\cdot\mathrm{DD}}(\theta)=(1-\gamma)\mathbb{E}_{\rho,t,X_{t}\sim q_{t|0}(\mid S^{\prime})}\Big[\big\|\tilde{s}_{t}(X_{t}\mid S,A;\theta)-\nabla_{x}\log p_{t|0}(X_{t}\mid S^{\prime})\big\|^{2}\Big]}}\\ {{\qquad\qquad+\gamma\mathbb{E}_{\rho,t,X_{t}\sim q_{t|T}^{(n)}(\mid S^{\prime},\pi(S^{\prime}))}\Big[\big\|\tilde{s}(X_{t}\mid S,A;\theta)-\tilde{s}_{t}^{(n)}(X_{t}\mid S^{\prime},\pi(S^{\prime})\big\|^{2}\Big]\,.}}\end{array}$$
2i(23)

## C. Experimental Details C.1. Evaluation

Evaluating a GHM can be challenging, TD-based losses employing bootstrapping do not provide a good signal as to the quality of the learned model. Instead, we opt to measure 1) the likelihood of a trajectory coming from the true discounted occupancy of a given policy, 2) the Earth Mover's Distance (EMD; Rubner et al., 2000) between samples from the true occupancy and our GHM which provides an estimate of the distance between these two probability distributions, and 3) the value-function approximation error. In all cases, to obtain samples from the true discounted occupancy, we collect trajectories
{(s0, s1*, . . . , s*T )}
N
i=1 from policy π and subsequently resample states according to t ∼ Geometic(1 − γ) for a particular discount factor γ ∈ [0, 1). Armed with samples from mπ we compute the aforementioned metrics following the procedures stated below along with the parameter values outlined in Table 3.

| experiments. Evaluation         | Hyperparameter               | Value   |
|---------------------------------|------------------------------|---------|
| Number of states s0             | 64                           |         |
| Number of m-samples per state   | 2048                         |         |
| EMD                             | Number of episodes per state | 1       |
| Episode length                  | 1000                         |         |
| Number of state s0              | 64                           |         |
| Number of GHM-samples per state | 2048                         |         |
| Number of episodes per state    | 1                            |         |
| Episode length                  | 1000                         |         |
| MSE(V)                          | Number of z samples          | 256     |
| GPI                             | Number of GHM samples        | 128     |
| Number of FB inference samples  | 250, 000                     |         |

Normalized Negative Log-Likelihood. To compute the log-likelihood of our flow matching and diffusion methods, we take advantage of the following change in variables formula (Dinh et al., 2015; Rezende & Mohamed, 2015; Chen et al., 2018),

$$\log\left(\widetilde{m}(x_{1}\,|\,s,a;\theta)\right)=\log\varphi(x_{0})+\int_{0}^{1}\frac{\partial\log\left(\widetilde{m}(x_{t}\,|\,s,a;\theta)\right)}{\partial x_{t}}\;d t\,,$$

where φ is the probability density function of a standard Gaussian distribution, which acts as the prior on x0. The change in log density over time can be written as the following differential equation called the instantaneous change of variables formula (Chen et al., 2018, Theorem 1),

$$\frac{\partial\log\left(\widetilde{m}(x_{t}\,|\,s,a;\theta)\right)}{\partial x_{t}}=-\,\mathrm{Tr}\left(\frac{\partial\,\bar{v}_{t}(x_{t}\,|\,s,a;\theta)}{\partial x_{t}}\right)\,.$$

We can now compute the log-likelihood for a sample X ∼ mπ(· | *s, a*) by integrating the total change in log-density backward in time from x1 = X to obtain x0 which has tractable likelihood. In practice, we solve the following coupled initial value problem using numerical integration (Grathwohl et al., 2019),

$$\begin{array}{c}{{x_{0}}}\\ {{\left[\log\widetilde{m}(x_{1}\,|\,s,a;\theta)-\log\varphi(x_{0})\right]=\int_{1}^{0}\left[\begin{array}{c}{{-\tilde{v}_{t}(x_{t}\,|\,s,a;\theta)}}\\ {{\mathrm{Tr}\left(\partial\,\frac{\tilde{v}_{t}(x_{t}\,|\,s,a;\theta)}{\partial x_{t}}\right)}}\end{array}\right]d t\,,}}\\ {{\mathrm{where}\ \ \left[\begin{array}{c}{{x_{1}}}\\ {{\log\widetilde{m}(x\,|\,s,a;\theta)-\log\widetilde{m}(x_{1}\,|\,s,a;\theta)}}\end{array}\right]=\left[\begin{array}{c}{{X}}\\ {{0}}\end{array}\right]\,.}}\end{array}$$

$$(24)$$

For all experiments we report the negative log-likelihood *normalized by the dimension of the observation space*. Earth Mover's Distance We compute the Earth Mover's Distance (EMD; Rubner et al., 2000), also known as the Wasserstein-1 distance, between m = 2048 samples from the ground truth distribution X ∼ mπ(· | Sk, Ak) and our learned GHM Xe ∼ me (· | Sk, Ak; θ) for a set of randomly sampled state-action pairs {(Sk, Ak)}
nk=1. Intuitively, the EMD quantifies the minimum cost required to transform one distribution into another, where the cost is defined in terms of the Euclidean distance between states X(i), X(j). Formally, we have,

$$\mathrm{EMD}(\{X^{(1)},\ldots,X^{(m)}\},\{\widetilde{X}^{(1)},\ldots,\widetilde{X}^{(m)}\})=\min_{\xi\in\Xi}\sum_{i,j}\xi_{ij}\sum_{k=1}^{d}\left(X_{k}^{(i)}-\widetilde{X}_{k}^{(j)}\right)^{2}\,,$$

where ξ is a transport plan such that ξij specifies the proportion of mass moved from Xito Xej . We report the average EMD
across n = 64 source states using the Python Optimal Transport (Flamary et al., 2021) library. Value Function Mean Square Error (MSE(V)). We compute the mean square error between a Monte-Carlo estimation Ve π MC of the value function V
π(s) and the estimation VeGHM obtained using the learned model. We obtain Ve π MC by collecting a trajectory {(s0, s1*, . . . , s*T )} from policy π and computing the discounted sum of rewards. We generate a single trajectory since both the policy and the environment are deterministic. The GHM estimate is given by (2), i.e.,

$$\widetilde{V}_{\mathrm{GHM}}^{\pi}(s)=(1-\gamma)^{-1}\mathbb{E}_{\widetilde{X}\sim\widetilde{m}(\cdot|s,\pi(s))}\bigg[r(\widetilde{X})\bigg]\,.$$

Then, MSE(Ve π MC, Ve π GHM) = ES0∼ν h(Ve π GHM(S0) − Ve π MC(S0))2i. We average our results over 64 initial states S0 sampled from the initial state distribution ν. Planning with GPI. We evaluate planning performance by computing the average return over 100 episodes, each lasting 1, 000 steps, for every task. For the Forward-Backward representation (Touati & Ollivier, 2021), we directly follow the policy πwr(thus at = πwr(st)) where wr = E(S,R)∼ρ[B(s) · R] is the zero-shot policy embedding inferred using 250, 000 transitions labeled with the task reward function r. Given that FB provides a direct way of estimating the value function of a policy (i.e., Qπw r(s, a) = F(*s, a, w*)
Tzr), we can do planning in the policy embedding space by solving the following problem:

$$w_{t}^{\mathrm{FB-GP}}\in\arg\operatorname*{max}F(s_{t},\pi_{w}(s_{t}),w)^{T}w_{r}.$$

This optimization problem requires no generation except sampling from D(W). We approximate the max using 255 samples from D(W) and additionally incorporating wr to ultimately maximize over 256 policies. On the other hand, for GHM-GPI,
we solve the following optimization problem,

$$w_{t}^{\mathrm{GHM-GPI}}\in\arg\max_{w\sim D(\mathsf{W})}\ \underbrace{(1-\gamma)^{-1}\,\mathbb{E}_{X\sim m^{\pi_{w}}(\cdot\left|s_{t},\pi_{w}(s_{t})\right.))}\big[r(X)\big]}_{Q^{\pi_{w}}(s_{t},\pi_{w}(s_{t}))},$$

which requires generating samples from mπw . In our experiments we generate 128 samples from mπw .

## C.2. Environments

Experiments in this paper were conducted with a subset of domains from the DeepMind Control Suite (Tunyasuvunakool et al., 2020) highlighted in Figure 4.

## C.3. Geometric Horizon Models

This section describes each class of generative model used for our empirical experiments.

C.3.1. FLOW MATCHING

| Table 4. Summary of how different TD-flow algorithms generate the target probability path and vector field. ↷ The neural ode ψt is defined by the vector field v t computed at iteration n. ↷ pt ↷ vt   |         |                     |                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------------------|------------------------|
| Algorithm 1 Template for TD-Flow algorithms 1: Inputs: offline dataset D, policy π, batch size n, Polyak coefficient ζ, weight decay λ, randomly initialized weights θ, discount factor γ, learning rate η, one-step conditional path →pt|1 and conditional vector-field →ut|1, bootstrap path ↷pt and vector-field ↷vt. 2: for n = 1, . . . do 3: Sample mini-batch {(Sk, Ak, S′ k)} K k=1 from D 4: for k = 1, . . . , K do 5: Sample tk ∼ U([0, 1]) → →ptk|1(· | S ′ 6: Sample Xk ∼ k) 7: → ℓk(θ) = vtk ( Xk | Sk, Ak; θ) − → →utk|1( Xk | S → k) ′ 2 ↷ ↷ptk 8: Sample Xk ∼ (· | S ′ k, π(S ′ k); ¯θ) 9: ↷ ℓk(θ) = vtk ( ↷ ↷vtk ( ↷ ′ ′ k); ¯θ) 2 Xk | Sk, Ak; θ) − Xk | S k, π(S 10: end for 11: # Compute loss → ↷ 12: ℓ(θ) = 1 K PK k=1(1 − γ) ℓk(θ) + γ ℓk(θ) 13: # Perform gradient step 14: θ ← θ − η∇θ  ℓ(θ) + λ∥θ∥ 2  15: # Update parameters of target vector field 16: ¯θ ← ζ ¯θ + (1 − ζ)θ 17: end for                                                                                                                                                                                                         | X0 ∼ m0 |                     |                        |
| M                                                                                                                                                                                                       |         |                     |                        |
| CFD-T                                                                                                                                                                                                   | X       | ut|1(Xt | X1)       |                        |
| 1 = ψ1(X0 | S ′ , A′ ; ¯θ) Xt ∼ pt|1(· | X1)                                                                                                                                                            |         |                     |                        |
| )                                                                                                                                                                                                       | X0 ∼ m0 |                     |                        |
| (CMCFD-T                                                                                                                                                                                                | X       | ut|0,1(Xt | X0, X1) |                        |
| ′ , A′ ; ¯θ)                                                                                                                                                                                            |         |                     |                        |
| 1 = ψ1(X0 | S Xt ∼ pt|0,1(· | X0, X1)                                                                                                                                                                   |         |                     |                        |
| FM                                                                                                                                                                                                      |         | X0 ∼ m0             | vt(Xt | S ′ , A′ ; ¯θ) |
| -C2 TD Xt = ψt(X0 | S ′ , A′ ; ¯θ)                                                                                                                                                                      |         |                     |                        |

To discuss the TD-Flow methods introduced herein, we first unify the loss function through defining a general template for the loss as,

$$\ell(\theta)=(1-\gamma)\mathbb{E}_{\rho,t,X_{t}\sim\vec{p}_{t1}(\cdot\mid S^{\prime})}\Big{[}\big{\|}v_{t}(X_{t}\mid S,A;\theta)-\vec{u}_{t\mid1}(X_{t}\mid S^{\prime})\big{\|}^{2}\Big{]}$$ $$\quad+\gamma\mathbb{E}_{\rho,t,X_{t}\sim\vec{p}_{t}^{(n)}(\cdot\mid Z)}\left[\big{\|}v_{t}(X_{t}\mid S,A;\theta)-\vec{v}_{t}^{(n)}(X_{t}\mid Z)\big{\|}^{2}\right].$$

```
We can now recover each algorithm by a specific choice of the target probability path 
                                                                                     ↷
                                                                                       
                                                                                     p
                                                                                       (n)
                                                                                      tand vector field 
                                                                                                          ↷
                                                                                                            
                                                                                                          v
                                                                                                            (n)
                                                                                                            tas illustrated
in Figure 4. Based on this unified structure, we present pseudo-code for the TD flow methods in Figure 1. In practice,
instead of proceeding through full iterations, we use standard mini-batch gradient updates with a target network ¯θ updated
as a moving average of θ.

```

When employing the conditional probability path →
pt|1 and vector field →
ut|1 we use the standard Gaussian linear interpolation defined as →
pt|1(· | X1) = N (· | tX1,(1 − t)
2I), hence Xt = tX1 + (1 − t)X0 ∼ pt|1, resulting in →
ut|1(Xt | X1) =
(X1 − Xt)/(1 − t) (Lipman et al., 2023). The source distribution for all experiments is m0(·) = N (· | 0, I). To sample from the Neural ODE we use the Midpoint method with a constant step size of dt = t/10 for a total of 10 steps. We found both coupled and TD2 methods do not require many solver steps and hypothesize this is due to the reduction in transport cost as analyzed in Appendix E.7. For all flow and diffusion-based methods, we employ a U-Net-style architecture (Ronneberger et al., 2015) that has hierarchical skip connections throughout an MLP. We embed the timestep t by first increasing its dimensionality with a sinusoidal embedding before transforming it through a two-layer MLP with mish activations (Misra, 2019). We further process additional conditioning information, such as the state-action pair and Forward-Backward embedding z through an additional two-layer MLP, whose result then gets concatenated with our time embedding. Finally, the network integrates all prior conditioning information through FiLM modulation (Perez et al., 2018) that replaces the learned affine transformation for layer normalization (Ba et al., 2016).