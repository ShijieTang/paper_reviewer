# Adjoint Schrödinger Bridge Sampler

Guan-Horng Liu1,∗, Jaemoo Choi2,∗, Yongxin Chen2**, Benjamin Kurt Miller**1, Ricky T. Q. Chen1,∗
1FAIR at Meta, 2Georgia Institute of Technology, ∗Core contributors

## Abstract

Computational methods for learning to sample from the Boltzmann distributionwhere the target distribution is known only up to an unnormalized energy functionhave advanced significantly recently. Due to the lack of explicit target samples, however, prior diffusion-based methods, known as *diffusion samplers*, often require importance-weighted estimation or complicated learning processes. Both trade off scalability with extensive evaluations of the energy and model, thereby limiting their practical usage. In this work, we propose **Adjoint Schrödinger Bridge Sampler** (ASBS), a new diffusion sampler that employs simple and scalable matchingbased objectives yet without the need to estimate target samples during training.

ASBS is grounded on a mathematical model—the Schrödinger Bridge—which enhances sampling efficiency via kinetic-optimal transportation. Through a new lens of stochastic optimal control theory, we demonstrate how SB-based diffusion samplers can be learned at scale via Adjoint Matching and prove convergence to the global solution. Notably, ASBS generalizes the recent Adjoint Sampling (Havens et al., 2025) to arbitrary source distributions by relaxing the so-called memoryless condition that largely restricts the design space. Through extensive experiments, we demonstrate the effectiveness of ASBS on sampling from classical energy functions, amortized conformer generation, and molecular Boltzmann distributions. Codes are available at https://github.com/facebookresearch/adjoint_samplers.

## 1 Introduction

Sampling from Boltzmann distributions is a fundamental problem in computational science, with widespread applications in Bayesian inference, statistical physics, and chemistry (Box and Tiao, 2011; Binder et al., 1992; Tuckerman, 2023). Mathematically, we aim to sample from a target distribution ν(x) known up to a unnormalized, often differentiable, energy function E(x) : X ⊆ R
d → R,

$$\nu(x):={\frac{e^{-E(x)}}{Z}},\qquad{\mathrm{where~}}Z:=\int_{\mathcal{X}}e^{-E(x)}\mathrm{d}x$$
$$(1)$$

−E(x)dx (1)
is an intractable normalization constant. For instance, the energy function E(x) of a molecular system quantifies the stability of a chemical structure based on the 3D positions of particles. A lower energy indicates a more stable structure and hence a higher likelihood of its occurrence, *i.e.,* ν(x) ∝ e
−E(x).

Classical methods that generate samples from ν(x) rely on Markov Chain Monte Carlo algorithms, which run a Markov chain whose stationary distribution is ν(x) (Metropolis et al., 1953; Neal, 2001; Del Moral et al., 2006). These methods, however, tend to suffer from slow mixing time and require extensive evaluations of energy function, limiting their practical usages due to prohibitive complexity. To improve sampling efficiency, modern samplers focus on learning better proposal distributions (Noé et al., 2019; Midgley et al., 2023). Among those, recent advances in diffusion-based generative models (Song et al., 2021; Ho et al., 2020) have given rise to a family of *Diffusion Samplers*, which

| Design condition for (2)                                        | Learning method for u θ t                                             |    |    |    |
|-----------------------------------------------------------------|-----------------------------------------------------------------------|----|----|----|
| Method                                                          | Non-memoryless Arbitrary prior Matching objective1 No reliance on IWs |    |    |    |
| PIS (Zhang and Chen, 2022) DDS (Vargas et al., 2023)            | ✗                                                                     | ✗  | ✗  | ✓  |
| LV-PIS & LV-DDS (Richter and Berner, 2024)                      | ✗                                                                     | ✗  | ✗  | ✗  |
| PDDS (Phillips et al., 2024) iDEM (Akhound-Sadegh et al., 2024) | ✗                                                                     | ✗  | ✓  | ✗  |
| AS (Havens et al., 2025)                                        | ✗                                                                     | ✗  | ✓  | ✓  |
| Sequential SB (Bernton et al., 2019)                            | ✓                                                                     | ✓  | ✗  | ✗  |
| Adjoint Schrödinger Bridge Sampler (Ours)                       | ✓                                                                     | ✓  | ✓  | ✓  |

Table 1: Compared to prior diffusion samplers, Adjoint Schrödinger Bridge Sampler (**ASBS**) offers the most flexible design for diffusion samplers (2), while learning the drift u θ t via scalable matching objectives that do not rely on computation of importance weights (IWs). consider stochastic differential equations (SDEs) of the following form:

$$\mathrm{d}X_{t}=\left[f_{t}(X_{t})+\sigma_{t}u_{t}^{\theta}(X_{t})\right]\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t},\qquad X_{0}\sim\mu(X_{0}),$$
$$(2)$$
t(Xt)dt + σtdWt, X0 ∼ µ(X0), (2)
where ft(x) : [0, 1] *× X → X* the base drift, σt : [0, 1] → R>0 the noise schedule, and µ(x) the initial source distribution. Given (ft, σt, µ), the diffusion sampler learns a parametrized drift u θ t(x)
transporting samples to the target distribution ν(x) at the terminal time t = 1. Computational methods for learning diffusion samplers have grown significantly recently (Zhang and Chen, 2022; Vargas et al., 2023; Berner et al., 2024; Chen et al., 2025). Due to the distinct problem setup in (1), the target distribution is defined exclusively by its energy E(x), rather than by explicit target samples. This characteristic renders modern generative modeling techniques for scalabilityparticularly the score matching objectives1—less applicable. As such, prior matching-based diffusion samplers (Phillips et al., 2024; Akhound-Sadegh et al., 2024; De Bortoli et al., 2024) often require computationally intensive estimation of target samples via importance weights (IWs). Recently, Havens et al. (2025) introduced Adjoint Sampling (AS), a new class of diffusion samplers whose matching objectives rely only on on-policy samples, thereby greatly enhancing scalability. By incorporating stochastic optimal control (SOC) theory (Kappen, 2005; Todorov, 2007), AS facilitates the use of Adjoint Matching (Domingo-Enrich et al., 2025), a novel matching objective that imposes self-consistency in generated samples, effectively eliminating the needs for target samples. The efficiency of AS, however, is achieved through a specific instantiation of the SDE (2) to satisfy the so-called *memoryless* condition. This condition—formally discussed in Section 2—restricts its source distribution to be Dirac delta µ(x*) :=* δ, precluding the use of common priors such as Gaussian or domain-specific priors such as the harmonic oscillators in molecular systems (Jing et al., 2023). Notably, the memoryless condition underlies all previous matching-based diffusion samplers, restricting the design space of (2) from other choices known to enhance transportation efficiency (Shaul et al., 2023). While the condition has been relaxed in non-matching-based methods at extensive computational complexity (Richter and Berner, 2024; Bernton et al., 2019), no existing diffusion sampler—to our best understanding—has successfully combined matching objectives with non-memoryless condition. Table 1 summarizes the comparison between prior diffusion samplers. In this work, we propose **Adjoint Schrödinger Bridge Sampler (ASBS)**, a new adjoint-matchingbased diffusion sampler that eliminates the requirement for memoryless condition entirely. Formally, ASBS recasts learning diffusion sampler as a distributionally constrained optimization, known as the Schrödinger Bridge (SB) problem (Schrödinger, 1931, 1932; Léonard, 2013; Chen et al., 2016):

$$\min_{u}D_{\text{KL}}(p^{u}||p^{\text{base}})=\mathbb{E}_{X\sim p^{u}}\left[\int_{0}^{1}\tfrac{1}{2}\|u_{t}^{\theta}(X_{t})\|^{2}\text{d}t\right],$$ (3a) s.t. $$\text{d}X_{t}=\left[f_{t}(X_{t})+\sigma_{t}u_{t}^{\theta}(X_{t})\right]\text{d}t+\sigma_{t}\text{d}W_{t},\qquad X_{0}\sim\mu(X_{0}),\qquad X_{1}\sim\nu(X_{1}).\tag{3b}$$

Here, p u denotes the path distribution induced by the SDE in (3b), whereas p base := p u:=0 denotes the path distribution induced by the "base" SDE when ut := 0. By minimizing their KL divergence, the SB problem (3) seeks the kinetic-optimal drift u
⋆
t—an optimality structure well correlated 1The matching objective is a simple regression loss, E∥u θ t (Xt) − vt(Xt, X1)∥
2, w.r.t. some tractable vt.

with sampling efficiency in generative modeling (Finlay et al., 2020; Liu et al., 2023). Since the SOC problem in AS corresponds to a specific case of the SB problem with (ft, µ) := (0, δ),
ASBS extends AS to handle non-memoryless conditions by solving more general SB problems (see Theorem 3.1). Computationally, ASBS retains all scalability advantages from AS by utilizing an adjoint-matching objective that removes the need for estimating target samples. It also introduces a corrector-matching objective to correct nontrivial biases arising from non-memoryless conditions. We prove that alternating optimization between the two matching objectives is equivalent to executing the Iterative Proportional Fitting algorithm (Kullback, 1968), ensuring global convergence of ASBS
to u
⋆
t(see Theorem 3.2). Though extensive experiments, we show superior performance of ASBS
over prior diffusion samplers across various benchmarks on sampling multi-particle energy functions. In summary, we present the following contributions:
- We introduce **ASBS**, an SB-based diffusion sampler capable of sampling target distributions using only unnormalized energy functions, by solving general SB problems with arbitrary priors.

- We base ASBS on a new SOC framework that removes the restrictive memoryless condition, develop a scalable matching-based algorithm, and prove theoretical convergence to global solution.

- We show ASBS's superior performance over prior methods on sampling Boltzmann distributions of classical energy functions, alanine dipeptide molecule and amortized conformer generation.

## 2 Preliminary

We revisit the memoryless condition introduced by Domingo-Enrich et al. (2025) and examine its impact on the constructions of SOC-based diffusion samplers (Zhang and Chen, 2022; Havens et al., 2025), which are closely related to our ASBS. Additional review can be found in Appendix A. Stochastic Optimal Control (SOC) The SOC problem (4) studies an optimization problem:

$$\mathrm{{\bf{Control}}\;({\bf SOC})\;\;\;\;\;\Gamma}$$
$$\operatorname*{min}_{u}\mathbb{E}_{X\sim p^{u}}\left[\int_{0}^{1}{\frac{1}{2}}\|u_{t}(X_{t})\|^{2}\mathrm{d}t+g(X_{1})\right]\quad{\mathrm{s.t.~}}(2),$$

s.t. (2), (4)
which, unlike the SB problem (3), includes an additional *terminal cost* g(x) : X → R at the terminal time t = 1 and considers the SDE without the terminal constraint X1 ∼ ν. The primary reason for studying this specific optimization problem is that the optimal distribution is known analytically by2

$$p^{*}(X_{0},X_{1})=p^{\rm base}(X_{0},X_{1})e^{-g(X_{1})+V_{0}(X_{0})},\ \ \ {\rm where}\ \ V_{0}(x)=-\log\int\!p_{1|0}^{\rm base}(y|x)e^{-g(y)}{\rm d}y\tag{5}$$

is the initial value function. That is, the optimal distribution p
⋆is an exponentially tilted version of the base distribution, p base := p u:=0. Specifically, p base is tilted by the terminal cost "−g(X1)" and the initial value function V0(X0), which is intractable. Consequently, to ensure its marginal p
⋆(X1)
follows the target distribution ν(X1), we must eliminate the *initial value function bias* from V0(X0).

Memoryless condition & SOC-based diffusion sampler A common approach to eliminate the aforementioned initial value function bias, adopted by most diffusion samplers, is to restrict the class of base processes to be *memoryless*. Formally, the memoryless condition assumes statistical independency between X0 and X1 in the base distribution:

$$p^{\mathrm{base}}(X_{0},X_{1})\ {\stackrel{\mathrm{memoryless}}{:=}}\ p^{\mathrm{base}}(X_{0})p^{\mathrm{base}}(X_{1}).$$
base(X1). (6)
This memoryless condition (6) simplifies the optimal distribution at the terminal time t = 1 and, upon choosing a proper terminal cost g(x), recovers the target distribution ν,

p
$${}^{*}(X_{1})\stackrel{\mathrm{memoryless}}{{=}}\int\!p^{\mathrm{base}}(X_{0})p^{\mathrm{base}}(X_{1})e^{-g(X_{1})+V_{0}(X_{0})}\mathrm{d}X_{0}\propto p^{\mathrm{base}}(X_{1})e^{-g(X_{1})}=\nu(X_{1}),$$

```
where the last equality is due to setting the terminal cost to g(x) := log p
                                                                            base
                                                                            1
                                                                               (x)
                                                                             ν(x). Typically, the

```

memoryless condition (6) is enforced by a careful design of the base distribution p base or, equivalently,

$$(6)^{\frac{1}{2}}$$

tim e t
the parameters (ft, σt, µ) in (2). For instance, the variance-preserving process (VP; Song et al., 2021) considers a linear base drift ft, a noise schedule σt that grows significantly with time, and a Gaussian prior µ; see Figure 1. Alternatively, one could implement (6) with Dirac delta prior µ(x*) :=* δ0(x) and ft := 0, leading to the following SOC problem (Zhang and Chen, 2022):

$$\min_{u}\mathbb{E}_{X\sim p^{u}}\Big{[}\int_{0}^{1}\frac{1}{2}\|u_{t}(X_{t})\|^{2}\mathrm{d}t+\log\frac{p_{1}^{\mathrm{bus}}(X_{1})}{\nu(X_{1})}\Big{]}\ \ \ \mathrm{s.t.}\ \mathrm{d}X_{t}=\sigma_{t}u_{t}(X_{t})\mathrm{d}t+\sigma_{t}\mathrm{d}W_{t},\ \ X_{0}{=}0.\tag{7}$$

Based on the aforementioned reasoning, solving (7) results in a diffusion sampler that transports samples to the target distribution at t=1, with Adjoint Sampling (Havens et al., 2025) as the only scalable method of this class. Despite encouraging, the SOC problem in (7) is nevertheless limited by its trivial source, precluding potentially more effective options for sampling Boltzmann distributions.

## 3 Adjoint Schrödinger Bridge Sampler

We introduce a new diffusion sampler by solving the SB problem (3), where the target distribution ν(x) is given by its energy function E(x) rather than explicit samples. All proofs are left in Appendix B.

## 3.1 Soc Characteristics Of The Sb Problem

The SB problem (3)—as an optimization problem with distribution constraints—is widely explored in optimal transport, stochastic control, and recently machine learning (Léonard, 2012; Chen et al., 2021; De Bortoli et al., 2021). Its kinetic-optimal drift u
⋆satisfies the following optimality equations:

$$u_{t}^{*}(x)=\sigma_{t}\nabla\log\varphi_{t}(x),\quad\text{where}\left\{\begin{aligned}\varphi_{t}(x)&=\int_{1|t}^{\text{bus}}(y|x)\varphi_{1}(y)\mathrm{d}y,\quad\varphi_{0}(x)\hat{\varphi}_{0}(x)=\mu(x)\\ \hat{\varphi}_{t}(x)&=\int_{1|0}^{\text{bus}}(x|y)\hat{\varphi}_{0}(y)\mathrm{d}y,\quad\varphi_{1}(x)\hat{\varphi}_{1}(x)=\nu(x)\end{aligned}\right.\tag{8a}$$
$$({\mathfrak{g}})$$

and p base t|s
(y|x) := p base(Xt=y|Xs=x) is the transition kernel of the base process for observing y at time t given x at time s. The *SB potentials* φt(x), φˆt(x) ∈ C
1,2([0, 1], R
d) are then defined (up to some multiplicative constant) as solutions to forward and backward time integrations w.r.t. p base t|s
.

Equation (8) are computationally challenging to solve—even when p base t|s has an analytical solutiondue to the intractable integration and coupled boundaries at t = 0 and 1. Our key observation is that the first equation (8a) resembles the optimality condition of the SOC problem (4) (see Appendix A.1).

This implies that the optimality conditions of SB hints an SOC reinterpretation, which, as we will demonstrate, is more tractable than solving (8) directly. We formalize our finding below. Theorem 3.1 (SOC characteristics of SB). *The kinetic-optimal drift* u
⋆
tin (8) *solves an SOC problem*

$$\operatorname*{min}_{u}\mathbb{E}_{X\sim p^{u}}\left[\int_{0}^{1}{\frac{1}{2}}\|u_{t}(X_{t})\|^{2}\mathrm{d}t+\log{\frac{\hat{\varphi}_{1}(X_{1})}{\nu(X_{1})}}\right]\quad s.t.{\mathrm{~(2)}}.$$

Theorem 3.1 suggests that *every* SB problem (3) can be solved like an SOC problem (4) with the terminal cost g(x) := log φˆ1(x)
ν(x). Comparing to the formulation in Adjoint Sampling (Havens et al.,
2025), the two SOC problems, namely (7) and (9), differ in their terminal costs—where p base 1is replaced by φˆ1—and the relaxation of the source distribution from Dirac delta X0 = 0 to general source µ(X0). How φˆ1(·) **debiases non-memoryless SOC problems** Taking a closer look at the effect of φˆ1, notice that the optimal distribution of the SB problem—according to Theorem 3.1 and (5)—follows

$$p^{*}(X_{0},X_{1})=p^{\mathrm{base}}(X_{0},X_{1})\exp\left(-\log{\frac{\dot{\varphi}_{1}(X_{1})}{\nu(X_{1})}}-\log\varphi_{0}(X_{0})\right),$$
$$(10)$$
, (10)
where "− log φ0" is the equivalent initial value function. One can verify that the marginal at the terminal time t = 1 indeed satisfies the target distribution,

$$\begin{array}{c}{{p^{*}(X_{1})=\int\!p^{*}(X_{0},X_{1})\mathrm{d}X_{0}\stackrel{(10)}{=}\frac{\nu(X_{1})}{\hat{\varphi}_{1}(X_{1})}\!\int\!p^{\mathrm{base}}(X_{0},X_{1})\frac{1}{\hat{\varphi}_{0}(X_{0})}\mathrm{d}X_{0}}}\\ {{\stackrel{(\boxtimes)}{=}\frac{\nu(X_{1})}{\hat{\varphi}_{1}(X_{1})}\!\int\!p^{\mathrm{base}}(X_{1}|X_{0})\hat{\varphi}_{0}(X_{0})\mathrm{d}X_{0}\stackrel{(\boxtimes)}{=}\nu(X_{1}).}}\end{array}$$
$$(11)$$

That is, the optimality equations in (8), in their essence, construct a specific function φˆ1(·) that eliminates the initial value function bias associated with any non-memoryless processes, thereby ensuring that the optimal distribution satisfies the target ν at t = 1.

## 3.2 Adjoint Sampling With General Source Distribution

We now specialize Theorem 3.1 to sampling Boltzmann distributions (1), where ν(x) ∝ e
−E(x),

and hence the terminal cost of the new SOC problem in (9) becomes log 
φˆ1(x)
ν(x) 
= E(x) + log ˆφ1(x).
To encourage minimal transportation cost (Chen and Georgiou, 2015; Peyré and Cuturi, 2017), we
consider the Brownian-motion base process with a degenerate base drift ft := 0. Applying Adjoint
Matching (AM; Domingo-Enrich et al., 2025) to the resulting SOC problem leads to
u ⋆ = arg min Ep base t|0,1 p u¯ 0,1 -∥ut(Xt) + σt (∇E + ∇ log ˆφ1) (X1)∥ 2, u¯ = stopgrad(u). (12)
u Note that the AM objective in (12) functions as a self-consistency loss—in that both the regression and its expectation depend on the optimization variable u. This makes (12) particularly suitable for learning SB-based diffusion samplers, unlike previous matching-based SB methods (Shi et al., 2023; Liu et al., 2024), which all require ground-truth target samples from X1 ∼ ν. Computing the AM objective in (12) requires knowing ∇ log ˆφ1(x), which, as we discussed in (11),
serves as a *corrector* that debiases the optimization toward the desired target. Notably, this corrector function ∇ log ˆφ1(x) also admits a variational form (Peluchetti, 2022, 2023; Shi et al., 2023):3

$$\nabla\log\hat{\varphi}_{1}=\operatorname*{arg\,min}_{h}\mathbb{E}_{p_{0,1}^{\mathrm{sat}}}\left[\|h(X_{1})-\nabla_{x_{1}}\log p^{\mathrm{base}}(X_{1}|X_{0})\|^{2}\right].$$

To summarize, Equations (12) and (13) characterize two distinct matching objectives that any kineticoptimal drift u
⋆
t of SBs must satisfy. When the source distribution degenerates to Dirac delta X0 := 0,
(13) is minimized at ∇ log p base 1, and (12) simply recovers the objective used in Adjoint Sampling
(Havens et al., 2025). In other words, (12) and (13) should be understood as a generalization of Adjoint Sampling to handle arbitrary—including *non-memoryless*—source distributions.

## 3.3 Alternating Optimization With Adjoint And Corrector Matching

Building upon the theoretical characterization in Section 3.2, we aim to design a learning algorithm that finds a diffusion sampler satisfying (12) and (13), which correspond to two simple matchingbased objectives. However, these matching objectives cannot be naively implemented due to their interdependency: Solving (12) for the kinetic-optimal drift u
⋆requires knowing ∇ log ˆφ1. Likewise, solving (13) for the corrector function ∇ log ˆφ1 requires samples from u
⋆. We relax the interdependency with an alternating optimization scheme. Specifically, given an approximation of
∇ log ˆφ1 ≈ h
(k−1) from the previous stage k −1, we first update the drift u
(k) with the AM objective:
3Formally, ∇ log ˆφt(x) is the kinetic-optimal drift along the reversed time coordinate s := 1 − t, and (13) is its variational formulation, *i.e.,* the Markovian projection at s = 0; see Appendix A.2 for details.

$$(13)$$

Algorithm 1 Adjoint Schrödinger Bridge Sampler (ASBS)
Require: Sample-able source X0 ∼ µ, differentiable energy E(x), parametrized uθ(*t, x*) and hϕ(x)
1: Initialize h
(0)
ϕ:= 0 2: for stage k in 1, 2*, . . .* do 3: Update drift u
(k)
θby solving (14) ▷ adjoint matching 4: Update corrector h
(k)
ϕby solving (15) ▷ corrector matching 5: **end for**

AMCM
u (k):= arg min u Ep base t|0,1 p u¯0,1 h∥ut(Xt) + σt(∇E + h (k−1))(X1)∥ 2i, u¯ = stopgrad(u). (14)
$$u^{(k)}:=$$
Then, we use the resulting drift u
(k)to update h
(k) by minimizing the following matching objective, which—in light of the corrector role of ∇ log ˆφ1—we refer to as the *Corrector Matching* objective:

$$h^{(k)}:=\arg\operatorname*{min}_{h}\mathbb{E}_{p_{0,1}^{u(k)}}\left[\|h(X_{1})-\nabla_{x_{1}}\log p^{\mathrm{base}}(X_{1}|X_{0})\|^{2}\right].$$
2. (15)
Equation (15) should be distinguish from the bridge-matching objectives in data-driven SB methods
(Shi et al., 2023; Somnath et al., 2023), where X1 must be drawn from the target distribution ν. In contrast, the matching objectives in (14) and (15) depend only on model samples at the current stage X1 ∼ p u
(k)
θ (X1|X0), hence can be used to learn SB-based diffusion samplers at scale.

The alternating optimization between (14) and (15) creates a sequence of updates, (u
(0), h(0)) →
· · ·(u
(k), h(k)) *→ · · ·* , that may be thought of as running coordinate descent between the control u and the corrector h. Intuitively, at each stage k, we first find the control u
(k)that best aligns with the corrector from previous stage, h
(k−1), then update the corrector h
(k)accordingly to reflect the
"memorylessness" of the current control u
(k). We summarize our method, **Adjoint Schrödinger**
Bridge Sampler (ASBS), in Algorithm 1, while leaving the full details with additional components, such as replay buffers, in Appendix C. Finally, we prove that this alternating optimization indeed converges to the kinetic-optimal drift u
⋆in (8).

Theorem 3.2 (Global convergence of ASBS). Algorithm 1 converges to the Schrödinger bridge solution of (3)*, provided all matching stages achieve their critical points, i.e.,*

$$(15)$$
$$\operatorname*{lim}_{k\to\infty}u^{(k)}=u^{\star}.$$

## 4 Theoretical Analysis

We provide the proof of Theorem 3.2 and highlight theoretical insights throughout. While ASBS is specialized to a degenerate base drift ft := 0, all theoretical results here apply to general ft. To simplify notation, we omit the parameters θ, ϕ and reparametrize the corrector by h
(k) = ∇ log h¯(k).

All proofs are left in Appendix B. Our first result presents a variational characteristic to the solution of the AM objective in (14). Theorem 4.1 (Adjoint Matching solves a forward half bridge). Let p u
(k)be the path distribution induced by the drift u
(k)in (14) at stage k*. Then,* p u
(k)*solves the following variational problem:*

p
$u^{(k)}=\arg\min\left\{D_{\rm KL}(p||q^{\bar{h}^{(k-1)}}):p_{0}=\mu\right\}$,
$$(16)$$
$$\mathbf{T}\mathbf{J}\uparrow$$
$$(17)$$
$\left[\left(\frac{\partial\phi}{\partial x}\right)^2\right]$
p
h¯(k−1) ) : p0 = µ	, (16)
where q h¯(k−1) *is the path distribution induced by a "backward" SDE on the reversed time coordinate* s := 1 − t*, defined by the corrector from the previous stage* h¯(k−1):

$$\mathrm{d}Y_{s}=\left[-f_{s}(Y_{s})+\sigma_{s}^{2}\right]$$

2 s∇ log ϕs(Ys)ds + σsdWs, ϕs(y) = 
Zp

$$\mathrm{\frac{{\operatorname{ase}}}{-s|0}}(y|z)\phi_{1}(z)\mathrm{d}z,$$

with the boundary conditions Y0 ∼ ν and ϕ0(y) = h¯(k−1)(y). Theorem 4.1 suggests that any SOC problems with the terminal cost g(x) := log h¯(k)(x)
ν(x)can be reinterpreted as KL minimization w.r.t. a specific *backward* SDE (17) that is fully characterized by ν—which serves as its source distribution—and h¯(k)—which defines its drift through the function ϕs(y). The objective in (16) differs from the one in the original SB problem (3) by disregarding the target boundary constraint, X1 ∼ ν. Consequently, (16) only solves a forward half bridge.

Next, we show that the CM objective (15) admits a similar variational form, except backward in time.

Theorem 4.2 (Corrector Matching solves a backward half bridge). Let h¯(k) *be the corrector in* (15)
at stage k*. Then, the path distribution* q h¯(k)*solves the following variational problem:*

$$\begin{array}{r}{{\left\{D_{\mathrm{KL}}(p^{u^{(k)}})\right\}}}\\ {{q^{\bar{h}^{(k)}}=\operatorname*{arg\,min}_{q}\left\{D_{\mathrm{KL}}(p^{u^{(k)}}||q):q_{1}=\nu\right\}}}\\ {{q^{\bar{h}^{(k)}}=\operatorname*{arg\,min}_{q}\left\{D_{\mathrm{KL}}(p^{u^{(k)}}||q):q_{1}=\nu\right\}}}\end{array}$$
$$(18)$$
(k)||q) : q1 = ν	(18)
Unlike (16), the objective in (18) disregards the source boundary constraint µ instead, thereby solving a backward half bridge. Theorems 4.1 and 4.2 imply that our ASBS in Algorithm 1 *implicitly* employs an optimization scheme that alternates between solving forward and backward half bridges, thereby instantiating the celebrated Iterative Proportional Fitting algorithm (IPF; Fortet, 1940; Kullback, 1968). Combining with the analysis by (De Bortoli et al., 2021) leads to our final result in Theorem 3.2.

## 5 Related Works

We provide additional clarification on SB-related works and leave the full review to Appendix A.3. Data-driven Schrödinger Bridges The SB problem has attracted notable interests in machine learning due to its connection to diffusion-based generative models (Wang et al., 2021). Earlier methods implemented classical IPF algorithms (De Bortoli et al., 2021; Vargas et al., 2021; Chen et al., 2022), with scalability later enhanced by bridge matching-based methods (Shi et al., 2023; Liu et al., 2024). Unlike ASBS, all of them focus on generative modeling and assume access to extensive target samples during training, making them unsuitable for sampling from Boltzmann distributions. SB-inspired Diffusion Samplers Notably, in the context of diffusion samplers, the SB formulation has been constantly emphasized as a mathematically appealing framework for both theoretical analysis and method motivation (Zhang and Chen, 2022; Vargas et al., 2024; Richter and Berner, 2024; Havens et al., 2025). None of the prior methods, however, offers general solutions to learning SB-based diffusion samplers, instead specializing to either the memoryless condition or non-matching-based objectives, which largely complicate the learning process (see Table 1). Conceptually, our ASBS stands closest to SSB (Bernton et al., 2019) by learning general SB samplers. However, the two methods differ fundamentally in scalability: SSB is a Sequential Monte Carlo-based method (Chopin, 2002) augmented with learned transition kernels using Gaussian-approximated SB potentials. As with many MCMC-augmented samplers (Gabrié et al., 2022; Matthews et al., 2022), SSB requires extensive evaluations on the energy E(x), in contrast to ASBS, which is much more energy-efficient.

Method Sinkhorn ↓ W2 ↓ E(·) W2 ↓ W2 ↓ E(·) W2 ↓ W2 ↓ E(·) W2 ↓

PDDS (Phillips et al., 2024) - 0.92±0.08 0.58±0.25 4.66±0.87 56.01±10.80 - — SCLD (Chen et al., 2025) 0.44±0.06 1.30±0.64 0.40±0.19 2.93±0.19 27.98± 1.26 - — PIS (Zhang and Chen, 2022) 0.65±0.25 0.68±0.28 0.65±0.25 1.93±0.07 18.02± 1.12 4.79±0.45 228.70±131.27 DDS (Vargas et al., 2023) 0.63±0.24 0.92±0.11 0.90±0.37 1.99±0.13 24.61± 8.99 4.60±0.09 173.09± 18.01 LV-PIS (Richter and Berner, 2024) - 1.04±0.29 1.89±0.89 - — - — iDEM (Akhound-Sadegh et al., 2024) - 0.70±0.06 0.55±0.14 1.61±0.01 30.78±24.46 4.69±1.52 93.53± 16.31 AS (Havens et al., 2025) 0.32±0.06 0.62±0.06 0.55±0.12 1.67±0.01 2.40± 1.25 4.04±0.05 30.83± 8.19

ASBS (**Ours**) 0.15±0.02 0.43±0.05 0.20±0.11 1.59±0.03 1.99± 1.01 4.00±0.03 28.10± 8.15

MW-5 (d=5) DW-4 (d = 8) LJ-13 (d = 39) LJ-55 (d = 165)

Norm aliz ed Den sit y DW-4 ASBS (ours) Ground Truth
-60 -45 -30 -15 Energy E(x)
.00 .02 .04

.06 LJ-13
-26 -22 -18 -14 Energy E(x)
.0 .1 .2 .3 Complexity per Grad. Update Ave rag e N
FE on Mo del 10 3 10 2 10 1 10 0 10 1 10 2 10 3 Average NFE on Energy 10 0 10 1 10 2 10 3 PIS, DDS
ASBS AS
iDEM

## 6 Experiments

Benchmarks We evaluate our ASBS on three classes of multi-particle energy functions E(x).

- *Synthetic energy functions* These are classical potentials based on pair-wise distances of an n-particle system, where E(x) is known analytically. Following (Akhound-Sadegh et al., 2024; Chen et al., 2025), we consider a 2D 4-particle Double-Well potential (DW-4), a 1D 5-particle Many-Well potential (MW-5), a 3D 13-particle Lennard-Jones potential (LJ-13) and a 3D 55particle Lennard-Jones potential (LJ-55). For the ground-truth samples, we sample analytically from MW-5 and use the MCMC samples from (Klein et al., 2023) for the rest of three potentials.

- *Alanine dipeptide* This is a molecule consisting of 22 atoms in 3D. Specifically, we consider the alanine dipeptide in an implicit solvent and aim to sample from its Boltzmann distribution at a temperature 300K. Following prior methods (Zhang and Chen, 2022; Wu et al., 2020), we use the energy function E(x) from the OpenMM library (Eastman et al., 2017) and consider a more structural internal coordinate with the dimension d = 60. The ground-truth samples contain 107 configurations, simulated from Molecular Dynamics (Midgley et al., 2023).

- *Amortized conformer generation* Finally, we consider a new benchmark proposed in (Havens et al., 2025) for large-scale conformer generation. Conformers are locally stable configurations located at the local minima of the molecule's potential energy surface (Hawkins, 2017). Sampling conformers is essentially a conditional generation task, targeting a Boltzmann distribution ν(x|g) ∝ e
− 1τ E(x|g)at a low temperature τ ≪ 1, conditioned on the molecular topology g ∈ G.

The training set Gtrain contains 24,477 molecular topologies from SPICE (Eastman et al., 2023), represented by the SMILES strings (Weininger, 1988), whereas the test set Gtest contains 80 topologies from SPICE and another 80 from GEOM-DRUGS (Axelrod and Gomez-Bombarelli, 2022). As with (Havens et al., 2025), we consider E(x|g) a foundation model *eSEN* from (Fu et al., 2025), which predicts energy with density-functional-theory accuracy at a much lower computational cost. We use CREST conformers (Pracht et al., 2024) as the ground-truth samples.

Baselines and evaluation We compare ASBS with a wide range of diffusion samplers, including PIS (Zhang and Chen, 2022), DDS (Vargas et al., 2023), PDDS (Phillips et al., 2024), SCLD (Chen

Table 3: Comparison between diffusion samplers on sampling the molecular Boltzmann distribution

of the alanine dipeptide. We report the KL divergence DKL for the 1D marginal across five torsion

angles and the Wasserstein-2 W2 on jointly (*ϕ, ψ*), known as Ramachandran plots (see Figure 5). Best results are highlighted.

DKL on each torsion's marginal ↓ W2 on joint ↓

Method *ϕ ψ γ*1 γ2 γ3 (ϕ, ψ)

PIS (Zhang and Chen, 2022) 0.05±0.03 0.38±0.49 5.61±1.24 4.49±0.03 4.60±0.03 1.27±1.19 DDS (Vargas et al., 2023) 0.03±0.01 0.16±0.07 2.44±0.96 0.03±0.00 0.03±0.00 0.68±0.09 AS (Havens et al., 2025) 0.09±0.09 0.04±0.04 0.17±0.17 0.56±0.09 0.51±0.06 0.65±0.52

ASBS (**Ours**) 0.02±0.00 0.01±0.00 0.03±0.01 0.02±0.00 0.02±0.00 0.25±0.01

Table 4: Results on large-scale amortized conformer generation, evaluated on two test sets, SPICE and GEOM-DRUGS, both with and without post-processing relaxation. We report the coverage (%) and Absolute Mean RMSD (AMR) of the recall at the threshold **1.0Å**. Note that "*+RDKit warmup*"

refers to warm-starting the model uθ using RDKit conformers; see Appendix D for details. Best

results without and with RDKit warm-up are highlighted separately.

without relaxation with relaxation

SPICE GEOM-DRUGS SPICE GEOM-DRUGS

Method Coverage ↑ AMR ↓ Coverage ↑ AMR ↓ Coverage ↑ AMR ↓ Coverage ↑ AMR ↓ RDKit ETKDG (Riniker and Landrum, 2015) 56.94±35.82 1.04±0.52 50.81±34.69 1.15±0.61 70.21±31.70 0.79±0.44 62.55±31.67 0.93±0.53

AS (Havens et al., 2025) 56.75±38.15 0.96±0.26 36.23±33.42 1.20±0.43 82.41±25.85 0.68±0.28 64.26±34.57 0.89±0.45

ASBS w/ Gaussian prior (**Ours**) 73.04±31.95 0.83±0.24 50.23±35.98 1.05±0.43 88.26±20.57 0.60±0.24 72.32±29.68 0.77±0.35 ASBS w/ harmonic prior (**Ours**) 74.05±31.61 0.82±0.23 53.14±35.69 1.03±0.42 88.71±18.63 0.59±0.24 72.77±29.94 0.78±0.35 AS +RDKit warmup (Havens et al., 2025) 72.21±30.22 0.84±0.24 52.19±35.20 1.02±0.34 87.84±19.20 0.60±0.23 73.88±28.63 0.76±0.34 ASBS +RDKit warmup (**Ours**) 77.84±28.37 0.79±0.23 57.19±35.14 0.98±0.40 88.08±18.84 0.58±0.24 73.18±30.09 0.76±0.37

et al., 2025), LV (Richter and Berner, 2024), iDEM (Akhound-Sadegh et al., 2024) and finally Adjoint Sampling (AS; Havens et al., 2025). For the conformer generation task, we include additionally a domain-specific baseline, RDKit ETKDG (Riniker and Landrum, 2015), which relies on chemistrybased heuristics. The evaluation pipelines are consistent with prior methods, where we adopt the SCLD setup for MW-5, the PIS setup for alanine dipeptide, and the AS setup for all the rest; see Appendix D for details.

ASBS models For all tasks, we consider a degenerate base drift ft := 0, as discussed in Section 3.2, and set σt a geometric noise schedule. For energy functions that directly take particle systems as inputs—such as DW, LJ, and eSEN—we parametrize the models uθ, hϕ with two Equivariant Graph Neural Networks (Satorras et al., 2021) and consider a domain-specific source distribution—the harmonic prior (Jing et al., 2023). Formally, for an n-particle system x = {xi}
n i=0, the harmonic prior µharmonic(x) is a quadratic potential that can be sampled analytically from an anisotropic Gaussian:
µharmonic(x) ∝ exp(−
α 2 Pi,j ∥xi − xj∥
2). (19)
For other energy functions, we use standard fully-connected neural networks and consider Gaussian priors. All models are trained with Adam (Kingma and Ba, 2015) and, following standard practices (Havens et al., 2025; Akhound-Sadegh et al., 2024), utilize replay buffers; see Appendix C for details.

Results Table 2 presents the results on synthetic energy functions. Notably, ASBS consistently outperforms prior diffusion samplers across all energy functions. In Figure 3, we compare the energy histograms of DW-4 and LJ-13 potentials between the ground-truth MCMC samples and those from ASBS. It is evident that ASBS generates samples that closely resemble the target Boltzmann distribution ν(x) ∝ e
−E(x), resulting in energy profiles E(x) that are almost indistinguishable from the ground truth. Computationally, Figure 4 shows the average number of evaluation required on the energy E(x) and the model uθ(*t, x*) for each gradient update. ASBS is much more efficient than most diffusion samplers, with a slight overhead compared to AS due to the additional network hϕ(x).

Table 3 summarizes the results for alanine dipeptide. Following standard pipeline (Zhang and Chen, 2022), we generate model samples X1 ∈ R
60 and extract five torsion angles—including the backbone Figure 6: Example of ASBS generative process on amortized conformer generation. Given an unseen molecular topology g ∈ Gtest from the test set—COCSc1sc2ccccc2[n+]1[O-] in this case—ASBS transports samples from the harmonic prior X0 ∼ µharmonic to generate conformers X1.

Co ve rage Re ca ll (
%
)

SPICE
0 0.5 1.0 1.5 2.0 Threshold (Ångström)
0 20 40 60 80 100SPICE + relax 0 0.5 1.0 1.5 2.0 Threshold (Ångström)
0 20 40 60 80 100GEOM-DRUG
0 0.5 1.0 1.5 2.0 Threshold (Ångström)
0 20 40 60 80 100GEOM-DRUG + relax 0 0.5 1.0 1.5 2.0 Threshold (Ångström)
0 20 40 60 80 100 RDKit AS ASBS gauss ASBS harmonic
angles ϕ, ψ and methyl rotation angles γ1, γ2, γ3—all of them exhibit multi-modal distributions.

Notably, ASBS achieves lowest KL divergence to the ground-truth marginals across all five torsions. Figure 5 further compares the joint distributions of (*ϕ, ψ*), known as the Ramachandran plots (Spencer et al., 2019), between ground-truth and ASBS. While ASBS identifies all high-density modes in the region ϕ ∈ [−π, 0], it misses few low-density modes. This mode-seeking behavior, inherit in all SOC-based diffusion samplers, could be improved with important weighting. We provide further discussions in Appendix D.4. Table 4 presents the recall for amortized conformer generation compared to ground-truth samples. For prior diffusion samplers, we primarily compare to AS (Havens et al., 2025) due to the benchmark's scale. Following AS, we ablate a warm-start stage using RDKit conformers, which are close but not identical to ground-truth samples, and include results with relaxation for post-generation optimization. Since AS is a specific instance of ASBS with a Dirac delta prior—as discussed in Section 3.2—any performance improvements from AS to ASBS highlight the added capability to handle arbitrary priors and, consequently, non-memoryless processes. Remarkably, without any warm-start, ASBS with the harmonic prior (19) already matches and, in many cases, surpasses the RDKit-warm-up AS. With warm-start, ASBS achieves best performance across most metrics. This highlights the significance of domain-specific priors, aiding exploration as effectively as warm-start with additional data, which may not always be available. Finally, we visualize the generation process of ASBS with harmonic prior (19) in Figure 6 and report the recall curves in Figure 7. In practice, we observe that ASBS achieves slightly better results with a harmonic prior compared to a Gaussian prior, with both significantly outperforming AS (Havens et al., 2025). See Appendix D.4 for further ablation studies.

## 7 Conclusion And Limitation

We introduced **Adjoint Schrödinger Bridge Sampler (ASBS)**, a new diffusion sampler for Boltzmann distributions that solves general SB problems given only target energy functions. ASBS is based on a scalable matching framework, converges theoretically to the global solution, and performs superiorly across various benchmarks. Despite these encouraging results, further enhancement with importance sampling techniques is worth investigating to mitigate the mode collapse inherent in SOC-inspired diffusion samplers. Exploring its effectiveness in sampling amortized Boltzmann distributions would also be valuable.

## Acknowledgements

The authors would like to thank Aaron Havens, Juno Nam, Xiang Fu, Bing Yan, Brandon Amos, and Brian Karrer for the helpful discussions and comments. JC and YC acknowledge support from NSF Grants ECCS-1942523, DMS-2206576, and CMMI-2450378.

## References

Tara Akhound-Sadegh, Jarrid Rector-Brooks, Avishek Joey Bose, Sarthak Mittal, Pablo Lemos, Cheng-Hao Liu, Marcin Sendera, Siamak Ravanbakhsh, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, and Alexander Tong. Iterated denoising energy matching for sampling from boltzmann densities. In *International Conference on Machine Learning (ICML)*, 2024.

Michael S Albergo and Eric Vanden-Eijnden. Nets: A non-equilibrium transport sampler. *arXiv* preprint arXiv:2410.02711, 2024.

Michael Arbel, Alex Matthews, and Arnaud Doucet. Annealed flow transport monte carlo. In International Conference on Machine Learning (ICML), 2021.

Simon Axelrod and Rafael Gomez-Bombarelli. GEOM, energy-annotated molecular conformations for property prediction and molecular generation. *Scientific Data*, 9(1):185, 2022.

Richard Bellman. The theory of dynamic programming. Technical report, Rand corp santa monica ca, 1954.

Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. *Transactions on Machine Learning Research (TMLR)*, 2024.

Espen Bernton, Jeremy Heng, Arnaud Doucet, and Pierre E Jacob. Schrödinger bridge samplers.

arXiv preprint arXiv:1912.13170, 2019.

Kurt Binder, Dieter W Heermann, and K Binder. *Monte Carlo simulation in statistical physics*,
volume 8. Springer, 1992.

Denis Blessing, Xiaogang Jia, Johannes Esslinger, Francisco Vargas, and Gerhard Neumann. Beyond elbos: a large-scale evaluation of variational methods for sampling. In Proceedings of the 41st International Conference on Machine Learning, pages 4205–4229, 2024.

George EP Box and George C Tiao. *Bayesian inference in statistical analysis*. John Wiley & Sons, 2011.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. http:
//github.com/google/jax.

Junhua Chen, Lorenz Richter, Julius Berner, Denis Blessing, Gerhard Neumann, and Anima Anandkumar. Sequential controlled langevin diffusions. In *International Conference on Learning* Representations (ICLR), 2025.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.

Tianrong Chen, Guan-Horng Liu, and Evangelos A Theodorou. Likelihood training of Schrödinger bridge using forward-backward SDEs theory. In International Conference on Learning Representations (ICLR), 2022.

Yongxin Chen and Tryphon Georgiou. Stochastic bridges of linear systems. *IEEE Transactions on* Automatic Control, 61(2):526–531, 2015.

Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. On the relation between optimal transport and schrödinger bridges: A stochastic control viewpoint. Journal of Optimization Theory and Applications, 169:671–691, 2016.

Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrödinger bridge. *SIAM Review*, 63(2):249–313, 2021.

Nicolas Chopin. A sequential particle filter method for static models. *Biometrika*, 89(3):539–552, 2002.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrödinger bridge with applications to score-based generative modeling. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Valentin De Bortoli, Michael Hutchinson, Peter Wirnsberger, and Arnaud Doucet. Target score matching. *arXiv preprint arXiv:2402.08667*, 2024.

Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. *Journal of the* Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006.

Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint Matching:
Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations (ICLR), 2025.

Peter Eastman, Jason Swails, John D Chodera, Robert T McGibbon, Yutong Zhao, Kyle A Beauchamp, Lee-Ping Wang, Andrew C Simmonett, Matthew P Harrigan, Chaya D Stern, Rafal P. Wiewiora, Bernard R. Brooks, and Vijay S. Pande. OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLoS computational biology*, 13(7):e1005659, 2017.

Peter Eastman, Pavan Kumar Behara, David L Dotson, Raimondas Galvelis, John E Herr, Josh T
Horton, Yuezhi Mao, John D Chodera, Benjamin P Pritchard, Yuanqing Wang, Gianni De Fabritiis, and Thomas E. Markland. SPICE, a dataset of drug-like molecules and peptides for training machine learning potentials. *Scientific Data*, 10(1):11, 2023.

Chris Finlay, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam Oberman. How to train your neural ODE: The world of jacobian and kinetic regularization. In International Conference on Machine Learning (ICML), 2020.

Robert Fortet. Résolution d'un système d'équations de M. Schrödinger. *Journal de Mathématiques* Pures et Appliquées, 19(1-4):83–105, 1940.

Xiang Fu, Brandon M Wood, Luis Barroso-Luque, Daniel S Levine, Meng Gao, Misko Dzamba, and C Lawrence Zitnick. Learning smooth and expressive interatomic potentials for physical property prediction. In *International Conference on Machine Learning (ICML)*, 2025.

Marylou Gabrié, Grant M Rotskoff, and Eric Vanden-Eijnden. Adaptive monte carlo augmented with normalizing flows. *Proceedings of the National Academy of Sciences*, 119(10):e2109420119, 2022.

WK HASTINGS. Monte carlo sampling methods using markov chains and their applications.

Biometrika, 57(1):97–109, 1970.

Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon Wood, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, Xiang Fu, Guan-Horng Liu, and Ricky T. Q. Chen. Adjoint Sampling: Highly scalable diffusion samplers via Adjoint Matching. In International Conference on Machine Learning (ICML), 2025.

Paul CD Hawkins. Conformation generation: The state of the art. Journal of chemical information and modeling, 57(8):1747–1756, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Kiyosi Itô. *On stochastic differential equations*, volume 4. American Mathematical Soc., 1951. Bowen Jing, Ezra Erives, Peter Pao-Huang, Gabriele Corso, Bonnie Berger, and Tommi S Jaakkola.

EigenFold: Generative protein structure prediction with diffusion models. In International Conference on Learning Representations (ICLR), Workshop Track, 2023.

Hilbert J Kappen. Path integrals and symmetry breaking for optimal control theory. Journal of Statistical Mechanics: Theory and Experiment, 2005(11):P11011, 2005.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

Leon Klein, Andrew Foong, Tor Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, and Ryota Tomioka. Timewarp: Transferable acceleration of molecular dynamics by learning time-coarsened dynamics. In Advances in Neural Information Processing Systems
(NeurIPS), 2023.

Jonas Köhler, Leon Klein, and Frank Noé. Equivariant flows: exact likelihood generative learning for symmetric densities. In *International conference on machine learning*, pages 5361–5370. PMLR, 2020.

Solomon Kullback. Probability densities with given marginals. *The Annals of Mathematical Statistics*,
39(4):1236–1243, 1968.

Greg Landrum. Rdkit: Open-source cheminformatics. https://www.rdkit.org, 2006. Jean-François Le Gall. *Brownian motion, martingales, and stochastic calculus*. Springer, 2016. Christian Léonard. From the schrödinger problem to the monge–kantorovich problem. Journal of Functional Analysis, 262(4):1879–1920, 2012.

Christian Léonard. A survey of the Schrödinger problem and some of its connections with optimal transport. *Discrete and Continuous Dynamical Systems*, 2013.

Christian Léonard, Sylvie Rœlly, and Jean-Claude Zambrini. Reciprocal processes. A measuretheoretical point of view. *Probability Surveys*, 2014.

Daniel S Levine, Muhammed Shuaibi, Evan Walter Clark Spotte-Smith, Michael G Taylor, Muhammad R Hasyim, Kyle Michel, Ilyes Batatia, Gábor Csányi, Misko Dzamba, Peter Eastman, et al. The open molecules 2025 (omol25) dataset, evaluations, and models. *arXiv preprint arXiv:2505.08762*, 2025.

Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I2SB: Image-to-Image Schrödinger bridge. In International Conference on Machine Learning (ICML), 2023.

Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos A Theodorou, and Ricky T. Q. Chen. Generalized Schrödinger bridge matching. In International Conference on Learning Representations (ICLR), 2024.

Alex Matthews, Michael Arbel, Danilo Jimenez Rezende, and Arnaud Doucet. Continual repeated annealed flow transport monte carlo. In *International Conference on Machine Learning (ICML)*,
2022.

Nicholas Metropolis, Arianna W Rosenbluth, Marshall N Rosenbluth, Augusta H Teller, and Edward Teller. Equation of state calculations by fast computing machines. *The journal of chemical physics*,
21(6):1087–1092, 1953.

Laurence Illing Midgley, Vincent Stimper, Gregor NC Simm, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Flow annealed importance sampling bootstrap. In International Conference on Learning Representations (ICLR), 2023.

Radford M Neal. Annealed importance sampling. *Statistics and computing*, 11:125–139, 2001.

F. Neese. The orca program system. *WIRES Comput. Molec. Sci.*, 2(1):73–78, 2012. doi: 10.1002/
wcms.81.

Kirill Neklyudov, Daniel Severo, and Alireza Makhzani. Action matching: A variational method for learning stochastic dynamics from samples. In International Conference on Machine Learning
(ICML), 2023.

Edward Nelson. *Dynamical theories of Brownian motion*, volume 106. Princeton university press, 2020.

Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. *Science*, 365(6457):eaaw1147, 2019.

Bernt Øksendal. Stochastic differential equations. In *Stochastic Differential Equations*, pages 65–84.

Springer, 2003.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In *Advances in neural information processing systems*, pages 8026–8037, 2019.

Stefano Peluchetti. Non-Denoising forward-time diffusions, 2022. https://openreview.net/
forum?id=oVfIKuhqfC.

Stefano Peluchetti. Diffusion bridge mixture transports, Schrödinger bridge problems and generative modeling. *arXiv preprint arXiv:2304.00917*, 2023.

Gabriel Peyré and Marco Cuturi. Computational optimal transport. Center for Research in Economics and Statistics Working Papers, 2017.

Gabriel Peyré and Marco Cuturi. Computational optimal transport: With applications to data science.

Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.

Angus Phillips, Hai-Dang Dau, Michael John Hutchinson, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. Particle denoising diffusion sampler. In *International Conference on Machine* Learning (ICML), 2024.

Philipp Pracht, Stefan Grimme, Christoph Bannwarth, Fabian Bohle, Sebastian Ehlert, Gereon Feldmann, Johannes Gorges, Marcel Müller, Tim Neudecker, Christoph Plett, Sebastian Spicher, Pit Steinbach, Patryk A. Wesołowski, and Felix Zeller. CREST—A program for the exploration of low-energy molecular chemical space. *The Journal of Chemical Physics*, 160(11), 2024.

Lorenz Richter and Julius Berner. Improved sampling via learned diffusions. In *International* Conference on Learning Representations (ICLR), 2024.

Sereina Riniker and Gregory A Landrum. Better informed distance geometry: using what we know to improve conformation generation. *Journal of chemical information and modeling*, 55(12):
2562–2574, 2015.

Simo Särkkä and Arno Solin. *Applied stochastic differential equations*, volume 10. Cambridge University Press, 2019.

Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In *International Conference on Machine Learning (ICML)*, 2021.

Erwin Schrödinger. *Über die Umkehrung der Naturgesetze*, volume IX. Sitzungsberichte der Preuss Akad. Wissen. Phys. Math. Klasse, Sonderausgabe, 1931.

Erwin Schrödinger. Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique.

In *Annales de l'institut Henri Poincaré*, 1932.

Neta Shaul, Ricky T. Q. Chen, Maximilian Nickel, Matthew Le, and Yaron Lipman. On kinetic optimal probability paths for generative models. In International Conference on Machine Learning
(ICML), 2023.

Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion Schrödinger bridge matching. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion Schrödinger bridges. In *Conference on Uncertainty in* Artificial Intelligence (UAI), 2023.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021.

Ryan K Spencer, Glenn L Butterfoss, John R Edison, James R Eastwood, Stephen Whitelam, Kent Kirshenbaum, and Ronald N Zuckermann. Stereochemistry of polypeptoid chain configurations.

Biopolymers, 110(6):e23266, 2019.

Vincent Stimper, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Resampling base distributions of normalizing flows. In International Conference on Artificial Intelligence and Statistics
(AISTATS), 2022.

Emanuel Todorov. Linearly-solvable Markov decision problems. In Advances in Neural Information Processing Systems (NeurIPS), 2007.

Mark E Tuckerman. *Statistical mechanics: theory and molecular simulation*. Oxford university press, 2023.

Francisco Vargas, Pierre Thodoroff, Neil D Lawrence, and Austen Lamacraft. Solving Schrödinger bridges via maximum likelihood. *Entropy*, 2021.

Francisco Vargas, Will Grathwohl, and Arnaud Doucet. Denoising diffusion samplers. In International Conference on Learning Representations (ICLR), 2023.

Francisco Vargas, Shreyas Padhy, Denis Blessing, and Nikolas Nüsken. Transport meets variational inference: Controlled monte carlo diffusions. In International Conference on Learning Representations (ICLR), 2024.

Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via Schrödinger bridge. In *International Conference on Machine Learning (ICML)*, 2021.

David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. *Journal of chemical information and computer sciences*, 28(1):31–36, 1988.

Hao Wu, Jonas Köhler, and Frank Noé. Stochastic normalizing flows. In *Advances in Neural* Information Processing Systems (NeurIPS), volume 33, pages 5933–5944, 2020.

Qinsheng Zhang and Yongxin Chen. Path integral sampler: A stochastic control approach for sampling. In *International Conference on Learning Representations (ICLR)*, 2022.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Our theoretical and empirical results validate the itemized claims made in the end of introduction. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Limitation is discussed in the last section, titled "Conclusion and Limitation". Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Proofs and assumptions of all theorems appearing in the main paper can be found in Appendix B. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems.

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Necessary information to reproduce our method is discussed in Section 6, with full details in Appendices C and D. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

5. **Open access to data and code**
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [No]

Justification: All data used in this work is open-source. Unfortunately, due to organizational policy, we are unable to release our source code at submission time. However, we plan to make it publicly available in the near future once administrative challenges are resolved. Guidelines:
- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All experimental setups are discussed in Section 6, with full details in Appendix D. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: All numerical values in Section 6 are averaged over a few random trials and we have reported their standard deviations. Guidelines:
- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We provide these details in the supplementary material. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: We read and comply with the NeurIPS Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [No] Justification: This work does not have novel societal impact beyond that of already existing diffusion samplers. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

11. **Safeguards**
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: This paper poses no such risks. Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: Original papers that produced the code package or dataset are all properly credited. Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.