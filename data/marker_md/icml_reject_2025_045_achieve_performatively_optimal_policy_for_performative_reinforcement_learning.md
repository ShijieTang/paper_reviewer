011

014 015 016

018

024

026

034

036

038

# Achieve Performatively Optimal Policy for Performative Reinforcement Learning

Anonymous Authors<sup>1</sup>

# Abstract

Performative reinforcement learning is an emerging dynamical decision making framework, which extends reinforcement learning to the common applications where the agent's policy can change the environmental dynamics. Existing works on performative reinforcement learning only aim at a performatively stable (PS) policy that maximizes an approximate value function. However, there is a provably positive constant gap between the PS policy and the desired performatively optimal (PO) policy that maximizes the original value function. In contrast, this work proposes a zerothorder performative policy gradient (0-PPG) algorithm that for the first time converges to the desired PO policy with polynomial computation complexity under mild conditions. For the convergence analysis, we prove two important properties of the nonconvex value function. First, when the policy regularizer dominates the environmental shift, the value function satisfies a certain gradient dominance property, so that any stationary point of the value function is a desired PO. Second, though the value function has unbounded gradient, we prove that all the sufficiently stationary points lie in a convex and compact policy subspace Π∆, where the policy value has a constant lower bound ∆ > 0 and thus the gradient becomes bounded and Lipschitz continuous.

# 1. Introduction

Reinforcement learning is a powerful dynamic decision making framework with many successes in AI, such as AlphaGo [\(Silver et al.,](#page-9-0) [2017\)](#page-9-0), AlphaStar [\(Vinyals et al.,](#page-9-1) [2019\)](#page-9-1), Pluribus [\(Brown and Sandholm,](#page-8-0) [2019\)](#page-8-0), large language model alignment [\(Bai et al.,](#page-8-1) [2022\)](#page-8-1) and reasoning

[\(Havrilla et al.,](#page-8-2) [2024\)](#page-8-2). However, most reinforcement learning works ignore the effect of the deployed policy on the environmental dynamics, including transition kernel and reward function. This effect is significant in some applications. For example, the behavior of the autonomous vehicles can affect the behavior of the pedestrians and the other vehicles, so the environment may become very different from the designers' imagination [\(Nikolaidis et al.,](#page-9-2) [2017\)](#page-9-2). Also, a recommender system formulated as a contextual Markov decision process not only affects the user demographics (context distribution) but also how users interact with the platforms [\(Chaney et al.,](#page-8-3) [2018;](#page-8-3) [Mansoury et al.,](#page-8-4) [2020\)](#page-8-4).

To account for such effect of deployed policy on environmental dynamics, performative reinforcement learning has been proposed by [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) where the transition kernel p<sup>π</sup> and reward function r<sup>π</sup> are modeled as functions of the deployed policy π. Similar to conventional reinforcement learning, the ultimate goal is to find the *performatively optimal (PO)* policy that maximizes the *performative value function*, defined as the accumulated discounted reward when deploying a policy π to its corresponding environment (pπ, rπ). However, the policy-dependent environmental dynamics pose significant challenge to achieve PO. Hence, [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) pursues a suboptimal *performatively stable (PS)* policy using repeated retraining method with environmental dynamics fixed for the current policy at each policy optimization step. However, [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) shows that PS can have a positive constant distance to PO.

Two extensions of the basic performative reinforcement learning problem [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) have been proposed and studied. [\(Rank et al.,](#page-9-3) [2024\)](#page-9-3) extends to the setting where the environmental dynamics gradually adjust to the currently deployed policy, and proposes a mixed delayed repeated retraining algorithm with accelerated convergence to a PS policy. [\(Mandal and Radanovic,](#page-8-6) [2024\)](#page-8-6) extends [\(Mandal](#page-8-5) [et al.,](#page-8-5) [2023\)](#page-8-5) from tabular setting to linear Markov decision processes with large number of states, and also obtains the convergence rate of the repeated retraining algorithm to a PS policy.

In sum, all these existing performative reinforcement learning works pursue a suboptimal PS policy. Therefore, we want to ask the following fundamental research question.

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109 *Q: Can we design an algorithm that converges to the desired performatively optimal (PO) policy?*

#### 1.1. Our Contributions

We will answer affirmatively to the research question above in the following steps. Each step yields a novel contribution.

• We study an entropy regularized performative reinforcement learning problem, compatible with the basic performative reinforcement learning problem in [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5). We prove that the objective function satisfies a certain gradient dominance condition, which implies that an approximate stationary point (not the suboptimal PS) is the desired approximate PO policy, under a mild regularizer dominance condition similar to that used by [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank et al.,](#page-9-3) [2024;](#page-9-3) [Mandal and Radanovic,](#page-8-6) [2024\)](#page-8-6) to ensure convergence to a suboptimal PS policy. The proof adopts novel techniques such as recursion for pπ-related error term and frequent switch among various necessary and sufficient conditions of smoothness and strong concavity like properties for various variables (see Section [3.2\)](#page-3-0).

• We obtain a policy lower bound as a decreasing function of a stationary measure. This bound not only implies the unbounded *performative policy gradient* (a challenge to obtain a stationary policy and thus PO), but also inspires us to find a stationary policy in the policy subspace Π<sup>∆</sup> with a constant policy lower bound ∆ > 0 where we prove the objective function to be Lipschitz continuous and Lipschitz smooth (a solution to this challenge). The policy lower bound is obtained using a novel technique which simplifies a complicated inequality of the minimum policy value π[amin(s)|s] in two cases (see Section [3.3\)](#page-4-0).

• We construct a zeroth-order estimation of the *performative policy gradient* and obtains its estimation error. This is more challenging than the existing zero-th order estimation methods since our objective function is only well-defined on the policy space, a compact subset of a linear subspace of the Euclidean space R |S||A|. To solve this puzzle, we adjust a two-point estimation to the linear subspace L<sup>0</sup> of policy difference, and simplify the estimation error analysis by mapping policies onto the Euclidean space R |S|(|A|−1) via orthogonal transformation (see Section [4.1\)](#page-5-0).

• We propose a zeroth-order performative policy gradient (0-PPG) algorithm (see Algorithm [1\)](#page-7-0) by combining the *performative policy gradient* estimation above with the Frank-Wolfe algorithm. Then we obtain a polynomial computation complexity of our 0-PPG algorithm to converge to a stationary policy, which is also the desired PO policy under the regularizer dominance condition above. The convergence analysis uses a policy averaging technique to show that an approximate stationary policy on Π<sup>∆</sup> is also approximately stationary on the whole policy space Π (see Section [4.2\)](#page-7-1).

Finally, we briefly show that the results above, including gradient dominance, Lipschitz properties and the finite-time convergence of 0-PPG algorithm to the desired PO, can be adjusted to the performative reinforcement learning problem with the quadratic regularizer used by [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank et al.,](#page-9-3) [2024\)](#page-9-3) (see Appendix [K\)](#page-36-0).

# 2. Preliminary: Performative Reinforcement Learning

#### 2.1. Problem Formulation

Performative reinforcement learning is characterized by a Markov decision process (MDP) M<sup>π</sup> = (S, A, pπ, rπ, ρ) that depends on a certain policy π. Here, S and A denote the finite state space with cardinality |S| and finite action space with cardinality |A| respectively. The policy π ∈ [0, 1]|S||A| , with entries π(a|s) for any state s ∈ S and action a ∈ A, lies in the following policy space, such that π(·|s) for any state s can be seen as a distribution over A.

$$\Pi \stackrel{\text{def}}{=} \left\{ \pi \in [0, 1]^{|\mathcal{S}||\mathcal{A}|} : \sum_{a \in \mathcal{A}} \pi(a|s) = 1, \forall s \in \mathcal{S} \right\}.$$

The transition kernel <sup>p</sup><sup>π</sup> ∈ [0, 1]|S|<sup>2</sup> |A| dependent on policy π ∈ Π, with entries pπ(s ′ |s, a) for any s, s′ ∈ S and a ∈ A, lies in the following transition kernel space such that pπ(·|s, a) can be seen as a state distribution on S.

$$\mathcal{P} \stackrel{\text{def}}{=} \left\{ p \in [0, 1]^{|\mathcal{S}|^2 \setminus \mathcal{A}|} : \sum_{s \in \mathcal{S}} p(s'|s, a) = 1, \forall s \in \mathcal{S}, a \in \mathcal{A} \right\}.$$

<sup>r</sup><sup>π</sup> ∈ R def = [0, 1]|S||A| is the reward function with entries rπ(s, a) ∈ [0, 1] for any s ∈ S and a ∈ A. ρ ∈ [0, 1]|S| is the initial state distribution such that P <sup>s</sup>∈S ρ(s) = 1. Note that we consider pπ, rπ, ρ, π as Euclidean vectors, so that we can conveniently define their Euclidean norm. For example, we define ∥pπ∥<sup>q</sup> = -P s,a,s′ |pπ(s ′ |s, a)| q <sup>1</sup>/q for any q > 1 and ∥pπ∥<sup>∞</sup> = maxs,a,s′ |pπ(s ′ |s, a)|. Such norms can be similarly defined over rπ, ρ, π by summing or maximizing over all the entries. Specifically, denote ∥ · ∥ = ∥ · ∥<sup>2</sup> by convention.

When an agent applies its policy π ∈ Π to MDP Mπ′ = (S, A, pπ′ , rπ′ , ρ), the initial environmental state s<sup>0</sup> ∈ S is generated from the distribution ρ. Then at each time t = 0, 1, 2, . . ., the agent takes a random action a<sup>t</sup> ∼ π(·|st) based on the current state s<sup>t</sup> ∈ S, the environment transitions to the next state st+1 ∼ pπ′ (·|st, at) and provides reward r<sup>t</sup> = rπ′ (st, at) ∈ [0, 1] to the agent. The value of applying policy π to Mπ′ can be characterized by the following *value function*.

$$V_{\lambda, \pi'} \stackrel{\text{def}}{=} \mathbb{E}_{\pi, p_{\pi'}, \rho} \left[ \sum_{t=0}^{\infty} \gamma^t r_{\pi'}(s_t, a_t) \right] - \lambda \mathcal{H}_{\pi'}(\pi). \quad (1)$$

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

164

Here, <sup>E</sup>π,pπ′ ,ρ is the expectation under policy π, transition kernel pπ′ and initial state distribution ρ. γ ∈ (0, 1) is the discount factor. Hπ′ (π) is a regularizer with coefficient λ ≥ 0 to ensure or accelerate algorithm convergence. Existing works use the quadratic regularizers such as Hπ′ (π) = <sup>1</sup> 2 ∥dπ,pπ′ ∥ [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank](#page-9-3) [et al.,](#page-9-3) [2024\)](#page-9-3) and Hπ′ (π) = <sup>1</sup> 2 ∥Φ <sup>⊤</sup>dπ,pπ′ ∥ 2 [\(Mandal and](#page-8-6) [Radanovic,](#page-8-6) [2024\)](#page-8-6) with a feature matrix Φ, where the occupancy measure dπ,p ∈ [0, 1]|S||A| for any policy π and transition kernel p is defined as the following distribution on S × A.

$$d_{\pi, p}(s, a) \stackrel{\text{def}}{=} (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}_{\pi, p, \rho}\{s_t = s, a_t = a\}. \quad (2)$$

Then the state occupancy measure defined as dπ,p(s) def <sup>P</sup> <sup>=</sup> a dπ,p(s, a) satisfies the following well-known Bellman equation for any state s ′ ∈ S.

$$d_{\pi, p}(s') = (1-\gamma)\rho(s') + \gamma \sum_{s, a} d_{\pi, p}(s) \pi(a|s) p(s'|s, a). \quad (3)$$

The ultimate goal of performative reinforcement learning is to find the *performatively optimal (PO)* policy π that maximizes the *performative value function* V π λ,π (with π ′ = π in Eq. [\(1\)](#page-1-0)), as formally defined below.

Definition 1 (Ultimate Goal: PO). *For any* ϵ ≥ 0*, a policy* π ∈ Π *is defined as* ϵ*-performatively optimal (*ϵ*-PO) if* maxπ′∈<sup>Π</sup> V π ′ λ,π′ − V π λ,π ≤ ϵ*. Specifically, we call a 0-PO policy as a PO policy.*

Conventional reinforcement learning can be seen as a special case of performative reinforcement learning with fixed environmental dynamics, namely, constant transition kernel p<sup>π</sup> ≡ p and constant reward function r<sup>π</sup> ≡ r. However, this may fail on applications with policy-dependent environmental dynamics, such as recommender system and autonomous driving [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) as explained in Section [1.](#page-0-0)

### 2.2. Performatively Stable (PS) Policy in Existing Works

Achieving an ϵ-PO policy (defined by Definition [1\)](#page-2-0) is challenging, due to the policy-dependent environmental dynamics p<sup>π</sup> and rπ. To alleviate the challenge, all the existing works [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank et al.,](#page-9-3) [2024;](#page-9-3) [Mandal and](#page-8-6) [Radanovic,](#page-8-6) [2024\)](#page-8-6) aim at a *performatively stable (PS)* policy πPS defined as follows, as an approximation of a *PO policy*.

$$\pi_{\text{PS}} \in \arg \max_{\pi \in \Pi} V_{\lambda, \pi_{\text{PS}}}^{\pi}. \quad (4)$$

In other words, a PS policy πPS has the optimal value on the fixed environment M<sup>π</sup>PS . However, [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) shows that a PS policy can be suboptimal, so these existing algorithms cannot converge to a PO policy. Nevertheless, we will briefly introduce these algorithms, to later

partially inspire and compare with our method for achieving a PO policy. Note that an occupancy measure d (a distribution on S × A) corresponds to the policy π <sup>d</sup> defined as π d (a|s) = <sup>d</sup>(s,a) d(s) (π d (a|s) = 1/|A| if d(s) = 0), where d(s)=P <sup>a</sup>′∈A <sup>d</sup>(s, a′ ). Hence, [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank](#page-9-3) [et al.,](#page-9-3) [2024;](#page-9-3) [Mandal and Radanovic,](#page-8-6) [2024\)](#page-8-6) transform the policy optimization problem [\(4\)](#page-2-1) into a problem of solving d. The basic performative reinforcement learning [\(Man](#page-8-5)[dal et al.,](#page-8-5) [2023\)](#page-8-5) considers the following dual optimization problem of d in the environment pd′ = pπd′ , rd′ = rπd′ corresponding to another occupancy measure d ′ .

$$\begin{cases} \max_{d: \text{distribution on } \mathcal{S} \times \mathcal{A} \sum_{s,a} d(s, a) r_{d'}(s, a) - \frac{\lambda}{2} \|d\|^2 \\ \text{s.t. } \sum_a d(s, a) = \rho(s) + \gamma \sum_{s',a} d(s', a) p_{d'}(s|s', a) \end{cases} \quad (5)$$

The objective function above corresponds to the value function V π λ,π′ defined in Eq. [\(1\)](#page-1-0) with quadratic regularizer Hπ′ (π)= <sup>1</sup> 2 ∥dπ,pπ′ ∥ 2 . The equality constraint above comes from the Bellman equation [\(3\)](#page-2-2). Denote ϕ(d ′ ) as the optimal solution to the problem [\(5\)](#page-2-3) above. Then the target becomes a performatively stable occupancy measure dPS defined as a fixed point dPS = ϕ(dPS), which corresponds to a PS policy πPS = π <sup>d</sup>PS . Suppose the transition kernel and reward function are sensitive with parameters ϵ ′ p , ϵ′ <sup>r</sup> > 0 respectively, that is, for any occupancy measures d, d′ .

$$\|p_{d'} - p_d\| \leq \epsilon'_p \|d' - d\|, \quad \|r_{d'} - r_d\| \leq \epsilon'_r \|d' - d\|. \quad (6)$$

It has been proved by [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5) that ϕ is a contraction mapping under a regularizer dominance condition that λ > O(ϵ ′ <sup>p</sup> + ϵ ′ r ). In this case, any repeated retraining method characterized by dt+1 ≈ ϕ(dt) with sufficient precision can converge to the PS policy.

Similarly, [\(Rank et al.,](#page-9-3) [2024;](#page-9-3) [Mandal and Radanovic,](#page-8-6) [2024\)](#page-8-6) also apply repeated retraining to optimization problems of occupancy measure, which converges to a PS policy for extensions of the basic performative reinforcement learning [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5). Next, we will propose our significantly different strategies to achieve the desired PO policy.

# 3. Entropy Regularized Performative Reinforcement Learning

In this section, we obtain critical properties of an entropy regularized performative reinforcement learning problem for achieving the desired PO policy.

#### 3.1. Negative Entropy Regularizer

To achieve the PO policy, one might attempt to solve the problem (Pd), adjusted from the dual problem [\(5\)](#page-2-3) above with fixed d ′ replaced by the decision variable d. The solution dPO will yield the PO policy π <sup>d</sup>PO . However, such

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

218

replacement will make the convex quadratic optimization problem [\(5\)](#page-2-3) much more complicated, due to the unknown and possibly complicated functions p<sup>d</sup> and rd. Therefore, we will instead focus on the primal problem max<sup>π</sup> V π λ,π.

We consider the following negative entropy regularizer of the policy π, which is widely used in reinforcement learning to encourage environment exploration and accelerate convergence [\(Mnih et al.,](#page-8-7) [2016;](#page-8-7) [Mankowitz et al.,](#page-8-8) [2019;](#page-8-8) [Cen](#page-8-9) [et al.,](#page-8-9) [2022;](#page-8-9) [Chen and Huang,](#page-8-10) [2024\)](#page-8-10).

$$\mathcal{H}_{\pi'}(\pi) = \mathbb{E}_{\pi, p_{\pi'}, \rho} \left[ \sum_{t=0}^{\infty} \gamma^t \log \pi(a_t | s_t) \right]. \quad (7)$$

In addition, this negative entropy regularizer can be seen as a strongly convex function of the occupancy measure dπ,pπ′ (proved in Appendix [B\)](#page-19-0), which is critical to develop algorithms convergent to a PO (see Theorem [1](#page-3-1) later) or PS policy [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5). For optimization problem on a probability simplex variable (policy π or occupancy measure d), negative entropy regularizer is more natural and yields faster theoretical convergence than the quadratic regularizers used in the existing performative reinforcment learning works [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank et al.,](#page-9-3) [2024\)](#page-9-3) (see pages 43-45 of [\(Chen,](#page-8-11) [2020\)](#page-8-11) for explanation).

Therefore, we will mainly focus on the following entropyregularized value function, which is obtained by substituting the negative entropy regularizer [\(7\)](#page-3-2) into the general value function [\(1\)](#page-1-0).

$$V_{\lambda, \pi'}^{\pi} \stackrel{\text{def}}{=} \mathbb{E}_{\pi, p_{\pi'}, \rho} \left[ \sum_{t=0}^{\infty} \gamma^t [r_{\pi'}(s_t, a_t) - \lambda \log \pi(a_t | s_t)] \right]. \quad (8)$$

Specifically, we will study the critical properties of the entropy-regularized value function [\(8\)](#page-3-3) (Section [4\)](#page-5-1) to develop algorithm that converges to PO (Sections [4.1-](#page-5-0)[4.2\)](#page-7-1). Then we will briefly discuss about how to adjust these results to the existing quadratic regularizers (Appendix [K\)](#page-36-0).

We make the following standard assumptions to study the properties of the entropy-regularized value function [\(8\)](#page-3-3).

Assumption 1 (Sensitivity). *There exist constants* ϵp, ϵ<sup>r</sup> > 0 *such that for any* π, π′ ∈ Π*,*

$$\|p_{\pi'} - p_{\pi}\| \leq \epsilon_p \|\pi' - \pi\|, \quad \|r_{\pi'} - r_{\pi}\| \leq \epsilon_r \|\pi' - \pi\| \quad (9)$$

Assumption 2 (Smoothness). p<sup>π</sup> *and* r<sup>π</sup> *are Lipschitz smooth with modulus* Sp, S<sup>r</sup> > 0 *respectively, that is, for any* π ∈ Π*,* s, s′ ∈ S*,* a ∈ A*, we have*

$$\|\nabla_{\pi} p_{\pi'}(s'|s, a) - \nabla_{\pi} p_{\pi}(s'|s, a)\| \leq S_p \|\pi' - \pi\|, \quad (10)$$

$$\|\nabla_{\pi} r_{\pi'}(s, a) - \nabla_{\pi} r_{\pi}(s, a)\| \leq S_r \|\pi' - \pi\|. \quad (11)$$

Assumption 3. *There exists a constant* D > 0 *such that* infπ∈Π,p∈P,s∈S dπ,p(s) ≥ D*.*

Assumptions [1](#page-3-4)[-2](#page-3-5) ensure that the environmental dynamics p<sup>π</sup> and r<sup>π</sup> adjust continuously and smoothly to policy π, and thus the *performative value function* V π λ,π is differentiable with *performative policy gradient* ∇πV π λ,π. Similar versions of Assumption [1](#page-3-4) on environmental sensitivity have been used in the performative reinforcement learning literature (e.g. Eq. [\(6\)](#page-2-4) in [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5)). Assumption [3](#page-3-6) has been used [\(Zhang et al.,](#page-9-4) [2021\)](#page-9-4) or implied by stronger assumptions [\(Wei et al.,](#page-9-5) [2021;](#page-9-5) [Chen et al.,](#page-8-12) [2022;](#page-8-12) [Agarwal](#page-8-13) [et al.,](#page-8-13) [2021;](#page-8-13) [Leonardos et al.,](#page-8-14) [2022;](#page-8-14) [Wang et al.,](#page-9-6) [2023;](#page-9-6) [Chen](#page-8-10) [and Huang,](#page-8-10) [2024;](#page-8-10) [Bhandari and Russo,](#page-8-15) [2024\)](#page-8-15) in conventional reinforcement learning (see Appendix [C](#page-20-0) for the proof), which guarantees that each state is visited sufficiently often.

#### 3.2. Gradient Dominance

For the nonconvex policy optimization problem maxπ∈<sup>Π</sup> V π λ,π with the entropy regularized value function [\(8\)](#page-3-3) on the convex policy space Π, it is natural to consider its approximate stationary solution as defined below.

Definition 2 (Stationary Policy). *For any* ϵ ≥ 0*, a policy* π ∈ Π *is* ϵ*-stationary if* maxπ˜∈<sup>Π</sup> ∇πV π λ,π, π˜ − π ≤ ϵ*. We call a 0-stationary policy as a stationary policy.*

Note that for a policy to be the desired PO, it is necessary to be stationary, while the PS policy targeted by existing works is neither necessary nor sufficient. Furthermore, we will show that stationary policy can also be a sufficient condition of the desired PO under mild conditions. As a preliminary step, we show the important gradient dominance property of the objective function as follows.

Theorem 1 (Gradient Dominance). *Under Assumptions [1-](#page-3-4) [3,](#page-3-6) the entropy regularized value function* [\(8\)](#page-3-3) *satisfies the following gradient dominance property for any* π0, π<sup>1</sup> ∈ Π*.*

$$V_{\lambda, \pi_1}^{\pi_1} \leq V_{\lambda, \pi_0}^{\pi_0} + D^{-1} \max_{\pi \in \Pi} \langle \nabla_{\pi_0} V_{\lambda, \pi_0}^{\pi_0}, \pi - \pi_0 \rangle - \frac{\mu}{2} \|\pi_1 - \pi_0\|^2, \quad (12)$$

*where the constant* µ ∈ R *is defined as follows.*

$$\mu = \frac{D\lambda}{1-\gamma} - \frac{6\gamma|\mathcal{S}|(1+\lambda|\mathcal{A}|)}{D(1-\gamma)^3} [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma\epsilon_p\sqrt{|\mathcal{S}|}) + S_p(1-\gamma)] - \frac{S_r(1-\gamma) + 4\epsilon_r(\sqrt{|\mathcal{A}|} + \gamma\epsilon_p\sqrt{|\mathcal{S}|})}{D^2(1-\gamma)^2}. \quad (13)$$

Remark: With sufficiently large regularizer strength λ and small environmental shift strength ϵp, ϵr, Sp, S<sup>r</sup> (i.e., when the regularizer dominates the environmental shift), we have µ ≥ 0, which implies the gradient dominance form (Eq. [\(12\)](#page-3-7) with µ = 0) that holds for conventional unregularized reinforcement learning (see Lemma 4 of [\(Agarwal](#page-8-13)

226

228

231

234

236

238

254

256

258

260

264

266

268

271

[et al.,](#page-8-13) [2021\)](#page-8-13)). In this case, stationary policy becomes a sufficient condition of the desired PO, as shown in the following Corollary [1.](#page-4-1) Note that the existing performative reinforcement learning works [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank et al.,](#page-9-3) [2024;](#page-9-3) [Perdomo et al.,](#page-9-7) [2020\)](#page-9-7) also require a regularizer dominance condition similar to our µ ≥ 0 (e.g. λ > O(ϵ ′ <sup>p</sup> + ϵ ′ r ) in [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5)) to ensure convergence to a PS policy.

Corollary 1. *Under Assumptions [1](#page-3-4)[-3,](#page-3-6) if* µ ≥ 0 *for* µ *defined in Eq.* [\(13\)](#page-3-8)*, then any* Dϵ*-stationary policy is also the desired* ϵ*-PO policy. Furthermore, if* µ > 0*, the PO policy is unique.*

Intuition and Novelty for Proving Theorem [1:](#page-3-1) Define the following more refined value function

$$J_\lambda(\pi, \pi', p, r) \stackrel{\text{def}}{=} \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t [r(s_t, a_t) - \lambda \log \pi'(a_t | s_t)] \middle| s_0 \sim \rho \right]. \quad (14)$$

To get the intuition, we consider the following three cases from the simplest conventional reinforcement learning to the hardest performative reinforcement learning.

(Case I): For conventional reinforcement learning with fixed dynamics p<sup>π</sup> ≡ p and r<sup>π</sup> ≡ r, denote d<sup>α</sup> = αd<sup>π</sup>1,p + [\(](#page-8-13)1 − α)d<sup>π</sup>0,p (α ∈ [0, 1]). Based on the Bellman equation [\(3\)](#page-2-2), d<sup>α</sup> = d<sup>π</sup>α,p is the occupancy measure of the policy <sup>π</sup>α(a|s) = <sup>d</sup>α(s,a) dα(s) . Therefore, V π<sup>α</sup> λ,π<sup>α</sup> can be rewritten as Jλ(πα, πα, p, r) = P s,a dα(s, a)[r(s, a) − λ log πα(a|s)], which has the following strong concavity like property by Pinsker's inequality (see Eq. [\(91\)](#page-24-0) for detail).

$$\begin{aligned} & J_\lambda(\pi_\alpha, \pi_\alpha, p, r) - \alpha J_\lambda(\pi_1, \pi_1, p, r) \\ & - (1 - \alpha) J_\lambda(\pi_0, \pi_0, p, r) \\ & = \frac{1}{1 - \gamma} \sum_s [\alpha d_1(s) \text{KL}[\pi_1(\cdot | s) \| \pi_\alpha(a | s)] \\ & + (1 - \alpha) d_0(s) \text{KL}[\pi_0(\cdot | s) \| \pi_\alpha(a | s)]] \\ & \geq \frac{D\lambda\alpha(1 - \alpha)}{2(1 - \gamma)} \|\pi_1 - \pi_0\|^2. \end{aligned} \quad (15)$$

(Case II): Consider a harder case with varying p<sup>π</sup> and constant reward r<sup>π</sup> ≡ r. Similarly, we denote d<sup>α</sup> = αd<sup>π</sup>1,pπ<sup>1</sup> + (1 − α)d<sup>π</sup>0,pπ<sup>0</sup> and <sup>π</sup>α(a|s) = <sup>d</sup>α(s,a) dα(s) . The varying p<sup>π</sup> brings a major challenge that d<sup>α</sup> = d<sup>π</sup>α,pπα required by Case I no longer holds. To solve this challenge, we prove that the error term eα(s) = d<sup>π</sup>α,p<sup>α</sup> (s) − dα(s) of interest satisfies the following novel recursion (see Eq. [\(89\)](#page-23-0) for the derivation based on the Bellman equation [\(3\)](#page-2-2)).

$$e_\alpha(s') = \gamma \sum_{s,a} [e_\alpha(s)\pi_\alpha(a|s)p_{\pi_\alpha}(s'|s,a) + h_\alpha(s,a,s')],$$

dα(s, a)p<sup>π</sup><sup>α</sup> (s ′ |s, a) is a Lipschitz smooth function of α with Lipschitz constant ℓdp(s, a) defined by Eq. [\(87\)](#page-23-1), we have |hα(s, a, s′ )| ≤ <sup>α</sup>(1−α) 2 ℓdp(s, a), which can be substituted into the recursion above and yields the following novel error bound (see Eq. [\(90\)](#page-24-1) for detail).

$$\sum_s |e_\alpha(s)| \leq \alpha(1-\alpha)\mathcal{O}(\epsilon_p + S_p) \|\pi_1 - \pi_0\|^2,$$

which implies the desired strong concavity like property as follows.

$$\begin{aligned} J_\lambda(\pi_\alpha, \pi_\alpha, p_\alpha, r) - \alpha J_\lambda(\pi_1, \pi_1, p_1, r) \\ - (1 - \alpha) J_\lambda(\pi_0, \pi_0, p_0, r) \\ \geq \text{Eq. (15)} - \alpha(1 - \alpha)(1 + \lambda)\mathcal{O}(\epsilon_p + S_p) \|\pi_1 - \pi_0\|^2 \\ \geq \frac{\alpha(1 - \alpha)\mu_1}{2} \|\pi_1 - \pi_0\|^2 \end{aligned} \quad (16)$$

where µ<sup>1</sup> = Dλ 2(1−γ)−(1+λ)O(ϵp+Sp) defined by Eq. [\(92\)](#page-25-0) equals µ defined by Eq. [\(13\)](#page-3-8) when ϵ<sup>r</sup> = S<sup>r</sup> = 0.

(Case III): Now we consider performative reinforcement learning with varying p<sup>π</sup> and rπ. The policy π<sup>α</sup> and its occupancy measure d<sup>α</sup> are the same as in Case II above. Then the function w(α) = αJλ(π1, π1, p1, rα) + (1 − α)Jλ(π0, π0, p0, rα) can be proved Lipschitz smooth with parameter µ<sup>2</sup> = O(S<sup>r</sup> + ϵr) defined by Eq. [\(94\)](#page-25-1), so using r = r<sup>α</sup> in Eq. [\(16\)](#page-4-3) we obtain the following strong concavity like property.

$$\begin{aligned} & J_\lambda(\pi_\alpha, \pi_\alpha, p_\alpha, r_\alpha) - \alpha J_\lambda(\pi_1, \pi_1, p_1, r_1) \\ & \quad - (1 - \alpha) J_\lambda(\pi_0, \pi_0, p_0, r_0) \\ & \geq \frac{\alpha(1-\alpha)\mu_1}{2} \|\pi_1 - \pi_0\|^2 + w(\alpha) - \alpha w(1) - (1-\alpha)w(0) \\ & \geq \frac{\alpha(1-\alpha)(\mu_1 - \mu_2)}{2} \|\pi_1 - \pi_0\|^2. \end{aligned}$$

Rearranging the inequality above, we obtain the following inequality of V π<sup>α</sup> λ,π<sup>α</sup> = Jλ(πα, πα, pα, rα).

$$\frac{V_{\lambda, \pi_\alpha}^{\pi_\alpha} - V_{\lambda, \pi_0}^{\pi_0}}{\alpha} \geq V_{\lambda, \pi_1}^{\pi_1} - V_{\lambda, \pi_0}^{\pi_0} + \frac{\mu(1-\alpha)}{2} \|\pi_1 - \pi_0\|^2,$$

where µ = µ<sup>1</sup> − µ<sup>2</sup> is exactly defined by Eq. [\(13\)](#page-3-8). Letting α → +0 above, we have

$$V_{\lambda, \pi_1}^{\pi_1} \leq V_{\lambda, \pi_0}^{\pi_0} + \left[ \frac{d}{d\alpha} V_{\lambda, \pi_\alpha}^{\pi_\alpha} \right]_{\alpha=0} - \frac{\mu}{2} \|\pi_1 - \pi_0\|^2.$$

Using the chain rule, we can find a policy π ∗ 0 such that d dα V π<sup>α</sup> λ,π<sup>α</sup> <sup>α</sup>=0 ≤ <sup>D</sup> ∇<sup>π</sup><sup>0</sup> V π<sup>0</sup> λ,π<sup>0</sup> , π<sup>∗</sup> <sup>0</sup> − π<sup>0</sup> , which along with the bound above proves the gradient dominance property [\(12\)](#page-3-7).

# 3.3. Policy Lower Bound and Lipschitz Properties

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

the unbounded *performative policy gradient* ∇πV π λ,π on Π. Specifically, we will show that as π(a|s) → 0 for any state s and action a, ∥∇πV π λ,π∥ → +∞. To tackle this challenge, we prove the following policy lower bound.

Theorem 2. *If Assumptions [1](#page-3-4) and [3](#page-3-6) hold, and* pπ*,* r<sup>π</sup> *are differentiable functions of* π*, then the following policy lower bound holds for any* π ∈ Π*,* s ∈ S*,* a ∈ A*.*

$$\pi(a|s) \geq \pi_{\min} \exp \left[ -\frac{2|\mathcal{A}|}{\lambda} (1-\gamma) \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \pi' - \pi \rangle \right]. \quad (17)$$

*Here, we define the following constant* πmin *and policy* π ′ *.*

$$\pi_{\min} \stackrel{\text{def}}{=} \frac{1}{2|\mathcal{A}|^{1/(1-\gamma)}} \exp \left\{ -\frac{1}{\lambda(1-\gamma)} - \frac{2|\mathcal{A}|\sqrt{2|\mathcal{S}|}}{\lambda} \left[ \frac{\epsilon_p \sqrt{|\mathcal{S}|}(1 + \lambda \log |\mathcal{A}|)}{1-\gamma} + \epsilon_r \right] \right\}, \quad (18)$$

$$\pi'(a|s) = \begin{cases} \pi[a_{\min}(s)|s], & a = a_{\max}(s) \\ \pi[a_{\max}(s)|s], & a = a_{\min}(s), \\ \pi(a|s), & \text{Otherwise} \end{cases} \quad (19)$$

*where* amax(s) ∈ arg maxaπ(a|s) *and* amin(s) ∈ arg minaπ(a|s)*.*

Implications of Theorem [2:](#page-5-2) First, as π(a|s) → 0, we have ⟨∇πV π λ,π, π′ −π⟩ → <sup>+</sup>∞, so ∥∇π<sup>V</sup> π λ,π∥ → +∞ as aforementioned. Second, any stationary policy π satisfies ⟨∇πV π λ,π, π′ − <sup>π</sup>⟩ ≤ <sup>0</sup>, so <sup>π</sup>(a|s) ≥ <sup>π</sup>min. Therefore, we can search ϵ-stationary policy on the convex and compact policy subspace Π<sup>∆</sup> def <sup>=</sup> {<sup>π</sup> ∈ Π : <sup>π</sup>(a|s) ≥ <sup>∆</sup>} with lower bound ∆ ∈ (0, πmin].

Intuition and Novelty for Proving Theorem [2:](#page-5-2) As a preliminary step, consider a conventional reinforcement learning problem with fixed environmental dynamics p<sup>π</sup> ≡ p and r<sup>π</sup> ≡ r. In this case, ∇πV π λ,π has analytical form (see Eq. [\(98\)](#page-27-0)) based on policy gradient theorem, so by direct computation we obtain the following bound (see Eq. [\(99\)](#page-27-1) for detail)

$$\langle \nabla_{\pi} \pi_{\lambda', \pi}' - \pi_{\lambda'} \rangle \geq \frac{1}{1-\gamma} \max_s \left\{ (\pi[a_{\max}(s)|s] - \pi[a_{\min}(s)|s]) \left[ \lambda \log \frac{\pi[a_{\max}(s)|s]}{\pi[a_{\min}(s)|s]} - 1 - \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1-\gamma} \right] \right\}.$$

To directly solve the inequality above of π[amin(s)|s] is not easy. To simplify this inequality, we consider two cases, either π[amin(s)|s] ≥ 1 2 π[amax(s)|s] ≥ <sup>2</sup>|A| or π[amin(s)|s] < 1 2 π[amax(s)|s]. In the second case, we can replace π[amax(s)|s] and π[amax(s)|s] − π[amin(s)|s] above with their lower bounds <sup>1</sup> |A| and <sup>1</sup> <sup>2</sup>|A| respectively. Then it becomes straightforward to obtain the policy lower bound.

$$\pi[a_{\min}(s)|s] \geq \pi'_{\min} \exp \left[ -\frac{2|\mathcal{A}|}{\lambda} (1-\gamma) \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \pi' - \pi \rangle \right],$$

where π ′ min is defined by Eq. [\(18\)](#page-5-3) with ϵ<sup>p</sup> = ϵ<sup>r</sup> = 0.

Then by extending conventional reinforcement learning to performative reinforcement learning, ∇πV π λ,π is perturbed by a magnitude of at most <sup>ϵ</sup><sup>p</sup> √ |S|(1+λ log |A|) <sup>1</sup>−<sup>γ</sup> + ϵ<sup>r</sup> (see Eq. [\(102\)](#page-28-0) for detail) based on the chain rule. This perturbation bound along with ∥π ′ − π∥ ≤ p 2|S| yields the second line of Eq. [\(18\)](#page-5-3) and proves Theorem [2.](#page-5-2)

Lipschitz Properties: Furthermore, in the policy subspace Π∆, the *performative value function* V π λ,π is actually Lipschitz continuous and Lipschitz smooth as shown below, which facilitates finding an ϵ-stationary policy in Π∆.

Theorem 3. *Under Assumptions [1](#page-3-4)[-2,](#page-3-5)* V π λ,π *satisfies the following Lipschitz propreties for any* ∆ > 0 *and* π, π′ ∈ Π∆*.*

$$|V_{\lambda,\pi'}^{\pi'} - V_{\lambda,\pi}^{\pi}| \leq \frac{L_{\lambda}}{\Delta} \|\pi' - \pi\|, \quad (20)$$

$$\|\nabla_{\pi'} V_{\lambda, \pi'}^{\pi'} - \nabla_{\pi} V_{\lambda, \pi}^{\pi}\| \leq \frac{\ell_{\lambda}}{\Delta} \|\pi' - \pi\|. \quad (21)$$

*where*

$$L_\lambda \stackrel{\text{def}}{=} \frac{\sqrt{|\mathcal{A}|(2-\gamma+\gamma\lambda \log|\mathcal{A}|)+\epsilon_p\sqrt{|\mathcal{S}|(1+\lambda \log|\mathcal{A}|)}}{(1-\gamma)^2} + \frac{\epsilon_r}{1-\gamma} \quad (22)$$

$$\begin{aligned} \ell_\lambda \stackrel{\text{def}}{=} & \frac{3|\mathcal{A}|(1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \frac{\epsilon_p \sqrt{|\mathcal{S}|} |\mathcal{A}|(5 + 6\lambda \log |\mathcal{A}|)}{(1 - \gamma)^3} \\ & + \frac{\epsilon_r [\sqrt{|\mathcal{A}|}(1 - \gamma) + \sqrt{|\mathcal{S}|}(\gamma + 2\epsilon_p)]}{|\mathcal{A}|(1 - \gamma)^2} \\ & + \frac{S_p \sqrt{|\mathcal{S}|}(1 + \lambda \log |\mathcal{A}|) + S_r(1 - \gamma)}{|\mathcal{A}|(1 - \gamma)^2}. \end{aligned} \quad (23)$$

# 4. Zeroth-Order Performative Policy Gradient (0-PPG) Algorithm

#### 4.1. Performative Policy Gradient Estimation

In Section [3,](#page-2-5) we have obtained important properties of the entropy regularized *performative value function* V π λ,π (defined by Eq. [\(8\)](#page-3-3)), which indicates that it suffices to find an ϵstationary policy in the subspace Π<sup>∆</sup> for ∆ ∈ (0, πmin]. To achieve this goal, an accurate estimation of the *performative policy gradient* ∇πV π λ,π is important, which has two challenges. First, unlike conventional reinforcement learning where policy gradient has analytical form, such analytical form does not exist in performative reinforcement learning due to the arbitrary forms of p<sup>π</sup> and rπ. Second, in practice, we cannot access the values of pπ(s ′ |s, a) and rπ(s, a) but can only obtain stochastic samples from them [\(Mandal et al.,](#page-8-5) [2023\)](#page-8-5).

Despite these challenges in estimating ∇πV π λ,π, note that V π λ,π for any policy π can be evaluated, since it is actually

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

the policy evaluation problem in conventional reinforcement learning under fixed environment p<sup>π</sup> and r<sup>π</sup> (for fixed π). Furthermore, for any ϵ<sup>V</sup> > 0 and η ∈ (0, 1), many existing policy evaluation algorithms such as temporal difference [\(Bhandari et al.,](#page-8-16) [2018;](#page-8-16) [Li et al.,](#page-8-17) [2023;](#page-8-17) [Samsonov et al.,](#page-9-8) [2023\)](#page-9-8), can obtain Vˆ <sup>π</sup> λ,π ≈ V π λ,π with the following ϵ<sup>V</sup> error with probability at least 1 − η.

$$|\hat{V}_{\lambda,\pi}^\pi - V_{\lambda,\pi}^\pi| \leq \epsilon_V. \quad (24)$$

As a result, we will consider a zeroth-order estimation of ∇πV π λ,π using policy evaluation. However, this has another challenge that V π λ,π is only well-defined on π ∈ Π, so we cannot directly apply the existing zeroth-order estimation methods [\(Agarwal et al.,](#page-8-18) [2010;](#page-8-18) [Shamir,](#page-9-9) [2017;](#page-9-9) [Malik](#page-8-19) [et al.,](#page-8-19) [2020\)](#page-8-19) which require the objective function to be welldefined on a sphere. Fortunately, for any π, π′ ∈ Π, the policy difference π ′ − π lies in the following linear subspace of dimensionality |S|(|A| − 1).

$$\mathcal{L}_0 \stackrel{\text{def}}{=} \left\{ u \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|} : \sum_a u(a|s) = 0, \forall s \in \mathcal{S} \right\}. \quad (25)$$

Therefore, inspired by the popular two-point zeroth-order estimations, we obtain the following estimation of ∇πV π λ,π.

$$\hat{g}_{\lambda, \delta}(\pi) = \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N (\hat{V}_{\lambda, \pi + \delta u_i}^{\pi + \delta u_i} - \hat{V}_{\lambda, \pi - \delta u_i}^{\pi - \delta u_i}) u_i, \quad (26)$$

where {ui} N <sup>i</sup>=1 are i.i.d. samples from the uniform distribution on U<sup>1</sup> ∩ L<sup>0</sup> with

$$U_1 \stackrel{\text{def}}{=} \{u \in \mathbb{R}^{|\mathcal{S}| |\mathcal{A}|} : \|u\| = 1\}. \quad (27)$$

Our estimation [\(26\)](#page-6-0) above is more tricky than the existing two-point zeroth-order estimations [\(Agarwal et al.,](#page-8-18) [2010;](#page-8-18) [Shamir,](#page-9-9) [2017;](#page-9-9) [Malik et al.,](#page-8-19) [2020\)](#page-8-19) where u<sup>i</sup> is uniformly distributed on U1. To elaborate, we replace their U<sup>1</sup> with U<sup>1</sup> ∩ L0, a complete unit sphere on the linear subspace L0, and further require π ∈ Π<sup>∆</sup> and δ < ∆, to guarantee that π + δu<sup>i</sup> , π − δu<sup>i</sup> ∈ Π for any u<sup>i</sup> ∈ U<sup>1</sup> ∩ L<sup>0</sup> and thus the stochastic gradient estimation [\(26\)](#page-6-0) is valid (see Appendix [H](#page-30-0) for the proof of validity). Moreover, we use the following three steps to obtain u<sup>i</sup> uniformly from U<sup>1</sup> ∩ L0: (1) Obtain v<sup>i</sup> from the uniform distribution on U1; (2) Project v<sup>i</sup> onto L<sup>0</sup> as Eq. [\(28\)](#page-6-1) below; (3) Normalize this projection as Eq. [\(29\)](#page-6-2) below.

$$\text{proj}_{\mathcal{L}_0}(v_i)(a|s) = v_i(a|s) - \frac{1}{|\mathcal{A}|} \sum_{a'} v_i(a'|s), \quad (28)$$

$$u_i = \frac{\text{proj}_{\mathcal{L}_0}(v_i)}{\|\text{proj}_{\mathcal{L}_0}(v_i)\|}. \quad (29)$$

The gradient estimation [\(26\)](#page-6-0) has the following provable error bound.

Proposition 1. *For any* ∆ > δ > 0*,* η ∈ (0, 1) *and* π ∈ Π∆*, then the stochastic gradient* gˆλ,δ(π) *defined by Eq.* [\(26\)](#page-6-0) *is valid and approximates the projected performative policy gradient* projL<sup>0</sup> (∇πV π λ,π) *with the following error bound with probability at least* 1 − η*.*

$$\begin{aligned} & \|\hat{g}_{\lambda,\delta}(\pi) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi})\| \\ & \leq \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3N(\Delta - \delta)} \log\left(\frac{3N|\mathcal{S}||\mathcal{A}|}{\eta}\right) \\ & + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log\left(\frac{3N|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta - \delta}. \quad (30) \end{aligned}$$

Remark: Proposition [1](#page-6-3) above aims to approximate projL<sup>0</sup> (∇πV π λ,π) instead of ∇πV π λ,π. This is sufficient to obtain an ϵ-stationary policy, because for any policies π, π′ , the stationarity measure only involves ⟨∇πV π λ,π, π′ − <sup>π</sup>⟩ which equals to ⟨projL<sup>0</sup> (∇πV π λ,π), π′ − <sup>π</sup>⟩ as <sup>π</sup> ′ − π ∈ L0. Therefore, we only care about projL<sup>0</sup> (∇πV π λ,π).

The approximation error [\(30\)](#page-6-4) has the order of O ϵ<sup>V</sup> <sup>δ</sup> + log(N/η) √ N + δ , which can be arbitrarily small with sufficiently large batchsize N (for reducing the variance), small δ (for reducing the bias), and smaller policy evaluation error ϵ<sup>V</sup> .

Intuition and Novelty for Proving Proposition [1:](#page-6-3) Unlike existing zeroth-order estimations on the whole Euclidean space, our estimation [\(30\)](#page-6-4) is made on the policy space Π, which lies in the linear manifold L<sup>0</sup> + |A|<sup>−</sup><sup>1</sup> ⊂ R |S||A|. The key to our proof is to find an orthogonal transformation T : R |S|(|A|−1) → L0, so that the goal is simplified to analyze the gradient estimation of fλ(x) def = V T(x)+|A|<sup>−</sup><sup>1</sup> λ,T(x)+|A|<sup>−</sup><sup>1</sup> on any <sup>x</sup> ∈ <sup>R</sup> |S|(|A|−1). In particular, the true gradient can be rewritten as ∇fλ(x) = T −1 projL0∇πV π λ,π π=T(x)+|A|<sup>−</sup><sup>1</sup> using differentiability, and when ϵ<sup>V</sup> = 0 (i.e., Vˆ <sup>π</sup> λ,π = V π λ,π for any π ∈ Π), our estimated gradient [\(30\)](#page-6-4) on the policy space Π can be rewritten as the following two-point estimator on R |S|(|A|−1) (see Eq. [\(112\)](#page-32-0) for details).

$$\hat{g}_{\lambda, \delta}(\pi) = \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N [f_\lambda(x + \delta \tilde{u}_i) - f_\lambda(x - \delta \tilde{u}_i)] \tilde{u}_i,$$

where u˜<sup>i</sup> = T −1 (ui) is uniformly distributed on a unit sphere in R |S|(|A|−1) and x = T −1 (π − |A|<sup>−</sup><sup>1</sup> ). Therefore, we can apply estimation analysis to the Euclidean space R |S|(|A|−1). Finally, it is straightforward extend the conclusion from ϵ<sup>V</sup> = 0 to ϵ<sup>V</sup> > 0 by adding the policy evaluation error terms (see Eq. [\(116\)](#page-32-1)).

394

396

Algorithm 1 Zeroth-Order Performative Policy Gradient (0-PPG) Algorithm

1: Inputs: T, N, ∆ > δ > 0, ϵ<sup>V</sup> ≥ 0, β > 0. 2: Initialize: policy π<sup>0</sup> ∈ Π∆. 3: for Iterations t = 0, 1, . . . , T − 1 do 4: Obtain i.i.d. vectors {vi} N <sup>i</sup>=1 uniformly from the unit sphere U<sup>1</sup> def <sup>=</sup> {u∈<sup>R</sup> |S||A| : ∥u∥= 1}. 5: Obtain {projL<sup>0</sup> (vi)} N <sup>i</sup>=1 from Eq. [\(28\)](#page-6-1). 6: Obtain {ui} N <sup>i</sup>=1 by Eq. [\(29\)](#page-6-2). 7: Obtain stochastic policy evaluation Vˆ <sup>π</sup> λ,π ≈ V π λ,π for π ∈ {π<sup>t</sup> ± δui} N <sup>i</sup>=1 with error bound [\(24\)](#page-6-5). 8: Obtain stochastic performative policy gradient estimation gˆλ,δ(πt) using Eq. [\(26\)](#page-6-0). 9: Obtain π˜<sup>t</sup> by Eq. [\(33\)](#page-7-2). 10: Update πt+1 by Eq. [\(32\)](#page-7-3). 11: end for 12: Output: <sup>π</sup>Te where <sup>T</sup>e <sup>∈</sup> arg min0≤t≤<sup>T</sup> <sup>−</sup><sup>1</sup> ⟨gˆλ,δ(πt), π˜<sup>t</sup> − πt⟩.

#### 4.2. Zeroth-Order Performative Policy Gradient (0-PPG) Algorithm

With the estimated gradient gˆλ,δ(πt) defined by Eq. [\(26\)](#page-6-0), we can consider the following Frank-Wolfe algorithm to find an ϵ-stationary policy.

$$\tilde{\pi}_t = \arg \max_{\pi \in \Pi_\Delta} \langle \pi, \hat{g}_\lambda, \delta(\pi_t) \rangle, \quad (31)$$

$$\pi_{t+1} = \pi_t + \beta(\tilde{\pi}_t - \pi_t). \quad (32)$$

Lemma 1. *The step* [\(31\)](#page-7-4) *has the analytical solution below.*

$$\tilde{\pi}_t(a|s) = \begin{cases} \Delta; a \neq \tilde{a}_t(s) \\ 1 - \Delta(|\mathcal{A}| - 1); a = \tilde{a}_t(s) \end{cases}, \quad (33)$$

*where* a˜t(s) ∈ arg max<sup>a</sup> gˆλ,δ(πt)(a|s)*.*

See the proof of Lemma [1](#page-7-5) in Section [A.1.](#page-10-0) Then combining the *performative policy gradient* estimation (see Section [3.1\)](#page-2-6) with the Frank-Wolfe algorithm, we propose our zerothorder performative policy gradient (0-PPG) algorithm (see Algorithm [1\)](#page-7-0). We obtain the following convergence result of Algorithm [1](#page-7-0) in Theorem [4,](#page-7-6) the main theoretical result of this work, as follows.

Theorem 4. *Suppose Assumptions [1](#page-3-4)[-3](#page-3-6) hold. For any* 0 < ϵ ≤ min -24p 2|S| <sup>ℓ</sup><sup>λ</sup> D , 2λ 5|A|D<sup>2</sup>(1−γ) , 288Lλ|S|<sup>1</sup>.<sup>5</sup> |A| Dπmin *and* η ∈ (0, 1)*, select the following hyperparameters for Algorithm [1:](#page-7-0)* ∆ = <sup>π</sup>min 3 *,* β = Dπminϵ <sup>36</sup>ℓλ|S| *,* <sup>δ</sup> <sup>=</sup> O(ϵ)*,* <sup>ϵ</sup><sup>V</sup> <sup>=</sup> O(<sup>ϵ</sup> 2 )*,* N = O[ϵ −2 log(η −1 ϵ −1 )]*, and the number of iterations* T = O(ϵ −2 ) *(see Eqs.* [\(122\)](#page-34-0)*-*[\(127\)](#page-35-0) *in Appendix [J](#page-33-0) for detailed expression of these hyperparameters). Then with probability at least* 1 − η*, the output policy* π˜T˜ *of Algorithm*

*[1](#page-7-0) is an* Dϵ*-stationary policy. Furthermore, if* µ ≥ 0*,* π˜T˜ *is also an* ϵ*-PO policy. The total number of policy evaluations is* 2NT = O[ϵ −4 log(η −1 ϵ −1 )]*.*

Comparison with Existing Works: Theorem [4](#page-7-6) indicates that our 0-PPG algorithm for the first time converges to the desire PO policy with arbitrarily small precision ϵ in polynomial computation complexity, under the mild regularizer dominance condition that µ ≥ 0. In contrast, existing works only converge to a suboptimal PS policy under a similar regularizer dominance condition [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank](#page-9-3) [et al.,](#page-9-3) [2024;](#page-9-3) [Mandal and Radanovic,](#page-8-6) [2024\)](#page-8-6). Our preferable convergence result is due to the major algorithmic difference that existing works adopt repeated retraining algorithms with iteration πt+1 ≈ arg maxπ∈ΠV π<sup>t</sup> λ,π where the policy π is deployed in a fixed environment M<sup>π</sup><sup>t</sup> with π ̸= πt, whereas our 0-PPG algorithm evaluates V π λ,π where each policy π is always deployed at its corresponding environment Mπ.

Intuition and Novelty for Proving Theorem [4:](#page-7-6) Standard convergence analysis of Frank-Wolfe algorithm yields that maxπ˜∈Π<sup>∆</sup> ⟨∇πV πT˜ λ,πT˜ , <sup>π</sup>˜ − <sup>π</sup>T˜⟩ ≤ Dϵ 2 on Π∆. However, it requires a trick to prove the following Proposition [2](#page-7-7) which implies that πT˜ is Dϵ-stationary on Π.

Proposition 2. *If* ∆ ≤ πmin/3 *and a policy* π *satisfies* maxπ˜∈Π<sup>∆</sup> ⟨∇πV π λ,π, <sup>π</sup>˜−π⟩ ≤ Dλ 5|A|(1−γ) *, then the stationary measures on* Π<sup>∆</sup> *and* Π *bound each other as follows.*

$$\max_{\tilde{\pi} \in \Pi} \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \tilde{\pi} - \pi \rangle \leq 2 \max_{\tilde{\pi} \in \Pi_{\Delta}} \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \tilde{\pi} - \pi \rangle \quad (34)$$

To prove Proposition [2,](#page-7-7) note that π ′ defined by Eq. [\(19\)](#page-5-4) also belongs to Π∆, so Theorem [2](#page-5-2) implies π(a|s) ≥ 2∆. Then for any π<sup>2</sup> ∈ Π, we have <sup>π</sup>2+<sup>π</sup> 2 ∈ Π<sup>∆</sup> and thus

$$\begin{aligned} \max_{\pi_2 \in \Pi} \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \pi_2 - \pi \rangle &= 2 \max_{\pi_2 \in \Pi} \left\langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \frac{\pi_2 + \pi}{2} - \pi \right\rangle \\ &\leq 2 \max_{\pi \in \Pi_{\Delta}} \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \tilde{\pi} - \pi \rangle. \end{aligned}$$

# 5. Conclusion

We have studied an entropy-regularized performative reinforcement learning problem, obtained its important properties including gradient dominance, policy lower bound, Lipschitz continuity and smoothness. Based on these properties, we have proposed a zeroth-order performative policy gradient (0-PPG) algorithm only using sample-based policy evaluation, which for the first time converges to a *performatively optimal (PO)* policy with polynomial number of policy evaluations under the regularizer dominance condition. These theoretical results also holds for the quadratice regularizers used in the existing works on performative reinforcement learning (see Appendix [K](#page-36-0) for discussion). A future direction is to extend the algorithm and results to more practical environments of large state and action spaces.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Agarwal, A., Dekel, O., and Xiao, L. (2010). Optimal algorithms for online convex optimization with multipoint bandit feedback. In *Colt*, pages 28–40. Citeseer. Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. *The Journal of Machine Learning Research*, 22(1):4431–4506. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. *ArXiv:2204.05862*. Bhandari, J. and Russo, D. (2024). Global optimality guarantees for policy gradient methods. *Operations Research*. Bhandari, J., Russo, D., and Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. In *Proceedings of the Conference on learning theory (COLT)*, pages 1691–1692. Brown, N. and Sandholm, T. (2019). Superhuman ai for multiplayer poker. *Science*, 365(6456):885–890. Cen, S., Cheng, C., Chen, Y., Wei, Y., and Chi, Y. (2022). Fast global convergence of natural policy gradient methods with entropy regularization. *Operations Research*, 70(4):2563–2578. Chaney, A. J., Stewart, B. M., and Engelhardt, B. E. (2018). How algorithmic confounding in recommendation systems increases homogeneity and decreases utility. In *Proceedings of the 12th ACM conference on recommender systems*, pages 224–232. Chen, Y. (2020). Mirror descent. [https://yuxinche](https://yuxinchen2020.github.io/ele522_optimization/lectures/mirror_descent.pdf) [n2020.github.io/ele522\\_optimization/](https://yuxinchen2020.github.io/ele522_optimization/lectures/mirror_descent.pdf) [lectures/mirror\\_descent.pdf](https://yuxinchen2020.github.io/ele522_optimization/lectures/mirror_descent.pdf). Chen, Z. and Huang, H. (2024). Accelerated policy gradient for s-rectangular robust mdps with large state spaces. In *Proceedings of the International Conference on Machine Learning (ICML)*. Chen, Z., Ma, S., and Zhou, Y. (2022). Sample efficient stochastic policy extragradient algorithm for zero-sum markov game. In *Proceedings of the International Conference on Learning Representations (ICLR)*. Flaxman, A. D., Kalai, A. T., and McMahan, H. B. (2005). Online convex optimization in the bandit setting: gradient descent without a gradient. In *Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms*, pages 385–394. Havrilla, A., Du, Y., Raparthy, S. C., Nalmpantis, C., Dwivedi-Yu, J., Hambro, E., Sukhbaatar, S., and Raileanu,
  - R. (2024). Teaching large language models to reason with reinforcement learning. In *AI for Math Workshop@ ICML 2024*. Leonardos, S., Overman, W., Panageas, I., and Piliouras, G. (2022). Global convergence of multi-agent policy gradient in markov potential games. In *ICLR 2022 Workshop on Gamification and Multiagent Solutions*. Li, G., Wu, W., Chi, Y., Ma, C., Rinaldo, A., and Wei,
  - Y. (2023). Sharp high-probability sample complexities for policy evaluation with linear function approximation. *ArXiv:2305.19001*. Malik, D., Pananjady, A., Bhatia, K., Khamaru, K., Bartlett,
  - P. L., and Wainwright, M. J. (2020). Derivative-free methods for policy optimization: Guarantees for linear quadratic systems. *Journal of Machine Learning Research*, 21(21):1–51. Mandal, D. and Radanovic, G. (2024). Performative reinforcement learning with linear markov decision process. *ArXiv:2411.05234*. Mandal, D., Triantafyllou, S., and Radanovic, G. (2023). Performative reinforcement learning. In *Proceedings of the International Conference on Machine Learning (ICML)*, pages 23642–23680. Mankowitz, D. J., Levine, N., Jeong, R., Abdolmaleki, A., Springenberg, J. T., Shi, Y., Kay, J., Hester, T., Mann, T., and Riedmiller, M. (2019). Robust reinforcement learning for continuous control with model misspecification. In *Proceedings of the International Conference on Learning Representations (ICLR)*. Mansoury, M., Abdollahpouri, H., Pechenizkiy, M., Mobasher, B., and Burke, R. (2020). Feedback loop and bias amplification in recommender systems. In *Proceedings of the 29th ACM international conference on information & knowledge management*, pages 2145–2148. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In *Proceedings of the International Conference on Machine Learning (ICML)*, volume 48, pages 1928–1937.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 549 Nikolaidis, S., Nath, S., Procaccia, A. D., and Srinivasa, S. (2017). Game-theoretic modeling of human adaptation in human-robot collaboration. In *Proceedings of the 2017 ACM/IEEE international conference on human-robot interaction*, pages 323–331. Perdomo, J., Zrnic, T., Mendler-Dünner, C., and Hardt,
  - M. (2020). Performative prediction. In *International Conference on Machine Learning*, pages 7599–7609. Rank, B., Triantafyllou, S., Mandal, D., and Radanovic, G. (2024). Performative reinforcement learning in gradually shifting environments. In *The 40th Conference on Uncertainty in Artificial Intelligence (UAI)*. Samsonov, S., Tiapkin, D., Naumov, A., and Moulines, E. (2023). Finite-sample analysis of the temporal difference learning. *ArXiv:2310.14286*. Shamir, O. (2017). An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. *Journal of Machine Learning Research*, 18(52):1–11. Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. *nature*, 550(7676):354–359. Tropp, J. A. et al. (2015). An introduction to matrix concentration inequalities. *Foundations and Trends® in Machine Learning*, 8(1-2):1–230. Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P., et al. (2019). Grandmaster level in starcraft ii using multi-agent reinforcement learning. *nature*, 575(7782):350–354. Wang, Q., Ho, C. P., and Petrik, M. (2023). Policy gradient in robust mdps with global convergence guarantee. In *Proceedings of the International Conference on Machine Learning (ICML)*, volume 202, pages 35763–35797. Wei, C.-Y., Lee, C.-W., Zhang, M., and Luo, H. (2021). Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games. In *Proceedings of the Conference on Learning Theory (COLT)*. Zhang, J., Bedi, A. S., Wang, M., and Koppel, A. (2021). Beyond cumulative returns via reinforcement learning over state-action occupancy measures. In *2021 American Control Conference (ACC)*, pages 894–901. IEEE.

554

556

| 557 |            |              |              |                                               |                      |
|-----|------------|--------------|--------------|-----------------------------------------------|----------------------|
| A   | Supporting |              | Lemmas       |                                               | 11                   |
|     | A.1        | Frank-Wolfe  |              | Step                                          | 11                   |
|     | A.2        | Lipschitz    |              | Property of Occupany Measure                  | 12                   |
|     | A.3        | Various      | Value        | Functions                                     | 13                   |
|     | A.4        | Zeroth-order |              | Gradient Estimation Error                     | 17                   |
|     | A.5        | Orthogonal   |              | Transformation                                | 19                   |
|     | A.6        | Basic        | Inequalities |                                               | 20                   |
| B   | Negative   |              | Entropy      | Regularizer as a Strongly Convex Function of  | Occupancy Measure 20 |
| C   | Existing   |              | Assumptions  | That Implies Assumption 3                     | 21                   |
| D   | Proof      | of           | Theorem      | 1                                             | 22                   |
| E   | Proof      | of           | Corollary    | 1                                             | 27                   |
| F   | Proof      | of           | Theorem      | 2                                             | 28                   |
| G   | Proof      | of           | Theorem      | 3                                             | 29                   |
| H   | Proof      | of           | Proposition  | 1                                             | 31                   |
| I   | Proof      | of           | Proposition  | 2                                             | 34                   |
| J   | Proof      | of           | Theorem      | 4                                             | 34                   |
| K   | Adjusting  |              | Our          | Results to the Existing Quadratic Regularizer | 37                   |

594

596

598

# Appendix

# Table of Contents

# A. Supporting Lemmas

#### A.1. Frank-Wolfe Step

We repeat Lemma [1](#page-7-5) as follows.

Lemma 2. *The step* [\(31\)](#page-7-4) *has the following analytical solution.*

$$\tilde{\pi}_t(a|s) = \begin{cases} \Delta; a \neq \tilde{a}_t(s) \\ 1 - \Delta(|\mathcal{A}| - 1); a = \tilde{a}_t(s) \end{cases}, \quad (35)$$

*where* a˜t(s) ∈ arg max<sup>a</sup> gˆλ,δ(πt)(a|s)*.*

*Proof.* For π˜<sup>t</sup> defined by Eq. [\(35\)](#page-10-2) and for any π ∈ Π∆, we have

$$\begin{aligned} & \langle \tilde{\pi}_t - \pi, \hat{g}_\lambda, \delta(\pi_t) \rangle \\ &= \sum_{s,a} \hat{g}_{\lambda,\delta}(\pi_t)(a|s) [\tilde{\pi}_t(a|s) - \pi(a|s)] \end{aligned}$$

$$\begin{aligned} &= \sum_s \left\{ \hat{g}_{\lambda, \delta}(\pi_t) [\tilde{a}_t(s) | s] [1 - \Delta(|\mathcal{A}| - 1) - \pi[\tilde{a}_t(s) | s]] - \sum_{a \neq \tilde{a}_t(s)} \hat{g}_{\lambda, \delta}(\pi_t) (a | s) [\pi(a | s) - \Delta] \right\} \\ &\stackrel{(a)}{\geq} \sum_s \left\{ \hat{g}_{\lambda, \delta}(\pi_t) [\tilde{a}_t(s) | s] [1 - \Delta(|\mathcal{A}| - 1) - \pi[\tilde{a}_t(s) | s]] - \sum_{a \neq \tilde{a}_t(s)} \hat{g}_{\lambda, \delta}(\pi_t) [\tilde{a}_t(s) | s] [\pi(a | s) - \Delta] \right\} \\ &= \sum_s \left\{ \hat{g}_{\lambda, \delta}(\pi_t) [\tilde{a}_t(s) | s] [1 - \Delta(|\mathcal{A}| - 1) - \pi[\tilde{a}_t(s) | s]] - \hat{g}_{\lambda, \delta}(\pi_t) [\tilde{a}_t(s) | s] [1 - \pi[\tilde{a}_t(s) | s] - \Delta(|\mathcal{A}| - 1)] \right\} \\ &= 0, \end{aligned}$$

where (a) uses π(a|s) − ∆ ≥ 0 and gˆλ,δ(πt)(a|s) ≤ gˆλ,δ(πt)[˜at(s)|s]. Therefore, Eq. [\(31\)](#page-7-4) holds, that is, π˜<sup>t</sup> = arg maxπ∈Π<sup>∆</sup> ⟨π, gˆλ,δ(πt)⟩.

# A.2. Lipschitz Property of Occupany Measure

Lemma 3. *The occupancy measure* dπ,p *defined by Eq.* [\(2\)](#page-2-7) *has the following Lipschitz properties for any* π, π′ ∈ Π*,* p, p′ ∈ P *and* s˜ ∈ S*.*

$$\sum_s |d_{\pi',p}(s) - d_{\pi,p}(s)| \leq \frac{\gamma}{1-\gamma} \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 \leq \frac{\gamma\sqrt{|\mathcal{A}|}}{1-\gamma} \|\pi' - \pi\| \quad (36)$$

$$\sum_s |d_{\pi, p'}(s) - d_{\pi, p}(s)| \leq \frac{\gamma}{1-\gamma} \max_{s, a} \|p'(\cdot | s, a) - p(\cdot | s, a)\|_1 \leq \frac{\gamma \sqrt{|\mathcal{S}|}}{1-\gamma} \|p' - p\| \quad (37)$$

$$\begin{aligned} \sum_{s,a} |d_{\pi',p'}(s,a) - d_{\pi,p}(s,a)| &\leq \frac{1}{1-\gamma} \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \frac{\gamma}{1-\gamma} \max_{s,a} \|p'(\cdot|s,a) - p(\cdot|s,a)\|_1 \\ &\leq \frac{\sqrt{|\mathcal{A}|}}{1-\gamma} \|\pi' - \pi\| + \frac{\gamma\sqrt{|\mathcal{S}|}}{1-\gamma} \|p' - p\| \end{aligned} \quad (38)$$

*Proof.* The first ≤ of Eqs. [\(36\)](#page-11-1) and [\(37\)](#page-11-2) follows from Lemma 5 of [\(Chen and Huang,](#page-8-10) [2024\)](#page-8-10). The second ≤ of Eqs. [\(36\)](#page-11-1) and [\(37\)](#page-11-2) uses ∥x∥<sup>1</sup> ≤ √ d∥x∥ for any x ∈ <sup>R</sup> d .

Eq. [\(38\)](#page-11-3) can be proved as follows.

$$\begin{aligned} & \sum_{s,a} |d_{\pi',p'}(s,a) - d_{\pi,p}(s,a)| \\ &= \sum_{s,a} |d_{\pi',p'}(s)\pi'(a|s) - d_{\pi,p}(s)\pi(a|s)| \\ &\leq \sum_{s,a} d_{\pi',p'}(s)|\pi'(a|s) - \pi(a|s)| + \pi(a|s)|d_{\pi',p'}(s) - d_{\pi,p}(s)| \\ &\leq \sum_s [d_{\pi',p'}(s) \max_{s'} \|\pi'(\cdot|s') - \pi(\cdot|s')\|_1] + \sum_s |d_{\pi',p'}(s) - d_{\pi,p}(s)| \\ &\stackrel{(a)}{\leq} \max_{s'} \|\pi'(\cdot|s') - \pi(\cdot|s')\|_1 + \frac{\gamma}{1-\gamma} \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \frac{\gamma}{1-\gamma} \max_{s,a} \|p'(\cdot|s,a) - p(\cdot|s,a)\|_1 \\ &\leq \frac{1}{1-\gamma} \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \frac{\gamma}{1-\gamma} \max_{s,a} \|p'(\cdot|s,a) - p(\cdot|s,a)\|_1 \\ &\leq \frac{\sqrt{|\mathcal{A}|}}{1-\gamma} \|\pi' - \pi\| + \frac{\gamma\sqrt{|\mathcal{S}|}}{1-\gamma} \|p' - p\|, \end{aligned}$$

where (a) uses Eqs. [\(36\)](#page-11-1) and [\(37\)](#page-11-2).

689 690

694

696

698

700

704

706

708 709

711

#### A.3. Various Value Functions

Define the following value functions.

$$\begin{aligned} J_\lambda(\pi, \pi', p, r) &\stackrel{\text{def}}{=} \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t [r(s_t, a_t) - \lambda \log \pi'(a_t|s_t)] \mid s_0 \sim \rho \right] \\ &= \frac{1}{1 - \gamma} \sum_{s, a} d_{\pi, p}(s, a) [r(s, a) - \lambda \log \pi'(a|s)], \end{aligned} \quad (39)$$

$$V_\lambda(\pi, \pi', p, r; s) \stackrel{\text{def}}{=} \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t [r(s_t, a_t) - \lambda \log \pi'(a_t | s_t)] \middle| s_0 = s \right], \quad (40)$$

$$Q_\lambda(\pi, \pi', p, r; s, a) \stackrel{\text{def}}{=} \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t [r(s_t, a_t) - \lambda \log \pi'(a_t | s_t)] \mid s_0 = s, a_0 = a \right]$$

$$= r(s, a) - \lambda \log \pi'(a | s) + \gamma \sum_{s'} p(s' | s, a) V_\lambda(\pi, \pi', p, r; s'). \quad (41)$$

Note that the value function [\(8\)](#page-3-3) of interest can be rewritten into the above functions as follows.

$$V_{\lambda, \pi'}^\pi = J_\lambda(\pi, \pi, p_{\pi'}, r_{\pi'}) = \sum_s \rho(s) V_\lambda(\pi, \pi, p_{\pi'}, r_{\pi'}; s) = \sum_{s, a} \rho(s) \pi(a|s) Q_\lambda(\pi, \pi, p_{\pi'}, r_{\pi'}; s, a). \quad (42)$$

Hence, we will investigate the properties of the value functions [\(39\)](#page-12-1)-[\(41\)](#page-12-2) as follows.

Lemma 4. *For any* π ∈ Π*,* p ∈ P*,* r ∈ R*, we have* V π λ,π, Jλ(π, π, p, r), Vλ(π, π, p, r; s), Qλ(π, π, p, r; s, a) ∈ h 0, 1+λ log |A| 1−γ i *.*

*Proof.* We will prove the range of Jλ(π, π, p, r) as follows using r(s, a) ∈ [0, 1]. The proof for the other value functions follow the same way.

$$\begin{aligned} 0 &\leq J_\lambda(\pi, \pi, p, r) = \mathbb{E}_{\pi, p, \rho} \left[ \sum_{t=0}^{\infty} \gamma^t [r(s_t, a_t) - \lambda \log \pi(a_t|s_t)] \right] \\ &\leq \sum_{t=0}^{\infty} \gamma^t + \lambda \mathbb{E}_{\pi, p, \rho} \left[ \sum_{t=0}^{\infty} \gamma^t \sum_a [-\pi(a|s_t) \log \pi(a|s_t)] \right] \\ &\leq \frac{1}{1-\gamma} + \lambda \sum_{t=0}^{\infty} \gamma^t \log |\mathcal{A}| \\ &\leq \frac{1 + \lambda \log |\mathcal{A}|}{1-\gamma}. \end{aligned}$$

Lemma 5. *The gradients of* Jλ(π, π′ , p, r) *defined by Eq.* [\(39\)](#page-12-1) *have the following expressions.*

$$\frac{\partial J_\lambda(\pi, \pi', p, r)}{\partial \pi(a|s)} = \frac{d_{\pi, p}(s)Q_\lambda(\pi, \pi', p, r; s, a)}{1 - \gamma}, \quad (43)$$

$$\frac{\partial J_\lambda(\pi, \pi', p, r)}{\partial \pi'(a|s)} = - \frac{\lambda d_{\pi, p}(s, a)}{(1 - \gamma) \pi'(a|s)}, \quad (44)$$

$$\frac{\partial J_\lambda(\pi, \pi', p, r)}{\partial p(s'|s, a)} = \frac{d_{\pi, p}(s, a)}{1 - \gamma} [r(s, a) - \lambda \log \pi'(a|s) + \gamma V_\lambda(\pi, \pi', p, r; s')], \quad (45)$$

$$\frac{\partial J_\lambda(\pi, \pi', p, r)}{\partial r(s, a)} = \frac{d_{\pi, p}(s, a)}{1 - \gamma}, \quad (46)$$

$$\frac{\partial J_\lambda(\pi, \pi, p, r)}{\partial \pi(a|s)} = \frac{d_{\pi, p}(s)[Q_\lambda(\pi, \pi, p, r; s, a) - \lambda]}{1 - \gamma}. \quad (47)$$

718

724

726

728

731

734

736

738

751

754

756

758

760

764

766

*Proof.* Eq. [\(43\)](#page-12-3) follows from the policy gradient expression in Eq. (7) of [\(Agarwal et al.,](#page-8-13) [2021\)](#page-8-13), with reward function r(s, a) replaced by r(s, a) − λ log π ′ (a|s).

Eq. [\(45\)](#page-12-4) can be proved as follows.

$$\begin{aligned} p(s'|s, a) &\stackrel{(a)}{=} \frac{d_{\pi, p}(s)\pi(a|s)}{1-\gamma} [r(s, a) - \lambda \log \pi(a|s) + \gamma V_\lambda(\pi, \pi', p, r; s')] \\ &= \frac{d_{\pi, p}(s, a)}{1-\gamma} [r(s, a) - \lambda \log \pi(a|s) + \gamma V_\lambda(\pi, \pi', p, r; s')], \end{aligned}$$

where (a) uses Eq. (9) in [\(Chen and Huang,](#page-8-10) [2024\)](#page-8-10).

Eqs. [\(44\)](#page-12-5) and [\(46\)](#page-12-6) can be proved by taking derivatives of Eq. [\(39\)](#page-12-1).

Based on the chain rule, Eq. [\(47\)](#page-12-7) can be proved as follows by adding Eqs. [\(43\)](#page-12-3) and [\(44\)](#page-12-5) with π ′ = π.

$$\begin{aligned} \frac{\partial J_\lambda(\pi, \pi, p, r)}{\partial \pi(a|s)} &= \left[ \frac{\partial J_\lambda(\pi, \pi', p, r)}{\partial \pi(a|s)} + \frac{\partial J_\lambda(\pi, \pi', p, r)}{\partial \pi'(a|s)} \right]_{\pi'=\pi} \\ &= \frac{d_{\pi, p}(s) Q_\lambda(\pi, \pi, p, r; s, a)}{1 - \gamma} - \frac{\lambda d_{\pi, p}(s, a)}{(1 - \gamma) \pi(a|s)} \\ &= \frac{d_{\pi, p}(s) [Q_\lambda(\pi, \pi, p, r; s, a) - \lambda]}{1 - \gamma}, \end{aligned}$$

where the final = uses dπ,p(s, a) = dπ,p(s)π(a|s).

Lemma 6. *The function* J<sup>λ</sup> *defined by eq.* [\(39\)](#page-12-1) *has the following Lipschitz properties for any* π, π′ ∈ Π*,* p, p′ ∈ P *and* r, r′ ∈ R*.*

$$|J_\lambda(\pi', \pi', p, r) - J_\lambda(\pi, \pi, p, r)| \leq L_\pi \max_r \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| \quad (48)$$

$$|J_\lambda(\pi, \pi, p', r) - J_\lambda(\pi, \pi, p, r)| \leq L_p \|p' - p\| \quad (49)$$

$$|J_\lambda(\pi, \pi, p, r') - J_\lambda(\pi, \pi, p, r)| \leq \frac{\|r' - r\|_\infty}{1 - \gamma} \leq \frac{\|r' - r\|}{1 - \gamma} \quad (50)$$

$$\|\nabla_p J_\lambda(\pi', \pi', p, r) - \nabla_p J_\lambda(\pi, \pi, p, r)\| \leq \ell_\pi \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| \quad (51)$$

$$\|\nabla_p J_\lambda(\pi, \pi, p', r) - \nabla_p J_\lambda(\pi, \pi, p, r)\| \leq \ell_p \|p' - p\| \quad (52)$$

$$\|\nabla_p J_\lambda(\pi', \pi', p', r') - \nabla_p J_\lambda(\pi, \pi, p, r)\| \leq \ell_\pi \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + \ell_p \|p' - p\| + \frac{\sqrt{|\mathcal{S}|}}{(1 - \gamma)^2} \|r' - r\|_\infty \quad (53)$$

$$\|\nabla_r J_\lambda(\pi', \pi', p', r') - \nabla_r J_\lambda(\pi, \pi, p, r)\| \leq \frac{\max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \gamma \max_{s,a} \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1}{(1 - \gamma)^2} \quad (54)$$

$$\begin{aligned} \|\nabla_{\pi} J_{\lambda}(\pi', \pi', p', r') - \nabla_{\pi} J_{\lambda}(\pi, \pi, p, r)\| &\leq \left( \frac{|\mathcal{A}|(1 + 2\lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \gamma L_{\pi} \right) \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| \\ &\quad + \gamma \sqrt{|\mathcal{A}|} \left[ \frac{2\sqrt{|\mathcal{S}|}(1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + L_p \right] \|p' - p\| + \frac{\sqrt{|\mathcal{A}|} \|r' - r\|_{\infty}}{1 - \gamma}, \quad (55) \end{aligned}$$

where 
$$L_\pi := \frac{\sqrt{|\mathcal{A}|(2-\gamma+\gamma\lambda \log |\mathcal{A}|)}}{(1-\gamma)^2}$$
,  $L_p := \frac{\sqrt{|\mathcal{S}|(1+\lambda \log |\mathcal{A}|)}}{(1-\gamma)^2}$ ,  $\ell_\pi := \frac{\sqrt{|\mathcal{S}||\mathcal{A}|(2+3\gamma\lambda \log |\mathcal{A}|)}}{(1-\gamma)^3}$  and  $\ell_p := \frac{2\gamma|\mathcal{S}|(1+\lambda \log |\mathcal{A}|)}{(1-\gamma)^3}$ .

*Proof.* Eqs. [\(48\)](#page-13-0), [\(49\)](#page-13-1), [\(51\)](#page-13-2) and [\(52\)](#page-13-3) directly follow from Lemma 6 of [\(Chen and Huang,](#page-8-10) [2024\)](#page-8-10). Eq. [\(50\)](#page-13-4) can be proved as follows.

$$\begin{aligned} |J_\lambda(\pi, p, r') - J_\lambda(\pi, p, r)| &= \left| \frac{1}{1-\gamma} \sum_{s,a} d_{\pi,p}(s, a) [r'(s, a) - r(s, a)] \right| \\ &\leq \frac{1}{1-\gamma} \sum_{s,a} d_{\pi,p}(s, a) |r'(s, a) - r(s, a)| \end{aligned}$$

774

776

778

794

796

800

804

806

808

$$\begin{aligned} &= \frac{1}{1-\gamma} \sum_{s,a} d_{\pi,p}(s,a) \|r' - r\|_\infty \\ &= \frac{1}{1-\gamma} \|r' - r\|_\infty \leq \frac{1}{1-\gamma} \|r' - r\|. \end{aligned}$$

To prove Eq. [\(53\)](#page-13-5), note that

$$\begin{aligned} & \left| \frac{\partial J_\lambda(\pi, \pi, p, r')}{\partial p(s'|s, a)} - \frac{\partial J_\lambda(\pi, \pi, p, r)}{\partial p(s'|s, a)} \right| \\ & \stackrel{(a)}{=} \frac{d_{\pi, p}(s, a)}{1 - \gamma} \left| r'(s, a) - r(s, a) + \gamma [V_\lambda(\pi, \pi', p, r'; s') - V_\lambda(\pi, \pi', p, r; s')] \right| \\ & \stackrel{(b)}{\leq} \frac{d_{\pi, p}(s, a)}{1 - \gamma} \left[ \|r' - r\|_\infty + \gamma \sum_{t=0}^{\infty} \gamma^t \|r' - r\|_\infty \right] \\ & \leq \frac{d_{\pi, p}(s, a)}{(1 - \gamma)^2} \|r' - r\|_\infty \end{aligned} \tag{56}$$

where (a) uses Eq. [\(45\)](#page-12-4) and (b) uses Eq. [\(40\)](#page-12-8). Therefore, we can prove Eq. [\(53\)](#page-13-5) as follows.

$$\begin{aligned} & \|\nabla_p J_\lambda(\pi', \pi', p', r') - \nabla_p J_\lambda(\pi, \pi, p, r)\| \\ & \leq \|\nabla_p J_\lambda(\pi', \pi', p', r') - \nabla_p J_\lambda(\pi, \pi, p', r')\| + \|\nabla_p J_\lambda(\pi, \pi, p', r') - \nabla_p J_\lambda(\pi, \pi, p, r')\| \\ & \quad + \|\nabla_p J_\lambda(\pi, \pi, p, r') - \nabla_p J_\lambda(\pi, \pi, p, r)\| \\ & \stackrel{(a)}{\leq} \ell_\pi \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + \ell_p \|p' - p\| + \sqrt{\sum_{s,a,s'} \left| \frac{\partial J_\lambda(\pi, \pi, p, r')}{\partial p(s'|s, a)} - \frac{\partial J_\lambda(\pi, \pi, p, r)}{\partial p(s'|s, a)} \right|^2} \\ & \stackrel{(b)}{\leq} \ell_\pi \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + \ell_p \|p' - p\| + \sqrt{\frac{\|r' - r\|_\infty^2}{(1-\gamma)^4} \sum_{s,a,s'} d_{\pi,p}^2(s, a)} \\ & \leq \ell_\pi \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + \ell_p \|p' - p\| + \frac{\sqrt{|\mathcal{S}|}}{(1-\gamma)^2} \|r' - r\|_\infty, \end{aligned}$$

where (a) uses Eqs. [\(51\)](#page-13-2)-[\(52\)](#page-13-3) and (b) uses Eq. [\(56\)](#page-14-0).

Then, we prove Eq. [\(54\)](#page-13-6) as follows.

$$\begin{aligned} & \|\nabla_r J_\lambda(\pi', \pi', p', r') - \nabla_r J_\lambda(\pi, \pi, p, r)\| \\ & \stackrel{(a)}{=} \frac{\|d_{\pi', p'} - d_{\pi, p}\|}{1 - \gamma} \\ & \leq \frac{\|d_{\pi', p'} - d_{\pi, p}\|_1}{1 - \gamma} \\ & \stackrel{(b)}{\leq} \frac{1}{(1 - \gamma)^2} \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \frac{\gamma}{(1 - \gamma)^2} \max_{s,a} \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1, \end{aligned}$$

where (a) uses Eq. [\(46\)](#page-12-6), (b) uses Eq. [\(38\)](#page-11-3).

To prove Eq. [\(55\)](#page-13-7), we will first prove the following auxiliary bounds.

$$Q_\lambda(\pi, \pi, p, r; s, a) - \lambda \stackrel{(a)}{\in} \left[ -\lambda, \frac{1 + \lambda \log |\mathcal{A}|}{1 - \gamma} - \lambda \right] \Rightarrow |Q_\lambda(\pi, \pi, p, r; s, a) - \lambda| \leq \frac{1 + \lambda \log |\mathcal{A}|}{1 - \gamma}, \quad (57)$$

where (a) uses Lemma [4.](#page-12-9)

$$\begin{aligned}
& |V_\lambda(\pi', \pi', p', r'; s) - V_\lambda(\pi, \pi, p, r; s)| \\
& \leq |V_\lambda(\pi', \pi', p', r'; s) - V_\lambda(\pi, \pi, p', r'; s)| + |V_\lambda(\pi, \pi, p', r'; s) - V_\lambda(\pi, \pi, p, r'; s)| + |V_\lambda(\pi, \pi, p, r'; s) - V_\lambda(\pi, \pi, p, r; s)|
\end{aligned}$$

828

831

834

836

838

854

856

858

860

864

866

868

874

876

$$\stackrel{(a)}{\leq} L_{\pi} \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + L_p \|p' - p\| + \frac{\|r' - r\|_{\infty}}{1 - \gamma}, \quad (58)$$

where (a) applies Eqs. [\(48\)](#page-13-0)-[\(50\)](#page-13-4) to the case where the initial state distribution ρ is probability 1 at s (so Jλ(π, π, p, r) becomes Vλ(π, π, p, r; s)).

$$\begin{aligned}
& |Q_\lambda(\pi, \pi, p, r'; s, a) - Q_\lambda(\pi, \pi, p, r; s, a)| \\
& \stackrel{(a)}{=} \left| \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t [r'(s_t, a_t) - r(s_t, a_t)] \middle| s_0 = s, a_0 = a \right] \right| \\
& \leq \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t [r'(s_t, a_t) - r(s_t, a_t)] \middle| s_0 = s, a_0 = a \right] \\
& \leq \mathbb{E}_{\pi, p} \left[ \sum_{t=0}^{\infty} \gamma^t \|r' - r\|_\infty \middle| s_0 = s, a_0 = a \right] \\
& \leq \frac{\|r' - r\|_\infty}{1 - \gamma},
\end{aligned} \tag{59}$$

where (a) uses Eq. [\(41\)](#page-12-2).

$$\begin{aligned}
& |Q_\lambda(\pi', \pi', p', r; s, a) - Q_\lambda(\pi, \pi, p, r; s, a)| \\
& \stackrel{(a)}{\leq} \lambda |\log \pi'(a|s) - \log \pi(a|s)| + \gamma \left| \sum_{s'} [p'(s'|s, a) V_\lambda(\pi', \pi', p', r; s) - p(s'|s, a) V_\lambda(\pi, \pi, p, r; s)] \right| \\
& \leq \lambda |\log \pi'(a|s) - \log \pi(a|s)| + \gamma \sum_{s'} p'(s'|s, a) |V_\lambda(\pi', \pi', p', r; s) - V_\lambda(\pi, \pi, p, r; s)| \\
& \quad + \gamma \sum_{s'} |p'(s'|s, a) - p(s'|s, a)| |V_\lambda(\pi, \pi, p, r; s)| \\
& \stackrel{(b)}{\leq} \lambda |\log \pi'(a|s) - \log \pi(a|s)| + \gamma L_\pi \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + \gamma L_p \|p' - p\| \\
& \quad + \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1, \tag{60}
\end{aligned}$$

where (a) uses Eq. [\(41\)](#page-12-2), and (b) uses Eq. [\(58\)](#page-15-0) and Lemma [4.](#page-12-9)

Note that

$$\begin{aligned} & (1 - \gamma) \left| \frac{\partial J_\lambda(\pi', \pi', p', r')}{\partial \pi'(a|s)} - \frac{\partial J_\lambda(\pi, \pi, p, r)}{\partial \pi(a|s)} \right| \\ & \stackrel{(a)}{=} \left| d_{\pi', p'}(s) [Q_\lambda(\pi', \pi', p', r'; s, a) - \lambda] - d_{\pi, p}(s) [Q_\lambda(\pi, \pi, p, r; s, a) - \lambda] \right| \\ & \leq \left| [d_{\pi', p'}(s) - d_{\pi, p}(s)] [Q_\lambda(\pi', \pi', p', r'; s, a) - \lambda] + d_{\pi, p}(s) [Q_\lambda(\pi', \pi', p', r'; s, a) - Q_\lambda(\pi', \pi', p', r; s, a)] \right. \\ & \quad \left. + d_{\pi, p}(s) [Q_\lambda(\pi', \pi', p', r; s, a) - Q_\lambda(\pi, \pi, p, r; s, a)] \right| \\ & \leq \left| d_{\pi', p'}(s) - d_{\pi, p}(s) \right| \cdot \left| Q_\lambda(\pi', \pi', p', r'; s, a) - \lambda \right| + d_{\pi, p}(s) \left| Q_\lambda(\pi', \pi', p', r'; s, a) - Q_\lambda(\pi', \pi', p', r; s, a) \right| \\ & \quad + d_{\pi, p}(s) \left| Q_\lambda(\pi', \pi', p', r; s, a) - Q_\lambda(\pi, \pi, p, r; s, a) \right| \\ & \stackrel{(b)}{\leq} \frac{1 + \lambda \log |\mathcal{A}|}{1 - \gamma} \left| d_{\pi', p'}(s) - d_{\pi, p}(s) \right| + \frac{d_{\pi, p}(s) \|r' - r\|_\infty}{1 - \gamma} + d_{\pi, p}(s) \left[ \lambda |\log \pi'(a|s) - \log \pi(a|s)| \right. \\ & \quad \left. + \gamma L_\pi \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + \gamma L_p \|p' - p\| + \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1 \right], \end{aligned}$$

where (a) uses Eq. [\(47\)](#page-12-7), (b) uses Eqs. [\(57\)](#page-14-1), [\(59\)](#page-15-1) and [\(60\)](#page-15-2). Applying triangular inequality to the bound above, we can prove Eq. [\(55\)](#page-13-7) as follows.

$$(1 - \gamma) \|\nabla_{\pi'} J_\lambda(\pi', \pi', p', r') - \nabla_\pi J_\lambda(\pi, \pi, p, r)\|$$

$$\begin{aligned}
& 880 \leq \frac{1 + \lambda \log |\mathcal{A}|}{1 - \gamma} \sqrt{\sum_{s,a} |d_{\pi',p'}(s) - d_{\pi,p}(s)|^2} + \frac{\|r' - r\|_\infty}{1 - \gamma} \sqrt{\sum_{s,a} d_{\pi,p}(s)^2} + \lambda \sqrt{\sum_{s,a} d_{\pi,p}(s)^2 |\log \pi'(a|s) - \log \pi(a|s)|^2} \\
& 880 \quad + [\gamma L_\pi \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + \gamma L_p \|p' - p\|] \sqrt{\sum_{s,a} d_{\pi,p}(s)^2} \\
& 884 \quad + \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \sqrt{\sum_{s,a} d_{\pi,p}(s)^2 \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1^2} \\
& 887 \quad + \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \sqrt{\sum_{s,a} d_{\pi,p}(s)^2 \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1^2} \\
& 888 \leq \frac{\sqrt{|\mathcal{A}|}(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \sum_s |d_{\pi',p'}(s) - d_{\pi,p}(s)| + \frac{\sqrt{|\mathcal{A}|} \|r' - r\|_\infty}{1 - \gamma} + \lambda \sqrt{\sum_s d_{\pi,p}(s) \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\|^2} \\
& 889 \quad + [\gamma L_\pi \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + \gamma L_p \|p' - p\|] \sqrt{|\mathcal{A}|} + \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \sqrt{|\mathcal{S}| \sum_{s,a} \|p'(\cdot|s, a) - p(\cdot|s, a)\|^2} \\
& 890 \quad + \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \sqrt{|\mathcal{S}| \sum_{s,a} \|p'(\cdot|s, a) - p(\cdot|s, a)\|^2} \\
& 891 \leq \frac{(a) \gamma \sqrt{|\mathcal{A}|}(1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} [\max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \max_{s,a} \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1] + \frac{\sqrt{|\mathcal{A}|} \|r' - r\|_\infty}{1 - \gamma} \\
& 896 \quad + \lambda \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + [\gamma L_\pi \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + \gamma L_p \|p' - p\|] \sqrt{|\mathcal{A}|} \\
& 898 \quad + \frac{\gamma \sqrt{|\mathcal{S}|}(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \|p' - p\| \\
& 900 \leq \frac{(b) [\frac{|\mathcal{A}|(\gamma + 2\lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \gamma L_\pi] \max_{s'} \|\log \pi'(\cdot|s') - \log \pi(\cdot|s')\| + \gamma \sqrt{|\mathcal{A}|} [\frac{2\sqrt{|\mathcal{S}|}(1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + L_p] \|p' - p\|}{(1 - \gamma)^2} \\
& 903 \quad + \frac{\sqrt{|\mathcal{A}|} \|r' - r\|_\infty}{1 - \gamma}, \\
& 905 \quad + \frac{\sqrt{|\mathcal{A}|} \|r' - r\|_\infty}{1 - \gamma},
\end{aligned}$$

911

914 915 916

918

924

928

where (a) uses Lemma [3,](#page-11-4) (b) uses ∥π ′ (·|s) − π(·|s)∥<sup>1</sup> ≤ ∥ log π ′ (·|s) − log π(·|s)∥1,

$$\begin{aligned} \|p'(\cdot|s, a) - p(\cdot|s, a)\|_1 &\leq \sqrt{|\mathcal{S}|} \|p'(\cdot|s, a) - p(\cdot|s, a)\| \leq \sqrt{|\mathcal{S}|} \|p' - p\|, \quad \frac{\gamma \sqrt{|\mathcal{S}|}(1+\lambda \log |\mathcal{A}|)}{1-\gamma} \leq \frac{\sqrt{|\mathcal{S}|} |\mathcal{A}| (1+\lambda \log |\mathcal{A}|)}{(1-\gamma)^2} \quad \text{and} \\ \lambda &\leq \frac{\lambda |\mathcal{A}| \log |\mathcal{A}|}{(1-\gamma)^2}. \quad \square \end{aligned}$$

#### A.4. Zeroth-order Gradient Estimation Error

We import Theorem 1.6.2 of [\(Tropp et al.,](#page-9-10) [2015\)](#page-9-10) as follows.

Lemma 7 (Matrix Bernstein Inequality). *Suppose complex-valued matrices* S1, . . . , S<sup>N</sup> ∈ <sup>C</sup> <sup>d</sup>1×d<sup>2</sup> *are independently distributed with* <sup>E</sup>S<sup>k</sup> = 0 *and* ∥Sk∥ ≤ C *for each* k = 1, . . . , N*. Denote the sum* Z<sup>N</sup> = P<sup>N</sup> <sup>k</sup>=1 S<sup>k</sup> *its variance statistic as follows*

$$v(Z_N) = \max \left[ \left\| \sum_{k=1}^N \mathbb{E}(S_k S_k^*) \right\|, \left\| \sum_{k=1}^N \mathbb{E}(S_k^* S_k) \right\| \right], \quad (61)$$

*where* S ∗ k *denotes the conjugate transpose of* Sk*. Then for any* ϵ ≥ 0*, we have*

$$\mathbb{P}\{\|Z_N\| \geq \epsilon\} \leq (d_1 + d_2) \exp \left[ \frac{-\epsilon^2/2}{v(Z_N) + C\epsilon/3} \right]. \quad (62)$$

Applying the above lemma to vectors, we obtain the following vector Bernstein inequality.

Lemma 8 (Vector Bernstein Inequality). *Suppose independently distributed vectors* x1, . . . , x<sup>N</sup> ∈ <sup>C</sup> d *satisfies* ∥xk∥ ≤ c *for each* k = 1, . . . , N*. Then for any* η ∈ (0, 1)*, with probability at least* 1 − η*, we have*

$$\left\| \frac{1}{N} \sum_{k=1}^N (x_k - \mathbb{E}x_k) \right\| < \frac{4c}{3N} \log\left(\frac{d+1}{\eta}\right) + 2c\sqrt{\frac{2}{N} \log\left(\frac{d+1}{\eta}\right)}. \quad (63)$$

938

954

956

958

971

974

976

978

*Proof.* Note that S<sup>k</sup> = x<sup>k</sup> − <sup>E</sup>x<sup>k</sup> satisfies the conditions of Lemma [7](#page-16-1) with d<sup>1</sup> = d, d<sup>2</sup> = 1 and C replaced by 2c. In addition, v(Z<sup>N</sup> ) defined by Eq. [\(61\)](#page-16-2) satisfies v(Z<sup>N</sup> ) ≤ 4N c<sup>2</sup> since

$$\max[\|S_k S_k^*\|, \|S_k^* S_k\|^2] \leq \|S_k^*\|^2 \|S_k\|^2 \leq 4c^2.$$

For any η ∈ (0, 1), let

$$\epsilon = \frac{4c}{3} \log\left(\frac{d+1}{\eta}\right) + c\sqrt{2N \log\left(\frac{d+1}{\eta}\right)}.$$

Therefore, Lemma [7](#page-16-1) implies that

$$\mathbb{P}\left\{\frac{1}{N} \left\| \sum_{k=1}^N (x_k - \mathbb{E}x_k) \right\| \geq \frac{\epsilon}{N} \right\} \leq (d+1) \exp \left[ \frac{-\epsilon^2/2}{4Nc^2 + 2c\epsilon/3} \right] \leq \eta,$$

which implies that with probability at least 1 − η, we have

$$\frac{1}{N} \left\| \sum_{k=1}^N (x_k - \mathbb{E}x_k) \right\| < \frac{\epsilon}{N} = \frac{4c}{3N} \log\left(\frac{d+1}{\eta}\right) + 2c\sqrt{\frac{2}{N}} \log\left(\frac{d+1}{\eta}\right).$$

For any function f : R <sup>d</sup> → <sup>R</sup>, obtain the following zeroth-order stochastic estimator of the gradient ∇f.

$$g_\delta(x) = \frac{d}{2N\delta} \sum_{i=1}^N [f(x + \delta u_i) - f(x - \delta u_i)] u_i \approx \nabla f(x) \quad (64)$$

where δ > 0 and {ui} N <sup>i</sup>=1 are i.i.d. samples of the uniform distribution on the sphere <sup>S</sup><sup>d</sup> = {u ∈ <sup>R</sup> d : ∥u∥ = 1}.

Lemma 9. *Suppose* f : R <sup>d</sup> → <sup>R</sup> *is an* L<sup>f</sup> *-Lipschitz continuous and* ℓ<sup>f</sup> *-smooth function. Then for any* η ∈ (0, 1)*, with probability at least* 1 − η*, the gradient estimator* g<sup>δ</sup> *defined by Eq.* [\(64\)](#page-17-0) *has the following error bound.*

$$\|g_\delta(x) - \nabla f(x)\| \leq \frac{4L_f d}{3N} \log\left(\frac{d+1}{\eta}\right) + 2L_f d \sqrt{\frac{2}{N} \log\left(\frac{d+1}{\eta}\right)} + \delta \ell_f. \quad (65)$$

*Proof.* Note that gδ,i(x) def = d 2δ [f(x + δui) − f(x − δui)]u<sup>i</sup> has the following norm bound

$$\|g_{\delta,i}(x)\| \leq \frac{d}{2\delta} |f(x + \delta u_i) - f(x - \delta u_i)| \cdot \|u_i\| \leq \frac{d}{2\delta} \cdot L_f \|2\delta u_i\| = L_f d. \quad (66)$$

Define the following smoothed approximation of f as follows.

$$f_\delta(x) \stackrel{\text{def}}{=} \mathbb{E}_{v \sim \text{Unif}(\mathbb{B}_d)}[f(x + \delta v)], \quad (67)$$

where Unif(<sup>B</sup>d) denotes the uniform distribution on the ball <sup>B</sup><sup>d</sup> def <sup>=</sup> {<sup>u</sup> ∈ <sup>R</sup> d : ∥u∥ ≤ 1}. Then based on Lemma 1 of [\(Flaxman et al.,](#page-8-20) [2005\)](#page-8-20), we have

$$\mathbb{E}[g_{\delta,i}(x)] = \nabla f_\delta(x) = \mathbb{E}_{v \sim \text{Unif}(\mathbb{E}_d)}[\nabla f(x + \delta v)]. \quad (68)$$

Therefore, applying Lemma [8](#page-16-3) to gδ,i(x), the following bound holds with probability at least 1 − η.

$$\frac{1}{N} \left\| \sum_{i=1}^N [g_{\delta,i}(x) - \nabla f_\delta(x)] \right\| < \frac{4L_f d}{3N} \log\left(\frac{d+1}{\eta}\right) + 2L_f d \sqrt{\frac{2}{N} \log\left(\frac{d+1}{\eta}\right)}. \quad (69)$$

994

996

998

1014 *Proof.* It can be verified that R d admits the following orthonormal basis with ⟨e<sup>i</sup> , e<sup>j</sup> ⟩ = 0 for any i ̸= j and ∥ei∥ = 1.

1016

1019

1024

1026

1029

1034

1036

Note that

$$\|\nabla f_\delta(x) - \nabla f(x)\| = \|\mathbb{E}_{v \sim \text{Unif}(\mathbb{B}_d)}[\nabla f(x + \delta v) - \nabla f(x)]\| \leq \delta \ell_f. \quad (70)$$

As a result, we can prove the conclusion as follows by using Eqs. [\(69\)](#page-17-1) and [\(70\)](#page-18-1) above.

$$\begin{aligned} \|g_\delta(x) - \nabla f(x)\| &= \left\| \left[ \frac{1}{N} \sum_{i=1}^N g_{\delta,i}(x) \right] - \nabla f(x) \right\| \\ &\leq \left\| \left[ \frac{1}{N} \sum_{i=1}^N g_{\delta,i}(x) \right] - \nabla f_\delta(x) \right\| + \|\nabla f_\delta(x) - \nabla f(x)\| \\ &< \frac{4L_f d}{3N} \log\left(\frac{d+1}{\eta}\right) + 2L_f d \sqrt{\frac{2}{N} \log\left(\frac{d+1}{\eta}\right)} + \delta \ell_f. \end{aligned}$$

#### A.5. Orthogonal Transformation

Lemma 10. *There exists an orthogonal transformation* T *from the space* R d−1 *to* Z<sup>d</sup> = {z = [z1, . . . , zd] ∈ <sup>R</sup> d : P i z<sup>i</sup> = 0}*, that is,* T *is invertible and satisfies the following properties for any* x, y ∈ Z<sup>d</sup> *and* α, β ∈ <sup>R</sup>*.*

$$\mathcal{T}(\alpha x + \beta y) = \alpha \mathcal{T}(x) + \beta \mathcal{T}(y), \quad (71)$$

$$\langle \mathcal{T}(x), \mathcal{T}(y) \rangle = \langle x, y \rangle. \quad (72)$$

$$e_k = \frac{1}{\sqrt{k(k+1)}} \underbrace{[1, 1, \dots, 1, -k, \underbrace{0, 0, \dots, 0}]}_{k \ 1's} \in \mathbb{R}^d; k = 1, 2, \dots, d-1.$$

$$e_d = \frac{1}{\sqrt{d}} \underbrace{[1, 1, \dots, 1]}_{d \ 1's} \in \mathbb{R}^d.$$

Define the transformation T at x = [x1, x2, . . . , xd−1] ∈ <sup>R</sup> d−1 as follows.

$$\mathcal{T}(x) = \sum_{i=1}^{d-1} x_i e_i. \quad (73)$$

Since Z<sup>d</sup> is a linear subspace of <sup>R</sup> <sup>d</sup> orthogonal to ed, Z<sup>d</sup> admits the orthonormal basis {ei} d−1 <sup>i</sup>=1 . Hence, T (x) ∈ Zd. Conversely, for any y ∈ Zd, there exists unique x ∈ <sup>R</sup> d−1 such that y = P<sup>d</sup>−<sup>1</sup> <sup>i</sup>=1 xie<sup>i</sup> . Hence, T : R <sup>d</sup>−<sup>1</sup> → Z<sup>d</sup> is invertible. For any x = [x1, . . . , xd−1], y = [y1, . . . , yd−1] ∈ <sup>R</sup> d−1 and α, β ∈ R, we can prove Eqs. [\(71\)](#page-18-2) and [\(72\)](#page-18-3) respectively as follows.

$$\begin{aligned}\mathcal{T}(\alpha x + \beta y) &= \sum_{i=1}^{d-1} (\alpha x_i + \beta y_i) e_i \\ &= \alpha \sum_{i=1}^{d-1} x_i e_i + \beta \sum_{i=1}^{d-1} y_i e_i \\ &= \alpha \mathcal{T}(x) + \beta \mathcal{T}(y).\end{aligned}$$

1054 Lemma 11. *For any* ϵ ∈ (0, 0.5] *and* x ≥ 4ϵ −1 log(ϵ −1 )*, the following inequality holds.*

1056

1059 1060 *Specifically, any* x ≥ 3 *satisfies* log <sup>x</sup> <sup>x</sup> ≤ 2 *.*

1061 1062 1063 *Proof.* As ϵ <sup>−</sup><sup>1</sup> ≥ 2, we have x ≥ 4ϵ −1 log(ϵ −1 ) ≥ (4)(2) log(2) > 5.54, so log x > log 5.54 > 1.71, which proves the first < of Eq. [\(74\)](#page-19-2).

1064 Note that the function f(x) = log <sup>x</sup> x has the following derivative

1065 1066 1067

1068 1069 where < uses log x > 1.71. Hence, f is monotonic decreasing in x ≥ 4ϵ −1 log(ϵ −1 ) > 5.54, Therefore, we prove the second ≤ of Eq. [\(74\)](#page-19-2) as follows.

1074

1076 When x ≥ 3, f ′ (x) = <sup>1</sup>−log <sup>x</sup> <sup>x</sup><sup>2</sup> <sup>&</sup>lt; <sup>0</sup>, so <sup>f</sup>(x) <sup>≤</sup> <sup>f</sup>(3) = log 3 <sup>3</sup> < 1 2 .

1079

1089 The negative entropy regularizer [\(7\)](#page-3-2) can be rewritten as follows

1090 1091 1092

1093 1094 1095 where dπ,pπ′ (s) = P <sup>a</sup>′ <sup>d</sup>π,pπ′ (s, a′ ). Hence, it suffices to prove that the following function of occupancy measure d is strongly convex.

$$\begin{aligned} &= \sum_{i=1}^{d-1} \sum_{j=1}^{d-1} x_i y_j \langle e_i, e_j \rangle \\ &= \sum_{i=1}^{d-1} x_i y_i = \langle x, y \rangle. \end{aligned}$$

#### A.6. Basic Inequalities

$$0 < \frac{\log x}{x} \leq \epsilon \quad (74)$$

$$f'(x) = \frac{1 - \log x}{x^2} < 0,$$

$$\frac{\log x}{x\epsilon} \leq \frac{\log[4\epsilon^{-1} \log(\epsilon^{-1})]}{\epsilon[4\epsilon^{-1} \log(\epsilon^{-1})]} = \frac{\log 4 + \log(\epsilon^{-1}) + \log[\log(\epsilon^{-1})]}{4 \log(\epsilon^{-1})} \stackrel{(a)}{\leq} \frac{\log 4}{4 \log(2)} + \frac{\log(\epsilon^{-1}) + \log(\epsilon^{-1})}{4 \log(\epsilon^{-1})} = 1, \quad (75)$$

where (a) uses ϵ <sup>−</sup><sup>1</sup> ≥ 2 and log u ≤ u for u = log(ϵ −1 ).

Lemma 12. *For any* π, π′ ∈ Π*, we have* ∥π ′ − π∥ ≤ p 2|S|*.*

*Proof.*

$$\|\pi' - \pi\|^2 = \sum_{s,a} |\pi'(a|s) - \pi(a|s)|^2 \leq \sum_{s,a} [\pi'^2(a|s) + \pi^2(a|s)] \leq \sum_{s,a} [\pi'(a|s) + \pi(a|s)] = 2|\mathcal{S}|.$$

# B. Negative Entropy Regularizer as a Strongly Convex Function of Occupancy Measure

$$\mathcal{H}_{\pi'}(\pi) = \mathbb{E}_{\pi, p_{\pi'}}, \left[ \sum_{t=0}^{\infty} \gamma^t \log \pi(a_t | s_t) \right] = \frac{1}{1-\gamma} \sum_{s, a} d_{\pi, p_{\pi'}}(s, a) \log \frac{d_{\pi, p_{\pi'}}(s, a)}{d_{\pi, p_{\pi'}}(s)}, \quad (76)$$

$$H(d) = \sum_{s,a} d(s, a) \log \frac{d(s, a)}{d(s)}, \quad (77)$$

1104

1106

1109

1111

1114

1116

1118 1119

1124

1126

1129

1134

1136

1151

where d(s) = P <sup>a</sup>′ <sup>d</sup>(s, a′ ). For any α ∈ [0, 1] and occupancy measures d1, d0, denote d<sup>α</sup> = αd<sup>1</sup> + (1 − α)d<sup>0</sup> and the corresponding policy as <sup>π</sup>α(a|s) = <sup>d</sup>α(s,a) dα(s) . Then we have

$$\begin{aligned}
& \alpha H(d_1) + (1 - \alpha)H(d_0) - H(d_\alpha) \\
&= \sum_{s,a} \left[ \alpha d_1(s,a) \log \pi_1(a|s) + (1 - \alpha)d_0(s,a) \log \pi_0(a|s) - [\alpha d_1(s,a) + (1 - \alpha)d_0(s,a)] \log \pi_\alpha(a|s) \right] \\
&= \sum_{s,a} \left[ \alpha d_1(s,a) \log \frac{\pi_1(a|s)}{\pi_\alpha(a|s)} + (1 - \alpha)d_0(s,a) \log \frac{\pi_0(a|s)}{\pi_\alpha(a|s)} \right] \\
&= \sum_{s,a} \left[ \alpha d_1(s) \pi_1(a|s) \log \frac{\pi_1(a|s)}{\pi_\alpha(a|s)} + (1 - \alpha)d_0(s) \pi_0(a|s) \log \frac{\pi_0(a|s)}{\pi_\alpha(a|s)} \right] \\
&= \sum_s \left[ \alpha d_1(s) \text{KL}[\pi_1(\cdot|s) \| \pi_\alpha(a|s)] + (1 - \alpha)d_0(s) \text{KL}[\pi_0(\cdot|s) \| \pi_\alpha(a|s)] \right] \\
&\stackrel{(a)}{\geq} \frac{1}{2} \sum_s \left[ \alpha d_1(s) \| \pi_1(\cdot|s) - \pi_\alpha(\cdot|s) \|_1^2 + (1 - \alpha)d_0(s) \| \pi_0(\cdot|s) - \pi_\alpha(\cdot|s) \|_1^2 \right] \\
&\stackrel{(b)}{\geq} \frac{D}{2} \sum_s \left[ \alpha \| \pi_1(\cdot|s) - \pi_\alpha(\cdot|s) \|_1^2 + (1 - \alpha) \| \pi_0(\cdot|s) - \pi_\alpha(\cdot|s) \|_1^2 \right] \\
&\geq \frac{D}{2} \left[ \alpha \max_s \| \pi_1(\cdot|s) - \pi_\alpha(\cdot|s) \|_1^2 + (1 - \alpha) \max_s \| \pi_0(\cdot|s) - \pi_\alpha(\cdot|s) \|_1^2 \right] \\
&\stackrel{(c)}{\geq} \frac{D(1 - \gamma)}{2} \left[ \alpha \| d_1 - d_\alpha \|_1^2 + (1 - \alpha) \max_s \| d_0 - d_\alpha \|_1^2 \right] \\
&= \frac{D(1 - \gamma)}{2} \left[ \alpha(1 - \alpha)^2 \| d_1 - d_0 \|_1^2 + (1 - \alpha) \alpha^2 \| d_1 - d_0 \|_1^2 \right] \\
&= \frac{\alpha(1 - \alpha)}{2} \cdot D(1 - \gamma) \| d_1 - d_0 \|_1^2. \tag{78}
\end{aligned}$$

where (a) uses Pinsker's inequality, (b) uses Assumption [3,](#page-3-6) (c) uses Eq. [\(38\)](#page-11-3) with p ′ = p. The inequality above implies that H(d) is D(1 − γ)-strongly convex, so the negative entropy regularizer [\(76\)](#page-19-3) can be seen as a D-strongly convex function of the occupancy measure dπ,pπ′ .

# C. Existing Assumptions That Implies Assumption [3](#page-3-6)

The following assumptions have been used in the reinforcement learning literature. We will show that each of these assumptions implies Assumption [3.](#page-3-6)

Assumption 4. *[\(Bhandari and Russo,](#page-8-15) [2024\)](#page-8-15)* ρ(s) > 0 *for any* s ∈ S*.*

Assumption 5. *[\(Agarwal et al.,](#page-8-13) [2021;](#page-8-13) [Leonardos et al.,](#page-8-14) [2022;](#page-8-14) [Wang et al.,](#page-9-6) [2023;](#page-9-6) [Chen and Huang,](#page-8-10) [2024\)](#page-8-10)*

D<sup>ρ</sup> := supπ∈Π,p∈P ∥dπ,p/ρ∥<sup>∞</sup> < ∞*.*

Assumption 6. *[\(Wei et al.,](#page-9-5) [2021;](#page-9-5) [Chen et al.,](#page-8-12) [2022\)](#page-8-12) There exists a constant* µmin > 0 *and mixing time* t*mix* ∈ <sup>N</sup> *such that under any policy* π ∈ Π *and transition kernel* p ∈ P*, the stationary state distribution* µπ,p(s) *has uniform lower bound* mins∈S µπ,p(s) ≥ µmin*, and*

$$d_{\text{TV}} [\mathbb{P}_{\pi, p, \rho}(s_{t_{\text{mix}}} = \cdot), \mu_{\pi, p}] \leq \frac{1}{4}, \quad (79)$$

*where* <sup>P</sup>π,p,ρ(s<sup>t</sup>*mix* = ·) *denotes the state distribution at time* t*mix, under the policy* π*, transition kernel* p *and initial state distribution* ρ*, and* dTV *denotes the total variation distance between two probability distributions.*

Proof of Assumption [4](#page-20-1)⇒Assumption [3:](#page-3-6) For any policy π ∈ Π, transition kernel p ∈ P and state s ∈ S, we have

$$d_{\pi,p}(s) = \sum_a d_{\pi,p}(s, a)$$

1159 1160 1161

1164

1174

1176

1194

1196

1199 1200

1204

1206

$$\begin{aligned} &\stackrel{(a)}{=} \sum_a (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}_{\pi,p,\rho}\{s_t = s, a_t = a\} \\ &= (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}_{\pi,p,\rho}\{s_t = s\} \\ &\geq (1-\gamma) \mathbb{P}_{\pi,p,\rho}\{s_0 = s\} \\ &= (1-\gamma) \rho(s) \\ &\geq (1-\gamma) \min_{s \in \mathcal{S}} \rho(s). \end{aligned}$$

As S is a finite state space, ρ(s) > 0, ∀s ∈ S implies that mins∈S ρ(s) > 0. Hence, Assumption [3](#page-3-6) holds with D = (1 − γ) mins∈S ρ(s) > 0.

Proof of Assumption [5](#page-20-2)⇒Assumption [3:](#page-3-6) If ρ(s) = 0 for a state s, then Assumption [5](#page-20-2) implies that dπ,p(s) = (1 − γ) P<sup>∞</sup> <sup>t</sup>=0 γ <sup>t</sup>Pπ,p,ρ{s<sup>t</sup> = s} = 0 for any π ∈ Π and p ∈ P, which means the state s will never be visited. Therefore, we can exclude all such states s from S such that Assumption [4](#page-20-1) holds, which implies Assumption [3](#page-3-6) as proved above.

Proof of Assumption [6](#page-20-3)⇒Assumption [3:](#page-3-6) Eq. [\(79\)](#page-20-4) implies that for any n ∈ <sup>N</sup>+, we have

$$d_{\text{TV}} [\mathbb{P}_{\pi, p, \rho}(s_{nt_{\text{mix}}} = \cdot), \mu_{\pi, p}] = \frac{1}{2} \sum_s |\mathbb{P}_{\pi, p, \rho}\{s_{nt_{\text{mix}}} = s\} - \mu_{\pi, p}(s)| \leq \frac{1}{4^n}.$$

Select n = ⌈log(µ −1 min)/ log 4⌉. Then the bound above implies |<sup>P</sup>π,p,ρ{sntmix = s} − µπ,p(s)| ≤ µmin/2 for any state s, which along with µπ,p(s) ≥ µmin implies that <sup>P</sup>π,p,ρ{sntmix = s} ≥ µmin/2. Therefore, we can prove Assumption [3](#page-3-6) as follows.

$$d_{\pi,p}(s) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}_{\pi,p,\rho}\{s_t = s\} \geq (1 - \gamma) \gamma^{n_{\text{tmix}}} \mathbb{P}_{\pi,p,\rho}\{s_{n_{\text{tmix}}} = s\} \geq \frac{\mu_{\text{min}}}{2} \gamma^{n_{\text{tmix}}} (1 - \gamma).$$

# D. Proof of Theorem [1](#page-3-1)

Fix any π0, π<sup>1</sup> ∈ Π. For any α ∈ [0, 1], denote d<sup>α</sup> = αd<sup>π</sup>1,pπ<sup>1</sup> + (1 − α)d<sup>π</sup>0,pπ<sup>0</sup> , <sup>π</sup>α(a|s) = <sup>d</sup>α(s,a) <sup>d</sup>α(s) where <sup>d</sup>α(s) = P <sup>a</sup>′ <sup>d</sup>α(s, a′ ), and p<sup>α</sup> = p<sup>π</sup><sup>α</sup> . It can be easily verified that d<sup>0</sup> = d<sup>π</sup>0,p<sup>0</sup> , d<sup>1</sup> = d<sup>π</sup>1,p<sup>1</sup> and d<sup>α</sup> = αd<sup>0</sup> + (1 − α)d1. Then we can obtain the following derivatives and their bounds about πα, d<sup>α</sup> in Eqs. [\(80\)](#page-21-1)-[\(86\)](#page-23-2).

$$\begin{aligned} \frac{d}{d\alpha}\pi_\alpha(a|s) &= \frac{d_\alpha(s)[d_1(s, a) - d_0(s, a)] - d_\alpha(s, a)[d_1(s) - d_0(s)]}{d_\alpha^2(s)} \\ &= \frac{[\alpha d_1(s) + (1 - \alpha)d_0(s)][d_1(s, a) - d_0(s, a)] - [\alpha d_1(s, a) + (1 - \alpha)d_0(s, a)][d_1(s) - d_0(s)]}{d_\alpha^2(s)} \\ &= \frac{d_0(s)d_1(s, a) - d_0(s, a)d_1(s)}{d_\alpha^2(s)} \\ &= \frac{d_0(s)d_1(s)[\pi_1(a|s) - \pi_0(a|s)]}{d_\alpha^2(s)}. \end{aligned} \tag{80}$$

Hence,

$$\begin{aligned} \left\| \frac{d\pi_\alpha}{d\alpha} \right\|^2 &= \sum_{s,a} \left| \frac{d_0(s)d_1(s)[\pi_1(a|s) - \pi_0(a|s)]}{d_\alpha^2(s)} \right|^2 \\ &\stackrel{(a)}{\leq} \sum_{s,a} \left[ \frac{\max[d_0(s), d_1(s)] \min[d_0(s), d_1(s)]}{\min^2[d_0(s), d_1(s)]} \right]^2 [\pi_1(a|s) - \pi_0(a|s)]^2 \\ &\stackrel{(b)}{\leq} D^{-2} \sum_{s,a} [\pi_1(a|s) - \pi_0(a|s)]^2 \leq D^{-2} \|\pi_1 - \pi_0\|^2, \end{aligned} \quad (81)$$

1216 Hence,

1218 1219

1224

1226

1229

1234

1236

1254

1256

1259 1260

where (a) uses dα(s) = αd1(s) + (1 − α)d0(s) ≥ min[d0(s), d1(s)] and (b) uses Assumption [3.](#page-3-6) Then by taking derivative of Eq. [\(80\)](#page-21-1), we have

$$\frac{d^2}{d\alpha^2}\pi_\alpha(a|s) = -\frac{2d_0(s)d_1(s)[\pi_1(a|s) - \pi_0(a|s)][d_1(s) - d_0(s)]}{d_\alpha^3(s)}. \quad (82)$$

$$\begin{aligned} \left\| \frac{d^2 \pi_\alpha}{d\alpha^2} \right\|^2 &= \sum_{s,a} \left| \frac{2d_0(s)d_1(s)[\pi_1(a|s) - \pi_0(a|s)][d_1(s) - d_0(s)]}{[\alpha d_1(s) + (1 - \alpha)d_0(s)]^3} \right|^2 \\ &\stackrel{(a)}{\leq} \sum_{s,a} \left[ \frac{2 \max[d_0(s), d_1(s)] \min[d_0(s), d_1(s)] |d_1(s) - d_0(s)|}{D^2 \min[d_0(s), d_1(s)]} \right]^2 [\pi_1(a|s) - \pi_0(a|s)]^2 \\ &\leq (2D^{-2})^2 \max_s [|d_1(s) - d_0(s)|^2] \sum_{s,a} [\pi_1(a|s) - \pi_0(a|s)]^2 \\ &\leq (2D^{-2})^2 \|\pi_1 - \pi_0\|^2 \left[ \sum_s |d_1(s) - d_0(s)| \right]^2 \\ &\stackrel{(b)}{\leq} (2D^{-2})^2 \|\pi_1 - \pi_0\|^2 \left[ \frac{\gamma \sqrt{|\mathcal{A}|}}{1 - \gamma} \|\pi_1 - \pi_0\| + \frac{\gamma \sqrt{|\mathcal{S}|}}{1 - \gamma} \|p_{\pi_1} - p_{\pi_0}\| \right]^2 \\ &\stackrel{(c)}{\leq} (2D^{-2})^2 \|\pi_1 - \pi_0\|^2 \left[ \frac{\gamma \sqrt{|\mathcal{A}|}}{1 - \gamma} \|\pi_1 - \pi_0\| + \frac{\gamma \epsilon_p \sqrt{|\mathcal{S}|}}{1 - \gamma} \|\pi_1 - \pi_0\| \right]^2 \\ &\leq (2D^{-2})^2 \|\pi_1 - \pi_0\|^4 \left[ \frac{\gamma(\epsilon_p \sqrt{|\mathcal{S}|} + \sqrt{|\mathcal{A}|})}{1 - \gamma} \right]^2, \end{aligned} \tag{83}$$

where (a) uses dα(s) = αd1(s) + (1 − α)d0(s) ≥ min[d0(s), d1(s)] ≥ D, (b) uses Lemma [3,](#page-11-4) and (c) uses Assumption [1.](#page-3-4)

$$\begin{aligned}
& d_0(s)d_1(s) \left| \frac{d}{d\alpha} \left[ \frac{d_\alpha(s, a)}{d_\alpha^2(s)} \right] \right| \\
&= \left| \frac{d_0(s)d_1(s)}{d_\alpha^2(s)} [d_1(s, a) - d_0(s, a)] - \frac{2d_0(s)d_1(s)d_\alpha(s, a)}{d_\alpha^3(s)} [d_1(s) - d_0(s)] \right| \\
&\leq \frac{d_0(s)d_1(s)}{d_\alpha^2(s)} \left[ |d_1(s, a) - d_0(s, a)| + \frac{2d_\alpha(s, a)}{d_\alpha(s)} |d_1(s) - d_0(s)| \right] \\
&\leq \frac{\max[d_0(s), d_1(s)] \min[d_0(s), d_1(s)]}{\min^2[d_0(s), d_1(s)]} [|d_1(s, a) - d_0(s, a)| + 2\pi_\alpha(a|s)|d_1(s) - d_0(s)|] \\
&\leq D^{-1} [|d_1(s, a) - d_0(s, a)| + 2\pi_\alpha(a|s)|d_1(s) - d_0(s)|]. \tag{84}
\end{aligned}$$

$$\begin{aligned} & \frac{d}{d\alpha}[d_\alpha(s, a)p_\alpha(s'|s, a)] \\ &= p_\alpha(s'|s, a)[d_1(s, a) - d_0(s, a)] + d_\alpha(s, a) \cdot \frac{d}{d\alpha}\pi_\alpha(a|s) \cdot \nabla_\pi p_{\pi_\alpha}(s'|s, a) \\ &= p_\alpha(s'|s, a)[d_1(s, a) - d_0(s, a)] + \frac{d_\alpha(s, a)d_0(s)d_1(s)[\pi_1(a|s) - \pi_0(a|s)]}{d_\alpha^2(s)} \cdot \nabla_\pi p_{\pi_\alpha}(s'|s, a) \end{aligned} \quad (85)$$

Then for any α, α′ ∈ [0, 1], we have

$$\begin{aligned} & \left| \frac{d}{d\alpha} [d_{\alpha'}(s, a) p_{\alpha'}(s'|s, a)] - \frac{d}{d\alpha} [d_{\alpha}(s, a) p_{\alpha}(s'|s, a)] \right| \\ & \stackrel{(a)}{\leq} |p_{\alpha'}(s'|s, a) - p_{\alpha}(s'|s, a)| \cdot |d_1(s, a) - d_0(s, a)| + d_0(s) d_1(s) |\pi_1(a|s) - \pi_0(a|s)| \cdot \\ & \left[ \left| \frac{d_{\alpha'}(s, a)}{d_{\alpha'}^2(s)} \left| \|\nabla_{\pi} p_{\pi_{\alpha'}}(s'|s, a) - \nabla_{\pi} p_{\pi_{\alpha}}(s'|s, a)\| + \left| \frac{d_{\alpha'}(s, a)}{d_{\alpha'}^2(s)} - \frac{d_{\alpha}(s, a)}{d_{\alpha}^2(s)} \right| \|\nabla_{\pi} p_{\pi_{\alpha}}(s'|s, a)\| \right] \right. \right] \end{aligned}$$

1269

1274

1276

1279

1289 1290 Therefore, the error term eα(s) satisfies the following recursion.

1294

1296

1299

$$\begin{aligned} & 1300 \quad \sum_{s'} |e_\alpha(s')| \\ & 1302 \quad \leq \gamma \sum_{s, a, s'} [|e_\alpha(s)| \pi_\alpha(a|s) p_\alpha(s'|s, a) + |d_\alpha(s, a) p_\alpha(s'|s, a) - \alpha d_1(s, a) p_1(s'|s, a) - (1 - \alpha) d_0(s, a) p_0(s'|s, a)|] \\ & 1304 \quad \\ & 1305 \quad \stackrel{(a)}{\leq} \gamma \sum_s |e_\alpha(s)| + \frac{\gamma \alpha (1 - \alpha)}{2} \sum_{s, a, s'} \ell_{dp}(s, a) \\ & 1307 \quad \\ & 1308 \quad \stackrel{(b)}{\leq} \gamma \sum_s |e_\alpha(s)| + \frac{\gamma |\mathcal{S}| \alpha (1 - \alpha)}{2} \left[ 2D^{-1} \epsilon_p \|\pi_1 - \pi_0\| \sum_{s, a} |d_1(s, a) - d_0(s, a)| + 4D^{-1} \epsilon_p \|\pi_1 - \pi_0\|_\infty \sum_s |d_1(s) - d_0(s)| \right. \\ & 1310 \quad \\ & \quad \left. + 4D^{-1} S_p \|\pi_1 - \pi_0\|_\infty \cdot \|\pi_1 - \pi_0\| \right] \\ & 1312 \quad \\ & 1313 \quad \stackrel{(c)}{\leq} \gamma \sum_s |e_\alpha(s)| + \frac{\gamma |\mathcal{S}| \alpha (1 - \alpha)}{2} \left[ 6D^{-1} \epsilon_p \|\pi_1 - \pi_0\| \cdot \frac{1}{1 - \gamma} \left( \sqrt{|\mathcal{A}|} \|\pi_1 - \pi_0\| + \gamma \sqrt{|\mathcal{S}|} \|p_{\pi_1} - p_{\pi_0}\| \right) + 4D^{-1} S_p \|\pi_1 - \pi_0\|^2 \right] \\ & 1314 \quad \\ & 1315 \quad \stackrel{(d)}{\leq} \gamma \sum_s |e_\alpha(s)| + 3D^{-1} \gamma |\mathcal{S}| \alpha (1 - \alpha) \|\pi_1 - \pi_0\|^2 \left[ \frac{\epsilon_p}{1 - \gamma} (\sqrt{|\mathcal{A}|} + \gamma \epsilon_p \sqrt{|\mathcal{S}|}) + S_p \right], \\ & 1317 \quad \end{aligned}$$

$$\begin{aligned}
& \stackrel{(b)}{\leq} \epsilon_p \|\pi_{\alpha'} - \pi_\alpha\| |d_1(s, a) - d_0(s, a)| \\
& \quad + \pi_{\alpha'}(a|s) |\pi_1(a|s) - \pi_0(a|s)| \cdot \frac{\max[d_0(s), d_1(s)] \min[d_0(s), d_1(s)]}{\min[d_0(s), d_1(s)]} \cdot S_p \|\pi_{\alpha'} - \pi_\alpha\| \\
& \quad + D^{-1} \epsilon_p |\pi_1(a|s) - \pi_0(a|s)| \cdot [|d_1(s, a) - d_0(s, a)| + 2\pi_\alpha(a|s)|d_1(s) - d_0(s)|] \cdot |\alpha' - \alpha| \\
& \stackrel{(c)}{\leq} \epsilon_p D^{-1} \|\pi_1 - \pi_0\| \cdot |\alpha' - \alpha| \cdot |d_1(s, a) - d_0(s, a)| \\
& \quad + S_p \pi_{\alpha'}(a|s) \cdot |\pi_1(a|s) - \pi_0(a|s)| \cdot [d_0(s) + d_1(s)] \cdot D^{-1} \|\pi_1 - \pi_0\| \cdot |\alpha' - \alpha| \\
& \quad + D^{-1} \epsilon_p |\pi_1(a|s) - \pi_0(a|s)| \cdot [|d_1(s, a) - d_0(s, a)| + 2\pi_\alpha(a|s)|d_1(s) - d_0(s)|] \cdot |\alpha' - \alpha| \\
& \stackrel{(d)}{\leq} \ell_{dp}(s, a) |\alpha' - \alpha|, \tag{86}
\end{aligned}$$

where (a) uses Eq. [\(85\)](#page-22-0), (b) uses Assumptions [1-](#page-3-4)[2,](#page-3-5) dα′ (s, a) = dα′ (s)πα′ (a|s), dα′ (s) = α ′d1(s) + (1 − α ′ )d0(s) ≥ min[d0(s), d1(s)] and Eq. [\(84\)](#page-22-1), (c) uses Assumption [3](#page-3-6) as well as Eq. [\(81\)](#page-21-2), (d) defines ℓdp(s, a) as the following Eq. [\(87\)](#page-23-1) and uses <sup>π</sup>α(a|s) = αd1(s)π1(a|s)+(1−α)d0(s)π0(a|s) αd1(s)+(1−α)d0(s) ≤ <sup>π</sup>0(a|s) + <sup>π</sup>1(a|s).

$$\begin{aligned} \ell_{dp}(s, a) = & 2D^{-1}\epsilon_p\|\pi_1 - \pi_0\| |d_1(s, a) - d_0(s, a)| + 2D^{-1}\epsilon_p[\pi_1(a|s) + \pi_0(a|s)] \cdot |\pi_1(a|s) - \pi_0(a|s)| \cdot |d_1(s) - d_0(s)| \\ & + D^{-1}S_p[\pi_1(a|s) + \pi_0(a|s)] \cdot |\pi_1(a|s) - \pi_0(a|s)| \cdot \|\pi_1 - \pi_0\| \cdot [d_0(s) + d_1(s)]. \end{aligned} \quad (87)$$

Denote eα(s) = d<sup>π</sup>α,p<sup>α</sup> (s) − dα(s) as the error term due to the policy-dependent transition kernel p<sup>α</sup> = p<sup>π</sup><sup>α</sup> . Note that the occupancy measure [\(2\)](#page-2-7) satisfies that the Bellman equation [\(3\)](#page-2-2) repeated as follows.

$$d_{\pi,p}(s') = (1 - \gamma)\rho(s') + \gamma \sum_{s,a} d_{\pi,p}(s)\pi(a|s)p(s'|s,a), \quad s' \in \mathcal{S}. \quad (88)$$

$$\begin{aligned}
& e_\alpha(s') \\
&= d_{\pi_\alpha, p_\alpha}(s') - \alpha d_1(s') - (1 - \alpha) d_0(s') \\
&= \gamma \sum_{s, a} [d_{\pi_\alpha, p_\alpha}(s) \pi_\alpha(a|s) p_\alpha(s'|s, a) - \alpha d_{\pi_1, p_1}(s) \pi_1(a|s) p_1(s'|s, a) - (1 - \alpha) d_{\pi_0, p_0}(s) \pi_0(a|s) p_0(s'|s, a)] \\
&= \gamma \sum_{s, a} [e_\alpha(s) \pi_\alpha(a|s) p_\alpha(s'|s, a) + d_\alpha(s, a) p_\alpha(s'|s, a) - \alpha d_1(s, a) p_1(s'|s, a) - (1 - \alpha) d_0(s, a) p_0(s'|s, a)]. \tag{89}
\end{aligned}$$

The above inequality implies that

<sup>1</sup> If p<sup>π</sup><sup>α</sup> ≡ p does not depend on the policy πα, it can be easily verified that eα(s) = 0 for all s ∈ S.

1326

$$\begin{aligned}
& 1329 \quad J_\lambda(\pi_\alpha, \pi_\alpha, p_\alpha, r) - \alpha J_\lambda(\pi_1, \pi_1, p_1, r) - (1 - \alpha) J_\lambda(\pi_0, \pi_0, p_0, r) \\
& 1330 \\
& 1331 \quad \stackrel{(a)}{=} \frac{1}{1 - \gamma} \sum_{s, a} \left[ d_{\pi_\alpha, p_\alpha}(s, a) [r(s, a) - \lambda \log \pi_\alpha(a|s)] - \alpha d_1(s, a) [r(s, a) - \lambda \log \pi_1(a|s)] \right] \\
& 1332 \\
& 1333 \quad - (1 - \alpha) d_0(s, a) [r(s, a) - \lambda \log \pi_0(a|s)] \\
& 1335 \\
& 1336 \quad = \frac{1}{1 - \gamma} \sum_{s, a} \left[ [d_{\pi_\alpha, p_\alpha}(s, a) - d_\alpha(s, a)] [r(s, a) - \lambda \log \pi_\alpha(a|s)] \right] \\
& 1337 \\
& 1338 \quad + d_\alpha(s, a) [r(s, a) - \lambda \log \pi_\alpha(a|s)] - \alpha d_1(s, a) [r(s, a) - \lambda \log \pi_1(a|s)] - (1 - \alpha) d_0(s, a) [r(s, a) - \lambda \log \pi_0(a|s)] \\
& 1339 \\
& 1340 \quad \stackrel{(b)}{=} \frac{1}{1 - \gamma} \sum_{s, a} [d_{\pi_\alpha, p_\alpha}(s) - d_\alpha(s)] \pi_\alpha(a|s) [r(s, a) - \lambda \log \pi_\alpha(a|s)] \\
& 1341 \\
& 1342 \quad + \frac{\lambda}{1 - \gamma} \sum_{s, a} \left[ \alpha d_1(s, a) \log \frac{\pi_1(a|s)}{\pi_\alpha(a|s)} + (1 - \alpha) d_0(s, a) \log \frac{\pi_0(a|s)}{\pi_\alpha(a|s)} \right] \\
& 1343 \\
& 1344 \\
& 1345 \quad \stackrel{(c)}{\geq} -\frac{1 + \lambda \log |\mathcal{A}|}{1 - \gamma} \sum_s |e_\alpha(s)| + \frac{\lambda}{1 - \gamma} \sum_s \left[ \alpha d_1(s) \sum_a \left( \pi_1(a|s) \log \frac{\pi_1(a|s)}{\pi_\alpha(a|s)} \right) + (1 - \alpha) d_0(s) \sum_a \left( \pi_0(a|s) \log \frac{\pi_0(a|s)}{\pi_\alpha(a|s)} \right) \right] \\
& 1346 \\
& 1347 \\
& 1348 \quad \stackrel{(d)}{\geq} -\frac{1 + \lambda \log |\mathcal{A}|}{1 - \gamma} \frac{3\gamma |\mathcal{S}| \alpha (1 - \alpha)}{D(1 - \gamma)^2} \|\pi_1 - \pi_0\|^2 [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma \epsilon_p \sqrt{|\mathcal{S}|}) + S_p(1 - \gamma)] \\
& 1349 \\
& 1350 \quad + \frac{\lambda}{1 - \gamma} \sum_s \left[ \alpha d_1(s) \text{KL}[\pi_1(\cdot|s) \| \pi_\alpha(\cdot|s)] + (1 - \alpha) d_0(s) \text{KL}[\pi_0(\cdot|s) \| \pi_\alpha(\cdot|s)] \right] \\
& 1351 \\
& 1352 \\
& 1353 \quad \stackrel{(e)}{\geq} -\frac{3\gamma |\mathcal{S}| \alpha (1 - \alpha) (1 + \lambda \log |\mathcal{A}|)}{D(1 - \gamma)^3} \|\pi_1 - \pi_0\|^2 [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma \epsilon_p \sqrt{|\mathcal{S}|}) + S_p(1 - \gamma)] \\
& 1354 \\
& 1355 \quad + \frac{\lambda}{2(1 - \gamma)} \sum_s \left[ \alpha d_1(s) \|\pi_1(\cdot|s) - \pi_\alpha(\cdot|s)\|_1^2 + (1 - \alpha) d_0(s) \|\pi_0(\cdot|s) - \pi_\alpha(\cdot|s)\|_1^2 \right] \\
& 1356 \\
& 1357 \\
& 1358 \quad \stackrel{(f)}{=} -\frac{3\gamma |\mathcal{S}| \alpha (1 - \alpha) (1 + \lambda \log |\mathcal{A}|)}{D(1 - \gamma)^3} \|\pi_1 - \pi_0\|^2 [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma \epsilon_p \sqrt{|\mathcal{S}|}) + S_p(1 - \gamma)] \\
& 1359 \\
& 1360 \quad + \frac{\lambda}{2(1 - \gamma)} \sum_s \left[ \alpha d_1(s) \left\| \frac{(1 - \alpha) d_0(s)}{d_\alpha(s)} [\pi_1(\cdot|s) - \pi_0(\cdot|s)] \right\|_1^2 + (1 - \alpha) d_0(s) \left\| \frac{\alpha d_1(s)}{d_\alpha(s)} [\pi_1(\cdot|s) - \pi_0(\cdot|s)] \right\|_1^2 \right] \\
& 1361 \\
& 1362 \\
& 1363 \quad \stackrel{(g)}{=} \frac{\lambda \alpha (1 - \alpha)}{2(1 - \gamma)} \sum_s \frac{d_0(s) d_1(s)}{d_\alpha(s)} \|\pi_1(\cdot|s) - \pi_0(\cdot|s)\|_1^2 \\
& 1364 \\
& 1365 \quad - \frac{3\gamma |\mathcal{S}| \alpha (1$$

1374 where (a) uses Eq. [\(39\)](#page-12-1), (b) uses d<sup>π</sup>α,p<sup>α</sup> (s, a) = d<sup>π</sup>α,p<sup>α</sup> (s)πα(a|s), dα(s, a) = dα(s)πα(a|s) and d<sup>α</sup> = αd<sup>1</sup> + (1 − α)d0, (c) uses r(s, a) ∈ [0, 1], − P a πα(a|s) log πα(a|s) ∈ [0, log |A|] and eα(s) = d<sup>π</sup>α,p<sup>α</sup> (s) − dα(s), (d) uses Eq. [\(90\)](#page-24-1), (e)

where (a) uses Eq. [\(86\)](#page-23-2) which implies that dα(s, a)pα(s ′ |s, a) is a Lipschitz smooth function with Lipschitz constant ℓdp(s, a) defined by Eq. [\(87\)](#page-23-1), (b) uses Eq. [\(87\)](#page-23-1), (c) uses ∥π<sup>1</sup> − π0∥<sup>∞</sup> ≤ ∥π<sup>1</sup> − π0∥ and Lemma [3,](#page-11-4) and (d) uses Assumption [1.](#page-3-4) Rearranging the above inequality, we get

$$\sum_s |e_\alpha(s)| \leq \frac{3\gamma |\mathcal{S}| \alpha (1-\alpha)}{D(1-\gamma)^2} \|\pi_1 - \pi_0\|^2 [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma \epsilon_p \sqrt{|\mathcal{S}|}) + S_p(1-\gamma)]. \quad (90)$$

Therefore, for any reward function r, we have

1379

1389 1390

1394

1396

1422 1423 1424 1425 where (a) uses Assumptions [1-](#page-3-4)[2,](#page-3-5) ∥∇rJλ(·, ·, ·, ·)∥ ≤ <sup>1</sup> 1−γ (implied by Eq. [\(50\)](#page-13-4)) as well as Eqs. [\(54\)](#page-13-6), [\(81\)](#page-21-2) and [\(83\)](#page-22-2), (b) uses Eq. [\(81\)](#page-21-2) and ∥x∥<sup>1</sup> ≤ √ d∥x∥ for any x ∈ <sup>R</sup> d , (c) uses Assumption [1,](#page-3-4) and (d) uses D, γ ∈ [0, 1]. The inequality above implies that w(α) is µ2∥π<sup>1</sup> − π0∥ 2 -Lipschitz smooth with the constant µ<sup>2</sup> defined as follows.

uses Pinsker's inequality, (f) uses <sup>π</sup>α(a|s) = <sup>d</sup>α(s,a) <sup>d</sup>α(s) = αd1(s) dα(s) <sup>π</sup>1(a|s) + (1−α)d0(s) dα(s) π0(a|s), (g) uses dα(s) = αd1(s) + (1 − α)d0(s), (h) uses Assumption [3](#page-3-6) and dα(s) ≤ max[d0(s), d1(s)], and (i) defines the constant µ<sup>1</sup> below.

$$\mu_1 \stackrel{\text{def}}{=} \frac{D\lambda}{1-\gamma} - \frac{6\gamma|\mathcal{S}|(1+\lambda\log|\mathcal{A}|)}{D(1-\gamma)^3} [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma\epsilon_p\sqrt{|\mathcal{S}|}) + S_p(1-\gamma)]. \quad (92)$$

Next, we begin to consider the policy-dependent reward r<sup>α</sup> = r<sup>π</sup><sup>α</sup> . Define the function w(α) = αJλ(π1, π1, p1, rα) + (1 − α)Jλ(π0, π0, p0, rα), which has the following derivative

$$\begin{aligned} w'(\alpha) = & J_\lambda(\pi_1, \pi_1, p_1, r_\alpha) - J_\lambda(\pi_0, \pi_0, p_0, r_\alpha) \\ & + [\alpha \nabla_r J_\lambda(\pi_1, \pi_1, p_1, r_\alpha) + (1 - \alpha) \nabla_r J_\lambda(\pi_0, \pi_0, p_0, r_\alpha)] (\nabla_r r_{\pi_\alpha}) \frac{d\pi_\alpha}{d\alpha} \end{aligned} \quad (93)$$

For any 0 ≤ α ≤ α ′ ≤ 1, we prove the smoothness of w(α) as follows.

$$\begin{aligned} & |w'(\alpha') - w'(\alpha)| \\ &= \left| \int_{\alpha}^{\alpha'} \nabla_r [J_{\lambda}(\pi_1, \pi_1, p_1, r_{\tilde{\alpha}}) - J_{\lambda}(\pi_0, \pi_0, p_0, r_{\tilde{\alpha}})] (\nabla_{\pi} r_{\pi_{\tilde{\alpha}}}) \frac{d\pi_{\tilde{\alpha}}}{d\tilde{\alpha}} d\tilde{\alpha}} \right. \\ &\quad + [\alpha' \nabla_r J_{\lambda}(\pi_1, \pi_1, p_1, r_{\alpha'}) + (1 - \alpha') \nabla_r J_{\lambda}(\pi_0, \pi_0, p_0, r_{\alpha'})] (\nabla_{\pi} r_{\pi_{\alpha'}}) \left( \frac{d\pi_{\alpha'}}{d\alpha'} - \frac{d\pi_{\alpha}}{d\alpha} \right) \\ &\quad + [\alpha' \nabla_r J_{\lambda}(\pi_1, \pi_1, p_1, r_{\alpha'}) + (1 - \alpha') \nabla_r J_{\lambda}(\pi_0, \pi_0, p_0, r_{\alpha'})] (\nabla_{\pi} r_{\pi_{\alpha'}} - \nabla_{\pi} r_{\pi_{\alpha}}) \frac{d\pi_{\alpha}}{d\alpha} \\ &\quad + \{ \alpha' [\nabla_r J_{\lambda}(\pi_1, \pi_1, p_1, r_{\alpha'}) - \nabla_r J_{\lambda}(\pi_1, \pi_1, p_1, r_{\alpha})] \\ &\quad + (1 - \alpha') [\nabla_r J_{\lambda}(\pi_0, \pi_0, p_0, r_{\alpha'}) - \nabla_r J_{\lambda}(\pi_0, \pi_0, p_0, r_{\alpha})] \} (\nabla_{\pi} r_{\pi_{\alpha}}) \frac{d\pi_{\alpha}}{d\alpha} \\ &\quad \left. + (\alpha' - \alpha) [\nabla_r J_{\lambda}(\pi_1, \pi_1, p_1, r_{\alpha}) - \nabla_r J_{\lambda}(\pi_0, \pi_0, p_0, r_{\alpha})] (\nabla_{\pi} r_{\pi_{\alpha}}) \frac{d\pi_{\alpha}}{d\alpha} \right| \\ &\stackrel{(a)}{\leq} \int_{\alpha}^{\alpha'} \frac{\epsilon_r \|\pi_1 - \pi_0\|}{D(1 - \gamma)^2} \left( \max_s \|\pi_1(\cdot|s) - \pi_0(\cdot|s)\|_1 + \gamma \max_{s,a} \|p_1(\cdot|s, a) - p_0(\cdot|s, a)\|_1 \right) d\tilde{\alpha} \\ &\quad + \frac{\epsilon_r}{1 - \gamma} \cdot 2D^{-2} \|\pi_1 - \pi_0\|^2 \left[ \frac{\gamma(\epsilon_p \sqrt{|\mathcal{S}|} + \sqrt{|\mathcal{A}|})}{1 - \gamma} \right] |\alpha' - \alpha| + \frac{S_r \|\pi_{\alpha'} - \pi_{\alpha}\|}{1 - \gamma} \cdot D^{-1} \|\pi_1 - \pi_0\| + 0 \\ &\quad + |\alpha' - \alpha| \cdot \frac{\epsilon_r \|\pi_1 - \pi_0\|}{D(1 - \gamma)^2} \left( \max_s \|\pi_1(\cdot|s) - \pi_0(\cdot|s)\|_1 + \gamma \max_{s,a} \|p_1(\cdot|s, a) - p_0(\cdot|s, a)\|_1 \right) \\ &\stackrel{(b)}{\leq} 2|\alpha' - \alpha| \cdot \frac{\epsilon_r \|\pi_1 - \pi_0\|}{D(1 - \gamma)^2} (\sqrt{|\mathcal{A}|} \|\pi_1 - \pi_0\| + \gamma \sqrt{|\mathcal{S}|} \|p_1 - p_0\|) \\ &\quad + \frac{2\epsilon_r \|\pi_1 - \pi_0\|^2}{D^2(1 - \gamma)} \left[ \frac{\gamma(\epsilon_p \sqrt{|\mathcal{S}|} + \sqrt{|\mathcal{A}|})}{1 - \gamma} \right] |\alpha' - \alpha| + \frac{S_r \|\pi_1 - \pi_0\|^2}{D^2(1 - \gamma)} |\alpha' - \alpha| \\ &\stackrel{(c)}{\leq} \frac{2\epsilon_r \|\pi_1 - \pi_0\|}{D(1 - \gamma)^2} (\sqrt{|\mathcal{A}|} \|\pi_1 - \pi_0\| + \gamma \epsilon_p \sqrt{|\mathcal{S}|} \|\pi_1 - \pi_0\|) |\alpha' - \alpha| \\ &\quad + \frac{2\gamma \epsilon_r \|\pi_1 - \pi_0\|^2}{D^2(1 - \gamma)^2} (\sqrt{|\mathcal{A}|} + \epsilon_p \sqrt{|\mathcal{S}|}) |\alpha' - \alpha| + \frac{S_r(1 - \gamma) \|\pi_1 - \pi_0\|^2}{D^2(1 - \gamma)^2} |\alpha' - \alpha| \\ &\stackrel{(d)}{\leq} \frac{4\epsilon_r(\sqrt{|\mathcal{A}|} + \gamma \epsilon_p \sqrt{|\mathcal{S}|}) + S_r(1 - \gamma)}{D^2(1 - \gamma)^2} \|\pi_1 - \pi_0\|^2 |\alpha' - \alpha|, \end{aligned}$$

$$\mu_2 = \frac{4\epsilon_r(\sqrt{|\mathcal{A}|} + \epsilon_p\sqrt{|\mathcal{S}|}) + S_r(1 - \gamma)}{D^2(1 - \gamma)^2} \quad (94)$$

Therefore,

$$\begin{aligned}
& V_{\lambda, \pi_\alpha}^{\pi_\alpha} - \alpha V_{\lambda, \pi_1}^{\pi_1} - (1 - \alpha) V_{\lambda, \pi_0}^{\pi_0} \\
&= J_\lambda(\pi_\alpha, \pi_\alpha, p_\alpha, r_\alpha) - \alpha J_\lambda(\pi_1, \pi_1, p_1, r_1) - (1 - \alpha) J_\lambda(\pi_0, \pi_0, p_0, r_0) \\
&\stackrel{(a)}{\geq} \alpha J_\lambda(\pi_1, \pi_1, p_1, r_\alpha) + (1 - \alpha) J_\lambda(\pi_0, \pi_0, p_0, r_\alpha) + \frac{\mu_1 \alpha (1 - \alpha)}{2} \|\pi_1 - \pi_0\|^2 \\
&\quad - \alpha J_\lambda(\pi_1, \pi_1, p_1, r_1) - (1 - \alpha) J_\lambda(\pi_0, \pi_0, p_0, r_0) \\
&= w(\alpha) - \alpha w(1) - (1 - \alpha) w(0) + \frac{\mu_1 \alpha (1 - \alpha)}{2} \|\pi_1 - \pi_0\|^2 \\
&\stackrel{(b)}{\geq} \frac{(\mu_1 - \mu_2) \alpha (1 - \alpha)}{2} \|\pi_1 - \pi_0\|^2 \\
&\stackrel{(c)}{=} \frac{\mu \alpha (1 - \alpha)}{2} \|\pi_1 - \pi_0\|^2, 
\end{aligned} \tag{95}$$

where (a) uses Eq. [\(91\)](#page-24-0) with r replaced by rα, (b) uses the fact proved above that w(α) is µ2∥π<sup>1</sup> − π0∥ 2 -Lipschitz smooth, and (c) defines the following constant µ which is the same as Eq. [\(13\)](#page-3-8).

$$\mu \stackrel{\text{def}}{=} \mu_1 - \mu_2 \quad , \quad (a) \quad \frac{D\lambda}{1-\gamma} - \frac{6\gamma|\mathcal{S}|(1+\lambda \log|\mathcal{A}|)}{D(1-\gamma)^3} [\epsilon_p(\sqrt{|\mathcal{A}|} + \gamma\epsilon_p\sqrt{|\mathcal{S}|}) + S_p(1-\gamma)] - \frac{S_r(1-\gamma) + 4\epsilon_r(\sqrt{|\mathcal{A}|} + \epsilon_p\sqrt{|\mathcal{S}|})}{D^2(1-\gamma)^2},$$

where (a) uses Eqs. [\(92\)](#page-25-0) and [\(94\)](#page-25-1). Rearranging Eq. [\(95\)](#page-26-1), we obtain that

$$\frac{V_{\lambda, \pi_\alpha}^{\pi_\alpha} - V_{\lambda, \pi_0}^{\pi_0}}{\alpha} \geq V_{\lambda, \pi_1}^{\pi_1} - V_{\lambda, \pi_0}^{\pi_0} + \frac{\mu(1 - \alpha)}{2} \|\pi_1 - \pi_0\|^2.$$

Letting α → +0 above, we can prove the conclusion as follows.

$$\begin{aligned} & V_{\lambda, \pi_1}^{\pi_1} - V_{\lambda, \pi_0}^{\pi_0} + \frac{\mu}{2} \|\pi_1 - \pi_0\|^2 \\ & \leq \left[ \frac{d}{d\alpha} V_{\lambda, \pi_\alpha}^{\pi_\alpha} \right]_{\alpha=0} \\ & \leq \sum_{s, a} \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a)} \left[ \frac{d}{d\alpha} \pi_\alpha(a|s) \right]_{\alpha=0} \\ & \stackrel{(a)}{=} \sum_s \frac{d_1(s)}{d_0(s)} \sum_a \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a)} [\pi_1(a|s) - \pi_0(a|s)] \\ & \leq \sum_s \frac{d_1(s)}{d_0(s)} \left[ \max_{a'} \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a')} - \sum_a \pi_0(a|s) \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a)} \right] \\ & \stackrel{(b)}{\leq} D^{-1} \sum_{s, a} \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a)} [\pi_0^*(a|s) - \pi_0(a|s)] \\ & \leq D^{-1} \frac{1}{\pi \in \Pi} \langle \nabla_{\pi_0} V_{\lambda, \pi_0}^{\pi_0}, \pi - \pi_0 \rangle, \end{aligned}$$

where (a) uses Eq. [\(80\)](#page-21-1), and (b) uses Assumption [3](#page-3-6) as well as the following Eq. [\(96\)](#page-26-2) where π ∗ <sup>0</sup> ∈ Π is defined as π ∗ 0 (a ∗ |s) = 1 for a certain a <sup>∗</sup> ∈ arg maxa′ ∂V <sup>π</sup><sup>0</sup> λ,π0 ∂π0(s,a′) and π ∗ 0 (a ′ |s) = 0 for a ′ ̸= a ∗ .

$$\sum_a \pi_0^*(a|s) \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a)} = \max_{a'} \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a')} \geq \sum_a \pi_0(a|s) \frac{\partial V_{\lambda, \pi_0}^{\pi_0}}{\partial \pi_0(s, a)}. \quad (96)$$

# E. Proof of Corollary [1](#page-4-1)

Based on Theorem [1,](#page-3-1) Eq. [\(12\)](#page-3-7) holds for any π0, π<sup>1</sup> ∈ Π as repeated below.

$$V_{\lambda,\pi_1}^{\pi_1} \leq V_{\lambda,\pi_0}^{\pi_0} + D^{-1} \max_{\pi \in \Pi} \langle \nabla_{\pi_0} V_{\lambda,\pi_0}^{\pi_0}, \pi - \pi_0 \rangle - \frac{\mu}{2} \|\pi_1 - \pi_0\|^2, \quad (97)$$

1490 1491 1492 If µ ≥ 0, the inequality above further implies that maxπ˜∈<sup>Π</sup> V π˜ λ,π˜ − V π λ,π ≤ ϵ, that is, the Dϵ-stationary policy π is also an ϵ-PO policy.

1498 1499 1500 Substituting the two equalities above into Eq. [\(97\)](#page-26-3), we obtain that <sup>µ</sup> 2 ∥π<sup>1</sup> − π0∥ <sup>2</sup> ≤ 0, which along with µ > 0 implies π<sup>1</sup> = π0, that is, the PO policy is unique.

1504

1506

1509

1518 1519

1524

1526

1529

1534

1536

In the above inequality, let π<sup>1</sup> ∈ arg maxπ∈ΠV π λ,π and π<sup>0</sup> = π is any a Dϵ-stationary policy of interest. Then the inequality above becomes

$$\max_{\tilde{\pi} \in \Pi} V_{\lambda, \tilde{\pi}}^{\tilde{\pi}} \leq V_{\lambda, \pi}^{\pi} + D^{-1} \cdot D\epsilon - \frac{\mu}{2} \|\pi_1 - \pi\|^2.$$

Furthermore, suppose µ > 0 and there are two PO policies π0, π<sup>1</sup> ∈ Π, which should satisfy

$$V_{\lambda, \pi_1}^{\pi_1} = V_{\lambda, \pi_0}^{\pi_0} = \max_{\pi \in \Pi} V_{\lambda, \pi}^{\pi},$$

$$\max_{\pi \in \Pi} \langle \nabla_{\pi_0} V_{\lambda, \pi_0}^{\pi_0}, \pi - \pi_0 \rangle = 0.$$

# F. Proof of Theorem [2](#page-5-2)

For any π ∈ Π, p ∈ P, r ∈ R, we have

$$\begin{aligned} \frac{\partial J_\lambda(\pi, \pi, p, r)}{\partial \pi(a|s)} & \underline{\underline{(a) d_{\pi,p}(s)[Q_\lambda(\pi, \pi, p, r; s, a) - \lambda]}} \\ & \underline{\underline{= \frac{(b) d_{\pi,p}(s)}{1-\gamma} \left[ r(s, a) - \lambda - \lambda \log \pi(a|s) + \gamma \sum_{s'} p(s'|s, a) V_\lambda(\pi, p, r; s') \right]}}, \end{aligned} \quad (98)$$

where (a) uses Eqs. [\(47\)](#page-12-7), and (b) uses Eq. [\(41\)](#page-12-2).

Then we have

$$\begin{aligned} & \nabla_{\pi} J_{\lambda}(\pi, \pi, p, r)^{\top} (\pi' - \pi) \\ &= \sum_s \left[ \frac{\partial J_{\lambda}(\pi, \pi, p, r)}{\partial \pi[a_{\max}(s)|s]} (\pi'[a_{\max}(s)|s] - \pi[a_{\max}(s)|s]) + \frac{\partial J_{\lambda}(\pi, \pi, p, r)}{\partial \pi[a_{\min}(s)|s]} (\pi'[a_{\min}(s)|s] - \pi[a_{\min}(s)|s]) \right] \\ &= \sum_s \left\{ \frac{d_{\pi, p}(s)}{1 - \gamma} (\pi[a_{\max}(s)|s] - \pi[a_{\min}(s)|s]) \left[ r[s, a_{\min}(s)] - r[s, a_{\max}(s)] + \lambda \log \frac{\pi[a_{\max}(s)|s]}{\pi[a_{\min}(s)|s]} \right. \right. \\ &\quad \left. \left. + \gamma \sum_{s'} [p(s'|s, a_{\min}(s)) - p(s'|s, a_{\max}(s))] V_{\lambda}(\pi, p, r; s') \right] \right\} \\ &\stackrel{(a)}{\geq} \frac{1}{1 - \gamma} \max_s \left\{ (\pi[a_{\max}(s)|s] - \pi[a_{\min}(s)|s]) \left[ \lambda \log \frac{\pi[a_{\max}(s)|s]}{\pi[a_{\min}(s)|s]} - 1 - \frac{\gamma(1 + \lambda \log |\mathcal{A}|)}{1 - \gamma} \right] \right\}, \end{aligned} \quad (99)$$

where (a) uses π[amax(s)|s] − π[amin(s)|s] ≥ 0, r(a|s) ∈ [0, 1], p(s ′ |s, a) ∈ [0, 1] for any s, a, s′ and Lemma [4.](#page-12-9)

Consider the following two cases.

(Case I) If π[amin(s)|s] ≥ 2 π[amax(s)|s], then as π[amax(s)|s] ≥ 1 |A| , we have <sup>π</sup>[amin(s)|s] ≥ 1 <sup>2</sup>|A| .

(Case II) π[amin(s)|s] < π[amax(s)|s], then as π[amax(s)|s] ≥ |A| , Eq. [\(99\)](#page-27-1) implies that

$$\begin{aligned} & \nabla_{\pi} J_{\lambda}(\pi, \pi, p, r)^{\top} (\pi' - \pi) \\ & \geq \max_s \left\{ \frac{\pi[a_{\max}(s)|s]}{2(1-\gamma)} \left[ \lambda \log \frac{1}{|\mathcal{A}|\pi[a_{\min}(s)|s]} - \frac{1 + \gamma \lambda \log |\mathcal{A}|}{1 - \gamma} \right] \right\} \\ & \geq - \frac{1}{2|\mathcal{A}|(1-\gamma)} \left[ \lambda \log (|\mathcal{A}| \min_s \pi[a_{\min}(s)|s]) + \frac{1 + \gamma \lambda \log |\mathcal{A}|}{1 - \gamma} \right], \end{aligned} \quad (100)$$

1554

1556

1559 1560 1561

1564

1566 1567

1569

1574

1576

1579

1589 1590

which further implies that for any s ∈ S and a ∈ A, we have

$$\begin{aligned} \pi(a|s) &\geq \pi[a_{\min}(s)|s] \geq \frac{1}{|\mathcal{A}|} \exp \left[ -\frac{1/\lambda + \gamma \log |\mathcal{A}|}{1 - \gamma} - \frac{2|\mathcal{A}|}{\lambda} (1 - \gamma) \nabla_{\pi} J_{\lambda}(\pi, \pi, p, r)^{\top} (\pi' - \pi) \right] \\ &\geq \frac{1}{2|\mathcal{A}|^{1/(1-\gamma)}} \exp \left[ -\frac{1}{\lambda(1-\gamma)} - \frac{2|\mathcal{A}|}{\lambda} (1 - \gamma) \nabla_{\pi} J_{\lambda}(\pi, \pi, p, r)^{\top} (\pi' - \pi) \right], \end{aligned} \quad (101)$$

Note that in the two cases above, Eq. [\(101\)](#page-28-2) always holds.

Furthermore, if Assumption [1](#page-3-4) holds and pπ, r<sup>π</sup> are differentiable functions of π, then we have

$$\begin{aligned} & \left\| \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) - \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\bar{\pi}}, r_{\bar{\pi}}) \right\|_{\bar{\pi}=\pi} \\ &= \left\| \nabla_p J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) \nabla_{\pi} p_{\pi} + \nabla_r J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) \nabla_{\pi} r_{\pi} \right\| \\ &\leq \left\| \nabla_p J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) \right\| \left\| \nabla_{\pi} p_{\pi} \right\| + \left\| \nabla_r J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) \right\| \left\| \nabla_{\pi} r_{\pi} \right\| \\ &\stackrel{(a)}{\leq} \frac{\epsilon_p \sqrt{|\mathcal{S}|} (1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \frac{\epsilon_r}{1 - \gamma}, \end{aligned} \tag{102}$$

where (a) uses Assumption [1](#page-3-4) as well as Eqs. [\(49\)](#page-13-1) and [\(50\)](#page-13-4). Therefore,

$$\begin{aligned}
& \left[ \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\tilde{\pi}}, r_{\tilde{\pi}}) |_{\tilde{\pi}=\pi} \right]^{\top} (\pi' - \pi) \\
&= \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})^{\top} (\pi' - \pi) - \left[ \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) - \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\tilde{\pi}}, r_{\tilde{\pi}}) |_{\tilde{\pi}=\pi} \right]^{\top} (\pi' - \pi) \\
&\leq \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})^{\top} (\pi' - \pi) + \| \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi}) - \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\tilde{\pi}}, r_{\tilde{\pi}}) |_{\tilde{\pi}=\pi} \| \| \pi' - \pi \| \\
&\stackrel{(a)}{\leq} \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})^{\top} (\pi' - \pi) + \sqrt{2|\mathcal{S}|} \left( \frac{\epsilon_p \sqrt{|\mathcal{S}|} (1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \frac{\epsilon_r}{1 - \gamma} \right),
\end{aligned} \tag{103}$$

where (a) uses Eq. [\(102\)](#page-28-0) and Lemma [12.](#page-19-4) Substituting p = pπ, r = r<sup>π</sup> and then Eq. [\(103\)](#page-28-3) into Eq. [\(101\)](#page-28-2), we can prove Eq. [\(17\)](#page-5-6) as follows.

$$\begin{aligned} \pi(a|s) &\geq \frac{1}{2|\mathcal{A}|^{1/(1-\gamma)}} \exp \left\{ - \frac{1}{\lambda(1-\gamma)} \right. \\ &\quad \left. - \frac{2|\mathcal{A}|}{\lambda} (1-\gamma) \left[ \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})^{\top} (\pi' - \pi) + \sqrt{2|\mathcal{S}|} \left( \frac{\epsilon_p \sqrt{|\mathcal{S}|} (1 + \lambda \log |\mathcal{A}|)}{(1-\gamma)^2} + \frac{\epsilon_r}{1-\gamma} \right) \right] \right\} \\ &= \pi_{\min} \exp \left[ - \frac{2|\mathcal{A}|}{\lambda} (1-\gamma) \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \pi' - \pi \rangle \right], \end{aligned}$$

where the = uses V π λ,π = Jλ(π, π, pπ, rπ) and πmin defined by Eq. [\(18\)](#page-5-3).

# G. Proof of Theorem [3](#page-5-5)

For any policies π, π′ , we have

$$\begin{aligned}
& |V_{\lambda, \pi'}^{\pi'} - V_{\lambda, \pi}^{\pi}| \\
& \leq |J_{\lambda}(\pi', p_{\pi'}, r_{\pi'}) - J_{\lambda}(\pi, p_{\pi}, r_{\pi})| \\
& \leq |J_{\lambda}(\pi', p_{\pi'}, r_{\pi'}) - J_{\lambda}(\pi', p_{\pi'}, r_{\pi})| + |J_{\lambda}(\pi', p_{\pi'}, r_{\pi}) - J_{\lambda}(\pi', p_{\pi}, r_{\pi})| + |J_{\lambda}(\pi', p_{\pi}, r_{\pi}) - J_{\lambda}(\pi, p_{\pi}, r_{\pi})| \\
& \stackrel{(a)}{\leq} \frac{\|r_{\pi'} - r_{\pi}\|}{1 - \gamma} + L_p \|p_{\pi'} - p_{\pi}\| + L_{\pi} \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| \\
& \stackrel{(b)}{\leq} \left( L_p \epsilon_p + \frac{\epsilon_r}{1 - \gamma} \right) \|\pi' - \pi\| + L_{\pi} \sqrt{\sum_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\|^2} \\
& \stackrel{(c)}{\leq} \left( L_p \epsilon_p + \frac{\epsilon_r}{1 - \gamma} \right) \|\log \pi' - \log \pi\| + L_{\pi} \|\log \pi' - \log \pi\| \\
& \stackrel{(d)}{=} L_{\lambda} \|\log \pi' - \log \pi\|, \tag{104}
\end{aligned}$$

1596 where (a) uses Eqs. [\(48\)](#page-13-0), [\(49\)](#page-13-1) and [\(50\)](#page-13-4), (b) uses Assumption [9,](#page-3-9) (c) uses | log y − log x| ≤ |y − x| for any x, y ∈ <sup>R</sup>, and (d) defines the following constant.

1599

1600 1601 Note that for any u, v ≥ ∆ > 0,

1602 1603 1604

1605 Therefore, for any π, π′ ∈ Π<sup>∆</sup> def <sup>=</sup> {<sup>π</sup> ∈ Π : <sup>π</sup>(a|s) ≥ <sup>∆</sup>}, we have

1606 1607 1608

1609 Substituting the above inequality into Eq. [\(104\)](#page-28-4) proves Eq. [\(20\)](#page-5-7).

1614

$$\begin{aligned}
& 16.16 \quad \text{for any } \pi, \pi \in \Pi\Delta, \text{ we prove Eq. (21) as follows.} \\
& 16.17 \quad \|\nabla_{\pi'} V_{\lambda, \pi'}^{\pi'} - \nabla_{\pi} V_{\lambda, \pi}^{\pi}\| \\
& 16.18 \quad \leq \|\nabla_{\pi'} J_{\lambda}(\pi', \pi', p_{\bar{\pi}}, r_{\bar{\pi}})|_{\bar{\pi}=\pi'} - \nabla_{\pi} J_{\lambda}(\pi, \pi, p_{\bar{\pi}}, r_{\bar{\pi}})|_{\bar{\pi}=\pi}\| \\
& 16.19 \quad + \|\nabla_{\pi'} p_{\pi'}\| \cdot \|\nabla_{p_{\pi'}} J_{\lambda}(\pi', \pi', p_{\pi'}, r_{\pi'}) - \nabla_{p_{\pi}} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})\| + \|\nabla_{p_{\pi}} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})\| \cdot \|\nabla_{\pi'} p_{\pi'} - \nabla_{\pi} p_{\pi}\| \\
& 1620 \quad + \|\nabla_{\pi'} r_{\pi'}\| \cdot \|\nabla_{r_{\pi'}} J_{\lambda}(\pi', \pi', p_{\pi'}, r_{\pi'}) - \nabla_{r_{\pi}} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})\| + \|\nabla_{r_{\pi}} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})\| \cdot \|\nabla_{\pi'} r_{\pi'} - \nabla_{\pi} r_{\pi}\| \\
& 1621 \quad + \|\nabla_{\pi'} r_{\pi'}\| \cdot \|\nabla_{r_{\pi'}} J_{\lambda}(\pi', \pi', p_{\pi'}, r_{\pi'}) - \nabla_{r_{\pi}} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})\| + \|\nabla_{r_{\pi}} J_{\lambda}(\pi, \pi, p_{\pi}, r_{\pi})\| \cdot \|\nabla_{\pi'} r_{\pi'} - \nabla_{\pi} r_{\pi}\| \\
& 1622 \quad 16.22 \\
& 1623 \quad \leq \left( \frac{|\mathcal{A}|(1 + 2\lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \gamma L_{\pi} \right) \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + \left[ \frac{2(1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \gamma L_p \right] \sqrt{|\mathcal{S}||\mathcal{A}|} \|p_{\pi'} - p_{\pi}\| \\
& 1624 \quad + \frac{\sqrt{|\mathcal{A}|} \|r_{\pi'} - r_{\pi}\|_{\infty}}{1 - \gamma} + \epsilon_p \left[ \ell_{\pi} \max_s \|\log \pi'(\cdot|s) - \log \pi(\cdot|s)\| + \ell_p \|p_{\pi'} - p_{\pi}\| + \frac{2 - \gamma}{1 - \gamma} \sqrt{|\mathcal{S}|} \|r_{\pi'} - r_{\pi}\|_{\infty} \right] \\
& 1625 \quad + L_p S_p \|\pi' - \pi\| + \frac{\gamma \epsilon_r}{(1 - \gamma)^2} \left( \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \max_{s, a} \|p_{\pi'}(\cdot|s, a) - p_{\pi}(\cdot|s, a)\|_1 \right) + \frac{S_r}{1 - \gamma} \|\pi' - \pi\| \\
& 1626 \quad + L_p S_p \|\pi' - \pi\| + \frac{\gamma \epsilon_r}{(1 - \gamma)^2} \left( \max_s \|\pi'(\cdot|s) - \pi(\cdot|s)\|_1 + \max_{s, a} \|p_{\pi'}(\cdot|s, a) - p_{\pi}(\cdot|s, a)\|_1 \right) + \frac{S_r}{1 - \gamma} \|\pi' - \pi\| \\
& 1630 \quad \leq \left( \frac{|\mathcal{A}|(1 + 2\lambda \log |\mathcal{A}|)}{\Delta(1 - \gamma)^2} + \frac{\gamma L_{\pi}}{\Delta} \right) \|\pi' - \pi\| + \epsilon_p \sqrt{|\mathcal{S}||\mathcal{A}|} \left[ \frac{2(1 + \lambda \log |\mathcal{A}|)}{(1 - \gamma)^2} + \gamma L_p \right] \|\pi' - \pi\| \\
& 1631 \quad + \frac{\epsilon_r \sqrt{|\mathcal{A}|} \|\pi' - \pi\|}{1 - \gamma} + \epsilon_p \left[ \frac{\ell_{\pi}}{\Delta} \|\pi' - \pi\| + \ell_p \epsilon_p \|\pi' - \pi\| + \frac{2 - \gamma}{1 - \gamma} \epsilon_r \sqrt{|\mathcal{S}|} \|\pi' - \pi\| \right] \\
& 1632 \quad + L_p S_p \|\pi' - \pi\| + \frac{\gamma \epsilon_r}{(1 - \gamma)^2} \left( \sqrt{|\mathcal{S}|} \|\pi' - \pi\| + \epsilon_p \sqrt{|\mathcal{S}|} \|\pi' - \pi\| \right) + \frac{S_r}{1 - \gamma} \|\pi' - \pi\| \\
& 1633 \quad 16.33 \\
& 1634 \quad \leq \left( \frac{|\mathcal{A}|(1 + 2\lambda \log |\mathcal{A}|)}{\Delta(1 - \gamma)^2} + \frac{\gamma L_{\pi}}{\Delta} \right) \|\pi' - \pi\| + \frac{\epsilon_p}{\Delta} \sqrt{\frac{|\mathcal{S}|}{|\mathcal{A}|}} \left[ \frac{2(1 + \lambda \log |\mathcal{A}|)}{($$

$$L_\lambda = L_p \epsilon_p + \frac{\epsilon_r}{1-\gamma} + L_\pi = \frac{\sqrt{|\mathcal{A}|}(2-\gamma + \gamma\lambda \log |\mathcal{A}|) + \epsilon_p \sqrt{|\mathcal{S}|}(1+\lambda \log |\mathcal{A}|) + \epsilon_r(1-\gamma)}{(1-\gamma)^2}.$$

$$|\log u - \log v| = \log \max(u, v) - \log \min(u, v) = \int_{\min(u, v)}^{\max(u, v)} \frac{1}{x} dx \leq \frac{1}{\Delta} [\max(u, v) - \min(u, v)] = \frac{|u - v|}{\Delta}.$$

$$\|\log \pi' - \log \pi\|^2 = \sum_{s,a} |\log \pi'(a|s) - \log \pi(a|s)|^2 \leq \Delta^{-2} \sum_{s,a} |\pi'(a|s) - \pi(a|s)|^2 = \Delta^{-2} \|\pi' - \pi\|^2.$$

Next, we will prove Eq. [\(21\)](#page-5-8) about the Lipschitz continuity of the following performative policy gradient.

$$\begin{aligned}\nabla_\pi V_{\lambda,\pi}^\pi &= \nabla_\pi J_\lambda(\pi, \pi, p_\pi, r_\pi) \\ &= \nabla_\pi J_\lambda(\pi, \pi, p_{\tilde{\pi}}, r_{\tilde{\pi}})|_{\tilde{\pi}=\pi} + (\nabla_\pi p_\pi) \nabla_{p_\pi} J_\lambda(\pi, \pi, p_\pi, r_\pi) + (\nabla_\pi r_\pi) \nabla_{r_\pi} J_\lambda(\pi, \pi, p_\pi, r_\pi).\end{aligned}\tag{105}$$

For any π, π′ ∈ Π∆, we prove Eq. [\(21\)](#page-5-8) as follows.

$$\begin{aligned}
& 1650 + \frac{\epsilon_p}{\Delta} \left[ \frac{\sqrt{|\mathcal{S}| |\mathcal{A}|} (2 + 3\gamma \lambda \log |\mathcal{A}|)}{(1 - \gamma)^3} + \frac{2\epsilon_p \gamma |\mathcal{S}| (1 + \lambda \log |\mathcal{A}|)}{|\mathcal{A}| (1 - \gamma)^3} + \frac{2 - \gamma}{|\mathcal{A}| (1 - \gamma)} \epsilon_r \sqrt{|\mathcal{S}|} \right] \|\pi' - \pi\| \\
& 1652 + \frac{\epsilon_r \sqrt{|\mathcal{A}|} (1 - \gamma) + \gamma \epsilon_r \sqrt{|\mathcal{S}|} (1 + \epsilon_p)}{\Delta |\mathcal{A}| (1 - \gamma)^2} \|\pi' - \pi\| + \frac{S_p \sqrt{|\mathcal{S}|} (1 + \lambda \log |\mathcal{A}|) + S_r (1 - \gamma)}{\Delta |\mathcal{A}| (1 - \gamma)^2} \|\pi' - \pi\| \\
& 1654 \leq \frac{3|\mathcal{A}|(1 + \lambda \log |\mathcal{A}|)}{\Delta(1 - \gamma)^2} \|\pi' - \pi\| + \frac{\epsilon_p \sqrt{|\mathcal{S}| |\mathcal{A}|} (5 + 6\lambda \log |\mathcal{A}|)}{\Delta(1 - \gamma)^3} \|\pi' - \pi\| \\
& 1656 + \frac{\epsilon_r [\sqrt{|\mathcal{A}|}(1 - \gamma) + \sqrt{|\mathcal{S}|}(\gamma + 2\epsilon_p)] + S_p \sqrt{|\mathcal{S}|}(1 + \lambda \log |\mathcal{A}|) + S_r(1 - \gamma)}{\Delta |\mathcal{A}| (1 - \gamma)^2} \|\pi' - \pi\| \\
& 1658 \stackrel{(e)}{=} \frac{\ell_\lambda}{\Delta} \|\pi' - \pi\|,
\end{aligned}$$

1662 1663 1664 1665 1666 where (a) uses Eqs. [\(49\)](#page-13-1), [\(50\)](#page-13-4) and [\(53\)](#page-13-5)-[\(55\)](#page-13-7) as well as Assumptions [1](#page-3-4)[-2,](#page-3-5) and (b) uses the following bounds for any π, π′ ∈ ∆ in which (d) uses Assumption [1,](#page-3-4) (c) uses ∆ ≤ |A|<sup>−</sup><sup>1</sup> (since for any π ∈ Π∆, 1 = P a π(a|s) ≥ ∆|A|), (d) uses L<sup>π</sup> := √ |A|(2−γ+γλ log |A|) (1−γ) <sup>2</sup> , L<sup>p</sup> := √ |S|(1+λ log |A|) (1−γ) <sup>2</sup> , ℓ<sup>π</sup> := √ |S||A|(2+3γλ log |A|) (1−γ) <sup>3</sup> and ℓ<sup>p</sup> := 2γ|S|(1+λ log |A|) (1−γ) <sup>3</sup> defined in Lemma [6,](#page-13-8) (e) uses ℓ<sup>λ</sup> defined by Eq. [\(23\)](#page-5-9).

$$\begin{aligned} & 1667 \max_s \| \log \pi'(\cdot|s) - \log \pi(\cdot|s) \| \leq \Delta^{-1} \max_s \| \pi'(\cdot|s) - \pi(\cdot|s) \| \leq \Delta^{-1} \| \pi' - \pi \|, \\ & 1668 \\ & 1699 \quad \| p_{\pi'} - p_\pi \| \stackrel{(d)}{\leq} \epsilon_p \| \pi' - \pi \|, \\ & 1670 \\ & 1671 \quad \| r_{\pi'} - r_\pi \|_\infty \leq \| r_{\pi'} - r_\pi \| \stackrel{(d)}{\leq} \epsilon_r \| \pi' - \pi \|, \\ & 1672 \\ & 1673 \quad \max_s \| \pi'(\cdot|s) - \pi(\cdot|s) \|_1 \leq \sqrt{|\mathcal{S}|} \max_s \| \pi'(\cdot|s) - \pi(\cdot|s) \| \leq \sqrt{|\mathcal{S}|} \| \pi' - \pi \|, \\ & 1674 \\ & 1675 \quad \max_{s,a} \| p_{\pi'}(\cdot|s, a) - p_\pi(\cdot|s, a) \|_1 \leq \sqrt{|\mathcal{S}|} \max_{s,a} \| p_{\pi'}(\cdot|s, a) - p_\pi(\cdot|s, a) \| \leq \sqrt{|\mathcal{S}|} \| p_{\pi'} - p_\pi \| \stackrel{(d)}{\leq} \epsilon_p \sqrt{|\mathcal{S}|} \| \pi' - \pi \|. \\ & 1676 \end{aligned}$$

1679

1689 1690

1694

1696

1699 1700

# H. Proof of Proposition [1](#page-6-3)

We prove the validity of the stochastic gradient [\(26\)](#page-6-0) first. For any π ∈ Π∆, s ∈ S and a ∈ A, we have π(a|s) ≥ ∆, so π(a|s) ≤ 1 − ∆ (since P <sup>a</sup>′ π(a ′ |s) = 1). For any u<sup>i</sup> ∈ U1, we have |ui(a|s)| ≤ 1. Therefore,

$$(\pi \pm \delta u_i)(a|s) \geq \pi(a|s) - \delta|u_i(a|s)| \geq \Delta - \delta > 0, \quad (106)$$

which means π ± δu<sup>i</sup> ∈ Π. Hence, V π λ,π′ is well defined for π ′ ∈ {π + δu<sup>i</sup> , π − δui}.

Then we will prove the estimation error [\(30\)](#page-6-4). Based on Lemma [10,](#page-18-4) there exists an orthogonal transformation T : R |A|→ Z|A|−<sup>1</sup> ={z=[z1, . . . , z|A|] ∈ <sup>R</sup> |A| : P i z<sup>i</sup> = 0}.

Note that any x ∈ R |S|(|A|−1) can be written as x = [xs]s∈S , a concatenation of |S| vectors x<sup>s</sup> ∈ <sup>R</sup> |A|. Therefore, we can define the transformation T : R |S|(|A|−1) → L<sup>0</sup> def = u ∈ R |S||A| : u(·|s) ∈ Z|A|−1, ∀s ∈ S as follows

$$[T(x)](\cdot|s) = \mathcal{T}(x_s), \forall s \in \mathcal{S} \quad (107)$$

where x<sup>s</sup> ∈ <sup>R</sup> |A| are extracted from |A| entries of x = [xs]s∈S . For any x = [xs]s∈S , y = [ys]s∈S ∈ <sup>R</sup> |S|(|A|−1) and α, β ∈ R, we can prove that T is an orthogonal transformation as follows.

$$\begin{aligned} [T(\alpha x + \beta y)](\cdot|s) &= \mathcal{T}(\alpha x_s + \beta y_s) = \alpha \mathcal{T}(x_s) + \beta \mathcal{T}(y_s) = \alpha[T(x)](\cdot|s) + \beta[T(x)](\cdot|s) \\ &\Rightarrow T(\alpha x + \beta y) = \alpha T(x) + \beta T(y). \end{aligned}$$

$$\langle T(x), T(y) \rangle = \sum_s \langle [T(x)](\cdot|s), [T(y)](\cdot|s) \rangle = \sum_s \langle \mathcal{T}(x_s), \mathcal{T}(y_s) \rangle = \sum_s \langle x_s, y_s \rangle = \langle x, y \rangle.$$

Define the following set.

$$T^{-1}(\Pi_{\Delta} - |\mathcal{A}|^{-1}) \stackrel{\text{def}}{=} \{\pi \in \Pi_{\Delta} : T^{-1}(\pi - |\mathcal{A}|^{-1})\}, \quad (108)$$

1706 where π − |A|<sup>−</sup><sup>1</sup> ∈ <sup>R</sup> |S||A| has entries (π − |A|<sup>−</sup><sup>1</sup> )(a|s) = π(a|s) − |A|<sup>−</sup><sup>1</sup> , so π − |A|<sup>−</sup><sup>1</sup> ∈ L0. Furthermore, since Π<sup>∆</sup> is a convex and compact set and T −1 is an orthogonal transformation, T −1 (Π<sup>∆</sup> − |A|<sup>−</sup><sup>1</sup> ) is a convex and compact subset of L0.

1709

1714

1716

1719

1724

1726

1729 Furthermore, we will show that fλ(x) is a Lipscthiz continuous and Lipschitz smooth function of x ∈ Π∆. For any x, x′ ∈ T −1 (Π<sup>∆</sup> − |A|<sup>−</sup><sup>1</sup> ), we have

1734

1736

1754

1756

Then for any x ∈ T −1 (Π<sup>∆</sup> − |A|<sup>−</sup><sup>1</sup> ), we have T(x) + |A|<sup>−</sup><sup>1</sup> ∈ Π∆, so we can define the function fλ(x) def = V T(x)+|A|<sup>−</sup><sup>1</sup> λ,T(x)+|A|<sup>−</sup><sup>1</sup> .

Note that as V π λ,π is a differentiable function of π, so for any π ′ ∈ Π and fixed π ∈ Π we have

$$\frac{V_{\lambda,\pi'}^{\pi'} - V_{\lambda,\pi}^{\pi} - \langle \nabla_{\pi} V_{\lambda,\pi}^{\pi}, \pi' - \pi \rangle}{\|\pi' - \pi\|} = \frac{V_{\lambda,\pi'}^{\pi} - V_{\lambda,\pi}^{\pi} - \langle \text{proj}_{\mathcal{L}_0}(\nabla_{\pi} V_{\lambda,\pi}^{\pi}), \pi' - \pi \rangle}{\|\pi' - \pi\|} \rightarrow 0 \text{ as } \pi' \in \Pi \text{ and } \pi' \rightarrow \pi, \quad (109)$$

where the above = uses π ′ − π ∈ L0. Then, we can prove that f<sup>λ</sup> is differentiable with gradient ∇fλ(x) = T −1 projL0∇πV π λ,π π=T(x)+|A|<sup>−</sup><sup>1</sup> , since for any x ′ ∈ T −1 (Π<sup>∆</sup> − |A|<sup>−</sup><sup>1</sup> ) and fixed x ∈ T −1 (Π<sup>∆</sup> − |A|<sup>−</sup><sup>1</sup> ) we have

$$\frac{f_\lambda(x') - f_\lambda(x) - \langle T^{-1}[\text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x)+|\mathcal{A}|^{-1}})], x' - x \rangle}{\|x' - x\|} \\ \underline{(a)} \frac{V_{\lambda,T(x')+|\mathcal{A}|^{-1}}^{T(x')+|\mathcal{A}|^{-1}} - V_{\lambda,T(x)+|\mathcal{A}|^{-1}}^{T(x)+|\mathcal{A}|^{-1}} - \langle \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x)+|\mathcal{A}|^{-1}}), [T(x') + |\mathcal{A}|^{-1}] - [T(x) + |\mathcal{A}|^{-1}] \rangle}{\|[T(x') + |\mathcal{A}|^{-1}] - [T(x) + |\mathcal{A}|^{-1}]\|} \\ \xrightarrow{(b)} 0 \text{ as } x' \in T^{-1}(\Pi_\Delta - |\mathcal{A}|^{-1}) \text{ and } x' \rightarrow x, \quad (110)$$

where (a) uses the property of the orthogonal transformation T, and (b) uses Eq. [\(109\)](#page-31-0) and the fact that x ′ → x means [T(x ′ ) + |A|<sup>−</sup><sup>1</sup> ] − [T(x) + |A|<sup>−</sup><sup>1</sup> = ∥x ′ − x∥ → 0.

$$|f_\lambda(x') - f_\lambda(x)| = |V_{\lambda, T(x')+|\mathcal{A}|^{-1}}^{T(x')+|\mathcal{A}|^{-1}} - V_{\lambda, T(x)+|\mathcal{A}|^{-1}}^{T(x)+|\mathcal{A}|^{-1}}| \stackrel{(a)}{\leq} \frac{L_\lambda}{\Delta} \|T(x') - T(x)\| \stackrel{(b)}{=} \frac{L_\lambda}{\Delta} \|x' - x\|,$$

$$\begin{aligned} \|\nabla f_\lambda(x') - \nabla f_\lambda(x)\| &= \|T^{-1}[\text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x')})] - T^{-1}[\text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x)})]\| \\ &\stackrel{(b)}{=} \|\text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x')+|\mathcal{A}|^{-1}}) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x)+|\mathcal{A}|^{-1}})\| \\ &\leq \|(\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x')+|\mathcal{A}|^{-1}}) - (\nabla_\pi V_{\lambda,\pi}^\pi|_{\pi=T(x)+|\mathcal{A}|^{-1}})\| \\ &\stackrel{(a)}{\leq} \frac{\ell_\lambda}{\Delta} \|T(x') - T(x)\| \stackrel{(b)}{=} \frac{\ell_\lambda}{\Delta} \|x' - x\|, \end{aligned}$$

In both the inequalities above, (a) applies Theorem [3](#page-5-5) to T(x) + |A|<sup>−</sup><sup>1</sup> , T(x ′ ) + |A|<sup>−</sup><sup>1</sup> ∈ Π<sup>∆</sup> and (b) uses the property of the orthogonal transformation T. The two inequalities above implies that f<sup>λ</sup> is an <sup>L</sup><sup>λ</sup> ∆ -Lipschitz continuous and <sup>ℓ</sup><sup>λ</sup> ∆ -Lipschitz smooth function on T −1 (Π<sup>∆</sup> − |A|<sup>−</sup><sup>1</sup> ).

Denote

$$g_{\lambda, \delta}(\pi) = \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N (V_{\lambda, \pi + \delta u_i}^{\pi + \delta u_i} - V_{\lambda, \pi - \delta u_i}^{\pi - \delta u_i}) u_i, \quad (111)$$

which replaces Vˆ <sup>π</sup> ′ λ,π′ with V π ′ λ,π′ in Eq. [\(26\)](#page-6-0). The estimation error of the performative policy gradient estimator above can be rewritten as follows for any π ∈ Π∆.

$$\begin{aligned} & g_{\lambda,\delta}(\pi) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi) \\ & \stackrel{(a)}{=} \left( \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N (V_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i} - V_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i}) u_i \right) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi) \\ & \stackrel{(b)}{=} \left( \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N (f_\lambda[T^{-1}(\pi - |\mathcal{A}|^{-1}) + \delta T^{-1}(u_i)] - f_\lambda[T^{-1}(\pi - |\mathcal{A}|^{-1})] - \delta T^{-1}(u_i)]) T^{-1}(u_i) \right) \end{aligned}$$

1764

1766

1769

1774

1776

1779

1790

1794

1796

$$\begin{aligned} & -T^{-1}[\text{proj}_{\mathcal{L}_0}(\nabla_{\pi}V_{\lambda,\pi}^{\pi})] \\ & \stackrel{(c)}{=} \left( \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N (f_{\lambda}[T^{-1}(\pi - |\mathcal{A}|^{-1}) + \delta T^{-1}(u_i)] - f_{\lambda}[T^{-1}(\pi - |\mathcal{A}|^{-1}) - \delta T^{-1}(u_i)])T^{-1}(u_i) \right) \\ & - \nabla f_{\lambda}[T^{-1}(\pi - |\mathcal{A}|^{-1})], \end{aligned} \tag{112}$$

where (a) uses Eq. [\(26\)](#page-6-0), (b) uses fλ(x) def = V T(x)+|A|<sup>−</sup><sup>1</sup> λ,T(x)+|A|<sup>−</sup><sup>1</sup> and the property of the orthogonal transformation T −1 , (c) uses ∇fλ(x) = T −1 projL0∇πV π λ,π π=T(x)+|A|<sup>−</sup><sup>1</sup> . Note that in the above Eq. [\(112\)](#page-32-0), π ∈ Π<sup>∆</sup> and u<sup>i</sup> is uniformly distributed on the sphere U<sup>1</sup> ∩ L<sup>0</sup> with U<sup>1</sup> defined by Eq. [\(27\)](#page-6-6), as repeated below.

$$U_1 \stackrel{\text{def}}{=} \{u \in \mathbb{R}^{|S||\mathcal{A}|} : \|u\| = 1\}. \quad (113)$$

Hence, π ± δu<sup>i</sup> ∈ Π∆−<sup>δ</sup> which implies T −1 (π − |A|<sup>−</sup><sup>1</sup> ) ± δT <sup>−</sup><sup>1</sup> (ui) = T −1 (π ± δu<sup>i</sup> − |A|<sup>−</sup><sup>1</sup> ) ∈ T −1 (Π∆−<sup>δ</sup> − |A|<sup>−</sup><sup>1</sup> ). Also, T −1 (ui) is uniformly distributed on the sphere T −1 (U1,0) = <sup>S</sup>|S|(|A|−1) = {u ∈ <sup>R</sup> |S|(|A|−1) : ∥u∥ = 1}. Therefore, we can apply Lemma [9](#page-17-2) to the above Eq. [\(112\)](#page-32-0) where the function f<sup>λ</sup> is an <sup>L</sup><sup>λ</sup> ∆−δ -Lipschitz continuous and <sup>ℓ</sup><sup>λ</sup> ∆−δ -Lipschitz smooth function on T −1 (Π∆−<sup>δ</sup> − |A|<sup>−</sup><sup>1</sup> ), and obtain the following bound which holds with probability at least 1 − η.

$$\begin{aligned} & \|g_{\lambda,\delta}(\pi) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi)\| \\ & \leq \frac{4L_\lambda |\mathcal{S}|(|\mathcal{A}| - 1)}{3N(\Delta - \delta)} \log \left( \frac{|\mathcal{S}|(|\mathcal{A}| - 1) + 1}{\eta} \right) + \frac{L_\lambda |\mathcal{S}|(|\mathcal{A}| - 1)}{\Delta - \delta} \sqrt{\frac{2}{N} \log \left( \frac{|\mathcal{S}|(|\mathcal{A}| - 1) + 1}{\eta} \right)} + \frac{\delta \ell_\lambda}{\Delta - \delta} \\ & \leq \frac{4L_\lambda |\mathcal{S}||\mathcal{A}|}{3N(\Delta - \delta)} \log \left( \frac{|\mathcal{S}||\mathcal{A}|}{\eta} \right) + \frac{L_\lambda |\mathcal{S}||\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log \left( \frac{|\mathcal{S}||\mathcal{A}|}{\eta} \right)} + \frac{\delta \ell_\lambda}{\Delta - \delta}. \end{aligned} \quad (114)$$

Note that Eq. [\(24\)](#page-6-5) holds for any a certain policy π with probability at least 1 − η. Therefore, with probability at least 1 − 2Nη, we have

$$|\hat{V}_{\lambda, \pi'}^{\pi'} - V_{\lambda, \pi'}^{\pi'}| \leq \epsilon_V, \forall \pi' \in \{\pi \pm \delta u_i\}_{i=1}^N \quad (115)$$

Therefore, with probability at least 1 − (2N + 1)η, Eqs. [\(114\)](#page-32-2) and [\(115\)](#page-32-3) hold and thus we have

$$\begin{aligned} & \|\hat{g}_{\lambda,\delta}(\pi) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi)\| \\ & \leq \|\hat{g}_{\lambda,\delta}(\pi) - g_{\lambda,\delta}(\pi)\| + \|g_{\lambda,\delta}(\pi) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda,\pi}^\pi)\| \\ & \stackrel{(a)}{\leq} \left\| \frac{|\mathcal{S}|(|\mathcal{A}|-1)}{2N\delta} \sum_{i=1}^N (\hat{V}_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i} - V_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i} - \hat{V}_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i} + V_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i}) u_i \right\| \\ & \quad + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3N(\Delta-\delta)} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta-\delta} \sqrt{\frac{2}{N} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta-\delta} \\ & \stackrel{(b)}{\leq} \frac{|\mathcal{S}||\mathcal{A}|}{N\delta} \sum_{i=1}^N \left\| (\hat{V}_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i} - V_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i} - \hat{V}_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i} + V_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i}) u_i \right\| \\ & \quad + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3N(\Delta-\delta)} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta-\delta} \sqrt{\frac{2}{N} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta-\delta} \\ & \leq \frac{|\mathcal{S}||\mathcal{A}|}{N\delta} \sum_{i=1}^N (|\hat{V}_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i} - V_{\lambda,\pi+\delta u_i}^{\pi+\delta u_i}| + |\hat{V}_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i} + V_{\lambda,\pi-\delta u_i}^{\pi-\delta u_i}|) \\ & \quad + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3N(\Delta-\delta)} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta-\delta} \sqrt{\frac{2}{N} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta-\delta} \\ & \stackrel{(c)}{\leq} \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3N(\Delta-\delta)} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta-\delta} \sqrt{\frac{2}{N} \log\left(\frac{|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta-\delta}, \end{aligned} \tag{116}$$

1860 1861 where (a) uses π˜<sup>t</sup> − πt, π˜ − π<sup>t</sup> ∈ L<sup>0</sup> for π˜t, π˜ ∈ Π∆, and (b) uses Eq. [\(117\)](#page-33-2) and Lemma [12.](#page-19-4)

where (a) uses Eqs. [\(26\)](#page-6-0), [\(64\)](#page-17-0) and [\(114\)](#page-32-2), (b) uses Jensen's inequality that ∥ 1 N P<sup>N</sup> <sup>i</sup>=1 xi∥ <sup>2</sup> ≤ 1 N P<sup>N</sup> <sup>i</sup>=1 ∥xi∥ 2 for any vectors {xi} N <sup>i</sup>=1 of the same dimensionality, (c) uses Eq. [\(24\)](#page-6-5). The conclusion can be proved by replacing <sup>η</sup> with <sup>η</sup> 3N in the inequality above.

# I. Proof of Proposition [2](#page-7-7)

For any π ∈ Π∆, it is easily seen that the corresponding π ′ defined by Eq. [\(19\)](#page-5-4) also belongs to Π∆. Therefore,

$$\langle \nabla_\pi V_{\lambda,\pi}^\pi, \pi' - \pi \rangle \leq \max_{\tilde{\pi} \in \Pi_\Delta} \langle \nabla_\pi V_{\lambda,\pi}^\pi, \tilde{\pi} - \pi \rangle \leq \frac{D\lambda}{5|\mathcal{A}|(1-\gamma)}.$$

Substituting the above inequality into Eq. [\(17\)](#page-5-6), we obtain that

$$\pi(a|s) \geq \pi_{\min} \exp \left[ -\frac{2|\mathcal{A}|}{D\lambda} (1-\gamma) \langle \nabla_{\pi} V_{\lambda,\pi}^{\pi}, \pi' - \pi \rangle \right] \geq \frac{2\pi_{\min}}{3} \geq 2\Delta.$$

Therefore, for any π<sup>2</sup> ∈ Π, we can prove that <sup>π</sup>2+<sup>π</sup> 2 ∈ Π<sup>∆</sup> as follows.

$$\frac{\pi_2(a|s) + \pi(a|s)}{2} \geq \frac{0 + 2\Delta}{2} = \Delta.$$

Therefore, we can prove Eq. [\(34\)](#page-7-8) as follows.

$$\max_{\pi_2 \in \Pi} \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \pi_2 - \pi \rangle = 2 \max_{\pi_2 \in \Pi} \left\langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \frac{\pi_2 + \pi}{2} - \pi \right\rangle \stackrel{(a)}{\leq} 2 \max_{\tilde{\pi} \in \Pi_{\Delta}} \langle \nabla_{\pi} V_{\lambda, \pi}^{\pi}, \tilde{\pi} - \pi \rangle.$$

where (a) uses <sup>π</sup>2+<sup>π</sup> 2 ∈ Π∆.

# J. Proof of Theorem [4](#page-7-6)

If π<sup>t</sup> ∈ Π∆, then πt+1 ∈ Π∆, since Π<sup>∆</sup> is a convex set and πt+1 obtained by Eq. [\(32\)](#page-7-3) is a convex combination of πt, π˜<sup>t</sup> ∈ Π∆. Since π<sup>0</sup> ∈ Π∆, we have π<sup>t</sup> ∈ Π<sup>∆</sup> for all t by induction. Therefore, Proposition [1](#page-6-3) implies that the following bound holds simultaneously for all {πt} T <sup>t</sup>=1 ⊆ Π<sup>∆</sup> with probability at least 1 − η.

$$\begin{aligned} & \|\hat{g}_{\lambda,\delta}(\pi_t) - \text{proj}_{\mathcal{L}_0}(\nabla_{\pi} V_{\lambda, \pi_t}^{\pi_t})\| \\ & \leq \frac{2|\mathcal{S}| |\mathcal{A}| \epsilon_V}{\delta} + \frac{4L_{\lambda} |\mathcal{S}| |\mathcal{A}|}{3TN(\Delta - \delta)} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_{\lambda} |\mathcal{S}| |\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N}} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{\delta\ell_{\lambda}}{\Delta - \delta}. \end{aligned} \quad (117)$$

The bound above further implies that for any π ∈ Π, we have

$$\begin{aligned} & |\langle \hat{g}_\lambda, \delta(\pi_t) - \nabla_\pi V_{\lambda, \pi_t}^{\pi_t}, \pi - \pi_t \rangle| \\ & \stackrel{(a)}{=} |\langle \hat{g}_\lambda, \delta(\pi_t) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda, \pi_t}^{\pi_t}), \pi - \pi_t \rangle| \\ & \leq \|\hat{g}_\lambda, \delta(\pi_t) - \text{proj}_{\mathcal{L}_0}(\nabla_\pi V_{\lambda, \pi_t}^{\pi_t})\| \cdot \|\pi - \pi_t\| \\ & \stackrel{(b)}{\leq} \sqrt{2|\mathcal{S}|} \left[ \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3TN(\Delta - \delta)} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta \ell_\lambda}{\Delta - \delta} \right], \end{aligned} \quad (118)$$

Under the conditions above, we have

$$\begin{aligned} & V_{\lambda, \pi_{t+1}}^{\pi_{t+1}} \\ & \stackrel{(a)}{\geq} V_{\lambda, \pi_t}^{\pi_t} + \langle \nabla_{\pi} V_{\lambda, \pi_t}^{\pi_t}, \pi_{t+1} - \pi_t \rangle - \frac{\ell_{\lambda}}{2\Delta} \|\pi_{t+1} - \pi_t\|^2 \\ & \stackrel{(b)}{\leq} V_{\lambda, \pi_t}^{\pi_t} + \beta \langle \nabla_{\pi} V_{\lambda, \pi_t}^{\pi_t}, \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} \beta^2}{2\Delta} \|\tilde{\pi}_t - \pi_t\|^2 \end{aligned}$$

$$\begin{aligned} 1870 &= V_{\lambda,\pi_t}^{\pi_t} + \beta \langle \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle + \beta \langle \nabla_{\pi} V_{\lambda,\pi_t}^{\pi_t} - \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} \beta^2}{2\Delta} \|\tilde{\pi}_t - \pi_t\|^2 \\ 1871 &\geq V_{\lambda,\pi_t}^{\pi_t} + \beta \langle \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} |\mathcal{S}| \beta^2}{\Delta} \\ 1872 &= V_{\lambda,\pi_t}^{\pi_t} + \beta \langle \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} |\mathcal{S}| \beta^2}{\Delta} \\ 1873 &= V_{\lambda,\pi_t}^{\pi_t} + \beta \langle \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} |\mathcal{S}| \beta^2}{\Delta} \\ 1874 &= V_{\lambda,\pi_t}^{\pi_t} + \beta \langle \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} |\mathcal{S}| \beta^2}{\Delta} \\ 1875 &= -\beta \sqrt{2|\mathcal{S}|} \left[ \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_{\lambda}|\mathcal{S}||\mathcal{A}|}{3TN(\Delta - \delta)} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right) + \frac{L_{\lambda}|\mathcal{S}||\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right)} + \frac{\delta \ell_{\lambda}}{\Delta - \delta} \right], \quad (119) \\ 1876 &= V_{\lambda,\pi_t}^{\pi_t} + \beta \langle \hat{g}_{\lambda,\delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle - \frac{\ell_{\lambda} |\mathcal{S}| \beta^2}{\Delta} \end{aligned}$$

1877 1878 where (a) uses the <sup>ℓ</sup><sup>λ</sup> ∆ -Lipschitz smoothness of V π λ,π on Π∆, (b) uses Eq. [\(32\)](#page-7-3), (c) uses Eq. [\(118\)](#page-33-3) and Lemma [12.](#page-19-4)

1879 Rearranging and averaging Eq. [\(119\)](#page-34-1) over t = 0, 1, . . . , T − 1, we obtain that

$$\begin{aligned}
& 1880 & \max_{\tilde{\pi} \in \Pi_\Delta} \langle \hat{g}_{\lambda, \delta}(\pi_{\tilde{T}}), \tilde{\pi} - \pi_{\tilde{T}} \rangle \\
& 1881 & \\
& 1882 & \\
& 1883 & \stackrel{(a)}{=} \langle \hat{g}_{\lambda, \delta}(\pi_{\tilde{T}}), \tilde{\pi}_{\tilde{T}} - \pi_{\tilde{T}} \rangle \\
& 1884 & \\
& 1885 & \stackrel{(b)}{\leq} \frac{1}{T} \sum_{t=0}^{T-1} \langle \hat{g}_{\lambda, \delta}(\pi_t), \tilde{\pi}_t - \pi_t \rangle \\
& 1886 & \\
& 1887 & \leq \frac{V_{\lambda, \pi_T}^{\pi_T} - V_{\lambda, \pi_0}^{\pi_0}}{T\beta} + \frac{\ell_\lambda |\mathcal{S}| \beta}{\Delta} \\
& 1888 & \\
& 1889 & \\
& 1890 & + \sqrt{2|\mathcal{S}|} \left[ \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3TN(\Delta-\delta)} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta-\delta} \sqrt{\frac{2}{N} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta-\delta} \right] \\
& 1891 & \\
& 1892 & \leq \frac{1 + \lambda \log |\mathcal{A}|}{T\beta(1-\gamma)} + \frac{\ell_\lambda |\mathcal{S}| \beta}{\Delta} \\
& 1893 & \\
& 1894 & + \sqrt{2|\mathcal{S}|} \left[ \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3TN(\Delta-\delta)} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) + \frac{L_\lambda|\mathcal{S}||\mathcal{A}|}{\Delta-\delta} \sqrt{\frac{2}{N} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right)} + \frac{\delta\ell_\lambda}{\Delta-\delta} \right], \quad (120) \\
& 1895 & \\
& 1896 & \\
& 1897 & 
\end{aligned}$$

1897

1898 1899 where (a) uses Lemma [<sup>1</sup>](#page-7-5) which means <sup>π</sup>˜<sup>t</sup> satisfies Eq. [\(31\)](#page-7-4) and (b) uses the output rule of Algorithm [<sup>1</sup>](#page-7-0) that <sup>T</sup>e ∈ arg min0≤t≤<sup>T</sup> <sup>−</sup><sup>1</sup> ⟨gˆλ,δ(πt), π˜<sup>t</sup> − πt⟩. Therefore,

$$\begin{aligned} 1900 \quad & \max_{\tilde{\pi} \in \Pi_\Delta} \langle \nabla_\pi V_{\lambda, \pi_{\tilde{T}}}^{\pi_{\tilde{T}}}, \tilde{\pi} - \pi_{\tilde{T}} \rangle \\ 1902 \quad & = \max_{\tilde{\pi} \in \Pi_\Delta} [\langle \nabla_\pi V_{\lambda, \pi_{\tilde{T}}}^{\pi_{\tilde{T}}} - \hat{g}_{\lambda, \delta}(\pi_{\pi_{\tilde{T}}}), \tilde{\pi} - \pi_{\tilde{T}} \rangle + \langle \hat{g}_{\lambda, \delta}(\pi_{\pi_{\tilde{T}}}), \tilde{\pi} - \pi_{\tilde{T}} \rangle] \\ 1904 \quad & \\ 1905 \quad & \stackrel{(a)}{\leq} \frac{1 + \lambda \log |\mathcal{A}|}{T\beta(1 - \gamma)} + \frac{\ell_\lambda |\mathcal{S}| \beta}{\Delta} \\ 1906 \quad & \\ 1907 \quad & + 2\sqrt{2|\mathcal{S}|} \left[ \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda |\mathcal{S}||\mathcal{A}|}{3TN(\Delta - \delta)} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right) + \frac{L_\lambda |\mathcal{S}||\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right)} + \frac{\delta \ell_\lambda}{\Delta - \delta} \right], \quad (121) \\ 1908 \quad & \\ 1909 \quad & \end{aligned}$$

1914

1916

1918 1919

where (a) uses Eqs. [\(118\)](#page-33-3) and [\(120\)](#page-34-2).

Use the following hyperparameter choices for Algorithm [1.](#page-7-0)

$$\Delta = \frac{\pi \min}{3}, \quad (122)$$

$$\beta = \frac{D\Delta\epsilon}{12\ell_\lambda|\mathcal{S}|} = \frac{D\pi_{\min}\epsilon}{36\ell_\lambda|\mathcal{S}|} = \mathcal{O}(\epsilon), \quad (123)$$

$$T = \frac{12(1 + \lambda \log |\mathcal{A}|)}{D\epsilon\beta(1 - \gamma)} = \frac{432\ell\lambda|\mathcal{S}|(1 + \lambda \log |\mathcal{A}|)}{\pi_{\min}D^2(1 - \gamma)\epsilon^2} = \mathcal{O}(\epsilon^{-2}) \quad (124)$$

$$\delta = \frac{D\Delta\epsilon}{48\sqrt{2|\mathcal{S}|\ell_\lambda}} = \frac{D\pi_{\min}\epsilon}{144\sqrt{2|\mathcal{S}|\ell_\lambda}} = \mathcal{O}(\epsilon) \stackrel{(a)}{\leq} \frac{\Delta}{2}, \quad (125)$$

$$\epsilon_V = \frac{D\delta\epsilon}{48|\mathcal{S}||\mathcal{A}|\sqrt{2|\mathcal{S}|}} = \frac{\pi_{\min}D^2\epsilon^2}{13824\ell_\lambda|\mathcal{S}|^2|\mathcal{A}|} = \mathcal{O}(\epsilon^2) \quad (126)$$

1929

1934

1936

1954

1956

1974

1976

$$N = \frac{663552L_\lambda^2|\mathcal{S}|^3|\mathcal{A}|^2}{D^2\pi_{\min}^2\epsilon^2} \log \max \left( \frac{165888L_\lambda^2|\mathcal{S}|^3|\mathcal{A}|^2}{D^2\pi_{\min}^2\epsilon^2}, \frac{1296\ell_\lambda|\mathcal{S}|^2|\mathcal{A}|(1 + \lambda\log|\mathcal{A}|)}{D^2\eta\pi_{\min}(1 - \gamma)\epsilon^2} \right) \\ + 2\log\left(\frac{3|\mathcal{S}||\mathcal{A}|}{\eta}\right) + 3 \\ = \mathcal{O}[\epsilon^{-2}\log(\eta^{-1}\epsilon^{-1})] \quad (127)$$

where (a) uses ϵ ≤ 24p 2|S|ℓλ/D. With the hyperparameter choices above, we obtain the following inequalities [\(128\)](#page-35-1)-[\(130\)](#page-35-2).

$$\begin{aligned} & 2\sqrt{2|\mathcal{S}|} \cdot \frac{L_\lambda |\mathcal{S}| |\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right)} \\ & \stackrel{(a)}{\leq} \frac{24L_\lambda |\mathcal{S}|^{1.5} |\mathcal{A}|}{\pi_{\min}} \sqrt{\frac{\log N}{N} + \frac{1}{N} \log \left( \frac{1296\ell_\lambda |\mathcal{S}|^2 |\mathcal{A}| (1 + \lambda \log |\mathcal{A}|)}{\eta \pi_{\min} D^2 (1 - \gamma) \epsilon^2} \right)} \\ & \stackrel{(b)}{\leq} \frac{24L_\lambda |\mathcal{S}|^{1.5} |\mathcal{A}|}{\pi_{\min}} \sqrt{\tilde{\epsilon} + \frac{\tilde{\epsilon}}{4}} \\ & = \frac{12\sqrt{5}L_\lambda |\mathcal{S}|^{1.5} |\mathcal{A}|}{\pi_{\min}} \cdot \frac{D\pi_{\min}\epsilon}{\sqrt{165888}L_\lambda |\mathcal{S}|^{1.5} |\mathcal{A}|} \\ & \leq \frac{D\epsilon}{12}, \end{aligned} \tag{128}$$

where (a) uses Eq. [\(124\)](#page-34-3) and δ ≤ ∆/2 = πmin/6 implied by Eqs. [\(122\)](#page-34-0) and [\(125\)](#page-34-4), (b) uses Eq. [\(127\)](#page-35-0) and its implication that N ≥ 4˜ϵ −1 log(˜ϵ −1 ) with ϵ˜ = π minϵ 165888D2L<sup>2</sup> |S|<sup>3</sup>|A|<sup>2</sup> ≤ <sup>0</sup>.<sup>5</sup> (since <sup>ϵ</sup> ≤ 288DLλ|S|<sup>1</sup>.<sup>5</sup> |A| πmin ), which implies log <sup>N</sup> <sup>N</sup> ≤ ϵ˜ based on Lemma [11.](#page-19-5)

$$\frac{1}{TN} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) = \frac{\log(TN)}{TN} + \frac{1}{TN} \log\left(\frac{3|\mathcal{S}||\mathcal{A}|}{\eta}\right) \stackrel{(a)}{\leq} \frac{1}{2} + \frac{1}{2} = 1, \quad (129)$$

where (a) uses NT <sup>≥</sup> <sup>N</sup> <sup>≥</sup> max h <sup>3</sup>, 2 log 3|S||A| η i and Lemma [11.](#page-19-5)

$$2\sqrt{2|\mathcal{S}|} \cdot \frac{4L_\lambda|\mathcal{S}||\mathcal{A}|}{3TN(\Delta - \delta)} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right) \stackrel{(a)}{\leq} 2\sqrt{2|\mathcal{S}|} \cdot \frac{\sqrt{2L_\lambda|\mathcal{S}||\mathcal{A}|}}{\Delta - \delta} \sqrt{\frac{1}{TN} \log\left(\frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta}\right)} \stackrel{(b)}{\leq} \frac{D\epsilon}{12} \quad (130)$$

where (a) uses <sup>4</sup> <sup>3</sup> < √ 2 and y ≤ √<sup>y</sup> for <sup>y</sup> <sup>=</sup> 1 T N log 3T N|S||A| η ≤ 1 (Eq. [\(129\)](#page-35-3)), and (b) uses T ≥ 1 and Eq. [\(128\)](#page-35-1). By substituting the hyperparameter choices [\(122\)](#page-34-0)-[\(127\)](#page-35-0) as well as Eqs. [\(128\)](#page-35-1) and [\(130\)](#page-35-2) into Eq. [\(121\)](#page-34-5), we have

$$\begin{aligned}
& \max_{\tilde{\pi} \in \Pi_\Delta} \langle \nabla_\pi V_{\lambda, \pi_{\tilde{T}}}^{\pi_{\tilde{T}}}, \tilde{\pi} - \pi_{\tilde{T}} \rangle \\
& \leq \frac{1 + \lambda \log |\mathcal{A}|}{T\beta(1 - \gamma)} + \frac{\ell_\lambda |\mathcal{S}| \beta}{\Delta} \\
& \quad + 2\sqrt{2|\mathcal{S}|} \left[ \frac{2|\mathcal{S}||\mathcal{A}|\epsilon_V}{\delta} + \frac{4L_\lambda |\mathcal{S}||\mathcal{A}|}{3TN(\Delta - \delta)} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right) + \frac{L_\lambda |\mathcal{S}||\mathcal{A}|}{\Delta - \delta} \sqrt{\frac{2}{N} \log \left( \frac{3TN|\mathcal{S}||\mathcal{A}|}{\eta} \right)} + \frac{\delta \ell_\lambda}{\Delta - \delta} \right] \\
& \leq \frac{1 + \lambda \log |\mathcal{A}|}{\beta(1 - \gamma)} \frac{\epsilon \beta(1 - \gamma)}{12D(1 + \lambda \log |\mathcal{A}|)} + \frac{\ell_\lambda |\mathcal{S}|}{\Delta} \cdot \frac{\Delta \epsilon}{12D\ell_\lambda |\mathcal{S}|} \\
& \quad + \frac{4\sqrt{2|\mathcal{S}||\mathcal{S}||\mathcal{A}|}}{\delta} \cdot \frac{\delta \epsilon}{48D|\mathcal{S}||\mathcal{A}|\sqrt{2|\mathcal{S}|}} + \frac{\epsilon}{12D} + \frac{\epsilon}{12D} + \frac{2\sqrt{2|\mathcal{S}|\ell_\lambda}}{\Delta/2} \cdot \frac{\Delta \epsilon}{48\sqrt{2|\mathcal{S}|D\ell_\lambda}} \\
& = \frac{D\epsilon}{2} \stackrel{(a)}{\leq} \frac{D\lambda}{5|\mathcal{A}|(1 - \gamma)},
\end{aligned}$$

where (a) uses ϵ ≤ 2λD<sup>2</sup> 5|A|(1−γ) . Then based on Proposition [2,](#page-7-7) the inequality above implies that

$$\max_{\tilde{\pi} \in \Pi} \langle \nabla_{\pi} V_{\lambda, \pi_{\tilde{T}}}^{\pi_{\tilde{T}}}, \tilde{\pi} - \pi_{\tilde{T}} \rangle \leq D\epsilon,$$

1981 1983 1984 1986 In Section [4,](#page-5-1) we have proposed a 0-PPG algorithm and obtain its finite-time convergence result to the desired PO policy for our entropy-regularized value function [\(8\)](#page-3-3). We will briefly show that 0-PPG algorithm can also converge to PO for the existing performative reinforcement learning defined by the value function [\(1\)](#page-1-0) with quadratic regularizer Hπ′ (π) = <sup>1</sup> 2 ∥dπ,pπ′ ∥ 2 [\(Mandal et al.,](#page-8-5) [2023;](#page-8-5) [Rank et al.,](#page-9-3) [2024\)](#page-9-3). The *performative value function* can be rewritten as the following λ-strongly concave function of dπ,p<sup>π</sup> .

1987

1989 1990 1991 1994 1996 We can prove the *performative value function* above also satisfies Theorem [1](#page-3-1) (gradient dominance) with a different µ, following the same proof logic, since both regularizers Hπ(π) are strongly convex functions of dπ,p<sup>π</sup> which implies that V π<sup>α</sup> λ,π<sup>α</sup> is a µ-strongly concave function of α as shown in the proof of Theorem [1](#page-3-1) in Appendix [D.](#page-21-0) By direct calculation, we can also show that V π λ,π above is a Lipschitz continuous and Lipschitz smooth function of π ∈ Π. With these two properties, we can follow the proof logic of Theorem [4](#page-7-6) to show that the 0-PPG algorithm (with the same procedure as that of Algorithm [1](#page-7-0) except the different values of V π<sup>α</sup> λ,π<sup>α</sup> in the policy evaluation step) converges to a stationary policy of the *performative value function* [\(131\)](#page-36-1), which by gradient dominance is a PO policy when the new value of µ satisfies µ ≥ 0.

1997

2014

2016

2018 2019

2024

2026

2029

# K. Adjusting Our Results to the Existing Quadratic Regularizer

$$V_{\lambda,\pi}^\pi = \langle d_{\pi,p_\pi}, r_\pi \rangle - \lambda \|d_{\pi,p_\pi}\|^2. \quad (131)$$