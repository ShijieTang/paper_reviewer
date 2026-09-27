# Achieve Performatively Optimal Policy For Performative Reinforcement Learning

Anonymous Authors1

## Abstract

Performative reinforcement learning is an emerging dynamical decision making framework, which extends reinforcement learning to the common applications where the agent's policy can change the environmental dynamics. Existing works on performative reinforcement learning only aim at a performatively stable (PS) policy that maximizes an approximate value function. However, there is a provably positive constant gap between the PS policy and the desired performatively optimal (PO) policy that maximizes the original value function. In contrast, this work proposes a zerothorder performative policy gradient (0-PPG) algorithm that **for the first time converges to the** desired PO policy with polynomial computation complexity under mild conditions. For the convergence analysis, we prove two important properties of the nonconvex value function. First, when the policy regularizer dominates the environmental shift, the value function satisfies a certain gradient dominance property, so that any stationary point of the value function is a desired PO.

Second, though the value function has unbounded gradient, we prove that all the sufficiently stationary points lie in a convex and compact policy subspace Π∆, where the policy value has a constant lower bound ∆ > 0 and thus the gradient becomes bounded and Lipschitz continuous.

## 1. Introduction

Reinforcement learning is a powerful dynamic decision making framework with many successes in AI, such as AlphaGo (Silver et al., 2017), AlphaStar (Vinyals et al., 2019), Pluribus (Brown and Sandholm, 2019), large language model alignment (Bai et al., 2022) and reasoning 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author
<anon.email@domain.com>.

1 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054
(Havrilla et al., 2024). However, most reinforcement learning works ignore the effect of the deployed policy on the environmental dynamics, including transition kernel and reward function. This effect is significant in some applications. For example, the behavior of the autonomous vehicles can affect the behavior of the pedestrians and the other vehicles, so the environment may become very different from the designers' imagination (Nikolaidis et al., 2017). Also, a recommender system formulated as a contextual Markov decision process not only affects the user demographics (context distribution) but also how users interact with the platforms (Chaney et al., 2018; Mansoury et al., 2020). To account for such effect of deployed policy on environmental dynamics, performative reinforcement learning has been proposed by (Mandal et al., 2023) where the transition kernel pπ and reward function rπ are modeled as functions of the deployed policy π. Similar to conventional reinforcement learning, the ultimate goal is to find the performatively optimal (PO) policy that maximizes the performative value function, defined as the accumulated discounted reward when deploying a policy π to its corresponding environment
(pπ, rπ). However, the policy-dependent environmental dynamics pose significant challenge to achieve PO. Hence, (Mandal et al., 2023) pursues a suboptimal performatively stable (PS) policy using repeated retraining method with environmental dynamics fixed for the current policy at each policy optimization step. However, (Mandal et al., 2023) shows that PS can have a positive constant distance to PO. Two extensions of the basic performative reinforcement learning problem (Mandal et al., 2023) have been proposed and studied. (Rank et al., 2024) extends to the setting where the environmental dynamics gradually adjust to the currently deployed policy, and proposes a mixed delayed repeated retraining algorithm with accelerated convergence to a PS policy. (Mandal and Radanovic, 2024) extends (Mandal et al., 2023) from tabular setting to linear Markov decision processes with large number of states, and also obtains the convergence rate of the repeated retraining algorithm to a PS policy. In sum, all these existing performative reinforcement learning works pursue a suboptimal PS policy. Therefore, we want to ask the following fundamental research question.

## 1.1. Our Contributions

We will answer affirmatively to the research question above in the following steps. Each step yields a novel contribution.

- We study an entropy regularized performative reinforcement learning problem, compatible with the basic performative reinforcement learning problem in (Mandal et al., 2023). We prove that the objective function satisfies a certain gradient dominance condition, which implies that an approximate stationary point (not the suboptimal PS) is the desired approximate PO policy, under a mild regularizer dominance condition similar to that used by (Mandal et al., 2023; Rank et al., 2024; Mandal and Radanovic, 2024) to ensure convergence to a suboptimal PS policy. The proof adopts novel techniques such as recursion for pπ-related error term and frequent switch among various necessary and sufficient conditions of smoothness and strong concavity like properties for various variables (see Section 3.2). - We obtain a policy lower bound as a decreasing function of a stationary measure. This bound not only implies the unbounded *performative policy gradient* (a challenge to obtain a stationary policy and thus PO), but also inspires us to find a stationary policy in the policy subspace Π∆ with a constant policy lower bound ∆ > 0 where we prove the objective function to be Lipschitz continuous and Lipschitz smooth (a solution to this challenge). The policy lower bound is obtained using a novel technique which simplifies a complicated inequality of the minimum policy value π[amin(s)|s]
in two cases (see Section 3.3).

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109
- We construct a zeroth-order estimation of the performative policy gradient and obtains its estimation error. This is more challenging than the existing zero-th order estimation methods since our objective function is only well-defined on the policy space, a compact subset of a linear subspace of the Euclidean space R
|S||A|. To solve this puzzle, we adjust a two-point estimation to the linear subspace L0 of policy difference, and simplify the estimation error analysis by mapping policies onto the Euclidean space R
|S|(*|A|−*1)
via orthogonal transformation (see Section 4.1). - We propose a zeroth-order performative policy gradient (0-PPG) algorithm (see Algorithm 1) by combining the performative policy gradient estimation above with the Frank- Wolfe algorithm. Then we obtain a polynomial computation complexity of our 0-PPG algorithm to converge to a stationary policy, which is also the desired PO policy under the regularizer dominance condition above. The convergence analysis uses a policy averaging technique to show that an approximate stationary policy on Π∆ is also approximately stationary on the whole policy space Π (see Section 4.2).

Finally, we briefly show that the results above, including gradient dominance, Lipschitz properties and the finite-time convergence of 0-PPG algorithm to the desired PO, can be adjusted to the performative reinforcement learning problem with the quadratic regularizer used by (Mandal et al., 2023; Rank et al., 2024) (see Appendix K).

## 2. Preliminary: Performative Reinforcement Learning 2.1. Problem Formulation

Performative reinforcement learning is characterized by a Markov decision process (MDP) Mπ = (S, A, pπ, rπ, ρ) that depends on a certain policy π. Here, S and A denote the finite state space with cardinality |S| and finite action space with cardinality |A| respectively. The policy π ∈ [0, 1]*|S||A|*,
with entries π(a|s) for any state s ∈ S and action a ∈ A, lies in the following policy space, such that π(·|s) for any state s can be seen as a distribution over A.

$$\Pi\stackrel{\mathrm{def}}{=}\Big\{\pi\in[0,1]^{|\mathcal{S}||\mathcal{A}|}:\sum_{a\in\mathcal{A}}\pi(a|s)=1,\forall s\in\mathcal{S}\Big\}.$$

The transition kernel pπ ∈ [0, 1]|S|2|A| dependent on policy π ∈ Π, with entries pπ(s
′|*s, a*) for any s, s′ ∈ S and a ∈
A, lies in the following transition kernel space such that pπ(·|*s, a*) can be seen as a state distribution on S.

$${\mathcal{P}}\ {\stackrel{\mathrm{def}}{=}}\ \Bigl\{p\in[0,1]^{|{\mathcal{S}}|^{2}|{\mathcal{A}}|}:\!\!\sum_{s\in{\mathcal{S}}}p(s^{\prime}|s,a)\!=\!1,\forall s\!\in\!{\mathcal{S}},a\!\in\!{\mathcal{A}}\Bigr\}.$$

rπ ∈ R 
def 
= [0, 1]*|S||A|* is the reward function with entries rπ(*s, a*) ∈ [0, 1] for any s ∈ S and a ∈ A. ρ ∈ [0, 1]|S| is the initial state distribution such that Ps∈S ρ(s) = 1.

Note that we consider pπ, rπ, ρ, π as Euclidean vectors, so that we can conveniently define their Euclidean norm. For example, we define ∥pπ∥q =P*s,a,s*′ |pπ(s
′|*s, a*)| q1/q for any q > 1 and ∥pπ∥∞ = max*s,a,s*′ |pπ(s
′|*s, a*)|. Such norms can be similarly defined over rπ, ρ, π by summing or maximizing over all the entries. Specifically, denote
∥ · ∥ = *∥ · ∥*2 by convention. When an agent applies its policy π ∈ Π to MDP Mπ′ =
(S, A, pπ′ , rπ′ , ρ), the initial environmental state s0 ∈ S
is generated from the distribution ρ. Then at each time t = 0, 1, 2*, . . .*, the agent takes a random action at ∼ π(·|st) based on the current state st ∈ S, the environment transitions to the next state st+1 ∼ pπ′ (·|st, at) and provides reward rt = rπ′ (st, at) ∈ [0, 1] to the agent. The value of applying policy π to Mπ′ can be characterized by the following *value function*.

$$V_{\lambda,\pi^{\prime}}^{\pi}\stackrel{\mathrm{def}}{=}\mathbb{E}_{\pi,p_{\pi^{\prime}},p}\bigg[\sum_{t=0}^{\infty}\gamma^{t}r_{\pi^{\prime}}(s_{t},a_{t})\bigg]-\lambda\mathcal{H}_{\pi^{\prime}}(\pi).\tag{1}$$

Q: *Can we design an algorithm that converges to* the desired performatively optimal (PO) policy?

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Here, Eπ,pπ′ ,ρ is the expectation under policy π, transition kernel pπ′ and initial state distribution ρ. γ ∈ (0, 1) is the discount factor. Hπ′ (π) is a regularizer with coefficient λ ≥ 0 to ensure or accelerate algorithm convergence. Existing works use the quadratic regularizers such as Hπ′ (π) = 12
∥dπ,pπ′ ∥
2(Mandal et al., 2023; Rank et al., 2024) and Hπ′ (π) = 12
∥Φ
⊤dπ,pπ′ ∥
2(Mandal and Radanovic, 2024) with a feature matrix Φ, where the occupancy measure dπ,p ∈ [0, 1]*|S||A|* for any policy π and transition kernel p is defined as the following distribution on *S × A*.

$$d_{\pi,p}(s,a)\stackrel{{\mathrm{def}}}{{=}}(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}\mathbb{P}_{\pi,p,\rho}\{s_{t}=s,a_{t}=a\}.\tag{2}$$

Then the state occupancy measure defined as dπ,p(s)
def P 
=
a dπ,p(s, a) satisfies the following well-known Bellman equation for any state s
′ ∈ S.

$$d_{\pi,p}(s^{\prime})=(1-\gamma)\rho(s^{\prime})+\gamma\sum_{s,a}d_{\pi,p}(s)\pi(a|s)p(s^{\prime}|s,a).\tag{3}$$

The ultimate goal of performative reinforcement learning is to find the *performatively optimal (PO)* policy π that maximizes the *performative value function* V
π λ,π (with π
′ =
π in Eq. (1)), as formally defined below.

Definition 1 (Ultimate Goal: PO). For any ϵ ≥ 0*, a policy* π ∈ Π is defined as ϵ-performatively optimal (ϵ*-PO) if* maxπ′∈Π V
π
′
λ,π′ − V
π λ,π ≤ ϵ. Specifically, we call a 0-PO
policy as a PO policy. Conventional reinforcement learning can be seen as a special case of performative reinforcement learning with fixed environmental dynamics, namely, constant transition kernel pπ ≡ p and constant reward function rπ ≡ r. However, this may fail on applications with policy-dependent environmental dynamics, such as recommender system and autonomous driving (Mandal et al., 2023) as explained in Section 1.

## 2.2. **Performatively Stable (Ps) Policy In Existing Works**

Achieving an ϵ-PO policy (defined by Definition 1) is challenging, due to the policy-dependent environmental dynamics pπ and rπ. To alleviate the challenge, all the existing works (Mandal et al., 2023; Rank et al., 2024; Mandal and Radanovic, 2024) aim at a *performatively stable (PS)* policy πPS defined as follows, as an approximation of a *PO policy*.

$$\pi_{\mathrm{PS}}\in\arg\operatorname*{max}_{\pi\in\Pi}V_{\lambda,\pi_{\mathrm{PS}}}^{\pi}.$$
λ,πPS . (4)
In other words, a PS policy πPS has the optimal value on the fixed environment MπPS . However, (Mandal et al.,
2023) shows that a PS policy can be suboptimal, so these existing algorithms cannot converge to a PO policy. Nevertheless, we will briefly introduce these algorithms, to later partially inspire and compare with our method for achieving a PO policy. Note that an occupancy measure d (a distribution on *S × A*) corresponds to the policy π d defined as π d(a|s) = d(s,a)
d(s)(π d(a|s) = 1/|A| if d(s) = 0), where d(s)=Pa′∈A d(*s, a*′). Hence, (Mandal et al., 2023; Rank et al., 2024; Mandal and Radanovic, 2024) transform the policy optimization problem (4) into a problem of solving d. The basic performative reinforcement learning (Mandal et al., 2023) considers the following dual optimization problem of d in the environment pd′ = pπd′ , rd′ = rπd′
corresponding to another occupancy measure d
′.

$$\left\{\begin{array}{l}\max_{d:\mbox{distribution on}\mathcal{S}\times\mathcal{A}\sum_{s,a}d(s,a)r_{d^{\prime}}(s,a)-\frac{\lambda}{2}\|d\|^{2}}\\ \mbox{s.t.}\sum_{a}d(s,a)=\rho(s)+\gamma\sum_{s^{\prime},a}d(s^{\prime},a)p_{d^{\prime}}(s|s^{\prime},a)\end{array}\right..\tag{5}$$

The objective function above corresponds to the value function V
π λ,π′ defined in Eq. (1) with quadratic regularizer Hπ′ (π)= 12
∥dπ,pπ′ ∥
2. The equality constraint above comes from the Bellman equation (3). Denote ϕ(d
′) as the optimal solution to the problem (5) above. Then the target becomes a performatively stable occupancy measure dPS defined as a fixed point dPS = ϕ(dPS), which corresponds to a PS policy πPS = π dPS . Suppose the transition kernel and reward function are sensitive with parameters ϵ
′p
, ϵ′r > 0 respectively, that is, for any occupancy measures *d, d*′.

$$||p_{d^{\prime}}-p_{d}||\leq\epsilon^{\prime}_{p}||d^{\prime}-d||,\ ||r_{d^{\prime}}-r_{d}||\leq\epsilon^{\prime}_{r}||d^{\prime}-d||.\tag{6}$$

It has been proved by (Mandal et al., 2023) that ϕ is a contraction mapping under a regularizer dominance condition that λ > O(ϵ
′p + ϵ
′r). In this case, any repeated retraining method characterized by dt+1 ≈ ϕ(dt) with sufficient precision can converge to the PS policy. Similarly, (Rank et al., 2024; Mandal and Radanovic, 2024) also apply repeated retraining to optimization problems of occupancy measure, which converges to a PS policy for extensions of the basic performative reinforcement learning (Mandal et al., 2023). Next, we will propose our significantly different strategies to achieve the desired PO policy.

## 3. Entropy Regularized Performative Reinforcement Learning

In this section, we obtain critical properties of an entropy regularized performative reinforcement learning problem for achieving the desired PO policy.

## 3.1. Negative Entropy Regularizer

To achieve the PO policy, one might attempt to solve the problem (Pd), adjusted from the dual problem (5) above with fixed d
′replaced by the decision variable d. The solution dPO will yield the PO policy π dPO . However, such replacement will make the convex quadratic optimization problem (5) much more complicated, due to the unknown and possibly complicated functions pd and rd. Therefore, we will instead focus on the primal problem maxπ V
π λ,π.

We consider the following negative entropy regularizer of the policy π, which is widely used in reinforcement learning to encourage environment exploration and accelerate convergence (Mnih et al., 2016; Mankowitz et al., 2019; Cen et al., 2022; Chen and Huang, 2024).

$${\mathcal{H}}_{\pi^{\prime}}(\pi)=\mathbb{E}_{\pi,p_{\pi^{\prime}},\rho}\Big[\sum_{t=0}^{\infty}\gamma^{t}\log\pi(a_{t}|s_{t})\Big].$$
i. (7)
In addition, this negative entropy regularizer can be seen as a strongly convex function of the occupancy measure dπ,pπ′(proved in Appendix B), which is critical to develop algorithms convergent to a PO (see Theorem 1 later) or PS policy (Mandal et al., 2023). For optimization problem on a probability simplex variable (policy π or occupancy measure d), negative entropy regularizer is more natural and yields faster theoretical convergence than the quadratic regularizers used in the existing performative reinforcment learning works (Mandal et al., 2023; Rank et al., 2024) (see pages 43-45 of (Chen, 2020) for explanation). Therefore, we will mainly focus on the following entropyregularized value function, which is obtained by substituting the negative entropy regularizer (7) into the general value function (1).

$$V_{\lambda,\pi^{\prime}}^{\pi}\stackrel{{\rm def}}{{=}}\mathbb{E}_{\pi,p_{\pi^{\prime}},\rho}\biggl{[}\sum_{t=0}^{\infty}\gamma^{t}[r_{\pi^{\prime}}(s_{t},a_{t})-\lambda\log\pi(a_{t}|s_{t})]\biggr{]}.\tag{8}$$

Specifically, we will study the critical properties of the entropy-regularized value function (8) (Section 4) to develop algorithm that converges to PO (Sections 4.1-4.2). Then we will briefly discuss about how to adjust these results to the existing quadratic regularizers (Appendix K). We make the following standard assumptions to study the properties of the entropy-regularized value function (8).

Assumption 1 (Sensitivity). There exist constants ϵp, ϵr > 0 such that for any *π, π*′ ∈ Π,

$$\|p_{\pi^{\prime}}-p_{\pi}\|\leq\epsilon_{p}\|\pi^{\prime}-\pi\|,\ \|r_{\pi^{\prime}}-r_{\pi}\|\leq\epsilon_{r}\|\pi^{\prime}-\pi\|\tag{9}$$

Assumption 2 (Smoothness). pπ and rπ are Lipschitz smooth with modulus Sp, Sr > 0 respectively, that is, for any π ∈ Π, s, s′ ∈ S, a ∈ A*, we have* 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219

$$\|\nabla_{\pi}p_{\pi^{\prime}}(s^{\prime}|s,a)-\nabla_{\pi}p_{\pi}(s^{\prime}|s,a)\|\leq S_{p}\|\pi^{\prime}-\pi\|,\tag{10}$$ $$\|\nabla_{\pi}r_{\pi^{\prime}}(s,a)-\nabla_{\pi}r_{\pi}(s,a)\|\leq S_{r}\|\pi^{\prime}-\pi\|.\tag{11}$$

Assumption 3. There exists a constant D > 0 *such that* infπ∈Π,p∈P,s∈S dπ,p(s) ≥ D.

Assumptions 1-2 ensure that the environmental dynamics pπ and rπ adjust continuously and smoothly to policy π, and thus the *performative value function* V
π λ,π is differentiable with performative policy gradient ∇πV
π λ,π. Similar versions of Assumption 1 on environmental sensitivity have been used in the performative reinforcement learning literature (e.g. Eq. (6) in (Mandal et al., 2023)). Assumption 3 has been used (Zhang et al., 2021) or implied by stronger assumptions (Wei et al., 2021; Chen et al., 2022; Agarwal et al., 2021; Leonardos et al., 2022; Wang et al., 2023; Chen and Huang, 2024; Bhandari and Russo, 2024) in conventional reinforcement learning (see Appendix C for the proof), which guarantees that each state is visited sufficiently often.

$$\mathbf{\Sigma}(7)$$

## 3.2. Gradient Dominance

For the nonconvex policy optimization problem maxπ∈Π V
π λ,π with the entropy regularized value function
(8) on the convex policy space Π, it is natural to consider its approximate stationary solution as defined below. Definition 2 (Stationary Policy). For any ϵ ≥ 0*, a policy* π ∈ Π is ϵ*-stationary if* maxπ˜∈Π∇πV
π λ,π, π˜ − π≤ ϵ. We call a 0-stationary policy as a stationary policy. Note that for a policy to be the desired PO, it is necessary to be stationary, while the PS policy targeted by existing works is neither necessary nor sufficient. Furthermore, we will show that stationary policy can also be a sufficient condition of the desired PO under mild conditions. As a preliminary step, we show the important gradient dominance property of the objective function as follows. Theorem 1 (Gradient Dominance). *Under Assumptions* 13, the entropy regularized value function (8) satisfies the following gradient dominance property for any π0, π1 ∈ Π.

$$V_{\lambda,\pi_{1}}^{\pi_{1}}\leq V_{\lambda,\pi_{0}}^{\pi_{0}}+D^{-1}\max_{\pi\in\Pi}\left\langle\nabla_{\pi_{0}}V_{\lambda,\pi_{0}}^{\pi_{0}},\pi-\pi_{0}\right\rangle$$ $$-\frac{\mu}{2}\|\pi_{1}-\pi_{0}\|^{2},$$
$$(12)$$

where the constant µ ∈ R *is defined as follows.*

$$\begin{array}{c}{{\mu=\!\!\frac{D\lambda}{1-\gamma}-\frac{6\gamma|\mathcal{S}|(1+\lambda\log|\mathcal{A}|)}{D(1-\gamma)^{3}}}}\\ {{\qquad\left[\epsilon_{p}\big(\sqrt{|\mathcal{A}|}+\gamma\epsilon_{p}\sqrt{|\mathcal{S}|}\big)+S_{p}(1-\gamma)\right]}}\\ {{\qquad-\frac{S_{r}(1-\gamma)+4\epsilon_{r}(\sqrt{|\mathcal{A}|}+\gamma\epsilon_{p}\sqrt{|\mathcal{S}|})}{D^{2}(1-\gamma)^{2}}.}}\end{array}$$
$$(13)$$
2. (13)
Remark: With sufficiently large regularizer strength λ and small environmental shift strength ϵp, ϵr, Sp, Sr (i.e.,
when the regularizer dominates the environmental shift), we have µ ≥ 0, which implies the gradient dominance form (Eq. (12) with µ = 0) that holds for conventional unregularized reinforcement learning (see Lemma 4 of (Agarwal

$$J_{\lambda}(\pi,\pi^{\prime},p,r)$$
$$J_{\lambda}(\pi,\pi\,,p,r)$$ $$\stackrel{{\rm def}}{{=}}\mathbb{E}_{\pi,p}\left[\sum_{t=0}^{\infty}\gamma^{t}[r(s_{t},a_{t})-\lambda\log\pi^{\prime}(a_{t}|s_{t})]\right|s_{0}\!\sim\!\rho\right].\tag{14}$$

To get the intuition, we consider the following three cases from the simplest conventional reinforcement learning to the hardest performative reinforcement learning. (Case I): For conventional reinforcement learning with fixed dynamics pπ ≡ p and rπ ≡ r, denote dα = αdπ1,p +
(1 − α)dπ0,p (α ∈ [0, 1]). Based on the Bellman equation
(3), dα = dπα,p is the occupancy measure of the policy πα(a|s) = 
dα(s,a)
dα(s)
. Therefore, V
πα λ,πα can be rewritten as Jλ(πα, πα*, p, r*) = Ps,a dα(s, a)[r(*s, a*) − λ log πα(a|s)],
which has the following strong concavity like property by Pinsker's inequality (see Eq. (91) for detail).

$$e_{\alpha}(s^{\prime})\!=\!\gamma\sum_{s,a}\left[e_{\alpha}(s)\pi_{\alpha}(a|s)p_{\pi_{\alpha}}(s^{\prime}|s,a)+h_{\alpha}(s,a,s^{\prime})\right],$$

Intuition and Novelty for Proving Theorem 1: Define the following more refined value function 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

$$\begin{array}{c}{{J_{\lambda}(\pi_{\alpha},\pi_{\alpha},p,r)-\alpha J_{\lambda}(\pi_{1},\pi_{1},p,r)}}\\ {{-(1-\alpha)J_{\lambda}(\pi_{0},\pi_{0},p,r)}}\\ {{=\frac{1}{1-\gamma}\sum_{s}\left[\alpha d_{1}(s)\mathrm{KL}[\pi_{1}(\cdot|s)\|\pi_{\alpha}(a|s)]\right.}}\\ {{\left.+(1-\alpha)d_{0}(s)\mathrm{KL}[\pi_{0}(\cdot|s)\|\pi_{\alpha}(a|s)]\right]}}\\ {{\geq\frac{D\lambda\alpha(1-\alpha)}{2(1-\gamma)}\|\pi_{1}-\pi_{0}\|^{2}.}}\end{array}$$

(Case II): Consider a harder case with varying pπ and constant reward rπ ≡ r. Similarly, we denote dα =
αdπ1,pπ1
+ (1 − α)dπ0,pπ0 and πα(a|s) = dα(s,a)
dα(s)
. The varying pπ brings a major challenge that dα = dπα,pπα required by Case I no longer holds. To solve this challenge, we prove that the error term eα(s) = dπα,pα
(s) − dα(s) of interest satisfies the following novel recursion (see Eq. (89) for the derivation based on the Bellman equation (3)).

dα(*s, a*)pπα
(s
′|*s, a*) is a Lipschitz smooth function of α with Lipschitz constant ℓdp(s, a) defined by Eq. (87), we have |hα(s, a, s′)| ≤ α(1−α)
2ℓdp(*s, a*), which can be substituted into the recursion above and yields the following novel error bound (see Eq. (90) for detail).

$$\sum_{s}|e_{\alpha}(s)|\leq\alpha(1-\alpha){\mathcal{O}}(\epsilon_{p}+S_{p})\|\pi_{1}-\pi_{0}\|^{2},$$

which implies the desired strong concavity like property as follows.

$$J_{\lambda}(\pi_{\alpha},\pi_{\alpha},p_{\alpha},r)-\alpha J_{\lambda}(\pi_{1},\pi_{1},p_{1},r)$$ $$-(1-\alpha)J_{\lambda}(\pi_{0},\pi_{0},p_{0},r)$$ $$\geq\mbox{Eq.~{}(15)}-\alpha(1-\alpha)(1+\lambda)\mathcal{O}(\epsilon_{p}+S_{p})\|\pi_{1}-\pi_{0}\|^{2}$$ $$\geq\frac{\alpha(1-\alpha)\mu_{1}}{2}\|\pi_{1}-\pi_{0}\|^{2}\tag{16}$$

where µ1 =Dλ 2(1−γ)−(1+λ)O(ϵp+Sp) defined by Eq. (92)
equals µ defined by Eq. (13) when ϵr = Sr = 0. (Case III): Now we consider performative reinforcement learning with varying pπ and rπ. The policy πα and its occupancy measure dα are the same as in Case II above. Then the function w(α) = αJλ(π1, π1, p1, rα) + (1 − α)Jλ(π0, π0, p0, rα) can be proved Lipschitz smooth with parameter µ2 = O(Sr + ϵr) defined by Eq. (94), so using r = rα in Eq. (16) we obtain the following strong concavity like property.

$J_{\lambda}(\pi_{\alpha},\pi_{\alpha},p_{\alpha},r_{\alpha})-\alpha J_{\lambda}(\pi_{1},\pi_{1},p_{1},r_{1})$  $-(1-\alpha)J_{\lambda}(\pi_{0},\pi_{0},p_{0},r_{0})$  $\alpha(1-\alpha)\cdot$
$$\geq\frac{\alpha(1-\alpha)\mu_{1}}{2}\|\pi_{1}-\pi_{0}\|^{2}+w(\alpha)-\alpha w(1)-(1-\alpha)w(0)$$ $$\geq\frac{\alpha(1-\alpha)(\mu_{1}-\mu_{2})}{2}\|\pi_{1}-\pi_{0}\|^{2}.$$
Rearranging the inequality above, we obtain the following inequality of V
πα λ,πα
= Jλ(πα, πα, pα, rα).

$$(15)$$
$$\frac{V_{\lambda,\pi_{\alpha}}^{\pi_{\alpha}}-V_{\lambda,\pi_{0}}^{\pi_{0}}}{\alpha}\geq V_{\lambda,\pi_{1}}^{\pi_{1}}-V_{\lambda,\pi_{0}}^{\pi_{0}}+\frac{\mu(1-\alpha)}{2}\|\pi_{1}-\pi_{0}\|^{2},$$

where µ = µ1 − µ2 is exactly defined by Eq. (13). Letting α → +0 above, we have

$$V_{\lambda,\pi_{1}}^{\pi_{1}}\leq V_{\lambda,\pi_{0}}^{\pi_{0}}+\left[\frac{d}{d\alpha}V_{\lambda,\pi_{\alpha}}^{\pi_{\alpha}}\right]\Big|_{\alpha=0}-\frac{\mu}{2}\|\pi_{1}-\pi_{0}\|^{2}.$$

Using the chain rule, we can find a policy π
∗
0such that
-d dα V
πα λ,πα
α=0 ≤ D∇π0 V
π0 λ,π0
, π∗0 − π0, which along with the bound above proves the gradient dominance property (12).

## 3.3. Policy Lower Bound And Lipschitz Properties

Policy Lower Bound: Based on Section 3.2, we can focus on achieving an ϵ-stationary policy. A major challenge is

where $h_{\alpha}(s,a,s^{\prime})=d_{\alpha}(s,a)p_{\pi_{\alpha}}(s^{\prime}|s,a)$  $\alpha$\(\
−αd1(s, a)pπ1
(s
′|s, a)−(1−α)d0(*s, a*)pπ0
(s
′|*s, a*). Since 5 et al., 2021)). In this case, stationary policy becomes a sufficient condition of the desired PO, as shown in the following Corollary 1. Note that the existing performative reinforcement learning works (Mandal et al., 2023; Rank et al., 2024; Perdomo et al., 2020) also require a regularizer dominance condition similar to our µ ≥ 0 (e.g. λ > O(ϵ
′p + ϵ
′r) in
(Mandal et al., 2023)) to ensure convergence to a PS policy. Corollary 1. Under Assumptions 1-3, if µ ≥ 0 for µ defined in Eq. (13), then any Dϵ-stationary policy is also the desired ϵ-PO policy. Furthermore, if µ > 0*, the PO policy is unique.*
the unbounded *performative policy gradient* ∇πV
π λ,π on Π.

Specifically, we will show that as π(a|s) → 0 for any state s and action a, ∥∇πV
π λ,π∥ → +∞. To tackle this challenge, we prove the following policy lower bound.

Theorem 2. If Assumptions 1 and 3 hold, and pπ, rπ are differentiable functions of π*, then the following policy lower* bound holds for any π ∈ Π, s ∈ S, a ∈ A.

$$\pi(a|s)\!\geq\!\pi_{\min}\exp\biggl{[}-\frac{2|{\cal A}|}{\lambda}(1-\gamma)\langle\nabla_{\pi}V_{\lambda,\pi}^{\pi},\pi^{\prime}-\pi\rangle\biggr{]}.\tag{17}$$  _Here, we define the following constant $\pi_{\min}$ and policy $\pi^{\prime}$._
$$\pi_{\min}\stackrel{{\rm def}}{{=}}\frac{1}{2|{\cal A}|^{1/(1-\gamma)}}\exp\bigg{\{}-\frac{1}{\lambda(1-\gamma)}$$ $$-\frac{2|{\cal A}|\sqrt{2|{\cal S}|}}{\lambda}\bigg{[}\frac{\epsilon_{p}\sqrt{|{\cal S}|}(1+\lambda\log|{\cal A}|)}{1-\gamma}+\epsilon_{r}\bigg{]}\bigg{\}},\tag{18}$$
$$\pi^{\prime}(a|s)=\begin{cases}\pi[a_{\min}(s)|s],&a=a_{\max}(s)\\ \pi[a_{\max}(s)|s],&a=a_{\min}(s)\;,\\ \pi(a|s),&\text{Otherwise}\end{cases}\tag{19}$$

where amax(s) ∈ arg maxaπ(a|s) and amin(s) ∈ arg minaπ(a|s).

Implications of Theorem 2: First, as π(a|s) → 0, we have ⟨∇πV
π λ,π, π′ −π⟩ → +∞, so ∥∇πV
π λ,π∥ → +∞ as aforementioned. Second, any stationary policy π satisfies
⟨∇πV
π λ,π, π′ − π⟩ ≤ 0, so π(a|s) ≥ πmin. Therefore, we can search ϵ-stationary policy on the convex and compact policy subspace Π∆
def = {π ∈ Π : π(a|s) ≥ ∆} with lower bound ∆ ∈ (0, πmin].

Intuition and Novelty for Proving Theorem 2: As a preliminary step, consider a conventional reinforcement learning problem with fixed environmental dynamics pπ ≡ p and rπ ≡ r. In this case, ∇πV
π λ,π has analytical form (see Eq. (98)) based on policy gradient theorem, so by direct computation we obtain the following bound (see Eq. (99) for detail)

$$\begin{array}{l}{{\langle\nabla_{\pi}V_{\lambda,\pi}^{\pi},\pi^{\prime}-\pi\rangle{\geq}{\frac{1}{1-\gamma}}{\underset{s}{\operatorname*{max}}}\biggl\{}\bigl(\pi[a_{\operatorname*{max}}(s)|s]-\pi[a_{\operatorname*{min}}(s)|s]\bigr)}}\\ {{\qquad\qquad\biggl[\lambda\log{\frac{\pi[a_{\operatorname*{max}}(s)|s]}{\pi[a_{\operatorname*{min}}(s)|s]}}-1-{\frac{\gamma(1+\lambda\log|{\mathcal A}|)}{1-\gamma}}\biggr]\biggr\}.}}\end{array}$$
$$\pi[a_{\mathrm{min}}(s)|s]\!\geq\!\pi_{\mathrm{min}}^{\prime}\mathrm{exp}\Big[-\frac{2|{\mathcal{A}}|}{\lambda}(1\!-\!\gamma)\langle\nabla_{\pi}V_{\lambda,\pi}^{\pi},\pi^{\prime}\!-\!\pi\rangle\Big],$$

6 where π
′min is defined by Eq. (18) with ϵp = ϵr = 0.

Then by extending conventional reinforcement learning to performative reinforcement learning, ∇πV
π λ,π is perturbed by a magnitude of at most ϵp
√|S|(1+λ log |A|)
1−γ + ϵr (see Eq.

(102) for detail) based on the chain rule. This perturbation bound along with ∥π
′ − π∥ ≤ p2|S| yields the second line of Eq. (18) and proves Theorem 2. Lipschitz Properties: Furthermore, in the policy subspace Π∆, the *performative value function* V
π λ,π is actually Lipschitz continuous and Lipschitz smooth as shown below, which facilitates finding an ϵ-stationary policy in Π∆.

Theorem 3. Under Assumptions *1-2,* V
π λ,π satisfies the following Lipschitz propreties for any ∆ > 0 and *π, π*′ ∈ Π∆.

$$\begin{array}{c}{{|V_{\lambda,\pi^{\prime}}^{\pi^{\prime}}-V_{\lambda,\pi}^{\pi}|\leq\frac{L_{\lambda}}{\Delta}\|\pi^{\prime}-\pi\|,}}\\ {{\|\nabla_{\pi^{\prime}}V_{\lambda,\pi^{\prime}}^{\pi^{\prime}}-\nabla_{\pi}V_{\lambda,\pi}^{\pi}\|\leq\frac{\ell_{\lambda}}{\Delta}\|\pi^{\prime}-\pi\|.}}\end{array}$$
$$(20)$$
$$(21)$$
$$\underline{{A|)}}$$
$$(22)^{\frac{1}{2}}$$
′ − π∥, (20)
′ − π∥. (21)
where

$$L_{\lambda}\stackrel{{\rm def}}{{=}}\frac{\sqrt{|{\cal A}|}(2-\gamma+\gamma\lambda\log|{\cal A}|)+\epsilon_{p}\sqrt{|{\cal S}|}(1+\lambda\log|{\cal A}|)}{(1-\gamma)^{2}}$$
$$+\,{\frac{\epsilon_{r}}{1-\gamma}}$$
$$\underline{{|A|)}}$$
1 − γ(22)
$$\ell_{\lambda}\stackrel{{\rm def}}{{=}}\frac{3|{\cal A}|(1+\lambda\log|{\cal A}|)}{(1-\gamma)^{2}}+\frac{\epsilon_{p}\sqrt{|{\cal S}||{\cal A}|}(5+6\lambda\log|{\cal A}|)}{(1-\gamma)^{3}}$$ $$+\frac{\epsilon_{r}\big{[}\sqrt{|{\cal A}|}(1-\gamma)+\sqrt{|{\cal S}|}(\gamma+2\epsilon_{p})\big{]}}{|{\cal A}|(1-\gamma)^{2}}$$ $$+\frac{S_{p}\sqrt{|{\cal S}|}(1+\lambda\log|{\cal A}|)+S_{r}(1-\gamma)}{|{\cal A}|(1-\gamma)^{2}}.\tag{23}$$

## 4. **Zeroth-Order Performative Policy Gradient** (0-Ppg) Algorithm

4.1. Performative Policy Gradient Estimation In Section 3, we have obtained important properties of the entropy regularized *performative value function* V
π λ,π (defined by Eq. (8)), which indicates that it suffices to find an ϵstationary policy in the subspace Π∆ for ∆ ∈ (0, πmin]. To achieve this goal, an accurate estimation of the performative policy gradient ∇πV
π λ,π is important, which has two challenges. First, unlike conventional reinforcement learning where policy gradient has analytical form, such analytical form does not exist in performative reinforcement learning due to the arbitrary forms of pπ and rπ. Second, in practice, we cannot access the values of pπ(s
′|*s, a*) and rπ(*s, a*) but can only obtain stochastic samples from them (Mandal et al., 2023).

Despite these challenges in estimating ∇πV
π λ,π, note that V
π λ,π for any policy π can be evaluated, since it is actually 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 To directly solve the inequality above of π[amin(s)|s] is not easy. To simplify this inequality, we consider two cases, either π[amin(s)|s] ≥12 π[amax(s)|s] ≥1 2|A| or π[amin(s)|s] <12 π[amax(s)|s]. In the second case, we can replace π[amax(s)|s] and π[amax(s)|s] − π[amin(s)|s] above with their lower bounds 1 |A| and 1 2|A| respectively.

Then it becomes straightforward to obtain the policy lower bound. the policy evaluation problem in conventional reinforcement learning under fixed environment pπ and rπ (for fixed π). Furthermore, for any ϵV > 0 and η ∈ (0, 1), many existing policy evaluation algorithms such as temporal difference (Bhandari et al., 2018; Li et al., 2023; Samsonov et al.,
2023), can obtain Vˆ π λ,π ≈ V
π λ,π with the following ϵV error with probability at least 1 − η.

$$|\hat{V}_{\lambda,\pi}^{\pi}-V_{\lambda,\pi}^{\pi}|\leq\epsilon_{V}.$$

As a result, we will consider a zeroth-order estimation of
∇πV
π λ,π using policy evaluation. However, this has another challenge that V
π λ,π is only well-defined on π ∈ Π, so we cannot directly apply the existing zeroth-order estimation methods (Agarwal et al., 2010; Shamir, 2017; Malik et al., 2020) which require the objective function to be welldefined on a sphere. Fortunately, for any *π, π*′ ∈ Π, the policy difference π
′ − π lies in the following linear subspace of dimensionality |S|(*|A| −* 1).

$${\mathcal{L}}_{0}\ {\stackrel{\mathrm{def}}{=}}\ \Big\{u\in\mathbb{R}^{|{\mathcal{S}}||{\mathcal{A}}|}\colon\sum_{a}u(a|s){=}0,\forall s\in{\mathcal{S}}\Big\}.$$

Therefore, inspired by the popular two-point zeroth-order estimations, we obtain the following estimation of ∇πV
π λ,π.

$$\hat{g}_{\lambda,\delta}(\pi)\!=\!\frac{|\mathcal{S}|(|\mathcal{A}|\!-\!1)}{2N\delta}\sum_{i=1}^{N}(\hat{V}_{\lambda,\pi}^{\pi+\delta u_{i}}-\hat{V}_{\lambda,\pi}^{\pi-\delta u_{i}})u_{i},\tag{26}$$

where {ui}
N
i=1 are i.i.d. samples from the uniform distribution on U1 ∩ L0 with

$$U_{1}\ {\stackrel{\mathrm{def}}{=}}\ \{u\in\mathbb{R}^{|{\mathcal{S}}||{\mathcal{A}}|}:\|u\|{=}1\}.$$

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 Our estimation (26) above is more tricky than the existing two-point zeroth-order estimations (Agarwal et al., 2010; Shamir, 2017; Malik et al., 2020) where uiis uniformly distributed on U1. To elaborate, we replace their U1 with U1 ∩ L0, a complete unit sphere on the linear subspace L0, and further require π ∈ Π∆ and δ < ∆, to guarantee that π + δui, π − δui ∈ Π for any ui ∈ U1 ∩ L0 and thus the stochastic gradient estimation (26) is valid (see Appendix H for the proof of validity). Moreover, we use the following three steps to obtain ui uniformly from U1 ∩ L0: (1) Obtain vi from the uniform distribution on U1; (2) Project vi onto L0 as Eq. (28) below; (3) Normalize this projection as Eq.

(29) below.

$$\mathrm{proj}_{\mathcal{L}_{0}}(v_{i})(a|s)=v_{i}(a|s)-\frac{1}{|\mathcal{A}|}\sum_{a^{\prime}}v_{i}(a^{\prime}|s),$$ $$u_{i}=\frac{\mathrm{proj}_{\mathcal{L}_{0}}(v_{i})}{\|\mathrm{proj}_{\mathcal{L}_{0}}(v_{i})\|}.$$
′|s), (28)
. (29)
The gradient estimation (26) has the following provable error bound.

7 Proposition 1. For any ∆ > δ > 0, η ∈ (0, 1) and π ∈ Π∆, then the stochastic gradient gˆλ,δ(π) *defined by Eq.* (26) is valid and approximates the projected performative policy gradient projL0
(∇πV
π λ,π) *with the following error bound* with probability at least 1 − η.

$$\|\hat{g}_{\lambda,\delta}(\pi)-\text{proj}_{\mathcal{L}_{0}}(\nabla_{\pi}V_{\lambda,\pi}^{\pi})\|$$ $$\leq\frac{2|\mathcal{S}||\mathcal{A}|\epsilon_{V}}{\delta}+\frac{4L_{\lambda}|\mathcal{S}||\mathcal{A}|}{3N(\Delta-\delta)}\log\left(\frac{3N|\mathcal{S}||\mathcal{A}|}{\eta}\right)$$ $$+\frac{L_{\lambda}|\mathcal{S}||\mathcal{A}|}{\Delta-\delta}\sqrt{\frac{2}{N}\log\left(\frac{3N|\mathcal{S}||\mathcal{A}|}{\eta}\right)}+\frac{\delta\ell_{\lambda}}{\Delta-\delta}.\tag{30}$$
$$(24)^{\frac{1}{2}}$$

Remark: Proposition 1 above aims to approximate projL0
(∇πV
π λ,π) instead of ∇πV
π λ,π. This is sufficient to obtain an ϵ-stationary policy, because for any policies *π, π*′, the stationarity measure only involves ⟨∇πV
π λ,π, π′ − π⟩
which equals to ⟨projL0
(∇πV
π λ,π), π′ − π⟩ as π
′ − π ∈ L0.

Therefore, we only care about projL0
(∇πV
π λ,π).

$$(25)$$
$$(27)$$

The approximation error (30) has the order of OϵV
δ +
log(N/η)
√N+ δ, which can be arbitrarily small with sufficiently large batchsize N (for reducing the variance), small δ (for reducing the bias), and smaller policy evaluation error ϵV .

Intuition and Novelty for Proving Proposition 1: Unlike existing zeroth-order estimations on the whole Euclidean space, our estimation (30) is made on the policy space Π, which lies in the linear manifold L0 + |A|−1 ⊂
R 
|S||A|. The key to our proof is to find an orthogonal transformation T : R
|S|(*|A|−*1) → L0, so that the goal is simplified to analyze the gradient estimation of fλ(x)
def 
= V
T(x)+|A|−1 λ,T(x)+|A|−1 on any x ∈ R
|S|(|A|−1). In particular, the true gradient can be rewritten as ∇fλ(x) =
T
−1projL0∇πV
π λ,π
π=T(x)+|A|−1 using differentiability, and when ϵV = 0 (i.e., Vˆ π λ,π = V
π λ,π for any π ∈ Π), our estimated gradient (30) on the policy space Π can be rewritten as the following two-point estimator on R
|S|(*|A|−*1) (see Eq.

(112) for details).

$${\hat{g}}_{\lambda,\delta}(\pi)\!=\!{\frac{|{\cal S}|(|{\cal A}|\!-\!1)}{2N\delta}}\sum_{i=1}^{N}\left[f_{\lambda}(x+\delta{\bar{u}}_{i})\!-\!f_{\lambda}(x-\delta{\bar{u}}_{i})\right]{\bar{u}}_{i},$$
(28)  $\binom{29}{2}$  . 
where u˜i = T
−1(ui) is uniformly distributed on a unit sphere in R
|S|(*|A|−*1) and x = T
−1(π *− |A|*−1). Therefore, we can apply estimation analysis to the Euclidean space R 
|S|(*|A|−*1). Finally, it is straightforward extend the conclusion from ϵV = 0 to ϵV > 0 by adding the policy evaluation error terms (see Eq. (116)).

## 4.2. Zeroth-Order Performative Policy Gradient (0-Ppg) Algorithm

With the estimated gradient gˆλ,δ(πt) defined by Eq. (26), we can consider the following Frank-Wolfe algorithm to find an ϵ-stationary policy.

$$\begin{array}{r l}{{}}&{{}{\tilde{\pi}}_{t}=\arg\operatorname*{max}_{\pi\in\Pi_{\Delta}}\langle\pi,{\hat{g}}_{\lambda,\delta}(\pi_{t})\rangle,}\\ {{}}&{{}{\pi}_{t+1}=\pi_{t}+\beta({\tilde{\pi}}_{t}-\pi_{t}).}\end{array}$$
⟨π, gˆλ,δ(πt)⟩, (31)
Lemma 1. *The step* (31) *has the analytical solution below.*

$$\tilde{\pi}_{t}(a|s)=\begin{cases}\Delta;a\neq\tilde{a}_{t}(s)\\ 1-\Delta(|{\cal A}|-1);a=\tilde{a}_{t}(s)\end{cases},\tag{33}$$

where a˜t(s) ∈ arg maxa gˆλ,δ(πt)(a|s).

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 See the proof of Lemma 1 in Section A.1. Then combining the *performative policy gradient* estimation (see Section 3.1) with the Frank-Wolfe algorithm, we propose our zerothorder performative policy gradient (0-PPG) algorithm (see Algorithm 1). We obtain the following convergence result of Algorithm 1 in Theorem 4, the main theoretical result of this work, as follows.

Theorem 4. Suppose Assumptions 1-3 *hold. For any* 0 < ϵ ≤ min -24p2|S| ℓλD
,2λ 5|A|D2(1−γ)
,
288Lλ|S|1.5|A| Dπmin and η ∈ (0, 1), select the following hyperparameters for Algorithm 1: ∆ = πmin 3, β =
Dπminϵ 36ℓλ|S| , δ = O(ϵ), ϵV = O(ϵ 2),
N = O[ϵ
−2log(η
−1ϵ
−1)]*, and the number of iterations* T = O(ϵ
−2) *(see Eqs.* (122)-(127) in Appendix J for detailed expression of these hyperparameters). Then with probability at least 1 − η, the output policy π˜T˜ of Algorithm 1 is an Dϵ*-stationary policy. Furthermore, if* µ ≥ 0, π˜T˜ is also an ϵ*-PO policy. The total number of policy evaluations* is 2NT = O[ϵ
−4log(η
−1ϵ
−1)].

Comparison with Existing Works: Theorem 4 indicates that our 0-PPG algorithm for the first time converges to the desire PO policy with arbitrarily small precision ϵ in polynomial computation complexity, under the mild regularizer dominance condition that µ ≥ 0. In contrast, existing works only converge to a suboptimal PS policy under a similar regularizer dominance condition (Mandal et al., 2023; Rank et al., 2024; Mandal and Radanovic, 2024). Our preferable convergence result is due to the major algorithmic difference that existing works adopt repeated retraining algorithms with iteration πt+1 ≈ arg maxπ∈ΠV
πt λ,π where the policy π is deployed in a fixed environment Mπt with π ̸= πt, whereas our 0-PPG algorithm evaluates V
π λ,π where each policy π is always deployed at its corresponding environment Mπ.

Intuition and Novelty for Proving Theorem 4: Standard convergence analysis of Frank-Wolfe algorithm yields that maxπ˜∈Π∆ ⟨∇πV
πT˜
λ,πT˜
, π˜ − πT˜⟩ ≤ 
Dϵ 2on Π∆. However, it requires a trick to prove the following Proposition 2 which implies that πT˜ is Dϵ-stationary on Π.

Proposition 2. If ∆ ≤ πmin/3 and a policy π *satisfies* maxπ˜∈Π∆ ⟨∇πV
π λ,π, π˜−π⟩ ≤ Dλ 5|A|(1−γ)
, then the stationary measures on Π∆ and Π *bound each other as follows.*

$$\max_{\pi\in\Pi}\langle\nabla_{\pi}V^{\pi}_{\lambda,\pi},\bar{\pi}-\pi\rangle\leq2\max_{\bar{\pi}\in\Pi_{\Delta}}\langle\nabla_{\pi}V^{\pi}_{\lambda,\pi},\bar{\pi}-\pi\rangle\tag{34}$$
$$(31)$$

To prove Proposition 2, note that π
′ defined by Eq. (19) also belongs to Π∆, so Theorem 2 implies π(a|s) ≥ 2∆. Then for any π2 ∈ Π, we have π2+π 2∈ Π∆ and thus

$$\max_{\pi_{2}\in\Pi}\langle\nabla_{\pi}V^{\pi}_{\lambda,\pi},\pi_{2}-\pi\rangle=2\max_{\pi_{2}\in\Pi}\left\langle\nabla_{\pi}V^{\pi}_{\lambda,\pi},\frac{\pi_{2}+\pi}{2}-\pi\right\rangle,$$ $$\leq2\max_{\tilde{\pi}\in\Pi_{\Delta}}\langle\nabla_{\pi}V^{\pi}_{\lambda,\pi},\tilde{\pi}-\pi\rangle.$$

## 5. Conclusion

We have studied an entropy-regularized performative reinforcement learning problem, obtained its important properties including gradient dominance, policy lower bound, Lipschitz continuity and smoothness. Based on these properties, we have proposed a zeroth-order performative policy gradient (0-PPG) algorithm only using sample-based policy evaluation, which for the first time converges to a performatively optimal (PO) policy with polynomial number of policy evaluations under the regularizer dominance condition. These theoretical results also holds for the quadratice regularizers used in the existing works on performative reinforcement learning (see Appendix K for discussion). A future direction is to extend the algorithm and results to more practical environments of large state and action spaces.

Algorithm 1 Zeroth-Order Performative Policy Gradient (0-PPG) Algorithm 1: **Inputs:** T, N, ∆ > δ > 0, ϵV ≥ 0, β > 0. 2: **Initialize:** policy π0 ∈ Π∆.

3: for Iterations t = 0, 1*, . . . , T* − 1 do 4: Obtain i.i.d. vectors {vi}
N
i=1 uniformly from the unit sphere U1 def 
= {u∈R
|S||A| : ∥u∥= 1}.

5: Obtain {projL0
(vi)}
N
i=1 from Eq. (28).

6: Obtain {ui}
N
i=1 by Eq. (29).

7: Obtain stochastic policy evaluation Vˆ π λ,π ≈ V
π λ,π for π ∈ {πt ± δui}
N
i=1 with error bound (24).

8: Obtain stochastic performative policy gradient estimation gˆλ,δ(πt) using Eq. (26).

9: Obtain π˜t by Eq. (33).

10: Update πt+1 by Eq. (32). 11: **end for**
12: **Output:** πTe where Te ∈ arg min0≤t≤T −1

$${}_{1}\langle\hat{g}_{\lambda,\delta}(\pi_{t}),\bar{\pi}_{t}-\pi_{t}\rangle.$$

## Impact Statement References

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Chen, Z., Ma, S., and Zhou, Y. (2022). Sample efficient stochastic policy extragradient algorithm for zero-sum markov game. In Proceedings of the International Conference on Learning Representations (ICLR).

Flaxman, A. D., Kalai, A. T., and McMahan, H. B. (2005).

Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms, pages 385–394.

Havrilla, A., Du, Y., Raparthy, S. C., Nalmpantis, C.,
Dwivedi-Yu, J., Hambro, E., Sukhbaatar, S., and Raileanu, R. (2024). Teaching large language models to reason with reinforcement learning. In *AI for Math Workshop@ ICML* 2024.

Agarwal, A., Dekel, O., and Xiao, L. (2010). Optimal algorithms for online convex optimization with multipoint bandit feedback. In *Colt*, pages 28–40. Citeseer.

Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G.

(2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. The Journal of Machine Learning Research, 22(1):4431–4506.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-
Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. ArXiv:2204.05862.

Bhandari, J. and Russo, D. (2024). Global optimality guarantees for policy gradient methods. *Operations Research*.

Bhandari, J., Russo, D., and Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. In Proceedings of the Conference on learning theory (COLT), pages 1691–1692.

Brown, N. and Sandholm, T. (2019). Superhuman ai for multiplayer poker. *Science*, 365(6456):885–890.

Cen, S., Cheng, C., Chen, Y., Wei, Y., and Chi, Y. (2022).

Fast global convergence of natural policy gradient methods with entropy regularization. *Operations Research*, 70(4):2563–2578.

Chaney, A. J., Stewart, B. M., and Engelhardt, B. E. (2018).

How algorithmic confounding in recommendation systems increases homogeneity and decreases utility. In Proceedings of the 12th ACM conference on recommender systems, pages 224–232.

Chen, Y. (2020). Mirror descent. https://yuxinche n2020.github.io/ele522_optimization/
lectures/mirror_descent.pdf.

Chen, Z. and Huang, H. (2024). Accelerated policy gradient for s-rectangular robust mdps with large state spaces. In Proceedings of the International Conference on Machine Learning (ICML).

Mansoury, M., Abdollahpouri, H., Pechenizkiy, M.,
Mobasher, B., and Burke, R. (2020). Feedback loop and bias amplification in recommender systems. In Proceedings of the 29th ACM international conference on information & knowledge management, pages 2145–2148.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T.,
Harley, T., Silver, D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), volume 48, pages 1928–1937.

Leonardos, S., Overman, W., Panageas, I., and Piliouras, G.

(2022). Global convergence of multi-agent policy gradient in markov potential games. In ICLR 2022 Workshop on Gamification and Multiagent Solutions.

Li, G., Wu, W., Chi, Y., Ma, C., Rinaldo, A., and Wei, Y. (2023). Sharp high-probability sample complexities for policy evaluation with linear function approximation. ArXiv:2305.19001.

Malik, D., Pananjady, A., Bhatia, K., Khamaru, K., Bartlett, P. L., and Wainwright, M. J. (2020). Derivative-free methods for policy optimization: Guarantees for linear quadratic systems. Journal of Machine Learning Research, 21(21):1–51.

Mandal, D. and Radanovic, G. (2024). Performative reinforcement learning with linear markov decision process. ArXiv:2411.05234.

Mandal, D., Triantafyllou, S., and Radanovic, G. (2023).

Performative reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), pages 23642–23680.

Mankowitz, D. J., Levine, N., Jeong, R., Abdolmaleki, A.,
Springenberg, J. T., Shi, Y., Kay, J., Hester, T., Mann, T., and Riedmiller, M. (2019). Robust reinforcement learning for continuous control with model misspecification. In Proceedings of the International Conference on Learning Representations (ICLR).

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Nikolaidis, S., Nath, S., Procaccia, A. D., and Srinivasa, S.

(2017). Game-theoretic modeling of human adaptation in human-robot collaboration. In Proceedings of the 2017 ACM/IEEE international conference on human-robot interaction, pages 323–331.

Perdomo, J., Zrnic, T., Mendler-Dünner, C., and Hardt, M. (2020). Performative prediction. In International Conference on Machine Learning, pages 7599–7609.

Rank, B., Triantafyllou, S., Mandal, D., and Radanovic, G.

(2024). Performative reinforcement learning in gradually shifting environments. In The 40th Conference on Uncertainty in Artificial Intelligence (UAI).

Samsonov, S., Tiapkin, D., Naumov, A., and Moulines, E.

(2023). Finite-sample analysis of the temporal difference learning. *ArXiv:2310.14286*.

Shamir, O. (2017). An optimal algorithm for bandit and zero-order convex optimization with two-point feedback.

Journal of Machine Learning Research, 18(52):1–11.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. *nature*, 550(7676):354–359.

Tropp, J. A. et al. (2015). An introduction to matrix concentration inequalities. Foundations and Trends® *in Machine* Learning, 8(1-2):1–230.

Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M.,
Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P., et al. (2019). Grandmaster level in starcraft ii using multi-agent reinforcement learning. *nature*, 575(7782):350–354.

Wang, Q., Ho, C. P., and Petrik, M. (2023). Policy gradient in robust mdps with global convergence guarantee. In Proceedings of the International Conference on Machine Learning (ICML), volume 202, pages 35763–35797.

Wei, C.-Y., Lee, C.-W., Zhang, M., and Luo, H. (2021).

Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games. In Proceedings of the Conference on Learning Theory (COLT).

Zhang, J., Bedi, A. S., Wang, M., and Koppel, A. (2021).

Beyond cumulative returns via reinforcement learning over state-action occupancy measures. In *2021 American* Control Conference (ACC), pages 894–901. IEEE.

# Appendix

## Table Of Contents

A. Supporting Lemmas A.1. Frank-Wolfe Step We repeat Lemma 1 as follows.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Lemma 2. *The step* (31) *has the following analytical solution.*

$$\tilde{\pi}_{t}(a|s)=\begin{cases}\Delta;a\neq\tilde{a}_{t}(s)\\ 1-\Delta(|{\mathcal A}|-1);a=\tilde{a}_{t}(s)\end{cases},$$
, (35)
where a˜t(s) ∈ arg maxagˆλ,δ(πt)(a|s).

Proof. For π˜t defined by Eq. (35) and for any π ∈ Π∆, we have
⟨π˜t − π, gˆλ,δ(πt)⟩
=X
s,a gˆλ,δ(πt)(a|s)[˜πt(a|s) − π(a|s)]

| Table of Contents A Supporting Lemmas                         | 11                                                                              |    |    |
|---------------------------------------------------------------|---------------------------------------------------------------------------------|----|----|
| A.1                                                           | Frank-Wolfe Step                                                                |    | 11 |
| A.2                                                           | Lipschitz Property of Occupany Measure                                          | 12 |    |
| A.3                                                           | Various Value Functions                                                         | 13 |    |
| A.4                                                           | Zeroth-order Gradient Estimation Error                                          | 17 |    |
| A.5                                                           | Orthogonal Transformation                                                       |    | 19 |
| A.6                                                           | Basic Inequalities                                                              |    | 20 |
| B                                                             | Negative Entropy Regularizer as a Strongly Convex Function of Occupancy Measure | 20 |    |
| C                                                             | Existing Assumptions That Implies Assumption 3                                  | 21 |    |
| D                                                             | Proof of Theorem 1                                                              | 22 |    |
| E                                                             | Proof of Corollary 1                                                            | 27 |    |
| F                                                             | Proof of Theorem 2                                                              | 28 |    |
| G                                                             | Proof of Theorem 3                                                              | 29 |    |
| H                                                             | Proof of Proposition 1                                                          | 31 |    |
| I                                                             | Proof of Proposition 2                                                          | 34 |    |
| J                                                             | Proof of Theorem 4                                                              | 34 |    |
| K Adjusting Our Results to the Existing Quadratic Regularizer | 37                                                                              |    |    |

where (a) uses π(a|s) − ∆ ≥ 0 and gˆλ,δ(πt)(a|s) ≤ gˆλ,δ(πt)[˜at(s)|s]. Therefore, Eq. (31) holds, that is, π˜t =
arg maxπ∈Π∆
⟨π, gˆλ,δ(πt)⟩.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641
642
643 644 645 646 647 648
$$\mathrm{(d)}$$

## 650 651 652 653 654 655 656 657 658 659 A.2. Lipschitz Property Of Occupany Measure

Lemma 3. The occupancy measure dπ,p *defined by Eq.* (2) has the following Lipschitz properties for any *π, π*′ ∈ Π,
p, p′ ∈ P and s˜ ∈ S.

where (a) uses Eqs. (36) and (37).

s |dπ′,p(s) − dπ,p(s)| ≤ γ 1 − γ max s ∥π ′(·|s) − π(·|s)∥1 ≤ γp|A| 1 − γ ∥π ′ − π∥ (36) X s |dπ,p′ (s) − dπ,p(s)| ≤  γ 1 − γ max s,a ∥p ′(·|s, a) − p(·|s, a)∥1 ≤ γp|S| 1 − γ ∥p ′ − p∥ (37) X s,a |dπ′,p′ (s, a) − dπ,p(s, a)| ≤ 1 1 − γ max s∥π ′(·|s) − π(·|s)∥1 +γ 1 − γ max s,a ∥p ′(·|s, a) − p(·|s, a)∥1 X ≤ p|A| 1 − γ ∥π ′ − π∥ + γp|S| 1 − γ ∥p ′ − p∥ (38)
$$(36)$$
$$(37)$$
$$(38)$$
Proof. The first ≤ of Eqs. (36) and (37) follows from Lemma 5 of (Chen and Huang, 2024). The second ≤ of Eqs. (36) and
(37) uses ∥x∥1 ≤
√d∥x∥ for any x ∈ R
d.

X s,a |dπ′,p′ (s, a) − dπ,p(s, a)| =X s,a |dπ′,p′ (s)π ′(a|s) − dπ,p(s)π(a|s)| ≤X s,a dπ′,p′ (s)|π ′(a|s) − π(a|s)| + π(a|s)|dπ′,p′ (s) − dπ,p(s)| ≤X s [dπ′,p′ (s) max s ′∥π ′(·|s ′) − π(·|s ′)∥1] +X s |dπ′,p′ (s) − dπ,p(s)| (a) ≤ max s ′∥π ′(·|s ′) − π(·|s ′)∥1 +γ 1 − γ max s ∥π ′(·|s) − π(·|s)∥1 +γ 1 − γ max s,a ∥p ′(·|s, a) − p(·|s, a)∥1 ≤1 1 − γ max s ∥π ′(·|s) − π(·|s)∥1 +γ 1 − γ max s,a ∥p ′(·|s, a) − p(·|s, a)∥1 ≤ p|A| 1 − γ ∥π ′ − π∥ + γp|S| 1 − γ ∥p ′ − p∥,
Eq. (38) can be proved as follows.

= X s ngˆλ,δ(πt)[˜at(s)|s]-1 − ∆(|A| − 1) − π[˜at(s)|s]−X a̸=˜at(s) gˆλ,δ(πt)(a|s)[π(a|s) − ∆]o (a) ≥X s ngˆλ,δ(πt)[˜at(s)|s]-1 − ∆(|A| − 1) − π[˜at(s)|s]−X a̸=˜at(s) gˆλ,δ(πt)[˜at(s)|s][π(a|s) − ∆]o = X s ngˆλ,δ(πt)[˜at(s)|s]-1 − ∆(|A| − 1) − π[˜at(s)|s]− gˆλ,δ(πt)[˜at(s)|s]-1 − π[˜at(s)|s] − ∆(|A| − 1)o =0,
A.3. Various Value Functions Define the following value functions. Note that the value function (8) of interest can be rewritten into the above functions as follows.

$$V_{\lambda,\pi^{\prime}}^{\pi}=J_{\lambda}(\pi,\pi,p_{\pi^{\prime}},r_{\pi^{\prime}})=\sum_{s}\rho(s)V_{\lambda}(\pi,\pi,p_{\pi^{\prime}},r_{\pi^{\prime}};s)=\sum_{s,a}\rho(s)\pi(a|s)Q_{\lambda}(\pi,\pi,p_{\pi^{\prime}},r_{\pi^{\prime}};s,a).$$

Hence, we will investigate the properties of the value functions (39)-(41) as follows. Lemma 4. For any π ∈ Π, p ∈ P, r ∈ R*, we have* V
π λ,π, Jλ(π, π, p, r), Vλ(π, π, p, r; s), Qλ(π, π, p, r; *s, a*) ∈
h0, 1+λ log |A| 1−γ i.

Proof. We will prove the range of Jλ(*π, π, p, r*) as follows using r(*s, a*) ∈ [0, 1]. The proof for the other value functions follow the same way.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 Lemma 5. The gradients of Jλ(π, π′, p, r) *defined by Eq.* (39) *have the following expressions.*

∂Jλ(π, π′, p, r) ∂π(a|s)= dπ,p(s)Qλ(π, π′, p, r; s, a) 1 − γ, (43) ∂Jλ(π, π′, p, r) ∂π′(a|s)= − λdπ,p(s, a) (1 − γ)π ′(a|s) , (44) ∂Jλ(π, π′, p, r) ∂p(s ′|s, a)= dπ,p(s, a) 1 − γ -r(s, a) − λ log π ′(a|s) + γVλ(π, π′, p, r; s ′), (45) ∂Jλ(π, π′, p, r) ∂r(s, a)= dπ,p(s, a) 1 − γ, (46) ∂Jλ(π, π, p, r) ∂π(a|s)= dπ,p(s)[Qλ(π, π, p, r; s, a) − λ] 1 − γ. (47)
$$(44)$$
$$(45)$$
$$(46)$$
$$(47)$$
13

$$0\leq J_{\lambda}(\pi,\pi,p,r)=\mathbb{E}_{\pi,p,\rho}\Big{[}\sum_{t=0}^{\infty}\gamma^{t}\big{[}r(s_{t},a_{t})-\lambda\log\pi(a_{t}|s_{t})\big{]}\Big{]}$$ $$\leq\sum_{t=0}^{\infty}\gamma^{t}+\lambda\mathbb{E}_{\pi,p,\rho}\Big{[}\sum_{t=0}^{\infty}\gamma^{t}\sum_{a}[-\pi(a|s_{t})\log\pi(a|s_{t})]\Big{]}$$ $$\leq\frac{1}{1-\gamma}+\lambda\sum_{t=0}^{\infty}\gamma^{t}\log|A|$$ $$\leq\frac{1+\lambda\log|\mathcal{A}|}{1-\gamma}.$$
$$(43)$$
Jλ(π, π′, p, r) def  = Eπ,phX ∞ t=0 γ t[r(st, at) − λ log π ′(at|st)] s0 ∼ ρ i =1 1 − γ X s,a dπ,p(s, a)[r(s, a) − λ log π Vλ(π, π′, p, r; s) def  = Eπ,phX ∞ t=0 γ t[r(st, at) − λ log π ′(at|st)] s0 = s Qλ(π, π′, p, r; s, a) def  = Eπ,phX∞ t=0 γ t[r(st, at) − λ log π ′(at|st)] s0 = s, a0 = a i =r(s, a) − λ log π ′(a|s) + γ X s′ p(s ′|s, a)Vλ(π, π′, p, r; s
′(a|s)], (39)
i, (40)

$$(39)$$
$$(40)$$
$$(41)$$
$$(42)$$
′). (41)
Proof. Eq. (43) follows from the policy gradient expression in Eq. (7) of (Agarwal et al., 2021), with reward function r(*s, a*) replaced by r(s, a) − λ log π
′(a|s).

Eq. (45) can be proved as follows. where (a) uses Eq. (9) in (Chen and Huang, 2024). Eqs. (44) and (46) can be proved by taking derivatives of Eq. (39).

Based on the chain rule, Eq. (47) can be proved as follows by adding Eqs. (43) and (44) with π
′ = π.

where the final = uses dπ,p(*s, a*) = dπ,p(s)π(a|s).

Lemma 6. The function Jλ *defined by eq.* (39) has the following Lipschitz properties for any π, π′ ∈ Π, p, p′ ∈ P and r, r′ ∈ R.

$$\|\nabla_{p}J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})-\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p,r)\|\leq\ell_{\pi}\max_{s}[\|\log\pi^{\prime}(\cdot|s)-\log\pi(\cdot|s)\|+\ell_{p}\|p^{\prime}-p\|+\frac{\sqrt{|S|}}{(1-\gamma)^{2}}\|r^{\prime}-r\|_{\infty}].$$
′−r∥∞ (53)
$$\|\nabla_{r}J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})-\nabla_{r}J_{\lambda}(\pi,\pi,p,r)\|\leq\Big{[}\frac{J_{\lambda}(1+\lambda\log|\mathcal{A}|)}{(1-\gamma)^{2}}+\gamma L_{p}\Big{]}\max_{\lambda}\|\pi^{\prime}(\cdot|s)-\log\pi^{\prime}(\cdot|s)\|$$ $$+\gamma\sqrt{|\mathcal{A}\Big{[}\frac{2\sqrt{|\mathcal{S}|}(1+\lambda\log|\mathcal{A}|)}{(1-\gamma)^{2}}+L_{p}\Big{]}|\mathcal{U}^{\prime}-p|+\frac{\sqrt{|\mathcal{A}||\mathcal{U}^{\prime}-r|}\infty}{1-\gamma},\tag{55}$$
where Lπ :=
$\overline{\mathbb{I}[(2-\gamma+\gamma\lambda\log|\mathcal{A}|)}$, $L_p:=\frac{\sqrt{|\mathcal{S}|}(1+\lambda\log|\mathcal{A}|)}{(1-\gamma)^2}$, $\ell_\pi:=\frac{\sqrt{|\mathcal{S}|[\mathcal{A}]}(2+3\gamma\lambda\log|\mathcal{A}|)}{(1-\gamma)^3}$ and $\ell_p:=\frac{2\gamma|\mathcal{S}|(1+\lambda\log|\mathcal{A}|)}{(1-\gamma)^3}$. 
Proof. Eqs. (48), (49), (51) and (52) directly follow from Lemma 6 of (Chen and Huang, 2024). Eq. (50) can be proved as follows.

$$\begin{split}|J_{\lambda}(\pi,p,r^{\prime})-J_{\lambda}(\pi,p,r)|=&\left|\frac{1}{1-\gamma}\sum_{s,a}d_{\pi,p}(s,a)[r^{\prime}(s,a)-r(s,a)]\right|\\ \leq&\frac{1}{1-\gamma}\sum_{s,a}d_{\pi,p}(s,a)|r^{\prime}(s,a)-r(s,a)|\end{split}$$

14

$\eqref{eq:walpha}$
715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

$$\begin{array}{c}{{p(s^{\prime}|s,a)\stackrel{(a)}{=}\frac{d\pi_{,p}(s)\pi(a|s)}{1-\gamma}\big[r(s,a)-\lambda\log\pi(a|s)+\gamma V_{\lambda}(\pi,\pi^{\prime},p,r;s^{\prime})\big]}}\\ {{=\frac{d\pi_{,p}(s,a)}{1-\gamma}\big[r(s,a)-\lambda\log\pi(a|s)+\gamma V_{\lambda}(\pi,\pi^{\prime},p,r;s^{\prime})\big],}}\end{array}$$
$$\frac{\partial J_{\lambda}(\pi,\pi,p,r)}{\partial\pi(a|s)}=\Big{[}\frac{\partial J_{\lambda}(\pi,\pi^{\prime},p,r)}{\partial\pi(a|s)}+\frac{\partial J_{\lambda}(\pi,\pi^{\prime},p,r)}{\partial\pi^{\prime}(a|s)}\Big{]}\Big{|}_{\pi^{\prime}=\pi}$$ $$=\frac{d_{\pi,p}(s)Q_{\lambda}(\pi,\pi,p,r;s,a)}{1-\gamma}-\frac{\lambda d_{\pi,p}(s,a)}{(1-\gamma)\pi(a|s)}$$ $$=\frac{d_{\pi,p}(s)[Q_{\lambda}(\pi,\pi,p,r;s,a)-\lambda]}{1-\gamma},$$
$$(48)$$
$$(49)$$
$$(50)$$
|Jλ(π ′, π′, p, r) − Jλ(π, π, p, r)| ≤ Lπ max s∥ log π ′(·|s) − log π(·|s)∥ (48) |Jλ(π, π, p′, r) − Jλ(π, π, p, r)| ≤ Lp∥p ′ − p∥ (49) |Jλ(π, π, p, r′) − Jλ(π, π, p, r)| ≤  ∥r ′ − r∥∞ 1 − γ≤ ∥r ′ − r∥ 1 − γ(50) ∥∇pJλ(π ′, π′, p, r) − ∇pJλ(π, π, p, r)∥ ≤ ℓπ max s∥ log π ′(·|s) − log π(·|s)∥ (51) ∥∇pJλ(π, π, p′, r) − ∇pJλ(π, π, p, r)∥ ≤ ℓp∥p ′ − p∥ (52)
To prove Eq. (53), note that

$$\leq\frac{d_{\pi,p}(s,a)}{(1-\gamma)^{2}}\|r^{\prime}-r\|_{\infty}$$
′ − r∥∞ (56)
$$\left|\frac{\partial J_{\lambda}(\pi,\pi,p,r^{\prime})}{\partial p(s^{\prime}|s,a)}-\frac{\partial J_{\lambda}(\pi,\pi,p,r)}{\partial p(s^{\prime}|s,a)}\right|$$ $$\frac{(a)}{1-\gamma}[r^{\prime}(s,a)-r(s,a)+\gamma[V_{\lambda}(\pi,\pi^{\prime},p,r^{\prime};s^{\prime})-V_{\lambda}(\pi,\pi^{\prime},p,r;s^{\prime})]]$$ $$\frac{(b)}{\leq}\frac{d_{\pi,p}(s,a)}{1-\gamma}\Big{[}\|r^{\prime}-r\|_{\infty}+\gamma\sum_{t=0}^{\infty}\gamma^{t}\|r^{\prime}-r\|_{\infty}\Big{]}$$ $$d_{\pi,p}(s,a)$$
$$(56)$$

where (a) uses Eq. (45) and (b) uses Eq. (40). Therefore, we can prove Eq. (53) as follows.

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

$$\begin{array}{l}{{\|\nabla_{p}J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})-\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p,r)\|}}\\ {{\leq\|\nabla_{p^{\prime}}J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})-\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p^{\prime},r^{\prime})\|+\|\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p^{\prime},r^{\prime})-\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p,r^{\prime})\|}}\\ {{+\|\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p,r^{\prime})-\nabla_{p^{\prime}}J_{\lambda}(\pi,\pi,p,r)\|}}\end{array}$$
$\stackrel{{(a)}}{{\leq}}\ell_{\pi}\max_{s}\|\log\pi^{\prime}(\cdot|s)-\log\pi(\cdot|s)\|+\ell_{p}\|p^{\prime}-p\|+\sqrt{\sum_{s,a,s^{\prime}}\left|\frac{\partial J_{\lambda}(\pi,\pi,p,r^{\prime})}{\partial p(s^{\prime}|s,a)}-\frac{\partial J_{\lambda}(\pi,\pi,p,r)}{\partial p(s^{\prime}|s,a)}\right|^{2}}$.  
$${\overset{(b)}{\leq}}\ell_{\pi}\operatorname*{max}_{s}\|\log\pi^{\prime}(\cdot|s)-\log\pi(\cdot|s)\|+\ell_{p}\|p^{\prime}-p\|+{\sqrt{\frac{\|r^{\prime}-r\|_{\infty}^{2}}{(1-\gamma)^{4}}}}\sum_{s,a,s^{\prime}}d_{\pi,p}^{2}(s,a)$$
$$\leq\!\ell_{\pi}\operatorname*{max}_{s}\|\log\pi^{\prime}(\cdot|s)-\log\pi(\cdot|s)\|+\ell_{p}\|p^{\prime}-p\|+{\frac{\sqrt{|S|}}{(1-\gamma)^{2}}}\|r^{\prime}-r\|_{\infty},$$

where (a) uses Eqs. (51)-(52) and (b) uses Eq. (56). Then, we prove Eq. (54) as follows. where (a) uses Eq. (46), (b) uses Eq. (38). To prove Eq. (55), we will first prove the following auxiliary bounds.

$$Q_{\lambda}(\pi,\pi,p,r;s,a)-\lambda\stackrel{{(a)}}{{\in}}\Big{[}-\lambda,\frac{1+\lambda\log|\mathcal{A}|}{1-\gamma}-\lambda\Big{]}\Rightarrow|Q_{\lambda}(\pi,\pi,p,r;s,a)-\lambda|\leq\frac{1+\lambda\log|\mathcal{A}|}{1-\gamma},\tag{57}$$

where (a) uses Lemma 4.

$$|V_{3}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime};s)-V_{3}(\pi,\pi,p,r;s)|$$ $$\leq|V_{3}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime};s)-V_{3}(\pi,\pi,p^{\prime},r^{\prime};s)|+|V_{3}(\pi,\pi,p^{\prime},r^{\prime};s)-V_{3}(\pi,\pi,p,r^{\prime};s)|+|V_{3}(\pi,\pi,p,p^{\prime};s)-V_{3}(\pi,\pi,p,r;s)|$$
$$\begin{array}{c}{{\|\nabla_{r}J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})-\nabla_{r}J_{\lambda}(\pi,\pi,p,r)\|}}\\ {{\stackrel{(a)}{=}\frac{\|d_{\pi^{\prime},p^{\prime}}-d_{\pi,p}\|}{1-\gamma}}}\\ {{\stackrel{<}{=}\frac{\|d_{\pi^{\prime},p^{\prime}}-d_{\pi,p}\|_{1}}{1-\gamma}}}\\ {{(b)=1.}}\end{array}$$
$$\stackrel{(b)}{\leq}\frac{1}{(1-\gamma)^{2}}\operatorname*{max}_{s}\|\pi^{\prime}(\cdot|s)-\pi(\cdot|s)\|_{1}+\frac{\gamma}{(1-\gamma)^{2}}\operatorname*{max}_{s,a}\|p^{\prime}(\cdot|s,a)-p(\cdot|s,a)\|_{1},$$

15

$$\begin{array}{l}{{=\frac{1}{1-\gamma}\sum_{s,a}d_{\pi,p}(s,a)\|r^{\prime}-r\|_{\infty}}}\\ {{=\frac{1}{1-\gamma}\|r^{\prime}-r\|_{\infty}\leq\frac{1}{1-\gamma}\|r^{\prime}-r\|.}}\end{array}$$

Achieve Performatively Optimal Policy for Performative Reinforcement Learning

$$\stackrel{(a)}{\leq}L_{\pi}\operatorname*{max}_{s}\|\log\pi^{\prime}(\cdot|s)-\log\pi(\cdot|s)\|+L_{p}\|p^{\prime}-p\|+\frac{\|r^{\prime}-r\|_{\infty}}{1-\gamma},$$
1 − γ, (58)
where (a) applies Eqs. (48)-(50) to the case where the initial state distribution ρ is probability 1 at s (so Jλ(*π, π, p, r*) becomes Vλ(*π, π, p, r*; s)).

where (a) uses Eq. (41).

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 Note that where (a) uses Eq. (41), and (b) uses Eq. (58) and Lemma 4.

$$\stackrel{{(b)}}{{\leq}}\lambda|\log\pi^{\prime}(a|s)-\log\pi(a|s)|+\gamma L_{\pi}\max_{s^{\prime}}\|\log\pi^{\prime}(\cdot|s^{\prime})-\log\pi(\cdot|s^{\prime})\|+\gamma L_{p}\|p^{\prime}-p\|$$ $$+\frac{\gamma(1+\lambda\log|A|)}{1-\gamma}\|p^{\prime}(\cdot|s,a)-p(\cdot|s,a)\|_{1},$$
$$(1-\gamma)\Big|\frac{\partial J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})}{\partial\pi^{\prime}(a|s)}-\frac{\partial J_{\lambda}(\pi,\pi,p,r)}{\partial\pi(a|s)}\Big|$$
(a)
=dπ′,p′ (s)[Qλ(π
′, π′, p′, r′; s, a) − λ] − dπ,p(s)[Qλ(π, π, p, r; *s, a*) − λ]
≤[dπ′,p′ (s) − dπ,p(s)][Qλ(π
′, π′, p′, r′; s, a) − λ] + dπ,p(s)[Qλ(π
′, π′, p′, r′; *s, a*) − Qλ(π
′, π′, p′, r; s, a)]
+ dπ,p(s)[Qλ(π
′, π′, p′, r; s, a) − Qλ(π, π, p, r; *s, a*)]
≤dπ′,p′ (s) − dπ,p(s)·Qλ(π
′, π′, p′, r′; s, a) − λ + dπ,p(s)Qλ(π
′, π′, p′, r′; *s, a*) − Qλ(π
′, π′, p′, r; *s, a*)
+ dπ,p(s)Qλ(π
′, π′, p′, r; s, a) − Qλ(π, π, p, r; *s, a*)
(b)
≤
1 + λ log |A|
1 − γ
dπ′,p′ (s) − dπ,p(s) +
dπ,p(s)∥r
′ − r∥∞
1 − γ+ dπ,p(s)
hλ| log π
′(a|s) − log π(a|s)|
+ γLπ max
s
′∥ log π
′(·|s
′) − log π(·|s
′)∥ + γLp∥p
′ − p∥ +
γ(1 + λ log |A|)
1 − γ∥p
′(·|s, a) − p(·|*s, a*)∥1
i,
where (a) uses Eq. (47), (b) uses Eqs. (57), (59) and (60). Applying triangular inequality to the bound above, we can prove
Eq. (55) as follows.
$$(1-\gamma)\|\nabla_{\pi^{\prime}}J_{\lambda}(\pi^{\prime},\pi^{\prime},p^{\prime},r^{\prime})-\nabla_{\pi}J_{\lambda}(\pi,\pi,p,r)\|$$
|Qλ(π, π, p, r′; s, a) − Qλ(π, π, p, r; s, a)| (a) = Eπ,phX ∞ t=0 γ t[r ′(st, at) − r(st, at)] s0 = s, a0 = a i ≤Eπ,phX ∞ t=0 γ t[r ′(st, at) − r(st, at)| s0 = s, a0 = a i ≤Eπ,phX ∞ t=0 γ t∥r ′ − r∥∞ s0 = s, a0 = a i ≤ ∥r ′ − r∥∞

$$(59)$$
$$(60)$$
1 − γ, (59)
|Qλ(π ′, π′, p′, r; s, a) − Qλ(π, π, p, r; s, a)| (a) ≤λ| log π ′(a|s) − log π(a|s)| + γ  X s ′ [p ′(s ′|s, a)Vλ(π ′, π′, p′, r; s) − p(s ′|s, a)Vλ(π, π, p, r; s)]  ≤λ| log π ′(a|s) − log π(a|s)| + γX s ′ p ′(s ′|s, a)|Vλ(π ′, π′, p′, r; s) − Vλ(π, π, p, r; s)| + γX s ′ |p ′(s ′|s, a) − p(s ′|s, a)||Vλ(π, π, p, r; s)|
Achieve Performatively Optimal Policy for Performative Reinforcement Learning

$$\leq\frac{1+\lambda\log|\mathcal{A}|}{1-\gamma}\sqrt{\sum_{s,a}\left|d_{s^{\prime},p^{\prime}}(s)-d_{s,p}(s)\right|^{2}}+\frac{\left|p^{\prime}-r\right|_{\infty}}{1-\gamma}\sqrt{\sum_{s,a}d_{s,p}(s)^{2}}+\lambda\sqrt{\sum_{s,a}d_{s,p}(s)^{2}|\log\pi^{\prime}(a|s)-\log\pi(a|s)|^{2}}.$$
$$+\left.\left[\gamma L_{\pi}\operatorname*{max}_{s^{\prime}}\|\log\pi^{\prime}(\cdot|s^{\prime})-\log\pi(\cdot|s^{\prime})\|+\gamma L_{p}\|p^{\prime}-p\|\right]\sqrt{\sum_{s,a}d_{\pi,p}(s)^{2}}\right]\right|$$

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

## A.4. Zeroth-Order Gradient Estimation Error

We import Theorem 1.6.2 of (Tropp et al., 2015) as follows.

Lemma 7 (Matrix Bernstein Inequality). Suppose complex-valued matrices S1*, . . . , S*N ∈ C
d1×d2 *are independently* distributed with ESk = 0 and ∥Sk∥ ≤ C for each k = 1, . . . , N*. Denote the sum* ZN =PN
k=1 Sk its variance statistic as follows where S
∗
k denotes the conjugate transpose of Sk. Then for any ϵ ≥ 0*, we have*

$$v(Z_{N})=\max\left[\left|\sum_{k=1}^{N}\mathbb{E}(S_{k}S_{k}^{*})\right|,\left|\sum_{k=1}^{N}\mathbb{E}(S_{k}^{*}S_{k})\right|\right],\tag{1}$$
$$(61)$$

$$(62)$$
$$\mathbb{P}\{\|Z_{N}\|\geq\epsilon\}\leq(d_{1}+d_{2})\exp\Big[\frac{-\epsilon^{2}/2}{v(Z_{N})+C\epsilon/3}\Big].$$
i. (62)
Applying the above lemma to vectors, we obtain the following vector Bernstein inequality.

Lemma 8 (Vector Bernstein Inequality). Suppose independently distributed vectors x1*, . . . , x*N ∈ C
dsatisfies ∥xk∥ ≤ c for each k = 1, . . . , N. Then for any η ∈ (0, 1), with probability at least 1 − η*, we have*

$$\left\|{\frac{1}{N}}\sum_{k=1}^{N}(x_{k}-\mathbb{E}x_{k})\right\|<{\frac{4c}{3N}}\log\left({\frac{d+1}{\eta}}\right)+2c{\sqrt{\frac{2}{N}}}\log\left({\frac{d+1}{\eta}}\right).$$

where (a) uses Lemma 3, (b) uses ∥π
′(·|s) − π(·|s)∥1 ≤ ∥ log π
′(·|s) − log π(·|s)∥1,
∥p
′(·|s, a) − p(·|s, a)∥1 ≤p*|S|∥*p
′(·|s, a) − p(·|s, a)∥ ≤ p*|S|∥*p
′ − p∥,
γ
√|S|(1+λ log |A|)
1−γ ≤
√*|S||A|*(1+λ log |A|)
(1−γ)
2 and λ ≤
λ|A| log |A|
(1−γ)
2 .

$$+\frac{1\cdot\gamma\cdot\gamma\cdot\gamma\cdot\gamma}{1-\gamma}\|p^{\prime}-p\|$$ $$\frac{\omega}{\gamma}\Big{[}\frac{|\mathcal{A}|(\gamma+2\lambda\log|\mathcal{A}|)}{(1-\gamma)^{2}}+\gamma L_{\pi}\Big{]}\max_{s^{\prime}}\|\log\pi^{\prime}(\cdot|s^{\prime})-\log\pi(\cdot|s^{\prime})\|+\gamma\sqrt{|\mathcal{A}|}\Big{[}\frac{2\sqrt{|\mathcal{S}|}(1+\lambda\log|\mathcal{A}|)}{(1-\gamma)^{2}}+L_{\tau}\Big{]}|p^{\prime}-p\|$$ $$+\frac{\sqrt{|\mathcal{A}||}|p^{\prime}-\tau|_{\infty}}{1-\gamma},$$
$$+\,{\frac{\gamma{\sqrt{|{\mathcal{S}}|}}(1+\lambda\log|{\mathcal{A}}|)}{1-\gamma}}\|p^{\prime}-p\|$$
(a) γp|A|(1 + λ log |A|) p|A|∥r ′ − r∥∞ + γ(1 + λ log |A|) 1 − γ sX s,a dπ,p(s) 2∥p ′(·|s, a) − p(·|s, a)∥ 21 ≤ p|A|(1 + λ log |A|) 1 − γ X s |dπ′,p′ (s) − dπ,p(s)| + p|A|∥r ′ − r∥∞ 1 − γ+ λ sX s dπ,p(s)∥ log π ′(·|s) − log π(·|s)∥ 2 +-γLπ max s ′∥ log π ′(·|s ′)−log π(·|s ′)∥+γLp∥p ′−p∥p|A| + γ(1+λ log |A|) 1 − γ s|S|X s,a ∥p ′(·|s, a) − p(·|s, a)∥ 2
$$\frac{(\circ)}{\leq}\frac{\gamma\sqrt{|\mathcal{A}|}(1+\lambda\log|\mathcal{A}|)}{(1-\gamma)^{2}}\big{[}\max_{s}\|\pi^{\prime}(\cdot|s)-\pi(\cdot|s)\|_{1}+\max_{s,a}\|p^{\prime}(\cdot|s,a)-p(\cdot|s,a)\|_{1}\big{]}+\frac{\sqrt{|\mathcal{A}|}\|r^{\prime}-r\|^{\prime}}{1-\gamma}$$ $$\quad+\lambda\max_{s^{\prime}}\|\log\pi^{\prime}(\cdot|s^{\prime})-\log\pi(\cdot|s^{\prime})\|+\big{[}\gamma L_{\pi}\max_{s^{\prime}}\|\log\pi^{\prime}(\cdot|s^{\prime})-\log\pi(\cdot|s^{\prime})\|+\gamma L_{p}\|p^{\prime}-p\|\big{]}\sqrt{|\mathcal{A}|}$$
$$(63)$$

17 Proof. Note that Sk = xk − Exk satisfies the conditions of Lemma 7 with d1 = d, d2 = 1 and C replaced by 2c. In addition, v(ZN ) defined by Eq. (61) satisfies v(ZN ) ≤ 4N c2since

$$\operatorname*{max}[\|S_{k}S_{k}^{*}\|,\|S_{k}^{*}S_{k}\|^{2}]\leq\|S_{k}^{*}\|^{2}\|S_{k}\|^{2}\leq4c^{2}.$$

For any η ∈ (0, 1), let Therefore, Lemma 7 implies that

$$\mathbb{P}\Big\{{\frac{1}{N}}\Big\|}\sum_{k=1}^{N}(x_{k}-\mathbb{E}x_{k})\Big\|\geq{\frac{\epsilon}{N}}\Big\}\leq(d+1)\exp\Big[{\frac{-\epsilon^{2}/2}{4N c^{2}+2c c/3}}\Big]\leq\eta,$$

which implies that with probability at least 1 − η, we have

$${\frac{1}{N}}\Big\|\sum_{k=1}^{N}(x_{k}-\mathbb{E}x_{k})\Big\|<{\frac{\epsilon}{N}}={\frac{4c}{3N}}\log\Big({\frac{d+1}{\eta}}\Big)+2c{\sqrt{\frac{2}{N}\log\Big({\frac{d+1}{\eta}}\Big)}}.$$

For any function f : R
d → R, obtain the following zeroth-order stochastic estimator of the gradient ∇f.

$$g_{\delta}(x)=\frac{d}{2N\delta}\sum_{i=1}^{N}[f(x+\delta u_{i})-f(x-\delta u_{i})]u_{i}\approx\nabla f(x)$$
$\square$
$$(64)$$

$$(65)$$
[f(x + δui) − f(x − δui)]ui ≈ ∇f(x) (64)
where δ > 0 and {ui}
N
i=1 are i.i.d. samples of the uniform distribution on the sphere Sd = {u ∈ R
d: ∥u∥ = 1}.

Lemma 9. *Suppose* f : R
d → R is an Lf -Lipschitz continuous and ℓf -smooth function. Then for any η ∈ (0, 1), with probability at least 1 − η, the gradient estimator gδ *defined by Eq.* (64) *has the following error bound.* Proof. Note that gδ,i(x)
def 
=d 2δ
[f(x + δui) − f(x − δui)]ui has the following norm bound 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 Define the following smoothed approximation of f as follows.

$$(66)$$
$$f_{\delta}(x)\ {\stackrel{\mathrm{def}}{=}}\ \mathbb{E}_{v\sim\mathrm{Unif}(\mathbb{B}_{d})}[f(x+\delta v)],$$

def = Ev∼Unif(Bd)[f(x + δv)], (67)
where Unif(Bd) denotes the uniform distribution on the ball Bd def = {u ∈ R
d: ∥u∥ ≤ 1}. Then based on Lemma 1 of
(Flaxman et al., 2005), we have

$$\mathbb{E}[g_{\delta,i}(x)]=\nabla f_{\delta}(x)=\mathbb{E}_{v\sim\operatorname{Unif}(\mathbb{B}_{d})}[\nabla f(x+\delta v)].$$
E[gδ,i(x)] = ∇fδ(x) = Ev∼Unif(Bd)[∇f(x + δv)]. (68)

Therefore, applying Lemma 8 to gδ,i(x), the following bound holds with probability at least 1 − η.

$$\frac{1}{N}\Big\|\sum_{i=1}^{N}[g_{\delta,i}(x)-\nabla f_{\delta}(x)]\Big\|<\frac{4L_{f}d}{3N}\log\Big(\frac{d+1}{\eta}\Big)+2L_{f}d\sqrt{\frac{2}{N}\log\Big(\frac{d+1}{\eta}\Big)}.$$
. (69)
$$\|g_{\delta}(x)-\nabla f(x)\|\leq\frac{4L_{f}d}{3N}\log\left(\frac{d+1}{\eta}\right)+2L_{f}d\sqrt{\frac{2}{N}\log\left(\frac{d+1}{\eta}\right)}+\delta\ell_{f}.$$
$$\|g_{\delta,i}(x)\|\leq{\frac{d}{2\delta}}|f(x+\delta u_{i})-f(x-\delta u_{i})|\cdot\|u_{i}\|\leq{\frac{d}{2\delta}}\cdot L_{f}\|2\delta u_{i}\|=L_{f}d.$$
$$\epsilon=\frac{4c}{3}\log\left(\frac{d+1}{\eta}\right)+c\sqrt{2N\log\left(\frac{d+1}{\eta}\right)}.$$
$$(67)^{\frac{1}{2}}$$
$$(68)$$
$$(69)$$

18 Note that

$\|\nabla f_{\delta}(x)-\nabla f(x)\|=\|\mathbb{E}_{v\sim\text{Unif}(\mathbb{B}_{d})}[\nabla f(x+\delta v)-\nabla f(x)]\|\leq\delta\ell_{f}$.  
As a result, we can prove the conclusion as follows by using Eqs. (69) and (70) above.

## A.5. Orthogonal Transformation

Lemma 10. There exists an orthogonal transformation T *from the space* R
d−1to Zd = {z = [z1*, . . . , z*d] ∈ R
d:Pizi =
0}, that is, T is invertible and satisfies the following properties for any x, y ∈ Zd and *α, β* ∈ R. Proof. It can be verified that R
dadmits the following orthonormal basis with ⟨ei, ej ⟩ = 0 for any i ̸= j and ∥ei∥ = 1.

$$e_{d}={\frac{1}{\sqrt{d}}}[\underbrace{1,1,\ldots,1}_{d\ 1^{\prime}s}]\in\mathbb{R}^{d}.$$

Define the transformation T at x = [x1, x2*, . . . , x*d−1] ∈ R
d−1as follows.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Since Zd is a linear subspace of R
d orthogonal to ed, Zd admits the orthonormal basis {ei}
d−1 i=1 . Hence, T (x) ∈ Zd.

Conversely, for any y ∈ Zd, there exists unique x ∈ R
d−1such that y =Pd−1 i=1 xiei. Hence, T : R
d−1 → Zdis invertible.

For any x = [x1, . . . , xd−1], y = [y1*, . . . , y*d−1] ∈ R
d−1and *α, β* ∈ R, we can prove Eqs. (71) and (72) respectively as follows.

$$\langle{\mathcal{T}}(x),{\mathcal{T}}(y)\rangle=\!\!\left\langle\sum_{i=1}^{d-1}x_{i}e_{i},\sum_{j=1}^{d-1}y_{j}e_{j}\right\rangle$$

19

$$\begin{array}{r}{\mathcal{T}(\alpha x+\beta y)=\sum_{i=1}^{d-1}(\alpha x_{i}+\beta y_{i})e_{i}}\\ {=\alpha\sum_{i=1}^{d-1}x_{i}e_{i}+\beta\sum_{i=1}^{d-1}y_{i}e_{i}}\\ {=\alpha\mathcal{T}(x)+\beta\mathcal{T}(y).}\end{array}$$
$${\mathcal{T}}(x)=\sum_{i=1}^{d-1}x_{i}e_{i}.$$
$$\begin{array}{l}{(71)}\\ {(72)}\end{array}$$

$$(73)$$
xiei. (73)
$$\leq\left\|\left[\frac{1}{N}\sum_{i=1}^{N}g_{\delta,i}(x)\right]-\nabla f_{\delta}(x)\right\|+\|\nabla f_{\delta}(x)-\nabla f(x)\|$$ $$<\frac{4L_{f}d}{3N}\log\left(\frac{d+1}{\eta}\right)+2L_{f}d\sqrt{\frac{2}{N}\log\left(\frac{d+1}{\eta}\right)}+\delta\ell_{f}.$$
$$\|g_{\delta}(x)-\nabla f(x)\|=\Bigr\|\Bigl[{\frac{1}{N}}\sum_{i=1}^{N}g_{\delta,i}(x)\Bigr]-\nabla f(x)\Bigr\|$$
$\left(70\right)^3$
$$\begin{array}{l}{{\mathcal{T}(\alpha x+\beta y)=\alpha\mathcal{T}(x)+\beta\mathcal{T}(y),}}\\ {{\langle\mathcal{T}(x),\mathcal{T}(y)\rangle=\langle x,y\rangle.}}\end{array}$$
$$e_{k}=\frac{1}{\sqrt{k(k+1)}}[\underbrace{1,1,\ldots,1}_{k\ 1^{\prime}s},-k,\underbrace{0,0,\ldots,0}_{(d-k-1)\ 0^{\prime}s}]\in\mathbb{R}^{d};k=1,2,\ldots,d-1.$$

A.6. Basic Inequalities Lemma 11. For any ϵ ∈ (0, 0.5] and x ≥ 4ϵ
−1log(ϵ
−1)*, the following inequality holds.*

$$0<{\frac{\log x}{x}}\leq\epsilon$$
x≤ ϵ (74)
Specifically, any x ≥ 3 *satisfies* log x x ≤
1 2
.

Proof. As ϵ
−1 ≥ 2, we have x ≥ 4ϵ
−1log(ϵ
−1) ≥ (4)(2) log(2) > 5.54, so log x > log 5.54 > 1.71, which proves the first < of Eq. (74).

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Note that the function f(x) = log x xhas the following derivative

$$f^{\prime}(x)={\frac{1-\log x}{x^{2}}}<0,$$
$$\begin{array}{l}{\square}\end{array}$$

where < uses log x > 1.71. Hence, f is monotonic decreasing in x ≥ 4ϵ
−1log(ϵ
−1) > 5.54, Therefore, we prove the second ≤ of Eq. (74) as follows.

$$\frac{\log x}{xe}\leq\frac{\log[4e^{-1}\log(e^{-1})]}{e(|e^{-1}\log(e^{-1})|}=\frac{\log4+\log(e^{-1})+\log[\log(e^{-1})]}{4\log(e^{-1})}\leq\frac{\log4}{4\log(2)}+\frac{\log(e^{-1})+\log(e^{-1})}{4\log(e^{-1})}=1,\tag{75}$$

where (a) uses ϵ
−1 ≥ 2 and log u ≤ u for u = log(ϵ
−1).

When x ≥ 3, f
′(x) = 1−log x x2 < 0, so f(x) ≤ f(3) = log 3 3 <
1 2
.

Lemma 12. For any π, π′ ∈ Π*, we have* ∥π
′ − π∥ ≤ p2|S|.

Proof.

$$\|\pi^{\prime}-\pi\|^{2}=\sum_{s,a}|\pi^{\prime}(a|s)-\pi(a|s)|^{2}\leq\sum_{s,a}[\pi^{\prime2}(a|s)+\pi^{2}(a|s)]\leq\sum_{s,a}[\pi^{\prime}(a|s)+\pi(a|s)]=2|{\cal S}|.$$

B. Negative Entropy Regularizer as a Strongly Convex Function of Occupancy Measure The negative entropy regularizer (7) can be rewritten as follows

$${\mathcal{H}}_{\pi^{\prime}}(\pi)=\mathbb{E}_{\pi,p_{\pi^{\prime}},\rho}\bigg[\sum_{t=0}^{\infty}\gamma^{t}\log\pi(a_{t}|s_{t})\bigg]={\frac{1}{1-\gamma}}\sum_{s,a}d_{\pi,p_{\pi^{\prime}}}(s,a)\log{\frac{d_{\pi,p_{\pi^{\prime}}}(s,a)}{d_{\pi,p_{\pi^{\prime}}}(s)}},$$
, (76)
where dπ,pπ′ (s) = Pa′ dπ,pπ′ (*s, a*′). Hence, it suffices to prove that the following function of occupancy measure d is strongly convex.

$$H(d)=\sum_{s,a}d(s,a)\log{\frac{d(s,a)}{d(s)}},$$
, (77)
$$(76)$$

$$(77)^{\frac{1}{2}}$$
$$=\sum_{i=1}^{d-1}x_{i}y_{i}=\langle x,y\rangle.$$
$$(74)$$
$$=\sum_{i=1}^{d-1}\sum_{j=1}^{d-1}x_{i}y_{j}\langle e_{i},e_{j}\rangle$$
$\square$