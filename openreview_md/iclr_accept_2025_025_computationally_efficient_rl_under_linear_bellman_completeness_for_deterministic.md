# Computationally Efficient Rl Under Linear Bellman Completeness For Deterministic Dy- Namics

Runzhe Wu∗
Cornell University rw646@cornell.edu Ayush Sekhari∗
MIT sekhari@mit.edu Wen Sun Cornell University ws455@cornell.edu Akshay Krishnamurthy Microsoft Research akshaykr@microsoft.com

## Abstract

We study computationally and statistically efficient Reinforcement Learning algorithms for the *linear Bellman Complete* setting. This setting uses linear function approximation to capture value functions and unifies existing models like linear Markov Decision Processes (MDP) and Linear Quadratic Regulators (LQR). While it is known from the prior works that this setting is statistically tractable, it remained open whether a computationally efficient algorithm exists. Our work provides a computationally efficient algorithm for the linear Bellman complete setting that works for MDPs with large action spaces, random initial states, and random rewards but relies on the underlying dynamics to be deterministic. Our approach is based on randomization: we inject random noise into least squares regression problems to perform optimistic value iteration. Our key technical contribution is to carefully design the noise to only act in the null space of the training data to ensure optimism while circumventing a subtle error amplification issue.

## 1 Introduction

Various application domains of Reinforcement Learning (RL)—including game playing, robotics, self-driving cars, and foundation models—feature environments with large state and action spaces. In such settings, the learner aims to find a well performing policy by repeated interactions with the environment to acquire knowledge. Due to the high dimensionality of the problem, function approximation techniques are used to generalize the knowledge acquired across the state and action space. Under the broad category of function approximation, model-free RL stands out as a particularly popular approach due to its simple implementation and relatively better sample efficiency in practice. In model-free RL, the learner uses function approximation (e.g., an expressive function class like deep neural networks) to model the state-action value function of various policies in the underlying MDP. In fact, the combination of model-free RL with various empirical exploration heuristics has led to notable empirical advances, including breakthroughs in game playing (Silver et al., 2016; Berner et al., 2019), robot manipulation (Andrychowicz et al., 2020), and self-driving (Chen et al., 2019). Theoretical advancements have paralleled the practical successes in RL, with tremendous progress in recent years in building rigorous statistical foundations to understand what structures in the environment and the function class suffice for sample-efficient RL. These advancements are supported by optimal exploration strategies that align with the corresponding structural assumptions, and by now we have a rich set of tools and techniques for sample-efficient RL in MDPs with large state/action spaces (Russo & Van Roy, 2013; Jiang et al., 2017; Sun et al., 2019; Wang et al., 2020; Du et al., 2021; Jin et al., 2021; Foster et al., 2021; Xie et al., 2022). However, despite a rigorous statistical foundation, a significant challenge remains: many of these theoretically rigorous approaches 1 for rich function approximation are not computationally feasible, and thus have limited practical applicability. For example, some require solving complex optimization problems that are computationally intractable in practice (Zanette et al., 2020b); others require deterministic dynamics and initial states (Du et al., 2020); and some methods depend on maintaining large and complex version spaces (Jin et al., 2021; Du et al., 2021) which are intractable in terms of memory and computation. One of the most striking examples of this statistical-computational gap is observed in the *Linear* Bellman Completeness setting, which is perhaps one of the simplest learning settings. Linear Bellman completeness serves as a bridge between RL and control theory literature as it provides a unified framework to capture Linear MDPs (Jin et al., 2020; Agarwal et al., 2019; Zanette et al., 2020b) and the Linear Quadratic Regulator (LQR), two popular models in RL and control respectively. In particular, the linear Bellman completeness setting captures MDPs where the state-action value function of the optimal policy is a linear function of some pre-specified feature representations (of states and actions), and the Bellman backups of linear state-action value functions are linear (w.r.t. some feature representation). Naturally, for this setting, the learner utilizes the function class F consisting of all linear functions over the given feature representation as the value function class for model-free RL. In addition to considering a linear class, we also assume that the class F exhibits low inherent Bellman error—a structural assumption that quantifies the error in approximating the Bellman backup of functions within F. The first assumption, i.e., linearity of optimal state-action value function, is perhaps the simplest modeling assumption one can make in RL with function approximation. Furthermore, emerging evidence suggests that linearity is practically useful, as with adequate feature representation, linear functions can represent value functions in various domains. The second assumption, i.e. low inherent Bellman error of the class, while being a bit mysterious, is a natural condition for statistical tractability for classic algorithms such as Fitted Q-iteration (FQI) and temporal difference (TD) learning with linear function approximation (Munos, 2005; Zanette et al., 2020b). It is also well-known that linearity alone does not suffice for efficient RL (Wang et al., 2021; Weisz et al., 2021). While the prior works have shown that RL with linear bellman completeness is statistically tractable, and one can learn with sample complexity that scales polynomially with both d and H (where d is the dimensionality of the feature representation and H is the horizon of the RL problem), the proposed algorithms that obtain such sample complexity in the online RL setting are not computationally efficient. Given the simplicity of the problem, it was conjectured that a computationally efficient algorithm should exist. However, no such algorithms were proposed. Unfortunately, the classical approaches of combining supervised learning techniques with RL in the online setting, e.g., value function iteration, which are computationally efficient by design, fail to extend to be statistically tractable due to exponential blowups from error compounding, especially without making norm-boundedness assumptions. On the other hand, the techniques of adding quadratic exploration bonuses, e.g., the one proposed in LinUCB (Li et al., 2010) and used in LSVI for linear MDPs, also fail here as Bellman backups of quadratic functions are not necessarily within the linear class F. In fact, the search for a computationally efficient algorithm with large action spaces is open even when the transition dynamics are deterministic. In this work, we provide the first computationally efficient algorithm for the linear Bellman complete setting with deterministic dynamics, that enjoys regret bound of Õ(d 5/2H5/2 + d 2H3/2T
1/2)
for feature dimension d, horizon H, and number of rounds T. Importantly, our algorithm works with large action spaces, stochastic reward functions, and stochastic initial states. The key ideas of our algorithm are twofold: using *randomization* to encourage exploration and leveraging a span argument to bound the regret. While adding random noise to the learned parameters has been quite successful in linear function approximation, unfortunately, for our specific setting, since we need to add sufficiently large noise to cancel out the estimation error, blind randomization can cause the corresponding parameters to grow exponentially with the horizon. We avoid paying for this blow-up by only adding noise to the null space of the data. In particular, when the dynamics are deterministic, by adding exploration noise only in the null space, we can learn the value function exactly for any trajectories that lie within the span of the data seen so far. Additionally, a simple span argument bounds the number of times the trajectories fall outside the span of the historical data. Together, these techniques leads to our polynomial sample complexity bound. The resulting algorithm relies on linear regression oracles under convex constraints, which we show can be approximately solved via a random-walk-based algorithm (Bertsimas & Vempala, 2004).

## 2 Related Works

Computational Efficient RL under Linear Bellman Completeness. Numerous works have focused on computationally efficient RL within the scope of linear Bellman completeness (LBC). The simplest setting is tabular MDPs where computationally efficient and near-optimal algorithms have been well known (Azar et al., 2017; Zhang et al., 2020; Jin et al., 2018). Tabular MDPs can be extended to linear MDPs (Jin et al., 2020), where computationally efficient algorithms are also known (Jin et al., 2020; Agarwal et al., 2023; He et al., 2023). However, in the setting of linear Bellman completeness, which captures linear MDPs, the existence of computationally efficient algorithms remain unclear. Previous works have resorted to various assumptions to achieve computational efficiency, such as few actions (Golowich & Moitra, 2024) and assuming MDPs are
"explorable" (Zanette et al., 2020c). We provide a detailed overview of the literature in Section 3.2.

Exploration via Randomization. Random noise has been a powerful alternative to bonus-based exploration in RL literature. A typical approach is Randomized Least-Squares Value Iteration (RLSVI) (Osband et al., 2016), which injects Gaussian noise into the least-squares estimate and achieves near-optimal worst-case regret for linear MDPs (Agrawal et al., 2021; Zanette et al., 2020a); Ishfaq et al. (2023) instead propose posterior sampling via Langevin Monte Carlo for Q-function and also obtain regret bounds for linear MDPs; Ishfaq et al. (2021) developed randomization algorithms for general function approximation assuming bounded eluder dimension and Bellman completeness for any function. Randomization is also explored in preference-based RL, leading to the first computationally efficient algorithm with near-optimal regret guarantees for linear MDPs (Wu & Sun, 2024). However, these approaches either have strong assumptions (e.g., Bellman completeness for any function), or inject random noise larger than the estimation error, causing exponential blowup of parameter values—to mitigate it, they truncate the value, but this is feasible only in low-rank MDPs and challenging under linear Bellman completeness as the Bellman backup of truncated value may no longer be linear. Consequently, existing algorithms cannot handle linear Bellman complete problems, and new techniques capable of managing exponential parameter values are needed. Beyond Linear Bellman Completeness. Many structural conditions capture linear Bellman completeness, such as Bilinear class (Du et al., 2021), Bellman eluder dimension (Jin et al., 2021), Bellman rank (Jiang et al., 2017), witness rank (Sun et al., 2019), and decision-estimation coefficient (Foster et al., 2021). While statistically efficient algorithms exist for these settings, no computationally efficient algorithms are known.

## 3 Preliminaries

A finite-horizon Markov Decision Process (MDP) is given by a tuple M = (S,A, H,T*, r, µ*) where S is the state space, A is the action space, H ∈ N is the horizon, T ∶ S × A → ∆(S) is the transition function, r ∶ S × A → [0, 1] is the reward function and µ ∈ ∆(S) is the initial state distribution.

Given a policy π ∶ S ↦ ∆(A), we denote Qπh(s, a) = Eπ [∑
H
i=hri∣ sh = *s, a*h = a] as the layered state-action value function of policy π and V
π h(s) = Qπh(*s, π*(s)) as the state value function. The optimal value function is denoted by V
⋆
h(s) = maxπ V
π h(s), and the optimal policy is π
⋆.

We focus on the setting of linear function approximation and consider the following linear Bellman completeness, which ensures that the Bellman backup of a linear function remains linear. Definition 1 (Linear Bellman Completeness). An MDP is said to be linear Bellman complete with respect to a feature mapping ϕ *if there exists a mapping* T ∶ R
d → R
d*so that, for all* θ ∈ R
d*and all*
(s, a) ∈ S × A*, it holds that*

$$\langle T\theta,\phi(s,a)\rangle=\operatorname*{\mathbb{E}}_{s^{\prime}\sim\mathrm{T}(s,a)}\operatorname*{max}_{a^{\prime}}(\theta,\phi(s^{\prime},a^{\prime})).$$

Moreover, we require that, for all h ∈ [H] and (s, a) ∈ S × A*, the random reward is bounded in* [0, 1] with mean rh(*s, a*) = ⟨ω
⋆h
, ϕ(s, a)⟩ *for some unknown* ω
⋆h∈ R
d.

We assume ∥ϕ(*s, a*)∥2 ≤ 1 for all s ∈ S and a ∈ A. Notably, we do not impose any upper bound on ∥ω
⋆h∥2 or any ℓ2-norm non-expansiveness of the Bellman backup, distinguishing us from some existing works—in Section 3.1, we discuss why many existing definitions of linear Bellman completeness fail to capture even tabular MDPs or linear MDPs due to certain ℓ2-norm boundedness assumptions.

We further assume the feature space spans R
d, i.e., span({ϕ(s, a) ∶ s ∈ S, a ∈ A}) = R
d; otherwise, we can project the feature space onto its span or use pseudo-inverse in the analysis when needed. We can verify that the linear Bellman completeness captures both linear MDPs and Linear Quadratic Regulators (for a convex subset of linear functions). The proof is in Appendix E. Next, we consider deterministic state transition.

Assumption 1 (Deterministic transitions). For all s ∈ S and a ∈ A*, there is a unique state* s
′∈ S to which the system transitions to after taking action a *on state* s. We emphasize that, although the transition is deterministic, the initial state distribution is stochastic
(although we assume that {st,1}t≤T is independently sampled from an initial distribution µ, our results extend to the scenarios when {st,1}t≤T are adversarially chosen). Additionally, the reward signals can be stochastic. Hence, learning is still challenging in this case. The goal is to achieve low regret over T rounds. The regret is defined as

$$\mathrm{Reg}_{T}:=\mathbb{E}\left[\sum_{t=1}^{T}\left(V^{\star}(s_{t,1})-V^{\pi_{t}}(s_{t,1})\right)\right],$$

The expectation here is taken over the randomness of algorithm and reward signals. While it is defined as an average for simplicity, a concentration inequality can yield the high-probability regret.

In this paper, we use asymptotic notations Θ̃(⋅) and Õ(⋅) to hides logarithmic and constant factors.

## 3.1 Other Linear Bellman Completeness Definitions In The Literature

Several closely related definitions of Linear Bellman Completeness have been considered in the literature. In the following, we demonstrate that some of these variant definitions face limitations due to additional ℓ2-norm assumptions. We present two commonly imposed assumptions in existing works below, and subsequently provide examples illustrating their potential limitations.

(1) Assuming Bounded ℓ2**-norm of Parameters.** Golowich & Moitra (2024); Zanette et al. (2020b;c) assume that any value function under consideration has its parameters bounded in ℓ2norm, i.e., when we apply the Bellman backup, the resulting state-action value function always lies in {Q ∶ Q(s, a) = ⟨ϕ(s, a), θ⟩, ∥θ∥2 ≤ R} where R is a pre-fixed polynomial in the dimension of the feature space. We will show that this assumption might not hold true since ∥θ∥2 is unnecessarily bounded under linear Bellman completeness.

(2) Assuming Non-expansiveness of Bellman Backup in ℓ2**-norm.** Song et al. (2022) assume that, after applying the Bellman backup, the ℓ2-norm of the value function parameters will not increase, i.e., for any θ, they assume the existence of parameter θ
′such that ∥θ
′∥2 ≤ ∥θ∥2 and ⟨ϕ(s, a), θ′⟩ =
Es
′∼T(s,a) maxa′⟨ϕ(s
′, a′), θ⟩ for all *s, a*. This assumption is stronger than the previous one and does not hold even in tabular MDPs, as we will show in the second example below. The following example demonstrates that the two assumptions above do not generally hold under linear Bellman completeness as the ℓ2-norm amplification can actually be arbitrarily large. Example 1 (Arbitrarily Large ℓ2-norm on Parameters). Consider a layered linear MDP with three states, s1, s2, s3, and a single action a1. Here s1 is in the first layer and s2 and s3 *are in the* second layer. For some ε and p, we define ϕ(s1, a1) = (
√ε, √p − ε), µ(s2) = (p/
√ε, 0)*, and* µ(s3) = (0, (1 − p)/√p − ε). We further define r(s2, ⋅) = ε and r(s3, ⋅) = 1*. We can verify that* P(s2∣s1, a1) = p and P(s3∣s1, a1) = 1 − p. Hence Q(s1, a1) = pε + 1 − p. We assume Q*-function is* parameterized by θ. Then, since ∥ϕ(s1, a1)∥ = p*, it must hold that* ∥θ∥ ≥ (pε+1−p)/p = ε+p
−1 −1.

While p can be arbitrarily small, the norm of θ *can be arbitrarily large.*
We may hope to "normalize" the features in this example so that the ℓ2-norm of the parameters is bounded. However, it is unclear how to do so since changing either ε or p will change the MDP,
and feature search is likely a hard problem. Essentially, this example breaks one of the assumptions in the original linear MDP (Jin et al., 2020) which requires the integral ∫ gµ to be bounded for any function g ∈ [0, 1]. Thus, while being a linear MDP, the original LSVI-UCB algorithm (Jin et al.,
2020) indeed will not work for this example. However, we note that our algorithm can still work. Nevertheless, as the above example leverages a careful design of the feature, we might hope that some non-expansiveness properties could hold under stronger representation assumptions (e.g., when state space is tabular). Unfortunately, the following example shows that Bellman backup can be expansive even in tabular MDPs.

Example 2 (Expansiveness of Bellman Backup in ℓ2-norm). *Consider a tabular MDP with horizon* H = 2, S states {s1, . . . , sS} in the first layer, a single state s in the second layer, and a single action a. On taking action a *in any state in the first-layer, the agent deterministically transitions* to s, and on taking action a in s *deterministically yields a reward of 1. Since linear Bellman* completeness captures tabular MDPs with one-hot encoded features where ϕ(si, a) = ei ∈ R
S+1 for i ≤ S and ϕ(s, a) = eS+1 = (0*, . . . ,* 0, 1)
⊺, the state-action value function at the second layer can be parameterized by θ2 = (0*, . . . ,* 0, 1)
⊺. However, applying the Bellman backup, since the returnto-go for any first-layer state siis 1 (because s always yields a reward of 1), the backed-up value function must be parameterized by θ1 = (1, 1*, . . . ,* 1)
⊺*. Here, we find that* ∥θ1∥2/∥θ2∥2 =
√S, thus showing that Bellman backup cannot guarantee non-expansiveness of the ℓ2*-norm.* Hence, in this paper, we aim not to assume any ℓ2-norm bound or ℓ2-norm non-expansiveness of the parameters. Unfortunately, without these assumptions, the ground truth parameter of the optimal value function can exponentially grow with the horizon as evidenced by the examples above, thus invalidating prior methods requiring bounded parameter. Our key contribution is an algorithm that remains efficient even if the parameter norm blows up but requiring deterministic transition.

## 3.2 Other Prior Works On Linear Bellman Completeness

In this section, we review prior efforts on RL under linear Bellman completeness and discuss various assumptions underlying these approaches. Efficient Algorithms under Generative Access. A generative model takes as input a state-action pair (*s, a*) and returns a sample s
′ ∼ T(⋅ ∣ *s, a*) and the reward signal. With such a generative model, Linear Least-Squares Value Iteration (LSVI) can achieve statistical and computational efficiency (Agarwal et al., 2019). However, generative access is a big assumption, and our work aims to operate with only online access. Efficient Algorithms under Explorability Assumption. Zanette et al. (2020c) propose a rewardfree algorithm under the assumption that every direction in the parameter space is reachable. This assumption, when translated into tabular MDPs, means that any state can be reached with a probability bounded below by some (large enough) positive constant. This does not hold if there are unreachable states or if the probability of reaching them is exponentially small. Computationally Intractable Algorithms. Zanette et al. (2020b) present a computationally intractable algorithm that requires solving an intractable optimization problem. In our work, we aim to only utilize a tractable squared loss minimization oracle. Few action MDPs. Golowich & Moitra (2024) propose a computationally efficient algorithm under linear Bellman completeness, inspired by the bonus-based exploration approach in LSVI-UCB (Jin et al., 2020) for Linear MDPs. While their algorithm extends to stochastic MDPs, both the sample complexity and running time have exponential dependence on the size of the action space. In comparison, our algorithm extends to infinite action spaces but relies on the transition dynamics to be deterministic. Deterministic Rewards or Deterministic Initial State. Several existing studies provide computationally and statistically efficient algorithms for more general settings but under stronger assumptions; these methods can be extended to linear Bellman completeness settings but similarly strong assumptions will also apply. Du et al. (2020) provide an algorithm based on a span argument that is efficient for MDPs that have linear optimal state-action value function (a.k.a. the Linear Q⋆setting),
deterministic transition dynamics, deterministic initial state, and stochastic rewards. Unfortunately, their approach cannot extend to settings with stochastic initial states, as we consider in our paper.

Another line of work due to Wen & Van Roy (2017) considers the Q⋆-realizable setting with deterministic dynamics, deterministic rewards, stochastic initial states, and bounded eluder dimension. Their approach can be extended to the linear bellman completeness setting when both rewards and dynamics are deterministic. However, their algorithm fails to converge when rewards are stochastic and thus may not apply to the problem setting that we consider. Efficient Algorithm in the hybrid RL setting. Song et al. (2022) develop efficient algorithms for the hybrid RL setting, where the learner has access to both online interaction and an offline dataset. However, they do not have a fully online algorithms. In summary, no previous work addresses the problem with stochastic initial states, stochastic rewards, and large action spaces. This is the gap that we aim to fill with this work.

## 4 Algorithm

In this section, we present our algorithm for online RL under linear Bellman completeness. See Algorithm 1 for pseudocode. The input to the algorithm consists of three components. First, the noise variances, {σh}
H
h=1 and σR, control the scale of the random noise. Second, a D-optimal design
(defined below) for the feature space.

Definition 2 (D-optimal design). The D-optimal design for the set of features Φ = {ϕ(s, a) ∶ s ∈
S, a ∈ A} is a distribution ρ over Φ *that maximizes* log det(∑ϕ∈Φ ρ(ϕ)ϕϕ⊺).

There always exist D-optimal designs with at most O(d 2) support points (Lemma 23). Many efficient algorithms can be applied to find approximate D-optimal designs such as the Frank-Wolfe.

The algorithm also requires a constrained squared loss minimization oracle Osq, and we introduce an instantiation of Osq in Section 6.

Algorithm 1 Null Space Randomization for Linear Bellman Completeness Require: - Noise variances {σh}
H
h=1and σR.

- A D-optimal design for Φ = {ϕ(s, a) ∶ s ∈ S, a ∈ A} given by {(ϕi, ρi)}m i=1.

- Squared loss minimization oracle Osq.

1: Define Σ1,h ∶= ∑
m i=1 ρiϕiϕ
⊺
ifor all h ∈ [H].

2: for t = 1*, . . . , T* do 3: Let θt,H+1 ← 0, Qt,H+1 ← 0, V t,H+1 ← 0.

4: for h = *H, . . . ,* 1 do 5: Let Pt,h be the orthogonal projection matrix onto span({ϕ(si,h, ai,h) ∶ i = 1*, . . . , t* − 1})
6: For i ∈ [m], define ϕ
∥
t,h,i = Pt,hϕi and ϕ
⊥
t,h,i = (I − Pt,h)ϕi 7: Let Λt,h ← ∑
m i=1 ρi(ϕ
∥
t,h,i(ϕ
∥
t,h,i)
⊺ + ϕ
⊥
t,h,i(ϕ
⊥
t,h,i)
⊺)
8: // Fit value function and reward using squared loss regression //
9: Compute θ̂t,h and ω̂t,h using the squared loss minimization oracle Osq as:

$$\begin{array}{l}{{\widetilde{\theta}_{t,h}\leftarrow\operatorname*{argmin}_{\theta\in\mathcal{O}(h)}\sum_{i=1}^{t-1}\left(\left\langle\theta,\phi\left(s_{i,h},a_{i,h}\right)\right\rangle-\overline{{{V}}}_{t,h+1}\left(s_{i,h+1}\right)\right)^{2}}}\\ {{\widetilde{\omega}_{t,h}\leftarrow\operatorname*{argmin}_{\omega\in\mathcal{O}(1)}\sum_{i=1}^{t-1}\left(\left\langle\omega,\phi\left(s_{i,h},a_{i,h}\right)\right\rangle-r_{i,h}\right)^{2}}}\end{array}$$
$$(1)$$
$$(2)$$

10: // Perturb the estimated parameters by adding Gaussian noise // 11: Update the parameters by sampling:

$$\begin{array}{l}{{\overline{{{\theta}}}_{t,h}\sim\widehat{\theta}_{t,h}+\mathcal{N}\Big(0,\,\sigma_{h}^{2}\big(I-P_{t,h}\big)\Lambda_{t,h}^{-1}\big(I-P_{t,h}\big)\Big)}}}\\ {{\overline{{{\omega}}}_{t,h}\sim\widehat{\omega}_{t,h}+\mathcal{N}\big(0,\,\sigma_{\mathrm{R}}^{2}\Sigma_{t,h}^{-1}\big)}}\end{array}$$

12: Define Qt,h(s, a) ← ⟨ωt,h + θt,h, ϕ(*s, a*)⟩ and V t,h(s) ← maxa Qt,h(*s, a*) for all (s, a)
13: **end for**
14: Define the policy πt such that πt,h(s) = argmaxa Qt,h(s, a)
15: Generate trajectory (st,1, at,1, rt,1, . . . , st,H, at,H, rt,H) ∼ πt 16: Define Σt+1,h ∶= Σt,h + ϕ(st,h, at,h)ϕ
⊺(st,h, at,h) for all h ∈ [H]
17: **end for**
The algorithm begins by initializing the covariance matrix Σ1,h for all h ∈ [H] using the optimal design, which differs from most standard LSVI-type algorithms where it is initialized to the identity matrix. We believe that the identity matrix is unsuitable here since we do not assume any ℓ2-norm bound on the parameters. Additionally, recalling that we assume the feature space spans R
d, it ensures Σt,h is invertible for all t and h. Otherwise, pseudo-inverses can be used instead. At each round t ∈ [T], the algorithm operates in a backward manner starting from the last horizon H. For each h ∈ [H], it first constructs the orthogonal projection matrix Pt,h onto the span of the historical data. It then decomposes the D-optimal design points into the span and null space components using the projection and constructs Λt,h. By separating the span and null space components, it facilitates clearer concentration bounds for the subsequent Gaussian noise. The algorithm then performs constrained squared loss regression to estimate the value function and reward function. Here we define O(W) ∶= {θ ∈ R
d∶ ∣⟨θ, ϕ(*s, a*)⟩∣ ≤ W for all s ∈ S, a ∈ A} for any W > 0. This *convex* constrained set is defined by the ℓ∞-functional-norm bound instead of the ℓ2-norm because we do not assume any bound on the ℓ2-norm of the learned parameters. Here we define Wh = Θ̃((d
√mH)
H−h(d 3/2 + d
√mH)) (detailed definition deferred to Appendix C).

We note that although Wh appears exponential, which may seem suspicious, this does not affect our sample efficiency due to the span argument that we introduce in the analysis. We note that prior RLSVI algorithms used truncation on value functions to explicitly avoid such an exponential blow-up. However, truncation does not work for linear Bellman completeness setting since the Bellman backup on a truncated value function is not necessarily a linear function anymore. Next, the algorithm perturbs the estimated parameters by adding Gaussian noise. The noise for the value function act *only in the null space* of the data covariance matrix. This ensures optimism while keeping the estimate accurate in the span space. It is a key modification from the standard RLSVI algorithm. The perturbation for the reward function is standard. Finally, the algorithm constructs the value function for the current horizon and the greedy policy with respect to it. It then generates the trajectory by executing the greedy policy, and the covariance matrix is updated.

## 5 Analysis

In this section, we provide the theoretical guarantees of Algorithm 1. A high-level proof sketch can be found in Appendix B and detailed proofs are in Appendix C. We first consider the case where the squared loss minimization oracle is exact. We then extend the analysis to the approximate oracle and the low inherent linear Bellman error setting in subsequent sections.

## 5.1 Prelude: Learning With Exact Square Loss Minimization Oracle

We first consider the most ideal setting where the squared loss minimization oracle is exact.

Assumption 2 (Exact Squared Loss Minimization Oracle). Line 9 of Algorithm 1 *is solved exactly.*
Then, we have the following regret bound. A proof sketch is provided in Appendix B for the readers convenience. Theorem 1 (Regret Bound with Exact Oracle). Under Assumptions 1 and *2, executing Algorithm* 1 with parameters σR = Θ̃(
√dH log(HT)) and σh = Θ̃((d
√mH)
H−h+1(
√d +
√mH))*, we have* RegT = Õ(d 5/2H
5/2 + d 2H
3/2
√T).

This result has several notable features. First, it does not depend on the number of actions. The only requirement for the action space is the ability to compute the argmax. Second, the 
√T-dependence on T is optimal, as it is necessary even in the bandit setting. Additionally, we emphasize that the dependence on 
√T arises solely from reward learning due to the application of elliptical potential lemma. In fact, if the reward function is known, our regret bound can be as small as Õ(dH2),
depending on T up to logarithmic factors. We elaborate on this observation in Appendix B. As a standard practice, Theorem 1 can be converted into a sample complexity bound below.

Corollary 1 (Sample Complexity Bound). Let ε ≤ 1. Under the same setting as Theorem 1, letting T ≥ Ω(d 4H3/ε 2), we get that the policy ̂π chosen uniformly from the set π1, . . . , πT enjoys performance guarantee E[V
⋆ − V
π̂] ≤ ε.

## 5.2 Learning With Approximate Square Loss Minimization Oracle

Assumption 3 (Approximate Squared Loss Minimization Oracle). We assume access to an approximate squared loss minimization oracle Osq apx *that takes as input a problem of the form:*
argminθ∈O(W)g(θ) ∶= ∑(ϕ(s,a),u)∈D(⟨θ, ϕ(*s, a*)⟩ − u)
2 *where* O(W) = {θ ∈ R
d∣ ∣⟨θ, ϕ(*s, a*)⟩∣ ≤
W} for some W ∈ R is a convex set, and D is a dataset of tuples {(ϕ(s, a), u)}. The oracle returns a point θ̂*that satisfies* g(θ̂)− minθ∈O(W) g(θ) ≤ ε 21 and θ̂∈ O(W +ε2) where ε1, ε2 ≤ 1 are precision parameters of the oracle. With an approximate oracle, the regret bound depends on an additional quantity defined below.

Assumption 4. There exists a constant γ > 1 such that, for any r ≤ d*, and for any* ϕ1, ϕ2*, . . . , ϕ*r ∈ Φ, the eigenvalues of the matrix Σ ∶= ∑
r i=1 ϕiϕ
⊺
i*are either zero or at least* 1/γ 2.

As a concrete example, it holds with γ = 1 when the MDP is tabular. This assumption implies that the eigenvalues of Σ
†are at most γ 2. Consequently, for any vector ϕ ∈ Φ, we have ∥ϕ∥Σ† ≤
∥ϕ∥2γ ≤ γ—this lower bound on the norm of any vector is exactly what we need for the analysis of an approximate oracle, while Assumption 4 simply serves as a sufficient condition for it. The following theorem provides the regret bound with the approximate oracle in terms of parameters ε1, ε2 and γ.

Theorem 2 (Regret Bound with Approximate Oracle). Under Assumptions 1, 3 and *4, executing* Algorithm 1 *with* σR = Θ̃(
√dH) and σh = Θ̃((d
√mH)
H−h+1(ε1γ
√H +
√d +
√mH)*, we have* RegT = Õ(d 5/2H
5/2 + d 2H
3/2
√T + ε1γ(dH2 + d 3/2H
√T)).

Compared to Theorem 1, the regret bound has an additional term that depends on the approximation error ε1γ. Typically, ε1 is from optimization and thus can be exponentially small with respect to the relevant parameters, as we later discuss in Section 6. Hence, we allow γ to be exponentially large.

Moreover, we note that ε2 does not appear in the regret bound since it only affects the constraint violation of the regression, whose effect to the statistical guarantees is of lower order and thus ignored. In addition, we note that the regret bound does not depend on the number of actions, and the dependence on T remains optimal, similar to the previous theorem.

## 5.3 Learning With Low Inherent Linear Bellman Error

Now we consider the setting where the MDP has low inherent linear Bellman error. Definition 3 (Inherent Linear Bellman Error). Given εB ≤ 1, an MDP M is said to have εB-
inherent linear Bellman error with respect to a feature mapping ϕ *if there exists a mapping* T ∶ R
d → R
d*so that, for all* θ ∈ R
d*and all* (s, a) ∈ S × A*, it holds that* ∣⟨T θ, ϕ(*s, a*)⟩ −
Es′∼T(s,a) maxa′⟨*θ, ϕ*(s
′, a′)⟩∣ ≤ εB. Moreover, we require that, for all h ∈ [H] and (s, a) ∈ S × A,
the random reward is bounded in [0, 1] with ∣rh(*s, a*) − ⟨ω
⋆h
, ϕ(s, a)⟩∣ ≤ εB *for some unknown* ω
⋆h∈ R
d.

With low inherent Bellman error, Assumption 4 is still necessary. The following theorem provides the regret bound in this case. We assume the exact oracle for simplicity.

Theorem 3 (Regret Bound with Low Inherent Bellman Error). Assume the MDP has εB-inherent Bellman error. Under Assumptions 1, 2 and 4, when executing Algorithm 1 *with parameters* σR =
Θ̃(
√dH + εBHT) and σh = Θ̃((d
√mH)
H−h+1(εBγ
√HT +
√εBT +
√d +
√mH))*, we have* RegT = Õ(d 5/2H
5/2 + d 2H
3/2
√T +
√εB(d 2H
5/2
√T + d 3/2H
3/2T) + εBγ(dH2
√T + d 3/2HT)).

Compared to Theorem 1, the regret bound includes two additional terms that depend on the inherent linear Bellman error εB. For both terms, the dependence on T is linear. We believe the T-dependence is unavoidable, as it also appears in similar settings (Zanette et al., 2020b). In addition, it is worth noting that the regret bound does not depend on the number of actions, and the other dependence on T remains optimal, similar to previous theorems.

## 6 Opening The Black-Box: Implementing Squared Loss Minimization Oracles In Algorithm 1

In this section, we detail a practical implementation of the desired squared loss oracle need by our algorithm. The implementation relies on the observation that a square loss minimization objective over a convex domain can be cast as a convex set feasibility problem—given a convex set K, return a point θ̂∈ K. Thus, we can use algorithms for convex set feasibility to implement the squared loss minimization oracles. However, even given this observation, our key challenge for an efficient algorithm is that the corresponding convex set could be exponentially large and only be described using exponentially many number of linear constraints. Fortunately, various works in the optimization literature propose computationally efficient procedures to find feasible points within such ill-defined sets, under mild oracle assumptions.

## 6.1 Computationally Efficient Convex Set Feasibility

We first paraphrase the work of Bertsimas & Vempala (2004) that provide a computationally efficient procedure for finding feasible points within a convex set by random walks. Notably, the computational complexity of their algorithm only depends logarithmically on the size of the convex set, and thus their approach is well suited for the corresponding convex feasibility problems that appear in our approach. At a high level, they provide an algorithm that takes an input an arbitrary convex set K ⊆ R
d, and returns a feasible point ẑ ∈ K. Their algorithm accesses the convex set K
via a separation oracle defined as follows.

Definition 4 (Separation oracle). A separation oracle for a convex set K*, denoted by* O
sep K
, is defined such that on any input z ∈ R
d, the oracle either confirms that z ∈ K or returns a hyperplane ⟨*a, z*⟩ ≤ b that separates the point z *from the set* K. In order to ensure finite time convergence for their procedure, they assume that the convex set K is not degenerate and is bounded in any direction. This is formalized by the following assumption. Assumption 5. *The convex set* K is (r, R)−Bounded, i.e. there exist parameters 0 < r ≤ R such that
(a) K ⊆ R∞(R)*, and (b) there exists a vector* z ∈ R
d*such that the shifted cube* (z + R∞(r)) ⊆ K.

The computational efficiency and the convergence guarantee of their algorithm are below.

Theorem 4 (Bertsimas & Vempala (2004)). Let δ ∈ (0, 1) and K ⊂ R
dbe an arbitrary convex set that satisfies Assumption 5 for some 0 ≤ r ≤ R. Then, Algorithm 2 (given in the appendix), when invoked with the separation oracle O
sep K w.r.t. K, returns a feasible point ẑ ∈ K with probability at least 1 − δ. Moreover, Algorithm 2 *makes* O(d log(R/δr)) *calls to the oracle* O
sep K*and runs in time* O(d 7log(R/δr)).

Notice that both the number of oracle calls and the running time only depend logarithmically on R and r, and thus their procedure can be efficiently implemented for our applications where R/r may be exponentially large in the corresponding problem parameters.

## 6.2 Computationally Efficient Estimation Of Value Function (Eqn (1))

We now described how to leverage the method by Bertsimas & Vempala (2004) to estimate the parameters for the value functions in (1) in Algorithm 1. Note that for any time t and horizon h ∈ [H], the objective in (1) is the optimization problem

$$\widehat{\theta}_{t,h}\leftarrow\operatorname*{argmin}_{\theta\in\mathcal{O}(W_{h})}\sum_{i=1}^{t-1}\left(\left\langle\theta,\phi\big(s_{i,h},a_{i,h}\big)\right\rangle-\overline{{{V}}}_{t,h+1}\big(s_{i,h+1}\big)\right)^{2},$$

where Wh = Θ̃((d
√mH)
H−h(ε1dγ√H +d 3/2 +d
√mH)). We provide a computationally efficient procedure to approximately solve the above given a linear optimization oracle over the feature space. Assumption 6 (Linear optimization oracle over the feature space). Learner has access to a linear optimization oracle Olin *that on taking input* θ ∈ R
d*, returns a feature* ϕ(s
′, a′) ∈
argmaxs,a⟨θ, ϕ(*s, a*)⟩.

$$(3)$$

The key observation we use is that under linear Bellman completeness (Definition 1) and deterministic dynamics (Assumption 1), any solution θ for (3) must satisfy ∑
t−1 i=1(⟨θ, ϕ(si,h, ai,h)⟩ −
V t,h+1(si,h+1))2 = 0. On the other hand, the converse also holds that any point θ ∈ O(Wh) for which the objective value is 0 must be a solution to (3). Thus, the minimization problem in (3) is equivalent to finding a feasible point within the convex set

$$\mathcal{K}:=\left\{\theta\in\mathbb{R}^{d}\,\left|\begin{array}{c}\left(\left(\theta,\phi(s_{i,h},a_{i,h}\right)\right)-\overline{V}_{t,h+1}(s_{i,h+1})\right)^{2}=0\text{for all}i\leq t\\ \left|\left(\theta,\phi(s,a)\right)\right|\leq W_{h}\text{for all}s,a\end{array}\right.\right\}.\tag{4}$$

Given the above reformulation of the optimization objective (3) as a feasibility problem, we can now use the procedure of Bertsimas & Vempala (2004) for finding θt,h ∈ K. However, we first need to define a separation oracle for the set K and verify Assumption 5. Unfortunately, there may not exist any r > 0 for which (z + R∞(r)) ⊆ K for some z ∈ R
din our case and thus the above K may not satisfy Assumption 5. This can, however, be easily fixed by artificially increasing the set K to allow for some approximation errors. In particular, let ε > 0 and define the convex set

$$\mathcal{K}_{\text{APX}}:=\begin{cases}\theta\in\mathbb{R}^{d}\\ \theta,\phi(s_{i,h},a_{i,h}))-\overline{V}_{t,h+1}(s_{i,h+1})\leq\varepsilon\text{for all}i\leq t\\ \langle\theta,\phi(s_{i,h},a_{i,h})\rangle-\overline{V}_{t,h+1}(s_{i,h+1})\geq-\varepsilon\text{for all}i\leq t\\ |\langle\theta,\phi(s,a)|\rangle\leq W_{h}+\varepsilon\text{for all}s,a\end{cases}.\tag{5}$$

Clearly, since there exists at least one point θt,h ∈ K, we must have that (θt,h + R∞(ε)) ⊆ KAPX. To ensure an outer bounding box for the set KAPX, we need to make an additional assumption. Assumption 7. Let Φ = {ϕ(*s, a*) ∣ s, a ∈ S × A}. There exist some R ≥ 0 *such that* 1 R
ei ∈ Φ*, where* ei denotes the unit basis vector along the i*-th direction in* R
d.

The above assumption ensures that K ⊆ B∞(WhR). Recall that we can tolerate the parameter R to be exponential in the dimension d or the horizon H. Finally, a separation oracle can be implemented using Olin (see Algorithm 4 for details). Thus, one can use Algorithm 2 (given in appendix), due to Bertsimas & Vempala (2004), and the guarantee in Theorem 4 to find a feasible point in KAPX, which corresponds to an approximate solution to (3).

Theorem 5. Let ε > 0, δ ∈ (0, 1), and suppose Assumption 7 holds with some parameter R > 0.

Additionally, suppose Assumption 6 *holds with the linear optimization oracle denoted by* Olin*. Then,*
there exists a computationally efficient procedure (given in Algorithm 4 *in the appendix), that for* any t ∈ [T] and h ∈ [H], returns a point θ̂t,h that, with probability at least 1 − δ*, satisfies*

$$\sum_{i=1}^{t-1}\left(\left\langle\widehat{\theta}_{t,h},\phi\big{(}s_{i,h},a_{i,h}\big{)}\right\rangle-\overline{{{V}}}_{t,h+1}\big{(}s_{i,h+1}\big{)}\right)^{2}\leq T\varepsilon\qquad{\mathrm{~and~}}\qquad\widehat{\theta}_{t,h}\in{\mathcal{O}}(W_{h}+\varepsilon).$$

Furthermore, Algorithm 4 *takes* O(d 7log( R
δε )) *time in addition to* O(d log( THR
δε )) *calls to* Olin.

The above techniques and Algorithm 4 can be similarly extended to get a computationally efficient procedure to estimate the reward parameter in (2). The main difference is that the value of the optimization objective in (2) is not zero at the minimizer (due to stochasticity). Thus, we need to construct a set feasibility problem for every desired target value of the objective function within the grid [0, ε, 2*ε, . . . ,* 2 − ε, 2] and use a separating hyperplane w.r.t. the ellipsoid constraint in (2) to implement the separating hyperplane for KAPX (which can be implemented using projections).

## 7 Conclusion

In this paper, we develop a computationally efficient RL algorithm under linear Bellman completeness with deterministic dynamics, aiming to bridge the statistical-computational gap in this setting.

Our algorithm injects random noise into regression estimates only in the null space to ensure optimism and leverages a span argument to bound regret. It handles large action spaces, random initial states, and stochastic rewards. Our key observation is that deterministic dynamics simplifies the learning process by ensuring accurate value estimates within the data span, allowing noise injection to be confined to the null space. Extending our algorithm to stochastic dynamics remains an open challenge.

## Acknowledgments

We thank Yuda Song, Zeyu Jia, Noah Golowich, and Sasha Rakhlin for useful discussions. AS acknowledges support from the Simons Foundation and NSF through award DMS-2031883, as well as from the DOE through award DE-SC0022199. WS acknowledges support from NSF IIS-2154711, NSF CAREER 2339395, and DARPA LANCER: LeArning Network CybERagents.

## References

Marc Abeille and Alessandro Lazaric. Linear thompson sampling revisited. In Artificial Intelligence and Statistics, pp. 176–184. PMLR, 2017.

Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. *CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep*, 32:96, 2019.

Alekh Agarwal, Yujia Jin, and Tong Zhang. Vo q l: Towards optimal regret in model-free rl with nonlinear function approximation. In *The Thirty Sixth Annual Conference on Learning Theory*, pp. 987–1063. PMLR, 2023.

Priyank Agrawal, Jinglin Chen, and Nan Jiang. Improved worst-case regret bounds for randomized least-squares value iteration. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pp. 6566–6573, 2021.

OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. *The International Journal of Robotics Research*, 39(1):3–20, 2020.

Mohammad Gheshlaghi Azar, Ian Osband, and Remi Munos. Minimax regret bounds for reinforce- ´
ment learning. In *International conference on machine learning*, pp. 263–272. PMLR, 2017.

Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. *arXiv preprint arXiv:1912.06680*, 2019.

Dimitris Bertsimas and Santosh Vempala. Solving convex programs by random walks. Journal of the ACM (JACM), 51(4):540–556, 2004.

Rajendra Bhatia. *Matrix analysis*, volume 169. Springer Science & Business Media, 2013. Jianyu Chen, Bodi Yuan, and Masayoshi Tomizuka. Model-free deep reinforcement learning for urban autonomous driving. In *2019 IEEE intelligent transportation systems conference (ITSC)*, pp. 2765–2771. IEEE, 2019.

Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning, pp. 2826–2836. PMLR, 2021.

Simon S Du, Jason D Lee, Gaurav Mahajan, and Ruosong Wang. Agnostic q-learning with function approximation in deterministic systems: Tight bounds on approximation error and sample complexity. *arXiv preprint arXiv:2002.07125*, 2020.

Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. *arXiv preprint arXiv:2112.13487*, 2021.

Noah Golowich and Ankur Moitra. Linear bellman completeness suffices for efficient online reinforcement learning with few actions. In *The Thirty Seventh Annual Conference on Learning* Theory. PMLR, 2024.

David Haussler. Decision theoretic generalizations of the pac model for neural net and other learning applications. In *The mathematics of generalization*, pp. 37–116. CRC Press, 2018.

Jiafan He, Heyang Zhao, Dongruo Zhou, and Quanquan Gu. Nearly minimax optimal reinforcement learning for linear markov decision processes. In *International Conference on Machine Learning*, pp. 12790–12822. PMLR, 2023.

Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Zhuoran Yang, Zhaoran Wang, Doina Precup, and Lin Yang. Randomized exploration in reinforcement learning with general value function approximation. In *International Conference on Machine Learning*, pp. 4607–4616. PMLR, 2021.

Haque Ishfaq, Qingfeng Lan, Pan Xu, A Rupam Mahmood, Doina Precup, Anima Anandkumar, and Kamyar Azizzadenesheli. Provable and practical: Efficient exploration in reinforcement learning via langevin monte carlo. *arXiv preprint arXiv:2305.18246*, 2023.

Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning, pp. 1704–1713. PMLR, 2017.

Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? *Advances in neural information processing systems*, 31, 2018.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In *Conference on learning theory*, pp. 2137–2143. PMLR, 2020.

Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. *Advances in neural information processing systems*, 34:13406–13418, 2021.

Tor Lattimore and Csaba Szepesvari. ´ *Bandit algorithms*. Cambridge University Press, 2020. Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pp. 661–670, 2010.

Aditya Modi, Jinglin Chen, Akshay Krishnamurthy, Nan Jiang, and Alekh Agarwal. Model-free representation learning and exploration in low-rank mdps. *Journal of Machine Learning Research*, 25(6):1–76, 2024.

Remi Munos. Error bounds for approximate value iteration. In ´ Proceedings of the National Conference on Artificial Intelligence, volume 20, pp. 1006. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2005.

Ian Osband, Benjamin Van Roy, and Zheng Wen. Generalization and exploration via randomized value functions. In *International Conference on Machine Learning*, pp. 2377–2386. PMLR, 2016.

Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. *Advances in Neural Information Processing Systems*, 26, 2013.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016.

Yuda Song, Yifei Zhou, Ayush Sekhari, J Andrew Bagnell, Akshay Krishnamurthy, and Wen Sun. Hybrid rl: Using both offline and online data can make rl efficient. arXiv preprint arXiv:2210.06718, 2022.

Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In *Conference on learning theory*, pp. 2898–2933. PMLR, 2019.

Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems, 33:6123–6135, 2020.

Yuanhao Wang, Ruosong Wang, and Sham Kakade. An exponential lower bound for linearly realizable mdp with constant suboptimality gap. *Advances in Neural Information Processing Systems*, 34:9521–9533, 2021.

Gellert Weisz, Philip Amortila, and Csaba Szepesv ´ ari. Exponential lower bounds for planning in ´
mdps with linearly-realizable optimal action-value functions. In *Algorithmic Learning Theory*, pp. 1237–1264. PMLR, 2021.

Zheng Wen and Benjamin Van Roy. Efficient reinforcement learning in deterministic systems with value function generalization. *Mathematics of Operations Research*, 42(3):762–782, 2017.

Runzhe Wu and Wen Sun. Making rl with preference-based feedback efficient via randomization.

In *The Twelfth International Conference on Learning Representations*, 2024.

Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. *arXiv preprint arXiv:2210.04157*, 2022.

Andrea Zanette, David Brandfonbrener, Emma Brunskill, Matteo Pirotta, and Alessandro Lazaric.

Frequentist regret bounds for randomized least-squares value iteration. In International Conference on Artificial Intelligence and Statistics, pp. 1954–1964. PMLR, 2020a.

Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent bellman error. In *International Conference on Machine Learning*, pp. 10978–10989. PMLR, 2020b.

Andrea Zanette, Alessandro Lazaric, Mykel J Kochenderfer, and Emma Brunskill. Provably efficient reward-agnostic navigation with linear value iteration. Advances in Neural Information Processing Systems, 33:11756–11766, 2020c.

Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learningvia reference-advantage decomposition. *Advances in Neural Information Processing Systems*, 33: 15198–15207, 2020.

Yinglun Zhu and Robert Nowak. Efficient active learning with abstention. *arXiv preprint* arXiv:2204.00043, 2022.

CONTENTS OF APPENDIX
1 Introduction 1 2 Related Works 3 3 Preliminaries 3 3.1 Other Linear Bellman Completeness Definitions in the Literature . . . . . . . . . . . . 4 3.2 Other Prior Works on Linear Bellman Completeness . . . . . . . . . . . . . . . . . . . 5 4 Algorithm 6 5 Analysis 7 5.1 Prelude: Learning with Exact Square Loss Minimization Oracle . . . . . . . . . . . . 7 5.2 Learning with Approximate Square Loss Minimization Oracle . . . . . . . . . . . . . 8 5.3 Learning with Low Inherent Linear Bellman Error . . . . . . . . . . . . . . . . . . . . 8 6 Opening the Black-Box: Implementing Squared Loss Minimization Oracles in **Algorithm 1** 9 6.1 Computationally Efficient Convex Set Feasibility . . . . . . . . . . . . . . . . . . . . . 9 6.2 Computationally Efficient Estimation of Value Function (Eqn (1)) . . . . . . . . . . . 9

## 7 Conclusion 10

A Table of Notation 16 B Proof Overview 17 B.1 Span Argument . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

| B.2   | Exploration in the Null Space    | 17   |
|-------|----------------------------------|------|

B.3 Proof Outline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

## C Full Proof For **Section 5** 18

C.1 High-probability Event and Boundedness . . . . . . . . . . . . . . . . . . . . . . . . . 19 C.2 Value Decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 C.3 Exploration in the Null Space . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27 C.4 Main Steps of the Proof . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30 D Supporting Lemmas 35 D.1 Pseudo Dimension and Covering Number . . . . . . . . . . . . . . . . . . . . . . . . . 38 E Linear MDPs and LQRs imply Linear Bellman Completeness 38 F Computationally Efficient Implementations for Optimization Oracles 39

## G Missing Details From **Section 6.2** 39

G.1 Computationally Efficient Estimation of Reward Function (Eqn. 2) . . . . . . . . . . 39

## A Table Of Notation

We list the notation used in this paper in table 1, for the convenience of reference.

| Table 1: Notation used in the paper.   |                                                                                        |                                              |           |
|----------------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------|-----------|
| Symbol                                 | Description                                                                            |                                              |           |
| O(W)                                   | {θ ∈ R d ∶ ∣⟨θ, ϕ(s, a)⟩∣ ≤ W for all s ∈ S, a ∈ A}                                    |                                              |           |
| R∞(W)                                  | {θ ∈ R d ∶ ∥θ∥∞ ≤ W}                                                                   |                                              |           |
| R2(W)                                  | {θ ∈ R d ∶ ∥θ∥2 ≤ W}                                                                   |                                              |           |
| ηt,h                                   | T (ωt,h+1 + θt,h+1) − θ̂t,h                                                             |                                              |           |
| R η t,h                                | ω ⋆ h − ω̂t,h                                                                           |                                              |           |
| ξ t                                    | ωt,h − ω̂t,h                                                                            |                                              |           |
| R                                      |                                                                                        |                                              |           |
| ξ t,h                                  | θt,h − θ̂t,h                                                                            |                                              |           |
| P                                      |                                                                                        |                                              |           |
| E high                                 | High probability event, defined in Definition 5                                        |                                              |           |
| E span t                               | Event that trajectory at round t is within the span of historical data, defined in (6) |                                              |           |
| E optm t                               | Optimism event at round t, defined in Lemma 14                                         |                                              |           |
| Ut,h                                   | Value function lower bound, defined in Appendix C.2                                    |                                              |           |
| B err                                  | Upper bound of ∥ω̂t,h − ω                                                               |                                              |           |
| R                                      | ⋆ h ∥Σt , defined in Definition 5                                                      |                                              |           |
| B err                                  | Upper bound of ∥θ̂t,h − T (ωt,h + θt,h+1)∥Σ̂t,h                                          |                                              |           |
| P                                      | , defined in Lemma 7                                                                   |                                              |           |
| B noise                                | Upper bound of ∥ξ                                                                      |                                              |           |
| R                                      | R t,h∥Σt,h , defined in Definition 5                                                   |                                              |           |
| P                                      | P                                                                                      |                                              |           |
| B noise,h                              | Upper bound of ∥ξ t,h∥Λt,h , defined in Definition 5                                   |                                              |           |
| R                                      | T                                                                                      |                                              |           |
| B ϕ                                    | Upper bound of ∑ t=1 ∥ϕ(st,h, at,h)∥Σ−1 defined in Lemma 16 t,h T span                 |                                              |           |
| B P ϕ                                  | Upper bound of ∑ t=1 1{E t                                                             | }∥ϕ(st,h, at,h)∥Σ̂† t,h , defined in Lemma 16 |           |
| BV                                     | Upper bound of ∣V t∣ conditioning on E span t and E high, defined in Lemma 13          |                                              |           |
| Σt,h                                   | ∑ m i=1 ρiϕiϕ ⊺                                                                        | t−1 i=1 ϕ(si,h, ai,h)ϕ ⊺ (si,h, ai,h)        |           |
| i + ∑                                  |                                                                                        |                                              |           |
| Σ̂t,h                                   | ∑ t−1                                                                                  | ⊺ (si,h, ai,h)                               |           |
| i=1 ϕ(si,h, ai,h)ϕ                     | √ 2d ⋅ B                                                                               | √ 2d ⋅ B                                     |           |
| P                                      | R                                                                                      |                                              |           |
| Wh                                     | Recursively defined as Wh−1 = Wh + 2ε2 +                                               | noise,h +                                    | noise + 1 |
| with WH+1 = 1                          |                                                                                        |                                              |           |

## B Proof Overview

In this section, we provide a sketch of the proof of Theorem 1 (exact oracle and zero inherent linear Bellman error) with the full proofs deferred to Appendix C. To better convey the intuition, we now assume that the reward function is known, as reward learning is largely standard. In particular, we temporarily remove the estimation and perturbation of rewards (Lines 9 and 11) and simply assume ωt,h = ω
⋆h in Line 12.

## B.1 Span Argument

The very first step of our analysis revolve around two complimentary cases - whether the trajectory at round t is in the span of the historical data or not. Let Dt,h ∶= {ϕ(si,h, ai,h)}ti=1 and define E
span t as the event that the trajectory at round t is in the span of the historical data, i.e.,
E
span t∶= {∀h ∈ [H] ∶ ϕ(st,h, at,h) ∈ span(Dt−1,h)} . (6)
(1) In-span case. When the trajectory generated in the round t is completely within the span of historical data, we can assert that the value function estimation is accurate under πt. Particularly, by linear Bellman completeness, the Bayes optimal of the regression in Line 9 zeros the empirical risk, as formally stated in the following lemma.

Lemma 1. For any t ∈ [T]*, we have* ∑
t−1 i=1(⟨θ̂t,h, ϕ(si,h, ai,h)⟩ − V t,h+1(si,h+1))2 = 0.

Define Ut(⋅) as a version of V t(⋅) that minimizes V t(st,1) while satisfying the high probability bound (precise definition provided at the beginning of Appendix C.2). It implies the following.

Lemma 2. For any t ∈ [T]*, whenever* E
span t*holds, we have* V t(st,1) = Ut(st,1) = V
πt (st,1).

To understand Lemma 2, we consider two fact: (1) πt is the optimal policy for the estimated value function V t, and (2) both V t and Ut has accurate value estimate for the trajectory induced by πt, starting from st,1, because it is in the span of the historical data when E
span tholds.

(2) Out-of-span case. When any segment of the trajectory is not within the span, we simply pay H in regret and can assert that this will not occur too many times. To see this, we observe the following fact: whenever E
span tdoes not hold, there must exists h ∈ [H] such that dim span(Dt,h) =
dim span(Dt−1,h) + 1 by definition. Since the dimension of spans cannot exceed d for any h ∈ [H],
the case that E
span tdoes not hold cannot happen for more than dH times. We formally state it in the following lemma.

Lemma 3. *We have* ∑
T
t=1 1{(E
span t)
∁} ≤ dH.

Hence, we have the following decomposition:

V
⋆(st,1) − V
πt (st,1) = 1{E
span
t}(V
⋆(st,1) − V
πt (st,1)) + 1{(E
$${\bf1}\{({\mathfrak{E}}_{t}^{\mathrm{span}})^{\mathbb{C}}\}\Big(V^{\star}(s_{t},$$
⋆(st,1) − V
πt (st,1))
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸ ¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶
≤ dH2 when summed over t
Therefore, we only need to focus on the rounds where E
span
tholds. This will be the aim of the
subsequent sections.

B.2 EXPLORATION IN THE NULL SPACE
Lemma 1 implies that the estimation error only comes from the null space of the historical data, i.e.,
null({ϕ(si,h, ai,h) ∶ i = 1*, . . . , t* − 1}). Therefore, we only need to explore in this null space. While adding explicit bonus is infeasible under linear Bellman completeness, we add noise (Line 11) that can cancel out the estimation error in the null space. This achieves the following: Lemma 4 (Optimism with constant probability). *Denote* E
optm t*as the event that* V
⋆(st,1) ≤
V t(st,1). Then, for any t ∈ [T]*, we have* Pr(E
optm t) ≥ Γ
2(−1) where Γ is the cumulative distribution function of the standard normal distribution. This result has been the key idea in randomized RL algorithms, such as RLSVI. In the next section, we will explore how this lemma is utilized.

$${}_{t,1})-V^{\pi_{t}}(s_{t,})$$

## B.3 Proof Outline

In this section, we outline the structure of the whole proof. Let Ṽ denote an i.i.d. copy of V , and Ẽspan t,Ẽoptm tdenote the counterpart of E
span t,E
optm tfor Ṽ. We first invoke Lemma 2 and get 1{E
span t}(V
⋆(st,1) − V
πt (st,1)) = 1{E
span t}(V
⋆(st,1) − Ut(st,1)) ≤ V
⋆(st,1) − 1{E
span t}Ut(st,1)
where the last step is by the non-negativity of V
⋆. Next, we apply Lemma 4 and get

$\leq\frac{\mathbb{E}}{V_{t}}\left[\min\{\widetilde{V}_{t}(s_{t,1}),H\}-1\{\mathbf{c}_{t}^{\text{span}}\}U_{t}(s_{t,1})\left|\widetilde{\mathbf{c}}_{t}^{\text{optm}}\right.\right]$
Split it into two parts:

$$=\operatorname*{\mathbb{E}}_{\mathbb{P}}\left[\mathbf{1}\{{\widehat{\mathfrak{E}}}_{t}^{\mathrm{span}}\}\Big(\min\{{\widehat{V}}_{t}(s_{t,1}),H\}-\mathbf{1}\{{\mathfrak{E}}_{t}^{\mathrm{span}}\}U_{t}(s_{t,1})\Big)\Big|\,{\widehat{\mathfrak{E}}}_{t}^{\mathrm{optm}}\right]$$
Ṽt
$+\frac{\mathbb{E}}{\widetilde{V}_{t}}\Big{[}\mathbf{1}\{(\widetilde{\mathbf{e}}_{t}^{\text{span}})^{\mathbb{C}}\}\Big{(}\min\{\widetilde{V}_{t}(s_{t,1}),H\}-\mathbf{1}\{\mathbf{e}_{t}^{\text{span}}\}U_{t}(s_{t,1})\Big{)}\Big{]}\widetilde{\mathbf{e}}_{t}^{\text{optim}}\Big{]}$
Note that the quantity inside the first expectation is non-negative, so we can peel off the conditioning event; the quantity in the second term is simply upper bounded by H. Hence, we have

≤1 Γ2(−1) E Ṽt [1{Ẽspan t}( min{Ṽt(st,1), H} − 1{E span t}Ut(st,1))] + 1 Γ2(−1) E Ṽt [1{(Ẽspan t) ∁}H] Now we split the first term into two parts again: = 1 Γ2(−1) E Ṽt [1{Ẽspan t} min{Ṽt(st,1), H} − 1{E span t}Ut(st,1)] + 1 Γ2(−1) E Ṽt [1{(Ẽspan t) ∁ ∩ E span t}Ut(st,1)] + 1 Γ2(−1) E Ṽt [1{(Ẽspan t) ∁}H] ≤ 1 Γ2(−1) E Ṽt [1{Ẽspan t} min{Ṽt(st,1), H} − 1{E span t}Ut(st,1)] + 2 Γ2(−1) E Ṽt [1{(Ẽspan t) ∁}H] where we used the fact that 1{E span t}Ut(st,1) ≤ H. Taking the expectation over the randomness of

the algorithm and use the tower property, which converts Ṽ into V , we obtain
≤1 Γ2(−1)
E[1{E
span t} min{V t(st,1), H} − 1{E
span t}Ut(st,1)] +2 Γ2(−1)
E[1{(E
span t)
∁}H]
The first term is upper bounded by zero due to Lemma 2, and the second term is upper bounded by dH2by Lemma 3 when summed over t. This finishes the proof.

Remark 1 (Span Argument and Exponential Blow-Up). In the proof sketch above, we did not utilize any ℓ2-norm bound on θt,h or θ̂t,h as did in many prior works. We actually cannot leverage them since they can be exponentially large due to the addition of exponentially large noise. This phenomenon is widely observed in the literature (e.g., Agrawal et al. (2021); Zanette et al. (2020a)) and is addressed through truncation. However, truncation does not work under linear Bellman completeness, as the Bellman backup of a truncated value function is not necessarily linear. This is why we use the span argument to circumvent this issue.

## C Full Proof For Section 5

In this section, we present and prove the following main theorem, which provides the regret bound in terms of parameters ε1, ε2, and εB. Setting ε1 = ε2 = εB = 0 yields Theorem 1, setting εB = 0 yields Theorem 2, and setting ε1 = ε2 = 0 yields Theorem 3. Theorem 6. Assume the MDP has εB-inherent linear Bellman error. Under Assumptions 1, 3 and 4, when executing Algorithm 1 *with parameters* σR =
√HBR
err and σh ≥
√H(
√3γBP
err +
√8m(Wh +

$\varepsilon_{2}$)), we have_  $$\mathbb{E}\left[\sum_{t=1}^{T}\left(V^{\star}\left(s_{t,1}\right)-V^{\pi_{t}}\left(s_{t,1}\right)\right)\right]=\widetilde{O}\left(d^{5/2}H^{5/2}+d^{2}H^{3/2}\sqrt{T}+\varepsilon_{1}\gamma\left(dH^{2}+d^{3/2}H\sqrt{T}\right)\right)$$
$$+\sqrt{\varepsilon_{\mathrm{B}}}\Big(d^{2}H^{5/2}\sqrt{T}+d^{3/2}H^{3/2}T\Big)+\varepsilon_{\mathrm{B}}\gamma\Big(d H^{2}\sqrt{T}+d^{3/2}H T\Big)\Big).$$
Exact value of parameters σR and σh **in Theorem** 6. We define WH+1 = 1 and recursively define Wh−1 = Wh + 2ε2 +
√2d ⋅ B
P
noise,h +
√2d ⋅ B
R
noise + 1. Plugging the definition of all these symbols involved and ignoring lower order terms (i.e., logarithmic and constant terms), we get

$${\overline{{H}}}+\varepsilon_{\mathrm{B}}\cdot d\gamma{\sqrt{H}}$$
$$I_{\gamma}{\lor}I$$

Wh−1 ≈ d
√mH ⋅ Wh + ε1 ⋅ dγ√H + εB ⋅ dγ√HT +
√εB ⋅ d
√T + d 3/2. (7)
Solving this recursion, we get

$$W_{h}\approx\left(d\sqrt{mH}\right)^{H+1-h}+\left(d\sqrt{mH}\right)^{H-h}\left(\varepsilon_{1}\cdot d\gamma\sqrt{H}+\varepsilon_{\rm B}\cdot d\gamma\sqrt{H\overline{T}}+\sqrt{\varepsilon_{\rm B}}\cdot d\sqrt{\overline{T}}+d^{3/2}\right)$$ $$\approx\left(d\sqrt{mH}\right)^{H-h}\left(\varepsilon_{1}\cdot d\gamma\sqrt{H}+\varepsilon_{\rm B}\cdot d\gamma\sqrt{H\overline{T}}+\sqrt{\varepsilon_{\rm B}}\cdot d\sqrt{T}+d^{3/2}+d\sqrt{mH}\right).$$

We insert this into the value of σh and get

$$\sigma_{h}\approx\left(d{\sqrt{m H}}\right)^{H-h+1}\left(\varepsilon_{1}\cdot\gamma{\sqrt{H}}+\varepsilon_{\mathrm{B}}\cdot\gamma{\sqrt{H T}}+{\sqrt{\varepsilon_{\mathrm{B}}}}\cdot{\sqrt{T}}+d^{1/2}+{\sqrt{m H}}\right).$$

We can also get the value of σR as

$$(T)$$
$$\gamma2_{\cdot}$$
$$\sigma_{\mathrm{R}}\approx{\sqrt{H}}{\Big(}{\sqrt{d\log(H T)}}+\varepsilon_{1}+{\sqrt{\varepsilon_{\mathrm{B}}T}}{\Big)}.$$

Define Λ = ∑
m i=1 ρiϕiϕ
⊺ i
. It is straightforward that both Λ and Λt,h (constructed in Line 7 of Algorithm 1) are invertible. We define λ ∶= maxs,a ∥ϕ(*s, a*)∥Λ−1 and λt,h ∶= maxs,a ∥ϕ(*s, a*)∥Λ−1 t,h
.

Lemma 5. The matrices Λ and Λt,h *are invertible. Furthermore, we also have that*
- λ ≤
√d;

- $\lambda_{t,h}\leq\sqrt{2d}\:for\:all\:t\in[T]\:and\:all\:h\in[H]$. 
Proof of Lemma 5. By the last item in Lemma 23, we have λ ≤
√d. In what follows, we will show that Λ ⪯ 2Λt,h, which implies λt,h ≤
√2λ ≤
√2d.

For any x ∈ R
d, we have

$$x^{\top}\Lambda x=\sum_{i=1}^{m}\rho_{i}(x^{\top}\phi_{i})^{2}=\sum_{i=1}^{m}\rho_{i}\left(x^{\top}\phi_{1,h,i}^{\dagger}+x^{\top}\phi_{t,h,i}^{\dagger}\right)^{2}$$ $$\leq2\sum_{i=1}^{m}\rho_{i}\left(x^{\top}\phi_{t,h,i}^{\dagger}\right)^{2}+2\sum_{i=1}^{m}\rho_{i}\left(x^{\top}\phi_{t,h,i}^{\dagger}\right)^{2}\qquad\qquad\text{(using}(a+b)^{2}\leq2a^{2}+2b^{2})$$ $$=2x^{\top}\Lambda_{t,h}x.$$

This implies that Λ ⪯ 2Λt,h.

C.1 HIGH-PROBABILITY EVENT AND BOUNDEDNESS
Lemma 6 (Reward estimation). With probability at least 1 − δ, for any t ∈ [T] and h ∈ [H],

$$\|\widehat{\omega}_{t,h}-\omega_{h}^{\star}\|_{\Sigma_{t}}\leq\sqrt{1030(1+\varepsilon_{2})^{4}d}$$

√
4d log (8(1 + ε2)e 2T2H/δ) + 4ε 2 1 + 16(1 + ε2)(1 + εBT).

Proof of Lemma 6. For the ease of notation, we fixed t and h in the proof and simply write the regression problem as

$\square$
$$\widehat{\omega}\leftarrow\operatorname{argmin}_{\omega\in{\mathcal{O}}(1)}\sum_{i=1}^{n}\left(\omega^{\top}\phi_{i}-r_{i}\right)^{2}$$
where we have dropped the subscripts t and h for notational simplicity. Here ϕi and ri are abbreviated notations for ϕ(si,h, ai,h) and ri,h, respectively, and n = t − 1. Note that, due to approximate oracle (Assumption 3), ω̂ actually belongs to O(1 + ε2) instead of O(1). Denote C as an ℓ1-norm α-cover (Definition 6) on O(1+ε2) such that for any ω ∈ O(1+ε2),
there exists a ω̃ ∈ C, such that ∑
n i=1∣ω
⊺ϕi − ω̃
⊺ϕi∣/n ≤ α. Since O(1 + ε2) is a linear function class, which has pseudo-dimension d (Definition 8), we have

$$|\mathcal{C}|\leq\left(8(1+\varepsilon_{2})e^{2}/\alpha\right)^{d}\tag{8}$$  by Lemma 27. Now define $z_{i}^{\omega}=(\omega^{\intercal}\phi_{i}-r_{i})^{2}-((\omega^{\star})^{\intercal}\phi_{i}-r_{i})^{2}$. Then we have $|z_{i}^{\omega}|\leq4(1+\varepsilon_{2})^{2}$, and 
$$(8)$$
and

E i [z ω i] = E i [(ω ⊺ϕi − (ω ⋆) ⊺ϕi)(ω ⊺ϕi + (ω ⋆) ⊺ϕi − 2ri)] = E i [(ω ⊺ϕi − (ω ⋆) ⊺ϕi) (ω ⊺ϕi − (ω ⋆) ⊺ϕi + 2((ω ⋆) ⊺ϕi − ri))] ≥ (ω ⊺ϕi − (ω ⋆) ⊺ϕi) 2 − 4(1 + ε2)εB,
and moreover,

$$\mathbb{E}\big{[}(z_{i}^{\omega})^{2}\big{]}=\mathbb{E}\big{[}(\omega^{\top}\phi_{i}-(\omega^{\star})^{\top}\phi_{i})^{2}(\omega^{\top}\phi_{i}+(\omega^{\star})^{\top}\phi_{i}-2r_{i})^{2}\big{]}\leq16(1+\varepsilon_{2})^{2}(\omega^{\top}\phi_{i}-(\omega^{\star})^{\top}\phi_{i})^{2}$$

We note that z ω i −Ei z ω i is a martingale difference sequence and ∣z ω i −Ei z ω i∣ ≤ 8(1+ε2)
2. Applying Freedman's inequality (Lemma 22) and a union bound over ω ∈ C, we have with probability at least 1 − δ, for all ω ∈ C,

$$\sum_{i=1}^{n}(\omega^{\top}\phi_{i}-(\omega^{*})^{\top}\phi_{i})^{2}-\sum_{i=1}^{n}z_{i}^{\omega}$$ $$\leq\eta\sum_{i=1}^{n}16(1+\varepsilon_{2})^{2}(\omega^{\top}\phi_{i}-(\omega^{*})^{\top}\phi_{i})^{2}+\frac{8(1+\varepsilon_{2})^{2}\log(|\mathcal{C}|/\delta)}{\eta}+4(1+\varepsilon_{2})\varepsilon_{\mathrm{B}}T.\tag{9}$$
Recall that ω̂ is the least square solution. Denote ω̃ ∈ C as the point that is closest to ω̂, which means
that: ∑
n
i=1∣ω̂
⊺ϕi − ω̃
⊺ϕi∣ ≤ nα. We can derive the following relationship between ω̂ and ω̃:
n ∑ i=1 (ω̂ ⊺ϕi − (ω ⋆) ⊺ϕi) 2≤ 2 n ∑ i=1 (ω̂ ⊺ϕi − ω̃ ⊺ϕi) 2 + 2 n ∑ i=1 (ω̃ ⊺ϕi − (ω ⋆) ⊺ϕi) 2≤ 2n 2α 2 + 2 n ∑ i=1 (ω̃ ⊺ϕi − (ω ⋆) ⊺ϕi) 2, n ∑ i=1 z ω̃ i − n ∑ i=1 z ω̂ i = n ∑ i=1 (ω̃ ⊺ϕi − ω̂ ⊺ϕi)(ω̃ ⊺ϕi + ω̂ ⊺ϕi − 2ri) ≤ 4(1 + ε2)nα.
Now plug ω̃ into (9) and re-arrange terms, we get:

Now plug $\omega$ into (3) and rearranging terms, we get:  $$\sum_{i=1}^{n}\left(\widetilde{\omega}^{\top}\phi_{i}-\left(\omega^{*}\right)^{\top}\phi_{i}\right)^{2}\leq\frac{1}{1-16(1+\varepsilon_{2})^{2}\eta}\sum_{i=1}^{n}z_{i}^{\top}+\frac{8(1+\varepsilon_{2})^{2}}{\eta(1-16(1+\varepsilon_{2})^{2}\eta)}\cdot\log(|\mathcal{C}|/\delta)+\frac{4(1+\varepsilon_{2})\varepsilon_{0}T}{1-16(1+\varepsilon_{2})^{2}\eta}.$$  Setting $\eta=(32(1+\varepsilon_{2})^{2})^{-1}$, we get 
n ∑ i=1 (ω̃ ⊺ϕi − (ω ⋆) ⊺ϕi) 2≤ 2 n ∑ i=1 z ω̃ i + 512(1 + ε2) 4log(∣C∣/δ) + 8(1 + ε2)εBT.
Using the relationships between ω̂ and ω̃ that we derived above, we have:

$$\sum_{i=1}^{n}({\widehat{\omega}}^{\top}\phi_{i}-(\omega^{\star})^{\top}\phi_{i})^{2}$$
$$\sum_{i=1}^{n}(\overline{\omega}^{\top}\phi_{i}-(\omega^{*})^{\top}\phi_{i})^{2}$$ $$\leq2n^{2}\alpha^{2}+4\sum_{i=1}^{n}z_{i}^{\frac{\alpha}{\alpha}}+1024(1+\varepsilon_{2})^{4}\log(|\mathcal{C}|/\delta)+16(1+\varepsilon_{2})\varepsilon_{\rm B}T.$$ $$\leq2n^{2}\alpha^{2}+4\sum_{i=1}^{n}z_{i}^{\overline{\omega}}+1024(1+\varepsilon_{2})^{4}\log(|\mathcal{C}|/\delta)+16(1+\varepsilon_{2})n\alpha+16(1+\varepsilon_{2})\varepsilon_{\rm B}T.$$

Since ω̂ is the (approximate) least square solution, we have ∑i z ω̂
i ≤ ε 21. This implies that:

$$\sum_{i=1}^{n}(\overline{{{\omega}}}^{\top}\phi_{i}-(\omega^{*})^{\top}\phi_{i})^{2}\leq2n^{2}\alpha^{2}+4\varepsilon_{1}^{2}+1024(1+\varepsilon_{2})^{4}\log(|C|/\delta)+16(1+\varepsilon_{2})(n\alpha+\varepsilon_{\rm B}T).$$