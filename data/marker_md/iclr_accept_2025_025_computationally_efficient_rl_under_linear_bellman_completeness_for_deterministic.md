# COMPUTATIONALLY EFFICIENT RL UNDER LINEAR BELLMAN COMPLETENESS FOR DETERMINISTIC DY-NAMICS

Runzhe Wu∗ Cornell University rw646@cornell.edu Ayush Sekhari∗ MIT sekhari@mit.edu

Akshay Krishnamurthy Microsoft Research akshaykr@microsoft.com Wen Sun Cornell University ws455@cornell.edu

## ABSTRACT

We study computationally and statistically efficient Reinforcement Learning algorithms for the *linear Bellman Complete* setting. This setting uses linear function approximation to capture value functions and unifies existing models like linear Markov Decision Processes (MDP) and Linear Quadratic Regulators (LQR). While it is known from the prior works that this setting is statistically tractable, it remained open whether a computationally efficient algorithm exists. Our work provides a computationally efficient algorithm for the linear Bellman complete setting that works for MDPs with large action spaces, random initial states, and random rewards but relies on the underlying dynamics to be deterministic. Our approach is based on randomization: we inject random noise into least squares regression problems to perform optimistic value iteration. Our key technical contribution is to carefully design the noise to only act in the null space of the training data to ensure optimism while circumventing a subtle error amplification issue.

## 1 INTRODUCTION

Various application domains of Reinforcement Learning (RL)—including game playing, robotics, self-driving cars, and foundation models—feature environments with large state and action spaces. In such settings, the learner aims to find a well performing policy by repeated interactions with the environment to acquire knowledge. Due to the high dimensionality of the problem, function approximation techniques are used to generalize the knowledge acquired across the state and action space. Under the broad category of function approximation, model-free RL stands out as a particularly popular approach due to its simple implementation and relatively better sample efficiency in practice. In model-free RL, the learner uses function approximation (e.g., an expressive function class like deep neural networks) to model the state-action value function of various policies in the underlying MDP. In fact, the combination of model-free RL with various empirical exploration heuristics has led to notable empirical advances, including breakthroughs in game playing [\(Silver et al.,](#page-11-0) [2016;](#page-11-0) [Berner](#page-10-0) [et al.,](#page-10-0) [2019\)](#page-10-0), robot manipulation [\(Andrychowicz et al.,](#page-10-1) [2020\)](#page-10-1), and self-driving [\(Chen et al.,](#page-10-2) [2019\)](#page-10-2).

Theoretical advancements have paralleled the practical successes in RL, with tremendous progress in recent years in building rigorous statistical foundations to understand what structures in the environment and the function class suffice for sample-efficient RL. These advancements are supported by optimal exploration strategies that align with the corresponding structural assumptions, and by now we have a rich set of tools and techniques for sample-efficient RL in MDPs with large state/action spaces [\(Russo & Van Roy,](#page-11-1) [2013;](#page-11-1) [Jiang et al.,](#page-11-2) [2017;](#page-11-2) [Sun et al.,](#page-11-3) [2019;](#page-11-3) [Wang et al.,](#page-11-4) [2020;](#page-11-4) [Du et al.,](#page-10-3) [2021;](#page-10-3) [Jin et al.,](#page-11-5) [2021;](#page-11-5) [Foster et al.,](#page-10-4) [2021;](#page-10-4) [Xie et al.,](#page-12-0) [2022\)](#page-12-0). However, despite a rigorous statistical foundation, a significant challenge remains: many of these theoretically rigorous approaches

<sup>∗</sup>Equal contribution.

for rich function approximation are not computationally feasible, and thus have limited practical applicability. For example, some require solving complex optimization problems that are computationally intractable in practice [\(Zanette et al.,](#page-12-1) [2020b\)](#page-12-1); others require deterministic dynamics and initial states [\(Du et al.,](#page-10-5) [2020\)](#page-10-5); and some methods depend on maintaining large and complex version spaces [\(Jin et al.,](#page-11-5) [2021;](#page-11-5) [Du et al.,](#page-10-3) [2021\)](#page-10-3) which are intractable in terms of memory and computation.

One of the most striking examples of this statistical-computational gap is observed in the *Linear Bellman Completeness* setting, which is perhaps one of the simplest learning settings. Linear Bellman completeness serves as a bridge between RL and control theory literature as it provides a unified framework to capture Linear MDPs [\(Jin et al.,](#page-11-6) [2020;](#page-11-6) [Agarwal et al.,](#page-10-6) [2019;](#page-10-6) [Zanette et al.,](#page-12-1) [2020b\)](#page-12-1) and the Linear Quadratic Regulator (LQR), two popular models in RL and control respectively. In particular, the linear Bellman completeness setting captures MDPs where the state-action value function of the optimal policy is a linear function of some pre-specified feature representations (of states and actions), and the Bellman backups of linear state-action value functions are linear (w.r.t. some feature representation). Naturally, for this setting, the learner utilizes the function class F consisting of all linear functions over the given feature representation as the value function class for model-free RL. In addition to considering a linear class, we also assume that the class F exhibits low inherent Bellman error—a structural assumption that quantifies the error in approximating the Bellman backup of functions within F. The first assumption, i.e., linearity of optimal state-action value function, is perhaps the simplest modeling assumption one can make in RL with function approximation. Furthermore, emerging evidence suggests that linearity is practically useful, as with adequate feature representation, linear functions can represent value functions in various domains. The second assumption, i.e. low inherent Bellman error of the class, while being a bit mysterious, is a natural condition for statistical tractability for classic algorithms such as Fitted Q-iteration (FQI) and temporal difference (TD) learning with linear function approximation [\(Munos,](#page-11-7) [2005;](#page-11-7) [Zanette](#page-12-1) [et al.,](#page-12-1) [2020b\)](#page-12-1). It is also well-known that linearity alone does not suffice for efficient RL [\(Wang et al.,](#page-12-2) [2021;](#page-12-2) [Weisz et al.,](#page-12-3) [2021\)](#page-12-3).

While the prior works have shown that RL with linear bellman completeness is statistically tractable, and one can learn with sample complexity that scales polynomially with both d and H (where d is the dimensionality of the feature representation and H is the horizon of the RL problem), the proposed algorithms that obtain such sample complexity in the online RL setting are not computationally efficient. Given the simplicity of the problem, it was conjectured that a computationally efficient algorithm should exist. However, no such algorithms were proposed. Unfortunately, the classical approaches of combining supervised learning techniques with RL in the online setting, e.g., value function iteration, which are computationally efficient by design, fail to extend to be statistically tractable due to exponential blowups from error compounding, especially without making norm-boundedness assumptions. On the other hand, the techniques of adding quadratic exploration bonuses, e.g., the one proposed in LinUCB [\(Li et al.,](#page-11-8) [2010\)](#page-11-8) and used in LSVI for linear MDPs, also fail here as Bellman backups of quadratic functions are not necessarily within the linear class F. In fact, the search for a computationally efficient algorithm with large action spaces is open even when the transition dynamics are deterministic.

In this work, we provide the first computationally efficient algorithm for the linear Bellman complete setting with deterministic dynamics, that enjoys regret bound of <sup>O</sup>̃(<sup>d</sup> <sup>5</sup>/<sup>2</sup>H<sup>5</sup>/<sup>2</sup> + <sup>d</sup> <sup>2</sup>H<sup>3</sup>/<sup>2</sup><sup>T</sup> <sup>1</sup>/<sup>2</sup> ) for feature dimension d, horizon H, and number of rounds T. Importantly, our algorithm works with large action spaces, stochastic reward functions, and stochastic initial states. The key ideas of our algorithm are twofold: using *randomization* to encourage exploration and leveraging a *span argument* to bound the regret. While adding random noise to the learned parameters has been quite successful in linear function approximation, unfortunately, for our specific setting, since we need to add sufficiently large noise to cancel out the estimation error, blind randomization can cause the corresponding parameters to grow exponentially with the horizon. We avoid paying for this blow-up by only adding noise to the null space of the data. In particular, when the dynamics are deterministic, by adding exploration noise only in the null space, we can learn the value function exactly for any trajectories that lie within the span of the data seen so far. Additionally, a simple span argument bounds the number of times the trajectories fall outside the span of the historical data. Together, these techniques leads to our polynomial sample complexity bound. The resulting algorithm relies on linear regression oracles under convex constraints, which we show can be approximately solved via a random-walk-based algorithm [\(Bertsimas & Vempala,](#page-10-7) [2004\)](#page-10-7).

#### 2 RELATED WORKS

Computational Efficient RL under Linear Bellman Completeness. Numerous works have focused on computationally efficient RL within the scope of linear Bellman completeness (LBC). The simplest setting is tabular MDPs where computationally efficient and near-optimal algorithms have been well known [\(Azar et al.,](#page-10-8) [2017;](#page-10-8) [Zhang et al.,](#page-12-4) [2020;](#page-12-4) [Jin et al.,](#page-11-9) [2018\)](#page-11-9). Tabular MDPs can be extended to linear MDPs [\(Jin et al.,](#page-11-6) [2020\)](#page-11-6), where computationally efficient algorithms are also known [\(Jin et al.,](#page-11-6) [2020;](#page-11-6) [Agarwal et al.,](#page-10-9) [2023;](#page-10-9) [He et al.,](#page-11-10) [2023\)](#page-11-10). However, in the setting of linear Bellman completeness, which captures linear MDPs, the existence of computationally efficient algorithms remain unclear. Previous works have resorted to various assumptions to achieve computational efficiency, such as few actions [\(Golowich & Moitra,](#page-10-10) [2024\)](#page-10-10) and assuming MDPs are "explorable" [\(Zanette et al.,](#page-12-5) [2020c\)](#page-12-5). We provide a detailed overview of the literature in Section [3.2.](#page-4-0)

Exploration via Randomization. Random noise has been a powerful alternative to bonus-based exploration in RL literature. A typical approach is Randomized Least-Squares Value Iteration (RLSVI) [\(Osband et al.,](#page-11-11) [2016\)](#page-11-11), which injects Gaussian noise into the least-squares estimate and achieves near-optimal worst-case regret for linear MDPs [\(Agrawal et al.,](#page-10-11) [2021;](#page-10-11) [Zanette et al.,](#page-12-6) [2020a\)](#page-12-6); [Ishfaq et al.](#page-11-12) [\(2023\)](#page-11-12) instead propose posterior sampling via Langevin Monte Carlo for Q-function and also obtain regret bounds for linear MDPs; [Ishfaq et al.](#page-11-13) [\(2021\)](#page-11-13) developed randomization algorithms for general function approximation assuming bounded eluder dimension and Bellman completeness for any function. Randomization is also explored in preference-based RL, leading to the first computationally efficient algorithm with near-optimal regret guarantees for linear MDPs [\(Wu & Sun,](#page-12-7) [2024\)](#page-12-7). However, these approaches either have strong assumptions (e.g., Bellman completeness for any function), or inject random noise larger than the estimation error, causing exponential blowup of parameter values—to mitigate it, they truncate the value, but this is feasible only in low-rank MDPs and challenging under linear Bellman completeness as the Bellman backup of truncated value may no longer be linear. Consequently, existing algorithms cannot handle linear Bellman complete problems, and new techniques capable of managing exponential parameter values are needed.

Beyond Linear Bellman Completeness. Many structural conditions capture linear Bellman completeness, such as Bilinear class [\(Du et al.,](#page-10-3) [2021\)](#page-10-3), Bellman eluder dimension [\(Jin et al.,](#page-11-5) [2021\)](#page-11-5), Bellman rank [\(Jiang et al.,](#page-11-2) [2017\)](#page-11-2), witness rank [\(Sun et al.,](#page-11-3) [2019\)](#page-11-3), and decision-estimation coefficient [\(Foster et al.,](#page-10-4) [2021\)](#page-10-4). While statistically efficient algorithms exist for these settings, no computationally efficient algorithms are known.

#### 3 PRELIMINARIES

A finite-horizon Markov Decision Process (MDP) is given by a tuple M = (S,A, H,T, r, µ) where S is the state space, A is the action space, <sup>H</sup> ∈ <sup>N</sup> is the horizon, <sup>T</sup> ∶ S × A → <sup>∆</sup>(S) is the transition function, <sup>r</sup> ∶ S × A → [0, <sup>1</sup>] is the reward function and <sup>µ</sup> ∈ <sup>∆</sup>(S) is the initial state distribution. Given a policy <sup>π</sup> ∶ S ↦ <sup>∆</sup>(A), we denote <sup>Q</sup><sup>π</sup> h (s, a) = <sup>E</sup><sup>π</sup> [∑ H <sup>i</sup>=<sup>h</sup> ri ∣ <sup>s</sup><sup>h</sup> = s, a<sup>h</sup> = <sup>a</sup>] as the layered state-action value function of policy π and V π h (s) = <sup>Q</sup><sup>π</sup> h (s, π(s)) as the state value function. The optimal value function is denoted by V ⋆ h (s) = max<sup>π</sup> <sup>V</sup> π h (s), and the optimal policy is <sup>π</sup> ⋆ .

We focus on the setting of linear function approximation and consider the following linear Bellman completeness, which ensures that the Bellman backup of a linear function remains linear.

Definition 1 (Linear Bellman Completeness). *An MDP is said to be linear Bellman complete with respect to a feature mapping* <sup>ϕ</sup> *if there exists a mapping* T ∶ <sup>R</sup> <sup>d</sup> → <sup>R</sup> d *so that, for all* <sup>θ</sup> ∈ <sup>R</sup> d *and all* (s, a) ∈ S × A*, it holds that*

$$\langle \mathcal{T}\theta, \phi(s, a) \rangle = \mathbb{E}_{s' \sim \mathcal{T}(s, a)} \max_{a'} \langle \theta, \phi(s', a') \rangle.$$

*Moreover, we require that, for all* <sup>h</sup> ∈ [H] *and* (s, a) ∈ S × A*, the random reward is bounded in* [0, <sup>1</sup>] *with mean* <sup>r</sup>h(s, a) = ⟨<sup>ω</sup> ⋆ h , ϕ(s, a)⟩ *for some unknown* <sup>ω</sup> ⋆ h ∈ R d *.*

We assume ∥ϕ(s, a)∥<sup>2</sup> ≤ <sup>1</sup> for all <sup>s</sup> ∈ S and <sup>a</sup> ∈ A. Notably, we do not impose any upper bound on ∥ω ⋆ h ∥<sup>2</sup> or any <sup>ℓ</sup>2-norm non-expansiveness of the Bellman backup, distinguishing us from some existing works—in Section [3.1,](#page-3-0) we discuss why many existing definitions of linear Bellman completeness fail to capture even tabular MDPs or linear MDPs due to certain ℓ2-norm boundedness assumptions.

We further assume the feature space spans R d , i.e., span({ϕ(s, a) ∶ <sup>s</sup> ∈ S, a ∈ A}) = <sup>R</sup> d ; otherwise, we can project the feature space onto its span or use pseudo-inverse in the analysis when needed. We can verify that the linear Bellman completeness captures both linear MDPs and Linear Quadratic Regulators (for a convex subset of linear functions). The proof is in Appendix [E.](#page-37-0)

Next, we consider deterministic state transition.

Assumption 1 (Deterministic transitions). *For all* <sup>s</sup> ∈ S *and* <sup>a</sup> ∈ A*, there is a unique state* <sup>s</sup> ′ ∈ S *to which the system transitions to after taking action* a *on state* s*.*

We emphasize that, although the transition is deterministic, the initial state distribution is stochastic (although we assume that {<sup>s</sup>t,1}t≤<sup>T</sup> is independently sampled from an initial distribution <sup>µ</sup>, our results extend to the scenarios when {<sup>s</sup>t,1}t≤<sup>T</sup> are adversarially chosen). Additionally, the reward signals can be stochastic. Hence, learning is still challenging in this case. The goal is to achieve low regret over T rounds. The regret is defined as

$$\text{Reg}_T := \mathbb{E} \left[ \sum_{t=1}^T (V^*(s_{t,1}) - V^{\pi_t}(s_{t,1})) \right].$$

The expectation here is taken over the randomness of algorithm and reward signals. While it is defined as an average for simplicity, a concentration inequality can yield the high-probability regret. In this paper, we use asymptotic notations <sup>Θ</sup>̃(⋅) and <sup>O</sup>̃(⋅) to hides logarithmic and constant factors.

#### 3.1 OTHER LINEAR BELLMAN COMPLETENESS DEFINITIONS IN THE LITERATURE

Several closely related definitions of Linear Bellman Completeness have been considered in the literature. In the following, we demonstrate that some of these variant definitions face limitations due to additional ℓ2-norm assumptions. We present two commonly imposed assumptions in existing works below, and subsequently provide examples illustrating their potential limitations.

(1) Assuming Bounded ℓ2-norm of Parameters. [Golowich & Moitra](#page-10-10) [\(2024\)](#page-10-10); [Zanette et al.](#page-12-1) [\(2020b](#page-12-1)[;c\)](#page-12-5) assume that any value function under consideration has its parameters bounded in ℓ2 norm, i.e., when we apply the Bellman backup, the resulting state-action value function always lies in {<sup>Q</sup> ∶ <sup>Q</sup>(s, a) = ⟨ϕ(s, a), θ⟩, ∥θ∥<sup>2</sup> ≤ <sup>R</sup>} where <sup>R</sup> is a pre-fixed polynomial in the dimension of the feature space. We will show that this assumption might not hold true since ∥θ∥<sup>2</sup> is unnecessarily bounded under linear Bellman completeness.

(2) Assuming Non-expansiveness of Bellman Backup in ℓ2-norm. [Song et al.](#page-11-14) [\(2022\)](#page-11-14) assume that, after applying the Bellman backup, the ℓ2-norm of the value function parameters will not increase, i.e., for any θ, they assume the existence of parameter θ ′ such that ∥<sup>θ</sup> ′ ∥<sup>2</sup> ≤ ∥θ∥<sup>2</sup> and ⟨ϕ(s, a), θ′ ⟩ = E<sup>s</sup> ′∼T(s,a) max<sup>a</sup>′⟨ϕ(<sup>s</sup> ′ , a′ ), θ⟩ for all s, a. This assumption is stronger than the previous one and does not hold even in tabular MDPs, as we will show in the second example below.

The following example demonstrates that the two assumptions above do not generally hold under linear Bellman completeness as the ℓ2-norm amplification can actually be arbitrarily large.

Example 1 (Arbitrarily Large ℓ2-norm on Parameters). *Consider a layered linear MDP with three states,* s1, s2, s3*, and a single action* a1*. Here* s<sup>1</sup> *is in the first layer and* s<sup>2</sup> *and* s<sup>3</sup> *are in the second layer. For some* <sup>ε</sup> *and* <sup>p</sup>*, we define* <sup>ϕ</sup>(<sup>s</sup>1, a1) = ( √ ε, √ <sup>p</sup> − <sup>ε</sup>)*,* <sup>µ</sup>(<sup>s</sup>2) = (p/ √ ε, <sup>0</sup>)*, and* <sup>µ</sup>(<sup>s</sup>3) <sup>=</sup> (0, (<sup>1</sup> <sup>−</sup> <sup>p</sup>)/√ <sup>p</sup> − <sup>ε</sup>)*. We further define* <sup>r</sup>(<sup>s</sup>2, ⋅) = <sup>ε</sup> *and* <sup>r</sup>(<sup>s</sup>3, ⋅) = <sup>1</sup>*. We can verify that* <sup>P</sup>(<sup>s</sup>2∣<sup>s</sup>1, a1) = <sup>p</sup> *and* <sup>P</sup>(<sup>s</sup>3∣<sup>s</sup>1, a1) = <sup>1</sup> − <sup>p</sup>*. Hence* <sup>Q</sup>(<sup>s</sup>1, a1) = pε + <sup>1</sup> − <sup>p</sup>*. We assume* <sup>Q</sup>*-function is parameterized by* <sup>θ</sup>*. Then, since* ∥ϕ(<sup>s</sup>1, a1)∥ = <sup>p</sup>*, it must hold that* ∥θ∥ ≥ (pε+1−p)/<sup>p</sup> = <sup>ε</sup>+<sup>p</sup> <sup>−</sup><sup>1</sup> −<sup>1</sup>*. While* p *can be arbitrarily small, the norm of* θ *can be arbitrarily large.*

We may hope to "normalize" the features in this example so that the ℓ2-norm of the parameters is bounded. However, it is unclear how to do so since changing either ε or p will change the MDP, and feature search is likely a hard problem. Essentially, this example breaks one of the assumptions in the original linear MDP [\(Jin et al.,](#page-11-6) [2020\)](#page-11-6) which requires the integral ∫ gµ to be bounded for any function <sup>g</sup> ∈ [0, <sup>1</sup>]. Thus, while being a linear MDP, the original LSVI-UCB algorithm [\(Jin et al.,](#page-11-6) [2020\)](#page-11-6) indeed will not work for this example. However, we note that our algorithm can still work.

Nevertheless, as the above example leverages a careful design of the feature, we might hope that some non-expansiveness properties could hold under stronger representation assumptions (e.g., when state space is tabular). Unfortunately, the following example shows that Bellman backup can be expansive even in tabular MDPs.

Example 2 (Expansiveness of Bellman Backup in ℓ2-norm). *Consider a tabular MDP with horizon* <sup>H</sup> = <sup>2</sup>*,* <sup>S</sup> *states* {<sup>s</sup>1, . . . , sS} *in the first layer, a single state* <sup>s</sup> *in the second layer, and a single action* a*. On taking action* a *in any state in the first-layer, the agent deterministically transitions to* s*, and on taking action* a *in* s *deterministically yields a reward of 1. Since linear Bellman completeness captures tabular MDPs with one-hot encoded features where* <sup>ϕ</sup>(<sup>s</sup><sup>i</sup> , a) = <sup>e</sup><sup>i</sup> ∈ <sup>R</sup> <sup>S</sup>+<sup>1</sup> *for* <sup>i</sup> ≤ <sup>S</sup> *and* <sup>ϕ</sup>(s, a) = <sup>e</sup>S+<sup>1</sup> = (0, . . . , <sup>0</sup>, <sup>1</sup>) ⊺ *, the state-action value function at the second layer can be parameterized by* <sup>θ</sup><sup>2</sup> = (0, . . . , <sup>0</sup>, <sup>1</sup>) ⊺ *. However, applying the Bellman backup, since the returnto-go for any first-layer state* s<sup>i</sup> *is 1 (because* s *always yields a reward of 1), the backed-up value function must be parameterized by* <sup>θ</sup><sup>1</sup> = (1, <sup>1</sup>, . . . , <sup>1</sup>) ⊺ *. Here, we find that* ∥<sup>θ</sup>1∥2/∥<sup>θ</sup>2∥<sup>2</sup> = √ S*, thus showing that Bellman backup cannot guarantee non-expansiveness of the* ℓ2*-norm.*

Hence, in this paper, we aim not to assume any ℓ2-norm bound or ℓ2-norm non-expansiveness of the parameters. Unfortunately, without these assumptions, the ground truth parameter of the optimal value function can exponentially grow with the horizon as evidenced by the examples above, thus invalidating prior methods requiring bounded parameter. Our key contribution is an algorithm that remains efficient even if the parameter norm blows up but requiring deterministic transition.

### 3.2 OTHER PRIOR WORKS ON LINEAR BELLMAN COMPLETENESS

In this section, we review prior efforts on RL under linear Bellman completeness and discuss various assumptions underlying these approaches.

Efficient Algorithms under Generative Access. A generative model takes as input a state-action pair (s, a) and returns a sample <sup>s</sup> ′ ∼ <sup>T</sup>(⋅ ∣ s, a) and the reward signal. With such a generative model, Linear Least-Squares Value Iteration (LSVI) can achieve statistical and computational efficiency [\(Agarwal et al.,](#page-10-6) [2019\)](#page-10-6). However, generative access is a big assumption, and our work aims to operate with only online access.

Efficient Algorithms under Explorability Assumption. [Zanette et al.](#page-12-5) [\(2020c\)](#page-12-5) propose a rewardfree algorithm under the assumption that every direction in the parameter space is reachable. This assumption, when translated into tabular MDPs, means that any state can be reached with a probability bounded below by some (large enough) positive constant. This does not hold if there are unreachable states or if the probability of reaching them is exponentially small.

Computationally Intractable Algorithms. [Zanette et al.](#page-12-1) [\(2020b\)](#page-12-1) present a computationally intractable algorithm that requires solving an intractable optimization problem. In our work, we aim to only utilize a tractable squared loss minimization oracle.

Few action MDPs. [Golowich & Moitra](#page-10-10) [\(2024\)](#page-10-10) propose a computationally efficient algorithm under linear Bellman completeness, inspired by the bonus-based exploration approach in LSVI-UCB [\(Jin](#page-11-6) [et al.,](#page-11-6) [2020\)](#page-11-6) for Linear MDPs. While their algorithm extends to stochastic MDPs, both the sample complexity and running time have exponential dependence on the size of the action space. In comparison, our algorithm extends to infinite action spaces but relies on the transition dynamics to be deterministic.

Deterministic Rewards or Deterministic Initial State. Several existing studies provide computationally and statistically efficient algorithms for more general settings but under stronger assumptions; these methods can be extended to linear Bellman completeness settings but similarly strong assumptions will also apply. [Du et al.](#page-10-5) [\(2020\)](#page-10-5) provide an algorithm based on a span argument that is efficient for MDPs that have linear optimal state-action value function (a.k.a. the Linear <sup>Q</sup>⋆ setting), deterministic transition dynamics, deterministic initial state, and stochastic rewards. Unfortunately, their approach cannot extend to settings with stochastic initial states, as we consider in our paper. Another line of work due to [Wen & Van Roy](#page-12-8) [\(2017\)](#page-12-8) considers the <sup>Q</sup>⋆ -realizable setting with deterministic dynamics, deterministic rewards, stochastic initial states, and bounded eluder dimension. Their approach can be extended to the linear bellman completeness setting when both rewards and dynamics are deterministic. However, their algorithm fails to converge when rewards are stochastic and thus may not apply to the problem setting that we consider.

Efficient Algorithm in the hybrid RL setting. [Song et al.](#page-11-14) [\(2022\)](#page-11-14) develop efficient algorithms for the hybrid RL setting, where the learner has access to both online interaction and an offline dataset. However, they do not have a fully online algorithms.

In summary, no previous work addresses the problem with stochastic initial states, stochastic rewards, and large action spaces. This is the gap that we aim to fill with this work.

## 4 ALGORITHM

In this section, we present our algorithm for online RL under linear Bellman completeness. See Algorithm [1](#page-5-0) for pseudocode. The input to the algorithm consists of three components. First, the noise variances, {<sup>σ</sup>h} H <sup>h</sup>=<sup>1</sup> and σR, control the scale of the random noise. Second, a D-optimal design (defined below) for the feature space.

Definition 2 (D-optimal design). *The D-optimal design for the set of features* <sup>Φ</sup> = {ϕ(s, a) ∶ <sup>s</sup> ∈ S, a ∈ A} *is a distribution* <sup>ρ</sup> *over* <sup>Φ</sup> *that maximizes* log det(∑ϕ∈<sup>Φ</sup> <sup>ρ</sup>(ϕ)ϕϕ<sup>⊺</sup> )*.*

There always exist D-optimal designs with at most <sup>O</sup>(<sup>d</sup> 2 ) support points (Lemma [23\)](#page-36-0). Many efficient algorithms can be applied to find approximate D-optimal designs such as the Frank-Wolfe. The algorithm also requires a constrained squared loss minimization oracle Osq, and we introduce an instantiation of Osq in Section [6.](#page-8-0)

Algorithm 1 Null Space Randomization for Linear Bellman Completeness

Require: • Noise variances {<sup>σ</sup>h} H and σR.

- <sup>h</sup>=<sup>1</sup>
- A D-optimal design for <sup>Φ</sup> = {ϕ(s, a) ∶ <sup>s</sup> ∈ S, a ∈ A} given by {(<sup>ϕ</sup><sup>i</sup> , ρi)}<sup>m</sup> <sup>i</sup>=<sup>1</sup>

- .
- Squared loss minimization oracle Osq . 1: Define <sup>Σ</sup>1,h ∶= ∑ m <sup>i</sup>=<sup>1</sup> ρiϕiϕ ⊺ i for all <sup>h</sup> ∈ [H]. 2: for <sup>t</sup> = <sup>1</sup>, . . . , T do 3: Let <sup>θ</sup>t,H+<sup>1</sup> ← <sup>0</sup>, <sup>Q</sup>t,H+<sup>1</sup> ← <sup>0</sup>, <sup>V</sup> t,H+<sup>1</sup> ← <sup>0</sup>. 4: for <sup>h</sup> = H, . . . , <sup>1</sup> do 5: Let <sup>P</sup>t,h be the orthogonal projection matrix onto span({ϕ(<sup>s</sup>i,h, ai,h) ∶ <sup>i</sup> = <sup>1</sup>, . . . , t − <sup>1</sup>}) 6: For <sup>i</sup> ∈ [m], define <sup>ϕ</sup> ∥ t,h,i = <sup>P</sup>t,hϕ<sup>i</sup> and <sup>ϕ</sup> ⊥ t,h,i = (<sup>I</sup> − <sup>P</sup>t,h)<sup>ϕ</sup><sup>i</sup> 7: Let <sup>Λ</sup>t,h ← ∑ m <sup>i</sup>=<sup>1</sup> <sup>ρ</sup>i(<sup>ϕ</sup> ∥ t,h,i(<sup>ϕ</sup> ∥ t,h,i) <sup>⊺</sup> + <sup>ϕ</sup> ⊥ t,h,i(<sup>ϕ</sup> ⊥ t,h,i) ⊺ ) 8: // Fit value function and reward using squared loss regression // 9: Compute <sup>θ</sup>̂t,h and <sup>ω</sup>̂t,h using the squared loss minimization oracle Osq as:

$$\widehat{\theta}_{t,h} \leftarrow \operatorname{argmin}_{\theta \in \mathcal{O}(W_h)} \sum_{i=1}^{t-1} \left( \langle \theta, \phi(s_{i,h}, a_{i,h}) \rangle - \overline{V}_{t,h+1}(s_{i,h+1}) \right)^2 \quad (1)$$

$$\widehat{\omega}_{t,h} \leftarrow \operatorname{argmin}_{\omega \in \mathcal{O}(1)} \sum_{i=1}^{t-1} \left( \langle \omega, \phi(s_{i,h}, a_{i,h}) \rangle - r_{i,h} \right)^2 \quad (2)$$

10: // Perturb the estimated parameters by adding Gaussian noise // 11: Update the parameters by sampling:

$$\bar{\theta}_{t,h} \sim \widehat{\theta}_{t,h} + \mathcal{N}\left(0, \sigma_h^2(I - P_{t,h})\Lambda_{t,h}^{-1}(I - P_{t,h})\right)$$

$$\overline{\omega}_{t,h} \sim \widehat{\omega}_{t,h} + \mathcal{N}(0, \sigma_R^2 \Sigma_{t,h}^{-1})$$

12: Define <sup>Q</sup>t,h(s, a) ← ⟨<sup>ω</sup>t,h + <sup>θ</sup>t,h, ϕ(s, a)⟩ and <sup>V</sup> t,h(s) ← max<sup>a</sup> <sup>Q</sup>t,h(s, a) for all (s, a) 13: end for 14: Define the policy <sup>π</sup><sup>t</sup> such that <sup>π</sup>t,h(s) = argmax<sup>a</sup> <sup>Q</sup>t,h(s, a) 15: Generate trajectory (<sup>s</sup>t,1, at,1, rt,1, . . . , st,H, at,H, rt,H) ∼ <sup>π</sup><sup>t</sup> 16: Define <sup>Σ</sup>t+1,h ∶= <sup>Σ</sup>t,h + <sup>ϕ</sup>(<sup>s</sup>t,h, at,h)<sup>ϕ</sup> ⊺ (<sup>s</sup>t,h, at,h) for all <sup>h</sup> ∈ [H] 17: end for

The algorithm begins by initializing the covariance matrix <sup>Σ</sup>1,h for all <sup>h</sup> ∈ [H] using the optimal design, which differs from most standard LSVI-type algorithms where it is initialized to the identity matrix. We believe that the identity matrix is unsuitable here since we do not assume any ℓ2-norm bound on the parameters. Additionally, recalling that we assume the feature space spans R d , it ensures Σt,h is invertible for all t and h. Otherwise, pseudo-inverses can be used instead.

At each round <sup>t</sup> ∈ [T], the algorithm operates in a backward manner starting from the last horizon <sup>H</sup>. For each <sup>h</sup> ∈ [H], it first constructs the orthogonal projection matrix <sup>P</sup>t,h onto the span of the historical data. It then decomposes the D-optimal design points into the span and null space components using the projection and constructs Λt,h. By separating the span and null space components, it facilitates clearer concentration bounds for the subsequent Gaussian noise.

The algorithm then performs constrained squared loss regression to estimate the value function and reward function. Here we define <sup>O</sup>(W) ∶= {<sup>θ</sup> ∈ <sup>R</sup> d ∶ ∣⟨θ, ϕ(s, a)⟩∣ ≤ <sup>W</sup> for all <sup>s</sup> ∈ S, a ∈ A} for any <sup>W</sup> > <sup>0</sup>. This *convex* constrained set is defined by the <sup>ℓ</sup>∞-functional-norm bound instead of the ℓ2-norm because we do not assume any bound on the ℓ2-norm of the learned parameters. Here we define <sup>W</sup><sup>h</sup> = <sup>Θ</sup>̃((<sup>d</sup> √ mH) <sup>H</sup>−<sup>h</sup> (d <sup>3</sup>/<sup>2</sup> + <sup>d</sup> √ mH)) (detailed definition deferred to Appendix [C\)](#page-17-0). We note that although W<sup>h</sup> appears exponential, which may seem suspicious, this does not affect our sample efficiency due to the span argument that we introduce in the analysis. We note that prior RLSVI algorithms used truncation on value functions to explicitly avoid such an exponential blow-up. However, truncation does not work for linear Bellman completeness setting since the Bellman backup on a truncated value function is not necessarily a linear function anymore.

Next, the algorithm perturbs the estimated parameters by adding Gaussian noise. The noise for the value function act *only in the null space* of the data covariance matrix. This ensures optimism while keeping the estimate accurate in the span space. It is a key modification from the standard RLSVI algorithm. The perturbation for the reward function is standard. Finally, the algorithm constructs the value function for the current horizon and the greedy policy with respect to it. It then generates the trajectory by executing the greedy policy, and the covariance matrix is updated.

#### 5 ANALYSIS

In this section, we provide the theoretical guarantees of Algorithm [1.](#page-5-0) A high-level proof sketch can be found in Appendix [B](#page-16-0) and detailed proofs are in Appendix [C.](#page-17-0) We first consider the case where the squared loss minimization oracle is exact. We then extend the analysis to the approximate oracle and the low inherent linear Bellman error setting in subsequent sections.

#### 5.1 PRELUDE: LEARNING WITH EXACT SQUARE LOSS MINIMIZATION ORACLE

We first consider the most ideal setting where the squared loss minimization oracle is exact.

Assumption 2 (Exact Squared Loss Minimization Oracle). *Line [9](#page-5-1) of Algorithm [1](#page-5-0) is solved exactly.*

Then, we have the following regret bound. A proof sketch is provided in Appendix [B](#page-16-0) for the readers convenience.

Theorem 1 (Regret Bound with Exact Oracle). *Under Assumptions [1](#page-3-1) and [2,](#page-6-0) executing Algorithm [1](#page-5-0) with parameters* <sup>σ</sup><sup>R</sup> = <sup>Θ</sup>̃( √ dH log(HT)) *and* <sup>σ</sup><sup>h</sup> = <sup>Θ</sup>̃((<sup>d</sup> √ mH) <sup>H</sup>−h+<sup>1</sup> ( √ d + √ mH))*, we have*

$$\text{Reg}_T = \tilde{O}(d^{5/2} H^{5/2} + d^2 H^{3/2} \sqrt{T}).$$

This result has several notable features. First, it does not depend on the number of actions. The only requirement for the action space is the ability to compute the argmax. Second, the √ T-dependence on T is optimal, as it is necessary even in the bandit setting. Additionally, we emphasize that the dependence on √ T arises solely from reward learning due to the application of elliptical potential lemma. In fact, if the reward function is known, our regret bound can be as small as <sup>O</sup>̃(dH<sup>2</sup> ), depending on T up to logarithmic factors. We elaborate on this observation in Appendix [B.](#page-16-0) As a standard practice, Theorem [1](#page-6-1) can be converted into a sample complexity bound below.

Corollary 1 (Sample Complexity Bound). *Let* <sup>ε</sup> ≤ <sup>1</sup>*. Under the same setting as Theorem [1,](#page-6-1) letting* <sup>T</sup> ≥ <sup>Ω</sup>(<sup>d</sup> <sup>4</sup>H<sup>3</sup> /ε 2 )*, we get that the policy* ̂<sup>π</sup> *chosen uniformly from the set* <sup>π</sup>1, . . . , π<sup>T</sup> *enjoys performance guarantee* <sup>E</sup>[<sup>V</sup> <sup>⋆</sup> − <sup>V</sup> π̂ ] ≤ <sup>ε</sup>*.*

#### 5.2 LEARNING WITH APPROXIMATE SQUARE LOSS MINIMIZATION ORACLE

Assumption 3 (Approximate Squared Loss Minimization Oracle). *We assume access to an approximate squared loss minimization oracle* Osq apx *that takes as input a problem of the form:* argminθ∈O(W) <sup>g</sup>(θ) ∶= ∑(ϕ(s,a),u)∈D(⟨θ, ϕ(s, a)⟩ − <sup>u</sup>) <sup>2</sup> *where* O(W) = {<sup>θ</sup> ∈ <sup>R</sup> d ∣ ∣⟨θ, ϕ(s, a)⟩∣ ≤ <sup>W</sup>} *for some* <sup>W</sup> ∈ <sup>R</sup> *is a convex set, and* D *is a dataset of tuples* {(ϕ(s, a), u)}*. The oracle returns a point* <sup>θ</sup>̂*that satisfies* <sup>g</sup>(θ̂)− minθ∈O(W) <sup>g</sup>(θ) ≤ <sup>ε</sup> 2 <sup>1</sup> *and* <sup>θ</sup>̂∈ <sup>O</sup>(<sup>W</sup> +<sup>ε</sup>2) *where* <sup>ε</sup>1, ε<sup>2</sup> ≤ <sup>1</sup> *are precision parameters of the oracle.*

With an approximate oracle, the regret bound depends on an additional quantity defined below.

Assumption 4. *There exists a constant* <sup>γ</sup> > <sup>1</sup> *such that, for any* <sup>r</sup> ≤ <sup>d</sup>*, and for any* <sup>ϕ</sup>1, ϕ2, . . . , ϕ<sup>r</sup> ∈ <sup>Φ</sup>*, the eigenvalues of the matrix* <sup>Σ</sup> ∶= ∑ r <sup>i</sup>=<sup>1</sup> ϕiϕ ⊺ i *are either zero or at least* <sup>1</sup>/<sup>γ</sup> 2 *.*

As a concrete example, it holds with <sup>γ</sup> = <sup>1</sup> when the MDP is tabular. This assumption implies that the eigenvalues of Σ † are at most γ 2 . Consequently, for any vector <sup>ϕ</sup> ∈ <sup>Φ</sup>, we have ∥ϕ∥Σ† ≤ ∥ϕ∥2<sup>γ</sup> ≤ <sup>γ</sup>—this lower bound on the norm of any vector is exactly what we need for the analysis of an approximate oracle, while Assumption [4](#page-7-0) simply serves as a sufficient condition for it. The following theorem provides the regret bound with the approximate oracle in terms of parameters ε1, ε<sup>2</sup> and γ.

Theorem 2 (Regret Bound with Approximate Oracle). *Under Assumptions [1,](#page-3-1) 3 and [4,](#page-7-0) executing Algorithm [<sup>1</sup>](#page-5-0) with* <sup>σ</sup><sup>R</sup> = <sup>Θ</sup>̃( √ dH) *and* <sup>σ</sup><sup>h</sup> = <sup>Θ</sup>̃((<sup>d</sup> √ mH) <sup>H</sup>−h+<sup>1</sup> (<sup>ε</sup>1<sup>γ</sup> √ <sup>H</sup> + √ d + [√](#page-7-1) mH)*, we have*

$$\text{Reg}_T = \tilde{O}(d^{5/2} H^{5/2} + d^2 H^{3/2} \sqrt{T} + \varepsilon_1 \gamma(d H^2 + d^{3/2} H \sqrt{T})).$$

Compared to Theorem [1,](#page-6-1) the regret bound has an additional term that depends on the approximation error ε1γ. Typically, ε<sup>1</sup> is from optimization and thus can be exponentially small with respect to the relevant parameters, as we later discuss in Section [6.](#page-8-0) Hence, we allow γ to be exponentially large. Moreover, we note that ε<sup>2</sup> does not appear in the regret bound since it only affects the constraint violation of the regression, whose effect to the statistical guarantees is of lower order and thus ignored. In addition, we note that the regret bound does not depend on the number of actions, and the dependence on T remains optimal, similar to the previous theorem.

#### 5.3 LEARNING WITH LOW INHERENT LINEAR BELLMAN ERROR

Now we consider the setting where the MDP has low inherent linear Bellman error.

Definition 3 (Inherent Linear Bellman Error). *Given* <sup>ε</sup><sup>B</sup> ≤ <sup>1</sup>*, an MDP* M *is said to have* <sup>ε</sup>B*inherent linear Bellman error with respect to a feature mapping* ϕ *if there exists a mapping* T ∶ <sup>R</sup> <sup>d</sup> → <sup>R</sup> d *so that, for all* <sup>θ</sup> ∈ <sup>R</sup> d *and all* (s, a) ∈ S × A*, it holds that* ∣⟨T θ, ϕ(s, a)⟩ − E<sup>s</sup> ′∼T(s,a) max<sup>a</sup>′⟨θ, ϕ(<sup>s</sup> ′ , a′ )⟩∣ ≤ <sup>ε</sup>B*. Moreover, we require that, for all* <sup>h</sup> ∈ [H] *and* (s, a) ∈ S × A*, the random reward is bounded in* [0, <sup>1</sup>] *with* ∣<sup>r</sup>h(s, a) − ⟨<sup>ω</sup> ⋆ h , ϕ(s, a)⟩∣ ≤ <sup>ε</sup><sup>B</sup> *for some unknown* ω ⋆ h ∈ R d *.*

With low inherent Bellman error, Assumption [4](#page-7-0) is still necessary. The following theorem provides the regret bound in this case. We assume the exact oracle for simplicity.

Theorem 3 (Regret Bound with Low Inherent Bellman Error). *Assume the MDP has* εB*-inherent Bellman error. Under Assumptions [1,](#page-3-1) [<sup>2</sup>](#page-6-0) and [4,](#page-7-0) when executing Algorithm [<sup>1</sup>](#page-5-0) with parameters* <sup>σ</sup><sup>R</sup> = Θ̃( √ dH + <sup>ε</sup>BHT) *and* <sup>σ</sup><sup>h</sup> = <sup>Θ</sup>̃((<sup>d</sup> √ mH) <sup>H</sup>−h+<sup>1</sup> (<sup>ε</sup>B<sup>γ</sup> √ HT + √ <sup>ε</sup>B<sup>T</sup> + √ d + √ mH))*, we have*

$$\text{Reg}_T = \tilde{O}\left(d^{5/2}H^{5/2} + d^2H^{3/2}\sqrt{T} + \sqrt{\varepsilon_B}(d^2H^{5/2}\sqrt{T} + d^{3/2}H^{3/2}T) + \varepsilon_B\gamma(dH^2\sqrt{T} + d^{3/2}HT)\right).$$

Compared to Theorem [1,](#page-6-1) the regret bound includes two additional terms that depend on the inherent linear Bellman error εB. For both terms, the dependence on T is linear. We believe the T-dependence is unavoidable, as it also appears in similar settings [\(Zanette et al.,](#page-12-1) [2020b\)](#page-12-1). In addition, it is worth noting that the regret bound does not depend on the number of actions, and the other dependence on T remains optimal, similar to previous theorems.

## 6 OPENING THE BLACK-BOX: IMPLEMENTING SQUARED LOSS MINIMIZATION ORACLES IN A[LGORITHM](#page-5-0) 1

In this section, we detail a practical implementation of the desired squared loss oracle need by our algorithm. The implementation relies on the observation that a square loss minimization objective over a convex domain can be cast as a convex set feasibility problem—given a convex set K, return a point <sup>θ</sup>̂∈ K. Thus, we can use algorithms for convex set feasibility to implement the squared loss minimization oracles. However, even given this observation, our key challenge for an efficient algorithm is that the corresponding convex set could be exponentially large and only be described using exponentially many number of linear constraints. Fortunately, various works in the optimization literature propose computationally efficient procedures to find feasible points within such ill-defined sets, under mild oracle assumptions.

#### 6.1 COMPUTATIONALLY EFFICIENT CONVEX SET FEASIBILITY

We first paraphrase the work of [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7) that provide a computationally efficient procedure for finding feasible points within a convex set by random walks. Notably, the computational complexity of their algorithm only depends logarithmically on the size of the convex set, and thus their approach is well suited for the corresponding convex feasibility problems that appear in our approach. At a high level, they provide an algorithm that takes an input an arbitrary convex set K ⊆ <sup>R</sup> d , and returns a feasible point <sup>z</sup>̂ ∈ K. Their algorithm accesses the convex set K via a separation oracle defined as follows.

Definition 4 (Separation oracle). *A separation oracle for a convex set* K*, denoted by* O sep K *, is defined such that on any input* <sup>z</sup> ∈ <sup>R</sup> d *, the oracle either confirms that* <sup>z</sup> ∈ K *or returns a hyperplane* ⟨a, z⟩ ≤ <sup>b</sup> *that separates the point* <sup>z</sup> *from the set* K*.*

In order to ensure finite time convergence for their procedure, they assume that the convex set K is not degenerate and is bounded in any direction. This is formalized by the following assumption.

Assumption 5. *The convex set* K *is* (r, R)−*Bounded, i.e. there exist parameters* <sup>0</sup> < <sup>r</sup> ≤ <sup>R</sup> *such that (a)* K ⊆ <sup>R</sup>∞(R)*, and (b) there exists a vector* <sup>z</sup> ∈ <sup>R</sup> d *such that the shifted cube* (<sup>z</sup> + <sup>R</sup>∞(r)) ⊆ K*.*

The computational efficiency and the convergence guarantee of their algorithm are below.

Theorem 4 [\(Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7)). *Let* <sup>δ</sup> ∈ (0, <sup>1</sup>) *and* K ⊂ <sup>R</sup> d *be an arbitrary convex set that satisfies Assumption [<sup>5</sup>](#page-8-1) for some* <sup>0</sup> ≤ <sup>r</sup> ≤ <sup>R</sup>*. Then, Algorithm [<sup>2</sup>](#page-39-0) (given in the appendix), when invoked with the separation oracle* O sep <sup>K</sup> *w.r.t.* K*, returns a feasible point* <sup>z</sup>̂ ∈ K *with probability at least* <sup>1</sup> − <sup>δ</sup>*. Moreover, Algorithm [<sup>2</sup>](#page-39-0) makes* <sup>O</sup>(<sup>d</sup> log(R/δr)) *calls to the oracle* O sep K *and runs in time* <sup>O</sup>(<sup>d</sup> 7 log(R/δr))*.*

Notice that both the number of oracle calls and the running time only depend logarithmically on R and <sup>r</sup>, and thus their procedure can be efficiently implemented for our applications where <sup>R</sup>/<sup>r</sup> may be exponentially large in the corresponding problem parameters.

#### 6.2 COMPUTATIONALLY EFFICIENT ESTIMATION OF VALUE FUNCTION (EQN [\(1\)\)](#page-5-2)

We now described how to leverage the method by [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7) to estimate the parameters for the value functions in [\(1\)](#page-5-2) in [Algorithm 1.](#page-5-0) Note that for any time t and horizon <sup>h</sup> ∈ [H], the objective in [\(1\)](#page-5-2) is the optimization problem

$$\widehat{\theta}_{t,h} \leftarrow \operatorname{argmin}_{\theta \in \mathcal{O}(W_h)} \sum_{i=1}^{t-1} \left( \langle \theta, \phi(s_{i,h}, a_{i,h}) \rangle - \overline{V}_{t,h+1}(s_{i,h+1}) \right)^2, \quad (3)$$

where <sup>W</sup><sup>h</sup> = <sup>Θ</sup>̃((<sup>d</sup> √ mH) <sup>H</sup>−<sup>h</sup> (<sup>ε</sup>1dγ√ <sup>H</sup> +<sup>d</sup> <sup>3</sup>/<sup>2</sup> +<sup>d</sup> √ mH)). We provide a computationally efficient procedure to approximately solve the above given a linear optimization oracle over the feature space.

Assumption 6 (Linear optimization oracle over the feature space). *Learner has access to a linear optimization oracle* Olin *that on taking input* <sup>θ</sup> ∈ <sup>R</sup> d *, returns a feature* <sup>ϕ</sup>(<sup>s</sup> ′ , a′ ) ∈ argmaxs,a⟨θ, ϕ(s, a)⟩*.*

The key observation we use is that under linear Bellman completeness (Definition [1\)](#page-2-0) and deterministic dynamics (Assumption [1\)](#page-3-1), any solution <sup>θ</sup> for [\(3\)](#page-8-2) must satisfy ∑ <sup>t</sup>−<sup>1</sup> <sup>i</sup>=<sup>1</sup> (⟨θ, ϕ(<sup>s</sup>i,h, ai,h)⟩ − <sup>V</sup> t,h+1(<sup>s</sup>i,h+1))<sup>2</sup> = <sup>0</sup>. On the other hand, the converse also holds that any point <sup>θ</sup> ∈ O(<sup>W</sup>h) for which the objective value is 0 must be a solution to [\(3\).](#page-8-2) Thus, the minimization problem in [\(3\)](#page-8-2) is equivalent to finding a feasible point within the convex set

$$\mathcal{K} := \left\{ \theta \in \mathbb{R}^d \mid \frac{(\langle \theta, \phi(s_{i,h}, a_{i,h}) \rangle - \overline{V}_{t,h+1}(s_{i,h+1}))^2}{|\langle \theta, \phi(s, a) \rangle| \leq W_h \text{ for all } s, a} = 0 \text{ for all } i \leq t \right\}. \quad (4)$$

Given the above reformulation of the optimization objective [\(3\)](#page-8-2) as a feasibility problem, we can now use the procedure of [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7) for finding <sup>θ</sup>t,h ∈ K. However, we first need to define a separation oracle for the set K and verify Assumption [5.](#page-8-1) Unfortunately, there may not exist any <sup>r</sup> > <sup>0</sup> for which (<sup>z</sup> + <sup>R</sup>∞(r)) ⊆ K for some <sup>z</sup> ∈ <sup>R</sup> d in our case and thus the above K may not satisfy Assumption [5.](#page-8-1) This can, however, be easily fixed by artificially increasing the set K to allow for some approximation errors. In particular, let <sup>ε</sup> > <sup>0</sup> and define the convex set

$$\mathcal{K}_{\text{APX}} := \left\{ \theta \in \mathbb{R}^d \mid \begin{array}{l} \langle \theta, \phi(s_{i,h}, a_{i,h}) \rangle - \overline{V}_{t,h+1}(s_{i,h+1}) \leq \varepsilon \text{ for all } i \leq t \\ \langle \theta, \phi(s_{i,h}, a_{i,h}) \rangle - \overline{V}_{t,h+1}(s_{i,h+1}) \geq -\varepsilon \text{ for all } i \leq t \\ |\langle \theta, \phi(s, a) \rangle| \leq W_h + \varepsilon \text{ for all } s, a \end{array} \right\}. \quad (5)$$

Clearly, since there exists at least one point <sup>θ</sup>t,h ∈ K, we must have that (<sup>θ</sup>t,h + <sup>R</sup>∞(ε)) ⊆ KAPX. To ensure an outer bounding box for the set KAPX, we need to make an additional assumption.

Assumption 7. *Let* <sup>Φ</sup> = {ϕ(s, a) ∣ s, a ∈ S × A}*. There exist some* <sup>R</sup> ≥ <sup>0</sup> *such that* <sup>1</sup> R <sup>e</sup><sup>i</sup> ∈ <sup>Φ</sup>*, where* e<sup>i</sup> *denotes the unit basis vector along the* i*-th direction in* <sup>R</sup> d

The above assumption ensures that K ⊆ <sup>B</sup>∞(<sup>W</sup>h<sup>R</sup>). Recall that we can tolerate the parameter <sup>R</sup> to be exponential in the dimension d or the horizon H. Finally, a separation oracle can be implemented using Olin (see Algorithm [<sup>4</sup>](#page-40-0) for details). Thus, one can use Algorithm [<sup>2</sup>](#page-39-0) (given in appendix), due to [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7), and the guarantee in Theorem [<sup>4</sup>](#page-8-3) to find a feasible point in KAPX, which corresponds to an approximate solution to [\(3\).](#page-8-2)

Theorem 5. *Let* <sup>ε</sup> > <sup>0</sup>*,* <sup>δ</sup> ∈ (0, <sup>1</sup>)*, and suppose Assumption [<sup>7</sup>](#page-9-0) holds with some parameter* <sup>R</sup> > <sup>0</sup>*. Additionally, suppose Assumption [<sup>6</sup>](#page-8-4) holds with the linear optimization oracle denoted by* Olin*. Then, there exists a computationally efficient procedure (given in Algorithm [4](#page-40-0) in the appendix), that for any* <sup>t</sup> ∈ [T] *and* <sup>h</sup> ∈ [H]*, returns a point* <sup>θ</sup>̂t,h *that, with probability at least* <sup>1</sup> − <sup>δ</sup>*, satisfies*

$$\sum_{i=1}^{t-1} \left( (\widehat{\theta}_{t,h}, \phi(s_{i,h}, a_{i,h})) - \overline{V}_{t,h+1}(s_{i,h+1}) \right)^2 \leq T\varepsilon \quad \text{and} \quad \widehat{\theta}_{t,h} \in \mathcal{O}(W_h + \varepsilon).$$

*Furthermore, Algorithm [<sup>4</sup>](#page-40-0) takes* <sup>O</sup>(<sup>d</sup> 7 log( R δε )) *time in addition to* <sup>O</sup>(<sup>d</sup> log( THR δε )) *calls to* Olin

The above techniques and Algorithm [4](#page-40-0) can be similarly extended to get a computationally efficient procedure to estimate the reward parameter in [\(2\).](#page-5-3) The main difference is that the value of the optimization objective in [\(2\)](#page-5-3) is not zero at the minimizer (due to stochasticity). Thus, we need to construct a set feasibility problem for every desired target value of the objective function within the grid [0, ε, <sup>2</sup>ε, . . . , <sup>2</sup> − ε, <sup>2</sup>] and use a separating hyperplane w.r.t. the ellipsoid constraint in [\(2\)](#page-5-3) to implement the separating hyperplane for KAPX (which can be implemented using projections).

## 7 CONCLUSION

In this paper, we develop a computationally efficient RL algorithm under linear Bellman completeness with deterministic dynamics, aiming to bridge the statistical-computational gap in this setting. Our algorithm injects random noise into regression estimates only in the null space to ensure optimism and leverages a span argument to bound regret. It handles large action spaces, random initial states, and stochastic rewards. Our key observation is that deterministic dynamics simplifies the learning process by ensuring accurate value estimates within the data span, allowing noise injection to be confined to the null space. Extending our algorithm to stochastic dynamics remains an open challenge.

## ACKNOWLEDGMENTS

We thank Yuda Song, Zeyu Jia, Noah Golowich, and Sasha Rakhlin for useful discussions. AS acknowledges support from the Simons Foundation and NSF through award DMS-2031883, as well as from the DOE through award DE-SC0022199. WS acknowledges support from NSF IIS-2154711, NSF CAREER 2339395, and DARPA LANCER: LeArning Network CybERagents.

## REFERENCES


[1] Marc Abeille and Alessandro Lazaric. Linear thompson sampling revisited. In *Artificial Intelligence and Statistics*, pp. 176–184. PMLR, 2017. Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. *CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep*, 32:96, 2019. Alekh Agarwal, Yujia Jin, and Tong Zhang. Vo q l: Towards optimal regret in model-free rl with nonlinear function approximation. In *The Thirty Sixth Annual Conference on Learning Theory*, pp. 987–1063. PMLR, 2023. Priyank Agrawal, Jinglin Chen, and Nan Jiang. Improved worst-case regret bounds for randomized least-squares value iteration. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pp. 6566–6573, 2021. OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. *The International Journal of Robotics Research*, 39(1):3–20, 2020. Mohammad Gheshlaghi Azar, Ian Osband, and Remi Munos. Minimax regret bounds for reinforce- ´ ment learning. In *International conference on machine learning*, pp. 263–272. PMLR, 2017. Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. *arXiv preprint arXiv:1912.06680*, 2019. Dimitris Bertsimas and Santosh Vempala. Solving convex programs by random walks. *Journal of the ACM (JACM)*, 51(4):540–556, 2004. Rajendra Bhatia. *Matrix analysis*, volume 169. Springer Science & Business Media, 2013. Jianyu Chen, Bodi Yuan, and Masayoshi Tomizuka. Model-free deep reinforcement learning for urban autonomous driving. In *2019 IEEE intelligent transportation systems conference (ITSC)*, pp. 2765–2771. IEEE, 2019. Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in rl. In *International Conference on Machine Learning*, pp. 2826–2836. PMLR, 2021. Simon S Du, Jason D Lee, Gaurav Mahajan, and Ruosong Wang. Agnostic q-learning with function approximation in deterministic systems: Tight bounds on approximation error and sample complexity. *arXiv preprint arXiv:2002.07125*, 2020. Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. *arXiv preprint arXiv:2112.13487*, 2021. Noah Golowich and Ankur Moitra. Linear bellman completeness suffices for efficient online reinforcement learning with few actions. In *The Thirty Seventh Annual Conference on Learning Theory*. PMLR, 2024. David Haussler. Decision theoretic generalizations of the pac model for neural net and other learning applications. In *The mathematics of generalization*, pp. 37–116. CRC Press, 2018.

[2] Jiafan He, Heyang Zhao, Dongruo Zhou, and Quanquan Gu. Nearly minimax optimal reinforcement learning for linear markov decision processes. In *International Conference on Machine Learning*, pp. 12790–12822. PMLR, 2023. Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Zhuoran Yang, Zhaoran Wang, Doina Precup, and Lin Yang. Randomized exploration in reinforcement learning with general value function approximation. In *International Conference on Machine Learning*, pp. 4607–4616. PMLR, 2021. Haque Ishfaq, Qingfeng Lan, Pan Xu, A Rupam Mahmood, Doina Precup, Anima Anandkumar, and Kamyar Azizzadenesheli. Provable and practical: Efficient exploration in reinforcement learning via langevin monte carlo. *arXiv preprint arXiv:2305.18246*, 2023. Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In *International Conference on Machine Learning*, pp. 1704–1713. PMLR, 2017. Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? *Advances in neural information processing systems*, 31, 2018. Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In *Conference on learning theory*, pp. 2137–2143. PMLR, 2020. Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. *Advances in neural information processing systems*, 34:13406–13418, 2021. Tor Lattimore and Csaba Szepesvari. ´ *Bandit algorithms*. Cambridge University Press, 2020. Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In *Proceedings of the 19th international conference on World wide web*, pp. 661–670, 2010. Aditya Modi, Jinglin Chen, Akshay Krishnamurthy, Nan Jiang, and Alekh Agarwal. Model-free representation learning and exploration in low-rank mdps. *Journal of Machine Learning Research*, 25(6):1–76, 2024. Remi Munos. Error bounds for approximate value iteration. In ´ *Proceedings of the National Conference on Artificial Intelligence*, volume 20, pp. 1006. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2005. Ian Osband, Benjamin Van Roy, and Zheng Wen. Generalization and exploration via randomized value functions. In *International Conference on Machine Learning*, pp. 2377–2386. PMLR, 2016. Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. *Advances in Neural Information Processing Systems*, 26, 2013. David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016. Yuda Song, Yifei Zhou, Ayush Sekhari, J Andrew Bagnell, Akshay Krishnamurthy, and Wen Sun. Hybrid rl: Using both offline and online data can make rl efficient. *arXiv preprint arXiv:2210.06718*, 2022. Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In *Conference on learning theory*, pp. 2898–2933. PMLR, 2019. Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. *Advances in Neural Information Processing Systems*, 33:6123–6135, 2020.

[3] Yuanhao Wang, Ruosong Wang, and Sham Kakade. An exponential lower bound for linearly realizable mdp with constant suboptimality gap. *Advances in Neural Information Processing Systems*, 34:9521–9533, 2021. Gellert Weisz, Philip Amortila, and Csaba Szepesv ´ ari. Exponential lower bounds for planning in ´ mdps with linearly-realizable optimal action-value functions. In *Algorithmic Learning Theory*, pp. 1237–1264. PMLR, 2021. Zheng Wen and Benjamin Van Roy. Efficient reinforcement learning in deterministic systems with value function generalization. *Mathematics of Operations Research*, 42(3):762–782, 2017. Runzhe Wu and Wen Sun. Making rl with preference-based feedback efficient via randomization. In *The Twelfth International Conference on Learning Representations*, 2024. Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. *arXiv preprint arXiv:2210.04157*, 2022. Andrea Zanette, David Brandfonbrener, Emma Brunskill, Matteo Pirotta, and Alessandro Lazaric. Frequentist regret bounds for randomized least-squares value iteration. In *International Conference on Artificial Intelligence and Statistics*, pp. 1954–1964. PMLR, 2020a. Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent bellman error. In *International Conference on Machine Learning*, pp. 10978–10989. PMLR, 2020b. Andrea Zanette, Alessandro Lazaric, Mykel J Kochenderfer, and Emma Brunskill. Provably efficient reward-agnostic navigation with linear value iteration. *Advances in Neural Information Processing Systems*, 33:11756–11766, 2020c. Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learningvia reference-advantage decomposition. *Advances in Neural Information Processing Systems*, 33: 15198–15207, 2020. Yinglun Zhu and Robert Nowak. Efficient active learning with abstention. *arXiv preprint arXiv:2204.00043*, 2022.
# CONTENTS OF APPENDIX

| 1 | Introduction                                                                   | 1  |
|---|--------------------------------------------------------------------------------|----|
| 2 | Related Works                                                                  | 3  |
| 3 | Preliminaries                                                                  | 3  |
|   | 3.1 Other Linear Bellman Completeness Definitions in the Literature            | 4  |
|   | 3.2 Other Prior Works on Linear Bellman Completeness                           | 5  |
| 4 | Algorithm                                                                      | 6  |
| 5 | Analysis                                                                       | 7  |
|   | 5.1 Prelude: Learning with Exact Square Loss Minimization Oracle               | 7  |
|   | 5.2 Learning with Approximate Square Loss Minimization Oracle                  | 8  |
|   | 5.3 Learning with Low Inherent Linear Bellman Error                            | 8  |
| 6 | Opening the Black-Box: Implementing Squared Loss Minimization Oracles in Algo |    |
|   | rithm 1                                                                        | 9  |
|   | 6.1 Computationally Efficient Convex Set Feasibility                           | 9  |
|   | 6.2 Computationally Efficient Estimation of Value Function (Eqn (1))           | 9  |
| 7 | Conclusion                                                                     | 10 |
| A | Table of Notation                                                              | 16 |
| B | Proof Overview                                                                 | 17 |
|   | B.1 Span Argument                                                              | 17 |
|   | B.2 Exploration in the Null Space                                              | 17 |
|   | B.3 Proof Outline                                                              | 18 |
| C | Full Proof for Section 5                                                       | 18 |
|   | C.1 High-probability Event and Boundedness                                     | 19 |
|   | C.2 Value Decomposition                                                        | 24 |
|   | C.3 Exploration in the Null Space                                              | 27 |
|   | C.4 Main Steps of the Proof                                                    | 30 |
| D | Supporting Lemmas                                                              | 35 |
|   | D.1 Pseudo Dimension and Covering Number                                       | 38 |
| E | Linear MDPs and LQRs imply Linear Bellman Completeness                         | 38 |
| F | Computationally Efficient Implementations for Optimization Oracles             | 39 |

[G Missing Details from](#page-38-1) [Section 6.2](#page-8-6) 39 [G.1 Computationally Efficient Estimation of Reward Function \(Eqn.](#page-38-2) [2\)](#page-5-3) . . . . . . . . . . 39

# A TABLE OF NOTATION

We list the notation used in this paper in table [1,](#page-15-1) for the convenience of reference.

Table 1: Notation used in the paper.

|               |             |      |             |       |            | Table 1: Notation used in the paper.                             |
|---------------|-------------|------|-------------|-------|------------|------------------------------------------------------------------|
| Symbol        | Description |      |             |       |            |                                                                  |
| O ( W ) { θ   | ∈ R         |      |             |       |            |                                                                  |
|               |             |      | ∶ ∣⟨        | θ, ϕ  | ( s,       | a )⟩∣ ≤ W for all s ∈ S , a ∈ A}                                 |
| R ∞ ( W ) { θ | ∈ R         |      |             |       |            |                                                                  |
|               |             |      | ∶ ∥ θ       | ∥ ∞   | ≤          | W }                                                              |
| R 2 ( W ) { θ | ∈ R         |      |             |       |            |                                                                  |
|               |             |      | ∶ ∥ θ       | ∥ 2   | ≤          | W }                                                              |
| η t,h T       | ( ω t,h     | +    | 1 +         | θ t,h | +          | 1 ) − θ ̂ t,h                                                    |
| t,h ω         |             |      |             |       |            |                                                                  |
| h             | − ω ̂       | t,h  |             |       |            |                                                                  |
| t ω           | t,h −       | ω ̂  | t,h         |       |            |                                                                  |
| t,h θ t,h     | −           | θ ̂  | t,h         |       |            |                                                                  |
| high High     |             |      | probability |       |            | event, defined in Definition 5                                   |
| t             | Event       | that |             |       | trajectory | at round t is within the span of historical data, defined in (6) |
| t             | Optimism    |      |             | event |            | at round t , defined in Lemma 14                                 |
| U t,h         | Value       |      | function    |       |            | lower bound, defined in Appendix C.2                             |
| err           | Upper       |      | bound       | of    |            | ∥ ω ̂ t,h − ω                                                    |
|               |             |      |             |       |            | ∥ Σ t                                                            |
|               |             |      |             |       |            | , defined in Definition 5                                        |
| err           | Upper       |      | bound       | of    |            | ∥ θ ̂ t,h − T ( ω t,h + θ t,h + 1 )∥ Σ ̂ t,h                     |
|               |             |      |             |       |            | , defined in Lemma 7                                             |
| noise         | Upper       |      | bound       | of    |            | ∥ ξ                                                              |
|               |             |      |             |       |            | t,h ∥ Σ t,h , defined in Definition 5                            |
| noise ,h      | Upper       |      | bound       | of    |            | ∥ ξ                                                              |
|               |             |      |             |       |            | t,h ∥ Λ t,h , defined in Definition 5                            |
| ϕ             | Upper       |      | bound       | of    |            | ∑                                                                |
|               |             |      |             |       |            | t = 1                                                            |
|               |             |      |             |       |            | ∥ ϕ ( s t,h , a t,h )∥ Σ − 1                                     |
|               |             |      |             |       |            | defined in Lemma 16                                              |
| ϕ             | Upper       |      | bound       | of    |            | ∑                                                                |
|               |             |      |             |       |            | t = 1 1 { E                                                      |
|               |             |      |             |       |            | }∥ ϕ ( s t,h , a t,h )∥ Σ ̂ †                                    |
|               |             |      |             |       |            | , defined in Lemma 16                                            |
| B V           | Upper       |      | bound       | of    | ∣          | V t ∣ conditioning on E                                          |
|               |             |      |             |       |            | and E                                                            |
|               |             |      |             |       |            | high , defined in Lemma 13                                       |
| Σ t,h ∑       |             |      |             |       |            |                                                                  |
| i             | = 1 ρ       | i ϕ  | i ϕ         |       |            |                                                                  |
|               |             |      |             | +     | ∑          |                                                                  |
|               |             |      |             |       |            | t − 1                                                            |
|               |             |      |             |       |            | i = 1 ϕ ( s i,h , a i,h ) ϕ                                      |
|               |             |      |             |       |            | ( s i,h , a i,h )                                                |
| Σ ̂ t,h ∑     |             |      |             |       |            |                                                                  |
| t             | − 1         |      |             |       |            |                                                                  |
| i             | = 1 ϕ       | (    | s i,h       | , a   | i,h        | ) ϕ                                                              |
|               |             |      |             |       |            | ( s i,h , a i,h )                                                |
| W h           | Recursively |      |             |       | defined    | as W h − 1 = W h + 2 ε 2 +                                       |
|               |             |      |             |       |            | 2 d ⋅ B                                                          |
|               |             |      |             |       |            | noise ,h +                                                       |
|               |             |      |             |       |            | 2 d ⋅ B                                                          |
|               |             |      |             |       |            | noise + 1                                                        |
| with          |             | W H  | + 1         | = 1   |            |                                                                  |

## B PROOF OVERVIEW

In this section, we provide a sketch of the proof of Theorem [1](#page-6-1) (exact oracle and zero inherent linear Bellman error) with the full proofs deferred to Appendix [C.](#page-17-0) To better convey the intuition, we now assume that the reward function is known, as reward learning is largely standard. In particular, we temporarily remove the estimation and perturbation of rewards (Lines [9](#page-5-1) and [11\)](#page-5-5) and simply assume <sup>ω</sup>t,h = <sup>ω</sup> ⋆ h in Line [12.](#page-5-6)

#### B.1 SPAN ARGUMENT

The very first step of our analysis revolve around two complimentary cases – whether the trajectory at round <sup>t</sup> is in the span of the historical data or not. Let Dt,h ∶= {ϕ(<sup>s</sup>i,h, ai,h)}<sup>t</sup> <sup>i</sup>=<sup>1</sup> and define E span t as the event that the trajectory at round t is in the span of the historical data, i.e.,

$$\mathfrak{E}_t^{\text{span}} := \{ \forall h \in [H] : \phi(s_{t,h}, a_{t,h}) \in \text{span}(\mathcal{D}_{t-1,h}) \}. \quad (6)$$

*(1) In-span case.* When the trajectory generated in the round t is completely within the span of historical data, we can assert that the value function estimation is accurate under πt. Particularly, by linear Bellman completeness, the Bayes optimal of the regression in Line [9](#page-5-1) zeros the empirical risk, as formally stated in the following lemma.

Lemma 1. *For any* <sup>t</sup> ∈ [T]*, we have* ∑ <sup>t</sup>−<sup>1</sup> <sup>i</sup>=<sup>1</sup> (⟨θ̂t,h, ϕ(<sup>s</sup>i,h, ai,h)⟩ − <sup>V</sup> t,h+1(<sup>s</sup>i,h+1))<sup>2</sup> = <sup>0</sup>*.*

**Lemma 1.** *For any 
$$t \in [T]$$
, we have  $\sum_{i=1}^{t-1} ((\widehat{\theta}_{t,h}, \phi(s_{i,h}, a_{i,h})) - \overline{V}_{t,h+1}(s_{i,h+1}))^2 = 0$ .*

Define <sup>U</sup>t(⋅) as a version of <sup>V</sup> <sup>t</sup>(⋅) that minimizes <sup>V</sup> <sup>t</sup>(<sup>s</sup>t,1) while satisfying the high probability bound (precise definition provided at the beginning of Appendix [C.2\)](#page-23-0). It implies the following.

Lemma 2. *For any* <sup>t</sup> ∈ [T]*, whenever* <sup>E</sup> span t *holds, we have* <sup>V</sup> <sup>t</sup>(<sup>s</sup>t,1) = <sup>U</sup>t(<sup>s</sup>t,1) = <sup>V</sup> <sup>π</sup><sup>t</sup> (<sup>s</sup>t,1).

To understand Lemma [2,](#page-16-4) we consider two fact: (1) π<sup>t</sup> is the optimal policy for the estimated value function V <sup>t</sup>, and (2) both V <sup>t</sup> and U<sup>t</sup> has accurate value estimate for the trajectory induced by πt, starting from st,1, because it is in the span of the historical data when E span t holds.

*(2) Out-of-span case.* When any segment of the trajectory is not within the span, we simply pay H in regret and can assert that this will not occur too many times. To see this, we observe the following fact: whenever E span t does not hold, there must exists <sup>h</sup> ∈ [H] such that dim span(Dt,h) = dim span(Dt−1,h) + <sup>1</sup> by definition. Since the dimension of spans cannot exceed <sup>d</sup> for any <sup>h</sup> ∈ [H], the case that E span t does not hold cannot happen for more than dH times. We formally state it in the following lemma.

Lemma 3. *We have* ∑ T <sup>t</sup>=<sup>1</sup> <sup>1</sup>{(<sup>E</sup> span t ) ∁ } ≤ dH.

Hence, we have the following decomposition:

$$V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) = \mathbf{1}\{\{\mathbf{E}_t^{\text{span}}\} \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) + \mathbf{1}\{(\mathbf{E}_t^{\text{span}})^C\} \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right)\} \leq dH^2 \text{ when summed over } t$$

Therefore, we only need to focus on the rounds where E span t holds. This will be the aim of the subsequent sections.

#### B.2 EXPLORATION IN THE NULL SPACE

Lemma [1](#page-16-5) implies that the estimation error only comes from the null space of the historical data, i.e., null({ϕ(<sup>s</sup>i,h, ai,h) ∶ <sup>i</sup> = <sup>1</sup>, . . . , t − <sup>1</sup>}). Therefore, we only need to explore in this null space. While adding explicit bonus is infeasible under linear Bellman completeness, we add noise (Line [11\)](#page-5-5) that can cancel out the estimation error in the null space. This achieves the following:

Lemma 4 (Optimism with constant probability). *Denote* E optm t *as the event that* V ⋆ (<sup>s</sup>t,1) ≤ <sup>V</sup> <sup>t</sup>(<sup>s</sup>t,1)*. Then, for any* <sup>t</sup> ∈ [T]*, we have* Pr(<sup>E</sup> optm t ) ≥ <sup>Γ</sup> 2 (−1) *where* <sup>Γ</sup> *is the cumulative distribution function of the standard normal distribution.*

#### B.3 PROOF OUTLINE

In this section, we outline the structure of the whole proof. Let <sup>V</sup>̃ denote an i.i.d. copy of <sup>V</sup> , and <sup>E</sup>̃span t ,Ẽoptm t denote the counterpart of E span t ,E optm t for <sup>V</sup>̃. We first invoke Lemma [<sup>2</sup>](#page-16-4) and get

<sup>1</sup>{<sup>E</sup> span t }(<sup>V</sup> ⋆ (<sup>s</sup>t,1) − <sup>V</sup> <sup>π</sup><sup>t</sup> (<sup>s</sup>t,1)) = <sup>1</sup>{<sup>E</sup> span t }(<sup>V</sup> ⋆ (<sup>s</sup>t,1) − <sup>U</sup>t(<sup>s</sup>t,1)) ≤ <sup>V</sup> ⋆ (<sup>s</sup>t,1) − <sup>1</sup>{<sup>E</sup> span t }<sup>U</sup>t(<sup>s</sup>t,1) where the last step is by the non-negativity of V ⋆ . Next, we apply Lemma [4](#page-16-6) and get

$$\leq \mathbb{E}_{\widetilde{V}_t} \left[ \min\{\widetilde{V}_t(s_{t,1}), H\} - \mathbf{1}\{\mathfrak{E}_t^{\text{span}}\} U_t(s_{t,1}) \mid \widetilde{\mathfrak{E}}_t^{\text{optm}} \right]$$

Split it into two parts:

$$\begin{aligned}
&= \mathbb{E}_{\widetilde{V}_t} \left[ \mathbf{1}\{(\widetilde{\mathbf{e}}_t^{\text{span}}) \left( \min\{\widetilde{V}_t(s_{t,1}), H\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1}) \right) \middle| \widetilde{\mathbf{e}}_t^{\text{optm}} \right] \\
&\quad + \mathbb{E}_{\widetilde{V}_t} \left[ \mathbf{1}\{(\widetilde{\mathbf{e}}_t^{\text{span}})^C \left( \min\{\widetilde{V}_t(s_{t,1}), H\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1}) \right) \middle| \widetilde{\mathbf{e}}_t^{\text{optm}} \right]
\end{aligned}$$

Note that the quantity inside the first expectation is non-negative, so we can peel off the conditioning event; the quantity in the second term is simply upper bounded by H. Hence, we have

$$\leq \frac{1}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathbf{e}}_t^{\text{span}}\} \left( \min\{\tilde{V}_t(s_{t,1}), H\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1}) \right) \right] + \frac{1}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{(\tilde{\mathbf{e}}_t^{\text{span}})^{\mathcal{C}}\} H \right]$$

Now we split the first term into two parts again:

$$\begin{aligned} &= \frac{1}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathfrak{C}}_t^{\text{span}}\} \min\{\tilde{V}_t(s_{t,1}), H\} - \mathbf{1}\{\tilde{\mathfrak{C}}_t^{\text{span}}\} U_t(s_{t,1}) \right] \\ &\quad + \frac{1}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{(\tilde{\mathfrak{C}}_t^{\text{span}})^{\mathbb{C}} \cap \tilde{\mathfrak{C}}_t^{\text{span}}\} U_t(s_{t,1}) \right] + \frac{1}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{(\tilde{\mathfrak{C}}_t^{\text{span}})^{\mathbb{C}}\} H \right] \\ &\leq \frac{1}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathfrak{C}}_t^{\text{span}}\} \min\{\tilde{V}_t(s_{t,1}), H\} - \mathbf{1}\{\tilde{\mathfrak{C}}_t^{\text{span}}\} U_t(s_{t,1}) \right] + \frac{2}{\Gamma^2(-1)} \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{(\tilde{\mathfrak{C}}_t^{\text{span}})^{\mathbb{C}}\} H \right] \end{aligned}$$

where we used the fact that <sup>1</sup>{<sup>E</sup> span t }<sup>U</sup>t(<sup>s</sup>t,1) ≤ <sup>H</sup>. Taking the expectation over the randomness of the algorithm and use the tower property, which converts <sup>V</sup>̃ into <sup>V</sup> , we obtain

$$\leq \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} \min\{\overline{V}_t(s_{t,1}), H\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1}) \right] + \frac{2}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{(\mathbf{e}_t^{\text{span}})^c\} H \right]$$

The first term is upper bounded by zero due to Lemma [2,](#page-16-4) and the second term is upper bounded by dH<sup>2</sup> by Lemma [3](#page-16-7) when summed over t. This finishes the proof.

Remark 1 (Span Argument and Exponential Blow-Up). *In the proof sketch above, we did not utilize any* <sup>ℓ</sup>2*-norm bound on* <sup>θ</sup>t,h *or* <sup>θ</sup>̂t,h *as did in many prior works. We actually cannot leverage them since they can be exponentially large due to the addition of exponentially large noise. This phenomenon is widely observed in the literature (e.g., [Agrawal et al.](#page-10-11) [\(2021\)](#page-10-11); [Zanette et al.](#page-12-6) [\(2020a\)](#page-12-6)) and is addressed through truncation. However, truncation does not work under linear Bellman completeness, as the Bellman backup of a truncated value function is not necessarily linear. This is why we use the span argument to circumvent this issue.*

## C FULL PROOF FOR S[ECTION](#page-6-2) 5

In this section, we present and prove the following main theorem, which provides the regret bound in terms of parameters <sup>ε</sup>1, <sup>ε</sup>2, and <sup>ε</sup>B. Setting <sup>ε</sup><sup>1</sup> = <sup>ε</sup><sup>2</sup> = <sup>ε</sup><sup>B</sup> = <sup>0</sup> yields Theorem [1,](#page-6-1) setting <sup>ε</sup><sup>B</sup> = <sup>0</sup> yields Theorem [2,](#page-7-4) and setting <sup>ε</sup><sup>1</sup> = <sup>ε</sup><sup>2</sup> = <sup>0</sup> yields Theorem [3.](#page-7-5)

Theorem 6. *Assume the MDP has* εB*-inherent linear Bellman error. Under Assumptions [1,](#page-3-1) [3](#page-7-1) and [4,](#page-7-0) when executing Algorithm [<sup>1</sup>](#page-5-0) with parameters* <sup>σ</sup><sup>R</sup> = √ HB<sup>R</sup> err *and* <sup>σ</sup><sup>h</sup> ≥ √ H( √ 3γB<sup>P</sup> err + √ <sup>8</sup>m(<sup>W</sup><sup>h</sup> + <sup>ε</sup>2))*, we have*

$$\mathbb{E} \left[ \sum_{t=1}^T \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] = \tilde{O} \left( d^{5/2} H^{5/2} + d^2 H^{3/2} \sqrt{T} + \varepsilon_1 \gamma \left( d H^2 + d^{3/2} H \sqrt{T} \right) \right. \\ \left. + \sqrt{\varepsilon_B} \left( d^2 H^{5/2} \sqrt{T} + d^{3/2} H^{3/2} T \right) + \varepsilon_B \gamma \left( d H^2 \sqrt{T} + d^{3/2} H T \right) \right).$$

Exact value of parameters <sup>σ</sup><sup>R</sup> and <sup>σ</sup><sup>h</sup> in Theorem [6.](#page-17-2) We define <sup>W</sup>H+<sup>1</sup> = <sup>1</sup> and recursively define <sup>W</sup>h−<sup>1</sup> = <sup>W</sup><sup>h</sup> + <sup>2</sup>ε<sup>2</sup> + √ <sup>2</sup><sup>d</sup> ⋅ <sup>B</sup> P noise,h + √ <sup>2</sup><sup>d</sup> ⋅ <sup>B</sup> R noise + <sup>1</sup>. Plugging the definition of all these symbols involved and ignoring lower order terms (i.e., logarithmic and constant terms), we get

$$W_{h-1} \approx d\sqrt{mH} \cdot W_h + \varepsilon_1 \cdot d\gamma\sqrt{H} + \varepsilon_B \cdot d\gamma\sqrt{HT} + \sqrt{\varepsilon_B} \cdot d\sqrt{T} + d^{3/2}. \quad (7)$$

Solving this recursion, we get

$$\begin{aligned} W_h \approx & \left( d\sqrt{mH} \right)^{H+1-h} + \left( d\sqrt{mH} \right)^{H-h} \left( \varepsilon_1 \cdot d\gamma\sqrt{H} + \varepsilon_B \cdot d\gamma\sqrt{HT} + \sqrt{\varepsilon_B} \cdot d\sqrt{T} + d^{3/2} \right) \\ \approx & \left( d\sqrt{mH} \right)^{H-h} \left( \varepsilon_1 \cdot d\gamma\sqrt{H} + \varepsilon_B \cdot d\gamma\sqrt{HT} + \sqrt{\varepsilon_B} \cdot d\sqrt{T} + d^{3/2} + d\sqrt{mH} \right). \end{aligned}$$

We insert this into the value of σ<sup>h</sup> and get

$$\sigma_h \approx (d\sqrt{mH})^{H-h+1} (\varepsilon_1 \cdot \gamma\sqrt{H} + \varepsilon_B \cdot \gamma\sqrt{HT} + \sqrt{\varepsilon_B} \cdot \sqrt{T} + d^{1/2} + \sqrt{mH}).$$

We can also get the value of σ<sup>R</sup> as

$$\sigma_R \approx \sqrt{H} \left( \sqrt{d \log(HT)} + \varepsilon_1 + \sqrt{\varepsilon_B T} \right).$$

Define <sup>Λ</sup> = ∑ m <sup>i</sup>=<sup>1</sup> ρiϕiϕ ⊺ i . It is straightforward that both Λ and Λt,h (constructed in Line [7](#page-5-7) of Algorithm [1\)](#page-5-0) are invertible. We define <sup>λ</sup> ∶= maxs,a ∥ϕ(s, a)∥<sup>Λ</sup>−<sup>1</sup> and <sup>λ</sup>t,h ∶= maxs,a ∥ϕ(s, a)∥<sup>Λ</sup>−<sup>1</sup> t,h .

Lemma 5. *The matrices* Λ *and* Λt,h *are invertible. Furthermore, we also have that*

- <sup>λ</sup> ≤ √ d*;*
- <sup>λ</sup>t,h ≤ √ <sup>2</sup><sup>d</sup> *for all* <sup>t</sup> ∈ [T] *and all* <sup>h</sup> ∈ [H]*.*

Proof of Lemma [5.](#page-18-1) By the last item in Lemma [23,](#page-36-0) we have <sup>λ</sup> ≤ √ d. In what follows, we will show that <sup>Λ</sup> ⪯ 2Λt,h, which implies <sup>λ</sup>t,h ≤ √ <sup>2</sup><sup>λ</sup> ≤ √ 2d.

For any <sup>x</sup> ∈ <sup>R</sup> d , we have

$$\begin{aligned} x^\top \Lambda x &= \sum_{i=1}^m \rho_i (x^\top \phi_i)^2 = \sum_{i=1}^m \rho_i \left( x^\top \phi_{t,h,i}^\perp + x^\top \phi_{t,h,i}^\perp \right)^2 \\ &\leq 2 \sum_{i=1}^m \rho_i \left( x^\top \phi_{t,h,i}^\perp \right)^2 + 2 \sum_{i=1}^m \rho_i \left( x^\top \phi_{t,h,i}^\perp \right)^2 \quad (\text{using } (a+b)^2 \leq 2a^2 + 2b^2) \\ &= 2x^\top \Lambda_{t,h} x. \end{aligned}$$

This implies that <sup>Λ</sup> ⪯ 2Λt,h.

#### C.1 HIGH-PROBABILITY EVENT AND BOUNDEDNESS

Lemma 6 (Reward estimation). *With probability at least* <sup>1</sup> − <sup>δ</sup>*, for any* <sup>t</sup> ∈ [T] *and* <sup>h</sup> ∈ [H]*,*

$$\| \widehat{\omega}_{t,h} - \omega_h^* \|_{\Sigma_t} \leq \sqrt{1030(1 + \varepsilon_2)^4 d \log (8(1 + \varepsilon_2)e^2 T^2 H/\delta) + 4\varepsilon_1^2 + 16(1 + \varepsilon_2)(1 + \varepsilon_B T)}.$$

Proof of Lemma [6.](#page-18-2) For the ease of notation, we fixed t and h in the proof and simply write the regression problem as

$$\widehat{\omega} \leftarrow \operatorname{argmin}_{\omega \in \mathcal{O}(1)} \sum_{i=1}^n \left( \omega^\top \phi_i - r_i \right)^2$$

where we have dropped the subscripts t and h for notational simplicity. Here ϕ<sup>i</sup> and r<sup>i</sup> are abbreviated notations for <sup>ϕ</sup>(<sup>s</sup>i,h, ai,h) and <sup>r</sup>i,h, respectively, and <sup>n</sup> = <sup>t</sup> − <sup>1</sup>.

Note that, due to approximate oracle (Assumption [3\)](#page-7-1), <sup>ω</sup>̂ actually belongs to O(<sup>1</sup> + <sup>ε</sup>2) instead of O(1). Denote C as an <sup>ℓ</sup>1-norm <sup>α</sup>-cover (Definition [6\)](#page-37-2) on O(1+<sup>ε</sup>2) such that for any <sup>ω</sup> ∈ O(1+<sup>ε</sup>2),

there exists a <sup>ω</sup>̃ ∈ C, such that ∑ n <sup>i</sup>=<sup>1</sup> ∣ω <sup>⊺</sup>ϕ<sup>i</sup> − <sup>ω</sup>̃ ⊺ϕi ∣/<sup>n</sup> ≤ <sup>α</sup>. Since O(<sup>1</sup> + <sup>ε</sup>2) is a linear function class, which has pseudo-dimension d (Definition [8\)](#page-37-3), we have

$$|\mathcal{C}| \leq (8(1 + \varepsilon_2)e^2/\alpha)^d \quad (8)$$

by Lemma [27.](#page-37-4) Now define z ω <sup>i</sup> = (<sup>ω</sup> <sup>⊺</sup>ϕ<sup>i</sup> − <sup>r</sup>i) <sup>2</sup> − ((<sup>ω</sup> ⋆ ) <sup>⊺</sup>ϕ<sup>i</sup> − <sup>r</sup>i) 2 . Then we have ∣<sup>z</sup> ω i ∣ ≤ <sup>4</sup>(<sup>1</sup> + <sup>ε</sup>2) 2 , and

$$\begin{aligned}\mathbb{E}_i[z_i^{\omega}] &= \mathbb{E}_i \left[ (\omega^\top \phi_i - (\omega^*)^\top \phi_i) (\omega^\top \phi_i + (\omega^*)^\top \phi_i - 2r_i) \right] \\ &= \mathbb{E}_i \left[ (\omega^\top \phi_i - (\omega^*)^\top \phi_i) (\omega^\top \phi_i - (\omega^*)^\top \phi_i + 2((\omega^*)^\top \phi_i - r_i)) \right] \\ &\geq (\omega^\top \phi_i - (\omega^*)^\top \phi_i)^2 - 4(1 + \varepsilon_2)\varepsilon_B,\end{aligned}$$

and moreover,

$$\mathbb{E}_i[(z_i^\omega)^2] = \mathbb{E}_i[(\omega^\top \phi_i - (\omega^*)^\top \phi_i)^2(\omega^\top \phi_i + (\omega^*)^\top \phi_i - 2r_i)^2] \leq 16(1 + \varepsilon_2)^2(\omega^\top \phi_i - (\omega^*)^\top \phi_i)^2$$

We note that z ω <sup>i</sup> −<sup>E</sup><sup>i</sup> <sup>z</sup> ω i is a martingale difference sequence and ∣<sup>z</sup> ω <sup>i</sup> −<sup>E</sup><sup>i</sup> <sup>z</sup> ω i ∣ ≤ <sup>8</sup>(1+<sup>ε</sup>2) 2 . Applying Freedman's inequality (Lemma [22\)](#page-36-1) and a union bound over <sup>ω</sup> ∈ C, we have with probability at least <sup>1</sup> − <sup>δ</sup>, for all <sup>ω</sup> ∈ C,

$$\begin{aligned} & \sum_{i=1}^n (\omega^\top \phi_i - (\omega^*)^\top \phi_i)^2 - \sum_{i=1}^n z_i^\omega \\ & \leq \eta \sum_{i=1}^n 16(1 + \varepsilon_2)^2 (\omega^\top \phi_i - (\omega^*)^\top \phi_i)^2 + \frac{8(1 + \varepsilon_2)^2 \log(|\mathcal{C}|/\delta)}{\eta} + 4(1 + \varepsilon_2)\varepsilon_B T. \end{aligned} \quad (9)$$

Recall that <sup>ω</sup>̂ is the least square solution. Denote <sup>ω</sup>̃ ∈ C as the point that is closest to <sup>ω</sup>̂, which means that: ∑ n <sup>i</sup>=<sup>1</sup> ∣ω̂ <sup>⊺</sup>ϕ<sup>i</sup> − <sup>ω</sup>̃ ⊺ϕi ∣ ≤ nα. We can derive the following relationship between <sup>ω</sup>̂ and <sup>ω</sup>̃:

$$\begin{aligned} \sum_{i=1}^n (\bar{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 &\leq 2 \sum_{i=1}^n (\bar{\omega}^\top \phi_i - \bar{\omega}^\top \phi_i)^2 + 2 \sum_{i=1}^n (\bar{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 \leq 2n^2\alpha^2 + 2 \sum_{i=1}^n (\bar{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2, \\ \sum_{i=1}^n z_i^{\bar{\omega}} - \sum_{i=1}^n z_i^{\bar{\omega}} &= \sum_{i=1}^n (\bar{\omega}^\top \phi_i - \bar{\omega}^\top \phi_i)(\bar{\omega}^\top \phi_i + \bar{\omega}^\top \phi_i - 2r_i) \leq 4(1 + \varepsilon_2)n\alpha. \end{aligned}$$

Now plug <sup>ω</sup>̃ into [\(9\)](#page-19-0) and re-arrange terms, we get:

$$\sum_{i=1}^n (\bar{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 \leq \frac{1}{1 - 16(1 + \varepsilon_2)^2 \eta} \sum_{i=1}^n z_i^{\bar{\omega}} + \frac{8(1 + \varepsilon_2)^2}{\eta(1 - 16(1 + \varepsilon_2)^2 \eta)} \cdot \log(|\mathcal{C}|/\delta) + \frac{4(1 + \varepsilon_2)\varepsilon_B T}{1 - 16(1 + \varepsilon_2)^2 \eta}.$$

Setting <sup>η</sup> = (32(<sup>1</sup> + <sup>ε</sup>2) 2 ) −1 , we get

$$\sum_{i=1}^n (\tilde{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 \leq 2 \sum_{i=1}^n z_i^{\tilde{\omega}} + 512(1 + \varepsilon_2)^4 \log(|\mathcal{C}|/\delta) + 8(1 + \varepsilon_2)\varepsilon_B T.$$

Using the relationships between <sup>ω</sup>̂ and <sup>ω</sup>̃ that we derived above, we have:

$$\begin{aligned} & \sum_{i=1}^n (\tilde{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 \\ & \leq 2n^2 \alpha^2 + 4 \sum_{i=1}^n z_i^{\tilde{\omega}} + 1024(1 + \varepsilon_2)^4 \log(|\mathcal{C}|/\delta) + 16(1 + \varepsilon_2)\varepsilon_B T. \\ & \leq 2n^2 \alpha^2 + 4 \sum_{i=1}^n z_i^{\tilde{\omega}} + 1024(1 + \varepsilon_2)^4 \log(|\mathcal{C}|/\delta) + 16(1 + \varepsilon_2)n\alpha + 16(1 + \varepsilon_2)\varepsilon_B T. \end{aligned}$$

Since <sup>ω</sup>̂ is the (approximate) least square solution, we have ∑<sup>i</sup> <sup>z</sup> ω̂ <sup>i</sup> ≤ <sup>ε</sup> 2 1 . This implies that:

$$\sum_{i=1}^n (\varpi^\top \phi_i - (\omega^*)^\top \phi_i)^2 \leq 2n^2\alpha^2 + 4\varepsilon_1^2 + 1024(1 + \varepsilon_2)^4 \log(|\mathcal{C}|/\delta) + 16(1 + \varepsilon_2)(n\alpha + \varepsilon_B T).$$

Now plugging the covering number [\(8\)](#page-19-1) and setting <sup>α</sup> = <sup>1</sup>/<sup>n</sup>, we obtain

$$\begin{aligned} \sum_{i=1}^n (\omega^\top \phi_i - (\omega^*)^\top \phi_i)^2 &\leq 2 + 4\varepsilon_1^2 + 1024(1 + \varepsilon_2)^4 d \log(8(1 + \varepsilon_2)e^2 n/\delta) + 16(1 + \varepsilon_2)(1 + \varepsilon_B T) \\ &\leq 1026(1 + \varepsilon_2)^4 d \log(8(1 + \varepsilon_2)e^2 n/\delta) + 4\varepsilon_1^2 + 16(1 + \varepsilon_2)(1 + \varepsilon_B T). \end{aligned}$$

Finally, we have

$$\|\widehat{\omega} - \omega_h^*\|_{\Sigma_t}^2 = \sum_{i=1}^n (\widehat{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 + \sum_{i=1}^m \rho_i (\widehat{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2.$$

Here, with some abuse of notation, the ϕi's in the right term are the support points of the optimal design. The first term is already bounded above. The second term can be bounded by

$$\sum_{i=1}^m \rho_i (\widehat{\omega}^\top \phi_i - (\omega^*)^\top \phi_i)^2 \leq \sum_{i=1}^m \rho_i \cdot 4(1 + \varepsilon_2) = 4(1 + \varepsilon_2).$$

We add it into the constant of the first term. Then, we apply the union bound over all <sup>t</sup> ∈ [T] and <sup>h</sup> ∈ [H] to get the desired result.

Lemma 7 (Value function estimation). *Suppose that* T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1) ∈ O(<sup>W</sup>h)*. Then,*

$$\sum_{i=1}^{t-1} (\langle \widehat{\theta}_{t,h}, \phi(s_{i,h}, a_{i,h}) \rangle - \overline{V}_{t,h+1}(s_{i,h+1}))^2 \leq \varepsilon_1^2 + T\varepsilon_{\text{B}}^2.$$

*Furthermore,* ∥θ̂t,h − T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1)∥Σ̂t,h ≤ √ 2ε 2 <sup>1</sup> + <sup>4</sup>T ε<sup>2</sup> B =∶ <sup>B</sup> P err.

Proof of Lemma [7.](#page-20-1) The Bayes optimal T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1) should achieve the empirical risk of at most εB, i.e.,

$$\forall i \in [t-1] : \quad \left| \left\langle \phi(s_{i,h}, a_{i,h}), \mathcal{T}(\overline{\omega}_{t,h} + \overline{\theta}_{t,h+1}) \right\rangle - \overline{V}_{t,h+1}(s_{i,h+1}) \right| \leq \varepsilon_{\text{B}}.$$

Since T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1) is realizable (i.e., T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1) ∈ <sup>O</sup>(<sup>W</sup>h)), and <sup>θ</sup>̂t,h minimizes the objective up to precision ε1, it should satisfy the following

$$\sum_{i=1}^{t-1} \left( (\widehat{\theta}_{t,h}, \phi(s_{i,h}, a_{i,h})) - \overline{V}_{t,h+1}(s_{i,h+1}) \right)^2 \leq \varepsilon_1^2 + T\varepsilon_{\text{B}}^2.$$

Combining the above two results, we arrive at the following:

$$\begin{aligned} & \sum_{i=1}^{t-1} \left\langle \phi(s_{i,h}, a_{i,h}), \widehat{\theta}_{t,h} - \mathcal{T}(\bar{\omega}_{t,h} + \bar{\theta}_{t,h+1}) \right\rangle^2 \\ & \leq 2 \sum_{i=1}^{t-1} \left( \left\langle \phi(s_{i,h}, a_{i,h}), \widehat{\theta}_{t,h} \right\rangle - \overline{V}_{t,h+1}(s_{i,h+1}) \right)^2 + 2 \sum_{i=1}^{t-1} \left( \overline{V}_{t,h+1}(s_{i,h+1}) - \left\langle \phi(s_{i,h}, a_{i,h}), \mathcal{T}(\bar{\omega}_{t,h} + \bar{\theta}_{t,h+1}) \right\rangle \right)^2 \\ & \quad \quad \quad \text{(using } (a+b)^2 \leq 2a^2 + 2b^2) \end{aligned}$$

$$\leq 2\varepsilon_1^2 + 4T\varepsilon_B^2.$$

This implies that

$$\left\|\widehat{\theta}_{t,h} - \mathcal{T}(\overline{\omega}_{t,h} + \overline{\theta}_{t,h+1})\right\|_{\widehat{\Sigma}_{t,h}}^2 \leq 2\varepsilon_1^2 + 4T\varepsilon_{\text{B}}^2.$$

Definition 5 (High-probability events). *Define event* E high *as*

$$\begin{aligned} \mathfrak{e}^{\text{high}} &:= \left\{ \forall t \in [T], \forall h \in [H] : \|\xi_{t,h}^P\|_{\Lambda_{t,h}} \leq \sigma_h \sqrt{2d \log(6dH^2 T^2)} \right\} =: B_{\text{noise},h}^P \\ &\cap \left\{ \forall t \in [T], \forall h \in [H] : \|\xi_{t,h}^R\|_{\Sigma_{t,h}} \leq \sigma_R \sqrt{2d \log(6dH^2 T^2)} \right\} =: B_{\text{noise}}^R \\ &\cap \left\{ \forall t \in [T], \forall h \in [H] : \|\eta_{t,h}^R\|_{\Sigma_{t,h}} \leq B_{\text{err}}^R \right\} \end{aligned}$$

$$\text{where } B_{\text{err}}^R := \sqrt{1030(1 + \varepsilon_2)^4 d \log (24(1 + \varepsilon_2)e^2 T^3 H^2) + 4\varepsilon_1^2 + 16(1 + \varepsilon_2)(1 + \varepsilon_B T)}.$$

Lemma 8. *We have* Pr(<sup>E</sup> high) > <sup>1</sup> − <sup>1</sup>/(HT)*.*

Proof of Lemma [8.](#page-20-2) Below we show that each event defined in Definition [5](#page-20-0) holds with probability at least <sup>1</sup> − <sup>1</sup>/(3HT). Then, by union bound, we have the desired result.

*Proof of the first event.* The way we generate ξ P t,h is equivalent to first sampling <sup>ζ</sup>t,h ∼ N (0, (<sup>σ</sup>h) <sup>2</sup>Λ −1 t,h) and then set <sup>ξ</sup> P t,h ← (<sup>I</sup> − <sup>P</sup>t,h)<sup>ζ</sup>t,h. By Lemma [<sup>20</sup>](#page-34-1) and the union bound, we have

$$\Pr \left( \forall t \in [T], \forall h \in [H] : \|\zeta_{t,h}\|_{\Lambda_{t,h}} > \sigma_h \sqrt{2d \log(6dH^2T^2)} \right) \leq 1/(3HT).$$

Then, by definition, we have

$$\begin{aligned} \|\xi_{t,h}^P\|_{\Lambda_{t,h}}^2 &= \|(1 - P_{t,h})\zeta_{t,h}\|_{\Lambda_{t,h}}^2 \\ &= \zeta_{t,h}^\top (I - P_{t,h}) \sum_{i=1}^m \left( \phi_{t,h,i}^\parallel (\phi_{t,h,i}^\parallel)^\top + \phi_{t,h,i}^\perp (\phi_{t,h,i}^\perp)^\top \right) (I - P_{t,h}) \zeta_{t,h} \\ &= \zeta_{t,h}^\top \sum_{i=1}^m \phi_{t,h,i}^\perp (\phi_{t,h,i}^\perp)^\top \zeta_{t,h} \\ &\leq \zeta_{t,h}^\top \sum_{i=1}^m \left( \phi_{t,h,i}^\parallel (\phi_{t,h,i}^\parallel)^\top + \phi_{t,h,i}^\perp (\phi_{t,h,i}^\perp)^\top \right) \zeta_{t,h} \end{aligned}$$

where the third step holds by the fact that ϕ ⊥ is in the null space and ϕ ∥ is in the span. Hence, we conclude that ∥<sup>ξ</sup> P t,h∥<sup>Λ</sup>t,h ≤ ∥<sup>ζ</sup>t,h∥<sup>Λ</sup>t,h .

*Proof of the second event.* Applying Lemma [20](#page-34-1) and the union bound, we have

$$\Pr \left( \forall t \in [T] : \|\xi_t^R\|_{\Sigma_t} > \sigma_R \sqrt{2d \log(6dHT^2)} \right) \leq 1/(3HT).$$

*Proof of the third event.* This is directly from Lemma [6.](#page-18-2)

Lemma 9 (Boundness of parameters). *Under Assumption [4,](#page-7-0) conditioning on* E high*, the following hold for all* <sup>t</sup> ∈ [T] *and* <sup>h</sup> ∈ [H]*:*

- *1.* maxs,a ∣⟨ϕ(s, a), <sup>θ</sup>̂t,h⟩∣ ≤ <sup>W</sup><sup>h</sup> + <sup>ε</sup>2*;*
- *2.* maxs,a ∣⟨ϕ(s, a), T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1)⟩∣ ≤ <sup>W</sup>h*;*
- *3.* ∥<sup>η</sup>t,h∥Σ̂t,h ≤ <sup>B</sup> P err*;*
- *4.* ∥<sup>η</sup>t,h∥<sup>Λ</sup> ≤ <sup>2</sup>(<sup>W</sup><sup>h</sup> + <sup>ε</sup>2) √ m*;*
- *5.* ∥<sup>η</sup>t,h∥<sup>Λ</sup>t,h ≤ √ 3γB<sup>P</sup> err + √ <sup>8</sup>m(<sup>W</sup><sup>h</sup> + <sup>ε</sup>2) *;*
- *6.* maxs,a ∣⟨ϕ(s, a), <sup>θ</sup>t,h⟩∣ ≤ <sup>W</sup>h−<sup>1</sup> − √ <sup>2</sup><sup>d</sup> ⋅ <sup>B</sup> R noise − <sup>1</sup> − <sup>ε</sup><sup>2</sup>
- *7.* max<sup>s</sup> <sup>V</sup> t,h(s) = maxs,a ∣<sup>Q</sup>t,h(s, a)∣ ≤ <sup>W</sup>h−1*.*

Proof of Lemma [9.](#page-21-0) Fix <sup>t</sup> ∈ [T]. We prove these items by induction on <sup>h</sup>. The base case (<sup>h</sup> = <sup>H</sup> +<sup>1</sup>) clearly holds since there is actually nothing at (<sup>H</sup> + <sup>1</sup>)-th step. Now assume they hold for <sup>h</sup>+ <sup>1</sup>, and we will show that they hold for h as well.

*Proof of Item [1.](#page-21-1)* It is simply by Line [9](#page-5-1) of Algorithm [1](#page-5-0) and Assumption [3.](#page-7-1)

*Proof of Item [2.](#page-21-2)* By linear Bellman completeness (Definition [1\)](#page-2-0), for any s, a, we have,

$$\begin{aligned} |\langle \phi(s, a), \mathcal{T}(\bar{\omega}_{t,h} + \bar{\theta}_{t,h+1}) \rangle| &= \left| \mathbb{E}_{s' \sim \mathcal{T}(s, a)} \max_{a'} \langle \phi(s', a'), \bar{\omega}_{t,h} + \bar{\theta}_{t,h+1} \rangle \right| \\ &\leq \max_{s,a} |\langle \phi(s, a), \bar{\omega}_{t,h} + \bar{\theta}_{t,h+1} \rangle| \end{aligned}$$

$$\begin{aligned} &\leq \max_{s,a} |\langle \phi(s, a), \bar{\omega}_t, h \rangle| + \max_{s,a} |\langle \phi(s, a), \xi_{t,h}^R \rangle| + \max_{s,a} |\langle \phi(s, a), \bar{\theta}_{t,h+1} \rangle| \\ &\leq (1 + \varepsilon_2) + \max_{s,a} \|\phi(s, a)\|_{\Sigma_{t,h}^{-1}} \|\xi_{t,h}^R\|_{\Sigma_{t,h}} + (W_h - \sqrt{2d} \cdot B_{\text{noise}}^R - 1 - \varepsilon_2) \\ &\leq 1 + \varepsilon_2 + \sqrt{2d} \cdot B_{\text{noise}}^R + (W_h - \sqrt{2d} \cdot B_{\text{noise}}^R - 1 - \varepsilon_2) = W_h. \end{aligned}$$

*Proof of Item [3.](#page-21-3)* This is directly from Lemma [7.](#page-20-1)

*Proof of Item [4.](#page-21-4)* By triangle inequality, we have

$$\|\eta_{t,h}\|_{\Lambda} = \|\widehat{\theta}_{t,h} - \mathcal{T}(\overline{\omega}_{t,h} + \overline{\theta}_{t,h+1})\|_{\Lambda} \leq \|\widehat{\theta}_{t,h}\|_{\Lambda} + \|\mathcal{T}(\overline{\omega}_{t,h} + \overline{\theta}_{t,h+1})\|_{\Lambda} \leq 2(W_h + \varepsilon_2)\sqrt{m}.$$

where the last step is by

$$\|\widehat{\theta}_{t,h}\|_{\Lambda} = \sqrt{\sum_{i=1}^m \langle \phi_i, \widehat{\theta}_{t,h} \rangle^2} \leq \sqrt{\sum_{i=1}^m (W_h + \varepsilon_2)^2} = (W_h + \varepsilon_2)\sqrt{m}$$

and the similar for T (<sup>ω</sup>t,h + <sup>θ</sup>t,h+1).

*Proof of Item [5.](#page-21-5)* By definition, we have

$$\begin{aligned} \|\eta_{t,h}\|_{\Lambda_{t,h}}^2 &= \sum_{i=1}^m \rho_i \left( \left\langle \phi_{t,h,i}^{\parallel}, \eta_{t,h} \right\rangle^2 + \left\langle \phi_{t,h,i}^{\perp}, \eta_{t,h} \right\rangle^2 \right) \\ &= \sum_{i=1}^m \rho_i \left( \langle P_{t,h}\phi_i, \eta_{t,h} \rangle^2 + \langle (I - P_{t,h})\phi_i, \eta_{t,h} \rangle^2 \right) \\ &\leq \sum_{i=1}^m \rho_i \left( 3\langle P_{t,h}\phi_i, \eta_{t,h} \rangle^2 + 2\langle \phi_i, \eta_{t,h} \rangle^2 \right) \quad (\text{using } (a+b)^2 \leq a^2 + b^2) \\ &\leq 3 \sum_{i=1}^m \rho_i \left( \left\| \phi_{t,h,i}^{\parallel} \right\|_{\overline{\Sigma}_{t,h}^{\uparrow}}^2 \|\eta_{t,h}\|_{\overline{\Sigma}_{t,h}}^2 \right) + 2\|\eta_{t,h}\|_{\Lambda}^2 \quad (\text{Cauchy-Schwartz, Lemma 25}) \end{aligned}$$

We have ∥<sup>ϕ</sup> ∥ t,h,i∥Σ̂† t,h = ∥<sup>P</sup>t,hϕi∥Σ̂† t,h = ∥<sup>ϕ</sup>i∥Σ̂† t,h by Lemma [26.](#page-36-3) By Assumption [4,](#page-7-0) this is upper bounded by <sup>γ</sup>. The second term, ∥<sup>η</sup>t,h∥Σ̂t,h , is upper bounded by B P err by Item [3.](#page-21-3)

Hence, we have

$$\begin{aligned} \|\eta_{t,h}\|_{\Lambda_{t,h}}^2 &\leq 3\gamma^2(B_{\text{err}}^{\text{P}})^2 + 2\|\eta_{t,h}\|_{\Lambda}^2 \\ &\leq 3\gamma^2(B_{\text{err}}^{\text{P}})^2 + 8(W_h + \varepsilon_2)^2 m. \end{aligned} \quad (\text{Item 4})$$

*Proof of Item [6.](#page-21-6)* We have

$$\begin{aligned} \max_{s,a} |\langle \phi(s, a), \bar{\theta}_{t,h} \rangle| &= \max_{s,a} |\langle \phi(s, a), \bar{\theta}_{t,h} + \xi_{t,h}^P \rangle| \\ &\leq \max_{s,a} |\langle \phi(s, a), \bar{\theta}_{t,h} \rangle| + \max_{s,a} |\langle \phi(s, a), \xi_{t,h}^P \rangle| \\ &\leq W_h + \varepsilon_2 + \max_{s,a} \|\phi(s, a)\|_{\Lambda_{t,h}^{-1}} \|\xi_{t,h}^P\|_{\Lambda_{t,h}} \\ &\leq W_h + \varepsilon_2 + \sqrt{2d} \cdot B_{\text{noise},h}^P \quad (\text{Lemma 5}) \\ &= W_{h-1} - \sqrt{2d} \cdot B_{\text{noise}}^R - 1 - \varepsilon_2. \end{aligned}$$

*Proof of Item [7.](#page-21-7)* We have

$$\begin{aligned}
|\bar{Q}_{t,h}(s, a)| &= |\langle \phi(s, a), \bar{\theta}_{t,h} \rangle + \langle \phi(s, a), \bar{\omega}_{t,h} \rangle| \\
&\leq |\langle \phi(s, a), \bar{\theta}_{t,h} \rangle| + |\langle \phi(s, a), \bar{\omega}_{t,h} \rangle| + |\langle \phi(s, a), \xi_t^R \rangle| \\
&\leq (W_{h-1} - \sqrt{2d} \cdot B_{\text{noise}}^R - 1 - \varepsilon_2) + (1 + \varepsilon_2) + \sqrt{2d} \cdot B_{\text{noise}}^R \\
&= W_{h-1}.
\end{aligned}$$

## C.2 VALUE DECOMPOSITION

We note that, at any round <sup>t</sup> ∈ [T], conditioning on all information collected up to round <sup>t</sup> − <sup>1</sup>, the randomness of <sup>V</sup> <sup>t</sup> only comes from the Gaussian noise {<sup>ξ</sup> P t,h, ξ<sup>R</sup> t,h} H <sup>h</sup>=<sup>1</sup> . In other words, V <sup>t</sup> can be considered *a functional of the Gaussian noise*. In light of this, we define

$$V_{t,h}[\check{\xi}_1^P, \dots, \check{\xi}_H^P, \check{\xi}_1^R, \dots, \check{\xi}_H^R](\cdot)$$

as a functional of the noise variable, which maps the given noise variable to the value function produced by the algorithm by replacing the random Gaussian noise with the variable ˇξ P 1 , . . . , ˇξ P H, ˇξ R 1 , . . . , ˇξ R <sup>H</sup>. By definition, we immediately have

$$\overline{V}_{t,h}(\cdot) = V_{t,h}[\xi_{t,1}^P, \dots, \xi_{t,H}^P, \xi_{t,1}^R, \dots, \xi_{t,H}^R](\cdot).$$

Next, we define U<sup>t</sup> as the minimum of the following program

$$\begin{aligned}
 & \min_{\substack{\xi_1^P, \dots, \xi_H^P, \xi_1^R, \dots, \xi_H^R}} V_{t,1} [\xi_1^P, \dots, \xi_H^P, \xi_1^R, \dots, \xi_H^R](s_{t,1}) \\
 \text{s.t.} \quad & \forall h \in [H] : \|\xi_{t,h}^P\|_{\Lambda_{t,h}} \leq B_{\text{noise},h}^P, \quad \|\xi_{t,h}^R\|_{\Sigma_{t,h}} \leq B_{\text{noise}}^R.
 \end{aligned}$$

In other words, U<sup>t</sup> achieves the minimum value at st,<sup>1</sup> while satisfying the high-probability constraints (E high) on the noise variable. We denote ξ P 1 , . . . , ξ<sup>P</sup> H , ξ<sup>R</sup> 1 , . . . , ξ<sup>R</sup> H as the minimizer of the above program, and will always use underlined variables to represent the intermediate variables corresponding to <sup>U</sup><sup>t</sup> (such as <sup>θ</sup>̂, <sup>θ</sup>, <sup>ω</sup>̂, <sup>ω</sup>, <sup>Q</sup>, <sup>V</sup> ) to distinguish them from the variables corresponding to <sup>V</sup> <sup>t</sup>, (<sup>θ</sup>̂, <sup>θ</sup>, <sup>ω</sup>̂, <sup>ω</sup>, <sup>Q</sup>, <sup>V</sup> ). We note that, under <sup>E</sup> high, we directly have <sup>U</sup>t(<sup>s</sup>t,1) ≤ <sup>V</sup> <sup>t</sup>(<sup>s</sup>t,1).

Below is a decomposition lemma under deterministic transition. Note that it slightly differs from the usual value decomposition lemma under stochastic transitions, where we have to take the expectation over trajectory randomness. This distinction is crucial to our analysis: by not accounting for trajectory randomness, we can effectively leverage our span argument.

We denote {<sup>s</sup>t,h, at,h} H <sup>h</sup>=<sup>1</sup> as the trajectory generated by executing π<sup>t</sup> with initial state st,1, and {s ⋆ t,h, a<sup>⋆</sup> t,h} H <sup>h</sup>=<sup>1</sup> as the trajectory generated by executing π ⋆ with initial state <sup>s</sup> ⋆ t,<sup>1</sup> = <sup>s</sup>t,1.

Lemma 10 (Value decomposition under deterministic transition). *Under deterministic transition (Assumption [1\)](#page-3-1), we have*

$$V^{\pi_t}(s_{t,1}) - \overline{V}_t(s_{t,1}) = \sum_{h=1}^H \left( \overline{V}_{t,h+1}(s_{t,h+1}) - \langle \overline{\theta}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle + \langle \omega_h^* - \overline{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right); \quad (10)$$

$$V^*(s_{t,1}) - \overline{V}_t(s_{t,1}) \leq \sum_{h=1}^H \left( \overline{V}_{t,h+1}(s_{t,h+1}^*) - \langle \overline{\theta}_{t,h}, \phi(s_{t,h}^*, a_{t,h}^*) \rangle + \langle \omega_h^* - \overline{\omega}_{t,h}, \phi(s_{t,h}^*, a_{t,h}^*) \rangle \right). \quad (11)$$

*Similarly, we have*

$$V^{\pi_t}(s_{t,1}) - U_t(s_{t,1}) \leq \sum_{h=1}^H \left( U_{t,h+1}(s_{t,h+1}) - \langle \bar{\theta}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle + \langle \omega_h^* - \bar{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right). \quad (12)$$

Proof of Lemma [10.](#page-23-1) We will prove [\(10\)](#page-23-2) and [\(11\)](#page-23-3) altogether, and then prove [\(12\)](#page-23-4).

*Proof of* [\(10\)](#page-23-2) *and* [\(11\)](#page-23-3)*.* We consider an arbitrary policy <sup>π</sup>. Let {<sup>s</sup> ′ t,h, a′ t,h} H <sup>h</sup>=<sup>1</sup> denote the deterministic trajectory generated by π with initial state s ′ t,<sup>1</sup> = <sup>s</sup>t,1. By definition, we have

$$\begin{aligned}
& V^\pi(s'_{t,1}) - \overline{V}_t(s'_{t,1}) \\
&= Q_1^\pi(s'_{t,1}, \pi(s'_{t,1})) - \max_a \overline{Q}_{t,1}(s'_{t,1}, a) \\
&\leq Q_1^\pi(s'_{t,1}, \pi(s'_{t,1})) - \overline{Q}_{t,1}(s'_{t,1}, \pi(s'_{t,1})) \\
&= V_2^\pi(s'_{t,2}) + r_h(s'_{t,1}, a'_{t,1}) - \langle \overline{\theta}_{t,1}, \phi(s'_{t,1}, \pi(s'_{t,1})) \rangle - \langle \overline{\omega}_{t,h}, \phi(s'_{t,1}, \pi(s'_{t,1})) \rangle \quad (\text{by definition})
\end{aligned}$$

$$= \left( V_2^\pi(s'_{t,2}) - \overline{V}_{t,2}(s'_{t,2}) \right) + \left( \overline{V}_{t,2}(s'_{t,2}) - \langle \overline{\theta}_{t,1}, \phi(s'_{t,1}, \pi(s'_{t,1})) \rangle \right) + \langle \omega_h^* - \overline{\omega}_{t,h}, \phi(s'_{t,1}, a'_{t,1}) \rangle$$

Recursively expanding the first term, we obtain

$$V^\pi(s'_{t,1}) - \overline{V}_t(s'_{t,1}) \leq \sum_{h=1}^H \left( \overline{V}_{t,h+1}(s'_{t,h+1}) - \langle \overline{\theta}_{t,h}, \phi(s'_{t,h}, a'_{t,h}) \rangle + \langle \omega_h^* - \overline{\omega}_{t,h}, \phi(s'_{t,h}, a'_{t,h}) \rangle \right).$$

This proves [\(11\)](#page-23-3) by specifying <sup>π</sup> = <sup>π</sup> ⋆ . Similarly, [\(10\)](#page-23-2) can be proved by observing that the only inequality [\(13\)](#page-23-5) becomes equality when <sup>π</sup> = <sup>π</sup>t.

*Proof of* [\(12\)](#page-23-4)*.* The proof is quite similar. We have

$$\begin{aligned}
& V^{\pi_t}(s_{t,1}) - U_t(s_{t,1}) \\
&= Q_1^{\pi_t}(s_{t,1}, \pi_t(s_{t,1})) - \max_a \overline{Q}_{t,1}(s_{t,1}, a) \\
&\leq Q_1^{\pi_t}(s_{t,1}, \pi_t(s_{t,1})) - \overline{Q}_{t,1}(s_{t,1}, \pi_t(s_{t,1})) \\
&= V_2^{\pi_t}(s_{t,2}) + r_h(s_{t,1}, a_{t,1}) - \langle \overline{\theta}_{t,1}, \phi(s_{t,1}, \pi_t(s_{t,1})) \rangle - \langle \overline{\omega}_{t,h}, \phi(s_{t,1}, a_{t,1}) \rangle \quad (\text{by definition}) \\
&= (V_2^{\pi_t}(s_{t,2}) - U_{t,2}(s_{t,2})) + (U_{t,2}(s_{t,2}) - \langle \overline{\theta}_{t,1}, \phi(s_{t,1}, \pi_t(s_{t,1})) \rangle) + (\omega_t^* - \overline{\omega}_{t,1}, \phi(s_{t,1}, a_{t,1}))
\end{aligned}$$

$$= \left( V_2^{\pi_t}(s_{t,2}) - U_{t,2}(s_{t,2}) \right) + \left( U_{t,2}(s_{t,2}) - \langle \overline{\theta}_{t,1}, \phi(s_{t,1}, \pi_t(s_{t,1})) \rangle \right) + \langle \omega_h^* - \overline{\omega}_{t,h}, \phi(s_{t,1}, a_{t,1}) \rangle$$

Recursively expanding the first term, we obtain

$$V^{\pi_t}(s_{t,1}) - U_t(s_{t,1}) \leq \sum_{h=1}^H \left( U_{t,h+1}(s_{t,h+1}) - \langle \overline{\varrho}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle + \langle \omega_h^* - \overline{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right).$$

This completes the proof.

Lemma 11. *For any* <sup>t</sup> ∈ [T]*, conditioning on* <sup>E</sup> span t *, we have the following (in)equalities:*

$$\begin{aligned} \overline{V}_t(s_{t,1}) &= \sum_{h=1}^H \left( (\widehat{\theta}_{t,h} - \mathcal{T}(\overline{\theta}_{t,h+1} + \overline{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h})) + \langle \overline{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right), \\ U_t(s_{t,1}) &\geq \sum_{h=1}^H \left( (\widehat{\theta}_{t,h} - \mathcal{T}(\overline{\theta}_{t,h+1} + \overline{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h})) + \langle \overline{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right). \end{aligned}$$

Proof of Lemma [11.](#page-24-0) We will prove the two statements separately, but the proofs are quite similar. *Proof of the first statement.* By Lemma [10,](#page-23-1) we have

$$\begin{aligned} & \overline{V}_t(s_{t,1}) - V^{\pi_t}(s_{t,1}) \\ &= \sum_{h=1}^H \left( \left\langle \widehat{\theta}_{t,h}, \phi(s_{t,h}, a_{t,h}) \right\rangle + \langle \xi_{t,h}^P, \phi(s_{t,h}, a_{t,h}) \rangle \right) - \overline{V}_{t,h+1}(s_{t,h+1}) + \langle \overline{\omega}_{t,h} - \omega_h^*, \phi(s_{t,h}, a_{t,h}) \rangle \end{aligned}$$

By linear Bellman completeness (Definition [1\)](#page-2-0), there exists a vector, denoted by T (<sup>θ</sup>t,h+<sup>1</sup> +<sup>ω</sup>t,h+1), such that <sup>V</sup> t,h+1(⋅) = ⟨ϕ(⋅, a), T (<sup>θ</sup>t,h+<sup>1</sup> + <sup>ω</sup>t,h+1)⟩. Hence, we can rewrite the above as

$$\begin{aligned} & \overline{V}_t(s_{t,1}) - V^{\pi_t}(s_{t,1}) \\ &= \sum_{h=1}^H \left( \left( \widehat{\theta}_{t,h} - \mathcal{T}(\overline{\theta}_{t,h+1} + \overline{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h}) \right) + \left\langle \xi_{t,h}^{\text{P}}, \phi(s_{t,h}, a_{t,h}) \right\rangle + \left\langle \overline{\omega}_{t,h} - \omega_h^*, \phi(s_{t,h}, a_{t,h}) \right\rangle \right). \end{aligned}$$

Note that by definition of V <sup>π</sup><sup>t</sup> we have V <sup>π</sup><sup>t</sup> (<sup>s</sup>t,1) = ∑ H <sup>h</sup>=<sup>1</sup> ⟨ω ⋆ , ϕ(<sup>s</sup>t,h, at,h)⟩. Hence, the above implies

$$\overline{V}_t(s_{t,1}) = \sum_{h=1}^H \left( (\widehat{\theta}_{t,h} - \mathcal{T}(\overline{\theta}_{t,h+1} + \overline{\omega}_{t,h+1}) + \xi_{t,h}^P, \phi(s_{t,h}, a_{t,h})) \right) + \langle \overline{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle.$$

*Proof of the second statement.* By Lemma [10,](#page-23-1) we have

$$V^{\pi_t}(s_{t,1}) - U_t(s_{t,1}) \leq \sum_{h=1}^H \left( U_{t,h+1}(s_{t,h+1}) - \langle \bar{\theta}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle + \langle \omega_h^* - \bar{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right)$$

Again, by the definition of V π<sup>t</sup> , we conclude that

$$U_t(s_{t,1}) \geq \sum_{h=1}^H \left( \left\langle \overline{\underline{w}}_{t,h}, \phi(s_{t,h}, a_{t,h}) \right\rangle + \left\langle \widehat{\underline{\theta}}_{t,h} - \mathcal{T}(\overline{\underline{\theta}}_{t,h+1} + \overline{\underline{w}}_{t,h+1}) + \xi_{t,h}^{\text{P}}, \phi(s_{t,h}, a_{t,h}) \right\rangle \right).$$

We can remove ξ P t,h since ⟨<sup>ξ</sup> P t,h, ϕ(<sup>s</sup>t,h, at,h)⟩ = <sup>0</sup> conditioning on <sup>E</sup> span t .

The following lemma shows that, conditioning on the span event E span t , the value function V <sup>t</sup> cannot deviate too much from the value function V <sup>π</sup><sup>t</sup> on average.

Lemma 12. *For any* <sup>t</sup> ∈ [T]*, under* Assumption [<sup>4</sup>](#page-7-0) *and conditioning on* <sup>E</sup> span t *and* E high*, we have*

$$\sum_{t=1}^T \left( \overline{V}_t(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \leq B_{\text{err}}^{\text{P}} \gamma H + (B_{\text{noise}}^{\text{R}} + B_{\text{err}}^{\text{R}}) \cdot B_{\phi}^{\text{R}}.$$

Proof of Lemma [12.](#page-25-1) We apply Lemma [11](#page-24-0) to decompose V <sup>t</sup> and obtain

$$\begin{aligned} & \sum_{t=1}^T \left( \overline{V}_t(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \\ &= \sum_{t=1}^T \left( \langle \widehat{\theta}_{t,h} - \mathcal{T}(\overline{\theta}_{t,h+1} + \overline{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h}) \rangle + \langle \overline{\omega}_{t,h} - \omega_h^*, \phi(s_{t,h}, a_{t,h}) \rangle \right) \end{aligned}$$

Applying Cauchy-Schwartz yields

$$\leq \sum_{t=1}^T \left( \left\| \widehat{\theta}_{t,h} - \mathcal{T}(\overline{\theta}_{t,h+1} + \overline{\omega}_{t,h+1}) \right\|_{\widehat{\Sigma}_{t,h}} \|\phi(s_{t,h}, a_{t,h})\|_{\widehat{\Sigma}_{t,h}^\dagger} + \|\overline{\omega}_{t,h} - \omega_h^*\|_{\Sigma_{t,h}}, \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} \right)$$

We apply Lemma [7](#page-20-1) and Assumption [4](#page-7-0) to the left term and Lemmas [6](#page-18-2) and [16](#page-28-0) and Definition [5](#page-20-0) to the right. Then, we obtain

$$\leq H \cdot B_{\text{err}}^{\text{P}} \gamma + (B_{\text{noise}}^{\text{R}} + B_{\text{err}}^{\text{R}}) \cdot B_{\phi}^{\text{R}}.$$

This completes the proof.

The following lemma establishes upper bounds on the value functions when conditioning on the span event E span t .

Lemma 13. *For any* <sup>t</sup> ∈ [T]*, conditioning on* <sup>E</sup> span t *and* E high*, we have*

$$|U_t(s_{t,1})| \leq H \cdot (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot \sqrt{d} + H \cdot (1 + B_{\text{err}}^P \gamma).$$

*Moreover, we have*

$$|\overline{V}_t(s_{t,1})| \leq H \cdot (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot \sqrt{d} + H \cdot (1 + B_{\text{err}}^P \gamma).$$

*We abbreviate* <sup>B</sup><sup>V</sup> ∶= <sup>H</sup> ⋅ (<sup>B</sup> R noise + <sup>B</sup> R err) ⋅ √ <sup>d</sup> + <sup>H</sup> ⋅ (<sup>1</sup> + <sup>B</sup> P err<sup>γ</sup>)*.*

Proof of Lemma [13.](#page-25-0) We will first prove the second statement and then the first statement.

*Proof of the second statement.* Applying Lemma [11](#page-24-0) and the triangle inequality, we have the following

$$\begin{aligned} |\overline{V}_t(s_{t,1})| &\leq \left| \sum_{h=1}^H \langle \bar{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right| + \left| \sum_{h=1}^H \langle \bar{\theta}_{t,h} - \mathcal{T}(\bar{\theta}_{t,h+1} + \bar{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h}) \rangle \right| \\ &=: \mathbf{T}_1 + \mathbf{T}_2. \end{aligned}$$

We bound the two terms separately. For T1, we have

$$\begin{aligned} \mathbf{T}_1 &= \left| \sum_{h=1}^H \langle (\bar{\omega}_{t,h} - \widehat{\omega}_{t,h}) + (\widehat{\omega}_{t,h} - \omega^*) + \omega_h^*, \phi(s_{t,h}, a_{t,h}) \rangle \right| \\ &\leq \sum_{h=1}^H (\|\bar{\omega}_{t,h} - \widehat{\omega}_{t,h}\|_{\Sigma_{t,h}} + \|\widehat{\omega}_{t,h} - \omega_h^*\|_{\Sigma_{t,h}}) \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} + V^{\pi_t} \quad (\text{Cauchy-Schwartz}) \\ &\leq H \cdot (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot \sqrt{d} + H. \quad (\text{Definition 5 and lemma 5}) \end{aligned}$$

For T2, we can use Cauchy-Schwartz:

$$\begin{aligned} \mathbf{T}_2 &= \left| \sum_{h=1}^H \langle \widehat{\theta}_{t,h} - \mathcal{T}(\bar{\theta}_{t,h+1} + \bar{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h}) \rangle \right| \\ &\leq \sum_{h=1}^H \| \widehat{\theta}_{t,h} - \mathcal{T}(\bar{\theta}_{t,h+1} + \bar{\omega}_{t,h+1}) \|_{\widehat{\Sigma}_{t,h}} \| \phi(s_{t,h}, a_{t,h}) \|_{\widehat{\Sigma}_{t,h}^\dagger} \quad (\text{Cauchy-Schwartz, Lemma 25}) \\ &\leq B_{\text{err}}^{\text{P}} \gamma H. \quad (\text{Assumption 4 and lemma 7}) \end{aligned}$$

*Proof of the first statement.* We prove it by establishing a lower bound and an upper bound of <sup>U</sup>t(<sup>s</sup>t,1) separately. We start with the lower bound, whose derivation is similar to the second statement we just proved above:

$$\begin{aligned}
 U_t(s_{t,1}) &\geq \sum_{h=1}^H \left( \left\langle \widehat{\varrho}_{t,h} - \mathcal{T}(\widehat{\varrho}_{t,h+1} + \bar{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h}) \right\rangle + \langle \bar{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right) \quad (\text{Lemma 11}) \\
 &\geq -B_{\text{err}}^P \gamma H - \left| \sum_{h=1}^H \left( (\bar{\omega}_{t,h} - \bar{\omega}_{t,h}) + (\bar{\omega}_{t,h} - \omega_h^*) + \omega_h^*, \phi(s_{t,h}, a_{t,h}) \right) \right| \\
 &\quad (\text{following a similar argument as above}) \\
 &\geq -B_{\text{err}}^P \gamma H - \sum_{h=1}^H \left( \|\bar{\omega}_{t,h} - \bar{\omega}_{t,h}\|_{\Sigma_{t,h}} + \|\bar{\omega}_{t,h} - \omega_h^*\|_{\Sigma_{t,h}} \right) \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} \\
 &\quad (\text{Cauchy-Schwartz}) \\
 &\geq -B_{\text{err}}^P \gamma H - H \cdot (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot \sqrt{d}. \quad (\text{Lemma 8})
 \end{aligned}$$

The upper bound of <sup>U</sup>t(<sup>s</sup>t,1) is a consequence of the second statement we just proved above:

$$\begin{aligned} U_t(s_{t,1}) &\leq \mathbb{E}[\overline{V}_t(s_{t,1}) \mid \mathfrak{E}^{\text{high}}] && \text{(by definition)} \\ &\leq B_{\text{err}}^{\text{P}} \gamma H + H \cdot (B_{\text{noise}}^{\text{R}} + B_{\text{err}}^{\text{R}}) \cdot \sqrt{d} + H. \end{aligned}$$

We finish the proof by combining the lower and upper bounds.

#### C.3 EXPLORATION IN THE NULL SPACE

Lemma 14 (optimism with constant probability). *For any* <sup>t</sup> ∈ [T]*, denote* <sup>E</sup> optm t *as the event that*

$$V^*(s_{t,1}) \leq \overline{V}_t(s_{t,1}) + B_{\text{err}}^{\text{P}} \gamma H.$$

*Then, under Assumption [4](#page-7-0) and conditioning on the high-probability event* E high*, we have*

$$\Pr \left( \mathfrak{E}_t^{\text{optm}} \right) \geq \Gamma^2(-1)$$

*where* <sup>Γ</sup>(⋅) *is the CDF of the standard normal distribution.*

Proof of Lemma [14.](#page-26-1) By Lemma [10,](#page-23-1) we have:

$$\begin{aligned} V^*(s_{t,1}) - \bar{V}_t(s_{t,1}) &\leq \sum_{h=1}^H \left( \bar{V}_{t,h+1}(s_{t,h+1}^*) - \langle \bar{\theta}_{t,h}, \phi(s_{t,h}^*, a_{t,h}^*) \rangle + \langle \omega_h^* - \bar{\omega}_{t,h}, \phi(s_{t,h}^*, a_{t,h}^*) \rangle \right) \\ &= \underbrace{\sum_{h=1}^H \left( \bar{V}_{t,h+1}(s_{t,h+1}^*) - \langle \widehat{\theta}_{t,h}, \phi(s_{t,h}^*, a_{t,h}^*) \rangle \right)}_{(i)} - \underbrace{\sum_{h=1}^H \langle \xi_{t,h}^P, \phi(s_{t,h}^*, a_{t,h}^*) \rangle}_{(ii)} \end{aligned}$$

$$+ \underbrace{\sum_{h=1}^H \langle \omega_{t,h}^* - \bar{\omega}_{t,h}, \phi(s_{t,h}^*, a_{t,h}^*) \rangle}_{\text{(iii)}} - \underbrace{\sum_{h=1}^H \langle \xi_{t,h}^R, \phi(s_{t,h}^*, a_{t,h}^*) \rangle}_{\text{(iv)}}.$$

Note that, given any state-action-state triple (s, a, s′ ), we have

$$\overline{V}_{t,h+1}(s') - \langle \widehat{\theta}_{t,h}, \phi(s, a) \rangle = \langle \mathcal{T}(\overline{\omega}_{t,h+1} + \overline{\theta}_{t,h+1}) - \widehat{\theta}_{t,h}, \phi(s, a) \rangle = \langle \eta_{t,h}, \phi(s, a) \rangle.$$

Plugging this back to (i), we obtain

$$(i) - (ii) \leq \sum_{h=1}^H \langle \eta_{t,h} - \xi_{t,h}^P, \phi(s_{t,h}^*, a_{t,h}^*) \rangle =: \sum_{h=1}^H \langle \eta_{t,h} - \xi_{t,h}^P, \phi_h^* \rangle$$

where we abbreviate ϕ ⋆ h ∶= <sup>ϕ</sup>(<sup>s</sup> ⋆ t,h, a<sup>⋆</sup> t,h). Next, we split it into two parts:

$$\begin{aligned}
 (i) - (ii) &\leq \sum_{h=1}^H \langle \eta_{t,h}, P_{t,h} \phi_h^* \rangle + \sum_{h=1}^H \langle \eta_{t,h}, (I - P_{t,h}) \phi_h^* \rangle - \sum_{h=1}^H \langle \xi_{t,h}^P, \phi_h^* \rangle \\
 &\leq \sum_{h=1}^H \|\eta_{t,h}\|_{\Sigma_{t,h}} \|P_{t,h} \phi_h^*\|_{\Sigma_{t,h}^*} + \sum_{h=1}^H \|\eta_{t,h}\|_{\Lambda_{t,h}} \|(I - P_{t,h}) \phi_h^*\|_{\Lambda_{t,h}^{-1}} - \sum_{h=1}^H \langle \xi_{t,h}^P, \phi_h^* \rangle \\
 &\quad (\text{Cauchy-Schwartz, Lemma 25}) \\
 &\leq B_{\text{err}}^P \gamma H + \sum_{h=1}^H \|\eta_{t,h}\|_{\Lambda_{t,h}} \|(I - P_{t,h}) \phi_h^*\|_{\Lambda_{t,h}^{-1}} - \sum_{h=1}^H \langle \xi_{t,h}^P, \phi_h^* \rangle \\
 &\quad (\text{Assumption 4 and Lemmas 7 and 26}) \\
 &\leq B_{\text{err}}^P \gamma H + \sqrt{H \sum_{h=1}^H \|\eta_{t,h}\|_{\Lambda_{t,h}}^2 \|(I - P_{t,h}) \phi_h^*\|_{\Lambda_{t,h}^{-1}}^2} - \sum_{h=1}^H \langle \xi_{t,h}^P, \phi_h^* \rangle \quad (\text{Cauchy-Schwartz})
 \end{aligned}$$

Recall that ξ P t,h is sampled from N (0, σ<sup>2</sup> h (<sup>I</sup> − <sup>P</sup>t,h)<sup>Λ</sup> −1 t,h(<sup>I</sup> − <sup>P</sup>t,h)). Therefore,

$$\sum_{h=1}^H \langle \xi_{t,h}^P, \phi_h^* \rangle \sim \mathcal{N} \left( 0, \sum_{h=1}^H \sigma_h^2 \| (I - P_{t,h}) \phi_h^* \|_{\Lambda_{t,h}^{-1}}^2 \right).$$

Since <sup>σ</sup><sup>h</sup> ≥ √ <sup>H</sup>∥<sup>η</sup>t,h∥<sup>Λ</sup>t,h under high-probability event <sup>E</sup> high, we have

$$\Pr \left( (i) - (ii) \leq B_{\text{err}}^P \gamma H \right) \geq \Gamma(-1).$$

Next, we consider (iii) − (iv). By a similar argument, we have

$$\begin{aligned} (\text{iii}) - (\text{iv}) &= \sum_{h=1}^H \langle \omega_h^* - \widehat{\omega}_{t,h}, \phi_h^* \rangle - \sum_{h=1}^H \langle \xi_{t,h}^R, \phi_h^* \rangle \\ &\leq \sum_{h=1}^H \|\omega_h^* - \widehat{\omega}_{t,h}\|_{\Sigma_{t,h}} \|\phi_h^*\|_{\Sigma_{t,h}^{-1}} - \sum_{h=1}^H \langle \xi_{t,h}^R, \phi_h^* \rangle \\ &\leq \sqrt{H \cdot \sum_{h=1}^H \|\omega_h^* - \widehat{\omega}_{t,h}\|_{\Sigma_{t,h}}^2 \|\phi_h^*\|_{\Sigma_{t,h}^{-1}}^2} - \sum_{h=1}^H \langle \xi_{t,h}^R, \phi_h^* \rangle. \end{aligned}$$

Recall that ξ R t is sampled from N (0, σ<sup>2</sup> <sup>R</sup>Σ −1 t,h), and thus, we have

$$\sum_{h=1}^H \langle \xi_t^R, \phi_h^* \rangle \sim \mathcal{N} \left( 0, \sum_{h=1}^H \sigma_R^2 \|\phi_h^*\|_{\Sigma_{t,h}^{-1}}^2 \right).$$

Therefore, since <sup>σ</sup><sup>R</sup> ≥ √ <sup>H</sup>∥<sup>ω</sup> ⋆ <sup>h</sup> − <sup>ω</sup>̂t,h∥<sup>Σ</sup><sup>t</sup> (Lemma [9\)](#page-21-0), we have

$$\Pr \left( (\text{iii}) - (\text{iv}) \leq 0 \right) \geq \Gamma(-1).$$

Lemma 15. *The number of times* E span t *does not hold will not exceed* dH*, i.e.,*

$$\sum_{t=1}^T \mathbf{1} \left\{ (\mathbf{e}_t^{\text{span}})^C \right\} \leq dH.$$

Proof. By definition, when E span t does not hold, there exists <sup>h</sup> ∈ [H] such that <sup>ϕ</sup>(<sup>s</sup>t,h, at,h) is not in the span of {ϕ(<sup>s</sup>i,h, ai,h)}t−<sup>1</sup> <sup>i</sup>=<sup>1</sup> . That means, the dimension of the span should increase by exactly one after this iteration, i.e.,

$$\dim \left( \text{span} \left( \{ \phi(s_{i,h}, a_{i,h}) \}_{i=1}^t \right) \right) = \dim \left( \text{span} \left( \{ \phi(s_{i,h}, a_{i,h}) \}_{i=1}^{t-1} \right) \right) + 1.$$

However, the dimension cannot exceed d, so it can only increase at most d times. This argument holds for any <sup>h</sup> ∈ [H], and thus, the total number of times <sup>E</sup> span t does not happen will not exceed dH.

Lemma 16. *For any* <sup>h</sup> ∈ [H]*, it holds that*

$$\begin{aligned} \sum_{t=1}^T \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} &\leq d\sqrt{2T \log(T+1)} =: B_{\phi}^R, \\ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} \|\phi(s_{t,h}, a_{t,h})\|_{\widehat{\Sigma}_{t,h}^i} &\leq \gamma d\sqrt{2dT \log(2T\gamma^2)} =: B_{\phi}^{\text{P}}. \end{aligned}$$

Proof of Lemma [16.](#page-28-0) We prove the two inequalities separately.

*Proof of the first inequality.* For any <sup>t</sup> ∈ [T] and <sup>h</sup> ∈ [H], we have the following bound on the norm of features (Lemma [5\)](#page-18-1):

$$\|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} \leq \|\phi(s_{t,h}, a_{t,h})\|_{\Lambda^{-1}} \leq \sqrt{d}.$$

Hence, by Cauchy-Schwartz, we have

$$\begin{aligned} \sum_{t=1}^T \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} &\leq \sqrt{T \cdot \sum_{t=1}^T \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}}^2} \\ &= \sqrt{T \cdot \sum_{t=1}^T \min \left\{ \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}}^2, d \right\}} \\ &\leq \sqrt{T d \cdot \sum_{t=1}^T \min \left\{ \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}}^2, 1 \right\}} \\ &\leq \sqrt{Td \cdot 2d \log(T+1)} \quad (\text{elliptical potential lemma, Lemma 21}) \\ &= d\sqrt{2T \log(T+1)}. \end{aligned}$$

*Proof of the second inequality.* We divide the rounds into d consecutive blocks, in each of which the rank of <sup>Σ</sup>̂t,h remains the same. To be specific, let <sup>t</sup>1, t2, . . . , td, td+<sup>1</sup> be a sequence of integers such that for any <sup>i</sup> ∈ [d] and any <sup>t</sup> ∈ {<sup>t</sup><sup>i</sup> , ti+1, . . . , ti+<sup>1</sup> − <sup>1</sup>}, we have rank(Σ̂t,h) = <sup>i</sup>.

We will apply the elliptical potential lemma to each block separately. Now let's fix <sup>i</sup> ∈ [d] and consider the <sup>i</sup>-th block. Let the reduced eigen-decomposition of <sup>Σ</sup>̂<sup>t</sup>i,h be <sup>Σ</sup>̂<sup>t</sup>i,h = UDU<sup>⊺</sup> where <sup>U</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>i</sup> and <sup>D</sup> ∈ <sup>R</sup> <sup>i</sup>×<sup>i</sup> . For each <sup>t</sup> ∈ {<sup>t</sup><sup>i</sup> , ti+1, . . . , ti+<sup>1</sup> − <sup>1</sup>}, since <sup>ϕ</sup>(<sup>s</sup>t,h, at,h) is in the span of <sup>Σ</sup>̂t,h conditioning on E span t , there exists a vector <sup>x</sup><sup>t</sup> such that <sup>ϕ</sup>(<sup>s</sup>t,h, at,h) = Uxt.

For any <sup>t</sup> ∈ {<sup>t</sup><sup>i</sup> , ti+1, . . . , ti+<sup>1</sup> − <sup>1</sup>}, we have

$$\begin{aligned}\|\phi(s_{t,h}, a_{t,h})\|_{\widehat{\Sigma}_{t,h}^2} &= \phi(s_{t,h}, a_{t,h})^\top \widehat{\Sigma}_{t,h}^\dagger \phi(s_{t,h}, a_{t,h}) \\ &= \phi(s_{t,h}, a_{t,h})^\top \left( \widehat{\Sigma}_{t_i, h} + \sum_{j=t_i}^{t-1} \phi(s_{j,h}, a_{j,h}) \phi^\top(s_{j,h}, a_{j,h}) \right)^\dagger \phi(s_{t,h}, a_{t,h})\end{aligned}$$

$$\begin{aligned} &= x_t^\top U^\top \left( UDU^\top + \sum_{j=t_i}^{t-1} Ux_j x_j^\top U^\top \right)^\dagger Ux_t \\ &= x_t^\top \left( D + \sum_{j=t_i}^{t-1} x_j x_j^\top \right)^{-1} x_t. \end{aligned}$$

Define <sup>D</sup><sup>t</sup> = <sup>D</sup> + ∑ <sup>t</sup>−<sup>1</sup> <sup>j</sup>=<sup>t</sup><sup>i</sup> xjx ⊺ j . Hence, we have

$$\sum_{t=t_i}^{t_{i+1}-1} \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} \|\phi(s_{t,h}, a_{t,h})\|_{\widehat{\mathcal{S}}_{t,h}^\dagger} = \sum_{t=t_i}^{t_{i+1}-1} \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} \|x_t\|_{D_t^{-1}}.$$

By Assumption [4,](#page-7-0) the eigenvalues of <sup>D</sup> are lower bounded by <sup>1</sup>/<sup>γ</sup> 2 . And clearly, its eigenvalues are upper bounded by <sup>t</sup><sup>i</sup> ≤ <sup>T</sup>. Therefore, we have

$$\begin{aligned} \sum_{t=t_i}^{t_{i+1}-1} \mathbf{1}\{\mathbf{E}_t^{\text{span}}\} \|x_t\|_{D_{\tau^{-1}}^{-1}} &\leq \sqrt{T \cdot \sum_{t=t_i}^{t_{i+1}-1} \mathbf{1}\{\mathbf{E}_t^{\text{span}}\} \|x_t\|_{D_{\tau^{-1}}^{-1}}^2} \\ &= \sqrt{T \cdot \sum_{t=t_i}^{t_{i+1}-1} \mathbf{1}\{\mathbf{E}_t^{\text{span}}\} \min \left\{ \|x_t\|_{D_{\tau^{-1}}^{-1}}^2, \gamma^2 \right\}} \\ &\leq \gamma \sqrt{T \cdot \sum_{t=t_i}^{t_{i+1}-1} \mathbf{1}\{\mathbf{E}_t^{\text{span}}\} \min \left\{ \|x_t\|_{D_{\tau^{-1}}^{-1}}^2, 1 \right\}} \\ &\leq \gamma \sqrt{T \cdot 2d \log(T\gamma^2(1+1/d))} \quad (\text{elliptical potential lemma, Lemma 2.1}) \\ &\leq \gamma \sqrt{T \cdot 2d \log(2T\gamma^2)}. \end{aligned}$$

This finishes the summation of one block. Notice that we have d such blocks, we complete the proof by multiplying the above by d.

#### C.4 MAIN STEPS OF THE PROOF

Let <sup>V</sup>̃t(<sup>s</sup>t,1) denote an i.i.d. copy of <sup>V</sup> <sup>t</sup> conditioned on initial state <sup>s</sup>t,<sup>1</sup> and <sup>E</sup>̃optm t and <sup>E</sup>̃high denote the counterparts of E optm t and E high but for <sup>V</sup>̃t(<sup>s</sup>t,1).

The proof starts with the following decomposition of the regret:

$$\begin{aligned}\mathbb{E} \left[ \sum_{t=1}^T \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] &\leq \mathbb{E} \left[ \mathbf{1}\{(\mathbf{e}^{\text{high}}) \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] \\ &\quad + \mathbb{E} \left[ \mathbf{1}\{((\mathbf{e}^{\text{high}})^C) \sum_{t=1}^T \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] \\ &\quad + \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{((\mathbf{e}^{\text{span}})^C) \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right]\end{aligned}$$

We will later show that the second and third terms can be easily bounded separately by observing the following two fact: (1) the probability that E high doesn't hold is very small, and (2) the number of times E span t doesn't hold is also small. Hence, it remains to bound the first term, which is the most challenging. The most of the proof below is devoted to bounding it.

As the first step, we will add some necessary event conditions to the first term, using the following lemma.

Lemma 17 (Adding necessary event conditions). *It holds that*

$$\begin{aligned} & \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] \\ & \leq \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E} \left[ \mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}} \cap \mathbf{e}^{\text{high}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap \widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} U_t(s_{t,1}) \right] \right] \end{aligned}$$

$$+ \frac{1}{\Gamma^2(-1)} \cdot \left( dHB_V + B_{\text{err}}^{\text{P}} \gamma H + (B_{\text{noise}}^{\text{R}} + B_{\text{err}}^{\text{R}}) \cdot B_{\phi}^{\text{R}} + dH^2 + 1 \right)$$

*where the expectation* <sup>E</sup>Ṽ<sup>t</sup> *is taken over the randomness of* <sup>V</sup>̃<sup>t</sup> *(an i.i.d. copy of* <sup>V</sup> <sup>t</sup>*) only.*

Proof of Lemma [17.](#page-29-1) We have

$$\begin{aligned} & \mathbb{E} \left[ \mathbf{1}\{\mathbf{E}^{\text{high}}\} \sum_{t=1}^T \mathbf{1}\{\mathbf{E}_t^{\text{span}}\} \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] \\ & \leq \mathbb{E} \left[ \mathbf{1}\{\mathbf{E}^{\text{high}}\} \sum_{t=1}^T (V^*(s_{t,1}) - \mathbf{1}\{\mathbf{E}_t^{\text{span}}\} V^{\pi_t}(s_{t,1})) \right] \quad (V^* \text{ is non-negative}) \end{aligned}$$

Plugging the condition on <sup>E</sup>̃optm t (Lemma [14\)](#page-26-1), we get

$$\leq \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E} \left[ \min\{H, \tilde{V}_t(s_{t,1})\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} V^{\pi_t}(s_{t,1}) \mid \tilde{\mathbf{e}}_t^{\text{optm}} \right] \right] + B_{\text{err}}^{\text{P}} \gamma H$$

We aim to add two event indicators, <sup>E</sup>̃high and <sup>E</sup>̃span t , and thus split the whole thing into several terms:

$$\begin{aligned} &\leq \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{\tilde{\mathbf{e}}^{\text{high}} \cap \tilde{\mathbf{e}}_t^{\text{span}}\} (\tilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} V^{\pi_t}(s_{t,1})) | \tilde{\mathbf{e}}_t^{\text{optm}}] \right] \\ &\quad + \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{(\tilde{\mathbf{e}}^{\text{high}})^C\} (\min\{H, \tilde{V}_t(s_{t,1})\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} V^{\pi_t}(s_{t,1})) | \tilde{\mathbf{e}}_t^{\text{optm}}] \right] \\ &\quad + \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{(\tilde{\mathbf{e}}_t^{\text{span}})^C\} (\min\{H, \tilde{V}_t(s_{t,1})\} - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} V^{\pi_t}(s_{t,1})) | \tilde{\mathbf{e}}_t^{\text{optm}}] \right] \\ &\quad + B_{\text{err}}^P \gamma H \\ &=: \mathbf{T}_1 + \mathbf{T}_2 + \mathbf{T}_3 + B_{\text{err}}^P \gamma H. \end{aligned}$$

Below we bound each term separately.

*Bounding* <sup>T</sup>1*.* To bound <sup>T</sup>1, we will first drop the conditioning event <sup>E</sup>̃optm t to make things cleaner. To that end, we re-arange it in the following way

$$\begin{aligned} \mathbf{T}_1 &= \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathbf{e}}^{\text{high}} \cap \tilde{\mathbf{e}}_t^{\text{span}}\} (\tilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1})) + \mathbf{1}\{(\mathbf{e}_t^{\text{span}})^C\} \cdot B_V \mid \tilde{\mathbf{e}}_t^{\text{optm}} \right] \right. \\ &\quad \left. + \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathbf{e}}^{\text{high}} \cap \tilde{\mathbf{e}}_t^{\text{span}}\} \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} (U_t(s_{t,1}) - V^{\pi_t}(s_{t,1})) \mid \tilde{\mathbf{e}}_t^{\text{optm}} \right] \right] \right. \\ &\quad \left. - \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{(\mathbf{e}_t^{\text{span}})^C\} \cdot B_V \mid \tilde{\mathbf{e}}_t^{\text{optm}} \right] \right] \right. \\ &=: \mathbf{T}_{1.1} + \mathbf{T}_{1.2} + \mathbf{T}_{1.3}. \end{aligned}$$

The reason we did this is that we want to make sure (∗) is non-negative, so we can drop the conditioning event <sup>E</sup>̃optm t . To see why it is non-negative, we consider two cases: first, if E span t holds, then we already have <sup>1</sup>{Ẽhigh}(Ṽt(<sup>s</sup>t,1) − <sup>1</sup>{<sup>E</sup> span t }<sup>U</sup>t(<sup>s</sup>t,1)) ≥ <sup>0</sup> by definition of <sup>U</sup>t(<sup>s</sup>t,1); second, if E span t does not hold, then we have <sup>1</sup>{Ẽhigh ∩ <sup>E</sup>̃span t }Ṽt(<sup>s</sup>t,1) + <sup>1</sup>{(<sup>E</sup> span t ) <sup>∁</sup>} ⋅ <sup>B</sup><sup>V</sup> ≥ <sup>0</sup> by Lemma [13.](#page-25-0)

Hence, for T1.1, we can drop the conditioning event using the following rule (for non-negative variable X):

$$\mathbb{E}[X \mid \mathfrak{E}] = \mathbb{E}[X \cdot \mathbf{1}\{\mathfrak{E}\}] / \Pr(\mathfrak{E}) \leq \mathbb{E}[X] / \Pr(\mathfrak{E})$$

$$\mathbf{T}_{1,1} \leq \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \mathbb{E}_{\widetilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} (\widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1})) + \mathbf{1}\{(\mathbf{e}_t^{\text{span}})^c\} \cdot B_V \right] \right]$$

$$\begin{aligned}
&= \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{\mathbf{\mathfrak{E}}^{\text{high}}\} \sum_{t=1}^T \mathbb{E} \left[ \mathbf{1}\{\widetilde{\mathbf{\mathfrak{E}}}^{\text{high}} \cap \widetilde{\mathbf{\mathfrak{E}}}^{\text{span}}\} (\widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{\mathfrak{E}}^{\text{span}}\} U_t(s_{t,1})) \right] \right] \\
&\quad + \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{\mathbf{\mathfrak{E}}^{\text{high}}\} \sum_{t=1}^T \mathbf{1}\{(\mathbf{\mathfrak{E}}^{\text{span}})^C\} \cdot B_V \right] \\
&\leq \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{\mathbf{\mathfrak{E}}^{\text{high}}\} \sum_{t=1}^T \mathbb{E} \left[ \mathbf{1}\{\widetilde{\mathbf{\mathfrak{E}}}^{\text{high}} \cap \widetilde{\mathbf{\mathfrak{E}}}^{\text{span}}\} (\widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{\mathfrak{E}}^{\text{span}}\} U_t(s_{t,1})) \right] \right] \\
&\quad + \frac{1}{\Gamma^2(-1)} \cdot dH B_V \tag{Lemma 15}
\end{aligned}$$

For T1.2, we apply Lemma [12](#page-25-1) to get

$$\begin{aligned} T_{1.2} &\leq \mathbb{E} \left[ \mathbf{1}\{\mathfrak{E}^{\text{high}}\} \sum_{t=1}^T \frac{\mathbb{E}}{\widetilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathfrak{E}}^{\text{high}} \cap \widetilde{\mathfrak{E}}_t^{\text{span}}\} \mathbf{1}\{\mathfrak{E}_t^{\text{span}}\} (\overline{V}_t(s_{t,1}) - V^{\pi_t}(s_{t,1})) \mid \widetilde{\mathfrak{E}}_t^{\text{optm}} \right] \right] \\ &\quad (\overline{V}_t \geq U_t \text{ conditioning on } \mathfrak{E}^{\text{high}}) \\ &\leq B_{\text{err}}^P \gamma H + (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot B_{\phi}^R. \end{aligned}$$

We simply upper bound T1.<sup>3</sup> by zero. Plugging all these upper bounds back, we obtain

$$\begin{aligned} T_1 &\leq \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \mathbf{1}\{\mathbf{e}^{\text{high}}\} \sum_{t=1}^T \frac{\mathbb{E}}{\tilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} (\widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}}\} U_t(s_{t,1})) \right] \right] \\ &\quad + \frac{1}{\Gamma^2(-1)} \cdot dHB_V + B_{\text{err}}^P \gamma H + (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot B_\phi^R \\ &= \frac{1}{\Gamma^2(-1)} \mathbb{E} \left[ \sum_{t=1}^T \frac{\mathbb{E}}{\tilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}} \cap \mathbf{e}^{\text{high}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap \widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} U_t(s_{t,1}) \right] \right] \\ &\quad + \frac{1}{\Gamma^2(-1)} \cdot dHB_V + B_{\text{err}}^P \gamma H + (B_{\text{noise}}^R + B_{\text{err}}^R) \cdot B_\phi^R \end{aligned}$$

This is the final bound of T<sup>1</sup> we need. Next, we go back to bound T<sup>2</sup> and T3.

*Bounding* T2*.* We upper bound the value function inside the expectation by H and obtain

$$\begin{aligned} T_2 &\leq H \cdot \mathbb{E} \left[ \mathbf{1}\{(\mathfrak{E}^{\text{high}})\} \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{((\widetilde{\mathfrak{E}}^{\text{high}})^{\mathbb{C}}) \mid \widetilde{\mathfrak{E}}_t^{\text{optm}}\}] \right] \\ &\leq H \cdot \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{((\widetilde{\mathfrak{E}}^{\text{high}})^{\mathbb{C}}) \mid \widetilde{\mathfrak{E}}_t^{\text{optm}}\}] \right] \quad (\text{dropping } \mathfrak{E}^{\text{high}}) \\ &= H \cdot \mathbb{E} \left[ \sum_{t=1}^T \Pr \left( ((\widetilde{\mathfrak{E}}^{\text{high}})^{\mathbb{C}} \cap \widetilde{\mathfrak{E}}_t^{\text{optm}}) / \Pr \left( \widetilde{\mathfrak{E}}_t^{\text{optm}} \right) \right) \right] \\ &\leq \frac{HT}{\Gamma^2(-1)} \cdot \Pr \left( ((\mathfrak{E}^{\text{high}})^{\mathbb{C}}) \right. \\ &\leq \frac{1}{\Gamma^2(-1)}. \quad (\text{Lemma 8}) \end{aligned}$$

*Bounding* T3*.* Similar, we upper bound the value function inside the expectation by H and obtain

$$\begin{aligned} T_3 &\leq H \cdot \mathbb{E} \left[ \mathbf{1}\{(\mathfrak{E}_t^{\text{high}}) \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{(\widetilde{\mathfrak{E}}_t^{\text{span}})^C\} \mid \widetilde{\mathfrak{E}}_t^{\text{optm}}]\} \right] \\ &\leq H \cdot \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{(\widetilde{\mathfrak{E}}_t^{\text{span}})^C\} \mid \widetilde{\mathfrak{E}}_t^{\text{optm}}] \right] \quad (\text{dropping } \mathfrak{E}_t^{\text{high}}) \\ &= H \cdot \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{(\widetilde{\mathfrak{E}}_t^{\text{span}})^C \cap \widetilde{\mathfrak{E}}_t^{\text{optm}}\}] / \Pr(\widetilde{\mathfrak{E}}_t^{\text{optm}}) \right] \\ &\leq H \cdot \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E} [\mathbf{1}\{(\widetilde{\mathfrak{E}}_t^{\text{span}})^C\}] / \Pr(\widetilde{\mathfrak{E}}_t^{\text{optm}}) \right] \end{aligned}$$

$$\begin{aligned} &\leq \frac{H}{\Gamma^2(-1)} \cdot \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{(\mathbf{E}_t^{\text{span}})^C\} \right] && \text{(tower rule)} \\ &\leq \frac{dH^2}{\Gamma^2(-1)} && \text{(Lemma 15)} \end{aligned}$$

Plugging all these back, we conclude the proof.

The following lemma refines the event conditions established in Lemma [17](#page-29-1) to make the whole thing more manageable.

Lemma 18 (Refining event conditions). *It holds that*

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathfrak{E}}^{\text{high}} \cap \widetilde{\mathfrak{E}}_t^{\text{span}} \cap \mathfrak{E}^{\text{high}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathfrak{E}_t^{\text{span}} \cap \mathfrak{E}^{\text{high}} \cap \widetilde{\mathfrak{E}}^{\text{high}} \cap \widetilde{\mathfrak{E}}_t^{\text{span}}\} U_t(s_{t,1}) \right] \right] \\ & \leq \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathfrak{E}}^{\text{high}} \cap \widetilde{\mathfrak{E}}_t^{\text{span}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathfrak{E}_t^{\text{span}} \cap \mathfrak{E}^{\text{high}}\} U_t(s_{t,1}) \right] \right] \\ & \quad + dHB_V + 2B_V/H. \end{aligned}$$

Proof of Lemma [18.](#page-32-0) We start with refining the event conditions on the first term. We remove unneeded events by splitting the first term into two parts:

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\widetilde{V}_t} [\mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}} \cap \mathbf{e}^{\text{high}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap \widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} U_t(s_{t,1})] \right] \\ &= \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\widetilde{V}_t} [\mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap \widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} U_t(s_{t,1})] \right] \\ &\quad - \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\widetilde{V}_t} [\mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}} \cap (\mathbf{e}^{\text{high}})^c\} \widetilde{V}_t(s_{t,1})] \right] \end{aligned}$$

Here, using Lemma [13,](#page-25-0) the last term can be bounded by

$$-\mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\widetilde{V}_t} \left[ \mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}^{\text{span}} \cap (\mathbf{e}^{\text{high}})^{\mathcal{C}}\} \widetilde{V}_t(s_{t,1}) \right] \right] \leq \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{(\mathbf{e}^{\text{high}})^{\mathcal{C}}\} B_V \right] \leq B_V / H.$$

where we used Lemma [8](#page-20-2) in the last inequality.

Now we seek to remove unneeded event conditions on U<sup>t</sup> as well. We notice the following decomposition

$$\begin{aligned} & 1\{ \{ \mathfrak{C}_t^{\text{span}} \cap \mathfrak{C}^{\text{high}} \cap \widetilde{\mathfrak{C}}^{\text{high}} \cap \widetilde{\mathfrak{C}}_t^{\text{span}} \} U_t(s_{t,1}) \\ & \geq 1\{ \{ \mathfrak{C}_t^{\text{span}} \cap \mathfrak{C}^{\text{high}} \} U_t(s_{t,1}) \\ & \quad - 1\{ \{ \mathfrak{C}_t^{\text{span}} \cap \mathfrak{C}^{\text{high}} \cap (\widetilde{\mathfrak{C}}^{\text{high}})^{\mathbb{C}} \} U_t(s_{t,1}) \\ & \quad - 1\{ \{ \mathfrak{C}_t^{\text{span}} \cap \mathfrak{C}^{\text{high}} \cap (\widetilde{\mathfrak{C}}_t^{\text{span}})^{\mathbb{C}} \} U_t(s_{t,1}). \end{aligned}$$

Plugging this back, we obtain

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathfrak{E}}^{\text{high}} \cap \tilde{\mathfrak{E}}_t^{\text{span}} \cap \mathfrak{e}^{\text{high}}\} \tilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathfrak{e}_t^{\text{span}} \cap \mathfrak{e}^{\text{high}} \cap \tilde{\mathfrak{E}}^{\text{high}} \cap \tilde{\mathfrak{E}}_t^{\text{span}}\} U_t(s_{t,1}) \right] \right] \\ & \leq \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\tilde{\mathfrak{E}}^{\text{high}} \cap \tilde{\mathfrak{E}}_t^{\text{span}}\} \tilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathfrak{e}_t^{\text{span}} \cap \mathfrak{e}^{\text{high}}\} U_t(s_{t,1}) \right] \right] \\ & + \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\mathfrak{e}_t^{\text{span}} \cap \mathfrak{e}^{\text{high}} \cap (\tilde{\mathfrak{E}}^{\text{high}})^{\mathbb{C}}\} U_t(s_{t,1}) \right] \right] \\ & + \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\mathfrak{e}_t^{\text{span}} \cap \mathfrak{e}^{\text{high}} \cap (\tilde{\mathfrak{E}}_t^{\text{span}})^{\mathbb{C}}\} U_t(s_{t,1}) \right] \right] \\ & + B_V / H \end{aligned}$$

The first term is exactly what we want. Now we bound the middle two terms separately below:

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap (\tilde{\mathbf{e}}^{\text{high}})^{\mathbb{C}}\} U_t(s_{t,1}) \right] \right] \\ & \leq \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap (\tilde{\mathbf{e}}^{\text{high}})^{\mathbb{C}}\} B_V \right] \right] \quad (\text{Lemma 13}) \\ & \leq T \cdot \Pr((\tilde{\mathbf{e}}^{\text{high}})^{\mathbb{C}}) B_V \\ & \leq B_V / H \quad (\text{Lemma 8}) \end{aligned}$$

and for the other term we also have

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}} \cap (\tilde{\mathbf{e}}_t^{\text{span}})^C\} U_t(s_{t,1}) \right] \right] \\ & \leq \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\tilde{V}_t} \left[ \mathbf{1}\{(\tilde{\mathbf{e}}_t^{\text{span}})^C\} \right] B_V \right] \quad (\text{Lemma 13}) \\ & = B_V \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{(\mathbf{e}_t^{\text{span}})^C\} \right] \quad (\text{tower rule}) \\ & \leq dH B_V. \quad (\text{Lemma 15}) \end{aligned}$$

Hence, putting all together, we complete the proof.

The following lemma provides a final bound for the first term in Lemma [18.](#page-32-0)

Lemma 19 (Final bound). *It holds that*

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E} \left[ \mathbf{1}\{\widetilde{\mathfrak{C}}^{\text{high}} \cap \widetilde{\mathfrak{C}}_t^{\text{span}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathfrak{C}_t^{\text{span}} \cap \mathfrak{C}^{\text{high}}\} U_t(s_{t,1}) \right] \right] \\ & \leq 2H B_{\text{err}}^{\text{P}} B_{\phi}^{\text{P}} + 2(B_{\text{err}}^{\text{R}} + B_{\text{noise}}^{\text{R}}) \cdot H B_{\phi}^{\text{R}}. \end{aligned}$$

Proof of Lemma [19.](#page-33-0) By tower rule, we have

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T \mathbb{E}_{\widetilde{V}_t} [\mathbf{1}\{\widetilde{\mathbf{e}}^{\text{high}} \cap \widetilde{\mathbf{e}}_t^{\text{span}}\} \widetilde{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}}\} U_t(s_{t,1})] \right] \\ &= \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \overline{V}_t(s_{t,1}) - \mathbf{1}\{\mathbf{e}_t^{\text{span}} \cap \mathbf{e}^{\text{high}}\} U_t(s_{t,1}) \right] \end{aligned}$$

We plug in the result in Lemma [11](#page-24-0) and get

$$\begin{aligned} &\leq \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \sum_{h=1}^H \langle \widehat{\theta}_{t,h} - \mathcal{T}(\widehat{\theta}_{t,h+1} + \bar{\omega}_{t,h+1}), \phi(s_{t,h}, a_{t,h}) \rangle \right] \\ &\quad + \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \sum_{h=1}^H \langle \mathcal{T}(\widehat{\theta}_{t,h+1} + \bar{\omega}_{t,h+1}) - \widehat{\theta}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right] \\ &\quad + \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \sum_{h=1}^H \langle \bar{\omega}_{t,h} - \bar{\omega}_{t,h}, \phi(s_{t,h}, a_{t,h}) \rangle \right] \end{aligned}$$

Applying Cauchy-Schwartz inequality to each term yields

$$\begin{aligned} &\leq \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \sum_{h=1}^H \|\widehat{\theta}_{t,h} - \mathcal{T}(\bar{\theta}_{t,h+1} + \bar{\omega}_{t,h+1})\|_{\Sigma_{t,h}} \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^\dagger} \right] \\ &\quad + \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \sum_{h=1}^H \|\mathcal{T}(\bar{\theta}_{t,h+1} + \bar{\omega}_{t,h+1}) - \widehat{\theta}_{t,h}\|_{\Sigma_{t,h}} \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^\dagger} \right] \\ &\quad + \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{\mathbf{e}_t^{\text{high}} \cap \mathbf{e}_t^{\text{span}}\} \sum_{h=1}^H \left( \|\bar{\omega}_{t,h} - \omega_h^*\|_{\Sigma_{t,h}} + \|\omega_h^* - \bar{\omega}_{t,h}\|_{\Sigma_{t,h}} \right) \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} \right]. \end{aligned}$$

The first two terms can be bounded by HB<sup>P</sup> errB P ϕ using Lemmas [7](#page-20-1) and [16.](#page-28-0) For the last term, conditioning on E high, we have

$$\|\bar{\omega}_{t,h} - \omega_h^*\|_{\Sigma_{t,h}} \leq \|\bar{\omega}_{t,h} - \widehat{\omega}_{t,h}\|_{\Sigma_{t,h}} + \|\widehat{\omega}_{t,h} - \omega_h^*\|_{\Sigma_{t,h}} \leq B_{\text{err}}^{\text{R}} + B_{\text{noise}}^{\text{R}}$$

and similarly for ∥<sup>ω</sup> ⋆ <sup>h</sup> − <sup>ω</sup>t,h∥<sup>Σ</sup>t,h . Also, applying Lemma [16,](#page-28-0) we have

$$\sum_{t=1}^T \sum_{h=1}^H \|\phi(s_{t,h}, a_{t,h})\|_{\Sigma_{t,h}^{-1}} \leq H B_{\phi}^{\text{R}}.$$

Inserting all these back, we get the upper bound of

$$2HB_{\text{err}}^{\text{P}}B_{\phi}^{\text{P}} + 2(B_{\text{err}}^{\text{R}} + B_{\text{noise}}^{\text{R}}) \cdot HB_{\phi}^{\text{R}}.$$

Hence, we complete the proof.

Proof of Theorem [6.](#page-17-2) We have

$$\begin{aligned}\mathbb{E} \left[ \sum_{t=1}^T \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] &\leq \mathbb{E} \left[ \mathbf{1}\{(\mathfrak{E}^{\text{high}}) \sum_{t=1}^T \mathbf{1}\{(\mathfrak{E}^{\text{span}}) \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right) \right] \\ &\quad + \mathbb{E} \left[ \mathbf{1}\{((\mathfrak{E}^{\text{high}}) \mathcal{C}) \sum_{t=1}^T \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] \\ &\quad + \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{((\mathfrak{E}^{\text{span}}) \mathcal{C}) \left( V^*(s_{t,1}) - V^{\pi_t}(s_{t,1}) \right) \right] \\ &=: \mathbf{T}_A + \mathbf{T}_B + \mathbf{T}_C.\end{aligned}$$

For TA, by Lemmas [17](#page-29-1) to [19](#page-33-0) and re-arranging the results, we have

$$\begin{aligned} \mathbf{T}_A &\leq \frac{1}{\Gamma^2(-1)} \cdot \left( 2B_V(dH + 1/H) + HB_{\text{err}}^P \gamma + dH^2 + 1 + (B_{\text{err}}^R + B_{\text{noise}}^R)(2H + 1)B_\phi^R + 2HB_{\text{err}}^P B_\phi^P \right) \\ &= \tilde{O}\left(d^{5/2}H^{5/2} + d^2H^{3/2}\sqrt{T} + \varepsilon_1 \gamma (dH^2 + d^{3/2}H\sqrt{T}) \right. \\ &\quad \left. + \sqrt{\varepsilon_B} (d^2H^{5/2}\sqrt{T} + d^{3/2}H^{3/2}T) + \varepsilon_B \gamma (dH^2\sqrt{T} + d^{3/2}HT) \right) \end{aligned}$$

For TB, by Lemma [8,](#page-20-2) we have

$$T_B \leq HT \cdot \Pr\left((\mathfrak{E}^{\text{high}})^C\right) \leq 1.$$

For TC, by Lemma [15,](#page-27-0) we have

$$T_C \leq H \cdot \mathbb{E} \left[ \sum_{t=1}^T \mathbf{1}\{(\mathbf{e}_t^{\text{span}})^C\} \right] \leq dH^2.$$

Putting everything together, we complete the proof.

## D SUPPORTING LEMMAS

Lemma 20 (Gaussian concentration). *[\(Abeille & Lazaric,](#page-10-12) [2017\)](#page-10-12) Let* <sup>x</sup> ∼ N (0, c<sup>Σ</sup> −1 ) *for* <sup>c</sup> ∈ <sup>R</sup> + *and* <sup>Σ</sup> *a positive definite matrix. Then, for any* <sup>δ</sup> > <sup>0</sup>*, we have* Pr (∥x∥<sup>Σ</sup> > √ <sup>2</sup>cd log(2d/δ)) ≤ <sup>δ</sup>

Lemma 21 (Elliptical potential lemma). *Assume that* <sup>X</sup> ⊆ {<sup>x</sup> ∶ ∥x∥<sup>2</sup> ≤ <sup>1</sup>} *is compact and* span(X) = <sup>R</sup> d *. Let* <sup>x</sup>1, . . . , x<sup>T</sup> ∈ <sup>X</sup> *be a sequence of vectors,* <sup>Σ</sup><sup>1</sup> *be a positive definite matrix with each eigenvalue bounded within the range of* [a, b] *for some* a, b > <sup>0</sup>*, and* <sup>Σ</sup>t+<sup>1</sup> = <sup>Σ</sup><sup>t</sup> + <sup>x</sup>t<sup>x</sup> ⊺ t *. Then, we have*

$$\sum_{t=1}^T \min \{1, x_t^\top \Sigma_t^{-1} x_t\} \leq 2d \log \left( \frac{b}{a} + \frac{T}{ad} \right).$$

*Furthermore, if* <sup>Σ</sup><sup>1</sup> *is constructed via optimal design, i.e.,* <sup>Σ</sup><sup>1</sup> = <sup>E</sup>x∼<sup>ρ</sup> xx<sup>⊺</sup> *where* <sup>ρ</sup> ∈ <sup>∆</sup>(X) *is an optimal design over* X*, then we have*

$$\sum_{t=1}^T \min \{1, x_t^\top \Sigma_t^{-1} x_t\} \leq 2d \log(T+1).$$

Proof of Lemma [21.](#page-34-2) First we claim that

$$\min \left\{ 1, x_t^\top \Sigma_t^{-1} x_t \right\} \leq 2x_t^\top \Sigma_{t+1}^{-1} x_t \quad (14)$$

To show this, we use Sherman-Morrison-Woodbury formula [\(Bhatia,](#page-10-13) [2013\)](#page-10-13) for rank-one updates to a matrix inverse:

$$x_t^\top \Sigma_{t+1}^{-1} x_t = x_t^\top (\Sigma_t + x_t x_t^\top)^{-1} x_t = x_t^\top \left( \Sigma_t^{-1} - \frac{\Sigma_t^{-1} x_t x_t^\top \Sigma_t^{-1}}{1 + \|x_t\|_{\Sigma_t^{-1}}^2} \right) x_t = \|x_t\|_{\Sigma_t^{-1}}^2 - \frac{\|x_t\|_{\Sigma_t^{-1}}^4}{1 + \|x_t\|_{\Sigma_t^{-1}}^2} = \frac{\|x_t\|_{\Sigma_t^{-1}}^2}{1 + \|x_t\|_{\Sigma_t^{-1}}^2}.$$

Now let us consider two cases for the right-hand side of the above:

Case 1 : x ⊺ <sup>t</sup> Σ −1 <sup>t</sup> <sup>x</sup><sup>t</sup> ≤ <sup>1</sup>. Then, we can lower bound the right-hand side above by ∥<sup>x</sup>t∥ 2 Σ<sup>−</sup><sup>1</sup> /2.

Case 1 : 
$$x_t^\top \Sigma_t^{-1} x_t \leq 1$$
. Then, we can lower bound the right-hand side above by  $\|x_t\|_{\Sigma_t^{-1}}^2/2$ .

Case 2 : x ⊺ <sup>t</sup> Σ −1 <sup>t</sup> <sup>x</sup><sup>t</sup> ≥ <sup>1</sup>. Then the right-hand side above is directly at least <sup>1</sup>/<sup>2</sup> since the function <sup>x</sup>/(<sup>1</sup> + <sup>x</sup>) is increasing in <sup>x</sup>.

Hence, in both cases, we have x ⊺ <sup>t</sup> Σ −1 <sup>t</sup>+1x<sup>t</sup> ≥ min {1, x<sup>⊺</sup> <sup>t</sup> Σ −1 <sup>t</sup> <sup>x</sup>t} /<sup>2</sup>, which finishes the proof of [\(14\)](#page-35-0).

Since the log-determinant function is concave, we can obtain that log det (<sup>Σ</sup>t) − log det Σt+<sup>1</sup> ≤ tr (<sup>Σ</sup> −1 <sup>t</sup>+<sup>1</sup> (<sup>Σ</sup><sup>t</sup> − <sup>Σ</sup>t+1)) via first-order Taylor approximation. This gives us the following

$$\sum_{t=1}^T x_t^\top \Sigma_{t+1}^{-1} x_t = \sum_{t=1}^T \text{tr} \left( \Sigma_{t+1}^{-1} (\Sigma_{t+1} - \Sigma_t) \right) \leq \sum_{t=1}^T (\log \det \Sigma_{t+1} - \log \det \Sigma_t) = \log \left( \frac{\det \Sigma_{T+1}}{\det \Sigma_1} \right)$$

where the last step follows from telescoping. Since each eigenvalue of Σ<sup>1</sup> is lower bounded by a, we have det Σ<sup>1</sup> ≥ <sup>a</sup> d . Towards an upper bound of det ΣT+<sup>1</sup> = det(<sup>Σ</sup><sup>1</sup> +∑ T <sup>t</sup>=<sup>1</sup> xtx ⊺ t ), let (<sup>λ</sup>1, . . . , λd) denote the eigenvalues of ∑ T <sup>t</sup>=<sup>1</sup> xtx ⊺ t , and then we have

$$\det \left( \Sigma_1 + \sum_{t=1}^T x_t x_t^\top \right) \leq \prod_{i=1}^d (b + \lambda_i) \leq \left( \frac{1}{d} \sum_{i=1}^d (b + \lambda_i) \right)^d \leq \left( b + \frac{1}{d} \text{tr} \left( \sum_{t=1}^T x_t x_t^\top \right) \right)^d \leq \left( b + \frac{T}{d} \right)^d$$

Here, the first step is Weyl's inequality, the second step is AM-GM inequality, and the last step is because the trace is bounded by T. Plugging this upper bound back, we have

$$\log\left(\frac{\det \Sigma_{T+1}}{\det \Sigma_1}\right) \leq d \log\left(\frac{b}{a} + \frac{T}{ad}\right).$$

This completes the proof of the first statement.

For the case where Σ<sup>1</sup> is constructed via optimal design, we can rewrite ΣT+<sup>1</sup> in the following way:

$$\Sigma_{T+1} = \mathbb{E}_{x \sim \rho} xx^\top + \sum_{t=1}^T x_t x_t^\top = (T+1) \underbrace{\left( \frac{1}{1+T} \cdot \mathbb{E}_{x \sim \rho} xx^\top + \sum_{t=1}^T \frac{1}{1+T} \cdot x_t x_t^\top \right)}_{(*)} =: (T+1) \mathbb{E}_{x \sim \rho'} xx^\top.$$

where we consider (∗) as an expectation of xx<sup>⊺</sup> over a new distribution that we denote by ρ ′ . Recall that <sup>Σ</sup><sup>1</sup> is constructed via optimal design, which implies det Σ<sup>1</sup> ≥ detEx∼<sup>ρ</sup>′ xx<sup>⊺</sup> (Lemma [23\)](#page-36-0). This gives us

$$\log\left(\frac{\det \Sigma_{T+1}}{\det \Sigma_1}\right) = \log\left(\frac{(T+1)^d \det \mathbb{E}_{x \sim \rho'} x x^\top}{\det \Sigma_1}\right) \leq \log\left((T+1)^d\right) = d \log(T+1).$$

The following inequality is well-known; we use the version stated in [Zhu & Nowak](#page-12-9) [\(2022\)](#page-12-9).

Lemma 22 (Freedman's inequality). *Let* {<sup>X</sup>t}t≤<sup>T</sup> *be a real-valued martingale different sequence adapted to the filtration* Ft*, and let* <sup>E</sup>t[⋅] ∶= <sup>E</sup>[⋅ ∣ Ft−1]*. If* ∣<sup>X</sup>t∣ ≤ <sup>B</sup> *almost surely, then for any* <sup>η</sup> ∈ (0, <sup>1</sup>/B)*, the following holds with probability at least* <sup>1</sup> − <sup>δ</sup>*:*

$$\sum_{t=1}^T X_t \leq \eta \sum_{t=1}^T \mathbb{E}_t[X_t^2] + \frac{B \log(1/\delta)}{\eta}.$$

Lemma 23. *[\(Lattimore & Szepesvari](#page-11-15) ´ , [2020\)](#page-11-15) Assume that* <sup>Φ</sup> ⊆ <sup>R</sup> d *is compact and* span(Φ) = <sup>R</sup> d *. For a distribution* <sup>ρ</sup> *over* <sup>Φ</sup>*, define* <sup>Λ</sup>(ρ) = ∑ϕ∈<sup>Φ</sup> <sup>ρ</sup>(ϕ)ϕϕ<sup>⊺</sup> *and* <sup>g</sup>(ρ) = maxϕ∈<sup>Φ</sup> ∥ϕ∥ 2 <sup>Λ</sup>(ρ) <sup>−</sup><sup>1</sup> *. Then, the following are equivalent:*

- ρ *is a minimizer of* g*.*
- <sup>ρ</sup> *is a maximizer of* <sup>f</sup>(ρ) ∶= log det Λ(ρ)*.*
- <sup>g</sup>(ρ) = <sup>d</sup>*.*

*Furthermore, there exists a minimizer* <sup>ρ</sup> *of* <sup>g</sup> *such that* ∣supp(ρ)∣ ≤ <sup>d</sup>(<sup>d</sup> + <sup>1</sup>)/<sup>2</sup>*.*

Below we show that the Cauchy-Schwarz inequality is still valid when the matrix is not invertible under some conditions. We start with the following lemma.

Lemma 24. *Let* <sup>A</sup> *be a positive semi-definite matrix. Let* <sup>B</sup> *be a square root of* <sup>A</sup>*, i.e.,* <sup>A</sup> = BB<sup>⊺</sup> *. Then* range(A) = range(B)*.*

Proof of Lemma [24.](#page-36-4) We first show that range(A) ⊆ range(B). To see this, for any <sup>y</sup> ∈ range(A), there exists <sup>x</sup> such that <sup>y</sup> = Ax = BB⊺<sup>x</sup> = <sup>B</sup>(<sup>B</sup> <sup>⊺</sup>x). Hence <sup>y</sup> ∈ range(B). Next, we show that range(B) ⊆ range(A). To see this, for any <sup>y</sup> ∈ range(B), there exists <sup>x</sup> such that <sup>y</sup> = Bx. Let <sup>x</sup> = <sup>x</sup><sup>0</sup> + <sup>x</sup><sup>1</sup> where <sup>x</sup><sup>0</sup> ∈ null(B) and <sup>x</sup><sup>1</sup> ∈ rowspace(B). Then, <sup>y</sup> = Bx = Bx1. Since <sup>x</sup><sup>1</sup> ∈ rowspace(B), there exists <sup>z</sup> such that <sup>x</sup><sup>1</sup> = <sup>B</sup> ⊺ <sup>z</sup>. Thus, <sup>y</sup> = Bx<sup>1</sup> = BB<sup>⊺</sup> <sup>z</sup> = Az. Hence, <sup>y</sup> ∈ range(A).

Lemma 25 (Cauchy-Schwarz under pseudo-inverse). *Let* Σ *be a positive semi-definite matrix (that is unnecessarily invertible). Then, for any* <sup>x</sup> ∈ range(Σ) *and any* <sup>y</sup> ∈ <sup>R</sup> d *, we have*

$$x^\top y \leq \|x\|_{\Sigma^\dagger} \|y\|_{\Sigma}.$$

Proof of Lemma [25.](#page-36-2) Let B denote the square root of Σ and force B to be positive semi-definite. One can verify that BB† is the orthogonal projection matrix onto range(B), and hence, range(Σ) (recalling that range(B) = range(Σ) by Lemma [24\)](#page-36-4). Therefore, for any <sup>x</sup> ∈ range(Σ), we have BB†<sup>x</sup> = <sup>x</sup>. Then, we have

$$x^\top y = x^\top B^\dagger B y \leq \sqrt{x^\top B^\dagger B^\dagger x} \sqrt{y^\top B B y} = \|x\|_{\Sigma^\dagger} \|y\|_{\Sigma}$$

where the inequality follows from the standard Cauchy-Schwarz inequality.

Lemma 26 (Invariance under projection). *Let* <sup>Σ</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> *be a positive semi-definite matrix of rank* r*. For any vector* <sup>ϕ</sup> ∈ <sup>R</sup> d *, we have* ∥ϕ∥Σ*†* = ∥P ϕ∥Σ*† where* <sup>P</sup> *is the projection onto the range of* <sup>Σ</sup>*.*

Proof of Lemma [26.](#page-36-3) Assume the eigen-decomposition of <sup>Σ</sup> = <sup>U</sup>Λ<sup>U</sup> ⊺ , so Σ † = <sup>U</sup><sup>Λ</sup> †U ⊺ . Without loss of generality, we assume Λ has all its non-zero elements at the front and zero elements at the back on the diagonal. Denote <sup>U</sup><sup>r</sup> as the matrix obtained by replacing the last <sup>n</sup> − <sup>r</sup> columns of <sup>U</sup> by 0. Note that the first <sup>r</sup> columns of <sup>U</sup> is in the range of <sup>Σ</sup>, so we must have P U = <sup>U</sup>r. Then, we have the following

$$\|P\phi\|_{\Sigma^\dagger}^2 = \phi^\top P^\top \Sigma^\dagger P\phi = \phi^\top P^\top U\Lambda^\dagger U^\top P\phi = \phi^\top P^\top U_r\Lambda^\dagger U_r^\top P\phi = \phi^\top U_r\Lambda^\dagger U_r^\top \phi = \phi^\top U\Lambda^\dagger U^\top \phi = \|\phi\|_{\Sigma^\dagger}^2.$$

#### D.1 PSEUDO DIMENSION AND COVERING NUMBER

Definition 6 (ℓ1-Covering number). *(Definition 4 of [Modi et al.](#page-11-16) [\(2024\)](#page-11-16)) Given a hypothesis class* H ⊆ (Z ↦ <sup>R</sup>) *and* <sup>Z</sup> <sup>n</sup> = (<sup>z</sup>1, . . . , zn) ∈ Z n *,* <sup>ε</sup> > <sup>0</sup>*, define* N (ε, H, Z<sup>n</sup> ) *as the minimum cardinality of a set* C ⊆ H*, such that for any* <sup>h</sup> ∈ H*, there exists* <sup>h</sup> ′ ∈ C *such that* ∑ n <sup>i</sup>=<sup>1</sup> ∣h(<sup>z</sup>i) − <sup>h</sup> ′ (<sup>z</sup>i)∣/<sup>n</sup> ≤ <sup>ε</sup>*. We define* N (ε, H, n) = maxZ<sup>n</sup>∈Z<sup>n</sup> N (ε, H, Z<sup>n</sup> )*.*

Below we define the pseudo-dimension [\(Haussler,](#page-10-14) [2018;](#page-10-14) [Modi et al.,](#page-11-16) [2024\)](#page-11-16).

Definition 7 (VC-dimension). *For hypothesis class* H ⊆ (X → {0, <sup>1</sup>})*, we define its VCdimension* VC-dim(H) *as the maximal cardinality of a set* <sup>X</sup> = {<sup>x</sup>1, . . . , x∣X∣} ⊆ X *that satisfies* ∣HX∣ = <sup>2</sup> ∣X∣ *(or* <sup>X</sup> *is shattered by* H)*, where* H<sup>X</sup> *is the restriction of* H *to* <sup>X</sup>*, i.e.,* {(<sup>h</sup> (<sup>x</sup>1) , . . . , h (<sup>x</sup>∣X∣)) ∶ <sup>h</sup> ∈ H}*.*

Definition 8 (Pseudo-dimension). *For hypothesis class* H ⊆ (X → <sup>R</sup>)*, we define its pseudo dimension* Pdim(H) *as* Pdim(H) = VCdim (H<sup>+</sup> )*, where* H<sup>+</sup> = {(x, ξ) ↦ <sup>1</sup>[h(x) > <sup>ξ</sup>] ∶ <sup>h</sup> ∈ H} ⊆ (X × <sup>R</sup> → {0, <sup>1</sup>})

The following lemma provides a bound on the covering number of a hypothesis class via pseudo dimension.

Lemma 27. *(Corollary 42 of [Modi et al.](#page-11-16) [\(2024\)](#page-11-16)) Given a hypothesis class* H ⊆ Z ↦ [a, b] *with* Pdim(H) ≤ <sup>d</sup>*, then, for any* <sup>n</sup>*, we have*

$$\mathcal{N}(\varepsilon, \mathcal{H}, n) \leq (4e^2(b-a)/\varepsilon)^d.$$

*Note that the right-hand side is independent of* n*.*

## E LINEAR MDPS AND LQRS IMPLY LINEAR BELLMAN COMPLETENESS

It is already well known that linear Bellman completeness captures linear MDPs, as demonstrated in works such as [Agarwal et al.](#page-10-6) [\(2019\)](#page-10-6); [Zanette et al.](#page-12-1) [\(2020b\)](#page-12-1). Here, we show that it also captures LQRs for a convex subset of linear functions (specifically, when the value function is parameterized by a PSD matrix). We start with the definition.

Definition 9 (Linear Quadratic Regulator). *A linear quadratic regulator (LQR) problem is defined by a tuple* (A, B, Q, R) *where* <sup>A</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> *,* <sup>B</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>m</sup>*,* <sup>Q</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> *, and* <sup>R</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup>*. The objective is to find a policy* π *that minimizes the following:*

$$J(\pi) = \mathbb{E} \left[ \sum_{h=1}^H x_h^\top Q x_h + u_h^\top R u_h \right]$$

*where* <sup>x</sup>h+<sup>1</sup> = Ax<sup>h</sup> + Bu<sup>h</sup> + <sup>w</sup><sup>h</sup> *where* <sup>w</sup><sup>h</sup> ∼ N (0, <sup>Σ</sup>)*.*

Let us focus on an arbitrary step h and simply write the transition as the following (ignoring the subscript h for notational simplicity):

$$x' = Ax + Bu + w, \quad \text{where} \quad w \sim \mathcal{N}(0, \Sigma). \quad (15)$$

We consider state-action value functions of the form:

$$Q(x, u) = \begin{bmatrix} x \\ u \end{bmatrix}^\top \begin{bmatrix} P_{xx} & P_{xu} \\ P_{ux} & P_{uu} \end{bmatrix} \begin{bmatrix} x \\ u \end{bmatrix} + c. \quad (16)$$

It is linear in the quadratic feature <sup>ϕ</sup>(x, u) = [<sup>x</sup> 2 , u<sup>2</sup> , xu, x, u, <sup>1</sup>]. Without loss of generality, we assume <sup>P</sup>xu = <sup>P</sup> ⊺ ux. Note that we may not have the Bellman completeness for *any* such Q. However, it does hold under the restriction that <sup>P</sup> <sup>=</sup> [ Pxx Pxu <sup>P</sup>ux <sup>P</sup>uu] is PSD. Recall that <sup>P</sup> is PSD if and only if (1) <sup>P</sup>uu is PSD, and (2) its Schur complement <sup>P</sup>xx − <sup>P</sup>xu<sup>P</sup> −1 uuPux is PSD. We note that such a set of feasible P is a convex set.

$$\tilde{Q}(x, u) = \mathbb{E}_{x'} \left[ \min_{u'} Q(x', u') \right] \quad (17)$$

$$\begin{aligned} &= \mathbb{E}_w \left[ \min_{u'} \left[ \begin{array}{cc} Ax + Bu + w \\ u' \end{array} \right]^T \begin{bmatrix} P_{xx} & P_{xu} \\ P_{ux} & P_{uu} \end{bmatrix} \begin{bmatrix} Ax + Bu + w \\ u' \end{bmatrix} + c \right] \\ &= \mathbb{E}_w \left[ \min_{u'} \left\{ [Ax + Bu + w]^T P_{xx} [Ax + Bu + w] + 2 [Ax + Bu + w]^T P_{xu} u' + u'^T P_{uu} u' \right\} \right] + c. \end{aligned} \quad (18)$$
(19)

Using first-order condition, we know that the optimal u ′ (for a fixed w) satisfies

$$u' = -P_{uu}^{-1} P_{ux}(Ax + Bu + w), \quad (20)$$

which implies that, the term in [\(17\)](#page-37-5) is equal to

$$\min_{u'} Q(x', u') = [Ax + Bu + w]^T [P_{xx} - P_{xu} P_{uu}^{-1} P_{ux}] [Ax + Bu + w] + c \quad (21)$$

Plugging the above in [\(19\),](#page-38-3) we get

$$\tilde{Q}(x, u) = \mathbb{E}_w \left[ [Ax + Bu + w]^T [P_{xx} - P_{xu}P_{uu}^{-1}P_{ux}] [Ax + Bu + w] + c \right] \quad (22)$$

$$= [Ax + Bu]^T [P_{xx} - P_{xu} P_{uu}^{-1} P_{ux}] [Ax + Bu] + c + \mathbf{Tr}((P_{xx} - P_{xu} P_{uu}^{-1} P_{ux}) \Sigma) \quad (23)$$

$$= [Ax + Bu]^T [P_{xx} - P_{xu} P_{uu}^{-1} P_{xu}^T] [Ax + Bu] + c' \quad (24)$$

$$= \begin{bmatrix} x \\ u \end{bmatrix}^T \begin{bmatrix} A^T \\ B^T \end{bmatrix} (P_{xx} - P_{xu} P_{uu}^{-1} P_{xu}^\top) [A \quad B] \begin{bmatrix} x \\ u \end{bmatrix} + c' \quad (25)$$

where c ′ is some constant. The middle matrix above is PSD if <sup>P</sup>xx ⪰ <sup>P</sup>xu<sup>P</sup> −1 uuP ⊺ xu, which holds since <sup>P</sup> is PSD. Thus, we conclude that <sup>Q</sup>̃ is also linear for some PSD matrix.

We can also easily verify that the reward (cost) function is linear in the quadratic feature. Hence, we complete the proof.

## F COMPUTATIONALLY EFFICIENT IMPLEMENTATIONS FOR OPTIMIZATION ORACLES

The convex programming algorithm given in Algorithm [2](#page-39-0) is due to [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7). In the following, we provide an informal description of Algorithm [2](#page-39-0) below but refer the reader to [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7) for the full details.

At an iteration <sup>t</sup> ≤ <sup>T</sup>, Algorithm [<sup>2</sup>](#page-39-0) stars with a set D<sup>t</sup> which contains the set K, and a set of <sup>2</sup><sup>N</sup> points U<sup>t</sup> sampled (approximately) uniformly from D<sup>t</sup> using the SAMPLER subroutine in Algorithm [3.](#page-39-1) It then uses the first <sup>N</sup> samples from U<sup>t</sup> to compute an approximate centroid <sup>z</sup><sup>t</sup> of the set D<sup>t</sup> in Line [23;](#page-39-2) the remaining points from U<sup>t</sup> are denoted by Vt. It then queries the separation oracle O sep K at the point <sup>z</sup>t. If <sup>z</sup><sup>t</sup> ∈ K, then we terminate and return <sup>z</sup>t. Otherwise, we use the separating hyperplane between <sup>z</sup><sup>t</sup> and K to shrink the set D<sup>t</sup> further into Dt+<sup>1</sup> in Line [29.](#page-39-3) Finally, it calls SAMPLER again using the set of points V<sup>t</sup> as a warm start to get <sup>2</sup><sup>N</sup> new (approximately) i.i.d. sample from Dt+<sup>1</sup> in Line [30.](#page-39-4) Equipped with the sets Dt+<sup>1</sup> and Ut+1, another iteration of the algorithm follows.

On receiving a convex set D and a set of points V, the SAMPLER protocol in Algorithm [<sup>3</sup>](#page-39-1) first refines V to V ′ by disposing off any points <sup>z</sup> ∈ V that do not lie in D. Then, it starts a random ball walk from the samples in V ′ : in order to update the current point <sup>z</sup>̂ we first sample a point <sup>z</sup> ′ uniformly from the ellipsoid <sup>z</sup>̂+ <sup>η</sup><sup>Λ</sup> <sup>1</sup>/<sup>2</sup>Bd(1) (where <sup>Λ</sup> is defined using the points in V ′ ) and then updates <sup>z</sup>̂ ← <sup>z</sup> ′ if z ′ ∈ D. The analysis of [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7) shows that this ball walk mixes fast to a uniform distribution over the set D.

# G MISSING DETAILS FROM S[ECTION](#page-8-6) 6.2

## G.1 COMPUTATIONALLY EFFICIENT ESTIMATION OF REWARD FUNCTION (EQN. [2\)](#page-5-3)

The convex set feasibility procedure of [Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7) can also be used to estimate the parameters for the reward functions in Equation [\(2\)](#page-5-3) in Algorithm [1.](#page-5-0) Note that for any time t and

Algorithm 2 Solving Convex Programs by Random Walks [\(Bertsimas & Vempala](#page-10-7) [\(2004\)](#page-10-7))

Require: • Separation oracle O sep K for the convex set K ⊆ <sup>R</sup> d .

• Parameters r, R, δ.

- 1: Let <sup>T</sup> = <sup>2</sup><sup>d</sup> log(R/δr) and <sup>N</sup> = <sup>O</sup>(<sup>d</sup> log(1/δ)) 2: Let D<sup>1</sup> be the axis-aligned cube with width <sup>R</sup> with center <sup>z</sup><sup>1</sup> = <sup>0</sup>. 3: Sample <sup>2</sup><sup>N</sup> points U<sup>1</sup> ∶= {<sup>z</sup> 1 1 , . . . , z<sup>2</sup><sup>N</sup> <sup>1</sup> } ← Uniform(D1). 4: Let V<sup>1</sup> ∶= {<sup>z</sup> 1 1 , . . . , z<sup>N</sup> <sup>1</sup> } and V¯ <sup>1</sup> ∶= U<sup>1</sup> ∖ V1. 5: for <sup>t</sup> = <sup>1</sup>, . . . , T do 6: Compute the point <sup>z</sup><sup>t</sup> ← <sup>1</sup> <sup>N</sup> ∑z∈V<sup>t</sup>
- z. 7: if <sup>z</sup><sup>t</sup> ∈ <sup>K</sup> then 8: Return z<sup>t</sup> and terminate. 9: else 10: // If <sup>z</sup><sup>t</sup> ∉ K, shrink the set D<sup>t</sup> using a separating hyperplane // 11: Let ⟨<sup>a</sup>t, z⟩ ≤ <sup>b</sup> be the separating hyperplane returned by O sep K (<sup>z</sup>t). 12: Let Dt+<sup>1</sup> ← D<sup>t</sup> ∩ H<sup>t</sup> where H<sup>t</sup> denotes the halfspace {<sup>z</sup> ∣ ⟨<sup>a</sup>t, z⟩ ≤ ⟨<sup>a</sup>t, zt⟩}. 13: Sample <sup>2</sup><sup>N</sup> points Ut+<sup>1</sup> ∶= {<sup>z</sup> 1 , . . . , z<sup>2</sup><sup>N</sup> <sup>1</sup> } ← <sup>S</sup>AMPLER(Dt+1, N,Vt). 14: Let Vt+<sup>1</sup> ∶= {<sup>z</sup> 1 1 , . . . , z<sup>N</sup> <sup>1</sup> } and V¯ <sup>t</sup>+<sup>1</sup> ∶= Ut+<sup>1</sup> ∖ Vt+1. 15: end if 16: end for 17: Terminate and report that K is empty.

Algorithm 3 SAMPLER used in Algorithm [2](#page-39-0)

Require: • Convex set D.

• Parameter N. • Points V = {<sup>z</sup>

1

, . . . , z<sup>N</sup> }.

1: Let step size <sup>η</sup> = <sup>Θ</sup>(1/

√

<sup>d</sup>), and number of iterations <sup>N</sup>′ = <sup>O</sup>̃(<sup>d</sup>

<sup>3</sup>N).

2: Let V ′ ∶= {<sup>z</sup> ∈ <sup>V</sup> ∣ <sup>z</sup> lies in D}, and define

$$\bar{z} = \frac{1}{|\mathcal{V}'|} z \quad \text{and} \quad \Lambda = \frac{1}{|\mathcal{V}'|} \sum_{z \in \mathcal{V}'} (z - \bar{z})(z - \bar{z})^T.$$

3: Let U = ∅ and <sup>z</sup>̂∈ V ′ be any arbitrary stating point (note that <sup>z</sup>̂∈ D). 4: while ∣U∣ ≤ <sup>2</sup><sup>N</sup> do 5: Initialize <sup>i</sup> ← <sup>1</sup>. 6: while <sup>i</sup> ≤ <sup>N</sup>′ do 7: Sample z ′ ∼ Uniform(ẑ+ <sup>η</sup><sup>Λ</sup> <sup>1</sup>/<sup>2</sup>Bd(1)). // Ball walk // 8: if z ′ ∈ D then 9: Update <sup>z</sup>̂ ← <sup>z</sup> ′ and <sup>i</sup> ← <sup>i</sup> + <sup>1</sup>. 10: end if 11: end while 12: Update U = U ∪ {ẑ}. 13: end while 14: Return U. // Distribution of samples in U closely approximates Uniform(D) //

horizon <sup>h</sup> ∈ [H], the objective in Equation [\(1\)](#page-5-2) is the optimization problem

$$\widehat{\omega}_{t,h} \leftarrow \operatorname{argmin}_{\omega \in \mathcal{O}(1)} \sum_{i=1}^{t-1} \left( \langle \omega, \phi(s_{i,h}, a_{i,h}) \rangle - r_{i,h} \right)^2. \quad (27)$$

In the following, we provide a computationally efficient procedure, based off on Algorithm [2,](#page-39-0) to approximately solve the above squared loss minimization problem given a linear optimization oracle over the feature space (Assumption [6\)](#page-8-4). Note that since <sup>r</sup>i,h ∈ [0, <sup>1</sup>], the constraint on the point ω implies that the objective value in Equation [\(27\)](#page-39-5) is at most 2. Thus, we can solve the above

Algorithm 4 Computationally Efficient Implementation of Osq apx for Value Estimation

Require: • Data samples {(<sup>s</sup><sup>i</sup> , a<sup>i</sup> , ui)}i≤t.

- Convex domain O(W).
- Approximation parameter ε.
- Linear optimization oracle Olin defined in Assumption [6.](#page-8-4)

1: // Convert Square Loss Minimization into a Set Feasibility Problem //

2: Define the convex set

$$\mathcal{K}_{\text{APX}} := \left\{ \theta \in \mathbb{R}^d \mid \begin{array}{l} \langle \theta, \phi(s_i, a_i) \rangle - u_i \leq \varepsilon \text{ for all } i \leq t \\ \langle \theta, \phi(s_i, a_i) \rangle - u_i \geq -\varepsilon \text{ for all } i \leq t \\ |\langle \theta, \phi(s, a) \rangle| \leq W_h + \varepsilon \text{ for all } s, a \end{array} \right\} \quad (26)$$

- 3: // Define a Separation Oracle for the set KAPX using Olin // 4: Definition O sep KAPX (Input: parameter <sup>θ</sup> ∈ <sup>R</sup> d )
  - For all <sup>i</sup> ≤ <sup>t</sup>, verify if −<sup>ε</sup> ≤ ⟨θ, ϕ(<sup>s</sup><sup>i</sup> , ai)⟩ − <sup>u</sup><sup>i</sup> ≤ <sup>ε</sup> for all <sup>i</sup> ≤ <sup>t</sup>. ▸ Output any violating constraint as a separating hyperplane. Terminate.
- Then, verify if max{maxs,a⟨θ, ϕ(s, a)⟩, maxs,a⟨−θ, ϕ(s, a)⟩} ≤ <sup>W</sup> + <sup>ε</sup> using the linear optimization oracle Olin (Assumption [6\)](#page-8-4). ▸ If violated, use Olin to compute a violating constraint and return it as the separating hyperplane. Terminate. ▸ Otherwise, return that the point <sup>θ</sup> ∈ KAPX. Terminate. 5: EndDefinition 6: // Find a feasible point in KAPX // 7: Invoke Algorithm [<sup>2</sup>](#page-39-0) to return a feasible point in the set KAPX with O sep KAPX as the separation oracle.

optimization problem upto precision <sup>ε</sup>, by iterating over the set <sup>∆</sup> ∈ {0, ε, <sup>2</sup>ε, . . . , <sup>2</sup> − ε, <sup>2</sup>} in order to solve the set feasibility problem

$$\mathcal{K}_{\text{APX}}^{\Delta} := \left\{ \omega \in \mathbb{R}^d \mid \sum_{i=1}^{t-1} (\langle \omega, \phi(s_{i,h}, a_{i,h}) \rangle - r_{i,h})^2 \leq \Delta + \varepsilon \mid |\langle \omega, \phi(s, a) \rangle| \leq 1 + \varepsilon \text{ for all } s, a \right\} \quad (28)$$

and stopping at the smallest point <sup>∆</sup> for which K ∆ APX has a feasible solution. It is easy to see that for any <sup>∆</sup>, either K ∆ APX is empty or the shifted cube <sup>ω</sup>̂t,h + <sup>R</sup>∞(ε) ⊆ K ∆ APX. Furthermore, under Assumption [<sup>7</sup>](#page-9-0) we also have that K ∆ APX ⊆ <sup>R</sup>∞(R) for any <sup>∆</sup>. Thus, for any <sup>∆</sup>, whenever a feasible solution exists, the set K ∆ APX satisfies the prerequisites for Theorem [4,](#page-8-3) where recall that we can tolerate the parameter R to be exponential in the dimension d or the horizon H. Furthermore, a separation oracle O sep K<sup>∆</sup> can be easily implemented by using the linear optimization oracle Olin w.r.t. the feature space (Assumption [6\)](#page-8-4) and by explicitly constructing a separation oracle for the ellipsoidal constraint

$$\sum_{i=1}^{t-1} (\langle \omega, \phi(s_{i,h}, a_{i,h}) \rangle - r_{i,h})^2 \leq \Delta + \varepsilon.$$

We provide the implementation of the above in Algorithm [5,](#page-41-0) which relies on Algorithm [2](#page-39-0) for solving the corresponding set feasibility problems. The guarantee in Theorem [4](#page-8-3) to find a feasible point in K ∆ APX (for each ∆) gives the following guarantee on computational efficiency for Algorithm [5.](#page-41-0)

Theorem 7. *Let* <sup>ε</sup> > <sup>0</sup>*,* <sup>δ</sup> ∈ (0, <sup>1</sup>)*, and suppose Assumption [<sup>7</sup>](#page-9-0) holds with some parameter* <sup>R</sup> > <sup>0</sup>*. Additionally, suppose Assumption [<sup>6</sup>](#page-8-4) holds with the linear optimization oracle denoted by* Olin*. Then, for any* <sup>t</sup> ∈ [T] *and* <sup>h</sup> ∈ [H]*, Algorithm [<sup>5</sup>](#page-41-0) returns a point* <sup>ω</sup>̂t,h *that, with probability at least* <sup>1</sup> − <sup>δ</sup>*, satisfies*

$$\sum_{i=1}^{t-1} (\langle \tilde{\omega}, \phi(s_{i,h}, a_{i,h}) \rangle - r_{i,h})^2 \leq \min_{\omega \in \mathcal{O}(1)} \sum_{i=1}^{t-1} (\langle \omega, \phi(s_{i,h}, a_{i,h}) \rangle - r_{i,h})^2 + \varepsilon \quad \text{and} \quad \tilde{\omega}_{t,h} \in \mathcal{O}(1 + \varepsilon).$$

Algorithm 5 Computationally Efficient Implementation of Osq apx for Reward Estimation

- Require: • Data samples {(<sup>s</sup><sup>i</sup> , a<sup>i</sup> , ri)}i≤t.
  - Convex domain O(1).
  - Approximation parameter ε.
  - Linear optimization oracle Olin defined in Assumption [6.](#page-8-4)

1: for <sup>∆</sup> ∈ {0, ε, <sup>2</sup>ε, . . . , <sup>2</sup> − ε, <sup>2</sup>} do 2: // Define a Set Feasibility Problem using ∆ // 3: Define the convex set

$$\mathcal{K}_{\text{APX}}^{\Delta} := \left\{ \omega \in \mathbb{R}^d \mid \sum_{i=1}^{t-1} (\langle \omega, \phi(s_i, a_i) \rangle - r_i)^2 \leq \Delta + \varepsilon \mid |\langle \omega, \phi(s, a) \rangle| \leq 1 + \varepsilon \text{ for all } s, a \right\} \quad (29)$$

- 4: // Define a Separation Oracle for K ∆ APX using Olin // 5: Definition O sep K<sup>∆</sup> (Input: parameter <sup>ω</sup> ∈ <sup>R</sup> d )
  - Verify if ∑ <sup>t</sup>−<sup>1</sup> <sup>i</sup>=<sup>1</sup> (⟨ω, ϕ(<sup>s</sup><sup>i</sup> , ai)⟩ − <sup>r</sup>i) 2 ≤ <sup>∆</sup> + <sup>ε</sup>. ▸ If not, return a separating hyperplane for the ellipsoid ∑ <sup>t</sup>−<sup>1</sup> <sup>i</sup>=<sup>1</sup> (⟨ω, ϕ(<sup>s</sup><sup>i</sup> , ai)⟩ − <sup>r</sup>i) ≤ <sup>∆</sup> + <sup>ε</sup> w.r.t. <sup>ω</sup>. Terminate.
- Then, verify if max{maxs,a⟨ω, ϕ(s, a)⟩, maxs,a⟨−ω, ϕ(s, a)⟩} ≤ <sup>1</sup> + <sup>ε</sup> using the linear optimization oracle Olin (Assumption [6\)](#page-8-4). ▸ If violated, use Olin to compute a violating constraint and return it as the separating hyperplane. Terminate. ▸ Otherwise, return that the point <sup>ω</sup> ∈ K ∆ APX. Terminate. 6: EndDefinition 7: // Find a feasible point in K ∆ APX // 8: Invoke Algorithm [<sup>2</sup>](#page-39-0) with O sep K<sup>∆</sup> as the separation oracle.
  - If succeeded in finding a feasible point <sup>ω</sup>̂ ∈ K ∆ APX. Return <sup>ω</sup>̂ and terminate.
- Else, continue. 9: end for