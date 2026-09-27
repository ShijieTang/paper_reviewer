# Monte-Carlo Tree Search with Uncertainty Propagation via Optimal Transport

Tuan Dam <sup>1</sup> Pascal Stenger <sup>2</sup> Lukas Schneider <sup>3</sup> Joni Pajarinen <sup>4</sup> Carlo D'Eramo 2 5 6 Odalric-Ambrym Maillard <sup>7</sup>

## Abstract

This paper introduces a novel backup strategy for Monte-Carlo Tree Search (MCTS) tailored for highly stochastic and partially observable Markov decision processes. We adopt a probabilistic approach, modeling both value and action-value nodes as Gaussian distributions, to introduce a novel backup operator that computes value nodes as the Wasserstein barycenter of their action-value children nodes; thus, propagating the uncertainty of the estimate across the tree to the root node. We study our novel backup operator when using a novel combination of L 1 -Wasserstein barycenter with α-divergence, by drawing a crucial connection to the generalized mean backup operator. We complement our probabilistic backup operator with two sampling strategies, based on optimistic selection and Thompson sampling, obtaining our Wasserstein MCTS algorithm. We provide theoretical guarantees of asymptotic convergence of O(n −1/2 ), with n as the number of visited trajectories, to the optimal policy and an empirical evaluation on several stochastic and partially observable environments, where our approach outperforms wellknown related baselines.

## 1. Introduction

Monte-Carlo Tree Search (MCTS) has become a crucial algorithmic paradigm for tackling challenging planning and Reinforcement Learning (RL) problems, particularly after its widespread success in deterministic games like Go and Chess [\(Silver et al., 2016a;](#page-9-0) [2017b\)](#page-10-0). However, moving beyond these deterministic settings toward highly stochastic or partially observable Markov Decision Processes (MDPs/POMDPs) reveals major difficulties. In these cases, two key obstacles arise: *Uncertainty in Value Estimates:* In problems with substantial randomness or limited observability, naive value backups may lead to erroneous or unstable estimates, which propagate through the tree and degrade overall performance. *Exploration-Exploitation Balancing:* Traditional UCT-based exploration bonuses [\(Koc](#page-9-1)[sis et al., 2006\)](#page-9-1) can falter under high variance transitions, often causing either over- or under-exploration. Recent works [\(Tesauro et al., 2012;](#page-10-1) [Bai et al., 2013;](#page-8-0) [2014\)](#page-8-1) have suggested Bayesian or distributional methods for MCTS to better quantify uncertainty. Meanwhile, [Metelli et al.](#page-9-2) [\(2019\)](#page-9-2) leveraged L 2 -Wasserstein barycenters to propagate distributional information in temporal-difference learning. Yet, several open questions remain on how to unify *distribution-based backups* and *flexible exploration strategies* within a single MCTS framework that provably handles high stochasticity and partial observability.

Our Approach. In this paper, we propose a new MCTS algorithm, *Wasserstein MCTS*, that models each node's value as a Gaussian distribution and propagates *both* mean and variance estimates throughout the tree. Crucially, we introduce a novel backup operator that computes *value nodes* as L 1 -Wasserstein barycenters of their *action-value children*, using an α-divergence as the distance measure. This yields:

• *Distributional Value Backups:* By tracking distributions (rather than point estimates), our method captures the inherent uncertainty of each node's value, especially valuable in stochastic or partially observable domains.

We complement these distributional backups with two exploration mechanisms—an optimistic UCT bonus, and a Thompson sampling approach that selects actions by sam-

<sup>1</sup>Hanoi University of Science and Technology, Hanoi, Vietnam <sup>2</sup>Department of Computer Science, Technical University of Darmstadt, Germany <sup>3</sup>ETHZ - ETH Zurich, Switzerland <sup>4</sup>Department of Electrical Engineering and Automation, Aalto University, Finland <sup>5</sup>Center for Artificial Intelligence and Data Science, University of Wurzburg, Germany ¨ <sup>6</sup>Hessian Center for Artificial Intelligence (Hessian.ai), Germany <sup>7</sup>Univ. Lille, Inria, CNRS, Centrale Lille, UMR 9189- CRIStAL, F-59000 Lille, France. Correspondence to: Tuan Dam <tuandq@soict.hust.edu.vn>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<sup>•</sup> *Generalized Mean Operator:* The α-divergence ties naturally to the power-mean backup [\(Dam et al., 2019;](#page-9-3) [2024a\)](#page-9-4), letting us interpolate between average-like and max-like updates to mitigate the overestimation often seen in RL [\(Hasselt, 2010\)](#page-9-5).

pling from the node's Gaussian posterior.

Our Key Contributions. *1. Uncertainty Propagation via* L 1 *-Wasserstein Barycenters.* We provide a principled way to back up distributions in an MCTS, unifying L 1 - Wasserstein geometry and α-divergences to handle high variance and partial observability. *2. Connection to Generalized Mean Backup.* Our backup operator yields a powermean update for node values, enabling a controllable continuum between overly optimistic (max-like) and riskaverse (average-like) estimates. *3. Polynomial Convergence Analysis.* We prove that *Wasserstein MCTS* with Thompson sampling converges to the optimal policy at a rate O(n −1/2 ), matching known lower bounds. This is in contrast to prior distributional MCTS methods that lacked explicit convergence guarantees. *4. Extensive Empirical Validation.* On a suite of highly stochastic MDPs (e.g. *River-Swim, Taxi*) and partially observable tasks (*Pocman, Rocksample*), our approach outperforms established baselines, including UCT, Power-UCT, and Bayesian MCTS variants.

Overall, *Wasserstein MCTS* offers a flexible and theoretically grounded framework for handling uncertainty within MCTS. By combining Gaussian node models, L 1 - Wasserstein barycenters, and α-divergences, it effectively balances exploration and exploitation in domains where noise or partial observability make traditional MCTS methods brittle.

## 2. Related Work

[Metelli et al.](#page-9-2) [\(2019\)](#page-9-2) use L 2 -Wasserstein barycenters to propagate uncertainty in temporal-difference learning. In MCTS, Bayesian methods handle uncertainty by treating values as Gaussian distributions [\(Tesauro et al., 2012\)](#page-10-1) or Dirichlet-NormalGamma posteriors [\(Bai et al., 2013;](#page-8-0) [2014\)](#page-8-1). Unlike these, we propagate uncertainty *throughout* the tree via L 1 -Wasserstein barycenters and α-divergences, linking to generalized-mean backups [\(Dam et al., 2019\)](#page-9-3) and maintaining both mean and variance estimates. This distributional perspective is effective in highly stochastic or partially observable tasks. In multi-armed bandits, optimism [\(Auer et al., 2002a\)](#page-8-2) and Thompson sampling [\(Thompson, 1933\)](#page-10-2) are standard; we combine these with our uncertainty propagation scheme to guide action selection in MCTS.

## 3. Background

## 3.1. Markov Decision Process

We consider an agent in an infinite-horizon discounted Markov decision process (MDP) M = ⟨S, A, R,P, γ⟩, where S is the state space, A is the finite action space, R : S × A × S → R is the reward function, P : S × A → S is the transition kernel, and γ ∈ [0, 1) is the discount factor. A policy π ∈ Π : S → A defines the action selection probabilities based on states. The action-value function Q<sup>π</sup> is given by Q<sup>π</sup> (s, a) ≜ <sup>E</sup> -P<sup>∞</sup> <sup>k</sup>=0 γ k ri+k+1 | s<sup>i</sup> = s, a<sup>i</sup> = a, π , representing the expected cumulative discounted reward for executing action a in state s and following policy π. The objective is to find the optimal policy that maximizes Q<sup>π</sup> , satisfying the Bellman equation [\(Bellman, 1954\)](#page-8-3): Q<sup>∗</sup> (s, a) ≜ R S P(s ′ |s, a) [R(s, a, s′ ) + γ maxa′ Q<sup>∗</sup> (s ′ , a′ )] ds′ , ∀s ∈ S, a ∈ A. From the optimal action-value function, we derive the optimal value function as V ∗ (s) ≜ maxa∈A Q<sup>∗</sup> (s, a), ∀s ∈ S.

#### 3.2. Monte-Carlo Tree Search

Monte-Carlo Tree Search (MCTS) combines Monte-Carlo sampling, tree search, and exploration strategies from multi-armed bandits [\(Auer et al., 2002b\)](#page-8-4) to solve MDPs. It builds a search tree where states are nodes and actions are edges. MCTS involves four key steps: Selection: Navigate from the root to a leaf node using a *tree-policy*. Expansion: Expand the reached node based on the tree policy. Simulation: Perform a rollout (Monte-Carlo simulation) from the child node to estimate its value, or use a pretrained neural network [\(Silver et al., 2016a\)](#page-9-0) for this estimation. Backup: Update the action-values Q(·) along the visited trajectory using the collected rewards.

## 4. Formalization

Problem Setup Monte Carlo Tree Search (MCTS) is an algorithm for exploring and evaluating trajectories in an MDP. Starting from an initial state s0, MCTS incrementally builds a planning tree by simulating trajectories. Each trajectory either reaches a leaf node or terminates when a predetermined maximum depth H is reached. At the end of each trajectory, a *playout policy* (which may be deterministic or stochastic) is executed from the final node reached, allowing the algorithm to evaluate the associated state. After running for t trajectories, the MCTS algorithm provides the following outputs:

- at: estimate of the optimal action to take in state s0,
- V <sup>t</sup>(s0): estimate of the optimal value function at s0.

Evaluating MCTS Performance The performance of the MCTS algorithm is assessed based on its *convergence rate*, r(t), which quantifies how quickly the algorithm approaches the optimal policy. Specifically, the following bounds hold:

$$\begin{aligned} \mathbb{E} [V^*(s_0) - Q^*(s_0, \bar{a}_t)] &\leq r(t), \\ |\mathbb{E} [V^*(s_0) - \bar{V}_t(s_0)]| &\leq r(t), \end{aligned}$$

where V ⋆ (s0) and Q<sup>⋆</sup> (s0, a) are the true optimal value and action-value functions at state s0, respectively.

Recursive Value Estimation To analyze the MCTS algorithm, we consider a planning horizon H and a playout policy π<sup>0</sup> with an associated value function V0. For each node s<sup>h</sup> at depth h (i.e., the state reached after h steps from s0), we recursively define the value function <sup>V</sup>e(sh) as follows. At the leaf nodes (h = H), the value function is simply the playout policy's value:

$$\tilde{V}(s_H) = V_0(s_H).$$

For all other depths h ⩽ H − 1, we compute the actionvalue function <sup>Q</sup>e(sh, a) and value function <sup>V</sup>e(sh) as:

$$\tilde{Q}(s_h, a) = r(s_h, a) + \gamma \sum_{s_{h+1} \in \mathcal{A}_s} \mathcal{P}(s_{h+1} \mid s_h, a) \tilde{V}(s_{h+1}),$$

$$\tilde{V}(s_h) = \max_a \tilde{Q}(s_h, a),$$

where r(sh, a) is the mean immediate reward obtained by taking action a in state sh, P(sh+1 | sh, a) is the probability of transitioning to state sh+1 from s<sup>h</sup> given action a and γ is the discount factor.

Bounding the Error The recursive structure of the value estimates gives rise to a bound on the error between the true optimal action-value function Q<sup>⋆</sup> (s0, a) and the MCTS estimate <sup>Q</sup>e(s0, a). Specifically, we have:

$$\left| Q^*(s_0, a) - \tilde{Q}(s_0, a) \right| \leq \gamma^H \|V^* - V_0\|_\infty,$$

where the supremum norm ∥V <sup>⋆</sup> − V0∥<sup>∞</sup> can be restricted to states reachable within H steps from s0.

Goal of MCTS The ultimate aim of the MCTS algorithm is to minimize the convergence rate r(t) by constructing accurate estimates of <sup>Q</sup>e(s0, a) and <sup>V</sup>e(s0), which in turn approach the true optimal functions Q<sup>⋆</sup> (s0, a) and V ⋆ (s0), and then identify the best action at the root node:

$$a_\star = \arg \max_a Q^\star(s_0, a).$$

### 5. Wasserstein Barycenter With α-Divergence

We introduce the key notions behind our distribution-based backups: the *Wasserstein barycenter* and the α*-divergence*. Unlike prior works that use L -based Wasserstein distances [\(Metelli et al., 2019\)](#page-9-2), we adopt an L 1 -Wasserstein distance combined with α-divergences. This combination yields more robust value backups in stochastic and partially observable settings.

#### 5.1. Wasserstein Barycenter

on X whose q-th moment is finite. For two distributions µ, ν ∈ Pq(X ), the L q *-Wasserstein distance* is

$$W_q(\mu, \nu) = \left( \inf_{\rho \in \Gamma(\mu, \nu)} \mathbb{E}_{(X, Y) \sim \rho} [d(X, Y)^q] \right)^{1/q},$$

where Γ(µ, ν) is the set of joint couplings whose marginals match µ and ν. Given n distributions {νi} n <sup>i</sup>=1 and weights {wi} summing to 1, the L q *-Wasserstein barycenter* is

$$\bar{\nu} = \arg \min_{\nu} \sum_{i=1}^n w_i W_q(\nu, \nu_i)^q.$$

Our work focuses on q = 1.

#### 5.2. α-divergence and the L<sup>1</sup> Wasserstein Barycenter

In many distribution-based backup schemes, the *Wasserstein distance* is a natural choice to quantify how "far apart" two distributions are. A commonly used approach [\(Metelli](#page-9-2) [et al., 2019\)](#page-9-2) is to employ the L 2 -Wasserstein metric. In contrast, we consider an L 1 -Wasserstein formulation coupled with an α-divergence for two main reasons:

• *Robustness & Aggregation Control.* An L 1 -based metric can be more robust to outliers and large deviations than L 2 . Furthermore, combining it with the α-divergence allows a continuous interpolation between averaging and max-like backups (through the α parameter).

• *Connection to Power-Mean Updates.* Modeling nodes as Gaussians (or particle distributions) and relying on L 1 - Wasserstein with an α-divergence yields closed-form updates that coincide with the power-mean operator. This unifies average and maximum backups in a single formula and lets us propagate both means and variances (uncertainty) through the tree.

f-divergences and the α-divergence. An f-divergence [\(Csiszar, 1964\)](#page-9-6) between two points ´ X and Y over a Manifold M defined as

$$D_{f_\alpha}(X \parallel Y) = \sum_i \xi_Y^{(i)} f_\alpha\left(\frac{\xi_X^{(i)}}{\xi_Y^{(i)}}\right), f_\alpha(x) = \frac{x^\alpha - 1 - \alpha(x-1)}{\alpha(\alpha-1)},$$

where varying α controls how aggressively or conservatively we measure the "distance" between X and Y .

Constructing the L 1 -Wasserstein Barycenter. In our approach, the L 1 -Wasserstein distance between ν and ν<sup>i</sup> is defined via

$$W_1(\nu, \nu_i) = \inf_{\rho \in \Gamma(\nu, \nu_i)} \mathbb{E}_{(X, Y) \sim \rho}[D_{f_\alpha}(X, Y)]. \quad (1)$$

The L 1 -Wasserstein barycenter then solves

$$\bar{\nu} = \arg \inf_{\nu} \left\{ \sum_{i=1}^n w_i W_1(\nu, \nu_i) \right\},$$

i.e., we seek the single distribution ν¯ that jointly minimizes its L 1 -Wasserstein distance (defined via the α-divergence) to all the ν<sup>i</sup> .

Why L instead of L . Using the L <sup>1</sup> distance in equation [1](#page-2-0) naturally leads to a backup rule resembling the *power mean* operator Proposition [1.](#page-3-0) This power-mean update is more robust to high-variance samples and connects smoothly to both the average backup (when α → 0 or p = 1) and the max backup (as α → ∞ or p → ∞). Hence, L 1 - Wasserstein with α-divergences offers a principled way to blend distributions in highly stochastic environments while controlling the balance between underestimation and overestimation in the final backup.

Why Use an α-Divergence Instead of L<sup>2</sup>? Although αdivergences are not strict metrics (they can be asymmetric and need not satisfy the triangle inequality), their use within an L 1 -Wasserstein framework provides distinct benefits for MCTS under stochastic or partially observable conditions:

- *Greater Flexibility via Generalized Means.* When combined with the L 1 -Wasserstein distance, an α-divergence naturally yields a *power-mean* style backup operator [\(Dam et al., 2019\)](#page-9-3). By adjusting the parameter α, one smoothly interpolates between average-like and max-like backups, allowing precise control over how conservative or aggressive the updates should be. This stands in contrast to L 2 -based distances, which only yield fixed (e.g. purely quadratic) aggregation behavior.
- *Robustness to Stochastic Variations.* Because αdivergences can emphasize or de-emphasize portions of the distribution differently depending on α, they help mitigate overestimation or underestimation in highly stochastic settings. Empirical studies in distributional RL [\(Metelli et al., 2019\)](#page-9-2) suggest that more adaptive divergence measures can significantly improve stability and performance when the underlying dynamics involve heavy noise.
- *No Need for Symmetry in Backups.* MCTS requires a *cost functional* to aggregate posterior distributions across children nodes, rather than a strict metric. Hence, the lack of symmetry or the triangle inequality does not undermine its validity here. An f-divergence—including αdivergences—is sufficient to drive consistent updates of belief distributions in the tree.
- *Unified Framework for Various Divergences.* The αdivergence family subsumes and generalizes many standard divergences (e.g. KL, reverse KL). This singleparameter approach enables users to easily switch or finetune the update behavior for different problem characteristics, rather than designing separate algorithms for each

divergence.

- *Direct Theoretical Connections.* Under mild assumptions, L 1 -Wasserstein geometry paired with α-divergences admits closed-form or near-closed-form power-mean formulas [\(Dam et al., 2019\)](#page-9-3). This not only streamlines theoretical analysis but also simplifies implementation by allowing straightforward computation of mean and variance updates at each node.

In practice, these properties make α-divergences wellsuited for uncertainty propagation within MCTS: despite not being a metric, their adaptability and connection to generalized means allow them to effectively handle complex, high-variance environments.

#### 5.3. V-posterior

It is natural to define a value node as the V-posterior computed with L 1 -Wasserstein barycenters of the children nodes Q-posteriors, following a procedure inspired by Metelli et al. 2019 [\(Metelli et al., 2019\)](#page-9-2) and tailored to MCTS.

Definition 1 (V-posterior). *Given a policy* π¯ *and a state* s ∈ S*, we define the V-posterior* V(s) *induced by Qposteriors* Q(s, a) *with* a ∈ A *as the* L 1 *-Wassertein barycenter of the* Q*(s, a):*

$$\mathcal{V}(s) \in \arg \inf_{\mathcal{V}} \left\{ \mathbb{E}_{a \sim \bar{\pi}(\cdot | s)} [W_1(\mathcal{V}, \mathcal{Q}(s, a))] \right\}.$$

In this work, we model each node in the tree as a Gaussian distribution. We define p = 1−α and derive the following.

Proposition 1. *Consider the V-posterior value function* V(s) *as a Gaussian:* N (m(s), σ 2 (s))*. Define each* Q(s, a) *as the action-value function child node of* V(s)*. Each* Q(s, a) *is assumed as a Gaussian distributions* Q(s, a) : N (m(s, a), σ(s, a) 2 )*. If the value function* V(s) *is defined as the Wasserstein barycenter of the action-value function* Q(s, a)*, given the policy* π¯*, we have*

$$\begin{aligned}\overline{m}(s) &= (\mathbb{E}_{a \sim \pi}[m(s, a)^p])^{\frac{1}{p}} \\ \overline{\delta}(s) &= (\mathbb{E}_{a \sim \pi}[\delta(s, a)^p])^{\frac{1}{p}}.\end{aligned}$$

Proposition [1](#page-3-0) shows the closed form solutions of the mean and standard deviation of the Gaussian value function V(s) considering it as the L 1 -Wasserstein barycenter Qposteriors. In detail, the mean of V(s) are the power mean of all mean values of all the Q(s, a) function, considering the finite set of actions. When p = 1, we derive the expected form solutions.

We point out that our approach is not restricted to the Gaussian distribution model. We get the following result by considering each tree node as a particle model.

Proposition 2. *Consider the V-posterior value function* V(s) *as an equally weighted Particle model:* xi(s) : i ∈ [1, M]*.* M *is an integer and* M ⩾ 1*. Assume each action-value function* Q(s, a) *has* M *particles* xi(s, a), i ∈ [1, M]*. If the value function* V(s) *is defined as the Wasserstein barycenter of the action-value function* Q(s, a)*, given the policy* π¯*, each particle* xi(s), i ∈ [1, M] *can be estimated as*

$$\overline{x_i}(s) = (\mathbb{E}_{a \sim \pi}[x_i(s, a)^p])^{1/p},$$

Proposition [2](#page-4-0) shows that each particle of the V-posterior value function V(s) can be derived as the power mean of the respective particles of all the Q(s, a) function. If p = 1, we again get the closed-form solutions as the expectation of the respective particles of all the Q(s, a) functions. The results in Proposition [1,](#page-3-0) and Proposition [2](#page-4-0) can be considered as the generalized result of Proposition A.3 in [Metelli](#page-9-2) [et al.](#page-9-2) [\(2019\)](#page-9-2). In the next section, we present our Wasserstein Monte-Carlo tree search (W-MCTS ) algorithm, assuming each tree node is a Gaussian distribution.

## 6. Wasserstein Monte-Carlo Tree Search

We introduce our Wasserstein Monte-Carlo Tree Search (W-MCTS), where V-posteriors are modeled as Wasserstein barycenters of action-value distributions. With Gaussian distributions at each node, we define backup operators for mean and variance. Additionally, we propose two action selection strategies: optimistic selection and Thompson sampling.

#### 6.1. Backup Operator

We model each V -node and Q-node as a Gaussian with mean and standard deviation:

$$V_m(s), V_{std}(s)$$
 and  $Q_m(s, a), Q_{std}(s, a)$ .

We denote V <sup>m</sup>(s, N(s)) as the *empirical mean estimate* of the V -node at state s after N(s) total visits, and Qm(s, a, n(s, a)) as the *empirical mean estimate* of the Qnode at (s, a) after n(s, a) visits. Likewise, V std(s, N(s)) and Qstd(s, a, n(s, a)) are their corresponding *empirical standard deviation estimates*.

V -nodes. From Proposition [1,](#page-3-0) the mean and the standard deviation of a V -node is a power-mean aggregation of its Q-children:

$$\overline{V}_m(s, N(s)) \leftarrow \left( \sum_a \frac{n(s,a)}{N(s)} [\overline{Q}_m(s, a, n(s, a))]^p \right)^{1/p},$$

$$\overline{V}_{\text{std}}(s, N(s)) \leftarrow \left( \sum_a \frac{n(s,a)}{N(s)} [\overline{Q}_{\text{std}}(s, a, n(s, a))]^p \right)^{1/p},$$

where n(s, a) is the visit count of action a at state s, and N(s) = P <sup>a</sup> n(s, a). For p = 1, this reduces to the standard average, whereas p > 1 induces a more "max-like" backup [\(Dam et al., 2019\)](#page-9-3).

Q-nodes. Under the Bellman-style backup for each Qnode,

$$Q_m(s, a) = \mathbb{E}[r(s, a)] + \gamma \mathbb{E}[V_m(s')], \quad Q_{\text{std}}(s, a) = \gamma V_{\text{std}}(s'),$$

we replace expectations by empirical sums and visitation counts:

$$\overline{Q}_m(s, a, n(s, a)) \leftarrow \frac{\sum r(s, a) + \gamma \sum_{s'} N(s') \overline{V}_m(s', N(s'))}{n(s, a)},$$

$$\overline{Q}_m(s, a, n(s, a)) \leftarrow \frac{\sum r(s, a) + \gamma \sum_{s'} N(s') \overline{V}_m(s', N(s'))}{n(s, a)},$$

$$\overline{Q}_{\text{std}}(s, a, n(s, a)) \leftarrow \frac{\gamma \sum_{s'} \frac{N(s')}{n(s, a)} \overline{V}_{\text{std}}(s')}{n(s, a)}.$$

Here, the sums range over transitions and children states s ′ , weighted by their visit counts N(s ′ ). As n(s, a) grows large, both the variance and mean estimators stabilize, eventually converging to deterministic values.

#### 6.2. Action Selection

Monte Carlo Tree Search can adopt a variety of exploration strategies based on the original UCT framework [\(Kocsis](#page-9-1) [et al., 2006\)](#page-9-1). In practice, multiple refinements exist, such as the variants used in AlphaGo [\(Silver et al., 2016b\)](#page-9-7), AlphaZero [\(Silver et al., 2017c](#page-10-3)[;a\)](#page-9-8), MuZero [\(Schrittwieser](#page-9-9) [et al., 2020\)](#page-9-9), Stochastic MuZero [\(Antonoglou et al., 2021\)](#page-8-5), and Stochastic-Power-UCT [\(Dam et al., 2024b\)](#page-9-10). Although different choices of the exploration constant or bonus lead to different performance characteristics, we retain the standard, state-of-the-art designs described below. In our theoretical analysis, however, we focus specifically on Thompson sampling, since the UCT-like optimistic selection can be viewed as a special case of the well-studied Power-UCT algorithm [\(Dam et al., 2019;](#page-9-3) [2024b\)](#page-9-10).

Optimistic Selection. A classic UCT-style selection picks actions using upper confidence bounds on Q-values,

$$a = \operatorname{argmax}_{a_i} \left[ m(s, a_i) + C \sqrt{\frac{\log N(s)}{n(s, a_i)}} \right],$$

where m(s, ai) is the empirical mean, n(s, ai) is the visit count of action a<sup>i</sup> , and N(s) is the total visit count at state <sup>s</sup>. Replacing the √ 1 n(s,ai) term by the empirical standard deviation σ(s, ai) yields an *optimistic* variant of Wasserstein MCTS (W-MCTS-OS):

$$a = \operatorname{argmax}_{a_i} \left[ m(s, a_i) + C \sigma(s, a_i) \sqrt{\log N(s)} \right].$$

Thompson Sampling. In contrast, Thompson sampling stochastically samples an action from the Q-posterior:

$$a = \operatorname{argmax}_{a_i} \{ \theta_i \sim \mathcal{N}(m(s, a_i), \sigma^2(s, a_i)) \}.$$

We refer to this Thompson variant as *Wasserstein MCTS-TS* (W-MCTS-TS). In Section [7,](#page-5-0) we analyze its convergence properties under non-stationary multi-armed bandits and then leverage these results to establish convergence in the planning tree.

### 7. Theoretical Analysis

#### 7.1. Analysis Setup

We define the setting for our theoretical analysis using a class of non-stationary Multi-Armed Bandit (MAB) problems at each state s in the MCTS tree. Consider K arms (actions), each with a mean reward µk, for k ∈ [K]. At time step t, pulling arm k yields a random reward Xk,t, bounded within [0, R]. The average reward for arm k after n trials is:

$$\overline{X}_{k,n} = \frac{1}{n} \sum_{t=1}^n X_{k,t}, \quad \text{with} \quad \mu_{k,n} = \mathbb{E}[\overline{X}_{k,n}]$$

Let ⋆ represent quantities related to the optimal arm, and denote Tk(n) as the number of times arm k has been played by step n. We assume the following *concentration* condition holds:

Assumption 1. *We assume that the reward sequence,* {Xk,t : t ⩾ 1}*, is a non-stationary process satisfying the assumption: for all* 1 > ε > 0*,* ∃c > 0 *that*

$$\Pr \left( |\overline{X}_{k,n} - \mu_k| > \varepsilon \right) \leq cn^{-1}\varepsilon^{-2}, k \in [K]. \quad (2)$$

#### 7.2. Main Results

We show the polynomial convergence of the expected estimated mean value function at the root node in Theorem [1.](#page-5-1)

## 7.2.1. CONVERGENCE OF W-MCTS

#### We start with an important result as shown below

Proposition 3. *Applying* W-MCTS *to an MCTS tree of depth* (H)*, at any depth* h *of the tree, we have*

- *(i) At any depth* h*,* ∃ *constant* C<sup>0</sup> > 0 *that for any* 0 < ε < 0, n ⩾ 1*, we can derive*

$$\Pr\left(\left|\overline{V}_m(s_h, a_k, n) - \tilde{V}(s_h, a_k)\right| \geq \varepsilon\right) \leq C_0 n^{-1} \varepsilon^{-2}.$$

- *(ii) At any depth* h*,* ∃ *constant* C<sup>0</sup> > 0 *that for any* 0 < ε < 0, n ⩾ 1*, we can derive*

$$\Pr \left( \left| \overline{Q}_m(s_h, a_k, n) - \tilde{Q}(s_h, a_k) \right| \geq \varepsilon \right) \leq C_0 n^{-1} \varepsilon^{-2}.$$

#### Proof Sketch

MCTS as a Hierarchical Bandit Structure. The Monte Carlo Tree Search (MCTS) algorithm can be viewed as a hierarchy of multi-armed bandits (MABs), where each node in the search tree represents an independent bandit problem. In this framework, the reward for each node, or current bandit, is influenced by the performance of the bandit algorithms applied to its child nodes. Since the W-MCTS policy adapts dynamically to balance exploitation and exploration, the rewards at each node are inherently *non-stationary*. The proof of Theorem [1](#page-5-1) unfolds through three essential steps:

1. Analyzing Non-stationary Bandits The initial step focuses on the analysis of a non-stationary multi-armed bandit, which reflects the behavior of MABs at each MCTS node. We establish that if the rewards of these nonstationary bandits meet specific *concentration* properties, the regret induced by the W-MCTS algorithm will exhibit corresponding concentration guarantees. This outcome is formally stated in Theorem [2.](#page-6-0)

2. Induction Argument Next, we utilize an inductive argument to transfer the convergence and concentration properties from the lower tree levels to the root node. As the rewards from one level inform those of the next, the findings from Step 1 can be recursively applied. We begin at depth H − 1 and move upward, demonstrating inductively that the bandit rewards at each level H of the MCTS satisfy the criteria required by Theorem [2.](#page-6-0) This process propagates the desired properties up to the root node, completing the induction.

3. Error Analysis from the Oracle The final step examines the error introduced by the leaf node estimator, represented by the value function oracle V0. With this oracle, the depth-H MCTS can be interpreted as performing H steps of value iteration, starting from V<sup>0</sup> at the leaf nodes (as mentioned in [\(Dam et al., 2024b\)](#page-9-10)). Importantly, the oracle's error decreases geometrically at a rate of γ due to the contraction mapping property of value iteration, leading to diminishing error as we ascend from the leaf nodes to the root. The complete proof for Proposition [3](#page-5-2) can be found in the supplemental material. Finally, we get the main result.

Theorem 1. *We have at the root node* s0*,*

$$|\mathbb{E}[\overline{V}_m(s_0, n)] - \widetilde{V}(s_0)| \leq \mathcal{O}(n^{-1/2}).$$

Our proposed method, W-MCTS, achieves a polynomial convergence rate of O(n −1/2 ), matching the results of [Dam et al.](#page-9-10) [\(2024b\)](#page-9-10). In contrast, [Xiao et al.](#page-10-4) [\(2019\)](#page-10-4) introduced MENTS, followed by RENTS and TENTS from [Dam](#page-9-11) [et al.](#page-9-11) [\(2021\)](#page-9-11), which leverage exponential convergence to a regularized value function through maximum entropy regularization. However, these methods face bias due to errors in the regularized value function, potentially leading to incorrect action selection. Conversely, [Painter et al.](#page-9-12) [\(2024\)](#page-9-12) employ a similar action selection strategy with a maximum backup operator for value estimation, resulting in exponential reductions in simple regret. However, their method's effectiveness heavily relies on the temperature parameter in Boltzmann exploration, limiting its practical use.

#### 7.2.2. WASSERSTEIN NON-STATIONARY MULTI-ARMED BANDIT

A crucial part of the proof for Theorem [1](#page-5-1) is to derive the following result for the W-MCTS in bandit setting. Under the Assumption [1,](#page-5-3) we consider applying Thompson Sampling strategy as the action selection method for the nonstationary multi-armed bandit (MAB) problems describes above. At each time step n, an action is selected as

$$a = \operatorname{argmax}_{a_i, i \in \{1 \dots K\}} \{\theta_i \sim \mathcal{N}(\bar{X}_{k,n}, V_k/T_k(n))\}. \quad (3)$$

Let's define <sup>X</sup>n(p) = P<sup>K</sup> <sup>a</sup>=1 Ta(n) n X p a,Ta(n) 1/p as the power mean value backup at the root node, Ta(n) = P<sup>n</sup>−<sup>1</sup> <sup>t</sup>=1 <sup>1</sup>(a<sup>t</sup> = a) is the number of selections of a prior to round n. We show theoretical results of our method as follows. Under the Assumption [1,](#page-5-3) we establish the concentration properties of the power mean backup operator Xn(p) towards the mean value of the optimal arm µ<sup>∗</sup> = maxa{µa}, a ∈ [K], as shown in Theorem [2.](#page-6-0)

Theorem 2. *Consider a non-stationary bandit problem described as in [7.1](#page-5-4) with action selection as Equation [\(3\)](#page-5-5). Then,*

$$\mathbf{Pr}(|\overline{X}_n(p) - \mu_\star| \geq \varepsilon) \leq Cn^{-1}\varepsilon^{-2}.$$

Theorem [2](#page-6-0) states the concentration properties of the power mean estimation by W-MCTS for a non-stationary continuous-armed bandit problem, and play an important role for the induction proof of Proposition [3](#page-5-2) leading to the main result presented at Theorem [1.](#page-5-1)

## 8. Experiments

#### 8.1. Fully Observable, Highly Stochastic Tasks

We compare W-MCTS to UCT [\(Kocsis et al., 2006\)](#page-9-1), Power-UCT [\(Dam et al., 2019\)](#page-9-3), and DNG [\(Bai et al., 2013\)](#page-8-0) in five benchmark environments: *FrozenLake*, *NChain*, *RiverSwim*, *SixArms*, and *Taxi*. These tasks all feature significant stochasticity or long-horizon exploration challenges.

FrozenLake. A 4 × 4 grid with slippery transitions, implemented in OpenAI Gym [\(Brockman et al., 2016\)](#page-9-13). The agent aims to reach a goal in the bottom-right corner. Due to frequent slips, each move has high uncertainty. Figure [1](#page-7-0) shows that W-MCTS-TS (Thompson sampling) outperforms DNG, UCT, Power-UCT, and W-MCTS (optimistic selection), with W-MCTS at p = 1 performing comparably to W-MCTS-TS.

NChain. An agent can move forward or backward along a chain of length 5. Actions may reverse with 20% probability, making consistent forward progress difficult. In Figure [1,](#page-7-0) both W-MCTS-TS and W-MCTS-OS exceed UCT and Power-UCT in convergence speed and final returns.

RiverSwim. Similar to *NChain* but more complex transitions: sometimes the agent remains in the same state or only partially moves. This rewards long-term planning to reach high-value states. As in Figure [1,](#page-7-0) W-MCTS-OS converges fastest and attains the best performance, while Power-UCT eventually reaches similar returns more slowly.

SixArms. A 7-state chain with 6 possible arms (actions) leading to different rewards that scale inversely with their success probabilities. This environment demands high exploration. Figure [1](#page-7-0) shows that W-MCTS is the only method consistently securing strong returns.

Taxi. A 7 × 6 grid where the agent must pick up three passengers, then reach a goal region. Slips occur 10% of the time, adding further uncertainty. Only W-MCTS-TS manages to collect all passengers reliably, outperforming Power-UCT and W-MCTS with optimistic selection.

#### 8.2. Partially Observable, Highly Stochastic Tasks

We also test W-MCTS against POMCP(UCT), D2NG, and DESPOT in classic POMDP benchmarks: *rocksample*, *pocman*, *Tag*, and *LaserTag*. Code for POMCP(UCT) [\(Silver](#page-9-14) [& Veness, 2010b\)](#page-9-14), D2NG [\(Bai et al., 2014\)](#page-8-1), and DESPOT [\(Somani et al., 2013\)](#page-10-5) is used as released by the original authors.

Rocksample. A robot on an n×n grid can sample or ignore k rocks, then exit. We test three variants: (11,11), (15,15), and (15,35). Figure [2](#page-7-1) shows that W-MCTS-TS consistently outperforms both UCT and D2NG.

Pocman. A partially observed maze [\(Silver & Veness,](#page-9-15) [2010a\)](#page-9-15) where the agent must collect pellets while avoiding ghosts. Table [1](#page-7-2) indicates that W-MCTS-TS with p = 100 outperforms UCT and D2NG across most rollout-budget settings, and W-MCTS-OS also matches or surpasses these baselines in some configurations.

Comparison with **DESPOT**. We additionally compare W-MCTS to DESPOT across *Tag*, *LaserTag*, *rocksample* (15 × 15), and *Pocman*. Table [2](#page-7-3) shows that W-MCTS-OS and W-MCTS-TS achieve higher returns than AB-DESPOT and AR-DESPOT in rocksample. Similarly, W-MCTS-TS surpasses DESPOT in *Pocman*, *Tag*, and *LaserTag*, while W-MCTS-OS outperforms AB-DESPOT in *Pocman*. Role

![](_page_7_Figure_1.jpeg)

Figure 1: Performance of W-MCTS vs. DNG, Power-UCT, and UCT on five MDPs. Each curve shows the mean discounted return (averaged over 50 runs), with shaded regions indicating standard error.

![](_page_7_Figure_3.jpeg)

Figure 2: Performance of W-MCTS vs. D2NG in *rocksample*, averaged over 1000 runs (except UCT, 100 runs). Shaded areas denote standard error.

Table 1: Discounted total reward in *pocman*. Mean ± standard error are computed from 1000 random seeds.

|                     | 1024         | 4096         | 32768        | 65536        |
|---------------------|--------------|--------------|--------------|--------------|
| W-MCTS-OS , p = 1   | 50 9 ± 0 6   | 51 0 ± 0 62  | 52 2 ± 0 79  | 54 6 ± 1 08  |
| W-MCTS-TS , p = 100 | 67 38 ± 0 53 | 75 64 ± 0 51 | 77 68 ± 0 77 | 77 70 ± 1 22 |
| D2NG                | 71 55 ± 0 57 | 75 39 ± 1 47 | 76 90 ± 6 40 | 72 2 ± 0 0   |
| UCT                 | 23 4 ± 0 99  | 23 6 ± 1 09  | 24 90 ± 3 40 | 28 5 ± 3 8   |

Table 2: Average total discounted reward. The results for POMCP, and DESPOT are taken from [\(Somani et al., 2013\)](#page-10-5).

|           |     |    | T ag |    |      |    |     |    |    | RS | (15 , | 15) |     |    |     |    |
|-----------|-----|----|------|----|------|----|-----|----|----|----|-------|-----|-----|----|-----|----|
| W-MCTS-OS | − 6 | 05 | ± 0  | 56 | − 18 | 17 | ± 0 | 46 | 19 | 76 | ± 0   | 28  | 297 | 98 | × 2 | 83 |
| W-MCTS-TS | − 5 | 90 | ± 0  | 66 | − 8  | 75 | ± 0 | 5  | 20 | 29 | ± 0   | 22  | 315 | 45 | ± 2 | 15 |
| POMCP     | − 7 | 14 | ± 0  | 28 | − 19 | 58 | ± 0 | 06 | 12 | 23 | ± 0   | 32  | 294 | 16 | ± 4 | 06 |
| AB-DESPOT | − 6 | 57 | ± 0  | 26 | − 11 | 13 | ± 0 | 30 | 18 | 18 | ± 0   | 30  | 290 | 34 | ± 4 | 12 |
| AR-DESPOT | − 6 | 26 | ± 0  | 28 | − 9  | 34 | ± 0 | 26 | 18 | 57 | ± 0   | 30  | 307 | 96 | ± 4 | 22 |

of α-Divergence. We explored several values of α to vary how aggressively our backups shift between average-like and max-like behavior. When α approaches 0 or ∞, the update becomes nearly a pure average (p = 1) or nearly a max backup, respectively. In practice, we found that moderate α values often provide a suitable balance between these extremes, and we report results with the bestperforming choices. Although a more extensive sensitivity analysis could be conducted, the core takeaway is that combining power-mean backups with variance propagation significantly enhances performance in highly stochastic tasks.

## 8.3. Key Performance Factors

itations in existing MCTS approaches for stochastic and partially observable environments:

Explicit Variance Propagation. Unlike previous methods that only propagate point estimates or use fixed variance models, our approach dynamically updates both means and variances at each node through the L 1 -Wasserstein barycenter formulation. This capability is particularly crucial in highly stochastic and partially observable environments where uncertainty quantification directly impacts decision quality. Our experimental results demonstrate consistent improvements over Bayesian MCTS methods: we achieve up to 80% improvement over DNG in *Frozen-Lake*, and significant gains over POMCP across all POMDP environments, with particularly notable improvements of 55.31% in *LaserTag* and 65.90% in *rocksample*(15,15). Additionally, we observe improvements of up to 21.38% over AB-DESPOT in *LaserTag*, highlighting the effectiveness of our distributional approach.

Flexibility in Balancing Exploration-Exploitation. Our approach's ability to interpolate between average-like and max-like backups through the α-divergence parameter allows adaptive behavior across varying levels of stochasticity. In highly stochastic environments such as *FrozenLake* and *NChain*, we found that moderate α values (leading to more average-like updates with p closer to 1) performed optimally by preventing overestimation bias. Conversely, in environments with more deterministic regions of the state space, larger α values (yielding more max-like behavior) proved beneficial for faster convergence to optimal policies. This flexibility, combined with our Thompson sampling strategy, enables our algorithm to automatically adapt its exploration-exploitation balance based on the empirical variance observed at each node.

The synergy between these two components—principled uncertainty propagation and adaptive backup operators explains why W-MCTS consistently outperforms both classical MCTS variants and existing Bayesian approaches across our diverse set of benchmark environments.

## 9. Conclusion

We proposed *Wasserstein MCTS*, an algorithm that models node values as Gaussian distributions and employs L 1 -Wasserstein barycenters with α-divergences to unify average- and max-like backups. Coupled with Thompson sampling or optimistic selection, our method achieves strong empirical performance while offering O(n −1/2 ) convergence guarantees. Experiments in both stochastic MDPs and POMDPs show significant improvements over classic baselines and Bayesian MCTS variants. Future work includes extending these Wasserstein-based ideas to open-loop planning [\(Leurent & Maillard, 2020;](#page-9-16) [Bubeck &](#page-9-17) [Munos, 2010\)](#page-9-17) for even broader applicability.

## Impact Statement

Our proposed *Wasserstein MCTS* algorithm offers a principled way to tackle complex, stochastic tasks in both fully and partially observable domains. Potential applications include robotics, autonomous systems, and large-scale resource management, all of which require adaptive planning strategies to handle real-world variability. While we do not anticipate immediate negative societal implications, responsible deployment remains essential. As with any AIdriven technology, understanding ethical, economic, and security ramifications—such as autonomy in safety-critical systems—should guide practical use.

#### Acknowledgments

This research is funded by Hanoi University of Science and Technology (HUST) under Project No. T2024-TD-024, the French Ministry of Higher Education and Research, the Hautsde-France region, Inria, the MEL, the French National Research Agency under PEPR IA FOUNDRY project (ANR-23-PEIA-0003).

## References


[1] Antonoglou, I., Schrittwieser, J., Ozair, S., Hubert, T. K., and Silver, D. Planning in stochastic environments with a learned model. In *International Conference on Learning Representations*, 2021. Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. *Machine learning*, 47(2):235–256, 2002a. Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. *Mach. Learn.*, 47(2–3):235–256, may 2002b. ISSN 0885-6125. doi: 10. 1023/A:1013689704352. URL [https://doi.org/10.1023/A:](https://doi.org/10.1023/A:1013689704352) [1013689704352.](https://doi.org/10.1023/A:1013689704352) Bai, A., Wu, F., and Chen, X. Bayesian mixture modelling and inference based thompson sampling in monte-carlo tree search. *Advances in neural information processing systems*, 26, 2013. Bai, A., Wu, F., Zhang, Z., and Chen, X. Thompson sampling based monte-carlo planning in pomdps. *the International Conference on Automated Planning and Scheduling*, 24(1), 2014. Bellman, R. The theory of dynamic programming. Technical report, Rand corp santa monica ca, 1954.

[2] Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. Openai gym. *arXiv preprint arXiv:1606.01540*, 2016. Bubeck, S. and Munos, R. Open loop optimistic planning. In *COLT 2010-The 23rd Conference on Learning Theory*, 2010. Cichocki, A. and Amari, S.-i. Families of alpha- betaand gamma- divergences: Flexible and robust measures of similarities. *Entropy*, 12(6):1532–1568, 2010. ISSN 1099-4300. doi: 10.3390/e12061532. URL [https://www.](https://www.mdpi.com/1099-4300/12/6/1532) [mdpi.com/1099-4300/12/6/1532.](https://www.mdpi.com/1099-4300/12/6/1532) Csiszar, I. Eine informationstheoretische ungleichung und ´ ihre anwendung auf beweis der ergodizitaet von markoffschen ketten. *Magyer Tud. Akad. Mat. Kutato Int. Koezl.*, 8:85–108, 1964. Dam, T., Klink, P., D'Eramo, C., Peters, J., and Pajarinen, J. Generalized mean estimation in monte-carlo tree search. *arXiv preprint arXiv:1911.00384*, 2019. Dam, T., D'Eramo, C., Peters, J., and Pajarinen, J. A unified perspective on value backup and exploration in monte-carlo tree search. *Journal of Artificial Intelligence Research*, 81:511–577, 2024a. Dam, T., Maillard, O.-A., and Kaufmann, E. Power mean estimation in stochastic monte-carlo tree search. *The 40th Conference on Uncertainty in Artificial Intelligence (UAI)*, 2024b. Dam, T. Q., D'Eramo, C., Peters, J., and Pajarinen, J. Convex regularization in monte-carlo tree search. In *International Conference on Machine Learning*, pp. 2365–2375. PMLR, 2021. Gerchinovitz, S., Menard, P., and Stoltz, G. Fano's in- ´ equality for random variables. *Statist. Sci*, 2020. Hasselt, H. V. Double q-learning. In *Advances in Neural Information Processing Systems*, 2010. Jin, T., Xu, P., Xiao, X., and Anandkumar, A. Finite-time regret of thompson sampling algorithms for exponential family multi-armed bandits. *Advances in Neural Information Processing Systems*, 35:38475–38487, 2022. Kocsis, L., Szepesvari, C., and Willemson, J. Improved ´ monte-carlo search. *Univ. Tartu, Estonia, Tech. Rep*, 1, 2006. Leurent, E. and Maillard, O.-A. Practical open-loop optimistic planning. In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019, Wurzburg, Germany, September 16–20, ¨ 2019, Proceedings, Part III*, pp. 69–85. Springer, 2020. Metelli, A. M., Likmeta, A., and Restelli, M. Propagating uncertainty in reinforcement learning via wasserstein barycenters. *Advances in Neural Information Processing Systems*, 32, 2019. Painter, M., Baioumy, M., Hawes, N., and Lacerda, B. Monte carlo tree search with boltzmann exploration. *Advances in Neural Information Processing Systems*, 36, 2024. Perlman, M. D. Jensen's inequality for a convex vectorvalued function on an infinite-dimensional space. *Journal of Multivariate Analysis*, 4(1):52–65, 1974. ISSN 0047-259X. doi: https://doi.org/10.1016/0047-259X(74) 90005-0. URL [https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/0047259X74900050) [article/pii/0047259X74900050.](https://www.sciencedirect.com/science/article/pii/0047259X74900050) Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., et al. Mastering atari, go, chess and shogi by planning with a learned model. *Nature*, 588 (7839):604–609, 2020. Silver, D. and Veness, J. Monte-carlo planning in large pomdps. In *Advances in neural information processing systems*, 2010a. Silver, D. and Veness, J. Monte-carlo planning in large pomdps. In Lafferty, J., Williams, C., Shawe-Taylor, J., Zemel, R., and Culotta, A. (eds.), *Advances in Neural Information Processing Systems*, volume 23. Curran Associates, Inc., 2010b. URL [https://proceedings.neurips.cc/paper/2010/file/](https://proceedings.neurips.cc/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf) [edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf.](https://proceedings.neurips.cc/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf) Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587):484–489, January 2016a. doi: 10.1038/nature16961. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484, 2016b. Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. *arXiv preprint arXiv:1712.01815*, 2017a.

[3] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., and Hassabis, D. Mastering the game of go without human knowledge. *Nature*, 550:354–, October 2017b. URL [http://dx.doi.org/](http://dx.doi.org/10.1038/nature24270) [10.1038/nature24270.](http://dx.doi.org/10.1038/nature24270) Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. Mastering the game of go without human knowledge. *Nature*, 550(7676):354–359, 2017c. Soch, J. The book of statistical proofs. *https://statproofbook.github.io*, 2020. URL [https://statproofbook.github.io/P/norm-qf.html.](https://statproofbook.github.io/P/norm-qf.html) Somani, A., Ye, N., Hsu, D., and Lee, W. S. Despot: Online pomdp planning with regularization. *Advances in neural information processing systems*, 26, 2013. Tesauro, G., Rajan, V., and Segal, R. Bayesian inference in monte-carlo tree search. *arXiv preprint arXiv:1203.3519*, 2012. Thompson, W. R. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4):285–294, 1933. Xiao, C., Huang, R., Mei, J., Schuurmans, D., and Muller, ¨

[4] M. Maximum entropy monte-carlo planning. In *Advances in Neural Information Processing Systems*, pp.

[5] 9516–9524, 2019.
#### Outline

- Notations will be described in Section A.
- Hyperparameters are provided in Section B.
- Derivation of Wasserstein barycenter with Gaussian and particle filter distributions will be described in Section C.
- Supporting Lemmas will be provided in Section D.
- Full proof for the convergence of Wasserstein Non-stationary multi-armed bandit will be provided in Section E.
- Full proof for the convergence of Wasserstein Monte-Carlo tree search will be provided in Section F.

## A. Notations

Table 3: List of all notations of Wasserstein barycenter with Gaussian and particle filter distributions.

| Notation     | Type  | Description                                              |
|--------------|-------|----------------------------------------------------------|
| N ( m, δ 2   |       |                                                          |
| )            | R     | Gaussian distribution with mean m , standard deviation δ |
| ( X , d )    |       | complete separable metric (Polish) space                 |
| W q ( µ, ν ) |       | L                                                        |
|              |       | -Wasserstein distance between µ, ν                       |
| W 1 ( µ, ν ) |       | L                                                        |
|              |       | -Wasserstein distance between µ, ν                       |
| − 1          |       |                                                          |
| p ( x )      |       |                                                          |
| ( t )        |       | quantile function of a distribution p ( x )              |
| Γ( µ, ν )    | X × Y | set of measures on X × Y with marginals µ, ν             |
| d ( X, Y )   | R     | distance between X and Y                                 |
| D f α        |       |                                                          |
| ( X    Y )   | R     | α -divergence distance between X and Y                   |
| erf − 1      |       |                                                          |
| ( t )        |       | the inverse of the function √                            |
|              |       | R t                                                      |
|              |       | exp {− x                                                 |
|              |       | 2 } dx                                                   |

## B. Experimental setup and Parameters selection

All the experiments were done on an Intel(R) Core(TM) i9-14900K 3.20 GHz 24 cores/CPU.

To compare the performance of W-MCTS to other state-of-the-art planning algorithms, we run several experiments on standard MDP as well as POMDP environments. For comparison, we consider UCT [\(Kocsis et al., 2006\)](#page-9-1), Power-UCT [\(Dam](#page-9-3) [et al., 2019\)](#page-9-3), DNG [\(Bai et al., 2013\)](#page-8-0) and D2NG [\(Bai et al., 2014\)](#page-8-1). The hyperparameters are tuned using grid-search. Except for the case of *Pocman* environment, we scale the rewards into the range [0, 1]. We use the discount factor γ = 0.95. For DNG, D2NG, we set hyperparameters as recommended in the paper and from the author's source code [\(Bai et al., 2013;](#page-8-0) [2014\)](#page-8-1). We set exploration constant for UCT, Power-UCT to √ 2. We set initial standard deviation value to std = 30. In all *Rocksample* and *Pocman* environments, we set the heuristic for rollouts as treeknowledge = 0, rolloutknowledge = 1. For all environments, we increase the value of p and choose the best power mean p value for Power-UCT, and W-MCTS . Details can be found in Table [6.](#page-13-0) For POMDP environments such as *Rocksample*, *Pocman* we get the source code released from the author of DNG [\(Bai et al., 2013\)](#page-8-0) and D2NG [\(Bai et al., 2014\)](#page-8-1)[<sup>1</sup>](#page-11-0) .

<sup>1</sup> https://github.com/aijunbai/thompson-sampling

Table 4: List of all notations of Wasserstein Non-stationary multi-armed bandits.

| Notation      | Type | Description                                           |
|---------------|------|-------------------------------------------------------|
| K             | N    | number of arms/actions                                |
| µ k           | R    | mean value of arm k                                   |
| ∗             | R    | optimal mean value                                    |
| △ k           | R    | △ k = µ                                               |
|               |      | ∗ − µ k                                               |
| △             | R    | △ = max k ∈ [ K ] {△ k }                              |
| s             | R    | average reward of the optimal arm after s visitations |
| s             | R    | CDF of Gaussian with mean X                           |
| T k ( n )     | N    | number of visitations of arm k at timesteps n         |
| X n ( p )     | R    | power mean backup operator with power p               |
| X k,T k ( n ) | R    | average rewards of arm k after T k ( n ) visits       |

## C. Derivation of Wasserstein barycenter with Gaussian and particle filter distributions

We revisit the definition of Wasserstein distance: The L q -Wasserstein distance (with q > 0) between two distributions µ, ν with the cost function d(x, y) : X × Y → <sup>R</sup> is defined as

$$W_q(\mu, \nu) = \left( \inf_{\rho \in \Gamma(\mu, \nu) X, Y \sim \rho} [d(X, Y)^q] \right)^{1/q}, \quad (4)$$

here Γ(µ, ν) is the set of measures on X × Y with marginals µ, ν.

Define F −1 p(x) (t) as the quantile function of a distribution

$$p(x) : F_{p(x)}^{-1}(t) = \inf\{x \in \mathbb{R}, t \leq F_p(x)\}. \quad (5)$$

With d(X, Y ) = |X − Y | as the Euclidean distance, we can derive

$$W_q^q(\mu, \nu) = \left( \int_0^1 |F_\mu^{-1}(t) - F_\nu^{-1}(t)|^q dt \right). \quad (6)$$

With d(X, Y ) = Df<sup>α</sup> (X||Y ), as the α-divergence distance (defined in section 4.1), we can derive

$$W_q^q(\mu, \nu) = \left( \int_0^1 D_{f_\alpha}(F_\mu^{-1}(t)||F_\nu^{-1}(t))^q dt \right). \quad (7)$$

#### C.1. L 1 -Wasserstein barycenter with α-divergence distance

We have

$$W_1(\mu, \nu) = \inf_{\rho \in \Gamma(\mu, \nu) \times Y \sim \rho} [d(X, Y)] = \inf_{\rho \in \Gamma(\mu, \nu) \times Y \sim \rho} [D_{f_a}(X, Y)]. \quad (8)$$

Table 5: List of all notations of Wasserstein Monte-Carlo Tree Search.

| Notation              | Type | Description                                                                                              |
|-----------------------|------|----------------------------------------------------------------------------------------------------------|
| KL                    |      | KL divergence                                                                                            |
| V m ( s h )           | R    | optimal mean of V value at root state s h , at depth ( h )                                               |
| Q m ( s h , a k )     | R    | mean of Q value function at state s h , action a k , at depth ( h )                                      |
| V m ( s h , n )       | R    | empirical estimated mean of V value at state s h after n visitations at depth ( h )                      |
| Q m ( s h , a k , n ) | R    | empirical estimated mean of Q value at root at state s h , action a k after n visitations at depth ( h ) |
| V m ( s h )           | R    | optimal mean of V value at depth ( h ) at state s h                                                      |
| Q m ( s h , a k )     | R    | mean of Q value function at depth ( h ) at state s h , action a k                                        |
| V m ( s h , n )       | R    | empirical estimated mean of V value at depth ( h ) at state s h after n visitations                      |
| Q m ( s h , a k , n ) | R    | empirical estimated mean of Q value at depth ( h ) at state s h , action a k after n visitations         |
| T s h ,a k            |      |                                                                                                          |
| ( n )                 | N    | number of plays of action a k at state s h at timestep n                                                 |
| s,a k                 |      |                                                                                                          |
| ( n )                 | N    | number of plays of action a k at state s to state s                                                      |
|                       |      | at timestep n                                                                                            |

Table 6: List of all hyperparameters.

| Environments      |     | p   | Value |   |    |   |   |    | Search |    |       |       |       |           |           | Best      | p Value                  |
|-------------------|-----|-----|-------|---|----|---|---|----|--------|----|-------|-------|-------|-----------|-----------|-----------|--------------------------|
| FrozenLake        |     | p = |       | 1 | ,  | 2 | , | 4  | ,      | 10 |       | , 100 |       | W-MCTS-OS | (p=100),  | W-MCTS-TS | (p=100),Power-UCT(p=100) |
| NChain            | p   | =   | 1     | , | 2  | , | 4 | ,  | 8      | ,  | 15    | ,     | 100   | W-MCTS-OS | (p=4),    | W-MCTS-TS | (p=100),Power-UCT(p=8)   |
| RiverSwim         | p   | =   | 1     | , | 2  | , | 4 | ,  | 8      | ,  | 15    | ,     | 100   | W-MCTS-OS | (p=100),  | W-MCTS-TS | (p=15),Power-UCT(p=15)   |
| SixArms           | p   | =   | 1     | , | 2  | , | 4 | ,  | 8      | ,  | 15    | ,     | 100   | W-MCTS-OS | (p=100),  | W-MCTS-TS | (p=100),Power-UCT(p=8)   |
| Taxi              | p   | =   | 1     | , | 2  | , | 4 | ,  | 8      | ,  | 15    | ,     | 100   | W-MCTS-OS | (p=15),   | W-MCTS-TS | (p=15),Power-UCT(p=15)   |
| Rocksample(11x11) | p = |     | 10    | , | 50 |   | , | 80 |        | ,  | 100   |       | , 150 |           | W-MCTS-OS | (p=150),  | W-MCTS-TS (p=100)        |
| Rocksample(15x15) | p = |     | 10    | , | 50 |   | , | 80 |        | ,  | 100   |       | , 150 |           | W-MCTS-OS | (p=100),  | W-MCTS-TS (p=100)        |
| Rocksample(15x35) |     | p   | =     |   | 10 |   |   | ,  | 80     |    | , 100 |       |       |           | W-MCTS-OS | (p=150),  | W-MCTS-TS (p=10)         |
| Pocman            | p   | =   | 1     | , | 2  | , | 4 | ,  | 8      | ,  | 10    | ,     | 100   |           | W-MCTS-OS | (p=1),    | W-MCTS-TS (p=100)        |

We find the lower bound of W1(µ, ν) with α-divergence as a measure cost function.

Let denote N (m, δ<sup>2</sup> ) as a Gaussian distribution with mean m and standard deviation δ. With µ = N (m1, δ<sup>2</sup> 1 ), ν = N (m2, δ<sup>2</sup> 2 ) We first want to show that by applying Data Processing Inequalities (Lemma 2.1 [\(Gerchinovitz et al., 2020\)](#page-9-18)), with h(X) = X − m1, and g(X) = X − m2, we can derive

$$\begin{aligned} W_1(\mu, \nu) &= \inf_{\rho \in \Gamma(\mu, \nu) X, Y \sim \rho} \mathbb{E}[D_{f_\alpha}(X, Y)] \geq \inf_{\rho \in \Gamma(\mu, \nu) X, Y \sim \rho} [D_{f_\alpha}(X - m_1, Y - m_1)] \\ &= W_1(\mathcal{N}(0, \delta_1^2), \mathcal{N}(m_2 - m_1, \delta_2^2)), \end{aligned} \tag{9}$$

and

$$\begin{aligned} W_1(\mu, \nu) &= \inf_{\rho \in \Gamma(\mu, \nu) \times Y \sim \rho} \mathbb{E}[D_{f_\alpha}(X, Y)] \geq \inf_{\rho \in \Gamma(\mu, \nu) \times Y \sim \rho} [D_{f_\alpha}(X - m_2, Y - m_2)] \\ &\geq \inf_{\rho \in \Gamma(\mu, \nu) \times Y \sim \rho} \mathbb{E}[D_{f_\alpha}(m_2 - X, m_2 - Y)] \text{ (with the transform function } f(X) = -X) \\ &= W_1(\mathcal{N}(m_2 - m_1, \delta_1^2), \mathcal{N}(0, \delta_2^2)). \end{aligned} \tag{10}$$

Now according to [\(7\)](#page-12-0), the L 1 -Wasserstein distance with α-divergence distance is defined as

$$W_1(\mu, \nu) = \left( \int_0^1 D_{f_\alpha}(F_\mu^{-1}(t)||F_\nu^{-1}(t))dt \right). \quad (11)$$

We show that the quantile function of a Gaussian distribution [\(Soch, 2020\)](#page-10-6) F = N (µ, δ<sup>2</sup> ) is

$$F^{-1}(t) = \sqrt{2} \operatorname{erf}^{-1}(2t-1) + \mu, \quad (12)$$

where erf−<sup>1</sup> (t) is the inverse of the function √ 2 π R t 0 exp{−x <sup>2</sup>}dx.

Therefore, the L 1 -Wasserstein distance with α-divergence distance as the cost function between two Gaussian distributions µ = N (m1, δ<sup>2</sup> 1 ), ν = N (m2, δ<sup>2</sup> 2 ) can be measured as

$$W_1(\mu, \nu) = \left( \int_0^1 D_{f_\alpha}(\sqrt{2}\delta_1 \operatorname{erf}^{-1}(2t-1) + m_1 || \sqrt{2}\delta_2 \operatorname{erf}^{-1}(2t-1) + m_2) dt \right).$$

Applying the convexity properties of α-divergence [\(Cichocki & Amari, 2010\)](#page-9-19), and from [\(9\)](#page-14-0),[\(10\)](#page-14-1) we have

$$\begin{aligned} W_1(\mu, \nu) &\geq \frac{1}{2} \left( \int_0^1 D_{f_\alpha}(\sqrt{2}\delta_1 \text{erf}^{-1}(2t-1) || \sqrt{2}\delta_2 \text{erf}^{-1}(2t-1) + m_2 - m_1) dt \right. \\ &\quad \left. + \int_0^1 D_{f_\alpha}(\sqrt{2}\delta_1 \text{erf}^{-1}(2t-1) + m_2 - m_1 || \sqrt{2}\delta_2 \text{erf}^{-1}(2t-1)) dt \right) \\ &\geq \left( \int_0^1 D_{f_\alpha}(\sqrt{2}\delta_1 \text{erf}^{-1}(2t-1) + \frac{m_2 - m_1}{2} || \sqrt{2}\delta_2 \text{erf}^{-1}(2t-1) + \frac{m_2 - m_1}{2}) dt \right) \\ &= W_1\left(\mathcal{N}\left(\frac{m_2 - m_1}{2}, \delta_1^2\right), \mathcal{N}\left(\frac{m_2 - m_1}{2}, \delta_2^2\right)\right). \end{aligned}$$

Applying Data Processing Inequalities (Lemma 2.1 [\(Gerchinovitz et al., 2020\)](#page-9-18)), with h(X) = X − m2−m<sup>1</sup> 2 , we can derive

$$W_1(\mu, \nu) \geq W_1(\mathcal{N}(0, \delta_1^2), \mathcal{N}(0, \delta_2^2)) = \left( \int_0^1 D_{f_\alpha}(\sqrt{2}\delta_1 \operatorname{erf}^{-1}(2t-1) || \sqrt{2}\delta_2 \operatorname{erf}^{-1}(2t-1)) dt \right).$$

Let us consider the sequences 0 = t<sup>0</sup> ⩽ t<sup>1</sup> ⩽ ... ⩽ t<sup>N</sup> = 1, there exists ξ<sup>i</sup> ∈ [t<sup>i</sup> , ti+1] that

$$\begin{aligned} W_1(\mu, \nu) &\geq \sum_{i=0}^{i=N} (t_{i+1} - t_i) D_{f_\alpha}(\sqrt{2}\delta_1 \text{erf}^{-1}(2\xi_i - 1) \|\sqrt{2}\delta_2 \text{erf}^{-1}(2\xi_i - 1)) \\ &= \sum_{i=0}^{i=N} \Delta_i D_{f_\alpha}(\sqrt{2}\delta_1 \text{erf}^{-1}(2\xi_i - 1) \|\sqrt{2}\delta_2 \text{erf}^{-1}(2\xi_i - 1)), \end{aligned}$$

with ∆<sup>i</sup> = (ti+1 − ti). Since Df<sup>α</sup> (cP||cQ) = Df<sup>α</sup> (P||Q) where c is a constant. We can derive

$$W_1(\mu, \nu) \geq \sum_{i=0}^{i=N} \Delta_i D_{f_\alpha}(\delta_1 || \delta_2) = D_{f_\alpha}(\delta_1 || \delta_2). \quad (13)$$

We start with the first Proposition about the closed solutions of mean and variance of a Gaussian value function V(s) as V-posterior L -Wasserstein barycenter of all action value function distributions Q(s, a).

Proposition 1. *Consider the V-posterior value function* V(s) *as a Gaussian:* N (m(s), δ 2 (s))*. Let's define each* Q(s, a) *as the Q function child node of* V(s)*. Each* Q(s, a) *is assumed as a Gaussian distributions* Q(s, a) : N (m(s, a), δ(s, a) 2 )*. If the value function* V(s) *is defined as the Wasserstein barycenter of the Q function* Q(s, a) *given the policy* π¯*, we will have:*

$$\overline{m}(s) = (\mathbb{E}_{a \sim \pi}[m(s, a)^p])^{\frac{1}{p}} \quad (14)$$

$$\bar{\delta}(s) = (\mathbb{E}_{a \sim \pi}[\delta(s, a)^p])^{\frac{1}{p}}, \quad (15)$$

*with* p = 1 − α*.*

*Proof.* By the definition of the V-posterior value function, we have:

$$(\bar{\mu}(s), \bar{\delta}(s)) = \arg \min_{\mu, \delta} \left\{ \mathbb{E}_{\bar{\pi}} [W_1(\mathcal{V}(s) || \mathcal{Q}(s, a))] \right\}. \quad (16)$$

We first compute the standard deviation δ(s).

From [\(13\)](#page-14-2), and [\(16\)](#page-15-0), we want to find δ(s) that is the minimizer of

$$\bar{\delta}(s) = \arg \min_{\delta(s)} \left\{ \mathbb{E}_{\bar{\pi}} [D_{f_\alpha}(\delta(s) || \delta(s, a))] \right\}.$$

we derive δ(s) is the solution of

$$\frac{\nabla \mathbb{E}_{a \sim \pi}[D_{f_\alpha}(\delta(s)||\delta(s, a))]}{\nabla \delta(s)} = 0. \quad (17)$$

Since

$$\frac{\nabla f_\alpha(x)}{\nabla x} = \frac{\alpha(x^{\alpha-1} - 1)}{\alpha(\alpha - 1)} = \frac{x^{\alpha-1} - 1}{\alpha - 1}. \quad (18)$$

With Df<sup>α</sup> (x||y) = P y yfα( x y ), we can have

$$\frac{\nabla D_{f_\alpha}(x||y)}{\nabla x} = \sum_y \frac{(\frac{x}{y})^{\alpha-1} - 1}{\alpha - 1}. \quad (19)$$

We can derive

$$\mathbb{E}_{a \sim \bar{\pi}} \left[ \frac{(\frac{\bar{\delta}(s)}{\delta(s,a)})^{\alpha-1} - 1}{(\alpha-1)} \right] = 0 \implies \mathbb{E}_{a \sim \bar{\pi}} \left[ (\frac{\bar{\delta}(s)}{\delta(s,a)})^{\alpha-1} - 1 \right] = 0. \quad (20)$$

Now we can define p = 1 − α that leads to

$$\bar{\delta}(s) = (\mathbb{E}_{a \sim \pi}[\delta(s, a)^p])^{\frac{1}{p}}. \quad (21)$$

To compute µ¯(s). Let's revisit here again the definition of L <sup>1</sup>−Wasserstein distance between two Gaussian distributions µ(m1, δ<sup>2</sup> 1 ), ν(m2, δ<sup>2</sup> ).

$$W_1(\mu, \nu) = \inf\{\mathbb{E}[D_{f_\alpha}(\mu||\nu)]\}. \quad (22)$$

According to Jensen's inequality[\(Perlman, 1974\)](#page-9-20) we can derive

$$\mathbb{E}[D_{f_\alpha}(\mu||\nu)] \geq D_{f_\alpha}(\mathbb{E}[\mu]||\mathbb{E}[\nu]) = D_{f_\alpha}(m_1||m_2). \quad (23)$$

Therefore, according to the definition of Wasserstein barycenter, the mean of a Gaussian V-posterior value function V(s) can be derived as

$$\bar{m}(s) = \arg \min_{m(s)} \mathbb{E}_{a \sim \pi} [D_{f_\alpha}(m(s) || m(s, a))]. \quad (24)$$

Following the same steps as to compute δ(s), we can get

$$\overline{m}(s) = (\mathbb{E}_{a \sim \pi}[m(s, a)^p])^{\frac{1}{p}}, \quad (25)$$

Next, we consider each node as an equally weighted Particle model and derive the following proposition.

Proposition 2. *Let's assume the V-posterior value function* V(s) *as a equally weighted Particle model:* xi(s) : i ∈ [1, M]*.* M *is an integer and* M ⩾ 1*. Let's assume each Q function* Q(s, a) *has* M *particles* xi(s, a), i ∈ [1, M]*. If the value function* V(s) *is defined as the Wasserstein barycenter of the Q function* Q(s, a) *given the policy* π¯*, each particle (*xi(s), i ∈ [1, M]*) can be estimated as*

$$\overline{x}_i(s) = (\mathbb{E}_{a \sim \pi}[x_i(s, a)^p])^{1/p}, \quad (26)$$

*with* p = 1 − α*.*

*Proof.* We can compute the quantile function of µ and ν as

$$F_{\mu}^{-1}(t) = \sum_{i=1}^M x_i \mathbf{1}_{I_i}(t), F_{\nu}^{-1}(t) = \sum_{i=1}^M y_i \mathbf{1}_{I_i}(t). \quad (27)$$

Therefore from [\(11\)](#page-14-3) we can get

$$W_1(\mu, \nu) = \left( \int_0^1 D_{f_\alpha}(F_\mu^{-1}(t)||F_\nu^{-1}(t)) dt \right) \quad (28)$$

$$= \sum_{i=1}^M \left( \int_{I_i} D_{f_\alpha}(F_\mu^{-1}(t)||F_\nu^{-1}(t))dt \right) \quad (29)$$

$$= \sum_{i=1}^M \left( \int_{I_i} D_{f_\alpha}(x_i || y_i) dt \right) \quad (30)$$

$$= \sum_{i=1}^M D_{f_\alpha}(x_i || y_i) \left( \int_{I_i} dt \right) \quad (31)$$

$$= \sum_{i=1}^M w_i D_{f_\alpha}(x_i || y_i). \quad (32)$$

We can see that for each particle (xi(s), i ∈ [1, M]), we can derive

$$\bar{x}_i(s) = \arg \min_{x_i(s)} \mathbb{E}_{a \sim \pi} [D_{f_a}(x_i(s) | x_i(s, a))] \quad (33)$$

$$\implies \overline{x_i}(s) = (\mathbb{E}_{a \sim \pi}[x_i(s, a)^p])^{1/p}, \quad (34)$$

with p = 1 − α.

## D. Supporting Lemmas

We will make use of the following basic results.

Lemma 1. *(Minkowski's inequality) Given* p ⩾ 1, {x<sup>i</sup> , yi} ∈ <sup>R</sup>, i = 1, 2, ..., n*, then we have the following inequality*

$$\left( \sum_i (|x_i + y_i|)^p \right)^{\frac{1}{p}} \leq \left( \sum_i (|x_i|)^p \right)^{\frac{1}{p}} + \left( \sum_i (|y_i|)^p \right)^{\frac{1}{p}}. \quad (35)$$

*Proof.* This is a basic result.

Lemma 2. *(Markov's inequality) If* X *is a nonnegative random variable and* a > 0*, then the probability that X is at least a is at most the expectation of X divided by a:*

$$\Pr(X > a) \leq \frac{\mathbb{E}[X]}{a}. \quad (36)$$

### E. Convergence of Wasserstein Non-stationary multi-armed bandits

We note that in an MCTS tree, each node is considered a non-stationary multi-armed bandit where the average mean drifts due to the given action selection strategy. Therefore, we first study the convergence of Wasserstein non-stationary multiarmed bandits where the action selection is Thompson sampling, with the power mean backup operator at the root node. Detailed descriptions of the Wasserstein Non-stationary multi-armed bandits settings can be found in the main article in the Theoretical Analysis section.

We briefly summarize the theoretical results below. Lemma [6](#page-18-0) is about the upper bound on the expectation of the number of suboptimal arms playing, following the corresponding Theorem 4.2 in [\(Jin et al., 2022\)](#page-9-21). Lemma [7](#page-18-1) is about the bias of the expected value of the power mean backup operator, which follows the result as Theorem 1 in Stochastic-Power-UCT [\(Dam](#page-9-10) [et al., 2024b\)](#page-9-10). Theorem [2](#page-20-0) deals with the polynomial concentration of the power mean backup operator around the optimal mean at the root node of the non-stationary Wasserstein problem for multi-armed bandits. This theorem plays an important role in deriving the polynomial convergence of the choice of the optimal action at the root node in the Wasserstein MCTS tree, described in the next section.

Now, we will find an upper bound for the expectation of numbers of pulling a suboptimal arm. Let us define the event Ek,ε(t) = {θk(t) ⩽ µ <sup>∗</sup> − ε} for all k ∈ [K], ε > 0, θk(t) is sampled from N (Xk, V /Tk(n)) at timestep t. Let us consider the decomposition

$$\mathbb{E}[T_k(n)] = 1 + \mathbb{E}\left[\sum_{t=K+1}^n \mathbf{1}\{A_t = a_k, E_{k,\varepsilon}(t)\} + \sum_{t=K+1}^n \mathbf{1}\{A_t = a_k, E_{k,\varepsilon}^c(t)\}\right] \quad (37)$$

$$= 1 + \mathbb{E} \left[ \underbrace{\sum_{t=K+1}^n \mathbf{1}\{A_t = a_k, E_{k,\varepsilon}(t)\}}_A \right] + \mathbb{E} \left[ \underbrace{\sum_{t=K+1}^n \mathbf{1}\{A_t = a_k, E_{k,\varepsilon}^c(t)\}}_B \right]. \quad (38)$$

Here E<sup>c</sup> is the complement of an event E, ε > 8 p V /n is an arbitrary constant.

Bounding Term A: Let's define

$$\alpha_s = \sup_{x \in [0, \mu^* - \varepsilon]} \left\{ \text{KL}(\mu^* - \varepsilon - x, \mu^*) \leq 4 \log\left(\frac{n}{s}\right) / s \right\}. \quad (39)$$

Lemma 3. *(Lemma A.1 [\(Jin et al., 2022\)](#page-9-21)) Let* M = ⌈16V log(nε<sup>2</sup>/V )/ε<sup>2</sup> ⌉*, and* α<sup>s</sup> *be the same as defined in [\(39\)](#page-17-0) then*

$$\mathbb{E} \left[ \sum_{t=K+1}^n \mathbf{1}\{A_t = a_k, E_{k,\varepsilon}(t)\} \right] \leq \sum_{s=1}^M \mathbb{E} \left[ \left( \frac{1}{1 - F_s^*(\mu^* - \varepsilon)} - 1 \right) \cdot \mathbf{1}\{\bar{X}_s^* \in (\mu^* - \varepsilon - \alpha_s, 1]\} \right] + \odot \left( \frac{V}{\varepsilon^2} \right), \quad (40)$$

*where* F ∗ s *is the CDF of Gaussian with mean* X ∗ s *,* X ∗ s *is the average reward of the optimal arm after* s *visitations.*

Lemma 4. *(Lemma A.2 [\(Jin et al., 2022\)](#page-9-21)) Let* M = ⌈16V log(nε<sup>2</sup>/V )/ε<sup>2</sup> ⌉*. Then*

$$\sum_{s=1}^M \mathbb{E}_{\bar{X}_s^*} \left[ \left( \frac{1}{1 - F_s^*(\mu^* - \varepsilon)} \right) \cdot \mathbf{1}_{\{\bar{X}_s^* \in (\mu^* - \varepsilon - \alpha_s, 1]\}} \right] = \Theta \left( \frac{V \log(n\varepsilon^2/V)}{\varepsilon^2} \right). \quad (41)$$

#### Bounding Term B:

Lemma 5. *(Lemma C.1 [\(Jin et al., 2022\)](#page-9-21)) Let* N = min{ 1 1− *KL*(µk+ρk,µ∗−ε) log(nε2/V ) , 2}*. For any* ρk, ε > 0 *that satisfies* ε+ρ<sup>k</sup> < ∆<sup>i</sup> *, then*

$$\mathbb{E} \left[ \sum_{t=K+1}^n \mathbf{1}\{A_t = k, E_{k,\varepsilon}^c(t)\} \right] \leqslant 1 + \frac{2V}{\rho_k^2} + \frac{V}{\varepsilon^2} + \frac{N \log(n\varepsilon^2/V)}{KL(\mu_k + \rho_k, \mu^* - \varepsilon)}. \quad (42)$$

From Assumption 1, we derive the upper bound for the expectation of the number of plays of a suboptimal arm.

Lemma 6. *Consider Thompson Sampling strategy (using power mean estimator) applied to a non-stationary problem where the pay-off sequence satisfies Assumption 1. Fix* ε ⩾ 0*. Let* Tk(n) *denote the number of plays of arm* k*. Then if* k *is the index of a suboptimal arm, then each sub-optimal arm* k *is played in expectation at most*

$$\mathbb{E}[T_k(n)] \leq \Theta\left(1 + \frac{V \log(n \Delta_k^2 / V)}{\Delta_k^2}\right). \quad (43)$$

*Proof.* The proof of Lemma [6](#page-18-0) closely follows Theorem 4.2([\(Jin et al., 2022\)](#page-9-21)) by observing results from Lemma [3,](#page-17-1) [4,](#page-17-2) [5.](#page-17-3) From equation [38,](#page-17-4) putting all Lemma [3,](#page-17-1) [4,](#page-17-2) [5,](#page-17-3) we have

$$\mathbb{E}[T_k(n)] = \Theta\left(1 + \frac{V \log(n\varepsilon^2/V)}{(\Delta_k - \varepsilon - \rho_k)^2} + \frac{V}{\rho_k^2} + \frac{V \log(n\varepsilon^2/V)}{\varepsilon^2}\right). \quad (44)$$

Set ε = ρ<sup>k</sup> = ∆k/4, we derive

$$\mathbb{E}[T_k(n)] \leq \Theta\left(1 + \frac{V \log(n\Delta_k^2/V)}{\Delta_k^2}\right). \quad (45)$$

Lemma 7. *Consider a non-stationary problem where the pay-off sequence satisfies Assumption 1. We consider a bandit algorithm that selects each arm as*

$$a = \operatorname{argmax}_{a_i, i \in \{1 \dots K\}} \{\theta_i \sim \mathcal{N}(\overline{X}_{k,n}, V/T_k(n))\}.$$

*Let us define the power mean estimator* <sup>X</sup>n(p) *as* <sup>X</sup>n(p) = P<sup>K</sup> a=1 Ta(n) <sup>n</sup> X p a,Ta(n) 1 p *, and* δ⋆,n = µ<sup>⋆</sup> − µ⋆,n *For any* p ⩾ 1, ε<sup>0</sup> > 0*, we have*

$$|\mathbb{E}[\overline{X}_n(p)] - \mu_*| \leq |\delta_{*,n}| + \frac{R}{n} \sum_{a=1, a \neq a_*}^K \Theta\left(1 + \frac{V \log(n \Delta_k^2 / V)}{\Delta_k^2}\right) \quad (46)$$

*Proof.* We observe that

$$|\bar{X}_n(p) - \mu_*| \leq |\bar{X}_n(p) - \mu_{*,n}| + |\mu_* - \mu_{*,n}| = |\bar{X}_n(p) - \mu_{*,n}| + |\delta_{*,n}| \quad (47)$$

Furthermore,

$$\overline{X}_{a,T_a(n)} \leqslant \mu_{a,n} + |\overline{X}_{a,T_a(n)} - \mu_{a,n}|. \quad (48)$$

Since µ⋆,n = maxa∈[K]{µa,n}, we have

$$\bar{X}_n(p) - \mu_{\star,n} = \bar{X}_n(p) - \sum_{a=1}^K T_a(n) \mu_{\star,n} \leqslant \left( \sum_{a=1}^K \frac{T_a(n)}{n} (\bar{X}_a, T_a(n))^p \right)^{\frac{1}{p}} - \left( \sum_{a=1}^K \frac{T_a(n)}{n} (\mu_{a,n})^p \right)^{\frac{1}{p}} \quad (49)$$

$$= \frac{\left( \sum_{a=1}^K T_a(n) (\bar{X}_{a,T_a(n)})^p \right)^{\frac{1}{p}} - \left( \sum_{a=1}^K T_a(n) (\mu_{a,n})^p \right)^{\frac{1}{p}}}{n^{\frac{1}{p}}} \quad (50)$$

Applying Minkowski's inequality from Lemma [1,](#page-16-0) and the result of equation [48,](#page-18-2) we have

$$\overline{X}_n(p) - \mu_{\star,n} \leq \frac{\left(\sum_{a=1}^K T_a(n) (\mu_a + |\overline{X}_{a,T_a(n)} - \mu_{a,n}|)^p\right)^{\frac{1}{p}} - \left(\sum_{a=1}^K T_a(n) (\mu_{a,n})^p\right)^{\frac{1}{p}}}{n^{\frac{1}{p}}} \quad (51)$$

$$\leq \frac{\left(\sum_{a=1}^K T_a(n) (|\overline{X}_{a,T_a(n)} - \mu_{a,n}|)^p\right)^{\frac{1}{p}}}{n^{\frac{1}{p}}} \quad (52)$$

On the other hand,

$$\mu_{\star,n} - \overline{X}_n(p) = \frac{n\mu_{\star,n} - n\overline{X}_n(p)}{n} = \frac{n\mu_{\star,n} - (\sum_{a=1}^K T_a(n)\mu_{a,n}) + \sum_{a=1}^K T_a(n)\mu_{a,n} - n\overline{X}_n(p)}{n} \quad (53)$$

$$= \frac{\sum_{a=1, a \neq a_*}^K T_a(n) |\mu_{\star, n} - \mu_{a, n}| + \sum_{a=1}^K T_a(n) \mu_{a, n} - n \bar{X}_n(p)}{n} \quad (54)$$

$$\leq R \sum_{a=1, a \neq a_*}^K \frac{T_a(n)}{n} + \sum_{a=1}^K \frac{T_a(n)}{n} \mu_{a,n} - \overline{X}_n(p) \quad (55)$$

Because power mean is an increasing function of p, so that

$$\sum_{a=1}^K \frac{T_a(n)}{n} \mu_{a,n} \leq \left( \sum_{a=1}^K \frac{T_a(n)}{n} (\mu_{a,n})^p \right)^{1/p}.$$

Furthermore, we observe that

$$\mu_{a,n} \leq \overline{X}_{a,T_a(n)} + \left| \overline{X}_{a,T_a(n)} - \mu_{a,n} \right|.$$

So that, from equation [55](#page-19-0) we have

$$\mu_{\star,n} - \overline{X}_n(p) \leqslant R \sum_{a=1, a \neq a_*}^K \frac{T_a(n)}{n} + \left( \sum_{a=1}^K \frac{T_a(n)}{n} (\mu_{a,n})^p \right)^{1/p} - \overline{X}_n(p) \quad (56)$$

$$\leq R \sum_{a=1, a \neq a_*}^K \frac{T_a(n)}{n} \quad (57)$$

$$+ \frac{\left( \sum_{a=1}^K T_a(n) (\bar{X}_{a,T_a(n)} + |\bar{X}_{a,T_a(n)} - \mu_{a,n}|)^p \right)^{\frac{1}{p}} - \left( \sum_{a=1}^K T_a(n) (\bar{X}_{a,T_a(n)})^p \right)^{\frac{1}{p}}}{n^{\frac{1}{p}}} \quad (58)$$

$$(\text{Minkovski's inequality}) \quad R \sum_{a=1, a \neq a_*}^K \frac{T_a(n)}{n} + \frac{\left( \sum_{a=1}^K T_a(n) (|\overline{X}_{a, T_a(n)} - \mu_{a, n}|)^p \right)^{\frac{1}{p}}}{n^{\frac{1}{p}}} \quad (59)$$

$$(\text{Properties of } L^p \text{ norm}) \leq R \sum_{a=1, a \neq a_*}^K \frac{T_a(n)}{n} + \frac{\left( \sum_{a=1}^K T_a(n) (|\overline{X}_{a, T_a(n)} - \mu_{a, n}|) \right)}{n^{\frac{1}{p}}} \quad (60)$$

$$= R \sum_{a=1, a \neq a_*}^K \frac{T_a(n)}{n} + \frac{\sum_{a=1}^K \left( \left| \sum_t^{T_a(n)} X_{a,t} - T_a(n) \mu_{a,n} \right| \right)}{n^{\frac{1}{p}}} \quad (61)$$

Therefore

$$|\mathbb{E}[\bar{X}_n(p) - \mu_{\star,n}]| \leq R \sum_{a=1, a \neq a_*}^K \frac{\mathbb{E}[T_a(n)]}{n} + \frac{\mathbb{E}\left[\left(\left|\sum_{a=1}^K \sum_t^{T_a(n)} X_{a,t} - T_a(n)\mu_{a,n}\right|\right)\right]}{n^{\frac{1}{p}}} \quad (62)$$

$$= R \sum_{a=1, a \neq a_*}^K \frac{\mathbb{E}[T_a(n)]}{n} \quad (63)$$

Please note that because we study non-stationary bandits, <sup>E</sup>[ P<sup>n</sup> <sup>t</sup> Xa,t] = nµa,n, therefore,

$$\mathbb{E} \left[ \left( \left| \sum_{a=1}^K \sum_t^{T_a(n)} X_{a,t} - T_a(n) \mu_{a,n} \right| \right) \right] = 0$$

According to Lemma [7,](#page-18-1) we have

$$|\mathbb{E}[\overline{X}_n(p) - \mu_{\star,n}]| \leq |\delta_{\star,n}| + R \sum_{a=1, a \neq a_*}^{\infty} \frac{\mathbb{E}[T_a(n)]}{n} \leq |\delta_{\star,n}| + \frac{R}{n} \sum_{a=1, a \neq a_*}^{\infty} \Theta\left(1 + \frac{V \log(n \Delta_k^2/V)}{\Delta_k^2}\right), \quad (64)$$

which concludes the proof.

Theorem 2. *For* a ∈ [K]*, let* (Xa,n)<sup>n</sup>⩾<sup>1</sup> *be a sequence of estimator satisfying Assumption 1 and let* µ<sup>⋆</sup> = max a {µa}*. Assume that all the estimators are bounded in* [0, R]*. We consider a bandit algorithm that selects each arm as*

$$a = \operatorname{argmax}_{a_i, i \in \{1 \dots K\}} \{\theta_i \sim \mathcal{N}(\bar{X}_{k,n}, V/T_k(n))\}.$$

*Then, for all* p ∈ [1, ∞)*, the sequence of estimators*

$$\overline{X}_n(p) = \left( \sum_{a=1}^K \frac{T_a(n)}{n} \overline{X}_{a,T_a(n)}^p \right)^{\frac{1}{p}},$$

*where* <sup>T</sup>a(n) = P<sup>n</sup>−<sup>1</sup> <sup>t</sup>=1 <sup>1</sup>(a<sup>t</sup> = a) *is the number of selections of* a *prior to round* n *satisfies*

$$\mathbf{Pr}(|\overline{X}_n(p) - \mu_\star| \geq \varepsilon) \leq Cn^{-1}\varepsilon^{-2}.$$

*Proof.* We first prove that limn→∞ <sup>E</sup>[Xn(p)] = µ∗. According to the result of Lemma [7,](#page-18-1) we have

$$|\mathbb{E}[\overline{X}_n(p)] - \mu_*| \leq |\delta_{*,n}| + R \sum_{a=1, a \neq a_*}^K \frac{\mathbb{E}[T_a(n)]}{n} \quad (65)$$

$$\leq |\delta_{\star, n}| + \frac{R}{n} \sum_{a=1, a \neq a_*}^K \left\{ \frac{(1 + \varepsilon_0) \log n}{\mathcal{K}^{(N)}(F_a, \mu_\star)} + o(\log n) + O(1) \right\} \quad (66)$$

with <sup>δ</sup>⋆,n <sup>=</sup> <sup>µ</sup><sup>⋆</sup> <sup>−</sup> <sup>µ</sup>⋆,n, and because limn→∞ µ∗,n = µ⋆, we can concludes that

$$\lim_{n \rightarrow \infty} \mathbb{E}[\overline{X}_n(p)] = \mu_*.$$

Second, we prove that

$$\forall n \geqslant 1, \forall \varepsilon > 0, \exists c > 0 \text{ that } \mathbb{P}\left(|\overline{X}_n(p) - \mu_*| > \varepsilon\right) \leqslant cn^{-1}\varepsilon^{-2}.$$

We observe that

$$|\overline{X}_n(p) - \mu_*| \leq |\overline{X}_n(p) - \mu_{*,n}| + |\mu_* - \mu_{*,n}| = |\overline{X}_n(p) - \mu_{*,n}| + |\delta_{*,n}| \quad (67)$$

$$\Rightarrow \mathbb{P}(|\bar{X}_n(p) - \mu_*| \geq \varepsilon) \leq \mathbb{P}(|\bar{X}_n(p) - \mu_{*,n}| \geq \varepsilon/2) + \mathbb{P}(|\delta_{*,n}| \geq \varepsilon/2). \quad (68)$$

Because limn→<sup>n</sup> |δ⋆,n| = 0, therefore, ∃N<sup>0</sup> > 0 such that ∀n ⩾ N0, we have |δ⋆,n| < ε/2 that means

$$\forall n > N_0, \mathbb{P}(|\delta_{\star,n}| \geq \varepsilon/2) = 0.$$

Next, according to Lemma [6,](#page-18-0)

$$|\mathbb{E}[\overline{X}_n(p)] - \mu_{\star,n}| \leq \frac{R}{n} \sum_{a=1, a \neq a_*}^K \left\{ \frac{(1 + \varepsilon_0) \log n}{\mathcal{K}^{(N)}(F_a, \mu_\star)} + o(\log n) + O(1) \right\} = O(n^{-1}), \quad (69)$$

that leads to

$$\mathbb{P}(|\overline{X}_n(p) - \mu_{\star,n}| \geq \varepsilon/2) \leq \frac{|\mathbb{E}[\overline{X}_n(p)] - \mu_{\star,n}|}{\varepsilon/2} = \frac{O(n^{-1})}{\varepsilon/2}. \quad (70)$$

Therefore, ∃c > 0, ∀0 < ε < 1 such that

$$\mathbb{P}(|\overline{X}_n(p) - \mu_{*,n}| \geq \varepsilon/2) \leq cn^{-1}\varepsilon^{-2}, \quad (71)$$

which means

$$\forall n \geqslant N_0, \forall 0 < \varepsilon < 1, \exists c > 0 \text{ that } \mathbb{P}\left(|\overline{X}_n(p) - \mu_*| > \varepsilon\right) \leqslant cn^{-1}\varepsilon^{-2}.$$

Now we see that |Xn(p) − µ⋆| ⩽ R. With ε > max{1, R}, we have |Xn(p) − µ⋆| > ε ⇔ |Xn(p) − µ⋆| > R, therefore the inequality holds as

$$\mathbb{P} \left( |\overline{X}_n(p) - \mu_\star| > \varepsilon \right) = 0 \leq cn^{-1}\varepsilon^{-2}.$$

with 0 < ε < max{1, R}, 1 ⩽ n < N<sup>0</sup> ⇒ nε < max{1, R}N<sup>0</sup> ⇒ n −1 ε <sup>−</sup><sup>1</sup> > 1/ max{1, R}N0. Therefore

$$\forall C > 1/\max\{1, R\}N_0 \Rightarrow \mathbb{P}(|\overline{X}_n(p) - \mu_*| > \varepsilon) \leqslant 1 < Cn^{-1}\varepsilon^{-1} < Cn^{-1}\varepsilon^{-2},$$

which means

$$\forall n \geqslant 1, \forall 0 < \varepsilon < 1, \exists C > 0 \text{ that } \mathbb{P} \left( |\overline{X}_n(p) - \mu_*| > \varepsilon \right) \leqslant C n^{-1} \varepsilon^{-2}.$$

That concludes the proof.

## F. Convergence of Wasserstein Monte-Carlo tree search

We start with Lemma [8,](#page-21-0) which shows the concentration of empirical Q value at any internal node in the tree. This plays an important role in the analysis of our MCTS algorithm.

From the results of Lemma [8](#page-21-0) and Theorem [2,](#page-20-0) we derive Propostion [3](#page-5-2) which shows the concentration of any internal V-node and Q-node in the tree. Finally, we get the expected simple bias with convergence rate of O(n −1/2 ) in Theorem [1.](#page-5-1)

Let us start with Lemma [8.](#page-21-0)

Lemma 8. *(Lemma 1[\(Dam et al., 2024b\)](#page-9-10)) For* <sup>m</sup> ∈ [M]*, let* (Vb(m, n))<sup>n</sup>⩾<sup>1</sup> *be a sequence of estimator satisfying*

$$\Pr \left( |\hat{V}(m, n) - V(m)| > \varepsilon \right) \leq C n^{-1} \varepsilon^{-2}$$

*Assume that there exists a constant* L > <sup>0</sup> *such that* <sup>L</sup> <sup>=</sup> *supremum*{Vb(m, n)}<sup>n</sup>⩾1*. Let* <sup>R</sup><sup>i</sup> *be an iid sequence with mean* µ *and* S<sup>i</sup> *be an iid sequence from a distribution* p = (p1, . . . , pM) *supported on* {1, . . . , M}*. Introducing the random variables* N<sup>n</sup> <sup>m</sup> = #|{i ⩽ n : S<sup>i</sup> = sm}|*, we define the sequence of estimator*

$$\bar{Q}(n) = \frac{1}{n} \sum_{i=1}^n R_i + \gamma \sum_{m=1}^M \frac{N_m^m}{n} \hat{V}(m, N_m^m).$$

*Then there exists some constant* c ′ *(which depends on* p<sup>i</sup> *(i=1,2,...,M),* γ*,* µ*) such that*

$$\Pr \left( \left| \overline{Q}(n) - \mu - \sum_{m=1}^M p_m V(m) \right| \geq \varepsilon \right) \leqslant C n^{-1} \varepsilon^{-2}.$$

Based on the results of the described nonstationary multi-armed bandit problem, we derive theoretical results for W-MCTS .

We derive Proposition [3,](#page-5-2) which shows the polynomial concentration of the estimated mean of the Q-value function at the root node. In Proposition [3,](#page-5-2) we also show that the estimated mean of the V-value function at the root node converges polynomially to the optimal mean. Based on Proposition [3,](#page-5-2) we derive the result in Theorem [1,](#page-5-1) which shows the bias of the expected payoff of the power mean backup at the root node.

At any node of state s at depth h in the tree, the mean of the Q value function, and the mean value of the optimal value function are defined as

$$\tilde{Q}(s_h, a) = R(s_h, a) + \gamma \tilde{V}(s_{h+1}), \quad (72)$$

$$\tilde{V}(s_h) = \operatorname{argmax}_a \tilde{Q}(s_h, a), \quad (73)$$

with <sup>h</sup> = [<sup>H</sup> − <sup>1</sup>, ..., <sup>1</sup>, 0], <sup>V</sup>e(sh) is the value return from rollouts at state <sup>s</sup>h, <sup>R</sup>(sh, a) is the mean reward received at state s<sup>h</sup> after taking action a. Let us denote ak<sup>∗</sup> as the optimal action at the root node.

Proposition 3. *When we apply the* W-MCTS *algorithm to an MCTS tree of depth* (H)*, at any depth* h *of the tree, we have*

*(i) At any depth* h*,* ∃ *constant* C<sup>0</sup> > 0 *that for any* 0 < ε < 1, n ⩾ 1*, we can derive*

$$\Pr\left(\left|\overline{V}_m(s_h, a_k, n) - \tilde{V}(s_h, a_k)\right| \geq \varepsilon\right) \leq C_0 n^{-1} \varepsilon^{-2}. \quad (74)$$

*(ii) At any depth* h*,* ∃ *constant* C<sup>0</sup> > 0 *that for any* 0 < ε < 1, n ⩾ 1*, we can derive*

$$\Pr\left(\left|\overline{Q}_m(s_h, a_k, n) - \tilde{Q}(s_h, a_k)\right| \geq \varepsilon\right) \leq C_0 n^{-1} \varepsilon^{-2}. \quad (75)$$

*Proof.* We will prove this by induction on the depth D of the tree.

#### Base case (depth H = 1):

At depth 1, the tree consists of only the root node. The state at the root is denoted by s0. At time step t, suppose the agent takes action a<sup>k</sup> at s0, resulting in an intermediate reward rt(s0, ak), and transitions to the next state s1.

We assume that the reward R(s0, ak) represents the mean reward received at state s<sup>0</sup> after taking action ak.

We recall the definition of <sup>Q</sup>e(s0, ak), defined as

$$\tilde{Q}(s, a) = R(s, a) + \gamma \tilde{V}(s). \quad (76)$$

where Vm(s1) is the value of the rollout policy at state s1, A<sup>s</sup><sup>0</sup> is the set of feasible actions at state s0, |A<sup>s</sup><sup>0</sup> | = M, <sup>P</sup>(s1|s0, ak) is the probability transition of taking action a<sup>k</sup> at state s<sup>0</sup> to state s1. We have

$$\bar{Q}_m(s_0, a_k, n) = \frac{1}{n} \sum_{t=1}^n r_t(s_0, a_k) + \gamma \sum_{s_1 \sim \mathcal{T}(s_0, a_k)} \frac{T_{s_0, a_k}^{s_1}(n)}{n} \bar{V}_m(s_1, T_{s_0, a_k}^{s_1}(n))$$

Equation [\(74\)](#page-22-0) is a direct result of Lemma [8,](#page-21-0) where X<sup>t</sup> represents the intermediate reward rt(s0, ak) at time step t. The probability distribution p = (p1, p2, . . . , pM) ∼ <sup>P</sup>(·|s0, ak), where <sup>P</sup>(·|s0, ak) is the transition probability dynamic for taking action a<sup>k</sup> in state s0.

For each <sup>m</sup> ∈ [M], the sequence (<sup>V</sup> m,t)<sup>t</sup>⩾<sup>1</sup> at time step <sup>t</sup> corresponds to the deterministic initial value function <sup>V</sup>em(s1), where:

$$\mathbf{Pr} \left( \left| \overline{V}_m(s_m, n) - \tilde{V}(s_1) \right| > \varepsilon \right) \leqslant C n^{-1} \varepsilon^{-2},$$

with m = 1, 2, 3, . . . , M, and s<sup>m</sup> ∼ τ (·|s0, ak). Here, τ (·|s0, ak) denotes the transition kernel from state s<sup>0</sup> to sm, given action ak.

Equation [\(75\)](#page-22-1) is the direct results from Theorem [2.](#page-20-0) In detail, we have from equation [\(74\)](#page-22-0),

$$\Pr \left( \left| \bar{Q}_m(s_0, a_k, n) - \tilde{Q}(s_0, a_k) \right| > \varepsilon \right) \leq C n^{-1} \varepsilon^{-2}, \text{ with } a_k \in \mathcal{A}_{s_0}$$

Because by definition:

$$\tilde{V}(s_0) = \max_{a_k \in \mathcal{A}_{s_0}} \tilde{Q}(s_0, a_k) \quad (77)$$

$$\bar{V}_m(s_0, n) = \left( \sum_{a \in \mathcal{A}_{s_0}} \frac{T_{s_0, a}(n)}{n} (\bar{Q}_m(s_0, a, T_{s_0, a}(n)))^p \right)^{\frac{1}{p}} \text{ with } p \in [1, +\infty) \quad (78)$$

Then we have

$$\Pr \left( \left| \overline{V}_m(s_0, n) - \tilde{V}(s_0) \right| > \varepsilon \right) \leq C n^{-1} \varepsilon^{-2}$$

that concludes for Equation [\(75\)](#page-22-1)

Let us assume that for a tree of depth H − 1, the theorem holds for all its children.

Now, consider a tree with depth H. When an action is taken at the root node, where the state is s0, the tree transitions into a subtree of depth H. By the induction hypothesis, the results hold for any internal node of the tree after taking the first action.

We have s<sup>1</sup> ∼ τ (s0, ak), where τ (s0, ak) denotes the transition dynamics. By definition, the value function at the leaf nodes is <sup>V</sup>e(sH) = <sup>V</sup>0(sH), and for all <sup>h</sup> <sup>⩽</sup> <sup>H</sup> − <sup>1</sup>, the following holds:

$$\begin{aligned} \tilde{Q}(s_h, a) &= R(s_h, a) + \gamma \sum_{s_{h+1} \in \mathcal{A}_s} \mathbb{P}(s_{h+1} \mid s_h, a) \tilde{V}(s_{h+1}), \\ \tilde{V}(s_h) &= \max_a \tilde{Q}(s_h, a), \end{aligned}$$

where R(sh, a) represents the immediate reward at state s<sup>h</sup> after taking action a, γ is the discount factor, and <sup>P</sup>(sh+1 | sh, a) is the probability of transitioning to state sh+1 from s<sup>h</sup> by taking action a.

By the assumption of the induction the root node of a subtree with depth (H − 1) at state s<sup>1</sup> we have

$$\mathbf{Pr} \left( \left| \overline{V}_m(s_1, n) - \tilde{V}(s_1) \right| > \varepsilon \right) \leqslant C n^{-1} \varepsilon^{-2}$$

[\(75\)](#page-22-1) Let's apply Lemma [8](#page-21-0) with {Xt} is the intermediate reward {rt(s0, ak)}, p = (p1, p2, ...pM) ∼ <sup>P</sup>(·|s0, ak). For m ∈ [M], each (V m,t)<sup>t</sup>⩾<sup>1</sup> at time step t is the empirical Value function V <sup>t</sup>(s1). We will have

$$\Pr \left( \left| \overline{Q}_m(s_0, a_k, n) - \tilde{Q}(s_0, a_k) \right| > \varepsilon \right) \leqslant C n^{-1} \varepsilon^{-2}, \text{ with } a_k \in \mathcal{A}_{s_0}$$

[\(74\)](#page-22-0) follows the results of Theorem [2](#page-20-0) as at the root node s<sup>0</sup> of depth H, with

$$\tilde{V}(s_0) = \max_{a_k \in \mathcal{A}_{s_0}} \tilde{Q}^{(0)}(s_0, a_k) \quad (79)$$

$$\bar{V}_m(s_0, n) = \left( \sum_{a \in \mathcal{A}_s} \frac{T_{s_0, a}(n)}{n} (\bar{Q}_m(s_0, a, T_{s_0, a}(n)))^p \right)^{\frac{1}{p}} \text{ for some } p \in [1, +\infty) \quad (80)$$

And because

$$\Pr \left( \left| \overline{Q}_m(s_0, a_k, n) - \widetilde{Q}(s_0, a_k) \right| > \varepsilon \right) \leqslant Cn^{-1}\varepsilon^{-2}, \text{ with } a_k \in \mathcal{A}_{s_0}$$

Then, we have

$$\Pr \left( |\overline{V}_m(s_0, n) - V(s_0)| > \varepsilon \right) \leq C n^{-1} \varepsilon^{-2}.$$

that concludes for [\(74\)](#page-22-0)

Theorem 1. *We have at the root node* s0*,*

$$|\mathbb{E}[\overline{V}_m(s_0, n)] - V(s_0)| \leq \mathcal{O}(n^{-1/2}).$$

*Proof.* Using the convexity of f(x) = |x| and applying Jensen's inequality we have

$$\begin{aligned} |\mathbb{E}[\overline{V}_m(s_0, n)] - V(s_0)| &\leq \mathbb{E}[|\overline{V}_m(s_0, n)| - V(s_0)] \\ &= \int_0^{+\infty} \mathbb{P}(|\overline{V}_m(s_0, n) - V(s_0)| \geq s) ds \\ &\leq \int_0^{n^{-\frac{1}{2}}} 1 ds + \int_{n^{-\frac{1}{2}}}^{+\infty} C_0 n^{-1} s^{-2} ds \\ &\leq n^{-\frac{1}{2}} + C_0 n^{-1} \left( \frac{s^{-2+1}}{-2+1} \right) \Big|_{n^{-\frac{1}{2}}}^{+\infty} \\ &= \left( \frac{C_0}{2-1} + 1 \right) n^{-\frac{1}{2}}. \end{aligned}$$

Then,

$$|\mathbb{E}[\overline{V}_m(s_0, n)] - V(s_0)| \leq \mathcal{O}(n^{-1/2}).$$

That concludes the proof.