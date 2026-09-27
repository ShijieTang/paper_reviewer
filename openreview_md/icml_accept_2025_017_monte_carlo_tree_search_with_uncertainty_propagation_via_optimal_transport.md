# 

Tuan Dam 1 Pascal Stenger 2 Lukas Schneider 3 Joni Pajarinen 4 **Carlo D'Eramo** 2 5 6 Odalric-Ambrym Maillard 7

## Abstract

This paper introduces a novel backup strategy for Monte-Carlo Tree Search (MCTS) tailored for highly stochastic and partially observable Markov decision processes. We adopt a probabilistic approach, modeling both value and action-value nodes as Gaussian distributions, to introduce a novel backup operator that computes value nodes as the Wasserstein barycenter of their action-value children nodes; thus, propagating the uncertainty of the estimate across the tree to the root node. We study our novel backup operator when using a novel combination of L
1-Wasserstein barycenter with α-divergence, by drawing a crucial connection to the generalized mean backup operator. We complement our probabilistic backup operator with two sampling strategies, based on optimistic selection and Thompson sampling, obtaining our Wasserstein MCTS algorithm. We provide theoretical guarantees of asymptotic convergence of O(n
−1/2),
with n as the number of visited trajectories, to the optimal policy and an empirical evaluation on several stochastic and partially observable environments, where our approach outperforms wellknown related baselines.

## 1. Introduction

Reinforcement Learning (RL) problems, particularly after its widespread success in deterministic games like Go and Chess (Silver et al., 2016a; 2017b). However, moving beyond these deterministic settings toward highly stochastic or partially observable Markov Decision Processes (MDP- s/POMDPs) reveals major difficulties. In these cases, two key obstacles arise: *Uncertainty in Value Estimates:* In problems with substantial randomness or limited observability, naive value backups may lead to erroneous or unstable estimates, which propagate through the tree and degrade overall performance. Exploration-Exploitation Balancing: Traditional UCT-based exploration bonuses (Kocsis et al., 2006) can falter under high variance transitions, often causing either over- or under-exploration. Recent works (Tesauro et al., 2012; Bai et al., 2013; 2014) have suggested Bayesian or distributional methods for MCTS to better quantify uncertainty. Meanwhile, Metelli et al. (2019) leveraged L
2-Wasserstein barycenters to propagate distributional information in temporal-difference learning. Yet, several open questions remain on how to unify distribution-based backups and flexible exploration strategies within a single MCTS framework that provably handles high stochasticity and partial observability. Our Approach. In this paper, we propose a new MCTS algorithm, *Wasserstein MCTS*, that models each node's value as a Gaussian distribution and propagates *both* mean and variance estimates throughout the tree. Crucially, we introduce a novel backup operator that computes *value nodes* as L
1-Wasserstein barycenters of their *action-value children*,
using an α-divergence as the distance measure. This yields:
- *Distributional Value Backups:* By tracking distributions
(rather than point estimates), our method captures the inherent uncertainty of each node's value, especially valuable in stochastic or partially observable domains.

- *Generalized Mean Operator:* The α-divergence ties naturally to the power-mean backup (Dam et al., 2019; 2024a), letting us interpolate between average-like and max-like updates to mitigate the overestimation often seen in RL (Hasselt, 2010).

We complement these distributional backups with two exploration mechanisms—an optimistic UCT bonus, and a Thompson sampling approach that selects actions by sam1 pling from the node's Gaussian posterior. Our Key Contributions. *1. Uncertainty Propagation* via L
1*-Wasserstein Barycenters.* We provide a principled way to back up distributions in an MCTS, unifying L
1-
Wasserstein geometry and α-divergences to handle high variance and partial observability. 2. Connection to Generalized Mean Backup. Our backup operator yields a powermean update for node values, enabling a controllable continuum between overly optimistic (max-like) and riskaverse (average-like) estimates. 3. Polynomial Convergence Analysis. We prove that *Wasserstein MCTS* with Thompson sampling converges to the optimal policy at a rate O(n
−1/2), matching known lower bounds. This is in contrast to prior distributional MCTS methods that lacked explicit convergence guarantees. 4. Extensive Empirical Validation. On a suite of highly stochastic MDPs (e.g. River- Swim, Taxi) and partially observable tasks (Pocman, Rocksample), our approach outperforms established baselines, including UCT, Power-UCT, and Bayesian MCTS variants.

Overall, *Wasserstein MCTS* offers a flexible and theoretically grounded framework for handling uncertainty within MCTS. By combining Gaussian node models, L
1-
Wasserstein barycenters, and α-divergences, it effectively balances exploration and exploitation in domains where noise or partial observability make traditional MCTS methods brittle.

## 2. Related Work

Metelli et al. (2019) use L
2-Wasserstein barycenters to propagate uncertainty in temporal-difference learning. In MCTS, Bayesian methods handle uncertainty by treating values as Gaussian distributions (Tesauro et al., 2012) or Dirichlet-NormalGamma posteriors (Bai et al., 2013; 2014). Unlike these, we propagate uncertainty *throughout* the tree via L
1-Wasserstein barycenters and α-divergences, linking to generalized-mean backups (Dam et al., 2019) and maintaining both mean and variance estimates. This distributional perspective is effective in highly stochastic or partially observable tasks. In multi-armed bandits, optimism (Auer et al., 2002a) and Thompson sampling (Thompson, 1933) are standard; we combine these with our uncertainty propagation scheme to guide action selection in MCTS.

## 3. Background 3.1. Markov Decision Process

We consider an agent in an infinite-horizon discounted Markov decision process (MDP) M = ⟨S, A, R,P, γ⟩, where S is the state space, A is the finite action space, R : *S × A × S →* R is the reward function, P : *S × A → S* is the transition kernel, and γ ∈ [0, 1) is the discount factor. A policy π ∈ Π : *S → A* defines the action selection probabilities based on states. The action-value function Qπis given by Qπ(*s, a*) ≜ EP∞
k=0 γ kri+k+1 | si = s, ai = *a, π*,
representing the expected cumulative discounted reward for executing action a in state s and following policy π. The objective is to find the optimal policy that maximizes Qπ, satisfying the Bellman equation (Bellman, 1954): Q∗(*s, a*) ≜
RS
P(s
′|*s, a*) [R(*s, a, s*′) + γ maxa′ Q∗(s
′, a′)] ds′, ∀s ∈
S, a ∈ A. From the optimal action-value function, we derive the optimal value function as V
∗(s) ≜ maxa∈A Q∗(s, a), ∀s ∈ S.

## 3.2. Monte-Carlo Tree Search

Monte-Carlo Tree Search (MCTS) combines Monte-Carlo sampling, tree search, and exploration strategies from multi-armed bandits (Auer et al., 2002b) to solve MDPs.

It builds a search tree where states are nodes and actions are edges. MCTS involves four key steps: Selection: Navigate from the root to a leaf node using a *tree-policy*. Expansion: Expand the reached node based on the tree policy. Simulation: Perform a rollout (Monte-Carlo simulation) from the child node to estimate its value, or use a pretrained neural network (Silver et al., 2016a) for this estimation. Backup: Update the action-values Q(·) along the visited trajectory using the collected rewards.

## 4. Formalization

Problem Setup Monte Carlo Tree Search (MCTS) is an algorithm for exploring and evaluating trajectories in an MDP. Starting from an initial state s0, MCTS incrementally builds a planning tree by simulating trajectories. Each trajectory either reaches a leaf node or terminates when a predetermined maximum depth H is reached. At the end of each trajectory, a *playout policy* (which may be deterministic or stochastic) is executed from the final node reached, allowing the algorithm to evaluate the associated state. After running for t trajectories, the MCTS algorithm provides the following outputs:
- at: estimate of the optimal action to take in state s0, - V t(s0): estimate of the optimal value function at s0.

Evaluating MCTS Performance The performance of the MCTS algorithm is assessed based on its convergence rate, r(t), which quantifies how quickly the algorithm approaches the optimal policy. Specifically, the following bounds hold:

$$\begin{array}{c}{{\mathbb{E}\left[V^{\star}(s_{0})-Q^{\star}(s_{0},\overline{{{a}}}_{t})\right]\leqslant r(t),}}\\ {{\left|\mathbb{E}\left[V^{\star}(s_{0})-\overline{{{V}}}_{t}(s_{0})\right]\right|\leqslant r(t),}}\end{array}$$

where V
⋆(s0) and Q⋆(s0, a) are the true optimal value and action-value functions at state s0, respectively.

Recursive Value Estimation To analyze the MCTS algorithm, we consider a planning horizon H and a playout policy π0 with an associated value function V0. For each node sh at depth h (i.e., the state reached after h steps from s0),
we recursively define the value function Ve(sh) as follows.

At the leaf nodes (h = H), the value function is simply the playout policy's value:

$$\widetilde{V}(s_{H})=V_{0}(s_{H}).$$

For all other depths h ⩽ H − 1, we compute the actionvalue function Qe(sh, a) and value function Ve(sh) as:

$$\begin{array}{l}{{\widetilde{Q}(s_{h},a)=r(s_{h},a)+\gamma\sum_{s_{h+1}\in{\mathcal A}_{s}}{\mathcal P}(s_{h+1}\mid s_{h},a)\widetilde{V}(s_{h+1}),}}\\ {{\widetilde{V}(s_{h})=\operatorname*{max}_{a}\widetilde{Q}(s_{h},a),}}\end{array}$$

where r(sh, a) is the mean immediate reward obtained by taking action a in state sh, P(sh+1 | sh, a) is the probability of transitioning to state sh+1 from sh given action a and γ is the discount factor.

Bounding the Error The recursive structure of the value estimates gives rise to a bound on the error between the true optimal action-value function Q⋆(s0, a) and the MCTS estimate Qe(s0, a). Specifically, we have:

$$\left|Q^{\star}(s_{0},a)-\tilde{Q}(s_{0},a)\right|\leqslant\gamma^{H}\|V^{\star}-V_{0}\|_{\infty},$$

where the supremum norm ∥V
⋆ − V0∥∞ can be restricted to states reachable within H steps from s0.

Goal of MCTS The ultimate aim of the MCTS algorithm is to minimize the convergence rate r(t) by constructing accurate estimates of Qe(s0, a) and Ve(s0), which in turn approach the true optimal functions Q⋆(s0, a) and V
⋆(s0),
and then identify the best action at the root node:

$$a_{\star}=\arg\operatorname*{max}_{a}Q^{\star}(s_{0},a).$$

## 5. Wasserstein Barycenter With Α**-Divergence**

We introduce the key notions behind our distribution-based backups: the *Wasserstein barycenter* and the α*-divergence*. Unlike prior works that use L
2-based Wasserstein distances (Metelli et al., 2019), we adopt an L
1-Wasserstein distance combined with α-divergences. This combination yields more robust value backups in stochastic and partially observable settings.

## 5.1. Wasserstein Barycenter

Let (X , d) be a Polish (complete, separable metric) space. For q ≥ 1, define Pq(X ) as the set of probability measures on X whose q-th moment is finite. For two distributions µ, ν ∈ Pq(X ), the L
q*-Wasserstein distance* is

$$W_{q}(\mu,\nu)\ =\ \left(\operatorname*{inf}_{\rho\in\Gamma(\mu,\nu)}\mathbb{E}_{(X,Y)\sim\rho}\big[d(X,Y)^{q}\big]\right)^{1/q},$$

where Γ(*µ, ν*) is the set of joint couplings whose marginals match µ and ν. Given n distributions {νi}
n i=1 and weights
{wi} summing to 1, the L
q*-Wasserstein barycenter* is

$$\bar{\nu}\ =\ \arg\operatorname*{min}_{\nu}\,\sum_{i=1}^{n}w_{i}\,W_{q}(\nu,\nu_{i})^{q}.$$

Our work focuses on q = 1.

## 5.2. Α-Divergence And The L1 **Wasserstein Barycenter**

In many distribution-based backup schemes, the Wasserstein distance is a natural choice to quantify how "far apart" two distributions are. A commonly used approach (Metelli et al., 2019) is to employ the L
2-Wasserstein metric. In contrast, we consider an L
1-Wasserstein formulation coupled with an α-divergence for two main reasons:
- *Robustness & Aggregation Control.* An L
1-based metric can be more robust to outliers and large deviations than L
2.

Furthermore, combining it with the α-divergence allows a continuous interpolation between averaging and max-like backups (through the α parameter).

- *Connection to Power-Mean Updates.* Modeling nodes as Gaussians (or particle distributions) and relying on L
1-
Wasserstein with an α-divergence yields closed-form updates that coincide with the power-mean operator. This unifies average and maximum backups in a single formula and lets us propagate both means and variances (uncertainty) through the tree. f-divergences and the α**-divergence.** An f-divergence (Csiszar, 1964) between two points ´ X and Y over a Manifold M defined as

$$D_{f_{\alpha}}(X\|Y)\;=\;\sum_{i}\;\xi_{Y}^{(i)}\,f_{\alpha}\!\!\left(\frac{\xi_{X}^{(i)}}{\xi_{Y}^{(i)}}\right)\!,f_{\alpha}(x)=\frac{x^{\alpha}-1-\alpha(x-1)}{\alpha(\alpha-1)},$$

where varying α controls how aggressively or conservatively we measure the "distance" between X and Y .

Constructing the L
1**-Wasserstein Barycenter.** In our approach, the L
1-Wasserstein distance between ν and νiis defined via

$$W_{1}(\nu,\nu_{i})\ =\ \operatorname*{inf}_{\rho\,\in\,\Gamma(\nu,\nu_{i})}\mathbb{E}_{(X,Y)\sim\rho}\big[D_{f_{\alpha}}(X,Y)\big].\tag{1}$$

The L
1-Wasserstein barycenter then solves

$$\bar{\nu}\ =\ \arg\operatorname*{inf}_{\nu}\,\Bigl\{\sum_{i=1}^{n}\,w_{i}\,W_{1}\bigl(\nu,\nu_{i}\bigr)\Bigr\},$$

i.e., we seek the single distribution ν¯ that jointly minimizes its L
1-Wasserstein distance (defined via the α-divergence)
to all the νi.

Why L
1**instead of** L
2. Using the L
1 distance in equation 1 naturally leads to a backup rule resembling the *power mean* operator Proposition 1. This power-mean update is more robust to high-variance samples and connects smoothly to both the average backup (when α → 0 or p = 1) and the max backup (as α → ∞ or p → ∞). Hence, L
1-
Wasserstein with α-divergences offers a principled way to blend distributions in highly stochastic environments while controlling the balance between underestimation and overestimation in the final backup.

Why Use an α**-Divergence Instead of** L2? Although αdivergences are not strict metrics (they can be asymmetric and need not satisfy the triangle inequality), their use within an L
1-Wasserstein framework provides distinct benefits for MCTS under stochastic or partially observable conditions:
- *Greater Flexibility via Generalized Means.* When combined with the L
1-Wasserstein distance, an α-divergence naturally yields a *power-mean* style backup operator (Dam et al., 2019). By adjusting the parameter α, one smoothly interpolates between average-like and max-like backups, allowing precise control over how conservative or aggressive the updates should be. This stands in contrast to L
2-based distances, which only yield fixed (e.g.

purely quadratic) aggregation behavior.

- *Robustness to Stochastic Variations.* Because αdivergences can emphasize or de-emphasize portions of the distribution differently depending on α, they help mitigate overestimation or underestimation in highly stochastic settings. Empirical studies in distributional RL (Metelli et al., 2019) suggest that more adaptive divergence measures can significantly improve stability and performance when the underlying dynamics involve heavy noise.

- *No Need for Symmetry in Backups.* MCTS requires a cost functional to aggregate posterior distributions across children nodes, rather than a strict metric. Hence, the lack of symmetry or the triangle inequality does not undermine its validity here. An f-divergence—including αdivergences—is sufficient to drive consistent updates of belief distributions in the tree.

- *Unified Framework for Various Divergences.* The αdivergence family subsumes and generalizes many standard divergences (e.g. KL, reverse KL). This singleparameter approach enables users to easily switch or finetune the update behavior for different problem characteristics, rather than designing separate algorithms for each divergence.

- *Direct Theoretical Connections.* Under mild assumptions, L
1-Wasserstein geometry paired with α-divergences admits closed-form or near-closed-form power-mean formulas (Dam et al., 2019). This not only streamlines theoretical analysis but also simplifies implementation by allowing straightforward computation of mean and variance updates at each node.

In practice, these properties make α-divergences wellsuited for uncertainty propagation within MCTS: despite not being a metric, their adaptability and connection to generalized means allow them to effectively handle complex, high-variance environments.

## 5.3. V-Posterior

It is natural to define a value node as the V-posterior computed with L
1-Wasserstein barycenters of the children nodes Q-posteriors, following a procedure inspired by Metelli et al. 2019 (Metelli et al., 2019) and tailored to MCTS. Definition 1 (V-posterior). Given a policy π¯ *and a state* s ∈ S, we define the V-posterior V(s) induced by Q- posteriors Q(s, a) with a ∈ A *as the* L
1-Wassertein barycenter of the Q*(s, a):*

$${\mathcal{V}}(s)\in{\underset{\mathcal{V}}{\operatorname{arg\,inf}}}\left\{\mathbb{E}_{a\sim{\tilde{\pi}}(.|s)}\left[W_{1}({\mathcal{V}},{\mathcal{Q}}(s,a))\right]\right\}.$$

In this work, we model each node in the tree as a Gaussian distribution. We define p = 1−α and derive the following.

Proposition 1. *Consider the V-posterior value function* V(s) *as a Gaussian:* N (m(s), σ 2(s)). Define each Q(*s, a*)
as the action-value function child node of V(s)*. Each* Q(s, a) *is assumed as a Gaussian distributions* Q(s, a) : N (m(s, a), σ(*s, a*)
2). If the value function V(s) is defined as the Wasserstein barycenter of the action-value function Q(s, a), given the policy π¯*, we have*

$$\begin{array}{l}{{\overline{{{m}}}(s)=(\mathbb{E}_{a\sim\overline{{{\pi}}}}[m(s,a)^{p}])^{\frac{1}{p}}}}\\ {{\overline{{{\delta}}}(s)=(\mathbb{E}_{a\sim\overline{{{\pi}}}}[\delta(s,a)^{p}])^{\frac{1}{p}}.}}\end{array}$$

Proposition 1 shows the closed form solutions of the mean and standard deviation of the Gaussian value function V(s) considering it as the L
1-Wasserstein barycenter Q-
posteriors. In detail, the mean of V(s) are the power mean of all mean values of all the Q(*s, a*) function, considering the finite set of actions. When p = 1, we derive the expected form solutions. We point out that our approach is not restricted to the Gaussian distribution model. We get the following result by considering each tree node as a particle model.

Proposition 2. *Consider the V-posterior value function* V(s) *as an equally weighted Particle model:* xi(s) : i ∈
[1, M]. M is an integer and M ⩾ 1. Assume each action-value function Q(s, a) has M particles xi(s, a), i ∈
[1, M]. If the value function V(s) is defined as the Wasserstein barycenter of the action-value function Q(s, a), given the policy π¯, each particle xi(s), i ∈ [1, M] can be estimated as

$$\overline{{{x_{i}}}}(s)=(\mathbb{E}_{a\sim\bar{\pi}}[x_{i}(s,a)^{p}])^{1/p},$$

Proposition 2 shows that each particle of the V-posterior value function V(s) can be derived as the power mean of the respective particles of all the Q(*s, a*) function. If p = 1, we again get the closed-form solutions as the expectation of the respective particles of all the Q(*s, a*) functions. The results in Proposition 1, and Proposition 2 can be considered as the generalized result of Proposition A.3 in Metelli et al. (2019). In the next section, we present our Wasserstein Monte-Carlo tree search (W-MCTS ) algorithm, assuming each tree node is a Gaussian distribution.

## 6. Wasserstein Monte-Carlo Tree Search

We introduce our Wasserstein Monte-Carlo Tree Search (W-MCTS), where V-posteriors are modeled as Wasserstein barycenters of action-value distributions. With Gaussian distributions at each node, we define backup operators for mean and variance. Additionally, we propose two action selection strategies: optimistic selection and Thompson sampling.

## 6.1. Backup Operator

We model each V -node and Q-node as a Gaussian with mean and standard deviation:

$$V_{\mathrm{m}}(s),\;V_{\mathrm{std}}(s)\quad\mathrm{and}\quad Q_{\mathrm{m}}(s,a),\;Q_{\mathrm{std}}(s,a).$$

We denote V m(*s, N*(s)) as the *empirical mean estimate* of the V -node at state s after N(s) total visits, and Qm(s, a, n(*s, a*)) as the *empirical mean estimate* of the Q-
node at (s, a) after n(*s, a*) visits. Likewise, V std(*s, N*(s))
and Qstd(s, a, n(*s, a*)) are their corresponding empirical standard deviation estimates. V **-nodes.** From Proposition 1, the mean and the standard deviation of a V -node is a power-mean aggregation of its Q-children:

$$\begin{array}{r c l}{{}}&{{}}&{{\overline{{{V}}}_{\mathrm{m}}(s,N(s))\ \leftarrow\ \left(\sum_{a}\,\frac{n(s,a)}{N(s)}\,\big[\overline{{{Q}}}_{\mathrm{m}}(s,a,n(s,a))\big]^{p}\right)^{1/p},}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}\end{array}$$

where n(*s, a*) is the visit count of action a at state s, and N(s) = Pa n(*s, a*). For p = 1, this reduces to the standard average, whereas p > 1 induces a more "max-like" backup (Dam et al., 2019). Q**-nodes.** Under the Bellman-style backup for each Q- node,

$$Q_{\mathrm{m}}(s,a)=\mathbb{E}[r(s,a)]+\gamma\,\mathbb{E}[V_{\mathrm{m}}(s^{\prime})],Q_{\mathrm{std}}(s,a)=\gamma\,V_{\mathrm{std}}(s^{\prime}),$$

we replace expectations by empirical sums and visitation counts:

$$\begin{array}{r c l}{{\overline{{{Q}}}_{\mathrm{m}}(s,a,n(s,a))}}&{{\leftarrow}}&{{\frac{\sum r(s,a)+\gamma\sum_{s^{\prime}}N(s^{\prime})\,\overline{{{V}}}_{\mathrm{m}}(s^{\prime},N(s^{\prime}))}{n(s,a)},}}\\ {{}}&{{}}&{{}}\\ {{}}&{{\overline{{{Q}}}_{\mathrm{std}}(s,a,n(s,a))}}&{{\leftarrow}}&{{\frac{\gamma\sum_{s^{\prime}}N(s^{\prime})\,\overline{{{V}}}_{\mathrm{std}}(s^{\prime})}{n(s,a)}.}}\end{array}$$

Here, the sums range over transitions and children states s
′, weighted by their visit counts N(s
′). As n(s, a) grows large, both the variance and mean estimators stabilize, eventually converging to deterministic values.

## 6.2. Action Selection

Monte Carlo Tree Search can adopt a variety of exploration strategies based on the original UCT framework (Kocsis et al., 2006). In practice, multiple refinements exist, such as the variants used in AlphaGo (Silver et al., 2016b), AlphaZero (Silver et al., 2017c;a), MuZero (Schrittwieser et al., 2020), Stochastic MuZero (Antonoglou et al., 2021), and Stochastic-Power-UCT (Dam et al., 2024b). Although different choices of the exploration constant or bonus lead to different performance characteristics, we retain the standard, state-of-the-art designs described below. In our theoretical analysis, however, we focus specifically on Thompson sampling, since the UCT-like optimistic selection can be viewed as a special case of the well-studied Power-UCT algorithm (Dam et al., 2019; 2024b). Optimistic Selection. A classic UCT-style selection picks actions using upper confidence bounds on Q-values,

$$a\;=\;\operatorname{argmax}_{a_{i}}\left[m(s,a_{i})\;+\;C\,{\sqrt{\frac{\log N(s)}{n(s,a_{i})}}}\right],$$

where m(*s, a*i) is the empirical mean, n(s, ai) is the visit count of action ai, and N(s) is the total visit count at state s. Replacing the √
1 n(s,ai)
term by the empirical standard deviation σ(*s, a*i) yields an *optimistic* variant of Wasserstein MCTS (W-MCTS-OS):

$$a\;=\;\operatorname*{argmax}_{a_{i}}\left[m(s,a_{i})\;+\;C\,\sigma(s,a_{i})\,\sqrt{\log N(s)}\right].$$
The factor $\sigma(s,a_i)\approx1/\sqrt{n(s,a_i)}$ follows from a CLT-. 
based argument.
5 Thompson Sampling. In contrast, Thompson sampling stochastically samples an action from the Q-posterior:

$$a\;=\;\operatorname*{argmax}_{a_{i}}\left\{\theta_{i}\sim{\mathcal{N}}\big(m(s,a_{i}),\,\sigma^{2}(s,a_{i})\big)\right\}.$$

We refer to this Thompson variant as Wasserstein MCTS- TS (W-MCTS-TS). In Section 7, we analyze its convergence properties under non-stationary multi-armed bandits and then leverage these results to establish convergence in the planning tree.

## 7. Theoretical Analysis 7.1. Analysis Setup

We define the setting for our theoretical analysis using a class of non-stationary Multi-Armed Bandit (MAB) problems at each state s in the MCTS tree. Consider K arms
(actions), each with a mean reward µk, for k ∈ [K]. At time step t, pulling arm k yields a random reward Xk,t, bounded within [0, R]. The average reward for arm k after n trials is:

$${\overline{{X}}}_{k,n}={\frac{1}{n}}\sum_{t=1}^{n}X_{k,t},\quad{\mathrm{with}}\quad\mu_{k,n}=\mathbb{E}[{\overline{{X}}}_{k,n}]$$

Let ⋆ represent quantities related to the optimal arm, and denote Tk(n) as the number of times arm k has been played by step n. We assume the following *concentration* condition holds: Assumption 1. *We assume that the reward sequence,*
{Xk,t : t ⩾ 1}, is a non-stationary process satisfying the assumption: for all 1 > ε > 0, ∃c > 0 *that*

$$\mathbf{Pr}\left(|{\overline{X}}_{k,n}-\mu_{k}|>\varepsilon\right)\leqslant c n^{-1}\varepsilon^{-2},k\in[K].\tag{2}$$

## 7.2. Main Results

We show the polynomial convergence of the expected estimated mean value function at the root node in Theorem 1.

$$\pi$$
$\quad\;\;$CONVERO. 
$\mathbb{N}-\mathbb{M}\mathbb{C}\mathbb{T}\mathbb{S}_+$
7.2.1. CONVERGENCE OF W-MCTS
We start with an important result as shown below Proposition 3. Applying W-MCTS to an MCTS tree of depth (H), at any depth h of the tree, we have
(i) At any depth h, ∃ constant C0 > 0 *that for any* 0 <
ε < 0, n ⩾ 1*, we can derive*

$$\mathbf{Pr}{\bigg(}\left|\nabla_{m}(s_{h},a_{k},n)-{\tilde{V}}(s_{h},a_{k})\right|\geqslant\varepsilon{\bigg)}\leqslant C_{0}n^{-1}\varepsilon^{-2}.$$
_(ii) At any depth $h$, $\exists$ constant $C_{0}>0$ that for any $0$._
ε < 0, n ⩾ 1*, we can derive*
$$\mathbf{E}\mathbf{F}\mathbf{E}$$
$=\;\cos x\;=\;\frac{\pi}{2}$  . 
$$\mathbf{Pr}\bigg(\left|\overline{{{Q}}}_{m}(s_{h},a_{k},n)-\overline{{{Q}}}(s_{h},a_{k})\right|\geqslant\varepsilon\bigg)\leqslant C_{0}n^{-1}\varepsilon^{-2}.$$

## Proof Sketch

MCTS as a Hierarchical Bandit Structure. The Monte Carlo Tree Search (MCTS) algorithm can be viewed as a hierarchy of multi-armed bandits (MABs), where each node in the search tree represents an independent bandit problem. In this framework, the reward for each node, or current bandit, is influenced by the performance of the bandit algorithms applied to its child nodes. Since the W-MCTS policy adapts dynamically to balance exploitation and exploration, the rewards at each node are inherently non-stationary. The proof of Theorem 1 unfolds through three essential steps: 1. Analyzing Non-stationary Bandits The initial step focuses on the analysis of a non-stationary multi-armed bandit, which reflects the behavior of MABs at each MCTS node. We establish that if the rewards of these nonstationary bandits meet specific *concentration* properties, the regret induced by the W-MCTS algorithm will exhibit corresponding concentration guarantees. This outcome is formally stated in Theorem 2. 2. Induction Argument Next, we utilize an inductive argument to transfer the convergence and concentration properties from the lower tree levels to the root node. As the rewards from one level inform those of the next, the findings from Step 1 can be recursively applied. We begin at depth H − 1 and move upward, demonstrating inductively that the bandit rewards at each level H of the MCTS satisfy the criteria required by Theorem 2. This process propagates the desired properties up to the root node, completing the induction. 3. Error Analysis from the Oracle The final step examines the error introduced by the leaf node estimator, represented by the value function oracle V0. With this oracle, the depth-H MCTS can be interpreted as performing H steps of value iteration, starting from V0 at the leaf nodes (as mentioned in (Dam et al., 2024b)). Importantly, the oracle's error decreases geometrically at a rate of γ due to the contraction mapping property of value iteration, leading to diminishing error as we ascend from the leaf nodes to the root. The complete proof for Proposition 3 can be found in the supplemental material. Finally, we get the main result.

Theorem 1. *We have at the root node* s0,

$$\left|\mathbb{E}[\overline{{{V}}}_{m}(s_{0},n)]-\widetilde{V}(s_{0})\right|\leqslant\mathcal{O}(n^{-1/2}).$$
$$t h a t\,f o r\,a n y\,\,0\,<\,$$
$${\mathrm{y}}\ 0\ <$$

Our proposed method, W-MCTS, achieves a polynomial convergence rate of O(n
−1/2), matching the results of Dam et al. (2024b). In contrast, Xiao et al. (2019) introduced MENTS, followed by RENTS and TENTS from Dam et al. (2021), which leverage exponential convergence to a regularized value function through maximum entropy regularization. However, these methods face bias due to errors in the regularized value function, potentially leading to incorrect action selection. Conversely, Painter et al. (2024) employ a similar action selection strategy with a maximum backup operator for value estimation, resulting in exponential reductions in simple regret. However, their method's effectiveness heavily relies on the temperature parameter in Boltzmann exploration, limiting its practical use.

## 7.2.2. Wasserstein Non-Stationary Multi-Armed Bandit

A crucial part of the proof for Theorem 1 is to derive the following result for the W-MCTS in bandit setting. Under the Assumption 1, we consider applying Thompson Sampling strategy as the action selection method for the nonstationary multi-armed bandit (MAB) problems describes above. At each time step n, an action is selected as

$$a=\operatorname*{argmax}_{a_{i},i\in\{1...K\}}\{\theta_{i}\sim\mathcal{N}(\overline{X}_{k,n},V_{k}/T_{k}(n))\}.\tag{3}$$

Let's define Xn(p) = 
PK
a=1 Ta(n)
n X
p a,Ta(n)
1/pas the power mean value backup at the root node, Ta(n) =
Pn−1 t=1 1(at = a) is the number of selections of a prior to round n. We show theoretical results of our method as follows. Under the Assumption 1, we establish the concentration properties of the power mean backup operator Xn(p) towards the mean value of the optimal arm µ∗ =
maxa{µa}, a ∈ [K], as shown in Theorem 2. Theorem 2. Consider a non-stationary bandit problem described as in 7.1 with action selection as Equation (3). Then,

$$\mathbf{Pr}(|{\overline{{X}}}_{n}(p)-\mu_{\star}|\geqslant\varepsilon)\leqslant C n^{-1}\varepsilon^{-2}.$$

Theorem 2 states the concentration properties of the power mean estimation by W-MCTS for a non-stationary continuous-armed bandit problem, and play an important role for the induction proof of Proposition 3 leading to the main result presented at Theorem 1.

## 8. Experiments 8.1. Fully Observable, Highly Stochastic Tasks

We compare W-MCTS to UCT (Kocsis et al., 2006), Power-UCT (Dam et al., 2019), and DNG (Bai et al., 2013)
in five benchmark environments: FrozenLake, NChain, RiverSwim, *SixArms*, and *Taxi*. These tasks all feature significant stochasticity or long-horizon exploration challenges.

FrozenLake. A 4 × 4 grid with slippery transitions, implemented in OpenAI Gym (Brockman et al., 2016). The agent aims to reach a goal in the bottom-right corner. Due to frequent slips, each move has high uncertainty. Figure 1 shows that W-MCTS-TS (Thompson sampling) outperforms DNG, UCT, Power-UCT, and W-MCTS (optimistic selection), with W-MCTS at p = 1 performing comparably to W-MCTS-TS. NChain. An agent can move forward or backward along a chain of length 5. Actions may reverse with 20% probability, making consistent forward progress difficult. In Figure 1, both W-MCTS-TS and W-MCTS-OS exceed UCT and Power-UCT in convergence speed and final returns. RiverSwim. Similar to *NChain* but more complex transitions: sometimes the agent remains in the same state or only partially moves. This rewards long-term planning to reach high-value states. As in Figure 1, W-MCTS-OS
converges fastest and attains the best performance, while Power-UCT eventually reaches similar returns more slowly. SixArms. A 7-state chain with 6 possible arms (actions) leading to different rewards that scale inversely with their success probabilities. This environment demands high exploration. Figure 1 shows that W-MCTS is the only method consistently securing strong returns. Taxi. A 7 × 6 grid where the agent must pick up three passengers, then reach a goal region. Slips occur 10% of the time, adding further uncertainty. Only W-MCTS-TS manages to collect all passengers reliably, outperforming Power-UCT and W-MCTS with optimistic selection.

## 8.2. Partially Observable, Highly Stochastic Tasks

We also test W-MCTS against POMCP(UCT), D2NG, and DESPOT in classic POMDP benchmarks: rocksample, pocman, Tag, and *LaserTag*. Code for POMCP(UCT) (Silver & Veness, 2010b), D2NG (Bai et al., 2014), and DESPOT
(Somani et al., 2013) is used as released by the original authors.

Rocksample. A robot on an n×n grid can sample or ignore k rocks, then exit. We test three variants: (11,11), (15,15), and (15,35). Figure 2 shows that W-MCTS-TS consistently outperforms both UCT and D2NG.

Pocman. A partially observed maze (Silver & Veness, 2010a) where the agent must collect pellets while avoiding ghosts. Table 1 indicates that W-MCTS-TS with p = 100 outperforms UCT and D2NG across most rollout-budget settings, and W-MCTS-OS also matches or surpasses these baselines in some configurations. Comparison with **DESPOT**. We additionally compare W-MCTS to DESPOT across Tag, LaserTag, *rocksample* (15 × 15), and *Pocman*. Table 2 shows that W-MCTS-OS and W-MCTS-TS achieve higher returns than AB-DESPOT and AR-DESPOT in rocksample. Similarly, W-MCTS-TS surpasses DESPOT in Pocman, Tag, and *LaserTag*, while W-MCTS-OS outperforms AB-DESPOT in *Pocman*. **Role**

FrozenLake NChain RiverSwim 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 2 3 4 5 6 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.0 0.1 0.2 0.3 0.4 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations1e5 0.0 0.2 0.4 0.6 0.8 Dis counted Re tur n Dis counted Re tur n Dis counted Re tur n SixArms Taxi 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.0 0.1 0.2 0.3 0.4 0.5 Dis counted Re tur n Dis counted Re tur n DNG Power-UCT UCT W-MCTS-OS W-MCTS-TS W-MCTS-TS, p=1 Dis co un ted Re turn rocksample 11x11 (16 actions)
Dis co un ted Re turn rocksample 15x15 (20 actions)
Dis co un ted Re turn rocksample 15x35 (40 actions)
0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 5 0 5 10 15 20 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 5 0 5 10 15 20 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations1e5 5 0 5 10 15 20 D2NG UCT W-MCTS-OS W-MCTS-TS
Table 1: Discounted total reward in *pocman*. Mean ± standard error are computed from 1000 random seeds.

| 1024                | 4096                      | 32768                                  | 65536        |             |
|---------------------|---------------------------|----------------------------------------|--------------|-------------|
| W-MCTS-OS , p = 1   | 50.9 ± 0.6                | 51.0 ± 0.62                            | 52.2 ± 0.79  | 54.6 ± 1.08 |
| W-MCTS-TS , p = 100 | 67.38 ± 0.53              | 75.64 ± 0.51 77.68 ± 0.77 77.70 ± 1.22 |              |             |
| D2NG                | 71.55 ± 0.57 75.39 ± 1.47 | 76.90 ± 6.40                           | 72.2 ± 0.0   |             |
| UCT                 | 23.4 ± 0.99               | 23.6 ± 1.09                            | 24.90 ± 3.40 | 28.5 ± 3.8  |

Table 2: Average total discounted reward. The results for POMCP, and DESPOT are taken from (Somani et al., 2013).

| T ag      | LaserT ag                  | RS(15, 15)   | P ocman       |               |
|-----------|----------------------------|--------------|---------------|---------------|
| W-MCTS-OS | −6.05 ± 0.56 −18.17 ± 0.46 | 19.76 ± 0.28 | 297.98 × 2.83 |               |
| W-MCTS-TS | −5.90 ± 0.66               | −8.75 ± 0.5  | 20.29 ± 0.22  | 315.45 ± 2.15 |
| POMCP     | −7.14 ± 0.28 −19.58 ± 0.06 | 12.23 ± 0.32 | 294.16 ± 4.06 |               |
| AB-DESPOT | −6.57 ± 0.26 −11.13 ± 0.30 | 18.18 ± 0.30 | 290.34 ± 4.12 |               |
| AR-DESPOT | −6.26 ± 0.28               | −9.34 ± 0.26 | 18.57 ± 0.30  | 307.96 ± 4.22 |

of α**-Divergence.** We explored several values of α to vary how aggressively our backups shift between average-like and max-like behavior. When α approaches 0 or ∞, the update becomes nearly a pure average (p = 1) or nearly a max backup, respectively. In practice, we found that moderate α values often provide a suitable balance between these extremes, and we report results with the bestperforming choices. Although a more extensive sensitivity analysis could be conducted, the core takeaway is that combining power-mean backups with variance propagation significantly enhances performance in highly stochastic tasks.

## 8.3. Key Performance Factors

The superior performance of our method stems from two complementary components that address fundamental limitations in existing MCTS approaches for stochastic and partially observable environments:
Munos, 2010) for even broader applicability.

Explicit Variance Propagation. Unlike previous methods that only propagate point estimates or use fixed variance models, our approach dynamically updates both means and variances at each node through the L
1-Wasserstein barycenter formulation. This capability is particularly crucial in highly stochastic and partially observable environments where uncertainty quantification directly impacts decision quality. Our experimental results demonstrate consistent improvements over Bayesian MCTS methods: we achieve up to 80% improvement over DNG in Frozen- Lake, and significant gains over POMCP across all POMDP
environments, with particularly notable improvements of 55.31% in *LaserTag* and 65.90% in *rocksample*(15,15). Additionally, we observe improvements of up to 21.38%
over AB-DESPOT in *LaserTag*, highlighting the effectiveness of our distributional approach. Flexibility in Balancing Exploration-Exploitation. Our approach's ability to interpolate between average-like and max-like backups through the α-divergence parameter allows adaptive behavior across varying levels of stochasticity. In highly stochastic environments such as *FrozenLake* and *NChain*, we found that moderate α values (leading to more average-like updates with p closer to 1) performed optimally by preventing overestimation bias. Conversely, in environments with more deterministic regions of the state space, larger α values (yielding more max-like behavior) proved beneficial for faster convergence to optimal policies. This flexibility, combined with our Thompson sampling strategy, enables our algorithm to automatically adapt its exploration-exploitation balance based on the empirical variance observed at each node. The synergy between these two components—principled uncertainty propagation and adaptive backup operatorsexplains why W-MCTS consistently outperforms both classical MCTS variants and existing Bayesian approaches across our diverse set of benchmark environments.

## 9. Conclusion

We proposed *Wasserstein MCTS*, an algorithm that models node values as Gaussian distributions and employs L
1-Wasserstein barycenters with α-divergences to unify average- and max-like backups. Coupled with Thompson sampling or optimistic selection, our method achieves strong empirical performance while offering O(n
−1/2)
convergence guarantees. Experiments in both stochastic MDPs and POMDPs show significant improvements over classic baselines and Bayesian MCTS variants. Future work includes extending these Wasserstein-based ideas to open-loop planning (Leurent & Maillard, 2020; Bubeck &

## Impact Statement

Our proposed *Wasserstein MCTS* algorithm offers a principled way to tackle complex, stochastic tasks in both fully and partially observable domains. Potential applications include robotics, autonomous systems, and large-scale resource management, all of which require adaptive planning strategies to handle real-world variability. While we do not anticipate immediate negative societal implications, responsible deployment remains essential. As with any AI- driven technology, understanding ethical, economic, and security ramifications—such as autonomy in safety-critical systems—should guide practical use.

## Acknowledgments

This research is funded by Hanoi University of Science and Technology (HUST) under Project No. T2024-TD-024, the French Ministry of Higher Education and Research, the Hautsde-France region, Inria, the MEL, the French National Research Agency under PEPR IA FOUNDRY project (ANR-23-PEIA-0003).

## References

Antonoglou, I., Schrittwieser, J., Ozair, S., Hubert, T. K., and Silver, D. Planning in stochastic environments with a learned model. In International Conference on Learning Representations, 2021. Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002a. Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. *Mach. Learn.*, 47(2–3):235–256, may 2002b. ISSN 0885-6125. doi: 10. 1023/A:1013689704352. URL https://doi.org/10.1023/A: 1013689704352. Bai, A., Wu, F., and Chen, X. Bayesian mixture modelling and inference based thompson sampling in monte-carlo tree search. *Advances in neural information processing* systems, 26, 2013. Bai, A., Wu, F., Zhang, Z., and Chen, X. Thompson sampling based monte-carlo planning in pomdps. the International Conference on Automated Planning and Scheduling, 24(1), 2014. Bellman, R. The theory of dynamic programming. Technical report, Rand corp santa monica ca, 1954.

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. Openai gym. arXiv preprint arXiv:1606.01540, 2016. Bubeck, S. and Munos, R. Open loop optimistic planning.

In *COLT 2010-The 23rd Conference on Learning Theory*,
2010. Cichocki, A. and Amari, S.-i. Families of alpha- betaand gamma- divergences: Flexible and robust measures of similarities. *Entropy*, 12(6):1532–1568, 2010. ISSN 1099-4300. doi: 10.3390/e12061532. URL https://www. mdpi.com/1099-4300/12/6/1532. Csiszar, I. Eine informationstheoretische ungleichung und ´ ihre anwendung auf beweis der ergodizitaet von markoffschen ketten. *Magyer Tud. Akad. Mat. Kutato Int. Koezl.*, 8:85–108, 1964. Dam, T., Klink, P., D'Eramo, C., Peters, J., and Pajarinen, J. Generalized mean estimation in monte-carlo tree search. *arXiv preprint arXiv:1911.00384*, 2019. Dam, T., D'Eramo, C., Peters, J., and Pajarinen, J. A unified perspective on value backup and exploration in monte-carlo tree search. Journal of Artificial Intelligence Research, 81:511–577, 2024a.

Dam, T., Maillard, O.-A., and Kaufmann, E. Power mean estimation in stochastic monte-carlo tree search. The 40th Conference on Uncertainty in Artificial Intelligence (UAI), 2024b. Dam, T. Q., D'Eramo, C., Peters, J., and Pajarinen, J. Convex regularization in monte-carlo tree search. In International Conference on Machine Learning, pp. 2365–2375.

PMLR, 2021.

Gerchinovitz, S., Menard, P., and Stoltz, G. Fano's in- ´ equality for random variables. *Statist. Sci*, 2020. Hasselt, H. V. Double q-learning. In Advances in Neural Information Processing Systems, 2010. Jin, T., Xu, P., Xiao, X., and Anandkumar, A. Finite-time regret of thompson sampling algorithms for exponential family multi-armed bandits. Advances in Neural Information Processing Systems, 35:38475–38487, 2022. Kocsis, L., Szepesvari, C., and Willemson, J. Improved ´ monte-carlo search. *Univ. Tartu, Estonia, Tech. Rep*, 1, 2006. Leurent, E. and Maillard, O.-A. Practical open-loop optimistic planning. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019, Wurzburg, Germany, September 16–20, ¨ 2019, Proceedings, Part III, pp. 69–85. Springer, 2020.

Metelli, A. M., Likmeta, A., and Restelli, M. Propagating uncertainty in reinforcement learning via wasserstein barycenters. Advances in Neural Information Processing Systems, 32, 2019. Painter, M., Baioumy, M., Hawes, N., and Lacerda, B. Monte carlo tree search with boltzmann exploration. Advances in Neural Information Processing Systems, 36, 2024. Perlman, M. D. Jensen's inequality for a convex vectorvalued function on an infinite-dimensional space. Journal of Multivariate Analysis, 4(1):52–65, 1974. ISSN 0047-259X. doi: https://doi.org/10.1016/0047-259X(74) 90005-0. URL https://www.sciencedirect.com/science/ article/pii/0047259X74900050. Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., et al. Mastering atari, go, chess and shogi by planning with a learned model. *Nature*, 588 (7839):604–609, 2020. Silver, D. and Veness, J. Monte-carlo planning in large pomdps. In *Advances in neural information processing* systems, 2010a.

Silver, D. and Veness, J. Monte-carlo planning in large pomdps. In Lafferty, J., Williams, C., Shawe-Taylor, J., Zemel, R., and Culotta, A. (eds.), Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc., 2010b. URL https://proceedings.neurips.cc/paper/2010/file/ edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587):484–489, January 2016a. doi: 10.1038/nature16961. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484, 2016b. Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815, 2017a.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., and Hassabis, D. Mastering the game of go without human knowledge. Nature, 550:354–, October 2017b. URL http://dx.doi.org/ 10.1038/nature24270. Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. Mastering the game of go without human knowledge. *Nature*, 550(7676):354–359, 2017c. Soch, J. The book of statistical proofs. https://statproofbook.github.io, 2020. URL https://statproofbook.github.io/P/norm-qf.html. Somani, A., Ye, N., Hsu, D., and Lee, W. S. Despot: Online pomdp planning with regularization. Advances in neural information processing systems, 26, 2013.

Tesauro, G., Rajan, V., and Segal, R. Bayesian inference in monte-carlo tree search. *arXiv preprint* arXiv:1203.3519, 2012. Thompson, W. R. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4):285–294, 1933. Xiao, C., Huang, R., Mei, J., Schuurmans, D., and Muller, ¨ M. Maximum entropy monte-carlo planning. In Advances in Neural Information Processing Systems, pp. 9516–9524, 2019.

## Outline

- Notations will be described in Section A. - Hyperparameters are provided in Section B. - Derivation of Wasserstein barycenter with Gaussian and particle filter distributions will be described in Section C. - Supporting Lemmas will be provided in Section D.

- Full proof for the convergence of Wasserstein Non-stationary multi-armed bandit will be provided in Section E.

- Full proof for the convergence of Wasserstein Monte-Carlo tree search will be provided in Section F.

## A. Notations

| Notation     | Type                                     | Description                                             |      |
|--------------|------------------------------------------|---------------------------------------------------------|------|
| N (m, δ2 )   | R                                        | Gaussian distribution with mean m, standard deviation δ |      |
| (X , d)      | complete separable metric (Polish) space |                                                         |      |
| Wq(µ, ν)     | L q -Wasserstein distance between µ, ν   |                                                         |      |
| W1(µ, ν)     | L 1 -Wasserstein distance between µ, ν   |                                                         |      |
| F −1 (t)     | quantile function of a distribution p(x) |                                                         |      |
| p(x) Γ(µ, ν) | X × Y                                    | set of measures on X × Y with marginals µ, ν            |      |
| d(X, Y )     | R                                        | distance between X and Y                                |      |
| Dfα (X||Y )  | R                                        | α-divergence distance between X and Y                   |      |
| erf−1 (t)    | the inverse of the function √ 2          | R t                                                     | 2}dx |
| exp{−x       |                                          |                                                         |      |
| π            | 0                                        |                                                         |      |

Table 3: List of all notations of Wasserstein barycenter with Gaussian and particle filter distributions.

## B. Experimental Setup And Parameters Selection

All the experiments were done on an Intel(R) Core(TM) i9-14900K 3.20 GHz 24 cores/CPU.

To compare the performance of W-MCTS to other state-of-the-art planning algorithms, we run several experiments on standard MDP as well as POMDP environments. For comparison, we consider UCT (Kocsis et al., 2006), Power-UCT (Dam et al., 2019), DNG (Bai et al., 2013) and D2NG (Bai et al., 2014). The hyperparameters are tuned using grid-search. Except for the case of *Pocman* environment, we scale the rewards into the range [0, 1]. We use the discount factor γ = 0.95. For DNG, D2NG, we set hyperparameters as recommended in the paper and from the author's source code (Bai et al., 2013; 2014). We set exploration constant for UCT, Power-UCT to √2. We set initial standard deviation value to std = 30. In all Rocksample and *Pocman* environments, we set the heuristic for rollouts as treeknowledge = 0*, rolloutknowledge* = 1.

For all environments, we increase the value of p and choose the best power mean p value for Power-UCT, and W-MCTS .

Details can be found in Table 6. For POMDP environments such as Rocksample, *Pocman* we get the source code released from the author of DNG (Bai et al., 2013) and D2NG (Bai et al., 2014)1.

1https://github.com/aijunbai/thompson-sampling

| Notation   | Type   | Description                                           |
|------------|--------|-------------------------------------------------------|
| K          | N      | number of arms/actions                                |
| µk         | R      | mean value of arm k                                   |
| ∗          | R      | optimal mean value                                    |
| µ △k       | R      | △k = µ ∗ − µk                                         |
| △          | R      | △ = maxk∈[K]{△k}                                      |
| X ∗ s      | R      | average reward of the optimal arm after s visitations |
| F s        | R      | CDF of Gaussian with mean X                           |
| ∗          | ∗ s    |                                                       |
| Tk(n)      | N      | number of visitations of arm k at timesteps n         |
| Xn(p)      | R      | power mean backup operator with power p               |
| Xk,Tk(n)   | R      | average rewards of arm k after Tk(n) visits           |

## C. Derivation Of Wasserstein Barycenter With Gaussian And Particle Filter Distributions

We revisit the definition of Wasserstein distance: The L
q-Wasserstein distance (with q > 0) between two distributions *µ, ν* with the cost function d(*x, y*) : *X × Y →* R is defined as

$$W_{q}(\mu,\nu)=\left(\operatorname*{inf}_{\rho\in\Gamma(\mu,\nu)X,Y\sim\rho}[d(X,Y)^{q}]\right)^{1/q},$$
$$(4)$$

here Γ(µ, ν) is the set of measures on *X × Y* with marginals *µ, ν*.

Define F
−1 p(x)
(t) as the quantile function of a distribution

$$p(x):F_{p(x)}^{-1}(t)=\operatorname*{inf}\{x\in\mathbb{R},t\leqslant F_{p}(x)\}.$$
$$(5)$$
$$\mathbb{n}\,\operatorname{derivative}$$
(t) = inf{x ∈ R, t ⩽ Fp(x)}. (5)
With $d(X,Y)=|X-Y|$. 
With d(*X, Y* ) = |X − Y | as the Euclidean distance, we can derive

$$W_{q}^{q}(\mu,\nu)=\bigg(\int_{0}^{1}|F_{\mu}^{-1}(t)-F_{\nu}^{-1}(t)|^{q}d t\bigg).$$
qdt. (6)
$\zeta,Y)=D_{f_2}(X||Y)$
With d(*X, Y* ) = Dfα
(X||Y ), as the α-divergence distance (defined in section 4.1), we can derive

$$W_{q}^{q}(\mu,\nu)=\bigg(\int_{0}^{1}D_{f_{\alpha}}(F_{\mu}^{-1}(t)||F_{\nu}^{-1}(t))^{q}d t\bigg).$$
ν(t))qdt. (7)
C.1. L
1-Wasserstein barycenter with α**-divergence distance**
We have

$$W_{1}(\mu,\nu)=\operatorname*{inf}_{\rho\in\Gamma(\mu,\nu)X,Y\sim\rho}[d(X,Y)]=\operatorname*{inf}_{\rho\in\Gamma(\mu,\nu)X,Y\sim\rho}[D_{f_{\alpha}}(X,Y)].$$

[Dfα(*X, Y* )]. (8)
$$(6)^{\frac{1}{2}}$$

$$(7)^{\frac{1}{2}}$$

$$({\boldsymbol{\delta}})$$

13

| Notation      | Type          | Description                                                                                         |                                                                    |
|---------------|---------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| KL            | KL divergence |                                                                                                     |                                                                    |
| Vm(sh)        | R             | optimal mean of V value at root state sh, at depth (h)                                              |                                                                    |
| Qm(sh, ak)    | R             | mean of Q value function at state sh, action ak, at depth (h)                                       |                                                                    |
| V m(sh, n)    | R             | empirical estimated mean of V value at state sh after n visitations at depth (h)                    |                                                                    |
| Qm(sh, ak, n) | R             | empirical estimated mean of Q value at root at state sh, action ak after n visitations at depth (h) |                                                                    |
| Vm(sh)        | R             | optimal mean of V value at depth (h) at state sh                                                    |                                                                    |
| Qm(sh, ak)    | R             | mean of Q value function at depth (h) at state sh, action ak                                        |                                                                    |
| V m(sh, n)    | R             | empirical estimated mean of V value at depth (h) at state sh after n visitations                    |                                                                    |
| Qm(sh, ak, n) | R             | empirical estimated mean of Q value at depth (h) at state sh, action ak after n visitations         |                                                                    |
| Tsh,ak (n)    | N             | number of plays of action ak at state sh at timestep n                                              |                                                                    |
| ′             |               |                                                                                                     |                                                                    |
| T s           | (n)           | N                                                                                                   | number of plays of action ak at state s to state s ′ at timestep n |
| s,ak          |               |                                                                                                     |                                                                    |

| Environments      | p Value Search           | Best p Value                                         |
|-------------------|--------------------------|------------------------------------------------------|
| FrozenLake        | p = 1, 2, 4, 10, 100     | W-MCTS-OS (p=100),W-MCTS-TS (p=100),Power-UCT(p=100) |
| NChain            | p = 1, 2, 4, 8, 15, 100  | W-MCTS-OS (p=4),W-MCTS-TS (p=100),Power-UCT(p=8)     |
| RiverSwim         | p = 1, 2, 4, 8, 15, 100  | W-MCTS-OS (p=100),W-MCTS-TS (p=15),Power-UCT(p=15)   |
| SixArms           | p = 1, 2, 4, 8, 15, 100  | W-MCTS-OS (p=100),W-MCTS-TS (p=100),Power-UCT(p=8)   |
| Taxi              | p = 1, 2, 4, 8, 15, 100  | W-MCTS-OS (p=15),W-MCTS-TS (p=15),Power-UCT(p=15)    |
| Rocksample(11x11) | p = 10, 50, 80, 100, 150 | W-MCTS-OS (p=150),W-MCTS-TS (p=100)                  |
| Rocksample(15x15) | p = 10, 50, 80, 100, 150 | W-MCTS-OS (p=100),W-MCTS-TS (p=100)                  |
| Rocksample(15x35) | p = 10, 80, 100          | W-MCTS-OS (p=150),W-MCTS-TS (p=10)                   |
| Pocman            | p = 1, 2, 4, 8, 10, 100  | W-MCTS-OS (p=1),W-MCTS-TS (p=100)                    |

We find the lower bound of W1(*µ, ν*) with α-divergence as a measure cost function.

Let denote N (*m, δ*2) as a Gaussian distribution with mean m and standard deviation δ. With µ = N (m1, δ2 1
), ν =
N (m2, δ22
) We first want to show that by applying Data Processing Inequalities (Lemma 2.1 (Gerchinovitz et al., 2020)),

with h(X) = X − m1, and g(X) = X − m2, we can derive W1(µ, ν) = inf ρ∈Γ(µ,ν) E X,Y ∼ρ [Dfα (X, Y )]] ⩾ inf ρ∈Γ(µ,ν) E X,Y ∼ρ [Dfα (X − m1, Y − m1)] = W1(N (0, δ2 1), N (m2 − m1, δ2 2)), (9) and W1(µ, ν) = inf ρ∈Γ(µ,ν) E X,Y ∼ρ [Dfα(X, Y )]] ⩾ inf ρ∈Γ(µ,ν) E X,Y ∼ρ [Dfα(X − m2, Y − m2)] ⩾ inf ρ∈Γ(µ,ν) E X,Y ∼ρ [Dfα(m2 − X, m2 − Y )]( with the transform function f(X) = −X) = W1(N (m2 − m1, δ21), N (0, δ2 2)). (10) Now according to (7), the L 1-Wasserstein distance with α-divergence distance is defined as

$$W_{1}(\mu,\nu)=\left(\int_{0}^{1}D_{f_{\alpha}}(F_{\mu}^{-1}(t)||F_{\nu}^{-1}(t))dt\right).$$  of a Gaussian distribution (Soch, 2020) $F={\cal N}(\mu,\delta^{2})$ is 
We show that the quantile function of a Gaussian distribution (Soch, 2020) F = N (*µ, δ*2) is

$${}^{(9)}$$
$$(10)^{2}$$
$$(11)$$
$$F^{-1}(t)=\sqrt{2}\delta\mathrm{erf}^{-1}(2t-1)+\mu,$$
$$(12)$$

−1(t) = √2δerf−1(2t − 1) + µ, (12)
where erf−1(t) is the inverse of the function √
2 π R t 0 exp{−x 2}dx.

Therefore, the L
1-Wasserstein distance with α-divergence distance as the cost function between two Gaussian distributions µ = N (m1, δ2 1
), ν = N (m2, δ22
) can be measured as

$$W_{1}(\mu,\nu)=\bigg{(}\int_{0}^{1}D_{f_{\alpha}}(\sqrt{2}\delta_{1}\mathrm{erf}^{-1}(2t-1)+m_{1}||\sqrt{2}\delta_{2}\mathrm{erf}^{-1}(2t-1)+m_{2})dt\bigg{)}.$$  Applying the convexity properties of $\alpha$-divergence (Clichocki & Amari, 2010), and from (9),(10) we have 
W1(µ, ν) ⩾ 1 2  Z 1 0 Dfα( √2δ1erf−1(2t − 1)||√2δ2erf−1(2t − 1) + m2 − m1)dt + Z 1 0 Dfα( √2δ1erf−1(2t − 1) + m2 − m1||√2δ2erf−1(2t − 1))dt ⩾  Z 1 0 Dfα( √2δ1erf−1(2t − 1) + m2 − m1 2||√2δ2erf−1(2t − 1) + m2 − m1 2)  = W1(N ( m2 − m1 2, δ21), N ( m2 − m1 2, δ2 2)). Applying Data Processing Inequalities (Lemma 2.1 (Gerchinovitz et al., 2020)), with h(X) = X − m2−m1

2, we can derive
$$W_{1}(\mu,\nu)\geqslant W_{1}({\cal N}(0,\delta_{1}^{2}),{\cal N}(0,\delta_{2}^{2}))=\left(\int_{0}^{1}D_{f_{\alpha}}(\sqrt{2}\delta_{1}{\rm erf}^{-1}(2t-1)||\sqrt{2}\delta_{2}{\rm erf}^{-1}(2t-1))dt\right).$$  Consider the sequence $0=t_{1}\leq t_{1}\leq\ldots\leq t_{N}=1$, there exists $\delta_{i}\in[t_{1},t_{\alpha-1}]$ that 
$$W_{1}(\mu,\nu)\geqslant\sum_{i=0}^{i=N}(t_{i+1}-t_{i})D_{f_{\alpha}}(\sqrt{2}\delta_{1}\mathrm{erf}^{-1}(2\xi_{i}-1)||\sqrt{2}\delta_{2}\mathrm{erf}^{-1}(2\xi_{i}-1))$$ $$=\sum_{i=0}^{i=N}\Delta_{i}D_{f_{\alpha}}(\sqrt{2}\delta_{1}\mathrm{erf}^{-1}(2\xi_{i}-1)||\sqrt{2}\delta_{2}\mathrm{erf}^{-1}(2\xi_{i}-1)),$$  $\lambda$: Since $D_{f_{\alpha}}(P^{||}\,Q)=D_{f_{\alpha}}(P^{||}Q)$ does not exist. We can claim an 
with $\Delta_{i}=(t_{i+1}-t_{i})$. Since $D_{f_{n}}(cP||cQ)=D_{f_{n}}(P||Q)$ where $c$ is a constant. We can derive 
$$W_{1}(\mu,\nu)\geqslant\sum_{i=0}^{i=N}\Delta_{i}D_{f_{\alpha}}(\delta_{1}||\delta_{2})=D_{f_{\alpha}}(\delta_{1}||\delta_{2}).$$  In other words, the $\alpha$-function is a function of $\alpha$.  
$$(13)$$

Let us consider the sequences 0 = t0 ⩽ t1 ⩽ ... ⩽ tN = 1, there exists ξi ∈ [ti, ti+1] that We start with the first Proposition about the closed solutions of mean and variance of a Gaussian value function V(s) as V-posterior L
1-Wasserstein barycenter of all action value function distributions Q(*s, a*).

Proposition 1. Consider the V-posterior value function V(s) *as a Gaussian:* N (m(s), δ 2(s)). Let's define each Q(s, a) as the Q function child node of V(s). Each Q(s, a) is assumed as a Gaussian distributions Q(*s, a*) : N (m(s, a), δ(s, a)
2). If the value function V(s) *is defined as the Wasserstein barycenter of the Q function* Q(s, a) given the policy π¯*, we will have:*

$$\overline{m}(s)=\left(\mathbb{E}_{a\sim\overline{\pi}}[m(s,a)^{p}]\right)^{\frac{1}{p}}$$ $$\overline{\delta}(s)=\left(\mathbb{E}_{a\sim\overline{\pi}}[\delta(s,a)^{p}]\right)^{\frac{1}{p}},\tag{1}$$
$$(14)$$
$$(15)$$
$$(16)^{\frac{1}{2}}$$
$$(\overline{{{\mu}}}(s),\overline{{{\delta}}}(s))=\operatorname*{arg\,min}_{\mu,\delta}\left\{\mathbb{E}_{\pi}[W_{1}(\mathcal{V}(s)||\mathcal{Q}(s,a))]\right\}.$$
nEπ¯[W1(V(s)||Q(s, a))]o. (16)
$$\overline{{{\delta}}}(s)=\operatorname*{arg\,min}_{\delta(s)}\left\{\mathbb{E}_{\overline{{{\pi}}}}[D_{f_{\alpha}}(\delta(s)||\delta(s,a))]\right\}.$$
$$\frac{\nabla\mathbb{E}_{a\sim\bar{\pi}}[D_{f_{a}}(\delta(s)||\delta(s,a))]}{\nabla\delta(s)}=0.$$
$$\stackrel{\nabla}{=}$$

Since∇fα(x)
$${\frac{\nabla f_{\alpha}(x)}{\nabla x}}={\frac{\alpha(x^{\alpha-1}-1)}{\alpha(\alpha-1)}}={\frac{x^{\alpha-1}-1}{\alpha-1}}.$$
With Dfα
(x||y) = Py
yfα(
x
y
), we can have
$$(17)$$
$$(18)^{\frac{1}{2}}$$
$${\frac{\nabla D_{f_{n}}(x||y)}{\nabla x}}=\sum_{y}{\frac{({\frac{x}{y}})^{\alpha-1}-1}{\alpha-1}}.$$
$$(19)$$
$$(20)$$
$$(21)$$
$$(22)^{\frac{1}{2}}$$
$$\mathbb{E}_{\alpha\sim\mathbb{R}}\left[\frac{\left(\frac{\tilde{\delta}(s)}{\delta(s,a)}\right)^{\alpha-1}-1}{(\alpha-1)}\right]=0\Longrightarrow\mathbb{E}_{\alpha\sim\mathbb{R}}\left[\left(\frac{\tilde{\delta}(s)}{\delta(s,a)}\right)^{\alpha-1}-1\right]=0.$$ $=\alpha$ that leads to 
$$\overline{{{\delta}}}(s)=(\mathbb{E}_{a\sim\overline{{{\pi}}}}[\delta(s,a)^{p}])^{\frac{1}{p}}.$$
p]) 1p . (21)
$$W_{1}(\mu,\nu)=\operatorname*{inf}\{\mathbb{E}[D_{f_{\alpha}}(\mu||\nu)]\}.$$
(µ||ν)]}. (22)
$$\mathbb{E}[D_{f_{\alpha}}(\mu||\nu)]\geqslant D_{f_{\alpha}}(\mathbb{E}[\mu]||\mathbb{E}[\nu])=D_{f_{\alpha}}(m_{1}||m_{2}).$$
$$\mathbb{E}[D_{f_{\alpha}}(\mu||\nu)]\geqslant D_{f_{\alpha}}$$
E[Dfα(µ||ν)] ⩾ Dfα(E[µ]||E[ν]) = Dfα(m1||m2). (23)
$$\overline{{{m}}}(s)=\operatorname*{arg\,min}_{m(s)}\mathbb{E}_{a\sim\pi}[D_{f_{\alpha}}(m(s)||m(s,a))].$$
$$\overline{{{m}}}(s)=(\mathbb{E}_{a\sim\bar{\pi}}[m(s,a)^{p}])^{\frac{1}{p}},$$
p]) 1p , (25)
$\eqref{eq:walpha}$. 
$$(24)$$
$$(25)$$
with p = 1 − α. Proof. By the definition of the V-posterior value function, we have: We first compute the standard deviation δ(s). From (13), and (16), we want to find δ(s) that is the minimizer of we derive δ(s) is the solution of We can derive Now we can define p = 1 − α that leads to To compute µ¯(s). Let's revisit here again the definition of L
1−Wasserstein distance between two Gaussian distributions µ(m1, δ2 1), ν(m2, δ22).

According to Jensen's inequality(Perlman, 1974) we can derive Therefore, according to the definition of Wasserstein barycenter, the mean of a Gaussian V-posterior value function V(s) can be derived as Following the same steps as to compute δ(s), we can get with p = 1 − α that concludes the proof.

Next, we consider each node as an equally weighted Particle model and derive the following proposition.

Proposition 2. Let's assume the V-posterior value function V(s) *as a equally weighted Particle model:* xi(s) : i ∈
[1, M]. M is an integer and M ⩾ 1. Let's assume each Q function Q(s, a) has M particles xi(s, a), i ∈ [1, M]. If the value function V(s) is defined as the Wasserstein barycenter of the Q function Q(s, a) given the policy π¯, each particle
(xi(s), i ∈ [1, M]*) can be estimated as*

$$\overline{{{x_{i}}}}(s)=(\mathbb{E}_{a\sim\bar{\pi}}[x_{i}(s,a)^{p}])^{1/p},$$
p])1/p, (26)
with p = 1 − α.

Proof. We can compute the quantile function of µ and ν as

$$F_{\mu}^{-1}(t)=\sum_{i=1}^{M}x_{i}\mathbf{1}_{I_{i}}(t),F_{\nu}^{-1}(t)=\sum_{i=1}^{M}y_{i}\mathbf{1}_{I_{i}}(t).$$
$$(26)$$
$$(27)$$
$$(28)$$
$$(29)$$
$$(30)$$
$$(31)$$
$$(32)$$

Therefore from (11) we can get We can see that for each particle (xi(s), i ∈ [1, M]), we can derive

$$\overline{x_{i}}(s)=\operatorname*{arg\,min}_{x_{i}(s)}\mathbb{E}_{a\sim\overline{x}}[D_{f_{a}}(x_{i}(s)||x_{i}(s,a))]$$ $$\Longrightarrow\overline{x_{i}}(s)=(\mathbb{E}_{a\sim\overline{x}}[x_{i}(s,a)^{p}])^{1/p},$$

with p = 1 − α.

## D. Supporting Lemmas

We will make use of the following basic results.

Lemma 1. **(Minkowski's inequality)** Given p ⩾ 1, {xi, yi} ∈ R, i = 1, 2, ..., n*, then we have the following inequality*

$$\left(\sum_{i}(|x_{i}+y_{i}|)^{p}\right)^{\frac{1}{p}}\leqslant\left(\sum_{i}(|x_{i}|)^{p}\right)^{\frac{1}{p}}+\left(\sum_{i}(|y_{i}|)^{p}\right)^{\frac{1}{p}}.$$

Proof. This is a basic result.

Lemma 2. **(Markov's inequality)** If X is a nonnegative random variable and a > 0, then the probability that X is at least a is at most the expectation of X divided by a:

$$\mathbf{Pr}(X>a)\leqslant{\frac{\mathbb{E}[X]}{a}}.$$
a. (36)
$$W_{1}(\mu,\nu)=\left(\int_{0}^{1}D_{f_{\alpha}}(F_{\mu}^{-1}(t)||F_{\nu}^{-1}(t))dt\right)$$ $$=\sum_{i=1}^{M}\left(\int_{I_{i}}D_{f_{\alpha}}(F_{\mu}^{-1}(t)||F_{\nu}^{-1}(t))dt\right)$$ $$=\sum_{i=1}^{M}\left(\int_{I_{i}}D_{f_{\alpha}}(x_{i}||y_{i})dt\right)$$ $$=\sum_{i=1}^{M}D_{f_{\alpha}}(x_{i}||y_{i})\bigg{(}\int_{I_{i}}dt\bigg{)}$$ $$=\sum_{i=1}^{M}w_{i}D_{f_{\alpha}}(x_{i}||y_{i}).$$
$$(33)$$
$$(34)$$
$\square$
$$(35)$$
$$(36)^{\frac{1}{2}}$$

17

## E. Convergence Of Wasserstein Non-Stationary Multi-Armed Bandits

We note that in an MCTS tree, each node is considered a non-stationary multi-armed bandit where the average mean drifts due to the given action selection strategy. Therefore, we first study the convergence of Wasserstein non-stationary multiarmed bandits where the action selection is Thompson sampling, with the power mean backup operator at the root node. Detailed descriptions of the Wasserstein Non-stationary multi-armed bandits settings can be found in the main article in the Theoretical Analysis section. We briefly summarize the theoretical results below. Lemma 6 is about the upper bound on the expectation of the number of suboptimal arms playing, following the corresponding Theorem 4.2 in (Jin et al., 2022). Lemma 7 is about the bias of the expected value of the power mean backup operator, which follows the result as Theorem 1 in Stochastic-Power-UCT (Dam et al., 2024b). Theorem 2 deals with the polynomial concentration of the power mean backup operator around the optimal mean at the root node of the non-stationary Wasserstein problem for multi-armed bandits. This theorem plays an important role in deriving the polynomial convergence of the choice of the optimal action at the root node in the Wasserstein MCTS tree, described in the next section. Now, we will find an upper bound for the expectation of numbers of pulling a suboptimal arm. Let us define the event Ek,ε(t) = {θk(t) ⩽ µ
∗ − ε} for all k ∈ [K], ε > 0, θk(t) is sampled from N (Xk*, V /T*k(n)) at timestep t. Let us consider the decomposition

$$\mathbb{E}[T_{k}(n)]=1+\mathbb{E}\Big{[}\sum_{t=K+1}^{n}\mathbf{1}\{A_{t}=a_{k},E_{k,\varepsilon}(t)\}+\sum_{t=K+1}^{n}\mathbf{1}\{A_{t}=a_{k},E_{k,\varepsilon}^{c}(t)\}\Big{]}$$ $$=1+\underbrace{\mathbb{E}\Big{[}\sum_{t=K+1}^{n}\mathbf{1}\{A_{t}=a_{k},E_{k,\varepsilon}(t)\}\Big{]}}_{A_{t}}+\underbrace{\mathbb{E}\Big{[}\sum_{t=K+1}^{n}\mathbf{1}\{A_{t}=a_{k},E_{k,\varepsilon}^{c}(t)\}\Big{]}}_{B_{t}}.$$
(37)  $$\left(38\right)$$ .... 

$$(39)$$
$$(40)$$
$$(41)$$
. (38)
Here Ecis the complement of an event E, ε > 8p*V /n* is an arbitrary constant.

Bounding Term A: Let's define

$$\alpha_{s}=\operatorname*{sup}_{x\in[0,\mu^{*}-\varepsilon)}\Big\{\mathrm{KL}(\mu^{*}-\varepsilon-x,\mu^{*})\leqslant4\log({\frac{n}{s}})/s\Big\}.$$
)/so. (39)
Lemma 3. *(Lemma A.1 (Jin et al., 2022)) Let* M = ⌈16V log(nε2/V )/ε2⌉, and αs *be the same as defined in (39) then*

$$\mathbb{E}\Big{[}\sum_{t=K+1}^{n}\mathbf{1}\{A_{t}=a_{k},E_{k,\varepsilon}(t)\}\Big{]}\leqslant\sum_{s=1}^{M}\mathbb{E}\Big{[}\Big{(}\frac{1}{1-F_{s}^{*}(\mu^{*}-\varepsilon)}-1\Big{)}.\mathbf{1}\{\overline{X}_{s}^{*}\in(\mu^{*}-\varepsilon-\alpha_{s},1]\}\Big{]}+\odot\Big{(}\frac{V}{\varepsilon^{2}}\Big{)},$$

, (40)
where F
∗
s*is the CDF of Gaussian with mean* X
∗
s, X
∗
sis the average reward of the optimal arm after s *visitations.*
Lemma 4. *(Lemma A.2 (Jin et al., 2022)) Let* M = ⌈16V log(nε2/V )/ε2⌉*. Then*

$$\sum_{s=1}^{M}\mathbb{E}\overline{{{X}}}_{s}^{*}\left[\left(\frac{1}{1-F_{s}^{*}(\mu^{*}-\varepsilon)}\right).1\{\overline{{{X}}}_{s}^{*}\in(\mu^{*}-\varepsilon-\alpha_{s},1]\}\right]=\Theta\Big(\frac{V\log(n\varepsilon^{2}/V)}{\varepsilon^{2}}\Big).$$

Bounding Term B:
Lemma 5. *(Lemma C.1 (Jin et al., 2022)) Let* N = min{1 1−
KL(µk+*ρk,µ*∗−ε)
log(nε2/V )
, 2}. For any ρk, ε > 0 that satisfies ε+ρk < ∆i, then

$$\mathbb{E}\Big[\sum_{t=K+1}^{n}\mathbf{1}\{A_{t}=k,E_{k,\varepsilon}^{c}(t)\}\Big]\leqslant1+\frac{2V}{\rho_{k}^{2}}+\frac{V}{\varepsilon^{2}}+\frac{N\log(n\varepsilon^{2}/V)}{K L(\mu_{k}+\rho_{k},\mu^{*}-\varepsilon)}.$$

$$(42)$$

18 From Assumption 1, we derive the upper bound for the expectation of the number of plays of a suboptimal arm. Lemma 6. Consider Thompson Sampling strategy (using power mean estimator) applied to a non-stationary problem where the pay-off sequence satisfies Assumption 1. Fix ε ⩾ 0. Let Tk(n) denote the number of plays of arm k*. Then if* k is the index of a suboptimal arm, then each sub-optimal arm k *is played in expectation at most*

$$\mathbb{E}[T_{k}(n)]\leqslant\Theta\left(1+{\frac{V\log(n\Delta_{k}^{2}/V)}{\Delta_{k}^{2}}}\right).$$
$$(43)$$
$$(444)$$
$$(45)$$
$\square$
Proof. The proof of Lemma 6 closely follows Theorem 4.2((Jin et al., 2022)) by observing results from Lemma 3, 4, 5. From equation 38, putting all Lemma 3, 4, 5, we have

$$\mathbb{E}[T_{k}(n)]=\Theta\Big(1+\frac{V\log(n\varepsilon^{2}/V)}{(\Delta_{k}-\varepsilon-\rho_{k})^{2}}+\frac{V}{\rho_{k}^{2}}+\frac{V\log(n\varepsilon^{2}/V)}{\varepsilon^{2}}\Big).$$

Set ε = ρk = ∆k/4, we derive

$$\mathbb{E}[T_{k}(n)]\leqslant\Theta\Big(1+\frac{V\log(n\Delta_{k}^{2}/V)}{\Delta_{k}^{2}}\Big).$$

Lemma 7. Consider a non-stationary problem where the pay-off sequence satisfies Assumption 1. We consider a bandit algorithm that selects each arm as

$$a=\operatorname*{argmax}_{a_{i},i\in\{1...K\}}\{\theta_{i}\sim{\mathcal{N}}({\overline{{X}}}_{k,n},V/T_{k}(n))\}.$$

p ⩾ 1, ε0 > 0*, we have*

Let us define the power mean estimator Xn(p) as Xn(p) = PK
$=\left(\sum_{a=1}^K\frac{T_a(n)}{n}\overline{X}_{a,T_a(n)}^p\right)^{\frac{1}{p}}$, and $\delta_{\star,n}=\mu_{\star}-\mu_{\star,n}$ For any $n$, $p$ is a constant. 
$$\left|\mathbb{E}[\overline{{{X}}}_{n}(p)]-\mu_{\star}\right|\leqslant|\delta_{\star,n}|+\frac{R}{n}\sum_{a=1,a\neq a_{\star}}^{K}\Theta\Big(1+\frac{V\log(n\Delta_{k}^{2}/V)}{\Delta_{k}^{2}}\Big)$$
Proof. We observe that

$$\left|\overline{{{X}}}_{n}(p)-\mu_{\star}\right|\leqslant\left|\overline{{{X}}}_{n}(p)-\mu_{\star,n}\right|+\left|\mu_{\star}-\mu_{\star,n}\right|=\left|\overline{{{X}}}_{n}(p)-\mu_{\star,n}\right|+\left|\delta_{\star,n}\right|$$
 + |δ⋆,n| (47)
Furthermore,
$$\overline{{{X}}}_{a,T_{a}(n)}\leqslant\mu_{a,n}+\left|\overline{{{X}}}_{a,T_{a}(n)}-\mu_{a,n}\right|.$$
. (48)
Since µ⋆,n = maxa∈[K]{µa,n}, we have
$$\overline{X}_{n}(p)-\mu_{+,n}=\overline{X}_{n}(p)-\sum_{n=1}^{K}T_{n}(n)\mu_{+,n}\leqslant\left(\sum_{n=1}^{K}\frac{T_{n}(n)}{n}\left(\overline{X}_{a,T_{n}(n)}\right)^{p}\right)^{\frac{1}{p}}-\left(\sum_{n=1}^{K}\frac{T_{n}(n)}{n}\left(\mu_{a,n}\right)^{p}\right)^{\frac{1}{p}}$$ $$=\frac{\left(\sum_{n=1}^{K}T_{a}(n)\left(\overline{X}_{a,T_{n}(n)}\right)^{p}\right)^{\frac{1}{p}}-\left(\sum_{n=1}^{K}T_{a}(n)\left(\mu_{a,n}\right)^{p}\right)^{\frac{1}{p}}}{n^{\frac{1}{p}}}$$
$$(46)$$
$$(47)^{\frac{1}{2}}$$
$$(48)$$
(49)  $$\begin{array}{l}\small\mathbf{(50)^{}}\end{array}$$ . 
Applying Minkowski's inequality from Lemma 1, and the result of equation 48, we have

$$\overline{X}_{n}(p)-\mu_{*,n}\leqslant\frac{\left(\sum_{\alpha=1}^{K}T_{\alpha}(n)\left(\mu_{\alpha}+\left|\overline{X}_{\alpha,T_{\alpha}(n)}-\mu_{\alpha,n}\right|\right)^{p}\right)^{\frac{1}{p}}-\left(\sum_{\alpha=1}^{K}T_{\alpha}(n)\left(\mu_{\alpha,n}\right)^{p}\right)^{\frac{1}{p}}}{n^{\frac{1}{p}}}$$ $$\leqslant\frac{\left(\sum_{\alpha=1}^{K}T_{\alpha}(n)\left(\left|\overline{X}_{\alpha,T_{\alpha}(n)}-\mu_{\alpha,n}\right|\right)^{p}\right)^{\frac{1}{p}}}{n^{\frac{1}{p}}}$$
$$(51)$$

$$(52)$$

On the other hand,

nd hand, $$\begin{split}\mu_{*,n}-\overline{X}_n(p)&=\frac{n\mu_{*,n}-n\overline{X}_n(p)}{n}=\frac{n\mu_{*,n}-(\sum_{n=1}^K T_a(n)\mu_{a,n})+\sum_{n=1}^K T_a(n)\mu_{a,n}-n\overline{X}_n(p)}{n}\\ &=\frac{\sum_{n=1,a\neq a_n}^{K}T_a(n)\left[\mu_{*,n}-\mu_{*,n}\right]+\sum_{n=1}^K T_a(n)\mu_{a,n}-n\overline{X}_n(p)}{n}\\ &\leqslant R\sum_{n=1,a\neq a_n}^K\frac{T_a(n)}{n}+\sum_{n=1}^K\frac{T_a(n)}{n}\mu_{a,n}-\overline{X}_n(p)\end{split}$$
Because power mean is an increasing function of p, so that

$$\sum_{a=1}^{K}\frac{T_{a}(n)}{n}\mu_{a,n}\leqslant\left(\sum_{a=1}^{K}\frac{T_{a}(n)}{n}\left(\mu_{a,n}\right)^{p}\right)^{1/p}.$$

Furthermore, we observe that
$$\mu_{a,n}\leqslant\overline{{{X}}}_{a,T_{a}(n)}+\left|\overline{{{X}}}_{a,T_{a}(n)}-\mu_{a,n}\right|.$$
So that, from equation 55 we have
a=1 Ta(n) n(µa,n) p !1/p − Xn(p) (56) µ⋆,n − Xn(p) ⩽ RX K a=1,a̸=a∗ Ta(n) n+  X K ⩽ RX K a=1,a̸=a∗ Ta(n) n(57) + PK a=1 Ta(n)Xa,Ta(n) +Xa,Ta(n) − µa,n p 1p− PK a=1 Ta(n)Xa,Ta(n)p 1p n 1 p a=1,a̸=a∗ Ta(n) n+ PK a=1 Ta(n)Xa,Ta(n) − µa,np 1p n 1 p (Minkovski's inequality) ⩽ RX K a=1,a̸=a∗ Ta(n) n+ PK a=1 Ta(n)Xa,Ta(n) − µa,n  n 1 p (Properties of L pnorm) ⩽ RX K a=1,a̸=a∗ Ta(n) n+ PK a=1  PTa(n) t Xa,t − Ta(n)µa,n   = RX K n 1 p
$$(S6)$$
$$(S7)$$
(58)  $$\begin{array}{l}\small\mathbf{(59)^{}}\end{array}$$ . 
(60)  $$\begin{array}{l}\small\mathbf{(61)}\end{array}$$ . 
Therefore

$$\mathbb{E}[\overline{X}_{n}(p)-\mu_{*,n}]|\leqslant R\sum_{a=1,n\neq a_{*}}^{K}\frac{\mathbb{E}[T_{a}(n)]}{n}+\frac{\mathbb{E}\left[\left(\left|\sum_{a=1}^{K}\sum_{t}^{T_{a}(n)}X_{a,t}-T_{a}(n)\mu_{a,n}\right|\right)\right]}{n^{\frac{1}{p}}}$$ $$=R\sum_{a=1,n\neq a_{*}}^{K}\frac{\mathbb{E}[T_{a}(n)]}{n}$$

(62)  $\binom{63}{2}$  . 
Please note that because we study non-stationary bandits, E[Pn t Xa,t] = nµa,n, therefore,

$$\frac{\mathbb{E}\left[\left(\left|\sum_{a=1}^{K}\sum_{t}^{T_{a}(n)}X_{a,t}-T_{a}(n)\mu_{a,n}\right|\right)\right]}{n^{\frac{1}{p}}}=0$$