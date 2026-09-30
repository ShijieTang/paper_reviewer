# High-Dimensional Prediction for Sequential Decision Making

Georgy Noarov <sup>1</sup> Ramya Ramalingam <sup>1</sup> Aaron Roth <sup>1</sup> Stephan Xie <sup>2</sup>

## Abstract

We give an efficient algorithm for producing multidimensional forecasts in an online adversarial environment that have low bias subject to any polynomial number of conditioning events, that can depend both on external context and on our predictions themselves. We demonstrate the use of this algorithm with several applications. We show how to make predictions that can be transparently consumed by any polynomial number of downstream decision makers with different utility functions, guaranteeing them diminishing swap regret at optimal rates. We also give the first efficient algorithms for guaranteeing diminishing conditional regret in online combinatorial optimization problems for an arbitrary polynomial number of conditioning events — i.e. on an arbitrary number of intersecting subsequences determined both by context and our own predictions. Finally, we give the first efficient algorithm for online multicalibration with O(T 2/3 ) rates in the ECE metric.

## 1. Introduction

Decision making in sequential settings is a challenging problem from both the theoretical and the applied perspective. Tackling the nonstationarity inherent to sequential interactive decision making settings is the subject of the online learning, reinforcement learning, and bandits literatures (see e.g. [Slivkins et al.](#page-11-0) [\(2019\)](#page-11-0)).

The dominant paradigm is *regret minimization*, which places emphasis on optimizing various notions of *regret* for decision-making agents, thus ensuring that these agents' cumulative rewards stay ahead of nontrivial benchmark classes of strategies. The most basic and tractable notion of regret, called *external* regret (with the qualifier "external" usually

omitted), compares the utility of an agent's play to that of the best-in-hindsight fixed strategy. It is a *marginal* guarantee, in the sense that it forces the agent to do as well as the benchmark cumulatively over all rounds t = 1 . . . T, but does not imply locally optimal performance on any subset of rounds. Thus, various extensions to more challenging benchmark classes have been developed, giving rise to notions such as swap regret, adaptive regret, and beyond.

Typically, regret minimization algorithms directly optimize the agent's loss or reward function. This works well in single-agent settings and for simple regret guarantees. However, this direct approach does not naturally address sequential environments which (a) involve multiple decisionmaking agents with different utilities/rewards; or (b) demand performance guarantees *conditionally* on the agents' own actions. Motivated by these challenges, in this paper we take a different route and shift from direct reward optimization to a more subtle but flexible approach.

#### Recipe

1. Distill the decision-making-relevant aspects of the sequential environment into a sequence of "sufficient statistics": vector-valued *states* of the environment, such that all agents could easily identify their reward-maximizing actions *if* they perfectly knew the next state.

2. Devise an efficient online adversarial algorithm that predicts the upcoming state of the environment with such granular accuracy guarantees that the agents can *treat our predictions as the true states* for the purposes of picking reward-maximizing actions.

Step 1 of this recipe will naturally always require insight into the specific setting at hand, and identifying appropriate sufficient statistics can often be an art. However — as we will show in this paper — Step 2, which may appear just as hard to execute, turns out to have a "silver-bullet" solution that applies to a wide variety of complex decision-making settings: our *event-unbiased prediction* framework.

## 1.1. Our Contributions

<sup>1</sup>Department of Computer and Information Science, University of Pennsylvania <sup>2</sup>Machine Learning Department, Carnegie Mellon University. Correspondence to: Georgy Noarov <gnoarov@seas.upenn.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

tities in the online adversarial setting. It accommodates arbitrary finite collections of (unweighted or weighted) *conditioning events*, which may depend on external contexts or on the predictions themselves, and guarantees optimally converging bias conditional on each of these events. We introduce the framework, and formally derive the algorithm and its guarantees, in Section [2.](#page-2-0)

We then show how our framework implements the above recipe in complex environments shared by one or more decision-making agents. In this context, as our Recipe prescribes, we use it in a "Predict-then-Act" manner, which allows us to bypass direct optimization of policy rewards for any given agent, instead focusing on forecasting the evolving *state* of the environment, which the agents' rewards ultimately depend on. The main advantages this offers are:

(1) Strong *conditional* regret guarantees: By leveraging our predictions, the agents can minimize their regret not just marginally over the whole sequence of the interaction, but also on arbitrary context- and action- defined subsequences of the rounds. This is enabled by our framework's ability to de-bias its state predictions over arbitrary such subsequences of rounds, thus endowing the agents' decisions with corresponding regret guarantees on these subsequences.

(2) *Coordination* of multiple agents: Instead of letting all agents take the burden of running their respective regret minimization algorithms locally, we are able to instead provide a single public-knowledge state prediction, which all agents can then use to inform their actions in a straightforward way: They can just *best-respond* to the announced state.

We illustrate the power of our framework on the following concrete examples:

Online Combinatorial Optimization We consider a general *online combinatorial optimization* setting, which models a sequential interaction of multiple agents with combinatorially large action spaces corresponding to structured subsets of some set of d base actions, and which encompasses a variety of classical problems such as online routing. We show how to endow all agents with conditional regret guarantees, letting the agents incur only sublinear regret on any given finite collection of subsequences defined by external contexts and even by their *own actions*.

An example instantiation is the online routing problem in which multiple agents are trying to get from home to work as fast as possible every day. We can issue *trustworthy* daily road congestion forecasts, such that each agent who always selects the fastest route according to our forecast is guaranteed no regret conditional both on salient covariate information (such as the weather) and on their own choice of route (e.g. downtown route; interstate route; route visiting their favorite coffee shop). Thus, each agent will then be happy with their average travel time not just overall, but also, e.g., over those days when it rained and they went to get coffee on their way to work; or over those days when their local football team was playing.

Our algorithm has an efficient dependence on the dimension d of the combinatorial problem (e.g. d is the number of edges in online routing): it runs using only poly(d) calls to the offline oracle for the problem (e.g. a shortest-path algorithm for routing) and has <sup>O</sup>e(d) regret dependence. Prior techniques for giving conditional regret guarantees by direct regret minimization (e.g. the algorithm of [Blum & Mansour](#page-8-0) [\(2007\)](#page-8-0)) have running time scaling with the number of actions (e.g. paths in online routing), which can be exponential in d in online combinatorial optimization. See Section [4.](#page-6-0)

Swap Regret for Multiple Agents In the classical experts setting but with multiple regret-minimizing agents, we show how to use our framework to issue a single coordinating prediction at all rounds that will guarantee *swap regret* at optimal rates to all agents simultaneously. See Section [3.](#page-5-0)

Online Multicalibration In Section [2.2](#page-4-0) we show that our algorithm, when appropriately instantiated, gives the first efficient O(T 2/3 ) online multicalibration algorithm. Multicalibration [\(Hebert-Johnson et al.](#page-10-0) ´ , [2018\)](#page-10-0) is a strengthening of the classical statistical concept of calibration [\(Dawid,](#page-9-0) [1985\)](#page-9-0), and promises calibrated predictions on rich collections of contextually defined groups in the data. The best-known efficient algorithm for (vanilla) online calibration is due to [Abernethy et al.](#page-8-1) [\(2011\)](#page-8-1), and we match its rate in the more challenging multicalibration setting. The best-known multicalibration bound achieved in the ECE metric was O(T 3/4 ) [\(Gupta et al.,](#page-9-1) [2022\)](#page-9-1).

## 1.2. Follow-up Work and Impact

Since the appearance of the preprint of this paper, our methodology has been employed in several subsequent works, showing the broad utility of our framework.

[Collina et al.](#page-8-2) [\(2024b\)](#page-8-2) apply our algorithms in a repeated principal-agent setting defined by [Camara et al.](#page-8-3) [\(2020\)](#page-8-3) to obtain exponentially improved bounds. Briefly, [Camara](#page-8-3) [et al.](#page-8-3) [\(2020\)](#page-8-3) gave a mechanism that replaced the standard "common prior" assumptions that underlie principal-agent models with calibrated forecasts of an underlying state, and is applicable in adversarial settings. [Camara et al.](#page-8-3) [\(2020\)](#page-8-3) use the traditional notion of calibration, and as a result inherit exponential computational and statistical dependencies on the cardinality of the state space. [Collina et al.](#page-8-2) [\(2024b\)](#page-8-2) show how to apply our techniques to recover the same results (under weaker assumptions) with an exponentially improved dependence on the cardinality of the state space.

[Roth & Shi](#page-11-1) [\(2024\)](#page-11-1) apply our algorithms to produce forecasts that guarantee *all* downstream decision makers O( √ T) *swap regret* without the need to know their utilities in advance, improving on the simultaneous no-external regret guarantees of [Kleinberg et al.](#page-10-1) [\(2023\)](#page-10-1).

[Hu & Wu](#page-10-2) [\(2024\)](#page-10-2), using our algorithm, further remove any dependence on the cardinality of the decision maker's action space in the 1-dimensional setting.

[Collina et al.](#page-8-4) [\(2024a\)](#page-8-4) use our algorithm as part of their construction giving computationally tractable "agreement" protocols generalizing Aumann's agreement theorem.

Our original extended preprint contains another application of our algorithm, to decision conditional "score-free" conformal prediction. By appropriately de-biasing the scores of any online multiclass predictor, we can make them look like correct class probability vectors to downstream prediction set algorithms, letting prediction sets with valid conditional coverage guarantees be simply "read off" from the multiclass probability vectors.

#### 1.3. Related Work

Our framework extends an array of recent prior works on decision-focused prediction. [Zhao et al.](#page-11-2) [\(2021\)](#page-11-2) introduce decision calibration: a calibration-based framework for the offline (batch) setting. Decision calibration is less expressive than our event-unbiasedness notion: e.g., unlike our framework, it does not imply swap regret guarantees. [Gopalan](#page-9-2) [et al.](#page-9-2) [\(2022a\)](#page-9-2) introduce omniprediction: an approach for making predictions that simultaneously can be used to optimize multiple downstream loss functions. Omniprediction is related to our Predict-then-Act approach to optimizing utilities for multiple agents: the omni-predictions need to be post-processed downstream to optimize each loss, but can be treated as if they are real probabilities in that downstream optimization. [Dwork et al.](#page-9-3) [\(2021\)](#page-9-3) study outcome indistinguishability: a complexity-theoretic perspective on making predictions that appear indistinguishable from the ground truth to a rich class of distinguishers. These works have generated considerable follow-up research, which we discuss in Appendix [A.](#page-12-0) We extend these insights into a broadly applicable, efficient *online adversarial* framework with optimal-rate guarantees.

No-Regret Guarantees in Online Learning No-regret learning has been studied at least since [Hannan](#page-10-3) [\(1957\)](#page-10-3); see [Hazan](#page-10-4) [\(2016\)](#page-10-4) for a modern treatment of this literature. [Kalai & Vempala](#page-10-5) [\(2005\)](#page-10-5) gave efficient no regret algorithms in online linear and combinatorial optimization problems. Internal regret, which corresponds to regret on the subsequences defined by the play of each action, was first defined by [Foster & Vohra](#page-9-4) [\(1999\)](#page-9-4), who also showed it could be obtained by best responding to calibrated forecasts.

The seminal contribution of [Foster & Vohra](#page-9-4) [\(1999\)](#page-9-4) has led to a long list of works exploring the interplay of no-regret and online calibration algorithms, discussed in more detail in the Appendix. As one important precursor to our work, [Haghtalab et al.](#page-10-6) [\(2023a\)](#page-10-6) develop a general online multiobjective learning framework based on a game between no-regret and best-response algorithms, with the focus on deriving improved multicalibration guarantees in the online and batch settings. Their reduction to no-regret learning allows them, in particular, to obtain small-loss group-calibration bounds, mirroring our small-loss event-conditional bias bounds.

[Lehrer](#page-10-7) [\(2003\)](#page-10-7) defined a notion of "wide-range regret" which is equivalent to conditional regret: that a player should have no regret not just overall on the whole sequence of rounds, but also conditional on various events — subsequences that can be defined both as a function of time ("time selection functions") and as a function of the actions of the learner. [Blum & Mansour](#page-8-0) [\(2007\)](#page-8-0) gave algorithms for obtaining this kind of conditional regret guarantees (including, notably internal (or "swap") regret as a special case). The algorithm of [Blum & Mansour](#page-8-0) [\(2007\)](#page-8-0) is efficient when the action space is polynomially sized: it requires computing eigenvectors of a square matrix of dimension equal to the number of actions in the game. Motivated by fairness concerns, [Blum](#page-8-5) [& Lykouris](#page-8-5) [\(2020\)](#page-8-5) give an algorithm for obtaining diminishing "groupwise" regret, which is equivalent to regret with respect to a collection of time selection functions. These results do not accommodate events that can depend on the actions of the learner, which are crucial for our applications.

## 2. General Framework and Algorithm

Unbiased Prediction Setting Let X be a *context space* which can be arbitrary. Let the *state space* S be any convex and compact subset of R d and assume without loss of generality that maxs∈S ∥s∥<sup>∞</sup> ≤ 1. Any element s ∈ S is called a *state*. The space of distributions over S is denoted ∆S.

In this section, we consider the task of online adversarial contextual prediction of the states over t ∈ [T] := {1, . . . , T} time steps. The learner sequentially observes contexts (xt)t∈[T] ∈ (X ) T , and makes randomized state predictions (¯st)t∈[T] ∈ (∆S) T . The *adversary* sequentially responds by generating the true states (st)t∈[T] ∈ (S) T .

The learner aims to make predictions *unbiased* conditional on a given collection E = (E<sup>j</sup> )j∈[n] of n ≥ 1 *events*.

Definition 2.1 (Event; Event-Conditional Bias). An *event* is a mapping E : X × S → [0, 1]; the event's value in round t is E(xt, sˆt). If the range of E is {0, 1} then we call E a *binary* event.

The cumulative E*-conditional bias* in coordinate i ∈ [d] of the state predictions after T rounds is defined as:

$$\text{Bias}_T(E, i) := \mathbb{E}_{\hat{s}_t \sim \hat{s}_t \forall t} \left[ \sum_{t=1}^T E(x_t, \hat{s}_t) \cdot (\hat{s}_{t,i} - s_{t,i}) \right].$$

The general protocol is as follows. In rounds t = 1 . . . T:

- 1. The learner observes context x<sup>t</sup> ∈ X , and receives event functions E(xt, ·) : S → [0, 1] for E ∈ E.
- 2. The learner makes *state prediction* s¯ <sup>t</sup> ∈ ∆S.
- 3. The adversary sees s¯ t and generates *true state* s <sup>t</sup> ∈ S.
- 4. The *(realized) prediction* sˆ <sup>t</sup> ∈ S is sampled: sˆ <sup>t</sup> ∼ s¯ t .

Objective: The learner's goal is to make predictions that are *unbiased* in all coordinates i ∈ [d] conditional on all events E ∈ E. In fact, we define our desideratum by requiring the bias rate conditional on every event to diminish as a function of the event's frequency (rather than as a function of the time horizon T); such strengthened bounds are referred to as *small-loss* in online learning.

Definition 2.2 (Unbiased Prediction). Let n<sup>T</sup> (E) = Esˆt∼s¯t∀<sup>t</sup> hP<sup>T</sup> <sup>t</sup>=1 (<sup>E</sup> (xt, <sup>s</sup>ˆt))<sup>2</sup> i denote the *incidence*[<sup>1</sup>](#page-3-0) of event E. Then we call the learner's predictions *unbiased conditional on event collection* E if for all events E ∈ E,

$$\max_{i \in [d]} \text{Bias}_T(E, i) = O\left(\log(d |\mathcal{E}| T) + \sqrt{n_T(E) \log(d |\mathcal{E}| T)}\right)$$

. The General Unbiased Prediction Algorithm We now apply the OLO tools described above to obtain an efficient unbiased prediction algorithm that achieves the guarantee of Definition [2.2](#page-3-1) for any finite event collection E, state space S and feature space X .

We further denote Bias<sup>T</sup> (E) := maxi∈[d] Bias<sup>T</sup> (E, i).

### 2.1. General Algorithm with Bounds

OLO Primitives We now develop a general algorithm that achieves the above bias bounds for any given finite event collection. It will rely on online linear optimization (OLO) methods. We briefly review the OLO protocol over any d ′ dimensional convex domain C ⊆ R d . In rounds t = 1 . . . T, an OLO algorithm AOLO plays some c<sup>t</sup> ∈ C, the adversary observes that and generates a loss vector ℓ<sup>t</sup> ∈ <sup>R</sup> d , and AOLO observes ℓ<sup>t</sup> and suffers loss ⟨ℓt, ct⟩ in that round. The overall performance of AOLO is measured via *OLO regret* to the best point in C that could have been played. Letting the regret to any admissible point be defined as Reg<sup>T</sup> (AOLO, c) := P<sup>T</sup> <sup>t</sup>=1⟨ℓt, c<sup>t</sup> − c⟩, the OLO regret of AOLO is defined as:

$$\text{Reg}_T(\mathcal{A}_{\text{OLO}}) := \max_{c \in \mathcal{C}} \text{Reg}_T(\mathcal{A}_{\text{OLO}}, c),$$

Many OLO algorithms AOLO achieve the classic minimax regret bound Reg<sup>T</sup> (AOLO) = O( √ T) for all convex compact domains C and bounded losses; the simplest one is online gradient descent (OGD) of [Zinkevich](#page-11-3) [\(2003\)](#page-11-3).

However, for particular domains C, algorithms with even stronger regret bounds have been developed. We will use

one such method called MsMwC (Multiscale Multiplicative Weights with Correction) due to [Chen et al.](#page-8-6) [\(2021\)](#page-8-6) whose domain is the d ′ -dimensional simplex: C = ∆d′ . This special setting is also called the *experts setting*, as each of the vertices (ei)i∈[d′ ] of the simplex (e<sup>i</sup> ∈ <sup>R</sup> d denoting the ith standard basis vector) can be viewed as an expert. Rather than only promising O( √ T) regret to the best expert, MsMwC obtains small-loss bounds simultaneously to each expert, which scale with the losses of the expert:

Theorem 2.3 (Theorem 2 of [Chen et al.](#page-8-6) [\(2021\)](#page-8-6)). *There exists an experts OLO algorithm* AMsMwC *with per-round time complexity* poly(d ′ ) *for* d ′ *experts, whose chosen points* w<sup>t</sup> ∈ ∆′ d , t ∈ [T]*, achieve the following regret bound* to every expert e<sup>i</sup> ∈ [d ′ ] *provided all losses* ℓ<sup>t</sup> ∈ [−1, 1]<sup>d</sup> *:*

$$\text{Reg}_T(\mathcal{A}_{\text{MsMwC}}, e_i) = O\left(\log(d'T) + \sqrt{\log(d'T) \cdot \sum_{t=1}^T \ell_{t,i}^2}\right)$$

For notational convenience, we will represent any event collection E = (E<sup>j</sup> )j∈[n] as a single vector-valued *event function* E⃗ : X × S → [0, 1]<sup>n</sup>. Definition [2.2](#page-3-1) essentially requires us to learn to make randomized state predictions (ˆst)t∈[T] to optimize the quantity: Ψ (ˆst) T 1 ,(st) T 1 := maxi∈[d],j∈[n] P<sup>T</sup> t=1E⃗ <sup>j</sup> (xt, sˆt)·(ˆst,i−st,i) . However, this objective has a complex, and generally nonconvex and nondifferentiable, dependence on the predictions sˆt, so directly optimizing it appears out of reach. Yet, we will now show how to achieve this via a two-layer algorithmic technique: first, a reduction to a surrogate minimax objective, followed by a "simulated play" solution of that minimax problem. For both layers, we will use OLO algorithms as subroutines.

First Step: We identify weights w<sup>t</sup> ∈ ∆2dn, t ∈ [T], in an online fashion such that the following surrogate objective u = P<sup>T</sup> <sup>t</sup>=1 u<sup>t</sup> closely approximates Ψ:

$$\sum_{t=1}^T \underbrace{\sum_{\substack{i \in [d], j \in [n], \\ \sigma = \pm 1}} w_{t, (\sigma, i, j)} \cdot \sigma \cdot \vec{E}_j(x_t, \hat{s}_t) \cdot (\hat{s}_{t, i} - s_{t, i})}_{:=u_t(\hat{s}_t, s_t)}.$$

Second Step: While the surrogate function P t u<sup>t</sup> usefully separates the original objective across rounds t ∈ [T], each ut(ˆst, st) still depends on sˆ<sup>t</sup> through the event mappings, which need not be convex or differentiable. However, it is linear in the adversary's choice of st, and this can be exploited due to the following observation: If the adversary committed to s<sup>t</sup> first, the learner could achieve value 0 in the zero-sum game max<sup>s</sup><sup>t</sup> minsˆ<sup>t</sup> ut(ˆst, st) by simply *copy-*

<sup>1</sup>Note: n<sup>T</sup> (E) is at most the expected count of E's occurrences (i.e., rounds where E(xt, sˆt) = 1), with equality for binary E.

*ing* the adversary, i.e., with sˆ<sup>t</sup> = st. Therefore, *simulating* the *reverse playthrough* of this game, with the adversary going first and the learner copying, can give us a randomized saddle-point strategy s¯<sup>t</sup> for the learner: namely, the empirical distribution of the learner's simulated plays. This will suffice so long as the adversary plays to optimize the variable s<sup>t</sup> using *any* no-regret OLO algorithm. Therefore, by simulating sufficiently many rounds of this "no-regret adversary vs. copycat learner" game, we can get as close as we want to the value of the game.

Now we are ready to present our general Algorithm [1.](#page-4-1) For the first step, it instantiates the [Chen et al.](#page-8-6) [\(2021\)](#page-8-6) MsMwC algorithm for 2dn experts corresponding to signs σ = ±1, coordinates i ∈ [d] and events j ∈ [n], to enable bias bounds that depend on each event's incidence count. For the second step, it uses any no-regret OLO algorithm A<sup>S</sup> that can optimize over the state space S; this could be OGD or any other general-purpose O( √ T)-regret algorithm.

Algorithm 1 Unbiased Prediction

Initialize sˆ<sup>0</sup> = 0 d , s<sup>0</sup> = 0 d , E⃗ <sup>0</sup> (·) ≡ 0 n , and AMsMwC. for t = 1 . . . T do

Get context x<sup>t</sup> ∈ X , and define E⃗ <sup>t</sup> (·) := E⃗ (xt, ·).

Get new weights w<sup>t</sup> by updating AMsMwC with losses:

ℓ outer <sup>t</sup>−<sup>1</sup> ← σ · E⃗ <sup>t</sup>−<sup>1</sup> j (ˆst−1) · (st−1,i − sˆt−1,i) 

σ,i,j Initialize a new instance of A<sup>S</sup> and any s tent <sup>0</sup> ∈ S.

for τ = 1 . . . t<sup>2</sup> do

Get simulated prediction s tent <sup>τ</sup> by updating A<sup>S</sup> with:

$$\ell_{\tau-1}^{\text{inner}} \leftarrow \left( \sum_{\sigma=\pm 1} \sigma \sum_{j \in [n]} w_{t,(\sigma,i,j)} \cdot \vec{E}_j^t \left( s_{\tau-1}^{\text{tent}} \right) \right)_{i \in [d]}$$

end for

Set s¯<sup>t</sup> ← Unif {s tent 0 , stent 1 , . . . , stent t <sup>2</sup> } .

Predict sˆ<sup>t</sup> ∼ s¯t.

Observe true state st.

end for

Theorem 2.4 (Bias of Algorithm [1\)](#page-4-1). *For any time horizon* T*, and instantiated with any* O( √ T)*-regret OLO method* A<sup>S</sup> *over domain* S*, Algorithm [1](#page-4-1) produces (randomized) predictions* (¯st)t∈[T] *whose realizations* (ˆst)t∈[T] *achieve the desired bias bounds for all* i ∈ [d] *and* E<sup>j</sup> , j ∈ [n]*:*

$$\text{Bias}_T(E, i) \leq O \left( \log(d|\mathcal{E}|T) + \sqrt{n_T(E) \log(d|\mathcal{E}|T)} \right).$$

*Proof.* Step 1: We instantiate AMsMwC for 2dn experts, corresponding to signs σ = ±1, coordinates i ∈ [d] and events j ∈ [n]. Let the weights of MsMwC be (wt)t≥1, and denote our loss vectors for MsMwC by (ℓ outer t )t≥1, as defined in Algorithm [1.](#page-4-1) Denote experts' basis vectors by eσ,i,j . For every σ <sup>∗</sup> = ±1, i<sup>∗</sup> ∈ [d], j<sup>∗</sup> ∈ [n]:

$$\begin{aligned} \text{Reg}_T(\mathcal{A}_{\text{MsMwC}}, e_{\sigma^*, i^*, j^*}) &= \sum_{t \in [T]} \langle \ell_t^{\text{outer}}, w_t - e_{\sigma^*, i^*, j^*} \rangle \\ &= \sum_{T, \sigma, i, j} w_{t, (\sigma, i, j)} \cdot \sigma \cdot \vec{E}_j^t(\hat{s}_t) \cdot (s_{t, i} - \hat{s}_{t, i}) \\ &\quad + \sigma^* \sum_{t=1}^T \vec{E}_{j^*}^t(\hat{s}_t) \cdot (\hat{s}_{t, i^*} - s_{t, i^*}) \end{aligned}$$

Rearranging and taking a max over σ ∗ , we get for all i ∗ , j<sup>∗</sup> :

$$\begin{aligned} & \left| \sum_{t=1}^T \vec{E}_{j^*}^t (\hat{s}_t) \cdot (\hat{s}_{t,i^*} - s_{t,i^*}) \right| \\ & \leq \max_{\sigma^* \in \pm 1} \text{Reg}_T(\mathcal{A}_{\text{MsMwC}}, e_{\sigma^*,i^*,j^*}) \\ & + \sum_{T, \sigma, i, j} w_{t,(\sigma,i,j)} \cdot \sigma \cdot \vec{E}_j^t(\hat{s}_t) \cdot (\hat{s}_{t,i} - s_{t,i}) \\ & = O\left(\log(dnT) + \sqrt{\log(dnT)n_T (\vec{E}_j^*)}\right) + \sum_{t=1}^T u_t(\hat{s}_t, s_t), \end{aligned}$$

where u<sup>t</sup> is as defined above, and the regret bound follows from Theorem [2.3](#page-3-2) since the total squared loss of each expert (σ, i, j) is: P<sup>T</sup> <sup>t</sup>=1 ℓ outer t,(σ,i,j) 2 = O P<sup>T</sup> t=1(E⃗ <sup>t</sup> j (xt, sˆt))<sup>2</sup> = O(n<sup>T</sup> (E⃗ <sup>j</sup> )).

Step 2: By definition of regret for A<sup>S</sup> , we have for any t:

$$\begin{aligned} \text{Reg}_{t^2}(\mathcal{A}_S) &= \max_{s \in \mathcal{S}} \sum_{\tau \in [t^2]} \langle \ell_\tau^{\text{inner}}, s_\tau^{\text{tent}} - s \rangle \\ &= \max_{s \in \mathcal{S}} \sum_{\tau \in [t^2]} \sum_{\sigma, i, j} w_{t, (\sigma, i, j)} \cdot \sigma \cdot \vec{E}_j^t(s_\tau^{\text{tent}}) \cdot (s_{\tau, i}^{\text{tent}} - s_i) \\ &= \max_{s \in \mathcal{S}} \sum_{\tau \in [t^2]} u_t(s_\tau^{\text{tent}}, s) = t^2 \cdot \max_{s \in \mathcal{S}} \mathbb{E}_{\hat{s}_t \sim \bar{s}_t} [u_t(\hat{s}_t, s)]. \end{aligned}$$

The last line uses s¯<sup>t</sup> ← Unif {s tent 0 , stent 1 , . . . , stent t <sup>2</sup> } .

Thus, using that A<sup>S</sup> has regret O( √ T), we have E sˆt∼s¯t∀t hP<sup>T</sup> <sup>t</sup>=1 ut(ˆst, st) i ≤ P<sup>T</sup> <sup>t</sup>=1 t <sup>−</sup><sup>2</sup>Reg<sup>t</sup> <sup>2</sup> (A<sup>S</sup> ) = O( P<sup>T</sup> <sup>t</sup>=1 t −1 ) = O(log T): a lower-order term. Taking the expectation of the Step 1 bound thus gives the result.

#### 2.2. Efficient O(T 2/3 ) Online Multicalibration: Sketch

We now sketch a simple application illustrating that our highdimensional prediction methodology can be useful even for single-dimensional forecasting. This application also showcases the utility of our per-event bias bounds scaling optimally as O( p n<sup>T</sup> (E)) rather than as O( √ T).

Online Multicalibration In this setting [\(Gupta et al.,](#page-9-1) [2022;](#page-9-1) [Hebert-Johnson et al.](#page-10-0) ´ , [2018\)](#page-10-0), a learner, in each round

t ∈ [T], receives a context x<sup>t</sup> ∈ X , makes (randomized) prediction p<sup>t</sup> ∈ [0, 1], and receives true adversarial label y<sup>t</sup> ∈ [0, 1]. Upfront, a *group collection* G ⊆ 2 <sup>X</sup> is specified, and the learner's goal is to minimize the expected calibration error (ECE) conditional on every group G ∈ G: i.e. to minimize, for all groups G ∈ G, the expectation of:[<sup>2</sup>](#page-5-1)

$$ECE(G) = \sum_{p \in [0,1]} \left| \sum_{t=1}^T \mathbb{1}[x_t \in G] \mathbb{1}[p_t = p](y_t - p) \right|.$$

Obtaining the O(T 2/3 ) Bound via Unbiased Prediction We instantiate our framework by letting P = (Pi)i∈[m] be the m-point uniform discretization of [0, 1], and defining the following m · |G| group-calibration events:

$$E_{G,i} = \mathbb{1} \left[ x_t \in G, p_t \in (P_i - \frac{1}{2m}, P_i + \frac{1}{2m}] \right].$$

For each "bucket" (Pi± <sup>2</sup><sup>m</sup> ), imagine reassigning all predictions p<sup>t</sup> that fell in this bucket to be P<sup>i</sup> . For each group G, let (ni,G)i,G be the incidences of all discretized predicted values on G. We can then derive the bound:

$$ECE(G) = O(T/m) + \sum_{i \in [m]} O(\sqrt{n_{i,G}}),$$

where the first term is the discretization error, and the rest are bias bounds. In the worst case, this is O(T /m + m p T /m) = O(T /m + √ Tm). By tuning m = T 1/3 , we thus obtain *the first efficient* O(T 2/3 ) online multicalibration method. This rate in particular matches the rate of the algorithm of [Abernethy et al.](#page-8-1) [\(2011\)](#page-8-1) for vanilla calibration.

## 3. Unbiased Prediction for Decision Making

We now apply the unbiased framework to making predictions in the service of online adversarial decision making.

Agents (Decision Makers) We study agents (decision makers) who can choose amongst a set of actions A = {1, . . . , K}. They want to maximize utility as a function of both the action they take and of the state s ∈ S ⊆ R d .

Definition 3.1 (Agent's Utility). A utility function u : A × S → [0, 1] maps an action a ∈ A and a state s ∈ S to u(a, s). We assume that for every action a ∈ A, u is *linear* and L-*Lipschitz* in s, so that |u(a, s1) − u(a, s2)| ≤ L∥s<sup>1</sup> − s2∥<sup>∞</sup> for all s1, s<sup>2</sup> ∈ S and some L > 0.

Definition 3.2 (Best-Response). The *best response function*[<sup>3</sup>](#page-5-2) BRu: S→A for utility u is: BRu(s)= argmax a∈A u(a,s).

Suppose we make predictions sˆ1, . . . , sˆt. An agent with utility u may use them to take corresponding actions a1, . . . , at. We call an agent *straightforward* if they trust the predictions as correct (as if s<sup>t</sup> = ˆst) and thus always best respond:

Definition 3.3 (Straightforward Agent). An agent with utility u who treats predictions as correct and on every round t chooses a<sup>t</sup> = BRu(ˆst) is called *straightforward*.

Regret Since our predictions need not be correct, a straightforward agent may regret not having taken some other sequence of actions in hindsight (i.e. with knowledge of the true states s1, . . . , st). We study several regret notions.

Definition 3.4 (External regret). The external regret of a utility-u agent is defined as:

$$\text{Reg}_T(u) := \max_{a \in \mathcal{A}} \sum_{t=1}^T u(a, s_t) - u(a_t, s_t).$$

Definition 3.5 (Swap regret). A mapping ϕ : A → A is called a strategy modification mapping. Let Φ be the set of all such mappings. The *swap regret* of a utility-u agent is:

$$\text{SwapReg}_T(u) := \max_{\phi \in \Phi} \sum_{t=1}^T u(\phi(a_t), s_t) - u(a_t, s_t).$$

External regret compares the agent's play to the best fixed action. Swap regret [\(Blum & Mansour,](#page-8-0) [2007\)](#page-8-0) is strictly more challenging (indeed, external regret is equivalent to competing against the K constant strategy modification functions (ϕa)a∈A, where ϕ<sup>a</sup> : A → A is given by ϕa(a ′ ) = a for a ′ ∈ A), and allows the agent to compete against all re-mappings of their actions into other actions.

We now introduce the strong notion of *conditional regret*, parameterized by collections of events that may depend on the contexts and on the agent's actions. It requires the agent to have no external regret conditional on every event.

Definition 3.6 (Conditional Regret [\(Lehrer,](#page-10-7) [2003;](#page-10-7) [Blum &](#page-8-0) [Mansour,](#page-8-0) [2007;](#page-8-0) [Lee et al.,](#page-10-8) [2022\)](#page-10-8)). Fix Ξ, a finite collection of covariate-dependent and action-dependent subsequences of rounds: each member ξ ∈ Ξ is a mapping X ×A → [0, 1]. The Ξ*-conditional regret* of a utility-u agent is:

$$\begin{aligned} & \text{CReg}_T(\Xi, u) \\ &= \max_{\xi \in \Xi, a \in \mathcal{A}} \sum_{t=1}^T \xi(x_t, a_t) (u(a, s_t) - u(a_t, s_t)). \end{aligned}$$

#### 3.1. Swap Regret Guarantees for Many Agents

Environment Consider any convex compact state space S ⊆ R d . Suppose there are n agents, each with K discrete actions and with utility functions (ui)i∈[n] that are linear and L-Lipschitz in the state variable s ∈ S. We will now show how to make predictions (ˆst)t∈[T] to *simultaneously* guarantee no swap regret to every agent, given

<sup>2</sup>The summation over p ∈ [0, 1], despite looking uncountable, only has T nonzero terms, corresponding to p ∈ {p1, . . . , p<sup>T</sup> }.

<sup>3</sup>We assume that all ties are broken lexicographically.

that all agents are straightforward, i.e. they all best-respond: at,i = BR<sup>u</sup><sup>i</sup> (ˆst) for i ∈ [n]. We will do it by applying the unbiased prediction framework with the following natural collection of nK events.

Best-Response Events We write Eu,a(s) = <sup>1</sup>[BRu(s) = a] to denote the binary event that a is a best response to s for utility u. These events are essentially *level-set* events of the agent's best-response correspondence. We will now see that producing E-conditionally unbiased predictions (ˆst)t∈[T] for the nK-sized event collection E = (E<sup>u</sup>i,a)i∈[n],a∈[K] will suffice to guarantee no swap regret to all agents.

Informally, the reason these events give no swap regret to each agent u<sup>i</sup> is the following. Fix any strategy modification function ϕ ∈ Φ. Then each event E<sup>u</sup>i,a will ensure that on those rounds t where u<sup>i</sup> played a ∈ [K], the predictions sˆ<sup>t</sup> are sufficiently unbiased that a = BR<sup>u</sup><sup>i</sup> (ˆst), the best response that assumes the predictions are correct, is in fact the (approximately) best action to play over those rounds; in particular, a will have no regret to the re-mapped action ϕ(a) ∈ [K] on those rounds. Since this argument applies to all re-mappings ϕ ∈ Φ and all actions a ∈ [K], it will by definition ensure no swap regret to agent u<sup>i</sup> .

Formally, fix any u<sup>i</sup> and swap ϕ : A → A. We express the agent's regret to swap ϕ in terms of (E<sup>u</sup>i,a)a∈[K] as:

$$\begin{aligned} & \sum_{t=1}^T u_i(\phi(a_{t,i}), s_t) - u_i(a_{t,i}, s_t) \\ &= \sum_{a \in \mathcal{A}} \sum_{t: \text{BR}_{u_i}(\hat{s}_t)=a} u_i(\phi(a), s_t) - u_i(a, s_t) \\ &= \sum_{a \in \mathcal{A}} \sum_{t=1}^T E_{u_i, a}(\hat{s}_t) (u_i(\phi(a), s_t) - u_i(a, s_t)). \end{aligned}$$

By linearity of u<sup>i</sup> in st, we combine terms to get:

$$- \sum_{a \in \mathcal{A}} u_i \left( \phi(a), \sum_{t=1}^T E_{u_i, a}(\hat{s}_t) s_t \right)$$

Now, by L-Lipschitzness of u<sup>i</sup> , for a ′ ∈ {a, ϕ(a)} we have

$$\begin{aligned} \mathbb{E} \left| u_i \left( a', \sum_{t=1}^T E_{u_i, a}(\hat{s}_t) \hat{s}_t \right) - u_i \left( a', \sum_{t=1}^T E_{u_i, a}(\hat{s}_t) s_t \right) \right| \\ \leq L \cdot \mathbb{E} \left\| \sum_{t=1}^T E_{u_i, a}(\hat{s}_t) \cdot (\hat{s}_t - s_t) \right\|_{\infty} = L \cdot \text{Bias}_T(E_{u_i, a}). \end{aligned}$$

pected L

P

<sup>a</sup>∈[K] Bias<sup>T</sup> (E<sup>u</sup>i,a) error away from:

P a∈A ui ϕ(a), P T t=1

E<sup>u</sup>i,a(ˆst)ˆs<sup>t</sup>

 −u<sup>i</sup> a,P T t=1

E<sup>u</sup>i,a(ˆst)ˆs<sup>t</sup>

.

However, this last expression can be rewritten as P<sup>T</sup> <sup>t</sup>=1 ui(ϕ(at,i), sˆt) − ui(at,i, sˆt), which is nonpositive since actions at,i = BR<sup>u</sup><sup>i</sup> (ˆst) obtain the best utility *when evaluated on predicted states* sˆt. This means that we have shown a L P <sup>a</sup>∈[K] Bias<sup>T</sup> (E<sup>u</sup>i,a) expected regret bound for any agent to any swap function ϕ : A → A, which implies that each agent has expected swap regret at most L P <sup>a</sup>∈[K] Bias<sup>T</sup> (E<sup>u</sup>i,a).

Note that exactly one event from {E<sup>u</sup>i,a}a∈A occurs at each time t. Thus, P <sup>a</sup>∈A n<sup>T</sup> (Eu,a) ≤ T, so that L P <sup>a</sup>∈[K] Bias<sup>T</sup> (E<sup>u</sup>i,a) = O(L P a∈[K] p n<sup>T</sup> (E<sup>u</sup>i,a)) ≤ O(LKp T /K) = O(L √ KT). Therefore, we have shown: Theorem 3.7 (No Swap Regret for Multiple Agents). *In the above setting with* n *agents, all with* K *actions and Lipschitz utilities, if all agents best-respond to our forecasts* (ˆst)t∈[T] *, then by making these forecasts via* E*-unbiased prediction for* E = (E<sup>u</sup>i,a)i∈[n],a∈[K] *, the we efficiently obtain* O( √ KT) *swap regret bounds for all agents simultaneously.*

In the context of experts learning for a single decision maker, similar observations about the relevance of best-response partitions have been previously made by [Perchet](#page-10-9) [\(2011\)](#page-10-9) and [Haghtalab et al.](#page-10-10) [\(2023b\)](#page-10-10).

## 4. Conditional Regret Guarantees for Online Combinatorial Optimization

The regret guarantees that we just obtained for n agents with size-K action sets required unbiased predictions conditional on O(nK) events, resulting in poly(nK) runtime. This method applies to any finite action sets, and will be efficient where agents' action sets are modestly sized. However, when the agents' action sets are combinatorially large, this runtime is prohibitive. Below we identify an important setting in which an exponentially improved (oracle-) complexity poly(n log K) can be obtained.

### 4.1. Setting: Online Combinatorial Optimization

In a combinatorial optimization setting (as studied by [Kalai](#page-10-5) [& Vempala](#page-10-5) [\(2005\)](#page-10-5)), there are d ≥ 1 *base elements*, or *base actions*, e ∈ B := {1, . . . , d}, each offering an associated reward r<sup>e</sup> ∈ [−1, 1]. In this setting, an agent has action space A ⊆ 2 <sup>B</sup> — an arbitrarily structured collection of subsets of the base action set — and their utility u : A×S → [−d, d] is defined as the sum of the rewards of the base actions in the chosen action:

$$u(a, r) := \sum_{e \in a} r_e \quad \text{for } a \in \mathcal{A}, r \in \mathcal{S} := [-1, 1]^d.$$

Thus, given a vector r = (re)e∈[d] of d base rewards, the agent's optimization task is to identify, from among their actions a ∈ A, the highest-reward subset of the base action set. An *offline oracle* for this problem is any algorithm that, given r, computes the agent's best action in A.

Now, we define a contextual online n-agent setting in which base rewards vectors r<sup>t</sup> are generated by an adversary in rounds t ∈ [T]. In this setting, each agent i's goal will be to learn to play actions (at,i)t∈[T] that will minimize an appropriate notion of regret to the hindsight-best policy from some benchmark policy class.

Formally, consider an arbitrary context space X , and n ≥ 1 combinatorial agents with action sets (Ai)i∈[n] repeatedly playing the following game in rounds t ∈ [T]. In round t:

- 1. Agents i ∈ [n] observe context x<sup>t</sup> ∈ X , and commit to their actions at,i ∈ A<sup>i</sup> ;
- 2. Adversary produces base rewards vector r<sup>t</sup> ∈ [−1, 1]<sup>d</sup> ;
- 3. Agents see r<sup>t</sup> and get utilities ui(at,i, rt) := P e∈at,i rt,e.

Examples Suppose the base elements B correspond to the roads in a road network, the feasible subsets A<sup>i</sup> for each agent i ∈ [n] correspond to collections of roads that form source-to-sink paths for that agent in the network, and the reward for each road (edge) e ∈ B in the network is the (negative) latency on this edge. This classic instance of online combinatorial optimization is called online routing or online shortest paths [\(Takimoto & Warmuth,](#page-11-4) [2003;](#page-11-4) [Kalai](#page-10-5) [& Vempala,](#page-10-5) [2005\)](#page-10-5). More generally, the action spaces A<sup>i</sup> could represent *any* combinatorial structure; other wellstudied examples include spanning trees, Hamiltonian paths, and fixed-size subsets of the base set. For many of these classical examples, there exist efficient offline oracles, such as Bellman-Ford for shortest paths or Prim or Kruskal for spanning trees.

Regret Guarantees The FTPL algorithm of [Kalai & Vem](#page-10-5)[pala](#page-10-5) [\(2005\)](#page-10-5) reduces the problem of obtaining efficient *external* regret bounds for combinatorial optimization problems to the offline problem of linear optimization over the action spaces A<sup>i</sup> . Here we show for the first time how to efficiently obtain much stronger and more granular regret bounds: namely, Ξ*-conditional* regret bounds for any polynomially large collection of events Ξ. Moreover, unlike prior results, our result will provide these guarantees simultaneously for any finite collection of agents, letting us publish a concise forecast that is simultaneously useful for many downstream consumers.

Some existing general-purpose online algorithms [\(Blum &](#page-8-0) [Mansour,](#page-8-0) [2007;](#page-8-0) [Lee et al.,](#page-10-8) [2022;](#page-10-8) [Haghtalab et al.,](#page-10-6) [2023a\)](#page-10-6) could be used to obtain conditional regret bounds sublinear

in T, by directly optimizing over the entire action set A<sup>i</sup> of each agent. However, their runtime will then scale polynomially in |A<sup>i</sup> | which can be as large as Ω(2<sup>d</sup> ), thus making the runtime exponentially large in the problem size.

Our framework, by contrast, will let us give the agents conditional regret guarantees simply by unbiasedly predicting the d-dimensional base reward vectors (conditionally on appropriate events). Our general algorithm will thus run in time poly(d), giving us an efficient algorithm with an exponential runtime improvement compared to prior work.

#### 4.2. Conditional Regret via Unbiased Prediction

To derive conditional regret guarantees via unbiased prediction, we will use the same Predict-then-Act approach as in Section [3.](#page-5-0) At the beginning of each round t, we will (appropriately unbiasedly) predict the rewards vector rˆ<sup>t</sup> ∈ S. Every agent i ∈ [n] will then best-respond to our prediction and select action at,i = BR<sup>u</sup><sup>i</sup> (ˆrt).

However, to make this approach efficient, we must now design our event collection differently than before. Indeed, consider the simplest case where we ask for no external regret to downstream agents. The collection of "level set" events {E<sup>u</sup>i,a<sup>i</sup> }i∈[n],ai∈A<sup>i</sup> studied in Section [3](#page-5-0) will imply sublinear regret as before — but will be too big as it scales with |A<sup>i</sup> |, which can be exponential in d.

To overcome this, we will take advantage of the special structure of the payoffs, which are all linear in the base element rewards. The idea is to condition on events defined by the *base elements* e ∈ B = [d]. Again starting with no external regret, it turns out that for each agent i ∈ [n] it suffices to condition on the always-on event, and on d events (Ee)e∈[d] : for each base element e, E<sup>e</sup> will be the event that *the agent's chosen action* at,i *contains* e, i.e., Ee(s) = 1[e ∈ BR<sup>u</sup><sup>i</sup> (s)]. This requires just nd events over all agents. From here, if we now desire Ξ-conditional regret guarantees, it will suffice to expand this event collection to O(|Ξ|·d) events per agent: for each ξ ∈ Ξ, the intersectional events Ee,ξ(x, s) = ξ(x, s) · Ee(s) for e ∈ [d] (that both ξ is active and E<sup>e</sup> is active), as well as the event that is active whenever ξ is active, will imply no external regret conditional on ξ.

Importantly, observe that all these events can be evaluated via direct calls to the offline optimization oracle for the problem (e.g., Bellman-Ford for routing); therefore, the unbiased prediction algorithm will be oracle efficient. We now state and prove our conditional regret bound.

Theorem 4.1. *Consider online combinatorial optimization over context space* X *with* d *base actions and* n *agents with action sets* A<sup>i</sup> ⊆ 2 [d] *. Suppose each agent* i *is straightforward, and wants to obtain no* Ξi*-conditional regret for some*

*events* Ξ<sup>i</sup> *. Define the following set of events:*

$$\mathcal{E} = \bigcup_{i \in [n]} \bigcup_{\xi \in \Xi_i} \left\{ \{E_{e,\xi}^i\}_{e \in [d]} \cup \{E_{\xi}^i\} \right\}.$$

*Here,* E<sup>i</sup> e,ξ(xt, rˆt) := <sup>1</sup>[e ∈ BR<sup>u</sup><sup>i</sup> (ˆrt)] · ξ(xt, BR<sup>u</sup><sup>i</sup> (ˆrt))*; and* E<sup>i</sup> (xt, rˆt) := ξ(xt, BR<sup>u</sup><sup>i</sup> (ˆrt))*, for all* e, i, ξ ∈ Ξ<sup>i</sup> *.*

*Then, running Unbiased Prediction on state space* S = [−1, 1]<sup>d</sup> *with event collection* E *will produce a sequence of predictions* (ˆrt)t∈[T] *such that each agent* i ∈ [n]*, by playing their best-response actions* at,i = BR<sup>u</sup><sup>i</sup> (ˆrt) *at all rounds, will obtain* <sup>O</sup>e(<sup>d</sup> √ T) *expected* Ξi*-conditional regret. The runtime will consist of* poly (dT P<sup>n</sup> <sup>i</sup>=1 |Ξ<sup>i</sup> |) *oracle calls to the offline optimization oracle for the setting.*

*Proof.* It suffices to fix any agent i ∈ [n] and event ξ ∈ Ξ<sup>i</sup> and show that best-responding to the E-unbiased predictions (ˆrt)t∈[T] gets i no external regret on subsequence ξ. Denote the agent's external regret on the subsequence ξ by:

$$\begin{aligned} & \text{CReg}_T(\xi, i) \\ &= \max_{a^* \in \mathcal{A}_i} \sum_{t=1}^T \xi(x_t, a_{t,i}) \cdot (u_i(a^*, r_t) - u_i(a_{t,i}, r_t)). \end{aligned}$$

Now, consider the hypothetical "ideal" scenario in which our predictions are exactly correct on every round, i.e., rˆ<sup>t</sup> = r<sup>t</sup> for all t. Then, our "ideal" regret would be nonpositive:

$$\begin{aligned} & \text{IdReg}_T(\xi, i) \\ &= \max_{a^* \in \mathcal{A}_i} \sum_{t=1}^T \xi(x_t, a_{t,i}) \cdot (u_i(a^*, \hat{r}_t) - u_i(a_{t,i}, \hat{r}_t)) \leq 0 \end{aligned}$$

since for each t, at,i = BR<sup>u</sup><sup>i</sup> (ˆrt) and thus ui(at,i, rˆt) = maxa∈A<sup>i</sup> ui(a, rˆt) ≥ ui(a ∗ , rˆt). Therefore,

$$\text{CReg}_T(\xi, i) \leq \text{CReg}_T(\xi, i) - \text{IdReg}_T(\xi, i).$$

This difference in regrets can be expressed as:

$$\begin{aligned} & \max_{a^* \in \mathcal{A}_i} \sum_{t=1}^T \xi(x_t, a_{t,i}) \cdot u_i(a^*, r_t) \\ & - \max_{a^* \in \mathcal{A}_i} \sum_{t=1}^T \xi(x_t, a_{t,i}) \cdot u_i(a^*, \hat{r}_t) \\ & + \sum_{t=1}^T \xi(x_t, a_{t,i}) \cdot (u_i(a_{t,i}, \hat{r}_t) - u_i(a_{t,i}, r_t)). \end{aligned}$$

It is easy to check that the first line's expectation is at most Bias<sup>T</sup> (E<sup>i</sup> ). Similarly, by decomposing the second line's expression across the d coordinates, it can be seen that its expectation is at most P <sup>e</sup>∈[d] Bias<sup>T</sup> (E<sup>i</sup> e,ξ). Hence, the expected regret of agent P i conditional on ξ is at most <sup>e</sup>∈[d] Bias<sup>T</sup> (E<sup>i</sup> e,ξ) + Bias<sup>T</sup> (E<sup>i</sup> ) = <sup>O</sup>e(<sup>d</sup> √ T).

### Acknowledgements

We are grateful to Edgar Dobriban, Amy Greenwald, Jason Hartline, Michael Jordan, Shuo Li, Jon Schneider, and Rakesh Vohra for insightful conversations at various stages of this work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References


[1] Abernethy, J., Bartlett, P. L., and Hazan, E. Blackwell approachability and no-regret learning are equivalent. In *Proceedings of the 24th Annual Conference on Learning Theory*, pp. 27–46. JMLR Workshop and Conference Proceedings, 2011. Bastani, O., Gupta, V., Jung, C., Noarov, G., Ramalingam, R., and Roth, A. Practical adversarial multivalid conformal prediction. In *Advances in Neural Information Processing Systems*, 2022. Blum, A. and Lykouris, T. Advancing subgroup fairness via sleeping experts. In *11th Innovations in Theoretical Computer Science Conference (ITCS 2020)*. Schloss Dagstuhl-Leibniz-Zentrum fur Informatik, 2020. ¨ Blum, A. and Mansour, Y. From external to internal regret. *Journal of Machine Learning Research*, 8(6), 2007. Camara, M. K., Hartline, J. D., and Johnsen, A. Mechanisms for a no-regret agent: Beyond the common prior. In *2020 ieee 61st annual symposium on foundations of computer science (focs)*, pp. 259–270. IEEE, 2020. Chen, L., Luo, H., and Wei, C.-Y. Impossible tuning made possible: A new expert algorithm and its applications. In *Conference on Learning Theory*, pp. 1216–1259. PMLR, 2021. Collina, N., Goel, S., Gupta, V., and Roth, A. Tractable agreement protocols. *arXiv preprint arXiv:2411.19791*, 2024a. Collina, N., Roth, A., and Shao, H. Efficient prior-free mechanisms for no-regret agents. In *Proceedings of the 25th ACM Conference on Economics and Computation*, pp. 511–541, 2024b. Condat, L. Fast projection onto the simplex and the l 1 ball. *Mathematical Programming*, 158(1-2):575–585, 2016.

[2] Dawid, A. P. Calibration-based empirical probability. *The Annals of Statistics*, 13(4):1251–1274, 1985. Demirovic, E., Stuckey, P. J., Bailey, J., Chan, J., Leckie, ´ C., Ramamohanarao, K., and Guns, T. An investigation into prediction+ optimisation for the knapsack problem. In *Integration of Constraint Programming, Artificial Intelligence, and Operations Research: 16th International Conference, CPAIOR 2019, Thessaloniki, Greece, June 4–7, 2019, Proceedings 16*, pp. 241–257. Springer, 2019. Donti, P., Amos, B., and Kolter, J. Z. Task-based end-to-end model learning in stochastic optimization. *Advances in neural information processing systems*, 30, 2017. Dwork, C., Kim, M. P., Reingold, O., Rothblum, G. N., and Yona, G. Outcome indistinguishability. In *Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing*, pp. 1095–1108, 2021. Dwork, C., Kim, M., Reingold, O., Rothblum, G., and Yona, G. Beyond bernoulli: Generating random outcomes that cannot be distinguished from nature. In Dasgupta, S. and Haghtalab, N. (eds.), *Proceedings of The 33rd International Conference on Algorithmic Learning Theory*, volume 167 of *Proceedings of Machine Learning Research*, pp. 342–380. PMLR, 29 Mar–01 Apr 2022. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v167/dwork22a.html) [v167/dwork22a.html](https://proceedings.mlr.press/v167/dwork22a.html). El Balghiti, O., Elmachtoub, A. N., Grigas, P., and Tewari,

[3] A. Generalization bounds in the predict-then-optimize framework. *Advances in neural information processing systems*, 32, 2019. Elmachtoub, A. N. and Grigas, P. Smart "predict, then optimize". *Management Science*, 68(1):9–26, 2022. Fisch, A., Jaakkola, T., and Barzilay, R. Calibrated selective classification. *arXiv preprint arXiv:2208.12084*, 2022. Foster, D. P. and Hart, S. Smooth calibration, leaky forecasts, finite recall, and nash dynamics. *Games and Economic Behavior*, 109:271–293, 2018. Foster, D. P. and Kakade, S. M. Calibration via regression. In *2006 IEEE Information Theory Workshop-ITW'06 Punta del Este*, pp. 82–86. IEEE, 2006. Foster, D. P. and Vohra, R. Regret in the on-line decision problem. *Games and Economic Behavior*, 29(1-2):7–35, 1999. Foster, D. P. and Vohra, R. V. Asymptotic calibration. *Biometrika*, 85(2):379–390, 1998. Foster, D. P., Rakhlin, A., Sridharan, K., and Tewari, A. Complexity-based approach to calibration with checking rules. In *Proceedings of the 24th Annual Conference on Learning Theory*, pp. 293–314. JMLR Workshop and Conference Proceedings, 2011. Garg, S., Jung, C., Reingold, O., and Roth, A. Oracle efficient online multicalibration and omniprediction. In *ACM-SIAM Symposium on Discrete Algorithms*, 2024. Globus-Harris, I., Harrison, D., Kearns, M., Roth, A., and Sorrell, J. Multicalibration as boosting for regression. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp. 11459–11492. PMLR, 2023. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v202/globus-harris23a.html) [v202/globus-harris23a.html](https://proceedings.mlr.press/v202/globus-harris23a.html). Gopalan, P., Kalai, A. T., Reingold, O., Sharan, V., and Wieder, U. Omnipredictors. In Braverman, M. (ed.), *13th Innovations in Theoretical Computer Science Conference, ITCS 2022, January 31 - February 3, 2022, Berkeley, CA, USA*, volume 215 of *LIPIcs*, pp. 79:1–79:21. Schloss Dagstuhl - Leibniz-Zentrum fur Informatik, 2022a. doi: ¨ 10.4230/LIPIcs.ITCS.2022.79. URL [https://doi.](https://doi.org/10.4230/LIPIcs.ITCS.2022.79) [org/10.4230/LIPIcs.ITCS.2022.79](https://doi.org/10.4230/LIPIcs.ITCS.2022.79). Gopalan, P., Kim, M. P., Singhal, M. A., and Zhao, S. Lowdegree multicalibration. In *Conference on Learning Theory*, pp. 3193–3234. PMLR, 2022b. Gopalan, P., Hu, L., Kim, M. P., Reingold, O., and Wieder,
  - U. Loss minimization through the lens of outcome indistinguishability. In *14th Innovations in Theoretical Computer Science Conference (ITCS 2023)*. Schloss Dagstuhl-Leibniz-Zentrum fur Informatik, 2023a. ¨ Gopalan, P., Kim, M. P., and Reingold, O. Characterizing notions of omniprediction via multicalibration. *arXiv preprint arXiv:2302.06726*, 2023b. Gopalan, P., Kim, M., and Reingold, O. Swap agnostic learning, or characterizing omniprediction via multicalibration. *Advances in Neural Information Processing Systems*, 36, 2024. Grotschel, M., Lov ¨ asz, L., and Schrijver, A. ´ *Geometric Algorithms and Combinatorial Optimization*, volume 2 of *Algorithms and Combinatorics*. Springer, 1988. ISBN 978-3-642-97883-8. doi: 10.1007/978-3-642-97881-4. Gupta, C. and Ramdas, A. Top-label calibration and multiclass-to-binary reductions. *arXiv preprint arXiv:2107.08353*, 2021. Gupta, V., Jung, C., Noarov, G., Pai, M. M., and Roth,
    - A. Online multivalid learning: Means, moments, and

[4] prediction intervals. In *13th Innovations in Theoretical Computer Science Conference (ITCS 2022)*. Schloss Dagstuhl-Leibniz-Zentrum fur Informatik, 2022. ¨ Haghtalab, N., Jordan, M., and Zhao, E. A unifying perspective on multi-calibration: Game dynamics for multiobjective learning. *Advances in Neural Information Processing Systems*, 36:72464–72506, 2023a. Haghtalab, N., Podimata, C., and Yang, K. Calibrated stackelberg games: Learning optimal commitments against calibrated agents. *arXiv preprint arXiv:2306.02704*, 2023b. Hannan, J. Approximation to bayes risk in repeated play. *Contributions to the Theory of Games*, 3:97–139, 1957. Hazan, E. Introduction to online convex optimization. *Foundations and Trends® in Optimization*, 2(3-4):157–325, 2016. Hazan, E. and Kakade, S. M. (weak) calibration is computationally hard. In *Conference on Learning Theory*, pp. 3–1. JMLR Workshop and Conference Proceedings, 2012. Hebert-Johnson, U., Kim, M., Reingold, O., and Rothblum, ´

[5] G. Multicalibration: Calibration for the (computationallyidentifiable) masses. In *International Conference on Machine Learning*, pp. 1939–1948. PMLR, 2018. Hu, L. and Wu, Y. Predict to minimize swap regret for all payoff-bounded tasks. In *65th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2024, Chicago, IL, USA, October 27-30, 2024*, pp. 244–

263. IEEE, 2024. doi: 10.1109/FOCS61266.2024.00024. URL [https://doi.org/10.1109/FOCS61266.](https://doi.org/10.1109/FOCS61266.2024.00024) [2024.00024](https://doi.org/10.1109/FOCS61266.2024.00024). Jung, C., Lee, C., Pai, M., Roth, A., and Vohra, R. Moment multicalibration for uncertainty estimation. In *Conference on Learning Theory*, pp. 2634–2678. PMLR, 2021. Jung, C., Noarov, G., Ramalingam, R., and Roth, A. Batch multivalid conformal prediction. In *International Conference on Learning Representations (ICLR)*, 2023. Kakade, S. M. and Foster, D. P. Deterministic calibration and nash equilibrium. *Journal of Computer and System Sciences*, 74(1):115–130, 2008. Kalai, A. and Vempala, S. Efficient algorithms for online decision problems. *Journal of Computer and System Sciences*, 71(3):291–307, 2005. Khalil, E., Dai, H., Zhang, Y., Dilkina, B., and Song,

[7] L. Learning combinatorial optimization algorithms over graphs. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf) [cc/paper\\_files/paper/2017/file/](https://proceedings.neurips.cc/paper_files/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf) [d9896106ca98d3d05b8cbdf4fd8b13a1-Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf). Kim, M. P. and Perdomo, J. C. Making decisions under outcome performativity. In *14th Innovations in Theoretical Computer Science Conference (ITCS 2023)*. Schloss-Dagstuhl-Leibniz Zentrum fur Informatik, 2023. ¨ Kim, M. P., Ghorbani, A., and Zou, J. Multiaccuracy: Blackbox post-processing for fairness in classification. In *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society*, pp. 247–254, 2019. Kleinberg, R., Leme, R. P., Schneider, J., and Teng, Y. Ucalibration: Forecasting for an unknown agent. *arXiv preprint arXiv:2307.00168*, 2023. Kuleshov, V. and Liang, P. S. Calibrated structured prediction. *Advances in Neural Information Processing Systems*, 28, 2015. Lee, D., Noarov, G., Pai, M., and Roth, A. Online minimax multiobjective optimization: Multicalibeating and other applications. *Advances in Neural Information Processing Systems*, 35:29051–29063, 2022. Lehrer, E. A wide range no-regret theorem. *Games and Economic Behavior*, 42(1):101–115, 2003. Liu, H. and Grigas, P. Risk bounds and calibration for a smart predict-then-optimize method. *Advances in Neural Information Processing Systems*, 34:22083–22094, 2021. Mandi, J., Stuckey, P. J., Guns, T., et al. Smart predict-andoptimize for hard combinatorial optimization problems. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34 (02), pp. 1603–1610, 2020. Noarov, G. and Roth, A. The statistical scope of multicalibration. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp. 26283–26310. PMLR, 2023. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v202/noarov23a.html) [press/v202/noarov23a.html](https://proceedings.mlr.press/v202/noarov23a.html). Perchet, V. Internal regret with partial monitoring: Calibration-based optimal algorithms. *Journal of Machine Learning Research*, 12(6), 2011. Qiao, M. and Valiant, G. Stronger calibration lower bounds via sidestepping. In *Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing*, pp. 456–466, 2021.

[8] Roth, A. Uncertain: Modern topics in uncertainty estimation. https://www.cis.upenn.edu/ aaroth/uncertainty-notes.pdf, 2022. Roth, A. and Shi, M. Forecasting for swap regret for all downstream agents. In *Proceedings of the 25th ACM Conference on Economics and Computation*, pp. 466– 488, 2024. Rothblum, G. N. and Yona, G. Decision-making under miscalibration. *arXiv preprint arXiv:2203.09852*, 2022. Slivkins, A. et al. Introduction to multi-armed bandits. *Foundations and Trends® in Machine Learning*, 12(1- 2):1–286, 2019. Takimoto, E. and Warmuth, M. K. Path kernels and multiplicative updates. *The Journal of Machine Learning Research*, 4:773–818, 2003. Vanderschueren, T., Verdonck, T., Baesens, B., and Verbeke,

[9] W. Predict-then-optimize or predict-and-optimize? an empirical evaluation of cost-sensitive learning strategies. *Information Sciences*, 594:400–415, 2022. Wang, L., Joachims, T., and Rodriguez, M. G. Improving screening processes via calibrated subset selection. In *International Conference on Machine Learning*, pp. 22702–22726. PMLR, 2022. Wilder, B., Dilkina, B., and Tambe, M. Melding the datadecisions pipeline: Decision-focused learning for combinatorial optimization. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33 (01), pp. 1658–1665, 2019. Zadrozny, B. and Elkan, C. Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining*, pp. 694–699, 2002. Zhao, S., Kim, M., Sahoo, R., Ma, T., and Ermon, S. Calibrating predictions to decisions: A novel approach to multi-class calibration. *Advances in Neural Information Processing Systems*, 34:22313–22324, 2021. Zinkevich, M. Online convex programming and generalized infinitesimal gradient ascent. In *Proceedings of the 20th international conference on machine learning (icml-03)*, pp. 928–936, 2003.
## A. Additional Related Work

Calibration The study of sequential calibration goes back to [\(Dawid,](#page-9-0) [1985\)](#page-9-0) who viewed it as a way to define the foundations of probability, and algorithms for producing calibrated forecasts in an adversarial setting were first given by [\(Foster & Vohra,](#page-9-5) [1998\)](#page-9-5). [\(Foster & Vohra,](#page-9-4) [1999\)](#page-9-4) were the first to connect sequential calibration to sequential decision making, showing that a decision maker who best responds to (fully) calibrated forecasts obtains diminishing internal regret (and that when all agents in a game do so, empirical play converges to correlated equilibrium). [\(Kakade & Foster,](#page-10-11) [2008\)](#page-10-11) and [\(Foster](#page-9-6) [& Hart,](#page-9-6) [2018\)](#page-9-6) make a similar connection between "smooth calibration" (which in contrast to classical calibration can be obtained with deterministic algorithms) and Nash equilibrium.

Much of the calibration literature has focused on the binary prediction and one dimensional regression settings, where labels are in {0, 1} or in [0, 1], and predictions are in [0, 1]. Comparatively few works, including [\(Zadrozny & Elkan,](#page-11-5) [2002;](#page-11-5) [Kuleshov & Liang,](#page-10-12) [2015;](#page-10-12) [Gupta & Ramdas,](#page-9-7) [2021\)](#page-9-7), have addressed higher dimensional predictions, which, as discussed, are challenging because of the curse of dimensionality; many of these works have sought to *reduce* multiclass calibration problems to binary calibration. In this context, our work in particular proposes tractable notions of online multiclass calibration for cases when there is a specific downstream task that the forecasts will be used for.

Multicalibration In the recent computer science literature, there has been interest in constructive calibration guarantees (obtained by efficient algorithms and obtaining good rates) that hold conditional on context in various ways, called *multicalibration* [\(Hebert-Johnson et al.](#page-10-0) ´ , [2018\)](#page-10-0). Multicalibration has been studied both in the batch setting [\(Hebert-Johnson et al.](#page-10-0) ´ , [2018;](#page-10-0) [Kim et al.,](#page-10-13) [2019;](#page-10-13) [Globus-Harris et al.,](#page-9-8) [2023;](#page-9-8) [Haghtalab et al.,](#page-10-6) [2023a\)](#page-10-6) and in the online sequential setting [\(Foster &](#page-9-9) [Kakade,](#page-9-9) [2006;](#page-9-9) [Foster et al.,](#page-9-10) [2011;](#page-9-10) [Gupta et al.,](#page-9-1) [2022;](#page-9-1) [Garg et al.,](#page-9-11) [2024\)](#page-9-11). For the most part (with a few notable exceptions [\(Gopalan et al.,](#page-9-12) [2022b;](#page-9-12) [Zhao et al.,](#page-11-2) [2021\)](#page-11-2)) multicalibration has been studied in the 1-dimensional setting in which the outcome being predicted is boolean. This has been extended to predicting real-valued outcomes, with notions of calibration tailored to variances [\(Jung et al.,](#page-10-14) [2021\)](#page-10-14), quantiles [\(Bastani et al.,](#page-8-7) [2022;](#page-8-7) [Jung et al.,](#page-10-15) [2023\)](#page-10-15), and other distributional properties [\(Noarov & Roth,](#page-10-16) [2023\)](#page-10-16). See [\(Roth,](#page-11-6) [2022\)](#page-11-6) for an introductory exposition of this literature. Our algorithm can be used to recover many of the above online multicalibration guarantees by plugging in appropriate events, but it goes beyond multicalibration constraints.

Omniprediction A growing line of work [\(Gopalan et al.,](#page-9-2) [2022a;](#page-9-2) [2023a;](#page-9-13)[b;](#page-9-14) [2024\)](#page-9-15) aims to use (multi)calibration as a tool for a one-dimensional form of downstream decision making, called omniprediction. The goal of omniprediction is to make probabilistic predictions of a binary outcome as a function of contextual information that are useful for simultaneously optimizing a variety of downstream loss functions. E.g., [\(Gopalan et al.,](#page-9-2) [2022a\)](#page-9-2) show that a predictor that is multicalibrated with respect to a benchmark class of functions H and of a binary label can be used to optimize any convex, Lipschitz loss function of an action and a binary label. Also related is the *outcome indistinguishability* strand of research [\(Dwork et al.,](#page-9-3) [2021;](#page-9-3) [2022\)](#page-9-16), which studies producing decisions that are indistinguishable from the ground truth according to a collection of tests.

Conceptually, our motivation is slightly different than for omniprediction: while omniprediction aims to produce forecasts that are good enough to optimize for a *large* (typically infinitely large) family of *possible* downstream tasks characterized by their associated losses — such that we may not know ahead of time which task will present itself — our framework is developed to handle finitely many arbitrary but *specific* (i.e., known-in-advance) downstream tasks. The above mentioned results are in the batch setting.

In the online setting, [\(Kleinberg et al.,](#page-10-1) [2023\)](#page-10-1) defined "U-calibration", which can be viewed as a non-contextual version of omniprediction where the goal is to make predictions that guarantee an arbitrary downstream decision maker no external regret. In comparison to [\(Kleinberg et al.,](#page-10-1) [2023\)](#page-10-1), our goal is to give both stronger guarantees than external regret, and to be able to do so even when the state space is very large.

Calibration for Decision Making The most closely related work is [\(Zhao et al.,](#page-11-2) [2021\)](#page-11-2), who define and study "decision calibration" in the batch setting in the context of predicting a probability distribution over k discrete outcomes. Decision calibration is a slightly weaker requirement than what we study, also defined in terms of the best-response correspondence of a decision maker's utility function. Decision calibration asks, informally, that a decision maker be able to correctly estimate the expected reward of their best response policy; we ask for a slightly stronger condition that requires them to also be able to estimate the utility of deviations as a function of their play. This kind of unbiased estimation (based on the best-response correspondence of a decision maker) has also been previously observed to be related to swap regret in [\(Perchet,](#page-10-9) [2011\)](#page-10-9) and [\(Haghtalab et al.,](#page-10-10) [2023b\)](#page-10-10). The algorithmic portion of our work can be viewed as extending [\(Zhao et al.,](#page-11-2) [2021\)](#page-11-2) from the batch to the online adversarial setting; Our applications hinge crucially on both the online aspect of our algorithm and on the more general setting we consider, beyond predicting distributions on k outcomes.

Subsequently to [\(Zhao et al.,](#page-11-2) [2021\)](#page-11-2), decision calibration has been extended to, or applied in, several specific downstream tasks in the batch setting. For instance, [\(Fisch et al.,](#page-9-17) [2022;](#page-9-17) [Wang et al.,](#page-11-7) [2022\)](#page-11-7) applied decision calibration in the presence of downstream selection or screening processes. These and omniprediction ideas were also used to obtain new *performative prediction* algorithms in [\(Kim & Perdomo,](#page-10-17) [2023\)](#page-10-17). In the opposite direction, [\(Rothblum & Yona,](#page-11-8) [2022\)](#page-11-8) study how downstream decision policies can be modified in response to miscalibrated forecasts.

Predict then Optimize An expansive recent literature has focused on the similarly named *predict-then-optimize* problem [\(Elmachtoub & Grigas,](#page-9-18) [2022;](#page-9-18) [El Balghiti et al.,](#page-9-19) [2019;](#page-9-19) [Liu & Grigas,](#page-10-18) [2021\)](#page-10-18). This line of work investigates a setup in which predictions made from data are to be used in a linear optimization problem downstream in the pipeline. This is similar in motivation to our framework, but with two important differences: (1) the predict-then-optimize framework aims to optimize for a single downstream problem, whereas we aim to simultaneously provide guarantees to an arbitrary finite collection of downstream decision makers; and (2) the surrogate loss approach studied in this literature is naturally embedded in a batch/distributional setting, where the goal is to exactly optimize for the Bayes optimal downstream decision policy, up to generalization/risk bounds; meanwhile, our framework naturally lives in the online adversarial setting, and aims for different notions of optimality defined in terms of regret bounds, as well as omniprediction-type 'best-in-class' optimality. Both frameworks can be used to solve downstream combinatorial optimization problems [\(Mandi et al.,](#page-10-19) [2020;](#page-10-19) [Demirovic et al.](#page-9-20) ´ , [2019\)](#page-9-20); but our framework appears to have a broader set of applications — as a consequence of its strong calibration properties, we are able to apply our framework to derive strong uncertainty quantification guarantees, which do not appear to naturally fit within the predict-then-optimize framework. There also exist other approaches for learning in batch decision making pipelines, that are different from the predict-then-optimize method; see e.g. [\(Donti et al.,](#page-9-21) [2017;](#page-9-21) [Khalil et al.,](#page-10-20) [2017;](#page-10-20) [Wilder et al.,](#page-11-9) [2019;](#page-11-9) [Vanderschueren et al.,](#page-11-10) [2022\)](#page-11-10).

## B. Calibration and Decision Making

The notion of calibration [\(Dawid,](#page-9-0) [1985\)](#page-9-0) requires that predictors make forecasts that are consistent with the ground truth conditional on the predicted values themselves. For instance, for a binary predictor f : X → [0, 1], it enforces, roughly speaking, that <sup>E</sup>(x,y) [y|f(x) ≈ v] ≈ v, for all v ∈ [0, 1].

Calibration has very strong decision-theoretic properties. When making predictions about a payoff-relevant state, in a very general setting it is a dominant strategy amongst all *prediction-to-action policies* for every downstream decision maker to best-respond to calibrated predictions as if they were correct. This has strong semantics as "trustworthiness" — as one can do no better than to trust calibrated predictions and *act accordingly*. It also implies strong performance guarantees for the downstream decision makers. Here are two examples: First, decision makers who best respond to calibrated forecasts are guaranteed to have no swap regret — meaning that they obtain utility that is as high as the best action they could have played in hindsight, not just marginally, but also *conditionally* on each action that they played [\(Foster & Vohra,](#page-9-4) [1999\)](#page-9-4). The second example concerns multi-class prediction, where a decision-maker observes features and predicts an unknown label from some large set. Standard machine learning methods for multi-class classification will, given features, predict "scores" for each label that look like probabilities in that they are non-negative and sum to 1. These scores are not probabilities; nevertheless, decision makers who produce "prediction sets" of labels by treating *calibrated* scores as if they were real probabilities will find that their prediction sets cover the true label with the same frequency that they would if the scores really were conditional label probabilities. So sequential calibration offers very strong guarantees — and it has been known since [\(Foster & Vohra,](#page-9-5) [1998\)](#page-9-5) that it is possible to produce calibrated forecasts even in adversarial environments.

But calibration is in different senses both too strong and too weak. On the one hand, calibration is too weak in that it provides only a *marginal* guarantee; calibrated forecasts will in general fail to be calibrated conditional on external information. Thus, the property that downstream agents can do no better than to treat calibrated forecasts as correct will fail to hold if the downstream agents have access to external context. In one-dimensional prediction settings (such as real-valued regression), *multicalibration* [\(Hebert-Johnson et al.](#page-10-0) ´ , [2018\)](#page-10-0) mitigates this weakness: it allows one to enforce calibration not just marginally, but also conditionally on any collection of context-dependent *groups* or *subpopulations* in the data.

On the other hand, calibration is too strong in that (because it is agnostic to downstream decisions) it conditions on fine distinctions in its predictions that may be irrelevant to the downstream task at hand. As a result, calibration is intractable in *high-dimensional* settings. Since calibrated predictions must be statistically unbiased conditional on their own values, then for d-dimensional prediction problems — in which, up to discretization, there are Ω(2<sup>d</sup> ) possible values we may predict we, naively, need to respect Ω(2<sup>d</sup> ) possible conditioning events to stay calibrated. Given this intuition, it should come as no surprise that the best known calibration algorithms have exponential computational and statistical complexity in dimension d of the outcome space; there exist some lower bounds in the literature that confirm this hardness, see e.g. the PPAD-hardness result of [\(Hazan & Kakade,](#page-10-21) [2012\)](#page-10-21)). In fact, even in 1 dimension, it is known that achieving adversarial calibration at a rate of O( √ T) is impossible [\(Qiao & Valiant,](#page-10-22) [2021\)](#page-10-22) — even though it is possible to obtain swap regret at this rate [\(Blum &](#page-8-0) [Mansour,](#page-8-0) [2007\)](#page-8-0). Thus, despite its remarkable guarantees, calibration has been of little utility in designing online algorithms for high-dimensional problems.

## C. Connecting Prediction and Decision Making

We next make connections between our ability to make unbiased predictions and the quality of decisions that are made downstream as a function of our predictions in a general setting. The form of the argument will proceed in the same way that it will in our main applications, and so is instructive.

Specifically, we will show that a straightforward decision maker who simply best responds to our predictions can be guaranteed *no swap regret* — if we just make our predictions *unbiased* conditional on the events defined by the decision maker's best-response correspondence.

The Predict-then-Act Paradigm These results, whose formal statements are given below, suggest a natural design paradigm for sequential decision algorithms, which we call *predict-then-act*. The idea is simple: first we make a prediction sˆ<sup>t</sup> for an unknown payoff-relevant parameter st, and then we choose an action as if our prediction were correct — i.e. we best respond to sˆt. We can parameterize the predict-then-act algorithm with various events E, such that our predictions will be unbiased with respect to events in E. Whenever E is a collection of polynomially many events that can each be evaluated in polynomial time, the predict-then-act algorithm can be implemented in polynomial time per step. While Predict-Then-Act is quite simple, its flexibility in a variety of settings lies in the design of the event set E and prediction space S. By choosing the events E to be appropriately tailored to the task at hand, we can arrange that Predict-Then-Act has guarantees of various sorts.

Algorithm 2 Predict-Then-Act(T, U, E, S, A)

for t in 1 . . . T do Compute ψ<sup>t</sup> ←UnbiasedPrediction(E, t) Predict sˆ<sup>t</sup> ∼ ψ<sup>t</sup> for u<sup>i</sup> ∈ U do Decision maker i selects action at,i = BR<sup>u</sup><sup>i</sup> (ˆst) = argmaxa∈A ui(a, sˆt) end for Observe outcome s<sup>t</sup> ∈ S end for

We now develop, and use, our machinery based on the Predict-then-Act approach powered by our Unbiased Prediction algorithm's guarantees, to give algorithms with strong no-regret guarantees in a variety of sequential settings. In all cases, the scenario we analyze is that in rounds t, a predictor makes state predictions sˆt, after which one or more decision makers choose actions that are best responses to sˆ<sup>t</sup> (i.e. they function as *straightforward* decision makers). When we are designing an algorithm for a single decision maker, we always use the *predict-then-act* paradigm (Algorithm [2\)](#page-14-0). When we are designing a coordination mechanism, we imagine that our predictions are simultaneously issued, on every round, to multiple decision makers, who each independently act. We show how guaranteeing that our predictions are unbiased subject to appropriately chosen events gives desirable guarantees for downstream decision makers of various sorts.

## C.1. Transparent Policy Evaluation and Its Downstream Benefits

In this subsection, we introduce the main benefit of calibration that underlies, in one form or another, all our following applications. Namely, calibrated (or, as we will see, sufficiently unbiased) predictions result in what we call a *transparent policy evaluation* property, whereby downstream agents' prediction-to-action policies will in hindsight (once the true states are revealed) bring as much utility to the agents as they would get *had the predictions been exactly correct*. In that sense, the predicted states are *indistinguishable* from true states *for the purposes of policy evaluation*. In a nutshell, this enables straightforward downstream optimization of the next action to play while only having access to predicted, rather than realized, quantities — and is the main vehicle that drives our Predict-then-Act approach where agents simply best-respond to the predictions. This general "transparency" property of calibration is also what underlies outcome indistinguishability [\(Dwork et al.,](#page-9-3) [2021\)](#page-9-3) and omniprediction [\(Gopalan et al.,](#page-9-13) [2023a\)](#page-9-13) in 1-dimensional batch settings, and [\(Zhao et al.,](#page-11-2) [2021\)](#page-11-2) gave similar transparency guarantees in multi-dimensional batch settings — but as we will see, will be especially useful for us in problems arising in high-dimensional online settings.

In this and the next subsection, we begin by deriving this transparency property (and its downstream consequences) in the idealized scenario where the predictions sˆ are *exactly* calibrated (i.e., there is no error term). This, of course, is a statistically unachievable goal in high dimensions, and so these results won't yet be implementable. However, this presents no issue for our algorithmic results — which do have error rates — because given access to approximately calibrated or unbiased predictions (which we will generally obtain by invoking our Unbiased Prediction algorithm), bias error propagation through our idealized proof templates is quite easy. Meanwhile, these idealized statements and their proofs provide the core statistical intuition about why our approach works.

We begin by formally defining how we want to evaluate the long-term success of any prediction-to-action policy in our online setting.

Definition C.1 (Policy Evaluation Function). A policy evaluation function is a mapping U : (S → A) × S<sup>∗</sup> × S<sup>∗</sup> → <sup>R</sup>. The interpretation is that for any prediction-to-action policy f : S → A and any two sequences of states s, sˆ ∈ S<sup>∗</sup> with len(s) = len(ˆs) < ∞, U(f, s, s ˆ ) will give the total utility of the policy f evaluated on the ground truth state sequence s when actions are taken according to the predicted state sequence sˆ.

We will usually instantiate this definition as follows. Fixing any time horizon T and any decision maker with utility function u : A × S → R, we will let the evaluation function U be defined as the decision maker's cumulative (total) utility of employing the policy f applied to predictions sˆ across rounds 1, . . . , T, namely,

$$U_u(f, \hat{s}, s) = \sum_{t=1}^T u(f(\hat{s}_t), s_t),$$

where s = (s1, . . . , s<sup>T</sup> ) are the ground truth states and sˆ = (ˆs1, . . . , sˆ<sup>T</sup> ) are the predicted states. Note that due to our assumption that the decision maker's utility u is linear and Lipschitz in its second argument, the policy evaluation function U<sup>u</sup> will be linear and Lipschitz in its last (ground-truth) argument s.

Our main results will all follow from appropriately instantiating our event collection such that it guarantees the following property of our sequence of predicted states, relative to (relevant) prediction-to-action policies.

Definition C.2 (Transparent Policy Evaluation). Fix a prediction-to-action policy f : S → A and a policy evaluation function U : (S → A) × S<sup>∗</sup> × S<sup>∗</sup> → <sup>R</sup>. Consider any two sequences of states s, sˆ ∈ S<sup>∗</sup> with len(s) = len(ˆs) < ∞. We interpret s as the ground truth state sequence, and sˆ as a predicted state sequence.

Then, the predictions sˆ are (f, U, s)*-transparent* if:

$$U(f, \hat{s}, s) = U(f, \hat{s}, \hat{s}).$$

Henceforth, we will keep the ground truth sequence s implicit and refer to predictions sˆ as (f, U)*-transparent*.

This notion of transparency it at the core of our proposal for how to define trustworthy predictions in decision pipelines: If the predictions are (f, Uu)-transparent, then a decision maker with utility u who employs prediction-to-action policy f can safely view the predictions sˆ as exactly coinciding with the ground truth states s — in the sense that she can measure the performance of f (using U) against the predictions rather than against the true states. This is often a useful property on its own — but when it holds for multiple prediction-to-action policies f, then optimizing amongst these policies on the basis of the predicted outcomes implies performance guarantees corresponding to optimality within this benchmark set.

Enforcing Transparency via (Full) Calibration We now see how full calibration lends transparency to *all* prediction-toaction policies with respect to *all* evaluation functions U<sup>u</sup> as defined above.

Theorem C.3 (Calibration Lends Transparency to All Prediction-to-Action Policies). *Fix any time horizon* T*. Consider any ground truth sequence of states* s = (s1, . . . , s<sup>T</sup> ) *and any fully calibrated sequence of predictions* sˆ = (ˆs1, . . . , sˆ<sup>T</sup> )*, meaning that for all* v ∈ {s1, . . . , s<sup>T</sup> }*, it holds that* P t∈[T]:ˆst=v s<sup>t</sup> = v · #{t ∈ [T] : ˆs<sup>t</sup> = v}*. Then,* sˆ *is* (f, Uu)*-transparent for all* f : S → A *and every* u : A × S → R *that is linear in its second argument.*

*Proof.* The proof follows directly by definition of (full) calibration (equality (2)) and by linearity of u in the state (equalities (1) and (3)). Letting n<sup>v</sup> := #{t ∈ [T] : ˆs<sup>t</sup> = v} for any value v ∈ <sup>R</sup>, we get:

$$U_u(f, \hat{s}, s) = \sum_{t=1}^T u(f(\hat{s}_t), s_t) = \sum_{v \in \mathbb{R}} \sum_{t \in [T]: \hat{s}_t = v} u(f(v), s_t) \stackrel{(1)}{=} \sum_{v \in \mathbb{R}} u\left(f(v), \sum_{t \in [T]: \hat{s}_t = v} s_t\right)$$

$$\stackrel{(2)}{=} \sum_{v \in \mathbb{R}} u(f(v), v \cdot n_v) \stackrel{(3)}{=} \sum_{v \in \mathbb{R}} n_v \cdot u(f(v), v) = \sum_{t=1}^T u(f(\hat{s}_t), \hat{s}_t) = U_u(f, \hat{s}, \hat{s}). \quad \square$$

Using Transparency for Downstream Optimization Transparency on its own is valuable insofar as it means that a decision maker can follow a prediction-to-action policy f with respect to predicted outcomes without being surprised about her long-run utility. But as we will now observe, it is also directly useful to decision makers for the purposes of optimizing their prediction-to-action policy. In fact, we will now see that enforcing the transparency property over all policies f in any prediction-to-action policy class F that includes the best-response policy f BR u (·) := BRu(·) lets the decision maker obtain no regret to any other policy in that class F by simply playing f BR u in all rounds (i.e., trusting the predictions and acting accordingly).

Theorem C.4 (Transparency over Policy Class F Implies Best-Response Optimality over F). *Consider a decision maker with utility function* u : A × S → <sup>R</sup>*. Consider any collection of prediction-to-action policies* F ⊆ S<sup>A</sup> *such that the decision maker's best response policy is included in it:* f BR <sup>u</sup> ∈ F*.*

*Suppose the sequence of predictions* sˆ *is* (f, Uu)*-transparent for all* f ∈ F*. Then, committing to* f <sup>u</sup> *gives the decision maker no regret with respect to the policy class* F*:*

$$U_u(f_u^{\text{BR}}, \hat{s}, s) = \max_{f \in \mathcal{F}} U_u(f, \hat{s}, s).$$

*Proof.* Fix any policy f ∈ F. By the definition of transparency (noting that f BR <sup>u</sup> ∈ F) and the definition of the best response policy:

$$U_u(f_u^{\text{BR}}, \hat{s}, s) - U_u(f, \hat{s}, s) = U_u(f_u^{\text{BR}}, \hat{s}, \hat{s}) - U_u(f, \hat{s}, \hat{s}) = \max_{f: S \rightarrow \mathcal{A}} U_u(f', \hat{s}, \hat{s}) - U_u(f, \hat{s}, \hat{s}) \geq 0,$$

implying the desired statement.

As an immediate corollary, we see that full calibration implies that the best response policy is simultaniously optimal for all downstream decision makers, amongst all prediction-to-action policies.

Corollary C.5 (Calibration Implies Global Optimality of Best-Response Policy to All Decision-Makers). *Suppose the predictions* sˆ *are fully calibrated. Then, simultaneously* for all downstream decision makers *(i.e., for all utilities* u : A×S → R*), playing the best-response policy* f BR <sup>u</sup> *gives the decision maker* no regret to all prediction-to-action policies*:*

$$U_u(f_u^{\text{BR}}, \hat{s}, s) = \max_{f: S \rightarrow \mathcal{A}} U_u(f, \hat{s}, s) \quad \text{for all decision makers' utilities } u : \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}.$$

*Proof.* As established, full calibration gives transparency to every decision maker (with any utility u) with respect to the entire class of all prediction-to-action policies Ffull := S <sup>A</sup>. Thus, by the preceding theorem, playing the best-response policy gives no regret to *all* f : S → A.

#### C.2. Transparency via Level-Set Unbiasedness and Swap Regret

Let us re-examine what properties our predictors should have in order to achieve transparency for various prediction-to-action policies. Fix any such policy f : S → A, and suppose for a moment that we only need (f, Uu) transparency for this specific f and for all u : A × S → R.

We already know that calibration — which demands unbiasedness from our predictions conditional on every possible prediction value v ∈ S — is sufficient for this purpose [\(Foster & Vohra,](#page-9-4) [1999\)](#page-9-4). But it is not necessary. Intuitively, this is because the total utility U<sup>u</sup> of policy f does not require such granular predictions for estimation. In particular, consider the collection of level sets of policy f, defined as LS(f) := {f −1 (a)}a∈A. These level sets form a partition of the state space S, but unless the mapping f : S → A is injective (which would necessitate the action space A being at least as complex as the space of predictions S), this partition will be (likely much) less granular than the partition of S into single points as required by full calibration. To determine which action to play, policy f only requires knowledge of the level set that the prediction belongs to, not the exact predicted value — and in this sense, LS(f) provides the right level of granularity over the state space S for us to confidently estimate the total utility of f. We formalize this as follows.

Theorem C.6 (Transparent Policy Evaluation via Level-Set Unbiasedness). *Consider any policy* f : S → A*. Suppose the predictions* sˆ *are unbiased on the level sets of* f*, in the sense that for each level set* V ∈ LS(f) *(note that* V ⊆ S*) it holds that* P t∈[T]:ˆst∈V sˆ<sup>t</sup> − s<sup>t</sup> = 0*. Then, the predictions* sˆ *are* (f, Uu)*-transparent for all possible decision makers' utilities* u : A × S → R*.*

*Proof.* For each level set V ∈ LS(f), let f(V ) ∈ A denote the action to which f maps every prediction in V .

$$\begin{aligned} U_u(f, \hat{s}, s) &= \sum_{t=1}^T u(f(\hat{s}_t), s_t) = \sum_{V \in \text{LS}_f} \sum_{t \in [T]: \hat{s}_t \in V} u(f(\hat{s}_t), s_t) = \sum_{V \in \text{LS}_f} \sum_{t \in [T]: \hat{s}_t \in V} u(f(V), s_t) \\ &= \sum_{V \in \text{LS}_f} u\left(f(V), \sum_{t \in [T]: \hat{s}_t \in V} s_t\right) = \sum_{V \in \text{LS}_f} u\left(f(V), \sum_{t \in [T]: \hat{s}_t \in V} \hat{s}_t\right) = \sum_{V \in \text{LS}_f} \sum_{t \in [T]: \hat{s}_t \in V} u(f(V), \hat{s}_t) \\ &= \sum_{V \in \text{LS}_f} \sum_{t \in [T]: \hat{s}_t \in V} u(f(\hat{s}_t), \hat{s}_t) = \sum_{t=1}^T u(f(\hat{s}_t), \hat{s}_t) = U_u(f, \hat{s}, \hat{s}). \end{aligned}$$

But in fact, this result can be significantly strengthened at no cost. Consider the set Φ<sup>A</sup> = {ϕ : A → A} of all self-maps of the action set A. For reasons that will become clear very soon, we will also refer to a self-map ϕ ∈ Φ<sup>A</sup> as a *swap*. As it turns out, predictions that are unbiased on the level sets LS(f) of a policy f lend transparency not just to the f itself but also to each prediction-to-action policy f<sup>ϕ</sup> that is a post-processing of the map f by a swap ϕ ∈ A, that is, f<sup>ϕ</sup> = ϕ ◦ f.

Theorem C.7 (Level-Set Unbiasedness Gives Transparency under All Swaps). *Consider any policy* f : S → A*. As in the above theorem, suppose the state predictions* sˆ *are unbiased on all level sets* V ∈ LS(f) *of* f*. Then, they are* (ϕ ◦ f, Uu)-transparent for all swaps ϕ : A → A *and for all decision makers' utilities* u : A × S → <sup>R</sup>*.*

*Proof.* Fix any swap ϕ : A → A. Then, the level sets of ϕ ◦ f either coincide with, or are strictly coarser than, the level sets of f. Indeed, viewing LS(f) and LS(ϕ ◦ f) as partitions of S, it is easy to see that LS(f) is a *refinement* of LS(ϕ ◦ f), in the sense that for any V ∈ LS(f) there exists some V ′ ∈ LS(ϕ ◦ f) with V ⊆ V ′ . As a result, each level set of ϕ ◦ f is a disjoint union of one or more level sets of f. Thus, since the predictions sˆ are unbiased on the level sets LS(f) of f, they are also unbiased on the level sets LS(ϕ ◦ f) of ϕ ◦ f, implying by the above theorem that they are (ϕ ◦ f, Uu)-transparent for all decision maker's utilities u.

This strengthened result is very useful because it implies *no swap regret guarantees* for decision makers when the predictions sˆ are unbiased on the *level sets of the decision maker's best-response policy*.

Theorem C.8 (No Swap Regret via Unbiasedness on Best-Response Level Sets). *Fix any decision maker with utility function* u : A × S → R*. Consider her best-response policy* f BR u : S → A*. Then, if the predictions* sˆ *are unbiased on the level sets* LS(f BR u ) *of the best-response policy, the decision maker will obtain* no swap regret *by employing the best-response policy:*

$$U_u \left( f_u^{\text{BR}}, \hat{s}, s \right) = \max_{\phi: \mathcal{A} \rightarrow \mathcal{A}} U_u \left( \phi \circ f_u^{\text{BR}}, \hat{s}, s \right).$$

*Proof.* Since the predictions sˆ are (ϕ ◦ f BR u , Uu)-transparent for all swaps ϕ : A → A, by the definition of the best-response policy, we get:

$$U_u(f_u^{\text{BR}}, \hat{s}, s) - \max_{\phi: \mathcal{A} \rightarrow \mathcal{A}} U_u(\phi \circ f_u^{\text{BR}}, \hat{s}, s) = U_u(f_u^{\text{BR}}, \hat{s}, \hat{s}) - \max_{\phi: \mathcal{A} \rightarrow \mathcal{A}} U_u(\phi \circ f_u^{\text{BR}}, \hat{s}, \hat{s}) = 0. \quad \square$$

## D. Faster Unbiased Prediction for Disjoint Events

In this section we show how to find an ϵ-approximate solution to the minimax problems min max u<sup>t</sup> at all rounds t ∈ [T], defined in Section [2,](#page-2-0) with running time that is polynomial in d, |E|, and log(1/ϵ) in the case in which the events E ∈ E are binary valued and disjoint: for all x, sˆ: P <sup>E</sup>∈E E(x, sˆ) ≤ 1. We will also assume that for every history π and context x, the predictions sˆ that satisfy E(π, x, sˆ) = 1 form a convex set for which we have a polynomial time separation oracle.

Throughout this appendix, we refer to weights w<sup>t</sup> from Section [2](#page-2-0) as qt, and to the randomized strategies s¯<sup>t</sup> from Section [2](#page-2-0) as ψt.

Our goal is to solve for the learner's equilibrium strategy ψ<sup>t</sup> ∈ ∆(S) in the game with utility function

$$u_t(\hat{s}, s) = \sum_{i=1}^d \sum_{\sigma \in \{-1,1\}} \sum_{E \in \mathcal{E}} q_{t,(i,\sigma,E)} \cdot \sigma \cdot E(\pi_{t-1}, x_t, \hat{s}) \cdot (\hat{s}_i - s_i)$$

corresponding to the per-round gain of MsMwC. In other words, we need to approximately solve:

$$\psi_t^* = \operatorname{argmin}_{\psi \in \Delta(\mathcal{S})} \max_{s \in \mathcal{S}} \mathbb{E} [u_t(\hat{s}, s)]. \quad (1)$$

By relaxing the minimization player's domain from S to ∆(S), the set of distributions over predictions, we have made the objective linear (and hence convex/concave), but we have continuously many optimization variables — both primal variables (for the minimization player) and dual variables (for the maximization player). Our strategy for solving this problem in polynomial time will be to argue that it has a solution in which only |E| many primal variables take non-zero values, that we can efficiently identify those variables, and that we can implement a separation oracle for the dual "constraints" in polynomial time. This will allow us to construct a reduced but equivalent linear program that we can efficiently solve with the Ellipsoid algorithm.

We first observe that in the utility function ut(ˆs, s), the learner's predictions sˆ "interact" with the outcomes s only through the activation of the events E(πt−1, xt, sˆ). This implies that *conditional* on the values of the events E(πt−1, xt, sˆ), there is a unique sˆ that minimizes ut(·, s) *simultaneously for all* s. In general, the collection of events E(πt−1, xt, sˆ) could take on many different combinations of values — but our assumption in this section that the events are disjoint and binary means that there are in fact only |E| different candidate values of sˆ for us to consider — namely, those defined by the following efficiently solvable convex programs:

Definition D.1. For E ∈ E, let sˆ ∗,E <sup>t</sup> be a solution to the following convex program (selecting arbitrarily if there are multiple optimal solutions):

$$\begin{aligned} \text{minimize } & \hat{s} \in \mathcal{S} & \sum_{i=1}^d \sum_{\sigma \in \{-1,1\}} q_{t,(i,\sigma,E)} \cdot \sigma \cdot \hat{s}_i \\ \text{subject to} & E(\pi_{t-1}, x_t, \hat{s}) = 1. \end{aligned}$$

Let P<sup>t</sup> = {sˆ ∗,E <sup>t</sup> }E∈E be a collection of |E| vectors in S constituting solutions to the above programs.

*Remark* D.2*.* As we have assumed in this section, the set of sˆ such that E(πt−1, xt, sˆ) = 1 is a convex region endowed with a separation oracle, and so these are indeed convex programs that we can efficiently solve with the Ellipsoid algorithm. This is often the case: for example, if we have a decision maker with a utility function u over K actions, the disjoint binary events Eu,a (for each action a ∈ [K]) are defined by K linear inequalities, and so form a convex polytope with a small number of explicitly defined constraints; this collection of events is relevant for obtaining diminishing swap regret for downstream decision makers.

We next verify that the prediction values defined in Definition [D.1](#page-18-0) are best responses for the minimization player against all possible realizations s<sup>t</sup> that the maximization player might choose, *conditional* on a positive value of a particular event:

Lemma D.3. *Simultaneously for all* s ∈ S*, we have:*

$$\hat{s}_t^{*,E} \in \operatorname{argmin}_{\hat{s}: E(\pi_{t-1}, x_t, \hat{s})=1} u_t(\hat{s}, s).$$

A consequence of this is that solutions to the following reduced minimax problem (which now has only |E| variables for the minimization player — the weights defining a distribution over the |E| points sˆ ∗,E t ) are also solutions to our original minimax problem [1:](#page-18-1)

$$\psi_t^* = \operatorname{argmin}_{\psi \in \Delta(\mathcal{P}_t)} \max_{s \in \mathcal{S}} \mathbb{E} [u_t(\hat{s}, s)]. \quad (2)$$

Lemma D.4. *Fix any optimal solution* ψ ∗ t *to minimax problem [2.](#page-19-0) Then* ψ ∗ t *is also an optimal solution to minimax problem [1.](#page-18-1)*

Thus, to find a solution to minimax problem [1,](#page-18-1) it suffices to find a solution to minimax problem [2.](#page-19-0) Minimax problem [2](#page-19-0) can be expressed as a linear program with |E| + 1 variables but with continuously many constraints, one for each s ∈ S:

$$\begin{aligned} \text{minimize}_{\psi \in \Delta(\mathcal{P}_t)} \quad & \gamma \\ \text{subject to} \quad & \mathbb{E}_{\hat{s} \sim \psi}[u_t(\hat{s}, s)] \leq \gamma \quad \forall s \in \mathcal{S}. \end{aligned} \tag{3}$$

We can find an ϵ-approximate solution to a polynomial-variable linear program using the Ellipsoid algorithm in time polynomial in the number of variables and log(1/ϵ) so long as we have an efficient *separation oracle* — i.e., an algorithm to find an ϵ-violated constraint whenever one exists, given a candidate solution. In this case, implementing a separation oracle corresponds to computing a *best response* for the adversary (the maximization player) in our game—and since the utility function in our game is *linear* in the adversary's chosen action s, implementing a separation oracle corresponds to solving a linear maximization problem over the convex feasible region S—a problem that we can solve efficiently assuming we have a separation oracle for S. There are a number of technical details involved in making this rigorous, which can be found in Appendix [D.](#page-18-2) Here we state the final algorithm and guarantee.

Algorithm 3 Get-Approx-Equilibrium-LP(t, ϵ, E)

for E ∈ E do Solve the convex program from Definition [D.1](#page-18-0) to obtain sˆ ∗,E t .

end for Let P<sup>t</sup> = {sˆ ∗,E <sup>t</sup> }E∈E .

Solve linear program [3](#page-19-1) over P<sup>t</sup> using the weak Ellipsoid algorithm to obtain solution ψ ′ t .

Let ψ ∗ <sup>t</sup> be the Euclidean projection of ψ ′ <sup>t</sup> onto ∆(Pt) returned by the simplex projection algorithm.

Return ψ ∗ t .

Theorem D.5. *Given a polynomial-time separation oracle for* S*, for any* ϵ > 0*, there exists an algorithm (Algorithm [3\)](#page-19-2) that returns an* ϵ*-approximately optimal solution* ψ ∗ t *to minimax problem [1](#page-18-1) and runs in time polynomial in* d*,* |E|, log( <sup>1</sup> ϵ )*.*

We solve for an ϵ-approximate solution of linear program [3](#page-19-1) using a *weak separation oracle*, using an approximate version of the Ellipsoid algorithm.

Definition D.6. For any ϵ > 0 and any convex set S, let

$$S^{+\epsilon} = \{s : \|s - \tilde{s}\|_2 \leq \epsilon \quad \text{for some } \tilde{s} \in S\} \quad S^{-\epsilon} = \{s : B_2(s, \epsilon) \subseteq S\}$$

be the positive and negative ϵ-approximate sets of S, where B2(x, r) is a ball of radius r under the ℓ<sup>2</sup> norm.

Definition D.7. A *weak separation oracle* for a convex set S is an algorithm that, when given input ψ ∈ Q<sup>d</sup> and positive ϵ ∈ Q, confirms that ψ ∈ S +ϵ if true, and otherwise returns a hyperplane a ∈ Q<sup>d</sup> such that ||a||<sup>∞</sup> = 1 and ⟨a, ψ⟩ ≤ ⟨a, ψ′ ⟩ + ϵ for all ψ ′ ∈ S −ϵ .

We express a separation oracle for linear program [3](#page-19-1) as the convex program that solves for the most violated constraint given a candidate solution ψ, which is simply the best response problem for the maximization player in minimax problem [2.](#page-19-0) This is the problem of maximizing a d-variable linear function over the convex set S. To make sure that we can control the bit complexity of the constraint returned by the separation oracle we round the coordinates of the constraint a ∈ R <sup>d</sup> output by the separation oracle to a rational-valued vector within ± ϵ 2 of the exact solution by truncating each coordinate of a to log( <sup>1</sup> ϵ ) bits.

Definition D.8. A solution ψ ∈ S +ϵ is ϵ*-weakly optimal* if, given ϵ > 0, <sup>E</sup>sˆ∼ψ[u(ˆs, s)] ≤ <sup>E</sup>sˆ∼ψ′ [u(ˆs, s)] + ϵ for all ψ ′ ∈ S −ϵ and for all s.

For an ϵ-approximate solution to minimax problem [2,](#page-19-0) it suffices to find an ϵ-weakly optimal solution to linear program [3,](#page-19-1) which we can do using the Ellipsoid method. However, the solution to the weak optimization may not even be a valid probability distribution (since it only approximately satisfies the constraints) – in this case, we can project our infeasible solution back to feasibility. We use the simplex Euclidean projection algorithm given by [\(Condat,](#page-8-8) [2016\)](#page-8-8) to project the candidate solution back to a feasible region and show that this projected feasible solution is still ϵ-approximately optimal.

Theorem D.5. *Given a polynomial-time separation oracle for* S*, for any* ϵ > 0*, there exists an algorithm (Algorithm [3\)](#page-19-2) that returns an* ϵ*-approximately optimal solution* ψ ∗ t *to minimax problem [1](#page-18-1) and runs in time polynomial in* d*,* |E|, log( <sup>1</sup> ϵ )*.*

*Proof.* Linear program [3](#page-19-1) encodes minimax problem [2.](#page-19-0) To solve LP [3,](#page-19-1) we use the Ellipsoid algorithm, which gives an approximate solution in polynomial time under the following conditions:

Theorem D.9 ([\(Grotschel et al.](#page-9-22) ¨ , [1988\)](#page-9-22), Theorem 4.4.7). *Given a weak separation oracle over convex constraint set* S *and* ϵ > 0*, the Ellipsoid algorithm finds a* ϵ*-weakly optimal solution over* S *in time polynomial in the bit complexity of the constraints returned by the separation oracle, the bit complexity of the objective function, and the bit complexity of* ϵ*.*

Fix some ϵ > 0. Let S be the constraint set, which are a set of linear constraints over a convex compact set (i.e. s ∈ S) and constraints enforcing a probability simplex (i.e. ψ ∈ ∆(P)), implying that S is a convex set. Let ϵ ′ = ϵ 2C √ |E| . Given an exact separation oracle over S, preserving log( <sup>1</sup> ϵ ′ ) bits of the most violated constraint given by the separation oracle and rounding to a rational number yields an rational ϵ ′ -approximate most violated constraint, which satisfies the conditions for a weak separation oracle. Thus, we can find an ϵ ′ -weakly optimal solution (γ ′ , ψ′ ) to minimax problem [2,](#page-19-0) where ψ ′ ∈ S +ϵ . In the case that ψ ′ is a valid probability distribution, we have found an ϵ-approximate optimal solution ψ ∗ <sup>t</sup> = ψ ′ .

Otherwise, ψ ′ may violate conditions for a valid probability distribution if the linear constraints do not constrain the feasible set (i.e. S = ∆(P)). Since ψ ′ ∈ S +ϵ , there exists some ψ <sup>ϵ</sup> ∈ S such that ||ψ <sup>ϵ</sup> − ψ ′ || ≤ ϵ ′ . We find this point ψ <sup>ϵ</sup> via the simplex projection algorithm in [\(Condat,](#page-8-8) [2016\)](#page-8-8).

We show that this projection back to a feasible probability distribution still leaves us with an ϵ-approximately optimal solution. Let ut(ˆs ∗ t , s) be the |E|-dimensional vector such that each coordinate E has entry ut(ˆs ∗,E t , s). First, we show that |ut(ˆs ∗,E t , s)| is bounded by C = 2 maxs∈S ||s||<sup>∞</sup> for E ∈ E:

$$\begin{aligned} |u(\hat{s}_t^{*,E}, s)| &\leq \sum_{i=1}^d \sum_{\sigma \in \{-1,1\}} \sum_{E \in \mathcal{E}} q_{t,(i,\sigma,E)} \cdot |\sigma| \cdot E(x_t, \hat{s}_t^{*,E}) \cdot |\hat{s}_{t,i}^{*,E} - s_i| \\ &\leq \sum_{i=1}^d \sum_{\sigma \in \{-1,1\}} \sum_{E \in \mathcal{E}} q_{t,(i,\sigma,E)} \cdot |\hat{s}_{t,i}^{*,E} - s_i| \leq \sum_{i=1}^d \sum_{\sigma \in \{-1,1\}} \sum_{E \in \mathcal{E}} q_{t,(i,\sigma,E)} \cdot C = C, \end{aligned}$$

where we used that E(xt, sˆ ∗,E t ) ≤ 1 and |σ| = 1, and that q ∈ ∆(2d|E|), implying it must sum to 1. From this, we find that ||ut(ˆs ∗ t , s)||<sup>2</sup> ≤ q C<sup>2</sup> <sup>1</sup> + . . . + C<sup>2</sup> |E| ≤ C p |E|.

Next, by continuity of inner product, given ϵ > 0, s ∈ S, there exists δ > 0 such that ||ψ <sup>ϵ</sup> − ψ ′ || ≤ δ implies that || <sup>E</sup>sˆ∼ψ<sup>ϵ</sup> [ut(ˆs, s)] − <sup>E</sup>sˆ∼ψ′ [ut(ˆs, s)]|| ≤ ϵ. By Cauchy-Schwarz, we can bound the difference between the expectations as follows:

$$\left\| \mathbb{E}_{\hat{s} \sim \psi^\epsilon} [u_t(\hat{s}, s)] - \mathbb{E}_{\hat{s} \sim \psi'} [u_t(\hat{s}, s)] \right\|_2 = \langle \psi^\epsilon - \psi^*, u_t(\hat{s}_t^*, s) \rangle \leq \|\psi_t^\epsilon - \psi'\|_2 \cdot \|u_t(\hat{s}_t^*, s)\|_2 \leq \delta \cdot C \sqrt{|\mathcal{E}|}.$$

Thus, using ψ ∗ <sup>t</sup> = ψ ϵ as the solution and setting δ = ϵ ′ gives us an ϵ ′ · C p |E| + ϵ ′ = ϵ <sup>2</sup> + ϵ 2C √ |E| ≤ ϵ approximate solution. By Lemma [D.4,](#page-19-3) any optimal solution to minimax problem [2](#page-19-0) is an optimal solution to minimax problem [1,](#page-18-1) so we must have that ψ ∗ t is an ϵ-approximate solution to minimax problem [1](#page-18-1).

Now we consider the runtime of the algorithm. In order for LP [3](#page-19-1) to be well-formulated, we first solve |E| convex programs (one for each sˆ <sup>∗</sup>,E), which takes time polynomial in d. Now, consider the bit complexity of the constraints. For the inequality constraints, the bit complexity of each constraint bounding the objective function is given by the bit complexity of <sup>E</sup>sˆ∼ψ[u(ˆs, s)]. Each coefficient of ψ(ˆs ∗,E t ) is ut(ˆs ∗,E t , s), which is bounded by C from above. Since there are |E| variables in this constraint, the maximum bit complexity of any constraint is bounded by O(log(C|E|)). Similarly, the objective function has polynomial bit complexity on the scale of O(log(C|E|)). Finally, ϵ has a bit complexity of log( <sup>1</sup> ϵ ). The simplex projection algorithm has quadratic runtime in the dimension of the vector, which takes O(|E|<sup>2</sup> ) time. Thus, the runtime of the algorithm is polynomial in d, |E|, log(C|E|), and log( <sup>1</sup> ϵ ).

Lemma D.3. *Simultaneously for all* s ∈ S*, we have:*

$$\hat{s}_t^{*,E} \in \operatorname{argmin}_{\hat{s}: E(\pi_{t-1}, x_t, \hat{s})=1} u_t(\hat{s}, s).$$

*Proof.* The constraint that E(πt−1, xt, sˆ) = 1 together with the fact that the set of events E is disjoint and binary implies that for all other events E′ ∈ E, E′ (πt−1, xt, sˆ) = 0. For any sˆ such that E(πt−1, xt, sˆ) = 1, we therefore have that ut(ˆs, s) reduces to:

$$u_t(\hat{s}, s) = \sum_{i=1}^d \sum_{\sigma \in \{-1,1\}} q_{t,(i,\sigma,E)} \cdot \sigma \cdot \hat{s}_i - q_{t,(i,\sigma,E)} \cdot \sigma \cdot s_i.$$

But in this expression, the sˆ terms have no interaction with the s terms, and hence we have that for any s:

$$\begin{aligned} \argmin_{\hat{s}: E(\pi_{t-1}, x_t, \hat{s})=1} u_t(\hat{s}, s) &= \argmin_{\hat{s}: E(\pi_{t-1}, x_t, \hat{s})=1} \left( \sum_{i=1}^d \sum_{\sigma \in \{-1, 1\}} q_{t, (i, \sigma, E)} \cdot \sigma \cdot \hat{s}_i \right) = \hat{s}_t^{*, E}. \end{aligned} \quad \square$$

Lemma D.4. *Fix any optimal solution* ψ ∗ t *to minimax problem [2.](#page-19-0) Then* ψ ∗ t *is also an optimal solution to minimax problem [1.](#page-18-1)*

*Proof.* We first observe that minimax problem [2](#page-19-0) is only a more constrained problem for the minimization player than minimax problem [1,](#page-18-1) as P<sup>t</sup> ⊂ S. Thus it suffices to show that given a solution ψˆ <sup>t</sup> for minimax problem [1,](#page-18-1) we can transform it into a new solution ψ<sup>t</sup> such that:

- 1. ψ<sup>t</sup> has support only over points in Pt, and
- 2. For all s ∈ S, <sup>E</sup>sˆt∼ψ<sup>t</sup> [u(ˆst, s)] ≥ <sup>E</sup>sˆt∼ψˆ<sup>t</sup> [u(ˆst, s)].

Given ψˆ <sup>t</sup>, we construct ψ<sup>t</sup> as follows: for each event E, we take all of the weight that ψˆ <sup>t</sup> places on points sˆ such that E(πt−1, xt, sˆ) = 1, and place that weight on sˆ ∗,E <sup>t</sup> ∈ Pt:

$$\psi_t(\hat{s}_t^{*,E}) = \hat{\psi}_t(\{\hat{s} : E(\pi_{t-1}, x_t, \hat{s}) = 1\}).$$

By construction ψ<sup>t</sup> has support over points in Pt. It remains to show that ψ<sup>t</sup> has objective value that is at least as high as ψˆ t for every s ∈ S:

$$\begin{aligned} \mathbb{E}_{\hat{s}_t \sim \psi_t} [u(\hat{s}_t, s)] &= \sum_{E \in \mathcal{E}} \Pr_{\hat{s}_t \sim \psi_t} [E(\pi_{t-1}, x_t, \hat{s}_t) = 1] \mathbb{E}_{\hat{s}_t \sim \psi_t} [u(\hat{s}_t, s) | E(\pi_{t-1}, x_t, \hat{s}_t) = 1] \\ &\leq \sum_{E \in \mathcal{E}} \Pr_{\hat{s}_t \sim \psi_t} [E(\pi_{t-1}, x_t, \hat{s}_t) = 1] u(\hat{s}_t^*, E, s) \\ &= \mathbb{E}_{\hat{s}_t \sim \psi_t} [u(\hat{s}_t, s)] \end{aligned}$$

The inequality follows from Lemma [D.3.](#page-18-3)