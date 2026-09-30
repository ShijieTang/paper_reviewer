# High-Dimensional Calibration from Swap Regret

Maxwell Fishelson<sup>∗</sup> maxfish@mit.edu

Noah Golowich† nzg@mit.edu

Mehryar Mohri‡ mohri@google.com

Jon Schneider§ jschnei@google.com

# Abstract

We study the online calibration of multi-dimensional forecasts over an arbitrary convex set P ⊂ R d relative to an arbitrary norm k·k. We connect this with the problem of external regret minimization for online linear optimization, showing that if it is possible to guarantee O( √ ρT) worst-case regret after T rounds when actions are drawn from P and losses are drawn from the dual k·k<sup>∗</sup> unit norm ball, then it is also possible to obtain -calibrated forecasts after T = exp(O(ρ/<sup>2</sup> )) rounds. When P is the d-dimensional simplex and k·k is the `1-norm, the existence of O( √ T log d)-regret algorithms for learning with experts implies that it is possible to obtain -calibrated forecasts after T = exp(O(log d/<sup>2</sup> )) = d O(1/<sup>2</sup> ) rounds, recovering a recent result of [\[Pen25\]](#page-10-0).

Interestingly, our algorithm obtains this guarantee without requiring access to any online linear optimization subroutine or knowledge of the optimal rate ρ – in fact, our algorithm is identical for every setting of P and k·k. Instead, we show that the optimal regularizer for the above OLO problem can be used to upper bound the above calibration error by a swap regret, which we then minimize by running the recent TreeSwap algorithm ([\[DDFG24,](#page-9-0) [PR24\]](#page-10-1)) with Follow-The-Leader as a subroutine. The resulting algorithm is highly efficient and plays a distribution over simple averages of past observations in each round.

Finally, we prove that any online calibration algorithm that guarantees T `1 calibration error over the d-dimensional simplex requires T ≥ exp(poly(1/)) (assuming d ≥ poly(1/)). This strengthens the corresponding d Ω(log 1/) lower bound of [\[Pen25\]](#page-10-0), and shows that an exponential dependence on 1/ is necessary.

# 1 Introduction

Consider the problem faced by a forecaster who must report probabilistic predictions for a sequence of events (e.g. whether it will rain or not tomorrow). One of the most common methods to evaluate the quality of such a forecaster is to verify whether they are *calibrated*: for example, does it indeed rain with probability 40% on days where the forecaster makes this prediction? In addition to calibration being a natural property to expect from predictions, several applications across machine learning, fairness, and game theory require the ability to produce online calibrated predictions [\[ZME20,](#page-11-0) [GPSW17,](#page-9-1) [HJKRR18,](#page-10-2) [FV97\]](#page-9-2).

When events have binary outcomes, calibration can be quantified by the notion of *expected calibration error*, which measures the expected distance between a prediction made by a forecaster and the actual empirical probability of the outcome on the days where they made that prediction. In a seminal result by Foster and Vohra [\[FV98\]](#page-9-3), it was proved that it is possible for an online forecaster to efficiently

<sup>∗</sup>MIT.

<sup>†</sup>MIT. Supported by a NSF Graduate Research Fellowship and a Fannie & Hertz Foundation Graduate Fellowship.

<sup>‡</sup>Google Research and Courant Institute of Mathematical Sciences, New York.

<sup>§</sup>Google Research.

guarantee a sublinear calibration error of O(T 2/3 ) against any adversarial sequence of T binary events. Equivalently, this can be interpreted as requiring at most O( −3 ) rounds of forecasting to guarantee an per-round calibration error on average.

However, many applications require forecasting sequences of *multi-dimensional* outcomes. The previous definition of calibration error easily extends to the multi-dimensional setting where predictions and outcomes belong to a d-dimensional convex set P ⊂ R d . Specifically, if a forecaster makes a sequence of predictions p1, p2, . . . , p<sup>T</sup> ∈ P for the outcomes y1, y2, . . . , y<sup>T</sup> ∈ P, their k·k-calibration error (for any norm k·k over <sup>R</sup> d ) is given by

$$\text{Cal}_T^{\|\cdot\|} = \sum_{t=1}^T \|p_t - \nu_{p_t}\|$$

where ν<sup>p</sup><sup>t</sup> is the average of the outcomes y<sup>t</sup> on rounds where the learner predicted pt.

The algorithm of Foster and Vohra extends to the multidimensional calibration setting, but at the cost of producing bounds that decay exponentially in the dimension d. In particular, their algorithm only guarantees that the forecaster achieves an average calibration error of after (1/) Ω(d) rounds. Until recently, no known algorithm achieved a sub-exponential dependence on d in any non-trivial instance of multi-dimensional calibration.

In 2025, [\[Pen25\]](#page-10-0) presented a new algorithm for high-dimensional calibration, demonstrating that it is possible to obtain `1-calibration rates of T in d O(1/<sup>2</sup> ) rounds for predictions over the d-dimensional simplex (i.e., multi-class calibration). In particular, this is the first known algorithm achieving polynomial calibration rates in d for fixed constant . [\[Pen25\]](#page-10-0) complements this with a lower bound, showing that in the worst case d Ω(log 1/) rounds are necessary to obtain this rate (implying that a fully polynomial bound poly(d, 1/) is impossible).

### 1.1 Our results

Although the algorithm of [\[Pen25\]](#page-10-0) is simple to describe, its analysis is fairly nuanced and tailored to `1-calibration over the simplex (e.g., by analyzing the KL divergence between predictions and distributions of historical outcomes). We present a very similar algorithm (TreeCal) for multidimensional calibration over an arbitrary convex set P ⊂ R d , but with a simple, unified analysis that provides simultaneous guarantees for calibration with respect to any norm k·k. In particular, we prove the following theorem.

Theorem 1.1 (Informal restatement of Corollary [C.5\)](#page-26-0). *Fix a convex set* P *and a norm* k · k*. Assume there exists a function* R : P → <sup>R</sup> *that is* 1*-strongly-convex with respect to* k · k *and has range (*maxx∈P R(x) − minp∈P R(x)*) at most* ρ*. Then* TreeCal *guarantees that the calibration error of its predictions is bounded by* Calk·k <sup>T</sup> ≤ T *for* T ≥ (diamk·k(P)/) O(ρ/<sup>2</sup> ) *.*

Interestingly, the function R(p) and parameter ρ appearing in the statement of Theorem [1.1](#page-1-0) have an independent learning-theoretic interpretation: if we consider the *online linear optimization* problem where a learner plays actions in P and the adversary plays linear losses that are unit bounded in the dual norm k·k<sup>∗</sup> , then it is possible for the learner to guarantee a regret bound of at most O( √ ρT) by playing Follow-The-Regularized-Leader (FTRL) with R(p) as a regularizer. In fact, since universality results for mirror descent guarantee that some instantiation of FTRL achieves near-optimal rates for online linear optimization (as long as the action and loss sets are centrally convex) [\[SST11,](#page-11-1) [GSJ24\]](#page-10-3), this allows us to relate the performance of Theorem [3.1](#page-7-0) directly to what rates are possible in online linear optimization.

Corollary 1.2 (Informal restatement of Corollary [C.6\)](#page-27-0). *Let* P ⊆ R <sup>d</sup> *be a centrally symmetric convex set, and let* L = {y ∈ <sup>R</sup> d | kyk<sup>∗</sup> ≤ 1} *for some norm* k·k*. Then if there exists an algorithm for online linear optimization with action set* P *and loss set* L *that incurs regret at most* O( √ ρT)*,* TreeCal *guarantees that the calibration error of its predictions is bounded by* Calk·k <sup>T</sup> ≤ T *for* T ≥ (diamk·k(P)/) O(ρ/<sup>2</sup> ) *.*

- When P is the d-simplex ∆<sup>d</sup> and k·k is the `1-norm, the existence of the negative entropy regularizer R(x) = P<sup>d</sup> <sup>i</sup>=1 x<sup>i</sup> log x<sup>i</sup> (which is 1-strongly convex w.r.t. the `<sup>1</sup> norm with range ρ = log d) implies that the `<sup>1</sup> calibration error of TreeCal is at most (1/) O(log d/<sup>2</sup> ) = d O˜(1/<sup>2</sup> . This recovers the result of [\[Pen25\]](#page-10-0).
- When P is the `<sup>2</sup> ball and k·k is the `<sup>2</sup> norm, the Euclidean regularizer (R(x) = kxk 2 ) implies a calibration bound of (1/) O(1/<sup>2</sup> ) (notably, this bound is independent of d).

It should be emphasized here that running TreeCal does not require any online linear optimization subroutine, nor any knowledge of these regularizers R(x) or optimal rates ρ. TreeCal has no functional dependence on any specific k·k. It achieves k·k-calibration at the above rate (Theorem [1.1\)](#page-1-0) for all k·k simultaneously. The TreeCal algorithm is nearly identical[<sup>5</sup>](#page-2-0) to the algorithm of [\[Pen25\]](#page-10-0) – both algorithms initialize a tree of sub-forecasters and at each round play a uniform combination of some subset of them (see Figure [1\)](#page-6-0).

The novelty in our analysis stems from the observation that TreeCal is simply a specific instantiation of the TreeSwap swap regret minimization algorithm [\[DDFG24,](#page-9-0) [PR24\]](#page-10-1) and can be analyzed directly in this way. In particular, our analysis consists of the following steps:

- 1. First, minimizing calibration error can be reduced to minimizing swap regret, generalizing an idea of [\[LSS25,](#page-10-4) [FKO](#page-9-4)<sup>+</sup>25]. That is, it is possible to assign the learner loss functions `<sup>t</sup> : P → <sup>R</sup> at each round such that their calibration error is upper bounded by the gap between the total loss they received, and the minimal loss they could have received after applying an arbitrary "swap function" π : P → P to their predictions. In fact, any strongly convex function R (w.r.t. the norm k·k) gives rise to one such reduction, by setting the loss function `t(p) to equal the Bregman divergence DR(yt|p).
- 2. Second, the TreeSwap algorithm of [\[DDFG24,](#page-9-0) [PR24\]](#page-10-1) provides a general recipe for converting external regret minimization algorithms into swap regret minimization algorithms. We obtain TreeCal by plugging in the Follow-The-Leader algorithm (the learning algorithm which simply always best responds to the current history) into TreeSwap.
- 3. Instead of analyzing the swap regret bound of TreeSwap with Follow-The-Leader (which may not have a good enough external regret bound, as discussed in Section [3.3\)](#page-7-1), we instead analyze the swap regret of TreeSwap with *Be-The-Leader* (the fictitious algorithm that best responds to the current history, including the current round). Though it is not possible to actually implement Be-The-Leader due to its clairvoyance, we use it as a tool for analysis. We then relate the calibration error of TreeSwap with *Be-The-Leader* to that of TreeSwap with *Follow-The-Leader* using the fact that Be-The-Leader and Follow-The-Leader make similar predictions.

In the above step 1, we will choose R to be k·k-norm 1-strongly convex, which guarantees that DR(y|p) ≥ ky − pk 2 . Going through the analysis, this actually leads to the stronger guarantee that TreeCal minimizes *squared-norm* calibration error.

Theorem 1.3 (Informal restatement of Theorem [3.1\)](#page-7-0). *Fix a convex set* P *and a norm* k · k*. Assume there exists a function* R : P → <sup>R</sup> *that is* 1*-strongly-convex with respect to* k · k *and has range (*maxx∈P R(x) − minp∈P R(x)*) at most* ρ*. Then* TreeCal *guarantees that the calibration error of its predictions is bounded by* Calk·k<sup>2</sup> <sup>T</sup> ≤ T *for* T ≥ (diamk·k(P)/ √ ) O(ρ/) *.*

Note here we have only singly-exponential dependence on 1/. We arrive at Theorem [1.1](#page-1-0) as a corollary of this result by simply applying Cauchy-Schwarz. Finally, we strengthen the lower bound of [\[Pen25\]](#page-10-0) by showing an exponential dependence on 1/ is necessary.

Theorem 1.4 (Informal restatement of Theorem [4.3\)](#page-8-0). *There is a sufficiently small constant* c > 0 *so that the following holds. Fix any* > 0, d ∈ N*. Then for any* T ≤ exp(c · min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>6</sup>})*, there is an oblivious adversary producing a sequence of outcomes so that any learning algorithm must incur* `1*-calibration error* Calk·k<sup>1</sup> <sup>T</sup> ≥ · T.

<sup>5</sup>One minor difference is that the algorithm of [\[Pen25\]](#page-10-0) regularizes each sub-forecaster by slightly mixing their prediction with the uniform distribution, which TreeCal does not require.

Unlike the lower bound of [\[Pen25\]](#page-10-0), this lower bound requires no specialized construction. Instead, it follows from the original observation of [\[FV98\]](#page-9-3) that any algorithm for online calibration can be used to construct an algorithm for swap regret minimization by simply best responding to a sequence of calibrated predictions of the adversary's losses. The existing lower bound for swap regret in [\[DFG](#page-9-5)<sup>+</sup>24] then immediately precludes the existence of sufficiently strong calibration bounds (e.g., of the form d O(log 1/) , which was still allowed by the work of [\[Pen25\]](#page-10-0)).

Using a similar technique, in Theorem [D.2,](#page-29-0) we show a similar lower bound for `<sup>2</sup> calibration, namely that exp(Ω(min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>7</sup>})) time steps are needed to achieve `<sup>2</sup> calibration error at most · T. For d ≥ −2 , this bound is tight up a polynomial in the exponent.

We discuss additional related work in the appendix.

# 2 Setup

For a positive integer n, we let [0 : n − 1] denote the sequence 0, 1, . . . , n − 1, and [n] denote the sequence 1, 2, . . . , n. We say a convex set S ⊆ R d is *centrally symmetric* if s ∈ S ⇔ −s ∈ S for all s ∈ R d . A norm k·k is a function corresponding to a convex, bounded, centrally-symmetric set S of the form ksk = inf {c ∈ <sup>R</sup>≥0|s ∈ cS}. The corresponding *dual norm* is defined kvk<sup>∗</sup> = sup {hs, vi | ksk ≤ 1}.

### 2.1 Calibration

We consider the following setting of *multi-dimensional calibration*. Positive integers d ∈ N representing the number of dimensions and T ∈ N representing the number of rounds are given. We let P ⊂ R <sup>d</sup> denote a bounded convex subset of <sup>R</sup> d . An *adversary* and a *learning algorithm* interact for a total of T timesteps; at each time step t ∈ [T]:

- The learning algorithm chooses a distribution[<sup>6</sup>](#page-3-0) x<sup>t</sup> ∈ ∆(P) with finite support.
- The adversary observes x<sup>t</sup> and chooses an *outcome* y<sup>t</sup> ∈ P.

In order for the learner to be calibrated, we would like the average outcome conditional on the learner making a specific prediction p to be "close" to p. We formalize this as follows. For a point p ∈ P, we define ν<sup>p</sup> to be the average outcome conditioned on the learner predicting p, that is:

$$\nu_p := \frac{\sum_{t=1}^T \mathbf{x}_t(p) \cdot y_t}{\sum_{t=1}^T \mathbf{x}_t(p)}. \quad (1)$$

Fix a *distance measure* D : P × P → <sup>R</sup>≥0, namely an arbitrary non-negative valued function on P × P. Given a distance measure D, we define the D*-calibration error* as follows:

$$\text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}) := \sum_{p \in \mathcal{P}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot D(\nu_p, p).$$

In the event that <sup>D</sup>(p, q) = k<sup>p</sup> − <sup>q</sup>k, we will write Calk·k T (x1:<sup>T</sup> , y1:<sup>T</sup> ) = Cal<sup>D</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ), and we define Calk·k<sup>2</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) analogously.

### 2.2 Regret minimization

For a sequence of actions p1, · · · , p<sup>T</sup> ∈ P and loss functions `1, · · · , `<sup>T</sup> : P → <sup>R</sup>, we define

$$\text{ExtReg}_T(p_{1:T}, \ell_{1:T}) := \sup_{p^* \in \mathcal{P}} \sum_{t=1}^T \sum_{p \in \mathcal{P}} \ell_t(p_t) - \ell_t(p^*)$$

<sup>6</sup> Some authors refer to this setting as "pseudo-calibration" or "distributional calibration", and reserve the term "calibration" for the setting where the learner is required to randomly select a pure forecast p<sup>t</sup> ∈ P each round instead of a distribution. In Appendix [E](#page-30-0) we describe how to extend our results to this pure-strategy setting of calibration.

For a sequence of distributions x1, · · · , x<sup>T</sup> ∈ ∆(P) and loss functions `1, · · · , `<sup>T</sup> : P → <sup>R</sup>, we define

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) := \sup_{\pi: \mathcal{P} \rightarrow \mathcal{P}} \sum_{t=1}^T \sum_{p \in \mathcal{P}} \mathbf{x}_t(p) \cdot (\ell_t(p) - \ell_t(\pi(p))). \quad (2)$$

Here, we adopt the convention of [\[FKO](#page-9-4)<sup>+</sup>25], referring to the latter quantity as *Full* Swap Regret to emphasize that we consider *all* swap transformations π : P → P (instead of e.g. just linear transformations π).

Throughout, we consider the performance of *regret minimizing* algorithms. These algorithms sequentially map loss functions `1, · · · , `<sup>T</sup> to actions p1, · · · , p<sup>T</sup> or action distributions x1, · · · , x<sup>T</sup> with the goal of minimizing the above quantities. We consider the performance of these algorithms on adversarially selected loss functions from a set L. Abusing notation slightly, for an external regret minimizing algorithm Alg : L <sup>T</sup> → P<sup>T</sup> , we define

$$\text{ExtReg}_T(\text{Alg}) := \sup_{\ell_{1:T} \in \mathcal{L}^T} \text{ExtReg}_T(\text{Alg}(\ell_{1:T}), \ell_{1:T}) \quad (3)$$

and for a full swap regret minimizing algorithm Alg : L <sup>T</sup> → ∆(P) T , we define

$$\text{FullSwapReg}_T(\text{Alg}) := \sup_{\ell_{1:T} \in \mathcal{L}^T} \text{FullSwapReg}_T(\text{Alg}(\ell_{1:T}), \ell_{1:T}).$$

We will denote the tth action played by Alg on a sequence of losses `1:<sup>T</sup> by Alg<sup>t</sup> (`1:<sup>T</sup> ). One important subclass of external regret minimization problems is the setting of *online linear optimization (OLO)*, where all loss functions in ` are linear. Here we slightly abuse notation and identify L with a subset of R d (with the understanding that an element ` ∈ L refers to the linear loss function `(p) = hp, `i). Although we will never actually employ any OLO algorithms themselves, the calibration bounds we obtain will be closely related to optimal regret bounds for instances of OLO (we discuss this further in Section [2.4\)](#page-5-0).

### 2.3 From swap regret to calibration

As noted in [\[LSS25,](#page-10-4) [FKO](#page-9-4)<sup>+</sup>25], calibration with a distance measure D that corresponds to a *Bregman divergence* can be written as a full swap regret with loss functions given by the associated *proper scoring rule*. Given a convex function R : P → R, the *Bregman divergence* associated to R, D<sup>R</sup> : P × P → <sup>R</sup>≥0, is defined as[<sup>7</sup>](#page-4-0)

$$D_R(y|p) := R(y) - R(p) - \langle \nabla R(p), y - p \rangle$$

Geometrically, this divergence is defined by taking the hyperplane tangent to R at p and computing the difference in height between R and the hyperplane at y (see Figure [2\)](#page-19-0).

When viewed as a loss function in p, the Bregman divergence DR(y|p) also has the property that it is a *proper scoring rule*. This refers to the fact that if y is drawn from some distribution y ∈ ∆(P), the optimal response p (to minimize the expected loss DR(y|p)) is simply the expectation y¯ = <sup>E</sup>y∼y[y]. In particular, we have the following lemma.

Lemma 2.1. *For any* y ∈ ∆(P) *and convex function* R : P → <sup>R</sup>*, let* y¯ = <sup>E</sup>y∼y[y]*. and* R(y) = <sup>E</sup>y∼y[R(y)]*. For all* p ∈ P*,* <sup>E</sup>y∼y[DR(y|p)] = DR(¯y|p) + R(y) − R(¯y). *In particular,* `(p) = <sup>E</sup>y∼y[DR(y|p)] *is minimized at* p = ¯y *at a value of* R(y) − R(¯y) *[\(Figure 3\)](#page-20-0).*

This implies the following connection between full swap regret and calibration.

Lemma 2.2. *Fix any convex function* R : P → R*. For any sequence of distributions* x1, x2, . . . , x<sup>T</sup> ∈ ∆(P) *and outcomes* y1, y2, . . . , y<sup>T</sup> ∈ P*, define the sequence of loss functions* `1, `2, . . . , `<sup>T</sup> *via* `t(p) = DR(yt|p)*. Then,*

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) = \text{Cal}_T^{D_R}(\mathbf{x}_{1:T}, y_{1:T}).$$

The proofs of Lemmas [2.1](#page-4-1) and [2.2](#page-4-2) may be found in Appendix [B.](#page-19-1)

In the event that R is not differentiable, we can replace the ∇R(p) term with any element of the sub-gradient at p. When P is not open and p is on the boundary, the ∇R(p) term represents the inward directional gradient.

### 2.4 Rates and regularization

In order to reduce our general calibration problem to a swap regret minimization problem (via Lemma [2.2\)](#page-4-2), we will need to construct a convex function R whose Bregman divergence upper bounds our distance measure. It turns out that the optimal choice of such a function is closely related to the design of optimal regularizers for online linear optimization. In this section, we describe this functional optimization problem and detail this connection.

We say that a convex function R : P → <sup>R</sup> is α*-strongly convex* with respect to a given norm k·k if for any points y, p ∈ P it is the case that R(y) ≥ R(p)+h∇R(p), y−pi+α ky − pk 2 . Equivalently, the Bregman divergence must satisfy DR(y|p) ≥ α ky − pk 2 . Thus, k·k<sup>2</sup> -calibration error is bounded by DR-calibration error if R is k·k-norm 1-strongly convex.

Our later analysis will need not only R to be strongly convex with respect to our norm, but for the Bregman divergence to have a small maximal value. Motivated by this, we will say that a convex function R : P → <sup>R</sup> has *rate* ρ with respect to a given norm k·k if: (1) R is 1-strongly convex with respect to k·k, and (2) the range of the Bregman divergence is at most ρ, i.e., maxy,p∈P DR(y|p) ≤ ρ. We define Rate(P, k·k) to be the infimum of the rates of all 1-strongly convex functions R : P → <sup>R</sup>.

As mentioned earlier, we call this quantity a "rate" due to its connection with the optimal regret rates for online linear optimization. For a learning algorithm Alg : L <sup>T</sup> → P<sup>T</sup> , we defined (in [\(3\)](#page-4-3)) ExtReg<sup>T</sup> (Alg) to be the worst-case regret against any sequence `1:<sup>T</sup> of T losses. It is known that for any fixed action set and loss set, the optimal worst-case regret bound is of the form p RateOLO(P,L) · T + o( √ T), for some constant RateOLO(P,L). Formally, we define RateOLO(P,L) = lim supT→∞ infAlg T · ExtReg<sup>T</sup> (Alg) 2 .

One important class of learning algorithms for online linear optimization is the class of Follow-The-Regularized-Leader (FTRL) algorithms. Each algorithm in this class is specified by a convex "regularizer" function <sup>R</sup> : P → <sup>R</sup>, and at round <sup>t</sup> selects the action <sup>p</sup><sup>t</sup> = argminp∈P P<sup>t</sup>−<sup>1</sup> <sup>s</sup>=1 hp, `ti + R(p). The work of [\[SST11\]](#page-11-1) and [\[GSJ24\]](#page-10-3) shows that there always exists some instantiation of FTRL which achieves (up to a universal constant factor) the optimal regret rate of p RateOLO(P,L) · T + o( √ T) defined above. Moreover, the optimal regularizer for this instance can be constructed by solving a similar functional optimization problem over strongly convex regularizers R, as described in the following theorem.

Theorem 2.3. *Let* P *and* L *be centrally symmetric convex sets. Then, if the function* R : P → R *is 1 strongly-convex with respect to the norm* k·kL<sup>∗</sup> *and has range* ρ *(i.e.,* maxp∈P R(p)−minp∈P R(p) = ρ*), then* RateOLO(P,L) ≤ ρ*. Conversely, there exists a function* R : P → <sup>R</sup> *that is* 1*-strongly-convex with respect to* k·kL<sup>∗</sup> *and has range* O(RateOLO(P,L))*.*

*Proof.* The first result (that RateOLO(P,L) ≤ ρ) follows from the standard analysis of FTRL – see e.g. Theorem 5.2 in [H <sup>+</sup>[16\]](#page-10-5). The converse result follows from Theorem 2 of [\[GSJ24\]](#page-10-3).

Theorem [2.3](#page-5-1) allows us to relate the quantity Rate(P, k·k) to the quantity RateOLO(P,L) (where L is chosen to be the unit dual norm ball). Note that there is a slight difference in the two functional optimization problems defined above – the one for Rate(P, k·k) asks us to bound the range of the Bregman divergence of R, while the one for RateOLO(P,L) asks us to bound the range of R itself. While these two quantities do not directly bound each other (the negative entropy function R(p) = Pp<sup>i</sup> log p<sup>i</sup> has bounded range over the simplex but unbounded Bregman divergence), we can nonetheless show that optimal solutions to one problem can be used to construct optimal solutions to the other problem of similar quality.

Lemma 2.4. *If the action set* P *is centrally symmetric and* L = {y ∈ <sup>R</sup> d | kyk<sup>∗</sup> ≤ 1} *(i.e., the unit ball in the dual norm to* k·k*), then* RateOLO(P,L) = Θ(Rate(P, k·k))*.*

# 3 Main result

We now describe our main algorithm for calibration, TreeCal (Algorithm [1\)](#page-22-0). As we will see, it is equivalent to the TreeSwap algorithm for Full Swap Regret minimization ([\[DDFG24,](#page-9-0) [PR24\]](#page-10-1); Algorithm [2\)](#page-23-0), where the loss functions are given by appropriate Bregman divergences as determined by

Lemma [2.2.](#page-4-2) Moreover, TreeCal is effectively the same as the main algorithm of [\[Pen25\]](#page-10-0). However, the perspective that TreeCal can be viewed as a particular instance of TreeSwap (Lemma [3.2\)](#page-7-2) is novel to this work, and it enables us to tackle a much more general set of calibration problems (Theorem [3.1\)](#page-7-0). We first describe the TreeCal and TreeSwap algorithms, then state Theorem [3.1](#page-7-0) which establishes our main upper bound for TreeCal, and finally discuss the proof of Theorem [3.1,](#page-7-0) which uses the TreeSwap algorithm as a tool in the analysis.

### 3.1 Algorithm description

Given some number of rounds T ∈ N, TreeCal and TreeSwap sequentially produce distributions x1, · · · , x<sup>T</sup> ∈ ∆(P). TreeCal receives from the adversary an outcome sequence y1, · · · , y<sup>T</sup> ∈ P whereas TreeSwap receives loss functions `1, · · · , `<sup>T</sup> : P → <sup>R</sup>.

To describe how the algorithms use the adversary's actions to produce the distributions xt, we need some additional ntation. The algorithms take as input parameters H, L ∈ N satisfying H ≥ 2 and H<sup>L</sup>−<sup>1</sup> ≤ T ≤ H<sup>L</sup>. We index time steps t ∈ [T] via base-H L-tuples: in particular, for t ∈ [T], we let t1, . . . , t<sup>L</sup> ∈ [0 : H − 1] be the base-H representation of t − 1; we will write t − 1 = (t1t<sup>2</sup> · · ·tL). For all 0 ≤ l ≤ L, for all k ∈ [0 : H − 1]<sup>l</sup> , let Γ (l) <sup>k</sup> ⊂ [T] represent the interval of times t with prefix k. That is, t ∈ Γ (l) k iff t<sup>i</sup> = k<sup>i</sup> for all i ∈ [1 : l]. These intervals may be arranged to form an H-ary depth-L tree, where the children of Γ (l) k are Γ (l+1) k0 , Γ (l+1) k1 , · · · , Γ (l+1) k,H−1 . [8](#page-6-1)

Both TreeCal and TreeSwap operate by assigning an action p (l) k to each node Γ (l) k of the tree, except the root. At time t, both algorithms return the uniform distribution over the actions on the root-to-leaf-<sup>t</sup> path, namely <sup>x</sup><sup>t</sup> := Unif n<sup>p</sup> (1) t1 , p (2) t1t<sup>2</sup> , · · · , p (L) t1t2···t<sup>L</sup> o (see [Figure 1\)](#page-6-0). The algorithms differ in how the actions p (l) k are chosen:

|       | (2)   |     |   |  | (1) p 0 (2) |  |       |     | k (2) |     |   |     | (2)    |     |  |     | t (1) p 1 (2) |
|-------|-------|-----|---|--|-------------|--|-------|-----|-------|-----|---|-----|--------|-----|--|-----|---------------|
|       | 00    |     |   |  | p           |  |       |     |       |     |   |     |        |     |  |     |               |
|       |       |     |   |  | 01          |  |       |     | p     |     |   |     |        |     |  |     |               |
|       |       |     |   |  |             |  |       |     | 02    |     |   |     | p      |     |  |     |               |
| (3)   | p (3) |     |   |  | (3)         |  |       |     | (3)   | (3) |   |     | 10 (3) |     |  | (3) | p (3)         |
| 000 p |       |     |   |  |             |  |       |     |       |     |   |     |        |     |  |     |               |
|       | 001 p |     |   |  |             |  |       |     |       |     |   |     |        |     |  |     |               |
|       |       | 002 | p |  |             |  |       |     |       |     |   |     |        |     |  |     |               |
|       |       |     |   |  | 010 p       |  |       |     |       |     |   |     |        |     |  |     |               |
|       |       |     |   |  | 011 p       |  |       |     |       |     |   |     |        |     |  |     |               |
|       |       |     |   |  |             |  | 012 p |     |       |     |   |     |        |     |  |     |               |
|       |       |     |   |  |             |  |       | 020 | p     |     |   |     |        |     |  |     |               |
|       |       |     |   |  |             |  |       |     | 021   | p   |   |     |        |     |  |     |               |
|       |       |     |   |  |             |  |       |     |       | 022 | p |     |        |     |  |     |               |
|       |       |     |   |  |             |  |       |     |       |     |   | 100 | p      |     |  |     |               |
|       |       |     |   |  |             |  |       |     |       |     |   |     | 101 p  |     |  |     |               |
|       |       |     |   |  |             |  |       |     |       |     |   |     |        | 102 |  | p   |               |
| p     |       |     |   |  |             |  |       |     |       |     |   |     |        |     |  | 110 | p             |

. . Figure 1: Visualization of the state of TreeCal/TreeSwap at time step t (about half-way through the algorithm). For H = 3, we depict the intervals Γ of the first three non-root levels of the tree (l = 1, 2, 3). Each rectangular node represents an interval, with sibling nodes separated by red lines. We represent the specific time step t via the vertical dashed green line. The yellow intervals it intersects at each level correspond to the nodes on the root-to-leaf-t path. Accordingly, x<sup>t</sup> will be the uniform distribution over the labels p of these yellow intervals. We see that the algorithm has committed to the labels of all intervals that started at or before time t, and has yet to label the future intervals.

- TreeCal (Algorithm [1\)](#page-22-0) assigns actions to nodes as follows. For all 1 ≤ l ≤ L, k ∈ [0 : H − 1]<sup>l</sup>−<sup>1</sup> , h ∈ [0 : H − 1], at the start of Γ
  - (l) kh, TreeCal sets p
  - (l) kh to be the average over all y<sup>t</sup> that have been observed thus far in the parent interval Γ (l−1) k . That is,

$$p_{kh}^{(l)} = \frac{1}{hH^{L-l}} \sum_{i=0}^{h-1} \sum_{t \in \Gamma_{ki}^{(l)}} y_t \quad (4)$$

- The more general TreeSwap algorithm (Algorithm [2\)](#page-23-0) also takes as a parameter an external regret-minimizing algorithm Alg, which operates with horizon of length H: we denote the resulting algorithm by TreeSwap.Alg. TreeSwap.Alg associates each internal node of the tree, Γ (l−1) k (with <sup>1</sup> ≤ <sup>l</sup> ≤ <sup>L</sup>), with an instance Alg, denoted Alg(l−1) k . The subroutine Alg(l−1) k is responsible for choosing the actions p
  - (l) k0 , p
- (l) k1 , · · · , p
- (l) <sup>k</sup>(H−1). It does so by responding to the average losses over each of its child intervals. In particular: at the end of

<sup>8</sup>We ignore the truncated branches that exist if T < H<sup>L</sup> .

each child interval Γ (l) kh, we pass Alg(l−1) k the average loss over that interval. Alg(l−1) k then outputs the action p (l) <sup>k</sup>(h+1) assigned to the next child interval.

### 3.2 Main result

Theorem [3.1](#page-7-0) upper bounds the calibration error of TreeCal with respect to the squared norm k·k<sup>2</sup> .

Theorem 3.1 (Main theorem). *Let* P ⊂ R <sup>d</sup> *be a bounded convex set and* k·k *be an arbitrary norm. Then,* TreeCal *(Algorithm [1\)](#page-22-0) guarantees that for an arbitrary sequence of outcomes* y1, . . . , y<sup>T</sup> ∈ P*, the* k·k<sup>2</sup> *calibration error of its predictions* x1, . . . , x<sup>T</sup> ∈ ∆(P) *is bounded as follows:*

$$\text{Cal}_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T}) \leq \epsilon T \quad \text{for} \quad T \geq (\text{diam}(\mathcal{P})/\sqrt{\epsilon})^{O(\text{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon)}$$

It is straightforward to derive from Theorem [3.1](#page-7-0) via an application of Jensen's inequality an upper bound on the calibration error of TreeCal with respect to the (non-squared) norm k·k, as stated in Theorem [1.1;](#page-1-0) see Corollary [C.5.](#page-26-0) In Appendix [E,](#page-30-0) we additionally consider a variant of TreeCal which plays *pure actions* in P (i.e., not distributions) by sampling from the distributions x<sup>t</sup> for each t ∈ [T]. We show that the *pure calibration* error of this variant can be bounded by a similar quantity to that in Theorem [3.1.](#page-7-0)

### 3.3 Outline of the proof of Theorem [3.1](#page-7-0)

Step 1: Reduction from calibration error to swap regret. Let us choose a convex function R : P → <sup>R</sup> given P, k·k as described in Section [2.4.](#page-5-0) The first step in the proof of Theorem [3.1](#page-7-0) is to reduce the problem of minimizing (squared-norm) calibration error to that of minimizing full swap regret for an appropriate sequence of loss functions. In particular, for any sequence x1, . . . , x<sup>T</sup> ∈ ∆(P) and y1, . . . , y<sup>T</sup> ∈ P, we have

$$\text{Cal}_T^{\parallel \cdot \parallel^2}(\mathbf{x}_{1:T}, y_{1:T}) \leq \text{Cal}_T^{D_R}(\mathbf{x}_{1:T}, y_{1:T}) = \text{FullSwapReg}_R(\mathbf{x}_{1:T}, \ell_{1:T}), \quad (5)$$

where `<sup>t</sup> : P → <sup>R</sup> is the loss function given by `t(p) := DR(yt|p): the inequality uses strong convexity of R, and the subsequent equality uses Lemma [2.2.](#page-4-2)

Step 2: Equivalence with TreeSwap. Thus, it suffices to find an algorithm which minimizies the full swap regret quantity on the right-hand side of [\(5\).](#page-7-3) Fortunately, the TreeSwap algorithm is known to do exactly this! (See Theorem [C.1,](#page-23-1) from [\[DDFG24\]](#page-9-0), for a formal statement for the swap regret bound of TreeSwap.) In order to apply the swap regret bound of Theorem [C.1,](#page-23-1) we need to ensure that the TreeCal algorithm is an instantiation of TreeSwap.Alg for an appropriate choice of (a) the loss functions fed as input to TreeSwap and (b) the Alg subroutine. The loss functions have already been defined: given a sequence y1, . . . , y<sup>T</sup> , recall that we chose `t(p) := DR(yt|p). Moreover, we let the Alg subroutine be given by *Follow-the-Leader* (FTL), which simply chooses an action at each step minimizing the sum of losses up to the previous time step. The following lemma shows that TreeSwap with the losses `<sup>t</sup> and the FTL subroutine produces the same action distributions as TreeCal:

Lemma 3.2. *Let* P ⊂ R <sup>d</sup> *be a bounded convex set and let* R : P → <sup>R</sup> *be a convex function. For a sequence of loss functions* `1, · · · , `<sup>H</sup> : P → <sup>R</sup>*, define* FTLh(`1:H) = arg minp∈P P<sup>h</sup>−<sup>1</sup> <sup>s</sup>=1 `s(p)*. For all sequences of outcomes* y1:<sup>T</sup> ∈ P<sup>T</sup> *, the action distributions* x<sup>t</sup> *produced by* TreeCal *on* y1:<sup>T</sup> *equal those produced by* TreeSwap.FTL *on loss functions* `t(p) = DR(yt|p) *for all* t*.*

The proof of Lemma [3.2](#page-7-2) (given in full in the appendix) is a straightforward consequence of the fact that the Bregman divergence is a proper scoring rule: the action p ∈ P minimizing an average of Bregman divergences DR(y|p) is simply the average of the constituent points y (Lemma [2.1\)](#page-4-1).

Step 3: Applying the swap regret bound of TreeSwap to BTL. Finally, we want to apply the main result of [\[DDFG24\]](#page-9-0) (restated as Theorem [C.1\)](#page-23-1) to bound the full swap regret for the iterates x1:<sup>T</sup> produced by TreeSwap.Alg, for an appropriate choice of Alg. The most natural way to do so would be to try to directly apply this result in the case when Alg = FTL (which corresponds to how we actually implement TreeSwap). However, applying this theorem requires an external regret bound on FTL for an arbitrary sequence of losses. While FTL is known to possess strong external regret bounds in some situations (e.g., when all the loss functions are strongly convex), the loss functions p 7→ DR(y|p) are not necessarily even convex in p and so it is not a priori clear how to establish such bounds.

Instead, the main idea is to consider the "Be-The-Leader" algorithm BTL, which is the same as FTL but where actions are shifted ahead in time by 1 time step: in particular, the action chosen by BTL at time step h given a sequence `1, `2, . . . , `<sup>H</sup> : P → <sup>R</sup> is BTLh(`1:H) = FTLh+1(`1:H) = argminp∈P P<sup>h</sup> <sup>s</sup>=1 `s(p). BTL is not implementable since its action at time step h depends on the (unobserved) loss `<sup>h</sup> at that time step. However, since its regret is always non-positive (i.e., ExtRegH(BTL) ≤ 0 for any H), if we apply Theorem [C.1](#page-23-1) to the algorithm TreeSwap.BTL, we get that FullSwapReg<sup>T</sup> (TreeSwap.BTL) ≤ · T as long as T ≥ HO(ρ/) for *any* choice of H (the arity parameter H used in TreeSwap). Using [\(5\),](#page-7-3) this implies that the *calibration error* of the iterates produced by TreeSwap.BTL can also be bounded above by · T.

Of course, this result on its own is uninteresting (since BTL is unimplementable, as mentioned above). However, the key insight is that we can show that the actions chosen by TreeSwap.BTL are close to (as measured by the norm k·k) those chosen by TreeSwap.FTL, which in turn is equivalent to TreeCal (Lemma [3.2\)](#page-7-2). This closeness is an immediate consequence of the fact that the actions chosen by FTL for our loss functions DR(y1|·), DR(y2|·), . . . are simply the empirical average of all actions y1, y2, . . . ∈ P of the adversary up to the previous time step.[<sup>9</sup>](#page-8-1) In turn, we can use this closeness to show that the calibration error of TreeSwap.FTL is close to that of TreeSwap.BTL. This latter part of the argument becomes slightly tricky due to the possibility that different nodes of the tree might output the same action p ∈ P; accordingly, we need to work with a *labeled* variant of the action set and bound the swap regret over this labeled variant; see Appendix [C](#page-21-0) for further details.

# 4 Lower bound

To prove our calibration lower bound, we make use of the following swap regret lower bound.

Theorem 4.1 (Theorem 4.1 of [\[DFG](#page-9-5)<sup>+</sup>24]). *There is a sufficiently small constant* c[4](#page-8-2).<sup>1</sup> > 0 *so that the following holds. Fix any* > 0*. For any* d ∈ <sup>N</sup>*, there is a subset* X ⊂ [−1, 1]<sup>d</sup> *so that the following holds for any* T ≤ exp c[4](#page-8-2).<sup>1</sup> min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>6</sup>} *. There is an oblivious adversary producing a sequence* v1, . . . , v<sup>T</sup> *with* kvtk<sup>1</sup> ≤ 1 *and* kvtk<sup>∞</sup> ≤ max{d <sup>−</sup>13/<sup>14</sup>, <sup>13</sup>/<sup>6</sup>} *for all* t*, which satisfies the following property. For linear loss functions* `(x, v) = hv, xi *for vectors* v ∈ <sup>R</sup> <sup>d</sup> *and* x ∈ <sup>R</sup> d *, any learning algorithm producing* x1, . . . , x<sup>T</sup> ∈ ∆(X )*,*

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell(\cdot, v_{1:T})) = \sup_{\pi: \mathcal{X} \rightarrow \mathcal{X}} \sum_{t=1}^T \sum_{p \in \mathcal{X}} \mathbf{x}_t(p) \cdot (\langle v_t, p \rangle - \langle v_t, \pi(p) \rangle) \geq \epsilon \cdot T.$$

We leverage the classic reduction from swap-regret minimization to calibration [\[FV98\]](#page-9-3): by producing calibrated predictions of the upcoming loss and best-responding to it, we can effectively minimize swap regret. This is formalized in the following lemma, proved in Appendix [D.](#page-27-1)

Lemma 4.2. *Fix a set* P ⊂ R d *, a norm* k · k*, and write* D(p, p<sup>0</sup> ) := kp − p <sup>0</sup>k*. Suppose that, for some* > 0, T ∈ <sup>N</sup>*, there is an algorithm which chooses* x1, . . . , x<sup>T</sup> ∈ ∆(P) *and which ensures that for every oblivious adversary choosing* y1, . . . , y<sup>T</sup> ∈ P*, we have* Cal<sup>D</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ · T*. Then for every set* P <sup>0</sup> ⊂ <sup>R</sup> d *, there is an algorithm which chooses* x 0 1 , . . . , x 0 <sup>T</sup> ∈ ∆(P 0 ) *and which ensures that for every oblivious adversary choosing* y1, . . . , y<sup>T</sup> ∈ P*, we have*

$$\text{FullSwapReg}_T(\mathbf{x}'_{1:T}, \ell(\cdot, y_{1:T})) \leq \epsilon \cdot T \cdot \text{diam}_{\|\cdot\|_*}(\mathcal{P}').$$

Combining these two ideas, we demonstrate that an algorithm -calibrated predictions of outcomes on the simplex in T ≤ exp(poly(1/)) rounds could be used in Lemma [4.2](#page-8-3) to achieve a swap regret algorithm contradicting Theorem [4.1.](#page-8-2) This gives the following (proved in Appendix [D\)](#page-27-1).

<sup>9</sup>An observant reader might note that this same argument also lets us provide bounds on the regret of FTL for these losses. One subtlety in the analysis is that we obtain better calibration bounds by bounding the distance between the predictions of FTL and BTL in the k·k norm rather than in the losses DR(yt|·), and so it is important that we directly analyze TreeSwap.BTL instead of TreeSwap.FTL (the latter causes us to pick up an extra factor related to the *smoothness* of R).

Theorem 4.3. *There is a sufficiently small constant* c > 0 *so that the following holds. Write* D(p, p<sup>0</sup> ) = kp−p <sup>0</sup>k1*, and fix any* > 0, d ∈ <sup>N</sup>*. Then for any* T ≤ exp(c · min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>6</sup>})*, there is an oblivious adversary producing a sequence* y1, . . . , y<sup>T</sup> ∈ ∆<sup>d</sup> *so that for any learning algorithm producing* x1, . . . , x<sup>T</sup> ∈ ∆(∆<sup>d</sup> )*,* Cal<sup>D</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≥ · T.

In Theorem [D.2](#page-29-0) (see Appendix [D.2\)](#page-29-1), we show a similar lower bound for `<sup>2</sup> calibration over the unit `<sup>2</sup> ball.

# References


[ACRS25] Eshwar Ram Arunachaleswaran, Natalie Collina, Aaron Roth, and Mirah Shi. An elementary predictor obtaining distance to calibration. In *Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pages 1366–1370. SIAM, 2025. [AM11] Jacob Abernethy and Shie Mannor. Does an efficient calibrated forecasting strategy exist? In *Proceedings of the 24th Annual Conference on Learning Theory*, pages 809–812. JMLR Workshop and Conference Proceedings, 2011. [BGHN23] Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, and Preetum Nakkiran. A unifying theory of distance from calibration. In *Proceedings of the 55th Annual ACM Symposium on Theory of Computing*, pages 1727–1740, 2023. [BM07] Avrim Blum and Yishay Mansour. From external to internal regret. *Journal of Machine Learning Research*, 8(6), 2007. [Daw82] A Philip Dawid. The well-calibrated bayesian. *Journal of the American statistical Association*, 77(379):605–610, 1982. [DDF<sup>+</sup>24] Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, Noah Golowich, Robert Kleinberg, and Princewill Okoroafor. Breaking the t <sup>2</sup>/<sup>3</sup> barrier for sequential calibration. *arXiv preprint arXiv:2406.13668*, 2024. [DDFG24] Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. From external to swap regret 2.0: An efficient reduction for large action spaces. In *Proceedings of the 56th Annual ACM Symposium on Theory of Computing*, pages 1216–1222, 2024. [DFG<sup>+</sup>24] Constantinos Daskalakis, Gabriele Farina, Noah Golowich, Tuomas Sandholm, and Brian Hu Zhang. A lower bound on swap regret in extensive-form games. *arXiv preprint arXiv:2406.13116*, 2024. [FH18] Dean P Foster and Sergiu Hart. Smooth calibration, leaky forecasts, finite recall, and nash dynamics. *Games and Economic Behavior*, 109:271–293, 2018. [FKO<sup>+</sup>25] Maxwell Fishelson, Robert Kleinberg, Princewill Okoroafor, Renato Paes Leme, Jon Schneider, and Yifeng Teng. Full swap regret and discretized calibration. *arXiv preprint arXiv:2502.09332*, 2025. [FL99] Drew Fudenberg and David K Levine. An easier way to calibrate. *Games and economic behavior*, 29(1-2):131–137, 1999. [Fos99] Dean P Foster. A proof of calibration via blackwell's approachability theorem. *Games and Economic Behavior*, 29(1-2):73–78, 1999. [FV97] Dean P Foster and Rakesh V Vohra. Calibrated learning and correlated equilibrium. *Games and Economic Behavior*, 21(1-2):40–55, 1997. [FV98] Dean P Foster and Rakesh V Vohra. Asymptotic calibration. *Biometrika*, 85(2):379–390, 1998. [GJRR24] Sumegha Garg, Christopher Jung, Omer Reingold, and Aaron Roth. Oracle efficient online multicalibration and omniprediction. In *Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pages 2725–2792. SIAM, 2024. [GPSW17] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Doina Precup and Yee Whye Teh, editors, *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pages 1321–1330. PMLR, 06–11 Aug 2017.

[GSJ24] Khashayar Gatmiry, Jon Schneider, and Stefanie Jegelka. Computing optimal regularizers for online linear optimization. *arXiv preprint arXiv:2410.17336*, 2024. [H<sup>+</sup>16] Elad Hazan et al. Introduction to online convex optimization. *Foundations and Trends® in Optimization*, 2(3-4):157–325, 2016. [Har22] Sergiu Hart. Calibrated forecasts: The minimax proof. *arXiv preprint arXiv:2209.05863*, 2022. [HJKRR18] Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Multicalibration: Calibration for the (Computationally-identifiable) masses. In Jennifer Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 1939–1948. PMLR, 10–15 Jul 2018. [HK12] Elad Hazan and Sham M Kakade. (weak) calibration is computationally hard. In *Conference on Learning Theory*, pages 3–1. JMLR Workshop and Conference Proceedings, 2012. [HW24] Lunjia Hu and Yifan Wu. Predict to minimize swap regret for all payoff-bounded tasks. In *2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS)*, pages 244–263. IEEE, 2024. [KF08] Sham M Kakade and Dean P Foster. Deterministic calibration and nash equilibrium. *Journal of Computer and System Sciences*, 74(1):115–130, 2008. [KLST23] Bobby Kleinberg, Renato Paes Leme, Jon Schneider, and Yifeng Teng. U-calibration: Forecasting for an unknown agent. In *The Thirty Sixth Annual Conference on Learning Theory*, pages 5143–5145. PMLR, 2023. [LSS24] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Optimal multiclass u-calibration error and beyond. *arXiv preprint arXiv:2405.19374*, 2024. [LSS25] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Simultaneous swap regret minimization via kl-calibration. *arXiv preprint arXiv:2502.16387*, 2025. [MS10] Shie Mannor and Gilles Stoltz. A geometric proof of calibration. *Mathematics of Operations Research*, 35(4):721–727, 2010. [MSA07] Shie Mannor, Jeff S Shamma, and Gürdal Arslan. Online calibrated forecasts: Memory efficiency versus universality for learning in games. *Machine Learning*, 67:77–115, 2007. [NRRX23] Georgy Noarov, Ramya Ramalingam, Aaron Roth, and Stephan Xie. High-dimensional prediction for sequential decision making. *arXiv preprint arXiv:2310.17651*, 2023. [Pen25] Binghui Peng. High dimensional online calibration in polynomial time. *arXiv preprint arXiv:2504.09096*, 2025. [PR24] Binghui Peng and Aviad Rubinstein. Fast swap regret minimization and applications to approximate correlated equilibria. In *Proceedings of the 56th Annual ACM Symposium on Theory of Computing*, pages 1223–1234, 2024. [QV21] Mingda Qiao and Gregory Valiant. Stronger calibration lower bounds via sidestepping. In *Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing*, pages 456–466, 2021. [QZ24] Mingda Qiao and Letian Zheng. On the distance from calibration in sequential prediction. In *The Thirty Seventh Annual Conference on Learning Theory*, pages 4307–4357. PMLR, 2024. [RS24] Aaron Roth and Mirah Shi. Forecasting for swap regret for all downstream agents. In *Proceedings of the 25th ACM Conference on Economics and Computation*, pages 466–488, 2024. [RST15] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Sequential complexities and uniform martingale laws of large numbers. *Probability Theory and Related Fields*, 161(1-2):111–153, 2015. [SS11] Shai Shalev-Shwartz. Online learning and online convex optimization. *Foundations and Trends in Machine Learning*, 4(2):107–194, 2011.

[SST11] Nati Srebro, Karthik Sridharan, and Ambuj Tewari. On the universality of online mirror descent. *Advances in neural information processing systems*, 24, 2011. [ZME20] Shengjia Zhao, Tengyu Ma, and Stefano Ermon. Individual calibration with randomized forecasting. In Hal Daumé III and Aarti Singh, editors, *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pages 11387–11397. PMLR, 13–18 Jul 2020.
# NeurIPS Paper Checklist

### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We prove all stated claims.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss limitations.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Justification: We prove all theorems and lemmas.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
  - (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
  - (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
  - (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
  - (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not include experiments requiring code.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.

# A Additional Related Work

There is a large range of other existing work on online (sequential) calibration [\[Daw82,](#page-9-6) [FV97,](#page-9-2) [FV98,](#page-9-3) [QV21,](#page-10-6) [DDF](#page-9-7)<sup>+</sup>24, [Har22,](#page-10-7) [Fos99,](#page-9-8) [FL99,](#page-9-9) [KF08,](#page-10-8) [MSA07,](#page-10-9) [MS10,](#page-10-10) [AM11,](#page-9-10) [HK12,](#page-10-11) [FH18,](#page-9-11) [LSS24,](#page-10-12) [NRRX23,](#page-10-13) [KLST23,](#page-10-14) [GJRR24,](#page-9-12) [QZ24,](#page-10-15) [ACRS25\]](#page-9-13). We briefly survey some of these areas below.

Binary outcomes. For binary outcomes (i.e., one-dimensional calibration), classical results of [\[FV97,](#page-9-2) [Fos99,](#page-9-8) [BM07,](#page-9-14) [AM11\]](#page-9-10) demonstrate that it is possible to efficiently guarantee O(T 2/3 ) `1 calibration. The optimal possible rates for `1-calibration remain a major unsolved problem in online learning. Recently [\[QV21\]](#page-10-6) improved over the naive lower bound of Ω(√ T) by demonstrating a lower bound of Ω(T <sup>0</sup>.<sup>528</sup>); this was further improved to Ω(T <sup>0</sup>.<sup>543</sup>) by [\[DDF](#page-9-7)<sup>+</sup>24], who also improved on the upper bound, demonstrating the existence of an algorithm with O(T 2/3− ) calibration for some constant > 0.

Calibration and swap regret. The connection between calibration and swap regret has been acknowledged since the earliest works on swap regret. For example, the earliest algorithms for minimizing swap regret worked by best responding to online calibrated predictions [\[FV97\]](#page-9-2) (later algorithms for swap regret minimization, such as [\[BM07\]](#page-9-14) and [\[DDF](#page-9-7)<sup>+</sup>24] obtain better swap regret bounds by side-stepping the need to generate calibrated predictions). In the other direction, several works minimize calibration via relating it to a swap regret that can then be minimized [\[FKO](#page-9-4)<sup>+</sup>25, [LSS25,](#page-10-4) [AM11,](#page-9-10) [Fos99\]](#page-9-8).

Other forms of calibration. Due to the difficulty of minimizing (high-dimensional) calibration, there has been a line of work on designing forecasting algorithms that minimize weaker forms of calibration that recover some of the important guarantees of calibration (e.g., trustworthy-ness by a decision-maker). These include *distance from calibration* [\[BGHN23,](#page-9-15) [QZ24,](#page-10-15) [ACRS25\]](#page-9-13), *omniprediction error / U-calibration* [\[KLST23,](#page-10-14) [LSS24,](#page-10-12) [GJRR24\]](#page-9-12), *calibration conditioned on downstream outcomes* [\[NRRX23\]](#page-10-13), and *prediction for downstream swap regret* [\[RS24,](#page-10-16) [HW24\]](#page-10-17). Other work focuses on minimizing notions of calibration designed to lead to specific classes of equilibria, e.g. weak calibration [\[HK12\]](#page-10-11), deterministic calibration [\[KF08\]](#page-10-8), and smooth calibration [\[FH18\]](#page-9-11).

# B Proofs of preliminary results

![](_page_19_Diagram_6.jpeg)

![](_page_19_Picture_7.jpeg)

Figure 2: Geometric depiction of the Bregman divergence from p to y.

*Proof of Lemma [2.1.](#page-4-1)*

$$\begin{aligned} \mathbb{E}_{y \sim \mathcal{Y}}[D_R(y|p)] &= \mathbb{E}_{y \sim \mathcal{Y}}[R(y) - R(p) - \langle \nabla R(p), y - p \rangle] \\ &= \overline{R(y)} - R(p) - \langle \nabla R(p), \bar{y} - p \rangle \\ &= D_R(\bar{y}|p) + \overline{R(y)} - R(\bar{y}) \end{aligned}$$

![](_page_20_Figure_0.jpeg)

Figure 3: [Proof of Lemma [2.1\]](#page-4-1) the average Bregman divergence (orange + purple) decomposes into the Jensen error (orange) and the Bregman divergence to the mean (purple). For example, when R(p) = kpk 2 2 , DR(y|p) = ky − pk 2 2 and we recover the bias-variance decomposition.

*Proof of Lemma [2.2.](#page-4-2)* Fix any p ∈ P, and consider the quantity maxp∗∈P P <sup>t</sup> xt(p)(DR(yt|p) − DR(yt|p ∗ )). By considering the distribution y that has weight xt(p)/ P <sup>t</sup> xt(p) on yt, Lemma [2.1](#page-4-1) implies that this quantity is maximized when p <sup>∗</sup> = ν<sup>p</sup> = (P <sup>t</sup> xt(p)yt)/( P <sup>t</sup> xt(p)). At this optimal value of p ∗ , this quantity can be rewritten as:

$$\begin{aligned}
& \sum_t \mathbf{x}_t(p) (D_R(y_t|p) - D_R(y_t|\nu_p)) \\
= & \sum_t \mathbf{x}_t(p) [(R(y_t) - R(p) - \langle \nabla R(p), y_t - p \rangle) - (R(y_t) - R(\nu_p) - \langle \nabla R(\nu_p), y_t - \nu_p \rangle)] \\
= & \sum_t \mathbf{x}_t(p) [(R(\nu_p) - R(p) - \langle \nabla R(p), \nu_p - p \rangle) + \langle \nabla R(\nu_p) - \nabla R(p), y_t - \nu_p \rangle] \\
= & \sum_t \mathbf{x}_t(p) D_R(\nu_p|p) + \left\langle \nabla R(\nu_p) - \nabla R(p), \sum_t \mathbf{x}_t(p)(y_t - \nu_p) \right\rangle \\
= & \sum_t \mathbf{x}_t(p) D_R(\nu_p|p).
\end{aligned}$$

$$\begin{aligned} \text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) &= \sup_{\pi:\mathcal{P} \rightarrow \mathcal{P}} \sum_{t=1}^T \sum_{p \in \mathcal{P}} \mathbf{x}_t(p) \cdot (\ell_t(p) - \ell_t(\pi(p))) \\ &= \sum_{p \in \mathcal{P}} \max_{p^* \in \mathcal{P}} \sum_{t=1}^T \mathbf{x}_t(p) \cdot (\ell_t(p) - \ell_t(p^*)) \\ &= \sum_{p \in \mathcal{P}} \max_{p^* \in \mathcal{P}} \sum_{t=1}^T \mathbf{x}_t(p) \cdot (D_R(y_t|p) - D_R(y_t|\nu_p)) \\ &= \sum_{p \in \mathcal{P}} \sum_t \mathbf{x}_t(p) D_R(\nu_p|p) \\ &= \text{Cal}_T^{D_R}(\mathbf{x}_{1:T}, y_{1:T}). \end{aligned}$$

*Proof of Lemma [2.4.](#page-5-2)* Note that if we define L = {y ∈ <sup>R</sup> d | kyk<sup>∗</sup> ≤ 1} to be the unit dual norm ball for some norm k·k, then by duality the norm k·kL<sup>∗</sup> corresponding to L ∗ is simply the original norm k·k. It therefore suffices to show that given a 1-strongly convex function R with bounded range ρ, it is possible to construct a 1-strongly convex function R<sup>0</sup> with bounded Bregman divergence O(ρ) (and vice versa).

Assume R(p) is 1-strongly convex and satisfies maxp∈P R(p) − minp∈P R(p) = ρ. Define R<sup>0</sup> (p) = 4R p 2 (since P is centrally symmetric, p/2 is guaranteed to belong to P). If R is 1-strongly convex, then R(p/2) is 1/4-strongly convex, and so R<sup>0</sup> (p) is also 1-strongly convex. We claim the maximum Bregman divergence of R<sup>0</sup> is at most O(ρ). To show this, we first argue that for any z1, z<sup>2</sup> ∈ P, ∇R( z<sup>1</sup> ), z<sup>2</sup> ≤ 2ρ. To see this, note that since R(p) is convex and has range bounded by ρ, we have that ρ ≥ R(p) − R( z<sup>1</sup> 2 ) ≥ ∇R( z<sup>1</sup> 2 ), x − z<sup>1</sup> 2 . If we set p = z1+z<sup>2</sup> 2 , it then follows that ∇R( z<sup>1</sup> 2 ), z<sup>2</sup> ≤ 2ρ. Now, note that

$$\begin{aligned} \max_{y,p \in \mathcal{P}} D_{R'}(y|p) &= R'(y) - R'(p) - \langle \nabla R'(p), y - p \rangle \\ &= R\left(\frac{y}{2}\right) - R\left(\frac{p}{2}\right) - \frac{1}{2} \langle \nabla R\left(\frac{p}{2}\right), y - p \rangle \\ &\leq \left| R\left(\frac{y}{2}\right) - R\left(\frac{p}{2}\right) \right| + \left\langle \nabla R\left(\frac{p}{2}\right), \frac{y-p}{2} \right\rangle \leq 3\rho. \end{aligned}$$

Conversely, if R(p) is 1-strongly convex and satisfies maxy,p∈P DR(y|p) ≤ ρ, define R<sup>0</sup> (p) = R(p) − h∇R(0), pi − R(0) (i.e., subtracting a linear function to make zero a minimizer of R<sup>0</sup> (p)). Since R and R<sup>0</sup> differ by a linear function, R<sup>0</sup> is also 1-strongly convex. But also, note that DR(y|0) = R(y) − R(0) − h∇R(0), yi = R<sup>0</sup> (y); since D<sup>R</sup> is bounded in range by ρ, it follows that so is R<sup>0</sup> .

# C Proof of Theorem [3.1](#page-7-0)

In this section, we prove Theorem [3.1.](#page-7-0) First, in Appendix [C.1,](#page-21-1) we introduce a slightly stronger notion of calibration error and swap regret to deal with a technicality in the proof. We then give the proof of Theorem [3.1.](#page-7-0)

### C.1 Labeled calibration and swap regret

Intuition. Recall that the TreeCal algorithm labels each interval Γ (l) k of the tree with some action, p (l) <sup>k</sup> ∈ P. At each time step t, the algorithm outputs the uniform distribution over all p (l) k

Algorithm 1 TreeCal(P, T, H, L)

Require: Action set P ⊂ R d , time horizon T, parameters H, L with T ≤ H<sup>L</sup>.

1: for 1 ≤ t ≤ T do 2: Write the base-H representation of t − 1 as t = (h<sup>1</sup> · · · hL), for h1, . . . , h<sup>L</sup> ∈ [0 : H − 1]. 3: for 1 ≤ l ≤ L do 4: Write k := (h<sup>1</sup> · · · hl−1) ∈ [0 : H − 1]<sup>l</sup>−<sup>1</sup> . 5: if hl+1 = · · · = h<sup>L</sup> = 0 or l = L then 6: If h<sup>l</sup> > 0, define ν (l) k,hl−1 := <sup>1</sup> HL−<sup>l</sup> · P s∈Γ k,hl−<sup>1</sup> ys. 7: Define p (l) k,h<sup>l</sup> := <sup>1</sup> h<sup>l</sup> P<sup>h</sup>l−<sup>1</sup> <sup>i</sup>=0 ν (l) k,i if h<sup>l</sup> > 0, otherwise choose arbitrary p (l) k,h<sup>l</sup> ∈ P. 8: end if 9: end for 10: Output the uniform mixture x<sup>t</sup> := Unif({p (1) h<sup>1</sup> , . . . , p (L) h1···h<sup>L</sup> }), and observe yt. 11: end for

with Γ (l) <sup>k</sup> 3 t. When evaluating the calibration error, suppose that the actions p (l) k are all distinct, for l ∈ [L], k ∈ [0 : H − 1]<sup>l</sup>−<sup>1</sup> (as we discuss below, this case is in some sense the "worst case"). In this event, each action p (l) k is compared to the average outcome over the interval Γ (l) k : y¯ (l) <sup>k</sup> = 1 Γ k P t∈Γ k yt. Formally, this would give

$$\text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}) = \sum_{l=1}^L \frac{H^{L-l}}{L} \sum_{k \in [H]^l} D\left(\bar{y}_k^{(l)}, p_k^{(l)}\right). \quad (6)$$

as each level l action is selected with <sup>1</sup> <sup>L</sup> mass for <sup>H</sup><sup>L</sup>−<sup>l</sup> rounds.

If it happened that two distinct intervals Γ (l1) k<sup>1</sup> , Γ (l2) k<sup>2</sup> were assigned the same action p = p (l1) k<sup>1</sup> = p (l1) k<sup>1</sup> , then the calibration error would be *at most* the quantity on the right-hand side of [\(6\)](#page-22-1) (by Jensen's inequality). In particular, rather than having to compare p to two potentially distinct quantities D(¯y (l1) k<sup>1</sup> , p), D(¯y (l2) k<sup>2</sup> , p), the mass placed on p would be categorized under the same forecast and we would only compare p to an appropriately-weighted average of y¯ (l1) k<sup>1</sup> and y¯ (l2) k<sup>2</sup> .

For technical reasons, it will turn out to be necessary to upper bound the "worst case quantity" on the right-hand side of [\(6\)](#page-22-1) (and an analogous version for swap regret), even in the even that the actions p (l) k are *not all distinct*. To streamline our notation, we introduce a generalization of these quantities which apply for arbitrary algorithms, which we call *labeled* calibration error and *labeled* swap regret.

Formal definitions. Given a convex set P ⊂ R d , we define its *labeled* extension to be P¯ := P × {0, 1} ? , i.e., elements of P¯ are tuples (p, σ), where σ ∈ {0, 1} ? is a string that is said to *label* p. For a loss function ` : P → <sup>R</sup>, we extend its domain to P¯ in the natural way, i.e., `((p, σ)) := `(p) for (p, σ) ∈ P¯. Given a sequence of distributions over the labeled extension, x1, . . . , x<sup>T</sup> ∈ ∆(P¯), and loss functions `1, . . . , `<sup>T</sup> : P → <sup>R</sup>, we define

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) := \sup_{\pi: \bar{\mathcal{P}} \rightarrow \bar{\mathcal{P}}} \sum_{t=1}^T \sum_{p \in \bar{\mathcal{P}}} \mathbf{x}_t(p) \cdot (\ell_t(p) - \ell_t(\pi(p))).$$

In words, the full swap regret of x1:<sup>T</sup> with respect to `1:<sup>T</sup> is defined identically as in [\(2\)](#page-4-4) except that the swap function π can now depend on the label σ. In particular, the labeled extension allows us to consider a more refined notion of swap regret where identical actions played in different rounds can be swapped (via π) to different alternatives as long as they have different labels.

In a similar manner we define the calibration error for a sequence of labeled distributions: given x1, . . . , x<sup>T</sup> ∈ ∆(P¯) and y1, . . . , y<sup>T</sup> ∈ P, we define

$$\text{Cal}_T^D := \sum_{(p,\sigma) \in \bar{\mathcal{P}}} \left( \sum_{t=1}^T \mathbf{x}_t((p, \sigma)) \right) \cdot D(\nu_{(p,\sigma)}, p), \quad \nu_{(p,\sigma)} := \frac{\sum_{t=1}^T \mathbf{x}_t((p, \sigma)) \cdot y_t}{\sum_{t=1}^T \mathbf{x}_t((p, \sigma))}.$$

The main result of [\[DDFG24\]](#page-9-0) shows that the swap regret of TreeSwap is bounded, even when one labels the action produced at each node of the tree by the node of the tree. This labeled variant of TreeSwap is given in Algorithm [2.](#page-23-0) It functions exactly as discussed in Section [3.1,](#page-6-2) except that the distribution x<sup>t</sup> output at time step t is in ∆(P¯) instead of ∆(P). In particular, each p (l) <sup>k</sup> ∈ P in the support of x<sup>t</sup> is labeled by the tuple k ∈ [0 : H − 1]<sup>l</sup> . [10](#page-23-2)

Theorem C.1 (TreeSwap; Theorem 3.1 of [\[DDFG24\]](#page-9-0)). *Suppose that* H, L ∈ N *satisfy* H ≥ 2 *and* H<sup>L</sup>−<sup>1</sup> ≤ T ≤ H<sup>L</sup>*. For bounded convex action set* P ⊂ <sup>R</sup> <sup>d</sup> *and loss function set* L ⊂ {` : P → [0, b]}*, let* Alg<sup>H</sup> : L <sup>H</sup> → P<sup>H</sup> *be any algorithm. Then, the labeled* TreeSwap *algorithm (Algorithm [2\)](#page-23-0) parametrized by* T, H, L,P,L, Alg<sup>H</sup> *outputs labeled distributions* <sup>x</sup>1, . . . , <sup>x</sup><sup>T</sup> <sup>∈</sup> ∆(P¯) *satisfying the following: for any sequence* `1, . . . , `<sup>T</sup> ∈ L*,*

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) \leq T \cdot \left( \frac{\text{ExtReg}_H(\text{Alg}_H)}{H} + \frac{3b}{L} \right).$$

Algorithm 2 TreeSwap.Alg(P,L, T, H, L), labeled variant (see Appendix [C.1\)](#page-21-1)

Require: Action set P ⊂ R d , convex loss class L ⊂ (P → <sup>R</sup>), no-external regret algorithm Alg, time horizon T, parameters H, L with T ≤ H<sup>L</sup>.

- 1: For each sequence h<sup>1</sup> · · · hl−<sup>1</sup> ∈ S<sup>L</sup> <sup>l</sup>=1[0 : <sup>H</sup> − 1]<sup>l</sup>−<sup>1</sup> , initialize an instance of Alg with time horizon H, denoted Algh1:l−<sup>1</sup> . 2: for 1 ≤ t ≤ T do 3: Write the base-H representation of t−1 as t−1 = (h<sup>1</sup> · · · hL), for h1, . . . , h<sup>L</sup> ∈ [0 : H −1]. 4: for 1 ≤ l ≤ L do 5: Write k := (h<sup>1</sup> · · · hl−1) ∈ [0 : H − 1]<sup>l</sup>−<sup>1</sup> . 6: if hl+1 = · · · = h<sup>L</sup> = 0 or l = L then 7: If h<sup>l</sup> > 0, define `
- (l) k,hl−1 := <sup>1</sup> HL−<sup>l</sup> · P s∈Γ k,hl−<sup>1</sup> `<sup>s</sup> ∈ L. 8: Define p
  - (l) k,h<sup>l</sup> = Algk,hl+1(`
  - (l) k,0:hl−1 ) ∈ P. . *The* h<sup>l</sup> *th action of* Alg<sup>k</sup> *given the loss sequence* `
- (l) k,1:hl−1 *.* 9: end if 10: end for 11: Output the uniform mixture x<sup>t</sup> := Unif({(p
- (1) h<sup>1</sup> , h1), . . . ,(p
- (L) h1···h<sup>L</sup> , h1:L)}) ∈ ∆(P¯), and observe `t. . *Each action* p
- (l) k *is* labeled *by the sequence* k *(see Appendix [C.1\)](#page-21-1).* 12: end for

### C.2 Proof of the main theorem

First, we recall some definitions from Section [3.](#page-5-3) For all l ∈ [0 : L], for all k ∈ [H] l , let Γ (l) k represent the interval of times t with prefix k. That is, t ∈ Γ (l) k iff t<sup>i</sup> = k<sup>i</sup> for all i ∈ [1 : l]. These intervals form an H-ary depth-L tree, where the children of Γ (l) k are Γ (l+1) k0 , Γ (l+1) k1 , · · · , Γ (l+1) <sup>k</sup>(H−1). In the calibration setting where the learner receives outcomes y1:<sup>T</sup> , let ν (l) <sup>k</sup> = Γ k P t∈Γ k y<sup>t</sup> (as defined on Line [6](#page-22-2) of Algorithm [1\)](#page-22-0). In the swap regret setting where the learner receives loss functions `1:<sup>T</sup> , let ` (l) <sup>k</sup> =  Γ k  P t∈Γ `<sup>t</sup> (as defined in Line [7](#page-22-3) of Algorithm [2\)](#page-23-0).

Finally, recall that for an online learning algorithm Alg with time horizon H, we define its action at time step h ∈ [H] given losses `1, . . . , `<sup>H</sup> : P → <sup>R</sup> by Alg<sup>h</sup> (`1, . . . , `H). If Alg<sup>h</sup> only depends on the first g losses, then we will write Alg<sup>h</sup> (`1, . . . , `g). In the proof of Theorem [3.1](#page-7-0) we will consider two algorithms in particular; the first, Follow-The-Leader (FTL) is defined as follows: for

<sup>10</sup>Technically, the analysis of [\[DDFG24\]](#page-9-0) does not analyse the labeled version, but the proof goes through as is – the only step where labeling changes any of the reasoning in the argument is in Eq. (8) of [\[DDFG24\]](#page-9-0), where the upper bound as written in that equation holds even for the labeled version.

`1, . . . , `h−<sup>1</sup> : P → <sup>R</sup>, we have

$$\text{FTL}_h(\ell_1, \dots, \ell_{h-1}) = \operatorname{argmin}_{p \in \mathcal{P}} \sum_{i=1}^{h-1} \ell_i(p).$$

The second algorithm we consider is the Be-The-Leader algorithm (BTL), which is defined as follows: for `1, . . . , `<sup>h</sup> : P → <sup>R</sup>, we have

$$\text{BTL}_h(\ell_1, \dots, \ell_h) = \operatorname{argmin}_{p \in \mathcal{P}} \sum_{i=1}^h \ell_i(p).$$

Note that since BTLh(`1:h) depends on the unobserved loss `<sup>h</sup> at time step h, it is unimplementable. Nevertheless, it will be useful in our analysis.

Next we prove Lemma [3.2,](#page-7-2) establishing the equivalence of TreeCal and TreeSwap.FTL. In fact, we establish the stronger claim, which immediately implies Lemma [3.2.](#page-7-2)

Lemma C.2. *Fix distributions* q0, . . . , q<sup>h</sup> ∈ ∆(P)*, and define* `h(p) := <sup>E</sup>y∼q<sup>h</sup> [DR(y|p)]*. Then for each* h > 0*,* FTLh(`0, . . . , `h−1) = <sup>1</sup> h P<sup>h</sup>−<sup>1</sup> <sup>i</sup>=0 <sup>E</sup>y∼q<sup>i</sup> [y]*.*

*Proof.* The lemma is an immediate consequence of Lemma [2.1,](#page-4-1) noting that

$$\text{FTL}_h(\ell_0, \dots, \ell_{h-1}) = \operatorname{argmin}_{p \in \mathcal{P}} \sum_{i=0}^{h-1} \ell_i(p) = \operatorname{argmin}_{p \in \mathcal{P}} \mathbb{E}_{i \sim [0:h-1], y \sim q_i} [D_R(y_i|p)] = \frac{1}{h} \sum_{i=0}^{h-1} \mathbb{E}_{y \sim q_i} [y]. \quad (7)$$

*Proof of Lemma [3.2.](#page-7-2)* At time t, both TreeCal (Line [10](#page-22-4) of Algorithm [1\)](#page-22-0) and TreeSwap.FTL (Line [11](#page-22-5) of Algorithm [2\)](#page-23-0) select <sup>x</sup><sup>t</sup> <sup>=</sup> Unif n<sup>p</sup> (1) t1 , p (2) t1t<sup>2</sup> , · · · , p (L) t1t2···t<sup>L</sup> o). It remains to demonstrate that both algorithms assign actions p (l) k to intervals Γ (l) k identically. Fixing a choice of l ∈ [L] and k ∈ [0 : H−1]<sup>l</sup>−<sup>1</sup> , this is an immediate consequence of Lemma [C.2](#page-24-0) with q<sup>h</sup> = Unif({y<sup>t</sup> : t ∈ Γ (l) k,h}) and the fact that:

- In TreeCal, p
- (l) k,h = h P<sup>h</sup>−<sup>1</sup> <sup>i</sup>=0 ν
- (l) k,i with ν
- (l) k,i = <sup>E</sup>t∼Unif(Γ(l) k,i) [yt];
- Whereas in TreeSwap.FTL, p
- (l) k,h = FTLh+1(`
- (l) k,0 , . . . , `(l) k,h−1 ) with `
- (l) k,i = <sup>E</sup>t∼Unif(Γ(l) k,i) [DR(yt|·)].

We are now ready to prove Theorem [3.1.](#page-7-0)

*Proof of Theorem [3.1.](#page-7-0)* Fix any convex set P and a norm k·k, and let R : P → <sup>R</sup> be chosen to be 1-strongly convex which has range ρ > 0. Lemma [3.2](#page-7-2) gives that the actions x1, . . . , x<sup>T</sup> ∈ ∆(P) are identical to the actions played by TreeSwap.FTL with losses `t(p) = DR(yt|p) (Algorithm [2;](#page-23-0) we are ignoring the labels here). Thus, from here on, it suffices to bound the calibration error of the corresponding distributions x1, . . . , x<sup>T</sup> of TreeSwap.FTL. The actions p (l) k,h (for l ∈ [L], k ∈ [0 : H − 1]<sup>l</sup>−<sup>1</sup> , h ∈ [0 : H − 1]) of TreeSwap.FTL satisfy p (l) k,h = FTLh+1(` (l) k,0 , . . . , `(l) k,h−1 ).

Next, let p˜ (l) k,h denote the corresponding actions played by TreeSwap.BTL, i.e., p˜ (l) k,h = BTLh+1(` (l) k,0 , . . . , `(l) k,h). We let <sup>x</sup><sup>t</sup> <sup>∈</sup> ∆(P¯) denote the (labeled) distribution chosen by TreeSwap.FTL (Line [11](#page-22-5) of Algorithm [2\)](#page-23-0), and let x˜<sup>t</sup> ∈ ∆(P¯) denote the corresponding distribution for TreeSwap.BTL. To be concrete, if t − 1 = (h<sup>1</sup> · · · hL), then

$$\mathbf{x}_t = \text{Unif}(\{(p_{h_1}^{(1)}, h_1), \dots, (p_{h_1\dots h_L}^{(L)}, h_{1:L})\}), \quad \tilde{\mathbf{x}}_t = \text{Unif}(\{(\tilde{p}_{h_1}^{(1)}, h_1), \dots, (\tilde{p}_{h_1\dots h_L}^{(L)}, h_{1:L})\}), \quad (8)$$

We state the below claim, whose proof is deferred to the end of the section. (We remark that the primary purpose of introducing labeling is so that it is possible to establish Claim [C.3.](#page-25-0))

Claim C.3. *It holds that*

$$\text{Cal}_T^{\parallel \cdot \parallel^2}(\mathbf{x}_{1:T}, y_{1:T}) - 2\text{Cal}_T^{\parallel \cdot \parallel^2}(\tilde{\mathbf{x}}_{1:T}, y_{1:T}) \leq \frac{2 \cdot \text{diam}(\mathcal{P})^2}{H^2} \cdot T. \quad (9)$$

The fact that BTL enjoys non-positive external regret (e.g., [\[SS11,](#page-10-18) Lemma 2.1] gives that for an arbitrary sequence of loss functions `<sup>t</sup> : P → <sup>R</sup>, the external regret of BTL<sup>H</sup> satisfies ExtRegH(BTLH) ≤ 0. Thus, by Theorem [C.1,](#page-23-1) the swap regret of (the labeled version of) TreeSwap<sup>T</sup> applied with Alg<sup>H</sup> = BTL<sup>H</sup> may be bounded as follows: for any sequence of losses `1, . . . , `<sup>T</sup> : P → [0, ρ],

$$\text{FullSwapReg}_T(\tilde{\mathbf{x}}_{1:T}, \ell_{1:T}) \leq T \cdot \frac{3\rho}{L}.$$

Using Lemma [2.2](#page-4-2)[<sup>11</sup>](#page-25-1) and [\(9\),](#page-25-2) we get that for an arbitrary sequence y1, . . . , y<sup>T</sup> ∈ P,

$$\begin{aligned} \text{Cal}_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T}) &\leq 2 \cdot \text{Cal}_T^{\|\cdot\|^2}(\tilde{\mathbf{x}}_{1:T}, y_{1:T}) + \frac{2 \cdot \text{diam}(\mathcal{P})^2}{H^2} \cdot T \\ &\leq 2 \cdot \text{Cal}_T^{D_R}(\tilde{\mathbf{x}}_{1:T}, y_{1:T}) + \frac{2 \cdot \text{diam}(\mathcal{P})^2}{H^2} \cdot T \\ &= 2 \cdot \text{FullSwapReg}_T(\tilde{\mathbf{x}}_{1:T}, D_R(y_{1:T}|\cdot)) + \frac{2 \cdot \text{diam}(\mathcal{P})^2}{H^2} \cdot T \\ &\leq \frac{6\rho \cdot T}{L} + \frac{2 \cdot \text{diam}(\mathcal{P})^2 \cdot T}{H^2}. \end{aligned}$$

Given any desired accuracy > 0, choosing L = 12ρ/ and H = diam(P)/ √ gives that we can guarantee Calk·k<sup>2</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ · T as long as T ≥ H<sup>L</sup> = (diam(P)/ √ ) 12ρ/<sup>2</sup> .

*Proof of Claim [C.3.](#page-25-0)* For each t ∈ [T], we can write t − 1 = h1h<sup>2</sup> · · · h<sup>L</sup> with h<sup>i</sup> ∈ [0 : H − 1] for all i ∈ [L], and xt, x˜<sup>t</sup> are as given in [\(8\).](#page-24-1) Let us write, for (p, σ) ∈ P¯,

$$\begin{aligned} \nu_{(p,\sigma)} &:= \frac{\sum_{t=1}^T \mathbf{x}_t((p,\sigma)) \cdot y_t}{\sum_{t=1}^T \mathbf{x}_t((p,\sigma))}, & \tilde{\nu}_{(p,\sigma)} &:= \frac{\sum_{t=1}^T \tilde{\mathbf{x}}_t((p,\sigma)) \cdot y_t}{\sum_{t=1}^T \mathbf{x}_t((p,\sigma))}, \\ \nu_{\sigma} &:= \frac{\sum_{p \in \mathcal{P}} \sum_{t=1}^T \mathbf{x}_t((p,\sigma)) \cdot y_t}{\sum_{p \in \mathcal{P}} \sum_{t=1}^T \mathbf{x}_t((p,\sigma))}. \end{aligned} \quad (10)$$

Since each p (l) h1···h<sup>l</sup> and each p˜ (l) h1···h<sup>l</sup> is labeled by h1:<sup>l</sup> in x<sup>t</sup> and x˜t, respectively, it holds that for each σ of the form σ = h<sup>1</sup> · · · h<sup>l</sup> (for some l ∈ [L]), there are unique p, p˜ ∈ P so that ν<sup>σ</sup> = ν(p,σ) = ν(˜p,σ) :

<sup>11</sup>Technically, we need a labeled version of Lemma [2.2,](#page-4-2) where the distribution x<sup>t</sup> are over the labeled set ∆(P); it is immediate to see that the proof of Lemma [2.2](#page-4-2) extends to the labeled case.

in particular, we have p = p (l) h1···h<sup>l</sup> , p˜ = ˜p (l) h1···h<sup>l</sup> . We can therefore bound

$$\begin{aligned}
& \text{Cal}_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T}) - 2\text{Cal}_T^{\|\cdot\|^2}(\tilde{\mathbf{x}}_{1:T}, y_{1:T}) \\
&= \sum_{l \in [L], h_{1:l} \in [0:H-1]^l} \left( \sum_{t=1}^T \mathbf{x}_t((p_{h_1\dots h_l}^{(l)}, h_1 \dots h_l)) \right) \cdot \|\nu_{h_1\dots h_l} - p_{h_1\dots h_l}^{(l)}\|^2 \\
&\quad - 2 \left( \sum_{t=1}^T \tilde{\mathbf{x}}_t((\tilde{p}_{h_1\dots h_l}^{(l)}, h_1 \dots h_l)) \right) \cdot \|\nu_{h_1\dots h_l} - \tilde{p}_{h_1\dots h_l}^{(l)}\|^2 \\
&= \sum_{l \in [L], h_{1:l} \in [0:H-1]^l} \frac{H^{L-l}}{L} \cdot \left( \left\| \|\nu_{h_1\dots h_l} - p_{h_1\dots h_l}^{(l)}\|^2 - 2 \left\| \|\nu_{h_1\dots h_l} - \tilde{p}_{h_1\dots h_l}^{(l)}\|^2 \right\|^2 \right) \right. \\
&\leq 2 \sum_{l \in [L], h_{1:l} \in [0:H-1]^l} \frac{H^{L-l}}{L} \cdot \left\| p_{h_1\dots h_l}^{(l)} - \tilde{p}_{h_1\dots h_l}^{(l)} \right\|^2 \\
&\leq \frac{2}{L} \sum_{l=1}^L \sum_{h_{1:l-1} \in [0:H-1]^{l-1}} \text{diam}(\mathcal{P})^2 \cdot H^{L-l} \\
&\leq \frac{2}{L} \sum_{l=1}^L \text{diam}(\mathcal{P})^2 \cdot H^{L-1} \\
&= \frac{2T \text{diam}(\mathcal{P})^2}{H},
\end{aligned} \tag{11}$$

where the second-to-last inequality uses that P<sup>H</sup>−<sup>1</sup> hl=0 p (l) h1···h<sup>l</sup> − p˜ (l) h1···h<sup>l</sup> ≤ diam(P) 2 for all choices of h<sup>1</sup> · · · hl−<sup>1</sup> (a consequence of Lemma [C.4](#page-26-1) and Lemma [C.2\)](#page-24-0).

Lemma C.4. *Fix any convex set* P ⊂ R <sup>d</sup> *and a convex function* R : P → <sup>R</sup>*. Fix a sequence* y1, . . . , y<sup>H</sup> ∈ P*, and set*

$$p_h = \frac{1}{h-1} \sum_{i=1}^{h-1} y_i \quad \forall h \in [H], h > 1, \quad \tilde{p}_h = \frac{1}{h} \sum_{i=1}^h y_i \quad \forall h \in [H],$$

*as well as* p<sup>1</sup> ∈ P *arbitrarily. Then*

$$\sum_{h=1}^H \|p_h - \tilde{p}_h\|^2 \leq 2 \cdot \text{diam}(\mathcal{P})^2.$$

*Proof.* Note that

$$\tilde{p}_h - p_h = \frac{y_h}{h} - \frac{1}{h(h-1)} \sum_{i=1}^{h-1} y_i,$$

which implies that kp˜<sup>h</sup> − phk <sup>2</sup> ≤ π 6 · diam(P) <sup>2</sup> < 2diam(P) 2 .

Applying Cauchy-Schwarz, we get the following corollary,

Corollary C.5. *Let* P ⊂ R <sup>d</sup> *be a bounded convex set and* k·k *be an arbitrary norm. Then,* TreeCal *(Algorithm [1\)](#page-22-0) guarantees that for an arbitrary sequence of outcomes* y1, . . . , y<sup>T</sup> ∈ P*, the* k·k *calibration error of its predictions* <sup>x</sup>1, . . . , <sup>x</sup><sup>T</sup> ∈ ∆(P) *is bounded* Calk·k T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ T *for* T ≥ (diamk·k(P)/) O(Rate(P,k·k)/<sup>2</sup> )

*Proof.* Using the fact that P p∈P P<sup>T</sup> <sup>t</sup>=1 xt(p) = 1 together with Jensen's inequality, we have

$$\begin{aligned} \frac{1}{T} \text{Cal}_T^{\|\cdot\|}(\mathbf{x}_{1:T}, y_{1:T}) &= \frac{1}{T} \sum_{p \in \mathcal{P}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\nu_p - p\| \\ &\leq \sqrt{\frac{1}{T} \sum_{p \in \mathcal{P}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\nu_p - p\|^2} \\ &= \sqrt{\frac{1}{T} \text{Cal}_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T})} \leq \epsilon \end{aligned}$$

for T ≥ (diam(P)/) O(Rate(P,k·k)/<sup>2</sup> ) by Theorem [3.1.](#page-7-0) Thus, Calk·k T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ T for T ≥ (diam(P)/) O(Rate(P,k·k)/<sup>2</sup> ) , incurring an additional factor of 2 in the exponent constant, as desired.

Finally, for the setting of centrally symmetric P, we can apply Lemma [2.4](#page-5-2) to directly relate this regret bound to the optimal possible rate of an online linear optimization problem.

Corollary C.6. *Let* P ⊂ R <sup>d</sup> *be a bounded centrally symmetric convex set and* k·k *be an arbitrary norm. Then,* TreeCal *(Algorithm [1\)](#page-22-0) guarantees that for an arbitrary sequence of outcomes* y1, . . . , y<sup>T</sup> ∈ P*, the* k·k *calibration error of its predictions* x1, . . . , x<sup>T</sup> ∈ ∆(P) *is bounded* Calk·k T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ T *for* T ≥ (diamk·k(P)/) O(RateOLO(P,k·k)/<sup>2</sup> )

*Proof.* Follows immediately by applying Lemma [2.4](#page-5-2) to Corollary [C.5.](#page-26-0)

# D Proofs for Section [4](#page-8-4)

In this section, we prove lower bounds on high-dimensional calibration that tell us that in order to achieve calibration error at most · T, we need to take T & exp(poly(1/)). First, in Appendix [D.1,](#page-27-2) we prove a lower bound for `<sup>1</sup> calibration over the d-dimensional simplex, and then, in Appendix [D.2,](#page-29-1) we prove a lower bound for `<sup>2</sup> calibration over the unit d-dimensional Euclidean ball.

# D.1 Lower bound on `<sup>1</sup> calibration

First, we prove Theorem [4.3](#page-8-0) which gives a lower bound on `<sup>1</sup> calibration over the simplex P = ∆<sup>d</sup> .

*Proof of Lemma [4.2.](#page-8-3)* Fix an algorithm Alg which ensures that Cal<sup>D</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ · T as in the statement of the lemma. We construct the following algorithm Alg<sup>0</sup> : it simulates Alg, but whenever Alg outputs the distribution x<sup>t</sup> ∈ ∆(P), Alg<sup>0</sup> chooses instead x 0 <sup>t</sup> ∈ ∆(P 0 ), defined by

$$\mathbf{x}'_t(p') := \sum_{p \in \mathcal{P}:} \mathbf{x}_t(p).$$

$$p' = \operatorname{argmin}_{q \in \mathcal{P}'} \langle q, p \rangle$$

To simplify notation, we define BR(p) := argminq∈P<sup>0</sup> hq, pi. It follows that, for any oblivious adversary choosing a (random) sequence y1, . . . , y<sup>T</sup> ∈ P,

$$\begin{aligned}
& \text{FullSwapReg}_T(\mathbf{x}'_{1:T}, \ell(\cdot, y_{1:T})) \\
&= \sup_{\pi: \mathcal{P}' \rightarrow \mathcal{P}'} \sum_{p' \in \mathcal{P}} \sum_{t \in [T]} \mathbf{x}'_t(p') \cdot (\langle y_t, p' - \pi(p') \rangle) \\
&= \sup_{\pi: \mathcal{P}' \rightarrow \mathcal{P}'} \sum_{p \in \mathcal{P}} \sum_{t \in [T]} \mathbf{x}_t(p) \cdot (\langle y_t, \text{BR}(p) - \pi(\text{BR}(p)) \rangle) \\
&= \sup_{\pi: \mathcal{P}' \rightarrow \mathcal{P}'} \sum_{p \in \mathcal{P}} \left( \sum_{t \in [T]} \mathbf{x}_t(p) \right) \cdot (\langle \nu_p, \text{BR}(p) - \pi(\text{BR}(p)) \rangle) \\
&= \sup_{\pi: \mathcal{P}' \rightarrow \mathcal{P}'} \sum_{p \in \mathcal{P}} \left( \sum_{t \in [T]} \mathbf{x}_t(p) \right) \cdot (\langle \nu_p - p, \text{BR}(p) - \pi(\text{BR}(p)) \rangle + \langle p, \text{BR}(p) - \pi(\text{BR}(p)) \rangle) \\
&\leq \sup_{\pi: \mathcal{P}' \rightarrow \mathcal{P}'} \sum_{p \in \mathcal{P}} \left( \sum_{t \in [T]} \mathbf{x}_t(p) \right) \cdot (\|\nu_p - p\| \cdot \|\text{BR}(p) - \pi(\text{BR}(p))\|_*) \\
&\leq \text{diam}_{\|\cdot\|_*}(\mathcal{P}') \cdot \text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}),
\end{aligned}$$

where in the final inequality we have used the fact that kBR(p) − π(BR(p))k? ≤ diamk·k? (P 0 ).

For p > 0, d ∈ N, write B d p := {x ∈ <sup>R</sup> d | kxk<sup>p</sup> ≤ 1} to denote the unit `<sup>p</sup> norm ball.

To map the lower bound Theorem [4.1](#page-8-2) from the k·k<sup>1</sup> -norm unit ball B d 1 to the simplex and arrive at the desired contradiction using the above lemma, we use the following.

Lemma D.1. *Fix* d ∈ <sup>N</sup>*, and write* D(x, y) := kx − yk<sup>1</sup> *for* x, y ∈ <sup>R</sup> d *. Suppose that there is an algorithm* Alg *for calibration over the domain* P = ∆<sup>2</sup>d+1 *which produces* x1:<sup>T</sup> *given the choices of an adversary* y1:<sup>T</sup> *achieving calibration error* Cal<sup>D</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≤ R(T)*, for* T ∈ <sup>N</sup>*. Then there is an algorithm* Alg<sup>0</sup> *for calibration over the domain* B d <sup>1</sup> *which produces* x 0 1:T *given* y 0 1:T *achieving calibration error* Cal<sup>D</sup> T (x 0 1:T , y<sup>0</sup> 1:T ) ≤ R(T)*.*

*Proof of Lemma [D.1.](#page-28-0)* We define a mapping φ : B d <sup>1</sup> → ∆<sup>2</sup>d+1 as follows: for y ∈ B<sup>d</sup> <sup>1</sup> ⊂ <sup>R</sup> d , we define

$$\phi(y)_i = \begin{cases} [y_j]_+ & : i = 2j-1, j \in [d] \\ [y_j]_- & : i = 2j, j \in [d] \\ 1 - \|y\| & : i = 2d+1. \end{cases}$$

It is straightforward to see that φ has a left inverse ψ, defined as follows: for z ∈ ∆<sup>2</sup>d+1 ,

$$\psi(z)_i = z_{2i-1} - z_{2i}, \quad i \in [d],$$

so that ψ ◦ φ(y) = y for all y ∈ <sup>R</sup> d .

We define the algorithm Alg<sup>0</sup> as follows: given y 0 <sup>t</sup> ∈ B<sup>d</sup> <sup>1</sup> ⊂ <sup>R</sup> d , it defines y<sup>t</sup> ∈ ∆<sup>2</sup>d+1 by y<sup>t</sup> = φ(y 0 t ). Alg<sup>0</sup> then feeds y<sup>t</sup> into Alg, and if we denote the distribution output by Alg at time step t by xt, Alg<sup>0</sup> then plays the push-forward measure x 0 t := ψ ◦ x<sup>t</sup> ∈ ∆(B d 1 ).

Our bound on the calibration error of Alg gives

$$\text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}) = \sum_{p \in \Delta^{2d+1}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\nu_p - p\|_1 \leq R(T),$$

where ν<sup>p</sup> = P<sup>T</sup> P<sup>t</sup>=1 <sup>x</sup>t(p)·y<sup>t</sup> T <sup>t</sup>=1 xt(p) ∈ ∆<sup>2</sup>d+1. For p <sup>0</sup> ∈ B<sup>d</sup> 1 , let us denote ν 0 <sup>p</sup><sup>0</sup> := P<sup>T</sup> <sup>t</sup>=1 x t (p )·y <sup>P</sup> <sup>t</sup> T <sup>t</sup>=1 x (p<sup>0</sup>) = ψ P<sup>T</sup> <sup>t</sup>=1 x (p P )·y<sup>t</sup> T <sup>t</sup>=1 x 0 t (p<sup>0</sup>) , using linearity of ψ.

We may now bound the calibration error of Alg<sup>0</sup> by

$$\begin{aligned} \text{Cal}_T^D(\mathbf{x}'_{1:T}, y'_{1:T}) &= \sum_{p' \in \mathcal{B}_1^d} \left( \sum_{t=1}^T \mathbf{x}'_t(p') \right) \cdot \|\nu'_{p'} - p'\|_1 \\ &\leq \sum_{p \in \Delta^{2d+1}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\psi(\nu_p) - \psi(p)\|_1 \\ &\leq \text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}). \end{aligned}$$

*Proof of Theorem [4.3.](#page-8-0)* Suppose to the contrary that there was an algorithm Alg which bounded calibration error by T for T ≤ exp(c · min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>6</sup>}). Then by Lemma [D.1,](#page-28-0) for d <sup>0</sup> = b(d − 1)/2c there is an algorithm Alg<sup>0</sup> for calibration on the domain B d <sup>1</sup> ⊂ <sup>R</sup> d produces x 0 1:T given y 0 1:T satisfying Cal<sup>D</sup> T (x 0 1:T , y<sup>0</sup> 1:T ) ≤ · T for any T ≤ exp(c · min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>6</sup>}).

We now apply Lemma [4.2](#page-8-3) for P = B d <sup>1</sup> ⊂ <sup>R</sup> d , the norm given by the `<sup>1</sup> norm k · k1, and P 0 := [−1, 1]<sup>d</sup> . Note that diamk·k∞(P 0 ) = 1. Then Lemma [4.2](#page-8-3) ensures that there is an algorithm Alg<sup>00</sup> which chooses x 00 1 , . . . , x 00 <sup>T</sup> ∈ ∆(P 0 ) which ensures that for every oblivious adversary choosing y 00 1 , . . . , y<sup>00</sup> <sup>T</sup> ∈ B<sup>d</sup> 0 , we have FullSwapReg<sup>T</sup> (x 00 1:T , `(·, y<sup>00</sup> 1:T )) ≤ · T.

But if T ≤ exp(c[4](#page-8-2).1·min{(d 0 ) <sup>1</sup>/<sup>14</sup>, −1/<sup>6</sup>}), we have a contradiction to Theorem [4.1,](#page-8-2) thus completing the proof of the theorem.

# D.2 Lower bound for `<sup>2</sup> calibration

Next, we prove a lower bound for `<sup>2</sup> calibration.

Theorem D.2. *There is a sufficiently small constant* c > 0 *so that the following holds. Write* D(p, p<sup>0</sup> ) = kp − p 0k2 *and fix any* > 0*,* d ∈ N*. Then for any* T ≤ exp(c · min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>7</sup>})*, there is an oblivious adversary producing a sequence* y1, . . . , y<sup>T</sup> ∈ B<sup>d</sup> 2 *so that for any learning algorithm producing* x1, . . . , x<sup>T</sup> ∈ ∆(B d 2 )*,* Cal<sup>D</sup> T (x1:<sup>T</sup> , y1:<sup>T</sup> ) ≥ · T*.*

*Proof.* Fix > 0, d ∈ N, and write ˜ = 6/7 . We may assume without loss of generality that d ≤ ˜ −14/6 , so that min{d 1/14 , ˜ <sup>−</sup>1/<sup>6</sup>} = min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>7</sup>} = d <sup>1</sup>/<sup>14</sup>: if this were not the case, we simply use the adversary resulting from ˜ <sup>−</sup>14/<sup>6</sup> dimensions and project the forecaster's predictions down into this lower-dimensional subspace, which can only decrease calibration error. Now suppose to the contrary that there was an algorithm Alg which bounded calibration error by T for T ≤ exp(c·min{d <sup>1</sup>/<sup>14</sup>, −1/<sup>7</sup>}) = exp(c·d <sup>1</sup>/<sup>14</sup>). Then by Lemma [4.2](#page-8-3) with P = B d 2 and norm k·k = k·k<sup>2</sup> , for any subset P <sup>0</sup> ⊂ B<sup>d</sup> <sup>2</sup> we get that there is an algorithm which chooses x 0 1 , . . . , x 0 <sup>T</sup> ∈ ∆(P 0 ) and which ensures that for every oblivious adversary choosing y1, . . . , y<sup>T</sup> ∈ B<sup>d</sup> 2 , we have

$$\text{FullSwapReg}_T(\mathbf{x}'_{1:T}, (\langle \cdot, y_1 \rangle, \dots, \langle \cdot, y_T \rangle)) \leq \epsilon \cdot T. \quad (12)$$

On the other hand, the oblivious adversary of Theorem [4.1](#page-8-2) guarantees a subset X ⊂ [−1, 1]<sup>d</sup> ⊂ and an oblivious adversary producing a sequence v1, . . . , v<sup>T</sup> ∈ <sup>R</sup> <sup>d</sup> with kvtk<sup>∞</sup> ≤ <sup>d</sup> <sup>−</sup>13/<sup>14</sup> for all t ∈ [T], so that

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, (\langle \cdot, v_1 \rangle, \dots, \langle \cdot, v_T \rangle)) \geq \tilde{\epsilon} \cdot T \quad (13)$$

as long as T ≤ exp(c4.<sup>1</sup> · d <sup>1</sup>/<sup>14</sup>). We have kvtk<sup>2</sup> ≤ <sup>d</sup> <sup>1</sup>/2−13/<sup>14</sup> = d −3/7 for all t, and scaling X down by a factor of 1/ [√](#page-8-2) d (i.e., letting P <sup>0</sup> = X / √ d) and all vectors v<sup>t</sup> up by a factor of d 3/7 (i.e., letting v 0 <sup>t</sup> = √ d · v<sup>t</sup> ensures that any algorithm producing x 0 1 , . . . , x 0 <sup>T</sup> ∈ P<sup>0</sup> must still have full swap regret

FullSwapReg\_T(
$$\mathbf{x}'_{1:T}, (\langle \cdot, v'_1 \rangle, \dots, \langle \cdot, v'_T \rangle)$$
) >  $\tilde{\epsilon} \cdot T \cdot d^{-1/14} \geq \tilde{\epsilon}^{7/6} \cdot T = \epsilon \cdot T$ ,

# E Pure calibration and pure full swap regret

### E.1 Pure calibration

In certain settings of calibration, the learner is required to randomly select a pure forecast p<sup>t</sup> ∈ P rather than a distribution x<sup>t</sup> ∈ ∆(P). In these settings, the above definition of calibration is instead referred to as "pseudo-calibration". Here, we stick to calling the above calibration, as we believe it to be the more natural definition, and instead refer to this alternative setting as "pure-calibration". The learning task changes as follows.

At each time step t ∈ [T]:

- The learning algorithm chooses a distribution x<sup>t</sup> ∈ ∆(P).
- The adversary observes x<sup>t</sup> and chooses an *outcome* y<sup>t</sup> ∈ P.
- The learner samples p<sup>t</sup> ∼ xt.

We adjust the definitions of the "pure average outcome" and "pure-calibration" accordingly:

$$\dot{\nu}_p := \frac{\sum_{t=1}^T \mathbb{1}[p_t = p] \cdot y_t}{\sum_{t=1}^T \mathbb{1}[p_t = p]}, \quad \text{PureCal}_T^D(p_{1:T}, y_{1:T}) := \sum_{p \in \mathcal{P}} \left( \sum_{t=1}^T \mathbb{1}[p_t = p] \right) \cdot D(\dot{\nu}_p, p)$$

Algorithm 3 SampleTreeCal(P, T, H, L, S)

Require: Action set P ⊂ R d , time horizon T, repetition parameter S parameters H, L with T /S ≤ H<sup>L</sup>.

1: Instantiate an instance TreeCal(P, T /S, H, L). 2: for 1 ≤ i ≤ T /S do 3: Let x<sup>i</sup> ∈ ∆(P) denote the prediction of TreeCal at step i. 4: for 1 ≤ j ≤ S do 5: Sample pS(i−1)+<sup>j</sup> ∼ x<sup>i</sup> , and observe outcome yS(i−1)+<sup>j</sup> . 6: end for 7: Feed the outcome y¯<sup>i</sup> := <sup>1</sup> S P<sup>S</sup> <sup>j</sup>=1 yS(i−1)+<sup>j</sup> to TreeCal. 8: end for

To obtain a bound on the (expected) pure calibration error, we use a slight modification of TreeCal, namely SampleTreeCal (Algorithm [3\)](#page-30-1). It functions identically to TreeCal except that for each time step t of TreeCal, it samples S actions from x<sup>t</sup> on each of S contiguous time steps. (Hence, TreeCal is used with time horizon T /S.) At a high level, we will use an appropriate concentration inequality to show that the calibration upper bound of Theorem [3.1](#page-7-0) implies a *pure calibration* upper bound for SampleTreeCal.

Theorem E.1 (Pure calibration error). *Let* P ⊂ R <sup>d</sup> *be a bounded convex set and* k·k *be an arbitrary norm with unit dual ball* L := {f ∈ <sup>R</sup> d | kfk? ≤ 1}*. Then,* SampleTreeCal *(Algorithm [3,](#page-30-1) with an appropriate choice of parameters* H, L, S*) guarantees that for an arbitrary sequence of outcomes* y1, . . . , y<sup>T</sup> ∈ P*, the* k·k<sup>2</sup> *calibration error of its predictions* x1, . . . , x<sup>T</sup> ∈ ∆(P) *is bounded as follows:*

$$\mathbb{E}[\text{PureCal}_T^{\|\cdot\|}(p_{1:T}, y_{1:T})] \leq \epsilon T, \quad \text{for} \quad T \geq \text{Rate}(\mathcal{L}, \|\cdot\|_*) \cdot (\text{diam}_{\|\cdot\|}(\mathcal{P})/\epsilon)^{O(\text{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon^2)}.$$

*Proof.* The proof uses Theorem [3.1](#page-7-0) together with an appropriate concentration inequality, and closely follows that of [\[Pen25,](#page-10-0) Lemma 3.4].

Fix any 1 ≤ i ≤ T /S and 1 ≤ j ≤ S, and let FS(i−1)+<sup>j</sup> denote the σ-algebra generated by y1, . . . , yS(i−1)+j+1 and p1, . . . , pS(i−1)+<sup>j</sup> ; since TreeCal is deterministic, it follows that x1, . . . , x<sup>i</sup> ∈ ∆(P) are Fi-measurable. For any 1 ≤ j ≤ S, we have that, for any p ∈ supp(xi),

$$\mathbb{E} \left[ (p - y_{S(i-1)+j}) \cdot \mathbb{1}[pS_{(i-1)+j} = p] \mid \mathcal{F}_{S(i-1)+j-1} \right] = (p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p).$$

Fixing any i ∈ [T /S], By Lemma [E.2](#page-32-0) applied to the sequence MS(i−1)+<sup>j</sup> = (p − yS(i−1)+<sup>j</sup> ) · <sup>1</sup>[pS(i−1)+<sup>j</sup> = p], for 1 ≤ j ≤ S (and the filtration FS(i−1)+<sup>j</sup> ), we see that

$$\mathbb{E} \left[ \left\| \sum_{j=1}^S (p - y_{S(i-1)+j}) \cdot \mathbb{1}[p_{S(i-1)+j} = p] - \sum_{j=1}^S (p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p) \right\| \right] \leq \text{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{8S \cdot \text{Rate}(\mathcal{L}, \|\cdot\|_*)}.$$

It follows by summing over the L values of p ∈ supp(xi) that

$$\begin{aligned} & \mathbb{E} \left[ \sum_{p \in \mathcal{P}} \left\| \sum_{j=1}^S (p - y_{S(i-1)+j}) \cdot \mathbb{1}[p_{S(i-1)+j} = p] - \sum_{j=1}^S (p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p) \right\| \right] \\ & \leq L \cdot \text{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{8S \cdot \text{Rate}(\mathcal{L}, \|\cdot\|_*)} \leq \epsilon \cdot S, \end{aligned} \quad (14)$$

as long as S ≥ 8·Rate(L,k·k? )·diamk·k(P) ·L <sup>2</sup> .

The guarantee of Corollary [C.5](#page-26-0) gives that, as long as T /S ≥ (diamk·k(P)/) O(Rate(P,k·k)/<sup>2</sup> ) , then

$$\begin{aligned} \text{Cal}_T^{\parallel \parallel}(\mathbf{x}_{1:T/S}, \bar{y}_{1:T/S}) &= \sum_{p \in \mathcal{P}} \left\| \sum_{i=1}^{T/S} \mathbf{x}_i(p) \cdot (p - \bar{y}_i) \right\| \\ &= \sum_{p \in \mathcal{P}} \left\| \sum_{i=1}^{T/S} \frac{1}{S} \sum_{j=1}^S \mathbf{x}_i(p) \cdot (p - y_{S(i-1)+j}) \right\| \leq \frac{\epsilon T}{S}. \end{aligned} \quad (15)$$

By combining Equations [\(14\)](#page-31-0) and [\(15\)](#page-31-1), it follows that for an arbitrary adaptive adversary who chooses a sequence y1, . . . , y<sup>T</sup> ∈ P,

$$\begin{aligned} & \mathbb{E} \left[ \text{PureCal}_T^{\|\cdot\|} (p_{1:T}, y_{1:T}) \right] \\ &= \mathbb{E} \left[ \sum_{p \in \mathcal{P}} \left\| \sum_{t=1}^T (p - y_t) \cdot \mathbb{1}[p_t = p] \right\| \right] \\ &\leq \mathbb{E} \left[ \sum_{p \in \mathcal{P}} \left\| \sum_{i=1}^{T/S} \sum_{j=1}^S (p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p) \right\| + \sum_{i=1}^{T/S} \left\| \sum_{j=1}^S ((p - y_{S(i-1)+j}) \cdot (\mathbb{1}[p_{S(i-1)+j} = p] - \mathbf{x}_i(p))) \right\| \right] \\ &\leq 2\epsilon T. \end{aligned}$$

The result follows by rescaling and our choice of L = O(Rate(P, k·k)/<sup>2</sup> ).

As example applications of Theorem [E.1:](#page-30-2)

- When k·k is the `<sup>1</sup> norm and P is the simplex, we have diamk·k(P) = 1, L = {f ∈ <sup>R</sup> d kfk<sup>∞</sup> ≤ 1} satisfies Rate(L, k·k? ) ≤ d (as we can take the function R(x) = kxk 2 2 ), which gives that for T ≥ d O(1/<sup>2</sup> ) , we can have <sup>E</sup>[PureCalk·k<sup>1</sup> T ] ≤ T. This result recovers the main upper bound of [\[Pen25\]](#page-10-0) (Theorem 1.1 therein).
- When k·k is the `<sup>2</sup> norm and P is the unit `<sup>2</sup> ball, we have diamk·k(P) = 1, L = {f ∈ <sup>R</sup> d kfk<sup>2</sup> ≤ 1} satisfies Rate(L, k·k? ) ≤ 1 (as we can take the function R(x) = kxk 2 2 ), which gives that for T ≥ exp(O(1/<sup>2</sup> )), we can have <sup>E</sup>[PureCalk·k<sup>1</sup> T ] ≤ T.

# E.2 Sequential law of large numbers

Fix a convex set P ⊂ R d and a norm k·k on <sup>R</sup> d . We define

$$\mathcal{R}_n(\mathcal{P}, \|\cdot\|) := \sup_{\mathbf{p}} \mathbb{E}_\epsilon \left[ \left\| \frac{1}{n} \sum_{i=1}^n \epsilon_i \mathbf{p}_i(\epsilon) \right\| \right],$$

where the supremum is over all sequences of mappings p1, . . . , pn, where p<sup>i</sup> : {−1, 1} <sup>i</sup>−<sup>1</sup> → P, and the expectation is over an i.i.d. sequence of Rademacher random variables = (1, . . . , n), <sup>i</sup> ∼ Unif({±1}). The below lemma (essentially contained in [\[RST15\]](#page-10-19)) establishes a martingale law of large numbers for P-valued martingales, in terms of geometric properties of P and k·k.

Lemma E.2 ([\[RST15\]](#page-10-19)). *Consider a convex set* P ⊂ R <sup>d</sup> *a norm* k·k *on* <sup>R</sup> d *, and let* M1, . . . , M<sup>n</sup> *denote a sequence of random variables adapted to a filtration* (Fi)i∈[n] *. Let* L = {f | kfk? ≤ 1} *be the unit ball of the dual norm* k·k*. Then*

$$\mathbb{E} \left[ \left\| \sum_{i=1}^n M_i - \mathbb{E}[M_i \mid \mathcal{F}_{i-1}] \right\| \right] \leq \text{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{8n \cdot \text{Rate}(\mathcal{L}, \|\cdot\|_*)}.$$

*Proof.* By applying an appropriate translation to P, we can assume that P contains the origin. We apply Theorem 2 of [\[RST15\]](#page-10-19) with the domain Z equal to P and the function class F equal to the class of mappings {z 7→ hz, fi : kfk? ≤ 1} indexed by unit-dual norm linear functions on Z. The theorem implies that

$$\begin{aligned}\mathbb{E} \left[ \frac{1}{n} \left\| \sum_{i=1}^n M_i - \mathbb{E}[M_i \mid \mathcal{F}_{i-1}] \right\| \right] &\leq 2 \cdot \sup_{\mathbf{p}} \mathbb{E}_{\epsilon} \left[ \sup_{\|f\|_* \leq 1} \frac{1}{n} \left\langle \sum_{i=1}^n \epsilon_i \mathbf{p}_i(\epsilon), f \right\rangle \right] \\ &= 2 \cdot \mathcal{R}_n(\mathcal{P}, \|\cdot\|).\end{aligned}$$

Write L = {f ∈ <sup>R</sup> d : kfk? ≤ 1} denote the unit ball for the dual norm k·k? . Proposition 16 of [\[RST15\]](#page-10-19) gives that, if there is a function R : L → R which is 1-strongly convex with respect to k·k? and which has range ρ, then Rn(P, k·k) ≤ q2<sup>ρ</sup> n · diamk·k(P). In particular, Rn(P, k·k) ≤ diamk·k(P) · q2Rate(L,k·k? ) n .