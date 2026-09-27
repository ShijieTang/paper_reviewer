# High-Dimensional Calibration From Swap Regret

| Noah Golowich†   | Mehryar Mohri‡   | Jon Schneider§     |
|------------------|------------------|--------------------|
| nzg@mit.edu      | mohri@google.com | jschnei@google.com |

## Abstract

We study the online calibration of multi-dimensional forecasts over an arbitrary convex set P ⊂ R
drelative to an arbitrary norm k·k. We connect this with the problem of external regret minimization for online linear optimization, showing that if it is possible to guarantee O(
√ρT) worst-case regret after T rounds when actions are drawn from P and losses are drawn from the dual k·k∗
unit norm ball, then it is also possible to obtain -calibrated forecasts after T = exp(O(ρ/2)) rounds.

When P is the d-dimensional simplex and k·k is the `1-norm, the existence of O(
√T log d)-regret algorithms for learning with experts implies that it is possible to obtain -calibrated forecasts after T = exp(O(log d/2)) = d O(1/2)rounds, recovering a recent result of [Pen25]. Interestingly, our algorithm obtains this guarantee without requiring access to any online linear optimization subroutine or knowledge of the optimal rate ρ - in fact, our algorithm is identical for every setting of P and k·k. Instead, we show that the optimal regularizer for the above OLO problem can be used to upper bound the above calibration error by a swap regret, which we then minimize by running the recent TreeSwap algorithm ([DDFG24, PR24]) with Follow-The-Leader as a subroutine. The resulting algorithm is highly efficient and plays a distribution over simple averages of past observations in each round.

Finally, we prove that any online calibration algorithm that guarantees T `1calibration error over the d-dimensional simplex requires T ≥ exp(poly(1/)) (assuming d ≥ poly(1/)). This strengthens the corresponding d Ω(log 1/)lower bound of [Pen25], and shows that an exponential dependence on 1/ is necessary.

## 1 Introduction

Consider the problem faced by a forecaster who must report probabilistic predictions for a sequence of events (e.g. whether it will rain or not tomorrow). One of the most common methods to evaluate the quality of such a forecaster is to verify whether they are *calibrated*: for example, does it indeed rain with probability 40% on days where the forecaster makes this prediction? In addition to calibration being a natural property to expect from predictions, several applications across machine learning, fairness, and game theory require the ability to produce online calibrated predictions [ZME20, GPSW17, HJKRR18, FV97]. When events have binary outcomes, calibration can be quantified by the notion of expected calibration error, which measures the expected distance between a prediction made by a forecaster and the actual empirical probability of the outcome on the days where they made that prediction. In a seminal result by Foster and Vohra [FV98], it was proved that it is possible for an online forecaster to efficiently
∗MIT. †MIT. Supported by a NSF Graduate Research Fellowship and a Fannie & Hertz Foundation Graduate Fellowship.

‡Google Research and Courant Institute of Mathematical Sciences, New York. §Google Research.

guarantee a sublinear calibration error of O(T
2/3) against any adversarial sequence of T binary events. Equivalently, this can be interpreted as requiring at most O(
−3) rounds of forecasting to guarantee an  per-round calibration error on average.

However, many applications require forecasting sequences of *multi-dimensional* outcomes. The previous definition of calibration error easily extends to the multi-dimensional setting where predictions and outcomes belong to a d-dimensional convex set P ⊂ R
d. Specifically, if a forecaster makes a sequence of predictions p1, p2, . . . , pT ∈ P for the outcomes y1, y2, . . . , yT ∈ P, their k·k-calibration error (for any norm k·k over R
d) is given by

$$\mathsf{C a l}_{T}^{\parallel\cdot\parallel}=\sum_{t=1}^{T}\left\|p_{t}-\nu_{p_{t}}\right\|$$

where νptis the average of the outcomes yt on rounds where the learner predicted pt.

The algorithm of Foster and Vohra extends to the multidimensional calibration setting, but at the cost of producing bounds that decay exponentially in the dimension d. In particular, their algorithm only guarantees that the forecaster achieves an average calibration error of  after (1/)
Ω(d)rounds. Until recently, no known algorithm achieved a sub-exponential dependence on d in any non-trivial instance of multi-dimensional calibration. In 2025, [Pen25] presented a new algorithm for high-dimensional calibration, demonstrating that it is possible to obtain `1-calibration rates of T in d O(1/2)rounds for predictions over the d-dimensional simplex (i.e., multi-class calibration). In particular, this is the first known algorithm achieving polynomial calibration rates in d for fixed constant . [Pen25] complements this with a lower bound, showing that in the worst case d Ω(log 1/)rounds are necessary to obtain this rate (implying that a fully polynomial bound poly(d, 1/) is impossible).

## 1.1 Our Results

Although the algorithm of [Pen25] is simple to describe, its analysis is fairly nuanced and tailored to `1-calibration over the simplex (e.g., by analyzing the KL divergence between predictions and distributions of historical outcomes). We present a very similar algorithm (TreeCal) for multidimensional calibration over an arbitrary convex set P ⊂ R
d, but with a simple, unified analysis that provides simultaneous guarantees for calibration with respect to any norm k·k. In particular, we prove the following theorem.

Theorem 1.1 (Informal restatement of Corollary C.5). Fix a convex set P and a norm k · k. Assume there exists a function R : P → R that is 1-strongly-convex with respect to k · k and has range
(maxx∈P R(x) − minp∈P R(x)) at most ρ*. Then* TreeCal guarantees that the calibration error of its predictions is bounded by Calk·k T ≤ T for T ≥ (diamk·k(P)/)
O(ρ/2).

Interestingly, the function R(p) and parameter ρ appearing in the statement of Theorem 1.1 have an independent learning-theoretic interpretation: if we consider the *online linear optimization* problem where a learner plays actions in P and the adversary plays linear losses that are unit bounded in the dual norm k·k∗
, then it is possible for the learner to guarantee a regret bound of at most O(
√ρT) by playing Follow-The-Regularized-Leader (FTRL) with R(p) as a regularizer. In fact, since universality results for mirror descent guarantee that some instantiation of FTRL achieves near-optimal rates for online linear optimization (as long as the action and loss sets are centrally convex) [SST11, GSJ24], this allows us to relate the performance of Theorem 3.1 directly to what rates are possible in online linear optimization.

Corollary 1.2 (Informal restatement of Corollary C.6). Let P ⊆ R
d be a centrally symmetric convex set, and let L = {y ∈ R
d| kyk∗ ≤ 1} for some norm k·k*. Then if there exists an algorithm for* online linear optimization with action set P and loss set L *that incurs regret at most* O(
√ρT),
TreeCal *guarantees that the calibration error of its predictions is bounded by* Calk·k T ≤ T for T ≥ (diamk·k(P)/)
O(ρ/2).

Theorem 1.1 and its corollary allow us to immediately recover several existing and novel bounds on calibration error in a variety of settings:
- When P is the d-simplex ∆d and k·k is the `1-norm, the existence of the negative entropy regularizer R(x) = Pd i=1 xilog xi (which is 1-strongly convex w.r.t. the `1 norm with range ρ = log d) implies that the `1 calibration error of TreeCal is at most (1/)
O(log d/2) =
d O˜(1/2). This recovers the result of [Pen25].

- When P is the `2 ball and k·k is the `2 norm, the Euclidean regularizer (R(x) = kxk 2)
implies a calibration bound of (1/)
O(1/2)(notably, this bound is independent of d).

It should be emphasized here that running TreeCal does not require any online linear optimization subroutine, nor any knowledge of these regularizers R(x) or optimal rates ρ. TreeCal has no functional dependence on any specific k·k. It achieves k·k-calibration at the above rate (Theorem 1.1)
for all k·k simultaneously. The TreeCal algorithm is nearly identical5to the algorithm of [Pen25] –
both algorithms initialize a tree of sub-forecasters and at each round play a uniform combination of some subset of them (see Figure 1). The novelty in our analysis stems from the observation that TreeCal is simply a specific instantiation of the TreeSwap swap regret minimization algorithm [DDFG24, PR24] and can be analyzed directly in this way. In particular, our analysis consists of the following steps:
1. First, minimizing calibration error can be reduced to minimizing swap regret, generalizing an idea of [LSS25, FKO+25]. That is, it is possible to assign the learner loss functions
`t : P → R at each round such that their calibration error is upper bounded by the gap between the total loss they received, and the minimal loss they could have received after applying an arbitrary "swap function" π : *P → P* to their predictions. In fact, any strongly convex function R (w.r.t. the norm k·k) gives rise to one such reduction, by setting the loss function `t(p) to equal the Bregman divergence DR(yt|p).

2. Second, the TreeSwap algorithm of [DDFG24, PR24] provides a general recipe for converting external regret minimization algorithms into swap regret minimization algorithms. We obtain TreeCal by plugging in the Follow-The-Leader algorithm (the learning algorithm which simply always best responds to the current history) into TreeSwap.

3. Instead of analyzing the swap regret bound of TreeSwap with Follow-The-Leader (which may not have a good enough external regret bound, as discussed in Section 3.3), we instead analyze the swap regret of TreeSwap with *Be-The-Leader* (the fictitious algorithm that best responds to the current history, including the current round). Though it is not possible to actually implement Be-The-Leader due to its clairvoyance, we use it as a tool for analysis.

We then relate the calibration error of TreeSwap with *Be-The-Leader* to that of TreeSwap with *Follow-The-Leader* using the fact that Be-The-Leader and Follow-The-Leader make similar predictions.

In the above step 1, we will choose R to be k·k-norm 1-strongly convex, which guarantees that DR(y|p) ≥ ky − pk 2. Going through the analysis, this actually leads to the stronger guarantee that TreeCal minimizes *squared-norm* calibration error. Theorem 1.3 (Informal restatement of Theorem 3.1). Fix a convex set P and a norm k · k*. Assume* there exists a function R : P → R that is 1-strongly-convex with respect to k · k and has range
(maxx∈P R(x) − minp∈P R(x)) at most ρ*. Then* TreeCal *guarantees that the calibration error of* its predictions is bounded by Calk·k2 T ≤ T for T ≥ (diamk·k(P)/
√)
O(ρ/).

Note here we have only singly-exponential dependence on 1/. We arrive at Theorem 1.1 as a corollary of this result by simply applying Cauchy-Schwarz. Finally, we strengthen the lower bound of [Pen25] by showing an exponential dependence on 1/ is necessary.

Theorem 1.4 (Informal restatement of Theorem 4.3). There is a sufficiently small constant c > 0 so that the following holds. Fix any  > 0, d ∈ N*. Then for any* T ≤ exp(c · min{d 1/14, −1/6}), there is an oblivious adversary producing a sequence of outcomes so that any learning algorithm must incur `1*-calibration error* Calk·k1 T ≥  · T.

5One minor difference is that the algorithm of [Pen25] regularizes each sub-forecaster by slightly mixing their prediction with the uniform distribution, which TreeCal does not require.

Unlike the lower bound of [Pen25], this lower bound requires no specialized construction. Instead, it follows from the original observation of [FV98] that any algorithm for online calibration can be used to construct an algorithm for swap regret minimization by simply best responding to a sequence of calibrated predictions of the adversary's losses. The existing lower bound for swap regret in
[DFG+24] then immediately precludes the existence of sufficiently strong calibration bounds (e.g.,
of the form d O(log 1/), which was still allowed by the work of [Pen25]).

Using a similar technique, in Theorem D.2, we show a similar lower bound for `2 calibration, namely that exp(Ω(min{d 1/14, −1/7})) time steps are needed to achieve `2 calibration error at most  · T.

For d ≥ 
−2, this bound is tight up a polynomial in the exponent.

We discuss additional related work in the appendix.

## 2 Setup

For a positive integer n, we let [0 : n − 1] denote the sequence 0, 1*, . . . , n* − 1, and [n] denote the sequence 1, 2*, . . . , n*. We say a convex set S ⊆ R
dis centrally symmetric if s ∈ S ⇔ −s ∈ S
for all s ∈ R
d. A norm k·k is a function corresponding to a convex, bounded, centrally-symmetric set S of the form ksk = inf {c ∈ R≥0|s ∈ cS}. The corresponding *dual norm* is defined kvk∗ =
sup {hs, vi | ksk ≤ 1}.

## 2.1 Calibration

We consider the following setting of *multi-dimensional calibration*. Positive integers d ∈ N representing the number of dimensions and T ∈ N representing the number of rounds are given. We let P ⊂ R
d denote a bounded convex subset of R
d. An *adversary* and a *learning algorithm* interact for a total of T timesteps; at each time step t ∈ [T]:
- The learning algorithm chooses a distribution6 xt ∈ ∆(P) with finite support.

- The adversary observes xt and chooses an outcome yt ∈ P.

In order for the learner to be calibrated, we would like the average outcome conditional on the learner making a specific prediction p to be "close" to p. We formalize this as follows. For a point p ∈ P, we define νp to be the average outcome conditioned on the learner predicting p, that is:

$$\nu_{p}:={\frac{\sum_{t=1}^{T}\mathbf{x}_{t}(p)\cdot y_{t}}{\sum_{t=1}^{T}\mathbf{x}_{t}(p)}}.$$
. (1)
Fix a distance measure D : *P × P →* R≥0, namely an arbitrary non-negative valued function on P × P. Given a distance measure D, we define the D*-calibration error* as follows:

$$\mathbf{Cal}_{T}^{D}(\mathbf{x}_{1:T},y_{1:T}):=\sum_{p\in{\mathcal{P}}}\left(\sum_{t=1}^{T}\mathbf{x}_{t}(p)\right)\cdot D(\nu_{p},p).$$
$$(1)$$

In the event that D(*p, q*) = kp − qk, we will write Calk·k T(x1:T , y1:T ) = CalD
T(x1:T , y1:T ), and we define Calk·k2 T(x1:T , y1:T ) analogously.

## 2.2 Regret Minimization

For a sequence of actions p1, · · · , pT ∈ P and loss functions `1, · · · , `T : P → R, we define

$\left(\begin{array}{cccc}\text{Ext}\text{Reg}_{T}(p_{1:T},\ell_{1:T}):=\sup_{p^{*}\in\mathcal{P}}\sum_{t=1}^{T}\sum_{p\in\mathcal{P}}\ell_{t}(p_{t})-\ell_{t}(p^{*})\right)$
For a sequence of distributions x1, *· · ·* , xT ∈ ∆(P) and loss functions `1, · · · , `T : P → R, we define

$$\text{FullSwapReg}_{T}(\mathbf{x}_{1:T},\ell_{1:T}):=\sup_{\pi:\mathcal{P}\to\mathcal{P}}\sum_{t=1}^{T}\sum_{p\in\mathcal{P}}\mathbf{x}_{t}(p)\cdot(\ell_{t}(p)-\ell_{t}(\pi(p))).\tag{2}$$
$$({\mathfrak{I}})$$

Here, we adopt the convention of [FKO+25], referring to the latter quantity as *Full* Swap Regret
to emphasize that we consider all swap transformations π : *P → P* (instead of e.g. just linear
transformations π). Throughout, we consider the performance of *regret minimizing* algorithms. These algorithms sequentially map loss functions `1, · · · , `T to actions p1, · · · , pT or action distributions x1, *· · ·* , xT with
the goal of minimizing the above quantities. We consider the performance of these algorithms on
adversarially selected loss functions from a set L. Abusing notation slightly, for an external regret minimizing algorithm Alg : L
T → PT, we define
$\text{Alg}_{T}(\text{Alg}):=\sup\limits_{\ell_{1:T}\in\mathcal{L}^{T}}\text{Ext}\text{Reg}_{T}\left(\text{Alg}(\ell_{1:T}),\ell_{1:T}\right)$
ExtRegT(Alg(`1:T ), `1:T ) (3)
and for a full swap regret minimizing algorithm Alg : L
$$\cdot\mathrm{g}:{\mathcal{L}}^{T}\to\Delta({\mathcal{P}})^{T},\,\mathrm{w}$$
T, we define FullSwapRegT(Alg) := sup
`1:T ∈LT
FullSwapRegT(Alg(`1:T ), `1:T ).

We will denote the tth action played by Alg on a sequence of losses `1:T by Algt(`1:T ). One important subclass of external regret minimization problems is the setting of *online linear optimization (OLO)*,
where all loss functions in ` are linear. Here we slightly abuse notation and identify L with a subset of R
d(with the understanding that an element ` ∈ L refers to the linear loss function `(p) = h*p, `*i).

Although we will never actually employ any OLO algorithms themselves, the calibration bounds we obtain will be closely related to optimal regret bounds for instances of OLO (we discuss this further in Section 2.4).

## 2.3 From Swap Regret To Calibration

As noted in [LSS25, FKO+25], calibration with a distance measure D that corresponds to a Bregman divergence can be written as a full swap regret with loss functions given by the associated *proper* scoring rule. Given a convex function R : P → R, the *Bregman divergence* associated to R,
DR : *P × P →* R≥0, is defined as7 DR(y|p) := R(y) − R(p) − h∇R(p), y − pi Geometrically, this divergence is defined by taking the hyperplane tangent to R at p and computing the difference in height between R and the hyperplane at y (see Figure 2). When viewed as a loss function in p, the Bregman divergence DR(y|p) also has the property that it is a *proper scoring rule*. This refers to the fact that if y is drawn from some distribution y ∈ ∆(P), the optimal response p (to minimize the expected loss DR(y|p)) is simply the expectation y¯ = Ey∼y[y].

In particular, we have the following lemma.

Lemma 2.1. For any y ∈ ∆(P) and convex function R : P → R, let y¯ = Ey∼y[y]*. and* R(y) = Ey∼y[R(y)]. For all p ∈ P, Ey∼y[DR(y|p)] = DR(¯y|p) + R(y) − R(¯y). *In particular,* `(p) = Ey∼y[DR(y|p)] *is minimized at* p = ¯y at a value of R(y) − R(¯y) *(Figure 3).*
This implies the following connection between full swap regret and calibration.

Lemma 2.2. Fix any convex function R : P → R*. For any sequence of distributions* x1, x2, . . . , xT ∈ ∆(P) *and outcomes* y1, y2, . . . , yT ∈ P*, define the sequence of loss functions* `1, `2, . . . , `T via `t(p) = DR(yt|p)*. Then,*
FullSwapRegT(x1:T , `1:T ) = CalDR
T(x1:T , y1:T ).

The proofs of Lemmas 2.1 and 2.2 may be found in Appendix B.

$\lambda=c$ . 
$D_{\mathbb{R}^d}(\cdot)$

## 2.4 Rates And Regularization

In order to reduce our general calibration problem to a swap regret minimization problem (via Lemma 2.2), we will need to construct a convex function R whose Bregman divergence upper bounds our distance measure. It turns out that the optimal choice of such a function is closely related to the design of optimal regularizers for online linear optimization. In this section, we describe this functional optimization problem and detail this connection.

We say that a convex function R : P → R is α*-strongly convex* with respect to a given norm k·k if for any points y, p ∈ P it is the case that R(y) ≥ R(p)+h∇R(p), y−pi+α ky − pk 2. Equivalently, the Bregman divergence must satisfy DR(y|p) ≥ α ky − pk 2. Thus, k·k2-calibration error is bounded by DR-calibration error if R is k·k-norm 1-strongly convex.

Our later analysis will need not only R to be strongly convex with respect to our norm, but for the Bregman divergence to have a small maximal value. Motivated by this, we will say that a convex function R : P → R has *rate* ρ with respect to a given norm k·k if: (1) R is 1-strongly convex with respect to k·k, and (2) the range of the Bregman divergence is at most ρ, i.e., maxy,p∈P DR(y|p) ≤ ρ.

We define Rate(P, k·k) to be the infimum of the rates of all 1-strongly convex functions R : P → R.

As mentioned earlier, we call this quantity a "rate" due to its connection with the optimal regret rates for online linear optimization. For a learning algorithm Alg : L
T → PT, we defined (in (3)) ExtRegT(Alg) to be the worst-case regret against any sequence `1:T of T losses.

It is known that for any fixed action set and loss set, the optimal worst-case regret bound is of the form pRateOLO(P,L) · T + o(
√T), for some constant RateOLO(P,L). Formally, we define RateOLO(P,L) = lim supT→∞ infAlg 1 T
· ExtRegT
(Alg)
2.

One important class of learning algorithms for online linear optimization is the class of Follow- The-Regularized-Leader (FTRL) algorithms. Each algorithm in this class is specified by a convex
"regularizer" function R : P → R, and at round t selects the action pt = argminp∈P Pt−1 s=1 h*p, `*ti +
R(p). The work of [SST11] and [GSJ24] shows that there always exists some instantiation of FTRL
which achieves (up to a universal constant factor) the optimal regret rate of pRateOLO(P,L) · T +
o(
√T) defined above. Moreover, the optimal regularizer for this instance can be constructed by solving a similar functional optimization problem over strongly convex regularizers R, as described in the following theorem.

Theorem 2.3. Let P and L be centrally symmetric convex sets. Then, if the function R : P → R *is 1-* strongly-convex with respect to the norm k·kL∗ and has range ρ *(i.e.,* maxp∈P R(p)−minp∈P R(p) =
ρ*), then* RateOLO(P,L) ≤ ρ. Conversely, there exists a function R : P → R that is 1-strongly-convex with respect to k·kL∗ *and has range* O(RateOLO(P,L)).

Proof. The first result (that RateOLO(P,L) ≤ ρ) follows from the standard analysis of FTRL - see e.g. Theorem 5.2 in [H
+16]. The converse result follows from Theorem 2 of [GSJ24].

Theorem 2.3 allows us to relate the quantity Rate(P, k·k) to the quantity RateOLO(P,L) (where L
is chosen to be the unit dual norm ball). Note that there is a slight difference in the two functional optimization problems defined above - the one for Rate(P, k·k) asks us to bound the range of the Bregman divergence of R, while the one for RateOLO(P,L) asks us to bound the range of R
itself. While these two quantities do not directly bound each other (the negative entropy function R(p) = Ppilog pi has bounded range over the simplex but unbounded Bregman divergence), we can nonetheless show that optimal solutions to one problem can be used to construct optimal solutions to the other problem of similar quality.

Lemma 2.4. If the action set P *is centrally symmetric and* L = {y ∈ R
d| kyk∗ ≤ 1} (i.e., the unit ball in the dual norm to k·k*), then* RateOLO(P,L) = Θ(Rate(P, k·k)).

## 3 Main Result

We now describe our main algorithm for calibration, TreeCal (Algorithm 1). As we will see, it is equivalent to the TreeSwap algorithm for Full Swap Regret minimization ([DDFG24, PR24];
Algorithm 2), where the loss functions are given by appropriate Bregman divergences as determined by Lemma 2.2. Moreover, TreeCal is effectively the same as the main algorithm of [Pen25]. However, the perspective that TreeCal can be viewed as a particular instance of TreeSwap (Lemma 3.2) is novel to this work, and it enables us to tackle a much more general set of calibration problems (Theorem 3.1). We first describe the TreeCal and TreeSwap algorithms, then state Theorem 3.1 which establishes our main upper bound for TreeCal, and finally discuss the proof of Theorem 3.1, which uses the TreeSwap algorithm as a tool in the analysis.

## 3.1 Algorithm Description

Given some number of rounds T ∈ N, TreeCal and TreeSwap sequentially produce distributions x1, *· · ·* , xT ∈ ∆(P). TreeCal receives from the adversary an outcome sequence y1, · · · , yT ∈ P
whereas TreeSwap receives loss functions `1, · · · , `T : P → R.

To describe how the algorithms use the adversary's actions to produce the distributions xt, we need some additional ntation. The algorithms take as input parameters *H, L* ∈ N satisfying H ≥ 2 and HL−1 ≤ T ≤ HL. We index time steps t ∈ [T] via base-H L-tuples: in particular, for t ∈ [T], we let t1*, . . . , t*L ∈ [0 : H − 1] be the base-H representation of t − 1; we will write t − 1 = (t1t2 *· · ·*tL). For all 0 ≤ l ≤ L, for all k ∈ [0 : H − 1]l, let Γ
(l)
k ⊂ [T] represent the interval of times t with prefix k. That is, t ∈ Γ
(l)
kiff ti = ki for all i ∈ [1 : l]. These intervals may be arranged to form an H-ary depth-L tree, where the children of Γ
(l)
kare Γ
(l+1)
k0, Γ
(l+1)
k1, *· · ·* , Γ
(l+1)
k,H−1.

8 Both TreeCal and TreeSwap operate by assigning an action p
(l)
kto each node Γ
(l)
kof the tree, except the root. At time t, both algorithms return the uniform distribution over the actions on the root-to-leaf-t path, namely xt := Unif np
(1)
t1
, p
(2)
t1t2
, · · · , p
(L)
t1t2···tL
o (see Figure 1). The algorithms

| differ in how the actions p (l) k are chosen:                                                                                                                                               | t   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| (3) (3) (3) (3) (3) (3) (3) (3) (3) (3) (3) (3) (3) (3) p 000p 001p 002p 010p 011p 012p 020p 021p 022p 100p 101p 102p 110p 111 p (1) 0 p (1) 1 (2) (2) (2) (2) (2) p 00 p 01 p 02 p 10 p 11 |     |

Figure 1: Visualization of the state of TreeCal/TreeSwap at time step t (about half-way through the algorithm).

For H = 3, we depict the intervals Γ of the first three non-root levels of the tree (l = 1, 2, 3). Each rectangular node represents an interval, with sibling nodes separated by red lines. We represent the specific time step t via the vertical dashed green line. The yellow intervals it intersects at each level correspond to the nodes on the root-to-leaf-t path. Accordingly, xt will be the uniform distribution over the labels p of these yellow intervals. We see that the algorithm has committed to the labels of all intervals that started at or before time t, and has yet to label the future intervals.

- TreeCal (Algorithm 1) assigns actions to nodes as follows. For all 1 ≤ l ≤ L, k ∈ [0 :
H − 1]l−1, h ∈ [0 : H − 1], at the start of Γ
(l)
kh, TreeCal sets p
(l)
kh to be the average over all yt that have been observed thus far in the parent interval Γ
(l−1)
k. That is,

$$p_{k h}^{(l)}=\frac{1}{h H^{L-l}}\sum_{i=0}^{h-1}\sum_{t\in\Gamma_{k i}^{(l)}}y_{t}$$
$$(4)$$
yt (4)
- The more general TreeSwap algorithm (Algorithm 2) also takes as a parameter an external regret-minimizing algorithm Alg, which operates with horizon of length H: we denote the resulting algorithm by TreeSwap.Alg. TreeSwap.Alg associates each internal node of the tree, Γ
(l−1)
k(with 1 ≤ l ≤ L), with an instance Alg, denoted Alg(l−1)
k. The subroutine Alg(l−1)
kis responsible for choosing the actions p
(l)
k0
, p
(l)
k1
, · · · , p
(l)
k(H−1). It does so by responding to the average losses over each of its child intervals. In particular: at the end of each child interval Γ
(l)
kh, we pass Alg(l−1)
kthe average loss over that interval. Alg(l−1)
kthen outputs the action p
(l) k(h+1) 
assigned to the next child interval.

## 3.2 Main Result

Theorem 3.1 upper bounds the calibration error of TreeCal with respect to the squared norm k·k2.

Theorem 3.1 (Main theorem). Let P ⊂ R
d be a bounded convex set and k·k be an arbitrary norm.

Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes y1, . . . , yT ∈ P,
the k·k2calibration error of its predictions x1, . . . , xT ∈ ∆(P) *is bounded as follows:*
Calk·k2 T(x1:T , y1:T ) ≤ T for T ≥ (diam(P)/
√)
O(Rate(P,k·k)/)
It is straightforward to derive from Theorem 3.1 via an application of Jensen's inequality an upper bound on the calibration error of TreeCal with respect to the (non-squared) norm k·k, as stated in Theorem 1.1; see Corollary C.5. In Appendix E, we additionally consider a variant of TreeCal which plays *pure actions* in P (i.e., not distributions) by sampling from the distributions xt for each t ∈ [T]. We show that the *pure calibration* error of this variant can be bounded by a similar quantity to that in Theorem 3.1.

## 3.3 Outline Of The Proof Of Theorem 3.1

Step 1: Reduction from calibration error to swap regret. Let us choose a convex function R : P → R given P, k·k as described in Section 2.4. The first step in the proof of Theorem 3.1 is to reduce the problem of minimizing (squared-norm) calibration error to that of minimizing full swap regret for an appropriate sequence of loss functions. In particular, for any sequence x1*, . . . ,* xT ∈ ∆(P) and y1, . . . , yT ∈ P, we have Calk·k2 T(x1:T , y1:T ) ≤ CalDR
T(x1:T , y1:T ) = FullSwapRegR(x1:T , `1:T ), (5)
where `t : P → R is the loss function given by `t(p*) :=* DR(yt|p): the inequality uses strong convexity of R, and the subsequent equality uses Lemma 2.2. Step 2: Equivalence with TreeSwap. Thus, it suffices to find an algorithm which minimizies the full swap regret quantity on the right-hand side of (5). Fortunately, the TreeSwap algorithm is known to do exactly this! (See Theorem C.1, from [DDFG24], for a formal statement for the swap regret bound of TreeSwap.) In order to apply the swap regret bound of Theorem C.1, we need to ensure that the TreeCal algorithm is an instantiation of TreeSwap.Alg for an appropriate choice of (a) the loss functions fed as input to TreeSwap and (b) the Alg subroutine. The loss functions have already been defined: given a sequence y1*, . . . , y*T , recall that we chose `t(p) := DR(yt|p). Moreover, we let the Alg subroutine be given by *Follow-the-Leader* (FTL), which simply chooses an action at each step minimizing the sum of losses up to the previous time step. The following lemma shows that TreeSwap with the losses `t and the FTL subroutine produces the same action distributions as TreeCal:
Lemma 3.2. Let P ⊂ R
d be a bounded convex set and let R : P → R be a convex function. For a sequence of loss functions `1, · · · , `H : P → R*, define* FTLh(`1:H) = arg minp∈P Ph−1 s=1 
`s(p).

For all sequences of outcomes y1:T ∈ PT, the action distributions xt *produced by* TreeCal on y1:T
equal those produced by TreeSwap.FTL *on loss functions* `t(p) = DR(yt|p) *for all* t.

The proof of Lemma 3.2 (given in full in the appendix) is a straightforward consequence of the fact that the Bregman divergence is a proper scoring rule: the action p ∈ P minimizing an average of Bregman divergences DR(y|p) is simply the average of the constituent points y (Lemma 2.1). Step 3: Applying the swap regret bound of TreeSwap to BTL. Finally, we want to apply the main result of [DDFG24] (restated as Theorem C.1) to bound the full swap regret for the iterates x1:T
produced by TreeSwap.Alg, for an appropriate choice of Alg. The most natural way to do so would be to try to directly apply this result in the case when Alg = FTL (which corresponds to how we actually implement TreeSwap). However, applying this theorem requires an external regret bound on FTL for an arbitrary sequence of losses. While FTL is known to possess strong external regret bounds in some situations (e.g., when all the loss functions are strongly convex), the loss functions p 7→ DR(y|p) are not necessarily even convex in p and so it is not a priori clear how to establish such bounds.

Instead, the main idea is to consider the "Be-The-Leader" algorithm BTL, which is the same as FTL but where actions are shifted ahead in time by 1 time step: in particular, the action chosen by BTL at time step h given a sequence `1, `2, . . . , `H : P → R is BTLh(`1:H) = FTLh+1(`1:H) =
argminp∈P Ph s=1 `s(p). BTL is not implementable since its action at time step h depends on the (unobserved) loss `h at that time step. However, since its regret is always non-positive (i.e.,
ExtRegH(BTL) ≤ 0 for any H), if we apply Theorem C.1 to the algorithm TreeSwap.BTL, we get that FullSwapRegT(TreeSwap.BTL) ≤  · T as long as T ≥ HO(ρ/)for any choice of H (the arity parameter H used in TreeSwap). Using (5), this implies that the *calibration error* of the iterates produced by TreeSwap.BTL can also be bounded above by  · T.

Of course, this result on its own is uninteresting (since BTL is unimplementable, as mentioned above). However, the key insight is that we can show that the actions chosen by TreeSwap.BTL are close to (as measured by the norm k·k) those chosen by TreeSwap.FTL, which in turn is equivalent to TreeCal (Lemma 3.2). This closeness is an immediate consequence of the fact that the actions chosen by FTL for our loss functions DR(y1|·), DR(y2|·)*, . . .* are simply the empirical average of all actions y1, y2, . . . ∈ P of the adversary up to the previous time step.9In turn, we can use this closeness to show that the calibration error of TreeSwap.FTL is close to that of TreeSwap.BTL. This latter part of the argument becomes slightly tricky due to the possibility that different nodes of the tree might output the same action p ∈ P; accordingly, we need to work with a *labeled* variant of the action set and bound the swap regret over this labeled variant; see Appendix C for further details.

## 4 Lower Bound

To prove our calibration lower bound, we make use of the following swap regret lower bound.

Theorem 4.1 (Theorem 4.1 of [DFG+24]). There is a sufficiently small constant c4.1 > 0 *so that the* following holds. Fix any  > 0. For any d ∈ N, there is a subset X ⊂ [−1, 1]d*so that the following* holds for any T ≤ exp c4.1 min{d 1/14, −1/6}. There is an oblivious adversary producing a sequence v1, . . . , vT with kvtk1 ≤ 1 and kvtk∞ ≤ max{d
−13/14, 13/6} for all t, which satisfies the following property. For linear loss functions `(x, v) = hv, xi *for vectors* v ∈ R
d and x ∈ R
d, any learning algorithm producing x1*, . . . ,* xT ∈ ∆(X ),

$$\operatorname{FullSwapReg}_{T}(\mathbf{x}_{1:T},\ell(\cdot,v_{1:T}))=\operatorname*{sup}_{\pi:X\to X}\sum_{t=1}^{T}\sum_{p\in X}\mathbf{x}_{t}(p)\cdot(\langle v_{t},p\rangle-\langle v_{t},\pi(p)\rangle)\geq\epsilon\cdot T.$$

We leverage the classic reduction from swap-regret minimization to calibration [FV98]: by producing calibrated predictions of the upcoming loss and best-responding to it, we can effectively minimize swap regret. This is formalized in the following lemma, proved in Appendix D.

Lemma 4.2. Fix a set P ⊂ R
d, a norm k · k, and write D(*p, p*0) := kp − p 0k*. Suppose that, for some*
 > 0, T ∈ N, there is an algorithm which chooses x1, . . . , xT ∈ ∆(P) *and which ensures that for* every oblivious adversary choosing y1, . . . , yT ∈ P*, we have* CalD
T(x1:T , y1:T ) ≤  · T. Then for every set P
0 ⊂ R
d*, there is an algorithm which chooses* x 01*, . . . ,* x 0 T 
∈ ∆(P
0) and which ensures that for every oblivious adversary choosing y1, . . . , yT ∈ P*, we have* FullSwapRegT
(x 0 1:T
, `(·, y1:T )) ≤  · T · diamk·k?

(P
0).

Theorem 4.3. There is a sufficiently small constant c > 0 *so that the following holds. Write* D(*p, p*0) = kp−p 0k1, and fix any  > 0, d ∈ N*. Then for any* T ≤ exp(c · min{d 1/14, −1/6}), there is an oblivious adversary producing a sequence y1, . . . , yT ∈ ∆d*so that for any learning algorithm* producing x1*, . . . ,* xT ∈ ∆(∆d), CalD
T
(x1:T , y1:T ) ≥  · T.

In Theorem D.2 (see Appendix D.2), we show a similar lower bound for `2 calibration over the unit `2 ball.

| References [ACRS25] Eshwar Ram Arunachaleswaran, Natalie Collina, Aaron Roth, and Mirah Shi.                                                                             | An                                                                                                                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| elementary predictor obtaining distance to calibration. In Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1366–1370. SIAM, 2025. |                                                                                                                                                                                                                                                                                                                     |
| [AM11]                                                                                                                                                                   | Jacob Abernethy and Shie Mannor. Does an efficient calibrated forecasting strategy exist? In Proceedings of the 24th Annual Conference on Learning Theory, pages 809–812. JMLR Workshop and Conference Proceedings, 2011.                                                                                           |
| [BGHN23]                                                                                                                                                                 | Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, and Preetum Nakkiran. A unifying theory of distance from calibration. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 1727–1740, 2023.                                                                                                |
| [BM07]                                                                                                                                                                   | Avrim Blum and Yishay Mansour. From external to internal regret. Journal of Machine Learning Research, 8(6), 2007.                                                                                                                                                                                                  |
| [Daw82]                                                                                                                                                                  | A Philip Dawid. The well-calibrated bayesian. Journal of the American statistical Association, 77(379):605–610, 1982.                                                                                                                                                                                               |
| [DDF+24]                                                                                                                                                                 | Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, Noah Golowich, Robert Kleinberg, and Princewill Okoroafor. Breaking the t 2/3 barrier for sequential calibration. arXiv preprint arXiv:2406.13668, 2024.                                                                                                   |
| [DDFG24]                                                                                                                                                                 | Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. From external to swap regret 2.0: An efficient reduction for large action spaces. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 1216–1222, 2024.                                                        |
| [DFG+24]                                                                                                                                                                 | Constantinos Daskalakis, Gabriele Farina, Noah Golowich, Tuomas Sandholm, and Brian Hu Zhang. A lower bound on swap regret in extensive-form games. arXiv preprint arXiv:2406.13116, 2024.                                                                                                                          |
| [FH18]                                                                                                                                                                   | Dean P Foster and Sergiu Hart. Smooth calibration, leaky forecasts, finite recall, and nash dynamics. Games and Economic Behavior, 109:271–293, 2018.                                                                                                                                                               |
| [FKO+25]                                                                                                                                                                 | Maxwell Fishelson, Robert Kleinberg, Princewill Okoroafor, Renato Paes Leme, Jon Schneider, and Yifeng Teng. Full swap regret and discretized calibration. arXiv preprint arXiv:2502.09332, 2025.                                                                                                                   |
| [FL99]                                                                                                                                                                   | Drew Fudenberg and David K Levine. An easier way to calibrate. Games and economic behavior, 29(1-2):131–137, 1999.                                                                                                                                                                                                  |
| [Fos99]                                                                                                                                                                  | Dean P Foster. A proof of calibration via blackwell's approachability theorem. Games and Economic Behavior, 29(1-2):73–78, 1999.                                                                                                                                                                                    |
| [FV97]                                                                                                                                                                   | Dean P Foster and Rakesh V Vohra. Calibrated learning and correlated equilibrium. Games and Economic Behavior, 21(1-2):40–55, 1997.                                                                                                                                                                                 |
| [FV98]                                                                                                                                                                   | Dean P Foster and Rakesh V Vohra. Asymptotic calibration. Biometrika, 85(2):379–390, 1998.                                                                                                                                                                                                                          |
| [GJRR24]                                                                                                                                                                 | Sumegha Garg, Christopher Jung, Omer Reingold, and Aaron Roth. Oracle efficient online multicalibration and omniprediction. In Proceedings of the 2024 Annual ACMSIAM Symposium on Discrete Algorithms (SODA), pages 2725–2792. SIAM, 2024.                                                                                                                                                                                                                                                                                                                     |
| [GPSW17]                                                                                                                                                                 | Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 06–11 Aug 2017. |

| [GSJ24]   | Khashayar Gatmiry, Jon Schneider, and Stefanie Jegelka. Computing optimal regularizers for online linear optimization. arXiv preprint arXiv:2410.17336, 2024.                                                                                                                              |
|-----------|------------------------------------------------------------------------------------------------------------------------------|
| [H+16]    | Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4):157–325, 2016. |
| [Har22]   | Sergiu Hart. Calibrated forecasts: The minimax proof. arXiv preprint arXiv:2209.05863, 2022.                                 |
| [HJKRR18] Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Multicalibration: Calibration for the (Computationally-identifiable) masses. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1939–1948. PMLR, 10–15 Jul 2018. [HK12] Elad Hazan and Sham M Kakade. (weak) calibration is computationally hard. In Conference on Learning Theory, pages 3–1. JMLR Workshop and Conference Proceedings, 2012. [HW24] Lunjia Hu and Yifan Wu. Predict to minimize swap regret for all payoff-bounded tasks. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS), pages 244–263. IEEE, 2024. [KF08] Sham M Kakade and Dean P Foster. Deterministic calibration and nash equilibrium. Journal of Computer and System Sciences, 74(1):115–130, 2008. [KLST23] Bobby Kleinberg, Renato Paes Leme, Jon Schneider, and Yifeng Teng. U-calibration: Forecasting for an unknown agent. In The Thirty Sixth Annual Conference on Learning Theory, pages 5143–5145. PMLR, 2023. [LSS24] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Optimal multiclass u-calibration error and beyond. arXiv preprint arXiv:2405.19374, 2024. [LSS25] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Simultaneous swap regret minimization via kl-calibration. arXiv preprint arXiv:2502.16387, 2025. [MS10] Shie Mannor and Gilles Stoltz. A geometric proof of calibration. Mathematics of Operations Research, 35(4):721–727, 2010. [MSA07] Shie Mannor, Jeff S Shamma, and Gürdal Arslan. Online calibrated forecasts: Memory efficiency versus universality for learning in games. Machine Learning, 67:77–115, 2007. [NRRX23] Georgy Noarov, Ramya Ramalingam, Aaron Roth, and Stephan Xie. High-dimensional prediction for sequential decision making. arXiv preprint arXiv:2310.17651, 2023. [Pen25] Binghui Peng. High dimensional online calibration in polynomial time. arXiv preprint arXiv:2504.09096, 2025. [PR24] Binghui Peng and Aviad Rubinstein. Fast swap regret minimization and applications to approximate correlated equilibria. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 1223–1234, 2024. [QV21] Mingda Qiao and Gregory Valiant. Stronger calibration lower bounds via sidestepping. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, pages 456–466, 2021. [QZ24] Mingda Qiao and Letian Zheng. On the distance from calibration in sequential prediction. In The Thirty Seventh Annual Conference on Learning Theory, pages 4307–4357. PMLR, 2024. [RS24] Aaron Roth and Mirah Shi. Forecasting for swap regret for all downstream agents. In Proceedings of the 25th ACM Conference on Economics and Computation, pages 466–488, 2024. [RST15] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Sequential complexities and uniform martingale laws of large numbers. Probability Theory and Related Fields, 161(1-2):111–153, 2015. [SS11] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning, 4(2):107–194, 2011.           |                                                                                                                              |

[SST11] Nati Srebro, Karthik Sridharan, and Ambuj Tewari. On the universality of online mirror descent. *Advances in neural information processing systems*, 24, 2011.

[ZME20] Shengjia Zhao, Tengyu Ma, and Stefano Ermon. Individual calibration with randomized forecasting. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 11387–11397. PMLR, 13–18 Jul 2020.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We prove all stated claims. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss limitations. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We prove all theorems and lemmas. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]
Justification: The paper does not include experiments. Guidelines:
- The answer NA means that the paper does not include experiments.

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments requiring code. Guidelines:
- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA] Justification: The paper does not include experiments. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper does not include experiments. Guidelines:
- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [NA] Justification: The paper does not include experiments. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes]
Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] Justification: There is no societal impact of the work performed. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper poses no such risks.

Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: The paper does not use existing assets. Guidelines:
- The answer NA means that the paper does not use existing assets.

- The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets. Guidelines:
- The answer NA means that the paper does not release new assets. - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional review board (IRB) approvals or equivalent for research with human**

## Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

16. **Declaration of LLM usage**
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines:
- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.

## A Additional Related Work

There is a large range of other existing work on online (sequential) calibration [Daw82, FV97, FV98, QV21, DDF+24, Har22, Fos99, FL99, KF08, MSA07, MS10, AM11, HK12, FH18, LSS24, NRRX23, KLST23, GJRR24, QZ24, ACRS25]. We briefly survey some of these areas below.

Binary outcomes. For binary outcomes (i.e., one-dimensional calibration), classical results of [FV97, Fos99, BM07, AM11] demonstrate that it is possible to efficiently guarantee O(T
2/3) `1calibration. The optimal possible rates for `1-calibration remain a major unsolved problem in online learning. Recently [QV21] improved over the naive lower bound of Ω(√T) by demonstrating a lower bound of Ω(T
0.528); this was further improved to Ω(T
0.543) by [DDF+24], who also improved on the upper bound, demonstrating the existence of an algorithm with O(T
2/3−) calibration for some constant  > 0.

Calibration and swap regret. The connection between calibration and swap regret has been acknowledged since the earliest works on swap regret. For example, the earliest algorithms for minimizing swap regret worked by best responding to online calibrated predictions [FV97] (later algorithms for swap regret minimization, such as [BM07] and [DDF+24] obtain better swap regret bounds by side-stepping the need to generate calibrated predictions). In the other direction, several works minimize calibration via relating it to a swap regret that can then be minimized [FKO+25, LSS25, AM11, Fos99]. Other forms of calibration. Due to the difficulty of minimizing (high-dimensional) calibration, there has been a line of work on designing forecasting algorithms that minimize weaker forms of calibration that recover some of the important guarantees of calibration (e.g., trustworthy-ness by a decision-maker). These include *distance from calibration* [BGHN23, QZ24, ACRS25], omniprediction error / U-calibration [KLST23, LSS24, GJRR24], *calibration conditioned on downstream* outcomes [NRRX23], and *prediction for downstream swap regret* [RS24, HW24]. Other work focuses on minimizing notions of calibration designed to lead to specific classes of equilibria, e.g. weak calibration [HK12], deterministic calibration [KF08], and smooth calibration [FH18].

p, R(p) *y, R(y*)
y,h∇R(p), y − pi + R(p)
DR(y | p)
Proof of Lemma *2.1.*

$$\mathbb{E}_{y\sim\mathbf{y}}[D_{R}(y|p)]=\mathbb{E}_{y\sim\mathbf{y}}\left[R(y)-R(p)-\langle\nabla R(p),y-p\rangle\right]$$ $$=\overline{R(y)}-R(p)-\langle\nabla R(p),\bar{y}-p\rangle$$ $$=D_{R}(\bar{y}|p)+\overline{R(y)}-R(\bar{y})$$

See Figure 3 for a visual proof.