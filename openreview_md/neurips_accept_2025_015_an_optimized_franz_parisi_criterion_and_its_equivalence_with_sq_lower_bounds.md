# An Optimized Franz-Parisi Criterion And Its Equivalence With Sq Lower Bounds

Siyu Chen Department of Statistics and Data Science Yale University siyu.chen.sc3226@yale.edu Theodor Misiakiewicz Department of Statistics and Data Science Yale University theodor.misiakiewicz@yale.edu

| Ilias Zadik                                                                    |
|--------------------------------------------------------------------------------|
| Department of Statistics and Data Science Yale University ilias.zadik@yale.edu |

## Abstract

Bandeira et al. (2022) introduced the Franz-Parisi (FP) criterion for characterizing the computational hard phases in statistical detection problems. The FP criterion, based on an annealed version of the celebrated Franz-Parisi potential from statistical physics, was shown to be equivalent to low-degree polynomial (LDP) lower bounds for Gaussian additive models, thereby connecting two distinct approaches to understanding the computational hardness in statistical inference. In this paper, we propose a refined FP criterion that aims to better capture the geometric "overlap" structure of statistical models. Our main result establishes that this optimized FP criterion is equivalent to Statistical Query (SQ) lower bounds—another foundational framework in computational complexity of statistical inference. Crucially, this equivalence holds under a mild, verifiable assumption satisfied by a broad class of statistical models, including Gaussian additive models, planted sparse models, as well as non-Gaussian component analysis (NGCA), single-index (SI) models, and convex truncation detection settings. For instance, in the case of convex truncation tasks, the assumption is equivalent with the Gaussian correlation inequality (Royen, 2014) from convex geometry. In addition to the above, our equivalence not only unifies and simplifies the derivation of several known SQ lower bounds—such as for the NGCA model (Diakonikolas et al., 2017) and the SI model (Damian et al., 2024)—but also yields new SQ lower bounds of independent interest, including for the computational gaps in mixed sparse linear regression (Arpino et al., 2023) and convex truncation (De et al., 2023).

| Peiyuan Zhang                                                                    |
|----------------------------------------------------------------------------------|
| Department of Statistics and Data Science Yale University peiyuan.zhang@yale.edu |

## 1 Introduction

Over the past decades, a central focus in statistical inference has been to understand the transition from computationally easy to hard regimes—that is, to characterize when a statistical task can be solved by polynomial-time algorithms. A key insight from this line of work is the emergence of computational-statistical tradeoffs: in many models, there exist broad parameter regimes where information-theoretic recovery is possible, yet no known polynomial-time algorithm succeeds. Evidence for such tradeoffs spans multiple disciplines with varying levels of mathematical rigor. In particular, the statistical physics community has played an instrumental role by leveraging nonrigorous but highly predictive techniques to study average-case hardness. Their approach typically analyzes the geometry of solution spaces and identifies structural properties that correlate with algorithmic intractability (see [40] for a survey). Remarkably, for many statistical models, the predictions from statistical physics have been in striking agreement with the performance of the best-known polynomial-time algorithms. Alongside these heuristic predictions, rigorous frameworks from statistics and theoretical computer science have been developed to analyze the limitations of efficient algorithms. While ruling out all polynomial-time algorithms would require resolving P ̸= N P, substantial progress has been made by studying broad, expressive classes of polynomial-time algorithms. Two frameworks have emerged as particularly influential: *low-degree* (LD) *polynomial* lower bounds [31] and *statistical query* (SQ) lower bounds [20]. For many "nice enough" detection problems, the lower bounds derived from these frameworks align closely with the performance of the best-known polynomial-time algorithms1. This striking consistency has motivated the formulation of the so-called *low-degree conjecture* by Hopkins [26], which posits that for sufficiently "symmetric and noisy" models, the failure of degree-O(log n) polynomials is indicative of the failure of all polynomial-time algorithms [31]. Given this context, a natural question arises: can one formally connect these two seemingly distinct approaches? At first glance, the answer appears negative, due to a fundamental mismatch in scope. Statistical physics techniques are primarily geared toward estimation problems, where the goal is to recover a hidden signal, while the rigorous frameworks discussed above—such as LD and SQ lower bounds—are focused on detection or hypothesis testing, where the task is to distinguish between the presence or absence of a signal in a noisy environment2. Nevertheless, a major step towards bridging this gap was taken by Bandeira et al. (2022) [6], who introduced the *Franz-Parisi* (FP) *criterion* for computational hardness in detection tasks. Inspired by the seminal work of Franz and Parisi in spin glass theory [21], the FP criterion provides a geometric perspective on computational hardness rooted in overlap structures. Crucially, Bandeira et al. showed that for Gaussian additive models, the FP criterion is mathematically equivalent to the low-degree (LD) lower bounds, thereby establishing a rigorous link between statistical physics heuristics and formal algorithmic barriers.

Specifically, consider the following general detection problem between two distributions P and Q supported on a subset of R
n, which in what follows we refer to as a "P versus Q**" task**. Under the *planted* distribution P = EuPu, a signal u is drawn from a prior distribution π supported on Θ ⊆ SN−1, and one observes m independent samples Y1*, . . . , Y*m ∼ Pu. Under the *null* distribution Q, the samples are drawn independently from Y1*, . . . , Y*m ∼ Q. The goal in the detection task3is to distinguish between these two hypotheses based on the observed data, that is to find a test statistics with vanishing Type I and Type II errors, as n grows. Note that the computational question then is whether such a successful test statistic exists that also terminates in polynomial-in-mn time. To characterize the hardness of detection problems from the statistical physics perspective, Bandeira et al. in [6] introduced the following notion of Franz-Parisi (FP) hardness:
Definition 1 (FP hardness). For D, m ∈ N, ε > 0, we say that a P versus Q *detection task is*
(q, m, ε)*-FP hard if*

_FP:_$\mathbb{E}\left[\langle L_{v}^{\otimes m},L_{v}^{\otimes m}\rangle\cdot\mathbf{1}(|\langle u,v\rangle|\leq\delta(q))\right]\leq1+\varepsilon,\quad$where  $$\delta(q)=\sup\{\delta>0:\pi^{2}(|\langle u,v\rangle|\geq\delta)\geq q^{-2}\}.$$
In the definition we denoted as customary Lu =
dPu dQ
, u ∈ Θ, and for f, g ∈ L2(R
n), the Hilbert space L
2(Q) of (square integrable) functions from R
n to R, we use ⟨*f, g*⟩Q = EY ∼Qf(Y )g(Y ).

(1)  (2) $\frac{1}{2}$  (3) $\frac{1}{2}$  (4) $\frac{1}{2}$  (5) $\frac{1}{2}$  (6) $\frac{1}{2}$  (7) $\frac{1}{2}$  (8) $\frac{1}{2}$  (9) $\frac{1}{2}$  (10) $\frac{1}{2}$  (11) $\frac{1}{2}$  (12) $\frac{1}{2}$  (13) $\frac{1}{2}$  (14) $\frac{1}{2}$  (15) $\frac{1}{2}$  (16) $\frac{1}{2}$  (17) $\frac{1}{2}$  (18) $\frac{1}{2}$ 
We elaborate in Section A.1 on the statistical physics motivations behind this criterion, and only briefly highlight its core intuition here. The left-hand side of the FP condition integrates the function Fann(t) := E [⟨L
⊗m u, L⊗m v⟩ · 1(⟨*u, v*⟩ = t)] , over a (1 − q
−2)-typical region of the overlap variable t, corresponding to the constraint |⟨u, v*⟩| ≤* δ(q). This function Fann(t) is an annealed proxy for the Franz-Parisi potential, a central object in statistical physics that has long served as a predictor of algorithmic hardness [40]. Intuitively, the Franz-Parisi potential captures the energy landscape experienced by local algorithms—such as Langevin or Glauber dynamics—whose performance is constrained by the geometry of the underlying signal space. The overlap ⟨*u, v*⟩ naturally quantifies a local "geometric" similarity between signals, making it a meaningful argument for Fann(t) and explaining its role within the FP criterion. Returning to the definition of FP hardness, the parameter m corresponds to the sample size, and one should interpret q as a proxy for the required runtime. In this light, Bandeira et al. (2022) proved that, for Gaussian additive models, FP-hardness is equivalent to the failure of degree-D = log q polynomials to solve the detection task with m samples—i.e., roughly the authors of [6] showed that the problem is (*q, m, O*(1))-FP hard if and only if it is "hard" for degree-log q polynomials to solve the detection task4. Hence, based on the current belief in the literature of low-degree lower bounds that a D-degree lower bound implies that the detection task requires at least e D runtime to be solved, e.g., see [18], proving a task is ((mn)
ω(1)*, m, O*(1))-FP hard for a Gaussian additive model provides rigorous evidence for polynomial-time hardness for the task. Despite this success, the connection between the FP potential and other rigorous notions of algorithmic hardness remains limited. [6] only established a formal equivalence for Gaussian additive models and an one-sided implication for planted sparse models between the FP criterion and "low-degree" lower bounds. They further presented counterexamples where the equivalence fails entirely. In this work, our aim is to extend the Franz-Parisi criterion to rigorously characterize hardness beyond Gaussian additive models, and to clarify the scope and limitations of this framework across a broader class of statistical models.

## 1.1 Main Contributions

Our main contribution is to propose a slight modification of the FP-hardness criterion from [6], motivated by the observation that sticking to the Euclidean geometry assumption (and hence the
"overlap" ⟨*u, v*⟩) may fail to capture the "true" hardness of some statistical models. We remark that this is an arguably natural modification, as (1) there are many statistical models for which the Euclidean geometry appears unnatural for navigating their parameter space (see Section 5 for a simple such construction), and (2) even in statistical physics settings, the Franz-Parisi potential is often considered under a more general notion of overlap [22]. Motivated by these considerations, we propose optimizing the "overlap" event inside the FP-hardness definition, subject only to a mild symmetry assumption for technical reasons. This leads to the following new criterion of FP-hardness:
Definition 2 (Generalized Franz Parisi (GFP) hardness under symmetry G). Fix q, m ∈ N*, ε >* 0 and a group G of finite order acting on the parameter space of the signal. We say a "P *versus* Q"
problem is (q, m, ε)-GFPG *hard if*

$$G F P_{G}:\quad\operatorname*{inf}_{\begin{array}{c}{{A:=\pi^{\otimes2}(A)\geq1-a-2}}\\ {{A\in G^{2}\mathrm{~mm}}}\end{array}}\mathbb{E}\Big[\big<L_{u}^{\otimes m},L_{v}^{\otimes m}\big>_{\mathbb{Q}}\mathbf{1}(A)\Big]\leq1+\varepsilon.$$
A is G2*-invariant*

i≤ 1 + ε. (3)
As in the original FP-hardness framework of [6], one should interpret q as a proxy for runtime, and therefore ((mn)
ω(1)*, m, O*(1))-GFP hardness should be providing evidence of polynomial-time hardness with m samples in this framework. We highlight that the assumption on the invariance of the optimizing event under group G is made for technical reasons to enhance the applicability of our hardness criterion. We point the reader to Section 3.1 for further discussion of this assumption. The main result of this work is that the "optimized" notion of GFP-hardness is fundamentally connected with the well-established framework of Statistical Query (SQ) hardness. The SQ framework 4[6] established this equivalence for m = 1, but the argument extends directly to general m.

(3)  $\frac{1}{2}$
was initially proposed by Kearns in [30] to capture the power of noise-tolerant algorithms. The notion of a statistical dimension proposed by [20] allowed for achieving powerful lower bounds against SQ methods, which we refer to from now on as SQ-hardness results. We employ here a slight strengthening of the notion of SQ-hardness from [20], introduced in [8].

Definition 3 (SQ hardness). Fix q, m ∈ N. We say a "P versus Q*" detection problem is* (q, m)-SQ
hard if

$$S Q\colon\quad\operatorname*{sup}_{A:\pi^{2}(A)\geq q^{-2}}\mathbb{E}\left[\left|\left\langle L_{u},L_{v}\right\rangle_{\mathbb{Q}}-1\right|\mid A\right]\leq{\frac{1}{m}}.$$
$$\left(4\right)$$

. (4)
Roughly, a detection problem is (q, m)-SQ hard if any Statistical Query method succeeding at solving it with m samples requires q queries, which should be interpreted as requiring runtime q (see [8, Appendix A] for more details and motivation). Hence, proving a task is ((mn)
ω(1), m)-SQ hard provides evidence for polynomial-time hardness for the task. Our main result is informally described as follows.

Theorem 1. (Informal, GFP and SQ equivalence) Consider any P versus Q detection task which we assume (1) it satisfies a mild assumption with respect to a group G of finite order acting on the parameter space (namely Assumption 1 *below), and (2) it is information-theoretically impossible to* be solved with mIT samples. Then the following holds for any samples size m *and proxy runtime* q = mΩ(1).

$$n\,],0$$

- If the task is (q, m)*-SQ hard, then it is also* (Θ(q), Θ(m), O(1))-GFPG*-hard.*
- If the task is (*q, m, O*(1))-GFPG-hard, then it is also (mΘ(mIT), m1−o(1))*-SQ hard.*
Note that often in statistical tasks of interest mIT = ω(log n) (in fact, more often than not mIT =
poly(n)). Under this condition, Theorem 1 implies that a task is ((mn)
ω(1), m1−o(1), O(1))-GFP
hard if and only if it is ((mn)
ω(1), m1−o(1))-SQ hardness, matching the two criteria for hardness.

On top of that, as we mentioned above and discuss in Section 3.1, the required Assumption 1 on the detection task is rather mild. In fact, it turns out that it is satisfied for several models of recent interest in the community, making a strong case of how the Generalized Franz-Parisi criterion now correctly predicts the hardness phase for them. Importantly, these models include the Gaussian additive models and also greatly extend beyond them, significantly extending the key message from [6] about connecting the physics-based forms of hardness to more rigorous frameworks. We list now some of the tasks that satisfy Assumption 1.

1. *All Gaussian additive models* (GAMs), under any symmetric prior, satisfy Assumption 1 with G = Z2 that flips the sign of the signal. Moreover, in that case the Generalized Franz Parisi criterion is equivalent to the Franz-Parisi criterion, that is the optimizing event A in (3) is of the form {|⟨u, v*⟩| ≤* δ(q)}. Hence, Theorem 1 allows us to extend the result of [6] which proved the equivalence of FP-hardness to Low-degree hardness for GAMs, to also proving FP-hardness equivalent with SQ-hardness for these settings5.

2. *All Planted Sparse Models* satisfy Assumption 1 for the trivial group G = {id}. In particular, using Theorem 1 we can prove that GFP-hardness is equivalent to SQ-hardness for multiple well-studied models such as sparse phase retrieval [5], sparse regression [24, 6], (multisample) sparse PCA [7], and Bernoulli group testing [11]. As a corollary of this connection, we present a straightforward argument to obtain an SQ lower bound for the mixed sparse linear regression problem [5]. We remark that in [6] it has been proven that FP-hardness implies low-degree hardness for all Planted Sparse Models, but no result was presented for the other direction.

3. *All Non-Gaussian component analysis (NGCA) models* and all single-index models (under any symmetric prior) satisfy Assumption 1 with G = Z2, Therefore, via Theorem 1 GFP-
hardness is again equivalent with SQ-hardness for these tasks.

5We remark that such a connection could also be made via the results of [8], since GAMs are noise-robust.

4. *All Gaussian convex truncation models* satisfy Assumption 1 for G = {id}. In particular, interestingly Assumption 1 for these models is exactly equivalent to the celebrated Gaussian correlation inequality for convex bodies in probability theory, which was a multi-decade open problem posed in 1972 in [25] that was finally proven by Royen in 2014 [37]. Leveraging the equivalence between GFP-hardness and SQ-hardness in Theorem 1, we establish an SQ- lower bound for the convex truncation detection task. This allows us to provide, to the best of our knowledge, the first formal evidence that the current state-of-the-art polynomial-time detection method for convex truncation proposed in [15] has optimal sample complexity.

We also complement our results, with a simple example satisfying Assumption 1 where FP-hardness does not coincide with GFP-hardness, which we interpret as a model where the Euclidean geometry is not appropriate. We finally conclude the paper with a discussion. For completeness, we prove in Appendix B the equivalence between GFP-hardness and low-degree (LD) polynomial hardness for noise-robust models. This result follows by combining our GFP–SQ equivalence with the equivalence between SQ-hardness and LD-hardness under noise robustness shown by Brennan et al. [8]. In particular, our GFP-hardness results for the examples presented in this paper immediately imply low-degree lower bounds in all those settings. This substantially extends the equivalence established in [6]. For clarity and readers' convenience, we also include succinct proofs of the SQ-to-LD equivalence, adapted from [8].

## 2 Setting And Definitions

We first recall the definition of a "P versus Q**" task** mentioned in the Introduction. Under the *planted* distribution P = EuPu, a signal u is drawn from a prior distribution π supported on Θ ⊆ SN−1, and one observes m independent samples Y1*, . . . , Y*m ∼ Pu. Under the *null* distribution Q, the samples are drawn independently from Y1*, . . . , Y*m ∼ Q. The goal in the detection task6is the so-called strong detection task to distinguish between these two hypotheses based on the observed data, that is to find a test statistics with vanishing Type I and Type II errors, as n grows. We will also be interested in the weak detection task, which is that the sum of type I and type II errors is at most 1 − ε for some fixed ε > 0 (not depending on n). In other words, strong detection means the test succeeds with high probability, while weak detection means the test has some non-trivial advantage over random guessing. Throughout, we will work in the Hilbert space L
2(Q) of (square integrable) functions R
N → R
with inner product ⟨*f, g*⟩Q := EY ∼Q[f(Y )g(Y )] and corresponding norm ∥f∥Q := ⟨*f, f*⟩
1/2 Q
. We will assume that Pu is absolutely continuous with respect to Q for all u ∈ supp(π), use Lu := dPu dQ
to denote the likelihood ratio, and assume that Lu ∈ L
2(Q) for all u ∈ supp(π). The likelihood ratio between P and Q is denoted by L := dP
dQ = Eu∼µLu. Observe that for m samples, we denote by Lm = Eu∼µLu the m-sample likelihood ratio. Finally, for a function f : R
N → R and integer D ∈ N, we let f
≤D denote the orthogonal (w.r.t. ⟨·, ·⟩Q) projection of f onto the subspace of polynomials of degree at most D. An important identity between the (squared) norm of the likelihood ratio with m samples and the chi-squared divergence χ 2(P
⊗m ∥ Q⊗m) is
∥L∥
2Q
 = ∥Eu∼µLu∥
2 Q = χ 2(P ∥ Q) + 1 ≥ 1 .

This quantity has the following standard implications for *information-theoretic* impossibility of testing, in the asymptotic regime n → ∞. The proofs can be found in e.g. [34, Lemma 2].

- If ∥L∥
2Q
 
= O(1) then strong detection is impossible.

- If ∥L∥
2Q
 
= 1 + o(1) then weak detection is impossible.

## 3 Main Results

In this section, we formally present our equivalence between GFP-hardness and SQ-hardness.

## 3.1 The Assumption

As mentioned in the Introduction, all our results operate under a crucial assumption on the "P versus Q" detection task. The assumption is as follows. Assumption 1. Given any "P versus Q" task, there exists a π-preserving finite group G acting on the parameter space Θ*, i.e., for all* g ∈ G, g(v)
(d)
= v for v ∼ π, such that for any sample size m for any u, v ∈ Θ*, the following "correlation" inequality holds for any* k ∈ N

$$\mathbb{E}_{g,g^{\prime}\sim U n i f(G)}\left(\langle I\rangle\right)$$

Eg,g′∼Unif(G)(⟨Lg(u), Lg
′(v)⟩Q − 1)k ≥ 0. (5)
We first remark that (5) is a natural condition even if G is the trivial group, G = {id}. Indeed in that case (5) asks that for all *u, v* ∈ Θ,

$$(5)$$
$$\mathbb{Q}-1)^{k}\geq0.$$
$$\langle L_{u},L_{v}\rangle_{\mathbb{Q}}\geq1.$$
$$(6)$$
⟨Lu, Lv⟩Q ≥ 1. (6)
Recall that if one averages over all (u, v) ∼ π
⊗2, we have by standard identities

$\mathbb{E}\langle L_{\mathbf{x}},L_{\mathbf{y}}\rangle_{\mathbb{Q}}=\mathbb{E}$
E⟨Lu, Lv⟩Q = EQ∥EuLu∥
2 2 = 1 + χ 2(P, Q) ≥ 1.

Thus, (6) should be understood as a pointwise condition that is guaranteed to hold in expectation over the product measure π
⊗2for any P, Q. While this pointwise condition turns out to be vanilla satisfied
in many models (such as Planted Sparse Models or Convex Truncation settings), a slight modification of it—leading to (5)—applies more broadly. Specifically, this modified condition requires (6) to hold for a pair *u, v* only after performing a "small" averaging over the a group orbit that preserves the prior
π. For instance, if the prior is symmetric around 0 and the group G is Z2, which acts by flipping the
sign of the signal u, then for k = 1, condition (5) reduces to demonstrating that, for all *u, v*,
$$\frac{1}{4}(\mathbb{E}\langle L_{u},L_{v}\rangle_{\mathbb{Q}}+\mathbb{E}\langle L_{-u},L_{v}\rangle_{\mathbb{Q}}+\mathbb{E}\langle L_{u},L_{-v}\rangle_{\mathbb{Q}}+\mathbb{E}\langle L_{-u},L_{-v}\rangle_{\mathbb{Q}})\geq1,$$
4 which is significantly less restrictive than the original pointwise condition (6). This averaging approach allows for much greater generality, making it applicable to various settings, including Gaussian additive models, single-index models, and Non-Gaussian component analysis settings. Remark 3.1. We finally make a trivial remark that will be useful in verifying (5) in our examples in Section 4 with symmetric prior. In all of them by symmetry we have for all u, v ⟨Lu, L−v⟩Q =
⟨L−u, Lv⟩Q and ⟨Lu, Lv⟩Q = ⟨L−u, L−v⟩Q. Using that and the trivial fact that for all *x, y* ∈ R, if x + y ≥ 0 then x k + y k ≥ 0 for all k ∈ N, we conclude that if G is either the trivial group or Z2
(which will be the case in all examples of Section 4) it suffices to check the case k = 1 in (5), and then it automatically holds for all k ∈ N.

Remark 3.2. As mentioned in the previous remark, we highlight that in all our examples in Section 4 of our GFP-SQ equivalence theorem below, we either use G to be the trivial group or Z2. The reason we state our assumption Assumption 1 for a general finite group G is for potential further applications of our work.

## 3.2 The Gfp-Sq Equivalence 3.2.1 Simplifying Gfp-Hardness

We present our equivalence theorem in two steps. First, we identify an approximate optimal "overlap" event A in the definition of GFP-hardness, which simplifies GFP-hardness significantly and makes GFP-hardness easier to establish in applications. Then, we prove the equivalence between this simplified version and SQ-hardness. Given a group G acting on the parameter space of the signal, it turns out that the approximately optimal "overlap" event A takes the form {ρG(*u, v*) ≤ r} for the following notion of "overlap" between u, v,

$$\rho_{G}(u,v)=\operatorname*{max}_{g,g^{\prime}\in G}\{|\langle L_{g(u)},L_{g^{\prime}(v)}\rangle_{\mathbb{Q}}-1|\}.$$

In particular, focusing only on such type of events we define the following version of FP-hardness.

Definition 4 (ρG-FP hardness). Fix q, m ∈ N, ε > 0 and a finite group G acting on the parameter space of the signal. We say a "P versus Q" problem is (q, m, ε)-ρG*-FP hard if*

$$\begin{array}{r l}{\rho_{G}\mathbf{\text{-}}\mathbf{F}\mathbf{P}:}&{{}\mathbb{E}\left[\langle L_{u}^{\otimes m},L_{v}^{\otimes m}\rangle_{\mathbb{Q}}\cdot\mathbf{1}(\rho_{G}(u,v)<r(q))\right]\leq1+\varepsilon,}\\ {r(q)=\operatorname*{sup}\{r:\pi^{2}(\rho_{G}(u,v)\geq r)\geq q^{-2}\},}\end{array}$$
$$\left(7\right)$$
$$({\boldsymbol{8}})$$

We prove that GFPG-hardness is equivalent to ρG-FP hardness under Assumption 1.

Theorem 2. Consider any "P versus Q" task that satisfies Assumption 1 for a group G*. Suppose* m, q ∈ N and ε > 0. Then the following statements hold.

1. If the task is (q, m, ε)-ρG-FP hard, then the task is also (q, m, ε)-GFPG hard. 2. Assume there exists an r > 0 *such that* π 2(ρG(u, v) < r) = 1 − q
−2 and that m is even.

Then, if the task is (q, m, ε)-GFPG hard, then it is (*q, m,* 3 |G|
(1 +ε) +m·χ 2(P, Q))-ρG-FP
hard. In particular, if mχ2(P, Q) = O(1), the task is (q, m, O(1 + ε))-ρG*-FP hard.*
The proof of this theorem can be found in Appendix C.1. Remark 3.3. While the first implication is immediate to grasp, the second implication has some additional conditions we now elaborate upon. First, both the requirements of the existence of r with the desired probability mass and the parity of m are for technical convenience, and both can be easily remove with some tedious work. Second, any potential "blow-up" in the ε-term for ρG-FP hard depends only on |G|, which should be treated as constant, and the term m · χ 2(P, Q), which is an easy to compute quantity (usually n = 1 and χ 2(P, Q) is an one-dimensional integral). Moreover, it is almost always of order O(1) for detection tasks that are conjecturally hard with m samples. Indeed, the mathematical reason behind this is exactly that it is equal to the squared L
2-norm of the projection of the likelihood onto the degree-1 polynomial space, i.e., on the span of linear functions. On top of that, if the detection task is (*q, m*)-SQ hard *for any* q then it holds directly mχ2(P, Q) = O(1) as well. We elaborate more on this in Remark B.1 in Section B.

## 3.2.2 The Equivalence

As we have already proven an equivalence between GFP-hardness and ρG-FP hardness, it suffices to connect the latter with SQ-hardness. This is the topic of the next theorem.

Theorem 3 (SQ and ρG-FP Equivalence). Suppose a "P versus Q" task satisfies Assumption 1 for a group G.

1. If the task is (q, m)-SQ hard for some q, m with q > 2 then, it is also (q
′, m′, e|G|
−1m′/m)-
ρG*-FP hard for any integers* q
′ < q/√2 and m′ ≤ m/2.

2. Suppose the task is (q, m, ε)-ρG-FP hard for some q, m *integers. Assume that there exists* an r = r(q) > 0 *such that* π 2(ρG(u, v) < r) = 1 − q
−2 and m is even. Then, the model is also (q
′, m′)-SQ hard for any even integer t *with* t ≤ log q/ log m *and any integer* q
′ > 0,

_where_  $$m^{\prime}=\frac{m}{(t(1+\varepsilon)^{1/t}+\chi^{2}(\mathbb{P}^{\otimes4t}\parallel\mathbb{Q}^{\otimes4t}))(q^{\prime})^{2/t}}.$$  _In particular, if for some sample size $m_{\rm IT}$, we have_
(a) *(Bounded* χ 2for mIT *samples)*

$\mathfrak{sl}$ ]. 
$$\chi^{2}(\mathbb{P}^{\otimes m_{\mathrm{IT}}}\parallel\mathbb{Q}^{\otimes m_{\mathrm{IT}}})=O(1).$$
then the model is (mδmIT , Θ( m1−O(δ)
mIT(1+ε)
))-SQ hard for any δ > 0.

The proof of this theorem can be found in Appendix C.2. Similar to Theorem 2, the conditions on *r, m* of part 2 in Theorem 3 are for technical convenience and can be easily removed. As we discussed in the Introduction the assumption that there exists some sufficiently growing mIT (e.g., growing super-logarithmically in n) is natural for multiple commonly studied models. We remark that the condition on the information theory threshold mIT to be growing with n is also necessary, by constructing a variant of the planted clique problem which satisfies Assumption 1, it is not SQ-hard and is GFP-hard. Lastly, we also note that our introduced Assumption 1 is also necessary for the equivalence. In Section A, we discuss a counterexample not satisfying Assumption 1 that is GFP-hard, but not SQ-hard. Remark 3.4. We note that while our bounds in the equivalence of Theorem 2 deteriorate when |G| becomes large, a slightly more general equivalence between GFP and SQ, using a variant of ρG, can also be proven for infinite groups G under an "hypercontractivity" assumption on ⟨Lu, Lv⟩Q with respect to the pair (*u, v*). We omit this generalization as in all relevant examples in this work a small group action using either the trivial or 2-cyclic group suffices.

## 4 Examples

In this section, we discuss two popular classes of detection tasks that satisfy Assumption 1 and hence fall under our GFP-SQ equivalence. Further examples are deferred to Appendix D.

## 4.1 Gaussian Additive Models

A P versus Q task is a Gaussian additive model (GAM) if it satisfies:
1. Under the null model, Q = N (0, In). 2. Under the planted model Pu (for u ∈ S
n−1), for some signal-to-noise ratio (SNR) λ > 0 we set Y = λu + Z, for some Z ∼ Q.

GAMs includes multiple well-studied models in the literature, with the predominant examples being (multisample variants) of tensor PCA [36] and sparse PCA [2]. For such models, it can be straightforwardly checked (see [6, Proposition 2.3]) that for all *u, v*,

⟨Lu, Lv⟩Q = e
λ
2⟨u,v⟩.
So for instance, in the case of non-negative sparse PCA where *u, v* are binary k-sparse vectors in (see e.g., [4, 10]) we always have ⟨u, v⟩ ≥ 0, and therefore Assumption 1 is always satisfied for the trivial group G. On top of that, Assumption 1 remains true for any prior which is symmetric around 0; this
time Assumption 1 is also always satisfied by choosing the action of G = Z2 which flips the sign
of u. We remark that symmetric priors encompass most commonly used priors for GAMs, e.g., for tensor PCA where u = vec(x
⊗r), x ∼ Unif(S
d−1).
Lemma 1. Consider any GAM with symmetric π, i.e., v = −v, v ∼ π. For any *u, v* ∈ support(π),
$$\frac{1}{4}(\langle L_{u},L_{v}\rangle_{\mathbb{Q}}+\langle L_{-u},L_{v}\rangle_{\mathbb{Q}}+\langle L_{u},L_{-v}\rangle_{\mathbb{Q}}+\langle L_{-u},L_{-v}\rangle_{\mathbb{Q}})\geq1.$$
 
Moreover, any GAM satisfies Assumption 1 for G = Z2 *acting by flipping the sign of* u.

Proof. Notice
$$\frac{1}{4}(\langle L_{u},L_{v}\rangle_{\mathbb{Q}}+\langle L_{-u},L_{v}\rangle_{\mathbb{Q}}+\langle L_{u},L_{-v}\rangle_{\mathbb{Q}}+\langle L_{-u},L_{-v}\rangle_{\mathbb{Q}})=\frac{1}{2}(\exp\left(\lambda^{2}\langle u,v\rangle\right)+\exp\left(-\lambda^{2}\langle u,v\rangle\right))\geq1.$$  Hence, given Remark 3.1, the conclusion follows.  
Given the above lemma, we conclude the (almost) equivalence between GFP-hardness and SQ- hardness from Theorem 3.

Remark 4.1. We remark that in the symmetric prior case for a GAM and G = Z2 acting by flipping the sign of u, ρG(*u, v*) = exp λ 2|⟨*u, v*⟩|is an increasing function of |⟨*u, v*⟩|. Hence, for such GAMs we conclude via Theorem 2 that FP-hardness is equivalent to GFP-hardness, and therefore also to SQ-hardness. This is in agreement with the results of [6] establishing that FP-hardness is equivalent to LD-hardness; in fact, our approach can offer an alternative proof of their result via the LD-SQ equivalence [8] and the noise robustness of GAMs (see Theorem Theorem 6).

## 4.2 Planted Sparse Models

In [6], the authors introduced the family of planted sparse models (PSM) and proved that FP-hardness for a PSM implies it's also low-degree hard. We start with the definition.

A P versus Q task is a planted sparse model (PSM) if it satisfies:
1. Under the null model, the one sample is given by Y = (Y1*, . . . , Y*n) ∼ Q, where each entry Yi, i = 1*, . . . , n* is drawn independently from some distribution Qi, i = 1*, . . . , n* on R.

2. Under the planted model Pu, we associate u with a set of planted entries Φu ⊂ [n]. Then on sample is generated as follows. For the entries i /∈ Φu, we draw Yiindependently from Qi
(which is identical as in the Q measure). For the entries in Φu we draw from an arbitrary joint distribution Pu|Φu with the following symmetry condition: for any subset S ⊆ Φu, the marginal distribution Pu|ϕu(S) does not depend on u but only on S, i.e. Pu|S = PS.

Multiple well-known detection models satisfy this definition, such as, a well studied model of sparse regression [24, 6], Bernoulli group testing [1, 11], sparse phase retrieval [5], as well as multi-sample variants [8] of planted clique [29] and sparse (Wigner) PCA [2]. Satisfyingly, all planted sparse models directly satisfy Assumption 1 for the trivial group. In fact this result has already been established for a different use in [6, Proposition 3.6], proving that any *u, v* we have ⟨Lu, Lv⟩Q
 
≥ 1. We state here for completeness.

Lemma 2. Consider any PSM. For any *u, v* ∈ support(π), ⟨Lu, Lv⟩Q
 
≥ 1. This is to say, any PSM
satisfies Assumption 1 *for the trivial group.* The proof follows from [6, Proposition 3.6] for D = 0. Using this, one can apply our main equivalence Theorem 3 to multiple interesting planted sparse models and obtain old and new SQ- hardness results in a rather streamlined fashion. As an instantiation of this, in Appendix D.1 we prove the GFP-hardness for the mixed sparse linear regression setting studied in [5] in its conjecturally hard regime. We then use our equivalence theorem to translate it into an SQ-hardness result in the same regime. Our results complement the existing low-degree lower bound [5], providing further evidence for the hard phase of the problem.

## 4.2.1 Other Examples

Due to space constraint, we defer the following examples to Appendix D:
- *Non-Gaussian Component Analysis (NGCA):* Assumption 1 holds with G = Z2 for any symmetric prior. We recover the SQ-hardness result of [16] for the uniform prior via its equivalence with GFP-hardness, and establish a new SQ lower bound under a sparse prior.

- *Single-Index Models (SIM):* Again, Assumption 1 holds with G = Z2. We rederive the SQ-hardness result of [12] for the uniform prior through the GFP-hardness equivalence, and prove a new SQ lower bound for sparse priors.

- *Convex truncation detection:* Here Assumption 1 holds with the trivial group. In fact this assumption is precisely equivalent to the celebrated Gaussian Correlation Inequality on convex bodies [37, 32]. Using the GFP-hardness correspondence, we derive a new SQ lower bound that matches the current state-of-the-art polynomial-time algorithm of [15].

## 5 Gfp-Hardness Is Not Always Equal To Fp-Hardness

Recall that by definition, FP-hardness implies GFP-hardness. In this section, we show that the converse does not necessarily hold: we construct a P versus Q detection task that satisfies Assumption 1 and is easy under the FP criterion but hard under the GFP criterion. In particular, by using Theorem 2 and Theorem 3, the problem is also SQ-hard. Thus, while the FP criterion fails to capture the SQ-hardness in this case, our optimized GFP criterion correctly predicts it. As our initial departure from FP-hardness was that in many models the Euclidean overlap ⟨*u, v*⟩ might not be the "correct" choice, our example is carefully creating a model where the natural "overlap" ρG(*u, v*) (based on Theorem 3) is not a function of the Euclidean dot product.

The P versus Q problem is defined as follows. The null model is Q = Rad 12
⊗(n+1), i.e., each coordinate is an independent Rademacher random variable. For a signal u ∈ {0, 1}
n+1, the sample x ∼ Pu is generated by drawing each coordinate independently according to

$$x_{i}={\begin{cases}+1,&\text{w.p.}\frac12+r\cdot\frac{1-(1-\alpha)\cdot u_{i}}{2},\\ -1,&\text{w.p.}\frac12-r\cdot\frac{1-(1-\alpha)\cdot u_{i}}{2},\end{cases}}$$
$$(9)$$

where *α, r* ∈ (0, 1) are fixed constants to be chosen later. The following holds. Lemma 3. Let u, v ∈ {0, 1}
n+1. For any u, v ∈ {0, 1}
n+1, ⟨Lu, Lv⟩Q =Qn i=0 1 + r 2· α ui+vi.

Notice that our construction importantly ensures that the likelihood ratio inner product ⟨Lu, Lv⟩ is not solely a function of ⟨*u, v*⟩, but instead has a more intricate dependence on u and v. It is exactly this reason that leads to the discrepancy between GFP and FP hardness stated below.

Theorem 4. There exist a two-point prior π on u *such that, for* r = n
−1/2, α = n
−1+2ε, m = n 1−ε and D = n ε, where ε > 0 is any small constant, the following hold. The m-sample hypothesis testing problem Eu∼πP
⊗m u*versus* Q⊗m is (e D/2*, m,* Θ(n
−ε))*-GFP hard but not* (n
−1*, m,* exp (Θ(n ε)))-FP
hard. Moreover, via our equivalence theorem the model is (e n Θ(ε), n1−Θ(ε))*-SQ hard.*
The proof of this Theorem and the above Lemma can be found in Appendix E.

## 6 Conclusion

In this work, we generalize the Franz-Parisi (FP) criterion introduced by [6], motivated by the observation that the Euclidean dot product may not be the most natural geometry for all statistical task—a point partially illustrated by our example in Section 5. Our main result shows that optimizing the overlap event in the FP definition of [6] leads to a Generalized Franz-Parisi (GFP) hardness criterion, which is equivalent to SQ-hardness for models satisfying the mild Assumption 1. This assumption holds in a broad range of well-studied problems, including Gaussian additive models, planted sparse models, single-index models, and convex truncation. Our work significantly strengthens the theoretical foundation behind the (annealed) FP potential's predictions from statistical physics, but also opens several questions:
1. *(Algorithmic implications)* Does the optimal overlap function ρG(u, v)—as characterized in Theorem 2—yield meaningful algorithmic insights, particularly for local search or geometric methods?

2. *(The annealed potential)* Can similar equivalences be established for the original (also known as quenched) FP potential, or is the choice of the annealed version fundamental?

3. *(Interpretation of FP Area)* Why does the area under the FP curve appear to govern detection hardness? Is there some physical/algorithmic interpretation of this phenomenon?

4. *(Generalization to estimation)* Can our techniques be extended from detection to estimation tasks, for which the Franz-Parisi potential was originally introduced?

We believe these questions point toward promising future directions, with the potential to unify different approaches on the computational complexity of statistical inference.

## References

[1] Matthew Aldridge, Oliver Johnson, Jonathan Scarlett, et al. Group testing: an information theory perspective. Foundations and Trends® *in Communications and Information Theory*,
15(3-4):196–392, 2019.

[2] Arash A Amini and Martin J Wainwright. High-dimensional analysis of semidefinite relaxations for sparse principal components. In *2008 IEEE international symposium on information theory*, pages 2454–2458. IEEE, 2008.

[3] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Algorithmic thresholds for tensor pca. *The Annals of Probability*, 48(4):2052–2087, 2020.

[4] Gérard Ben Arous, Alexander S Wein, and Ilias Zadik. Free energy wells and overlap gap property in sparse pca. *Communications on Pure and Applied Mathematics*, 76(10):2410–2473, 2023.

[5] Gabriel Arpino and Ramji Venkataramanan. Statistical-computational tradeoffs in mixed sparse linear regression. In *The Thirty Sixth Annual Conference on Learning Theory*, pages 921–986. PMLR, 2023.

[6] Afonso S Bandeira, Ahmed El Alaoui, Samuel Hopkins, Tselil Schramm, Alexander S Wein, and Ilias Zadik. The franz-parisi criterion and computational trade-offs in high dimensional statistics. *Advances in Neural Information Processing Systems*, 35:33831–33844, 2022.

[7] Matthew Brennan, Guy Bresler, and Wasim Huleihel. Universality of computational lower bounds for submatrix detection. In *Conference on Learning Theory*, pages 417–468. PMLR, 2019.

[8] Matthew S Brennan, Guy Bresler, Sam Hopkins, Jerry Li, and Tselil Schramm. Statistical query algorithms and low degree tests are almost equivalent. In *Conference on Learning Theory*,
pages 774–774. PMLR, 2021.

[9] Siyu Chen, Beining Wu, Miao Lu, Zhuoran Yang, and Tianhao Wang. Can neural networks achieve optimal computational-statistical tradeoff? an analysis on single-index model. In The Thirteenth International Conference on Learning Representations.

[10] Zongchen Chen, Conor Sheehan, and Ilias Zadik. On the low-temperature mcmc threshold: the cases of sparse tensor pca, sparse regression, and a geometric rule. *arXiv preprint* arXiv:2408.00746, 2024.

[11] Amin Coja-Oghlan, Oliver Gebhard, Max Hahn-Klimroth, Alexander S Wein, and Ilias Zadik.

Statistical and computational phase transitions in group testing. In Conference on Learning Theory, pages 4764–4781. PMLR, 2022.

[12] Alex Damian, Loucas Pillaud-Vivien, Jason Lee, and Joan Bruna. Computational-statistical gaps in gaussian single-index models. In The Thirty Seventh Annual Conference on Learning Theory, pages 1262–1262. PMLR, 2024.

[13] Constantinos Daskalakis, Themis Gouleakis, Chistos Tzamos, and Manolis Zampetakis. Efficient statistics, in high dimensions, from truncated samples. In *2018 IEEE 59th Annual* Symposium on Foundations of Computer Science (FOCS), pages 639–649. IEEE, 2018.

[14] Constantinos Daskalakis, Themis Gouleakis, Christos Tzamos, and Manolis Zampetakis. Computationally and statistically efficient truncated regression. In *Conference on learning theory*, pages 955–960. PMLR, 2019.

[15] Anindya De, Shivam Nadimpalli, and Rocco A Servedio. Testing convex truncation. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 4050–4082. SIAM, 2023.

[16] Ilias Diakonikolas, Daniel M Kane, and Alistair Stewart. Statistical query lower bounds for robust estimation of high-dimensional gaussians and gaussian mixtures. In *2017 IEEE 58th* Annual Symposium on Foundations of Computer Science (FOCS), pages 73–84. IEEE, 2017.

[17] Ilias Diakonikolas, Weihao Kong, and Alistair Stewart. Efficient algorithms and lower bounds for robust linear regression. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 2745–2754. SIAM, 2019.

[18] Yunzi Ding, Dmitriy Kunisky, Alexander S Wein, and Afonso S Bandeira. Subexponential-time algorithms for sparse pca. *Foundations of Computational Mathematics*, 24(3):865–914, 2024.

[19] Jianqing Fan, Han Liu, Zhaoran Wang, and Zhuoran Yang. Curse of heterogeneity: Computational barriers in sparse mixture models and phase retrieval. *arXiv preprint arXiv:1808.06996*, 2018.

[20] Vitaly Feldman, Elena Grigorescu, Lev Reyzin, Santosh S Vempala, and Ying Xiao. Statistical algorithms and a lower bound for detecting planted cliques. *Journal of the ACM (JACM)*, 64(2):1–37, 2017.

[21] Silvio Franz and Giorgio Parisi. Recipes for metastable states in spin glasses. Journal de Physique I, 5(11):1401–1415, 1995.

[22] Silvio Franz and Giorgio Parisi. Effective potential in glassy systems: theory and simulations.

Physica A: Statistical Mechanics and its Applications, 261(3-4):317–339, 1998.

[23] Francis Galton. An examination into the registered speeds of american trotting horses, with remarks on their value as hereditary data. *Proceedings of the Royal Society of London*, 62(379387):310–315, 1898.

[24] David Gamarnik and Ilias Zadik. Sparse high-dimensional linear regression. estimating squared error and a phase transition. *The Annals of Statistics*, 50(2):880–903, 2022.

[25] S Das Gupta, Morris L Eaton, Ingram Olkin, Michael Perlman, Leonard J Savage, and Milton Sobel. Inequalities on the probability content of convex regions for elliptically contoured distributions. In Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability (Univ. California, Berkeley, Calif., 1970/1971), volume 2, pages 241–265, 1972.

[26] Samuel Hopkins. *Statistical Inference and the Sum of Squares Method*. PhD thesis, Cornell University, 2018.

[27] Daniel J Hsu, Clayton H Sanford, Rocco Servedio, and Emmanouil Vasileios Vlatakis-
Gkaragkounis. Near-optimal statistical query lower bounds for agnostically learning intersections of halfspaces with gaussian marginals. In *Conference on Learning Theory*, pages 283–312. PMLR, 2022.

[28] Hidehiko Ichimura. Semiparametric least squares (sls) and weighted sls estimation of singleindex models. *Journal of econometrics*, 58(1-2):71–120, 1993.

[29] Mark Jerrum. Large cliques elude the metropolis process. *Random Structures & Algorithms*,
3(4):347–359, 1992.

[30] Michael Kearns. Efficient noise-tolerant learning from statistical queries. Journal of the ACM
(JACM), 45(6):983–1006, 1998.

[31] Dmitriy Kunisky, Alexander S Wein, and Afonso S Bandeira. Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio. In ISAAC Congress
(International Society for Analysis, its Applications and Computation), pages 1–50. Springer, 2019.

[32] Rafał Latała and Dariusz Matlak. Royen's proof of the gaussian correlation inequality. In Geometric Aspects of Functional Analysis: Israel Seminar (GAFA) 2014–2016, pages 265–275.

Springer, 2017.

[33] Peter McCullagh. *Generalized linear models*. Routledge, 2019. [34] Andrea Montanari, Daniel Reichman, and Ofer Zeitouni. On the limitation of spectral methods: From the gaussian hidden clique problem to rank-one perturbations of gaussian tensors.

Advances in Neural Information Processing Systems, 28, 2015.

[35] Karl Pearson. On the systematic fitting of curves to observations and measurements. *Biometrika*,
1(3):265–303, 1902.

[36] Emile Richard and Andrea Montanari. A statistical model for tensor pca. *Advances in neural* information processing systems, 27, 2014.

[37] Thomas Royen. A simple proof of the gaussian correlation conjecture extended to multivariate gamma distributions. *arXiv preprint arXiv:1408.1028*, 2014.

[38] Tselil Schramm and Alexander S Wein. Computational barriers to estimation from low-degree polynomials. *The Annals of Statistics*, 50(3):1833–1858, 2022.

[39] Ilias Zadik, Min Jae Song, Alexander S Wein, and Joan Bruna. Lattice-based methods surpass sum-of-squares in clustering. In *Conference on Learning Theory*, pages 1247–1248. PMLR, 2022.

[40] Lenka Zdeborová and Florent Krzakala. Statistical physics of inference: Thresholds and algorithms. *Advances in Physics*, 65(5):453–552, 2016.

## Neurips Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: **The papers not including the checklist will be desk rejected.** The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit. Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
- You should answer [Yes] , [No] , or [NA] .

- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

- Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper. The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found. IMPORTANT, please:
- **Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist"**, - **Keep the checklist subsection headings, questions/answers and guidelines below.** - **Do not modify the questions and only use the provided macros for your answers**.

## 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Yes, the abstract and introduction make accurate claims about the paper's contributions and scope. We describe our results accurately. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss sufficiently the limitations of our results and provide clear assumptions under which they apply. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]
Justification: We write clearly all the assumptions that are required for our results to hold. The proofs are all provided in the appendices. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] Justification: There are no experimental results, as this is a theory paper. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: There are no experimental results, as this is a theory paper. Guidelines:
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

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: There are no experimental results, as this is a theory paper. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: There are no experimental results, as this is a theory paper. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [NA] Justification: There are no experimental results, as this is a theory paper. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have conducted research and presented our work in accordance with the Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: There are no direct societal impact of our work that we are aware of. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: There are no such risks that we are aware of. Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: Yes, we cite all the relevant works to be acknowledged to establish our results. Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

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

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: As this is a theory paper with no such crowdsourcing involved on human related subjects. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional review board (IRB) approvals or equivalent for research with human**

## Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: As this is a theory paper with no such crowdsourcing involved on human related subjects. Guidelines: