# An Optimized Franz-Parisi Criterion and its Equivalence with SQ Lower Bounds

Siyu Chen Department of Statistics and Data Science Yale University siyu.chen.sc3226@yale.edu

Theodor Misiakiewicz Department of Statistics and Data Science Yale University theodor.misiakiewicz@yale.edu

Ilias Zadik Department of Statistics and Data Science Yale University ilias.zadik@yale.edu

Peiyuan Zhang Department of Statistics and Data Science Yale University peiyuan.zhang@yale.edu

## Abstract

Bandeira et al. (2022) introduced the Franz-Parisi (FP) criterion for characterizing the computational hard phases in statistical detection problems. The FP criterion, based on an annealed version of the celebrated Franz-Parisi potential from statistical physics, was shown to be equivalent to low-degree polynomial (LDP) lower bounds for Gaussian additive models, thereby connecting two distinct approaches to understanding the computational hardness in statistical inference. In this paper, we propose a refined FP criterion that aims to better capture the geometric "overlap" structure of statistical models. Our main result establishes that this optimized FP criterion is equivalent to Statistical Query (SQ) lower bounds—another foundational framework in computational complexity of statistical inference. Crucially, this equivalence holds under a mild, verifiable assumption satisfied by a broad class of statistical models, including Gaussian additive models, planted sparse models, as well as non-Gaussian component analysis (NGCA), single-index (SI) models, and convex truncation detection settings. For instance, in the case of convex truncation tasks, the assumption is equivalent with the Gaussian correlation inequality (Royen, 2014) from convex geometry. In addition to the above, our equivalence not only unifies and simplifies the derivation of several known SQ lower bounds—such as for the NGCA model (Diakonikolas et al., 2017) and the SI model (Damian et al., 2024)—but also yields new SQ lower bounds of independent interest, including for the computational gaps in mixed sparse linear regression (Arpino et al., 2023) and convex truncation (De et al., 2023).

## 1 Introduction

Over the past decades, a central focus in statistical inference has been to understand the transition from computationally easy to hard regimes—that is, to characterize when a statistical task can be solved by polynomial-time algorithms. A key insight from this line of work is the emergence of *computational-statistical tradeoffs*: in many models, there exist broad parameter regimes where information-theoretic recovery is possible, yet no known polynomial-time algorithm succeeds.

Evidence for such tradeoffs spans multiple disciplines with varying levels of mathematical rigor. In particular, the statistical physics community has played an instrumental role by leveraging nonrigorous but highly predictive techniques to study average-case hardness. Their approach typically analyzes the geometry of solution spaces and identifies structural properties that correlate with algorithmic intractability (see [\[40\]](#page-12-0) for a survey). Remarkably, for many statistical models, the predictions from statistical physics have been in striking agreement with the performance of the best-known polynomial-time algorithms.

Alongside these heuristic predictions, rigorous frameworks from statistics and theoretical computer science have been developed to analyze the limitations of efficient algorithms. While ruling out *all* polynomial-time algorithms would require resolving P ̸= N P, substantial progress has been made by studying broad, expressive classes of polynomial-time algorithms. Two frameworks have emerged as particularly influential: *low-degree* (LD) *polynomial* lower bounds [\[31\]](#page-11-0) and *statistical query* (SQ) lower bounds [\[20\]](#page-11-1). For many "nice enough" detection problems, the lower bounds derived from these frameworks align closely with the performance of the best-known polynomial-time algorithms[<sup>1</sup>](#page-1-0) . This striking consistency has motivated the formulation of the so-called *low-degree conjecture* by Hopkins [\[26\]](#page-11-2), which posits that for sufficiently "symmetric and noisy" models, the failure of degree-O(log n) polynomials is indicative of the failure of all polynomial-time algorithms [\[31\]](#page-11-0).

Given this context, a natural question arises: can one formally connect these two seemingly distinct approaches? At first glance, the answer appears negative, due to a fundamental mismatch in scope. Statistical physics techniques are primarily geared toward estimation problems, where the goal is to recover a hidden signal, while the rigorous frameworks discussed above—such as LD and SQ lower bounds—are focused on detection or hypothesis testing, where the task is to distinguish between the presence or absence of a signal in a noisy environment[<sup>2</sup>](#page-1-1) . Nevertheless, a major step towards bridging this gap was taken by Bandeira et al. (2022) [\[6\]](#page-10-0), who introduced the *Franz-Parisi* (FP) *criterion* for computational hardness in detection tasks. Inspired by the seminal work of Franz and Parisi in spin glass theory [\[21\]](#page-11-3), the FP criterion provides a geometric perspective on computational hardness rooted in overlap structures. Crucially, Bandeira et al. showed that for Gaussian additive models, the FP criterion is mathematically equivalent to the low-degree (LD) lower bounds, thereby establishing a rigorous link between statistical physics heuristics and formal algorithmic barriers.

Specifically, consider the following general detection problem between two distributions P and Q supported on a subset of R <sup>n</sup>, which in what follows we refer to as a "<sup>P</sup> versus Q" task. Under the *planted* distribution <sup>P</sup> = <sup>E</sup>uPu, a signal u is drawn from a prior distribution π supported on Θ ⊆ S<sup>N</sup>−<sup>1</sup> , and one observes m independent samples Y1, . . . , Y<sup>m</sup> ∼ <sup>P</sup>u. Under the *null* distribution Q, the samples are drawn independently from Y1, . . . , Y<sup>m</sup> ∼ Q. The goal in the detection task[<sup>3</sup>](#page-1-2) is to distinguish between these two hypotheses based on the observed data, that is to find a test statistics with vanishing Type I and Type II errors, as n grows. Note that the computational question then is whether such a successful test statistic exists that also terminates in polynomial-in-mn time. To characterize the hardness of detection problems from the statistical physics perspective, Bandeira et al. in [\[6\]](#page-10-0) introduced the following notion of Franz-Parisi (FP) hardness:

Definition 1 (FP hardness). *For* D, m ∈ N, ε > 0, *we say that a* P *versus* Q *detection task is* (q, m, ε)*-FP hard if*

$$\mathbf{FP}: \quad \mathbb{E} \left[ \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(|\langle u, v \rangle| \leq \delta(q)) \right] \leq 1 + \varepsilon, \quad \text{where} \quad (1)$$

$$\delta(q) = \sup\{\delta > 0 : \pi^2(|\langle u, v \rangle| \geq \delta) \geq q^{-2}\}. \quad (2)$$

In the definition we denoted as customary L<sup>u</sup> = dP<sup>u</sup> dQ , u ∈ Θ, and for f, g ∈ L<sup>2</sup> (<sup>R</sup> <sup>n</sup>), the Hilbert space L 2 (Q) of (square integrable) functions from <sup>R</sup> <sup>n</sup> to <sup>R</sup>, we use ⟨f, g⟩<sup>Q</sup> = <sup>E</sup><sup>Y</sup> <sup>∼</sup>Qf(Y )g(Y ).

<sup>1</sup> For exceptions to this correspondence, see [\[39\]](#page-12-1) and discussion therein.

<sup>2</sup> Some recent work has extended low-degree lower bounds to estimation settings, beginning with [\[38\]](#page-12-2), though this direction remains relatively underdeveloped.

<sup>3</sup>The associated *estimation* problem consists in recovering the planted signal u from Y1, . . . , Y<sup>m</sup> ∼ <sup>P</sup>u.

We elaborate in Section [A.1](#page-22-0) on the statistical physics motivations behind this criterion, and only briefly highlight its core intuition here. The left-hand side of the FP condition integrates the function Fann(t) := <sup>E</sup> [⟨L ⊗m u , L⊗<sup>m</sup> v ⟩ · 1(⟨u, v⟩ = t)] , over a (1 − q −2 )-typical region of the overlap variable t, corresponding to the constraint |⟨u, v⟩| ≤ δ(q). This function Fann(t) is an annealed proxy for the Franz-Parisi potential, a central object in statistical physics that has long served as a predictor of algorithmic hardness [\[40\]](#page-12-0). Intuitively, the Franz-Parisi potential captures the energy landscape experienced by local algorithms—such as Langevin or Glauber dynamics—whose performance is constrained by the geometry of the underlying signal space. The overlap ⟨u, v⟩ naturally quantifies a local "geometric" similarity between signals, making it a meaningful argument for Fann(t) and explaining its role within the FP criterion.

Returning to the definition of FP hardness, the parameter m corresponds to the sample size, and one should interpret q as a proxy for the required runtime. In this light, Bandeira et al. (2022) proved that, for Gaussian additive models, FP-hardness is equivalent to the failure of degree-D = log q polynomials to solve the detection task with m samples—i.e., roughly the authors of [\[6\]](#page-10-0) showed that the problem is (q, m, O(1))-FP hard if and only if it is "hard" for degree-log q polynomials to solve the detection task[<sup>4</sup>](#page-2-0) . Hence, based on the current belief in the literature of low-degree lower bounds that a D-degree lower bound implies that the detection task requires at least e <sup>D</sup> runtime to be solved, e.g., see [\[18\]](#page-11-4), proving a task is ((mn) <sup>ω</sup>(1), m, O(1))-FP hard for a Gaussian additive model provides rigorous evidence for polynomial-time hardness for the task.

Despite this success, the connection between the FP potential and other rigorous notions of algorithmic hardness remains limited. [\[6\]](#page-10-0) only established a formal equivalence for Gaussian additive models and an one-sided implication for planted sparse models between the FP criterion and "low-degree" lower bounds. They further presented counterexamples where the equivalence fails entirely. In this work, our aim is to extend the Franz-Parisi criterion to rigorously characterize hardness beyond Gaussian additive models, and to clarify the scope and limitations of this framework across a broader class of statistical models.

#### 1.1 Main Contributions

Our main contribution is to propose a slight modification of the FP-hardness criterion from [\[6\]](#page-10-0), motivated by the observation that sticking to the Euclidean geometry assumption (and hence the "overlap" ⟨u, v⟩) may fail to capture the "true" hardness of some statistical models. We remark that this is an arguably natural modification, as (1) there are many statistical models for which the Euclidean geometry appears unnatural for navigating their parameter space (see Section [5](#page-9-0) for a simple such construction), and (2) even in statistical physics settings, the Franz-Parisi potential is often considered under a more general notion of overlap [\[22\]](#page-11-5). Motivated by these considerations, we propose optimizing the "overlap" event inside the FP-hardness definition, subject only to a mild symmetry assumption for technical reasons. This leads to the following new criterion of FP-hardness:

Definition 2 (Generalized Franz Parisi (GFP) hardness under symmetry G). *Fix* q, m ∈ N, ε > 0 *and a group* G *of finite order acting on the parameter space of the signal. We say a "*P *versus* Q*" problem is* (q, m, ε)*-GFP*<sup>G</sup> *hard if*

$$GFP_G : \inf_{\substack{A: \pi \otimes^2(A) \geq 1 - q^{-2} \\ A \text{ is } G^2\text{-invariant}}} \mathbb{E} \left[ \left\langle L_u^{\otimes m}, L_v^{\otimes m} \right\rangle_{\mathbb{Q}} \mathbf{1}(A) \right] \leq 1 + \varepsilon. \quad (3)$$

As in the original FP-hardness framework of [\[6\]](#page-10-0), one should interpret q as a proxy for runtime, and therefore ((mn) <sup>ω</sup>(1), m, O(1))-GFP hardness should be providing evidence of polynomial-time hardness with m samples in this framework. We highlight that the assumption on the invariance of the optimizing event under group G is made for technical reasons to enhance the applicability of our hardness criterion. We point the reader to Section [3.1](#page-5-0) for further discussion of this assumption.

The main result of this work is that the "optimized" notion of GFP-hardness is fundamentally connected with the well-established framework of Statistical Query (SQ) hardness. The SQ framework

<sup>4</sup> [\[6\]](#page-10-0) established this equivalence for m = 1, but the argument extends directly to general m.

was initially proposed by Kearns in [\[30\]](#page-11-6) to capture the power of noise-tolerant algorithms. The notion of a statistical dimension proposed by [\[20\]](#page-11-1) allowed for achieving powerful lower bounds against SQ methods, which we refer to from now on as SQ-hardness results. We employ here a slight strengthening of the notion of SQ-hardness from [\[20\]](#page-11-1), introduced in [\[8\]](#page-10-1).

Definition 3 (SQ hardness). *Fix* q, m ∈ N*. We say a "*<sup>P</sup> *versus* Q*" detection problem is* (q, m)*-SQ hard if*

$$SQ: \sup_{A:\pi^2(A)\geq q^{-2}} \mathbb{E} \left[ \left| \langle L_u, L_v \rangle_{\mathbb{Q}} - 1 \right| \mid A \right] \leq \frac{1}{m}. \quad (4)$$

Roughly, a detection problem is (q, m)-SQ hard if any Statistical Query method succeeding at solving it with m samples requires q queries, which should be interpreted as requiring runtime q (see [\[8,](#page-10-1) Appendix A] for more details and motivation). Hence, proving a task is ((mn) <sup>ω</sup>(1), m)-SQ hard provides evidence for polynomial-time hardness for the task.

Our main result is informally described as follows.

Theorem 1. *(Informal, GFP and SQ equivalence) Consider any* P *versus* Q *detection task which we assume (1) it satisfies a mild assumption with respect to a group* G *of finite order acting on the parameter space (namely Assumption [1](#page-5-1) below), and (2) it is information-theoretically impossible to be solved with* mIT *samples. Then the following holds for any samples size* m *and proxy runtime* q = mΩ(1) .

- *If the task is* (q, m)*-SQ hard, then it is also* (Θ(q), Θ(m), O(1))*-GFP*G*-hard.*
- *If the task is* (q, m, O(1))*-GFP*G*-hard, then it is also* (mΘ(mIT) , m<sup>1</sup>−o(1))*-SQ hard.*

Note that often in statistical tasks of interest mIT = ω(log n) (in fact, more often than not mIT = poly(n)). Under this condition, Theorem [1](#page-3-0) implies that a task is ((mn) <sup>ω</sup>(1), m<sup>1</sup>−o(1), O(1))-GFP hard if and only if it is ((mn) <sup>ω</sup>(1), m<sup>1</sup>−o(1))-SQ hardness, matching the two criteria for hardness.

On top of that, as we mentioned above and discuss in Section [3.1,](#page-5-0) the required Assumption [1](#page-5-1) on the detection task is rather mild. In fact, it turns out that it is satisfied for several models of recent interest in the community, making a strong case of how the Generalized Franz-Parisi criterion now correctly predicts the hardness phase for them. Importantly, these models include the Gaussian additive models and also greatly extend beyond them, significantly extending the key message from [\[6\]](#page-10-0) about connecting the physics-based forms of hardness to more rigorous frameworks. We list now some of the tasks that satisfy Assumption [1.](#page-5-1)

- 1. *All Gaussian additive models* (GAMs), under any symmetric prior, satisfy Assumption [1](#page-5-1) with G = <sup>Z</sup><sup>2</sup> that flips the sign of the signal. Moreover, in that case the Generalized Franz Parisi criterion is equivalent to the Franz-Parisi criterion, that is the optimizing event A in
  - [\(3\)](#page-2-1) is of the form {|⟨u, v⟩| ≤ δ(q)}. Hence, Theorem [1](#page-3-0) allows us to extend the result of [\[6\]](#page-10-0) which proved the equivalence of FP-hardness to Low-degree hardness for GAMs, to also proving FP-hardness equivalent with SQ-hardness for these settings[<sup>5</sup>](#page-3-1) .
- 2. *All Planted Sparse Models* satisfy Assumption [1](#page-5-1) for the trivial group G = {id}. In particular, using Theorem [1](#page-3-0) we can prove that GFP-hardness is equivalent to SQ-hardness for multiple well-studied models such as sparse phase retrieval [\[5\]](#page-10-2), sparse regression [\[24,](#page-11-7) [6\]](#page-10-0), (multisample) sparse PCA [\[7\]](#page-10-3), and Bernoulli group testing [\[11\]](#page-10-4). As a corollary of this connection, we present a straightforward argument to obtain an SQ lower bound for the mixed sparse linear regression problem [\[5\]](#page-10-2). We remark that in [\[6\]](#page-10-0) it has been proven that FP-hardness implies low-degree hardness for all Planted Sparse Models, but no result was presented for the other direction.
- 3. *All Non-Gaussian component analysis (NGCA) models* and *all single-index models (under any symmetric prior)* satisfy Assumption [1](#page-5-1) with G = <sup>Z</sup>2, Therefore, via Theorem [1](#page-3-0) GFPhardness is again equivalent with SQ-hardness for these tasks.

<sup>5</sup>We remark that such a connection could also be made via the results of [\[8\]](#page-10-1), since GAMs are noise-robust.

- 4. *All Gaussian convex truncation models* satisfy Assumption [1](#page-5-1) for G = {id}. In particular, interestingly Assumption [1](#page-5-1) for these models is exactly equivalent to the celebrated Gaussian correlation inequality for convex bodies in probability theory, which was a multi-decade open problem posed in 1972 in [\[25\]](#page-11-8) that was finally proven by Royen in 2014 [\[37\]](#page-12-3). Leveraging the equivalence between GFP-hardness and SQ-hardness in Theorem [1,](#page-3-0) we establish an SQlower bound for the convex truncation detection task. This allows us to provide, to the best of our knowledge, the first formal evidence that the current state-of-the-art polynomial-time detection method for convex truncation proposed in [\[15\]](#page-10-5) has optimal sample complexity.

We also complement our results, with a simple example satisfying Assumption [1](#page-5-1) where FP-hardness does *not* coincide with GFP-hardness, which we interpret as a model where the Euclidean geometry is not appropriate. We finally conclude the paper with a discussion.

For completeness, we prove in Appendix [B](#page-26-0) the equivalence between GFP-hardness and low-degree (LD) polynomial hardness for noise-robust models. This result follows by combining our GFP–SQ equivalence with the equivalence between SQ-hardness and LD-hardness under noise robustness shown by Brennan et al. [\[8\]](#page-10-1). In particular, our GFP-hardness results for the examples presented in this paper immediately imply low-degree lower bounds in all those settings. This substantially extends the equivalence established in [\[6\]](#page-10-0). For clarity and readers' convenience, we also include succinct proofs of the SQ-to-LD equivalence, adapted from [\[8\]](#page-10-1).

## 2 Setting and Definitions

We first recall the definition of a "P versus Q" task mentioned in the Introduction. Under the *planted* distribution <sup>P</sup> = <sup>E</sup>uPu, a signal u is drawn from a prior distribution π supported on Θ ⊆ S<sup>N</sup>−<sup>1</sup> , and one observes m independent samples Y1, . . . , Y<sup>m</sup> ∼ <sup>P</sup>u. Under the *null* distribution Q, the samples are drawn independently from Y1, . . . , Y<sup>m</sup> ∼ Q. The goal in the detection task[<sup>6</sup>](#page-4-0) is the so-called strong detection task to distinguish between these two hypotheses based on the observed data, that is to find a test statistics with vanishing Type I and Type II errors, as n grows. We will also be interested in the weak detection task, which is that the sum of type I and type II errors is at most 1 − ε for some fixed ε > 0 (not depending on n). In other words, strong detection means the test succeeds with high probability, while weak detection means the test has some non-trivial advantage over random guessing.

Throughout, we will work in the Hilbert space L 2 (Q) of (square integrable) functions <sup>R</sup> <sup>N</sup> → <sup>R</sup> with inner product ⟨f, g⟩<sup>Q</sup> := <sup>E</sup><sup>Y</sup> <sup>∼</sup>Q[f(Y )g(Y )] and corresponding norm ∥f∥<sup>Q</sup> := ⟨f, f⟩ 1/2 Q . We will assume that <sup>P</sup><sup>u</sup> is absolutely continuous with respect to Q for all u ∈ supp(π), use L<sup>u</sup> := <sup>d</sup>P<sup>u</sup> dQ to denote the likelihood ratio, and assume that L<sup>u</sup> ∈ L 2 (Q) for all u ∈ supp(π). The likelihood ratio between <sup>P</sup> and Q is denoted by L := <sup>d</sup><sup>P</sup> <sup>d</sup><sup>Q</sup> = <sup>E</sup>u∼µLu. Observe that for m samples, we denote by L<sup>m</sup> = <sup>E</sup>u∼µL<sup>u</sup> the m-sample likelihood ratio. Finally, for a function f : <sup>R</sup> <sup>N</sup> → <sup>R</sup> and integer D ∈ N, we let f <sup>≤</sup><sup>D</sup> denote the orthogonal (w.r.t. ⟨·, ·⟩Q) projection of f onto the subspace of polynomials of degree at most D.

An important identity between the (squared) norm of the likelihood ratio with m samples and the *chi-squared divergence* χ 2 (<sup>P</sup> <sup>⊗</sup><sup>m</sup> ∥ Q⊗<sup>m</sup>) is

$$\|L\|_{\mathbb{Q}}^2 = \|\mathbb{E}_{u \sim \mu} L_u\|_{\mathbb{Q}}^2 = \chi^2(\mathbb{P} \parallel \mathbb{Q}) + 1 \geq 1.$$

This quantity has the following standard implications for *information-theoretic* impossibility of testing, in the asymptotic regime n → ∞. The proofs can be found in e.g. [\[34,](#page-12-4) Lemma 2].

- If ∥L∥ 2 <sup>Q</sup> = O(1) then strong detection is impossible.
- If ∥L∥ 2 <sup>Q</sup> = 1 + o(1) then weak detection is impossible.

<sup>6</sup>The associated *estimation* problem consists in recovering the planted signal u from Y1, . . . , Y<sup>m</sup> ∼ <sup>P</sup>u.

## 3 Main Results

In this section, we formally present our equivalence between GFP-hardness and SQ-hardness.

#### 3.1 The Assumption

As mentioned in the Introduction, all our results operate under a crucial assumption on the "P versus Q" detection task. The assumption is as follows.

Assumption 1. *Given any "*P *versus* Q*" task, there exists a* π*-preserving finite group* G *acting on the parameter space* Θ*, i.e., for all* g ∈ G*,* g(v) (d) = v *for* v ∼ π*, such that for any sample size* m *for any* u, v ∈ Θ*, the following "correlation" inequality holds for any* k ∈ N

$$\mathbb{E}_{g,g' \sim \text{Unif}(G)}(\langle L_{g(u)}, L_{g'(v)} \rangle_{\mathbb{Q}} - 1)^k \geq 0. \quad (5)$$

We first remark that [\(5\)](#page-5-2) is a natural condition even if G is the trivial group, G = {id}. Indeed in that case [\(5\)](#page-5-2) asks that for all u, v ∈ Θ,

$$\langle L_u, L_v \rangle_{\mathbb{Q}} \geq 1. \quad (6)$$

Recall that if one averages over all (u, v) ∼ π ⊗2 , we have by standard identities

$$\mathbb{E}\langle L_u, L_v \rangle_{\mathbb{Q}} = \mathbb{E}_{\mathbb{Q}} \|\mathbb{E}_u L_u\|_2^2 = 1 + \chi^2(\mathbb{P}, \mathbb{Q}) \geq 1.$$

Thus, [\(6\)](#page-5-3) should be understood as a pointwise condition that is guaranteed to hold in expectation over the product measure π ⊗2 for any P, Q. While this pointwise condition turns out to be vanilla satisfied in many models (such as Planted Sparse Models or Convex Truncation settings), a slight modification of it—leading to [\(5\)](#page-5-2)—applies more broadly. Specifically, this modified condition requires [\(6\)](#page-5-3) to hold for a pair u, v only after performing a "small" averaging over the a group orbit that preserves the prior π. For instance, if the prior is symmetric around 0 and the group G is <sup>Z</sup>2, which acts by flipping the sign of the signal u, then for k = 1, condition [\(5\)](#page-5-2) reduces to demonstrating that, for all u, v,

$$\frac{1}{4}(\mathbb{E}\langle L_u, L_v \rangle_{\mathbb{Q}} + \mathbb{E}\langle L_{-u}, L_v \rangle_{\mathbb{Q}} + \mathbb{E}\langle L_u, L_{-v} \rangle_{\mathbb{Q}} + \mathbb{E}\langle L_{-u}, L_{-v} \rangle_{\mathbb{Q}}) \geq 1,$$

which is significantly less restrictive than the original pointwise condition [\(6\)](#page-5-3). This averaging approach allows for much greater generality, making it applicable to various settings, including Gaussian additive models, single-index models, and Non-Gaussian component analysis settings.

Remark 3.1. We finally make a trivial remark that will be useful in verifying [\(5\)](#page-5-2) in our examples in Section [4](#page-7-0) with symmetric prior. In all of them by symmetry we have for all u, v ⟨Lu, L−v⟩<sup>Q</sup> = ⟨L−u, Lv⟩<sup>Q</sup> and ⟨Lu, Lv⟩<sup>Q</sup> = ⟨L−u, L−v⟩Q. Using that and the trivial fact that for all x, y ∈ <sup>R</sup>, if x + y ≥ 0 then x <sup>k</sup> + y <sup>k</sup> ≥ 0 for all k ∈ <sup>N</sup>, we conclude that if G is either the trivial group or <sup>Z</sup><sup>2</sup> (which will be the case in all examples of Section [4\)](#page-7-0) it suffices to check the case k = 1 in [\(5\)](#page-5-2), and then it automatically holds for all k ∈ N.

Remark 3.2. As mentioned in the previous remark, we highlight that in all our examples in Section [4](#page-7-0) of our GFP-SQ equivalence theorem below, we either use G to be the trivial group or <sup>Z</sup>2. The reason we state our assumption Assumption [1](#page-5-1) for a general finite group G is for potential further applications of our work.

#### 3.2 The GFP-SQ equivalence

#### 3.2.1 Simplifying GFP-hardness

We present our equivalence theorem in two steps. First, we identify an approximate optimal "overlap" event A in the definition of GFP-hardness, which simplifies GFP-hardness significantly and makes GFP-hardness easier to establish in applications. Then, we prove the equivalence between this simplified version and SQ-hardness.

Given a group G acting on the parameter space of the signal, it turns out that the approximately optimal "overlap" event A takes the form {ρG(u, v) ≤ r} for the following notion of "overlap" between u, v,

$$\rho_G(u, v) = \max_{g, g' \in G} \{|\langle L_g(u), L_{g'}(v) \rangle_{\mathbb{Q}} - 1|\}.$$

In particular, focusing only on such type of events we define the following version of FP-hardness.

Definition 4 (ρG-FP hardness). *Fix* q, m ∈ <sup>N</sup>, ε > 0 *and a finite group* G *acting on the parameter space of the signal. We say a "*<sup>P</sup> *versus* Q*" problem is* (q, m, ε)*-*ρG*-FP hard if*

$$\rho_{G^*} \mathbf{FP} : \quad \mathbb{E} \left[ \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \leq 1 + \varepsilon, \quad \text{where} \quad (7)$$

$$r(q) = \sup\{r : \pi^2(\rho_G(u, v) \geq r) \geq q^{-2}\}, \quad (8)$$

We prove that GFPG-hardness is equivalent to ρG-FP hardness under Assumption [1.](#page-5-1)

Theorem 2. *Consider any "*P *versus* Q*" task that satisfies Assumption [1](#page-5-1) for a group* G*. Suppose* m, q ∈ N *and* ε > 0. *Then the following statements hold.*

- *1. If the task is* (q, m, ε)*-*ρG*-FP hard, then the task is also* (q, m, ε)*-GFP*<sup>G</sup> *hard.*
- *2. Assume there exists an* r > 0 *such that* π 2 (ρG(u, v) < r) = 1 − q <sup>−</sup><sup>2</sup> *and that* m *is even. Then, if the task is* (q, m, ε)*-GFP*<sup>G</sup> *hard, then it is* (q, m, <sup>3</sup> |G| (1 +ε) +m·χ 2 (P, Q))*-*ρG*-FP hard. In particular, if* mχ<sup>2</sup> (P, Q) = O(1), *the task is* (q, m, O(1 + ε))*-*ρG*-FP hard.*

The proof of this theorem can be found in Appendix [C.1.](#page-31-0)

Remark 3.3. While the first implication is immediate to grasp, the second implication has some additional conditions we now elaborate upon. First, both the requirements of the existence of r with the desired probability mass and the parity of m are for technical convenience, and both can be easily remove with some tedious work. Second, any potential "blow-up" in the ε-term for ρG-FP hard depends only on |G|, which should be treated as constant, and the term m · χ 2 (P, Q), which is an easy to compute quantity (usually n = 1 and χ 2 (P, Q) is an one-dimensional integral). Moreover, it is almost always of order O(1) for detection tasks that are conjecturally hard with m samples. Indeed, the mathematical reason behind this is exactly that it is equal to the squared L 2 -norm of the projection of the likelihood onto the degree-1 polynomial space, i.e., on the span of linear functions. On top of that, if the detection task is (q, m)-SQ hard *for any* q then it holds directly mχ<sup>2</sup> (P, Q) = O(1) as well. We elaborate more on this in Remark [B.1](#page-27-0) in Section [B.](#page-26-0)

#### 3.2.2 The equivalence

As we have already proven an equivalence between GFP-hardness and ρG-FP hardness, it suffices to connect the latter with SQ-hardness. This is the topic of the next theorem.

Theorem 3 (SQ and ρG-FP Equivalence). *Suppose a "*<sup>P</sup> *versus* Q*" task satisfies Assumption [1](#page-5-1) for a group* G.

- *1. If the task is* (q, m)*-SQ hard for some* q, m *with* q > 2 *then, it is also* (q ′ , m′ , e|G| <sup>−</sup><sup>1</sup>m′/m) ρG*-FP hard for any integers* q ′ < q/√ 2 *and* m′ ≤ m/2*.*
- *2. Suppose the task is* (q, m, ε)*-*ρG*-FP hard for some* q, m *integers. Assume that there exists an* r = r(q) > 0 *such that* π 2 (ρG(u, v) < r) = 1 − q <sup>−</sup><sup>2</sup> *and* m *is even. Then, the model is also* (q ′ , m′ )*-SQ hard for any even integer* t *with* t ≤ log q/ log m *and any integer* q ′ > 0*, where*

$$m' = \frac{m}{(t(1 + \varepsilon)^{1/t} + \chi^2(\mathbb{P}^{\otimes 4t} \parallel \mathbb{Q}^{\otimes 4t}))(q')^{2/t}}.$$

*In particular, if for some sample size* mIT, *we have*

$$\chi^2(\mathbb{P}^{\otimes m_{\text{IT}}} \parallel \mathbb{Q}^{\otimes m_{\text{IT}}}) = O(1).$$

$$(b) \text{ (Large enough } q) \qquad q \geq m^{m_{1T}}$$
then the model is  $(m^{\delta m_{1T}}, \Theta(\frac{m^{1-O(\delta)}}{m_{1T}(1+\varepsilon)}))$ - $SQ$  hard for any  $\delta > 0$ .

The proof of this theorem can be found in Appendix [C.2.](#page-33-0) Similar to Theorem [2,](#page-6-0) the conditions on r, m of part 2 in Theorem [3](#page-6-1) are for technical convenience and can be easily removed. As we discussed in the Introduction the assumption that there exists some sufficiently growing mIT (e.g., growing super-logarithmically in n) is natural for multiple commonly studied models. We remark that the condition on the information theory threshold mIT to be growing with n is also necessary, by constructing a variant of the planted clique problem which satisfies Assumption [1,](#page-5-1) it is not SQ-hard and is GFP-hard. Lastly, we also note that our introduced Assumption [1](#page-5-1) is also necessary for the equivalence. In Section [A,](#page-22-1) we discuss a counterexample not satisfying Assumption [1](#page-5-1) that is GFP-hard, but not SQ-hard.

Remark 3.4. We note that while our bounds in the equivalence of Theorem [2](#page-6-0) deteriorate when |G| becomes large, a slightly more general equivalence between GFP and SQ, using a variant of ρG, can also be proven for infinite groups G under an "hypercontractivity" assumption on ⟨Lu, Lv⟩<sup>Q</sup> with respect to the pair (u, v). We omit this generalization as in all relevant examples in this work a small group action using either the trivial or 2-cyclic group suffices.

## 4 Examples

In this section, we discuss two popular classes of detection tasks that satisfy Assumption [1](#page-5-1) and hence fall under our GFP-SQ equivalence. Further examples are deferred to Appendix [D.](#page-35-0)

#### 4.1 Gaussian Additive Models

A P versus Q task is a Gaussian additive model (GAM) if it satisfies:

- 1. Under the null model, Q = N (0, In).
- 2. Under the planted model <sup>P</sup><sup>u</sup> (for u ∈ S n−1 ), for some signal-to-noise ratio (SNR) λ > 0 we set

$$Y = \lambda u + Z, \quad \text{for some } Z \sim \mathbb{Q}.$$

GAMs includes multiple well-studied models in the literature, with the predominant examples being (multisample variants) of tensor PCA [\[36\]](#page-12-5) and sparse PCA [\[2\]](#page-10-6). For such models, it can be straightforwardly checked (see [\[6,](#page-10-0) Proposition 2.3]) that for all u, v,

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = e^{\lambda^2 \langle u, v \rangle}.$$

So for instance, in the case of non-negative sparse PCA where u, v are binary k-sparse vectors in (see e.g., [\[4,](#page-10-7) [10\]](#page-10-8)) we always have ⟨u, v⟩ ≥ 0, and therefore Assumption [1](#page-5-1) is always satisfied for the trivial group G. On top of that, Assumption [1](#page-5-1) remains true for any prior which is symmetric around 0; this time Assumption [1](#page-5-1) is also always satisfied by choosing the action of G = <sup>Z</sup><sup>2</sup> which flips the sign of u. We remark that symmetric priors encompass most commonly used priors for GAMs, e.g., for tensor PCA where u = vec(x ⊗r ), x ∼ Unif(S d−1 ).

Lemma 1. *Consider any GAM with symmetric* π, *i.e.,* v = −v, v ∼ π*. For any* u, v ∈ support(π)*,*

$$\frac{1}{4}(\langle L_u, L_v \rangle_{\mathbb{Q}} + \langle L_{-u}, L_v \rangle_{\mathbb{Q}} + \langle L_u, L_{-v} \rangle_{\mathbb{Q}} + \langle L_{-u}, L_{-v} \rangle_{\mathbb{Q}}) \geq 1.$$

*Moreover, any GAM satisfies Assumption [1](#page-5-1) for* G = <sup>Z</sup><sup>2</sup> *acting by flipping the sign of* u*.*

*Proof.* Notice

$$\frac{1}{4}(\langle L_u, L_v \rangle_{\mathbb{Q}} + \langle L_{-u}, L_v \rangle_{\mathbb{Q}} + \langle L_u, L_{-v} \rangle_{\mathbb{Q}} + \langle L_{-u}, L_{-v} \rangle_{\mathbb{Q}}) = \frac{1}{2}(\exp(\lambda^2 \langle u, v \rangle) + \exp(-\lambda^2 \langle u, v \rangle)) \geq 1.$$

Given the above lemma, we conclude the (almost) equivalence between GFP-hardness and SQhardness from Theorem [3.](#page-6-1)

Remark 4.1. We remark that in the symmetric prior case for a GAM and G = <sup>Z</sup><sup>2</sup> acting by flipping the sign of u, ρG(u, v) = exp λ |⟨u, v⟩| is an increasing function of |⟨u, v⟩|. Hence, for such GAMs we conclude via Theorem [2](#page-6-0) that FP-hardness is equivalent to GFP-hardness, and therefore also to SQ-hardness. This is in agreement with the results of [\[6\]](#page-10-0) establishing that FP-hardness is equivalent to LD-hardness; in fact, our approach can offer an alternative proof of their result via the LD-SQ equivalence [\[8\]](#page-10-1) and the noise robustness of GAMs (see Theorem Theorem [6\)](#page-29-0).

#### 4.2 Planted Sparse Models

In [\[6\]](#page-10-0), the authors introduced the family of planted sparse models (PSM) and proved that FP-hardness for a PSM implies it's also low-degree hard. We start with the definition.

A P versus Q task is a planted sparse model (PSM) if it satisfies:

- 1. Under the null model, the one sample is given by Y = (Y1, . . . , Yn) ∼ Q, where each entry Yi , i = 1, . . . , n is drawn independently from some distribution Q<sup>i</sup> , i = 1, . . . , n on R.
- 2. Under the planted model <sup>P</sup>u, we associate u with a set of planted entries Φ<sup>u</sup> ⊂ [n]. Then on sample is generated as follows. For the entries i /∈ Φu, we draw Y<sup>i</sup> independently from Q<sup>i</sup> (which is identical as in the Q measure). For the entries in Φ<sup>u</sup> we draw from an arbitrary joint distribution Pu|<sup>Φ</sup><sup>u</sup> with the following symmetry condition: for any subset S ⊆ Φu, the marginal distribution Pu|<sup>ϕ</sup><sup>u</sup>
  - (S) does not depend on u but only on S, i.e. Pu|<sup>S</sup> = PS.

Multiple well-known detection models satisfy this definition, such as, a well studied model of sparse regression [\[24,](#page-11-7) [6\]](#page-10-0), Bernoulli group testing [\[1,](#page-10-9) [11\]](#page-10-4), sparse phase retrieval [\[5\]](#page-10-2), as well as multi-sample variants [\[8\]](#page-10-1) of planted clique [\[29\]](#page-11-9) and sparse (Wigner) PCA [\[2\]](#page-10-6).

Satisfyingly, all planted sparse models directly satisfy Assumption [1](#page-5-1) for the trivial group. In fact this result has already been established for a different use in [\[6,](#page-10-0) Proposition 3.6], proving that any u, v we have ⟨Lu, Lv⟩<sup>Q</sup> ≥ 1. We state here for completeness.

Lemma 2. *Consider any PSM. For any* u, v ∈ support(π)*,* ⟨Lu, Lv⟩<sup>Q</sup> ≥ 1. *This is to say, any PSM satisfies Assumption [1](#page-5-1) for the trivial group.*

The proof follows from [\[6,](#page-10-0) Proposition 3.6] for D = 0. Using this, one can apply our main equivalence Theorem [3](#page-6-1) to multiple interesting planted sparse models and obtain old and new SQhardness results in a rather streamlined fashion. As an instantiation of this, in Appendix [D.1](#page-35-1) we prove the GFP-hardness for the mixed sparse linear regression setting studied in [\[5\]](#page-10-2) in its conjecturally hard regime. We then use our equivalence theorem to translate it into an SQ-hardness result in the same regime. Our results complement the existing low-degree lower bound [\[5\]](#page-10-2), providing further evidence for the hard phase of the problem.

#### 4.2.1 Other examples

Due to space constraint, we defer the following examples to Appendix [D:](#page-35-0)

- *Non-Gaussian Component Analysis (NGCA):* Assumption [1](#page-5-1) holds with G = <sup>Z</sup><sup>2</sup> for any symmetric prior. We recover the SQ-hardness result of [\[16\]](#page-11-10) for the uniform prior via its equivalence with GFP-hardness, and establish a new SQ lower bound under a sparse prior.
- *Single-Index Models (SIM):* Again, Assumption [1](#page-5-1) holds with G = <sup>Z</sup>2. We rederive the SQ-hardness result of [\[12\]](#page-10-10) for the uniform prior through the GFP-hardness equivalence, and prove a new SQ lower bound for sparse priors.
- *Convex truncation detection:* Here Assumption [1](#page-5-1) holds with the trivial group. In fact this assumption is precisely equivalent to the celebrated Gaussian Correlation Inequality on convex bodies [\[37,](#page-12-3) [32\]](#page-12-6). Using the GFP-hardness correspondence, we derive a new SQ lower bound that matches the current state-of-the-art polynomial-time algorithm of [\[15\]](#page-10-5).

## 5 GFP-hardness is not always equal to FP-hardness

Recall that by definition, FP-hardness implies GFP-hardness. In this section, we show that the converse does not necessarily hold: we construct a P versus Q detection task that satisfies Assumption [1](#page-5-1) and is easy under the FP criterion but hard under the GFP criterion. In particular, by using Theorem [2](#page-6-0) and Theorem [3,](#page-6-1) the problem is also SQ-hard. Thus, while the FP criterion fails to capture the SQ-hardness in this case, our optimized GFP criterion correctly predicts it. As our initial departure from FP-hardness was that in many models the Euclidean overlap ⟨u, v⟩ might not be the "correct" choice, our example is carefully creating a model where the natural "overlap" ρG(u, v) (based on Theorem [3\)](#page-6-1) is not a function of the Euclidean dot product.

The <sup>P</sup> versus Q problem is defined as follows. The null model is Q = Rad 2 <sup>⊗</sup>(n+1), i.e., each coordinate is an independent Rademacher random variable. For a signal u ∈ {0, 1} <sup>n</sup>+1, the sample x ∼ <sup>P</sup><sup>u</sup> is generated by drawing each coordinate independently according to

$$x_i = \begin{cases} +1, & \text{w.p. } \frac{1}{2} + r \cdot \frac{1-(1-\alpha) \cdot u_i}{2}, \\ -1, & \text{w.p. } \frac{1}{2} - r \cdot \frac{1-(1-\alpha) \cdot u_i}{2}, \end{cases} \quad (9)$$

where α, r ∈ (0, 1) are fixed constants to be chosen later. The following holds.

Lemma 3. *Let* u, v ∈ {0, 1} <sup>n</sup>+1*. For any* u, v ∈ {0, 1} n+1 *,* ⟨Lu, Lv⟩<sup>Q</sup> = Q<sup>n</sup> <sup>i</sup>=0 1 + r · α ui+v<sup>i</sup> *.*

Notice that our construction importantly ensures that the likelihood ratio inner product ⟨Lu, Lv⟩ is not solely a function of ⟨u, v⟩, but instead has a more intricate dependence on u and v. It is exactly this reason that leads to the discrepancy between GFP and FP hardness stated below.

Theorem 4. *There exist a two-point prior* π *on* u *such that, for* r = n −1/2 *,* α = n −1+2ε *,* m = n 1−ε *and* D = n ε *, where* ε > 0 *is any small constant, the following hold. The* m*-sample hypothesis testing problem* Eu∼πP ⊗m u *versus* Q⊗<sup>m</sup> *is* (e D/2 , m, Θ(n −ε ))*-GFP hard but not* (n −1 , m, exp (Θ(n ε )))*-FP hard. Moreover, via our equivalence theorem the model is* (e n Θ(ε) , n<sup>1</sup>−Θ(ε) )*-SQ hard.*

The proof of this Theorem and the above Lemma can be found in Appendix [E.](#page-44-0)

### 6 Conclusion

In this work, we generalize the Franz-Parisi (FP) criterion introduced by [\[6\]](#page-10-0), motivated by the observation that the Euclidean dot product may not be the most natural geometry for all statistical task—a point partially illustrated by our example in Section [5.](#page-9-0) Our main result shows that optimizing the overlap event in the FP definition of [\[6\]](#page-10-0) leads to a Generalized Franz-Parisi (GFP) hardness criterion, which is equivalent to SQ-hardness for models satisfying the mild Assumption [1.](#page-5-1) This assumption holds in a broad range of well-studied problems, including Gaussian additive models, planted sparse models, single-index models, and convex truncation. Our work significantly strengthens the theoretical foundation behind the (annealed) FP potential's predictions from statistical physics, but also opens several questions:

- 1. *(Algorithmic implications)* Does the optimal overlap function ρG(u, v)—as characterized in Theorem [2—](#page-6-0)yield meaningful algorithmic insights, particularly for local search or geometric methods?
- 2. *(The annealed potential)* Can similar equivalences be established for the original (also known as quenched) FP potential, or is the choice of the annealed version fundamental?
- 3. *(Interpretation of FP Area)* Why does the area under the FP curve appear to govern detection hardness? Is there some physical/algorithmic interpretation of this phenomenon?
- 4. *(Generalization to estimation)* Can our techniques be extended from detection to estimation tasks, for which the Franz-Parisi potential was originally introduced?

We believe these questions point toward promising future directions, with the potential to unify different approaches on the computational complexity of statistical inference.

## References


[1] Matthew Aldridge, Oliver Johnson, Jonathan Scarlett, et al. Group testing: an information theory perspective. *Foundations and Trends® in Communications and Information Theory*, 15(3-4):196–392, 2019. [2] Arash A Amini and Martin J Wainwright. High-dimensional analysis of semidefinite relaxations for sparse principal components. In *2008 IEEE international symposium on information theory*, pages 2454–2458. IEEE, 2008. [3] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Algorithmic thresholds for tensor pca. *The Annals of Probability*, 48(4):2052–2087, 2020. [4] Gérard Ben Arous, Alexander S Wein, and Ilias Zadik. Free energy wells and overlap gap property in sparse pca. *Communications on Pure and Applied Mathematics*, 76(10):2410–2473, 2023. [5] Gabriel Arpino and Ramji Venkataramanan. Statistical-computational tradeoffs in mixed sparse linear regression. In *The Thirty Sixth Annual Conference on Learning Theory*, pages 921–986. PMLR, 2023. [6] Afonso S Bandeira, Ahmed El Alaoui, Samuel Hopkins, Tselil Schramm, Alexander S Wein, and Ilias Zadik. The franz-parisi criterion and computational trade-offs in high dimensional statistics. *Advances in Neural Information Processing Systems*, 35:33831–33844, 2022. [7] Matthew Brennan, Guy Bresler, and Wasim Huleihel. Universality of computational lower bounds for submatrix detection. In *Conference on Learning Theory*, pages 417–468. PMLR, 2019. [8] Matthew S Brennan, Guy Bresler, Sam Hopkins, Jerry Li, and Tselil Schramm. Statistical query algorithms and low degree tests are almost equivalent. In *Conference on Learning Theory*, pages 774–774. PMLR, 2021. [9] Siyu Chen, Beining Wu, Miao Lu, Zhuoran Yang, and Tianhao Wang. Can neural networks achieve optimal computational-statistical tradeoff? an analysis on single-index model. In *The Thirteenth International Conference on Learning Representations*. [10] Zongchen Chen, Conor Sheehan, and Ilias Zadik. On the low-temperature mcmc threshold: the cases of sparse tensor pca, sparse regression, and a geometric rule. *arXiv preprint arXiv:2408.00746*, 2024. [11] Amin Coja-Oghlan, Oliver Gebhard, Max Hahn-Klimroth, Alexander S Wein, and Ilias Zadik. Statistical and computational phase transitions in group testing. In *Conference on Learning Theory*, pages 4764–4781. PMLR, 2022. [12] Alex Damian, Loucas Pillaud-Vivien, Jason Lee, and Joan Bruna. Computational-statistical gaps in gaussian single-index models. In *The Thirty Seventh Annual Conference on Learning Theory*, pages 1262–1262. PMLR, 2024. [13] Constantinos Daskalakis, Themis Gouleakis, Chistos Tzamos, and Manolis Zampetakis. Efficient statistics, in high dimensions, from truncated samples. In *2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS)*, pages 639–649. IEEE, 2018. [14] Constantinos Daskalakis, Themis Gouleakis, Christos Tzamos, and Manolis Zampetakis. Computationally and statistically efficient truncated regression. In *Conference on learning theory*, pages 955–960. PMLR, 2019. [15] Anindya De, Shivam Nadimpalli, and Rocco A Servedio. Testing convex truncation. In *Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pages 4050–4082. SIAM, 2023.

[16] Ilias Diakonikolas, Daniel M Kane, and Alistair Stewart. Statistical query lower bounds for robust estimation of high-dimensional gaussians and gaussian mixtures. In *2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS)*, pages 73–84. IEEE, 2017. [17] Ilias Diakonikolas, Weihao Kong, and Alistair Stewart. Efficient algorithms and lower bounds for robust linear regression. In *Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms*, pages 2745–2754. SIAM, 2019. [18] Yunzi Ding, Dmitriy Kunisky, Alexander S Wein, and Afonso S Bandeira. Subexponential-time algorithms for sparse pca. *Foundations of Computational Mathematics*, 24(3):865–914, 2024. [19] Jianqing Fan, Han Liu, Zhaoran Wang, and Zhuoran Yang. Curse of heterogeneity: Computational barriers in sparse mixture models and phase retrieval. *arXiv preprint arXiv:1808.06996*, 2018. [20] Vitaly Feldman, Elena Grigorescu, Lev Reyzin, Santosh S Vempala, and Ying Xiao. Statistical algorithms and a lower bound for detecting planted cliques. *Journal of the ACM (JACM)*, 64(2):1–37, 2017. [21] Silvio Franz and Giorgio Parisi. Recipes for metastable states in spin glasses. *Journal de Physique I*, 5(11):1401–1415, 1995. [22] Silvio Franz and Giorgio Parisi. Effective potential in glassy systems: theory and simulations. *Physica A: Statistical Mechanics and its Applications*, 261(3-4):317–339, 1998. [23] Francis Galton. An examination into the registered speeds of american trotting horses, with remarks on their value as hereditary data. *Proceedings of the Royal Society of London*, 62(379- 387):310–315, 1898. [24] David Gamarnik and Ilias Zadik. Sparse high-dimensional linear regression. estimating squared error and a phase transition. *The Annals of Statistics*, 50(2):880–903, 2022. [25] S Das Gupta, Morris L Eaton, Ingram Olkin, Michael Perlman, Leonard J Savage, and Milton Sobel. Inequalities on the probability content of convex regions for elliptically contoured distributions. In *Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability (Univ. California, Berkeley, Calif., 1970/1971)*, volume 2, pages 241–265, 1972. [26] Samuel Hopkins. *Statistical Inference and the Sum of Squares Method*. PhD thesis, Cornell University, 2018. [27] Daniel J Hsu, Clayton H Sanford, Rocco Servedio, and Emmanouil Vasileios Vlatakis-Gkaragkounis. Near-optimal statistical query lower bounds for agnostically learning intersections of halfspaces with gaussian marginals. In *Conference on Learning Theory*, pages 283–312. PMLR, 2022. [28] Hidehiko Ichimura. Semiparametric least squares (sls) and weighted sls estimation of singleindex models. *Journal of econometrics*, 58(1-2):71–120, 1993. [29] Mark Jerrum. Large cliques elude the metropolis process. *Random Structures & Algorithms*, 3(4):347–359, 1992. [30] Michael Kearns. Efficient noise-tolerant learning from statistical queries. *Journal of the ACM (JACM)*, 45(6):983–1006, 1998. [31] Dmitriy Kunisky, Alexander S Wein, and Afonso S Bandeira. Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio. In *ISAAC Congress (International Society for Analysis, its Applications and Computation)*, pages 1–50. Springer, 2019.

[32] Rafał Latała and Dariusz Matlak. Royen's proof of the gaussian correlation inequality. In *Geometric Aspects of Functional Analysis: Israel Seminar (GAFA) 2014–2016*, pages 265–275. Springer, 2017. [33] Peter McCullagh. *Generalized linear models*. Routledge, 2019. [34] Andrea Montanari, Daniel Reichman, and Ofer Zeitouni. On the limitation of spectral methods: From the gaussian hidden clique problem to rank-one perturbations of gaussian tensors. *Advances in Neural Information Processing Systems*, 28, 2015. [35] Karl Pearson. On the systematic fitting of curves to observations and measurements. *Biometrika*, 1(3):265–303, 1902. [36] Emile Richard and Andrea Montanari. A statistical model for tensor pca. *Advances in neural information processing systems*, 27, 2014. [37] Thomas Royen. A simple proof of the gaussian correlation conjecture extended to multivariate gamma distributions. *arXiv preprint arXiv:1408.1028*, 2014. [38] Tselil Schramm and Alexander S Wein. Computational barriers to estimation from low-degree polynomials. *The Annals of Statistics*, 50(3):1833–1858, 2022. [39] Ilias Zadik, Min Jae Song, Alexander S Wein, and Joan Bruna. Lattice-based methods surpass sum-of-squares in clustering. In *Conference on Learning Theory*, pages 1247–1248. PMLR, 2022. [40] Lenka Zdeborová and Florent Krzakala. Statistical physics of inference: Thresholds and algorithms. *Advances in Physics*, 65(5):453–552, 2016.
## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

- Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, the abstract and introduction make accurate claims about the paper's contributions and scope. We describe our results accurately.

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss sufficiently the limitations of our results and provide clear assumptions under which they apply.

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

Answer: [Yes]

Justification: We write clearly all the assumptions that are required for our results to hold. The proofs are all provided in the appendices.

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: There are no experimental results, as this is a theory paper.

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

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: There are no experimental results, as this is a theory paper.

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: There are no experimental results, as this is a theory paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: There are no experimental results, as this is a theory paper.

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: There are no experimental results, as this is a theory paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: We have conducted research and presented our work in accordance with the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There are no direct societal impact of our work that we are aware of.

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: There are no such risks that we are aware of.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, we cite all the relevant works to be acknowledged to establish our results. Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets>

has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: As this is a theory paper with no such crowdsourcing involved on human related subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: As this is a theory paper with no such crowdsourcing involved on human related subjects.

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper does not involve the use of LLMs for our scientific methodologies. Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.

## Contents

| 1             | Introduction |                                               | 1                      |
|---------------|--------------|-----------------------------------------------|------------------------|
| 1.1           | Main         | Contributions                                 | 3                      |
| 2 Setting     | and          | Definitions                                   | 5                      |
| 3 Main        | Results      |                                               | 6                      |
| 3.1           | The          | Assumption                                    | 6                      |
| 3.2           | The          | GFP-SQ equivalence                            | 6                      |
|               | 3.2.1        | Simplifying GFP-hardness                      | 6                      |
|               | 3.2.2        | The equivalence                               | 7                      |
| 4 Examples    |              |                                               | 8                      |
| 4.1           | Gaussian     | Additive Models                               | 8                      |
| 4.2           | Planted      | Sparse Models                                 | 9                      |
|               | 4.2.1        | Other examples                                | 9                      |
| 5             | GFP-hardness | is not always equal to FP-hardness            | 10                     |
| 6 Conclusion  |              |                                               | 10                     |
| A Discussions |              | on FP criterion and assumptions               | 23                     |
| A.1           | Connection   | of the FP criterion with statistical          | physics 23             |
| A.2           | Necessity    | of assumptions in main theorem                | 24                     |
|               | A.2.1        | Necessacity of Assumption 1                   | 24                     |
|               | A.2.2        | Necessity of a non-trivial information-theory | threshold 25           |
| B             | Equivalence  | between LD, SQ, and GFP                       | 27                     |
| B.1           | Low-Degree   | lower bounds definitions                      | 27                     |
|               | B.1.1        | Low-Degree Lower Bounds                       | 27                     |
|               | B.1.2        | Low Samplewise Degree Lower Bounds            | 27                     |
| B.2           |              | Unconditional SQ hardness                     | 28                     |
| B.3           | Noise-robust | models and SQ-LD equivalence                  | 29                     |
| B.4           | Equivalence  | of GFP, SQ, and LD hardness for               | noise-robust models 30 |
| C Proofs      | of           | main theorems                                 | 32                     |
| C.1           | Proof        | of Theorem 2                                  | 32                     |
| C.2           | Proof        | of Theorem 3                                  | 34                     |
| D Details     | of           | examples and proofs                           | 36                     |

| D.1          | Symmetric    | mixed  | sparse       | linear       | regression             | 36 |
|--------------|--------------|--------|--------------|--------------|------------------------|----|
| D.1.1        |              | Proofs | for          | mSLR         |                        | 37 |
| D.2          | Non-Gaussian |        | component    |              | analysis               | 39 |
| D.2.1        |              | Proofs | for          | Non-Gaussian | Component Analysis     | 40 |
| D.3          | Single-index |        | Models       |              |                        | 42 |
| D.4          | Truncated    |        | statistics:  | convex       | truncation             | 42 |
| D.4.1        | A            | new    | SQ lower     | bound        |                        | 43 |
| D.4.2        |              | Proofs | for          | Convex       | Truncation             | 43 |
| E Details on | the          |        | GFP-hardness | and          | FP-hardness separation | 45 |

## A Discussions on FP criterion and assumptions

In this Section, we provide additional discussions on the Franz-Parisi criterion and its connection with Statistical physics, as well as, on the necessity of our assumptions in our main theorem (Theorem [3\)](#page-6-1).

## A.1 Connection of the FP criterion with statistical physics

We begin by discussing the connection between the Franz-Parisi (FP) criterion and statistical physics methods. For a more detailed overview and additional references, we refer the reader to [\[6,](#page-10-0) Section 1.3].

A natural algorithm for solving the estimation problem of recovering u from Y = (Y1, . . . , Ym) ∼ <sup>P</sup><sup>u</sup> is to run some "local" dynamics (e.g., Langevin or Glauber dynamics) to sample from the posterior

$$\mathbb{P}(u|Y) \propto \pi(v)\mathbb{P}(Y|v) = \pi(v) \prod_{i=1}^m \mathbb{P}_v(Y_i), v \in \Theta,$$

where Y = (Y1, . . . , Ym). In statistical physics, a powerful heuristic exists for predicting the success of local dynamics in sampling from random distributions of the form p<sup>Y</sup> (v)ν(v), v ∈ Θ where ν is a reference measure and Y ∼ µ is a "disorder". The heuristic approach is to check the monotonicity of the so-called Franz-Parisi potential defined as

$$F(t) = \mathbb{E}_{u \sim \nu, Y \sim \mu} [\log \mathbb{E}_{v \sim \nu} [p_Y(v) \mathbf{1}(d(v, u) = t)]], t \in [0, 1],$$

where d(·, ·) is some notion of (normalized) distance between the states u, v in agreement with the operations of the local dynamics on the state space. The prediction, introduced by Franz and Parisi in [\[21\]](#page-11-3), is that local dynamics can efficiently sample from the distribution if and only if the potential is monotonic, i.e., it lacks "bad" local minima. Remarkably, this prediction has been empirically validated across a range of problems in statistical physics, often yielding accurate forecasts of algorithmic tractability. For instance, when d is the Euclidean distance, this criterion has proven effective in the study of spin glasses [\[21\]](#page-11-3). Other, more intricate distance functions have also been used successfully in non-spin glass settings, such as binary fluids [\[22\]](#page-11-5).

Now, returning to statistical estimation settings, researchers in statistical physics have applied this rule for <sup>p</sup><sup>Y</sup> (v) := <sup>P</sup>(<sup>Y</sup> |v) = Q<sup>m</sup> <sup>i</sup>=1 <sup>P</sup>v(Yi) and ν := π to arrive at a prediction of success for "local" algorithms based on the geometry defined by the distance d. The prediction [\[40\]](#page-12-0) is then based on the monotonicity of the curve

$$F(t) = \mathbb{E}_{u \sim \pi, Y \sim \mathbb{P}_u} [\log \mathbb{E}_{v \sim \pi} (\mathbb{P}(Y|v)1 [d(v, u) = t])], t \in [0, 1],$$

or equivalently for

$$F(t) = \mathbb{E}_{u \sim \pi, Y \sim \mathbb{P}_u} \left[ \log \mathbb{E}_{v \sim \pi} \left( \prod_{i=1}^m \frac{\mathbb{P}_v(Y_i)}{\mathbb{Q}(Y_i)} 1(d(v, u) = t) \right) \right], t \in [0, 1], \quad (10)$$

Interestingly, when d(·, ·) is the Euclidean distance, recent mathematical works have indeed produced one-sided results linking the potential to the performance of local methods for estimation tasks in the context of the so-called Gaussian additive models (e.g., [\[3,](#page-10-11) [4,](#page-10-7) [6\]](#page-10-0)). This connection with the choice of the Euclidean distance can be perhaps cast as natural by a well-known analogy between spin glasses and GAMs, where GAMs often take the form of "spiked" spin glass models. Now, given the above successes, both heuristic and rigorous, it is natural to conjecture a potential link between general algorithmic hardness and the monotonicity of F(t). However, this connection remains unproven in general, and known counterexamples exist. For instance, in sparse tensor PCA [\[10\]](#page-10-8), there are regimes where the FP potential is non-monotonic (suggesting hardness), but some polynomial-time methods do succeed.

Despite the above issue, [\[6\]](#page-10-0) used the Franz-Parisi potential to arrive at a different criterion, but now for algorithmic hardness of detection. Following an application of Jensen's inequality described in [\[6,](#page-10-0) Section 1.3] one get the following "annealed" upped bound for any t ∈ [0, 1] F(t) ≤ log F˜(t) for,

$$\tilde{F}(t) = \mathbb{E}_{u,v \sim p}, [\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle 1 [d(v, u) = t]], t \in [0, 1]. \quad (11)$$

Then by focusing on the Euclidean distance d (or equivalently the Euclidean dot product ⟨u, v⟩) they suggested the Franz-Parisi (FP) criterion Definition [1,](#page-1-3) restated here.

Definition 5 (FP hardness). *We say a problem is* (q, m, ε)*-FP hard if*

$$\mathbf{FP}: \quad \mathbb{E} \left[ \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(|\langle u, v \rangle| \leq \delta(q)) \right] \leq 1 + \varepsilon, \quad \text{where} \quad (12)$$

$$\delta(q) = \sup\{\delta : \delta^2(\langle u, v \rangle \geq \delta) \geq q^{-2}\}, \quad (13)$$

Notice that the FP criterion says that to check for "hardness" of detection one should integrate the annealed FP curve is an (1 − q −2 )-typical overlap t-region. Moreover, as we elaborated in the Introduction, one should understand q in the above definition as a proxy for q run-time. In that light, [\[6\]](#page-10-0) roughly proved that for any GAMs is a (q, m, O(1))-FP hard if and only if D = log q-degree polynomials fail to detect between P and Q with m samples. We remark that, albeit this is an equivalence for detection, this is a first-of-a-kind result for GAMs as it is mathematical connection between the FP curve and a rigorous form of hardness. However, [\[6\]](#page-10-0) also presented counterexamples where this equivalence breaks down when we move away from GAMs.

The central idea of this work is to optimize over the integration region in the FP criterion, rather than fixating on the Euclidean dot product. This leads us to propose the Generalized Franz-Parisi (GFP) criterion (see Definition [2\)](#page-2-3). Our motivation arises from the observation that while the Euclidean distance is natural for GAMs (and spin glass models), it may be inappropriate in other statistical settings (see Section [5\)](#page-9-0). This echoes insights from statistical physics, where non-Euclidean distances are used in models beyond spin glasses [\[22\]](#page-11-5). Satisfyingly, this generalization enables a broad equivalence with statistical query (SQ) lower bounds, as shown in Theorem [3.](#page-6-1)

#### A.2 Necessity of assumptions in main theorem

In this section, we comment on the necessity of our assumptions for the GFP-hardness and SQhardness equivalence.

#### A.2.1 Necessacity of Assumption [1](#page-5-1)

We first show that there is a strong separation between GFP-hardness and SQ-hardness unless some non trivial lower bound on minu,v⟨Lu, Lv⟩<sup>Q</sup> is assumed, providing support for our Assumption [1.](#page-5-1) Indeed, we present here a (very) simple counterexample when one allows for minu,v⟨Lu, Lv⟩<sup>Q</sup> = 0.

Let us define a variant of the following model from [\[6,](#page-10-0) Section 4.2.] (assume for simplicity n is a multiple of 10 in what follows):

- Under <sup>P</sup>, we first sample a u ∼ Unif({x ∈ {−1, 1} <sup>n</sup> : Px<sup>i</sup> = 8n/10}). Then under <sup>P</sup><sup>u</sup> each sample always equals to u, i.e., <sup>P</sup><sup>u</sup> is the Dirac measure on u.

- Under Q for each sample, we sample u ∼ Unif({−1, 1}
- <sup>n</sup>) and output u.

It is easy to see that for all u, v ∈ {x ∈ {−1, 1} <sup>n</sup> : Px<sup>i</sup> = 8n/10},

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = 2^n \mathbf{1}(u = v).$$

We first prove that the task is *not even* (1, 1)*-SQ hard.* Indeed (1, 1)-SQ hard implies <sup>E</sup>[|⟨Lu, Lv⟩<sup>Q</sup> − 1|] ≤ 1, but

$$\mathbb{E}[|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|] \geq |\langle L_u, L_u \rangle_{\mathbb{Q}} - 1|\pi^2(u = v) \quad (14)$$

$$= (2^n - 1) \binom{n}{9n/10}^{-1} = \omega(1). \quad (15)$$

On the contrary, we now show that the task is (e n Θ(1) , ∞, O(1))*-GFP-hard,* more specifically we prove that it is (m, q, O(1))-GFP-hard for any sample size m and q = <sup>n</sup> 9n/10<sup>1</sup>/<sup>2</sup> = e n Θ(1) . Indeed, we have π 2 (u ̸= v) = 1 − q −2 and therefore to prove (m, q)-GFP hardness it suffices

$$\mathbb{E}[\langle L_u, L_v \rangle_{\mathbb{Q}}^m \mathbf{1}(u \neq v)] = O(1).$$

But in fact it even holds <sup>E</sup>[⟨Lu, Lv⟩ m <sup>Q</sup> 1(u ̸= v)] = 0 completing this proof.

#### A.2.2 Necessity of a non-trivial information-theory threshold

The second assumption that our equivalence operates under is that mIT is non-trivial. We now claim that some non-trivial lower bound on mIT is also necessary for the connection between the notions of GFP-hardness and SQ-hardness, even under Assumption [1.](#page-5-1)

Indeed, consider the following multisample problem over graphs. Let n ∈ N, p = 1 − n −1/4 and k = n 1/3+o(1) .

- Under <sup>P</sup> we choose a u being a k-clique in Kn, chosen uniformly at random (we see u as a k-vertex set). Then under <sup>P</sup><sup>u</sup> one sample consists of the union of a G(n, p) with the k-clique on u.
- Under Q one sample is a sample from G(n, p).

We note this is a multi-sample variant of the classic planted clique model [\[29\]](#page-11-9), but on the (very) dense regime, which has been recently used to establish circuit lower bounds for the model in [? ].

Now, even for m = 1 the problem is information-theoretically solvable; indeed, under P there is always a k-clique, while under Q there is no k-clique with probability 1 − o(1), and a brute force method can distinguish the two cases. Indeed, by a union bound the probability there is a k-clique under Q is at most

$$n^k(1 - n^{-1/4})^{\binom{k}{2}} \leq \exp\left(n^{1/3} \log n - \Theta(n^{2/3-1/4})\right) = \exp\left(-\Theta(n^{5/12})\right) = o(1).$$

Moreover, the model satisfies Assumption [1.](#page-5-1) One can see this because the model is a PSM and use Lemma [2.](#page-8-2) Alternatively, one can just directly observe that for any u, we have Lu(G) = 1(u is a k-clique in G)p <sup>−</sup>( k 2) . Hence, for any u, v

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = p^{-\binom{|u \cap v|}{2}} = (1 - n^{-1/4})^{-\binom{|u \cap v|}{2}} \geq 1.$$

Now, we prove that this PSM is not SQ-hard even for m = 1 and q = 1, but it is GFP-hard even for m = n Θ(1)-samples.

We first prove that the task *is not* (1, 1)*-SQ-hard*. Notice that (1, 1)-SQ hardness is equivalent with the condition <sup>E</sup>[|⟨Lu, Lv⟩<sup>Q</sup> − 1|] ≤ 1. But in this context

$$\mathbb{E}[|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|] \geq |\langle L_u, L_u \rangle_{\mathbb{Q}} - 1|\pi^2(u = v)$$

$$\begin{aligned} &= (1 - n^{-1/4}) \Theta(k^2) \binom{n}{k} \\ &= \exp \left( \Theta(k^2 n^{-1/4}) - \Theta(k \log n) \right) \\ &= \exp \left( \Theta(n^{5/12}) \right) = \omega(1). \end{aligned}$$

On the contrary, the task is (e n Θ(1) , n<sup>1</sup>/<sup>8</sup> , O(1))*-GFP-hard.* Fix m samples. Notice that for i.i.d. u, v ∼ π, the overlap |u ∩ v| follows an (n, k, k)-Hypergeometric. Hence by [\[6,](#page-10-0) Lemma 6.6.] for any q > 0 if δ = log(k 2 q) > 0 it holds

$$\pi^2(|u \cap v| \geq \delta) \leq k(k^2/n)^\delta \leq k2^{-\delta} = q^{-2}.$$

Hence, to prove GFP-hardness it suffices to prove that <sup>E</sup>[⟨Lu, Lv⟩ m <sup>Q</sup> 1(|u ∩ v| ≤ δ)] = O(1). Therefore for q = exp (n <sup>α</sup>) for some α > 0, and m = n 1/8+o(1) ,

$$\begin{aligned} \mathbb{E}[\langle L_u, L_v \rangle_{\mathbb{Q}}^m \mathbf{1}(|u \cap v| \leq \delta)] &\leq \mathbb{E}[(1 - n^{-1/4})^{-m(\lfloor u \cap v \rfloor)} \mathbf{1}(|u \cap v| \leq \delta)] \\ &\leq (1 - n^{-1/4})^{-m(\frac{\delta}{2})} \\ &= (1 - n^{-1/4})^{-\Theta(m(\log(kq^2))^2)} \\ &\leq \exp\left(\Theta(mn^{-1/4}(\log(kq^2))^2)\right) \\ &= \exp\left(n^{-1/8+2\alpha}\right) = O(1), \end{aligned}$$

where the last equality hold say for any 0 < α < 1/16. Hence, choosing say q = e n <sup>1</sup>/<sup>32</sup> concludes the proof.

## B Equivalence between LD, SQ, and GFP

In this Appendix, we discuss the equivalence between GFP and low-degree (LD) polynomial hardness. This result is obtained by combining the GFP-SQ equivalence from Section [3](#page-5-5) with the equivalence between SQ and LD hardness under noise robustness proved by Brennan et al. [\[8\]](#page-10-1). We recall and provide a succinct proof of Brennan et al.'s result for completeness.

#### B.1 Low-Degree lower bounds definitions

We start by recalling the definition of a low-degree lower bound. The definition is based on the *low-degree likelihood ratio* L <sup>≤</sup><sup>D</sup>, where we recall that L <sup>≤</sup><sup>D</sup> denotes the projection of the likelihood ratio onto the subspace of degree-at-most-D polynomials.

#### B.1.1 Low-Degree Lower Bounds

The following is the standard definition of Low-Degree hardness as originally stated, for example, in [\[26\]](#page-11-2).

Definition 6 (Low-Degree Likelihood Ratio). *For* m *samples, define the squared norm of the degree-*D *likelihood ratio (also called the "low-degree likelihood ratio") to be the quantity*

$$\text{LD}(D) := \|L_m^{\leq D}\|_{\mathbb{Q}}^2 = \left\| (\mathbb{E}_{u \sim \pi} L_u^{\otimes m})^{\leq D} \right\|_{\mathbb{Q}}^2 = \mathbb{E}_{u, v \sim \pi} [(L_u^{\otimes m})^{\leq D}, (L_v^{\otimes m})^{\leq D}]_{\mathbb{Q}}. \quad (16)$$

*For some increasing sequence* D = Dn*, we say that the hypothesis testing problem above is* hard for the degree-D likelihood *or simply* D-degree hard *if* LD(D) = O(1)*.*

While we direct the reader to [\[6,](#page-10-0) Section 1.2] a relation between the Low-degree likelihood ratio and the performance of low-degree algorithms we highlight some key conjectures in the community.

- We expect the class of degree-<sup>D</sup> polynomials to be as powerful as all exp Θ( ˜ D) -time tests (which is the runtime needed to naively evaluate the polynomial term-by-term). Thus, if LD(D) = O(1) (or 1 + o(1)), we take this as evidence that strong (or weak, respectively) detection requires runtime e Ω( ˜ D) ; see Hypothesis 2.1.5 of [\[26\]](#page-11-2).
- On a finer scale, we expect the class of degree-O(log n) polynomials to be at least as powerful as all polynomial-time tests. Thus, if LD(D) = O(1) (or 1 + o(1)) for some D = ω(log n), we take this as evidence that strong (or weak, respectively) detection cannot be achieved in polynomial time; see Conjecture 2.2.4 of [\[26\]](#page-11-2).

We emphasize that the above statements are not true in general (see for instance [\[39\]](#page-12-1) for some discussion of counterexamples) and depend on the choice of P and Q, yet remarkably often appear to hold up for a broad class of distributions arising in high-dimensional statistics.

#### B.1.2 Low Samplewise Degree Lower Bounds

In multisample settings like ours, a similar notion of "samplewise" low degree lower bounds have been considered in [\[8\]](#page-10-1).

Definition 7. *For* d, k ∈ N ∪ {∞} *a function* f : (<sup>R</sup> n) <sup>⊗</sup><sup>m</sup> → <sup>R</sup> *has samplewise degree* (d, k) *if it can be written as a linear combination of functions which have degree at most* d *in each* x<sup>i</sup> *and non-zero degree in at most* k *of the* xi*'s (if* d < ∞ *the function is therefore a polynomial).*

Let's state the hardness criterion associated with this samplewise low degree polynomials:

Definition 8 (Low Degree (LD) Hardness). *We say a "*<sup>P</sup> *versus* Q*" detection problem is* (m, d, k, ε)*- LD hard if*

$$LD: \quad \mathbb{E} \left[ \langle (L_u^{\otimes m})^{\leq d,k}, (L_v^{\otimes m})^{\leq d,k} \rangle \right] \leq 1 + \varepsilon. \quad (17)$$

Notice that this notion of (d, k)-low degree hardness is the natural generalization to [\(16\)](#page-26-4). As a point of comparison, dk-degree polynomials contain all (d, k)-degree polynomials and (d, d)-degree polynomials contain all d-degree polynomial.

Remark B.1 (Explaining Remark [3.3\)](#page-6-3). A nice property of the low samplewise-degree degree projection is that it is easy to relate it to d-degree projections. Indeed, using a binomial expansion argument (see [\[8,](#page-10-1) Claim 3.3.]),

$$\|L_m^{\leq(d,k)}\|_{\mathbb{Q}}^2 = \mathbb{E}_{u,v \sim \pi} \left[ (\langle L_u^{\otimes m} \rangle^{\leq(d,k)}, (L_v^{\otimes m})^{\leq(d,k)} \rangle_{\mathbb{Q}} \right] = \sum_{t=0}^m \binom{m}{t} \mathbb{E}_{u,v \sim \pi} \left[ (\langle L_u^{\leq d}, L_v^{\leq d} \rangle_{\mathbb{Q}} - 1)^t \right].$$

In particular, if k = 1, d = ∞, since <sup>E</sup>u,v∼<sup>π</sup> [⟨Lu, Lv⟩ − 1] = χ 2 (P, Q) we have

$$\|L_m^{\leq (\infty,1)}\|_{\mathbb{Q}}^2 = 1 + m\chi^2(\mathbb{P}, \mathbb{Q}).$$

In particular, notice that the condition mχ<sup>2</sup> (P, Q) = O(1) discussed in Theorem [2](#page-6-0) and Remark [3.3](#page-6-3) is equivalent with a samplewise (∞, 1)-degree lower bound for the task, i.e., a lower bound against function that are linear combination of functions of one sample at a time. In [\[8\]](#page-10-1) the authors prove that SQ lower bounds are (almost) equivalent with sample-wise degree lower bounds, therefore it is perhaps no surprise that the condition mχ<sup>2</sup> (P, Q) = O(1) can be also explained as a (very) weak consequence of any SQ lower bounds against m samples. Indeed, assume a P versus Q detection problem is (q, m)-SQ hard for any q (even q = 1). Then setting A = support(π) <sup>⊗</sup><sup>2</sup> we have that it must hold mEu,v∼<sup>π</sup> [|⟨Lu, Lv⟩<sup>Q</sup> − 1|] ≤ 1 and therefore

$$m\chi^2(\mathbb{P}, \mathbb{Q}) = m\mathbb{E}_{u,v \sim \pi} [\langle L_u, L_v \rangle_{\mathbb{Q}} - 1] \leq 1.$$

#### B.2 Unconditional SQ hardness

Before stating the equivalence between the above LD-hardness criterion and SQ-hardness, we define an *Unconditional Statistical Query* (USQ) hardness criterion, which is equivalent to SQ and often appears as a convenient intermediate step in proofs. This hardness measure appeared, often implicitly, in several prior work (e.g., [\[8\]](#page-10-1)):

Definition 9 (Unconditional SQ hardness). *We say a "*<sup>P</sup> *versus* Q*" detection problem is* (m, t) *unconditional SQ hard for some even* t *if*

$$USQ: \mathbb{E} [\chi_{\mathbb{Q}}(\mathbb{P}_u, \mathbb{P}_v)^t] \leq m^{-t}. \quad (18)$$

The USQ criterion removes the conditioning on event A from the SQ criterion, which makes it much easier to manipulate. USQ hardness is essentially equivalent to SQ hardness as stated in the next proposition:

Proposition 1 (Equivalence USQ and SQ hardness).

- *(i) If a model is* (m, t)*-USQ hard, then it is* (q, m/q<sup>2</sup>/t)*-SQ hard for all integers* q ≥ 1*.*
- *(ii) If a model is* (q, m/q<sup>2</sup>/t)*-SQ hard for all integers* q ≥ 1*, then it is* (m′ , t′ )*-USQ hard for all* t ′ < t *and* m′ ≤ m · 2 <sup>−</sup>1/t(t − t ′ ) 1/t′ *.*

For simplicity, for t ≥ 4, we can set t ′ = t/2 and m′ = m in Proposition [1.](#page-27-2)(ii). Proposition [1](#page-27-2) was proven in [\[8\]](#page-10-1). We provide a succinct proof for completeness.

*Proof of Proposition [1.](#page-27-2)* USQ hardness implies SQ hardness. By Hölder's inequality,

$$\begin{aligned}\mathbb{E} [|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1| \mid A] &\leq \frac{(\mathbb{E} [|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^t])^{1/t} \cdot (\mathbb{E} [1[(u, v) \in A]])^{1-1/t}}{\pi^2(A)} \\ &= \left( \frac{\mathbb{E} [|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^t]}{\pi^2(A)} \right)^{1/t}.\end{aligned}$$

Assuming that we have (m, t)-USQ hardness, this implies that for any q ≥ 1,

$$\sup_{A: \pi^2(A) \geq q^{-2}} \mathbb{E} [|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1| \mid A] \leq \frac{q^{2/t}}{m},$$

which establishes the (q, m/q<sup>2</sup>/t)-SQ hardness.

SQ hardness implies USQ hardness. For convenience, introduce the random variable X = |⟨Lu, Lv⟩<sup>Q</sup> − 1| with (u, v) ∼ π . Assume that we have (q, m/q<sup>2</sup>/t)-SQ hardness for all q ≥ 1. In particular, for all A, we have

$$\mathbb{E}[X|A] \leq \frac{1}{\pi^2(A)^{1/t}m}.$$

Using [\[8,](#page-10-1) Fact 4.3], we have for every t > t′ > 0,

$$\mathbb{E}[X^{t'}] \leq \left( 2 \sup_A \pi^2(A) \cdot \mathbb{E}[X|A]^t \right)^{t'/t} \cdot \frac{t}{t-t'} \leq \frac{2^{t'/t}}{m^{t'}} \cdot \frac{t}{t-t'},$$

which establishes (t ′ , m′ )-USQ hardness for any t ′ < t and m′ = m · 2 <sup>−</sup>1/t(t − t ′ ) 1/t′ .

### B.3 Noise-robust models and SQ-LD equivalence

An advantage of USQ is that it is directly related to Low Degree lower bounds: USQ hardness is equivalent to LD hardness with d = ∞, that is, with no degree-constraint on each sample in the projection.

Proposition 2 (Equivalence between USQ and LD hardness with d = ∞).

- *(i) If a model is* (m, ∞, k, ε)*-LD hard, then it is* (m′ , k)*-USQ hard with* m′ = m/(kε<sup>1</sup>/k)*.*
- *(ii) If a model is* (m, k)*-USQ hard, it is* (m, ∞, k, e − 1)*-LD hard. More generally, it will be* (m′ , ∞, k, em′/m)*-LD hard for all* m′ < m*.*

*Proof of Proposition [2.](#page-28-1)* We follow the proof in [\[8\]](#page-10-1). Assume that the model is (m, ∞, k, ε)-LD hard. Then

$$\|\mathbb{E}_u[(L_u^{\otimes m})^{\leq \infty, k}] - 1\|_{\mathbb{Q}}^2 = \sum_{s=1}^k \binom{m}{s} \mathbb{E}_{u,v}[\chi_{\mathbb{Q}}(\mathbb{P}_u, \mathbb{P}_v)^s] \leq \varepsilon,$$

and in particular, for k even,

$$\|\mathbb{E}_u[(L_u^{\otimes m})^{\leq \infty, k}] - 1\|_{\mathbb{Q}}^2 - \|\mathbb{E}_u[(L_u^{\otimes m})^{\leq \infty, k-1}] - 1\|_{\mathbb{Q}}^2 = \binom{m}{k} \mathbb{E}_{u,v}[\chi_{\mathbb{Q}}(\mathbb{P}_u, \mathbb{P}_v)^k] \leq \varepsilon.$$

This implies that <sup>E</sup>u,v[χQ(<sup>P</sup>u, <sup>P</sup>v) k ] ≤ ε/<sup>m</sup> k ≤ εk<sup>k</sup>/m<sup>k</sup> . On the other hand, (m, k)-USQ hardness implies that

$$\|\mathbb{E}_u[(L_u^{\otimes m})^\leq \infty, k] - 1\|_{\mathbb{Q}}^2 \leq \sum_{s=1}^k \frac{m^s}{s!} \mathbb{E}_{u,v}[\chi_{\mathbb{Q}}(L_u, L_v)^s] \leq (e-1).$$

More generally, we will have for m′ < m

$$\|\mathbb{E}_u[(L_u^{\otimes m'})^{\leq \infty, k}] - 1\|_{\mathbb{Q}}^2 \leq \sum_{s=1}^k \frac{(m')^s}{s!} \mathbb{E}_{u,v}[\chi_{\mathbb{Q}}(L_u, L_v)^s] \leq \sum_{s=1}^k \frac{1}{s!} \left(\frac{m'}{m}\right)^s \leq e \frac{m'}{m},$$

which concludes the proof.

Combining this equivalence of USQ and LD(d = ∞) with the equivalence between USQ and SQ in Proposition [1,](#page-27-2) we can directly state an (unconditional) equivalence between SQ and LD(d = ∞) hardness. In order to transfer this equivalence to LD with d < ∞, one can assume that the model with d = ∞ and d < ∞ are close to each other: this assumption is equivalent to being *noise-robust* (in some sense, see discussions in [\[8\]](#page-10-1)).

Assumption 2 (Noise robustness). *We say a "*<sup>P</sup> *versus* Q*" detection problem is* (d, k, δ)*-noise robust if*

$$\|\mathbb{E}_u[(L_u^{>d})^{\otimes k}]\|_{L^2(\mathbb{Q})}^2 \leq \delta. \quad (19)$$

Under this assumption, one can state the equivalence between LD and USQ:

Proposition 3 (Equivalence between LD and USQ Hardness).

- *(i) If the model is* (m, t)*-USQ hard, then the model is also* (m′ , d, k′ , em′/m)*-LD hard for all* m′ ≤ m*,* k ′ ≤ t *and* d ≥ 1*.*
- *(ii) If the model is* (m, d, k, ε)*-LD hard and we further assume that it is* (d, k, δ)*-noise robust, then the model is* (m′ , k)*-USQ hard with*

$$m' = \frac{m}{m\delta^{1/k} + k\varepsilon^{1/k}}.$$

*Proof of Proposition [3.](#page-29-2)* Part (i) is directly implied by Proposition [2.](#page-28-1)(ii). For part (ii), following the same argument as in the proof of Proposition [2.](#page-28-1)(i), we get

$$\mathbb{E}[|\langle L_u^{\leq d}, L_v^{\leq d} \rangle - 1|^k] \leq \varepsilon \frac{k^k}{m^k}.$$

Then using [\[8,](#page-10-1) Lemma 3.4], we obtain

$$\begin{aligned} \mathbb{E}[|\langle L_u, L_v \rangle - 1|^{k]^1/k} &\leq \mathbb{E}[|\langle L_u^{\leq d}, L_v^{\leq d} \rangle - 1|^{k]^1/k} + \mathbb{E}[|\langle L_u^{>d}, L_v^{>d} \rangle|^{k]^1/k}] \\ &\leq \varepsilon^{1/k} \frac{k}{m} + \delta^{1/k} = \frac{k\varepsilon^{1/k} + m\delta^{1/k}}{m}, \end{aligned}$$

which concludes the proof.

Then, the equivalence between LD and SQ hardness in [\[8\]](#page-10-1) is obtained by combining Proposition [3](#page-29-2) and Proposition [1.](#page-27-2) We state it below for completeness:

Theorem 5 (Equivalence between LD and SQ hardness).

- *(i) If the model is* (q, m/q<sup>2</sup>/t)*-SQ hard for all* q ≥ 1 *(with* t ≥ 4*, for simplicity), then it is* (m′ , d, k′ , em′/m)*-LD hard for all* m′ ≤ m*,* k ′ ≤ t/2*. and* d ≥ 1*.*
- *(ii) If the model is* (m, d, k, ε)*-LD hard and we further assume that it is* (d, k, δ)*-noise robust, then the model is* (q, m′/q<sup>2</sup>/t)*-SQ hard for all* q ≥ 1 *with*

$$m' = \frac{m}{m\delta^{1/k} + k\varepsilon^{1/k}}.$$

### B.4 Equivalence of GFP, SQ, and LD hardness for noise-robust models

Based on the SQ-LD equivalence stated in the previous section (Theorem [5\)](#page-29-3) and the equivalence between GFP and SQ (Theorem [3\)](#page-6-1), we can state an equivalence between GFP and LD hardness for noise-robust models.

Theorem 6 (LD and ρG-FP Equivalence). *Suppose a "*<sup>P</sup> *versus* Q*" task satisfies Assumption [1](#page-5-1) for a group* G.

- *(i) If the model is* (m, d, k, ε)*-LD hard and we further assume that it is* (d, k, δ)*-noise robust, then the model is* (q ′ , m′ , e|G| <sup>−</sup><sup>1</sup>mq ˜ <sup>2</sup>/t/m˜ )*-*ρG*-FP hard for any integers* q ≥ 1*,* q ′ <sup>≤</sup> q/√ 2*, and* m′ ≤ m/ ˜ 2*, with*

$$\tilde{m} = \frac{m}{m\delta^{1/k} + k\varepsilon^{1/k}}.$$

- *(ii) If a task is* (q, m, ε)*-*ρG*-FP hard for some* q, m *integers. Assume that there exists an* r = r(q) > 0 *such that* π 2 (ρG(u, v) < r) = 1 − q <sup>−</sup><sup>2</sup> *and* m *is even. Then, for all even integer* 4 ≤ t ≤ log(q)/ log(m)*, the model is also* (m′ , d, k′ , em′/m˜ )*-LD hard for all* m′ ≤ m˜ *and* k ′ ≤ t/2*, and* d ≥ 1*, where*

$$\tilde{m} = \frac{m}{t(1 + \varepsilon)^{1/t} + \chi^2(\mathbb{P}^{\otimes 4t} \parallel \mathbb{Q}^{\otimes 4t})}.$$

Note that the implication of GFP hardness to LD hardness is unconditional. LD hardness with d = ∞ implies GFP hardness, while for d < ∞, this implication holds under the noise robustness assumption.

### C Proofs of main theorems

#### C.1 Proof of Theorem [2](#page-6-0)

*Proof of Theorem [2.](#page-6-0)* It is clear that (q, m, ε)-ρG-FP hard implies (q, m, ε)-GFP<sup>G</sup> hard as for the event

$$A := \{\rho_G(u, v) < r(q)\},$$

it clearly holds π ⊗2 (A) ≥ 1 − q −2 and, since G is a group, G⊗<sup>2</sup> (A) = A. Hence,

$$\inf_{\substack{A: \pi \otimes 2(A) \geq 1 - q^{-2} \\ G \otimes 2(A) = A}} \mathbb{E} \left[ \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \mathbf{1}(A) \right] \leq \mathbb{E} \left[ \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \leq 1 + \varepsilon,$$

implying the desired result.

We now focus on the other direction. By decomposing the likelihood ratio inner product, we obtain

$$\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} = (\langle L_u, L_v \rangle_{\mathbb{Q}} - 1 + 1)^m = \sum_{t=0}^m \binom{m}{t} \cdot (\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t. \quad (20)$$

Taking expectation over the prior π ⊗2 conditioned on *any* event A satisfying G⊗<sup>2</sup> (A) = A and π 2 (A) = 1 − q −2 , we have

$$\begin{aligned}\mathbb{E}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} | A] &= \sum_{t=0}^m \binom{m}{t} \cdot \mathbb{E}[\langle \langle L_u, L_v \rangle_{\mathbb{Q}} - 1 \rangle^t | A] \\ &= \sum_{t=1}^m \binom{m}{t} \cdot \left( \mathbb{E}_{g \sim \text{Unif}(G)} \mathbb{E}[\langle \langle L_{g(u)}, L_{g(v)} \rangle_{\mathbb{Q}} - 1 \rangle^t | A] \right) + 1 \\ &\geq \sum_{t=1}^{\lfloor m/2 \rfloor} \binom{m}{2t} \cdot \left( \mathbb{E}[\mathbb{E}_{g \sim \text{Unif}(G)} \langle \langle L_{g(u)}, L_{g(v)} \rangle_{\mathbb{Q}} - 1 \rangle^{2t} | A] \right) + 1.\end{aligned}$$

where in the second equality, we used that G is a π-preserving transformation, and for the inequality we use Assumption [1.](#page-5-1)

Clearly for all t ≥ 0,

$$\mathbb{E}_{g \sim \text{Unif}(G)} \left( \langle L_{g(u)}, L_{g(v)} \rangle - 1 \right)_{\mathbb{Q}}^{2t} \geq |G|^{-1} \rho_G(u, v)^{2t}. \quad (21)$$

Therefore, we further conclude that

$$\mathbb{E}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1 \mid A] \geq |G|^{-1} \sum_{t=1}^{\lfloor m/2 \rfloor} \binom{m}{2t} \cdot \mathbb{E}[\rho_G(u, v)^{2t} \mid A]. \quad (22)$$

Recall that r(q) satisfies

$$\pi^2((u, v) : \rho_G(u, v) \leq r(q)) = \pi^2(A) = 1 - q^{-2}.$$

Hence, by definition of r(q) we have

$$\begin{aligned} |G|^{-1} \sum_{t=1}^{\lfloor m/2 \rfloor} \binom{m}{2t} \cdot \mathbb{E}[\rho_G(u, v)^{2t} \mid A] &\geq |G|^{-1} \sum_{t=1}^{\lfloor m/2 \rfloor} \binom{m}{2t} \cdot \mathbb{E}[\rho_G(u, v)^{2t} \mid \rho_G(u, v) \leq r(q)] \\ &\geq |G|^{-1} \sum_{t=1}^{\lfloor m/2 \rfloor} \binom{m}{2t} \cdot \mathbb{E}[|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t} \mid \rho_G(u, v) \leq r(q)]. \end{aligned} \quad (23)$$

$$(24)$$

In addition, we notice that for each even order 2t+ 1 with t = 1, . . . , ⌊m/2⌋−1, it holds by Lemma [4](#page-32-0) that

$$\frac{({m \choose 2t+1}) \cdot |\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t+1}}{\sqrt{({m \choose 2t}) \cdot |\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t} \cdot ({m \choose 2t+2}) \cdot |\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t+2}}} = \frac{({m \choose 2t+1})}{\sqrt{({m \choose 2t}) \cdot ({m \choose 2t+2})}} \leq 2.$$

Therefore, using the inequality 2 √ ab ≤ a + b for a, b ≥ 0, we obtain

$$\binom{m}{2t+1} \cdot |\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t+1} \leq \binom{m}{2t} \cdot |\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t} + \binom{m}{2t+2} \cdot |\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t+2}.$$

Consequently, the right-hand-side of [\(24\)](#page-31-2) can be further lower bounded by

$$|G|^{-1} \sum_{t=1}^{\lfloor m/2 \rfloor} \binom{m}{2t} \cdot \mathbb{E}[|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^{2t} \mid \rho_G(u, v) \leq r(q)] \quad (25)$$

$$\geq \frac{|G|^{-1}}{3} \sum_{t=2}^m \binom{m}{t} \cdot \mathbb{E}[|\langle L_u, L_v \rangle_{\mathbb{Q}} - 1|^t \mid \rho_G(u, v) \leq r(q)] \quad (26)$$

$$\geq \frac{|G|^{-1}}{3} \sum_{t=2}^m \binom{m}{t} \cdot \mathbb{E}[(\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t \mid \rho_G(u, v) \leq r(q)]. \quad (27)$$

Combining [\(22\)](#page-31-3), [\(24\)](#page-31-2), and [\(27\)](#page-32-1), with the condition of (q, m, ε)-GFP<sup>T</sup> hardness, we obtain

$$\sum_{t=2}^m \binom{m}{t} \cdot \mathbb{E}[(\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t \mid \rho_G(u, v) \leq r(q)] \leq 3|G| \cdot \mathbb{E}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1 \mid A]. \quad (28)$$

Again, by the definition of r(A) it follows that

$$\sum_{t=2}^m \binom{m}{t} \cdot \mathbb{E}[(\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t \cdot \mathbf{1}(\rho_G(u, v) \leq r(q))] \leq 3|G| \mathbb{E}[(\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1) \mathbf{1}(A)]$$

and therefore by [\(20\)](#page-31-4)

$$\begin{aligned} & \mathbb{E}[(\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1)\mathbf{1}(\rho_G(u, v) \leq r(q))] \\ & \leq 3|G|\mathbb{E}[(\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1)\mathbf{1}(A)] + m\mathbb{E}[(\langle L_u, L_v \rangle_{\mathbb{Q}} - 1) \cdot \mathbf{1}(\rho(u, v) \leq r(q))] \end{aligned}$$

Next, we aim to upper bound the first order term, namely m · <sup>E</sup>[(⟨Lu, Lv⟩ − 1) · 1(ρG(u, v) ≤ r(q))]. Note that A′ := {(u, v) : ρG(u, v) ≤ r(q)} is also G⊗<sup>2</sup> -invariant. Hence, employing also Assumption [1](#page-5-1) we also have

$$\begin{aligned} & \mathbb{E}[(\langle L_u, L_v \rangle - 1) \cdot \mathbf{1}(\rho_G(u, v) \leq r(q))] \\ &= \mathbb{E}[(\mathbb{E}_{g \sim \text{Unif}(G)} \langle L_g(u), L_g(v) \rangle_{\mathbb{Q}} - 1)) \cdot \mathbf{1}(\rho_G(u, v) \leq r(q))] \\ &\leq \mathbb{E}[(\mathbb{E}_{g \sim \text{Unif}(G)} \langle L_g(u), L_g(v) \rangle_{\mathbb{Q}} - 1))] \\ &= \mathbb{E}[(\langle L_u, L_v \rangle - 1)] \\ &= \chi^2(\mathbb{P}, \mathbb{Q}). \end{aligned}$$

Therefore,

$$\mathbb{E}[(\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1)\mathbf{1}(\rho_G(u, v) \leq r(q))] \leq 3|G|\mathbb{E}[(\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1)\mathbf{1}(A)] + m\chi^2(\mathbb{P}, \mathbb{Q}).$$

from which the result follows.

Lemma 4. *For any* t ∈ {1, 2, . . . , n − 1} *and* n ≥ 3*, we have*

$$\frac{\binom{n}{t}^2}{\binom{n}{t-1} \cdot \binom{n}{t+1}} \leq 4. \quad (29)$$

*Proof.* Note that by the successive ratio between binomial coefficients, we have

$$\frac{\binom{n}{t}^2}{\binom{n}{t-1} \cdot \binom{n}{t+1}} = 1 + \frac{1+n}{t(n-t)} \leq 1 + \frac{1+n}{n-1} = 2 + \frac{2}{n-1} \leq 4. \quad (30)$$

#### C.2 Proof of Theorem [3](#page-6-1)

*Proof of Theorem [3.](#page-6-1)* SQ implies ρG-FP. We have that

$$\sup_{A: \pi^2(A) \geq q^{-2}} \mathbb{E} [|\langle L_u, L_v \rangle - 1| \mid A] \leq \frac{1}{m}. \quad (31)$$

Now as G is π-preserving that easily implies that for any A such that G⊗<sup>2</sup> (A) = A, that

$$\sup_{A: \pi^2(A) \geq q^{-2}} \mathbb{E}[\rho_G(u, v) \mid A] \leq |G| \sup_{A: \pi^2(A) \geq q^{-2}} \mathbb{E}[\mathbb{E}_{g \sim \text{Unif}(G)} |\langle L_{g(u)}, L_{g(v)} \rangle - 1 \mid A] \leq \frac{|G|}{m}.$$

Hence for any r > 0 if we set A<sup>r</sup> = {ρG(u, v) ≥ r} since G⊗<sup>2</sup> (A) = A we conclude that π (Ar) ≥ q −2 implies r ≤ |G|/m. Recall that r(q) > 0 satisfies π (Ar(q)) ≥ q −2 . In particular, r(q) ≤ |G|/m, and therefore for any m′ ≤ m/2,

$$\begin{aligned} & \mathbb{E} \left[ \langle L_u^{\otimes m'}, L_v^{\otimes m'} \rangle_{\mathbb{Q}} \cdot \mathbf{1}(\rho_G(u, v) \leq r(q)) \right] \\ &= \mathbb{E} \left[ (\langle L_u, L_v \rangle_{\mathbb{Q}} - 1 + 1)^{m'} \cdot \mathbf{1}(\rho_G(u, v) \leq r(q)) \right] \\ &\leq \mathbb{E} \left[ (\rho_G(u, v) + 1)^{m'} \cdot \mathbf{1}(\rho_G(u, v) \leq r(q)) \right] \\ &< (r(a) + 1)^{m'} < (|G|/m + 1)^{m'} < 1 + e|G|m'/m \end{aligned} \quad (32)$$

$$\leq (r(q) + 1)^{m'} \leq (|G|/m + 1)^{m'} \leq 1 + e|G|m'/m. \quad (33)$$

This concludes the (q, m′ , e|G|m′/m)-ρG-FP hardness.

ρG-FP hardness implies SQ-hardness. Suppose we have (q, m, ε)-ρG-FP hardness

$$\mathbb{E} \left[ \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \leq 1 + \varepsilon, \quad \text{where} \quad \pi^2(\rho_G(u, v) \geq r(q)) = q^2.$$

By definition [4,](#page-6-4) we have that

$$\begin{aligned} 1 + \varepsilon &\geq \mathbb{E} \left[ (\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} - 1) \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \\ &= \mathbb{E} \left[ \sum_{t=1}^m \binom{m}{t} \cdot (\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \\ &= \mathbb{E} \left[ \sum_{t=1}^m \binom{m}{t} \mathbb{E}_{g \sim \text{Unif}(G)} [(\langle L_{g(u)}, L_{g(v)} \rangle_{\mathbb{Q}} - 1)^t] \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right], \end{aligned}$$

where the first inequality holds by the definition of the ρG-FP hardness and the second equality holds by using the elementary ⟨L ⊗m u , L⊗<sup>m</sup> v ⟩<sup>Q</sup> = (⟨Lu, Lv⟩<sup>Q</sup> − 1 + 1)<sup>m</sup>. The last equality holds by using that G is π-measure preserving. As crucially <sup>E</sup>g∼Unif(G) ⟨Lg(u) , Lg(v)⟩<sup>Q</sup> − 1 t ] ≥ 0 for all integers t ≥ 0, we have

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^m \binom{m}{t} \mathbb{E}_{g \sim \text{Unif}(G)} [(\langle L_{g(u)}, L_{g(v)} \rangle_{\mathbb{Q}} - 1)^t] \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \\ & \geq \mathbb{E} \left[ \sum_{t \leq m, t \text{ even}} \binom{m}{t} \cdot \mathbb{E}_{g \sim \text{Unif}(G)} [(\langle L_{g(u)}, L_{g(v)} \rangle_{\mathbb{Q}} - 1)^t] \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \\ & \geq \max_{\substack{1 \leq t \leq m, \\ t \text{ even}}} \mathbb{E} \left[ \binom{m}{t} \cdot (\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right]. \end{aligned}$$

Hence, combining the two for all even t, with 1 ≤ t ≤ m,

$$\max_{\substack{1 \leq t \leq m, \\ t \text{ even}}} \mathbb{E} \left[ (\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^t \cdot \mathbf{1}(\rho_G(u, v) < r(q)) \right] \leq \frac{1 + \varepsilon}{\binom{m}{t}}. \quad (34)$$

Therefore, we have for any even t with t ≤ m that

$$\begin{aligned} \mathbb{E}[(\langle L_u, L_v \rangle - 1)^t] &= \mathbb{E}[(\langle L_u, L_v \rangle - 1)^t \mathbf{1}(\rho_G(u, v) < r(q))] + \mathbb{E}[(\langle L_u, L_v \rangle - 1)^t \mathbf{1}(\rho_G(u, v) \geq r(q))] \\ &\leq \frac{1 + \varepsilon}{\binom{m}{t}} + \mathbb{E}[(\langle L_u, L_v \rangle - 1)^{2t}]^{1/2} \cdot q^{-1} \\ &\leq \left( \frac{t(1 + \varepsilon)^{1/t}}{m} + \frac{\chi^2(\mathbb{P}^{\otimes 4t} \parallel \mathbb{Q}^{\otimes 4t})^{1/2t}}{q^{1/t}} \right)^t. \end{aligned}$$

where in the first inequality, we use the Cauchy-Schwarz inequality for the second term and the fact that π 2 (ρG(u, v) ≥ r(q)) ≤ q . In the second term, we use the elementary <sup>m</sup> t ≥ (m/t) t .

Now focusing on t ≤ log q/ log m we further have

$$\mathbb{E}[(\langle L_u, L_v \rangle - 1)^t] \leq \left( \frac{t(1 + \varepsilon)^{1/t} + \chi^2(\mathbb{P}^{\otimes 4t} \parallel \mathbb{Q}^{\otimes 4t})}{m} \right)^t. \quad (35)$$

Hence the model is ( m t(1+ε) <sup>1</sup>/t+χ<sup>2</sup>(P⊗4<sup>t</sup> ∥ Q⊗4<sup>t</sup>) , t)-USQ hard. By Proposition [1](#page-27-2) we conclude for any q ′ > 0 that the model is (q ′ , m(q ) −2/t t(1+ε) <sup>1</sup>/t+χ<sup>2</sup>(P⊗4<sup>t</sup> ∥ Q⊗4<sup>t</sup>) )-SQ hard. The second part follows by setting t = (log m) s and q ′ = e δ(log m) s+1 .

## D Details of examples and proofs

#### D.1 Symmetric mixed sparse linear regression

The symmetric mixed sparse linear regression (mSLR) setting, is a P versus Q detection task defined as follows. Given k, n ∈ N with k ≤ n and σ <sup>2</sup> > 0, we have:

- Under the planted model, we first sample u ∼ π uniformly from set u ∈ {0, 1} <sup>n</sup> with ∥u∥<sup>0</sup> = k. Then, the sample (x<sup>i</sup> , yi) ∼ <sup>P</sup><sup>u</sup> is generated by

$$y_i = (k + \sigma)^{-1} [z_i \odot \langle x_i, u \rangle + (1 - z) \odot \langle x_i, -u \rangle + w_i],$$

for independent w<sup>i</sup> ∼ N (0, σ<sup>2</sup> ), x<sup>i</sup> ∼ N (0, In) and z<sup>i</sup> ∼ Bern(1/2). Following [\[5\]](#page-10-2) we also denote SNR := k/σ<sup>2</sup> .

- Under the null model, the sample (y<sup>i</sup> , xi) ∼ Q is generated by y<sup>i</sup> ∼ N (0, 1) and independently x<sup>i</sup> ∼ N (0, In).

To see that mSLR is a PSM, set for any u, ϕ<sup>u</sup> := suppport(u) ∪ {n + 1}, that is the coordinates of ((xi)<sup>j</sup> )j∈support(u) and of y<sup>i</sup> . Then it is easy to confirm that for any subset S ⊆ Φu, the marginal distribution Pu|<sup>ϕ</sup><sup>u</sup> (S) does not depend on u but only on S; the choice of suppport(u) \ S does not alter this distribution.

It is known that the information theory sample size threshold of the problem is

$$m_{\text{STATS}} = \tilde{\Theta} \left( \frac{k}{\log\left(\frac{\text{SNR}^2}{2\text{SNR}_{+1}} + 1\right)} \right),$$

see e.g., [\[19\]](#page-11-11). Also in [\[5\]](#page-10-2) it was proven that in the similar mSLR setting where u's coordinates can take values in {−1, 0, 1}, if

$$m \leq m_{\text{ALG}} = \tilde{\Theta} \left( \frac{(\text{SNR} + 1)^2}{\text{SNR}^2} k^2 \right),$$

then the problem is O(log n)-degree hard. Here we prove that with sample size m ≤ (mALG) 1−o(1) the problem is also GFP-hard, and hence via Theorem [3](#page-6-1) also SQ-hard. Our result holds under a very mild assumption on SNR being not exponential in k. Interestingly, the proof is relatively short.

The first step is to calculate the inner product ⟨L ⊗m u , L⊗<sup>m</sup> v ⟩<sup>Q</sup> which accounts to a calculation over the Gaussian measure.

Lemma 5. *For any sample size* m *and any* u, v *binary* k*-sparse vector, the following holds for the mSLR model:*

$$\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} = \left( 1 - \left( \frac{\langle u, v \rangle}{k + \sigma^2} \right)^2 \right)^{-m} \leq \exp \left( \frac{m \langle u, v \rangle^2}{(k + \sigma^2)^2 - \langle u, v \rangle^2} \right).$$

Using Lemma [5](#page-35-2) one can prove the GFP-hardness, and therefore the SQ-hardness.

Theorem 7. *If* n Ω(1) = k = o(n 1/2 ) *then for any* m = o k log( SNR<sup>2</sup> 2SNR+1 +1) , *it holds*

$$\chi^2(\mathbb{P}^{\otimes m}, \mathbb{Q}^{\otimes m}) = 1 + o(1).$$

*Moreover, for any constant* T > 1, *for any* m = O (SNR+1)<sup>2</sup>k SNR<sup>2</sup>(log n) <sup>2</sup><sup>T</sup> +2 *and* q = e Θ((log n) <sup>T</sup> ) , *the mSLR task is* (q, m, O(1))*-GFP hard. In particular, if* SNR ≤ e k 1−α *for some* α > 0, *then for any* T > 1 *the mSLR task is* (e Θ((log n) <sup>T</sup> ) ,( (SNR+1)<sup>2</sup> SNR<sup>2</sup> k 2 ) <sup>1</sup>−o(1))*-SQ hard.*

#### D.1.1 Proofs for mSLR

*Proof of Lemma [5.](#page-35-2)* Let λ = p k/σ<sup>2</sup> + 1 and since ⟨L ⊗m u , L⊗<sup>m</sup> v ⟩<sup>Q</sup> = (⟨L ⊗m u , L⊗<sup>m</sup> v ⟩Q) <sup>m</sup> we focus on the case m = 1.

Let Y = Y1, X = X1. By definition and Bayes' rule,

$$L_u = L_u(X, Y) = \frac{\mathbb{P}(Y|X, u)}{\mathbb{Q}(Y)}$$

Under Q we have λσY ∼ N (0, λ<sup>2</sup>σ 2 ), while under <sup>P</sup> conditional on (X, u) we have

$$\lambda\sigma Y = \sqrt{k + \sigma^2} Y \sim \frac{1}{2}\mathcal{N}(\langle X, u \rangle, \sigma^2) + \frac{1}{2}\mathcal{N}(-\langle X, u \rangle, \sigma^2),$$

and so

$$\begin{aligned} L_u &= \frac{\mathbb{P}(Y|X, u)}{\mathbb{Q}(Y)} \\ &= \frac{1}{2} \lambda \exp \left( -\frac{1}{2\sigma^2} (\lambda \sigma Y - \langle X, u \rangle)^2 + \frac{1}{2\lambda^2 \sigma^2} (\lambda \sigma Y)^2 \right) \\ &\quad + \frac{1}{2} \lambda \exp \left( -\frac{1}{2\sigma^2} (\lambda \sigma Y + \langle X, u \rangle)^2 + \frac{1}{2\lambda^2 \sigma^2} (\lambda \sigma Y)^2 \right) \\ &= \frac{\lambda^m}{2} \left\{ \exp \left( -\frac{\lambda^2 - 1}{2} Y^2 + \frac{\lambda}{\sigma} Y \langle X, u \rangle - \frac{1}{2\sigma^2} \langle X, u \rangle^2 \right) \right. \\ &\quad \left. + \exp \left( -\frac{\lambda^2 - 1}{2} Y^2 - \frac{\lambda}{\sigma} Y \langle X, u \rangle - \frac{1}{2\sigma^2} \langle X, u \rangle^2 \right) \right\}. \end{aligned}$$

Now a standard integration argument using the MGF of the χ <sup>2</sup> distribution (see e.g., the proof of [\[6,](#page-10-0) Proposition 6.8.] for an almost identical argument) gives for any u, v binary k-sparse vectors,

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = \frac{\lambda^2}{2(2\lambda^2 - 1)^{1/2}} \mathbb{E}_{X \sim \mathbb{Q}}(\exp\left(\frac{1}{2\sigma^2(2\lambda^2 - 1)} [(1 - \lambda^2)(\langle X, u \rangle^2 + \langle X, v \rangle^2) + 2\lambda^2 \langle X, u \rangle \langle X, v \rangle]\right)) \quad (36)$$

$$+ \exp \left( \frac{1}{2\sigma^2(2\lambda^2 - 1)} [(1 - \lambda^2) (\langle X, u \rangle^2 + \langle X, v \rangle^2) - 2\lambda^2 \langle X, u \rangle \langle X, v \rangle] \right) \quad (37)$$

Now of course the pair (⟨X, u⟩,⟨X, v⟩) follows a bivariate Gaussian law with variances equals to k and covariance ⟨u, v⟩. Hence, some standard manipulations (see again the proof of [\[6,](#page-10-0) Proposition 6.8.] for an almost identical argument) allow us to derive that for Z ∈ R <sup>1</sup>×<sup>3</sup> with i.i.d. N (0, 1) entries and

$$t := \frac{1}{2\sigma^2(2\lambda^2 - 1)} = \frac{1}{2\sigma^2(2k/\sigma^2 + 1)} = \frac{1}{4k + 2\sigma^2}. \quad (38)$$

it holds

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = \frac{\lambda^{2m}}{2(2\lambda^2 - 1)^{m/2}} \mathbb{E}_Z(\exp(t\langle M_1, Z^\top Z \rangle) + \exp(t\langle M_2, Z^\top Z \rangle)) \quad (39)$$

where for ℓ := ⟨u, v⟩,

$$M_1 = M_1(\ell) := \begin{pmatrix} \frac{2\ell}{\sqrt{\ell(k-\ell)}} & \sqrt{\ell(k-\ell)} & \sqrt{\ell(k-\ell)} \\ \sqrt{\ell(k-\ell)} & (1-\lambda^2)(k-\ell) & \lambda^2(k-\ell) \\ \sqrt{\ell(k-\ell)} & \lambda^2(k-\ell) & (1-\lambda^2)(k-\ell) \end{pmatrix}.$$

and

$$M_2 = M_2(\ell) := \begin{pmatrix} 2(1 - 2\lambda^2)\ell & (1 - 2\lambda^2)\sqrt{\ell(k-\ell)} & (1 - 2\lambda^2)\sqrt{\ell(k-\ell)} \\ (1 - 2\lambda^2)\sqrt{\ell(k-\ell)} & (1 - \lambda^2)(k-\ell) & -\lambda^2(k-\ell) \\ (1 - 2\lambda^2)\sqrt{\ell(k-\ell)} & -\lambda^2(k-\ell) & (1 - \lambda^2)(k-\ell) \end{pmatrix}.$$

The eigendecompositions of <sup>M</sup>1, M<sup>2</sup> of the form P<sup>3</sup> <sup>i</sup>=1 λ<sup>i</sup> uiu ⊤ i <sup>∥</sup>ui∥<sup>2</sup> are, first for M1,

$$\begin{aligned} u_1^\top &= (0 \quad 1 \quad -1) & \lambda_1 &= (1 - 2\lambda^2)(k - \ell) \\ u_2^\top &= (\sqrt{k - \ell} \quad -\sqrt{\ell} \quad -\sqrt{\ell}) & \lambda_2 &= 0 \\ u_3^\top &= (2\sqrt{\ell} \quad \sqrt{k - \ell} \quad \sqrt{k - \ell}) & \lambda_3 &= k + \ell. \end{aligned} \tag{40}$$

and for M2,

$$\begin{aligned} u_1^\top &= \begin{pmatrix} 0 & 1 & -1 \\ & & \\ & & \\ & & & \end{pmatrix} & \lambda_1 = k - \ell \\ u_2^\top &= (\sqrt{k-\ell} & -\sqrt{\ell} & -\sqrt{\ell}) & \lambda_2 = 0 & (41) \\ u_3^\top &= (2\sqrt{\ell} & \sqrt{k-\ell} & \sqrt{k-\ell}) & \lambda_3 = (1 - 2\lambda^2)(k + \ell). & \end{aligned}$$

As t < 1/(4k) we have 2t max{∥M1∥op, ∥M2∥op} < (k + ℓ)/2k ≤ 1. Hence, using [\[6,](#page-10-0) Lemma A.5.] for B(U) = <sup>R</sup> <sup>n</sup>×<sup>m</sup>, we have

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = \frac{\lambda^2}{2(2\lambda^2 - 1)^{1/2}} (\det(I_3 - 2tM_1)^{-1/2} + \det(I_3 - 2tM_2)^{-1/2}). \quad (42)$$

Using [\(38\)](#page-36-1) and [\(40\)](#page-37-0), [\(41\)](#page-37-1) the eigenvalues of the matrices I<sup>3</sup> − 2tM1, I<sup>3</sup> − 2tM<sup>2</sup> are

$$\{1, 1 - 2t(k + \ell), 1 - 2t(1 - 2\lambda^2)(k - \ell)\} = \left\{1, 1 - \frac{k + \ell}{\sigma^2(2\lambda^2 - 1)}, 1 + \frac{k - \ell}{\sigma^2}\right\}$$

and

$$\{1, 1 - 2t(k - \ell), 1 - 2t(1 - 2\lambda^2)(k + \ell)\} = \left\{1, 1 - \frac{k - \ell}{\sigma^2(2\lambda^2 - 1)}, 1 + \frac{k + \ell}{\sigma^2}\right\}.$$

Since λ <sup>2</sup> = k/σ<sup>2</sup> + 1 we have

$$\frac{\lambda^2}{\sqrt{2\lambda^2 - 1}} \det(I_3 - 2tM_1)^{-1/2} = \lambda^2 \left[ \left( 2\lambda^2 - 1 - \frac{k + \ell}{\sigma^2} \right) \left( 1 + \frac{k - \ell}{\sigma^2} \right) \right]^{-1/2} \quad (43)$$

$$= \frac{\frac{k}{\sigma^2} + 1}{1 + \frac{k-\ell}{\sigma^2}} \quad (44)$$

$$= \left( 1 - \frac{\ell}{k + \sigma^2} \right)^{-1}. \quad (45)$$

and by symmetry,

$$\frac{\lambda^2}{\sqrt{2\lambda^2 - 1}} \det(I_3 - 2tM_2)^{-1/2} = \left(1 + \frac{\ell}{k + \sigma^2}\right)^{-1}. \quad (46)$$

Combining the above,

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = \frac{1}{2} \left( \left( 1 - \frac{\ell}{k + \sigma^2} \right)^{-1} + \left( 1 + \frac{\ell}{k + \sigma^2} \right)^{-1} \right) \quad (47)$$

$$= \left( 1 - \left( \frac{\ell}{k + \sigma^2} \right)^2 \right)^{-1} \quad (48)$$

$$\leq \exp \left( \frac{m\ell^2}{(k + \sigma^2)^2 - \ell^2} \right), \quad (49)$$

where for the last inequality we used that log x ≥ 1 − 1/x, for x > 0.

*Proof of Theorem [7.](#page-35-3)* We have from the first part of Lemma [5,](#page-35-2)

$$\chi^2(\mathbb{P}^{\otimes m}, \mathbb{Q}^{\otimes m}) - 1 = \mathbb{E}_{u, v \sim \pi} \langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \leq \mathbb{E}_{u, v \sim \pi} \left( 1 - \frac{\langle u, v \rangle^2}{(k + \sigma^2)^2} \right)^{-m} \quad (50)$$

But, in this setting ⟨u, v⟩ follows an Hypergeometric distribution with parameters n, k, k. Hence, by [\[6,](#page-10-0) Lemma 6.6],

$$\begin{aligned} \chi^2(\mathbb{P}^{\otimes m}, \mathbb{Q}^{\otimes m}) - 1 &\leq \sum_{\ell=0}^k \left( 1 - \frac{\ell^2}{(k + \sigma^2)^2} \right)^{-m} \left( \frac{k^2}{n - k} \right)^\ell \\ &\leq \sum_{\ell=0}^{\lfloor k/2 \rfloor} \exp \left( \frac{m \ell^2}{(k + \sigma^2)^2 - \ell^2} \right) e^{-\ell \log \left( \frac{n-k}{k^2} \right)} + \sum_{\ell=\lfloor k/2 \rfloor}^k \left( 1 - \frac{k^2}{(k + \sigma^2)} \right)^{-m} e^{-\ell \log \left( \frac{n-k}{k^2} \right)} \\ &\leq \sum_{\ell=0}^{\lfloor k/2 \rfloor} e^{\ell m / (k-\ell) - \ell \log \left( \frac{n-k}{k^2} \right)} + k \left( \frac{(\frac{k}{\sigma^2})^2}{2 \frac{k}{\sigma^2} + 1} + 1 \right)^m e^{-\Theta(k \log \left( \frac{n-k}{k^2} \right))}, \end{aligned} \tag{51}$$

where for the last inequality we used log x ≥ 1 − 1/x for x > 0.

Since k <sup>2</sup> = o(n), m = o( k log( SNR<sup>2</sup> 2SNR+1 +1) ), SNR = k/σ<sup>2</sup> we have for large enough n,

$$k\left(\frac{\frac{k}{\sigma^2})^2}{\frac{2}{\sigma^2} + 1} + 1\right)^m e^{-\Theta(k \log(\frac{n-k}{k^2}))} = e^{-\Theta(k \log(\frac{n-k}{k^2}))} = o(1).$$

Moreover, since k <sup>2</sup> = o(n), m = o(k) we have for large enough n,

$$\sum_{\ell=0}^{\lfloor k/2 \rfloor} e^{\ell m/(k-\ell) - \ell \log\left(\frac{n-k}{k^2}\right)} \leq \sum_{\ell=0}^{\lfloor k/2 \rfloor} e^{2\ell m/k - \ell \log\left(\frac{n-k}{k^2}\right)} \leq \sum_{\ell=0}^{\lfloor k/2 \rfloor} e^{-\ell \log\left(\frac{n-k}{k^2}\right)/2} = 1 + o(1).$$

Now, fix any T > 1. Notice ⟨Lu, Lv⟩<sup>Q</sup> is an increasing function of ⟨u, v⟩. Hence, for any q > 0 there exists δ0(q) > 0 such that {ρid(u, v) ≥ r(q)} = {⟨u, v⟩ ≥ δ0(q)}. Moreover, from the tail of ⟨u, v⟩ which is an (n, k.k) Hypergeometric distribution, there exists some q = q<sup>T</sup> = e Θ((log n) <sup>T</sup> ) for which there exists r(T) with π({ρid(u, v) ≥ r(T)}) = q −2 .

Let us then fix q = q<sup>T</sup> . Notice that if we choose δ := log(kq<sup>2</sup> ) = Θ((log n) <sup>T</sup> +1), then we have for large enough n, by [\[6,](#page-10-0) Lemma 6.6] π ⊗2 (⟨u, v⟩ ≥ δ) ≤ k( k 2 n ) <sup>δ</sup> ≤ k2 <sup>−</sup><sup>δ</sup> = 1/q<sup>2</sup> . Hence, δ<sup>0</sup> ≤ δ. Combining the above with the second part of Lemma [5,](#page-35-2)

$$\begin{aligned}\mathbb{E}_{u,v \sim \pi}(\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \mathbf{1}(\langle u, v \rangle \leq \delta_0)) &\leq \mathbb{E}_{u,v \sim \pi} \exp \left( \frac{m \langle u, v \rangle^2}{(k + \sigma^2)^2 - \langle u, v \rangle^2} \right) \mathbf{1}(\langle u, v \rangle \leq \delta) \\ &\leq \exp \left( \frac{m \delta^2}{(k + \sigma^2)^2 - \delta^2} \right) \\ &= \exp \left( m \Theta \left( \frac{(\log n)^{2(T+1)}}{k^2 (1 + \text{SNR}^{-1})^2} \right) \right) = O(1),\end{aligned}$$

for any m = O( k (1+SNR<sup>−</sup><sup>1</sup> ) (log n) 2(<sup>T</sup> +1) ). Hence, the model is (<sup>q</sup> <sup>=</sup> <sup>q</sup><sup>T</sup> , m = Θ( <sup>k</sup> (1+SNR<sup>−</sup><sup>1</sup> ) (log n) 2(<sup>T</sup> +1) ), O(1))-ρidhard. Using now Theorem [3](#page-6-1) for mIT = (log n) T , which is permissible to use using our χ <sup>2</sup> bound and that SNR ≤ e k 1−α for some α > 0, we conclude for all T > 1, the (e Θ((log n) T−1 ) ,(k 2 (1 + SNR−<sup>1</sup> ) 2 ) <sup>1</sup>−o(1))-SQ hardness. The result follows.

#### D.2 Non-Gaussian component analysis

The following model was introduced in [\[16\]](#page-11-10) to capture the complexity of learning Gaussian mixtures. Definition 10 (Non-Gaussian component analysis model). *A "*P *versus* Q*" detection problem is a Non-Gaussian component (NGCA) model if:*

- *There exists* µ ∈ P(R) *such that, under the planted hypothesis* <sup>P</sup><sup>u</sup> *with* u ∈ S<sup>n</sup>−<sup>1</sup> *, we sample* x ∼ N (0, In) *and replace the component* ⟨x, u⟩ · u *by* z · u *where* z ∼ µ *independently;*

- *Under the null model, we sample* x ∼ N (0, In)*.*

In other words, an NGCA model is an isotropic Gaussian distribution with a non-Gaussian marginal in direction u. The SQ-hardness for NGCA models established also in [\[16\]](#page-11-10) has been of big importance in proving many recent SQ-hardness for learning tasks, such as for robust estimation of Gaussians [\[17\]](#page-11-12) and robust linear regression [\[17\]](#page-11-12) among others.

Interestingly, we can also connect all NGCA models with GFP-hardness for G = <sup>Z</sup>2. By a direct Hermite expansion, we can decompose the likelihood function (in L (Q))

$$L_u(x) = 1 + \sum_{i=s^*}^{\infty} \nu_i h_i(\langle u, x \rangle), \quad \nu_i := \mathbb{E}_{z \sim \mu}[h_i(z)],$$

where h<sup>i</sup> is the (normalized) degree-i Hermite polynomial. Here we denoted s <sup>∗</sup> > 0 the first non-zero coefficient ν<sup>i</sup> ̸= 0, that is, the smallest moment of µ that disagrees with N(0, 1) moments (we call s ∗ the generative exponent of the NGCA model). The inner-product of the likelihood ratios is given for all u, v ∈ S<sup>n</sup>−<sup>1</sup> by

$$\langle L_u, L_v \rangle = 1 + \sum_{i=s^*}^{\infty} \nu_i^2 \cdot (\langle u, v \rangle)^s, \quad (54)$$

where we used that <sup>E</sup>[hs(⟨u, x⟩)hk(⟨v, x⟩)] = δks⟨u, v⟩ s . Similar to GAMs, using again the group G = <sup>Z</sup><sup>2</sup> acting on flipping the sign of each parameter, we get

$$\mathbb{E}_{g,g' \sim \text{Unif}(G)}(\langle L_{g(u)}, L_{g'(v)} \rangle_{\mathbb{Q}} - 1) = \sum_{i=s^*, i \text{ even}}^{\infty} \nu_i^2 \cdot (\langle u, v \rangle)^i \geq 0,$$

concluding that NGCA satisfy Assumption [1](#page-5-1) with G = <sup>Z</sup>2. Hence, based on our equivalence, for any NGCA model the SQ-hardness is equivalent with the GFP-hardness for any symmetric prior (that is π(−u) = π(u)). We illustrate this equivalence for two standard priors: the uniform prior π = Unif(S n−1 ) and the <sup>k</sup>-sparse prior <sup>π</sup> <sup>=</sup> Unif({<sup>u</sup> ∈ ± √ 1 k {0, 1} <sup>n</sup> : ∥u∥<sup>0</sup> = k}).

Theorem 8 (GFP-hardness of NGCA, uniform prior). *Consider a NGCA model with generative exponent* s <sup>∗</sup> *and the uniform prior* π = *Unif*(S n−1 )*. For any* ε ∈ (0, 1/2)*, the NGCA model is* (exp (Θ(n ε )), m, O(1))*-GFP hard with*

$$m = \frac{1}{\nu_s^2} n^{s^*/2 - \Theta(\varepsilon)}.$$

*Moreover, via our equivalence theorem, the model is* (exp n Θ(ε) , m<sup>1</sup>−Θ(ε) )*-SQ hard.*

Theorem 9 (GFP-hardness of NGCA, sparse prior). *Consider a NGCA model with generative exponent* s <sup>∗</sup> *and the* <sup>k</sup>*-sparse prior* <sup>π</sup> <sup>=</sup> *Unif*({<sup>u</sup> ∈ ± √ 1 k {0, 1} <sup>n</sup> : ∥u∥<sup>0</sup> = k})*. For any* ε ∈ (0, 1/2) *so that* k = n Ω(ε) *, the NGCA model is* (exp (Θ(n ε )), m, O(1))*-GFP hard with*

$$m = \frac{1}{\nu_{s^*}^2} \min(n^{s^*/2 - \Theta(\varepsilon)}, k^{s^*} n^{-\Theta(\varepsilon)}).$$

*Moreover, via our equivalence theorem, the model is* (exp n Θ(ε) , m<sup>1</sup>−Θ(ε) )*-SQ hard.*

The SQ lower bound in Theorem [8](#page-39-1) was proven in [\[16\]](#page-11-10) by a direct argument: here, we prove this SQ-hardness via equivalence to GFP-hardness. The sparse prior was not considered previously and we include it to illustrate the broad applicability of our equivalence.

### D.2.1 Proofs for Non-Gaussian Component Analysis

*Proof of Theorem [8.](#page-39-1)* Let us prove that the model is ρ<sup>Z</sup><sup>2</sup> -FP hard and conclude using the implication in Theorem [2.](#page-6-0)1. Note that ρ<sup>Z</sup><sup>2</sup> (u, v) = ⟨Lu, Lsign(⟨u,v⟩)v⟩ − 1, that is

$$\rho_{\mathbb{Z}_2}(u, v) = \sum_{s=s^*}^{\infty} \nu_s^2 |\langle u, v \rangle|^s,$$

and ρ<sup>Z</sup><sup>2</sup> (u, v) is an increasing function of |⟨u, v⟩|. Thus, ρ<sup>Z</sup><sup>2</sup> -FP hardness is equivalent to FP hardness. Using that ⟨u, v⟩ under the uniform prior is distributed as the first coordinate of z ∼ Unif(S d−1 ), we get

$$\pi(|\langle u, v \rangle| \geq \kappa) \leq 2\exp\left(-cn\kappa^2\right),$$

for some universal constant c > 0. For simplicity denote ρ = |⟨u, v⟩|. Using that ν 2 <sup>s</sup> ≤ 1 for any s ∈ N by Jensen's inequality, we can write

$$\langle L_u, L_v \rangle - 1 \leq \sum_{s \geq s^*} \nu_s^2 \rho^s \leq \nu_{s^*}^2 \rho^{s^*} + \rho^{s^{*+1}} \sum_{s \geq s^{*+1}} \rho^s \leq \rho^{s^*} \left( \nu_{s^*}^2 + \frac{\rho}{1-\rho} \right),$$

so that for ρ = on(1) and n large enough, ⟨Lu, Lv⟩−1 ≤ 2ν 2 s <sup>∗</sup> ρ s ∗ . We deduce that for κ = n −1/2+ε , we have

$$\begin{aligned} \mathbb{E} [\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle_{\mathbb{Q}} \cdot \mathbf{1}(|\langle u, v \rangle| < \kappa)] &\leq 1 + \sum_{j=1}^m \binom{m}{j} \mathbb{E} [(\langle L_u, L_v \rangle_{\mathbb{Q}} - 1)^j \cdot \mathbf{1}(|\langle u, v \rangle| < \kappa)] \\ &\leq 1 + \sum_{j=1}^m (2m\nu_{s^*}^2 \kappa^{s^*})^j. \end{aligned}$$

Thus we deduce that the problem is (exp (Θ(n ε )), m, Θ(1))-GFP hard with m = n s <sup>∗</sup>/2−Θ(ε)/ν<sup>2</sup> s ∗ . To use the equivalence with SQ, we need to compute the χ 2 -divergence, that is

$$\mathbb{E}[\langle L_u, L_v \rangle^{4t}] = \mathbb{E}[\langle L_u, L_v \rangle^{4t}] = 1 + \sum_{j=1}^{4t} \binom{4t}{j} \mathbb{E}[\langle \langle L_u, L_v \rangle - 1 \rangle^j].$$

Let us bound

$$\begin{aligned}\mathbb{E}[(\langle L_u, L_v \rangle - 1)^j] &= \mathbb{E}[(\langle L_u, L_v \rangle - 1)^j \cdot \mathbf{1}(|\rho| \leq \kappa)] + \mathbb{E}[(\langle L_u, L_v \rangle - 1)^j \cdot \mathbf{1}(|\rho| > \kappa)] \\ &\leq (2\nu_{s^*}^2 \kappa^{s^*})^j + M^j \exp(-cn^{2\varepsilon}),\end{aligned}$$

where we denoted M = ∥Lu∥ <sup>Q</sup> − 1 = O(exp n ε/2 ). Thus

$$\mathbb{E}[(\langle L_u, L_v \rangle - 1)^j] = 1 + \sum_{j=1}^{4t} (8t\nu_{s*}^2 \kappa^{s^*})^j + (4tM)^j \exp(-cn^{2\varepsilon}) = O(1),$$

where we used that tlog(t) = Θ( ˜ n ε/2 ) by assumption. We can therefore apply Theorem [3](#page-6-1) with q ′ = exp n ε/2 and t = n ε/2 (so that t ≤ log(q)/ log(m) = Θ( ˜ n ε )). The model is (q ′ , m′ )-SQ hard with

$$m' = \frac{m}{(t(1 + \varepsilon)^{1/t} + \chi^2(\mathbb{P}^{\otimes 4t} \parallel \mathbb{Q}^{\otimes 4t}))(q')^{2/t}} = \Theta(m/t) = m^{1-\Theta(\varepsilon)},$$

which concludes the proof.

*Proof of Theorem [9.](#page-39-2)* The proof proceeds similarly as the proof of Theorem [8.](#page-39-1) The main difference is the new tail bound on ⟨u, v⟩ given in Lemma [6.](#page-40-0) We now set κ = n <sup>ε</sup> max(n −1/2 , k−<sup>1</sup> ), so that

$$\pi(|\langle u, v \rangle| \geq \kappa) \leq 2\exp(-cn^\varepsilon).$$

With this modification, the rest of the proof is identical and we omit it.

Lemma 6 (Tail bound for sparse prior). *Let* u, v *be independently sampled from the prior* π = *Unif*({<sup>u</sup> ∈ ± √ 1 k {0, 1} <sup>n</sup> : ∥u∥<sup>0</sup> = k})*. Then for any* t ≥ 0*, we have*

$$\pi^2(\langle u, v \rangle \geq t) \leq \exp\left(-c \min\{nt^2, kt\}\right), \quad (55)$$

#### D.3 Single-index Models

Another extremely popular class of models in statistics dating back to the 80s [\[33,](#page-12-7) [28\]](#page-11-13) are the so-called single-index models.

Definition 11 (Single-index model). *A "*P *versus* Q*" detection problem is a Single-index model if:*

- *There exists a distribution* µ ∈ P(<sup>R</sup> × <sup>R</sup>) *such that, under the planted hypothesis* <sup>P</sup>u*, we sample* x ∼ N (0, In) *and* y ∼ µ(·|zu)*, where* z<sup>u</sup> := ⟨x, u⟩*;*
- *Under the null model, we sample* x ∼ N (0, In) *and* y ∼ µy*, where* µ<sup>y</sup> *is the marginal distribution of* µ*.*

Also, all single index models satisfy Assumption [1](#page-5-1) for G = <sup>Z</sup>2. Indeed, if s ∗ is the generative exponent of the model [\[12\]](#page-10-10), following [\[12\]](#page-10-10) we know that an Hermite expansion gives for some s <sup>∗</sup> ∈ <sup>N</sup> (s ∗ is called the generative exponent) that for all u, v ∈ S<sup>n</sup>−<sup>1</sup> ,

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = 1 + \sum_{i=s^*}^{\infty} \lambda_i^2 \cdot (\langle u, v \rangle)^i, \quad \lambda_i := \|\zeta_i(Y)\|_{\mu_y}, \quad \zeta_i(y) := \mathbb{E}[h_s(z)|y].$$

From this point on, the argument is identical as in the case of NGCA, including the nonnegativity with G = <sup>Z</sup><sup>2</sup> as well as the examples of GFP-hardness with uniform and sparse priors. For completeness, we state separate theorems for single-index models:

Theorem 10 (GFP-hardness of SI models, uniform prior). *Consider a SI model with generative exponent* s <sup>∗</sup> *and the uniform prior* π = *Unif*(S n−1 )*. For any* ε ∈ (0, 1/2)*, the SI model is* (exp (Θ(n ε )), m, O(1))*-GFP hard with*

$$m = \frac{1}{\lambda_{s^*}^2} n^{s^*/2 - \Theta(\varepsilon)}.$$

*Moreover, via our equivalence theorem, the model is* (exp n Θ(ε) , m<sup>1</sup>−Θ(ε) )*-SQ hard.*

Theorem 11 (GFP-hardness of SI models, sparse prior). *Consider a SI model with generative exponent* s <sup>∗</sup> *and the* <sup>k</sup>*-sparse prior* <sup>π</sup> <sup>=</sup> *Unif*({<sup>u</sup> ∈ ± √ 1 k {0, 1} <sup>n</sup> : ∥u∥<sup>0</sup> = k})*. For any* ε ∈ (0, 1/2) *so that* k = n Ω(ε) *, the SI model is* (exp (Θ(n ε )), m, O(1))*-GFP hard with*

$$m = \frac{1}{\lambda_{s^*}^2} \min(n^{s^*/2 - \Theta(\varepsilon)}, k^{s^*} n^{-\Theta(\varepsilon)}).$$

*Moreover, via our equivalence theorem, the model is* (exp n Θ(ε) , m<sup>1</sup>−Θ(ε) )*-SQ hard.*

The SQ lower bounds in Theorem [10](#page-41-2) and Theorem [11](#page-41-3) were proven in [\[12\]](#page-10-10) and [\[9\]](#page-10-12) via direct argument. Here, we obtain these bounds via the equivalence of the SQ-hardness and GFP-hardness.

#### D.4 Truncated statistics: convex truncation

Learning from truncated data has been a topic of interest since the late 1800s and the pioneering works of Galton [\[23\]](#page-11-14) and Pearson [\[35\]](#page-12-8). Interestingly, there has been some recent line of works on truncated statistics tasks that seeks to revisit these old questions from a computational viewpoint, see e.g., [\[13\]](#page-10-13), [\[14\]](#page-10-14) and references therein. In this line of recent work, the problem of detecting a convex truncation in Gaussian noise has been proposed.

Definition 12. *Fix* α ∈ (0, 1)*. A hypothesis testing "*<sup>P</sup> *versus* Q*" problem is called an* α*-Convex Truncation model if it satisfies:*

- *1. Under the null hypothesis* Q*,* x ∼ N(0, In)*.*
- *2. Under the planted hypothesis* <sup>P</sup>K*,* x ∼ N(0, In)|K *where* K *is a symmetric convex body with Gaussian volume at most* 1 − α*.*

Interestingly, also all α-Convex Truncation models satisfy Assumption [1](#page-5-1) for the trivial group G. Perhaps this fact is even more interesting because it turns out Assumption [1](#page-5-1) is exactly equivalent with the celebrated Gaussian Correlation Inequality on convex bodies [\[37,](#page-12-3) [32\]](#page-12-6).

Lemma 7. *Consider an* α*-convex truncated model in Definition [12.](#page-41-4) For any* K, K′ *two symmetric convex bodies of Gaussian volume* 1 − α, *it holds* ⟨LK, LK′ ⟩<sup>Q</sup> ≥ 1. *This is to say, and* α*-Convex Truncated Model satisfies Assumption [1](#page-5-1) for the trivial group.*

*Proof.* For any K, it holds LK(x) = 1(x ∈ K)/Q(K), x ∈ <sup>R</sup> <sup>n</sup>. Hence,

$$\langle L_K, L_{K'} \rangle = \frac{\mathbb{Q}(K \cap K')}{\mathbb{Q}(K)\mathbb{Q}(K')}.$$

But the so-called Gaussian correlation inequality for symmetric convex bodies in convex geometry [\[37\]](#page-12-3) states exactly that for any symmetric convex bodies K, K′ it holds Q(K ∩ K′ ) ≥ Q(K)Q(K′ ) yielding the result.

Now, for the α-Convex truncation models, the state-of-the-art polynomial-time algorithms require O(n/α<sup>2</sup> ) samples [\[15\]](#page-10-5), and the best known information-theoretic lower bound is Ω(n/α) samples [\[15\]](#page-10-5). Using the GFP-hardness to SQ-hardness framework we prove that for some prior on K, it is SQ-hard to distinguish with o˜(n/α<sup>2</sup> ) samples, providing evidence that the polynomial-time method from [\[15\]](#page-10-5) cannot be improved.

#### D.4.1 A new SQ lower bound

To apply our framework, we focus on the following prior on K, a variant of which has been studied in [\[15\]](#page-10-5) to prove their information-theoretic lower bound of Ω(n/α) samples. To define it we let

$$K = K_v = \{x \in \mathbb{R}^d : |\langle x, v \rangle| \leq \kappa\},$$

for any v ∈ Unif({−1/ √ d, 1/ √ d} d ). Here, we choose κ = κ(α, d) is such that the Gaussian measure of each K<sup>v</sup> is 1 − α. Then our prior is uniform among Kv, v ∼ Unif({−1/ √ d, 1/ √ d} d ). We refer to the α-convex truncation setting with this prior as the *"*α*-Slice Convex Truncation"* model.

We first point out that for any m = ω(n/α), detection with m samples is always possible in the α-Slice Convex Truncation model from a time-inefficient method. Indeed, one can brute-force search for some v ∈ {−1/ √ d, 1/ √ d} d for which it holds: for all i = 1, 2, . . . m, |⟨x<sup>i</sup> , v⟩| ≤ κ. Under P, there always exists such a vector v and hence the brute force search algorithm will find it with probability 1. Under Q though a direct union bund gives that such a v exists only with probability at most 2 d (1 − α) <sup>m</sup> = o(1) for any m = ω(d/α). Hence, the algorithm can detect with probability 1 − o(1). In that context, we prove the following result.

Theorem 12 (ρId-FP- and SQ-hardness of Convex Truncation). *Let* n ∈ <sup>N</sup> *growing and arbitrary* α = α<sup>n</sup> ∈ (0, 1)*. There exists a universal constant* C > 0 *and a prior* π *on the convex bodies* K *of Gaussian volume* 1 − α *such that for any* q ∈ N *with* q = e o(αn) *, the* α*-Convex Truncation model under* π *is* (q, Cn α<sup>2</sup> log(1/α) <sup>3</sup>/<sup>2</sup> log q )*-*ρId*-FP-hard.*

*In particular, for any constant* T > 0 *if* α = ω( (log n) T n ) *then the* α*-Convex Truncation model under* π *is* (e Θ((log n) <sup>T</sup> ) , Θ( <sup>n</sup> α<sup>2</sup> log(1/α) <sup>3</sup>/<sup>2</sup>(log n) <sup>2</sup><sup>T</sup> +1 ))*-SQ hard.*

Satisfyingly the proof of this result is also relatively short. The proof of this Theorem can be found in Appendix [D.](#page-35-0)

#### D.4.2 Proofs for Convex Truncation

*Proof of Theorem [12.](#page-42-2)* Observe that for L<sup>u</sup> := L<sup>K</sup><sup>u</sup> we have via standard Hermite expansion (identical to the argument in [\[15,](#page-10-5) Line (32), proof of Claim 24]),

$$\langle L_u, L_v \rangle_{\mathbb{Q}} = \frac{\mathbb{Q}(K_u \cap K_v)}{(1-\alpha)^2} = 1 + (1-\alpha)^{-2} \langle u, v \rangle^2 \left[ \sum_{i=1}^{\infty} f_{2i}^2 \langle u, v \rangle^{2(i-1)} \right], \quad (56)$$

where f<sup>i</sup> is the i-th Hermite weight of 1(x ∈ [−κ, κ]), x ∈ <sup>R</sup> for κ such that Φ(κ) = 1 − α/2 where Φ is the CDF of a standard Gaussian.

Now, conveniently, the authors [\[27\]](#page-11-15) have already studied the Hermite mass of indicators of symmetric intervals around 0. Indeed, applying [\[27,](#page-11-15) Lemma 27] for j = 2, θ = k imply that

$$f_2^2 = O(\kappa\phi(\kappa)^2),$$

where ϕ is the PDF of a standard Gaussian. But observe that by standard tail bounds κ = O( p log(1/α)) and from the Mill's ratio bound ϕ(κ) = Θ((1 − Φ(κ))κ). Combining the above we conclude

$$f_2^2 = O\left(\alpha^2 \log(1/\alpha)^{3/2}\right).$$

Parseval's identity gives P i>0 f 2 <sup>i</sup> = α(1 − α) ≤ α, and hence for some constant C > 0

$$\langle L_u, L_v \rangle_{\mathbb{Q}} \leq 1 + C \left( (1 - \alpha)^{-2} \langle u, v \rangle^2 \left( \alpha^2 \log(1/\alpha)^{3/2} + \langle u, v \rangle^2 \alpha \right) \right).$$

Now, notice that from [\(56\)](#page-42-3), ⟨Lu, Lv⟩<sup>Q</sup> is an increasing function of ⟨u, v⟩ 2 . Hence, for any q > 0 there exists δ0(q) > 0 such that {ρid(u, v) ≥ r(q)} = {⟨u, v⟩ <sup>2</sup> ≥ δ0(q)}. From Hoeffding's inequality we have that for some constant C ′ > 0 if δ = C ′ log q n then π 2 (⟨u, v⟩ <sup>2</sup> ≥ δ) ≤ q −2 . Hence δ0(q) ≤ δ = C ′ log q n .

Combining the above we have that for any q = e o(αn) ,

$$\begin{aligned}\mathbb{E}[\langle L_u, L_v \rangle_{\mathbb{Q}}^m \mathbf{1}(\langle u, v \rangle^2 \leq \delta_0)] &\leq \left[ 1 + C \left( (1 - \alpha)^{-2} \delta_0 (\alpha^2 \log(1/\alpha)^{3/2} + \delta_0 \alpha) \right) \right]^m \\ &\leq \left[ 1 + C \left( (1 - \alpha)^{-2} C' \frac{\log q}{n} (\alpha^2 \log(1/\alpha)^{3/2} + C' \frac{\log q}{n} \alpha) \right) \right]^m \\ &\leq \left[ 1 + 2C \left( C' \frac{\log q}{n(1 - \alpha)^2} (\alpha^2 \log(1/\alpha)^{3/2}) \right) \right]^m \\ &= O(1),\end{aligned}$$

as long as m = O(d/(α 2 log(1/α) 3/2 log q)). So we conclude the (q, Θ(n/(α 2 log(1/α) 3/2 log q))) ρid-FP-hard for any q = e o(αn) .

Now via an identical proof to [\[15,](#page-10-5) Theorem 23] we have for any m = o(n/α) that

$$\chi^2(\mathbb{P}^{\otimes m}, \mathbb{Q}^{\otimes m}) = O(1).$$

In particular, for any constant T > 0,

$$\chi^2(\mathbb{P}^{\otimes(\log n)^T}, \mathbb{Q}^{\otimes(\log n)^T}) = O(1).$$

Finally, notice that again since ⟨Lu, Lv⟩<sup>Q</sup> is a strictly increasing function of ⟨u, v⟩ 2 , and ⟨u, v⟩ is a sum of iid Rademacher random variables we conclude via standard Central Limit Theorem arguments that for any T > 0 there exists q = q(T) = e Θ((log n) <sup>T</sup> +1) for which for some r ′ = r ′ (T), r = r(T) > 0 it holds that π 2 (⟨u, v⟩ <sup>2</sup> ≥ r) = π 2 (ρid(u, v) ≥ r ′ ) = q −2 .

Hence, for any T > 0 we can apply our equivalence Theorem [3](#page-6-1) for mIT = (log n) T , q = q(T + 1) (so log q = Θ((log n) <sup>T</sup> +1)), and appropriate t = Θ((log n) T ) to conclude the (e Θ((log n) T ), Θ( <sup>n</sup> α<sup>2</sup> log(1/α) <sup>3</sup>/<sup>2</sup>(log n) <sup>2</sup><sup>T</sup> +1 ))-SQ hardness of the task.

## E Details on the GFP-hardness and FP-hardness separation

Below, we provide details on the counterexample described in Section [5.](#page-9-0)

*Proof of Lemma [3.](#page-9-2)* By definition, for any u ∈ {0, 1} n+1 ,

$$\begin{aligned} L_u(x) &= \prod_{i=0}^n \left( \mathbf{1}(x_i = 1) \cdot \frac{1/2 + r \cdot \frac{1-(1-\alpha) \cdot u_i}{2}}{1/2} + \mathbf{1}(x_i = -1) \frac{1/2 - r \cdot \frac{1-(1-\alpha) \cdot u_i}{2}}{1/2} \right) \\ &= \prod_{i=0}^n (1 + rx_i \cdot [1 - (1 - \alpha) \cdot u_i]). \end{aligned}$$

For any u, v ∈ {0, 1} <sup>n</sup>+1, the inner product ⟨Lu, Lv⟩ satisfies

$$\begin{aligned} \langle L_u, L_v \rangle &= \mathbb{E}_{x \sim \mathbb{Q}} \left[ \prod_{i=0}^n (1 + rx_i \cdot [1 - (1 - \alpha) \cdot u_i]) (1 + rx_i \cdot [1 - (1 - \alpha) \cdot v_i]) \right] \\ &= \prod_{i=0}^n \mathbb{E}_{x_i \sim \text{Rad}(1/2)} (1 + rx_i \cdot [1 - (1 - \alpha) \cdot u_i]) (1 + rx_i \cdot [1 - (1 - \alpha) \cdot v_i]) \\ &= \prod_{i=0}^n (1 + r^2 \cdot (1 - (1 - \alpha) \cdot u_i)(1 - (1 - \alpha) \cdot v_i)) \end{aligned}$$

Denote a<sup>i</sup> = 1 + r 2 · (1 − (1 − α) · ui)(1 − (1 − α) · vi). When u<sup>i</sup> = v<sup>i</sup> = 0, we have a<sup>i</sup> = 1 + r 2 ; when u<sup>i</sup> = v<sup>i</sup> = 1, a<sup>i</sup> = 1 + r 2 · α 2 ; when there is exactly one 1 and one 0 in u<sup>i</sup> , v<sup>i</sup> , we get a<sup>i</sup> = 1 + r 2 · α. We deduce that a<sup>i</sup> = 1 + r 2 · α <sup>u</sup>i+v<sup>i</sup> and the lemma follows.

Let us consider the m-sample version of the hypothesis testing problem. The null hypothesis is then Q⊗<sup>m</sup> and the alternative hypothesis is <sup>E</sup>u∼π<sup>P</sup> ⊗m u , where u is sampled from the following two-point prior π:

$$u = \begin{cases} (1, 0, \dots, 0), & \text{w.p. } \rho, \\ (0, 1, \dots, 1), & \text{w.p. } 1 - \rho. \end{cases} \quad (57)$$

We abbreviate these vectors as u<sup>1</sup> = (1, 0, . . . , 0) and u<sup>2</sup> = (0, 1, . . . , 1) for convenience. Using Lemma [3,](#page-9-2) it holds that

$$\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle = \prod_{i=0}^n (1 + r^2 \cdot \alpha^{u_i + v_i})^m. \quad (58)$$

Let's next show that this problem is GFP hard but FP easy. Note that ⟨Lu, Lv⟩ ≥ 1 for all u, v and therefore the model verifies Assumption [1](#page-5-1) for the trivial group.

Theorem 13. *For the two-point prior* π *in* [\(58\)](#page-44-1) *with* ρ = exp (−n <sup>ε</sup>/2)*, and for* r = n −1/2 *,* α = n −1+2ε *,* m = n <sup>1</sup>−<sup>ε</sup> *and* D = n ε *, where* ε > 0 *is any small constant, the following hold. The* m*-sample hypothesis testing problem* <sup>E</sup>u∼π<sup>P</sup> ⊗m u *versus* Q⊗<sup>m</sup> *is* (e D/2 , m, Θ(n −ε ))*-GFP hard but not* (n −1 , m, exp (Θ(n ε )))*-FP hard. Moreover, via our equivalence theorem the model is* (e n Θ(ε) , n<sup>1</sup>−Θ(ε) )*-SQ hard.*

*Proof of Theorem [13.](#page-44-2)* Let us first show it is FP easy. Define δ := δ(n −1/2 ) the supremum over δ such that π 2 (⟨u, v⟩ ≥ δ) ≥ 1/n. We observe when u ̸= v, then we must have ⟨u, v⟩ = 0 < δ by the choice of the two points prior with ⟨u1, u2⟩ = 0. Therefore,

$$\pi^2(u \neq v) = 2\rho(1 - \rho) \leq 2e^{-n^\varepsilon/2} \ll n^{-1} \leq \pi^2(\langle u, v \rangle \geq \delta). \quad (59)$$

$$\mathbb{E}_{u,v}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(\langle u, v \rangle < \delta)] \geq \mathbb{E}_{u,v}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(u \neq v)]$$

$$= \pi^2 [u \neq v] \cdot \mathbb{E}_{u,v} [\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \mid u \neq v]$$

When conditioned on u ̸= v, we get

$$\mathbb{E}_{u,v}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \mid u \neq v] = (1 + \alpha r^2)^{(n+1)m},$$

by applying Eq. [\(58\)](#page-44-1), with u<sup>i</sup> + v<sup>i</sup> = 1 for all 0 ≤ i ≤ n. Inserting the parameters stated in the lemma, we obtain

$$\begin{aligned}\mathbb{E}_{u,v}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(\langle u, v \rangle < \delta)] &\geq 2\rho(1-\rho) \cdot (1 + \alpha r^2)^{(n+1)m} \\ &\geq \exp\left(-\frac{1}{2}n^\varepsilon\right) \cdot \left(1 + n^{-2+2\varepsilon}\right)^{(n+1)n^{1-\varepsilon}} \\ &= \Omega(1) \cdot \exp\left(-\frac{1}{2}n^\varepsilon\right) \cdot \exp\left(n^\varepsilon\right) \geq \Omega(1) \cdot \exp\left(\frac{1}{2}n^\varepsilon\right).\end{aligned}$$

This shows that under our parameter choice, the task is (n −1/2 , m, exp (Θ(n ε )))-FP easy.

Let us now show that this model is GFP hard. We will prove that the model is ρId-FP hard and conclude using the implication Theorem [2.](#page-6-0)1. Under the trivial group, we have ρId(u, v) = ⟨Lu, Lv⟩<sup>Q</sup> − 1. From Eq. [\(58\)](#page-44-1), the m-sample inner product of likelihood ratio is given for u = v = u<sup>1</sup> by

$$\langle L_{u_1}^{\otimes m}, L_{u_1}^{\otimes m} \rangle = (1 + \alpha^2 r^2)^m \cdot (1 + r^2)^{mn} \quad (60)$$

and for u = v = u<sup>2</sup> by

$$\langle L_{u_2}^{\otimes m}, L_{u_2}^{\otimes m} \rangle = (1 + \alpha^2 r^2)^{nm} \cdot (1 + r^2)^m. \quad (61)$$

Because α ≪ 1, it is not hard to notice

$$\langle L_{u_1}^{\otimes m}, L_{u_2}^{\otimes m} \rangle < \langle L_{u_2}^{\otimes m}, L_{u_2}^{\otimes m} \rangle < \langle L_{u_1}^{\otimes m}, L_{u_1}^{\otimes m} \rangle. \quad (62)$$

From the definition of π, it holds that

$$\pi^2(\{u = u_2, v = u_2\} \cup \{u \neq v\}) = 1 - e^{-n^\varepsilon} \geq 1 - q^{-2}, \quad (63)$$

using that we set q = exp (D/2). Combining with Eq. [\(62\)](#page-45-0), we conclude that the event {ρ(u, v) ≤ r(q)} ⊂ {u = u2, v = u2} ∪ {u ̸= v}. This allows us to estimate the upper bound as

$$\begin{aligned} \mathbb{E}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(r \leq r(q))] &\leq \mathbb{E}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(\{u = u_2, v = u_2\} \cup \{u \neq v\}] \\ &\leq (1 + \alpha^2 r^2)^{nm} \cdot (1 + r^2)^m. \end{aligned} \quad (64)$$

Inserting our choice of parameters, we obtain

$$\begin{aligned}\mathbb{E}[\langle L_u^{\otimes m}, L_v^{\otimes m} \rangle \cdot \mathbf{1}(r \leq r(q))] &\leq (1 + n^{-3+4\varepsilon})^{n^{2-\varepsilon}} \cdot (1 + n^{-1})^{n^{1-\varepsilon}} \\ &\leq \exp(n^{-1+3\varepsilon} + n^{-\varepsilon}) \leq 1 + 2n^{-\varepsilon}.\end{aligned}\tag{65}$$

Thus, the model is (e D/2 , m, Θ(n −ε ))-ρId-FP hard, and therefore (e D/2 , m, Θ(n −ε ))-GFP hard.

Finally, let us use the SQ-GFP equivalence in Theorem [3](#page-6-1) to show that the model is also SQ hard, with parameters q ′ = e n ε/2 and t = n ε/2 (where indeed t ≤ log(q)/ log(m) = Θ( ˜ n ε )). To apply the theorem, we need to compute the χ 2 -divergence. Denoting X = ⟨L ⊗4t u , L⊗4<sup>t</sup> v ⟩ with t = n ε/2 ,

$$\begin{aligned} \chi^2(\mathbb{P}^{\otimes 4t}, \mathbb{Q}^{\otimes 4t}) + 1 &= \pi^2(u = u_1, v = u_1) \cdot \mathbb{E}[X|u = u_1, v = u_1] + \pi^2(u \neq v) \cdot \mathbb{E}[X|u \neq v] \\ &\quad + \pi^2(u = u_2, v = u_2) \cdot \mathbb{E}[X|u = u_2, v = u_2] \\ &= (1 - \rho)^2 \cdot (1 + \alpha^2 r^2)^{4nt} \cdot (1 + r^2)^{4t} + \rho^2 \cdot (1 + \alpha^2 r^2)^{4t} \cdot (1 + r^2)^{4nt} \\ &\quad + 2\rho(1 - \rho) \cdot (1 + \alpha^2 r^2)^{(n+1)4t} \\ &\leq (1 + n^{-3+4\varepsilon})^{n^{1+\varepsilon}} \cdot (1 + n^{-1})^{n^\varepsilon} \\ &\quad + e^{-n^\varepsilon} \cdot (1 + n^{-3+4\varepsilon})^{n^\varepsilon} \cdot (1 + n^{-1})^{n^{1+\varepsilon/2}} + 2n^{-\varepsilon} \cdot (1 + n^{-2+3\varepsilon})^{2n^{1+\varepsilon}} \\ &\leq 1 + 4n^{-1+\varepsilon}. \end{aligned}$$

Thus, we obtain

$$m' = \frac{m}{(t(1 + \varepsilon)^{1/t} + \chi^2(\mathbb{P}^{\otimes 4t} \parallel \mathbb{Q}^{\otimes 4t}))(q')^{2/t}} = m^{1-\Theta(\varepsilon)},$$