# 

Peter Halmos 1 * **Julian Gold** 1 2 * Xinhao Liu 1 **Benjamin J. Raphael** 1

## Abstract

Optimal transport (OT) has enjoyed great success in machine learning as a principled way to align datasets via a least-cost correspondence, driven in large part by the runtime efficiency of the Sinkhorn algorithm (Cuturi, 2013). However, Sinkhorn has quadratic space and time complexity in the number of points, limiting scalability to larger datasets. Low-rank OT achieves linear complexity, but by definition, cannot compute a oneto-one correspondence between points. When the optimal transport problem is an assignment problem between datasets then an optimal mapping, known as the *Monge map*, is guaranteed to be a bijection. In this setting, we show that the factors of an optimal low-rank coupling co-cluster each point with its image under the Monge map. We leverage this invariant to derive an algorithm, Hierarchical Refinement (HiRef), that dynamically constructs a multiscale partition of each dataset using low-rank OT subproblems, culminating in the bijective Monge map. Hierarchical Refinement runs in log-linear time and linear space, retaining the advantages of low-rank OT while overcoming its limited resolution. We demonstrate the advantages of Hierarchical Refinement on several datasets, including ones containing over a million points, scaling full-rank OT to problems previously beyond Sinkhorn's reach.

## 1. Introduction

Optimal transport (OT) is a mathematical framework for comparing probability distributions µ and ν. Given a cost function c, the *Monge problem* is to find a mapping T transforming a distribution µ into ν (i.e. T♯µ = ν) with least-cost.

A relaxation of this problem, called the *Kantorovich prob-*
*Equal contribution 1Department of Computer Science, Princeton University 2Center for Statistics and Machine Learning, Princeton University. Correspondence to: Benjamin J. Raphael <braphael@princeton.edu>.

lem, instead seeks a least-cost coupling γ between µ and ν. In the Kantorovich formulation, mass splitting is allowed and thus a solution always exists; in contrast, a Monge map between µ and ν may not exist. When a Monge map T does exist, the solution to the Kantorovich problem is a coupling γ = (id × T)♯ µ supported on its graph, and the Monge and Kantorovich problems coincide (Brenier, 1991). When µ and ν are discrete uniform measures on n points the optimal transport problem reduces to an assignment problem. Classical algorithms such as the Hungarian algorithm and Network Simplex (Tarjan, 1997; Orlin, 1997), solve this in cubic time. The Sinkhorn algorithm (Cuturi, 2013) solves the entropy-regularized Kantorovich problem with quadratic runtime, greatly expanding the applicability of computational OT. However, the Sinkhorn algorihtm requires quadratic space to store the coupling γ. In recent years, OT has found numerous applications in machine learning and across science, including: domain adaptation (Courty et al., 2014; Solomon et al., 2015), selfattention (Tay et al., 2020; Sander et al., 2022; Geshkovski et al., 2023), computational biology (Schiebinger et al., 2019; Yang et al., 2020; Zeira et al., 2022; Bunne et al.,
2023; Halmos et al., 2025b; Klein et al., 2025), unpaired data translation (Korotin et al., 2021; De Bortoli et al., 2024; Tong et al., 2024; Klein et al., 2024), and alignment problems in transformers and large language models (Melnyk et al., 2024; Li et al., 2024). The *least-cost* principle of optimal transport is crucial for training high-quality generative models using Schrodinger bridges, flow-matching, ¨ diffusion models, or neural ordinary differential equations (Finlay et al., 2020; Tong et al., 2024; De Bortoli et al., 2024; Kornilov et al., 2024; Klein et al., 2024). These models typically require millions to hundreds of millions of data-points to achieve high-performance at scale (Ramesh et al., 2021), limiting the scope of OT for generative modeling. As modern datasets grow to have tens of thousands or even millions of points, the quadratic space and time complexity of Sinkhorn becomes increasingly prohibitive. This limitation is widely recognized in the machine learning literature, with (De Bortoli et al., 2024) noting that the quadratic complexity of optimal transport renders its application to modern datasets on the order of millions of points impractical. A number of approaches have been proposed to address scal1 ing OT to massive datasets which avoid instantiating a full coupling matrix. Mini-batch OT (Genevay et al., 2018) improves scalability, but incurs significant biases (Sommerfeld et al., 2019; Korotin et al., 2021; Fatras et al., 2021a) as each mini-batch alignment is a poor representation of the global coupling. Multiple works have investigated the theoretical properties of mini-batch estimators of the coupling (Fatras et al., 2020; 2021b), while others have attempted to mitigate this bias using partial or unbalanced OT that allows mass variation between mini-batches (Nguyen et al., 2022a; Fatras et al., 2021a). However, these approaches introduce additional hyperparameters to control the degree of unbalancedness, and ultimately remain biased, local approximations of the global coupling. Neural optimal transport methods (Makkuva et al., 2020; Bunne et al., 2023; Fan et al., 2023; Korotin et al., 2023; Buzun et al., 2024), parametrize the Monge map as a neural network instead of materializing a quadratic coupling matrix. However, these methods have noted limitations recovering faithful maps (Korotin et al., 2021).

Another approach to improve space complexity of OT is to introduce a *low-rank* constraint on the coupling matrix in the Kantorovich problem. This has been done by parameterizing the coupling through a set of low-rank factors (Scetbon et al., 2021; 2022; Scetbon & Cuturi, 2022; Scetbon et al., 2023; Halmos et al., 2024) or by using a proxy objective for the low-rank problem, factoring the transport through a small number of anchor points (Forrow et al., 2019; Lin et al., 2021). For a given rank r these approaches have O(nr) space complexity, enabling *linear* time and space scaling. Low-rank OT has been used successfully on datasets on the order of 105samples with ranks on the order of 101
(Scetbon et al., 2023; Halmos et al., 2024; 2025a; Klein et al., 2025), but computing *full-rank* couplings between datasets of sizes on the order of 105and greater has not yet been accomplished. Contributions We introduce Hierarchical Refinement (HiRef), an algorithm to scalably compute a full-rank alignment between two equally-sized input datasets X and Y by solving a hierarchy of low-rank OT sub-problems. The success of this refinement is driven by a theoretical result, Proposition 3.1, stating that factors of an optimal low-rank coupling between X and Y co-cluster points X with their image under the Monge map. We use Proposition 3.1 recursively to obtain increasingly fine partitions of X and Y. At each scale, the solutions to low-rank OT sub-problems are bijections between the partitions of X and Y. Iterating to the finest scale gives a bijection between X and Y. Hierarchical Refinement constructs a *multiscale partition* of each dataset, and thus is related to (Gerber & Maggioni, 2017), which introduced a general framework for multiscale optimal transport using such partitions, and the earlier work of (Merigot ´ , 2011). Unlike (Merigot ´ , 2011; Gerber & Maggioni, 2017), Hierarchical Refinement (i) does not assume multiscale partitions for each dataset are given, instead constructing them on the fly; and (ii) operates intrinsically to the data, without a mesh or anchor points in the ambient space of the data, avoiding the curse of dimensionality. We demonstrate that Hierarchical Refinement computes OT maps efficiently in high-dimensional spaces, often matching or even outperforming Sinkhorn in terms of primal cost. Moreover, HiRef has linear space complexity and time complexity scaling log-linearly in the dataset size. Unlike low-rank OT, Hierarchical Refinement places X and Y in bijective correspondence. Hierarchical Refinement scales to over a million points, enabling the use of OT on massive datasets without incurring the bias of mini-batching.

## 2. Background And Related Work

Suppose X = {xi}
n i=1 and Y = {yj}
m j=1 are datasets in the same metric space (X , dX ). Let c : *X × X →* R+
be a cost function. This cost c is often assumed to satisfy strict convexity or to be a metric. Datasets X and Y are represented as discretely supported probability measures µ =Pn i=1 aiδxiand ν =Pm j=1 bj δyjfor probability vectors a ∈ ∆n and b ∈ ∆m. Throughout, ∆k denotes the k*-simplex* {p ∈ R
k+ :Pi pi = 1}, the set of probability vectors of length k. Monge Problem Optimal transport has its origin in the Monge problem (Monge, 1781), concerned with finding an optimal map T : X → Y pushing µ forward to ν:

$${\rm M}_{c}(\mu,\nu)=\min_{T:T_{\sharp}\mu=\nu}\mathbb{E}_{\mu}c(x,T(x))\,.\tag{1}$$

Above, T♯µ is the pushforward of µ under T, the measure on Y with T♯µ(B) := µ(T
−1(B)) for any (measurable)
set B ⊂ Y. In general, a Monge map may not exist (e.g. if *m > n*). However, when |X| = |Y| = n and a, b are uniform then the Monge problem becomes the assignment problem and has a bijective solution (Thorpe, 2018).

Kantorovich Problem The *Kantorovich problem* (Kantorovich, 1942) was introduced as a relaxation of the Monge problem. In contrast to the Monge problem, the Kantorovich problem allows mass-splitting and a solution is always guaranteed to exist. Define the *transport polytope* Πa,b as the following set of coupling matrices

$$\Pi_{\bf a,b}:=\left\{{\bf P}\in\mathbb{R}_{+}^{n\times m}:{\bf P1}_{m}={\bf a},{\bf P}^{\top}{\bf1}_{n}={\bf b}\right\},\tag{2}$$

respectively with left (or "source") marginal a and with right (or "target") marginal b. For the cost c(·, ·), define the cost matrix C by Cij = c(xi, yj ). In this discrete setting, the Kantorovich problem seeks a least cost coupling matrix P ∈ Πa,b between the probability vectors a, b associated to each measure *µ, ν*:

$$\mathrm{W}_{c}(\mu,\nu)=\operatorname*{min}_{{\bf P}\in\Pi_{{\bf n},{\bf b}}}\langle{\bf C},{\bf P}\rangle_{F}\,.$$
⟨C, P⟩F . (3)
The optimal value Wc(*µ, ν*) of (3) is called the c-
Wasserstein distance between µ and ν. Sinkhorn Algorithm and the ϵ**-schedule** The Sinkhorn algorithm (Cuturi, 2013) relaxes the classical linearprogramming formulation of optimal transport by solving an entropy regularized version of (3),

$$\mathrm{W}_{\epsilon}(\mu,\nu):=\operatorname*{min}_{{\bf P}\in\Pi_{\bf n,b}}\langle{\bf C},{\bf P}\rangle_{F}-\epsilon H({\bf P}),$$

where H(P) := −Pij Pij (log Pij − 1) is the Shannon entropy, and the parameter ϵ > 0 is the regularization strength. The Sinkhorn algorithm improved the O(n 3log n)
time complexity of classical techniques used for OT such as the Hungarian algorithm (Kuhn, 1955) and Network Simplex (Orlin, 1997; Tarjan, 1997) to O(n 2log n) (Luo et al., 2023). As ϵ ↓ 0, the optimal coupling P⋆,ϵ for (4)
converges to a sparse optimal coupling for (3) at an extremal point of the transport polytope (c.f. (Peyre & Cu- ´ turi, 2019)). However, the number of iterations required scales as poly(1/ϵ), diverging as ϵ decreases. A technique used to improve this scaling is the ϵ-schedule, an adaptive, monotone-decreasing and step-dependent set of entropy parameters ϵ1 > ϵ2 > · · · > ϵtfin . This anneals Problem 4 from high-entropy to low-entropy, gradually driving a dense initial condition to a sparse solution with a log (1/ϵ) rate
(Chen et al., 2023).

Low-rank Optimal Transport The nonnegative rank rk+(M) of a nonnegative matrix M ≽ 0 is the smallest number of nonnegative rank-1 matrices summing to M;
i.e. rk+(M) is the smallest integer z such that there exist nonnegative vectors q1*, . . . ,* qz ≽ 0 and r1*, . . . ,* rz ≽ 0 satisfying M =Pz i=1 qir
⊤
i. Let Πa,b(r) := {P ∈ Πa,b :
rk+(P) = r} be the set of rank-r couplings. The low-rank Wasserstein problem for general cost matrix C is:

$$\mathbf{P}^{*}=\operatorname*{arg\,min}_{\mathbf{P}\in\Pi_{\mathbf{n},\mathbf{b}}(r)}\langle\mathbf{C},\mathbf{P}\rangle_{F}\,.$$
From (Cohen & Rothblum, 1993), each P ∈ Πa,b(r) may
be decomposed as  $$\mathbf{P}=\sum_{i=1}(1/\mathbf{g}_{i})\mathbf{Q}_{.,i}\mathbf{R}_{.,i}^{\top}:=\mathbf{Q}\mathrm{diag}(1/\mathbf{g})\mathbf{R}^{\top},\tag{6}$$

where g ∈ ∆r, Q ∈ Πa,g and R ∈ Πb,g. This factorization was introduced to optimal transport by (Scetbon et al., 2021) in the context of the general low-rank problem (5). The factors Q and R constitute co-clusterings of datasets X and Y onto the *same* set of r components. Other factorizations have recently been proposed (Halmos et al., 2024), using Q, R and an intermediate latent coupling T to solve (5) where X and Y have r1 and r2 components, respectively.

$$({\mathfrak{I}}{\mathfrak{J}})$$

Hierarchical and Multiscale Approaches to OT Hierarchical optimal transport (Schmitzer & Schnorr ¨ , 2013) is a variant of OT modeling data and transport at two scales, using Wasserstein distances as the coarse-scale ground costs. It has been applied to document representation (Yurochkin et al., 2019), domain adaptation (El Hamri et al., 2022), sliced Wasserstein distances (Bonneel et al., 2015; Nguyen et al., 2022b) and to give a discrete formulation of transport between Gaussian mixture models (Chen et al., 2018; Delon & Desolneux, 2020). These works build interpretable, coarse-grained structure into a single coupling, rather than solving for a sequence of couplings at progressively finer scales as in the present work.

$$(4)$$

Multiscale approaches to OT generalize hierarchical OT to a progression of scales. Building on the semidiscrete approach of (Aurenhammer et al., 1998), (Merigot ´ , 2011) uses Lloyd's algorithm to progressively coarse-grain the target measure. More recently, using a regular family of multiscale partitions (Definition C.3) on each dataset, (Gerber & Maggioni, 2017) formalize a general hierarchical approach to the Kantorovich problem (3). They propose: (i) solving a Kantorovich problem between the coarsest partitions of X and Y in their respective multiscale families; and (ii) propagation of the optimal coupling at scale t ∈ {1*, . . . , κ* − 1} to initialize the optimization at scale t + 1. They take as input a chain of partitions and measures across scales (X
(1), µ1) *→ · · · →* (X
(κ), µκ) and
(Y
(1), ν1) *→ · · · →* (Y
(κ), νκ) where each dataset X, Y is identified with the trivial partitions X
(κ) = {{x} : x ∈ X}
and Y
(κ) = {{y} : y ∈ Y}. At the finest scale κ, (Gerber
& Maggioni, 2017) recover the original datasets and a near optimal coupling for (3). A naive implementation of the above idea requires quadratic memory complexity, but (Gerber & Maggioni, 2017) propose several propagation strategies to mitigate this, following (Glimm & Henscheid, 2013; Oberman & Ruan, 2015; Schmitzer, 2016). These strategies use the optimal coupling at scale t to restrict the support of the coupling computed at the next scale using local optimality criteria. In the next section, we give our own such criterion, Proposition 3.1.

$$({\boldsymbol{5}})$$

## 3. Hierarchical Refinement 3.1. Low-Rank Optimal Transport Co-Clusters Source-Target Pairs Under The Monge Map

We first show that under a few assumptions, the optimal lowrank factors (Q⋆, R⋆) for a *variant* of the low-rank Wasserstein problem (5) have qualities suited to our refinement strategy. Specifically, we parameterize low-rank couplings P of rank-r using the factorization P = Qdiag(1/g)R⊤ of
(Scetbon et al., 2021), fixing g ∈ ∆r to be uniform. Define the following variant of (5):

$$\begin{array}{r l}{{\left(\mathbf{Q}^{\star},\mathbf{R}^{\star}\right)=\arg\operatorname*{min}_{\left(\mathbf{Q},\mathbf{R}\right)}\left\langle\mathbf{C},\mathbf{Q}\mathrm{diag}(1/\mathbf{g})\mathbf{R}^{\top}\right\rangle_{F}}}&{{}}\\ {{\mathrm{s.t.}}}&{{\mathbf{Q}\in\Pi_{\mathbf{a},\mathbf{g}},\,\mathbf{R}\in\Pi_{\mathbf{b},\mathbf{g}},\,\mathbf{g}=(1/r)\mathbf{1}_{r}}}\end{array}$$

Proposition 3.1 below is the main structural result behind Hierarchical Refinement. It says that when optimal Q⋆and R⋆for (7) correspond to hard-clusterings (partitions) of each dataset, given by clustering functions q
⋆: X → [r],r
⋆:
Y → [r], one has q
⋆ = r
⋆ ◦ T
⋆, where T
⋆is a Monge map.

Proposition 3.1 (Optimal low-rank factors co-cluster Monge pairs). Let X, Y ⊂ R
d with |X| = |Y| = n, with cost matrix C that is strictly r-Monge separable (Definition *B.2).*
Let a, b ∈ ∆n *be uniform so that a Monge map* T
⋆: X → Y
exists. If (Q⋆, R⋆) *are minimizers of* (7) and correspond to clustering functions q
⋆: X → [r],r
⋆: Y → [r]*, then for all* x ∈ X *one has* q
⋆(x) = r
⋆(T
⋆(x)).

The proof of Proposition 3.1 is in two steps. First, we use the existence of a Monge map and its coupling P†to permute the cost C to cost C†(Definition B.1) for which the identity matrix is a Monge map. Second, supposing that strict r-Monge separability (Definition B.2) holds, we show the solution to Problem 7 with cost C†is symmetric, so that minQ,R∈Πa,g⟨C†, QR⊤⟩F = minQ∈Πa,g⟨C†, QQ⊤⟩F .

Returning to the coordinate frame of the original cost C, we find that Q = P†R, implying Proposition 3.1. We note that when r = 2, optimal Q, R are hard-partitions (Lemma B.5) automatically satisfying one of the assumptions of Proposition 3.1.

## 3.2. Hierarchical Refinement Algorithm

The Hierarchical Refinement algorithm (Algorithm 1) uses Proposition 3.1 to guarantee that each low-rank step coclusters the datasets optimally, in that x and T
⋆(x) are assigned the same label by q
⋆and r
⋆. Using the same label set to partition X and Y automatically places the blocks of each partition in bijective correspondence. One then recurses on each pair of corresponding blocks (which we call a *co-cluster*) at the previous scale, until all blocks have size one. This guarantee holds despite that optimal (Q⋆, R⋆) for (7) may not constitute an optimal triple (Q⋆, R⋆, g
⋆) for the original low-rank problem (5) under the (Scetbon et al., 2021) factorization. A hierarchy-depth κ denotes the total number of times Algorithm 1 refines the initial trivial partitions {X}, {Y}. The effective rank at scale t is ρt := Qts=1 rs, given rankannealing schedule (r1, r2*, . . . , r*κ) for which ρκ divides n. The base rank is rbase =n ρκ
. Note that n/ρt is also the size of each partition at scale t: n/ρt = |X
(t)| = |Y
(t)|, and that any sequence of any factorization of n corresponds to a rank-annealing schedule.

Proposition 3.2. For any n*, there exists a rank-schedule*
(r1, · · · , rκ) factorizing n such that all partitions of Algorithm 1 *at level* t ∈ [0 : κ − 1] satisfy strict rt+1-Monge separability (Definition *B.2). Let* LROT denote an optimal rank-r solver for (7) over hard-partitions. For any satisfying rank-schedule, the map returned by Algorithm 1 *is optimal* and supported on the graph of the Monge map T
⋆.

Proof. Existence follows from the trivial (r1) = (n) rankschedule. For any schedule (r1, · · · , rκ) satisfying Monge separability, applying the invariant of Proposition 3.1 inductively on t to level κ yields n tuples {(x, T ⋆(x))} containing each x ∈ X and its image T
⋆(x) under the Monge map.

If the black-box subroutine LROT in Algorithm 1 solves (7) optimally, then Hierarchical Refinement is guaranteed to recover a Monge map. In practice, we implement LROT using the low-rank solver (Halmos et al., 2024) and enforce that inner marginal g is uniform.

Let Γt,q denote the q-th co-cluster at scale t generated by Hierarchical Refinement:

$$\Gamma_{t,q}:=\left\{({\bf x},{\bf y}):{\bf x}\in{\sf X}_{q}^{(t)},\,{\bf y}\in{\sf Y}_{q}^{(t)}\right\}\,,\qquad(8)$$

where X
(t) = {X
(t)
q }
ρt q=1, Y
(t) = {Y
(t)
q }
ρt q=1, and define the co-clustering Γt at scale t by:

$$\Gamma_{t}:=\Big\{({\mathsf{X}}_{q}^{(t)},{\mathsf{Y}}_{q}^{(t)})\Big\}_{q=1}^{\rho_{t}}\ .$$

At scale t ∈ [κ], Hierarchical Refinement refines Γt to Γt+1 by running a rank rt+1 low-rank optimal transport problem between uniform gt+1 = (1/rt+1)1rt+1 and measures supported on each pair (X
(t)
q , Y
(t)
q ) in Γt for q ∈ [ρt], yielding factors specific to this q ∈ [ρt]: For each q ∈ [ρt] we use the Q, R from (9) to co-cluster X
(t)
q with Y
(t)
q using rt+1 labels. Within this pair, each xi ∈ X
(t)
q is assigned a label z ∈ [rt+1] by taking the argmax over the i-th row of Q, and likewise each yj ∈ Y
(t)
q is assigned the argmax over the j-th row of R. This corresponds to the Assign step in Algorithm 1, and coincides with the hard assignment of q
⋆and r
⋆for an optimal (Q∗, R∗)
(Lemma B.5).

The uniform constraint g = 1rt+1 /rt+1 in (7) enforces an even split of the dataset, which by Lemma B.5 ensures a partition at optimality (for rt = 2). Repeating for all q ∈ [ρt], one obtains a co-clustering with rt+1 components within each co-cluster at the previous scale, leading to a total of ρt+1 = rt+1ρt co-clusters at scale t + 1 (Fig. 1). If the base-case rank rbase is one, Algorithm 1 returns a bijection between X and Y as a collection of n tuples. Note that Hierarchical Refinement defines an implicit hierarchy of block-couplings at each scale t.

Definition 3.3 (Hierarchical block-coupling). For each scale t ∈ [κ], given the Hierarchical Refinement co-cluster partition Γt, the *hierarchical block-coupling* at scale t is defined by the matrix

$$\mathbf{P}_{ij}^{(t)}:=\frac{\rho_{t}}{n^{2}}\sum_{q=1}^{\rho_{t}}\delta_{(\mathbf{x}_{i},\mathbf{y}_{j})\in\Gamma_{t,q}},\tag{10}$$

Without loss of generality, P(t) may be block diagonalized into ρt square blocks, as discussed in Appendix B (see Equation (S13)). By Proposition 3.1, for any rank-schedule
(rj )
κ j=1 satisfying Monge separability, the final P(κ)corresponds to an optimal coupling supported on the graph of the Monge map T
⋆, P(κ):= (id × T
⋆)♯µX. While these intermediate couplings are never instantiated, one can still use them to define a transport cost ⟨C, P(t)⟩ at each scale.

In Appendix B.8, we show the following bounds on the cost difference across scales.

Proposition 3.4. Let c(·, ·) be a strictly-convex and Lipschitz cost function, let (r1, r2, · · · , rκ) be a rank-schedule, and let P(t) *denote the coupling defined in* (10), obtained from step t of Algorithm 1. Define ∆t,t+1 = ⟨C, P(t)⟩F − ⟨C, P(t+1)⟩F *. Then,*

$\rho_{t}$)$F$. Then,  $$0\leq\Delta_{t,t+1}\leq\|\nabla c\|_{\infty}\frac{1}{\rho_{t}}\sum_{q=1}^{\rho_{t}}\mbox{diam}(\Gamma_{t,q})\;,\tag{11}$$
where q indexes co-clusters Γt,q at scale t*, defined in* (8).

$$(9)$$
$${\bf\Phi}({\bf\Phi})\,,{\bf g}_{t+1}\,\Big)\,.$$

Thus, the lower-bound implies that each step of refinement improves the coarse partition, and the upper-bound implies that the difference in solution value is bounded above by a factor depending on the Lipschitz constant and the mean diameter of the coarse partitions at each level t. The proof of Proposition 3.4 roughly follows that of Proposition 1 of (Gerber & Maggioni, 2017). In Remark B.9, we discuss how Proposition 3.4 compares, noting that our result makes fewer geometric assumptions on our multiscale partitions
(X
(t))
κ t=1 and (Y
(t))
κ t=1 and therefore does not quantify the rate of decay of diamΓt,q.

## 3.3. On The Rank-Annealing Schedule

As observed by (Forrow et al., 2019; Scetbon et al., 2021), rank behaves like a temperature parameter, inverse to the strength ϵ of entropy regularization. The correspondence between small ϵ and large rank implies that annealing in the parameter ϵ is, from the perspective of rank, analogous to initializing the optimization at a low-rank coupling, and then gradually increasing the rank constraint from low to full. In Hierarchical Refinement, this gradual rank increase is accomplished implicitly. At each scale t = 1*, . . . , κ* the implicit coupling P(t)is made explicit in the hierarchical block coupling defined in equation (10). A rank-annealing schedule (r1*, . . . , r*κ) describes the sequence of multiplicative factors by which the rank of this explicit coupling will increase

$\mathbb{L}(\mathsf{X},\mathsf{Y})$
Algorithm 1 Hierarchical Refinement Require: Data X , Y; **Low-rank OT solver** LROT(·);
Rank schedule (r1, r2*, . . . , r*κ); **Base rank** rbase (=1).
Initialize:
1: t ← 0, Γ0 ← { (X, Y)}
2: **while** ∃ (X
(t)
q , Y
(t)
q ) ∈ Γt **such that**
3: min{|X
(t)
q |, |Y
(t)
q |} > rbase do
4: Γt+1 ← ∅
5: for (X
(t) q , Y (t) q ) ∈ Γt do 6: if min{|X (t) q |, |Y (t) q |} ≤ rbase then 7: Γt+1 ← Γt+1 ∪ {(X (t) q , Y (t) q )} (t) q=1 |X (t) q | Px∈X (t) qδx (t) q=1 |Y (t) q | Py∈Y (t) qδy. 11: gt+1 ← (1/rt+1)1rt+1 12: (Q, R) ← LROT(µX (t) q , µY (t) q , gt+1) 13: for z = 1 → rt+1 do (t+1) z ← Assign(X (t), Q, z) (t+1) z ← Assign(Y (t), R, z) 16: Γt+1 ← Γt+1 ∪ { (X (t+1) z , Y (t+1) z )} 18: ▷ Assign(S,M, z) = {s ∈ S | arg maxz′ Msz′ = z}
8: **else**
9: µX
10: µY
14: X 15: Y 17: **end for** 19: **end if** 20: **end for** 21: t ← t + 1 22: **end while**
23: **Output:** Γκ = {(xi, T(xi))n
i=1} ▷ Mapped pairs.
$\mathbf{\hat{x}}_i=\{(\mathbf{x}_i)\}$
$$\mathbf{\Pi}_{=1}^{1}\}$$
$t\left|\cdot\right|$. 
$$(\mathbf{Q},\mathbf{R})\leftarrow\mathrm{LR}$$
(Q, R) ← LROT(µX
$$\Gamma(\mu_{\chi_{q}^{(t)}},\mu_{\chi_{q}^{(t)}})$$
(t) q
, gt+1). (9)
at successive scales. The partial products of these, denoted
(ρ1*, . . . , ρ*κ), are the ranks of the couplings P(1)*, . . . ,* P(κ).

Note that small values of ri generate coarse partitions of the points at the next scale, while large values of ri generate finer partitions at the next scale.

We now turn to the question of how to efficiently choose such a schedule under given memory constraints. For an integer n, Algorithm 1 has log-linear complexity for depth κ = logr n (Section 3.4). However, the large constants required by low-rank OT in practice encourage minimizing the number of calls to LROT as a subroutine, so that if memory permits, it may be advantageous to decrease the depth by storing couplings of higher rank. If desired, memory constraints can be enforced by imposing a maximum rank rmax ≥ rt for all t ∈ [κ] to ensure Hierarchical Refinement only requires O(nrmax) space at each step. Thus, we seek factorizations with *minimal* partial sums of ranks while remaining below a desired memory-capacity:

$$\min_{(r_{i})_{i=1}^{\kappa}}\sum_{j=1}^{\kappa}\rho_{j}\quad\mbox{s.t.}\quad\rho_{\kappa}=n,\quad r_{i}\leq r_{\max}\,.\tag{12}$$

The above optimization assumes a base-rank rbase of 1; we describe how to handle the general case in Appendix E.1.

Importantly, the recursive structure min(ri)
κ i=1 Pκ j=1 ρj =
min(ri)
κ i=1 r1 + r1Pκ j=2 Qj i=2 ri enables a dynamic programming approach to (12), storing a table of factors up to rmax to optimize (12) in O(rmaxκn) time. Assuming *κ, r*max are small constants chosen to ensure that all matrices can fit within memory, determining the optimal rank-schedule with respect to *κ, n, r*max is a simple lineartime procedure.

## 3.4. **Complexity And Scaling Of Hierarchical Refinement**

For two datasets X, Y of size n, the space complexity of Hierarchical Refinement is Θ(n), since at each level, one must store Γt which is a set of subsets of X and Y. To derive the time-complexity of Hierarchical Refinement, note that if n = r k, a rank-r schedule at each layer requires nr instances of LROT over rapidly decaying dataset sizes. The complexity of low-rank OT (Scetbon et al., 2021; 2022; Halmos et al., 2024) is linear (Kn) for a constant K = O(*BLrd*) dependent on B the number of inner Sinkhorn (Halmos et al., 2024) or Dykstra (Scetbon et al., 2021) iterations, L the number of mirror-descent steps, r the rank of the coupling, and d the rank of the factorization of the cost matrix C. In this setting, for n a power of r, the runtime of Algorithm 1 is given by the sum r 0Θ(n) + r 1Θ( n r
) + ... + r i−1Θ( n r i−1 ) = Θ(ndr logr n)
for i = logr n, achieving *linear* space with *log-linear* time for constant ranks r, d. In cases where the cost matrix does not admit a low-rank factorization C = UV⊤, i.e., when d = O(n), one requires Θ(n 2) space to store the cost matrix and Hierarchical Refinement exhibits time complexity O˜(n 2), as in Sinkhorn.

For kernel costs such as squared Euclidean cost, as noted in
(Scetbon et al., 2021), one may efficiently compute a (d+ 2) dimensional factorization where d is the ambient dimension, to achieve log-linear scaling with exact distances. We also use the sample-linear algorithm of (Indyk et al., 2019) to compute approximate factorizations for distances c(·, ·) satisfying metric properties such as the triangle inequality (e.g. Euclidean distance, see Appendix E.1). At each level, pairing such sample-linear approximations with each lowrank step only requires O(n logd n) time. We observe this scaling empirically, as reported in Fig. S2.

## 4. Experiments

We benchmark Hierarchical Refinement (HiRef) against the full-rank OT methods Sinkhorn (Cuturi, 2013), ProgOT (Kassraie et al., 2024), and mini-batch OT (Genevay et al., 2018; Fatras et al., 2020; 2021b). We additionally benchmark against the low-rank OT methods LOT (Scetbon et al., 2021) and FRLC (Halmos et al., 2024). We use the default implementations of Sinkhorn, ProgOT, and LOT in the high-performance ott-jax library (Cuturi et al., 2022). In particular, Sinkhorn is run with the default entropy regularization parameter of ϵ = 0.05. We also benchmark against the multiscale method MOP (Gerber & Maggioni, 2017), which requires multiscale partitions of the input datasets –
akin to a family of dyadic cubes across scales - to compute alignments. This leads to a transport cost that depends on the choice of this partition. For simplicity, we choose the default partitions of MOP which are computed from the GMRA (Geometric Multi-Resolution Analysis) R package.

Hierarchical-Refinement Map Sinkhorn Barycentric Map Optimal Map (Dual Revised Simplex)
a.

b.

## 4.1. Evaluation On Synthetic Datasets.

We first evaluate the performance of Hierarchical Refinement against optimal transport methods returning primal couplings, namely Sinkhorn (Cuturi, 2013) (as implemented in ott-jax (Cuturi et al., 2022)) and ProgOT (Kassraie et al., 2024). We evaluate the methods with respect to the Wasserstein-1 and Wasserstein-2 distance on an alignment of 1024 pairs of samples on the Checkerboard (Makkuva et al., 2020), MAFMoons and Rings (Buzun et al., 2024), and Half-Moon and S-Curve (Buzun et al., 2024) synthetic datasets (Fig. 3, Table S6). All methods are similarly effective at minimizing the primal OT cost ⟨C, P⟩F , with small absolute difference in cost between the final couplings. Hierarchical Refinement achieves slightly lower primal cost on 4 out of the 6 evaluations. Notably, there is a massive difference in the number of non-zero entries (defined as entries Pij > 10−8) in the couplings output by HiRef, Sinkhorn, and ProgOT (Table S3). Specifically, across the experiments HiRef outputs a bijection with exactly 1024 non-zero elements in the coupling matrix, equal to the number of aligned samples. In constrast, Sinkhorn and ProgOT output couplings with 624733 to 678720 and 271087 to 337258 non-zero entries.

We evaluate the scalability of Hierarchical Refinement relative to other full-rank solvers on varying numbers of samples from the Half Moon & S-Curve (Buzun et al., 2024) synthetic dataset. We vary the rank from 2 5 = 32 (64 points aligned) up to 2 20 = 1048576 points (2097152 points aligned) in R
2, the latter dataset of a size that is beyond the capabilities of current optimal transport solvers. We observe that Sinkhorn (Cuturi, 2013) and ProgOT - methods which produce dense mappings - require a coupling matrix with O(n 2) non-zero entries and thus run only up to 16384 points. HiRef yields solutions with comparable primal cost to ProgOT and Sinkhorn on the sample sizes where all methods run.

We also find that HiRef achieves an OT cost that is competitive with the dual revised simplex solver (Huangfu & Hall, 2018), a solver which only scales up to 512 points (Table S4). This solver computes an *optimal* coupling, unlike ProgOT and Sinkhorn which rely on entropic regularization. While we benchmark Sinkhorn in place of mini-batch OT on the synthetic datasets due to their limited complexity, we also evaluate the multi-scale method MOP on the 512 point instance (Table S4). Although MOP outputs a fast approximation to optimal transport, its primal cost on the Checkerboard (Makkuva et al., 2020) dataset is twice as high as that of the other methods, and it performs significantly worse on the MAF Moons & Rings and Half Moon & S-Curve datasets (Buzun et al., 2024). Lastly, we observe that Hierarchical Refinement scales to over a million points, two orders of magnitude greater than ProgOT and Sinkhorn, two full-rank OT methods that compute global alignments. We find HiRef scales linearly with the size of the problem instance (Fig. S2a) in contrast to the quadratic scaling in time complexity of Sinkhorn (Fig. S2b).

## 4.2. Large-Scale Matching Problems And Transcriptomics

Recently, optimal transport has been applied to single-cell and spatial transcriptomics datasets to compute couplings between cells taken from different timepoints from developmental processes or perturbations (Schiebinger et al., 2019; Lavenant et al., 2024; Bunne et al., 2022; Huizing et al., 2024; Halmos et al., 2025b; Klein et al., 2025). However, the size of current datasets (Chen et al., 2022) (>100k cells)

| Table 1. Cost Values ⟨C, P⟩F Across Later Embryonic Stages Method E12-13.5 E13-14.5 E14-15.5 E15-16.5 HiRef 14.35 13.78 14.29 12.79 Sinkhorn - - - - MB 128 14.86 14.14 14.75 13.32 MB 1024 14.45 13.86 14.43 12.91 FRLC 15.47 14.64 15.51 14.00   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

has exceeded the capacity of existing full-rank solvers, requiring low-rank approximations of the coupling (Scetbon et al., 2023; Klein et al., 2025; Halmos et al., 2025a) to produce alignments. We evaluate whether the full-rank solver of Hierarchical Refinement exhibits competitive alignments for such datasets. Specifically, we analyze the mouse organogenesis spatiotemporal transcriptomic atlas (MOSTA) datasets, which include spatial transcriptomics data from mouse embryos at successive 1-day time-intervals with increasing number n of cells at each stage: E9.5 (n = 5913), E10.5 (n = 18408), E11.5 (n = 30124), E12.5 (n = 51365), E13.5 (n = 77369), E14.5 (n = 102519), E15.5 (n = 113350), and E16.5 (n = 121767). For the cost we use the Euclidean distance Cij = ∥xi − yj∥2 in 60-dimensional PCA space of expression vectors, so xi, yj ∈ R
60.

Sinkhorn and ProgOT are unable to produce alignments for the stages beyond E10.5 (n = 18408 cells), whereas HiRef, the low-rank solvers, and mini-batch OT (batchsizes B = 128 to B = 2048) are able to continue scaling to
> 105(Table 1, Table S6). We observe that the Kantorovich cost of HiRef is consistently lower than all other methods for all timepoints (Table 1, Table S6). HiRef achieves a substantially lower cost than the lowrank solvers FRLC and LOT for rank r = 40, even though HiRef relies on low-rank optimal transport (FRLC) as a subroutine. This result underscores the empirical trend observed in Fig. S3, where the refinement step of HiRef progressively decreases the primal cost of coarser low-rank couplings (Proposition 3.4). While the mini-batch solvers exhibit competitive scaling up to the last pair, the primal cost of mini-batch is higher for all tested batch-sizes (Table S6).

Unlike HiRef, mini-batch OT does not compute a global alignment and exhibits batch-size dependent error.

## 4.3. Merfish Brain Atlas Alignment

We ran HiRef on two slices of MERFISH Mouse Brain Receptor Map data from Vizgen to test whether HiRef can produce biologically valid alignments using the *only* spatial densities of each tissue. These spatial transcriptomics data consist of spatial and gene expression measurements at individual spots in three full coronal slices across three biological replicates. Our "source" dataset (X1, S
1) is replicate 3 of slice 2, while our "target" dataset (X2, S
2) is replicate 2 of slice 2, following the expression transfer task described (Clifton et al., 2023) between these two slices. Each dataset has roughly 84k spots, where memory constraints prohibit instantiation a full-rank alignment as a matrix. Thus, solvers such as Sinkhorn (Cuturi, 2013) and ProgOT (Kassraie et al., 2024) are unable to run on the dataset. We use only spatial information when building a map between the two slices, using the spatial Euclidean cost Cij := ∥s 1 i − s 2 j∥2, after registering spatial coordinates S
1 = {s 1 i}
n i=1 and S
2 = {s 2 i}
n i=1 with an affine transformation. We gauged the quality of the HiRef alignment (Fig. 4a), using gene expression abundances of five
"spatially-varying" genes. Specifically, we observe that expression vector v 1 of gene *Slc17a7* in the source slice (
Fig. 4b) when transferred to target slice through the bijective mapping output by HiRef, denoted as vˆ (Fig. 4c), closely matches the observed expression vector v 2 of *Slc17a7* in the target slice (Fig. 4d) with cosine similarity equal to 0.8098. For genes *Slc17a7*, Grm4, Olig1, Gad1, *Peg10*, the corresponding cosine similarities between the transferred and observed expression vectors are 0.8098, 0.7959, 0.7526, 0.4932, 0.6015, respectively. For comparison, we also ran the low-rank methods FRLC (Halmos et al., 2024) and LOT (Scetbon et al., 2021) with and without subsampling, reporting their best scores, as discussed in Section D.3. For the gene *Slc17a7*, FRLC's cosine similarity was 0.2373, while LOT's cosine similarity was 0.3390. For all five genes Slc17a7, Grm4, Olig1, Gad1, *Peg10*, FRLC's scores were (0.2373, 0.2124, 0.1929, 0.0963, 0.1550, respectively, while LOT's scores were 0.3390, 0.2712, 0.3186, 0.1666, 0.1080. Across all five genes HiRef's scores were at least twice those of FRLC or LOT (Table S7) with gene abundances shown in Fig. S1. On the same task, we compared against MOP, the method of (Gerber & Maggioni, 2017), whose scores for the five genes were: (0.5211, 0.4714, 0.5972, 0.3571, 0.2719). Finally, we also benchmarked against mini-batch OT using batch sizes ranging from 128 to 2048 in powers of two, whose best scores (0.7434, 0.7822, 0.7056, 0.4912, 0.5683) were more comparable to that of the performance of HiRef. Across all methods and genes compared in Table S7, HiRef had greatest cosine similarity scores in the expression transfer task, while also having lowest transport cost. Further experimental details are in Section D.3.

## 4.4. Imagenet Alignment

We demonstrate the scalability of Hierarchical Refinement on a large-scale and high-dimensional dataset by aligning 2048-dimensional embeddings of 1.281 million images from the ImageNet ILSVRC dataset (Deng et al., 2009;

a. b. c. d.

0 *Slc17a7* counts 30

| Table 2. Cost Values ⟨C, P⟩F for ImageNet (Deng et al., 2009; Russakovsky et al., 2015) Alignment Task. Method HiRef MB 128 MB 256 MB 512 MB 1024 FRLC OT Cost 18.97 21.89 21.11 20.34 19.58 24.12   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Russakovsky et al., 2015). Each image is embedded using using the ResNet50 architecture (He et al., 2016), and we construct two datasets, X and Y, by taking a random 50:50 split of the embedded images. We align X and Y using HiRef, FRLC, and mini-batch OT with batch-sizes ranging from B = 128 to B = 1024. ProgOT, Sinkhorn, and LOT could not be run on the datasets due to memory constraints. HiRef yielded a primal OT cost of 18.974, while FRLC (Halmos et al., 2024) solution had a primal OT cost of 24.119 for rank r = 40 and mini-batch OT has costs of 21.89 (B = 128) to 19.58 (B = 1024) (Table 2).

## 5. Discussion

Hierarchical Refinement computes the Monge map between large-scale datasets in linear space, but has several limitations. First, we currently assume that the datasets X and Y have the same number of samples. In many machine learning applications, this is not a limiting factor, as one generally seeks to pair an equal number of source points x to target points y. Second, while Hierarchical Refinement scales linearly in space and log-linearly in time, it still involves a constant dependent on the low-rank OT subprocedure used - this underscores the need to accelerate and stabilize low-rank OT solvers further (Scetbon & Cuturi, 2022; Halmos et al., 2024). Finally, while Hierarchical Refinement guarantees an optimal solution given an optimal black-box low-rank solver (Proposition 3.1), the low-rank solvers (Scetbon et al., 2022; Halmos et al., 2024) used in practice are not necessarily optimal, owing to the nonconvexity of low-rank problems.

Optimal transport has been successfully applied in deep learning frameworks, such as OT flow-matching (Tong et al., 2024), computer vision and point cloud registration, (Yu et al., 2021; Qin et al., 2022), among many others. The mini-batch procedure used to train many of these methods involves sampling two datasets XB ∼ µ and YB ∼ ν with batch-size B and aligning them with Sinkhorn at every training iteration. HiRef suggests an alternative approach: one can precompute millions of *globally aligned* pairs and then sample XB ∼ µ and the optimal mapping T(XB) ∼ ν by indexing into these precomputed pairs. This approach applies to any loss function dependent on an OT alignment. Hierarchical Refinement may also be useful in neural OT approaches which learn a continuous Monge map between the densities of two datasets. For example, (Seguy et al.,
2018) minimize a loss minθ 1 2 Eµ∥Tθ(xi) − T(xi)∥
22 between a neural network Tθ with parameters θ and a Monge map T over samples xi ∼ µ (Remark B.11). Thus, the procedure outlined above may be used to directly regress a neural network Tθ on the Monge map T without the bias of mini-batching or entropy.

## 6. Conclusion

We introduce Hierarchical Refinement (HiRef), an algorithm to solve optimal transport with linear complexity in the number of points, making sparse, full-rank optimal transport feasible for large-scale datasets. Our algorithm leverages that low-rank optimal transport co-clusters points with their image under the Monge map, refining bijections between partitions of each dataset across a hierarchy of scales, down to a bijective Monge map between the datasets at the finest scale. Hierarchical Refinement couplings achieve comparable primal cost to couplings obtained through full-rank entropic solvers, and scales to datasets with over a million points, opening the door to applications previously infeasible for optimal transport.

## Acknowledgements Impact Statement Code Availability References

using neural optimal transport. *Nature Methods*, 20(11): 1759–1768, 2023.

We thank Henri Schmidt for many helpful conversations. This research was supported by NIH/NCI grant U24CA248453 to B.J.R. J.G. is supported by the Schmidt DataX Fund at Princeton University made possible through a major gift from the Schmidt Futures Foundation.

Buzun, N., Bobrin, M., and Dylov, D. V. Expectile regularization for fast and accurate training of neural optimal transport. In Advances in Neural Information Processing Systems, volume 37, pp. 119811–119837, 2024. URL https://openreview.net/forum? id=4DA5vaPHFb.

Chen, A., Liao, S., Cheng, M., Ma, K., Wu, L., Lai, Y.,
Qiu, X., Yang, J., Xu, J., Hao, S., et al. Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. *Cell*, 185(10):1777–
1792, 2022.

Optimal transport has emerged as a powerful tool in generative modeling, yet its widespread use has been limited by scalability constraints. HiRef overcomes this limitation by enabling the application of OT to datasets with millions of points. This advancement paves the way for integrating OT into large-scale deep generative models and modern vision and language tasks.

Chen, J., Chen, L., Liu, Y. P., Peng, R., and Ramaswami, A. Exponential convergence of Sinkhorn under regularization scheduling. In *SIAM Conference on Applied and* Computational Discrete Algorithms, pp. 180–188. SIAM, 2023.

As with any computational tool which may advance largescale generative modeling, there are potential issues with bias in training datasets and a possibility of misuse. Use of HiRef in applications should be careful and transparent about these risks and utilize appropriate mitigation strategies.

Chen, X. and Price, E. Condition number-free query and active learning of linear families. *CoRR, abs/1711.10051*, 24, 2017.

Chen, Y., Georgiou, T. T., and Tannenbaum, A. Optimal transport for Gaussian mixture models. *IEEE Access*, 7: 6269–6278, 2018.

Our implementation of Hierarchical Refinement is available at https://github.com/raphael-group/HiRef.

Clifton, K., Anant, M., Aihara, G., Atta, L., Aimiuwu, O. K.,
Kebschull, J. M., Miller, M. I., Tward, D., and Fan, J. STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping. *Nature Communications*, 14(1):8123, 2023.

Aurenhammer, F., Hoffmann, F., and Aronov, B.

Minkowski-type theorems and least-squares clustering. Algorithmica, 20:61–76, 1998.

Cohen, J. E. and Rothblum, U. G. Nonnegative ranks, decompositions, and factorizations of nonnegative matrices. Linear Algebra and its Applications, 190:149–168, 1993.

Birkhoff, G. Tres observaciones sobre el algebra lineal.

Univ. Nac. Tucuman, Ser. A, 5:147–154, 1946.

Courty, N., Flamary, R., and Tuia, D. Domain adaptation with regularized optimal transport. In European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 274–289. Springer, 2014.

Bonneel, N., Rabin, J., Peyre, G., and Pfister, H. Sliced and ´
Radon Wasserstein barycenters of measures. *Journal of* Mathematical Imaging and Vision, 51:22–45, 2015.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in Neural Information Processing Systems, 26:2292–2300, 2013.

Brenier, Y. Polar factorization and monotone rearrangement of vector-valued functions. Communications on pure and applied mathematics, 44(4):375–417, 1991.

Cuturi, M., Meng-Papaxanthos, L., Tian, Y., Bunne, C.,
Davis, G., and Teboul, O. Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein. arXiv preprint arXiv:2201.12324, 2022.

Bunne, C., Papaxanthos, L., Krause, A., and Cuturi, M.

Proximal optimal transport modeling of population dynamics. In International Conference on Artificial Intelligence and Statistics, volume 25, pp. 6511–6528. PMLR, 2022.

De Bortoli, V., Korshunova, I., Mnih, A., and Doucet, A.

Schrodinger bridge flow for unpaired data translation. ¨ Advances in Neural Information Processing Systems, 37: 103384–103441, 2024. URL https://openreview.

net/forum?id=1F32iCJFfa.

Bunne, C., Stark, S. G., Gut, G., del Castillo, J. S., Levesque, M., Lehmann, K.-V., Pelkmans, L., Krause, A., and Ratsch, G. Learning single-cell perturbation responses ¨
De Loera, J. A. and Kim, E. D. Combinatorics and geometry of transportation polytopes: An update. Discrete Geometry and Algebraic Combinatorics, 625:37–76, 2013.

Genevay, A., Peyre, G., and Cuturi, M. Learning gen- ´
erative models with Sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics, volume 84, pp. 1608–1617. PMLR, 2018. URL https://proceedings.mlr.press/v84/ genevay18a.html.

Delon, J. and Desolneux, A. A Wasserstein-type distance in the space of Gaussian mixture models. *SIAM Journal* on Imaging Sciences, 13(2):936–970, 2020.

Gerber, S. and Maggioni, M. Multiscale strategies for computing optimal transport. Journal of Machine Learning Research, 18(72):1–32, 2017.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A large-scale hierarchical image database. In *IEEE Conference on Computer Vision and Pattern* Recognition, pp. 248–255. IEEE, 2009.

Geshkovski, B., Letrouit, C., Polyanskiy, Y., and Rigollet, P. A mathematical perspective on Transformers. arXiv preprint arXiv:2312.10794, 2023.

El Hamri, M., Bennani, Y., and Falih, I. Hierarchical optimal transport for unsupervised domain adaptation. Machine Learning, 111(11):4159–4182, 2022.

Glimm, T. and Henscheid, N. Iterative scheme for solving optimal transportation problems arising in reflector design. *International Scholarly Research Notices*, 2013(1): 635263, 2013.

Fan, J., Liu, S., Ma, S., Zhou, H.-M., and Chen, Y. Neural Monge map estimation and its applications. Transactions on Machine Learning Research, 2023. URL https: //openreview.net/forum?id=2mZSlQscj3.

Halmos, P., Liu, X., Gold, J., and Raphael, B. Low-
Rank Optimal Transport through Factor Relaxation with Latent Coupling. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=hGgkdFF2hR.

Fatras, K., Zine, Y., Flamary, R., Gribonval, R., and Courty, N. Learning with minibatch Wasserstein: asymptotic and gradient properties. In International Conference on Artificial Intelligence and Statistics, volume 108, pp. 2131– 2141. PMLR, 2020. URL http://proceedings. mlr.press/v108/fatras20a.html.

Halmos, P., Gold, J., Liu, X., and Raphael, B. J. Learning latent trajectories in developmental time series with Hidden-Markov optimal transport. In International Conference on Research in Computational Molecular Biology, pp. 367–370. Springer, 2025a.

Fatras, K., Sejourn ´ e, T., Flamary, R., and Courty, N. Un- ´
balanced minibatch optimal transport; applications to domain adaptation. In International Conference on Machine Learning, volume 139, pp. 3186–3197. PMLR, 2021a.

URL http://proceedings.mlr.press/v139/ fatras21a.html.

Halmos, P., Liu, X., Gold, J., Chen, F., Ding, L., and Raphael, B. J. DeST-OT: Alignment of spatiotemporal transcriptomics data. *Cell Systems*, 16(2), 2025b.

Fatras, K., Zine, Y., Majewski, S., Flamary, R., Gribonval, R., and Courty, N. Minibatch optimal transport distances; analysis and applications. arXiv preprint arXiv:2101.01792, 2021b.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE* conference on computer vision and pattern recognition, pp. 770–778, 2016.

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., and Oberman, A. How to train your neural ODE: the world of Jacobian and kinetic regularization. In International Conference on Machine Learning, pp. 3154–3164. PMLR, 2020.

Huangfu, Q. and Hall, J. A. J. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018.

Forrow, A., Hutter, J.-C., Nitzan, M., Rigollet, P., ¨
Schiebinger, G., and Weed, J. Statistical optimal transport via factored couplings. In International Conference on Artificial Intelligence and Statistics, volume 89, pp. 2454– 2465. PMLR, 2019. URL https://proceedings. mlr.press/v89/forrow19a.html.

Huizing, G.-J., Peyre, G., and Cantini, L. Learn- ´
ing cell fate landscapes from spatial transcriptomics using Fused Gromov-Wasserstein. bioRxiv preprint bioRxiv:2024.07.26.605241, 2024.

Indyk, P., Vakilian, A., Wagner, T., and Woodruff, D. P.

Sample-optimal low-rank approximation of distance matrices. In *Conference on Learning Theory*, volume 99, pp.

1723–1751. PMLR, 2019.

Frieze, A., Kannan, R., and Vempala, S. Fast Monte-Carlo Algorithms for Finding Low-rank Approximations. J. ACM, 51(6):1025–1041, nov 2004. ISSN 0004-5411. doi: 10.1145/1039488.1039494. URL https://doi.

org/10.1145/1039488.1039494.

Kantorovich, L. On the translocation of masses. Doklady Akademii Nauk SSSR, 37(7-8):227–229, 1942.

Kassraie, P., Pooladian, A.-A., Klein, M., Thornton, J.,
Niles-Weed, J., and Cuturi, M. Progressive entropic optimal transport solvers. Advances in Neural Information Processing Systems, 37:19561–19590, 2024.

Klein, D., Uscidda, T., Theis, F. J., and Cuturi, M. Generative entropic neural optimal transport to map within and across space, 2024. URL https://openreview. net/forum?id=gBLEHzKOfF.

Klein, D., Palla, G., Lange, M., Klein, M., Piran, Z., Gander, M., Meng-Papaxanthos, L., Sterr, M., Saber, L., Jing, C., Bastidas-Ponce, A., Cota, P., Tarquis-Medina, M., Parikh, S., Gold, I., Lickert, H., Bakhti, M., Nitzan, M., Cuturi, M., and Theis, F. J. Mapping cells through time and space with moscot. *Nature*, pp. 1–11, 2025.

Kornilov, N., Mokrov, P., Gasnikov, A., and Korotin, A.

Optimal flow matching: Learning straight trajectories in just one step. Advances in Neural Information Processing Systems, 37:104180–104204, 2024. URL https:// openreview.net/forum?id=kqmucDKVcU.

Korotin, A., Li, L., Genevay, A., Solomon, J. M., Filippov, A., and Burnaev, E. Do neural optimal transport solvers work? A continuous Wasserstein-2 benchmark. *Advances* in Neural Information Processing Systems, 34:14593–
14605, 2021.

Korotin, A., Selikhanovych, D., and Burnaev, E. Neural optimal transport. *International Conference on Learning* Representations, 2023. URL https://openreview. net/forum?id=d8CBRlWNkqH.

Kuhn, H. W. The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2): 83–97, 1955.

Lavenant, H., Zhang, S., Kim, Y.-H., and Schiebinger, G.

Toward a mathematical theory of trajectory inference. The Annals of Applied Probability, 34(1A):428–500, 2024.

Li, X., Chen, J., Chai, Y., and Xiong, H. GiLOT: Interpreting generative language models via optimal transport. *International Conference on Machine Learning*,
2024. URL https://openreview.net/forum? id=qKL25sGjxL.

Lin, C.-H., Azabou, M., and Dyer, E. L. Making transport more robust and interpretable by moving data through a small number of anchor points. *International Conference* on Machine Learning, 139:6631, 2021.

Luo, J., Yang, D., and Wei, K. Improved complexity analysis of the sinkhorn and greenkhorn algorithms for optimal transport. *arXiv preprint arXiv:2305.14939*, 2023.

Makkuva, A., Taghvaei, A., Oh, S., and Lee, J. Optimal transport mapping via input convex neural networks. International Conference on Machine Learning, 119:6672– 6681, 2020.

Melnyk, I., Mroueh, Y., Belgodere, B., Rigotti, M., Nitsure, A., Yurochkin, M., Greenewald, K., Navratil, J., and Ross, J. Distributional preference alignment of LLMs via optimal transport. Advances in Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=2LctgfN6Ty.

Merigot, Q. A multiscale approach to optimal transport. ´
Computer Graphics Forum, 30(5):1583–1592, 2011.

Monge, G. Memoire sur la th ´ eorie des d ´ eblais et des rem- ´
blais. *Mem. Math. Phys. Acad. Royale Sci.*, pp. 666–704, 1781.

Nguyen, K., Nguyen, D., Pham, T., and Ho, N. Improving mini-batch optimal transport via partial transportation. In Proceedings of the 39th International Conference on Machine Learning, 2022a.

Nguyen, K., Ren, T., Nguyen, H., Rout, L., Nguyen, T. M.,
and Ho, N. Hierarchical sliced Wasserstein distance. International Conference on Learning Representations, 2022b.

Oberman, A. M. and Ruan, Y. An efficient linear programming method for optimal transportation. arXiv preprint arXiv:1509.03668, 2015.

Orlin, J. B. A polynomial time primal network simplex algorithm for minimum cost flows. Mathematical Programming, 78(2):109–129, 1997.

Peyre, G. and Cuturi, M. Computational optimal transport: ´
With applications to data science. Foundations and Trends in Machine Learning, 11(5–6):355–607, 2019.

Qin, Z., Yu, H., Wang, C., Guo, Y., Peng, Y., and Xu, K.

Geometric transformer for fast and robust point cloud registration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11143–11152, 2022.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. International Conference on Machine Learning, 139:8821–8831, 2021.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S.,
Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015.

Sander, M. E., Ablin, P., Blondel, M., and Peyre, G. Sink- ´
formers: Transformers with doubly stochastic attention. In International Conference on Artificial Intelligence and Statistics, pp. 3515–3530. PMLR, 2022.

Scetbon, M. and Cuturi, M. Low-rank optimal transport:
Approximation, statistics and debiasing. Advances in Neural Information Processing Systems, 35:6802–6814, 2022.

Scetbon, M., Cuturi, M., and Peyre, G. Low-rank Sinkhorn ´
factorization. International Conference on Machine Learning, pp. 9344–9354, 2021.

Scetbon, M., Peyre, G., and Cuturi, M. Linear-time Gro- ´
mov Wasserstein distances using low rank couplings and costs. *International Conference on Machine Learning*, pp. 19347–19365, 2022.

Scetbon, M., Klein, M., Palla, G., and Cuturi, M. Unbalanced low-rank optimal transport solvers. *Advances* in Neural Information Processing Systems, 36:52312–
52325, 2023.

Schiebinger, G., Shu, J., Tabaka, M., Cleary, B., Subramanian, V., Solomon, A., Gould, J., Liu, S., Lin, S., and Berube, P. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell*, 176(4):928–943, 2019.

Schmitzer, B. A sparse multiscale algorithm for dense optimal transport. *Journal of Mathematical Imaging and* Vision, 56:238–259, 2016.

Schmitzer, B. and Schnorr, C. A hierarchical approach to ¨
optimal transport. In *International Conference on Scale* Space and Variational Methods in Computer Vision, pp. 452–464. Springer, 2013.

Seguy, V., Damodaran, B. B., Flamary, R., Courty, N., Rolet, A., and Blondel, M. Large-scale optimal transport and mapping estimation. *International Conference on* Learning Representations, 2018.

Solomon, J., De Goes, F., Peyre, G., Cuturi, M., Butscher, ´
A., Nguyen, A., Du, T., and Guibas, L. Convolutional Wasserstein distances: Efficient optimal transportation on geometric domains. *ACM Transactions on Graphics*, 34 (4):1–11, 2015.

Sommerfeld, M., Schrieber, J., Zemel, Y., and Munk, A.

Optimal transport: Fast probabilistic approximation with exact solvers. *Journal of Machine Learning Research*, 20
(105):1–23, 2019.

Stahl, P. L., Salm ˚ en, F., Vickovic, S., Lundmark, A., ´
Navarro, J. F., Magnusson, J., Giacomello, S., Asp, M.,
Westholm, J. O., and Huss, M. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. *Science*, 353(6294):78–82, 2016.

Tarjan, R. E. Dynamic trees as search trees via Euler tours, applied to the network simplex algorithm. Mathematical Programming, 78(2):169–177, 1997.

Tay, Y., Bahri, D., Yang, L., Metzler, D., and Juan, D.-C.

Sparse Sinkhorn attention. *International Conference on* Machine Learning, 119:9438–9447, 2020.

Thorpe, M. Introduction to optimal transport. *Notes of* Course at University of Cambridge, 2018.

Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y.,
Rector-Brooks, J., Wolf, G., and Bengio, Y. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024. URL https://openreview. net/forum?id=CD9Snc73AW.

Wolf, F. A., Angerer, P., and Theis, F. J. SCANPY: Largescale single-cell gene expression data analysis. *Genome* Biology, 19:1–5, 2018.

Yang, K. D., Damodaran, K., Venkatachalapathy, S., Soylemezoglu, A. C., Shivashankar, G. V., and Uhler, C. Predicting cell lineages using autoencoders and optimal transport. *PLoS Computational Biology*, 16(4):e1007828, 2020.

Yu, H., Li, F., Saleh, M., Busam, B., and Ilic, S. CoFiNet:
Reliable coarse-to-fine correspondences for robust pointcloud registration. Advances in Neural Information Processing Systems, 34:23872–23884, 2021.

Yurochkin, M., Claici, S., Chien, E., Mirzazadeh, F., and Solomon, J. M. Hierarchical optimal transport for document representation. Advances in Neural Information Processing Systems, 32, 2019.

Zeira, R., Land, M., Strzalkowski, A., and Raphael, B. J.

Alignment and integration of spatial transcriptomics data.

Nature Methods, 19(5):567–575, 2022.

## A. Hierarchical-Refinement Algorithm

Algorithm 2 Hierarchical Refinement for Full-Rank OT
Require: **Datasets** X = {xi}
n i=1, Y = {yi}
n i=1; **Low-rank OT solver** LROT(·); **Rank schedule** (r1, r2*, . . . , r*κ);
Base rank rbase = Q n κ t=1 rt
(e.g. 1).

Initialize:
1: t ← 0, Γ0 ← { (X, Y)}
2: **while** ∃ (X
(t), Y
(t)) ∈ Γt **such that**
3: min{|X
(t)|, |Y
(t)|} > rbase do 4: Γt+1 ← ∅
5: for (X
(t)
q , Y
(t)
q ) ∈ Γt do 6: if min{|X
(t)
q |, |Y
(t)
q *|} ≤* rbase **then**
7: Γt+1 ← Γt+1 ∪ {(X
(t)
q , Y
(t)
q )}
8: **else**
9: µX
(t)
q=1 |X
(t)
q | Px∈X
(t)
qδx 10: µY
(t)
q=1 |Y
(t)
q | Py∈Y
(t)
qδy.

11: gt+1 ← (1/rt+1)1rt+1 12: (Q, R) ← LROT(µX
(t) q
, µY
(t) q
, gt+1)
13: for z = 1 → rt+1 do 14: X
(t+1)
z ← Assign(X
(t), Q, z)
15: Y
(t+1)
z ← Assign(Y
(t), R, z)
16: Γt+1 ← Γt+1 ∪ { (X
(t+1)
z , Y
(t+1)
z )}
17: **end for**
18: ▷ Assign(S,M, z) = {s ∈ S | arg maxz′ Msz′ = z}
19: **end if** 20: **end for** 21: t ← t + 1 22: **end while** 23: **Output:** Γκ = {(xi, T(xi))} ▷ Set of refined pairs.

## B. Proofs

Datasets X and Y are represented as discretely supported probability measures µ =Pn i=1 aiδxi and ν =Pn j=1 bj δyj for probability vectors a, b ∈ ∆n, which we assume to be uniform: a = b = un = (1/n)1n ∈ ∆n. We form the cost matrix C defined by

$$(\mathrm{S1})$$

Cij := c(xi, yj ). (S1)
In all cases below, we are concerned with the assignment problem (1) for this cost matrix.

Let perm(n) = {P˜ ∈ R
n×n : P1˜n = P˜ ⊤1n = (1/n) 1n} denote the set of (scaled) n × n permutation matrices. By the Birkhoff-von Neumann theorem (Birkhoff, 1946), an optimal solution to the n × n assignment problem is attained at a permutation matrix in perm(n).

Definition B.1. Say that cost matrix C ∈ R
n×n is *Monge rotated* if the identity matrix I is a solution to the assignment problem associated to C, i.e.

$$\mathbf{I}\in\operatorname*{arg\,min}_{\mathbf{P}\in\mathbf{perm}(n)}\langle\mathbf{C},\mathbf{P}\rangle.$$

For arbitrary cost matrix C ∈ R
n×n, let P† ∈ arg min P∈**perm**(n) ⟨C, P⟩F
, and note that the column-permuted cost matrix C†:= CP†,⊤ is Monge rotated by construction. This is a consequence of the following identity, which holds for any permutation P˜ ∈ perm(n).

$$\langle\mathbf{C},\mathbf{P}\rangle_{F}=\text{tr}(\mathbf{C}^{\top}\mathbf{P})$$ $$=\text{tr}(\hat{\mathbf{P}}^{-1}\hat{\mathbf{P}}\mathbf{C}^{\top}\mathbf{P})$$ $$=\text{tr}(\hat{\mathbf{P}}\mathbf{C}^{\top}\hat{\mathbf{P}}\hat{\mathbf{P}}^{\top})=\langle\mathbf{C}\hat{\mathbf{P}}^{\top},\hat{\mathbf{P}}\hat{\mathbf{P}}^{\top}\rangle_{F}.\tag{1}$$
$$(\mathrm{S2})$$

Let Π(un,ur) ≡ Πun,u2denote the transport polytope between two uniform measures. For Q ∈ Π(un,ur), say that a row of Q is *soft* if at least two of its entries are positive, and call the row *hard* otherwise. For rank r ≪ n such that r divides n, let Π•(un,ur) be the subset of Π(un,ur) consisting of transport plans Q with only hard rows. Below, we consider two low-rank OT problems associated to C†. The first low-rank problem considered is

$$\min_{{\bf Q},{\bf R}\in\Pi_{\bullet}({\bf u}_{n},{\bf u}_{r})}\ \langle{\bf C}^{\dagger},{\bf QR}^{\top}\rangle_{F}.\tag{10}$$

while the second low-rank problem considered is restricted to symmetric couplings:

$$(\mathbf{S3})$$
$$\operatorname*{min}_{{\bf Q}\in\Pi_{\bullet}({\bf u}_{n},{\bf u}_{r})}\langle{\bf C}^{\dagger},{\bf Q}{\bf Q}^{\top}\rangle_{F}.$$
$$(\mathbf{S4})$$
⟨C†, QQ⊤⟩F . (S4)
In either case, we have omitted the constant factor of r coming from diag(1/ur). We next introduce a technical condition on C. Let C ∈ R
n×n be a cost matrix and let P† ∈ arg minP∈**perm**(n)⟨C, P⟩ corresponding to permutation σ
†: [n] →
[n], σ† ∈ Sn. Given partitions I = {I1*, . . . , I*r} and J = {J1*, . . . , J*r} of [n] and *a, b* ∈ [r], define the cost between two sets Ia, Jb to be

$$\mathbf{C}_{I_{a},J_{b}}:=\sum_{i\in I_{a},j\in J_{b}}\mathbf{C}_{i\sigma^{\dagger}(j)}.\tag{1}$$

We call partition I *balanced* if each block Ia of I has the same number of elements, | Ia | = (n/r).

Definition B.2. Cost matrix C ∈ R
n×n is r*-Monge separable* if there exists a balanced partition I
⋆ = {I
⋆k
}
rk=1, such that for any two permutations π1, π2 ∈ Sn, one has

$$\sum_{k=1}^{r}{\bf C}_{I_{k}^{*},I_{k}^{*}}\leq\sum_{k=1}^{r}{\bf C}_{\pi_{1}(I_{k}^{*}),\pi_{2}(I_{k}^{*})}.$$
$$(\mathbf{S}\mathbf{S})$$
$$(\mathbf{S6})$$

We say that C is strictly r*-Monge separable* if (S6) holds with strict inequality (<) for any π1(I
⋆
k) ̸= π2(I
⋆
k).

One interesting feature of this definition is that while the sum is over r ≤ n terms, where it may occur that r ≪ n, this inequality must hold over all permutations π1 and π2 acting on the individual data points, rather than partition blocks. This captures the notion of finding low-rank or low-resolution solutions which are nevertheless compatible with the optimal bijective Monge map. Remark B.3. If C is r-Monge separable, the distinguished partition I
⋆ may be represented as Q⋆ ∈ Π•(un,ur) such that Q⋆is optimal for (S4) and the pair (Q⋆, Q⋆) is optimal for (S3). After proving the next lemma, we will relate r-Monge separability to cyclic monotonicity.

Proposition B.4. Let C ∈ R
n×n be strictly r*-Monge separable. If* Q⋆, R⋆ ∈ arg minQ,R∈Π•(un,u2)⟨C, QR⊤⟩ then, for all i ∈ [n],

$$\arg\max_{z\in[r]}{\bf Q}_{iz}^{\star}=\arg\max_{z\in[r]}{\bf R}_{\sigma^{\dagger}(i)z}^{\star},\tag{10}$$

where σ
†: [n] → [n] *is the permutation corresponding to* P† ∈ arg minP∈**perm**(n)
⟨C, P⟩F .

Proof. Let σ
†, P† be as in the statement of the lemma, and define C†:= CP†,⊤. The same reasoning as in (S2) implies that if (Q⋆, R⋆) ∈ arg minQ,R∈Π•(un,u2)⟨C, QR⊤⟩F , then

$$(\mathbf{Q}^{*},\mathbf{P}^{\dagger}\mathbf{R}^{*})\in\operatorname*{arg\,min}_{\mathbf{Q},\mathbf{R}\in\Pi_{\bullet}(\mathbf{u}_{n},\mathbf{u}_{2})}(\mathbf{C}^{\dagger},\mathbf{Q}\mathbf{R}^{\top})_{F}.\tag{1}$$
$$(\mathbf{S}7)$$
$$(\mathrm{S}8)$$

The membership (S8) follows from the identities

$$\langle\mathbf{C}^{\dagger},\mathbf{Q}^{\star}\mathbf{R}^{\star}\mathbf{P}^{\dagger,\top}\rangle_{F}=\langle\mathbf{CP}^{\dagger,\top},\mathbf{Q}^{\star}\mathbf{R}^{\star,\top}\mathbf{P}^{\dagger,\top}\rangle_{F},$$ $$=\mathbf{tr}(\mathbf{P}^{\dagger}\mathbf{C}^{\top}\mathbf{Q}^{\star}\mathbf{R}^{\star,\top}\mathbf{P}^{\dagger,\top}),$$ $$=\mathbf{tr}\,\mathbf{C}^{\top}\mathbf{Q}^{\star}\mathbf{R}^{\star,\top}=\langle\mathbf{C},\mathbf{Q}^{\star}\mathbf{R}^{\star,\top}\rangle_{F}.$$
$$(\mathrm{S9})$$

Remark B.3 above follows from the requirement that the variables Q, R have all hard rows, and are subject to uniform marginal constraints, so that all non-zero entries of QR⊤ have the same value. Thus, if C is r-Monge separable, there exists Q˜ ∈ Π•(un,u2) corresponding to distinguished balanced partition I˜ from Definition B.2 such that

$$(\bar{\mathbf{Q}},\bar{\mathbf{Q}})\in\operatorname*{arg\,min}_{\mathbf{Q},\mathbf{R}\in\Pi_{\bullet}(\mathbf{u}_{n},\mathbf{u}_{2})}\langle\mathbf{C}^{\dagger},\mathbf{Q}\mathbf{R}^{\top}\rangle.\tag{14}$$

Moreover, this pair (Q˜ , Q˜ ) is the unique optimum when C is strictly r-Monge separable. From (S8), (S9), we must have

$$\bar{\mathbf{Q}}=\mathbf{Q}^{*},\quad\bar{\mathbf{Q}}=\mathbf{P}^{\dagger}\mathbf{R},$$

from which (S7) follows immediately.

Let us now discuss how the notion of r-Monge separability is related to c-cyclic monotonicity. Recall that for a cost matrix C ∈ R
n×n derived from ground cost c the support of an optimal plan is c-cyclically monotone if for all permutations π : [n] → [n], π ∈ Sn, one has

$$\sum_{i=1}^{n}{\bf C}_{ii}\leq\sum_{i=1}^{n}{\bf C}_{i\pi(i)}.\tag{1}$$
$$\boxed{\bot}$$
$$(\mathrm{S}10)$$

As it amounts to a reindexing of the sum on the right side of (S10) , one can equivalently define the support of the optimal plan to be c-cyclically monotone if for any *pair* of permutations π1, π2 ∈ Sn,

$$\sum_{i=1}^{n}\mathbf{C}_{i i}\leq\sum_{i=1}^{n}\mathbf{C}_{\pi_{1}(i)\pi_{2}(i)},$$

from which we see that c-cyclical monotonicity is equivalent to r-Monge separability with r = n.

We next show that the optimal factors Q⋆, R⋆for the rank-2 Wasserstein problem given in (5) correspond to hard-partitions of each dataset, so that for this problem the optimal Q⋆, R⋆ ∈ Π(un,u2) satisfy Q⋆, R⋆ ∈ Π•(un,ur). Below, let suppi(Q⋆) ⊂ [n] be the indices on which column i of Q⋆is supported, where i = 1, 2.

Lemma B.5. Let (Q⋆, R⋆) *be optimal for the rank-2 Wasserstein problem* (5) *subject to the additional constraint that* a = b = un, and g = u2 are uniform and n = m *is even. Then,* (supp1
(Q⋆),supp2
(Q⋆)) is a partition of [n]*, and* symmetrically, so is (supp1
(R⋆),supp2
(R⋆))*, so* (Q⋆, R⋆) ∈ Π•(un,u2).

Proof. The cost is linear in (Q, R) respectively: the minimization in each variable given the other fixed can be expressed as

$$\begin{array}{c c}{{\arg\min}}&{{2\left\langle{\bf Q},{\bf C}{\bf R}\right\rangle_{F},}}&{{\arg\min}}&{{2\left\langle{\bf R},{\bf C}^{\top}{\bf Q}\right\rangle_{F}.}}\\ {{{\bf Q}\in\Pi({\bf u}_{n},{\bf u}_{2})}}&{{{\bf R}\in\Pi({\bf u}_{n},{\bf u}_{2})}}&{{{\bf R}\in\Pi({\bf u}_{n},{\bf u}_{2})}}\end{array}$$

Thus for any optimal Q⋆ or R⋆ fixed the minimization in the other variable is a linear optimal transport problem, where by Corollary 2.11 in (De Loera & Kim, 2013) it holds that since the constraint matrix is totally unimodular with marginals integral (on rescaling), the optima R⋆and Q⋆ must be vertices on the transport polytope Πun,u2 with integral entries (on rescaling, by 2n or 2m). There are ≤ n + 1 positive entries in any optimal rank r = 2 solution (De Loera & Kim, 2013; Peyre & Cuturi ´ , 2019), so that n (resp. m) being even and the rescaled rows and columns summing to 2 and n implies that there are exactly n positive entries and thus that the vertices define partitions of [n] and [m]. Thus, solutions to S11 satisfy
(Q⋆, R⋆) ∈ Π•(un,u2).

$$(\mathrm{S}11)$$

Notably, in the case of an odd number of points n or m this likewise implies that one has a single row which has 2 entries 1/2n 1/2n, with all other rows of the form 0 1/nor 1/n 0defining a partition of the remaining even subset of size (n − 1) or (m − 1). In the general case of ranks r ̸= 2 there are maximally n + r + 1 (Peyre & Cuturi ´ , 2019) non-zero edges (so that the graph is acyclic), and for n ≫ r the optimal solution remains close to a partition given mild assumptions on C.

Lemma B.5 states optimal low-rank couplings (Q⋆, R⋆) for Problem 7 over Π(un,u2) are in Π•(un,u2). Thus, by Proposition B.4 these solutions co-cluster points x ∈ X with their image under Monge map T
⋆(x), supposing the cost is strictly 2-Monge separable (Definition B.2). This co-clustering is in the sense of the clustering functions q
⋆,r
⋆from Proposition 3.1 corresponding to each factor Q⋆, R⋆. We note that when µ and ν are discretely supported measures with supports of equal cardinality, a Monge map, T
⋆: X → Y, is guaranteed to exist by Theorem 2.7 of (Thorpe, 2018).

On the Rank Schedule. At each intermediate scale t ∈ [κ], the rank-schedule (r1*, . . . , r*κ) determines the effective rank of the coupling computed so far. For each t ∈ [κ], define the *effective rank* at scale t as ρt := Qts=1 rs. This effective rank corresponds to the number of partitions, which are placed in bijective correspondence

$$\mathsf{X}_{q}^{(t)}\leftrightarrow\mathsf{Y}_{q}^{(t)}\quad\ t\in[\rho_{t}]\,.$$
$$(\mathrm{S}12)$$

qt ∈ [ρt] . (S12)
at the t-th step of HiRef. The size of the partitions at scale t is given by n/ρt = |X
(t)| = |Y
(t)|. Given these preliminaries, we show that for an appropriate rank-schedule Hierarchical Refinement yields optimal transport maps. Proposition B.6 (Optimality of Hierarchical Refinement). Suppose the Monge-map exists between two datasets X, Y of size n. Then there exists a rank-schedule (r1, · · · , rκ) which factorizes n such that all size n/ρt partitions generated by Hierarchical Refinement at level t satisfy strict rt+1-Monge separability (Definition *B.2) for* t ∈ [0 : κ − 1]. For any such rank-schedule, given an optimal black-box low-rank solver over Π•(·, ·)*, Hierarchical Refinement returns the Monge-map.*
Proof. For existence, observe that taking r1 = n implies the statement Pn k=1 CI
⋆ k
,I⋆k
≤Pn k=1 Cπ1(I
⋆ k
),π2(I
⋆k
). For partitions Ik of size one, this is equivalent to the statement of c-cyclical monotonicity Pn i=1 Cii ≤Pn i=1 Ciπ(i), so that for the trivial rank-schedule (r1) := (n) the cost is always n-Monge separable.

Given the existence of such a schedule (r1, · · · , rκ) with rt+1-Monge separability, we proceed by induction on t ∈ [0, κ]. For the base case of t = 0, as we assume the Monge map exists, for the initial partition Γ0 = {(X, Y)} one has that Y = T(X). We want to show the variant that Γt contains sets which are co-clusters of sets with their image under T. As the inductive hypothesis, at scale t > 0 with ρt co-clusters Γt = {(X
(t)
i, Y
(t)
i)}
ρt i=1 each satisfies Y
(t)
i = T(X
(t)
i). As strict rt+1-Monge separability holds for each size n/ρt bipartition (X
(t)
i, Y
(t)
i) ∈ Γt, using Proposition B.4 each such set is divided into rt+1 co-clusters {(X
(t+1)
j, Y
(t+1)
j)}
rt+1 j=1 which satisfy Y
(t+1)
j = T(X
(t+1)
j). Thus, taking the union of these rt+1 bi-partitions across the ρt elements of Γt we form a set Γt+1 of size ρt+1 = rt+1ρt which maintains the invariant that
(X
(t+1)
j, Y
(t+1)
j) ∈ Γt+1 =⇒ Y
(t+1)
j = T(X
(t+1)
j). At the final level rκ Monge separability holds, so one may conclude on singleton sets of the form Γκ = {(xi, T(xi))}
n i=1.

Remark B.7. Strict Monge separability applies unconditionally at the terminal level. Observe that all sets in Γκ−1 have size equal to the rank (n/ρκ−1) = rκ, and that we have maintained the invariant that Y
(κ−1)
j = T(X
(κ−1)
j). Let Jκ ⊂ [n] denote the size rκ set of indices for X
(κ−1)
jin X. By c-cyclical monotonicity, one has for all permutations π ∈ perm(n)

$$\sum_{i=1}^{n}\mathbf{C}_{i i}=\sum_{i\in J_{n}}\mathbf{C}_{i i}+\sum_{j\in[n]\setminus J_{n}}\mathbf{C}_{j j}\leq\sum_{i\in J_{n}}\mathbf{C}_{i\pi(i)}+\sum_{j\in[n]\setminus J_{n}}\mathbf{C}_{j\pi(j)}=\sum_{i=1}^{n}\mathbf{C}_{i\pi(i)}$$
$$(\mathrm{S}13)$$

Thus, for the subset of permutations on n where π : π |[n]\Jκ = id, we have Pi∈Jκ Cii ≤Pi∈Jκ Ciπ(i)implying that one may solve a constant time O(r 2κ) solution to the assignment problem on each size rκ bipartition to recover the final map.

We call ρt the effective rank because (to avoid quadratic space complexity) we never instantiate the transport coupling corresponding to the bijective mapping (S12) as a matrix T(t). Were we to instantiate T(t), it would have rank ρt, and moreover we can evaluate its transport cost by using T(t)to induce a transport coupling P(t) between the full datasets X, Y.

$$\mathbf{P}_{i j}^{(t)}:={\begin{cases}\rho_{t}/n^{2}&{\mathrm{if}}\quad q(n/\rho_{t})<i,j\leq(q+1)(n/\rho_{t})\\ 0&{\mathrm{otherwise}}\end{cases}}\ ,$$
, (S13)
where q ∈ [ρt], and where the mass ρt/n2is a simplified form of (ρt/n)
2(1/ρt). We note that this is a rewriting of ρt n2Pρt q=1 δ(xi,yj )∈Γt,q to have the indices ordered into a contiguous block-structure. Using coupling (S13), which again we never instantiate, one can define:

$$\operatorname{cost}(\mathbf{T}^{(t)}):=\langle\mathbf{C},\mathbf{P}^{(t)}\rangle.$$

The next proposition shows that the costs ⟨C, P(t)⟩ decrease as t increases from 1 to κ, and also provides a bound on their consecutive differences. Below, recall that each Γt denotes the co-clustering (X
(t), Y
(t)), where

$$\mathbf{\chi}^{(t)}=\{\mathbf{\chi}_{q}^{(t)}\}_{q=1}^{\rho_{t}},\quad\mathbf{\gamma}^{(t)}=\{\mathbf{\gamma}_{q}^{(t)}\}_{q=1}^{\rho_{t}}\,,$$

and where co-cluster Γt,q is defined as:

$$\Gamma_{t,q}:=\left\{(\mathbf{x},\mathbf{y}):\mathbf{x}\in\mathsf{X}_{q}^{(t)},\,\mathbf{y}\in\mathsf{Y}_{q}^{(t)}\right\}.$$

Proposition B.8 (Proposition 3.4). *Let cost function* c : R
d ×R
d → R+ *be of the form* c(x, y) = h(x−y) for some strictly convex function h : R
d → R+ and suppose that h is Lipschitz. Let P(t) *be as defined above in* (S13). Then one has the following bound on the difference in cost between iterations of refinement:

$$0\leq\langle{\bf C},{\bf P}^{(t)}\rangle-\langle{\bf C},{\bf P}^{(t+1)}\rangle\leq\|\nabla c\|_{\infty}\frac{1}{\rho_{t}}\sum_{q=1}^{\rho_{t}}{\rm diam}\big{(}\Gamma_{t,q}\big{)}\,,$$ (S14)
where

$$\operatorname{diam}\bigl{(}\Gamma_{t,q}\bigr{)}\equiv\operatorname{diam}\bigl{(}\mathsf{X}_{q}^{(t)}\cup T(\mathsf{X}_{q}^{(t)})\bigr{)}\ :=\ \max_{\mathbf{x}_{i},\,\mathbf{x}_{j},\,\mathbf{x}_{k},\,\mathbf{x}_{l}\in\mathsf{X}_{q}^{(t)}}\Bigl\|\bigl{(}\mathbf{x}_{i},\,T(\mathbf{x}_{j})\bigr{)}\ -\ \bigl{(}\mathbf{x}_{k},\,T(\mathbf{x}_{l})\bigr{)}\Bigr\|.$$

Proof. By definition (S13) of P(t),

⟨C, P (t)⟩ − ⟨C, P (t+1)⟩ = ρt n2 Xn i=1 Xn j=1 c(xi, yj ) X ρt q=1 δ(xi,yj )∈Γt,q − ρt+1 n2 Xn i=1 Xn j=1 c(xi, yj ) ρ Xt+1 q=1 δ(xi,yj )∈Γt+1,q = ρt n2  q ′=1 δ(xi,yj )∈Γt+1,q′  q=1 δ(xi,yj )∈Γt,q − rt+1Xn i=1 Xn j=1 c(xi, yj ) ρ Xt+1 Xn i=1 Xn j=1 c(xi, yj )X ρt  = ρt n2  q=1 Xn i=1 Xn j=1 c(xi, yj )δ(xi,yj )∈Γt,q − rt+1 ρ Xt+1 q ′=1 Xn i=1 Xn j=1 c(xi, yj )δ(xi,yj )∈Γt+1,q′  X ρt  .
By Proposition B.4, one then has:

$$=\frac{\rho_{t+1}}{n^{2}}\left(\sum_{j=1}^{\rho_{t}}\left(\underbrace{\frac{1}{r_{t+1}}\sum_{i\in\mathbb{N}_{j}^{(t)},j\in\mathbb{N}_{j}^{(t)}}c(\mathbf{x}_{i},T(\mathbf{x}_{j}))}_{\text{merge Change flowing in}\Gamma_{t,\text{eq}}\text{over net scale}}-\underbrace{\sum_{i=1}^{r_{t+1}}\sum_{i\in\mathbb{N}_{j}^{(t+1)},j\in\mathbb{N}_{j}^{(t+1)}}c(\mathbf{x}_{i},T(\mathbf{x}_{j}))}_{\text{Mang duration}^{\text{\tiny{$\alpha$}}}\text{work}t+1}\right)\right)\tag{515}$$
$$(\mathrm{S15})$$

Note that the inner summands of (S15) (indexed by q) are non-negative by definition of the refinement step, where *within* each cluster, one has a minimization over a larger set of couplings. This shows ⟨C, P(t)*⟩ − ⟨*C, P(t+1)⟩ ≥ 0. Towards an upper bound, we will bound each summand of (S15):

$$\left(\frac{1}{r_{t+1}}\sum_{i\in\mathsf{X}_{q}^{(t)}}\sum_{j\in\mathsf{X}_{q}^{(t)}}c(\mathbf{x}_{i},T(\mathbf{x}_{j}))-\sum_{z=1}^{r_{t+1}}\sum_{i\in\mathsf{X}_{q\neq t-\bar{z}}^{(t+1)}}\sum_{j\in\mathsf{X}_{q\neq t-\bar{z}}^{(t+1)}}c(\mathbf{x}_{i},T(\mathbf{x}_{j}))\right)\,.$$ (S16)
Define st+1 := n/ρt+1 as well as barycenters

$${\bar{\mathbf{x}}}^{(t)}:=\sum_{\mathbf{x}_{i}\in\mathsf{X}_{q p t+z}^{(t+1)}}{\frac{\mathbf{x}_{i}}{s_{t+1}}},\quad{\bar{\mathbf{y}}}^{(t)}:=\sum_{\mathbf{x}\in\mathsf{X}_{q p t+z}^{(t+1)}}{\frac{T(\mathbf{x}_{i})}{s_{t+1}}}\,,$$

and note that by Jensen's inequality, for convex cost c(·, ·) one has:

$$\sum_{z=1}^{r_{t+1}}\sum_{\mathbf{x}_{i}\in\mathcal{C}_{q_{t+1}+z}^{(t+1)}}\sum_{\mathbf{x}_{j}\in\mathcal{C}_{q_{t+1}+z}^{(t+1)}}c(\mathbf{x}_{i},T(\mathbf{x}_{j}))=s_{t+1}^{2}\sum_{z=1}^{r_{t+1}}\sum_{\mathbf{x}_{i}\in\mathcal{C}_{q_{t+1}+z}^{(t+1)}}\frac{1}{s_{t+1}}\sum_{j\in\mathcal{C}_{q_{t+1}+z}^{(t+1)}}\frac{1}{s_{t+1}}c(\mathbf{x}_{i},T(\mathbf{x}_{j}))$$ $$\geq s_{t+1}^{2}r_{t+1}c(\tilde{\mathbf{x}}^{(t)},\tilde{\mathbf{y}}^{(t)}),$$
 $$\left(\textbf{S17}\right)$$  $$\left(\textbf{S18}\right)$$  $$\left(\textbf{S19}\right)$$. 
so that we may continue upper-bounding the difference (S16):

$$\leq$$
≤1
rt+1  xj∈X (t) q c(xi, T(xj ))  X xi∈X (t) q X  − s 2 t+1rt+1c(x¯ (t), y¯ (t)) (S17) rt+1     X xj∈X (t) q c(xi, T(xj ))   − n 2 ρt c(x¯ (t), y¯ (t))   (S18) X xi∈X (t) q rt+1   X c(xi, T(xj )) − c(x¯ (t), y¯ (t))    . (S19) X xi∈X (t) q xj∈X (t) q
$=\;\hdots$ . 
=1
$=\;\frac{1}{2}$  . 
=1
Now, define the diameter of co-cluster Γt,q as follows:

$$\operatorname{diam}\bigl(\Gamma_{t,q}\bigr)\equiv\operatorname{diam}\bigl(\mathsf{X}_{q}^{(t)}\cup T(\mathsf{X}_{q}^{(t)})\bigr)\;:=\;\max_{\mathbf{x}_{i},\,\mathbf{x}_{j},\,\mathbf{x}_{k},\,\mathbf{x}_{l}\in\mathsf{X}_{q}^{(t)}}\,\Bigl\|\bigl(\mathbf{x}_{i},\,T(\mathbf{x}_{j})\bigr)\;-\;\bigl(\mathbf{x}_{k},\,T(\mathbf{x}_{l})\bigr)\Bigr\|,$$

Using our Lipschitz assumption on h made at the beginning of the section, where c(x, y) = h(x−y) (we will write ∥∇c∥∞ for ∥∇h∥∞), one has the inequality:
|c(xi, T(xi)) − c(xj , T(xj ))*| ≤ ∥∇*c∥∞diamΓt,q.

Thus, returning to the bound on each summand (S16), we obtain the upper bound:

$$\leq\frac{1}{r_{t+1}}\sum_{{\bf x}_{i}\in{\sf X}_{q}^{(t)}}\sum_{{\bf x}_{j}\in{\sf X}_{q}^{(t)}}\|\nabla c\|_{\infty}\Big{\|}\big{(}{\bf x}_{i},\,T({\bf x}_{j})\big{)}\ -\ \big{(}\bar{\bf x}^{(t)},\,\bar{\bf y}^{(t)}\big{)}\Big{\|}$$
As partition $\mathsf{X}^{(t+1)}$ is a refinement of $\mathsf{X}^{(t)}$ and $\mathsf{Y}^{(t+1)}$ is a refinement of $\mathsf{Y}^{(t)}$, it holds that (S16) is upper bounded by  $$\leq\frac{1}{r_{t+1}}\sum_{i\in\mathsf{X}^{(t)}_{t}\cup\mathsf{X}^{(t)}_{t}}\|\nabla\mathsf{c}\|_{\infty}\mathrm{diam}\left(\mathsf{L}_{t,0}\right),$$ $$=\frac{1}{r_{t+1}}|\mathsf{X}^{(0)}|^{2}\|\nabla\mathsf{c}\|_{\infty}\mathrm{diam}\left(\mathsf{L}_{t,0}\right),$$ $$=\frac{1}{r_{t+1}}\frac{n^{2}\|\nabla\mathsf{c}\|_{\infty}}{\rho_{t}^{2}}\mathrm{diam}\left(\mathsf{L}_{t,0}\right).$$  To conclude, we plug these bounds into each summand of (S15), obtaining the following bound on the full sum:
$$(\mathrm{S20})$$
$$=\frac{\rho_{t+1}}{n^{2}}\frac{1}{r_{t+1}}\frac{n^{2}\|\nabla c\|_{\infty}}{\rho_{t}^{2}}\sum_{q=1}^{\rho_{t}}\operatorname{diam}\bigl{(}\Gamma_{t,q}\bigr{)}$$ $$=\|\nabla c\|_{\infty}\frac{1}{\rho_{t}}\sum_{q=1}^{\rho_{t}}\operatorname{diam}\bigl{(}\Gamma_{t,q}\bigr{)}.$$
$$(\mathbf{S21})$$
(S22)  $\left(\text{S23}\right)$  . 
$$(\mathbf{S24})$$  $$(\mathbf{S25})$$

completing the proof.

Remark B.9. Proposition B.8 should be considered a *conditional* result. Our proof follows that of (Proposition 1, (Gerber & Maggioni, 2017)), but they are able to provide sharper bounds between elements of a cluster and the centroid of the cluster using the properties assumed to hold in their definition of a multiscale family of partitions (Definition C.3), which mimick the structure of dyadic cubes in Euclidean space. As we do not make any geometric assumptions of our partitions, the above result is a priori weaker, through we leave the exploration of the geometry of partitions induced by low-rank OT to future work.

Remark *B.10*. Note, if c(xi, T(xj )) = γ is constant (i.e., if all points are equidistant in a block), one has that refinement offers no gain from level Γt → Γt+1:

$$\leq\frac{\rho_{t+1}}{n^{2}}\sum_{q=1}^{\rho_{t}}\left|\gamma\frac{|\chi_{q}^{(t)}|^{2}}{r_{t+1}}-\gamma r_{t+1}|\chi_{q}^{(t+1)}|^{2}\right|=\frac{\rho_{t+1}}{n^{2}}\sum_{q=1}^{\rho_{t}}\left|\gamma\frac{(n/\rho_{t})^{2}}{r_{t+1}}-\gamma r_{t+1}(n/\rho_{t+1})^{2}\right|=0\,.$$

Remark B.11. The work (Seguy et al., 2018) suggests a loss dependent on an (entropic) coupling γ. If γ is sparse and supported on the graph of the Monge map so that γ = (id × T)♯ µ, this loss becomes a regression of a neural network Tθ on the Monge map T over the support of µ: minTθ Eµc (Tθ(xi), T(xi)).

Proof. By linearity of the push-forward map one immediately obtains

$$\int_{\mathsf{X}\times\mathsf{Y}}\|T_{\theta}(x)-y\|_{2}^{p}(\mathrm{id}\times T)_{\sharp}\sum_{i=1}^{n}\mu_{i}\delta_{x_{i}}\mathrm{d}x\mathrm{d}y=\int_{\mathsf{X}\times\mathsf{Y}}\|T_{\theta}(x)-y\|_{2}^{p}\sum_{i=1}^{n}\mu_{i}(\mathrm{id}\times T)_{\sharp}\delta_{x_{i}}\mathrm{d}x\mathrm{d}y$$ $$=\sum_{i=1}^{n}\mu_{i}\int_{\mathsf{X}\times\mathsf{Y}}\|T_{\theta}(x)-y\|_{2}^{p}\delta_{(x_{i},T(x_{i}))}\mathrm{d}y\mathrm{d}x=\sum_{i=1}^{n}\mu_{i}\|T_{\theta}(x_{i})-T(x_{i})\|_{2}^{p},$$
$$\square$$

By integrating against the δ. As µi > 0, it holds that this loss is identically zero if and only if Tθ = T on the dataset (xi)
n i=1

$$\min_{T_{\theta}}\int_{\mathsf{X}\times\mathsf{Y}}\|T_{\theta}(x)-y\|_{2}^{p}d\gamma(x,y)=0\iff\|T_{\theta}(x_{i})-T(x_{i})\|_{2}^{p}=0\iff T_{\theta}(x_{i})=T(x_{i})$$

In other words, when one minimizes the objective of (Seguy et al., 2018) using the bijective Monge map γ = (id × T)♯µ as opposed to an entropic coupling, the objective of (Seguy et al., 2018) reduces to an unbiased regression. That is, the neural map Tθ directly matches T over the dataset support as if trained on supervised (*x, y*) pairs y = T(x).

## C. Background: Multiscale Optimal Transport C.1. Multiscale Partitions

(Gerber & Maggioni, 2017) describe a general multiscale strategy for computing OT couplings between metric measure spaces (X, dX, µ) and (Y, dY, ν). They state this in the Kantorovich setting, using a general cost function c : X × Y → R+.

Their framework consists of several elements:
1. A way of *coarsening* the set of source points X and the measure µ across multiple scales:

$$(\mathsf{X},\mu)=:(\mathsf{X}_{J},\mu_{J})\to(\mathsf{X}_{J-1},\mu_{J-1})\to\cdots\to(\mathsf{X}_{1},\mu_{1})\,,$$
(X, µ) =: (XJ , µJ ) → (XJ−1, µJ−1) → · · · → (X1, µ1), (S26)
as well as an analogous coarsening for the set of target points Y:

$$(\Upsilon,\nu)=:(\Upsilon_{J},\nu_{J})\rightarrow(\Upsilon_{J-1},\nu_{J-1})\rightarrow\cdots\rightarrow(\Upsilon_{1},\nu_{1})\,,$$
(Y, ν) =: (YJ , νJ ) → (YJ−1, νJ−1) → · · · → (Y1, ν1), (S27)
where at each scale j, supp(µj ) = Xj and supp(νj ) = Yj , and the cardinality of each Xj and Yj decreases with j.

2. A way of *propagating* coupling πj solving the transport problem µj → νj at scale j to a coupling πj+1 at scale j + 1.

3. A way of *refining the coupling* from scale j to an optimal solution at scale j + 1.

$$(\mathrm{S26})$$
$$(\mathrm{S27})$$