# Hierarchical Refinement: Optimal Transport to Infinity and Beyond

Peter Halmos 1 \* Julian Gold 1 2 \* Xinhao Liu <sup>1</sup> Benjamin J. Raphael <sup>1</sup>

## Abstract

Optimal transport (OT) has enjoyed great success in machine learning as a principled way to align datasets via a least-cost correspondence, driven in large part by the runtime efficiency of the Sinkhorn algorithm [\(Cuturi,](#page-9-0) [2013\)](#page-9-0). However, Sinkhorn has quadratic space and time complexity in the number of points, limiting scalability to larger datasets. Low-rank OT achieves linear complexity, but by definition, cannot compute a oneto-one correspondence between points. When the optimal transport problem is an assignment problem between datasets then an optimal mapping, known as the *Monge map*, is guaranteed to be a bijection. In this setting, we show that the factors of an optimal low-rank coupling co-cluster each point with its image under the Monge map. We leverage this invariant to derive an algorithm, *Hierarchical Refinement* (HiRef), that dynamically constructs a multiscale partition of each dataset using low-rank OT subproblems, culminating in the bijective Monge map. Hierarchical Refinement runs in log-linear time and linear space, retaining the advantages of low-rank OT while overcoming its limited resolution. We demonstrate the advantages of Hierarchical Refinement on several datasets, including ones containing over a million points, scaling full-rank OT to problems previously beyond Sinkhorn's reach.

# 1. Introduction

Optimal transport (OT) is a mathematical framework for comparing probability distributions µ and ν. Given a cost function c, the *Monge problem* is to find a mapping T transforming a distribution µ into ν (i.e. T♯µ = ν) with least-cost. A relaxation of this problem, called the *Kantorovich prob-* *lem*, instead seeks a least-cost coupling γ between µ and ν. In the Kantorovich formulation, mass splitting is allowed and thus a solution always exists; in contrast, a Monge map between µ and ν may not exist. When a Monge map T does exist, the solution to the Kantorovich problem is a coupling γ = (id × T) <sup>♯</sup> µ supported on its graph, and the Monge and Kantorovich problems coincide [\(Brenier,](#page-9-1) [1991\)](#page-9-1).

When µ and ν are discrete uniform measures on n points the optimal transport problem reduces to an assignment problem. Classical algorithms such as the Hungarian algorithm and Network Simplex [\(Tarjan,](#page-12-0) [1997;](#page-12-0) [Orlin,](#page-11-0) [1997\)](#page-11-0), solve this in cubic time. The Sinkhorn algorithm [\(Cuturi,](#page-9-0) [2013\)](#page-9-0) solves the entropy-regularized Kantorovich problem with quadratic runtime, greatly expanding the applicability of computational OT. However, the Sinkhorn algorihtm requires quadratic space to store the coupling γ.

In recent years, OT has found numerous applications in machine learning and across science, including: domain adaptation [\(Courty et al.,](#page-9-2) [2014;](#page-9-2) [Solomon et al.,](#page-12-1) [2015\)](#page-12-1), selfattention [\(Tay et al.,](#page-12-2) [2020;](#page-12-2) [Sander et al.,](#page-12-3) [2022;](#page-12-3) [Geshkovski](#page-10-0) [et al.,](#page-10-0) [2023\)](#page-10-0), computational biology [\(Schiebinger et al.,](#page-12-4) [2019;](#page-12-4) [Yang et al.,](#page-12-5) [2020;](#page-12-5) [Zeira et al.,](#page-12-6) [2022;](#page-12-6) [Bunne et al.,](#page-9-3) [2023;](#page-9-3) [Halmos et al.,](#page-10-1) [2025b;](#page-10-1) [Klein et al.,](#page-11-1) [2025\)](#page-11-1), unpaired data translation [\(Korotin et al.,](#page-11-2) [2021;](#page-11-2) [De Bortoli et al.,](#page-9-4) [2024;](#page-9-4) [Tong et al.,](#page-12-7) [2024;](#page-12-7) [Klein et al.,](#page-11-3) [2024\)](#page-11-3), and alignment problems in transformers and large language models [\(Melnyk](#page-11-4) [et al.,](#page-11-4) [2024;](#page-11-4) [Li et al.,](#page-11-5) [2024\)](#page-11-5). The *least-cost* principle of optimal transport is crucial for training high-quality generative models using Schrodinger bridges, flow-matching, ¨ diffusion models, or neural ordinary differential equations [\(Finlay et al.,](#page-10-2) [2020;](#page-10-2) [Tong et al.,](#page-12-7) [2024;](#page-12-7) [De Bortoli et al.,](#page-9-4) [2024;](#page-9-4) [Kornilov et al.,](#page-11-6) [2024;](#page-11-6) [Klein et al.,](#page-11-3) [2024\)](#page-11-3). These models typically require millions to hundreds of millions of data-points to achieve high-performance at scale [\(Ramesh et al.,](#page-11-7) [2021\)](#page-11-7), limiting the scope of OT for generative modeling.

As modern datasets grow to have tens of thousands or even millions of points, the quadratic space and time complexity of Sinkhorn becomes increasingly prohibitive. This limitation is widely recognized in the machine learning literature, with [\(De Bortoli et al.,](#page-9-4) [2024\)](#page-9-4) noting that the quadratic complexity of optimal transport renders its application to modern datasets on the order of millions of points impractical. A number of approaches have been proposed to address scal-

<sup>\*</sup>Equal contribution <sup>1</sup>Department of Computer Science, Princeton University <sup>2</sup>Center for Statistics and Machine Learning, Princeton University. Correspondence to: Benjamin J. Raphael <braphael@princeton.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

ing OT to massive datasets which avoid instantiating a full coupling matrix. Mini-batch OT [\(Genevay et al.,](#page-10-3) [2018\)](#page-10-3) improves scalability, but incurs significant biases [\(Sommerfeld](#page-12-8) [et al.,](#page-12-8) [2019;](#page-12-8) [Korotin et al.,](#page-11-2) [2021;](#page-11-2) [Fatras et al.,](#page-10-4) [2021a\)](#page-10-4) as each mini-batch alignment is a poor representation of the global coupling. Multiple works have investigated the theoretical properties of mini-batch estimators of the coupling [\(Fatras et al.,](#page-10-5) [2020;](#page-10-5) [2021b\)](#page-10-6), while others have attempted to mitigate this bias using partial or unbalanced OT that allows mass variation between mini-batches [\(Nguyen et al.,](#page-11-8) [2022a;](#page-11-8) [Fatras et al.,](#page-10-4) [2021a\)](#page-10-4). However, these approaches introduce additional hyperparameters to control the degree of unbalancedness, and ultimately remain biased, local approximations of the global coupling.

Neural optimal transport methods [\(Makkuva et al.,](#page-11-9) [2020;](#page-11-9) [Bunne et al.,](#page-9-3) [2023;](#page-9-3) [Fan et al.,](#page-10-7) [2023;](#page-10-7) [Korotin et al.,](#page-11-10) [2023;](#page-11-10) [Buzun et al.,](#page-9-5) [2024\)](#page-9-5), parametrize the Monge map as a neural network instead of materializing a quadratic coupling matrix. However, these methods have noted limitations recovering faithful maps [\(Korotin et al.,](#page-11-2) [2021\)](#page-11-2).

Another approach to improve space complexity of OT is to introduce a *low-rank* constraint on the coupling matrix in the Kantorovich problem. This has been done by parameterizing the coupling through a set of low-rank factors [\(Scetbon et al.,](#page-12-9) [2021;](#page-12-9) [2022;](#page-12-10) [Scetbon & Cuturi,](#page-12-11) [2022;](#page-12-11) [Scetbon et al.,](#page-12-12) [2023;](#page-12-12) [Halmos et al.,](#page-10-8) [2024\)](#page-10-8) or by using a proxy objective for the low-rank problem, factoring the transport through a small number of anchor points [\(Forrow et al.,](#page-10-9) [2019;](#page-10-9) [Lin et al.,](#page-11-11) [2021\)](#page-11-11). For a given rank r these approaches have O(nr) space complexity, enabling *linear* time and space scaling. Low-rank OT has been used successfully on datasets on the order of 10<sup>5</sup> samples with ranks on the order of 10<sup>1</sup> [\(Scetbon et al.,](#page-12-12) [2023;](#page-12-12) [Halmos et al.,](#page-10-8) [2024;](#page-10-8) [2025a;](#page-10-10) [Klein](#page-11-1) [et al.,](#page-11-1) [2025\)](#page-11-1), but computing *full-rank* couplings between datasets of sizes on the order of 10<sup>5</sup> and greater has not yet been accomplished.

Contributions We introduce Hierarchical Refinement (HiRef), an algorithm to scalably compute a full-rank alignment between two equally-sized input datasets X and Y by solving a hierarchy of low-rank OT sub-problems. The success of this refinement is driven by a theoretical result, Proposition [3.1,](#page-3-0) stating that factors of an optimal low-rank coupling between X and Y co-cluster points X with their image under the Monge map. We use Proposition [3.1](#page-3-0) recursively to obtain increasingly fine partitions of X and Y. At each scale, the solutions to low-rank OT sub-problems are bijections between the partitions of X and Y. Iterating to the finest scale gives a bijection between X and Y.

Hierarchical Refinement constructs a *multiscale partition* of each dataset, and thus is related to [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11), which introduced a general framework for multiscale optimal transport using such partitions, and the earlier work

of [\(Merigot](#page-11-12) ´ , [2011\)](#page-11-12). Unlike [\(Merigot](#page-11-12) ´ , [2011;](#page-11-12) [Gerber & Mag](#page-10-11)[gioni,](#page-10-11) [2017\)](#page-10-11), Hierarchical Refinement (i) does not assume multiscale partitions for each dataset are given, instead constructing them on the fly; and (ii) operates intrinsically to the data, without a mesh or anchor points in the ambient space of the data, avoiding the curse of dimensionality.

We demonstrate that Hierarchical Refinement computes OT maps efficiently in high-dimensional spaces, often matching or even outperforming Sinkhorn in terms of primal cost. Moreover, HiRef has linear space complexity and time complexity scaling log-linearly in the dataset size. Unlike low-rank OT, Hierarchical Refinement places X and Y in bijective correspondence. Hierarchical Refinement scales to over a million points, enabling the use of OT on massive datasets without incurring the bias of mini-batching.

## 2. Background and Related Work

Suppose X = {xi} n <sup>i</sup>=1 and Y = {yj} m <sup>j</sup>=1 are datasets in the same metric space (X , d<sup>X</sup> ). Let c : X × X → <sup>R</sup><sup>+</sup> be a cost function. This cost c is often assumed to satisfy strict convexity or to be a metric. Datasets X and Y are represented as discretely supported probability measures µ = P<sup>n</sup> <sup>i</sup>=1 aiδ<sup>x</sup><sup>i</sup> and ν = P<sup>m</sup> <sup>j</sup>=1 b<sup>j</sup> δ<sup>y</sup><sup>j</sup> for probability vectors a ∈ ∆<sup>n</sup> and b ∈ ∆m. Throughout, ∆<sup>k</sup> denotes the k*-simplex* {p ∈ <sup>R</sup> k <sup>+</sup> : P <sup>i</sup> p<sup>i</sup> = 1}, the set of probability vectors of length k.

Monge Problem Optimal transport has its origin in the *Monge problem* [\(Monge,](#page-11-13) [1781\)](#page-11-13), concerned with finding an optimal map T : X → Y pushing µ forward to ν:

$$M_c(\mu, \nu) = \min_{T: T_{\sharp}\mu = \nu} \mathbb{E}_{\mu} c(x, T(x)). \quad (1)$$

Above, T♯µ is the pushforward of µ under T, the measure on Y with T♯µ(B) := µ(T −1 (B)) for any (measurable) set B ⊂ Y. In general, a Monge map may not exist (e.g. if m > n). However, when |X| = |Y| = n and a, b are uniform then the Monge problem becomes the *assignment problem* and has a bijective solution [\(Thorpe,](#page-12-13) [2018\)](#page-12-13).

Kantorovich Problem The *Kantorovich problem* [\(Kan](#page-10-12)[torovich,](#page-10-12) [1942\)](#page-10-12) was introduced as a relaxation of the Monge problem. In contrast to the Monge problem, the Kantorovich problem allows mass-splitting and a solution is always guaranteed to exist. Define the *transport polytope* Πa,<sup>b</sup> as the following set of coupling matrices

$$\Pi_{\mathbf{a},\mathbf{b}} := \{\mathbf{P} \in \mathbb{R}_+^{n \times m} : \mathbf{P}\mathbf{1}_m = \mathbf{a}, \mathbf{P}^\top \mathbf{1}_n = \mathbf{b}\}, \quad (2)$$

respectively with left (or "source") marginal a and with right (or "target") marginal b. For the cost c(·, ·), define the cost matrix C by Cij = c(x<sup>i</sup> , y<sup>j</sup> ). In this discrete setting, the Kantorovich problem seeks a least cost coupling matrix

P ∈ Πa,<sup>b</sup> between the probability vectors a, b associated to each measure µ, ν:

$$W_c(\mu, \nu) = \min_{\mathbf{P} \in \Pi_{\mathbf{a}, \mathbf{b}}} \langle \mathbf{C}, \mathbf{P} \rangle_F. \quad (3)$$

The optimal value Wc(µ, ν) of [\(3\)](#page-2-0) is called the c*-Wasserstein distance* between µ and ν.

Sinkhorn Algorithm and the ϵ-schedule The Sinkhorn algorithm [\(Cuturi,](#page-9-0) [2013\)](#page-9-0) relaxes the classical linearprogramming formulation of optimal transport by solving an entropy regularized version of [\(3\)](#page-2-0),

$$W_\epsilon(\mu, \nu) := \min_{\mathbf{P} \in \Pi_{\mathbf{a}, \mathbf{b}}} \langle \mathbf{C}, \mathbf{P} \rangle_F - \epsilon H(\mathbf{P}), \quad (4)$$

where H(P) := − P ij Pij (log Pij − 1) is the Shannon entropy, and the parameter ϵ > 0 is the regularization strength. The Sinkhorn algorithm improved the O(n 3 log n) time complexity of classical techniques used for OT such as the Hungarian algorithm [\(Kuhn,](#page-11-14) [1955\)](#page-11-14) and Network Simplex [\(Orlin,](#page-11-0) [1997;](#page-11-0) [Tarjan,](#page-12-0) [1997\)](#page-12-0) to O(n 2 log n) [\(Luo](#page-11-15) [et al.,](#page-11-15) [2023\)](#page-11-15). As ϵ ↓ 0, the optimal coupling P⋆,ϵ for [\(4\)](#page-2-1) converges to a sparse optimal coupling for [\(3\)](#page-2-0) at an extremal point of the transport polytope (c.f. [\(Peyre & Cu-](#page-11-16) ´ [turi,](#page-11-16) [2019\)](#page-11-16)). However, the number of iterations required scales as poly(1/ϵ), diverging as ϵ decreases. A technique used to improve this scaling is the ϵ-schedule, an adaptive, monotone-decreasing and step-dependent set of entropy parameters ϵ<sup>1</sup> > ϵ<sup>2</sup> > · · · > ϵ<sup>t</sup>fin . This anneals Problem [4](#page-2-1) from high-entropy to low-entropy, gradually driving a dense initial condition to a sparse solution with a log (1/ϵ) rate [\(Chen et al.,](#page-9-6) [2023\)](#page-9-6).

Low-rank Optimal Transport The nonnegative rank rk+(M) of a nonnegative matrix M ≽ 0 is the smallest number of nonnegative rank-1 matrices summing to M; i.e. rk+(M) is the smallest integer z such that there exist nonnegative vectors q1, . . . , q<sup>z</sup> ≽ 0 and r1, . . . , r<sup>z</sup> ≽ 0 satisfying M = P<sup>z</sup> <sup>i</sup>=1 qir ⊤ i . Let Πa,b(r) := {P ∈ Πa,<sup>b</sup> : rk+(P) = r} be the set of rank-r couplings. The low-rank Wasserstein problem for general cost matrix C is:

$$\mathbf{P}^* = \arg \min_{\mathbf{P} \in \Pi_{\mathbf{a}, \mathbf{b}}(r)} \langle \mathbf{C}, \mathbf{P} \rangle_F. \quad (5)$$

From [\(Cohen & Rothblum,](#page-9-7) [1993\)](#page-9-7), each P ∈ Πa,b(r) may be decomposed as

$$\mathbf{P} = \sum (1/g_i) \mathbf{Q}_{\cdot,i} \mathbf{R}_{\cdot,i}^\top := \mathbf{Q} \text{diag}(1/g) \mathbf{R}^\top, \quad (6)$$

i=1 where g ∈ ∆r, Q ∈ Πa,<sup>g</sup> and R ∈ Πb,g. This factorization was introduced to optimal transport by [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) in the context of the general low-rank problem [\(5\)](#page-2-2). The factors Q and R constitute co-clusterings of datasets X and Y onto the *same* set of r components. Other factorizations have recently been proposed [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8), using Q, R and an intermediate latent coupling T to solve [\(5\)](#page-2-2) where X and Y have r<sup>1</sup> and r<sup>2</sup> components, respectively.

Hierarchical and Multiscale Approaches to OT Hierarchical optimal transport [\(Schmitzer & Schnorr](#page-12-14) ¨ , [2013\)](#page-12-14) is a variant of OT modeling data and transport at two scales, using Wasserstein distances as the coarse-scale ground costs. It has been applied to document representation [\(Yurochkin](#page-12-15) [et al.,](#page-12-15) [2019\)](#page-12-15), domain adaptation [\(El Hamri et al.,](#page-10-13) [2022\)](#page-10-13), sliced Wasserstein distances [\(Bonneel et al.,](#page-9-8) [2015;](#page-9-8) [Nguyen](#page-11-17) [et al.,](#page-11-17) [2022b\)](#page-11-17) and to give a discrete formulation of transport between Gaussian mixture models [\(Chen et al.,](#page-9-9) [2018;](#page-9-9) [De](#page-10-14)[lon & Desolneux,](#page-10-14) [2020\)](#page-10-14). These works build interpretable, coarse-grained structure into a single coupling, rather than solving for a sequence of couplings at progressively finer scales as in the present work.

Multiscale approaches to OT generalize hierarchical OT to a progression of scales. Building on the semidiscrete approach of [\(Aurenhammer et al.,](#page-9-10) [1998\)](#page-9-10), [\(Merigot](#page-11-12) ´ , [2011\)](#page-11-12) uses Lloyd's algorithm to progressively coarse-grain the target measure. More recently, using a regular family of multiscale partitions (Definition [C.3\)](#page-20-0) on each dataset, [\(Ger](#page-10-11)[ber & Maggioni,](#page-10-11) [2017\)](#page-10-11) formalize a general hierarchical approach to the Kantorovich problem [\(3\)](#page-2-0). They propose: (i) solving a Kantorovich problem between the coarsest partitions of X and Y in their respective multiscale families; and (ii) propagation of the optimal coupling at scale t ∈ {1, . . . , κ − 1} to initialize the optimization at scale t + 1. They take as input a chain of partitions and measures across scales (X (1), µ1) → · · · → (X (κ) , µκ) and (Y (1), ν1) → · · · → (Y (κ) , νκ) where each dataset X, Y is identified with the trivial partitions X (κ) = {{x} : x ∈ X} and Y (κ) = {{y} : y ∈ Y}. At the finest scale κ, [\(Gerber](#page-10-11) [& Maggioni,](#page-10-11) [2017\)](#page-10-11) recover the original datasets and a near optimal coupling for [\(3\)](#page-2-0).

A naive implementation of the above idea requires quadratic memory complexity, but [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) propose several propagation strategies to mitigate this, following [\(Glimm & Henscheid,](#page-10-15) [2013;](#page-10-15) [Oberman & Ruan,](#page-11-18) [2015;](#page-11-18) [Schmitzer,](#page-12-16) [2016\)](#page-12-16). These strategies use the optimal coupling at scale t to restrict the support of the coupling computed at the next scale using local optimality criteria. In the next section, we give our own such criterion, Proposition [3.1.](#page-3-0)

# 3. Hierarchical Refinement

### 3.1. Low-rank optimal transport co-clusters source-target pairs under the Monge map

We first show that under a few assumptions, the optimal lowrank factors (Q<sup>⋆</sup> , R<sup>⋆</sup> ) for a *variant* of the low-rank Wasserstein problem [\(5\)](#page-2-2) have qualities suited to our refinement strategy. Specifically, we parameterize low-rank couplings P of rank-r using the factorization P = Qdiag(1/g)R<sup>⊤</sup> of [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9), fixing g ∈ ∆<sup>r</sup> to be uniform. Define

![](_page_3_Picture_1.jpeg)

Figure 1. Hierarchical Refinement algorithm: low-rank optimal transport is used to progressively refine partitions at the previous scale, with the coarsest scale partitions denoted X (1) , Y (1), and the finest scale partitions X (κ) , Y (κ) corresponding to the individual points in the datasets.

the following variant of [\(5\)](#page-2-2):

$$(\mathbf{Q}^*, \mathbf{R}^*) = \arg \min_{(\mathbf{Q}, \mathbf{R})} \langle \mathbf{C}, \mathbf{Q} \text{diag}(1/\mathbf{g}) \mathbf{R}^\top \rangle_F \quad (7)$$
s.t.  $\mathbf{Q} \in \Pi_{\mathbf{a}, \mathbf{g}}, \mathbf{R} \in \Pi_{\mathbf{b}, \mathbf{g}}, \mathbf{g} = (1/r)\mathbf{1}_r$ 

Proposition [3.1](#page-3-0) below is the main structural result behind Hierarchical Refinement. It says that when optimal Q<sup>⋆</sup> and R<sup>⋆</sup> for [\(7\)](#page-3-1) correspond to hard-clusterings (partitions) of each dataset, given by clustering functions q ⋆ : X → [r],r ⋆ : Y → [r], one has q <sup>⋆</sup> = r <sup>⋆</sup> ◦ T ⋆ , where T ⋆ is a Monge map.

Proposition 3.1 (Optimal low-rank factors co-cluster Monge pairs). *Let* X, Y ⊂ R <sup>d</sup> *with* |X| = |Y| = n*, with cost matrix* C *that is strictly* r*-Monge separable (Definition [B.2\)](#page-14-0). Let* a, b ∈ ∆<sup>n</sup> *be uniform so that a Monge map* T ⋆ : X → Y *exists. If* (Q<sup>⋆</sup> , R<sup>⋆</sup> ) *are minimizers of* [\(7\)](#page-3-1) *and correspond to clustering functions* q ⋆ : X → [r],r ⋆ : Y → [r]*, then for all* x ∈ X *one has* q ⋆ (x) = r ⋆ (T ⋆ (x)).

The proof of Proposition [3.1](#page-3-0) is in two steps. First, we use the existence of a Monge map and its coupling P† to permute the cost C to cost C† (Definition [B.1\)](#page-13-0) for which the identity matrix is a Monge map. Second, supposing that strict r-Monge separability (Definition [B.2\)](#page-14-0) holds, we show the solution to Problem [7](#page-3-1) with cost C† is symmetric, so that minQ,R∈Πa,<sup>g</sup> ⟨C† , QR<sup>⊤</sup>⟩<sup>F</sup> = minQ∈Πa,<sup>g</sup> ⟨C† , QQ<sup>⊤</sup>⟩<sup>F</sup> . Returning to the coordinate frame of the original cost C, we find that Q = P†R, implying Proposition [3.1.](#page-3-0) We note that when r = 2, optimal Q, R are hard-partitions (Lemma [B.5\)](#page-15-0) automatically satisfying one of the assumptions of Proposition [3.1.](#page-3-0)

## 3.2. Hierarchical Refinement Algorithm

The Hierarchical Refinement algorithm (Algorithm [1\)](#page-4-0) uses Proposition [3.1](#page-3-0) to guarantee that each low-rank step coclusters the datasets optimally, in that x and T ⋆ (x) are assigned the same label by q ⋆ and r ⋆ . Using the same label set to partition X and Y automatically places the blocks of each partition in bijective correspondence. One then recurses on each pair of corresponding blocks (which we call a *co-cluster*) at the previous scale, until all blocks have size one. This guarantee holds despite that optimal (Q<sup>⋆</sup> , R<sup>⋆</sup> ) for [\(7\)](#page-3-1) may not constitute an optimal triple (Q<sup>⋆</sup> , R<sup>⋆</sup> , g ⋆ ) for the original low-rank problem [\(5\)](#page-2-2) under the [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) factorization.

A hierarchy-depth κ denotes the total number of times Algorithm [1](#page-4-0) refines the initial trivial partitions {X}, {Y}. The effective rank at scale <sup>t</sup> is ρ<sup>t</sup> := Q<sup>t</sup> <sup>s</sup>=1 rs, given rankannealing schedule (r1, r2, . . . , rκ) for which ρ<sup>κ</sup> divides n. The base rank is rbase = n ρ<sup>κ</sup> . Note that n/ρ<sup>t</sup> is also the size of each partition at scale t: n/ρ<sup>t</sup> = |X (t) | = |Y (t) |, and that any sequence of any factorization of n corresponds to a rank-annealing schedule.

Proposition 3.2. *For any* n*, there exists a rank-schedule* (r1, · · · , rκ) *factorizing* n *such that all partitions of Algorithm [1](#page-4-0) at level* t ∈ [0 : κ − 1] *satisfy strict* rt+1*-Monge separability (Definition [B.2\)](#page-14-0). Let* LROT *denote an optimal rank-*r *solver for [\(7\)](#page-3-1) over hard-partitions. For any satisfying rank-schedule, the map returned by Algorithm [1](#page-4-0) is optimal and supported on the graph of the Monge map* T ⋆ *.*

*Proof.* Existence follows from the trivial (r1) = (n) rankschedule. For any schedule (r1, · · · , rκ) satisfying Monge separability, applying the invariant of Proposition [3.1](#page-3-0) inductively on t to level κ yields n tuples {(x, T <sup>⋆</sup> (x))} containing each x ∈ X and its image T ⋆ (x) under the Monge map.

If the black-box subroutine LROT in Algorithm [1](#page-4-0) solves [\(7\)](#page-3-1) optimally, then Hierarchical Refinement is guaranteed to recover a Monge map. In practice, we implement LROT using the low-rank solver [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) and enforce that inner marginal g is uniform.

Let Γt,q denote the q-th co-cluster at scale t generated by Hierarchical Refinement:

$$\Gamma_{t,q} := \left\{ (\mathbf{x}, \mathbf{y}) : \mathbf{x} \in X_q^{(t)}, \mathbf{y} \in Y_q^{(t)} \right\}, \quad (8)$$

where X (t) = {X (t) <sup>q</sup> } ρ<sup>t</sup> <sup>q</sup>=1, Y (t) = {Y (t) <sup>q</sup> } ρ<sup>t</sup> <sup>q</sup>=1, and define the co-clustering Γ<sup>t</sup> at scale t by:

$$\Gamma_t := \left\{ (\mathbf{X}_q^{(t)}, \mathbf{Y}_q^{(t)}) \right\}_{q=1}^{\rho_t}.$$

At scale t ∈ [κ], Hierarchical Refinement refines Γ<sup>t</sup> to Γt+1 by running a rank rt+1 low-rank optimal transport problem between uniform gt+1 = (1/rt+1)1<sup>r</sup>t+1 and measures supported on each pair (X (t) <sup>q</sup> , Y (t) <sup>q</sup> ) in Γ<sup>t</sup> for q ∈ [ρt], yielding

Algorithm 1 Hierarchical Refinement

Require: Data X , Y; Low-rank OT solver LROT(·); Rank schedule (r1, r2, . . . , rκ); Base rank rbase (=1).

Initialize: 1: t ← 0, Γ<sup>0</sup> ← { (X, Y)} 2: while ∃ (X (t) <sup>q</sup> , Y (t) <sup>q</sup> ) ∈ Γ<sup>t</sup> such that 3: min{|X (t) <sup>q</sup> |, |Y (t) <sup>q</sup> |} > rbase do 4: Γt+1 ← <sup>∅</sup> 5: for (X (t) <sup>q</sup> , Y (t) <sup>q</sup> ) ∈ Γ<sup>t</sup> do 6: if min{|X (t) <sup>q</sup> |, |Y (t) <sup>q</sup> |} ≤ rbase then 7: Γt+1 ← Γt+1 ∪ {(X (t) <sup>q</sup> , Y (t) <sup>q</sup> )} 8: else 9: µ<sup>X</sup> q = 1 |X <sup>q</sup> | P x∈X q δx 10: µ<sup>Y</sup> = 1 |Y <sup>q</sup> | P y∈Y δy. 11: gt+1 ← (1/rt+1)1<sup>r</sup>t+1 12: (Q, <sup>R</sup>) ← LROT(µ<sup>X</sup> q , µ<sup>Y</sup> q , gt+1) 13: for z = 1 → rt+1 do 14: X (t+1) <sup>z</sup> ← Assign(X (t) , Q, z) 15: Y (t+1) <sup>z</sup> ← Assign(Y (t) , R, z) 16: Γt+1 ← Γt+1 ∪ { (X (t+1) <sup>z</sup> , Y (t+1) <sup>z</sup> )} 17: end for 18: ▷ Assign(S,M, z) = {<sup>s</sup> <sup>∈</sup> <sup>S</sup> | arg maxz′ <sup>M</sup>sz′ <sup>=</sup> <sup>z</sup>} 19: end if 20: end for 21: t ← t + 1 22: end while 23: Output: Γ<sup>κ</sup> = {(x<sup>i</sup> , T(xi))<sup>n</sup> <sup>i</sup>=1} ▷ Mapped pairs.

factors specific to this q ∈ [ρt]:

$$(\mathbf{Q}, \mathbf{R}) \leftarrow \text{LROT}(\mu_{\chi_q^{(t)}}, \mu_{\gamma_q^{(t)}}, \mathbf{g}_{t+1}). \quad (9)$$

For each q ∈ [ρt] we use the Q, R from [\(9\)](#page-4-1) to co-cluster X (t) <sup>q</sup> with Y (t) <sup>q</sup> using rt+1 labels. Within this pair, each x<sup>i</sup> ∈ X (t) <sup>q</sup> is assigned a label z ∈ [rt+1] by taking the argmax over the i-th row of Q, and likewise each y<sup>j</sup> ∈ Y (t) q is assigned the argmax over the j-th row of R. This corresponds to the Assign step in Algorithm [1,](#page-4-0) and coincides with the hard assignment of q ⋆ and r ⋆ for an optimal (Q<sup>∗</sup> , R<sup>∗</sup> ) (Lemma [B.5\)](#page-15-0).

The uniform constraint g = 1<sup>r</sup>t+1 /rt+1 in [\(7\)](#page-3-1) enforces an even split of the dataset, which by Lemma [B.5](#page-15-0) ensures a partition at optimality (for r<sup>t</sup> = 2). Repeating for all q ∈ [ρt], one obtains a co-clustering with rt+1 components within each co-cluster at the previous scale, leading to a total of ρt+1 = rt+1ρ<sup>t</sup> co-clusters at scale t + 1 (Fig. [1\)](#page-3-2). If the base-case rank rbase is one, Algorithm [1](#page-4-0) returns a bijection between X and Y as a collection of n tuples.

Note that Hierarchical Refinement defines an implicit hierarchy of block-couplings at each scale t.

Definition 3.3 (Hierarchical block-coupling). For each scale t ∈ [κ], given the Hierarchical Refinement co-cluster partition Γt, the *hierarchical block-coupling* at scale t is defined by the matrix

$$\mathbf{P}_{ij}^{(t)} := \frac{\rho_t}{n^2} \sum_{q=1}^{\rho_t} \delta(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t,q}, \quad (10)$$

Without loss of generality, P(t) may be block diagonalized into ρ<sup>t</sup> square blocks, as discussed in Appendix [B](#page-13-1) (see Equation [\(S13\)](#page-16-0)). By Proposition [3.1,](#page-3-0) for any rank-schedule (r<sup>j</sup> ) κ <sup>j</sup>=1 satisfying Monge separability, the final <sup>P</sup>(κ) corresponds to an optimal coupling supported on the graph of the Monge map T ⋆ , P(κ) := (id × T ⋆ )♯µX. While these intermediate couplings are never instantiated, one can still use them to define a transport cost ⟨C, P(t) ⟩ at each scale. In Appendix [B.8,](#page-17-0) we show the following bounds on the cost difference across scales.

Proposition 3.4. *Let* c(·, ·) *be a strictly-convex and Lipschitz cost function, let* (r1, r2, · · · , rκ) *be a rank-schedule, and let* P(t) *denote the coupling defined in* [\(10\)](#page-4-2)*, obtained from step* t *of Algorithm [1.](#page-4-0) Define* ∆t,t+1 = ⟨C, P(t) ⟩<sup>F</sup> − ⟨C, P(t+1)⟩<sup>F</sup> *. Then,*

$$0 \leq \Delta_{t,t+1} \leq \|\nabla c\|_\infty \frac{1}{\rho_t} \sum_{q=1}^{\rho_t} \text{diam}(\Gamma_{t,q}), \quad (11)$$

*where* q *indexes co-clusters* Γt,q *at scale* t*, defined in* [\(8\)](#page-3-3)*.*

Thus, the lower-bound implies that each step of refinement improves the coarse partition, and the upper-bound implies that the difference in solution value is bounded above by a factor depending on the Lipschitz constant and the mean diameter of the coarse partitions at each level t. The proof of Proposition [3.4](#page-4-3) roughly follows that of Proposition 1 of [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11). In Remark [B.9,](#page-18-0) we discuss how Proposition [3.4](#page-4-3) compares, noting that our result makes fewer geometric assumptions on our multiscale partitions (X (t) ) κ <sup>t</sup>=1 and (Y (t) ) κ <sup>t</sup>=1 and therefore does not quantify the rate of decay of diam Γt,q .

### 3.3. On the Rank-Annealing Schedule

As observed by [\(Forrow et al.,](#page-10-9) [2019;](#page-10-9) [Scetbon et al.,](#page-12-9) [2021\)](#page-12-9), rank behaves like a temperature parameter, inverse to the strength ϵ of entropy regularization. The correspondence between small ϵ and large rank implies that annealing in the parameter ϵ is, from the perspective of rank, analogous to initializing the optimization at a low-rank coupling, and then gradually increasing the rank constraint from low to full. In Hierarchical Refinement, this gradual rank increase is accomplished implicitly. At each scale t = 1, . . . , κ the implicit coupling P(t) is made explicit in the hierarchical block coupling defined in equation [\(10\)](#page-4-2). A rank-annealing schedule (r1, . . . , rκ) describes the sequence of multiplicative factors by which the rank of this explicit coupling will increase

![](_page_5_Figure_1.jpeg)

Figure 2. Primal OT cost for varying sample size on the synthetic half-moon S-curve dataset of [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) for HiRef, Sinkhorn, and ProgOT

at successive scales. The partial products of these, denoted (ρ1, . . . , ρκ), are the ranks of the couplings P(1) , . . . , P(κ) . Note that small values of r<sup>i</sup> generate coarse partitions of the points at the next scale, while large values of r<sup>i</sup> generate finer partitions at the next scale.

We now turn to the question of how to efficiently choose such a schedule under given memory constraints. For an integer n, Algorithm [1](#page-4-0) has log-linear complexity for depth κ = log<sup>r</sup> n (Section [3.4\)](#page-5-0). However, the large constants required by low-rank OT in practice encourage minimizing the number of calls to LROT as a subroutine, so that if memory permits, it may be advantageous to decrease the depth by storing couplings of higher rank. If desired, memory constraints can be enforced by imposing a maximum rank rmax ≥ r<sup>t</sup> for all t ∈ [κ] to ensure Hierarchical Refinement only requires O(nrmax) space at each step. Thus, we seek factorizations with *minimal* partial sums of ranks while remaining below a desired memory-capacity:

$$\min_{(r_i)_{i=1}^\kappa} \sum_{j=1}^\kappa \rho_j \quad \text{s.t.} \quad \rho_\kappa = n, \quad r_i \leq r_{\max}. \quad (12)$$

The above optimization assumes a base-rank rbase of 1; we describe how to handle the general case in Appendix [E.1.](#page-28-0) Importantly, the recursive structure min(ri) i=1 P<sup>κ</sup> <sup>j</sup>=1 ρ<sup>j</sup> = min(ri) κ <sup>i</sup>=1 r<sup>1</sup> + r<sup>1</sup> P<sup>κ</sup> j=2 Q<sup>j</sup> <sup>i</sup>=2 r<sup>i</sup> enables a dynamic programming approach to [\(12\)](#page-5-1), storing a table of factors up to rmax to optimize [\(12\)](#page-5-1) in O(rmaxκn) time. Assuming κ, rmax are small constants chosen to ensure that all matrices can fit within memory, determining the optimal rank-schedule with respect to κ, n, rmax is a simple lineartime procedure.

## 3.4. Complexity and Scaling of Hierarchical Refinement

must store Γ<sup>t</sup> which is a set of subsets of X and Y. To derive the time-complexity of Hierarchical Refinement, note that if n = r k , a rank-r schedule at each layer requires <sup>n</sup> r instances of LROT over rapidly decaying dataset sizes. The complexity of low-rank OT [\(Scetbon et al.,](#page-12-9) [2021;](#page-12-9) [2022;](#page-12-10) [Halmos et al.,](#page-10-8) [2024\)](#page-10-8) is linear (Kn) for a constant K = O(BLrd) dependent on B the number of inner Sinkhorn [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) or Dykstra [\(Scetbon](#page-12-9) [et al.,](#page-12-9) [2021\)](#page-12-9) iterations, L the number of mirror-descent steps, r the rank of the coupling, and d the rank of the factorization of the cost matrix C. In this setting, for n a power of r, the runtime of Algorithm [1](#page-4-0) is given by the sum r <sup>0</sup>Θ(n) + r <sup>1</sup>Θ( <sup>n</sup> r ) + ... + r <sup>i</sup>−<sup>1</sup>Θ( <sup>n</sup> r <sup>i</sup>−<sup>1</sup> ) = Θ(ndr log<sup>r</sup> n) for i = log<sup>r</sup> n, achieving *linear* space with *log-linear* time for constant ranks r, d.

In cases where the cost matrix does not admit a low-rank factorization C = UV⊤, i.e., when d = O(n), one requires Θ(n 2 ) space to store the cost matrix and Hierarchical Refinement exhibits time complexity O˜(n 2 ), as in Sinkhorn. For kernel costs such as squared Euclidean cost, as noted in [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9), one may efficiently compute a (d+ 2) dimensional factorization where d is the ambient dimension, to achieve log-linear scaling with exact distances. We also use the sample-linear algorithm of [\(Indyk et al.,](#page-10-16) [2019\)](#page-10-16) to compute approximate factorizations for distances c(·, ·) satisfying metric properties such as the triangle inequality (e.g. Euclidean distance, see Appendix [E.1\)](#page-28-0). At each level, pairing such sample-linear approximations with each lowrank step only requires O(n log<sup>d</sup> n) time. We observe this scaling empirically, as reported in Fig. [S2.](#page-5-2)

## 4. Experiments

We benchmark Hierarchical Refinement (HiRef) against the full-rank OT methods Sinkhorn [\(Cuturi,](#page-9-0) [2013\)](#page-9-0), ProgOT [\(Kassraie et al.,](#page-11-19) [2024\)](#page-11-19), and mini-batch OT [\(Genevay et al.,](#page-10-3) [2018;](#page-10-3) [Fatras et al.,](#page-10-5) [2020;](#page-10-5) [2021b\)](#page-10-6). We additionally benchmark against the low-rank OT methods LOT [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) and FRLC [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8). We use the default implementations of Sinkhorn, ProgOT, and LOT in the high-performance ott-jax library [\(Cuturi et al.,](#page-9-11) [2022\)](#page-9-11). In particular, Sinkhorn is run with the default entropy regularization parameter of ϵ = 0.05. We also benchmark against the multiscale method MOP [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11), which requires multiscale partitions of the input datasets – akin to a family of dyadic cubes across scales – to compute alignments. This leads to a transport cost that depends on the choice of this partition. For simplicity, we choose the default partitions of MOP which are computed from the GMRA (Geometric Multi-Resolution Analysis) R package.

![](_page_6_Picture_1.jpeg)

Figure 3. Comparison of the Hierarchical Refinement Mapping, the Sinkhorn Barycentric Map, and an optimal map computing using dual revised simplex for the a. Half-moon and S-curve dataset [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) of 4096 points (512 points for dual revised simplex) and b. Checkerboard dataset [\(Makkuva et al.,](#page-11-9) [2020\)](#page-11-9).

### 4.1. Evaluation on Synthetic Datasets.

We first evaluate the performance of Hierarchical Refinement against optimal transport methods returning primal couplings, namely Sinkhorn [\(Cuturi,](#page-9-0) [2013\)](#page-9-0) (as implemented in ott-jax [\(Cuturi et al.,](#page-9-11) [2022\)](#page-9-11)) and ProgOT [\(Kassraie](#page-11-19) [et al.,](#page-11-19) [2024\)](#page-11-19). We evaluate the methods with respect to the Wasserstein-1 and Wasserstein-2 distance on an alignment of 1024 pairs of samples on the Checkerboard [\(Makkuva](#page-11-9) [et al.,](#page-11-9) [2020\)](#page-11-9), MAFMoons and Rings [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5), and Half-Moon and S-Curve [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) synthetic datasets (Fig. [3,](#page-6-0) Table [S6\)](#page-26-0).

All methods are similarly effective at minimizing the primal OT cost ⟨C, P⟩<sup>F</sup> , with small absolute difference in cost between the final couplings. Hierarchical Refinement achieves slightly lower primal cost on 4 out of the 6 evaluations. Notably, there is a massive difference in the number of non-zero entries (defined as entries Pij > 10−<sup>8</sup> ) in the couplings output by HiRef, Sinkhorn, and ProgOT (Table [S3\)](#page-24-0). Specifically, across the experiments HiRef outputs a bijection with exactly 1024 non-zero elements in the coupling matrix, equal to the number of aligned samples. In constrast, Sinkhorn and ProgOT output couplings with 624733 to 678720 and 271087 to 337258 non-zero entries.

We evaluate the scalability of Hierarchical Refinement relative to other full-rank solvers on varying numbers of samples from the Half Moon & S-Curve [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) synthetic dataset. We vary the rank from 2 <sup>5</sup> = 32 (64 points aligned) up to 2 <sup>20</sup> = 1048576 points (2097152 points aligned) in R 2 , the latter dataset of a size that is beyond the capabilities of current optimal transport solvers. We observe that Sinkhorn [\(Cuturi,](#page-9-0) [2013\)](#page-9-0) and ProgOT – methods which produce dense mappings – require a coupling matrix with O(n ) non-zero entries and thus run only up

to 16384 points. HiRef yields solutions with comparable primal cost to ProgOT and Sinkhorn on the sample sizes where all methods run.

We also find that HiRef achieves an OT cost that is competitive with the dual revised simplex solver [\(Huangfu &](#page-10-17) [Hall,](#page-10-17) [2018\)](#page-10-17), a solver which only scales up to 512 points (Table [S4\)](#page-24-1). This solver computes an *optimal* coupling, unlike ProgOT and Sinkhorn which rely on entropic regularization. While we benchmark Sinkhorn in place of mini-batch OT on the synthetic datasets due to their limited complexity, we also evaluate the multi-scale method MOP on the 512 point instance (Table [S4\)](#page-24-1). Although MOP outputs a fast approximation to optimal transport, its primal cost on the Checkerboard [\(Makkuva et al.,](#page-11-9) [2020\)](#page-11-9) dataset is twice as high as that of the other methods, and it performs significantly worse on the MAF Moons & Rings and Half Moon & S-Curve datasets [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5).

Lastly, we observe that Hierarchical Refinement scales to over a million points, two orders of magnitude greater than ProgOT and Sinkhorn, two full-rank OT methods that compute global alignments. We find HiRef scales linearly with the size of the problem instance (Fig. [S2a](#page-5-2)) in contrast to the quadratic scaling in time complexity of Sinkhorn (Fig. [S2b](#page-5-2)).

### 4.2. Large-scale Matching Problems and Transcriptomics

Recently, optimal transport has been applied to single-cell and spatial transcriptomics datasets to compute couplings between cells taken from different timepoints from developmental processes or perturbations [\(Schiebinger et al.,](#page-12-4) [2019;](#page-12-4) [Lavenant et al.,](#page-11-20) [2024;](#page-11-20) [Bunne et al.,](#page-9-12) [2022;](#page-9-12) [Huizing et al.,](#page-10-18) [2024;](#page-10-18) [Halmos et al.,](#page-10-1) [2025b;](#page-10-1) [Klein et al.,](#page-11-1) [2025\)](#page-11-1). However, the size of current datasets [\(Chen et al.,](#page-9-13) [2022\)](#page-9-13) (>100k cells)

Table 1. Cost Values ⟨C, P⟩<sup>F</sup> Across Later Embryonic Stages

| Method         | E12-13.5 | E13-14.5 | E14-15.5 | E15-16.5 |
|----------------|----------|----------|----------|----------|
| HiRef Sinkhorn | 14.35    | 13.78    | 14.29    | 12.79    |
| MB 128         | 14.86    | 14.14    | 14.75    | 13.32    |
| MB 1024        | 14.45    | 13.86    | 14.43    | 12.91    |
| FRLC           | 15.47    | 14.64    | 15.51    | 14.00    |

has exceeded the capacity of existing full-rank solvers, requiring low-rank approximations of the coupling [\(Scetbon](#page-12-12) [et al.,](#page-12-12) [2023;](#page-12-12) [Klein et al.,](#page-11-1) [2025;](#page-11-1) [Halmos et al.,](#page-10-10) [2025a\)](#page-10-10) to produce alignments.

We evaluate whether the full-rank solver of Hierarchical Refinement exhibits competitive alignments for such datasets. Specifically, we analyze the mouse organogenesis spatiotemporal transcriptomic atlas (MOSTA) datasets, which include spatial transcriptomics data from mouse embryos at successive 1-day time-intervals with increasing number n of cells at each stage: E9.5 (n = 5913), E10.5 (n = 18408), E11.5 (n = 30124), E12.5 (n = 51365), E13.5 (n = 77369), E14.5 (n = 102519), E15.5 (n = 113350), and E16.5 (n = 121767). For the cost we use the Euclidean distance Cij = ∥x<sup>i</sup> − yj∥<sup>2</sup> in 60-dimensional PCA space of expression vectors, so x<sup>i</sup> , y<sup>j</sup> ∈ <sup>R</sup> 60 .

Sinkhorn and ProgOT are unable to produce alignments for the stages beyond E10.5 (n = 18408 cells), whereas HiRef, the low-rank solvers, and mini-batch OT (batchsizes B = 128 to B = 2048) are able to continue scaling to > 10<sup>5</sup> (Table [1,](#page-7-0) Table [S6\)](#page-26-0). We observe that the Kantorovich cost of HiRef is consistently lower than all other methods for all timepoints (Table [1,](#page-7-0) Table [S6\)](#page-26-0).

HiRef achieves a substantially lower cost than the lowrank solvers FRLC and LOT for rank r = 40, even though HiRef relies on low-rank optimal transport (FRLC) as a subroutine. This result underscores the empirical trend observed in Fig. [S3,](#page-6-0) where the refinement step of HiRef progressively decreases the primal cost of coarser low-rank couplings (Proposition [3.4\)](#page-4-3). While the mini-batch solvers exhibit competitive scaling up to the last pair, the primal cost of mini-batch is higher for all tested batch-sizes (Table [S6\)](#page-26-0). Unlike HiRef, mini-batch OT does not compute a global alignment and exhibits batch-size dependent error.

### 4.3. MERFISH Brain Atlas Alignment

We ran HiRef on two slices of MERFISH Mouse Brain Receptor Map data from [Vizgen](https://info.vizgen.com/mouse-brain-map) to test whether HiRef can produce biologically valid alignments using the *only* spatial densities of each tissue. These spatial transcriptomics data consist of spatial and gene expression measurements at individual spots in three full coronal slices across three biological replicates. Our "source" dataset (X<sup>1</sup> , S 1 ) is replicate 3 of slice 2, while our "target" dataset (X<sup>2</sup> , S 2 ) is replicate 2 of slice 2, following the expression transfer task described [\(Clifton et al.,](#page-9-14) [2023\)](#page-9-14) between these two slices. Each dataset has roughly 84k spots, where memory constraints prohibit instantiation a full-rank alignment as a matrix. Thus, solvers such as Sinkhorn [\(Cuturi,](#page-9-0) [2013\)](#page-9-0) and ProgOT [\(Kassraie et al.,](#page-11-19) [2024\)](#page-11-19) are unable to run on the dataset.

We use only spatial information when building a map between the two slices, using the spatial Euclidean cost Cij := ∥s 1 <sup>i</sup> − s 2 j ∥2, after registering spatial coordinates S <sup>1</sup> = {s i } n <sup>i</sup>=1 and S <sup>2</sup> = {s 2 i } n <sup>i</sup>=1 with an affine transformation. We gauged the quality of the HiRef alignment (Fig. [4a](#page-8-0)), using gene expression abundances of five "spatially-varying" genes. Specifically, we observe that expression vector v <sup>1</sup> of gene *Slc17a7* in the source slice ( Fig. [4b](#page-8-0)) when transferred to target slice through the bijective mapping output by HiRef, denoted as vˆ (Fig. [4c](#page-8-0)), closely matches the observed expression vector v <sup>2</sup> of *Slc17a7* in the target slice (Fig. [4d](#page-8-0)) with cosine similarity equal to 0.8098. For genes *Slc17a7*, *Grm4*, *Olig1*, *Gad1*, *Peg10*, the corresponding cosine similarities between the transferred and observed expression vectors are 0.8098, 0.7959, 0.7526, 0.4932, 0.6015, respectively.

For comparison, we also ran the low-rank methods FRLC [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) and LOT [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) with and without subsampling, reporting their best scores, as discussed in Section [D.3.](#page-25-0) For the gene *Slc17a7*, FRLC's cosine similarity was 0.2373, while LOT's cosine similarity was 0.3390. For all five genes *Slc17a7*, *Grm4*, *Olig1*, *Gad1*, *Peg10*, FRLC's scores were (0.2373, 0.2124, 0.1929, 0.0963, 0.1550, respectively, while LOT's scores were 0.3390, 0.2712, 0.3186, 0.1666, 0.1080. Across all five genes HiRef's scores were at least twice those of FRLC or LOT (Table [S7\)](#page-27-0) with gene abundances shown in Fig. [S1.](#page-3-2) On the same task, we compared against MOP, the method of [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11), whose scores for the five genes were: (0.5211, 0.4714, 0.5972, 0.3571, 0.2719). Finally, we also benchmarked against mini-batch OT using batch sizes ranging from 128 to 2048 in powers of two, whose best scores (0.7434, 0.7822, 0.7056, 0.4912, 0.5683) were more comparable to that of the performance of HiRef. Across all methods and genes compared in Table [S7,](#page-27-0) HiRef had greatest cosine similarity scores in the expression transfer task, while also having lowest transport cost. Further experimental details are in Section [D.3.](#page-25-0)

### 4.4. ImageNet Alignment

We demonstrate the scalability of Hierarchical Refinement on a large-scale and high-dimensional dataset by aligning 2048-dimensional embeddings of 1.281 million images from the ImageNet ILSVRC dataset [\(Deng et al.,](#page-10-19) [2009;](#page-10-19)

![](_page_8_Figure_1.jpeg)

![](_page_8_Picture_2.jpeg)

Figure 4. a. Hierarchical Refinement alignment on MERFISH mouse brain data, using only spatial coordinates. b. Abundance v 1 of gene *Slc17a7* in the source slice. c. Predicted *Slc17a7* abundance vˆ from the source slice to the target slice, through the HiRef coupling. d. Abundance v 2 of the same gene in the target slice. Transferred abundances vˆ have cosine similarity 0.8098 with true abundances v 2 in the target.

Table 2. Cost Values ⟨C, P⟩<sup>F</sup> for ImageNet [\(Deng et al.,](#page-10-19) [2009;](#page-10-19) [Russakovsky et al.,](#page-11-21) [2015\)](#page-11-21) Alignment Task.

| <b>Method</b>  | <b>HireRef</b> | <b>MB 128</b> | <b>MB 256</b> | <b>MB 512</b> | <b>MB 1024</b> | <b>FRLC</b> |
|----------------|----------------|---------------|---------------|---------------|----------------|-------------|
| <b>OT Cost</b> | <b>18.97</b>   | 21.89         | 21.11         | 20.34         | 19.58          | 24.12       |

[Russakovsky et al.,](#page-11-21) [2015\)](#page-11-21). Each image is embedded using using the ResNet50 architecture [\(He et al.,](#page-10-20) [2016\)](#page-10-20), and we construct two datasets, X and Y, by taking a random 50:50 split of the embedded images. We align X and Y using HiRef, FRLC, and mini-batch OT with batch-sizes ranging from B = 128 to B = 1024. ProgOT, Sinkhorn, and LOT could not be run on the datasets due to memory constraints. HiRef yielded a primal OT cost of 18.974, while FRLC [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) solution had a primal OT cost of 24.119 for rank r = 40 and mini-batch OT has costs of 21.89 (B = 128) to 19.58 (B = 1024) (Table [2\)](#page-8-1).

## 5. Discussion

Hierarchical Refinement computes the Monge map between large-scale datasets in linear space, but has several limitations. First, we currently assume that the datasets X and Y have the same number of samples. In many machine learning applications, this is not a limiting factor, as one generally seeks to pair an equal number of source points x to target points y. Second, while Hierarchical Refinement scales linearly in space and log-linearly in time, it still involves a constant dependent on the low-rank OT subprocedure used – this underscores the need to accelerate and stabilize low-rank OT solvers further [\(Scetbon & Cu](#page-12-11)[turi,](#page-12-11) [2022;](#page-12-11) [Halmos et al.,](#page-10-8) [2024\)](#page-10-8). Finally, while Hierarchical Refinement guarantees an optimal solution given an optimal black-box low-rank solver (Proposition [3.1\)](#page-3-0), the low-rank solvers [\(Scetbon et al.,](#page-12-10) [2022;](#page-12-10) [Halmos et al.,](#page-10-8) [2024\)](#page-10-8) used in practice are not necessarily optimal, owing to the nonconvexity of low-rank problems.

Optimal transport has been successfully applied in deep learning frameworks, such as OT flow-matching [\(Tong et al.,](#page-12-7) [2024\)](#page-12-7), computer vision and point cloud registration, [\(Yu](#page-12-17) [et al.,](#page-12-17) [2021;](#page-12-17) [Qin et al.,](#page-11-22) [2022\)](#page-11-22), among many others. The mini-batch procedure used to train many of these methods involves sampling two datasets X<sup>B</sup> ∼ µ and Y<sup>B</sup> ∼ ν with batch-size B and aligning them with Sinkhorn at every training iteration. HiRef suggests an alternative approach: one can precompute millions of *globally aligned* pairs and then sample X<sup>B</sup> ∼ µ and the optimal mapping T(XB) ∼ ν by indexing into these precomputed pairs. This approach applies to any loss function dependent on an OT alignment.

Hierarchical Refinement may also be useful in neural OT approaches which learn a continuous Monge map between the densities of two datasets. For example, [\(Seguy et al.,](#page-12-18) [2018\)](#page-12-18) minimize a loss min<sup>θ</sup> 2 <sup>E</sup>µ∥Tθ(xi) − T(xi)∥ 2 <sup>2</sup> between a neural network T<sup>θ</sup> with parameters θ and a Monge map T over samples x<sup>i</sup> ∼ µ (Remark [B.11\)](#page-19-0). Thus, the procedure outlined above may be used to directly regress a neural network T<sup>θ</sup> on the Monge map T without the bias of mini-batching or entropy.

# 6. Conclusion

We introduce Hierarchical Refinement (HiRef), an algorithm to solve optimal transport with linear complexity in the number of points, making sparse, full-rank optimal transport feasible for large-scale datasets. Our algorithm leverages that low-rank optimal transport co-clusters points with their image under the Monge map, refining bijections between partitions of each dataset across a hierarchy of scales, down to a bijective Monge map between the datasets at the finest scale. Hierarchical Refinement couplings achieve comparable primal cost to couplings obtained through full-rank entropic solvers, and scales to datasets with over a million points, opening the door to applications previously infeasible for optimal transport.

- Acknowledgements We thank Henri Schmidt for many helpful conversations. This research was supported by NIH/NCI grant U24CA248453 to B.J.R. J.G. is supported by the Schmidt DataX Fund at Princeton University made possible through a major gift from the Schmidt Futures Foundation. Impact Statement Optimal transport has emerged as a powerful tool in generative modeling, yet its widespread use has been limited by scalability constraints. HiRef overcomes this limitation by enabling the application of OT to datasets with millions of points. This advancement paves the way for integrating OT into large-scale deep generative models and modern vision and language tasks. As with any computational tool which may advance largescale generative modeling, there are potential issues with bias in training datasets and a possibility of misuse. Use of HiRef in applications should be careful and transparent about these risks and utilize appropriate mitigation strategies. Code Availability Our implementation of Hierarchical Refinement is available at [https://github.com/raphael-group/HiRef.](https://github.com/raphael-group/HiRef) References Aurenhammer, F., Hoffmann, F., and Aronov, B. Minkowski-type theorems and least-squares clustering. *Algorithmica*, 20:61–76, 1998. Birkhoff, G. Tres observaciones sobre el algebra lineal. *Univ. Nac. Tucuman, Ser. A*, 5:147–154, 1946. Bonneel, N., Rabin, J., Peyre, G., and Pfister, H. Sliced and ´ Radon Wasserstein barycenters of measures. *Journal of Mathematical Imaging and Vision*, 51:22–45, 2015. Brenier, Y. Polar factorization and monotone rearrangement of vector-valued functions. *Communications on pure and applied mathematics*, 44(4):375–417, 1991. Bunne, C., Papaxanthos, L., Krause, A., and Cuturi, M. Proximal optimal transport modeling of population dynamics. In *International Conference on Artificial Intelligence and Statistics*, volume 25, pp. 6511–6528. PMLR, 2022. Bunne, C., Stark, S. G., Gut, G., del Castillo, J. S., Levesque, M., Lehmann, K.-V., Pelkmans, L., Krause, A., and Ratsch, G. Learning single-cell perturbation responses ¨ using neural optimal transport. *Nature Methods*, 20(11): 1759–1768, 2023. Buzun, N., Bobrin, M., and Dylov, D. V. Expectile regularization for fast and accurate training of neural optimal transport. In *Advances in Neural Information Processing Systems*, volume 37, pp. 119811–119837, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=4DA5vaPHFb) [id=4DA5vaPHFb](https://openreview.net/forum?id=4DA5vaPHFb). Chen, A., Liao, S., Cheng, M., Ma, K., Wu, L., Lai, Y., Qiu, X., Yang, J., Xu, J., Hao, S., et al. Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. *Cell*, 185(10):1777– 1792, 2022. Chen, J., Chen, L., Liu, Y. P., Peng, R., and Ramaswami,
  - A. Exponential convergence of Sinkhorn under regularization scheduling. In *SIAM Conference on Applied and Computational Discrete Algorithms*, pp. 180–188. SIAM, 2023. Chen, X. and Price, E. Condition number-free query and active learning of linear families. *CoRR, abs/1711.10051*, 24, 2017. Chen, Y., Georgiou, T. T., and Tannenbaum, A. Optimal transport for Gaussian mixture models. *IEEE Access*, 7: 6269–6278, 2018. Clifton, K., Anant, M., Aihara, G., Atta, L., Aimiuwu, O. K., Kebschull, J. M., Miller, M. I., Tward, D., and Fan, J. STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping. *Nature Communications*, 14(1):8123, 2023. Cohen, J. E. and Rothblum, U. G. Nonnegative ranks, decompositions, and factorizations of nonnegative matrices. *Linear Algebra and its Applications*, 190:149–168, 1993. Courty, N., Flamary, R., and Tuia, D. Domain adaptation with regularized optimal transport. In *European Conference on Machine Learning and Knowledge Discovery in Databases*, pp. 274–289. Springer, 2014. Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in Neural Information Processing Systems*, 26:2292–2300, 2013. Cuturi, M., Meng-Papaxanthos, L., Tian, Y., Bunne, C., Davis, G., and Teboul, O. Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein. *arXiv preprint arXiv:2201.12324*, 2022. De Bortoli, V., Korshunova, I., Mnih, A., and Doucet, A. Schrodinger bridge flow for unpaired data translation. ¨ *Advances in Neural Information Processing Systems*, 37: 103384–103441, 2024. URL [https://openreview.](https://openreview.net/forum?id=1F32iCJFfa) [net/forum?id=1F32iCJFfa](https://openreview.net/forum?id=1F32iCJFfa).

- De Loera, J. A. and Kim, E. D. Combinatorics and geometry of transportation polytopes: An update. *Discrete Geometry and Algebraic Combinatorics*, 625:37–76, 2013. Delon, J. and Desolneux, A. A Wasserstein-type distance in the space of Gaussian mixture models. *SIAM Journal on Imaging Sciences*, 13(2):936–970, 2020. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,
- L. ImageNet: A large-scale hierarchical image database. In *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255. IEEE, 2009. El Hamri, M., Bennani, Y., and Falih, I. Hierarchical optimal transport for unsupervised domain adaptation. *Machine Learning*, 111(11):4159–4182, 2022. Fan, J., Liu, S., Ma, S., Zhou, H.-M., and Chen, Y. Neural Monge map estimation and its applications. *Transactions on Machine Learning Research*, 2023. URL [https:](https://openreview.net/forum?id=2mZSlQscj3) [//openreview.net/forum?id=2mZSlQscj3](https://openreview.net/forum?id=2mZSlQscj3). Fatras, K., Zine, Y., Flamary, R., Gribonval, R., and Courty,
- N. Learning with minibatch Wasserstein: asymptotic and gradient properties. In *International Conference on Artificial Intelligence and Statistics*, volume 108, pp. 2131– 2141. PMLR, 2020. URL [http://proceedings.](http://proceedings.mlr.press/v108/fatras20a.html) [mlr.press/v108/fatras20a.html](http://proceedings.mlr.press/v108/fatras20a.html). Fatras, K., Sejourn ´ e, T., Flamary, R., and Courty, N. Un- ´ balanced minibatch optimal transport; applications to domain adaptation. In *International Conference on Machine Learning*, volume 139, pp. 3186–3197. PMLR, 2021a. URL [http://proceedings.mlr.press/v139/](http://proceedings.mlr.press/v139/fatras21a.html) [fatras21a.html](http://proceedings.mlr.press/v139/fatras21a.html). Fatras, K., Zine, Y., Majewski, S., Flamary, R., Gribonval, R., and Courty, N. Minibatch optimal transport distances; analysis and applications. *arXiv preprint arXiv:2101.01792*, 2021b. Finlay, C., Jacobsen, J.-H., Nurbekyan, L., and Oberman,
- A. How to train your neural ODE: the world of Jacobian and kinetic regularization. In *International Conference on Machine Learning*, pp. 3154–3164. PMLR, 2020. Forrow, A., Hutter, J.-C., Nitzan, M., Rigollet, P., ¨ Schiebinger, G., and Weed, J. Statistical optimal transport via factored couplings. In *International Conference on Artificial Intelligence and Statistics*, volume 89, pp. 2454– 2465. PMLR, 2019. URL [https://proceedings.](https://proceedings.mlr.press/v89/forrow19a.html) [mlr.press/v89/forrow19a.html](https://proceedings.mlr.press/v89/forrow19a.html). Frieze, A., Kannan, R., and Vempala, S. Fast Monte-Carlo Algorithms for Finding Low-rank Approximations. *J. ACM*, 51(6):1025–1041, nov 2004. ISSN 0004-5411. doi: 10.1145/1039488.1039494. URL [https://doi.](https://doi.org/10.1145/1039488.1039494) [org/10.1145/1039488.1039494](https://doi.org/10.1145/1039488.1039494). Genevay, A., Peyre, G., and Cuturi, M. Learning gen- ´ erative models with Sinkhorn divergences. In *International Conference on Artificial Intelligence and Statistics*, volume 84, pp. 1608–1617. PMLR, 2018. URL [https://proceedings.mlr.press/v84/](https://proceedings.mlr.press/v84/genevay18a.html) [genevay18a.html](https://proceedings.mlr.press/v84/genevay18a.html). Gerber, S. and Maggioni, M. Multiscale strategies for computing optimal transport. *Journal of Machine Learning Research*, 18(72):1–32, 2017. Geshkovski, B., Letrouit, C., Polyanskiy, Y., and Rigollet,
  - P. A mathematical perspective on Transformers. *arXiv preprint arXiv:2312.10794*, 2023. Glimm, T. and Henscheid, N. Iterative scheme for solving optimal transportation problems arising in reflector design. *International Scholarly Research Notices*, 2013(1): 635263, 2013. Halmos, P., Liu, X., Gold, J., and Raphael, B. Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=hGgkdFF2hR) [id=hGgkdFF2hR](https://openreview.net/forum?id=hGgkdFF2hR). Halmos, P., Gold, J., Liu, X., and Raphael, B. J. Learning latent trajectories in developmental time series with Hidden-Markov optimal transport. In *International Conference on Research in Computational Molecular Biology*, pp. 367–370. Springer, 2025a. Halmos, P., Liu, X., Gold, J., Chen, F., Ding, L., and Raphael, B. J. DeST-OT: Alignment of spatiotemporal transcriptomics data. *Cell Systems*, 16(2), 2025b. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016. Huangfu, Q. and Hall, J. A. J. Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10(1):119–142, 2018. Huizing, G.-J., Peyre, G., and Cantini, L. Learn- ´ ing cell fate landscapes from spatial transcriptomics using Fused Gromov-Wasserstein. *bioRxiv preprint bioRxiv:2024.07.26.605241*, 2024. Indyk, P., Vakilian, A., Wagner, T., and Woodruff, D. P. Sample-optimal low-rank approximation of distance matrices. In *Conference on Learning Theory*, volume 99, pp. 1723–1751. PMLR, 2019. Kantorovich, L. On the translocation of masses. *Doklady Akademii Nauk SSSR*, 37(7-8):227–229, 1942.

Kassraie, P., Pooladian, A.-A., Klein, M., Thornton, J., Niles-Weed, J., and Cuturi, M. Progressive entropic optimal transport solvers. *Advances in Neural Information Processing Systems*, 37:19561–19590, 2024. Klein, D., Uscidda, T., Theis, F. J., and Cuturi, M. Generative entropic neural optimal transport to map within and across space, 2024. URL [https://openreview.](https://openreview.net/forum?id=gBLEHzKOfF) [net/forum?id=gBLEHzKOfF](https://openreview.net/forum?id=gBLEHzKOfF). Klein, D., Palla, G., Lange, M., Klein, M., Piran, Z., Gander, M., Meng-Papaxanthos, L., Sterr, M., Saber, L., Jing, C., Bastidas-Ponce, A., Cota, P., Tarquis-Medina, M., Parikh, S., Gold, I., Lickert, H., Bakhti, M., Nitzan, M., Cuturi, M., and Theis, F. J. Mapping cells through time and space with moscot. *Nature*, pp. 1–11, 2025. Kornilov, N., Mokrov, P., Gasnikov, A., and Korotin, A. Optimal flow matching: Learning straight trajectories in just one step. *Advances in Neural Information Processing Systems*, 37:104180–104204, 2024. URL [https://](https://openreview.net/forum?id=kqmucDKVcU) [openreview.net/forum?id=kqmucDKVcU](https://openreview.net/forum?id=kqmucDKVcU). Korotin, A., Li, L., Genevay, A., Solomon, J. M., Filippov, A., and Burnaev, E. Do neural optimal transport solvers work? A continuous Wasserstein-2 benchmark. *Advances in Neural Information Processing Systems*, 34:14593– 14605, 2021. Korotin, A., Selikhanovych, D., and Burnaev, E. Neural optimal transport. *International Conference on Learning Representations*, 2023. URL [https://openreview.](https://openreview.net/forum?id=d8CBRlWNkqH) [net/forum?id=d8CBRlWNkqH](https://openreview.net/forum?id=d8CBRlWNkqH). Kuhn, H. W. The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2): 83–97, 1955. Lavenant, H., Zhang, S., Kim, Y.-H., and Schiebinger, G. Toward a mathematical theory of trajectory inference. *The Annals of Applied Probability*, 34(1A):428–500, 2024. Li, X., Chen, J., Chai, Y., and Xiong, H. GiLOT: Interpreting generative language models via optimal transport. *International Conference on Machine Learning*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=qKL25sGjxL) [id=qKL25sGjxL](https://openreview.net/forum?id=qKL25sGjxL). Lin, C.-H., Azabou, M., and Dyer, E. L. Making transport more robust and interpretable by moving data through a small number of anchor points. *International Conference on Machine Learning*, 139:6631, 2021. Luo, J., Yang, D., and Wei, K. Improved complexity analysis of the sinkhorn and greenkhorn algorithms for optimal transport. *arXiv preprint arXiv:2305.14939*, 2023. Makkuva, A., Taghvaei, A., Oh, S., and Lee, J. Optimal transport mapping via input convex neural networks. *International Conference on Machine Learning*, 119:6672– 6681, 2020. Melnyk, I., Mroueh, Y., Belgodere, B., Rigotti, M., Nitsure, A., Yurochkin, M., Greenewald, K., Navratil, J., and Ross, J. Distributional preference alignment of LLMs via optimal transport. *Advances in Neural Information Processing Systems*, 2024. URL [https:](https://openreview.net/forum?id=2LctgfN6Ty) [//openreview.net/forum?id=2LctgfN6Ty](https://openreview.net/forum?id=2LctgfN6Ty). Merigot, Q. A multiscale approach to optimal transport. ´ *Computer Graphics Forum*, 30(5):1583–1592, 2011. Monge, G. Memoire sur la th ´ eorie des d ´ eblais et des rem- ´ blais. *Mem. Math. Phys. Acad. Royale Sci.*, pp. 666–704, 1781. Nguyen, K., Nguyen, D., Pham, T., and Ho, N. Improving mini-batch optimal transport via partial transportation. In *Proceedings of the 39th International Conference on Machine Learning*, 2022a. Nguyen, K., Ren, T., Nguyen, H., Rout, L., Nguyen, T. M., and Ho, N. Hierarchical sliced Wasserstein distance. *International Conference on Learning Representations*, 2022b. Oberman, A. M. and Ruan, Y. An efficient linear programming method for optimal transportation. *arXiv preprint arXiv:1509.03668*, 2015. Orlin, J. B. A polynomial time primal network simplex algorithm for minimum cost flows. *Mathematical Programming*, 78(2):109–129, 1997. Peyre, G. and Cuturi, M. Computational optimal transport: ´ With applications to data science. *Foundations and Trends in Machine Learning*, 11(5–6):355–607, 2019. Qin, Z., Yu, H., Wang, C., Guo, Y., Peng, Y., and Xu, K. Geometric transformer for fast and robust point cloud registration. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11143–11152, 2022. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. *International Conference on Machine Learning*, 139:8821–8831, 2021. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3):211–252, 2015.

Sander, M. E., Ablin, P., Blondel, M., and Peyre, G. Sink- ´ formers: Transformers with doubly stochastic attention. In *International Conference on Artificial Intelligence and Statistics*, pp. 3515–3530. PMLR, 2022. Scetbon, M. and Cuturi, M. Low-rank optimal transport: Approximation, statistics and debiasing. *Advances in Neural Information Processing Systems*, 35:6802–6814, 2022. Scetbon, M., Cuturi, M., and Peyre, G. Low-rank Sinkhorn ´ factorization. *International Conference on Machine Learning*, pp. 9344–9354, 2021. Scetbon, M., Peyre, G., and Cuturi, M. Linear-time Gro- ´ mov Wasserstein distances using low rank couplings and costs. *International Conference on Machine Learning*, pp. 19347–19365, 2022. Scetbon, M., Klein, M., Palla, G., and Cuturi, M. Unbalanced low-rank optimal transport solvers. *Advances in Neural Information Processing Systems*, 36:52312– 52325, 2023. Schiebinger, G., Shu, J., Tabaka, M., Cleary, B., Subramanian, V., Solomon, A., Gould, J., Liu, S., Lin, S., and Berube, P. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell*, 176(4):928–943, 2019. Schmitzer, B. A sparse multiscale algorithm for dense optimal transport. *Journal of Mathematical Imaging and Vision*, 56:238–259, 2016. Schmitzer, B. and Schnorr, C. A hierarchical approach to ¨ optimal transport. In *International Conference on Scale Space and Variational Methods in Computer Vision*, pp. 452–464. Springer, 2013. Seguy, V., Damodaran, B. B., Flamary, R., Courty, N., Rolet, A., and Blondel, M. Large-scale optimal transport and mapping estimation. *International Conference on Learning Representations*, 2018. Solomon, J., De Goes, F., Peyre, G., Cuturi, M., Butscher, ´ A., Nguyen, A., Du, T., and Guibas, L. Convolutional Wasserstein distances: Efficient optimal transportation on geometric domains. *ACM Transactions on Graphics*, 34 (4):1–11, 2015. Sommerfeld, M., Schrieber, J., Zemel, Y., and Munk, A. Optimal transport: Fast probabilistic approximation with exact solvers. *Journal of Machine Learning Research*, 20 (105):1–23, 2019. Stahl, P. L., Salm ˚ en, F., Vickovic, S., Lundmark, A., ´ Navarro, J. F., Magnusson, J., Giacomello, S., Asp, M., Westholm, J. O., and Huss, M. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. *Science*, 353(6294):78–82, 2016. Tarjan, R. E. Dynamic trees as search trees via Euler tours, applied to the network simplex algorithm. *Mathematical Programming*, 78(2):169–177, 1997. Tay, Y., Bahri, D., Yang, L., Metzler, D., and Juan, D.-C. Sparse Sinkhorn attention. *International Conference on Machine Learning*, 119:9438–9447, 2020. Thorpe, M. Introduction to optimal transport. *Notes of Course at University of Cambridge*, 2018. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., and Bengio, Y. Improving and generalizing flow-based generative models with minibatch optimal transport. *Transactions on Machine Learning Research*, 2024. URL [https://openreview.](https://openreview.net/forum?id=CD9Snc73AW) [net/forum?id=CD9Snc73AW](https://openreview.net/forum?id=CD9Snc73AW). Wolf, F. A., Angerer, P., and Theis, F. J. SCANPY: Largescale single-cell gene expression data analysis. *Genome Biology*, 19:1–5, 2018. Yang, K. D., Damodaran, K., Venkatachalapathy, S., Soylemezoglu, A. C., Shivashankar, G. V., and Uhler, C. Predicting cell lineages using autoencoders and optimal transport. *PLoS Computational Biology*, 16(4):e1007828, 2020. Yu, H., Li, F., Saleh, M., Busam, B., and Ilic, S. CoFiNet: Reliable coarse-to-fine correspondences for robust pointcloud registration. *Advances in Neural Information Processing Systems*, 34:23872–23884, 2021. Yurochkin, M., Claici, S., Chien, E., Mirzazadeh, F., and Solomon, J. M. Hierarchical optimal transport for document representation. *Advances in Neural Information Processing Systems*, 32, 2019. Zeira, R., Land, M., Strzalkowski, A., and Raphael, B. J. Alignment and integration of spatial transcriptomics data. *Nature Methods*, 19(5):567–575, 2022.

## A. Hierarchical-Refinement Algorithm

Algorithm 2 Hierarchical Refinement for Full-Rank OT Require: Datasets X = {xi} n <sup>i</sup>=1, Y = {yi} n <sup>i</sup>=1; Low-rank OT solver LROT(·); Rank schedule (r1, r2, . . . , rκ); Base rank <sup>r</sup>base <sup>=</sup> <sup>Q</sup> <sup>n</sup> κ <sup>t</sup>=1 r<sup>t</sup> (e.g. 1). Initialize: 1: t ← 0, Γ<sup>0</sup> ← { (X, Y)} 2: while ∃ (X (t) , Y (t) ) ∈ Γ<sup>t</sup> such that 3: min{|X (t) |, |Y (t) |} > rbase do 4: Γt+1 ← <sup>∅</sup> 5: for (X (t) <sup>q</sup> , Y (t) <sup>q</sup> ) ∈ Γ<sup>t</sup> do 6: if min{|X (t) <sup>q</sup> |, |Y (t) <sup>q</sup> |} ≤ rbase then 7: Γt+1 ← Γt+1 ∪ {(X (t) <sup>q</sup> , Y (t) <sup>q</sup> )} 8: else 9: µ<sup>X</sup> q = |X <sup>q</sup> | P x∈X q δx 10: µ<sup>Y</sup> = 1 |Y <sup>q</sup> | P y∈Y δy. 11: gt+1 ← (1/rt+1)1<sup>r</sup>t+1 12: (Q, <sup>R</sup>) ← LROT(µ<sup>X</sup> q , µ<sup>Y</sup> q , gt+1) 13: for z = 1 → rt+1 do 14: X (t+1) <sup>z</sup> ← Assign(X (t) , Q, z) 15: Y (t+1) <sup>z</sup> ← Assign(Y (t) , R, z) 16: Γt+1 ← Γt+1 ∪ { (X (t+1) <sup>z</sup> , Y (t+1) <sup>z</sup> )} 17: end for 18: ▷ Assign(S,M, z) = {<sup>s</sup> <sup>∈</sup> <sup>S</sup> | arg maxz′ <sup>M</sup>sz′ <sup>=</sup> <sup>z</sup>} 19: end if 20: end for 21: t ← t + 1 22: end while 23: Output: Γ<sup>κ</sup> = {(x<sup>i</sup> , T(xi))} ▷ Set of refined pairs.

## B. Proofs

Datasets X and Y are represented as discretely supported probability measures µ = P<sup>n</sup> <sup>i</sup>=1 aiδ<sup>x</sup><sup>i</sup> and ν = P<sup>n</sup> <sup>j</sup>=1 b<sup>j</sup> δ<sup>y</sup><sup>j</sup> for probability vectors a, b ∈ ∆n, which we assume to be uniform: a = b = u<sup>n</sup> = (1/n)1<sup>n</sup> ∈ ∆n. We form the cost matrix C defined by

$$\mathbf{C}_{ij} := c(\mathbf{x}_i, \mathbf{y}_j). \quad (\text{S1})$$

In all cases below, we are concerned with the assignment problem [\(1\)](#page-1-0) for this cost matrix.

Let perm(n) = {P˜ ∈ <sup>R</sup> <sup>n</sup>×<sup>n</sup> : P1˜ <sup>n</sup> = P˜ <sup>⊤</sup>1<sup>n</sup> = (1/n) 1n} denote the set of (scaled) n × n permutation matrices. By the Birkhoff-von Neumann theorem [\(Birkhoff,](#page-9-15) [1946\)](#page-9-15), an optimal solution to the n × n assignment problem is attained at a permutation matrix in perm(n).

Definition B.1. Say that cost matrix C ∈ R <sup>n</sup>×<sup>n</sup> is *Monge rotated* if the identity matrix I is a solution to the assignment problem associated to C, i.e.

$$\mathbf{I} \in \arg \min_{\mathbf{P} \in \text{perm}(n)} \langle \mathbf{C}, \mathbf{P} \rangle.$$

For arbitrary cost matrix C ∈ R <sup>n</sup>×<sup>n</sup>, let <sup>P</sup>† ∈ arg min <sup>P</sup>∈perm(n) ⟨C, <sup>P</sup>⟩<sup>F</sup> , and note that the column-permuted cost matrix C† := CP†,<sup>⊤</sup> is Monge rotated by construction. This is a consequence of the following identity, which holds for any permutation P˜ ∈ perm(n).

$$\begin{aligned}\langle \mathbf{C}, \mathbf{P} \rangle_F &= \text{tr}(\mathbf{C}^\top \mathbf{P}) \\ &= \text{tr}(\hat{\mathbf{P}}^{-1} \hat{\mathbf{P}} \mathbf{C}^\top \mathbf{P}) \\ &= \text{tr}(\hat{\mathbf{P}} \mathbf{C}^\top \mathbf{P} \hat{\mathbf{P}}^\top) = \langle \mathbf{C} \hat{\mathbf{P}}^\top, \mathbf{P} \hat{\mathbf{P}}^\top \rangle_F.\end{aligned}\tag{S2}$$

Let Π(un,ur) ≡ Π<sup>u</sup>n,u<sup>2</sup> denote the transport polytope between two uniform measures. For Q ∈ Π(un,ur), say that a row of Q is *soft* if at least two of its entries are positive, and call the row *hard* otherwise. For rank r ≪ n such that r divides n, let Π•(un,ur) be the subset of Π(un,ur) consisting of transport plans Q with only hard rows. Below, we consider two low-rank OT problems associated to C† . The first low-rank problem considered is

$$\min_{\mathbf{Q}, \mathbf{R} \in \Pi_\bullet(\mathbf{u}_n, \mathbf{u}_r)} \langle \mathbf{C}^\dagger, \mathbf{Q} \mathbf{R}^\top \rangle_F. \quad (\text{S3})$$

while the second low-rank problem considered is restricted to symmetric couplings:

$$\min_{\mathbf{Q} \in \Pi_\bullet(\mathbf{u}_n, \mathbf{u}_r)} \langle \mathbf{C}^\dagger, \mathbf{Q} \mathbf{Q}^\top \rangle_F. \quad (\text{S4})$$

In either case, we have omitted the constant factor of r coming from diag(1/ur). We next introduce a technical condition on C. Let C ∈ R <sup>n</sup>×<sup>n</sup> be a cost matrix and let <sup>P</sup>† ∈ arg minP∈perm(n) ⟨C, P⟩ corresponding to permutation σ † : [n] → [n], σ† ∈ Sn. Given partitions I = {I1, . . . , Ir} and J = {J1, . . . , Jr} of [n] and a, b ∈ [r], define the cost between two sets Ia, J<sup>b</sup> to be

$$\mathbf{C}_{I_a, J_b} := \sum_{i \in I_a, j \in J_b} \mathbf{C}_{i\sigma^\dagger(j)}. \quad (\text{S5})$$

We call partition I *balanced* if each block I<sup>a</sup> of I has the same number of elements, | I<sup>a</sup> | = (n/r).

Definition B.2. Cost matrix C ∈ R <sup>n</sup>×<sup>n</sup> is r*-Monge separable* if there exists a balanced partition I <sup>⋆</sup> = {I ⋆ k } r <sup>k</sup>=1, such that for any two permutations π1, π<sup>2</sup> ∈ Sn, one has

$$\sum_{k=1}^r \mathbf{C}_{I_k^*, I_k^*} \leq \sum_{k=1}^r \mathbf{C}_{\pi_1(I_k^*), \pi_2(I_k^*)}. \quad (\text{S6})$$

We say that C is *strictly* r*-Monge separable* if [\(S6\)](#page-14-1) holds with strict inequality (<) for any π1(I ⋆ k ) ̸= π2(I ⋆ k ).

One interesting feature of this definition is that while the sum is over r ≤ n terms, where it may occur that r ≪ n, this inequality must hold over all permutations π<sup>1</sup> and π<sup>2</sup> acting on the individual data points, rather than partition blocks. This captures the notion of finding low-rank or low-resolution solutions which are nevertheless compatible with the optimal bijective Monge map.

*Remark* B.3*.* If C is r-Monge separable, the distinguished partition I <sup>⋆</sup> may be represented as Q<sup>⋆</sup> ∈ Π•(un,ur) such that Q<sup>⋆</sup> is optimal for [\(S4\)](#page-14-2) and the pair (Q<sup>⋆</sup> , Q<sup>⋆</sup> ) is optimal for [\(S3\)](#page-14-3). After proving the next lemma, we will relate r-Monge separability to cyclic monotonicity.

Proposition B.4. *Let* C ∈ R <sup>n</sup>×<sup>n</sup> *be strictly* r*-Monge separable. If* Q<sup>⋆</sup> , <sup>R</sup><sup>⋆</sup> ∈ arg min<sup>Q</sup>,R∈Π•(un,u2) ⟨C, QR<sup>⊤</sup>⟩ *then, for all* i ∈ [n]*,*

$$\arg \max_{z \in [r]} \mathbf{Q}_{iz}^* = \arg \max_{z \in [r]} \mathbf{R}_{\sigma^\dagger(i)z}^*, \quad (S7)$$

*where* σ † : [n] → [n] *is the permutation corresponding to* <sup>P</sup>† ∈ arg minP∈perm(n) ⟨C, P⟩<sup>F</sup> *.*

*Proof.* Let σ † , P† be as in the statement of the lemma, and define C† := CP†,⊤. The same reasoning as in [\(S2\)](#page-14-4) implies that if (Q<sup>⋆</sup> , R<sup>⋆</sup> ) ∈ arg min<sup>Q</sup>,R∈Π•(un,u2) ⟨C, QR<sup>⊤</sup>⟩<sup>F</sup> , then

$$(\mathbf{Q}^*, \mathbf{P}^\dagger \mathbf{R}^*) \in \underset{\mathbf{Q}, \mathbf{R} \in \Pi_\bullet(\mathbf{u}_n, \mathbf{u}_2)}{\operatorname{argmin}} \langle \mathbf{C}^\dagger, \mathbf{Q} \mathbf{R}^\top \rangle_F. \quad (\text{S8})$$

The membership [\(S8\)](#page-14-5) follows from the identities

$$\begin{aligned}\langle \mathbf{C}^\dagger, \mathbf{Q}^* \mathbf{R}^* \mathbf{P}^{\dagger, \top} \rangle_F &= \langle \mathbf{C} \mathbf{P}^{\dagger, \top}, \mathbf{Q}^* \mathbf{R}^{*, \top} \mathbf{P}^{\dagger, \top} \rangle_F, \\ &= \text{tr}(\mathbf{P}^\dagger \mathbf{C}^\top \mathbf{Q}^* \mathbf{R}^{*, \top} \mathbf{P}^{\dagger, \top}), \\ &= \text{tr} \mathbf{C}^\top \mathbf{Q}^* \mathbf{R}^{*, \top} = \langle \mathbf{C}, \mathbf{Q}^* \mathbf{R}^{*, \top} \rangle_F.\end{aligned}$$

Remark [B.3](#page-14-6) above follows from the requirement that the variables Q, R have all hard rows, and are subject to uniform marginal constraints, so that all non-zero entries of QR<sup>⊤</sup> have the same value. Thus, if C is r-Monge separable, there exists Q˜ ∈ Π•(un,u2) corresponding to distinguished balanced partition I˜ from Definition [B.2](#page-14-0) such that

$$(\tilde{\mathbf{Q}}, \tilde{\mathbf{Q}}) \in \underset{\mathbf{Q}, \mathbf{R} \in \Pi_\bullet(u_n, u_2)}{\operatorname{arg\,min}} \langle \mathbf{C}^\dagger, \mathbf{Q} \mathbf{R}^\top \rangle. \quad (\text{S9})$$

Moreover, this pair (Q˜ , Q˜ ) is the unique optimum when C is strictly r-Monge separable. From [\(S8\)](#page-14-5), [\(S9\)](#page-15-1), we must have

$$\tilde{\mathbf{Q}} = \mathbf{Q}^*, \quad \tilde{\mathbf{Q}} = \mathbf{P}^\dagger \mathbf{R},$$

from which [\(S7\)](#page-14-7) follows immediately.

Let us now discuss how the notion of r-Monge separability is related to c-cyclic monotonicity. Recall that for a cost matrix C ∈ R <sup>n</sup>×<sup>n</sup> derived from ground cost c the support of an optimal plan is c-cyclically monotone if for all permutations π : [n] → [n], π ∈ Sn, one has

$$\sum_{i=1}^n \mathbf{C}_{ii} \leq \sum_{i=1}^n \mathbf{C}_{i\pi(i)}. \quad (\text{S10})$$

As it amounts to a reindexing of the sum on the right side of [\(S10\)](#page-15-2) , one can equivalently define the support of the optimal plan to be c-cyclically monotone if for any *pair* of permutations π1, π<sup>2</sup> ∈ Sn,

$$\sum_{i=1}^n \mathbf{C}_{ii} \leq \sum_{i=1}^n \mathbf{C}_{\pi_1(i)\pi_2(i)},$$

from which we see that c-cyclical monotonicity is equivalent to r-Monge separability with r = n.

We next show that the optimal factors Q<sup>⋆</sup> , R<sup>⋆</sup> for the rank-2 Wasserstein problem given in [\(5\)](#page-2-2) correspond to hard-partitions of each dataset, so that for this problem the optimal Q<sup>⋆</sup> , R<sup>⋆</sup> ∈ Π(un,u2) satisfy Q<sup>⋆</sup> , R<sup>⋆</sup> ∈ Π•(un,ur). Below, let supp<sup>i</sup> (Q<sup>⋆</sup> ) ⊂ [n] be the indices on which column i of Q<sup>⋆</sup> is supported, where i = 1, 2.

Lemma B.5. *Let* (Q<sup>⋆</sup> , R<sup>⋆</sup> ) *be optimal for the rank-2 Wasserstein problem* [\(5\)](#page-2-2) *subject to the additional constraint that* a = b = un, *and* g = u<sup>2</sup> *are uniform and* n = m *is even. Then,* (supp<sup>1</sup> (Q<sup>⋆</sup> ),supp<sup>2</sup> (Q<sup>⋆</sup> )) *is a partition of* [n]*, and symmetrically, so is* (supp<sup>1</sup> (R<sup>⋆</sup> ),supp<sup>2</sup> (R<sup>⋆</sup> ))*, so* (Q<sup>⋆</sup> , R<sup>⋆</sup> ) ∈ Π•(un,u2)

*Proof.* The cost is linear in (Q, R) respectively: the minimization in each variable given the other fixed can be expressed as

$$\text{arg min}_{\mathbf{Q} \in \Pi(\mathbf{u}_n, \mathbf{u}_2)} 2 \langle \mathbf{Q}, \mathbf{CR} \rangle_F, \quad \text{arg min}_{\mathbf{R} \in \Pi(\mathbf{u}_n, \mathbf{u}_2)} 2 \langle \mathbf{R}, \mathbf{C}^T \mathbf{Q} \rangle_F. \quad (\text{S11})$$

Thus for any optimal Q<sup>⋆</sup> or R<sup>⋆</sup> fixed the minimization in the other variable is a linear optimal transport problem, where by Corollary 2.11 in [\(De Loera & Kim,](#page-10-21) [2013\)](#page-10-21) it holds that since the constraint matrix is totally unimodular with marginals integral (on rescaling), the optima R<sup>⋆</sup> and Q<sup>⋆</sup> must be vertices on the transport polytope Π<sup>u</sup>n,u<sup>2</sup> with integral entries (on rescaling, by 2n or 2m). There are ≤ n + 1 positive entries in any optimal rank r = 2 solution [\(De Loera & Kim,](#page-10-21) [2013;](#page-10-21) [Peyre & Cuturi](#page-11-16) ´ , [2019\)](#page-11-16), so that n (resp. m) being even and the rescaled rows and columns summing to 2 and n implies that there are exactly n positive entries and thus that the vertices define partitions of [n] and [m]. Thus, solutions to [S11](#page-15-3) satisfy (Q<sup>⋆</sup> , R<sup>⋆</sup> ) ∈ Π•(un,u2) .

Notably, in the case of an odd number of points n or m this likewise implies that one has a single row which has 2 entries 1/2n 1/2n , with all other rows of the form 0 1/n or 1/n 0 defining a partition of the remaining even subset of size (n − 1) or (m − 1). In the general case of ranks r ̸= 2 there are maximally n + r + 1 [\(Peyre & Cuturi](#page-11-16) ´ , [2019\)](#page-11-16) non-zero edges (so that the graph is acyclic), and for n ≫ r the optimal solution remains close to a partition given mild assumptions on C.

Lemma [B.5](#page-15-0) states optimal low-rank couplings (Q<sup>⋆</sup> , R<sup>⋆</sup> ) for Problem [7](#page-3-1) over Π(un,u2) are in Π•(un,u2) . Thus, by Proposition [B.4](#page-14-8) these solutions co-cluster points x ∈ X with their image under Monge map T ⋆ (x), supposing the cost is strictly 2-Monge separable (Definition [B.2\)](#page-14-0). This co-clustering is in the sense of the clustering functions q ⋆ ,r ⋆ from Proposition [3.1](#page-3-0) corresponding to each factor Q<sup>⋆</sup> , R<sup>⋆</sup> . We note that when µ and ν are discretely supported measures with supports of equal cardinality, a Monge map, T ⋆ : X → Y, is guaranteed to exist by Theorem 2.7 of [\(Thorpe,](#page-12-13) [2018\)](#page-12-13).

On the Rank Schedule. At each intermediate scale t ∈ [κ], the *rank-schedule* (r1, . . . , rκ) determines the effective rank of the coupling computed so far. For each <sup>t</sup> ∈ [κ], define the *effective rank* at scale <sup>t</sup> as ρ<sup>t</sup> := Q<sup>t</sup> <sup>s</sup>=1 rs. This effective rank corresponds to the number of partitions, which are placed in bijective correspondence

$$X_q^{(t)} \leftrightarrow Y_q^{(t)} \quad t \in [\rho_t]. \quad (\text{S12})$$

at the t-th step of HiRef. The size of the partitions at scale t is given by n/ρ<sup>t</sup> = |X (t) | = |Y (t) |. Given these preliminaries, we show that for an appropriate rank-schedule Hierarchical Refinement yields optimal transport maps.

Proposition B.6 (Optimality of Hierarchical Refinement). *Suppose the Monge-map exists between two datasets* X*,* Y *of size* n*. Then there exists a rank-schedule* (r1, · · · , rκ) *which factorizes* n *such that all size* n/ρ<sup>t</sup> *partitions generated by Hierarchical Refinement at level* t *satisfy strict* rt+1*-Monge separability (Definition [B.2\)](#page-14-0) for* t ∈ [0 : κ − 1]*. For any such rank-schedule, given an optimal black-box low-rank solver over* Π•(·, ·)*, Hierarchical Refinement returns the Monge-map.*

*Proof.* For existence, observe that taking <sup>r</sup><sup>1</sup> <sup>=</sup> <sup>n</sup> implies the statement P<sup>n</sup> <sup>k</sup>=1 C<sup>I</sup> ⋆ ,I<sup>⋆</sup> k ≤ P<sup>n</sup> <sup>k</sup>=1 C<sup>π</sup>1(<sup>I</sup> ⋆ k ),π2(I ⋆ k ) . For partitions <sup>I</sup><sup>k</sup> of size one, this is equivalent to the statement of <sup>c</sup>-cyclical monotonicity P<sup>n</sup> <sup>i</sup>=1 Cii ≤ P<sup>n</sup> <sup>i</sup>=1 Ciπ(i) , so that for the trivial rank-schedule (r1) := (n) the cost is always n-Monge separable.

Given the existence of such a schedule (r1, · · · , rκ) with rt+1-Monge separability, we proceed by induction on t ∈ [0, κ]. For the base case of t = 0, as we assume the Monge map exists, for the initial partition Γ<sup>0</sup> = {(X, Y)} one has that Y = T(X). We want to show the variant that Γ<sup>t</sup> contains sets which are co-clusters of sets with their image under T. As the inductive hypothesis, at scale t > 0 with ρ<sup>t</sup> co-clusters Γ<sup>t</sup> = {(X (t) i , Y (t) i )} ρ<sup>t</sup> <sup>i</sup>=1 each satisfies Y (t) <sup>i</sup> = T(X (t) i ). As strict rt+1-Monge separability holds for each size n/ρ<sup>t</sup> bipartition (X (t) i , Y (t) i ) ∈ Γt, using Proposition [B.4](#page-14-8) each such set is divided into rt+1 co-clusters {(X (t+1) j , Y (t+1) j )} rt+1 <sup>j</sup>=1 which satisfy Y (t+1) <sup>j</sup> = T(X (t+1) j ). Thus, taking the union of these rt+1 bi-partitions across the ρ<sup>t</sup> elements of Γ<sup>t</sup> we form a set Γt+1 of size ρt+1 = rt+1ρ<sup>t</sup> which maintains the invariant that (X (t+1) j , Y (t+1) j ) ∈ Γt+1 =⇒ Y (t+1) <sup>j</sup> = T(X (t+1) j ). At the final level r<sup>κ</sup> Monge separability holds, so one may conclude on singleton sets of the form Γ<sup>κ</sup> = {(x<sup>i</sup> , T(xi))} n <sup>i</sup>=1.

*Remark* B.7*.* Strict Monge separability applies unconditionally at the terminal level. Observe that all sets in Γκ−<sup>1</sup> have size equal to the rank (n/ρκ−1) = rκ, and that we have maintained the invariant that Y (κ−1) <sup>j</sup> = T(X (κ−1) j ). Let J<sup>κ</sup> ⊂ [n] denote the size r<sup>κ</sup> set of indices for X (κ−1) j in X. By c-cyclical monotonicity, one has for all permutations π ∈ perm(n)

$$\sum_{i=1}^n \mathbf{C}_{ii} = \sum_{i \in J_K} \mathbf{C}_{ii} + \sum_{j \in [n \setminus J_K]} \mathbf{C}_{jj} \leq \sum_{i \in J_K} \mathbf{C}_{i\pi(i)} + \sum_{j \in [n \setminus J_K]} \mathbf{C}_{j\pi(j)} = \sum_{i=1}^n \mathbf{C}_{i\pi(i)}$$

Thus, for the subset of permutations on n where π : π |[n]\J<sup>κ</sup> = id, we have P <sup>i</sup>∈J<sup>κ</sup> Cii ≤ P <sup>i</sup>∈J<sup>κ</sup> Ciπ(i) implying that one may solve a constant time O(r 2 κ ) solution to the assignment problem on each size r<sup>κ</sup> bipartition to recover the final map.

We call ρ<sup>t</sup> the effective rank because (to avoid quadratic space complexity) we never instantiate the transport coupling corresponding to the bijective mapping [\(S12\)](#page-16-1) as a matrix T(t) . Were we to instantiate T(t) , it would have rank ρt, and moreover we can evaluate its transport cost by using T(t) to induce a transport coupling P(t) between the full datasets X, Y.

$$\mathbf{P}_{ij}^{(t)} := \begin{cases} \rho_t/n^2 & \text{if } q(n/\rho_t) < i, j \leq (q+1)(n/\rho_t) \\ 0 & \text{otherwise} \end{cases} , \quad (\text{S13})$$

where q ∈ [ρt], and where the mass ρt/n<sup>2</sup> is a simplified form of (ρt/n) 2 (1/ρt). We note that this is a rewriting of ρ<sup>t</sup> n<sup>2</sup> P<sup>ρ</sup><sup>t</sup> <sup>q</sup>=1 δ(xi,y<sup>j</sup> )∈Γt,q to have the indices ordered into a contiguous block-structure. Using coupling [\(S13\)](#page-16-0), which again we never instantiate, one can define:

$$\text{cost}(\mathbf{T}^{(t)}) := \langle \mathbf{C}, \mathbf{P}^{(t)} \rangle.$$

The next proposition shows that the costs ⟨C, P(t) ⟩ decrease as t increases from 1 to κ, and also provides a bound on their consecutive differences. Below, recall that each Γ<sup>t</sup> denotes the co-clustering (X (t) , Y (t) ), where

$$\mathbf{X}^{(t)} = \{\mathbf{X}_q^{(t)}\}_{q=1}^{\rho_t}, \quad \mathbf{Y}^{(t)} = \{\mathbf{Y}_q^{(t)}\}_{q=1}^{\rho_t},$$

and where co-cluster Γt,q is defined as:

$$\Gamma_{t,q} := \{(\mathbf{x}, \mathbf{y}) : \mathbf{x} \in X_q^{(t)}, \mathbf{y} \in Y_q^{(t)}\} .$$

Proposition B.8 (Proposition [3.4\)](#page-4-3). *Let cost function* c : R <sup>d</sup> ×<sup>R</sup> <sup>d</sup> → <sup>R</sup><sup>+</sup> *be of the form* c(x, y) = h(x−y) *for some strictly convex function* h : R <sup>d</sup> → <sup>R</sup><sup>+</sup> *and suppose that* h *is Lipschitz. Let* P(t) *be as defined above in* [\(S13\)](#page-16-0)*. Then one has the following bound on the difference in cost between iterations of refinement:*

$$0 \leq \langle \mathbf{C}, \mathbf{P}^{(t)} \rangle - \langle \mathbf{C}, \mathbf{P}^{(t+1)} \rangle \leq \|\nabla c\|_{\infty} \frac{1}{\rho_t} \sum_{q=1}^{\rho_t} \text{diam}(\Gamma_{t,q}) , \quad (\text{S14})$$

*where*

$$\text{diam}(\Gamma_{t,q}) \equiv \text{diam}(X_q^{(t)} \cup T(X_q^{(t)})) := \max_{\mathbf{x}_i, \mathbf{x}_j, \mathbf{x}_k, \mathbf{x}_l \in X_q^{(t)}} \left\| (\mathbf{x}_i, T(\mathbf{x}_j)) - (\mathbf{x}_k, T(\mathbf{x}_l)) \right\|.$$

*Proof.* By definition [\(S13\)](#page-16-0) of P(t) ,

$$\begin{aligned} \langle \mathbf{C}, \mathbf{P}^{(t)} \rangle - \langle \mathbf{C}, \mathbf{P}^{(t+1)} \rangle &= \frac{\rho_t}{n^2} \sum_{i=1}^n \sum_{j=1}^n c(\mathbf{x}_i, \mathbf{y}_j) \sum_{q=1}^{\rho_t} \delta_{(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t,q}} - \frac{\rho_{t+1}}{n^2} \sum_{i=1}^n \sum_{j=1}^n c(\mathbf{x}_i, \mathbf{y}_j) \sum_{q=1}^{\rho_{t+1}} \delta_{(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t+1,q}} \\ &= \frac{\rho_t}{n^2} \left( \sum_{i=1}^n \sum_{j=1}^n c(\mathbf{x}_i, \mathbf{y}_j) \sum_{q=1}^{\rho_t} \delta_{(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t,q}} - r_{t+1} \sum_{i=1}^n \sum_{j=1}^n c(\mathbf{x}_i, \mathbf{y}_j) \sum_{q'=1}^{\rho_{t+1}} \delta_{(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t+1,q'}} \right) \\ &= \frac{\rho_t}{n^2} \left( \sum_{q=1}^{\rho_t} \sum_{i=1}^n \sum_{j=1}^n c(\mathbf{x}_i, \mathbf{y}_j) \delta_{(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t,q}} - r_{t+1} \sum_{q'=1}^{\rho_{t+1}} \sum_{i=1}^n \sum_{j=1}^n c(\mathbf{x}_i, \mathbf{y}_j) \delta_{(\mathbf{x}_i, \mathbf{y}_j) \in \Gamma_{t+1,q'}} \right). \end{aligned}$$

By Proposition [B.4,](#page-14-8) one then has:

$$= \frac{\rho_{t+1}}{n^2} \left( \sum_{q=1}^{\rho_t} \left( \underbrace{\frac{1}{r_{t+1}} \sum_{i \in \mathbf{X}_q^{(t)}} \sum_{j \in \mathbf{X}_q^{(t)}} c(\mathbf{x}_i, T(\mathbf{x}_j))}_{\text{average "Monge distortion" in } \Gamma_{t,q} \text{ over next scale}} - \sum_{z=1}^{r_{t+1}} \sum_{i \in \mathbf{X}_{q\rho_{t+z}}^{(t+1)}} \sum_{j \in \mathbf{X}_{q\rho_{t+z}}^{(t+1)}} c(\mathbf{x}_i, T(\mathbf{x}_j)) \right) \right) \quad (\text{S15})$$

Note that the inner summands of [\(S15\)](#page-17-1) (indexed by q) are non-negative by definition of the refinement step, where *within* each cluster, one has a minimization over a larger set of couplings. This shows ⟨C, P(t) ⟩ − ⟨C, P(t+1)⟩ ≥ 0. Towards an upper bound, we will bound each summand of [\(S15\)](#page-17-1):

$$\left( \frac{1}{r_{t+1}} \sum_{i \in X_q^{(t)}} \sum_{j \in X_q^{(t)}} c(\mathbf{x}_i, T(\mathbf{x}_j)) - \sum_{z=1}^{r_{t+1}} \sum_{i \in X_{q_{t+z}}^{(t+1)}} \sum_{j \in X_{q_{t+z}}^{(t+1)}} c(\mathbf{x}_i, T(\mathbf{x}_j)) \right). \quad (\text{S16})$$

Define st+1 := n/ρt+1 as well as barycenters

$$\bar{\mathbf{x}}^{(t)} := \sum_{\mathbf{x}_i \in X_{q_{\rho_t+z}}^{(t+1)}} \frac{\mathbf{x}_i}{s_{t+1}}, \quad \bar{\mathbf{y}}^{(t)} := \sum_{\mathbf{x} \in X_{q_{\rho_t+z}}^{(t+1)}} \frac{T(\mathbf{x}_i)}{s_{t+1}},$$

and note that by Jensen's inequality, for convex cost c(·, ·) one has:

$$\begin{aligned} \sum_{z=1}^{r_{t+1}} \sum_{\mathbf{x}_i \in \mathbf{X}_{q\rho_t+z}^{(t+1)}} \sum_{\mathbf{x}_j \in \mathbf{X}_{q\rho_t+z}^{(t+1)}} c(\mathbf{x}_i, T(\mathbf{x}_j)) &= s_{t+1}^2 \sum_{z=1}^{r_{t+1}} \sum_{\mathbf{x}_i \in \mathbf{X}_{q\rho_t+z}^{(t+1)}} \frac{1}{s_{t+1}} \sum_{j \in \mathbf{X}_{q\rho_t+z}^{(t+1)}} \frac{1}{s_{t+1}} c(\mathbf{x}_i, T(\mathbf{x}_j)) \\ &\geq s_{t+1}^2 r_{t+1} c(\bar{\mathbf{x}}^{(t)}, \bar{\mathbf{y}}^{(t)}), \end{aligned}$$

so that we may continue upper-bounding the difference [\(S16\)](#page-17-2):

$$\leq \frac{1}{r_{t+1}} \left( \sum_{\mathbf{x}_i \in \mathbf{X}_q^{(t)}} \sum_{\mathbf{x}_j \in \mathbf{X}_q^{(t)}} c(\mathbf{x}_i, T(\mathbf{x}_j)) \right) - s_{t+1}^2 r_{t+1} c(\bar{\mathbf{x}}^{(t)}, \bar{\mathbf{y}}^{(t)}) \quad (\text{S17})$$

$$= \frac{1}{r_{t+1}} \left( \left( \sum_{\mathbf{x}_i \in \mathbf{X}_q^{(t)}} \sum_{\mathbf{x}_j \in \mathbf{X}_q^{(t)}} c(\mathbf{x}_i, T(\mathbf{x}_j)) \right) - \frac{n^2}{\rho_t} c(\bar{\mathbf{x}}^{(t)}, \bar{\mathbf{y}}^{(t)}) \right) \quad (\text{S18})$$

$$= \frac{1}{r_{t+1}} \left( \sum_{\mathbf{x}_i \in \mathbf{X}_q^{(t)}} \sum_{\mathbf{x}_j \in \mathbf{X}_q^{(t)}} (c(\mathbf{x}_i, T(\mathbf{x}_j)) - c(\bar{\mathbf{x}}^{(t)}, \bar{\mathbf{y}}^{(t)})) \right). \quad (\text{S19})$$

Now, define the diameter of co-cluster Γt,q as follows:

$$\text{diam}(\Gamma_{t,q}) \equiv \text{diam}(X_q^{(t)} \cup T(X_q^{(t)})) := \max_{\mathbf{x}_i, \mathbf{x}_j, \mathbf{x}_k, \mathbf{x}_l \in X_q^{(t)}} \left\| (\mathbf{x}_i, T(\mathbf{x}_j)) - (\mathbf{x}_k, T(\mathbf{x}_l)) \right\|,$$

Using our Lipschitz assumption on h made at the beginning of the section, where c(x, y) = h(x−y) (we will write ∥∇c∥<sup>∞</sup> for ∥∇h∥∞), one has the inequality:

$$|c(\mathbf{x}_i, T(\mathbf{x}_i)) - c(\mathbf{x}_j, T(\mathbf{x}_j))| \leq \|\nabla c\|_\infty \text{diam}(\Gamma_{t,q}) .$$

Thus, returning to the bound on each summand [\(S16\)](#page-17-2), we obtain the upper bound:

$$\leq \frac{1}{r_{t+1}} \sum_{\mathbf{x}_i \in \mathbf{X}_q^{(t)}} \sum_{\mathbf{x}_j \in \mathbf{X}_q^{(t)}} \|\nabla c\|_{\infty} \left\| (\mathbf{x}_i, T(\mathbf{x}_j)) - (\bar{\mathbf{x}}^{(t)}, \bar{\mathbf{y}}^{(t)}) \right\| \quad (\text{S20})$$

As partition X (t+1) is a refinement of X (t) and Y (t+1) is a refinement of Y (t) , it holds that [\(S16\)](#page-17-2) is upper bounded by:

$$\leq \frac{1}{r_{t+1}} \sum_{i \in \mathbf{X}_q^{(t)}} \sum_{j \in \mathbf{X}_q^{(t)}} \|\nabla c\|_{\infty} \text{diam}(\Gamma_{t,q}), \quad (\text{S21})$$

$$= \frac{1}{r_{t+1}} |\mathbf{X}_q^{(t)}|^2 \|\nabla c\|_{\infty} \text{diam}(\Gamma_{t,q}) , \quad (\text{S22})$$

$$= \frac{1}{r_{t+1}} \frac{n^2 \|\nabla c\|_\infty}{\rho_t^2} \text{diam}(\Gamma_{t,q}). \quad (\text{S23})$$

To conclude, we plug these bounds into each summand of [\(S15\)](#page-17-1), obtaining the following bound on the full sum:

$$= \frac{\rho_{t+1}}{n^2} \frac{1}{r_{t+1}} \frac{n^2 \|\nabla c\|_\infty}{\rho_t^2} \sum_{q=1}^{\rho_t} \text{diam}(\Gamma_{t,q}) \quad (\text{S24})$$

$$= \|\nabla c\|_\infty \frac{1}{\rho_t} \sum_{q=1}^{\rho_t} \text{diam}(\Gamma_{t,q}). \quad (\text{S25})$$

*Remark* B.9*.* Proposition [B.8](#page-17-0) should be considered a *conditional* result. Our proof follows that of (Proposition 1, [\(Gerber &](#page-10-11) [Maggioni,](#page-10-11) [2017\)](#page-10-11)), but they are able to provide sharper bounds between elements of a cluster and the centroid of the cluster using the properties assumed to hold in their definition of a multiscale family of partitions (Definition [C.3\)](#page-20-0), which mimick the structure of dyadic cubes in Euclidean space. As we do not make any geometric assumptions of our partitions, the above result is a priori weaker, through we leave the exploration of the geometry of partitions induced by low-rank OT to future work.

*Remark* B.10*.* Note, if c(x<sup>i</sup> , T(x<sup>j</sup> )) = γ is constant (i.e., if all points are equidistant in a block), one has that refinement offers no gain from level Γ<sup>t</sup> → Γt+1:

$$\leq \frac{\rho_{t+1}}{n^2} \sum_{q=1}^{\rho_t} \left| \gamma \frac{|\mathbf{X}_q^{(t)}|^2}{r_{t+1}} - \gamma r_{t+1} |\mathbf{X}_q^{(t+1)}|^2 \right| = \frac{\rho_{t+1}}{n^2} \sum_{q=1}^{\rho_t} \left| \gamma \frac{(n/\rho_t)^2}{r_{t+1}} - \gamma r_{t+1} (n/\rho_{t+1})^2 \right| = 0.$$

*Remark* B.11*.* The work [\(Seguy et al.,](#page-12-18) [2018\)](#page-12-18) suggests a loss dependent on an (entropic) coupling γ. If γ is sparse and supported on the graph of the Monge map so that γ = (id × T) <sup>♯</sup> µ, this loss becomes a regression of a neural network T<sup>θ</sup> on the Monge map T over the support of µ: min<sup>T</sup><sup>θ</sup> <sup>E</sup>µc (Tθ(xi), T(xi)).

*Proof.* By linearity of the push-forward map one immediately obtains

$$\begin{aligned} & \int_{\mathbb{X} \times \mathbb{Y}} \|T_\theta(x) - y\|_2^p (\mathrm{id} \times T)_\# \sum_{i=1}^n \mu_i \delta_{x_i} \mathrm{d}x \mathrm{d}y = \int_{\mathbb{X} \times \mathbb{Y}} \|T_\theta(x) - y\|_2^p \sum_{i=1}^n \mu_i (\mathrm{id} \times T)_\# \delta_{x_i} \mathrm{d}x \mathrm{d}y \\ &= \sum_{i=1}^n \mu_i \int_{\mathbb{X} \times \mathbb{Y}} \|T_\theta(x) - y\|_2^p \delta_{(x_i, T(x_i))} \mathrm{d}y \mathrm{d}x = \sum_{i=1}^n \mu_i \|T_\theta(x_i) - T(x_i)\|_2^p, \end{aligned}$$

By integrating against the δ. As µ<sup>i</sup> > 0, it holds that this loss is identically zero if and only if T<sup>θ</sup> = T on the dataset (xi) n i=1

$$\min_{T_\theta} \int_{X \times Y} \|T_\theta(x) - y\|_2^p d\gamma(x, y) = 0 \iff \|T_\theta(x_i) - T(x_i)\|_2^p = 0 \iff T_\theta(x_i) = T(x_i)$$

In other words, when one minimizes the objective of [\(Seguy et al.,](#page-12-18) [2018\)](#page-12-18) using the bijective Monge map γ = (id × T)♯µ as opposed to an entropic coupling, the objective of [\(Seguy et al.,](#page-12-18) [2018\)](#page-12-18) reduces to an unbiased regression. That is, the neural map T<sup>θ</sup> directly matches T over the dataset support as if trained on supervised (x, y) pairs y = T(x).

## C. Background: Multiscale Optimal Transport

## C.1. Multiscale Partitions

[\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) describe a general multiscale strategy for computing OT couplings between metric measure spaces (X, dX, µ) and (Y, dY, ν). They state this in the Kantorovich setting, using a general cost function c : X × Y → <sup>R</sup>+. Their framework consists of several elements:

- 1. A way of *coarsening* the set of source points X and the measure µ across multiple scales:

$$(X, \mu) =: (X_J, \mu_J) \rightarrow (X_{J-1}, \mu_{J-1}) \rightarrow \cdots \rightarrow (X_1, \mu_1), \quad (\text{S26})$$

as well as an analogous coarsening for the set of target points Y:

$$(Y, \nu) =: (Y_J, \nu_J) \rightarrow (Y_{J-1}, \nu_{J-1}) \rightarrow \cdots \rightarrow (Y_1, \nu_1), \quad (S27)$$

where at each scale j, supp(µ<sup>j</sup> ) = X<sup>j</sup> and supp(ν<sup>j</sup> ) = Y<sup>j</sup> , and the cardinality of each X<sup>j</sup> and Y<sup>j</sup> decreases with j.

- 2. A way of *propagating* coupling π<sup>j</sup> solving the transport problem µ<sup>j</sup> → ν<sup>j</sup> at scale j to a coupling πj+1 at scale j + 1.
- 3. A way of *refining the coupling* from scale j to an optimal solution at scale j + 1.

To derive approximation bounds for the error incurred by the multiscale transport problem at each scale, [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) use regular families of multiscale partitions (Definition [C.3](#page-20-0) below) to define approximations to µ, ν and c at all scales.

For z ∈ X, define Bx(r) := {x ′ ∈ X : dX(x, x′ ) < r} as the metric ball of radius r centered at x. Functions f, g : X → <sup>R</sup> have the *same order of magnitude* if there is c1, c<sup>2</sup> > 0 with c1f(x) ≤ g(x) ≤ c2f(x) for all x ∈ X, and in this case we write f ≍ g. Write M(X) for the space of unsigned measures on X, and write P(X) for the subspace of probability measures.

Definition C.1. A metric space (X, dX) has *doubling dimension* d > 0 if every Bz(r) admits a covering by at most 2 <sup>d</sup> balls of radius r/2.

A metric space is said to be *doubling* if it has doubling dimension d for some d > 0. A related notion to a doubling metric space is a doubling measure.

Definition C.2. Measure µ ∈ M(X) is a *doubling measure with dimension* d if there is a constant c<sup>1</sup> > 0 such that for all x ∈ X and all r > 0, one has c −1 1 r <sup>d</sup> ≤ µ(Bx(r)) ≤ c1r d , i.e. µ(Bx(r)) ≍ r d .

Note that if (X, dX, µ) is doubling, then d<sup>X</sup> is doubling, and up to modification of d<sup>X</sup> to an equivalent metric, the dimension d can be taken as the same in either case.

Definition C.3. Given metric measure space (X, dX, µ), a *regular family of multiscale partitions* with scaling parameter θ > 1 is a family of sets

$$\left\{ \{C_{j,k}\}_{k=1}^{K_j} \right\}_{j=1}^J,$$

with each Cj,k ⊂ X such that:

- 1. For each scale j, the sets {Cj,k} K<sup>j</sup> <sup>k</sup>=1 partition X.
- 2. For each scale j ∈ [J − 1], either Cj+1,k′ ∩ Cj,k = ∅ or Cj+1,k′ ⊂ Cj,k. In this latter case, we say that (j + 1, k′ ) is a *child* of (j, k), or equivalently that (j, k) is a *parent* of (j + 1, k′ ), writing (j + 1, k′ ) ≺ (j, k).
- 3. There is a constant A > 0 such that for all j, k, we have diameter diam(Cj,k) ≤ Aθ−<sup>j</sup> .
- 4. Each Cj,k contains a "center point" cj,k such that B<sup>c</sup><sup>j</sup> ,k(θ −j ) ⊂ Cj,k.

We take θ = 2 for simplicity. As the child-parent terminology suggests, these partitions (through the second point) have a tree structure, like dyadic cubes in R d . Though the measure µ is not explicitly used in the above definition, the third and fourth points imply µ(Cj,k) ≍ 2 <sup>−</sup>jd and K<sup>j</sup> ≍ 2 jd .

Coarsening spaces and measures Now suppose that each of X and Y are each discrete metric measure spaces, each equipped with regular families Γ(X), Γ(Y) of multiscale partitions:

$$\begin{aligned}\Gamma(\mathbf{X}) &:= \{\Gamma_j(\mathbf{X})\}_{j=0}^J, & \Gamma_j(\mathbf{X}) &:= \{C_{j,k}(\mathbf{X})\}_{k=1}^{K_j} \\ \Gamma(\mathbf{Y}) &:= \{\Gamma_j(\mathbf{Y})\}_{j=0}^J, & \Gamma_j(\mathbf{Y}) &:= \{C_{j,k}(\mathbf{Y})\}_{k=1}^{K_j},\end{aligned}$$

and these yield the coarsening chains in [\(S26\)](#page-19-1), [\(S27\)](#page-19-2) in the most natural way possible at each scale j, defining the coarse-grained spaces X<sup>j</sup> , Y<sup>j</sup> to be the clusters at scale j:

$$X_j := \Gamma_j(X), \quad Y_j := \Gamma_j(Y),$$

while the measures at scale j are defined from the measures at scale j + 1 via:

$$\mu_j(C_{j,k}(\mathbf{X})) := \sum_{(j+1,k') \prec (j,k)} \mu_{j+1}(C_{j+1,k'}(\mathbf{X})), \quad \nu_j(C_{j,k}(\mathbf{Y})) := \sum_{(j+1,k') \prec (j,k)} \nu_{j+1}(C_{j+1,k'}(\mathbf{Y})).$$

The fourth item of Definition [C.3](#page-20-0) requires that we define cluster centers c¯j,k(X) for each Cj,k(X). At the finest scale j = J, all clusters CJ,k(X) correspond to singletons {xJ,k}, so we define c¯J,k(X) := xJ,k in this case. At coarser scales, these centers can be defined recursively from the next finest scale, depending on the structure of X.

For example, if X has vector space structure (in addition to being a metric measure space), a natural choice for cluster centers xj,k at scale j = 0, . . . , J − 1 is the weighted average xj,k := ¯cj,k(X), where

$$\bar{c}_{j,k}(\mathbf{X}) := \sum_{(j+1,k') \prec (j,k)} \mu_{j+1}(C_{j+1,k'}(\mathbf{X})) x_{j+1,k'}.$$

On the other hand, in the absence of vector space structure, one can still define

$$\bar{c}_{j,k}(\mathbf{X}) = \arg \min_{\mathbf{x} \in \mathbf{X}} \sum_{(j+1,k') \prec (j,k)} d_{\mathbf{X}}^p(x, x_{j+1,k'}),$$

with analogous constructions for Y yielding centers yj,k.

Coarsening the cost function [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) suggest three ways to coarsen the cost function using the multiscale partition. To condense the notation slightly, let us write xj,k in place of Cj,k(X) and yj,k′ in place of Cj,k(X) and Cj,k′ (Y).

(c-i) The pointwise value

$$c_j(x_{j,k}, y_{j,k'}) := c(x_{j,k}, y_{j,k'}), \quad (\text{S28})$$

using centers xj,k and yj,k′ defined in any of the ways above.

(c-ii) The local average

$$c_j(x_{j,k}, y_{j,k'}) := \frac{\sum_{x \in C_{j,k}(\mathbf{X}), y \in C_{j,k'}(\mathbf{Y})} c(x, y)}{|C_{j,k}(\mathbf{X})| |C_{j,k'}(\mathbf{Y})|}$$

(c-iii) The local weighted average:

$$c_j(x_{j,k}, y_{j,k'}) := \frac{\sum_{x \in C_{j,k}(X), y \in C_{j,k'}(Y)} c(x, y) \pi_{j-1}^*(x_{j-1,k_1}, y_{j-1,k'_1})}{\sum_{x \in C_{j,k}(X), y \in C_{j,k'}(Y)} \pi_{j-1}^*(x_{j-1,k_1}, y_{j-1,k'_1})},$$

where π ⋆ j−1 is the optimal (or approximately optimal) OT coupling at scale j − 1, defined below. The indices k<sup>1</sup> and k ′ are defined using the tree structure of the partition: k<sup>1</sup> is the unique index among [Kj−1(X)] such that (j, k) ≺ (j − 1, k1), and likewise k ′ is unique among [Kj−1(X)].

### C.2. Propagation of OT solutions across scales

For each scale j, consider the OT problem given as follows.

$$\begin{aligned} \pi_j^* &:= \arg \min_{\pi \in \Pi(\mu_j, \nu_j)} \text{cost}(\pi_j), \quad \text{where:} \\ \text{cost}(\pi_j) &:= \sum_{k \in [K_j(\mathbf{X})], k' \in [K_j(\mathbf{Y})]} c(x_{j,k}, y_{j,k'}) \pi_j(x_{j,k}, y_{j,k'}) \end{aligned} \quad (\text{S29})$$

[\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) show bounds on |cost(π ⋆ j ) − cost(π ⋆ J )| of a constant times 2 <sup>−</sup><sup>j</sup>∥∇c∥∞, but note that this only implies closeness of the couplings in terms of their cost, not necessarily in any other sense.

Given an optimal coupling π ⋆ j at scale j, [\(Glimm & Henscheid,](#page-10-15) [2013\)](#page-10-15) proposed a direct propagation strategy to initialize the problem at scale j + 1, distributing the mass π ⋆ j (xj,k, yj,k′ ) equally to all combinations of paths between children(xj,k) and children(yj,k′ ). In this context, a path is understood to mean a source-target pair at the next scale, e.g. a pair of the form (xj+1,ℓ, yj+1,ℓ′ ). To formalize this, let

$$\mathcal{A}_j := \{(x_{j,k}, y_{j,k'}) : k \in [K_j(\mathbf{X})], k' \in [K_j(\mathbf{Y})]\}$$

denote *all* paths between points in X<sup>j</sup> and Y<sup>j</sup> . The drawback of this warm-start procedure is that if supp(µ<sup>j</sup> ) ⊂ A<sup>j</sup> , which is always the case, the refinement procedure still requires quadratic space complexity at the finest scale.

To mitigate the ultimate quadratic space complexity of retaining all possible paths at all scales, [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) allow for a refinement procedure where the support of couplings at scale j + 1 is restricted to a subset Rj+1 ⊂ Aj+1 of all possible paths (with Rj+1 defined by the optimal coupling at the previous iteration). Given R<sup>j</sup> ⊂ A<sup>j</sup> , let π ⋆ j |<sup>R</sup><sup>j</sup> denote the optimal solution to the path-restricted or *restricted problem* at scale j:

$$\pi_j^*|_{\mathcal{R}_j} := \arg \min_{\pi \in \Pi(\mu_j, \nu_j), \text{ supp}(\pi) \subset \mathcal{R}_j} \text{cost}(\pi_j). \quad (\text{S30})$$

Simple propagation. The simplest way to restrict the number of paths considered at subsequent scales is to use paths at scale j whose endpoints are children of mass-bearing paths at scale j + 1:

$$\mathcal{R}_{j+1} := \{(x_{j+1,\ell}, y_{j+1,\ell}) : \exists (x_{j,k}, y_{j,k'}) \in \text{supp}(\pi_j^*) \text{ s.t. } (j+1, \ell) \prec (j,k) \text{ and } (j+1, \ell') \prec (j,k')\}.$$

The optimal Kantorovich plan at scale j has at most (K<sup>j</sup> (X) + K<sup>j</sup> (Y) + 1) non-zero entries. Using the above simple propagation strategy constrains plan at scale j + 1 to be supported on at most

$$\alpha_j^2(K_j(\mathbf{X}) + K_j(\mathbf{Y}))$$

entries, where α<sup>j</sup> is the maximum number of children of any (j, k) across both datasets. When the ambient space has doubling dimension d, for any j one has α<sup>j</sup> ≍ 2 d , yielding a plan with *linear* space complexity at the finest scale.

Capacity constraint propagation. This propagation strategy solves a modified minimum flow problem at scale j in order to include additional paths at scale j + 1 likely to be included in the optimal solution π ⋆ <sup>j</sup>+1. Concretely, one first computes an unconstrained optimal plan π ⋆ j |<sup>R</sup><sup>j</sup> at scale j. Then, a new OT plan π˜ ⋆ j |<sup>R</sup><sup>j</sup> is solved for at scale j now subject to the capacity constraint

$$\tilde{\pi}_j^*|_{\mathcal{R}_j} \leq U_{k,k'} \min(\mu(x_{j,k}), \nu(y_{j,k'}))$$

for each (xj,k, yj,k′ ) ∈ supp(π ⋆ j |<sup>R</sup><sup>j</sup> ), where the random variables Uk,k′ are i.i.d. Uniform([0.1, 0.9]). This can also be iterated several times, in all cases leading to linear space complexity in the optimization at the finest scale.

## C.3. Refinement of the propagated solution

Propagation of a solution to the restricted transport problem [\(S30\)](#page-22-0) at scale j, in general cannot guarantee reaching an optimal solution to the restricted problem at scale j + 1, and can lead to accumulation of errors across all scales. Several *refinement* strategies are proposed in [\(Gerber & Maggioni,](#page-10-11) [2017\)](#page-10-11) to address this.

Potential Refinement. One refinement strategy leverages the problem dual to [\(3\)](#page-2-0), here stated at the finest scale:

$$\max_{\substack{\mathbf{f} \in \mathbb{R}^n, \mathbf{g} \in \mathbb{R}^m \\ \mathbf{f}_i + \mathbf{g}_j \leq \mathbf{C}_{ij}}} \sum_{i=1}^n \mu(\{x_i\}) \mathbf{f}_i + \sum_{j=1}^m \mu(\{y_j\}) \mathbf{g}_j. \quad (\text{S31})$$

The refinement strategy uses optimal dual variables f ⋆ , g ⋆ to select paths to include at the next scale. From the dual formulation, an optimal solution (f ⋆ , g ⋆ ) to [\(S31\)](#page-22-1) must have all nonnegative entries in the *reduced cost matrix*, defined as the matrix C − f ⊕ g with entries Ckk′ − f<sup>k</sup> − gk′ . Note that the dual to the restricted problem [\(S30\)](#page-22-0) is well-defined, and for this dual we denote the optimal dual potentials by f ⋆ |<sup>R</sup><sup>j</sup> and g ⋆ |<sup>R</sup><sup>j</sup> . With slight abuse of notation, let (f <sup>⋆</sup> ⊕ g ⋆ )|<sup>R</sup><sup>j</sup> be

$$(\mathbf{f}^* \oplus \mathbf{g}^*)|_{\mathcal{R}_j} := (\mathbf{f}^*|_{\mathcal{R}_j} \oplus \mathbf{g}^*|_{\mathcal{R}_j}) \odot \mathbf{M}^{(j)},$$

where M(j) is the indicator matrix of the restricted set of paths R<sup>j</sup> at scale j, and where ⊙ denotes the Hadamard (entrywise) product. While the restricted set of paths R<sup>j</sup> is inherited from the previous scale, one can define a new set of paths V 0 j based on where the restricted reduced cost C − (f <sup>⋆</sup> ⊕ g ⋆ )|<sup>R</sup><sup>j</sup> is nonpositive:

$$\mathcal{V}_j^0(\pi_j^*|_{\mathcal{R}_j}) := \{(x_{j,k}, y_{j,k'}) \in \mathcal{A}_j : \mathbf{C}_{kk'} - [(\mathbf{f}^* \oplus \mathbf{g}^*)|_{\mathcal{R}_j}]_{kk'} \leq 0\}.$$

Table S1. Hyperparameters for Synthetic Experiments

| Parameter Name            | Variable        | Value    |
|---------------------------|-----------------|----------|
| Rank-Annealing Schedule   | ( r 1 , , r κ ) | [2, 512] |
| Hierarchy Depth           | κ               | 2        |
| Maximal Base Rank         | Q               | 2        |
| Maximal Intermediate Rank | C               | 16       |

With a new set of paths Q<sup>0</sup> j := V 0 j (π ⋆ j |<sup>R</sup><sup>j</sup> ), one can compute a new optimal plan π ⋆ j |Q<sup>0</sup> j at scale j restricted to these paths, as well as *new* optimal dual potentials f ⋆ |V<sup>0</sup> j and g ⋆ |V<sup>0</sup> leading to a new reduced cost C − (f <sup>⋆</sup> ⊕ g ⋆ )|V<sup>0</sup> j . This strategy can be iterated via

$$\mathcal{Q}_j^i := \mathcal{V}_j(\pi_j^*|_{\mathcal{Q}_j^{i-1}}), \quad (\text{S32})$$

yielding the sequence of transport plans π ⋆ j |Q<sup>i</sup> , all at scale j, which converge on a solution whose reduced cost is nonnegative, necessarily making it optimal. The potential refinement strategy was used by [\(Glimm & Henscheid,](#page-10-15) [2013\)](#page-10-15), with [\(Schmitzer,](#page-12-16) [2016\)](#page-12-16) introducing shielding neighborhoods in a similar spirit, using dual potentials to locally verify global optimality.

# D. Experimental Details

## D.1. Synthetic Experiments

For all of the synthetic experiments, we first generate n = 1024 points from three datasets: the checkerboard dataset ([\(Makkuva et al.,](#page-11-9) [2020\)](#page-11-9)), the MAFMoons and Rings dataset ([\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5)), and the Half-moon and S-curve dataset ([\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5)). Following [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) the random seed was set to 0 for data-generation with jax.random.key(0). We evaluate the OT cost ⟨C, P⟩<sup>F</sup> of HiRef Sinkhorn [\(Cuturi,](#page-9-0) [2013\)](#page-9-0), and ProgOT [\(Kassraie](#page-11-19) [et al.,](#page-11-19) [2024\)](#page-11-19) on each of these three datasets, where we use (1) the Euclidean cost ∥·∥2, and (2) the squared Euclidean cost ∥·∥<sup>2</sup> 2 (Table [S2\)](#page-8-1). We additionally quantify the number of non-zero entries in the plan and its entropy (Table [S3\)](#page-24-0).

We also compare the cost of couplings computed by Hierarchical Refinement to low-rank couplings [\(Scetbon et al.,](#page-12-9) [2021;](#page-12-9) [Halmos et al.,](#page-10-8) [2024\)](#page-10-8) of varying rank. We observe that as the latent rank r → n, the OT cost ⟨C, Pr⟩<sup>F</sup> asymptotically approaches the cost achieved by Hierarchical Refinement (Figure [S3\)](#page-6-0). In the limit limr→n⟨C, Pr⟩<sup>F</sup> low-rank OT recovers Sinkhorn [\(Scetbon & Cuturi,](#page-12-11) [2022\)](#page-12-11) and approaches quadratic memory complexity, while HiRef remains linear in space.

### Checkerboard

The checkerboard dataset [\(Makkuva et al.,](#page-11-9) [2020\)](#page-11-9) is defined by random variables Y ∼ Q sampled from the source distribution according to Y = X + Z where X and Z are sampled from Uniform distributions defined by

$$\mathbf{X} \sim \text{Uniform}(\{(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)\})$$
,  
 $\mathbf{Z} \sim \text{Uniform}([-0.5, 0.5] \times [-0.5, 0.5])$ .

the target distribution P has random variable Y′ where the random variable Y′ is defined as Y′ = X′ + Z with components

$$\mathbf{X}' \sim \text{Uniform}(\{(0,1), (0,-1), (1,0), (-1,0)\})$$
,  
 $\mathbf{Z} \sim \text{Uniform}([-0.5, 0.5] \times [-0.5, 0.5])$ .

## MAFMoons and Rings

The MAFMoon dataset [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) defines a source distribution Q by sampling X ∼ N (0, <sup>1</sup>2) and applying the non-linear transformation defined by

$$\mathbf{Y} = \begin{bmatrix} Y_1 \\ Y_2 \end{bmatrix} = \begin{bmatrix} 0.5(X_1 + X_2^2) - 5 \\ X_2 \end{bmatrix}$$

The target distribution P representing concentric rings is generated by first sampling θ ∼ Uniform(2π), with fixed radii r<sup>i</sup> ∈ {0.25, 0.55, 0.9, 1.2} from which one transforms to Cartesian coordinates as x<sup>i</sup> = 3r<sup>i</sup> cos θ<sup>i</sup> and y<sup>i</sup> = 3r<sup>i</sup> sin θ<sup>i</sup> . Gaussian noise is added to each of these, as ϵ ∼ N (0, <sup>1</sup>σ 2 ) for σ = 0.08.

## Half-moon and S-Curve

The Half-moon and S-curve dataset [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5) is generated from Y = make moons and make S curve from the scikit-learn library. Both datasets are transformed further with a rotation R(θ), a scaling λ, and a translation µ applied as Y′ ← R(θ)(λY) + µ.

Table S2. Comparison Table for Coupling-Based OT Methods on Primal Cost ⟨C, P⟩<sup>F</sup> for ∥·∥<sup>2</sup> and ∥·∥<sup>2</sup>

| Method   | Table S2. Checkerboard | Comparison Table (Makkuva | for Coupling-Based 2020) MAFMoons | OT Methods on & Rings (Buzun | Primal Cost ⟨ C , P ⟩ F 2024) Half Moon | for ∥·∥ 2 and ∥·∥ 2 & S-Curve (Buzun 2024) |
|----------|------------------------|---------------------------|-----------------------------------|------------------------------|-----------------------------------------|--------------------------------------------|
|          | ∥·∥ 2                  | ∥·∥ 2                     |                                   |                              |                                         |                                            |
|          |                        | 2                         | ∥·∥ 2                             | ∥·∥ 2                        |                                         |                                            |
|          |                        |                           |                                   | 2                            | ∥·∥ 2                                   | ∥·∥ 2                                      |
| Sinkhorn | 0.3573                 | 0.1319                    | 0.4422                            | 0.4440                       | 0.5663                                  | 0.5663                                     |
| ProgOT   | N/A                    | 0.1320                    | N/A                               | 0.4443                       | N/A                                     | 0.5709                                     |
| HiRef    | 0.3533                 | 0.1248                    | 0.4398                            | 0.4414                       | 0.5741                                  | 0.5737                                     |

Table S3. Entropy and Non-Zero Entries ( > 10−<sup>8</sup> ) of Coupling Matrices for Each Method and Dataset (Wasserstein-2 distance cost, ∥·∥<sup>2</sup> 2)

| Method   | Checkerboard Entropy | (Makkuva Non-Zeros | 2020) MAFMoons Entropy | & Rings (Buzun Non-Zeros | 2024) Half Moon Entropy | & S-Curve (Buzun 2024) Non-Zeros |
|----------|----------------------|--------------------|------------------------|--------------------------|-------------------------|----------------------------------|
| Sinkhorn | 12.8509              | 624733             | 12.6117                | 678720                   | 12.7776                 | 652993                           |
| ProgOT   | 12.3830              | 271087             | 11.6158                | 327764                   | 12.1170                 | 337258                           |
| HiRef    | 6.9314               | 1024               | 6.9314                 | 1024                     | 6.9314                  | 1024                             |

Table S4. Comparison of Coupling-Based OT Methods on Primal Cost ⟨C, P⟩<sup>F</sup> (Wasserstein-2) on 512 point small instance.

| Method MOP (Gerber & Maggioni, 2017) Sinkhorn ( ott-jax ) | Checkerboard 0.393 0.136 | MAF Moons & Rings 0.276 0.221 | Half Moon & S-Curve 0.401 0.338 |
|-----------------------------------------------------------|--------------------------|-------------------------------|---------------------------------|
| ProgOT                                                    | 0.136                    | 0.216                         | 0.334                           |
| HiRef                                                     | 0.129                    | 0.216                         | 0.334                           |
| Dual Revised Simplex Solver                               | 0.127                    | 0.214                         | 0.332                           |

Table S5. Hyperparameters for Mouse-Embryo Spatial Transcriptomics Experiment (E15.5-16.5)

| Parameter Name            | Variable        | Value        |
|---------------------------|-----------------|--------------|
| Rank-Annealing Schedule   | ( r 1 , , r κ ) | [2, 86, 659] |
| Hierarchy Depth           | κ               | 3            |
| Maximal Base Rank         | Q               | 2            |
| Maximal Intermediate Rank | C               | 128          |

### D.2. Large-scale Transcriptomics Matching on Mouse-Embryo

In this problem, we use HiRef to find a full-rank alignment matrix between successive pairs of spatial transcriptomics (ST) [\(Stahl et al.](#page-12-19) ˚ , [2016\)](#page-12-19) slices. These are from a dataset of whole-mouse embryogenesis [\(Chen et al.,](#page-9-13) [2022\)](#page-9-13) on the Stereo-Seq platform. These datasets have been measured at successive 1-day time-intervals: E9.5 (n = 5913), E10.5 (n = 18408), E11.5 (n = 30124), E12.5 (n = 51365), E13.5 (n = 77369), E14.5 (n = 102519), E15.5 (n = 113350), and E16.5 (n = 121767), where the embryonic mouse is growing across the stages so that the sample-complexity n increases with the numeric stage. For each pair of datasets of size n and m, we sub-sample the datasets so that the size of the two datasets is given as n ← min{n, m}.

In the context of spatial transcriptomics, an experiment conducted on a two-dimensional tissue slice produces a data pair (X, Z). Here, X ∈ <sup>R</sup> n×p represents the gene expression matrix, where n denotes the number of cells (or spatial spots) analyzed on the slice, and p signifies the number of genes measured. Specifically, the entry Xij ∈ <sup>R</sup><sup>+</sup> corresponds to the expression level of gene j in cell i, with higher values indicating greater expression intensity. Concurrently, Z ∈ R n×2 is the spatial coordinate matrix, where each row i contains the (x, y) coordinates of cell i on the tissue slice. Consequently, every cell on the slice is characterized by a gene expression vector of length p, capturing its molecular features, and a coordinate vector of length two, detailing its spatial position within the slice.

We utilize the extensive, real-world dataset on mouse embryo development presented in [\(Chen et al.,](#page-9-13) [2022\)](#page-9-13), which encompasses eight temporal snapshots of spatial transcriptomics (ST) slices throughout the entire mouse embryo development process. And align all consecutive timepoints. The preprocessing of this dataset is conducted using the standard SCANPY [\(Wolf et al.,](#page-12-20) [2018\)](#page-12-20) workflow. Initially, we ensure that both slices contain an identical set of genes by filtering, which results in a common gene set across all cells for each pair of timepoints. Subsequently, we apply log-normalization to the gene expression data of all cells from the two slices. To compress the data, we perform Principal Component Analysis (PCA), reducing the dimensionality of the gene expression profiles to d = 60 PCs. Finally, we compute the Euclidean distances between gene expression vectors in the PCA-transformed space to construct the cost matrix C, on which we solve a Wasserstein problem to obtain the optimal coupling P of full-rank. We offer hyperparameters for the E15-16.5 experiment (the largest alignment) in Table [S5.](#page-25-1) For the other experiments, the maximal intermediate rank is r = 16 up to E10.5, r = 32 to E11.5, r = 64 up to E13.5, and 128 for E14.5-16.5. The rank-annealing schedule is generated according to the dynamic program in each case by the rank annealing.optimal rank schedule( n, hierarchy depth , max Q , max rank ) function.

In this experiment, we benchmark against the default implementation of Sinkhorn in ott-jax [\(Cuturi et al.,](#page-9-11) [2022\)](#page-9-11) with entropy parameter ϵ = 0.05, and additionally benchmark against the default implementations of ProgOT [\(Kassraie et al.,](#page-11-19) [2024\)](#page-11-19) and LOT [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) in ott-jax. For the low-rank methods LOT and FRLC [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) we fix a constant rank of r = 40 for these experiments. While LOT [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) provides a robust, scalable low-rank procedure for the Wasserstein-2 distance, the LOT solver with point cloud input on Wasserstein-1 cost only runs for the first pair (E9.5:E10.5). For subsequent pairs we input the cost C directly, resulting in the LOT solver running up to the third pair (E11.5:E12.5). Mini-batch OT is run with batch-sizes ranging from 128 to 2048, and is performed without replacement. As noted in prior works [\(Fatras et al.,](#page-10-5) [2020;](#page-10-5) [2021a;](#page-10-4)[b\)](#page-10-6), this is a standard choice for instantiating a full-rank coupling using mini-batch OT. Sinkhorn is used to solve each mini-batch coupling, as implemented in ott-jax with a default setting of the entropy parameter ϵ = 0.05.

### D.3. Brain Atlas Spatial Alignment

We took inspiration from MERFISH-MERFISH alignment experiments of [\(Clifton et al.,](#page-9-14) [2023\)](#page-9-14), particularly gene abundance transfer tasks that STalign is exhibited on. The data are available on the Vizgen website for MERFISH Mouse Brain Receptor Map data release [\(https://info.vizgen.com/mouse-brain-map\)](https://info.vizgen.com/mouse-brain-map). The two spatial transcriptomics slices used for the experiment

Table S6. Cost Values ⟨P, C⟩<sup>F</sup> for Different Methods Across Embryonic Stages

| Method   | E9.5-E10.5 | E10.5-E11.5 | E11.5-E12.5 | E12.5-E13.5 | E13.5-E14.5 | E14.5-E15.5 | E15.5-E16.5 |
|----------|------------|-------------|-------------|-------------|-------------|-------------|-------------|
| HiRef    | 21.81      | 14.81       | 16.14       | 14.35       | 13.78       | 14.29       | 12.79       |
| Sinkhorn | 21.91      | 14.89       |             |             |             |             |             |
| ProgOT   | 22.56      | 15.35       |             |             |             |             |             |
| MB 128   | 22.44      | 15.35       | 16.69       | 14.86       | 14.14       | 14.75       | 13.32       |
| MB 512   | 22.15      | 15.05       | 16.33       | 14.54       | 13.92       | 14.50       | 13.01       |
| MB 1024  | 22.05      | 15.02       | 16.24       | 14.45       | 13.86       | 14.43       | 12.91       |
| MB 2048  | 21.98      | 14.98       | 16.18       | 14.39       | 13.81       | 14.39       | 12.85       |
| FRLC     | 23.14      | 16.09       | 17.74       | 15.47       | 14.64       | 15.51       | 14.00       |
| LOT      | 26.06      | 19.06       | 21.64       |             |             |             |             |

are slice 2, replicate 3 ("source" dataset) and slice 2, replicate 2 ("target" dataset). The datasets will be denoted (X<sup>1</sup> , S 1 ) for the source and (X<sup>2</sup> , S ) for the target.

The source dataset consists of 85, 958 spots, while the target dataset consists of 84, 172 spots. To apply HiRef to these data, we subsampled the source dataset to have 84, 172 spots also (uniformly at random), removing a total of 1786 spots. We use this sub-sampled n × n dataset for all methods, but as discussed below, note that this sub-sampling incurs little error. We ran HiRef using the settings max rank = 11 and hierarchy depth=4, for a total runtime of 10 minutes 6 seconds, on an A100 GPU. The random seed was set to 44. For the cost function used by HiRef, we only use the *spatial* modalities S 1 , S <sup>2</sup> of the two datasets. We centered the spatial coordinates of both datasets, and applied a rotation by 45 degrees to the first dataset. With these registered spatial data, here denoted S˜<sup>1</sup> = {s 1 i } n <sup>i</sup>=1 and <sup>S</sup>˜<sup>2</sup> <sup>=</sup> {<sup>s</sup> 2 i } n <sup>i</sup>=1, we formed the cost matrix C given by:

$$\mathbf{C}_{ij} = \|\mathbf{s}_i^1 - \mathbf{s}_j^2\|_2,$$

where ∥ · ∥<sup>2</sup> denotes the Euclidean distance between the spatial coordinates. This cost C was used as input to HiRef, which produced as output a 1-1 mapping T between the two datasets (a permutation matrix is too large to instantiate).

We then evaluated the performance of HiRef through cosine similarity of predicted gene abundance with target gene abundance, across five "spatially-patterned genes" (using the terminology of [\(Clifton et al.,](#page-9-14) [2023\)](#page-9-14)): *Slc17a7*, *Grm4*, *Olig1*, *Gad1*, *Peg10*. Writing g to stand in for any of these genes, we formed the abundance vectors v 1,g and v <sup>2</sup>,<sup>g</sup> using the raw counts for gene g in each datasets' expression component X<sup>1</sup> , X<sup>2</sup> . Using HiRef output T, we also formed the *predicted* abundance vector vˆ g , which maps the raw counts from v 1,g to the spots in the second dataset through T.

Moreover, to compute cosine similarities between predicted and true expression abundances, [\(Clifton et al.,](#page-9-14) [2023\)](#page-9-14) employ a spatial binning on their output, using windows of 200µm to tile each slice. The diameter of each slice is roughly 10, 000µm, and to make our output comparable, we used the spatial coordinates S ′ to bin and average the vectors v 2,g and vˆ locally. We used a total of 5625 bins, corresponding to a 15-to-1 mapping from spots to bins. Averaging the abundance of gene g in each bin, we obtain spatially smoothed versions of v 2,g and vˆ, as in [\(Clifton et al.,](#page-9-14) [2023\)](#page-9-14). Denote these smoothed vectors by w<sup>2</sup>,<sup>g</sup> and wˆ . For each gene g among { *Slc17a7*, *Grm4*, *Olig1*, *Gad1*, *Peg10* }, we computed the cosine similarity between w<sup>2</sup>,<sup>g</sup> and wˆ , listing our results in Table [S7.](#page-27-0) In the same table, we list scores obtained by the low-rank methods FRLC [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) and LOT [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9) for comparison. While HiRef is restricted to running on datasets of the same size, LOT and FRLC have no such restriction, and can run on the pair of MERFISH slices without any subsampling. To address this, in each case of LOT and FRLC, we give results from the methods run on the datasets with *and* without subsampling, reporting the highest scores for each method in main. In particular, we compared the cosine similarities for the original and sub-sampled dataset on a downstream task, as the primal OT cost is no longer directly comparable. Without the sub-sampling, the cosine score is only slightly higher than with: (0.3390, 0.2712, 0.3186, 0.1666, 0.1080) vs (0.3241, 0.2279, 0.3029, 0.1653, 0.0719). These scores remain significantly lower than those of hierarchical refinement on the sub-sampled data: (0.8098, 0.7959, 0.7526, 0.4932, 0.6015).

For the FRLC algorithm, we set α = 0, γ = 200, τin = 500, rank r = 500, using 20 outer iterations and 300 inner iterations. The runtime of FRLC was 1 minute 26 seconds on an A100 GPU. For the LOT algorithm, we were unable to pass a low-rank factorization of the distance matrix, so we had to use a smaller rank r = 20 in order to avoid exceeding GPU memory (the choice r = 20 led to memory usage of 30GB). We set ϵ = 0.01 and otherwise used the default settings of the method. The

Table S7. Cosine Similarity Scores for Expression Transfer & Spatial Transport Cost

| Method       |                       | Slc17a7      | Grm4   | Olig1  | Gad1   | Peg10  | Transport Cost |
|--------------|-----------------------|--------------|--------|--------|--------|--------|----------------|
| HiRef        | (this work)           | 0.8098       | 0.7959 | 0.7526 | 0.4932 | 0.6015 | 330.3301       |
| FRLC         | (Halmos et al., 2024) | 0.2180       | 0.2124 | 0.1929 | 0.0963 | 0.0991 | 415.0683       |
| FRLC,        | no subsampling        | 0.2373       | 0.1896 | 0.1579 | 0.0644 | 0.1550 | 634.4158       |
| LOT (Scetbon | et al., 2021)         | 0.3241       | 0.2279 | 0.3029 | 0.1653 | 0.0719 | 3722.3171      |
| LOT, no      | subsampling           | 0.3390       | 0.2712 | 0.3186 | 0.1666 | 0.1080 | 3722.1360      |
| MOP          | (Gerber & Maggioni,   | 2017) 0.5211 | 0.4714 | 0.5972 | 0.3571 | 0.2719 | 2479.6117      |
| Mini-batch   | (128)                 | 0.6693       | 0.6637 | 0.6442 | 0.4150 | 0.4932 | 653.0491       |
| Mini-batch   | (512)                 | 0.7089       | 0.7383 | 0.6771 | 0.4562 | 0.5383 | 438.1703       |
| Mini-batch   | (1,024)               | 0.7256       | 0.7621 | 0.6918 | 0.4733 | 0.5557 | 384.2498       |
| Mini-batch   | (2,048)               | 0.7434       | 0.7822 | 0.7056 | 0.4912 | 0.5683 | 349.2964       |

Table S8. Cost Values ⟨C, P⟩<sup>F</sup> for ImageNet [\(Deng et al.,](#page-10-19) [2009;](#page-10-19) [Russakovsky et al.,](#page-11-21) [2015\)](#page-11-21) Alignment Task.

| <b>Method</b>  | HiRef        | Sinkhorn | MB 128 | MB 256 | MB 512 | MB 1024 | FLRC  | LOT | ProgTd |
|----------------|--------------|----------|--------|--------|--------|---------|-------|-----|--------|
| <b>OT Cost</b> | <b>18.97</b> | N/A      | 21.89  | 21.11  | 20.34  | 19.58   | 24.12 | N/A | N/A    |

total runtime was 36 minutes 8 seconds on an A100 GPU. To form a spot-to-spot mapping from each transport plan output by FRLC and LOT, we mapped the spot with index i in the first slice to the index argmax of the i-th row of the transport plan. Note that we ran LOT using the squared Euclidean cost as default, as passing cost fn=costs.Euclidean() as an argument to ott-jax's PointCloud raised an error. The discrepancy in transport cost between the two low rank methods reported in Table [S7](#page-27-0) is explained by (i) needing to use squared-Euclidean cost in the case of LOT, and (ii) using a rank-20 plan of LOT versus the rank-500 plan of FRLC. We applied the exact same spatial averaging to the outputs of all methods. We plot the ground-truth and HiRef-predicted abundances in Figure [S1.](#page-3-2)

### D.4. Alignment of ImageNet Embeddings

To demonstrate the scalability of HiRef to massive and high-dimensional datasets, we perform an alignment unprecedented for OT solvers: aligning 1.281 million images from the ImageNet ILSVRC dataset [\(Russakovsky et al.,](#page-11-21) [2015;](#page-11-21) [Deng et al.,](#page-10-19) [2009\)](#page-10-19). A negligible amount of sub-sampling, 167 points out of 1281167, was applied so that n divided into two integers n/2 = 640500 of which neither is prime. From this, rank annealing.optimal rank schedule( n, hierarchy depth , max Q , max rank ) was called to generate the depth 3 rank-annealing schedule of (r1, r2, r3) = (7, 50, 1830) for HiRef. We used the ResNet50 architecture [\(He et al.,](#page-10-20) [2016\)](#page-10-20) available at <https://download.pytorch.org/models/resnet50-0676ba61.pth> to generate embeddings of each image of dimension d = 2048. We then took a 50:50 split of the dataset as the two image datasets X, Y to be aligned, where we used a random permutation of the indices of the dataset using torch.randperm so that the splits approximately represent the same distribution over images. We then aligned these image datasets using HiRef FRLC, and mini-batch OT. For z<sup>i</sup> , z<sup>j</sup> ∈ <sup>R</sup> <sup>2048</sup> we used the standard Euclidean cost defined by

$$\mathbf{C}_{ij} = \|\mathbf{z}_i - \mathbf{z}_j\|_2$$

We use the sample-linear algorithm [\(Indyk et al.,](#page-10-16) [2019\)](#page-10-16) to factorize C into low-rank factors of dimensions (d1, d2, d3) = (r1, r2, r3) = (7, 50, 1830) paralleling the rank-schedule. The final cost values for each are shown in Table [S8.](#page-27-1)

# E. Additional Information

There are a number of additional practical details regarding Algorithm [1](#page-4-0) in its actual implementation. In particular, to achieve linear scaling, one must also have sample-linear approximation of the distance matrix C. We use the algorithm of [\(Indyk et al.,](#page-10-16) [2019\)](#page-10-16) to accomplish this, as discussed in Section [E.1.](#page-28-0) In addition, one requires parallel sequence of ranks for the distance matrices used at each step, (d1, · · · , dκ). As a default, we set (d1, · · · , dκ) = (r1, · · · , rκ) so that the ranks of the distance matrices parallel those of the coupling matrices. Moreover, HiRef has the capacity to be heavily parallelized: since Algorithm [1](#page-4-0) breaks each instance into independent partitions, one may also parallelize the low-rank sub-problems of

Table S9. Hyperparameters for ImageNet Experiment

| Parameter Name            | Variable        | Value         |
|---------------------------|-----------------|---------------|
| Rank-Annealing Schedule   | ( r 1 , , r κ ) | [7, 50, 1830] |
| Hierarchy Depth           | κ               | 3             |
| Maximal Base Rank         | Q               | 2             |
| Maximal Intermediate Rank | C               | 64            |

Algorithm [1](#page-4-0) across compute nodes.

## E.1. Optimizing the Rank-Annealing Schedule

As discussed in Section [3.3,](#page-4-4) the large constants required by low-rank OT (LROT) in practice encourage factorizations which have *minimal* partial sums. In particular, one seeks a factorization which minimizes the number of times LROT is run as a sub-procedure. Suppose one defines the maximal admissible rank of the low-rank solutions to be C ∈ <sup>Z</sup>+, the hierarchy-depth to be κ, the number of data-points to be n, and the maximal-rank permissible for the base-case alignment to be Q. If Q ̸= 1, then one may take n ← n/Q, κ ← κ − 1, to observe that the total number of runs required is 1 + r<sup>1</sup> + r1r<sup>2</sup> + ... + Q<sup>κ</sup> <sup>i</sup>=1 r<sup>i</sup> , where the ranks factor the sample-size as Q<sup>κ</sup> <sup>i</sup>=1 r<sup>i</sup> = n. Thus, to optimize the number of LROT calls for a given hierarchy-depth κ, one can optimize for the rank-annealing schedule by minimizing the sum of partial products defined by min(ri) κ P<sup>κ</sup> j=1 Q<sup>j</sup> <sup>i</sup>=1 <sup>r</sup><sup>i</sup> subject to Q<sup>κ</sup> <sup>i</sup>=1 r<sup>i</sup> = n, r<sup>i</sup> ∈ <sup>Z</sup>+, r<sup>i</sup> ≤ C. Observing that this equals min(ri) κ i=1 r<sup>1</sup> + r<sup>1</sup> P<sup>κ</sup> j=2 Q<sup>j</sup> <sup>i</sup>=2 r<sup>i</sup> implies a standard dynamic-programming approach and store a table of factors up to C to optimize this in O(Cκn) time for C, κ generally pre-fixed constants.

Low-rank distance matrix C. A key work [\(Indyk et al.,](#page-10-16) [2019\)](#page-10-16) showed that one may approximately factor a distance matrix C with linear complexity in the number of points n (Algorithm [E.1\)](#page-28-0). For certain costs, e.g. squared Euclidean, this factorization can be given for free [\(Scetbon et al.,](#page-12-9) [2021\)](#page-12-9). We rely on both of these for low-rank factorizations of the distance matrix, so that both the space of the coupling and pairwise distance matrix scale linearly.

Algorithm 3 Low-Rank approximation for distance matrix C

Input point sets {xi} n <sup>i</sup>=1, {yj}<sup>M</sup> <sup>j</sup>=1 in metric space X and metric d Pick indices i <sup>∗</sup> ∈ [n], j<sup>∗</sup> ∈ [m] uniformly at random for i = 1 to n do Update sample probability p<sup>i</sup> = d(x<sup>i</sup> , y<sup>j</sup> ∗ ) <sup>2</sup> + d(x<sup>i</sup> <sup>∗</sup> , y<sup>j</sup> ∗ ) <sup>2</sup> + 1 m P<sup>m</sup> <sup>j</sup>=1 d(x<sup>i</sup> <sup>∗</sup> , y<sup>j</sup> ) 2 end for Sample <sup>O</sup>(r/ε) rows <sup>C</sup>i,. <sup>∼</sup> Categorical Ppi pi Compute U using [\(Frieze et al.,](#page-10-22) [2004\)](#page-10-22) Compute V using [\(Chen & Price,](#page-9-16) [2017\)](#page-9-16) return V, U

![](_page_29_Figure_1.jpeg)

Figure S1. Abundance of 5 genes (a. *Slc17a7*, b. *Grm4*, c. *Olig1*, d. *Gad1*, e. *Peg10*) in Allen Brain Atlas MERFISH dataset [\(Clifton](#page-9-14) [et al.,](#page-9-14) [2023\)](#page-9-14). From left to right are plotted (1) abundance in the first dataset, (2) abundance in the second dataset, and (3) predicted abundance via transfer of the abundances in the first dataset under the mapping of HiRef.

![](_page_30_Figure_1.jpeg)

Figure S2. Runtime scaling across sample-complexities n of a. Hierarchical Refinement (HiRef) and b. Sinkhorn for Euclidean cost, ∥·∥<sup>2</sup> (single CPU core). Hierarchical Refinement exhibits linear scaling for increasing n, whereas Sinkhorn exhibits quadratic scaling and is unable to run beyond 16k points. c. Runtime scaling of HiRef (GPU).

![](_page_30_Figure_3.jpeg)

Figure S3. HiRef cost and the cost of the low-rank OT solution of FRLC [\(Halmos et al.,](#page-10-8) [2024\)](#page-10-8) across the coupling rank r ∈ [5, 100].

![](_page_31_Figure_1.jpeg)

Figure S4. Comparison of optimal transport maps under (1) the HiRef alignment, and (2) the Sinkhorn [\(Cuturi,](#page-9-0) [2013\)](#page-9-0) barycentric projection. a. The checkerboard dataset of [\(Makkuva et al.,](#page-11-9) [2020\)](#page-11-9), b. the Half-moon and S-curve dataset of [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5), and c. the MAF-Moons Rings dataset of [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5).

![](_page_32_Figure_1.jpeg)

Figure S5. Alignments of the synthetic datasets of [\(Makkuva et al.,](#page-11-9) [2020;](#page-11-9) [Buzun et al.,](#page-9-5) [2024\)](#page-9-5) using the optimal dual revised simplex [\(Huangfu & Hall,](#page-10-17) [2018\)](#page-10-17) algorithm for small instances (512 points). a. The checkerboard dataset of [\(Makkuva et al.,](#page-11-9) [2020\)](#page-11-9), b. the Half-moon and S-curve dataset of [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5), and c. the MAF-Moons Rings dataset of [\(Buzun et al.,](#page-9-5) [2024\)](#page-9-5).