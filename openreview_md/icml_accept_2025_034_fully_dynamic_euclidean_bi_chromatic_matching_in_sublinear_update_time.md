# 

Gramoz Goranci 1 Peter Kiss 2 Neel Patel 3 Martin P. Seybold 4 Eva Szilagyi 5 **David Zheng** 6

## Abstract

We consider the Euclidean bi-chromatic matching problem in the dynamic setting, where the goal is to efficiently process point insertions and deletions while maintaining a high-quality solution. Computing the minimum cost bi-chromatic matching is one of the core problems in geometric optimization that has found many applications, most notably in estimating Wasserstein distance between two distributions. In this work, we present the first fully dynamic algorithm for Euclidean bi-chromatic matching with sub-linear update time. For any fixed ε > 0, our algorithm achieves O(1/ε)-approximation and handles updates in O(n ε) time. Our experiments show that our algorithm enables effective monitoring of the distributional drift in the Wasserstein distance on real and synthetic data sets, while outperforming the runtime of baseline approximations by orders of magnitudes.

## 1. Introduction

1 points in R
dand a set B of n blue points in R
d, the goal is to compute a bijection µ : A → B such that Pa∈A∥a −
µ(a)∥2 is minimized. A key application of minimum-cost bi-chromatic matching in machine learning is estimating the 1-Wasserstein distance of two spatial distributions ν and ν
′, whose supports satisfy Sν, Sν′ ⊆ R
d. This distance, also known as Euclidean Earth Mover's Distance (EMD), Kantorovich-Rubinstein metric, or Mallow's distance, is defined as

$$W_{1}(\nu,\nu^{\prime})=\operatorname*{inf}_{P\in\Gamma(\nu,\nu^{\prime})}\int_{S_{\nu}\times S_{\nu^{\prime}}}\|a-b\|_{2}\;d P(a,b)$$

were Γ(*ν, ν*′) denotes the set of all probability measures on Sν × Sν′ with marginal distributions ν and ν
′.

Estimating the 1-Wasserstein distance measure, or the Euclidean bi-chromatic matching problem, has been extensively studied across a wide range of disciplines, including machine learning (Liu et al., 2018; 2017; 2018; Cao et al., 2019; Balaji et al., 2020; Tolstikhin et al., 2018), computer vision (Rubner et al., 2000), statistics (Panaretos and Zemel, 2019) and economics (Galichon, 2018). This broad interest has led to different problem varaiants, e.g., the point reweighted setting, and the development of algorithms with high-accuracy approximation guarantees or those that only estimate the solution size without explicitly outputting the underlying matching (for a more detailed discussion, please see Appendix A). In this work, we consider the Euclidean bi-chromatic matching in the *dynamic setting*, where the input undergoes updates such as insertions or deletions of pairs of points. The goal is to process these updates as *fast* as possible while maintaining a matching that achieves a small, provable approximation ratio compared to the optimal matching at any given time. The dynamic setting is particularly relevant in the statistical estimation of the 1-Wasserstein distance (Gattani et al., 2023; Beugnot et al., 2021; Liu et al., 2018; Panaretos and Zemel, 2019), when the closed-form expressions for ν and ν
′are unknown. In such cases, one can efficiently draw independent and identically distributed (i.i.d.) samples from both distributions. The resulting empirical distribution νn and ν
′n, assign a uniform probability mass of 1/n to each sample in A and B, respectively. It is well known that W1(νn, ν′n
) converges to W1(*ν, ν*′), as n → ∞. Rather than computing a matching from scratch after new samples are drawn, a dynamic algorithm guarantees efficient maintenance of accurate estimates for minµ 1 n Pa∈A∥a − µ(a)∥2 under such changes. Further real-world applications of dynamic geometric matching include measuring the similarity between evolving data sets (Alvarez-Melis and Fusi, 2020), measuring the change in patient data (such as MRI images) over a long period of time (Gramfort et al., 2015; Janati et al., 2019), or applications of matching (or EMD) as a metric in time series analysis (Cheng et al., 2021). Despite the fundamental importance of this problem, dynamic geometric matching has only recently been considered in (Xu and Ding, 2024a) and the existing algorithms cannot break the linear barrier on the update time.

## 1.1. Main Contributions

We obtain the first sub-linear dynamic algorithm for Euclidean bi-chromatic bipartite matching. Specifically, our algorithm achieves O(1) approximation in sub-linear in n update time. The exact guarantees of our result are summarized in the theorem below.

Theorem 1.1. For any 0 < ε ≤ 1*, there exists a fully* dynamic algorithm that maintains an expected O(1/ε)- approximate solution to the Euclidean bi-chromatic matching problem defined on the point-sets *A, B* ⊂ R
2, |A| = |B|, while pairs of points are inserted into and deleted from *A, B* in O(n ε· ε
−1) worst-case update time. Here, n *denotes the* maximum size of A *throughout the update sequence.*
In the static setting, the Euclidean bi-chromatic matching problems admits known (1 + ε)-approximation algorithms that run in near-optimal running time (Agarwal et al., 2022b; Raghvendra and Agarwal, 2020). In comparison, the O(1/ε) approximation guarantee in Theorem 1.1 may initially seem less competitive. However, as we show in Appendix F (Theorem F.1), a simple lower bound construction reveals that, in the dynamic setting (even if restricted to only point insertions or deletions), there is no algorithm that simultaneously achieves (2 − δ)-approximation and runs in sub-linear update time, for any δ > 0. This in turn implies that no non-trivial dynamic algorithm can maintain a solution to the problem with an arbitrarily tight approximation ratio. While the matching problem on graphs has received significant attention in the dynamic model (see (Azarmehr et al., 2024) for a complete list of works), the only prior known algorithm for the geometric version of the problem was presented by (Xu and Ding, 2024a) with update time O(n). Notably, even disregarding edge weights, the maintenance of a perfect matching in the graph setting requires Ω(n)
update time under well accepted hardness conjectures (Henzinger et al., 2015; Dahlgaard, 2016). On the other hand, there is a long line of work focusing on maintaining an O(1)-approximate matching in near optimal update time in dynamic graphs. Furthermore, dynamic graph matching algorithms focus on singular edge updates. As in the dynamic Euclidean bichromatic matching problem we have to work with updates affecting Ω(n) point-wise distances, in order to obtain our results, we need techniques which are new to dynamic literature and heavily exploit the guarantees of the geometric setting. To complement our primary contribution (Theorem 1.1), we present experimental evaluations of the performance of our dynamic algorithm for estimating the 1-Wasserstein distance in contrast to static state of the art approximate algorithms. Our experimental results reveal that in the dynamic setting the total running time of our algorithm over a series of updates outperforms periodic static recomputation by orders of magnitudes. Furthermore, we find that the approximation ratio obtained by our algorithm on real-world datasets is smaller than that of our theoretical guarantees and almost matches that of static algorithms. Our results confirm the expected approximation ratio drop-off and running time decrease we would expect with the reduction of the ε parameter.

## 1.2. Technical Overview

In vein of related works, we discuss algorithms and analysis for d = 2 to simplify the presentation, though our theorems and implementation extend to constant d > 2. Our dynamic algorithm builds on the static algorithm of (Agarwal and Varadarajan, 2004). On a very high level, the algorithm partitions the input points into a series of nested grid cells. The nested grid is constructed in a such a way that each grid cell contains O(n ε) sub-cells of the next level, bottom level cells contain at most O(n ε) points and the data-structure consists of O(ε
−1) levels. Intuitively, the algorithm aims to match every point within the smallest grid cell possible, however, certain cells might not have the same number of red and blue points. A crucial difference between our implementation and that of (Agarwal and Varadarajan, 2004) is that we describe a process which constructs the matching in a bottom-up manner, starting with the smallest cells and progressing towards the cell containing all of the input, whereas the prior work iterates in a top-to-bottom order. This change allows for a shorter algorithm description; however, it requires a different but arguably simpler analysis, which we present in Appendix C. As bottom cells contain only a small number of input points, they may select a maximal color-balanced subset of them, match its points optimally using a polynomial running time static algorithm, and forward the remaining points to their parent cell.

If a set of points can't be matched within a cell due to color dis-balance, we can argue that an optimal cost matching must match the dis-balance with points outside of the cell, and hence with longer edges. This allows us to introduce some slack and represent these forwarded points implicitly by the center of their respective cell and their cardinality. Larger cells aiming to match these forwarded implicitly represented points can efficiently find an implicit matching of the underlying points by an optimal polynomial running time geometric transportation algorithm. These implicit matchings can then efficiently be turned into an actual matching of the underlying points of roughly the same cost. To dynamize the algorithm, we show that, due to an update, the matching structure might only change in cells containing the updated points. After an update for each affected cell in a bottom-up manner we carefully adjust the matchings calculated. Since in each cell we either have O(n ε) points or a larger set of points represented with O(n ε) sub-cell center points, this can be done with polynomial running time static sub-routines for each affected cell. As the nested grid contains O(ε
−1) levels, the update time works out to O(n ε· ε
−1).

Updating the matching of a bottom cell is straightforward, as we may run an optimal polynomial time algorithm on their updated points. However, the sub-cell center-points of intermediate cells may represent Ω(n) input points implicitly. Hence, if the implicit matching of the cell significantly changes due to an update, it may take Ω(n) time to update the actual output matching within the cell. To overcome this difficulty, we first carefully analyze the update process and show that the points matched in each cell may only change by 2 due to an update. Afterwards, we present a graph shortest s − t path sub-routine based procedure which calculates an updated version of the geometric transport solution stored in the cell. This solution only differs in at most O(n ε) edges compared to its unupdated state. In turn, this allows us to efficiently update the matching of the actual input points implicitly represented by the geometric transportation sub-solutions.

## 1.3. Related Work

Numerous results exist on special cases and generalizations of Euclidean bi-chromatic bipartite matching across various computational settings. In the static setting, an exact solution can be found through minimum cost max-flow graph algorithms in Oˆ(n 2) time (van den Brand et al., 2020). For restricted cases of the problem, (Sharathkumar, 2013) has presented an algorithm with Oˆ(n 3/2) running time. In the approximate setting, (Agarwal et al., 2022a) presented the first deterministic O˜(n) time algorithm for arbitrarily tight approximations. Due to space restrictions, a more detailed overview and discussion on related work can be found in Appendix A. Paper Outline: We describe a static algorithm and its dynamic implementation in sections 3 and 4 respectively. In Section 5, we demonstrate the practicality and efficiency of the dynamic algorithm for estimating the 1-Wasserstein distance.

## 2. Preliminaries

The Euclidean matching problem Given a set of red points A and a set of blue points B in the plane, where |A| = |B| = n, a *matching* is a bijection of the form µ P
: A → B. The *cost* of a matching µ is given by c(µ) =
a∈A ∥a − µ(a)∥2. The objective is to find a matching of *minimum* cost. We say that a pair of points e := (*a, b*), referred to as an *edge*, belongs to matching µ if b = µ(a). We define a matching µ to be α-approximate for α ≥ 1 if c(µ) · α ≤ c(µ
∗) for any optimal solution µ
∗.

To simplify presentation, we assume that X = A ∪ B ⊆ D × D for some D = poly(n), where D is a power of 2.

Throughout this paper we make the standard assumption that the *spread* of the input points, i.e. the ratio between the largest and the smallest inter-point distance in X, remains bounded by some large U = poly(n).

A generalization of the matching problem, supporting point multiplicities, is the so-called Euclidean transportation problem, defined as follows.

The Euclidean transportation problem Let *A, B* ⊂ R
2 be any two sets of points such that each point a ∈ A has a non-negative integer *supply* sa ≥ 0 and each b ∈ B has a non-positive integer *demand* db ≤ 0 satisfying P
P 
a∈A sa +
b∈B db = 0. An *assignment*, γ : A × B → Z≥0, is a P
non-negative set of weights on the edges such that sa =
b∈B γ(*a, b*), ∀a ∈ A and −db =Pa∈A γ(*a, b*), ∀b ∈ B.

The P
cost of an assignment γ is c(γ) =
a∈A,b∈B γ(*a, b*)∥a − b∥2. An *optimal* assignment for the Euclidean transportation problem is an assignment of *minimum* cost. Theorem 2.1. (Atkinson and Vaidya, 1995) There exists a deterministic algorithm which, given an instance of the Euclidean transportation problem *A, B* ⊆ R
2, s ∈ Z
A
≥0, d ∈ Z
B
≤0*, returns an optimal assignment* γ : A × B → Z≥0 *in time* O(n 5/2log(n log M))*, where* M = max(∥s∥∞ , ∥d∥∞), n = |A ∪ B|.

The results of this paper rely on a standard data structure for representing points in a geometric setting (see (Agarwal and Varadarajan, 2004; Agarwal et al., 2022a)), the underlying and restricted p-trees, two quadtree-like structures. Throughout the paper think of the parameter p as some n ε sized power of 2.

The underlying and the restricted p**-tree** We define a cell of side-length L > 0 as a square of the form [*a, a*+L)×
[*b, b* + L) for some *a, b* ∈ R. We will frequently subdivide cells of side-length L into *subcells* of side length L/p. We will call this a *grid* of side length L/p. For an integer p ≥ 2, which is also a power of 2, and a point set X, we define an underlying p-*tree* T on X to be a quadtree-like structure with larger branching factor (see e.g., (Agarwal and Varadarajan, 2004)), formally described as follows.

(1) The **root** r¯ consists of a bounding cell with side-length D, containing all points in X, and its subdivision into p 2smaller subcells. The root r¯ has p 2children, denoted by Cr¯, corresponding to its subcells, each of side-length D/p. The root is at level 0 of T.

(2) Each **internal node** v at level i > 0 corresponds to one of the subcells of its parent node. Thus, an internal node v at level i consists of a cell of side length D/pi, and its subdivision into p 2smaller subcells, each of size D/pi+1 × D/pi+1.

For a node v of the underlying p-tree T, let Xv be the set of points from X belonging to the bounding cell of v. We say that a point x ∈ X *belongs to node* v if x ∈ Xv. Now, notice that for some nodes v (on both low and high levels)
the set Xv might be empty. To avoid storing unnecessary empty cells (which would result in a slower data structure), we define the restricted p-*tree* or shortly p-*tree*, which is a subtree of the underlying p-tree, where the root and leaves are defined as follows:
(1) The **root** r is a node of the underlying p-tree such that Xr = X, but which does not have a child v with Xv = X. In other words, r is the node with Xr = X
located on the lowest possible level of the underlying p-tree.

(2) A **leaf** of the p-tree is a cell v such that |Xv| ≤ p 2, but whose parent u has |Xu| > p2.

As we insert or delete points in the dynamic setting, notice that the restricted p-tree might change over time: the root, as well as the leaves might change to higher or lower levels of the underlying p-tree. However, as we assume that the spread of the pointset X is bounded by U at all times, we know that the height of the restricted p-tree is bounded by d := logp U and is therefore constant if p is chosen to be sublinear in n. It can be shown that a restricted p-tree w.r.t. set X, |X| = n, can be maintained under insertions and deletions of points in O(p 2 + logp U) amortized update time (see e.g. (de Berg et al., 2007)).

z**-ordering** The z-ordering of the point-set X w.r.t. an underlying p-tree T is defined via specific ordering of children nodes in T. Starting from the root of T, we traverse the children cells of each internal node row by row, then recursively traverse the subcells of each cell until a subcell contains only one point.

## 3. Basic Framework And The Static Algorithm

The starting point of our work is the static algorithm for computing bi-chromatic Euclidean matchings due to (Agarwal and Varadarajan, 2004). Specifically, their top-down algorithm computes an O(1/ε)-approximate matching for an instance (A, B) in time O(n 1+ε) for sufficiently small constants ε > 0. We begin by presenting our modified variant of this algorithm for the static setting, referred to as STATIC-MATCHING(*A, B*). The next section shows that our variant can be extended to a bottom-up algorithm for the fully dynamic setting achieving sub-linear update time. For the input sets *A, B* with bounded spread U we first apply a random shift to all the points by a fixed random vector r ∈ [U]
2to get A′ = {a + r : a ∈ A} and B′ = {b + r : b ∈ B}. From now on, we are only interested in sets A′and B′. Slightly abusing notation, we will simply refer to these sets as A and B. Our first step is to construct a restricted p-tree T on the point set X := A ∪ B. Using T in a *bottom-up* manner, we obtain a matching between the point set A and B as follows. Intuitively, as the number of unmatched red and blue points belonging to some node might not be balanced, we greedily match as many red-blue pairs of points as possible at that node. Handling the unmatched points is deferred to the parent node, which matches as many excess points from its children as possible, and so forth. Therefore, for any node v of T, we keep track of the set of red points Yv(A) and blue points Yv(B) that we match at that node, and the set of unmatched monochromatic points that are in excess, denoted by Ev.

An essential element of our algorithm is the choice of the matching algorithm used throughout the traversal of T. Observe that the excess points forwarded to an internal node of T by its children could be of size Ω(n). To overcome this problem, our algorithm will rely on a data structure representing excess points forwarded by node v of T by the center point of the cell of v. Using this data structure we are able to efficiently compute (and maintain) an implicit matching of the unmatched points at all v ∈ V , which we can efficiently turn into an *explicit matching* of the underlying points when queried.

## 3.1. Sub-Instances Via Transportation Solver

An *implicit matching* is an optimal assignment γ
∗
v[Yv] : v ∈
T associated with a vertex v of T of an Euclidean transportation problem defined by some inputs (Yv(A), Yv(B)*, s, d*).

To make sense of the above vague definition, we describe procedures IMPLICIT-MATCHING and EXPLICIT-
MATCHING that will act as a sub-routines in both static and dynamic algorithm for Euclidean matchings. The purpose of these functions is to obtain two representations of a maximal matching between specific points forwarded to an internal vertex from its children in a restricted p-tree. Given a subset of points belonging to a node in the p-tree, in the IMPLICIT-MATCHING, we first define the notion of a aggregated sub-instance for that node. To aggregate the points of a given cell, we move all of them to the cell's center. We define an Euclidean transportation problem on this aggregated sub-instance, and the resulting optimal assignment will correspond to an *implicit matching* of v. EXPLICIT-
MATCHING converts this implicit matching into a matching of the original subset of points, which we refer to as the explicit matching.

IMPLICIT-MATCHING **procedure** The input of the procedure is a vertex v of p-tree T, the monochromatic sets of points {Eu : u ∈ Cv} (denote their union by E, and partition into blue EB and red EA points) forwarded to v from its children and their cardinalities {|Eu| : u ∈ Cv}.

Without loss of generality, assume that |EB| ≥ k = |EA|.

Its output consists of an optimal assignment of a transportation problem (i.e., an implicit matching) defined by k points of EB and all points of EA (i.e., the points we can match within the cell of v) and |EB| − k blue points, which will be forwarded to the parent of v in T. Choosing the points to match. Let Yv(B) stand for the first k points of EB with respect to some ordering (say the z-ordering). Let Ev = EB \ Yv(B) be the *excess set*, i.e. the set of points not matched at v, Yv(A) = EA and Yv = Yv(A) ∪ Yv(B). Excess set Ev will be forwarded to the parent of v. Computing the implicit matching. We define an instance of Euclidean transportation problem over the center points of all p 2sub-cells of the cell of v. Note that every such sub-cell u ∈ Cv contains a monochromatic subset of E. Define Ac and Bcto be center points of sub-cells containing only red and blue points of Yv respectively. We set sa′ for a
′ ∈ Ac and −db
′ for b
′ ∈ Bcto correspond to the number of red and blue points of Yv of the cells with centers a
′and b
′,
respectively. The optimal assignment γ
∗v[Yv] obtained by solving the Euclidean transport problem defined by (Ac, Bc*, s, d*) using the algorithm of Theorem 2.1 will be returned as the implicit matching of vertex v. Crucially, observe that setting up the Eucledian transport instance doesn't require the knowledge of the exact location of vertices in Yv(A), Yv(B) just their cardinalities in the sub-cells of v.

EXPLICIT-MATCHING **procedure** The inputs of the procedure are sets Yv(A) and Yv(B), as well as implicit matching γ
∗
v[Yv] between those points at an internal node v. Its output is a matching between Yv(A) and Yv(B).

The procedure naturally translates the assignment γ
∗
v[Yv] to a perfect matching µ
′v[Yv] between the points Yv(A) and Yv(B) as follows: match γ
∗
v[Yv](a
′, b′) many different pairs of points of the form (*a, b*), where a and b lies in the sub-cell whose centers are a
′and b
′respectively. We refer to the latter matching as an *explicit* matching of node v w.r.t. its corresponding point set Yv. For an example involving both implicit and aggregated sub-instances, see Figure 1. In the following lemmas, we summarize the above algorithms and their running times.

Lemma 3.1. Let v be an internal node of a p-tree T
on X := A ∪ B with |A| = |B| = n*. Let* E =
{Eu : u ∈ Cv} be the pointers to some monochromatic sets of v's children forwarded to v, as well as their cardinalities {|Eu| : u ∈ Cv}*. There is an algorithm* IMPLICIT-MATCHING(v, E, {|Eu| : u ∈ Cv}) *that chooses* balanced red and blue sets Yv(A) and Yv(B), computes an implicit (perfect) matching γ
∗
v[Yv] *between them, in time* O(p 5log p log n). Additionally, the function returns the
(pointer to the) monochromatic set of unmatched points Ev at node v. A crucial property of the Lemma 3.1 is that it has a running time only polylogarithmic with respect to n although E could consist of Ω(n) points. This can be naturally achieved through an implicit representation of the point sets {Eu :
u ∈ Cv}. We formalize this process in Appendix B.

Now, as described above, it is easy to convert an implicit matching to a matching of the input points. Lemma 3.2. Let T be a p-tree on X = A ∪ B and d its depth. The following statements hold:
1. Given a vertex v of a p-tree T*, and an implicit matching* γ
∗v[Yv] w.r.t. v and set Yv, the corresponding explicit matching γ
′v[Yv] can be obtained in time O(|Yv|d).

2. Given tree T where each leaf stores a matching and each non-leaf node an implicit matching, the corresponding perfect matching on X *can be reported in* O(n) *time.*

## 3.2. Static Algorithm

Next, we present the formal description of the bottom-up algorithm. To this end, we define the subroutine MATCH(v)
as follows. If node v is *leaf* of the restricted p-tree:
1. **Determine sets** Yv(A), Yv(B) and Ev: Let Y
′v
(A) =
Xv ∩ A be the red points, Y
′v
(B) = Xv ∩ B the blue points, and ηv = |Y
′v
(A)*| − |*Y
′
v
(B)| the difference in cardinality, at node v.

-3 2 2 1 3 4
-3 
If ηv = 0, we set Ev = ∅. If ηv > 0, we assign a subset of ηv red points chosen as the last ηv points in the z-ordering from Y
′v(A) to Ev, and set Yv(A) =
Y
′
v(A) \ Ev and Yv(B) = Y
′v(B) (in the same way as presented in Section 3.1). Analogously, if ηv < 0, we assign a subset of ηv blue points from Y
′
v(A) to Ev, and set Yv(B) = Y
′
v(B) \ Ev and Yv(A) = Y
′
v(A).

We refer to Ev as the *excess points* at v.

2. **Compute matching:** Obtain a matching µv between red points Yv(A) and blue points Yv(B) using any efficient exact algorithm (e.g. the Hungarian algorithm (Munkres, 1957)), and compute its cost c(µv).

3. Return the cost and the excess set (c(µv), Ev).

Otherwise, node v is *internal*:
1. **Compute implicit matching:** For each child u of v, let Eu be the set of excess points returned by MATCH(u). Invoke algorithm IMPLICIT-
MATCHING(v, E, {|Eu| : u ∈ Cv}) from Section 2
(see Lemma 3.1). Let γ
∗v[Yv] be the resulting implicit matching w.r.t. node v and set Yv computed via the algorithm, and Ev the set returned by the algorithm.

Set the estimate of the cost for the matching in v to c(µv) := c(γ
∗v[Yv]) + Pu∈Cv c(µu).

## 2. Return The Cost And Excess Set (C(Μv), Ev).

Our algorithm STATIC-MATCHING(*A, B*) first invokes the MATCH(r) subroutine, where r is the root of T on A ∪ B.

Note that this yields implicit matchings γ
∗
v[Yv] at internal nodes v of T. Using Lemma 3.2, Part (2), we can transform the implicit matchings γ
∗
v[Yv] of the aggregated versions of Yv into matchings of the original points µ
′[Yv] of roughly the same cost (see Claim C.4 for a formal bound on the slack introduced by the aggregation). Finally, we return a maximal matching µr on the entire point set X, along with its already computed cost c(µr).

Our analysis of the algorithm builds on the following key idea from (Agarwal and Varadarajan, 2004): when all points are shifted randomly by the vector r, the probability of an edge being cut by a grid in a node of the p-tree is proportional to the length of the edge and inversely proportional to the spacing of the grid. However, the algorithm of (Agarwal and Varadarajan, 2004) works in a top-to-bottom approach (building the matching starting from the root and progressing to the leaves) in contrast to our algorithm, which takes a bottom-up approach. The bottom-up approach allows us to guarantee that for any node v ∈ T the number of nodes not matched within the cell of v simply corresponds to the difference in the number of blue and red nodes inside the cell of v. Intuitively, this implies that points are matched to near-by neighbor whenever possible. This results in an arguably simpler analysis differing from that of (Agarwal and Varadarajan, 2004), and crucially relying on the concept of implicit matchings. We provide a selfcontained analysis of the static algorithm in Appendix C.

Theorem 3.3 (Static). *For any Euclidean matching instance*
(A, B), with |A| = |B| = n*, and spread* U ≤ n c*, for* some constant c > 0*, and for any constant* ε ∈ (0, 1 10 
),
algorithm STATIC-MATCHING(A, B) *computes an* O(1/ε)- approximate Euclidean matching with high probability in O(n 1+ε· ε
−1) *time.*

## 4. Dynamic Algorithm

In this paper, we present two dynamic data structures for maintaining perfect matchings under insertions and deletions of pairs of red and blue points. Both data structures naturally convert our static, bottom-up algorithm from the previous section into a dynamic one. Our first algorithm EUCLMATCH1 implicitly maintains a matching for each node of the tree. This allows us to (i)
estimate the cost of the entire matching after each update and (ii) report the current approximate matching upon a query in time proportional to its size. Our second algorithm EUCLMATCH2 explicitly maintains the output matching at all times. We defer the second algorithm to Appendix E due to page limitations. Initialization For any given set of points A and B, we will next describe how to initialize a matching by essentially running the bottom-up algorithm STATIC-MATCHING on the instance (*A, B*). Formally, the INITIALIZE(*A, B*) initializes the restricted p-tree T w.r.t. X := A ∪ B, and the following corresponding values: the set of unmatched points Y
′v(A), Y ′
v(B), the excess set Ev at each node v of T, the implicit matching γ
∗[Yv] and the cost c(µv) of the matching w.r.t. the subtree rooted at each internal node, and the matching µv and its cost c(µv) at each leaf. This process takes O(n 1+ε) time by Theorem 3.3.

Handling updates As the procedures for insertions and deletions are similar (see Appendix D), we focus on presenting the insertion procedure. Without loss of generality assume that we insert a red point a to A. The algorithm INSERT(a) (see Algorithm 2) proceeds in two steps: updating the restricted p-tree T, and updating the matchings at each *affected* node, i.e. node v where a ∈ Xv.

First, updating T follows a standard approach, similar to the method described in e.g. (Har-Peled, 2011). Namely, one or both of the following cases can happen:
1. After insertion of a to A, a leaf v of T might contain more than p 2 points. In this case, v now becomes an internal node of T, while a new subtree has to be built under v. We refer to these nodes as *new subroots*, while all other nodes belonging to this subtree are called marginal nodes. Finally, all nodes in T that are not marginal are refered to as *non-marginal nodes*.

2. If a does not belong to the current root of the restricted p-tree, a internal node of the underlying p-tree that is on a higher level than the previous root now becomes the new root of T.

Second, we define the recursive procedure INSERT-UPDATE(*v, a*) which updates the matchings at each affected node of T in a bottom-up manner (see Alorithm 1). Specifically, we define INSERT-UPDATE(v, a) for three types of affected nodes of T: non-marginal internal nodes, non-marginal leaves, and new subroots. We focus on presenting the case for non-marginal internal nodes, as the other two cases are simpler, and can be found in Appendix D. If a node v is a non-marginal internal node, the procedure INSERT-UPDATE(*v, a*) first updates set Eu for the affected child u of v, via INSERT-UPDATE(*u, a*). Then, we update the excess set Ev accordingly. Finally, we invoke the IMPLICIT-MATCHING(·) procedure to obtain an implicit matching for the remaining points Yv at v.

Crucially, observe that throughout this process the affected nodes of T, that is nodes at which either the implicit matching or the excess set changes, lie along a path from a leaf to the root of T. Hence, the most costly operation of the algorithm, the updating of the implicit matchings, only has to be repeated at a small number of nodes. The correctness of the algorithm follows from the algorithm always satisfying that after handling an update its output corresponds to the run of a static algorithm on the current state of the input, up until the different choices the underlying optimal bi-chromatic matching algorithm (Hungarian algorithm) and optimal geometric transport algorithm (see Theorem 2.1) makes. Note that the choice of underlying exact algorithms is arbitrary as long as they run in polynomial time. Queries To report the changes to the matching, the algorithm converts the implicit matchings at the affected nodes to an explicit matching of X via Lemma 3.2. To query the cost of the approximate solution the algorithm simply returns the approximate cost value c(µr) that is maintained at root r of T. The following theorem summarizes the results of our simpler dynamic algorithm. Theorem 4.1. Let ε ∈ (0, 1] and (A, B) *two point sets in* R 
2 with |A| = |B| = n, and spread U < ncfor some constant c > 0. There is a dynamic data structure that supports insertions and deletions in O(n ε· ε
−1) worst-case update time and maintains an expected O(1/ε)-approximate matching between the dynamic set (A, B)*. Initialization takes* O(n 1+ε) time. Reporting the matching, the changes to the matching when an insertion or deletion occurs (recourse),
and the cost of the matching, take O(n), O(n) and O(1)
time, respectively. Improved dynamic algorithm Ideally, the algorithm would maintain the solution matching explicitly. This does require however, that the explicit matchings implicitly represented at all nodes of T are explicitly maintained at all times. This requires a careful propagation of changes throughout the p-tree and the ability to update explicit matching efficiently when the underlying implicit matching changes. To dynamize the maintenance of explicit matchings, we rely on the fact that a single insertion might only change the excess set stored at each node by 1 point. This implies that the difference between the updated and pre-updated matching lies along a short alternating path of old and new matching edges. We find this augmenting path using a graph shortest-path based sub-routine, which, due to page limitations, we defer to Appendix E.

## 5. Experimental Evaluation

With this work, we provide1an opensource, single-threaded C++ implementation of our algorithm in Appendix E. To demonstrate the practicality and effectiveness our proposed algorithms, we conducted experiments on a 2.2 GHz Ubuntu 22.04.4 system with AMD Opteron 6174, using synthetic and real world data sets. For reproducibility and comparability, we report on measurements for the same 2D distributions and data sources used in the recent work (Gattani et al., 2023). For the *synthetic data*, we drew the point coordinates from the uniform distribution on integers between 1 and 500 to obtain the Uniform datasets. For the Gaussian datasets, we drew the point coordinates from the normal distribution (µ = 0.5, σ = 0.25) prior to scaling by a factor of 500 and rounding to integers. For the real data, we extracted all trips from the Yellow-Cap Taxi dataset2 of December 2009, and, following (Gattani et al.,
2023), extracted those having at least 3 minutes duration, at most 110 mph speed, and latitude/longitude in the range of [−74.5, −73.5] × [40, 41], yielding 14 324 017 trips with one 'Pickup' and 'Dropoff' location each.

## 5.1. Experimental Results

Speedup: Dynamic vs Static Approximation Algorithms To measure the speedup or of dynamic over baseline approximations, we measured the update time and time to compute a approximate matching from scratch, for various levels of approximation quality. Figure 2 shows the speedup factors
(p = 8) for inserting ≤ 8 000 samples from Uniform vs Guassian (top) and Pickup vs Dropoff (bottom). Insertion-Workloads: Estimating Wasserstein distance using Exact vs Approximate Matchings To demonstrate the usability of approximations of minimum cost bichromatic matchings to estimate the Wasserstein distance, we estimated the empirical Wasserstein distance using exact and approximate matchings on identical and different distributions for various levels of approximations and sample sizes. Figure 3 shows the empirical 1-Wasserstein distance of two Uniform distributions (top) using exact minimum cost matchings and approximations with p = 2, p = 8, and p = 32, for up to 2 000 samples. The bottom figure shows the same estimates for Uniform vs Gaussian.

Insertion-Workloads: Convergence and Update Time for Very Large Sample Sizes To demonstrate the scalability of our approach, we measured the convergence and update time of our dynamic approximations for sample sizes up to 1 000 000, rendering the exact solvers impractical. That is, running the code of (Xu and Ding, 2024b) on Uniform 1https://github.com/Zhengdw/dyn-euc-match 2See https://www.nyc.gov/site/tlc/about/
tlc-trip-record-data.page.

vs Gaussian slowed down substantially with increasing number of samples. Inserting 2 000 samples took more than 12GB RAM and longer than 3 hours in total, which is when we had to terminate the process as it was running out of memory. (Note that this is already > 5s per update on a *very small* instance.) In contrast, Figure 4 (top) shows the convergence of empirical 1-Wasserstein distance, using p = 2, 4, 8 for approximations, for the Uniform vs Uniform benchmark (bottom three curves) and the Uniform vs Gaussian benchmark (top three curves). The bottom figure shows the update time, in *milliseconds*, of our dynamic algorithm on both benchmark data sets, having sample sizes up to 1 000 000. Fully-Dynamic Workloads: Drift of Pickup vs Dropoff distributions over time To demonstrate the effectiveness of our dynamic algorithms to monitor the drift of timedependent, spatial distributions, we sorted the trips in the Taxi dataset by pickup time and use a sliding window, having a width of 10 000 samples, to obtain update sequences that contain insertions and deletions. Figure 5 (top) shows the empirical 1-Wasserstein distance between the Pickup and Dropoff locations, using approximations with p = 4 and p = 16. The bottom figure shows the update time of our fully dynamic algorithm on the Taxi benchmark.

## 5.2. Discussion Of Experimental Results

The experimental results show that the dynamic algorithm is orders of magnitudes faster than computing static approximations for minimum cost bi-chromatic matching (Figure 2). In practice, the approximate matching costs are within a very small factor (i.e. < 2) of the minimum bi-chromatic matching cost, which enables very accurate and effectiv estimation the 1-Wasserstein distance (Figure 3 and top part of Figure 4). As suggested by our Theorem 1.1, we observe an tradeoff between approximation quality and update time (bottom part of Figure 4). Updates of the dynamic algorithm took typically between one and ten millisecond and showed a clear separation, between inputs with zero and non-zero distance, at sample sizes around 10 000 samples. The time dependent spatial distributions of Pickup and Dropoff locations in the Taxi dataset show a drift in the empirical 1-Wasserstein distance (Figure 5 top), and the dynamic algorithm allows for effective monitoring of the change even on large window sizes on real data sets (Figure 5 bottom).

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

unit-capacity graphs. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 93–104.

Pankaj K. Agarwal, Hsien-Chih Chang, Sharath Raghvendra, and Allen Xiao. 2022a. Deterministic, near-linear ε-approximation algorithm for geometric bipartite matching. In STOC '22: 54th Annual ACM SIGACT Symposium on Theory of Computing, Rome, Italy, June 20 - 24, 2022, Stefano Leonardi and Anupam Gupta (Eds.). ACM, 1052–1065. https://doi.org/10.1145/ 3519935.3519977 Amir Azarmehr, Soheil Behnezhad, and Mohammad Roghani. 2024. Fully Dynamic Matching: -
Approximation in Polylog Update Time. In Proceedings of the 2024 ACM-SIAM Symposium on Discrete Algorithms, SODA 2024, Alexandria, VA, USA, January 7-10, 2024, David P. Woodruff (Ed.). SIAM, 3040–3061. https://doi.org/10.1137/1.

9781611977912.109 Pankaj K. Agarwal, Sharath Raghvendra, Pouyan Shirzadian, and Rachita Sowle. 2022b. An Improved ε-Approximation Algorithm for Geometric Bipartite Matching. In 18th Scandinavian Symposium and Workshops on Algorithm Theory, SWAT 2022, June 2729, 2022, Torshavn, Faroe Islands (LIPIcs, Vol. 227) ´ . Schloss Dagstuhl - Leibniz-Zentrum fur Informatik, ¨ 6:1–6:20. https://doi.org/10.4230/LIPICS. SWAT.2022.6 Yogesh Balaji, Rama Chellappa, and Soheil Feizi. 2020.

Robust Optimal Transport with Applications in Generative Modeling and Domain Adaptation. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Richard Bellman. 1958. On a routing problem. Quarterly of applied mathematics 16, 1 (1958), 87–90.

Pankaj K. Agarwal and Kasturi R. Varadarajan. 2004. A
near-linear constant-factor approximation for euclidean bipartite matching?. In Proceedings of the 20th ACM Symposium on Computational Geometry, Brooklyn, New York, USA, June 8-11, 2004, Jack Snoeyink and Jean- Daniel Boissonnat (Eds.). ACM, 247–252. https:// doi.org/10.1145/997817.997856 Gaspard Beugnot, Aude Genevay, Kristjan H. Greenewald, and Justin Solomon. 2021. Improving approximate optimal transport distances using quantization. In Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, UAI 2021, Virtual Event, 27-30 July 2021 (Proceedings of Machine Learning Research, Vol. 161). AUAI Press, 290–300. https://proceedings.mlr.press/ v161/beugnot21a.html Jason Altschuler, Jonathan Niles-Weed, and Philippe Rigollet. 2017. Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration. Advances in neural information processing systems 30 (2017).

Sergio Cabello, Panos Giannopoulos, Christian Knauer, and Gunter Rote. 2005. Matching Point Sets with Respect to ¨
the Earth Mover's Distance. In Algorithms - ESA 2005, 13th Annual European Symposium, Palma de Mallorca, Spain, October 3-6, 2005, Proceedings (Lecture Notes in Computer Science, Vol. 3669). Springer, 520–531.

https://doi.org/10.1007/11561071_47 David Alvarez-Melis and Nicolo Fusi. 2020. Geometric `
Dataset Distances via Optimal Transport. In *Advances* in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria- Florina Balcan, and Hsuan-Tien Lin (Eds.).

Sergio Cabello, Panos Giannopoulos, Christian Knauer, and Gunter Rote. 2008. Matching point sets with respect ¨ to the Earth Mover's Distance. *Comput. Geom.* 39, 2
(2008), 118–133. https://doi.org/10.1016/J.

COMGEO.2006.10.001 Alexandr Andoni, Aleksandar Nikolov, Krzysztof Onak, and Grigory Yaroslavtsev. 2014. Parallel algorithms for geometric graph problems. In Symposium on Theory of Computing, STOC 2014, New York, NY, USA, May 31 - June 03, 2014, David B. Shmoys (Ed.).

ACM, 574–583. https://doi.org/10.1145/
2591796.2591805 Jiezhang Cao, Langyuan Mo, Yifan Zhang, Kui Jia, Chunhua Shen, and Mingkui Tan. 2019. Multi-marginal Wasserstein GAN. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada. 1774–1784.

David S Atkinson and Pravin M. Vaidya. 1995. Using geometry to solve the transportation problem in the plane. Algorithmica 13, 5 (1995), 442–461. https://doi. org/10.1007/BF01190848 Kyriakos Axiotis, Aleksander Madry, and Adrian Vladu.

2020. Circulation control for faster minimum cost flow in Kevin C. Cheng, Shuchin Aeron, Michael C. Hughes, and Eric L. Miller. 2021. Dynamical Wasserstein Barycenters for Time-series Modeling. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (Eds.). 27991–28003.

Monika Henzinger, Sebastian Krinninger, Danupon Nanongkai, and Thatchaphol Saranurak. 2015. Unifying and Strengthening Hardness for Dynamic Problems via the Online Matrix-Vector Multiplication Conjecture. In Proceedings of the Forty-Seventh Annual ACM on Symposium on Theory of Computing, STOC 2015, Portland, OR, USA, June 14-17, 2015, Rocco A. Servedio and Ronitt Rubinfeld (Eds.). ACM, 21–30. https: //doi.org/10.1145/2746539.2746609 Søren Dahlgaard. 2016. On the Hardness of Partially Dynamic Graph Problems and Connections to Diameter.

In 43rd International Colloquium on Automata, Languages, and Programming, ICALP 2016, July 11-15, 2016, Rome, Italy (LIPIcs, Vol. 55), Ioannis Chatzigiannakis, Michael Mitzenmacher, Yuval Rabani, and Davide Sangiorgi (Eds.). Schloss Dagstuhl - Leibniz-Zentrum fur Informatik, 48:1–48:14. ¨ https://doi.org/10. 4230/LIPICS.ICALP.2016.48 Piotr Indyk. 2007. A near linear time constant factor approximation for Euclidean bichromatic matching (cost). In Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2007, New Orleans, Louisiana, USA, January 7-9, 2007. SIAM, 39– 42. http://dl.acm.org/citation.cfm?id= 1283383.1283388 Mark de Berg, Herman Haverkort, Shripad Thite, and Laura Toma. 2007. I/O-efficient map overlay and point location in low-density subdivisions. In International Symposium on Algorithms and Computation. Springer, 500–511.

Hicham Janati, Thomas Bazeille, Bertrand Thirion, Marco Cuturi, and Alexandre Gramfort. 2019. Group Level MEG/EEG Source Imaging via Optimal Transport: Minimum Wasserstein Estimates. In Information Processing in Medical Imaging - 26th International Conference, IPMI 2019, Hong Kong, China, June 2-7, 2019, Proceedings (Lecture Notes in Computer Science, Vol. 11492), Albert C. S. Chung, James C. Gee, Paul A. Yushkevich, and Siqi Bao (Eds.). Springer, 743–754. https://doi.org/ 10.1007/978-3-030-20351-1_58 Harold N Gabow and Robert E Tarjan. 1989. Faster scaling algorithms for network problems. *SIAM J. Comput.* 18, 5 (1989), 1013–1036.

Alfred Galichon. 2018. Optimal transport methods in economics. Princeton University Press.

Harold W Kuhn. 1955. The Hungarian method for the assignment problem. *Naval research logistics quarterly* 2, 1-2 (1955), 83–97.

Akshaykumar Gattani, Sharath Raghvendra, and Pouyan Shirzadian. 2023. A Robust Exact Algorithm for the Euclidean Bipartite Matching Problem. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 -
16, 2023.

Huidong Liu, Xianfeng Gu, and Dimitris Samaras. 2018.

A Two-Step Computation of the Exact GAN Wasserstein Distance. In *Proceedings of the 35th International* Conference on Machine Learning, ICML 2018, Stockholmsmassan, Stockholm, Sweden, July 10-15, 2018 (Pro- ¨ ceedings of Machine Learning Research, Vol. 80). PMLR,
3165–3174. http://proceedings.mlr.press/
v80/liu18d.html Alexandre Gramfort, Gabriel Peyre, and Marco Cuturi. ´
2015. Fast Optimal Transport Averaging of Neuroimaging Data. In Information Processing in Medical Imaging - 24th International Conference, IPMI 2015, Sabhal Mor Ostaig, Isle of Skye, UK, June 28 - July 3, 2015, Proceedings (Lecture Notes in Computer Science, Vol. 9123), Sebastien Ourselin, Daniel C. Alexander, ´ Carl-Fredrik Westin, and Manuel Jorge Cardoso (Eds.).

Springer, 261–272. https://doi.org/10.1007/
978-3-319-19992-4_20 Yongchao Liu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. 2017. Primal-dual hybrid gradient method for distributionally robust optimization problems. Oper. Res.

Lett. 45, 6 (2017), 625–630. https://doi.org/10.

1016/J.ORL.2017.10.001 James Munkres. 1957. Algorithms for the assignment and transportation problems. Journal of the society for industrial and applied mathematics 5, 1 (1957), 32–38.

Joachim Gudmundsson, Oliver Klein, Christian Knauer, and Michiel Smid. 2007. Small Manhattan networks and algorithmic applications for the earth mover's distance. In Proceedings of the 23rd European Workshop on Computational Geometry. Citeseer, 174–177.

Victor M. Panaretos and Yoav Zemel. 2019. Statistical Aspects of Wasserstein Distances. Annual Review of Statistics and Its Application 6, Volume 6, 2019 (2019), 405–431. https://doi.org/10.1146/
annurev-statistics-030718-104938 Sariel Har-Peled. 2011. Geometric approximation algorithms. Number 173. American Mathematical Soc.

Sharath Raghvendra and Pankaj K. Agarwal. 2020. A Nearlinear Time ε-Approximation Algorithm for Geometric Bipartite Matching. *J. ACM* 67, 3 (2020), 18:1–18:19. https://doi.org/10.1145/3393694 Yossi Rubner, Carlo Tomasi, and Leonidas J. Guibas.

2000. The Earth Mover's Distance as a Metric for Image Retrieval. *Int. J. Comput. Vis.* 40, 2 (2000), 99–121. https://doi.org/10.1023/A: 1026543900054 R. Sharathkumar. 2013. A sub-quadratic algorithm for bipartite matching of planar points with bounded integer coordinates. In Symposium on Computational Geometry 2013, SoCG '13, Rio de Janeiro, Brazil, June 17-20, 2013. ACM, 9–16. https://doi.org/10.1145/
2462356.2480283 R. Sharathkumar and Pankaj K. Agarwal. 2012a. Algorithms for the transportation problem in geometric settings. In Proceedings of the Twenty-Third Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2012, Kyoto, Japan, January 17-19, 2012. SIAM, 306–317. https://doi.org/10.1137/ 1.9781611973099.29 R. Sharathkumar and Pankaj K. Agarwal. 2012b. A nearlinear time ε-approximation algorithm for geometric bipartite matching. In Proceedings of the 44th Symposium on Theory of Computing Conference, STOC 2012, New York, NY, USA, May 19 - 22, 2012. ACM, 385– 394. https://doi.org/10.1145/2213977.

2214014 Ilya O. Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Scholkopf. 2018. Wasserstein Auto-Encoders. ¨ In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30
- May 3, 2018, Conference Track Proceedings. Open-
Review.net. https://openreview.net/forum? id=HkL7n1-0b Jan van den Brand, Yin Tat Lee, Danupon Nanongkai, Richard Peng, Thatchaphol Saranurak, Aaron Sidford, Zhao Song, and Di Wang. 2020. Bipartite Matching in Nearly-linear Time on Moderately Dense Graphs. In *61st* IEEE Annual Symposium on Foundations of Computer Science, FOCS 2020, Durham, NC, USA, November 1619, 2020. IEEE, 919–930. https://doi.org/10.

1109/FOCS46700.2020.00090 Xiaoyang Xu and Hu Ding. 2024a. A Novel Skip Orthogonal List for Dynamic Optimal Transport Problem. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 20838–20846.

Xiaoyang Xu and Hu Ding. 2024b. A Novel Skip Orthogonal List for Dynamic Optimal Transport Problem. In *Thirty-Eighth AAAI Conference on Artificial* Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, 20838–20846. https://doi.org/10.1609/
AAAI.V38I18.30073

## A. Related Works

Static Algorithms: The classical Hungarian algorithm (Kuhn, 1955) allows computing an optimal Euclidean bi-chromatic matching in O(n 3) time. (Atkinson and Vaidya, 1995) presented a Primal-Dual based scaling algorithm for points from R 
2that, supporting points with multiplicity at most W, runs in O(n 5/2log(n) log(W)) time. Near-quadratic algorithms for computing an optimal matching (of points with multiplicity) can be derived from recent advances for minimum cost max-flows in graphs (van den Brand et al., 2020).

Sub-quadratic algorithms are known for the following two important cases. In case d = 2 and all coordinates are integers ≤U, (Sharathkumar, 2013) presented an O(n 3/2+δlog(nU)), for any fixed δ ∈ (0, 1]. In case both *A, B* are i.i.d. samples
(from two distributions), (Gattani et al., 2023) recently presented an Oe(n 2− 1 2d log(U)Φ(n)), where Φ(n) = poly(log n) for d = 2, expected time algorithm, and showed its practicality for estimating the 1-Wasserstein distance of distributions on R
2 for n ≤ 50 000 samples in approximately 700 seconds. Static Approximations: (Agarwal and Varadarajan, 2004) presented a top-down algorithm that computes, with with high probability, an O(log 1ε
)-approximation in O(n 1+ε)-time, for any fixed ε ∈ (0, 1]. (Sharathkumar and Agarwal, 2012b),
(Raghvendra and Agarwal, 2020) obtained an O(n poly(log *n, ε*−1)) time algorithm that, using augmenting paths in a randomly shifted quad-tree, computes with high probability a (1 + ε)-approximate matching. The poly(log *n, ε*−1)-factor was further improved in (Agarwal et al., 2022b). Recently, (Agarwal et al., 2022a) presented the first deterministic
(1 + ε)-approximation algorithm with near-linear n( log n ε)
O(d)time.

Variants: Using spanners and fast minimum cost max-flow solvers, it is possible to compute a (1 + ε)-approximation of the *cost-value* in Oe(n 3/2) or in Oe(n 4/3+o(1)) time, also supporting point multiplicity (see (Gabow and Tarjan, 1989) and
(Axiotis et al., 2020)). See also e.g., (Cabello et al., 2005; 2008) and (Gudmundsson et al., 2007). (Indyk, 2007) showed that importance sampling can be used for obtaining, with high probability, an O(1)-approximation of the cost-value in O(n poly(log n)) time, which however also does not allow to report a matching. (Altschuler et al., 2017) presented an algorithm that also reports a matching, but with an additive error ε in time O˜(n 2/ε3). For the parallel and streaming model,
(Andoni et al., 2014) presented an n 1+oε(1) time algorithm for R
2that allows to compute an O(1)-approximation for the cost-value of an optimal matching. (Sharathkumar and Agarwal, 2012a) also presented a dynamic approximation algorithm that maintains a partial-matching. The more general problem of dynamic optimal transport has been considered in recent work (Xu and Ding, 2024b). Specifically, the goal is to design a data structure that accelerates executing one iteration of the Network Simplex Algorithm on an explicit complete bi-partite graph. Their work is concerned with maintaining an exact solution, within numeric computing accuracy, after the insertion/deletion of a vertex to the min-cost flow problem. Since their algorithm operates on a complete, bipartite graph, it requires Ω(n 2) space to store the edge weights and Ω(n) update time to insert a new vertex.

Moreover, in the worst case, the update time can take up to a quadratic time. As such, the algorithm and implementation cannot handle very large instances in practice.

## B. Formal Description Of The Implicit And Explicit Matching Sub-Routines

IMPLICIT M**ATCHING** Recall that the input of the procedure is a vertex v of p-tree T and the monochromatic set of points E := {Eu : u ∈ Cv} forwarded to v from its children and their cardinalities {|Eu| : u ∈ Cv}.

Choosing the points to match at v. Formally, first divide E into collections of red and blue monochromatic sets E1 := {Eu :
u ∈ Cv, Eu is red} and E2 := {Eu : u ∈ Cv, Eu is blue}, respectively.

Then, determine the total number of red and blue points in E1 and E2, respectively, by summing up the provided set cardinalities in each collection. Suppose that there are k1 red points in E1, k2 blue points in E2, and that without loss of generality k1 ≤ k2. In this case, we aim to match all red points from sets in E1. To pick k1 points from blue sets in E2, we loop over Cv in a fixed order (e.g. z-order w.r.t. the underlying p-tree) and pick all elements Eu ∈ E2 until one child u provides sufficiently many points to satisfy stopping (i.e. picking all |Eu| points would lead to a set of blue points of size at least k).

From this child, the selection implicitly chooses the remaining set E′u of k
′ ≤ k1 points with the smallest z-order. Define set Ev := Eu \ E′u of points that will not be matched at v, which can be represented by an integer k and a pointer to child u a pointer to the (k
′ + 1)-th element in set Eu. Finally, denote the set of chosen red and blue points by Yv(A) and Yv(B),
respectively, and let Yv = Yv(A) ∪ Yv(B). Note that (implicitly) choosing these subsets at a node v takes O(p 2) time, though reporting the points in Yv(A), or in Yv(B), takes O(k · d) time, where d = O(logp U) is the depth of the p-tree.

Obtaining the aggregated sub-instance. For any cell u ∈ Cv, define zu to be the *center* point of u. For x ∈ Yv, let u(x) be the subcell of Cv that x is contained in. We define a mapping m(x) from x to zu(x), to be the function that maps a point x to the center of its (bounding) subcell.

For any point set S, let m(S) := {m(x) | x ∈ S}. We call m(Yv) a *aggregated* sub-instance w.r.t. to node v and the corresponding point set Yv. Note that the function m(Yv) can be implicitly represented by the p 2sub-grid cells of the cell of v and their corresponding points in Yv.

Computing the implicit matching: We solve the following Euclidean transport instance on the points of m(Yv(A)), m(Yv(B)): For every point a
′ ∈ m(Yv(A)), we set the supply of a
′to be the number of points in Yv(A)
that map to a
′, i.e., sa′ := |{a ∈ Yv(A) : m(a) = a
′}|. Similarly, for every point b
′ ∈ m(Yv(B)), we set the demand of b
′
to be the negative number of points in Yv(B) that map to b
′, i.e., db
′ := −|{b ∈ Yv(B) : m(b) = b
′}|. Let γ
∗v[Yv] be an optimal assignment (or implicit matching) obtained by solving the Euclidean transportation problem on the sets m(Yv(A)) and m(Yv(B)) using the algorithm of Theorem 2.1.

EXPLICIT M**ATCHING** The inputs to the explicit matching procedure are a vertex of the restricted p-tree v, the point sets Yv (partitioned by their corresponding cells in u ∈ Cv and an optimal assignment γ
∗
v[Yv] to the Euclidean transport instance on (m(Yv(A)), m(Yv(B))*, s, d*) described above. Its output is a matching of points Yv.

The procedure iterates through the edges (a
′, b′) ∈ m(Yv(A)) × m(Yv(B)) with non-zero assignment in γ
∗
v[Yv] in an arbitrary manner and adds γ
∗
v[Yv](a
′, b′) arbitrary pairs of points *a, b*, where a ∈ Yv(A) and b ∈ Yv(B), and m(a) = a
′and m(b) = b
′, as edges to the output matching from the cells with centers a
′, b′.

## C. Proof Of Theorem 3.3

Theorem 3.3 (Static). For any Euclidean matching instance (A, B), with |A| = |B| = n*, and spread* U ≤ n c, for some constant c > 0*, and for any constant* ε ∈ (0, 1 10 )*, algorithm* STATIC-MATCHING(A, B) computes an O(1/ε)-approximate Euclidean matching with high probability in O(n 1+ε· ε
−1) time.

Proof of Theorem 3.3. For this proof, we will assume that both n and p are powers of 2. While a similar approach extends to the general case, we present the special case here for the sake of clarity. We show that p = n ε/10 yields the runtime and approximation bounds. Run-time analysis: By definition each leaf of p-tree T contains at most p 2 many points. Now, since, we can compute the exact optimal matching between the points Yv for leaf node v in O(|Yv| 3) time using the Hungarian algorithm, the total runtime of computing exact matchings of all leaves is bounded by O(p 4· n).

Next, we discuss the running time of step 1 in IMPLICIT-MATCHING(v, Yv). By Lemma 3.1, for each non-leaf node v, the IMPLICIT-MATCHING in the execution of MATCH(v) can be implemented in O(p 2 + p 5· log2n) = O(p 5· log2n) time.

In addition, we need to solve IMPLICIT-MATCHING for at most n nodes in the p-tree T. The total run-time of solving the IMPLICIT-MATCHING at each node is bounded by O(n · p 5· log2n).

For any node v, the total time to report the points in excess set Ev is bounded by O(|Ev|). Thus, Lemma 3.2 yields the perfect matching on set X takes O(n) time. Hence, the runtime of STATIC-MATCHING(*A, B*) is at most O
n + p 4· n + n · p 5· log2n + n ·
log U
log p
, since the number of levels in T is bounded by log U
log p
. Setting p = n ε/10, and using that U ≤ n cfor a fixed c > 0, we obtain the stated run-time bound.

Note that while the upper bound we present of the approximation ratio holds in expectation, in the static setting we may repeat the algorithm O(log n) times and output the smallest cost solution to find an O(1/ε)-approximate solution with high probability. In the dynamic setting, this simple trick guaranteeing a high-probability approximation ratio does not work.

Correctness analysis: Let µ
∗ be a matching on set X with optimal cost c(µ
∗). Say that an edge is crossing at level i if it crosses the grid of level i of T. Observe that an edge may be crossing at multiple levels. Let **Cross**(*M, i*) stand for the number of crossing edges of matching M at level i.

Note that if none of the edges of µ
∗is cut by a grid at level ℓ, this implies that all nodes at level ℓ have a balanced number of blue and red points. Also note that no edge of µ
∗is crossing at level 0, as we assume input points are contained in the cell D × D for some D = n c(both in the static and dynamic settings).

Let µa be the matching computed with our algorithm w.r.t. T. Our goal is to prove that

$$\mathbb{E}[c(\mu_{a})]=O(\varepsilon^{-1}\cdot c(\mu^{*}))\quad,$$
$$(1)$$
∗)) , (1)
where the expectation is taken with respect to the random shift introduced at initialization. Let Tℓ stand for the vertices of T
at level ℓ. Let λℓ = D/pℓstand for the length of the sides of the grid at level ℓ. Slightly overloading notation let λv stand for the length of side of the cell of v.

Matching µa has the following convenient property: every point of Yv is matched with a point of Yv (inside the cell of v) under µa due to the construction of our implicit-explicit matching procedure. Unfortunately, µ
∗ might not have this convenient property, making the comparison between them difficult. We now describe a procedure which translates µ
∗into a matching µ
′ with this property. Later, we compare the costs of µ
′, µ∗and µa.

Generating µ
′**from** µ
∗ We describe an iterative method which generates µ
′from µ
∗. Initially, we set µ
′ = µ
∗. Assume that the tree T has k levels. Iterating ℓ from k to 0 for every v ∈ Tℓ, we keep repeating the following as long as possible:
If there exists edges of µ
′, denoted (r1, b1),(r2, b2), such that r1, b2 ∈ Yv for some v ∈ Tℓ, r1, b2 are of opposite color and b1, r2 ∈/ Yv, then replace (r1, b1),(r2, b2) with (r1, b2),(r2, b1) in µ
′.

Let µ
′ℓ for ℓ ∈ {*k, . . . ,* 0} stand for the state of µ
′after not being able to repeat this process at level ℓ + 1. Specifically, µ
∗ = µ
′k
.

Claim C.1. The above process terminates in n iterations and for any ℓ it must hold that for all v ∈ T>ℓ the points of Yv are matched exclusively to each other by µℓ. Proof. Observe that before the execution of any of the two steps none of the two edges run between points of Yw for any w ∈ T. However, after the execution we have (r1, b2) ∈ µ
′and r1, b2 ∈ Yv for some v ∈ T. Hence, the process must terminate in n steps.

Also, note that any Yv contains the same number of red and blue points. If, without loss of generality, any red point of some Yv is matched outside of Yv by µ
′, then there must also exist a blue point of Yv matched outside of Yv and the process can take a step. Hence, once the process can't make any further steps for some Yv where r1, b2 ∈ Yv, then µ
′is matching the points of Yv to each other.

Claim C.2. *Cross*(µ
′
ℓ, ℓ′) ≤ *Cross*(µ
∗, ℓ′) *for all* ℓ
′ < ℓ.

Proof. Assume that the process is progressing through vertices of level ℓ. We will argue that any step it may take will not increase the number of crossing edges at lower levels, implying the claim. If the process takes a step for some v ∈ Tℓ then it adds edges (r1, b2),(r2, b1) to µ
′. (r1, b2) is in the same cell at level ℓ hence it is not crossing at lover levels. (r2, b1) might be crossing at some level ℓ
′ < ℓ. However, in this case as (r1, b2) are in the same cell of level ℓ
′it must hold that in this case either (r1, b1) or (r2, b2) was crossing at level ℓ
′ before their removal from µ
′.

We will first relate the costs of matchings µ
∗and µ
′.

Claim C.3. c(µ
′) ≤ c(µ
∗) + Pk ℓ=0 8 · λℓ · *Cross*(µ
∗, ℓ).

Proof. Consider the state of µ
′after having finished with vertices of level ℓ + 1, that is µ
′
ℓ. Call a point bad at this point if it belongs to some Yv : v ∈ Tℓ and is matched to some point outside of Yv by µ
′
ℓ. In this case there are two options: it is either matched outside of its cell at level ℓ, or it is matched to a point forwarded by node v to its ancestor but still inside Cv. Note that it can't be matched to any point of Yw for a node on higher level then v due to Claim C.1. The set of points forwarded by v to its parent is monochromatic. Hence, for any bad point of the later type we will be able to associate a bad point of the earlier type. This implies, that at least half of the bad points, denote their number by Bℓ, are incident on edges crossing at level ℓ in µ
′ℓ, that is µ
′ℓhas at least Bℓ/2 crossing edges at level ℓ. Observe that, due to Claim C.2, this implies that µ
∗also has at least Bℓ/2 crossing edges at level ℓ. Once the process gets to level ℓ in every step, it will reduce the number of bad edges by 1. Furthermore, we claim that in each step it may increase the cost of µ
′ by at most 4 · λℓ. By every step some (r1, b2),(r2, b1) are added to µ
′and some (r1, b1),(r2, b2) are removed.

We have that c(r1, b2) ≤ 2 · λℓ as r1 and b2 are contained in the same cell at level ℓ. By triangle inequality, we have c(r2, b1) ≤ c(b1, r1) + c(r1, b2) + c(b2, r2) and therefore c(r2, b1) + c(r1, b2) − c(b1, r1) − c(b2, r2) ≤ 2 · c(r1, b2). Hence the total increase is indeed at most 4 · λℓ. That is, the process will take at most Bℓ steps, in each increasing µ
′ by at most 4 · λℓ, where Bℓ ≤ **Cross**(µ
∗, ℓ)/2. Summing over all levels proves the claim.

We now relate the costs of matchings µ
′and µa. Let M[Y ] stand for the edges of matching M restricted to points Y . By construction of the algorithm and due to Claim C.1, we know that both of µ
′and µa match the points of Yv to each other for any v ∈ T. First, observe that if v is a leaf then our algorithm calculated an optimal minimum cost matching of Yv, that is c(µ
′[Yv]) ≤ c(µa[Yv]) for all leafs v ∈ V .

$\mathbf{a}\in\mathbf{R}$
Claim C.4. c(µa[Yv]) ≤ c(µ
′[Yv]) + 2 · λv/p · |Yv| *for all non-leaf* v ∈ T.

Proof. Recall that the algorithm computes µa[Yv] by aggregating all points of Yv to the respective center of their sub-cell in Cv and computing an optimal solution of the resulting Euclidean geometric transport instance, then moving the points back to their original location. That is, assuming the points were already in the center of their respective cells, µa[Yv] is a minimum cost matching of Yv. Moving a point to the center of it's respective sub-cell increases the cost of an optimal solution by at most λv/p (the length of the sides of the sub-cells of v), similarly the cost of moving them back to their original position is also upper bounded by λv/p.

$$\operatorname*{erf}|Y_{v}|\leq2\cdot\operatorname{Cross}(\mu^{*},\ell+1)$$

Claim C.5. Pv∈Tℓ|v is non-leaf |Yv| ≤ 2 · *Cross*(µ
∗, ℓ + 1) for ℓ ∈ {0 *. . . k* − 1}.

Proof. Consider a particular vertex v ∈ Tℓ. The points matched at v (the points of Yv) are all inherited by v by its children nodes due to an imbalance of red and blue points in the respective cells of v-s children nodes. This implies that µ
∗ matched at least |Yv| points in Cv with points in a separate sub-cell of layer ℓ + 1. Hence, |Yv| points of Cv are incident on crossing edges at level ℓ + 1 in µ
∗. Summing over all vertices of {Yv|v ∈ Tℓ} proves the claim.

We now have to tie the number of crossing edges to the cost of µ
∗.

Claim C.6. E[**Cross**(µ
∗, ℓ)]/λℓ = O(c(µ
∗)) for all ℓ ∈ {0 . . . k} where the expectation is taken with respect to the random shift introduced at initialization.

Proof. Consider an edge e ∈ µ
∗ of length L. With probability Ω(L/λℓ) it crosses the grid at level ℓ due to the random shift.

Hence, it contributes L to both sides of the inequality.

We are now ready to upper bound the approximation ratio of the algorithm in expectation with respect to the random shift at initialization.

E[c(µa)] = X v∈T|v is leaf E[c(µa[Yv])] + X v∈T|v is non-leaf E[c(µa[Yv])] ≤ E[c(µ1)] + X v∈T|v is non-leaf 2 · E[|Yv|] · λv/p (2) ≤ c(µ ∗) +X k ℓ=0 8 · λℓ · E[Cross(µ ∗, ℓ)] + X v∈T|v is non-leaf 2 · E[|Yv|] · λv/p (3) ℓ=0 8 · λℓ · E[Cross(µ ∗, ℓ)] + k X −1 ≤ c(µ ∗) +X k ℓ=0 4 · λℓ/p · E[Cross(µ ∗, ℓ + 1)] (4) ≤ O(c(µ ∗) · k) (5) ≤ O(c(µ ∗)/ε)
$$(2)$$
$$(3)$$
$${\mathrm{(4)}}$$
$$(S)$$
$\square$
The first inequality follows from the fact that µa matches vertices of Yv exclusively with each other. Equation 2 holds due to Claim C.4. Equation 3 follows from Claim C.3, while equation 4 follows from Claim C.5. Equation 5 follows from Claim C.6. Finally, the last inequality holds as the number of layers in the p-tree T is upper bounded by O*(log* U/ log P) = O(1/ε).

## D. Basic Algorithm

In this section we first present the pseudo-code of our basic algorithm which can answer queries about the solution efficiently (see Theorem 4.1). Afterwards we argue about it's running time and approximation ratio.

## D.1. Running Time Analysis

We will first argue about the update time of the algorithm. We will analyze the update time of an insertion as the process of handling of deletions is similar. An update is handled in a top-to-bottom manner: the algorithm first calls the update procedure at the root and then always recuses towards the child which contains the newly inserted point. Hence, throughout an update the algorithm will visit nodes of T along a path form the root to some leaf. Apart from the visited leaf node the only non-constant time operation executed at any node is the implicit-matching procedure which runs a geometric Euclidean transport sub-routine on an aggregated sub-instance. These aggregated sub-instances consist of points corresponding to the centers of the sub-cells of the respective node, that is p 2 points. The weights assigned to each center are at most n. Hence, by Theorem 2.1 they run in time O(p 5· log(p log n)) =
O(n O(ε)). That is, updating the non-leaf nodes along the path takes O(n O(ε)/ε) worst case update time, as T will always have at most O(1/ε)-levels. The leaf node might end up having more then p 2input points in its cell after the insertion (otherwise we simply call the Hungarian algorithm on it). This might result in the leaf splitting into a new sub-tree. This sub-tree can similarly only have O(1/ε)-levels and in total at most O(p 2/ε) nodes. On each of its non-leaf nodes similarly the only non-constant time operation will be the implicit matching procedure adding O(n O(ε)/ε) worst-case update time.

There might be ∼ p 2leaves of the new sub-tree, but the number of points inside the cells of the new leaves is at most O(p 2).

In each such leaf the only non-constant time calculation the algorithm completes is a call to the Hungarian algorithm to obtain an optimal solution inside the leaf, hence this contributes at most O(p 6 = n O(ε)) to the update time.

Additionally, we can maintain the p-tree T itself in O(p 2/ε) = O(n O(ε)/ε) worst-case update time. Hence, the total worst-case update time of the algorithm is O(n O(ε)/ε).

## D.2. Algorithm Correctness

The analysis of the approximation ratio of the static algorithm presented in Section C relies on the following properties of the output:
Algorithm 1 INSERT-UPDATE(*v, a*)
if v is non-marginal internal node of T **then**
u ← child u ∈ Cv such that a ∈ Xu Update Eu and c(µu) via INSERT-UPDATE(*u, a*)
Update γ
∗v[Yv] and Ev via IMPLICIT-MATCHING(v, E, {|Eu| : u ∈ Cv})
c(µv) ←Pu∈Cv c(µu) + c(γ
∗
v[Yv])
end if if v is non-marginal leaf in T **then**
Y
′
v
(A) ← Y
′v
(A) ∪ {a}
if η ≥ 0 **then**
Ev ← Ev ∪ {a}
η ← η + 1 Yv(A) ← Yv(A) \ Ev else b ← any point from Ev Ev ← Ev \ {b}
η ← η − 1 Yv(B) ← Yv(B) \ Ev end if µv ← HUNGARIAN-ALGORITHM(Yv(A), Yv(B))
end if if v is a new subroot of T **then**
if η ≥ 0 **then**
Ev ← Ev ∪ {a}
η ← η + 1 else b ← any point from Ev Ev ← Ev \ {b}
η ← η − 1 end if Update γ
∗v
[Yv] and c(µv) via MATCH(v)
c(µv) ←Pu∈Cv c(µu) + c(γ
∗
v
[Yv])
end if Algorithm 2 INSERT(a)
Update restricted p-tree T
INSERT-UPDATE(*r, a*)
Algorithm 3 DELETE-UPDATE(*v, a*)
if v is an internal node of T **then**
u ← child u ∈ Tv such that a ∈ Xu Update Eu and c(µu) via INSERT-UPDATE(*u, a*)
Update γ
∗v[Yv] and Ev via IMPLICIT-MATCHING(v, E, {|Eu| : u ∈ Cv})
c(µv) ←Pu∈Cv c(µu) + c(γ
∗
v[Yv])
end if if v is a leaf in T **then**
Y
′
v(A) ← Y
′v(A) \ {a}
if a ∈ Ev **then**
Ev ← Ev \ {a}
else if η > 0 **then**
b ← any point from Ev Ev ← Ev \ {b} η ← η − 1 Yv(B) ← Yv(B) \ Ev else Ev ← Ev ∪ {a}
η ← η + 1 Yv(A) ← Yv(A) \ Ev end if µv ← HUNGARIAN-ALGORITHM(Yv(A), Yv(B))
end if end if Algorithm 4 DELETE(a)
Update restricted p-tree T
DELETE-UPDATE(*r, a*)
(1) For any node v ∈ T points of Yv (which all lie in the cell of v) are matched exclusively to each other. (2) For any leaf node v ∈ T the points matched in the cell of v, Yv, are matched via an optimal Euclidean bi-chromatic matching algorithm (Hungarian algorithm).

(3) For any internal node v ∈ T the points matched in the cell of v, Yv are matched via the implicit-explicit matching procedures relying on an underlying optimal Euclidean geometric transport sub-routine (Theorem 2.1).

(4) For any node v ∈ V the algorithm matches all but a monochromatic set of points out of the sets of points forwarded to v by its children inside the cell of v (or in the case of leaves among all the points in the corresponding cell Cv).

We will argue that once an update is processed by the dynamic algorithm, all of these properties of the output are restored. The only vertices of T where any of the mentioned properties might be violated are the ones whose cell contains the updated point or ones which have been created due to the update to the p-tree. Leaves, which have contained the updated point or are created due to the update, all select an arbitrary maximal color balanced set of points in their respective cell. The algorithm runs the Hungarian algorithm on these sets restoring item (2). Afterwards, leaves pass on a monochromatic set of points to their parent internal nodes. Every internal node runs the implicit-explicit matching procedure on the points passed to it by its children in a bottom-up order. This implies they will satisfy item (3) once the update has been processed. The implicit-explicit matching procedure matches all but a monochromatic set of points in a cell, which set is then forwarded to the parent node. This implies that by the time the root has finished its internal calculations, both items (4) and (1) are satisfied.

Note that we may guarantee that the approximation ratio holds with high probability if we simply maintain O(log n) solutions in parallel and always query the one with smallest cost. However, in this case in order to maintain the approximation ratio with high probability over super-polynomially long update sequences the data-structure has to be periodically rebuilt reducing the update time guarantee to be amortized.

## E. Advanced Algorithm

In this section, we describe an improvement of the algorithm described in Section 4. The main difference is that in the advanced algorithm, we maintain an explicit matching at each node. As shown previously, we may recompute the explicit matching at every node efficiently after an update through a black-box Euclidean geometric transport algorithm. However, without opening the black-box, we can't guarantee that the implicit matching computed will not differ significantly from its previous version. This means that in order to maintain a matching of the actual input points we would have to re-run the explicit matching procedure, but that takes time proportional to the number of points matched at the specific node which could be Ω(n). To overcome this difficulty, we show that we can update the implicit and explicit matchings stored at each node v ∈ T in time O(p 3) when two points are added to the respective point set Yv. We will call this procedure AUGMENT-MATCHING.

We then show that the size of the excess set of each node of T changes by exactly 1 after each update. We finally present an algorithm which maintains the explicit matchings stored at each node efficiently.

## E.1. Updating Implicit/Explicit Matchings Under Insertions

Let µ be a matching of some red and blue point sets (A, B) and *x, y* be some arbitrary points of opposite color not in *A, B*. Define an augmenting path with respect to µ to be a path between *x, y* consisting of edges between *x, y, A, B*, alternating between edges not in µ
∗and in µ
∗. Define the cost of such a path to be the sum of the costs of its edges not in µ minus the cost of its edges in µ.

For such an augmenting path Π, define the matching µ
′consisting of all of the edges of µ not in Π and the edges of Π not in µ to the matching µ augmented by Π. Observe that c(µ
′) = c(µ) + c(Π) and c(µ
′) is a perfect matching of A ∪ {x}, B ∪ {y}.

Claim E.1. Let A, B *be a set of input points for the bi-chromatic Euclidean matching problem and* µ
∗ be an optimal matching of points A, B. Let x, y be some arbitrary points in the plane of opposite colors and let Π∗*the smallest cost* augmenting path with respect to µ
∗starting and finishing at x and y*, respectively. Then the matching obtained by augmenting* µ
∗ by Π∗is a minimum cost perfect matching of A ∪ {x}, B ∪ {y}.

Proof. Define (µ
∗)
′to be the matching obtained by augmenting µ
∗ with Π∗. Assume that it's not of minimum cost and let µ
∗
E stand for a min cost matching of the extended point set A ∪ {x}, B ∪ {y}. Look at the union of µ
∗and µ
∗
E. It consists of disjoint edges, cycles and a single augmenting path Π∗ with respect to µ
∗ between x and y. Observe, that the cost of matching all points not part of Π′is the same under both matchings due to their optimality in their respective point sets, hence c(µ
∗
E) = c(µ
∗) + c(Π′) . However, as c((µ
∗)
′) = c(µ
∗) + c(Π∗) this would imply that c(Π∗) > c(Π′) which is a contradiction. Now extend the problem bi-chromatic Euclidean matching to point sets where certain points might have the same coordinates. It can be thought as extending the problem to inputs where input points have multiplicities. Claim E.2. Let A, B *a set of red and blue points where some pairs of points might share locations and* µ
∗ *be a min cost* bi-chromatic Euclidean matching between them. Assume that points A, B are located on k disjoint points of the plane, and points of A ∪ B at the same location are monochromatic. For arbitrary red and blue points x, y, there exists a min cost augmenting path between x and y *with respect to* µ
∗ which visits each of the k disjoint locations at most once.

Proof. For sake of contradiction assume that there is no such min cost augment path and let π
∗ be a minimum length
(in terms of number of edges) min cost augmenting path between x and y with respect to µ
∗. We will show that we can
'shortcut' π
∗.

Assume that Π∗ visits one of the k disjoint locations a twice. Then it must be the case that the Π∗contains a sub-path C
which is a cycle starting and ending with a consisting of µ
∗and non-µ
∗edges. There are three possible cases.

- If the total cost of non-µ
∗edges of C is the same as that of the µ
∗edges, then we may remove this cycle meaning that Π∗is not of minimum length.