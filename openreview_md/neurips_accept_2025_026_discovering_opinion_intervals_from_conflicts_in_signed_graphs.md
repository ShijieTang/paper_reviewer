# Discovering Opinion Intervals From Conflicts In Signed Graphs

| Peter Blohm∗ Aalto University Espoo, Finland   |
|------------------------------------------------|
| peter.blohm@aalto.fi                           |

| Aristides Gionis                                                                   |
|------------------------------------------------------------------------------------|
| KTH Royal Institute of Technology Digital Futures Stockholm, Sweden argioni@kth.se |

## Abstract

| Stefan Neumann TU Wien Vienna, Austria   |
|------------------------------------------|
| stefan.neumann@tuwien.ac.at              |

| Florian Chen∗                   |
|---------------------------------|
| University of Oxford Oxford, UK |
| florian.chen@cs.ox.ac.uk        |

Online social media provide a platform for people to discuss current events and exchange opinions with their peers. While interactions are predominantly positive, in recent years, there has been a lot of research to understand the conflicts in social networks and how they are based on different views and opinions. In this paper, we ask whether the conflicts in a network reveal a small and interpretable set of prevalent opinion ranges that explain the users' interactions. More precisely, we consider signed graphs, where the edge signs indicate positive and negative interactions of node pairs, and our goal is to infer opinion intervals that are consistent with the edge signs. We introduce an optimization problem that models this question, and we give strong hardness results and a polynomial-time approximation scheme by utilizing connections to interval graphs and the CORRELATION CLUSTERING problem. We further provide scalable heuristics and show that in experiments they yield more expressive solutions than CORRELATION CLUSTERING baselines. We also present a case study on a novel real-world dataset from the German parliament, showing that our algorithms can recover the political leaning of German parties based on co-voting behavior.

## 1 Introduction

Online social networks are essential parts of modern societies and are used by billions of people to discuss current events. Even though a majority of the interactions on such networks are positive, there are a substantial number of conflicts, particularly due to tensions among people with differing viewpoints [48, 47]. As a result, gaining a deeper understanding of these conflicts has become essential. This question is often studied using *signed graphs* [29, 48, 47], where each edge has a sign that is either positive (+)
if two nodes interact amicably, or negative (−) if the interaction is conflicting. A classic formulation used to analyze signed graphs and gain insights about the graph structure and potentially the opinions of nodes is the CORRELATION CLUSTERING problem [4]. In CORRELATION CLUSTERING, we ask to partition the nodes of a given signed graph into clusters, so as to maximize the number of edges that are consistent with the clustering (or minimize the number of inconsistent edges).

1 2 3 4 5 6 7 8 LINKE (Left) GRÜNE (Green) SPD (Social Democrat)
FDP (Liberal) CDU/CSU (Conservative) AfD (Far Right)
One drawback of the CORRELATION CLUSTERING formulation is that it makes hard decisions for the assignment of nodes into clusters and does not allow for a nuanced model in the presence of complex node interactions. For instance, in the landscape of European political parties, the opinions of representatives typically do not align perfectly with party lines; instead, members of ideologically-neighboring parties may agree on certain issues, while at the same time, members of the same party may disagree on other issues. Similar observations have been made for the US House of Representatives [2], however, when more parties are involved, modeling the interactions between representatives becomes increasingly complex. In this paper, we introduce a novel problem to analyze signed graphs and discover structure that explains the nodes' interactions (conflicts and agreements) more accurately. Instead of assigning nodes to disjoint clusters, we seek to assign nodes to a small number of potentially-overlapping opinion intervals. The resulting structure can lead to meaningful insights and intuitive visualization
(e.g., see Figure 1). We show that our problem is more expressive than CORRELATION CLUSTERING, thus resolving the drawback mentioned above. At the same time, our problem only requires the edge signs in the network as input, making it widely applicable. Our results. First, we introduce the BEST INTERVAL APPROXIMATION problem: Given a signed graph G = (*V, E*+ ∪ E−), assign an interval Iv ⊂ R to every vertex v ∈ V such that we maximize the number of edges {u, v} ∈ E+ with Iu ∩ Iv ̸= ∅ and {u, v} ∈ E− with Iu ∩ Iv = ∅. In other words, if two nodes are connected by a positive edge, then their corresponding intervals should overlap, whereas if they are connected by a negative edge, then their intervals should be disjoint. Note that for a node v, we can think of Iv as the range of opinions that are acceptable to v and yield an amicable interaction; all opinions outside of Iv are not acceptable and yield a conflict. This problem is more expressive than the CORRELATION CLUSTERING problem of Bansal et al. [4], and a related problem by Kermarrec and Thraves [34] as we explain below.

Second, we show that BEST INTERVAL APPROXIMATION is NP-hard *even when the graph* G+ =
(V, E+) *induced by the positive edges forms a cycle*. In a sense, this is the strongest possible hardness result one could hope for since removing a single edge from the cycle G+ = (*V, E*+) produces a path, for which intervals can always be assigned without any error. This implies that (unless P = NP) for BEST INTERVAL APPROXIMATION there is no FPT algorithm that parameterizes by the number of required edge removals and that the disagreement version of BEST INTERVAL APPROXIMATION cannot be approximated within any multiplicative factor. It also provides novel insights into the hardness of finding forbidden induced subgraphs, which rules out several algorithm design approaches. Our reduction is based on a result of Cygan et al. [22], but making it work for cycles requires several new ideas and gets significantly more complicated. We provide an overview of the reduction in Section 2.1. Third, we consider a constrained version of BEST INTERVAL APPROXIMATION, where we are given a complete signed graph and a parameter ε > 0. Now, we are only allowed to use k distinct intervals and each node must be assigned to one of them. This provides highly interpretable insights since the number of intervals is small. For this problem, we provide a polynomial-time approximation scheme (PTAS); specifically, we present an algorithm that computes a (1 + ε)-approximation in time 2 O(k 2log(k/(εδ))/ε3)· n. This generalizes an algorithm by Giotis and Guruswami [28] that was developed for CORRELATION CLUSTERING with a fixed number of clusters. We provide an overview of the PTAS in Section 2.2.

Fourth, from a practical point of view, we introduce heuristics that we describe in Section 3. Our heuristics are inspired by the PTAS above and include several practical improvements. Our experiments find that BEST INTERVAL APPROXIMATION is substantially more expressive than CORRE- LATION CLUSTERING and that our heuristic algorithms succeed in exploiting this expressivity. On 8 real-world datasets, our methods find overlapping opinion interval assignments that represent the data with 38% fewer disagreements on average compared to CORRELATION CLUSTERING solutions found by state-of-the-art methods. This holds even when we use only 8 intervals, showing that already a small number of intervals yields expressive and interpretable representations. Furthermore, we perform a case study on a novel dataset based on co-voting behavior in the German parliament, which we make publicly available. The output of our algorithm allows us to reconstruct the leaning of the political parties, as we demonstrate in Figure 1. Besides accurately reflecting the German political spectrum, the figure also reveals the coalition governments throughout the past decade (see Section 4). We stress that, due to the overlapping spectrum from the left to the right, finding such a structure would not be possible in existing problems like CORRELATION CLUSTERING. We conclude the paper with several interesting questions for further research in Section 5, and present our proofs and additional experimental results in the supplementary material.

Related work. The BEST INTERVAL APPROXIMATION problem is closely related to the SITTING ARRANGEMENT problem by Kermarrec and Thraves [34]: Given a signed graph G = (*V, E*+ ∪ E−),
can we assign a vector xu ∈ R
ℓto each u ∈ V such that for all positive edges {u, v} ∈ E+
and negative edges {u, w} ∈ E− the inequality ∥xu − xv∥2 < ∥xu − xw∥2 holds? Kermarrec and Thraves [34] presented several results for the case of ℓ = 1, i.e., embedding G into the real line. Cygan et al. [22] improved upon this and showed that for a complete signed graph G such an assignment exists if and only if the subgraph induced by its positive edges G+ = (*V, E*+) is a unit interval graph [59]. Given this characterization, we note that our problem is more expressive since we allow general (non-unit) intervals. Besides these theoretical insights, Pardo et al. [55, 56] provided heuristics for an optimization version that aims to minimize the number of violated constraints on the vectors xu above. However, this objective is substantially different from ours and thus incomparable.

As mentioned before, CORRELATION CLUSTERING [4] is highly related to our work and is stated as follows: Given a signed graph G = (*V, E*+ ∪ E−), partition its vertices into disjoint clusters C1*, . . . , C*k ⊆ V such that the number of positive edges within the clusters Ci and the number of negative edges between different clusters Ci and Cj , i ̸= j, is maximized. Here, the value of k can be picked by the algorithm. CORRELATION CLUSTERING has received a lot of attention in the past two decades in social network analysis and image segmentation, spanning approximation algorithms [13, 60, 28, 20, 18, 17], more expressive formulations [7], and results in dynamic, online, parallel, and streaming settings [16, 15, 19, 3, 41, 51, 43]. There has also been continued interest in developing heuristics [50, 1, 61, 66, 42, 6, 9]. We refer to the book by Bonchi et al. [8] for more references. Another closely related problem is that of (Unit) Interval Editing. Specifically, in (Unit) Interval Editing, the task is to transform an unsigned graph into a (unit) interval graph using a minimum number of edge deletions and insertions. This problem is known to be NP-hard already since the seminal work of Garey and Johnson [27] and it is fixed-parameter tractable (FPT) when parameterized by the number of edge insertions and deletions [11, 33, 64, 12]. We further discuss the relation of BEST INTERVAL APPROXIMATION to these problems in Section 2. Opinion formation models, such as the DeGroot model [24], the Friedkin–Johnsen model [26], or the bounded-confidence model [38, 23], are also related. These models have recently received a significant amount of attention in computer science and machine learning [53, 65, 67, 62] and assign a real-valued opinion to each node in a graph, which allows a more fine-grained understanding of conflicts than CORRELATION CLUSTERING. However, estimating the parameters of such models is highly challenging and requires more information than the edge signs of a signed graph [5, 45, 44]. Thus, our method is more easily applicable as it requires substantially less (and particularly less sensitive) data. In relation to our case study on the German parliament, the (DW)-NOMINATE algorithm [57, 58, 52]
also predicts ideological positions of legislators based on co-voting data. It models legislators and roll-call votes as a signed bipartite graph and applies maximum-likelihood estimation to infer each legislator's ideological location in a low-dimensional Euclidean space, together with a Gaussian utility function centered at that point. However, since (DW)-NOMINATE operates on a bipartite graph, it is not applicable in more general social network settings where the input is given as a unipartite graph. This is in contrast to our methods, which only require a unipartite signed graph as input. Several related works consider versions of the CORRELATION CLUSTERING objective for partitioning signed graphs to reveal community structures [21, 40, 14, 54, 63]. However, similar to CORRELATION CLUSTERING, these methods do not allow finding overlapping communities. Thus, they cannot explicitly consider individual tolerance of other opinions in a way comparable to opinion intervals. Further, Dwork et al. [25] employed opinion intervals to analyze content moderation in online communities. They use them to study the effectiveness of moderation strategies on online platforms.

Preliminaries. A *signed graph* G = (*V, E*+ ∪ E−) is given by its vertices V , positive edges E+,
and negative edges E−, where E+ ∩ E− = ∅. It is *complete* if E+ ∪ E− =V2
. For u ∈ V , we write N +(u) to denote its neighbors in E+ and N −(u) to denote its neighbors in E−.

A graph G = (*V, E*) is an *interval graph* if we can assign an interval Iv ⊂ R to all vertices v ∈ V
such that for all *u, v* ∈ V , it holds that {u, v} ∈ E if and only if Iu ∩ Iv ̸= ∅. Additionally, we say that G is a *unit* interval graph if all intervals have length 1.

## 2 Problem Definition And Theoretical Results

In this section, we define our novel problem and state our main theoretical results.

Problem 2.1 (BEST INTERVAL APPROXIMATION). Given a signed graph G = (*V, E*+ ∪ E−), find a set I = {Iv ⊂ R: v ∈ V } of non-empty, contiguous intervals that maximizes

$${\tt a g r e e}(G,{\mathcal{I}})=\sum_{\{u,v\}\in E^{+}}{\mathds{1}}(I_{u}\cap I_{v}\neq\emptyset)+\sum_{\{u,v\}\in E^{-}}{\mathds{1}}(I_{u}\cap I_{v}=\emptyset),$$
1(Iu ∩ Iv = ∅), (1)
where 1(E) is indicator function, which takes value 1 if E is true and 0 otherwise.

Intuitively, the problem assigns an interval Iv to each vertex v and asks that two intervals overlap if their corresponding vertices are connected with a positive edge and do not overlap if their vertices are connected with a negative edge. To connect this problem to opinions, we may consider an interval Iv for a node v as the range of opinions that are acceptable to v. The length |Iv| can further be seen as a measure of v's tolerance towards the opinion spectrum to the left and to the right. We will refer to the formulation in Problem 2.1 as the *agreement* version of BEST INTERVAL APPROXIMATION, which asks to satisfy as many edges as possible. We will also talk about the disagreement version, which aims to minimize the number of edges violating the interval assignment. Their complexity is the same for exact solutions, but they differ w.r.t. approximation guarantees. Relationship to CORRELATION CLUSTERING. Next, we observe that BEST INTERVAL APPROXIMATION is more expressive than CORRE- LATION CLUSTERING: First, consider any CORRELATION CLUSTER-
ING solution C1*, . . . , C*k, and consider k non-overlapping intervals I1*, . . . , I*k. Now we assign each vertex in Cito the same interval Ii. Thus, if *u, v* ∈ Ci, then their intervals overlap, and, if u ∈ Ci and v ∈ Cj for i ̸= j, then their intervals do not overlap. This implies that the optimal solution of BEST INTERVAL APPROXIMATION will always yield an agreement at least as large as for CORRELATION CLUSTERING. Second, our interval representation is strictly more expressive and, for instance, allows us to model non-transitive node relationships and this property is illustrated in Figure 2. This is neither possible in CORRELATION CLUSTERING nor in the structural balance

$$(1)$$

v2 123
−
+ +
v1 v3
theory of Harary [29], in both of which no cycle with exactly one negative edge can be represented without error.

The case of complete graphs and relationship to Interval Editing. If G is a complete signed graph then it can be represented without error in BEST INTERVAL APPROXIMATION if and only if G+ = (*V, E*+)
is an interval graph. That is because missing edges in G+ correspond to negative edges in G (since G is complete). Thus, making the minimum number of edge deletions/insertions to turn G+ into an interval graph is equivalent to flipping the minimum number of edge signs in G such that we have agreement for all edges. Hence, for complete graphs, we can rely on the rich literature on Interval Editing which asks for the minimum number of edge changes to G+ such that it becomes an interval graph. The results of Cao [10] now imply that BEST INTERVAL APPROXIMATION is FPT for complete graphs when only allowing a fixed number of sign changes (in one direction). However, in social networks this number will be large for real-world instances and thus these algorithms are not applicable in practice. Furthermore, our hardness results show that such FPT results are not possible in incomplete graphs when parameterized by the number of required edge deletions (see Section 2.1).

## 2.1 Computational Hardness

Next, we show that BEST INTERVAL APPROXIMATION is NP-hard. We show this by using a reduction from the NP-complete problem ACYCLIC DIGRAPH PARTITION [22], where we are given a directed graph H = (V, E) and have to decide whether one can partition V into two sets V1 and V2, such that both H[V1] and H[V2] are directed acyclic graphs. Our hardness result is stated below. In the theorem, we say that an interval representation is conflictfree if it achieves agreement for all edges, i.e., if Equation (1) equals the number of edges in the graph. Further, we will consider the minimum number of *edge deletions* required to make the graph conflict-free, which is identical to the optimal objective function value for the disagreement version of BEST INTERVAL APPROXIMATION.

1 Theorem 2.2. *There exists a polynomial-time algorithm that, given an instance* H = (*V, E*) of ACYCLIC DIGRAPH PARTITION, outputs an instance G = (V
′, E+ ∪ E−) of BEST INTERVAL
APPROXIMATION with the following properties: (1) H is a YES-instance if and only if a conflict-free interval representation of G exists. (2) If H *is a NO-instance, then only a single edge deletion is* required to obtain a conflict-free interval representation of G*. (3)* |V
′| = O(|V |), |E+∪E−| <|V
′| 2
,
and G+ = (V
′, E+) *is a cycle. Thus,* BEST INTERVAL APPROXIMATION is NP*-hard.*
The theorem has several implications for BEST INTERVAL APPROXIMATION in incomplete graphs: (1) The disagreement version is hard to approximate within any factor. (2) It is not FPT when parameterized by the number of required edge deletions (unless P = NP), separating it from the problem in complete graphs. (3) The result holds even when restricted to graphs G = (V
′, E+ ∪ E−)
where G+ = (V
′, E+) is a chordless cycle. This is intriguing because many algorithmic results on interval graphs rely on detecting forbidden induced subgraphs like chordless cycles of four or more vertices [10, 64, 37]. Our hardness result implies that detecting these forbidden structures is NP-hard for incomplete signed graphs.

We prove Theorem 2.2 in Appendix A, where we construct a new graph G from an ACYCLIC DIGRAPH PARTITION instance H, and show that G can be represented conflict-free if and only if H
is a YES-instance. In G, we introduce two auxiliary vertices L and R and we show that all vertices whose intervals overlap with the interval of L (R) must be in partition V1 (V2) in the optimal solution of ACYCLIC DIGRAPH PARTITION. Thus, the overlap structure of the intervals encodes a partition of the vertices of H. Crucially, we use negative edges to enforce a topological ordering over these partitions and the induced subgraphs H[V1] and H[V2], and we introduce further auxiliary vertices to ensure that G forms a cycle.

## 2.2 A Ptas For Fixed K **In Complete Graphs**

INTERVAL APPROXIMATION in which we must find k intervals I1*, . . . , I*k ⊂ R and each vertex v ∈ V must be assigned to one of these intervals. In practice, the small number of intervals makes the results highly interpretable. Additionally, it applies to scenarios such as analyzing political votes, where we would like to have one interval representing each party, and the number of parties is small. Our result for this restricted version of the problem is as follows.

Theorem 2.3. Let G be a complete signed graph and let ε > 0, δ > 0 and k ∈ N be parameters.

There exists an algorithm that, with probability at least 1 − δ*, returns a* (1 + ε)-approximate solution for BEST INTERVAL APPROXIMATION when the algorithm can only use k different intervals and it runs in time 2 O(k 2log(k/(εδ))/ε3)· n.

The complete description and analysis of the PTAS are provided in Appendix B. An overview to obtain this result is as follows. Since k is fixed, we can enumerate all possible choices of k intervals with respect to their overlap structure. Now, given a fixed set of k intervals, the main observation is that this corresponds to a generalized instance of CORRELATION CLUSTERING where we are given k fixed clusters that might overlap. Specifically, when two clusters Vi and Vj overlap, we want their vertices to be connected by positive edges (rather than negative edges in the classic version of CORRELATION CLUSTERING). Then, we show that we can generalize a PTAS from Giotis and Guruswami [28] as described below.

We solve the generalized version of CORRELATION CLUSTERING by partitioning V into m = O(1/ε) equally-sized subsets V1*, . . . , V*m. Then, for each i = 1*, . . . , m*, we proceed as follows. We sample a set of vertices Si ⊆ V \ Vi of size O˜1/ε2. Now, we enumerate all possible assignments of Siinto (Si,1, . . . , Si,k), where Si,ℓ ⊆ Si are the vertices assigned to interval Iℓ, and for each such assignment, we greedily assign the vertices v ∈ Vito the interval that maximizes the agreement of v's edges to the clustering of Si given by (Si,1, . . . , Si,k). This process gives a clustering of Vi and we show how the clusterings of the Vi can be merged to obtain a global clustering of V . Our analysis is similar to that of [28] and shows that the sets Si are small enough such that enumerating all assignments is not too expensive, and simultaneously large enough that for most vertices they give us a good estimate for the agreement of their edges w.r.t. a fixed clustering. This is the key to arguing that the greedy assignment will yield a good result when we consider the correct clustering of the Si. In contrast to [28], we have to take into account the overlap of intervals when computing the estimates. As for [28], the approach does not extend to incomplete graphs or large k, since then the sets Si become too large and enumeration would not be possible anymore.

## 3 Heuristic Algorithms

Next, we present our heuristic *Greedy Agreement Interval Assignment* (GAIA) for BEST INTERVAL APPROXIMATION, which is given intervals I1*, . . . , I*k as input to which all vertices must be assigned.

We use the following notation. For an interval Iℓ we let overlap(ℓ) = {ℓ
′: Iℓ ∩ Iℓ
′ ̸= ∅} denote the set of intervals Iℓ
′ that overlap with Iℓ. Furthermore, we will consider disjoint vertex clusters C1*, . . . , C*k ⊆ V that correspond to an assignment of the vertices to the intervals, i.e., Cℓ contains all vertices assigned to interval Iℓ. Now, for a vertex u and C1*, . . . , C*k as before, we write

$\text{agree}(u,\ell,(C_{1},\ldots,C_{k}))=\sum_{\ell^{\prime}\in\text{rowfMap}(\ell)}|N^{+}(u)\cap C_{\ell^{\prime}}|+\sum_{\ell^{\prime}\notin\text{rowfMap}(\ell)}|N^{-}(u)\cap C_{\ell^{\prime}}|$
for the number of agreeing edges when assigning vertex u to interval Iℓ for the clustering C1*, . . . , C*k.

Now, we describe GAIA and state its pseudocode in Algorithm 1. GAIA is based on *iterative refinement*: After computing an initial greedy assignment of all vertices, the solution is improved by reassigning vertices in multiple epochs. This reassignment procedure is carried out in batches to avoid local minima. In each epoch, the vertex set is partitioned into random batches V1*, . . . , V*m, and the algorithm iterates over these batches one at a time. When processing a batch Vi, all vertices in the batch are first unassigned and then reassigned using the greedy procedure described below. This can be viewed as a practical version of PTAS from Theorem 2.3, where, instead of brute-forcing solutions on out-of-batch vertices, the algorithm leverages the previously constructed greedy solution.

The core of GAIA is the *greedy assignment* of vertices in Vito intervals in Line 5–8 in Algorithm 1. Here, each vertex v ∈ Viis assigned to the interval Iℓ (and its corresponding cluster Cℓ) that Algorithm 1: Greedy Agreement Interval Assignment (GAIA)
Input: Signed graph G = (*V, E*+ ∪ E−), intervals I1*, . . . , I*k Output: Interval assignment (C1*, . . . , C*k) where Cℓ are the vertices assigned to interval Iℓ 1 Compute an initial assignment of the vertices to the intervals; 2 **for each** epoch do 3 Randomly partition V into m sets V1*, . . . , V*m of size nm each; 4 for i = 1*, . . . , m* do 5 Cℓ ← Cℓ \ Vi for all ℓ = 1*, . . . , k*; // Unassign all vertices in Vi 6 for v ∈ Vi*in order of maximum agreement* do 7 ℓ ← argmaxℓ=1*...k* agree(v, ℓ,(C1*, . . . , C*k)); 8 Cℓ ← Cℓ ∪ {v}; // Assign v to Iℓ 9 **return** (C1*, . . . , C*k);
maximizes agree(v, ℓ,(C1*, . . . , C*k)) (breaking ties at random). Crucially, we assign the vertices with the highest agreement values first, as these vertices are easier to assign and their assignment provides more information when assigning later vertices.

We also provide a version of GAIA called *Variable ENergy Uphill Search* (VENUS), which additionally uses *simulated annealing* [36] to further increase the variability of its solutions. In VENUS, vertices are not necessarily assigned to the interval that maximizes agree(v, ℓ,(C1*, . . . , C*k)), but instead, each vertex is assigned to an interval selected probabilistically according to a temperature-scaled softmax distribution over agreement values. To that end, Line 7 of Algorithm 1 is replaced with ℓ ∼ softmaxℓ=1*...k* agree(v,ℓ,(C1*,...,C*k))
t. Here, t is a temperature parameter and controls the level of randomness during the assignment. A temperature t close to 0 corresponds to a more greedy approach, while higher temperatures lead to increasingly uniform random assignments. The annealing schedule follows exponential decay: the temperature is initialized at t0 and multiplied by a decay factor α ∈ (0, 1) after every τ epochs. This gradually reduces randomness and encourages convergence.

## 4 Experiments

Next, we experimentally evaluate our algorithms. Our code is available in a GitHub repository.2 We aim to answer the following research questions:
(RQ1) Does BEST INTERVAL APPROXIMATION yield a substantial increase in expressiveness compared to CORRELATION CLUSTERING?

(RQ2) How computationally efficient and scalable are our proposed algorithms? (RQ3) What is the trade-off between solution quality and the number of intervals? (RQ4) Are the solutions produced by our method interpretable? (RQ5) Are our algorithms able to recover ground-truth interval structures?

We evaluate our algorithms on real-world datasets from SNAP [46] and KONECT [39]. We further provide a novel dataset based on voting data from the German Bundestag (parliament) between the years of 2012 and 2025 and make it available in our repository. In this dataset, each Bundestag member corresponds to a vertex in the graph, and two members are connected by a positive edge if they vote the same way in at least 75% of the sessions they both attended. Conversely, they are connected by a negative edge if their votes align in 25% of sessions or less.

In our experiments, we evaluate our base algorithm, GAIA, as well as the VENUS variant that uses simulated annealing. For VENUS, we use an initial temperature of 100 and a decay factor of 2/3, applied every 5 epochs. Both are run with 10 batches for vertex reassignment. For the interval structure that our algorithms receive as input, unless stated otherwise, we use a chain-like structure of 8 intervals, where each interval overlaps with the next, e.g., [0, 1], [1, 2]*, . . . ,* [7, 8], and we call this interval structure an 8-Chain. This structure was chosen to find a trade-off between increased 2https://github.com/Peter-Blohm/discovering_opinion_intervals

| Our algorithms   | CORRELATION CLUSTERING baselines   |           |          |       |       |       |         |         |       |             |
|------------------|------------------------------------|-----------|----------|-------|-------|-------|---------|---------|-------|-------------|
| Dataset          | |V |                               | |E|       | |E+| |E| | GAIA  | VENUS | GAEC  | GAECKLj | SCMLEvo | RAMA  | Improvement |
| BitcoinOTC       | 5 881                              | 21 434    | 0.85     | 3.32  | 3.55  | 5.58  | 5.57    | 5.57    | 5.64  | 40.39%      |
| Chess            | 7 301                              | 32 650    | 0.58     | 19.82 | 19.63 | 28.64 | 28.10   | 27.33   | 39.98 | 28.17%      |
| WikiElec         | 7 115                              | 100 355   | 0.78     | 11.24 | 11.26 | 14.13 | 14.13   | 14.13   | 14.45 | 20.45%      |
| Bundestag        | 1 480                              | 397 497   | 0.81     | 0.25  | 0.25  | 3.06  | 2.95    | 2.95    | 3.72  | 91.53%      |
| Slashdot         | 82 140                             | 498 532   | 0.76     | 9.05  | 8.94  | 13.75 | 13.66   | 13.52   | 17.17 | 33.88%      |
| Epinions         | 131 580                            | 708 507   | 0.83     | 4.47  | 4.42  | 6.83  | 6.68    | 6.67    | 6.86  | 33.73%      |
| WikiSigned       | 138 587                            | 712 337   | 0.88     | 4.94  | 4.85  | 6.17  | 6.17    | 6.17    | 6.96  | 21.39%      |
| WikiConflict     | 116 836                            | 2 014 053 | 0.38     | 3.44  | 3.43  | 5.87  | 5.82    | 5.82    | 6.02  | 41.06%      |

expressivity and intuitive interpretation (see also discussion of RQ3 below). Where applicable, experiments were repeated 50 times on different random seeds, and standard deviations are reported. Numerous approaches have been proposed for solving CORRELATION CLUSTERING in social networks analysis [49, 9, 31, 30] and in computer vision [35, 66, 42, 1]. To provide a representative performance overview, we selected four state-of-the-art algorithms for comparison: - GAEC [35]: A method that incrementally merges clusters to minimize disagreement. - GAECKLj [35]: An extension of GAEC that additionally applies local search postprocessing.

- SCMLEvo [30]: An algorithm combining multilevel local search with evolutionary techniques. - RAMA [1]: An algorithm using polyhedral relaxation and message passing to guide cluster merging.

For each of these algorithms, we run the authors' publicly available implementations. In contrast to our algorithms, which only use 8 intervals, the baselines may use an unrestricted number of clusters. Throughout our experiments, we report the *disagreement*, i.e., the fraction of violated edges in solutions found across all real-world datasets (rather than the number of agreeing edges as in Equation (1)), as this makes the algorithms' performance easier to compare. Further details on the experiment setup, as well as additional results, are described in Appendix C.

Expressivity analysis (RQ1). To compare the expressivity of BEST INTERVAL APPROXIMATION with CORRELATION CLUSTERING, we run our algorithms and the baselines on the real-world datasets. The results in Table 1 show that our algorithms consistently find interval assignments that achieve 20% to 90% fewer disagreements than the best CORRELATION CLUSTERING solution. Across all datasets, our results have 38% less disagreement on average, even though our methods only use 8 intervals, whereas the CORRELATION CLUSTERING baselines use an unrestricted number of clusters. Hence, our heuristic algorithms manage to effectively use the increased expressivity of the overlapping interval structure. Additionally, we find that VENUS tends to outperform GAIA slightly, particularly for larger graphs. Computational efficiency (RQ2). Next, we assess the runtime efficiency of our algorithms by tracking the progression of the objective value over time. Representative results for Slashdot are presented in Figure 3a. GAIA makes the most progress in the first 15 seconds, followed by slower, incremental gains. VENUS exhibits a similar pattern, though slightly delayed, likely due to its initially high temperature which slows early convergence. However, this high initial temperature appeared to be necessary to achieve improvement over GAIA's results. In most instances, both heuristics stopped after 50 epochs without improvement in the first five minutes of runtime, with GAIA often terminating after a few seconds. The algorithms' running time until convergence scales approximately linearly in the number of edges and on all datasets our methods terminate within 30 minutes; we also elaborate on this in the appendix.

Number of intervals (RQ3). To investigate the relationship between the number of intervals and the solution quality, we ran our algorithms with 4, 8, 12, and 16 intervals. As for the 8-Chain, in each interval structure, every interval overlaps with its successor and predecessor. In Figure 3a, we illustrate the convergence behavior, and Figure 3b presents the solution quality after convergence, both on the Slashdot dataset. Our results show that using only 4 intervals leads to poor solution quality

45k 50k 55k 0s 25s 50s 75s 100s 125s Runtime Di sag ree me nt Algorithm GAIA VENUS
Number of Intervals 4 8 12 16
(a) Objective value over time.

45k 50k 55k 4 8 12 16 Number of Intervals Di sag ree m en t Algorithm GAIA VENUS
compared to the higher numbers, suggesting that such a limited structure may not adequately capture the complexity of the graph. While the solution quality improves with more intervals, 8 intervals seem sufficiently expressive for this graph, with only marginal improvements beyond that. This behavior is typical for other problem instances as well. Again, we see that VENUS tends to perform slightly better, and most notably, its results have much less variance compared to GAIA.

Interpretability (RQ4). To study the interpretability of our solutions, we perform a case study on the Bundestag dataset. We present a representative solution found by VENUS in Figure 1. As the dataset models co-voting behavior of politicians, we expect our interval representation to resemble the German political spectrum, and, indeed, this is the case. Our result assigns most members from the same party to the same or neighboring intervals. For each party, except the FDP, we can identify one interval consisting mainly of members of that party. We note that the slight splitting up of parties is natural due to government coalitions they formed throughout the years. Also, the behavior of the FDP can be traced back to different coalition governments they were part of (they formed governments with the conservative CDU/CSU, as well as with the left/center GRÜNE and SPD). We consider the ability of our algorithms to extract such highly overlapping structure as a substantial improvement over CORRELATION CLUSTERING, and this is also emphasized by the objective function values reported in Table 1, where our methods have 91% fewer disagreeing edges on this dataset. Reconstruction of ground-truth data (RQ5). Next, we evaluate our algorithms on synthetic data. We fix the 8-Chain and generate a graph with n = 800 vertices as follows. We assign n 8 vertices to each interval, and we introduce edges with signs corresponding to the interval structure for dn2 random pairs of vertices, where d ∈ [0, 1] is the desired density of the graph. Each edge obtains a correct edge sign based on the interval structure with probability 1 − p and we flip the sign with probability p. In our experiments, we measure the relative change of the objective function achieved by our algorithms compared to the ground-truth assignment in percent, agree(G,ground truth)−agree(G,ALG)
|E|· 100, and we also report the accuracy with which vertices are assigned to their corresponding interval. Figure 4 shows the result of our experiments. Without sign noise, the solutions are always within 6.5% of the ground truth for VENUS, and within 12.5% of the ground truth for GAIA. We also obtain a high accuracy in reconstructing the ground-truth assignment. With increasing sign noise, the true solution becomes increasingly suboptimal to the point where both GAIA and VENUS find alternative solutions, with *better* objective values than the ground truth (this is the case when we have negative y-axis values in the plot). This increased objective value, however, comes at the cost of less accuracy in the vertex assignment. The point at which alternative solutions become viable depends heavily on the density of the graph, with denser graphs being more resilient to this phenomenon.

## 5 Conclusion

We introduced the BEST INTERVAL APPROXIMATION problem and showed that it is more expressive than CORRELATION CLUSTERING, both theoretically and in experiments. We gave strong hardness results for incomplete graphs, as well as a PTAS for complete graphs and fixed k. We also provided efficient heuristics, which find interval assignments with significantly better objective values than CORRELATION CLUSTERING solutions found by state-of-the-art algorithms, and we showed that these interval assignments are highly interpretable.

-40 -30 -20 -10 0 10 Sign Flip Probability Density: 0.01Sign Flip Probability Density: 0.03Sign Flip Probability Density: 0.1Sign Flip Probability Density: 1 0 0.1 0.2 0.3 0.4 0.5 0 0.1 0.2 0.3 0.4 0.5 0 0.1 0.2 0.3 0.4 0.5 0 0.1 0.2 0.3 0.4 0.5
-40 -30 -20 -10 0 10
-40 -30 -20 -10 0 10
-40 -30 -20 -10 0 10 Di ff e re n ce 
(%
)

0.00 0.25 0.50 0.75 1.00 Sign Flip Probability Density: 0.01Sign Flip Probability Density: 0.03Sign Flip Probability Density: 0.1Sign Flip Probability Density: 1 0 0.1 0.2 0.3 0.4 0.5 0 0.1 0.2 0.3 0.4 0.5 0 0.1 0.2 0.3 0.4 0.5 0 0.1 0.2 0.3 0.4 0.5 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 Ac cu rac y GAIA VENUS Random Assignment

## We Believe There Are Several Interesting Directions For Future Work, Which We Describe Next.

From a more theoretical point of view, several problems remain unresolved. First, we conjecture that in the agreement version of the problem (for k not fixed), the optimal solution can always satisfy a 3 4
-fraction of the edges. This claim is supported by ILP-solutions that we computed on small instances, and it is tight, for instance, when taking two cliques with negative edges and connecting each pair of their vertices with a positive edge. Second, it is interesting to study whether a PTAS exists in this setting. Third, our hardness results do not allow us to rule out that for fixed k and complete graphs a PTAS exists for the disagreement version of our problem. While the techniques of Giotis and Guruswami [28] for CORRELATION CLUSTERING do not seem to extend to this setting, obtaining such a result would be interesting. From a modeling perspective, several extensions are well-motivated. First, a natural extension is to move beyond one-dimensional intervals. Interestingly, neither our PTAS nor our heuristic algorithms are inherently restricted to intervals on a line. Rather, they only require knowledge of which clusters overlap and which do not. Hence, an empirical study using higher-dimensional intervals could allow a more nuanced discovery of opinions along multiple axes. Second, it might be interesting to consider temporal or dynamic settings in which opinion ranges expand or contract over time. Here, one could consider making the opinion intervals expand or contract depending on the nodes' centrality or the homophily of their immediate neighborhood. From a machine learning perspective, it is interesting to study whether methods like GNNs can outperform our algorithms. This might be particularly promising when additional information, such as node labels, are available, which can be exploited by the GNNs.

## Acknowledgments And Disclosure Of Funding

The authors thank the anonymous reviewers for their helpful comments, which have helped us to improve the presentation of the paper. We further thank Sebastian Lüderssen for useful discussions.

This research has been funded by the Vienna Science and Technology Fund (WWTF) [Grant ID:
10.47379/VRG23013], the ERC Advanced Grant REBOUND [834862], the Swedish Research Council (VR) [2024-05603], the European Commission MSCA DN ARMADA [101168951], and the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

## References

[1] A. Abbas and Paul Swoboda. Rama: A rapid multicut algorithm on gpu. *CVPR*, pages 8183–8192, 2021.

[2] Samin Aref and Zachary P. Neal. Identifying hidden coalitions in the us house of representatives by optimally partitioning signed networks based on generalized balance. *Scientific Reports*, 11, 2021.

[3] Sepehr Assadi, Vihan Shah, and Chen Wang. Streaming algorithms and lower bounds for estimating correlation clustering cost. In *NeurIPS*, 2023.

[4] Nikhil Bansal, Avrim Blum, and Shuchi Chawla. Correlation clustering. *Mach. Learn.*, 56(1-3):
89–113, 2004.

[5] Pablo Barberá. Birds of the same feather tweet together: Bayesian ideal point estimation using twitter data. *Political analysis*, 23(1):76–91, 2015.

[6] Thorsten Beier, Thorben Kröger, Jörg H. Kappes, U. Köthe, and Fred A. Hamprecht. Cut, glue,
& cut: A fast, approximate solver for multicut partitioning. *CVPR*, pages 73–80, 2014.

[7] Francesco Bonchi, A. Gionis, and Antti Ukkonen. Overlapping correlation clustering. *Knowledge and Information Systems*, 35:1–32, 2011.

[8] Francesco Bonchi, David García-Soriano, and Francesco Gullo. *Correlation Clustering*. Synthesis Lectures on Data Mining and Knowledge Discovery. Springer, 2022.

[9] Michael J. Brusco and Patrick Doreian. Partitioning signed networks using relocation heuristics, tabu search, and variable neighborhood search. *Social Networks*, 56:70–80, 2019.

[10] Yixin Cao. Linear recognition of almost interval graphs. In *SODA*, pages 1096–1115, 2016. [11] Yixin Cao. Unit interval editing is fixed-parameter tractable. *Inf. Comput.*, 253:109–126, 2017.

[12] Yixin Cao and Dániel Marx. Interval deletion is fixed-parameter tractable. ACM Trans.

Algorithms, 11(3):21:1–21:35, 2015.

[13] Moses Charikar, Venkatesan Guruswami, and Anthony Wirth. Clustering with qualitative information. *FOCS*, pages 524–533, 2003.

[14] Kai-Yang Chiang, Joyce Jiyoung Whang, and Inderjit S. Dhillon. Scalable clustering of signed networks using balance normalized cut. In *CIKM*, pages 615–624, 2012.

[15] Vincent Cohen-Addad, Silvio Lattanzi, Slobodan Mitrovic, Ashkan Norouzi-Fard, Nikos Parotsidis, and Jakub Tarnawski. Correlation clustering in constant many parallel rounds. In *ICML*,
volume 139, pages 2069–2078, 2021.

[16] Vincent Cohen-Addad, Silvio Lattanzi, Andreas Maggiori, and Nikos Parotsidis. Online and consistent correlation clustering. In *ICML*, volume 162, pages 4157–4179, 2022.

[17] Vincent Cohen-Addad, Euiwoong Lee, and Alantha Newman. Correlation Clustering with Sherali-Adams. In *FOCS*, pages 651–661, 2022.

[18] Vincent Cohen-Addad, Euiwoong Lee, Shi Li, and Alantha Newman. Handling correlated rounding error via preclustering: A 1.73-approximation for correlation clustering. In *FOCS*,
pages 1082–1104, 2023.

[19] Vincent Cohen-Addad, Silvio Lattanzi, Andreas Maggiori, and Nikos Parotsidis. Dynamic correlation clustering in sublinear update time. In *ICML*, 2024.

[20] Vincent Cohen-Addad, David Rasmussen Lolck, Marcin Pilipczuk, Mikkel Thorup, Shuyi Yan, and Hanwen Zhang. Combinatorial correlation clustering. In *STOC*, pages 1617–1628. ACM,
2024.

[21] Mihai Cucuringu, Peter Davies, Aldo Glielmo, and Hemant Tyagi. SPONGE: A generalized eigenproblem for clustering signed networks. In *AISTATS*, volume 89, pages 1088–1098. PMLR, 2019.

[22] Marek Cygan, Marcin Pilipczuk, Michal Pilipczuk, and Jakub Onufry Wojtaszczyk. Sitting closer to friends than enemies, revisited. *Theory Comput. Syst.*, 56:394–405, 2015.

[23] Guillaume Deffuant, David Neau, Frederic Amblard, and Gérard Weisbuch. Mixing beliefs among interacting agents. *Advances in Complex Systems*, 3(01n04):87–98, 2000.

[24] Morris H DeGroot. Reaching a consensus. *J Am Stat Assoc*, 69(345):118–121, 1974. [25] Cynthia Dwork, Chris Hays, Jon M. Kleinberg, and Manish Raghavan. Content moderation and the formation of online communities: A theoretical framework. In WWW, pages 1307–1317.

ACM, 2024.

[26] Noah E Friedkin and Eugene C Johnsen. Social influence and opinions. Journal of Mathematical Sociology, 15(3-4):193–206, 1990.

[27] M. R. Garey and David S. Johnson. Computers and Intractability: A Guide to the Theory of NP-Completeness. W. H. Freeman, 1979. ISBN 0-7167-1044-7.

[28] Ioannis Giotis and Venkatesan Guruswami. Correlation clustering with a fixed number of clusters. *Theory Comput.*, 2(13):249–266, 2006.

[29] Frank Harary. On the notion of balance of a signed graph. *Michigan Mathematical Journal*, 2
(2):143–146, 1953.

[30] Felix Hausberger, Marcelo Fonseca Faraj, and Christian Schulz. Scalable multilevel and memetic signed graph clustering. In *ALENEX*, pages 81–94, 2025.

[31] Jia-Lin Hua, Jian Yu, and Miin-Shen Yang. Fast clustering for signed graphs based on random walk gap. *Social Networks*, 60:113–128, 2020.

[32] Kevin G. Jamieson and Ameet Talwalkar. Non-stochastic best arm identification and hyperparameter optimization. In *AISTATS*, pages 240–248, 2016.

[33] Haim Kaplan, Ron Shamir, and Robert Endre Tarjan. Tractability of parameterized completion problems on chordal, strongly chordal, and proper interval graphs. *SIAM J. Comput.*, 28:
1906–1922, 1999.

[34] Anne-Marie Kermarrec and Christopher Thraves. Can everybody sit closer to their friends than their enemies? In *MFCS*, pages 388–399, 2011.

[35] Margret Keuper, Evgeny Levinkov, Nicolas Bonneel, Guillaume Lavoué, Thomas Brox, and Bjoern Andres. Efficient decomposition of image and mesh graphs by lifted multicuts. *ICCV*, pages 1751–1759, 2015.

[36] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi. Optimization by simulated annealing. *Science*,
220(4598):671–680, 1983.

[37] Dieter Kratsch, Ross M. McConnell, Kurt Mehlhorn, and Jeremy P. Spinrad. Certifying algorithms for recognizing interval graphs and permutation graphs. *SIAM J. Comput.*, 36(2): 326–353, 2006.

[38] Ulrich Krause. A discrete nonlinear and non-autonomous model of consensus formation.

Communications in Difference Equations, 2000, 07 2000. doi: 10.1201/b16999-21.

[39] Jérôme Kunegis. Konect: the koblenz network collection. WWW, 2013. [40] Jérôme Kunegis, Stephan Schmidt, Andreas Lommatzsch, Jürgen Lerner, Ernesto William De Luca, and Sahin Albayrak. Spectral analysis of signed graphs for clustering, prediction and visualization. In SDM, pages 559–570. SIAM, 2010.

[41] Yuko Kuroki, Atsushi Miyauchi, Francesco Bonchi, and Wei Chen. Query-efficient correlation clustering with noisy oracle. In *NeurIPS*, 2024.

[42] Jan-Hendrik Lange, Andreas Karrenbauer, and Bjoern Andres. Partial optimality and fast lower bounds for weighted correlation clustering. In *ICML*, volume 80, pages 2892–2901, 2018.

[43] Silvio Lattanzi, Benjamin Moseley, Sergei Vassilvitskii, Yuyan Wang, and Rudy Zhou. Robust online correlation clustering. In *NeurIPS*, pages 4688–4698, 2021.

[44] Jacopo Lenti, Corrado Monti, and Gianmarco De Francisci Morales. Likelihood-based methods improve parameter estimation in opinion dynamics models. In *WSDM*, pages 350–359, 2024.

[45] Jacopo Lenti, Fabrizio Silvestri, and Gianmarco De Francisci Morales. Variational inference of parameters in opinion dynamics models. *CoRR*, abs/2403.05358, 2024.

[46] Jure Leskovec and Andrej Krevl. SNAP Datasets: Stanford large network dataset collection.

http://snap.stanford.edu/data, June 2014.

[47] Jure Leskovec, Daniel P. Huttenlocher, and Jon M. Kleinberg. Predicting positive and negative links in online social networks. In WWW, pages 641–650, 2010.

[48] Jure Leskovec, Daniel P. Huttenlocher, and Jon M. Kleinberg. Signed networks in social media.

In CHI, pages 1361–1370, 2010.

[49] Evgeny Levinkov, Alexander Kirillov, and Bjoern Andres. A comparative study of local search algorithms for correlation clustering. In *German Conference on Pattern Recognition*, 2017.

[50] Mário Levorato, Rosa Figueiredo, Yuri Frota, and Lúcia M. A. Drummond. Evaluating balancing on social networks through the efficient solution of correlation clustering problems. *EURO*
Journal on Computational Optimization, 5:467–498, 2017.

[51] Konstantin Makarychev and Sayak Chakrabarty. Single-pass pivot algorithm for correlation clustering. keep it simple! In *NeurIPS*, 2023.

[52] Nolan McCarty, Keith T. Poole, and Howard Rosenthal. Income redistribution and the realignment of American politics. AEI Press, 1997.

[53] Cameron Musco, Christopher Musco, and Charalampos E. Tsourakakis. Minimizing polarization and disagreement in social networks. In WWW, pages 369–378, 2018.

[54] Stefan Neumann and Pan Peng. Sublinear-time clustering oracle for signed graphs. In *ICML*,
volume 162, pages 16496–16528, 2022.

[55] Eduardo G. Pardo, Mauricio Soto, and Christopher Thraves. Embedding signed graphs in the line - heuristics to solve minsa problem. *J. Comb. Optim.*, 29:451–471, 2015.

[56] Eduardo G. Pardo, Antonio García-Sánchez, Marc Sevaux, and Abraham Duarte. Basic variable neighborhood search for the minimum sitting arrangement problem. *J. Heuristics*, 26:249–268, 2020.

[57] Keith T. Poole and Howard Rosenthal. Patterns of congressional voting. American Journal of Political Science, 35:228, 1991.

[58] Keith T. Poole and Howard Rosenthal. Congress: A Political-Economic History of Roll Call Voting. Oxford University Press, 1997.

[59] Fred S. Roberts. Indifference graphs. *Proof Techniques in Graph Theory*, pages 139–146, 1969. [60] Chaitanya Swamy. Correlation clustering: maximizing agreements via semidefinite programming. In *SODA*, 2004.

[61] Paul Swoboda and Bjoern Andres. A message passing algorithm for the minimum cost multicut problem. CVPR, pages 4990–4999, 2016.

[62] Sijing Tu and Stefan Neumann. A viral marketing-based model for opinion dynamics in online social networks. In WWW, pages 1570–1578, 2022.

[63] Ruo-Chun Tzeng, Bruno Ordozgoiti, and Aristides Gionis. Discovering conflicting groups in signed networks. In *NeurIPS*, 2020.

[64] Yngve Villanger, Pinar Heggernes, Christophe Paul, and Jan Arne Telle. Interval completion is fixed parameter tractable. *SIAM J. Comput.*, 38(5):2007–2020, 2009.

[65] Yanbang Wang and Jon M. Kleinberg. On the relationship between relevance and conflict in online social link recommendations. In *NeurIPS*, 2023.

[66] Steffen Wolf, Constantin Pape, Alberto Bailoni, Nasim Rahaman, Anna Kreshuk, U. Köthe, and Fred A. Hamprecht. The mutex watershed: Efficient, parameter-free image partitioning. In ECCV, 2018.

[67] Liwang Zhu, Qi Bao, and Zhongzhi Zhang. Minimizing polarization and disagreement in social networks via link recommendation. In *NeurIPS*, pages 2072–2084, 2021.

## Neurips Paper Checklist 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claimed theoretical results are formalized in Section 2 with additional proofs in Appendices A and B. The heuristic algorithms are described in Section 3, and the claimed empirical results are supported in Section 4.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Our theorems clearly state the assumptions that are necessary for them to hold. Regarding the performance of our practical algorithms, we have evaluated them to the best of our knowledge and also provide further information on their scalability in the appendix.

3. **Theory assumptions and proofs**
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: For each theoretical result (Theorems 2.2 and 2.3) we clearly state assumptions and provide proof sketches in Section 2. Formal proofs are provided in Appendices A and B and cross-referenced to Theorems 2.2 and 2.3, respectively.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The heuristic algorithms are described in Section 3, the dataset generation process and the parameters used to run the experiments are described in Section 4 and Appendix C. Additionally, our code is available in a GitHub repository.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Our code, including the experiment setup, is available as a GitHub repository, referenced in Footnote 2 in Section 4. All datasets used are either publicly available or included in the linked repository.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The parameters used to run all experiments are described in Section 4 and Appendix C. Additionally, the code to reproduce our experiments is available in a GitHub repository.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: All experimental results in Section 4 and Appendix C include averages, standard deviations, and confidence intervals where applicable. In the box plots, the notches extend 1.58 · *IQR/*√n, giving roughly a 95% confidence interval for comparing medians.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: The computer resources used in the experiments are described in Appendix C.

9. **Code of ethics**
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The authors have ensured that the research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA]
Justification: We believe that our work has no broader societal impact, given that our focus is on foundational research. While we offer a new perspective on modeling opinion diversity using limited interaction data, similar insights are already frequently obtained in the real world by existing methods using richer user-level data.

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]
Justification: The paper poses no such risks.

12. **Licenses for existing assets**
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We cite all original papers where code or datasets are used in our work. In particular, for the scraped Bundestag dataset, we provide copyright- and legal notice in the GitHub repository.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: The generation process for the newly introduced dataset is described in Section 4. The dataset and the code used for its generation are available in a GitHub repository.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No crowdsourcing or experiments with human subjects were used in the paper.

15. **Institutional review board (IRB) approvals or equivalent for research with human**
subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No crowdsourcing or experiments with human subjects were used in the paper.

## 16. **Declaration Of Llm Usage**

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLMs were not involved in the core method development.

## A Hardness Result

In this section, we prove our hardness result from Theorem 2.2. In the proof, we use the notation I = [ℓ(I), r(I)], where ℓ(I) denotes the infimum and r(I) denotes the supremum of the interval.

## A.1 Construction

We describe a reduction from an instance H = (*V, E*) of ACYCLIC DIGRAPH PARTITION to an instance G = (V
′, E+ ∪ E−) of BEST INTERVAL APPROXIMATION, where (V
′, E+) forms a cycle.

First, the set of vertices V
′consists of:
1. Seven constant vertices V
′
c = {S, L, M, R, T, HS, HT }. These vertices will be used to constrain the structure of the solution. With sets of negative edges, we will force any conflict-free representation to assign each vertex in V to a sub-interval of either L or R. The names of the vertices stand for start, left, middle, right, target, help-start, and help-target, respectively.

2. Four vertices for each vertex v ∈ V : V
′
v = {Mv, Av, Xv, Bv}, where Xv corresponds to the original vertex v in H and the other vertices are used to structure the instance.

Next, we define the set of positive edges E+ to form a cycle over V
′. For this, we use an arbitrary ordering of V = {v1*, . . . , v*n}, and construct E+ as the union of the following sets:
1. E+
c = {{S, L}, {L, M}, {M, R}, {R, T}, {S, HS}, {HT , T}}
2. for all v ∈ V : E+
v = {{Mv, Av}, {Av, Xv}, {Xv, Bv}}
3. E
+
V 
= {{HS, Mv1
}} ∪ Si∈[1,n−1]{{Bvi
, Mvi+1 *}} ∪ {{*Bvn
, HT }}
Finally, we construct E− as the union of the following sets:
1. E
−
S = {{*S, v*′} : v
′ ∈ V
′ \ {*L, H*S}}, E
−
T = {{*T, v*′} : v
′ ∈ V
′ \ {*R, H*T }}
We connect negatively S and T to each vertex in the graph besides their positive neighbors.

This forces IS and IT to be the outermost intervals in any conflict-free interval representation, as otherwise the interval of some negatively connected vertex intersects either of them. See Lemma A.1. To break symmetry, we assume without loss of generality that r(IS) < ℓ(IT ).

2. E−
c = {{*L, R*}}
This edge ensures that IL and IR are disjoint, and in any conflict-free representation, r(IL) < ℓ(IR), due to their respective positive edges to S and T.

3. E
−
M =Sv∈V
{{Mv, L}, {Mv, R}}
These edges ensure that for all vertices v ∈ V the interval IMvlies in IM. See Lemma A.3.

4. E−
∗ = {{Xv, M} : v ∈ V }
These edges ensure that for all vertices v ∈ V , the interval IXveither lies in IL or in IR.

See Lemma A.4.

5. E
−
V = {{Xu, Xv} : u, v ∈ *V, u* ̸= v}
These edges ensure that for all vertices *u, v* ∈ V with u ̸= v the intervals IXuand IXvare disjoint.

6. E
−
E 
=S(u,v)∈E
{{Xv, Au}, {Xv, Bu}}
These edges enforce topological orderings of the vertices. See Lemma A.5.

This concludes the construction. We refer to Figure 5 for an illustration. It is clear that V
′, E+, and E− have the sizes claimed in the theorem.

## A.2 Structural Lemmas

To prove the correctness of the reduction, we will make use of a few smaller results that describe the structure of any conflict-free interval representation of G. First, notice that {S, T} ∈ E−, so for any conflict-free interval representation, it must hold that IS ∩ IT = ∅. For the rest of this analysis, assume without loss of generality r(IS) < ℓ(IT ).

v1 v2 v4 v3 V1 V2 S L M R 
T
HS
v1 v2 v3 v4 HT
Lemma A.1. For any conflict-free interval representation of G *it must hold that for all* u ∈ V
′ \ {S} :
r(IS) < r(Iu) *and for all* u ∈ V
′ \ {T} : ℓ(Iu) < ℓ(IT ).

Proof. Towards a contradiction assume there exists a vertex u ∈ V
′ \ {S} such that r(Iu) ≤ r(IS).

As (V
′, E+) is a cycle, there exists a path from u to T that does not include S. The union Ip of the intervals corresponding to the vertices in this path must form an interval itself. As r(Iu) ≤ r(IS), but ℓ(IT ) > r(IS), IS ∩ Ip ̸= ∅, and consequently there exists some vertex x ∈ V
′ \ {S}, x ̸= u along this path such that IS ∩ Ix ̸= ∅. Note that by the construction of this path, x cannot be a positive neighbor of S, as we explicitly choose one of the two paths from u to T that does not include S.

Then, {S, x} ∈ E−, leading to a contradiction. Finally, for all u ∈ V
′ \ {T} : ℓ(Iu) < ℓ(IT ) holds by a symmetric argument. Lemma A.2. For any conflict-free interval representation of G *it must hold that (i) for all* u ∈ V
′ \ {S, L, HS} : r(IS) < ℓ(Iu) *and (ii) for all* u ∈ V
′ \ {T, R, HT } : r(Iu) < ℓ(IT ).

Proof. Assume towards a contradiction that there exists a vertex u ∈ V
′ \ {*S, L, H*S} such that ℓ(Iu) ≤ r(IS). From Lemma A.1, we know r(Iu) > r(IS), so it follows that Iu ∩ IS ̸= ∅. However, since {S, u} ∈ E− this leads to a contradiction with the assumption of a conflict-free interval representation. The proof for (ii) follows symmetrically. Lemma A.3. For any conflict-free interval representation of G, it must hold that for all v ∈ V :
IMv ⊂ IM. Proof. By construction, the open interval (r(IS), ℓ(IT )) ⊂ IL ∪ IM ∪ IR. From Lemma A.2, we know that IMv ⊂ IL ∪ IM ∪ IR. Finally, as {Mv, L}, {Mv, R} ∈ E−, the claim holds.

Lemma A.4. For any conflict-free interval representation of G, it must hold that for all v ∈ V *either* IXv ⊂ IL or IXv ⊂ IR *but not both.* Proof. By construction, the open interval (r(IS), ℓ(IT )) ⊂ IL ∪ IM ∪ IR. From Lemma A.2, we know that IXv ⊂ IL ∪IM ∪IR. Furthermore, we know that IL ∩IR = ∅. Finally, as {Xv, M} ∈ E−
the claim holds.

Building on this statement, we can further characterize the relative location of the intervals IXvinside IL and IR. Lemma A.5. For any conflict-free interval representation of G *it must hold that for all edges*
(u, v) ∈ E if IXu ⊂ IR and IXv ⊂ IR*, then* r(IXu
) < ℓ(IXv
), and, symmetrically, if IXu ⊂ IL and IXv ⊂ IL, then r(IXv) < ℓ(IXu)
Proof. As we assumed that r(IS) < ℓ(IT ), it follows that ℓ(IM) < ℓ(IR) ≤ r(IM) < r(IR).

Towards a contradiction, assume there exists an edge (*u, v*) ∈ E such that IXu ⊂ IR and IXv ⊂ IR, but ℓ(IXv
) ≤ r(IXu
). As in any conflict-free interval representation the intervals IXuand IXvare disjoint, this implies that r(IXv
) < ℓ(IXu
). Now, consider the intervals IMuand IAu.

From Lemma A.3, we know that IMu ⊂ IM, hence r(IMu) < ℓ(IXv) < r(IXv) < ℓ(IXu). By construction, IAu must overlap with IMuand IXu, hence ℓ(IAu) ≤ r(IMu) and r(IAu) ≥ ℓ(IXu).

However, this implies that IAu ∩IXv̸= ∅, leading to a violation of the {Xv, Au} constraint introduced in E
− E
. By symmetry, this also proves the case where IXu ⊂ IL and IXv ⊂ IL.

## A.3 Proof Of **Theorem 2.2**

Equipped with Lemmas A.1 to A.5 we can now prove Theorem 2.2. We first show that if H = (*V, E*) is a YES-instance of ACYCLIC DIGRAPH PARTITION, then the constructed signed graph instance G has a conflict-free interval-representation in BEST INTERVAL
APPROXIMATION. Assume H[V1] and H[V2] are the two acyclic induced subgraphs of H corresponding to the partition and let k = |V1|. Further, let [v(1,1)*, . . . , v*(1,k)] and [v(2,1)*, . . . , v*(2,n−k)] be topological orderings of V1 and V2, respectively. Now, we define intervals for V
′
cas follows and depicted in Figure 5:
IS := [0, 0.2], IL := [0.2, 0.4], IM := [0.4, 0.6], IR := [0.6, 0.8], IT := [0.8, 1], IHS:= [0.1, 0.5],
IHT:= [0.5, 0.9].

This satisfies the constraints set by E−
c. Then, for each v ∈ V , we assign IMv = [0.45, 0.55]. This satisfies all constraints imposed by E
−
M. Next, we define

for all $i\in\{1,\ldots,k\}:I_{X_{v_{(1,i)}}}=\left[0.4-\frac{2i+1}{16k},0.4-\frac{2i}{16k}\right],$ and  for all $i\in\{1,\ldots,n-k\}:I_{X_{v_{(2,i)}}}=\left[0.6+\frac{2i}{16(n-k)},0.6+\frac{2i+1}{16(n-k)}\right].$
This ensures that for all vertices u ∈ V1, the interval IXulies in (0.2, 0.4), and symmetrically for all vertices v ∈ V2, the interval IXvlies in (0.6, 0.8), hence satisfying E−
∗. Further, for all vertices u, v ∈ V with u ̸= v their intervals IXu
, IXvare disjoint, thereby satisfying E
−
V. To conclude the construction of the interval representation, we set

for all $v\in V_1:I_{A_v}=I_{B_v}=[\ell(I_{X_v}),0.5]$, and for all $v\in V_2:I_{A_v}=I_{B_v}=[0.5,r(I_{X_v})]$. 
Now, all the constraints set in E+, E
−
Sand E
−
Tare satisfied by construction. It is left to check whether the constraints set by E
−
E
are satisfied. Here, IXv must not overlap IAuor IBuif there exists a directed edge (*u, v*) ∈ E. This is trivially satisfied if u ∈ V1 and v ∈ V2 or vice-versa. If both *u, v* ∈ V2, then in the constructed interval representation we must have that ℓ(IXv) > r(IAu) = r(IBu) = r(IXu). As the intervals {IXv: v ∈ V2} were constructed according to a topological ordering of V2, this is always satisfied. The argument works symmetrically for V1, and hence the interval representation is conflict-free. Conversely, suppose the constructed instance G admits a conflict-free interval representation in BEST INTERVAL APPROXIMATION. We claim that this implies H is a YES-instance of ACYCLIC DIGRAPH PARTITION. First, by Lemmas A.1 and A.2, any conflict-free interval representation places IS and IT at the extreme left and extreme right, respectively. Consequently, in the open interval r(IS), ℓ(IT ), the intervals IL, IM, and IR appear in that left-to-right order. Next, Lemma A.3 guarantees that every interval IMvfor v ∈ V is contained in IM. Meanwhile, Lemma A.4 ensures that each IXvis contained entirely in either IL or IR. This setup naturally suggests a bipartition of the set V :

$V_{1}=\{\,v\in V:I_{X_{v}}\subset I_{L}\}\quad\mbox{and}\quad V_{2}=\{\,v\in V:I_{X_{v}}\subset I_{R}\}$.  
We claim that H[V1] and H[V2] must each be acyclic. Indeed, in Lemma A.5 we show that for any directed edge (u, v) ∈ E with *u, v* ∈ V1, the intervals IXuand IXvin IL must satisfy r(IXv) < ℓ(IXu
). Hence, the interval IXv must lie to the left of IXu. This implies a topological ordering of vertices in V1, and thus prevents directed cycles in H[V1]. A symmetric argument shows that H[V2] is acyclic. Thus, H admits a partition of its vertex set into two DAGs H[V1] and H[V2]. Therefore, H is a YES-instance of ACYCLIC DIGRAPH PARTITION, completing the proof of Theorem 2.2.