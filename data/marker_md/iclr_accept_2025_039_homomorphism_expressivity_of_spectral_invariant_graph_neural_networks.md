# HOMOMORPHISM EXPRESSIVITY OF SPECTRAL IN-VARIANT GRAPH NEURAL NETWORKS

Jingchu Gai <sup>1</sup> Yiheng Du <sup>1</sup> Bohang Zhang<sup>1</sup><sup>∗</sup> Haggai Maron <sup>2</sup>,<sup>3</sup> Liwei Wang <sup>1</sup> 1 Peking University <sup>2</sup>Technion <sup>3</sup>NVIDIA Research

gaijingchu@stu.pku.edu.cn, zhangbohang@pku.edu.cn, duyiheng@stu.pku.edu.cn hmaron@nvidia.com, wanglw@pku.edu.cn

## ABSTRACT

Graph spectra are an important class of structural features on graphs that have shown promising results in enhancing Graph Neural Networks (GNNs). Despite their widespread practical use, the theoretical understanding of the power of spectral invariants — particularly their contribution to GNNs — remains incomplete. In this paper, we address this fundamental question through the lens of homomorphism expressivity, providing a comprehensive and quantitative analysis of the expressive power of spectral invariants. Specifically, we prove that spectral invariant GNNs can homomorphism-count exactly a class of specific tree-like graphs which we refer to as *parallel trees*. We highlight the significance of this result in various contexts, including establishing a quantitative expressiveness hierarchy across different architectural variants, offering insights into the impact of GNN depth, and understanding the subgraph counting capabilities of spectral invariant GNNs. In particular, our results significantly extend [Arvind et al.](#page-10-0) [\(2024\)](#page-10-0) and settle their open questions. Finally, we generalize our analysis to higher-order GNNs and answer an open question raised by [Zhang et al.](#page-13-0) [\(2024b\)](#page-13-0).

# 1 INTRODUCTION

The graph spectrum, defined as the eigenvalues of a graph matrix, is an important class of graph invariants. It encapsulates rich graph structural information including the graph connectivity, bipartiteness, node clustering patterns, diameter, and more [\(Brouwer & Haemers,](#page-10-1) [2011\)](#page-10-1). Besides eigenvalues, generalized spectral information may also include projection matrices, which further encodes node relations such as distances and random walk properties, enabling the definition of more fine-grained graph invariants [\(Furer](#page-11-0) ¨ , [2010\)](#page-11-0). These spectral invariants possesses strong *expressive power*. For example, a well-known conjecture raised by [Van Dam & Haemers](#page-12-0) [\(2003\)](#page-12-0); [Haemers](#page-11-1) [& Spence](#page-11-1) [\(2004\)](#page-11-1) claimed that almost all graphs can be uniquely determined by their spectra up to isomorphism. The rare exceptions, known as cospectral graphs, tend to be highly similar in their structure and continue to be an active area of research in graph theory [\(Lorenzen,](#page-12-1) [2022\)](#page-12-1).

In the machine learning community, spectral invariants have recently gained increasing popularity in designing Graph Neural Networks (GNNs) [\(Bruna et al.,](#page-10-2) [2013;](#page-10-2) [Defferrard et al.,](#page-11-2) [2016;](#page-11-2) [Lim et al.,](#page-12-2) [2023;](#page-12-2) [Huang et al.,](#page-11-3) [2024;](#page-11-3) [Feldman et al.,](#page-11-4) [2023;](#page-11-4) [Zhang et al.,](#page-13-0) [2024b;](#page-13-0) [Black et al.,](#page-10-3) [2024\)](#page-10-3), owing to several reasons. From a practical perspective, graph spectra have been shown to be closely related to certain practical applications such as molecular property prediction [\(Bonchev,](#page-10-4) [2018\)](#page-10-4). Moreover, a recent line of works [\(Xu et al.,](#page-12-3) [2019;](#page-12-3) [Morris et al.,](#page-12-4) [2019;](#page-12-4) [Li et al.,](#page-12-5) [2020;](#page-12-5) [Chen et al.,](#page-10-5) [2020;](#page-10-5) [Zhang](#page-13-1) [et al.,](#page-13-1) [2023b\)](#page-13-1) has pointed out that the expressive power of classic message-passing GNNs (MPNNs) are inherently limited, and cannot encode important graph structure like connectivity or distance. Incorporating spectral invariants into the design of MPNNs can naturally alleviate the limitations.

Therefore, from both theoretical and practical perspectives, it is beneficial to give a systematic understanding of the power of spectral invariants and their corresponding GNNs. The earliest study in this area may be traced back to [Furer](#page-11-0) ¨ [\(2010\)](#page-11-0), who first linked the power of several spectral invariants to the classic Weisfeiler-Lehman test [\(Weisfeiler & Lehman,](#page-12-6) [1968\)](#page-12-6) by proving that these invariants are upper bounded by 2-FWL. More recently, [Rattan & Seppelt](#page-12-7) [\(2023\)](#page-12-7) further revealed a strict

<sup>∗</sup> Project lead.

expressivity gap between Furer's spectral invariants and 2-FWL. ¨ [Zhang et al.](#page-13-0) [\(2024b\)](#page-13-0) and [Arvind](#page-10-0) [et al.](#page-10-0) [\(2024\)](#page-10-0) analyzed *refinement-based* spectral invariants, which offer insights into the power of real GNN architectures. Yet, all of these works study expressiveness through the lens of Weisfeiler-Lehman tests, which has inherent limitations. So far, there remains a lack of *comprehensive* understanding of the *practical* power of spectral invariants and their corresponding GNN architectures.

Current work. In this paper, we investigate the aforementioned questions via a novel perspective called *graph homomorphism*. Specifically, [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) recently proposed homomorphism expressivity as a quantitative framework to better understand the expressive power of various GNN architectures. As homomorphism expressivity is a fine-grained and practical measure, it naturally addresses several limitations of the WL test. However, extending this framework to other architectures, such as spectral invariant GNNs, poses significant challenges. In fact, whether homomorphism expressivity exists for a given architecture remains an open research direction (see [Zhang](#page-13-2) [et al.](#page-13-2) [\(2024a\)](#page-13-2)). In our context, this problem becomes even challenging since homomorphism and spectral invariants correspond to two orthogonal branches in graph theory. Here, we provide affirmative answers to all these questions by formally proving that the homomorphism expressivity for spectral invariant GNNs exists and can be elegantly characterized as a special class of *parallel trees* (Theorem [3.3\)](#page-3-0). This offers deep insights into a series of previous studies, extending their results and answering several open questions. We summarize our results below:

- Separation power of spectral invariants/GNNs. We offer a new proof that projection-based spectral invariants and corresponding GNNs are strictly bounded by 2-FWL (Corollary [3.4\)](#page-4-0). Moreover, we establish a *quantitative hierarchy* among raw spectra information, projection, refinement-based spectral invariant, and various combinatorial variants of WL tests (see Figure [4\)](#page-6-0). This (i) recovers and extends results in [Rattan & Seppelt](#page-12-7) [\(2023\)](#page-12-7), and (ii) provides clear insights into the hierarchy established in [Zhang et al.](#page-13-0) [\(2024b\)](#page-13-0).
- The power of refinement. We offer a systematic understanding of the role of refinement in spectral invariant GNNs. We show increasing the number of iterations always leads to a strict improvement in expressive power (Corollary [3.11\)](#page-5-0), thus settling a key open question raised in [Arvind et al.](#page-10-0) [\(2024\)](#page-10-0). Moreover, our counterexamples establish a tight lower bound on the number of iterations required to achieve maximal expressivity, which is in the same order of graph size. This advances a line of research regarding iteration numbers in WL tests [\(Furer](#page-11-5) ¨ , [2001;](#page-11-5) [Kiefer & Schweitzer,](#page-11-6) [2016;](#page-11-6) [Lichter et al.,](#page-12-8) [2019\)](#page-12-8).
- Substructure counting power of spectral invariants/GNNs. On the practical side, we precisely characterize the power of spectral invariants/GNNs in counting certain subgraphs as well as the required iterations. For example, they can count all cycles within 7 vertices, while using 1 iteration already suffices to count all cycles within 6 vertices (Corollary [3.15\)](#page-6-1).

Empirically, a set of experiments on both synthetic and real-world tasks validate our theoretical results, showing that the homomorphism expressivity of spectral invariant GNNs well reflects their performance in down-stream tasks.

## 2 PRELIMINARIES

Notations. We use { } and {{ }} to denote sets and multisets, respectively. The cardinality of a given (multi)set S is denoted as |S|. In this paper, we consider finite, undirected, simple graphs with no self-loops or repeated edges, and without loss of generality we only consider connected graphs. Let G = (VG, EG) be a graph with vertex set V<sup>G</sup> and edge set EG, where each edge in E<sup>G</sup> is a set {u, v} ⊂ V<sup>G</sup> of cardinality two. The *neighbors* of vertex u is denoted as NG(u) := {v ∈ VG|{u, v} ∈ EG}. A *walk* of length k is a sequence of vertices u0, · · · , u<sup>k</sup> ∈ V<sup>G</sup> such that {ui−1, ui} ∈ E<sup>G</sup> for all i ∈ [k]. It is further called a *path* if u<sup>i</sup> ̸= u<sup>j</sup> for all i < j, and it is called a *cycle* if u0, · · · , uk−<sup>1</sup> is a path and u<sup>0</sup> = uk. The shortest path distance between two nodes u, v ∈ VG, denoted as disG(u, v), is the minimum length of walk from u to v. A graph F = (V<sup>F</sup> , E<sup>F</sup> ) is a *subgraph* of G if V<sup>F</sup> ⊂ V<sup>G</sup> and E<sup>F</sup> ⊂ EG. We use P<sup>n</sup> (resp. Cn) to denote a graph corresponding to a path (resp. cycle) of n vertices. A graph is called a tree if it is connected and contains no cycle as a subgraph. We denote by T r the rooted tree T with root r. The depth of a rooted tree T r is defined as dep(T r ) = maxu∈V<sup>T</sup> dis<sup>T</sup> (r, u), and the depth of T is defined as dep(T) = minr∈V<sup>T</sup> dep(T r ).

#### 2.1 SPECTRAL INVARIANT GNNS

Let G be a graph of n vertices where V<sup>G</sup> = [n], and denote by A ∈ {0, 1} <sup>n</sup>×<sup>n</sup> the adjacency matrix of G. The *spectrum* of G is defined as the multiset of all eigenvalues of A. In addition to eigenvalues, eigenspaces also provide important spectral information. Formally, the eigenspace associated with some eigenvalue λ can be characterized by its projection matrix Pλ. It follows that there exist a unique set of orthogonal projection matrices {Pλ}λ∈Λ, where Λ is the set of all distinct eigenvalues of A, such that A = P λ∈Λ λPλ, and the following conditions hold: P <sup>λ</sup> P<sup>λ</sup> = I, PλPλ′ = 0 for λ ̸= λ ′ , and AP<sup>λ</sup> = PλA for all λ ∈ Λ. Combining the projection matrices with the associated eigenvalues naturally define an invariant between node pairs, which we denote by P:

$$\mathcal{P}(u, v) := \{\{(\lambda, \mathbf{P}_\lambda(u, v)) | \lambda \in \Lambda\}\} \quad \text{for } u, v \in V_G.$$

Then, one can define the so-called "spectral invariant" of a graph as follows. Consider the following color refinement process by treating P(u, v) as the edge feature between vertices u and v:

$$\chi_G^{\text{Spec},(d+1)}(u) = \text{hash}\left(\chi_G^{\text{Spec},(d)}(u), \{\{\chi_G^{\text{Spec},(d)}(v), \mathcal{P}(u, v)\} | v \in V_G\}\}\right) \quad \text{for } u \in V_G, d \in N_+,$$

where all colors χ Spec,(0) <sup>G</sup> (u) (u ∈ VG) are constant in initialization, and hash is a perfect hash function. For each iteration d, the mapping χ Spec,(d) <sup>G</sup> induces an equivalence relation over vertex set VG, and the relation gets *refined* with the increase of d. Therefore, with a sufficiently large number of iterations d ≤ |VG|, the relations get *stable*. The spectral invariant χ Spec,(∞) <sup>G</sup> (G) is then defined to be the multiset of stable node colors. We can similarly define χ Spec,(d) <sup>G</sup> (G) to be the multiset of node colors after d iterations [\(Arvind et al.,](#page-10-0) [2024\)](#page-10-0). We remark that χ Spec,(1) <sup>G</sup> (G) is exactly the Furer's ¨ (weak) spectral invariant proposed in [Furer](#page-11-0) ¨ [\(2010\)](#page-11-0).

Owing to the relation between GNNs and color refinement algorithms, one can easily transform the above refinement process into a GNN architecture by replacing hash function with a continuous, non-linear, parameterized function, while maintaining the same expressive power [\(Xu et al.,](#page-12-3) [2019;](#page-12-3) [Morris et al.,](#page-12-4) [2019\)](#page-12-4). We call the resulting architecture Spectral Invariant GNNs (see [Zhang et al.](#page-13-0) [\(2024b\)](#page-13-0) for concrete implementations of spectral invariant GNN layer). Without ambiguity, we may also refer to χ Spec,(d) <sup>G</sup> (G) as the graph representation computed by a d-layer spectral invariant GNN.

#### 2.2 HOMOMORPHISM EXPRESSIVITY

Given two graphs F and G, a homomorphism from F to G is a mapping f : V<sup>F</sup> → V<sup>G</sup> that preserves edge relations, i.e., {f(u), f(v)} ∈ E<sup>G</sup> for all {u, v} ∈ E<sup>F</sup> . We denote by Hom(F, G) the set of all homomorphisms from F to G and define hom(F, G) = |Hom(F, G)|, which counts the number of homomorphisms. If f is further surjective on both vertices and edges of G, we call G a *homomorphic image* of F. A mapping f : V<sup>F</sup> → V<sup>G</sup> is called an isomorphism if f is a bijection and both f and its inverse f −1 are homomorphisms. We denote by sub(F, G) the number of subgraphs of G that is isomorphic to F.

In [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), the authors introduced the concept the homomorphism expressivity to quantify the expressive power of a color refinement algorithm (or GNN). It is formally defined as follows:

Definition 2.1. Let M be a color refinement algorithm (or GNN) that outputs a graph invariant χ<sup>M</sup> <sup>G</sup> (G) given graph G. The homomorphism expressivity of M, denoted by F<sup>M</sup>, is a family of connected graphs[<sup>1</sup>](#page-2-0) satisfying the following conditions:

- a) For any two graphs G, H, χ<sup>M</sup> <sup>G</sup> (G) = χ<sup>M</sup> <sup>H</sup> (H) *iff* hom(F, G) = hom(F, H) for all F ∈ F<sup>M</sup>;
- b) F<sup>M</sup> is maximal, i.e., for any connected graph F /∈ F<sup>M</sup>, there exists a pair of graphs G, H such that χ<sup>M</sup> <sup>G</sup> (G) = χ<sup>M</sup> <sup>H</sup> (H) and hom(F, G) ̸= hom(F, H).

By characterizing the set F<sup>M</sup> for different GNN models M, one can quantitatively understand the expressivity gap between two models by simply computing their set inclusion relation and set difference. [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) examines several representative GNNs under this framework, including

<sup>1</sup> For simplicity, we focus on *connected* graphs in this paper. The results can be easily generalized to disconnected graphs following [Seppelt](#page-12-9) [\(2024\)](#page-12-9).

(a) A parallel edge with endpoints (u, v) (b) An example of parallel tree and its tree skeleton Figure 1: Illustration of a parallel edge with endpoints (u, v) in (a) and a parallel tree with its skeleton on the right in (b).

the standard MPNNs and Folklore GNNs [\(Maron et al.,](#page-12-10) [2019;](#page-12-10) [Azizian & Lelarge,](#page-10-6) [2021\)](#page-10-6), and recent architectures such as Subgraph GNN [\(Bevilacqua et al.,](#page-10-7) [2022;](#page-10-7) [Qian et al.,](#page-12-11) [2022;](#page-12-11) [Cotta et al.,](#page-10-8) [2021\)](#page-10-8) and Local GNN [\(Morris et al.,](#page-12-12) [2020;](#page-12-12) [Zhang et al.,](#page-13-3) [2023a\)](#page-13-3). However, one implicit challenge not reflected in Definition [2.1\(](#page-2-1)a) is that the set F<sup>M</sup> may not even exist for a general GNN M. Proving the existence corresponds to an involved research topic known as homomorphism distinguishing closedness [\(Roberson,](#page-12-13) [2022;](#page-12-13) [Seppelt,](#page-12-9) [2024;](#page-12-9) [Neuen,](#page-12-14) [2023\)](#page-12-14), which is highly non-trivial. In the next section, we will give affirmative results showing that the homomorphism expressivity of spectral invariant GNNs does exist and give an elegant description of the graph family.

## 3 HOMOMORPHISM EXPRESSIVITY OF SPECTRAL INVARIANT GNNS

In this section, we investigate the homomorphism expressivity of spectral invariants and the corresponding GNNs. We will provide a complete characterization of the set F Spec,(d) for arbitrary model depth d ∈ N ∪ {∞}. This allows us to analyze spectral invariants in a novel perspective, significantly extending prior research and resolving previously unanswered questions.

#### 3.1 MAIN RESULTS

Our idea is motivated by the previous finding that the homomorphism expressivity of MPNNs is exactly the family of all trees [\(Zhang et al.,](#page-13-2) [2024a\)](#page-13-2). Note that in the definition of spectral invariant GNN, if one replaces P(u, v) by the standard adjacency Auv, the resulting architecture is just an MPNN. Such a relationship perhaps implies that the homomorphism expressivity of spectral invariant GNNs also comprises "tree-like" graphs. We will show this is indeed true. To present our results, let us define a special class of graphs, referred to as *parallel trees*:

Definition 3.1 (Parallel Edge). A graph G is called a *parallel edge* if there exist two different vertices u, v ∈ V<sup>G</sup> such that the edge set E<sup>G</sup> can be partitioned into a sequence of simple paths P1, . . . , Pm, where all paths share endpoints (u, v). We refer to (u, v) as the endpoints of G.

Definition 3.2 (Parallel Tree). A graph F is called a *parallel tree* if there exists a tree T such that F can be obtained from T by replacing each edge {u, v} ∈ E<sup>T</sup> with a parallel edge that has endpoints {u, v}. We refer to T as the *parallel tree skeleton* of graph F. Given a parallel tree F, define the *parallel tree depth* of F as the minimum depth of any parallel tree skeleton of F.

We give an illustration of parallel edge and parallel tree in Figure [1.](#page-3-1) With the above definitions, we are ready to state our main theorem:

Theorem 3.3. *For any* d ∈ N*, the homomorphism expressivity of spectral invariant GNNs with* d *iterations exists and can be characterized as follows:*

$$\mathcal{F}^{\text{Spec},(d)} = \{F \mid F \text{ has parallel tree depth at most } d\}.$$

*Specifically, the following properties hold:*

- *Given any graphs* G *and* H*,* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) *if and only if, for all connected graphs* F *with parallel tree depth at most* d*,* hom(F, G) = hom(F, H)*.*
- F Spec,(d) *is maximal; that is, for any connected graph* F /∈ FSpec,(d) *, there exist graphs* G *and* H *such that* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) *and* hom(F, G) ̸= hom(F, H)*.*

We will present a concise proof sketch of Theorem [3.3](#page-3-0) in Section [3.3.](#page-6-2) Next, in Section [3.2,](#page-4-1) we will interpret this result in the context of GNNs and discuss its significance, including how it extends previous findings and addresses open problems identified in earlier studies.

#### 3.2 IMPLICATIONS

Our theory has a wide range of applications, which will be separately discussed in detail below.

#### 3.2.1 COMPARISON WITH 2-FWL

Firstly, we compare the expressive power of spectral invariant GNNs with the expressive power of the standard Weisfeiler-Lehman (WL) test. It immediately follows that the expressive power of spectral invariant GNNs strictly lies between the expressive power of 1-WL and 2-FWL test.

![](_page_4_Diagram_6.jpeg)

Figure 2: A counterexample graph in F <sup>2</sup>−FWL\FSpec,(∞) .

Corollary 3.4. *The expressive power of spectral invariant GNNs is strictly stronger than* 1*-WL and strictly weaker than* 2*-FWL.*

*Proof.* According to [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), the homomorphism expressivity of 2-FWL encompasses the set of all graphs with treewidth at most 2. A classical result in graph theory states that any subgraph of any series-parallel graph has treewidth at most 2 [\(Diestel,](#page-11-7) [2017\)](#page-11-7). Since any parallel tree is clearly a subgraph of some series-parallel graph, its treewidth is at most 2. It follows that the homomorphism expressivity of parallel trees is contained within that of the 2-FWL. To show the gap, we give a counterexample graph in Figure [2.](#page-4-2) This implies that the expressive power of spectral invariant GNNs is strictly weaker than that of the 2-FWL. The proof for the case of 1-WL is similar and we omit it for clarity.

#### 3.2.2 HIERARCHY

Theorem [3.3](#page-3-0) not only provides insights into the relationship between the expressive power of spectral invariant GNNs and 2-FWL, but also allows for a comparison with a wide range of graph invariants and the corresponding GNNs. Specifically, similar to the analysis in Corollary [3.4,](#page-4-0) for any GNN models A and B such that their homomorphism expressivity exists, if F <sup>A</sup> ⊊ F <sup>B</sup>, then A is strictly weaker than B in expressive power. We now use this property to establish a comprehensive hierarchy by linking spectral invariant GNNs to other fundamental graph invariants and GNNs.

Corollary 3.5. *Spectral invariant GNN with* 1 *iteration is strictly weaker than subgraph GNN (also referred to as* (1, 1)*-WL in [Rattan & Seppelt](#page-12-7) [\(2023\)](#page-12-7)).*

*Proof.* According to [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), the homomorphism expressivity of subgraph GNNs contains all graphs that become a forest upon the deletion of a specific vertex. On the other hand, Theorem [3.3](#page-3-0) states that the homomorphism expressivity of spectral invariant GNNs with one iteration contains all parallel trees of depth 1. Since any parallel tree of depth 1 becomes a forest when deleting the root vertex, we have proved that F Spec,(1) is a subset of that of subgraph GNNs. Finally, one can easily construct a counterexample graph to prove the strict separation.

Remark 3.6. Our result recovers and strengthens the main result in [Rattan & Seppelt](#page-12-7) [\(2023\)](#page-12-7), which only studied spectral invariants with 1 iteration (Furer's weak spectral invariant). We will next show ¨ this result actually does *not* hold in case of more than 1 iterations.

Corollary 3.7. *Spectral invariant GNNs with* 2 *iterations are incomparable to subgraph GNNs.*

We provide a counterexample in Figure [3.](#page-5-1) Nevertheless, we can still bound the expressive power of spectral invariant GNNs with multiple iterations to that of Local 2-GNN, as stated in the following:

Corollary 3.8. *For any* d ∈ <sup>N</sup>+∪{∞}*, spectral invariant GNNs with* d *iterations are strictly weaker than Local 2-GNN [\(Morris et al.,](#page-12-12) [2020;](#page-12-12) [Zhang et al.,](#page-13-2) [2024a\)](#page-13-2).*

*Proof.* According to [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), the homomorphism expressivity of Local 2-GNNs contains all graphs that admit a strong nested ear decomposition. Since any parallel edge can be partitioned into ears with the same endpoints, one can easily construct a nested ear decomposition for any parallel tree. This shows F Spec,(d) is a subset of that of Local 2-GNN. The expressivity gap can be seen using the same counterexample graph in Figure [2.](#page-4-2)

![](_page_5_Diagram_1.jpeg)

Remark 3.9. Corollaries [3.7](#page-4-3) and [3.8](#page-4-4) significantly extend the findings of [Arvind et al.](#page-10-0) [\(2024,](#page-10-0) Theorem 17) and provide additional insights into [Zhang et al.](#page-13-0) [\(2024b,](#page-13-0) Theorem 4.3).

The power of projection. We next conduct a fine-grained analysis by separating eigenvalues and projections to better understand their individual contributions to enhancing the expressive power of GNN models. We first prove the following theorem:

Theorem 3.10. *The homomorphism expressivity of graph spectra is the set of all cycles* C<sup>n</sup> *(*n ≥ 3*) plus paths* P<sup>1</sup> *and* P2*, i.e.,* {Cn|n ≥ 3} ∪ {P1, P2}*.*

The proof of Theorem [3.10](#page-5-2) is provided in Appendix [C,](#page-33-0) which has the same structure as that of Theorem [3.3.](#page-3-0) Previously, [Van Dam & Haemers\(2003\)](#page-12-0); [Dell et al.](#page-11-8) [\(2018\)](#page-11-8) have proved that the spectra of two graphs G and H are identical if and only if for every cycle F, hom(F, G) = hom(F, H). We extend their result by further proving the maximal property (Definition [2.1\(](#page-2-1)b)), which only adds two trivial graphs P<sup>1</sup> and P<sup>2</sup> to the homomorphism expressivity. From this result, one can easily see that using eigenvalues alone can already improve the expressive power of an MPNN since the homomorphism expressivity of MPNN contains only trees (but not cycles).

To understand the role of projection, one can compare the set {Cn|n ≥ 3}∪ {P1, P2} with F Spec,(1) (the homomorphism expressivity of Furer's spectral invariant). Clearly, the set of all parallel trees of ¨ depth 1 is strictly larger than {Cn|n ≥ 3}∪ {P1, P2}, confirming that adding projection information significantly enhances the expressive power beyond graph spectra.

The power of refinement. We finally investigate the power of iterations d (or number of GNN layers) in enhancing the model's expressive power. We have the following result:

Corollary 3.11. *For any* d ∈ N*, spectral invariant GNNs with* d + 1 *iterations are strictly more powerful than spectral invariant GNNs with* d *iterations.*

*Proof.* For any k ∈ N, we can construct a counterexample formed by replacing each edge in the path graph P2k+2 with a parallel edge. We illustrate the construction in Figure [3\(](#page-5-1)b). One can easily see that the resulting graph is in F Spec,(k+1) but not F Spec,(k) .

Remark 3.12. Corollary [3.11](#page-5-0) addresses the key open question posed in [Arvind et al.](#page-10-0) [\(2024\)](#page-10-0), who conjectured that spectral invariant GNNs converge within *constant* iterations. Specifically, the authors questioned whether, for d ≥ 4, spectral invariant GNNs with d + 1 iterations are as powerful as those with d iterations. We disproved this conjecture by providing a family of example graphs that cannot be distinguished in d iterations but can be distinguished in d + 1 iterations.

Our counterexamples further leads to the following result:

Corollary 3.13. *For any* d ∈ <sup>N</sup>+*, There exist two graphs with* O(d) *vertices such that spectral invariant GNNs require at least* d *iterations to distinguish between them.*

Corollary [3.13](#page-5-3) establishes a tight bound on the number of layers needed for spectral invariant GNNs to reach maximal expressivity, showing that it scales with the order of graph size. This advances an important research topic that aims to study the relation between expressiveness and iteration number of color refinement algorithms [\(Furer](#page-11-5) ¨ , [2001;](#page-11-5) [Kiefer & Schweitzer,](#page-11-6) [2016;](#page-11-6) [Lichter et al.,](#page-12-8) [2019\)](#page-12-8).

To summarize all the above results, we illustrate the hierarchy established for spectral invariant GNNs and other mainstream GNNs in Figure [4.](#page-6-0)

## 3.2.3 SUBGRAPH COUNT

In fact, our results can go beyond the WL framework and reveal the expressive power of spectral invariant GNNs in a more practical perspective. As an example, we will show below how Theorem [3.3](#page-3-0) can be used to understand the subgraph counting capabilities of spectral invariant GNNs.

![](_page_6_Diagram_1.jpeg)

Figure 4: Hierarchy of spectral invariant GNN (abbreviated as Spectral IGN) and other mainstream GNNs. Each arrow points to the strictly stronger architecture.

Given any graph F, we say a GNN model M can subgraph-count substructure F if for any graphs G and H, the condition χ<sup>M</sup> <sup>G</sup> (G) = χ<sup>M</sup> <sup>H</sup> (H) implies sub(F, G) = sub(F, H). Denote by Spasm(F) the set of all homomorphic images of F. Previous results have proved that, if the homomorphism expressivity F<sup>M</sup> exists for model M, then M can subgraph-count F if and only if Spasm(F) ⊂ F<sup>M</sup> [\(Seppelt,](#page-12-15) [2023;](#page-12-15) [Zhang et al.,](#page-13-2) [2024a\)](#page-13-2). This allows us to precisely analyze which substructure can be subgraph-counted by spectral invariant GNNs.

Corollary 3.14. *Spectral invariant GNN can count cycles and paths with up to* 7 *vertices.*

*Proof.* For cycles or paths with at most 7 vertices, one can check by enumeration that their homomorphic images are all parallel trees. For cycles or paths with at least 8 vertices, the 4-clique is a valid homomorphic image but is not a parallel tree.

We can further strengthen the above results by studying the number of iterations needed to count substructures. We have the following results:

Corollary 3.15. *The following holds:*

- *1. Spectral invariant GNNs can subgraph-count all cycles up to* 7 *vertices within* 2 *iterations.*
- *2. The above upper bound is tight: spectral invariant GNNs with only* 1 *iteration (i.e., Furer's ¨ weak spectral invariant) cannot subgraph-count* 7*-cycle.*
- *3. Spectral invariant GNNs with* 1 *iteration suffice to subgraph-count all cycles up to* 6 *vertices.*

Remark 3.16. The subgraph counting power of spectral invariant has long been studied in the literature. [Cvetkovic et al.](#page-10-9) [\(1997\)](#page-10-9) proved that the graph angles (which can be determined by projection) can subgraph-count all cycles of length no more than 5. In comparison, our results significantly extend their findings, which even match the cycle counting power of 2-FWL [\(Arvind et al.,](#page-10-10) [2020\)](#page-10-10). Moreover, we show that Furer's weak spectral invariant can already count ¨ 6-cycles, thus extending the work of [Furer](#page-11-9) ¨ [\(2017\)](#page-11-9).

#### 3.3 PROOF SKETCH

In this section, we provide a proof sketch of Theorem [3.3,](#page-3-0) with the complete proof presented in the Appendix. We begin by demonstrating that the information encoded by spectral invariants is closely related to encoding *walk information* in the aggregation process of GNNs. This corresponds to the following lemma (proved in Appendix [B.2,](#page-17-0) see also [Arvind et al.](#page-10-0) [\(2024\)](#page-10-0)):

Lemma 3.17. *(Equivalence of encoding walk and encoding spectral information) Let* G = (VG, EG) *be a graph, with its adjacency matrix denoted by* A*. For vertices* x, y ∈ VG*, define* ω k <sup>G</sup>(x, y) = A<sup>k</sup> x,y *for all* k ∈ {0, 1, 2, . . . , |VG|}*, which represents the number of* k*-walks from vertex* x *to vertex* y*. Define the tuple* ω ∗ <sup>G</sup>(x, y) = (ω 0 G(x, y), ω<sup>1</sup> G(x, y), . . . , ω<sup>n</sup>−<sup>1</sup> <sup>G</sup> (x, y))*, where* n = |VG|*. Define the walk-encoding GNN with the following update rule:*

$$\chi_G^{\text{Walk},(d+1)}(x) = \text{hash}(\chi_G^{\text{Walk},(d)}(x), \{\{(\omega_G^*(x, y), \chi_G^{\text{Walk},(d)}(y)) \mid y \in V_G\}\}).$$

*The walk-encoding GNN outputs a representation* χ Walk,(d) <sup>G</sup> (G) = {{χ Walk,(d) <sup>G</sup> (u)|u ∈ VG}}*. For any graphs* G*,* H*, we have* χ Walk,(d) <sup>G</sup> (G) = χ Walk,(d) <sup>H</sup> (H) *if and only if* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H)*.*

Our next step aims to prove that for graphs G and H, χ Walk,(d) <sup>G</sup> (G) = χ Walk,(d) <sup>H</sup> (H) iff, for all graphs F with parallel tree depth at most d, hom(F, G) = hom(F, H). This will yield the first property outlined in Theorem [3.3.](#page-3-0) The proof has a similar structure to that in [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), which is based on the tools of tree-decomposed graphs and algebraic graph theory (see Theorems [B.14](#page-21-0) and [B.20](#page-24-0) and Lemma [B.17\)](#page-22-0). This part corresponds to Appendix [B.3.](#page-19-0)

Now, it remains to prove that the set F Spec,(d) is maximal (the second property in Theorem [3.3\)](#page-3-0). To achieve this, we leverage the technique known as pebble game [\(Cai et al.,](#page-10-11) [1992\)](#page-10-11), which was originally used to construct counterexample graphs that cannot be distinguished by the k-FWL test. We extend the framework and define the pebble game for spectral invariant GNNs as follows:

Definition 3.18. (Pebble game for spectral invariant GNNs) The pebble game is conducted on two graphs G = (VG, EG) and H = (VH, EH). Without loss of generality, we assume V<sup>G</sup> = VH. Initially, each graph is equipped with two distinct pebbles denoted as u and v, which initially lie outside the graphs. The game involves two players: the spoiler and the duplicator. The game process is described as follows:

- *Initialization:* The spoiler first selects a non-empty subset V S from either V<sup>G</sup> or VH, and the duplicator responds with a subset V <sup>D</sup> from the other graph, ensuring that |V <sup>D</sup>| = |V S |. Then, the spoiler places the pebble u on some vertex in V <sup>D</sup>, and the duplicator places the corresponding pebble u on some vertex in V S . Similarly, the spoiler and duplicator repeat the process to place two pebbles v. After the initialization, all pebbles will lie on the two graphs.
- *Main Process:* The game iteratively repeats the following steps, where in each iteration the spoiler may choose freely between the following two actions:
  - 1. Action 1 (moving pebble v). The spoiler first selects a non-empty subset V S from either V<sup>G</sup> or VH, and the duplicator responds with a subset V <sup>D</sup> from the other graph, ensuring that |V <sup>D</sup>| = |V S |. The spoiler then moves pebble v to some vertex in V <sup>D</sup>, and the duplicator moves the corresponding pebble v to some vertex in V S .
  - 2. Action 2 (moving pebble u). This action is similar to the above one except that both players move pebble u instead of pebble v.
- *Termination:* The spoiler wins if, after a certain number of rounds, ω ⋆ <sup>G</sup>(u, v) for graph G differs from ω ⋆ <sup>H</sup>(u, v) for graph H. Conversely, the duplicator wins if the spoiler is unable to win after any number of rounds.

With the above definition, we can now prove the equivalence between the outcome of a pebble game and the ability to distinguish non-isomorphic graphs using spectral invariant GNNs:

Lemma 3.19. *(Equivalence of pebble game and spectral invariant GNNs) Given graphs* G *and* H *and the number of steps* d ∈ N*, the spoiler cannot win the pebble game in* d *steps iff* χ Spec,(d+1) <sup>G</sup> (G) = χ Spec,(d+1) <sup>H</sup> (H)*.*

We give a proof in Appendix [B.4.](#page-25-0) Next, to identify counterexamples G and H for any F /∈ FSpec,(d) such that χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) and hom(F, G) ̸= hom(F, H), we draw inspiration from a special class of graphs called Furer graphs ( ¨ [Furer](#page-11-5) ¨ , [2001\)](#page-11-5), which is a principled approach to constructing pairs of non-isomorphic but structurally similar graphs. If graphs G and H are the Furer ¨ graph and twisted Furer graph constructed from the same base graph ¨ F, we show that the pebble game can be significantly simplified. Importantly, the simplified pebble game will be played on the base graph F instead of the complex Furer graphs, making the subsequent analysis much easier. ¨ Due to space constraints, a detailed description of the simplified pebble game is provided in Appendix [B.5.](#page-27-0) We then establish the following lemma, which relates the simplified pebble game to spectral invariant GNNs:

Lemma 3.20. *(Equivalence of pebble game on Furer graphs and spectral invariant GNNs ¨ ) Given a base graph* F*, let* G(F) *and* H(F) *be the Furer graph and twisted F ¨ urer graph of ¨* F*, respectively. Then, the spoiler cannot win the simplified pebble game on* F *in* d *steps iff* χ Spec,(d+1) <sup>G</sup> (G(F)) = χ Spec,(d+1) <sup>H</sup> (H(F))*.*

Note that for any connected graph F, hom(F, G(F)) ̸= hom(F, H(F)) [\(Roberson,](#page-12-13) [2022;](#page-12-13) [Zhang](#page-13-2) [et al.,](#page-13-2) [2024a\)](#page-13-2). Furthermore, we demonstrate that the spoiler has a winning strategy on F in d steps if and only if F is a parallel tree with parallel tree depth at most d + 1 (see Appendix [B.6\)](#page-30-0). By combining these results with Lemma [3.20,](#page-7-0) we establish the following lemma:

Lemma 3.21. *For any* F /∈ FSpec,(d) *, the spoiler cannot win the simplified pebble game on* F*. Consequently,* χ Spec,(d) <sup>G</sup> (G(F)) = χ Spec,(d) <sup>H</sup> (H(F))*.*

This yields the second property in Theorem [3.3](#page-3-0) and concludes the proof.

#### 3.4 EXTENSIONS

So far, this paper mainly analyzes the standard spectral invariant GNNs, which refines *node features* based on projection information. In this subsection, we will show the flexibility of our proposed homomorphism expressivity framework, which can also be used to analyze other spectral-based GNN models such as higher-order spectral invariant GNNs.

#### 3.4.1 HIGHER ORDER

Let us consider generalizing Section [2.1](#page-2-2) to higher order spectral invariant GNNs. A natural update rule of higher order spectral invariant GNN can be defined as follows:

Definition 3.22 (Higher-Order Spectral Invariant GNN). For any k ∈ <sup>N</sup>+, the k-order spectral invariant GNN maintains a color χ k-Spec <sup>G</sup> (u) for each vertex k-tuple u = (u1, . . . , uk) ∈ V k G. Initially, χ k-Spec,(0) <sup>G</sup> (u) = (P(u1, u2), . . . ,P(u1, uk), . . . ,P(uk−1, uk)). In each iteration t + 1, the color is updated as follows:

$$\chi_G^{k\text{-Spec},(t+1)}(\mathbf{u}) = \text{hash}(\chi_G^{k\text{-Spec},(t)}(\mathbf{u}), \{\{(\chi_G^{k\text{-Spec},(t)}(v, u_2, \dots, u_k), \mathcal{P}(u_1, v)) : v \in V_G\}\}, \dots, \{\{(\chi_G^{k\text{-Spec},(t)}(u_1, u_2, \dots, u_{k-1}, v), \mathcal{P}(u_k, v)) : v \in V_G\}\}).$$

Denote the stable color of vertex tuple u ∈ V k <sup>G</sup> as χ k-Spec <sup>G</sup> (u). The graph representation is defined as χ k-Spec <sup>G</sup> (G) := {{χ k-Spec <sup>G</sup> (u) : u ∈ V k <sup>G</sup>}}.

One can see that when k = 1, the above definition degenerates to the standard spectral invariant GNN defined in Section [2.1.](#page-2-2) To illustrate the homomorphism expressivity of higher-order spectral invariant GNNs, we extend the concept of strong nested ear decomposition (NED) introduced by [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) and define the parallel strong NED. Our main result is stated below:

Theorem 3.23 (informal). *A graph* F *is said to have a parallel* k*-order strong nested ear decomposition (NED) if there exists a graph* G *such that* G *admits a strong NED and* F *can be obtained from* G *by replacing each edge* {u, v} ∈ E<sup>G</sup> *with a parallel edge that has endpoints* (u, v)*. Then, the homomorphism expressivity of* k*-order spectral invariant GNNs is the set of all graphs that admit a parallel* k*-order strong NED.*

Due to space constraints, we leave the formal definition of k-order strong NED and the technical proof of Theorem [3.23](#page-8-0) to the Appendix.

#### 3.4.2 SYMMETRIC POWER

To generalize spectrum and projection to higher order, another classic approach in the literature is to use the symmetric power of a graph (also called the *token graph*). [Audenaert et al.](#page-10-12) [\(2005\)](#page-10-12) first introduced the graph symmetric power to generalize eigenvalues into higher-order graph invariants. The formal definition of the symmetric k-th power is presented as follows:

Definition 3.24 (Symmetric Power). For any k ∈ <sup>N</sup><sup>+</sup> and graph G, the symmetric k-th power of G, denoted by G{k} , is a graph where its vertices are k-subsets of VG, and two subsets are adjacent if and only if their symmetric difference is an edge in G.

Our homomorphism expressivity framework can be used to study the ability of mainstream GNNs to encode the symmetric power of graphs. Our main result is stated as follows:

Theorem 3.25. *The Local* 2k*-GNN defined in [Morris et al.](#page-12-12) [\(2020\)](#page-12-12); [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) can encode the symmetric* k*-th power. Specifically, for given graphs* G *and* H*, if* G *and* H *have the same representation under Local* 2k*-GNN, then* G{k} *and* H{k} *have the same representation under the spectral invariant GNN defined in Section [2.1.](#page-2-2)*

Discussions with prior work. Regarding the expressive power of symmetric power, [Alzaga et al.](#page-10-13) [\(2008\)](#page-10-13); [Barghi & Ponomarenko](#page-10-14) [\(2009\)](#page-10-14) gave the first upper bound, showing that if 2k-FWL fails to distinguish between two non-isomorphic graphs, then their symmetric k-th powers are cospectral.

Table 1: Experimental results on homomorphism counting, real-world tasks and substructure count.

| Task Model             |     |     |     | Count |     | Subset | ZINC | Full  |     |     |     | Substructure | Count |     |     |
|------------------------|-----|-----|-----|-------|-----|--------|------|-------|-----|-----|-----|--------------|-------|-----|-----|
| MPNN                   | 300 | 261 | 276 | 233   | 341 | 138 ±  | 006  | 030 ± | 002 | 358 | 208 | 188          | 146   | 261 | 205 |
| Spectral Invariant GNN | 045 | 046 | 053 | 048   | 303 | 103 ±  | 006  | 028 ± | 003 | 072 | 072 | 089          | 089   | 060 | 099 |
| Subgraph GNN           | 011 | 013 | 010 | 015   | 260 | 110 ±  | 007  | 028 ± | 002 | 010 | 020 | 024          | 046   | 007 | 027 |
| Local 2-GNN            | 008 | 006 | 008 | 008   | 112 | 069 ±  | 001  | 024 ± | 002 | 008 | 011 | 017          | 034   | 007 | 016 |

However, it remains unclear whether the conclusion extends to the more powerful projection information (beyond eigenvalues), or if the stated upper bound is tight. These open questions are further highlighted in [Zhang et al.](#page-13-0) [\(2024b\)](#page-13-0). Our result answers both questions by bounding the stronger refinement-based spectral invariant for the k-th symmetric power graphs to Local 2k-GNN, which is strictly weaker than 2k-FWL [\(Zhang et al.,](#page-13-2) [2024a\)](#page-13-2). This offers a deeper understanding of the capability of mainstream GNNs in encoding higher-order spectral information.

## 4 EXPERIMENT

In this section, we validate our theoretical findings through empirical experiments. We evaluate the performance of GNN models on both synthetic and real-world tasks. For the synthetic tasks, we assess the homomorphic counting power and subgraph counting power of the GNN models. These experiments serve to confirm our theoretical results, including Theorem [3.3](#page-3-0) and Corollary [3.14.](#page-6-3) In addition, for the real-world task, we focus on molecular reaction prediction, specifically evaluating GNN performance on the ZINC dataset [\(Dwivedi et al.,](#page-11-10) [2020\)](#page-11-10). Our primary objective is not to achieve SOTA results but to validate our theoretical findings. We compare the performance of spectral invariant GNNs to both MPNNs and subgraph GNNs on the ZINC dataset. Details about model architectures are in Appendix [D.](#page-34-0)

Homomorphism Count We use the benchmark dataset from [Zhao et al.](#page-13-4) [\(2022\)](#page-13-4) to evaluate the homomorphism expressivity of four mainstream GNN models. The reported performance is measured by the normalized Mean Absolute Error (MAE) on the test set. The empirical results are presented in Table [1.](#page-9-0) We can see that concerning homomorphism: (i) MPNN is unable to encode any of the five substructures, and none of the five substructures is a tree; (ii) Spectral invariant GNN can only encode the 1st and 2nd substructures; (iii) Subgraph GNN can encode the 1st, 2nd, and 3rd substructures; and (iv) Local 2-GNN can encode the 1st, 2nd, 3rd, and 4th substructures. The empirical results basically align with our theoretical findings.

Subgraph Count Cycle counting is a fundamental problem in chemical and biological tasks. Following the settings in [Frasca et al.](#page-11-11) [\(2022\)](#page-11-11); [Zhang et al.](#page-13-3) [\(2023a\)](#page-13-3); [Huang et al.](#page-11-12) [\(2023\)](#page-11-12), we evaluate the cycle counting power of four GNNs. The empirical results in Table [1](#page-9-0) demonstrate that the spectral invariant GNN can accurately count 3-, 4-, 5-, and 6-cycles, indicating its strong performance in cycle counting tasks. This empirical result is also consistent with our theoretical predictions.

Real-World Task We evaluate our GNN models on the ZINC-subset and ZINC-full dataset [\(Dwivedi et al.,](#page-11-10) [2020\)](#page-11-10). Following the standard configuration, all models are constrained to a 500K parameter budget. The results show that the spectral invariant GNN outperforms MPNN while demonstrating comparable performance to the subgraph GNN on the real-world task. These findings are consistent with our theoretical predictions.

# 5 CONCLUSION

In this work, we investigate the expressive power of spectral invariant graph neural networks (GNNs). By leveraging the framework of homomorphism expressivity, we give a precise characterization the homomorphism expressivity of these networks. We then establish a comprehensive hierarchy of spectral invariant GNNs relative to other mainstream GNNs based on their homomorphism expressivity. Additionally, we analyze the subgraph counting capabilities of spectral invariant GNNs, with a focus on their ability to count essential substructures. Our results are extended to higher-order contexts and address additional problems related to spectral structures using our homomorphism framework. We demonstrate the significance of our findings by showing how our results extend previous work and address open problems identified in the literature. Finally, we conduct experiments to validate our theoretical results.

# ACKNOWLEDGEMENTS

This work is supported by National Science and Technology Major Project (2022ZD0114902)and National Science Foundation of China (NSFC92470123, NSFC62276005).

# REFERENCES


[1] Alfredo Alzaga, Rodrigo Iglesias, and Ricardo Pignol. Spectra of symmetric powers of graphs and the weisfeiler-lehman refinements, 2008. URL <https://arxiv.org/abs/0801.2322>. Vikraman Arvind, Frank Fuhlbruck, Johannes K ¨ obler, and Oleg Verbitsky. On weisfeiler-leman ¨ invariance: Subgraph counts and related graph properties. *Journal of Computer and System Sciences*, 113:42–59, 2020. Vikraman Arvind, Frank Fuhlbruck, Johannes K ¨ obler, and Oleg Verbitsky. On a hierarchy of spec- ¨ tral invariants for graphs. In *41st International Symposium on Theoretical Aspects of Computer Science (STACS 2024)*. Schloss Dagstuhl–Leibniz-Zentrum fur Informatik, 2024. ¨ Koenraad Audenaert, Chris Godsil, Gordon Royle, and Terry Rudolph. Symmetric squares of graphs, 2005. URL <https://arxiv.org/abs/math/0507251>. Waiss Azizian and Marc Lelarge. Expressive power of invariant and equivariant graph neural networks. In *International Conference on Learning Representations*, 2021. Muhammet Balcilar, Pierre Heroux, Benoit Gauzere, Pascal Vasseur, S ´ ebastien Adam, and Paul ´ Honeine. Breaking the limits of message passing graph neural networks. In *International Conference on Machine Learning*, pp. 599–608. PMLR, 2021. Amir Rahnamai Barghi and Ilya Ponomarenko. Non-isomorphic graphs with cospectral symmetric powers. *the electronic journal of combinatorics*, pp. R120–R120, 2009. Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai, Gopinath Balamurugan, Michael M Bronstein, and Haggai Maron. Equivariant subgraph aggregation networks. In *International Conference on Learning Representations*, 2022. Mitchell Black, Zhengchao Wan, Gal Mishne, Amir Nayyeri, and Yusu Wang. Comparing graph transformers via positional encodings. *arXiv preprint arXiv:2402.14202*, 2024. Danail Bonchev. *Chemical graph theory: introduction and fundamentals*. Routledge, 2018. Andries E Brouwer and Willem H Haemers. *Spectra of graphs*. Springer Science & Business Media, 2011. Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. *arXiv preprint arXiv:1312.6203*, 2013. Jin-Yi Cai, Martin Furer, and Neil Immerman. An optimal lower bound on the number of variables ¨ for graph identification. *Combinatorica*, 12(4):389–410, 1992. Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count substructures? In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, pp. 10383–10395, 2020. Leonardo Cotta, Christopher Morris, and Bruno Ribeiro. Reconstruction for powerful graph representations. In *Advances in Neural Information Processing Systems*, volume 34, pp. 1713–1726, 2021. Radu Curticapean, Holger Dell, and Daniel Marx. Homomorphisms are a good basis for counting ´ small subgraphs. In *Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing*, pp. 210–223, 2017. Dragos Cvetkovic, Dragos M Cvetkovi ˇ c, Peter Rowlinson, and Slobodan Simic. ´ *Eigenspaces of graphs*. Cambridge University Press, 1997.

[2] Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on ¨ graphs with fast localized spectral filtering. In *Advances in neural information processing systems*, volume 29, 2016. Holger Dell, Martin Grohe, and Gaurav Rattan. Lovasz meets weisfeiler and leman. In ´ *45th International Colloquium on Automata, Languages, and Programming (ICALP 2018)*, volume 107, pp. 40. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018. Reinhard Diestel. *Graph Theory*. Springer Publishing Company, Incorporated, 5th edition, 2017. ISBN 3662536218. Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. *arXiv preprint arXiv:2003.00982*, 2020. Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. *arXiv preprint arXiv:2110.07875*, 2021. Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. *Journal of Machine Learning Research*, 24(43):1–48, 2023. Or Feldman, Amit Boyarski, Shai Feldman, Dani Kogan, Avi Mendelson, and Chaim Baskin. Weisfeiler and leman go infinite: Spectral and combinatorial pre-colorings. *Transactions on Machine Learning Research*, 2023. ISSN 2835-8856. Fabrizio Frasca, Beatrice Bevilacqua, Michael M Bronstein, and Haggai Maron. Understanding and extending subgraph gnns by rethinking their symmetries. In *Advances in Neural Information Processing Systems*, 2022. Martin Furer. Weisfeiler-lehman refinement requires at least a linear number of iterations. In ¨ *International Colloquium on Automata, Languages, and Programming*, pp. 322–333. Springer, 2001. Martin Furer. On the power of combinatorial and spectral invariants. ¨ *Linear algebra and its applications*, 432(9):2373–2380, 2010. Martin Furer. On the combinatorial power of the weisfeiler-lehman algorithm. In ¨ *International Conference on Algorithms and Complexity*, pp. 260–271. Springer, 2017. Floris Geerts and Juan L Reutter. Expressiveness and approximation properties of graph neural networks. In *International Conference on Learning Representations*, 2022. Willem H Haemers and Edward Spence. Enumeration of cospectral graphs. *European Journal of Combinatorics*, 25(2):199–211, 2004. Yinan Huang, Xingang Peng, Jianzhu Ma, and Muhan Zhang. Boosting the cycle counting power of graph neural networks with i\$ˆ2\$-GNNs. In *The Eleventh International Conference on Learning Representations*, 2023. Yinan Huang, William Lu, Joshua Robinson, Yu Yang, Muhan Zhang, Stefanie Jegelka, and Pan

[3] Li. On the stability of expressive positional encodings for graphs. In *The Twelfth International Conference on Learning Representations*, 2024. Charilaos Kanatsoulis and Alejandro Ribeiro. Counting graph substructures with graph neural networks. In *The Twelfth International Conference on Learning Representations*. Sandra Kiefer and Pascal Schweitzer. Upper bounds on the quantifier depth for graph differentiation in first order logic. In *Proceedings of the 31st Annual ACM/IEEE Symposium on Logic in Computer Science*, pp. 287–296, 2016. Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Letourneau, and Prudencio Tossou. Re- ´ thinking graph transformers with spectral attention. In *Advances in Neural Information Processing Systems*, volume 34, 2021.

[4] Ron Levie, Federico Monti, Xavier Bresson, and Michael M Bronstein. Cayleynets: Graph convolutional neural networks with complex rational spectral filters. *IEEE Transactions on Signal Processing*, 67(1):97–109, 2018. Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec. Distance encoding: design provably more powerful neural networks for graph representation learning. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, pp. 4465–4478, 2020. Moritz Lichter, Ilia Ponomarenko, and Pascal Schweitzer. Walk refinement, walk logic, and the iteration number of the weisfeiler-leman algorithm. In *2019 34th Annual ACM/IEEE Symposium on Logic in Computer Science (LICS)*, pp. 1–13. IEEE, 2019. Derek Lim, Joshua David Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie Jegelka. Sign and basis invariant networks for spectral graph representation learning. In *The Eleventh International Conference on Learning Representations*, 2023. Kate Lorenzen. Cospectral constructions for several graph matrices using cousin vertices. *Special Matrices*, 10(1):9–22, 2022. Laszl ´ o Lov ´ asz. ´ *Large networks and graph limits*, volume 60. American Mathematical Soc., 2012. Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. In *Advances in neural information processing systems*, volume 32, pp. 2156–2167, 2019. Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In *Proceedings of the AAAI conference on artificial intelligence*, volume 33, pp. 4602–4609, 2019. Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: towards scalable higher-order graph embeddings. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, pp. 21824–21840, 2020. Daniel Neuen. Homomorphism-distinguishing closedness for graphs of bounded tree-width. *arXiv preprint arXiv:2304.07011*, 2023. Chendi Qian, Gaurav Rattan, Floris Geerts, Mathias Niepert, and Christopher Morris. Ordered subgraph aggregation networks. In *Advances in Neural Information Processing Systems*, 2022. Ladislav Rampa´sek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Do- ˇ minique Beaini. Recipe for a general, powerful, scalable graph transformer. *Advances in Neural Information Processing Systems*, 35:14501–14515, 2022. Gaurav Rattan and Tim Seppelt. Weisfeiler-leman and graph spectra. In *Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pp. 2268–2285. SIAM, 2023. David E Roberson. Oddomorphisms and homomorphism indistinguishability over graphs of bounded degree. *arXiv preprint arXiv:2206.10321*, 2022. Tim Seppelt. Logical equivalences, homomorphism indistinguishability, and forbidden minors. *arXiv preprint arXiv:2302.11290*, 2023. Tim Seppelt. Logical equivalences, homomorphism indistinguishability, and forbidden minors. *Information and Computation*, pp. 105224, 2024. Edwin R Van Dam and Willem H Haemers. Which graphs are determined by their spectrum? *Linear Algebra and its applications*, 373:241–272, 2003. Boris Weisfeiler and Andrei Lehman. The reduction of a graph to canonical form and the algebra which appears therein. *NTI, Series*, 2(9):12–16, 1968. Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *International Conference on Learning Representations*, 2019.

[5] Bohang Zhang, Guhao Feng, Yiheng Du, Di He, and Liwei Wang. A complete expressiveness hierarchy for subgraph GNNs via subgraph weisfeiler-lehman tests. In *International Conference on Machine Learning*, volume 202, pp. 41019–41077. PMLR, 2023a. Bohang Zhang, Shengjie Luo, Di He, and Liwei Wang. Rethinking the expressive power of gnns via graph biconnectivity. In *International Conference on Learning Representations*, 2023b. Bohang Zhang, Jingchu Gai, Yiheng Du, Qiwei Ye, Di He, and Liwei Wang. Beyond weisfeilerlehman: A quantitative framework for GNN expressiveness. In *The Twelfth International Conference on Learning Representations*, 2024a. Bohang Zhang, Lingxiao Zhao, and Haggai Maron. On the expressive power of spectral invariant graph neural networks. *arXiv preprint arXiv:2406.04336*, 2024b. Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any gnn with local structure awareness. In *International Conference on Learning Representations*, 2022.
# Appendix

# Table of Contents

| A | Additional   |              | Related        | Work                                   | 16                          |
|---|--------------|--------------|----------------|----------------------------------------|-----------------------------|
| B | Proof        | of           | Theorem        | 3.3                                    | 17                          |
|   | B.1          | Preparation: |                | Parallel Tree and Unfolding Tree       | 17                          |
|   | B.2          | Step         | 1: Equivalence | of Encoding Walk information           | and Spectral Information 18 |
|   | B.3          | Step         | 2: Finding     | the Homomorphic Expressivity           | 20                          |
|   | B.4          | Step         | 3: Finding     | Pebble Game for Spectral Invariant     | GNN 26                      |
|   | B.5          | Step         | 4: Introducing | Furer graphs ¨                         | 28                          |
|   | B.6          | Step         | 5: Proving     | the Maximality of Homomorphism         | Expressivity 31             |
| C | Proof        | of           | Theorem        | 3.10                                   | 34                          |
| D | Experimental |              | Details        |                                        | 35                          |
| E | Higher       | Order        | Spectral       | Invariant GNN                          | 36                          |
|   | E.1          | Update       | Rule           | of Higher-Order Spectral Invariant GNN | 36                          |
|   | E.2          | Homomorphism |                | Expressivity of Higher-Order Spectral  | Invariant GNN 36            |
|   | E.3          | Proof        | of Theorem     | E.6                                    | 37                          |
| F | Proof        | for          | Symmetric      | Power                                  | 39                          |
|   | F.1          | Properties   |                | of Local k − GNN                       | 39                          |
|   | F.2          | Main         | Result         |                                        | 40                          |
|   | F.3          | Proof        | of Theorem     | F.5                                    | 40                          |

# A ADDITIONAL RELATED WORK

Spectral Based Graph Neural Network. Spectral invariants refer to eigenvalues, projection matrices, and other generalized spectral information. In recent studies, spectral invariants have gained significant attention in the fields of graph learning and graph theory [\(Furer](#page-11-0) ¨ , [2010;](#page-11-0) [Van Dam &](#page-12-0) [Haemers,](#page-12-0) [2003;](#page-12-0) [Haemers & Spence,](#page-11-1) [2004\)](#page-11-1). For instance, a well-known conjecture proposed by [Van Dam & Haemers\(2003\)](#page-12-0); [Haemers & Spence](#page-11-1) [\(2004\)](#page-11-1) posits that almost all graphs can be uniquely determined by their spectra, up to isomorphism. Given the importance and widespread application of graph spectral information [\(Bonchev,](#page-10-4) [2018\)](#page-10-4), the machine learning community has also focused on analyzing the ability of graph neural networks (GNNs) to encode spectral information and on designing GNN models that incorporate more spectral features. As a result, several recent works have concentrated on the spectral-based design of GNNs [\(Bruna et al.,](#page-10-2) [2013;](#page-10-2) [Defferrard et al.,](#page-11-2) [2016;](#page-11-2) [Lim](#page-12-2) [et al.,](#page-12-2) [2023;](#page-12-2) [Huang et al.,](#page-11-3) [2024;](#page-11-3) [Feldman et al.,](#page-11-4) [2023;](#page-11-4) [Zhang et al.,](#page-13-0) [2024b\)](#page-13-0). Specifically, [Dwivedi](#page-11-13) [et al.](#page-11-13) [\(2023;](#page-11-13) [2021\)](#page-11-14); [Kreuzer et al.](#page-11-15) [\(2021\)](#page-11-15); [Rampa´sek et al.](#page-12-16) ˇ [\(2022\)](#page-12-16) have designed spectral GNNs by encoding Laplacian eigenvectors as absolute positional encodings. A key drawback of using Laplacian eigenvectors is the ambiguity in choosing eigenvectors; thus, follow-up works have sought to design GNNs that are invariant to the choice of eigenvectors. [Lim et al.](#page-12-2) [\(2023\)](#page-12-2) introduced BasisNet, which achieves spectral invariance for the first time using projection matrices. [Huang et al.](#page-11-3) [\(2024\)](#page-11-3) further generalized BasisNet by proposing the Spectral Projection Encoding (SPE), which performs soft aggregation across different eigenspaces, as opposed to the hard separation implemented in BasisNet.

In addition to the design of spectral-based GNNs, several recent works have also focused on analyzing the expressive power of spectral GNNs and comparing them with other mainstream GNN models. [Balcilar et al.](#page-10-15) [\(2021\)](#page-10-15) investigate the relationship between ChebNet [\(Defferrard et al.,](#page-11-2) [2016\)](#page-11-2) and the 1-WL test, demonstrating that for graphs with similar maximum eigenvalues, ChebNet is as expressive as 1-WL. [Geerts & Reutter](#page-11-16) [\(2022\)](#page-11-16) revisit this analysis and prove that CaleyNet [\(Levie](#page-12-17) [et al.,](#page-12-17) [2018\)](#page-12-17) is bounded by the 2-WL test.

[Black et al.](#page-10-3) [\(2024\)](#page-10-3) introduced several new WL algorithms based on absolute and relative positional encodings (PE). The authors further established a bunch of equivalence relationships among these algorithms. Notably, there exists a strong connection between the proposed "stack of power of matrices" PE and Spectral Invariant GNNs. We can prove that the proposed (I, L, · · · , L<sup>2</sup>n−<sup>1</sup> )-WL (see Theorem 4.6 in [Black et al.](#page-10-3) [\(2024\)](#page-10-3)) is as expressive as spectral invariant GNNs with matrix L, and similarly, (I, A, · · · , A<sup>2</sup>n−<sup>1</sup> )-WL is as expressive as spectral invariant GNNs with the ordinary adjacency matrix. Therefore, all results in our paper can be used to understand the power of these WL variants. Since [Zhang et al.](#page-13-0) [\(2024b\)](#page-13-0) has shown that the expressive power of RD-WL is bounded by Spectral Invariant GNNs, it follows that the proposed L † -WL (see Theorem 4.6 in [Black et al.](#page-10-3) [\(2024\)](#page-10-3)) is also bounded in expressive power by Spectral Invariant GNNs. This conclusion reproduces their key result (Theorem 4.4 in [Black et al.](#page-10-3) [\(2024\)](#page-10-3)).

Homomorphism Count and Subgraph Count. Subgraph counting is a fundamental problem in chemical and biological tasks, as the ability to count subgraphs is strongly correlated with the performance of GNN in molecular prediction tasks. [Kanatsoulis & Ribeiro](#page-11-17) studies subgraph counting power for a novel GNN framework, where classic message-passing GNNs are enhanced with random node features, and the GNN output is computed by taking the expectation over the introduced randomness. The paper demonstrates that such GNNs can learn to count various substructures, including cycles and cliques. These findings share similarities with our work, as both studies characterize the cycle-counting power of certain GNN models. Notably, the GNN framework proposed in [Kanatsoulis & Ribeiro](#page-11-17) can count more complex substructures, such as 4-cliques and 8-cycles, which exceed the expressive power of 2-FWL.

Moreover, based on the foundational theory of [Lovasz](#page-12-18) ´ [\(2012\)](#page-12-18); [Curticapean et al.](#page-10-16) [\(2017\)](#page-10-16), it follows that the subgraph counting power of a GNN can be inferred from its ability to count homomorphisms [\(Seppelt,](#page-12-15) [2023;](#page-12-15) [Zhang et al.,](#page-13-2) [2024a\)](#page-13-2). Consequently, recent research has also focused on the homomorphism counting power of GNNs. [Dell et al.](#page-11-8) [\(2018\)](#page-11-8) demonstrates that two graphs have the same representation under the k-WL algorithm if and only if the number of homomorphisms to the two graphs from any substructure with bounded tree width k is equal. Additionally, [Zhang](#page-13-2) [et al.](#page-13-2) [\(2024a\)](#page-13-2) introduce the concept of homomorphism expressivity as a quantitative framework for assessing the expressive power of GNNs. This paper specifically focuses on the subgraph counting power of spectral invariant GNNs. Related works in this area include [Cvetkovic et al.](#page-10-9) [\(1997\)](#page-10-9), which shows that the graph angles (which can be determined through projection) are capable of counting all cycles of length up to 5, and [Lim et al.](#page-12-2) [\(2023\)](#page-12-2), which demonstrates that GNNs can count cycles with up to 5 vertices. A detailed comparison of our results with these previous studies is provided in the main text.

## B PROOF OF THEOREM [3.3](#page-3-0)

#### B.1 PREPARATION: PARALLEL TREE AND UNFOLDING TREE

#### B.1.1 ADDITIONAL EXPLANATION FOR PARALLEL TREE

For the reader's convenience, we begin by restating the definition of the parallel tree, as introduced in the main paper.

Definition B.1 (Parallel Edge:). We denote a graph G as a *parallel edge* if there exist vertices u, v ∈ V<sup>G</sup> such that the edge set E<sup>G</sup> can be partitioned into a sequence of simple paths P1, . . . , Pm, where each path has endpoints (u, v). We refer to (u, v) as the endpoints of the parallel edge G.

Definition B.2 (Parallel Tree:). We define a graph F as a *parallel tree* if there exists a tree T such that we can obtain a graph isomorphic to F by replacing each edge (u, v) ∈ E<sup>T</sup> with a parallel edge having endpoints (u, v). We refer to T as the *parallel tree skeleton* of the graph F. Additionally, we denote the minimum depth of any parallel tree skeleton of F as the *parallel tree depth* of F.

We further define parallel tree decomposition for any parallel tree as follows:

Definition B.3 (Parallel tree decomposition). For a parallel tree F = (V<sup>F</sup> , E<sup>F</sup> ), its parallel tree decomposition involves constructing a rooted tree T <sup>r</sup> = (VT<sup>r</sup> , ET<sup>r</sup> ) along with mapping functions βT<sup>r</sup> and γT<sup>r</sup> that satisfy the following conditions:

- 1. The label function for nodes, βT<sup>r</sup> : VT<sup>r</sup> → V<sup>F</sup> , maps each node in T r to a unique vertex in F.
- 2. Let E<sup>F</sup> denote the union of all paths in the graph F. The edge label function, γT<sup>r</sup> : ET<sup>r</sup> → 2 <sup>E</sup><sup>F</sup> , satisfies the condition that for all (t1, t2) ∈ ET<sup>r</sup> , each P ∈ γT<sup>r</sup> (t1, t2) is a path connecting βT<sup>r</sup> (t1) and βT<sup>r</sup> (t2). Moreover, for each edge e ∈ E<sup>F</sup> , there exists a unique tuple (t1, t2, P), where (t1, t2) ∈ V<sup>T</sup> × V<sup>T</sup> and P ∈ γ<sup>T</sup> (t1, t2), such that e lies on the path P.

We denote T <sup>r</sup> = (VT<sup>r</sup> , ET<sup>r</sup> , βT<sup>r</sup> , γT<sup>r</sup> ) as the decomposition skeleton of graph F, and the ordered pair (F, T<sup>r</sup> ) as a parallel-tree decomposed graph.

Let S pt denote the set of all parallel trees, and we use S pt d to denote the set of all parallel trees whose parallel tree skeleton has depth at most d.

#### B.1.2 UNFOLDING TREE OF SPECTRAL INVARIANT GNN

We now introduce a process of constructing a parallel tree from any vertex of a given graph.

Definition B.4 (Constructing an unfolding tree of spectral invariant GNN). Given a graph G, vertex u ∈ V (G) and a non-negative integer d, the depth-d spectral GNN unfolding tree of graph G at vertex u, denoted as (F (d) <sup>G</sup> (u), T(d) <sup>G</sup> (u)), is a parallel-tree decomposed graph constructed as follows: At the beginning, F = {u}, and T only has a root node r with βT<sup>r</sup> (r) = {u}. We can define a mapping π : V<sup>F</sup> → V<sup>G</sup> as π(u) = u.

For each leaf node t in T r , do the following procedure: Let βT<sup>r</sup> (t) = x. For each w ∈ VG, add a fresh node t<sup>w</sup> to T r and designate t as its parent. Then, consider the following case:

- 1. If w ̸= π(x), add x<sup>w</sup> to F and extend π with π(xw) = w. We define βT<sup>r</sup> (tw) = xw. For every walk w = v1, v2, . . . , v<sup>n</sup> = π(x) with n ≤ |VG|, where v<sup>1</sup> = π(x), v<sup>n</sup> = w, we introduce a path x<sup>v</sup><sup>1</sup> , x<sup>v</sup><sup>2</sup> , . . . , x<sup>v</sup><sup>n</sup> linking x<sup>w</sup> and x to graph F, where x<sup>v</sup><sup>1</sup> = x, x<sup>v</sup><sup>n</sup> = xw. We can also extend mapping π with π(x<sup>v</sup><sup>1</sup> ) = v1, π(x<sup>v</sup><sup>2</sup> ) = v2, . . . , π(x<sup>v</sup><sup>n</sup> ) = vn. We define γT<sup>r</sup> (t, tw) to be the set of all path x<sup>v</sup><sup>1</sup> , x<sup>v</sup><sup>2</sup> , . . . , x<sup>v</sup><sup>n</sup> connecting x and x<sup>w</sup> introduced in this step.
- 2. If w = π(x), we define βT<sup>r</sup> (tw) = x. Similarly, for every walk w = v1, v2, . . . , v<sup>n</sup> = π(x) with n ≤ |VG|, we introduce a loop x<sup>v</sup><sup>1</sup> , x<sup>v</sup><sup>2</sup> , . . . , x<sup>v</sup><sup>n</sup> to graph F, where x<sup>v</sup><sup>1</sup> = x = x<sup>v</sup><sup>n</sup> . We can also extend mapping π with π(x<sup>v</sup><sup>1</sup> ) = v1, π(x<sup>v</sup><sup>2</sup> ) = v2, . . . , π(x<sup>v</sup><sup>n</sup> ) = vn. We define

γT<sup>r</sup> (t, tw) to be the set of all path x<sup>v</sup><sup>1</sup> , x<sup>v</sup><sup>2</sup> , . . . , x<sup>v</sup><sup>n</sup> connecting x and x<sup>w</sup> introduced in this step.

We terminate the process once T <sup>r</sup> becomes a complete tree of depth d.

The following fact is straightforward from the construction of the unfolding tree:

Fact B.5. *For any graph* G*, any vertex* u ∈ VG*, and any non-negative integer* D*, there is a homomorphism from* F (D) <sup>G</sup> (u) *to* G*.*

With additional Explanation for parallel tree and construction of unfolding tree, we are now ready to prove Theorem [3.3](#page-3-0) step by step.

#### B.2 STEP 1: EQUIVALENCE OF ENCODING WALK INFORMATION AND SPECTRAL INFORMATION

In this section, we aim to prove Lemma [3.17.](#page-6-4) The key idea is to use the Cayley-Hamilton theorem to demonstrate that the walk-encoding GNN, as defined in Lemma [3.17,](#page-6-4) is equivalent to the spectral invariant GNN.

#### B.2.1 PROOF OF LEMMA [3.17](#page-6-4)

Lemma B.6. *Let* G = (VG, EG) *be a graph, with its adjacency matrix denoted by* A*. For vertices* x, y ∈ VG*, define* ω k <sup>G</sup>(x, y) = A<sup>k</sup> x,y *for all* k ∈ {0, 1, 2, . . . , |VG|}*, which represents the number of* k*-walks from vertex* x *to vertex* y*. Define the tuple* ω ∗ <sup>G</sup>(x, y) = (ω 0 G(x, y), ω<sup>1</sup> G(x, y), . . . , ω<sup>n</sup>−<sup>1</sup> <sup>G</sup> (x, y))*, where* n = |VG|*. Define the walk-encoding GNN with the following update rule:*

$$\chi_G^{\text{Walk},(d+1)}(x) = \text{hash}(\chi_G^{\text{Walk},(d)}(x), \{\{(\omega_G^*(x, y), \chi_G^{\text{Walk},(d)}(y)) \mid y \in V_G\}\}).$$

*The walk-encoding GNN outputs a graph invariant* χ Walk,(d) <sup>G</sup> (G) = {{χ Walk,(d) <sup>G</sup> (u)|u ∈ VG}}*. For any graphs* G *and* H*, we have* χ Walk,(d) <sup>G</sup> (G) = χ Walk,(d) <sup>H</sup> (H) *if and only if* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H)*.*

*Proof.* We begin by proving the following statement: If the spectra of graph G and graph H are identical (denoted as (λ1, λ2, . . . , λm)), then for x, u ∈ V<sup>G</sup> and y, v ∈ VH, P(x, u) = P(y, v) if and only if ω ⋆ <sup>G</sup>(x, u) = ω ⋆ <sup>H</sup>(y, v).

- 1. First, we prove that if P(x, u) = P(y, v), then ω ⋆ <sup>G</sup>(x, u) = ω ⋆ <sup>H</sup>(y, v).

By the properties of diagonalizable matrices, for any k ∈ {1, 2, . . . , |VG|}, we have:

$$\omega_G^k(x, u) = \lambda_1^k \mathbf{P}_{\lambda_1}(x, u) + \lambda_{\lambda_2}^k \mathbf{P}_2(x, u) + \cdots + \lambda_m^k \mathbf{P}_{\lambda_m}(x, u).$$

Therefore, if

$$P_{\lambda_r}(x, u) = P_{\lambda_r}(y, v), \quad \forall r \in [m],$$

it follows that:

$$\omega_G^k(x, u) = \sum_{r=1}^m \lambda_r^k \mathbf{P}_{\lambda_r}(x, u) = \sum_{r=1}^m \lambda_r^k \mathbf{P}_{\lambda_r}(y, v) = \omega_H^k(y, v).$$

Thus, we have proven the first direction of the statement.

- 2. Now, we prove that if ω ⋆ <sup>G</sup>(x, u) = ω ⋆ <sup>H</sup>(y, v), then P(x, u) = P(y, v).

Let A<sup>G</sup> and A<sup>H</sup> denote the adjacency matrices of graphs G and H, respectively. By the Cayley-Hamilton theorem, the minimal annihilating polynomial of matrix A<sup>G</sup> is given by:

$$f(\lambda) = (\lambda - \lambda_1)(\lambda - \lambda_2) \cdots (\lambda - \lambda_m).$$

For each r ∈ {1, 2, . . . , m}, the eigenspace corresponding to eigenvalue λ<sup>r</sup> is Ker(λrI −AG). Since:

$$\mathbb{R}^n = \mathbf{Ker}(\lambda_1 I - \mathbf{A}_G) \oplus \mathbf{Ker}(\lambda_2 I - \mathbf{A}_G) \oplus \cdots \oplus \mathbf{Ker}(\lambda_m I - \mathbf{A}_G),$$

for each r ∈ {1, 2, . . . , m}, the projection matrix onto the kernel space Ker(λrI − AG) is:

$$f_r(\mathbf{A}_G) = \prod_{j \neq r} (\lambda_j I - \mathbf{A}_G) = \mathbf{P}_{\lambda_r}.$$

Therefore, there exist coefficients c r 0 , . . . , c<sup>r</sup> m−1 such that:

$$\begin{aligned} \mathbf{P}_{\lambda,r}(x, u) &= c_0^r \cdot \omega_G^0(x, u) + c_1^r \cdot \omega_G^1(x, u) + \cdots + c_{m-1}^r \cdot \omega_G^{m-1}(x, u), \\ \mathbf{P}_{\lambda,r}(y, v) &= c_0^r \cdot \omega_H^0(y, v) + c_1^r \cdot \omega_H^1(y, v) + \cdots + c_{m-1}^r \cdot \omega_H^{m-1}(y, v). \end{aligned}$$

Finally, we conclude that if ω ⋆ <sup>G</sup>(x, u) = ω ⋆ <sup>H</sup>(y, v), then P(x, u) = P(y, v) for all x, u ∈ V<sup>G</sup> and y, v ∈ VH.

Armed with the statement proven above, we are now prepared to prove Lemma [3.17.](#page-6-4) We will prove the two directions of the lemma separately as follows:

- 1. First, we prove that if χ Spec <sup>G</sup> (G) = χ Spec <sup>H</sup> (H), then χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H). To do so, it suffices to show that for all t ∈ N, if χ Spec,(t) <sup>G</sup> (u) = χ Spec,(t) <sup>H</sup> (v) for all (u, v) ∈ V<sup>G</sup> × VH, then χ Walk,(t) <sup>G</sup> (u) = χ Walk,(t) <sup>H</sup> (v).

We prove this by induction. Initially, the statement holds trivially for t = 0. We then assume the statement holds for t = d and aim to prove it for t = d + 1. If χ Spec,(d+1) <sup>G</sup> (u) = χ Spec,(d+1) <sup>H</sup> (v), then the following conditions are satisfied:

$$\begin{aligned} \chi_G^{\text{Spec},(d)}(u) &= \chi_H^{\text{Spec},(d)}(v), \\ \{\{\mathcal{P}(u, x), \chi_G^{\text{Spec},(d)}(x)) \mid x \in V_G\}\} &= \{\{\mathcal{P}(v, y), \chi_H^{\text{Spec},(d)}(y)) \mid y \in V_H\}\}. \end{aligned} \quad (1)$$

For any x ∈ V<sup>G</sup> and y ∈ VH, if (P(u, x), χ Spec,(d) <sup>G</sup> (x)) = (P(v, y), χ Spec,(d) <sup>H</sup> (y)), then by our previous result and the induction hypothesis, we have:

$$(\omega_G^*(u, x), \chi_G^{\text{Walk}, (d)}(x)) = (\omega_H^*(v, y), \chi_H^{\text{Walk}, (d)}(y)). \quad (2)$$

By combining equation [1](#page-18-0) and equation [2,](#page-18-1) we conclude:

$$\begin{aligned} \chi_G^{\text{Walk},(d)}(u) &= \chi_H^{\text{Walk},(d)}(v), \\ \{\{(\omega_G^*(u, x), \chi_G^{\text{Walk},(d)}(x)) \mid x \in V_G\}\} &= \{\{(\omega_H^*(v, y), \chi_H^{\text{Walk},(d)}(y)) \mid y \in V_H\}\}. \end{aligned}$$

Thus, we conclude that χ Walk,(d+1) <sup>G</sup> (u) = χ Walk,(d+1) <sup>H</sup> (v). Therefore, we have proven that χ Spec <sup>G</sup> (G) = χ Spec <sup>H</sup> (H) implies χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H).

- 2. Now, we prove the converse: if χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H), then χ Spec <sup>G</sup> (G) = χ Spec <sup>H</sup> (H). Initially, χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H) implies {{χ Walk,(1) <sup>G</sup> (u) | u ∈ VG}} = {{χ Walk,(1) <sup>H</sup> (v) | v ∈ VH}}. If χ Walk,(1) <sup>G</sup> (u) = χ Walk,(1) <sup>H</sup> (v), then ω ⋆ <sup>G</sup>(u, u) = ω ⋆ <sup>H</sup>(v, v). This leads to: {{ω ⋆ <sup>G</sup>(u, u) | u ∈ VG}} = {{ω ⋆ <sup>H</sup>(v, v) | v ∈ VH}}.

Hence, we derive that for all k ∈ [n]:

$$\text{tr}(\mathbf{A}_G^k) = \sum_{u \in V_G} \mathbf{A}_G^k(u, u) = \sum_{u \in V_G} \omega_G^k(u, u) = \sum_{v \in V_H} \omega_H^k(v, v) = \sum_{v \in V_H} \mathbf{A}_H^k(v, v) = \text{tr}(\mathbf{A}_H^k).$$

By standard results from linear algebra, the spectra of graphs G and H must be identical.

Similar to the first direction, we now prove that for all t ∈ N, if χ Walk,(t) <sup>G</sup> (u) = χ Walk,(t) <sup>H</sup> (v) for all (u, v) ∈ V<sup>G</sup> × VH, then χ Spec,(t) <sup>G</sup> (u) = χ Spec,(t) <sup>H</sup> (v).

We again proceed by induction. Initially, the statement holds trivially for t = 0. Assuming the statement holds for t = d, we aim to prove it for t = d + 1. If χ Walk,(d+1) <sup>G</sup> (u) = χ Walk,(d+1) <sup>H</sup> (v), we have:

$$\begin{aligned} \chi_G^{\text{Walk},(d)}(u) &= \chi_H^{\text{Walk},(d)}(v), \\ \{\{(\omega_G^*(u,x), \chi_G^{\text{Walk},(d)}(x)) \mid x \in V_G\}\} &= \{\{(\omega_H^*(v,y), \chi_H^{\text{Walk},(d)}(y)) \mid y \in V_H\}\}. \end{aligned}$$

According to the statement proven earlier, for any x ∈ V<sup>G</sup> and y ∈ VH, ω ⋆ <sup>G</sup>(u, x) = ω ⋆ <sup>H</sup>(v, y) implies that P(u, x) = P(v, y). Thus, we obtain:

$$\begin{aligned} \chi_G^{\text{Spec},(d)}(u) &= \chi_H^{\text{Spec},(d)}(v), \\ \{\{(\mathcal{P}(u, x), \chi_G^{\text{Spec},(d)}(x)) \mid x \in V_G\}\} &= \{\{(\mathcal{P}(v, y), \chi_H^{\text{Spec},(d)}(y)) \mid y \in V_H\}\}. \end{aligned}$$

Therefore, we conclude that χ Spec,(d+1) <sup>G</sup> (u) = χ Spec,(d+1) <sup>H</sup> (v). Finally, we have proven that χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H) implies χ Spec <sup>G</sup> (G) = χ Spec <sup>H</sup> (H).

By combining both directions, we conclude that for any two graphs G and H, χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H) if and only if χ Spec <sup>G</sup> (G) = χ Spec <sup>H</sup> (H). Hence, the walk-encoding GNN is as expressive as the spectralinvariant GNN.

#### B.3 STEP 2: FINDING THE HOMOMORPHIC EXPRESSIVITY

We first define the isomorphism between parallel-tree decomposed graphs.

Definition B.7. Given two parallel-tree decomposed graphs (F, T<sup>r</sup> ) and (F , ˜ T˜<sup>r</sup> ), a pair of mappings (ρ, τ ) is called an isomorphism from (F, T<sup>r</sup> ) to (F , ˜ T˜<sup>r</sup> ), denoted by (F, T<sup>r</sup> ) ∼= (F , ˜ T˜<sup>r</sup> ), if the following hold:

- 1. ρ is an isomorphism from F to F˜, while τ is an isomorphism from T r to T˜<sup>r</sup> (ignoring labels β and γ).
- 2. For any t ∈ VT<sup>r</sup> , ρ(βT<sup>r</sup> (t)) = βT˜<sup>r</sup> (τ (t)). Moreover, for any (t1, t2) ∈ ET<sup>r</sup> , ρ(γT<sup>r</sup> (t1, t2)) = γT<sup>r</sup> (τ (t1, t2))

Theorem B.8. *For any two graphs* G, H*, any vertices* u ∈ VG*,* x ∈ VH*,and any non-negative integer* D*,* χ Walk,(D) <sup>G</sup> (u) = χ Walk,(D) <sup>H</sup> (x) *iff there exists an isomorphism* (ρ, τ ) *from* (F (D) <sup>G</sup> (u), T(D) <sup>G</sup> (u)) *to* (F (D) <sup>H</sup> (x), T(D) <sup>H</sup> (x)) *such that* ρ(u) = x*.*

*Proof.* The proof proceeds by induction on D. The base case is straightforward: for D = 0, the theorem holds trivially. Now assume the theorem holds for all D ≤ d, and we will prove it for D = d + 1.

We first prove that χ Walk,(d+1) <sup>G</sup> (u) = χ Walk,(d+1) <sup>H</sup> (x) implies the existence of an isomorphism (ρ, τ ) from (F (d+1) <sup>G</sup> (u), T(d+1) <sup>G</sup> (u)) to (F (d+1) <sup>H</sup> (x), T(d+1) <sup>H</sup> (x)) such that ρ(u) = x. Given that χ (d+1) <sup>G</sup> (u) = χ (d+1) <sup>H</sup> (x), it follows that:

$$\{\{\omega_G^*(u, v), \chi_G^{\text{Walk}, (d)}(v)\}\}_{v \in V_G} = \{\{\omega_H^*(x, y), \chi_H^{\text{Walk}, (d)}(y)\}\}_{y \in V_H}.$$

Let n = |VG| = |VH|, and denote V<sup>G</sup> = {v1, v2, . . . , vn}, V<sup>H</sup> = {y1, y2, . . . , yn} such that:

$$\omega_G^*(u, v_i) = \omega_H^*(x, y_i), \quad \chi_G^{\text{Walk},(d)}(v_i) = \chi_H^{\text{Walk},(d)}(y_i) \quad \text{for all } i \in [n].$$

By the definition of tree unfolding, we have:

$$F_G^{(d+1)}(u) = \left( \bigcup_{v_i} F_G^{(d)}(v_i) \right) \cup F_G^{(1)}(u), \quad F_H^{(d+1)}(x) = \left( \bigcup_{y_i} F_H^{(d)}(y_i) \right) \cup F_H^{(1)}(x),$$

where we use ∪ to represent graph union. By the inductive hypothesis, there exists an isomorphism (ρ<sup>i</sup> , τi) from (F (d) <sup>G</sup> (vi), T(d) <sup>G</sup> (vi)) to (F (d) <sup>H</sup> (yi), T(d) <sup>H</sup> (yi)) such that ρi(vi) = y<sup>i</sup> . Additionally, since ω ∗ <sup>G</sup>(u, vi) = ω ∗ <sup>H</sup>(x, yi), F (1) <sup>G</sup> (u) is isomorphic to F (1) <sup>H</sup> (x). Therefore, by merging all ρ<sup>i</sup> and τ<sup>i</sup> into ρ˜ and τ˜, and constructing an approximate mapping between tree nodes at depth no more than 1 in T (d+1) <sup>G</sup> (u) and T (d+1) <sup>H</sup> (x), it follows that (˜ρ, τ˜) is a well-defined isomorphism from (F (d+1) <sup>G</sup> (u), T(d+1) <sup>G</sup> (u)) to (F (d+1) <sup>H</sup> (x), T(d+1) <sup>H</sup> (x)), satisfying ρ˜(u) = x.

Next, we prove that if there exists an isomorphism (ρ, τ ) between the parallel-tree decomposed graphs (F (d+1) <sup>G</sup> (u), T(d+1) <sup>G</sup> (u)) and (F (d+1) <sup>H</sup> (x), T(d+1) <sup>H</sup> (x)) such that ρ(u) = x, then χ Walk,(d+1) <sup>G</sup> (u) = χ Walk,(d+1) <sup>H</sup> (x). Since τ is an isomorphism from T (d+1) <sup>G</sup> (u) to T (d+1) <sup>H</sup> (x), it maps all depth-1 nodes in T (d+1) <sup>G</sup> (u) to depth-1 nodes in T (d+1) <sup>H</sup> (x). Let s1, s2, . . . , s<sup>n</sup> be the depth-1 nodes in T (d+1) <sup>G</sup> (u), and t1, t2, . . . , t<sup>n</sup> be the corresponding nodes in T (d+1) <sup>H</sup> (x). For i ∈ [n], we denote the subtree induced by s<sup>i</sup> and its descendants as T (d+1) G,s<sup>i</sup> (u), and similarly, the subtree induced by t<sup>i</sup> and its descendants as T (d+1) G,t<sup>i</sup> (x). Additionally, we define the subgraph of F (d+1) <sup>G</sup> (u) induced by T (d+1) G,s<sup>i</sup> (u) as F (d+1) G,s<sup>i</sup> (u). Likewise, we define the subgraph of F (d+1) <sup>H</sup> (u) induced by T (d+1) H,t<sup>i</sup> (u) as F (d+1) H,t<sup>i</sup> (u). Without loss of generality, we assume the following:

- τ is an isomorphism from the subtree T (d+1) G,s<sup>i</sup>
- (u) to T (d+1) H,t<sup>i</sup> (x).
- For all <sup>s</sup> ∈ <sup>V</sup><sup>T</sup> (d+1) G,si
- (u) , ρ(β<sup>T</sup> (d+1) <sup>G</sup> (u) (s)) = β<sup>T</sup> (d+1) <sup>H</sup> (x) (τ (s)).
- For all <sup>e</sup> ∈ <sup>E</sup><sup>T</sup> (d+1) G,si
- (u) , ρ(γ T (d+1) <sup>G</sup> (u) (e)) = γ T (d+1) <sup>H</sup> (x) (τ (e)).
- ρ is an isomorphism between the subgraphs F (d+1) G,s<sup>i</sup>
- (u) and F (d+1) H,t<sup>i</sup> (x).

According to our assumption, (F (d+1) G,s<sup>i</sup> (u), T(d+1) G,s<sup>i</sup> (u)) is isomorphic to (F (d+1) H,t<sup>i</sup> (x), T(d+1) H,t<sup>i</sup> (x)). Additionally, by the definition of the unfolding tree, (F (d+1) G,s<sup>i</sup> (u), T(d+1) G,s<sup>i</sup> (u)) is isomorphic to the depth-d unfolding tree (F (d) <sup>G</sup> (vi), T(d) <sup>G</sup> (vi)) for some v<sup>i</sup> ∈ VG. Similarly, (F (d+1) H,t<sup>i</sup> (x), T(d+1) H,t<sup>i</sup> (x)) is isomorphic to (F (d) <sup>H</sup> (yi), T(d) <sup>H</sup> (yi)) for some y<sup>i</sup> ∈ VH. By induction, we know that χ Walk,(d) <sup>G</sup> (vi) = χ Walk,(d) <sup>H</sup> (yi) and ω ∗ <sup>G</sup>(u, vi) = ω ∗ <sup>H</sup>(x, yi). Therefore, we conclude:

$$\left( \omega_G^*(u, v_i), \chi_G^{\text{Walk}, (d)}(v_i) \right) = \left( \omega_H^*(x, y_i), \chi_H^{\text{Walk}, (d)}(y_i) \right)$$

for all i ∈ [n], implying that:

$$\{ \left( \omega_G^*(u, v_i), \chi_G^{\text{Walk}, (d)}(v_i) \right) \}_{v_i \in V_G} = \{ \left( \omega_H^*(x, y_i), \chi_H^{(d)}(y_i) \right) \}_{y_i \in V_H}. \quad (3)$$

It remains to prove that χ Walk,(d) <sup>G</sup> (u) = χ Walk,(d) <sup>H</sup> (x). To prove this, note that equation [3](#page-20-0) implies that

$$\left\{ \left( \omega_G^*(u, v_i), \chi_G^{\text{Walk}, (d')}(v_i) \right) \right\}_{y_i \in V_G} = \left\{ \left( \omega_H^*(x, y_i), \chi_H^{\text{Walk}, (d')}(y_i) \right) \right\}_{y_i \in V_H}.$$

holds for all 0 ≤ d ′ ≤ d. Combined this with the fact that χ Walk,(0) <sup>G</sup> (u) = χ Walk,(0) <sup>H</sup> (x), we can incrementally prove that χ Walk,(d ′ ) <sup>G</sup> (u) = χ Walk,(d ′ ) <sup>H</sup> (x) for all d ′ ≤ d + 1. We have thus concluded the proof. Thus, the proof is complete.

Definition B.9. Given a graph G and a parallel-tree decomposed graph (F, T<sup>r</sup> ), we define the function treeCount((F, T<sup>r</sup> ), G) as the number of ordered pairs (u, d) ∈ V<sup>G</sup> × <sup>N</sup> such that the depth-d unfolding tree (F (d) <sup>G</sup> (u), T(d) <sup>G</sup> (u)) at vertex <sup>u</sup> is isomorphic to (F, T<sup>r</sup> ).

Corollary B.10. *For any graph* G, H*,* χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H) *iff* treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) *holds for all parallel-tree decomposed graph* (F, T<sup>r</sup> )*.*

*Proof.* We first prove one direction of the corollary. We aim to prove that if χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H), then treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H). If χ Walk <sup>G</sup> (G) = χ Walk <sup>H</sup> (H), then {{χ Walk <sup>G</sup> (u) : u ∈ VG}} = {{χ Walk <sup>H</sup> (x) : x ∈ VH}}. For each color c in the above multiset, pick u ∈ V<sup>G</sup> with χ Walk <sup>G</sup> (u) = c. It follows that if (F, T<sup>r</sup> ) ∼= (F (D) <sup>G</sup> (u), T(D) <sup>G</sup> (u)) for some D, then treeCount((F, T<sup>r</sup> ), G) = |{{u ∈ V<sup>G</sup> : χ Walk <sup>G</sup> (u) = c}}| = |{{x ∈ V<sup>H</sup> : χH(x) = c}}| = treeCount((F, T<sup>r</sup> ), H) by Theorem [B.8.](#page-19-1) On the other hand, if (F, T<sup>r</sup> ) ̸∼= (F (D) <sup>G</sup> (u), T(D) <sup>G</sup> (u)) for all u ∈ V<sup>G</sup> and all D, then clearly treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) = 0.

We then aim to prove the second direction of the corollary. If treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) holds for all parallel-tree decomposed graph (F, T<sup>r</sup> ), it clearly holds for all (F (D) <sup>G</sup> (u), T(D) <sup>G</sup> (u)) with u ∈ V<sup>G</sup> and a sufficiently large D. This guarantees that for all color c, |{{u ∈ V<sup>G</sup> : χ Walk <sup>G</sup> (u) = c}}| = |{{x ∈ V<sup>H</sup> : χ Walk <sup>H</sup> (x) = c}}| by Theorem [B.8.](#page-19-1) Therefore, {{χ Walk <sup>G</sup> (u) : u ∈ VG}} = {{χ Walk <sup>H</sup> (x) : x ∈ VH}}, concluding the proof.

Definition B.11. For parallel-tree decomposed graph (F, T<sup>r</sup> ), we use Dep(T r ) to denote the depth of tree T. For any tree note t ∈ V<sup>T</sup> , we use dep<sup>T</sup> (t) to denote the depth of node t in T r .

Using techniques similar to those in Corollary [B.10,](#page-20-1) we can derive a finite-iteration version of Corollary [B.10](#page-20-1) as follows:

Corollary B.12. *For any graphs* G *and* H*,* χ Walk,(d) <sup>G</sup> (G) = χ Walk,(d) <sup>H</sup> (H) *if and only if* treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) *holds for all parallel-tree decomposed graphs* (F, T<sup>r</sup> ) *with* Dep(T r ) ≤ d*.*

In the following theorem, we will bridge homomorphic count with unfolding tree count. Before presenting the result, we first introduce some notations used to present the theorem.

Definition B.13. Given two parallel-tree decomposed graphs (F, T<sup>r</sup> ) and (F , ˜ T˜<sup>r</sup> ), a pair of mappings (ρ, τ ) is called a *strong homomorphism* from (F, T<sup>r</sup> ) to (F , ˜ T˜<sup>s</sup> ) if it satisfies the following conditions: First, τ is a homomorphism from T to T˜, ignoring the labels β and γ, and is depthpreserving, i.e., depT<sup>r</sup> (t) = depT˜<sup>s</sup> (τ (t)) for all t ∈ V<sup>T</sup> . Additionally, ρ is a homomorphism from <sup>F</sup>[γ<sup>T</sup> (t1, t2)] to <sup>F</sup>˜[γT˜(<sup>τ</sup> (t1), τ (t2))]. Finally, the depth of <sup>T</sup> r is equal to the depth of T˜<sup>s</sup> .

We use strHom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )) to denote the set of all strong homomorphism from (F, T<sup>r</sup> ) to (F , ˜ T˜<sup>r</sup> ), and let strhom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )) = |strHom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> ))|.

Theorem B.14. *Let* (F, T<sup>r</sup> ) *be parallel-tree decomposed graph and let* G *be a graph. We have*

$$\text{hom}(F, G) = \sum_{(\tilde{F}, \tilde{T}^r) \in \mathcal{S}^{pt}} \text{strhom}\left((F, T^r), (\tilde{F}, \tilde{T}^r)\right) \cdot \text{treeCount}((\tilde{F}, \tilde{T}^r), G)$$

*Proof.* We assume that βT<sup>r</sup> (r) = u for (F, T<sup>r</sup> ), and the depth of (F, T<sup>r</sup> ) is d. Let x ∈ V<sup>G</sup> be any vertex in G, and denote (F (d) <sup>G</sup> (x), T(d) <sup>G</sup> (x)) as the depth-d unfolding tree at x. We define S1(x) as the set of all homomorphisms from F to G that map the vertex u ∈ V<sup>F</sup> to x ∈ VG. Furthermore, we define S2(x) as the set of strong homomorphisms (ρ, τ ) from (F, T<sup>r</sup> ) to (F (d) <sup>G</sup> (x), T(d) <sup>G</sup> (x)), such that ρ(u) = x. Then Theorem [B.14](#page-21-0) is equivalent to the following equation: P x∈V<sup>G</sup> |S1(x)| = P x∈V<sup>G</sup> |S2(x)| . We will prove that |S1(x)| = |S2(x)| for all x ∈ VG. Given x ∈ VG, according to Fact [B.5,](#page-17-1) there exists a homomorphism π from F (d) <sup>G</sup> (x) to graph G. Define a mapping σ such that σ(ρ, τ ) = π ◦ ρ for all (ρ, τ ) ∈ S2(x). It suffices to prove that σ is a bijection from S2(x) to S1(x).

We first prove that σ is a mapping from S2(x) to S1(x). Since ρ is a homomorphism from F to F (d) <sup>G</sup> (x), and π is a homomorphism from F (d) <sup>G</sup> (x) to G. The composition of homomorphism is still a homomorphism. Therefore, π ◦ ρ is a homomorphism from F to graph G.

We then prove that σ is a surjection. For all g ∈ S1(x), we define a mapping (ρ, τ ) from (F, T<sup>r</sup> ) to (F (d) <sup>G</sup> (x), T(d) <sup>G</sup> (x)) as follows. First define ρ(u) = x and set τ (r) to be the root of (F (d) <sup>G</sup> (x), T(d) <sup>G</sup> (x)). Let v1, v2, . . . , v<sup>m</sup> ∈ VT<sup>r</sup> be the tree nodes of depth 1. Similarly, by definition of the unfolding tree, let y1, y2, . . . , y<sup>n</sup> ∈ <sup>V</sup><sup>F</sup> (d) <sup>G</sup> (x) be tree nodes of depth 1. For all i ∈ [m], we denote {Pi1, Pi2, . . . , Pia<sup>i</sup> } = γT<sup>r</sup> (u, vi), to be the paths associated with edge (u, vi) ∈ ET<sup>r</sup> . Similarly, for i ∈ [n] we denote {P˜ <sup>i</sup>1, P˜ <sup>i</sup>2, . . . , P˜ ib<sup>i</sup> } = γT<sup>r</sup> (x, yi) to be the paths associated with edge (x, yi) ∈ <sup>E</sup><sup>T</sup> (d) <sup>G</sup> (x) . Since g and π are both homomorphism, we have:

- For every <sup>v</sup>i(<sup>i</sup> ∈ [m]), there exists y<sup>j</sup> (j ∈ [n]), such that g(βT<sup>r</sup> (vi)) = <sup>π</sup>(β<sup>T</sup>
- (d) <sup>G</sup> (x) (y<sup>j</sup> )) = ˜z<sup>j</sup> for some z˜<sup>j</sup> ∈ VG.
- For every path Pik ∈ γT<sup>r</sup> (u, vi) (k ∈ [a<sup>i</sup> ]) linking u and βT<sup>r</sup> (vi), there exists P˜ jl (l ∈ [b<sup>j</sup> ]) linking x and β<sup>T</sup>
  - (d) <sup>G</sup> (x) (y<sup>j</sup> ) such that g(Pik) = π(P˜ jl).

We then define ρ(βT<sup>r</sup> (vi)) = β<sup>T</sup> (d) <sup>G</sup> (x) (y<sup>j</sup> ) and ρ(Pik) = P˜ jl for each i ∈ [m] and k ∈ [a<sup>i</sup> ]. Based on the above two items, one can easily define τ such that each node s in T <sup>r</sup> of depth 1 is mapped by τ to a node t in T (d) <sup>G</sup> (x) of the same depth, such that ρ(βT<sup>r</sup> (s)) = β<sup>T</sup> (d) <sup>G</sup> (x) (t) and ρ(γT<sup>r</sup> (r, s)) = γ T (d) <sup>G</sup> (x) (x, t). Continuing, we denote the subtree of T r induced by s and all its descendants as T r s , and the subgraph of F induced by T r s as Fs. Similarly, we denote the subtree of T (d) <sup>G</sup> (x) induced by τ (s) and its descendants as T (d) G,τ(s) (x), and the subgraph of F (d) <sup>G</sup> (x) induced by T (d) G,τ(s) (x) as F (d) G,τ(s) (x). We can recursively define the image of ρ on F<sup>s</sup> for each tree node of depth 1, following the same construction described above. This recursive definition holds because g remains a homomorphism from (Fs, T<sup>r</sup> s ) to G, and π remains a homomorphism from (F (d) G,τ(s) (x), T(d) G,τ(s) (x)) to G, with g(βT<sup>r</sup> (s)) = <sup>π</sup>(β<sup>T</sup> (d) <sup>G</sup> (x) (τ (s))). By recursively applying this procedure, we can construct (ρ, τ ) such that it becomes a strong homomorphism (denoted strHom) from (F, T<sup>r</sup> ) to (F (d) <sup>G</sup> (x), T(d) <sup>G</sup> (x)). Therefore, we have shown that for any g ∈ S1(x), there exists a preimage (ρ, τ ) ∈ S2(x) such that σ(ρ, τ ) = g.

#### Finally, we prove that σ is an injection.

Let (ρ1, τ1),(ρ2, τ2) ∈ S2(x) such that π ◦ ρ<sup>1</sup> = π ◦ ρ2. Similar to previous item, we define v1, v2, . . . , v<sup>m</sup> ∈ VT<sup>r</sup> to be the tree nodes of depth 1. Similarly, by definition of the unfolding tree, let y1, y2, . . . , y<sup>n</sup> ∈ <sup>V</sup><sup>T</sup> (d) <sup>G</sup> (x) be tree nodes of depth 1.

- For all i ∈ [m], we denote {Pi1, Pi2, . . . , Pia<sup>i</sup> } = γT<sup>r</sup> (u, vi), to be the paths associated with edge (u, vi) ∈ ET<sup>r</sup> . Similarly, for i ∈ [n] we denote {P˜ <sup>i</sup>1, P˜ <sup>i</sup>2, . . . , P˜ ib<sup>i</sup> } = γT<sup>r</sup> (x, yi) to be the paths associated with edge (x, yi) ∈ <sup>E</sup><sup>T</sup>
  - (d) <sup>G</sup> (x) . For each i ∈ [m], let j1(i) and j2(i) be indices satisfying ρ1(wi) = xj1(i) and ρ2(wi) = xj2(i) . It follows that π(xj1(i)) = π(xj2(i)). By the definition of unfolding tree, we must have xj1(i) = xj2(i) , and thus ρ1(wi) = ρ2(wi).
- For each k ∈ [a<sup>i</sup> ], Pik ∈ γT<sup>r</sup> (u, vi), let l1(k) and l2(k) be indices satisfying ρ1(Pik) = P˜ jl1(j) and ρ2(Pik) = P˜ jl2(j) , where we use j to denote j = j1(i) = j2(i). With similar analysis as the previous item, we have π(P˜ jl1(j)) = <sup>π</sup>(P˜ jl2(j)). By the definition of the unfolding tree, we must have P˜ jl1(j) <sup>=</sup> <sup>P</sup>˜ jl2(j) , and thus ρ1(Pik) = ρ2(Pik).

Next, we recursively apply the previously described procedure to the subtree induced by the tree node s at depth 1 and its descendants, following the same steps outlined earlier. Through this process, we can ultimately demonstrate that ρ<sup>1</sup> = ρ2. Consequently, σ is injective.

Combining the above three parts completes the proof.

Theorem B.15. *Let* (F, T<sup>r</sup> ) *be parallel-tree decomposed graph with* Dep(T r ) ≤ d *and let* G *be a graph. We have*

$$\text{hom}(F, G) = \sum_{(\tilde{F}, \tilde{T}^r) \in \mathcal{S}_d^{pt}} \text{strhom}\left((F, T^r), (\tilde{F}, \tilde{T}^r)\right) \cdot \text{treeCount}\left((\tilde{F}, \tilde{T}^r), G\right)$$

*Proof.* According to the third condition in Definition [B.13,](#page-21-1) for (F, T<sup>r</sup> ) ∈ Spt d and (F , ˜ T˜<sup>r</sup> ) ∈ Spt , if strhom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )) ̸= 0, then Dep(T r ) = Dep(T˜<sup>r</sup> ). Therefore, we have (F , ˜ T˜<sup>r</sup> ) ∈ Spt d . Thus, the conclusion of the lemma follows.

Definition B.16. Given two parallel-tree decomposed graphs (F, T<sup>r</sup> ) and (F , ˜ T˜<sup>r</sup> ), along with a strong homomorphism (ρ, τ ), we define (ρ, τ ) as a *surjective strong homomorphism* if both ρ and τ are surjective mappings, and as an *injective strong homomorphism* if both ρ and τ are injective mappings. We denote the set of all surjective strong homomorphisms from (F, T<sup>r</sup> ) to (F , ˜ T˜<sup>r</sup> ) by strSurj((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )), and further define strsurj((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )) = |strSurj((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> ))|. Similarly, we denote the set of all injective strong homomorphisms from (F, T<sup>r</sup> ) to (F , ˜ T˜<sup>r</sup> ) by strInj((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )), and further define strinj((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> )) = |strInj((F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> ))|.

We now present the following lemma regarding the relationships between strong homomorphisms, surjective strong homomorphisms, and injective strong homomorphisms.

Lemma B.17. *For any parallel-tree decomposed graph* (F, T<sup>r</sup> ) *and* (F , ˜ T˜<sup>s</sup> )*, we have*

$$\text{strhom}\left(\left(F, T^r\right), \left(\tilde{F}, \tilde{T}^s\right)\right) = \sum_{\left(\hat{F}, \hat{T}^t\right) \in \mathcal{S}^{pt}} \frac{\text{strsurj}\left(\left(F, T^r\right), \left(\hat{F}, \hat{T}^t\right)\right) \cdot \text{strinj}\left(\left(\hat{F}, \hat{T}^t\right), \left(\tilde{F}, \tilde{T}^s\right)\right)}{\text{aut}\left(\hat{F}, \hat{T}^t\right)},$$

*where* aut(F , ˆ Tˆ<sup>t</sup> ) *denotes the number of automorphism of* (F , ˆ Tˆ<sup>r</sup> )*. Here, the summation ranges over all non-isomorphic (parallel-tree decomposed) graphs in* S pt *and is well-defined as there are only a finite number of graphs making the value in the summation non-zero.*

*Proof.* We initially define the set S as the set of triples((F , ˆ Tˆ<sup>t</sup> ),(ρ, τ ),(ϕ, ψ)) that satisfy (F , ˆ Tˆ<sup>t</sup> ) ∈ S pt , (ρ, τ ) ∈ strSurj((F, T<sup>r</sup> ),(F , ˆ Tˆ<sup>t</sup> )), and (ϕ, ψ) ∈ strInj((F , ˆ Tˆ<sup>t</sup> ),(F , ˜ T˜<sup>s</sup> )). We define a mapping σ such that σ((F , ˆ Tˆ<sup>t</sup> ),(ρ, τ ),(ϕ, ψ)) = (ϕ ◦ ρ, ψ ◦ τ ) for all ((F , ˆ Tˆ<sup>t</sup> ),(ρ, τ ),(ϕ, ψ)) ∈ S. Our goal is to prove that σ is a mapping from S to strHom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>s</sup> )). Moreover, we aim to show that σ((Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> ),(ρ1, τ1),(ϕ1, ψ1)) = σ((Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> ),(ρ2, τ2),(ϕ2, ψ2)) if and only if there exists an isomorphism (ˆρ, τˆ) from (Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> 1 ) to (Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> 2 ) such that ρˆ◦ ρ<sup>1</sup> = ρ2, τˆ ◦ τ<sup>1</sup> = τ2, ϕ<sup>1</sup> = ϕ<sup>2</sup> ◦ ρˆ, and ψ<sup>1</sup> = ψ<sup>2</sup> ◦ τˆ.

We will prove these statements one by one. We first prove that σ is a mapping from S to strHom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>s</sup> )). This simply follows from the fact that strSurj and strInj are both strHom, and the composition of two strHoms are still a strHom.

Next, we will prove that σ is surjective. Given (˜ρ, τ˜) ∈ strHom((F, T<sup>r</sup> ),(F , ˜ T˜<sup>s</sup> )), we define (F , ˆ Tˆ<sup>r</sup> ), (ρ, τ ), and (ϕ, ψ) as follows:

- 1. We define Fˆ as the subgraph of F˜ induced by ρ˜(V<sup>F</sup> ), and we define Tˆ<sup>t</sup> as the subgraph of T˜<sup>s</sup> induced by τ <sup>H</sup>(V<sup>T</sup> ). We clearly have (F , ˆ Tˆ<sup>t</sup> ) ∈ Spt .
- 2. Let ρ = ˜ρ and τ = ˜τ . Obviously, (ρ, τ ) is a strSurj from (F, T<sup>r</sup> ) to (F , ˆ Tˆ<sup>t</sup> ).
- 3. Define identity mappings ϕ(u) = u for all u ∈ VF<sup>ˆ</sup> for ψ(t) = t for all t ∈ VT<sup>ˆ</sup>. Obviously, (ϕ, ψ) is a strInj from (F , ˆ Tˆ<sup>t</sup> ) to (F , ˜ T˜<sup>s</sup> ).

We clearly have ρ˜ = ϕ ◦ ρ and τ˜ = ψ ◦ τ . Thus, σ is a surjection.

We will now prove that σ((Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> 1 ),(ρ1, τ1),(ϕ1, ψ1))=σ((Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> 2 ),(ρ2, τ2),(ϕ2, ψ2)) iff there exist an isomorphism (ˆρ, τˆ) from (Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> 1 ) to (Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> 2 ) such that ρˆ ◦ ρ<sup>1</sup> = ρ2, τˆ ◦ τ<sup>1</sup> = τ2, ϕ<sup>1</sup> = ϕ<sup>2</sup> ◦ ρˆ, ψ<sup>1</sup> = ψ<sup>2</sup> ◦ τˆ. It suffices to prove only one direction, namely, σ((Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> 1 ),(ρ1, τ1),(ϕ1, ψ1))=σ((Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> ),(ρ2, τ2),(ϕ2, ψ2)) implies that there exist an isomorphism (ˆρ, τˆ) from (Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> 1 ) to (Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> 2 ) such that ρˆ◦ρ<sup>1</sup> = ρ2, τˆ◦τ<sup>1</sup> = τ2, ϕ<sup>1</sup> = ϕ<sup>2</sup> ◦ρˆ, ψ<sup>1</sup> = ψ<sup>2</sup> ◦τˆ.

- 1. We first prove that Fˆ 1 ∼= Fˆ <sup>2</sup> and Tˆ<sup>t</sup><sup>1</sup> 1 ∼= Tˆ<sup>t</sup><sup>2</sup> 2 . For any u, v ∈ V<sup>F</sup> , if ρ1(u) ̸= ρ1(v), then ϕ<sup>1</sup> ◦ ρ1(u) ̸= ϕ<sup>1</sup> ◦ ρ1(v) since ϕ is an injection. Therefore, ϕ<sup>2</sup> ◦ ρ2(u) ̸= ϕ<sup>2</sup> ◦ ρ2(v), and thus ρ2(u) ̸= ρ2(v). By symmetry, we also have that ρ2(u) ̸= ρ2(v) implies ρ1(u) ̸= ρ1(v). This proves that ρ1(u) = ρ1(v) iff ρ2(u) = ρ2(v). For any u, v ∈ V<sup>F</sup> , if {ρ1(u), ρ1(v)} ∈ EFˆ<sup>1</sup> , then {u, v} ∈ E<sup>F</sup> since ρ<sup>1</sup> is a surjection. Therefore, {ρ2(u), ρ2(v)} ∈ EFˆ<sup>2</sup> since ρ<sup>2</sup> is a homomorphism.

By symmetry, it follows that if {ρ2(u), ρ2(v)} ∈ EFˆ<sup>2</sup> , then {ρ1(u), ρ1(v)} ∈ EFˆ<sup>1</sup> . Therefore, we conclude that Fˆ 1 ∼= Fˆ <sup>2</sup>. Similarly, it follows that Tˆ<sup>t</sup><sup>1</sup> ∼= Tˆ<sup>t</sup><sup>2</sup> .

- 1 2
- 2. Consequently, there exist isomorphism ρˆ and τˆ such that ρˆ◦ ρ<sup>1</sup> = ρ2, τˆ τ<sup>1</sup> = τ2. For any node q ∈ V<sup>T</sup> ,

$$\hat{\rho}(\beta_{\hat{T}_1}(\tau_1(q))) = \hat{\rho} \circ \rho_1(\beta_T(q)) = \rho_2(\beta_T(q)) = \beta_{\hat{T}_2}(\tau_2(q)) = \beta_{\hat{T}_2}(\hat{\tau} \circ \tau_1(q)).$$

Moreover, for any {q1, q2} ∈ E<sup>T</sup> ,

ρˆ(γTˆ<sup>1</sup> (τ1(q1, q2))) = ˆρ ◦ ρ1(γ<sup>T</sup> (q1, q2)) = ρ2(γ<sup>T</sup> (q1, q2)) = γTˆ<sup>2</sup> (τ2(q1, q2)) = γTˆ<sup>2</sup> (ˆτ ◦ τ1(q1, q2)).

$$\hat{\rho}(\gamma_{\hat{T}_1}(\tau_1(q_1, q_2))) = \hat{\rho} \circ \rho_1(\gamma_{T}(q_1, q_2)) = \rho_2(\gamma_{T}(q_1, q_2)) = \gamma_{\hat{T}_2}(\tau_2(q_1, q_2)) = \gamma_{\hat{T}_2}(\hat{\tau} \circ \tau_1(q_1, q_2)).$$

Since τ<sup>1</sup> is surjective, τ1(q) ranges over all nodes in Tˆ<sup>t</sup><sup>1</sup> <sup>1</sup> when q ranges over V<sup>T</sup> , and τ1(q1, q2) ranges over all edges in Tˆ<sup>t</sup><sup>1</sup> <sup>1</sup> when (q1, q2) ranges over E<sup>T</sup> . We thus conclude that (ρ, τ ) is an isomorphism from (Fˆ <sup>1</sup>, Tˆ<sup>t</sup><sup>1</sup> ) to (Fˆ <sup>2</sup>, Tˆ<sup>t</sup><sup>2</sup> )

- 3. We finally prove that ϕ<sup>1</sup> = ϕ<sup>2</sup> ρˆ and ψ<sup>1</sup> = ψ<sup>2</sup> τˆ. Pick any u ∈ V<sup>F</sup> , we have ϕ<sup>2</sup> ρρˆ <sup>1</sup>(u) = ϕ<sup>2</sup> ◦ ρ2(u) = ϕ<sup>1</sup> ◦ ρ1(u). Since ρ<sup>1</sup> is surjective, ϕ1(u) ranges over all vertices in Fˆ <sup>1</sup> when u ranges over V<sup>F</sup> . This proves that ϕ<sup>1</sup> = ϕ<sup>2</sup> ◦ ρˆ. Following the same procedure, we can prove that ψ<sup>1</sup> = ψ<sup>2</sup> ◦ τˆ.

Combining the above three items concludes the proof.

From Lemma [B.17,](#page-22-0) we can also obtain the finite-iteration version of Lemma [B.17](#page-22-0) as follows:

Lemma B.18. *For any parallel-tree decomposed graph* (F, T<sup>r</sup> ) ∈ Spt d *and* F , ˜ T˜<sup>s</sup> ∈ Spt d *, we have*

$$\text{strhom}\left(\left(F, T^r\right), \left(\tilde{F}, \tilde{T}^s\right)\right) = \sum_{\left(\hat{F}, \hat{T}^t\right) \in \mathcal{S}_d^{\text{pt}}} \frac{\text{strsurj}\left(\left(F, T^r\right), \left(\hat{F}, \hat{T}^t\right)\right) \cdot \text{strinj}\left(\left(\hat{F}, \hat{T}^t\right), \left(\tilde{F}, \tilde{T}^s\right)\right)}{\text{aut}\left(\hat{F}, \hat{T}^t\right)},$$

*Proof.* According to the third condition in Definition [B.13,](#page-21-1) for (F , ˆ Tˆ<sup>r</sup> ) ∈ Spt, if strsurj((F, T<sup>r</sup> ),(F , ˆ Tˆ<sup>t</sup> )) · strinj((F , ˆ Tˆ<sup>t</sup> ),(F , ˜ T˜<sup>s</sup> )) ̸= 0, it follows that (F , ˆ Tˆ<sup>r</sup> ) ∈ Spt d . Therefore, the conclusion of the lemma is immediate.

Definition B.19. We can list all non-isomorhpic parallel-tree decomposed graphs into an infinite sequence (F1, T<sup>r</sup><sup>1</sup> 1 ),(F2, T<sup>r</sup><sup>2</sup> 2 ), . . . with the following order.

- The order requires |V<sup>T</sup><sup>i</sup> | ≤ |VT<sup>j</sup> | for any i < j.
- If |V<sup>T</sup><sup>i</sup> | = |VT<sup>j</sup> | for any i < j, then |F<sup>T</sup><sup>i</sup> | ≤ |F<sup>T</sup><sup>j</sup> |.

Then we define following function matrix and function vector based on the order defined above.

- 1. Let f : S pt × Spt → <sup>N</sup> be any mapping. Define the associated matrix M<sup>f</sup> ∈ <sup>N</sup> <sup>N</sup>+×N<sup>+</sup> , where A f i,j = f (F<sup>i</sup> , T<sup>r</sup><sup>i</sup> i ),(F<sup>j</sup> , T<sup>r</sup><sup>j</sup> j ) . Similarly, we consider the finite-iteration version. Let f : S pt <sup>d</sup> × Spt <sup>d</sup> → <sup>N</sup> be any mapping. Define the associated matrix <sup>M</sup><sup>f</sup> ∈ <sup>N</sup> <sup>N</sup>+×N<sup>+</sup> , where M f,(d) i,j = f (F<sup>i</sup> , T<sup>r</sup><sup>i</sup> i ),(F<sup>j</sup> , T<sup>r</sup><sup>j</sup> j ) .
- 2. Let g : S pt × G → <sup>N</sup> be any mapping. Given a graph G ∈ G, define the (infinite) vector l g <sup>G</sup> ∈ <sup>N</sup> <sup>N</sup><sup>+</sup> , where l g G,i = g ((F<sup>i</sup> , T<sup>r</sup><sup>i</sup> i ), G). For the finite-iteration version, let g : S pt <sup>d</sup> × G → <sup>N</sup> be any mapping. Given a graph G ∈ G, define the (infinite) vector l g,(d) <sup>G</sup> ∈ <sup>N</sup> <sup>N</sup><sup>+</sup> , where l g,(d) G,i = g ((F<sup>i</sup> , T<sup>r</sup><sup>i</sup> i ), G).
- 3. Let h : G×G → N be any mapping. Given a graph G ∈ G, define the (infinite) vector l h <sup>G</sup> ∈ <sup>N</sup> <sup>N</sup><sup>+</sup> , where l h G,i = h(F<sup>i</sup> , G). In the finite-iteration setting, let h : G × G → N be any mapping. Given a graph G ∈ G, define the (infinite) vector l h,(d) <sup>G</sup> ∈ <sup>N</sup> <sup>N</sup><sup>+</sup> , where l h,(d) G,i = h(F<sup>i</sup> , G).

Theorem B.20. *For any two graphs* G *and* H*, we have* hom((F, T<sup>r</sup> ), G) = hom((F, T<sup>r</sup> ), H) *for all parallel-tree decomposed graphs* (F, T<sup>r</sup> ) *iff* treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) *for all parallel-tree decomposed graphs. Similarly, in the finite-iteration setting,* hom((F, T<sup>r</sup> ), G) = hom((F, T<sup>r</sup> ), H) *holds for all* (F, T<sup>r</sup> ) ∈ Spt d *iff* treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) *for all* (F, T<sup>r</sup> ) ∈ Spt d *.*

*Proof.* We consider each direction separately.

- 1. First, we prove that if treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) for all parallel-tree decomposed graphs, then hom((F, T<sup>r</sup> ), G) = hom((F, T<sup>r</sup> ), H) for all such graphs (F, T<sup>r</sup> ). According to Theorem [B.14,](#page-21-0) this result can be expressed in matrix form as l hom <sup>G</sup> = Mstrhom · l treeCount <sup>G</sup> and l hom <sup>H</sup> = Mstrhom · l treeCount <sup>H</sup> for all parallel trees F. This directly implies that l treeCount <sup>G</sup> = l treeCount <sup>H</sup> leads to l hom <sup>G</sup> = l hom <sup>H</sup> . Similarly, in the finite-iteration setting, the result from Theorem [B.15](#page-22-1) can be rewritten in matrix form as l hom,(d) <sup>G</sup> <sup>=</sup> <sup>M</sup>strhom,(d) · l treeCount,(d) <sup>G</sup> . Therefore, if l treeCount,(d) <sup>G</sup> = l treeCount,(d) <sup>H</sup> , it follows that l hom,(d) <sup>G</sup> = l hom,(d) <sup>H</sup> .

- 2. For the second direction of the lemma, it suffices to prove the finite-iteration setting, as the general case directly follows. According to Lemma [B.18,](#page-24-1) we have the following equations:

$$\begin{aligned} l_G^{\text{strhom},(d)} &= M^{\text{strsurj},(d)} \cdot M^{\text{strinj},(d)} \cdot (M^{\text{aut},(d)})^{-1} \cdot l_G^{\text{treeCount},(d)}, \\ l_H^{\text{strhom},(d)} &= M^{\text{strsurj},(d)} \cdot M^{\text{strinj},(d)} \cdot (M^{\text{aut},(d)})^{-1} \cdot l_H^{\text{treeCount},(d)}. \end{aligned}$$

By simple observation, Maut is a diagonal matrix where all diagonal elements are positive integers. Moreover, Mstrinj is an upper triangular matrix with positive diagonal elements. This holds because strinj((F<sup>i</sup> , T<sup>r</sup><sup>i</sup> i ),(F<sup>j</sup> , T<sup>r</sup><sup>j</sup> j )) > 0 only when |V<sup>T</sup><sup>i</sup> | ≤ |V<sup>T</sup><sup>j</sup> |. Since Mstrsurj,(d) is a lower triangular matrix with positive diagonal elements, it is invertible. Thus,

$$M^{\text{strinj},(d)} \cdot l_G^{\text{treeCount},(d)} = M^{\text{strinj},(d)} \cdot l_H^{\text{treeCount},(d)}.$$

Additionally, by the definition of an unfolding tree, there are only finitely many non-zero elements in both l treeCount,(d) <sup>G</sup> and l treeCount,(d) <sup>H</sup> , and the corresponding non-zero indices are restricted to a fixed (finite) set. In this case, the upper triangular matrix Mstrinj,(d) reduces to a finite-dimensional matrix, so we conclude that l treeCount,(d) <sup>G</sup> = l treeCount,(d) <sup>H</sup> . By enumerating over all d ≥ 0, we obtain that l treeCount <sup>G</sup> = l treeCount <sup>H</sup> .

Combining item 1 and item 2, we finish the proof of the lemma.

#### B.4 STEP 3: FINDING PEBBLE GAME FOR SPECTRAL INVARIANT GNN

In this section, we introduce the pebble game and demonstrate its equivalence to the expressive power of spectral invariant GNN.

#### B.4.1 PEBBLE GAME

We first formally define the rules of pebble game.

Definition B.21 (Pebble game for spectral invariant GNN). The pebbling game is conducted on two graphs G = (VG, EG) and H = (VH, EH). Initially, each graph is equipped with two distinct pebbles, denoted as u and v, which start off outside the graphs. The game involves two players: the Spoiler and the duplicator. We now describe the procedure of the game as follows:

- *Initialization:*The Spoiler first selects a non-empty subset V <sup>S</sup> from either V<sup>G</sup> or VH, and the duplicator responds with a subset V <sup>D</sup> from the other graph, ensuring that |V <sup>D</sup>| = |V <sup>S</sup>|. The duplicator loses the game if no feasible choice is available. The Spoiler places a pebble u on a vertex in V <sup>D</sup>, and the duplicator places a corresponding pebble u in V
  - <sup>S</sup>. Similarly, the Spoiler and duplicator repeat the process to place two pebbles, v. Specifically, the Spoiler selects a non-empty subset V <sup>S</sup> from either V<sup>G</sup> or VH, and the duplicator responds by selecting a subset V <sup>D</sup> from the other graph, maintaining |V <sup>S</sup>| = |V <sup>D</sup>|. The Spoiler then places v on a vertex in V <sup>D</sup>, while the duplicator places the corresponding v in V
    - S.
- *Main Process:* The game iteratively repeats the following steps, where, in each iteration, the Spoiler may choose freely between the following two actions:
  - 1. Action 1 (moving pebble v): The Spoiler first selects a non-empty subset V <sup>S</sup> from either V<sup>G</sup> or VH, and the duplicator responds with a subset V <sup>D</sup> from the other graph, ensuring that |V <sup>D</sup>| = |V <sup>S</sup>|. The Spoiler then moves pebble v to a vertex in V <sup>D</sup>, and the duplicator moves the corresponding pebble v to a vertex in V
    - S.
  - 2. Action 2 (moving pebble u): The Spoiler first selects a non-empty subset V <sup>S</sup> from either V<sup>G</sup> or VH, and the duplicator responds with a subset V <sup>D</sup> from the other graph, ensuring that |V <sup>D</sup>| = |V <sup>S</sup>|. The Spoiler then moves pebble u to a vertex in V <sup>D</sup>, and the duplicator moves the corresponding pebble u to a vertex in V
    - S.
- *Termination:* The Spoiler wins if, after a certain number of rounds, ω ⋆ <sup>G</sup>(u, v) for graph G differs from ω ⋆ <sup>H</sup>(u, v) for graph H. Conversely, the duplicator wins if the Spoiler is unable to achieve a win after any number of rounds.

#### B.4.2 EQUIVALENCE BETWEEN SPECTRAL GNNS AND PEBBLING GAMES

Lemma B.22. *Let* l ∈ <sup>N</sup> *be any integer. For any vertices* uG, v<sup>G</sup> ∈ V<sup>G</sup> *and* uH, v<sup>H</sup> ∈ VH*, if* χ Walk,(l) <sup>G</sup> (u) ̸= χ Walk,(l) <sup>H</sup> (v)*, then the Spoiler can win the game in* l − 1 *rounds when the two pebbles* u *are initially placed on vertices* u<sup>G</sup> ∈ V<sup>G</sup> *and* u<sup>H</sup> ∈ V<sup>H</sup> *in graphs* G *and* H*, respectively.*

*Proof.* The proof proceeds by induction on l. First, consider the base case where l = 0. In this case, the statement is trivially true.

Now, assume that the lemma holds for all l ≤ L, and consider the case where l = L + 1. Suppose χ Walk,(L+1) <sup>G</sup> (uG) ̸= χ Walk,(L+1) <sup>H</sup> (uH). If χ Walk,(L) <sup>G</sup> (uG) ̸= χ Walk,(L) <sup>H</sup> (uH), then by the inductive hypothesis, Spoiler wins. Otherwise, we have

$$\{\{(\omega_G^*(u_G, v_G), \chi_G^{\text{Walk}, (L)}(v_G)) : v_G \in \mathcal{V}_G\} \neq \{\{(\omega_H^*(u_H, v_H), \chi_H^{\text{Walk}, (L)}(v_H)) : v_H \in \mathcal{V}_H\}\}.$$

Therefore, there exists a color c and x ∈ R |VG| such that |CG(uG, c, x)| ̸= |CH(uH, c, x)|, where

$$\mathcal{C}_G(u_G, c, x) = \left\{ v_G \in \mathcal{V}_G : \chi_G^{\text{Walk}, (L)}(v_G) = c, \omega_G^*(u_G, v_G) = x \right\}.$$

If |CG(uG, c, x)| > |CH(uH, c, x)|, the Spoiler can select the vertex subset V <sup>S</sup> = CG(uG, c, x) ⊂ VG. Regardless of how the Duplicator responds with a subset V <sup>D</sup> ⊂ VH, there exists a vertex v<sup>H</sup> ∈ V <sup>D</sup> such that (ω ⋆ <sup>H</sup>(uH, vH), χ Walk,(L) <sup>H</sup> (vH)) ̸= (x, c). The Spoiler then selects this vertex x <sup>S</sup> = vH, and no matter how the Duplicator responds with x <sup>D</sup> = v<sup>G</sup> ∈ V <sup>S</sup>, we have either ω ⋆ <sup>G</sup>(uG, vG) ̸= ω ⋆ <sup>H</sup>(uH, vH) or χ Walk,(L) <sup>G</sup> (vG) ̸= χ Walk,(L) <sup>H</sup> (vH). If ω ⋆ <sup>G</sup>(uG, vG) ̸= ω ⋆ <sup>H</sup>(uH, vH), the Spoiler wins the game immediately. If χ Walk,(L) <sup>G</sup> (vG) ̸= χ Walk,(L) <sup>H</sup> (vH), the remainder of the game is equivalent to one where the two pebbles u are initially placed on v<sup>G</sup> ∈ V<sup>G</sup> and v<sup>H</sup> ∈ V<sup>H</sup> in graphs G and H respectively. By the inductive hypothesis, the Spoiler wins the game.

If |CG(uG, c, x)| < |CH(uH, c, x)|, Spoiler can select the vertex subset V <sup>S</sup> = CH(uH, c, x) ⊂ VH, and the conclusion follows analogously.

Lemma B.23. *For any vertices* u<sup>G</sup> ∈ V<sup>G</sup> *and* u<sup>H</sup> ∈ VH*, if* χ Walk,(l+1) <sup>G</sup> (uG) = χ Walk,(l+1) <sup>H</sup> (uH)*, then the Spoiler cannot win the game within* l *rounds when the two pebbles are initially placed on vertices* u<sup>G</sup> ∈ V<sup>G</sup> *and* u<sup>H</sup> ∈ V<sup>H</sup> *in graphs* G *and* H*, respectively.*

*Proof.* The proof proceeds by induction on l. The base case l = 0 is trivially true. Now, assume the statement holds for l ≤ L, and consider the case l = L + 1. Suppose χ Walk,(L+2) <sup>G</sup> (uG) = χ Walk,(L+2) <sup>H</sup> (uH). Then,

$$\{ \left( \omega_G^*(u_G, v_G), \chi_G^{\text{Walk}, (L+1)}(v_G) \right) : v_G \in \mathcal{V}_G \} = \{ \left( \omega_H^*(u_H, v_H), \chi_H^{\text{Walk}, (L+1)}(v_H) \right) : v_H \in \mathcal{V}_H \}.$$

If Spoiler selects a subset V <sup>S</sup>, and if V <sup>S</sup> ⊂ VG, Duplicator can respond with a subset V <sup>D</sup> ⊂ V<sup>H</sup> such that

$$\{ \left( \omega_G^*(u_G, v_G), \chi_G^{\text{Walk}, (L+1)}(v_G) \right) : v_G \in V^S \} = \{ \left( \omega_H^*(u_H, v_H), \chi_H^{\text{Walk}, (L+1)}(v_H) \right) : v_H \in V^D \}.$$

Similarly, if V <sup>S</sup> ⊂ VH, Duplicator can respond with a subset V <sup>D</sup> ⊂ V<sup>G</sup> such that

$$\{ \left( \omega_G^*(u_G, v_G), \chi_G^{\text{Walk}, (L+1)}(v_G) \right) : v_G \in V^D \} = \{ \left( \omega_H^*(u_H, v_H), \chi_H^{\text{Walk}, (L+1)}(v_H) \right) : v_H \in V^S \}.$$

In both cases, it is clear that |V <sup>S</sup>| = |V <sup>D</sup>|. Next, regardless of how Spoiler moves the pebble v to a vertex x <sup>S</sup> ∈ V <sup>D</sup>, Duplicator can always respond by moving the corresponding pebble v to a vertex x <sup>D</sup> ∈ V <sup>S</sup>, such that

$$\left(\omega_G^*(u_G, \tilde{v}_G), \chi_G^{\text{Walk}, (L+1)}(\tilde{v}_G)\right) = \left(\omega_H^*(u_H, \tilde{v}_H), \chi_H^{\text{Walk}, (L+1)}(\tilde{v}_H)\right),$$

where (˜vG, v˜H) represents the new positions of the pebbles. The remaining game is then equivalent to a game in which the two pebbles are initially placed on vertices v˜<sup>G</sup> ∈ V<sup>G</sup> and v˜<sup>H</sup> ∈ V<sup>H</sup> in graphs G and H, respectively.

Combining previous two lemmas, we have the following result:

Lemma B.24. *Given graph* G *and* H*, Spoiler cannot wins the pebble game in* d *steps iff* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H)*.*

Therefore, we have proven Lemma [3.19](#page-7-1) in the main paper.

#### B.5 STEP 4: INTRODUCING FURER GRAPHS ¨

To continue, we draw introduce Furer graphs, and we further prove that pebble games restricted on ¨ Furer graphs can be greatly simplified. ¨

### B.5.1 PROPERTIES OF FURER GRAPHS ¨

We first introduce the definition of Furer graphs, introduced by ¨ [Furer](#page-11-5) ¨ [\(2001\)](#page-11-5).

Definition B.25 (Connected components). Let F = (V<sup>F</sup> , E<sup>F</sup> ) be a connected graph, and let U ⊂ V<sup>F</sup> be a set of vertices, referred to as separation vertices. We define two edges {u, v}, {x, y} ∈ E<sup>F</sup> as belonging to the same connected component if there exists a simple path {{y0, y1}, {y1, y2}, . . . , {yk−1, yk}} such that {y0, y1} = {u, v}, {yk−1, yk} = {x, y}, and y<sup>i</sup> ∈/ U for all i ∈ [1, k −1]. It is straightforward to verify that this relation between edges induces an *equivalence relation*. Consequently, the edge set E<sup>F</sup> can be partitioned into disjoint subsets, denoted by CC<sup>F</sup> (U) = {P<sup>i</sup> : i ∈ [m]}, where each P<sup>i</sup> ⊂ E<sup>F</sup> represents a connected component for some m.

Definition B.26 (Furer graphs ¨ ). Given any connected graph F = (V<sup>F</sup> , E<sup>F</sup> ), the Furer graph ¨ G(F) = (VG(<sup>F</sup> ) , EG(<sup>F</sup> )) is constructed as follows:

$$V_{G(F)} = \{(x, X) : x \in V_F, X \subset N_F(x), |X| \bmod 2 = 0\},$$

$$E_{G(F)} = \{\{(x, X), (y, Y)\} \subset V_G : \{x, y\} \in E_F, (x \in Y \leftrightarrow y \in X)\}.$$

Here, x ∈ Y ↔ y ∈ X holds when either (x ∈ Y and y ∈ X) or (x /∈ Y and y /∈ X) holds. For each x ∈ V<sup>F</sup> , denote the set

$$\text{Meta}_F(x) := \{(x, X) : X \subset N_F(x), |X| \bmod 2 = 0\}, \quad (4)$$

which is called the meta vertices of G(F) associated to x. Note that VG(<sup>F</sup> ) = S <sup>x</sup>∈V<sup>F</sup> Meta<sup>F</sup> (x).

We next define an operation called "twist":

Definition B.27 (Twist). Let G(F) = (VG(<sup>F</sup> ) , EG(<sup>F</sup> )) be the Furer graph of ¨ F = (V<sup>F</sup> , E<sup>F</sup> , ℓ<sup>F</sup> ), and let {x, y} ∈ E<sup>F</sup> be an edge of F. The *twisted* Furer graph of ¨ G(F) for edge {x, y}, is constructed as follows: twist(G(F), {x, y}) := (VG(<sup>F</sup> ) , Etwist(G(<sup>F</sup> ),{x,y})), where

$$E_{\text{twist}(G(F), \{x, y\})} := E_{G(F)} \triangle \{\{\xi, \eta\} : \xi \in \text{Meta}_F(x), \eta \in \text{Meta}_F(y)\},$$

and △ is the symmetric difference operator, i.e., A△B = (A\B) ∪ (B\A). For an edge set S = {e1, · · · , ek} ⊂ E<sup>F</sup> , we further define

$$\text{twist}(G(F), S) := \text{twist}(\cdots \text{twist}(G(F), e_1) \cdots, e_k). \quad (5)$$

Note that Equation [\(5\)](#page-27-1) is well-defined as the resulting graph does not depend on the order of edges e1, · · · , e<sup>k</sup> for twisting.

The following result is well-known (see e.g., [Zhang et al.,](#page-13-3) [2023a,](#page-13-3) Corollary I.5 and Lemma I.7)):

Theorem B.28. *For any connected graph* F *and any set* S1, S<sup>2</sup> ⊂ E<sup>F</sup> *,* twist(G(F), S1) ≃ twist(G(F), S2) *iff* |S1| ≡ |S2| (mod2)*.*

We now present an essential property of Furer graphs in terms of walk number: ¨

Theorem B.29. *Let* G(F) = (VG, EG) *be the Furer graph of ¨* F = (V<sup>F</sup> , E<sup>F</sup> )*, and let* H(F) = *twist*(G(F), E) *for some* E ⊂ E<sup>F</sup> *. Given* (x, X ),(y, Y) ∈ VG*, and a connected component* P ∈ CC<sup>F</sup> ({x, y}) *with* |P ∩ E| = 1*, the number of* n*-walks from* (x, X ) *to* (y, Y)*, passing through* Meta(x1), Meta(x2), . . . , Meta(xn) *sequentially in* G(F)*, is equal to the number of such walks in* H(F) *for all* n ∈ N <sup>+</sup> *and vertices* x1, . . . , x<sup>n</sup> *on* P*, iff* P *is a path.*

*Proof.* If P is a path, we denote P = {{x1, x2}, . . . , {xn−1, xn}} with x<sup>1</sup> = x and x<sup>n</sup> = y. It follows that the number of n-walks starting from (x, X ) and ending at (y, Y), passing through Meta(x1), . . . , Meta(xn) sequentially on G(F), is not equal to the number of such walks on H(F). Thus, one direction of the lemma is established.

If P is not a path, then there exists at least one vertex, besides x and y, on P whose degree is greater than 2. We define ω G(F ) <sup>n</sup> ((x, X ), Meta(x2), . . . , Meta(xn−1),(y, Y)) and ω H(F ) <sup>n</sup> ((x, X ), Meta(x2), . . . , Meta(xn−1),(y, Y)) as the number of n-walks starting from (x, X ), ending at (y, Y), and passing through Meta<sup>F</sup> (x1), . . . , Meta<sup>F</sup> (xn) sequentially in G(F) and H(F), respectively. We use the notation deg<sup>F</sup> (v) to denote the degree of a vertex v in the graph F. We proceed by induction on n to prove the following stronger statement: If the degrees of x2, . . . , xn−<sup>1</sup> are not all less than or equal to 2, then there exists a function f F n : V n <sup>F</sup> → <sup>N</sup> such that

$$\begin{aligned} &\omega_n^{G(F)}((x, \mathcal{X}), \text{Meta}(x_2), \dots, \text{Meta}(x_{n-1}), (y, \mathcal{Y})) \\ &= \omega_n^{H(F)}((x, \mathcal{X}), \text{Meta}(x_2), \dots, \text{Meta}(x_{n-1}), (y, \mathcal{Y})) = f_n(x_1, \dots, x_n) \end{aligned}$$

for all n ∈ N, (x, X ) ∈ Meta(x), and (y, Y) ∈ Meta(y).

We first consider the case when n = 2. In this case, we can straightforwardly define the function f F n as f(x1, x2, x3) = 2deg(x2)−<sup>3</sup> .

Next, assume that the statement holds for n ≤ N. We now consider the case when n = N + 1, and analyze two separate cases:

#### 1. Not all degrees of x3, x4, . . . , xn−<sup>1</sup> are less than or equal to 2.

The n-walk passing Meta<sup>F</sup> (x1), . . . , Meta<sup>F</sup> (xn) sequentially can be decomposed into a 1-walk from (x, X ) to Meta<sup>F</sup> (x2), followed by an n − 1-walk passing Meta<sup>F</sup> (x2), . . . , Meta<sup>F</sup> (xn) sequentially and ending at (y, Y). According to the induction hypothesis, the number of (n − 1)-walks passing Meta<sup>F</sup> (x2), . . . , Meta<sup>F</sup> (xn) sequentially and ending at (y, Y) equals f F n−1 (x2, x3, . . . , xn). Since the number of 1-walks from (x, X ) to Meta<sup>F</sup> (x2) equals 2 deg<sup>F</sup> (x2)−2 , we can define the function f F n as f F n (x1, x2, . . . , xn) = 2deg<sup>F</sup> (x2)−<sup>2</sup> · f F n−1 (x2, x3, . . . , xn).

- 2. All degrees of x3, x4, . . . , xn−<sup>1</sup> are less than or equal to 2. In this case, we have deg<sup>F</sup> (x2) ≥ 3. The number of (n − 1)-walks passing Meta<sup>F</sup> (x2), . . . , Meta<sup>F</sup> (xn) sequentially and ending at (y, Y) is either 1 or 0. Therefore, we can define the function f F n as f F <sup>n</sup> = 2deg<sup>F</sup> (x2)−<sup>3</sup> .

Combining the two cases, we conclude that if the degrees of x2, x3, . . . , xn−<sup>1</sup> are not all equal to 2, then there exists a function f F n : V (F) <sup>n</sup> → <sup>N</sup> such that

$$\begin{aligned} & \omega_n^{G(F)}((x, \mathcal{X}), \text{Meta}(x_2), \dots, \text{Meta}(x_{n-1}), (y, \mathcal{Y})) \\ &= \omega_n^{H(F)}((x, \mathcal{X}), \text{Meta}(x_2), \dots, \text{Meta}(x_{n-1}), (y, \mathcal{Y})) = f_n(x_1, \dots, x_n) \end{aligned}$$

for all X ∈ N<sup>F</sup> (x), Y ∈ N<sup>F</sup> (y), and any E ⊂ E<sup>F</sup> .

By combining all previous analyses, we have proven the result of the theorem.

#### B.5.2 SIMPLIFIED PEBBLE GAME ON FURER GRAPHS ¨

Definition B.30 (Simplified Pebble Game). The simplified pebble game is defined as follows. Let F = (V<sup>F</sup> , E<sup>F</sup> ) represent the base graph of a proper Furer graph. The game is played on ¨ F with two pebbles, u and v, each of a different type. Initially, both pebbles are placed outside the graph F. The game begins with Spoiler placing pebble u on any vertex of F, while pebble v remains outside the graph. The game then proceeds in cycles, following these steps: Spoiler places pebble v on any vertex of F, swaps the positions of u and v, and then places pebble v back outside the graph. Duplicator, on the other hand, maintains a subset Q of connected components, where Q ⊂ CC<sup>S</sup> (F) and S is the set of vertices in F where pebbles u and v are currently placed.

When Spoiler places a pebble on a vertex of F, one of two scenarios occurs. If CC<sup>S</sup> (F) remains unchanged, Duplicator takes no action. However, if the new pebble placement causes a connected component to split into smaller regions, Duplicator updates Q by replacing any original component P ⊂ <sup>E</sup><sup>F</sup> that splits into P1, . . . ,P<sup>k</sup> (where S<sup>k</sup> <sup>i</sup>=1 P<sup>i</sup> = P) with a subset of the newly formed components. That is, <sup>Q</sup>˜ = (Q \ P) ∪ {P<sup>j</sup><sup>1</sup> , . . . ,P<sup>j</sup><sup>l</sup> } for some j1, . . . , j<sup>l</sup> ∈ [k], ensuring that |Q˜| ≡ 1 (mod 2). In other words, Duplicator removes the old component P (if present) and adds some of the new components while preserving the parity of |Q|. When Spoiler removes a pebble and places it outside the graph, two cases arise. If CC<sup>S</sup> (F) remains unchanged, Duplicator again takes no action. However, if the removal of the pebble causes multiple connected components P1, . . . ,P<sup>k</sup> to merge into a larger component P = S<sup>k</sup> <sup>i</sup>=1 P<sup>i</sup> , Duplicator updates Q by either removing the smaller components, i.e., Q˜ = Q \ {P1, . . . ,Pk}, or adding the merged component, i.e., Q˜ = (Q \ {P1, . . . ,Pk}) ∪ P, depending on which option preserves |Q˜| ≡ 1 (mod 2). When Spoiler swaps the positions of the two pebbles, the connected components CC<sup>S</sup> (F) do not change, so Duplicator does not modify Q.

Spoiler wins the game if, after any round, Q contains a connected component that forms a path. Duplicator wins if Spoiler is unable to achieve this outcome after any number of rounds.

Lemma B.31. *Given a base graph* F*, Spoiler cannot win the simplified pebble game on* F *in* d *steps iff* χ Spec,(d+1) <sup>G</sup> (G(F)) = χ Spec,(d+1) <sup>H</sup> (H(F))*.*

The proof of Lemma [B.31](#page-29-0) follows a similar structure to the proof of Theorem 17 in [Zhang et al.](#page-13-3) [\(2023a\)](#page-13-3), and thus we omit the details here for the sake of simplicity. Notably, the main idea behind the proof is to show that the original pebble game is equivalent to the following 'half-simplified' version of the game:

Let F = (V<sup>F</sup> , E<sup>F</sup> ) be the base graph of a proper Furer graph. This version of the pebble game is ¨ also played on F, with two pebbles u and v. Initially, both pebbles are outside the graph F.

- First, we describe the rules for the Spoiler. Spoiler maintains a subset Q<sup>1</sup> ⊂ CC<sup>S</sup> (F) of connected components, where the set S consists of the vertices in F currently occupied by the pebbles u and v. (If pebble v is outside F, then S contains only the vertex where u is placed.) Initially, Spoiler places u on any vertex of F and leaves v outside the graph, maintaining Q<sup>1</sup> = {E<sup>F</sup> }. Then, the game proceeds cyclically as follows:
  - Spoiler places v on any vertex of F. Two cases arise for maintaining Q1: if CC<sup>S</sup> (F) does not change, Spoiler leaves Q<sup>1</sup> unchanged. Otherwise, the new pebble may split some connected components into smaller regions. For each original component P ⊂ E<sup>F</sup> that splits into P1, . . . ,P<sup>k</sup> with S<sup>k</sup> <sup>i</sup>=1 <sup>P</sup><sup>i</sup> <sup>=</sup> <sup>P</sup>, Spoiler updates <sup>Q</sup><sup>1</sup> to <sup>Q</sup>˜ <sup>1</sup> = (Q<sup>1</sup> \P)∪{P<sup>j</sup><sup>1</sup> , . . . ,P<sup>j</sup><sup>l</sup> }, where j1, . . . , j<sup>l</sup> ∈ [k] and |Q˜ <sup>1</sup>| ≡ 0 (mod 2). This ensures that the parity of |Q1| remains unchanged.
  - Spoiler swaps the positions of u and v, leaving Q<sup>1</sup> unchanged.
  - Spoiler removes v from the graph, leaving it outside F. Again, two cases arise for maintaining Q1: if CC<sup>S</sup> (F) does not change, Spoiler does nothing. Otherwise, several connected components P1, . . . ,P<sup>k</sup> may merge into a larger component P = S<sup>k</sup> <sup>i</sup>=1 P<sup>i</sup> . Spoiler then updates Q<sup>1</sup> to either Q˜ <sup>1</sup> = Q1\{P1, . . . ,Pk} or Q˜ <sup>1</sup> = (Q1\{P1, . . . ,Pk})∪P, whichever satisfies |Q˜ <sup>1</sup>| ≡ 0 (mod 2).
- Next, we describe the rules for the Duplicator, which are analogous to the Spoiler's rules but with a key difference: Duplicator maintains a subset Q<sup>2</sup> ⊂ CC<sup>S</sup> (F) where the parity of |Q2| is always odd. Initially, Q<sup>2</sup> = {E<sup>F</sup> }, and throughout the game, Duplicator performs the following updates:
  - When Spoiler places a pebble, Duplicator updates Q<sup>2</sup> in the same manner as Q1, but ensuring |Q˜ <sup>2</sup>| ≡ 1 (mod 2).
  - When Spoiler removes a pebble, Duplicator updates Q<sup>2</sup> as in the previous case, ensuring that the parity of |Q2| remains odd.
  - When Spoiler swaps the pebbles, Duplicator does nothing, as CC<sup>S</sup> (F) remains unchanged.

The result of the game is determined as follows: Suppose that pebbles u and v are placed on vertices of F. Spoiler maintains the subset Q1, and Duplicator maintains Q2. We then construct two twisted Furer graphs: ¨ G˜(F) = twist(G(F), E˜) and Gˆ(F) = twist(G(F), Eˆ), where |E| ˜ = |Q1| and, for each P ∈ Q1, we select a single edge E ∩ P ˜ = 1. Similarly, |E| ˆ = |Q2|, and for each P ∈ Q2,

we select a single edge E ∩ P ˆ = 1. Spoiler wins if the walk vector satisfies ω ⋆ G˜(F ) ((u, ∅),(v, ∅)) ̸= ω ⋆ Gˆ(F ) ((u, ∅),(v, ∅)), meaning that there exists an n-walk in G˜(F) from (u, ∅) to (v, ∅) that differs from the corresponding n-walk in Gˆ(F). By following a similar analysis to that in Theorem 17 of [Zhang et al.](#page-13-3) [\(2023a\)](#page-13-3), we can demonstrate that the 'half-simplified' pebble game is equivalent to the original pebble game. Specifically, the Spoiler can win in d steps in the original pebble game if and only if they can win in d steps in the half-simplified pebble game. Furthermore, since it is clear that the 'half-simplified' game is equivalent to the simplified game, we can conclude that the original game and the simplified game are equivalent on Furer graphs. ¨

#### B.6 STEP 5: PROVING THE MAXIMALITY OF HOMOMORPHISM EXPRESSIVITY

Before presenting the proof, we redefine the concept of the game state graph for clarity in the technical exposition. Notably, there is a slight difference between the definition of the game state graph here and the one in the previous section: we only consider game states with a single pebble in the game state graph.

Definition B.32. We define the game state (u, Q) as in the previous section, where u ∈ V<sup>F</sup> represents the position of the pebble, and Q is the connected component maintained by the Duplicator. The game state graph is formed by all game states (u, Q). There is an edge from (u, Q) to (˜u, Q˜) if there exists a game transition from (u, Q) to ((ˆu, vˆ), Qˆ), followed by a transition from ((ˆu, vˆ), Qˆ) to (˜u, Q˜), for some connected component set Q ⊂ˆ E<sup>F</sup> and vertex vˆ ∈ V<sup>F</sup> .

Definition B.33. A game state (u, Q) is called a terminal game state if there is a transition from (u, Q) to a game state ((u, v˜), Q˜) for some connected component set Q ⊂˜ E<sup>F</sup> and vertex v˜ ∈ V<sup>F</sup> , such that Q˜ consists only of a single path. In this case, the game state (u, Q) is called a terminal game state. It is straightforward to see that the Spoiler can win in the terminal state.

Definition B.34. Given a game state graph G<sup>S</sup> , a state (u, Q) is termed "contracted" if, for any transition (u, Q) → (u ′ , Q′ ) ∈ EG<sup>S</sup> , it holds that Q′ ⊂ Q. The state is called "strictly contracted" if, for any transition (u, Q) → (u ′ , Q′ ) ∈ EG<sup>S</sup> , it holds that Q′ ⊊ Q.

Definition B.35. A game state (u, Q) is defined as "unreachable" if any path starting from the initial state (∅, E<sup>F</sup> ) and ending at (u, Q) passes through a terminal state.

We do not need to consider unreachable states since the Spoiler always wins before reaching them.

Lemma B.36. *For any graph* F*, if the Spoiler can win the pebble game on* F*, then there exists a game state graph* G<sup>S</sup> *corresponding to a winning strategy for the Spoiler such that all reachable and non-terminal states are strictly contracted.*

*Proof.* 1. First, we prove that there exists a strategy for the Spoiler such that every reachable and non-terminal state is contracted. Since the Spoiler can win the pebble game, they can win at any reachable state (u, Q). Consider any strategy where (u, Q) is not contracted. Note that the game state graph induced by all reachable states is a Directed Acyclic Graph (DAG), so we can choose a state (u, Q) such that no path from the initial state (∅, {E<sup>F</sup> }) to (u, Q) passes through any intermediate state that is not contracted. Next, we construct a new strategy to make the state (u, Q) unreachable. We clearly have u ̸= ∅. Without loss of generality, assume there is a transition (u, Q) → (u ′ , Q′ ) such that Q′ ̸⊂ Q. Let (u0, Q0),(u1, Q1), . . . ,(u<sup>T</sup> , Q<sup>T</sup> ) be any path from the initial state (∅, {E<sup>F</sup> }) to (u, Q). We modify the strategy as follows: at state (u<sup>T</sup> <sup>−</sup>1, Q<sup>T</sup> <sup>−</sup>1), the Spoiler places the pebble p<sup>v</sup> on u ′ , swaps the pebbles at u and v, and then removes p<sup>v</sup> from the graph. This process can be repeated for every path from the initial state (∅, E<sup>F</sup> ) to the state (u, Q). In the new strategy, (u, Q) will become unreachable. However, the state (u<sup>T</sup> <sup>−</sup>1, Q<sup>T</sup> <sup>−</sup>1) may now violate the contraction condition. In this case, we recursively apply the above procedure to (u<sup>T</sup> <sup>−</sup>1, Q<sup>T</sup> <sup>−</sup>1). Note that this process will terminate after a finite number of steps, as the length of the path from the initial state to (u<sup>T</sup> <sup>−</sup>1, Q<sup>T</sup> <sup>−</sup>1) is strictly

- shorter than the path to (u, Q).
- 2. Next, we prove that every reachable and non-terminal state can be strictly contracted. Suppose, for contradiction, that (u, Q) is reachable and non-terminal, but not strictly contracted. Then there exists a transition ((u, Q) → (u ′ , Q′ )) ∈ EG<sup>S</sup> . Since u is at the boundary of all connected

components, we have u = u ′ . This implies that the game state graph is not acyclic, which contradicts the assumption that it is a DAG.

Combining the above two points, we conclude that for any given graph F, if the Spoiler can win the pebble game on F, then there exists a game state graph G<sup>S</sup> corresponding to a winning strategy for the Spoiler such that every reachable and non-terminal state is strictly contracted.

Lemma B.37. *Given any connected graph* F*, if the Spoiler can win the pebble game on* F*, then* F *is a parallel tree. Specifically, there exists a tree skeleton* T <sup>r</sup> = (VT<sup>r</sup> , ET<sup>r</sup> , βT<sup>r</sup> , γT<sup>r</sup> ) *such that* (F, T<sup>r</sup> ) ∈ Spt *.*

*Proof.* Let G<sup>S</sup> be the game state graph satisfying Lemma [B.36.](#page-30-1) For each game state s, denote nextG<sup>S</sup> (s) as the set of states s ′ such that (s, s′ ) is a transition in G<sup>S</sup> and s ′ contains only a single component, i.e., s ′ has the form (u, {P}). By definition, nextG<sup>S</sup> (∅, {E<sup>F</sup> }) = {(u, Q1), . . . ,(u, Qm)} for some u ∈ V<sup>F</sup> , where Q1, . . . , Q<sup>m</sup> is the finest partition of CC<sup>F</sup> ({u}).

The tree T <sup>r</sup> will be recursively constructed as follows. First, create the tree root r with β<sup>T</sup> (r) = u. As will be explained later, the root node will be associated with the set of states S(r) := nextG<sup>S</sup> (∅, {E<sup>F</sup> }). We then proceed with the following procedure:

Let t be a leaf node in the current tree associated with a non-empty set of game states S(t) such that | ∪(u,{<sup>P</sup> })∈S(t) P| > 1. For each state (u, {P}) ∈ S(t), create a new node t˜ and set its parent to be t. Pick any state (v, {P ′}) ∈ nextG<sup>S</sup> (u, {P}), and set β<sup>T</sup> (t˜) = v. Then, node t˜will be associated with the set of states <sup>S</sup>(t˜) = {(v, {P˜}) : (v, {P˜}) ∈ nextG<sup>S</sup> (u, {P})}.

We now prove that T r is indeed a valid tree skeleton for F. By definition of a parallel tree, when constructing T r and defining the label function β<sup>T</sup> : VT<sup>r</sup> → V<sup>F</sup> , we can naturally define the label function for edges γ<sup>T</sup> : ET<sup>r</sup> → 2 <sup>E</sup><sup>F</sup> . For any edge (t1, t2) ∈ ET<sup>r</sup> , there exist only paths connecting β<sup>T</sup> (t1) and β<sup>T</sup> (t2) in F. Therefore, the image of (t1, t2) is naturally defined as the set of paths connecting β<sup>T</sup> (t1) and β<sup>T</sup> (t2). Since β<sup>T</sup> is already defined for the nodes of T r , it remains to prove that for every edge (t1, t2) ∈ ET<sup>r</sup> , there exist only paths connecting β<sup>T</sup> (t1) and β<sup>T</sup> (t2) in F.

We revisit the construction of T r . Let t be a leaf node associated with the game states S(t). For each game state (u, {P}) ∈ S(t), create a new node t˜ and set its parent to t. Pick any state (v, {P ′}) ∈ nextG<sup>S</sup> (u, {P}), and set β<sup>T</sup> (t˜) = v. Since (v, {P ′}) ∈ nextG<sup>S</sup> (u, {P}), the transition ((u, {P}),(v, {P ′})) is a legal move in the pebble game.

Moreover, since we assume that G<sup>S</sup> satisfies Lemma [B.36,](#page-30-1) we can conclude that the game state (u, {P}) is strictly contracted. In other words, P ′ ⊂ P. This implies that when the Spoiler places the pebble p<sup>v</sup> on vertex v ∈ V<sup>F</sup> , the Duplicator can only choose a strictly contracted connected component set. Hence, we deduce that there are only paths connecting u and v. Consequently, there exist only paths connecting β<sup>T</sup> (t) and β<sup>T</sup> (t˜).

By recursively applying this analysis throughout the construction of T r , we conclude that for every edge (t1, t2) ∈ ET<sup>r</sup> , there exist only paths connecting β<sup>T</sup> (t1) and β<sup>T</sup> (t2) in graph F.

We now prove finite-iteration version of Lemma [B.37](#page-31-0) as follows:

Lemma B.38. *Given any base graph* F*, Spoiler can win the simplified pebble game on* F *in* d *steps iff there exsits a parallel tree skeleton* T <sup>r</sup> *of* F *such that* T <sup>r</sup> *has depth at most* d + 1*.*

*Proof.* Initially, it is evident that if F is a parallel tree with a tree skeleton of depth at most d + 1, then the Spoiler has a winning strategy in d steps. Therefore, we are left to consider the converse direction of the lemma. Now, consider the case where, for a base graph F, the Spoiler has a winning strategy in d steps. According to the analysis in Lemma [B.36,](#page-30-1) if the Spoiler has a winning strategy in d steps, then he can guarantee that all reachable non-terminal states in the game state graph G<sup>S</sup> are strictly contracted. We will prove this statement by induction. The statement trivially holds for d = 1. Assume that if the Spoiler has a winning strategy in d − 1 steps, then the base graph is a

parallel tree with a tree skeleton of depth at most d. Now, we consider the case where the Spoiler can win in d steps.

By Lemma [B.37,](#page-31-0) the Spoiler can win the game on F, implying that F is a parallel tree. Let T <sup>r</sup> be the tree skeleton of F. At the beginning of the game, we first consider the case where the Spoiler places a pebble on a vertex u such that u /∈ {v : ∃t ∈ V<sup>T</sup> , β<sup>T</sup> (t) = v}. We assume that the Duplicator selects connected component P (since F is a parallel tree, the Duplicator can only select one connected component in this case). Assume further that there exist t, t′ ∈ V<sup>T</sup> such that u is on a path connecting β<sup>T</sup> (t) and β<sup>T</sup> (t ′ ). We now consider two separate cases:

- If there is more than one path connecting β<sup>T</sup> (t) and β<sup>T</sup> (t ′ ) in F, i.e., |γ<sup>T</sup> (t, t′ )| > 1, then placing the pebble on u does not split F, and it remains as one connected component. In this case, we can directly eliminate (u, P) from the game state graph, and the remaining game state graph still represents a winning strategy for the Spoiler.
- If there is only one path connecting β<sup>T</sup> (t) and β<sup>T</sup> (t ′ ) in F, i.e., |γ<sup>T</sup> (t, t′ )| = 1, then placing the pebble on u splits the base graph F into two connected components. In this case, we replace the game state (u, {P}) in the game state graph with {u ′ , {P ′}}, where u ′ ∈ {β<sup>T</sup> (t), β<sup>T</sup> (t ′ )} and P ⊂ P ′ .

Following this discussion, we only need to consider the case where, at the beginning of the game, the Spoiler places the pebble on a vertex u ∈ V<sup>F</sup> such that there exists t ∈ V<sup>T</sup> and u = β<sup>T</sup> (t). Without loss of generality, assume u = β<sup>T</sup> (r), and the children of r are {t1, . . . , tn}. Further, assume that among all subtrees induced by t1, t2, . . . , tn, the subtree induced by t<sup>1</sup> ∈ V<sup>T</sup> has the greatest depth. We now consider the case where the Duplicator picks the connected component formed by the subtree induced by t<sup>1</sup> and the path in γ<sup>T</sup> (r, t1). If the Spoiler must ensure that the subsequent game state is strictly contracted, he must place the pebble on t1. The remaining game now reduces to a game played on the graph induced by the subtree formed by all descendants of t1. By the induction hypothesis, the subtree induced by t<sup>1</sup> has depth at most d. Thus, T <sup>r</sup> has depth at most d + 1.

We now prove Lemma [3.21](#page-7-2) from the main paper.

Lemma B.39. *For any* F /∈ FSpec,(d) *, the Spoiler cannot win the simplified pebble game on* F *in* d − 1 *steps. Consequently,* χ Spec,(d) <sup>G</sup> (G(F)) = χ Spec,(d) <sup>H</sup> (H(F))*.*

*Proof.* By Lemma [B.38,](#page-31-1) since F /∈ FSpec,(d) , the Spoiler cannot win the simplified pebble game on the base graph F. Thus, by Lemma [3.20,](#page-7-0) we conclude that χ Spec,(d) <sup>G</sup> (G(F)) = χ Spec,(d) <sup>H</sup> (H(F)).

Combining all the results from steps 1 through 5, we now conclude the proof of our main theorem.

Theorem B.40. *The homomorphism expressivity of spectral invariant GNNs with* d *iterations can be characterized as follows:*

$$\mathcal{F}^{\text{Spec},(d)} = \{F \mid F \text{ has parallel tree depth at most } d\}.$$

*Specifically, the following properties hold:*

- *For graphs* G *and* H*,* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) *if and only if, for all graphs* F *with parallel tree depth at most* d*,* hom(F, G) = hom(F, H)*.*
- F Spec,(d) *is maximal; that is, for any graph* F /∈ FSpec,(d) *, there exist graphs* G *and* H *such that* χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) *and* hom(F, G) ̸= hom(F, H)*.*

*Proof.* By Theorem [B.20](#page-24-0) and Corollary [B.10,](#page-20-1) we obtain that for graphs G and H, χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) if and only if, for all graphs F with parallel tree depth at most d, hom(F, G) = hom(F, H). Furthermore, by Lemma [3.21,](#page-7-2) there exist counterexamples G and H for any F /∈ F Spec,(d) such that χ Spec,(d) <sup>G</sup> (G) = χ Spec,(d) <sup>H</sup> (H) and hom(F, G) ̸= hom(F, H). Thus, we conclude the proof of the main theorem.

## C PROOF OF THEOREM [3.10](#page-5-2)

In this section, we provide the proof of Theorem [3.10](#page-5-2) from the main paper.

Theorem C.1. *The homomorphism expressivity of graph spectra is the set of all cycles* C<sup>n</sup> *(*n ≥ 3*) plus paths* P<sup>1</sup> *and* P2*, i.e.,* {Cn|n ≥ 3} ∪ {P1, P2}*.*

*Proof.* We separately prove that the set of all cycles satisfies the two conditions of homomorphism expressivity. For a graph G, we denote A<sup>G</sup> ∈ <sup>R</sup> |VG|×|VG| as the adjacency matrix of G, and Spec(G) = {λG,1, λG,2, . . . , λG,|VG<sup>|</sup>} as the spectrum of G.

- We first prove that for any two graphs G and H, their spectra are identical if and only if for every F ∈ {Cn|n ≥ 3} ∪ {P1, P2}, hom(F, G) = hom(F, H). Let C<sup>n</sup> denote a cycle with n vertices. For any graph G, we have hom(Cn, G) = tr(A<sup>n</sup>
  - <sup>G</sup>) for all n ∈ <sup>N</sup>≥3, and for n = 2, we denote C<sup>2</sup> = P2. Moreover, by a basic result from linear algebra, we further obtain:

$$\text{hom}(\mathcal{C}_n, G) = \text{tr}(\mathbf{A}_G^n) = \lambda_{G,1}^n + \lambda_{G,2}^n + \cdots + \lambda_{G,|V_G|}^n.$$

Therefore, if hom(Cn, G) = hom(Cn, H) for all n ∈ <sup>N</sup> <sup>+</sup>, then we have:

$$\lambda_{G,1}^n + \lambda_{G,2}^n + \cdots + \lambda_{G,|V_{G|}}^n = \lambda_{H,1}^n + \lambda_{H,2}^n + \cdots + \lambda_{H,|V_{H|}}^n, \quad \text{for all } n \in \mathbb{N}^+.$$

Thus, Spec(G) = Spec(H). Conversely, if we are given that Spec(G) = Spec(H), then:

$$\text{hom}(\mathcal{C}_n, G) = \text{tr}(\mathbf{A}_G^n) = \text{tr}(\mathbf{A}_H^n) = \text{hom}(\mathcal{C}_n, H), \quad \text{for all } n \in \mathbb{N}^+.$$

Therefore, we have proven that for any two graphs G and H, their spectra are identical if and only if for every F ∈ {Cn|n ≥ 3} ∪ {P1, P2}, hom(F, G) = hom(F, H).

- We now prove that for any graph F that is not a cycle nor a path, there exists a pair of graphs G and H such that their spectra are identical, but hom(F, G) ̸= hom(F, H). Specifically, we show that for any graph F that is not a cycle, Spec(G(F)) = Spec(H(F)) holds, where G(F) and H(F) denote the pair of Furer graphs constructed with ¨ F as the base graph.

If F is not nor a path, then there exist vertices x, y ∈ V<sup>F</sup> such that the degree of x is greater 2. We then consider the Furer graph ¨ G(F) and the twisted Furer graph ¨ H(F) = twist(G(F), {x, y}). According to ??, for vertices v, x2, . . . , x<sup>n</sup> ∈ V<sup>F</sup> and V ⊂ V<sup>F</sup> , the number of n-walks passing through (v, V), Meta(x2), . . . , Meta(xn),(v, V) sequentially in G(F) and H(F) is unequal. Specifically, x2, . . . , x<sup>n</sup> satisfy the following properties:

- 1. (v, x2),(x2, x3), . . . ,(xn−1, xn),(xn, v) ∈ E<sup>F</sup> .
- 2. The degree of x2, . . . , x<sup>n</sup> is 2 in the base graph F.
- 3. Let x<sup>1</sup> = xn+1 = v, then:

$$|\{\{x_i, x_{i+1}\}, i = 1, 2, \dots, n\} \cap \{\{x, y\}\}| \equiv 1 \pmod{2}.$$

From this, we deduce that v = x, and we have:

$$\sum_{(v, \mathcal{V}') \in \text{Meta}(v)} c_{G(F)}^n((v, \mathcal{V}'), x_2, \dots, x_n) = \sum_{(v, \mathcal{V}') \in \text{Meta}(v)} c_{H(F)}^n((v, \mathcal{V}'), x_2, \dots, x_n), \quad (6)$$

where for any vertex (v, V ′ ) ∈ Meta(v), we use notations c n G(F ) ((v, V ′ ), x2, . . . , xn) and c n H(F ) ((v, V ′ ), x2, . . . , xn) to denote the number of n-walks passing through (v, V ′ ), Meta(x2), . . . , Meta(xn),(v, V ′ ) in G(F) and H(F), respectively. If x2, . . . , x<sup>n</sup> do not satisfy the above properties, then for all (v, V ′ ) ∈ Meta(v), the number of n-walks passing through (v, V ′ ), Meta(x2), . . . , Meta(xn),(v, V ′ ) in G(F) and H(F) is equal. Thus, equation [6](#page-33-1) holds for all v, x2, . . . , x<sup>n</sup> ∈ V<sup>F</sup> . Consequently, we observe the following property in terms of walk counts:

$$\sum_{(v, \mathcal{V}') \in \text{Meta}(v)} \omega_{G(F)}^n((v, \mathcal{V}'), (v, \mathcal{V}')) = \sum_{(v, \mathcal{V}') \in \text{Meta}(v)} \omega_{H(F)}^n((v, \mathcal{V}'), (v, \mathcal{V}')),$$

where ω n G(F ) ((v, V ′ ),(v, V ′ )) and ω n H(F ) ((v, V ′ ),(v, V ′ )) denote the number of n-walks starting and ending at (v, V ′ ) in G(F) and H(F), respectively. This holds for all n ∈ N <sup>+</sup> and

![](_page_34_Diagram_1.jpeg)

(a) Counterexample for Theorem [3.10](#page-5-2) (Graph G) (b) Counterexample for Theorem [3.10](#page-5-2) (Graph H) Figure 5: Counterexample for Theorem [3.10](#page-5-2)

all (v, V) ∈ Meta(v). Thus, we conclude that tr(A<sup>n</sup> G(F ) ) = tr(A<sup>n</sup> H(F ) ) for all n ∈ N. By a basic result from linear algebra, this implies that Spec(G(F)) = Spec(H(F)). However, since hom(F, G(F)) ̸= hom(F, H(F)), we have proven that for any graph F that is not a cycle, there exists a pair of graphs G and H such that their spectra are identical, but hom(F, G) ̸= hom(F, H).

- We now prove that for any path F of length at least 2, there exist graphs G and H such that hom(F, G) = hom(F, H). A pair of counterexamples is provided in Figure [5.](#page-34-1) Initially, we observe that the two graphs are cospectral. Furthermore, for any path P of length k (k ≥ 2), hom(F, G) = 4 · 2 <sup>k</sup> + 2. For the graph H, let the number of k-walks starting from the vertex with degree 3 be denoted as ak. We then have the following recurrence relation:

$$a_k = a_{k-1} + 2 \cdot a_{k-2}, \quad a_0 = 1, \quad a_1 = 3.$$

From this relationship, we can deduce that:

$$\begin{aligned} a_{2k+1} &= 2^{2k+1} - 2^{2k+1} + \cdots + 2^2 - a_0 = 1 + 2 \cdot (1 + 4 + \cdots + 2^{2k}) = \frac{1}{3} (2^{2k+3} + 1), \\ a_{2k} &= 2^{2k+2} - \frac{1}{3} (2^{2k+3} + 1) = \frac{1}{3} (2^{2k+2} - 1). \end{aligned}$$

Therefore, the total number of homomorphisms from a path of length 2k + 1 to H is given by:

$$\begin{aligned} \text{hom}(P_{2k+2}, H) &= 4 \cdot a_{2k} + 2 \cdot a_{2k+1} \\ &= \frac{1}{3} (2 \cdot 2^{2k+3} - 4) + \frac{1}{3} (2 \cdot 2^{2k+3} + 2) + 3 \\ &= \frac{1}{3} (4 \cdot 2^{2k+3} - 2). \end{aligned}$$

Similarly, the total number of homomorphisms from a path of length 2k + 2 to H is:

$$\begin{aligned} \text{hom}(P_{2k+2}, H) &= 4 \cdot a_{2k+1} + 2 \cdot a_{2k+2} \\ &= \frac{4}{3} \cdot (2^{2k+3} + 1) + \frac{2}{3} \cdot (4 \cdot 2^{2k+5} - 2) \\ &= 3 \cdot 2^{2k+5}. \end{aligned}$$

Thus, for all k ≥ 3, we conclude that:

$$\text{hom}(P_k, G) \neq \text{hom}(P_k, H)$$
, for all  $k \geq 3$ .

Combining the previous three results, we have proven that homomorphic expressivity is {Cn|n ≥ 3} ∪ {P1, P2}..

# D EXPERIMENTAL DETAILS

In this section, we provide details on the experiments in Section [4.](#page-9-1) For dataset setup and training parameters, we follow [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2). We also use exactly the same model architecture for MPNN, subgraph GNN, and local 2-GNN as [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) did.

Model architecture of spectral invariant GNN. For spectral invariant GNN, we use the same feature initialization and final pooling layer as other models. The feature propogation in each layer is implemented to incorporate the eigenvalues and their projection matrices of the graph. Specifically, suppose {{(λ, Pλ(u, v))}} are all eigenvalues and eigenvectors of the input graph, h l (u) ∈ <sup>R</sup> d is the feature vector of node u in layer l. Then, the feature in next layer l + 1 is updated according to the following rule:

$$\begin{aligned} h^{(l+1)}(u) &= \text{ReLU}(\text{BN}^{(l)}(\text{MLP}_1^{(l)}((1 + \epsilon^{(l)}h^{(l)}(u) + f^{(l)}(u))), \\ f^{(l)}(u) &= \sum_{v \in \mathcal{V}} \text{ReLU}(h^{(l)}(v) + \sum_{\lambda} \text{MLP}_2^{(l)}(\lambda) P_{\lambda}(u, v)), \end{aligned} \quad (7)$$

where MLP1,<sup>2</sup> are two-layer feed-forward networks with batch normalization in the hidden layer.

Similar to [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), for graphs with edge features, we maintain a learnable edge embedding, g (l) (u, v), for each type of edges, and add them to the aggregation rule f (l) (u). The number of layers and hidden dimensions is set to match MPNN, such that all four models have roughly the same, and obey the 500K parameter budget in ZINC, as [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) did.

## E HIGHER ORDER SPECTRAL INVARIANT GNN

#### E.1 UPDATE RULE OF HIGHER-ORDER SPECTRAL INVARIANT GNN

A natural update rule for higher-order spectral invariant GNNs is as follows:

Definition E.1 (Higher-Order Spectral Invariant GNN). For any k ∈ <sup>N</sup>+, the k-order spectral invariant GNN maintains a color χ k-Spec <sup>G</sup> (u) for each vertex k-tuple u = (u1, . . . , uk) ∈ V k G. Initially, χ k-Spec,(0) <sup>G</sup> (u) = (P(u1, u2), . . . ,P(u1, uk), . . . ,P(uk−1, uk)). In each iteration t + 1, the color is updated as follows:

$$\chi_G^{k\text{-Spec},(t+1)}(\mathbf{u}) = \text{hash}(\chi_G^{k\text{-Spec},(t)}(\mathbf{u}), \{\{(\chi_G^{k\text{-Spec},(t)}(v, u_2, \dots, u_k), \mathcal{P}(u_1, v)) : v \in V_G\}\}, \dots, \\ \{\{(\chi_G^{k\text{-Spec},(t)}(u_1, u_2, \dots, u_{k-1}, v), \mathcal{P}(u_k, v)) : v \in V_G\}\}).$$

Denote the stable color of vertex tuple u ∈ V k <sup>G</sup> as χ k-Spec <sup>G</sup> (u). The graph representation is defined as χ k-Spec <sup>G</sup> (G) := {{χ k-Spec <sup>G</sup> (u) : u ∈ V k <sup>G</sup>}}.

#### E.2 HOMOMORPHISM EXPRESSIVITY OF HIGHER-ORDER SPECTRAL INVARIANT GNN

To describe the homomorphism expressivity of higher-order spectral invariant GNNs, we draw inspiration from the concept of "strong nested ear decomposition" from [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2). For the reader's convenience, we restate the relevant definitions here:

Definition E.2 (k-order Ear). A k-order ear is a graph G formed by the union of k paths P1, · · · , P<sup>k</sup> (possibly of zero length), along with an edge set Q, satisfying the following conditions:

- For each path P<sup>i</sup> , let its two endpoints be u<sup>i</sup> (outer endpoint) and v<sup>i</sup> (inner endpoint). All edges in Q are between inner endpoints, i.e., Q ⊂ {{v<sup>i</sup> , vj} : 1 ≤ i, j ≤ k, v<sup>i</sup> ̸= vj}.
- Any two distinct paths P<sup>i</sup> and P<sup>j</sup> intersect only at their inner endpoints (if v<sup>i</sup> = v<sup>j</sup> ).
- G is a connected graph.

The endpoints of the k-order ear are the outer endpoints u1, · · · , uk.

Definition E.3 (Nested Interval). Let G and H be two k-order ears with inner(G) = {v1, · · · , vk}, outer(G) = {u1, · · · , uk}, and outer(H) = {w1, · · · , wk}, where each {u<sup>i</sup> , vi} corresponds to the endpoints of a path P<sup>i</sup> ∈ path(G). We say H is nested on G if at least one endpoint w<sup>i</sup> of H (i ∈ [k]) lies on the path P<sup>i</sup> , and all other vertices of H are not part of G. The nested interval is defined as the union of the subpaths subpath<sup>P</sup><sup>i</sup> (w<sup>i</sup> , vi) for all i ∈ [k] such that w<sup>i</sup> lies on P<sup>i</sup> .

Definition E.4 (k-Order Strong Nested Ear Decomposition (NED)). A k-order strong NED P of a graph G is a partition of the edge set E<sup>G</sup> into a sequence of edge sets Q1, · · · , Qm, satisfying the following conditions:

- Each Q<sup>i</sup> is a k-order ear.
- Any two ears Q<sup>i</sup> and Q<sup>j</sup> with indices 1 ≤ i < j ≤ c do not intersect, where c is the number of connected components of G.

- For each Q<sup>j</sup> with index j > c, it is nested on some k-order ear Q<sup>i</sup> with index 1 ≤ i < j. Moreover, except for the endpoints of Q<sup>j</sup> on Q<sup>i</sup> , no other vertices in Q<sup>j</sup> belong to any previous ear Q<sup>k</sup> for 1 ≤ k < i.
- Denote by I(Q<sup>j</sup> ) ⊂ Q<sup>i</sup> the *nested interval* of Q<sup>j</sup> in Q<sup>i</sup> . For all Q<sup>j</sup> and Q<sup>k</sup> with c < j < k ≤ m, if Q<sup>j</sup> and Q<sup>k</sup> are nested on the same ear, then I(Q<sup>j</sup> ) ⊂ I(Qk).

Definition E.5 (Parallel k-Order Strong NED). A graph F is said to have a parallel k-order strong nested ear decomposition (NED) if there exists a graph G such that F can be obtained from G by replacing each edge (u, v) ∈ E<sup>G</sup> with a parallel edge that has endpoints (u, v).

With the definition of parallel k-order strong NED, we now state the homomorphism expressivity of k-spectral invariant GNN as follows:

Theorem E.6. *The homomorphism expressivity of a* k*-spectral invariant GNN is characterized by the set of all graphs that possess a parallel* k*-order strong NED.*

#### E.3 PROOF OF THEOREM [E.6](#page-36-1)

The proof of Theorem [E.6](#page-36-1) follows a similar structure to the analysis of Theorem [3.3](#page-3-0) and Theorem 3.4 in [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2). Therefore, we provide only a brief sketch, emphasizing the key differences between the proof of Theorem [E.6](#page-36-1) and the previous analyses.

Lemma E.7. *For any given graphs* G *and* H*, we have* χ k−Spec <sup>G</sup> (G) = χ k−Spec <sup>H</sup> (H) *if and only if, for every graph* F *that has a parallel* k*-order strong NED,* hom(F, G) = hom(F, H)*.*

*Proof.* We first define a parallel tree decomposition, which is a variant of the standard tree decomposition. Given a graph G = (VG, EG), its tree decomposition is represented as a tree T <sup>r</sup> = (V<sup>T</sup> , E<sup>T</sup> , β<sup>T</sup> , γ<sup>T</sup> ). The label functions β<sup>T</sup> : V<sup>T</sup> → 2 <sup>V</sup><sup>G</sup> and γ<sup>T</sup> : V<sup>T</sup> → 2 <sup>P</sup><sup>G</sup> are defined, where P<sup>G</sup> denotes the set of paths in G. The tree T = (V<sup>T</sup> , E<sup>T</sup> , β<sup>T</sup> , γ<sup>T</sup> ) satisfies the following conditions:

- 1. Each tree node t ∈ V<sup>T</sup> is associated with a non-empty subset of vertices β<sup>T</sup> (t) ⊂ V<sup>G</sup> in G, referred to as a bag. Each node t ∈ V<sup>T</sup> is also associated with a set of paths γ<sup>T</sup> (t), called a sub-bag, which includes paths in G that begin and end with vertices in β<sup>T</sup> (t). We say that a tree node t contains a vertex u if u ∈ β<sup>T</sup> (t), and contains a path p if p ∈ γ<sup>T</sup> (t).
- 2. For each path (u1, u2, . . . , un) with u<sup>i</sup> ∈ V<sup>G</sup> for i ∈ [n], there exists a tree node t ∈ V<sup>T</sup> that contains the path, i.e., (u1, . . . , un) ∈ γ<sup>T</sup> (t).
- 3. For each vertex u ∈ VG, the set of tree nodes t that contain u, denoted by B<sup>T</sup> (u) = {t ∈ V<sup>T</sup> : u ∈ β<sup>T</sup> (t)}, forms a non-empty connected subtree of T.
- 4. The depth of T is even, i.e., maxt∈V<sup>T</sup> depthT<sup>r</sup> (t) is an even number.
- 5. |β<sup>T</sup> (t)| = k if depthT<sup>r</sup> (t) is even, and |β<sup>T</sup> (t)| = k + 1 if depthT<sup>r</sup> (t) is odd.
- 6. For all tree edges {s, t} ∈ E<sup>T</sup> , where depthT<sup>r</sup> (s) is even and depthT<sup>r</sup> (t) is odd, we have β<sup>T</sup> (s) ⊂ β<sup>T</sup> (t).

We refer to (G, T<sup>r</sup> ) as a parallel tree-decomposed graph and k as the width of G's parallel tree decomposition. The set of parallel tree-decomposed graphs with width at most k is denoted as S k−Spec .

Similar to the low-dimensional case, we define the unfolding tree of a k-spectral invariant graph neural network as follows. Given a graph G, a vertex k-tuple u = (u1, . . . , uk) ∈ V k <sup>G</sup>, and a non-negative integer D, the depth-2D spectral k-spectral invariant tree of G at u, denoted (F k−Spec,(D) <sup>G</sup> (u), T <sup>k</sup>−Spec,(D) <sup>G</sup> (u)), is a parallel tree-decomposed graph (F, T<sup>r</sup> ) ∈ S<sup>k</sup>−Spec constructed as follows:

- 1. *Initialization.* Initialize F = G[u], and T with a root node r such that β<sup>T</sup> (r) = {u1, . . . , uk}. Define a mapping π : V<sup>F</sup> → V<sup>G</sup> by setting π(u) = u. For all i, j ∈ [k] with i ̸= j and r ∈ [n], if there exists an r-length walk (v1, . . . , vr) with v<sup>1</sup> = u<sup>i</sup> and v<sup>r</sup> = u<sup>j</sup> , we add a path (w1, . . . , wr) with w<sup>1</sup> = u<sup>i</sup> and w<sup>r</sup> = u<sup>j</sup> to F, and include (w1, . . . , wr) in the sub-bag γ<sup>T</sup> (r). Moreover, we extend π by setting π(wi) = v<sup>i</sup> for all i ∈ [r].

2. *Iterate for* D *rounds.* For each leaf node t ∈ T

r

, execute the following for each j ∈ [n]:

- (a) If w /∈ {π(u1), . . . , π(uk)}, add a new vertex z to F and extend π by setting π(z) = w. Set β<sup>T</sup> (tw) = β<sup>T</sup> (t) ∪ {z}. Initialize γ<sup>T</sup> (tw) = γ<sup>T</sup> (t). For all i ∈ [k] and r ∈ [n], if there exists a path of length r, (v1, . . . , vr), where v<sup>1</sup> = w and v<sup>r</sup> = π(wi), we construct a corresponding path (w1, . . . , wr), with w<sup>1</sup> = z and w<sup>r</sup> = u<sup>i</sup> , and include (w1, . . . , wr) in the sub-bag γ<sup>T</sup> (tw).
- (b) If w = π(ur) for some r ∈ [k], set β<sup>T</sup> (tw) = β<sup>T</sup> (t) ∪ {ur} without modifying F.

For each tw, add a child node t ′ <sup>w</sup> to T r , designate t<sup>w</sup> as its parent, and update β<sup>T</sup> (t ′ <sup>w</sup>) based on the following cases:

- (a) If w /∈ {π(u1), . . . , π(uk)}, set β<sup>T</sup> (t ′
  - <sup>w</sup>) = {u1, . . . , uj−1, w, uj+1, . . . , uk}.
- (b) If w = π(ur) for some r ∈ [k], set β<sup>T</sup> (t ′
  - <sup>w</sup>) = {u1, . . . , uj−1, ur, uj+1, . . . , uk}.

Finally, set γ<sup>T</sup> (t ′ <sup>w</sup>) as the set of all paths in F that connect pairs of vertices in β<sup>T</sup> (t ′ <sup>w</sup>).

Following a similar analysis as in the low-dimensional setting, we can first prove that for any two graphs G and H, χ k−Spec <sup>G</sup> (G) = χ k−Spec <sup>H</sup> (H) if and only if treeCount<sup>k</sup>−Spec((F, T<sup>r</sup> ), G) = treeCount<sup>k</sup>−Spec((F, T<sup>r</sup> ), H) holds for all (F, T<sup>r</sup> ) ∈ S<sup>k</sup>−Spec. We define

$$\text{treeCount}^{\text{Spec}}((F, T^r), G) := \left| \left\{ \mathbf{u} \in V_G^k : \exists D \in \mathbb{N}_+ \text{ such that } \left( F_G^{k-\text{Spec}, (D)}(\mathbf{u}), T_G^{k-\text{Spec}, (D)}(\mathbf{u}) \right) \cong (F, T^r) \right\} \right|.$$

With similar arguments as in Theorem 3.4 in [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2), we can further prove that for any two graphs G and H, treeCount((F, T<sup>r</sup> ), G) = treeCount((F, T<sup>r</sup> ), H) holds for all treedecomposed graphs (F, T<sup>r</sup> ) if and only if hom(F, G) = hom(F, H) holds. We now prove that a graph F has a parallel tree decomposition with width at most k if and only if F admits a parallel k-order strong NED. We prove each direction separately. First, we use induction on the number of vertices in F to show that for any (F, T<sup>r</sup> ) ∈ S<sup>k</sup>−Spec with β<sup>T</sup> (r) = {u1, u2, . . . , uk}, there exists a graph F˜ with a strong NED such that {u1, . . . , uk} are the endpoints of the first ear. We can construct F by replacing edges in F˜ with parallel edges. For the converse direction, assume that F admits a parallel k-order strong NED. We aim to prove that there exists a parallel tree decomposition T <sup>r</sup> of F such that (F, T<sup>r</sup> ) ∈ S<sup>k</sup>−Spec. We proceed by induction on the number of vertices and prove a stronger statement. For any connected graph F, if F can be constructed by replacing edges in a graph F˜ with parallel edges, where F˜ has a k-order strong NED and the endpoints of the first ear are {u1, u2, . . . , uk}, then there exists a tree decomposition T <sup>r</sup> of F. This decomposition satisfies (F, T<sup>r</sup> ) ∈ S<sup>k</sup>−Spec, and β<sup>T</sup> (r) = {u1, u2, . . . , uk}. By combining the proofs for both directions, we conclude the proof of the lemma.

We then prove the maximality of homomorphism expressivity as follows.

Lemma E.8. *For any connected graph* F /∈ F<sup>k</sup>−Spec*, there exist graphs* G *and* H *such that* hom(F, G) ̸= hom(F, H) *and* χ k−Spec <sup>G</sup> (G) = χ k−Spec <sup>H</sup> (H)*.*

*Proof.* As in the low-dimensional case, we consider a pebble game between two players, the Spoiler and the Duplicator. The game involves a graph F and several pebbles. Initially, all pebbles are placed outside the graph. During the course of the game, some pebbles are placed on the vertices of F, which divides the edges E<sup>F</sup> into connected components. In each round, the Spoiler updates the position of the pebbles, while the Duplicator manages a subset of connected components, ensuring that the number of selected components is *odd*. There are three main types of operations:

- 1. *Adding a pebble* p: the Spoiler places a pebble p (which was previously outside the graph) on some vertex of F. If adding this pebble does not change the connected components, the Duplicator does nothing. Otherwise, some connected component P is divided into several components P = S <sup>i</sup>∈[m] P<sup>i</sup> for some m. the Duplicator updates his selection as follows: if P was selected, he removes P and adds a subset of {P1, . . . , Pm}, while ensuring that the total number of selected components remains odd.
- 2. *Removing a pebble* p: the Spoiler removes a pebble p from a vertex. If this action does not alter the connected components, the Duplicator again does nothing. Otherwise, several connected

components P1, . . . , P<sup>m</sup> merge into a single component P = S <sup>i</sup>∈[m] P<sup>i</sup> . the Duplicator updates his selection by removing all selected P<sup>i</sup> and optionally adding P, while ensuring the total number of selected components is odd.

- 3. *Swapping two pebbles* p *and* p ′ : the Spoiler swaps the positions of two pebbles, which does not affect the connected components, and therefore the Duplicator does nothing.

the Spoiler wins the game if, at any point, there exists a path p such that both of its endpoints are covered by pebbles and the connected component containing {p} is selected by the Duplicator. If the Spoiler cannot achieve this throughout the game, the Duplicator wins. In the case of the k-spectral invariant GNN, there are k + 1 pebbles, denoted p<sup>u</sup><sup>1</sup> , . . . , p<sup>u</sup><sup>k</sup> , pv. Initially, all pebbles are placed outside the graph. the Spoiler first sequentially adds the pebbles p<sup>u</sup><sup>1</sup> , . . . , p<sup>u</sup><sup>k</sup> (using operation 1). The game proceeds in a cyclical manner. In each round, Spoiler selects an r ∈ [k] and freely chooses one of the following two actions:

- For r = 1, 2, . . . , k, Spoiler removes pebble p<sup>u</sup><sup>r</sup> (operation 2), and then re-adds it (operation 1).
- For r = 1, 2, . . . , k, Spoiler adds pebble p<sup>w</sup> (operation 1) adjacent to p<sup>u</sup><sup>r</sup> , swaps p<sup>u</sup><sup>r</sup> with p<sup>w</sup> (operation 3), and then removes p<sup>w</sup> (operation 2).

For a given graph F, let G(F) and H(F) denote the Furer graph and the twisted F ¨ urer graph with ¨ respect to F. Using similar reasoning as in the low-dimensional case, we can show that if the Spoiler cannot win the pebble game on F, then χ k−Spec G(F ) (G(F)) = χ k−Spec H(F ) (H(F)). Furthermore, analogous to the analysis of Lemma [B.37,](#page-31-0) we can conclude that if the Spoiler wins the pebble game on F, then there exists a parallel tree decomposition T <sup>r</sup> of F such that (F, T<sup>r</sup> ) ∈ S<sup>k</sup>−Spec. Thus, for any connected graph F /∈ F<sup>k</sup>−Spec, there exist graphs G(F) and H(F) such that hom(F, G(F)) ̸= hom(F, H(F)) and χ k−Spec <sup>G</sup> (G(F)) = χ k−Spec <sup>H</sup> (H(F)). This completes the proof of the lemma.

Finally, the proof of Theorem [E.6](#page-36-1) is completed by combining the results from Lemma [E.7](#page-36-2) and Lemma [E.8.](#page-37-0)

# F PROOF FOR SYMMETRIC POWER

#### F.1 PROPERTIES OF LOCAL k−GNN

In this section, we review key properties of the local k-GNN as presented in previous works. We begin by formally introducing the update rule for the local k-GNN.

Definition F.1. Local k-GNN maintains a color χ L(k) <sup>G</sup> (u) for each vertex k-tuple u ∈ V k <sup>G</sup>. Initially, χ L(k),(0) <sup>G</sup> (u) = atpG(u), called the isomorphism type of vertex k-tuple u, where atpG(u) is the *atomic type* of u. Then, in each iteration t + 1,

$$\chi_G^{L(k),(t+1)}(\mathbf{u}) = \text{hash} \left( \chi_G^{L(k),(t)}(\mathbf{u}), \{ \chi_G^{L(k),(t)}(\mathbf{v}) : \mathbf{v} \in N_G^{(1)}(\mathbf{u}) \}, \dots, \{ \chi_G^{L(k),(t)}(\mathbf{v}) : \mathbf{v} \in N_G^{(k)}(\mathbf{u}) \} \right), \quad (8)$$

where N (j) <sup>G</sup> (u) = {(u1, · · · , uj−1, w, uj+1, · · · , uk) : w ∈ NG(u<sup>j</sup> )}. Denote the stable color as χ L(k) <sup>G</sup> (u). The representation of graph G is defined as χ L(k) <sup>G</sup> (G) := {{χ L(k) <sup>G</sup> (u) : u ∈ V k <sup>G</sup>}}.

where 
$$N_G^{(j)}(\mathbf{u}) = \{(u_1, \dots, u_{j-1}, w, u_{j+1}, \dots, u_k) : w \in N_G(u_j)\}$$
. Denote the stable color as  $\chi_G^{\text{L}(k)}(\mathbf{u})$ . The representation of graph  $G$  is defined as  $\chi_G^{\text{L}(k)}(G) := \{\{\chi_G^{\text{L}(k)}(\mathbf{u}) : \mathbf{u} \in V_G^k\}\}$ .

Definition F.2 (Canonical Tree Decomposition). Given a graph G = (VG, EG), a canonical tree decomposition of width k is a rooted tree T <sup>r</sup> = (V<sup>T</sup> , E<sup>T</sup> , β<sup>T</sup> ) satisfying the following conditions:

- 1. The depth of T is even, i.e., maxt∈V<sup>T</sup> depT<sup>r</sup> (t) is even;
- 2. Each tree node t ∈ V<sup>T</sup> is associated to a multiset of vertices β<sup>T</sup> (t) ⊂ VG, called a *bag*. Moreover, |β<sup>T</sup> (t)| = k if depT<sup>r</sup> (t) is *even* and |β<sup>T</sup> (t)| = k + 1 if depT<sup>r</sup> (t) is *odd*;
- 3. For all tree edges {s, t} ∈ E<sup>T</sup> where depT<sup>r</sup> (s) is even and depT<sup>r</sup> (t) is odd, β<sup>T</sup> (s) ⊂ β<sup>T</sup> (t) (where "⊂" denotes the multiset inclusion relation);
- 4. For each edge {u, v} ∈ VG, there exists at least one tree node t ∈ V<sup>T</sup> that contains the edge, i.e., {u, v} ⊂ β<sup>T</sup> (t);

5. For each vertex u ∈ VG, all tree nodes t whose bag contains u form a (non-empty) collection.

We further define set S L(k) as follows:

Definition F.3. (F, T<sup>r</sup> ) ∈ S<sup>L</sup>(k) iff (F, T<sup>r</sup> ) satisfies [F.2](#page-38-2) with width k, and any tree node t of odd depth has only one child. Moreover, all vertex of F is contained in at least one node of t.

Then, we can obtain the following theorem of the homomorphic expressivity of Local k-GNN.

Theorem F.4. *Any graph* G *and* H *have the same representation under Local* k−*GNN (i.e.,* χ L(k) <sup>G</sup> (G) = χ L(k) <sup>H</sup> (H)*) iff* hom(F, G) = hom(F, H) *for all* (F, T<sup>r</sup> ) ∈ S<sup>L</sup>(k) *.*

#### F.2 MAIN RESULT

Since 2k-local GNN is strictly weaker than 2k-WL, we aim to extend previous result by showing that 2k-local GNN can encode k-symmetric power of a graph. We state our main result as follows:

Theorem F.5. *The Local* 2k*-GNN defined in [Morris et al.](#page-12-12) [\(2020\)](#page-12-12); [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) can encode the symmetric* k*-th power. Specifically, for given graphs* G *and* H*, if* G *and* H *have the same representation under Local* 2k*-GNN, then* G{k} *and* H{k} *have the same representation under the spectral invariant GNN defined in Section [2.1.](#page-2-2)*

#### F.3 PROOF OF THEOREM [F.5](#page-39-2)

Definition F.6. Let µ<sup>1</sup> < µ<sup>2</sup> < · · · < µ<sup>m</sup> represent the distinct eigenvalues of the k-th order symmetric power matrix of a graph G. Let E<sup>i</sup> denote the eigenspace corresponding to µ<sup>i</sup> , and P S i the orthogonal projection matrix from R C k <sup>n</sup> onto E<sup>i</sup> . For u1, u2, . . . , u2<sup>k</sup> ∈ VG, if both {{u1, u2, . . . , uk}} and {{uk+1, uk+2, . . . , u2k}} are multisets of k distinct vertices, then we define

$$P_*^S(S_1, S_2) = (P_1^S(S_1, S_2), \dots, P_m^S(S_1, S_2)),$$

where S<sup>1</sup> = {{u1, u2, . . . , uk}} and S<sup>2</sup> = {{uk+1, uk+2, . . . , u2k}}. Otherwise, we define

$$P_*^S(\{\{u_1, u_2, \dots, u_k\}\}, \{\{u_{k+1}, u_{k+2}, \dots, u_{2k}\}\}) = \mathbf{0}.$$

We encode the spectral information of the symmetric power into the aggregation of local 2k-GNN, resulting in a variant of the local 2k-GNN, defined as follows:

Definition F.7. A local 2k-GNN with symmetric power maintains a color χ SL(2k) <sup>G</sup> (u) for each vertex 2k-tuple u ∈ V 2k <sup>G</sup> . Initially, the color is defined as

$$\chi_G^{\text{SL}(2k),(0)}(\mathbf{u}) = \left( P_*^S(\{u_1, \dots, u_k\}), \{u_{k+1}, \dots, u_{2k}\}), \text{atp}_G(\mathbf{u}) \right).$$

Then, at each iteration t + 1, the update rule is given by:

$$\chi_G^{\text{SL}(2k),(t+1)}(\mathbf{u}) = \text{hash}\left(\chi_G^{\text{SL}(2k),(t)}(\mathbf{u}), \{\{\chi_G^{\text{SL}(2k),(t)}(\mathbf{v}) : \mathbf{v} \in N_G^{(1)}(\mathbf{u})\}\}, \dots, \{\{\chi_G^{\text{L}(2k),(t)}(\mathbf{v}) : \mathbf{v} \in N_G^{(k)}(\mathbf{u})\}\}\right), \quad (9)$$

where N (j) <sup>G</sup> (u) = {(u1, · · · , uj−1, w, uj+1, · · · , uk) : w ∈ NG(u<sup>j</sup> )}.

The stable color is denoted as χ SL(k) <sup>G</sup> (u). The graph representation is then defined as

$$\chi_G^{\text{SL}(k)}(G) := \{ \chi_G^{\text{SL}(k)}(\mathbf{u}) : \mathbf{u} \in V_G^{2k} \}.$$

Next, we define the concept of a k-dimensional path as follows:

Definition F.8. For a graph G and vertices u1, . . . , u<sup>k</sup> ∈ VG, we define the neighboring multiset of {{u1, u2, . . . , uk}} as:

$$N_G(\{u_1, u_2, \dots, u_k\}) = \bigcup_{r=1}^k \{\{u_1, \dots, u_{r-1}, v, u_{r+1}, \dots, u_k\} \mid v \in N_G(u_r)\}.$$

A k-dimensional walk of length n is defined as a sequence (S1, S2, . . . , Sn), where each S1, . . . , S<sup>n</sup> is a multiset of k elements, and for all r ∈ [n − 1], S<sup>r</sup> ∈ NG(Sr+1). If the path further satisfies the condition that for all u ∈ S<sup>r</sup> with r ∈ {2, 3, . . . , n − 1} and v ∈ VG, u ∈ NG(v) implies v ∈ S<sup>i</sup> for some i ∈ {r − 1, r, r + 1}, then we denote (S1, . . . , Sn) as a k-dimensional path of length n.

We then define set S SL(k) base on the definition of set S L(k) .

Definition F.9. (F, T<sup>r</sup> ) ∈ SSL(2k) iff (F, T<sup>r</sup> ) satisfies definition [F.2](#page-38-2) with width 2k, and any tree node t of odd depth has only one child. Furthermore, for tree node t ∈ V<sup>T</sup> if depT<sup>r</sup> (t) is even, we further associate it with a set of k−dimensional path γ<sup>T</sup> (t), called sub-bag. Specifically, for node t ∈ V<sup>T</sup> , let β<sup>T</sup> (t) = {{u1, . . . , u2k}}, then γ<sup>T</sup> (t) contains k-dimensional path linking {{u1, . . . , uk}} and {{uk+1, . . . , u2k}}. Each vertex of F is contained in at least one node of T r , either in bags or sub-bags.

Lemma F.10. *Any graph* G *and* H *have the same representation under Local* 2k*-GNN with symmetric power if* hom(F, G) = hom(F, H) *for all* (F, T<sup>r</sup> ) ∈ SSL(2k) *.*

*Proof.* To prove the theorem, we first define unfolding tree of local 2k-GNN with symmetric power. Given a graph G, 2k-tuple u ∈ V 2k <sup>G</sup> and a non-negative integer D, the depth-D unfolding tree of graph G at tuple u, denoted as (F SL(D) <sup>G</sup> (u), T SL(D) <sup>G</sup> (u)) is constructed as follows:

- 1. *Initialization.* We assume multiset u = {{u1, u2, · · · , u2k}}. At the beginning, F = G[{{u1, u2, · · · , u2k}}], and T only has a root node r with β<sup>T</sup> (r) = {{u1, u2, · · · , u2k}}. Define a mapping π : V<sup>F</sup> → V<sup>G</sup> as π(ui) = u<sup>i</sup> , ∀i ∈ [2k]. For every k-dimensional walk {{u1, . . . , uk}} = S1, . . . , S<sup>n</sup> = {{uk+1, . . . , u2k}} with n ≤ |VG| k , we introduce a k-dimensional path {{u1, . . . , uk}} = S ′ 1 , S′ 2 , . . . , S′ <sup>n</sup> = {{uk+1, . . . , u2k}}, and we add (S ′ 1 , S′ 2 , . . . , S′ n ) to sub-bag γ<sup>T</sup> (r).
- 2. *Loop for* D *rounds.* For each leaf node t in T r , do the following procedure for all i ∈ [2k]: Let β<sup>T</sup> (t) = {{u1, . . . , u2k}}. For each w ∈ VG, add a fresh child node t<sup>w</sup> to T and designate t as its parent. Then, consider the following two cases:
  - (a) If w /∈ {{π(u1), . . . , π(u2k)}}, then add a fresh vertex z to F and extend π with π(z) = w. Define β<sup>T</sup> (tw) = β<sup>T</sup> (t) ∪ {{z}}. Then, add edges between z and β<sup>T</sup> (t), so that π is an isomorphism from F[β<sup>T</sup> (tw)] to G[π(β<sup>T</sup> (tw))].
  - (b) If w ∈ {{π(u1), . . . , π(u2k)}}, let w = π(ur). Then, we simply set β<sup>T</sup> (tw) = β<sup>T</sup> (t)∪{{ur}} without modifying graph F.

Next, add a fresh child node t ′ <sup>w</sup> in T r , designate t<sup>w</sup> as its parent, and set β<sup>T</sup> (t ′ <sup>w</sup>) and γ<sup>T</sup> (t ′ w) based on the following two cases:

- (a) If w /∈ {{π(u1), . . . , π(u2k)}}, then β<sup>T</sup> (t ′
  - <sup>w</sup>) = {{u1, . . . , ui−1, z, ui+1, . . . , u2k}}. For every k-dimensional walk linking π(S1) and π(Sn) of length n (n ≤ |VG| k ), we introduce k−dimensional path S<sup>1</sup> = S ′ 1 , S′ 2 , . . . , S′ <sup>n</sup> = Sn. If i < k, then S<sup>1</sup> = {{u1, . . . , ui−1, z, . . . , uk}}. If i = k, then S<sup>1</sup> = {{u1, . . . , ui−1, z}}, while if i > k, then S<sup>1</sup> = {{u1, . . . , uk}}. We denote S<sup>2</sup> = {{u1, . . . , ui−1, z, ui+1, . . . , u2k}} \ S1. We add (S ′ 1 , S′ 2 , . . . , S′ n ) into sub-bag γ<sup>T</sup> (t ′ <sup>w</sup>).
- (b) Conversely, if w ∈ {{π(u1), . . . , π(u2k)}}, we assume that w = π(ur). Then β<sup>T</sup> (t ′
  - <sup>w</sup>) = {{u1, . . . , ui−1, ur, ui+1, . . . , u2k}}. For every k-dimensional walk linking π(S1) and π(Sn) of length n (n ≤ |VG| k ), we introduce k−dimensional path S<sup>1</sup> = S ′ 1 , S′ 2 , . . . , S′ <sup>n</sup> = Sn. If i < k, then S<sup>1</sup> = {{u1, . . . , ui−1, ur, . . . , uk}}. If i = k, then S<sup>1</sup> = {{u1, . . . , ui−1, ur}}, while if i > k, then S<sup>1</sup> = {{u1, . . . , uk}}. We denote S<sup>2</sup> = {{u1, . . . , ui−1, ur, ui+1, . . . , u2k}} \ S1. We add (S ′ 1 , S′ 2 , . . . , S′ n ) into sub-bag γ<sup>T</sup> (t ′ <sup>w</sup>).

We can see from the construction of unfolding tree that for all k-tuple u ∈ V 2k <sup>G</sup> and D > 0, (F SL(D) <sup>G</sup> (u), T SL(D) <sup>G</sup> (u)) ∈ SSL(2k) . Given (F, T<sup>r</sup> ),(F , ˜ T˜<sup>r</sup> ) ∈ SSL(2k) , we define a pair of mapping (ρ, τ ) as an isomorphism from (F, T<sup>r</sup> ) to (F , ˜ T˜<sup>r</sup> ), denoted by (F, T) ∼= (F , ˜ T˜<sup>r</sup> ), if the following hold:

- 1. ρ is an isomorphism from F to F˜.
- 2. τ is an isomorphism from T r to T˜<sup>r</sup> (ignoring β and γ).
- 3. For any t ∈ VT<sup>r</sup> , ρ(βT<sup>r</sup> (t)) = βT˜<sup>r</sup> (τ (t)), and ρ(γT<sup>r</sup> (t)) = γT˜<sup>r</sup> (τ (t)).

With similar analysis as Theorem [B.8](#page-19-1) we obtain that for any k-tuple u ∈ V k <sup>G</sup> and v ∈ V k H, χ SL(2k)(D) <sup>G</sup> (u) = χ SL(2k)(D) <sup>H</sup> (v) if there exists an isomorphism (ρ, τ ) from (F SL(D) <sup>G</sup> (u), T SL(D) <sup>G</sup> (u))

$$\text{treeCount}^{\text{SL}(2k)}((F, T^r), G) := \left| \left\{ \mathbf{u} \in V_G^{2k} : \exists D \in \mathbb{N}_+ \text{ s.t. } \left( F_G^{\text{SL}(D)}(\mathbf{u}), T_G^{\text{SL}(D)}(\mathbf{u}) \right) \cong (F, T^r) \right\} \right|.$$

Therefore, we can obtain that if treeCountSL(2k) ((F, T<sup>r</sup> ), G) = treeCountSL(2k) ((F, T<sup>r</sup> ), H) for all (F, T<sup>r</sup> ) ∈ SSL(2k) , then χ SL(2k) <sup>G</sup> (G) = χ SL(2k) <sup>H</sup> (H). Additionally, with similar analysis as Theorem [B.20](#page-24-0) and Theorem [B.14,](#page-21-0) we can obtain that treeCountSL(2k) ((F, T<sup>r</sup> ), G) = treeCountSL(2k) ((F, T<sup>r</sup> ), H) for all (F, T<sup>r</sup> ) ∈ SSL(2k) if and only if hom((F, T<sup>r</sup> ), G) = hom((F, T<sup>r</sup> ), H) for all (F, T<sup>r</sup> ) ∈ SSL(2k) . Therefore, we can obtain that if hom((F, T<sup>r</sup> ), G) = hom((F, T<sup>r</sup> ), H) for all (F, T<sup>r</sup> ) ∈ SSL(2k) , then χ SL(2k) <sup>G</sup> (G) = χ SL(2k) <sup>H</sup> (H). Thus, we finish the proof of the lemma.

Lemma F.11. *For all* k ≥ 1*, we have* S <sup>L</sup>(2k) = S SL(2k) *.*

*Proof.* We can directly see that S <sup>L</sup>(2k) ⊂ SSL(2k) , so it is sufficed to prove that for all (F, T<sup>r</sup> ) ∈ S SL(2k) , there exists an alternative tree decomposition T˜<sup>r</sup> such that (F, T˜<sup>r</sup> ) ∈ SSL(2k) . We will prove that for (F, T<sup>r</sup> ) ∈ SSL(2k) , if maxt∈V<sup>T</sup> <sup>r</sup> |γ<sup>T</sup> (t)| ≥ <sup>1</sup>, then we can construct <sup>T</sup>˜<sup>r</sup> such that maxt∈VT˜<sup>r</sup> |γT˜(t)| <sup>&</sup>lt; maxt∈V<sup>T</sup> <sup>r</sup> |γ<sup>T</sup> (t)|. For (F, T<sup>r</sup> ) ∈ SSL(2k) , let t = arg maxt˜∈V<sup>T</sup> <sup>r</sup> γ<sup>T</sup> (t˜)  and suppose k-dimensional path (S1, S2, . . . , Sn) ∈ γ<sup>T</sup> (t). We apply the following modification to T r to construct T˜<sup>r</sup> :

- 1. We construct tree node <sup>t</sup>1, t2, . . . , tn−<sup>1</sup> and <sup>t</sup>ˆ1,tˆ2, . . . ,tˆn−<sup>1</sup> such that <sup>β</sup>T˜(tr) = <sup>S</sup>r+1 ∪S<sup>r</sup> ∪S<sup>n</sup> and <sup>β</sup>T˜(tˆr) = <sup>S</sup>r+1 ∪ <sup>S</sup><sup>n</sup> for all <sup>r</sup> ∈ [<sup>n</sup> − 1].
- 2. We add tˆ<sup>r</sup> as the child node of t<sup>r</sup> for all r ∈ [n − 1] and add t<sup>r</sup> as the child node of tˆr−<sup>1</sup> for all r ∈ {2, 3, . . . , n − 1}. Eventually, we add t<sup>1</sup> as the child node of t.
- 3. We delete k-dimensional path from γ<sup>T</sup> (t) and keep the bags of all t ∈ V<sup>T</sup> and sub-bags of vertices in V<sup>T</sup> \ {t} unchanged. Namely, we assume that γT˜(t) = γ<sup>T</sup> (t) \ {(S1, . . . , Sn)}. Moreover, β<sup>T</sup> (t) = βT˜(t) for all t ∈ V<sup>T</sup> and γ<sup>T</sup> (t) = γT˜(t) for all V<sup>T</sup> \ {t}.

With the procedure above we can obtain (F, T˜<sup>r</sup> ) ∈ SSL(2k) such that maxt∈T˜<sup>r</sup> |γT˜(t)| < maxt∈T<sup>r</sup> |γ<sup>T</sup> (t)|. If we recursively apply this procedure to modify T˜<sup>r</sup> , we can eventually obtain Tˆr such that maxt∈Tˆ<sup>r</sup> |γT<sup>ˆ</sup>(t)| = 0. Therefore, (F, <sup>T</sup>ˆ<sup>r</sup> ) ∈ S<sup>L</sup>(2k) . Eventually, we have proven that for all (F, T<sup>r</sup> ) ∈ SSL(2k) , there exists an alternative decomposition T˜<sup>r</sup> such that (F, T˜<sup>r</sup> ) ∈ S<sup>L</sup>(2k) . Thus, for all k ≥ 1, S <sup>L</sup>(2k) = S SL(2k) .

Finally, we can finish the proof of Theorem [F.5.](#page-39-2)

Theorem F.12. *The Local* 2k*-GNN defined in [Morris et al.](#page-12-12) [\(2020\)](#page-12-12); [Zhang et al.](#page-13-2) [\(2024a\)](#page-13-2) can encode the symmetric* k*-th power. Specifically, for given graphs* G *and* H*, if* G *and* H *have the same representation under Local* 2k*-GNN, then* G{k} *and* H{k} *have the same representation under the spectral invariant GNN defined in Section [2.1.](#page-2-2)*

*Proof.* According to Lemma [F.11,](#page-41-0) the homomorphism expressivity of the vanilla Local 2k-GNN is equivalent to that of the Local 2k-GNN with symmetric power. Hence, the expressive power of the Local 2k-GNN is the same as that of the Local 2k-GNN with symmetric power. If there exist graphs G and H such that χ L(k) <sup>G</sup> (G) = χ L(k) <sup>H</sup> (H), then it must follow that χ SL(k) <sup>G</sup> (G) = χ SL(k) <sup>H</sup> (H). Therefore, we also have χ SL(k),(0) <sup>G</sup> (G) = χ SL(k),(0) <sup>H</sup> (H), meaning that the symmetric k-th powers of G and H are cospectral.Moreover, it is straightforward that if graphs G and H have the same representation under a Local 2k-GNN with symmetric power, then G{k} and H{k} also have the same representation under the spectral invariant GNN. Thus, the proof of the theorem is complete.