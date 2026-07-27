# Homomorphism Expressivity Of Spectral In- Variant Graph Neural Networks

Jingchu Gai 1 Yiheng Du 1 Bohang Zhang1∗ Haggai Maron 2,3 **Liwei Wang** 1 1Peking University 2Technion 3NVIDIA Research gaijingchu@stu.pku.edu.cn, zhangbohang@pku.edu.cn, duyiheng@stu.pku.edu.cn hmaron@nvidia.com, wanglw@pku.edu.cn

## Abstract

Graph spectra are an important class of structural features on graphs that have shown promising results in enhancing Graph Neural Networks (GNNs). Despite their widespread practical use, the theoretical understanding of the power of spectral invariants - particularly their contribution to GNNs - remains incomplete. In this paper, we address this fundamental question through the lens of homomorphism expressivity, providing a comprehensive and quantitative analysis of the expressive power of spectral invariants. Specifically, we prove that spectral invariant GNNs can homomorphism-count exactly a class of specific tree-like graphs which we refer to as *parallel trees*. We highlight the significance of this result in various contexts, including establishing a quantitative expressiveness hierarchy across different architectural variants, offering insights into the impact of GNN depth, and understanding the subgraph counting capabilities of spectral invariant GNNs. In particular, our results significantly extend Arvind et al. (2024) and settle their open questions. Finally, we generalize our analysis to higher-order GNNs and answer an open question raised by Zhang et al. (2024b).

## 1 Introduction

The graph spectrum, defined as the eigenvalues of a graph matrix, is an important class of graph invariants. It encapsulates rich graph structural information including the graph connectivity, bipartiteness, node clustering patterns, diameter, and more (Brouwer & Haemers, 2011). Besides eigenvalues, generalized spectral information may also include projection matrices, which further encodes node relations such as distances and random walk properties, enabling the definition of more fine-grained graph invariants (Furer ¨ , 2010). These spectral invariants possesses strong expressive power. For example, a well-known conjecture raised by Van Dam & Haemers (2003); Haemers & Spence (2004) claimed that almost all graphs can be uniquely determined by their spectra up to isomorphism. The rare exceptions, known as cospectral graphs, tend to be highly similar in their structure and continue to be an active area of research in graph theory (Lorenzen, 2022). In the machine learning community, spectral invariants have recently gained increasing popularity in designing Graph Neural Networks (GNNs) (Bruna et al., 2013; Defferrard et al., 2016; Lim et al., 2023; Huang et al., 2024; Feldman et al., 2023; Zhang et al., 2024b; Black et al., 2024), owing to several reasons. From a practical perspective, graph spectra have been shown to be closely related to certain practical applications such as molecular property prediction (Bonchev, 2018). Moreover, a recent line of works (Xu et al., 2019; Morris et al., 2019; Li et al., 2020; Chen et al., 2020; Zhang et al., 2023b) has pointed out that the expressive power of classic message-passing GNNs (MPNNs) are inherently limited, and cannot encode important graph structure like connectivity or distance. Incorporating spectral invariants into the design of MPNNs can naturally alleviate the limitations. Therefore, from both theoretical and practical perspectives, it is beneficial to give a systematic understanding of the power of spectral invariants and their corresponding GNNs. The earliest study in this area may be traced back to Furer ¨ (2010), who first linked the power of several spectral invariants to the classic Weisfeiler-Lehman test (Weisfeiler & Lehman, 1968) by proving that these invariants are upper bounded by 2-FWL. More recently, Rattan & Seppelt (2023) further revealed a strict
∗Project lead.

1 expressivity gap between Furer's spectral invariants and 2-FWL. ¨ Zhang et al. (2024b) and Arvind et al. (2024) analyzed *refinement-based* spectral invariants, which offer insights into the power of real GNN architectures. Yet, all of these works study expressiveness through the lens of Weisfeiler- Lehman tests, which has inherent limitations. So far, there remains a lack of *comprehensive* understanding of the *practical* power of spectral invariants and their corresponding GNN architectures. Current work. In this paper, we investigate the aforementioned questions via a novel perspective called *graph homomorphism*. Specifically, Zhang et al. (2024a) recently proposed homomorphism expressivity as a quantitative framework to better understand the expressive power of various GNN architectures. As homomorphism expressivity is a fine-grained and practical measure, it naturally addresses several limitations of the WL test. However, extending this framework to other architectures, such as spectral invariant GNNs, poses significant challenges. In fact, whether homomorphism expressivity exists for a given architecture remains an open research direction (see Zhang et al. (2024a)). In our context, this problem becomes even challenging since homomorphism and spectral invariants correspond to two orthogonal branches in graph theory. Here, we provide affirmative answers to all these questions by formally proving that the homomorphism expressivity for spectral invariant GNNs exists and can be elegantly characterized as a special class of *parallel trees*
(Theorem 3.3). This offers deep insights into a series of previous studies, extending their results and answering several open questions. We summarize our results below:
- **Separation power of spectral invariants/GNNs**. We offer a new proof that projection-based spectral invariants and corresponding GNNs are strictly bounded by 2-FWL (Corollary 3.4). Moreover, we establish a *quantitative hierarchy* among raw spectra information, projection, refinement-based spectral invariant, and various combinatorial variants of WL tests (see Figure 4). This (i) recovers and extends results in Rattan & Seppelt (2023), and (ii) provides clear insights into the hierarchy established in Zhang et al. (2024b).

- **The power of refinement**. We offer a systematic understanding of the role of refinement in spectral invariant GNNs. We show increasing the number of iterations always leads to a strict improvement in expressive power (Corollary 3.11), thus settling a key open question raised in Arvind et al. (2024). Moreover, our counterexamples establish a tight lower bound on the number of iterations required to achieve maximal expressivity, which is in the same order of graph size. This advances a line of research regarding iteration numbers in WL tests (Furer ¨ , 2001; Kiefer & Schweitzer, 2016; Lichter et al., 2019).

- **Substructure counting power of spectral invariants/GNNs**. On the practical side, we precisely characterize the power of spectral invariants/GNNs in counting certain subgraphs as well as the required iterations. For example, they can count all cycles within 7 vertices, while using 1 iteration already suffices to count all cycles within 6 vertices (Corollary 3.15).

Empirically, a set of experiments on both synthetic and real-world tasks validate our theoretical results, showing that the homomorphism expressivity of spectral invariant GNNs well reflects their performance in down-stream tasks.

## 2 Preliminaries

Notations. We use { } and *{{ }}* to denote sets and multisets, respectively. The cardinality of a given (multi)set S is denoted as |S|. In this paper, we consider finite, undirected, simple graphs with no self-loops or repeated edges, and without loss of generality we only consider connected graphs. Let G = (VG, EG) be a graph with vertex set VG and edge set EG, where each edge in EG is a set {u, v} ⊂ VG of cardinality two. The *neighbors* of vertex u is denoted as NG(u) := {v ∈ VG|{u, v} ∈ EG}. A *walk* of length k is a sequence of vertices u0, · · · , uk ∈ VG such that {ui−1, ui} ∈ EG for all i ∈ [k]. It is further called a *path* if ui ̸= uj for all *i < j*, and it is called a cycle if u0, · · · , uk−1 is a path and u0 = uk. The shortest path distance between two nodes *u, v* ∈ VG, denoted as disG(*u, v*), is the minimum length of walk from u to v. A graph F = (VF , EF ) is a *subgraph* of G if VF ⊂ VG and EF ⊂ EG. We use Pn (resp. Cn) to denote a graph corresponding to a path (resp. cycle) of n vertices. A graph is called a tree if it is connected and contains no cycle as a subgraph. We denote by T
rthe rooted tree T with root r. The depth of a rooted tree T
ris defined as dep(T
r) = maxu∈VT disT (*r, u*), and the depth of T is defined as dep(T) = minr∈VT dep(T
r).

## 2.1 Spectral Invariant Gnns

Let G be a graph of n vertices where VG = [n], and denote by A ∈ {0, 1}
n×n the adjacency matrix of G. The *spectrum* of G is defined as the multiset of all eigenvalues of A. In addition to eigenvalues, eigenspaces also provide important spectral information. Formally, the eigenspace associated with some eigenvalue λ can be characterized by its projection matrix Pλ. It follows that there exist a unique set of orthogonal projection matrices {Pλ}λ∈Λ, where Λ is the set of all distinct eigenvalues of A, such that A =Pλ∈ΛλPλ, and the following conditions hold: Pλ Pλ = I,
PλPλ′ = 0 for λ ̸= λ
′, and APλ = PλA for all λ ∈ Λ. Combining the projection matrices with the associated eigenvalues naturally define an invariant between node pairs, which we denote by P:

$${\mathrm{for~}}u\in V_{G},d\in N_{+},$$

P(*u, v*) := {{(λ, Pλ(*u, v*))|λ ∈ Λ}} for *u, v* ∈ VG.

Then, one can define the so-called "spectral invariant" of a graph as follows. Consider the following color refinement process by treating P(u, v) as the edge feature between vertices u and v:

$$\chi_{G}^{\mathsf{Spec},(d+1)}(u)=\mathsf{hash}\left(\chi_{G}^{\mathsf{Spec},(d)}(u),\P(u,v),{\mathcal{P}}(u,v))|v\in V_{G}\right)\quad\mathrm{for}\quad\chi_{G}^{\mathsf{Spec},(d)}(u)=\mathsf{hash}\left(\chi_{G}^{\mathsf{Spec},(d)}(u),\P(u,v),{\mathcal{P}}(u,v)\right).$$
G (v),P(*u, v*))|v ∈ VG}}for u ∈ VG, d ∈ N+,
where all colors χ Spec,(0)
G (u) (u ∈ VG) are constant in initialization, and hash is a perfect hash function. For each iteration d, the mapping χ Spec,(d)
G induces an equivalence relation over vertex set VG, and the relation gets *refined* with the increase of d. Therefore, with a sufficiently large number of iterations d ≤ |VG|, the relations get *stable*. The spectral invariant χ Spec,(∞)
G (G) is then defined to be the multiset of stable node colors. We can similarly define χ Spec,(d)
G (G) to be the multiset of node colors after d iterations (Arvind et al., 2024). We remark that χ Spec,(1)
G (G) is exactly the Furer's ¨
(weak) spectral invariant proposed in Furer ¨ (2010). Owing to the relation between GNNs and color refinement algorithms, one can easily transform the above refinement process into a GNN architecture by replacing hash function with a continuous, non-linear, parameterized function, while maintaining the same expressive power (Xu et al., 2019; Morris et al., 2019). We call the resulting architecture Spectral Invariant GNNs (see Zhang et al. (2024b) for concrete implementations of spectral invariant GNN layer). Without ambiguity, we may also refer to χ Spec,(d)
G (G) as the graph representation computed by a d-layer spectral invariant GNN.

## 2.2 Homomorphism Expressivity

Given two graphs F and G, a homomorphism from F to G is a mapping f : VF → VG that preserves edge relations, i.e., {f(u), f(v)} ∈ EG for all {u, v} ∈ EF . We denote by Hom(*F, G*)
the set of all homomorphisms from F to G and define hom(*F, G*) = |Hom(*F, G*)|, which counts the number of homomorphisms. If f is further surjective on both vertices and edges of G, we call G a homomorphic image of F. A mapping f : VF → VG is called an isomorphism if f is a bijection and both f and its inverse f
−1are homomorphisms. We denote by sub(F, G) the number of subgraphs of G that is isomorphic to F. In Zhang et al. (2024a), the authors introduced the concept the homomorphism expressivity to quantify the expressive power of a color refinement algorithm (or GNN). It is formally defined as follows: Definition 2.1. Let M be a color refinement algorithm (or GNN) that outputs a graph invariant χM
G (G) given graph G. The homomorphism expressivity of M, denoted by FM, is a family of connected graphs1satisfying the following conditions:
a) For any two graphs *G, H*, χM
G (G) = χM
H (H) iff hom(*F, G*) = hom(*F, H*) for all F ∈ FM;
b) FM is maximal, i.e., for any connected graph F /∈ FM, there exists a pair of graphs *G, H*
such that χM
G (G) = χM
H (H) and hom(*F, G*) ̸= hom(*F, H*).

By characterizing the set FM for different GNN models M, one can quantitatively understand the expressivity gap between two models by simply computing their set inclusion relation and set difference. Zhang et al. (2024a) examines several representative GNNs under this framework, including the standard MPNNs and Folklore GNNs (Maron et al., 2019; Azizian & Lelarge, 2021), and recent architectures such as Subgraph GNN (Bevilacqua et al., 2022; Qian et al., 2022; Cotta et al., 2021) and Local GNN (Morris et al., 2020; Zhang et al., 2023a). However, one implicit challenge not reflected in Definition 2.1(a) is that the set FM may not even exist for a general GNN M. Proving the existence corresponds to an involved research topic known as homomorphism distinguishing closedness (Roberson, 2022; Seppelt, 2024; Neuen, 2023), which is highly non-trivial. In the next section, we will give affirmative results showing that the homomorphism expressivity of spectral invariant GNNs does exist and give an elegant description of the graph family.

## 3 Homomorphism Expressivity Of Spectral Invariant Gnns

In this section, we investigate the homomorphism expressivity of spectral invariants and the corresponding GNNs. We will provide a complete characterization of the set F
Spec,(d)for arbitrary model depth d ∈ N *∪ {∞}*. This allows us to analyze spectral invariants in a novel perspective, significantly extending prior research and resolving previously unanswered questions.

## 3.1 Main Results

Our idea is motivated by the previous finding that the homomorphism expressivity of MPNNs is exactly the family of all trees (Zhang et al., 2024a). Note that in the definition of spectral invariant GNN, if one replaces P(*u, v*) by the standard adjacency Auv, the resulting architecture is just an MPNN. Such a relationship perhaps implies that the homomorphism expressivity of spectral invariant GNNs also comprises "tree-like" graphs. We will show this is indeed true. To present our results, let us define a special class of graphs, referred to as *parallel trees*:
Definition 3.1 (**Parallel Edge**). A graph G is called a *parallel edge* if there exist two different vertices *u, v* ∈ VG such that the edge set EG can be partitioned into a sequence of simple paths P1*, . . . , P*m, where all paths share endpoints (u, v). We refer to (*u, v*) as the endpoints of G.

Definition 3.2 (**Parallel Tree**). A graph F is called a *parallel tree* if there exists a tree T such that F
can be obtained from T by replacing each edge {u, v} ∈ ET with a parallel edge that has endpoints
{*u, v*}. We refer to T as the *parallel tree skeleton* of graph F. Given a parallel tree F, define the parallel tree depth of F as the minimum depth of any parallel tree skeleton of F. We give an illustration of parallel edge and parallel tree in Figure 1. With the above definitions, we are ready to state our main theorem:
Theorem 3.3. For any d ∈ N*, the homomorphism expressivity of spectral invariant GNNs with* d iterations exists and can be characterized as follows:
F
Spec,(d) = {F | F *has parallel tree depth at most* d}.

Specifically, the following properties hold:
- Given any graphs G and H, χ Spec,(d)
G (G) = χ Spec,(d)
H (H) if and only if, for all connected graphs F *with parallel tree depth at most* d, hom(*F, G*) = hom(*F, H*).

- F
Spec,(d)is maximal; that is, for any connected graph F /∈ FSpec,(d), there exist graphs G and H *such that* χ Spec,(d)
G (G) = χ Spec,(d)
H (H) and hom(*F, G*) ̸= hom(*F, H*).

We will present a concise proof sketch of Theorem 3.3 in Section 3.3. Next, in Section 3.2, we will interpret this result in the context of GNNs and discuss its significance, including how it extends previous findings and addresses open problems identified in earlier studies.

## 3.2 Implications

Our theory has a wide range of applications, which will be separately discussed in detail below.

## 3.2.1 Comparison With 2-Fwl

Firstly, we compare the expressive power of spectral invariant GNNs with the expressive power of the standard Weisfeiler-Lehman (WL) test. It immediately follows that the expressive power of spectral invariant GNNs strictly lies between the expressive power of 1-WL and 2-FWL test. Corollary 3.4. The expressive power of spectral invariant GNNs is strictly stronger than 1-WL and strictly weaker than 2*-FWL.*
Proof. According to Zhang et al. (2024a), the homomorphism expressivity of 2-FWL encompasses the set of all graphs with treewidth at most 2. A
classical result in graph theory states that any subgraph of any series-parallel graph has treewidth at most 2 (Diestel, 2017). Since any parallel tree is clearly a subgraph of some series-parallel graph, its treewidth is at most 2. It follows that the homomorphism expressivity of parallel trees is contained within that of the 2-FWL. To show the gap, we give a counterexample graph in Figure 2. This implies that the expressive power of spectral invariant GNNs is strictly weaker than that of the 2-FWL. The proof for the case of 1-WL is similar and we omit it for clarity.

1 2 3 4 5 6

## 3.2.2 Hierarchy

Theorem 3.3 not only provides insights into the relationship between the expressive power of spectral invariant GNNs and 2-FWL, but also allows for a comparison with a wide range of graph invariants and the corresponding GNNs. Specifically, similar to the analysis in Corollary 3.4, for any GNN models A and B such that their homomorphism expressivity exists, if F
A ⊊
 F
B, then A is strictly weaker than B in expressive power. We now use this property to establish a comprehensive hierarchy by linking spectral invariant GNNs to other fundamental graph invariants and GNNs. Corollary 3.5. Spectral invariant GNN with 1 *iteration is strictly weaker than subgraph GNN (also* referred to as (1, 1)-WL in Rattan & Seppelt (2023)). Proof. According to Zhang et al. (2024a), the homomorphism expressivity of subgraph GNNs contains all graphs that become a forest upon the deletion of a specific vertex. On the other hand, Theorem 3.3 states that the homomorphism expressivity of spectral invariant GNNs with one iteration contains all parallel trees of depth 1. Since any parallel tree of depth 1 becomes a forest when deleting the root vertex, we have proved that F
Spec,(1) is a subset of that of subgraph GNNs. Finally, one can easily construct a counterexample graph to prove the strict separation. Remark 3.6. Our result recovers and strengthens the main result in Rattan & Seppelt (2023), which only studied spectral invariants with 1 iteration (Furer's weak spectral invariant). We will next show ¨ this result actually does not hold in case of more than 1 iterations. Corollary 3.7. Spectral invariant GNNs with 2 *iterations are incomparable to subgraph GNNs.* We provide a counterexample in Figure 3. Nevertheless, we can still bound the expressive power of spectral invariant GNNs with multiple iterations to that of Local 2-GNN, as stated in the following:
Corollary 3.8. For any d ∈ N+∪{∞}, spectral invariant GNNs with d iterations are strictly weaker than Local 2-GNN (Morris et al., 2020; Zhang et al., *2024a).* Proof. According to Zhang et al. (2024a), the homomorphism expressivity of Local 2-GNNs contains all graphs that admit a strong nested ear decomposition. Since any parallel edge can be partitioned into ears with the same endpoints, one can easily construct a nested ear decomposition for any parallel tree. This shows F
Spec,(d)is a subset of that of Local 2-GNN. The expressivity gap can be seen using the same counterexample graph in Figure 2. Remark 3.9. Corollaries 3.7 and 3.8 significantly extend the findings of Arvind et al. (2024, Theorem 17) and provide additional insights into Zhang et al. (2024b, Theorem 4.3). The power of projection. We next conduct a fine-grained analysis by separating eigenvalues and projections to better understand their individual contributions to enhancing the expressive power of GNN models. We first prove the following theorem:
Theorem 3.10. *The homomorphism expressivity of graph spectra is the set of all cycles* Cn (n ≥ 3) plus paths P1 and P2, i.e., {Cn|n ≥ 3} ∪ {P1, P2}.

The proof of Theorem 3.10 is provided in Appendix C, which has the same structure as that of Theorem 3.3. Previously, Van Dam & Haemers(2003); Dell et al. (2018) have proved that the spectra of two graphs G and H are identical if and only if for every cycle F, hom(*F, G*) = hom(*F, H*). We extend their result by further proving the maximal property (Definition 2.1(b)), which only adds two trivial graphs P1 and P2 to the homomorphism expressivity. From this result, one can easily see that using eigenvalues alone can already improve the expressive power of an MPNN since the homomorphism expressivity of MPNN contains only trees (but not cycles).

To understand the role of projection, one can compare the set {Cn|n ≥ 3*}∪ {*P1, P2} with F
Spec,(1)
(the homomorphism expressivity of Furer's spectral invariant). Clearly, the set of all parallel trees of ¨
depth 1 is strictly larger than {Cn|n ≥ 3}∪ {P1, P2}, confirming that adding projection information significantly enhances the expressive power beyond graph spectra.

The power of refinement. We finally investigate the power of iterations d (or number of GNN
layers) in enhancing the model's expressive power. We have the following result:
Corollary 3.11. For any d ∈ N*, spectral invariant GNNs with* d + 1 *iterations are strictly more* powerful than spectral invariant GNNs with d iterations.

Proof. For any k ∈ N, we can construct a counterexample formed by replacing each edge in the path graph P2k+2 with a parallel edge. We illustrate the construction in Figure 3(b). One can easily see that the resulting graph is in F
Spec,(k+1) but not F
Spec,(k).

Remark 3.12. Corollary 3.11 addresses the key open question posed in Arvind et al. (2024), who conjectured that spectral invariant GNNs converge within *constant* iterations. Specifically, the authors questioned whether, for d ≥ 4, spectral invariant GNNs with d + 1 iterations are as powerful as those with d iterations. We disproved this conjecture by providing a family of example graphs that cannot be distinguished in d iterations but can be distinguished in d + 1 iterations. Our counterexamples further leads to the following result:
Corollary 3.13. For any d ∈ N+, There exist two graphs with O(d) *vertices such that spectral* invariant GNNs require at least d *iterations to distinguish between them.* Corollary 3.13 establishes a tight bound on the number of layers needed for spectral invariant GNNs to reach maximal expressivity, showing that it scales with the order of graph size. This advances an important research topic that aims to study the relation between expressiveness and iteration number of color refinement algorithms (Furer ¨ , 2001; Kiefer & Schweitzer, 2016; Lichter et al., 2019). To summarize all the above results, we illustrate the hierarchy established for spectral invariant GNNs and other mainstream GNNs in Figure 4.

## 3.2.3 Subgraph Count

In fact, our results can go beyond the WL framework and reveal the expressive power of spectral invariant GNNs in a more practical perspective. As an example, we will show below how Theorem 3.3 can be used to understand the subgraph counting capabilities of spectral invariant GNNs. Figure 4: Hierarchy of spectral invariant GNN (abbreviated as Spectral IGN) and other mainstream GNNs. Each arrow points to the strictly stronger architecture. Given any graph F, we say a GNN model M can subgraph-count substructure F if for any graphs G
and H, the condition χM
G (G) = χM
H (H) implies sub(*F, G*) = sub(*F, H*). Denote by Spasm(F) the set of all homomorphic images of F. Previous results have proved that, if the homomorphism expressivity FM exists for model M, then M can subgraph-count F if and only if Spasm(F) ⊂ FM
(Seppelt, 2023; Zhang et al., 2024a). This allows us to precisely analyze which substructure can be subgraph-counted by spectral invariant GNNs. Corollary 3.14. Spectral invariant GNN can count cycles and paths with up to 7 *vertices.* Proof. For cycles or paths with at most 7 vertices, one can check by enumeration that their homomorphic images are all parallel trees. For cycles or paths with at least 8 vertices, the 4-clique is a valid homomorphic image but is not a parallel tree. We can further strengthen the above results by studying the number of iterations needed to count substructures. We have the following results: Corollary 3.15. *The following holds:*
1. Spectral invariant GNNs can subgraph-count all cycles up to 7 vertices within 2 *iterations.* 2. The above upper bound is tight: spectral invariant GNNs with only 1 iteration (i.e., Furer's ¨
weak spectral invariant) cannot subgraph-count 7*-cycle.*
3. Spectral invariant GNNs with 1 iteration suffice to subgraph-count all cycles up to 6 *vertices.*
Remark 3.16. The subgraph counting power of spectral invariant has long been studied in the literature. Cvetkovic et al. (1997) proved that the graph angles (which can be determined by projection) can subgraph-count all cycles of length no more than 5. In comparison, our results significantly extend their findings, which even match the cycle counting power of 2-FWL (Arvind et al., 2020). Moreover, we show that Furer's weak spectral invariant can already count ¨ 6-cycles, thus extending the work of Furer ¨ (2017).

## 3.3 Proof Sketch

In this section, we provide a proof sketch of Theorem 3.3, with the complete proof presented in the Appendix. We begin by demonstrating that the information encoded by spectral invariants is closely related to encoding *walk information* in the aggregation process of GNNs. This corresponds to the following lemma (proved in Appendix B.2, see also Arvind et al. (2024)):
Lemma 3.17. (**Equivalence of encoding walk and encoding spectral information**) Let G = (VG, EG) be a graph, with its adjacency matrix denoted by A. For vertices x, y ∈ VG*, define* ω kG(*x, y*) = Akx,y for all k ∈ {0, 1, 2, . . . , |VG|}, which represents the number of k-walks from vertex x to vertex y*. Define the tuple* ω
∗G(*x, y*) = (ω 0G(x, y), ω1G(x, y)*, . . . , ω*n−1 G (x, y))*, where* n = |VG|*. Define the walk-encoding GNN with the following update rule:*
χ Walk,(d+1)
G (x) = hash(χ Walk,(d)
G (x), {{(ω
∗
G(x, y), χ Walk,(d)
G (y)) | y ∈ VG}}).

The walk-encoding GNN outputs a representation χ Walk,(d)
G (G) = {{χ Walk,(d)
G (u)|u ∈ VG}}. For any graphs G, H*, we have* χ Walk,(d)
G (G) = χ Walk,(d)
H (H) *if and only if* χ Spec,(d)
G (G) = χ Spec,(d)
H (H).

Our next step aims to prove that for graphs G and H, χ Walk,(d)
G (G) = χ Walk,(d)
H (H) iff, for all graphs F with parallel tree depth at most d, hom(*F, G*) = hom(*F, H*). This will yield the first property outlined in Theorem 3.3. The proof has a similar structure to that in Zhang et al. (2024a), which is based on the tools of tree-decomposed graphs and algebraic graph theory (see Theorems B.14 and B.20 and Lemma B.17). This part corresponds to Appendix B.3. Now, it remains to prove that the set F
Spec,(d)is maximal (the second property in Theorem 3.3).

To achieve this, we leverage the technique known as pebble game (Cai et al., 1992), which was originally used to construct counterexample graphs that cannot be distinguished by the k-FWL test. We extend the framework and define the pebble game for spectral invariant GNNs as follows: Definition 3.18. (**Pebble game for spectral invariant GNNs**) The pebble game is conducted on two graphs G = (VG, EG) and H = (VH, EH). Without loss of generality, we assume VG = VH.

Initially, each graph is equipped with two distinct pebbles denoted as u and v, which initially lie outside the graphs. The game involves two players: the *spoiler* and the *duplicator*. The game process is described as follows:
- *Initialization:* The spoiler first selects a non-empty subset V
Sfrom either VG or VH, and the duplicator responds with a subset V
D from the other graph, ensuring that |V
D| = |V
S|. Then, the spoiler places the pebble u on some vertex in V
D, and the duplicator places the corresponding pebble u on some vertex in V
S. Similarly, the spoiler and duplicator repeat the process to place two pebbles v. After the initialization, all pebbles will lie on the two graphs.

- *Main Process:* The game iteratively repeats the following steps, where in each iteration the spoiler may choose freely between the following two actions:
1. Action 1 (moving pebble v). The spoiler first selects a non-empty subset V
Sfrom either VG
or VH, and the duplicator responds with a subset V
D from the other graph, ensuring that |V
D| = |V
S|. The spoiler then moves pebble v to some vertex in V
D, and the duplicator moves the corresponding pebble v to some vertex in V
S.

2. Action 2 (moving pebble u). This action is similar to the above one except that both players move pebble u instead of pebble v.

- *Termination:* The spoiler wins if, after a certain number of rounds, ω
⋆
G(u, v) for graph G differs from ω
⋆
H(*u, v*) for graph H. Conversely, the duplicator wins if the spoiler is unable to win after any number of rounds.

With the above definition, we can now prove the equivalence between the outcome of a pebble game and the ability to distinguish non-isomorphic graphs using spectral invariant GNNs:
Lemma 3.19. (**Equivalence of pebble game and spectral invariant GNNs***) Given graphs* G
and H and the number of steps d ∈ N, the spoiler cannot win the pebble game in d *steps iff* χ Spec,(d+1)
G (G) = χ Spec,(d+1)
H (H).

We give a proof in Appendix B.4. Next, to identify counterexamples G and H for any F /∈ FSpec,(d)
such that χ Spec,(d)
G (G) = χ Spec,(d)
H (H) and hom(*F, G*) ̸= hom(*F, H*), we draw inspiration from a special class of graphs called Furer graphs ( ¨ Furer ¨ , 2001), which is a principled approach to constructing pairs of non-isomorphic but structurally similar graphs. If graphs G and H are the Furer ¨ graph and twisted Furer graph constructed from the same base graph ¨ F, we show that the pebble game can be significantly simplified. Importantly, the simplified pebble game will be played on the base graph F instead of the complex Furer graphs, making the subsequent analysis much easier. ¨ Due to space constraints, a detailed description of the simplified pebble game is provided in Appendix B.5. We then establish the following lemma, which relates the simplified pebble game to spectral invariant GNNs: Lemma 3.20. (**Equivalence of pebble game on Furer graphs and spectral invariant GNNs** ¨ *) Given* a base graph F, let G(F) and H(F) be the Furer graph and twisted F ¨ urer graph of ¨ F*, respectively.* Then, the spoiler cannot win the simplified pebble game on F in d *steps iff* χ Spec,(d+1)
G (G(F)) =
χ Spec,(d+1)
H (H(F)).

Note that for any connected graph F, hom(*F, G*(F)) ̸= hom(*F, H*(F)) (Roberson, 2022; Zhang et al., 2024a). Furthermore, we demonstrate that the spoiler has a winning strategy on F in d steps if and only if F is a parallel tree with parallel tree depth at most d + 1 (see Appendix B.6). By combining these results with Lemma 3.20, we establish the following lemma:
Lemma 3.21. For any F /∈ FSpec,(d)*, the spoiler cannot win the simplified pebble game on* F.

Consequently, χ Spec,(d)
G (G(F*)) =* χ Spec,(d)
H (H(F)).

This yields the second property in Theorem 3.3 and concludes the proof.

## 3.4 Extensions

So far, this paper mainly analyzes the standard spectral invariant GNNs, which refines *node features* based on projection information. In this subsection, we will show the flexibility of our proposed homomorphism expressivity framework, which can also be used to analyze other spectral-based GNN models such as higher-order spectral invariant GNNs.

## 3.4.1 Higher Order

Let us consider generalizing Section 2.1 to higher order spectral invariant GNNs. A natural update rule of higher order spectral invariant GNN can be defined as follows:
Definition 3.22 (**Higher-Order Spectral Invariant GNN**). For any k ∈ N+, the k-order spectral invariant GNN maintains a color χ k-Spec G (u) for each vertex k-tuple u = (u1*, . . . , u*k) ∈ V
kG. Initially, χ k-Spec,(0)
G (u) = (P(u1, u2), . . . ,P(u1, uk), . . . ,P(uk−1, uk)). In each iteration t + 1, the color is updated as follows:

$$\chi_{G}^{k\cdot\mathsf{S p e c},(t+1)}$$

G (u) = hash(χ k-Spec,(t)
G (u), {{(χ k-Spec,(t)
G (v, u2*, . . . , u*k),P(u1, v)) : v ∈ VG}}, · · · ,

) $\ast$ 1. 
{{(χ k-Spec,(t)
G (u1, u2, . . . , uk−1, v),P(uk, v)) : v ∈ VG}}).

Denote the stable color of vertex tuple u ∈ V
kG as χ k-Spec G (u). The graph representation is defined as χ k-Spec G (G) := {{χ k-Spec G (u) : u ∈ V
k G}}.

One can see that when k = 1, the above definition degenerates to the standard spectral invariant GNN defined in Section 2.1. To illustrate the homomorphism expressivity of higher-order spectral invariant GNNs, we extend the concept of strong nested ear decomposition (NED) introduced by Zhang et al. (2024a) and define the parallel strong NED. Our main result is stated below:
Theorem 3.23 (informal). A graph F is said to have a parallel k-order strong nested ear decomposition (NED) if there exists a graph G such that G admits a strong NED and F *can be obtained from* G by replacing each edge {u, v} ∈ EG with a parallel edge that has endpoints (u, v). Then, the homomorphism expressivity of k-order spectral invariant GNNs is the set of all graphs that admit a parallel k*-order strong NED.* Due to space constraints, we leave the formal definition of k-order strong NED and the technical proof of Theorem 3.23 to the Appendix.

## 3.4.2 Symmetric Power

To generalize spectrum and projection to higher order, another classic approach in the literature is to use the symmetric power of a graph (also called the *token graph*). Audenaert et al. (2005) first introduced the graph symmetric power to generalize eigenvalues into higher-order graph invariants. The formal definition of the symmetric k-th power is presented as follows:
Definition 3.24 (**Symmetric Power**). For any k ∈ N+ and graph G, the symmetric k-th power of G, denoted by G{k}, is a graph where its vertices are k-subsets of VG, and two subsets are adjacent if and only if their symmetric difference is an edge in G. Our homomorphism expressivity framework can be used to study the ability of mainstream GNNs to encode the symmetric power of graphs. Our main result is stated as follows:
Theorem 3.25. The Local 2k-GNN defined in Morris et al. (2020); Zhang et al. (2024a) can encode the symmetric k-th power. Specifically, for given graphs G and H, if G and H *have the same* representation under Local 2k-GNN, then G{k} and H{k} have the same representation under the spectral invariant GNN defined in Section *2.1.* Discussions with prior work. Regarding the expressive power of symmetric power, Alzaga et al. (2008); Barghi & Ponomarenko (2009) gave the first upper bound, showing that if 2k-FWL fails to distinguish between two non-isomorphic graphs, then their symmetric k-th powers are cospectral.

| Table 1: Experimental results on homomorphism counting, real-world tasks and substructure count. Task Homomorphism Count ZINC Substructure Count Model Subset Full MPNN .300 .261 .276 .233 .341 .138 ± .006 .030 ± .002 .358 .208 .188 .146 .261 .205 Spectral Invariant GNN .045 .046 .053 .048 .303 .103 ± .006 .028 ± .003 .072 .072 .089 .089 .060 .099 Subgraph GNN .011 .013 .010 .015 .260 .110 ± .007 .028 ± .002 .010 .020 .024 .046 .007 .027 Local 2-GNN .008 .006 .008 .008 .112 .069 ± .001 .024 ± .002 .008 .011 .017 .034 .007 .016   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

However, it remains unclear whether the conclusion extends to the more powerful projection information (beyond eigenvalues), or if the stated upper bound is tight. These open questions are further highlighted in Zhang et al. (2024b). Our result answers both questions by bounding the stronger refinement-based spectral invariant for the k-th symmetric power graphs to Local 2k-GNN, which is strictly weaker than 2k-FWL (Zhang et al., 2024a). This offers a deeper understanding of the capability of mainstream GNNs in encoding higher-order spectral information.

## 4 Experiment

In this section, we validate our theoretical findings through empirical experiments. We evaluate the performance of GNN models on both synthetic and real-world tasks. For the synthetic tasks, we assess the homomorphic counting power and subgraph counting power of the GNN models. These experiments serve to confirm our theoretical results, including Theorem 3.3 and Corollary 3.14. In addition, for the real-world task, we focus on molecular reaction prediction, specifically evaluating GNN performance on the ZINC dataset (Dwivedi et al., 2020). Our primary objective is not to achieve SOTA results but to validate our theoretical findings. We compare the performance of spectral invariant GNNs to both MPNNs and subgraph GNNs on the ZINC dataset. Details about model architectures are in Appendix D. Homomorphism Count We use the benchmark dataset from Zhao et al. (2022) to evaluate the homomorphism expressivity of four mainstream GNN models. The reported performance is measured by the normalized Mean Absolute Error (MAE) on the test set. The empirical results are presented in Table 1. We can see that concerning homomorphism: (i) MPNN is unable to encode any of the five substructures, and none of the five substructures is a tree; (ii) Spectral invariant GNN can only encode the 1st and 2nd substructures; (iii) Subgraph GNN can encode the 1st, 2nd, and 3rd substructures; and (iv) Local 2-GNN can encode the 1st, 2nd, 3rd, and 4th substructures. The empirical results basically align with our theoretical findings. Subgraph Count Cycle counting is a fundamental problem in chemical and biological tasks. Following the settings in Frasca et al. (2022); Zhang et al. (2023a); Huang et al. (2023), we evaluate the cycle counting power of four GNNs. The empirical results in Table 1 demonstrate that the spectral invariant GNN can accurately count 3-, 4-, 5-, and 6-cycles, indicating its strong performance in cycle counting tasks. This empirical result is also consistent with our theoretical predictions. Real-World Task We evaluate our GNN models on the ZINC-subset and ZINC-full dataset (Dwivedi et al., 2020). Following the standard configuration, all models are constrained to a 500K parameter budget. The results show that the spectral invariant GNN outperforms MPNN while demonstrating comparable performance to the subgraph GNN on the real-world task. These findings are consistent with our theoretical predictions.

## 5 Conclusion

In this work, we investigate the expressive power of spectral invariant graph neural networks (GNNs). By leveraging the framework of homomorphism expressivity, we give a precise characterization the homomorphism expressivity of these networks. We then establish a comprehensive hierarchy of spectral invariant GNNs relative to other mainstream GNNs based on their homomorphism expressivity. Additionally, we analyze the subgraph counting capabilities of spectral invariant GNNs, with a focus on their ability to count essential substructures. Our results are extended to higher-order contexts and address additional problems related to spectral structures using our homomorphism framework. We demonstrate the significance of our findings by showing how our results extend previous work and address open problems identified in the literature. Finally, we conduct experiments to validate our theoretical results.

## Acknowledgements

This work is supported by National Science and Technology Major Project (2022ZD0114902)and National Science Foundation of China (NSFC92470123, NSFC62276005).

## References

Alfredo Alzaga, Rodrigo Iglesias, and Ricardo Pignol. Spectra of symmetric powers of graphs and the weisfeiler-lehman refinements, 2008. URL https://arxiv.org/abs/0801.2322.

Vikraman Arvind, Frank Fuhlbruck, Johannes K ¨ obler, and Oleg Verbitsky. On weisfeiler-leman ¨
invariance: Subgraph counts and related graph properties. *Journal of Computer and System Sciences*, 113:42–59, 2020.

Vikraman Arvind, Frank Fuhlbruck, Johannes K ¨ obler, and Oleg Verbitsky. On a hierarchy of spec- ¨
tral invariants for graphs. In *41st International Symposium on Theoretical Aspects of Computer* Science (STACS 2024). Schloss Dagstuhl–Leibniz-Zentrum fur Informatik, 2024. ¨
Koenraad Audenaert, Chris Godsil, Gordon Royle, and Terry Rudolph. Symmetric squares of graphs, 2005. URL https://arxiv.org/abs/math/0507251.

Waiss Azizian and Marc Lelarge. Expressive power of invariant and equivariant graph neural networks. In *International Conference on Learning Representations*, 2021.

Muhammet Balcilar, Pierre Heroux, Benoit Gauzere, Pascal Vasseur, S ´ ebastien Adam, and Paul ´
Honeine. Breaking the limits of message passing graph neural networks. In International Conference on Machine Learning, pp. 599–608. PMLR, 2021.

Amir Rahnamai Barghi and Ilya Ponomarenko. Non-isomorphic graphs with cospectral symmetric powers. *the electronic journal of combinatorics*, pp. R120–R120, 2009.

Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai, Gopinath Balamurugan, Michael M Bronstein, and Haggai Maron. Equivariant subgraph aggregation networks. In *International Conference on Learning Representations*, 2022.

Mitchell Black, Zhengchao Wan, Gal Mishne, Amir Nayyeri, and Yusu Wang. Comparing graph transformers via positional encodings. *arXiv preprint arXiv:2402.14202*, 2024.

Danail Bonchev. *Chemical graph theory: introduction and fundamentals*. Routledge, 2018. Andries E Brouwer and Willem H Haemers. *Spectra of graphs*. Springer Science & Business Media, 2011.

Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. *arXiv preprint arXiv:1312.6203*, 2013.

Jin-Yi Cai, Martin Furer, and Neil Immerman. An optimal lower bound on the number of variables ¨
for graph identification. *Combinatorica*, 12(4):389–410, 1992.

Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count substructures? In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 10383–10395, 2020.

Leonardo Cotta, Christopher Morris, and Bruno Ribeiro. Reconstruction for powerful graph representations. In *Advances in Neural Information Processing Systems*, volume 34, pp. 1713–1726, 2021.

Radu Curticapean, Holger Dell, and Daniel Marx. Homomorphisms are a good basis for counting ´
small subgraphs. In *Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of* Computing, pp. 210–223, 2017.

Dragos Cvetkovic, Dragos M Cvetkovi ˇ c, Peter Rowlinson, and Slobodan Simic. ´ *Eigenspaces of* graphs. Cambridge University Press, 1997.

Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on ¨
graphs with fast localized spectral filtering. In *Advances in neural information processing systems*, volume 29, 2016.

Holger Dell, Martin Grohe, and Gaurav Rattan. Lovasz meets weisfeiler and leman. In ´ 45th International Colloquium on Automata, Languages, and Programming (ICALP 2018), volume 107, pp. 40. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018.

Reinhard Diestel. *Graph Theory*. Springer Publishing Company, Incorporated, 5th edition, 2017.

ISBN 3662536218.

Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson.

Benchmarking graph neural networks. *arXiv preprint arXiv:2003.00982*, 2020.

Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson.

Graph neural networks with learnable structural and positional representations. arXiv preprint arXiv:2110.07875, 2021.

Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. *Journal of Machine Learning Research*, 24(43):1–48, 2023.

Or Feldman, Amit Boyarski, Shai Feldman, Dani Kogan, Avi Mendelson, and Chaim Baskin. Weisfeiler and leman go infinite: Spectral and combinatorial pre-colorings. Transactions on Machine Learning Research, 2023. ISSN 2835-8856.

Fabrizio Frasca, Beatrice Bevilacqua, Michael M Bronstein, and Haggai Maron. Understanding and extending subgraph gnns by rethinking their symmetries. In Advances in Neural Information Processing Systems, 2022.

Martin Furer. Weisfeiler-lehman refinement requires at least a linear number of iterations. In ¨ International Colloquium on Automata, Languages, and Programming, pp. 322–333. Springer, 2001.

Martin Furer. On the power of combinatorial and spectral invariants. ¨ Linear algebra and its applications, 432(9):2373–2380, 2010.

Martin Furer. On the combinatorial power of the weisfeiler-lehman algorithm. In ¨ *International* Conference on Algorithms and Complexity, pp. 260–271. Springer, 2017.

Floris Geerts and Juan L Reutter. Expressiveness and approximation properties of graph neural networks. In *International Conference on Learning Representations*, 2022.

Willem H Haemers and Edward Spence. Enumeration of cospectral graphs. European Journal of Combinatorics, 25(2):199–211, 2004.

Yinan Huang, Xingang Peng, Jianzhu Ma, and Muhan Zhang. Boosting the cycle counting power of graph neural networks with i$ˆ2$-GNNs. In The Eleventh International Conference on Learning Representations, 2023.

Yinan Huang, William Lu, Joshua Robinson, Yu Yang, Muhan Zhang, Stefanie Jegelka, and Pan Li. On the stability of expressive positional encodings for graphs. In *The Twelfth International* Conference on Learning Representations, 2024.

Charilaos Kanatsoulis and Alejandro Ribeiro. Counting graph substructures with graph neural networks. In *The Twelfth International Conference on Learning Representations*.

Sandra Kiefer and Pascal Schweitzer. Upper bounds on the quantifier depth for graph differentiation in first order logic. In Proceedings of the 31st Annual ACM/IEEE Symposium on Logic in Computer Science, pp. 287–296, 2016.

Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Letourneau, and Prudencio Tossou. Re- ´
thinking graph transformers with spectral attention. In Advances in Neural Information Processing Systems, volume 34, 2021.

Ron Levie, Federico Monti, Xavier Bresson, and Michael M Bronstein. Cayleynets: Graph convolutional neural networks with complex rational spectral filters. IEEE Transactions on Signal Processing, 67(1):97–109, 2018.

Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec. Distance encoding: design provably more powerful neural networks for graph representation learning. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 4465–4478, 2020.

Moritz Lichter, Ilia Ponomarenko, and Pascal Schweitzer. Walk refinement, walk logic, and the iteration number of the weisfeiler-leman algorithm. In 2019 34th Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), pp. 1–13. IEEE, 2019.

Derek Lim, Joshua David Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie Jegelka. Sign and basis invariant networks for spectral graph representation learning. In The Eleventh International Conference on Learning Representations, 2023.

Kate Lorenzen. Cospectral constructions for several graph matrices using cousin vertices. Special Matrices, 10(1):9–22, 2022.

Laszl ´ o Lov ´ asz. ´ *Large networks and graph limits*, volume 60. American Mathematical Soc., 2012. Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. In *Advances in neural information processing systems*, volume 32, pp. 2156–2167, 2019.

Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In *Proceedings of the AAAI conference on artificial intelligence*, volume 33, pp. 4602–4609, 2019.

Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: towards scalable higher-order graph embeddings. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 21824–21840, 2020.

Daniel Neuen. Homomorphism-distinguishing closedness for graphs of bounded tree-width. arXiv preprint arXiv:2304.07011, 2023.

Chendi Qian, Gaurav Rattan, Floris Geerts, Mathias Niepert, and Christopher Morris. Ordered subgraph aggregation networks. In *Advances in Neural Information Processing Systems*, 2022.

Ladislav Rampa´sek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Do- ˇ
minique Beaini. Recipe for a general, powerful, scalable graph transformer. *Advances in Neural* Information Processing Systems, 35:14501–14515, 2022.

Gaurav Rattan and Tim Seppelt. Weisfeiler-leman and graph spectra. In *Proceedings of the 2023* Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pp. 2268–2285. SIAM, 2023.

David E Roberson. Oddomorphisms and homomorphism indistinguishability over graphs of bounded degree. *arXiv preprint arXiv:2206.10321*, 2022.

Tim Seppelt. Logical equivalences, homomorphism indistinguishability, and forbidden minors.

arXiv preprint arXiv:2302.11290, 2023.

Tim Seppelt. Logical equivalences, homomorphism indistinguishability, and forbidden minors. Information and Computation, pp. 105224, 2024.

Edwin R Van Dam and Willem H Haemers. Which graphs are determined by their spectrum? *Linear* Algebra and its applications, 373:241–272, 2003.

Boris Weisfeiler and Andrei Lehman. The reduction of a graph to canonical form and the algebra which appears therein. *NTI, Series*, 2(9):12–16, 1968.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *International Conference on Learning Representations*, 2019.

Bohang Zhang, Guhao Feng, Yiheng Du, Di He, and Liwei Wang. A complete expressiveness hierarchy for subgraph GNNs via subgraph weisfeiler-lehman tests. In International Conference on Machine Learning, volume 202, pp. 41019–41077. PMLR, 2023a.

Bohang Zhang, Shengjie Luo, Di He, and Liwei Wang. Rethinking the expressive power of gnns via graph biconnectivity. In *International Conference on Learning Representations*, 2023b.

Bohang Zhang, Jingchu Gai, Yiheng Du, Qiwei Ye, Di He, and Liwei Wang. Beyond weisfeilerlehman: A quantitative framework for GNN expressiveness. In The Twelfth International Conference on Learning Representations, 2024a.

Bohang Zhang, Lingxiao Zhao, and Haggai Maron. On the expressive power of spectral invariant graph neural networks. *arXiv preprint arXiv:2406.04336*, 2024b.

Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any gnn with local structure awareness. In *International Conference on Learning Representations*, 2022.

# Appendix

## Table Of Contents

| Table of Contents A Additional Related Work   | 16                                                                        |     |    |
|-----------------------------------------------|---------------------------------------------------------------------------|-----|----|
| B                                             | Proof of Theorem 3.3                                                      | 17  |    |
| B.1                                           | Preparation: Parallel Tree and Unfolding Tree                             |     | 17 |
| B.2                                           | Step 1: Equivalence of Encoding Walk information and Spectral Information | . . | 18 |
| B.3                                           | Step 2: Finding the Homomorphic Expressivity                              | 20  |    |
| B.4                                           | Step 3: Finding Pebble Game for Spectral Invariant GNN                    |     | 26 |
| B.5                                           | Step 4: Introducing Furer graphs ¨                                        |     | 28 |
| B.6                                           | Step 5: Proving the Maximality of Homomorphism Expressivity               | 31  |    |
| C                                             | Proof of Theorem 3.10                                                     | 34  |    |
| D                                             | Experimental Details                                                      | 35  |    |
| E                                             | Higher Order Spectral Invariant GNN                                       | 36  |    |
| E.1                                           | Update Rule of Higher-Order Spectral Invariant GNN                        |     | 36 |
| E.2                                           | Homomorphism Expressivity of Higher-Order Spectral Invariant GNN          | 36  |    |
| E.3                                           | Proof of Theorem E.6                                                      | 37  |    |
| F                                             | Proof for Symmetric Power                                                 | 39  |    |
| F.1                                           | Properties of Local k−GNN                                                 | 39  |    |
| F.2                                           | Main Result                                                               | 40  |    |
| F.3                                           | Proof of Theorem F.5                                                      |     | 40 |

## A Additional Related Work

Spectral Based Graph Neural Network. Spectral invariants refer to eigenvalues, projection matrices, and other generalized spectral information. In recent studies, spectral invariants have gained significant attention in the fields of graph learning and graph theory (Furer ¨ , 2010; Van Dam & Haemers, 2003; Haemers & Spence, 2004). For instance, a well-known conjecture proposed by Van Dam & Haemers(2003); Haemers & Spence (2004) posits that almost all graphs can be uniquely determined by their spectra, up to isomorphism. Given the importance and widespread application of graph spectral information (Bonchev, 2018), the machine learning community has also focused on analyzing the ability of graph neural networks (GNNs) to encode spectral information and on designing GNN models that incorporate more spectral features. As a result, several recent works have concentrated on the spectral-based design of GNNs (Bruna et al., 2013; Defferrard et al., 2016; Lim et al., 2023; Huang et al., 2024; Feldman et al., 2023; Zhang et al., 2024b). Specifically, Dwivedi et al. (2023; 2021); Kreuzer et al. (2021); Rampa´sek et al. ˇ (2022) have designed spectral GNNs by encoding Laplacian eigenvectors as absolute positional encodings. A key drawback of using Laplacian eigenvectors is the ambiguity in choosing eigenvectors; thus, follow-up works have sought to design GNNs that are invariant to the choice of eigenvectors. Lim et al. (2023) introduced BasisNet, which achieves spectral invariance for the first time using projection matrices. Huang et al. (2024) further generalized BasisNet by proposing the Spectral Projection Encoding (SPE), which performs soft aggregation across different eigenspaces, as opposed to the hard separation implemented in BasisNet. In addition to the design of spectral-based GNNs, several recent works have also focused on analyzing the expressive power of spectral GNNs and comparing them with other mainstream GNN models. Balcilar et al. (2021) investigate the relationship between ChebNet (Defferrard et al., 2016) and the 1-WL test, demonstrating that for graphs with similar maximum eigenvalues, ChebNet is as expressive as 1-WL. Geerts & Reutter (2022) revisit this analysis and prove that CaleyNet (Levie et al., 2018) is bounded by the 2-WL test. Black et al. (2024) introduced several new WL algorithms based on absolute and relative positional encodings (PE). The authors further established a bunch of equivalence relationships among these algorithms. Notably, there exists a strong connection between the proposed "stack of power of matrices" PE and Spectral Invariant GNNs. We can prove that the proposed (I, L, · · · , L2n−1)-WL
(see Theorem 4.6 in Black et al. (2024)) is as expressive as spectral invariant GNNs with matrix L,
and similarly, (I, A, · · · , A2n−1)-WL is as expressive as spectral invariant GNNs with the ordinary adjacency matrix. Therefore, all results in our paper can be used to understand the power of these WL variants. Since Zhang et al. (2024b) has shown that the expressive power of RD-WL is bounded by Spectral Invariant GNNs, it follows that the proposed L
†-WL (see Theorem 4.6 in Black et al.

(2024)) is also bounded in expressive power by Spectral Invariant GNNs. This conclusion reproduces their key result (Theorem 4.4 in Black et al. (2024)). Homomorphism Count and Subgraph Count. Subgraph counting is a fundamental problem in chemical and biological tasks, as the ability to count subgraphs is strongly correlated with the performance of GNN in molecular prediction tasks. Kanatsoulis & Ribeiro studies subgraph counting power for a novel GNN framework, where classic message-passing GNNs are enhanced with random node features, and the GNN output is computed by taking the expectation over the introduced randomness. The paper demonstrates that such GNNs can learn to count various substructures, including cycles and cliques. These findings share similarities with our work, as both studies characterize the cycle-counting power of certain GNN models. Notably, the GNN framework proposed in Kanatsoulis & Ribeiro can count more complex substructures, such as 4-cliques and 8-cycles, which exceed the expressive power of 2-FWL. Moreover, based on the foundational theory of Lovasz ´ (2012); Curticapean et al. (2017), it follows that the subgraph counting power of a GNN can be inferred from its ability to count homomorphisms (Seppelt, 2023; Zhang et al., 2024a). Consequently, recent research has also focused on the homomorphism counting power of GNNs. Dell et al. (2018) demonstrates that two graphs have the same representation under the k-WL algorithm if and only if the number of homomorphisms to the two graphs from any substructure with bounded tree width k is equal. Additionally, Zhang et al. (2024a) introduce the concept of homomorphism expressivity as a quantitative framework for assessing the expressive power of GNNs. This paper specifically focuses on the subgraph counting power of spectral invariant GNNs. Related works in this area include Cvetkovic et al. (1997), which shows that the graph angles (which can be determined through projection) are capable of counting all cycles of length up to 5, and Lim et al. (2023), which demonstrates that GNNs can count cycles with up to 5 vertices. A detailed comparison of our results with these previous studies is provided in the main text.

## B Proof Of Theorem 3.3

B.1 PREPARATION: PARALLEL TREE AND UNFOLDING TREE B.1.1 ADDITIONAL EXPLANATION FOR PARALLEL TREE For the reader's convenience, we begin by restating the definition of the parallel tree, as introduced in the main paper. Definition B.1 (**Parallel Edge:**). We denote a graph G as a *parallel edge* if there exist vertices u, v ∈ VG such that the edge set EG can be partitioned into a sequence of simple paths P1*, . . . , P*m, where each path has endpoints (u, v). We refer to (*u, v*) as the endpoints of the parallel edge G.

Definition B.2 (**Parallel Tree:**). We define a graph F as a *parallel tree* if there exists a tree T such that we can obtain a graph isomorphic to F by replacing each edge (*u, v*) ∈ ET with a parallel edge having endpoints (*u, v*). We refer to T as the *parallel tree skeleton* of the graph F. Additionally, we denote the minimum depth of any parallel tree skeleton of F as the *parallel tree depth* of F. We further define parallel tree decomposition for any parallel tree as follows:
Definition B.3 (**Parallel tree decomposition**). For a parallel tree F = (VF , EF ), its parallel tree decomposition involves constructing a rooted tree T
r = (VTr , ETr ) along with mapping functions βTr and γTr that satisfy the following conditions:
1. The label function for nodes, βTr : VTr → VF , maps each node in T
rto a unique vertex in F.

2. Let EF denote the union of all paths in the graph F. The edge label function, γTr : ETr → 2 EF ,
satisfies the condition that for all (t1, t2) ∈ ETr , each P ∈ γTr (t1, t2) is a path connecting βTr (t1) and βTr (t2). Moreover, for each edge e ∈ EF , there exists a unique tuple (t1, t2, P), where (t1, t2) ∈ VT × VT and P ∈ γT (t1, t2), such that e lies on the path P.

We denote T
r = (VTr , ETr , βTr , γTr ) as the decomposition skeleton of graph F, and the ordered pair (*F, T*r) as a parallel-tree decomposed graph. Let S
pt denote the set of all parallel trees, and we use S
pt dto denote the set of all parallel trees whose parallel tree skeleton has depth at most d.

## B.1.2 Unfolding Tree Of Spectral Invariant Gnn

We now introduce a process of constructing a parallel tree from any vertex of a given graph. Definition B.4 (**Constructing an unfolding tree of spectral invariant GNN**). Given a graph G, vertex u ∈ V (G) and a non-negative integer d, the depth-d spectral GNN unfolding tree of graph G at vertex u, denoted as (F
(d)
G (u), T(d)
G (u)), is a parallel-tree decomposed graph constructed as follows: At the beginning, F = {u}, and T only has a root node r with βTr (r) = {u}. We can define a mapping π : VF → VG as π(u) = u. For each leaf node t in T
r, do the following procedure: Let βTr (t) = x. For each w ∈ VG, add a fresh node tw to T
rand designate t as its parent. Then, consider the following case:
1. If w ̸= π(x), add xw to F and extend π with π(xw) = w. We define βTr (tw) = xw. For every walk w = v1, v2*, . . . , v*n = π(x) with n ≤ |VG|, where v1 = π(x), vn = w, we introduce a path xv1
, xv2
, . . . , xvn linking xw and x to graph F, where xv1 = *x, x*vn = xw. We can also extend mapping π with π(xv1
) = v1, π(xv2
) = v2*, . . . , π*(xvn
) = vn. We define γTr (t, tw) to be the set of all path xv1, xv2*, . . . , x*vnconnecting x and xw introduced in this step.

2. If w = π(x), we define βTr (tw) = x. Similarly, for every walk w = v1, v2*, . . . , v*n = π(x)
with n ≤ |VG|, we introduce a loop xv1, xv2*, . . . , x*vnto graph F, where xv1 = x = xvn. We can also extend mapping π with π(xv1) = v1, π(xv2) = v2*, . . . , π*(xvn) = vn. We define γTr (*t, t*w) to be the set of all path xv1, xv2*, . . . , x*vnconnecting x and xw introduced in this step.

We terminate the process once T
r becomes a complete tree of depth d.

The following fact is straightforward from the construction of the unfolding tree:
Fact B.5. For any graph G, any vertex u ∈ VG, and any non-negative integer D, there is a homomorphism from F
(D)
G (u) to G.

With additional Explanation for parallel tree and construction of unfolding tree, we are now ready to prove Theorem 3.3 step by step. B.2 STEP 1: EQUIVALENCE OF ENCODING WALK INFORMATION AND SPECTRAL
INFORMATION
In this section, we aim to prove Lemma 3.17. The key idea is to use the Cayley-Hamilton theorem to demonstrate that the walk-encoding GNN, as defined in Lemma 3.17, is equivalent to the spectral invariant GNN. B.2.1 PROOF OF LEMMA 3.17 Lemma B.6. Let G = (VG, EG) be a graph, with its adjacency matrix denoted by A. For vertices x, y ∈ VG*, define* ω kG(*x, y*) = Akx,y for all k ∈ {0, 1, 2, . . . , |VG|}, which represents the number of k-walks from vertex x to vertex y*. Define the tuple* ω
∗
G(*x, y*) =
(ω 0G(x, y), ω1G(x, y)*, . . . , ω*n−1 G (x, y)), where n = |VG|. Define the walk-encoding GNN with the following update rule:
χ Walk,(d+1)
G (x) = hash(χ Walk,(d)
G (x), {{(ω
∗G(x, y), χ Walk,(d)
G (y)) | y ∈ VG}}).

The walk-encoding GNN outputs a graph invariant χ Walk,(d)
G (G) = {{χ Walk,(d)
G (u)|u ∈ VG}}.

For any graphs G and H*, we have* χ Walk,(d)
G (G) = χ Walk,(d)
H (H) *if and only if* χ Spec,(d)
G (G) =
χ Spec,(d)
H (H).

Proof. We begin by proving the following statement: If the spectra of graph G and graph H are identical (denoted as (λ1, λ2*, . . . , λ*m)), then for *x, u* ∈ VG and *y, v* ∈ VH, P(x, u) = P(*y, v*) if and only if ω
⋆G(*x, u*) = ω
⋆H(*y, v*).

1. First, we prove that if P(*x, u*) = P(*y, v*), then ω
⋆G(x, u) = ω
⋆H(*y, v*).

By the properties of diagonalizable matrices, for any k ∈ {1, 2*, . . . ,* |VG|}, we have:
ω k G(*x, u*) = λ k 1Pλ1(x, u) + λ k λ2P2(*x, u*) + *· · ·* + λ k mPλm(x, u).

Therefore, if Pλr(*x, u*) = Pλr(y, v), ∀r ∈ [m],
it follows that:

$$\omega_{G}^{k}(x,u)=\sum_{r=1}^{m}\lambda_{r}^{k}\mathbf{P}_{\lambda_{r}}(x,u)=\sum_{r=1}^{m}\lambda_{r}^{k}\mathbf{P}_{\lambda_{r}}(y,v)=\omega_{H}^{k}(y,v).$$
$\mathbf{A}$ 2. 
$\mathbf{a}=\mathbf{a}^{\dagger}\mathbf{a}$. 
Thus, we have proven the first direction of the statement.

2. Now, we prove that if ω
⋆G(x, u) = ω
⋆H(*y, v*), then P(x, u) = P(*y, v*).

Let AG and AH denote the adjacency matrices of graphs G and H, respectively. By the Cayley- Hamilton theorem, the minimal annihilating polynomial of matrix AG is given by:
f(λ) = (λ − λ1)(λ − λ2)*· · ·*(λ − λm).

For each r ∈ {1, 2*, . . . , m*}, the eigenspace corresponding to eigenvalue λr is Ker(λrI −AG).

Since:
R

n = Ker(λ1I − AG) ⊕ Ker(λ2I − AG) *⊕ · · · ⊕* Ker(λmI − AG),
for each r ∈ {1, 2*, . . . , m*}, the projection matrix onto the kernel space Ker(λrI − AG) is:

$$f_{r}(\mathbf{A}_{G})=\prod_{j\neq r}(\lambda_{j}I-\mathbf{A}_{G})=\mathbf{P}_{\lambda_{r}}.$$
.
Therefore, there exist coefficients c r 0*, . . . , c*rm−1such that:

Pλr(*x, u*) = c
r
0· ω
0
G(x, u) + c r 1· ω 1 G(x, u) + · · · + c H(y, v) + c r 1· ω 1 H(y, v) + · · · + c r

r
m−1· ω
m−1
G (*x, u*),
Pλr(*y, v*) = c
r
0· ω
0
m−1· ω
m−1
H (*y, v*).

Finally, we conclude that if ω
⋆G(*x, u*) = ω
⋆H(*y, v*), then P(*x, u*) = P(*y, v*) for all *x, u* ∈ VG
and *y, v* ∈ VH.

Armed with the statement proven above, we are now prepared to prove Lemma 3.17. We will prove the two directions of the lemma separately as follows:
1. First, we prove that if χ Spec G (G) = χ Spec H (H), then χ Walk G (G) = χ Walk H (H). To do so, it suffices to show that for all t ∈ N, if χ Spec,(t)
G (u) = χ Spec,(t)
H (v) for all (u, v) ∈ VG × VH, then χ Walk,(t)
G (u) = χ Walk,(t)
H (v).

We prove this by induction. Initially, the statement holds trivially for t = 0. We then assume the statement holds for t = d and aim to prove it for t = d + 1. If χ Spec,(d+1)
G (u) = χ Spec,(d+1)
H (v),
then the following conditions are satisfied:

$\chi_{G}^{\mathsf{Spec},(d)}(u)=\chi_{H}^{\mathsf{Spec},(d)}(v),$  $\{(\mathcal{P}(u,x),\chi_{G}^{\mathsf{Spec},(d)}(x))\mid x\in V_{G}\}=\{(\mathcal{P}(v,y),\chi_{H}^{\mathsf{Spec},(d)}(y))\mid y\in V_{H}\}\}.$
$$(d)\,(y)).$$
$$\left(2\right)$$

$$(1)$$
For any x ∈ VG and y ∈ VH, if (P(u, x), χ
Spec,(d)
G (x)) = (P(v, y), χ
Spec,(d)
H (y)), then by our
previous result and the induction hypothesis, we have:
$$(\omega_{G}^{\star}(u,x),\chi_{G}^{\mathrm{Walk},(d)}(x))=(\omega_{H}^{\star}(v,y),\chi_{H}^{\mathrm{Walk},(d)}(y)).$$
H (y)). (2)
By combining equation 1 and equation 2, we conclude:
χ Walk,(d)
G (u) = χ Walk,(d)
H (v),
{{(ω
⋆
G(u, x), χ Walk,(d)
G (x)) | x ∈ VG}} = {{(ω
⋆
H(v, y), χ Walk,(d)
H (y)) | y ∈ VH}}.

Thus, we conclude that χ Walk,(d+1)
G (u) = χ Walk,(d+1)
H (v). Therefore, we have proven that χ Spec G (G) = χ Spec H (H) implies χ Walk G (G) = χ Walk H (H).

2. Now, we prove the converse: if χ Walk G (G) = χ Walk H (H), then χ Spec G (G) = χ Spec H (H). Initially, χ Walk G (G) = χ Walk H (H) implies {{χ Walk,(1)
G (u) | u ∈ VG}} = {{χ Walk,(1)
H (v) | v ∈ VH}}. If χ Walk,(1)
G (u) = χ Walk,(1)
H (v), then ω
⋆
G(*u, u*) = ω
⋆
H(*v, v*). This leads to:
{{ω
⋆
G(*u, u*) | u ∈ VG}} = {{ω
⋆
H(*v, v*) | v ∈ VH}}.

Hence, we derive that for all k ∈ [n]:

$$\mathrm{tr}\left(\mathbf{A}_{G}^{k}\right)=\sum_{u\in V_{G}}\mathbf{A}_{G}^{k}(u,u)=\sum_{u\in V_{G}}\omega_{G}^{k}(u,u)=\sum_{v\in V_{H}}\omega_{H}^{k}(v,v)=\sum_{v\in V_{H}}\mathbf{A}_{H}^{k}(v,v)=\mathrm{tr}\left(\mathbf{A}_{H}^{k}\right).$$  By standard results from linear algebra, the spectra of graphs $G$ and $H$ must be identical.  
Similar to the first direction, we now prove that for all t ∈ N, if χ Walk,(t)
G (u) = χ Walk,(t)
H (v) for all (*u, v*) ∈ VG × VH, then χ Spec,(t)
G (u) = χ Spec,(t)
H (v).

We again proceed by induction. Initially, the statement holds trivially for t = 0. Assuming the statement holds for t = d, we aim to prove it for t = d + 1. If χ Walk,(d+1)
G (u) = χ Walk,(d+1)
H (v),
we have:

χ Walk,(d) G (u) = χ Walk,(d) H (v), {{(ω ⋆ G(u, x), χ Walk,(d) G (x)) | x ∈ VG}} = {{(ω ⋆ H(v, y), χ Walk,(d) H (y)) | y ∈ VH}}.
According to the statement proven earlier, for any x ∈ VG and y ∈ VH, ω
⋆G(u, x) = ω
⋆H(*v, y*)
implies that P(*u, x*) = P(*v, y*). Thus, we obtain:

$\square$
χ Spec,(d)
G (u) = χ Spec,(d)
H (v),
{{(P(u, x), χ Spec,(d)
G (x)) | x ∈ VG}} = {{(P(v, y), χ Spec,(d)
H (y)) | y ∈ VH}}.

Therefore, we conclude that χ Spec,(d+1)
G (u) = χ Spec,(d+1)
H (v). Finally, we have proven that χ Walk G (G) = χ Walk H (H) implies χ Spec G (G) = χ Spec H (H).

By combining both directions, we conclude that for any two graphs G and H, χ Walk G (G) = χ Walk H (H)
if and only if χ Spec G (G) = χ Spec H (H). Hence, the walk-encoding GNN is as expressive as the spectralinvariant GNN.

## B.3 Step 2: Finding The Homomorphic Expressivity

We first define the isomorphism between parallel-tree decomposed graphs.

Definition B.7. Given two parallel-tree decomposed graphs (*F, T*r) and (F , ˜ T˜r), a pair of mappings
(*ρ, τ* ) is called an isomorphism from (F, Tr) to (F , ˜ T˜r), denoted by (F, Tr) ∼= (F , ˜ T˜r), if the following hold:
1. ρ is an isomorphism from F to F˜, while τ is an isomorphism from T
rto T˜r(ignoring labels β and γ).

2. For any t ∈ VTr , ρ(βTr (t)) = βT˜r (τ (t)). Moreover, for any (t1, t2) ∈ ETr , ρ(γTr (t1, t2)) =
γTr (τ (t1, t2))
Theorem B.8. For any two graphs G, H, any vertices u ∈ VG, x ∈ VH,and any non-negative integer D, χ Walk,(D)
G (u) = χ Walk,(D)
H (x) iff there exists an isomorphism (ρ, τ ) *from* (F
(D)
G (u), T(D)
G (u))
to (F
(D)
H (x), T(D)
H (x)) such that ρ(u) = x.

Proof. The proof proceeds by induction on D. The base case is straightforward: for D = 0, the theorem holds trivially. Now assume the theorem holds for all D ≤ d, and we will prove it for D = d + 1. We first prove that χ Walk,(d+1)
G (u) = χ Walk,(d+1)
H (x) implies the existence of an isomorphism
(*ρ, τ* ) from (F
(d+1)
G (u), T(d+1)
G (u)) to (F
(d+1)
H (x), T(d+1)
H (x)) such that ρ(u) = x. Given that χ
(d+1)
G (u) = χ
(d+1)
H (x), it follows that:
{{ω
∗
G(u, v), χ Walk,(d)
G (v)}}v∈VG = {{ω
∗
H(x, y), χ Walk,(d)
H (y)}}y∈VH .

Let n = |VG| = |VH|, and denote VG = {v1, v2, . . . , vn}, VH = {y1, y2*, . . . , y*n} such that:
ω
∗
G(*u, v*i) = ω
∗
H(x, yi), χ Walk,(d)
G (vi) = χ Walk,(d)
H (yi) for all i ∈ [n].

By the definition of tree unfolding, we have:

$$F_{G}^{(d+1)}(u)=\left(\bigcup_{v_{i}}F_{G}^{(d)}(v_{i})\right)\cup F_{G}^{(1)}(u),\quad F_{H}^{(d+1)}(x)=\left(\bigcup_{y_{i}}F_{H}^{(d)}(y_{i})\right)\cup F_{H}^{(1)}(x),$$

where we use ∪ to represent graph union. By the inductive hypothesis, there exists an isomorphism (ρi, τi) from (F
(d)
G 
(vi), T(d)
G 
(vi)) to (F
(d)
H 
(yi), T(d)
H 
(yi)) such that ρi(vi) = yi. Additionally, since ω
∗G(*u, v*i) = ω
∗H(*x, y*i), F
(1)
G 
(u) is isomorphic to F
(1)
H 
(x). Therefore, by merging all ρi and τiinto ρ˜ and τ˜, and constructing an approximate mapping between tree nodes at depth no more than 1 in T
(d+1)
G (u) and T
(d+1)
H (x), it follows that (˜ρ, τ˜) is a well-defined isomorphism from
(F
(d+1)
G (u), T(d+1)
G (u)) to (F
(d+1)
H (x), T(d+1)
H (x)), satisfying ρ˜(u) = x.

Next, we prove that if there exists an isomorphism (*ρ, τ* ) between the parallel-tree decomposed graphs (F
(d+1)
G (u), T(d+1)
G (u)) and (F
(d+1)
H (x), T(d+1)
H (x)) such that ρ(u) = x, then