# Ah-Ugc: Adaptive And Heterogeneous-Universal Graph Coarsening

Anonymous Author(s)
Affiliation Address email

## Abstract

1 **Graph Coarsening (GC)** is a prominent graph reduction technique that com2 presses large graphs to enable efficient learning and inference. However, existing 3 GC methods generate only one coarsened graph per run and must recompute from 4 scratch for each new coarsening ratio, resulting in unnecessary overhead. Moreover, 5 most prior approaches are tailored to *homogeneous* graphs and fail to accommodate 6 the semantic constraints of *heterogeneous* graphs, which comprise multiple node 7 and edge types. To overcome these limitations, we introduce a novel framework 8 that combines Locality-Sensitive Hashing (LSH) with Consistent Hashing to en9 able *adaptive graph coarsening*. Leveraging hashing techniques, our method is 10 inherently fast and scalable. For heterogeneous graphs, we propose a *type-isolated* 11 *coarsening* strategy that ensures semantic consistency by restricting merges to 12 nodes of the same type. Our approach is the first unified framework to support both 13 adaptive and heterogeneous coarsening. Extensive evaluations on 23 real-world 14 datasets—including homophilic, heterophilic, homogeneous, and heterogeneous 15 graphs demonstrate that our method achieves superior scalability while preserving 16 the structural and semantic integrity of the original graph. Our code is available 17 here.

18 1 **Introduction**
19 Graphs are ubiquitous and have emerged as a fundamental data structure in numerous real-world 20 applications [1–3]. Broadly, graphs can be categorized into two types: (a) *Homogeneous graphs* 21 [4–6], which consist of a single type of nodes and edges. For instance, in a homogeneous citation 22 graph, all nodes represent papers, and all edges represent the "cite" relation between them; (b) 23 *Heterogeneous graphs* [7–9], which involve multiple types of nodes and/or edges, enabling the 24 modeling of richer and more realistic interactions. For example, in a recommendation system, a 25 heterogeneous graph may contain nodes of different types, such as users, items, and categories, and 26 edge types such as "(user, buys, item)", "(user, views, item)", and "(item, belongs-to, category)".

27 Although many real-world datasets are inherently heterogeneous, early research in graph machine 28 learning predominantly focused on homogeneous graphs due to their modeling simplicity, availabil29 ity of standardized benchmarks, and theoretical tractability [10, 11]. However, the limitations of 30 homogeneous representations in capturing rich semantic information have shifted attention toward 31 heterogeneous graph modeling [8, 12]. 32 As real-world networks continue to grow rapidly in size and complexity, large-scale graphs have 33 become increasingly common across various domains [1, 13–15]. This surge in scale poses signifi34 cant computational and memory challenges for learning and inference tasks on such graphs. This 35 underscores the growing importance of developing efficient and effective methodologies for process36 ing large-scale graph data. To address the issue, an expanding line of research investigates graph 37 reduction methods that compress structures without compromising essential properties. Most existing 38 graph reduction techniques, including pooling [16], sampling-based [17], condensation [18], and

1 7 8 5 6 12 6 1 411 1 411 8 2 3 5 8 2 3 5 12 9 1 12 12 9 9 7 6 7 8 12 12 2 10 10 10 12 1 411 10 8 2 3 5 5 7 7 6 6 9 4 3 11 Clockwise-merge of supernodes 7 F
A
D
C
9 3 2 4 11 10 1 7 5 12 6 8 D C
0 0 1 1 B
3 4 5 6 7 8 9 10 11 12 A
1 2 6 7 1 45 11 3 2 10 9 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 G

0 0 0 0 1 G

B
E
F
E
0 0 0 0 1 0 0 0 0 0 0 1 Adaptive GC using Coarsening Matrix 8

## 68 2 **Background**

69 **Definition 2.1 (Graph)** A graph is represented as G(V, A, X), where V = {v1*, . . . , v*N } is the set of N *nodes,* A ∈ R
N×N *is the adjacency matrix, and* X ∈ R
N×de 70 is the node fea39 coarsening-based methods [4, 19, 20]. Coarsening methods have demonstrated effectiveness in 40 preserving structural and semantic information [4, 19, 20], this study focuses on graph coarsening 41 (GC) as the primary reduction strategy. Despite advancements in existing GC frameworks, two key 42 challenges remain: 43 - **Lack of "Adaptive Reduction".** Many applications, such as interactive visualization and real-time 44 recommendations, benefit from multi-resolution graph representations. These scenarios often 45 require dynamically adjusting the coarsening ratio based on user interaction or task demands. 46 However, most existing methods generate a single fixed-size coarsened graph and must recompute 47 from scratch for each new ratio, incurring high overhead. This highlights the need for adaptive 48 coarsening frameworks that enable efficient, progressive refinement without redundant computation. 49 - **Lack of "Heterogeneous Graph Coarsening" Framework.** Existing methods typically assume 50 homogeneous node types, making them unsuitable for heterogeneous graphs with semantically 51 distinct nodes. This can result in invalid supernodes for example, merging an *author* with a *paper* 52 node in a citation graph thus violating type semantics. Moreover, node types often have different 53 feature dimensions, which standard coarsening techniques are not designed to handle. 54 **Key Contribution.** To address the dual challenges of adaptive reduction and heterogeneous GC, 55 we propose **AH-UGC**, a unified framework for Adaptive and Heterogeneous Universal Graph 56 Coarsening. We integrate locality-sensitive hashing (LSH) [4, 21, 22] with consistent hashing (CH) 57 [23, 24]. While LSH ensures that similar nodes are coarsened together based on their features and 58 connectivity, CH—a technique originally developed for load balancing—enables us to design a 59 coarsening process that supports multi-level adaptive coarsening without reprocessing the full graph.

60 To handle heterogeneous graphs, AH-UGC enforces *type-isolated coarsening*, wherein nodes are first 61 grouped by their types, and coarsening is applied independently within each type group. This ensures 62 that nodes and edges of incompatible types are never merged, preserving the semantic structure of the 63 original heterogeneous graph. Additionally, AH-UGC is naturally suited for streaming or evolving 64 graph settings, where new nodes and edges arrive over time. Our LSH- and CH-based method allows 65 new nodes to be integrated into the existing coarsened structure with minimal recomputation. To 66 summarize, **AH-UGC** is a general-purpose graph coarsening framework that supports *adaptive,* 67 *streaming, expanding, heterophilic, and heterogeneous graphs*.

ture matrix with each row Xi ∈ R
dedenoting the feature vector of node vi 71 . An edge between nodes vi and vj is indicated by Aij > 0*. Let* D ∈ R
N×N 72 *be the degree matrix with* Dii =P 73 j Aij , and let L = D − A denote the unnormalized Laplacian matrix. L ∈ SL*, where* SL =
nL ∈ R
N×NLij = Lji ≤ 0 for i ̸= j; Lii = −Pj̸=i Lijo 74 . For i ̸= j*, the matrices are* 75 related by Aij = −Lij *, and* Aii = 0. Hence, the graph G(V, A, X) *may equivalently be denoted* 76 G(L, X)*, and we use either form as contextually appropriate.* 77 **Definition 2.2** *A heterogeneous graph can be represented in two equivalent forms, with either* 78 *representation utilized as required within the paper.*
79 - **Entity-based:** A heterogeneous graph extends the standard graph structure by incorporating 80 multiple types of nodes and/or edges. Formally, a heterogeneous graph is defined as G(*V, E,* Φ, Ψ),
81 *where* Φ : V → TV and Ψ : E → TE *are node-type and edge-type mapping functions, respectively* 82 [9]. Here, TV and TE denote the sets of possible node types and edge types. When the total 83 number of node types |TV | and edge types |TE| *is equal to 1, the graph degenerates into a standard* 84 homogeneous graph (Definition *2.1).* 85 - **Type-based:** *Alternatively, a heterogeneous graph can be described as* 86 G ({X(node_type)}, {A(edge_type)}, {y(target_type)}), where feature matrices X, adjacency matri87 ces A, and target labels y *are grouped and indexed by their corresponding node, edge, and target* 88 *types [25].* 89 **Definition 2.3** Following [4, 19, 20], The Graph Coarsening (GC) problem involves learning a coarsening matrix C ∈ R
N×n 90 , which linearly maps nodes from the original graph G *to a reduced* 91 graph Gc, i.e., V → Ve. This linear mapping should ensure that similar nodes in G are grouped into the same super-node in Gc, such that the coarsened feature matrix is given by Xe = C 92 T X*. Each* non-zero entry Cij denotes the assignment of node vi 93 to super-node vej . The matrix C *must satisfy the* 94 *following structural constraints:*
S = *{C ∈* R
N×n, Cij ∈ {0, 1}, ∥Ci∥ = 1, ⟨CT
i, C
T
j⟩ = 0 ∀i ̸= j, ⟨Cl, Cl⟩ = dVel
, ∥CT
i ∥0 ≥ 1}
where dVel means the number of nodes in the l th*-supernode. The condition* ⟨CT
i
, C
T
j 95 ⟩ = 0 *ensures* that each node of G *is mapped to a unique super-node. The constraint* ∥CT
i 96 ∥0 ≥ 1 *requires that each* 97 *super-node contains at least one node.* 98 2.1 **Problem formulation and Related Work** 99 We formalize the problem through two key objectives: Goal 1. Adaptive Coarsening and Goal 2.

100 Graph Coarsening for Heterogeneous Graphs.

101 102 **Goal 1.** The objective is to compute multiple coarsened graphs {G(r)
c }
R
r=1 103 from input graph G(*V, A, X*), where each G
(r)
104 c corresponds to a target coarsening ratio r ∈ (0, 1], with105 out recomputing from scratch for each resolution. Formally, 106 the goal is to construct a family of coarsening matrices
{C(r) ∈ R
N×n
(r)
107 } such that

$$\widetilde{X}^{(r)}=({\mathcal{C}}^{(r)})^{\top}X,\quad\widetilde{A}^{(r)}=({\mathcal{C}}^{(r)})^{\top}A{\mathcal{C}}^{(r)},$$
$$\tilde{v}_{j})=\Phi(v_{i}),\quad\forall$$

with the constraint that all C
(r)
108 are derived from a single, shared projection s = HASH(X), thereby 109 ensuring consistency across coarsening levels and enabling adaptive GC.

Goal 2. The objective is to learn a coarsening matrix C ∈ R
N×n 110 , such that the resulting coarsened 111 graph Gc(V , e E, e Φe, Ψ) e satisfies the following constraints:
Φ( e vej ) = Φ(vi), ∀vej ∈ V , e ∀vi ∈ π
−1(vej ),
Ψ( e vej , vek) ∈ TE only if ∃(vi, vl) ∈ E s.t. π(vi) = vej , π(vl) = vek, 112 where π : V → Ve is the node-to-supernode mapping induced by C. These constraints guarantee 113 that: a) nodes of different types are not merged into the same supernode, and b) edge types between 114 supernodes are consistent with the original heterogeneous schema. 115 **Related Work.** Graph reduction methods have been extensively studied and can be broadly catego116 rized into optimization-based and GNN-based approaches. Among optimization-driven heuristics, 117 Loukas's spectral coarsening methods [20] including edge-based (LVE) and neighborhood-based 118 (LVN) variants aim to preserve the spectral properties of the original graph. Other techniques, such 119 as Heavy Edge Matching (HEM)[17, 26], Algebraic Distance[27], Affinity [28], and Kron reduc120 tion [29], rely on topological heuristics or structural similarity principles. FGC [19] incorporates node 121 features to learn a feature-aware reduction matrix. Despite their diverse designs, a common drawback 122 of these methods is that they are computationally demanding, often with time complexities ranging from O(n 2) to O(n 3 123 ), and are not well suited for large-scale or adaptive graph reduction settings. 124 UGC [4], a recent LSH-based framework, addresses these challenges by operating in linear time 125 and supporting heterophilic graphs. However, it produces only a single coarsened graph and must 126 recompute reductions for different coarsening levels, limiting its adaptability. GNN-based condensa127 tion methods like GCond [30] and SFGC [31] learn synthetic graphs through gradient matching but 128 require full supervision, are model-specific, and lack scalability. HGCond [25] is the only approach 129 designed for heterogeneous graphs, yet it inherits the inefficiencies of condensation-based techniques.

130 While some methods are model-agnostic, others offer partial support for heterophilic or streaming 131 graphs. Yet, no existing approach simultaneously addresses all these challenges—model-agnosticism, 132 adaptability, and support for heterophilic, heterogeneous, and streaming graphs. As illustrated in 133 Figure 2, HA-UGC is the first framework to meet all six criteria comprehensively. For details on LSH 134 and consistent hashing, see Appendix B.

## 135 3 **The Proposed Framework: Adaptive And Heterogeneous Universal Graph** 136 **Coarsening**

137 In this section we propose our framework AH-UGC to address the issues of adaptive and heteroge138 neous graph coarsening. Figure 1 shows the outline of AH-UGC. 139 3.1 **Adaptive Graph Coarsening(Goal 1)** 140 The AH-UGC pipeline closely follows the recently proposed structure of UGC but incorporates 141 consistent hashing principles to enable adaptive i.e., multi-level coarsening. Our framework introduces 142 an innovative and flexible approach to graph coarsening that removes the UGC's dependency on 143 fixed bin widths and enables the generation of multiple coarsened graphs. Similar to UGC [4], 144 AH-UGC employs an augmented representation to jointly encode both node attributes and graph 145 topology. For a given graph G(*V, A, X*), we compute a heterophily factor α ∈ [0, 1], which quantifies 146 the relative emphasis on structural information based on label agreement between connected nodes i.e., α =
|{(v,u)∈E:yv=yu}| |E| 147 . This factor is then used to blend node features Xi and adjacency vectors Ai 148 . For each node vi we calculate Fi = (1 − α) · Xi ⊕ α · Ai where ⊕ denotes concatenation.

149 This hybrid representation ensures that both local attribute similarity and topological proximity are 150 captured before the coarsening process. Importantly, this design enables our framework to handle 151 heterophilic graphs robustly by incorporating structural properties beyond mere feature similarity.

Adaptive Coarsening via Consistent and LSH Hashing. Let Fi ∈ R
d 152 denote the augmented feature vector for node vi 153 . AH-UGC applies l random projection functions using a projection matrix W ∈ R
d×land bias vector b ∈ R
l 154 , both sampled from a p-stable distribution [32]. The scalar hash score for each projection for i th 155 node is given by:
s k i = Wk · Fi + bk, ∀k ∈ {1*, . . . , l*}
156 UGC relies on a bin-width parameter (r) to control the coarsening ratio (R), but determining 157 appropriate bin-widths for different target ratios can be computationally expensive. In contrast, 158 AH-UGC eliminates the need for bin width by leveraging consistent hashing. Once the hash scores 159 (si) across projections are computed, AH-UGC enables efficient construction of coarsened graphs 160 at multiple coarsening ratios without requiring reprocessing, making it well-suited for adaptive 161 settings. We define an AGGREGATE function to combine projection scores across multiple random projectors. For each node i, the final score si 162 is computed as:

$$s_{i}=\mathrm{AGREGATE}\left(\left\{s_{i}^{k}\right\}_{k=1}^{l}\right)={\frac{1}{l}}\sum_{k=1}^{l}s_{i}^{k}$$

163 Alternative aggregation functions such as max, median, or weighted averaging can also be used, 164 depending on the design objectives. After computing the scalar hash scores {si} for all nodes vi ∈ V , we sort the nodes in increasing order of si 165 to form an ordered list L, represented as a list of super-node 166 and mapped nodes: L = [{u1 : {v1}}, {u2 : {v2}}*, . . . ,* {un : {vn}}] , where each key uj denotes a 167 super-node index, and the associated value is the set of nodes currently assigned to that super-node.

Initially, each node is its own super-node, and the number of super-nodes is |V
(0)
168 c | = |V |. At each iteration t, a super-node uj is randomly selected from the current list L
(t)
169 and merged with its 170 immediate clockwise neighbor uj+1. The updated super-node entry is given by:

$${\mathcal{L}}^{(t+1)}[j]=\{u_{j}:{\mathcal{L}}^{(t)}[u_{j}]\cup{\mathcal{L}}^{(t)}[u_{j+1}]\},$$

171 followed by the removal of uj+1 from the list. This reduces the number of super-nodes by one:
|V
(t+1)
c | = |V
(t)
c |−1. The process is repeated until the desired coarsening ratio is reached: R =
|Vc| |V | 172 . 173 Furthermore, this coarsening strategy is inherently adaptive, enabling transitions between any two 174 coarsening ratios R → T directly from the sorted list without reprocessing. 175 Since the list L is constructed using locality-sensitive hashing (LSH) principles [32], similar nodes 176 are positioned adjacently. Through Theorem 3.1 and Lemma 1, we show that the clockwise merging 177 operations in Consistent Hashing (CH) are locality-aware and effectively preserve feature similarity.

Theorem 3.1 Let *x, y* ∈ R
d 178 *, and let the projection function be defined as:* h(x) =
Pℓj=1 r
⊤
j x, rj ∼ N (0, Id) i.i.d. Then the difference h(x) − h(y) ∼ N (0, ℓ∥x − y∥
2 179 )*, and for any* 180 ε > 0:

$$\operatorname*{Pr}\left[|h(x)-h(y)|\leq\varepsilon\right]=\operatorname{erf}\left({\frac{\varepsilon}{\sqrt{2\ell}\|x-y\|}}\right)$$

181 **Proof:** The proof is deferred in Appendix D. 182 This gives the probability that two nodes, initially close in the feature space, are projected within an 183 ϵ-range in the projection space.

Lemma 1 Let *x, y, z* ∈ R
d 184 , with ∥x − y∥ ≪ ∥x − z∥*. Then the probability that a distant point* z 185 lies between x and y *after projection is:*

$$\operatorname*{Pr}[h(x)<h(z)<h(y)]\leq\Phi\left({\frac{\|x-y\|}{{\sqrt{\ell}}\|x-z\|}}\right)$$

186 where Φ *is the cumulative distribution function (CDF) of the standard normal distribution. This* 187 *result ensures that distant nodes rarely interrupt merge candidates that are close in feature space,* 188 *preserving the structural consistency of coarsened regions.* 189 **Remark 1** *Our framework also supports de-coarsening i.e., given the final sorted list and merge* 190 *history, the graph can be reconstructed to finer resolutions by reversing the merging process. However,* 191 *in this work, we restrict our focus to the coarsening direction only.*
192 **Construction of Coarsening Matrix** C. Given the score-based node assignments π : V → Ve, where π[vi] is the super-node index of vi, the binary coarsening matrix *C ∈ {*0, 1}
N×n 193 is defined such that Cij = 1 if π[vi 194 ] = vej , and Cij = 0 otherwise. Each entry Cij of the coarsening matrix is set to 1 if node viis assigned to super-node vej . Since each node receives a unique hash value hi 195 , it is exclusively 196 mapped to a single super-node. This one-to-one assignment guarantees that every super-node has at 197 least one associated node. As a result, each row of C contains exactly one non-zero entry, ensuring 198 that its columns are mutually orthogonal. The matrix C therefore adheres to the structural properties 199 defined in Equation 2.3. The adaptiveness of C stems from its sensitivity to local projection scores 200 rather than fixed bin constraints.

201 **Construction of the Coarsened Graph** Gc. The final coarsened graph Gc = (V , e A, e Fe) is constructed 202 from the coarsening matrix C. Two super-nodes vei and vej are connected if there exists at least one edge (*u, v*) ∈ E with u ∈ π
−1(vei) and v ∈ π
−1 203 (vej ). The weighted adjacency matrix is obtained via matrix multiplication: Ae = C 204 T AC. The super-node features are computed as the average of the features of the original nodes merged into the super-node: Fei =1 |π−1(vei)| Pu∈π−1 205 (vei) Fu. This 206 ensures that the coarsened representation preserves the aggregate semantic and structural content of 207 its constituent nodes. Since each super-edge aggregates multiple edges from the original graph, Ae is 208 significantly sparser than A, leading to lower memory and computation requirements downstream.

209 Algorithm 1 in Appendix G outlines the sequence of steps in our AH-UGC framework. 210 3.2 **Heterogeneous Graph Coarsening** 211 In this section, we present AH-UGC's capability to handle heterogeneous graphs. Given a heteroge212 neous graph, GA{A(author, write, paper), A(reader, read, paper)}, X{X(author), X(reader), X(paper)}, Y {y(paper)},
213 AH-UGC proceeds by first partitioning G by node type and independently applying the coarsening 214 framework to each subgraph. This ensures that only semantically similar nodes are grouped into 215 supernodes and that type-specific structure and features are preserved. Our approach naturally 216 supports varying feature dimensions and allows different coarsening ratios ηtype across node types.

217 Figure 7 in Appendix H illustrates this process, highlighting how AH-UGC preserves semantic 218 meaning compared to other GC methods that merge heterogeneous nodes indiscriminately.

219 **Construction of the Coarsened Heterogeneous Graph** Gc. The output of AH-UGC consists of a 220 set of coarsening matrices CH = {C(t) ∈ {0, 1}
|V(t)|×|Ve(t)|}t∈T ,
each of which maps original nodes of type t i.e., V(t)to their corresponding super-nodes Ve(t)
221 . Using 222 these mappings, we construct the coarsened graph

$\uparrow_{c}f$
Gc
$\left(\widetilde{A}\{\widetilde{A}(\text{author,write,paper})\,,\ \widetilde{A}(\text{reader,rand,paper})\,\},\ \widetilde{X}\{\widetilde{X}(\text{author})\,,\ \widetilde{X}(\text{reader}),\ \widetilde{X}(\text{paper})\,\},\ \widetilde{Y}\{\widetilde{y}(\text{paper})\,\}\right),$
For each node type t, the coarsened feature matrix is computed as: Xe(t) = C(t)· X(t)
223 , where rows
224 of C(t) are row-normalized so that super-node features represent the average of their constituent
nodes. The label matrix ye(paper)
225 is computed by majority voting over the labels of nodes merged 226 into each super-node. To compute the coarsened edge matrices, for each edge type Te ∈ TE, we
consider the interaction between supernodes of types node-type1
and node-type2
227 , corresponding to
the edge relation e = (node-type1
, Te, node-type2
) ∈ Ee. The coarsened adjacency matrix Ae(e)
228 is 229 then computed as:
$${\widetilde{A}}_{(e)}={\mathcal{C}}_{\mathrm{(node-type_{1})}}\cdot{\mathbf{A}}_{(e)}\cdot{\mathcal{C}}_{\mathrm{(node-type_{2})}}^{T}$$
230 This formulation accumulates the edge weights between the original nodes to define the inter231 supernode connections, thereby preserving the structural connectivity patterns between different 232 node-types of the original graph. Since each edge type is coarsened independently based on the map233 pings from its corresponding node types, Gc preserves the heterogeneous semantics and topological
234 relationships of the original graph G. Algorithm 2 in Appendix G outlines the sequence of steps in 235 our AH-UGC framework. By leveraging consistent hashing, our method ensures balanced supernode 236 formation. Theorem 3.2 provides a probabilistic upper bound on the number of nodes mapped to any 237 supernode, thereby guaranteeing load balance across supernodes with high probability. 238 **Theorem 3.2 (Explicit Load Balance via Random Rightward Merges)** Let n nodes be sorted ac239 cording to the consistent hashing scores defined earlier. Let k *supernodes be formed by performing* 240 n − k random rightward merges in the sorted list. Then, for any constant c > 0*, the maximum* 241 number of nodes in any supernode Si *satisfies:*

$$\operatorname*{Pr}{\biggl[}\operatorname*{max}_{i}|S_{i}|\leq{\frac{n}{k}}+{\frac{n(\log k+c)}{k}}{\biggr]}\geq1-e^{-c}$$

242 **Proof:** The proof is deferred in Appendix C.

243 4 **Experiments**
244 We conduct comprehensive experiments to evaluate the effectiveness of AH-UGC. First, we validate 245 its ability to perform *adaptive graph coarsening*. Second, we assess the quality of coarsened graphs 246 using node classification accuracy and spectral similarity. Finally, we demonstrate AH-UGC's 247 generalizability by evaluating its performance on *heterogeneous graphs*. 248 **Datasets:** We experiment on 23 widely-used benchmark datasets grouped into four categories:
249 - **Homophilic**: *Cora ,Citeseer, Pubmed* [33], *CS, Physics* [34], *DBLP* [35]; 250 - **Heterophilic**: *Squirrel, Chameleon, Texas, Cornell, Film, Wisconsin* [36–39], Penn49, deezer251 *europe, Amherst41, John Hopkins55, Reed98* [11]; 252 - **Heterogeneous**: *IMDB, DBLP, ACM* [7, 25]; 253 - **Large-scale**: *Flickr, Yelp*, [14] *ogbn-arxiv* [6] , *Reddit* [40]. 254 These datasets enable us to evaluate all six key components outlined in Section 2.1. For detailed 255 dataset statistics and characteristics, refer to Table 5 in Appendix A. 256 **System Specifications:** All experiments are conducted on a server equipped with two **NVIDIA RTX** 257 **A6000** GPUs (48 GB memory each) and an **Intel Xeon Platinum 8360Y** CPU with **1 TB RAM**.

Dataset VAN VAE VAC HE aJC aGS Kron FGC LAGC UGC AH-UGC

Cora 19 13 29 9 13 30 9 OOT OOT 30 7 Citeseer 28 23 37 21 22 31 20 OOT OOT 28 6

DBLP 162 138 388 204 206 1270 184 OOT OOT 131 20 PubMed 166 224 510 213 231 2351 155 OOT OOT 137 29

CS 174 237 343 216 256 1811 204 OOT OOT 233 23 Physics 411 798 943 705 906 9341 755 OOT OOT 331 54

Texas 1.59 0.91 2.66 0.77 0.96 1.32 0.8 OOT OOT 11 0.73 Cornell 1.76 0.99 2.72 0.86 1.11 1.35 0.68 OOT OOT 9 0.79

Chameleon 31 17 104 20 32 82 15 OOT OOT 21 6.73 Squirrel 384 61 398 66 342 1113 68 OOT OOT 53 4.69 Film 64 34 255 36 44 257 30 OOT OOT 92 11 Flickr 1199 2301 24176 2866 3421 59585 2858 OOT OOT 187 51 ogbn-arxiv OOT OOT OOT OOT OOT OOT OOT OOT OOT 1394 185 Reddit OOT OOT OOT OOT OOT OOT OOT OOT OOT 1595 290 Yelp OOT OOT OOT OOT OOT OOT OOT OOT OOT 6904 1374

## 258 4.1 **Adaptive Coarsening Run-Time.**

259 Given a graph G, we evaluate AH-UGC's abil260 ity to adaptively coarsen it to multiple resolu261 tions, targeting a set of coarsening ratios R = 262 {55, 50, 45, 40, 35, 30, 25, 20, 15, 10}. As described in 263 Section 3, AH-UGC leverages LSH and consistent hash264 ing to group similar nodes into supernodes, enabling the 265 construction of multiple coarsened graphs in a single pass. 266 This adaptivity significantly reduces computational over267 head compared to existing methods, which typically re268 quire reprocessing the entire graph for each target reso269 lution. The computational advantages of our approach 270 are evident in Table 1, where AH-UGC outperforms all 271 baseline methods by a significant margin, achieving the lowest coarsening time across all datasets and 272 coarsening ratios, while maintaining scalability even on large-scale graphs where other methods fail. 273 4.2 **Spectral Properties Preservation.** 274 Following the experimental setup of [4, 19, 20] we use Hyperbolic Error (HE), Reconstruction 275 Error (RcE) and Relative Eigen Error (REE) to indicate the structural similarity between G and 276 Gc. A more detailed discussion about these properties is included in Appendix F. Across three 277 spectral evaluation metrics AH-UGC delivers performance that is comparable to, and in several cases 278 surpasses, state-of-the-art methods, see Table 2. While there are minor dips in performance on a few 279 datasets, this trade-off can be justified given the significant computational efficiency and scalability 280 gains offered by our framework. These results underscore that AH-UGC achieves strong structural 281 fidelity without compromising on runtime, making it especially suitable for large-scale or adaptive 282 coarsening scenarios.

Figure 3: Empirical proof that two feature vectors remain close in projection space.

283 **LSH and consistent hashing results.** We empirically validates Theorem 3.1, see Figure 3. As 284 ϵ increases, Pr [|h(x) − h(y)| ≤ ε] approaches 1, consistent with the theoretical erf-based bound. 285 These results justify the use of consistent hashing, where each node is merged with its nearest 286 clockwise neighbor. Theorem 3.1 and Figure 3 together guarantee that similar nodes are projected to 287 nearby locations and are thus highly likely to be merged into a supernode.

Dataset VAN VAE VAC HE aJC aGS Kron UGC AH-UGC

HE

Error

DBLP 2.20 2.07 2.21 2.21 2.12 2.06 2.24 2.10 1.99

Pubmed 2.49 3.33 3.46 3.19 2.77 2.48 2.74 1.72 1.53 Squirrel 4.17 2.61 2.72 1.52 1.92 2.01 1.87 0.69 0.82

Chameleon 2.77 2.55 2.99 1.80 1.86 1.97 1.86 1.28 1.71

ReC

Error

DBLP 4.94 4.89 5.03 5.06 5.03 4.73 5.08 5.24 5.11

Pubmed 4.48 5.13 5.14 5.08 5.03 4.78 4.99 4.60 4.43 Squirrel 10.36 9.90 10.31 9.13 9.88 10.00 9.39 9.09 9.07

Chameleon 7.90 7.72 8.05 7.55 7.52 7.58 7.13 7.40 7.16

REE

Error

DBLP 0.10 0.05 0.13 0.07 0.06 0.03 0.18 0.44 0.32

Pubmed 0.05 0.97 0.88 0.71 0.48 0.06 0.42 0.31 0.21

Squirrel 0.88 0.58 0.42 0.44 0.34 0.36 0.48 0.05 0.07

Chameleon 0.76 0.69 0.67 0.38 0.38 0.35 0.52 0.09 0.12

Dataset Model VAN VAE VAC HE aJC aGS Kron UGC AH-UGC Base Citeseer GCN 59.90 60.36 58.40 61.26 60.81 61.26 62.76 65.31 65.46 70.12

SAGE 66.51 65.01 64.41 63.96 66.06 65.31 63.51 61.71 64.26 74.47 APPNP 62.16 63.36 62.46 60.21 62.91 63.81 63.21 68.61 69.06 73.12

PubMed GCN 74.34 72.46 74.06 71.72 67.36 72.87 69.59 84.66 85.47 87.60

SAGE 74.36 73.04 73.68 66.45 69.04 74.06 71.70 87.34 72.16 88.28 APPNP 76.34 77.00 73.55 75.55 71.75 76.72 70.46 85.64 85.80 87.88

Physics GCN 94.75 94.62 94.57 94.73 94.39 94.75 94.40 95.20 94.88 95.79

SAGE 96.26 96.04 96.08 95.97 96.04 96.18 96.01 95.21 95.78 96.44

APPNP 96.20 96.20 96.28 96.11 95.97 96.07 96.21 96.17 96.10 96.28

Chameleon SGC 38.60 51.58 45.79 54.91 52.63 53.15 54.39 58.60 59.65 57.46

Mixhop 40.53 51.40 43.33 50.35 49.82 49.30 54.39 58.25 58.60 63.16

GPR-GNN 40.53 46.32 41.05 39.64 40.35 43.68 51.05 54.74 52.28 55.04

Cornell SGC 67.24 67.09 68.26 68.02 68.35 69.02 68.33 76.68 76.08 72.78

Mixhop 66.79 67.67 67.14 66.07 66.45 66.71 66.41 70.64 71.61 76.49

GPR-GNN 64.98 64.27 65.17 65.00 63.55 63.67 63.48 69.66 68.00 67.46

Penn94 SGC 62.93 62.33 62.23 62.13 63.52 63.03 63.52 75.74 75.87 66.78

Mixhop 71.71 69.62 69.35 68.36 67.98 68.40 67.98 73.36 72.13 80.28 GPR-GNN 68.18 68.19 68.36 68.20 67.77 68.15 68.11 67.93 68.55 79.43

288 4.3 **Node Classification Accuracy** 289 Graph Neural Networks (GNNs) are widely used for node classification tasks [5, 40–42], where the 290 goal is to predict labels for nodes based on both node features and the underlying graph structure. In 291 this context, we evaluate the effectiveness of AH-UGC by examining how well it preserves predictive 292 performance when downstream models are trained on coarsened graphs [43]. Specifically, we 293 train several GNN models on the coarsened version of the original graph while evaluating their 294 performance on the original graph's test nodes. As discussed earlier, our experimental setup spans 295 a diverse collection of datasets, each with distinct structural characteristics. Following established 296 practice in the literature, we employ different GNN backbones tailored to each graph type. For 297 "homophilic" datasets, we use GCN [5], *Sage* [40], GAT [41], GIN [42] and *APPNP* [43], which are 298 well-suited to leverage dense neighborhood similarity. For "heterophilic" datasets, we adopt *GPRGNN* 299 [44], *MixHop* [45], *H2GNN* [46], *GCN-II* [47], *GatJK* [48] and SGC [49], which are designed to 300 handle weak or inverse homophily. For "heterogeneous" graphs, we use *HeteroSGC, HeteroGCN,*
301 *HeteroGCN2* [25] models that respect node and edge types during message passing. Complete 302 architectural and hyperparameter details are provided in Appendix E. Due to space constraints, Table 303 3 reports node classification accuracy for homophilic and heterophilic graphs on a representative 304 subset of datasets and GNN models. Please refer to Table 8 in Appendix E for comprehensive results 305 across additional datasets and architectures. The AH-UGC framework consistently delivers results 306 that are either on par with or exceed the performance of existing coarsening methods. As shown in 307 Table 3, the framework is independent of any particular GNN architecture, highlighting its robustness 308 and model-agnostic characteristics. 309 **Performance on Heterogeneous Graphs:** As outlined in Section 3, conventional graph coarsening 310 techniques struggle with preserving the semantic integrity of heterogeneous graphs. In contrast, Table 3: Node classification accuracy across various datasets and models at 50% coarsening ratio.

Dataset Model VAN VAE VAC HE aJC aGS Kron UGC AH-UGC Base

IMDB HeteroSGC 27.42 27.30 27.42 27.42 27.42 27.30 27.42 50.05 57.4 66.74

HeteroGCN 35.78 36.05 35.82 35.46 35.7 35.7 35.93 37.33 57.75 61.72 HeteroGCN2 35.78 35.82 35.82 35.82 35.82 35.82 35.82 37.65 58.57 63.47

DBLP HeteroSGC 30.95 29.43 29.43 53.07 56.65 29.43 29.43 37.06 79.18 94.10

HeteroGCN 32.38 31.77 32.75 32.75 33 35.46 31.28 63.66 66.74 84.18

HeteroGCN2 31.69 31.52 31.77 33.25 31.12 32.01 32.63 39.08 66 79.33

ACM HeteroSGC 84.46 42.31 OOT 34.54 42.31 34.54 42.31 63.63 59 92.06

HeteroGCN 36.52 35.2 OOT 35.7 35.2 35.53 35.1 38.51 84.95 92.72 HeteroGCN2 38.67 37.35 OOT 36.19 37.35 35.04 37.35 42.64 83.47 92.72

311 AH-UGC explicitly enforces type-aware coarsening, ensuring that supernodes are composed of nodes 312 from a single type, thus maintaining the heterogeneity semantics. Table 4 presents node classification 313 accuracies across various heterogeneous GNN models. AH-UGC consistently outperforms other 314 methods due to its ability to preserve type purity within supernodes. This structural consistency 315 enables all tested GNN architectures to achieve significantly higher classification performance.

316 Figure 4 illustrates the degree of supernode impurity for each method. Each bar corresponds to a 317 supernode and depicts the percentage distribution of node types within it. While supernodes generated 318 by AH-UGC are entirely type-pure, those produced by baseline methods exhibit substantial cross-type 319 mixing, leading to semantic drift and reduced model performance. Figure 5 analyzes the effect of 320 increasing coarsening ratios on node classification accuracy. As expected, all methods experience 321 performance degradation with aggressive coarsening. However, the drop is exponential for existing 322 approaches due to rising impurity levels. In contrast, AH-UGC maintains structural purity across 323 coarsening levels, resulting in a gradual, near-linear decline in accuracy. This robustness demonstrates 324 AH-UGC's superior capacity to coarsen heterogeneous graphs while preserving their semantic and 325 structural fidelity.

Figure 4: Supernode impurity across AH-UGC (left), UGC (center) and VAN (right) on IMDB dataset. Different colors represent different node types(*Movie, Director, Actor*). Figure 5: Node classification accuracy on the hDBLP dataset under decreasing coarsening ratios for three heteroGNN models: HeteroSGC (left), HeteroGCN (center), and HeteroGCN2 (right).

326 5 **Conclusion**
327 In this paper, we propose AH-UGC, a unified framework for adaptive and heterogeneous graph 328 coarsening. By integrating Locality-Sensitive Hashing (LSH) with Consistent Hashing, AH-UGC 329 efficiently produces multiple coarsened graphs with minimal overhead. Additionally, its type330 aware design ensures semantic preservation in heterogeneous graphs by avoiding cross-type node 331 merges. The framework is model-agnostic, scalable, and capable of handling both heterophilic and 332 heterogeneous graphs. We demonstrate that AH-UGC preserves key spectral properties, making it 333 applicable across diverse graph types. Extensive experiments on 23 real-world datasets with various 334 GNN architectures show that AH-UGC consistently outperforms existing methods in scalability, 335 classification accuracy, and structural fidelity.

336 **References**
337 [1] M. Kataria, E. Srivastava, K. Arjun, S. Kumar, I. Gupta, *et al.*, "A novel coarsened graph 338 learning method for scalable single-cell data analysis," *Computers in Biology and Medicine*, 339 vol. 188, p. 109873, 2025. 340 [2] A. Fout, J. Byrd, B. Shariat, and A. Ben-Hur, "Protein interface prediction using graph convolu341 tional networks," *Advances in neural information processing systems*, vol. 30, 2017. 342 [3] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and S. Y. Philip, "A comprehensive survey on graph 343 neural networks," *IEEE transactions on neural networks and learning systems*, vol. 32, no. 1, 344 pp. 4–24, 2020. 345 [4] M. Kataria, S. Kumar, *et al.*, "Ugc: Universal graph coarsening," Advances in Neural Informa346 *tion Processing Systems*, vol. 37, pp. 63057–63081, 2024. 347 [5] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," 348 *arXiv preprint arXiv:1609.02907*, 2016. 349 [6] K. Wang, Z. Shen, C. Huang, C.-H. Wu, Y. Dong, and A. Kanakia, "Microsoft academic graph: 350 When experts are not enough," *Quantitative Science Studies*, vol. 1, no. 1, pp. 396–413, 2020. 351 [7] Y. Liu, H. Zhang, C. Yang, A. Li, Y. Ji, L. Zhang, T. Li, J. Yang, T. Zhao, J. Yang, *et al.*, "Datasets 352 and interfaces for benchmarking heterogeneous graph neural networks," in *Proceedings of the* 353 *32nd ACM International Conference on Information and Knowledge Management*, pp. 5346–
354 5350, 2023. 355 [8] C. Yang, Y. Xiao, Y. Zhang, Y. Sun, and J. Han, "Heterogeneous network representation learning: 356 A unified framework with survey and benchmark," *IEEE Transactions on Knowledge and Data* 357 *Engineering*, vol. 34, no. 10, pp. 4854–4873, 2020. 358 [9] Q. Lv, M. Ding, Q. Liu, Y. Chen, W. Feng, S. He, C. Zhou, J. Jiang, Y. Dong, and J. Tang, "Are 359 we really making much progress? revisiting, benchmarking and refining heterogeneous graph 360 neural networks," in *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery* 361 *& data mining*, pp. 1150–1160, 2021. 362 [10] V. P. Dwivedi, C. K. Joshi, A. T. Luu, T. Laurent, Y. Bengio, and X. Bresson, "Benchmarking 363 graph neural networks," *Journal of Machine Learning Research*, vol. 24, no. 43, pp. 1–48, 2023. 364 [11] D. Lim, F. Hohne, X. Li, S. L. Huang, V. Gupta, O. Bhalerao, and S. N. Lim, "Large scale 365 learning on non-homophilous graphs: New benchmarks and strong simple methods," *Advances* 366 *in neural information processing systems*, vol. 34, pp. 20887–20902, 2021. 367 [12] C. Zhang, D. Song, C. Huang, A. Swami, and N. V. Chawla, "Heterogeneous graph neural 368 network," in *Proceedings of the 25th ACM SIGKDD international conference on knowledge* 369 *discovery & data mining*, pp. 793–803, 2019. 370 [13] K. Kong, J. Chen, J. Kirchenbauer, R. Ni, C. B. Bruss, and T. Goldstein, "Goat: A global trans371 former on large-scale graphs," in *International Conference on Machine Learning*, pp. 17375– 372 17390, PMLR, 2023. 373 [14] H. Zeng, H. Zhou, A. Srivastava, R. Kannan, and V. Prasanna, "Graphsaint: Graph sampling 374 based inductive learning method," *arXiv preprint arXiv:1907.04931*, 2019. 375 [15] K. Bhatia, K. Dahiya, H. Jain, P. Kar, A. Mittal, Y. Prabhu, and M. Varma, "The extreme 376 classification repository: Multi-label datasets and code," 2016. 377 [16] F. M. Bianchi, D. Grattarola, and C. Alippi, "Spectral clustering with graph neural networks for 378 graph pooling," in *International conference on machine learning*, pp. 874–883, PMLR, 2020. 379 [17] I. S. Dhillon, Y. Guan, and B. Kulis, "Weighted graph cuts without eigenvectors a multilevel 380 approach," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 29, no. 11, 381 pp. 1944–1957, 2007. 382 [18] W. Jin, L. Zhao, S. Zhang, Y. Liu, J. Tang, and N. Shah, "Graph condensation for graph neural 383 networks," *arXiv preprint arXiv:2110.07580*, 2021. 384 [19] M. Kumar, A. Sharma, and S. Kumar, "A unified framework for optimization-based graph 385 coarsening," *Journal of Machine Learning Research*, vol. 24, no. 118, pp. 1–50, 2023. 386 [20] A. Loukas, "Graph reduction with spectral and cut guarantees.," *J. Mach. Learn. Res.*, vol. 20, 387 no. 116, pp. 1–42, 2019. 388 [21] M. Datar, N. Immorlica, P. Indyk, and V. S. Mirrokni, "Locality-sensitive hashing scheme based 389 on p-stable distributions," in *Proceedings of the twentieth annual symposium on Computational* 390 *geometry*, pp. 253–262, 2004. 391 [22] M. Kataria, A. Khandelwal, R. Das, S. Kumar, and J. Jayadeva, "Linear complexity framework 392 for feature-aware graph coarsening via hashing," in *NeurIPS 2023 Workshop: New Frontiers in* 393 *Graph Learning*, 2023. 394 [23] D. Karger, E. Lehman, T. Leighton, R. Panigrahy, M. Levine, and D. Lewin, "Consistent 395 hashing and random trees: Distributed caching protocols for relieving hot spots on the world 396 wide web," in *Proceedings of the twenty-ninth annual ACM symposium on Theory of computing*, 397 pp. 654–663, 1997. 398 [24] J. Chen, B. Coleman, and A. Shrivastava, "Revisiting consistent hashing with bounded loads," 399 in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, pp. 3976–3983, 2021. 400 [25] J. Gao, J. Wu, and J. Ding, "Heterogeneous graph condensation," *IEEE Transactions on* 401 *Knowledge and Data Engineering*, vol. 36, no. 7, pp. 3126–3138, 2024. 402 [26] D. Ron, I. Safro, and A. Brandt, "Relaxation-based coarsening and multiscale graph organiza403 tion," 2010.

404 [27] J. Chen and I. Safro, "Algebraic distance on graphs," *SIAM J. Scientific Computing*, vol. 33, 405 pp. 3468–3490, 12 2011. 406 [28] O. E. Livne and A. Brandt, "Lean algebraic multigrid (lamg): Fast graph laplacian linear solver," 407 2011. 408 [29] F. Dorfler and F. Bullo, "Kron reduction of graphs with applications to electrical networks," 409 *IEEE Transactions on Circuits and Systems I: Regular Papers*, vol. 60, no. 1, pp. 150–163, 410 2013. 411 [30] W. Jin, L. Zhao, S. Zhang, Y. Liu, J. Tang, and N. Shah, "Graph condensation for graph neural 412 networks," 2021. 413 [31] X. Zheng, M. Zhang, C. Chen, Q. V. H. Nguyen, X. Zhu, and S. Pan, "Structure-free graph 414 condensation: From large-scale graphs to condensed graph-free data," *Advances in Neural* 415 *Information Processing Systems*, vol. 36, 2024. 416 [32] P. Indyk and R. Motwani, "Approximate nearest neighbors: Towards removing the curse of 417 dimensionality," in Proceedings of the Thirtieth Annual ACM Symposium on Theory of Com418 *puting*, STOC '98, (New York, NY, USA), p. 604–613, Association for Computing Machinery, 419 1998. 420 [33] Z. Yang, W. W. Cohen, and R. Salakhutdinov, "Revisiting semi-supervised learning with graph 421 embeddings," in *Proceedings of the 33nd International Conference on Machine Learning, ICML* 422 *2016, New York City, NY, USA, June 19-24, 2016*, JMLR Workshop and Conference Proceedings, 423 2016. 424 [34] O. Shchur, M. Mumme, A. Bojchevski, and S. Günnemann, "Pitfalls of graph neural network 425 evaluation," *ArXiv preprint*, 2018. 426 [35] X. Fu, J. Zhang, Z. Meng, and I. King, "Magnn: Metapath aggregated graph neural network for 427 heterogeneous graph embedding," in *Proceedings of The Web Conference 2020*, pp. 2331–2341, 428 2020. 429 [36] J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra, "Beyond homophily in graph 430 neural networks: Current limitations and effective designs," in *Advances in Neural Information* 431 *Processing Systems* (H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, eds.), 432 vol. 33, pp. 7793–7804, Curran Associates, Inc., 2020. 433 [37] H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang, "Geom-gcn: Geometric graph convolutional 434 networks," *arXiv preprint arXiv:2002.05287*, 2020.

435 [38] J. Zhu, R. A. Rossi, A. Rao, T. Mai, N. Lipka, N. K. Ahmed, and D. Koutra, "Graph neural 436 networks with heterophily," in *Proceedings of the AAAI conference on artificial intelligence*, 437 vol. 35, pp. 11168–11176, 2021. 438 [39] L. Du, X. Shi, Q. Fu, X. Ma, H. Liu, S. Han, and D. Zhang, "Gbk-gnn: Gated bi-kernel graph 439 neural networks for modeling both homophily and heterophily," in *Proceedings of the ACM* 440 *Web Conference 2022*, pp. 1550–1558, 2022. 441 [40] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," 442 2017. 443 [41] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention 444 networks," in *6th International Conference on Learning Representations, ICLR 2018, Vancouver,* 445 *BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*, 2018. 446 [42] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?," *arXiv* 447 *preprint arXiv:1810.00826*, 2018. 448 [43] Z. Huang, S. Zhang, C. Xi, T. Liu, and M. Zhou, "Scaling up graph neural networks via graph 449 coarsening," 2021. 450 [44] E. Chien, J. Peng, P. Li, and O. Milenkovic, "Adaptive universal generalized pagerank graph 451 neural network," *arXiv preprint arXiv:2006.07988*, 2020.

452 [45] S. Abu-El-Haija, B. Perozzi, A. Kapoor, N. Alipourfard, K. Lerman, H. Harutyunyan, 453 G. Ver Steeg, and A. Galstyan, "Mixhop: Higher-order graph convolutional architectures via 454 sparsified neighborhood mixing," in *international conference on machine learning*, pp. 21–29, 455 PMLR, 2019. 456 [46] J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra, "Beyond homophily in graph 457 neural networks: Current limitations and effective designs," *Advances in neural information* 458 *processing systems*, vol. 33, pp. 7793–7804, 2020. 459 [47] M. Chen, Z. Wei, Z. Huang, B. Ding, and Y. Li, "Simple and deep graph convolutional networks," 460 in *International conference on machine learning*, pp. 1725–1735, PMLR, 2020. 461 [48] K. Xu, C. Li, Y. Tian, T. Sonobe, K.-i. Kawarabayashi, and S. Jegelka, "Representation learning 462 on graphs with jumping knowledge networks," in *International conference on machine learning*, 463 pp. 5453–5462, PMLR, 2018. 464 [49] F. Wu, A. Souza, T. Zhang, C. Fifty, T. Yu, and K. Weinberger, "Simplifying graph convolutional 465 networks," in *International conference on machine learning*, pp. 6861–6871, Pmlr, 2019.

466 [50] B. Kulis and K. Grauman, "Kernelized locality-sensitive hashing for scalable image search," in 467 *2009 IEEE 12th international conference on computer vision*, (Kyoto, Japan), pp. 2130–2137, 468 IEEE, IEEE, 2009.

469 [51] J. Buhler, "Efficient large-scale sequence comparison by locality-sensitive hashing," Bioinfor470 *matics*, vol. 17, no. 5, pp. 419–428, 2001. 471 [52] O. Chum, J. Philbin, M. Isard, and A. Zisserman, "Scalable near identical image and shot 472 detection," in *Proceedings of the 6th ACM international conference on Image and video retrieval*, 473 pp. 549–556, 2007. 474 [53] H. A. David and H. N. Nagaraja, *Order statistics*. John Wiley & Sons, 2004. 475 [54] N. Malik, R. Gupta, and S. Kumar, "Hyperdefender: A robust framework for hyperbolic gnns," 476 *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, pp. 19396–19404, Apr. 477 2025. 478 [55] C. Li and D. Goldwasser, "Encoding social information with graph convolutional networks 479 forPolitical perspective detection in news media," in *Proceedings of the 57th Annual Meeting of* 480 *the Association for Computational Linguistics*, (Florence, Italy), pp. 2594–2604, Association 481 for Computational Linguistics, July 2019. 482 [56] A. Paliwal, F. Gimeno, V. Nair, Y. Li, M. Lubin, P. Kohli, and O. Vinyals, "Reinforced genetic 483 algorithm learning for optimizing computation graphs," 2019.

484 [57] T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, and P. W. Battaglia, "Learning mesh-based 485 simulation with graph networks," *arXiv preprint arXiv:2010.03409*, vol. 32, p. 18, 2020. 486 [58] R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and J. Leskovec, "Graph convolu487 tional neural networks for web-scale recommender systems," in *Proceedings of the 24th ACM* 488 *SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD '18, (New 489 York, NY, USA), p. 974–983, Association for Computing Machinery, 2018. 490 [59] G. Bravo Hermsdorff and L. Gunderson, "A unifying framework for spectrum-preserving graph 491 sparsification and coarsening," *Advances in Neural Information Processing Systems*, vol. 32, 492 p. 12, 2019. 493 [60] Y. Liu, T. Safavi, A. Dighe, and D. Koutra, "Graph summarization methods and applications: A 494 survey," *ACM computing surveys (CSUR)*, vol. 51, no. 3, pp. 1–34, 2018.

## 495 A **Datasets**

496 We experiment on 24 widely-used benchmark datasets grouped into four categories: **(a) Homophilic**: 497 *Cora ,Citeseer, Pubmed* [33], *CS, Physics* [34], *DBLP* [35]; **(b) Heterophilic**: *Squirrel, Chameleon,* 498 *Texas, Cornell, Film, Wisconsin* [36–39], *Penn49, deezer-europe, Amherst41, John Hopkins55,* 499 *Reed98* [11]; **(c) Heterogeneous**: *IMDB, DBLP, ACM* [7, 25]; and **(d) Large-scale**: *Flickr, Yelp*, 500 [14] *ogbn-arxiv* [6] , *Reddit* [40]. These datasets enable us to evaluate all six key components 501 outlined in Section 2.1. Please refer to Table 5 and 6 for detailed dataset statistics and characteristics.

Category Data Nodes Edges Feat. Class H.R(α)

Cora 2,708 5,429 1,433 7 0.19

Citeseer 3,327 9,104 3,703 6 0.26

DBLP 17,716 52,867 1,639 4 0.18

CS 18,333 163,788 6,805 15 0.20

PubMed 19,717 44,338 500 3 0.20

Physics 34,493 247,962 8,415 5 0.07

| Homophilic dataset Heterophilic dataset Large dataset   |
|---------------------------------------------------------|

Texas 183 309 1703 5 0.91

Cornell 183 295 1703 5 0.70

Film 7600 33544 931 5 0.78

Squirrel 5201 217073 2089 5 0.78

Chameleon 2277 36101 2325 5 0.75

Penn94 41,554 1.36M 5 2 0.53

Deezer-europe 28,281 185.5k 31.24k 2 -

Amherst41 2235 181.9k 1193 3 -

John-Hopkin55 41,554 2.7M 4,814 3 -

Reed98 962 37.6k 745 3 -

Large dataset

Flickr 89,250 899,756 500 7 -

Reddit 232,965 11.60M 602 41 -

Ogbn-arxiv 169,343 1.16M 128 40 -

Yelp 716,847 13.95M 300 100 -

502

Table 6: Summary of Heterogeneous graph datasets

Table 5: Summary of the datasets.

## 503 B **Locality-Sensitive Hashing And Consistent Hashing**

504 Locality-Sensitive Hashing (LSH) is a technique for hashing high-dimensional data points so that 505 similar items are more likely to collide (i.e., hash to the same bucket) [32, 50, 51]. It is commonly 506 used in approximate nearest neighbor search, dimensionality reduction, and randomized algorithms 507 [52]. For example, a hash function h(·) is locality-sensitive with respect to a similarity measure s(·, ·) 508 if Pr[h(x) = h(y)] increases with s(*x, y*). Gaussian LSH schemes, such as those using random 509 projections, are particularly effective for preserving Euclidean distances [4, 22]. 510 In the consistent hashing (CH) [23, 24] scheme, objects/requests are hashed to random bins/servers 511 on the unit circle, as shown in Figure 6. Objects are then assigned to the closest bin in the clockwise 512 direction. CH was originally proposed for load balancing in distributed systems; it maps data points

| Dataset                         | Nodes                                                                               | Edges                  | Features   | Classes   |
|---------------------------------|-------------------------------------------------------------------------------------|------------------------|------------|-----------|
| Movie - 4278                    | (Movie, to, Director) - 4278                                                        |                        |            |           |
| Director - 2081                 | (Movie, to, Actor) - 12828                                                          |                        |            |           |
| IMDB                            | 3061                                                                                | Movie: 3               |            |           |
| Actor - 5257                    | (Director, to, Movie) - 4278 (Actor, to, Movie) - 12828 (Author, to, Paper) - 19645 |                        |            |           |
| Author - 4057                   | (Paper, to, Author) - 19645                                                         | Author - 334           |            |           |
| Paper - 4231                    | (Paper, to, Term) - 85810                                                           | Paper - 4231           | Author: 4  |           |
| Term - 7723                     | (Paper, to, Conference) - 14328                                                     | Term - 50              |            |           |
| Conference - 50                 | (Term, to, Paper) - 85810                                                           | Conference - NA        |            |           |
| (Conference, to, Paper) - 14328 |                                                                                     |                        |            |           |
| DBLP                            | (Paper, cite, Paper) - 5343 (Paper, ref, Paper) - 5343                              |                        |            |           |
| Paper - 3025                    | (Paper, to, Author) - 9949                                                          |                        |            |           |
| Author - 5959                   | (Author, to, Paper) - 9949                                                          | All except term - 1902 |            |           |
| Subject - 56                    | (Paper, to, Subject) - 3025                                                         | Term - NA              | Paper: 3   |           |
| Term - 1902                     | (Subject, to, Paper) - 3025 (Paper, to, Term) - 255619 (Term, to, Paper) - 255619   |                        |            |           |
| ACM                             |                                                                                     |                        |            |           |

513 to buckets such that small changes in input (e.g., adding or removing an object) do not drastically 514 affect the overall assignment. We aim to employ CH for adaptive graph coarsening, as it enables 515 stable and scalable grouping of similar objects/nodes. When combined with LSH, consistent hashing 516 offers a powerful mechanism for adaptive graph reduction.

517 C Proof of Theorem 3.2 518 **Theorem C.1 (Explicit Load Balance via Random Rightward Merges)** Let n nodes be sorted ac519 cording to the consistent hashing scores defined earlier. Let k *supernodes be formed by performing* 520 n − k random rightward merges in the sorted list. Then, for any constant c > 0*, the maximum* 521 number of nodes in any supernode Si *satisfies:*

$$\operatorname*{Pr}\left[\operatorname*{max}_{i}|S_{i}|\leq{\frac{n}{k}}+{\frac{n(\log k+c)}{k}}\right]\geq1-e^{-c}$$

522 **Proof** Let U1*, . . . , U*k−1 ∼ Uniform(0, 1) and let U(1) < · · · < U(k−1) be their order statistics.

523 Define the spacings:
I1 = U(1) − 0, I2 = U(2) − U(1)*, . . . , I*k = 1 − U(k−1)
524 Then (I1*, . . . , I*k) form a random partition of the unit interval [0, 1]. It is a classical result (e.g., [53])
525 that:
526 - The vector (I1*, . . . , I*k) ∼ Dirichlet(1*, . . . ,* 1), 527 - Each individual spacing Ii ∼ Beta(1, k − 1).

Tail bound on Ii. The PDF of Ii 528 is:

$$f(t)=(k-1)(1-t)^{k-2},\quad t\in[0,1]$$

529 and its tail probability is:
$$\Pr[I_{i}>t]=(1-t)^{k-1}$$

Choose t =
log k+c k 530 . Then:

+ $\begin{matrix}1\\ 1\end{matrix}$). 
$\mathbf{a}=\mathbf{a}\cdot\mathbf{a}$. 
$$\operatorname*{Pr}[I_{i}>t]\leq\exp\left(-(\log k+c)\right)={\frac{1}{k}}e^{-c}$$
531 **Union bound.** Over all k intervals:

 Prove that every unit ball is $ \frac{1}{k}$  $ \Pr\left[\max_i I_i>\frac{\log k+c}{k}\right]\leq k\cdot\frac{1}{k}e^{-c}=e^{-c}\Rightarrow\Pr\left[\max_i I_i\leq\frac{\log k+c}{k}\right]\geq1-e^{-c}$
532 Scaling to n **nodes.** We model the sorted list of n nodes as uniformly spaced over [0, 1]. Each spacing Ii 533 then corresponds to a fraction of the list, and multiplying by n yields the expected number 534 of nodes in that supernode:

$$|S_{i}|=n\cdot I_{i}\Rightarrow\operatorname*{max}_{i}|S_{i}|=n\cdot\operatorname*{max}_{i}I_{i}\leq{\frac{n}{k}}+{\frac{n(\log k+c)}{k}}$$

535 This completes the proof.

536 D Proof of Theorem 3.1 Theorem D.1 (Projection Proximity for Similar Points) Let *x, y* ∈ R
d 537 *, and define the projection* 538 *function:*

$$h(x)=\sum_{j=1}^{\ell}r_{j}^{\top}x,\quad r_{j}\sim{\mathcal{N}}(0,I_{d})\;i.i.d.$$
_Then the difference $h(x)-h(y)\sim\mathcal{N}(0,\ell\|x-y\|^2)$, and for any $\varepsilon>0$:_
$$\operatorname*{Pr}\left[|h(x)-h(y)|\leq\varepsilon\right]=\operatorname{erf}\left({\frac{\varepsilon}{\sqrt{2\ell}\|x-y\|}}\right)$$

Proof Let z = x − y ∈ R
d 540 . Then:

$$h(x)-h(y)=\sum_{j=1}^{\ell}r_{j}^{\top}x-\sum_{j=1}^{\ell}r_{j}^{\top}y=\sum_{j=1}^{\ell}r_{j}^{\top}(x-y)=\sum_{j=1}^{\ell}r_{j}^{\top}z$$

$|\ensuremath{\widehat{\phantom{\rule{0.04pt}{0.04pt}}}}\ensuremath{\leftarrow}\ensuremath{\downarrow}$
Each term r
⊤ j 541 z is a linear projection of a standard Gaussian vector, hence:

$$r_{j}^{\top}z\sim{\mathcal{N}}(0,\|z)$$
2) = N (0, ∥x − y∥
2)
542 Since the rj are independent, the sum of ℓ such independent variables is:

$\downarrow$ . 
$$h(x)-h(y)\sim{\mathcal{N}}(0)$$

h(x) − h(y) ∼ N (0, ℓ∥x − y∥
2)
543 Now consider the probability:
$$\operatorname*{Pr}\left[|h(x)-h(y)|\leq\varepsilon\right]$$
This is the cumulative probability within ε of a zero-mean Gaussian with variance ℓ∥x − y∥
2
544 . Let
σ
2 = ℓ∥x − y∥
2
545 . Then:
$$\operatorname*{Pr}\left[|Z|\leq\varepsilon\right]=\operatorname{erf}\left({\frac{\varepsilon}{\sqrt{2\sigma^{2}}}}\right)=\operatorname{erf}\left({\frac{\varepsilon}{\sqrt{2\ell}\|x-y\|}}\right)$$
546 as required.

547 E **Node Classification Accuracy**
548 Graph Neural Networks (GNNs), designed to operate on graph data [4, 54], have demonstrated strong 549 performance across a range of applications [55–58]. Nevertheless, their scalability to large graphs 550 remains a significant bottleneck. Motivated by recent efforts in scalable learning [43], we explore 551 how our graph coarsening framework can improve the efficiency and scalability of GNN training, 552 enabling more effective processing of large-scale graph data. Specifically, we train several GNN 553 models on the coarsened version of the original graph while evaluating their performance on the 554 original graph's test nodes. As discussed earlier in 4.3, our experimental setup spans a diverse 555 collection of datasets, each with distinct structural characteristics. For *homophilic* graph settings, we 556 follow the architectural configurations proposed in UGC [4], see Table 7. For *heterophilic* graphs, the 557 GNN model designs are based on the implementations introduced in [11]. The *heterogeneous* GNN 558 architectures are adopted directly from [25]. 559 Table 8 reports node classification accuracy for homophilic and Table 9 reports node classification 560 accuracy for heterophilic graphs. The AH-UGC framework consistently delivers results that are 561 either on par with or exceed the performance of existing coarsening methods. As shown in Table 3, 562 the framework is independent of any particular GNN architecture, highlighting its robustness and 563 model-agnostic characteristics.

| Model     | Layers                  | Hidden Units          | Activation   | Dropout                    | Learning rate   | Decay   | Epoch   |
|-----------|-------------------------|-----------------------|--------------|----------------------------|-----------------|---------|---------|
| GCN       | 3 × GCNConv             | 64 → 64 → Output      | ReLU         | Yes (intermediate layers)  | 0.003           | 0.0005  | 500     |
| APPNP     | Linear → Linear → APPNP | 64 → 64 → 10 → Output | ReLU         | Yes (before Linear layers) | 0.003           | 0.0005  | 500     |
| GAT       | 2 × GATv2Conv           | 64 × 8 → Output       | ELU          | Yes (p=0.6)                | 0.003           | 0.0005  | 500     |
| GIN       | 2 × GATv2Conv           | 64 × 8 → Output       | ELU          | Yes (p=0.6)                | 0.003           | 0.0005  | 500     |
| GraphSAGE | 2 × SAGEConv            | 64 → Output           | ReLU         | Yes (after first layer)    | 0.003           | 0.0005  | 500     |

Dataset Model VAN VAE VAC HE aJC aGS Kron UGC AH-UGC Base Cora GCN 77.34 83.79 81.58 81.58 83.05 82.32 79.18 79.00 77.34 85.81

SAGE 80.47 82.87 81.95 81.76 83.97 82.87 82.87 76.61 76.24 89.87 GIN 78.63 77.53 74.58 76.79 79.18 78.08 77.16 55.43 77.34 87.29 GAT 77.16 78.08 75.87 74.40 81.21 80.47 74.58 78.26 81.03 87.10

APPNP 82.87 84.53 82.50 84.53 84.34 85.26 82.87 86.37 84.53 88.58

DBLP GCN 79.65 80.36 80.55 79.99 80.55 79.26 79.40 85.75 80.27 84.00

SAGE 80.58 80.07 80.16 80.81 80.61 81.57 79.48 68.56 68.31 84.08 GIN 79.40 79.20 80.38 78.83 77.96 78.18 78.01 73.95 79.82 83.26 GAT 74.43 78.32 76.49 77.56 78.97 77.51 75.93 77.93 79.48 82.25 APPNP 84.25 83.80 83.63 83.60 83.29 84.25 84.05 84.84 85.18 85.75

CS GCN 91.63 92.01 91.19 92.03 91.41 87.26 92.55 92.66 92.47 93.51

SAGE 94.32 94.19 94.57 94.24 93.94 93.70 94.02 89.17 89.83 94.82 GIN 89.80 89.69 89.83 90.70 89.61 88.00 90.64 86.77 81.07 83.50 GAT 91.98 91.52 92.31 91.57 90.67 91.19 89.50 89.83 90.48 91.84

Citeseer GCN 59.90 60.36 58.40 61.26 60.81 61.26 62.76 65.31 65.46 70.12

SAGE 66.51 65.01 64.41 63.96 66.06 65.31 63.51 61.71 64.26 74.47 GIN 59.60 60.36 59.00 59.45 56.15 62.91 57.50 64.41 63.66 71.62 GAT 53.45 58.55 54.95 53.45 62.76 59.75 57.35 65.76 69.21 71.32 APPNP 62.16 63.36 62.46 60.21 62.91 63.81 63.21 68.61 69.06 73.12

PubMed GCN 74.34 72.46 74.06 71.72 67.36 72.87 69.59 84.66 85.47 87.60

SAGE 74.36 73.04 73.68 66.45 69.04 74.06 71.70 87.34 72.16 88.28 GIN 57.17 66.53 61.53 60.11 65.66 60.85 63.46 82.42 83.97 85.75 GAT 46.85 40.03 52.68 50.60 53.29 56.99 69.09 84.66 84.63 87.39 APPNP 76.34 77.00 73.55 75.55 71.75 76.72 70.46 85.64 85.80 87.88

Physics GCN 94.75 94.62 94.57 94.73 94.39 94.75 94.40 95.20 94.88 95.79

SAGE 96.26 96.04 96.08 95.97 96.04 96.18 96.01 95.21 95.78 96.44 GIN 94.90 94.56 94.78 94.49 93.79 94.79 92.65 94.41 94.94 95.66 GAT 94.97 95.01 95.00 94.65 95.36 94.60 94.85 96.02 95.10 94.28 APPNP 96.20 96.20 96.28 96.11 95.97 96.07 96.21 96.17 96.10 96.28

## 564 F **Spectral Properties**

565 1. **Relative Eigen Error (REE):** REE used in [4, 19, 20] gives the means to quantify the measure 566 of the eigen properties of the original graph G that are preserved in coarsened graph Gc.

567 **Definition F.1** *REE is defined as follows:*

$$R E E(L,L_{c},k)=\frac{1}{k}\sum_{i=1}^{k}\frac{|\widetilde{\lambda_{i}}-\lambda_{i}|}{\lambda_{i}}$$

$$(1)$$

where λi and λe 568 i are top k eigenvalues of original graph Laplacian (L*) and coarsened graph* 569 Laplacian (Lc*) matrix, respectively.* 570 2. **Hyperbolic error (HE):** HE [59] indicates the structural similarity between G and Gc with the 571 help of a lifted matrix along with the feature matrix X of the original graph. 572 **Definition F.2** *HE is defined as follows:*

$$H E=a r c c o s h(\frac{||(L-L_{\mathrm{lift}})X||_{F}^{2}||X||_{F}^{2}}{2t r a c e(X^{T}L X)t r a c e(X^{T}L_{\mathrm{lift}}X)}+1)$$
+ 1) (2)

$$(2)$$

| Dataset         | Model   | VAN   | VAE   | VAC   | HE    | aJC   | aGS   | Kron   | UGC   | AH-UGC   | Base   |
|-----------------|---------|-------|-------|-------|-------|-------|-------|--------|-------|----------|--------|
| Film            | SGC     | 29.36 | 27.84 | 29.95 | 26.15 | 26.89 | 25.74 | 27.74  | 21.47 | 21.68    | 27.63  |
| Mixhop          | 28.21   | 30.68 | 29.84 | 29.52 | 29.10 | 29.15 | 31.15 | 21.57  | 21.79 | 30.92    |        |
| GCN2            | 26.15   | 28.47 | 28.00 | 26.94 | 27.63 | 25.84 | 29.42 | 19.47  | 20.42 | 28.36    |        |
| GPR-GNN         | 26.52   | 27.95 | 27.10 | 27.74 | 26.78 | 28.36 | 28.26 | 20.68  | 21.31 | 29.73    |        |
| GatJK           | 26.11   | 25.89 | 25.79 | 25.10 | 25.31 | 25.31 | 26.63 | 22.42  | 21.21 | 23.94    |        |
| deezer-europe   | SGC     | 54.55 | 55.31 | 54.50 | 55.38 | 54.48 | 54.69 | 55.15  | 54.49 | 55.06    | 57.08  |
| Mixhop          | 58.42   | 59.10 | 58.48 | 58.82 | 58.34 | 57.38 | 58.80 | 59.78  | 60.98 | 64.31    |        |
| GCN2            | 57.79   | 58.34 | 57.76 | 58.34 | 57.15 | 57.57 | 58.25 | 58.00  | 58.46 | 60.88    |        |
| GPR-GNN         | 56.30   | 56.85 | 56.70 | 56.77 | 55.73 | 55.55 | 56.31 | 58.44  | 58.46 | 56.97    |        |
| GatJK           | 55.21   | 57.50 | 54.63 | 55.76 | 55.31 | 56.03 | 56.87 | 57.01  | 57.33 | 59.01    |        |
| Amherst41       | SGC     | 61.42 | 63.19 | 59.06 | 60.83 | 63.39 | 62.99 | 63.78  | 78.74 | 73.82    | 73.46  |
| Mixhop          | 59.25   | 58.46 | 57.68 | 58.66 | 59.06 | 63.78 | 58.66 | 69.29  | 64.37 | 72.48    |        |
| GCN2            | 62.99   | 62.01 | 60.63 | 59.25 | 58.66 | 60.63 | 56.50 | 71.06  | 68.50 | 71.74    |        |
| GPR-GNN         | 59.45   | 58.86 | 58.07 | 55.91 | 57.68 | 59.25 | 55.71 | 66.73  | 63.98 | 60.93    |        |
| GatJK           | 57.48   | 63.58 | 60.24 | 62.99 | 61.61 | 64.76 | 62.60 | 64.37  | 67.72 | 78.13    |        |
| Johns Hopkins55 | SGC     | 62.72 | 69.19 | 68.77 | 69.35 | 68.85 | 70.28 | 69.19  | 73.80 | 72.96    | 73.77  |
| Mixhop          | 63.64   | 65.74 | 68.18 | 64.90 | 62.22 | 64.90 | 63.73 | 69.94  | 67.25 | 73.56    |        |
| GCN2            | 66.16   | 67.51 | 67.42 | 64.23 | 65.49 | 65.74 | 64.40 | 71.12  | 65.24 | 73.45    |        |
| GPR-GNN         | 62.05   | 63.06 | 62.30 | 62.80 | 60.37 | 61.96 | 61.71 | 66.33  | 63.31 | 64.95    |        |
| GatJK           | 62.80   | 69.10 | 67.34 | 66.41 | 65.99 | 65.58 | 67.00 | 69.77  | 65.32 | 77.12    |        |
| Reed98          | SGC     | 53.46 | 57.14 | 53.92 | 52.07 | 55.30 | 58.06 | 53.92  | 57.60 | 57.60    | 68.79  |
| Mixhop          | 50.69   | 58.99 | 49.77 | 48.85 | 55.30 | 59.45 | 53.46 | 60.37  | 52.53 | 62.43    |        |
| GCN2            | 56.68   | 59.45 | 51.61 | 50.69 | 51.61 | 56.68 | 50.69 | 61.75  | 57.14 | 64.16    |        |
| GPR-GNN         | 48.39   | 57.60 | 48.39 | 45.62 | 55.76 | 58.06 | 53.46 | 57.60  | 54.84 | 56.07    |        |
| GatJK           | 55.30   | 58.99 | 53.00 | 51.61 | 51.61 | 56.22 | 53.92 | 62.67  | 60.83 | 69.94    |        |
| Squirrel        | SGC     | 31.97 | 33.13 | 30.98 | 36.66 | 34.97 | 36.59 | 35.59  | 40.89 | 39.51    | 43.61  |
| Mixhop          | 36.28   | 30.21 | 24.60 | 34.90 | 28.44 | 27.90 | 37.05 | 46.12  | 43.97 | 46.40    |        |
| GCN2            | 39.74   | 42.28 | 39.20 | 41.74 | 37.97 | 39.12 | 41.51 | 43.12  | 44.35 | 50.72    |        |
| GPR-GNN         | 29.36   | 25.67 | 28.82 | 28.82 | 26.44 | 27.06 | 30.59 | 45.12  | 43.74 | 34.39    |        |
| GatJK           | 31.44   | 37.43 | 32.82 | 46.12 | 38.36 | 37.89 | 46.81 | 40.89  | 39.43 | 46.01    |        |
| Chameleon       | SGC     | 38.60 | 51.58 | 45.79 | 54.91 | 52.63 | 53.15 | 54.39  | 58.60 | 59.65    | 57.46  |
| Mixhop          | 40.53   | 51.40 | 43.33 | 50.35 | 49.82 | 49.30 | 54.39 | 58.25  | 58.60 | 63.16    |        |
| GCN2            | 47.37   | 52.11 | 56.84 | 59.30 | 59.65 | 58.95 | 59.12 | 51.40  | 49.82 | 67.11    |        |
| GPR-GNN         | 40.53   | 46.32 | 41.05 | 39.64 | 40.35 | 43.68 | 51.05 | 54.74  | 52.28 | 55.04    |        |
| GatJK           | 41.40   | 52.46 | 36.49 | 60.00 | 56.49 | 55.96 | 62.63 | 54.39  | 55.44 | 71.05    |        |
| Cornell         | SGC     | 67.24 | 67.09 | 68.26 | 68.02 | 68.35 | 69.02 | 68.33  | 76.68 | 76.08    | 72.78  |
| Mixhop          | 66.79   | 67.67 | 67.14 | 66.07 | 66.45 | 66.71 | 66.41 | 70.64  | 71.61 | 76.49    |        |
| GCN2            | 66.31   | 66.83 | 66.98 | 67.64 | 67.17 | 62.91 | 66.50 | 72.71  | 70.90 | 77.18    |        |
| GPR-GNN         | 64.98   | 64.27 | 65.17 | 65.00 | 63.55 | 63.67 | 63.48 | 69.66  | 68.00 | 67.46    |        |
| GatJK           | 63.48   | 65.31 | 68.28 | 66.00 | 67.40 | 66.21 | 66.64 | 70.09  | 70.35 | 78.37    |        |
| Penn94          | SGC     | 62.93 | 62.33 | 62.23 | 62.13 | 63.52 | 63.03 | 63.52  | 75.74 | 75.87    | 66.78  |
| Mixhop          | 71.71   | 69.62 | 69.35 | 68.36 | 67.98 | 68.40 | 67.98 | 73.36  | 72.13 | 80.28    |        |
| GCN2            | 71.79   | 69.55 | 70.75 | 69.52 | 69.61 | 71.41 | 69.61 | 71.85  | 72.07 | 81.75    |        |
| GPR-GNN         | 68.18   | 68.19 | 68.36 | 68.20 | 67.77 | 68.15 | 68.11 | 67.93  | 68.55 | 79.43    |        |
| GatJK           | 67.94   | 67.05 | 66.73 | 66.21 | 66.34 | 66.06 | 66.33 | 69.23  | 69.26 | 80.74    |        |

where L is the Laplacian matrix and X ∈ R
N×d 573 is the feature matrix of the original input graph, Llift is the lifted Laplacian matrix defined in [20] as Llift = CLcC
T where C ∈ R
N×n 574 is the 575 coarsening matrix and Lc is the Laplacian of Gc.

576 3. **Reconstruction Error (RcE)**
577 **Definition F.3** Let L be the original Laplacian matrix and Llift *be the lifted Laplacian matrix,*
578 then the reconstruction error (RE) [19, 60] is defined as:

$\eqref{eq:walpha}$
$$-\ L_{l i j\hbar}\|{\hat{\vec{F}}}$$
RcE = ∥L − L*lift*∥
2 F (3)

## 579 G **Algorithms** 580 H **Heterogenous Graph Coarsening**

| HE Error ReC Error REE Error   |
|--------------------------------|

Dataset VAN VAE VAC HE aJC aGS Kron UGC AH-UGC

Cora 2.04 2.08 2.14 2.19 2.13 1.95 2.14 1.96 2.03

DBLP 2.20 2.07 2.21 2.21 2.12 2.06 2.24 2.10 1.99

Pubmed 2.49 3.33 3.46 3.19 2.77 2.48 2.74 1.72 1.53

Squirrel 4.17 2.61 2.72 1.52 1.92 2.01 1.87 0.69 0.82

Chameleon 2.77 2.55 2.99 1.80 1.86 1.97 1.86 1.28 1.71

Deezer-Europe 1.90 1.97 2.04 1.95 1.90 1.62 1.90 1.76 1.61

Penn94 1.96 1.52 1.65 1.57 1.51 1.43 1.55 1.05 1.09

Cora 3.78 3.83 3.90 3.95 3.91 3.71 3.92 4.07 4.14

DBLP 4.94 4.89 5.03 5.06 5.03 4.73 5.08 5.24 5.11

Pubmed 4.48 5.13 5.14 5.08 5.03 4.78 4.99 4.60 4.43

Squirrel 10.36 9.90 10.31 9.13 9.88 10.00 9.39 9.09 9.07

Chameleon 7.90 7.72 8.05 7.55 7.52 7.58 7.13 7.40 7.16

Deezer-Europe 5.08 5.06 5.19 5.04 5.04 4.68 5.01 8.03 8.05

Penn94 7.77 7.71 7.77 7.73 7.73 7.63 7.76 7.71 7.74

Cora 0.09 0.07 0.05 0.04 0.11 0.09 0.03 0.64 0.66

DBLP 0.10 0.05 0.13 0.07 0.06 0.03 0.18 0.44 0.32

Pubmed 0.05 0.97 0.88 0.71 0.48 0.06 0.42 0.31 0.21

Squirrel 0.88 0.58 0.42 0.44 0.34 0.36 0.48 0.05 0.07

Chameleon 0.76 0.69 0.67 0.38 0.38 0.35 0.52 0.09 0.12

Deezer-Europe 0.48 0.29 0.47 0.25 0.21 0.02 0.19 0.35 0.35

Penn94 0.31 0.02 0.05 0.02 0.09 0.05 0.08 0.22 0.23

## Algorithm 1 Ah-Ugc: Adaptive Universal Graph Coarsening

Require: Input G(*V, A, X)*, l ← Number of Projectors 1: α =
|{(v,u)∈E:yv=yu}| |E|; α is heterophily factor, yi ∈ R
N is node labels, E denotes edge list 2: F =(1 − α) · X ⊕ α · A	
3: S ← F · W + b; S ∈ R
n×l// compute projections 4: W ∈ R
d×l, b ∈ R
l ∼ D(·) // sample projections 5: S ← F · W + b; S ∈ R
n×l// compute projections 6: si ← AGGREGATE({Si,k}
l k=1) = 1 l Plk=1 Si,k ∀i ∈ {1*, . . . , n*} // mean aggregation 7: L ← sort({vi}
n i=1) by ascending si // ordered node list 8: L ← [{u1 : {v1}}, {u2 : {v2}}*, . . . ,* {un : {vn}}] // initial super-nodes 9: **while** |L|/|V | > r do 10: uj ∼ Uniform(L) // sample a super-node 11: L[uj ] ← L[uj ] ∪ L[uj+1] // merge with right neighbor 12: *L ← L \ {*uj+1} // remove right neighbor

13: *C ∈ {*0, 1}
|L|×|V |, Cij ←
(1 if vj ∈ L[ui]
0 otherwise// partition matrix 14: C ← row-normalize(C) // normalize rows: Pj Cij = 1 15: Fe ← CF ; Ae ← CAC
T// coarsened features and adjacency 16: **return** Gc = (V , 
e A, e Fe), C
Algorithm 2 Heterogeneous Graph Coarsening Require: Graph G{X(node_type)}, {A(edge_type)}, {y(target_type)}, compression ratio η Ensure: Condensed graph Gc
{Xe(node_type)}, {Ae(edge_type)}, {Ye(target_type)}

1: for each node type t do 2: rt ← η · |Vt| 3: G
coarse t, Ct ← AH-UGC(Xt, At, rt)
4: Xet ← node features from G
coarse t 5: if t is target type **then**
6: yet[i] ← majority vote of yj for vj ∈ Ct[i]
7: for each edge type e = (t1, t2) do 8: Initialize Aee ∈ R
|Vet1 |×|Vet2 | 9: for each (vi, vj ) ∈ Ae do 10: u ← super-node index of vi via Ct1 11: v ← super-node index of vj via Ct2 12: Aee[u, v] ← Aee[u, v] + 1 13: **return** Gc
{Xe(node_type)}, {Ae(edge_type)}, {Ye(target_type)}

User 1 Paper 2 Dept. 1 User 2 User 3 Dept. 2 User 4 User 4 User 2 Paper 2 Paper 1 Dept. 2 Dept. 1 User 5 Paper 1 Paper 3 Paper 4 User 1 Paper 3 a) Other GC methods: Semantic Meaning Lost Paper 4 User 5 User 1 User 5 User 2 User 3 User 4 User 3 Dept. 1 Dept. 2 Node Type Edge Type User User Paper User Department writes / reads works in collaborates Paper 2 Paper 1 Paper 3 Paper 4 b) HA-UGC: Semantic Meaning Preserved