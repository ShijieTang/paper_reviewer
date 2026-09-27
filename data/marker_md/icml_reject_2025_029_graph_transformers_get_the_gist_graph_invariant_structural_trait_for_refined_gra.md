011

014 015 016

018

024

026

034

036

038

# Graph Transformers Get the GIST: Graph Invariant Structural Trait for Refined Graph Encoding

Anonymous Authors<sup>1</sup>

# Abstract

Graph classification is a core machine learning task with diverse applications across scientific fields. Transformers have recently gained significant attention in this area, addressing key limitations of traditional Graph Neural Networks (GNNs), including oversmoothing and oversquashing, while leveraging the attention mechanism. However, a key challenge remains: effectively encoding graph structure information within the all-to-all attention mechanism, arguably the first step of all Graph Transformers. To address this, we propose a novel structural feature, termed Graph Invariant Structural Trait (GIST), designed to capture substructures within a graph through estimated pairwise node intersections. Furthermore, we extend GIST into a structural encoding method tailored for the attention mechanism in graph transformers. Our theoretical analysis and empirical observations demonstrate that GIST effectively captures structural information critical for graph classification. Extensive experiments further reveal that graph transformers incorporating GIST into their attention mechanism achieve superior performance compared to state-of-the-art baselines. These findings highlight the potential of GIST to enhance the structural encoding of Graph Transformers.

# 1. Introduction

Graph classification is a fundamental problem in machine learning with widespread applications in various domains, including chemistry, biology, and drug discovery [\(Dwivedi](#page-8-0) [et al.,](#page-8-0) [2022a](#page-8-0)[;c;](#page-8-1) [Irwin et al.,](#page-8-2) [2012;](#page-8-2) [Wu et al.,](#page-9-0) [2017\)](#page-9-0). The ability to classify graphs accurately enables advancements in predicting molecular properties, understanding complex biological interactions, and discovering novel therapeutic compounds. Traditional Graph Neural Networks (GNNs) [\(Kipf](#page-8-3) [& Welling,](#page-8-3) [2017;](#page-8-3) [Han et al.,](#page-8-4) [2022\)](#page-8-4) have been the cornerstone for such tasks, leveraging neighborhood aggregation to learn node and graph representations. However, GNNs often suffer from limitations such as oversmoothing [\(Keriven,](#page-8-5) [2022\)](#page-8-5), oversquashing [\(Black et al.,](#page-8-6) [2023\)](#page-8-6), and restricted expressivity [\(Wang & Zhang,](#page-9-1) [2024\)](#page-9-1) due to their reliance on local message-passing mechanisms.

Recently, Transformers [\(Vaswani et al.,](#page-9-2) [2017\)](#page-9-2) have emerged as a promising alternative for graph representation learning due to their global attention mechanism, which addresses many of the inherent limitations of GNNs. Transformers' ability to model complex interactions between entities makes them particularly attractive for graph classification [\(Ying et al.,](#page-9-3) [2021\)](#page-9-3). However, applying Transformers to graph data is not a seamless procedure, still posing unique challenges. Unlike sequential or image data, graph nodes typically lack inherent self-identity, making it difficult for Transformers to distinguish between entities purely based on their features. Without incorporating meaningful structural information, the attention mechanism in Transformers struggles to capture complex graph relationships effectively.

Existing approaches have attempted to improve Transformers with graph structural inductive bias by integrating positional or structural features, such as shortest path distances [\(Ying et al.,](#page-9-3) [2021\)](#page-9-3), Laplacian eigenvector-based encodings [\(Dwivedi et al.,](#page-8-0) [2022a\)](#page-8-0), and random walk-based features [\(Rampa´sek et al.](#page-9-4) ˇ , [2022;](#page-9-4) [Ma et al.,](#page-9-5) [2023\)](#page-9-5). While these methods provide some structural context, they either fail to capture comprehensive substructural information essential for distinguishing complex graph patterns [\(Rampa´sek et al.](#page-9-4) ˇ , [2022\)](#page-9-4) or focus predominantly on a limited set of substructures while neglecting higher-order structural relationships [\(Wollschlager et al.,](#page-9-6) [2024\)](#page-9-6). The challenge remains to identify a more expressive and comprehensive set of structural features, and devise efficient methods for encoding them within the Transformer's self-attention mechanism.

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

In this work, we introduce a novel structural feature called Graph Invariant Structural Trait (GIST), which captures the inherent substructures within a graph by estimating k-hop pairwise node intersections. Our approach is grounded in

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109 the theoretical understanding that the cardinality of the intersection between two nodes' k-hop neighborhoods can serve as an effective permutation-invariant feature for substructure characterization, providing a robust foundation for graph classification. Incorporating GIST as a structural bias enhances the Transformer's capability to discern complex graph patterns, leading to improved classification performance. We further propose an efficient randomized algorithm to estimate GIST, ensuring scalability across large (number of) graphs. Through extensive experiments on various graph classification benchmarks, we demonstrate that integrating GIST into Graph Transformers achieves stateof-the-art performance and offers deeper insights into the structural properties of graph data.

Our key contributions are as follows:

- We introduce GIST, a method that encodes graph structure using pairwise k-hop substructure vector. These substructure vectors are efficiently computed by estimating the interaction cardinality between the k-hop neighborhoods of node pairs.
- We incorporate GIST into attention mechanisms of graph Transformers to enhance structural encoding. We provide both theoretical and empirical evidence demonstrating its effectiveness as a graph-invariant representation.
- We evaluate GIST-augmented graph Transformers on standard graph classification benchmarks, showing consistent performance improvements.

The introduction of GIST opens new avenues for enhancing the structural encoding capabilities of Transformers, paving the way for more effective and interpretable graph classification models.[<sup>1</sup>](#page-1-0)

# 2. Motivation

Transformers, originally designed for sequential data, lack an inherent mechanism to capture the structural biases of graph data as highlighted in [\(Ying et al.,](#page-9-3) [2021;](#page-9-3) [Rampa´sek](#page-9-4) ˇ [et al.,](#page-9-4) [2022\)](#page-9-4). Without a well-designed structural bias (structural encoding), they treat all nodes as equally related, failing to utilize the relational dependencies critical for graph tasks [\(Ying et al.,](#page-9-3) [2021;](#page-9-3) [Brody et al.,](#page-8-7) [2022\)](#page-8-7).

Challenge 1. Capturing Graph Substructures in Structural Encoding. The first key challenge in designing effective structural encodings for Graph Transformers is capturing the substructures within a graph, as these substructures often represent critical local patterns, or fragments that define the graph's overall characteristics [\(Ying et al.,](#page-9-3) [2021;](#page-9-3) [Ma et al.,](#page-9-5) [2023;](#page-9-5) [Wollschlager et al.,](#page-9-6) [2024\)](#page-9-6). While many early-stage structural encoding methods, such as shortest path distance (SPD) [\(Ying et al.,](#page-9-3) [2021\)](#page-9-3), provide a notion of

(a) (u, v1) from the same 6-ring substructure

(b) (u, v2) from different substructures: a 6-ring and a 2-path

Figure 1. k-hop Substructure Vector Visualization (Def. [3.1\)](#page-3-0) of ZINC molecule. The substructures of node pairs in the form of intersection cardinality of their common neighborhood at different distances from u and v are "GIST"-ed into the Substructure Vector. Specifically, each cell (ku, kv) in the Substructure Vector denotes the number of nodes that are exactly k<sup>u</sup> hops from u and k<sup>v</sup> hops from v. The variations in the Substructure Vector help the selfattention mechanism distinguish structural differences between node pairs, such as (u, v1) and (u, v2). For example, in Figure [1a,](#page-1-1) the pair (u, v1), which belongs to the same 6-ring substructure, has intersection cardinalities I(2,2) = I(4,2) = I(2,4) = 1. In contrast, the pair (u, v2), where u and v<sup>2</sup> belong to different substructures (a 6-ring and a 2-path), has I(2,2) = I(4,2) = I(2,4) = 0.

proximity between nodes, they often struggle to effectively capture and represent substructures.

Challenge 2. Aggregating Diverse Substructures Information. As highlighted in [\(Wollschlager et al.,](#page-9-6) [2024\)](#page-9-6), it is equally important for structural encodings to enable the aggregation of information across diverse substructures, rather than restricting it to similar or localized patterns. Graphs, such as molecules, often exhibit a variety of substructures that interact in complex ways, and limiting information flow to nodes in different structures can hinder the model's ability to capture global dependencies and cross-pattern interactions. This is particularly important in domains like chemistry, biology, and social networks, where functional or structural properties often arise from specific subgraph

<sup>1</sup>The code will be made publicly available upon publication.

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

164

![Diagram of a complex graph with nodes and edges. Node u is labeled at the top right. The graph consists of several complex segments of connected nodes. Edges are shown with colored arrows: a blue arrow points from node u to the left, and a purple arrow points from node u to the right. The graph is divided into two main segments: a top segment on the left and a bottom segment on the right. The top segment contains three nodes and four edges, while the bottom segment contains three nodes and four edges. A label 'u' is positioned at the top right of the top segment.]()

Figure 2. Node Clustering via Spectral Clustering Using Learned GIST Features in Graph Transformers on ZINC molecule graph. Nodes within the same local substructures are clustered together: 6-rings (purple), 2-path (cyan), and X-shape (light blue).

arrangements (i.e., rings and bonds in molecules) rather than the global graph structure alone [\(Yang et al.,](#page-9-7) [2018;](#page-9-7) [Yu &](#page-9-8) [Gao,](#page-9-8) [2022\)](#page-9-8). Many recent structural biases, such as shortest path distance [\(Ying et al.,](#page-9-3) [2021\)](#page-9-3) or those based on random walks [\(Rampa´sek et al.](#page-9-4) ˇ , [2022;](#page-9-4) [Ma et al.,](#page-9-5) [2023\)](#page-9-5), are effective at capturing simple substructures like cycles but tend to focus predominantly on these patterns, neglecting the interactions between different substructures [\(Wollschlager](#page-9-6) [et al.,](#page-9-6) [2024\)](#page-9-6). For example, in Figure [2,](#page-2-0) it is more beneficial for u to aggregate information from the 6-ring, X-shape, and 2-path substructures rather than solely focusing on another 6-ring that mirrors its own structural pattern. This highlights the need for a structural encoding that can help attention mechanisms effectively learn the substructures while enabling nodes to distinguish their own substructures from those of others, guiding attention based on the distinct structural relationships between nodes.

Observation 1: Intersection Cardinality as a Discriminative Subgraph Feature. Empirically, we observe that the intersection cardinality of common neighborhoods between two nodes (u, v) can also serve as a powerful and discriminative feature encoding the k−hop subgraph structures. As illustrated in Figure [1,](#page-1-1) the intersections of common neighborhoods at different hop distances provide a structured way for u to differentiate between the ring structure containing v<sup>1</sup> and the 2-path structure containing v2, based on the differences in the in-between graph structures. Specifically, for (u, v1), which belongs to the same 6-ring substructure, the intersection cardinality values I(2,2), I(4,2), and I(2,4) are all nonzero, indicating strong shared neighborhood connectivity. In contrast, (u, v2), which belongs to different substructures (a 6-ring and a 2-path), lacks these intersection values but instead exhibits nonzero intersection cardinality in positions such as I(3,2) and I(2,3), which are absent for

(u, v1). This contrast highlights how different substructure compositions lead to distinct intersection patterns, enabling the model to effectively distinguish between structurally similar and dissimilar node pairs, guiding the self-attention mechanism to weigh higher-order interactions accordingly.

Observation 2: Intersection Cardinality Enhances Structural Awareness in Self-Attention Mechanisms. Moreover, we empirically observe that incorporating an attention mechanism with intersection cardinality as an attention bias enables the attention mechanism to learn distinct substructures within the graph. In Figure [2,](#page-2-0) we train a Transformer architecture on the on ZINC dataset [\(Dwivedi et al.,](#page-8-0) [2022a\)](#page-8-0), introducing only the intersection cardinality (formally defined in Section [4](#page-4-0) as GIST) as a bias in the attention scores. After training the model, we apply Spectral Clustering to group nodes based on the learned GIST features. The GIST features facilitate representation aggregation across structurally similar regions, allowing node u to integrate information from another ring structure. This effect is evident as nodes from both rings are grouped into the same clusters, marked in dark blue and cyan. Furthermore, certain nodes positioned at the boundaries of these substructures act as "information exchange points", facilitating communication between distant regions of the graph. For example, the cyan-colored node within the "X" substructure is assigned to the same cluster as the ring nodes, effectively facilitating representation aggregation between two different substructures—an ability that current GNNs and Graph Transformers struggle with due to their inherent locality constraints. We note that this is not a cherry-picked example; rather, this phenomenon consistently occurs across multiple samples in the ZINC dataset after the Transformer is trained.

# 3. GIST: Graph Invariant Structural Trait

In this section, we formally introduce the graph invariant structural trait (GIST). We start by introducing how to encode the k-hop substructure of a node pair (u, v) based on the k-hop common neighborhood between them. Next, we introduce how to use encoded k-hop substructures in a graph to form GIST. Finally, we introduce how to efficiently compute GIST with randomized hashing algorithms.

Notation: We denote an undirected graph G = (V, E), which contains a set V of *n* nodes (vertices) and a set E of *m* edges (links). Each node v ∈ V has d<sup>n</sup> associated node features x<sup>v</sup> ∈ <sup>R</sup> <sup>d</sup><sup>n</sup> , while each edge eu,v ∈ E connecting node pair (u, v) has d<sup>e</sup> associated edge features yu,v ∈ <sup>R</sup> d<sup>e</sup> (yu,v = 0 d<sup>e</sup> if there is no edge between u and v). For every node v ∈ V, we denote its k-hop neighborhoods as Nk(v). Nk(v) consists of all vertices that can be reached from v with less or equal to k edges. Subsequently, we define the k-hop common neighborhood of a node pair (u, v) as

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

218

C<sup>k</sup>u,k<sup>v</sup> (u, v) = N<sup>k</sup><sup>u</sup> (u) ∩ N<sup>k</sup><sup>v</sup> (v), which is a set of nodes in the graph that can be reached within k<sup>u</sup> from u and k<sup>v</sup> edges from v, respectively.

### 3.1. Encoding k-hop Substructure of a Node Pair

We encode the k-hop substructure of a node pair (u, v) in a vector. This vector is computed based on the k-hop common neighborhood C<sup>k</sup>u,k<sup>v</sup> (u, v).

Definition 3.1 (k-hop substructure vector). Given a pair of node (u, v) ∈ G, we propose capturing the k−hop graph structure between u and v with two types of features computed by k-hop common neighborhood C<sup>k</sup>u,k<sup>v</sup> (u, v) as follows:

- I<sup>k</sup>u,k<sup>v</sup> (u, v) as the cardinality of common neighborhoods that are exactly k<sup>u</sup> hops from node u and k<sup>v</sup> hops from node v, computed as:

$$\mathcal{I}_{k_u, k_v}(u, v) = |\mathcal{C}_{k_u, k_v}(u, v)| - \sum_{\substack{x \leq k_u, y \leq k_v \\ (x, y) \neq (k_u, k_v)}} \mathcal{I}_{x, y}(u, v),$$

where I1,1(u, v) = |C1,1(u, v)| for u and v.

- T<sup>k</sup><sup>u</sup> (u, v): the cardinality of nodes that are exactly k<sup>u</sup> hop from vertex u and greater than k hop from v (and vice-versa for T<sup>k</sup><sup>v</sup> (v, u)), computed as:

$$\mathcal{T}_{k_u,k}(u, v) = |\mathcal{N}_{k_u}(u)| - \mathcal{T}_{k_u-1,k}(u) - \sum_{i=1}^{k_u} \sum_{j=1}^k \mathcal{I}_{i,j}(u, v)$$

For any node pair (u, v), there would be k <sup>2</sup> numbers of I<sup>k</sup>u,k<sup>v</sup> (u, v), k numbers of T<sup>k</sup>u,k(u, v), and k numbers of T<sup>k</sup>v,k(v, u). Finally, we encode the k−hop graph substructure surrounding node pair (u, v) as a k−hop substructure vector Sk(u, v). Sk(u, v) starts with I<sup>k</sup>u,k<sup>v</sup> (u, v) for every pair of ku, k<sup>v</sup> ≤ k. Next, we fill the rest of the dimension in Sk(u, v) with T<sup>k</sup>u,k(u, v) for each k<sup>u</sup> ≤ k hop and T<sup>k</sup>v,k(v, u) for each k<sup>v</sup> ≤ k hop.

As we see from Definition [3.1,](#page-3-0) computing the k−hop substructure vector requires first compute the cardinality of the k-hop common neighborhood C<sup>k</sup>u,k<sup>v</sup> (u, v).

## 3.2. GIST: Graph Invariant Structural Trait

We define GIST as a three-dimensional matrix defined on the k-hop common neighborhood C<sup>k</sup>u,k<sup>v</sup> (u, v) (see Definition [3.1\)](#page-3-0) between every pair of node (u, v) in graph G.

Definition 3.2 (Graph Invariant Structural Trait (GIST)). Let G = (V, E) denote a graph with n nodes (|V| = n). We define the k-hop graph invariant structural trait (GIST) as a matrix S(G) ∈ <sup>R</sup> n×n×(k <sup>2</sup>+2k) , where each entry Si,j (G) ∈ R k <sup>2</sup>+2k is the k-hop substructure between node v<sup>i</sup> , v<sup>j</sup> (see Definition [3.1\)](#page-3-0). We also use S(G)u,v to represent the GIST value between node u, v ∈ G.

GIST provides a compact representation of a graph's structural properties, encoding its topology and connectivity patterns by capturing higher-order relational dependencies among nodes and substructures. This encoding enables the differentiation of substructures, offering a detailed understanding of complex higher-order relationships, as illustrated in Figure [2](#page-2-0) and Section [2.](#page-1-2) We would like to note one component of this representation: the diagonal entry Si,i(G), which essentially encodes the k-hop neighborhood surrounding a node v<sup>i</sup> ∈ V. This local structure provides a positional reference that differentiates nodes based on their placement within the global graph topology, enabling the model to capture long-range dependencies beyond direct connectivity. Mathematically, GIST represents pairwise node interactions as a matrix, where each interaction is encoded as a vector of dimension (k <sup>2</sup> + 2k). This formulation preserves both local and global structural information, making GIST a comprehensive descriptor of graph architecture suitable for various analytical and learning-based applications.

### 3.3. Efficiently Compute GIST with Randomized Hashing

In this section, we show how to efficiently compute GIST by reducing the time complexity from O(k <sup>2</sup>n 4 ) to O(k <sup>2</sup>n 2 ). It is obvious that computing GIST S(G) requires O(k <sup>2</sup>n 4 ) time complexity. We note that for a node pair (u, v), the exact computation of their k-hop common neighborhood C<sup>k</sup>u,k<sup>v</sup> (u, v) incurs a cost of O(n 2 ), while calculating Su,v(G) requires O(k <sup>2</sup>n 2 ). Consequently, computing Su,v(G) for all node pairs in a graph G results in an overall complexity of O(k <sup>2</sup>n 4 ). Exact intersection calculations are computationally expensive, making them impractical for large graphs. Following [\(Chamberlain et al.,](#page-8-8) [2022;](#page-8-8) [Le et al.,](#page-8-9) [2024\)](#page-8-9), we propose to efficiently and unbiasedly estimate the cardinality of k-hop common neighborhood C<sup>k</sup>u,k<sup>v</sup> (u, v) by decomposing it as:

$$|\mathcal{C}_{k_u, k_v}(u, v)| = \mathcal{J}_{k_u, k_v}(u, v) \cdot \mathcal{U}_{k_u, k_v}(u, v) \quad (1)$$

Here, J<sup>k</sup>u,k<sup>v</sup> (u, v) represents the Jaccard similarity between ku-hop neighborhoods N<sup>k</sup><sup>u</sup> (u) and kv-hop neighborhoods N<sup>k</sup><sup>v</sup> (v). U<sup>k</sup>u,k<sup>v</sup> (u, v) denotes the cardinality of the union N<sup>k</sup><sup>u</sup> (u)∪ N<sup>k</sup><sup>v</sup> (v). Next, we can estimate J<sup>k</sup>u,k<sup>v</sup> (u, v) with the constant-time collisions of the MinHash signatures of N<sup>k</sup><sup>u</sup> (u) and N<sup>k</sup><sup>v</sup> (v) as shown in Algorithm [1.](#page-4-1) We note that MinHash provides an unbiased estimator to the J<sup>k</sup>u,k<sup>v</sup> (u, v) since the collision probability between the MinHash signatures of N<sup>k</sup><sup>u</sup> (u) and N<sup>k</sup><sup>v</sup> are equal to J<sup>k</sup>u,k<sup>v</sup> (u, v) We can also estimate U<sup>k</sup>u,k<sup>v</sup> (u, v) with the mergeable Hyper-LogLog sketch as Algorithm [1.](#page-4-1) We note that HyperLogLog also provides an unbiased estimator to U<sup>k</sup>u,k<sup>v</sup> (u, v).

Finally, we multiply the estimated J˜ ku,k<sup>v</sup> (u, v) and

226

228

231

234

236

238

254

256

258

260

264

266

268

271

274

Algorithm 1 Algorithm for computing intersection cardinality |C<sup>k</sup>u,k<sup>v</sup> (u, v)|

Input: Graph G = (V, E), max hops k, hops ku, kv, m MinHash functions H = {h1, . . . , hm}, HyperLogLog

parameter p and regularizer constant α<sup>p</sup> Output: Intersection cardinality |C<sup>k</sup>u,k<sup>v</sup> (u, v)| {Step 1. Pre-compute MinHash signatures} for v ∈ V, h<sup>j</sup> ∈ H do Mv[j, 0] ← h<sup>j</sup> (v) {Initialize MinHash signatures} end for for i = 1 to k do for v ∈ V, h<sup>j</sup> ∈ H do Mv[j, i] ← min u∈N(v) Mu[j, i − 1], Mv[j, i − 1] end for end for {Step 2. Pre-compute HyperLogLog sketches} m ← 2 p for v ∈ V do Compute k-hop HyperLogLog sketch H<sup>v</sup> ∈ <sup>R</sup> m×k end for {Step 3. Compute intersection cardinality} for (u, v) ∈ V × V do J˜ ku,k<sup>v</sup> (u, v) ← JACCARD-EST(ku, kv, m, Mu, Mv) U˜ ku,k<sup>v</sup> (u, v) ← HLL-EST(ku, kv, Hu, Hv) |C<sup>k</sup>u,k<sup>v</sup> (u, v)| ← J˜ ku,k<sup>v</sup> (u, v) · U˜ ku,k<sup>v</sup> (u, v) end for return |C<sup>k</sup>u,k<sup>v</sup> (u, v)| Function: JACCARD-EST(ku, kv, m, Mu, Mv) Input: hops ku, kv, number of MINHASH functions m, and k−hop MinHash values Mu, M<sup>v</sup> Output: Jaccard similarity J˜ ku,k<sup>v</sup> (u, v) J˜ ku,k<sup>v</sup> (u, v) ← 0 for j = 1 to m do if Mu(j, ku) = Mv(j, kv) then J˜ ku,k<sup>v</sup> (u, v) ← J˜ ku,k<sup>v</sup> (u, v) + 1 end if end for J˜ ku,k<sup>v</sup> (u, v) ← J˜ ku,k<sup>v</sup> (u, v)/m return J˜ ku,k<sup>v</sup> (u, v) EndFunction Function: HLL-EST(ku, kv, Hu, Hv) Input: hops ku, kv, HyperLogLog sketches Hu, H<sup>v</sup> Output: Union cardinality U˜ ku,k<sup>v</sup> (u, v) H<sup>k</sup>u,k<sup>v</sup> ← 0 m for j = 1 to m do H<sup>k</sup>u,k<sup>v</sup> [j] ← max Hu[j, ku], Hv[j, kv] end for U˜ ku,k<sup>v</sup> (u, v) ← αpm<sup>2</sup> ( P<sup>m</sup> <sup>i</sup>=0 2 −Hku,kv [i] ) −1 return U˜ ku,k<sup>v</sup> (u, v) EndFunction

U˜ ku,k<sup>v</sup> (u, v) together and form an unbiased estimator to |C<sup>k</sup>u,k<sup>v</sup> (u, v)|. This unbiased estimation can serve as an efficient alternative to exact computation for |C<sup>k</sup>u,k<sup>v</sup> (u, v)|. With MinHash and HyperLogLog, we reduce the computation time for Su,v(G) from O(k <sup>2</sup>n 2 ) to O(k 2 ), leading to O(k <sup>2</sup>n 2 ) time for compute GIST.

# 4. Graph Transformers Get the GIST

We see GIST can be naturally integrated into graph tansformers for graph structural encoding in the self-attention mechanism. As a result, we introduce the GIST attention for graph transformers.

Definition 4.1 (GIST attention). Let G = (V, E) denote a graph with n nodes (|V| = n). Let x<sup>u</sup> ∈ <sup>R</sup> <sup>d</sup><sup>n</sup> denote the representation of node u ∈ V. Let yu,v ∈ <sup>R</sup> <sup>d</sup><sup>e</sup> denote the representation of edge between nodes u, v ∈ V. Let w<sup>v</sup> ∈ <sup>R</sup> <sup>d</sup>n×d<sup>n</sup> and w<sup>e</sup> ∈ <sup>R</sup> <sup>d</sup>n×<sup>d</sup> denote the model weight. Let S(G) denote the k-hop GIST computed from G (see Definition [3.2\)](#page-3-1). We define the GIST attention as a transform ψ : R <sup>d</sup><sup>n</sup> → <sup>R</sup> <sup>d</sup><sup>n</sup> on every node feature x<sup>u</sup> as:

$$\psi(x_u) = \sum_{v \in \mathcal{V}} \mathcal{A}_{u,v} \cdot (w_v x_v + w_e \hat{\mathcal{A}}_{u,v}),$$

where Aˆ u,v ∈ <sup>R</sup> d and attention score Au,v ∈ <sup>R</sup> are:

$$\begin{aligned} e_{u,v} &= \phi_y(y_{u,v}) + \phi_S(S_{u,v}(\mathcal{G})) \\ \mathcal{A}_{u,v} &= \sigma(\langle w_Q x_u + w_K x_v + w_b, e_{u,v} \rangle). \\ \hat{\mathcal{A}}_{u,v} &= (w_Q x_u + w_K x_v + w_b) \odot e_{u,v}. \end{aligned}$$

Here ϕ<sup>y</sup> : <sup>R</sup> <sup>d</sup><sup>e</sup> → <sup>R</sup> d and ϕ<sup>S</sup> : <sup>R</sup> k <sup>2</sup>+2<sup>k</sup> → <sup>R</sup> d are MLP networks that align the representation of edge and GIST (see Definition [3.2\)](#page-3-1) into same d-dimensional vector for addition. wQ, w<sup>K</sup> ∈ <sup>R</sup> <sup>d</sup>×d<sup>n</sup> and w<sup>b</sup> ∈ <sup>R</sup> d are model weights and bias, respectively.

GIST attention can be viewed as a graph invariant with the following statement.

Theorem 4.2 (Informal version of Theorem [A.1\)](#page-10-0). *Let* G = (V, E) *denote a graph with* n *nodes (*|V| = n*). Let* S(G) ∈ *denote the* k*-hop GIST (see Definition [3.2\)](#page-3-1) computed on* G*. We show that the GIST attention (see Definition [4.1\)](#page-4-2)* ψ(xu) *for every node* u ∈ V *is invariant under graph isomorphism.*

We provide the formal version of this theorem and proof in Appendix [A.](#page-10-1) In other words, the permutation of node orders in the graph does not break the substructure in the graph due to graph isomorphism. As a result, it does not affect the value of GIST.

We use GIST attention as the building blocks and form a graph transformer with multiple GIST attention blocks. We view GIST attention as a way of modelling node interactions with the awareness of the graph structure.

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

328

## 5. Experiment

In this section, we aim to rigorously evaluate the effectiveness of GIST by addressing the following key research questions and providing corresponding insights:

- RQ 1: How well does GIST facilitate the learning and differentiation of substructures in graph classification tasks?
- RQ 2: To what extent does GIST enable long-range dependencies in Graph Transformers?
- RQ 3: How sensitive is GIST to the maximum hop distance for computing intersection cardinality?

#### 5.1. Settings

We evaluate the proposed method on three benchmark suites comprising a total of 12 datasets, spanning small-scale to large-scale settings: the Long-Range Graph Benchmark (LRGB) [\(Dwivedi et al.,](#page-8-1) [2022c\)](#page-8-1), MoleculeNet [\(Wu et al.,](#page-9-0) [2017\)](#page-9-0), ZINC [\(Dwivedi et al.,](#page-8-0) [2022a\)](#page-8-0), and ZINC-full [\(Irwin](#page-8-2) [et al.,](#page-8-2) [2012\)](#page-8-2). These datasets are specifically curated to emphasize challenges in structural encoding and long-range dependency modeling, with diverse applications in domains such as chemistry and biology.

Baselines. We benchmark the performance of our method against recent state-of-the-art baselines across multiple categories, including Graph Transformers, Graph Neural Networks (GNNs), hybrid models combining Transformers and GNNs, as well as pretrained graph models: GraphGPS [\(Rampa´sek et al.](#page-9-4) ˇ , [2022\)](#page-9-4), GRIT [\(Ma et al.,](#page-9-5) [2023\)](#page-9-5), Subgraphormer [\(Bar-Shalom et al.,](#page-8-10) [2024\)](#page-8-10), Frag-Net [\(Wollschlager et al.,](#page-9-6) [2024\)](#page-9-6), GatedGCN [\(Dwivedi et al.,](#page-8-1) [2022c\)](#page-8-1), SAN [\(Kreuzer et al.,](#page-8-11) [2021\)](#page-8-11), Graphormer [\(Ying et al.,](#page-9-3) [2021\)](#page-9-3), Graphormer-GD [\(Zhang et al.,](#page-9-9) [2023b\)](#page-9-9), GCN [\(Kipf](#page-8-3) [& Welling,](#page-8-3) [2017\)](#page-8-3), GIN [\(Xu et al.,](#page-9-10) [2018\)](#page-9-10), NGNN [\(Zhang &](#page-9-11) [Li,](#page-9-11) [2021\)](#page-9-11), DS-GNN [\(Bevilacqua et al.,](#page-8-12) [2022\)](#page-8-12), DSS-GNN [\(Bevilacqua et al.,](#page-8-12) [2022\)](#page-8-12), GNN-AK [\(Zhao et al.,](#page-9-12) [2022\)](#page-9-12), GNN-AK+ [\(Zhao et al.,](#page-9-12) [2022\)](#page-9-12), SUN [\(Frasca et al.,](#page-8-13) [2022\)](#page-8-13), OSAN [\(Qian et al.,](#page-9-13) [2022\)](#page-9-13), DS-GNN [\(Bevilacqua et al.,](#page-8-14) [2023\)](#page-8-14), GNN-SSWL [\(Zhang et al.,](#page-9-14) [2023a\)](#page-9-14), GNN-SSWL+ [\(Zhang et al.,](#page-9-14) [2023a\)](#page-9-14), GraphMVP [\(Liu et al.,](#page-9-15) [2022\)](#page-9-15), MGSSL [\(Zhang et al.,](#page-9-16) [2021\)](#page-9-16), and GraphFP [\(Luong & Singh,](#page-9-17) [2023\)](#page-9-17).

Experimental Settings. For each dataset, we train our proposed method on the training set and select the epoch with the best validation performance. We then report the test results corresponding to this selected epoch. The performance of our method is presented as the mean ± standard deviation over 5 runs with different random seeds. The performance metrics for each baseline are obtained either directly from their original publications or reproduced by us using the best hyperparameters reported in their studies.

Hyperparameters. Particularly for our method, we perform a grid search to find the optimal hyperparameter combination for each dataset whenever feasible. The intersection

features are within [1,2,3,4,5,6]-hops of each node, the batch size is chosen among [32, 64, 128, 256], the number of layers is chosen among [2, 4, 6, 8], the number of heads is chosen among [2, 4, 8, 16, 32], the number of hidden dimensions is chosen among [16, 32, 64, 128], and learning rate is chosen among [0.0001, 0.0003, 0.0005, 0.002]. The chosen optimizer is AdamW. Our model is trained at 200 epochs for all datasets, except for MUV and HIV, where it is trained for 100 epochs. All model training and evaluations were conducted on NVIDIA A100 GPUs with 80G memory.

Dataset Statistics. We provide the statistics of 12 datasets used in our experiments to evaluate the performance of our proposed GIST in Table [1.](#page-5-0)

Table 1. Datasets' Statistics

|                 |          | Table 1.     | Datasets’    | Statistics |                |                 |
|-----------------|----------|--------------|--------------|------------|----------------|-----------------|
| Dataset         | # Graphs | Avg. # nodes | Avg. # edges | Prediction | task           | Metric          |
| BBBP            | 2,050    | 23.9         | 51.6         | binary     | classification | ROC-AUC         |
| Tox21           | 7,831    | 18.6         | 38.6         | 12-task    | classification | ROC-AUC         |
| Toxcast         | 8,597    | 18.7         | 38.4         | 617-task   | classification | ROC-AUC         |
| Sider           | 1,427    | 33.6         | 70.7         | 27-task    | classification | ROC-AUC         |
| Clintox         | 1,484    | 26.1         | 55.5         | 2-task     | classification | ROC-AUC         |
| Bace            | 1513     | 34.1         | 73.7         | binary     | classification | ROC-AUC         |
| MUV             | 93,087   | 24.2         | 52.6         | 17-task    | classification | ROC-AUC         |
| HIV             | 41,127   | 25.5         | 54.9         | binary     | classification | ROC-AUC         |
| Peptides-func   | 15,535   | 150.94       | 307.30       | 10-task    | classification | Avg. Precision  |
| Peptides-struct | 15,535   | 150.94       | 307.30       | 11-task    | regression     | Mean Abs. Error |
| Zinc Subset     | 12,000   | 23.2         | 49.8         |            | regression     | Mean Abs. Error |
| Zinc Full       | 249,456  | 23.2         | 49.8         |            | regression     | Mean Abs. Error |

## 5.2. Long-Range Graph Benchmark (LRGB)

We evaluate the ability of our proposed GIST to learn longrange dependencies using two graph classification datasets from LRGB [\(Dwivedi et al.,](#page-8-1) [2022c\)](#page-8-1): Peptides-func and Peptides-struct. These datasets provide a robust benchmark for assessing graph classification methods in handling longrange dependencies and addressing structural challenges such as over-squashing and over-smoothing of many GNNs. As shown in Table [2,](#page-6-0) GIST significantly enhances the capability of Transformers, achieving state-of-the-art performance on LRGB. This demonstrates that encoding structural information into Transformer-based architectures can mitigate the limitations of existing GNNs in capturing longrange interactions. Regarding RQ2, our results demonstrate that GIST effectively captures long-range dependencies by encoding structural relationships beyond local neighborhoods, leading to improved classification performance.

## 5.3. ZINC and ZINC-full

We further evaluate our proposed GIST on two molecular property prediction datasets: ZINC [\(Dwivedi et al.,](#page-8-0) [2022a\)](#page-8-0) and ZINC-full [\(Irwin et al.,](#page-8-2) [2012\)](#page-8-2). These datasets are widely used benchmarks for assessing the ability of graph-based models to learn molecular representations and predict chemical properties. ZINC, with its constrained molecular structures and well-defined tasks, serves as a standard benchmark

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

Table 2. Performance of GIST on Peptides datasets from LRGB: Top-3 Results Highlighted in Red, Blue, and Orange.

| Model         |               | g        | g    |                |        |      | MAE      | ↓    |        |      | AP ↑     |      |
|---------------|---------------|----------|------|----------------|--------|------|----------|------|--------|------|----------|------|
| GCN (Kipf     | & Welling,    | 2017)    |      |                | 0      | 3496 | ± 0      | 0013 | 0      | 5930 | ± 0      | 0023 |
| GIN (Xu       | et al., 2018) |          |      |                | 0      | 3547 | ± 0      | 0045 | 0      | 5498 | ± 0      | 0079 |
| Subgraphormer | (Bar-Shalom   |          |      | et al., 2024)  | 0      | 2494 | ± 0      | 0020 | 0      | 6415 | ± 0      | 052  |
| FragNet       | (Wollschlager | et       | al., | 2024)          | 0.2462 |      | ± 0.0021 |      | 0.6678 |      | ± 0.0050 |      |
| GatedGCN+RWSE |               | (Dwivedi |      | et al., 2022c) | 0      | 3357 | ± 0      | 0006 | 0      | 6069 | ± 0      | 0035 |
| GRIT (Ma      | et al., 2023) |          |      |                | 0.2460 |      | ± 0.0012 |      | 0.6988 |      | ± 0.0082 |      |
| GraphGPS      | (Rampa´sek    | et ˇ     | al.  | , 2022)        | 0      | 2500 | ± 0      | 0012 | 0      | 6535 | ± 0      | 0041 |
| SAN+LapPE     | (Kreuzer      | et       | al., | 2021)          | 0.2683 |      | ± 0.0043 |      | 0.6384 |      | ± 0.0121 |      |
| SAN+RWSE      | (Kreuzer      | et       | al., | 2021)          | 0      | 2545 | ± 0      | 0012 | 0      | 6439 | ± 0      | 0075 |
| GNN-SSWL+     | (Zhang        | et       | al., | 2023a)         | 0      | 2570 | ± 0      | 006  | 0      | 5847 | ± 0      | 0050 |
| GIST (ours)   |               |          |      |                | 0.2442 |      | ± 0.0011 |      | 0.6783 |      | ± 0.0087 |      |

for evaluating a model's effectiveness in capturing molecular topology and learning chemically relevant features. In contrast, ZINC-full provides a large-scale and more diverse dataset, offering a more rigorous test of a model's generalization capability across a broader range of molecular structures and chemical compositions. As shown in Table [3,](#page-6-1) our approach significantly improves the ability of Transformers to learn molecular graph representations, achieving superior predictive performance. These results demonstrate that incorporating structural priors into Transformer architectures can enhance molecular property prediction, making GIST a promising approach for advancing deep learning methods in computational chemistry and drug discovery.

Table 3. Performance of GIST on ZINC and ZINC-full: Top-3 Results Highlighted in Red, Blue, and Orange.

| Model         |               |             |       |                |             |     | ZINC MAE | ZINC-full ↓ MAE ↓ |
|---------------|---------------|-------------|-------|----------------|-------------|-----|----------|-------------------|
| GCN           | (Kipf &       | Welling,    |       | 2017)          | 0           | 367 | ± 0      | 011 0 113 ± 0 002 |
| GIN           | (Xu et al.,   | 2018)       |       |                | 0           | 526 | ± 0      | 051 0 088 ± 0 002 |
| NGNN          | (Zhang        | & Li,       | 2021) |                | 0           | 111 | ± 0      | 003 0 029 ± 0 001 |
| DS-GNN        | (Bevilacqua   |             | et    | al., 2022)     | 0           | 116 | ± 0      | 009               |
| DSS-GNN       | (Bevilacqua   |             |       | et al., 2022)  | 0           | 102 | ± 0      | 003 0 029 ± 0 003 |
| GNN-AK        | (Zhao         | et al.,     |       | 2022)          | 0           | 105 | ± 0      | 010               |
| GNN-AK+       | (Zhao         | et          | al.,  | 2022)          | 0           | 091 | ± 0      | 002               |
| SUN           | (Frasca et    | al.,        | 2022) |                | 0           | 083 | ± 0      | 003 0 024 ± 0 003 |
| OSAN          | (Qian et      | al.,        | 2022) |                | 0           | 154 | ± 0      | 008               |
| DS-GNN        | (Bevilacqua   |             | et    | al., 2023)     | 0           | 087 | ± 0      | 003               |
| GNN-SSWL      | (Zhang        |             | et    | al., 2023a)    | 0           | 082 | ± 0      | 003 0 026 ± 0 001 |
| GNN-SSWL+     |               | (Zhang      |       | et al., 2023a) | 0.070       |     | ± 0.005  | 0.022 ± 0.001     |
| Subgraphormer |               | (Bar-Shalom |       | et al.,        | 2024) 0.063 |     | ± 0.001  | 0.023 ± 0.001     |
| FragNet       | (Wollschlager |             |       | et al., 2024)  | 0.078       |     | ± 0.005  | 0.024             |
| GatedGCN-LSPE |               | (Dwivedi    |       | et al., 2022c) | 0           | 090 | ± 0      | 001               |
| GRIT          | (Ma et al.,   | 2023)       |       |                | 0.059       |     | ± 0.002  | 0.023 ± 0.001     |
| GraphGPS      | (Rampa´sek    | ˇ           |       | et al. , 2022) | 0           | 070 | ± 0      | 004               |
| SAN           | (Kreuzer      | et al.,     | 2021) |                | 0           | 139 | ± 0      | 006               |
| Graphormer    | (Kreuzer      |             | et    | al., 2021)     | 0           | 122 | ± 0      | 006 0.052 ± 0.005 |
| Graphormer-GD |               | (Kreuzer    |       | et al., 2021)  | 0           | 081 | ± 0      | 009 0.025 ± 0.004 |
| GIST          | (ours)        |             |       |                | 0.055       |     | ± 0.002  | 0.019 ± 0.002     |

## 5.4. MoleculeNet Benchmark

iments to the MoleculeNet benchmark [\(Wu et al.,](#page-9-0) [2017\)](#page-9-0). MoleculeNet encompasses a diverse collection of graphbased molecular property prediction tasks, specifically designed to assess a model's ability to capture chemical interactions, molecular toxicity, and bioactivity. These tasks span a range of real-world applications, including drug discovery, environmental toxicity assessment, and material science, making MoleculeNet a comprehensive benchmark for evaluating graph-based learning approaches. As shown in Table [5,](#page-7-0) GIST consistently outperforms—or at least maintains competitive performance against—existing state-of-the-art pre-trained graph models and Graph Transformers across multiple tasks. These results highlight GIST's strong capability in molecular representation learning, demonstrating that structural information can be effectively integrated into Transformer-based architectures without the need for extensive pretraining, making it a promising approach for molecular property prediction in low-data regimes.

## 5.5. Ablation Study on different k−hop

Finally, to analyze the impact of different k-hop neighborhood sizes in our proposed GIST, we conduct an ablation study on the ZINC dataset. The value of k influences how much local and long-range information is incorporated into the model. For RQ3, results from our ablation study on the ZINC dataset (Table [4\)](#page-6-2) indicate that GIST is robust to variations in the maximum hop distance k. While performance improves as k increases from 1 to 3, capturing richer structural dependencies, the fluctuations beyond k = 3 remain minimal, suggesting that GIST maintains stability across different neighborhood sizes. The slight decrease in performance at higher k is marginal, indicating that GIST effectively balances local expressiveness and global aggregation without being overly sensitive to the choice of k.

Table 4. Ablation study on different k-hop neighborhood sizes in GIST on the ZINC dataset.

| <i>k</i> -hop | 1     | 2     | 3     | 4     | 5     |
|---------------|-------|-------|-------|-------|-------|
| MAE ↓         | 0.100 | 0.058 | 0.054 | 0.065 | 0.063 |

For RQ1, our competitive results in Tables [5,](#page-7-0) [2,](#page-6-0) and [3](#page-6-1) show that GIST effectively facilitates the learning and differentiation of substructures in graph classification tasks by encoding rich structural relationships through intersection cardinality. This enables Graph Transformers to capture fine-grained substructure information and complex substructure relationships, leading to improved performance.

# 6. Related Works

394

396

Table 5. Performance of GIST on MoleculeNet benchmark: Top-3 Results Highlighted in Red, Blue, and Orange.

| Model        |            |            |          |                |      | BBBP  |      | Tox21 |      | Toxcast |      | Sider |      | Clintox |      | Bace  |      | MUV   |      | HIV   | Avg. AUC |
|--------------|------------|------------|----------|----------------|------|-------|------|-------|------|---------|------|-------|------|---------|------|-------|------|-------|------|-------|----------|
| AttrMasking  | (Hu        |            | et al.,  | 2020a)         | 64.3 | ± 2.8 | 76.7 | ± 0.4 | 64.2 | ± 0.5   | 61.0 | ± 0.7 | 71.8 | ± 4.1   | 79.3 | ± 1.6 | 74.7 | ± 1.4 | 77.2 | ± 1.1 | 71.2     |
| GRIT         | (Ma et     | al., 2023) |          |                | 69.9 | ± 1.3 | 75.9 | ± 0.6 | 65.6 | ± 0.4   | 60.3 | ± 1.2 | 85.9 | ± 2.9   | 84.4 | ± 1.2 | 77.1 | ± 1.7 | 77.3 | ± 1.5 | 74.8     |
| GraphGPS     | (Rampa´sek |            | ˇ        | et al. , 2022) | 56.2 | ± 4.4 | 71.4 | ± 0.7 | 60.6 | ± 1.0   | 60.2 | ± 1.1 | 79.2 | ± 3.6   | 71.5 | ± 6.0 | 65.2 | ± 1.6 | 66.0 | ± 9.4 | 66.3     |
| GraphLoG     | (Xu        | et         | al.,     | 2021)          | 67.8 | ± 1.9 | 75.1 | ± 1.0 | 62.4 | ± 0.2   | 59.5 | ± 1.5 | 65.3 | ± 3.2   | 80.2 | ± 3.5 | 73.6 | ± 1.2 | 73.7 | ± 0.9 | 69.7     |
| GraphCL      | (You       | et         | al.,     | 2020)          | 69.7 | ± 0.7 | 73.9 | ± 0.7 | 62.4 | ± 0.6   | 60.5 | ± 0.9 | 76.0 | ± 2.7   | 75.4 | ± 1.4 | 69.8 | ± 2.7 | 78.5 | ± 1.2 | 70.8     |
| G-Motif      | (Rong      | et         | al.,     | 2020)          | 66.9 | ± 3.1 | 73.6 | ± 0.7 | 62.3 | ± 0.6   | 61.0 | ± 1.5 | 77.7 | ± 2.7   | 73.0 | ± 3.3 | 73.0 | ± 1.8 | 73.8 | ± 1.2 | 70.2     |
| G-Contextual |            | (Rong      |          | et al., 2020)  | 69.2 | ± 3.0 | 75.0 | ± 0.6 | 62.8 | ± 0.7   | 58.7 | ± 1.0 | 60.6 | ± 5.2   | 79.3 | ± 1.1 | 72.1 | ± 0.7 | 76.3 | ± 1.5 | 69.3     |
| GPT-GNN      | (Hu        | et         | al.,     | 2020b)         | 64.5 | ± 1.4 | 74.9 | ± 0.3 | 62.5 | ± 0.4   | 58.1 | ± 0.3 | 58.3 | ± 5.2   | 77.9 | ± 3.2 | 75.9 | ± 2.3 | 65.2 | ± 2.1 | 67.2     |
| GraphFP      | (Luong     |            | & Singh, | 2023)          | 72.0 | ± 1.7 | 74.0 | ± 0.7 | 63.9 | ± 0.9   | 63.6 | ± 1.2 | 84.7 | ± 5.8   | 80.5 | ± 1.8 | 75.4 | ± 1.9 | 78.0 | ± 1.5 | 74.0     |
| MGSSL        | (Zhang     | et         | al.,     | 2021)          | 68.9 | ± 2.5 | 74.9 | ± 0.6 | 63.3 | ± 0.5   | 57.7 | ± 0.7 | 67.5 | ± 5.5   | 82.1 | ± 2.7 | 73.2 | ± 1.9 | 75.7 | ± 1.3 | 70.4     |
| GraphMVP     | (Liu       | et         | al.,     | 2022)          | 68.5 | ± 0.2 | 74.5 | ± 0.4 | 62.7 | ± 0.1   | 62.3 | ± 1.6 | 79.0 | ± 2.5   | 76.8 | ± 1.1 | 75.0 | ± 1.4 | 74.8 | ± 1.4 | 71.7     |
| GIST         | (ours)     |            |          |                | 70.6 | ± 1.8 | 77.2 | ± 0.4 | 67.3 | ± 0.9   | 61.3 | ± 2.7 | 88.2 | ± 2.2   | 86.0 | ± 1.9 | 75.5 | ± 3.2 | 77.0 | ± 0.2 | 75.4     |

terns and improving representation learning in graph-based tasks. However, GNNs remain fundamentally constrained by their reliance on localized message passing, which limits their ability to capture long-range dependencies and effectively model complex substructure interactions, due to over-smoothing and over-squashing issues [\(Xu et al.,](#page-9-10) [2018;](#page-9-10) [Alon & Yahav,](#page-8-17) [2021\)](#page-8-17). To address this, later works have introduced spectral features [\(Balcilar et al.,](#page-8-18) [2021\)](#page-8-18), motif-based methods [\(Rong et al.,](#page-9-20) [2020;](#page-9-20) [Zhang et al.,](#page-9-16) [2021;](#page-9-16) [Bar-Shalom et al.,](#page-8-10) [2024;](#page-8-10) [Wollschlager et al.,](#page-9-6) [2024\)](#page-9-6), and Weisfeiler-Lehman (WL) kernel-based approaches [\(Morris](#page-9-21) [et al.,](#page-9-21) [2019\)](#page-9-21) to improve graph representation learning by explicitly capturing local and global structural patterns. While motif-based methods improve expressivity by incorporating recurring substructures, they often depend on predefined motifs, restricting their adaptability to unseen graph patterns. Similarly, WL kernel-based approaches enhance structural discrimination but struggle with distinguishing graphs that are structurally different yet WL-equivalent. Furthermore, spectral features capture global graph properties but introduce additional computational complexity, making them less practical for large-scale applications. These limitations underscore the need for alternative architectures that can more effectively integrate structural biases while maintaining both scalability and expressiveness in graph learning.

Graph Transformers. Transformers have demonstrated remarkable success in natural language processing and computer vision by leveraging self-attention to model long-range dependencies effectively [\(Vaswani et al.,](#page-9-2) [2017\)](#page-9-2). More recently, their adaptation to graph-structured data has led to the emergence of Graph Transformers, where self-attention replaces traditional message-passing mechanisms to enable more flexible and expressive learning [\(Zhang et al.,](#page-9-22) [2020;](#page-9-22) [Dwivedi & Bresson,](#page-8-19) [2021\)](#page-8-19). However, a fundamental challenge in applying Transformers to graphs is the absence of a natural node ordering, making it difficult to encode structural information directly. To address this, positional encodings have been introduced to assign meaningful node representations within the graph topology. Among these, Laplacian eigenvector-based encodings (LapPE) [\(Dwivedi et al.,](#page-8-0) [2022a\)](#page-8-0) and random walk positional encodings (RWPE) [\(Dwivedi et al.,](#page-8-20) [2022b\)](#page-8-20) inject global structural awareness, enhancing the model's ability to differentiate nodes with similar local neighborhoods. Beyond positional encodings, researchers have explored incorporating structural biases into self-attention to ensure that Graph Transformers respect the underlying graph topology. GPS [\(Rampa´sek et al.](#page-9-4) ˇ , [2022\)](#page-9-4) combines message passing with attention, allowing models to capture both local and global dependencies within the graph. More recently, GRIT [\(Ma et al.,](#page-9-5) [2023\)](#page-9-5) introduced a fully Transformer-based framework that eliminates explicit message passing while embedding structure-aware attention, achieving state-of-the-art performance across multiple graph learning benchmarks. These advancements reflect a growing shift toward pure Transformer architectures that effectively incorporate graph-specific inductive biases, paving the way for more scalable and expressive models in graph representation learning.

# 7. Conclusion

This paper introduces the Graph Invariant Structural Trait (GIST) to enhance Graph Transformers by improving their ability to encode graph structures. GIST estimates pairwise node intersections to capture substructures within a graph, integrating this information into the attention mechanism. This refinement enables Graph Transformers to better represent structural relationships that traditional all-to-all attention struggles to capture. Theoretical analysis and empirical results confirm that GIST effectively preserves essential structural information critical for graph classification. Extensive experiments across multiple datasets demonstrate that incorporating GIST into Graph Transformers consistently improves performance over state-of-the-art methods. These findings highlight the importance of structural encoding in enhancing Graph Transformers, contributing to more robust and interpretable graph-based learning models across scientific domains.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. References Alon, U. and Yahav, E. On the bottleneck of graph neural networks and its practical implications. In *The Tenth International Conference on Learning Representations*, 2021. Balcilar, M., Heroux, P., Ga ´ uz¨ ere, B., Vasseur, P., Adam, ` S., and Honeine, P. Breaking the limits of message passing graph neural networks. In *The 38th International Conference on Machine Learning*, 2021. Bar-Shalom, G., Bevilacqua, B., and Maron, H. Subgraphormer: Unifying subgraph gnns and graph transformers via graph products. In *The Forty-first International Conference on Machine Learning*, 2024. Bevilacqua, B., Frasca, F., Lim, D., Srinivasan, B., Cai, C., Balamurugan, G., Bronstein, M. M., and Maron, H. Equivariant subgraph aggregation networks. In *International Conference on Learning Representations (ICLR)*, 2022. Bevilacqua, B., Eliasof, M., Meirom, E., Ribeiro, B., and Maron, H. Efficient subgraph gnns by learning effective selection policies. In *International Conference on Learning Representations (ICLR)*, 2023. Black, M., Wan, Z., Nayyeri, A., and Wang, Y. Understanding oversquashing in gnns through the lens of effective resistance. In *International Conference on Machine Learning*, pp. 2528–2547. PMLR, 2023. Brody, S., Alon, U., and Yahav, E. How attentive are graph attention networks? In *The Eleventh International Conference on Learning Representations*, 2022. Chamberlain, B. P., Shirobokov, S., Rossi, E., Frasca, F., Markovich, T., Hammerla, N. Y., Bronstein, M. M., and Hansmire, M. Graph neural networks for link prediction with subgraph sketching. In *The Eleventh International Conference on Learning Representations*, 2022. Dwivedi, V. P. and Bresson, X. A generalization of transformer networks to graphs. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2021. Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., and Bresson, X. Benchmarking graph neural networks. In *Journal of Machine Learning Research*, 2022a. Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., and Bresson, X. Graph neural networks with learnable structural and positional representations. In *The Eleventh International Conference on Learning Representations*, 2022b. Dwivedi, V. P., Rampa´sek, L., Galkin, M., Parviz, A., Wolf, ˇ G., Luu, A. T., and Beaini, D. Recipe for a general, powerful, scalable graph transformer. In *36th Conference on Neural Information Processing Systems*, 2022c. Frasca, F., Bevilacqua, B., Bronstein, M., and Maron, H. Understanding and extending subgraph gnns by rethinking their symmetries. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 35, pp. 31376–31390, 2022. Han, X., Jiang, Z., Liu, N., and Hu, X. G-mixup: Graph data augmentation for graph classification. In *International Conference on Machine Learning*, pp. 8230–8248. PMLR, 2022. Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., and Leskovec, J. Strategies for pre-training graph neural networks. In *International Conference on Learning Representations (ICLR)*, 2020a. Hu, Z., Dong, Y., Wang, K., Chang, K.-W., and Sun, Y. Gptgnn: Generative pre-training of graph neural networks. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)*, 2020b. Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., and Coleman, R. G. Zinc: a free tool to discover chemistry for biology. In *Journal of Chemical Information and Modeling*, 2012. Keriven, N. Not too little, not too much: a theoretical analysis of graph (over) smoothing. *Advances in Neural Information Processing Systems*, 35:2268–2281, 2022. Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. In *5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings*. OpenReview.net, 2017. URL [https://](https://openreview.net/forum?id=SJU4ayYgl) [openreview.net/forum?id=SJU4ayYgl](https://openreview.net/forum?id=SJU4ayYgl). Kreuzer, D., Beaini, D., Hamilton, W. L., Letourneau, V., and Tossou, P. Rethinking graph transformers with spectral attention. In *35th Conference on Neural Information Processing Systems*, 2021. Le, D., Zhong, S. H., Liu, Z., Xu, S., Chaudhary, V., Zhou, K., and Xu, Z. Knowledge graphs can be learned with just intersection features. In *The Forty-first International Conference on Machine Learning*, 2024.

494

504

506

508 509

511

514 515 516

518

524

526

528

531

534

536

538

- Liu, S., Wang, H., Liu, W., Lasenby, J., Guo, H., and Tang,
- J. Pre-training molecular graph representation with 3d geometry. In *The Eleventh International Conference on Learning Representations*, 2022. Luong, K.-D. and Singh, A. Fragment-based pretraining and finetuning on molecular graphs. In *37th Conference on Neural Information Processing Systems*, 2023. Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania,
- P. K., Coates, M., Torr, P. H., and Lim, S.-N. Graph inductive biases in transformers without message passing. In *The Fortieth International Conference on Machine Learning*, 2023. Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen,
- J. E., Rattan, G., and Grohe, M. Weisfeiler and leman go neural: Higher-order graph neural networks. In *Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI)*, pp. 4602–4609, 2019. URL <https://arxiv.org/abs/1810.02244>. Qian, C., Rattan, G., Geerts, F., Niepert, M., and Morris,
- C. Ordered subgraph aggregation networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 35, pp. 21030–21045, 2022. Rampa´sek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, ˇ G., and Beaini, D. Recipe for a general, powerful, scalable graph transformer. In *36th Conference on Neural Information Processing Systems*, 2022. Rong, Y., Bian, Y., Xu, T., Xie, W., Wei, Y., Huang, W., and HUang, J. Self-supervised graph transformer on large-scale molecular data. In *34th Conference on Neural Information Processing Systems*, 2020. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In *31th Conference on Neural Information Processing Systems*, 2017. Wang, Y. and Zhang, M. An empirical study of realized gnn expressiveness. In *Forty-first International Conference on Machine Learning*, 2024. Wollschlager, T., Kemper, N., Hetzel, L., Sommer, J., and Gunneman, S. Expressivity and generalization: Fragmentbiases for molecular gnns. In *The Forty-first International Conference on Machine Learning*, 2024. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., and Pande, V. Moleculenet: A benchmark for molecular machine learning. In *Chemical Science*, 2017. Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In *The Seventh International Conference on Learning Representations*, 2018. Xu, M., Wang, H., Ni, B., Guo, m. H., and Tang, J. Selfsupervised graph-level representation learning with local and global structure. In *The 38th International Conference on Machine Learning*, 2021. Yang, C., Liu, M., Zheng, V. W., and Han, J. Node, motif and subgraph: Leveraging network functional blocks through structural convolution. In *International Conference on Advances in Social Network Analysis and Mining*, 2018. Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., and Liu, T.-Y. Do transformers really perform bad for graph representation? In *35th Conference on Neural Information Processing Systems*, 2021. You, Y., Chen, T., Shen, Y., and Wang, Z. Graph contrastive learning with augmentations. In *34th Conference on Neural Information Processing Systems*, 2020. Yu, Z. and Gao, H. Molecular representation learning via heterogeneous motif graph neural networks. In *Proceedings of the 39th International Conference on Machine Learning*, 2022. Zhang, B., Feng, G., Du, Y., He, D., and Wang, L. A complete expressiveness hierarchy for subgraph gnns via subgraph weisfeiler-lehman tests. In *International Conference on Machine Learning (ICML)*, 2023a. Zhang, B., Luo, S., Wang, L., and He, D. Rethinking the expressive power of gnns via graph biconnectivity. In *The Twelfth International Conference on Learning Representations*, 2023b. Zhang, M. and Li, P. Nested graph neural networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 34, 2021. Zhang, Z., Cui, P., and Zhu, W. Graph-bert: Only attention is needed for learning graph representations. *arXiv preprint arXiv:2001.05140*, 2020. Zhang, Z., Liu, Q., Wang, H., Lu, C., and Lee, C.-K. Motifbased graph self-supervised learning for molecular property prediction. In *35th Conference on Neural Information Processing Systems*, 2021. Zhao, L., Jin, W., Akoglu, L., and Shah, N. From stars to subgraphs: Uplifting any gnn with local structure awareness. In *International Conference on Learning Representations (ICLR)*, 2022.

 Theorem A.1 (Formal version of Theorem [4.2\)](#page-4-3). *Let* G = (V, E) *denote a graph with* n *nodes (*|V| = n*). Let* S(G) ∈ *denote the* k*-hop GIST (see Definition [3.2\)](#page-3-1) computed on* G*. We show that the GIST attention* ψ(xu) *for every node* u ∈ V *(see Definition [4.1\)](#page-4-2) is invariant under graph isomorphism.*

*Proof.* Let f denote isomorphic transform on nodes V such that if u and v are adjacent in G, f(u) and f(v) are also adjacent. Without loss of generally, we see that C<sup>k</sup>u,k<sup>v</sup> (f(u), f(v)) = C<sup>k</sup>u,k<sup>v</sup> (u, v).

## A. Proofs

Following Definition [3.1,](#page-3-0) we show that I<sup>k</sup>u,k<sup>v</sup> (f(u), f(v)) = I<sup>k</sup>u,k<sup>v</sup> (u, v), T<sup>k</sup>u,k<sup>v</sup> (f(u), f(v)) = T<sup>k</sup>u,k<sup>v</sup> (u, v).

As a result, we show that Sf(u),f(v)(f(G)) = Su,v(f(G)).

Following Definition [4.1,](#page-4-2) since the order of node v does not affect the computation of ψ(xu), we show that ψ(xf(u)) = ψ(xu).

As a result, we show that the isomorphic transform f does not change ψ(xu), making ψ a graph invariant.