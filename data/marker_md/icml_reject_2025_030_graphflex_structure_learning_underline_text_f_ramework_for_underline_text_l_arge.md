011

014 015 016

018

024

026

034

036

038

054

# GraphFLEx: Structure Learning Framework for Large Expanding Graphs

#### Anonymous Authors<sup>1</sup>

## Abstract

Graph structure learning is a fundamental problem critical for interpretability and uncovering relationships in data. While graphical data is central to information representation, inferring graph structures remains challenging. Existing methods falter with expanding graphs, requiring costly relearning of the entire structure for new nodes, and face severe computational and memory demands on large graphs. To overcome these challenges, we propose GraphFLEx: a unified framework for structure learning in Large and Expanding Graphs. GraphFLEx efficiently limits potential connections to relevant nodes by leveraging clustering and coarsening techniques, significantly reducing computational costs and enhancing scalability. GraphFLEx provides 48 flexible methods for graph structure learning by integrating diverse learning, coarsening, and clustering approaches. Extensive experiments with various GNN models demonstrate its effectiveness. Our code is available [here.](https://anonymous.4open.science/r/Scaling_Graph_Learning-5644)

## 1. Introduction

Graph representations capture relationships between entities, vital across diverse fields like biology, finance, sociology, engineering, and operations research [\(Zhou et al.,](#page-10-0) [2020;](#page-10-0) [Fout et al.,](#page-8-0) [2017;](#page-8-0) [Wu et al.,](#page-10-1) [2020\)](#page-10-1). While some relationships, such as social connections or sensor networks, are directly observable, many, including gene regulatory networks, scene graph generation [\(Gu et al.,](#page-8-1) [2019\)](#page-8-1), brain networks, [\(Zhu et al.,](#page-11-0) [2021\)](#page-11-0) and drug interactions, require inference [\(Allen et al.,](#page-8-2) [2012\)](#page-8-2). Even when available, graph data often contains noise, requiring denoising and recalibration. Thus, inferring graph structures becomes crucial, often surpassing the choice of graph or algorithm itself. *Graph Structure Learning (GSL)* offers a solution, enabling the construction and refinement of graph topologies. GSL has been widely studied in both supervised and unsupervised

contexts [\(Liu et al.,](#page-9-0) [2022;](#page-9-0) [Chen & Wu,](#page-8-3) [2022\)](#page-8-3). In supervised GSL (s-SGL), the adjacency matrix and Graph Neural Networks (GNNs) are jointly optimized for a downstream task, such as node classification. Notable examples of s-GSL include NodeF ormer [\(Wu et al.,](#page-10-2) [2022\)](#page-10-2), P ro−GNN [\(Jin](#page-9-1) [et al.,](#page-9-1) [2020\)](#page-9-1), W SGNN [\(Lao et al.,](#page-9-2) [2022\)](#page-9-2), and SLAP S [\(Fatemi et al.,](#page-8-4) [2021\)](#page-8-4). Unsupervised GSL (u-SGL), on the other hand, focuses solely on learning the underlying graph structure, typically through adjacency or Laplacian matrices. Methods in this category include approximate nearest neighbours (A−NN) [\(Dong et al.,](#page-8-5) [2011;](#page-8-5) [Muja & Lowe,](#page-9-3) [2014\)](#page-9-3), knearest neighbours (k−NN) [\(MacQueen et al.,](#page-9-4) [1967;](#page-9-4) [Wang](#page-10-3) [& Zhang,](#page-10-3) [2006\)](#page-10-3), covariance estimation (emp.Cov.) [\(Hsieh](#page-8-6) [et al.,](#page-8-6) [2011\)](#page-8-6), graphical lasso (GLasso) [\(Friedman et al.,](#page-8-7) [2008\)](#page-8-7), and signal processing techniques like l2-model,logmodel, and large-model [\(Dong et al.,](#page-8-8) [2016;](#page-8-8) [Kalofolias,](#page-9-5) [2016\)](#page-9-5).

While s-SGL methods offer promising results, they have limitations: (1) they rely on label information, restricting their applicability in settings without annotations; (2) they are often task-specific, optimizing for node classification rather than general graph topology [\(Liu et al.,](#page-9-0) [2022\)](#page-9-0). These issues are avoided in u-SGL approaches, which are the focus of this work. However, both s-SGL and u-SGL face challenges when applied to large-scale or expanding datasets.

![](_page_0_Figure_8.jpeg)

Figure 1: High computational time required to learn graph structures using existing methods, whereas GraphFLEx effectively controls computational growth, achieving near-linear scalability. Notably, Vanilla KNN failed to construct graph structures with fewer than 10k nodes due to memory limitations.

As contemporary datasets grow in size, scalability becomes a critical challenge, with existing methods proving too computationally expensive for large-scale graphs. In such cases, Approximate Nearest Neighbours (A−NN), with time com-

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

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

108 109

![](_page_1_Diagram_1.jpeg)

Figure 2: General pipeline of GraphFLEx, it processes a graph (Gt−1) and incoming nodes (Et) at time t, comprising three main components: a) Clustering, which infers E<sup>t</sup> nodes to existing communities using a pre-trained model Mclust(G0); b) Coarsening, reduces the size of the desired community; and c) Learning, where the structure associated with E<sup>t</sup> nodes are learned using the coarsened graph, followed by projecting this structure onto the original graph to create graph G<sup>t</sup> at time t.

plexity O(N log(N)), is often the only feasible solution. In contrast, methods like k-NN, log-model, and l2-model are significantly more costly, with time complexities exceeding O(N<sup>2</sup> ).

The aforementioned techniques are ineffective for learning large-scale graphs because they consider the entire collection of nodes to determine connections for every individual node. All nodes, however, only have connections to a very small set of nodes. Therefore, we need to devise a method that can refine the entire graph's node set to a smaller subset of potential node sets, with the aim of identifying feasible connections. Additionally, expanding graphs where new nodes continuously arrive further complicates the issue, as existing methods require re-learning the entire graph structure with each new node [\(Khazane et al.,](#page-9-6) [2019;](#page-9-6) [Holme &](#page-8-9) [Saramaki](#page-8-9) ¨ , [2012\)](#page-8-9). This makes them inefficient for expanding data. To address these challenges, we propose GraphFLEx, a comprehensive framework that tackles both scalability for large datasets and adaptability for growing graphs.

As shown in Figure [2,](#page-1-0) GraphFLEx comprises three key modules: (i) Graph Clustering, (ii) Graph Coarsening, and (iii) Graph Learning. By leveraging clustering and coarsening, GraphFLEx significantly reduces computational overhead by restricting possible connections to only relevant nodes. Figure [1](#page-0-0) compares the graph structure learning time, highlighting GraphFLEx's efficiency over existing methods. Key contributions of GraphFLEx include:

Key Contributions and Novelty.

- We provide strong *theoretical guarantees* that the structure learned from a small subset of nodes is equivalent to that learned from the full set. This is supported by empirical results using real-world and synthetic datasets, demonstrating the effectiveness of GraphFLEx across diverse graph structures.
- GraphFLEx is composed of independently operating modules, allowing the creation of new learning frameworks by modifying any of its three modules. It currently supports *48 distinct methods* for learning graph structure, offering flexibility across various domains.
- GraphFLEx efficiently handles *large-scale and expanding graphs*, enhancing scalability for graph learning tasks.
- GraphFLEx serves as a *comprehensive framework* applicable individually for clustering, coarsening, and learning tasks.

# 2. Problem Formulation and Background

A graph G is represented using G(V, A, X) where V = {v1, v2...v<sup>N</sup> } is the set of N nodes, each node v<sup>i</sup> has a d−dimensional feature vector x<sup>i</sup> in X ∈ R N×d and A ∈ R <sup>N</sup>×<sup>N</sup> is adjacency matrix representing connection between i th and j th nodes when entry Aij > 0. An expanding graph E<sup>G</sup> can be considered a variant of graph G where nodes v now have an associated timestamp τv. We can represent a expanding graph as a sequence of graphs, i.e., E<sup>G</sup> = {G0, G1, ...G<sup>T</sup> } where {G<sup>0</sup> ⊆ G1.... ⊆ G<sup>T</sup> } at

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

τ ∈ {0, ...T} timestamps. New nodes arriving at different timestamps are seamlessly integrating into initial graph G0.

*Problem statement.* Given a partially known or missing graph structure, our goal is to incrementally learn the whole graph, i.e., learn adjacency or laplacian matrix. Specifically, we consider two unsupervised GSL tasks:

Goal 1. *Large Datasets with Missing Graph Structure: In this setting, the graph structure is entirely unavailable, and existing methods are computationally infeasible for learning the whole graph in a single step. To address this issue, we first randomly partition the dataset into exclusive subsets. We then learn the initial graph* G0(V0, X0) *over a small subset of nodes and incrementally expand it by integrating additional partitions, ultimately reconstructing the full graph* G<sup>T</sup> *.*

Goal 2. *Partially Available Graph: In this case, we only have access to the graph* G<sup>t</sup> *at timestamp* t*, with new nodes arriving over time. The goal is to update the graph incrementally to obtain* G<sup>T</sup> *, without re-learning it from scratch at each timestamp.*

GraphFlex addresses these challenges with a unified framework, outlined in Section [3.](#page-2-0) Before delving into the framework, we review some key concepts.

#### 2.1. Graph Reduction

Graph reduction encompasses sparsification, clustering, coarsening, and condensation [\(Hashemi et al.,](#page-8-10) [2024\)](#page-8-10). Graph-Flex employs clustering and coarsening to refine the set of relevant nodes for potential connections.

Graph Clustering. Graphs often exhibit global heterogeneity with localized homogeneity, making them well-suited for clustering [\(Fortunato,](#page-8-11) [2010\)](#page-8-11). Clusters capture higher-order structures, aiding graph learning. Methods like DMoN [\(Tsit](#page-10-4)[sulin et al.,](#page-10-4) [2023\)](#page-10-4) use GNNs for soft cluster assignments, while Spectral Clustering (SC) [\(Kamvar et al.,](#page-9-7) [2003\)](#page-9-7) and K-means [\(Wagstaff et al.,](#page-10-5) [2001;](#page-10-5) [MacQueen et al.,](#page-9-4) [1967\)](#page-9-4) efficiently detect communities. DiffPool [\(Bruna et al.,](#page-8-12) [2014;](#page-8-12) [Defferrard et al.,](#page-8-13) [2016\)](#page-8-13) applies SC for pooling in GNNs.

Graph Coarsening. Graph Coarsening (GC) reduces a graph G(V, E, X) with N nodes and features X ∈ <sup>R</sup> N×d into a smaller graph Gc(V , e E, e <sup>X</sup>e) with <sup>n</sup> ≪ <sup>N</sup> nodes and <sup>X</sup>e ∈ <sup>R</sup> n×d . This is achieved via learning a coarsening matrix P ∈ R <sup>n</sup>×<sup>N</sup> , mapping similar nodes in G to super-nodes in Gc, ensuring <sup>X</sup>e <sup>=</sup> P<sup>X</sup> while preserving key properties [\(Loukas,](#page-9-8) [2019;](#page-9-8) [Kataria et al.,](#page-9-9) [2023;](#page-9-9) [Kumar et al.,](#page-9-10) [2023;](#page-9-10) [Kataria et al.,](#page-9-11) [2024\)](#page-9-11).

#### 2.2. Unsupervised Graph Structure Learning

Unsupervised graph learning spans from simple k-NN weighting [\(Wang & Zhang,](#page-10-3) [2006;](#page-10-3) [Zhu et al.,](#page-10-6) [2003\)](#page-10-6) to advanced statistical and graph signal processing (GSP) tech-

Table 1: Unsupervised Graph Structure Learning Methods

| Method Time Complexity           |      |      | Formulation |     |             |
|----------------------------------|------|------|-------------|-----|-------------|
| GLasso O ( N                     |      |      |             |     |             |
| )                                | max  | Θ    | log         | det | Θ           |
| −                                | tr ˆ | (ΣΘ) | −           | ρ   | ∥ Θ ∥ 1     |
| log -model O ( N                 |      |      |             |     |             |
| ) min                            | W    | ∈W   | ∥           | W   | ◦ Z ∥ 1 , 1 |
| − α 1                            |      |      |             |     |             |
|                                  | log( | W    | 1           | ) + | β           |
|                                  |      |      |             |     | ∥ W ∥       |
| l 2 -model O ( N                 |      |      |             |     |             |
| ) min                            | W    | ∈W   | ∥           | W   | ◦ Z ∥ 1 , 1 |
| + α                              | ∥ W  | 1 ∥  |             |     |             |
|                                  |      |      | 2           | + α | ∥ W ∥       |
| +                                | 1 {∥ | W    | ∥ 1         | , 1 | = n }       |
| large -model O ( N log( N )) min | W ∈  | W ˜  | ∥           | W   | ◦ Z ∥ 1 , 1 |
| − α 1                            |      |      |             |     |             |
|                                  | log( | W    | 1           | ) + | β           |
|                                  |      |      |             |     | ∥ W ∥       |

niques. Statistical methods, also known as probabilistic graphical models, assume an underlying graph G governs the joint distribution of data X ∈ R N×d [\(Koller & Fried](#page-9-12)[man,](#page-9-12) [2009;](#page-9-12) [Banerjee et al.,](#page-8-14) [2008;](#page-8-14) [Friedman et al.,](#page-8-7) [2008\)](#page-8-7). Some approaches [\(Dempster,](#page-8-15) [1972\)](#page-8-15) prune elements in the inverse sample covariance matrix Σ =b <sup>1</sup> d−1XX<sup>T</sup> and sparse inverse covariance estimators, such as Graphical Lasso (GLasso) [\(Friedman et al.,](#page-8-7) [2008\)](#page-8-7): maximize<sup>Θ</sup> log det Θ − tr(ΣΘb )−ρ∥Θ∥1, where <sup>Θ</sup> is the inverse covariance matrix. However, these methods struggle with small sample sizes. Graph Signal Processing (GSP) techniques analyze signals on known graphs, ensuring properties like smoothness and sparsity. Signal smoothness on a graph G is quantified by the Laplacian quadratic form:

$$Q(\mathbf{L}) = \mathbf{x}^T \mathbf{L} \mathbf{x} = \frac{1}{2} \sum_{i,j} w_{ij} (\mathbf{x}(i) - \mathbf{x}(j))^2.$$

For a set of vectors X, smoothness is measured using the Dirichlet energy [\(Belkin et al.,](#page-8-16) [2006\)](#page-8-16): tr(X<sup>T</sup>LX). Stateof-the-art methods [\(Dong et al.,](#page-8-8) [2016;](#page-8-8) [Kalofolias,](#page-9-5) [2016;](#page-9-5) [Hu et al.,](#page-9-13) [2013\)](#page-9-13) optimize Dirichlet energy while enforcing sparsity or specific structural constraints. Table [1](#page-2-1) compares various graph learning methods based on their formulations and time complexities.

*Remark* 1*.* Graph Structure Learning (GSL) differs significantly from Continual Learning (CL) [\(Van de Ven & Tolias,](#page-10-7) [2019;](#page-10-7) [Zhang et al.,](#page-10-8) [2022;](#page-10-8) [Parisi et al.,](#page-9-14) [2019\)](#page-9-14) and Dynamic Graph Learning (DGL) [\(Kim et al.,](#page-9-15) [2022;](#page-9-15) [Wu et al.,](#page-10-9) [2023;](#page-10-9) [You et al.,](#page-10-10) [2022\)](#page-10-10), as discussed in Appendix [C.](#page-13-0)

## 3. GraphFLEx

In this section, we introduce GraphFLEx, which has three main modules:

- Graph Clustering. Identifies communities and extracts higher-order structural information,
- Graph Coarsening. Is used to coarsen down the desired community, if the community itself is large,
- Graph Learning. Learns the graph's structure using a

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

limited subset of nodes from the clustering and coarsening modules, *enabling scalability*.

For more details, see Algorithm [1](#page-15-0) in Appendix [E.](#page-14-0)

#### 3.1. Incremental Graph Learning for Large Datasets

Real-world graph data is continuously expanding. For instance, e-commerce networks accumulate new clicks and purchases daily [\(Xiang et al.,](#page-10-11) [2010\)](#page-10-11), while academic networks grow with new researchers and publications [\(Wang](#page-10-12) [et al.,](#page-10-12) [2020\)](#page-10-12). This expanding behaviour suggests that large graphs can be efficiently processed by learning them incrementally in smaller segments.

Given a large dataset L(VL, XL), where V<sup>L</sup> is the node set and X<sup>L</sup> represents node features, we define an *expanding dataset* setting L<sup>E</sup> = {E<sup>T</sup> <sup>τ</sup>=0}. Initially, L is split into: (i) a *static dataset* E0(V0, X0) and (ii) an *expanding dataset* E = {E<sup>τ</sup> (V<sup>τ</sup> , X<sup>τ</sup> )} T <sup>τ</sup>=1. Both *Goal [1](#page-2-2)* (large datasets with missing graph structure) and *Goal [2](#page-2-3)* (partially available graphs with incremental updates), discussed in Section [2,](#page-1-1) share the common objective of incrementally learning and updating the graph structure as new data arrives. Graph-FLEx handles these by decomposing the problem into two key components:

- Initial Graph G0(V0, A0, X0): For *Goal [1](#page-2-2)*, where the graph structure is entirely missing, E0(V0, X0) is used to construct G<sup>0</sup> from scratch using structure learning methods (see Section [2.2\)](#page-2-4). For *Goal [2](#page-2-3)*, the initial graph G0(V0, A0, X0) is already available and serves as the starting point for incremental updates.
- Expanding Dataset E = {E<sup>τ</sup> (V<sup>τ</sup> , X<sup>τ</sup> )} T <sup>τ</sup>=1: In both cases, E consists of incoming nodes and features arriving over T timestamps. These nodes are progressively integrated into the existing graph, enabling continuous adaptation and growth.

The partition is controlled by a parameter r, which determines the proportion of static nodes: r = ∥V0∥ ∥VL∥ . For example, r = 0.2 implies that 20% of V<sup>L</sup> is treated as static, while the remaining 80% arrives incrementally over T timestamps. In our experiments, we set r = 0.5 and T = 25.

*Remark* 2*.* We can learn G<sup>τ</sup> (V<sup>τ</sup> , A<sup>τ</sup> , X<sup>τ</sup> ) by aggregating E<sup>τ</sup> nodes in Gτ−<sup>1</sup> graph. Our goal is to learn G<sup>T</sup> (V<sup>T</sup> , A<sup>T</sup> , X<sup>T</sup> ) after T th-timestamp.

### 3.2. Detecting Communities

From the static graph G0, our goal is to learn higher-order structural information, identifying potential communities to which incoming nodes (V ∈ V τ ) may belong. We train the community detection/clustering model Mclust once using G0, allowing subsequent inference of clusters for all incoming nodes. While our framework supports spectral and k-means clustering, our primary focus has been on Graph Neural Network (GNN)-based clustering methods. Specifically, we use DMoN [\(Tsitsulin et al.,](#page-10-4) [2023;](#page-10-4) [Bianchi](#page-8-17) [et al.,](#page-8-17) [2020;](#page-8-17) [Bianchi,](#page-8-18) [2022\)](#page-8-18), which maximizes spectral modularity. Modularity [\(Newman,](#page-9-16) [2006\)](#page-9-16) measures the divergence between intra-cluster edges and the expected number. These methods use a GNN layer to compute the partition matrix <sup>C</sup> <sup>=</sup> softmax(MLP(X, θ e MLP)) ∈ <sup>R</sup> <sup>N</sup>×<sup>K</sup>, where <sup>K</sup> is the number of clusters and <sup>X</sup>e is the updated feature embedding generated by one or more message-passing layers. To optimize the C matrix, we minimize the loss function ∆(C; A) = − 1 <sup>2</sup><sup>m</sup> Tr(C <sup>T</sup> BC) + √ k n |ΣiC T i |<sup>F</sup> − 1, which combines spectral modularity maximization with regularization to prevent trivial solutions, where B is the modularity matrix [\(Tsitsulin et al.,](#page-10-4) [2023\)](#page-10-4). Our static graph G<sup>0</sup> and incoming nodes E follow Assumption [1.](#page-3-0)

Assumption 1. *We assume that the generated graphs adhere to the Degree-Corrected Stochastic Block Model (DC-SBM) [\(Zhao et al.,](#page-10-13) [2012\)](#page-10-13), where intra-class (or intracommunity) links are more likely than inter-class links.*

For more details on DC-SBM, see Appendix [A.](#page-12-0)

Lemma 1. M*clust Consistency. We adopt the theoretical framework of [\(Zhao et al.,](#page-10-13) [2012\)](#page-10-13) for a DC-SBM with* N *nodes and* k *classes. The edge probability matrix is parameterized as* P<sup>N</sup> = ρ<sup>N</sup> P*, where* P ∈ <sup>R</sup> k×k *is a symmetric matrix containing the between/within community edge probabilities and it is independent of* N*,* ρ<sup>N</sup> = λ<sup>N</sup> /N*, and* λ<sup>N</sup> *is the average degree of the network. Let* yˆ<sup>N</sup> = [ˆy1, yˆ2, . . . , yˆ<sup>N</sup> ] *denote the predicted class labels, and let* Cˆ<sup>N</sup> *be the corresponding* N × k *one-hot matrix. Let the true class label matrix is* C<sup>N</sup> *, and* µ *is any* k × k *permutation matrix. Under the adjacency matrix* A(N) *, the global maximum of the objective* ∆(·; A(N) ) *is denoted as* Cˆ<sup>∗</sup> <sup>N</sup> *. The consistency of class predictions is defined as:*

- *1. Strong Consistency.* P<sup>N</sup> min µ ∥Cˆ<sup>∗</sup> <sup>N</sup> µ − C<sup>N</sup> ∥ 2 <sup>F</sup> = 0 → 1 *as* N → ∞,
- *2. Weak Consistency.* ∀ε > 0, P<sup>N</sup> min µ 1 N ∥Cˆ<sup>∗</sup> <sup>N</sup> µ − C<sup>N</sup> ∥ <sup>F</sup> < ε → 1 *as* N → ∞.

*where* ∥ · ∥<sup>F</sup> *is the Frobenius norm. Under the conditions of Theorem 3.1 from [\(Zhao et al.,](#page-10-13) [2012\)](#page-10-13):*

- *The* M*clust objective is strongly consistent if* λ<sup>N</sup> / log(N) → ∞*, and*
- *It is weakly consistent when* λ<sup>N</sup> → ∞*.*

*Remark* 3*.* Structure Learning within Communities. In GraphF LEx, we focus on learning the structure within each community rather than the structure of the entire dataset at once. Strong consistency ensures perfect community recovery, meaning no inter-community edges exist

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

representing the ideal case. Weak consistency, however, allows for a small fraction (ϵ) of inter-community edges, where ϵ is controlled by ρ<sup>n</sup> in P<sup>n</sup> = ρnP, influencing graph sparsity.

By Lemma [1](#page-3-1) and Assumption [1,](#page-3-0) stronger consistency leads to more precise structure learning, whereas weaker consistency permits a limited number of inter-community edges.

#### 3.3. Learning Graph Structure on a Coarse Graph

After training Mclust, we identify communities for incoming nodes, starting with τ = 1. Once assigned, we determine significant communities those with at least one incoming node and learn their connections to the respective community subgraphs. For large datasets, substantial community sizes may again introduce scalability issues. To mitigate this, we first coarsen the large community graph into a smaller graph and use it to identify potential connections for incoming nodes. This process constitutes the second module of GraphFLEx, denoted as Mcoar, which employs LSH-based hashing for graph coarsening. The supernode index for i th node is given as:

$$\mathcal{H}_i = \max \text{Occurance} \left\{ \left\lfloor \frac{1}{r} \cdot (\mathcal{W} \cdot X_i + b) \right\rfloor \right\} \quad (1)$$

where r (bin width) controls the coarsened graph size, W represents random projection matrix, X is the feature matrix, and b is the bias term. For further details, refer to UGC [\(Kataria et al.,](#page-9-11) [2024\)](#page-9-11). After coarsening the i th community (Ci), Mcoar(Ci) = {P<sup>i</sup> , Si} yields a partition matrix P<sup>i</sup> ∈ <sup>R</sup> ∥Si∥×∥Ci∥ and a set of coarsened supernodes (Si), as discussed in Section [2.](#page-1-1)

Definition 1. *The neighborhood of a set of nodes* E<sup>i</sup> *is defined as the union of the top* k *most similar nodes in* C<sup>i</sup> *for each node* v ∈ E<sup>i</sup> *, where similarity is measured by the distance function* d(v, u)*. A node* u ∈ C<sup>i</sup> *is considered part of the neighborhood if its distance* d(v, u) *is among the* k *smallest distances for all* u ′ ∈ C<sup>i</sup> *.*

$$\mathcal{N}_k(\mathcal{E}_i) = \bigcup_{v \in \mathcal{E}_i} \{u \in C_i \mid d(v, u) \leq \text{top-}k[d(v, u') : u' \in C_i]\}$$

Goal 3. *The neighborhood of incoming nodes* Nk(Ei) *represents the ideal set of nodes where the incoming nodes* E<sup>i</sup> *are likely to establish connections when the entire community is provided to a structure learning framework.. A robust coarsening framework must reduce the number of nodes within each community* C<sup>i</sup> *while ensuring that the neighborhood of the incoming nodes is preserved.*

## 3.4. Graph Learning only with Potential Nodes

As we now have a smaller representation of the community, we can employ any graph learning algorithms discussed in

Section [2.2](#page-2-4) to learn a graph between coarsened supernodes S<sup>i</sup> and incoming nodes (V i <sup>τ</sup> ∈ V<sup>τ</sup> ). This is the third module of GraphFLEx, i.e., graph learning; we denote it as Mgl. The number of supernodes in S<sup>i</sup> is much smaller compared to the original size of the community, i.e., ∥Si∥ ≪ ∥Ci∥; scalability is not an issue now. We learn a small graph first using Mgl(S<sup>i</sup> , X<sup>i</sup> τ ) = Ge<sup>i</sup> τ (V c τ , A<sup>c</sup> τ ) where X<sup>i</sup> τ represents features of new nodes belonging to i th community at time <sup>τ</sup> , Ge<sup>i</sup> τ (V c τ , A<sup>c</sup> τ ) representing the graph between supernodes and incoming nodes. Utilizing the partition matrix P<sup>i</sup> obtained from Mcoar, we can precisely determine the set of nodes associated with each supernode. For every new node V ∈ V i τ , we identify the connected supernodes and subsequently select nodes within those supernodes. This subset of nodes is denoted by ω<sup>V</sup> <sup>i</sup> τ , the sub-graph associated with ω<sup>V</sup> <sup>i</sup> τ represented by G i τ−1 (ω<sup>V</sup> <sup>i</sup> τ ) then undergoes an additional round of graph learning Mgl(G i τ−1 (ω<sup>V</sup> <sup>i</sup> τ ), X<sup>i</sup> τ ), ultimately providing a clear and accurate connection of new nodes V i <sup>τ</sup> with nodes of Gτ−1, ultimately updating it to G<sup>τ</sup> . This multi-step approach, characterized by coarsening, learning on coarsened graphs, and translation to the original graph, ensures scalability.

Theorem 1. *Neighborhood Preservation. Let* Nk(Ei) *denote the neighborhood of incoming nodes* E<sup>i</sup> *for the* i *th community. With partition matrix* P<sup>i</sup> *and* Mgl(S<sup>i</sup> , X<sup>i</sup> τ ) = G c τ (V c τ , A<sup>c</sup> τ ) *we identify the supernodes connected to incoming nodes* E<sup>i</sup> *and subsequently select nodes within those supernodes; this subset of nodes is denoted by* ω<sup>V</sup> <sup>i</sup> τ *. Formally,*

$$\omega_{V_\tau^i} = \bigcup_{v \in \mathcal{E}_i} \left\{ \bigcup_{s \in S_i} \{\pi^{-1}(s) | A_\tau^c(v, s) \neq 0\} \right\}$$

*Then, with probability* Π{c∈ϕ}p(c)*, it holds that* Nk(Ei) ⊆ ω<sup>V</sup> <sup>i</sup> τ *where*

$$p(c) \leq 1 - \frac{2}{\sqrt{2\pi}} \frac{c}{r} \left[ 1 - e^{-r^2/(2c^2)} \right],$$

*and* ϕ *is a set containing all pairwise distance values*(c = ∥v−u∥) *between every node* v ∈ E<sup>i</sup> *and the nodes* u ∈ ω<sup>V</sup> <sup>i</sup> τ *. Here,* π −1 (s) *denotes the set of nodes mapped to supernode s,* r *is the bin-width hyperparameter of* M*coar.*

*Proof.* The proof is deferred in Appendix [B.](#page-12-1)

*Remark* 4*.* Theorem [1](#page-4-0) establishes that, with a constant probability of success, the neighborhood of incoming nodes Nk(Ei) can be effectively recovered using the GraphFLEx multistep approach, which involves coarsening and learning on the coarsened graph, i.e., Nk(Ei) ⊆ ω<sup>V</sup> <sup>i</sup> τ . The set ω<sup>V</sup> <sup>i</sup> τ , estimated by GraphFLEx, identifies potential candidates where incoming nodes are likely to connect. The probability of failure can be reduced by regulating the average degree of connectivity in Mgl(S<sup>i</sup> , X<sup>i</sup> τ ) = G c τ (V c τ , A<sup>c</sup> τ ). While a fully connected G c τ ensures all nodes in the community are candidates, it significantly increases computational costs for large communities.

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

Table 2: Time complexity analysis of GraphFLEx. Here, N is the number of nodes in the graph, k is the number of nodes in the static subgraph used for clustering (k ≪ N), and c represents the number of detected communities. k<sup>τ</sup> denotes the number of nodes at timestamp τ . Finally, α = ∥S i <sup>τ</sup> ∥ + ∥E<sup>i</sup> <sup>τ</sup> ∥ is the sum of coarsened and incoming nodes in the relevant community at τ timestamp.

| M clust                     | M   | coar |     | M gl      |     |     |   |     |   | GraphFLEx     |
|-----------------------------|-----|------|-----|-----------|-----|-----|---|-----|---|---------------|
| Best (kNN-UGC-ANN) O ( k    |     |      |     |           |     |     |   |     |   |               |
|                             | ) O |      |     |           |     |     |   |     |   |               |
|                             |     | k τ  |     |           |     |     |   |     |   |               |
|                             |     |      |     | O ( α log | α ) | O   |   | ( k |   |               |
|                             |     |      |     |           |     |     |   |     | 2 | +             |
|                             |     |      |     |           |     |     |   |     |   | k τ           |
|                             |     |      |     |           |     |     |   |     |   | c + α log α ) |
| Worst (SC-FGC-GLasso) O ( k |     |      |     |           |     |     |   |     |   |               |
|                             | ) O |      |     |           |     |     |   |     |   |               |
|                             | k τ |      |     |           |     |     |   |     |   |               |
|                             |     |  2  |     |           |     |     |   |     |   |               |
|                             |     | ∥    | S   |           |     |     |   |     |   |               |
|                             |     |      | τ ∥ |           |     |     |   |     |   |               |
|                             |     |      |     | O ( α     |     |     |   |     |   |               |
|                             |     |      |     |           | )   | O ( | k |     |   |               |
|                             |     |      |     |           |     |     |   | 3   | + |               |
|                             |     |      |     |           |     |     |   |     |   | k τ           |
|                             |     |      |     |           |     |     |   |     |   |  2           |
|                             |     |      |     |           |     |     |   |     |   | ∥ S           |
|                             |     |      |     |           |     |     |   |     |   | τ ∥ + α       |

#### 3.5. GraphFLEx Offering Multiple SGL Frameworks

Each module in Figure [3,](#page-5-0) controls distinct properties: clustering influences community detection, coarsening governs supernode formation to reduce graph complexity, and the learning module enforces diverse structural properties. Altering any of these modules results in a new graph learning method. Currently, we support 48 different graph learning configurations, and this number scales exponentially with the addition of new methods to any module. The number of possible frameworks is given by α × β × γ, where α, β, and γ represent the number of clustering, coarsening, and learning methods, respectively.

Figure 3: The versatility of GraphFlex in supporting multiple methods for structure learning.

#### 3.6. Run Time Analysis

We evaluate the run-time complexity of GraphFLEx in two scenarios: (a) the worst-case scenario, where computationally intensive clustering and coarsening modules are selected, providing an upper bound on time complexity, and (b) the best-case scenario, where the most efficient modules are chosen. Table [2](#page-5-1) summarizes the analysis. The run time of GraphFLEx is primarily determined by the learning module (Mgl). GraphFLEx computational time is always bounded by existing approaches, as it operates on a significantly reduced graph space, ensuring efficient performance, especially for larger or expanding graphs. This is also illustrated in Table [3.](#page-6-0)

### 4. Experiments

In this section, we conclude the experiments to back up our findings.

Tasks and Datasets. The experiments focus on four key aspects of GraphFLEx: its computational efficiency, scalability in handling large graphs, the quality of the learned graph structure, and its ability to efficiently handle expanding graphs. To validate the characteristics of GraphFLEx, we conduct extensive experiments on 22 different datasets, including (a) datasets that already have a complete graph structure (allowing comparison between the learned and the original structure), (b) datasets with missing graph structures, (c) synthetic datasets, and (d) small datasets for visualizing the graph structure. More details about datasets are presented in Table [6](#page-14-1) in Appendix [D.](#page-13-1)

![](_page_5_Diagram_8.jpeg)

*System Specifications:* All the experiments conducted for this work were performed on an Intel Xeon W-295 CPU and 64GB of RAM desktop using the Python environment.

Computational Efficiency. Existing methods like k-NN and log-model struggle to learn graph structures even for 20k nodes due to out-of-memory (OOM) or out-of-time (OOT) issues, while l2-model and large-model struggle beyond 50k nodes. Although A-NN and emp-Covar. are faster, GraphFLEx outperforms them on sufficiently large graphs (Table [3\)](#page-6-0). While traditional methods may be efficient for small graphs, GraphFLEx scales significantly better, excelling on large datasets like *Pubmed* and *Syn 5*, where most methods fail. It accelerates structure learning, making A-NN 3× faster and emp-Covar. 2× faster.

### 4.1. Node Classification Accuracy

Experimental Setup. We now evaluate the prediction performance of GNN models when trained on graph structures learned from three distinct scenarios: 1) Original Structure: GNN models trained on the original graph structure, which we refer to as the Base Structure, 2) GraphFLEx Structure: GNN models trained on the graph structure learned from GraphFLEx, and 3)Vanilla Structure: GNN models trained on the graph structure learned from other existing methods.

For each scenario, a unique graph structure is obtained. We trained GNN models on each of these three structure. For more details on GNN model parameters, see Appendix [F.](#page-15-1)

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

Table 3: Computational time for learning graph structures using GraphFLEx (GFlex) with existing methods (Vanilla referred to as Van.). The experimental setup involves treating 50% of the data as static, while the remaining 50% of nodes are treated as incoming nodes arriving in 25 different timestamps. The best times are highlighted by color Green. OOM and OOT denote out-of-memory and out-of-time, respectively.

| Data     | Van.  | ANN GFlex | Van. | KNN GFlex | Van.  | log-model GFlex | Van.  | l2-model GFlex | Van. | emp-Covar. GFlex | Van. | large-model GFlex |
|----------|-------|-----------|------|-----------|-------|-----------------|-------|----------------|------|------------------|------|-------------------|
| Cora     | 335   | 100       | 8.4  | 36.1      | 869   | 81.6            | 424   | 55             | 8.6  | 30               | 2115 | 18.4              |
| Citeseer | 1535  | 454       | 21.9 | 75        | 1113  | 64.5            | 977   | 54.0           | 14.7 | 59.2             | 8319 | 43.9              |
| DBLP     | 2731  | 988       | OOM  | 270       | 77000 | 919             | OOT   | 1470           | 359  | 343              | OOT  | 299               |
| CS       | 22000 | 12000     | OOM  | 789       | OOT   | 838             | 32000 | 809            | 813  | 718              | OOT  | 1469              |
| PubMed   | 770   | 227       | OOM  | 164       | OOT   | 176             | OOT   | 165            | 488  | 299              | OOT  | 262               |
| Phy.     | 61000 | 21000     | OOM  | 903       | OOT   | 959             | OOT   | 908            | 2152 | 1182             | OOT  | 2414              |
| Syn 3    | 95    | 37        | OOM  | 30        | 58000 | 346             | 859   | 53             | 88   | 59               | 5416 | 42                |
| Syn 4    | 482   | 71        | OOM  | 73        | OOT   | 555             | OOT   | 145            | 2072 | 1043             | OOT  | 392               |

Table 4: Node classification accuracies on different GNN models using GraphFLEx (GFlex) with existing Vanilla (Van.) methods. The experimental setup involves treating 70% of the data as static, while the remaining 30% of nodes are treated as new nodes coming in 25 different timestamps. The best and the second-best accuracies in each row are highlighted by dark and lighter shades of Green, respectively. GraphFLEx's structure beats all of the vanilla structures for every dataset. OOM and OOT denotes out-of-memory and out-of-time respectively.

| Data   | Model |    | Van. | ANN | GFlex | Van. | KNN   | GFlex | Van. | log-model | GFlex | Van.  | l2-model GFlex |       | Van. | COVA  | GFlex | Van. | large-model | GFlex | Base Struct. |
|--------|-------|----|------|-----|-------|------|-------|-------|------|-----------|-------|-------|----------------|-------|------|-------|-------|------|-------------|-------|--------------|
| G      | AT    | 34 | 23   | 67  | 37    | OOM  | 69    | 83    | OOT  | 69        | 83    | OOT   | 68.98          | 50    | 48   | 68    | 56    | OOT  | 66          | 38    | 70.84        |
| S AGE  |       | 34 | 23   | 69  | 58    | OOM  | 70    | 28    | OOT  | 70        | 28    | OOT   | 70 68          | 51    | 47   | 70    | 51    | OOT  | 69          | 32    | 72.57        |
| DBLP G | CN    | 34 | 12   | 69  | 41    | OOM  | 73    | 39    | OOT  | 73        | 39    | OOT   | 73.05          | 51    | 50   | 71    | 75    | OOT  | 68          | 55    | 74.43        |
| G      | IN    | 34 | 01   | 69  | 69    | OOM  | 68    | 19    | OOT  | 68        | 19    | OOT   | 73 08          | 52    | 77   | 72    | 03    | OOT  | 71          | 18    | 73.92        |
| G      | AT    | 12 | 47   | 60  | 89    | OOM  | 61.09 |       | OOT  | 60.95     |       | 18.64 | 61.06          | 58.96 |      | 88    | 06    | OOT  | 86          | 22    | 60.75        |
| S AGE  |       | 12 | 70   | 78  | 81    | OOM  | 79.43 |       | OOT  | 79.06     |       | 19.24 | 78.94          | 56.97 |      | 93    | 30    | OOT  | 92          | 79    | 80.33        |
| CS G   | CN    | 12 | 59   | 63  | 81    | OOM  | 67.94 |       | OOT  | 69.33     |       | 19.21 | 66.01          | 58.35 |      | 91.07 |       | OOT  | 84          | 85    | 67.43        |
| G      | IN    | 13 | 07   | 77  | 62    | OOM  | 78.41 |       | OOT  | 78.55     |       | 19.24 | 77.61          | 58.26 |      | 92.07 |       | OOT  | 86          | 03    | 55.65        |
| G      | AT    | 49 | 49   | 83  | 71    | OOM  | 84    | 60    | OOT  | 84        | 60    | OOT   | 84.04          | 72    | 63   | 83    | 97    | OOT  | 81          | 15    | 84.04        |
| S AGE  |       | 50 | 43   | 87  | 27    | OOM  | 87    | 34    | OOT  | 87        | 34    | OOT   | 87.42          | 73    | 57   | 86    | 68    | OOT  | 87          | 34    | 88.88        |
| Pub. G | CN    | 50 | 45   | 82  | 06    | OOM  | 83    | 56    | OOT  | 83        | 56    | OOT   | 83.74          | 73    | 14   | 82    | 39    | OOT  | 78          | 03    | 85.54        |
| G      | IN    | 51 | 82   | 83  | 13    | OOM  | 84    | 31    | OOT  | 84        | 07    | OOT   | 82.93          | 73    | 15   | 83    | 51    | OOT  | 82          | 85    | 86.50        |
| G      | AT    | 29 | 18   | 88  | 06    | OOM  | 88    | 47    | OOT  | 88        | 47    | OOT   | 88.68          | 58    | 96   | 88    | 06    | OOT  | 86          | 22    | 88.58        |
| S AGE  |       | 29 | 57   | 93  | 47    | OOM  | 93    | 47    | OOT  | 93        | 47    | OOT   | 93.78          | 56    | 97   | 93    | 60    | OOT  | 92          | 79    | 94.19        |
| Phy. G | CN    | 27 | 84   | 91  | 27    | OOM  | 91    | 08    | OOT  | 91        | 08    | OOT   | 91.78          | 58    | 35   | 91    | 07    | OOT  | 84          | 85    | 91.48        |
| G      | IN    | 28 | 38   | 92  | 69    | OOM  | 92    | 04    | OOT  | 92        | 04    | OOT   | 92.27          | 58    | 26   | 92    | 07    | OOT  | 86          | 03    | 88.89        |

GNN Models. Graph neural networks (GNNs) such as GCN [\(Kipf & Welling,](#page-9-17) [2016\)](#page-9-17), GraphSage [\(Hamilton](#page-8-19) [et al.,](#page-8-19) [2017\)](#page-8-19), GIN [\(Xu et al.,](#page-10-14) [2018\)](#page-10-14), and GAT [\(Velick](#page-10-15)[ovic et al.,](#page-10-15) [2017\)](#page-10-15) rely on accurate message passing, dictated by the graph structure, for effective embedding. We use these models to evaluate the above-mentioned learned structures. Table [4](#page-6-1) reports node classification performance across all methods. Notably, GraphFLEx outperforms vanilla structures by a significant margin across all datasets, achieving accuracies close to those obtained with the original structure. Figure [9](#page-16-0) in Appendix [F](#page-15-1) illustrates GraphSage classification results, highlighting GraphFLEx's superior performance. For the CS dataset, GraphFLEx (large-model) and GraphFLEx (empCovar.-model) even surpass the original

structure, demonstrating its ability to preserve key structural properties while denoising edges, leading to improved accuracy.

#### 4.2. Clustering Quality

We measure three metrics to evaluate the resulting clusters or community assignments: a) Normalized Mutual Information (NMI) [\(Tsitsulin et al.,](#page-10-4) [2023\)](#page-10-4) between the cluster assignments and original labels; b) Conductance (C) [\(Jerrum](#page-9-18) [& Sinclair,](#page-9-18) [1988\)](#page-9-18) which measures the fraction of total edge volume that points outside the cluster; and c) Modularity (Q) [\(Newman,](#page-9-16) [2006\)](#page-9-16) which measures the divergence between the intra-community edges and the expected one. Table [5](#page-7-0)

394

396

![](_page_7_Diagram_1.jpeg)

Figure 4: Figures (a), (b), and (c) illustrate the growing structure learned using GraphFLEx for *HE* synthetic dataset. Figures (d), (e), and (f) illustrate the learned structure on Zachary's karate dataset when existing methods are employed with GraphFLEx. New nodes are denoted using black color.

illustrates these metrics for single-cell RNA and MNIST dataset (where the whole structure is missing), and Figure [5](#page-7-1) shows the PHATE [\(Moon et al.,](#page-9-19) [2019\)](#page-9-19) visualization of clusters learned using GraphFLEx's clustering module Mclust. We also train the aforementioned GNN models for the node classification task in order to illustrate the efficacy of the learned structures; the accuracies values presented in Table [5,](#page-7-0) clearly highlight the significance of the learned structures, as reflected by the high accuracy values.

Table 5: Clustering results and node classification accuracies. Left: Clustering metrics - NMI, graph conductance C, and Modularity Q. Right: Node classification accuracy for GCN, GraphSAGE, GIN, GAT.

| Data    | NMI   | ↑ C ↓ | Q ↑   | G CN | S AGE | G IN | G AT |
|---------|-------|-------|-------|------|-------|------|------|
| Bar. M. | 0.716 | 0.057 | 0.741 | 91.2 | 96.2  | 95.1 | 94.9 |
| Seger.  | 0.678 | 0.102 | 0.694 | 91.0 | 93.9  | 94.2 | 92.3 |
| Mura.   | 0.843 | 0.046 | 0.706 | 96.9 | 97.4  | 97.5 | 96.4 |
| Bar. H. | 0.674 | 0.078 | 0.749 | 95.3 | 96.4  | 97.2 | 95.8 |
| Xin     | 0.741 | 0.045 | 0.544 | 98.6 | 99.3  | 98.9 | 99.8 |
| MNIST   | 0.677 | 0.082 | 0.712 | 92.9 | 94.5  | 94.9 | 82.6 |

![](_page_7_Figure_11.jpeg)

Figure 5: PHATE visualization of clusters learned using Graph-FLEx clustering module for scRNA-seq datasets.

## 4.3. Structure Visualization

We evaluate the structures generated by GraphFLEx through visualizations on four small datasets: (i) MNIST [\(LeCun](#page-9-20) [et al.,](#page-9-20) [2010\)](#page-9-20), consisting of handwritten digit images, where Figure [6\(](#page-7-2)a) shows that images of the same digit are mostly connected; (ii) Pre-trained GloVe embeddings [\(Pennington](#page-9-21) [et al.,](#page-9-21) [2014\)](#page-9-21) of English words, with Figure [6\(](#page-7-2)b) revealing that frequently used words are closely connected; (iii) A synthetic H.E dataset (see Appendix [D\)](#page-13-1), demonstrating Graph-FLEx's ability to handle expanding networks without requiring full relearning. Figure [4\(](#page-7-3)a-c) shows the graph structure evolving as 30 new nodes are added over three timestamps; and (iv) Zachary's karate club network [\(Zachary,](#page-10-16) [1977\)](#page-10-16), which highlights GraphFLEx's multi-framework capability. Figure [4\(](#page-7-3)d-f) shows three distinct graph structures after altering the learning module.

![](_page_7_Diagram_5.jpeg)

Figure 6: Figures demonstrate the effectiveness of our framework in learning meaningful structure between similar MNIST digit images and pre-trained GloVe embeddings.

## 5. Conclusion

Large or expanding graphs challenge the best of graph learning approaches. GraphFLEx, introduced in this paper, seamlessly adds new nodes into an existing graph structure. It offers diverse methods for acquiring the graph's structure. GraphFLEx consists of three key modules: Clustering, Coarsening, and Learning which empowers Graph-FLEx to serves as a comprehensive framework applicable individually for clustering, coarsening, and learning tasks. GraphFLEx is typically 3X faster than other state of the art methods and scales well with large graphs. It achieves accuracies close to training on the original graph, in most instances. The performance across multiple real and synthetic datasets affirms the utility and efficacy of GraphFLEx for graph structure learning.

Limitations and Future Work. GraphFLEx is designed assuming minimal inter-community connectivity, which aligns well with many real-world scenarios. However, its applicability to heterophilic graphs may require further adaptation. Future work will focus on extending the framework to supervised GSL methods and heterophilic graphs, broadening its scalability and versatility.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Allen, J. D., Xie, Y., Chen, M., Girard, L., and Xiao, G. Comparing statistical methods for constructing large scale gene networks. *PloS one*, 7(1):e29348, 2012. (Cited at
  - p. [1.](#page-0-1)) Banerjee, O., El Ghaoui, L., and d'Aspremont, A. Model selection through sparse maximum likelihood estimation for multivariate gaussian or binary data. *The Journal of Machine Learning Research*, 9:485–516, 2008. (Cited at
  - p. [3.](#page-2-5)) Belkin, M., Niyogi, P., and Sindhwani, V. Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. *Journal of machine learning research*, 7(11), 2006. (Cited at p. [3.](#page-2-5)) Bianchi, F. M. Simplifying clustering with graph neural networks. *arXiv preprint arXiv:2207.08779*, 2022. (Cited at p. [4.](#page-3-2)) Bianchi, F. M., Grattarola, D., and Alippi, C. Spectral clustering with graph neural networks for graph pooling. In *International conference on machine learning*, pp. 874–
  - 883. PMLR, 2020. (Cited at p. [4.](#page-3-2)) Bruna, J., Zaremba, W., Szlam, A., and LeCun, Y. Spectral networks and deep locally connected networks on graphs. arxiv. *arXiv preprint arXiv:1312.6203*, 2014. (Cited at
  - p. [3.](#page-2-5)) Chen, Y. and Wu, L. Graph neural networks: Graph structure learning. *Graph Neural Networks: Foundations, Frontiers, and Applications*, pp. 297–321, 2022. (Cited at p. [1.](#page-0-1)) Datar, M., Immorlica, N., Indyk, P., and Mirrokni, V. S. Locality-sensitive hashing scheme based on p-stable distributions. In *Proceedings of the twentieth annual symposium on Computational geometry*, pp. 253–262, 2004. (Cited at p. [13.](#page-12-2)) Defferrard, M., Martin, L., Pena, R., and Perraudin, N. Pygsp: Graph signal processing in python. URL [https:](https://github.com/epfl-lts2/pygsp/) [//github.com/epfl-lts2/pygsp/](https://github.com/epfl-lts2/pygsp/). (Cited at
  - p. [14.](#page-13-2)) Defferrard, M., Bresson, X., and Vandergheynst, P. Convolutional neural networks on graphs with fast localized spectral filtering. *Advances in neural information processing systems*, 29, 2016. (Cited at p. [3.](#page-2-5)) Dempster, A. P. Covariance selection. *Biometrics*, pp. 157– 175, 1972. (Cited at p. [3.](#page-2-5)) Dong, W., Moses, C., and Li, K. Efficient k-nearest neighbor graph construction for generic similarity measures. In *Proceedings of the 20th international conference on World wide web*, pp. 577–586, 2011. (Cited at p. [1.](#page-0-1)) Dong, X., Thanou, D., Frossard, P., and Vandergheynst,
    - P. Learning laplacian matrix in smooth graph signal representations. *IEEE Transactions on Signal Processing*, 64(23):6160–6173, 2016. (Cited at pp. [1](#page-0-1) and [3.](#page-2-5)) Fatemi, B., El Asri, L., and Kazemi, S. M. Slaps: Selfsupervision improves structure learning for graph neural networks. *Advances in Neural Information Processing Systems*, 34:22667–22681, 2021. (Cited at p. [1.](#page-0-1)) Fortunato, S. Community detection in graphs. *Physics reports*, 486(3-5):75–174, 2010. (Cited at p. [3.](#page-2-5)) Fout, A., Byrd, J., Shariat, B., and Ben-Hur, A. Protein interface prediction using graph convolutional networks. *Advances in neural information processing systems*, 30, 2017. (Cited at p. [1.](#page-0-1)) Friedman, J., Hastie, T., and Tibshirani, R. Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3):432–441, 2008. (Cited at pp. [1](#page-0-1) and [3.](#page-2-5)) Fu, X., Zhang, J., Meng, Z., and King, I. Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding. In *Proceedings of The Web Conference 2020*, pp. 2331–2341, 2020. (Cited at p. [14.](#page-13-2)) Gu, J., Zhao, H., Lin, Z., Li, S., Cai, J., and Ling, M. Scene graph generation with external knowledge and image reconstruction. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1969–1978, 2019. (Cited at p. [1.](#page-0-1)) Hamilton, W., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30, 2017. (Cited at p. [7.](#page-6-2)) Hashemi, M., Gong, S., Ni, J., Fan, W., Prakash, B. A., and Jin, W. A comprehensive survey on graph reduction: Sparsification, coarsening, and condensation. *arXiv preprint arXiv:2402.03358*, 2024. (Cited at p. [3.](#page-2-5)) Holme, P. and Saramaki, J. Temporal networks. ¨ *Physics reports*, 519(3):97–125, 2012. (Cited at p. [2.](#page-1-2)) Hsieh, C.-J., Dhillon, I., Ravikumar, P., and Sustik,
      - M. Sparse inverse covariance matrix estimation using quadratic approximation. *Advances in neural information processing systems*, 24, 2011. (Cited at p. [1.](#page-0-1))

## Impact Statement

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 549 Hu, C., Cheng, L., Sepulcre, J., El Fakhri, G., Lu, Y. M., and Li, Q. A graph theoretical regression model for brain connectivity learning of alzheimer's disease. In *2013 IEEE 10th International Symposium on Biomedical Imaging*, pp. 616–619. IEEE, 2013. (Cited at p. [3.](#page-2-5)) Jerrum, M. and Sinclair, A. Conductance and the rapid mixing property for markov chains: the approximation of permanent resolved. In *Proceedings of the twentieth annual ACM symposium on Theory of computing*, pp. 235–244, 1988. (Cited at p. [7.](#page-6-2)) Jin, W., Ma, Y., Liu, X., Tang, X., Wang, S., and Tang, J. Graph structure learning for robust graph neural networks. In *Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining*, pp. 66–74, 2020. (Cited at p. [1.](#page-0-1)) Kalofolias, V. How to learn a graph from smooth signals. In *Artificial intelligence and statistics*, pp. 920–929. PMLR, 2016. (Cited at pp. [1](#page-0-1) and [3.](#page-2-5)) Kamvar, S. D., Klein, D., and Manning, C. D. Spectral learning. In *IJCAI*, volume 3, pp. 561–566, 2003. (Cited at p. [3.](#page-2-5)) Kataria, M., Khandelwal, A., Das, R., Kumar, S., and Jayadeva, J. Linear complexity framework for featureaware graph coarsening via hashing. In *NeurIPS 2023 Workshop: New Frontiers in Graph Learning*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=HKdsrm5nCW) [id=HKdsrm5nCW](https://openreview.net/forum?id=HKdsrm5nCW). (Cited at p. [3.](#page-2-5)) Kataria, M., Kumar, S., and Jayadeva, J. UGC: Universal graph coarsening. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=nN6NSd1Qds) [id=nN6NSd1Qds](https://openreview.net/forum?id=nN6NSd1Qds). (Cited at pp. [3,](#page-2-5) [5,](#page-4-1) and [13.](#page-12-2)) Khazane, A., Rider, J., Serpe, M., Gogoglou, A., Hines, K., Bruss, C. B., and Serpe, R. Deeptrax: Embedding graphs of financial transactions. In *2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA)*, pp. 126–133. IEEE, 2019. (Cited at
  - p. [2.](#page-1-2)) Kim, S., Yun, S., and Kang, J. Dygrain: An incremental learning framework for dynamic graphs. In *IJCAI*, pp. 3157–3163, 2022. (Cited at pp. [3](#page-2-5) and [14.](#page-13-2)) Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*, 2016. (Cited at pp. [7](#page-6-2) and [14.](#page-13-2)) Koller, D. and Friedman, N. *Probabilistic graphical models: principles and techniques*. MIT press, 2009. (Cited at
- p. [3.](#page-2-5)) Kumar, M., Sharma, A., and Kumar, S. A unified framework for optimization-based graph coarsening. *Journal of Machine Learning Research*, 24(118):1– 50, 2023. URL [http://jmlr.org/papers/v24/](http://jmlr.org/papers/v24/22-1085.html) [22-1085.html](http://jmlr.org/papers/v24/22-1085.html). (Cited at p. [3.](#page-2-5)) Lao, D., Yang, X., Wu, Q., and Yan, J. Variational inference for training graph neural networks in low-data regime through joint structure-label estimation. In *Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining*, pp. 824–834, 2022. (Cited at
  - p. [1.](#page-0-1)) LeCun, Y., Cortes, C., and Burges, C. Mnist handwritten digit database. *ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist*, 2, 2010. (Cited at pp. [8](#page-7-4) and [15.](#page-14-2)) Liu, Y., Zheng, Y., Zhang, D., Chen, H., Peng, H., and Pan,
  - S. Towards unsupervised deep graph structure learning. In *Proceedings of the ACM Web Conference 2022*, pp. 1392–1403, 2022. (Cited at p. [1.](#page-0-1)) Loukas, A. Graph reduction with spectral and cut guarantees.
  - *J. Mach. Learn. Res.*, 20(116):1–42, 2019. (Cited at p. [3.](#page-2-5)) Lu, L. and Zhou, T. Link prediction in complex networks: ¨ A survey. *Physica A: statistical mechanics and its applications*, 390(6):1150–1170, 2011. (Cited at p. [14.](#page-13-2)) MacQueen, J. et al. Some methods for classification and analysis of multivariate observations. In *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*, volume 1, pp. 281–297. Oakland, CA, USA, 1967. (Cited at pp. [1](#page-0-1) and [3.](#page-2-5)) Moon, K. R., Van Dijk, D., Wang, Z., Gigante, S., Burkhardt,
  - D. B., Chen, W. S., Yim, K., Elzen, A. v. d., Hirn, M. J., Coifman, R. R., et al. Visualizing structure and transitions in high-dimensional biological data. *Nature biotechnology*, 37(12):1482–1492, 2019. (Cited at pp. [8](#page-7-4) and [19.](#page-18-0)) Muja, M. and Lowe, D. G. Scalable nearest neighbor algorithms for high dimensional data. *IEEE transactions on pattern analysis and machine intelligence*, 36(11): 2227–2240, 2014. (Cited at p. [1.](#page-0-1)) Newman, M. E. Modularity and community structure in networks. *Proceedings of the national academy of sciences*, 103(23):8577–8582, 2006. (Cited at pp. [4](#page-3-2) and [7.](#page-6-2)) Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., and Wermter,
  - S. Continual lifelong learning with neural networks: A review. *Neural networks*, 113:54–71, 2019. (Cited at pp. [3](#page-2-5) and [14.](#page-13-2)) Pennington, J., Socher, R., and Manning, C. D. Glove: Global vectors for word representation. In *Proceedings*

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 *of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pp. 1532–1543, 2014. (Cited at pp. [8](#page-7-4) and [15.](#page-14-2)) Shchur, O., Mumme, M., Bojchevski, A., and Gunnemann, ¨
  - S. Pitfalls of graph neural network evaluation. *arXiv preprint arXiv:1811.05868*, 2018. (Cited at p. [14.](#page-13-2)) Tsitsulin, A., Palowitch, J., Perozzi, B., and Muller, E. ¨ Graph clustering with graph neural networks. *Journal of Machine Learning Research*, 24(127):1–21, 2023. (Cited at pp. [3,](#page-2-5) [4,](#page-3-2) and [7.](#page-6-2)) Van de Ven, G. M. and Tolias, A. S. Three scenarios for continual learning. *arXiv preprint arXiv:1904.07734*, 2019. (Cited at pp. [3](#page-2-5) and [14.](#page-13-2)) Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y., et al. Graph attention networks. *stat*, 1050 (20):10–48550, 2017. (Cited at p. [7.](#page-6-2)) Vogelstein, J. T., Roncal, W. G., Vogelstein, R. J., and Priebe,
  - C. E. Graph classification using signal-subgraphs: Applications in statistical connectomics. *IEEE transactions on pattern analysis and machine intelligence*, 35(7):1539– 1551, 2012. (Cited at p. [14.](#page-13-2)) Wagstaff, K., Cardie, C., Rogers, S., Schrodl, S., et al. Con- ¨ strained k-means clustering with background knowledge. In *Icml*, volume 1, pp. 577–584, 2001. (Cited at p. [3.](#page-2-5)) Wang, F. and Zhang, C. Label propagation through linear neighborhoods. In *Proceedings of the 23rd international conference on Machine learning*, pp. 985–992, 2006. (Cited at pp. [1](#page-0-1) and [3.](#page-2-5)) Wang, K., Shen, Z., Huang, C., Wu, C., Dong, Y., and Kanakia, A. Microsoft academic graph: When experts are not enough. quantitative science studies, 1 (1), 396– 413, 2020. (Cited at p. [4.](#page-3-2)) Watts, D. J. and Strogatz, S. H. Collective dynamics of 'small-world'networks. *nature*, 393(6684):440–442, 1998. (Cited at p. [15.](#page-14-2)) Wu, Q., Zhao, W., Li, Z., Wipf, D. P., and Yan, J. Nodeformer: A scalable graph structure learning transformer for node classification. *Advances in Neural Information Processing Systems*, 35:27387–27401, 2022. (Cited at
- p. [1.](#page-0-1)) Wu, T., Liu, Q., Cao, Y., Huang, Y., Wu, X.-M., and Ding, J. Continual graph convolutional network for text classification. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 13754–13762, 2023. (Cited at pp. [3](#page-2-5) and [14.](#page-13-2)) Wu, Y., Lian, D., Xu, Y., Wu, L., and Chen, E. Graph convolutional networks with markov random field reasoning for social spammer detection. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 1054–1061, 2020. (Cited at p. [1.](#page-0-1)) Xiang, L., Yuan, Q., Zhao, S., Chen, L., Zhang, X., Yang, Q., and Sun, J. Temporal recommendation on graphs via long-and short-term preference fusion. In *Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining*, pp. 723–732, 2010. (Cited at p. [4.](#page-3-2)) Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? *arXiv preprint arXiv:1810.00826*, 2018. (Cited at p. [7.](#page-6-2)) Yang, F., Wang, W., Wang, F., Fang, Y., Tang, D., Huang, J., Lu, H., and Yao, J. scbert as a large-scale pretrained deep language model for cell type annotation of single-cell rnaseq data. *Nature Machine Intelligence*, 4(10):852–866, 2022. (Cited at p. [14.](#page-13-2)) Yang, Z., Cohen, W., and Salakhudinov, R. Revisiting semi-supervised learning with graph embeddings. In *International conference on machine learning*, pp. 40–48. PMLR, 2016. (Cited at p. [14.](#page-13-2)) You, J., Du, T., and Leskovec, J. Roland: graph learning framework for dynamic graphs. In *Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining*, pp. 2358–2366, 2022. (Cited at pp. [3](#page-2-5) and [14.](#page-13-2)) Zachary, W. W. An information flow model for conflict and fission in small groups. *Journal of anthropological research*, 33(4):452–473, 1977. (Cited at pp. [8,](#page-7-4) [15,](#page-14-2) and [19.](#page-18-0)) Zhang, X., Song, D., and Tao, D. Cglb: Benchmark tasks for continual graph learning. *Advances in Neural Information Processing Systems*, 35:13006–13021, 2022. (Cited at pp. [3](#page-2-5) and [14.](#page-13-2)) Zhao, Y., Levina, E., and Zhu, J. Consistency of community detection in networks under degree-corrected stochastic block models. 2012. (Cited at pp. [4](#page-3-2) and [13.](#page-12-2)) Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., and Sun, M. Graph neural networks: A review of methods and applications. *AI open*, 1:57–81, 2020. (Cited at p. [1.](#page-0-1)) Zhu, X., Ghahramani, Z., and Lafferty, J. D. Semisupervised learning using gaussian fields and harmonic functions. In *Proceedings of the 20th International conference on Machine learning (ICML-03)*, pp. 912–919, 2003. (Cited at p. [3.](#page-2-5))

Zhu, Y., Xu, W., Zhang, J., Du, Y., Zhang, J., Liu, Q., Yang, C., and Wu, S. A survey on graph structure learning: Progress and opportunities. *arXiv preprint arXiv:2103.03036*, 2021. (Cited at p. [1.](#page-0-1))

689 690

694

696

698

700

704

706

708 709

711

# Appendix

# A. Degree-Corrected Stochastic Block Model(DC-SBM)

The DC-SBM is one of the most commonly used models for networks with communities and postulates that, given node labels c = c1, ...cn, the edge variables A′ ij s are generated via the formula

$$E[A_{ij}] = \theta_i \theta_j P_{c_i} P_{c_j}$$

, where θ<sup>i</sup> is a "degree parameter" associated with node i, reflecting its individual propernsity to form ties, and P is a K × K symmetric matrix containing the between/withincommunity edge probabilities and P<sup>c</sup>iP<sup>c</sup><sup>j</sup> denotes the edge probabilities between community c<sup>i</sup> and c<sup>j</sup> .

For DC-SBM model [\(Zhao et al.,](#page-10-13) [2012\)](#page-10-13) assumed P<sup>n</sup> on n nodes with k classes, each node v<sup>i</sup> is given a label/degree pair(c<sup>i</sup> , θi), drawn from a discrete joint distribution ΠK×<sup>m</sup> which is fixed and does not depend on n. This implies that each θi is one of a fixed set of values 0 ≤ x<sup>1</sup> ≤ .... ≤ xm. To facilitate analysis of asymptotic graph sparsity, we parameterize the edge probability matrix P as P<sup>n</sup> = ρnP where P is independent of n, and ρ<sup>n</sup> = λn/n where λ<sup>n</sup> is the average degree of the network.

# B. Neighbourhood Preservation

Theorem 2. *Neighborhood Preservation. Let* Nk(Ei) *denote the neighborhood of incoming nodes* E<sup>i</sup> *for the* i *th community. With partition matrix* P<sup>i</sup> *and* Mgl(S<sup>i</sup> , X<sup>i</sup> τ ) = G c τ (V c τ , A<sup>c</sup> τ ) *we identify the supernodes connected to incoming nodes* E<sup>i</sup> *and subsequently select nodes within those supernodes; this subset of nodes is denoted by* ω<sup>V</sup> <sup>i</sup> τ *. Formally,*

$$\omega_{V_\tau^i} = \bigcup_{v \in \mathcal{E}_i} \left\{ \bigcup_{s \in S_i} \{\pi^{-1}(s) | A_\tau^c(v, s) \neq 0\} \right\}$$

*Then, with probability* Π{c∈ϕ}p(c)*, it holds that* Nk(Ei) ⊆ ω<sup>V</sup> <sup>i</sup> τ *where*

$$p(c) \leq 1 - \frac{2}{\sqrt{2\pi}} \frac{c}{r} \left[ 1 - e^{-r^2/(2c^2)} \right],$$

*and* ϕ *is a set containing all pairwise distance values* (c = ∥v − u∥) *between every node* v ∈ E<sup>i</sup> *and the nodes* u ∈ ω<sup>V</sup> <sup>i</sup> τ *. Here,* π −1 (s) *denotes the set of nodes mapped to supernode s,* r *is the bin-width hyperparameter of* M*coar.*

Proof: The probability that LSH random projection [\(Kataria et al.,](#page-9-11) [2024;](#page-9-11) [Datar et al.,](#page-8-20) [2004\)](#page-8-20) preserves the distance between two nodes v and u i.e., d(u, v) = c, is given by:

$$p(c) = \int_0^r \frac{1}{c} f_2 \left( \frac{t}{c} \right) \left( 1 - \frac{t}{r} \right) dt,$$

where f2(x) = √ 2 2π e −x <sup>2</sup>/2 represents the Gaussian kernel when the projection matrix is randomly sampled from pstable(p = 2) distribution [\(Datar et al.,](#page-8-20) [2004\)](#page-8-20).

The probability p(c) can be decomposed into two terms:

$$p(c) = S_1(c) - S_2(c),$$

S1(c) and S2(c) are defined as follows:

$$S_1(c) = \frac{2}{\sqrt{2\pi}} \int_0^r e^{-(t/c)^2/2} dt \leq 1,$$

$$S_2(c) = \frac{2}{\sqrt{2\pi}} \int_0^r e^{-(t/c)^2/2} \frac{t}{r} dt.$$

$$S_2(c) = \frac{2}{\sqrt{2\pi}} \cdot \frac{c}{r} \int_0^r e^{-(t/c)^2/2} \frac{t}{c^2} dt$$

718

724

726

728

731

734

736

738

751

754

756

758

760

764

766

Expanding S2(c) :

$$S_2(c) = \frac{2}{\sqrt{2\pi}} \cdot \frac{c}{r} \int_0^{r^2/(2c^2)} e^{-y} dy$$

$$S_2(c) = \frac{2}{\sqrt{2\pi}} \cdot \frac{c}{r} \left[ 1 - e^{-r^2/(2c^2)} \right]$$

Thus, the probability p(c) can be bounded as:

$$p(c) \leq 1 - \frac{2}{\sqrt{2\pi}} \frac{c}{r} \left[ 1 - e^{-r^2/(2c^2)} \right].$$

Now, let ϕ be the set of all pairwise distances d(u, v), where v ∈ E<sup>i</sup> and nodeω<sup>V</sup> <sup>i</sup> τ . The probability that all nodes in Nk(Ei) are preserved within ω<sup>V</sup> <sup>i</sup> τ , requires that all distances c ∈ ϕ are also preserved. The probability is then given by:

$$\prod_{c \in \phi} p(c) \leq \prod_{c \in \phi} \left( 1 - \frac{2}{\sqrt{2\pi}} \frac{c}{r} \left[ 1 - e^{-r^2/(2c^2)} \right] \right).$$

## C. Continual Learning and Dynamic Graph Learning

In this subsection, we highlight the key distinctions between Graph Structure Learning (GSL) and related fields to justify our specific selection of related works in Section [2.2.](#page-2-4) GSL is often confused with topics such as Continual Learning (CL) and Dynamic Graph Learning (DGL).

CL [\(Van de Ven & Tolias,](#page-10-7) [2019;](#page-10-7) [Zhang et al.,](#page-10-8) [2022;](#page-10-8) [Parisi et al.,](#page-9-14) [2019\)](#page-9-14) addresses the issue of catastrophic forgetting, where a model's performance on previously learned tasks degrades significantly after training on new tasks. In CL, the model has access only to the current task's data and cannot utilize data from prior tasks. Conversely, DGL [\(Kim et al.,](#page-9-15) [2022;](#page-9-15) [Wu et al.,](#page-10-9) [2023;](#page-10-9) [You et al.,](#page-10-10) [2022\)](#page-10-10) focuses on capturing the evolving structure of graphs and maintaining updated graph representations, with access to all prior information.

While both *CL and DGL* aim to *enhance model adaptability* to dynamic data, GSL is primarily concerned with generating *high-quality graph structures* that can be leveraged for downstream tasks such as node classification [\(Kipf & Welling,](#page-9-17) [2016\)](#page-9-17), link prediction [\(Lu & Zhou](#page-9-22) ¨ , [2011\)](#page-9-22), and graph classification [\(Vogelstein et al.,](#page-10-17) [2012\)](#page-10-17). Moreover, in CL and DGL, different tasks typically involve distinct data distributions, whereas GSL assumes a consistent data distribution throughout.

## D. Datasets

Datasets used in our experiments vary in size, with nodes ranging from 1k to 60k. Table [6](#page-14-1) lists all the datasets we used in our work. We evaluate our proposed framework GraphFlex on real-world datasets *Cora ,Citeseer, Pubmed* [\(Yang et al.,](#page-10-18) [2016\)](#page-10-18), *CS, Physics* [\(Shchur et al.,](#page-10-19) [2018\)](#page-10-19), *DBLP* [\(Fu et al.,](#page-8-21) [2020\)](#page-8-21), all of which include graph structures. These datasets allow us to compare the learned structures with the originals. Additionally, we utilize single-cell RNA pancreas datasets [\(Yang et al.,](#page-10-20) [2022\)](#page-10-20), including Baron, Muraro, Segerstolpe, and Xin, where the graph structure is missing. The Baron dataset was downloaded from the Gene Expression Omnibus (GEO) (accession no. GSE84133). The Muraro dataset was downloaded from GEO (accession no. GSE85241). The Segerstolpe dataset was accessed from ArrayExpress (accession no. E-MTAB-5061). The Xin dataset was downloaded from GEO (accession no. GSE81608). We simulate the expanding graph scenario by splitting the original dataset across different T timestamps. We assumed 50% of the nodes were static, with the remaining nodes arriving as incoming nodes at different timestamps.

Synthetic datasets: Different data generation techniques validate that our results are generalized to different settings. Please refer to Table [6](#page-14-1) for more details about the number of nodes, edges, features, and classes, Syn denotes the type of synthetic datasets. Figure [7](#page-14-3) shows graphs generated using different methods. We have employed three different ways to generate synthetic datasets which are mentioned below:

- PyGSP(PyGsp): We used synthetic graphs created by PyGSP [\(Defferrard et al.\)](#page-8-22) library. PyG-G and PyG-S denotes grid and sensor graphs from PyGSP.

774

776

778

794

796

800

804

806

808

![](_page_14_Diagram_5.jpeg)

- • Watts–Strogatz's small world(SW): [\(Watts & Strogatz,](#page-10-21) [1998\)](#page-10-21) proposed a generation model that produces graphs with small-world properties, including short average path lengths and high clustering.
- Heterophily(HE): We propose a method for creating synthetic datasets to explore graph behavior across a heterophily spectrum by manipulating heterophilic factor α, and classes. α is determined by dividing the number of edges connecting nodes from different classes by the total number of edges in the graph.

Visulization Datasets: To evaluate, the learned graph structure, we have also included three datasets: (i) MNIST [\(LeCun](#page-9-20) [et al.,](#page-9-20) [2010\)](#page-9-20), consisting of handwritten digit images; (ii) Pre-trained GloVe embeddings [\(Pennington et al.,](#page-9-21) [2014\)](#page-9-21) of English words; and (iii) Zachary's karate club network [\(Zachary,](#page-10-16) [1977\)](#page-10-16).

| Category Data         | Nodes  | Edges  | Feat.  | Class |               | Type         |
|-----------------------|--------|--------|--------|-------|---------------|--------------|
| Cora                  | 2,708  | 5,429  | 1,433  | 7     | Citation      | network      |
| Citeseer              | 3,327  | 9,104  | 3,703  | 6     | Citation      | network      |
| DBLP Original         | 17,716 | 52.8k  | 1,639  | 4     | Research      | paper        |
| CS Structure          | 18,333 | 163.7k | 6,805  | 15    | Co-authorship | network      |
| PubMed Known          | 19,717 | 44.3k  | 500    | 3     | Citation      | network      |
| Physics               | 34,493 | 247.9k | 8,415  | 5     | Co-authorship | network      |
| Not Known             |        |        |        |       |               |              |
| Xin                   | 1,449  | NA     | 33,889 | 4     | Human         | Pancreas     |
| Baron Mouse Original  | 1,886  | NA     | 14,861 | 13    | Mouse         | Pancreas     |
| Muraro Structure      | 2,122  | NA     | 18,915 | 9     | Human         | Pancreas     |
| Segerstolpe           | 2,133  | NA     | 22,757 | 13    | Human         | Pancreas     |
| Baron Human           | 8,569  | NA     | 17,499 | 14    | Human         | Pancreas     |
| Syn 1                 | 2,000  | 8,800  | 150    | 4     |               | SW           |
| Syn 2                 | 5,000  | 22k    | 150    | 4     |               | SW           |
| Syn 3                 | 10,000 | 44k    | 150    | 7     |               | SW           |
| Syn 4 Synthetic       | 50,000 | 220k   | 150    | 7     |               | SW           |
| Syn 5                 | 400    | 1,520  | 100    | 4     |               | PyG-G        |
| Syn 6                 | 2,500  | 9,800  | 100    | 4     |               | PyG-S        |
| Syn 7                 | 1,000  | 9,990  | 150    | 4     |               | HE           |
| Syn 8                 | 2,000  | 40k    | 150    | 4     |               | HE           |
| Visulization Datasets |        |        |        |       |               |              |
| MNIST                 | 60,000 | NA     | 784    | 10    |               | Images       |
| Zachary’s karate      | 34     | 156    | 34     | 4     | Karate        | club network |
| Glove                 | 2,000  | NA     | 50     | NA    | GloVe         | embeddings   |

Table 6: Summary of the datasets.

Figure 7: This figure illustrates different types of synthetic graphs generated using i)PyGSP, ii) Watts–Strogatz's small world(SW), and iii) Heterophily(HE). N denotes the number of nodes, while α denotes the number of classes.

826 828 831 834 836 838 840 841 842 843 844 845 846 847 848 Algorithm 1 GraphFlex: A Unified Structure Learning framework for expanding and Large Scale Graphs Input: Graph G0(X0, A0), expanding nodes set E T <sup>1</sup> = {E<sup>τ</sup> (V<sup>τ</sup> , X<sup>τ</sup> )} T τ=1 Parameter: GClust, GCoar, GL ← Clustering, Coarsening and Learning Module Output: Graph G<sup>T</sup> (X<sup>T</sup> , A<sup>T</sup> ) 1: Train clustering module *train*(Mclust, GClust, G0) 2: for each Et(Vt, Xt) in E T <sup>1</sup> do 3: C<sup>t</sup> = *infer*(Mclust, Xt), C<sup>t</sup> ∈ <sup>R</sup> <sup>N</sup><sup>t</sup> denotes the communities of N<sup>t</sup> nodes at time t. 4: I<sup>t</sup> = *unique*(Ct). 5: for each I i t in I<sup>t</sup> do 6: G<sup>i</sup> <sup>t</sup>−<sup>1</sup> = subgraph(Gt−1, I i t ) 7: {S i t−1 , P<sup>i</sup> <sup>t</sup>−1} = Mcoar(G<sup>i</sup> t−1 ), S i <sup>t</sup>−<sup>1</sup> ∈ <sup>R</sup> k×d are features of k supernodes, P i <sup>t</sup>−<sup>1</sup> ∈ <sup>R</sup> k×N<sup>i</sup> <sup>t</sup> is the partition matrix. 8: Gc<sup>i</sup> t−1 (S i t−1 , A<sup>i</sup> t−1 ) = Mgl(S i t−1 , X<sup>i</sup> t ), Gc<sup>i</sup> t−1 is the learned graph on super-nodes S i t−1 and new node X<sup>i</sup> t . 9: ω i <sup>t</sup> ← [] 10: for x ∈ X<sup>i</sup> <sup>t</sup> do 11: ω i t .append(x) 12: n<sup>p</sup> = {n | A<sup>i</sup> t−1 [n] > 0} 13: ω i t .append(np) 14: end for 15: Gt−<sup>1</sup> = *update*(Gt−1,Mgl(ω i t )) 16: end for 17: G<sup>t</sup> = Gt−<sup>1</sup> 18: end for 19: return G<sup>T</sup> (X<sup>T</sup> , A<sup>T</sup> )

849

854

856

858

860

864

866

868

874

876

# F. Other GNN models

We used four GNN models, namely GCN, GraphSage, GIN, and GAT. Table [7](#page-16-1) contains parameter details we used to train GraphFlex. We have used these parameters across all methods.

![](_page_15_Diagram_5.jpeg)

![](_page_16_Figure_2.jpeg)

911

914 915 916

918

924

928

934

Figure 9: GraphSage accuracies when structure is learned or given with 3 different scenarios(Vanilla, GraphFlex, Original) across different datasets, highlighting performance with 30% node growth over 25 timestamps.

Figure [8](#page-15-2) illustrates the pipeline for training our GNN models. Graph structures were learned using both existing methods and GraphFlex, and GNN models were subsequently trained on both structures. Results across all datasets are presented in Table [8](#page-17-0) and Table [4.](#page-6-1)

Table 7: GNN model parameters.

| Model     | Hidden |      | Layers L.R | Decay  | Epoch |
|-----------|--------|------|------------|--------|-------|
| GCN       | { 64   | , 64 | } 0.003    | 0.0005 | 500   |
| GraphSage | { 64   | , 64 | } 0.003    | 0.0005 | 500   |
| GIN       | { 64   | , 64 | } 0.003    | 0.0005 | 500   |
| GAT       | { 64   | , 64 | } 0.003    | 0.0005 | 500   |

We randomly split data in 60%, 20%, 20% for training-validation-test. The results for these models on synthetic datasets are presented in Table [8.](#page-17-0)

Figure [8](#page-15-2) illustrates the pipeline for training our GNN models. Graph structures were learned using both existing methods and GraphFlex, and GNN models were subsequently trained on both structures.

# G. Computational Efficiency

Table [9](#page-17-1) illustrates the remaining computational time for learning graph structures using GraphFLEx with existing Vanilla methods on Synthetic datasets. While traditional methods may be efficient for small graphs, GraphFLEx scales significantly better, excelling on large datasets like *Pubmed* and *Syn 5*, where most methods fail.

938

954

956

958

971

974

976

978

987 988

Table 8: Node classification accuracies on different GNN models using GraphFLEx (GFlex) with existing Vanilla (Van.) methods. The experimental setup involves treating 70% of the data as static, while the remaining 30% of nodes are treated as new nodes coming in 25 different timestamps. The best and the second-best accuracies in each row are highlighted by dark and lighter shades of Green, respectively. GraphFLEx's structure beats all of the vanilla structures for every dataset. OOM and OOT denotes out-of-memory and out-of-time respectively.

| Dataset    | Model |    | Van.  | ANN | GFlex |    | Van.  | KNN | GFlex |    | Van.  |    | GFlex |    | Van.  | l2-model | GFlex |    | Van.  | COVA | GFlex |    | Van.  |    | GFlex | Base Struc. |
|------------|-------|----|-------|-----|-------|----|-------|-----|-------|----|-------|----|-------|----|-------|----------|-------|----|-------|------|-------|----|-------|----|-------|-------------|
| G          | AT    | 18 | 73    | 73  | 84    | 20 | 96    | 73  | 65    | 16 | 14    | 72 | 36    | 18 | 74    | 73       | 10    | 49 | 72    | 77   | 55    | 14 | 28    | 76 | 43    | 79.77       |
| S          | AGE   | 17 | 25    | 77  | 37    | 18 | 00    | 76  | 99    | 19 | 48    | 77 | 40    | 19 | 85    | 75       | 51    | 49 | 35    | 76   | 99    | 14 | 28    | 77 | 55    | 82.37       |
| Cora G     | CN    | 17 | 99    | 78  | 11    | 17 | 81    | 77  | 92    | 18 | 55    | 77 | 74    | 20 | 41    | 79       | 22    | 47 | 31    | 80   | 52    | 14 | 28    | 79 | 03    | 84.60       |
|            | G IN  | 16 | 69    | 76  | 44    | 18 | 74    | 80  | 52    | 17 | 44    | 76 | 25    | 19 | 29    | 76       | 62    | 48 | 79    | 78   | 85    | 14 | 28    | 76 | 06    | 81.63       |
| G          | AT    | 16 | 51    | 61  | 82    | 25 | 00    | 62  | 27    | 19 | 24    | 64 | 70    | 18 | 18    | 63       | 48    | 20 | 91    | 62   | 73    | 16 | 67    | 62 | 27    | 66.42       |
| S          | AGE   | 16 | 66    | 68  | 48    | 16 | 67    | 68  | 64    | 22 | 12    | 69 | 39    | 22 | 42    | 69       | 85    | 22 | 88    | 71   | 52    | 16 | 67    | 69 | 39    | 72.57       |
| Citeseer G | CN    | 28 | 18    | 60  | 00    | 16 | 67    | 61  | 97    | 20 | 45    | 65 | 45    | 19 | 70    | 64       | 24    | 21 | 06    | 64   | 70    | 16 | 67    | 63 | 18    | 68.03       |
|            | G IN  | 16 | 66    | 64  | 39    | 16 | 67    | 63  | 94    | 20 | 15    | 59 | 85    | 18 | 64    | 63       | 64    | 22 | 12    | 60   | 30    | 16 | 67    | 61 | 81    | 67.38       |
| G          | AT    |    | 29.55 |     | 92.07 |    | OOM   |     | 90.86 |    | OOT   |    | 91.64 |    | OOT   |          | 91.64 |    | 35.79 |      | 92.52 |    | OOT   |    | 93.74 | 89.49       |
| S          | AGE   |    | 26.75 |     | 87.89 |    | OOM   |     | 91.05 |    | OOT   |    | 86.64 |    | OOT   |          | 86.64 |    | 32.92 |      | 90.44 |    | OOT   |    | 86.01 | 90.03       |
| Syn 4 G    | CN    |    | 28.85 |     | 51.97 |    | OOM   |     | 19.58 |    | OOT   |    | 18.29 |    | OOT   |          | 18.92 |    | 33.80 |      | 26.60 |    | OOT   |    | 36.85 | 21.43       |
|            | GIN   |    | 28.50 |     | 65.61 |    | OOM   |     | 31.06 |    | OOT   |    | 26.51 |    | OOT   |          | 26.56 |    | 34.03 |      | 46.40 |    | OOT   |    | 47.10 | 29.35       |
| G          | AT    |    | 44.00 |     | 86.80 |    | 43.60 |     | 86.60 |    | 30.00 |    | 78.75 |    | 55.40 |          | 92.80 |    | 36.20 |      | 93.60 |    | 31.80 |    | 92.80 | 97.20       |
| S          | AGE   |    | 41.00 |     | 93.80 |    | 41.40 |     | 93.60 |    | 33.75 |    | 88.75 |    | 57.60 |          | 94.00 |    | 35.20 |      | 94.80 |    | 28.20 |    | 95.60 | 97.40       |
| Syn 6 G    | CN    |    | 43.60 |     | 88.80 |    | 42.20 |     | 87.40 |    | 26.25 |    | 81.25 |    | 55.60 |          | 92.40 |    | 31.40 |      | 94.40 |    | 25.20 |    | 94.00 | 99.40       |
|            | GIN   |    | 39.60 |     | 89.00 |    | 40.40 |     | 86.60 |    | 21.25 |    | 82.50 |    | 55.20 |          | 91.80 |    | 30.00 |      | 94.60 |    | 30.40 |    | 92.00 | 98.80       |
| G          | AT    |    | 29.55 |     | 99.75 |    | 33.75 |     | 88.75 |    | 88.25 |    | 99.25 |    | 88.25 |          | 99.25 |    | 26.00 |      | 85.50 |    | 94.00 |    | 96.00 | 98.50       |
| S          | AGE   |    | 26.75 |     | 100.0 |    | 32.50 |     | 100.0 |    | 88.75 |    | 99.50 |    | 88.75 |          | 99.50 |    | 26.75 |      | 100.0 |    | 92.50 |    | 100.0 | 100.0       |
| Syn 8 G    | CN    |    | 28.85 |     | 98.75 |    | 31.75 |     | 99.75 |    | 88.75 |    | 99.00 |    | 88.75 |          | 99.00 |    | 28.50 |      | 99.25 |    | 95.00 |    | 100.0 | 100.0       |
|            | GIN   |    | 28.50 |     | 50.00 |    | 30.50 |     | 91.00 |    | 82.25 |    | 91.50 |    | 82.25 |          | 91.50 |    | 27.25 |      | 81.75 |    | 91.75 |    | 92.25 | 78.25       |

Table 9: Computational time for learning graph structures using GraphFLEx (GFlex) with existing methods (Vanilla referred to as Van.). The experimental setup involves treating 50% of the data as static, while the remaining 50% of nodes are treated as incoming nodes arriving in 25 different timestamps. The best times are highlighted by color Green. OOM and OOT denote out-of-memory and out-of-time, respectively.

|     | Data | Van. | ANN GFlex | Van. | KNN GFlex | Van.  | log-model GFlex | Van. | l2-model GFlex | Van. | COVA GFlex | Van. | large-model GFlex |
|-----|------|------|-----------|------|-----------|-------|-----------------|------|----------------|------|------------|------|-------------------|
| Syn | 1    | 19.4 | 9.8       | 2.5  | 10.5      | 2418  | 56.4            | 37.2 | 8.8            | 3.5  | 8.3        | 205  | 9.4               |
| Syn | 2    | 47.3 | 16.9      | 6.6  | 18.3      | 14000 | 144             | 214  | 22.6           | 20.3 | 18.6       | 1259 | 16.4              |
| Syn | 5    | 5.1  | 11.5      | 0.8  | 7.3       | 57.4  | 28              | 1.1  | 5.8            | 0.2  | 4.8        | 3.2  | 5.3               |
| Syn | 6    | 16.6 | 9.9       | 2.8  | 11.4      | 1766  | 96.3            | 193  | 101            | 5.3  | 8.9        | 324  | 9.6               |
| Syn | 7    | 10.6 | 7.4       | 1.4  | 8.9       | 704   | 85.2            | 10.3 | 7.9            | 0.9  | 6.4        | 36.5 | 8.2               |
| Syn | 8    | 19.6 | 11.2      | 2.5  | 11.7      | 2416  | 457             | 37.2 | 17.0           | 3.4  | 10.9       | 204  | 11.7              |

## H. Visualization of Growing graphs

This section helps us visualize the phases of our growing graphs. We have generated a synthetic graph of 60 nodes using PyGSP-Sensor and HE methods mentioned in Appendix [D.](#page-13-1) We then added 40 new nodes denoted using black color in these existing graphs at four different timestamps. Figure [10](#page-18-1) and Figure [11](#page-18-2) shows the learned graph structure after each timestamp for two different Synthetic graphs.

994

996

998

![](_page_18_Diagram_3.jpeg)

1029

1034 1036 This section involves a comparison of the graph structure learned from GraphFlex with existing methods. Six nodes were randomly selected and considered as new nodes. Figure [12](#page-18-3) visually depicts the structures learned using GraphFlex compared to other methods. It is evident from the figure that the structure known with GraphFlex closely resembles the original graph structure. Figure [13](#page-19-0) shows the original structure of Zachary's karate club network [\(Zachary,](#page-10-16) [1977\)](#page-10-16). We assumed six random nodes to be dynamic nodes, and the structure learned using GraphFlex compared to existing methods is shown in Figure [12.](#page-18-3)

1039

1040 1041 1042 Figure [14](#page-19-1) shows the PHATE [\(Moon et al.,](#page-9-19) [2019\)](#page-9-19) visualization of clusters learned using GraphFLEx's clustering module Mclust for Xin, MNIST, and Baron − Human datasets.

PyGsp

(a) Initial graph G<sup>0</sup> (b) α= 10, G<sup>1</sup> (c) α= 20, G<sup>2</sup> (d) α = 30, G<sup>3</sup> (e) α = 40, G<sup>4</sup>

Figure 10: This figure illustrates the growing structure learned using GraphFlex for dynamic nodes. New nodes are denoted using black color, and α denotes number of new nodes. *PyGsp* denotes type synthetic graph. HE

(a) Initial graph G<sup>0</sup> (b) α = 10, G<sup>1</sup> (c) α = 20, G<sup>2</sup> (d) α= 30, G<sup>3</sup> (e) α = 40, G<sup>4</sup>

Figure 11: This figure illustrates the growing structure learned using GraphFlex for dynamic nodes. New nodes are denoted using black color, and α denotes the number of new nodes. *HE* denotes the type of synthetic graph.

Figure 12: This figure compares the structures learned on Zachary's karate dataset when existing methods are employed with GraphFlex and when existing methods are used individually. We consider six nodes, denoted in black, as dynamic nodes.

# I. Structure Comparison on Karate Dataset

# J. Clustering Quality

![](_page_19_Diagram_2.jpeg)

Figure 13: Original Karate Graph

![](_page_19_Figure_4.jpeg)

Figure 14: PHATE visualization of clusters learnt using GraphFlex clustering module for scRNA-seq datasets.