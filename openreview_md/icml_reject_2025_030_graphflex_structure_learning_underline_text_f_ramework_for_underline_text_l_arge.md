# Graphflex: Structure Learning Framework For Large Expanding Graphs

## Anonymous Authors1 Abstract

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031

## 032

033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 Graph structure learning is a fundamental problem critical for interpretability and uncovering relationships in data. While graphical data is central to information representation, inferring graph structures remains challenging. Existing methods falter with expanding graphs, requiring costly relearning of the entire structure for new nodes, and face severe computational and memory demands on large graphs. To overcome these challenges, we propose **GraphFLEx**: a unified framework for structure learning in Large and Expanding Graphs. GraphFLEx efficiently limits potential connections to relevant nodes by leveraging clustering and coarsening techniques, significantly reducing computational costs and enhancing scalability. **GraphFLEx** provides 48 flexible methods for graph structure learning by integrating diverse learning, coarsening, and clustering approaches.

Extensive experiments with various GNN models demonstrate its effectiveness. Our code is available here.

## 1. Introduction

Graph representations capture relationships between entities, vital across diverse fields like biology, finance, sociology, engineering, and operations research (Zhou et al., 2020; Fout et al., 2017; Wu et al., 2020). While some relationships, such as social connections or sensor networks, are directly observable, many, including gene regulatory networks, scene graph generation (Gu et al., 2019), brain networks, (Zhu et al., 2021) and drug interactions, require inference (Allen et al., 2012). Even when available, graph data often contains noise, requiring denoising and recalibration. Thus, inferring graph structures becomes crucial, often surpassing the choice of graph or algorithm itself. Graph Structure Learning (GSL) offers a solution, enabling the construction and refinement of graph topologies. GSL has been widely studied in both supervised and unsupervised 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 contexts (Liu et al., 2022; Chen & Wu, 2022). In supervised GSL (s-SGL), the adjacency matrix and Graph Neural Networks (GNNs) are jointly optimized for a downstream task, such as node classification. Notable examples of s-GSL
include *NodeF ormer* (Wu et al., 2022), P ro−GNN (Jin et al., 2020), *W SGNN* (Lao et al., 2022), and *SLAP S*
(Fatemi et al., 2021). Unsupervised GSL (u-SGL), on the other hand, focuses solely on learning the underlying graph structure, typically through adjacency or Laplacian matrices. Methods in this category include approximate nearest neighbours (A−NN) (Dong et al., 2011; Muja & Lowe, 2014), knearest neighbours (k−NN) (MacQueen et al., 1967; Wang & Zhang, 2006), covariance estimation (*emp.Cov.*) (Hsieh et al., 2011), graphical lasso (*GLasso*) (Friedman et al., 2008), and signal processing techniques like l2-model,logmodel, and *large*-model (Dong et al., 2016; Kalofolias, 2016). While s-SGL methods offer promising results, they have limitations: (1) they rely on label information, restricting their applicability in settings without annotations; (2) they are often task-specific, optimizing for node classification rather than general graph topology (Liu et al., 2022). These issues are avoided in u-SGL approaches, which are the focus of this work. However, both s-SGL and u-SGL face challenges when applied to large-scale or expanding datasets.

Tim e i n Second s Time in Seconds vs Number of Nodes (Each Line Represents an Experiment)

GraphFlex ANN GraphFlex KNN GraphFlex Log model GraphFlex L2 model GraphFlex Emp. Covar. GraphFlex Large model Time in Seconds vs Number of Nodes (Each Line Represents an Experiment)

ANN KNN Log model L2 model Emp. Covar. Large model 10000 20000 30000 40000 50000 Number of Nodes 0 150 300 450 600 750 900 1050 1200 10000 20000 30000 40000 50000 Number of Nodes 0 150 300 450 600 750 900 1050 1200 Tim e i n Second s

(a) GraphFLEx
(b) Vanilla
Figure 1: High computational time required to learn graph structures using existing methods, whereas GraphFLEx effectively controls computational growth, achieving near-linear scalability. Notably, Vanilla KNN failed to construct graph structures with fewer than 10k nodes due to memory limitations.

As contemporary datasets grow in size, scalability becomes a critical challenge, with existing methods proving too computationally expensive for large-scale graphs. In such cases, Approximate Nearest Neighbours (A−NN), with time com-

GraphFLEx: Structure Learning Framework for Large Expanding Graphs Graph Coarsening t1 t6 t4 24 1 1 2 3 2 15 3 15 1 19 2 C1 C1 15 22 1 2 11 12 13 15 16 17 14 18 C2 16 6 10 7 8 3 19 19 20 20 21 24 21 23 22 3 4 5 17 20 7 8 4 5 7 7 8 8 6 14 10 9 11 12 13 11 11 12 12 13 14 14 18 9 9 9 t3 t2 10 10 C3 21 13 C3 GNN Clustering Graph Communities t5 23 Most relevant nodes formation Most relevant communities Super-nodes Incoming Nodes 15 15 2 2 2 2 20 11 12 13 15 16 17 14 18 3 3 1 2 19 3 4 5 7 8 6 10 9 1 1 1 1 G
rap h L
ea rni n g G
rap h L
ea rni n g 7 7 7 8 8 8 19 19 19 19 7 8 21 9 11 2 20 21 1 14 9 19 7 8 20 21 21 12 11 9 9 9 13 14 10 10 13 12 20 21 11 12 11 11 12 21 21 12 Projection back to original space 20 20

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 plexity O(N log(N)), is often the only feasible solution. In contrast, methods like k-NN, log-model, and l2-model are significantly more costly, with time complexities exceeding O(N2).

The aforementioned techniques are ineffective for learning large-scale graphs because they consider the entire collection of nodes to determine connections for every individual node. All nodes, however, only have connections to a very small set of nodes. Therefore, we need to devise a method that can refine the entire graph's node set to a smaller subset of potential node sets, with the aim of identifying feasible connections. Additionally, expanding graphs where new nodes continuously arrive further complicates the issue, as existing methods require re-learning the entire graph structure with each new node (Khazane et al., 2019; Holme & Saramaki ¨ , 2012). This makes them inefficient for expanding data. To address these challenges, we propose GraphFLEx, a comprehensive framework that tackles both scalability for large datasets and adaptability for growing graphs.

As shown in Figure 2, **GraphFLEx** comprises three key modules: (i) Graph Clustering, (ii) Graph Coarsening, and
(iii) Graph Learning. By leveraging clustering and coarsening, GraphFLEx significantly reduces computational overhead by restricting possible connections to only relevant nodes. Figure 1 compares the graph structure learning time, highlighting GraphFLEx's efficiency over existing methods. Key contributions of GraphFLEx include: Key Contributions and Novelty.

- We provide strong *theoretical guarantees* that the structure learned from a small subset of nodes is equivalent to that learned from the full set. This is supported by empirical results using real-world and synthetic datasets, demonstrating the effectiveness of GraphFLEx across diverse graph structures.

- GraphFLEx is composed of independently operating modules, allowing the creation of new learning frameworks by modifying any of its three modules. It currently supports 48 distinct methods for learning graph structure, offering flexibility across various domains.

- GraphFLEx efficiently handles large-scale and expanding graphs, enhancing scalability for graph learning tasks.

- GraphFLEx serves as a *comprehensive framework* applicable individually for clustering, coarsening, and learning tasks.

## 2. Problem Formulation And Background

A graph G is represented using G(*V, A, X*) where V = {v1, v2*...v*N } is the set of N nodes, each node vi has a d−dimensional feature vector xiin X ∈ R
N×dand A ∈ R
N×N is adjacency matrix representing connection between i th and j th nodes when entry Aij > 0. An expanding graph EG can be considered a variant of graph G where nodes v now have an associated timestamp τv. We can represent a expanding graph as a sequence of graphs, i.e., EG = {G0, G1*, ...*GT } where {G0 ⊆ G1.... ⊆ GT } at τ ∈ {0*, ...T*} timestamps. New nodes arriving at different timestamps are seamlessly integrating into initial graph G0.

Problem statement. Given a partially known or missing graph structure, our goal is to incrementally learn the whole graph, i.e., learn adjacency or laplacian matrix. Specifically, we consider two unsupervised GSL tasks: Goal 1. *Large Datasets with Missing Graph Structure:* In this setting, the graph structure is entirely unavailable, and existing methods are computationally infeasible for learning the whole graph in a single step. To address this issue, we first randomly partition the dataset into exclusive subsets. We then learn the initial graph G0(V0, X0) over a small subset of nodes and incrementally expand it by integrating additional partitions, ultimately reconstructing the full graph GT .

Goal 2. **Partially Available Graph:** In this case, we only have access to the graph Gt at timestamp t, with new nodes arriving over time. The goal is to update the graph incrementally to obtain GT , without re-learning it from scratch at each timestamp.

GraphFlex addresses these challenges with a unified framework, outlined in Section 3. Before delving into the framework, we review some key concepts.

## 2.1. Graph Reduction

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Graph reduction encompasses sparsification, clustering, coarsening, and condensation (Hashemi et al., 2024). Graph- Flex employs clustering and coarsening to refine the set of relevant nodes for potential connections. Graph Clustering. Graphs often exhibit global heterogeneity with localized homogeneity, making them well-suited for clustering (Fortunato, 2010). Clusters capture higher-order structures, aiding graph learning. Methods like DMoN (Tsitsulin et al., 2023) use GNNs for soft cluster assignments, while Spectral Clustering (SC) (Kamvar et al., 2003) and K-means (Wagstaff et al., 2001; MacQueen et al., 1967) efficiently detect communities. DiffPool (Bruna et al., 2014; Defferrard et al., 2016) applies SC for pooling in GNNs.

Graph Coarsening. Graph Coarsening (GC) reduces a graph G(*V, E, X*) with N nodes and features X ∈ R
N×d into a smaller graph Gc(V , 
e E, e Xe) with n ≪ N nodes and Xe ∈ R
n×d. This is achieved via learning a coarsening matrix P ∈ R
n×N , mapping similar nodes in G to super-nodes in Gc, ensuring Xe = PX while preserving key properties
(Loukas, 2019; Kataria et al., 2023; Kumar et al., 2023; Kataria et al., 2024).

## 2.2. Unsupervised Graph Structure Learning

Unsupervised graph learning spans from simple k-NN weighting (Wang & Zhang, 2006; Zhu et al., 2003) to advanced statistical and graph signal processing (GSP) tech-

| Table 1: Unsupervised Graph Structure Learning Methods   |                   |                                                      |
|----------------------------------------------------------|-------------------|------------------------------------------------------|
| Method                                                   | Time Complexity   | Formulation                                          |
| GLasso                                                   | O(N 3 )           | maxΘ log det Θ −tr(ΣΘ) ˆ − ρ∥Θ∥1                     |
| 2 )                                                      | minW∈W ∥W ◦ Z∥1,1 |                                                      |
| log-model                                                | O(N               | −α1 T log(W1) + β ∥W∥ 2 2 F                          |
| l2-model                                                 | O(N 2 )           | minW∈W ∥W ◦ Z∥1,1 +α∥W1∥ 2 + α∥W∥ 2 F +1{∥W∥1,1 = n} |
| large-model                                              | O(N log(N))       | minW∈W˜ ∥W ◦ Z∥1,1                                   |
| −α1 T log(W1) + β ∥W∥ 2 2 F                              |                   |                                                      |

niques. Statistical methods, also known as probabilistic graphical models, assume an underlying graph G governs the joint distribution of data X ∈ R
N×d(Koller & Friedman, 2009; Banerjee et al., 2008; Friedman et al., 2008). Some approaches (Dempster, 1972) prune elements in the inverse sample covariance matrix Σ =
b 1 d−1XXTand sparse inverse covariance estimators, such as Graphical Lasso (GLasso) (Friedman et al., 2008): maximizeΘ log det Θ −
tr(ΣΘ
b )−ρ∥Θ∥1, where Θ is the inverse covariance matrix.

However, these methods struggle with small sample sizes. Graph Signal Processing (GSP) techniques analyze signals on known graphs, ensuring properties like smoothness and sparsity. Signal smoothness on a graph G is quantified by the Laplacian quadratic form:

$$Q(\mathbf{L})=\mathbf{x}^{T}\mathbf{L}\mathbf{x}={\frac{1}{2}}\sum_{i,j}w_{i j}(\mathbf{x}(i)-\mathbf{x}(j))^{2}.$$

For a set of vectors X, smoothness is measured using the Dirichlet energy (Belkin et al., 2006): tr(XTLX). Stateof-the-art methods (Dong et al., 2016; Kalofolias, 2016; Hu et al., 2013) optimize Dirichlet energy while enforcing sparsity or specific structural constraints. Table 1 compares various graph learning methods based on their formulations and time complexities. Remark 1. Graph Structure Learning (GSL) differs significantly from Continual Learning (CL) (Van de Ven & Tolias, 2019; Zhang et al., 2022; Parisi et al., 2019) and Dynamic Graph Learning (DGL) (Kim et al., 2022; Wu et al., 2023; You et al., 2022), as discussed in Appendix C.

## 3. Graphflex

In this section, we introduce GraphFLEx, which has three main modules:
- **Graph Clustering.** Identifies communities and extracts higher-order structural information,
- **Graph Coarsening.** Is used to coarsen down the desired community, if the community itself is large,
- **Graph Learning.** Learns the graph's structure using a limited subset of nodes from the clustering and coarsening modules, *enabling scalability*.

For more details, see Algorithm 1 in Appendix E.

## 3.1. Incremental Graph Learning For Large Datasets

Real-world graph data is continuously expanding. For instance, e-commerce networks accumulate new clicks and purchases daily (Xiang et al., 2010), while academic networks grow with new researchers and publications (Wang et al., 2020). This expanding behaviour suggests that large graphs can be efficiently processed by learning them incrementally in smaller segments.

Given a large dataset L(VL, XL), where VL is the node set and XL represents node features, we define an *expanding* dataset setting LE = {ET
τ=0}. Initially, L is split into: (i)
a static dataset E0(V0, X0) and (ii) an *expanding dataset* E = {Eτ (Vτ , Xτ )}
T
τ=1. Both *Goal* 1 (large datasets with missing graph structure) and *Goal* 2 (partially available graphs with incremental updates), discussed in Section 2, share the common objective of incrementally learning and updating the graph structure as new data arrives. Graph- FLEx handles these by decomposing the problem into two key components:
165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219
- **Initial Graph** G0(V0, A0, X0): For *Goal* 1, where the graph structure is entirely missing, E0(V0, X0) is used to construct G0 from scratch using structure learning methods (see Section 2.2). For *Goal* 2, the initial graph G0(V0, A0, X0) is already available and serves as the starting point for incremental updates.

- **Expanding Dataset** E = {Eτ (Vτ , Xτ )}
T
τ=1: In both cases, E consists of incoming nodes and features arriving over T timestamps. These nodes are progressively integrated into the existing graph, enabling continuous adaptation and growth.

The partition is controlled by a parameter r, which determines the proportion of static nodes: r =
∥V0∥
∥VL∥
. For example, r = 0.2 implies that 20% of VL is treated as static, while the remaining 80% arrives incrementally over T timestamps. In our experiments, we set r = 0.5 and T = 25.

Remark 2. We can learn Gτ (Vτ , Aτ , Xτ ) by aggregating Eτ nodes in Gτ−1 graph. Our goal is to learn GT (VT , AT , XT )
after T
th-timestamp.

## 3.2. Detecting Communities

From the static graph G0, our goal is to learn higher-order structural information, identifying potential communities to which incoming nodes (V ∈ V τ ) may belong. We train the community detection/clustering model Mclust once using G0, allowing subsequent inference of clusters for all incoming nodes. While our framework supports spectral and k-means clustering, our primary focus has been on Graph Neural Network (GNN)-based clustering methods. Specifically, we use DMoN (Tsitsulin et al., 2023; Bianchi et al., 2020; Bianchi, 2022), which maximizes spectral modularity. Modularity (Newman, 2006) measures the divergence between intra-cluster edges and the expected number. These methods use a GNN layer to compute the partition matrix C = softmax(MLP(*X, θ* e MLP)) ∈ R
N×K,
where K is the number of clusters and Xe is the updated feature embedding generated by one or more message-passing layers. To optimize the C matrix, we minimize the loss function ∆(C; A) = −
1 2m Tr(C
T BC) +
√k n |ΣiC
T
i|F − 1, which combines spectral modularity maximization with regularization to prevent trivial solutions, where B is the modularity matrix (Tsitsulin et al., 2023). Our static graph G0 and incoming nodes E follow Assumption 1.

Assumption 1. We assume that the generated graphs adhere to the Degree-Corrected Stochastic Block Model (DC- SBM) (Zhao et al., 2012), where intra-class (or intracommunity) links are more likely than inter-class links. For more details on DC-SBM, see Appendix A.

Lemma 1. Mclust **Consistency.** *We adopt the theoretical* framework of (Zhao et al., *2012) for a DC-SBM with* N nodes and k classes. The edge probability matrix is parameterized as PN = ρN P*, where* P ∈ R
k×kis a symmetric matrix containing the between/within community edge probabilities and it is independent of N, ρN = λN /N, and λN is the average degree of the network. Let yˆN = [ˆy1, yˆ2*, . . . ,* yˆN ]
denote the predicted class labels, and let CˆN be the corresponding N × k one-hot matrix. Let the true class label matrix is CN , and µ is any k × k permutation matrix. Under the adjacency matrix A(N), the global maximum of the objective ∆(·; A(N)) is denoted as Cˆ∗N . The consistency of class predictions is defined as:

## 1. **Strong Consistency.**

$$i s t e m c y.$$
PN
  **Comolem:**  $$\left[\min_{\mu}\|\hat{C}_{N}^{*}\mu-C_{N}\|_{F}^{2}=0\right]\to1\quad as\ N\to\infty,$$
2. _Weak Consistency._ $$\forall\varepsilon>0,P_{N}\left[\min_{\mu}\frac{1}{N}\|\hat{C}_{N}^{*}\mu-C_{N}\|_{F}^{2}<\varepsilon\right]\to1\;as\;N\to\infty.$$

where ∥ · ∥F is the Frobenius norm. Under the conditions of Theorem 3.1 from (Zhao et al., *2012):*
- The Mclust *objective is strongly consistent if* λN / log(N) → ∞*, and*
- It is weakly consistent when λN → ∞. Remark 3. **Structure Learning within Communities.** In GraphF LEx, we focus on learning the structure within each community rather than the structure of the entire dataset at once. Strong consistency ensures perfect community recovery, meaning no inter-community edges exist representing the ideal case. Weak consistency, however, allows for a small fraction (ϵ) of inter-community edges, where ϵ is controlled by ρn in Pn = ρnP, influencing graph sparsity. By Lemma 1 and Assumption 1, stronger consistency leads to more precise structure learning, whereas weaker consistency permits a limited number of inter-community edges.

## 3.3. Learning Graph Structure On A Coarse Graph

After training Mclust, we identify communities for incoming nodes, starting with τ = 1. Once assigned, we determine significant communities those with at least one incoming node and learn their connections to the respective community subgraphs. For large datasets, substantial community sizes may again introduce scalability issues. To mitigate this, we first coarsen the large community graph into a smaller graph and use it to identify potential connections for incoming nodes. This process constitutes the second module of GraphFLEx, denoted as Mcoar, which employs LSH-based hashing for graph coarsening. The supernode index for i th node is given as:

$${\mathcal{H}}_{i}=m a x{O c u r a n c e}\left\{\left[{\frac{1}{r}}\cdot({\mathcal{W}}\cdot X_{i}+b)\right]\right\}\quad(1)$$

where r (bin width) controls the coarsened graph size, W represents random projection matrix, X is the feature matrix, and b is the bias term. For further details, refer to UGC (Kataria et al., 2024). After coarsening the i th community (Ci), Mcoar(Ci) = {Pi, Si} yields a partition matrix Pi ∈ R
∥Si∥×∥Ci∥and a set of coarsened supernodes
(Si), as discussed in Section 2. Definition 1. The neighborhood of a set of nodes Eiis defined as the union of the top k most similar nodes in Ci for each node v ∈ Ei*, where similarity is measured by the* distance function d(v, u). A node u ∈ Ciis considered part of the neighborhood if its distance d(v, u) *is among the* k smallest distances for all u
′ ∈ Ci.

$\mathcal{N}_{k}(\mathcal{E}_{i})=\bigcup\limits_{v\in\mathcal{E}_{i}}\{u\in C_{i}\mid d(v,u)\leq top\text{-}k[d(v,u^{\prime}):u^{\prime}\in C_{i}]\}$.  
Goal 3. The neighborhood of incoming nodes Nk(Ei) represents the ideal set of nodes where the incoming nodes Ei are likely to establish connections when the entire community is provided to a structure learning framework.. A robust coarsening framework must reduce the number of nodes within each community Ci while ensuring that the neighborhood of the incoming nodes is preserved.

## 3.4. Graph Learning Only With Potential Nodes

As we now have a smaller representation of the community, we can employ any graph learning algorithms discussed in Section 2.2 to learn a graph between coarsened supernodes Si and incoming nodes (V
i τ ∈ Vτ ). This is the third module of GraphFLEx, i.e., graph learning; we denote it as Mgl. The number of supernodes in Siis much smaller compared to the original size of the community, i.e., ∥Si*∥ ≪ ∥*Ci∥;
scalability is not an issue now. We learn a small graph first using Mgl(Si, Xiτ) = Geiτ(V
c τ, Acτ) where Xiτrepresents features of new nodes belonging to i th community at time τ , Geiτ(V
c τ, Acτ) representing the graph between supernodes and incoming nodes. Utilizing the partition matrix Pi obtained from Mcoar, we can precisely determine the set of nodes associated with each supernode. For every new node V ∈ V
i τ, we identify the connected supernodes and subsequently select nodes within those supernodes. This subset of nodes is denoted by ωV i τ
, the sub-graph associated with ωV i τ represented by G
i τ−1(ωV i τ
) then undergoes an additional round of graph learning Mgl(G
iτ−1(ωV i τ
), Xiτ),
ultimately providing a clear and accurate connection of new nodes V
i τ with nodes of Gτ−1, ultimately updating it to Gτ . This multi-step approach, characterized by coarsening, learning on coarsened graphs, and translation to the original graph, ensures scalability.

Theorem 1. **Neighborhood Preservation.** Let Nk(Ei) denote the neighborhood of incoming nodes Eifor the i th community. With partition matrix Pi and Mgl(Si, Xiτ) =
G
c τ
(V
c τ
, Acτ
) we identify the supernodes connected to incoming nodes Ei and subsequently select nodes within those supernodes; this subset of nodes is denoted by ωV i τ
. Formally,

$$\omega_{V_{\tau}^{i}}=\bigcup_{v\in{\mathcal E}_{i}}\Big\{\bigcup_{s\in S_{i}}\{\pi^{-1}(s)|A_{\tau}^{c}(v,s)\neq0\}\Big\}$$
$$\mathsf{V}_{k}({\mathcal{E}}_{i})\subseteq\omega_{V_{\tau}^{i}}$$
$\mathbf{a}=\mathbf{b}$. 
Then, with probability Π{c∈ϕ}p(c)*, it holds that* Nk(Ei) ⊆ ωV iτ
where
$p(c)\leq1-\frac{2}{\sqrt{2\pi}}\frac{c}{r}\left[1-e^{-r^{2}/(2e^{2})}\right]$, $\frac{1}{r}$
and ϕ *is a set containing all pairwise distance values*(c = ∥v−u∥)
between every node v ∈ Ei *and the nodes* u ∈ ωV i τ
. Here, π
−1(s)
denotes the set of nodes mapped to supernode s, r *is the bin-width* hyperparameter of Mcoar.

$$\operatorname{lim}_{\mathbf{x}\to\mathbf{0}}.$$
$\square$

## Proof. The Proof Is Deferred In Appendix B.

Remark 4. Theorem 1 establishes that, with a constant probability of success, the neighborhood of incoming nodes Nk(Ei) can be effectively recovered using the GraphFLEx multistep approach, which involves coarsening and learning on the coarsened graph, i.e., Nk(Ei) ⊆ ωV i τ
. The set ωV i τ
,
estimated by GraphFLEx, identifies potential candidates where incoming nodes are likely to connect. The probability of failure can be reduced by regulating the average degree of connectivity in Mgl(Si, Xiτ
) = G
c τ
(V
c τ
, Acτ
). While a fully connected G
cτensures all nodes in the community are candidates, it significantly increases computational costs for large communities.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 Table 2: Time complexity analysis of GraphFLEx. Here, N is the number of nodes in the graph, k is the number of nodes in the static subgraph used for clustering (k ≪ N), and c represents the number of detected communities. kτ denotes the number of nodes at timestamp τ . Finally, α = ∥S
i τ ∥ + ∥Eiτ ∥ is the sum of coarsened and incoming nodes in the relevant community at τ timestamp.

| Mclust                | Mcoar   | Mgl   | GraphFLEx   |                       |             |               |
|-----------------------|---------|-------|-------------|-----------------------|-------------|---------------|
| Best (kNN-UGC-ANN)    | O(k 2 ) | O  kτ        | O(α log α)  | O(k 2 + kτ + α log α) |             |               |
| c                     | c       |       |             |                       |             |               |
|  kτ 2 ∥S                       |          | O(α   | 2 ∥S             |                       |             |               |
| Worst (SC-FGC-GLasso) | O(k 3 ) | O     | i τ ∥       | 3 )                   | O(k 3 +  kτ | i τ ∥ + α 3 ) |
| c                     | c       |       |             |                       |             |               |

## 3.5. Graphflex Offering Multiple Sgl Frameworks 4. Experiments

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Each module in Figure 3, controls distinct properties: clustering influences community detection, coarsening governs supernode formation to reduce graph complexity, and the learning module enforces diverse structural properties. Altering any of these modules results in a new graph learning method. Currently, we support 48 different graph learning configurations, and this number scales exponentially with the addition of new methods to any module. The number of possible frameworks is given by α × β × γ, where α, β, and γ represent the number of clustering, coarsening, and learning methods, respectively. We evaluate the run-time complexity of GraphFLEx in two scenarios: (a) the worst-case scenario, where computationally intensive clustering and coarsening modules are selected, providing an upper bound on time complexity, and (b) the best-case scenario, where the most efficient modules are chosen. Table 2 summarizes the analysis. The run time of GraphFLEx is primarily determined by the learning module (Mgl). GraphFLEx computational time is always bounded by existing approaches, as it operates on a significantly reduced graph space, ensuring efficient performance, especially for larger or expanding graphs. This is also illustrated in Table 3.

## 4.1. Node Classification Accuracy 3.6. Run Time Analysis

In this section, we conclude the experiments to back up our findings. Tasks and Datasets. The experiments focus on four key aspects of GraphFLEx: its computational efficiency, scalability in handling large graphs, the quality of the learned graph structure, and its ability to efficiently handle expanding graphs. To validate the characteristics of GraphFLEx, we conduct extensive experiments on 22 different datasets, including (a) datasets that already have a complete graph structure (allowing comparison between the learned and the original structure), (b) datasets with missing graph structures, (c) synthetic datasets, and (d) small datasets for visualizing the graph structure. More details about datasets are presented in Table 6 in Appendix D.

System Specifications: All the experiments conducted for this work were performed on an Intel Xeon W-295 CPU and 64GB of RAM desktop using the Python environment. Computational Efficiency. Existing methods like k-NN and log-model struggle to learn graph structures even for 20k nodes due to out-of-memory (OOM) or out-of-time (OOT) issues, while l2-model and *large*-model struggle beyond 50k nodes. Although A-NN and emp-Covar. are faster, GraphFLEx outperforms them on sufficiently large graphs (Table 3). While traditional methods may be efficient for small graphs, GraphFLEx scales significantly better, excelling on large datasets like Pubmed and *Syn 5*, where most methods fail. It accelerates structure learning, making A-NN 3× faster and emp-Covar. 2× faster. Experimental Setup. We now evaluate the prediction performance of GNN models when trained on graph structures learned from three distinct scenarios: 1) Original Structure: GNN models trained on the original graph structure, which we refer to as the Base Structure, **2) GraphFLEx** Structure: GNN models trained on the graph structure learned from GraphFLEx, and **3)Vanilla Structure:** GNN models trained on the graph structure learned from other existing methods. For each scenario, a unique graph structure is obtained. We trained GNN models on each of these three structure. For more details on GNN model parameters, see Appendix F.

| Data     | ANN   | KNN   | log-model   | l2-model   | emp-Covar.   | large-model   |       |      |      |      |      |      |
|----------|-------|-------|-------------|------------|--------------|---------------|-------|------|------|------|------|------|
| Cora     | 335   | 100   | 8.4         | 36.1       | 869          | 81.6          | 424   | 55   | 8.6  | 30   | 2115 | 18.4 |
| Citeseer | 1535  | 454   | 21.9        | 75         | 1113         | 64.5          | 977   | 54.0 | 14.7 | 59.2 | 8319 | 43.9 |
| DBLP     | 2731  | 988   | OOM         | 270        | 77000        | 919           | OOT   | 1470 | 359  | 343  | OOT  | 299  |
| CS       | 22000 | 12000 | OOM         | 789        | OOT          | 838           | 32000 | 809  | 813  | 718  | OOT  | 1469 |
| PubMed   | 770   | 227   | OOM         | 164        | OOT          | 176           | OOT   | 165  | 488  | 299  | OOT  | 262  |
| Phy.     | 61000 | 21000 | OOM         | 903        | OOT          | 959           | OOT   | 908  | 2152 | 1182 | OOT  | 2414 |
| Syn 3    | 95    | 37    | OOM         | 30         | 58000        | 346           | 859   | 53   | 88   | 59   | 5416 | 42   |
| Syn 4    | 482   | 71    | OOM         | 73         | OOT          | 555           | OOT   | 145  | 2072 | 1043 | OOT  | 392  |

Table 4: Node classification accuracies on different GNN models using GraphFLEx (GFlex) with existing Vanilla (Van.) methods. The experimental setup involves treating 70% of the data as static, while the remaining 30% of nodes are treated as new nodes coming in 25 different timestamps. The best and the second-best accuracies in each row are highlighted by dark and lighter shades of Green, respectively. GraphFLEx's structure beats all of the vanilla structures for every dataset. OOM and OOT denotes out-of-memory and out-of-time respectively.

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366

367

368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

Data Model ANN KNN log-model l2-model COVA large-model **Base Struct.**

Van. GFlex Van. GFlex Van. GFlex Van. GFlex Van. GFlex **Van. GFlex**

GAT 34*.23 67.*37 OOM 69.83 OOT 69.83 OOT 68.98 50*.48 68.*56 OOT 66.38 70.84

SAGE 34.23 69.58 OOM 70.28 OOT 70.28 OOT 70.68 51.47 70.51 OOT 69.32 72.57

DBLP GCN 34*.12 69.*41 OOM 73.39 OOT 73.39 OOT 73.05 51*.50 71.*75 OOT 68.55 74.43

GIN 34.01 69.69 OOM 68.19 OOT 68.19 OOT 73.08 52.77 72.03 OOT 71.18 73.92

GAT 12*.47 60.*89 OOM 61.09 OOT 60.95 18.64 61.06 58.96 88.06 OOT 86.22 60.75

SAGE 12*.70 78.*81 OOM 79.43 OOT 79.06 19.24 78.94 56.97 93.30 OOT 92.79 80.33

CS GCN 12.59 63.81 OOM 67.94 OOT 69.33 19.21 66.01 58.35 91.07 OOT 84.85 67.43

GIN 13.07 77.62 OOM 78.41 OOT 78.55 19.24 77.61 58.26 92.07 OOT 86.03 55.65

GAT 49*.49 83.*71 OOM 84.60 OOT 84.60 OOT 84.04 72*.63 83.*97 OOT 81.15 84.04

SAGE 50.43 87.27 OOM 87.34 OOT 87.34 OOT 87.42 73.57 86.68 OOT 87.34 88.88

Pub. GCN 50*.45 82.*06 OOM 83.56 OOT 83.56 OOT 83.74 73*.14 82.*39 OOT 78.03 85.54

GIN 51.82 83.13 OOM 84.31 OOT 84.07 OOT 82.93 73.15 83.51 OOT 82.85 86.50

GAT 29*.18 88.*06 OOM 88.47 OOT 88.47 OOT 88.68 58*.96 88.*06 OOT 86.22 88.58

SAGE 29.57 93.47 OOM 93.47 OOT 93.47 OOT 93.78 56.97 93.60 OOT 92.79 94.19

Phy. GCN 27.84 91.27 OOM 91.08 OOT 91.08 OOT 91.78 58*.35 91.*07 OOT 84.85 91.48

GIN 28.38 92.69 OOM 92.04 OOT 92.04 OOT 92.27 58.26 92.07 OOT 86.03 88.89

GNN Models. Graph neural networks (GNNs) such as GCN (Kipf & Welling, 2016), *GraphSage* (Hamilton et al., 2017), GIN (Xu et al., 2018), and GAT (Velickovic et al., 2017) rely on accurate message passing, dictated by the graph structure, for effective embedding. We use these models to evaluate the above-mentioned learned structures. Table 4 reports node classification performance across all methods. Notably, GraphFLEx outperforms vanilla structures by a significant margin across all datasets, achieving accuracies close to those obtained with the original structure. Figure 9 in Appendix F illustrates *GraphSage* classification results, highlighting GraphFLEx's superior performance. For the CS dataset, GraphFLEx (*large*-model) and GraphFLEx (*empCovar.*-model) even surpass the original structure, demonstrating its ability to preserve key structural properties while denoising edges, leading to improved accuracy.

## 4.2. Clustering Quality

We measure three metrics to evaluate the resulting clusters or community assignments: a) Normalized Mutual Information (NMI) (Tsitsulin et al., 2023) between the cluster assignments and original labels; b) Conductance (C) (Jerrum
& Sinclair, 1988) which measures the fraction of total edge volume that points outside the cluster; and c) Modularity (Q) (Newman, 2006) which measures the divergence between the intra-community edges and the expected one. Table 5 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 illustrates these metrics for single-cell RNA and MNIST dataset (where the whole structure is missing), and Figure 5 shows the PHATE (Moon et al., 2019) visualization of clusters learned using GraphFLEx's clustering module M*clust*.

We also train the aforementioned GNN models for the node classification task in order to illustrate the efficacy of the learned structures; the accuracies values presented in Table 5, clearly highlight the significance of the learned structures, as reflected by the high accuracy values.

Table 5: Clustering results and node classification accuracies. Left: Clustering metrics - NMI, graph conductance C, and Modularity Q. Right: Node classification accuracy for GCN, GraphSAGE, GIN, GAT.

## 4.3. Structure Visualization

We evaluate the structures generated by GraphFLEx through visualizations on four small datasets: (i) MNIST (LeCun et al., 2010), consisting of handwritten digit images, where Figure 6(a) shows that images of the same digit are mostly connected; (ii) Pre-trained GloVe embeddings (Pennington et al., 2014) of English words, with Figure 6(b) revealing that frequently used words are closely connected; (iii) A synthetic H.E dataset (see Appendix D), demonstrating Graph-
FLEx's ability to handle expanding networks without requir-

(a) 10 incoming nodes (b) 20 incoming nodes (c) 30 incoming nodes (d) ANN as Mgl (e) Emp. Covr. as Mgl (f) kNN as Mgl

| Data    | NMI ↑   | C ↓   | Q ↑   | GCN   | SAGE   | GIN   | GAT   |
|---------|---------|-------|-------|-------|--------|-------|-------|
| Bar. M. | 0.716   | 0.057 | 0.741 | 91.2  | 96.2   | 95.1  | 94.9  |
| Seger.  | 0.678   | 0.102 | 0.694 | 91.0  | 93.9   | 94.2  | 92.3  |
| Mura.   | 0.843   | 0.046 | 0.706 | 96.9  | 97.4   | 97.5  | 96.4  |
| Bar. H. | 0.674   | 0.078 | 0.749 | 95.3  | 96.4   | 97.2  | 95.8  |
| Xin     | 0.741   | 0.045 | 0.544 | 98.6  | 99.3   | 98.9  | 99.8  |
| MNIST   | 0.677   | 0.082 | 0.712 | 92.9  | 94.5   | 94.9  | 82.6  |

(a) Muraro (b) Baron Mouse (c) Segerstolpe
ing full relearning. Figure 4(a-c) shows the graph structure evolving as 30 new nodes are added over three timestamps; and (iv) Zachary's karate club network (Zachary, 1977),
which highlights GraphFLEx's multi-framework capability. Figure 4(d-f) shows three distinct graph structures after altering the learning module.

(a) MNIST (b) Glove

## 5. Conclusion

Large or expanding graphs challenge the best of graph learning approaches. GraphFLEx, introduced in this paper, seamlessly adds new nodes into an existing graph structure. It offers diverse methods for acquiring the graph's structure. GraphFLEx consists of three key modules: Clustering, Coarsening, and Learning which empowers Graph- FLEx to serves as a comprehensive framework applicable individually for clustering, coarsening, and learning tasks. GraphFLEx is typically 3X faster than other state of the art methods and scales well with large graphs. It achieves accuracies close to training on the original graph, in most instances. The performance across multiple real and synthetic datasets affirms the utility and efficacy of GraphFLEx for graph structure learning. Limitations and Future Work. GraphFLEx is designed assuming minimal inter-community connectivity, which aligns well with many real-world scenarios. However, its applicability to heterophilic graphs may require further adaptation. Future work will focus on extending the framework to supervised GSL methods and heterophilic graphs, broadening its scalability and versatility.

## Impact Statement References

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476

477

478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494

Defferrard, M., Bresson, X., and Vandergheynst, P. Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29, 2016. (Cited at p. 3.)
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. Allen, J. D., Xie, Y., Chen, M., Girard, L., and Xiao, G.

Comparing statistical methods for constructing large scale gene networks. *PloS one*, 7(1):e29348, 2012. **(Cited at** p. 1.)
Banerjee, O., El Ghaoui, L., and d'Aspremont, A. Model selection through sparse maximum likelihood estimation for multivariate gaussian or binary data. The Journal of Machine Learning Research, 9:485–516, 2008. **(Cited at** p. 3.)
Belkin, M., Niyogi, P., and Sindhwani, V. Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. Journal of machine learning research, 7(11), 2006. (Cited at p. 3.)
Bianchi, F. M., Grattarola, D., and Alippi, C. Spectral clustering with graph neural networks for graph pooling. In *International conference on machine learning*, pp. 874– 883. PMLR, 2020. (Cited at p. 4.)
Bruna, J., Zaremba, W., Szlam, A., and LeCun, Y. Spectral networks and deep locally connected networks on graphs. arxiv. *arXiv preprint arXiv:1312.6203*, 2014. **(Cited at** p. 3.)
Datar, M., Immorlica, N., Indyk, P., and Mirrokni, V. S.

Locality-sensitive hashing scheme based on p-stable distributions. In Proceedings of the twentieth annual symposium on Computational geometry, pp. 253–262, 2004. (Cited at p. **13.)**
Defferrard, M., Martin, L., Pena, R., and Perraudin, N.

Pygsp: Graph signal processing in python. URL https: //github.com/epfl-lts2/pygsp/. **(Cited at** p. **14.)**

| Bianchi, F. M. Simplifying clustering with graph neural networks. arXiv preprint arXiv:2207.08779, 2022. (Cited at p. 4.)   |
|-----------------------------------------------------------------------------------------------------------------------------|

Holme, P. and Saramaki, J. Temporal networks. ¨ Physics reports, 519(3):97–125, 2012. (Cited at p. 2.)
Hsieh, C.-J., Dhillon, I., Ravikumar, P., and Sustik, M. Sparse inverse covariance matrix estimation using quadratic approximation. Advances in neural information processing systems, 24, 2011. (Cited at p. 1.)
Dempster, A. P. Covariance selection. *Biometrics*, pp. 157–
175, 1972. (Cited at p. 3.)

| Chen, Y. and Wu, L. Graph neural networks: Graph structure learning. Graph Neural Networks: Foundations, Frontiers, and Applications, pp. 297–321, 2022. (Cited at p. 1.)   |
|---|

Dong, W., Moses, C., and Li, K. Efficient k-nearest neighbor graph construction for generic similarity measures.

In *Proceedings of the 20th international conference on* World wide web, pp. 577–586, 2011. (Cited at p. 1.)
Dong, X., Thanou, D., Frossard, P., and Vandergheynst, P. Learning laplacian matrix in smooth graph signal representations. *IEEE Transactions on Signal Processing*, 64(23):6160–6173, 2016. (Cited at pp. 1 and 3.)
Fatemi, B., El Asri, L., and Kazemi, S. M. Slaps: Selfsupervision improves structure learning for graph neural networks. Advances in Neural Information Processing Systems, 34:22667–22681, 2021. (Cited at p. 1.)
Fortunato, S. Community detection in graphs. Physics reports, 486(3-5):75–174, 2010. (Cited at p. 3.)
Fout, A., Byrd, J., Shariat, B., and Ben-Hur, A. Protein interface prediction using graph convolutional networks. Advances in neural information processing systems, 30, 2017. (Cited at p. 1.)
Friedman, J., Hastie, T., and Tibshirani, R. Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3):432–441, 2008. (Cited at pp. 1 and 3.)
Fu, X., Zhang, J., Meng, Z., and King, I. Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding. In *Proceedings of The Web Conference 2020*, pp. 2331–2341, 2020. (Cited at p. **14.)**
Gu, J., Zhao, H., Lin, Z., Li, S., Cai, J., and Ling, M. Scene graph generation with external knowledge and image reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.

1969–1978, 2019. (Cited at p. 1.)
Hamilton, W., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. *Advances in neural* information processing systems, 30, 2017. (Cited at p. 7.)
Hashemi, M., Gong, S., Ni, J., Fan, W., Prakash, B. A.,
and Jin, W. A comprehensive survey on graph reduction: Sparsification, coarsening, and condensation. arXiv preprint arXiv:2402.03358, 2024. (Cited at p. 3.)
Hu, C., Cheng, L., Sepulcre, J., El Fakhri, G., Lu, Y. M.,
and Li, Q. A graph theoretical regression model for brain connectivity learning of alzheimer's disease. In 2013 IEEE 10th International Symposium on Biomedical Imaging, pp. 616–619. IEEE, 2013. (Cited at p. 3.)
Kumar, M., Sharma, A., and Kumar, S. A unified framework for optimization-based graph coarsening. Journal of Machine Learning Research, 24(118):1– 50, 2023. URL http://jmlr.org/papers/v24/ 22-1085.html. (Cited at p. 3.)
495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Lao, D., Yang, X., Wu, Q., and Yan, J. Variational inference for training graph neural networks in low-data regime through joint structure-label estimation. In *Proceedings* of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pp. 824–834, 2022. **(Cited at** p. 1.)
Jerrum, M. and Sinclair, A. Conductance and the rapid mixing property for markov chains: the approximation of permanent resolved. In Proceedings of the twentieth annual ACM symposium on Theory of computing, pp. 235–244, 1988. (Cited at p. 7.)
Jin, W., Ma, Y., Liu, X., Tang, X., Wang, S., and Tang, J.

Graph structure learning for robust graph neural networks.

In *Proceedings of the 26th ACM SIGKDD international* conference on knowledge discovery & data mining, pp.

66–74, 2020. (Cited at p. 1.)
LeCun, Y., Cortes, C., and Burges, C. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010. **(Cited at** pp. 8 and **15.)**
Liu, Y., Zheng, Y., Zhang, D., Chen, H., Peng, H., and Pan, S. Towards unsupervised deep graph structure learning.

In *Proceedings of the ACM Web Conference 2022*, pp.

1392–1403, 2022. (Cited at p. 1.)
Kalofolias, V. How to learn a graph from smooth signals. In Artificial intelligence and statistics, pp. 920–929. PMLR, 2016. (Cited at pp. 1 and 3.)
Kamvar, S. D., Klein, D., and Manning, C. D. Spectral learning. In *IJCAI*, volume 3, pp. 561–566, 2003. **(Cited** at p. 3.)
Loukas, A. Graph reduction with spectral and cut guarantees.

J. Mach. Learn. Res., 20(116):1–42, 2019. (Cited at p. 3.)
Lu, L. and Zhou, T. Link prediction in complex networks: ¨
A survey. Physica A: statistical mechanics and its applications, 390(6):1150–1170, 2011. (Cited at p. **14.)**
Kataria, M., Khandelwal, A., Das, R., Kumar, S., and Jayadeva, J. Linear complexity framework for featureaware graph coarsening via hashing. In NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023. URL https://openreview.net/forum? id=HKdsrm5nCW. (Cited at p. 3.)
MacQueen, J. et al. Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability, volume 1, pp. 281–297. Oakland, CA, USA, 1967. (Cited at pp. 1 and 3.)
Kataria, M., Kumar, S., and Jayadeva, J. UGC: Universal graph coarsening. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=nN6NSd1Qds. (Cited at pp. 3, 5, and **13.)**
Moon, K. R., Van Dijk, D., Wang, Z., Gigante, S., Burkhardt, D. B., Chen, W. S., Yim, K., Elzen, A. v. d., Hirn, M. J., Coifman, R. R., et al. Visualizing structure and transitions in high-dimensional biological data. Nature biotechnology, 37(12):1482–1492, 2019. (Cited at pp. 8 and **19.)**
Khazane, A., Rider, J., Serpe, M., Gogoglou, A., Hines, K., Bruss, C. B., and Serpe, R. Deeptrax: Embedding graphs of financial transactions. In 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), pp. 126–133. IEEE, 2019. **(Cited at**
p. 2.)
Muja, M. and Lowe, D. G. Scalable nearest neighbor algorithms for high dimensional data. IEEE transactions on pattern analysis and machine intelligence, 36(11): 2227–2240, 2014. (Cited at p. 1.)
Kim, S., Yun, S., and Kang, J. Dygrain: An incremental learning framework for dynamic graphs. In *IJCAI*, pp. 3157–3163, 2022. (Cited at pp. 3 and **14.)**
Newman, M. E. Modularity and community structure in networks. *Proceedings of the national academy of sciences*, 103(23):8577–8582, 2006. (Cited at pp. 4 and 7.)
Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. (Cited at pp. 7 and **14.)**
Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., and Wermter, S. Continual lifelong learning with neural networks: A
review. *Neural networks*, 113:54–71, 2019. **(Cited at**
pp. 3 and **14.)**
Koller, D. and Friedman, N. *Probabilistic graphical models:*
principles and techniques. MIT press, 2009. **(Cited at**
p. 3.)
Pennington, J., Socher, R., and Manning, C. D. Glove:
Global vectors for word representation. In *Proceedings* of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532–1543, 2014. (Cited at pp. 8 and **15.)**
Shchur, O., Mumme, M., Bojchevski, A., and Gunnemann, ¨
S. Pitfalls of graph neural network evaluation. arXiv preprint arXiv:1811.05868, 2018. (Cited at p. **14.)**
Tsitsulin, A., Palowitch, J., Perozzi, B., and Muller, E. ¨
Graph clustering with graph neural networks. *Journal of* Machine Learning Research, 24(127):1–21, 2023. **(Cited** at pp. 3, 4, and 7.)
Van de Ven, G. M. and Tolias, A. S. Three scenarios for continual learning. *arXiv preprint arXiv:1904.07734*,
2019. (Cited at pp. 3 and **14.)**
Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y., et al. Graph attention networks. *stat*, 1050 (20):10–48550, 2017. (Cited at p. 7.)
Vogelstein, J. T., Roncal, W. G., Vogelstein, R. J., and Priebe, C. E. Graph classification using signal-subgraphs: Applications in statistical connectomics. *IEEE transactions on* pattern analysis and machine intelligence, 35(7):1539–
1551, 2012. (Cited at p. **14.)**
Wagstaff, K., Cardie, C., Rogers, S., Schrodl, S., et al. Con- ¨
strained k-means clustering with background knowledge. In *Icml*, volume 1, pp. 577–584, 2001. (Cited at p. 3.)
Wang, F. and Zhang, C. Label propagation through linear neighborhoods. In Proceedings of the 23rd international conference on Machine learning, pp. 985–992, 2006. (Cited at pp. 1 and 3.)
Wang, K., Shen, Z., Huang, C., Wu, C., Dong, Y., and Kanakia, A. Microsoft academic graph: When experts are not enough. quantitative science studies, 1 (1), 396– 413, 2020. (Cited at p. 4.)
Watts, D. J. and Strogatz, S. H. Collective dynamics of 'small-world'networks. *nature*, 393(6684):440–442, 1998. (Cited at p. **15.)**
Wu, Q., Zhao, W., Li, Z., Wipf, D. P., and Yan, J. Nodeformer: A scalable graph structure learning transformer for node classification. *Advances in Neural Information* Processing Systems, 35:27387–27401, 2022. **(Cited at**
p. 1.)
Wu, T., Liu, Q., Cao, Y., Huang, Y., Wu, X.-M., and Ding, J.

Continual graph convolutional network for text classification. In *Proceedings of the AAAI Conference on Artificial* Intelligence, volume 37, pp. 13754–13762, 2023. **(Cited**
at pp. 3 and **14.)**
550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Wu, Y., Lian, D., Xu, Y., Wu, L., and Chen, E. Graph convolutional networks with markov random field reasoning for social spammer detection. In *Proceedings of the* AAAI conference on artificial intelligence, volume 34, pp. 1054–1061, 2020. (Cited at p. 1.)
Xiang, L., Yuan, Q., Zhao, S., Chen, L., Zhang, X., Yang, Q., and Sun, J. Temporal recommendation on graphs via long-and short-term preference fusion. In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 723–732, 2010. (Cited at p. 4.)
Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018. (Cited at p. 7.)
Yang, F., Wang, W., Wang, F., Fang, Y., Tang, D., Huang, J.,
Lu, H., and Yao, J. scbert as a large-scale pretrained deep language model for cell type annotation of single-cell rnaseq data. *Nature Machine Intelligence*, 4(10):852–866, 2022. (Cited at p. **14.)**
Yang, Z., Cohen, W., and Salakhudinov, R. Revisiting semi-supervised learning with graph embeddings. In International conference on machine learning, pp. 40–48. PMLR, 2016. (Cited at p. **14.)**
You, J., Du, T., and Leskovec, J. Roland: graph learning framework for dynamic graphs. In *Proceedings of the* 28th ACM SIGKDD conference on knowledge discovery and data mining, pp. 2358–2366, 2022. **(Cited at pp.** 3 and **14.)**
Zachary, W. W. An information flow model for conflict and fission in small groups. *Journal of anthropological* research, 33(4):452–473, 1977. (Cited at pp. 8, 15, and **19.)**
Zhang, X., Song, D., and Tao, D. Cglb: Benchmark tasks for continual graph learning. Advances in Neural Information Processing Systems, 35:13006–13021, 2022. **(Cited at** pp. 3 and **14.)**
Zhao, Y., Levina, E., and Zhu, J. Consistency of community detection in networks under degree-corrected stochastic block models. 2012. (Cited at pp. 4 and **13.)**
Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., and Sun, M. Graph neural networks: A review of methods and applications. *AI open*, 1:57–81, 2020. (Cited at p. 1.)
Zhu, X., Ghahramani, Z., and Lafferty, J. D. Semisupervised learning using gaussian fields and harmonic functions. In Proceedings of the 20th International conference on Machine learning (ICML-03), pp. 912–919, 2003. (Cited at p. 3.)
Zhu, Y., Xu, W., Zhang, J., Du, Y., Zhang, J., Liu, Q.,
Yang, C., and Wu, S. A survey on graph structure learning: Progress and opportunities. arXiv preprint arXiv:2103.03036, 2021. (Cited at p. 1.)
605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

## Appendix A. Degree-Corrected Stochastic Block Model(Dc-Sbm)

The DC-SBM is one of the most commonly used models for networks with communities and postulates that, given node labels c = c1*, ...c*n, the edge variables A′ij s are generated via the formula

$$E[A_{i j}]=\theta_{i}\theta_{j}P_{c_{i}}P_{c_{j}}$$

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714
, where θiis a "degree parameter" associated with node i, reflecting its individual propernsity to form ties, and P is a K × K
symmetric matrix containing the between/withincommunity edge probabilities and PciPcjdenotes the edge probabilities between community ci and cj . For DC-SBM model (Zhao et al., 2012) assumed Pn on n nodes with k classes, each node viis given a label/degree pair(ci, θi), drawn from a discrete joint distribution ΠK×m which is fixed and does not depend on n. This implies that each θiis one of a fixed set of values 0 ≤ x1 ≤ *....* ≤ xm. To facilitate analysis of asymptotic graph sparsity, we parameterize the edge probability matrix P as Pn = ρnP where P is independent of n, and ρn = λn/n where λn is the average degree of the network.

## B. Neighbourhood Preservation

Theorem 2. **Neighborhood Preservation.** Let Nk(Ei) denote the neighborhood of incoming nodes Ei*for the* i th *community.*
With partition matrix Pi and Mgl(Si, Xiτ) = G
c τ(V
c τ, Acτ) we identify the supernodes connected to incoming nodes Ei and subsequently select nodes within those supernodes; this subset of nodes is denoted by ωV i τ
. Formally, Then, with probability Π{c∈ϕ}p(c)*, it holds that* Nk(Ei) ⊆ ωV i τ where

$$\omega_{V_{\tau}^{i}}=\bigcup_{v\in{\mathcal E}_{i}}\Big\{\bigcup_{s\in S_{i}}\{\pi^{-1}(s)|A_{\tau}^{c}(v,s)\neq0\}\Big\}$$
$$p(c)\leq1-{\frac{2}{\sqrt{2\pi}}}{\frac{c}{r}}\left[1-e^{-r^{2}/\left(2c^{2}\right)}\right],$$

and ϕ is a set containing all pairwise distance values (c = ∥v − u∥) between every node v ∈ Ei *and the nodes* u ∈ ωV i τ
. Here, π
−1(s)
denotes the set of nodes mapped to supernode s, r is the bin-width hyperparameter of M*coar*.

Proof: The probability that LSH random projection (Kataria et al., 2024; Datar et al., 2004) preserves the distance between two nodes v and u i.e., d(*u, v*) = c, is given by:

$$p(c)=\int_{0}^{r}{\frac{1}{c}}f_{2}\left({\frac{t}{c}}\right)\left(1-{\frac{t}{r}}\right)d t,$$

where f2(x) = √
2 2π e
−x 2/2represents the Gaussian kernel when the projection matrix is randomly sampled from pstable(p = 2) distribution (Datar et al., 2004). The probability p(c) can be decomposed into two terms:

$$p(c)=S_{1}(c)-S_{2}(c),$$

S1(c) and S2(c) are defined as follows:

$$S_{2}(c)=\frac{2}{\sqrt{2\pi}}\int_{0}^{r}e^{-(t/c)^{2}/2}\frac{t}{r}d t.$$
$$S_{2}(c)={\frac{2}{\sqrt{2\pi}}}\cdot{\frac{c}{r}}\int_{0}^{r}e^{-(t/c)^{2}/2}{\frac{t}{c^{2}}}d t$$

13

$$S_{1}(c)=\frac{2}{\sqrt{2\pi}}\int_{0}^{r}e^{-(t/c)^{2}/2}d t\leq1,$$

Expanding S2(c) :
Thus, the probability p(c) can be bounded as:

$$S_{2}(c)={\frac{2}{\sqrt{2\pi}}}$$
·
c
Z r
2/(2c
2)
$${\overline{{r}}}\int_{0}$$
$\tau^2$ (). 
e
−ydy
$$S_{2}(c)={\frac{2}{\sqrt{2\pi}}}\cdot{\frac{c}{r}}\left[1-e^{-r^{2}/(2c^{2})}\right]$$
$$p(c)\leq1-\frac{2}{\sqrt{2\pi}}\frac{c}{r}\left[1-e^{-r^{2}/(2c^{2})}\right].$$

Now, let ϕ be the set of all pairwise distances d(*u, v*), where v ∈ Ei and nodeωV i τ
. The probability that all nodes in Nk(Ei)
are preserved within ωV i τ
, requires that all distances c ∈ ϕ are also preserved. The probability is then given by:

$$\prod_{c\in\phi}p(c).$$

c∈ϕ
$$\prod_{c\in\phi}p(c)\leq\prod_{r\in\phi}\left(1-\frac{2}{\sqrt{2\pi}}\frac{c}{r}\left[1-e^{-r^{2}/(2c^{2})}\right]\right)$$

## C∈Φ C. Continual Learning And Dynamic Graph Learning

In this subsection, we highlight the key distinctions between Graph Structure Learning (GSL) and related fields to justify our specific selection of related works in Section 2.2. GSL is often confused with topics such as Continual Learning (CL) and Dynamic Graph Learning (DGL). CL (Van de Ven & Tolias, 2019; Zhang et al., 2022; Parisi et al., 2019) addresses the issue of catastrophic forgetting, where a model's performance on previously learned tasks degrades significantly after training on new tasks. In CL, the model has access only to the current task's data and cannot utilize data from prior tasks. Conversely, DGL (Kim et al., 2022; Wu et al.,
2023; You et al., 2022) focuses on capturing the evolving structure of graphs and maintaining updated graph representations, with access to all prior information. While both *CL and DGL* aim to *enhance model adaptability* to dynamic data, GSL is primarily concerned with generating high-quality graph structures that can be leveraged for downstream tasks such as node classification (Kipf & Welling, 2016), link prediction (Lu & Zhou ¨ , 2011), and graph classification (Vogelstein et al., 2012). Moreover, in CL and DGL, different tasks typically involve distinct data distributions, whereas GSL assumes a consistent data distribution throughout.

## D. Datasets

Datasets used in our experiments vary in size, with nodes ranging from 1k to 60k. Table 6 lists all the datasets we used in our work. We evaluate our proposed framework *GraphFlex* on real-world datasets *Cora ,Citeseer, Pubmed* (Yang et al., 2016), *CS, Physics* (Shchur et al., 2018), *DBLP* (Fu et al., 2020), all of which include graph structures. These datasets allow us to compare the learned structures with the originals. Additionally, we utilize single-cell RNA pancreas datasets (Yang et al., 2022), including Baron, Muraro, Segerstolpe, and Xin, where the graph structure is missing. The Baron dataset was downloaded from the Gene Expression Omnibus (GEO) (accession no. GSE84133). The Muraro dataset was downloaded from GEO (accession no. GSE85241). The Segerstolpe dataset was accessed from ArrayExpress (accession no. E-MTAB-5061). The Xin dataset was downloaded from GEO (accession no. GSE81608). We simulate the expanding graph scenario by splitting the original dataset across different T timestamps. We assumed 50% of the nodes were static, with the remaining nodes arriving as incoming nodes at different timestamps. Synthetic datasets: Different data generation techniques validate that our results are generalized to different settings. Please refer to Table 6 for more details about the number of nodes, edges, features, and classes, Syn denotes the type of synthetic datasets. Figure 7 shows graphs generated using different methods. We have employed three different ways to generate synthetic datasets which are mentioned below: - **PyGSP(PyGsp):** We used synthetic graphs created by PyGSP (Defferrard et al.) library. PyG-G and PyG-S denotes grid and sensor graphs from PyGSP.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769
- **Watts–Strogatz's small world(SW):** (Watts & Strogatz, 1998) proposed a generation model that produces graphs with small-world properties, including short average path lengths and high clustering.

- **Heterophily(HE):** We propose a method for creating synthetic datasets to explore graph behavior across a heterophily spectrum by manipulating heterophilic factor α, and classes. α is determined by dividing the number of edges connecting nodes from different classes by the total number of edges in the graph.

Visulization Datasets: To evaluate, the learned graph structure, we have also included three datasets: (i) MNIST (LeCun et al., 2010), consisting of handwritten digit images; (ii) Pre-trained GloVe embeddings (Pennington et al., 2014) of English words; and (iii) Zachary's karate club network (Zachary, 1977).

E. Algorithm Figure 7: This figure illustrates different types of synthetic graphs generated using i)PyGSP, ii) Watts–Strogatz's small world(SW), and iii) Heterophily(HE). N denotes the number of nodes, while α denotes the number of classes.

Table 6: Summary of the datasets.

| Cora             | 2,708   | 5,429   | 1,433   | 7   | Citation network      |
|------------------|---------|---------|---------|-----|-----------------------|
| Citeseer         | 3,327   | 9,104   | 3,703   | 6   | Citation network      |
| DBLP             | 17,716  | 52.8k   | 1,639   | 4   | Research paper        |
| CS               | 18,333  | 163.7k  | 6,805   | 15  | Co-authorship network |
| PubMed           | 19,717  | 44.3k   | 500     | 3   | Citation network      |
| Physics          | 34,493  | 247.9k  | 8,415   | 5   | Co-authorship network |
| Xin              | 1,449   | NA      | 33,889  | 4   | Human Pancreas        |
| Baron Mouse      | 1,886   | NA      | 14,861  | 13  | Mouse Pancreas        |
| Muraro           | 2,122   | NA      | 18,915  | 9   | Human Pancreas        |
| Segerstolpe      | 2,133   | NA      | 22,757  | 13  | Human Pancreas        |
| Baron Human      | 8,569   | NA      | 17,499  | 14  | Human Pancreas        |
| Syn 1            | 2,000   | 8,800   | 150     | 4   | SW                    |
| Syn 2            | 5,000   | 22k     | 150     | 4   | SW                    |
| Syn 3            | 10,000  | 44k     | 150     | 7   | SW                    |
| Syn 4            | 50,000  | 220k    | 150     | 7   | SW                    |
| Syn 5            | 400     | 1,520   | 100     | 4   | PyG-G                 |
| Syn 6            | 2,500   | 9,800   | 100     | 4   | PyG-S                 |
| Syn 7            | 1,000   | 9,990   | 150     | 4   | HE                    |
| Syn 8            | 2,000   | 40k     | 150     | 4   | HE                    |
| MNIST            | 60,000  | NA      | 784     | 10  | Images                |
| Zachary's karate | 34      | 156     | 34      | 4   | Karate club network   |
| Glove            | 2,000   | NA      | 50      | NA  | GloVe embeddings      |

(a) PyGSP-Sensor, N = 50, α=3 (b) PyGSP-Grid, N = 80, α=3 (c) SW, N = 50, α=3 (d) HE, N = 50, α=3

Category Data Nodes Edges Feat. Class Type

Visulization Datasets

MNIST 60,000 NA 784 10 Images

Zachary's karate 34 156 34 4 Karate club network

Glove 2,000 NA 50 NA GloVe embeddings

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

| Original Structure Known Original Structure Not Known Synthetic   |
|-------------------------------------------------------------------|
| Visulization Datasets                                             |

Algorithm 1 GraphFlex: A Unified Structure Learning framework for expanding and Large Scale Graphs 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879

## F. Other Gnn Models

16

17 Log model 24 23 GLasso 4 6 18 L2 model 16 11 Large model 5 21 20 19 KNN 
9 t 10 1 t6 t4 7 12 1 ANN

2 22 24 Empirical Covariance 8 22 13 19 3 14 15 15 16 1 Vanilla 2 3 4 5 Random Graph 17 7 8 6 14 Dynamic Nodes 9 18 10 t3 t2 11 21 12 13 20 t5 GNN Models 23 20 11 12 13 15 16 17 14 18 1 2 19 24 Log model L2 model KNN 
Empirical Covariance GLasso Large model ANN

GNN FACH

22 3 5 7 8 4 6 10 21 9 23 Compare Accuracy GraphFlex

Input: Graph G0(X0, A0), expanding nodes set E
T
1 = {Eτ (Vτ , Xτ )}
T τ=1 Parameter: GClust, GCoar, GL ← Clustering, Coarsening and Learning Module Output: Graph GT (XT , AT )
1: Train clustering module train(M*clust*, GClust, G0)
2: for each Et(Vt, Xt) in E
T
1 do 3: Ct = infer(Mclust, Xt), Ct ∈ R
Nt denotes the communities of Nt nodes at time t.

4: It = *unique*(Ct). 5: for each I
i tin It do 6: Git−1 = subgraph(Gt−1, I
i t)
7: {S
i t−1, Pi t−1} = M*coar*(Git−1), S
i t−1 ∈ R
k×dare features of k supernodes, P
i t−1 ∈ R
k×Nit is the partition matrix.

8: Gcit−1(S
i t−1, Ait−1) = Mgl(S
it−1, Xi t), Gcit−1is the learned graph on super-nodes S
i t−1and new node Xi t.

9: ω it ← []
10: for x ∈ Xi t do 11: ω i t*.append*(x)
12: np = {n | Ait−1
[n] > 0}
13: ω i t
.append(np)
14: **end for**
15: Gt−1 = update(Gt−1,Mgl(ω it
))
16: **end for**
17: Gt = Gt−1 18: **end for**
19: **return** GT (XT , AT )
We used four GNN models, namely GCN, GraphSage, GIN, and GAT. Table 7 contains parameter details we used to train GraphFlex. We have used these parameters across all methods.

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 We randomly split data in 60%, 20%, 20% for training-validation-test. The results for these models on synthetic datasets are presented in Table 8. Figure 8 illustrates the pipeline for training our GNN models. Graph structures were learned using both existing methods and GraphFlex, and GNN models were subsequently trained on both structures.

## G. Computational Efficiency

Table 9 illustrates the remaining computational time for learning graph structures using GraphFLEx with existing Vanilla methods on Synthetic datasets. While traditional methods may be efficient for small graphs, GraphFLEx scales significantly better, excelling on large datasets like *Pubmed* and *Syn 5*, where most methods fail.

| Model     | Hidden Layers   | L.R   | Decay   | Epoch   |
|-----------|-----------------|-------|---------|---------|
| GCN       | {64, 64}        | 0.003 | 0.0005  | 500     |
| GraphSage | {64, 64}        | 0.003 | 0.0005  | 500     |
| GIN       | {64, 64}        | 0.003 | 0.0005  | 500     |
| GAT       | {64, 64}        | 0.003 | 0.0005  | 500     |

Figure 9: GraphSage accuracies when structure is learned or given with 3 different scenarios(Vanilla, GraphFlex, Original) across different datasets, highlighting performance with 30% node growth over 25 timestamps. Figure 8 illustrates the pipeline for training our GNN models. Graph structures were learned using both existing methods and GraphFlex, and GNN models were subsequently trained on both structures. Results across all datasets are presented in Table 8 and Table 4.

Vanilla SLdgSL FullDataset 0 10 20 30 40 50 60 70 80 90 100 A
cc u ra c ie s Large Mod el Em p. Cova Larg e Mo del AN
N

KNN
LO
G M
odel L2 Mo del Em p. C
ova Lar ge M
odel AN
N

KNN
LOG Mo del L2 M
odel Em p. Cov a AN
N

KNN
LOG Mo del L2 M
ode l Larg e Mod el LOG Mo del L2 Mod el Em p. Co va Larg e M
ode l Large Mod el L2 M
odel Emp. 

Cov a L2 Mod el Emp. C
ova AN
N

KNN
LO
G M ode l AN
N

KNN
LOG Mod el AN
N

KNN
Cora Citeseer DBLP Pubmed Physics CS
Datasets
Table 8: Node classification accuracies on different GNN models using GraphFLEx (GFlex) with existing Vanilla (Van.) methods. The experimental setup involves treating 70% of the data as static, while the remaining 30% of nodes are treated as new nodes coming in 25 different timestamps. The best and the second-best accuracies in each row are highlighted by dark and lighter shades of Green, respectively. GraphFLEx's structure beats all of the vanilla structures for every dataset. OOM and OOT denotes out-of-memory and out-of-time respectively.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971

972

973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989

| Dataset    | Model       | ANN         | KNN         | log-model   | l2-model    | COVA        | large-model   | Base Struc.   |       |       |       |
|------------|-------------|-------------|-------------|-------------|-------------|-------------|---------------|---------------|-------|-------|-------|
| Van. GFlex | Van. GFlex  | Van. GFlex  | Van. GFlex  | Van. GFlex  | Van. GFlex  |             |               |               |       |       |       |
| GAT 18.73  | 73.84 20.96 | 73.65 16.14 | 72.36 18.74 | 73.10 49.72 | 77.55 14.28 | 76.43       | 79.77         |               |       |       |       |
| SAGE 17.25 | 77.37 18.00 | 76.99 19.48 | 77.40 19.85 | 75.51 49.35 | 76.99 14.28 | 77.55       | 82.37         |               |       |       |       |
| Cora       | GCN 17.99   | 78.11 17.81 | 77.92 18.55 | 77.74 20.41 | 79.22 47.31 | 80.52 14.28 | 79.03         | 84.60         |       |       |       |
| GIN 16.69  | 76.44 18.74 | 80.52 17.44 | 76.25 19.29 | 76.62 48.79 | 78.85 14.28 | 76.06       | 81.63         |               |       |       |       |
| GAT 16.51  | 61.82 25.00 | 62.27 19.24 | 64.70 18.18 | 63.48 20.91 | 62.73 16.67 | 62.27       | 66.42         |               |       |       |       |
| SAGE 16.66 | 68.48 16.67 | 68.64 22.12 | 69.39 22.42 | 69.85 22.88 | 71.52 16.67 | 69.39       | 72.57         |               |       |       |       |
| Citeseer   | GCN 28.18   | 60.00 16.67 | 61.97 20.45 | 65.45 19.70 | 64.24 21.06 | 64.70 16.67 | 63.18         | 68.03         |       |       |       |
| GIN 16.66  | 64.39 16.67 | 63.94 20.15 | 59.85 18.64 | 63.64 22.12 | 60.30 16.67 | 61.81       | 67.38         |               |       |       |       |
| GAT 29.55  | 92.07 OOM   | 90.86       | OOT         | 91.64       | OOT         | 91.64 35.79 | 92.52         | OOT           | 93.74 | 89.49 |       |
| SAGE 26.75 | 87.89 OOM   | 91.05       | OOT         | 86.64       | OOT         | 86.64 32.92 | 90.44         | OOT           | 86.01 | 90.03 |       |
| Syn 4      | GCN 28.85   | 51.97 OOM   | 19.58       | OOT         | 18.29       | OOT         | 18.92 33.80   | 26.60         | OOT   | 36.85 | 21.43 |
| GIN 28.50  | 65.61 OOM   | 31.06       | OOT         | 26.51       | OOT         | 26.56 34.03 | 46.40         | OOT           | 47.10 | 29.35 |       |
| GAT 44.00  | 86.80 43.60 | 86.60 30.00 | 78.75 55.40 | 92.80 36.20 | 93.60 31.80 | 92.80       | 97.20         |               |       |       |       |
| SAGE 41.00 | 93.80 41.40 | 93.60 33.75 | 88.75 57.60 | 94.00 35.20 | 94.80 28.20 | 95.60       | 97.40         |               |       |       |       |
| Syn 6      | GCN 43.60   | 88.80 42.20 | 87.40 26.25 | 81.25 55.60 | 92.40 31.40 | 94.40 25.20 | 94.00         | 99.40         |       |       |       |
| GIN 39.60  | 89.00 40.40 | 86.60 21.25 | 82.50 55.20 | 91.80 30.00 | 94.60 30.40 | 92.00       | 98.80         |               |       |       |       |
| GAT 29.55  | 99.75 33.75 | 88.75 88.25 | 99.25 88.25 | 99.25 26.00 | 85.50 94.00 | 96.00       | 98.50         |               |       |       |       |
| SAGE 26.75 | 100.0 32.50 | 100.0 88.75 | 99.50 88.75 | 99.50 26.75 | 100.0 92.50 | 100.0       | 100.0         |               |       |       |       |
| Syn 8      | GCN 28.85   | 98.75 31.75 | 99.75 88.75 | 99.00 88.75 | 99.00 28.50 | 99.25 95.00 | 100.0         | 100.0         |       |       |       |
| GIN 28.50  | 50.00 30.50 | 91.00 82.25 | 91.50 82.25 | 91.50 27.25 | 81.75 91.75 | 92.25       | 78.25         |               |       |       |       |

Table 9: Computational time for learning graph structures using GraphFLEx (GFlex) with existing methods (Vanilla referred to as Van.). The experimental setup involves treating 50% of the data as static, while the remaining 50% of nodes are treated as incoming nodes arriving in 25 different timestamps. The best times are highlighted by color Green. OOM and OOT denote out-of-memory and out-of-time, respectively.

Data ANN KNN log-model l2-model COVA large-model

Van. GFlex Van. GFlex Van. GFlex Van. GFlex Van. GFlex Van. GFlex

Syn 1 19.4 9.8 2.5 10.5 2418 56.4 37.2 8.8 3.5 8.3 205 9.4 Syn 2 47.3 16.9 6.6 18.3 14000 144 214 22.6 20.3 18.6 1259 16.4 Syn 5 5.1 11.5 0.8 7.3 57.4 28 1.1 5.8 0.2 4.8 3.2 5.3 Syn 6 16.6 9.9 2.8 11.4 1766 96.3 193 101 5.3 8.9 324 9.6 Syn 7 10.6 7.4 1.4 8.9 704 85.2 10.3 7.9 0.9 6.4 36.5 8.2 Syn 8 19.6 11.2 2.5 11.7 2416 457 37.2 17.0 3.4 10.9 204 11.7

## H. Visualization Of Growing Graphs

This section helps us visualize the phases of our growing graphs. We have generated a synthetic graph of 60 nodes using PyGSP-Sensor and HE methods mentioned in Appendix D. We then added 40 new nodes denoted using black color in these existing graphs at four different timestamps. Figure 10 and Figure 11 shows the learned graph structure after each timestamp for two different Synthetic graphs.

Figure 10: This figure illustrates the growing structure learned using GraphFlex for dynamic nodes. New nodes are denoted using black color, and α denotes number of new nodes. *PyGsp* denotes type synthetic graph.

HE
(a) Initial graph G0 (b) α = 10, G1 (c) α = 20, G2 (d) α= 30, G3 (e) α = 40, G4 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Figure 11: This figure illustrates the growing structure learned using GraphFlex for dynamic nodes. New nodes are denoted using black color, and α denotes the number of new nodes. HE denotes the type of synthetic graph.

Vanilla
(a) ANN (b) Emp. Cov. (c) KNN (d) L2 model (e) Log model GraphFlex
(f) ANN (g) Emp Cov. (h) KNN (i) L2 model (j) Log model

## I. Structure Comparison On Karate Dataset

This section involves a comparison of the graph structure learned from GraphFlex with existing methods. Six nodes were randomly selected and considered as new nodes. Figure 12 visually depicts the structures learned using GraphFlex compared to other methods. It is evident from the figure that the structure known with GraphFlex closely resembles the original graph structure. Figure 13 shows the original structure of Zachary's karate club network (Zachary, 1977). We assumed six random nodes to be dynamic nodes, and the structure learned using GraphFlex compared to existing methods is shown in Figure 12.

## J. Clustering Quality

Figure 14 shows the PHATE (Moon et al., 2019) visualization of clusters learned using GraphFLEx's clustering module M*clust* for Xin, *MNIST*, and Baron − *Human* datasets.

PyGsp
(a) Initial graph G0 (b) α= 10, G1 (c) α= 20, G2 (d) α = 30, G3 (e) α = 40, G4 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099
(a) Xin (b) MNIST (c) Baron Human