**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# TOWARD HUMAN-INTERPRETABLE EXPLANATIONS IN A UNIFIED FRAMEWORK FOR GNNS

Anonymous authors Paper under double-blind review

### ABSTRACT

As Graph Neural Networks (GNNs) are increasingly applied across various domains, explainability has become a critical factor for real-world applications. Existing post-hoc explainability methods primarily focus on estimating the importance of edges, nodes, or subgraphs in the input graph to identify substructures crucial for predictions. However, these methods often lack human interpretability and do not provide a unified framework that incorporates both model-level and instance-level explanations. In this context, we propose leveraging a set of graphlets—small, connected, non-isomorphic induced subgraphs widely used in various scientific fields—and their associated orbits as human-interpretable units to decompose GNN predictions. Domain experts can select the most relevant graphlets as interpretable units and request unified explanations based on these units. To address this problem, we introduce UO-Explainer, the Unified and Orbit-based Explainer for GNNs, which utilizes predefined orbits that are generalizable and universal across graph domains as interpretable units. Our model decomposes GNN weights into orbit units to extract class-specific graph patterns (model-level) and to identify important subgraphs within individual data instances for prediction (instance-level). Extensive experimental results demonstrate that UO-Explainer outperforms existing baselines in providing meaningful and interpretable explanations across both synthetic and real-world datasets. Our code and datasets are available at <https://anonymous.4open.science/r/uoexplainer-F12C>.

## 1 INTRODUCTION

Graph Neural Networks (GNNs) have achieved state-of-the-art performance in various domains including real-world graph-structured data, such as social networks [\(Fan et al., 2019\)](#page-10-0), molecules [\(Duve](#page-10-1)[naud et al., 2015\)](#page-10-1), and knowledge graphs [\(Hogan et al., 2021\)](#page-10-2). Despite the remarkable advancements in GNN architectures [\(Hamilton et al., 2017;](#page-10-3) [Kipf & Welling, 2017;](#page-11-0) [Xu et al., 2019;](#page-12-0) [Velickovi](#page-12-1) ˇ c´ [et al., 2018\)](#page-12-1), they are still perceived as black box models due to their lack of explainability. This deficiency limits trust in GNN predictions, hindering their application in areas such as drug development [\(Gaudelet et al., 2021\)](#page-10-4) and education [\(Nakagawa et al., 2019\)](#page-11-1). Therefore, interpreting the prediction of GNNs has become crucial and has led to the emergence of various explanatory approaches.

Explainability methods in graph domains deliver explanations through subgraphs that play a significant role in predictions regarding the input graph. Two primary issues arise regarding explainability: i) a human-interpretability and ii) a unified framework encompassing both model and instance levels. Many existing post-hoc explanability is grounded on perturbation-based methods [\(Ying et al., 2019;](#page-12-2) [Luo et al., 2020;](#page-11-2) [Xie et al., 2022;](#page-12-3) [Zhang et al., 2023;](#page-12-4) [Schlichtkrull et al., 2022\)](#page-11-3) and gradient-based methods [\(Baldassarre & Azizpour, 2019;](#page-10-5) [Pope et al., 2019\)](#page-11-4) to approximate the importance of edges or nodes within subgraphs. This stochastic optimization of importance is computationally effective in applying any graph-structured data, these methods have the potential risk that the output subgraph does not align with prior human assumption or knowledge. Specifically, consider a scientist studying gene networks who is interested in understanding whether the presence of certain structures, such as triangles or rectangles, plays a crucial role in predicting specific properties. This scientist would want to audit the model's predictions by utilizing an explanation method that can highlight the importance of these structures. With existing methods, it is challenging to obtain explicit insights into whether a triangular or rectangular structure in the network is more important; instead, users often have to rely

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106** on conjecture based on the generated explanations. Additionally, these explanations do not always present results in a human-intuitive form, as they may include isolated nodes or disconnected edges, further hindering their interpretability. Therefore, we emphasize the need for explainability that centers on human interpretability, a perspective that has been largely underexplored.

Last but not least, human-interpretable explanations should be delivered in a unified framework, incorporating both global and local levels in terms of the scope of explanation [\(Yuan et al., 2022;](#page-12-5) [Prado-Romero et al., 2024\)](#page-11-5). Most of the existing methods primarily specialize one one-level explanation either model or instance-level explanations. Model-level methods reveal patterns that GNNs deem significant for specific classes, eventually offering a broad understanding of the GNNs' general behavior. On the other hand, instance-level methods focus on individual predictions, identifying the subgraphs most relevant to the target node or graph. As each type of explanation complements the other from different perspectives, understanding at both levels enhances the explainability necessary for grasping the decision-making process of GNNs. However, State-of-the-art [\(Azzolin et al., 2022;](#page-10-6) [Chen et al., 2023\)](#page-10-7) studies also have rarely explored a unified framework that simultaneously provides both model-level and instance-level explanations in the user-centric perspective of interest.

In this paper, we prioritize the two crucial perspectives that explanation should be human-interpretable in the unified framework incorporating both model and instance levels. Our proposed model, UO-Explainer, the Unified and Orbit-based Explainer for GNNs, allows users to harness their prior knowledge by giving room to select the user-defined explanation units in unified views. Considering the uniqueness of the graph domain to define explanation unit, we exploit orbits within graphlets that have been studied as a generalizable and universal pattern in many scientific fields such as protein interaction networks [Przulj et al.](#page-11-6) [\(2004\)](#page-11-6); Pr ˇ [zulj](#page-11-7) [\(2007\)](#page-11-7), social networks [Chen & Lui](#page-10-8) [\(2018\)](#page-10-8); [Ahmed](#page-10-9) ˇ [et al.](#page-10-9) [\(2015\)](#page-10-9), and molecular structure networks [Kondor et al.](#page-11-8) [\(2009\)](#page-11-8), while users can also adopt a unique prior perspective to define their unit instead of orbit. To provide unified explanations, we decompose the weights into orbit representation vectors to understand the contribution of each orbit for a specific class or prediction. When we break down the weight into orbits, we acknowledge the contribution of each orbit for model decision-making, which is supposed to be further discussed in the method section in detail. Through extensive experiments, we demonstrate the superior performance of UO-Explainer on eight well-known datasets. Consequently, UO-Explainer shows the concrete performance in a unified framework leveraging the human prior knowledge as orbit generalizable unit in graph-structured datasets.

In summary, the contributions of our research are as follows:

- We propose UO-Explainer, a unified framework to provide both model-level and instancelevel explanations for node classification based on human-defined explanation units.
- We demonstrate the human interpretable explanation based on orbits generalizable and effective on graph-structured datasets.
- We perform rigorous and extensive experiments on 8 datasets to evaluate the quality of our explanations in both model and instance-level explanations.

## 2 PRELIMINARY

## 2.1 GRAPH NEURAL NETWORKS (GNNS)

We represents a graph as G = (V, E; <sup>A</sup>, <sup>X</sup>) where E denotes a edge set and V = {<sup>v</sup>1, v2,⋯, v∣V∣} denotes a node set. <sup>A</sup> ∈ <sup>R</sup> ∣V∣×∣V∣ denotes the adjacent matrix and <sup>X</sup> ∈ <sup>R</sup> ∣V∣×din denotes the node feature matrix. In this study, we focus on GNNs for node classification tasks as presented in [\(Kipf](#page-11-0) [& Welling, 2017;](#page-11-0) [Xu et al., 2019\)](#page-12-0). A GNN model <sup>f</sup>(⋅) maps input graph into prediction matrix <sup>f</sup>(<sup>A</sup>, <sup>X</sup>) = <sup>Z</sup> ∈ <sup>R</sup> ∣V∣×∣C∣ , C = {<sup>c</sup>1,⋯, c∣C∣} in where C as the set of classes. GNNs can be expressed as a composite function <sup>f</sup> = <sup>f</sup><sup>D</sup> ○ <sup>f</sup><sup>E</sup> of an embedding-model <sup>f</sup>E(⋅) and a downstream-model <sup>f</sup>D(⋅). The embedding model embeds an adjacent matrix and node feature matrix into a node representation matrix <sup>H</sup>, i.e., <sup>f</sup>E(<sup>A</sup>, <sup>X</sup>) = <sup>H</sup> ∈ <sup>R</sup> ∣V∣×d . The representation vector of each node v<sup>n</sup> is denoted by the h<sup>v</sup><sup>n</sup> , n-th row vector of matrix H. The downstream-model maps the node representation matrix into the prediction matrix to classify nodes into each class, i.g., <sup>f</sup>D(H) = HW+<sup>b</sup> = <sup>Z</sup> where <sup>W</sup> ∈ <sup>R</sup> d×∣C∣ denotes weight matrix and <sup>b</sup> ∈ <sup>R</sup> ∣C∣ denotes bias vectors. In this operation, only the m-th column

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

![](_page_2_Diagram_1.jpeg)

Figure 2: Overview of UO-Explainer: The propagation process of the GNN is depicted in the blue box, while the orange box represents the pipeline of UO-Explainer. The provided explanations are illustrated in the green box. The dark green node of the input graph denotes the target node. The colors (red, orange, blue, and yellow) correspond to the orbits within the graphlets.

vector w<sup>c</sup><sup>m</sup> of weight matrix W and the m-th element b<sup>c</sup><sup>m</sup> of bias vectors b are involved in the computation to predict the class <sup>c</sup>m, i.g., <sup>z</sup><sup>c</sup><sup>m</sup> = Hw<sup>c</sup><sup>m</sup> + <sup>b</sup><sup>c</sup><sup>m</sup> where <sup>z</sup><sup>c</sup><sup>m</sup> denotes the <sup>m</sup>-th row vector of matrix Z. Weight regards to specific class as w<sup>c</sup><sup>m</sup> affects only the prediction of the class cm, so we call this weight vector a class weight. Furthermore, the prediction value z<sup>v</sup>n,c<sup>m</sup> for the class of each node (the <sup>n</sup>-th row and <sup>m</sup>-th column element of <sup>Z</sup>) can be expressed as <sup>z</sup><sup>v</sup>n,c<sup>m</sup> = <sup>h</sup><sup>v</sup><sup>n</sup> ⋅ <sup>w</sup><sup>c</sup><sup>m</sup> + <sup>b</sup><sup>c</sup>m, computed by the operation between each node representation vector and class weight.

#### 2.2 ORBITS WITHIN GRAPHLETS

![](_page_2_Diagram_5.jpeg)

![Figure 3: A graph diagram showing the connection of the two graphs in Figure 2. The graph in Figure 2 is a complex graph with nodes 0 through 14 and edges g1 through g8. The graph in Figure 3 is a similar graph with nodes 0 through 14 and edges g1 through g12. The graph in Figure 3 is a similar graph with nodes 0 through 14 and edges g1 through g12. The graph in Figure 3 is a similar graph with nodes 0 through 14 and edges g1 through g12.]()

Figure 1: Graphlets having 2-5 nodes and 0-72 orbits. The same color nodes within each graphlet belong to the same orbit.

Graphlets as predominantly observed patterns are pre-defined subgraphs with a small number of nodes [\(Przulj et al., 2004\)](#page-11-6). Figure [1](#page-2-0) shows ˇ some examples of 2-5 node graphlets [\(Przulj](#page-11-6) ˇ [et al., 2004;](#page-11-6) [Przulj, 2007\)](#page-11-7), where ˇ g<sup>l</sup> denotes the l-th graphlet. The full set of 2-5 node graphlets used in our study is presented in Appendix [A.](#page-13-0) All graphlets are non-isomorphic to each other, indicating that each graphlet has a unique structure. Each graphlet contains nodes with identical or distinguishable topological positions known as orbits, e.g., the central node of g<sup>1</sup> belongs to o<sup>2</sup> due to its distinguishable position, whereas

the remaining nodes belong to <sup>o</sup>1. The set of orbits is represented as O = {<sup>o</sup>0,⋯, ok,⋯, o∣O∣} where o<sup>k</sup> denotes the k-th orbit.

Previous studies have highlighted the usefulness of graphlet-based analysis in various graph data domains, including protein interaction networks [\(Przulj et al., 2004;](#page-11-6) Pr ˇ [zulj, 2007\)](#page-11-7), social networks [\(Chen](#page-10-8) ˇ [& Lui, 2018;](#page-10-8) [Ahmed et al., 2015\)](#page-10-9), and molecular structure networks [\(Kondor et al., 2009\)](#page-11-8). For example, [\(Przulj, 2007\)](#page-11-7) defined the Graphlet Degree Distribution (GDD) using 2-5 node graphlets and ˇ orbits as units to analyze agreement among biological and chemical networks. [\(Shervashidze et al.,](#page-11-9) [2009;](#page-11-9) [Espejo et al., 2020\)](#page-10-10) further compared the empirical similarity of various chemical compound networks using a 2-5 node graphlet kernel. Since these analyses indicate that 2- to 5-node graphlets can serve as simple yet effective units for interpreting graph data, we primarily employ them to provide orbit-based explanations, unless otherwise specified by the user.

## 3 UO-EXPLAINER

UO-Explainer serves as a unified explainer capable of delivering both model-level and instance-level explanations for node classification tasks. To deliver explanations in a human-interpretable way, we exploit a pre-defined set of 0-72 orbits as the explanatory unit, which is recognized as essential and human-interpretable units within graph domains, while users can also apply other meaning units regarding their perspective instead of orbits. Upon the interpretable unit, we decompose the weights

**166 167**

**169**

**171**

**204**

**206**

that directly influence classification concerning two components such as an embedding model and a downstream task model. An overview of UO-Explainer is presented in Figure [2.](#page-2-1)

#### 3.1 ORBIT BASIS LEARNING

To decompose class weights into orbit units requires orbit bases, which are the representation vectors of each orbit. Orbit bases must necessarily reflect the following two aspects: (1) The distribution of each orbit within the input graph, and (2) The message passing and aggregation behavior of the embedding model. To meet the first requirement, we pre-process the orbit-existences on each node in the input graph. Orbit existence is denoted as y<sup>v</sup>n,o<sup>k</sup> and determines whether each node v<sup>n</sup> belongs to the orbit ok.

$$y_{v_n, o_k} = \begin{cases} 0 & \text{if } v_n \text{ doesn't belong to } o_k \\ 1 & \text{if } v_n \text{ belongs to } o_k. \end{cases} \quad (1)$$

We present the toy example of this pre-processing through Figure [3.](#page-3-0) (a) portrays the input graph. (b) represents the graphlets and orbits employed in the pre-processing. For simplicity, let us assume that only the graphlets g2, g3, g<sup>6</sup> are utilized and the orbits used for pre-processing are o3, o4, o5, o11. Different color nodes inside each graphlet refer to nodes belonging to different orbits. (c) is a substantial pre-processing process, which checks whether each node can belong to each orbit and assigns 1 or 0 to the orbit's existence. The dark green node represents the pre-processing node. You can see that node 3 belong to o3, o5, o11. Therefore, y<sup>v</sup>3,o<sup>3</sup> , y<sup>v</sup>3,o<sup>4</sup> , y<sup>v</sup>3,o<sup>5</sup> , y<sup>v</sup>3,o<sup>11</sup> are assigned values of 1, 1, 0, 1, respectively. This pre-processing is performed for all orbits in 2-5 node graphlets at all nodes within the input graph.

![](_page_3_Diagram_7.jpeg)

Figure 3: Detailed example of the pre-processing step. The dark green node represents the target node for explanation. The colors of nodes within the graphlets represent the orbits respectively.

Next, we train a logistic binary classifier to predict the existence of each orbit, initializing each orbit with pˆ<sup>o</sup><sup>k</sup> vector, and taking node representation as input, described by the following equation:

$$\hat{y}_{v_n, o_k} = \text{sigmoid}(\hat{\mathbf{p}}_{o_k} \cdot h_{v_n}). \quad (2)$$

To satisfy the second aspect, we learn the orbit basis by incorporating the node representations from the node embedding model when training. Then, we apply normalization in Equation as <sup>p</sup><sup>o</sup><sup>k</sup> = pˆok ∣∣pˆok ∣∣ to ensure that the size of the orbit basis remains constant. Orbit-basis learning is conducted for all orbits, and details of orbit basis learning can be found in Algorithm [1.](#page-4-0)

#### 3.2 MODEL-LEVEL EXPLANATIONS

Model-level explanations are provided by decomposing class weights into a linear combination of orbit bases, as the following equation:

$$\begin{aligned} \mathbf{w}_{cm} &\approx s_{cm,0_0} \mathbf{p}_{00} + \dots + s_{cm,o_k} \mathbf{p}_{o_k} + \dots + s_{cm,o_K} \mathbf{p}_{o_K} \\ &\approx \sum_{o_k \in \mathcal{O}} s_{cm,o_k} \mathbf{p}_{o_k}. \end{aligned} \quad (3)$$

Generally, when a vector is expressed as a linear combination of bases, the coefficients of each basis indicate to what extent they contribute to forming the vector. Accordingly, the coefficients of orbit bases are regarded as contributions to the class weights. Furthermore, the bases are learned by considering each orbit distribution, thereby treating the contribution of orbit basis as the contribution of each orbit. We define the contribution of orbit o<sup>k</sup> to the class c<sup>m</sup> classification as a class-orbit score s<sup>c</sup>m,o<sup>k</sup> .

The class-orbit scores are trained by the following objective function derived from Equation [3:](#page-3-1)

$$\min_{s_{c_m, o_k} > 0} \left\| \mathbf{w}_{c_m} - \sum_{o_k \in \mathcal{O}} s_{c_m, o_k} \mathbf{p}_{o_k} \right\|. \quad (4)$$

**224**

**236 237**

**254**

**256**

**259**

**269**

Algorithm 1 Orbit Basis Learning

Input: A set of node representation vectors {hv<sup>1</sup> , <sup>⋯</sup>, <sup>h</sup>vn , <sup>⋯</sup>, <sup>h</sup>v∣V∣ }, a set of existences of each orbits for each nodes {yv1,o<sup>0</sup> , yv2,o<sup>0</sup> , <sup>⋯</sup>, yv1,o<sup>23</sup> , <sup>⋯</sup>, yv∣V∣ ,o<sup>72</sup> }. Output: A set of orbit bases P <sup>=</sup> {po<sup>0</sup> , <sup>⋯</sup>, pok , <sup>⋯</sup>, po<sup>72</sup> } Initialize P as an empty set 1: for k = 0 to 72 (the number of orbits used) do 2: Initialize <sup>p</sup>ˆok as a random vector. 3: for n=1 to ∣V∣ (the number of nodes) do 4: <sup>y</sup>ˆvn,ok <sup>=</sup> sigmoid(pˆok <sup>⋅</sup> hvn ) 5: <sup>L</sup> <sup>=</sup> BCE(yvn,ok , yˆvn,ok ) 6: Update <sup>p</sup>ˆok via <sup>∇</sup>pˆok L 7: end for 8: <sup>p</sup>ok = <sup>p</sup>ˆok ∣∣pˆok 9: P.add(pok ) 10: end for 11: Return P

To consider only the positive impact of the contributing orbits, we limit the class-orbit score to positive. However, directly optimizing the objective function for all orbit bases involves a significant amount of randomness. For example, in the worst case, if all orbit bases are orthogonal, and the dimension of the class weight is <sup>d</sup> < ∣O∣, an infinite number of combinations of <sup>s</sup><sup>c</sup>m,o<sup>k</sup> can be found to optimize the Equation [4.](#page-3-2) This randomness hinders the learning of the correct contribution of orbits. Therefore, we modify the objective function using a greedy approach by selecting the orbit that minimizes the difference between the class weight and the linear combination of selected orbits in each iteration, as shown in the following equation:

$$\arg \min_{o_k \in \mathcal{O}} \min_{\mathbf{S}_{c_m} > 0} \|\mathbf{w}_{c_m} - [\mathbf{P}_{c_m} | \mathbf{p}_{o_k}] \mathbf{S}_{c_m}\|. \quad (5)$$

P<sup>c</sup><sup>m</sup> is a matrix consisting of the selected p<sup>o</sup><sup>k</sup> as columns, and S<sup>c</sup><sup>m</sup> represents a column vector consisting s<sup>c</sup>m,o<sup>k</sup> of the selected orbits. [<sup>P</sup><sup>c</sup>m∣<sup>p</sup><sup>o</sup><sup>k</sup> ] denotes concatenation of the <sup>p</sup><sup>o</sup><sup>k</sup> to P<sup>c</sup><sup>m</sup> as a column, e.g., if orbits 1, 3, and 5 are selected, then <sup>P</sup><sup>c</sup><sup>m</sup> = [<sup>p</sup><sup>o</sup><sup>1</sup> ∣<sup>p</sup><sup>o</sup><sup>3</sup> ∣<sup>p</sup><sup>o</sup><sup>5</sup> ] and <sup>S</sup><sup>c</sup><sup>m</sup> is a vector composed of s<sup>c</sup>m,o<sup>1</sup> , s<sup>c</sup>m,o<sup>3</sup> and s<sup>c</sup>m,o<sup>5</sup> . By stopping the selection when the difference between the class weight and the linear combination does not decrease, we reduce the randomness and prevent too many orbits from being included in the explanation for each class. The detailed procedure can be found in Algorithm [2.](#page-5-0)

UO-Explainer uses the orbit o ∗ <sup>c</sup><sup>m</sup> with the highest class-orbit score from Equation [6](#page-4-1) as the model-level explanation.

$$o_{c_m}^* = \arg \max_{o_k \in \mathcal{O}} \{s_{c_m, o_0}, \dots, s_{c_m, o_k}, \dots, s_{c_m, o_{72}}\}. \quad (6)$$

The orbit with the highest class-orbit score is always accompanied by its corresponding graphlet. Therefore, the model-level explanation is provided in the form of graph patterns, with the given orbit as the target node and its corresponding graphlet.

### 3.3 INSTANCE-LEVEL EXPLANATIONS

Instance-level explanations are provided by decomposing the prediction value of the target node into orbit units. The class weight decomposition of Equation [3](#page-3-1) extends to the decomposition of prediction values as follows:

$$\begin{aligned} & z_{v_n, c_m} \approx \mathbf{h}_{v_n} \cdot \mathbf{w}_{c_m} + \mathbf{b}_{c_m} \\ & \approx \underbrace{s_{c_m, o_0} \mathbf{h}_{v_n} \cdot \mathbf{p}_{o_0}}_{s_{v_n, c_m, o_0}} + \dots + \underbrace{s_{c_m, o_k} \mathbf{h}_{v_n} \cdot \mathbf{p}_{o_k}}_{s_{v_n, c_m, o_k}} + \dots + \underbrace{s_{c_m, o_K} \mathbf{h}_{v_n} \cdot \mathbf{p}_{o_K}}_{s_{v_n, c_m, o_K}} + \mathbf{b}_{c_m}. \end{aligned} \quad (7)$$

The equation above directly decomposes the prediction value of the target node into orbit units. Each term represents the magnitude of the decomposed prediction value, similarly indicating the contribution of orbits, akin to class weight decomposition. Therefore, we define the contribution of orbit o<sup>k</sup> to the class c<sup>m</sup> prediction of target node v<sup>n</sup> as node-class-orbit score, s<sup>v</sup>n,cm,o<sup>k</sup> . To provide

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

Algorithm 2 Class-Orbit Score Learning

Input: A set of orbit bases {po<sup>0</sup> , <sup>⋯</sup>, pok , <sup>⋯</sup>, po<sup>72</sup> }, a set of class weight vectors {wc<sup>1</sup> , <sup>⋯</sup>, <sup>w</sup>cm, <sup>⋯</sup>, <sup>w</sup>c∣C∣ }. Output: A set S of selected orbit's class-orbit scores scm,ok Variables: A vector composed of an element as a selected orbit's class-orbit score Scm. Initialize S as an empty set 1: for m=1 to ∣C∣ (the number of classes) do 2: Initialize lmin <sup>=</sup> <sup>∞</sup> 3: Initialize <sup>P</sup>cm as an empty matrix 4: while do 5: selected orbit <sup>=</sup> arg minok∈O minScm<sup>&</sup>gt;<sup>0</sup> ∣∣wcm <sup>−</sup> [Pcm∣pok ]Scm∣∣ 6: <sup>l</sup> <sup>=</sup> ∣∣wcm <sup>−</sup> [Pcm∣pselected orbit]Scm∣∣ 7: if l <sup>&</sup>lt; lmin then 8: lmin <sup>=</sup> l 9: <sup>P</sup>cm <sup>=</sup> [Pcm∣pselected orbit] 10: else 11: S.add(all scm,ok in the Scm) 12: break 13: end if 14: end while 15: end for 16: Return S

instance-level explanations, UO-Explainer extracts the orbit o ∗ <sup>v</sup>n,c<sup>m</sup> with the highest node-class-score as follows:

$$O_{v_n, c_m}^* = \arg \max_{o_k \in \mathcal{O}} \{ s_{v_n, c_m, o_0}, \dots, s_{v_n, c_m, o_k}, \dots, s_{v_n, c_m, o_K} \}. \quad (8)$$

Unlike the model-level explanation, which provides the orbit with the highest contribution and its corresponding graphlet as an explanation, instance-level explanations must provide subgraphs around the target node within the input graph. To achieve this, we use the search algorithm based on the Breadth-First Search (BFS). These algorithms initiate the search from the target node and explore neighboring nodes by verifying whether their connectivity matches the highest contributed orbit's corresponding graphlets. A detailed algorithm is shown in Appendix [B.](#page-13-1) Through this search, UO-Explainer can extract a subgraph within the input graph that matches the highest-contributing orbit for the target node, i.e., the explored subgraph is provided as an instance-level explanation along with the target node.

#### 3.4 TIME COMPLEXITY ANALYSIS

Training UO-Explainer consists of two main processes: orbit basis learning and class-orbit score learning, with the time complexity for each process detailed below. The time complexity of orbit basis learning is <sup>O</sup>(∣O∣ ∣V∣), where training is conducted for each orbit basis across all nodes in the input graph, as outlined in Algorithm [1.](#page-4-0) Here, ∣O∣ represents the number of orbits used for explanation, and ∣V∣ denotes the total number of nodes in the input graph. The time complexity for class-orbit score learning depends on the number of orbits selected through greedy search for each class weight. In our experiments, the number of orbits selected did not exceed 5, rendering this complexity component negligible. Consequently, the overall time complexity for this phase is represented as <sup>O</sup>(∣C∣), following the procedure in Algorithm [2,](#page-5-0) where C indicates the number of classes. In short, the overall time complexity for training UO-Explainer is <sup>O</sup>(∣O∣ ∣V∣ + ∣C∣) ≈ <sup>O</sup>(∣V∣).

The time complexities of the baseline methods are as follows: D4Explainer has a time complexity of <sup>O</sup>(∣V∣ 3 ), GNNExplainer is <sup>O</sup>(∣V∣ ∣E∣) where ∣E∣ denotes the number of edges, PGExplainer and TAGE have a time complexity of <sup>O</sup>(∣E∣), and MotifExplainer operates with a complexity of <sup>O</sup>(∣V∣ ∣M∣), where ∣M∣ represents the number of motifs used in explanations. Therefore, in terms of time complexity, UO-Explainer is less demanding compared to D4Explainer and GNNExplainer. Alongside such analysis, our unified model is capable of providing both model-level and instancelevel explanations simultaneously, thus demonstrating its competitiveness in terms of time efficiency. A more detailed time complexity analysis and experiments are described in Appendix [C.](#page-14-0)

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

## 4 RELATED WORK

GNN explanation methods can be categorized into model-level and instance-level, each with its own scope of explanation [\(Yuan et al., 2022\)](#page-12-5). Model-level methods [\(Yuan et al., 2020;](#page-12-6) [Shin et al.,](#page-11-10) [2022;](#page-11-10) [Azzolin et al., 2022\)](#page-10-6) offer explanations that describe the general behavior underlying GNN predictions regarding to specific class. For example, XGNN [\(Yuan et al., 2020\)](#page-12-6) aims to generate model-level patterns that maximize the predictive probability of a certain class by training a graph generator through reinforcement learning. Similarly, PAGE [\(Shin et al., 2022\)](#page-11-10) employs graph representation vectors to iteratively search for human-interpretable prototype graphs. Moreover, GLGExplainer [\(Azzolin et al., 2022\)](#page-10-6) provides general model-level patterns by aggregating instancelevel explanations into logical formulas, utilizing an Entropy-Logic Explainer (E-LEN) [\(Barbiero](#page-10-11) [et al., 2022;](#page-10-11) [Ciravegna et al., 2023\)](#page-10-12). However, these studies primarily focus on graph classification tasks and do not directly apply to node classification tasks. To bridge this gap, D4Explainer [\(Chen](#page-10-7) [et al., 2023\)](#page-10-7) introduces a method for providing counterfactual model-level explanations for node classification tasks, alongside instance-level explanations, based on a diffusion model. This approach, however, results in a high training cost for the explainer incurred by iterative diffusion- and deonising steps. Moreover, none of the methods for model-level explanations fully consider the user-centric explanation unit, hindering human-interpretability.

Instance-level methods [\(Yu & Gao, 2022;](#page-12-7) [Ying et al., 2019;](#page-12-2) [Luo et al., 2020;](#page-11-2) [Xie et al., 2022;](#page-12-3) [Vu & Thai, 2020;](#page-12-8) [Wang et al., 2023;](#page-12-9) [Xiong et al., 2023;](#page-12-10) [Zhang et al., 2023;](#page-12-4) [Ye et al., 2024;](#page-12-11) [Lu](#page-11-11) [et al., 2024\)](#page-11-11) provide explanations in the form of subgraph to elucidate the prediction of a specific instance by unveiling relational structures with high contributions in the input graph. Notably, GNNExplainer [\(Ying et al., 2019\)](#page-12-2) stands as an early instance-level explainer, optimizing edge and feature masks to maximize mutual information with GNN prediction results. PGExplainer [\(Luo](#page-11-2) [et al., 2020\)](#page-11-2) leverages node representation vectors and trains a parameterized mask predictor to optimize edge masks for explanation in inductive settings. TAGE [\(Xie et al., 2022\)](#page-12-3) explains the GNN embedding models, allowing efficient explanations for multiple downstream tasks. All these methods entail learning edge masks to present masked graphs as instance-level explanation subgraphs. Despite connectivity constraints, it often fails to generate connected subgraphs, resulting in less intuitive explanations. On the other hand, MotifExplainer [\(Yu & Gao, 2022\)](#page-12-7) provides an instance-level explanation that does not rely on edge masks. It computes embedding vectors for each motif and extracts explanations by restoring the GNN's prediction value using an attention network. However, it overlooks both the universal importance of specific motifs and the orbit-based isomorphism when extracting motifs. Furthermore, the abovementioned methods are limited to providing explanations in the unified framework capable of simultaneously providing both model-level and instance-level explanations under the human-interpretable units of interest.

## 5 EXPERIMENT

We evaluate explanations provided by UO-Explainer at the model-level and instance-level on synthetic and real-world datasets. These extensive experiments include quantitative and qualitative analysis of UO-Explainer's performance compared to recent baselines. For detailed experimental settings, please refer to Appendix [D.](#page-15-0)

#### 5.1 DATASETS AND BASELINES

We conducted experiments using five synthetic datasets and three real-world data sets. Synthetic datasets such as Random Graph [\(Holme & Kim, 2002\)](#page-10-13), BA-Shapes [\(Ying et al., 2019\)](#page-12-2), BA-Community, Tree-Cycle, and Tree-Grid are used to evaluate GNN explanation methods to compare the generated explanation based on pre-defined ground truths of each dataset. Also, real-world datasets such as Protein-Protein Interaction (PPI) [\(Zitnik & Leskovec, 2017\)](#page-12-12), LastFM-Asia [\(Rozemberczki](#page-11-12) [& Sarkar, 2020\)](#page-11-12), and Gene [\(Zitnik & Leskovec, 2017\)](#page-12-12) are used for node classification tasks. The detailed information and statistics of each dataset are described in Appendix [F.](#page-17-0)

Among the existing methods, D4Explainer [\(Chen et al., 2023\)](#page-10-7) and GLGExplainer [\(Azzolin et al.,](#page-10-6) [2022\)](#page-10-6) are the only existing framework that provides model-level explanations for the node classification task. For instance-level explanations, we set baselines based on common methods that learn edge masks, such as GNNExplainer [\(Ying et al., 2019\)](#page-12-2), PGExplainer [\(Luo et al., 2020\)](#page-11-2), TAGE

**381**

**384**

**386**

Table 1: Model-level explanation on random graph datasets: (a) with 2 or 3-layer GCN, and (b) with 2 or 3-layer GIN. Each task is to classify whether the node belongs to the orbit corresponding task number. The evaluation metric is the *Sub-recall*. The best performances are shown in bold.

|              |         |   |    |    |    |    |    |    |    |    |    |    |    |    |    | (a) |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|--------------|---------|---|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Task number  |         | 8 | 11 |    | 16 |    | 21 |    | 27 |    | 31 |    | 32 |    | 33 |     | 35 |    | 39 |    | 45 |    | 47 |    | 49 |    | 57 |    | 59 |    | 60 |    | 61 |    | 62 |    | 64 |    |
| Ground-truth | orbit o | 8 | o  | 11 | o  | 16 | o  | 21 | o  | 27 | o  | 31 | o  | 32 | o  | 33  | o  | 35 | o  | 39 | o  | 45 | o  | 47 | o  | 49 | o  | 57 | o  | 59 | o  | 60 | o  | 61 | o  | 62 | o  | 64 |
| D4Explainer  | 0       | 2 | 0  | 8  | 0  | 4  | 0  | 4  | 0  | 2  | 0  | 4  | 0  | 4  | 0  | 4   | 1  | 0  | 0  | 8  | 0  | 2  | 0  | 4  | 0  | 4  | 0  | 8  | 0  | 2  | 0  | 0  | 0  | 6  | 0  | 6  | 0  | 2  |
| GLGExplainer | 0       | 8 | 0  | 6  | 0  | 8  | 0  | 6  | 0  | 8  | 1  | 0  | 1  | 0  | 0  | 8   | 0  | 8  | 0  | 8  | 0  | 8  | 0  | 6  | 0  | 6  | 1  | 0  | 0  | 6  | 0  | 6  | 0  | 8  | 0  | 0  | 1  | 0  |
| UO-Explainer | 1       | 0 | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 0  | 0   | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 0  | 0  | 1  | 0  | 0  | 0  | 1  | 0  |
| Task number  |         | 8 | 11 |    | 16 |    | 21 |    | 27 |    | 31 |    | 32 |    | 33 |     | 35 |    | 39 |    | 45 |    | 47 |    | 49 |    | 57 |    | 59 |    | 60 |    | 61 |    | 62 |    | 64 |    |
| Ground-truth | orbit o | 8 | o  | 11 | o  | 16 | o  | 21 | o  | 27 | o  | 31 | o  | 32 | o  | 33  | o  | 35 | o  | 39 | o  | 62 | o  | 47 | o  | 49 | o  | 57 | o  | 59 | o  | 60 | o  | 61 | o  | 45 | o  | 64 |
| D4Explainer  | 0       | 8 | 0  | 6  | 0  | 6  | 1  | 0  | 0  | 6  | 0  | 8  | 0  | 8  | 0  | 6   | 1  | 0  | 1  | 0  | 0  | 6  | 0  | 4  | 0  | 8  | 1  | 0  | 0  | 8  | 0  | 8  | 0  | 2  | 0  | 4  | 0  | 6  |
| GLGExplainer | 0       | 8 | 0  | 8  | 1  | 0  | 0  | 4  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0   | 0  | 8  | 1  | 0  | 1  | 0  | 0  | 6  | 0  | 8  | 1  | 0  | 0  | 6  | 0  | 8  | 1  | 0  | 0  | 0  | 1  | 0  |
| UO-Explainer | 1       | 0 | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0   | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 1  | 0  |

Table 2: Model-level explanation results on synthetic datasets. The evaluation metric is the Sub-recall.

|              | Orbit o | class1 58 | o | BA-Shapes class2 57 | o | class3 56 | o | class1 58 | o | class2 57 | o | class3 56 | o | class5 58 | o | class6 57 | o | class7 56 | o | class1 73 | o | Tree-Grid class2 74 | o | class3 75 | o | Tree-Cycle class1 76 |
|--------------|---------|-----------|---|---------------------|---|-----------|---|-----------|---|-----------|---|-----------|---|-----------|---|-----------|---|-----------|---|-----------|---|---------------------|---|-----------|---|----------------------|
| D4Explainer  | 0       | 4         | 0 | 6                   | 0 | 4         | 0 | 8         | 0 | 2         | 0 | 8         | 0 | 0         | 0 | 6         | 0 | 4         | 0 | 6         | 0 | 0                   | 0 | 2         | 0 | 8                    |
| GLGExplainer | 1       | 0         | 1 | 0                   | 1 | 0         | 1 | 0         | 0 | 8         | 0 | 2         | 0 | 8         | 1 | 0         | 0 | 8         | 0 | 0         | 0 | 8                   | 0 | 2         | 1 | 0                    |
| UO-Explainer | 1       | 0         | 1 | 0                   | 1 | 0         | 1 | 0         | 1 | 0         | 1 | 0         | 0 | 8         | 1 | 0         | 1 | 0         | 0 | 8         | 1 | 0                   | 1 | 0         | 1 | 0                    |

[\(Xie et al., 2022\)](#page-12-3), MixupExplainer [\(Zhang et al., 2023\)](#page-12-4), SAME [\(Ye et al., 2024\)](#page-12-11), and EIG [\(Lu et al.,](#page-11-11) [2024\)](#page-11-11). Additionally, MotifExplainer [\(Yu & Gao, 2022\)](#page-12-7), which provides explanations based on motifs similar to our model, was also set as a baseline.

## 5.2 EVALUATION METRIC

We evaluate the quality of explanations using Sparsity, Fidelity, Edge-recall, and Sub-recall as evaluation metrics. *Sparsity* [\(Li et al., 2022\)](#page-11-13) refers to the ratio of edges in the explanation compared to the total number of edges in the computation graph of the target node. High sparsity implies that the proposed explanation has a small number of edges. *Fidelity* [\(Li et al., 2022\)](#page-11-13) measures the difference in the probability values when the explanation is excluded from the computation graph based on the target node. *Edge-recall* indicates how many edges in the explanations match the edges in the ground truths. *Sub-Recall* indicates the proportion of correct answers that the entire presented explanations match with ground truths. Notably, equations of *Sparsity* and *Fidelity* are described in the Appendix [D.4.](#page-16-0) Additionally, we conducted experiments five times and then reported the average and standard deviations in Appendix [E.](#page-17-1)

### 5.3 RESULTS: MODEL-LEVEL EXPLANATIONS

We first validate whether the UO-Explainer can identify the correct orbits for model-level explanations. We pre-train 2 or 3-layer GCN models [\(Kipf & Welling, 2017\)](#page-11-0) and 2 or 3-layer GIN models [\(Xu](#page-12-0) [et al., 2019\)](#page-12-0) on Random Graph datasets. We pre-train 2 or 3-layer GCN models [\(Kipf & Welling,](#page-11-0) [2017\)](#page-11-0) and 2 or 3-layer GIN models [\(Xu et al., 2019\)](#page-12-0) on Random Graph datasets. Each task is to classify whether the node belongs to the orbit corresponding task number. Consequently, explanation methods are expected to provide the ground-truth pattern in the form of the orbit (target node) with its corresponding graphlet (pattern) for each task. Tasks with accuracy below 0.8 are excluded as they are unlikely to yield accurate explanations. We use the *Sub-recall* metric to evaluate whether the provided explanation matches the ground-truth pattern.

In Table [1,](#page-7-0) UO-Explainer shows superior performance compared to other baselines. In particular, the UO-Explainer constantly provides model-level explanations matching to ground truths for all tasks except 33, 60, and 62 in the GCN model as shown in (a). This limitation may arise from the GNN's expressiveness, failing to learn intended orbits during the pre-training. To address this, we conducted experiments in the same manner using GIN, known for better expressiveness. The results are shown in (b) of Table [1](#page-7-0) that UO-Explainer accurately provides the explanations matching the

Table 3: Instance-level explanation results on synthetic datasets. The best performances on each dataset are shown in bold.

|                |   | Sub-recall |   | BA-Shapes Edge-recall |   | Fidelity |   | Sub-recall |   | BA-Community Edge-recall |   | Fidelity |   | Sub-recall |   | Tree-Grid Edge-recall |   | Fidelity |   | Sub-recall |   | Tree-Cycle Edge-recall |   | Fidelity |
|----------------|---|------------|---|-----------------------|---|----------|---|------------|---|--------------------------|---|----------|---|------------|---|-----------------------|---|----------|---|------------|---|------------------------|---|----------|
| GNNExplainer   | 0 | 004        | 0 | 616                   | 0 | 580      | 0 | 006        | 0 | 491                      | 0 | 653      | 0 | 000        | 0 | 629                   | 0 | 872      | 0 | 119        | 0 | 699                    | 0 | 724      |
| PGExplainer    | 0 | 760        | 0 | 915                   | 0 | 574      | 0 | 238        | 0 | 667                      | 0 | 652      | 0 | 000        | 0 | 647                   | 0 | 876      | 0 | 926        | 0 | 992                    | 0 | 732      |
| TAGE           | 0 | 682        | 0 | 900                   | 0 | 601      | 0 | 352        | 0 | 754                      | 0 | 672      | 0 | 003        | 0 | 693                   | 0 | 874      | 0 | 963        | 0 | 994                    | 0 | 734      |
| MixupExplainer | 0 | 696        | 0 | 906                   | 0 | 612      | 0 | 496        | 0 | 857                      | 0 | 693      | 0 | 047        | 0 | 712                   | 0 | 8770     | 0 | 930        | 0 | 994                    | 0 | 734      |
| SAME           | 0 | 343        | 0 | 720                   | 0 | 547      | 0 | 132        | 0 | 680                      | 0 | 642      | 0 | 000        | 0 | 238                   | 0 | 846      | 0 | 101        | 0 | 635                    | 0 | 692      |
| EiG-Search     | 0 | 878        | 0 | 520                   | 0 | 605      | 0 | 078        | 0 | 681                      | 0 | 695      | 0 | 004        | 0 | 723                   | 0 | 885      | 0 | 083        | 0 | 812                    | 0 | 703      |
| MotifExplainer | 0 | 873        | 0 | 890                   | 0 | 548      | 0 | 423        | 0 | 714                      | 0 | 683      | 0 | 793        | 0 | 857                   | 0 | 879      | 0 | 991        | 0 | 993                    | 0 | 736      |
| UO-Explainer   | 0 | 948        | 0 | 984                   | 0 | 623      | 0 | 921        | 0 | 970                      | 0 | 716      | 0 | 859        | 0 | 900                   | 0 | 888      | 1 | 000        | 1 | 000                    | 0 | 737      |

Table 4: Instance-level explanation results on real datasets. The best fidelity on each dataset is shown in bold.∗ notation indicates the lower sparsity setting.

|                |   | Fidelity | Task0 | Sparsity |   | Fidelity | Task1 | Sparsity |   | Fidelity | Task2 | Sparsity | PPI | Fidelity | Task3 | Sparsity |   | Fidelity | Task4 | Sparsity |   | Fidelity | Task5 | Sparsity |   | LastFM Fidelity |   | Asia Sparsity |
|----------------|---|----------|-------|----------|---|----------|-------|----------|---|----------|-------|----------|-----|----------|-------|----------|---|----------|-------|----------|---|----------|-------|----------|---|-----------------|---|---------------|
| GNNExplainer   | 0 | 100      | 0     | 973      | 0 | 325      | 0     | 999      | 0 | 417      | 0     | 999      | 0   | 480      | 0     | 999      | 0 | 250      | 0     | 999      | 0 | 331      | 0     | 999      | 0 | 114             | 0 | 974           |
| PGExplainer*   | 0 | 023      | 0     | 973      | 0 | 155      | 0     | 999      | 0 | 223      | 0     | 999      | 0   | 101      | 0     | 999      | 0 | 200      | 0     | 999      | 0 | 005      | 0     | 999      | 0 | 011             | 0 | 974           |
| TAGE*          | 0 | 031      | 0     | 973      | 0 | 109      | 0     | 999      | 0 | 233      | 0     | 999      | 0   | 138      | 0     | 999      | 0 | 214      | 0     | 999      | 0 | 101      | 0     | 999      | 0 | 086             | 0 | 974           |
| MixupExplainer | 0 | 005      | 0     | 973      | 0 | 002      | 0     | 999      | 0 | 257      | 0     | 999      | 0   | 246      | 0     | 999      | 0 | 246      | 0     | 999      | 0 | 129      | 0     | 999      | 0 | 100             | 0 | 974           |
| EiG-Search     | 0 | 180      | 0     | 973      | 0 | 269      | 0     | 999      | 0 | 180      | 0     | 999      | 0   | 379      | 0     | 999      | 0 | 100      | 0     | 999      | 0 | 180      | 0     | 999      | 0 | 095             | 0 | 974           |
| SAME           | 0 | 022      | 0     | 973      | 0 | 189      | 0     | 999      | 0 | 194      | 0     | 999      | 0   | 189      | 0     | 999      | 0 | 034      | 0     | 999      | 0 | 129      | 0     | 999      | 0 | 049             | 0 | 974           |
| MotifExplainer | 0 | 070      | 0     | 992      | 0 | 074      | 0     | 999      | 0 | 012      | 0     | 999      | 0   | 129      | 0     | 999      | 0 | 097      | 0     | 999      | 0 | 050      | 0     | 999      | 0 | 085             | 0 | 994           |
| UO-Explainer   | 0 | 423      | 0     | 999      | 0 | 358      | 0     | 999      | 0 | 425      | 0     | 999      | 0   | 510      | 0     | 999      | 0 | 623      | 0     | 999      | 0 | 413      | 0     | 999      | 0 | 115             | 0 | 993           |

correct ground-truth for all tasks. These findings confirm that UO-Explainer is capable of detecting various orbits and providing correct explanations while UO-Explainer's explanations mirror the expressiveness of the GNN's embedding model. Notably, comparing the performance of original GNNs to the decomposed class weights model, the performance degradation is less than 5%. On the other hand, D4Explainer and GLGExplainer show slightly improved or even decreased performance in the GIN setting and fail to provide consistent explanations matched with the ground truth.

As observed in Table [2,](#page-7-1) UO-Explainer also outperforms other baselines on the BA-Shapes and BA-Community dataset. Using 3-layer GCNs for this experiment, UO-Explainer successfully provides accurate model-level explanations for each class on both datasets. Specifically on the BA-Shapes dataset, we observe the explanations that the house-like motif plays a crucial role in node classification by detecting the orbit such as o56, o57, and o<sup>58</sup> in the graphlet g<sup>23</sup> as explanations. This explanation sheds light on the overall behavior of the GNN beyond individual nodes, enabling a broader interpretation of GNNs. Moreover, our approach provides orbit-corresponding graphlets, allowing us to determine the topological positions of nodes within the motif for each class. Thus, UO-Explainer shows that the GNN recognizes the house-like motif as an important pattern for node classification, assigning nodes on the roof, floor, and top of the roof of the house-like motif to classes 1, 2, and 3, respectively. On the BA-Community dataset, UO-Explainer also finds the ground-truth pattern as the model-level explanation by detecting the orbit such as o56, o57, and o<sup>58</sup> in the graphlet g<sup>23</sup> as explanations, since the dataset is a union of two BA-SHAPES graphs. In conclusion, the experimental result demonstrates that orbits as pre-defined explanation units of UO-Explainer serve crucial patterns of the specific classes for prediction, showing accurate and consistent explanations.

#### 5.4 RESULTS: INSTANCE-LEVEL EXPLANATIONS

Except for MotifExplainer, all baselines provide explanations in the form of subgraphs obtained by extracting edges exceeding specific threshold values or ranking the top-k edge considered important. For fair experiments, explanations composed of top-k edges were extracted considering a sparsity level similar to that of UO-Explainer. In the case of MotifExplainer, we extract one motif as an instance-level explanation. UO-Explainer used a subgraph within the input graph that matches the highest-contributing orbit for the target node for the instance-level explanation as mentioned in Section [3.3.](#page-4-2)

The experimental results on synthetic datasets are shown in Table [3.](#page-8-0) UO-Explainer outperforms the baseline methods across all evaluation metrics while maintaining a comparable sparsity level. Notably, UO-Explainer achieves higher sub-recall, indicating accurate detection of ground-truth subgraphs as explanations. In cases such as Tree-grid and Tree-cycle, where grid- and cycle-shaped graphlets do not exist within the 2- to 5-node graphlets, we employ grid and cycle graphlets along with their corresponding orbits and include them as units of explanation. We note that by defining custom graphlets based on background knowledge or extracted rules tailored to specific problem

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

![](_page_9_Figure_1.jpeg)

Figure 4: Visualization of the explanations provided in the Gene dataset. Each node represents the ID of a gene, and the red nodes correspond to the target gene mentioned in the explanation. The red edges denote the edges included in the subgraph provided as part of the explanation.

settings, the proposed method can be extended beyond the pre-defined 2- to 5-node graphlets and orbits.

Table [4](#page-8-1) shows the performance on real datasets, where UO-Explainer outperforms other baselines in the fidelity metric, except for task 0 of the PPI dataset. MotifExplainer struggles due to extracting too many motifs, leading to less meaningful explanations. PGExplainer and TAGE also show poor performance, often identifying the same or irrelevant edges across nodes, regardless of experimental changes. GNNExplainer, trained iteratively for node-specific explanations, extracts more relevant edges, especially on larger datasets. The ∗ notation in Table [4](#page-8-1) highlights that lower sparsity is needed for competing methods to match UO-Explainer, yet they still include noise edges or less important subgraphs. In contrast, UO-Explainer performs consistently well even under high sparsity, using just one orbit-based subgraph for explanations, demonstrating its ability to provide high-quality instance-level explanations.

#### 5.5 CASE STUDY ON GENE DATASET

As the qualitative analysis, we visualized the explanatory subgraphs described in our method and the baselines on the gene dataset. The visualization results are presented in Figure [4.](#page-9-0) The experiment results demonstrate that PGExplainer and TAGE provide scattered subgraphs with discontinuous edges while the sparsity remains at 0.900 for fair comparison. In contrast, UO-Explainer offers a connected subgraph that is more intuitive while maintaining relatively high fidelity. MotifExplainer also presents a connected subgraph as an explanation but with relatively lower fidelity. Additionally, several studies provide evidence that the genes (TGFBR2 [\(Massague & Gomis, 2006\)](#page-11-14), ENG [\(Breen](#page-10-14) ´ [et al., 2013\)](#page-10-14), INHBA, and ACVR2B [\(Attisano & Wrana, 2013\)](#page-10-15)) identified by UO-Explainer in the explanations have an impact on the surface receptor signaling pathway, which is the label of the dataset. For example, in [\(Bottino et al., 2021\)](#page-10-16), it was mentioned that TGFBR2 is one of the TGF-beta receptors that transmit signals within natural killer cells, exerting a significant influence on cell development and the function of natural killer cells. These results imply that UO-Explainer provides human-interpretable explanations compared to other baselines regarding the perspective of interest.

## 6 DISCUSSION AND CONCLUSION

We introduce UO-Explainer, a human-interpretable explanation method that leverages pre-defined units, as requested by users, in a unified framework for node classification models. By utilizing orbits as explanatory units, UO-Explainer decomposes model weights into orbit components, which serve as essential, human-interpretable units within graph domains. Experimental results on both synthetic and real-world datasets demonstrate the effectiveness of UO-Explainer, outperforming baseline methods and delivering higher-quality explanations. UO-Explainer is particularly valuable in scientific applications, such as drug development and education, where domain knowledge is critical. By using pre-defined explanation units, users can uncover meaningful patterns and gain deeper insights through the explanation method.

**554 555 556**

**559**

**561**

**564**

**569**

**579**

**584**

## REFERENCES


[1] Nesreen K Ahmed, Jennifer Neville, Ryan A Rossi, and Nick Duffield. Efficient graphlet counting for large networks. In *Proceeding of the International Conference on Data Mining*, pp. 1–10. IEEE, 2015. Liliana Attisano and Jeffrey L Wrana. Signal integration in tgf-β, wnt, and hippo pathways. *F1000prime reports*, 5, 2013. Steve Azzolin, Antonio Longa, Pietro Barbiero, Pietro Lio, and Andrea Passerini. Global explain- ` ability of gnns via logic combination of learned concepts. *arXiv preprint arXiv:2210.07147*, 2022. Federico Baldassarre and Hossein Azizpour. Explainability techniques for graph convolutional networks, 2019. URL <https://arxiv.org/abs/1905.13686>. Pietro Barbiero, Gabriele Ciravegna, Francesco Giannini, Pietro Lio, Marco Gori, and Stefano ´ Melacci. Entropy-based logic explanations of neural networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pp. 6046–6054, 2022. Cristina Bottino, Thierry Walzer, Angela Santoni, and Roberta Castriconi. Tgf-β as a key regulator of nk and ilcs development and functions, 2021. Michael J Breen, Diarmuid M Moran, Wenzhe Liu, Xiaoke Huang, Calvin PH Vary, and Raymond C Bergan. Endoglin-mediated suppression of prostate cancer invasion is regulated by activin and bone morphogenetic protein type ii receptors. *PLoS One*, 8(8):e72407, 2013. Jialin Chen, Shirley Wu, Abhijit Gupta, and Rex Ying. D4explainer: In-distribution gnn explanations via discrete denoising diffusion. *Advances in neural information processing systems*, 2023. Xiaowei Chen and John CS Lui. Mining graphlet counts in online social networks. *ACM Transactions on Knowledge Discovery from Data*, 12(4):1–38, 2018. Gabriele Ciravegna, Pietro Barbiero, Francesco Giannini, Marco Gori, Pietro Lio, Marco Maggini, ´ and Stefano Melacci. Logic explained networks. *Artificial Intelligence*, 314:103822, 2023. David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alan´ Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural information processing systems*, 28, 2015. Rafael Espejo, Guillermo Mestre, Fernando Postigo, Sara Lumbreras, Andres Ramos, Tao Huang, and Ettore Bompard. Exploiting graphlet decomposition to explain the structure of complex networks: the ghust framework. *Scientific reports*, 10(1):12884, 2020. Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. Graph neural networks for social recommendation. In *Proceedings of the Web Conference*, 2019. Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. *arXiv preprint arXiv:1903.02428*, 2019. Thomas Gaudelet, Ben Day, Arian R Jamasb, Jyothish Soman, Cristian Regep, Gertrude Liu, Jeremy BR Hayter, Richard Vickers, Charles Roberts, Jian Tang, et al. Utilizing graph machine learning within drug discovery and development. *Briefings in bioinformatics*, 22(6):bbab159, 2021. Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30, 2017. Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d'Amato, Gerard de Melo, Claudio Gutierrez, Sabrina Kirrane, Jose Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, et al. Knowledge ´ graphs. *ACM Computing Surveys (CSUR)*, 54(4):1–37, 2021. Petter Holme and Beom Jun Kim. Growing scale-free networks with tunable clustering. *Physical review E*, 65(2):026107, 2002.

[2] **594 595 596 597 598 599 604 606 608 609 610 611 614 615 617 619 624 625 626 627 629 634 636 639 640 641 642 643 644 645 646 647** Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In *Proceedings of the International Conference on Learning Representations*, 2017. Risi Kondor, Nino Shervashidze, and Karsten M. Borgwardt. The graphlet spectrum. In *Proceedings of the International Conference on Machine Learning*, 2009. Yiqiao Li, Jianlong Zhou, Sunny Verma, and Fang Chen. A survey of explainable graph neural networks: Taxonomy and evaluation metrics. *arXiv preprint arXiv:2207.12599*, 2022. Meng Liu, Youzhi Luo, Limei Wang, Yaochen Xie, Hao Yuan, Shurui Gui, Haiyang Yu, Zhao Xu, Jingtun Zhang, Yi Liu, et al. Dig: A turnkey library for diving into graph deep learning research. *The Journal of Machine Learning Research*, 22(1):10873–10881, 2021. Shengyao Lu, Bang Liu, Keith G. Mills, Jiao He, and Di Niu. Eig-search: Generating edge-induced subgraphs for gnn explanation in linear time. In *International Conference on Machine Learning*, 2024. Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang Zhang. Parameterized explainer for graph neural network. *Advances in neural information processing systems*, 2020. Joan Massague and Roger R Gomis. The logic of tgf ´ β signaling. *FEBS letters*, 580(12):2811–2820, 2006. Hiromi Nakagawa, Yusuke Iwasawa, and Yutaka Matsuo. Graph-based knowledge tracing: modeling student proficiency using graph neural network. In *Proceeding of the IEEE/WIC/ACM International Conference on Web Intelligence*, 2019. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019. Phillip E Pope, Soheil Kolouri, Mohammad Rostami, Charles E Martin, and Heiko Hoffmann. Explainability methods for graph convolutional neural networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10772–10781, 2019. Mario Alfonso Prado-Romero, Bardh Prenkaj, Giovanni Stilo, and Fosca Giannotti. A survey on graph counterfactual explanations: Definitions, methods, evaluation, and research challenges. *ACM Computing Surveys*, 56(7):1–37, April 2024. ISSN 1557-7341. doi: 10.1145/3618105. URL <http://dx.doi.org/10.1145/3618105>.
  - N. Przulj, D. G. Corneil, and I. Jurisica. Modeling interactome: scale-free or geometric? ˇ *Bioinformatics*, 20(18):3508–3515, 2004. Natasa Pr ˇ zulj. Biological network comparison using graphlet degree distribution. ˇ *Bioinformatics*, 23 (2):e177–e183, 2007. Benedek Rozemberczki and Rik Sarkar. Characteristic functions on graphs: Birds of a feather, from statistical descriptors to parametric models. In *Proceedings of the International Conference on Information Knowledge Management*, 2020. Michael Sejr Schlichtkrull, Nicola De Cao, and Ivan Titov. Interpreting graph neural networks for nlp with differentiable edge masking, 2022. URL <https://arxiv.org/abs/2010.00577>. Nino Shervashidze, SVN Vishwanathan, Tobias Petri, Kurt Mehlhorn, and Karsten Borgwardt. Efficient graphlet kernels for large graph comparison. In *Artificial intelligence and statistics*, pp. 488–495. PMLR, 2009. Yong-Min Shin, Sun-Woo Kim, Eun-Bi Yoon, and Won-Yong Shin. Prototype-based explanations for graph neural networks (student abstract). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2022.

[3] **654**

[4] **656**

[5] **659**

[6] **661**

[7] **664 665**

[8] **669**

[9] **674**

[10] **684**

[11] **686**

[12] **689 690 691**

[13] Aravind Subramanian, Pablo Tamayo, Vamsi K. Mootha, Sayan Mukherjee, Benjamin L. Ebert, Michael A. Gillette, Amanda Paulovich, Scott L. Pomeroy, Todd R. Golub, Eric S. Lander, and Jill P. Mesirov. Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles. *Proceedings of the National Academy of Sciences*, 102(43): 15545–15550, 2005. Petar Velickovi ˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua ´ Bengio. Graph attention networks. In *Proceedings of the International Conference on Learning Representations*, 2018. Minh Vu and My T Thai. Pgm-explainer: Probabilistic graphical model explanations for graph neural networks. *Advances in neural information processing systems*, 2020. Jihong Wang, Minnan Luo, Jundong Li, Yun Lin, Yushun Dong, Jin Song Dong, and Qinghua Zheng. Empower post-hoc graph explanations with information bottleneck: A pre-training and fine-tuning perspective. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2349–2360, 2023. Yaochen Xie, Sumeet Katariya, Xianfeng Tang, Edward Huang, Nikhil Rao, Karthik Subbian, and Shuiwang Ji. Task-agnostic graph explanations. *Advances in neural information processing systems*, 2022. Ping Xiong, Thomas Schnake, Michael Gastegger, Gregoire Montavon, Klaus Robert Muller, and ´ Shinichi Nakajima. Relevant walk search for explaining graph neural networks. In *International Conference on Machine Learning*, pp. 38301–38324, 2023. Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *Proceedings of the International Conference on Learning Representations*, 2019. Ziyuan Ye, Rihan Huang, Qilin Wu, and Quanying Liu. Same: Uncovering gnn black box with structure-aware shapley-based multipiece explanations. *Advances in Neural Information Processing Systems*, 36, 2024. Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer: Generating explanations for graph neural networks. *Advances in neural information processing systems*, 32, 2019. Zhaoning Yu and Hongyang Gao. Motifexplainer: a motif-based graph neural network explainer. *arXiv preprint arXiv:2202.00519*, 2022. Hao Yuan, Jiliang Tang, Xia Hu, and Shuiwang Ji. Xgnn: Towards model-level explanations of graph neural networks. In *Proceedings of the SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp. 430–438, 2020. Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in graph neural networks: A taxonomic survey, 2022. URL <https://arxiv.org/abs/2012.15445>. Jiaxing Zhang, Dongsheng Luo, and Hua Wei. Mixupexplainer: Generalizing explanations for graph neural networks with data augmentation. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 3286–3296, 2023. Marinka Zitnik and Jure Leskovec. Predicting multicellular function through multi-layer tissue networks. *Bioinformatics*, 33(14):i190–i198, 2017.
## A THE FULL SET OF GRAPHLETS AND ORBITS

![](_page_13_Figure_12.jpeg)

Figure 5: Entire graphlets having 2-5 nodes and orbits. The same color nodes within each graphlet belong to the same orbit.

In this work, we have defined the explanatory units as 0-72 orbits and their corresponding graphlet ranging from 0 to 29 consisting of 2 to 5 nodes. The full set of utilized graphlets and orbits can be observed in Figure 5 (Pržulj et al., 2004; Pržulj, 2007).

## B ORBIT SEARCH ALGORITHM

In Section 3.1, we have mentioned the implementation of a graphlet search algorithm based on the Breadth-First Search (BFS) algorithm for extracting the orbit with the highest score and its corresponding graphlet on the target node. These algorithms initiate the search from the target node and explore neighboring nodes by verifying whether their connectivity matches the desired graphlets. We present two algorithmic examples for identifying specific orbits in our study. Algorithm 3 describes the process of identifying the  $o_{10}$  with its corresponding graphlet  $g_5$ , while Algorithm 4 outlines the steps for identifying the  $o_{56}$  with its corresponding graphlet  $g_{23}$ . In both algorithms,  $N(v)$  denotes the neighbor function to construct the set of neighboring nodes around a given node  $v$ . Additionally,  $C(\cdot, n)$  denotes a combination function to generate the set of combinations consisting of  $n$  elements from a given set  $\cdot$ .

---

Algorithm 3 Orbit $o_{10}$ Search Algorithm

---

**Input:** A graph  $\mathcal{G} = (\mathcal{V}, \mathcal{E})$  where  $\mathcal{V} = \{v_1, \dots, v_i, \dots, v_{|\mathcal{V}|}\}$  denotes a set of node and  $\mathcal{E} = \{(v_i, v_j), \dots, (v_k, v_l)\}$  denotes a set of edges, Target node  $v_n$ .

**Output:** A set  $L$  which contains edge sets of  $g_5$  graphlets that make target node belong to orbit  $o_{10}$ .

**Initialize**  $L$  as an empty set

1. 1: Neighbor set of the target node  $\mathcal{N}_{v_n} = N(v_n)$  where  $N$  denotes the neighbor function.
2. 2: A combination set with three elements  $\mathcal{C}_1 = C(\mathcal{N}_{v_n}, 3)$  where  $C$  denotes Combination function
3. 3: **for**  $v'_a, v'_b, v'_c$  in  $\mathcal{C}_1$  **do**
4. 4:     a set of candidate combinations  $\mathcal{C}'_1 = \{(v'_a, v'_b, v'_c), (v'_b, v'_a, v'_c), (v'_c, v'_a, v'_b)\}$
5. 5:     **for**  $v_a, v_b, v_c$  in  $\mathcal{C}'_1$  **do**
6. 6:         **if**  $(v_b, v_c) \in \mathcal{E}$  and  $\{(v_a, v_b), (v_a, v_c)\} \notin \mathcal{E}$  **then**
7. 7:              $L.add(\{(v_n, v_a), (v_n, v_b), (v_n, v_c), (v_b, v_c)\})$
8. 8:         **end if**
9. 9:     **end for**
10. 10: **end for**
11. 11: **Return**  $L$

---

We present a detailed line-by-line explanation of Algorithm 3 as an exemplary demonstration of the orbit search process. This algorithm is designed to find  $g_5$  graphlets in which the target node belongs to orbit  $o_{10}$ . In line 1, we generate the neighbor set  $\mathcal{N}_{v_n}$  from the target node  $v_n$ . Since the number of neighboring nodes within orbit  $o_{10}$  in graphlet  $g_5$  is three, line 2 constructs the set

**759**

**761**

**764**

**766**

**769**

**779 780 781**

**784**

**804 805 806**

C<sup>1</sup> by forming combinations of three nodes from the neighbor set. Within graphlet <sup>g</sup>5, there is one orbit, o8, that distinguishes it from the two neighboring nodes, o9. Therefore, to ensure that each node combination in C<sup>1</sup> also distinguishes between <sup>o</sup><sup>10</sup> and <sup>o</sup>9, line 4 generates the set C 1 , which reflects requirement. Notably, the first nodes of each tuple in the set are distinct from the remaining nodes. By observing the connectivity of the nodes in lines 5-6, if it matches the connectivity of the orbits within g5, the edge set of g<sup>5</sup> can be obtained in line 7. It is added to the set L. This algorithm iterates by considering all possible combinations of neighboring nodes surrounding the target node vn. Therefore, all existing graphlet g<sup>5</sup> that make the target node belong to orbit o<sup>10</sup> are in the output set L. This format can be applied to find entire 2-5 node graphlets. However, it is computationally expensive. To alleviate this computational cost, by introducing randomness during the generation of the neighbor set (line 1) or combination set in the algorithm(line 2 and line 4), it becomes possible to sample the desired number of graphlets. The performance as the number of samples is discussed in Table [7.](#page-16-1)

Algorithm 4 Orbit o<sup>56</sup> Search Algorithm

Input: A graph G <sup>=</sup> (V, E) where V <sup>=</sup> {v1, <sup>⋯</sup>, vi, <sup>⋯</sup>, v∣V∣} denotes a set of nodes and E <sup>=</sup> {(vi, v<sup>j</sup> ), <sup>⋯</sup>, (vk, vl)} denotes a set of edges, Target node vn. Output: A set L which contains edge sets of g<sup>23</sup> graphlets that make target node belong to orbit o56. Initialize L as an empty set 1: Neighbor set of the target node Nvn <sup>=</sup> <sup>N</sup>(vn) where <sup>N</sup> denotes the neighbor function. 2: A combination set with three elements <sup>C</sup><sup>1</sup> <sup>=</sup> <sup>C</sup>(Nvn , <sup>2</sup>) where <sup>C</sup> denotes Combination function 3: for va, v<sup>b</sup> in C<sup>1</sup> do 4: if vb, v<sup>c</sup> <sup>∈</sup> E then 5: Neighbor set of <sup>v</sup>a, Nva <sup>=</sup> <sup>N</sup>(va) 6: Neighbor set of <sup>v</sup>b, <sup>N</sup>vb <sup>=</sup> N(vb) 7: for <sup>v</sup><sup>c</sup> in Nva do 8: for <sup>v</sup><sup>d</sup> in <sup>N</sup>vb do 9: if (vc, vd) <sup>∈</sup> E and {(vn, vc), (vn, vd), (vb, vc), (va, vd)} /⊂ E then 10: L.add({(vn, va), (vn, vb), (va, vb), (va, vc), (vb, vd), (vc, vd)}) 11: end if 12: end for 13: end for 14: end if 15: end for 16: Return L

## C TIME COMPLEXITY ANALYSIS

When UO-Explainer provides instance-level explanations, there are four time-consuming processes: 1) Pre-processing for finding the existence of the orbit, 2) Orbit basis learning, 3) Class-orbit score learning (Model-level explanation generation), and 4) Generating instance-level explanations.

1) Pre-processing for finding the existence of each orbit. As mentioned in Section [3.1,](#page-3-3) to learn the orbit basis, we must find the existence of each orbit for every node, as represented by the equation [1.](#page-3-4) This pre-processing can be conducted to search 2-5 node graphlets for each node. The time complexity of finding each graphlet that includes specific orbits, as can be inferred from Algorithm [3](#page-13-3) and Algorithm [4,](#page-14-1) is <sup>O</sup>(<sup>d</sup> k−1 ) [1](#page-14-2) . Here, d represents the maximum node degree of the graph data, and k denotes the number of nodes included in each graphlet and is less than 5. Therefore, the time complexity of pre-processing for finding all 2-5 node graphlets that include a total 0-72 orbit for an entire node of graph data is <sup>O</sup>(∣O∣ ∣V∣ <sup>d</sup> k−1 ) where ∣O∣ denotes the number of edges and ∣V∣ represents the number of nodes.

2) Orbit basis learning. The time complexity of orbit basis learning is <sup>O</sup>(∣O∣ ∣V∣) since training is performed for each orbit basis over all nodes as shown in Algorithm [1.](#page-4-0)

3) Class-orbit score learning. Class-orbit score is trained based on the number of orbits selected by the greedy search for each class weight. In our experiments, the number of selected orbits was less than or equal to 5, so this can be sufficiently neglected. Therefore, the time complexity of class-orbit score learning considers only the number or class C; <sup>O</sup>(∣C∣) as shown in Algorithm [2.](#page-5-0) The training time required for UO-Explainer and the baselines can be found in Table [7.](#page-16-1) Model-level explanations

<sup>1</sup>As referenced from N. K. Ahmed, J. Neville, R. A. Rossi, and N. Duffield. Efficient Graphlet Counting for Large Networks. *In Proceedings of the International Conference on Data Mining*, 2015, pp. 1-10.

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

Table 5: The training time required for the baselines and UO-Explainer for explanation in the gene dataset. The training time of the explainer is significantly influenced by the epochs. Therefore, we set the epochs for each method through a hyper-parameter search to achieve the highest fidelity.

|                  | <b>UO-Explainer</b>                                        | <b>GNNExplainer</b> | <b>MotifExplainer</b> | <b>PGExplainer</b> | <b>TAGE</b> |
|------------------|------------------------------------------------------------|---------------------|-----------------------|--------------------|-------------|
| Training Time(s) | 268 (orbit basis learning)+21 (class-orbit score learning) | 1,681               | 6,074                 | 98                 | 204         |

Table 6: The training time required for the baselines and UO-Explainer for explanation in the gene dataset. The training time of the explainer is significantly influenced by the epochs. Therefore, we set the epochs for each method through a hyper-parameter search to achieve the highest fidelity.

|                  | Instance and Model-level                                    |              | Instance-level             |      | Model-level |
|------------------|-------------------------------------------------------------|--------------|----------------------------|------|-------------|
|                  | UO-Explainer                                                | GNNExplainer | MotifExplainer PGExplainer | TAGE | D4Explainer |
| Training Time(s) | 268 (orbit basis learning)+ 21 (class-orbit score learning) | 1 , 681      | 6 , 074 98                 | 204  | 502         |

can be provided immediately after class-orbit score learning. Therefore, the training time of the UO-Explainer listed in Table [6](#page-15-1) equals the time required to provide model-level explanations.

4) Generating instance-level explanations. To provide instance-level explanations, it is necessary to search graphlets that include the given orbits as explanations for each node and select just one graphlet having the highest fidelity. Therefore, the time complexity is <sup>O</sup>(∣V∣ <sup>d</sup> k−1 ) since graphlets need to be directly found for each node. However, instead of finding all graphlets around each node, we sample a predetermined number of graphlets. Thus, the actual computational cost can be significantly reduced compared to <sup>O</sup>(∣V∣ <sup>d</sup> k−1 ), while preserving high explanation performance. The required time and fidelity based on the number of sampled graphlets are shown in Table [7.](#page-16-1) Even with a time difference of more than 7 times, as observed when comparing the cases of not sampling and sampling 70 graphlets for explanation extraction, it can be observed that the fidelity values do not significantly decrease.

## D DETAILED EXPERIMENTAL SETTINGS

In this section, we cover the details of experiments that were not discussed in the main text, such as the structure of the pre-trained GNN and the hyper-parameter settings of the UO-Explainer.

## D.1 DETAILS OF PRE-TRAINED GNNS

We provide detailed information about pre-trained GNNs for datasets other than the random graph, as already described in Section [5.3.](#page-7-2) For each dataset, we utilized a pre-trained GNN architecture consisting of a 3-layer GCN as the embedding model and a 1-layer MLP as the downstream model. The hyper-parameters and split ratio are shown in Table [8.](#page-16-2)

## D.2 HYPER-PARAMETER SETTINGS FOR UO-EXPLAINER

UO-Explainer has a total of five hyper-parameters including batch size, epochs, learning rate (LR) for orbit basis learning, and additional epochs, learning rate (LR) for class-orbit score learning. Hyper-parameters used in the experiments are reported in Table [9.](#page-17-2)

## D.3 DETAILS OF BASELINES

In our code implementation, we primarily utilized the PyTorch [\(Paszke et al., 2019\)](#page-11-15) and PyTorch Geometric [\(Fey & Lenssen, 2019\)](#page-10-17) frameworks. Additionally, for the GNNExplainer and PGExplainer, we employed implementations from the PyTorch Geometric framework. For the TAGE, we employed implementations from the Dive into Graph (DIG) [\(Liu et al., 2021\)](#page-11-16) framework. For MotfiExplainer, we utilized the code provided in the supplementary material available on OpenReview.net, which was suitably modified following advice received in correspondence with the authors. This revised implementation is incorporated within our publicly accessible code.

Furthermore, to ensure the equity of our experiments, we diligently endeavored to explore the hyper-parameter space of the baselines as extensively as possible. The explored hyper-parameter

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

Table 7: The time required and fidelity based on the number of sampled graphlets of UO-Explainerin the gene dataset. We conducted five experiments for each method and reported the mean and standard deviation values. The "Entire" refers to the extraction of all graphlets surrounding the target node without sampling.

| # of sampled graphlets | 1              | 10              | 30               | 50              | 70              | 100              | Entire           |
|------------------------|----------------|-----------------|------------------|-----------------|-----------------|------------------|------------------|
| Time(s)                | 14 572 ± 0 208 | 136 342 ± 2 305 | 379 342 ± 22 438 | 597 283 ± 3 865 | 838 991 ± 3 587 | 1183 459 ± 5 913 | 6321 783 ± 8 462 |
| Fildelity              | 0 157 ± 0 018  | 0 256 ± 0 009   | 0 297 ± 0 011    | 0 309 ± 0 014   | 0 326 ± 0 008   | 0 324 ± 0 007    | 0 3543 ± 0       |

Table 8: Details of pre-trained GNNs. LR means learning rate.

| Dataset          | BA-Sahpes | BA-Community | PPI0  | PPI1  | PPI2 PPI3   | PPI4  | PPI5  | LastFM Asia | Gen   | Tree-Cycle | Tree-Grid |
|------------------|-----------|--------------|-------|-------|-------------|-------|-------|-------------|-------|------------|-----------|
| LR               | 0.001     | 0.001        |       |       | 0.003       |       |       | 0.001       | 0.001 | 0.001      | 0.001     |
| Epoch            | 2,000     | 5,000        |       |       | 300         |       |       | 600         | 100   | 600        | 3,000     |
| Hidden Dimension | 10        | 30           |       |       | 200         |       |       | 30          | 100   | 30         | 30        |
| Train:Val:Test   | 8:1:1     | 8:1:1        |       |       | 10:1:1      |       |       | 8:1:1       | 8:1:1 | 8:1:1      | 10:1:1    |
| Train Accuracy   | 0.982     | 0.998        | 9.998 | 0.998 | 0.997 0.998 | 0.994 | 0.998 | 0.997       | 0.999 | 0.993      | 0.987     |
| Val Accuracy     | 0.943     | 0.723        | 0.953 | 0.959 | 0.970 0.962 | 0.976 | 0.981 | 0.814       | 0.744 | 0.966      | 0.878     |
| Test Accuracy    | 0.971     | 0.800        | 0.971 | 0.967 | 0.973 0.957 | 0.980 | 0.936 | 0.841       | 0.686 | 0.966      | 0.863     |

space for GNNExplainer includes (lr, epoch, edge size, node feature size, edge entropy, and node feature entropy). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (100-300-600, 0.01-0.05-0.1, 0.005-0.01-0.1, 0.05-0.1, 0.5-1.0, 0.5-1.0) that yielded the highest performance. The explored hyper-parameter space for PGExplainer includes (lr, epoch, edge size, edge entropy, and dimension of MLP). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (0.001- 0.003-0.01-0.5-0.1, 0.01-0.05-0.1, 0.05-0.1, 64-128) that yielded the highest performance. The explored hyper-parameter space for TAGE includes (lr, epoch, batch size coefficient size, coefficient entropy, and loss type). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (0.000005-0.00005-0.0005-0.005-0.05-0.1, 1-3-5-10-20, 4-16- 64-128, 0.01-0.05, 0.0005-0.005, 'NCE'-'JSE') that yielded the highest performance. The explored hyper-parameter space for MotifExplainer includes (lr, epoch, embedding dimension of attention module). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (0.001-0.005-0.01, 30-50-100) that yielded the highest performance. Furthermore, the motif extraction rules for MotifExplainer encompass three categories: 1) cycles within the graph, 2) edges excluding cycles, and 3) motifs constituted by the fusion of cycles involving two or more identical nodes. In our experiments, when the number of motifs extracted from the graphs was fewer than 5000, all three rules were employed to conduct the experiments. However, in cases where the number exceeded 5000 motifs, a random sample of 5000 motifs was utilized.

### D.4 EVALUATION METRICS

Sparsity [\(Li et al., 2022\)](#page-11-13) means the proportion of the presented explanation in the computation graph based on the target node as follows:

$$Sparsity = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} 1 - \frac{|\mathcal{G}_{v_i}^{ex}|}{|\mathcal{G}_{v_i}|} \quad (9)$$

where G ex vi denotes the subgraph presented as the explanation for the node v<sup>i</sup> , G<sup>v</sup><sup>i</sup> denotes to the computation graph, and ∣G∣ means the number of edges in the graph G.

Fidelity [\(Li et al., 2022\)](#page-11-13) is calculated by taking the difference in the probability values of the input graph and the probability values when the explanation is excluded from the computation graph based on the target node, as follows:

$$Fidelity = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} f_{prob}(\mathcal{G}_{v_i}) - f_{prob}(\mathcal{G}_{v_i} - \mathcal{G}_{v_i}^{ex}) \quad (10)$$

**924**

**929**

**954**

**956**

**959**

**961**

Table 9: Hyper-parameter settings for UO-Explainer. LR means learning rate.

|        |              | Dataset |           | Random Graph | BA-Shapes | BA-Community | PPI   | LastFM-Asia | Gene  | Tree-Cycle | Tree-Grid |
|--------|--------------|---------|-----------|--------------|-----------|--------------|-------|-------------|-------|------------|-----------|
| Epochs | (Orbit       | basis   | learning) | 3,000        | 3,000     | 3,000        | 3,000 | 1,000       | 1,000 | 1,000      | 1,000     |
| LR     | (Orbit       | basis   | learning) | 0.005        | 0.005     | 0.005        | 0.001 | 0.003       | 0.005 | 0.005      | 0.005     |
| Batch  | size (Orbit  | basis   | learning) | 256          | 256       | 256          | 256   | 2048        | 256   | 256        | 256       |
| Epochs | (Class-orbit | score   | learning) | 2,000        | 1,000     | 1,000        | 1,000 | 1,000       | 1,000 | 1,000      | 1,000     |
| LR     | (Class-orbit | score   | learning) | 0.003        | 0.005     | 0.005        | 0.005 | 0.005       | 0.005 | 0.005      | 0.005     |

Table 10: Statistics of the dataset used for the experiment. # symbols mean the number.

| Dataset        | Random Graph | BA-Shapes | BA-Community | PPI    | LastFM-Asia | Gene   | Tree-Cycle | Tree-Grid |
|----------------|--------------|-----------|--------------|--------|-------------|--------|------------|-----------|
| Avg # of nodes | 340          | 700       | 1,400        | 2,373  | 7,624       | 857    | 871        | 1,231     |
| Avg # of edges | 2,690        | 4,110     | 8,920        | 66,136 | 55,612      | 13,992 | 1,950      | 3,410     |
| # of classes   | 2            | 4         | 8            | 2      | 18          | 2      | 2          | 4         |
| # of tasks     | 73           | 1         | 1            | 121    | 1           | 1      | 1          | 1         |

## E EXPERIMENTAL RESULTS WITH STANDARD DEVIATION.

We conducted five experiments for each method by recording the mean and standard deviation in Table [11,](#page-18-0) [12,](#page-18-1) [13,](#page-19-0) and [14.](#page-19-1)

## F DATASETS

We conducted experiments using five synthetic datasets and three real-world datasets. The statistics of datasets are shown in Table [10.](#page-17-3) Additionally, the detailed information on each dataset is described below.

Random Graph [\(Holme & Kim, 2002\)](#page-10-13): This dataset does not have node feature values, so we performed node classification tasks using only the structural information by setting all node feature values to the same constant value. We set the class of each node based on the existence of its orbit same as each task number. That is, each node has two classes, c<sup>0</sup> (orbit doesn't exist) and c<sup>1</sup> (orbit exists) for a total of 73 independent tasks, one for each orbit. The ground truth for class c<sup>1</sup> is the target orbit o<sup>k</sup> for each task.

BA-Shapes [\(Ying et al., 2019\)](#page-12-2): The node classification dataset consists of a BA graph with 300 nodes as the base and 80 "house-like motifs" randomly attached, each consisting of 5 nodes. The house-like motif precisely matches graphlet g23. The class is determined by the positions of the nodes within the house-like motif; these node positions precisely match the orbits o58, o57, o<sup>56</sup> in g23. Therefore, the ground truth for each class is set to o58, o57, o56.

BA-Community [\(Ying et al., 2019\)](#page-12-2): The node classification dataset is generated by randomly combining two house-like motifs through edges. The first motif is assigned classes c0, c1, c2, c<sup>3</sup> based on the positions of the nodes, and the second motif is assigned c4, c5, c6, c7, resulting in a total of eight class labels. The ground truth of each motif is the same as that of the BA-Shapes dataset.

Tree-Cycle [\(Ying et al., 2019\)](#page-12-2): The dataset consists of a balanced binary tree as the base and 80 "six-node cycle motifs" randomly attached. The task is to classify nodes as either not belonging (c0) to or belonging to a circle (c1). The ground truth for c<sup>1</sup> class corresponds to the circle motif. However, 2-5 node graphlets do not encompass the six-node circle motif. Thus, we conducted experiments by incorporating the circle motif and the orbit within the circle into candidate graphlets and orbits.

Tree-Grid [\(Ying et al., 2019\)](#page-12-2): The base is identical to that of the Tree-Cycle, with a modification involving the attachment of a "nine-node grid" in place of the circle. The task has been further challengingly augmented to classify each node as belonging to the center (c1), periphery (c2), or cross-lines (c3) based on their positions within the grid. Similar to the case of Tree-Cycle, 2-5 node graphlets do not contain the nine-node grid. Therefore, we included the grid motif and the orbits within the grid in the candidate set for experimentation.

Protein-Protein Interaction (PPI) [\(Zitnik & Leskovec, 2017\)](#page-12-12): This is a node classification dataset that represents protein-protein interactions in various human tissues using 24 different graphs. Each node represents a protein and features such as positional, motif gene, and immunological signature are used with a size of 50 dimensions. Protein-to-protein interactions are represented as edges. Each

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

**1017**

Table 11: Model-level explanation on random graph datasets: (a) with 2 or 3-layer GCN, and (b) with 2 or 3-layer GIN. Each task is to classify whether the node belongs to the orbit corresponding task number. The evaluation metric is the *Sub-recall*. We conducted experiments five times and then reported the average and standard deviations. The best performances are shown in bold.

|                                       |                               |                               |                               |                               |                               |                               |                               |                             | (a)                           |                               |                               |                               |                               |                               |                               |                             |                               |                             |                               |
|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-----------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-----------------------------|-------------------------------|-----------------------------|-------------------------------|
| Task number                           | 8                             | 11                            | 16                            | 21                            | 27                            | 31                            | 32                            | 33                          | 35                            | 39                            | 45                            | 47                            | 49                            | 57                            | 59                            | 60                          | 61                            | 62                          | 64                            |
| Ground-truth orbit                    | o 8                           | o 11                          | o 16                          | o 21                          | o 27                          | o 31                          | o 32                          | o 33                        | o 35                          | o 39                          | o 45                          | o 47                          | o 49                          | o 57                          | o 59                          | o 60                        | o 61                          | o 62                        | o 64                          |
| D4Explainer GLGExplainer UO-Explainer | 0 2 ± 0 4 0 8 ± 0 4 1 0 ± 0 0 | 0 8 ± 0 4 0 6 ± 0 5 1 0 ± 0 0 | 0 4 ± 0 5 0 8 ± 0 4 1 0 ± 0 0 | 0 4 ± 0 5 0 6 ± 0 5 1 0 ± 0 0 | 0 2 ± 0 4 0 8 ± 0 4 1 0 ± 0 0 | 0 4 ± 0 5 1 0 ± 0 0 1 0 ± 0 0 | 0 4 ± 0 5 1 0 ± 0 0 1 0 ± 0 0 | 0 4 ± 0 5 0 8 ± 0 4 0 ± 0 0 | 1 0 ± 0 0 0 8 ± 0 4 1 0 ± 0 0 | 0 8 ± 0 4 0 8 ± 0 4 1 0 ± 0 0 | 0 2 ± 0 4 0 8 ± 0 4 1 0 ± 0 0 | 0 4 ± 0 5 0 6 ± 0 5 1 0 ± 0 0 | 0 4 ± 0 5 0 6 ± 0 5 1 0 ± 0 0 | 0 8 ± 0 4 1 0 ± 0 0 1 0 ± 0 0 | 0 2 ± 0 4 0 6 ± 0 5 1 0 ± 0 0 | 0 0 ± 0 0 0 6 ± 0 5 0 ± 0 0 | 0 6 ± 0 5 0 8 ± 0 4 1 0 ± 0 0 | 0 6 ± 0 5 0 0 ± 0 0 0 ± 0 0 | 0 2 ± 0 4 1 0 ± 0 0 1 0 ± 0 0 |

(b) Task number 8 11 16 21 27 31 32 33 35 39 45 47 49 57 59 60 61 62 64 Ground-truth orbit o<sup>8</sup> o<sup>11</sup> o<sup>16</sup> o<sup>21</sup> o<sup>27</sup> o<sup>31</sup> o<sup>32</sup> o<sup>33</sup> o<sup>35</sup> o<sup>39</sup> o<sup>62</sup> o<sup>47</sup> o<sup>49</sup> o<sup>57</sup> o<sup>59</sup> o<sup>60</sup> o<sup>61</sup> o<sup>45</sup> o<sup>64</sup> D4Explainer <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>4</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>2</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>4</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> GLGExplainer <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>4</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>6</sup> <sup>±</sup> <sup>0</sup>.<sup>5</sup> <sup>0</sup>.<sup>8</sup> <sup>±</sup> <sup>0</sup>.<sup>4</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>0</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> UO-Explainer <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup> <sup>1</sup>.<sup>0</sup> <sup>±</sup> <sup>0</sup>.<sup>0</sup>

Table 12: Model-level explanation results on synthetic datasets. The evaluation metric is the Subrecall. We conducted 5 experiments for each experiment and reported consistent results.

| Ground-truth Orbit | MA Balance     |                |                |                | MA Balance     |                |                |                | MA Balance     |                |                |  |
|--------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|--|
|                    | class1         | class2         | class3         | class4         | class3         | class3         | class4         | class7         | class1         | class2         | class3         |  |
| D4                 | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          | 0.958          |  |
| GLEGExplainer      | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        | 1.0±0.0        |  |
| <b>UnO</b>         | <b>1.0±0.0</b> | <b>1.0±0.0</b> | <b>1.0±0.0</b> | <b>1.0±0.0</b> | <b>1.0±0.0</b> | <b>0.8±0.4</b> | <b>1.0±0.0</b> | <b>1.0±0.0</b> | <b>0.8±0.4</b> | <b>1.0±0.0</b> | <b>1.0±0.0</b> |  |

node is labeled with 121 dimensions of gene ontology sets, allowing for binary classification of 121 categories, similar to a random graph. We pre-trained a GNN on 20 of these graphs and extracted explanations for randomly selected graphs to conduct experiments.

LastFM-Asia [\(Rozemberczki & Sarkar, 2020\)](#page-11-12): This is a social network dataset of LastFM users in Asia. Each node represents a LastFM user in Asian countries, and edges represent the following relationships between them. Node features are composed of artists that users like, which we converted to one-hot encoding for use in experiments. We perform a node classification task to predict the country of each user across 18 countries.

Gene: For qualitative experiments, we created a straightforward dataset using the gene network of natural killer cells in humans, as provided by [\(Zitnik & Leskovec, 2017\)](#page-12-12). Each node of the dataset represents genes, while the features include positional genes, motif genes, and immunological signature information of each gene, with each feature consisting of 50 dimensions. The Molecular Signatures Database [\(Subramanian et al., 2005\)](#page-12-13) was used to collect the features for each gene. The labels in the dataset indicate whether a given gene belongs to the cell surface receptor signaling pathway, with 1 indicating inclusion and 0 indicating exclusion in the gene ontology set.

## G LIMITATION AND FUTURE WORK

In this study, we propose UO-Explainer, a GNN explanation method that utilizes graphlets and orbits as explanation units. Though we have demonstrated that UO-Explainer can provide high-quality explanations compared to other baselines on extensive experiments, the motifs we can employ as explanatory units are limited to 2-5 node graphlets. Consequently, we assume thatthis constraint may occasionally hinder the capture of motifs or patterns when crucial patterns are extremely large when the input graph is complex. Since many existing works already can approximate the various shaped subgraphs but do not consider the user-centric perspective, we aim to more focus on the case when prior assumption or knowledge is crucial based on pre-defined units toward human-interpretable explanations.

Table 13: Instance-level explanation results on synthetic datasets. The best performances on each dataset are shown in bold. We conducted five experiments for each method by recording the mean and standard deviation.

|              | B-Shape       |               |               | BA-Community  |               |               | Free-Shape    |               |               | Tree-Circle   |               |  |
|--------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|--|
|              | Shape-recall  | Fidelity      | Fidelity      | Shape-recall  | Fidelity      | Fidelity      | Shape-recall  | Fidelity      | Fidelity      | Shape-recall  | Fidelity      |  |
| ENMESpanier  | 0.034 ± 0.016 | 0.016 ± 0.010 | 0.050 ± 0.003 | 0.036 ± 0.003 | 0.036 ± 0.003 | 0.053 ± 0.012 | 0.050 ± 0.003 | 0.029 ± 0.003 | 0.073 ± 0.003 | 0.119 ± 0.000 | 0.724 ± 0.009 |  |
| PEEpanier    | 0.076 ± 0.025 | 0.195 ± 0.003 | 0.574 ± 0.020 | 0.228 ± 0.015 | 0.676 ± 0.011 | 0.524 ± 0.017 | 0.000 ± 0.000 | 0.676 ± 0.042 | 0.876 ± 0.022 | 0.926 ± 0.005 | 0.992 ± 0.005 |  |
| PEEpanier    | 0.076 ± 0.025 | 0.195 ± 0.003 | 0.574 ± 0.020 | 0.228 ± 0.015 | 0.676 ± 0.011 | 0.524 ± 0.017 | 0.000 ± 0.000 | 0.676 ± 0.042 | 0.876 ± 0.022 | 0.926 ± 0.005 | 0.992 ± 0.005 |  |
| MixupEpanier | 0.096 ± 0.012 | 0.092 ± 0.012 | 0.012 ± 0.012 | 0.046 ± 0.029 | 0.057 ± 0.012 | 0.093 ± 0.014 | 0.047 ± 0.012 | 0.712 ± 0.003 | 0.877 ± 0.011 | 0.930 ± 0.012 | 0.734 ± 0.004 |  |
| SAME         | 0.324 ± 0.033 | 0.272 ± 0.009 | 0.547 ± 0.003 | 0.320 ± 0.009 | 0.620 ± 0.010 | 0.624 ± 0.006 | 0.000 ± 0.000 | 0.326 ± 0.013 | 0.86 ± 0.010  | 0.101 ± 0.017 | 0.65 ± 0.023  |  |
| Gire-Sareh   | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.078 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.724 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |
| Moot-Mixther | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.078 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |
| TOP-Epanier  | 0.038 ± 0.016 | 0.034 ± 0.016 | 0.032 ± 0.003 | 0.052 ± 0.003 | 0.037 ± 0.003 | 0.052 ± 0.003 | 0.035 ± 0.023 | 0.038 ± 0.023 | 0.038 ± 0.023 | 0.000 ± 0.000 | 0.000 ± 0.000 |  |

Table 14: Instance-level explanation results on real datasets. The best fidelity on each dataset is shown in bold. ∗ notation indicates the lower sparsity setting. We conducted five experiments for each method by recording the mean and standard deviation.

|                                  |       |             | Fidelity |       | Task0       |       | Sparsity    |       |             | Fidelity |       | Task1       |       | Sparsity    |       |             | Fidelity |       | Task2       |       | Sparsity    | PPI   |             | Fidelity |       | Task3       |       | Sparsity    |       |             | Fidelity |       | Task4       |       | Sparsity    |       |             | Fidelity |       | Task5       |       | Sparsity    |       |             | Fidelity |       | LastFM      | Asia  | Sparsity    |
|----------------------------------|-------|-------------|----------|-------|-------------|-------|-------------|-------|-------------|----------|-------|-------------|-------|-------------|-------|-------------|----------|-------|-------------|-------|-------------|-------|-------------|----------|-------|-------------|-------|-------------|-------|-------------|----------|-------|-------------|-------|-------------|-------|-------------|----------|-------|-------------|-------|-------------|-------|-------------|----------|-------|-------------|-------|-------------|
| GNNExplainer                     | 0     | 100         | ±        | 0     | 016         | 0     | 973         | 0     | 325         | ±        | 0     | 004         | 0     | 999         | 0     | 417         | ±        | 0     | 067         | 0     | 999         | 0     | 480         | ±        | 0     | 012         | 0     | 999         | 0     | 250         | ±        | 0     | 001         | 0     | 999         | 0     | 331         | ±        | 0     | 001         | 0     | 999         | 0     | 114         | ±        | 0     | 005         | 0     | 974         |
| PGExplainer*                     | 0     | 023         | ±        | 0     | 00          | 0     | 973         | 0     | 155         | ±        | 0     | 008         | 0     | 999         | 0     | 223         | ±        | 0     | 077         | 0     | 999         | 0     | 101         | ±        | 0     | 067         | 0     | 999         | 0     | 200         | ±        | 0     | 001         | 0     | 999         | 0     | 005         | ±        | 0     | 000         | 0     | 999         | 0     | 011         | ±        | 0     | 002         | 0     | 974         |
| TAGE* MixupExplainer             | 0 0   | 031 005     | ± ±      | 0 0   | 002 000     | 0 0   | 973 973     | 0 0   | 109 002     | ± ±      | 0 0   | 003 001     | 0 0   | 999 999     | 0 0   | 233 257     | ± ±      | 0 0   | 052 002     | 0 0   | 999 999     | 0 0   | 138 246     | ± ±      | 0 0   | 034 003     | 0 0   | 999 999     | 0 0   | 214 246     | ± ±      | 0 0   | 019 003     | 0 0   | 999 999     | 0 0   | 101 129     | ± ±      | 0 0   | 020 001     | 0 0   | 999 999     | 0 0   | 086 100     | ± ±      | 0 0   | 008 001     | 0 0   | 974 974     |
| EiG-Search                       | 0     | 180         | ±        | 0     | 072         | 0     | 973         | 0     | 269         | ±        | 0     | 032         | 0     | 999         | 0     | 180         | ±        | 0     | 053         | 0     | 999         | 0     | 379         | ±        | 0     | 073         | 0     | 999         | 0     | 100         | ±        | 0     | 079         | 0     | 999         | 0     | 180         | ±        | 0     | 072         | 0     | 999         | 0     | 095         | ±        | 0     | 021         | 0     | 974         |
| SAME MotifExplainer UO-Explainer | 0 0 0 | 022 070 423 | ± ± ±    | 0 0 0 | 002 008 007 | 0 0 0 | 973 992 999 | 0 0 0 | 189 074 358 | ± ± ±    | 0 0 0 | 003 003 022 | 0 0 0 | 999 999 999 | 0 0 0 | 194 012 425 | ± ± ±    | 0 0 0 | 001 028 000 | 0 0 0 | 999 999 999 | 0 0 0 | 189 129 510 | ± ± ±    | 0 0 0 | 004 034 057 | 0 0 0 | 999 999 999 | 0 0 0 | 034 097 623 | ± ± ±    | 0 0 0 | 005 005 029 | 0 0 0 | 999 999 999 | 0 0 0 | 129 050 413 | ± ± ±    | 0 0 0 | 033 021 027 | 0 0 0 | 999 999 999 | 0 0 0 | 049 085 115 | ± ± ±    | 0 0 0 | 012 004 009 | 0 0 0 | 974 994 993 |