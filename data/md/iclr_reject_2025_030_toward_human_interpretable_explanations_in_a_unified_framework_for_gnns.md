000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 As Graph Neural Networks (GNNs) are increasingly applied across various domains, explainability has become a critical factor for real-world applications. Existing post-hoc explainability methods primarily focus on estimating the importance of edges, nodes, or subgraphs in the input graph to identify substructures crucial for predictions. However, these methods often lack human interpretability and do not provide a unified framework that incorporates both model-level and instance-level explanations. In this context, we propose leveraging a set of graphlets—small, connected, non-isomorphic induced subgraphs widely used in various scientific fields—and their associated orbits as human-interpretable units to decompose GNN predictions. Domain experts can select the most relevant graphlets as interpretable units and request unified explanations based on these units. To address this problem, we introduce UO-Explainer, the Unified and Orbit-based Explainer for GNNs, which utilizes predefined orbits that are generalizable and universal across graph domains as interpretable units. Our model decomposes GNN weights into orbit units to extract class-specific graph patterns (model-level) and to identify important subgraphs within individual data instances for prediction (instance-level). Extensive experimental results demonstrate that UO-Explainer outperforms existing baselines in providing meaningful and interpretable explanations across both synthetic and real-world datasets. Our code and datasets are available at https://anonymous.4open.science/r/uoexplainer-F12C.

## 1 Introduction

Graph Neural Networks (GNNs) have achieved state-of-the-art performance in various domains including real-world graph-structured data, such as social networks (Fan et al., 2019), molecules (Duvenaud et al., 2015), and knowledge graphs (Hogan et al., 2021). Despite the remarkable advancements in GNN architectures (Hamilton et al., 2017; Kipf & Welling, 2017; Xu et al., 2019; Velickovi ˇ c´ et al., 2018), they are still perceived as black box models due to their lack of explainability. This deficiency limits trust in GNN predictions, hindering their application in areas such as drug development (Gaudelet et al., 2021) and education (Nakagawa et al., 2019). Therefore, interpreting the prediction of GNNs has become crucial and has led to the emergence of various explanatory approaches. Explainability methods in graph domains deliver explanations through subgraphs that play a significant role in predictions regarding the input graph. Two primary issues arise regarding explainability: i) a human-interpretability and ii) a unified framework encompassing both model and instance levels. Many existing post-hoc explanability is grounded on perturbation-based methods (Ying et al., 2019; Luo et al., 2020; Xie et al., 2022; Zhang et al., 2023; Schlichtkrull et al., 2022) and gradient-based methods (Baldassarre & Azizpour, 2019; Pope et al., 2019) to approximate the importance of edges or nodes within subgraphs. This stochastic optimization of importance is computationally effective in applying any graph-structured data, these methods have the potential risk that the output subgraph does not align with prior human assumption or knowledge. Specifically, consider a scientist studying gene networks who is interested in understanding whether the presence of certain structures, such as triangles or rectangles, plays a crucial role in predicting specific properties. This scientist would want to audit the model's predictions by utilizing an explanation method that can highlight the importance of these structures. With existing methods, it is challenging to obtain explicit insights into whether a triangular or rectangular structure in the network is more important; instead, users often have to rely

# Toward Human-Interpretable Explanations In A Unified Framework For Gnns

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 on conjecture based on the generated explanations. Additionally, these explanations do not always present results in a human-intuitive form, as they may include isolated nodes or disconnected edges, further hindering their interpretability. Therefore, we emphasize the need for explainability that centers on human interpretability, a perspective that has been largely underexplored. Last but not least, human-interpretable explanations should be delivered in a unified framework, incorporating both global and local levels in terms of the scope of explanation (Yuan et al., 2022; Prado-Romero et al., 2024). Most of the existing methods primarily specialize one one-level explanation either model or instance-level explanations. Model-level methods reveal patterns that GNNs deem significant for specific classes, eventually offering a broad understanding of the GNNs' general behavior. On the other hand, instance-level methods focus on individual predictions, identifying the subgraphs most relevant to the target node or graph. As each type of explanation complements the other from different perspectives, understanding at both levels enhances the explainability necessary for grasping the decision-making process of GNNs. However, State-of-the-art (Azzolin et al., 2022; Chen et al., 2023) studies also have rarely explored a unified framework that simultaneously provides both model-level and instance-level explanations in the user-centric perspective of interest. In this paper, we prioritize the two crucial perspectives that explanation should be human-interpretable in the unified framework incorporating both model and instance levels. Our proposed model, UO- Explainer, the Unified and Orbit-based Explainer for GNNs, allows users to harness their prior knowledge by giving room to select the user-defined explanation units in unified views. Considering the uniqueness of the graph domain to define explanation unit, we exploit orbits within graphlets that have been studied as a generalizable and universal pattern in many scientific fields such as protein interaction networks Przulj et al. (2004); Pr ˇ zulj (2007), social networks Chen & Lui (2018); Ahmed ˇ et al. (2015), and molecular structure networks Kondor et al. (2009), while users can also adopt a unique prior perspective to define their unit instead of orbit. To provide unified explanations, we decompose the weights into orbit representation vectors to understand the contribution of each orbit for a specific class or prediction. When we break down the weight into orbits, we acknowledge the contribution of each orbit for model decision-making, which is supposed to be further discussed in the method section in detail. Through extensive experiments, we demonstrate the superior performance of UO-Explainer on eight well-known datasets. Consequently, UO-Explainer shows the concrete performance in a unified framework leveraging the human prior knowledge as orbit generalizable unit in graph-structured datasets. In summary, the contributions of our research are as follows:
- We propose UO-Explainer, a unified framework to provide both model-level and instancelevel explanations for node classification based on human-defined explanation units.

- We demonstrate the human interpretable explanation based on orbits generalizable and effective on graph-structured datasets.

- We perform rigorous and extensive experiments on 8 datasets to evaluate the quality of our explanations in both model and instance-level explanations.

## 2 Preliminary 2.1 Graph Neural Networks (Gnns)

We represents a graph as G = (V, E; A, X) where E denotes a edge set and V = {v1, v2,⋯, v∣V∣}
denotes a node set. A ∈ R
∣V∣×∣V∣denotes the adjacent matrix and X ∈ R
∣V∣×din denotes the node feature matrix. In this study, we focus on GNNs for node classification tasks as presented in (Kipf & Welling, 2017; Xu et al., 2019). A GNN model f(⋅) maps input graph into prediction matrix f(A, X) = Z ∈ R
∣V∣×∣C∣, C = {c1,⋯, c∣C∣} in where C as the set of classes. GNNs can be expressed as a composite function f = fD - fE of an embedding-model fE(⋅) and a downstream-model fD(⋅).

The embedding model embeds an adjacent matrix and node feature matrix into a node representation matrix H, i.e., fE(A, X) = H ∈ R
∣V∣×d. The representation vector of each node vn is denoted by the hvn, n-th row vector of matrix H. The downstream-model maps the node representation matrix into the prediction matrix to classify nodes into each class, i.g., fD(H) = HW+b = Z where W ∈ R
d×∣C∣
denotes weight matrix and b ∈ R
∣C∣denotes bias vectors. In this operation, only the m-th column 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 2.2 Orbits Within Graphlets

2-node graphlet 3-node graphlet **4-node graphlet**
!

" #
$
'
(
% )
*

"%
⋯
5-node graphlet
! " # $ %

"*
##
#$
)( )*
)' (#
⋯ ⋯
"(
"& #!

")
"'
#"
& "! "" #$ #&
Graphlets as predominantly observed patterns are pre-defined subgraphs with a small number of nodes (Przulj et al., 2004). Figure 1 shows ˇ some examples of 2-5 node graphlets (Przulj ˇ
et al., 2004; Przulj, 2007), where ˇ gl denotes the l-th graphlet. The full set of 2-5 node graphlets used in our study is presented in Appendix A. All graphlets are non-isomorphic to each other, indicating that each graphlet has a unique structure. Each graphlet contains nodes with identical or distinguishable topological positions known as **orbits**, e.g., the central node of g1 belongs to o2 due to its distinguishable position, whereas the remaining nodes belong to o1. The set of orbits is represented as O = {o0,⋯, ok,⋯, o∣O∣} where ok denotes the k-th orbit. Previous studies have highlighted the usefulness of graphlet-based analysis in various graph data domains, including protein interaction networks (Przulj et al., 2004; Pr ˇ zulj, 2007), social networks (Chen ˇ
& Lui, 2018; Ahmed et al., 2015), and molecular structure networks (Kondor et al., 2009). For example, (Przulj, 2007) defined the Graphlet Degree Distribution (GDD) using 2-5 node graphlets and ˇ orbits as units to analyze agreement among biological and chemical networks. (Shervashidze et al., 2009; Espejo et al., 2020) further compared the empirical similarity of various chemical compound networks using a 2-5 node graphlet kernel. Since these analyses indicate that 2- to 5-node graphlets can serve as simple yet effective units for interpreting graph data, we primarily employ them to provide orbit-based explanations, unless otherwise specified by the user.

Figure 1: Graphlets having 2-5 nodes and 0-72 orbits. The same color nodes within each graphlet belong to the same orbit.

## 3 Uo-Explainer

UO-Explainer serves as a unified explainer capable of delivering both model-level and instance-level explanations for node classification tasks. To deliver explanations in a human-interpretable way, we exploit a pre-defined set of 0-72 orbits as the explanatory unit, which is recognized as essential and human-interpretable units within graph domains, while users can also apply other meaning units regarding their perspective instead of orbits. Upon the interpretable unit, we decompose the weights vector wcm of weight matrix W and the m-th element bcm of bias vectors b are involved in the computation to predict the class cm, i.g., zcm = Hwcm + bcm where zcm denotes the m-th row vector of matrix Z. Weight regards to specific class as wcm affects only the prediction of the class cm, so we call this weight vector a class weight. Furthermore, the prediction value zvn,cm for the class of each node (the n-th row and m-th column element of Z) can be expressed as zvn,cm = hvn⋅ wcm + bcm, computed by the operation between each node representation vector and class weight.

Instance-level Explanation Class 1 on target node &

!!,""
∗ =  !!,"",$&, ⋯ , !!,"",$# ⋯ , !!,"",$'
≈0.1× ⨂
+0.4× ⨂+ ⋯
Prediction value decomposition
⨂
Node representation vector (ℎ!!)Class weights ("") Prediction values
(!!,"")
Preprocessing Node-class-orbit score (!!,"",$&)
Class weight decomposition Model-level Explanation
""
∗ =  "",$&, ⋯ , "",$#, ⋯ , "",$'
Class 1 Class 2
≈0.1× +0.4× +0.3× +0.2× Orbit Basis Learning Class-orbit score

("",$#)Orbit basis
($#)
"$

"%
≈0.2× +0.2× +0.1× +0.4×

## 3.1 Orbit Basis Learning

that directly influence classification concerning two components such as an embedding model and a downstream task model. An overview of UO-Explainer is presented in Figure 2.

$$y_{v_{n},o_{k}}={\begin{cases}0&\quad{i f\;v_{n}\;d o e s n^{\prime}t\;b e l o n g\;t o\;o_{k}}\\ 1&\quad{i f\;v_{n}\;b e l o n g\;t o\;o_{k}.}\end{cases}}$$

1 if vn *belongs to o*k.(1)
We present the toy example of this pre-processing through Figure 3. (a) portrays the input graph. (b) represents the graphlets and orbits employed in the pre-processing. For simplicity, let us assume that only the graphlets g2, g3, g6 are utilized and the orbits used for pre-processing are o3, o4, o5, o11. Different color nodes inside each graphlet refer to nodes belonging to different orbits. (c) is a substantial pre-processing process, which checks whether each node can belong to each orbit and assigns 1 or 0 to the orbit's existence. The dark green node represents the pre-processing node. You can see that node 3 belong to o3, o5, o11. Therefore, yv3,o3, yv3,o4, yv3,o5, yv3,o11 are assigned values of 1, 1, 0, 1, respectively. This pre-processing is performed for all orbits in 2-5 node graphlets at all nodes within the input graph. To decompose class weights into orbit units requires orbit bases, which are the representation vectors of each orbit. Orbit bases must necessarily reflect the following two aspects: (1) The distribution of each orbit within the input graph, and (2) The message passing and aggregation behavior of the embedding model. To meet the first requirement, we pre-process the orbit-existences on each node in the input graph. Orbit existence is denoted as yvn,okand determines whether each node vn belongs to the orbit ok.

() () () ()
"
$!,&!": 1
$!,&#": 0 $!,&$": 1
$!,&"": 1 '
!

Pre-processing on node "
'
#
)

Existence of each orbit on node 3
" #
)
(

"
!!

$#,&!": 0 $#,&#": 1
$#,&$": 0 $#,&"": 1 Graphlets and orbits Input Graph
(

Pre-processing on node #
Existence of each orbit on node 4

$$(1)$$

Next, we train a logistic binary classifier to predict the existence of each orbit, initializing each orbit with pˆokvector, and taking node representation as input, described by the following equation:
yˆvn,ok = *sigmoid*(pˆok⋅ hvn). (2)
To satisfy the second aspect, we learn the orbit basis by incorporating the node representations from the node embedding model when training. Then, we apply normalization in Equation as pok =pˆok
∣∣pˆok
∣∣ to ensure that the size of the orbit basis remains constant. Orbit-basis learning is conducted for all orbits, and details of orbit basis learning can be found in Algorithm 1.

## 3.2 Model-Level Explanations

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Figure 3: Detailed example of the pre-processing step. The dark green node represents the target node for explanation. The colors of nodes within the graphlets represent the orbits respectively.

Model-level explanations are provided by decomposing class weights into a linear combination of orbit bases, as the following equation:
$$\begin{array}{l l}{{P_{0}}}&{{}}\\ {{}}&{{}}\\ {{1}}&{{\approx S_{c m},o_{0}\,{\bf p}o_{0}+\cdots+s_{c m},o_{k}\,{\bf p}o_{k}+\cdots+s_{c m},o_{K}\,{\bf p}o_{K}\,,}}\\ {{}}&{{}}\\ {{}}&{{\approx\sum_{o_{k}\in{\cal O}}s_{c m},o_{k}\,{\bf p}o_{k}\,.}}\end{array}$$
$\mathbf{W}_{\mathrm{C}\mathrm{T}}$
.(3)
$$\operatorname*{min}_{s_{c m},o_{k}>0}\left\|\mathbf{w}_{c_{m}}-\sum_{o_{k}\epsilon\mathcal{O}}s_{c_{m},o_{k}}\mathbf{p}_{o_{k}}\right\|.$$
. (4)
Generally, when a vector is expressed as a linear combination of bases, the coefficients of each basis indicate to what extent they contribute to forming the vector. Accordingly, the coefficients of orbit bases are regarded as contributions to the class weights. Furthermore, the bases are learned by considering each orbit distribution, thereby treating the contribution of orbit basis as the contribution of each orbit. We define the contribution of orbit ok to the class cm classification as a class-orbit score scm,ok.

The class-orbit scores are trained by the following objective function derived from Equation 3:

$$({\mathfrak{I}})$$
$$(4)$$

4

## Algorithm 1 Orbit Basis Learning

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Input: A set of node representation vectors {hv1
, ⋯, hvn , ⋯, hv∣V∣
}, a set of existences of each orbits for each nodes
{yv1,o0
, yv2,o0
, ⋯, yv1,o23 , ⋯, yv∣V∣
,o72 }.

Output: A set of orbit bases P = {po0
, ⋯, pok
, ⋯, po72 }
Initialize P as an empty set 1: for k = 0 to 72 (the number of orbits used) do 2: Initialize pˆok as a random vector.

3: for n=1 to ∣V∣ (the number of nodes) do 4: yˆ*vn,ok*
= *sigmoid*(pˆok
⋅ hvn )
5: L = BCE(y*vn,ok*
, yˆ*vn,ok*
)
6: Update pˆok via ∇pˆok L
7: **end for**
8: pok
=
pˆok
∣∣pˆok
∣∣
9: *P.add*(pok
)
10: **end for** 11: **Return** P
To consider only the positive impact of the contributing orbits, we limit the class-orbit score to positive. However, directly optimizing the objective function for all orbit bases involves a significant amount of randomness. For example, in the worst case, if all orbit bases are orthogonal, and the dimension of the class weight is d < ∣O∣, an infinite number of combinations of scm,ok can be found to optimize the Equation 4. This randomness hinders the learning of the correct contribution of orbits. Therefore, we modify the objective function using a greedy approach by selecting the orbit that minimizes the difference between the class weight and the linear combination of selected orbits in each iteration, as shown in the following equation:

$$\operatorname*{arg\,min}_{o_{k}\in\mathcal{O}}\operatorname*{min}_{\mathbf{S}_{c_{m}}>0}\|\mathbf{w}_{c_{m}}-[\mathbf{P}_{c_{m}}|\mathbf{p}_{o_{k}}]\mathbf{S}_{c_{m}}\|.$$
$$({\boldsymbol{5}})$$

∣∣wcm − [Pcm∣pok]Scm∣∣. (5)
Pcm is a matrix consisting of the selected pok as columns, and Scm represents a column vector consisting scm,okof the selected orbits. [Pcm∣pok] denotes concatenation of the pokto Pcm as a column, e.g., if orbits 1, 3, and 5 are selected, then Pcm = [po1∣po3∣po5] and Scm is a vector composed of scm,o1, scm,o3and scm,o5. By stopping the selection when the difference between the class weight and the linear combination does not decrease, we reduce the randomness and prevent too many orbits from being included in the explanation for each class. The detailed procedure can be found in Algorithm 2. UO-Explainer uses the orbit o
∗
cm with the highest class-orbit score from Equation 6 as the model-level explanation.

$$o_{c_{m}}^{*}=\arg\operatorname*{max}_{o_{k}\in\mathcal{O}}\left\{s_{c_{m},o_{0}},\cdots,s_{c_{m},o_{k}},\cdots,s_{c_{m},o_{72}}\right\}.$$
,⋯, scm,o72 }. (6)
The orbit with the highest class-orbit score is always accompanied by its corresponding graphlet. Therefore, the model-level explanation is provided in the form of graph patterns, with the given orbit as the target node and its corresponding graphlet.

## 3.3 Instance-Level Explanations

Instance-level explanations are provided by decomposing the prediction value of the target node into
orbit units. The class weight decomposition of Equation 3 extends to the decomposition of prediction
values as follows:
$$\mathcal{Z}_{v_{n},c_{m}}\approx\mathbf{h}_{v_{n}}\cdot\mathbf{w}_{c_{m}}+b_{c_{m}}$$ $$\approx\underbrace{s_{c_{m},o_{0}}\mathbf{h}_{v_{n}}\cdot\mathbf{p}_{o_{0}}}_{s_{v_{n},c_{m},o_{0}}}+\cdots+\underbrace{s_{c_{m},o_{k}}\mathbf{h}_{v_{n}}\cdot\mathbf{p}_{o_{k}}}_{s_{v_{n},c_{m},o_{k}}}+\cdots+\underbrace{s_{c_{m},o_{K}}\mathbf{h}_{v_{n}}\cdot\mathbf{p}_{o_{K}}}_{s_{v_{n},c_{m},o_{K}}}+b_{c_{m}}\,.$$  where $\mathbf{h}_{v_{n}}$ is the homogeneous orthogonal matrix and $\mathbf{h}_{v_{n}}$ is the vector valued vector 
$$(6)$$
$$\left(7\right)$$
+bcm. (7)
The equation above directly decomposes the prediction value of the target node into orbit units. Each term represents the magnitude of the decomposed prediction value, similarly indicating the contribution of orbits, akin to class weight decomposition. Therefore, we define the contribution of orbit ok to the class cm prediction of target node vn as node-class-orbit score, svn,cm,ok. To provide Input: A set of orbit bases {po0
, ⋯, pok
, ⋯, po72 }, a set of class weight vectors {wc1
, ⋯, wcm, ⋯, wc∣C∣
}.

Output: A set S of selected orbit's class-orbit scores s*cm,ok* Variables: A vector composed of an element as a selected orbit's class-orbit score Scm.

Initialize S as an empty set 1: for m=1 to ∣C∣ (the number of classes) do 2: Initialize lmin = ∞
3: Initialize Pcm as an empty matrix 4: **while do**
5: selected *orbit* = arg minok∈O minScm>0 ∣∣wcm − [Pcm∣pok
]Scm∣∣
6: l = ∣∣wcm − [Pcm∣pselected *orbit*]Scm∣∣
7: if l < lmin **then** 8: lmin = l 9: Pcm = [Pcm∣pselected *orbit*]
10: **else**
11: S.add(all s*cm,ok* in the Scm)
12: break 13: **end if** 14: **end while** 15: **end for** 16: **Return** S
instance-level explanations, UO-Explainer extracts the orbit o
∗
vn,cm with the highest node-class-score as follows:

$$({\mathfrak{s}})$$
$${\cal O}_{v_{n},c_{m}}^{*}=\arg\operatorname*{max}_{o_{k}\in\mathcal{O}}\left\{s_{v_{n},c_{m},o_{0}},\cdots,s_{v_{n},c_{m},o_{k}},\cdots,s_{v_{n},c_{m},o_{K}}\right\}.$$
,⋯, svn,cm,oK }. (8)
Unlike the model-level explanation, which provides the orbit with the highest contribution and its corresponding graphlet as an explanation, instance-level explanations must provide subgraphs around the target node within the input graph. To achieve this, we use the search algorithm based on the Breadth-First Search (BFS). These algorithms initiate the search from the target node and explore neighboring nodes by verifying whether their connectivity matches the highest contributed orbit's corresponding graphlets. A detailed algorithm is shown in Appendix B. Through this search, UO-Explainer can extract a subgraph within the input graph that matches the highest-contributing orbit for the target node, i.e., the explored subgraph is provided as an instance-level explanation along with the target node.

## 3.4 Time Complexity Analysis

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## Algorithm 2 Class-Orbit Score Learning

Training UO-Explainer consists of two main processes: orbit basis learning and class-orbit score learning, with the time complexity for each process detailed below. The time complexity of **orbit**
basis learning is O(∣O∣ ∣V∣), where training is conducted for each orbit basis across all nodes in the input graph, as outlined in Algorithm 1. Here, ∣O∣ represents the number of orbits used for explanation, and ∣V∣ denotes the total number of nodes in the input graph. The time complexity for **class-orbit score learning** depends on the number of orbits selected through greedy search for each class weight. In our experiments, the number of orbits selected did not exceed 5, rendering this complexity component negligible. Consequently, the overall time complexity for this phase is represented as O(∣C∣), following the procedure in Algorithm 2, where C indicates the number of classes. In short, the overall time complexity for training UO-Explainer is O(∣O∣ ∣V∣ + ∣C∣) ≈ O(∣V∣). The time complexities of the baseline methods are as follows: D4Explainer has a time complexity of O(∣V∣
3), GNNExplainer is O(∣V∣ ∣E∣) where ∣E∣ denotes the number of edges, PGExplainer and TAGE have a time complexity of O(∣E∣), and MotifExplainer operates with a complexity of O(∣V∣ ∣M∣), where ∣M∣ represents the number of motifs used in explanations. Therefore, in terms of time complexity, UO-Explainer is less demanding compared to D4Explainer and GNNExplainer. Alongside such analysis, our unified model is capable of providing both model-level and instancelevel explanations simultaneously, thus demonstrating its competitiveness in terms of time efficiency. A more detailed time complexity analysis and experiments are described in Appendix C.

## 4 Related Work 324

325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 GNN explanation methods can be categorized into model-level and instance-level, each with its own scope of explanation (Yuan et al., 2022). Model-level methods (Yuan et al., 2020; Shin et al., 2022; Azzolin et al., 2022) offer explanations that describe the general behavior underlying GNN predictions regarding to specific class. For example, XGNN (Yuan et al., 2020) aims to generate model-level patterns that maximize the predictive probability of a certain class by training a graph generator through reinforcement learning. Similarly, PAGE (Shin et al., 2022) employs graph representation vectors to iteratively search for human-interpretable prototype graphs. Moreover, GLGExplainer (Azzolin et al., 2022) provides general model-level patterns by aggregating instancelevel explanations into logical formulas, utilizing an Entropy-Logic Explainer (E-LEN) (Barbiero et al., 2022; Ciravegna et al., 2023). However, these studies primarily focus on graph classification tasks and do not directly apply to node classification tasks. To bridge this gap, D4Explainer (Chen et al., 2023) introduces a method for providing counterfactual model-level explanations for node classification tasks, alongside instance-level explanations, based on a diffusion model. This approach, however, results in a high training cost for the explainer incurred by iterative diffusion- and deonising steps. Moreover, none of the methods for model-level explanations fully consider the user-centric explanation unit, hindering human-interpretability. Instance-level methods (Yu & Gao, 2022; Ying et al., 2019; Luo et al., 2020; Xie et al., 2022; Vu & Thai, 2020; Wang et al., 2023; Xiong et al., 2023; Zhang et al., 2023; Ye et al., 2024; Lu et al., 2024) provide explanations in the form of subgraph to elucidate the prediction of a specific instance by unveiling relational structures with high contributions in the input graph. Notably, GNNExplainer (Ying et al., 2019) stands as an early instance-level explainer, optimizing edge and feature masks to maximize mutual information with GNN prediction results. PGExplainer (Luo et al., 2020) leverages node representation vectors and trains a parameterized mask predictor to optimize edge masks for explanation in inductive settings. TAGE (Xie et al., 2022) explains the GNN embedding models, allowing efficient explanations for multiple downstream tasks. All these methods entail learning edge masks to present masked graphs as instance-level explanation subgraphs. Despite connectivity constraints, it often fails to generate connected subgraphs, resulting in less intuitive explanations. On the other hand, MotifExplainer (Yu & Gao, 2022) provides an instance-level explanation that does not rely on edge masks. It computes embedding vectors for each motif and extracts explanations by restoring the GNN's prediction value using an attention network. However, it overlooks both the universal importance of specific motifs and the orbit-based isomorphism when extracting motifs. Furthermore, the abovementioned methods are limited to providing explanations in the unified framework capable of simultaneously providing both model-level and instance-level explanations under the human-interpretable units of interest.

## 5 Experiment

We evaluate explanations provided by UO-Explainer at the model-level and instance-level on synthetic and real-world datasets. These extensive experiments include quantitative and qualitative analysis of UO-Explainer's performance compared to recent baselines. For detailed experimental settings, please refer to Appendix D.

## 5.1 Datasets And Baselines

We conducted experiments using five synthetic datasets and three real-world data sets. Synthetic datasets such as Random Graph (Holme & Kim, 2002), BA-Shapes (Ying et al., 2019), BA- Community, Tree-Cycle, and Tree-Grid are used to evaluate GNN explanation methods to compare the generated explanation based on pre-defined ground truths of each dataset. Also, real-world datasets such as Protein-Protein Interaction (PPI) (Zitnik & Leskovec, 2017), LastFM-Asia (Rozemberczki & Sarkar, 2020), and Gene (Zitnik & Leskovec, 2017) are used for node classification tasks. The detailed information and statistics of each dataset are described in Appendix F. Among the existing methods, D4Explainer (Chen et al., 2023) and GLGExplainer (Azzolin et al., 2022) are the only existing framework that provides model-level explanations for the node classification task. For instance-level explanations, we set baselines based on common methods that learn edge masks, such as GNNExplainer (Ying et al., 2019), PGExplainer (Luo et al., 2020), TAGE
378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Table 2: Model-level explanation results on synthetic datasets. The evaluation metric is the Sub-recall. (Xie et al., 2022), MixupExplainer (Zhang et al., 2023), SAME (Ye et al., 2024), and EIG (Lu et al., 2024). Additionally, MotifExplainer (Yu & Gao, 2022), which provides explanations based on motifs similar to our model, was also set as a baseline.

## 5.2 Evaluation Metric

We evaluate the quality of explanations using Sparsity, Fidelity, Edge-recall, and Sub-recall as evaluation metrics. *Sparsity* (Li et al., 2022) refers to the ratio of edges in the explanation compared to the total number of edges in the computation graph of the target node. High sparsity implies that the proposed explanation has a small number of edges. *Fidelity* (Li et al., 2022) measures the difference in the probability values when the explanation is excluded from the computation graph based on the target node. *Edge-recall* indicates how many edges in the explanations match the edges in the ground truths. *Sub-Recall* indicates the proportion of correct answers that the entire presented explanations match with ground truths. Notably, equations of *Sparsity* and *Fidelity* are described in the Appendix D.4. Additionally, we conducted experiments five times and then reported the average and standard deviations in Appendix E.

## 5.3 Results: Model-Level Explanations

We first validate whether the UO-Explainer can identify the correct orbits for model-level explanations. We pre-train 2 or 3-layer GCN models (Kipf & Welling, 2017) and 2 or 3-layer GIN models (Xu et al., 2019) on Random Graph datasets. We pre-train 2 or 3-layer GCN models (Kipf & Welling, 2017) and 2 or 3-layer GIN models (Xu et al., 2019) on Random Graph datasets. Each task is to classify whether the node belongs to the orbit corresponding task number. Consequently, explanation methods are expected to provide the ground-truth pattern in the form of the orbit (target node) with its corresponding graphlet (pattern) for each task. Tasks with accuracy below 0.8 are excluded as they are unlikely to yield accurate explanations. We use the *Sub-recall* metric to evaluate whether the provided explanation matches the ground-truth pattern.

In Table 1, UO-Explainer shows superior performance compared to other baselines. In particular, the UO-Explainer constantly provides model-level explanations matching to ground truths for all tasks except 33, 60, and 62 in the GCN model as shown in (a). This limitation may arise from the GNN's expressiveness, failing to learn intended orbits during the pre-training. To address this, we conducted experiments in the same manner using GIN, known for better expressiveness. The results are shown in (b) of Table 1 that UO-Explainer accurately provides the explanations matching the Table 1: Model-level explanation on random graph datasets: (a) with 2 or 3-layer GCN, and (b) with 2 or 3-layer GIN. Each task is to classify whether the node belongs to the orbit corresponding task number. The evaluation metric is the *Sub-recall*. The best performances are shown in **bold**.

(a)
Task number 8 11 16 21 27 31 32 33 35 39 45 47 49 57 59 60 61 62 64 Ground-truth orbit o8 o11 o16 o21 o27 o31 o32 o33 o35 o39 o45 o47 o49 o57 o59 o60 o61 o62 o64 D4Explainer 0.2 0.8 0.4 0.4 0.2 0.4 0.4 0.4 1.0 0.8 0.2 0.4 0.4 0.8 0.2 0.0 0.6 0.6 0.2 GLGExplainer 0.8 0.6 0.8 0.6 0.8 1.0 1.0 0.8 0.8 0.8 0.8 0.6 0.6 1.0 0.6 0.6 0.8 0.0 1.0 UO-Explainer 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 1.0 0.0 1.0
(b)
Task number 8 11 16 21 27 31 32 33 35 39 45 47 49 57 59 60 61 62 64 Ground-truth orbit o8 o11 o16 o21 o27 o31 o32 o33 o35 o39 o62 o47 o49 o57 o59 o60 o61 o45 o64 D4Explainer 0.8 0.6 0.6 1.0 0.6 0.8 0.8 0.6 1.0 1.0 0.6 0.4 0.8 1.0 0.8 0.8 0.2 0.4 0.6 GLGExplainer 0.8 0.8 1.0 0.4 1.0 1.0 1.0 1.0 0.8 1.0 1.0 0.6 0.8 1.0 0.6 0.8 1.0 0.0 1.0 UO-Explainer 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 8

| BA-Shapes          | BA-Community   | Tree-Grid   | Tree-Cycle   |        |        |        |        |        |        |        |        |        |     |
|--------------------|----------------|-------------|--------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----|
| class1             | class2         | class3      | class1       | class2 | class3 | class5 | class6 | class7 | class1 | class2 | class3 | class1 |     |
| Ground-truth Orbit | o58            | o57         | o56          | o58    | o57    | o56    | o58    | o57    | o56    | o73    | o74    | o75    | o76 |
| D4Explainer        | 0.4            | 0.6         | 0.4          | 0.8    | 0.2    | 0.8    | 0.0    | 0.6    | 0.4    | 0.6    | 0.0    | 0.2    | 0.8 |
| GLGExplainer       | 1.0            | 1.0         | 1.0          | 1.0    | 0.8    | 0.2    | 0.8    | 1.0    | 0.8    | 0.0    | 0.8    | 0.2    | 1.0 |
| UO-Explainer       | 1.0            | 1.0         | 1.0          | 1.0    | 1.0    | 1.0    | 0.8    | 1.0    | 1.0    | 0.8    | 1.0    | 1.0    | 1.0 |

BA-Shapes BA-Community Tree-Grid Tree-Cycle

Sub-recall Edge-recall Fidelity Sub-recall Edge-recall Fidelity Sub-recall Edge-recall Fidelity Sub-recall Edge-recall Fidelity

GNNExplainer 0.004 0.616 0.580 0.006 0.491 0.653 0.000 0.629 0.872 0.119 0.699 0.724

PGExplainer 0.760 0.915 0.574 0.238 0.667 0.652 0.000 0.647 0.876 0.926 0.992 0.732

TAGE 0.682 0.900 0.601 0.352 0.754 0.672 0.003 0.693 0.874 0.963 0.994 0.734 

MixupExplainer 0.696 0.906 0.612 0.496 0.857 0.693 0.047 0.712 0.8770 0.930 0.994 0.734

SAME 0.343 0.720 0.547 0.132 0.680 0.642 0.000 0.238 0.846 0.101 0.635 0.692

EiG-Search 0.878 0.520 0.605 0.078 0.681 0.695 0.004 0.723 0.885 0.083 0.812 0.703 

MotifExplainer 0.873 0.890 0.548 0.423 0.714 0.683 0.793 0.857 0.879 0.991 0.993 0.736 UO-Explainer 0.948 0.984 0.623 0.921 0.970 0.716 0.859 0.900 0.888 1.000 1.000 0.737

Table 4: Instance-level explanation results on real datasets. The best fidelity on each dataset is shown

in **bold**.∗ notation indicates the lower sparsity setting.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Table 3: Instance-level explanation results on synthetic datasets. The best performances on each dataset are shown in **bold**. correct ground-truth for all tasks. These findings confirm that UO-Explainer is capable of detecting various orbits and providing correct explanations while UO-Explainer's explanations mirror the expressiveness of the GNN's embedding model. Notably, comparing the performance of original GNNs to the decomposed class weights model, the performance degradation is less than 5%. On the other hand, D4Explainer and GLGExplainer show slightly improved or even decreased performance in the GIN setting and fail to provide consistent explanations matched with the ground truth.

As observed in Table 2, UO-Explainer also outperforms other baselines on the BA-Shapes and BA-Community dataset. Using 3-layer GCNs for this experiment, UO-Explainer successfully provides accurate model-level explanations for each class on both datasets. Specifically on the BA-Shapes dataset, we observe the explanations that the house-like motif plays a crucial role in node classification by detecting the orbit such as o56, o57, and o58 in the graphlet g23 as explanations. This explanation sheds light on the overall behavior of the GNN beyond individual nodes, enabling a broader interpretation of GNNs. Moreover, our approach provides orbit-corresponding graphlets, allowing us to determine the topological positions of nodes within the motif for each class. Thus, UO-Explainer shows that the GNN recognizes the house-like motif as an important pattern for node classification, assigning nodes on the roof, floor, and top of the roof of the house-like motif to classes 1, 2, and 3, respectively. On the BA-Community dataset, UO-Explainer also finds the ground-truth pattern as the model-level explanation by detecting the orbit such as o56, o57, and o58 in the graphlet g23 as explanations, since the dataset is a union of two BA-SHAPES graphs. In conclusion, the experimental result demonstrates that orbits as pre-defined explanation units of UO-Explainer serve crucial patterns of the specific classes for prediction, showing accurate and consistent explanations.

| PPI                        | LastFM Asia       |                   |                   |                   |                   |          |       |       |       |       |       |       |       |       |
|----------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|-------|-------|-------|-------|-------|-------|-------|-------|
| Task0                      | Task1             | Task2             | Task3             | Task4             | Task5             |          |       |       |       |       |       |       |       |       |
| Fidelity Sparsity Fidelity | Sparsity Fidelity | Sparsity Fidelity | Sparsity Fidelity | Sparsity Fidelity | Sparsity Fidelity | Sparsity |       |       |       |       |       |       |       |       |
| GNNExplainer               | 0.100             | 0.973             | 0.325             | 0.999             | 0.417             | 0.999    | 0.480 | 0.999 | 0.250 | 0.999 | 0.331 | 0.999 | 0.114 | 0.974 |
| PGExplainer*               | 0.023             | 0.973             | 0.155             | 0.999             | 0.223             | 0.999    | 0.101 | 0.999 | 0.200 | 0.999 | 0.005 | 0.999 | 0.011 | 0.974 |
| TAGE*                      | 0.031             | 0.973             | 0.109             | 0.999             | 0.233             | 0.999    | 0.138 | 0.999 | 0.214 | 0.999 | 0.101 | 0.999 | 0.086 | 0.974 |
| MixupExplainer             | 0.005             | 0.973             | 0.002             | 0.999             | 0.257             | 0.999    | 0.246 | 0.999 | 0.246 | 0.999 | 0.129 | 0.999 | 0.100 | 0.974 |
| EiG-Search                 | 0.180             | 0.973             | 0.269             | 0.999             | 0.180             | 0.999    | 0.379 | 0.999 | 0.100 | 0.999 | 0.180 | 0.999 | 0.095 | 0.974 |
| SAME                       | 0.022             | 0.973             | 0.189             | 0.999             | 0.194             | 0.999    | 0.189 | 0.999 | 0.034 | 0.999 | 0.129 | 0.999 | 0.049 | 0.974 |
| MotifExplainer             | 0.070             | 0.992             | 0.074             | 0.999             | 0.012             | 0.999    | 0.129 | 0.999 | 0.097 | 0.999 | 0.050 | 0.999 | 0.085 | 0.994 |
| UO-Explainer               | 0.423             | 0.999             | 0.358             | 0.999             | 0.425             | 0.999    | 0.510 | 0.999 | 0.623 | 0.999 | 0.413 | 0.999 | 0.115 | 0.993 |

## 5.4 Results: Instance-Level Explanations

Except for MotifExplainer, all baselines provide explanations in the form of subgraphs obtained by extracting edges exceeding specific threshold values or ranking the top-k edge considered important. For fair experiments, explanations composed of top-k edges were extracted considering a sparsity level similar to that of UO-Explainer. In the case of MotifExplainer, we extract one motif as an instance-level explanation. UO-Explainer used a subgraph within the input graph that matches the highest-contributing orbit for the target node for the instance-level explanation as mentioned in Section 3.3.

The experimental results on synthetic datasets are shown in Table 3. UO-Explainer outperforms the baseline methods across all evaluation metrics while maintaining a comparable sparsity level.

Notably, UO-Explainer achieves higher sub-recall, indicating accurate detection of ground-truth subgraphs as explanations. In cases such as Tree-grid and Tree-cycle, where grid- and cycle-shaped graphlets do not exist within the 2- to 5-node graphlets, we employ grid and cycle graphlets along with their corresponding orbits and include them as units of explanation. We note that by defining custom graphlets based on background knowledge or extracted rules tailored to specific problem

TAGE MotifExplainer PGExplainer Methods UO-Explainer SMAD7 SMAD7 SMAD7 SMAD7 SMAD7 SMAD7 SMAD7 SMAD7 ACVR2A DAXX
INHBA
ACVR1B
DAXX **ACVR2A**
INHBA
ACVR2A
INHBA
ACVR1B
ACVR1B
ACVR2A
INHBA
ACVR1B
Explanation DAXX
ENG
ACVR1 ENG
ACVR1 ENG
ACVR1 DAXX
ENG
ACVR1 TGFBR2 TGFBR2 TGFBR2 TGFBR2 TRGFB1 CCN2 TRGFB1 CCN2 TRGFB1 CCN2 TRGFB1 CCN2 Fidelity 0.221 0.101 0.228
. 
settings, the proposed method can be extended beyond the pre-defined 2- to 5-node graphlets and orbits. Table 4 shows the performance on real datasets, where UO-Explainer outperforms other baselines in the fidelity metric, except for task 0 of the PPI dataset. MotifExplainer struggles due to extracting too many motifs, leading to less meaningful explanations. PGExplainer and TAGE also show poor performance, often identifying the same or irrelevant edges across nodes, regardless of experimental changes. GNNExplainer, trained iteratively for node-specific explanations, extracts more relevant edges, especially on larger datasets. The ∗ notation in Table 4 highlights that lower sparsity is needed for competing methods to match UO-Explainer, yet they still include noise edges or less important subgraphs. In contrast, UO-Explainer performs consistently well even under high sparsity, using just one orbit-based subgraph for explanations, demonstrating its ability to provide high-quality instance-level explanations.

## 5.5 Case Study On Gene Dataset

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 We introduce UO-Explainer, a human-interpretable explanation method that leverages pre-defined units, as requested by users, in a unified framework for node classification models. By utilizing orbits as explanatory units, UO-Explainer decomposes model weights into orbit components, which serve as essential, human-interpretable units within graph domains. Experimental results on both synthetic and real-world datasets demonstrate the effectiveness of UO-Explainer, outperforming baseline methods and delivering higher-quality explanations. UO-Explainer is particularly valuable in scientific applications, such as drug development and education, where domain knowledge is critical. By using pre-defined explanation units, users can uncover meaningful patterns and gain deeper insights through the explanation method. As the qualitative analysis, we visualized the explanatory subgraphs described in our method and the baselines on the gene dataset. The visualization results are presented in Figure 4. The experiment results demonstrate that PGExplainer and TAGE provide scattered subgraphs with discontinuous edges while the sparsity remains at 0.900 for fair comparison. In contrast, UO-Explainer offers a connected subgraph that is more intuitive while maintaining relatively high fidelity. MotifExplainer also presents a connected subgraph as an explanation but with relatively lower fidelity. Additionally, several studies provide evidence that the genes (TGFBR2 (Massague & Gomis, 2006), ENG (Breen ´ et al., 2013), INHBA, and ACVR2B (Attisano & Wrana, 2013)) identified by UO-Explainer in the explanations have an impact on the surface receptor signaling pathway, which is the label of the dataset. For example, in (Bottino et al., 2021), it was mentioned that TGFBR2 is one of the TGF-beta receptors that transmit signals within natural killer cells, exerting a significant influence on cell development and the function of natural killer cells. These results imply that UO-Explainer provides human-interpretable explanations compared to other baselines regarding the perspective of interest.

## 6 Discussion And Conclusion References

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Nesreen K Ahmed, Jennifer Neville, Ryan A Rossi, and Nick Duffield. Efficient graphlet counting for large networks. In *Proceeding of the International Conference on Data Mining*, pp. 1–10. IEEE, 2015.

Liliana Attisano and Jeffrey L Wrana. Signal integration in tgf-β, wnt, and hippo pathways.

F1000prime reports, 5, 2013.

Steve Azzolin, Antonio Longa, Pietro Barbiero, Pietro Lio, and Andrea Passerini. Global explain- `
ability of gnns via logic combination of learned concepts. *arXiv preprint arXiv:2210.07147*, 2022.

Federico Baldassarre and Hossein Azizpour. Explainability techniques for graph convolutional networks, 2019. URL https://arxiv.org/abs/1905.13686.

Pietro Barbiero, Gabriele Ciravegna, Francesco Giannini, Pietro Lio, Marco Gori, and Stefano ´
Melacci. Entropy-based logic explanations of neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 6046–6054, 2022.

Cristina Bottino, Thierry Walzer, Angela Santoni, and Roberta Castriconi. Tgf-β as a key regulator of nk and ilcs development and functions, 2021.

Michael J Breen, Diarmuid M Moran, Wenzhe Liu, Xiaoke Huang, Calvin PH Vary, and Raymond C
Bergan. Endoglin-mediated suppression of prostate cancer invasion is regulated by activin and bone morphogenetic protein type ii receptors. *PLoS One*, 8(8):e72407, 2013.

Jialin Chen, Shirley Wu, Abhijit Gupta, and Rex Ying. D4explainer: In-distribution gnn explanations via discrete denoising diffusion. *Advances in neural information processing systems*, 2023.

Xiaowei Chen and John CS Lui. Mining graphlet counts in online social networks. ACM Transactions on Knowledge Discovery from Data, 12(4):1–38, 2018.

Gabriele Ciravegna, Pietro Barbiero, Francesco Giannini, Marco Gori, Pietro Lio, Marco Maggini, ´
and Stefano Melacci. Logic explained networks. *Artificial Intelligence*, 314:103822, 2023.

David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alan´
Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural information processing systems*, 28, 2015.

Rafael Espejo, Guillermo Mestre, Fernando Postigo, Sara Lumbreras, Andres Ramos, Tao Huang, and Ettore Bompard. Exploiting graphlet decomposition to explain the structure of complex networks: the ghust framework. *Scientific reports*, 10(1):12884, 2020.

Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. Graph neural networks for social recommendation. In *Proceedings of the Web Conference*, 2019.

Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.

Thomas Gaudelet, Ben Day, Arian R Jamasb, Jyothish Soman, Cristian Regep, Gertrude Liu, Jeremy BR Hayter, Richard Vickers, Charles Roberts, Jian Tang, et al. Utilizing graph machine learning within drug discovery and development. *Briefings in bioinformatics*, 22(6):bbab159, 2021.

Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs.

Advances in neural information processing systems, 30, 2017.

Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d'Amato, Gerard de Melo, Claudio Gutierrez, Sabrina Kirrane, Jose Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, et al. Knowledge ´ graphs. *ACM Computing Surveys (CSUR)*, 54(4):1–37, 2021.

Petter Holme and Beom Jun Kim. Growing scale-free networks with tunable clustering. Physical review E, 65(2):026107, 2002.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks.

In *Proceedings of the International Conference on Learning Representations*, 2017.

Risi Kondor, Nino Shervashidze, and Karsten M. Borgwardt. The graphlet spectrum. In Proceedings of the International Conference on Machine Learning, 2009.

Yiqiao Li, Jianlong Zhou, Sunny Verma, and Fang Chen. A survey of explainable graph neural networks: Taxonomy and evaluation metrics. *arXiv preprint arXiv:2207.12599*, 2022.

Meng Liu, Youzhi Luo, Limei Wang, Yaochen Xie, Hao Yuan, Shurui Gui, Haiyang Yu, Zhao Xu, Jingtun Zhang, Yi Liu, et al. Dig: A turnkey library for diving into graph deep learning research.

The Journal of Machine Learning Research, 22(1):10873–10881, 2021.

Shengyao Lu, Bang Liu, Keith G. Mills, Jiao He, and Di Niu. Eig-search: Generating edge-induced subgraphs for gnn explanation in linear time. In *International Conference on Machine Learning*, 2024.

Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang Zhang.

Parameterized explainer for graph neural network. Advances in neural information processing systems, 2020.

Joan Massague and Roger R Gomis. The logic of tgf ´ β signaling. *FEBS letters*, 580(12):2811–2820, 2006.

Hiromi Nakagawa, Yusuke Iwasawa, and Yutaka Matsuo. Graph-based knowledge tracing: modeling student proficiency using graph neural network. In Proceeding of the IEEE/WIC/ACM International Conference on Web Intelligence, 2019.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Phillip E Pope, Soheil Kolouri, Mohammad Rostami, Charles E Martin, and Heiko Hoffmann.

Explainability methods for graph convolutional neural networks. In *Proceedings of the IEEE/CVF* conference on computer vision and pattern recognition, pp. 10772–10781, 2019.

Mario Alfonso Prado-Romero, Bardh Prenkaj, Giovanni Stilo, and Fosca Giannotti. A survey on graph counterfactual explanations: Definitions, methods, evaluation, and research challenges. ACM
Computing Surveys, 56(7):1–37, April 2024. ISSN 1557-7341. doi: 10.1145/3618105. URL http://dx.doi.org/10.1145/3618105.

N. Przulj, D. G. Corneil, and I. Jurisica. Modeling interactome: scale-free or geometric? ˇ Bioinformatics, 20(18):3508–3515, 2004.

Natasa Pr ˇ zulj. Biological network comparison using graphlet degree distribution. ˇ *Bioinformatics*, 23
(2):e177–e183, 2007.

Benedek Rozemberczki and Rik Sarkar. Characteristic functions on graphs: Birds of a feather, from statistical descriptors to parametric models. In *Proceedings of the International Conference on* Information Knowledge Management, 2020.

Michael Sejr Schlichtkrull, Nicola De Cao, and Ivan Titov. Interpreting graph neural networks for nlp with differentiable edge masking, 2022. URL https://arxiv.org/abs/2010.00577.

Nino Shervashidze, SVN Vishwanathan, Tobias Petri, Kurt Mehlhorn, and Karsten Borgwardt.

Efficient graphlet kernels for large graph comparison. In *Artificial intelligence and statistics*, pp. 488–495. PMLR, 2009.

Yong-Min Shin, Sun-Woo Kim, Eun-Bi Yoon, and Won-Yong Shin. Prototype-based explanations for graph neural networks (student abstract). In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

Aravind Subramanian, Pablo Tamayo, Vamsi K. Mootha, Sayan Mukherjee, Benjamin L. Ebert, Michael A. Gillette, Amanda Paulovich, Scott L. Pomeroy, Todd R. Golub, Eric S. Lander, and Jill P. Mesirov. Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles. *Proceedings of the National Academy of Sciences*, 102(43):
15545–15550, 2005.

Petar Velickovi ˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua ´
Bengio. Graph attention networks. In Proceedings of the International Conference on Learning Representations, 2018.

Minh Vu and My T Thai. Pgm-explainer: Probabilistic graphical model explanations for graph neural networks. *Advances in neural information processing systems*, 2020.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 Jihong Wang, Minnan Luo, Jundong Li, Yun Lin, Yushun Dong, Jin Song Dong, and Qinghua Zheng.

Empower post-hoc graph explanations with information bottleneck: A pre-training and fine-tuning perspective. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and* Data Mining, pp. 2349–2360, 2023.

Yaochen Xie, Sumeet Katariya, Xianfeng Tang, Edward Huang, Nikhil Rao, Karthik Subbian, and Shuiwang Ji. Task-agnostic graph explanations. Advances in neural information processing systems, 2022.

Ping Xiong, Thomas Schnake, Michael Gastegger, Gregoire Montavon, Klaus Robert Muller, and ´
Shinichi Nakajima. Relevant walk search for explaining graph neural networks. In International Conference on Machine Learning, pp. 38301–38324, 2023.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *Proceedings of the International Conference on Learning Representations*, 2019.

Ziyuan Ye, Rihan Huang, Qilin Wu, and Quanying Liu. Same: Uncovering gnn black box with structure-aware shapley-based multipiece explanations. Advances in Neural Information Processing Systems, 36, 2024.

Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer:
Generating explanations for graph neural networks. Advances in neural information processing systems, 32, 2019.

Zhaoning Yu and Hongyang Gao. Motifexplainer: a motif-based graph neural network explainer.

arXiv preprint arXiv:2202.00519, 2022.

Hao Yuan, Jiliang Tang, Xia Hu, and Shuiwang Ji. Xgnn: Towards model-level explanations of graph neural networks. In Proceedings of the SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 430–438, 2020.

Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in graph neural networks: A
taxonomic survey, 2022. URL https://arxiv.org/abs/2012.15445.

Jiaxing Zhang, Dongsheng Luo, and Hua Wei. Mixupexplainer: Generalizing explanations for graph neural networks with data augmentation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 3286–3296, 2023.

Marinka Zitnik and Jure Leskovec. Predicting multicellular function through multi-layer tissue networks. *Bioinformatics*, 33(14):i190–i198, 2017.

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 We present a detailed line-by-line explanation of Algorithm 3 as an exemplary demonstration of the orbit search process. This algorithm is designed to find g5 graphlets in which the target node belongs to orbit o10. In line 1, we generate the neighbor set Nvnfrom the target node vn. Since the number of neighboring nodes within orbit o10 in graphlet g5 is three, line 2 constructs the set

2-node	graphlet 3-node	graphlet 4-node	graphlet
"
! #
$()
&

'!!

%
*
!" !#
!$ !&
5-node	graphlet
" ! # $ & ' ( ) %
#& #( #'

!%
$#
$!

$$
$&
$'
$(
&# &'
!"
&$
&&
#!

!%
!#
$%
$*
!)
!* #"
!'
!(
#)
#%
#*
&"
&!

##
#$
$)
!$
$"
* !" !! !# !$ !%
!& !'
!( !*
!)
'*
("
(!

(#
($
(&
('
(%
(*
)"
)!

)#
')
'%
'(
'"
'!

'#
()
'$
'&
''
((
#" #& #' #( #) #% #*
#!

##
#$
Figure 5: Entire graphlets having 2-5 nodes and orbits. The same color nodes within each graphlet belong to the same orbit. In this work, we have defined the explanatory units as 0-72 orbits and their corresponding graphlet ranging from 0 to 29 consisting of 2 to 5 nodes. The full set of utilized graphlets and orbits can be observed in Figure 5 (Przulj et al., 2004; Pr ˇ zulj, 2007). ˇ

## B Orbit Search Algorithm

In Section 3.1, we have mentioned the implementation of a graphlet search algorithm based on the Breadth-First Search (BFS) algorithm for extracting the orbit with the highest score and its corresponding graphlet on the target node. These algorithms initiate the search from the target node and explore neighboring nodes by verifying whether their connectivity matches the desired graphlets. We present two algorithmic examples for identifying specific orbits in our study. Algorithm 3 describes the process of identifying the o10 with its corresponding graphlet g5, while Algorithm 4 outlines the steps for identifying the o56 with its corresponding graphlet g23. In both algorithms, N(v) denotes the neighbor function to construct the set of neighboring nodes around a given node v.

Additionally, C(⋅, n) denotes a combination function to generate the set of combinations consisting of n elements from a given set ⋅.

## Algorithm 3 Orbit O10 Search Algorithm

Input: A graph G = (V, E) where V = {v1, ⋯, vi, ⋯, v∣V∣} denotes a set of node and E = {(vi, vj ), ⋯, (vk, vl)} denotes a set of edges, Target node vn.

Output: A set L which contains edge sets of g5 graphlets that make target node belong to orbit o10.

Initialize L as an empty set 1: Neighbor set of the target node Nvn = N(vn) where N denotes the neighbor function. 2: A combination set with three elements C1 = C(Nvn , 3) where C denotes Combination function 3: for v
′
a
, v
′ b
, v
′
c in C1 do 4: a set of candidate combinations C
′
1 = {(v
′ a
, v
′
b
, v
′ c
), (v
′ b
, v
′ a
, v
′ c
), (v
′ c
, v
′ a
, v
′b)}
5: for va, vb, vc in C
′ 1 do 6: if (vb, vc) ∈ E and {(va, vb), (va, vc)} /⊂ E **then** 7: *L.add*({(vn, va), (vn, vb), (vn, vc), (vb, vc)}) 8: **end if** 9: **end for**
10: **end for** 11: **Return** L

## A The Full Set Of Graphlets And Orbits

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 Algorithm 4 Orbit o56 Search Algorithm Input: A graph G = (V, E) where V = {v1, ⋯, vi, ⋯, v∣V∣} denotes a set of nodes and E = {(vi, vj ), ⋯, (vk, vl)} denotes a set of edges, Target node vn.

Output: A set L which contains edge sets of g23 graphlets that make target node belong to orbit o56.

Initialize L as an empty set 1: Neighbor set of the target node Nvn = N(vn) where N denotes the neighbor function. 2: A combination set with three elements C1 = C(Nvn , 2) where C denotes Combination function 3: for va, vb in C1 do 4: if vb, vc ∈ E **then**
5: Neighbor set of va, Nva = N(va)
6: Neighbor set of vb, Nvb
= N(vb)
7: for vc in Nva do 8: for vd in Nvb do 9: if (vc, vd) ∈ E and {(vn, vc), (vn, vd), (vb, vc), (va, vd)} /⊂ E **then**
10: *L.add*({(vn, va), (vn, vb), (va, vb), (va, vc), (vb, vd), (vc, vd)}) 11: **end if** 12: **end for** 13: **end for** 14: **end if** 15: **end for**
16: **Return** L

## C Time Complexity Analysis

When UO-Explainer provides instance-level explanations, there are four time-consuming processes:
1) Pre-processing for finding the existence of the orbit, 2) Orbit basis learning, 3) Class-orbit score learning (Model-level explanation generation), and 4) Generating instance-level explanations.

1) Pre-processing for finding the existence of each orbit. As mentioned in Section 3.1, to learn the orbit basis, we must find the existence of each orbit for every node, as represented by the equation 1. This pre-processing can be conducted to search 2-5 node graphlets for each node. The time complexity of finding each graphlet that includes specific orbits, as can be inferred from Algorithm 3 and Algorithm 4, is O(d k−1)
1. Here, d represents the maximum node degree of the graph data, and k denotes the number of nodes included in each graphlet and is less than 5. Therefore, the time complexity of pre-processing for finding all 2-5 node graphlets that include a total 0-72 orbit for an entire node of graph data is O(∣O∣ ∣V∣ d k−1) where ∣O∣ denotes the number of edges and ∣V∣ represents the number of nodes.

2) Orbit basis learning. The time complexity of orbit basis learning is O(∣O∣ ∣V∣) since training is performed for each orbit basis over all nodes as shown in Algorithm 1. 3) Class-orbit score learning. Class-orbit score is trained based on the number of orbits selected by the greedy search for each class weight. In our experiments, the number of selected orbits was less than or equal to 5, so this can be sufficiently neglected. Therefore, the time complexity of class-orbit score learning considers only the number or class C; O(∣C∣) as shown in Algorithm 2. The training time required for UO-Explainer and the baselines can be found in Table 7. Model-level explanations 1As referenced from N. K. Ahmed, J. Neville, R. A. Rossi, and N. Duffield. Efficient Graphlet Counting for Large Networks. *In Proceedings of the International Conference on Data Mining*, 2015, pp. 1-10.

C1 by forming combinations of three nodes from the neighbor set. Within graphlet g5, there is one orbit, o8, that distinguishes it from the two neighboring nodes, o9. Therefore, to ensure that each node combination in C1 also distinguishes between o10 and o9, line 4 generates the set C
′
1, which reflects requirement. Notably, the first nodes of each tuple in the set are distinct from the remaining nodes. By observing the connectivity of the nodes in lines 5-6, if it matches the connectivity of the orbits within g5, the edge set of g5 can be obtained in line 7. It is added to the set L. This algorithm iterates by considering all possible combinations of neighboring nodes surrounding the target node vn. Therefore, all existing graphlet g5 that make the target node belong to orbit o10 are in the output set L. This format can be applied to find entire 2-5 node graphlets. However, it is computationally expensive. To alleviate this computational cost, by introducing randomness during the generation of the neighbor set (line 1) or combination set in the algorithm(line 2 and line 4), it becomes possible to sample the desired number of graphlets. The performance as the number of samples is discussed in Table 7.

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 Table 5: The training time required for the baselines and UO-Explainer for explanation in the gene dataset. The training time of the explainer is significantly influenced by the epochs. Therefore, we set the epochs for each method through a hyper-parameter search to achieve the highest fidelity.

| UO-Explainer     | GNNExplainer                                               | MotifExplainer   | PGExplainer   | TAGE   |     |
|------------------|------------------------------------------------------------|------------------|---------------|--------|-----|
| Training Time(s) | 268 (orbit basis learning)+21 (class-orbit score learning) | 1, 681           | 6, 074        | 98     | 204 |

Table 6: The training time required for the baselines and UO-Explainer for explanation in the gene dataset. The training time of the explainer is significantly influenced by the epochs. Therefore, we set the epochs for each method through a hyper-parameter search to achieve the highest fidelity.

| Instance and Model-level                                                    | Instance-level   | Model-level                                 |    |     |     |
|-----------------------------------------------------------------------------|------------------|---------------------------------------------|----|-----|-----|
| UO-Explainer                                                                | GNNExplainer     | MotifExplainer PGExplainer TAGE D4Explainer |    |     |     |
| Training Time(s) 268 (orbit basis learning)+21 (class-orbit score learning) | 1, 681           | 6, 074                                      | 98 | 204 | 502 |

can be provided immediately after class-orbit score learning. Therefore, the training time of the UO-Explainer listed in Table 6 equals the time required to provide model-level explanations.

4) Generating instance-level explanations. To provide instance-level explanations, it is necessary to search graphlets that include the given orbits as explanations for each node and select just one graphlet having the highest fidelity. Therefore, the time complexity is O(∣V∣ d k−1) since graphlets need to be directly found for each node. However, instead of finding all graphlets around each node, we sample a predetermined number of graphlets. Thus, the actual computational cost can be significantly reduced compared to O*(∣V∣* d k−1), while preserving high explanation performance. The required time and fidelity based on the number of sampled graphlets are shown in Table 7. Even with a time difference of more than 7 times, as observed when comparing the cases of not sampling and sampling 70 graphlets for explanation extraction, it can be observed that the fidelity values do not significantly decrease.

## D Detailed Experimental Settings

In this section, we cover the details of experiments that were not discussed in the main text, such as the structure of the pre-trained GNN and the hyper-parameter settings of the UO-Explainer.

## D.1 Details Of Pre-Trained Gnns

We provide detailed information about pre-trained GNNs for datasets other than the random graph, as already described in Section 5.3. For each dataset, we utilized a pre-trained GNN architecture consisting of a 3-layer GCN as the embedding model and a 1-layer MLP as the downstream model. The hyper-parameters and split ratio are shown in Table 8.

## D.2 Hyper-Parameter Settings For Uo-Explainer

UO-Explainer has a total of five hyper-parameters including batch size, epochs, learning rate (LR) for orbit basis learning, and additional epochs, learning rate (LR) for class-orbit score learning. Hyper-parameters used in the experiments are reported in Table 9.

## D.3 Details Of Baselines

In our code implementation, we primarily utilized the PyTorch (Paszke et al., 2019) and PyTorch Geometric (Fey & Lenssen, 2019) frameworks. Additionally, for the GNNExplainer and PGExplainer, we employed implementations from the PyTorch Geometric framework. For the TAGE, we employed implementations from the Dive into Graph (DIG) (Liu et al., 2021) framework. For MotfiExplainer, we utilized the code provided in the supplementary material available on OpenReview.net, which was suitably modified following advice received in correspondence with the authors. This revised implementation is incorporated within our publicly accessible code. Furthermore, to ensure the equity of our experiments, we diligently endeavored to explore the hyper-parameter space of the baselines as extensively as possible. The explored hyper-parameter 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 Table 7: The time required and fidelity based on the number of sampled graphlets of UO-Explainerin the gene dataset. We conducted five experiments for each method and reported the mean and standard deviation values. The "Entire" refers to the extraction of all graphlets surrounding the target node without sampling.

| # of sampled graphlets   | 1                                                                                                                 | 10            | 30            | 50            | 70            | 100           | Entire     |
|--------------------------|-------------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|---------------|---------------|------------|
| Time(s)                  | 14.572 ± 0.208 136.342 ± 2.305 379.342 ± 22.438 597.283 ± 3.865 838.991 ± 3.587 1183.459 ± 5.913 6321.783 ± 8.462 |               |               |               |               |               |            |
| Fildelity                | 0.157 ± 0.018                                                                                                     | 0.256 ± 0.009 | 0.297 ± 0.011 | 0.309 ± 0.014 | 0.326 ± 0.008 | 0.324 ± 0.007 | 0.3543 ± 0 |

Table 8: Details of pre-trained GNNs. LR means learning rate.

| Dataset          | BA-Sahpes BA-Community   | PPI0   | PPI1                                | PPI2   | PPI3   | PPI4   | PPI5   | LastFM Asia   | Gen   | Tree-Cycle Tree-Grid   |
|------------------|--------------------------|--------|-------------------------------------|--------|--------|--------|--------|---------------|-------|------------------------|
| LR               | 0.001                    | 0.001  | 0.003                               | 0.001  | 0.001  | 0.001  | 0.001  |               |       |                        |
| Epoch            | 2,000                    | 5,000  | 300                                 | 600    | 100    | 600    | 3,000  |               |       |                        |
| Hidden Dimension | 10                       | 30     | 200                                 | 30     | 100    | 30     | 30     |               |       |                        |
| Train:Val:Test   | 8:1:1                    | 8:1:1  | 10:1:1                              | 8:1:1  | 8:1:1  | 8:1:1  | 10:1:1 |               |       |                        |
| Train Accuracy   | 0.982                    | 0.998  | 9.998 0.998 0.997 0.998 0.994 0.998 | 0.997  | 0.999  | 0.993  | 0.987  |               |       |                        |
| Val Accuracy     | 0.943                    | 0.723  | 0.953 0.959 0.970 0.962 0.976 0.981 | 0.814  | 0.744  | 0.966  | 0.878  |               |       |                        |
| Test Accuracy    | 0.971                    | 0.800  | 0.971 0.967 0.973 0.957 0.980 0.936 | 0.841  | 0.686  | 0.966  | 0.863  |               |       |                        |

space for GNNExplainer includes (lr, epoch, edge size, node feature size, edge entropy, and node feature entropy). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (100-300-600, 0.01-0.05-0.1, 0.005-0.01-0.1, 0.05-0.1, 0.5-1.0, 0.5-1.0) that yielded the highest performance. The explored hyper-parameter space for PGExplainer includes (lr, epoch, edge size, edge entropy, and dimension of MLP). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (0.0010.003-0.01-0.5-0.1, 0.01-0.05-0.1, 0.05-0.1, 64-128) that yielded the highest performance. The explored hyper-parameter space for TAGE includes (lr, epoch, batch size coefficient size, coefficient entropy, and loss type). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (0.000005-0.00005-0.0005-0.005-0.05-0.1, 1-3-5-10-20, 4-1664-128, 0.01-0.05, 0.0005-0.005, 'NCE'-'JSE') that yielded the highest performance. The explored hyper-parameter space for MotifExplainer includes (lr, epoch, embedding dimension of attention module). For each dataset in this study, the experiment was conducted by selecting the combination of hyper-parameters (0.001-0.005-0.01, 30-50-100) that yielded the highest performance. Furthermore, the motif extraction rules for MotifExplainer encompass three categories: 1) cycles within the graph, 2) edges excluding cycles, and 3) motifs constituted by the fusion of cycles involving two or more identical nodes. In our experiments, when the number of motifs extracted from the graphs was fewer than 5000, all three rules were employed to conduct the experiments. However, in cases where the number exceeded 5000 motifs, a random sample of 5000 motifs was utilized.

## D.4 Evaluation Metrics

Sparsity (Li et al., 2022) means the proportion of the presented explanation in the computation graph based on the target node as follows:

$$Sparsity=\frac{1}{|\mathcal{V}|}\sum_{i=1}^{|\mathcal{V}|}1-\frac{|\mathcal{G}_{v_{i}}^{ex}|}{|\mathcal{G}_{v_{i}}|}\tag{9}$$

where G
ex vi denotes the subgraph presented as the explanation for the node vi, Gvi denotes to the computation graph, and ∣G∣ means the number of edges in the graph G.

Fidelity (Li et al., 2022) is calculated by taking the difference in the probability values of the input graph and the probability values when the explanation is excluded from the computation graph based on the target node, as follows:

$$Fidelity=\frac{1}{|\mathcal{V}|}\sum_{i=1}^{|\mathcal{V}|}f_{prob}(\mathcal{G}_{v_{i}})-f_{prob}(\mathcal{G}_{v_{i}}-\mathcal{G}_{v_{i}}^{ex})\tag{10}$$

where f*prob*(Gvi) refers to the probability values of each node vi from the trained GNN f(G) with respect to the correct class.

Table 9: Hyper-parameter settings for UO-Explainer. LR means learning rate.

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 We conducted five experiments for each method by recording the mean and standard deviation in Table 11, 12, 13, and 14.

## F Datasets

We conducted experiments using five synthetic datasets and three real-world datasets. The statistics of datasets are shown in Table 10. Additionally, the detailed information on each dataset is described below.

Random Graph (Holme & Kim, 2002): This dataset does not have node feature values, so we performed node classification tasks using only the structural information by setting all node feature values to the same constant value. We set the class of each node based on the existence of its orbit same as each task number. That is, each node has two classes, c0 (orbit doesn't exist) and c1 (orbit exists) for a total of 73 independent tasks, one for each orbit. The ground truth for class c1 is the target orbit ok for each task.

BA-Shapes (Ying et al., 2019): The node classification dataset consists of a BA graph with 300 nodes as the base and 80 "house-like motifs" randomly attached, each consisting of 5 nodes. The house-like motif precisely matches graphlet g23. The class is determined by the positions of the nodes within the house-like motif; these node positions precisely match the orbits o58, o57, o56 in g23. Therefore, the ground truth for each class is set to o58, o57, o56.

BA-Community (Ying et al., 2019): The node classification dataset is generated by randomly combining two house-like motifs through edges. The first motif is assigned classes c0, c1, c2, c3 based on the positions of the nodes, and the second motif is assigned c4, c5, c6, c7, resulting in a total of eight class labels. The ground truth of each motif is the same as that of the BA-Shapes dataset. Tree-Cycle (Ying et al., 2019): The dataset consists of a balanced binary tree as the base and 80
"six-node cycle motifs" randomly attached. The task is to classify nodes as either not belonging (c0)
to or belonging to a circle (c1). The ground truth for c1 class corresponds to the circle motif. However, 2-5 node graphlets do not encompass the six-node circle motif. Thus, we conducted experiments by incorporating the circle motif and the orbit within the circle into candidate graphlets and orbits.

Tree-Grid (Ying et al., 2019): The base is identical to that of the Tree-Cycle, with a modification involving the attachment of a "nine-node grid" in place of the circle. The task has been further challengingly augmented to classify each node as belonging to the center (c1), periphery (c2), or cross-lines (c3) based on their positions within the grid. Similar to the case of Tree-Cycle, 2-5 node graphlets do not contain the nine-node grid. Therefore, we included the grid motif and the orbits within the grid in the candidate set for experimentation. Protein-Protein Interaction (PPI) (Zitnik & Leskovec, 2017): This is a node classification dataset that represents protein-protein interactions in various human tissues using 24 different graphs. Each node represents a protein and features such as positional, motif gene, and immunological signature are used with a size of 50 dimensions. Protein-to-protein interactions are represented as edges. Each

| Dataset                             | Random Graph   | BA-Shapes   | BA-Community   | PPI   | LastFM-Asia   | Gene   | Tree-Cycle Tree-Grid   |       |
|-------------------------------------|----------------|-------------|----------------|-------|---------------|--------|------------------------|-------|
| Epochs (Orbit basis learning)       | 3,000          | 3,000       | 3,000          | 3,000 | 1,000         | 1,000  | 1,000                  | 1,000 |
| LR (Orbit basis learning)           | 0.005          | 0.005       | 0.005          | 0.001 | 0.003         | 0.005  | 0.005                  | 0.005 |
| Batch size (Orbit basis learning)   | 256            | 256         | 256            | 256   | 2048          | 256    | 256                    | 256   |
| Epochs (Class-orbit score learning) | 2,000          | 1,000       | 1,000          | 1,000 | 1,000         | 1,000  | 1,000                  | 1,000 |
| LR (Class-orbit score learning)     | 0.003          | 0.005       | 0.005          | 0.005 | 0.005         | 0.005  | 0.005                  | 0.005 |

Table 10: Statistics of the dataset used for the experiment. \# symbols mean the number.

| Dataset        | Random Graph   | BA-Shapes   | BA-Community   | PPI    | LastFM-Asia   | Gene   | Tree-Cycle   | Tree-Grid   |
|----------------|----------------|-------------|----------------|--------|---------------|--------|--------------|-------------|
| Avg # of nodes | 340            | 700         | 1,400          | 2,373  | 7,624         | 857    | 871          | 1,231       |
| Avg # of edges | 2,690          | 4,110       | 8,920          | 66,136 | 55,612        | 13,992 | 1,950        | 3,410       |
| # of classes   | 2              | 4           | 8              | 2      | 18            | 2      | 2            | 4           |
| # of tasks     | 73             | 1           | 1              | 121    | 1             | 1      | 1            | 1           |

## E Experimental Results With Standard Deviation.

Table 11: Model-level explanation on random graph datasets: (a) with 2 or 3-layer GCN, and (b) with 2 or 3-layer GIN. Each task is to classify whether the node belongs to the orbit corresponding task number. The evaluation metric is the *Sub-recall*. We conducted experiments five times and then reported the average and standard deviations. The best performances are shown in **bold**.

(a)
Task number 8 11 16 21 27 31 32 33 35 39 45 47 49 57 59 60 61 62 64 Ground-truth orbit o8 o11 o16 o21 o27 o31 o32 o33 o35 o39 o45 o47 o49 o57 o59 o60 o61 o62 o64 D4Explainer 0.2 ± 0.4 0.8 ± 0.4 0.4 ± 0.5 0.4 ± 0.5 0.2 ± 0.4 0.4 ± 0.5 0.4 ± 0.5 0.4 ± 0.5 1.0 ± 0.0 0.8 ± 0.4 0.2 ± 0.4 0.4 ± 0.5 0.4 ± 0.5 0.8 ± 0.4 0.2 ± 0.4 0.0 ± 0.0 0.6 ± 0.5 0.6 ± 0.5 0.2 ± 0.4 GLGExplainer 0.8 ± 0.4 0.6 ± 0.5 0.8 ± 0.4 0.6 ± 0.5 0.8 ± 0.4 1.0 ± 0.0 1.0 ± 0.0 0.8 ± 0.4 0.8 ± 0.4 0.8 ± 0.4 0.8 ± 0.4 0.6 ± 0.5 0.6 ± 0.5 1.0 ± 0.0 0.6 ± 0.5 0.6 ± 0.5 0.8 ± 0.4 0.0 ± 0.0 1.0 ± 0.0 UO-Explainer 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 0 ± 0.0 1.0 ± 0.0 0 ± 0.0 1.0 ± 0.0
(b)
Task number 8 11 16 21 27 31 32 33 35 39 45 47 49 57 59 60 61 62 64 Ground-truth orbit o8 o11 o16 o21 o27 o31 o32 o33 o35 o39 o62 o47 o49 o57 o59 o60 o61 o45 o64 D4Explainer 0.8 ± 0.4 0.6 ± 0.5 0.6 ± 0.5 1.0 ± 0.0 0.6 ± 0.5 0.8 ± 0.4 0.8 ± 0.4 0.6 ± 0.5 1.0 ± 0.0 1.0 ± 0.0 0.6 ± 0.5 0.4 ± 0.5 0.8 ± 0.4 1.0 ± 0.0 0.8 ± 0.4 0.8 ± 0.4 0.2 ± 0.4 0.4 ± 0.5 0.6 ± 0.5 GLGExplainer 0.8 ± 0.4 0.8 ± 0.4 1.0 ± 0.0 0.4 ± 0.5 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 0.8 ± 0.4 1.0 ± 0.0 1.0 ± 0.0 0.6 ± 0.5 0.8 ± 0.4 1.0 ± 0.0 0.6 ± 0.5 0.8 ± 0.4 1.0 ± 0.0 0.0 ± 0.0 1.0 ± 0.0 UO-Explainer 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 Table 12: Model-level explanation results on synthetic datasets. The evaluation metric is the Subrecall. We conducted 5 experiments for each experiment and reported consistent results.

| BA-Shapes          | BA-Community   | Tree-Grid   | Tree-Cycle          |           |           |                     |                               |                     |           |           |           |        |     |
|--------------------|----------------|-------------|---------------------|-----------|-----------|---------------------|-------------------------------|---------------------|-----------|-----------|-----------|--------|-----|
| class1             | class2         | class3      | class1              | class2    | class3    | class5              | class6                        | class7              | class1    | class2    | class3    | class1 |     |
| Ground-truth Orbit | o58            | o57         | o56                 | o58       | o57       | o56                 | o58                           | o57                 | o56       | o73       | o74       | o75    | o76 |
| D4Explainer        | 0.4 ± 0.5      | 0.6 ± 0.5   | 0.4 ± 0.5 0.8 ± 0.4 | 0.2 ± 0.4 | 0.8 ± 0.4 | 0 ± 0.0             | 0.6 ± 0.5                     | 0.4 ± 0.5 0.6 ± 0.4 | 0 ± 0     | 0.2 ± 0.4 | 0.8 ± 0.4 |        |     |
| GLGExplainer       | 1.0 ± 0.0      | 1.0 ± 0.0   | 1.0 ± 0.0 1.0 ± 0.0 | 0.8 ± 0.4 | 0.2 ± 0.4 | 0.8 ± 0.4 1.0 ± 0.0 | 0.8 ± 0.4 0.0 ± 0.0 0.8 ± 0.4 | 0.2 ± 0.4           | 1.0 ± 0.0 |           |           |        |     |
| UO-Explainer       | 1.0 ± 0.0      | 1.0 ± 0.0   | 1.0 ± 0.0 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 0.8 ± 0.4 1.0 ± 0.0 | 1.0 ± 0.0 0.8 ± 0.4 1.0 ± 0.0 | 1.0 ± 0.0           | 1.0 ± 0.0 |           |           |        |     |

node is labeled with 121 dimensions of gene ontology sets, allowing for binary classification of 121 categories, similar to a random graph. We pre-trained a GNN on 20 of these graphs and extracted explanations for randomly selected graphs to conduct experiments. LastFM-Asia (Rozemberczki & Sarkar, 2020): This is a social network dataset of LastFM users in Asia. Each node represents a LastFM user in Asian countries, and edges represent the following relationships between them. Node features are composed of artists that users like, which we converted to one-hot encoding for use in experiments. We perform a node classification task to predict the country of each user across 18 countries. Gene: For qualitative experiments, we created a straightforward dataset using the gene network of natural killer cells in humans, as provided by (Zitnik & Leskovec, 2017). Each node of the dataset represents genes, while the features include positional genes, motif genes, and immunological signature information of each gene, with each feature consisting of 50 dimensions. The Molecular Signatures Database (Subramanian et al., 2005) was used to collect the features for each gene. The labels in the dataset indicate whether a given gene belongs to the cell surface receptor signaling pathway, with 1 indicating inclusion and 0 indicating exclusion in the gene ontology set.

## G Limitation And Future Work

In this study, we propose UO-Explainer, a GNN explanation method that utilizes graphlets and orbits as explanation units. Though we have demonstrated that UO-Explainer can provide high-quality explanations compared to other baselines on extensive experiments, the motifs we can employ as explanatory units are limited to 2-5 node graphlets. Consequently, we assume thatthis constraint may occasionally hinder the capture of motifs or patterns when crucial patterns are extremely large when the input graph is complex. Since many existing works already can approximate the various shaped subgraphs but do not consider the user-centric perspective, we aim to more focus on the case when prior assumption or knowledge is crucial based on pre-defined units toward human-interpretable explanations.

972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 Table 13: Instance-level explanation results on synthetic datasets. The best performances on each dataset are shown in **bold**. We conducted five experiments for each method by recording the mean and standard deviation.

1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 Table 14: Instance-level explanation results on real datasets. The best fidelity on each dataset is shown in **bold**. ∗ notation indicates the lower sparsity setting. We conducted five experiments for each method by recording the mean and standard deviation.

| PPI                                                                                                                                                      | LastFM Asia         |                     |                     |                     |                     |                     |               |               |          |          |          |          |          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------|---------------|----------|----------|----------|----------|----------|
| Task0                                                                                                                                                    | Task1               | Task2               | Task3               | Task4               | Task5               |                     |               |               |          |          |          |          |          |
| Fidelity                                                                                                                                                 | Sparsity            | Fidelity            | Sparsity            | Fidelity            | Sparsity            | Fidelity            | Sparsity      | Fidelity      | Sparsity | Fidelity | Sparsity | Fidelity | Sparsity |
| GNNExplainer 0.100 ± 0.016                                                                                                                               | 0.973 0.325 ± 0.004 | 0.999 0.417 ± 0.067 | 0.999 0.480 ± 0.012 | 0.999 0.250 ± 0.001 | 0.999 0.331 ± 0.001 | 0.999               | 0.114 ± 0.005 | 0.974         |          |          |          |          |          |
| PGExplainer*                                                                                                                                             | 0.023 ± 0.00        | 0.973 0.155 ± 0.008 | 0.999 0.223 ± 0.077 | 0.999 0.101 ± 0.067 | 0.999 0.200 ± 0.001 | 0.999 0.005 ± 0.000 | 0.999         | 0.011 ± 0.002 | 0.974    |          |          |          |          |
| TAGE*                                                                                                                                                    | 0.031 ± 0.002       | 0.973 0.109 ± 0.003 | 0.999 0.233 ± 0.052 | 0.999 0.138 ± 0.034 | 0.999 0.214 ± 0.019 | 0.999 0.101 ± 0.020 | 0.999         | 0.086 ± 0.008 | 0.974    |          |          |          |          |
| MixupExplainer 0.005 ± 0.000                                                                                                                             | 0.973 0.002 ± 0.001 | 0.999 0.257 ± 0.002 | 0.999 0.246 ± 0.003 | 0.999 0.246 ± 0.003 | 0.999 0.129 ± 0.001 | 0.999               | 0.100 ± 0.001 | 0.974         |          |          |          |          |          |
| EiG-Search                                                                                                                                               | 0.180 ± 0.072       | 0.973 0.269 ± 0.032 | 0.999 0.180 ± 0.053 | 0.999 0.379 ± 0.073 | 0.999 0.100 ± 0.079 | 0.999 0.180 ± 0.072 | 0.999         | 0.095 ± 0.021 | 0.974    |          |          |          |          |
| SAME                                                                                                                                                     | 0.022 ± 0.002       | 0.973 0.189 ± 0.003 | 0.999 0.194 ± 0.001 | 0.999 0.189 ± 0.004 | 0.999 0.034 ± 0.005 | 0.999 0.129 ± 0.033 | 0.999         | 0.049 ± 0.012 | 0.974    |          |          |          |          |
| MotifExplainer 0.070 ± 0.008                                                                                                                             | 0.992 0.074 ± 0.003 | 0.999 0.012 ± 0.028 | 0.999 0.129 ± 0.034 | 0.999 0.097 ± 0.005 | 0.999 0.050 ± 0.021 | 0.999               | 0.085 ± 0.004 | 0.994         |          |          |          |          |          |
| UO-Explainer 0.423 ± 0.007 0.999 0.358 ± 0.022 0.999 0.425 ± 0.000 0.999 0.510 ± 0.057 0.999 0.623 ± 0.029 0.999 0.413 ± 0.027 0.999 0.115 ± 0.009 0.993 |                     |                     |                     |                     |                     |                     |               |               |          |          |          |          |          |