000
001
002
003
004
005
006
007
008
009
010
011
012
013
014
015
016
017
018
019
020
021
022
023
024
025
026
027
028
029
030
031
032
033
034
035
036
037
038
039
040
041
042
043
044
045
046
047
048
049
050
051
052
053
054

# **Graph Transformers Get the GIST: Graph Invariant Structural Trait for** **Refined Graph Encoding**

**Anonymous Authors** [1]

**Abstract**

Graph classification is a core machine learning task with diverse applications across scientific fields. Transformers have recently gained
significant attention in this area, addressing
key limitations of traditional Graph Neural Networks (GNNs), including oversmoothing and
oversquashing, while leveraging the attention
mechanism. However, a key challenge remains:
effectively encoding graph structure information
within the all-to-all attention mechanism, arguably the first step of all Graph Transformers.
To address this, we propose a novel structural
feature, termed Graph Invariant Structural Trait
(GIST), designed to capture substructures within
a graph through estimated pairwise node intersections. Furthermore, we extend GIST into a structural encoding method tailored for the attention
mechanism in graph transformers. Our theoretical
analysis and empirical observations demonstrate
that GIST effectively captures structural information critical for graph classification. Extensive
experiments further reveal that graph transformers incorporating GIST into their attention mechanism achieve superior performance compared to
state-of-the-art baselines. These findings highlight the potential of GIST to enhance the structural encoding of Graph Transformers.

**1. Introduction**

Graph classification is a fundamental problem in machine
learning with widespread applications in various domains,
including chemistry, biology, and drug discovery (Dwivedi
et al., 2022a;c; Irwin et al., 2012; Wu et al., 2017). The ability to classify graphs accurately enables advancements in
predicting molecular properties, understanding complex bio

1Anonymous Institution, Anonymous City, Anonymous Region,
Anonymous Country. Correspondence to: Anonymous Author
_<_ anon.email@domain.com _>_ .

Preliminary work. Under review by the International Conference
on Machine Learning (ICML). Do not distribute.

logical interactions, and discovering novel therapeutic compounds. Traditional Graph Neural Networks (GNNs) (Kipf
& Welling, 2017; Han et al., 2022) have been the cornerstone for such tasks, leveraging neighborhood aggregation
to learn node and graph representations. However, GNNs often suffer from limitations such as oversmoothing (Keriven,
2022), oversquashing (Black et al., 2023), and restricted
expressivity (Wang & Zhang, 2024) due to their reliance on
local message-passing mechanisms.

Recently, Transformers (Vaswani et al., 2017) have emerged
as a promising alternative for graph representation learning
due to their global attention mechanism, which addresses
many of the inherent limitations of GNNs. Transformers’ ability to model complex interactions between entities makes them particularly attractive for graph classification (Ying et al., 2021). However, applying Transformers to
graph data is not a seamless procedure, still posing unique
challenges. Unlike sequential or image data, graph nodes
typically lack inherent self-identity, making it difficult for
Transformers to distinguish between entities purely based
on their features. Without incorporating meaningful structural information, the attention mechanism in Transformers
struggles to capture complex graph relationships effectively.

Existing approaches have attempted to improve Transformers with graph structural inductive bias by integrating positional or structural features, such as shortest path distances (Ying et al., 2021), Laplacian eigenvector-based encodings (Dwivedi et al., 2022a), and random walk-based features (Rampa´sek et al.ˇ, 2022; Ma et al., 2023). While these
methods provide some structural context, they either fail to
capture comprehensive substructural information essential
for distinguishing complex graph patterns (Rampa´sek et al.ˇ,
2022) or focus predominantly on a limited set of substructures while neglecting higher-order structural relationships
(Wollschlager et al., 2024). The challenge remains to identify a more expressive and comprehensive set of structural
features, and devise efficient methods for encoding them
within the Transformer’s self-attention mechanism.

In this work, we introduce a novel structural feature called
Graph Invariant Structural Trait (GIST), which captures the
inherent substructures within a graph by estimating _k_ -hop
pairwise node intersections. Our approach is grounded in

1

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

055
056
057
058
059
060
061
062
063
064
065
066
067
068
069
070
071
072
073
074
075
076
077
078
079
080
081
082
083
084
085
086
087
088
089
090
091
092
093
094
095
096
097
098
099
100
101
102
103
104
105
106
107
108
109

the theoretical understanding that the cardinality of the intersection between two nodes’ _k_ -hop neighborhoods can
serve as an effective permutation-invariant feature for substructure characterization, providing a robust foundation
for graph classification. Incorporating GIST as a structural bias enhances the Transformer’s capability to discern
complex graph patterns, leading to improved classification
performance. We further propose an efficient randomized
algorithm to estimate GIST, ensuring scalability across large
(number of) graphs. Through extensive experiments on various graph classification benchmarks, we demonstrate that
integrating GIST into Graph Transformers achieves stateof-the-art performance and offers deeper insights into the
structural properties of graph data.

Our key contributions are as follows:

- We introduce GIST, a method that encodes graph structure
using pairwise _k_ -hop substructure vector. These substructure vectors are efficiently computed by estimating the
interaction cardinality between the _k_ -hop neighborhoods
of node pairs.

- We incorporate GIST into attention mechanisms of graph
Transformers to enhance structural encoding. We provide
both theoretical and empirical evidence demonstrating its
effectiveness as a graph-invariant representation.

- We evaluate GIST-augmented graph Transformers on standard graph classification benchmarks, showing consistent
performance improvements.

The introduction of GIST opens new avenues for enhancing the structural encoding capabilities of Transformers,
paving the way for more effective and interpretable graph
classification models. [1]

**2. Motivation**

Transformers, originally designed for sequential data, lack
an inherent mechanism to capture the structural biases of
graph data as highlighted in (Ying et al., 2021; Rampa´sekˇ
et al., 2022). Without a well-designed structural bias (structural encoding), they treat all nodes as equally related, failing to utilize the relational dependencies critical for graph
tasks (Ying et al., 2021; Brody et al., 2022).

**Challenge 1.** **Capturing Graph Substructures in Struc-**
**tural Encoding.** The first key challenge in designing effective structural encodings for Graph Transformers is capturing the substructures within a graph, as these substructures
often represent critical local patterns, or fragments that define the graph’s overall characteristics (Ying et al., 2021;
Ma et al., 2023; Wollschlager et al., 2024). While many
early-stage structural encoding methods, such as shortest
path distance (SPD) (Ying et al., 2021), provide a notion of

1The code will be made publicly available upon publication.

(a) ( _u, v_ 1) from the same 6-ring substructure

(b) ( _u, v_ 2) from different substructures: a 6-ring and a 2-path

_Figure 1. k_ -hop Substructure Vector Visualization (Def. 3.1) of
ZINC molecule. The substructures of node pairs in the form of **in-**
**tersection cardinality** of their common neighborhood at different
distances from _u_ and _v_ are **“GIST”-ed into the Substructure Vec-**
**tor** . Specifically, each cell ( _ku, kv_ ) in the Substructure Vector denotes the number of nodes that are **exactly** _ku_ hops from _u_ and _kv_
hops from _v_ . The variations in the Substructure Vector help the selfattention mechanism distinguish structural differences between
node pairs, such as ( _u, v_ 1) and ( _u, v_ 2). For example, in Figure 1a,
the pair ( _u, v_ 1), which belongs to the **same** 6-ring substructure,
has intersection cardinalities _I_ (2 _,_ 2) = _I_ (4 _,_ 2) = _I_ (2 _,_ 4) = 1. In contrast, the pair ( _u, v_ 2), where _u_ and _v_ 2 belong to **different** substructures (a 6-ring and a 2-path), has _I_ (2 _,_ 2) = _I_ (4 _,_ 2) = _I_ (2 _,_ 4) = 0.

proximity between nodes, they often struggle to effectively
capture and represent substructures.

**Challenge 2.** **Aggregating Diverse Substructures Infor-**
**mation.** As highlighted in (Wollschlager et al., 2024), it is
equally important for structural encodings to enable the aggregation of information across diverse substructures, rather
than restricting it to similar or localized patterns. Graphs,
such as molecules, often exhibit a variety of substructures
that interact in complex ways, and limiting information
flow to nodes in different structures can hinder the model’s
ability to capture global dependencies and cross-pattern interactions. This is particularly important in domains like
chemistry, biology, and social networks, where functional
or structural properties often arise from specific subgraph

2

110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

( _u, v_ 1). This contrast highlights how different substructure
compositions lead to distinct intersection patterns, enabling
the model to effectively distinguish between structurally
similar and dissimilar node pairs, guiding the self-attention
mechanism to weigh higher-order interactions accordingly.

_Figure 2._ Node Clustering via Spectral Clustering Using Learned
GIST Features in Graph Transformers on ZINC molecule graph.
**Nodes** **within** **the** **same** **local** **substructures** **are** **clustered** **to-**
**gether** : 6-rings (purple), 2-path (cyan), and X-shape (light blue).

arrangements (i.e., rings and bonds in molecules) rather than
the global graph structure alone (Yang et al., 2018; Yu &
Gao, 2022). Many recent structural biases, such as shortest
path distance (Ying et al., 2021) or those based on random
walks (Rampa´sek et al.ˇ, 2022; Ma et al., 2023), are effective at capturing simple substructures like cycles but tend
to focus predominantly on these patterns, neglecting the
interactions between different substructures (Wollschlager
et al., 2024). For example, in Figure 2, it is more beneficial
for _u_ to aggregate information from the 6-ring, X-shape,
and 2-path substructures rather than solely focusing on another 6-ring that mirrors its own structural pattern. This
highlights the need for a structural encoding that can help
attention mechanisms effectively learn the substructures
while enabling nodes to distinguish their own substructures
from those of others, guiding attention based on the distinct
structural relationships between nodes.

**Observation 1:** **Intersection Cardinality as a Discrimina-**
**tive Subgraph Feature.** Empirically, we observe that the
intersection cardinality of common neighborhoods between
two nodes ( _u, v_ ) can also serve as a powerful and discriminative feature encoding the _k−_ hop subgraph structures. As
illustrated in Figure 1, the intersections of common neighborhoods at different hop distances provide a structured way
for _u_ to differentiate between the ring structure containing
_v_ 1 and the 2-path structure containing _v_ 2, based on the differences in the in-between graph structures. Specifically,
for ( _u, v_ 1), which belongs to the same 6-ring substructure,
the intersection cardinality values _I_ (2 _,_ 2), _I_ (4 _,_ 2), and _I_ (2 _,_ 4)
are all nonzero, indicating strong shared neighborhood connectivity. In contrast, ( _u, v_ 2), which belongs to different
substructures (a 6-ring and a 2-path), lacks these intersection
values but instead exhibits nonzero intersection cardinality
in positions such as _I_ (3 _,_ 2) and _I_ (2 _,_ 3), which are absent for

**Observation 2: Intersection Cardinality Enhances Struc-**
**tural** **Awareness** **in** **Self-Attention** **Mechanisms.** Moreover, we empirically observe that incorporating an attention
mechanism with intersection cardinality as an attention bias
enables the attention mechanism to learn distinct substructures within the graph. In Figure 2, we train a Transformer
architecture on the on ZINC dataset (Dwivedi et al., 2022a),
introducing only the intersection cardinality (formally defined in Section 4 as GIST) as a bias in the attention scores.
After training the model, we apply Spectral Clustering to
group nodes based on the learned GIST features. The GIST
features facilitate representation aggregation across structurally similar regions, allowing node _u_ to integrate information from another ring structure. This effect is evident
as nodes from both rings are grouped into the same clusters, marked in dark blue and cyan. Furthermore, certain
nodes positioned at the boundaries of these substructures
act as “information exchange points”, facilitating communication between distant regions of the graph. For example,
the cyan-colored node within the ”X” substructure is assigned to the same cluster as the ring nodes, effectively
facilitating representation aggregation between two different substructures—an ability that current GNNs and Graph
Transformers struggle with due to their inherent locality
constraints. We note that this is **not a cherry-picked ex-**
**ample** ; rather, this phenomenon **consistently occurs across**
**multiple samples** in the ZINC dataset after the Transformer
is trained.

**3. GIST: Graph Invariant Structural Trait**

In this section, we formally introduce the graph invariant
structural trait (GIST). We start by introducing how to encode the _k_ -hop substructure of a node pair ( _u, v_ ) based
on the _k_ -hop common neighborhood between them. Next,
we introduce how to use encoded _k_ -hop substructures in a
graph to form GIST. Finally, we introduce how to efficiently
compute GIST with randomized hashing algorithms.

**Notation:** We denote an undirected graph _G_ = ( _V, E_ ),
which contains a set _V_ of _n_ nodes (vertices) and a set _E_ of
_m_ edges (links). Each node _v_ _∈V_ has _dn_ associated node
features _xv_ _∈_ R _[d][n]_, while each edge _eu,v_ _∈E_ connecting
node pair ( _u, v_ ) has _de_ associated edge features _yu,v_ _∈_ R _[d][e]_
( _yu,v_ = **0** _[d][e]_ if there is no edge between _u_ and _v_ ). For every
node _v_ _∈V_, we denote its _k_ -hop neighborhoods as _Nk_ ( _v_ ).
_Nk_ ( _v_ ) consists of all vertices that can be reached from _v_
with less or equal to _k_ edges. Subsequently, we define
the _k_ -hop common neighborhood of a node pair ( _u, v_ ) as

3

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219

_Cku,kv_ ( _u, v_ ) = _Nku_ ( _u_ ) _∩Nkv_ ( _v_ ), which is a set of nodes
in the graph that can be reached within _ku_ from _u_ and _kv_
edges from _v_, respectively.

**3.1. Encoding** _k_ **-hop Substructure of a Node Pair**

We encode the _k_ -hop substructure of a node pair ( _u, v_ ) in a
vector. This vector is computed based on the _k_ -hop common
neighborhood _Cku,kv_ ( _u, v_ ).

**Definition 3.1** ( _k_ -hop substructure vector) **.** Given a pair of
node ( _u, v_ ) _∈G_, we propose capturing the _k−_ hop graph
structure between _u_ and _v_ with two types of features computed by _k_ -hop common neighborhood _Cku,kv_ ( _u, v_ ) as follows:

- _Iku,kv_ ( _u, v_ ) as the cardinality of common neighborhoods
that are exactly _ku_ hops from node _u_ and _kv_ hops from
node _v_, computed as:

_Iku,kv_ ( _u, v_ ) = _|Cku,kv_ ( _u, v_ ) _| −_ 
_x≤ku_ _,_ _y≤kv_
( _x,y_ )=( _ku,kv_ )

_Ix,y_ ( _u, v_ ) _,_

where _I_ 1 _,_ 1( _u, v_ ) = _|C_ 1 _,_ 1( _u, v_ ) _|_ for _u_ and _v_ .

- _Tku_ ( _u, v_ ): the cardinality of nodes that are exactly _ku_
hop from vertex _u_ and greater than _k_ hop from _v_ (and
vice-versa for _Tkv_ ( _v, u_ )), computed as:

_k_

- _Ii,j_ ( _u, v_ )

_j_ =1

GIST provides a compact representation of a graph’s structural properties, encoding its topology and connectivity
patterns by capturing higher-order relational dependencies
among nodes and substructures. This encoding enables
the differentiation of substructures, offering a detailed understanding of complex higher-order relationships, as illustrated in Figure 2 and Section 2. We would like to note one
component of this representation: the diagonal entry _Si,i_ ( _G_ ),
which essentially encodes the _k_ -hop neighborhood surrounding a node _vi_ _∈V_ . This local structure provides a positional
reference that differentiates nodes based on their placement
within the global graph topology, enabling the model to
capture long-range dependencies beyond direct connectivity.
Mathematically, GIST represents pairwise node interactions
as a matrix, where each interaction is encoded as a vector of
dimension ( _k_ [2] + 2 _k_ ). This formulation preserves both local
and global structural information, making GIST a comprehensive descriptor of graph architecture suitable for various
analytical and learning-based applications.

**3.3. Efficiently Compute GIST with Randomized**
**Hashing**

In this section, we show how to efficiently compute GIST by
reducing the time complexity from _O_ ( _k_ [2] _n_ [4] ) to _O_ ( _k_ [2] _n_ [2] ). It
is obvious that computing GIST _S_ ( _G_ ) requires _O_ ( _k_ [2] _n_ [4] )
time complexity. We note that for a node pair ( _u, v_ ),
the exact computation of their _k_ -hop common neighborhood _Cku,kv_ ( _u, v_ ) incurs a cost of _O_ ( _n_ [2] ), while calculating _Su,v_ ( _G_ ) requires _O_ ( _k_ [2] _n_ [2] ). Consequently, computing
_Su,v_ ( _G_ ) for all node pairs in a graph _G_ results in an overall
complexity of _O_ ( _k_ [2] _n_ [4] ). Exact intersection calculations are
computationally expensive, making them impractical for
large graphs. Following (Chamberlain et al., 2022; Le et al.,
2024), we propose to efficiently and unbiasedly estimate the
cardinality of _k_ -hop common neighborhood _Cku,kv_ ( _u, v_ ) by
decomposing it as:

_|Cku,kv_ ( _u, v_ ) _|_ = _Jku,kv_ ( _u, v_ ) _· Uku,kv_ ( _u, v_ ) (1)

Here, _Jku,kv_ ( _u, v_ ) represents the Jaccard similarity between
_ku_ -hop neighborhoods _Nku_ ( _u_ ) and _kv_ -hop neighborhoods
_Nkv_ ( _v_ ). _Uku,kv_ ( _u, v_ ) denotes the cardinality of the union
_Nku_ ( _u_ ) _∪Nkv_ ( _v_ ). Next, we can estimate _Jku,kv_ ( _u, v_ ) with
the constant-time collisions of the MinHash signatures of
_Nku_ ( _u_ ) and _Nkv_ ( _v_ ) as shown in Algorithm 1. We note that
MinHash provides an unbiased estimator to the _Jku,kv_ ( _u, v_ )
since the collision probability between the MinHash signatures of _Nku_ ( _u_ ) and _Nkv_ are equal to _Jku,kv_ ( _u, v_ ) We
can also estimate _Uku,kv_ ( _u, v_ ) with the mergeable HyperLogLog sketch as Algorithm 1. We note that HyperLogLog
also provides an unbiased estimator to _Uku,kv_ ( _u, v_ ).

Finally, we multiply the estimated _J_ ˜ _ku,kv_ ( _u, v_ ) and

_Tku,k_ ( _u, v_ ) = _|Nku_ ( _u_ ) _|−Tku−_ 1 _,k_ ( _u_ ) _−_

_ku_

_i_ =1

For any node pair ( _u, v_ ), there would be _k_ [2] numbers of
_Iku,kv_ ( _u, v_ ), _k_ numbers of _Tku,k_ ( _u, v_ ), and _k_ numbers of
_Tkv,k_ ( _v, u_ ). Finally, we encode the _k−_ hop graph substructure surrounding node pair ( _u, v_ ) as a _k−_ hop substructure
vector _Sk_ ( _u, v_ ). _Sk_ ( _u, v_ ) starts with _Iku,kv_ ( _u, v_ ) for every
pair of _ku, kv_ _≤_ _k_ . Next, we fill the rest of the dimension in _Sk_ ( _u, v_ ) with _Tku,k_ ( _u, v_ ) for each _ku_ _≤_ _k_ hop and
_Tkv,k_ ( _v, u_ ) for each _kv_ _≤_ _k_ hop.

As we see from Definition 3.1, computing the _k−_ hop substructure vector requires first compute the cardinality of the
_k_ -hop common neighborhood _Cku,kv_ ( _u, v_ ).

**3.2. GIST: Graph Invariant Structural Trait**

We define GIST as a three-dimensional matrix defined on
the _k_ -hop common neighborhood _Cku,kv_ ( _u, v_ ) (see Definition 3.1) between every pair of node ( _u, v_ ) in graph _G_ .

**Definition** **3.2** (Graph Invariant Structural Trait (GIST)) **.**
Let _G_ = ( _V, E_ ) denote a graph with _n_ nodes ( _|V|_ = _n_ ). We
define the _k_ -hop graph invariant structural trait (GIST) as a
matrix _S_ ( _G_ ) _∈_ R _[n][×][n][×]_ [(] _[k]_ [2][+2] _[k]_ [)], where each entry _Si,j_ ( _G_ ) _∈_
R _[k]_ [2][+2] _[k]_ is the _k_ -hop substructure between node _vi, vj_ (see
Definition 3.1). We also use _S_ ( _G_ ) _u,v_ to represent the GIST
value between node _u, v_ _∈G_ .

4

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274

**Algorithm 1** Algorithm for computing intersection cardinality _|Cku,kv_ ( _u, v_ ) _|_

**Input:** Graph _G_ = ( _V, E_ ), max hops _k_, hops _ku, kv_, _m_
MinHash functions _H_ = _{h_ 1 _, . . ., hm}_, HyperLogLog
parameter _p_ and regularizer constant _αp_
**Output:** Intersection cardinality _|Cku,kv_ ( _u, v_ ) _|_
_{_ Step 1. Pre-compute MinHash signatures _}_
**for** _v_ _∈V, hj_ _∈_ _H_ **do**

_Mv_ [ _j,_ 0] _←_ _hj_ ( _v_ ) _{_ Initialize MinHash signatures _}_
**end for**
**for** _i_ = 1 **to** _k_ **do**

**for** _v_ _∈V, hj_ _∈_ _H_ **do**

_Mv_ [ _j, i_ ] _←_ min
_u∈N_ ( _v_ )

- _Mu_ [ _j, i −_ 1] _, Mv_ [ _j, i −_ 1]�

**end for**
**end for**
_{_ Step 2. Pre-compute HyperLogLog sketches _}_
_m ←_ 2 _[p]_

**for** _v_ _∈V_ **do**

Compute _k_ -hop HyperLogLog sketch _Hv_ _∈_ R _[m][×][k]_

**end for**
_{_ Step 3. Compute intersection cardinality _}_
**for** ( _u, v_ ) _∈V_ _× V_ **do**

_J_ ˜ _ku,kv_ ( _u, v_ ) _←_ JACCARD-EST( _ku, kv, m, Mu, Mv_ )

_U_ ˜ _ku,kv_ ( _u, v_ ) _←_ HLL-EST( _ku, kv, Hu, Hv_ )
_|Cku,kv_ ( _u, v_ ) _| ←_ _J_ [˜] _ku,kv_ ( _u, v_ ) _·_ _U_ [˜] _ku,kv_ ( _u, v_ )
**end for**
**return** _|Cku,kv_ ( _u, v_ ) _|_

**Function:** JACCARD-EST( _ku, kv, m, Mu, Mv_ )
**Input:** hops _ku, kv_, number of MINHASH functions _m_,
and _k−_ hop MinHash values _Mu, Mv_
**Output:** Jaccard similarity _J_ [˜] _ku,kv_ ( _u, v_ )
_J_ ˜ _ku,kv_ ( _u, v_ ) _←_ 0
**for** _j_ = 1 **to** _m_ **do**

**if** _Mu_ ( _j, ku_ ) = _Mv_ ( _j, kv_ ) **then**

_J_ ˜ _ku,kv_ ( _u, v_ ) _←_ _J_ [˜] _ku,kv_ ( _u, v_ ) + 1
**end if**
**end for**

_J_ ˜ _ku,kv_ ( _u, v_ ) _←_ _J_ [˜] _ku,kv_ ( _u, v_ ) _/m_
**return** _J_ [˜] _ku,kv_ ( _u, v_ )
**EndFunction**

**Function:** HLL-EST( _ku, kv, Hu, Hv_ )
**Input:** hops _ku, kv_, HyperLogLog sketches _Hu, Hv_
**Output:** Union cardinality _U_ [˜] _ku,kv_ ( _u, v_ )
_Hku,kv_ _←_ **0** _[m]_

**for** _j_ = 1 **to** _m_ **do**

_Hku,kv_ [ _j_ ] _←_ max� _Hu_ [ _j, ku_ ] _, Hv_ [ _j, kv_ ]�

**end for**

_U_ ˜ _ku,kv_ ( _u, v_ ) _←_ _αpm_ [2] ( [�] _[m]_ _i_ =0 [2] _[−][H][ku,kv]_ [ [] _[i]_ []][)] _[−]_ [1]

**return** _U_ [˜] _ku,kv_ ( _u, v_ )
**EndFunction**

_U_ ˜ _ku,kv_ ( _u, v_ ) together and form an unbiased estimator to
_|Cku,kv_ ( _u, v_ ) _|_ . This unbiased estimation can serve as an
efficient alternative to exact computation for _|Cku,kv_ ( _u, v_ ) _|_ .
With MinHash and HyperLogLog, we reduce the computation time for _Su,v_ ( _G_ ) from _O_ ( _k_ [2] _n_ [2] ) to _O_ ( _k_ [2] ), leading to
_O_ ( _k_ [2] _n_ [2] ) time for compute GIST.

**4. Graph Transformers Get the GIST**

We see GIST can be naturally integrated into graph tansformers for graph structural encoding in the self-attention
mechanism. As a result, we introduce the GIST attention
for graph transformers.

**Definition** **4.1** (GIST attention) **.** Let _G_ = ( _V, E_ ) denote
a graph with _n_ nodes ( _|V|_ = _n_ ). Let _xu_ _∈_ R _[d][n]_ denote
the representation of node _u_ _∈V_ . Let _yu,v_ _∈_ R _[d][e]_ denote
the representation of edge between nodes _u, v_ _∈V_ . Let
_wv_ _∈_ R _[d][n][×][d][n]_ and _we_ _∈_ R _[d][n][×][d]_ denote the model weight.
Let _S_ ( _G_ ) denote the _k_ -hop GIST computed from _G_ (see
Definition 3.2). We define the GIST attention as a transform
_ψ_ : R _[d][n]_ _→_ R _[d][n]_ on every node feature _xu_ as:

_ψ_ ( _xu_ ) =    - _Au,v ·_ ( _wvxv_ + _weA_ [ˆ] _u,v_ ) _,_

_v∈V_

where _A_ [ˆ] _u,v_ _∈_ R _[d]_ and attention score _Au,v_ _∈_ R are:

_eu,v_ = _ϕy_ ( _yu,v_ ) + _ϕS_ ( _Su,v_ ( _G_ ))

_Au,v_ = _σ_     - _⟨wQxu_ + _wKxv_ + _wb, eu,v⟩_     - _._
_A_ ˆ _u,v_ = ( _wQxu_ + _wKxv_ + _wb_ ) _⊙_ _eu,v._

Here _ϕy_ : R _[d][e]_ _→_ R _[d]_ and _ϕS_ : R _[k]_ [2][+2] _[k]_ _→_ R _[d]_ are MLP
networks that align the representation of edge and GIST (see
Definition 3.2) into same _d_ -dimensional vector for addition.
_wQ, wK_ _∈_ R _[d][×][d][n]_ and _wb_ _∈_ R _[d]_ are model weights and
bias, respectively.

GIST attention can be viewed as a graph invariant with the
following statement.

**Theorem 4.2** (Informal version of Theorem A.1) **.** _Let G_ =
( _V, E_ ) _denote a graph with n nodes (|V|_ = _n)._ _Let S_ ( _G_ ) _∈_
_denote the k-hop GIST (see Definition 3.2) computed on G._
_We show that the GIST attention (see Definition 4.1) ψ_ ( _xu_ )
_for every node u ∈V_ _is invariant under graph isomorphism._

We provide the formal version of this theorem and proof in
Appendix A. In other words, the permutation of node orders
in the graph does not break the substructure in the graph
due to graph isomorphism. As a result, it does not affect the
value of GIST.

We use GIST attention as the building blocks and form a
graph transformer with multiple GIST attention blocks. We
view GIST attention as a way of modelling node interactions
with the awareness of the graph structure.

5

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

features are within [1,2,3,4,5,6]-hops of each node, the batch
size is chosen among [32, 64, 128, 256], the number of layers is chosen among [2, 4, 6, 8], the number of heads is
chosen among [2, 4, 8, 16, 32], the number of hidden dimensions is chosen among [16, 32, 64, 128], and learning rate is
chosen among [0.0001, 0.0003, 0.0005, 0.002]. The chosen
optimizer is AdamW. Our model is trained at 200 epochs for
all datasets, except for MUV and HIV, where it is trained
for 100 epochs. All model training and evaluations were
conducted on NVIDIA A100 GPUs with 80G memory.

**Dataset Statistics.** We provide the statistics of 12 datasets
used in our experiments to evaluate the performance of our
proposed GIST in Table 1.

|Dataset|# Graphs|Avg. # nodes|Avg. # edges|Prediction task|Metric|
|---|---|---|---|---|---|
|BBBP<br>Tox21<br>Toxcast<br>Sider<br>Clintox<br>Bace<br>MUV<br>HIV|2,050<br>7,831<br>8,597<br>1,427<br>1,484<br>1513<br>93,087<br>41,127|23.9<br>18.6<br>18.7<br>33.6<br>26.1<br>34.1<br>24.2<br>25.5|51.6<br>38.6<br>38.4<br>70.7<br>55.5<br>73.7<br>52.6<br>54.9|binary classifcation<br>12-task classifcation<br>617-task classifcation<br>27-task classifcation<br>2-task classifcation<br>binary classifcation<br>17-task classifcation<br>binary classifcation|ROC-AUC<br>ROC-AUC<br>ROC-AUC<br>ROC-AUC<br>ROC-AUC<br>ROC-AUC<br>ROC-AUC<br>ROC-AUC|
|Peptides-func<br>Peptides-struct|15,535<br>15,535|150.94<br>150.94|307.30<br>307.30|10-task classifcation<br>11-task regression|Avg. Precision<br>Mean Abs. Error|
|Zinc Subset<br>Zinc Full|12,000<br>249,456|23.2<br>23.2|49.8<br>49.8|regression<br>regression|Mean Abs. Error<br>Mean Abs. Error|

**5.2. Long-Range Graph Benchmark (LRGB)**

We evaluate the ability of our proposed GIST to learn longrange dependencies using two graph classification datasets
from LRGB (Dwivedi et al., 2022c): Peptides-func and
Peptides-struct. These datasets provide a robust benchmark
for assessing graph classification methods in handling longrange dependencies and addressing structural challenges
such as over-squashing and over-smoothing of many GNNs.
As shown in Table 2, GIST significantly enhances the capability of Transformers, achieving state-of-the-art performance on LRGB. This demonstrates that encoding structural information into Transformer-based architectures can
mitigate the limitations of existing GNNs in capturing longrange interactions. Regarding **RQ2**, our results demonstrate
that GIST effectively captures long-range dependencies by
encoding structural relationships beyond local neighborhoods, leading to improved classification performance.

**5.3. ZINC and ZINC-full**

We further evaluate our proposed GIST on two molecular
property prediction datasets: ZINC (Dwivedi et al., 2022a)
and ZINC-full (Irwin et al., 2012). These datasets are widely
used benchmarks for assessing the ability of graph-based
models to learn molecular representations and predict chemical properties. ZINC, with its constrained molecular structures and well-defined tasks, serves as a standard benchmark

275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
296
297
298
299
300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
324
325
326
327
328
329

**5. Experiment**

In this section, we aim to rigorously evaluate the effectiveness of GIST by addressing the following key research
questions and providing corresponding insights:

- **RQ 1** : How well does GIST facilitate the learning and differentiation of substructures in graph classification tasks?

- **RQ 2** : To what extent does GIST enable long-range dependencies in Graph Transformers?

- **RQ 3** : How sensitive is GIST to the maximum hop distance for computing intersection cardinality?

**5.1. Settings**

We evaluate the proposed method on three benchmark suites
comprising a total of 12 datasets, spanning small-scale to
large-scale settings: the Long-Range Graph Benchmark
(LRGB) (Dwivedi et al., 2022c), MoleculeNet (Wu et al.,
2017), ZINC (Dwivedi et al., 2022a), and ZINC-full (Irwin
et al., 2012). These datasets are specifically curated to
emphasize challenges in structural encoding and long-range
dependency modeling, with diverse applications in domains
such as chemistry and biology.

**Baselines.** We benchmark the performance of our method
against recent state-of-the-art baselines across multiple
categories, including Graph Transformers, Graph Neural Networks (GNNs), hybrid models combining Transformers and GNNs, as well as pretrained graph models:
GraphGPS (Rampa´sekˇ et al., 2022), GRIT (Ma et al.,
2023), Subgraphormer (Bar-Shalom et al., 2024), FragNet (Wollschlager et al., 2024), GatedGCN (Dwivedi et al.,
2022c), SAN (Kreuzer et al., 2021), Graphormer (Ying et al.,
2021), Graphormer-GD (Zhang et al., 2023b), GCN (Kipf
& Welling, 2017), GIN (Xu et al., 2018), NGNN (Zhang &
Li, 2021), DS-GNN (Bevilacqua et al., 2022), DSS-GNN
(Bevilacqua et al., 2022), GNN-AK (Zhao et al., 2022),
GNN-AK+ (Zhao et al., 2022), SUN (Frasca et al., 2022),
OSAN (Qian et al., 2022), DS-GNN (Bevilacqua et al.,
2023), GNN-SSWL (Zhang et al., 2023a), GNN-SSWL+
(Zhang et al., 2023a), GraphMVP (Liu et al., 2022), MGSSL
(Zhang et al., 2021), and GraphFP (Luong & Singh, 2023).

**Experimental Settings.** For each dataset, we train our proposed method on the training set and select the epoch with
the best validation performance. We then report the test results corresponding to this selected epoch. The performance
of our method is presented as the mean ± standard deviation
over 5 runs with different random seeds. The performance
metrics for each baseline are obtained either directly from
their original publications or reproduced by us using the
best hyperparameters reported in their studies.

**Hyperparameters.** Particularly for our method, we perform
a grid search to find the optimal hyperparameter combination for each dataset whenever feasible. The intersection

6

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

330
331
332
333
334
335
336
337
338
339
340
341
342
343
344
345
346
347
348
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384

_Table 2._ Performance of GIST on Peptides datasets from LRGB:

|Top-3 Results Highlighted in Red,|Blue, and Oran|nge.|
|---|---|---|
|**Model**|**Peptides-struct**<br>MAE_ ↓_|**Peptides-func**<br>AP_ ↑_|
|GCN (Kipf & Welling, 2017)<br>GIN (Xu et al., 2018)<br>Subgraphormer (Bar-Shalom et al., 2024)<br>FragNet (Wollschlager et al., 2024)<br>GatedGCN+RWSE (Dwivedi et al., 2022c)<br>GRIT (Ma et al., 2023)<br>GraphGPS (Ramp´aˇsek et al., 2022)<br>SAN+LapPE (Kreuzer et al., 2021)<br>SAN+RWSE (Kreuzer et al., 2021)<br>GNN-SSWL+ (Zhang et al., 2023a)|0_._3496_ ±_ 0_._0013<br>0_._3547_ ±_ 0_._0045<br>0_._2494_ ±_ 0_._0020<br>**0.2462**_±_** 0.0021**<br>0_._3357_ ±_ 0_._0006<br>**0.2460**_±_** 0.0012**<br>0_._2500_ ±_ 0_._0012<br>0.2683_ ±_ 0.0043<br>0_._2545_ ±_ 0_._0012<br>0_._2570_ ±_ 0_._006|0_._5930_ ±_ 0_._0023<br>0_._5498_ ±_ 0_._0079<br>0_._6415_ ±_ 0_._052<br>**0.6678**_ ±_** 0.0050**<br>0_._6069_ ±_ 0_._0035<br>**0.6988**_±_** 0.0082**<br>0_._6535_ ±_ 0_._0041<br>0.6384_ ±_ 0.0121<br>0_._6439_ ±_ 0_._0075<br>0_._5847_ ±_ 0_._0050|
|GIST (ours)|**0.2442**_ ±_** 0.0011**|**0.6783**_ ±_** 0.0087**|

for evaluating a model’s effectiveness in capturing molecular topology and learning chemically relevant features. In
contrast, ZINC-full provides a large-scale and more diverse
dataset, offering a more rigorous test of a model’s generalization capability across a broader range of molecular
structures and chemical compositions. As shown in Table
3, our approach significantly improves the ability of Transformers to learn molecular graph representations, achieving
superior predictive performance. These results demonstrate
that incorporating structural priors into Transformer architectures can enhance molecular property prediction, making
GIST a promising approach for advancing deep learning
methods in computational chemistry and drug discovery.

_Table 3._ Performance of GIST on ZINC and ZINC-full: Top-3
Results Highlighted in **Red**, **Blue**, and **Orange** .

|Model|ZINC<br>MAE ↓|ZINC-full<br>MAE ↓|
|---|---|---|
|GCN (Kipf & Welling, 2017)<br>GIN (Xu et al., 2018)|0_._367_ ±_ 0_._011<br>0_._526_ ±_ 0_._051|0_._113_ ±_ 0_._002<br>0_._088_ ±_ 0_._002|
|NGNN (Zhang & Li, 2021)<br>DS-GNN (Bevilacqua et al., 2022)<br>DSS-GNN (Bevilacqua et al., 2022)<br>GNN-AK (Zhao et al., 2022)<br>GNN-AK+ (Zhao et al., 2022)<br>SUN (Frasca et al., 2022)<br>OSAN (Qian et al., 2022)<br>DS-GNN (Bevilacqua et al., 2023)<br>GNN-SSWL (Zhang et al., 2023a)<br>GNN-SSWL+ (Zhang et al., 2023a)|0_._111_ ±_ 0_._003<br>0_._116_ ±_ 0_._009<br>0_._102_ ±_ 0_._003<br>0_._105_ ±_ 0_._010<br>0_._091_ ±_ 0_._002<br>0_._083_ ±_ 0_._003<br>0_._154_ ±_ 0_._008<br>0_._087_ ±_ 0_._003<br>0_._082_ ±_ 0_._003<br>0.070_ ±_ 0.005|0_._029_ ±_ 0_._001<br>-<br>0_._029_ ±_ 0_._003<br>-<br>-<br>0_._024_ ±_ 0_._003<br>-<br>-<br>0_._026_ ±_ 0_._001<br>**0.022**_ ±_** 0.001**|
|Subgraphormer (Bar-Shalom et al., 2024)<br>FragNet (Wollschlager et al., 2024)<br>GatedGCN-LSPE (Dwivedi et al., 2022c)<br>GRIT (Ma et al., 2023)<br>GraphGPS (Ramp´aˇsek et al., 2022)<br>SAN (Kreuzer et al., 2021)<br>Graphormer (Kreuzer et al., 2021)<br>Graphormer-GD (Kreuzer et al., 2021)|**0.063**_ ±_** 0.001**<br>0.078_ ±_ 0.005<br>0_._090_ ±_ 0_._001<br>**0.059**_ ±_** 0.002**<br>0_._070_ ±_ 0_._004<br>0_._139_ ±_ 0_._006<br>0_._122_ ±_ 0_._006<br>0_._081_ ±_ 0_._009|**0.023**_ ±_** 0.001**<br>0.024<br>-<br>**0.023**_ ±_** 0.001**<br>-<br>-<br>0.052_ ±_0.005<br>0.025_ ±_0.004|
|GIST (ours)|**0.055**_ ±_** 0.002**|**0.019**_ ±_** 0.002**|

**5.4. MoleculeNet Benchmark**

To further evaluate the effectiveness of our proposed GIST
in molecular representation learning, we extend our exper

iments to the MoleculeNet benchmark (Wu et al., 2017).
MoleculeNet encompasses a diverse collection of graphbased molecular property prediction tasks, specifically designed to assess a model’s ability to capture chemical interactions, molecular toxicity, and bioactivity. These tasks span
a range of real-world applications, including drug discovery,
environmental toxicity assessment, and material science,
making MoleculeNet a comprehensive benchmark for evaluating graph-based learning approaches. As shown in Table
5, GIST consistently outperforms—or at least maintains
competitive performance against—existing state-of-the-art
pre-trained graph models and Graph Transformers across
multiple tasks. These results highlight GIST’s strong capability in molecular representation learning, demonstrating
that structural information can be effectively integrated into
Transformer-based architectures without the need for extensive pretraining, making it a promising approach for
molecular property prediction in low-data regimes.

**5.5. Ablation Study on different** _k−_ **hop**

Finally, to analyze the impact of different _k_ -hop neighborhood sizes in our proposed GIST, we conduct an **ablation**
**study** on the ZINC dataset. The value of _k_ influences how
much local and long-range information is incorporated into
the model. For **RQ3**, results from our ablation study on
the ZINC dataset (Table 4) indicate that GIST is robust to
variations in the maximum hop distance _k_ . While performance improves as _k_ increases from 1 to 3, capturing richer
structural dependencies, the fluctuations beyond _k_ = 3
remain minimal, suggesting that GIST maintains stability
across different neighborhood sizes. The slight decrease in
performance at higher _k_ is marginal, indicating that GIST
effectively balances local expressiveness and global aggregation without being overly sensitive to the choice of _k_ .

_Table 4._ Ablation study on different _k_ -hop neighborhood sizes in
GIST on the ZINC dataset.

|k-hop|1|2|3|4|5|
|---|---|---|---|---|---|
|MAE_ ↓_|0.100|0.058|0.054|0.065|0.063|

For **RQ1**, our competitive results in Tables 5, 2, and 3
show that GIST effectively facilitates the learning and differentiation of substructures in graph classification tasks by
encoding rich structural relationships through intersection
cardinality. This enables Graph Transformers to capture
fine-grained substructure information and complex substructure relationships, leading to improved performance.

**6. Related Works**

**Graph** **Substructures** **Modeling.** Modeling graph substructures is crucial for capturing fine-grained structural pat

7

385
386
387
388
389
390
391
392
393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
439

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

_Table 5._ Performance of GIST on MoleculeNet benchmark: Top-3 Results Highlighted in **Red**, **Blue**, and **Orange** .

|Model|BBBP|Tox21|Toxcast|Sider|Clintox|Bace|MUV|HIV|Avg. AUC|
|---|---|---|---|---|---|---|---|---|---|
|AttrMasking (Hu et al., 2020a)<br>GRIT (Ma et al., 2023)<br>GraphGPS (Ramp´aˇsek et al., 2022)<br>GraphLoG (Xu et al., 2021)<br>GraphCL (You et al., 2020)<br>G-Motif (Rong et al., 2020)<br>G-Contextual (Rong et al., 2020)<br>GPT-GNN (Hu et al., 2020b)<br>GraphFP (Luong & Singh, 2023)<br>MGSSL (Zhang et al., 2021)<br>GraphMVP (Liu et al., 2022)|64.3_ ±_ 2.8<br>**69.9**_ ±_** 1.3**<br>56.2_ ±_ 4.4<br>67.8_ ±_ 1.9<br>69.7_ ±_ 0.7<br>66.9_ ±_ 3.1<br>69.2_ ±_ 3.0<br>64.5_ ±_ 1.4<br>**72.0**_ ±_** 1.7**<br>68.9_ ±_ 2.5<br>68.5_ ±_ 0.2|**76.7**_ ±_** 0.4**<br>**75.9**_ ±_** 0.6**<br>71.4_ ±_ 0.7<br>75.1_ ±_ 1.0<br>73.9_ ±_ 0.7<br>73.6_ ±_ 0.7<br>75.0_ ±_ 0.6<br>74.9_ ±_ 0.3<br>74.0_ ±_ 0.7<br>74.9_ ±_ 0.6<br>74.5_ ±_ 0.4|**64.2**_ ±_** 0.5**<br>**65.6**_ ±_** 0.4**<br>60.6_ ±_ 1.0<br>62.4_ ±_ 0.2<br>62.4_ ±_ 0.6<br>62.3_ ±_ 0.6<br>62.8_ ±_ 0.7<br>62.5_ ±_ 0.4<br>63.9_ ±_ 0.9<br>63.3_ ±_ 0.5<br>62.7_ ±_ 0.1|61.0_ ±_ 0.7<br>60.3_ ±_ 1.2<br>60.2_ ±_ 1.1<br>59.5_ ±_ 1.5<br>60.5_ ±_ 0.9<br>61.0_ ±_ 1.5<br>58.7_ ±_ 1.0<br>58.1_ ±_ 0.3<br>**63.6**_ ±_** 1.2**<br>57.7_ ±_ 0.7<br>**62.3**_ ±_** 1.6**|71.8_ ±_ 4.1<br>**85.9**_ ±_** 2.9**<br>79.2_ ±_ 3.6<br>65.3_ ±_ 3.2<br>76.0_ ±_ 2.7<br>77.7_ ±_ 2.7<br>60.6_ ±_ 5.2<br>58.3_ ±_ 5.2<br>**84.7**_ ±_** 5.8**<br>67.5_ ±_ 5.5<br>79.0_ ±_ 2.5|79.3_ ±_ 1.6<br>**84.4**_ ±_** 1.2**<br>71.5_ ±_ 6.0<br>80.2_ ±_ 3.5<br>75.4_ ±_ 1.4<br>73.0_ ±_ 3.3<br>79.3_ ±_ 1.1<br>77.9_ ±_ 3.2<br>80.5_ ±_ 1.8<br>**82.1**_ ±_** 2.7**<br>76.8_ ±_ 1.1|74.7_ ±_ 1.4<br>**77.1**_ ±_** 1.7**<br>65.2_ ±_ 1.6<br>73.6_ ±_ 1.2<br>69.8_ ±_ 2.7<br>73.0_ ±_ 1.8<br>72.1_ ±_ 0.7<br>**75.9**_ ±_** 2.3**<br>75.4_ ±_ 1.9<br>73.2_ ±_ 1.9<br>75.0_ ±_ 1.4|77.2_ ±_ 1.1<br>**77.3**_ ±_** 1.5**<br>66.0_ ±_ 9.4<br>73.7_ ±_ 0.9<br>**78.5**_ ±_** 1.2**<br>73.8_ ±_ 1.2<br>76.3_ ±_ 1.5<br>65.2_ ±_ 2.1<br>**78.0**_ ±_** 1.5**<br>75.7_ ±_ 1.3<br>74.8_ ±_ 1.4|71.2<br>**74.8**<br>66.3<br>69.7<br>70.8<br>70.2<br>69.3<br>67.2<br>**74.0**<br>70.4<br>71.7|
|GIST (ours)|**70.6**_ ±_** 1.8**|**77.2**_ ±_** 0.4**|**67.3**_ ±_** 0.9**|**61.3**_ ±_** 2.7**|**88.2**_ ±_** 2.2**|**86.0**_ ±_** 1.9**|**75.5**_ ±_** 3.2**|77.0_ ±_ 0.2|**75.4**|

terns and improving representation learning in graph-based
tasks. However, GNNs remain fundamentally constrained
by their reliance on localized message passing, which limits their ability to capture long-range dependencies and
effectively model complex substructure interactions, due
to over-smoothing and over-squashing issues (Xu et al.,
2018; Alon & Yahav, 2021). To address this, later works
have introduced spectral features (Balcilar et al., 2021),
motif-based methods (Rong et al., 2020; Zhang et al., 2021;
Bar-Shalom et al., 2024; Wollschlager et al., 2024), and
Weisfeiler-Lehman (WL) kernel-based approaches (Morris
et al., 2019) to improve graph representation learning by explicitly capturing local and global structural patterns. While
motif-based methods improve expressivity by incorporating recurring substructures, they often depend on predefined
motifs, restricting their adaptability to unseen graph patterns.
Similarly, WL kernel-based approaches enhance structural
discrimination but struggle with distinguishing graphs that
are structurally different yet WL-equivalent. Furthermore,
spectral features capture global graph properties but introduce additional computational complexity, making them
less practical for large-scale applications. These limitations
underscore the need for alternative architectures that can
more effectively integrate structural biases while maintaining both scalability and expressiveness in graph learning.

**Graph** **Transformers.** Transformers have demonstrated
remarkable success in natural language processing and computer vision by leveraging self-attention to model long-range
dependencies effectively (Vaswani et al., 2017). More recently, their adaptation to graph-structured data has led to
the emergence of Graph Transformers, where self-attention
replaces traditional message-passing mechanisms to enable
more flexible and expressive learning (Zhang et al., 2020;
Dwivedi & Bresson, 2021). However, a fundamental challenge in applying Transformers to graphs is the absence of
a natural node ordering, making it difficult to encode structural information directly. To address this, positional encodings have been introduced to assign meaningful node representations within the graph topology. Among these, Lapla

cian eigenvector-based encodings (LapPE) (Dwivedi et al.,
2022a) and random walk positional encodings (RWPE)
(Dwivedi et al., 2022b) inject global structural awareness,
enhancing the model’s ability to differentiate nodes with
similar local neighborhoods. Beyond positional encodings,
researchers have explored incorporating structural biases
into self-attention to ensure that Graph Transformers respect
the underlying graph topology. GPS (Rampa´sek et al.ˇ, 2022)
combines message passing with attention, allowing models
to capture both local and global dependencies within the
graph. More recently, GRIT (Ma et al., 2023) introduced a
fully Transformer-based framework that eliminates explicit
message passing while embedding structure-aware attention, achieving state-of-the-art performance across multiple
graph learning benchmarks. These advancements reflect a
growing shift toward pure Transformer architectures that effectively incorporate graph-specific inductive biases, paving
the way for more scalable and expressive models in graph
representation learning.

**7. Conclusion**

This paper introduces the Graph Invariant Structural Trait
(GIST) to enhance Graph Transformers by improving their
ability to encode graph structures. GIST estimates pairwise
node intersections to capture substructures within a graph,
integrating this information into the attention mechanism.
This refinement enables Graph Transformers to better represent structural relationships that traditional all-to-all attention struggles to capture. Theoretical analysis and empirical
results confirm that GIST effectively preserves essential
structural information critical for graph classification. Extensive experiments across multiple datasets demonstrate
that incorporating GIST into Graph Transformers consistently improves performance over state-of-the-art methods.
These findings highlight the importance of structural encoding in enhancing Graph Transformers, contributing to more
robust and interpretable graph-based learning models across
scientific domains.

8

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., and Bresson, X. Graph neural networks with learnable structural
and positional representations. In _The Eleventh Interna-_
_tional Conference on Learning Representations_, 2022b.

Dwivedi, V. P., Rampa´sek, L., Galkin, M., Parviz, A., Wolf,ˇ
G., Luu, A. T., and Beaini, D. Recipe for a general,
powerful, scalable graph transformer. In _36th Conference_
_on Neural Information Processing Systems_, 2022c.

Frasca, F., Bevilacqua, B., Bronstein, M., and Maron, H. Understanding and extending subgraph gnns by rethinking
their symmetries. In _Advances in Neural Information Pro-_
_cessing Systems (NeurIPS)_, volume 35, pp. 31376–31390,
2022.

Han, X., Jiang, Z., Liu, N., and Hu, X. G-mixup: Graph
data augmentation for graph classification. In _Interna-_
_tional Conference on Machine Learning_, pp. 8230–8248.
PMLR, 2022.

Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V.,
and Leskovec, J. Strategies for pre-training graph neural networks. In _International Conference on Learning_
_Representations (ICLR)_, 2020a.

Hu, Z., Dong, Y., Wang, K., Chang, K.-W., and Sun, Y. Gptgnn: Generative pre-training of graph neural networks.
In _Proceedings of the 26th ACM SIGKDD International_
_Conference_ _on_ _Knowledge_ _Discovery_ _&_ _Data_ _Mining_
_(KDD)_, 2020b.

Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., and
Coleman, R. G. Zinc: a free tool to discover chemistry
for biology. In _Journal_ _of_ _Chemical_ _Information_ _and_
_Modeling_, 2012.

Keriven, N. Not too little, not too much: a theoretical
analysis of graph (over) smoothing. _Advances in Neural_
_Information Processing Systems_, 35:2268–2281, 2022.

Kipf, T. N. and Welling, M. Semi-supervised classification
with graph convolutional networks. In _5th International_
_Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2017,_
_Toulon,_ _France,_ _April_ _24-26,_ _2017,_ _Conference_ _Track_
_Proceedings_ . OpenReview.net, 2017. [URL https://](https://openreview.net/forum?id=SJU4ayYgl)
[openreview.net/forum?id=SJU4ayYgl.](https://openreview.net/forum?id=SJU4ayYgl)

Kreuzer, D., Beaini, D., Hamilton, W. L., Letourneau, V.,
and Tossou, P. Rethinking graph transformers with spectral attention. In _35th Conference on Neural Information_
_Processing Systems_, 2021.

Le, D., Zhong, S. H., Liu, Z., Xu, S., Chaudhary, V., Zhou,
K., and Xu, Z. Knowledge graphs can be learned with
just intersection features. In _The Forty-first International_
_Conference on Machine Learning_, 2024.

440
441
442
443
444
445
446
447
448
449
450
451
452
453
454
455
456
457
458
459
460
461
462
463
464
465
466
467
468
469
470
471
472
473
474
475
476
477
478
479
480
481
482
483
484
485
486
487
488
489
490
491
492
493
494

**Impact Statement**

This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none of which we feel must be
specifically highlighted here.

**References**

Alon, U. and Yahav, E. On the bottleneck of graph neural
networks and its practical implications. In _The_ _Tenth_
_International Conference on Learning Representations_,
2021.

Balcilar, M., Heroux,´ P., Gauz¨ ere,` B., Vasseur, P., Adam,
S., and Honeine, P. Breaking the limits of message passing graph neural networks. In _The_ _38th_ _International_
_Conference on Machine Learning_, 2021.

Bar-Shalom, G., Bevilacqua, B., and Maron, H. Subgraphormer: Unifying subgraph gnns and graph transformers via graph products. In _The Forty-first Interna-_
_tional Conference on Machine Learning_, 2024.

Bevilacqua, B., Frasca, F., Lim, D., Srinivasan, B., Cai,
C., Balamurugan, G., Bronstein, M. M., and Maron, H.
Equivariant subgraph aggregation networks. In _Interna-_
_tional Conference on Learning Representations (ICLR)_,
2022.

Bevilacqua, B., Eliasof, M., Meirom, E., Ribeiro, B., and
Maron, H. Efficient subgraph gnns by learning effective selection policies. In _International Conference on_
_Learning Representations (ICLR)_, 2023.

Black, M., Wan, Z., Nayyeri, A., and Wang, Y. Understanding oversquashing in gnns through the lens of effective resistance. In _International Conference on Machine_
_Learning_, pp. 2528–2547. PMLR, 2023.

Brody, S., Alon, U., and Yahav, E. How attentive are graph
attention networks? In _The Eleventh International Con-_
_ference on Learning Representations_, 2022.

Chamberlain, B. P., Shirobokov, S., Rossi, E., Frasca, F.,
Markovich, T., Hammerla, N. Y., Bronstein, M. M., and
Hansmire, M. Graph neural networks for link prediction
with subgraph sketching. In _The Eleventh International_
_Conference on Learning Representations_, 2022.

Dwivedi, V. P. and Bresson, X. A generalization of transformer networks to graphs. In _Proceedings of the AAAI_
_Conference on Artificial Intelligence_, 2021.

Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio,
Y., and Bresson, X. Benchmarking graph neural networks.
In _Journal of Machine Learning Research_, 2022a.

9

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

495
496
497
498
499
500
501
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525
526
527
528
529
530
531
532
533
534
535
536
537
538
539
540
541
542
543
544
545
546
547
548
549

Liu, S., Wang, H., Liu, W., Lasenby, J., Guo, H., and Tang,
J. Pre-training molecular graph representation with 3d
geometry. In _The Eleventh International Conference on_
_Learning Representations_, 2022.

Luong, K.-D. and Singh, A. Fragment-based pretraining
and finetuning on molecular graphs. In _37th Conference_
_on Neural Information Processing Systems_, 2023.

Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania,
P. K., Coates, M., Torr, P. H., and Lim, S.-N. Graph
inductive biases in transformers without message passing.
In _The_ _Fortieth_ _International_ _Conference_ _on_ _Machine_
_Learning_, 2023.

Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen,
J. E., Rattan, G., and Grohe, M. Weisfeiler and leman go neural: Higher-order graph neural networks.
In _Proceedings_ _of_ _the_ _33rd_ _AAAI_ _Conference_ _on_ _Arti-_
_ficial_ _Intelligence_ _(AAAI)_, pp. 4602–4609, 2019. URL
[https://arxiv.org/abs/1810.02244.](https://arxiv.org/abs/1810.02244)

Qian, C., Rattan, G., Geerts, F., Niepert, M., and Morris,
C. Ordered subgraph aggregation networks. In _Advances_
_in_ _Neural_ _Information_ _Processing_ _Systems_ _(NeurIPS)_,
volume 35, pp. 21030–21045, 2022.

Rampa´sek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf,ˇ
G., and Beaini, D. Recipe for a general, powerful, scalable graph transformer. In _36th Conference on Neural_
_Information Processing Systems_, 2022.

Rong, Y., Bian, Y., Xu, T., Xie, W., Wei, Y., Huang, W.,
and HUang, J. Self-supervised graph transformer on
large-scale molecular data. In _34th Conference on Neural_
_Information Processing Systems_, 2020.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention
is all you need. In _31th Conference on Neural Information_
_Processing Systems_, 2017.

Wang, Y. and Zhang, M. An empirical study of realized gnn
expressiveness. In _Forty-first International Conference_
_on Machine Learning_, 2024.

Wollschlager, T., Kemper, N., Hetzel, L., Sommer, J., and
Gunneman, S. Expressivity and generalization: Fragmentbiases for molecular gnns. In _The Forty-first International_
_Conference on Machine Learning_, 2024.

Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., and Pande, V.
Moleculenet: A benchmark for molecular machine learning. In _Chemical Science_, 2017.

Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful
are graph neural networks? In _The Seventh International_
_Conference on Learning Representations_, 2018.

Xu, M., Wang, H., Ni, B., Guo, m. H., and Tang, J. Selfsupervised graph-level representation learning with local
and global structure. In _The 38th International Confer-_
_ence on Machine Learning_, 2021.

Yang, C., Liu, M., Zheng, V. W., and Han, J. Node, motif and subgraph: Leveraging network functional blocks
through structural convolution. In _International Confer-_
_ence on Advances in Social Network Analysis and Mining_,
2018.

Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen,
Y., and Liu, T.-Y. Do transformers really perform bad
for graph representation? In _35th Conference on Neural_
_Information Processing Systems_, 2021.

You, Y., Chen, T., Shen, Y., and Wang, Z. Graph contrastive
learning with augmentations. In _34th_ _Conference_ _on_
_Neural Information Processing Systems_, 2020.

Yu, Z. and Gao, H. Molecular representation learning via
heterogeneous motif graph neural networks. In _Proceed-_
_ings_ _of_ _the_ _39th_ _International_ _Conference_ _on_ _Machine_
_Learning_, 2022.

Zhang, B., Feng, G., Du, Y., He, D., and Wang, L. A
complete expressiveness hierarchy for subgraph gnns via
subgraph weisfeiler-lehman tests. In _International Con-_
_ference on Machine Learning (ICML)_, 2023a.

Zhang, B., Luo, S., Wang, L., and He, D. Rethinking
the expressive power of gnns via graph biconnectivity.
In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_
_Representations_, 2023b.

Zhang, M. and Li, P. Nested graph neural networks.
In _Advances in Neural Information Processing Systems_
_(NeurIPS)_, volume 34, 2021.

Zhang, Z., Cui, P., and Zhu, W. Graph-bert: Only attention is
needed for learning graph representations. _arXiv preprint_
_arXiv:2001.05140_, 2020.

Zhang, Z., Liu, Q., Wang, H., Lu, C., and Lee, C.-K. Motifbased graph self-supervised learning for molecular property prediction. In _35th Conference on Neural Informa-_
_tion Processing Systems_, 2021.

Zhao, L., Jin, W., Akoglu, L., and Shah, N. From stars to
subgraphs: Uplifting any gnn with local structure awareness. In _International Conference on Learning Represen-_
_tations (ICLR)_, 2022.

10

550
551
552
553
554
555
556
557
558
559
560
561
562
563
564
565
566
567
568
569
570
571
572
573
574
575
576
577
578
579
580
581
582
583
584
585
586
587
588
589
590
591
592
593
594
595
596
597
598
599
600
601
602
603
604

**Graph Transformers Get the GIST: Graph Invariant Structural Traits for Refned Graph Encoding**

**A. Proofs**

**Theorem A.1** (Formal version of Theorem 4.2) **.** _Let G_ = ( _V, E_ ) _denote a graph with n nodes (|V|_ = _n)._ _Let S_ ( _G_ ) _∈_ _denote_
_the k-hop GIST (see Definition 3.2) computed on G._ _We show that the GIST attention ψ_ ( _xu_ ) _for every node u_ _∈V_ _(see_
_Definition 4.1) is invariant under graph isomorphism._

_Proof._ Let _f_ denote isomorphic transform on nodes _V_ such that if _u_ and _v_ are adjacent in _G_, _f_ ( _u_ ) and _f_ ( _v_ ) are also adjacent.
Without loss of generally, we see that _Cku,kv_ ( _f_ ( _u_ ) _, f_ ( _v_ )) = _Cku,kv_ ( _u, v_ ).

Following Definition 3.1, we show that _Iku,kv_ ( _f_ ( _u_ ) _, f_ ( _v_ )) = _Iku,kv_ ( _u, v_ ) _, Tku,kv_ ( _f_ ( _u_ ) _, f_ ( _v_ )) = _Tku,kv_ ( _u, v_ ).

As a result, we show that _Sf_ ( _u_ ) _,f_ ( _v_ )( _f_ ( _G_ )) = _Su,v_ ( _f_ ( _G_ )).

Following Definition 4.1, since the order of node _v_ does not affect the computation of _ψ_ ( _xu_ ), we show that _ψ_ ( _xf_ ( _u_ )) =
_ψ_ ( _xu_ ).

As a result, we show that the isomorphic transform _f_ does not change _ψ_ ( _xu_ ), making _ψ_ a graph invariant.

11

