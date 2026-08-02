# Query-Aware Subgraph Packing: A Knapsack Optimization Paradigm For Graph Retrieval-Augmented Generation

| Anonymous Author(s) Affiliation Address email   |
|-------------------------------------------------|

## Abstract 16 **1 Introduction**

1 Graph Retrieval-Augmented Generation (GraphRAG) has recently emerged as a 2 task paradigm for injecting graph-structured knowledge into large language models 3 (LLMs), yet most existing approaches still rely on flat, similarity-based retrieval that 4 ignores topology and uses static encoders, producing redundant or structurally inco5 herent evidence. In this paper, we propose GraphPack, a query-aware GraphRAG 6 framework that overcomes these limitations by casting subgraph selection as a 0–1 7 knapsack optimization. For every natural language query, GraphPack packs the 8 most informative subgraph under a size budget by jointly maximizing semantic 9 relevance and minimizing structural redundancy. The selected subgraph is then 10 encoded by a query-aware graph encoder whose parameters are conditioned on the 11 query, allowing node representations to adapt dynamically to user intent. Extensive 12 experiments on multiple knowledge-intensive graph benchmarks demonstrate that 13 GraphPack achieves state-of-the-art performance, showcasing its strong capabil14 ity in addressing structural and contextual challenges under supervised learning, 15 cross-domain settings, and zero-shot scenarios. 17 Graph-structured data plays a central role in real-world applications such as recommendation systems 18 [He et al., 2020], social network analysis [Huang et al., 2024], and knowledge-intensive reasoning 19 tasks [Fu et al., 2020, Lan et al., 2021]. Large language models (LLMs) have demonstrated impressive 20 capabilities in natural language understanding and generation. However, their ability to effectively 21 integrate structured knowledge and user intent remains limited, leading to suboptimal performance 22 on tasks such as query-focused summarization (QFS). A key challenge lies in retrieving and encoding 23 task-relevant entities from large-scale textual graphs in a manner that aligns with the user's intent. 24 Graph Retrieval-Augmented Generation (GraphRAG) [Edge et al., 2025] has emerged as an innovative 25 solution to address the challenges of integrating structured knowledge into LLMs. Unlike traditional 26 retrieval-augmented generation (RAG) [Lewis et al., 2020, Guu et al., 2020, Ram et al., 2023, Izacard 27 et al., 2022], which primarily operates over flat textual corpora, GraphRAG retrieves graph elements 28 - such as nodes, triples, paths, or subgraphs - that are semantically relevant to a given query 29 from a pre-constructed graph database. These retrieved elements provide rich relational knowledge 30 that enhances both the depth and accuracy of LLM-based reasoning. By retrieving subgraphs or 31 graph communities, GraphRAG enables comprehensive understanding of the underlying knowledge 32 structure, making it particularly effective in tasks such as query-focused summarization, where 33 concise yet informative responses must align closely with user intent.

Description: 
Question: What category does the following paper belong to?

Description: Attention Is All You Need \n Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. …
Question: Where does fabio capello come from?

Question: What's the deal between Ronaldo and Real Madrid?

Description: 
0,football \n3,male \n4,football player \n8,catholicism \n10,san canzian d'isonzo \n20,italy \n22,fabio capello …
Ronaldo Luis Nazario de Lima description: Brazilian footballer 22,common.topic.notable_types,4 10,location.location.people_born_here,22 22,sports.sports_team_coach.sports_coached,0 22,people.person.nationality,20 
…
Real Madrid Club description: Spanish association football club fabio capello football player Citation graph:

Knowledge graph:

football Knowledge graph:

Attention Is All You Need Segment anythingThe llama 3 herd of models Gemini: A Family of Highly Mamba: Linear Capable Multimodal Models -Time Sequence Modeling with Selective State Spaces male san canzian d'isonzo catholicism italy notable_types sports_coached people_born

_here person.religion person.nationality person.gender Real Madridfootball_player team Ronaldo containedby LLMs LLMs LLMs Answer: The paper "Attention Is All You Need" is primarily categorized under Computation and Language 
(cs.CL) due to its core contribution to the field of Natural Language Processing through the introduction of the Transformer architecture.

Answer: Based on the provided knowledge graph, fabio capello come from San Canzian d'Isonzo. a town in the province of Gorizia, located in the Friuli Venezia Giulia region of northeastern Italy.

Answer: Ronaldo is a former Real Madrid player. He joined the club from Inter Milan in 2002 and played there for five seasons (2002–2007).
34 A key challenge in applying LLMs to graph-structured data lies in designing retrieval mechanisms 35 that are not only semantically informative but also adaptable across diverse graph tasks. As shown in 36 figure 1, Knowledge-intensive tasks such as multi-hop question answering require global structural 37 reasoning, demanding the model to identify and integrate information from semantically related, 38 yet topologically distant entities. A major limitation of current graph-augmented LLMs lies in their 39 reliance on similarity-based retrieval mechanisms, which often neglect the rich topological structure 40 embedded in the graph. For example, GRAG [Hu et al., 2025] re-ranks candidate subgraphs based 41 on both their relational alignment with the query and fine-grained concept-level similarity. KELP 42 [Liu et al., 2024] trains a pretrained language model to score the relevance between retrieved paths 43 and input queries. While these methods perform well at identifying nodes or subgraphs that are 44 semantically close to a given query, they tend to treat the graph as a flat collection of textual elements, 45 neglecting the relational patterns that define its underlying structure. 46 To address this issue, we propose GraphPack, a novel framework for query-aware graph retrieval47 augmented generation. Specifically, we formulate subgraph packing as a 0-1 knapsack problem, 48 allowing the model to dynamically identify query-relevant regions of the graph by jointly considering 49 semantic relevance and structural cost. We further introduce Query-LM, a graph encoder with 50 query-aware capabilities that enhances node representations through conditional linear modulation 51 modules. This enables the model to adaptively adjust node embeddings based on the input query, 52 leading to more accurate and context-sensitive graph encoding. Additionally, we design an auxil53 iary graph-to-text reconstruction objective. This training signal improves the expressiveness and 54 interpretability of graph embeddings without requiring any architectural changes - making our 55 approach both general and practical. Our method goes beyond traditional GraphRAG frameworks 56 by explicitly modeling what the user is asking and how the graph structure should respond. This 57 leads to a more principled integration of structured knowledge into the language generation process. 58 Extensive experiments demonstrate that GraphLLM achieves strong performance across multiple 59 graph benchmarks, highlighting its effectiveness in bridging structured knowledge with LLMs for 60 downstream applications.

## 61 **2 Method** 62 **2.1 Large Language Model For Graph**

63 GraphLLM aims to effectively incorporate graph-structured contextual information into both the 64 retrieval and generation stages, thereby enhancing the relevance between the generated outputs and the 65 textual graph knowledge. Specifically, given a user query xq and a textual graph G = (V, E, Xv, Xe),
66 we expect GraphLLM to generate answers that are aligned with the intended semantics of the query.

l-th Graph Encoder Text Input node_id,node_attr 9,lionel messi 48,argentina national football team 125,fc barcelona
…
src,edge_attr,dst 256, player, 9 125, football_team ,256 … Query: What team does Messi play for 2011? Answer:

barcelona argentina national | fc barcelona Knapsack Problem fc trainable Query: What team does Messi play for 2011?

frozen Value =
leo messi 5 Weight = 
football_team 2 LLMs decoder Semantic-based Retrieval football_team 4 lfp bestforward 1 football_player 1 lfp bestforward 2 Node-level Readouts lionel messi
…
…
n-hop Subgraph Optimization Goal:
awards_wonm.010nx8b0 award_wi nner lfp best forward m.0j_8pbk award_honor Query-LM
selected elements leo messi best ibero-american soccer player award_
honor leo messi football_team football_team football_player
…
Graph Transformer lfp bestforward
…
leo messi argentina national football team football_team m.0w9d7mr football_player footbal l_playe r football_t eam football_team q Subgraph Construct argentina national football team fc barcelona football_
team q LLMEmbed m.0w8wsdx fc barcelona Query football_player football_player m.0w8wsdx m.0w9d7mr
67 However, real-world graphs can be large in scale and contain substantial amounts of irrelevant or 68 redundant information. Directly feeding the entire graph into the model is not only computationally 69 expensive but may also lead to generated outputs that deviate from the user's actual intent. To 70 address this challenge, we emphasize the integration of a subgraph retrieval mechanism in the design 71 of GraphLLM, ensuring that the model can leverage the rich semantic information present in the 72 graph while remaining highly sensitive to the specific query intent during the generation process. we 73 formally define the generation process of GraphLLM under the graph-augmented retrieval mechanism.

74 Given a user query xq and the original textual graph G, the model first retrieves the most relevant subgraph G
∗
75 with respect to the query through a retrieval mechanism:

$${\mathcal{G}}^{*}={\mathrm{Retrieval}}(x_{q},{\mathcal{G}})$$
∗ = Retrieval(xq, G) (1)
76 We model GraphLLM with graph-retrieval-augmented generation as a likelihood-based model that 77 defines the probability of generating a query-related answer y:

$$p(y\mid x_{q},{\mathcal{G}}^{*})=\prod_{l=1}^{L}p(y_{l}\mid y_{<l},x_{q},{\mathcal{G}}^{*})$$
$$(\mathbf{l})$$
$$\left(2\right)$$

78 where yl denotes the l-th element in the output sequence, and y<l represents the first l-1 generated words. G
∗
79 contains both the structural and textual information of the graph, which assists the model 80 in generating y. This modeling approach not only preserves the topological information of the graph 81 structure but also enables joint modeling of context and query intent, encouraging the model to 82 develop strong capabilities in understanding and utilizing graph-structured knowledge.

## 83 **2.2 Semantic-Aware Subgraph Retrieval Via Knapsack Optimization**

84 **Graph Indexing** We adopt a retrieval approach similar to RAG to efficiently retrieve subgraphs 85 relevant to user needs from large textual graphs. Specifically, we use a frozen text encoder such as 86 sentence-bert [Reimers and Gurevych, 2019] to map various types of text into a unified vector space:
zv = TextEncoder(xv) ∈ R
dLM , ze = TextEncoder(xe) ∈ R
dLM (3)
87 Here, zv and ze denote the embeddings of the node and edge. dLM represents the dimension of the 88 pretrained language model. To enable efficient graph retrieval, we precompute the textual embeddings 89 of the graph for subsequent use.

$\eqref{eq:walpha}$
90 **Anchor Node Identification** Traditional graph retrieval methods often struggle to balance semantic 91 relevance with structural coherence, especially in large and complex graphs. A promising approach is 92 to first identify a small set of semantically relevant nodes as anchor point and then expand the search 93 within their local neighborhoods. This two-step strategy not only addresses computational challenges 94 but also introduces a novel way to harmonize semantic alignment with topological connectivity. We 95 process the user's question in the same manner as the textual information of the graph to obtain the 96 embedding zq.

V*anchor* = argtopkn∈V cos(zq, zn) (4)
97 We use the cosine similarity function cos(·, ·) to measure the similarity between the question rep98 resentation and the node representations. The argtopk operation retrieves the top-k nodes with the 99 highest similarity scores, which are then selected as anchor nodes.

100 **Knapsack Optimization** We model subgraph packing as a 0-1 knapsack problem [Freville, 2004], 101 integrating both semantic relevance and structural redundancy into the subgraph retrieval frame102 work. Our method dynamically balances the value of each graph element (node or edge) against 103 its construction cost, aiming to achieve a trade-off between accuracy and efficiency in subgraph 104 construction. 105 Formally, we model the subgraph retrieval task as a 0-1 knapsack problem. For an n-hop subgraph
g
in = (V
′, E
′) rooted at an anchor node v
i
106 a ∈ V*anchor*, each graph element is treated as an element e
107 in the knapsack formulation. A value function value(e) measures the semantic relevance of e, while a 108 weight function weight(e) quantifies its structural cost. The goal is to maximize the total value of
109 selected items under a capacity constraint C:
$${\mathrm{for~a~capacity~constraint}}\,{\mathcal{C}}\colon$$ $${\mathrm{arg~max}}\sum_{\mathbf{e}\in S}{\mathrm{value}}(\mathbf{e}),{\mathrm{s.t.}}\sum_{\mathbf{e}\in S}{\mathrm{weight}}(\mathbf{e})\leq{\mathcal{C}},S\subseteq{\mathcal{V}}\cup{\mathcal{E}}^{\prime}$$
weight(e) ≤ C, S ⊆ V′ ∪ E′(5)
116 110 **Rank-Based Value Assignment** To evaluate semantic relevance, we introduce a ranking-based 111 decaying value mechanism. We first sort all elements in descending order based on their semantic 112 relevance scores and assign each element a rank(e). The value of each element is then computed as 113 followed:
value(e) = max_score − rank(e) (6)
114 This design ensures that elements with higher semantic relevance within the local subgraph re115 ceive higher value scores, and are therefore prioritized for inclusion in the final subgraph. 117 Structure-Aware Weight Assign118 **ment** In terms of measuring struc119 tural cost, we adopt a structure-aware 120 weighting mechanism to suppress re121 dundancy. For each element e, the 122 weight is determined by the smallest 123 n-hop subgraph in which it appears 124 - in other words, the minimum hop 125 level at which the element is first en126 countered:
weight(e) = min{n | e ∈ g i n} (7)
127 This means that nearby elements (e.g., 128 those within 1-hop) are assigned lower 129 weights, while incorporating distant 130 elements (e.g., those beyond 3-hops) 131 incurs a higher cost. In this way, the 132 inclusion of remote and potentially re133 dundant elements - which may con134 tribute little semantic value but sig135 nificantly increase structural complex136 ity - is effectively discouraged. This 137 leads to the construction of more com138 pact and effective subgraphs. We use an efficient dynamic programming Algorithm 1 to solve the 139 subgraph optimization problem. Finally, we use the query embedding as a prompt node to connect 140 all retrieved elements and construct a coherent subgraph. We present discussions on the algorithm 141 implementation in Appendix A.

$$({\boldsymbol{S}})$$

Algorithm 1 Dynamic Programming for 0-1 Knapsack Problem Input: Values v[1..n], Weights w[1..n], Capacity C Output: Selected items maximizing total value within C Initialize A ← array of (n + 1) × (C + 1) with 0 Initialize *keep* ← boolean array of (n + 1) × (C + 1) with False for i = 1 to n do for c = 0 to C do if w[i] ≤ c and v[i]+A[i−1][c−w[i]] > A[i−1][c]
then A[i][c] ← v[i] + A[i − 1][c − w[i]] keep[i][c] ← True else A[i][c] ← A[i − 1][c]
Initialize S ← [], c ← C for i = n *downto* 1 do if *keep*[i][c] **then**
Append i to S c ← c − w[i]
return S

## 142 **2.3 Query-Aware Graph Encoder**

143 We employ a graph neural network to encode the topological structure of the retrieved subgraph. 144 However, traditional GNNs rely solely on local neighborhood topology and edge attributes for 145 message passing and feature aggregation. As a result, they lack the ability to dynamically adjust their 146 modeling focus based on the input query - a critical limitation in knowledge-intensive question 147 answering tasks that require identifying task-specific paths or substructures. 148 To address this issue, we propose a query-aware graph encoder, which introduces conditional 149 modulation into the GNN architecture through FiLM-style transformations. we perform multi-layer GNN message passing over the retrieved subgraph G
∗
150 . At each layer, node representations are 151 updated by aggregating information from their neighbors, preserving contextual relationships within 152 the graph structure. Formally, the output of the l-th GNN layer is given by:

$$\bar{h}_{v}^{(l)}=\mathrm{GNN}^{(l)}\left(\mathbf{h}_{v}^{(l-1)},\left\{\left(\mathbf{h}_{u}^{(l-1)},\mathbf{e}_{u v}\right)\mid u\in{\mathcal{N}}(v)\right\}\right)$$
$$({\mathfrak{s}})$$
$$(9)$$
o (8)
153 where N (v) denotes the neighborhood of node v in the retrieved subgraph. To overcome the 154 limitations of traditional GNNs in static modeling, inspired by the FiLM [Perez et al., 2017], we 155 introduce the Query-aware Linear Modulation (Query-LM), which serves as a conditional control 156 mechanism within the GNN message passing process. Specifically, we encode the natural language 157 question into a vector representation:
hq = Pooling (LLMEmbedded(xq)) (9)
158 which serves as a guiding signal for the subsequent graph encoding process. This allows the model to 159 adaptively steer feature learning according to the specific requirements of the given task. We then 160 define the Query-FiLM module at each layer as follows:

$${\mathrm{abcdded}}(x_{q}))$$
$h_q=\mathbf{P}$. 
$${\mathrm{coloring~}}({\mathrm{LLMEm}}$$
$$\begin{array}{c}{{\gamma_{j}^{(l)}=\sigma\left(\mathbf{W}_{\gamma_{1}}^{(l)}\cdot h_{q}+\mathbf{b}_{\gamma_{1}}^{(l)}\right),\quad\beta_{j}^{(l)}=\sigma\left(\mathbf{W}_{\beta_{1}}^{(l)}\cdot h_{q}+\mathbf{b}_{\beta_{1}}^{(l)}\right)}}\\ {{h_{v}^{(l)}=\gamma_{v}^{(l)}\odot\bar{h}_{v}^{(l)}+\beta_{v}^{(l)}}}\end{array}$$
$$(10)$$

$$(11)$$
(10)
$$1{\mathfrak{f}}{\mathfrak{h}}1$$

162 where ⊙ denotes the Hadamard product, and σ represents an activation function. Query-FiLM uses the query embedding hq to generate the affine transformation parameters γ
(l)
jand β
(l) j 163 , which are then applied to scale and shift the intermediate node representations h˜
(l)
164 v output by the GNN in a channel-wise manner, resulting in the updated node representations h
(l)
165 v . Through the Query-FiLM,
166 the model translates the semantics of the natural language query into explicit modulation signals 167 over the GNN feature space, enabling the acquisition of query-aware graph representations while 168 preserving the original capability to model graph structure. 169 Then we use a graph readout method based on node-level nonlinear transformations. We obtain the 170 final graph-level representation by applying average pooling to the transformed embeddings of all 171 nodes:

$$h_{g}=\frac{1}{|{\cal V}|}\sum_{v\in{\cal V}}\sigma({\bf W}_{1}h_{v}^{(L)}+{\bf b}_{1}){\bf W}_{2}+{\bf b}_{2}$$

172 Here, W1, W2 and b1, b2 denote the learnable weight matrices and bias terms. Before the node 173 embeddings are pooled into a graph-level representation, they are first mapped through independent 174 nonlinear transformations. This enhances the expressive power of each node embedding while 175 maintaining geometric consistency with the LLM's textual semantic space.

## 176 **2.4 Llms Supervised Fine-Tuning**

177 During the supervised fine-tuning (SFT) phase, we use the original user query xq and the textual 178 description of the subgraph xg as the initial input to the decoder. The graph representation hg is 179 concatenated with the embeddings of the input text to form the contextual representation for the 180 language model. For the target answer sequence y corresponding to the query, we optimize the 181 model parameters by maximizing the standard log-likelihood of the output sequence. This process 182 effectively learns the conditional probability distribution defined in Equation 1, enabling the model to 183 generate accurate and semantically coherent answers.

184 However, a challenge arises as the input length increases - the attention weights allocated to the 185 graph embedding inevitably decrease, leading to a potential loss of structural information [Ma et al.,

$$(12)$$

186 2024, Kong et al., 2025]. To address this issue, we design an auxiliary graph-to-text reconstruction 187 task . Specifically, we train the model to answer the user query only based on the abstracted graph 188 embedding, by maximizing the standard log-likelihood of the target answer sequence y.

189 The purpose of this auxiliary task is to enhance the invertibility and interpretability of the graph 190 embedding, ensuring that it not only captures the underlying graph structure effectively but also can 191 independently guide high-quality answer generation within the language model. Importantly, this 192 strategy does not require any modification to the model architecture itself; instead, it improves the 193 representational power of the graph embeddings purely through adjustments to the training objective, 194 making it both general and practical.

## 195 **3 Related Works**

196 Here, we mainly introduce the generation-based GraphLLM [Ren et al., 2024] and GRAG [Peng 197 et al., 2024]. The classification-based GraphLLM and its connection to graph neural networks will be 198 discussed in the Appendix B.

## 199 **3.1 Llms With Graphs**

200 Recent research has explored how to apply LLMs to tasks involving graph-structured data. One 201 intuitive approach is to serialize the textual graph into structured descriptions, which are then directly 202 fed into the LLMs for fine-tuning [Wang et al., 2024, Ye et al., 2024, Zhao et al., 2023, Fatemi et al., 203 2023, Tan et al., 2024]. These methods can leverage LLMs to improve the generalization of tasks, 204 but they fail to model the unique structural information of graph data, leading to suboptimal results. 205 Subsequent works use specialized graph encoders to handle structural information [Tang et al., 2024a, 206 Chen et al., 2024, Kong et al., 2025, Tian et al., 2024, He et al., 2025, Tang et al., 2024b, Zhang et al., 207 2024]. GraphGPT [Tang et al., 2024a] trains a graph encoder by aligning structural and semantic 208 information using CLIP [Radford et al., 2021]. LLaGA [Chen et al., 2024] uses Laplacian embeddings 209 as the structural encoder to help the model recognize graph-structured knowledge. GOFA [Kong 210 et al., 2025] incorporates the embeddings of LLMs into the GNN message passing process to allow 211 interaction between the graph encoder and LLMs. Despite these efforts, most existing approaches 212 either treat the graph as static input or fail to dynamically adapt to user queries. This significantly 213 limits their ability to perform complex reasoning over large-scale graphs. In contrast, GraphPack 214 explicitly models the interplay between query intent and graph structure through a semantic-aware 215 subgraph retrieval mechanism , enabling more effective and targeted reasoning.

## 216 **3.2 Retrieval On Graphs**

217 In GraphRAG, various retrieval methods exhibit distinct advantages when addressing different aspects 218 of the retrieval task. We categorize them into two main types: Parameter-free Retrievers and Model219 based Retrievers. **Parameter-free Retrievers** do not rely on deep learning models, enabling efficient 220 and scalable retrieval. For instance, QA-GNN [Yasunaga et al., 2022] connect the QA context and KG
221 to form a joint graph. OpenCSR [Han et al., 2023] constructs a question-dependent open knowledge 222 graph based on retrieved supporting facts. GraphRAG [Edge et al., 2025] structures the corpus to 223 enable query-centric retrieval. GRAG [Hu et al., 2025] retrieves subgraphs based on the similarity 224 between the query and entities. G-Retriever [He et al., 2024] extracts relevant subgraphs using 225 Prize-Collecting Steiner Tree optimization. **Model-based Retrievers** train specialized models to 226 extract relevant entities or subgraphs, achieving higher accuracy at the cost of increased computational 227 overhead. Some studies [Mavromatis and Karypis, 2024, Han et al., 2023] employs GNN to identify 228 entities from the knowledge graph. Subgraph Retriever[Zhang et al., 2022] uses RoBERTa [Liu et al., 229 2019] to expand from the topic entity and retrieves the relevant paths in a sequential decision process. 230 Unlike previous methods, GraphPack formulates subgraph retrieval as an optimization problem akin 231 to the knapsack problem, ensuring that the selected subgraphs are both highly relevant and minimally 232 noisy. Moreover, our approach can adapt to new tasks without requiring retraining, making it more 233 practical and versatile than existing model-based retrievers.

| second-best results are marked with underlines. Model Cora Citeseer   | Wikics   | Instagram   | ogbn-arxiv   |       |       |       |       |       |       |       |
|-----------------------------------------------------------------------|----------|-------------|--------------|-------|-------|-------|-------|-------|-------|-------|
| Acc                                                                   | F1       | Acc         | F1           | Acc   | F1    | Acc   | F1    | Acc   | F1    |       |
| OFA                                                                   | 75.24    | 74.20       | 73.04        | 68.98 | 77.34 | 74.97 | 60.85 | 55.44 | 73.23 | 57.38 |
| InstructGLM                                                           | 69.10    | 65.74       | 51.87        | 50.65 | 45.73 | 42.70 | 57.94 | 54.87 | 39.09 | 24.65 |
| GraphText                                                             | 76.21    | 74.51       | 59.43        | 56.43 | 67.35 | 64.55 | 62.64 | 54.00 | 49.47 | 24.76 |
| GraphAdapter                                                          | 72.85    | 70.66       | 69.57        | 66.21 | 70.85 | 66.49 | 67.40 | 58.40 | 74.45 | 56.04 |
| LLaGA                                                                 | 74.42    | 72.50       | 55.73        | 54.83 | 73.88 | 70.90 | 62.94 | 54.62 | 72.78 | 53.86 |
| GraphPack                                                             | 76.40    | 75.45       | 69.95        | 67.59 | 79.59 | 77.18 | 66.40 | 59.34 | 75.01 | 58.51 |

| Model       | WebQSP   | CWQ   |       |       |
|-------------|----------|-------|-------|-------|
| F1          | Hit@1    | F1    | Hit@1 |       |
| Llama-2-7B  | 42.95    | 61.86 | 32.29 | 36.92 |
| Mistral-7B  | 43.11    | 62.52 | 32.87 | 36.46 |
| G-Retriever | 50.23    | 70.16 | 39.89 | 47.75 |
| GRAG        | 50.41    | 72.75 | 39.62 | 47.43 |
| GraphPack   | 51.79    | 73.01 | 41.03 | 48.50 |

F1 Scor e (%
)

□ F1 Score - Avg. Node 10 20 30 40 40.5 41 41.5 5 10 15 20 Avg
. Node Capacity

## 234 **4 Experiments**

235 We conducted comprehensive experiments to validate the effectiveness of our framework under 236 various settings, aiming to address the following key research questions: 237 **RQ1.** How does GraphPack perform overall on different graph tasks? 238 **RQ2.** How does GraphPack affect the reasoning of LLMs? 239 **RQ3.** How well does GraphPack generalize across different tasks under the zero-shot setting? 240 **RQ4.** What is the role of query-aware modeling in GraphPack?

## 241 **4.1 Experimental Settings**

242 **Datasets.** The datasets and tasks used in our evaluation represent knowledge-intensive graph 243 reasoning , where successful performance requires not only semantic understanding but also the 244 ability to integrate complex relational structures. These tasks span multiple domains and reasoning 245 paradigms, including citation graphs, social networks, and knowledge graphs, etc. We present the 246 details of the datasets we used in Appendix C.1.

Implement Details. To ensure a fair comparison, we employ the Llama-2-7b1 247 base model as the 248 baseline. Additionally, we select Sentence-BERT [Reimers and Gurevych, 2019] as the text encoder 249 and GraphTransformer [Shi et al., 2021] as the graph encoder. All training and experiment details, 250 including baseline, hyperparameters and templates, are provided in the Appendix C.

## 251 **4.2 Overall Performance On Supervised Learning (Rq1)**

252 As shown in Table 1 and Table 2, Across a range of benchmark tests, our framework demonstrates 253 significantly improved performance compared to traditional baseline models. Notably, the methods 1https://huggingface.co/meta-llama/Llama-2-7b-hf Table 3: Comparison of Prediction Results Between ChatGPT and GraphPack on the WebQSP
Dataset. Predictions with a ★ symbol match the ground truth.

Question: What are some inventions that leonardo da vinci invented? Ground Truth: Diving suit | Triple barrel canon | Viola organista | Double hull | Aerial screw | Anemometer | 33-barreled organ | Armored car | Parachute | Ornithopter ChatGPT: Flying Machine, Anemometer★, Diving Suit★, Ball Bearings, Helicopter GraphPack: Anemometer★, Triple barrel canon★, Aerial screw★, 33-barreled organ★,
Double hull★
Question: What languages do they speak in costa rica? Ground Truth: Bribri language | Spanish language | Limonese creole | Jamaican creole english language ChatGPT: In Costa Rica, the official language is Spanish★. Additionally, English is also commonly spoken GraphPack: Spanish language★ | Limonese creole★ | Bribri language★ | Jamaican creole english language★
254 employed in the baseline model are not well-suited for various types of graph tasks, whereas 255 GraphPack highlights its versatility and outstanding effectiveness in tackling diverse graph-related 256 challenges. Furthermore, as task size and complexity grow, GraphPack consistently maintains robust 257 and efficient performance, offering a universal and powerful solution for a broader spectrum of graph 258 tasks. Further performance reports on more graph benchmark tasks and knowledge-intensive tasks 259 are presented in Appendix D.1.

## 260 **4.3 Subgraph Retrieval Strategy (Rq2)**

261 To verify the effectiveness of GraphPack's graph-enhanced retrieval strategy, we evaluate its impact on 262 LLMs without fine-tuning. Table 4 demonstrates the performance improvements achieved by different 263 strategies during the inference of LLMs without any fine-tuning. It is noteworthy that GraphPack 264 achieves a 18.61% increase in F1 Score compared to the baseline model. This is particularly important 265 in real-world question answering scenarios, as it can provide users with more correct candidate entities 266 to choose from. Furthermore, As shown in Table 3, we analyze the performance of ChatGPT and 267 GraphPack when addressing questions involving multiple entities within labels. The results reveal 268 that ChatGPT exhibit false detection issues, whereas GraphPack demonstrates higher reliability in 269 handling multi-entity problems. This validates the perspective raised in RQ2: GraphPack significantly 270 enhances the practicality of the model in graph-based question-answering scenarios by offering users 271 more accurate and diverse candidate entities. We present a comparison of subgraph retrieval time and 272 efficiency between GraphPack and other methods in Appendix D.2. Notably, GraphPack retrieves the 273 optimal subgraph in less than 0.25 seconds - even in graphs containing millions of nodes. These 274 advantages make the GraphPack strategy significantly valuable in practical applications.

275 Furthermore, We conduct an ablation study over a range of knapsack capacities C to examine the 276 impact of subgraph size on retrieval effectiveness and computational efficiency. As shown in Figure 3, 277 increasing C allows the model to retrieve more nodes on average - from 8.34 nodes at C=10 to 17.96 278 nodes at C=30 - suggesting improved coverage of the graph structure. However, this increase in 279 coverage does not translate into consistent gains in performance. On the WebQSP dataset, the best 280 result (41.03 F1 score) is achieved at C=20. Further increasing C to 30 leads to a drop in performance 281 (40.72 F1 score), likely due to the inclusion of noisy or irrelevant entities that distract the LLM during 282 generation. This trend highlights a key insight: the optimal setting strikes a balance between semantic 283 richness and structural compactness, ensuring both high-quality retrieval and efficient reasoning.

## 284 **4.4 Zero-Shot Adaptation And Transfer Performance (Rq3)**

285 Zero-shot learning involves training the model on a specific dataset and then evaluating it on un286 seen datasets or tasks. This approach is crucial for assessing the generalization capability of the

| ments.   |
|----------|
| Table 4: Impact of different retrieval strategies. Model WebQSP F1 Hit@1 Recall Llama2-7B 0.2555 0.4148 0.2920 G-Retriever 0.2571 0.4760 0.2954 GraphPack 0.3023 0.4732 0.3061 Mistral-7B 0.2589 0.4213 0.2967 G-Retriever 0.2634 0.4832 0.2981 GraphPack 0.3071 0.4878 0.3088          |

Cora→Wikics Llama2-7B 0.4115 0.3772

GraphPack 0.5589 0.5367

Cora→Instagram Llama2-7B 0.4078 0.4369

GraphPack 0.4543 0.4698

CWQ→Wikics Llama2-7B 0.1534 0.1802

GraphPack 0.4279 0.4167

CWQ→Instagram Llama2-7B 0.1679 0.2421

GraphPack 0.39.87 0.4021

287 model. Specifically, we design two experimental settings to evaluate different aspects of zero-shot 288 performance. The first setting focuses on cross-domain generalization , where the model is trained 289 on citation graph datasets and evaluated on social network graphs. The second setting examines 290 cross-task generalization , involving different textual description templates of the graph and varying 291 user intents. As shown in Table 5, we compare the zero-shot performance of LLMs and GraphPack 292 under various settings. The results indicate that GraphPack consistently outperforms the fine-tuned 293 LLM in all conditions. In particular, when evaluated on cross-task scenarios, the fine-tuned LLM 294 struggles to answer domain-specific questions, whereas GraphPack maintains strong zero-shot perfor295 mance. This suggests that the structural knowledge encoded through our retrieval and modulation 296 framework transfers well across domains and task formulations, even without access to target-domain 297 supervision. Furthermore, in more complex and resource-constrained settings - such as when only 298 partial graph structures are available or when the target domain exhibits significant divergence - 299 GraphPack still demonstrates robust performance. Additional experiments presented in Appendix 300 D.3 explore these challenging zero-shot and few-shot scenarios.

## 301 **4.5 Effectiveness Of Query-Aware Modeling (Rq4)**

302 We conduct ablation studies by systematically removing different components of the query-aware 303 modeling framework and evaluating their impact on performance. In one variant, we remove the 304 ranking-based value assignment for both nodes and edges, thereby eliminating the model's ability to 305 prioritize semantically meaningful connections during subgraph selection. Additionally, we evaluate 306 the effect of excluding the Query-LM module from the graph encoder, effectively replacing the 307 conditional modulation mechanism with a standard static aggregation scheme commonly used in 308 traditional GNNs. Experimental results in Appendix D.4 demonstrate that the removal of any of these 309 query-aware components leads to consistent performance degradation across a range of knowledge310 intensive tasks. This highlights the importance of integrating explicit query signals into both the 311 retrieval and encoding stages, as doing so enables the model to dynamically align its focus with user 312 intent while preserving structural coherence.

## 313 **5 Conclusion, Limitations, And Future Works**

314 In this paper, we propose GraphPack, a query-aware framework for Graph Retrieval-Augmented 315 Generation. Its core idea is to cast subgraph selection as a 0-1 knapsack optimisation that simultane316 ously maximises semantic relevance and minimises topological redundancy, then encode the chosen 317 subgraph with a query-aware graph encoder whose parameters adapt to the user's intent. Extensive 318 experiments on citation, social-network and knowledge-graph benchmarks demonstrate that Graph319 Pack consistently outperforms strong GraphRAG baselines in supervised, cross-domain and zero-shot 320 settings. Two practical limitations remain: the framework's dependence on high-quality semantic 321 embeddings means noisy or sparse signals can degrade anchor node identification. Additionally, 322 GraphPack depends on downstream task fine-tuning, restricting its potential to become a general 323 graph foundation model. Addressing these challenges, by improving robustness to noisy semantics 324 and developing GFM—forms promising directions for future work.

## 325 **References**

326 Runjin Chen, Tong Zhao, Ajay Kumar Jaiswal, Neil Shah, and Zhangyang Wang. LLaGA: Large 327 language and graph assistant. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian 328 Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, *Proceedings of the 41st* 329 *International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning* 330 *Research*, pages 7809–7823. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/ 331 v235/chen24bh.html. 332 Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, 333 Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A 334 graph rag approach to query-focused summarization, 2025. URL https://arxiv.org/abs/ 335 2404.16130. 336 Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large 337 language models, 2023. URL https://arxiv.org/abs/2310.04560.

338 Arnaud Freville. The multidimensional 0-1 knapsack problem: An overview. *European Journal* 339 *of Operational Research*, 155(1):1–21, May 2004. URL https://ideas.repec.org/a/eee/ 340 ejores/v155y2004i1p1-21.html. 341 Bin Fu, Yunqi Qiu, Chengguang Tang, Yang Li, Haiyang Yu, and Jian Sun. A survey on complex 342 question answering over knowledge base: Recent advances and challenges, 2020. URL https: 343 //arxiv.org/abs/2007.13069. 344 Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrieval345 augmented language model pre-training, 2020. URL https://arxiv.org/abs/2002.08909. 346 Zhen Han, Yue Feng, and Mingming Sun. A graph-guided reasoning approach for open-ended 347 commonsense question answering, 2023. URL https://arxiv.org/abs/2303.10395.

348 Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. Lightgcn: 349 Simplifying and powering graph convolution network for recommendation, 2020. URL https: 350 //arxiv.org/abs/2002.02126. 351 Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, 352 and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and 353 question answering, 2024. URL https://arxiv.org/abs/2402.07630. 354 Yufei He, Yuan Sui, Xiaoxin He, and Bryan Hooi. Unigraph: Learning a unified cross-domain foun355 dation model for text-attributed graphs, 2025. URL https://arxiv.org/abs/2402.13630.

356 Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. GRAG: Graph 357 retrieval-augmented generation. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, *Findings* 358 *of the Association for Computational Linguistics: NAACL 2025*, pages 4145–4157, Albuquerque, 359 New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. 360 URL https://aclanthology.org/2025.findings-naacl.232/. 361 Xuanwen Huang, Kaiqiao Han, Yang Yang, Dezheng Bao, Quanjin Tao, Ziwei Chai, and Qi Zhu. 362 Can gnn be good adapter for llms? WWW, 2024.

363 Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane 364 Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with 365 retrieval augmented language models, 2022. URL https://arxiv.org/abs/2208.03299.

366 Lecheng Kong, Jiarui Feng, Hao Liu, Chengsong Huang, Jiaxin Huang, Yixin Chen, and Muhan 367 Zhang. Gofa: A generative one-for-all model for joint graph language modeling, 2025. URL 368 https://arxiv.org/abs/2407.09709. 369 Yunshi Lan, Gaole He, Jinhao Jiang, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. A survey on 370 complex knowledge base question answering: Methods, challenges and solutions. In Zhi-Hua 371 Zhou, editor, *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence,* 372 *IJCAI-21*, pages 4483–4491. International Joint Conferences on Artificial Intelligence Organization, 373 8 2021. doi: 10.24963/ijcai.2021/611. URL https://doi.org/10.24963/ijcai.2021/611.

374 Survey Track. 375 Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman 376 Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, 377 and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In 378 H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in* 379 *Neural Information Processing Systems*, volume 33, pages 9459–9474. Curran Associates, 380 Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/ 381 6b493230205f780e1bc26945df7481e5-Paper.pdf. 382 Haochen Liu, Song Wang, Yaochen Zhu, Yushun Dong, and Jundong Li. Knowledge graph-enhanced 383 large language models via path selection. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, 384 editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 6311–6321, 385 Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 386 2024.findings-acl.376. URL https://aclanthology.org/2024.findings-acl.376/. 387 Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike 388 Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining 389 approach, 2019. URL https://arxiv.org/abs/1907.11692. 390 Qiyao Ma, Xubin Ren, and Chao Huang. XRec: Large language models for explainable recommenda391 tion. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, *Findings of the Association* 392 *for Computational Linguistics: EMNLP 2024*, pages 391–402, Miami, Florida, USA, November 393 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.22. URL 394 https://aclanthology.org/2024.findings-emnlp.22/. 395 Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language model 396 reasoning, 2024. URL https://arxiv.org/abs/2405.20139.

397 Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang 398 Tang. Graph retrieval-augmented generation: A survey, 2024. URL https://arxiv.org/abs/ 399 2408.08921. 400 Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual rea401 soning with a general conditioning layer, 2017. URL https://arxiv.org/abs/1709.07871.

402 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, 403 Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 404 Learning transferable visual models from natural language supervision, 2021. URL https: 405 //arxiv.org/abs/2103.00020. 406 Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and 407 Yoav Shoham. In-context retrieval-augmented language models. *Transactions of the Association* 408 *for Computational Linguistics*, 11:1316–1331, 2023. doi: 10.1162/tacl_a_00605. URL https: 409 //aclanthology.org/2023.tacl-1.75/. 410 Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-
411 networks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, *Proceedings* 412 *of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th* 413 *International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3982– 414 3992, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 415 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410/. 416 Xubin Ren, Jiabin Tang, Dawei Yin, Nitesh Chawla, and Chao Huang. A survey of large language 417 models for graphs. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery* 418 *and Data Mining*, KDD '24, page 6616–6626. ACM, August 2024. doi: 10.1145/3637528.3671460.

419 URL http://dx.doi.org/10.1145/3637528.3671460. 420 Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked 421 label prediction: Unified message passing model for semi-supervised classification, 2021. URL 422 https://arxiv.org/abs/2009.03509.

423 Yanchao Tan, Hang Lv, Xinyi Huang, Jiawei Zhang, Shiping Wang, and Carl Yang. Musegraph:
424 Graph-oriented instruction tuning of large language models for generic graph mining, 2024. URL
425 https://arxiv.org/abs/2403.04780.

426 Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang.

427 Graphgpt: Graph instruction tuning for large language models, 2024a. URL https://arxiv.

428 org/abs/2310.13023. 429 Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. Higpt: 430 Heterogeneous graph language model, 2024b. URL https://arxiv.org/abs/2402.16024. 431 Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V. Chawla, and 432 Panpan Xu. Graph neural prompting with large language models. In Proceedings of the Thirty433 *Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative* 434 *Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in* 435 *Artificial Intelligence*, AAAI'24/IAAI'24/EAAI'24. AAAI Press, 2024. ISBN 978-1-57735-887-9.

436 doi: 10.1609/aaai.v38i17.29875. URL https://doi.org/10.1609/aaai.v38i17.29875.

437 Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, and Julian McAuley. Instructgraph: 438 Boosting large language models via graph-centric instruction tuning and preference alignment, 439 2024. URL https://arxiv.org/abs/2402.08785. 440 Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: 441 Reasoning with language models and knowledge graphs for question answering, 2022. URL 442 https://arxiv.org/abs/2104.06378. 443 Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. Language is all a graph 444 needs, 2024. URL https://arxiv.org/abs/2308.07134. 445 Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong Chen. Subgraph 446 retrieval enhanced model for multi-hop knowledge base question answering. In Smaranda Muresan, 447 Preslav Nakov, and Aline Villavicencio, editors, *Proceedings of the 60th Annual Meeting of the* 448 *Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5773–5784, Dublin, 449 Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 450 396. URL https://aclanthology.org/2022.acl-long.396/. 451 Mengmei Zhang, Mingwei Sun, Peng Wang, Shen Fan, Yanhu Mo, Xiaoxiao Xu, Hong Liu, Cheng 452 Yang, and Chuan Shi. Graphtranslator: Aligning graph model to large language model for 453 open-ended tasks, 2024. URL https://arxiv.org/abs/2402.07197.

454 Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, and 455 Jian Tang. Graphtext: Graph reasoning in text space, 2023. URL https://arxiv.org/abs/ 456 2310.01089.

## 457 **Neurips Paper Checklist**

458 1. **Claims** 459 Question: Do the main claims made in the abstract and introduction accurately reflect the 460 paper's contributions and scope? 461 Answer: [Yes] 462 Justification: Yes, the main claims made in the abstract and introduction accurately reflect 463 the paper's contributions and scope. 464 Guidelines:
465 - The answer NA means that the abstract and introduction do not include the claims 466 made in the paper.

467 - The abstract and/or introduction should clearly state the claims made, including the 468 contributions made in the paper and important assumptions and limitations. A No or 469 NA answer to this question will not be perceived well by the reviewers. 470 - The claims made should match theoretical and experimental results, and reflect how 471 much the results can be expected to generalize to other settings. 472 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 473 are not attained by the paper. 474 2. **Limitations** 475 Question: Does the paper discuss the limitations of the work performed by the authors? 476 Answer: [Yes] 477 Justification: Check Section 5. 478 Guidelines: 479 - The answer NA means that the paper has no limitation while the answer No means that 480 the paper has limitations, but those are not discussed in the paper. 481 - The authors are encouraged to create a separate "Limitations" section in their paper. 482 - The paper should point out any strong assumptions and how robust the results are to 483 violations of these assumptions (e.g., independence assumptions, noiseless settings, 484 model well-specification, asymptotic approximations only holding locally). The authors 485 should reflect on how these assumptions might be violated in practice and what the 486 implications would be. 487 - The authors should reflect on the scope of the claims made, e.g., if the approach was 488 only tested on a few datasets or with a few runs. In general, empirical results often 489 depend on implicit assumptions, which should be articulated. 490 - The authors should reflect on the factors that influence the performance of the approach. 491 For example, a facial recognition algorithm may perform poorly when image resolution 492 is low or images are taken in low lighting. Or a speech-to-text system might not be 493 used reliably to provide closed captions for online lectures because it fails to handle 494 technical jargon. 495 - The authors should discuss the computational efficiency of the proposed algorithms 496 and how they scale with dataset size. 497 - If applicable, the authors should discuss possible limitations of their approach to 498 address problems of privacy and fairness. 499 - While the authors might fear that complete honesty about limitations might be used by 500 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 501 limitations that aren't acknowledged in the paper. The authors should use their best 502 judgment and recognize that individual actions in favor of transparency play an impor503 tant role in developing norms that preserve the integrity of the community. Reviewers 504 will be specifically instructed to not penalize honesty concerning limitations.

## 505 3. **Theory Assumptions And Proofs**

506 Question: For each theoretical result, does the paper provide the full set of assumptions and 507 a complete (and correct) proof? 508 Answer: [Yes] 509 Justification: 510 Guidelines: 511 - The answer NA means that the paper does not include theoretical results. 512 - All the theorems, formulas, and proofs in the paper should be numbered and cross513 referenced. 514 - All assumptions should be clearly stated or referenced in the statement of any theorems. 515 - The proofs can either appear in the main paper or the supplemental material, but if 516 they appear in the supplemental material, the authors are encouraged to provide a short 517 proof sketch to provide intuition. 518 - Inversely, any informal proof provided in the core of the paper should be complemented 519 by formal proofs provided in appendix or supplemental material. 520 - Theorems and Lemmas that the proof relies upon should be properly referenced.

## 521 4. **Experimental Result Reproducibility**

522 Question: Does the paper fully disclose all the information needed to reproduce the main ex523 perimental results of the paper to the extent that it affects the main claims and/or conclusions 524 of the paper (regardless of whether the code and data are provided or not)? 525 Answer: [Yes]
526 Justification: The paper fully discloses all the information needed to reproduce the main 527 experimental results as they pertain to the paper's main claims and conclusions. The code is 528 provided in the supplementary materials. 529 Guidelines: 530 - The answer NA means that the paper does not include experiments. 531 - If the paper includes experiments, a No answer to this question will not be perceived 532 well by the reviewers: Making the paper reproducible is important, regardless of 533 whether the code and data are provided or not. 534 - If the contribution is a dataset and/or model, the authors should describe the steps taken 535 to make their results reproducible or verifiable.

536 - Depending on the contribution, reproducibility can be accomplished in various ways.

537 For example, if the contribution is a novel architecture, describing the architecture fully 538 might suffice, or if the contribution is a specific model and empirical evaluation, it may 539 be necessary to either make it possible for others to replicate the model with the same 540 dataset, or provide access to the model. In general. releasing code and data is often 541 one good way to accomplish this, but reproducibility can also be provided via detailed 542 instructions for how to replicate the results, access to a hosted model (e.g., in the case 543 of a large language model), releasing of a model checkpoint, or other means that are 544 appropriate to the research performed. 545 - While NeurIPS does not require releasing code, the conference does require all submis546 sions to provide some reasonable avenue for reproducibility, which may depend on the 547 nature of the contribution. For example 548 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 549 to reproduce that algorithm. 550 (b) If the contribution is primarily a new model architecture, the paper should describe 551 the architecture clearly and fully. 552 (c) If the contribution is a new model (e.g., a large language model), then there should 553 either be a way to access this model for reproducing the results or a way to reproduce 554 the model (e.g., with an open-source dataset or instructions for how to construct 555 the dataset). 556 (d) We recognize that reproducibility may be tricky in some cases, in which case 557 authors are welcome to describe the particular way they provide for reproducibility. 558 In the case of closed-source models, it may be that access to the model is limited in 559 some way (e.g., to registered users), but it should be possible for other researchers 560 to have some path to reproducing or verifying the results. 561 5. **Open access to data and code** 562 Question: Does the paper provide open access to the data and code, with sufficient instruc563 tions to faithfully reproduce the main experimental results, as described in supplemental 564 material? 565 Answer: [Yes] 566 Justification: The paper provides open access to the data and code, along with sufficient 567 instructions in the supplemental material to faithfully reproduce the main experimental 568 results. 569 Guidelines: 570 - The answer NA means that paper does not include experiments requiring code. 571 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/ 572 public/guides/CodeSubmissionPolicy) for more details. 573 - While we encourage the release of code and data, we understand that this might not be 574 possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not 575 including code, unless this is central to the contribution (e.g., for a new open-source 576 benchmark). 577 - The instructions should contain the exact command and environment needed to run to 578 reproduce the results. See the NeurIPS code and data submission guidelines (https: 579 //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

580 - The authors should provide instructions on data access and preparation, including how 581 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 582 - The authors should provide scripts to reproduce all experimental results for the new 583 proposed method and baselines. If only a subset of experiments are reproducible, they 584 should state which ones are omitted from the script and why. 585 - At submission time, to preserve anonymity, the authors should release anonymized 586 versions (if applicable). 587 - Providing as much information as possible in supplemental material (appended to the 588 paper) is recommended, but including URLs to data and code is permitted. 589 6. **Experimental setting/details** 590 Question: Does the paper specify all the training and test details (e.g., data splits, hyper591 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 592 results? 593 Answer: [Yes] 594 Justification: The paper specifies all the training and test details necessary to understand the 595 results, including data splits, hyperparameter settings, optimizer types, and how these were 596 chosen. These details are provided in the Appendix C. 597 Guidelines: 598 - The answer NA means that the paper does not include experiments. 599 - The experimental setting should be presented in the core of the paper to a level of detail 600 that is necessary to appreciate the results and make sense of them. 601 - The full details can be provided either with the code, in appendix, or as supplemental 602 material.

## 603 7. **Experiment Statistical Significance**

604 Question: Does the paper report error bars suitably and correctly defined or other appropriate 605 information about the statistical significance of the experiments? 606 Answer: [Yes] 607 Justification: The paper reports the average results across all experiments based on five runs 608 of training and testing, which is sufficient to demonstrate the consistency and reliability of 609 the experimental outcomes. 610 Guidelines: 611 - The answer NA means that the paper does not include experiments.

612 - The authors should answer "Yes" if the results are accompanied by error bars, confi613 dence intervals, or statistical significance tests, at least for the experiments that support 614 the main claims of the paper. 615 - The factors of variability that the error bars are capturing should be clearly stated (for 616 example, train/test split, initialization, random drawing of some parameter, or overall 617 run with given experimental conditions). 618 - The method for calculating the error bars should be explained (closed form formula, 619 call to a library function, bootstrap, etc.) 620 - The assumptions made should be given (e.g., Normally distributed errors). 621 - It should be clear whether the error bar is the standard deviation or the standard error 622 of the mean. 623 - It is OK to report 1-sigma error bars, but one should state it. The authors should 624 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis 625 of Normality of errors is not verified. 626 - For asymmetric distributions, the authors should be careful not to show in tables or 627 figures symmetric error bars that would yield results that are out of range (e.g. negative 628 error rates). 629 - If error bars are reported in tables or plots, The authors should explain in the text how 630 they were calculated and reference the corresponding figures or tables in the text. 631 8. **Experiments compute resources** 632 Question: For each experiment, does the paper provide sufficient information on the com633 puter resources (type of compute workers, memory, time of execution) needed to reproduce 634 the experiments? 635 Answer: [Yes] 636 Justification: The paper provides sufficient information on the computer resources required to 637 reproduce each experiment, including the type of compute workers, memory, and execution 638 time. This information can be found in Appendix C. 639 Guidelines: 640 - The answer NA means that the paper does not include experiments. 641 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, 642 or cloud provider, including relevant memory and storage. 643 - The paper should provide the amount of compute required for each of the individual 644 experimental runs as well as estimate the total compute. 645 - The paper should disclose whether the full research project required more compute 646 than the experiments reported in the paper (e.g., preliminary or failed experiments that 647 didn't make it into the paper). 648 9. **Code of ethics** 649 Question: Does the research conducted in the paper conform, in every respect, with the 650 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 651 Answer: [Yes] 652 Justification: The research conducted in the paper conforms with the NeurIPS Code of 653 Ethics in every respect. 654 Guidelines: 655 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 656 - If the authors answer No, they should explain the special circumstances that require a 657 deviation from the Code of Ethics. 658 - The authors should make sure to preserve anonymity (e.g., if there is a special consid659 eration due to laws or regulations in their jurisdiction). 660 10. **Broader impacts** 661 Question: Does the paper discuss both potential positive societal impacts and negative 662 societal impacts of the work performed? 663 Answer: [NA]
664 Justification: The paper does no discuss both potential positive and negative societal impacts 665 of the proposed method.

## 666 Guidelines:

667 - The answer NA means that there is no societal impact of the work performed. 668 - If the authors answer NA or No, they should explain why their work has no societal 669 impact or why the paper does not address societal impact. 670 - Examples of negative societal impacts include potential malicious or unintended uses 671 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 672 (e.g., deployment of technologies that could make decisions that unfairly impact specific 673 groups), privacy considerations, and security considerations. 674 - The conference expects that many papers will be foundational research and not tied 675 to particular applications, let alone deployments. However, if there is a direct path to 676 any negative applications, the authors should point it out. For example, it is legitimate 677 to point out that an improvement in the quality of generative models could be used to 678 generate deepfakes for disinformation. On the other hand, it is not needed to point out 679 that a generic algorithm for optimizing neural networks could enable people to train 680 models that generate Deepfakes faster. 681 - The authors should consider possible harms that could arise when the technology is 682 being used as intended and functioning correctly, harms that could arise when the 683 technology is being used as intended but gives incorrect results, and harms following 684 from (intentional or unintentional) misuse of the technology. 685 - If there are negative societal impacts, the authors could also discuss possible mitigation 686 strategies (e.g., gated release of models, providing defenses in addition to attacks, 687 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 688 feedback over time, improving the efficiency and accessibility of ML). 689 11. **Safeguards**
690 Question: Does the paper describe safeguards that have been put in place for responsible 691 release of data or models that have a high risk for misuse (e.g., pretrained language models, 692 image generators, or scraped datasets)? 693 Answer: [NA] 694 Justification: 695 Guidelines: 696 - The answer NA means that the paper poses no such risks. 697 - Released models that have a high risk for misuse or dual-use should be released with 698 necessary safeguards to allow for controlled use of the model, for example by requiring 699 that users adhere to usage guidelines or restrictions to access the model or implementing 700 safety filters. 701 - Datasets that have been scraped from the Internet could pose safety risks. The authors 702 should describe how they avoided releasing unsafe images. 703 - We recognize that providing effective safeguards is challenging, and many papers do 704 not require this, but we encourage authors to take this into account and make a best 705 faith effort.

## 706 12. **Licenses For Existing Assets**

707 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 708 the paper, properly credited and are the license and terms of use explicitly mentioned and 709 properly respected? 710 Answer: [Yes] 711 Justification: 712 Guidelines: 713 - The answer NA means that the paper does not use existing assets. 714 - The authors should cite the original paper that produced the code package or dataset. 715 - The authors should state which version of the asset is used and, if possible, include a 716 URL. 717 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 718 - For scraped data from a particular source (e.g., website), the copyright and terms of 719 service of that source should be provided. 720 - If assets are released, the license, copyright information, and terms of use in the 721 package should be provided. For popular datasets, paperswithcode.com/datasets 722 has curated licenses for some datasets. Their licensing guide can help determine the 723 license of a dataset. 724 - For existing datasets that are re-packaged, both the original license and the license of 725 the derived asset (if it has changed) should be provided. 726 - If this information is not available online, the authors are encouraged to reach out to 727 the asset's creators. 728 13. **New assets** 729 Question: Are new assets introduced in the paper well documented and is the documentation 730 provided alongside the assets? 731 Answer: [Yes] 732 Justification: Yes, any new assets introduced in the paper are well documented, and the 733 documentation is provided alongside the assets in the appendix. 734 Guidelines: 735 - The answer NA means that the paper does not release new assets. 736 - Researchers should communicate the details of the dataset/code/model as part of their 737 submissions via structured templates. This includes details about training, license, 738 limitations, etc. 739 - The paper should discuss whether and how consent was obtained from people whose 740 asset is used. 741 - At submission time, remember to anonymize your assets (if applicable). You can either 742 create an anonymized URL or include an anonymized zip file.

## 743 14. **Crowdsourcing And Research With Human Subjects**

744 Question: For crowdsourcing experiments and research with human subjects, does the paper 745 include the full text of instructions given to participants and screenshots, if applicable, as 746 well as details about compensation (if any)? 747 Answer: [NA] 748 Justification:
749 Guidelines:
750 - The answer NA means that the paper does not involve crowdsourcing nor research with 751 human subjects.

752 - Including this information in the supplemental material is fine, but if the main contribu753 tion of the paper involves human subjects, then as much detail as possible should be 754 included in the main paper. 755 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, 756 or other labor should be paid at least the minimum wage in the country of the data 757 collector.

## 758 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human**

759 **subjects** 760 Question: Does the paper describe potential risks incurred by study participants, whether 761 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) 762 approvals (or an equivalent approval/review based on the requirements of your country or 763 institution) were obtained? 764 Answer: [NA] 765 Justification: 766 Guidelines:
767 - The answer NA means that the paper does not involve crowdsourcing nor research with 768 human subjects. 769 - Depending on the country in which research is conducted, IRB approval (or equivalent) 770 may be required for any human subjects research. If you obtained IRB approval, you 771 should clearly state this in the paper. 772 - We recognize that the procedures for this may vary significantly between institutions 773 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the 774 guidelines for their institution. 775 - For initial submissions, do not include any information that would break anonymity (if 776 applicable), such as the institution conducting the review. 777 16. **Declaration of LLM usage**
778 Question: Does the paper describe the usage of LLMs if it is an important, original, or 779 non-standard component of the core methods in this research? Note that if the LLM is used 780 only for writing, editing, or formatting purposes and does not impact the core methodology, 781 scientific rigorousness, or originality of the research, declaration is not required. 782 Answer: [Yes] 783 Justification: The paper describes the use of LLMs as a pre-trained model in the research, 784 Guidelines: 785 - The answer NA means that the core method development in this research does not 786 involve LLMs as any important, original, or non-standard components. 787 - Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) 788 for what should or should not be described.