000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Karpa: A Training-Free Method Of Adapting Knowledge Graph As References For Large Language Model'S Reasoning Path Aggrega- Tion

Anonymous authors Paper under double-blind review

## Abstract

Large language models (LLMs) demonstrate exceptional performance across a variety of tasks, yet they are often affected by hallucinations and the timeliness of knowledge. Leveraging knowledge graphs (KGs) as external knowledge sources has emerged as a viable solution, but existing methods for LLM-based knowledge graph question answering (KGQA) are often limited by step-by-step decision-making on KGs, restricting the global planning and reasoning capabilities of LLMs, or they require fine-tuning or pre-training on specific KGs. To address these challenges, we propose **Knowledge graph Assisted Reasoning Path** Aggregation (KARPA), a novel framework that harnesses the global planning abilities of LLMs for efficient and accurate KG reasoning on KGs. KARPA operates through a three-step process: pre-planning, retrieving, and reasoning. First, KARPA uses the LLM's global planning ability to pre-plan logically coherent relation paths based on the provided question and relevant relations within the KG. Next, in the retrieving phase, relation paths with high semantic similarity to the pre-planned paths are extracted as candidate paths using a semantic embedding model. Finally, these candidate paths are provided to the LLM for comprehensive reasoning. Unlike existing LLM-based KGQA methods, KARPA fully leverages the global planning and reasoning capabilities of LLMs without requiring stepwise traversal or additional training, and it is compatible with various LLM architectures. Extensive experimental results show that KARPA achieves state-of-the-art performance in KGQA tasks, delivering both high efficiency and accuracy. Our code is available on https://anonymous.4open.science/r/KARPA/.

## 1 Introduction

In recent years, large language models (LLMs) (Touvron et al., 2023a;b; Achiam et al., 2023; Bai et al., 2023) have revolutionized natural language processing, demonstrating remarkable capabilities in understanding and generating human-like text across a range of tasks. Their ability to leverage vast amounts of data leads to impressive performance in areas such as information extraction (Xu et al., 2023), summarization (Jin et al., 2024), and question answering (Louis et al., 2024). However, these models face notable challenges, particularly in maintaining up-to-date knowledge, domainspecific knowledge (Zhang et al., 2024), or dealing with hallucinations (Zhang et al., 2023b; Huang et al., 2023) where the models produce incorrect or nonsensical outputs. Knowledge graphs (KGs) present a promising solution to enhance the reasoning capabilities of LLMs by providing structured, reliable external knowledge (Zhu et al., 2024; Pan et al., 2024). Existing approaches that integrate LLMs with KGs generally fall into two categories. The first category involves direct interaction between the LLM and the KGs (Sun et al., 2023; Jiang et al., 2023),
where the LLM explores the KG step-by-step. The second category, including methods such as reasoning on graphs (RoG) (Luo et al., 2023), involves generating retrieval information to extract knowledge from KGs. This often requires fine-tuning or pre-training the LLM on specific KG data (Li et al., 2023b; Huang et al., 2024). However, both approaches have notable limitations: (1) The direct interaction method often relies on local search strategies such as beam search, which can result 1

Eduardo graduate from Saverin Question: Who is Mark Zuckerberg's wife? entities relations graduate from Harvard University co-founder drop out from KG
occupation Mark Zuckerberg born in occupation Priscilla Chan wife of CEO Meta CEO of born in CEO Pediatrician USA
LLM Question Pre-planning LLMQuestion Stepwise Beam Search LLM Question Pre-train or Fine-tune Relevant relations Candidate paths Retrieving Embedding Model Candidate paths Iteration LLM
Unseen KGs Q
Step1: [co-founder, born in]
Sorry, I do not know ...

Hallucination
[Eduardo Saverin, USA]
Retrieved paths
[CEO of -> wife of CEO, ...]
Retrieved Step2: [graduate from, ...]
Q
LLM
[wife of, spouse of]
Reasoning:
[Harvard University, ...]
LLM Retrieved paths Priscilla Chan Step3: ...

(c)
(a)
(b)
054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 in suboptimal answers by overlooking the LLM's potential for global reasoning and planning across the entire path. Moreover, this method typically demands a high number of interactions between the LLM and the KG, as illustrated in Figure 1(b). (2) In contrast, methods that involve pre-training or fine-tuning the LLM struggle with unseen KGs, often necessitating retraining. Additionally, they remain prone to hallucinations during the information generation process, as shown in Figure 1(a). To address these limitations, we propose **Knowledge graph Assisted Reasoning Path Aggregation** (KARPA), an innovative framework that leverages the global planning capabilities of LLMs alongside semantic embedding models for efficient and accurate KG reasoning. Our approach consists of three key steps: pre-planning, retrieving, and reasoning, as shown in Figure 1(c). In the preplanning phase, KARPA enables the LLM to generate initial relation paths for the provided question using LLM's inherent reasoning and planning capabilities. With these inital relation paths, KARPA employs a semantic embedding model (Ruder et al., 2019) to identify candidate relations that are semantically similar to the relations within the initial paths. The LLM can then create coherent relation paths that logically connect the topic entity to potential answer entities using these candidate relations. During the retrieving phase, KARPA employs an embedding model to identify candidate paths within the KG that exhibit the highest similarity to the relation paths generated by the LLM in the pre-planning phase. This avoids locally optimal issues encountered in previous methods. Finally, during the reasoning step, the candidate paths and their corresponding tail entities are provided to the LLM to formulate final answers. The detail of our framework is shown in Figure 2.

KARPA offers several key advantages over existing LLM-based KGQA methods: (1) KARPA fully exploits the global planning and reasoning abilities of LLMs, generating comprehensive relation paths without the need for iterative traversal within KGs, which significantly reduces interactions between the LLM and the KG. (2) Our embedding-based extraction strategy avoids the locally optimal solution that arises from the stepwise interactions between LLMs and KGs, ensuring more effective exploration of the KGs. (3) KARPA operates in a training-free manner, making it adaptable to various LLMs while enhancing the reasoning capabilities of LLMs over KGs through techniques such as chain-of-thought (CoT) (Wei et al., 2022). Our contributions can be summarized as follows:
- We propose KARPA, a framework that leverages the complementary strengths of LLMs and embedding models to improve both the accuracy and efficiency of KGQA tasks, while addressing the limitations of existing LLM-based methods.

- KARPA fully leverages the global planning and reasoning capabilities of LLMs in conjunction with a novel semantic embedding-based extraction method. In the pre-planning phase, the LLM is empowered to generate initial relation paths that are not restricted to adjacent relations, but can instead select from all potential relations within the KG, constructing logically coherent paths leading to answer entities. By integrating an embedding model to extract relation paths based on semantic similarity, KARPA mitigates the risk of the LLM getting trapped in local optima and significantly reduces the required interactions between the LLM and KGs. Techniques such as CoT prompting can also be incorporated to further enhance the LLM's reasoning abilities over KGs.

- Our KARPA framework operates in a training-free manner and can be seamlessly integrated with various LLMs, providing a plug-and-play solution that achieves state-of-the-art performance across multiple metrics on several KGQA benchmark datasets.

## 2 Related Work

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Prompt-Based Reasoning with LLMs. Large Language Models (LLMs), such as LLaMA (Touvron et al., 2023a;b), Qwen (Bai et al., 2023), and GPT-4 (Achiam et al., 2023), have made substantial progress in enhancing reasoning capabilities by leveraging their vast internal knowledge. Various prompt-based methods have been proposed to further optimize these capabilities. For instance, Chain-of-Thought (CoT) prompting (Wei et al., 2022) facilitates a structured reasoning process by breaking down intricate tasks into manageable steps, significantly boosting performance in areas such as mathematical reasoning (Jie et al., 2023) and logical inference (Zhao et al., 2023). Building on CoT, several variants have been introduced to further optimize reasoning effectiveness, including Auto-CoT (Zhang et al., 2022), Zero-Shot-CoT (Kojima et al., 2022), and Complex-CoT (Fu et al., 2022). Additionally, newer frameworks like the Tree of Thoughts (ToT) (Yao et al., 2024) and Graph of Thoughts (GoT) (Besta et al., 2024) have expanded the scope of LLM reasoning, enabling the models to generate intermediate steps and sub-goals, thereby enhancing their versatility across diverse reasoning tasks. Lately, OpenAI o1 series models represent a significant advancement in LLM reasoning, allowing the LLM to develop an extensive internal chain of thought. These developments underscore the importance of tailored prompts in maximizing LLMs' reasoning potential.

## 3 Preliminary

In this section, we introduce key concepts and definitions relevant to our work, including Knowledge Graphs (KGs), relation paths, reasoning paths, Knowledge Graph Question Answering (KGQA), as well as embedding models and semantic similarity. LLM-Based Knowledge Graph Question Answering. The integration of KGs with LLMs for question answering has emerged as a promising approach to enhance reasoning capabilities and mitigate hallucination phenomena. Unlike traditional CoT method that leverage the internal knowledge of LLMs, the incorporation of KGs facilitates access to structured external knowledge (He et al., 2022; Wang et al., 2023). Approaches such as Think-on-Graph (ToG) (Sun et al., 2023), Interactive- KBQA (Xiong et al., 2024) and StructGPT (Jiang et al., 2023) enable real-time interactions between LLMs and KGs. However, these methods often entail extensive interactions that can lead to inefficiencies. Reasoning on graphs (RoG) (Luo et al., 2023) uses instruction-tuned LLaMa2-Chat-7B to generate reasoning paths and achieves state-of-the-art performance on KGQA tasks. Similarly, methods such as chain of knowledge (Li et al., 2023c) and other approaches (Huang et al., 2024; Pan et al., 2024) employ LLMs to generate retrieval information for KGQA tasks. However, these methods require pre-training or fine-tuning process, which can be both costly and time-consuming. Additionally, methods such as UniKGQA (Jiang et al., 2022) and KG-CoT (Zhao et al., 2024) require the training of specific models for KG information retrieval, further complicating their implementation.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

Eduardo graduate from Saverin Question: Who is Mark Zuckerberg's wife? **Knowledge**
Graph (KG)
graduate from Harvard University Step 1: Pre-planning:
All Relations born in co-founder drop out from CoT [...]
[spouse of]
Initial Planning [wife]
[has child, parent of]
occupation Mark Zuckerberg born in occupation Priscilla Chan wife of CEO Meta LLM
Q
CEO of Plan USA
CEO Pediatrician Relation Extraction
[wife, spouse of, has child, parent of, ...]
Embedding Model
[All relations]
Relation Paths Step 2: Retrieving:
Top-K
Paths Step 3: Reasoning:
Candidate Paths Embedding Model
{wife of CEO}
{CEO of, wife of CEO} Candidate Paths:
Top-K Retrieved Paths LLM
Top-K
Paths Q CoT
Candidate Relations:
[wife of CEO (0.61), occupation (0.44), ...]
...{CEO of, wife of CEO}
Score: 0.96
{co-founder}
Score: 0.43
{occupation}
Score: 0.37
{born in}
Score: 0.24 LLM
Q CoT
Re-planning Relation Paths:
Candidate paths:
Let's analyze these relation paths step-by-step: ...

Therefore, the answer is:
{Priscilla Chan}. ... ...

Plan Let's think step-by-step...

Length 1 path might be:
{wife of CEO};
Length 2 path might be: {CEO of, wife of CEO}; 
{occupation}
{wife of CEO}
{born in}
{CEO of, wife of CEO}
Semantic Similarity Candidate Relations
...
Figure 2: The framework of our KARPA. Our framework consists of three main steps: (1) Preplanning: The LLM generates initial relation paths based on the given question. These paths are then decomposed for relation extraction using an embedding model. Utilizing the set of candidate relations, the LLM is able to re-plan logically coherent relation paths that potentially connect the topic entity and answer entities. (2) Retrieving: Candidate relation paths are extracted based on their similarity with re-planned initial paths, utilizing an embedding model. Our retrieval method accommodates paths that may differ in length from the re-planned initial paths. (3) Reasoning: The selected top-K candidate relation paths are combined with the question and relevant entities to form a comprehensive prompt for the LLM, facilitating accurate question answering over the KG. In this section, we present our proposed Knowledge graph Assisted Reasoning Path Aggregation (KARPA) framework, which leverages the strengths of LLMs and an embedding model to enhance KGQA. The approach consists of three key steps: pre-planning, retrieving, and reasoning.

$$(1)$$

Knowledge Graphs (KGs). A Knowledge Graph (KG) is a structured representation of information, which can be represented as G = (*E, R*), where E denotes the set of entities and R denotes the set of relations. Each relation r ∈ R connects a pair of entities (ei, ej ) such that ei, ej ∈ E.

Relation Paths and Reasoning Paths. Relation paths are sequences of relations that connect two entities within a KG. A relation path P from topic entity et to answer entity ea can be expressed as: P = (r1, r2*, . . . , r*n), where each ri ∈ R denotes the relations along the path. Reasoning paths extend this concept of relation paths by incorporating intermediate entities alone the path. A reasoning path Pr from et to ea can be represented as Pr =
net →
r1e1 →
r2e2 *. . .* →
rnea o.

Knowledge Graph Question Answering (KGQA). Knowledge Graph Question Answering (KGQA) involves the task of responding to questions by leveraging the information stored within KGs. Given a query Q, the goal of KGQA is to retrieve an answer A defined as: A = f(*Q, G*), where f is a function that extracts the answer based on query Q over the KG G. Embedding Models and Semantic Similarity. Embedding Models facilitate the representation of words and sentences in a continuous vector space, enabling semantic embedding and similarity measurement. An embedding function Φ : R → R
d maps a sentence R to d-dimensional vectors.

The similarity between two embeddings can be quantified using metrics such as cosine similarity:

$$\begin{split}\text{\emph{sim}}(r_{i},r_{j})=\frac{\Phi(r_{i})\cdot\Phi(r_{j})}{\|\Phi(r_{i})\|\|\Phi(r_{j})\|},\end{split}\tag{1}$$

where · denotes the dot product and *∥· ∥* represents the Euclidean norm. This metric provides a measure of similarity between vectors, aiding in the retrieval and comparison of semantic information.

## 4 Approach 4.1 Pre-Planning With Llm

The pre-planning phase is a crucial component of our KARPA framework, where we leverage the global planning capabilities of LLMs to generate initial relation paths P*initial*. This phase initiates the reasoning process by allowing the LLM to analyze the input question Q and the associated topic entity et. By leveraging the reasoning capability of LLM, KARPA is able to propose paths that are not only logically coherent but also have the potential to lead to the answer entities Ea.

Initial Planning Using LLM KARPA start by leveraging the LLM's global planning capabilities to generate initial relation paths based on the provided question Q, as shown in Figure 2. The LLM outputs a set of potential relation paths P as follows:

P = {p1, p2*, . . . , p*m} where pi = (r
i
$\vdots\ldots,r^i_{n_i})$ f. 
$${\mathrm{or}}\;i=1,2,\ldots,m.$$
) for i = 1, 2*, . . . , m.* (2)
In Equation 2, each pi represents a relation path consisting of ni relations, r ij ∈ R, that are logically coherent and could connect a topic entity et to potential answer entities ea. The goal is to create several paths of varying lengths that could serve as candidates for relations extraction. Relation Extraction Strategy Once the initial relation paths P are generated, we decompose each path piinto its constituent relations. For each path pi ∈ P, the relations are organized into a relation list denoted as Ri = {r i1
, ri2
, . . . , rini
}. For each relation r ijin list Ri, we utilize an embedding model to extract top-K semantically similar relations from the entire KG, as shown in Figure 2.

This can be represented as:

$$R_{j}^{i}=\{r_{j1},r_{j2},\ldots,r_{j k}\}=\mathrm{Top-K}(\mathrm{sim}(\mathbf{r_{j}^{i}},\mathbf{r}))\quad\mathrm{for}\;r\in R,$$
$\eqref{eq:walpha}$. 
, r)) for r ∈ R, (3)
where sim(·) denotes the semantic similarity function (e.g., cosine similarity) between the embedding of relation r ij and all relations r ∈ R using Equation 1. The resulting set Rij contains the relations that best align semantically with the initial relations, ensuring that the LLM has access to relevant relations beyond just the immediate neighbors of current entity in the KG. Re-planning Relation Paths with LLM In the re-planning step, we leverage the candidate relations Rijidentified in the previous phase to construct formal relation paths that potentially connect the topic entity et to the answer entity ea. The process can be described as follows:

$\mathbf{I}\;=\;\frac{\pi}{2}$ . 
P*initial* = LLM(*Q, R*ij), for each r i j ∈ R
i j ⊂ R. (4)
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Given the question Q and candidate relations Rij, the LLM utilizes its global planning and reasoning capabilities to output initial relation paths P*initial*, as shown in Figure 2. During this phase, we can integrate reasoning techniques like Chain-of-Thought (CoT) to further enhance the LLM's inference abilities on KGs. The CoT process encourages the LLM to consider the semantic connections between relations, leading to paths that are logically coherent. By employing candidate relations extracted from the entire KG rather than being restricted to neighboring relations, our KARPA framework allows the LLM to construct the most logical reasoning chains without stepwise interactions between the LLM and KGs. This mitigates the risk of becoming trapped in local optima while reducing the required number of interactions. Through pre-planning process, we set the stage for effective retrieval and reasoning in the subsequent steps of our KARPA.

## 4.2 Relation Paths Retrieval

In this section, we outline the retrieving step of our KARPA framework, which is designed to retrieve candidate relation paths in KGs. As shown in Figure 2, the retrieving process systematically explores potential relation paths derived from the initial paths generated by the LLM, providing candidate paths for reasoning step.

## 4.2.1 Conventional Relation Paths Retrieval

Conventional methods for LLM-based KG exploration ToG(Sun et al., 2023), typically involve the LLM selecting top-K promising relations Rt from the adjacent relations of the current entity e at

each step. This strategy resembles a greedy algorithm, such as beam search. Formally, let R(e) denote the set of relations available for the current entity e. The selection process can be defined as:
$R_{\rm selected}=\mbox{argmax}_{r\in R(e)}\ f(r)$, $r\in KG$.  
f(r), r ∈ KG. (5)
$\left({\mathfrak{H}}\right)$
In Equation 5, f(r) is a scoring function indicating the potential of relation r. Since embedding similarity represents the similarity between two relations, we use 1−sim(ri, rj ) as the cost function for beam search. However, this approach does not guarantee finding the optimal path, as it may overlook globally optimal solutions. To enhance relation path extraction, we employ traditional pathfinding algorithms like Dijkstra's, which can be expressed as:
cost(v) = min{cost(v)*, cost*(v
′) + *cost*(v
′, v) | v
′is a predecessor of v}. (6)
In Equation 6, the cost to reach node v is determined by either its current known cost or the cost of reaching one of its predecessors v
′ plus *cost*(v
′, v), the cost of the edge connecting v
′to v.

In KARPA, we begin from the topic entity et and compute the semantic similarity sim(ri, rj )
using Equation 1 for relations at each step, scoring the relations based on their similarity to the corresponding relations in the initial relation paths P*initial*. The cost for each step is defined as: cost(r) = 1 − sim(ri, rj ). This modification ensures that higher similarity scores correspond to lower costs, facilitating optimal path discovery. Since similarity scores range from 0 to 1, we average the total cost of relation paths of different lengths so that shorter paths can be fairly compared with longer paths. The path retrieval function based on Dijkstra's algorithm can be defined as:

$\mathbf{a}^{\prime}\mathbf{a}^{\prime}=\mathbf{a}\mathbf{b}$
$$\mathbf{f}\;v\}.$$
$$cost(e)=\min\left\{\frac{1}{n_{e}}cost(e),\frac{1}{n_{e^{\prime}}+1}\left[cost(e^{\prime})+sim(r_{(e^{\prime},e)},r_{initial})\right]\right\},\tag{7}$$
where the cost of entity e is compared between *cost*(e) averaged by the number of relations ne to reach entity e, and the cost of its predecessor *cost*(e
′) plus the current cost sim(r(e
′,e)), r*initial*),
averaged by number of relations ne
′ plus one. All current costs are computed between current relation and the corresponding relation in initial relation paths P*initial* using Equation 1.

## 4.2.2 Heuristic Value-Based Relation Paths Retrieval

Since the conventional relation paths retrieval methods require the cost of each relations alone the paths, the similarity between initial relation paths and current paths within the KG can only be calculated when current paths have the same length as initial paths P*initial*. Inspired by the heuristic value in A* algorithm, we design a heuristic value-based relation paths retrieval method. In the traditional A* algorithm, the heuristic value serves as the a guiding function that indicates the distance between current node and target node. In KARPA, the heuristic value h indicate the semantic similarity between the initial relation paths P*initial* and current path within the KG. By using heuristic value h as an indicator, we are able to compute the similarity between paths of differing lengths, such as A
father
−−−−→
father
−−−−→ B and A
grandfather
−−−−−−−−→ B, as shown in Figure 2. For paths Pa and Pb, we concatenate all relations into one sentence and use the embedding model to calculate their similarity:

$$s i m(P_{a},P_{b})={\frac{\mathrm{emb}(\mathrm{concat}(R(P_{a})))\cdot\mathrm{emb}(\mathrm{concat}(R(P_{b})))}{\|\mathrm{emb}(\mathrm{concat}(R(P_{a})))\|\|\mathrm{emb}(\mathrm{concat}(R(P_{b})))\|}}.$$
. (8)
In Equation 8, the similarity between path Pa and Pb can be calculated using the concatenation of their internal relations R(P). Since the heuristic value represents the semantic distance between Pa and Pb, it can be defined as h = 1 − sim(Pa, Pb). The top-K candidate relation paths Pc with
lowest heuristic value can be extracted as:
$P_{c}=\mbox{argmax}_{P\in P_{all}}\mbox{sim}(P,P_{initial})$, $P_{all}\in KG$.  
sim(P, Pinitial), Pall ∈ KG. (9)
Through Equation 9, we are able to identify and select the top-K relevant paths from a diverse range of lengths as candidate paths Pc for further reasoning.
The relation paths retrieval method in KARPA effectively broadens the search space and mitigates the risk of missing potentially optimal paths that traditional methods might overlook. The KARPA framework can dynamically adapt to various lengths of relation paths, even if the initial path of corresponding length does not exist. Through the retrieving step, we are able to extract the top-K candidate relation paths for LLM to predict the finial answer for KGQA tasks.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

$$({\mathfrak{s}})$$

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

| WebQSP                              | CWQ                           |          |       |      |          |       |      |
|-------------------------------------|-------------------------------|----------|-------|------|----------|-------|------|
| Type of Model                       | Method                        | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| Answering with Internal Knowledge   |                               |          |       |      |          |       |      |
| GPT-4                               | IO prompt                     | -        | 62.5  | -    | -        | 44.3  | -    |
| GPT-4                               | CoT* (Sun et al., 2023)       | -        | 67.3  | -    | -        | 46.0  | -    |
| Training-based Methods              |                               |          |       |      |          |       |      |
| LLaMA2-7B (Fine-tune)               | KD-CoT* (Wang et al., 2023)   | -        | 68.6  | 52.5 | -        | 55.7  | -    |
| Graph Reasoning Model               | KG-CoT* (Zhao et al., 2024)   | -        | 84.9  | -    | -        | 62.3  | -    |
| FiD-3B                              | DECAF* (Yu et al., 2022)      | -        | 82.1  | 78.8 | -        | 70.4  | -    |
| PLM (Pretrain)                      | UniKGQA* (Jiang et al., 2022) | -        | 77.2  | 72.2 | -        | 51.2  | 49.0 |
| LLaMA2-7B (Fine-tune)               | RoG                           | 80.4     | 84.6  | 70.1 | 60.5     | 61.3  | 54.2 |
| Direct Inferance over KGs with LLMs |                               |          |       |      |          |       |      |
| GPT-4o                              | ToG                           | 58.6     | 78.5  | 50.9 | 53.3     | 56.8  | 41.9 |
| GPT-4                               | ToG* (Sun et al., 2023)       | -        | 82.6  | -    | -        | 69.5  | -    |
| GPT-4o                              | KARPA                         | 76.1     | 87.7  | 69.2 | 69.8     | 75.3  | 58.4 |
| GPT-4                               | KARPA                         | 80.9     | 91.2  | 72.1 | 73.6     | 78.4  | 61.5 |

Table 1: Comparison between our proposed KARPA and other baseline approaches. The table summarizes the performance of three categories of methods: (1) Answering with internal knowledge of LLMs, (2) Training-based methods, which require constant re-train for unseen KGs, and (3)
Direct inference over KGs with LLMs. *Results are cited from corresponding publications. **Bold** represents the best result, underline represents the second best, and fbox represents the third best.

## 4.3 Reasoning With Llm

In the reasoning step, we combine the candidate relation paths with their respective entities into a prompt for the LLM to reference during the final answer determination, as shown in Figure 2. The reasoning process of LLM can be formally expressed as:
$$A n s w e r=\mathrm{LLM}(Q,P_{c},e_{t},e_{a}),\;P_{c}=\{r_{1},r_{2},\ldots,r_{n}\}.$$
$\left(10\right)^3$

## Given The Top-K Candidate Relation Paths Pc And The Question Q, The Llm Can Effectively Assess
Whether The Provided Connections Lead To A Valid Answer To Q. If The Top-K Candidate Paths Do Not Yield A Precise Answer, We Leverage The Llm'S Inherent Knowledge To Provide An Appropriate Response. The Karpa Framework Facilitates The Llm'S Ability To Evaluate Multiple Reasoning Paths In Parallel, Thereby Enhancing The Overall Efficiency Of Llm-Based Kgqa Tasks. 5 Experiments

In this section, we detail the experimental setup, present our main results, and conduct further analysis to evaluate the performance of our proposed Knowledge graph Assisted Reasoning Path Aggregation (KARPA) framework.

## 5.1 Experimental Settings

Datasets and Evaluation Metrics We evaluate KARPA on two widely used multi-hop KGQA datasets: WebQuestionSP (WebQSP) (Yih et al., 2016) and Complex WebQuestions (CWQ) (Talmor, 2018). These two datasets are designed for Multi-hop KGQA tasks. We compare our proposed KARPA and other LLM-based KGQA methods to demonstrate the effectiveness of our framework. For evaluation, we employ three metrics: Accuracy, Hit@1, and F1 score. Accuracy measures the proportion of correctly answered questions. Hit@1 evaluates whether the correct answer is among the top predicted answers. F1 score combines precision and recall into a single metric, offering a balance evaluation between the two metrics. Baselines for Comparison We compare KARPA against several baselines: (1) To demonstrate that KARPA derives answers through KG reasoning rather than relying on the internal knowledge of the LLM, we report the result of IO Prompt (Brown et al., 2020), which directly answers questions without a reasoning process. The result of CoT (Wei et al., 2022) is also included as a baseline 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

WebQSP CWQ

Model Tpye Method Accuracy Hit@1 F1 Accuracy Hit@1 F1

GPT-4o-mini

CoT - 61.3 - - 49.5 -

ToG 56.4 75.2 51.6 50.2 54.0 34.5

KARPA **71.9 85.3 64.5 68.1 73.3 56.5**

GPT-4o

CoT - 67.0 - - 52.3 -

ToG 58.6 78.5 50.9 53.3 56.8 41.9

KARPA **76.1 87.7 69.2 69.8 75.3 58.4**

GPT-4

CoT - 66.1 - - 54.7 -

ToG* - 82.6 - - 69.5 -

KARPA **80.9 91.2 72.1 73.6 78.4 61.5**

Claude-3.5-Sonnet

CoT - 72.3 - - 57.4 -

ToG 61.5 79.2 53.4 54.1 60.3 43.5

KARPA **82.6 89.5 69.7 70.7 73.6 54.9**

Gemini-1.5-Pro

CoT - 65.3 - - 52.1 -

ToG 62.3 78.4 52.5 51.7 57.9 40.5

KARPA **80.7 90.5 68.6 69.8 75.0 54.8**

Table 2: Comparison of our proposed KARPA, ToG, and CoT using various LLMs. The results demonstrate that KARPA consistently outperforms ToG, the previous state-of-the-art for direct KG- based reasoning using LLM. *Results of ToG are cited from corresponding paper (Sun et al., 2023). to evaluate the LLM's reasoning performance without external knowledge. (2) KARPA is further compared with training-based KGQA methods, including KD-CoT (Wang et al., 2023), UniKGQA (Jiang et al., 2022), DECAF (Yu et al., 2022), and RoG (Luo et al., 2023). This comparison demonstrates that KARPA effectively leverages the LLM's planning and reasoning capabilities without additional training. (3) Lastly, KARPA is compared with ToG (Sun et al., 2023), the current stateof-the-art method that operates without training. Experimental Details We test various LLMs including GPT-4 (OpenAI, 2023), GPT-4o (OpenAI, 2024), GPT-4-mini, Claude-3.5-Sonnet (Anthropic, 2024), Gemini-1.5-pro (Team et al., 2024) and other models via API calls. We employ all-MiniLM-L6-v2 based on sentence-transformers (Reimers, 2019) as the embedding model. For each LLM, we randomly select 300 KGs from each datasets (WebQSP, CWQ) to evaluate KARPA's performance, aiming to reduce computational costs. In implementing KARPA, we determine that the initial relation paths planned by the LLM during pre-planning step represent the most reasonable path lengths. Therefore, during the retrieving step, we only extract paths that match the length of the initial paths predicted by the LLM. In the retrieving step based on beam search and pathfinding algorithms, we set the number of top-K paths to 16, selecting 16 paths with the highest semantic similarity for each initial relation path as candidate paths. In the heuristic value-based retrieval step, since our method can compute the similarity between paths of different lengths, we select 16 paths with the highest similarity for each initial path from relation paths of various lengths, which are then used as candidate paths for the reasoning step.

## 5.2 Main Results 5.2.1 Comparison Between Baselines

We evaluate our method against the following approaches: direct answering with GPT-4 (IO prompt), reasoning with internal knowledge (CoT), training-based methods and direct interaction with KGs (ToG). We present the results in Table 1. The results show that our method significantly outperforms existing approaches across most metrics, achieving state-of-the-art performance. When comparing our framework to the direct answering with internal knowledge, we demonstrate that leveraging KGs as external knowledge sources enables the LLM to yield superior answers. In contrast to training-based methods, our approach offers the advantage of being plug-and-play, requiring no additional training while still ensuring effective reasoning based on the KGs. Furthermore, our results indicate that KARPA generalizes well across different KGQA datasets. When comparing with the ToG method, which also utilizes LLMs for reasoning over KGs without ad-

## 5.2.2 Performance Across Different Llms

| Method            | WebQSP   | CWQ   |
|-------------------|----------|-------|
| ToG*              | 11.2     | 14.3  |
| KARPA+GPT-4o-mini | 5.1      | 6.2   |
| KARPA+GPT-4o      | 4.8      | 5.3   |
| KARPA+GPT-4       | 5.5      | 6.0   |
| KARPA+Claude      | 6.6      | 7.3   |
| KARPA+Gemini      | 5.8      | 7.4   |

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 5.3 Further Analysis

We evaluate the average number of interactions required to obtain an answer for both ToG and KARPA across multiple LLMs and datasets. The results, presented in Table 3, show that KARPA In this section, we conduct a deeper analysis of KARPA, exploring two key aspects: (a) the comparison of interaction steps between KARPA and the baseline method ToG, and (b) ablation studies to evaluate the impact of different retrieval methods and LLMs on the performance of KARPA.

## 5.3.1 Interaction Steps Comparison

We also evaluate ToG and KARPA with different LLMs, including GPT-4, GPT-4o, GPT-4-mini, Claude-3.5 - Sonnet, and Gemini-1.5-pro. Both ToG and our KARPA approach rely on the reasoning capabilities of these LLMs without requiring additional training. The results, shown in Table 2, indicate that KARPA consistently outperforms ToG, regardless of the LLM used. This demonstrates that KARPA's ability to harness LLMs' global planning and reasoning capabilities allows it to construct more logically sound and complete reasoning chains, which ultimately lead to more accurate answers. In contrast, ToG's reliance on stepwise relation selection limits its effectiveness, as it neglects the LLM's inherent planning capabilities. Additionally, we evaluate the performance of these LLMs when using CoT prompting. Our results clearly show that when KG information is incorporated, the LLMs are able to provide more accurate and complete answers, further emphasizing the value of external knowledge sources like KGs in enhancing LLM reasoning capabilities.

Table 3: Comparison of LLM call frequency. The LLM call of ToG are cited from its paper.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 consistently reduces the number of interactions by more than half compared to ToG, while maintaining superior performance in terms of answer accuracy and reasoning quality. The primary reason for this efficiency lies in the differences between the interaction mechanisms of the two approaches. In ToG, the stepwise relation selection on KGs is not only time-consuming but also leads to a higher demand for computational resources during interaction with the KG. In contrast, KARPA requires only two interactions with the LLM during the pre-planning step to generate the initial relation paths. These initial paths form a coherent reasoning chain that serves as the backbone for the subsequent retrieval process. Instead of repeatedly invoking the LLM for relation extracting, KARPA leverages an embedding model to extract similar relation paths from the KG based on semantic similarity. This significantly reduces the overall interaction steps and the computational cost of KG-based reasoning.

## 5.3.2 Ablation Studies

We perform two sets of ablation studies to further understand the components of our approach and how they contribute to its effectiveness.

| Method      | WebQSP   | CWQ   |
|-------------|----------|-------|
| GPT-4o-mini | Hit@1    | Hit@1 |
| KARPA-B     | 82.3     | 72.1  |
| KARPA-P     | 82.6     | 71.8  |
| KARPA-H     | 85.3     | 73.3  |
| GPT-4o      | Hit@1    | Hit@1 |
| KARPA-B     | 85.2     | 70.5  |
| KARPA-P     | 86.8     | 74.0  |
| KARPA-H     | 87.7     | 75.3  |

Impact of different retrieval methods. In the retrieving phase of KARPA, we experiment with different methods to extract relation paths and analyze their impact on the final results. The comparison is shown in Table 4, where we evaluate three retrieval strategies: (1) **KARPA-B**: A beam searchbased retrieval method with a fixed beam width to extract relation paths. This method is similar to ToG in that it calculates semantic similarity for paths using stepwise interactions. (2) KARPA-P: A pathfinding-based retrieval method that calculates the semantic similarity between relation paths based on pre-defined distance metrics, constrained to extracting paths of the same length as the initial relation paths. (3) **KARPA-H**: A heuristic value-based retrieval method that is able to compute semantic similarity between paths of different lengths, allowing more flexibility in the candidate path selection process. The results indicate that KARPA-H outperforms other retrieval methods, providing superior KGQA results when using the same LLMs. Additional results are provided in Appendix C. Influence of different LLMs. We also examine how different LLMs affect the performance of our method, as shown in Figure 3. Since KARPA relies on the global planning and reasoning capabilities of LLMs, the strength of the LLM plays a significant role in the overall performance of the KARPA. The results indicate that more powerful LLMs (such as GPT-4) generate better initial paths, leading to more accurate question answering (Kaplan et al., 2020). Conversely, when using the weaker LLM (e.g., GPT-4o-mini), the performance of KARPA slightly declines, though it still outperforms the ToG method. This demonstrates the importance of strong reasoning capabilities in the LLMs for KG-based tasks. The findings also suggest that LLMs with better planning and reasoning abilities can extract more meaningful insights from KGs, thus enhancing overall accuracy of KGQA tasks.

Table 4: Hit@1 value of KARPA with various retrieval strategies.

## 6 Conclusion

In this paper, we propose KARPA, a novel framework designed to enhance LLM-based KGQA by utilizing the global planning and reasoning capabilities of LLMs. KARPA addresses key limitations of existing approaches by improving both accuracy and efficiency, while providing a plug-and-play solution through its structured pre-planning, retrieving, and reasoning processes. Our experiments demonstrate that KARPA consistently outperforms state-of-the-art methods across multiple datasets and evaluation metrics. Furthermore, its training-free nature enables seamless integration with a variety of LLMs, offering broad applicability to different KGQA tasks. By optimizing LLM-KG interactions, KARPA improves reasoning efficiency and effectiveness, highlighting its potential as a robust approach for future retrieval-augmented generation (RAG) systems.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17682–17690, 2024.

Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pp. 1247–1250, 2008.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*, 2024.

Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. In *The Eleventh International Conference on Learning Representations*, 2022.

Tiezheng Guo, Qingwen Yang, Chen Wang, Yanyi Liu, Pan Li, Jiawei Tang, Dapeng Li, and Yingyou Wen. Knowledgenavigator: Leveraging large language models for enhanced reasoning over knowledge graph. *Complex & Intelligent Systems*, 10(5):7063–7076, 2024a.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrievalaugmented generation. *arXiv preprint arXiv:2410.05779*, 2024b.

Hangfeng He, Hongming Zhang, and Dan Roth. Rethinking with retrieval: Faithful large language model inference. *arXiv preprint arXiv:2301.00303*, 2022.

Sebastian Hofstatter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. Effi- ¨
ciently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 113–122, 2021.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *arXiv preprint arXiv:2311.05232*,
2023.

Rikui Huang, Wei Wei, Xiaoye Qu, Wenfeng Xie, Xianling Mao, and Dangyang Chen. Joint multifacts reasoning network for complex temporal question answering over knowledge graph. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 10331–10335. IEEE, 2024.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.

Anthropic. Claude 3.5 sonnet model card addendum, 2024. URL https://www.

paperswithcode.com/paper/claude-3-5-sonnet-model-card-addendum. Accessed: 2024-09-21.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Unikgqa: Unified retrieval and reasoning for solving multi-hop question answering over knowledge graph. arXiv preprint arXiv:2212.00959, 2022.

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt:
A general framework for large language model to reason over structured data. arXiv preprint arXiv:2305.09645, 2023.

Zhanming Jie, Trung Quoc Luong, Xinbo Zhang, Xiaoran Jin, and Hang Li. Design of chain-ofthought in math problem solving. *arXiv preprint arXiv:2309.11054*, 2023.

Hanlei Jin, Yang Zhang, Dan Meng, Jun Wang, and Jinghua Tan. A comprehensive survey on process-oriented automatic text summarization with exploration of llm-based methods. *arXiv* preprint arXiv:2403.02901, 2024.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213, 2022.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨ aschel, et al. Retrieval-augmented genera- ¨ tion for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33: 9459–9474, 2020.

Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao Zhang, and Bing Yin. Graph reasoning for question answering with triplet retrieval. *arXiv preprint* arXiv:2305.18742, 2023a.

Wendi Li, Wei Wei, Xiaoye Qu, Xian-Ling Mao, Ye Yuan, Wenfeng Xie, and Dangyang Chen.

Trea: Tree-structure reasoning schema for conversational recommendation. arXiv preprint arXiv:2307.10543, 2023b.

Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Lidong Bing, Shafiq Joty, and Soujanya Poria. Chain of knowledge: A framework for grounding large language models with structured knowledge bases. *arXiv preprint arXiv:2305.13269*, 2023c.

Haochen Liu, Song Wang, Yaochen Zhu, Yushun Dong, and Jundong Li. Knowledge graphenhanced large language models via path selection. *arXiv preprint arXiv:2406.13862*, 2024.

Antoine Louis, Gijs van Dijck, and Gerasimos Spanakis. Interpretable long-form legal question answering with retrieval-augmented large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 22266–22275, 2024.

Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. *arXiv preprint arXiv:2310.01061*, 2023.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.

Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. Text and code embeddings by contrastive pre-training. *arXiv preprint arXiv:2201.10005*, 2022.

OpenAI. Gpt-4 technical report. Technical report, OpenAI, 2023. URL https://cdn.openai.

com/papers/gpt-4.pdf.

OpenAI. Gpt-4o system card. Technical report, OpenAI, 2024. https://www.openai.com/
research/gpt-4o.

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. *IEEE Transactions on Knowledge and Data* Engineering, 2024.

Reimers. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), *Proceedings of the 2019 Conference on* Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410. URL
https://aclanthology.org/D19-1410.

Sebastian Ruder, Ivan Vulic, and Anders Søgaard. A survey of cross-lingual word embedding mod- ´
els. *Journal of Artificial Intelligence Research*, 65:569–631, 2019.

Yiheng Shu, Zhiwei Yu, Yuhan Li, Borje Karlsson, Tingting Ma, Yuzhong Qu, and Chin-Yew Lin. ¨
TIARA: Multi-grained retrieval for robust question answering over large knowledge base. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 8108–8121, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.

emnlp-main.555. URL https://aclanthology.org/2022.emnlp-main.555.

Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and William W Cohen. Open domain question answering using early fusion of knowledge bases and text. *arXiv preprint arXiv:1809.00782*, 2018.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Heung-
Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model with knowledge graph. *arXiv preprint arXiv:2307.07697*, 2023.

Talmor. The web as a knowledge-base for answering complex questions. *arXiv preprint* arXiv:1803.06643, 2018.

Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions. In Marilyn Walker, Heng Ji, and Amanda Stent (eds.), Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 641–651, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1059. URL https://aclanthology.org/N18-1059.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, and Wenjie Zhang. Paths-over-graph:
Knowledge graph empowered large language model reasoning. *arXiv preprint arXiv:2410.14211*, 2024.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. URL https://arxiv.org/abs/2403.05530.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´
Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and ` efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang Xiong. Knowledge-driven cot: Exploring faithful reasoning in llms for knowledgeintensive question answering. *arXiv preprint arXiv:2308.13259*, 2023.

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. *arXiv preprint* arXiv:2212.09561, 2022.

Guanming Xiong, Junwei Bao, and Wen Zhao. Interactive-kbqa: Multi-turn interactions for knowledge base question answering with large language models. *arXiv preprint arXiv:2402.15131*, 2024.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. *arXiv preprint arXiv:2007.00808*, 2020.

Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, and Enhong Chen. Large language models for generative information extraction: A survey. *arXiv preprint arXiv:2312.17617*, 2023.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.

Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The value of semantic parse labeling for knowledge base question answering. In *Proceedings of the 54th* Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 201–206, 2016.

Donghan Yu, Chenguang Zhu, Yuwei Fang, Wenhao Yu, Shuohang Wang, Yichong Xu, Xiang Ren, Yiming Yang, and Michael Zeng. Kg-fid: Infusing knowledge graph in fusion-in-decoder for open-domain question answering. *arXiv preprint arXiv:2110.04330*, 2021.

Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Yiqun Hu, William Yang Wang, Zhiguo Wang, and Bing Xiang. Decaf: Joint decoding of answers and logical forms for question answering over knowledge bases. In The Eleventh International Conference on Learning Representations, 2022.

Mengqi Zhang, Xiaotian Ye, Qiang Liu, Pengjie Ren, Shu Wu, and Zhumin Chen. Knowledge graph enhanced large language model editing. *arXiv preprint arXiv:2402.13593*, 2024.

Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. Retrieve anything to augment large language models. *arXiv preprint arXiv:2310.07554*, 2023a.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. Siren's song in the ai ocean: A survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023b.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. *arXiv preprint arXiv:2210.03493*, 2022.

Ruilin Zhao, Feng Zhao, Long Wang, Xianzhi Wang, and Guandong Xu. Kg-cot: Chain-of-thought prompting of large language models over knowledge graphs for knowledge-aware question answering. 2024.

Xufeng Zhao, Mengdi Li, Wenhao Lu, Cornelius Weber, Jae Hee Lee, Kun Chu, and Stefan Wermter.

Enhancing zero-shot chain-of-thought reasoning in large language models through logic. *arXiv* preprint arXiv:2309.13339, 2023.

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao, Yixin Ou, Yunzhi Yao, Shumin Deng, Huajun Chen, and Ningyu Zhang. Llms for knowledge graph construction and reasoning: Recent capabilities and future opportunities. *World Wide Web*, 27(5):58, 2024.

## A Algorithm For Karpa

In this section, we present the pseudo-code for the Knowledge graph Assisted Reasoning Path Aggregation (KARPA) framework, as shown in Algorithm 1. The pseudo-code outlines the key components of our approach, including the pre-planning, retrieval, and reasoning phases. It demonstrates the interaction between the large language model (LLM) and the embedding model in generating, retrieving, and refining relation paths, which are crucial for improving LLM-based KGQA tasks.

## Algorithm 1: Karpa Framework

Input: Question Q, Topic entity et, Knowledge Graph KG, Large Language Model LLM,
Embedding Model Output: Answers Ea Pre-Planning Phase:
Generate initial paths Pi = {p1, p2*, . . . , p*m} using LLM(*Q, e*t); for *each path* pi = (r i1, ri2*, . . . , r*ini
) do Decompose piinto relation list Ri = {r i 1, ri2*, . . . , r*ini
};
for *each relation* r ij in Ri do Retrieve top-K similar relations Rij = Top-K(sim(r ij, r));
end end Re-plan relation paths P*replan* = LLM(*Q, R*ij
) based on retrieved relations Rij
;
Retrieving Phase:
Extract relation paths Pr with length L ∈ len(P*replan*); for each path p in P*replan* do Compute similarity between paths using heuristic value P*retrieved* = Heuristic(sim(p, pr), pr ∈ Pr); Retrieve top-K similar paths P = Top-K(Pretrieved) as P*candidate*;
end Reasoning Phase:
Combine candidate relation paths Pcandidate = {r1, r2*, . . . , r*n} with et, ea into prompt; Predict final answer Ea = LLM(Q, Pcandidate, et, ea); return Ea

## B Implementation Details

Model Invocation. Our method, KARPA, along with the baseline comparison methods such as CoT (Wei et al., 2022) and ToG (Sun et al., 2023), is all implemented via API calls to various large language models (LLMs). These LLMs are queried dynamically throughout the experimental pipeline to perform pre-planning, retrieving, and reasoning steps. Experimental Setup. During the pre-planning stage, the initial paths generated by the LLM are decomposed and stored, along with the query, into a list. For each element in this list, we retrieve the top-k relations, where the total number of retrieved relations does not exceed 30. These relations are semantically closest to the elements based on the LLM's initial output. In the retrieving step, KARPA selects the top 16 relation paths with the highest similarity for each initial relation path. These paths serve as candidate paths for reasoning step. In the reasoning step, we limit the number of candidate paths input to the LLM at one time to a maximum of 8, ensuring that the reasoning process remains manageable and focused on the most relevant paths.

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863

| WebQSP            |         |          |       |      |           |
|-------------------|---------|----------|-------|------|-----------|
| Model Tpye        | Method  | Accuracy | Hit@1 | F1   | Precision |
| KARPA-B           | 67.2    | 82.3     | 61.5  | 64.1 |           |
| GPT-4o-mini       | KARPA-P | 67.8     | 82.6  | 62.4 | 64.9      |
| KARPA-H           | 71.9    | 85.3     | 64.5  | 65.9 |           |
| KARPA-B           | 73.8    | 85.2     | 67.3  | 72.3 |           |
| GPT-4o            | KARPA-P | 73.7     | 86.8  | 69.7 | 70.5      |
| KARPA-H           | 76.1    | 87.7     | 69.2  | 71.5 |           |
| KARPA-B           | 73.5    | 85.5     | 68.4  | 71.7 |           |
| GPT-4             | KARPA-P | 74.1     | 86.8  | 69.3 | 73.6      |
| KARPA-H           | 80.9    | 91.2     | 72.1  | 73.1 |           |
| KARPA-B           | 71.8    | 84.0     | 63.1  | 65.9 |           |
| DeepSeek-V2.5     | KARPA-P | 73.4     | 85.3  | 64.1 | 66.3      |
| KARPA-H           | 78.1    | 88.4     | 68.7  | 67.6 |           |
| KARPA-B           | 70.1    | 84.5     | 65.9  | 64.7 |           |
| Gemini-1.5-Pro    | KARPA-P | 73.8     | 88.0  | 67.4 | 66.1      |
| KARPA-H           | 80.7    | 90.5     | 68.6  | 67.8 |           |
| KARPA-B           | 75.1    | 85.7     | 66.0  | 67.6 |           |
| Claude-3.5-Sonnet | KARPA-P | 80.4     | 89.0  | 69.7 | 70.4      |
| KARPA-H           | 82.6    | 89.5     | 69.7  | 69.1 |           |

Answer Evaluation. To determine if the LLM correctly answers the question, KARPA enforces a specific output format. The final answer must be enclosed in curly brackets in the LLM's output. We consider an answer correct only when the tail entities of the reasoning paths match the text enclosed within the curly brackets in the LLM's output. For CoT, we consider an answer correct if the LLM's response contains the correct answer entities. This difference reflects the distinct reasoning and output expectations between KARPA and CoT.

## C Additional Results

In this section, we present additional experimental results to further evaluate the performance of KARPA when using different retrieval methods: KARPA-B (beam search-based retrieval), KARPA- P (pathfinding-based retrieval), and KARPA-H (heuristic value-based retrieval). We conduct these experiments across various LLMs, analyzing the effectiveness of each retrieval strategy in conjunction with different LLMs. These results provide a deeper insight into how different retrieval mechanisms impact the overall performance of KARPA, showcasing the versatility and adaptability of our approach under varying model conditions. The results presented in Table 5 and Table 6 consistently demonstrate the superior performance of KARPA-H (heuristic value-based retrieval) compared to the other two retrieval strategies, KARPA-B (beam search-based) and KARPA-P (pathfinding-based), across different LLMs and datasets (WebQSP and CWQ). In the majority of LLMs, KARPA-H outperforms the other methods in most metrics. This suggests that KARPA-H is more effective at extracting the correct relation paths, which in turn leads to more accurate and contextually relevant answers. These results highlight KARPA-H as the most robust and reliable retrieval method among the three, reinforcing its advantage in handling complex KG- based reasoning tasks.

## D Additional Experiments

864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917

| CWQ               |         |          |       |      |           |
|-------------------|---------|----------|-------|------|-----------|
| Model Tpye        | Method  | Accuracy | Hit@1 | F1   | Precision |
| KARPA-B           | 66.0    | 72.1     | 57.8  | 58.6 |           |
| GPT-4o-mini       | KARPA-P | 66.4     | 71.7  | 58.7 | 59.8      |
| KARPA-H           | 68.1    | 73.3     | 56.5  | 55.1 |           |
| KARPA-B           | 65.0    | 70.5     | 55.8  | 57.8 |           |
| GPT-4o            | KARPA-P | 69.2     | 74.1  | 59.8 | 58.4      |
| KARPA-H           | 69.8    | 75.3     | 58.4  | 59.5 |           |
| KARPA-B           | 71.2    | 75.4     | 61.1  | 62.7 |           |
| GPT-4             | KARPA-P | 73.4     | 77.9  | 63.0 | 62.5      |
| KARPA-H           | 73.6    | 78.4     | 61.5  | 63.1 |           |
| KARPA-B           | 61.6    | 63.2     | 48.4  | 50.1 |           |
| DeepSeek-V2.5     | KARPA-P | 60.9     | 63.0  | 51.8 | 52.6      |
| KARPA-H           | 62.6    | 64.1     | 51.9  | 53.5 |           |
| KARPA-B           | 69.1    | 74.0     | 57.2  | 59.5 |           |
| Gemini-1.5-Pro    | KARPA-P | 69.6     | 73.5  | 57.7 | 60.3      |
| KARPA-H           | 69.8    | 75.0     | 54.8  | 55.8 |           |
| KARPA-B           | 62.8    | 65.7     | 49.6  | 52.1 |           |
| Claude-3.5-Sonnet | KARPA-P | 61.5     | 64.3  | 52.9 | 55.5      |
| KARPA-H           | 70.6    | 73.7     | 54.9  | 56.9 |           |

Table 6: Performance of KARPA with different retrieval strategies (KARPA-B, KARPA-P, and KARPA-H) and LLMs on the CWQ dataset. In this section, we provide additional experiments to validate KARPA's performance from different perspectives.

To demonstrate that KARPA has better generalization capabilities than methods based on instruction-tuned LLMs, we conducted an experiment using GPT-4o-mini with a modified version of the WebQSP dataset. Specifically, we slightly alter the questions in WebQSP dataset while preserving their original meaning, using the prompt: "Please revise the question to make it more clear, but the original meaning of the question and the corresponding answers remain unchanged." We test RoG using its instruction-tuned LLaMa2-Chat-7B from in the planning step and GPT-4o-mini for reasoning. In KARPA, we use GPT-4o-mini for both pre-planning and reasoning steps.

Question Method Accuracy Hit@1 F1 Method Accuracy Hit@1 F1

Origin RoG 67.6 84.1 69.7 KARPA 73.1 85.4 68.1

Revised RoG 63.5 74.3 64.1 KARPA 72.6 84.5 68.9

Variation RoG -4.1 -9.8 -5.6 KARPA -0.5 -0.9 +0.8

Table 7: Comparison of RoG and KARPA on the WebQSP dataset with original and revised questions. The results in Table 7 show that KARPA's performance remains consistent and robust to question modifications, while RoG's performance drops due to path mismatches. This further highlights the advantage of KARPA's training-free framework, maintaining superior robustness and adaptability across all KGs.

We also conduct an additional experiment using instruction-tuned LLaMa2-Chat-7B as the backbone LLM for both KARPA and RoG, while using untrained Qwen2.5-7B and Qwen2.5-14B for final answer reasoning in both methods. The results in Table 8 show that with the same backbone LLM, KARPA's semantic similarity-based retrieval methods successfully extract more accurate reasoning paths, leading to higher accuracy in final answers.

| WebQSP                     | CWQ      |       |      |          |       |      |
|----------------------------|----------|-------|------|----------|-------|------|
| Embedding Model            | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| all-MiniLM-L6-v2           | 72.3     | 86.4  | 67.2 | 64.6     | 67.7  | 55.1 |
| all-mpnet-base-v2          | 74.5     | 86.1  | 68.6 | 64.1     | 68.3  | 53.7 |
| multilingual-MiniLM-L12-v2 | 74.1     | 85.3  | 68.3 | 65.3     | 69.5  | 55.4 |

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 In Table 9, 1-hop and 2-hop represent the F1 scores on the WebQSP dataset for KG with reasoning paths of length 1 and length 2, respectively. Overall refers to the overall F1 score on the WebQSP dataset. Random Hit@1 (RHit@1) is calculated following the method used in TIARA (Shu et al., 2022), where an answer is randomly selected for each question 100 times, and the average Hits@1 is reported. Overall (CWQ) represents the overall F1 score on the CWQ dataset. The results show that KARPA outperforms Interactive-KBQA on WebQSP and CWQ datasets with GPT-4-turbo. To demonstrate the impact of different embedding models on KARPA, we conduct additional experiments comparing various embedding models to evaluate their effects on KARPA's performance when using GPT-4o-mini. Table 10: Performance comparison of different embedding models on WebQSP and CWQ datasets. In Table 10, all-MiniLM-L6-v2 is the default embedding model used in KARPA, with a size of approximately 86MB. all-mpnet-base-v2, a more powerful embedding model, is around 417MB. paraphrase-multilingual-MiniLM-L12-v2, which supports embedding between multiple languages, has a size of approximately 448MB. The results demonstrate that KARPA's robust design ensures that its overall performance remains consistent across different embedding models. This is because the candidate paths generated by KARPA during the pre-planning phase are very distinct. While they are semantically close to the correct reasoning paths, they differ significantly from incorrect reasoning paths. Therefore, a basic embedding model is sufficient to assist KARPA in extracting the correct paths. Table 8: Comparison of RoG and KARPA performance on WebQSP and CWQ datasets using instruction-tuned LLaMa2-Chat-7B as the backbone LLM. We also compare KARPA with Interactive-KBQA (Xiong et al., 2024), a robust agent-like method which directly perform inference over KGs with LLMs. Interactive-KBQA shares similarities with ToG as both approaches rely on direct, step-by-step interaction between LLMs and KGs to infer answers. In contrast, KARPA eliminates the need for iterative interaction by directly generating a complete reasoning path based on relations extracted from the KG. Our approach significantly reduces the computational cost for LLMs and improves the logical coherence of reasoning paths. To further substantiate KARPA's advantages, we conduct an additional experiment comparing KARPA with Interactive-KBQA, using GPT-4-turbo as the backbone LLM. The results of Interactive-KBQA are cited from its paper. Table 9: Comparison of Interactive-KBQA and KARPA performance on WebQSP and CWQ datasets.

| Method           | 1-hop   | 2-hop   | Overall   | RHits@1   | Overall (CWQ)   |
|------------------|---------|---------|-----------|-----------|-----------------|
| Interactive-KBQA | 69.99   | 72.41   | 71.20     | 72.47     | 49.07           |
| KARPA            | 74.21   | 72.97   | 73.78     | 74.14     | 61.45           |

| WebQSP                  | CWQ    |          |       |      |          |       |      |
|-------------------------|--------|----------|-------|------|----------|-------|------|
| Base-model              | Method | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| LLaMa2-7B + Qwen2.5-7B  | RoG    | 54.5     | 73.8  | 57.2 | 38.6     | 43.5  | 35.8 |
| KARPA                   | 66.4   | 82.7     | 63.6  | 54.1 | 59.2     | 46.3  |      |
| LLaMa2-7B + Qwen2.5-14B | RoG    | 58.7     | 77.2  | 60.9 | 43.9     | 48.0  | 42.5 |
| KARPA                   | 69.8   | 84.2     | 67.4  | 55.0 | 60.4     | 47.2  |      |

972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 To demonstrate the effectiveness of KARPA with smaller LLMs, we conduct additional experiments with Qwen2.5-7B and Qwen2.5-14B as the LLM backbones for KARPA. The results in Table 12 demonstrate that KARPA consistently outperforms stepwise direct inference baselines such as ToG, even when using smaller LLMs. This reinforces the robustness and adaptability of our method across different LLM scales.

| WebQSP      | CWQ    |          |       |      |          |       |      |
|-------------|--------|----------|-------|------|----------|-------|------|
| Base-Model  | Method | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| CoT         | -      | 41.5     | -     | -    | 28.3     | -     |      |
| Qwen2.5-7B  | ToG    | 24.6     | 30.2  | 21.9 | 22.4     | 25.8  | 20.2 |
| KARPA       | 65.6   | 79.2     | 58.6  | 47.6 | 52.7     | 38.8  |      |
| CoT         | -      | 49.6     | -     | -    | 31.2     | -     |      |
| Qwen2.5-14B | ToG    | 45.0     | 55.9  | 42.7 | 31.2     | 36.6  | 29.5 |
| KARPA       | 72.6   | 84.1     | 65.0  | 51.5 | 57.9     | 41.6  |      |

Table 12: Performance comparison of different methods on WebQSP and CWQ datasets using smaller LLMs. Also, the results in Table 12 show that KARPA can perform well with LLMs that have weaker planning and reasoning capabilities, further highlighting KARPA's robustness and its reduced dependence on the LLM's planning and reasoning abilities compared to other inference-based methods. To quantify the impact of the re-planning step, we provide an ablation study that removes the replanning step from the pre-planning stage. The re-planning step is designed to handle mismatches between LLMs and KGs. In re-planning step, the extracted relations are used to refine and re-plan candidate paths. This guarantees that the candidate paths are both logically coherent and aligned with the KG.

| WebQSP                  | CWQ      |       |      |          |       |      |
|-------------------------|----------|-------|------|----------|-------|------|
| Pre-Planning            | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| Origin                  | 72.3     | 86.4  | 67.2 | 64.6     | 67.7  | 55.1 |
| Remove Re-Planning Step | 64.1     | 79.6  | 61.5 | 54.3     | 59.5  | 47.1 |

Table 13: Ablation study of removing re-planning step from the pre-planning stage.

The results in Table 13 show that the re-planning step is crucial for KARPA's performance. Additionally, in the retrieval step, KARPA employs semantic similarity as the cost function for pathfinding algorithms. This ensures that the final reasoning paths selected not only exist in the KG but are also semantically closest to the paths generated by the LLM, thereby maintaining the validity of the LLM's output across diverse query problems. To demonstrate that KARPA reduces the logical complexity of LLM reasoning on KGs, we provide a comparison of the average number of input and output tokens between ToG and KARPA using the Table 11: Exact Match (EM) performance comparison between ToG and KARPA on WebQSP and CWQ datasets. We also provide the Exact Match (EM) metric (Talmor & Berant, 2018) for a more comprehensive analysis. The results in Table 11 demonstrate that KARPA achieves higher EM scores compared to ToG, showing its effectiveness in accurately extracting reasoning paths and final answers.

| Base-Model   | Method   | EM (WebQSP)   | EM (CWQ)   |
|--------------|----------|---------------|------------|
| GPT-4o       | ToG      | 39.5          | 37.6       |
| GPT-4o       | KARPA    | 44.6          | 41.3       |
| GPT-4        | ToG      | 43.1          | 40.9       |
| GPT-4        | KARPA    | 51.7          | 47.2       |

1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 tokenizer of GPT-4o-mini. Methods that rely on step-by-step interactions between the LLM and KG must select the next relations from hundreds or even thousands of adjacent relations at each step, and repeat this process until the answer entities are found. This results in a high computational burden, and also fails to leverage the LLM's global planning capabilities.

| WebQSP   | CWQ             |                  |                 |                  |
|----------|-----------------|------------------|-----------------|------------------|
| Method   | Input Tokens/KG | Output Tokens/KG | Input Tokens/KG | Output Tokens/KG |
| ToG      | 6351.5          | 1836.5           | 7935.7          | 2931.6           |
| KARPA    | 2465.9          | 1492.3           | 3612.1          | 2267.1           |

Table 14: Token usage comparison between ToG and KARPA on WebQSP and CWQ datasets.

The results in Table 14 show that KARPA significantly reduces both input and output token usage compared to ToG, which means we have not only lowered the reasoning complexity for the LLM but also saved on the computational costs of the LLM, further demonstrating the superiority of KARPA. The multilingual scenarios can be effectively addressed by using multilingual embedding models. For instance, in a multilingual setting, we test KARPA with paraphrase-multilingual-MiniLM-L12v2, a multilingual embedding model. In the multilingual experiment, we use GPT-4o-mini to generate relation paths in Chinese, and then use the multilingual embedding model to calculate the semantic similarity between the candidate paths and paths in the KG.

| WebQSP          | CWQ      |       |      |          |       |      |
|-----------------|----------|-------|------|----------|-------|------|
| Language        | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| English-English | 74.1     | 85.3  | 68.3 | 65.3     | 69.5  | 55.4 |
| Chinese-English | 74.6     | 84.5  | 67.6 | 63.1     | 68.0  | 54.2 |

Table 15: Performance comparison of different languages using a multilingual embedding model.

These results in Table 15 demonstrate that with a multilingual embedding model, KARPA performs effectively across languages, maintaining its robustness. They also indicate that language variations do not significantly impact KARPA's performance. To demonstrate the necessity of extending relation paths with different lengths, we restrict the retrieval step to use only single-relation candidate paths provided by the LLM during re-planning step, and compare the performance of the heuristic value-based retrieval method (KARPA-H) with the pathfinding-based retrieval method (KARPA-P) using GPT-4o-mini.

| WebQSP                | CWQ     |          |       |      |          |       |      |
|-----------------------|---------|----------|-------|------|----------|-------|------|
| Candidate Path        | Method  | Accuracy | Hit@1 | F1   | Accuracy | Hit@1 | F1   |
| Original Paths        | KARPA-P | 66.0     | 81.2  | 63.8 | 61.0     | 64.5  | 53.4 |
| Original Paths        | KARPA-H | 72.3     | 86.4  | 67.2 | 64.6     | 67.7  | 55.1 |
| Single-Relation Paths | KARPA-P | 63.6     | 77.3  | 60.7 | 40.5     | 43.9  | 39.3 |
| Single-Relation Paths | KARPA-H | 71.4     | 85.5  | 68.9 | 55.1     | 59.6  | 47.4 |

Table 16: Performance of KARPA-P and KARPA-H using different candidate paths on the WebQSP and CWQ datasets. The results in the Table 16 demonstrate that the heuristic value-based retrieval method outperforms pathfinding-based retrieval methods in such scenarios, as it effectively addresses the semantic similarity issues that arise from differing path lengths. Moreover, as the questions in the CWQ dataset generally require longer reasoning paths compared to WebQSP, both methods exhibit a more significant decline in various metrics on CWQ. However, the heuristic value-based retrieval method shows a less pronounced drop compared to pathfinding-based retrieval methods, further demonstrating its superiority. To validate the performance of KARPA on KGs outside the training scope, we compare KARPA with Chain-of-Thought (CoT) reasoning, where the LLM directly relies on its internal knowledge to an-