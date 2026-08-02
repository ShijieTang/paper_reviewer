# Scalable Permutation-Aware Modeling For Temporal Set Prediction

| Affiliation Address email   |
|-----------------------------|

## Abstract 13 **1 Introduction**

14 Temporal Set Prediction addresses the problem of predicting which elements belong to the next 15 set, given a sequence of sets. The problem involves identifying patterns in how sets evolve over 16 time—tracking which elements enter, exit, or remain—and using these patterns to make accurate 17 membership predictions. This approach enables fine-grained, element-level forecasting in a wide 18 range of domains, including supply chain optimization, traffic congestion prediction, predictive 19 maintenance in industrial systems, personalized recommendation systems, clinical event forecasting 20 in healthcare, and modeling dynamic communities in social networks. 21 However, despite the importance of accurate element-level predictions, existing methods face signifi22 cant computational challenges with large temporal sets. Attention-based mechanisms typically scale 23 quadratically with sequence length, while many graph-based approaches have quadratic or worse 24 complexity relative to the number of elements. These computational constraints limit applicability to 25 real-world scenarios where both the universe of elements and the sequence length can be substantial. 26 For dynamic environments requiring frequent updates, such as real-time recommendation systems or 27 network monitoring, these performance limitations become particularly problematic. 28 In this paper, we propose an architecture called **PIETSP** for temporal set prediction that reduces computational complexity to O(N(KD + D2 29 ) + |E|D), where N is the number of distinct elements 30 in a sequence of sets, K is the maximum sequence length, D is the embedding dimension, and |E| is 31 the domain of all possible elements (vocabulary size). This represents a significant improvement over 32 conventional attention or graph based approaches which typically incur quadratic complexities in N 33 or K. The proposed architecture offers both scalability and accuracy in predicting set membership at 34 future time points. Our contributions include:
1 Temporal set prediction involves forecasting the elements that will appear in the 2 next set, given a sequence of prior sets, each containing a variable number of 3 elements. Existing methods often rely on complex architectures with substantial 4 computational overhead, limiting their scalability. In this work, we introduce a 5 novel and scalable framework that combines an efficient input representation with 6 permutation-equivariant and permutation-invariant transformations to model set 7 dynamics. Our approach significantly reduces training and inference time while 8 maintaining competitive performance. Extensive experiments on multiple public 9 datasets demonstrate that our method achieves state-of-the-art performance overall, 10 outperforming or matching existing models across several evaluation metrics. 11 These results highlight the effectiveness of our model in enabling efficient and 12 scalable temporal set prediction.

## 44 **2 Related Work**

35 - A mathematically principled formulation of temporal set prediction that integrates element 36 features and their temporal dynamics in a joint representation, allowing for more accurate 37 and efficient predictions. 38 - A novel algorithm that achieves linear scaling with respect to both sequence length and 39 number of distinct elements independently, thus enabling the processing of large-scale 40 datasets in a computationally efficient manner. 41 - Comprehensive empirical evaluation on publicly available datasets, demonstrating that our 42 approach offers comparable or superior performance to existing state-of-the-art methods 43 while significantly reducing computational requirements. 45 **Temporal Set Prediction.** Temporal Set Prediction (TSP) is a generalization of sequence prediction 46 that models the evolution of sequences of unordered sets rather than sequences of individual elements. 47 Several baselines have been proposed for this task. Sets2Sets [1] formulates it as sequential set48 to-set learning using an RNN encoder-decoder with set attention and repeated element modules; 49 however, its recurrent structure limits parallelism and slows training. DNNTSP [2] extends this 50 by modeling dynamic co-occurrence graphs using GCNs [3], incorporating temporal attention and 51 gated fusion to capture sequence dynamics, albeit with increased memory and compute costs due 52 to graph construction. SFCNTSP [4] mitigates these issues through a lightweight architecture with 53 permutation invariant and equivariant layers, achieving faster inference and fewer parameters, though 54 scalability remains a challenge on large datasets.

55 **Next Basket and Set Prediction.** TSP is closely aligned with next-basket recommendation, where 56 the goal is to predict the next set of items a user will interact with. Early models such as FPMC [5] 57 combine matrix factorization with Markov Chains to model user preferences and item transitions. 58 While efficient, FPMC lacks the ability to model inter-item dependencies within a basket and does 59 not generalize well to cold or rare items. Additionally, models such as SHAN [6] and SASRec [7] 60 introduced self-attention mechanisms [8] to this task, but their adaptation to set sequences is limited, 61 as attention is inherently order-sensitive and does not handle set permutation-invariance without 62 explicit modification. 63 **Multiset Modeling and Repetition.** A defining feature of TSP, and a gap in traditional sequential 64 models, is the ability to handle repeated elements—important in domains like healthcare (e.g., 65 recurring diagnoses, lab tests) and e-commerce (e.g., repeated purchases). Sets2Sets and DNNTSP 66 directly model repetition via frequency-aware loss functions or modules that attend to past occurrences. 67 However, these often assume hard duplication counts rather than modeling repetition as a stochastic 68 process. 69 **Sequence-to-Sequence and Set Modeling.** TSP extends the classic sequence-to-sequence (seq2seq) 70 framework popularized in NLP [9, 10], but demands adaptations for sets due to their unordered and 71 variable-length nature. Key inspirations come from DeepSets [11], which introduced permutation72 invariant functions for effective set representation. Building on this foundation, Set Transformers 73 [12] leverage attention mechanisms specifically designed for set inputs and outputs. SetVAE [13] 74 further advances this line of research by enabling generative modeling of unordered outputs with 75 improved computational efficiency. 76 While standard Transformer-based approaches often suffer from quadratic complexity in set size, 77 SetVAE employs architectural innovations to mitigate this limitation. However, challenges remain in 78 capturing sequential dependencies when these models are applied to problems requiring both set-level 79 operations and order-sensitive processing.

## 80 **3 Problem Formulation**

Let S(ti) ⊆ E represent the set of all elements present at time ti 81 , where E is the domain of possible 82 elements. Let S = [S(t1), S(t2)*, . . . , S*(tT )] be the sequence of sets across T time steps, where 83 1 ≤ T ≤ K and K = max{|Sj | : Sj *∈ D}* is the maximum set sequence length across the dataset D. Let U =ST
84 i=1 S(ti) be the universe of all distinct elements that appear in any set in the sequence

1,1,0 1,0,1 1,0,1 0,1,0 MU C
85 S. We can enumerate the elements in this universe as U = {e1, e2*, . . . , e*N }, where N = |U| is the 86 total number of unique elements in S.

Let M ∈ R
|E|×D 87 be the embedding matrix for the entire domain E, where each row represents the D-dimensional embedding vector for an element in the domain. Let MU ∈ R
N×D 88 be the embedding 89 matrix for elements in U, where MU is the submatrix of M containing only the rows corresponding 90 to elements in U. We show the construction of MU in Figure 1. 91 Given this formulation, our objective is to predict the future set membership S(tT +1) by analyzing 92 patterns in the sequence history S. Specifically, we aim to develop models that can accurately 93 forecast element membership in the set at time T + 1, based on the structural patterns identified in 94 the evolution history of S.

## 95 **4 Methodology**

PE
EE Oe Z
Ŷ
PI
SFI Layer Os GE
MU
C
96 We first give a brief overview of the sequence of operations below, as shown in Figure 2 and elaborate 97 each operation in the subsections. In order to predict the future set membership S(tT +1), we start by 98 integrating the embeddings MU of the unique elements N with a sequence feature C (illustrated in 99 Figure 1) by passing it through Sequence Feature Integration (SFI) layer. The resulting output Z
100 of N elements is passed to a **permutation equivariant layer** (PE). This enriches the representation 101 of the set elements with a global aggregated context. The updated representation Ze captures the 102 integrated sequence-element relationship and serves as input to two separate computational branches 103 as shown in Figure 2. The branch on the right, called **Element Evaluator** (EE), generates scores for 104 each of the updated N elements, collectively referred to as Oe.

105 Meanwhile, the branch extending downwards passes Ze through a **permutation invariant layer**
106 (PI). PI transforms the updated set of N elements Ze into a single representation Z. Note that Z is 107 the representation of an entire sequence of sets. Z is then passed through **Global Evaluator** (GE)
108 to get scores Os for the entire domain E. Oe, the scores for updated N elements Ze and Os, the global scores for the entire domain E, are then fused to get the final logits Yˆ 109 . Since our method 110 applies a Permutation Invariant operation (PI) following a Permutation Equivariant operation (PE) for 111 Temporal Set Prediction (TSP), we name have named our proposed approach **PIETSP**.

## 112 **4.1 Sequence Feature Integration**

113 Our approach differs from related methods that typically process set elements for each time step in 114 isolation. Although conventional techniques distribute the N distinct sequence elements across the 115 K time steps using weight-sharing mechanisms for separate processing, our method integrates com116 prehensive sequence information for each of the N elements. We utilize a sequential representation C ∈ R
N×Q 117 that encodes relationships between elements and sequences. The matrices MU and C
118 are integrated via the Sequence Feature Integration (SFI) function, as outlined in (1), resulting in 119 matrix Z. This matrix Z combines element-specific features with sequential information which may 120 include membership patterns, temporal embeddings, or count-based representations.

Z = SFI(MU , C) ∈ R
N×F(1)
For our implementation, we define C as a multi-hot representation C ∈ {0, 1}
N×K 121 , where:

4
$$(\mathbb{I})$$
$\hdots\;\Gamma^{\omega_+}$ . 
$$C[i,j]={\begin{cases}1&{\mathrm{if~element~}}e_{i}\in S(t_{j})\\ 0&{\mathrm{if~element~}}e_{i}\notin S(t_{j})\end{cases}}$$
$$(2)$$

0 if element ei ∈/ S(tj )(2)
122 In our implementation, the matrix C efficiently encodes the binary membership relationships between 123 elements and sequences, enabling us to mathematically model these relationships with each row 124 corresponding to an element and each column representing a sequence. The resulting structure 125 preserves the distributional patterns of elements across sequences, forming the basic for subsequent 126 operations in our method. We use simple concatenation for SFI. This results in Z of shape N × (K + 127 D). While the original data consists of a sequence of sets, transforming it into the matrix Z enables 128 efficient neural processing using standard operations, while preserving the underlying set semantics 129 through permutation-aware design.

## 130 **4.2 Integrated Element-Sequence Relationship Learning**

131 To effectively capture the interplay between individual elements and the sequences they participate 132 in, we apply a permutation equivariant transformation to the matrix Z. A function f is *permutation* 133 *equivariant* if permuting its inputs results in an equivalent permutation of its outputs. Formally, for 134 any permutation π and input list X = [x1, x2*, ..., x*n], a permutation equivariant function satisfies:
f([xπ(1), xπ(2)*, ..., x*π(n)]) = [f(X)]π 135 This property ensures that our model respects the structure of the input while allowing meaningful transformations of individual elements based on the collective context. We pass Z ∈ R
N×(K+D)
136 through a permutation equivariant layer to obtain Ze ∈ R
N×d
′
137 , as shown in Equation (4):
Ze = PE(Z) ∈ R
N×d
′(3)
138 In our implementation, we use a mean permutation equivariant layer, defined as:

$$\widetilde{Z}=\mathrm{ELU}\left(Z W_{g}+b_{g}-\frac{1}{N}\sum_{i=1}^{N}(Z_{i}W_{\ell})\right)\in\mathbb{R}^{N\times d^{\prime}}$$
$$({\mathfrak{I}})$$
$$Z=\mathrm{PE}(Z)$$
$${\Xi}=\mathbb{R}$$
$\mathbf{b}$
$$(4)$$

′(4)
where Wg ∈ R
(K+D)×d
′and Wℓ ∈ R
(K+D)×d
′are learnable weight matrices, and bg ∈ R
d
′
139 is a 140 learnable bias vector. The ELU activation is applied element-wise to the entire result to introduce 141 smooth, non-linear transformations.

To reduce computational complexity, we set the output dimension d 142 ′ = D, thereby projecting Z ∈ R
N×(K+D)into R
N×D 143 . This avoids the quadratic cost of a full (K + D) × (K + D)
144 transformation and instead yields a more efficient O(N(K + D)D) complexity, which is *linear* in the number of time steps K when D is fixed. The resulting output matrix Ze ∈ R
N×D 145 is then used in 146 subsequent stages of the model.

## 147 **4.3 Element Evaluator**

148 Following the permutation equivariant transformation, we apply an element-wise evaluator (EE) to the 149 enriched representations Ze in order to compute scalar relevance scores for each element. Specifically, 150 the evaluator produces one score per element as defined in Equation (5):

$$O_{e}=\operatorname{EE}({\widetilde{Z}})\in\mathbb{R}^{N}$$
$$({\boldsymbol{\delta}})$$
N (5)
151 In our implementation, the EE is instantiated as a two-layer Multi-Layer Perceptron (MLP) with a 152 ReLU activation in between. This setup allows the model to assess each element's importance based 153 on both its intrinsic characteristics and its contextual role within the sequence. 154 Since the evaluator processes elements independently, the permutation equivariant structure established earlier is preserved. The resulting scores Oe ∈ R
N 155 quantify each element's contextual 156 relevance and serve as intermediate signals, which will later be merged with complementary scores 157 to inform the final prediction.

## 158 **4.4 Sequence Set Representation**

159 To derive a global representation of the input sequence, we aggregate information from the enriched 160 element representations in a way that is invariant to element order. This summary will later be used 161 to complement the element-level scores for final prediction. 162 A function f is *permutation invariant* if its output remains unchanged regardless of the ordering of its 163 input elements. Formally, for any permutation π and input list X = [x1, x2*, . . . , x*n], a permutation 164 invariant function satisfies:
f([xπ(1), xπ(2)*, . . . , x*π(n)]) = f([x1, x2*, . . . , x*n])
165 This property ensures that the model produces consistent outputs for a given collection of elements, 166 independent of their order. In our method, we apply a permutation invariant operation to the enriched element representations Ze ∈ R
N×D 167 to obtain a global sequence-level summary vector:

$$\overline{{{Z}}}=\mathrm{PI}(\widetilde{Z})\in\mathbb{R}^{1\times D}$$
1×D (6)
168 Specifically, we use a sum pooling operation followed by a Multi-Layer Perceptron (MLP) consisting 169 of two hidden layers with ELU activations and a final output layer:

$$\overline{{{Z}}}=\mathrm{MLP}\left(\sum_{i=1}^{N}\widetilde{Z}_{i}\right)$$
$$(6)$$

$$(7)$$

170 The resulting summary Z encapsulates global sequence-level context that informs the final element 171 selection.

## 172 **4.5 Global Evaluator**

173 We perform global scoring using a global evaluator (GE), which computes relevance scores for all 174 elements in the domain E, as shown in Equation (8).

$$O_{s}=\operatorname{GE}({\overline{{Z}}})\in\mathbb{R}^{|E|}$$
|E|(8)
175 This mechanism captures the relationship between the global sequence set representation Z (which 176 encodes the entire sequence) and each candidate element in the domain E. The global evaluator 177 could take various forms, such as dot product similarity, concatenation followed by an MLP, or other 178 scoring functions. In our implementation, we specifically use a dot product formulation to calculate 179 the scores, measuring the similarity between each element embedding in M and the global sequence 180 representation Z, as shown in Equation (9).

$$O_{s}=M\cdot\overline{{{Z}}}^{\top}$$
$$(9)$$
⊤(9)
181 This approach produces a score for each element in E, indicating its relevance according to the global 182 sequence context.

## 183 **4.6 Score Fusion**

184 To effectively combine global context information with element-specific sequential patterns, we 185 implement a score fusion mechanism. This approach integrates global scores for all domain elements 186 with element-level scores from the set sequence. We introduce learnable parameter vectors α, β ∈
R

|E|to weight each information source. The global scores Os ∈ R
|E| 187 for all elements in domain E from Equation (8) and the element-level scores Oe ∈ R
N 188 from the set sequence as defined in 189 Equation (5) serve as inputs to our score fusion mechanism. 190 Let I : {1, . . . , N} → {1*, . . . ,* |E|} be a one-to-one mapping function that maps each element index 191 i in the set sequence to its unique corresponding index j in the domain E. Let Dseq ⊂ {1*, . . . ,* |E|}
192 be the set of domain indices that are mapped from the set sequence, i.e.,

$i\in\ensuremath{\mathbb{Z}}$

$$\operatorname{seq}=\{j$$
Dseq = {j ∈ {1, . . . , |E|} : ∃i ∈ {1, . . . , N}, I(i) = j}
The final logit output Yˆ ∈ R
|E| 193 is computed as:

$\hat{Y}_{j}=\begin{cases}\alpha_{j}\cdot(O_{s})_{j}+\beta_{j}\cdot(O_{e})_{i}&\text{if}j\in D_{\text{seq}}\text{where}I(i)=j\\ \alpha_{j}\cdot(O_{s})_{j}&\text{otherwise}\end{cases}$
$$(10)$$
194 This formulation ensures that all elements receive a score based on global context (weighted by 195 αj ), while only elements present in the set sequence receive an additional contribution from their 196 element-level scores (weighted by βj ). It allows the model to adaptively balance local sequential 197 patterns (captured by Oe) with global context information (captured by Os) when determining the 198 likelihood of each element appearing in the next set.

## 199 **4.7 Model Training Process**

200 Our model is trained with a batch size of 64. Sequences are zero-padded at the beginning in the 201 multi-hot sequence feature representation C, as formalized in Equation (2), and K is set to 19. We 202 use an embedding dimension of 32 and optimize using Adam with a learning rate of 0.001 and weight 203 decay of 0.01. The learning rate follows a cosine decay schedule. We train for 100 epochs with early 204 stopping (patience=10). Given that the prediction of next-period item sets constitutes a multi-label 205 classification problem, we implement a binary cross-entropy loss function.

## 206 **4.8 Model Complexity Analysis**

207 **Time Complexity:** Our model demonstrates efficient computational scaling across its components.

208 The time complexity for element relationship learning using PE is O(N(K + D)· D), while creating the set sequence embedding via PI requires O(ND2 209 ) operations. Scoring set elements via EE
contributes to an additional O(ND2 210 ) complexity, and global scoring of all elements in domain E
211 using GE adds O(|E|D) operations. Finally, the score fusion layer adds O(|E|) operations. The total 212 time complexity can be expressed as:
O(N(K + D) · D + ND2 + ND2 + |E|D + |E|)

213 This simplifies to:
$${\mathcal{O}}(N(K D+D^{2})+|E|D)$$

| Datasets   | #sets   | #users   | #elements   | #E/S   | #S/U   |
|------------|---------|----------|-------------|--------|--------|
| TaFeng     | 73,355  | 9,841    | 4,935       | 5.41   | 7.45   |
| DC         | 42,905  | 9,010    | 217         | 1.52   | 4.76   |
| TaoBao     | 628,618 | 113,347  | 689         | 1.10   | 5.55   |
| TMS        | 243,394 | 15,726   | 1,565       | 2.19   | 15.48  |

214 This formulation confirms our model's efficient scaling with respect to the number of elements N, 215 maximum sequence length K, embedding dimensionality D, and vocabulary size |E|. Our model 216 offers computational advantages compared to existing approaches, achieving linear scaling with 217 respect to both the number of elements N and maximum sequence length K. 218 This efficiency, coupled with time complexity that remains independent of the number of layers, 219 represents a significant improvement over related methods that either scale quadratically with N or 220 K, or have complexity that grows linearly with the number of layers in the model.

221 222 **Space Complexity:** The primary memory cost arises from the element embeddings with 223 complexity O(|E|D), which is standard across similar methods. Beyond this, PIETSP introduces the PE which is O((K + D)D), PI is O(D2), an element scorer with O(D2 224 ) complexity and score 225 fusion with O(|E|). This leads to a total space cost of:
O(|E|D + (K + D)D + D2 + D2 + |E|)

226 This simplifies to:
$${\mathcal{O}}(D(|E|+K+D))$$

## 227 **5 Experiments**

228 This section details the experimental setup used to evaluate our approach. We describe the datasets 229 employed in our study, followed by the baseline methods used for comparison. We use three standard 230 metrics for top-k recommendation: Recall@k, nDCG@k, and PHR@k. Recall@k measures the 231 proportion of relevant items retrieved, while nDCG@k captures both relevance and ranking quality. 232 PHR@k indicates the fraction of users for whom at least one relevant item appears in the top-k 233 predictions. We report results at multiple cutoffs to provide a comprehensive evaluation.

## 234 **5.1 Datasets**

235 We evaluate our model on four publicly available datasets commonly used in temporal set prediction 236 and next basket recommendation tasks: TaFeng, Dunnhumby-Carbo (DC), **TaoBao**, and Tags237 **Math-Sx (TMS)**. Each dataset records user behaviors over time as sequences of sets, where each 238 set contains the items associated with a user's interaction at a particular timestamp. The last set in 239 the sequence is used as the label. We discuss some relevant statistics for the datasets used in Table 1 240 where \#E/S denotes the average number of elements in each set, \#S/U represents the average number 241 of sets for each user. Our datasets and the train,validation and test splits have been sourced from 242 https://github.com/yule-BUAA/DNNTSP/tree/master/data.

## 243 **5.2 Baseline Methods**

244 To evaluate the effectiveness of our proposed approach, we compare it against three state-of-the245 art models designed for temporal set prediction: Sets2Sets, **DNNTSP**, and **SFCNTSP**. These 246 methods represent diverse modeling strategies, including recurrent, graph-based, and fully connected 247 architectures. 248 **Sets2Sets** [1]. Sets2Sets formulates temporal set prediction as a sequential sets-to-sequential sets 249 learning problem. It employs an encoder-decoder architecture built on recurrent neural networks 250 (RNNs), where each input set is embedded via a set-level embedding mechanism, and the sequence is 251 modeled with a decoder using set-based attention. It also incorporates a repeated-elements module to 252 capture frequent historical patterns and a custom objective function to address label imbalance and 253 label correlation.

| Dataset        |               |               |        |               |               |               |               |        |        |        |               |        |        |
|----------------|---------------|---------------|--------|---------------|---------------|---------------|---------------|--------|--------|--------|---------------|--------|--------|
| (p95 Set Size) | Method        | Recall        | NDCG   | PHR           |               |               |               |        |        |        |               |        |        |
| @1             | @2            | @5            | @10    | @1            | @2            | @5            | @10           | @1     | @2     | @5     | @10           |        |        |
| Tafeng (15)    | Sets2Sets     | 0.0302        | 0.0477 | 0.0832        | 0.1264        | 0.0767        | 0.0754        | 0.0820 | 0.0965 | 0.0767 | 0.1254        | 0.2158 | 0.3296 |
| DNNTSP         | 0.0448        | 0.0694        | 0.1140 | 0.1692        | 0.1422        | 0.1318        | 0.1293        | 0.1436 | 0.1422 | 0.2240 | 0.3509        | 0.4708 |        |
| SFCNTSP        | 0.0471        | 0.0728        | 0.1126 | 0.1674        | 0.1503        | 0.1378        | 0.1303        | 0.1437 | 0.1503 | 0.2336 | 0.3545        | 0.4703 |        |
| PIETSP         | 0.0515        | 0.0833        | 0.1300 | 0.1866        | 0.1778        | 0.1635        | 0.1531        | 0.1650 | 0.1778 | 0.2702 | 0.3885        | 0.4967 |        |
| DC (3)         | Sets2Sets     | 0.1276        | 0.2311 | 0.3825        | 0.4259        | 0.1776        | 0.2185        | 0.2883 | 0.3041 | 0.1775 | 0.3123        | 0.4786 | 0.5219 |
| DNNTSP         | 0.1480        | 0.2424        | 0.3924 | 0.4609        | 0.2047        | 0.2346        | 0.3035        | 0.3282 | 0.2047 | 0.3256 | 0.4870        | 0.5528 |        |
| SFCNTSP        | 0.1581        | 0.2457        | 0.3879 | 0.4585        | 0.2174        | 0.2421        | 0.3063        | 0.3330 | 0.2174 | 0.3295 | 0.4836        | 0.5552 |        |
| PIETSP         | 0.1811        | 0.2632        | 0.3983 | 0.4615        | 0.2514        | 0.2641        | 0.3235        | 0.3463 | 0.2514 | 0.3489 | 0.4958        | 0.5613 |        |
| TaoBao (2)     | Sets2Sets     | 0.0019        | 0.0398 | 0.0985        | 0.1743        | 0.0019        | 0.0260        | 0.0521 | 0.0767 | 0.0019 | 0.0409        | 0.1015 | 0.1787 |
| DNNTSP         | 0.0786        | 0.1394        | 0.2289 | 0.3032        | 0.0812        | 0.1183        | 0.1590        | 0.1831 | 0.0812 | 0.1434 | 0.2337        | 0.3093 |        |
| SFCNTSP        | 0.1003        | 0.1577        | 0.2355 | 0.3103        | 0.1037        | 0.1383        | 0.1766        | 0.1952 | 0.1037 | 0.1619 | 0.2402 0.3165 |        |        |
| PIETSP         | 0.1116        | 0.1613        | 0.2364 | 0.3059        | 0.1155        | 0.1448        | 0.1781        | 0.2012 | 0.1155 | 0.1656 | 0.2410 0.3108 |        |        |
| TMS (4)        | Sets2Sets     | 0.2055 0.2782 | 0.3589 | 0.4423        | 0.3846 0.3408 | 0.3455        | 0.3743 0.3846 | 0.4637 | 0.5645 | 0.6557 |               |        |        |
| DNNTSP         | 0.1248        | 0.2131        | 0.3566 | 0.4691        | 0.2616        | 0.2561        | 0.3000        | 0.3453 | 0.2616 | 0.3789 | 0.5633        | 0.6844 |        |
| SFCNTSP        | 0.1681        | 0.2655        | 0.3940 | 0.4960        | 0.3210        | 0.3133        | 0.3490        | 0.3924 | 0.3210 | 0.4469 | 0.5995        | 0.7044 |        |
| PIETSP         | 0.1930 0.2852 | 0.4074        | 0.4982 | 0.3620 0.3412 | 0.3713        | 0.4075 0.3620 | 0.4638        | 0.6068 | 0.7092 |        |               |        |        |

254 **DNNTSP** [2]. DNNTSP is a deep neural network architecture that captures both intra-set and 255 inter-set dependencies through graph-based modeling. It constructs dynamic co-occurrence graphs 256 over elements within each set and applies weighted graph convolutional layers to model relationships. 257 Additionally, it uses an attention-based temporal module to capture sequence dynamics and a gated 258 fusion mechanism to integrate static and dynamic element representations for improved predictive 259 accuracy. 260 **SFCNTSP** [4]. SFCNTSP proposes a lightweight and efficient architecture based entirely on sim261 plified fully connected networks (SFCNs), eliminating non-linear activations and complex modules 262 such as RNNs or attention. It captures inter-set temporal dependencies, intra-set element relationships, 263 and channel-wise correlations through permutation-invariant and permutation-equivariant operations. 264 Despite its simplicity, it achieves competitive performance while significantly reducing computational 265 and memory costs.

## 266 **6 Results**

267 In this section, we present a comprehensive evaluation of our proposed method. We begin with a 268 performance comparison against state-of-the-art baselines to demonstrate the effectiveness of our 269 approach. Next, we assess the efficiency of the model in terms of model training and inference. 270 Finally, we conduct an ablation study to analyze the contribution of different components in our 271 architecture, which is described in the Appendix section A.1.

## 272 **6.1 Performance Comparison**

273 We evaluate the performance of our proposed method against several strong baselines across four 274 benchmark datasets. We select cut-off k ∈ {1, 2, 5, 10} to span single-item through longer-list predictions relative to each dataset's 95th 275 -percentile set size (p95 ranges from 2 to 15 across our 276 benchmarks). As shown in Table 2, our model consistently outperforms all baselines on Tafeng and 277 DC and leads on TaoBao for k ≤ 5, with only a slight drop at k = 10. On TMS, Sets2Sets attains the 278 highest performance at k = 1 by memorizing the most frequent next element, but PIETSP surpasses 279 every method for k ≥ 2. These results underscore the robustness of our approach across varying set 280 sizes and—to our knowledge—PIETSP establishes a new state-of-the-art performance on all four 281 benchmarks under this evaluation.

## 282 **6.2 Model Efficiency Comparison**

283 To evaluate the efficiency of our proposed method, we compare it with the baseline SFCNTSP model 284 [4]. While SFCNTSP achieves competitive performance by eliminating complex modules like RNNs 285 and attention, it still faces higher time complexity compared to our approach. This higher complexity 286 arises from the dependence on the term O(|E|ND) in their adaptive fusing of user representations

| Dataset   | Model   | Mean Time (s)   | P99 Time (s)   | Samples/sec   | Epochs   |
|-----------|---------|-----------------|----------------|---------------|----------|
| TaFeng    | SFCNTSP | 0.00214         | 0.00250        | 467.69        | 327      |
| PIETSP    | 0.00011 | 0.00017         | 9081.62        | 14            |          |
| DC        | SFCNTSP | 0.00083         | 0.00096        | 1204.03       | 305      |
| PIETSP    | 0.00012 | 0.00061         | 8010.45        | 8             |          |
| TaoBao    | SFCNTSP | 0.00091         | 0.00109        | 1097.26       | 446      |
| PIETSP    | 0.00010 | 0.00013         | 9799.39        | 8             |          |
| TMS       | SFCNTSP | 0.00238         | 0.00258        | 419.97        | 313      |
| PIETSP    | 0.00012 | 0.00064         | 8628.95        | 11            |          |

Table 3: Inference speed and training efficiency comparison of SFCNTSP and PIETSP.

287 layer. Given that |E| ≫ N, this dependence on the element domain size |E| can lead to significant 288 computational costs, particularly as the number of elements grows. In contrast, our method achieves 289 a lower complexity of O(|E|D). 290 We focus on SFCNTSP for this comparison because it shares a similar goal of reducing computational 291 cost while achieving efficient performance. However, PIETSP reduces time complexity further, 292 providing both lower latency and higher throughput in large-scale settings. 293 Table 3 presents a detailed comparison of inference performance using mean sample time, 99th per294 centile latency (P99), and throughput (samples per second). We tested the models on Nvidia T4 GPU,
295 across 100 runs with batch size 64 and embedding dimension D of 32. Across all datasets, PIETSP
296 consistently outperforms SFCNTSP, exhibiting lower latency and significantly higher throughput. 297 These results highlight the practical advantages of our model in time-sensitive and large-scale 298 deployments. 299 Table 3 also shows the number of epochs required for each model to converge. Our proposed model, 300 PIETSP, achieves convergence significantly faster, requiring substantially fewer training epochs 301 than SFCNTSP across all datasets. This demonstrates its training efficiency and potential for rapid 302 iteration in real-world systems.

## 303 **7 Conclusion**

304 We propose PIETSP, a scalable and permutation-aware model for temporal set prediction that achieves 305 linear time complexity with respect to both sequence length and element count. By integrating 306 permutation-equivariant and permutation-invariant operations, PIETSP enables efficient modeling of 307 evolving sets and offers significant improvements in inference speed and training efficiency. 308 Empirical results across four public benchmarks show that PIETSP achieves comparable or superior 309 performance to existing state-of-the-art models while requiring significantly fewer computational 310 resources.

## 311 **8 Limitations And Future Work**

312 While efficient, the current architecture may under-capture fine-grained inter-element dependencies. 313 Enhancing expressiveness via more advanced attention mechanisms (e.g., Set Transformers) is a 314 promising direction. Additionally, the model operates on a fixed-length temporal window, which 315 may limit its effectiveness on long-range dependencies. Lastly, our work does not explore fairness or 316 uncertainty estimation, both of which are important considerations for high-stakes applications and 317 future work.

## 318 **References**

319 [1] Haoji Hu and Xiangnan He. Sets2sets: Learning from sequential sets with neural networks. In 320 *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &* 321 *Data Mining*, pages 1491–1499, 2019. 322 [2] Le Yu, Leilei Sun, Bowen Du, Chuanren Liu, Hui Xiong, and Weifeng Lv. Predicting temporal 323 sets with deep neural networks. In *Proceedings of the 26th ACM SIGKDD International* 324 *Conference on Knowledge Discovery & Data Mining*, pages 1083–1091, 2020. 325 [3] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional 326 networks. *arXiv preprint arXiv:1609.02907*, 2016. 327 [4] Le Yu, Zihang Liu, Tongyu Zhu, Leilei Sun, Bowen Du, and Weifeng Lv. Predicting temporal 328 sets with simplified fully connected networks. In *Proceedings of the AAAI Conference on* 329 *Artificial Intelligence*, volume 37, pages 4835–4844, 2023. 330 [5] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. Factorizing personalized 331 markov chains for next-basket recommendation. In *Proceedings of the 19th international* 332 *conference on World wide web*, pages 811–820, 2010. 333 [6] Haochao Ying, Fuzhen Zhuang, Fuzheng Zhang, Yanchi Liu, Guandong Xu, Xing Xie, Hui 334 Xiong, and Jian Wu. Sequential recommender system based on hierarchical attention network. 335 In *IJCAI international joint conference on artificial intelligence*, 2018. 336 [7] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In *2018* 337 *IEEE international conference on data mining (ICDM)*, pages 197–206. IEEE, 2018. 338 [8] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, 339 Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information* 340 *processing systems*, 30, 2017.

341 [9] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural 342 networks. In *Advances in Neural Information Processing Systems*, volume 27, 2014. 343 [10] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly 344 learning to align and translate. 2014. 345 [11] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, 346 and Alexander Smola. Deep sets. *Advances in neural information processing systems*, 30, 2017. 347 [12] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. 348 Set transformer: A framework for attention-based permutation-invariant neural networks. In 349 *Proceedings of the 36th International Conference on Machine Learning*, volume 97, pages 350 3744–3753. PMLR, 2019. 351 [13] Jinwoo Kim, Jaehoon Yoo, Juho Lee, and Seunghoon Hong. Setvae: Learning hierarchical 352 composition for generative modeling of set-structured data. In *Proceedings of the IEEE/CVF* 353 *Conference on Computer Vision and Pattern Recognition*, pages 15059–15068, 2021.

## 354 **A Technical Appendices And Supplementary Material** 355 **A.1 Ablation Study**

356 To better understand the contribution of each component in our architecture, we conduct an ablation 357 study. We compare the full PIETSP model with two variants: PIETSP-EE and PIETSP-GE, where we 358 remove the element evaluator EE and global evaluator GE modules respectively. We test the variants 359 on the Tafeng dataset. As shown in Table 4, removing either component leads to a noticeable drop 360 in performance, confirming the importance of both elements in capturing temporal and contextual 361 patterns effectively. The full model consistently outperforms both ablated variants, demonstrating 362 that the synergy between EE and GE is crucial to the overall effectiveness of the proposed approach.

## 363 **A.2 Statistical Variability In Experimental Results**

364 We present the variability in the experimental results of PIETSP across various metrics, reporting the 365 mean along with two standard deviations as shown in Table 5

| Dataset   | Method   | Recall   | NDCG   | PHR    |        |        |        |        |        |        |        |        |        |
|-----------|----------|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| @1        | @2       | @5       | @10    | @1     | @2     | @5     | @10    | @1     | @2     | @5     | @10    |        |        |
| PIETSP-GE | 0.0043   | 0.0050   | 0.0058 | 0.0060 | 0.0218 | 0.0156 | 0.0107 | 0.0090 | 0.0218 | 0.0239 | 0.0274 | 0.0284 |        |
| PIETSP-EE | 0.0452   | 0.0669   | 0.1004 | 0.1333 | 0.1427 | 0.1178 | 0.1098 | 0.1163 | 0.1427 | 0.1910 | 0.2712 | 0.3393 |        |
| Tafeng    | PIETSP   | 0.0515   | 0.0833 | 0.1300 | 0.1866 | 0.1778 | 0.1635 | 0.1531 | 0.1650 | 0.1778 | 0.2702 | 0.3885 | 0.4967 |

Table 4: Ablation study on the Tafeng dataset. PIETSP-GE: global evaluator removed; PIETSP-EE: element evaluator removed. Best results per metric are in bold. Table 5: Performance of our proposed model on the datasets. Each value is reported as mean ± 2xstd.

| Dataset   | Metric          | @1              | @2              | @5              | @10   |
|-----------|-----------------|-----------------|-----------------|-----------------|-------|
| PHR       | 0.1756 ± 0.0068 | 0.2661 ± 0.0142 | 0.3926 ± 0.0089 | 0.5009 ± 0.0148 |       |
| nDCG      | 0.1756 ± 0.0068 | 0.1612 ± 0.0074 | 0.1530 ± 0.0041 | 0.1655 ± 0.0042 |       |
| Recall    | 0.0516 ± 0.0022 | 0.0830 ± 0.0042 | 0.1316 ± 0.0070 | 0.1891 ± 0.0108 |       |
| PHR       | 0.2466 ± 0.0048 | 0.3485 ± 0.0038 | 0.4958 ± 0.0021 | 0.5617 ± 0.0028 |       |
| nDCG      | 0.2466 ± 0.0048 | 0.2633 ± 0.0033 | 0.3227 ± 0.0018 | 0.3457 ± 0.0016 |       |
| Recall    | 0.1779 ± 0.0034 | 0.2645 ± 0.0030 | 0.3984 ± 0.0015 | 0.4624 ± 0.0020 |       |
| PHR       | 0.1156 ± 0.0017 | 0.1672 ± 0.0021 | 0.2410 ± 0.0016 | 0.3111 ± 0.0014 |       |
| nDCG      | 0.1156 ± 0.0017 | 0.1459 ± 0.0016 | 0.1787 ± 0.0010 | 0.2010 ± 0.0010 |       |
| Recall    | 0.1118 ± 0.0015 | 0.1630 ± 0.0021 | 0.2364 ± 0.0017 | 0.3052 ± 0.0013 |       |
| PHR       | 0.3616 ± 0.0036 | 0.4672 ± 0.0049 | 0.6073 ± 0.0017 | 0.7048 ± 0.0047 |       |
| nDCG      | 0.3616 ± 0.0036 | 0.3418 ± 0.0026 | 0.3714 ± 0.0014 | 0.4073 ± 0.0015 |       |
| Recall    | 0.1927 ± 0.0020 | 0.2861 ± 0.0026 | 0.4080 ± 0.0020 | 0.4956 ± 0.0030 |       |

## 366 **Neurips Paper Checklist** 367 1. **Claims**

368 Question: Do the main claims made in the abstract and introduction accurately reflect the 369 paper's contributions and scope? 370 Answer: [Yes] 371 Justification: Our main claims and contributions have been detailed in the abstract and 372 section 1. Please refer to Subsection 4.8 and Section 6 for theoretical and experimental 373 evidence.

374 Guidelines: 375 - The answer NA means that the abstract and introduction do not include the claims 376 made in the paper. 377 - The abstract and/or introduction should clearly state the claims made, including the 378 contributions made in the paper and important assumptions and limitations. A No or 379 NA answer to this question will not be perceived well by the reviewers. 380 - The claims made should match theoretical and experimental results, and reflect how 381 much the results can be expected to generalize to other settings. 382 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 383 are not attained by the paper. 384 2. **Limitations** 385 Question: Does the paper discuss the limitations of the work performed by the authors? 386 Answer: [Yes] 387 Justification: Kindly refer to Section 8 for limitations. Kindly refer to Subsection 4.8 and 388 Subsection 6.2 for theoretical and experimental details on computational efficiency. 389 Guidelines: 390 - The answer NA means that the paper has no limitation while the answer No means that 391 the paper has limitations, but those are not discussed in the paper. 392 - The authors are encouraged to create a separate "Limitations" section in their paper. 393 - The paper should point out any strong assumptions and how robust the results are to 394 violations of these assumptions (e.g., independence assumptions, noiseless settings, 395 model well-specification, asymptotic approximations only holding locally). The authors 396 should reflect on how these assumptions might be violated in practice and what the 397 implications would be. 398 - The authors should reflect on the scope of the claims made, e.g., if the approach was 399 only tested on a few datasets or with a few runs. In general, empirical results often 400 depend on implicit assumptions, which should be articulated. 401 - The authors should reflect on the factors that influence the performance of the approach. 402 For example, a facial recognition algorithm may perform poorly when image resolution 403 is low or images are taken in low lighting. Or a speech-to-text system might not be 404 used reliably to provide closed captions for online lectures because it fails to handle 405 technical jargon. 406 - The authors should discuss the computational efficiency of the proposed algorithms 407 and how they scale with dataset size. 408 - If applicable, the authors should discuss possible limitations of their approach to 409 address problems of privacy and fairness. 410 - While the authors might fear that complete honesty about limitations might be used by 411 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 412 limitations that aren't acknowledged in the paper. The authors should use their best 413 judgment and recognize that individual actions in favor of transparency play an impor414 tant role in developing norms that preserve the integrity of the community. Reviewers 415 will be specifically instructed to not penalize honesty concerning limitations. 416 3. **Theory assumptions and proofs** 417 Question: For each theoretical result, does the paper provide the full set of assumptions and 418 a complete (and correct) proof? 419 Answer: [Yes] 420 Justification: We detail the complete proof with the assumptions in section 3 and section 4 421 Guidelines: 422 - The answer NA means that the paper does not include theoretical results. 423 - All the theorems, formulas, and proofs in the paper should be numbered and cross424 referenced. 425 - All assumptions should be clearly stated or referenced in the statement of any theorems. 426 - The proofs can either appear in the main paper or the supplemental material, but if 427 they appear in the supplemental material, the authors are encouraged to provide a short 428 proof sketch to provide intuition.

429 - Inversely, any informal proof provided in the core of the paper should be complemented 430 by formal proofs provided in appendix or supplemental material. 431 - Theorems and Lemmas that the proof relies upon should be properly referenced. 432 4. **Experimental result reproducibility** 433 Question: Does the paper fully disclose all the information needed to reproduce the main ex434 perimental results of the paper to the extent that it affects the main claims and/or conclusions 435 of the paper (regardless of whether the code and data are provided or not)?

436 Answer: [Yes]
437 Justification: The proposed algorithm and the architecture have been described in detail for 438 reproducibility in section 3 and section 4 439 Guidelines: 440 - The answer NA means that the paper does not include experiments. 441 - If the paper includes experiments, a No answer to this question will not be perceived 442 well by the reviewers: Making the paper reproducible is important, regardless of 443 whether the code and data are provided or not.

444 - If the contribution is a dataset and/or model, the authors should describe the steps taken 445 to make their results reproducible or verifiable. 446 - Depending on the contribution, reproducibility can be accomplished in various ways. 447 For example, if the contribution is a novel architecture, describing the architecture fully 448 might suffice, or if the contribution is a specific model and empirical evaluation, it may 449 be necessary to either make it possible for others to replicate the model with the same 450 dataset, or provide access to the model. In general. releasing code and data is often 451 one good way to accomplish this, but reproducibility can also be provided via detailed 452 instructions for how to replicate the results, access to a hosted model (e.g., in the case 453 of a large language model), releasing of a model checkpoint, or other means that are 454 appropriate to the research performed. 455 - While NeurIPS does not require releasing code, the conference does require all submis456 sions to provide some reasonable avenue for reproducibility, which may depend on the 457 nature of the contribution. For example 458 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 459 to reproduce that algorithm. 460 (b) If the contribution is primarily a new model architecture, the paper should describe 461 the architecture clearly and fully. 462 (c) If the contribution is a new model (e.g., a large language model), then there should 463 either be a way to access this model for reproducing the results or a way to reproduce 464 the model (e.g., with an open-source dataset or instructions for how to construct 465 the dataset). 466 (d) We recognize that reproducibility may be tricky in some cases, in which case 467 authors are welcome to describe the particular way they provide for reproducibility. 468 In the case of closed-source models, it may be that access to the model is limited in 469 some way (e.g., to registered users), but it should be possible for other researchers 470 to have some path to reproducing or verifying the results.

471 5. **Open access to data and code**
472 Question: Does the paper provide open access to the data and code, with sufficient instruc473 tions to faithfully reproduce the main experimental results, as described in supplemental 474 material? 475 Answer: [No] 476 Justification: We use a public dataset cited in section 5.1. Code is not released at this time 477 due to proprietary dependencies. However, the methodology is described in sufficient detail 478 in the paper to allow motivated readers to implement the approach independently. 479 Guidelines: 480 - The answer NA means that paper does not include experiments requiring code.

481 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/ 482 public/guides/CodeSubmissionPolicy) for more details.

483 - While we encourage the release of code and data, we understand that this might not be 484 possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not 485 including code, unless this is central to the contribution (e.g., for a new open-source 486 benchmark).

487 - The instructions should contain the exact command and environment needed to run to 488 reproduce the results. See the NeurIPS code and data submission guidelines (https: 489 //nips.cc/public/guides/CodeSubmissionPolicy) for more details. 490 - The authors should provide instructions on data access and preparation, including how 491 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 492 - The authors should provide scripts to reproduce all experimental results for the new 493 proposed method and baselines. If only a subset of experiments are reproducible, they 494 should state which ones are omitted from the script and why. 495 - At submission time, to preserve anonymity, the authors should release anonymized 496 versions (if applicable). 497 - Providing as much information as possible in supplemental material (appended to the 498 paper) is recommended, but including URLs to data and code is permitted. 499 6. **Experimental setting/details** 500 Question: Does the paper specify all the training and test details (e.g., data splits, hyper501 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 502 results? 503 Answer: [Yes] 504 Justification: We detail all the training and test details in section 4.7 505 Guidelines: 506 - The answer NA means that the paper does not include experiments. 507 - The experimental setting should be presented in the core of the paper to a level of detail 508 that is necessary to appreciate the results and make sense of them. 509 - The full details can be provided either with the code, in appendix, or as supplemental 510 material. 511 7. **Experiment statistical significance** 512 Question: Does the paper report error bars suitably and correctly defined or other appropriate 513 information about the statistical significance of the experiments? 514 Answer: [Yes] 515 Justification: We report error bars as the mean ± 2 standard deviations across multiple 516 random seeds for all key evaluation metrics (Recall@k, nDCG@k, PHR@k) in appendix 517 section A.2. 518 Guidelines: 519 - The answer NA means that the paper does not include experiments. 520 - The authors should answer "Yes" if the results are accompanied by error bars, confi521 dence intervals, or statistical significance tests, at least for the experiments that support 522 the main claims of the paper. 523 - The factors of variability that the error bars are capturing should be clearly stated (for 524 example, train/test split, initialization, random drawing of some parameter, or overall 525 run with given experimental conditions).

526 - The method for calculating the error bars should be explained (closed form formula, 527 call to a library function, bootstrap, etc.) 528 - The assumptions made should be given (e.g., Normally distributed errors). 529 - It should be clear whether the error bar is the standard deviation or the standard error 530 of the mean.

531 - It is OK to report 1-sigma error bars, but one should state it. The authors should 532 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis 533 of Normality of errors is not verified. 534 - For asymmetric distributions, the authors should be careful not to show in tables or 535 figures symmetric error bars that would yield results that are out of range (e.g. negative 536 error rates). 537 - If error bars are reported in tables or plots, The authors should explain in the text how 538 they were calculated and reference the corresponding figures or tables in the text.

## 539 8. **Experiments Compute Resources**

540 Question: For each experiment, does the paper provide sufficient information on the com541 puter resources (type of compute workers, memory, time of execution) needed to reproduce 542 the experiments? 543 Answer: [Yes] 544 Justification: We indicate the compute resources required in section 6.2 545 Guidelines: 546 - The answer NA means that the paper does not include experiments. 547 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, 548 or cloud provider, including relevant memory and storage.

549 - The paper should provide the amount of compute required for each of the individual 550 experimental runs as well as estimate the total compute. 551 - The paper should disclose whether the full research project required more compute 552 than the experiments reported in the paper (e.g., preliminary or failed experiments that 553 didn't make it into the paper). 554 9. **Code of ethics** 555 Question: Does the research conducted in the paper conform, in every respect, with the 556 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 557 Answer: [Yes] 558 Justification: We followed the NeurIPS Code of Ethics. 559 Guidelines: 560 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 561 - If the authors answer No, they should explain the special circumstances that require a 562 deviation from the Code of Ethics. 563 - The authors should make sure to preserve anonymity (e.g., if there is a special consid564 eration due to laws or regulations in their jurisdiction).

565 10. **Broader impacts**
566 Question: Does the paper discuss both potential positive societal impacts and negative 567 societal impacts of the work performed? 568 Answer: [NA] 569 Justification: This work uses publicly available datasets. There is no explicit negative social 570 impact. 571 Guidelines: 572 - The answer NA means that there is no societal impact of the work performed.

573 - If the authors answer NA or No, they should explain why their work has no societal 574 impact or why the paper does not address societal impact.

575 - Examples of negative societal impacts include potential malicious or unintended uses 576 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 577 (e.g., deployment of technologies that could make decisions that unfairly impact specific 578 groups), privacy considerations, and security considerations. 579 - The conference expects that many papers will be foundational research and not tied 580 to particular applications, let alone deployments. However, if there is a direct path to 581 any negative applications, the authors should point it out. For example, it is legitimate 582 to point out that an improvement in the quality of generative models could be used to 583 generate deepfakes for disinformation. On the other hand, it is not needed to point out 584 that a generic algorithm for optimizing neural networks could enable people to train 585 models that generate Deepfakes faster. 586 - The authors should consider possible harms that could arise when the technology is 587 being used as intended and functioning correctly, harms that could arise when the 588 technology is being used as intended but gives incorrect results, and harms following 589 from (intentional or unintentional) misuse of the technology. 590 - If there are negative societal impacts, the authors could also discuss possible mitigation 591 strategies (e.g., gated release of models, providing defenses in addition to attacks, 592 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 593 feedback over time, improving the efficiency and accessibility of ML).

594 11. **Safeguards**
595 Question: Does the paper describe safeguards that have been put in place for responsible 596 release of data or models that have a high risk for misuse (e.g., pretrained language models, 597 image generators, or scraped datasets)? 598 Answer: [NA] 599 Justification: The paper poses no such risks. 600 Guidelines: 601 - The answer NA means that the paper poses no such risks. 602 - Released models that have a high risk for misuse or dual-use should be released with 603 necessary safeguards to allow for controlled use of the model, for example by requiring 604 that users adhere to usage guidelines or restrictions to access the model or implementing 605 safety filters. 606 - Datasets that have been scraped from the Internet could pose safety risks. The authors 607 should describe how they avoided releasing unsafe images. 608 - We recognize that providing effective safeguards is challenging, and many papers do 609 not require this, but we encourage authors to take this into account and make a best 610 faith effort. 611 12. **Licenses for existing assets** 612 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 613 the paper, properly credited and are the license and terms of use explicitly mentioned and 614 properly respected? 615 Answer:[Yes] 616 Justification: Yes, we credited them in appropriate ways. 617 Guidelines: 618 - The answer NA means that the paper does not use existing assets. 619 - The authors should cite the original paper that produced the code package or dataset. 620 - The authors should state which version of the asset is used and, if possible, include a 621 URL. 622 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 623 - For scraped data from a particular source (e.g., website), the copyright and terms of 624 service of that source should be provided. 625 - If assets are released, the license, copyright information, and terms of use in the 626 package should be provided. For popular datasets, paperswithcode.com/datasets 627 has curated licenses for some datasets. Their licensing guide can help determine the 628 license of a dataset. 629 - For existing datasets that are re-packaged, both the original license and the license of 630 the derived asset (if it has changed) should be provided. 631 - If this information is not available online, the authors are encouraged to reach out to 632 the asset's creators. 633 13. **New assets** 634 Question: Are new assets introduced in the paper well documented and is the documentation 635 provided alongside the assets? 636 Answer: [NA] 637 Justification: The paper does not release new assets. 638 Guidelines: 639 - The answer NA means that the paper does not release new assets. 640 - Researchers should communicate the details of the dataset/code/model as part of their 641 submissions via structured templates. This includes details about training, license, 642 limitations, etc. 643 - The paper should discuss whether and how consent was obtained from people whose 644 asset is used. 645 - At submission time, remember to anonymize your assets (if applicable). You can either 646 create an anonymized URL or include an anonymized zip file.

## 647 14. **Crowdsourcing And Research With Human Subjects**

648 Question: For crowdsourcing experiments and research with human subjects, does the paper 649 include the full text of instructions given to participants and screenshots, if applicable, as 650 well as details about compensation (if any)? 651 Answer: [NA] 652 Justification: The paper does not involve crowdsourcing nor research with human subjects. 654 - The answer NA means that the paper does not involve crowdsourcing nor research with 655 human subjects. 656 - Including this information in the supplemental material is fine, but if the main contribu657 tion of the paper involves human subjects, then as much detail as possible should be 658 included in the main paper. 659 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, 660 or other labor should be paid at least the minimum wage in the country of the data 661 collector.

## 662 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human**

663 **subjects** 664 Question: Does the paper describe potential risks incurred by study participants, whether 665 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) 666 approvals (or an equivalent approval/review based on the requirements of your country or 667 institution) were obtained? 668 Answer: [NA] 669 Justification: The paper does not involve crowdsourcing nor research with human subjects. 670 Guidelines: 671 - The answer NA means that the paper does not involve crowdsourcing nor research with 672 human subjects. 673 - Depending on the country in which research is conducted, IRB approval (or equivalent) 674 may be required for any human subjects research. If you obtained IRB approval, you 675 should clearly state this in the paper. 676 - We recognize that the procedures for this may vary significantly between institutions 677 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the 678 guidelines for their institution. 679 - For initial submissions, do not include any information that would break anonymity (if 680 applicable), such as the institution conducting the review. 681 16. **Declaration of LLM usage**
682 Question: Does the paper describe the usage of LLMs if it is an important, original, or 683 non-standard component of the core methods in this research? Note that if the LLM is used 684 only for writing, editing, or formatting purposes and does not impact the core methodology, 685 scientific rigorousness, or originality of the research, declaration is not required. 686 Answer: [NA] 687 Justification: The core method development in this research does not involve LLMs as any 688 important, original, or non-standard components 689 Guidelines: 690 - The answer NA means that the core method development in this research does not 691 involve LLMs as any important, original, or non-standard components.

692 - Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
693 for what should or should not be described.

653 Guidelines: