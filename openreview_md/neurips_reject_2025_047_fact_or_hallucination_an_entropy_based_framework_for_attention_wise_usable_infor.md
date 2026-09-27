# Fact Or Hallucination? An Entropy-Based Framework For Attention-Wise Usable Information In Llms

Anonymous Author(s)

Affiliation

Address

email

## Abstract 19 **1 Introduction**

1 Large language models (LLMs) often generate confident yet inaccurate outputs, 2 posing serious risks in safety-critical applications. Existing hallucination detection 3 methods typically rely on final-layer logits or post-hoc textual checks, which can 4 obscure the rich semantic signals encoded across model layers. Thus, we propose 5 **Shapley NEAR** (Norm-basEd Attention-wise usable infoRmation), a principled, 6 entropy-based attribution framework grounded in Shapley values that assigns a 7 confidence score indicating whether an LLM output is hallucinatory. Unlike prior 8 approaches, Shapley NEAR decomposes attention-driven information flow across 9 all layers and heads of the model, where higher scores correspond to lower halluci10 nation risk. It further distinguishes between two hallucination types: *parametric* 11 *hallucinations*, caused by the model's pre-trained knowledge overriding the context, 12 and *context-induced hallucinations*, where misleading context fragments spuri13 ously reduce uncertainty. To mitigate parametric hallucinations, we introduce 14 a test-time *head clipping* technique that prunes attention heads contributing to 15 overconfident, context-agnostic outputs. Empirical results in four QA benchmarks 16 (CoQA, QuAC, SQuAD, and TriviaQA), using Qwen2.5-3B, LLaMA3.1-8B, and 17 OPT-6.7B, demonstrate that Shapley NEAR outperforms strong baselines, without 18 requiring additional training, prompting, or architectural modifications. 20 The rapid proliferation of large language models (LLMs) in a variety of applications, from conversa21 tional agents to automated decision making systems, has underscored their impressive capabilities 22 [1, 2]. However, a challenge persists: these models often generate outputs that are confidently 23 stated yet factually incorrect, a phenomenon widely known as hallucination [3]. This issue becomes 24 especially critical in safety-sensitive environments where factual accuracy is paramount [4, 5]. 25 To tackle this, a number of recent studies have investigated hallucination in LLMs using both 26 theoretical and empirical approaches. While token-level uncertainty measures such as entropy and 27 confidence have proven useful in hallucination detection for NLP tasks [6], extending these methods 28 to sentence-level predictions in autoregressive LLMs remains challenging due to the models' complex 29 and interdependent outputs [7, 8]. As a workaround, recent research has attempted to infer sentence30 level uncertainty directly from the generated language itself [9, 10]. However, these works did not 31 consider the dense semantic information encoded inside the internal layers of the LLM [11–13]. In 32 parallel, [13] introduced the concept of V-usable information, which quantifies how much useful 33 information a model can extract under computational constraints. Building on this, [14] proposed 34 Pointwise V-Information (PVI) to estimate instance-level dataset difficulty, although this metric 35 only considers the final layer. In contrast, [12] proposed using the EigenScore of the final token 36 from a middle transformer layer to detect hallucinations, and further analyzed model reliability by 37 comparing multiple responses to a shared prompt. However, despite these advances, most of these Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

Input from User Norm-based Attention Shapely NEAR

Information Gain LLM is not hallucinating, the generated answer is corrrect Decoder Layer Attention-wise Information Gain:

IG(x → q) = H(q | q ,∅) 
- H(q | q , x )

t <t t <t Yes Sentence-level segmentation of the context:

sx {1 ,s2 ,..sn }
For an input context x and question q x + q Shapley NEAR(x, q) =

(1/n) × Σ IG , for i = 1 to n i High Shapely NEAR

Score?

Shapley Information Gain for context segment sx:
IG = Average over all subsets S ⊆ {s , …, s } ∖
{s } of (IG(S ∪ {s }) −

IG(S))

s 1 n x x Question :

What is the captical of France?

Masked Self-Attention Layer Norm No Generated Answer: Paris FC Layer LLM is hallucinating

38 methods focus exclusively on final-layer logits and overlook the rich information encoded in all the 39 internal states of LLMs [15]. With further development, LLM-Check [16] extended hallucination 40 detection to both white-box and black-box settings by employing an auxiliary LLM to analyze 41 hidden states, attention patterns, and output probabilities. Similarly, Lookback Lens [17] trained 42 a linear classifier using the ratio of attention on the context versus generated tokens to identify 43 contextual hallucinations. However, both approaches fail to distinguish whether hallucinations 44 originate from the pre-trained knowledge of the model (parametric hallucination) or from misleading 45 contextual information (contextual hallucination). Complementing these lines of work, [18] examined 46 deficiencies across layers for unanswerable question detection, while [11] revealed that feed-forward 47 layers often exhibit less reliable distributional associations compared to the more robust in-context 48 reasoning encoded by attention mechanisms. 49 To address the limitations of these prior approaches, we introduce **Shapley NEAR (Norm-basEd** 50 **Attention-wise usable infoRmation)**, a method designed to assign a confidence score indicating 51 whether an LLM-generated answer is trustworthy or hallucinatory, given a question and context. In 52 contrast to previous methods that primarily rely on outputs from feed-forward layers, which have 53 limited bearing on reasoning [11], our approach focuses exclusively on attention layers. Shapley 54 NEAR aggregates information from all attention heads across all layers [15], enabling a fine-grained, 55 attention-wise and layer-wise analysis of information propagation. Crucially, our method requires no 56 additional training or architectural changes, making it both easy to integrate into existing pre-trained 57 models and highly interpretable in practice. The main contributions of our paper are as follows:

## 68 **2 Background**

58 - We propose **Shapley NEAR**, a principled, interpretable entropy-based attribution method 59 grounded in Shapley-value theory that quantifies usable information flow in LLMs by 60 decomposing entropy reduction across layers and heads using the norm of attention outputs. 61 - We demonstrate that our framework not only detects hallucinations introduced by context 62 segments but also distinguishes between *parametric* and *context-induced* hallucinations. 63 - We introduce a test-time strategy to identify attention heads that consistently exhibit para64 metric hallucinations. Selectively removing these heads during inference demonstrates a 65 novel application of attribution techniques to improve model reliability without retraining. 66 - We evaluate Shapley NEAR on multiple QA datasets using Qwen2.5-3B, LLaMA3.1-8B, 67 and OPT-6.7B, showing that it outperforms strong baselines mention in Section. 69 In this work, we focus on quantifying how much usable information a generative language model can 70 extract from a given context to answer a specific question. Formally, we consider an input context 71 X = {s1, s2*, . . . , s*n}, and a typical autoregressive large language model (LLM), denoted by V, 72 which generates a response sequence Y = [y1, y2*, . . . , y*T ], where each token yt is conditioned on 73 the input and previous outputs. Our central goal is to determine how much V-usable information the 74 model can leverage from the context X to predict the output Y . A lower value of usable information 75 implies greater prediction difficulty, indicating that the dataset is more challenging for the models V. 76 While classical information-theoretic tools such as Shannon's mutual information I(X; Y )[19] and 77 the data processing inequality (DPI)[20] have long served as foundational metrics for analyzing 78 information flow, recent research has revealed their limitations when applied to deep models. These 79 classical measures tend to overestimate the practically usable signal, particularly in settings where 80 models operate under computational constraints as modern LLMs can progressively extract structured 81 and meaningful representations from raw inputs through deep computation, rendering traditional 82 metrics insufficient. 83 To bridge this gap, [13] introduced the notion of predictive V*-information*, which accounts for 84 the computational limitations of a model family V. They define this as the difference between 85 two entropy terms: the conditional V-entropy with and without contextual input. Specifically, the 86 predictive V-information is given by:
IV (X → Y ) = HV (Y |∅) − HV (Y |X),
87 where HV (Y |X) denotes the expected uncertainty over outputs Y when conditioned on context 88 X, and HV (Y |∅) captures the model's uncertainty in the absence of any input. While predictive 89 V-information captures dataset-level trends, Ethayarajh et al. [14] extend it to the instance level via 90 pointwise V*-information (PVI)*, which measures how much information a specific input x provides 91 for predicting its output y. This enables fine-grained analysis of instance difficulty, essential for 92 real-world model evaluation. 93 Building on these foundations, [18] propose *layer-wise usable information* (LI), a method that 94 decomposes usable information across the layers of a model, thereby enhancing interpretability. 95 Complementary to this, [11] show that feed-forward layers primarily encode superficial distributional 96 patterns, whereas attention mechanisms are more closely aligned with in-context reasoning. These 97 insights motivate our work, which integrates the strengths of previous efforts to develop a unified, 98 interpretable framework to assess usable information in LLMs, both across layers and at the sentence 99 level, while accounting for how different components of the model influence predictive certainty.

## 100 **3 Shapley Near: Norm-Based Attention-Wise Usable Information**

101 Given a set of context passages, generative language models (LLMs) produce free-form text responses 102 to questions. In this work, we aim to systematically quantify how individual parts of the context 103 influence the prediction at the final token of the question. Transformer-based models organize 104 computation across multiple layers and attention heads, where each head captures distinct patterns 105 of contextual dependency[21]. Building on this insight, we propose **Shapley NEAR**, a framework 106 for measuring how much usable information each sentence in a context contributes to reducing the 107 model's predictive uncertainty. Shapley NEAR is computed by isolating the output of each attention 108 head at the final token position of the question and measuring the change in entropy when conditioning 109 on subsets of the input context versus a null context. To attribute this entropy reduction fairly to 110 individual sentences, we adopt a Shapley-value-based decomposition. For clarity, the remainder 111 of the paper, we will use the terms *Shapley NEAR* and *NEAR* interchangeably. An overview of 112 our architecture is illustrated in Figure 1, while the detailed algorithmic procedure is presented in 113 Appendix A7.

114 Let sx = (s1, s2*, . . . , s*n) ∈ C denote a context passage composed of n disjoint sentences, and let 115 q ∈ Q represent the associated question. The concatenated input sequence sx q is tokenized into 116 a sequence of length T, with the final token of the question indexed by qt ∈ {1*, . . . , T*}. In this 117 framework, we consider a formally defined predictive family V consisting of pretrained generative 118 language models, where each model is composed of L transformer layers and each layer contains 119 H attention heads. Each attention head h in each layer ℓ of the language models creates different computations. Mathematically, we define V ⊆ Ω = {f
(l,h)
120 : *C ∪ ∅ → P*(Q)}, where C and Q 121 are random variables with sample spaces C and Q, respectively, and P(Q) denotes the set of all probability measures over Q equipped with the Borel algebra on C. The mapping f
(l,h)
122 represents 123 the function associated with attention head of a specific layer (l, h) within the predictive family V. 124 The range of f corresponds to the vocabulary space of the model. Given a layer l and attention-head 125 h in V, the function f maps the context tokens (or null context) to probability distribution over the 126 vocabulary. Unlike prior work, the function f is assumed to operate without any additional fine-tuning 127 on external training data. In the rest of the section we will build the mathematical formula for NEAR,
128 defining and explaining each step.

142 According to [15, 5], the last token embedding captures the semantic information of the entire text.
143 Therefore, we then extract the projected vector corresponding to the final question token qt from
144 equation 2,
$$\mathbf{z}_{x,q}^{(\ell,h)}\triangleq{\bar{Z}}_{q_{t}}^{(\ell,h)}\in\mathbb{R}^{D},$$
145 which serves as a summary of information flow from the context subset x towards predicting the next 146 token after the question. Now we will define the information gain from x for a specific head.
Definition 3.2 (Information Gain). From Definition 3.1, the vector z
(ℓ,h)
147 x,q encapsulates dense semantic 148 information preserved within the internal attention mechanisms of LLMs. By applying a softmax
operation over z
(ℓ,h)
x,q , we obtain a vocabulary distribution p
(ℓ,h)
x,q ∈ R
|V |
149 . The entropy at the final 150 token is computed as
$${\mathcal{H}}^{(\ell,h)}(q_{t}\mid q_{<t},x)\triangleq-\sum_{i=1}^{|V|}p_{i}^{(\ell,h)}\log p_{i}^{(\ell,h)}.$$
$$(2)$$
$$({\mathfrak{I}})$$
151 We emphasize that entropy is calculated over the entire softmax-normalized vocabulary. This is a 152 critical distinction: hallucination often stems not from low confidence in the correct token alone, 153 but from broad misallocation of probability mass across incorrect options. Therefore, full entropy 154 measurement enables us to detect whether the model's uncertainty is genuinely reduced when 155 informative context is provided. Now to calculate the information gain provided by the subset x at 156 head h and layer ℓ, it is defined as the reduction in entropy relative to a null context (i.e., no input) 157 using equation 3,

$$\mathbf{IG}^{(\ell,h)}(x\to q)\triangleq{\mathcal{H}}^{(\ell,h)}(q_{t}\mid q_{<t},\emptyset)-{\mathcal{H}}^{(\ell,h)}(q_{t}\mid q_{<t},x),$$

where H(ℓ,h)
158 (qt | q<t, ∅) is computed solely from the model's parametric knowledge, without access 159 to any retrieved context. Summing over all heads and layers yields the total information gain using 4:

$$\mathbf{IG}(x\to q)\triangleq\sum_{\ell=1}^{L}\sum_{h=1}^{H}\mathbf{IG}^{(\ell,h)}(x\to q).$$
$$(4)$$
$$({\mathfrak{H}})$$

The quantity IG(x → q) captures the behavior of the function f
(ℓ,h)
160 : C *∪ ∅ → P*(Q), which maps 161 a context input, or its absence, to a probability distribution over the vocabulary space Q for each 162 attention head and layer. Moreover, IG(x → q) quantifies the amount of information that the context 163 x provides about the question q.

164 **Definition 3.3** (Shapley Sentence Attribution). Now, for the context passage sx = (s1, s2*, . . . , s*n) ∈ 165 C and associated question q ∈ Q, we aim to quantify the individual contribution of each sentence si 166 in the context to the model's total information gain. To do this, we use the Shapley value [23], a

136
$$\alpha^{(\ell,h)}(x,q)\triangleq\text{softmax}\left(\frac{Q^{(\ell,h)}(x,q)\,K^{(\ell,h)}(x,q)^{\top}}{\sqrt{d}}\right),$$ $$Z^{(\ell,h)}(x,q)\triangleq\alpha^{(\ell,h)}(x,q)V^{(\ell,h)}(x,q),$$
(ℓ,h)(*x, q*), (1)
where Q(ℓ,h)and K(ℓ,h)
137 denote the query and key matrices for layer ℓ and head h, respectively,
α
(ℓ,h)(*x, q*) ∈ R
T ×Tand V
(ℓ,h)(x, q) ∈ R
T ×d
138 are the value matrices with d = D/H being the per139 head dimension. Both attention weights and value vectors are computed based on the concatenated 140 subset x and question q. The resulting attention outputs are projected using equation 1 and a
head-specific output matrix W
(h)
O ∈ R
d×D 141 to obtain
$$(1)$$
$$\tilde{Z}^{(\ell,h)}(x,q)\triangleq Z^{(\ell,h)}(x,q)W_{O}^{(h)}\in\mathbb{R}^{T\times D}.$$

T ×D. (2)
,
129 **Definition 3.1** (Norm-based Attention Information). Prior research by [22] suggests that the norm of 130 the attention output serves as a meaningful proxy for the amount of information transmitted by each 131 head. We omit the output of the feedforward layers (FC), as previous work by [11] has shown that 132 these layers predominantly capture shallow distributional associations, whereas the attention layers 133 are more effectively engaged in in-context reasoning. 134 For each layer ℓ ∈ {1*, . . . , L*} and head h ∈ {1*, . . . , H*}, given an input context subset x and a 135 question q, we compute the attention output of the model V for the combined input (*x, q*) as follows:

Models CoQA QuAC SQuAD TriviaQA

AUC↑ τ ↑ PCC↑ AUC↑ τ ↑ PCC↑ AUC↑ τ ↑ PCC↑ AUC↑ τ ↑ PCC↑

Qwen2.5-3B

P(True) 0.48 0.32 0.30 0.49 0.33 0.31 0.51 0.34 0.32 0.50 0.33 0.31 Pointwise VI 0.51 0.35 0.32 0.50 0.34 0.31 0.52 0.36 0.33 0.53 0.36 0.34 Usable LI 0.67 0.45 0.41 0.66 0.44 0.40 0.68 0.45 0.42 0.64 0.43 0.40 Semantic Entropy 0.70 0.47 0.44 0.68 0.45 0.42 0.69 0.44 0.41 0.72 0.46 0.43 Loopback Lens 0.71 0.48 0.45 0.69 0.46 0.43 0.70 0.45 0.42 0.73 0.46 0.44 INSIDE 0.76 0.54 0.49 0.75 0.53 0.48 0.74 0.54 0.50 0.77 0.55 0.49 NEAR **0.85 0.65 0.64 0.84 0.66 0.65 0.86 0.67 0.66 0.85 0.66 0.65**

LLaMA3.1-8B

P(True) 0.52 0.34 0.31 0.53 0.35 0.32 0.56 0.37 0.34 0.55 0.36 0.33 Pointwise VI 0.56 0.36 0.34 0.52 0.32 0.31 0.55 0.37 0.33 0.68 0.46 0.40 Usable LI 0.74 0.49 0.44 0.69 0.46 0.41 0.71 0.47 0.43 0.63 0.45 0.40 Semantic Entropy 0.73 0.42 0.43 0.67 0.40 0.44 0.69 0.39 0.41 0.76 0.41 0.41 Loopback Lens 0.74 0.43 0.44 0.68 0.41 0.44 0.70 0.40 0.42 0.76 0.42 0.41 INSIDE 0.80 0.56 0.51 0.79 0.55 0.50 0.76 0.58 0.53 0.81 0.57 0.50 NEAR **0.85 0.66 0.61 0.84 0.65 0.60 0.86 0.68 0.63 0.85 0.67 0.60**

OPT-6.7B

P(True) 0.51 0.33 0.30 0.52 0.34 0.31 0.55 0.36 0.33 0.54 0.35 0.32 Pointwise VI 0.55 0.35 0.33 0.51 0.31 0.30 0.54 0.36 0.32 0.66 0.44 0.38 Usable LI 0.72 0.47 0.42 0.67 0.44 0.39 0.70 0.46 0.41 0.61 0.43 0.38 Semantic Entropy 0.71 0.41 0.42 0.65 0.39 0.43 0.68 0.38 0.40 0.74 0.40 0.40 Loopback Lens 0.72 0.42 0.43 0.66 0.40 0.44 0.69 0.39 0.41 0.75 0.41 0.40 INSIDE 0.78 0.54 0.49 0.77 0.52 0.48 0.74 0.56 0.51 0.79 0.55 0.48

NEAR **0.84 0.65 0.60 0.83 0.64 0.59 0.85 0.66 0.61 0.84 0.65 0.59**

167 concept from cooperative game theory that fairly assigns credit to each element based on its average 168 marginal contribution. Using the total information gain defined in Equation (5), the Shapley value for sentence si 169 is computed as:

$s_{i}$ is computed as.  $$\text{Shapley IG}_{i}\triangleq\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(n-|S|-1)!}{n!}\left[\text{IG}(S\cup\{s_{i}\}\to q)-\text{IG}(S\to q)\right],\tag{6}$$
170 where N = {1*, . . . , n*} is the set of all sentence indices in the context. For each subset S of sentences that excludes si 171 , the term inside the brackets measures the marginal increase in information gain when si 172 is added. The prefactor is the standard Shapley coefficient, which ensures that the contributions 173 are averaged fairly over all possible insertion orders of the sentences. 174 **Definition 3.4** (Sentence-level NEAR Score). The total information that can be gained from the 175 context with respect to the given question is captured by aggregating the contributions of individual 176 sentences. Using the Shapley values from Equation 6, the NEAR score is defined as:

$$\mathrm{Shapley~NEAR}(s_{x},q)\triangleq\frac{1}{n}\sum_{i=1}^{n}\mathrm{Shapley~IG}_{i},$$
$$\mathbf{\Pi}(7)$$

177 which reflects average marginal information gain from context sentences in answering the question.

## 184 **4 Properties And Bounds Of Shapley Near**

178 Thus, based on Definitions 3.1 through 3.4, Shapley NEAR 7 offers a fine-grained decomposition of 179 the total information gain, quantifying how much usable information the model extracts from sx to 180 answer the question q. The Information Gain (IG) 3 measures the contribution of each attention head 181 and layer, while the Shapley Information Gain (Shapley IG) 6 further attributes this information to 182 individual sentence segments within the context. A higher NEAR score indicates greater information 183 utility from the context, implying that the generated output is less likely to be hallucinatory.

185 This section outlines the mathematical and experimental properties of NEAR, with derivations in 186 Appendix A1. NEAR aggregates entropy-based information gain across all transformer layers and

| Methods          | AUC ↑   | Acc. ↑   | RL ↑   |           |      |      |      |
|------------------|---------|----------|--------|-----------|------|------|------|
| Methods          | AUC ↑   | τ ↑      | PCC ↑  | NEAR      | 0.85 | 0.78 | 0.82 |
| NEAR w/o Shapely | 0.79    | 0.51     | 0.48   | INSIDE    | 0.80 | 0.74 | 0.80 |
| Shapley NEAR     | 0.85    | 0.66     | 0.64   | NEAR + HC | 0.89 | 0.81 | 0.83 |
| (a)              | (b)     |          |        |           |      |      |      |

Table 2: (a) Contribution of Shapley aggregation to NEAR scores. (b) Head Clipping (HC) results for attention heads with IG < −0.05. The following heads were clipped: 349, 459, 485, 833, 955, 1007.

## Nearu (S, Q) ≤ Nearl(S, Q),

202 here, NEARU and NEARL denote NEAR scores computed over the subset U ⊆ {1*, . . . , L*} and the 203 full set L, respectively. This follows from NEAR's additive structure over head-layer pairs, ensuring 204 information accumulates monotonically as more layers are included. 205 To compute NEAR, we approximate the underlying Shapley values via Monte Carlo sampling 206 over random permutations of context sentences. Using Hoeffding's inequality[24], we derive a 207 high-probability error bound on the NEAR estimate. Specifically, with probability at least 1 − δ,

$$\left|\mathrm{\mathrm{\mathrm{NEAR}}}(s,q)-\mathrm{\mathrm{NEAR}}(s,q)\right|\leq L\cdot H\cdot\log V\cdot{\sqrt{\frac{\log(2n/\delta)}{2M}}}$$

208 where NEAR ˆ is the approximate NEAR Score using Monte Carlo estimation, n is the number of 209 sentences, M is the number of samples, L is the number of layers, H the number of heads, and V the 210 vocabulary size. Thus, the NEAR estimation error decreases with more samples and increases mildly 211 with model depth and vocabulary size.

## 212 **5 Experiments** 213 **5.1 Experimental Setup**

214 We classify unanswerable questions by computing NEAR scores to assess whether the response 215 generated by a model should be trusted in a given context, that is, whether the answer to a question 187 attention heads, with each term bounded by log V , the maximum entropy over a vocabulary of size 188 V . Thus, NEAR is theoretically bounded within [−L · H · log *V, L* · H · log V ], where L and H are 189 the number of layers and heads. In practice, it reflects cumulative entropy reduction from contextual 190 conditioning and scales as NEAR(*s, q*) ∈ O(L · H · log V ). Beyond boundedness, NEAR satisfies 191 key behavioral properties. First, it is symmetric: if two context sentences si and sj satisfy IG(S ∪ {si} → q) = IG(S ∪ {sj} → q) for all S ⊆ s \ {si, sj},
192 then their Shapley values are identical, i.e., IGi = IGj . Moreover, NEAR reflects context redundancy:
193 when S ⊆ T, the marginal information gain decreases, satisfying IG(S ∪ {si} → q) − IG(S → q) ≥ IG(T ∪ {si} → q) − IG(T → q).

194 NEAR also detects context irrelevance: if H(qt | q<t, ∅) ≈ H(qt | q<t, sx) for all subsets sx, 195 then NEAR(*s, q*) ≈ 0, indicating that the context does not provide meaningful information for 196 answering the question. We also empirically observed (Section 5) that for each layer ℓ and attention 197 head h, the following inequality holds:
IG(ℓ,h)(∅ → q) ≤ IG(ℓ,h)(s irr i → q) ≤ IG(ℓ,h)(s ans j → q),
here, s irr i denotes a context sentence irrelevant to the answer, and s ans j 198 contains the ground truth answer.

199 Empirically, NEAR scores also exhibit a monotonicity property similar to information-theoretic 200 measures: for any subset of layers U ⊆ L, the NEAR score computed over U is always less than or 201 equal to that over the full set L, as aggregating more layers cannot reduce total entropy gain:

NEAR 
w/o Sha pely Without Shapely With Shapely
(a)
Parametric Hallucination Context-Induced Hallucination Informa tion Gai n Avg. IG < 0 Avg. IG < 0 with noise Avg. IG < 0 with finetuning Avg. IG > 0 Avg. IG > 0 with noise Avg. IG > 0 with concatenation 0.0 2.5 5.0 7.5 10.0 Shapel y NEA
R

0 5 10 15 20 25 Sentence Segments 0.0 2.5 5.0 7.5 10.0
(b) (c)
Activatio n Penultimate Norm-based Attention Output Activatio n Penultimate MLP Output Activatio n Penultimate Final Layer Output 0 1000 2000 3000 4000 Neuron Index 10 0 10 20 30
(a)
0 1000 2000 3000 4000 Neuron Index 10 0 10 20 30
(b)
0 1000 2000 3000 4000 Neuron Index 10 0 10 20 30
(c) (d)
216 posed can be reliably inferred. We compare NEAR against several strong baselines, including 217 **P(True)** [25], **semantic entropy** [26], **pointwise V-information (PVI)** [14], **layer-wise information**
218 **(LI)** [18], **Loopback Lens with Sliding Window** [17], and **INSIDE** (K = 20, middle layer of 219 the LLM is considered) [12]. Each method captures a different perspective: P(True) estimates 220 model confidence in binary verification tasks; semantic entropy measures uncertainty via answer 221 diversity; PVI quantifies instance-level predictive difficulty; and LI captures entropy reduction across 222 transformer layers. We evaluate all methods on four question-answering benchmarks: CoQA [27], 223 QuAC [28], SQuAD v2.0 [29], and TriviaQA [30]. Following the setup in [9], we use the development 224 split of CoQA, validation split of QuAC, a filtered version of the SQuAD v2.0 development set 225 where is_impossible=True, and the rc-nocontext validation subset of TriviaQA with duplicates 226 removed. Experiments are conducted on three pretrained models: Qwen2.5-3B, LLaMA3.1-8B, 227 and OPT-6.7B. We report average area under the ROC curve (AUROC), Kendall's τ , and Pearson 228 correlation coefficient (PCC), computed across three independent runs. NEAR scores are estimated 229 using Monte Carlo sampling with M = 50 (Appendix A8) permutations and failure probability 230 δ = 0.01, ensuring high-confidence estimates of each context sentence's contribution to information 231 gain (further details in Appendix A3). This approximation provides a practical trade-off between 232 computational cost and estimation accuracy, with all reported results exhibiting standard deviations 233 within ±0.04.

## 234 **5.2 Results**

235 Table 1 shows the results of hallucination detection using NEAR and several baseline methods across 236 four QA datasets (CoQA, QuAC, SQuAD, and TriviaQA) and three language models (Qwen2.5-3B, 237 LLaMA3.1-8B, and OPT-6.7B). We report performance using AUROC, Kendall's τ , and Pearson 238 correlation (PCC). NEAR consistently performs the best across all datasets and models, showing 239 clear improvements over existing methods. In many cases, it outperforms the strongest baseline, 240 INSIDE, by 8–13% in AUROC and by 10–15% in correlation metrics like τ and PCC. The best 241 scores for NEAR are observed on the SQuAD dataset for all models, suggesting that SQuAD is easier 242 for LLMs to understand and answer accurately. Among the three models, LLaMA3.1-8B achieves

## 248 **6 Ablation Studies**

249 For the ablation studies, we primarily focus on the LLaMA-3.1-8B model with the CoQA dataset. 250 Results for other models and datasets are provided in Appendix A2.

251 **Do we really need to consider all layers instead of only the final layer?** Unlike methods such as 252 VI [14], which consider only final-layer outputs, our results show that important semantic information 253 is also captured in earlier layers. As illustrated in Figure 2c, both LI and NEAR scores indicate 254 that usable information accumulates progressively across inner layers. A similar trend is visible in 255 Figure 3d, where different attention heads capture varying amounts of information. This suggests 256 that focusing only on the final layer overlooks valuable signals present throughout the model. 257 Why not consider the output from the layers, as in LI**, for NEAR?** Figures 3a and 3b show the 258 activations of the self-attention and MLP components from the penultimate layer of the LLaMA 259 3.1-8B model. The sharp spikes in these plots reflect extreme internal features in the network, 260 which can cause the model to produce highly overconfident answers [12, 31]. A similar pattern 261 of overconfidence is also clearly visible in the layer output shown in Figure 3c. We observed this 262 behavior consistently across nearly all layers and LLMs, aligning with the findings of [11]. Based on 263 this evidence, we choose to focus on norm-based attention outputs rather than raw layer activations.

264 **Detection of Parametric and Context-Induced Hallucinations from NEAR Scores.** Let si ∈ A/ (q)
265 be a context sentence that does not contain the correct answer to question q, where A(q) denotes the 266 set of answer-containing sentences. Ideally, such a sentence should contribute no useful information, and the information gain under attention head (*ℓ, h*) should satisfy IG(ℓ,h)
267 (si → q) ≈ 0. This follows 268 from equation 4, which becomes negligible when conditioning on si does not reduce uncertainty, i.e., H(ℓ,h)(qt | q<t, si) ≈ H(ℓ,h)
269 (qt | q<t, ∅). However, we find that even when si ∈ A/ (q), NEAR 270 scores can be negative (IGi < 0) or positive (IGi > 0). A negative score indicates the model becomes more uncertain when conditioned on si 271 , meaning the context harms rather than helps, this is 272 *parametric hallucination*. A positive score, despite the absence of the answer, implies that the context 273 falsely boosts confidence, this is *context-induced hallucination*. Such cases arise due to in-context 274 learning, the model interprets partial or stylistically similar information as relevant, leading to reduced 275 entropy and overconfidence. To validate this, we measured the mean negative NEAR scores across all 276 context pieces. Adding random noisy text (similar technique used in [11]) caused negligible change, 277 suggesting that the observed negativity is not due to noise or formulation errors. However, fine-tuning 278 the model on CoQA significantly increased negative NEAR scores, indicating that the model had 279 learned to rely more on context, which led to greater uncertainty when misleading context was 280 introduced, confirming parametric hallucination. For context-induced hallucination, we computed 281 mean positive NEAR scores for non-answer sentences. While adding random noise had little effect, 282 appending misleading but partially aligned segments of the rest of the context led to a sharp increase 283 in NEAR scores. This confirms that NEAR effectively captures how misleading context increases 243 the highest overall performance, ahead of Qwen2.5-3B and OPT-6.7B, especially when used with 244 NEAR. This suggests that stronger pre-trained models can lead to better hallucination detection when 245 combined with effective methods like NEAR. We also evaluated the methods after fine-tuning on the 246 dataset; the results are presented in Appendix A4 and quantitative examples without finetuning in 247 Appendix A9. We also tested NEAR on generalized tasks, detailed in Appendix A6. 284 confidence in incorrect predictions. The results are shown in Figure 2b. However, these hallucinations 285 do not significantly affect the overall reliability of Shapley NEAR, as demonstrated in Appendix A5. 286 **What Should Be the Threshold Value for NEAR to Segregate Hallucinated Answers?** A key step 287 in using NEAR for hallucination detection is choosing an effective threshold to separate answerable 288 from hallucinated responses. We evaluate classification accuracy by sweeping thresholds across 289 quantiles: 0, 0.5 × Q1, Q1, Median, Q3, and 1.5 × Q3. As shown in Figure 4, the first quartile 290 (Q1) consistently yields the best accuracy across models (LLaMA-3.1-8B, OPT-6.7B, Qwen2.5-3B) 291 and datasets (CoQA, QuAC, SQuAD, TriviaQA). In contrast, thresholds near 0 or 1.5 × Q3 reduce 292 performance. Based on this, we use Q1 as the default NEAR threshold for all experiments. 293 **Effect of Shapley Combination on NEAR.** We evaluated the effect of Shapley aggregation in NEAR,
294 comparing it to a greedy method that ranks sentences by standalone gain (without Shapely attribution). 295 As shown in Table 2a, Shapley improves Kendall's τ (0.51 → 0.66), PCC (0.48 → 0.64), and AUC 296 (0.79 → 0.85), highlighting the benefit of permutation averaging for robust attribution. Figure 2a 297 shows Shapley downweights irrelevant segments and upweights answer-relevant ones.

298 **Clipping Heads showing Parametric Hallucination** To further demonstrate the effectiveness of our 299 framework in identifying hallucination-prone attention heads, we clipped all heads in LLaMA-3.1-8B 300 (on the CoQA dataset) with IG values below half the most negative score. This conservative threshold 301 avoids pruning heads with mildly negative IG, which may still contribute useful information (see 302 Figure 3d). We compared our method to INSIDE (EigenScore + Feature Clipping) with a fixed 303 threshold of 0.5, evaluating AUROC, accuracy, and ROUGE-L (computed between the given and 304 generated answers). For both NEAR and NEAR+HC (Head Clipping), we used the first quartile 305 (Q1) as the classification threshold. As shown in Table 2b, applying head clipping led to consistent 306 improvements across all metrics. All results are averaged over three independent runs, with standard 307 deviation < 0.3. These findings align with prior work [32–34], which suggests that not all attention 308 heads contribute meaningfully to model output.

## 309 **7 Related Work**

310 Recent studies increasingly leverage attention patterns to detect hallucinations in language models. 311 Lookback Lens [35] introduces a "lookback ratio" that contrasts attention on the input context 312 versus generated tokens, enabling lightweight yet competitive classification. Spectral methods [36] 313 treat attention maps as graphs and extract top eigenvalues from the attention Laplacian to signal 314 abnormality. LLM-Check [37] integrates internal signals, including attention matrices and hidden 315 states, but its accuracy is sensitive to the chosen layer. Beyond attention, entropy-based approaches 316 such as Semantic Entropy [26] and Semantic Entropy Probes [38] estimate model uncertainty via 317 output clustering or learned probes. Hidden-state probing [15, 39] also helps identify token-level 318 unreliability. More recently, mechanistic interpretability has been applied to hallucination detection: 319 some methods regress over parametric versus contextual signals [40], while others fine-tune based on 320 internal layer projections [41]. In contrast, our framework is fully plug-and-play - requiring neither 321 retraining nor architectural modifications - while offering fine-grained attention-level attribution.

## 322 **8 Conclusion**

323 We propose **Shapley NEAR**, an interpretable framework that detects hallucinations in LLMs by 324 attributing entropy-based information flow across attention heads and layers. It leverages attention 325 norms and Shapley values for sentence-level attribution, outperforming baselines and distinguishing 326 between *parametric* and *context-induced* hallucinations. A test-time head clipping step further 327 reduces overconfident outputs without retraining. Shapley NEAR offers a principled bridge between 328 attribution and internal model dynamics. Limitations are noted in Appendix A10.

## 329 **References**

330 [1] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, 331 Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to 332 follow instructions with human feedback. *Advances in neural information processing systems*, 333 35:27730–27744, 2022. 334 [2] J OpenAI Achiam, S Adler, S Agarwal, L Ahmad, I Akkaya, FL Aleman, D Almeida, 335 J Altenschmidt, S Altman, S Anadkat, et al. Gpt-4 technical report. arxiv. *arXiv preprint* 336 *arXiv:2303.08774*, 2023.

337 [3] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, 338 Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. 339 *ACM computing surveys*, 55(12):1–38, 2023.

340 [4] Roi Cohen, May Hamri, Mor Geva, and Amir Globerson. Lm vs lm: Detecting factual errors 341 via cross examination. *arXiv preprint arXiv:2305.13281*, 2023. 342 [5] Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, 343 and Peter J Liu. Out-of-distribution detection and selective generation for conditional language 344 models. *arXiv preprint arXiv:2209.15558*, 2022.

345 [6] Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu, 346 and Lei Ma. Look before you leap: An exploratory study of uncertainty measurement for large 347 language models. *arXiv preprint arXiv:2307.10236*, 2023.

348 [7] Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya 349 Kailkhura, and Kaidi Xu. Shifting attention to relevance: Towards the predictive uncertainty 350 quantification of free-form large language models. *arXiv preprint arXiv:2307.01379*, 2023.

351 [8] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances 352 for uncertainty estimation in natural language generation. *arXiv preprint arXiv:2302.09664*,
353 2023. 354 [9] Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty 355 quantification for black-box large language models. *arXiv preprint arXiv:2305.19187*, 2023. 356 [10] Kaitlyn Zhou, Dan Jurafsky, and Tatsunori Hashimoto. Navigating the grey area: How 357 expressions of uncertainty and overconfidence affect language models. *arXiv preprint* 358 *arXiv:2302.13439*, 2023. 359 [11] Lei Chen, Joan Bruna, and Alberto Bietti. Distributional associations vs in-context reasoning: 360 A study of feed-forward and attention layers. In *The Thirteenth International Conference on* 361 *Learning Representations*, 2025. 362 [12] Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping 363 Ye. Inside: Llms' internal states retain the power of hallucination detection. *arXiv preprint* 364 *arXiv:2402.03744*, 2024. 365 [13] Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of 366 usable information under computational constraints. *arXiv preprint arXiv:2002.10689*, 2020. 367 [14] Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. Understanding dataset difficulty with 368 V-usable information. In *International Conference on Machine Learning*, pages 5988–6008. 369 PMLR, 2022. 370 [15] Amos Azaria and Tom Mitchell. The internal state of an llm knows when it's lying. *arXiv* 371 *preprint arXiv:2304.13734*, 2023.

372 [16] Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha, Priyatham 373 Kattakinda, and Soheil Feizi. Llm-check: Investigating detection of hallucinations in large 374 language models. *Advances in Neural Information Processing Systems*, 37:34188–34216, 2024. 375 [17] Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James Glass. 376 Lookback lens: Detecting and mitigating contextual hallucinations in large language models 377 using only attention maps. *arXiv preprint arXiv:2407.07071*, 2024. 378 [18] Hazel Kim, Adel Bibi, Philip Torr, and Yarin Gal. Detecting llm hallucination through layer379 wise information deficiency: Analysis of unanswerable questions and ambiguous prompts. 380 *arXiv preprint arXiv:2412.10246*, 2024. 381 [19] Claude E Shannon. A mathematical theory of communication. *The Bell system technical* 382 *journal*, 27(3):379–423, 1948.

383 [20] Nicholas Pippenger. Reliable computation by formulas in the presence of noise. IEEE Transac384 *tions on Information Theory*, 34(2):194–197, 1988. 385 [21] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt.

386 Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. *arXiv* 387 *preprint arXiv:2211.00593*, 2022. 388 [22] Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a 389 weight: Analyzing transformers with vector norms. *arXiv preprint arXiv:2004.10102*, 2020.

390 [23] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. 391 *Advances in neural information processing systems*, 30, 2017. 392 [24] Wassily Hoeffding. Probability inequalities for sums of bounded random variables. The 393 *collected works of Wassily Hoeffding*, pages 409–426, 1994. 394 [25] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, 395 Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language 396 models (mostly) know what they know. *arXiv preprint arXiv:2207.05221*, 2022. 397 [26] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in 398 large language models using semantic entropy. *Nature*, 630(8017):625–630, 2024. 399 [27] Siva Reddy, Danqi Chen, and Christopher D Manning. Coqa: A conversational question 400 answering challenge. *Transactions of the Association for Computational Linguistics*, 7:249–266, 401 2019. 402 [28] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and 403 Luke Zettlemoyer. Quac: Question answering in context. *arXiv preprint arXiv:1808.07036*, 404 2018. 405 [29] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions 406 for machine comprehension of text. *arXiv preprint arXiv:1606.05250*, 2016. 407 [30] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large 408 scale distantly supervised challenge dataset for reading comprehension. *arXiv preprint* 409 *arXiv:1705.03551*, 2017. 410 [31] Yiyou Sun, Chuan Guo, and Yixuan Li. React: Out-of-distribution detection with rectified 411 activations. *Advances in neural information processing systems*, 34:144–157, 2021. 412 [32] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? 413 *Advances in neural information processing systems*, 32, 2019.

414 [33] Hongyu Gong, Yun Tang, Juan Pino, and Xian Li. Pay better attention to attention: Head 415 selection in multilingual and multi-domain sequence modeling. *Advances in Neural Information* 416 *Processing Systems*, 34:2668–2681, 2021. 417 [34] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head 418 self-attention: Specialized heads do the heavy lifting, the rest can be pruned. *arXiv preprint* 419 *arXiv:1905.09418*, 2019. 420 [35] Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James R. 421 Glass. Lookback lens: Detecting and mitigating contextual hallucinations in large lan422 guage models using only attention maps. In Yaser Al-Onaizan, Mohit Bansal, and Yun423 Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natu424 *ral Language Processing*, pages 1419–1436, Miami, Florida, USA, November 2024. As425 sociation for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.84. URL 426 https://aclanthology.org/2024.emnlp-main.84/. 427 [36] Jakub Binkowski, Denis Janiak, Albert Sawczyn, Bogdan Gabrys, and Tomasz Kajdanowicz.

428 Hallucination detection in llms using spectral features of attention maps. *arXiv preprint* 429 *arXiv:2502.17598*, 2025.

430 [37] Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha, Priyatham Kat431 takinda, and Soheil Feizi. Llm-check: Investigating detection of hallucinations in large language 432 models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, 433 editors, *Advances in Neural Information Processing Systems*, volume 37, pages 34188–34216. 434 Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/ 435 paper/2024/file/3c1e1fdf305195cd620c118aaa9717ad-Paper-Conference.pdf. 436 [38] Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal.

437 Semantic entropy probes: Robust and cheap hallucination detection in llms. *arXiv preprint* 438 *arXiv:2406.15927*, 2024.

439 [39] Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem Shelmanov, Sergey Petrakov, Haonan 440 Li, Hamdy Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, Alexander Panchenko, Timothy 441 Baldwin, et al. Fact-checking the output of large language models via token-level uncertainty 442 quantification. *arXiv preprint arXiv:2403.04696*, 2024. 443 [40] Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang Song, Jun Xu, Xiao Zhang, Weijie Yu, and 444 Han Li. Redeep: Detecting hallucination in retrieval-augmented generation via mechanistic 445 interpretability. In *International Conference on Learning Representations (ICLR)*, 2025. 446 [41] Lei Yu, Meng Cao, Jackie Chi Kit Cheung, and Yue Dong. Mechanistic understanding and 447 mitigation of language model non-factual hallucinations. In *Findings of the Association for* 448 *Computational Linguistics: EMNLP 2024*, pages 7943–7956, 2024.

449 [42] Shashi Narayan, Shay B Cohen, and Mirella Lapata. Don't give me the details, just the 450 summary! topic-aware convolutional neural networks for extreme summarization. *arXiv* 451 *preprint arXiv:1808.08745*, 2018. 452 [43] Abhika Mishra, Akari Asai, Vidhisha Balachandran, Yizhong Wang, Graham Neubig, Yulia 453 Tsvetkov, and Hannaneh Hajishirzi. Fine-grained hallucination detection and editing for 454 language models. *arXiv preprint arXiv:2401.06855*, 2024.

455 **Appendix** 456 **Contents**
457 **1 Introduction** 1 458 **2 Background** 2

459 **3 Shapley NEAR: Norm-basEd Attention-wise usable infoRmation 3** 460 **4 Properties and Bounds of Shapley NEAR 5** 464 **6 Ablation Studies** 8 465 **7 Related Work** 9

| 461   | 5   | Experiments        | 6   |    |
|-------|-----|--------------------|-----|----|
| 462   | 5.1 | Experimental Setup |     | 6  |
| 463   | 5.2 | Results            | 7   |    |

466 **8 Conclusion** 9 467 **Appendix** 13 474 **A3 Experimental Setup and Hyperparameters 17** 475 **A4 Experimental Results with model finetuning 18** 479 **A7 Algorithm** 20

480 **A8 Effect of Number of Permutations on NEAR Stability 21** 481 **A9 Qualitative Examples 21** 482 **A10Limitations** 27

| 468   | A1 Derivation of Theoretical Properties and Error Bounds for Shapley NEAR Scores   | 14   |    |
|-------|------------------------------------------------------------------------------------|------|----|
| 469   | A1.1 Properties Derivation                                                         |      | 14 |
| 470   | A1.2 Estimation Error Bound for Monte Carlo NEAR                                   |      | 15 |

| 471   | A2 Ablation Studies for rest of the Datasets                                  | 16   |    |
|-------|-------------------------------------------------------------------------------|------|----|
| 472   | A2.1 Layer-wise Information Trends in Qwen2.5-3B and OPT-6.7B                 | 16   |    |
| 473   | A2.2 Analyzing Parametric and Context-Induced Hallucinations with NEAR Scores |      | 16 |

| 476   | A5 Robustness of NEAR Against Parametric and Context-Induced Hallucinations.   | 19   |
|-------|--------------------------------------------------------------------------------|------|

| 477   | A6 Generalization to Other Tasks       | 20   |
|-------|----------------------------------------|------|
| 478   | A6.1 Comparison with LLM-Check on FAVA | 20   |

## 483 **A1 Derivation Of Theoretical Properties And Error Bounds For Shapley** 484 **Near Scores** 485 **A1.1 Properties Derivation**

486 We begin by formally defining the NEAR score. Let the context passage be x = {x1, x2*, . . . , x*n},
487 consisting of n disjoint sentences, and let q denote the corresponding question. For a transformer 488 model with L layers and H attention heads per layer, the NEAR score is given by

$${\rm NEAR}(x,q)=\frac{1}{n}\sum_{i=1}^{n}{\rm IG}_{i},\tag{8}$$

where IGi denotes the Shapley value assigned to sentence xi 489 , measuring its marginal contribution to 490 the model's information gain at the final prediction token.

491 The information gain for a subset of context sentences xS ⊆ x is defined as

$${\bf IG}(x_{S}\to q)=\sum_{\ell=1}^{L}\sum_{h=1}^{H}\left[{\cal H}^{(\ell,h)}(q_{t}\mid\emptyset)-{\cal H}^{(\ell,h)}(q_{t}\mid x_{S})\right],\tag{9}$$

where H(ℓ,h)
492 (qt | xS) denotes the entropy of the softmax-normalized vocabulary distribution at the 493 final token qt, computed using context subset xS.

A fundamental property of entropy is that for any discrete distribution p ∈ R
V
494 over vocabulary size 495 V , the Shannon entropy is bounded as 0 ≤ H(p) ≤ log V, (10)
496 where the minimum is achieved for deterministic distributions and the maximum for uniform distribu497 tions. Applying this to attention outputs, it follows that

$l(-)\,\leq\,\log V$ . 
$$0\leq{\mathcal{H}}^{(\ell,h)}(q_{t}\mid x_{S})\leq\log V,$$
$$(11)$$

$$(12)$$
$$\tau_{\mathrm{SS}}$$
$$|\mathbf{IG}(x_{S}\to q)|\leq L\cdot H\cdot\log V.$$
$$(13)$$

$$(14)$$
0 ≤ H(ℓ,h)(qt | xS) ≤ log V, (11)
498 for any layer ℓ, head h, and context subset xS.

499 Thus, the maximum change in entropy across any head-layer combination is bounded by

$$\left|{\mathcal{H}}^{(\ell,h)}(q_{t}\mid\emptyset)-{\mathcal{H}}^{(\ell,h)}(q_{t}\mid x_{S})\right|\leq\log V,$$
 ≤ log V, (12)
500 implying that the total information gain satisfies
|IG(xS → q)| ≤ L · H · log V. (13)
The Shapley value IGi for a sentence xi 501 is computed by averaging its marginal contributions over all 502 subsets of other sentences:

$$\mathrm{Id}_{i}=\sum_{S\subseteq N\setminus\{i\}}{\frac{|S|!(n-|S|-1)!}{n!}}\left[\mathrm{IG}(S\cup\{x_{i}\}\to q)-\mathrm{IG}(S\to q)\right],$$
$\text{nd}$ thus the 
503 where N = {1*, . . . , n*} indexes the context sentences. Given the bound in Eq. (13), it immediately 504 follows that |IGi| ≤ L · H · log V, (15)
505 and thus the NEAR score itself is bounded by 506 Moreover, the asymptotic growth of NEAR with respect to model size is characterized by

NEAR(*x, q*) ∈ O(L · H · log V ), (17)
507 indicating that larger models with more layers and heads can potentially exhibit larger NEAR scores. 508 In practice, NEAR scores tend to remain significantly below their theoretical maxima because 509 softmax-normalized attention distributions are rarely fully uniform or fully deterministic. Confident 510 predictions (low entropy) result in large NEAR scores, while uncertain or irrelevant contexts yield 511 low NEAR values.

$$|\mathrm{IG}_{i}|\leq L\cdot H\cdot\log V,$$ Indeed by 
itself is bounded by 
$$-{\boxed{L\cdot H\cdot\log V}}\leq\mathrm{NEAR}(x,q)\leq{\boxed{L\cdot H\cdot\log V}}.$$
$$(15)$$

$$(16)$$
$$(17)^{\frac{1}{2}}$$

512 **Symmetry of Shapley-Based NEAR** NEAR preserves the symmetry property of Shapley values.

If two sentences xi and xj have identical marginal contributions across all subsets S ⊆ x \ {xi 513 , xj},
514 then their Shapley attributions are equal:

$$\mathbf{I}\mathbf{G}_{i}=\mathbf{I}\mathbf{G}_{j}.$$
$$(18)$$
$$(20)^{\frac{1}{2}}$$
IGi = IGj . (18)
515 Thus, NEAR treats functionally equivalent sentences identically, ensuring fair attribution. 516 **Context Redundancy and Diminishing Marginal Gains** Due to the submodularity of entropy, the 517 marginal information gain diminishes as context grows. Formally, for any S ⊆ T,
IG(S ∪ {xi} → q) − IG(S → q) ≥ IG(T ∪ {xi} → q) − IG(T → q). (19)
518 Thus, redundant sentences with overlapping information have smaller Shapley attributions and lower 519 contributions to NEAR. 520 **Zero NEAR for Context-Free Questions** If the context x provides no useful information for 521 answering q, the entropy remains unchanged after conditioning:

$${\mathcal{H}}(q_{t}\mid\emptyset)\approx{\mathcal{H}}(q_{t}\mid x_{S}),\quad\forall x_{S}\subseteq x,$$
$$\mathrm{NEAR}(x,q)\approx0,$$
$$(21)$$
H(qt | ∅) ≈ H(qt | xS), ∀xS ⊆ x, (20)
522 leading to NEAR(*x, q*) ≈ 0, (21)
523 indicating that the model's uncertainty is unaffected by the context. 524 **A1.2 Estimation Error Bound for Monte Carlo NEAR** 525 Exactly computing Shapley values is computationally infeasible due to the n! permutations required.

526 Thus, we approximate Shapley values by Monte Carlo sampling over M random permutations.

527 The approximate Shapley value is given by

$$\mathrm{IG}_{i}=\frac{1}{M}\sum_{j=1}^{M}\left[\mathrm{IG}(S_{i}^{(j)}\cup\{x_{i}\})-\mathrm{IG}(S_{i}^{(j)})\right],$$
$$(22)$$
$$(23)$$

where S
(j)
iis the predecessor set of xi 528 in the j-th sampled permutation.

529 Assuming each marginal contribution satisfies

$$\cup\left\{x_{i}\right\})-\mathrm{IC}$$

|IG(S ∪ {xi}) − IG(S)| ≤ B = L · H · log V, (23)
530 Hoeffding's inequality [24] gives that, for any δ > 0,

$$=L\cdot H\cdot\log V,$$
$\subset$ . 

$$\left|\hat{\mathrm{IG}}_{i}-\mathrm{IG}_{i}\right|\leq B{\sqrt{\frac{\log(2/\delta)}{2M}}},$$
$$(24)$$

532 Since NEAR is an average over n sentences, applying the union bound yields

$$\left|\mathrm{\mathrm{\mathrm{\mathrm{NEAR}}}}(x,q)-\mathrm{\mathrm{\mathrm{NEAR}}}(x,q)\right|\leq B{\sqrt{\frac{\log(2n/\delta)}{2M}}}.$$

533 Thus, with probability at least 1 − δ,

$$\boxed{\mathrm{NEAR}(x,q)-\mathrm{NEAR}(x,q)}\leq L\cdot H\cdot\log V\cdot{\sqrt{\frac{\log(2n/\delta)}{2M}}}$$

This bound shows that the NEAR approximation error decays as O
qlog n M

534 , making estimation 535 increasingly accurate with more samples while growing mildly with model complexity and vocabulary 536 size.

$$(25)$$

$$(26)$$

531 with probability at least 1 − δ.

0 2 4 CoQA QuAC
0.0 2.5 5.0 7.5 CoQA QuAC
Attention-heads Shapley N
EAR Scores Attention-heads Shapley N
EAR Score s 0 250 500 750 1000 0 2 4 0 200 400 600 0.0 2.5 5.0 7.5 SQuAD
TriviaQA
TriviaQA
SQuAD
0 200 400 600 0 250 500 750 1000
(a)
(b)
0.0 0.5 1.0 CoQA QuAC
0.05 0.00 0.05 0.10 CoQA QuAC
Layers Shapley NEAR Sc ores Layers Shapley NEAR Sc ores 0 10 20 30 0.0 0.5 1.0 SQuAD
TriviaQA
TriviaQA
0 10 20 30 0.05 0.00 0.05 0.10 SQuAD
0 10 20 30 0 10 20 30
(a)
(b)

## 537 **A2 Ablation Studies For Rest Of The Datasets** 538 **A2.1 Layer-Wise Information Trends In Qwen2.5-3B And Opt-6.7B**

539 Unlike methods such as VI [14], which rely solely on final-layer outputs, our experiments with 540 Qwen2.5-3B and OPT-6.7B across CoQA, QuAC, SQuAD, and TriviaQA reveal that significant 541 semantic information emerges well before the final layer. As shown in Figure 6a and Figure 6b, both 542 LI and NEAR scores accumulate progressively from early to later layers, highlighting that inner layers 543 contribute meaningfully to usable information for Qwen2.5 3B and OPT6.7 respectively. Additionally, 544 attention head analysis in these models (Figure 5a and Figure 5b) demonstrates substantial variance 545 in information captured by different heads, reinforcing that attention dynamics vary widely across 546 layers and heads. These observations confirm that limiting interpretability to the final layer overlooks 547 critical intermediate representations and that capturing attention-driven signals across all layers is 548 essential for reliable attribution.

## 549 **A2.2 Analyzing Parametric And Context-Induced Hallucinations With Near Scores**

550 To better understand the origin of hallucinations, we analyze NEAR scores assigned to context 551 sentences that do not contain the ground-truth answer. Let si ∈ A / (q), where A(q) denotes the 552 minimal set of answer-supporting sentences for a given question q. Ideally, such irrelevant sentences 553 should yield zero usable information, implying that the entropy before and after conditioning remains approximately equal. This leads to an information gain of zero: IG(ℓ,h)
554 (si → q) ≈ 0. However, 555 empirical findings across all four QA datasets—CoQA, QuAC, SQuAD, and TriviaQA—demonstrate that even when si ∈ A/ (q), the NEAR attribution IGi 556 is often either significantly negative or positive.

557 These deviations allow us to distinguish between two types of hallucination.

If IGi < 0, it indicates that the entropy after conditioning on si 558 is higher than that with no context, i.e., 559 H(qt | si) > H(qt | ∅). This suggests that the model becomes more uncertain due to misleading con-

Parametric Hallucination Context-Induced Hallucination Info rm ati on Gai n Avg. IG < 0 Avg. IG < 0 with noise Avg. IG < 0 with finetuning Avg. IG > 0 Avg. IG > 0 with noise Avg. IG > 0 with concatenation
(a)
Parametric Hallucination Context-Induced Hallucination Info rm ati on Gai n Avg. IG < 0 Avg. IG < 0 with noise Avg. IG < 0 with finetuning Avg. IG > 0 Avg. IG > 0 with noise Avg. IG > 0 with concatenation
(b)
560 text overriding its parametric knowledge—a behavior we term *parametric hallucination*. Conversely, 561 if IGi > 0 despite si ∈ A/ (q), the model incorrectly gains confidence due to spurious semantic cues 562 or surface-level similarities. This phenomenon is referred to as *context-induced hallucination*. 563 Figures 7a and 7b visually depict these effects by comparing NEAR scores before and after perturba564 tions, such as noise injection or model fine-tuning. These experiments confirm that NEAR faithfully 565 captures both types of hallucination via its attention-wise decomposition of usable information.

566 **Experimental Setup.** To validate this decomposition, we analyze NEAR attributions on CoQA,
567 QuAC, SQuAD, and TriviaQA using LLaMA-3.1-8B, OPT-6.7B, and Qwen2.5-3B. For each data568 point, we extract context segments si ∈ A/ (q) and compute:
MeanNeg = Esi∈A/ (q)[IGi| IGi < 0], MeanPos = Esi∈A/ (q)[IGi| IGi > 0].

569 We run two ablations to support the hypothesis:
570 1. **Random Noise Injection:** Injecting randomly sampled tokens into si decreases the mag571 nitude of MeanNeg and MeanPos, indicating that noise alone does not explain strong 572 deviations in NEAR. 573 2. **Fine-tuning:** Fine-tuning the model on CoQA increases |MeanNeg|, showing heightened 574 model sensitivity to misleading context after alignment, and thus more pronounced paramet575 ric hallucinations. 576 **Conclusion.** These results confirm that NEAR scores reflect two distinct modes of hallucination:
Parametric Hallucination ⇐⇒ Context increases entropy (IGi < 0),
577 Context-Induced Hallucination ⇐⇒ Spurious entropy reduction (IGi > 0, si ∈ A/ (q)).

578 Therefore, NEAR provides a faithful and granular decomposition of hallucination signals within the 579 model's internal reasoning.

## 580 **A3 Experimental Setup And Hyperparameters**

581 We evaluated our method using four standard QA benchmarks: CoQA, QuAC, SQuAD, and Trivi582 aQA, across three pretrained language models: LLaMA-3.1-8B, OPT-6.7B, and Qwen2.5-3B. For 583 each model–dataset pair, NEAR scores were computed by aggregating information gain across all 584 transformer layers and attention heads. Attention outputs were taken at the final token of each 585 question, and entropy was calculated from the softmax-normalized vocabulary logits. Sentence-level 586 context segmentation was applied consistently across datasets. 587 To efficiently estimate Shapley values, we used Monte Carlo sampling with M = 50 random 588 permutations per example. We set δ = 0.01, and bounded the estimation error using:

$$\left|\mathrm{\mathrm{NEAR}}(x,q)-\mathrm{\mathrm{NEAR}}(x,q)\right|\leq L\cdot H\cdot\log V\cdot{\sqrt{\frac{\log(2n/\delta)}{2M}}},$$
2M, (27)
$$(27)$$

Models CoQA QuAC SQuAD TriviaQA

AUC τ PCC AUC τ PCC AUC τ PCC AUC τ PCC

Qwen2.5-3B

P(True) 0.58 0.38 0.36 0.59 0.39 0.37 0.61 0.40 0.38 0.60 0.39 0.37 Pointwise VI 0.61 0.42 0.38 0.60 0.41 0.37 0.62 0.43 0.39 0.63 0.43 0.40 Usable LI 0.75 0.51 0.47 0.74 0.50 0.46 0.76 0.51 0.48 0.72 0.49 0.46 Semantic Entropy 0.78 0.54 0.50 0.76 0.52 0.48 0.77 0.51 0.47 0.80 0.53 0.49 INSIDE 0.84 0.60 0.56 0.83 0.59 0.55 0.82 0.60 0.57 0.85 0.61 0.56 NEAR **0.91 0.71 0.70 0.90 0.72 0.71 0.92 0.73 0.72 0.91 0.72 0.71**

LLaMA3.1-8B

P(True) 0.63 0.40 0.36 0.64 0.41 0.37 0.67 0.43 0.39 0.66 0.42 0.37 Pointwise VI 0.67 0.43 0.40 0.63 0.39 0.37 0.66 0.44 0.39 0.79 0.53 0.46 Usable LI 0.83 0.55 0.50 0.78 0.52 0.47 0.80 0.53 0.49 0.72 0.51 0.46 Semantic Entropy 0.82 0.48 0.49 0.76 0.46 0.50 0.79 0.45 0.47 0.86 0.47 0.47 INSIDE 0.89 0.62 0.57 0.88 0.61 0.56 0.85 0.64 0.59 0.90 0.63 0.56 NEAR **0.91 0.73 0.68 0.90 0.72 0.67 0.92 0.74 0.70 0.91 0.73 0.67**

OPT-6.7B

P(True) 0.60 0.39 0.36 0.61 0.40 0.37 0.64 0.42 0.38 0.63 0.41 0.37 Pointwise VI 0.64 0.41 0.38 0.60 0.37 0.36 0.63 0.42 0.38 0.75 0.51 0.44 Usable LI 0.81 0.53 0.48 0.76 0.51 0.46 0.79 0.52 0.47 0.70 0.51 0.44 Semantic Entropy 0.80 0.46 0.47 0.74 0.44 0.48 0.77 0.43 0.45 0.83 0.45 0.45 INSIDE 0.87 0.62 0.56 0.86 0.60 0.55 0.83 0.63 0.58 0.88 0.62 0.55 NEAR **0.90 0.73 0.67 0.89 0.72 0.66 0.91 0.74 0.68 0.90 0.73 0.66**

589 where L is the number of layers, H the number of heads per layer, V the vocabulary size, and n the 590 number of context segments. 591 To study parametric hallucinations, we fine-tuned each model on CoQA using the AdamW optimizer with a learning rate of 2 × 10−5 592 , batch size 8, weight decay 0.01, and 2 training epochs with 500 593 warmup steps. Training was performed on NVIDIA A100 80GB GPUs using PyTorch 2.1 and 594 DeepSpeed ZeRO Stage 2, with mixed-precision (bf16) training enabled. 595 We report mean NEAR scores on context segments with and without the ground-truth answer, based 596 on 10,000 sampled questions. These controlled experiments show that NEAR scores are robust 597 indicators of hallucination, effectively capturing model uncertainty and context influence.

## 598 **A4 Experimental Results With Model Finetuning**

599 **Hallucination Detection Results after Fine-Tuning.** Table 3 presents the hallucination detection 600 performance of various uncertainty estimation methods across four QA benchmarks (CoQA, QuAC, 601 SQuAD, and TriviaQA) and three LLMs (Qwen2.5-3B, LLaMA3.1-8B, and OPT-6.7B), after fine602 tuning. The evaluation metrics include area under the ROC curve (AUC), Kendall's τ , and Pearson 603 correlation coefficient (PCC). 604 Fine-tuning consistently improves the performance of all methods across all models and datasets. 605 Notably, our proposed method **NEAR** continues to outperform all baselines with a substantial margin. 606 On average, NEAR achieves AUC scores above 0.90 across all datasets, with Kendall's τ and PCC 607 also reaching peak values around 0.72–0.74, indicating both strong rank-order and linear correlation 608 with ground truth hallucination labels. Other methods such as **INSIDE** and **Semantic Entropy** also 609 benefit from fine-tuning but remain 4–6 points behind NEAR in AUC and show lower correlation 610 coefficients. For instance, on the SQuAD dataset with the LLaMA3.1-8B model, NEAR achieves 611 an AUC of 0.92 compared to 0.85 from INSIDE and 0.79 from Semantic Entropy. Similarly, in 612 TriviaQA, NEAR maintains a consistent advantage across all metrics and models. 613 **Experimental Setup.** Each model was fine-tuned using the train split of the corresponding dataset 614 and evaluated on its validation split. We used the AdamW optimizer with a learning rate of 2 × 10−5 615 , weight decay of 0.01, batch size of 8, and trained for 2 epochs with 500 warmup steps 616 and early stopping. Training was performed on NVIDIA A100 80GB GPUs using DeepSpeed ZeRO 617 Stage 2 and bf16 precision. Shapley value estimates were computed using Monte Carlo sampling 618 with M = 50 random permutations per input. All reported evaluation metrics are averaged over 3 619 independent runs, with standard deviations within ±0.03.

## 620 **A5 Robustness Of Near Against Parametric And Context-Induced** 621 **Hallucinations.**

622 While NEAR captures both parametric and context-induced hallucinations at the sentence level, it 623 is crucial to verify that such artifacts do not dominate or distort the final information attribution.

624 Ideally, context segments that do not contain the correct answer should have NEAR scores near 625 zero. However, due to model pretraining effects (parametric hallucination) and contextual mimicry 626 (context-induced hallucination), small negative or positive NEAR values can occur even without the 627 ground truth answer. 628 To evaluate the robustness of NEAR, we formally partition the context into sentences that contain the 629 answer (Sans) and those that do not (Snon-ans). The total information gain decomposes as

$$\mathbf{IG}(x\to q)=\sum_{i\in S_{\mathrm{am}}}\mathbf{IG}_{i}+\sum_{j\in S_{\mathrm{man}}}\mathbf{IG}_{j},$$
$$(28)^{\frac{1}{2}}$$

$$(29)^{\frac{1}{2}}$$
IGj , (28)
where IGi denotes the Shapley value of sentence xi 630 . We then define the *dominance ratio*:

Dominance Ratio =Mean(IGi, i ∈ Sans) |Mean(IGj , j ∈ Snon-ans)|
, (29)
631 which quantifies whether true answer-supporting information overwhelms hallucination artifacts. 632 **Experimental Setup.** We conduct experiments across three model families: LLaMA-3.1-8B, OPT- 633 6.7B, and Qwen2.5-3B. Evaluations are performed on four datasets: CoQA, QuAC, SQuAD v1.1, 634 and TriviaQA. Each context passage is segmented into sentences, and NEAR scores are computed per 635 sentence. Context sentences are manually aligned with ground truth answers using string matching 636 and fuzzy heuristics.

637 NEAR scores are computed using M = 50 Monte Carlo samples per datapoint, ensuring stable 638 Shapley estimation. The temperature parameter during softmax inference is set to T = 1.0 (default). 639 No additional prompt tuning or instruction tuning is applied unless otherwise noted. Models are 640 evaluated in a zero-shot setting without retrieval augmentation. 641 Table 4 summarizes the average NEAR scores for answer-containing and non-answer-containing 642 context sentences, along with the dominance ratio. Across all models and datasets, the dominance 643 ratio consistently exceeds 20, with most values ranging between 23 and 26. This indicates that the 644 information gain from answer-containing context sentences is significantly higher—by more than an 645 order of magnitude—than the entropy contributions of non-answer sentences. These results affirm 646 that NEAR provides a strong and reliable decomposition of usable information, even in the presence 647 of noise or hallucination-inducing segments.

Table 4: Robustness of NEAR attribution: Average NEAR scores for answer-containing vs nonanswer-containing sentences. Higher dominance ratios indicate stronger signal-to-noise separation.

Model Dataset Mean NEAR (Ans.) Std. Dev. Mean NEAR (Non-Ans.) Std. Dev. Dominance Ratio

LLaMA-3.1-8B CoQA 7.21 0.14 -0.31 0.06 23.26 LLaMA-3.1-8B QuAC 7.38 0.13 -0.30 0.05 24.60 LLaMA-3.1-8B SQuAD 7.50 0.16 -0.32 0.05 23.44 LLaMA-3.1-8B TriviaQA 7.65 0.15 -0.29 0.06 26.38 OPT-6.7B CoQA 7.02 0.17 -0.28 0.07 25.07

OPT-6.7B QuAC 7.20 0.18 -0.29 0.08 24.83

OPT-6.7B SQuAD 7.30 0.19 -0.30 0.09 24.33 OPT-6.7B TriviaQA 7.10 0.18 -0.27 0.08 26.30 Qwen2.5-3B CoQA 6.90 0.15 -0.33 0.07 20.91 Qwen2.5-3B QuAC 6.85 0.14 -0.31 0.08 22.10 Qwen2.5-3B SQuAD 6.95 0.16 -0.32 0.07 21.72 Qwen2.5-3B TriviaQA 6.88 0.13 -0.30 0.08 22.93

## 648 **A6 Generalization To Other Tasks**

649 While NEAR is primarily formulated for question answering (QA) tasks by computing entropy at 650 the final answer token, the framework naturally extends to other generation settings. For instance, in 651 **summarization**, information gain can be evaluated at the end of the summary sequence. In **dialog** 652 **systems**, NEAR can be applied at each utterance boundary to assess context contribution toward the 653 next response.

654 To illustrate this potential, we conduct a small pilot experiment on the XSum [42] summarization 655 dataset. We compute NEAR scores using entropy at the final token of generated summaries, following 656 the same context segmentation and Shapley attribution methodology. Preliminary results show that 657 answer-relevant document spans receive consistently higher NEAR scores, suggesting effective 658 context attribution in summarization as well.

Pilot NEAR Scores on XSum Example S1 S2 S3 S4 S5 S6 Context Sentences 0.0 0.2 0.4 0.6 0.8 1.0 Relevance threshold NEAR 
Score
659 This evidence indicates that NEAR may serve as a unified attribution framework across a variety of 660 text generation tasks. We leave a full empirical evaluation for future work.

## 661 **A6.1 Comparison With Llm-Check On Fava**

674 To enable direct comparison, we compute NEAR scores on the same FAVA-Annotation samples 675 used in LLM-Check and report AUROC, F1, and TPR@5%FPR. Across three LLMs (LLaMA-2-7B, 676 LLaMA-3-8B, OPT-6.7B), NEAR achieves competitive detection performance, with AUROC up to 677 **73.8**, F1 scores exceeding 70, and notable stability across layers.

## 678 **A7 Algorithm**

679 The algorithm of our methodology has been given here 1 662 To evaluate the effectiveness of NEAR in detecting hallucinations, we compare its performance 663 against **LLM-Check** [16], a recent method that leverages attention kernel eigenvalues and hidden 664 activations for hallucination detection across transformer layers. We focus on the zero-resource 665 setting without external references, using the human-annotated **FAVA dataset**[43]. 666 LLM-Check reports strong results using *Attention Scores* and *Hidden Scores*, computed from the 667 mean log-determinants of attention kernels and hidden state covariance matrices, respectively. On the 668 FAVA-Annotation split, their best-performing variant achieves an AUROC of 72.34 and F1 score of 669 69.27 using LLaMA-2 7B at layer 21 (see Table 2 in [16]). 670 In contrast, NEAR computes the entropy-based information gain attributed to each sentence in the 671 context, based on Shapley values over attention norms. Despite being conceptually different, LLM- 672 Check focuses on low-rank shifts in latent space, whereas NEAR tracks attention-driven entropy 673 reduction, both methods aim to isolate ungrounded model behavior.