011

014 015 016

018

024

026

034

036

038

# House of Cards: Massive Weights in LLMs

Anonymous Authors<sup>1</sup>

# Abstract

Massive activations, which manifest in specific feature dimensions of hidden states, introduce a significant bias in large language models (LLMs), leading to an overemphasis on the corresponding token. In this paper, we identify that massive activations originate not from the hidden state but from the intermediate state of a feed-forward network module in an early layer. Expanding on the previous observation that massive activations occur only in specific feature dimensions, we dive deep into the weights that cause massive activations. Specifically, we define *top-*k *massive weights* as the weights that contribute to the dimensions with the top-k magnitudes in the intermediate state. When these massive weights are set to zero, the functionality of LLMs is entirely disrupted. However, when all weights except for massive weights are set to zero, it results in a relatively minor performance drop, even though a much larger number of weights are set to zero. This implies that during the pretraining process, learning is dominantly focused on massive weights. Building on this observation, we propose a simple plug-and-play method called MacDrop (massive weights curriculum dropout), to rely less on massive weights during parameter-efficient fine-tuning. This method applies dropout to the pre-trained massive weights, starting with a high dropout probability and gradually decreasing it as fine-tuning progresses. Through various experiments, including zero-shot downstream tasks, long-context tasks, and ablation studies, we demonstrate that MacDrop generally improves performance and strengthens robustness.

# 1. Introduction

Large language models (LLMs), such as GPT [\(Achiam et al.,](#page-8-0) [2023\)](#page-8-0) and Llama [\(Touvron et al.,](#page-10-0) [2023;](#page-10-0) [Dubey et al.,](#page-8-1) [2024\)](#page-8-1), have achieved remarkable success across diverse natural language tasks [\(Roziere et al.,](#page-10-1) [2023;](#page-10-1) [Mitra et al.,](#page-9-0) [2024;](#page-9-0) [Labrak](#page-9-1) [et al.,](#page-9-1) [2024;](#page-9-1) [Wu et al.,](#page-10-2) [2023\)](#page-10-2). Their success is largely attributed to the pre-training phase, during which they are trained on extensive high-quality corpora datasets to predict the next token [\(Longpre et al.,](#page-9-2) [2024;](#page-9-2) [Zhao et al.,](#page-11-0) [2024;](#page-11-0) [Shen et al.,](#page-10-3) [2023\)](#page-10-3). However, despite the impressive achievements of LLMs, a crucial gap remains in our understanding of the underlying mechanisms that drive their remarkable performance.

Recently, [Xiao et al.](#page-10-4) [\(2024\)](#page-10-4) uncovered an intriguing phenomenon in LLMs, referred to as *attention sinks*: an unexpectedly large portion of attention is directed toward the initial tokens, regardless of their semantic context, after a small number of early layers. They demonstrated that under a restricted budget, focusing attention solely on recent window leads to poor performance, and that performance is recovered when initial tokens are included. Based on this observation, they proposed StreamingLLM, which retains the key-value caches of the initial sink tokens and the recent tokens for streaming use of LLMs. [Yu et al.](#page-10-5) [\(2024b\)](#page-10-5) further investigated the attention sinks phenomenon, finding that attention sinks can appear both in the initial tokens and in later tokens with less semantic importance (e.g., '.' and '\n'). They showed that when sink tokens appear later in a sequence, sink tokens can potentially result in performance degradation. Inspired by this observation, they proposed a head-wise attention calibration technique without requiring additional training. Concurrently, [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6) discovered the existence of *massive activations* in the hidden states of LLMs, with magnitudes substantially larger than the others. Massive activations are jointly identified based on their sequence and feature dimensions within the hidden states. Specifically, massive activations occur at the initial tokens and weak semantic tokens according to the model, and are consistently present in only a few fixed feature dimensions. Moreover, they connected massive activations with attention sinks, suggesting that massive activations inject implicit bias into the self-attention mechanism throughout the pretraining phase.

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

![](_page_1_Diagram_1.jpeg)

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

Figure 1. (a) Massive weights are defined as the rows of Wgate and Wup in a specific layer l using the *bos* token, which produce the top-k magnitudes of the intermediate state hˆinter . Because massive weights are defined within a single layer l, the ratio of massive weights is significantly low compared to the overall number of parameters. For instance, in the case of Llama-3-8B, the proportion of the top-5 massive weights is 0.0005% of the model's total parameters. (b) When the top-5 massive weights are zeroed out, instruction-tuned LLMs completely lose their ability to generate text. On the other hand, when only the top-5 massive weights remain unchanged in Wgate and Wup, instruction-tuned LLMs retain their generation capability.

In this paper, we first delve deeper into massive activations, providing two key observations. (1) The *bos* token placed at the starting position always has massive activations in the same feature dimensions and makes attention sinks. This observation enables us to focus only on the feature dimensions, instead of both the sequence and feature dimensions, when addressing massive activations. Namely, a simplified and consistent analysis of massive activations can be achieved by using only the *bos* token. (2) Massive activations originate in the intermediate state hˆinter <sup>l</sup> within an early layer l, before appearing in the hidden state h<sup>l</sup> , as illustrated in Figure [1\(](#page-1-0)a). Namely, massive activations triggered in hˆinter are subsequently and continuously propagated through skip connections. This observation implies that the feed-forward network in layer l plays a crucial role in LLMs.

Next, we shift our focus from activations to the weights, relying on the fact that massive activations consistently appear in the same feature dimensions. In detail, we define the *top-*k *massive weights* as the rows of Wup and Wgate in the feed-forward network at layer l that produce the top-k magnitudes of the intermediate state hˆinter l , as illustrated in Figure [1\(](#page-1-0)a). It is important to note that massive weights, defined within a single layer l, account for a substantially small fraction compared to the model's total parameters. This holds true even when compared to the entire Wup and Wgate. Nevertheless, massive weights are crucial factors that can completely influence the performance of LLMs. Figure [1\(](#page-1-0)b) presents the generated responses of three models to the given user prompt: original model, top-5 massive weights attacked model, and other weights attacked model. Here, other weights represent all weights in Wup and Wgate at layer l that do not belong to the top-5 massive weights,

and an attack sets corresponding weights to zero. When the massive weights are attacked, the model becomes poor and repeats the user prompt. On the contrary, when other weights are attacked, the model does not entirely lose its generation capability, even though a much greater number of weights are set to zero in the same projection matrices. These observations imply that massive weights are dominantly learned during pre-training and highly related to the performance of LLMs.

Finally, we propose a straightforward plug-and-play method during parameter-efficient fine-tuning, named massive weights curriculum dropout (MacDrop). This method applies dropout to the pre-trained massive weights, rather than additional trainable weights, starting with a high dropout rate that is progressively reduced throughout the fine-tuning phase. The intuition behind MacDrop is that a high initial dropout rate encourages the model to lessen dependence on the massive weights predominantly learned during the pretraining phase. Then, reducing the dropout rate facilitates a more stable convergence, ensuring the pre-trained model is leveraged with neglectable damage by the end of fine-tuning. Through various ablation studies, we examine the effects of dropout scope and dropout probability scheduling. Finally, we demonstrate that MacDrop generally enhances model performance and robustness in zero-shot downstream tasks and long context tasks.

# 2. Massive Weights

In this section, we review the key observations on massive activations reported by [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6) and extend the analysis by exploring various states using the *bos* token,

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

which was not covered thoroughly. Based on this expanded analysis, we formally define top-k massive weights in a specific layer and investigate their importance through two opposite types of attacks.

![](_page_2_Figure_2.jpeg)

#### 2.1. Prerequisite: Massive Activations

Autoregressive Transformers [\(Vaswani et al.,](#page-10-7) [2017\)](#page-10-7) are structured with L decoding layers. Each layer l ∈ [1, L] includes an attention (ATTN) module and a feed-forward network (FFN) module. These modules are connected via residual connections [\(He et al.,](#page-8-2) [2016\)](#page-8-2), each following a layer normalization (LN) layer [\(Ba,](#page-8-3) [2016\)](#page-8-3). The previous hidden state hl−<sup>1</sup> is fed into layer l and processed to produce the subsequent hidden state h<sup>l</sup> :

$$h_l = \hat{h}_l + \text{FFN}(\text{LN}(\hat{h}_l)), \quad (1)$$
where  $\hat{h}_l = h_{l-1} + \text{ATTN}(\text{LN}(h_{l-1}))$ 

[Sun et al.](#page-10-6) [\(2024a\)](#page-10-6) primarily concentrated on the activations within hidden states, identifying that certain activations exhibit exceptionally large magnitudes, which they termed *massive activations*. Massive activations are observed at the starting position (i.e., input-agnostic) or at the delimiter tokens, depending on the model. Furthermore, these activations are confined to a small number of fixed feature dimensions, even within these tokens. These activations initially emerge after passing through several early layers and then decreases as they near the last layer.

Massive activations are strongly tied to the attention sinks phenomenon, as identified by [Xiao et al.](#page-10-4) [\(2024\)](#page-10-4), in which attention is abnormally concentrated on a small subset of tokens. In detail, a given query state tends to have positive cosine similarity with the key states of the tokens exhibiting massive activations, and negative cosine similarity with those of other tokens. Consequently, attention is heavily skewed toward the tokens associated with massive activations. Detailed related work is explained in Appendix [H.](#page-26-0)

## 2.2. Further Analysis on Massive Acitvations

We primarily utilize the Llama-3-8B model [\(Dubey et al.,](#page-8-1) [2024\)](#page-8-1) and explicitly specify other models when necessary.

*bos* token placed at the starting position always has massive activations. We begin by examining whether any specific condition consistently triggers massive activations. The existence of such a condition would greatly facilitate the analysis and algorithm development for handling massive activations. Following [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6), massive activations are observed when any token is placed at the starting position; however, we find the cases where the token at the starting position does not trigger massive activations and attention sinks, such as Mistral-7B.

Figure 2. (Top) Magnitudes of the hidden state and (Bottom) attention scores after Softmax of Mistral-7B, according to the position of the *bos* token. The described hidden state is the output of layer 16 (i.e., h16). The attention scores are calculated at layer 17 (i.e., after massive activations appear) and averaged across different heads.

Figure [2](#page-2-0) describes the magnitudes of activations of the hidden state and normalized attention scores of Mistral-7B according to the position of the *bos* token, after massive activations appear. The reason for arbitrarily inserting the *bos* token is based on the previous observation that nonsemantic tokens can trigger massive activations in certain LLMs. We use the implementation[<sup>1</sup>](#page-2-1) of massive activations for this example and visualization. Mistral-7B has massive activations at the first delimiter token '.', not at the starting position (first column). However, when the *bos* token is placed at the starting position, it triggers massive activations and the first delimiter token loses its massive activations (second column). When the *bos* token is inserted in the middle or ending position after the first delimiter token, massive activations are observed in both tokens (third and fourth columns). On the other hand, there are models that respond to the starting position but not to the *bos* token, such as Llama-2, detailed in Appendix [C.](#page-13-0) Therefore, by considering both conditions, we use only the *bos* token placed at the starting position for the continuation of analysis and algorithm development.

Massive activations originate in the intermediate state of a FFN module. Next, we trace various states in early layers until the first massive activations appear, using the *bos* token. Specifically, we monitor hl−1, LN(hl−1), ATTN(LN(hl−1)), hˆ l , LN(hˆ <sup>l</sup>), and FFN(LN(hˆ <sup>l</sup>)) in Eq. [\(1\)](#page-2-2) throughout early layers. Figure [3\(](#page-3-0)a) illustrates the magnitudes of various states in layers l ∈ [1, 3]. It is observed that FFN(LN(hˆ <sup>2</sup>)) has massive activations before h2. With Figure [3\(](#page-3-0)b), which describes the top three and median magnitudes of the hidden state<sup>1</sup> , it is observed that the massive activations generated within a FFN module at layer 2 are transmitted directly to the next hidden state and then propagated solely through the residual connections.

<sup>1</sup> https://github.com/locuslab/massive-activations

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

Table 1. Perplexity and zero-shot downstream tasks performance according to the attack.

|                | Models |           | WikiText    | C4      | PG-19   | Avg. (  | ↓ ) ARC-E | ARC-C | BoolQ | PIQA | WG   | Avg. ( ↑ ) |
|----------------|--------|-----------|-------------|---------|---------|---------|-----------|-------|-------|------|------|------------|
| Llama-3-8B     |        |           | 5.75        | 9.94    | 8.98    | 8.22    | 77.4      | 52.7  | 81.4  | 80.8 | 72.9 | 73.0       |
| top-           | 5      | zeroing   | 104.57      | 132.83  | 130.83  | 122.74  | 29.1      | 22.0  | 41.8  | 53.8 | 50.3 | 39.4       |
| top-           | 5      | retaining | 10.15       | 23.98   | 28.72   | 20.95   | 75.0      | 48.0  | 80.2  | 77.7 | 74.1 | 71.0       |
| Llama-3-70B    |        |           | 2.68        | 7.59    | 6.02    | 5.43    | 85.8      | 64.2  | 85.3  | 84.6 | 80.5 | 80.1       |
| top-           | 5      | zeroing   | 11135.81    | 7288.86 | 4696.49 | 7707.05 | 28.6      | 22.8  | 38.0  | 55.5 | 49.5 | 38.9       |
| top-           | 5      | retaining | 3.47        | 9.93    | 7.26    | 6.89    | 45.9      | 25.6  | 83.9  | 64.6 | 77.0 | 59.4       |
| Llama-3.1-405B |        |           | (8bit) 1.41 | 6.20    | 3.23    | 3.61    | 86.3      | 66.0  | 88.2  | 85.0 | 81.1 | 81.3       |
| top-           | 5      | zeroing   | 1785.56     | 985.36  | 633.40  | 1134.77 | 26.6      | 25.9  | 37.8  | 49.7 | 50.3 | 38.1       |
| top-           | 5      | retaining | 2.47        | 9.36    | 5.61    | 5.81    | 83.0      | 62.5  | 83.1  | 83.2 | 70.3 | 76.4       |

![](_page_3_Figure_3.jpeg)

Figure 3. (a) Magnitudes of various states and (b and c) the top three and median magnitudes of hidden states and intermediate states across layers. These results show that massive activations in the hidden state originate from those in the intermediate state in a FFN module in an early layer.

Furthermore, we decompose a FFN module into Wdown(σ(Wgate(·)) ⊙ Wup(·)), to analyze the intermediate states (i.e., the output of σ(Wgate(·)) ⊙Wup(·)). Figure [3\(](#page-3-0)c) describes the top three and median magnitudes of the intermediate state across layers. It is demonstrated that *massive activations originate in the intermediate state of a FFN module in an early layer* l. This result implies that Wup and Wgate in layer l are closely tied to massive activations. Additional results for other LLMs are provided in the Appendix [D.](#page-19-0)

## 2.3. Massive Weights

Massive weights are defined based on massive activations in the intermediate state at layer l, denoted as hˆinter l , when the *bos* token is fed into LLMs. To elaborate, we define the rows in the projection matrix Wup (and Wgate, if it exists) that correspond to the indices of the top-k magnitudes in hˆinter l as *top-*k *massive weights*, depicted in Figure [1\(](#page-1-0)a). It is noted that massive weights are defined within one specific layer, which means the number of massive weights is significantly smaller compared to the total number of parameters in LLMs. For example, in Llama-3-8B, the number of top-k massive weights is calculated as 2 × k × 4096, where 4096

represents the dimensions of hidden state. If k is set to 5, massive weights account for approximately 0.0005% of the total parameters in Llama-3-8B, approximately 0.0001% in Llama-3-70B, and approximately 0.00004% in Llama-3.1- 405B.

Massive weights are extremely small in quantity, their impact is tremendous. To assess the significance of massive weights, we conduct two types of attacks: top-k zeroing and top-k retaining. Note that these attacks only affect the Wup and Wgate projection matrices in layer l, where massive weights are present. The first attack is to set the top-k massive weights to zero (i.e., darker orange weights in Figure [1\(](#page-1-0)a)). In essence, this attack is very similar to the one proposed in [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6), where massive activations in the hidden state are zeroed out in a single layer. The difference is that their attack targets the hidden state, while our attack targets the intermediate state. The second attack is to set all weights to zero except for top-k massive weights (i.e., lighter orange weights in Figure [1\(](#page-1-0)a)). That is, the number of rows being damaged in each attack is k and the dimensions of intermediate state − k, respectively.

Following [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6), we assess perplexity<sup>1</sup> on three datasets: WikiText [\(Merity et al.,](#page-9-3) [2017\)](#page-9-3), C4 [\(Raffel et al.,](#page-9-4) [2020\)](#page-9-4), and PG-19 [\(Rae et al.,](#page-9-5) [2020\)](#page-9-5). Additionally, we evaluate zero-shot accuracy[<sup>2</sup>](#page-3-1) on five tasks: Arc-Easy (ARC-E), Arc-Challenge (ARC-C) [\(Clark et al.,](#page-8-4) [2018\)](#page-8-4), BoolQ [\(Clark](#page-8-5) [et al.,](#page-8-5) [2019\)](#page-8-5), PIQA [\(Bisk et al.,](#page-8-6) [2020\)](#page-8-6), and WinoGrande (WG) [\(Sakaguchi et al.,](#page-10-8) [2021\)](#page-10-8). Table [1](#page-3-2) presents the results of two attacks on Llama-3-8B, Llama-3-70B, and Llama-3.1-405B (8bit)[<sup>3</sup>](#page-3-3) , when k is set to 5. Llama-3-8B has its massive weights in layer 2 out of 32 layers, Llama-3-70B in layer 4 out of 80 layers, and Llama-3.1-405B in layer 6 out of 126 layers. Detailed indices of massive weights are provided in Appendix [B,](#page-12-0) including various LLMs. The larger the difference from the original performance, the stronger the attack.

https://github.com/EleutherAI/lm-evaluation-harness

<sup>3</sup>We use 8xA100-80G GPUs for our work; therefore, we employ an 8-bit model.

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

![](_page_4_Figure_1.jpeg)

Figure 5. Probability of experts of Mixtral-8x7b for the *bos* token. In the layer with massive weights (i.e., layer 2), the router probability between experts becomes completely skewed to one expert.

Top-5 zeroing is a much stronger attack than top-5 retaining, even though top-5 retaining sets several thousands times more weights to zero compared to top-5 zeroing does for the same projection matrices. This means that in the projections Wup and Wgate at layer l, having only massive weights is significantly better than having all weights except for massive weights. In detail, similar to the findings of [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6), the top-k zeroing attack proves to be highly effective in disrupting the Llama-3 family, even for extremely large-scale models (e.g., 70B and 405B). On the other hand, the top-k retaining attack does not cause complete damage. In conclusion, these findings reveal that massive weights are predominantly learned during pre-training, highlighting their essential contribution to the performance of LLMs.

Moreover, massive weights also exist in instruction-tuned LLMs such as Llama-3-8B-Instruct. These attacks are effective, as depicted in Figure [1\(](#page-1-0)b). When massive weights are set to zero (i.e., darker orange box), the model repeats the same text as the user prompt. On the other hand, when all weights are set to zero except for massive weights (i.e., lighter orange box), the model retains its ability to generate text, although the generated text differs from the original.

The value of k, which affects performance, varies depending on the model architecture. We examine the robustness of various LLMs against the top-k zeroing and retaining attacks, with a focus on the impact of the parameter k. Llama-2 [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0), Llama-3 [\(Dubey et al.,](#page-8-1) [2024\)](#page-8-1), Mistral [\(Jiang et al.,](#page-9-6) [2023\)](#page-9-6) and Mixtral [\(Jiang et al.,](#page-9-7) [2024\)](#page-9-7), Phi-3 [\(Abdin et al.,](#page-8-7) [2024\)](#page-8-7), and Gemma-2 [\(Team](#page-10-9) [et al.,](#page-10-9) [2024\)](#page-10-9) families are used. Figure [4](#page-4-0) illustrates the mean zero-shot accuracy of LLMs under the two attacks, according to the k. In top-k zeroing, more weights are set to zero as k increases, whereas in top-k retaining, more weights are

set to zero as k decreases. When k is 0 in top-k zeroing, it corresponds to the original performance without any attack, whereas, when k is 0 in top-k retaining, it sets the entire weights of Wup and Wgate in layer l to zero.

The Llama families are highly sensitive to massive weights. In the top-k zeroing, a noticeable performance drop occurs even when k is as small as 3, irrespective of the model's scale. In top-k retaining, when k is set to 1 (i.e., with only one row active in Wup and Wgate in layer l), the performance nearly reaches the original level in smaller-scaled models (≤ 13B). While, for larger-scaled models (≥ 70B), the top-30 massive weights are required to maintain performance. Moreover, since different versions of Llama families pre-trained on varying datasets (e.g., with different cutoff dates) exhibit similar tendencies, it can be inferred that the pre-training dataset itself does not significantly influence the emergence of massive phenomena.

Similarly, Mistral is also significantly disrupted, when k is set to 5. Mixtral is a sparse Mixture of Experts (MoE) architecture that uses a top-2 routing mechanism, where two experts are activated among eight FFN modules in each layer. To attack the Mixtral model, we identify the active experts in the layer with massive weights using the *bos* token. Figure [5](#page-4-1) describes the probability distribution of experts in the Mixtral model across all layers. Notably, it is observed that when massive activations occur, a single expert (i.e., expert 4) is assigned a significantly higher probability than the others. Therefore, we target only the Wup and Wgate of this expert, rather than all experts. Although Mixtral does not completely break down, there is a considerable decline in performance when the top-50 massive weights are zeroed out. These results indicate that, despite the immense resources required to build high-performance LLMs, they can collapse like a house of cards even under minimal attacks.

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

The Phi-3 family exhibits different robustness against attacks depending on the model size. As noted by [Abdin et al.](#page-8-7) [\(2024\)](#page-8-7), the phi-3-mini (3.8B) is trained on 3.3T tokens, while the phi-3-medium (14B) is trained on 4.8T tokens. A key architectural difference from the Llama family is the use of dropout to the outputs of both the ATTN and FFN modules, formed by Eq. [\(2\)](#page-5-0). While a specific recipe for dropout is not provided in the technical report [\(Abdin et al.,](#page-8-7) [2024\)](#page-8-7), in the case of phi-3-medium, applying dropout with longer pre-training might ensure that the residual connections contribute meaningfully, mitigating the risk of excessive dependence on massive weights.

$$h_l = \hat{h}_l + \text{dropout}(\text{FFN}(\text{LN}(\hat{h}_l))), \quad (2)$$
where  $\hat{h}_l = h_{l-1} + \text{dropout}(\text{ATTN}(\text{LN}(h_{l-1})))$ 

The Gemma-2 family is exceptionally resilient to the top-k zeroing attack, maintaining almost no loss in performance even when k is large. Additionally, even if Wup and Wgate are entirely eliminated (i.e., k = 0 in top-k retaining), there is no noticeable performance degradation. This family incorporates two additional LN layers after both the ATTN and FFN modules, formed by Eq. [\(3\)](#page-5-1). These added normalization layers result in completely different hidden and intermediate states compared to other LLMs, as described in Appendix [D.](#page-19-0) Furthermore, attention sinks are not observed in the Gemma-2 family, as shown in Appendix [E.](#page-23-0)

$$h_l = \hat{h}_l + \text{LN}(\text{FFN}(\text{LN}(\hat{h}_l))), \quad (3)$$
where  $\hat{h}_l = h_{l-1} + \text{LN}(\text{ATTN}(\text{LN}(h_{l-1})))$ 

These observations imply that LLMs not sensitive to massive phenomena may not perform effectively with algorithms based on massive phenomena. Further discussion is provided in Appendix [F.](#page-24-0)

# 3. Massive Weights Curriculum Dropout

In this section, we propose a straightforward plug-andplay method, termed massive weights curriculum dropout (MacDrop), during parameter-efficient fine-tuning such as low rank adaptation [\(Hu et al.,](#page-9-8) [2022\)](#page-9-8). This method applies dropout to the pre-trained massive weights with a curriculum that gradually reduces the dropout probability. The reason for applying dropout to weights [\(Wan et al.,](#page-10-10) [2013\)](#page-10-10) instead of activations [\(Srivastava et al.,](#page-10-11) [2014\)](#page-10-11) is that the number of massive activations is only k, but that of massive weights is k × d, where d is the dimension of the hidden states. It is important to note that our method does not modify the trainable parameters of adapters; instead, it is applied to the pre-trained frozen weights. Therefore, MacDrop can be applied orthogonally to the process of training the adapter.

Algorithm 1: Top-k massive weights curriculum dropout (MacDrop) in pseudo PyTorch style

// Dropout is only executed in layer l,

where h

inter

<sup>l</sup> appears.

Input: k, massive intermediate state h

inter <sup>l</sup> of the *bos*

token, initial dropout probability p0, total steps T

// massive activations in the intermediate state <sup>1</sup> , sorted indices = torch.sort(torch.abs(h

inter ),

descending=True)

<sup>2</sup> massive indices = sorted indices[:k]

// pre-trained massive weights <sup>3</sup> massive W up = copy(W up[massive indices, :]) <sup>4</sup> massive W gate = copy(W gate[massive indices, :])

<sup>5</sup> for t = 1 *to* T do

// decreasing dropout probability

<sup>6</sup> p = p<sup>0</sup> × (1 −

t T )

// pre-trained massive weights

dropout

<sup>7</sup> mask = (torch.rand(massive W up.shape) > p).int()

<sup>8</sup> W up[massive indices, :] \*= mask <sup>9</sup> W gate[massive indices, :] \*= mask <sup>10</sup> tr loss step = training step(model, inputs)

// pre-trained massive weights

rollback

<sup>11</sup> W up[massive indices, :] = massive W up <sup>12</sup> W gate[massive indices, :] = massive W gate

MacDrop is motivated by the observation that massive weights are predominantly learned during pre-training, and that zeroing them can severely undermine LLMs. Therefore, at the early stages of fine-tuning, the objective is to reduce the reliance on massive weights, as their excessive dominance may lead the model to over-rely on specific patterns. Moreover, considering that the undamaged pretrained model is used after fine-tuning is finished, we develop a strategy to adjust the dropout rate using a curriculum.

Algorithm [1](#page-5-2) describes MacDrop in a pseudo PyTorch [\(Paszke et al.,](#page-9-9) [2019\)](#page-9-9) style, and is implemented within the trainer code of transformers[<sup>4</sup>](#page-5-3) . Initially, massive weights are identified using the *bos* token before fine-tuning (Lines 1-4). Subsequently, an adapter is trained while the pre-trained massive weights are dropped. Meanwhile, a curriculum strategy is applied to progressively enable the use of the original pre-trained weights without masking. Note that in Line 10, 'model' includes both the masked pre-trained network and a trainable adapter. When we implement Lines 1-4 in practice, we precomputed the massive indices and loaded them. Lines 6-9 and 11-12 are the additional computation for MacDrop, which requires neglectable overhead (e.g., approximately 0.35 second per step for Llama-3-8B using LoRA on 8xA100 GPUs).

<sup>4</sup> https://github.com/huggingface/transformers/tree/main/src/ transformers

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

Table 2. Long context tasks performance.

| Method    | NrtvQA | Single-document Qasper | MF-en | QA HotpotQA | Multi-document 2WikiMQA | QA Musique | GovReport Model: | Summarization QMSum | MultiNews Meta-Llama-3-8B | TREC  | Few-shot TriviaQA | learning SAMSum | PCount | Synthetic PRe | Lcc   | Code RB-P | Avg.  |
|-----------|--------|------------------------|-------|-------------|-------------------------|------------|------------------|---------------------|---------------------------|-------|-------------------|-----------------|--------|---------------|-------|-----------|-------|
| LoRA      | 26.03  | 30.38                  | 53.38 | 26.30       | 23.05                   | 11.96      | 29.00            | 22.81               | 26.43                     | 72.50 | 81.14             | 44.27           | 2.63   | 32.00         | 72.90 | 69.70     | 39.03 |
| + MacDrop | 25.31  | 34.05                  | 46.84 | 38.06       | 28.99                   | 17.92      | 29.62            | 22.86               | 26.64                     | 72.00 | 89.34             | 45.08           | 3.00   | 27.50         | 73.17 | 68.25     | 40.54 |
| DoRA      | 26.31  | 31.57                  | 52.10 | 27.04       | 23.57                   | 12.01      | 29.20            | 23.35               | 26.34                     | 73.50 | 81.35             | 43.22           | 2.61   | 30.00         | 73.46 | 69.44     | 39.07 |
| + MacDrop | 26.11  | 30.99                  | 53.37 | 29.35       | 25.66                   | 12.14      | 29.07            | 23.10               | 26.32                     | 73.50 | 86.63             | 44.64           | 2.05   | 25.50         | 73.90 | 69.32     | 39.48 |
|           |        |                        |       |             |                         |            | Model:           | Mistral-7B-v0.1     |                           |       |                   |                 |        |               |       |           |       |
| LoRA      | 22.52  | 34.64                  | 35.65 | 32.11       | 19.80                   | 12.73      | 27.22            | 21.98               | 26.73                     | 69.00 | 87.50             | 41.70           | 1.00   | 21.00         | 71.57 | 65.47     | 36.91 |
| + MacDrop | 23.49  | 38.51                  | 36.11 | 37.78       | 27.60                   | 14.91      | 26.40            | 22.53               | 26.92                     | 69.50 | 89.92             | 37.07           | 1.55   | 20.00         | 70.37 | 65.87     | 38.03 |
| DoRA      | 22.79  | 34.52                  | 35.55 | 30.87       | 17.84                   | 12.26      | 27.45            | 22.15               | 26.52                     | 70.00 | 88.05             | 41.56           | 1.00   | 20.50         | 71.88 | 65.38     | 36.77 |
| + MacDrop | 23.10  | 35.10                  | 35.00 | 29.53       | 23.77                   | 10.50      | 27.14            | 22.63               | 27.50                     | 69.00 | 89.56             | 38.96           | 1.00   | 21.42         | 71.17 | 65.53     | 36.93 |

Table 3. Zero-shot downstream tasks performance.

| Model Method         | ARC-E | ARC-C | BoolQ | PIQA | WG   | Avg. |
|----------------------|-------|-------|-------|------|------|------|
| LoRA                 | 79.6  | 58.2  | 83.9  | 82.4 | 75.9 | 76.0 |
| + MacDrop Llama-3-8B | 82.9  | 58.3  | 83.9  | 82.6 | 75.0 | 76.5 |
| DoRA                 | 80.8  | 57.7  | 83.9  | 82.5 | 75.8 | 76.1 |
| + MacDrop            | 81.9  | 58.2  | 83.9  | 82.2 | 75.6 | 76.4 |
| LoRA                 | 78.5  | 54.9  | 84.9  | 82.9 | 75.3 | 75.3 |
| + MacDrop Mistral-7B | 80.9  | 56.7  | 85.0  | 83.0 | 75.3 | 76.2 |
| DoRA                 | 78.4  | 55.1  | 85.0  | 82.9 | 75.1 | 75.3 |
| + MacDrop            | 80.6  | 56.7  | 85.3  | 82.9 | 75.1 | 76.1 |

# 4. Experiments

## 4.1. Zero-shot Downstream Task

We fine-tune the Llama-3-8B and Mistral-7B using the alpaca gpt4 en dataset [\(Peng et al.,](#page-9-10) [2023\)](#page-9-10) for 3 epochs (579 steps), and evaluate on five zero-shot tasks. We use two parameter-efficient fine-tuning (PEFT) methods, LoRA [\(Hu](#page-9-8) [et al.,](#page-9-8) [2022\)](#page-9-8) and DoRA [\(yang Liu et al.,](#page-10-12) [2024\)](#page-10-12). DoRA decomposes the pre-trained weights into two components, magnitude and direction, and applies LoRA to the direction component. Our method is based on the implementation of Llama-Factory [\(Zheng et al.,](#page-11-1) [2024\)](#page-11-1) [5](#page-6-0) . For MacDrop, k and p<sup>0</sup> are set to 5 and 0.8, respectively. Details for implementations are explained in Appendix [A.](#page-12-1)

Table [3](#page-6-1) presents the results on zero-shot downstream tasks. For both the models and methods, MacDrop consistently leads to performance gains, especially in ARC-Easy and ARC-Challenge tasks. Note that we do not claim MacDrop is superior to LoRA/DoRA; rather, we demonstrate that LoRA/DoRA performs better when MacDrop is applied than when it is not. Results for other LLMs are provided in Appendix [F.](#page-24-0)

Table 4. Zero-shot downstream tasks performance under the top-3 zeroing attack, related to robustness.

| Model Method         | ARC-E | ARC-C | BoolQ | PIQA | WG   | Avg. |
|----------------------|-------|-------|-------|------|------|------|
| LoRA                 | 29.9  | 22.9  | 45.7  | 52.9 | 50.7 | 40.4 |
| + MacDrop Llama-3-8B | 36.8  | 24.7  | 64.3  | 58.5 | 54.1 | 47.7 |
| DoRA                 | 29.8  | 23.0  | 46.0  | 52.6 | 50.0 | 40.3 |
| + MacDrop            | 78.7  | 53.7  | 79.9  | 79.0 | 72.4 | 72.7 |
| LoRA                 | 54.6  | 34.8  | 58.0  | 74.8 | 57.5 | 55.9 |
| + MacDrop Mistral-7B | 69.2  | 45.0  | 78.3  | 79.3 | 64.1 | 67.2 |
| DoRA                 | 55.7  | 35.3  | 58.6  | 74.4 | 58.2 | 56.4 |
| + MacDrop            | 78.1  | 52.4  | 84.1  | 82.3 | 69.5 | 73.3 |

MacDrop is designed to mitigate dependence on massive weights during PEFT. To assess whether this dependency is effectively reduced, we attack the previous fine-tuned models in Table [3.](#page-6-1) Table [4](#page-6-2) shows the performance changes whether applying MacDrop under the top-3 zeroing attack. The zeroing attack severely degrades the performance of fine-tuned models without MacDrop, as seen in the case of Llama-3-8B with DoRA, where performance drops from 76.1 to 40.3. However, fine-tuned models with MacDrop exhibit significantly better performance under attack, indicating better robustness. Especially, when MacDrop is combined with DoRA, it demonstrates remarkable robustness. For instance, in the case of Llama-3-8B with DoRA, where performance drops from 76.4 to 72.7.

## 4.2. Long Context Task

We evaluate the two LLMs on LongBench [\(Bai et al.,](#page-8-8) [2024\)](#page-8-8), a benchmark specifically designed to assess the ability to understand long contexts. This includes 5 sub-categories and 16 English datasets: single-document QA, multi-document QA, summarization, few-shot learning, synthetic, and code generation. We set the max length of models to 7,500. Table [2](#page-6-3) shows that MacDrop increases the performance when understanding long context is required.

<sup>5</sup> https://github.com/hiyouga/LLaMA-Factory

394

396

![](_page_7_Figure_1.jpeg)

![](_page_7_Figure_2.jpeg)

Figure 6. Mean zero-shot accuracy according to the dropout scope and probability p.

#### 4.3. Ablation Study

We further provide ablation studies related to MacDrop using Llama-3-8B. The variations used in this section can serve as a baseline.

#### 4.3.1. DROPOUT SCOPE AND PROBABILITY

We investigate the effect of dropout scope and probability compared to the original performance achieved through LoRA without dropout. This ablation study is also conducted on the Wup and Wgate projection matrices in layer l. The dropout scope is divided into three categories: all weights, massive weights, all weights except for massive weights. Additionally, to assess the impact of dropout probability, it is kept constant throughout the fine-tuning process, without using a curriculum.

Figure [6](#page-7-0) illustrates the mean zero-shot accuracy according to the dropout scope and dropout probability p. It is observed that among three scopes, the original performance (i.e., without dropout), represented by the dotted line at 76.0, can be surpassed only when dropout is applied solely to massive weights. Nevertheless, if strong dropout (e.g., p ≥ 0.85) is maintained on the pre-trained massive weights during fine-tuning, performance deteriorates. This highlights the need to safeguard the pre-trained massive weights during the final stages of fine-tuning, because we are ultimately using them without causing any damage.

#### 4.3.2. CURRICULUM METHODS AND INITIAL DROPOUT PROBABILITY

We investigate the effect of curriculum methods and initial dropout probability p<sup>0</sup> in MacDrop, when LoRA is applied. We compare four curriculum methods: step-wise linear (Step), before epoch-wise linear (Epoch(before)), after epoch-wise linear (Epoch(after)), and exponential (Exp.). In formula, Step is defined as p = p<sup>0</sup> × (1 − tstep Tstep ). Epoch(before) and Epoch(after) are defined as p = p0×(1− tepoch−1 Tepoch ) and p = p0×(1− tepoch Tepoch ), respectively. Exp. is defined as p = p0×exp(−αtstep). Figure [7](#page-7-1) describes dropout probability p according to curriculum methods, when p<sup>0</sup> is 1.0. The distinct difference between Epoch(before) and Epoch(after) is that at the final epoch, the former continues

Figure 7. Curriculum methods.

Table 5. Mean zero-shot accuracy according to curriculum methods and initial dropout probability.

| Curriculum        |      | p    | 0    |      |
|-------------------|------|------|------|------|
| Method            | 0 2  | 0 5  | 0 8  | 1 0  |
| Step              | 76.0 | 76.3 | 76.5 | 75.5 |
| Epoch(before)     | 76.1 | 76.1 | 76.1 | 75.5 |
| Epoch(after)      | 75.9 | 76.0 | 76.2 | 76.3 |
| Exp. ( α = 0 01 ) | 76.0 | 76.2 | 76.5 | 76.4 |
| Exp. ( α = 0 05 ) | 76.0 | 76.1 | 76.2 | 76.3 |
| Exp. ( α = 0 10 ) | 76.0 | 76.1 | 76.1 | 76.2 |
| Mean              | 76.0 | 76.1 | 76.3 | 76.0 |

to apply dropout to the pre-trained massive weights with a probability of p<sup>0</sup> × 1 Tepoch , while the latter fully utilizes the pre-trained massive weights.

Table [5](#page-7-2) presents mean zero-shot accuracy according to curriculum methods and initial dropout probability p0. It is observed that step-based curriculum methods (e.g., Step and Exp.) generally achieve greater performance improvements compared to epoch-based curriculum methods (e.g., Epoch(before) and Epoch(after)). Nevertheless, when the initial dropout probability is relatively low (e.g., p<sup>0</sup> ≤ 0.2), even step-based curriculum methods fail to bring any performance gain compared to the original performance of 76.0. Additionally, it is shown that using a smaller α in the Exp. method leads to greater performance improvements, suggesting that a rapid decline in dropout probability to zero can diminish the effectiveness of MacDrop. On the other hand, for the Step and Epoch(before) methods, a significant performance drop is observed at a p<sup>0</sup> value of 1.0, highlighting the necessity of a near-zero dropout probability for the end of fine-tuning. In conclusion, for MacDrop, we recommend using the Step or Exp. with a smaller α, initiated from a moderately high p0.

## 5. Conclusion

In this paper, we explore the weight space of LLMs and identify the presence of massive weights within a FFN module in an early layer, which are predominantly pre-trained and have a significant impact on the performance of LLMs. Based on our observation, we propose a plug-and-play finetuning method called MacDrop, which applies dropout to the pre-trained massive weights, rather than to the parameters of adapters, during parameter-efficient fine-tuning. We hope that our findings will inspire future research in weight space of LLMs.

## Impact Statement

- This paper seeks to contribute to the advancement of large language models, particularly by providing deeper insights into their internal mechanisms. Our research demonstrates, through various open-sourced LLMs, that in many cases, models can be easily disrupted by zeroing massive weights if access to their weights is available. Subsequently, we improve robustness against such attacks through MacDrop. The existence of massive weights does not inherently present a risk if the model weights remain undisclosed. However, we believe that model developers should understand these internal mechanisms and take responsibility for ensuring robustness. In this vein, our work has the potential to make a meaningful impact. References Abdin, M., Jacobs, S. A., Awan, A. A., Aneja, J., Awadallah, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H., et al. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv preprint arXiv:2404.14219*, 2024. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. An, Y., Zhao, X., Yu, T., Tang, M., and Wang, J. Fluctuationbased adaptive structured pruning for large language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pp. 10865–10873, 2024. Ba, J. L. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016. Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li,
- J. LongBench: A bilingual, multitask benchmark for long context understanding. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3119–3137, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.172. URL [https:](https://aclanthology.org/2024.acl-long.172) [//aclanthology.org/2024.acl-long.172](https://aclanthology.org/2024.acl-long.172). Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. *arXiv:2004.05150*, 2020. Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020. Bojar, O., Buck, C., Federmann, C., Haddow, B., Koehn, P., Leveling, J., Monz, C., Pecina, P., Post, M., Saint-Amand, H., Soricut, R., Specia, L., and Tamchyna, A. Findings of the 2014 workshop on statistical machine translation. In Bojar, O., Buck, C., Federmann, C., Haddow, B., Koehn, P., Monz, C., Post, M., and Specia, L. (eds.), *Proceedings of the Ninth Workshop on Statistical Machine Translation*, pp. 12–58, Baltimore, Maryland, USA, June 2014. Association for Computational Linguistics. doi: 10.3115/v1/W14-3302. URL <https://aclanthology.org/W14-3302>. Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Burstein, J., Doran, C., and Solorio, T. (eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/ N19-1300. URL [https://aclanthology.org/](https://aclanthology.org/N19-1300) [N19-1300](https://aclanthology.org/N19-1300). Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021. Darcet, T., Oquab, M., Mairal, J., and Bojanowski, P. Vision transformers need registers. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=2dnO3LLiJ1) [id=2dnO3LLiJ1](https://openreview.net/forum?id=2dnO3LLiJ1). Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024. Ge, S., Zhang, Y., Liu, L., Zhang, M., Han, J., and Gao, J. Model tells you what to discard: Adaptive KV cache compression for LLMs. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https:](https://openreview.net/forum?id=uNrFpDPMyo) [//openreview.net/forum?id=uNrFpDPMyo](https://openreview.net/forum?id=uNrFpDPMyo). He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney,
  - M. W., Shao, Y. S., Keutzer, K., and Gholami, A. Kvquant: Towards 10 million context length llm inference with kv cache quantization. *arXiv preprint arXiv:2401.18079*, 2024. Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2022. URL [https:](https://openreview.net/forum?id=nZeVKeeFYf9) [//openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9). Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. *arXiv preprint arXiv:2310.06825*, 2023. Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna,
  - E. B., Bressand, F., et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024. Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., Neubig, G., Lee, M., Lee, K., and Seo, M. Prometheus 2: An open source language model specialized in evaluating other language models. *arXiv preprint arXiv:2405.01535*, 2024. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and Petrov, S. Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:452–466, 2019. doi: 10. 1162/tacl a 00276. URL [https://aclanthology.](https://aclanthology.org/Q19-1026) [org/Q19-1026](https://aclanthology.org/Q19-1026). Labrak, Y., Bazoge, A., Morin, E., Gourraud, P.-A., Rouvier, M., and Dufour, R. BioMistral: A collection of opensource pretrained large language models for medical domains. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Findings of the Association for Computational Linguistics ACL 2024*, pp. 5848–5864, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL [https://aclanthology.org/](https://aclanthology.org/2024.findings-acl.348) [2024.findings-acl.348](https://aclanthology.org/2024.findings-acl.348). Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm knows what you are looking for before generation. *arXiv preprint arXiv:2404.14469*, 2024. Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., and Han, S. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. *Proceedings of Machine Learning and Systems*, 6:87–100, 2024. Longpre, S., Yauney, G., Reif, E., Lee, K., Roberts, A., Zoph, B., Zhou, D., Wei, J., Robinson, K., Mimno, D., and Ippolito, D. A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. In Duh, K., Gomez, H., and Bethard, S. (eds.), *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 3245–3276, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.179. URL [https://](https://aclanthology.org/2024.naacl-long.179) [aclanthology.org/2024.naacl-long.179](https://aclanthology.org/2024.naacl-long.179). Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In *International Conference on Learning Representations*, 2017. URL [https://](https://openreview.net/forum?id=Byj72udxe) [openreview.net/forum?id=Byj72udxe](https://openreview.net/forum?id=Byj72udxe). Mitra, A., Khanpour, H., Rosset, C., and Awadallah, A. Orca-math: Unlocking the potential of slms in grade school math. *arXiv preprint arXiv:2402.14830*, 2024. Nallapati, R., Zhou, B., dos Santos, C., Gu.lc¸ehre, C¸ ., and Xiang, B. Abstractive text summarization using sequenceto-sequence RNNs and beyond. In Riezler, S. and Goldberg, Y. (eds.), *Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning*, pp. 280–290, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/K16-1028. URL <https://aclanthology.org/K16-1028>. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alche-Buc, F., Fox, E., ´ and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf) [cc/paper\\_files/paper/2019/file/](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf) [bdbca288fee7f92f2bfa9f7012727740-Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf). Peng, B., Li, C., He, P., Galley, M., and Gao, J. Instruction tuning with gpt-4. *arXiv preprint arXiv:2304.03277*, 2023. Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., and Lillicrap, T. P. Compressive transformers for longrange sequence modelling. In *International Conference on Learning Representations*, 2020. URL [https://](https://openreview.net/forum?id=SylKikSYDH) [openreview.net/forum?id=SylKikSYDH](https://openreview.net/forum?id=SylKikSYDH). Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring

554

556

558

560

564

566

568

571

574

576

578

594

596

598

- the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21 (140):1–67, 2020. Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Sauvestre, R., Remez, T., et al. Code llama: Open foundation models for code. *arXiv preprint arXiv:2308.12950*, 2023. Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021. Shen, Z., Tao, T., Ma, L., Neiswanger, W., Liu, Z., Wang, H., Tan, B., Hestness, J., Vassilieva, N., Soboleva, D., et al. Slimpajama-dc: Understanding data combinations for llm training. *arXiv preprint arXiv:2309.10818*, 2023. Son, S., Park, W., Han, W., Kim, K., and Lee, J. Prefixing attention sinks can mitigate activation outliers for large language model quantization. *arXiv preprint arXiv:2406.12016*, 2024. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1):1929–1958, 2014. Sun, M., Chen, X., Kolter, J. Z., and Liu, Z. Massive activations in large language models. In *First Conference on Language Modeling*, 2024a. URL [https:](https://openreview.net/forum?id=F7aAhfitX6) [//openreview.net/forum?id=F7aAhfitX6](https://openreview.net/forum?id=F7aAhfitX6). Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. In *The Twelfth International Conference on Learning Representations*, 2024b. URL [https://openreview.](https://openreview.net/forum?id=PxoFut3dWW) [net/forum?id=PxoFut3dWW](https://openreview.net/forum?id=PxoFut3dWW). Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Rame, A., et al. Gemma 2: Improving open ´ language models at a practical size. *arXiv preprint arXiv:2408.00118*, 2024. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. *arXiv preprint arXiv:2307.09288*, 2023. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [cc/paper\\_files/paper/2017/file/](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [3f5ee243547dee91fbd053c1c4a845aa-Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Wan, L., Zeiler, M., Zhang, S., Le Cun, Y., and Fergus, R. Regularization of neural networks using dropconnect. In *International conference on machine learning*, pp. 1058– 1066. PMLR, 2013. Wang, Z., Jin, B., Yu, Z., and Zhang, M. Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks. *arXiv preprint arXiv:2407.08454*, 2024. Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., Kambadur, P., Rosenberg, D., and Mann,
  - G. Bloomberggpt: A large language model for finance. *arXiv preprint arXiv:2303.17564*, 2023. Xia, H., Yang, Z., Dong, Q., Wang, P., Li, Y., Ge, T., Liu, T., Li, W., and Sui, Z. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Findings of the Association for Computational Linguistics ACL 2024*, pp. 7655–7671, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL [https://aclanthology.org/2024.](https://aclanthology.org/2024.findings-acl.456) [findings-acl.456](https://aclanthology.org/2024.findings-acl.456). Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.](https://openreview.net/forum?id=NG7sS51zVF) [net/forum?id=NG7sS51zVF](https://openreview.net/forum?id=NG7sS51zVF). yang Liu, S., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., and Chen, M.-H. DoRA: Weight-decomposed low-rank adaptation. In *Fortyfirst International Conference on Machine Learning*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=3d5CIRG1n2) [id=3d5CIRG1n2](https://openreview.net/forum?id=3d5CIRG1n2). Yu, M., Wang, D., Shan, Q., and Wan, A. The super weight in large language models. *arXiv preprint arXiv:2411.07191*, 2024a. Yu, Z., Wang, Z., Fu, Y., Shi, H., Shaikh, K., and Lin, Y. C. Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 57659–57677. PMLR, 21– 27 Jul 2024b. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v235/yu24l.html) [press/v235/yu24l.html](https://proceedings.mlr.press/v235/yu24l.html).

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. *Advances in neural information processing systems*, 33:17283–17297, 2020. Zhang, Y., Du, Y., Luo, G., Zhong, Y., Zhang, Z., Liu, S., and Ji, R. Cam: Cache merging for memoryefficient LLMs inference. In *Forty-first International Conference on Machine Learning*, 2024. URL [https:](https://openreview.net/forum?id=LCTmppB165) [//openreview.net/forum?id=LCTmppB165](https://openreview.net/forum?id=LCTmppB165). Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Re, C., Barrett, C., ´ Wang, Z. A., and Chen, B. H2o: Heavy-hitter oracle for efficient generative inference of large language models. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 34661–34710. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf) [cc/paper\\_files/paper/2023/file/](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf) [6ceefa7b15572587b78ecfcebb2827f8-Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf)-Conference. [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf). Zhao, Y., Du, L., Ding, X., Xiong, K., Sun, Z., Jun, S., Liu, T., and Qin, B. Deciphering the impact of pretraining data on large language models through machine unlearning. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Findings of the Association for Computational Linguistics ACL 2024*, pp. 9386–9406, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL [https://aclanthology.org/](https://aclanthology.org/2024.findings-acl.559) [2024.findings-acl.559](https://aclanthology.org/2024.findings-acl.559). Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging LLM-as-a-judge with MT-bench and chatbot arena. In *Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2023. URL [https:](https://openreview.net/forum?id=uccHPGDlao) [//openreview.net/forum?id=uccHPGDlao](https://openreview.net/forum?id=uccHPGDlao). Zheng, Y., Zhang, R., Zhang, J., Ye, Y., and Luo, Z. Llamafactory: Unified efficient fine-tuning of 100+ language models. *arXiv preprint arXiv:2403.13372*, 2024.

689 690

694

696

698

700

704

706

708 709

711

## A. Implementation Details

We use 8xA100-80GB, for our all implementations. As discussed in Section [2.2,](#page-2-3) we use only the *bos* token to analyze massive activations and massive weights, and design MacDrop. In Section [2.2,](#page-2-3) we use the example and visualization of massive activations. For parameter-efficient fine-tuning, Llama-Factory is used and configurations are summarized in Table [6.](#page-12-2) For evaluation, we use the code of massive activations for perplexity, use lm-eval-harness for zero-shot accuracy, and use FastChat and Prometheus for generation tasks. Related papers and codes are cited in the main.

Table 6. Configuration for low rank adaptation (LoRA and DoRA).

| Argument   |        |              | Setting                                                  |
|------------|--------|--------------|----------------------------------------------------------|
| dataset    |        |              | alpaca gpt4 en                                           |
| validation |        | size         | 0.05                                                     |
| per        | device | train batch  | size 8                                                   |
| gradient   |        | accumulation | steps 4                                                  |
| learning   |        | rate         | 1e-4                                                     |
| num        | train  | epochs       | 3                                                        |
| warmup     |        | ratio        | 0.05                                                     |
| adam       | β 1    |              | 0.9                                                      |
| adam       | β 2    |              | 0.999                                                    |
| lora       | target |              | all linear layers except for embedding layer and lm head |
| lora       | rank   |              | 16                                                       |
| lora       | alpha  |              | 16                                                       |

# B. Position of Massive Weights

Table [7](#page-12-3) summarizes the position of massive weights across various models. These are selected based on the magnitudes of intermediate state in Appendix [D.](#page-19-0)

Table 7. Layer and indices of top-5 massive weights.

| Model                        | Layer        | Top-    | 5 indices |            |             |        |
|------------------------------|--------------|---------|-----------|------------|-------------|--------|
| Llama-2-7b-hf                | 2            | [7890,  | 10411,    | 1192,      | 8731,       | 5843]  |
| Llama-2-7b-chat-hf           | 2            | [7890,  | 10411,    | 1192,      | 8731,       | 5843]  |
| Llama-2-13b-hf               | 4            | [7678,  | 8811,     | 11371,     | 6619,       | 12281] |
| Llama-2-13b-chat-hf          | 4            | [7678,  | 8811,     | 11371,     | 6619,       | 12281] |
| Meta-Llama-3-8B              | 2            | [2427,  | 198,      | 6412,      | 12657,      | 591]   |
| Meta-Llama-3-8B-Instruct     | 2            | [2427,  | 198,      | 6412,      | 591, 12657] |        |
| Meta-Llama-3-70B             | 4            | [16581, | 3590,     | 16039,     | 19670,      | 13266] |
| Meta-Llama-3-70B-Instruct    | 4            | [16581, | 3590,     | 16039,     | 19670,      | 13266] |
| Meta-Llama-3.1-405B          | (8bit) 6     | [11891, | 30740,    | 2392,      | 36238,      | 12328] |
| Meta-Llama-3.1-405B-Instruct | (8bit) 6     | [11891, | 30740,    | 36238,     | 2392,       | 1073]  |
| Mistral-7B-v0.1              | 2            | [7310,  | 8572,     | 2514,      | 1878,       | 8693]  |
| Mistral-7B-Instruct-v0.1     | 2            | [7310,  | 8572,     | 2514,      | 2484,       | 1878]  |
| Mixtral-8x7B-v0.1            | 2 (expert 4) | [7310,  | 7530,     | 11981,     | 7492,       | 3178]  |
| Mixtral-8x7B-Instruct-v0.1   | 2 (expert 4) | [7310,  | 11981,    | 2514,      | 7530,       | 3178]  |
| Phi-3-mini-4k-instruct       | 3            | [808,   | 340,      | 3644,      | 2473, 2987] |        |
| Phi-3-medium-4k-instruct     | 6            | [181,   | 7540,     | 19, 15874, | 5137]       |        |
| gemma-2-2b                   | 2            | [1257,  | 2896,     | 6954,      | 8624,       | 7118]  |
| gemma-2-2b-it                | 2            | [1257,  | 2896,     | 6954,      | 8624,       | 9140]  |
| gemma-2-9b                   | 1            | [2769,  | 6656,     | 4889,      | 14293,      | 11065] |
| gemma-2-9b-it                | 1            | [2769,  | 6656,     | 4889,      | 14293,      | 10429] |
| gemma-2-27b                  | 10           | [34659, | 32862,    | 9590,      | 8959,       | 32744] |
| gemma-2-27b-it               | 10           | [34659, | 32862,    | 9590,      | 32744,      | 8959]  |

## C. *bos* Token Analysis for Various LLMs

In this section, we provide the magnitudes of activations of the hidden state and normalized attention scores according to the position of the *bos* token, after massive activations appear (specifically, in the middle layer), for various LLM families.

#### C.1. Llama-2 family

Llama-2-7B (Figure [8\)](#page-13-1) has massive activations at the starting token or first delimiter token (first column). When the *bos* token is placed in the starting position, it triggers massive activations and the 'Summer' token loses its massive activations, while first delimiter token '.' still keeps its massive activations (second column). When the *bos* token is placed in the middle or ending position after the first delimiter token, it does not trigger massive activations (third and fourth columns).

Llama-2-13B (Figure [9\)](#page-13-2) has massive activations only at the starting token, other than Llama-2-7B (first column). In cases where the *bos* token is inserted, the same tendencies are observed as with the LLaMA-2-7B model.

![](_page_13_Figure_6.jpeg)

Figure 8. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Llama-2-7b-hf.

![](_page_13_Figure_8.jpeg)

Figure 9. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Llama-2-13b-hf.

#### C.2. Llama-3 family

Llama-3-8B (Figure [10\)](#page-14-0) does not have massive activations at delimiter tokens such as '.' (first column). When the *bos* token is placed in the starting position, it triggers massive activations and the 'Summer' token loses its massive activations, similar to Llama-2 family (second column). When the *bos* token is placed in the middle or ending position, it also triggers massive activations in the same feature dimensions (third and fourth columns). Namely, the *bos* token has massive activations, regardless of its position. What is intriguing is that, despite the difference in magnitude according to the position, the *bos* token similarly exhibits attention sinks.

Llama-3-70B (Figure [11\)](#page-14-1) generally exhibits similar trends to Llama-3-8B. One notable difference is that the degree of sinking for the token at the first position is significantly stronger compared to that of the Llama-3-8B.

![](_page_14_Figure_4.jpeg)

Figure 10. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Meta-Llama-3-8B.

![](_page_14_Figure_6.jpeg)

Figure 11. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Meta-Llama-3-70B.

#### C.3. Mistral and Mixtral family

Mistral (Figure [12\)](#page-15-0) does not exhibit massive activations at the starting position and does not trigger attention sink, contrary to previous findings observed by [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6). Rather, massive activations are observed only at the first delimiter token (first column). When the *bos* token is placed in the starting position, the first delimiter token loses its massive activations (second column). However, when the *bos* token is placed in the middle or ending position after the first delimiter token, massive activations are observed in both tokens (third and fourth columns). Similar to Llama-3 family, the *bos* token has massive activations, regardless of its position.

Mixtral (Figure [13\)](#page-15-1) exhibits the same behavior as Mistral. The only difference is observed in the magnitude of its massive activations, with Mixtral producing values approximately ten times higher than Mistral.

![](_page_15_Figure_4.jpeg)

Figure 12. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Mistral-7B-v0.1.

![](_page_15_Figure_6.jpeg)

Figure 13. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Mixtral-8x7B-v0.1.

#### C.4. Phi-3 family

Phi-3-mini (Figure [14\)](#page-16-0) and Phi-3-medium (Figure [15\)](#page-16-1) exhibit a similar tendency to Llama-2-13B. This family has massive activations only at the starting token (first column), with a similar response when the *bos* token is inserted (second, third, and fourth column). A significant distinction between the Llama-2-13B model and the Phi-3 family lies in their attention mechanisms. Specifically, the Phi-3 family demonstrates weaker attention on the token at the first position than Llama-2-13B model. This reduced attention appears to be primarily redistributed to recent tokens.

![](_page_16_Figure_3.jpeg)

Figure 14. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Phi-3-mini-4k-instruct.

![](_page_16_Figure_5.jpeg)

Figure 15. (Top) Magnitudes of the hidden state and (Bottom) attention scores of Phi-3-medium-4k-instruct.

#### C.5. Gemma-2 family

The Gemma-2 family displays significantly distinct magnitudes of activations and attention scores when compared to other model families. This divergence remains consistent regardless of the absence (first column) or presence (other columns) of the *bos* token.

Gemma-2-2b (Figure [16\)](#page-17-0) and Gemma-2-9b (Figure [17\)](#page-17-1) do not exhibit noticeably large values along either the token axis or the feature dimension axis, from the perspective of magnitudes of activations (first column, top). This suggests that massive activations are not present. As a result, the attention mechanism avoids the attention sink phenomenon and demonstrates a strong attention on the locality of recent tokens (first column, bottom). However, when the *bos* token is fed into these models, it exhibits massive activations with extremely large values in certain feature dimensions, regardless of its position (second, third, and fourth columns). Nevertheless, compared to other models where attention sinks occur, they allocate significantly greater attention to recent tokens (especially, to its own tokens).

![](_page_17_Figure_4.jpeg)

Figure 16. (Top) Magnitudes of the hidden state and (Bottom) attention scores of gemma-2-2b.

![](_page_17_Figure_6.jpeg)

Figure 17. (Top) Magnitudes of the hidden state and (Bottom) attention scores of gemma-2-9b.

 Gemma-2-27b (Figure [18\)](#page-18-0) demonstrates a distinct behavior compared to smaller models. It exhibits noticeably large values along the feature dimension axis across all tokens, from the perspective of magnitudes of activations (first column, top). This distribution, where the differences between tokens are not pronounced, fails to create attention sinks (first column, bottom). When the *bos* token is placed in the starting position, it triggers massive activations and attention sinks by generating value that exceed the magnitudes of other tokens by more than tenfold, in the certain feature dimension (second column). However, when the *bos* token is placed in the middle or ending position, it does not trigger massive activations, similar to when the *bos* token is absent (third and fourth columns).

![](_page_18_Figure_3.jpeg)

 In summary,

Figure 18. (Top) Magnitudes of the hidden state and (Bottom) attention scores of gemma-2-27b.

- Llama-2, Llama-3, and Phi-3 families have massive activations at the first position.
- Llama-3, Mistral/Mixtral families, and Gemma-2-2/9b models have massive activations at the *bos* token.
- All families have massive activations at the *bos* token placed at the first position.

 When comparing pre-trained LLMs (e.g., Llama-2-7b-hf) and instruction-tuned LLMs (e.g., Llama-2-7b-chat-hf) of the same model, the shape of their graphs is almost identical. This suggests that massive weights are formed during the pre-training process. Llama-2, Llama-3, Mistral, Mixtral, and Phi-3 exhibit similar patterns in their hidden states: following a single explosive amplification in an early layer, massive activations are sustained through residual connections almost until the final layer, although Phi-3 experiences a few additional amplifications. In fact, as we discuss in the main, such an explosion initially occurs in the intermediate state, and this phenomenon is observed across different models. However, the behavior of the Gemma-2 family significantly deviates from that of other models. Firstly, instead of the values being maintained in the hidden state, Gemma-2 shows a continuous increase followed by a decrease. Secondly, the magnitude of the explosion observed in the intermediate state is considerably lower compared to other models. These unique characteristics suggest that Gemma-2 operates under different internal dynamics, which may influence its overall performance and stability.

## D. Futher Analysis for Various LLMs

We investigate various LLM families: Llama-2 [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0), Llama-3 [\(Dubey et al.,](#page-8-1) [2024\)](#page-8-1), Mistral [\(Jiang et al.,](#page-9-6) [2023\)](#page-9-6) and Mixtral [\(Jiang et al.,](#page-9-7) [2024\)](#page-9-7), Phi-3 [\(Abdin et al.,](#page-8-7) [2024\)](#page-8-7), and Gemma-2 [\(Team et al.,](#page-10-9) [2024\)](#page-10-9). Similar to Figure [3](#page-3-0) in the main, we provide the top-3 and median magnitudes in the hidden states and the intermediate states throughout the layers. In subcaptions, H and I represent the hidden state and the intermediate state, respectively.

## D.1. Llama-2 family

![](_page_19_Figure_5.jpeg)

Figure 19. (Left) Hidden state and (Right) intermediate state of Llama-2 family.

#### D.2. Llama-3 family

![](_page_20_Figure_3.jpeg)

Figure 20. (Left) Hidden state and (Right) intermediate state of Llama-3/3.1 family.

### D.3. Mistral and Mixtral family

![](_page_21_Figure_2.jpeg)

Figure 21. (Left) Hidden state and (Right) intermediate state of Mistral and Mixtral family.

D.4. Phi-3 family

![](_page_21_Figure_5.jpeg)

Figure 22. (Left) Hidden state and (Right) intermediate state of Phi-3 family.

## D.5. Gemma-2 family

![](_page_22_Figure_2.jpeg)

Figure 23. (Left) Hidden state and (Right) intermediate state of Gemma-2 family.

## E. Attention Sinks

Figure [24](#page-23-1) describes attention after Softmax in the early layers (from layer 1 to layer 8) across various models. Attention sinks are observed in the layers after the massive weights layer. In Llama-2-7B (Figure [24\(](#page-23-1)a)), Mistral-7B (Figure [24\(](#page-23-1)c)), and Mixtral-8x7B (Figure [24\(](#page-23-1)d)), sink tokens are the initial token ('Summer') and the first delimiter token ('.'), discovered by [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6). In Llama-3-8B (Figure [24\(](#page-23-1)b)) and Phi-3-mini (Figure [24\(](#page-23-1)e)), sink token is the only the initial token ('Summer'). Interestingly, in these five models, it is commonly observed that significant attention is concentrated on non-semantic tokens ('.') before attention sinks occur. However, in Gemma-2 (Figure [24\(](#page-23-1)f)), attention sinks do not happen and attention is primarily assigned to local tokens. Note that what we provide is the average of the heads, and there might be heads that do not fully sink when viewed individually.

![](_page_23_Figure_3.jpeg)

Figure 24. Attention after Softmax.

1324 We address the potential limitations of algorithms based on massive phenomena. Table [8](#page-24-1) presents the results on zero-shot downstream tasks across different LLMs. The results show that MacDrop is not effective for LLMs that are not sensitive to massive weights, such as Phi-3-medium and Gemma-2 family, as shown in Section [2.](#page-1-1)

1326 1329 Nevertheless, we believe that clearly demonstrating such limitations is itself meaningful. Most previous and ongoing research addressing massive phenomena have stated that various models exhibit massive phenomena, and have proposed algorithms and experimental results excluding Gemma-2 family. Because we conduct experiments on a wide variety of recent LLMs, we are able to reveal such limitations. Additionally, we would like to emphasize that, similar to previous studies, MacDrop demonstrates sufficient feasibility for LLMs exhibiting massive phenomena.

1334

1336

1354

1356

1369

# F. Zero-shot Downstream Task across Various LLMs

Table 8. Zero-shot downstream tasks performance across different LLMs.

| Model Method           | ARC-Easy | ARC-Challenge | BoolQ | PIQA | WinoGrande | Avg. |
|------------------------|----------|---------------|-------|------|------------|------|
| LoRA                   | 77.1     | 51.9          | 82.2  | 81.6 | 72.8       | 73.1 |
| + MacDrop Llama-2-13B  | 78.8     | 52.6          | 82.2  | 71.4 | 71.9       | 73.4 |
| DoRA                   | 77.4     | 51.8          | 82.0  | 81.6 | 72.8       | 73.1 |
| + MacDrop              | 77.8     | 52.5          | 81.9  | 81.8 | 72.3       | 73.3 |
| LoRA                   | 79.6     | 58.2          | 83.9  | 82.4 | 75.9       | 76.0 |
| + MacDrop Llama-3-8B   | 82.9     | 58.3          | 83.9  | 82.6 | 75.0       | 76.5 |
| DoRA                   | 80.8     | 57.7          | 83.9  | 82.5 | 75.8       | 76.1 |
| + MacDrop              | 81.9     | 58.2          | 83.9  | 82.2 | 75.6       | 76.4 |
| LoRA                   | 87.5     | 66.3          | 86.1  | 84.9 | 80.4       | 81.0 |
| + MacDrop Llama-3-70B  | 87.5     | 66.5          | 86.1  | 85.0 | 80.8       | 81.2 |
| DoRA                   | 87.5     | 66.6          | 86.1  | 85.0 | 80.8       | 81.2 |
| + MacDrop              | 87.4     | 66.6          | 86.0  | 85.1 | 80.9       | 81.2 |
| LoRA                   | 78.5     | 54.9          | 84.9  | 82.9 | 75.3       | 75.3 |
| + MacDrop Mistral-7B   | 80.9     | 56.7          | 85.0  | 83.0 | 75.3       | 76.2 |
| DoRA                   | 78.4     | 55.1          | 85.0  | 82.9 | 75.1       | 75.3 |
| + MacDrop              | 80.6     | 56.7          | 85.3  | 82.9 | 75.1       | 76.1 |
| LoRA                   | 72.5     | 53.7          | 86.4  | 80.1 | 74.0       | 73.3 |
| + MacDrop Phi-3-mini   | 75.0     | 54.7          | 86.2  | 80.5 | 74.0       | 74.1 |
| DoRA                   | 72.3     | 53.3          | 86.4  | 80.0 | 73.6       | 73.1 |
| + MacDrop              | 72.9     | 53.5          | 86.3  | 80.0 | 73.7       | 73.3 |
| LoRA                   | 81.4     | 62.2          | 88.7  | 82.6 | 76.4       | 78.3 |
| + MacDrop Phi-3-medium | 81.2     | 61.9          | 88.7  | 82.5 | 76.4       | 78.1 |
| DoRA                   | 80.92    | 62.0          | 88.6  | 82.4 | 76.2       | 78.0 |
| + MacDrop              | 80.8     | 61.9          | 88.5  | 82.4 | 76.2       | 77.9 |
| LoRA                   | 81.6     | 54.4          | 79.8  | 79.2 | 68.7       | 72.7 |
| + MacDrop Gemma-2-2b   | 81.5     | 54.2          | 79.6  | 79.2 | 68.6       | 72.6 |
| DoRA                   | 81.4     | 54.0          | 79.6  | 79.3 | 68.7       | 72.6 |
| + MacDrop              | 81.6     | 53.9          | 79.4  | 79.3 | 68.7       | 72.6 |
| LoRA                   | 89.6     | 69.0          | 86.5  | 82.8 | 75.3       | 80.6 |
| + MacDrop Gemma-2-9b   | 89.2     | 68.4          | 86.5  | 82.8 | 75.2       | 80.4 |
| DoRA                   | 89.4     | 68.9          | 86.2  | 82.7 | 75.8       | 80.6 |
| + MacDrop              | 89.2     | 68.5          | 86.3  | 82.8 | 75.7       | 80.5 |
| LoRA                   | 87.5     | 69.0          | 86.0  | 84.1 | 80.6       | 81.4 |
| + MacDrop Gemma-2-27b  | 87.3     | 68.8          | 86.0  | 84.2 | 80.5       | 81.4 |
| DoRA                   | 88.5     | 69.4          | 85.6  | 84.5 | 80.0       | 81.6 |
| + MacDrop              | 88.1     | 69.2          | 85.7  | 84.3 | 80.0       | 81.5 |

1379

1389 1390

1394

1396

## G. Generation Tasks

We evaluate on the generated texts of the same models in Section [4.1](#page-6-4) using the Spec-Bench dataset [\(Xia et al.,](#page-10-13) [2024\)](#page-10-13). This benchmark includes six subtasks, each containing 80 instances: multi-turn (MT) conversation from MT-bench [\(Zheng et al.,](#page-11-2) [2023\)](#page-11-2), translation from WMT14 DE-EN [\(Bojar et al.,](#page-8-9) [2014\)](#page-8-9), summarization from CNN/Daily Mail [\(Nallapati et al.,](#page-9-11) [2016\)](#page-9-11), question answering (QA) from Natural Questions [\(Kwiatkowski et al.,](#page-9-12) [2019\)](#page-9-12), mathematical reasoning from GSM8K [\(Cobbe](#page-8-10) [et al.,](#page-8-10) [2021\)](#page-8-10), and retrieval-augmented generation (RAG) from Natural Questions [\(Kwiatkowski et al.,](#page-9-12) [2019\)](#page-9-12). We utilize the direct assessment of Prometheus-2-7B [\(Kim et al.,](#page-9-13) [2024\)](#page-9-13) to evaluate the generated texts using a 5-point Likert scale. Prometheus-2-7B is an open-source language model specifically designed for evaluation purposes[<sup>6</sup>](#page-25-0) . Table [9](#page-25-1) presents the results on generation tasks. MT-1 and MT-2 indicate the first turn and second turn, respectively. Unfortunately, MacDrop shows limited performance improvements in generation tasks. Table [10](#page-25-2) provides examples of the generated texts and judgements. Assistant A and B indicate Llama-3-8B with and without MacDrop, respectively.

Table 9. Generation tasks performance measured by Prometheus-2-7B.

| Table Model Method   | 9. MT-1 | Generation MT-2 | tasks performance translation | measured summarization | by QA | Prometheus-2-7B. math reasoning | RAG  | Avg. |
|----------------------|---------|-----------------|-------------------------------|------------------------|-------|---------------------------------|------|------|
| LoRA                 | 3.71    | 3.54            | 4.50                          | 3.24                   | 4.35  | 3.64                            | 3.74 | 3.82 |
| + MacDrop Llama-3-8B | 3.76    | 3.49            | 4.51                          | 3.39                   | 4.29  | 3.50                            | 3.71 | 3.81 |
| DoRA                 | 3.85    | 3.71            | 4.56                          | 3.34                   | 4.35  | 3.59                            | 3.73 | 3.80 |
| + MacDrop            | 3.79    | 3.41            | 4.55                          | 3.26                   | 4.29  | 3.79                            | 3.85 | 3.85 |
| LoRA                 | 3.55    | 3.30            | 4.61                          | 3.33                   | 4.41  | 3.45                            | 4.05 | 3.81 |
| + MacDrop Mistral-7B | 3.75    | 3.42            | 4.58                          | 3.26                   | 4.35  | 3.30                            | 3.98 | 3.81 |
| DoRA                 | 3.55    | 3.29            | 4.64                          | 3.34                   | 4.49  | 3.35                            | 3.88 | 3.79 |
| + MacDrop            | 3.49    | 3.29            | 4.59                          | 3.52                   | 4.41  | 3.33                            | 3.95 | 3.80 |

Table 10. Example of generation task (translation).

| User             | Translate |               | German    | to         | English:    | St.       |               | Hubertus | war           | der Sage     | nach       | ein recht    | ¨          | rucksichtsloser |               |
|------------------|-----------|---------------|-----------|------------|-------------|-----------|---------------|----------|---------------|--------------|------------|--------------|------------|-----------------|---------------|
|                  | Jager, ¨  | der           | durch     | eine       | Vision      |           | bekehrt       |          | und sp ater ¨ | zum          | Bischof    | von          | L uttich ¨ |                 | ernannt       |
| Reference answer | St.       | Hubert        | was,      |            | according   | to        | legend,       | a        | truly         | ruthless     | hunter,    | who          | converted  |                 | following     |
| a                | vision    |               | and was   | later      |             | appointed |               | Bishop   | of Luttich. ¨ |              |            |              |            |                 |               |
| Assistant A      |           | According     | to        | legend,    | St.         | Hubertus  | was           | a        | rather        | ruthless     | hunter     | who          | was        | converted       | and           |
|                  | later     |               | appointed | as the     | Bishop      | of        | Liege `       |          | after a       | vision.      |            |              |            |                 |               |
| Judgment for A   | The       | response      |           | provided   | a           |           | translation   | that     | was           | generally    | accurate   | and          |            | relevant        | to the        |
|                  | user’s    |               | request.  | However,   |             | there     | were          | minor    |               | inaccuracies | in the     | phrasing,    | such       | as              | ”rather       |
|                  | ruthless” |               | instead   | of         | ”truly      |           | ruthless,”    |          | which         | slightly     | deviated   | from         | the        | original        | text.         |
|                  |           | Additionally, |           | the term   |             | ”Liege” ` | was           | used     | instead       | of ”L        | uttich,” ¨ | which        | is         | the             | correct       |
|                  | name      | of            | the city  | in the     |             | original  | German        |          | text.         | Despite      | these      | minor        | issues,    | the             | response      |
|                  | still     | conveyed      |           | the main   |             | points    | of the        |          | legend about  | St.          | Hubertus.  |              | Therefore, |                 | while the     |
|                  | response  |               | was       | helpful    | and         | mostly    |               | aligned  | with          | the user’s   |            | inquiry, it  | did not    |                 | perfectly     |
|                  | cater     | to the        | user’s    | needs      | due         | to        | the           | slight   | inaccuracies  | in           | the        | translation. | So         | the             | overall       |
|                  | score     | is 4.         |           |            |             |           |               |          |               |              |            |              |            |                 |               |
| Assistant B      |           | According     | to        | legend,    | St.         | Hubertus  |               | was      | a rather      | reckless     | hunter     | who          | was        |                 | converted     |
|                  | and       | later         | appointed | as         | the         | Bishop    | of            | Liege `  | after         | a vision.    |            |              |            |                 |               |
| Judgment for B   | The       | response      |           | provided   | a           |           | translation   | that     | was           | generally    | accurate   | and          |            | relevant        | to the        |
|                  | user’s    |               | request.  |            | However,    | there     | were          |          | slight        | deviations   | from       | the          | original   | text.           | For           |
|                  | instance, |               | the       | term       | ”rather     | reckless” |               | is       | less precise  | than         | ”truly     | ruthless,”   |            | which           | could         |
|                  | lead      | to            | some      | ambiguity. |             |           | Additionally, |          | the term      | ”Liege” `    | is         | used         | instead of | ”L              | uttich,” ¨    |
|                  | which     | is            | a minor   |            | inaccuracy. |           | Despite       |          | these minor   | issues,      | the        | response     | still      |                 | managed       |
| to               |           | convey        | the       | main       | points      | of        | the           | original | text,         | making       | it useful  | for          | the        | user’s          | needs.        |
|                  |           | Therefore,    | the       | response   |             | aligns    | well          | with     | the user’s    | inquiry,     | with       | only         | rare       |                 | inaccuracies, |
|                  | and       | thus          | meets     | the        | criteria    | for       | a score       | of       | 4.            |              |            |              |            |                 |               |

<sup>6</sup> https://github.com/prometheus-eval/prometheus-eval

 The attention sinks phenomenon and their importance, uncovered by [Xiao et al.](#page-10-4) [\(2024\)](#page-10-4), have been widely used to compress key-value caches. For quantization, KVQuant [\(Hooper et al.,](#page-9-14) [2024\)](#page-9-14) applies attention sink-aware quantization, which retains only the first token in fp16. CushionCache [\(Son et al.,](#page-10-14) [2024\)](#page-10-14) inserts sink tokens into the prefix to mitigate massive activations in the middle of the sequence, enhancing the performance of quantized models. For token eviction and token merging, sink tokens are never evicted or merged; they remain unchanged [\(Xiao et al.,](#page-10-4) [2024;](#page-10-4) [Ge et al.,](#page-8-11) [2024;](#page-8-11) [Li et al.,](#page-9-15) [2024;](#page-9-15) [Zhang et al.,](#page-11-3) [2023;](#page-11-3) [Wang et al.,](#page-10-15) [2024;](#page-10-15) [Zhang et al.,](#page-11-4) [2024\)](#page-11-4).

 Nevertheless, there has been limited in-depth research on the phenomenon itself. In fact, the idea of global attentions, such as [CLS] and [SEP] tokens–similar to attention sinks–was introduced and emphasized even before the LLM era [\(Zaheer et al.,](#page-11-5) [2020;](#page-11-5) [Beltagy et al.,](#page-8-12) [2020\)](#page-8-12). In the LLM era, [Yu et al.](#page-10-5) [\(2024b\)](#page-10-5) showed that sink tokens can appear not only at the beginning of a sentence but also in the middle, and they are often shown to be nonsemantic (e.g., '.'). [Sun et al.](#page-10-6) [\(2024a\)](#page-10-6) discovered the presence of massive activations in the hidden state space of sink tokens, demonstrating that massive activations trigger the attention sinks phenomenon. Meanwhile, in vision transformers, similar phenomenon is observed [\(Darcet et al.,](#page-8-13) [2024\)](#page-8-13). They showed that training with register tokens, which is additional meaningless tokens similar to sink tokens, resulted in improved dense prediction and interpretability. Different from previous work, we explore this phenomenon in the weight space.

 Specifically, we define the massive weights in an activation-aware manner using the *bos* token. Similarly, Wanda [\(Sun et al.,](#page-10-16) [2024b\)](#page-10-16) with a structured pruning [\(An et al.,](#page-8-14) [2024\)](#page-8-14) and AWQ [\(Lin et al.,](#page-9-16) [2024\)](#page-9-16) calculate weight importance scores based on a small of calibration data. However, it is important to note that the massive weights are confined to a specific single layer, whereas Wanda and AWQ identify important weights within every linear layer. In other words, the massive weights would be included among those selected through Wanda or AWQ. Our contribution focuses more deeply on a narrowly defined aspect compared to these studies.

 Concurrently with our work, [Yu et al.](#page-10-17) [\(2024a\)](#page-10-17) have defined "super weight," which is a single weight that can significantly influence the performance of LLMs. To compare more specifically, we identify that the intermediate state in the layer l possesses massive activations. Accordingly, we define massive weight as the corresponding rows of Wup. However, [Yu](#page-10-17) [et al.](#page-10-17) [\(2024a\)](#page-10-17) take into account both the intermediate state and the subsequent hidden state. In other words, they define super weight as the intersection of the corresponding column of Wdown to the super activation in the intermediate state and the corresponding row of Wdown to the super activation in the hidden state, at the layer l. In summary, we examine the intermediate state exclusively from an output perspective (i.e., Wup), while their approach appears to have considered it from an input perspective (i.e., Wdown), incorporating the hidden state.

## H. Related work