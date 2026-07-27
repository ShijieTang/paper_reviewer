Transformers can capture long-range dependencies using self-attention, allowing tokens to attend to all others directly. However, stacking multiple attention layers leads to attention concentration. One natural way to address this issue is to use cross-layer attention, allowing information from earlier layers to be directly accessible to later layers. However, this approach is computationally expensive. To address this problem, we propose Transformer with residual value (ResFormer) which approximates cross-layer attention through adding a residual connection from the values of the the first layer to all subsequent layers. Based on this method, one variant is the Transformer with single layer value (SVFormer), where all layers share the same value embedding from first layer, reducing the KV cache by nearly 50%. Comprehensive empirical evidence demonstrates that ResFormer mitigates attention concentration problem in deeper layers and enhances representation across most layers, outperforming the vanilla Transformer, DenseFormer, and NeuTRENO in training error as well as downstream tasks. SVFormer trains significantly faster than the vanilla Transformer and performs better than other methods like GQA and CLA, with performance influenced by sequence length and cumulative learning rate.

Llama 8B V3.1 Mistral 7B V0.2 Llama 8B Instruct V3.1 Mistral 7B Instruct V0.2 Entropy Token Similarity 0.50 0.55 0.60 0.65 0.70 0.75 0 5 10 15 20 25 30 Layer Index 2 3 4 5 6 0 5 10 15 20 Layer Index 7.194 7.196 7.198 7.200 7.202 7.204 0 5 10 15 20 25 Layer Index 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 Avera ge E
ntropy Avera ge En tropy Toke n Sim ilarit y Entrop y Transformer NeuTRENO
Resformer SVFormer

Figure 1: (Left) The average entropy of token importance and the average hidden-state similarity for a randomly initialized 468M model. (Middle) The average entropy of token importance across layers in Llama (8B) (Dubey et al., 2024) and Mistral (7B) (Jiang et al., 2023). (Right) The average entropy of token importance across layers in ResFormer vs. the vanilla Transformer, where token importance is derived from the attention matrix. Lower entropy indicates more focused attention on specific tokens. More details can be found in Eqn. 11.

## 1 Introduction

The Transformer (Vaswani et al., 2017) model has become one of the leading architectures in recent years, excelling in both language modeling (Devlin et al., 2019; Lan et al., 2020; Brown et al., 2020) and computer vision tasks (Dosovitskiy et al., 2021). The discovery of scaling laws (Hoffmann et al.,
2022; Kaplan et al., 2020) has driven the pursuit of larger Transformer models by increasing network depth and width. Training large models presents significant challenges. Balancing the depth and width of a Transformer model within a fixed parameter budget is particularly difficult. While research indicates that deeper models generalize more compositionally than shallower ones (Petty et al., 2024), the training and deployment of deep models remain problematic. Although Transformers use residual connections (He et al., 2016) to address the vanishing gradient issue, training very 1 Anonymous authors Paper under double-blind review

## Abstract

# Value Residual Learning For Alleviating At- Tention Concentration In Transformers

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

0 5 10 15 20 25 Layer Index 0 500 1000 1500 2000 2500 3000 0 5 10 15 20 25 Layer Index 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 Transformer NeuTRENO
Resformer SVformer Transformer NeuTRENO
Resformer SVformer Transformer NeuTRENO Resformer SVformer 0 5 10 15 20 25 Layer Index 0.0 0.1 0.2 0.3 0.4 0.5 Token Impo rtan ce Value
-stat e No rm Hidd en-sta te N
orm
(a) Token importance.
Figure 2: The token importance (Xiao et al., 2024), value-state norms (Guo et al., 2024b), and hidden-state norms (Sun et al., 2024) of the first token across layers of 468M models. More Visualization results are available in Appendix A.2. deep Transformers is still challenging. For example, a 32-layer Vision Transformer (ViT) may perform worse than a 24-layer one (Zhou et al., 2021). This is mainly due to the smoothing mechanism of attention (Shi et al., 2022), which can lead to an over-smoothing effect (Nguyen et al., 2023) where the token representations become the same as the model's depth increases. Existing solutions to alleviate the over-smoothing problem in Transformer include adding extra regularizers (Nguyen et al., 2023; Shi et al., 2022) and optimizing the information flow within the model (Pagliardini et al., 2024). During the era of convolutional neural network architectures, Stochastic Depth (Huang et al., 2016) reduces the likelihood of over-smoothing by randomly dropping layers during training and DenseNet (Huang et al., 2017) mitigates the impact of over-smoothing by allowing each layer to directly access the hidden states of all preceding layers. Recently, DenseFormer (Pagliardini et al., 2024) adopts the idea of DenseNet when training Transformer. Additionally, NeuTRENO (Nguyen et al., 2023) alleviates over-smoothing through incorporating the difference between the value vectors of the first layer and the current layer to the attention output. In this paper, we address the problem of multi-layer attention from another perspective. We introduce the phenomenon of attention concentration, which describes how a model's attention increasingly focuses on fewer tokens. We quantify the degree of attention concentration using the entropy of the distribution of token importance, where lower entropy indicates a more pronounced concentration. Unlike over-smoothing, which is inherent to model architecture, attention concentration emerges during training. Fig. 1 (Left) shows that randomly initialized models exhibit oversmoothing but not attention concentration. Trained ViT models often focus on low-informative background areas (Darcet et al., 2024), while language models concentrate on low-semantic tokens (Sun et al., 2024), particularly the start token (attention sink (Xiao et al., 2024)). While previous studies analyzed single-layer attention patterns, our research reveals a "concentration - dispersion - concentration" pattern in deep models, as shown in Fig. 1 (Middle), suggesting potential loss of information during concentrated phases. The analysis of over-smoothing is available in Appendix A.1. Mitigating attention concentration can lead to more interpretable attention maps and potentially improve downstream task performance (Darcet et al., 2024). This phenomenon typically emerges after the second or third network layer and is associated with value-state drains (decreased magnitude of value states) (Guo et al., 2024b), and hidden-state peaks (increased magnitude of hidden states) (Sun et al., 2024). Guo et al. (2024a) shows a mutual reinforcement mechanism exists between value-state drains and attention concentration. Recent studies have linked this to implicit biases during pretraining, with most existing solutions focusing on the use of additional tokens (registers) (Darcet et al., 2024) or additional keys and values (explicit attention bias) (Sun et al., 2024) to redirect this. Given that the first layer always shows no attention concentration, an effective method is to use crosslayer attention on information from this layer. However, due to computational costs, we propose ResFormer as an efficient alternative. ResFormer applies a residual connection between the value vectors of the current layer and the first layer before the attention operation. Unlike cross-layer attention, ResFormer indirectly mitigates attention concentration. It leverages the absence of valuestate drains in the first layer by introducing a value residual connection. This alleviates value-state drains in deeper layers, thereby disrupting the mutual reinforcement between attention concentration and value-state drains, as shown in Fig. 1 (Right) and Fig. 2. During inference, deep networks require substantial KV cache, severely impacting model deployment (Xiao et al., 2024). Existing KV -efficient methods often process keys and values simultane-

2 H
2 H
2 H
2 H
2 H
2 A
2 V
2 A
2 V
2 A
2 V
2 A
2 A
2 V
1 H
1 H
1 H
1 H
1 H
1 A
1 V
1 A
1 A
1 V
1 A
1 V
1 A
1 V
0 H
0 H
0 H
0 H
0 H
0 A
0 V
0 A
0 V
0 A
0 V
0 A
0 V
0 A
0 V
Figure 3: Simplified illustration of the vanilla Transformer, NeuTRENO, DenseFormer, ResFormer, and SVFormer, with only three-layer structures and no operations other than attention. Ai, V
i, and Hi denote the attention matrix, value vectors, and attention outputs at the i-th layer, respectively. ⊕,
⊖, and ⊗ represent standard matrix addition, subtraction, and multiplication, respectively.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 2 Related Work 2.1 Shortcut Connections For Better Information Flow

ously. Building on ResFormer, we decouple the value from the attention operation and propose a new kind of Transformer with single layer value (SVFormer). In SVFormer, the queries and keys of all layers share the value from the first layer, and thus it can also alleviate attention concentration. We experiment on a 20B SlimPajama sub-sampled dataset, using settings similar to popular large language models (Wei et al., 2023; Dubey et al., 2024; Kaplan et al., 2020). We compare different models by their relative training curves against the vanilla Transformer. Results show that Res- Former outperforms the vanilla Transformer, DenseFormer, and NeuTRENO. ResFormer achieves equivalent validation loss with 10.4% fewer model parameters and 13.6% less training data compared to Transformer, while maintaining similar memory usage and computational cost. Besides, SVFormer, while reducing the KV -cache by nearly half, requires a 12.2% increase in parameters to achieve the same validation loss as Transformer. And the performance of SVFormer is better when the training sequence length is longer. It further reduces the KV cache when integrated with classical method GQA (Ainslie et al., 2023).

Deep learning models often consist of multiple layers, posing a challenge to minimize information loss during transmission. ResNet (He et al., 2016) mitigates the vanishing gradient problem with identity connections. Stochastic Depth (Huang et al., 2016) enhances training by randomly dropping layers. DenseNet (Huang et al., 2017) allows subsequent layers to directly access the hidden states of all preceding layers. These two methods further enhance the information flow after ResNet. Related research indicates that for advanced Transformer architectures, although increasing depth continues to yield performance improvements in language modeling tasks, the gains become less significant with further increases (Petty et al., 2024). Furthermore, Zhou et al. (2021) illustrates that a 32-layer ViT underperforms a 24-layer ViT. Depth-Wise Attention (ElNokrashy et al., 2024)
allows each query to access the key and value at the same position from previous layers through an attention-like mechanism before the output layer. DenseFormer (Pagliardini et al., 2024) integrates weighted fusion of outputs from all preceding layers after each layer. To explore why increasing depth in Transformers does not yield expected gains, Wang et al. (2022) finds that self-attention acts as a low-pass filter, smoothing token representations in ViTs. Additionally, Shi et al. (2022) investigates over-smoothing from a graph perspective in BERT-based language modeling tasks. Neu162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 TRENO (Nguyen et al., 2023) adds the difference between the value vectors of the first and current layers to each layer's attention output and significantly alleviates the over-smoothing problem. In contrast to these methods, ResFormer accesses and integrates information from previous layers prior to the attention operation, as illustrated in Fig. 3. Moreover, it does not require the selection or tuning of additional hyperparameters.

## 2.2 Kv Cache Compressing

The KV cache is a key factor limiting the efficiency of long-text model inference. Research in this area can be broadly classified into Transformer-based methods, which target redundant information in Transformer models, and non-Transformer methods, which mainly addresses the quadratic time complexity of attention with respect to sequence length. For non-Transformer methods, Mamba (Gu & Dao, 2023) and RWKV (Peng et al., 2023) are two popular works. They replace the original softmax-based attention with SSM (Gu et al., 2021) and AFT (Zhai et al., 2021) mechanisms, respectively. Besides, several approaches have been proposed to enhance models' ability to process long texts while reducing the reliance on KV cache. Dai et al.

(2019) advocates segmenting long texts into smaller parts for attention computation. Furthermore, Munkhdalai et al. (2024) uses a fixed-size memory matrix for storing and retrieving past information.

Transformer-based methods can be categorized into three main groups. The first group consists of post-training methods like SnapKV (Li et al., 2024) and ThinK (Xu et al., 2024), which compress KV cache during inference based on attention matrices at token or hidden dimension levels. The second group focuses on quantization and adopts low-precision KV cache quantization rather than completely eliminating them (Hooper et al., 2024). The third group aims to maximize the efficiency of attention-based models via parameter or activation value sharing. The most representative works include Multi-Query Attention (Shazeer, 2019) and Grouped-Query Attention (Ainslie et al., 2023) which suggest to share key and value across a group of queries. MLKV (Zuhri et al., 2024) further suggest to share keys and values for queries across layers and MLA (Liu et al., 2024) introduces low-rank projection when processing keys and values. Besides, CLA (Brandon et al., 2024) and LISA (Mu et al., 2024) respectively point out that we can reuse keys, values, or the attention matrix across layers to reduce redundancy between layers. While these methods typically process both key and value simultaneously, SVFormer is the first approach to decouple value from query and key during attention computation. Moreover, it is compatible with other methods like GQA.

## 3 Method 3.1 Motivation: Information Transfer Via Cross Layer Attention

Let Hn ∈ R
l×d be the input hidden state of the n-th layer, where l denotes the sequence length and d is the dimension size. In standard attention, the hidden state Hn will be firstly projected into Qn, Kn, Vn ∈ R
l×dthrough three linear projections WQ,WK,WV ∈ R
d×drespectively. For simplicity, we introduce dot-product attention of layer n as

$$\mathrm{Attention}(\mathbf{Q}_{n},\mathbf{K}_{n},\mathbf{V}_{n})=\mathrm{Softmax}({\frac{\mathbf{Q}_{n}\mathbf{K}_{n}^{T}}{\sqrt{d}}})\mathbf{V}_{n}.$$

)Vn. (1)
An ideal way to incorporate previous layers' information is cross layer attention. The attention mechanism naturally extracts relevant information from previous layers. If these layers contain lowquality information, the similarity between the current layer's query and the previous layers' keys
will be low, thus minimizing negative impacts. Given *m < n* and the information (Qm, Km, Vm) of m-th layer, the cross layer mechanism calculates the attention output Un of n-th layer by the
following attention formula:
Un = Softmax Qn Concat(Kn, Km)
$\downarrow$ $\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$
√d
Concat(Vn, Vm). (2)
In practice, cross-layer attention enhances feature fusion by allowing information to flow between layers, capturing both intra-layer and inter-layer dependencies. However, this approach introduces additional computational overhead due to the concatenation of keys and values from multiple layers. For example, in scenarios described by Eqn. 2, the overall computational complexity of the model nearly doubles compared with vanilla attention described in Eqn. 1.
$$(\mathbb{I})$$

## 3.2 Efficient Cross Layer Attention

To solve this problem, we propose to replace the Km with Kn in Eqn. 2, as shown in Eqn. 3.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

$$\mathbf{U}_{n}\approx\text{Softmax}\left(\mathbf{Q}_{n}\,\text{Concat}(\mathbf{K}_{n},\mathbf{K}_{n})^{T}/\sqrt{d}\right)\text{Concat}(\mathbf{V}_{n},\mathbf{V}_{m})$$ $$=\frac{1}{2}\,\text{Softmax}\left(\mathbf{Q}_{n}\mathbf{K}_{n}^{T}/\sqrt{d}\right)(\mathbf{V}_{n}+\mathbf{V}_{m}).$$  Note $\mathbf{U}_{n}$ denotes the $n$-th order of the $n$-th order of $\mathbf{V}_{n}$. This is the $n$-th order of $\mathbf{V}_{n}$.  
Utilizing the concept of block matrices, Eqn. 3 can be further simplified into Eqn. 4. This simplification converts the concatenation operation of the two value matrices into an addition operation. Compared to Eqn. 1, this new method only brings a minimal increase in computational complexity while still leveraging the information from the m-th layer in the n-th layer. Furthermore, Eqn. 4 can be generalized to incorporate cross-layer attention across all preceding n − 1 layers as follows:

$${\bf U}_{n}\approx\frac{1}{n}{\bf A}_{n}\sum_{i=1}^{n}{\bf V}_{i}.\tag{1}$$

where An denotes the original attention matrix for layer n. From the perspective of information propagation, model described by Eqn. 3 projects the historical values into the current layer's embedding space using the current layer's attention as a weight matrix. For example, a naive approach would be to perform identity mapping, as described by

$$\mathbf{U}_{n}=\mathbf{A}_{n}\mathbf{V}_{n}+{\frac{1}{n-1}}\sum_{i=1}^{n-1}\mathbf{V}_{i}.$$
Vi. (6)
To evaluate the approximation effect of replacing the Km with Kn, we randomly select 1,000 pre-training data samples. For each layer of a trained baseline model, assuming cross-layer attention is required for each layer with respect to the previous one, we calculate the cosine similarity between the outputs from Eqn. 2 and Eqn. 4. We also calculate the cosine similarity between the outputs from Eqn. 2 and Eqn. 6 for comparison. Fig. 4 shows that our proposed method provides a good approximation for cross-layer attention.

Figure 4: Average token similarity between layer outputs. Lines show similarity of outputs using current attention (Eqn. 4) or identity attention (Eqn. 6) compared to the one 3.3 TRANSFORMER WITH RESIDUAL VALUE using cross-layer attention (Eqn. 2).

2 3 4 5 6 7 8 Layer Index 0.3 0.4 0.5 0.6 0.7 0.8 0.9

$$({\mathfrak{I}})$$
$$(4)$$
$$(S)$$
$$\left(T\right)$$

Current Attention Identity Mapping

$$(6)$$

Ave rage Cos ine Si milarity
$$\mathbf{U}_{n}=\sum_{i=1}^{n}\alpha_{i}\mathbf{A}_{i}\mathbf{V}_{i}.$$
n
$$({\mathfrak{s}})$$
αiAiVi. (9)
$\eqref{eq:walpha}$. 

## 3.4 A Unified View Of Neutreno And Denseformer

Using our framework, the NeuTRENO can be defined as

$$\mathbf{U}_{n}=\left(\mathbf{A}_{n}-\lambda\mathbf{I}\right)\mathbf{V}_{n}+\lambda\mathbf{V}_{1}.$$

where I denotes the identity matrix and λ is a hyper-parameter. It can be found that the term of λI may have certain negative impact on the learning of original attention. If we ignore the attention output projection and the MLP layer, DenseFormer can also be modeled within our framework as

  ## TransFORMER WITH ResIDUAL VALUE  $$\newcommand{\vecs}[1]{\overset{\rightharpoonup}{\mathbf{#1}}}$$  $$\newcommand{\vecd}[1]{\overset{-\!-\!\rightharpoonup}{\vphantom{a}\smash{#1}}}$$
Based on Eqn. 5, we propose a variant of Transformer with residual value (ResFormer) which only chooses first layer as the target of cross layer attention since the first layer contains all basic information of each token. The analysis of entropy in Fig. 1 (Right) supports this point, indicating that attention tends to be relatively dispersed across different tokens in the initial layers of the model. The attention mechanism of ResFormer can be formulated as
$\mathbf{U}_{n}=\frac{1}{2}\mathbf{A}_{n}(\mathbf{V}_{n}+\mathbf{V}_{1})$.  $\mathbf{u}$ is applied in the first layer. From the training per 
where n ≥ 2 and standard attention is applied in the first layer. From the training perspective, it explicitly learns a residual mapping instead of directly learning the desired underlying mapping and that's why we call it ResFormer.

where {αi}
n i=1 is a set of hyper-parameters. DenseFormer uses attention matrix of previous layer as the weight matrix of projecting values but this is not aligned with the concept shown in Eqn. 3.

## 3.5 Svformer: Single-Layer Value For Half Kv Cache

After ResFormer, a natural idea is whether we can remove the value vectors in each layer and have all layers share the value vectors from the first layer. We call this method SVFormer. Similar to ResFormer, SVFormer still adopts standard attention in the first layer and obtain the attention output Un for n-th layer where n ≥ 2 through

$$\mathbf{U}_{n}=\mathbf{A}_{n}\mathbf{V}_{1}.$$
Un = AnV1. (10)
Compared to previous methods, SVFormer is the first method that decouple value vectors from attention. Its main advantage is that it only requires computing and storing the value vectors for the first layer, saving nearly half of the KV cache during inference. Similar methods like CLA reduce KV cache by sharing both of the key and value vectors every two layers. However, the results in Fig. 5 show that sharing values has less negative impact compared with sharing keys.

Figure 5: Ablation study of sharing different parts of attention every two layers.

## 4 Pretrain Experiments

0 2000 4000 6000 8000 10000 Training Step 0.03 0.02 0.01 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Rela tive Tra ining Los s CLAttention Only Share Keys Only Share Values

## 4.1 Setting 4.1.1 Training Details

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.1.2 Relative Training Loss Curve On Slimpajama

We trained all models for only one epoch on SlimPajama subsets, and primarily use training loss to compare different models. Furthermore, we use the relative training loss curve for better visualizing the difference among different models from the perspective of loss landscape. Specifically, for each method, we will subtract the smoothed training curve of the vanilla Transformer, obtained under the same experimental settings, from the smoothed training curves of the method. The smoothing is done using a window size of 10 steps or 100 steps.

## 4.1.3 Entropy For Analyzing Attention Concentration Effects

Following Brandon et al. (2024), we choose the Llama-like architecture and SlimPajama (Soboleva et al., 2023) data for main experiments. Specifically, the architecture includes pre-normalization, SwiGLU activations (Shazeer, 2020), rotary position embedding (Su et al., 2024), and no dropout. For slimpajama, we randomly sample nearly 20B tokens according to the original data distribution of seven domains during training and adopt tokenizer used for "RedPajama-INCITE-7B-Base". The details of training data can be found in Table 2 in Appendix.

Unless otherwise noted, we train all models using AdamW optimizer with 0.1 weight decay, β1 = 0.9, β2 = 0.95 and the max grad norm 1.0. The batch size is set to be around 2M tokens (Zhang et al., 2024) with a sequence length of 2,048 and the total steps is fixed 10,000 steps (Kaplan et al., 2020). We adopt linear learning rate warmup for the first 1,200 steps with the initial learning rate and the peak learning rate to be 1e-7 and 6e-4 respectively. The cosine decay schedule gradually decays to 10% of the peak learning rate by the end of training (Zhou et al., 2024; Wei et al., 2023). The detailed hyperparameters for models of various sizes and different training sequence lengths in Appendix A.5. Moreover, All models are trained with 8 Nvidia A100 80G GPUs using mixedprecision training in FP16. We adopt deepspeed zero-2 optimizer and flash attention mechanism.

Given the attention matrix A ∈ R
l×lat one layer, we use entropy e to represent its concentration effect. To obtain entropy E, calculate the importance vector a =
1 l Plj=1 Aij firstly where A is a lower triangular matrix. The entropy can be formulated as follows:
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 We further test the variant of ResFormer defined as Un = An(Vn + λV1). As shown in Fig.8, ResFormer can accommodate a wide range of λ values and the performance improves as λ increases, achieving the best results at λ = 2. Regardless of the value of λ, ResFormer consistently outperforms Transformers. It suggests that the success of Res-
Former lies in the use of V1 and the mapping by An. The ablation study of different hyperparameters λ for NeuTRENO,
as defined in Equation 8, can be found in the Appendix A.3.

0 2000 4000 6000 8000 Training Step 0.10 0.08 0.06 0.04 0.02 0.00

Relat ive Tra ining Loss Resformer Lambda 0.2 Lambda 0.5 Lambda 0.8 Lambda 1 Lambda 2 Lambda 5 0 2000 4000 6000 8000 10000 Training Step 0.06 0.05 0.04 0.03 0.02 0.01 0.00 0.01 Averag e Valid Loss Transformer Resformer 10.4% reduction Averag e Valid Loss Transformer Resformer 13.6% reduction Sequence length:

2048 32000 64000 0 2000 4000 6000 8000 10000 Training Step 0.10 0.08 0.06 0.04 0.02 0.00 0.02 0.04 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5

# Parameters 1e8 2.3 2.4 2.5 2.6 2.7 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0

# Training Tokens 1e10 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 Relative Training Lo ss Relative Training Lo ss

$$(11)$$

DenseFormer 82M

NeuTRENO 82M

Resformer 82M

Resformer + NeuTRENO

Figure 6: (Left) Validation loss as model size scales from 82M to 468M parameters. (Right) Validation loss for the 468M parameter model evaluated every 2B tokens. ResFormer achieves approximately 10.4%-13.6% reduction in both model parameters and training data.

Figure 7: (Left) The relative training curve between a 82M ResFormer and Transformer across different training sequence lengths. (Right) Relative training loss of various Transformer variants compared to the vanilla Transformer model, with model size fixed at 82M parameters.

$$e=-\sum_{i=1}^{l}a_{i}^{\prime}\log a_{i}^{\prime}.$$

. (11)
where a
′i i = ai/
Pli=1 ai for i = 1, 2*, . . . , l* and the higher the entropy e, the greater the degree of clustering in a, i.e., attention matrix A is more likely to focus on several specific tokens.

## 4.1.4 Spectral Decomposition For Analyzing Representations

Spectral Decomposition is a classical method to analyze the representations of models. Zhu et al. (2021) suggests that the eigenvectors with larger eigenvalues are more transferable. Here we use spectral decomposition to analyze the feature space of value v of one layer as following:

$${\frac{1}{l}}\sum_{i=1}^{l}\mathbf{v}_{i}\mathbf{v}_{i}^{T}=\sum_{j=1}^{d}\mathbf{w}_{j}\mathbf{\lambda}_{j}\mathbf{w}_{j}^{T}.$$
$$(12)$$
j. (12)
where wj is the j-th eigenvector with eigenvalue λj for j = 1, 2*, . . . , d* and d is the dimensionality of the value's feature space.

## 4.2 Resformer Vs. Vanilla Transformer

We trained ResFormer and vanilla Transformer with different model size on data with different sequence lengths. In Fig. 7 (Left), ResFormer consistently outperforms vanilla Transformer throughout training across different training sequence lengths. Additionally, the results in Fig. 7 (Right) illustrate that ResFormer outperforms DenseFormer and NeuTRENO. Furthermore, integrating Res- Former with NeuTRENO leads to additional performance improvements. We also analyzed how ResFormer and Transformer scale at model size and data size. ResFormer and Transformer are trained on similar experiment setting. On the one hand, we trained model with 82M, 180M, 320M and 468M parameters on 20B training tokens and evaluated them on a separate validation set. As shown in Fig.6 (Left), ResFormer achieves equivalent validation loss to the Transformer while utilizing 10.4% fewer model parameters. On the other hand, we evaluated the 468M models every 2B tokens and ResFormer needs 13.6% fewer training tokens to achieve the same validation loss as Transformer. The validation loss for these models is available in Appendix A.6.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 We compare the different models on several classical reasoning tasks following (Zhang et al., 2024) in a zero-shot way. The tasks include Hellaswag (Zellers et al., 2019), OpenBookQA (Mihaylov et al., 2018), WinoGrande (Sakaguchi et al., 2019), ARC-Easy and ARC-Challenge (Clark et al., 2018) and PIQA (Bisk et al., 2020). The results in Table 1 show that ResFormer achieved an average accuracy improvement of nearly 3% compared to the vanilla Transformer.

| Model       | Max Length   | HellaSwag   | Obqa   | WinoGrande   | ARC-c   | ARC-e   | PIQA   | Avg   |
|-------------|--------------|-------------|--------|--------------|---------|---------|--------|-------|
| Transformer | 2,048        | 0.263       | 0.142  | 0.492        | 0.199   | 0.331   | 0.572  | 0.333 |
| ResFormer   | 2,048        | 0.273       | 0.148  | 0.512        | 0.182   | 0.414   | 0.604  | 0.355 |
| Transformer | 64,000       | 0.267       | 0.142  | 0.485        | 0.179   | 0.322   | 0.570  | 0.328 |
| ResFormer   | 64,000       | 0.274       | 0.136  | 0.513        | 0.184   | 0.407   | 0.588  | 0.350 |

## 4.5 Visualization Of Resformer

To figure out why ResFormer can achieve better performance on language modeling tasks than vanilla Transformer, we conduct visualization based on the eigenvalue decomposition discussed in Section 4.1.4. After sorting the eigenvalues in descending order, we compute the average eigenvalue for each layer across 1,000 randomly sampled pre-train data examples. The results in Fig. 12 indicate that the value states generated by most layers of the ResFormer exhibit stronger representational capacity compared to those of the vanilla Transformer. We also analyze the attention concentration effects mentioned in Section 4.1.3 using the same batch of test data. Fig. 1 (Right) illustrates that the clustering effect of attention increases significantly

0 2000 4000 6000 8000 10000 Training Step 0.05 0.04 0.03 0.02 0.01 0.00 0.01 0 2000 4000 6000 8000 Training Step 0.1 0.0 0.1 0.2 0.3 0.4 0.5 0 2000 4000 6000 8000 10000 Training Step 0.10 0.08 0.06 0.04 0.02 0.00 0.02 0.04 0.06 Relativ e Trai ning L
oss Relativ e Trai ning Loss ResFormer Cross Layer Attention Identity Mapping Relat ive Tr ainin g Los s Add residual of:
Value Key Query Value shortcut from:

Only First layer One Adjacent layer All Previous layer
In Eqn. 4, we employ residual connections for the values. We compare this approach with models that add residual connections to queries or keys. The results, shown in Fig. 9, indicate that only residual connections for values yield positive effects. One possible explanation is that attention mechanisms are sensitive to perturbations, and modifying queries or keys significantly impacts it. Moreover, we compare with the models based on Eqn. 2 and Eqn. 6. The results in Fig. 10 align with Fig. 4, showing that identity mapping causes significant perturbations, leading to poor performance. Interestingly, ResFormer achieves an even lower final loss than ResFormer. It suggests that ResFormer's impact on the attention optimization is better by mitigating value-state drains. When determining the mapping method and target value, it is crucial to consider which historical layers' values should be included in the residual connection. Fig. 11 shows that each Transformer layer should add a shortcut to the first layer's value rather than to the nearest preceding layer or all previous layers, highlighting the first-layer value's critical importance. A potential explanation is that incorporating values from other layers may dilute the impact of the first-layer value.

## 4.4 Downstream Evaluations 4.3 Ablation Study Of Residual Connection

Resformer 468M

Transformer 468M

Resformer 180M

Transformer 180M

Resformer 82M

Transformer 82M

0 200 400 600 800 1000 Index of eigenvectors 0 25 50 75 100 125 150 175 0 5 10 15 20 Index of eigenvectors 100 200 300 400 500 600 Eige nvalu e Eige nvalu e Resformer 468M Transformer 468M Resformer 180M Transformer 180M
0 2000 4000 6000 8000 Training Step 0.08 0.06 0.04 0.02 0.00 0.02 0.04 0.06 SVFormer (2048)
GQA2 (2048)

CLA2 (2048)

SVFormer (64000)

GQA2 (64000)

CLA2 (64000)

0 2000 4000 6000 8000 Training Step 0.08 0.06 0.04 0.02 0.00 0.02 0.04 0.06 0.08 0 2000 4000 6000 8000 10000 Training Step 0.08 0.06 0.04 0.02 0.00 0.02 0.04 0.06 Training sequence length:

512 2048 8192 32000 64000 0 10000 20000 30000 40000 50000 60000 Training Sequence Length 5000 10000 15000 20000 25000 30000 Relative Tra ining Los s Relative Tra ining Los s Relative Tra ining Lo ss Critica l Point Step SVFormer+GQA4 (64000)

GQA8 (64000)

CLA2+GQA4 (64000)

Figure 13: The relative training loss for SV- Former and other KV efficient model compared with vanilla attention. The numbers in parentheses represent the training sequence length. Left:
Model with nearly 1/2 KV cache. Right: Model with nearly 1/8 KV cache.

Figure 14: Left: The relative training loss for SV- Former under different sequence lengths with a fixed batch size of 2M tokens. Right: Analysis of critical point, and we predict it for length 64,000 using linear regression with the last 1,000 data points.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 with the number of layers for the vanilla Transformer, whereas the clustering effect is relatively less pronounced for the ResFormer. We further visualize the attention weights, value-state norms
∥v∥2, and hidden-state norms ∥h∥2 of tokens at different layers and positions, with detailed results in Appendix A.2. Given that attention clustering often occurs on the first token, we primarily show its results in Fig. 2. The results indicate that using ResFormer significantly mitigates attention sinks
(Xiao et al., 2024), value-state drains (Guo et al., 2024b) and residual-state peaks (Sun et al., 2024). Guo et al. (2024a) attributes these phenomena to the mutual reinforcement mechanism of model and we suggest that the value shortcut disrupts this mechanism by alleviating value-state drains. Specifically, for tokens lacking semantic information like start tokens, a large value state magnitude can adversely affect the prediction of subsequent tokens if they are overly attended to. When there is no value-state drains, models will reduce attention clustering to these tokens to minimize loss.

## 4.6 Svformer Vs. Gqa 4.7 Other Factors Influencing Svformer

Intuitively, the training effectiveness of SVFormer is influenced by factors such as the maximum learning rate, warmup steps, model size, and other factors beyond just the training sequence length. We conducted experiments to explore these relationships. In the Fig. 13, at a training sequence length of 64,000, SVFormer demonstrates lower final loss compared to existing KV -efficient methods such as CLA and GQA. Moreover, it can be used concurrently with GQA to enhance KV efficiency further. However, we observed that with a training sequence length of 2,048, SVFormer underperforms compared to GQA. The results indicate that sequence length significantly affects SVFormer's performance. Thus, we conducted more comprehensive experiments on sequence length. Results in Fig. 14 (Left) demonstrate that SVFormer will always be gradually surpassed by vanilla attention during training while its training speed is faster than vanilla Transformer at the early stage. However, as the training sequence length increases, the SVFormer model performs better. In this way, we focus on the critical point, defined as the number of training steps exceeded. Fig. 14 (Right) illustrates that the relationship between the critical point and sequence length exhibits an exponential trend. We argue that it's due to the challenge deep models face in fully optimizing the increasingly larger first-layer value matrix as the training sequence length grows.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## Ethics Statement

On the one hand, the data employed in this paper is sourced from publicly available datasets provided by the company, which have undergone a certain level of filtering. On the other hand, the models trained in our study are solely utilized for experimental analysis and will not be publicly deployed.

0 2000 4000 6000 8000 10000 Training Step 0.4 0.3 0.2 0.1 0.0 0.1 0 2000 4000 6000 8000 10000 Training Step 0.20 0.15 0.10 0.05 0.00 0.05 Transformer SVFormer 12.2% increase 0 2000 4000 6000 8000 10000 Training Step 0.4 0.3 0.2 0.1 0.0 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5
# Parameters 1e8 2.3 2.4 2.5 2.6 2.7 Relative Tra ining Lo ss Relativ e Training Lo ss Relativ e Training Lo ss Average Va lid Loss Peak learning rate:

1e-4 3e-4 6e-4 Warmup steps:
1200 steps 120 steps Architecture:
Llama 82M GPT2 78M
(a) Learning Rate.

(b) Warmup Steps.

(c) Architecture.

(d) Model Size.
Figure 15: The relative training loss for SVFormer under different hyper-parameter setting and the validation loss as model size scales from 82M to 468M parameters. Based on the results shown in Fig. 15a and Fig. 15b, a smaller learning rate benefits SVFormer more, with warmup's impact being comparatively small. This could be attributed to the model's outcomes being closely tied to the total summed learning rate, which has weak connection with warmup steps (Kaplan et al., 2020). Moreover, larger models often require smaller learning rates to ensure training stability, making them more suitable for using SVFormer. Llama-like models and GPT2-like models exhibit similar critical points and final losses (see Fig. 15c). This suggests that the difference between SVFormer and the vanilla Transformer is not sensitive to architecture. Compared with Transformer, SVFormer requires a 12.2% increase in parameters to achieve the same validation loss while reducing the KV -cache by nearly half.

0 2000 4000 6000 8000 10000 Training Step 0.2 0.1 0.0 0.1 0.2 0.3 0.4 0.5 0 2000 4000 6000 8000 10000 Training Step 0.10 0.08 0.06 0.04 0.02 0.00 0.02 0.04 Relati ve Tra ining Loss Relativ e Trai ning Loss Shared value from:

First Layer First Layer + Last Layer First Layer + One Middle Layer + Last Layer First Layer + Two Middle Layer + Last Layer Components to share:

Value Query Key
To better understand SVFormer, we conduct several ablation experiments. We first observe the effects of sharing the first layer's queries or keys across all layers in Fig. 16, finding that this significantly impacts model performance, similar to the results in Fig. 5. Additionally, sharing the first layer's values in a multi-layer network may reduce the network's "effective depth." By updating the shared values using intermediate layers as "anchors," we find that increasing the number of "anchors" improves performance, as shown in Fig. 17.

## 5 Conclusion

In this paper, we propose the concept of attention concentration, a problem that arises from stacking multiple attention layers. From the perspective of cross-layer attention, we derive ResFormer, which adds a residual connection between the value vectors of the current layer and those of the first layer before the attention operation to alleviate attention concentration. Additionally, we introduce SVFormer, based on ResFormer, which reduces the KV cache by nearly half. We conducted comprehensive experiments on the language modeling task to validate the advantages of these two Transformer variants in different scenarios.

## Reproducibility Statement References

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 We have detailed the complete experiments setup such as batch size, optimizer, learning rates in Section 4.1.1. Besides, we will release source codes once our paper is made public. These resources should be sufficient to reproduce results of the paper. Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit ´
Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In *International* Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In *AAAI*, pp. 7432–7439. AAAI Press, 2020.

William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan Ragan Kelly. Reducing transformer key-value cache size with cross-layer attention. arXiv preprint arXiv:2405.12981, 2024.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In *NeurIPS*, 2020.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR, abs/1803.05457, 2018.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In *ACL (1)*, pp. 2978–2988. Association for Computational Linguistics, 2019.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need ´
registers. In *International Conference on Learning Representations*. OpenReview.net, 2024.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In *NAACL-HLT (1)*, pp. 4171–4186. Association for Computational Linguistics, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*. OpenReview.net, 2021.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.

arXiv preprint arXiv:2407.21783, 2024.

Muhammad ElNokrashy, Badr AlKhamissi, and Mona Diab. Depth-wise attention (dwatt): A layer fusion method for data-efficient classification. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 4665–4674, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´
state spaces. *arXiv preprint arXiv:2111.00396*, 2021.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Tianyu Guo, Druv Pai, Yu Bai, Jiantao Jiao, Michael I Jordan, and Song Mei. Active-dormant attention heads: Mechanistically demystifying extreme-token phenomena in llms. arXiv preprint arXiv:2410.13835, 2024a.

Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. Attention score is not all you need for token importance indicator in kv cache reduction: Value also matters. *arXiv preprint arXiv:2406.12335*, 2024b.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. *arXiv preprint arXiv:2401.18079*, 2024.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 646–661. Springer, 2016.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern* recognition, pp. 4700–4708, 2017.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas ´ Wang, Timothee Lacroix, and William El Sayed. Mistral 7b. ´ *CoRR*, abs/2310.06825, 2023.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations. OpenReview.net, 2020.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. *arXiv preprint arXiv:2404.14469*, 2024.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixtureof-experts language model. *arXiv preprint arXiv:2405.04434*, 2024.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018.

Yongyu Mu, Yuzhang Wu, Yuchun Fan, Chenglong Wang, Hengyu Li, Qiaozhi He, Murun Yang, Tong Xiao, and Jingbo Zhu. Cross-layer attention sharing for large language models. arXiv preprint arXiv:2408.01890, 2024.

Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention. *arXiv preprint arXiv:2404.07143*, 2024.

Tam Nguyen, Tan Nguyen, and Richard Baraniuk. Mitigating over-smoothing in transformers via regularized nonlocal functionals. *Advances in Neural Information Processing Systems*, 36:80233– 80256, 2023.

Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, and Martin Jaggi. Denseformer:
Enhancing information flow in transformers via depth weighted averaging. *arXiv preprint* arXiv:2402.02622, 2024.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era. *arXiv preprint arXiv:2305.13048*, 2023.

Jackson Petty, Sjoerd Steenkiste, Ishita Dasgupta, Fei Sha, Dan Garrette, and Tal Linzen. The impact of depth on compositional generalization in transformer language models. In *Proceedings* of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 7232–7245, 2024.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale, 2019.

Noam Shazeer. Fast transformer decoding: One write-head is all you need. *CoRR*, abs/1911.02150, 2019.

Noam Shazeer. GLU variants improve transformer. *CoRR*, abs/2002.05202, 2020. Han Shi, Jiahui Gao, Hang Xu, Xiaodan Liang, Zhenguo Li, Lingpeng Kong, Stephen Lee, and James T Kwok. Revisiting over-smoothing in bert from the perspective of graph. arXiv preprint arXiv:2202.08625, 2022.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey.

SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023. URL https: //huggingface.co/datasets/cerebras/SlimPajama-627B.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. *arXiv preprint arXiv:2402.17762*, 2024.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need.(nips), 2017. arXiv preprint arXiv:1706.03762, 10:S0140525X16001837, 2017.

Peihao Wang, Wenqing Zheng, Tianlong Chen, and Zhangyang Wang. Anti-oversmoothing in deep vision transformers via the fourier domain analysis: From theory to practice. arXiv preprint arXiv:2203.05962, 2022.

Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei Lu, Rui Hu, et al. Skywork: A more open bilingual foundation model. ¨ arXiv preprint arXiv:2310.19341, 2023.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In *International Conference on Learning Representations*. OpenReview.net, 2024.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. *Advances in Neural Information Processing Systems*, 36, 2024.

Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang, Xudong Lu, Aojun Zhou, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Think: Thinner key cache by query-driven pruning. arXiv preprint arXiv:2407.21018, 2024.

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755

## A Appendix A.1 Token Similarity Analysis A.2 Attention Concentration Visualization

We visualize the token importance, norms of value states and norms of hidden states for tokens at different position across layers. The results are averaged from 1,000 different sequences so that only Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In *ACL (1)*, pp. 4791–4800. Association for Computational Linguistics, 2019.

Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh Susskind. An attention free transformer. *arXiv preprint arXiv:2105.14103*, 2021.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. *arXiv preprint arXiv:2401.02385*, 2024.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36, 2024.

Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Zihang Jiang, Qibin Hou, and Jiashi Feng. Deepvit: Towards deeper vision transformer. *arXiv preprint arXiv:2103.11886*, 2021.

Fei Zhu, Zhen Cheng, Xu-Yao Zhang, and Cheng-lin Liu. Class-incremental learning via dual augmentation. *Advances in Neural Information Processing Systems*, 34:14306–14318, 2021.

Zayd Muhammad Kawakibi Zuhri, Muhammad Farid Adilazuarda, Ayu Purwarianti, and Alham Fikri Aji. Mlkv: Multi-layer key-value heads for memory efficient transformer decoding. arXiv preprint arXiv:2406.09297, 2024.

Attention concentration tends to make embeddings of different tokens more similar, resulting in over-smoothing. The extent of over-smoothing can be assessed by calculating the average token similarity s of the hidden states using the following formula:

$$\mathbf{s}=\frac{1}{l(l-1)}\sum\nolimits_{i=1}^{l}\sum\nolimits_{j=1,j\neq i}^{l}\text{Sim}\left(\mathbf{h}_{i},\mathbf{h}_{j}\right).\tag{1}$$

where {hi}
l i=1 is the hidden state of the i-th token and Sim(·) denotes the operation of cosine similarity. The results in Fig. 18 are align with the results in Fig. 1. In the case of Llama and Mistral, the average token similarity demonstrates an "M"-shaped pattern with increasing network depth, while entropy follows a "W"-shaped pattern at corresponding positions. These trends indicate a strong correlation between attention concentration and over-smoothing.

0 5 10 15 20 25 30 Layer Index 0.05 0.10 0.15 0.20 0.25 0.30 Ave rag e Toke n Si mila rity

$$(13)$$

Llama 8B V3.1 Llama 8B Instruct V3.1 Mistral 7B V0.2 Mistral 7B Instruct V0.2 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 0.1 0.2 0.3 0.4 0.5 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 500 1000 1500 2000 2500 5 5 5 10 10 10 20 Layers 20 Layers 20 Layers Transformer 15 15 15 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 200 400 600 800 1000 1200 1400 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 5 5 5 10 10 10 20 Layers 20 Layers 20 Layers NeuTRENO
15 15 15 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 0.7 0.8 0.9 1.0 1.1 1.2 1.3 50 100 150 200 250 300 0.01 0.02 0.03 0.04 0.05 0.06 0.07 5 5 5 10 10 10 20 Layers 20 Layers 20 Layers ResFormer 15 15 15 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 50 100 150 200 250 300 350 400 0.01 0.02 0.03 0.04 0.05 1.82 1.83 1.84 1.85 1.86 1.87 1.88 5 5 5 10 10 10 20 Layers 20 Layers 20 Layers 15 15 15 SVFormer 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index 10 20 30 40 50 60 70 80 90 Token Index
(a) Token importance.

(b) Value-state norms.

(c) Hidden-state drains.
810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863

1 2 3 4 5 6 7 8 Layer Index 2.75 3.00 3.25 3.50 3.75 4.00 4.25 4.50 Token Import ance Transformer NeuTRENO
Resformer SVformer Cross Layer Attention Hidden
-state N
orm Transformer NeuTRENO Resformer SVformer Cross Layer Attention 1 2 3 4 5 6 7 8 Layer Index 0.2 0.4 0.6 0.8 1.0 1.2 1 2 3 4 5 6 7 8 Layer Index 0.0 0.1 0.2 0.3 0.4 0.5 1 2 3 4 5 6 7 8 9 Layer Index 0 50 100 150 200 250 Value-s tate N
orm Average Ent ropy Transformer NeuTRENO

Resformer SVFormer Cross Layer Attention Transformer NeuTRENO Resformer SVformer Cross Layer Attention
(a) Entropy.

(b) Token importance.

Token I
mportanc e Top-1 Top-2 Top-3 Top-10 Top-1%

Token I
mportanc e Top-1 Top-2 Top-3 Top-10 Top-1%

0 5 10 15 20 25 Layer Index 0.00 0.02 0.04 0.06 0.08 0.10 Token I
mportanc e Top-1 Top-2 Top-3 Top-10 Top-1%

0 5 10 15 20 25 Layer Index 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 0.200 Token Importance Top-1 Top-2 Top-3 Top-10 Top-1%
0 5 10 15 20 25 Layer Index 0.00 0.05 0.10 0.15 0.20 0.25 0 5 10 15 20 25 Layer Index 0.00 0.02 0.04 0.06 0.08 0.10

(a) Transformer.

(b) NeuTRENO.

(c) ResFormer.

(d) SVFormer.
the start token is the same and special across all sequences. Fig. 19 (First column) demonstrates that the start token easily attracts massive attention despite lacking semantic information for Transformer and NeuTRENO. For ResFormer, the importance of the start token is less than 10 times that of tokens at other positions, indicating that tokens carrying semantic information receive more attention. Moreover, both Transformer and NeuTRENO exhibit significant value-state drains (Guo et al., 2024b) and residual-state peaks (Guo et al., 2024a; Sun et al., 2024) on the start token at certain layers. In contrast, for ResFormer, the value state norm of the start token exceeds half the magnitude of other tokens, while the peak hidden state norm is less than triple the average. Fig.21 further illustrates the distribution of token importance, where TOP-i represents the i-th largest token importance within a sequence. Compared to Transformer and NeuTRENO, ResFormer and SVFormer exhibit a more uniform distribution of token importance. Similar to Fig.2, we conducted experiments on 82M models, with results shown in Fig.20. We also illustrate the attention pattern of cross-layer attention introduced in Eqn. 2. The results demonstrate that while cross-layer attention successfully mitigates the problem of attention concentration, it still exhibits value-state drains.

## A.3 Ablation Study Of Neutreno

NeuTRENO is sensitive to the choice of hyperparameter λ which is task-dependent. In the appendix of Nguyen et al. (2023), it is reported that λ is set to 0.6 for image classification and segmentation tasks, and 0.4 for language modeling tasks. Fig. 22 indicates that λ = 0.4 achieves the best results in our training dataset so that we choose λ = 0.4 for comparison. Besides, we empirically choose λ = 0.2 for NeuTRENO when combined with ResFormer.

## A.4 Pre-Train Dataset

Based on the equation D ≥ 5000 · N0.74 (Kaplan et al., 2020) where D is data size and N is the number of non-embedding parameters, we need to collect at least 17.5B for model has N = 700M
non-embedding parameters (corresponding to complete 1B model with 2,048 hidden size, 50,277 vocab size and 2,048 sequence length) to avoid over-fitting. Besides, Xie et al. (2024) indicates that the mixture proportions of pre-training data domains significantly affects the training results. In this way, we sampled 20B tokens data from original 627B data based on the original data proportions shown in the Table 2.

864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917

0 2000 4000 6000 8000 Training Step 0.04 0.03 0.02 0.01 Relat ive Tra inin g Lo ss NeuTRENO(Lambda 0.2)

NeuTRENO(Lambda 0.4)
NeuTRENO(Lambda 0.6)

NeuTRENO(Lambda 0.8)

| Data source   | proportions   | Tokens   |
|---------------|---------------|----------|
| Commoncrawl   | 50%           | 10 B     |
| C4            | 20%           | 4 B      |
| GitHub        | 10%           | 2 B      |
| Books         | 5%            | 1 B      |
| ArXiv         | 5%            | 1 B      |
| Wikpedia      | 5%            | 1 B      |
| StackExchange | 5%            | 1 B      |

## A.5 Training Details

| Max Sequence Length        | 512   | 2,048   | 8,192   | 32,000   | 64,000   |
|----------------------------|-------|---------|---------|----------|----------|
| Total Batch Size           | 4,096 | 1,024   | 256     | 64       | 32       |
| Per-GPU Batch Size         | 128   | 32      | 8       | 2        | 1        |
| Gradient Accumulation Step | 32    |         |         |          |          |
| GPUs                       | 8     |         |         |          |          |

Table 3: Training details for training dataset with different sequence length.

Section 4.1.1 introduces the main experimental hyperparameters used in the paper. This section further details the training parameters for various model sizes and training sequence lengths. Table 4 demonstrates the differences among models of various sizes. The configurations for the number of layers, attention heads, hidden dimensions, and FFN dimensions are based on Biderman et al. (2023). Additionally, the λ in Eqn. 8 is set to be 0.4 for NeuTRENO. Moreover, as reported in Table 3, the batch size that a single GPU can accommodate varies depending on the length of the training sequences. Note that the total number of tokens in each batch is consistently 2 million.

## A.6 Validation Loss On Slimpajama

Section 4.1.2 introduces to use relative training loss as a main evaluation matrix. Table 5 reports the validation loss for differnt model on the whole validation split of slimpajama.

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971

| Model              | Common   | Stack   |          |           |        |        |        |        |
|--------------------|----------|---------|----------|-----------|--------|--------|--------|--------|
| Crawl              | C4       | Github  | Exchange | Wikipedia | Book   | Arxiv  | Avg.   |        |
| Transformer (82M)  | 3.3595   | 3.5388  | 1.4247   | 2.3872    | 2.9047 | 3.3797 | 2.1779 | 2.7389 |
| Transformer (180M) | 3.0961   | 3.2834  | 1.2451   | 2.1651    | 2.5897 | 3.1309 | 2.0001 | 2.5015 |
| Transformer (468M) | 2.8514   | 3.0430  | 1.0908   | 1.9628    | 2.2821 | 2.8979 | 1.8362 | 2.2806 |
| ResFormer (82M)    | 3.3362   | 3.5191  | 1.3941   | 2.3592    | 2.8646 | 3.3572 | 2.1518 | 2.7117 |
| ResFormer (180M)   | 3.0631   | 3.2504  | 1.2200   | 2.1350    | 2.5435 | 3.0994 | 1.9732 | 2.4692 |
| ResFormer (468M)   | 2.8214   | 3.0115  | 1.0730   | 1.9388    | 2.2477 | 2.8696 | 1.8142 | 2.2537 |

Table 4: Training details for models with different size.

Table 5: Validation loss on slimpajama.

| Model Size                                | 2M                | 82M   | 180M   | 468M   |
|-------------------------------------------|-------------------|-------|--------|--------|
| Layers                                    | 4                 | 8     | 12     | 24     |
| Attention Heads                           | 2                 | 8     | 12     | 16     |
| Hidden Dimension                          | 16                | 512   | 768    | 1,024  |
| FFN Dimension                             | 56                | 1,792 | 2,688  | 3,584  |
| Tie Word Embedding                        | False             |       |        |        |
| (Peak Learning Rate, Final Learning Rate) | (6e − 4, 6e − 5)  |       |        |        |
| Learning Rate Schedule                    | Cosine Decay      |       |        |        |
| Vocabulary Size                           | 50,277            |       |        |        |
| Activation Function                       | SwiGLU            |       |        |        |
| Position Embedding                        | RoPE (θ = 10,000) |       |        |        |
| Batch Size                                | 2M tokens         |       |        |        |
| Data Size                                 | 20B tokens        |       |        |        |
| (Warmup Steps, Training Steps)            | (120, 10,000)     |       |        |        |
| Adam β                                    | (0.9, 0.95)       |       |        |        |
| Dropout                                   | 0.0               |       |        |        |
| Weight Decay                              | 0.1               |       |        |        |