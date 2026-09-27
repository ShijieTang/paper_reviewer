# Tdformer: A Top-Down Attention-Controlled Spiking Transformer

Anonymous Author(s)
Affiliation Address email

## Abstract 19 **1 Introduction**

1 Traditional spiking neural networks (SNNs) can be viewed as a combination of 2 multiple subnetworks with each running for one time step, where the parameters 3 are shared, and the membrane potential serves as the only information link between 4 them. However, the implicit nature of the membrane potential limits its ability 5 to effectively represent temporal information. As a result, each time step cannot 6 fully leverage information from previous time steps, seriously limiting the model's 7 performance. Inspired by the top-down mechanism in the brain, we introduce 8 TDFormer, a novel model with a top-down feedback structure that functions hi9 erarchically and leverages high-order representations from earlier time steps to 10 modulate the processing of low-order information at later stages. The feedback 11 structure plays a role from two perspectives: 1) During forward propagation, our 12 model increases the mutual information across time steps, indicating that richer 13 temporal information is being transmitted and integrated in different time steps. 2) 14 During backward propagation, we theoretically prove that the feedback structure 15 alleviates the problem of vanishing gradients along the time dimension. We find 16 that these mechanisms together significantly and consistently improve the model 17 performance on multiple datasets. In particular, our model achieves state-of-the-art 18 performance on ImageNet with an accuracy of 86.83%. 20 Spiking Neural Networks (SNNs) are more energy-efficient and biologically plausible than traditional 21 artificial neural networks (ANNs) [1]. Transformer-based SNNs combine the architectural advantages 22 of Transformers with the energy efficiency of SNNs, resulting in a powerful and efficient models 23 that have attracted increasing research interest in recent years [2, 3, 4, 5, 6]. However, there is 24 still a big performance gap between existing SNNs and ANNs. This is because SNNs represent 25 information using binary spike activations, whereas ANNs use floating-point numbers, resulting in 26 reduced representational capacity and degraded performance. Moreover, the non-differentiability of 27 spikes hinders effective training with gradient-based methods. 28 In traditional SNNs, a common approach to increase representational capacity is to expand the 29 time step T. However, SNNs trained with direct coding and standard learning methods [7] lack 30 structural mechanisms for temporal adaptation. Temporal information is solely conveyed through 31 membrane potential dynamics, while the network architecture, parameters, and inputs remain fixed 32 across time steps. This reliance on membrane dynamics imposes two fundamental limitations. First, 33 temporal information can only be expressed when spikes are fired, yet firing rates are typically low 34 across layers, restricting the bandwidth of information flow. Moreover, the cumulative nature of 35 membrane potentials leads to loss of temporal detail, as earlier spike patterns are summed. Second, 36 temporal gradients must propagate solely through membrane potentials, which can result in vanishing 49 While traditional SNNs rely on bottom-up signal propagation, top-down mechanisms are prevalent in 50 the brain, especially between the prefrontal and visual cortices [14, 15, 16, 17], as shown in Figure 2. 51 These mechanisms are fundamental to how the brain incrementally acquires visual information over 52 time, with higher-level cognitive processes guiding the extraction of lower-level sensory features, 53 and prior knowledge informing the interpretation and refinement of new sensory input. Inspired 54 by top-down mechanisms, we introduce TDFormer, a Transformer-based SNN architecture that 55 incorporates a top-down feedback structure to improve temporal information utilization. Our main 56 contributions can be summarized as follows:

## 65 **2 Related Works** 66 **2.1 Transformer-Based Snns**

67 Spikformer [2] presented the first Transformer architecture based on SNNs, laying the groundwork for 68 spike-based self-attention mechanisms. Spike-driven TransformerV1 [5] introduced a spike-driven 37 gradients[8, 9]. We further confirm these limitations through temporal correlation analysis shown 38 in Figure 1, which demonstrates the limited representational capacity of membrane potentials, and 39 theoretical derivation in appendix B.3. 40 Previous work has been done to enhance the ability of SNNs to represent temporal information, e.g., 41 by initializing the membrane potential and altering the surrogate gradients and dynamics equations 42 [10, 11, 12]. Furthermore, some approaches have incorporated the dimension of time into attention 43 mechanisms, resulting in time complexity that scales linearly with the number of simulation time steps 44 [13]. However, structural mechanisms to facilitate information flow across multiple time steps remain 45 largely unexplored. We argue that adding connections between different time steps has the following 46 two benefits: First, in forward propagation, such connections help the model better leverage features 47 from previous time steps. Second, in backpropagation, structural connections support gradient flow 48 and help mitigate vanishing gradients caused by the membrane potential dynamics. 57 - We identify structural limitations in traditional SNNs, showing that features across time steps 58 exhibit weak mutual information, indicating insufficient temporal integration and utilization. 59 - We propose TDFormer, a Transformer-based SNN with a novel top-down feedback structure. 60 We show that the proposed structure improves temporal information utilization, and provide 61 theoretical analysis showing it mitigates vanishing gradients along the temporal dimension. 62 - We demonstrate state-of-the-art performance across multiple benchmarks with minimal 63 energy overhead, achieving ANN-level accuracy on ImageNet while preserving the efficiency 64 of SNNs.

## 77 **2.2 Models With Top-Down Mechanisms** 94 **3 Preliminaries** 95 **3.1 The Spiking Neuron**

96 The fundamental distinction between SNNs and ANNs lies in their neuronal activation mechanisms. 97 Drawing on established research [2, 4, 5, 3], we select the Leaky Integrate-and-Fire (LIF) [25] neuron 98 model as our primary spike activation unit. LIF neuron dynamics can be formulated by:
V [t] = H[t](1 − S[t]) + VresetS[t], (1)

## 102 **3.2 Spike-Based Self-Attention Mechanisms**

84 Many works have explored top-down attention mechanisms to improve model performance in 85 traditional ANNs. For example, Zheng et al. [21] proposed FBTP-NN, which integrates bottom-up 86 and top-down pathways to enhance visual object recognition, where top-down expectations modulate 87 neuron activity in lower layers [21]. Similarly, Anderson et al. introduced a model combining bottom88 up and top-down attention for image captioning and visual question answering, where top-down 89 attention weights features based on task context [22]. Shi et al. introduced a top-down mechanism 90 for Visual Question Answering (VQA), where high-level cognitive hypotheses influence the focus 91 on relevant scene parts [23]. Finally, Abel and Ullman proposed a network that combines back92 propagation with top-down attention to adjust gradient distribution and focus on important features 93 [24].

$$V[t]=H[t](1-S[t])+V_{\mathrm{reset}}S[t],$$
$$(1)$$
$$\begin{array}{l l}{{\tau[t]}}&{{=H[t](t-U[t])+V_{\mathrm{rest}}[t],}}\\ {{}}&{{}}\\ {{H[t]=V[t-1]+\frac{1}{\tau}(X[t]-(V[t-1]-V_{\mathrm{rest}})),}}\\ {{}}&{{S[t]=\Theta(H[t]-V_{\mathrm{th}}),}}\end{array}$$
$$(2)$$
(X[t] − (V [t − 1] − Vreset)), (2)
where Vreset 99 is the reset potential. When a spike is generated, S[t] = 1, the membrane potential V [t]
is reset to Vreset 100 ; otherwise, it remains at the hidden membrane potential H[t]. Moreover, τ represents 101 the membrane time constant, and the input current X[t] is decay-integrated into H[t].

103 A critical challenge in designing spike-based self-attention is eliminating floating-point matrix 104 multiplication in Vanilla Self-Attention (VSA) [26], which is crucial for utilizing the additive 105 processing characteristics of SNNs.
106 **Spiking Self-Attention** (SSA) Zhou et al. [2] first leveraged spike dynamics to replace the softmax
107 operation in VSA, thereby avoiding costly exponential and division calculations, and reducing energy 108 consumption. The process of SSA is as follows:
$I_{s}=\mathcal{SN}(BN(XW_{I})),I\in\{Q,K,V\},$  $\mathrm{SSA}(Q_{s},K_{s},V_{s})=\mathcal{SN}(Q_{s}K_{s}^{\top}V_{s}*s),$
where W ∈ R
T ×N×D 109 denotes a learnable weight matrix, Is represents the spiking representations of
110 query Qs, key Ks, and value Vs. Here, SN (·) denotes the LIF neuron, and s is a scaling factor.
(4)  $\binom{5}{4}$  . 

69 mechanism to effectively process discrete-time spike signals and employed stacked transformer layers 70 to capture complex spatiotemporal features. Built on [5], Spike-driven TransformerV2 [6] enhanced 71 the spike-driven mechanism and added dynamic weight adjustment to improve adaptability and 72 accuracy in processing spike data. SpikformerV2 [18] was specifically optimized for high-resolution 73 image recognition tasks, incorporating an improved spike encoding method and a multi-layer self74 attention mechanism. SpikeGPT [19] proposed an innovative combination of generative pre-trained 75 Transformers with SNNs. SGLFormer [20] enhanced feature representations by effectively capturing 76 both global context and local details. 78 Unlike bottom-up processes that are driven by sensory stimuli, top-down attention is governed 79 by higher cognitive processes such as goals, previous experience, or prior knowledge[21]. This 80 mechanism progressively acquires information by guiding the focus of attention to specific regions 81 or features of the visual scene. It can be seen as a feedback loop where higher-level areas provide 82 signals that modulate the processing of lower-level sensory inputs, ensuring that the most relevant 83 information is prioritized.

111 **Spike-Driven Self-Attention** (SDSA) Yao et al. [5, 6] improved the SSA mechanism by replacing 112 the matrix multiplication with the Hadamard product and computing the attention via column-wise 113 summation, effectively utilizing the additive properties of SNNs. The first version of SDSA [5] is as 114 follows:
SDSA1(Qs, Ks, Vs) = Qs *⊗ SN* (SUMc(Ks ⊗ Vs)), (6)
115 where ⊗ denotes the Hadamard product, SUMc(·) represents the column-wise summation. Further116 more, the second version of SDSA [6] is described as follows:
SDSA2(Qs, Ks, Vs) = SN s((QsK⊤
s)Vs), (7)
117 where SN s denotes a spiking neuron with a threshold of s · Vth. **Q-K Attention** (QKA) The work 118 in [3] reduces the computational complexity from quadratic to linear by utilizing only the query 119 and key. QKA can be further divided into two variants: Q-K Token Attention (QKTA) and Q-K 120 Channel Attention (QKCA). The formulations for QKTA and QKCA are provided in Equations 8 121 and 9, respectively:

$$\mathrm{QKTA}(Q_{s},K_{s})=\mathcal{SN}(\sum_{i=0}^{D}Q_{s}(i,j))\otimes K_{s},$$  $$\mathrm{QKCA}(Q_{s},K_{s})=\mathcal{SN}(\sum_{j=0}^{N}Q_{s}(i,j))\otimes K_{s},$$
$$(8)$$
$$({\mathfrak{g}})$$

122 where N denotes the token number, D represents the channel number.

## 123 **4 Method**

124 In this section, we introduce TDFormer, a Transformer-based SNN model featuring a top-down 125 feedback structure. We describe its architecture, including the division into sub-networks for feed126 back processing. We theoretically show that the attention module prior to the LIF neuron in the 127 feedback pathway exhibits lower variance compared to SSA and QKTA, and we provide guidance 128 for hyperparameter selection. Finally, we introduce the training loss and inference process. Detailed 129 mathematical derivations are provided in appendix B.

## 130 **4.1 Tdformer Architecture**

131 This work is based on three backbones: SpikformerV1 [2], Spike-driven TransformerV1 [5] and 132 QKformer [3]. These can be summarized into a unified structure, as shown in Figure 2, which consists 133 of Lc Conv-based SNN blocks, Lt Transformer-based SNN blocks, and a classification head (CH).

134 Additionally, the Transformer-based SNN blocks incorporate spike-based self-attention modules and 135 Multi-Layer Perceptron (MLP) modules. 136 Apart from the backbone structure, the TDFormer architecture specifically introduces a top-down 137 pathway called TDAC that includes two modules: the control module (CM) and the processing 138 module (PM), as shown in Figure 2. 139 Viewing traditional SNNs as a sequence of T = 1 sub-networks with shared parameters and temporal 140 dynamics governed by membrane potentials, we propose two approaches to introducing the top141 down pathway. The first adds recurrent feedback connections between these fine-grained T = 1 142 sub-networks, enabling temporal context to propagate backward through time. The second adopts 143 a coarser temporal resolution by dividing a sequence (e.g., T = 4) into fewer segments (e.g., two 144 T = 2 blocks). Importantly, the additional power overhead introduced by both schemes remains 145 minimal. Detailed analysis of power consumption is provided in appendix C.1. Both approaches can 146 be expressed in the following unified formulation:

H1 = Ftr CM S (1) bu , ∅  H1 ∈ {0, 1} T ×N×C , S(1) bu ∈ {0, 1} T ×H×W×C (10) S (1) td = PM(H1) S (1) td ∈ {0, 1} T ×N×C , H1 ∈ {0, 1} T ×N×C (11) Hn = Ftr CM S (n) bu , S(n−1) td   S (n) bu ∈ {0, 1} T ×H×W×C , n = 1 . . . N (12) S (n) td = PM(Hn) S (n) td ∈ {0, 1} T ×N×C , n = 1 . . . N (13) On = CH(Hn) On ∈ {0, 1} T ×L, Hn ∈ {0, 1} T ×N×C , n = 1 . . . N (14)
Spatial Bottom-Up Pathway Prefrontal Cortex Inputs Classif ication Head Conv-Based SNN Blocks
× 
Spiking Transformer Blocks Ftr
× 
 1 ℒ = ℒ(
,)
High-Order Information Visual Cortex Processing Module 
(PM)
Control Module 
(CM)

Tem poral 2 Top-Down Pathway Prefrontal Cortex

+1 Inputs ℒ = ℒ(
Classi fication Conv-Based SNN Blocks Spiking Transformer Blocks Ftr
,)
×Head

× 
Visual Cortex
(a) The framework of TDFormer MLP
C
H × W
Spiking Feature Maps

(−1)

()
Linear BN
Linear BN
H × W
T
CLearnable Channel-Wise Weights Spiking Self-Attention PM PM

 
O(ND)T
2-D Spatial-Wise Weights V1 V2 2
(−1)
2
()
O(N)
BN BN

Linear Conv Token Mixer BN

(−1)

(−1)
Ftr & CM
Ftr & CM
V4 Linear BN
BN
Conv Enhanced Spiking Feature Maps C
T

()
V3
(b) Control module (CM) (c) Processing module (PM)
(e) PM variants
(d) Subnetwork dynamics PM Variants Spike Neuron (LIF) Spike Neuron (PLIF) AC Operation
In the above formulation, S
(n)
bu denotes the bottom-up input at time step n, while S
(n−1)
td 147 represents 148 the top-down feedback from the previous step. CM is a control module that integrates bottom-up and 149 top-down signals, and Ftr denotes the Transformer-based processing unit. The processing module PM generates the current feedback signal S
(n)
td 150 from the high-level representation Hn, and CH maps 151 Hn to the final output On, where N denotes the number of sub-networks. The bottom part of Figure 152 2 illustrates the feedback information flow between sub-networks. 153 **For the control module (CM),** CM derives the query Q, key K, and value V vectors from the 154 bottom-up information Sbu and the top-down information Std. In more detail, Std facilitates attention 155 correction by controlling the attention map. The CM can be formulated as follows:

$Q,K,V=CM(S_{bu},S_{td}),$  $K=\mathcal{SN}(\text{BN}(\text{TokenMix}\left((S_{bu},S_{td}))\right)),$  $Q=\mathcal{SN}(\text{BN}(\text{Linear}(S_{bu}))),V=\mathcal{SN}(\text{BN}(\text{Linear}(S_{bu}))).$
156 We choose concatenation along the channel dimension as the default token mixer, which allows us 157 to combine the features of the current time step with those from previous time steps, and use the 158 fused information to dynamically adjust the self-attention map. After passing through the CM, the 159 query Q, key K and value V vectors are fed into the self-attention module to obtain the top-down 160 attention map. To prevent the fusion of top-down information from altering the distribution of K 161 in the self-attention computation, we first normalize the combined features, and then apply spike 162 discretization before computing self-attention. Ablation studies on different CM variants are provided 163 in the appendix C.2.

$$(15)^{\frac{1}{2}}$$
$\eqref{eq:walpha}$. 
$$(17)$$

164 **The processing module (PM)** PM includes both channel-wise token mixer and spatial-wise token 165 mixer [27]. The feature enhancement component enhances the original spiking feature maps X
by learning channel-wise Wc and computing spatial-wise attention maps Mspatial 166 . This attention 167 mechanism requires very few parameters and has a time complexity of O(ND). This operation is 168 represented as:

$$\mathbf{M}_{\text{spatial}}(t,n)=\sum_{c=1}^{C}\mathbf{W}_{c}\cdot\mathbf{X}_{t,n,c},$$ $$\mathbf{M}_{\text{spatial}}=\text{clamp}\left(\mathbf{M}_{\text{spatial}},b,a\right).\tag{1}$$
$$(18)$$

$$(19)$$
$$(20)$$
$$\mathbf{O}={\mathcal{S N}}(\mathbf{X}\odot\mathbf{M}_{\mathrm{spatial}}).$$

169 where X*t,n,c* represents the spiking activation at time t, spatial position n (corresponding to the 170 2D coordinate (*h, w*) in the feature map), and channel c. Here, a and b are hyperparameters. We 171 theoretically derive their effects on the PM output, and the details are given in appendix B.2. The 172 spatial attention map Mspatial weights the spiking feature map X via element-wise multiplication, 173 with broadcasting over the channel dimension:
O = SN (X ⊙ Mspatial). (20)
174 The attention embedding spaces are different across layers, and we aim to use a PM variants to 175 align the top-down information with the embedding spaces of different layers. We explored four PM 176 variants that serve as the channel-wise token mixer, which are illustrated in Figure 2. 177 We introduce a clamp operation in the attention module to enforce a strict upper bound on the variance 178 of the attention map which is formally stated in Proposition 4.1. Excessive variance can lead to 179 gradient vanishing, as gradients in spiking neurons are only generated near the firing threshold of 180 the membrane potential. Outside this narrow region, the gradient tends to vanish. Furthermore, high 181 variance may introduce outliers, resulting in significant quantization errors during spike generation. 182 The effect of the clamp operation on the gradient is shown in the Figure appendix C.2.

Proposition 4.1. The upper bound Var(Ytnc) for the X ⊙ M*spatial* 183 *is given as follows:*

$${\overline{{V a r}}}(Y_{t n c})=\begin{cases}a^{2}(f^{2}-f+{\frac{1}{2}})+a b(1-2f)+{\frac{b^{2}}{2}},&{\text{if}}0\leq f\leq{\frac{a+b}{2a}},\\ {\frac{a^{2}+2a b+b^{2}-4f a b}{4}},&{\text{if}}{\frac{a+b}{2a}}\leq f\leq1,\end{cases}$$
$$(21)$$

184 where we assume each Xt,n,c is independent random variable Xtnc ∼ Bernoulli(f), with f *as the* 185 *firing rate.* 186 Additionally, the clamp operation eliminates the need for scaling operations in attention mechanisms 187 (e.g., QK product scaling), simplifying computations, reducing complexity, and improving energy 188 efficiency in hardware implementations. The detailed proofs of this proposition are provided in 189 appendix B.1.

## 190 **4.2 Loss Function**

191 The loss of the TDFormer can be formulated as follows:

$${\mathcal{L}}_{\mathrm{TDFormer}}=\sum_{n=1}^{N}\alpha_{n}{\mathcal{L}}(y,O_{n}),\quad\sum_{n=1}^{N}\alpha_{n}=1,\quad0\leq\alpha_{n}\leq1.$$
$$(22)$$

192 Here, αn are hyperparameters. To maintain the overall loss scale, we apply a weighted average over 193 the losses from all N stages, assigning a larger weight to the final output loss. This is because we 194 believe that the receptive field in the temporal dimension increases as time progresses. Since the 195 earlier stages lack feedback from future steps, their outputs are less accurate and thus subject to 196 weaker supervision. By contrast, the final stage benefits from a larger temporal receptive field due to 197 feedback, making its output more reliable. Therefore, during testing, only the output from the last 198 sub-network is used for evaluation.

## 199 **4.3 Top-Down Feedback Enhances Temporal Dependency**

200 Top-down feedback enhances temporal dependency from two perspectives. First, from the forward 201 propagation perspective, we compute the mutual information matrix between features at different time

| default PM variant is v1.   | ImageNet               |                           |                                |         |                |         |
|-----------------------------|------------------------|---------------------------|--------------------------------|---------|----------------|---------|
| Methods                     | Spike                  | Architecture              | Time Step Power (mJ) Param (M) | Acc (%) |                |         |
| ViT [28]                    | ✗                      | ViT-B/16(3842 )           | 1                              | 254.84  | 86.59          | 77.90   |
| DeiT [29]                   | ✗                      | DeiT-B(3842 )             | 1                              | 254.84  | 86.59          | 83.10   |
| Swin [30]                   | ✗                      | Swin Transformer-B(3842 ) | 1                              | 216.20  | 87.77          | 84.50   |
| Spikingformer [4]           | ✓                      | Spikingformer-8-768       | 4                              | 13.68   | 66.34          | 75.85   |
| ✓                           | Spikformer-8-512       | 4                         | 11.58                          | 29.68   | 73.38          |         |
| SpikformerV1 [2]            | ✓                      | Spikformer-8-768          | 4                              | 21.48   | 66.34          | 74.81   |
| ✓                           | Meta-SpikeFormer-8-384 | 4                         | 32.80                          | 31.30   | 77.20          |         |
| SDTV2 [6]                   | ✓                      | Meta-SpikeFormer-8-512    | 4                              | 52.40   | 55.40          | 80.00   |
| ✓                           | E-Spikeformer          | 8                         | 30.90                          | 83.00   | 84.00          |         |
| ✓                           | E-Spikeformer          | 8                         | 54.70                          | 173.00  | 85.10          |         |
| E-Spikeformer [31]          | ✓                      | E-Spikeformer             | 8                              | -       | 173.00         | 86.20 # |
| ✓                           | HST-10-768 (2242 )     | 4                         | 38.91                          | 64.96   | 84.22          |         |
| QKFormer [3]                | ✓                      | HST-10-768 (2882 )        | 4                              | 64.27   | 64.96          | 85.20   |
| ✓                           | HST-10-768 (3842 )     | 4                         | 113.64                         | 64.96   | 85.65          |         |
| ✓                           | HST-10-768 (2242 )     | 4                         | 38.93                          | 65.55   | 85.37(+1.15)   |         |
| ✓                           | HST-10-768 (2882 )     | 4                         | 64.39                          | 65.55   | 86.29(+1.09)   |         |
| ✓                           | HST-10-768 (2242 )     | 4                         | 39.10                          | 69.09   | 85.57(+1.35)   |         |
| ✓                           | HST-10-768 (2882 )     | 4                         | 64.45                          | 69.09   | 86.43 (+1.23)  |         |
| ✓                           | HST-10-768 (3842 )     | 4                         | 113.79                         | 69.09   | 86.83 (+1.18)* |         |
| TDFormer                    |                        |                           |                                |         |                |         |

202 steps, as shown in Figure 1. Second, from the backward propagation perspective, we demonstrate that 203 introducing top-down feedback helps alleviate the problem of vanishing gradients along the temporal 204 dimension. We present the following theorem:
Definition 4.2. ϵ l(t) is defined as the sensitivity of the membrane potential Hl 205 (t + 1) to its previous state Hl 206 (t), and is computed as:

$$\epsilon^{l}(t)\equiv\frac{\partial\mathbf{H}^{l}(t+1)}{\partial\mathbf{H}^{l}(t)}+\frac{\partial\mathbf{H}^{l}(t+1)}{\partial\mathbf{S}^{l}(t)}\frac{\partial\mathbf{S}^{l}(t)}{\partial\mathbf{H}^{l}(t)},$$
$$(23)$$

207 where l indexes the layer. 208 **Theorem 4.3.** *We adopt the rectangular function as the surrogate gradient, following the setting* 209 used in previous studies[8, 9, *12]. For a conventional SNN, the sensitivity of the membrane potential* 210 *is expressed as follows:*

$$\epsilon^{l}(t)_{j j}=\left\{\begin{array}{l l}{{0,}}&{{\frac{1}{2}\vartheta<H_{j}^{l}(t)<\frac{3}{2}\vartheta,}}\\ {{1-\frac{1}{\tau},}}&{{o t h e r w i s e~.}}\end{array}\right.$$

211 *For SNN with top-down feedback structure, the sensitivity of the membrane potential can be expressed* 212 as:

$$\epsilon^{l}(t)_{j j}=\left\{\begin{array}{l l}{{\frac{\partial\varphi_{\vartheta}(\mathbf{S}^{l}(t))}{\partial\mathbf{S}^{l}(t)},}}&{{\frac{1}{2}\vartheta<H_{j}^{l}(t)<\frac{3}{2}\vartheta,}}\\ {{1-\frac{1}{\tau},}}&{{o t h e r v i s e\ .}}\end{array}\right.$$
$$(25)$$

213 where ϑ is the spike threshold, τ is a time constant and φθ *is a differentiable feedback function* 214 *parameterized by* θ.

According to Equation 24, ϵ l 215 (t) becomes zero within an easily-reached interval, and outside that interval, it is upper-bounded by a small value 1 −
1 τ 216 , since τ is typically close to 1 in practice[32, 33, 34, 9]. In contrast, our method allows non-zero gradients within this interval, and the ∂φθ(S
l(t))
∂Sl(t)
217 can

$$(24)$$

| 100. Conventions align with those in Table 1. The default PM variant is v1. CIFAR-10   | CIFAR-100     |                |                |
|----------------------------------------------------------------------------------------|---------------|----------------|----------------|
| Methods                                                                                | Time          |                |                |
| [Architecture]                                                                         | Step          | Acc            | Acc            |
| (%)                                                                                    | (%)           |                |                |
| STBP-tdBN [33] [ResNet-19]                                                             | 4             | 92.92          | 70.86          |
| TET [32] [ResNet-19]                                                                   | 4             | 94.44          | 74.47          |
| SDTV1[5][SDT-2-512]                                                                    | 4             | 95.60          | 78.40          |
| QKformer [3] [HST-4-384]                                                               | 4             | 96.18 #        | 81.15 #        |
| SpikformerV1 [2] [Spikformer-4-384]                                                    | 2             | 93.59          | 76.28          |
| 4                                                                                      | 95.19         | 77.86          |                |
| SpikformerV1(ours)[Spikformer-4-384]                                                   | 2             | 93.65          | 75.29          |
| 4                                                                                      | 94.73         | 77.88          |                |
| TDFormer[Spikformer-4-384]                                                             | 2             | 94.17 (+0.52)  | 75.79 (+0.50)  |
| 4                                                                                      | 95.11 (+0.38) | 77.99 (+0.11)  |                |
| SDTV1(ours)[SDT-2-256]                                                                 | 4             | 94.47          | 76.05          |
| SDTV1(ours)[SDT-2-512]                                                                 | 4             | 95.78          | 79.15          |
| TDFormer[SDT-2-256]                                                                    | 4             | 94.61 (+0.14)  | 76.23 (+0.18)  |
| TDFormer[SDT-2-512]                                                                    | 4             | 96.07 (+0.29)  | 79.67 (+0.52)  |
| TDFormer [HST-4-384]                                                                   | 4             | 96.51 (+0.33)* | 81.45 (+0.30)* |

exceed 1 −
1 τ 218 . This property helps to alleviate the vanishing gradient problem along the temporal 219 dimension. The detailed proof is provided in the appendix B.3.

## 220 **5 Experiments**

221 We evaluate our models on several datasets: CIFAR-10 [35], CIFAR-100 [35], CIFAR10-DVS [36], 222 DVS128 Gesture [37], ImageNet [38], CIFAR-10C [39] and ImageNet-C [39]. For the smaller 223 datasets, we employ the feedback pathway on SpikformerV1 [2] , Spike-driven TransformerV1 [5] 224 and QKformer[3], experimenting with different configurations tailored to each dataset. For the large225 scale datasets, we utilize QKformer[3] as baselines. Specific implementation details are provided in 226 appendix A.

## 227 **5.1 Experiments On Imagenet**

228 Table 1 presents the results for the large-scale dataset ImageNet. The incorporation of top-down 229 feedback structure has demonstrated significant improvements on E-spikformer, which is the previous 230 SOTA model of SNNs. Notably, compared to QKFormer, increasing the model size by merely 0.02 231 million parameters and 0.59 millijoules of power consumption leads to a significant gain of 1.15% 232 in top-1 accuracy on the ImageNet dataset. Our model sets a new SOTA performance in the SNN 233 field. This milestone lays a solid foundation for advancing SNNs toward large-scale networks, further 234 bridging the gap between SNNs and traditional deep learning models. Furthermore, we calculate the 235 power of TDFormer following the method in [3], as detailed in Table 1. TDFormer results in a slight 236 increase in energy consumption due to the feedback structure, but it achieves superior performance 237 with minimal additional power usage. The detailed calculation of power consumption is provided in 238 the appendix C.1.

## 239 **5.2 Experiments On Neuromorphic And Cifar Datasets**

240 Table 3 presents the results for the neuromorphic datasets CIFAR10-DVS and DVS128 Gesture. Our 241 proposed TDFormer consistently outperforms the baselines across all experiments, except for the 242 Spiking Transformer-2-256 at a time step of 10. Furthermore, we achieve SOTA results, with an 243 accuracy of 85.83% on CIFAR10-DVS using the HST-2-256 (V1), marking a notable improvement

| align with those in Table 1. The default PM variant is v1. CIFAR10-DVS   | DVS128 Gesture   |                |                |               |
|--------------------------------------------------------------------------|------------------|----------------|----------------|---------------|
| Methods [Architecture]                                                   | Time             | Acc            | Time           | Acc           |
| Step                                                                     | (%)              | Step           | (%)            |               |
| STBP-tdBN [33] [ResNet-19]                                               | 10               | 67.80          | 40             | 96.90         |
| DSR [40] [VGG-11]                                                        | 10               | 77.30          | -              | -             |
| SDTV1 [5][SDT-2-256]                                                     | 16               | 80.00          | 16             | 99.30 #       |
| SpikformerV1 [2] [Spikformer-2-256]                                      | 10               | 78.90          | 10             | 96.90         |
| 16                                                                       | 80.90            | 16             | 98.30          |               |
| Spikingformer [4] [Spikingformer-2-256]                                  | 10               | 79.90          | 10             | 96.20         |
| 16                                                                       | 81.30            | 16             | 98.30          |               |
| Qkformer [3] [HST-2-256]                                                 | 16               | 84.00 #        | 16             | 98.60         |
| SpikformerV1(ours) [Spikformer-2-256]                                    | 10               | 78.08          | -              | -             |
| 16                                                                       | 79.40            | -              | -              |               |
| TDFormer [Spikformer-2-256]                                              | 10               | 78.90 (+0.82)  | -              | -             |
| 16                                                                       | 81.70 (+2.30)    | -              | -              |               |
| SDTV1(ours) [SDT-2-256]                                                  | 10               | 75.22          | 10             | 96.79         |
| 16                                                                       | 77.07            | 16             | 97.98          |               |
| TDFormer[SDT-2-256]                                                      | 10               | 75.05 (-0.17)  | 10             | 96.92 (+0.13) |
| 16                                                                       | 77.45 (+0.38)    | 16             | 99.65 (+1.67)* |               |
| TDFormer[HST-2-256]                                                      | 16               | 85.83 (+1.83)* | 16             | 98.96 (+0.36) |

244 of 1.83% compared to the previous SOTA model, QKformer. We also achieve 99.65% accuracy on 245 DVS128 Gesture using the Spiking Transformer-2-256 (V1) at 16 time steps. 246 In addition, the results for the static datasets CIFAR-10 and CIFAR-100 are summarized in Table 2. 247 Compared to the baselines, the proposed TDFormer consistently demonstrates significant performance 248 improvements across all experiments, with the exception of Spikformer-4-384 (V1) at time step 249 6. Furthermore, we achieve the SOTA performance, attaining 96.51% accuracy on CIFAR-10 and 250 81.45% on CIFAR-100 using the HST-2-256 (V1) at a time step of 4.

## 251 **5.3 Model Generalization Analysis**

252 As reported in Table 5, we report results averaged over five random seeds for reliability. Our model 253 consistently improves performance across time steps and depths. To assess robustness, we evaluate 254 on the CIFAR-10C dataset with 15 corruption types. As shown in Table 7, the model equipped with 255 the TDAC module consistently achieves higher accuracy under various distortion settings.

256 Moreover, we provide a visualization analysis of the TDFormer attention modules on CIFAR-10C 257 and ImageNet-C. The specific results can be seen in Figure 4 and Figure 5 of the appendix C. We 258 find that after adding the TDAC module, the model focuses more on the targets and their surrounding 259 areas. This indicates that TDAC can filter noise and irrelevant information, allowing the model to 260 focus more on task-related information.

## 261 **6 Conclusion**

262 In this study, we propose TDFormer, which integrates an adaptive top-down feedback structure into 263 Transformer-based SNNs, addressing a key limitation of temporal information utilization in existing 264 models by incorporating biological top-down mechanisms. The TDFormer model outperforms 265 traditional Transformer-based SNNs, achieving SOTA performance across all evaluated datasets. Our 266 work suggests that the top-down feedback structure could be a valuable component for Transformer267 based SNNs and offers insights for future research into more advanced, biologically inspired neural 268 architectures that better mimic human cognition.

## 269 **References**

270 [1] Kai Malcolm and Josue Casco-Rodriguez. A comprehensive review of spiking neural networks: 271 Interpretation, optimization, efficiency, and best practices. *arXiv preprint arXiv:2303.10780*, 272 2023.

273 [2] Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng YAN, Yonghong Tian, 274 and Li Yuan. Spikformer: When spiking neural network meets transformer. In *The Eleventh* 275 *International Conference on Learning Representations*, 2023. 276 [3] Chenlin Zhou, Han Zhang, Zhaokun Zhou, Liutao Yu, Liwei Huang, Xiaopeng Fan, Li Yuan, 277 Zhengyu Ma, Huihui Zhou, and Yonghong Tian. Qkformer: Hierarchical spiking transformer 278 using qk attention. *arXiv preprint arXiv:2403.16552*, 2024. 279 [4] Chenlin Zhou, Liutao Yu, Zhaokun Zhou, Zhengyu Ma, Han Zhang, Huihui Zhou, and Yonghong 280 Tian. Spikingformer: Spike-driven residual learning for transformer-based spiking neural 281 network. *arXiv preprint arXiv:2304.11954*, 2023. 282 [5] Man Yao, JiaKui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo XU, and Guoqi Li. Spike283 driven transformer. In *Thirty-seventh Conference on Neural Information Processing Systems*,
284 2023. 285 [6] Man Yao, JiaKui Hu, Tianxiang Hu, Yifan Xu, Zhaokun Zhou, Yonghong Tian, Bo XU, and 286 Guoqi Li. Spike-driven transformer v2: Meta spiking neural network architecture inspiring the 287 design of next-generation neuromorphic chips. In *The Twelfth International Conference on* 288 *Learning Representations*, 2024. 289 [7] Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, Yuan Xie, and Luping Shi. Direct training for spiking 290 neural networks: Faster, larger, better. In *Proceedings of the AAAI conference on artificial* 291 *intelligence*, volume 33, pages 1311–1318, 2019.

292 [8] Yongqi Ding, Lin Zuo, Mengmeng Jing, Pei He, and Hanpu Deng. Rethinking spiking neural 293 networks from an ensemble learning perspective. *arXiv preprint arXiv:2502.14218*, 2025. 294 [9] Qingyan Meng, Mingqing Xiao, Shen Yan, Yisen Wang, Zhouchen Lin, and Zhi-Quan Luo. 295 Towards memory-and time-efficient backpropagation for training spiking neural networks. In 296 *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 6166–6176, 297 2023. 298 [10] Hangchi Shen, Qian Zheng, Huamin Wang, and Gang Pan. Rethinking the membrane dynamics 299 and optimization objectives of spiking neural networks. *Advances in Neural Information* 300 *Processing Systems*, 37:92697–92720, 2024. 301 [11] Wei Liu, Li Yang, Mingxuan Zhao, Shuxun Wang, Jin Gao, Wenjuan Li, Bing Li, and Weiming 302 Hu. Deeptage: Deep temporal-aligned gradient enhancement for optimizing spiking neural 303 networks. In *The Thirteenth International Conference on Learning Representations*. 304 [12] Yulong Huang, Xiaopeng Lin, Hongwei Ren, Haotian Fu, Yue Zhou, Zunchang Liu, Biao Pan, 305 and Bojun Cheng. Clif: Complementary leaky integrate-and-fire neuron for spiking neural 306 networks. *arXiv preprint arXiv:2402.04663*, 2024. 307 [13] Donghyun Lee, Yuhang Li, Youngeun Kim, Shiting Xiao, and Priyadarshini Panda. Spiking 308 transformer with spatial-temporal attention. *arXiv preprint arXiv:2409.19764*, 2024. 309 [14] Charles D Gilbert and Wu Li. Top-down influences on visual processing. *Nature reviews* 310 *neuroscience*, 14(5):350–363, 2013. 311 [15] Timothy J Buschman and Earl K Miller. Top-down versus bottom-up control of attention in the 312 prefrontal and posterior parietal cortices. *science*, 315(5820):1860–1862, 2007.

313 [16] John H Reynolds and David J Heeger. The normalization model of attention. *Neuron*, 61(2):168–
314 185, 2009. 315 [17] Maurizio Corbetta, Erbil Akbudak, Thomas E Conturo, Abraham Z Snyder, John M Ollinger, 316 Heather A Drury, Martin R Linenweber, Steven E Petersen, Marcus E Raichle, David C 317 Van Essen, et al. A common network of functional areas for attention and eye movements. 318 *Neuron*, 21(4):761–773, 1998. 319 [18] Zhaokun Zhou, Kaiwei Che, Wei Fang, Keyu Tian, Yuesheng Zhu, Shuicheng Yan, Yonghong 320 Tian, and Li Yuan. Spikformer v2: Join the high accuracy club on imagenet with an snn ticket. 321 *arXiv preprint arXiv:2401.02020*, 2024. 322 [19] Rui-Jie Zhu, Qihang Zhao, Guoqi Li, and Jason K Eshraghian. Spikegpt: Generative pre-trained 323 language model with spiking neural networks. *arXiv preprint arXiv:2302.13939*, 2023. 324 [20] Han Zhang, Chenlin Zhou, Liutao Yu, Liwei Huang, Zhengyu Ma, Xiaopeng Fan, Huihui Zhou, 325 and Yonghong Tian. Sglformer: Spiking global-local-fusion transformer with high performance. 326 *Frontiers in Neuroscience*, 18:1371290, 2024. 327 [21] Yuhua Zheng, Yan Meng, and Yaochu Jin. Object recognition using a bio-inspired neuron 328 model with bottom-up and top-down pathways. *Neurocomputing*, 74(17):3158–3169, 2011. 329 [22] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, 330 and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question 331 answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 332 pages 6077–6086, 2018. 333 [23] Baifeng Shi, Trevor Darrell, and Xin Wang. Top-down visual attention from analysis by 334 synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern* 335 *Recognition*, pages 2102–2112, 2023. 336 [24] Roy Abel and Shimon Ullman. Top-down network combines back-propagation with attention. 337 *arXiv preprint arXiv:2306.02415*, 2023. 338 [25] Wulfram Gerstner, Werner M Kistler, Richard Naud, and Liam Paninski. *Neuronal dynamics:* 339 *From single neurons to networks and models of cognition*. Cambridge University Press, 2014. 340 [26] A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 341 2017. 342 [27] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and 343 Shuicheng Yan. Metaformer is actually what you need for vision. In *Proceedings of the* 344 *IEEE/CVF conference on computer vision and pattern recognition*, pages 10819–10829, 2022. 345 [28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, 346 Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 347 An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint* 348 *arXiv:2010.11929*, 2020. 349 [29] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and 350 Hervé Jégou. Training data-efficient image transformers & distillation through attention. In 351 *International conference on machine learning*, pages 10347–10357. PMLR, 2021. 352 [30] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining 353 Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings* 354 *of the IEEE/CVF international conference on computer vision*, pages 10012–10022, 2021. 355 [31] Man Yao, Xuerui Qiu, Tianxiang Hu, Jiakui Hu, Yuhong Chou, Keyu Tian, Jianxing Liao, 356 Luziwei Leng, Bo Xu, and Guoqi Li. Scaling spike-driven transformer with efficient spike 357 firing approximation training. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 358 2025. 359 [32] Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu. Temporal efficient training of 360 spiking neural network via gradient re-weighting. *arXiv preprint arXiv:2202.11946*, 2022.

361 [33] Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li. Going deeper with directly-trained 362 larger spiking neural networks. In *Proceedings of the AAAI conference on artificial intelligence*, 363 volume 35, pages 11062–11070, 2021. 364 [34] Yufei Guo, Xinyi Tong, Yuanpei Chen, Liwen Zhang, Xiaode Liu, Zhe Ma, and Xuhui Huang. 365 Recdis-snn: Rectifying membrane potential distribution for directly training spiking neural net366 works. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 367 pages 326–335, 2022. 368 [35] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 369 University of Toronto, 2009. 370 [36] Hongmin Li, Hanchao Liu, Xiangyang Ji, Guoqi Li, and Luping Shi. Cifar10-dvs: an event371 stream dataset for object classification. *Frontiers in neuroscience*, 11:309, 2017. 372 [37] Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo, 373 Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, et al. A low 374 power, fully event-based gesture recognition system. In *Proceedings of the IEEE conference on* 375 *computer vision and pattern recognition*, pages 7243–7252, 2017. 376 [38] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large377 scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern* 378 *recognition*, pages 248–255. Ieee, 2009. 379 [39] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common 380 corruptions and perturbations. *arXiv preprint arXiv:1903.12261*, 2019. 381 [40] Qingyan Meng, Mingqing Xiao, Shen Yan, Yisen Wang, Zhouchen Lin, and Zhi-Quan Luo.

382 Training high-performance low-latency spiking neural networks by differentiation on spike 383 representation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern* 384 *recognition*, pages 12444–12453, 2022. 385 [41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint* 386 *arXiv:1711.05101*, 2017. 387 [42] Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, and Guoqi Li. Integer-valued training and 388 spike-driven inference spiking neural network for high-performance and energy-efficient object 389 detection. In *European Conference on Computer Vision*, pages 253–272. Springer, 2024. 390 [43] Youngeun Kim, Joshua Chough, and Priyadarshini Panda. Beyond classification: Directly 391 training spiking neural networks for semantic segmentation. *Neuromorphic Computing and* 392 *Engineering*, 2(4):044015, 2022.

393 [44] Changze Lv, Jianhan Xu, and Xiaoqing Zheng. Spiking convolutional neural networks for text 394 classification. *arXiv preprint arXiv:2406.19230*, 2024.

## 395 **A Implementation Details** 396 **A.1 Training Protocols**

397 We adopted the following training protocols: 398 - **Spike Generation**: We used a rate-based method for spike generation [2]. 399 - **Data Augmentation and Training Duration**: SpikformerV1 experiments followed [2], 400 while Spike-driven TransformerV1 experiments followed [5], furthermore QKformer experi401 ments followed the experimental setting in and [3]. 402 - **Optimization**: We employed AdamW [41] as the optimizer for our experiments. The learning rate was set to 3 × 10−4 403 for the Spike-driven TransformerV1. For SpikformerV1, we used a learning rate of 5 × 10−4 on static datasets and 1 × 10−3 404 on neuromorphic 405 datasets. Additionally, we utilized a cosine learning rate scheduler to adjust the learning 406 rate dynamically during training. Specifically, for QKformer, we fine-tuned the pretrained network with a base learning rate of 2 × 10−5 407 for 15 epochs, due to the high cost of direct 408 training on ImageNet using 4 time steps.

- **Batch Size**: The batch sizes for different datasets and models are specified in Table 4.

| Dataset         | Model                      | Batch Size   |
|-----------------|----------------------------|--------------|
| CIFAR-10 and    | SpikeformerV1              | 128          |
| CIFAR-100       | Spike-driven TransformerV1 | 64           |
| CIFAR10-DVS and | SpikeformerV1              | 16           |
| DVS128 Gesture  | Spike-driven TransformerV1 | 16           |
| ImageNet        | QKformer                   | 57           |

409

## 410 **A.2 Datasets**

411 Our experiments evaluated the performance and robustness of the TDFormer model using the 412 following datasets: 413 - **CIFAR-10:** This dataset contains 60,000 32 × 32 color images divided into 10 classes [35]. 414 - **CIFAR-100:** This dataset is similar to CIFAR-10 but includes 100 classes, providing a more 415 challenging classification task [35].

416 - **CIFAR10-DVS:** This is an event-based version of the CIFAR-10 dataset [36]. 417 - **DVS128 Gesture:** This is an event-based dataset for gesture recognition with 11 classes 418 [37]. 419 - **ImageNet:** This large-scale dataset contains over 1.2 million images divided into 1,000 420 classes [38]. 421 - **CIFAR-10C:** This is a corrupted version of CIFAR-10 with 19 common distortion types, 422 used to assess robustness [39]. 423 - **ImageNet-C:** This dataset is a corrupted version of ImageNet, designed similarly to CIFAR- 424 10C [39].

## 425 **A.3 Computational Environment** 426 **A.3.1 Software Setup**

427 We utilized PyTorch version 2.0.1 with CUDA 11.8 support and SpikingJelly version 0.0.0.0.12 as 428 the primary software tools.

## 429 **A.3.2 Hardware Setup.**

430 For the smaller dataset experiments, we utilized the following configuration: 431 - **Hardware Used:** NVIDIA L40S and L40 GPUs.

432 - **Configuration:** Single-GPU for each experiment.

433 - **Memory Capacity:** Each GPU is equipped with 42 GB of memory. 434 For the large-scale dataset (ImageNet) experiments, we employed the following setup: 435 - **Hardware Used:** NVIDIA H20 GPUs. 436 - **Configuration:** Eight-GPU for each experiment. 437 - **Memory Capacity:** Each GPU provides 96 GB of memory.

## 438 **A.4 Random Seed**

439 To ensure the comparability of the results, we selected the same random seeds as those in the baseline 440 paper. To ensure robustness, we also conducted experiments with random seeds 0, 42, 2024, 3407 441 and 114514, averaging the results. Detailed results are presented in Table 5.

## 442 **B Mathematical Derivations**

443 **B.1 Detailed proofs of the upper bound on PM output variance**
444 *Proof.* We assume that each Mspatial(*t, n*) is an independent random variable Mtn. Given that 445 b ≤ Mtn ≤ a, it follows that b ≤ E[Mtn] ≤ a. Furthermore, when Xtnc ̸= 0, we have:
(XtncMtn − b)(a − XtncMtn) ≥ 0, (26)
446 which expands to:
−(XtncMtn)
2 + (a + b)(XtncMtn) − ab ≥ 0. (27)
447 Taking the expectation on both sides yields:
E
-(XtncMtn)
2≤ (a + b)E [XtncMtn] − ab. (28)
448 Using the Law of Total Variance, we can decompose the variance of Ytnc as:
Var(Ytnc) = E[Var(Ytnc|Xtnc)] + Var(E[Ytnc|Xtnc]). (29)
449 For the first term, the expectation of the conditional variance can be expressed as:
E[Var(Ytnc|Xtnc)] = f · Var(Ytnc|Xtnc = 1) + (1 − f) · Var(Ytnc|Xtnc = 0). (30)
450 For the second term, the variance of the conditional expectation can be expanded as:
Var(E[Ytnc|Xtnc]) = E[E[Ytnc|Xtnc]
2] − E[E[Ytnc|Xtnc]]2. (31)
451 By substituting the conditional probabilities, we have:
Var(E[Ytnc|Xtnc]) = f · E[Ytnc|Xtnc = 1]2 − f 2· E[Ytnc|Xtnc = 1]2. (32)
452 Combining the two terms, the total variance becomes:
Var(Ytnc) = f · Var(Ytnc|Xtnc = 1) + (f − f 2) · E[Ytnc|Xtnc = 1]2. (33)

453 From Equation 32, we define E[Ytnc|Xtnc = 1] = µ. Substituting this definition, the variance can be
454 rewritten as:
Var(Ytnc) = f · (E[Y
2
tnc|Xtnc = 1] − µ
$$F)+(f-f^{2})\cdot\mu^{2}.$$
2. (34)
455 Using the constraints b ≤ Mtn ≤ a, we have the following bound for Var(Ytnc|Xtnc = 1):
Var(Ytnc|Xtnc = 1) ≤ (a + b)µ − ab − µ 2. (35)

$$(34)$$

456 By substituting this into the total variance expression, the upper bound of Var(Ytnc) becomes:

$$\mathrm{Var}(Y_{t n c})\leq f\cdot((a+b)\mu-a b-\mu^{2})+(f-f^{2})\cdot\mu^{2}$$ $$\leq-f^{2}\cdot\left(\mu-\frac{a+b}{2f}\right)^{2}+\frac{a^{2}+2a b+b^{2}-4f a b}{4}.$$
$$(36)$$

457 Next, we will prove that this upper bound can be achieved with equality under specific conditions.

Case 1: When a+b 458 2a ≤ f ≤ 1, we assume that:

$$\mathbb{E}[Y_{t n c}|X_{t n c}=1]={\frac{a+b}{2f}},\quad M_{t n}=a\operatorname{or}b.$$
$$(37)$$

459 Here, Mtn is a binary random variable, taking the value a with probability p and the value b with
460 probability 1−p. Using this assumption, we can express the conditional expectation E[Ytnc|Xtnc = 1]
461 as:E[Ytnc|Xtnc = 1] = pa + (1 − p)b. (38)
Substituting E[Ytnc|Xtnc = 1] = a+b
$\mathbb{E}[Y_{tnc}|X_{tnc}=1]=pa+(1-p)b$.  $1]=\frac{a+b}{2f}$ into the above equation, we solve for $p$
462 into the above equation, we solve for p:
$$p a+(1-p)b={\frac{a+b}{2f}}\Rightarrow p={\frac{a+b-2b f}{2f(a-b)}}.$$
$$(39)$$
. (39)
463 The variance of Ytnc under this distribution is maximized when Mtn follows this binary distribution.

464 Substituting p into the variance formula, the maximum variance is given by:

$$\operatorname*{max}(\operatorname{Var}(Y_{t n c}))={\frac{a^{2}+2a b+b^{2}-4f a b}{4}}.$$
$$(40)$$
$$(41)$$

Case 2: When 0 ≤ f ≤
a+b 2a 465 , the upper bound is achieved when Mtn = a. In this scenario, Mtn is 466 deterministic, and therefore:

$$Y_{t n c}=X_{t n c}M_{t n}=X_{t n c}a,\quad\mathbb{E}[Y_{t n c}|X_{t n c}=1]=a.$$

467 Substituting this into the variance formula, the maximum variance simplifies to:

$$\operatorname*{max}(\operatorname{Var}(Y_{t n c}))=a^{2}(f^{2}-f+1/2)+a b(1-2f)+b^{2}/2.$$
2/2. (42)
468 The proof is now complete.

469 We observe that both SSA and QKTA exhibit significantly larger variance compared to our proposed 470 attention mechanism. Their variances are expressed as follows: 471 **Variance of QKTA:**
$$\mathrm{Var}(\mathrm{QKTA})=d f_{Q}(1-f_{Q}),$$ and $f_{Q}$ represents the firing rate of the query. 
472 where d is the feature dimension, and fQ represents the firing rate of the query.
473 **Variance of SSA:**
$${\rm Var}({\rm SSA})=Nd\Big{(}f_{Q}f_{K}f_{V}(1-f_{Q})(1-f_{K})(1-f_{V})\tag{44}$$ $$+f_{Q}f_{K}f_{V}^{2}(1-f_{Q})(1-f_{K})$$ $$+f_{Q}f_{K}^{2}f_{V}(1-f_{Q})(1-f_{V})$$ $$+f_{Q}^{2}f_{K}f_{V}(1-f_{K})(1-f_{V})$$ $$+f_{Q}f_{K}^{2}f_{V}^{2}(1-f_{Q})$$ $$+f_{Q}^{2}f_{K}f_{V}^{2}(1-f_{K})$$ $$+f_{Q}^{2}f_{K}^{2}f_{V}(1-f_{V})\Big{)},$$
$$(43)$$
474 where N is the number of spatial locations, d is the feature dimension, and fQ, fK, fV are the firing 475 rates of the query, key, and value. 476 **Comparison with Our Attention Mechanism:** The variance of QKTA scales linearly with d. 477 By contrast, the variance of SSA grows with both N and d, resulting in significantly larger values 478 compared to QKTA. Our proposed attention mechanism is particularly effective in scenarios with large 479 spatial (N) and feature (d) dimensions. The strict upper bound on output variance ensures numerical 480 stability, preventing vanishing during training. Additionally, this upper bound eliminates the need 481 for traditional scaling operations (e.g., scaling factors in QK products), simplifying computations, 482 reducing complexity, and enhancing energy efficiency.

## 483 **B.2 The Mathematical Properties Of Hyperparameters**

484 Next, we will analyze the expectation and variance of the PM and propose an appropriate selection of 485 hyperparameters to ensure output stability.

486 **Lemma B.1.** *if the set* {c ∈ N : wc = 0} is finite and ∃ m, M > 0, ∀ c ∈ N, m ≤ |wc| ≤ M*, then:*

 $v_c=0\}\text{}is\ finite\ and\ \exists\ m,M>0,\forall\ c\in\mathbb{N},\ m\leq|w_c|\leq M,$  $w'_c=\lim\limits_{C\to\infty}\dfrac{w_c}{\sqrt{\sum_{c=1}^C w_c^2}}=0$
$$\mathbb{R}[T]_{*}$$
$$(45)$$
$$(46)$$
487 *Proof.* We begin by defining the normalized weight:

$$w_{c}^{\prime}=\frac{w_{c}}{\sqrt{\sum_{c=1}^{C}w_{c}^{2}}}.\tag{1}$$

488 By assumption, there are k terms where wc = 0, and for the remaining C − k terms, the weights
489 satisfy:
$$m^{2}\leq w_{c}^{2}\leq M^{2}\quad\mathrm{for~all~}c.$$
c ≤ M2for all c. (47)
490 Thus, the sum of squares of the weights is bounded as follows:
$$(C-k)m^{2}\leq\sum_{c=1}^{C}w_{c}^{2}\leq(C-k)M^{2}.$$
$$(47)$$
$$(48)$$
$$(49)$$
491 Taking the square root, we find that the denominator grows as:

$${\sqrt{\sum_{c=1}^{C}w_{c}^{2}}}\geq{\sqrt{(C-k)m^{2}}}\sim O({\sqrt{C}}).$$

Using the bound |wc| ≤ M, the normalized weight w
′c 492 satisfies:

$$|w_{c}^{\prime}|={\frac{|w_{c}|}{\sqrt{\sum_{c=1}^{C}w_{c}^{2}}}}\leq{\frac{M}{\sqrt{\sum_{c=1}^{C}w_{c}^{2}}}}\leq{\frac{M}{\sqrt{(C-k)m^{2}}}}.$$

To ensure |w
′c 493 | < ϵ for a given ϵ > 0, it suffices to require:

$$(S0)$$
$$\frac{M}{\sqrt{(C-k)m^{2}}}<\epsilon.$$

$$(51)$$

< ϵ. (51)
494 Rearranging, this condition can be rewritten as:

$$C\geq{\frac{M^{2}}{m^{2}\epsilon^{2}}}+k.$$
$$(52)$$
+ k. (52)
As C → ∞, the condition C ≥M2 m2ϵ 2 + k is always satisfied. Thus, for any ϵ > 0, we have |w
′c 495 | < ϵ, 496 which implies:

$$\lim_{C\to\infty}w^{\prime}_{c}=0.$$

497 The proof is complete.

$$(\mathbb{S}3)$$

498 **Lemma B.2.** *We assume that the features across different channels are independent and identically* 499 distributed (i.i.d.). When the number of channels C *is large, we have:*

$$M_{tn}\sim\mathcal{N}\left(\sum_{c=1}^{C}w_{c}^{\prime}f_{r},\sum_{c=1}^{C}w_{c}^{\prime2}f_{r}(1-f_{r})\right),\quad C\to\infty,$$ $$M_{tn}=\sum_{c=1}^{C}x_{tnc}w_{c}^{\prime}.$$
$$({\mathsf{S}}4)$$
$$(\mathbf{55})$$
$$(56)$$

500 501 where x ∈ X, x ∼ Bernoulli(fr), fr represents the firing rate (the probability of xtnc = 1). 502 *Proof.* To prove this lemma, we use the characteristic function method. The characteristic function 503 of a Bernoulli random variable xtnc is given by:

$$\Phi_{x_{t n c}}(t)=\mathbb{E}\left[e^{i t x_{t n c}}\right]=f_{r}e^{i t}+(1-f_{r}).\tag{1}$$

For the weighted variable w
′
504 cxtnc, its characteristic function is:

 ### Using  $\Phi_{w'_c x_{t n c}}(t)=\mathbb{E}\left[e^{i t w'_c x_{t n c}}\right]=f_r e^{i t w'_c}+(1-f_r)$. 
505 Since the features across channels are independent, the characteristic function of Mtn is:

$$\Phi_{M_{t n}}(t)=\prod_{c=1}^{C}\Phi_{w_{c}^{\prime}x_{t n c}}(t).\tag{1}$$
$$(57)$$
$$(58)$$
$$(59)$$

Substituting the expression for Φw′cxtnc 506 (t):

$$\Phi_{M_{t n}}(t)=\prod_{c=1}^{C}\left(f_{r}e^{i t w_{c}^{\prime}}+(1-f_{r})\right).$$
$$(60)$$
$$f_{r}e^{i t w_{c}^{\prime}}+(1-f_{r})=f_{r}\left(1+i t w_{c}^{\prime}-\frac{1}{2}t^{2}w_{c}^{\prime2}+o(w_{c}^{\prime2})\right)+(1-f_{r})$$ $$\approx1+f_{r}(i t w_{c}^{\prime}-\frac{1}{2}t^{2}w_{c}^{\prime2}).$$

507 Thus, the characteristic function becomes:

$$\Phi_{M_{t n}}(t)\approx\prod_{c=1}^{C}\left(1+f_{r}(i t w_{c}^{\prime}-\frac{1}{2}t^{2}w_{c}^{\prime2})\right).$$
$$(61)$$

508 Taking the logarithm to simplify the product into a sum:

$$\ln\Phi_{M_{t n}}(t)=\sum_{c=1}^{C}\ln\left(1+f_{r}(i t w_{c}^{\prime}-\frac{1}{2}t^{2}w_{c}^{\prime2})\right)$$ $$=\sum_{c=1}^{C}f_{r}i t w_{c}^{\prime}-\frac{1}{2}t^{2}w_{c}^{\prime2}f_{r}+\frac{1}{2}t^{2}w_{c}^{\prime2}f_{r}^{2}+O(w_{c}^{\prime2}),$$

where we used ln(1 + x) = x −
1 2 x 2 + O(x 2 509 ) for small x.

510 Separating terms, we get:

$$\ln\Phi_{M_{t n}}(t)\approx i t\sum_{c=1}^{C}w_{c}^{\prime}f_{r}-\frac{1}{2}t^{2}\sum_{c=1}^{C}w_{c}^{\prime2}f_{r}(1-f_{r}).$$
$$(62)$$
$$(63)$$

17 511 Exponentiating the logarithm gives:

$$\Phi_{M_{t n}}(t)=\exp\left(i t\sum_{c=1}^{C}w_{c}^{\prime}f_{r}-\frac{1}{2}t^{2}\sum_{c=1}^{C}w_{c}^{\prime2}f_{r}(1-f_{r})\right).$$
$$(64)$$
$$(65)$$
$$(66)$$

512 This is the characteristic function of a normal distribution with:

Mean: $\mu=\sum_{c=1}^{C}w^{\prime}_{c}f_{r}$, Variance: $\sigma^{2}=\sum_{c=1}^{C}w^{\prime2}_{c}f_{r}(1-f_{r})$.  
513 Since the characteristic function corresponds to a normal distribution, we conclude:

$$M_{t n}\sim{\mathcal{N}}\left(\sum_{c=1}^{C}w_{c}^{\prime}f_{r},\sum_{c=1}^{C}w_{c}^{\prime2}f_{r}(1-f_{r})\right).$$
$$\phi_{M_{t n}}(t_{1})\cdot\phi_{X_{t n c}}(t_{2}),\quad C\rightarrow\infty,$$
$$(67)$$

514 The proof is complete.

515 **Lemma B.3.** The distributions of Xtnc and Mtn *can be considered independent when the number of* 516 channels C is large. Specifically, for all t1, t2 ∈ R*, we have:*

ϕMtn,Xtnc (t1, t2) = ϕMtn (t1) · ϕXtnc (t2), C → ∞, (68)
517 where ϕX(t) *represents the characteristic function of* X.

518 *Proof.* The joint characteristic function of Mtn and Xtnc is given by:

$$\phi_{M_{tn},X_{tnc}}(t_{1},t_{2})=\mathbb{E}\left[e^{\left(it_{1}M_{tn}+it_{2}X_{tnc}\right)}\right]$$ $$=\mathbb{E}\left[e^{\left(it_{1}\sum_{c}w^{\prime}_{c}X_{tnc}+it_{2}X_{tnc}\right)}\right].$$
$$(69)$$
$$(70)$$
$$(71)$$

Separating Xtnc and the sum Pi̸=c w
′iXtni 519 , we rewrite:

$$\phi_{M_{tn},X_{tne}}(t_{1},t_{2})=\mathbb{E}\left[e^{\left(i t_{1}\sum_{i\neq e}w_{i}^{\prime}X_{tni}+i X_{tne}(t_{2}+t_{1}w_{e}^{\prime})\right)}\right]$$ $$=\mathbb{E}\left[e^{\left(i t_{1}\sum_{i\neq e}w_{i}^{\prime}X_{tni}\right)}\right]\cdot\mathbb{E}\left[e^{\left(i X_{tne}(t_{2}+t_{1}w_{e}^{\prime})\right)}\right].$$

520 Using the independence of Xtni across channels:

$$\phi_{M_{t n},X_{t n c}}(t_{1},t_{2})=\prod_{i\neq c}\mathbb{E}\left[e^{\left(i t_{1}w_{i}^{\prime}X_{t n i}\right)}\right]\cdot\mathbb{E}\left[e^{\left(i X_{t n c}(t_{2}+t_{1}w_{c}^{\prime})\right)}\right].$$

521 Substituting the characteristic function of Bernoulli random variables Xtnc ∼ Bernoulli(f):

$$\mathbb{E}\left[e^{i t X_{t n c}}\right)]=(1-f)+f e^{i t}.$$
itXtnc ] = (1 − f) + feit. (72)
522 Thus:

$$\phi_{M_{t n},X_{t n c}}(t_{1},t_{2})=\prod_{i\neq c}\left[(1-f)+f e^{i t_{1}w_{i}^{\prime}}\right]\cdot\left[(1-f)+f e^{i(t_{2}+t_{1}w_{c}^{\prime})}\right].$$
c)i. (73)
Using Lemma B.2, for small w
′c 523 , we apply the Taylor expansion to approximate each term:

$$(1-f)+fe^{it_{1}w^{\prime}_{i}}\approx1+f(it_{1}w^{\prime}_{i}),$$ $$(1-f)+fe^{i(t_{2}+t_{1}w^{\prime}_{e})}\approx(1-f)+fe^{it_{2}}.$$
$$(73)$$
$$\begin{array}{l}{(74)}\\ {(75)}\end{array}$$

18 524 Substituting back:

$$\phi_{M_{t n},X_{t n c}}(t_{1},t_{2})\approx\prod_{i\neq c}^{}(1+f i t_{1}w_{i}^{\prime})\cdot\left[(1-f)+f e^{i t_{2}}\right].$$

525 Using Equation 59, Equation 72 and Taylor expansion, the product of the characteristic functions for 526 the two distributions is:

ϕXtnc (t2)ϕMtn (t1) = (1 − f + feit2)Y C i=1 (1 − f + feit1w ′ i ) = (1 − f + feit2)Y C i=1 (1 + f it1w ′ i) = (1 − f + feit2)(1 + f it1w ′ c )Y i̸=c (1 + f it1w ′ i ) = (1 − f + feit2)Y i̸=c (1 + f it1w ′ i)
$$(76)$$

$$(77)$$
$$(78)$$  $$(79)$$
$$=\phi_{M_{t n},X_{t n c}}(t_{1},t_{2})$$
= ϕMtn,Xtnc (t1, t2) (77)
527 Thus, the joint characteristic function factorizes into the product of the marginal characteristic 528 functions, which demonstrates that Mtn and Xtnc are asymptotically independent as C → ∞. 529 **Proposition B.4.** If b ≈ 0, a ≥ 1, and the firing rate f is relatively small value, the PM output Ytnc 530 *satisfies:*

$$\mathbb{E}(Y_{t n c})\approx{\sqrt{\frac{f(1-f)}{2\pi}}}\,\mathbb{E}(X_{t n c})$$ $$V a r(Y_{t n c})\approx{\frac{f(\pi-f)}{2\pi}}\,V a r(X_{t n c})$$

531 *Proof.* For convenience, we denote:

For convenience, we define  $ \mu=\sum_{c=1}^C w_c^\prime f,\quad\sigma^2=\sum_{c=1}^C w_c^\prime f(1-f)=f(1-f),\quad M_{tn}^\prime=\text{clamp}(M_{tn},b,a)$. 
532 According to Lemma B.2, the input distribution satisfies:

$$(81)$$

Mtn ∼ N (*µ, σ*2). (81)
The expectation of the clamped variable M′tnc 533 is:

$$\mathbb{E}(M^{\prime}_{tn})=\int_{-\infty}^{\infty}xf(x)dx$$ $$=\frac{1}{\sqrt{2\pi\sigma^{2}}}\int_{0}^{a}x\exp\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)dx+\frac{a}{\sqrt{2\pi\sigma^{2}}}\int_{a}^{\infty}\exp\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)dx.\tag{82}$$
For the first term, let t = (x − µ) 2 534 , if µ ≈ 0, then: 1 √2πσ2 Z a 0 x exp − (x − µ) 2 2σ 2 dx =1 2 √2πσ2 Z (a−µ) 2 µ2 exp −t 2σ 2 dt +µ √2πσ2 Z a 0 exp − (x − µ) 2 2σ 2 dx =−σ √2πσ  exp −t 2σ 2 (a−µ) 2 µ2 + µ Φ a − µ σ − Φ −µ σ  ≈σ √2π 1 − exp − a 2 2σ 2  . (83)
$$(83)$$
535 where Φ(x) is the CDF of the standard normal distribution. The second term in the expectation is
536 straightforward:
$${\frac{a}{\sqrt{2\pi\sigma^{2}}}}\int_{a}^{\infty}\exp\left(-{\frac{(x-\mu)^{2}}{2\sigma^{2}}}\right)d x={\frac{a}{\sqrt{2\pi\sigma^{2}}}}\int_{a-\mu}^{\infty}\exp\left(-{\frac{t^{2}}{2\sigma^{2}}}\right)d t,$$ multi-distribution function (CDF) again. 
537 Using the cumulative distribution function (CDF) again:

$$\begin{split}\frac{a}{\sqrt{2\pi\sigma^{2}}}\int_{a-\mu}^{\infty}\exp\left(-\frac{t^{2}}{2\sigma^{2}}\right)dt&=a\left(1-\Phi\left(\frac{a-\mu}{\sigma}\right)\right)\\ &\approx a\left(1-\Phi\left(\frac{a}{\sigma}\right)\right)\end{split}$$
$$(84)$$
$$(85)$$

The Φ( aσ
) and exp(−
a 2 538 2σ2 ) function decay rapidly as σ decreases. Now, combining the results from 539 the two integrals, we have:

s, we have:  $\begin{gathered}\mathbb{E}(M_{tn}')=\frac{\sigma}{\sqrt{2\pi}}-\frac{\sigma}{\sqrt{2\pi}}\exp\left(-\frac{a^2}{2\sigma^2}\right)+a\left(1-\Phi\left(\frac{a-\mu}{\sigma}\right)\right)\\ \approx\frac{\sigma}{\sqrt{2\pi}}\end{gathered}$
Based on B.3, we calculate the expectation and variance of M′2 tn 540 :

 - The problem are given by means $\mathbb{E}(M_{tn}^2)$: $$\mathbb{E}(M_{tn}^{\prime2})=\int_{-\infty}^{\infty}x^2f(x)dx$$ $$=\frac{1}{\sqrt{2\pi\sigma^2}}\int_0^a x^2\exp\left(-\frac{x^2}{2\sigma^2}\right)dx+a^2\cdot\int_a^{\infty}f(x)dx.$$  a first term using integration by parts, I got: ... 
$$(86)$$
$$(87)$$
$$(88)$$
$$(89)$$
541 We calculate the first term using integration by parts. Let:

$$u=x,\quad d v=x\exp\left(-{\frac{x^{2}}{2\sigma^{2}}}\right)d x,\quad d u=d x,\quad v=-\sigma^{2}\exp\left(-{\frac{x^{2}}{2\sigma^{2}}}\right).$$
. (88)
542 Then:

$${\frac{1}{2\pi\sigma^{2}}}\int_{0}^{a}x^{2}\exp\left(-{\frac{x^{2}}{2\sigma^{2}}}\right)d x$$ $${\frac{1}{\sqrt{2\pi\sigma^{2}}}}\left(\left[-\sigma^{2}x\exp\left(-{\frac{x^{2}}{2\sigma^{2}}}\right)\right]_{0}^{a}+\sigma^{2}\int_{0}^{a}\exp\left(-{\frac{x^{2}}{2\sigma^{2}}}\right)d x\right)$$ $${\frac{1}{\sqrt{2\pi\sigma^{2}}}}\left(-\sigma^{2}a\exp\left(-{\frac{a^{2}}{2\sigma^{2}}}\right)+\sigma^{2}\int_{0}^{a}\exp\left(-{\frac{x^{2}}{2\sigma^{2}}}\right)d x\right).$$  In this article we shall talk about the integral. 
$$\frac{1}{\sqrt{2\pi\sigma^{2}}}$$ $$=\frac{1}{\sqrt{2\pi}}$$ $$=\frac{1}{\sqrt{2\pi}}$$
$$(90)$$
$$(91)$$

543 The remaining integral is a standard normal distribution integral:

 $ \frac{\sigma^2}{\sqrt{2\pi\sigma^2}}\int_0^a\exp\left(-\frac{x^2}{2\sigma^2}\right)dx=\sigma^2\left(\Phi\left(\frac{a}{\sigma}\right)-\frac{1}{2}\right),$  is of the standard normal distribution. 
544 where Φ(x) is the CDF of the standard normal distribution. 545 Substituting (90) into (89):

ing (90) into (89):  ${\frac{1}{\sqrt{2\pi\sigma^2}}\int_0^a x^2\exp\left(-\frac{x^2}{2\sigma^2}\right)dx=\frac{-a\sigma}{\sqrt{2\pi}}\exp\left(-\frac{a^2}{2\sigma^2}\right)+\sigma^2\left(\Phi\left(\frac{a}{\sigma}\right)-\frac{1}{2}\right).}$  but not is that's of the second distribution. 
. (91)
546 The second term is the tail of the normal distribution:
$$\int_{a}^{\infty}f(x)dx=\Phi\left(-\frac{a}{\sigma}\right),\tag{1}$$
547 we have:

$$a^{2}\cdot\int_{a}^{\infty}f(x)d x=a^{2}\Phi\left(-{\frac{a}{\sigma}}\right).$$
$$(92)$$
$$(93)$$