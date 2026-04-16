# **TDFormer: A Top-Down Attention-Controlled** **Spiking Transformer**

**Anonymous Author(s)**
Affiliation
Address
```
                    email

```

**Abstract**

1 Traditional spiking neural networks (SNNs) can be viewed as a combination of

2 multiple subnetworks with each running for one time step, where the parameters

3 are shared, and the membrane potential serves as the only information link between

4 them. However, the implicit nature of the membrane potential limits its ability

5 to effectively represent temporal information. As a result, each time step cannot

6 fully leverage information from previous time steps, seriously limiting the model’s

7 performance. Inspired by the top-down mechanism in the brain, we introduce

8 TDFormer, a novel model with a top-down feedback structure that functions hi
9 erarchically and leverages high-order representations from earlier time steps to

10 modulate the processing of low-order information at later stages. The feedback

11 structure plays a role from two perspectives: 1) During forward propagation, our

12 model increases the mutual information across time steps, indicating that richer

13 temporal information is being transmitted and integrated in different time steps. 2)

14 During backward propagation, we theoretically prove that the feedback structure

15 alleviates the problem of vanishing gradients along the time dimension. We find

16 that these mechanisms together significantly and consistently improve the model

17 performance on multiple datasets. In particular, our model achieves state-of-the-art

18 performance on ImageNet with an accuracy of 86.83%.

19 **1** **Introduction**

20 Spiking Neural Networks (SNNs) are more energy-efficient and biologically plausible than traditional

21 artificial neural networks (ANNs) [1]. Transformer-based SNNs combine the architectural advantages

22 of Transformers with the energy efficiency of SNNs, resulting in a powerful and efficient models

23 that have attracted increasing research interest in recent years [2, 3, 4, 5, 6]. However, there is

24 still a big performance gap between existing SNNs and ANNs. This is because SNNs represent

25 information using binary spike activations, whereas ANNs use floating-point numbers, resulting in

26 reduced representational capacity and degraded performance. Moreover, the non-differentiability of

27 spikes hinders effective training with gradient-based methods.

28 In traditional SNNs, a common approach to increase representational capacity is to expand the

29 time step _T_ . However, SNNs trained with direct coding and standard learning methods [7] lack

30 structural mechanisms for temporal adaptation. Temporal information is solely conveyed through

31 membrane potential dynamics, while the network architecture, parameters, and inputs remain fixed

32 across time steps. This reliance on membrane dynamics imposes two fundamental limitations. First,

33 temporal information can only be expressed when spikes are fired, yet firing rates are typically low

34 across layers, restricting the bandwidth of information flow. Moreover, the cumulative nature of

35 membrane potentials leads to loss of temporal detail, as earlier spike patterns are summed. Second,

36 temporal gradients must propagate solely through membrane potentials, which can result in vanishing

Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

Figure 1: Visualization of mutual information matrices of features across time steps on ImageNet.
The left panel shows the baseline model; the right panel shows the model incorporating feedback
connections. A higher level of mutual information suggests that the model captures more consistent
and temporally dependent features across time steps

37 gradients[8, 9]. We further confirm these limitations through temporal correlation analysis shown

38 in Figure 1, which demonstrates the limited representational capacity of membrane potentials, and

39 theoretical derivation in appendix B.3.

40 Previous work has been done to enhance the ability of SNNs to represent temporal information, e.g.,

41 by initializing the membrane potential and altering the surrogate gradients and dynamics equations

42 [10, 11, 12]. Furthermore, some approaches have incorporated the dimension of time into attention

43 mechanisms, resulting in time complexity that scales linearly with the number of simulation time steps

44 [13]. However, structural mechanisms to facilitate information flow across multiple time steps remain

45 largely unexplored. We argue that adding connections between different time steps has the following

46 two benefits: First, in forward propagation, such connections help the model better leverage features

47 from previous time steps. Second, in backpropagation, structural connections support gradient flow

48 and help mitigate vanishing gradients caused by the membrane potential dynamics.

49 While traditional SNNs rely on bottom-up signal propagation, top-down mechanisms are prevalent in

50 the brain, especially between the prefrontal and visual cortices [14, 15, 16, 17], as shown in Figure 2.

51 These mechanisms are fundamental to how the brain incrementally acquires visual information over

52 time, with higher-level cognitive processes guiding the extraction of lower-level sensory features,

53 and prior knowledge informing the interpretation and refinement of new sensory input. Inspired

54 by top-down mechanisms, we introduce TDFormer, a Transformer-based SNN architecture that

55 incorporates a top-down feedback structure to improve temporal information utilization. Our main

56 contributions can be summarized as follows:

57 - We identify structural limitations in traditional SNNs, showing that features across time steps

58 exhibit weak mutual information, indicating insufficient temporal integration and utilization.

59 - We propose TDFormer, a Transformer-based SNN with a novel top-down feedback structure.

60 We show that the proposed structure improves temporal information utilization, and provide

61 theoretical analysis showing it mitigates vanishing gradients along the temporal dimension.

62 - We demonstrate state-of-the-art performance across multiple benchmarks with minimal

63 energy overhead, achieving ANN-level accuracy on ImageNet while preserving the efficiency

64 of SNNs.

65 **2** **Related Works**

66 **2.1** **Transformer-based SNNs**

67 Spikformer [2] presented the first Transformer architecture based on SNNs, laying the groundwork for

68 spike-based self-attention mechanisms. Spike-driven TransformerV1 [5] introduced a spike-driven

2

69 mechanism to effectively process discrete-time spike signals and employed stacked transformer layers

70 to capture complex spatiotemporal features. Built on [5], Spike-driven TransformerV2 [6] enhanced

71 the spike-driven mechanism and added dynamic weight adjustment to improve adaptability and

72 accuracy in processing spike data. SpikformerV2 [18] was specifically optimized for high-resolution

73 image recognition tasks, incorporating an improved spike encoding method and a multi-layer self
74 attention mechanism. SpikeGPT [19] proposed an innovative combination of generative pre-trained

75 Transformers with SNNs. SGLFormer [20] enhanced feature representations by effectively capturing

76 both global context and local details.

77 **2.2** **Models with Top-Down Mechanisms**

78 Unlike bottom-up processes that are driven by sensory stimuli, top-down attention is governed

79 by higher cognitive processes such as goals, previous experience, or prior knowledge[21]. This

80 mechanism progressively acquires information by guiding the focus of attention to specific regions

81 or features of the visual scene. It can be seen as a feedback loop where higher-level areas provide

82 signals that modulate the processing of lower-level sensory inputs, ensuring that the most relevant

83 information is prioritized.

84 Many works have explored top-down attention mechanisms to improve model performance in

85 traditional ANNs. For example, Zheng et al. [21] proposed FBTP-NN, which integrates bottom-up

86 and top-down pathways to enhance visual object recognition, where top-down expectations modulate

87 neuron activity in lower layers [21]. Similarly, Anderson et al. introduced a model combining bottom
88 up and top-down attention for image captioning and visual question answering, where top-down

89 attention weights features based on task context [22]. Shi et al. introduced a top-down mechanism

90 for Visual Question Answering (VQA), where high-level cognitive hypotheses influence the focus

91 on relevant scene parts [23]. Finally, Abel and Ullman proposed a network that combines back
92 propagation with top-down attention to adjust gradient distribution and focus on important features

93 [24].

94 **3** **Preliminaries**

95 **3.1** **The Spiking Neuron**

96 The fundamental distinction between SNNs and ANNs lies in their neuronal activation mechanisms.

97 Drawing on established research [2, 4, 5, 3], we select the Leaky Integrate-and-Fire (LIF) [25] neuron

98 model as our primary spike activation unit. LIF neuron dynamics can be formulated by:

_V_ [ _t_ ] = _H_ [ _t_ ](1 _−_ _S_ [ _t_ ]) + _V_ reset _S_ [ _t_ ] _,_ (1)

_H_ [ _t_ ] = _V_ [ _t −_ 1] + [1] (2)

_τ_ [(] _[X]_ [[] _[t]_ []] _[ −]_ [(] _[V]_ [ [] _[t][ −]_ [1]] _[ −]_ _[V]_ [reset][))] _[,]_

_S_ [ _t_ ] = Θ( _H_ [ _t_ ] _−_ _V_ th) _,_ (3)

99 where _V_ reset is the reset potential. When a spike is generated, _S_ [ _t_ ] = 1, the membrane potential _V_ [ _t_ ]

100 is reset to _V_ reset; otherwise, it remains at the hidden membrane potential _H_ [ _t_ ]. Moreover, _τ_ represents

101 the membrane time constant, and the input current _X_ [ _t_ ] is decay-integrated into _H_ [ _t_ ].

102 **3.2** **Spike-Based Self-Attention Mechanisms**

103 A critical challenge in designing spike-based self-attention is eliminating floating-point matrix

104 multiplication in Vanilla Self-Attention (VSA) [26], which is crucial for utilizing the additive

105 processing characteristics of SNNs.

106 **Spiking Self-Attention** (SSA) Zhou et al. [2] first leveraged spike dynamics to replace the softmax

107 operation in VSA, thereby avoiding costly exponential and division calculations, and reducing energy

108 consumption. The process of SSA is as follows:

_Is_ = _SN_ ( _BN_ ( _XWI_ )) _, I_ _∈{Q, K, V },_ (4)

SSA( _Qs, Ks, Vs_ ) = _SN_ ( _QsKs_ _[⊤][V][s]_ _[∗]_ _[s]_ [)] _[,]_ (5)

109 where _W_ _∈_ R _[T][ ×][N]_ _[×][D]_ denotes a learnable weight matrix, _Is_ represents the spiking representations of

110 query _Qs_, key _Ks_, and value _Vs_ . Here, _SN_ ( _·_ ) denotes the LIF neuron, and _s_ is a scaling factor.

3

111 **Spike-Driven Self-Attention** (SDSA) Yao et al. [5, 6] improved the SSA mechanism by replacing

112 the matrix multiplication with the Hadamard product and computing the attention via column-wise

113 summation, effectively utilizing the additive properties of SNNs. The first version of SDSA [5] is as

114 follows:

SDSA1( _Qs, Ks, Vs_ ) = _Qs ⊗SN_ (SUMc( _Ks ⊗_ _Vs_ )) _,_ (6)
115 where _⊗_ denotes the Hadamard product, SUMc( _·_ ) represents the column-wise summation. Further
116 more, the second version of SDSA [6] is described as follows:

SDSA2( _Qs, Ks, Vs_ ) = _SN s_ (( _QsKs_ _[⊤]_ [)] _[V][s]_ [)] _[,]_ (7)
117 where _SN s_ denotes a spiking neuron with a threshold of _s · V_ th. **Q-K Attention** (QKA) The work

118 in [3] reduces the computational complexity from quadratic to linear by utilizing only the query

119 and key. QKA can be further divided into two variants: Q-K Token Attention (QKTA) and Q-K

120 Channel Attention (QKCA). The formulations for QKTA and QKCA are provided in Equations 8

121 and 9, respectively:

_D_

QKTA( _Qs, Ks_ ) = _SN_ (

_Qs_ ( _i, j_ )) _⊗_ _Ks,_ (8)

_i_ =0

_N_

QKCA( _Qs, Ks_ ) = _SN_ (

_Qs_ ( _i, j_ )) _⊗_ _Ks,_ (9)

_j_ =0

122 where _N_ denotes the token number, _D_ represents the channel number.

123 **4** **Method**

124 In this section, we introduce TDFormer, a Transformer-based SNN model featuring a top-down

125 feedback structure. We describe its architecture, including the division into sub-networks for feed
126 back processing. We theoretically show that the attention module prior to the LIF neuron in the

127 feedback pathway exhibits lower variance compared to SSA and QKTA, and we provide guidance

128 for hyperparameter selection. Finally, we introduce the training loss and inference process. Detailed

129 mathematical derivations are provided in appendix B.

130 **4.1** **TDFormer Architecture**

131 This work is based on three backbones: SpikformerV1 [2], Spike-driven TransformerV1 [5] and

132 QKformer [3]. These can be summarized into a unified structure, as shown in Figure 2, which consists

133 of _Lc_ Conv-based SNN blocks, _Lt_ Transformer-based SNN blocks, and a classification head (CH).

134 Additionally, the Transformer-based SNN blocks incorporate spike-based self-attention modules and

135 Multi-Layer Perceptron (MLP) modules.

136 Apart from the backbone structure, the TDFormer architecture specifically introduces a top-down

137 pathway called TDAC that includes two modules: the control module (CM) and the processing

138 module (PM), as shown in Figure 2.

139 Viewing traditional SNNs as a sequence of _T_ = 1 sub-networks with shared parameters and temporal

140 dynamics governed by membrane potentials, we propose two approaches to introducing the top
141 down pathway. The first adds recurrent feedback connections between these fine-grained _T_ = 1

142 sub-networks, enabling temporal context to propagate backward through time. The second adopts

143 a coarser temporal resolution by dividing a sequence (e.g., _T_ = 4) into fewer segments (e.g., two

144 _T_ = 2 blocks). Importantly, the additional power overhead introduced by both schemes remains

145 minimal. Detailed analysis of power consumption is provided in appendix C.1. Both approaches can

146 be expressed in the following unified formulation:

       -       - ��
_H_ 1 = Ftr CM _Sbu_ [(1)] _[,]_ [ ∅] _H_ 1 _∈{_ 0 _,_ 1 _}_ _[T][ ×][N]_ _[×][C]_ _, Sbu_ [(1)] _[∈{]_ [0] _[,]_ [ 1] _[}][T][ ×][H][×][W][ ×][C]_ (10)

_Std_ [(1)] [= PM(] _[H]_ [1][)] _Std_ [(1)] _[∈{]_ [0] _[,]_ [ 1] _[}][T][ ×][N]_ _[×][C][, H]_ [1] _[∈{]_ [0] _[,]_ [ 1] _[}][T][ ×][N]_ _[×][C]_ (11)

       -       - ��
_Hn_ = Ftr CM _Sbu_ [(] _[n]_ [)] _[, S]_ _td_ [(] _[n][−]_ [1)] _Sbu_ [(] _[n]_ [)] _[∈{]_ [0] _[,]_ [ 1] _[}][T][ ×][H][×][W][ ×][C][, n]_ [ = 1] _[ . . . N]_ (12)

_Std_ [(] _[n]_ [)] [= PM(] _[H][n]_ [)] _Std_ [(] _[n]_ [)] _[∈{]_ [0] _[,]_ [ 1] _[}][T][ ×][N]_ _[×][C][, n]_ [ = 1] _[ . . . N]_ (13)

_On_ = CH( _Hn_ ) _On_ _∈{_ 0 _,_ 1 _}_ _[T][ ×][L]_ _, Hn_ _∈{_ 0 _,_ 1 _}_ _[T][ ×][N]_ _[×][C]_ _, n_ = 1 _. . . N_ (14)

4

Figure 2: Overview of the TDFormer architecture. (a) Overall design inspired by top-down pathways
in the brain, mimicking feedback from the prefrontal cortex to the visual cortex for temporal modulation in SNNs; (b) and (c) Detailed structures of the processing and control modules; (d) Information
flow within the subnetwork, highlighting processing of feedback signals; (e) Four processing module
variants, labeled v1–v4.

147 In the above formulation, _Sbu_ [(] _[n]_ [)] [denotes the bottom-up input at time step] _[ n]_ [, while] _[ S]_ _td_ [(] _[n][−]_ [1)] represents

148 the top-down feedback from the previous step. CM is a control module that integrates bottom-up and

149 top-down signals, and Ftr denotes the Transformer-based processing unit. The processing module

150 PM generates the current feedback signal _Std_ [(] _[n]_ [)] [from the high-level representation] _[ H][n]_ [, and][ CH][ maps]

151 _Hn_ to the final output _On_, where _N_ denotes the number of sub-networks. The bottom part of Figure

152 2 illustrates the feedback information flow between sub-networks.

153 **For** **the** **control** **module** **(CM),** CM derives the query _Q_, key _K_, and value _V_ vectors from the

154 bottom-up information _Sbu_ and the top-down information _Std_ . In more detail, _Std_ facilitates attention

155 correction by controlling the attention map. The CM can be formulated as follows:

_Q, K, V_ = _CM_ ( _Sbu, Std_ ) _,_ (15)
_K_ = _SN_ (BN(TokenMix (( _Sbu, Std_ )))) _,_ (16)
_Q_ = _SN_ (BN(Linear( _Sbu_ ))) _, V_ = _SN_ (BN(Linear( _Sbu_ ))) _._ (17)

156 We choose concatenation along the channel dimension as the default token mixer, which allows us

157 to combine the features of the current time step with those from previous time steps, and use the

158 fused information to dynamically adjust the self-attention map. After passing through the CM, the

159 query _Q_, key _K_ and value _V_ vectors are fed into the self-attention module to obtain the top-down

160 attention map. To prevent the fusion of top-down information from altering the distribution of _K_

161 in the self-attention computation, we first normalize the combined features, and then apply spike

162 discretization before computing self-attention. Ablation studies on different CM variants are provided

163 in the appendix C.2.

5

164 **The processing module (PM)** PM includes both channel-wise token mixer and spatial-wise token

165 mixer [27]. The feature enhancement component enhances the original spiking feature maps **X**

166 by learning channel-wise **W** _c_ and computing spatial-wise attention maps **M** spatial. This attention

167 mechanism requires very few parameters and has a time complexity of _O_ ( _ND_ ). This operation is

168 represented as:

**M** spatial( _t, n_ ) =

_C_

- **W** _c ·_ **X** _t,n,c,_ (18)

_c_ =1

**M** spatial = clamp ( **M** spatial _, b, a_ ) _._ (19)

169 where **X** _t,n,c_ represents the spiking activation at time _t_, spatial position _n_ (corresponding to the

170 2D coordinate ( _h, w_ ) in the feature map), and channel _c_ . Here, _a_ and _b_ are hyperparameters. We

171 theoretically derive their effects on the PM output, and the details are given in appendix B.2. The

172 spatial attention map **M** spatial weights the spiking feature map **X** via element-wise multiplication,

173 with broadcasting over the channel dimension:

**O** = _SN_ ( **X** _⊙_ **M** spatial) _._ (20)

174 The attention embedding spaces are different across layers, and we aim to use a PM variants to

175 align the top-down information with the embedding spaces of different layers. We explored four PM

176 variants that serve as the channel-wise token mixer, which are illustrated in Figure 2.

177 We introduce a clamp operation in the attention module to enforce a strict upper bound on the variance

178 of the attention map which is formally stated in Proposition 4.1. Excessive variance can lead to

179 gradient vanishing, as gradients in spiking neurons are only generated near the firing threshold of

180 the membrane potential. Outside this narrow region, the gradient tends to vanish. Furthermore, high

181 variance may introduce outliers, resulting in significant quantization errors during spike generation.

182 The effect of the clamp operation on the gradient is shown in the Figure appendix C.2.

183 **Proposition 4.1.** _The upper bound Var_ ( _Ytnc_ ) _for the_ **X** _⊙_ **M** _spatial is given as follows:_

_a_ [2] ( _f_ [2] _−_ _f_ + [1]

2 _[,]_ _if_ 0 _≤_ _f_ _≤_ _[a]_ 2 [+] _a_ _[b]_

[1] _[b]_ [2]

2 [) +] _[ ab]_ [(1] _[ −]_ [2] _[f]_ [) +] 2

(21)
2 [+] _a_ _[b]_ _[≤]_ _[f]_ _[≤]_ [1] _[,]_

2 _a_ _[,]_

_Var_ ( _Ytnc_ ) =




_a_ [2] +2 _ab_ + _b_ [2] _−_ 4 _fab_



4 _b_ _−_ 4 _fab_ _,_ _if_ _[a]_ 2 [+] _a_ _[b]_

184 _where we assume each_ **X** _t,n,c is independent random variable Xtnc_ _∼_ _Bernoulli_ ( _f_ ) _, with f_ _as the_

185 _firing rate._

186 Additionally, the clamp operation eliminates the need for scaling operations in attention mechanisms

187 (e.g., QK product scaling), simplifying computations, reducing complexity, and improving energy

188 efficiency in hardware implementations. The detailed proofs of this proposition are provided in

189 appendix B.1.

190 **4.2** **Loss Function**

191 The loss of the TDFormer can be formulated as follows:

_N_

- _αn_ = 1 _,_ 0 _≤_ _αn_ _≤_ 1 _._ (22)

_n_ =1

_L_ TDFormer =

_N_

- _αnL_ ( _y, On_ ) _,_

_n_ =1

192 Here, _αn_ are hyperparameters. To maintain the overall loss scale, we apply a weighted average over

193 the losses from all _N_ stages, assigning a larger weight to the final output loss. This is because we

194 believe that the receptive field in the temporal dimension increases as time progresses. Since the

195 earlier stages lack feedback from future steps, their outputs are less accurate and thus subject to

196 weaker supervision. By contrast, the final stage benefits from a larger temporal receptive field due to

197 feedback, making its output more reliable. Therefore, during testing, only the output from the last

198 sub-network is used for evaluation.

199 **4.3** **Top-down feedback enhances temporal dependency**

200 Top-down feedback enhances temporal dependency from two perspectives. First, from the forward

201 propagation perspective, we compute the mutual information matrix between features at different time

6

Table 1: Comparison with the baseline and previous work on ImageNet. The result in bold indicates
superior performance compared to the baseline. SOTA is marked with *, previous SOTA with #. The
default PM variant is v1.

ImageNet
Methods Spike Architecture

Time
Acc (%)
Step [Power (mJ) Param (M)]

ViT [28] ✗ ViT-B/16(384 [2] ) 1 254.84 86.59 77.90

DeiT [29] ✗ DeiT-B(384 [2] ) 1 254.84 86.59 83.10

Swin [30] ✗ Swin Transformer-B(384 [2] ) 1 216.20 87.77 84.50

Spikingformer [4] ✓ Spikingformer-8-768 4 13.68 66.34 75.85

✓ Spikformer-8-512 4 11.58 29.68 73.38
SpikformerV1 [2]
✓ Spikformer-8-768 4 21.48 66.34 74.81

✓ Meta-SpikeFormer-8-384 4 32.80 31.30 77.20
SDTV2 [6]
✓ Meta-SpikeFormer-8-512 4 52.40 55.40 80.00

✓ E-Spikeformer 8 30.90 83.00 84.00
E-Spikeformer [31] ✓ E-Spikeformer 8 54.70 173.00 85.10
✓ E-Spikeformer 8       - 173.00 86.20 #

✓ HST-10-768 (224 [2] ) 4 38.91 64.96 84.22
QKFormer [3] ✓ HST-10-768 (288 [2] ) 4 64.27 64.96 85.20
✓ HST-10-768 (384 [2] ) 4 113.64 64.96 85.65

TDFormer

✓ HST-10-768 (224 [2] ) 4 38.93 65.55 **85.37(+1.15)**
✓ HST-10-768 (288 [2] ) 4 64.39 65.55 **86.29(+1.09)**
✓ HST-10-768 (224 [2] ) 4 39.10 69.09 **85.57(+1.35)**
✓ HST-10-768 (288 [2] ) 4 64.45 69.09 **86.43 (+1.23)**
✓ HST-10-768 (384 [2] ) 4 113.79 69.09 **86.83 (+1.18)***

202 steps, as shown in Figure 1. Second, from the backward propagation perspective, we demonstrate that

203 introducing top-down feedback helps alleviate the problem of vanishing gradients along the temporal

204 dimension. We present the following theorem:

205 **Definition 4.2.** _ϵ_ _[l]_ ( _t_ ) is defined as the sensitivity of the membrane potential **H** _[l]_ ( _t_ + 1) to its previous

206 state **H** _[l]_ ( _t_ ), and is computed as:

_[l]_ [(] _[t]_ [ + 1)]

+ _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]
_∂_ **H** _[l]_ ( _t_ ) _∂_ **S** _[l]_ ( _t_ )

_ϵ_ _[l]_ ( _t_ ) _≡_ _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]

_[l]_ [(] _[t]_ [ + 1)] _∂_ **S** _[l]_ ( _t_ )

(23)
_∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ ) _[,]_

207 where _l_ indexes the layer.

208 **Theorem 4.3.** _We adopt the rectangular function as the surrogate gradient, following the setting_

209 _used in previous studies[8, 9, 12]._ _For a conventional SNN, the sensitivity of the membrane potential_

210 _is expressed as follows:_

_ϵ_ _[l]_ ( _t_ ) _jj_ = - 0 _,_ [1] 12 _[ϑ < H]_ _j_ _[l]_ [(] _[t]_ [)] _[ <]_ [3] 2

0 _,_ 2 _[ϑ < H]_ _j_ [(] _[t]_ [)] _[ <]_ 2 _[ϑ,]_

1 _−_ [1] _[,]_ _otherwise ._

2 2 (24)

_τ_ [1] _[,]_ _otherwise ._

211 _For SNN with top-down feedback structure, the sensitivity of the membrane potential can be expressed_

212 _as:_

_ϵ_ _[l]_ ( _t_ ) _jj_ =

- _∂φθ_ ( **S** _[l]_ ( _t_ ))

_∂θ_ **S** _[l]_ ( _t_ ) _,_ 2 _[ϑ < H]_ _j_ [(] _[t]_ [)] _[ <]_ 2 _[ϑ,]_

1 _−_ [1] _[,]_ _otherwise ._

_∂θ_ **S** ( **S** _[l]_ ( _t_ () _t_ )) _,_ 21 _[ϑ < H]_ _j_ _[l]_ [(] _[t]_ [)] _[ <]_ [3] 2

_[l]_ ( _t_ ) 2 _j_ 2 (25)

_τ_ [1] _[,]_ _otherwise ._

213 _where_ _ϑ_ _is_ _the_ _spike_ _threshold,_ _τ_ _is_ _a_ _time_ _constant_ _and_ _φθ_ _is_ _a_ _differentiable_ _feedback_ _function_

214 _parameterized by θ._

215 According to Equation 24, _ϵ_ _[l]_ ( _t_ ) becomes zero within an easily-reached interval, and outside that

216 interval, it is upper-bounded by a small value 1 _−_ _τ_ [1] [, since] _[ τ]_ [is typically close to 1 in practice[][32][,][ 33][,]

217 34, 9]. In contrast, our method allows non-zero gradients within this interval, and the _[∂φ]_ _∂_ _[θ]_ **S** [(] **[S]** _[l]_ ( _[l]_ _t_ [(] ) _[t]_ [))] can

interval, it is upper-bounded by a small value 1 _−_ [1]

34, 9]. In contrast, our method allows non-zero gradients within this interval, and the _[∂φ][θ]_ [(] **[S]** _[l]_ _[l]_ [(] _[t]_ [))]

7

Table 2: Comparison with the baselines and previous work on static datasets: CIFAR-10 and CIFAR100. Conventions align with those in Table 1. The default PM variant is v1.

CIFAR-10 CIFAR-100
Methods Time

[Architecture] Step Acc Acc
(%) (%)

STBP-tdBN [33] [ResNet-19] 4 92.92 70.86

TET [32] [ResNet-19] 4 94.44 74.47

SDTV1[5][SDT-2-512] 4 95.60 78.40

QKformer [3] [HST-4-384] 4 96.18 # 81.15 #

2 93.59 76.28
SpikformerV1 [2] [Spikformer-4-384]
4 95.19 77.86

2 93.65 75.29
SpikformerV1(ours)[Spikformer-4-384]
4 94.73 77.88

2 **94.17 (+0.52)** **75.79 (+0.50)**
TDFormer[Spikformer-4-384]
4 **95.11 (+0.38)** **77.99 (+0.11)**

SDTV1(ours)[SDT-2-256] 4 94.47 76.05
SDTV1(ours)[SDT-2-512] 4 95.78 79.15

TDFormer[SDT-2-256] 4 **94.61 (+0.14)** **76.23 (+0.18)**
TDFormer[SDT-2-512] 4 **96.07 (+0.29)** **79.67 (+0.52)**

TDFormer [HST-4-384] 4 **96.51 (+0.33)*** **81.45 (+0.30)***

exceed 1 _−_ [1]

218 exceed 1 _−_ _τ_ [.] [This property helps to alleviate the vanishing gradient problem along the temporal]

219 dimension. The detailed proof is provided in the appendix B.3.

220 **5** **Experiments**

221 We evaluate our models on several datasets: CIFAR-10 [35], CIFAR-100 [35], CIFAR10-DVS [36],

222 DVS128 Gesture [37], ImageNet [38], CIFAR-10C [39] and ImageNet-C [39]. For the smaller

223 datasets, we employ the feedback pathway on SpikformerV1 [2], Spike-driven TransformerV1 [5]

224 and QKformer[3], experimenting with different configurations tailored to each dataset. For the large
225 scale datasets, we utilize QKformer[3] as baselines. Specific implementation details are provided in

226 appendix A.

227 **5.1** **Experiments on ImageNet**

228 Table 1 presents the results for the large-scale dataset ImageNet. The incorporation of top-down

229 feedback structure has demonstrated significant improvements on E-spikformer, which is the previous

230 SOTA model of SNNs. Notably, compared to QKFormer, increasing the model size by merely 0.02

231 million parameters and 0.59 millijoules of power consumption leads to a significant gain of 1.15%

232 in top-1 accuracy on the ImageNet dataset. Our model sets a new SOTA performance in the SNN

233 field. This milestone lays a solid foundation for advancing SNNs toward large-scale networks, further

234 bridging the gap between SNNs and traditional deep learning models. Furthermore, we calculate the

235 power of TDFormer following the method in [3], as detailed in Table 1. TDFormer results in a slight

236 increase in energy consumption due to the feedback structure, but it achieves superior performance

237 with minimal additional power usage. The detailed calculation of power consumption is provided in

238 the appendix C.1.

239 **5.2** **Experiments on Neuromorphic and CIFAR Datasets**

240 Table 3 presents the results for the neuromorphic datasets CIFAR10-DVS and DVS128 Gesture. Our

241 proposed TDFormer consistently outperforms the baselines across all experiments, except for the

242 Spiking Transformer-2-256 at a time step of 10. Furthermore, we achieve SOTA results, with an

243 accuracy of 85.83% on CIFAR10-DVS using the HST-2-256 (V1), marking a notable improvement

8

Table 3: Comparison with the baselines and previous work on the Neuromorphic Dataset. Conventions
align with those in Table 1. The default PM variant is v1.

CIFAR10-DVS DVS128 Gesture
Methods [Architecture]
Time Acc Time Acc
Step (%) Step (%)

STBP-tdBN [33] [ResNet-19] 10 67.80 40 96.90

DSR [40] [VGG-11] 10 77.30        -        
SDTV1 [5][SDT-2-256] 16 80.00 16 99.30 #

10 78.90 10 96.90
SpikformerV1 [2] [Spikformer-2-256]
16 80.90 16 98.30

10 79.90 10 96.20
Spikingformer [4] [Spikingformer-2-256]
16 81.30 16 98.30

Qkformer [3] [HST-2-256] 16 84.00 # 16 98.60

10 78.08                      -                      SpikformerV1(ours) [Spikformer-2-256]
16 79.40                      -                      

10 **78.90 (+0.82)**                      -                      TDFormer [Spikformer-2-256]
16 **81.70 (+2.30)**                      -                      

10 75.22 10 96.79
SDTV1(ours) [SDT-2-256]
16 77.07 16 97.98

10 75.05 (-0.17) 10 **96.92 (+0.13)**
TDFormer[SDT-2-256]
16 **77.45 (+0.38)** 16 **99.65 (+1.67)***
TDFormer[HST-2-256] 16 **85.83 (+1.83)*** 16 **98.96 (+0.36)**

244 of 1.83% compared to the previous SOTA model, QKformer. We also achieve 99.65% accuracy on

245 DVS128 Gesture using the Spiking Transformer-2-256 (V1) at 16 time steps.

246 In addition, the results for the static datasets CIFAR-10 and CIFAR-100 are summarized in Table 2.

247 Compared to the baselines, the proposed TDFormer consistently demonstrates significant performance

248 improvements across all experiments, with the exception of Spikformer-4-384 (V1) at time step

249 6. Furthermore, we achieve the SOTA performance, attaining 96.51% accuracy on CIFAR-10 and

250 81.45% on CIFAR-100 using the HST-2-256 (V1) at a time step of 4.

251 **5.3** **Model Generalization Analysis**

252 As reported in Table 5, we report results averaged over five random seeds for reliability. Our model

253 consistently improves performance across time steps and depths. To assess robustness, we evaluate

254 on the CIFAR-10C dataset with 15 corruption types. As shown in Table 7, the model equipped with

255 the TDAC module consistently achieves higher accuracy under various distortion settings.

256 Moreover, we provide a visualization analysis of the TDFormer attention modules on CIFAR-10C

257 and ImageNet-C. The specific results can be seen in Figure 4 and Figure 5 of the appendix C. We

258 find that after adding the TDAC module, the model focuses more on the targets and their surrounding

259 areas. This indicates that TDAC can filter noise and irrelevant information, allowing the model to

260 focus more on task-related information.

261 **6** **Conclusion**

262 In this study, we propose TDFormer, which integrates an adaptive top-down feedback structure into

263 Transformer-based SNNs, addressing a key limitation of temporal information utilization in existing

264 models by incorporating biological top-down mechanisms. The TDFormer model outperforms

265 traditional Transformer-based SNNs, achieving SOTA performance across all evaluated datasets. Our

266 work suggests that the top-down feedback structure could be a valuable component for Transformer
267 based SNNs and offers insights for future research into more advanced, biologically inspired neural

268 architectures that better mimic human cognition.

9

269 **References**

270 [1] Kai Malcolm and Josue Casco-Rodriguez. A comprehensive review of spiking neural networks:

271 Interpretation, optimization, efficiency, and best practices. _arXiv preprint arXiv:2303.10780_,

272 2023.

273 [2] Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng YAN, Yonghong Tian,

274 and Li Yuan. Spikformer: When spiking neural network meets transformer. In _The Eleventh_

275 _International Conference on Learning Representations_, 2023.

276 [3] Chenlin Zhou, Han Zhang, Zhaokun Zhou, Liutao Yu, Liwei Huang, Xiaopeng Fan, Li Yuan,

277 Zhengyu Ma, Huihui Zhou, and Yonghong Tian. Qkformer: Hierarchical spiking transformer

278 using qk attention. _arXiv preprint arXiv:2403.16552_, 2024.

279 [4] Chenlin Zhou, Liutao Yu, Zhaokun Zhou, Zhengyu Ma, Han Zhang, Huihui Zhou, and Yonghong

280 Tian. Spikingformer: Spike-driven residual learning for transformer-based spiking neural

281 network. _arXiv preprint arXiv:2304.11954_, 2023.

282 [5] Man Yao, JiaKui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo XU, and Guoqi Li. Spike
283 driven transformer. In _Thirty-seventh Conference on Neural Information Processing Systems_,

284 2023.

285 [6] Man Yao, JiaKui Hu, Tianxiang Hu, Yifan Xu, Zhaokun Zhou, Yonghong Tian, Bo XU, and

286 Guoqi Li. Spike-driven transformer v2: Meta spiking neural network architecture inspiring the

287 design of next-generation neuromorphic chips. In _The Twelfth International Conference on_

288 _Learning Representations_, 2024.

289 [7] Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, Yuan Xie, and Luping Shi. Direct training for spiking

290 neural networks: Faster, larger, better. In _Proceedings_ _of_ _the_ _AAAI_ _conference_ _on_ _artificial_

291 _intelligence_, volume 33, pages 1311–1318, 2019.

292 [8] Yongqi Ding, Lin Zuo, Mengmeng Jing, Pei He, and Hanpu Deng. Rethinking spiking neural

293 networks from an ensemble learning perspective. _arXiv preprint arXiv:2502.14218_, 2025.

294 [9] Qingyan Meng, Mingqing Xiao, Shen Yan, Yisen Wang, Zhouchen Lin, and Zhi-Quan Luo.

295 Towards memory-and time-efficient backpropagation for training spiking neural networks. In

296 _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 6166–6176,

297 2023.

298 [10] Hangchi Shen, Qian Zheng, Huamin Wang, and Gang Pan. Rethinking the membrane dynamics

299 and optimization objectives of spiking neural networks. _Advances_ _in_ _Neural_ _Information_

300 _Processing Systems_, 37:92697–92720, 2024.

301 [11] Wei Liu, Li Yang, Mingxuan Zhao, Shuxun Wang, Jin Gao, Wenjuan Li, Bing Li, and Weiming

302 Hu. Deeptage: Deep temporal-aligned gradient enhancement for optimizing spiking neural

303 networks. In _The Thirteenth International Conference on Learning Representations_ .

304 [12] Yulong Huang, Xiaopeng Lin, Hongwei Ren, Haotian Fu, Yue Zhou, Zunchang Liu, Biao Pan,

305 and Bojun Cheng. Clif: Complementary leaky integrate-and-fire neuron for spiking neural

306 networks. _arXiv preprint arXiv:2402.04663_, 2024.

307 [13] Donghyun Lee, Yuhang Li, Youngeun Kim, Shiting Xiao, and Priyadarshini Panda. Spiking

308 transformer with spatial-temporal attention. _arXiv preprint arXiv:2409.19764_, 2024.

309 [14] Charles D Gilbert and Wu Li. Top-down influences on visual processing. _Nature_ _reviews_

310 _neuroscience_, 14(5):350–363, 2013.

311 [15] Timothy J Buschman and Earl K Miller. Top-down versus bottom-up control of attention in the

312 prefrontal and posterior parietal cortices. _science_, 315(5820):1860–1862, 2007.

313 [16] John H Reynolds and David J Heeger. The normalization model of attention. _Neuron_, 61(2):168–

314 185, 2009.

10

315 [17] Maurizio Corbetta, Erbil Akbudak, Thomas E Conturo, Abraham Z Snyder, John M Ollinger,

316 Heather A Drury, Martin R Linenweber, Steven E Petersen, Marcus E Raichle, David C

317 Van Essen, et al. A common network of functional areas for attention and eye movements.

318 _Neuron_, 21(4):761–773, 1998.

319 [18] Zhaokun Zhou, Kaiwei Che, Wei Fang, Keyu Tian, Yuesheng Zhu, Shuicheng Yan, Yonghong

320 Tian, and Li Yuan. Spikformer v2: Join the high accuracy club on imagenet with an snn ticket.

321 _arXiv preprint arXiv:2401.02020_, 2024.

322 [19] Rui-Jie Zhu, Qihang Zhao, Guoqi Li, and Jason K Eshraghian. Spikegpt: Generative pre-trained

323 language model with spiking neural networks. _arXiv preprint arXiv:2302.13939_, 2023.

324 [20] Han Zhang, Chenlin Zhou, Liutao Yu, Liwei Huang, Zhengyu Ma, Xiaopeng Fan, Huihui Zhou,

325 and Yonghong Tian. Sglformer: Spiking global-local-fusion transformer with high performance.

326 _Frontiers in Neuroscience_, 18:1371290, 2024.

327 [21] Yuhua Zheng, Yan Meng, and Yaochu Jin. Object recognition using a bio-inspired neuron

328 model with bottom-up and top-down pathways. _Neurocomputing_, 74(17):3158–3169, 2011.

329 [22] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould,

330 and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question

331 answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_,

332 pages 6077–6086, 2018.

333 [23] Baifeng Shi, Trevor Darrell, and Xin Wang. Top-down visual attention from analysis by

334 synthesis. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_

335 _Recognition_, pages 2102–2112, 2023.

336 [24] Roy Abel and Shimon Ullman. Top-down network combines back-propagation with attention.

337 _arXiv preprint arXiv:2306.02415_, 2023.

338 [25] Wulfram Gerstner, Werner M Kistler, Richard Naud, and Liam Paninski. _Neuronal dynamics:_

339 _From single neurons to networks and models of cognition_ . Cambridge University Press, 2014.

340 [26] A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_,

341 2017.

342 [27] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and

343 Shuicheng Yan. Metaformer is actually what you need for vision. In _Proceedings_ _of_ _the_

344 _IEEE/CVF conference on computer vision and pattern recognition_, pages 10819–10829, 2022.

345 [28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,

346 Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.

347 An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint_

348 _arXiv:2010.11929_, 2020.

349 [29] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and

350 Hervé Jégou. Training data-efficient image transformers & distillation through attention. In

351 _International conference on machine learning_, pages 10347–10357. PMLR, 2021.

352 [30] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining

353 Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings_

354 _of the IEEE/CVF international conference on computer vision_, pages 10012–10022, 2021.

355 [31] Man Yao, Xuerui Qiu, Tianxiang Hu, Jiakui Hu, Yuhong Chou, Keyu Tian, Jianxing Liao,

356 Luziwei Leng, Bo Xu, and Guoqi Li. Scaling spike-driven transformer with efficient spike

357 firing approximation training. _IEEE Transactions on Pattern Analysis and Machine Intelligence_,

358 2025.

359 [32] Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu. Temporal efficient training of

360 spiking neural network via gradient re-weighting. _arXiv preprint arXiv:2202.11946_, 2022.

361 [33] Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li. Going deeper with directly-trained

362 larger spiking neural networks. In _Proceedings of the AAAI conference on artificial intelligence_,

363 volume 35, pages 11062–11070, 2021.

11

364 [34] Yufei Guo, Xinyi Tong, Yuanpei Chen, Liwen Zhang, Xiaode Liu, Zhe Ma, and Xuhui Huang.

365 Recdis-snn: Rectifying membrane potential distribution for directly training spiking neural net
366 works. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_,

367 pages 326–335, 2022.

368 [35] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report,

369 University of Toronto, 2009.

370 [36] Hongmin Li, Hanchao Liu, Xiangyang Ji, Guoqi Li, and Luping Shi. Cifar10-dvs: an event
371 stream dataset for object classification. _Frontiers in neuroscience_, 11:309, 2017.

372 [37] Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo,

373 Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, et al. A low

374 power, fully event-based gesture recognition system. In _Proceedings of the IEEE conference on_

375 _computer vision and pattern recognition_, pages 7243–7252, 2017.

376 [38] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large
377 scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern_

378 _recognition_, pages 248–255. Ieee, 2009.

379 [39] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common

380 corruptions and perturbations. _arXiv preprint arXiv:1903.12261_, 2019.

381 [40] Qingyan Meng, Mingqing Xiao, Shen Yan, Yisen Wang, Zhouchen Lin, and Zhi-Quan Luo.

382 Training high-performance low-latency spiking neural networks by differentiation on spike

383 representation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern_

384 _recognition_, pages 12444–12453, 2022.

385 [41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv_ _preprint_

386 _arXiv:1711.05101_, 2017.

387 [42] Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, and Guoqi Li. Integer-valued training and

388 spike-driven inference spiking neural network for high-performance and energy-efficient object

389 detection. In _European Conference on Computer Vision_, pages 253–272. Springer, 2024.

390 [43] Youngeun Kim, Joshua Chough, and Priyadarshini Panda. Beyond classification: Directly

391 training spiking neural networks for semantic segmentation. _Neuromorphic Computing and_

392 _Engineering_, 2(4):044015, 2022.

393 [44] Changze Lv, Jianhan Xu, and Xiaoqing Zheng. Spiking convolutional neural networks for text

394 classification. _arXiv preprint arXiv:2406.19230_, 2024.

12

395 **A** **Implementation Details**

396 **A.1** **Training Protocols**

397 We adopted the following training protocols:

398 - **Spike Generation** : We used a rate-based method for spike generation [2].

399 - **Data Augmentation and Training Duration** : SpikformerV1 experiments followed [2],

400 while Spike-driven TransformerV1 experiments followed [5], furthermore QKformer experi
401 ments followed the experimental setting in and [3].

402 - **Optimization** : We employed AdamW [41] as the optimizer for our experiments. The

403 learning rate was set to 3 _×_ 10 _[−]_ [4] for the Spike-driven TransformerV1. For SpikformerV1,

404 we used a learning rate of 5 _×_ 10 _[−]_ [4] on static datasets and 1 _×_ 10 _[−]_ [3] on neuromorphic

405 datasets. Additionally, we utilized a cosine learning rate scheduler to adjust the learning

406 rate dynamically during training. Specifically, for QKformer, we fine-tuned the pretrained

407 network with a base learning rate of 2 _×_ 10 _[−]_ [5] for 15 epochs, due to the high cost of direct

408 training on ImageNet using 4 time steps.

       - **Batch Size** : The batch sizes for different datasets and models are specified in Table 4.

Table 4: Batch sizes for different datasets and models.

**Dataset** **Model** **Batch Size**

CIFAR-10 and
CIFAR-100

CIFAR10-DVS and
DVS128 Gesture

SpikeformerV1 128
Spike-driven TransformerV1 64

SpikeformerV1 16
Spike-driven TransformerV1 16

ImageNet QKformer 57

409

410 **A.2** **Datasets**

411 Our experiments evaluated the performance and robustness of the TDFormer model using the

412 following datasets:

413 - **CIFAR-10:** This dataset contains 60,000 32 _×_ 32 color images divided into 10 classes [35].

414 - **CIFAR-100:** This dataset is similar to CIFAR-10 but includes 100 classes, providing a more

415 challenging classification task [35].

416 - **CIFAR10-DVS:** This is an event-based version of the CIFAR-10 dataset [36].

417 - **DVS128 Gesture:** This is an event-based dataset for gesture recognition with 11 classes

418 [37].

419 - **ImageNet:** This large-scale dataset contains over 1.2 million images divided into 1,000

420 classes [38].

421 - **CIFAR-10C:** This is a corrupted version of CIFAR-10 with 19 common distortion types,

422 used to assess robustness [39].

423 - **ImageNet-C:** This dataset is a corrupted version of ImageNet, designed similarly to CIFAR
424 10C [39].

425 **A.3** **Computational Environment**

426 **A.3.1** **Software Setup**

427 We utilized PyTorch version 2.0.1 with CUDA 11.8 support and SpikingJelly version 0.0.0.0.12 as

428 the primary software tools.

13

429 **A.3.2** **Hardware Setup.**

430 For the smaller dataset experiments, we utilized the following configuration:

431 - **Hardware Used:** NVIDIA L40S and L40 GPUs.

432 - **Configuration:** Single-GPU for each experiment.

433 - **Memory Capacity:** Each GPU is equipped with 42 GB of memory.

434 For the large-scale dataset (ImageNet) experiments, we employed the following setup:

435 - **Hardware Used:** NVIDIA H20 GPUs.

436 - **Configuration:** Eight-GPU for each experiment.

437 - **Memory Capacity:** Each GPU provides 96 GB of memory.

438 **A.4** **Random Seed**

439 To ensure the comparability of the results, we selected the same random seeds as those in the baseline

440 paper. To ensure robustness, we also conducted experiments with random seeds 0, 42, 2024, 3407

441 and 114514, averaging the results. Detailed results are presented in Table 5.

442 **B** **Mathematical Derivations**

443 **B.1** **Detailed proofs of the upper bound on PM output variance**

444 _Proof._ We assume that each **M** spatial( _t, n_ ) is an independent random variable _Mtn_ . Given that

445 _b ≤_ _Mtn_ _≤_ _a_, it follows that _b ≤_ E[ _Mtn_ ] _≤_ _a_ . Furthermore, when _Xtnc_ = 0, we have:

( _XtncMtn −_ _b_ )( _a −_ _XtncMtn_ ) _≥_ 0 _,_ (26)

446 which expands to:

_−_ ( _XtncMtn_ ) [2] + ( _a_ + _b_ )( _XtncMtn_ ) _−_ _ab ≥_ 0 _._ (27)

447 Taking the expectation on both sides yields:

E �( _XtncMtn_ ) [2][�] _≤_ ( _a_ + _b_ )E [ _XtncMtn_ ] _−_ _ab._ (28)

448 Using the Law of Total Variance, we can decompose the variance of _Ytnc_ as:

Var( _Ytnc_ ) = E[Var( _Ytnc|Xtnc_ )] + Var(E[ _Ytnc|Xtnc_ ]) _._ (29)

449 For the first term, the expectation of the conditional variance can be expressed as:

E[Var( _Ytnc|Xtnc_ )] = _f_ _·_ Var( _Ytnc|Xtnc_ = 1) + (1 _−_ _f_ ) _·_ Var( _Ytnc|Xtnc_ = 0) _._ (30)

450 For the second term, the variance of the conditional expectation can be expanded as:

Var(E[ _Ytnc|Xtnc_ ]) = E[E[ _Ytnc|Xtnc_ ] [2] ] _−_ E[E[ _Ytnc|Xtnc_ ]] [2] _._ (31)

451 By substituting the conditional probabilities, we have:

Var(E[ _Ytnc|Xtnc_ ]) = _f_ _·_ E[ _Ytnc|Xtnc_ = 1] [2] _−_ _f_ [2] _·_ E[ _Ytnc|Xtnc_ = 1] [2] _._ (32)

452 Combining the two terms, the total variance becomes:

Var( _Ytnc_ ) = _f_ _·_ Var( _Ytnc|Xtnc_ = 1) + ( _f_ _−_ _f_ [2] ) _·_ E[ _Ytnc|Xtnc_ = 1] [2] _._ (33)

453 From Equation 32, we define E[ _Ytnc|Xtnc_ = 1] = _µ_ . Substituting this definition, the variance can be

454 rewritten as:

Var( _Ytnc_ ) = _f_ _·_ (E[ _Ytnc_ [2] _[|][X][tnc]_ [= 1]] _[ −]_ _[µ]_ [2][) + (] _[f]_ _[−]_ _[f]_ [ 2][)] _[ ·][ µ]_ [2] _[.]_ (34)

455 Using the constraints _b ≤_ _Mtn_ _≤_ _a_, we have the following bound for Var( _Ytnc|Xtnc_ = 1):

Var( _Ytnc|Xtnc_ = 1) _≤_ ( _a_ + _b_ ) _µ −_ _ab −_ _µ_ [2] _._ (35)

14

456 By substituting this into the total variance expression, the upper bound of Var( _Ytnc_ ) becomes:

Var( _Ytnc_ ) _≤_ _f_ _·_ (( _a_ + _b_ ) _µ −_ _ab −_ _µ_ [2] ) + ( _f_ _−_ _f_ [2] ) _· µ_ [2]

   _≤−f_ [2] _·_ _µ −_ _[a]_ [ +] _[ b]_

2 _f_

�2
+ _[a]_ [2][ + 2] _[ab]_ [ +] _[ b]_ [2] _[ −]_ [4] _[fab]_ _._ (36)

4

457 Next, we will prove that this upper bound can be achieved with equality under specific conditions.

458 **Case 1:** When _[a]_ 2 [+] _a_ _[b]_ _[≤]_ _[f]_ _[≤]_ [1][, we assume that:]

E[ _Ytnc|Xtnc_ = 1] = _[a]_ [ +] _[ b]_ _Mtn_ = _a_ or _b._ (37)

2 _f_ _[,]_

459 Here, _Mtn_ is a binary random variable, taking the value _a_ with probability _p_ and the value _b_ with

460 probability 1 _−p_ . Using this assumption, we can express the conditional expectation E[ _Ytnc|Xtnc_ = 1]

461 as:

E[ _Ytnc|Xtnc_ = 1] = _pa_ + (1 _−_ _p_ ) _b._ (38)

462 Substituting E[ _Ytnc|Xtnc_ = 1] = _[a]_ 2 [+] _f_ _[b]_ [into the above equation, we solve for] _[ p]_ [:]

[ +] _[ b]_

_⇒_ _p_ = _[a]_ [ +] _[ b][ −]_ [2] _[bf]_
2 _f_ 2 _f_ ( _a −_ _b_ )

_pa_ + (1 _−_ _p_ ) _b_ = _[a]_ [ +] _[ b]_

(39)
2 _f_ ( _a −_ _b_ ) _[.]_

463 The variance of _Ytnc_ under this distribution is maximized when _Mtn_ follows this binary distribution.

464 Substituting _p_ into the variance formula, the maximum variance is given by:

max(Var( _Ytnc_ )) = _[a]_ [2][ + 2] _[ab]_ [ +] _[ b]_ [2] _[ −]_ [4] _[fab]_ _._ (40)

4

**Case 2:** When 0 _≤_ _f_ _≤_ _[a]_ [+] _[b]_

465 **Case 2:** When 0 _≤_ _f_ _≤_ 2 _a_ [, the upper bound is achieved when] _[ M][tn]_ [=] _[ a]_ [.] [In this scenario,] _[ M][tn]_ [ is]

466 deterministic, and therefore:

_Ytnc_ = _XtncMtn_ = _Xtnca,_ E[ _Ytnc|Xtnc_ = 1] = _a._ (41)

467 Substituting this into the variance formula, the maximum variance simplifies to:

max(Var( _Ytnc_ )) = _a_ [2] ( _f_ [2] _−_ _f_ + 1 _/_ 2) + _ab_ (1 _−_ 2 _f_ ) + _b_ [2] _/_ 2 _._ (42)

468 The proof is now complete.

469 We observe that both SSA and QKTA exhibit significantly larger variance compared to our proposed

470 attention mechanism. Their variances are expressed as follows:

471 **Variance of QKTA:**

Var(QKTA) = _dfQ_ (1 _−_ _fQ_ ) _,_ (43)
472 where _d_ is the feature dimension, and _fQ_ represents the firing rate of the query.

473 **Variance of SSA:**

               Var(SSA) = _Nd_ _fQfKfV_ (1 _−_ _fQ_ )(1 _−_ _fK_ )(1 _−_ _fV_ )

+ _fQfKfV_ [2] [(1] _[ −]_ _[f][Q]_ [)(1] _[ −]_ _[f][K]_ [)]

+ _fQfK_ [2] _[f][V]_ [(1] _[ −]_ _[f][Q]_ [)(1] _[ −]_ _[f][V]_ [)]

+ _fQ_ [2] _[f][K][f][V]_ [(1] _[ −]_ _[f][K]_ [)(1] _[ −]_ _[f][V]_ [)]

+ _fQfK_ [2] _[f]_ _V_ [ 2] [(1] _[ −]_ _[f][Q]_ [)]

+ _fQ_ [2] _[f][K][f]_ _V_ [ 2] [(1] _[ −]_ _[f][K]_ [)]

+ _fQ_ [2] _[f]_ _K_ [ 2] _[f][V]_ [(1] _[ −]_ _[f][V]_ [)]             - _,_ (44)

474 where _N_ is the number of spatial locations, _d_ is the feature dimension, and _fQ, fK, fV_ are the firing

475 rates of the query, key, and value.

15

476 **Comparison** **with** **Our** **Attention** **Mechanism:** The variance of QKTA scales linearly with _d_ .

477 By contrast, the variance of SSA grows with both _N_ and _d_, resulting in significantly larger values

478 compared to QKTA. Our proposed attention mechanism is particularly effective in scenarios with large

479 spatial ( _N_ ) and feature ( _d_ ) dimensions. The strict upper bound on output variance ensures numerical

480 stability, preventing vanishing during training. Additionally, this upper bound eliminates the need

481 for traditional scaling operations (e.g., scaling factors in QK products), simplifying computations,

482 reducing complexity, and enhancing energy efficiency.

483 **B.2** **The mathematical properties of hyperparameters**

484 Next, we will analyze the expectation and variance of the PM and propose an appropriate selection of

485 hyperparameters to ensure output stability.

486 **Lemma B.1.** _if the set {c ∈_ N : _wc_ = 0 _} is finite and ∃_ _m, M_ _>_ 0 _, ∀_ _c ∈_ N _, m ≤|wc| ≤_ _M_ _, then:_

_wc_
_wc_ _[′]_ [=] [lim] = 0 (45)
_C→∞_ �� _C_
_c_ =1 _[w]_ _c_ [2]

487 _Proof._ We begin by defining the normalized weight:

_wc_
_wc_ _[′]_ [=] _._ (46)
�� _C_
_c_ =1 _[w]_ _c_ [2]

488 By assumption, there are _k_ terms where _wc_ = 0, and for the remaining _C_ _−_ _k_ terms, the weights

489 satisfy:

_m_ [2] _≤_ _wc_ [2] _[≤]_ _[M]_ [ 2] for all _c._ (47)

490 Thus, the sum of squares of the weights is bounded as follows:

( _C −_ _k_ ) _m_ [2] _≤_

_C_

- _wc_ [2] _[≤]_ [(] _[C][ −]_ _[k]_ [)] _[M]_ [ 2] _[.]_ (48)

_c_ =1

491 Taking the square root, we find that the denominator grows as:

~~�~~

       - _C_ _√_
��� _wc_ [2] _≥_ ~~�~~ ( _C −_ _k_ ) _m_ [2] _∼_ _O_ (

_c_ =1

492 Using the bound _|wc| ≤_ _M_, the normalized weight _wc_ _[′]_ [satisfies:]

_C_ ) _._ (49)

_|wc_ _[′]_ _[|]_ [ =] ~~�~~             - _|wCc|_ _≤_ �� _MC_ _≤_ �( _C −M_ _k_ ) _m_ [2] _[.]_ (50)
_c_ =1 _[w]_ _c_ [2] _c_ =1 _[w]_ _c_ [2]

493 To ensure _|wc_ _[′]_ _[|][ < ϵ]_ [ for a given] _[ ϵ >]_ [ 0][, it suffices to require:]

_M_
(51)

~~�~~ ( _C −_ _k_ ) _m_ [2] _[< ϵ.]_

494 Rearranging, this condition can be rewritten as:

_C_ _≥_ _m_ _[M]_ [2][ 2] _ϵ_ [2] [+] _[ k.]_ (52)

_M_ [2]
495 As _C_ _→∞_, the condition _C_ _≥_ _m_ [2] _ϵ_ [2] [+] _[ k]_ [ is always satisfied.] [Thus, for any] _[ ϵ >]_ [ 0][, we have] _[ |][w]_ _c_ _[′]_ _[|][ < ϵ]_ [,]

496 which implies:

lim _c_ [= 0] _[.]_ (53)
_C→∞_ _[w][′]_

497 The proof is complete.

16

498 **Lemma B.2.** _We assume that the features across different channels are independent and identically_

499 _distributed (i.i.d.)._ _When the number of channels C_ _is large, we have:_

- _wc_ _[′]_ [2] _[f][r]_ [(1] _[ −]_ _[f][r]_ [)]

_c_ =1

_,_ _C_ _→∞,_ (54)

_Mtn_ _∼N_

- _C_

 - _wc_ _[′]_ _[f][r][,]_

_c_ =1

_C_

500

_C_

- _xtncwc_ _[′]_ _[.]_ (55)

_c_ =1

_C_

_Mtn_ =

501 _where x ∈_ _X, x ∼_ _Bernoulli_ ( _fr_ ) _, fr_ _represents the firing rate (the probability of xtnc_ = 1 _)._

502 _Proof._ To prove this lemma, we use the characteristic function method. The characteristic function

503 of a Bernoulli random variable _xtnc_ is given by:

Φ _xtnc_ ( _t_ ) = E          - _e_ _[itx][tnc]_ [�] = _fre_ _[it]_ + (1 _−_ _fr_ ) _._ (56)

504 For the weighted variable _wc_ _[′]_ _[x][tnc]_ [, its characteristic function is:]

                    Φ _wc′_ _xtnc_ ( _t_ ) = E _e_ _[itw]_ _c_ _[′]_ _[x][tnc]_ [�] = _fre_ _[itw]_ _c_ _[′]_ + (1 _−_ _fr_ ) _._ (57)

505 Since the features across channels are independent, the characteristic function of _Mtn_ is:

_C_

         
Φ _Mtn_ ( _t_ ) = Φ _wc′_ _xtnc_ ( _t_ ) _._ (58)

_c_ =1

506 Substituting the expression for Φ _wc′_ _xtnc_ ( _t_ ):

Φ _Mtn_ ( _t_ ) =

_C_

_c_ =1

- _fre_ _[itw]_ _c_ _[′]_ + (1 _−_ _fr_ ) _._ (59)

- 1 + _itwc_ _[′]_ _[−]_ [1] _c_ [+] _[ o]_ [(] _[w]_ _c_ _[′]_ [2][)] + (1 _−_ _fr_ )

2 _[t]_ [2] _[w][′]_ [2]

1 + _itwc_ _[′]_ _[−]_ [1]

_fre_ _[itw]_ _c_ _[′]_ + (1 _−_ _fr_ ) = _fr_

_≈_ 1 + _fr_ ( _itwc_ _[′]_ _[−]_ [1] _c_ [)] _[.]_ (60)

2 _[t]_ [2] _[w][′]_ [2]

507 Thus, the characteristic function becomes:

Φ _Mtn_ ( _t_ ) _≈_

_C_

_c_ =1

- 1 + _fr_ ( _itwc_ _[′]_ _[−]_ [1] _c_ [)] _._ (61)

2 _[t]_ [2] _[w][′]_ [2]

508 Taking the logarithm to simplify the product into a sum:

_C_

[1] _c_ [)] 
2 _[t]_ [2] _[w][′]_ [2]

ln Φ _Mtn_ ( _t_ ) =

=

  
- ln 1 + _fr_ ( _itwc_ _[′]_ _[−]_ [1]

2

_c_ =1

- _fritwc_ _[′]_ _[−]_ [1]

2

_c_ =1

_C_

[1] _c_ _[f][r]_ [+] [1]

2 _[t]_ [2] _[w][′]_ [2] 2

_c_ _[f]_ _r_ [ 2] [+] _[ O]_ [(] _[w]_ _c_ _[′]_ [2][)] _[,]_ (62)
2 _[t]_ [2] _[w][′]_ [2]

509 where we used ln(1 + _x_ ) = _x −_ 2 [1] _[x]_ [2][ +] _[ O]_ [(] _[x]_ [2][)][ for small] _[ x]_ [.]

510 Separating terms, we get:

_C_

_C_

 
[1]

2 _[t]_ [2]

- _wc_ _[′]_ _[f][r]_ _[−]_ [1]

2

_c_ =1

- _wc_ _[′]_ [2] _[f][r]_ [(1] _[ −]_ _[f][r]_ [)] _[.]_ (63)

_c_ =1

ln Φ _Mtn_ ( _t_ ) _≈_ _it_

17

511 Exponentiating the logarithm gives:

           
Φ _Mtn_ ( _t_ ) = exp _it_

- _wc_ _[′]_ _[f][r]_ _[−]_ [1]

2

_c_ =1

_C_

_C_

 
[1]

2 _[t]_ [2]

- _wc_ _[′]_ [2] _[f][r]_ [(1] _[ −]_ _[f][r]_ [)]

_c_ =1

_._ (64)

512 This is the characteristic function of a normal distribution with:

Mean: _µ_ =

_C_

- _wc_ _[′]_ _[f][r][,]_ (65)

_c_ =1

Variance: _σ_ [2] =

_C_

- _wc_ _[′]_ [2] _[f][r]_ [(1] _[ −]_ _[f][r]_ [)] _[.]_ (66)

_c_ =1

513 Since the characteristic function corresponds to a normal distribution, we conclude:

- _wc_ _[′]_ [2] _[f][r]_ [(1] _[ −]_ _[f][r]_ [)]

_c_ =1

_._ (67)

_Mtn_ _∼N_

- _C_

 - _wc_ _[′]_ _[f][r][,]_

_c_ =1

_C_

514 The proof is complete.

515 **Lemma B.3.** _The distributions of Xtnc_ _and Mtn_ _can be considered independent when the number of_

516 _channels C_ _is large._ _Specifically, for all t_ 1 _, t_ 2 _∈_ R _, we have:_

_ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) = _ϕMtn_ ( _t_ 1) _· ϕXtnc_ ( _t_ 2) _,_ _C_ _→∞,_ (68)

517 _where ϕX_ ( _t_ ) _represents the characteristic function of X._

518 _Proof._ The joint characteristic function of _Mtn_ and _Xtnc_ is given by:

                      _ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) = E _e_ [(] _[it]_ [1] _[M][tn]_ [+] _[it]_ [2] _[X][tnc]_ [)][�]

  -   = E _e_ [(] _[it]_ [1] _c_ _[w]_ _c_ _[′]_ _[X][tnc]_ [+] _[it]_ [2] _[X][tnc]_ [)][�] _._ (69)

519 Separating _Xtnc_ and the sum [�] _i_ = _c_ _[w]_ _i_ _[′][X][tni]_ [, we rewrite:]

          -          _ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) = E _e_ [(] _[it]_ [1] _i_ = _c_ _[w]_ _i_ _[′]_ _[X][tni]_ [+] _[iX][tnc]_ [(] _[t]_ [2][+] _[t]_ [1] _[w]_ _c_ _[′]_ [)][)][�]

          -          -          = E _e_ [(] _[it]_ [1] _i_ = _c_ _[w]_ _i_ _[′]_ _[X][tni]_ [)][�] _·_ E _e_ [(] _[iX][tnc]_ [(] _[t]_ [2][+] _[t]_ [1] _[w]_ _c_ _[′]_ [)][)][�] _._ (70)

520 Using the independence of _Xtni_ across channels:

_ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) =          - E          - _e_ [(] _[it]_ [1] _[w]_ _i_ _[′]_ _[X][tni]_ [)][�] _·_ E          - _e_ [(] _[iX][tnc]_ [(] _[t]_ [2][+] _[t]_ [1] _[w]_ _c_ _[′]_ [)][)][�] _._ (71)

_i_ = _c_

521 Substituting the characteristic function of Bernoulli random variables _Xtnc_ _∼_ Bernoulli( _f_ ):

E            - _e_ _[itX][tnc]_ [�] ] = (1 _−_ _f_ ) + _fe_ _[it]_ _._ (72)

522 Thus:

    -    - _i_    -    - _c_ [)][�]
_ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) = (1 _−_ _f_ ) + _fe_ _[it]_ [1] _[w][′]_ _·_ (1 _−_ _f_ ) + _fe_ _[i]_ [(] _[t]_ [2][+] _[t]_ [1] _[w][′]_ _._ (73)

_i_ = _c_

523 Using Lemma B.2, for small _wc_ _[′]_ [, we apply the Taylor expansion to approximate each term:]

(1 _−_ _f_ ) + _fe_ _[it]_ [1] _[w]_ _i_ _[′]_ _≈_ 1 + _f_ ( _it_ 1 _wi_ _[′]_ [)] _[,]_ (74)

(1 _−_ _f_ ) + _fe_ _[i]_ [(] _[t]_ [2][+] _[t]_ [1] _[w]_ _c_ _[′]_ [)] _≈_ (1 _−_ _f_ ) + _fe_ _[it]_ [2] _._ (75)

18

524 Substituting back:

_ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) _≈_ - (1 + _fit_ 1 _wi_ _[′]_ [)] _[ ·]_ �(1 _−_ _f_ ) + _fe_ _[it]_ [2][�] _._ (76)

_i_ = _c_

525 Using Equation 59, Equation 72 and Taylor expansion, the product of the characteristic functions for

526 the two distributions is:

_ϕXtnc_ ( _t_ 2) _ϕMtn_ ( _t_ 1) = (1 _−_ _f_ + _fe_ _[it]_ [2] )

= (1 _−_ _f_ + _fe_ _[it]_ [2] )

_C_
�(1 _−_ _f_ + _fe_ _[it]_ [1] _[w]_ _i_ _[′]_ )

_i_ =1

_C_
�(1 + _fit_ 1 _wi_ _[′]_ [)]

_i_ =1

= (1 _−_ _f_ + _fe_ _[it]_ [2] )(1 + _fit_ 1 _wc_ _[′]_ [)] �(1 + _fit_ 1 _wi_ _[′]_ [)]

_i_ = _c_

= (1 _−_ _f_ + _fe_ _[it]_ [2] ) �(1 + _fit_ 1 _wi_ _[′]_ [)]

_i_ = _c_

= _ϕMtn,Xtnc_ ( _t_ 1 _, t_ 2) (77)

527 Thus, the joint characteristic function factorizes into the product of the marginal characteristic

528 functions, which demonstrates that _Mtn_ and _Xtnc_ are asymptotically independent as _C_ _→∞_ .

529 **Proposition B.4.** _If b ≈_ 0 _, a ≥_ 1 _, and the firing rate f_ _is relatively small value, the PM output Ytnc_

530 _satisfies:_

E( _Ytnc_ ) _≈_

~~�~~ _f_ (1 _−_ _f_ )

E( _Xtnc_ ) (78)
2 _π_

_Var_ ( _Ytnc_ ) _≈_ _[f]_ [(] _[π][ −]_ _[f]_ [)] _Var_ ( _Xtnc_ ) (79)

2 _π_

531 _Proof._ For convenience, we denote:

_C_

- _wc_ _[′]_ [2] _[f]_ [(1] _[ −]_ _[f]_ [) =] _[ f]_ [(1] _[ −]_ _[f]_ [)] _[,]_ _Mtn_ _[′]_ [=][ clamp][(] _[M][tn][, b, a]_ [)] _[.]_ (80)

_c_ =1

_µ_ =

_C_

- _wc_ _[′]_ _[f,]_ _σ_ [2] =

_c_ =1

532 According to Lemma B.2, the input distribution satisfies:

_Mtn_ _∼N_ ( _µ, σ_ [2] ) _._ (81)

533 The expectation of the clamped variable _Mtnc_ _[′]_ [is:]

       - _∞_
E( _Mtn_ _[′]_ [) =] _xf_ ( _x_ ) _dx_

_−∞_

- _∞_

_∞_ 
exp _−_ [(] _[x][ −]_ _[µ]_ [)][2]
_a_ 2 _σ_ [2]

1
= ~~_√_~~

2 _πσ_ [2]

- _a_

_a_ 
_x_ exp _−_ [(] _[x][ −]_ _[µ]_ [)][2]
0 2 _σ_ [2]

2 _σ_ [2]

- _a_
_dx_ + ~~_√_~~

2 _πσ_ [2]

2 _σ_ [2]

_dx._ (82)

534 For the first term, let _t_ = ( _x −_ _µ_ ) [2], if _µ ≈_ 0, then:

1
~~_√_~~

2 _πσ_ [2]

- _a_

_a_  
_x_ exp _−_ [(] _[x][ −]_ _[µ]_ [)][2]
0 2 _σ_ [2]

2 _σ_ [2]

_dx_

- _a_

_a_ 
exp _−_ [(] _[x][ −]_ _[µ]_ [)][2]
0 2 _σ_ [2]

1
= ~~_√_~~

- ( _a−µ_ ) [2]

~~_√_~~
2

- _µ_
_dt_ + ~~_√_~~

2 _πσ_ [2]

2 _σ_ [2]

_dx_

2 _πσ_ [2]

    exp _−_ _[t]_
_µ_ [2] 2 _σ_

2 _σ_ [2]

- - _−µ_

_−_ Φ
_σ_

��

��( _a−µ_ ) [2] - - _a −_ _µ_

+ _µ_ Φ
_µ_ [2] _σ_

��( _a−µ_ ) [2]

_−σ_
= ~~_√_~~

2 _πσ_

- exp _−_ _[t]_

2 _σ_ [2]

��
_._ (83)

19

_σ_
_≈_ ~~_√_~~

2 _π_

_σ_
_≈_ ~~_√_~~

- 1 _−_ exp _−_ _[a]_ [2]

2 _σ_ [2]

- 1 _−_ exp _−_ _[a]_ [2]

535 where Φ( _x_ ) is the CDF of the standard normal distribution. The second term in the expectation is

536 straightforward:

- _∞_

_∞_ 
exp _−_ _[t]_ [2]
_a−µ_ 2 _σ_

_a_
~~_√_~~

2 _πσ_ [2]

- _∞_

_∞_ 
exp _−_ [(] _[x][ −]_ _[µ]_ [)][2]
_a_ 2 _σ_ [2]

2 _σ_ [2]

- _a_
_dx_ = ~~_√_~~

2 _πσ_ [2]

2 _σ_ [2]

_dt,_ (84)

537 Using the cumulative distribution function (CDF) again:

��

_a_
~~_√_~~

2 _πσ_ [2]

- _∞_

_∞_ 
exp _−_ _[t]_ [2]
_a−µ_ 2 _σ_

2 _σ_ [2]

- - - _a −_ _µ_
_dt_ = _a_ 1 _−_ Φ
_σ_

  -  - _a_ ��
_≈_ _a_ 1 _−_ Φ (85)
_σ_

_[a]_

_σ_ [)][ and][ exp(] _[−]_ 2 _[a]_ _σ_ [2]

The Φ( _[a]_

538 The Φ( _σ_ [)][ and][ exp(] _[−]_ 2 _σ_ [2] [)][ function decay rapidly as] _[ σ]_ [ decreases.] [Now, combining the results from]

539 the two integrals, we have:

_σ_
E( _Mtn_ _[′]_ [) =] ~~_√_~~

_σ_
~~_√_~~
2 _π_ _[−]_ 2

_σ_ 
_−_ _[a]_ [2]
2 _π_ [exp] 2 _σ_ [2]

2 _σ_ [2]

- - - _a −_ _µ_
+ _a_ 1 _−_ Φ
_σ_

��

_σ_
_≈_ ~~_√_~~ (86)

2 _π_

540 Based on B.3, we calculate the expectation and variance of _Mtn_ _[′]_ [2] [:]

            - _∞_
E( _Mtn_ _[′]_ [2] [) =] _x_ [2] _f_ ( _x_ ) _dx_

_−∞_

1
= ~~_√_~~

2 _πσ_ [2]

- _a_

_a_  
_x_ [2] exp _−_ _[x]_ [2]
0 2 _σ_ [2]

2 _σ_ [2]

- - _∞_
_dx_ + _a_ [2] _·_ _f_ ( _x_ ) _dx._ (87)

_a_

541 We calculate the first term using integration by parts. Let:

      _u_ = _x,_ _dv_ = _x_ exp _−_ _[x]_ [2]

2 _σ_ [2]

      _u_ = _x,_ _dv_ = _x_ exp _−_ _[x]_ [2]

- _dx,_ _du_ = _dx,_ _v_ = _−σ_ [2] exp _−_ _[x]_ [2]

2 _σ_ [2]

_._ (88)

542 Then:

1
~~_√_~~

2 _πσ_ [2]

- _a_

_a_ 
_x_ [2] exp _−_ _[x]_ [2]
0 2 _σ_ [2]

2 _σ_ [2]

_dx_

_a_ 
exp _−_ _[x]_ [2]
0 2 _σ_ [2]

- _dx_

�� _a_

   - _a_
+ _σ_ [2]
0 0

2 _σ_ [2]

1
= ~~_√_~~

2 _πσ_ [2]

�� _−σ_ [2] _x_ exp _−_ _[x]_ [2]

2 _σ_ [2]

2 _σ_ [2]

- - _a_
+ _σ_ [2]

_a_ 
exp _−_ _[x]_ [2]
0 2 _σ_ [2]

1
= ~~_√_~~

2 _πσ_ [2]

- _−σ_ [2] _a_ exp _−_ _[a]_ [2]

2 _σ_ [2]

- _dx_ _._ (89)

543 The remaining integral is a standard normal distribution integral:

_σ_ [2]
~~_√_~~

2 _πσ_ [2]

- _a_

_a_ 
exp _−_ _[x]_ [2]
0 2 _σ_ [2]

_−_ [1]

2

_,_ (90)

2 _σ_ [2]

- - - _a_
_dx_ = _σ_ [2] Φ
_σ_

544 where Φ( _x_ ) is the CDF of the standard normal distribution.

545 Substituting (90) into (89):

2 _σ_ [2]

- - - _a_
+ _σ_ [2] Φ
_σ_

_−_ [1]

2

_._ (91)

1
~~_√_~~

2 _πσ_ [2]

- _a_

_a_ 
_x_ [2] exp _−_ _[x]_ [2]
0 2 _σ_ [2]

2 _σ_ [2]

_dx_ = ~~_√_~~ _[−][aσ]_

_[aσ]_ 
_−_ _[a]_ [2]
2 _π_ [exp] 2 _σ_ [2]

546 The second term is the tail of the normal distribution:

               - _∞_

      _f_ ( _x_ ) _dx_ = Φ _−_ _[a]_
_a_ _σ_

_σ_

_,_ (92)

547 we have:

  - _∞_
_a_ [2] _·_

_σ_

       _f_ ( _x_ ) _dx_ = _a_ [2] Φ _−_ _[a]_
_a_ _σ_

_._ (93)

20

548 Combining (91) and (93) into (87), we get:

_[aσ]_ 
_−_ _[a]_ [2]
2 _π_ [exp] 2 _σ_ [2]

_−_ [1]

2

E( _Mtn_ _[′]_ [2] [) =] ~~_√_~~ _[−][aσ]_

2 _σ_ [2]

- - - _a_
+ _σ_ [2] Φ
_σ_

- + _a_ [2] Φ _−_ _[a]_

_σ_

_≈_ _[σ]_ [2] (94)

2 _[.]_

Since Φ - _−_ _[a]_

_σ_ _[a]_ - is exponentially small for moderate _a_, the term _a_ [2] Φ - _−_ _σ_ _[a]_

      -       -       -       
549 Since Φ _−_ _σ_ _[a]_ is exponentially small for moderate _a_, the term _a_ [2] Φ _−_ _σ_ _[a]_ is negligible compared to

550 leading terms and is often omitted for simplicity.

551 Using Var( _Mtn_ _[′]_ [) =][ E][(] _[M][ ′]_ _tn_ [2] [)] _[ −]_ [E][(] _[M][ ′]_ _tn_ [)][2][, we calculate:]

Var( _Mtn_ _[′]_ [)] _[ ≈]_ _[σ]_ [2]

2 _π_

���2

[2]  - _σ_

~~_√_~~
2 _[−]_ 2

- 1 _−_ exp _−_ _[a]_ [2]

2 _σ_ [2]

_≈_ _[π][ −]_ [1]

2 _π_ _[σ]_ [2]

= _[π][ −]_ [1] (95)

2 _π_ _[f]_ [(1] _[ −]_ _[f]_ [)] _[.]_

552 Given that _Ytnc_ = _Mtnc_ _[′]_ _[·][ X][tnc]_ [, and based on Lemma B.3 that the distributions of] _[ X][tnc]_ [and] _[ M][ ′]_ _tn_ [can]

553 be considered independent, the expectation of _Ytnc_ is:

E( _Ytnc_ ) = E( _Mtn_ _[′]_ [)] _[ ·]_ [ E][(] _[X][tnc]_ [)]

_≈_

- _f_ (1 _−_ _f_ )

E( _Xtnc_ ) _._ (96)
2 _π_

554 The variance of _Ytnc_ is computed as:

Var( _Ytnc_ ) = Var( _Mtn_ _[′]_ [)] _[ ·]_ [ Var][(] _[X][tnc]_ [) +][ Var][(] _[M][ ′]_ _tn_ [)] _[ ·]_ [ E][(] _[X][tnc]_ [)][2][ +][ Var][(] _[X][tnc]_ [)] _[ ·]_ [ E][[] _[M][ ′]_ _tn_ []][2]

= _[f]_ [(] _[π][ −]_ _[f]_ [)] _f_ (1 _−_ _f_ )

2 _π_

_≈_ _[f]_ [(] _[π][ −]_ _[f]_ [)] Var( _Xtnc_ ) _._ (97)

2 _π_

555 Thus, the proposition is proven:

Var( _Xtnc_ ) _._ (98)
2 _π_

E( _Ytnc_ ) _≈_

- _f_ (1 _−_ _f_ )

_−_ _f_ )

E( _Xtnc_ ) _,_ Var( _Ytnc_ ) _≈_ _[f]_ [(] _[π][ −]_ _[f]_ [)]
2 _π_ 2 _π_

556

557 In practice, we recommend setting the hyperparameters as follows: _b_ = 0 and _a_ _∈_ [1 _,_ 2]. Setting

558 _b_ = 0 allows the processing module to completely eliminate certain features in the spatial domain.

559 Furthermore, selecting _a ∈_ [1 _,_ 2] enables the processing module to selectively enhance specific spatial

560 features. This also ensures that both the mean and variance do not become too large or too small,

561 maintaining the numerical stability.

562 **B.3** **Gradient Analysis**

563 This section on the derivation of the traditional SNN network is mainly referenced from [40, 7, 8].

564 First, we derive the temporal gradient of the traditional SNN network, where the temporal gradient

565 is primarily backpropagated through the membrane potential. Taking the vanilla LIF neuron as an

566 example, we use the following form to analyze the gradient problem:

    **H** _[l]_ ( _t_ + 1) = 1 _−_ [1]

_τ_

�� **H** _[l]_ ( _t_ ) _−_ _ϑ_ **S** _[l]_ ( _t_ )� + **W** _[l]_ **S** _[l][−]_ [1] ( _t_ + 1) _,_ (99)

567 The derivative of the loss with respect to the weights _Wl_ is:

_T −_ 1

  
_∇_ **W** _l_ _L_ =

_t_ =0

_∂L_ _⊤_

**S** _[l][−]_ [1] [ _t_ ] _[⊤]_ _, l_ = _L, L −_ 1 _, · · ·_ _,_ 1 _,_ (100)

_∂_ **H** _[l]_ ( _t_ )

21

568 The gradient expression can be written as:

_∂L_ _∂L_ _∂_ **H** _[l]_ [+1] ( _t_ ) _∂_ **S** _[l]_ ( _t_ )

+

_∂_ **H** _[l]_ ( _t_ ) [=] _∂_ **H** _[l]_ [+1] ( _t_ ) _∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ )

     - ��     _Spatial_ _Gradient_

_, l < L,_ (101)

_T −_ 1

_t_ _[′]_ = _t_ +1

_∂L_ _∂_ **H** _[l]_ [+1] ( _t_ _[′]_ ) _∂_ **S** _[l]_ ( _t_ _[′]_ )
_∂_ **H** _[l]_ [+1] ( _t_ _[′]_ ) _∂_ **S** _[l]_ ( _t_ _[′]_ ) _∂_ **H** _[l]_ ( _t_ _[′]_ )

_∂L_ _∂_ **H** _[l]_ [+1] ( _t_ _[′]_ )
_∂_ **H** _[l]_ [+1] ( _t_ _[′]_ ) _∂_ **S** _[l]_ ( _t_ _[′]_ )

_t_ _[′]_ _−t_

- _ϵ_ _[L]_ ( _t_ _[′]_ _−_ _t_ _[′′]_ )

_t_ _[′′]_ =1

- �� _T emporal_ _Gradient_

569

_, l_ = _L,_ (102)

_t_ _[′]_ _−t_

- _ϵ_ _[L]_ ( _t_ _[′]_ _−_ _t_ _[′′]_ )

_t_ _[′′]_ =1

_∂L_ _∂L_ _∂_ **S** _[l]_ ( _t_ )
_∂_ **H** _[l]_ ( _t_ ) [=] _∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ )

     - ��      _Spatial_ _Gradient_

- �� _T emporal_ _Gradient_

+

_T −_ 1

_t_ _[′]_ = _t_ +1

_∂L_ _∂_ **S** _[l]_ ( _t_ _[′]_ )
_∂_ **S** _[l]_ ( _t_ _[′]_ ) _∂_ **H** _[l]_ ( _t_ _[′]_ )

570 _ϵ_ _[L]_ is defined as the sensitivity of the membrane potential _H_ _[l]_ ( _t_ + 1) with respect to _H_ _[l]_ ( _t_ ) between

571 adjacent timesteps.

_[l]_ [(] _[t]_ [ + 1)]

+ _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]
_∂_ **H** _[l]_ ( _t_ ) _∂_ **S** _[l]_ ( _t_ )

_ϵ_ _[l]_ ( _t_ ) _≡_ _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]

_[l]_ [(] _[t]_ [ + 1)] _∂_ **S** _[l]_ ( _t_ )

(103)
_∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ ) _[.]_

572 If we use a simple rectangular function as a surrogate for the gradient.

_ϵ_ _[l]_ ( _t_ ) _jj_ = - 0 _,_ [1] 21 _[ϑ < H]_ _j_ _[l]_ [(] _[t]_ [)] _[ <]_ [3] 2

0 _,_ 2 _[ϑ < H]_ _j_ [(] _[t]_ [)] _[ <]_ 2 _[ϑ,]_

1 _−_ [1] _[,]_ otherwise _._

2 2 (104)

_τ_ [1] _[,]_ otherwise _._

573 From the above equation, it can be concluded that if the membrane potential approaches the threshold

574 at any given timestep, the temporal gradient [�] _t_ _[t][′′][′][−]_ =1 _[t]_ _[ϵ][L]_ [ (] _[t][′][ −]_ _[t][′′]_ [)] [will] [vanish.] [This] [highlights] [a]

575 common issue with temporal gradients in the vanilla LIF model, which remains a problem even with

576 short timesteps.

577 Next, we perform gradient analysis on neurons with a feedback structure. Assume the structure of the

578 feedback is _φ_, which includes PM and CM.

    **H** _[l]_ ( _t_ + 1) = 1 _−_ [1]

_τ_

�� **H** _[l]_ ( _t_ ) _−_ _ϑ_ **S** _[l]_ ( _t_ )� + **W** _[l]_ **S** _[l][−]_ [1] ( _t_ + 1) + _φθ_ ( **S** _[l]_ ( _t_ )) (105)

579 Following the above derivation, we similarly define the variable _ϵ_ :

_ϵ_ _[l]_ ( _t_ ) _≡_ _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]

_[l]_ [(] _[t]_ [ + 1)]

+ _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]
_∂_ **H** _[l]_ ( _t_ ) _∂_ **S** _[l]_ ( _t_ )

**[H]** _[l]_ [(] _[t]_ [ + 1)] _∂_ **S** _[l]_ ( _t_ ) _[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]

_∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ ) [+] _∂φθ_ ( **S** _[l]_ ( _t_ ))

_∂φθ_ ( **S** _[l]_ ( _t_ )) _∂_ **S** _[l]_ ( _t_ )

_[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]

_∂φθ_ ( **S** _[l]_ ( _t_ )) _∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ )

- �� _F eedback_ _gradient_

_∂φθ_ ( **S** _[l]_ ( _t_ ))

_[∂]_ **[H]** _[l]_ [(] _[t]_ [ + 1)]

_∂φθ_ ( **S** _[l]_ ( _t_ )) _∂_ **S** _[l]_ ( _t_ )

(106)

580

_[θ]_ [(] **[S]** _[l]_ [(] _[t]_ [))] _∂_ **S** _[l]_ ( _t_ )

(107)
_∂_ **S** _[l]_ ( _t_ ) _∂_ **H** _[l]_ ( _t_ )

  _ϵ_ _[l]_ ( _t_ ) = 1 _−_ [1]

_τ_

- 
_−_ 1 _−_ [1]

_τ_

_ϑ ·_ _[∂]_ **[S]** _[l]_ [(] _[t]_ [)]

_[∂]_ **[S]** _[l]_ [(] _[t]_ [)] _[∂φ][θ]_ [(] **[S]** _[l]_ [(] _[t]_ [))]

_∂_ **H** _[l]_ ( _t_ ) [+] _∂_ **S** _[l]_ ( _t_ )

581 Similarly we have:

_ϵ_ _[l]_ ( _t_ ) _jj_ =

- _∂φθ_ ( **S** _[l]_ ( _t_ ))

_∂θ_ **S** _[l]_ ( _t_ ) _,_ 2 _[ϑ < H]_ _j_ [(] _[t]_ [)] _[ <]_ 2 _[ϑ,]_

1 _−_ [1] _[,]_ otherwise _._

_∂θ_ **S** ( **S** _[l]_ ( _t_ () _t_ )) _,_ 21 _[ϑ < H]_ _j_ _[l]_ [(] _[t]_ [)] _[ <]_ [3] 2

_[l]_ ( _t_ ) 2 _j_ 2 (108)

_τ_ [1] _[,]_ otherwise _._

582 Then, in training, _[∂φ]_ _∂_ _[θ]_ **S** [(] **[S]** _[l]_ ( _[l]_ _t_ [(] ) _[t]_ [))] is not possible to be zero.

583 **C** **Supplementary Results**

584 **C.1** **Energy Consumption Calculation of TDFormer**

585 This section is mainly referenced from [3]. We calculate the number of Synaptic Operations (SOPs)

586 of spike before calculating theoretical energy consumption for TDFormer.

SOP = _fr × T_ _×_ FLOPs (109)

22

Table 5: Results averaged across seeds: 0, 42, 2024, 3407 and 114514. Bold results indicate superior
performance compared to the baselines.

Methods Dataset/Time Step Architecture Baseline CM1+V1

CIFAR-10/T = 2 **94.18±0.06** 94.07±0.07
Spikformer-2-384
CIFAR-10/T = 4 94.84±0.14 **94.86±0.05**
CIFAR-10/T = 2 93.65±0.23 **94.05±0.14**

SpikeformerV1

SDTV1

CIFAR-10/T = 2 93.65±0.23 **94.05±0.14**

CIFAR-100/T =2 75.25±0.19 **75.99±0.12**
CIFAR-10/T = 4 94.73±0.06 **95.13±0.07**
CIFAR-100/T = 4 77.56±0.22 **77.60±0.26**

Spikformer-4-384

CIFAR-10/T = 6 95.09±0.08 **95.16±0.14**
CIFAR-100/T = 6 **78.21±0.22** 77.99±0.05
CIFAR10-DVS/T = 10 78.08±0.70 **78.13±0.72**
CIFAR10-DVS/T= 16 79.40±0.36 **80.20±0.75**

CIFAR-10/T = 4 94.47±0.11 **94.64±0.04**

CIFAR-100/T =4 76.15±0.13 **76.26±0.13**
DVS128 Gesture/T=10 Spiking 96.79±0.67 **96.92±0.29**
DVS128 Gesture/T=16 Transformer-2-256 97.98±0.59 **99.04±0.28**
CIFAR10-DVS/T = 10 75.03±0.67 **75.05±0.11**
CIFAR10-DVS/T = 16 77.07±0.19 **77.45±0.43**

Spiking
Transformer-2-256

Spikformer-4-384

CIFAR-10/T = 4 Spiking 95.76±0.06 **95.92±0.02**
CIFAR-100/T =4 Transformer-2-512 79.15±0.14 **79.35±0.16**
CIFAR-10/T = 4 94.47±0.11 **94.64±0.04**

587 where _fr_ is the firing rate of the block and _T_ is the simulation time step of spiking neuron. FLOPS

588 refers to floating point operations of block, which is the number of multiply-and-accumulate (MAC)

589 operations and SOP is the number of spike-based accumulate (AC) operations.

_E_ TDFormer = _EBaseline_ + _EAC_ _×_ (SOPPM + SOPCM) (110)

590 The channel-wise token mixer in TDFormer is highly power-efficient, consisting of only a linear

591 layer, a LIF neuron, and a BN layer. The BN parameters can be fused into the linear layer via

592 reparameterization, making its power consumption negligible. The linear layer maintains a constant

593 channel dimension, resulting in much lower power usage than conventional MLPs. Furthermore, the

594 spatial-wsie token mixer in PM has a time complexity of only _O_ ( _ND_ ), which is much lower than

595 the _O_ ( _N_ [2] _D_ ) of SSA. In the CM module, although a token mixer is used, the firing rates in both PM

596 and CM are very low. In our experiments, we observed that the firing rate in both modules remains

597 around 0.05. As a result, the overall power overhead of TDFormer is marginal.

598 **C.2** **Additional Experiments and Visualizations**

23

Table 6: Results of different TDFormer variants. The results in bold indicate superior performance
compared to the baseline. The default configuration used in our work is indicated by *. CM1-CM3
denote different strategies for integrating top-down information with bottom-up features. CM1: _Std_
is fused into the computation of the attention map. CM2: _Std_ is fused into the value of self-attention.
CM3: _Std_ is incorporated into the input of the attention module.

SpikeformerV1 SDTV1
Model Type (Spikformer-4-384) (Spiking Transformer-2-256)

Acc (%) FLOPs (G) Param (M) Acc (%) FLOPs (G) Param (M)

Baseline 94.73 3.71 9.33 94.47 1.25 2.57
*CM1+V1 **95.14** 3.88 9.92 **94.77** 1.31 2.69
CM1+V2 **94.79** 3.88 9.92 **94.93** 1.31 2.69
CM1+V3 **94.90** 3,88 9.92 **94.61** 1.31 2.69
CM1+V4 **94.94** 3.88 9.92 **94.88** 1.31 2.69
CM2+V1 **94.88** 3.88 9.92 **94.73** 1.31 2.69
CM2+V2 **94.75** 3.88 9.92 **94.79** 1.31 2.69
CM2+V3 94.70 3.88 9.92 **94.75** 1.31 2.69
CM2+V4 **95.27** 3.88 9.92 **94.66** 1.31 2.69
CM3+V1 94.69 3.90 9.92 94.43 1.32 2.69
CM3+V2 **94.89** 3.90 9.92 **94.69** 1.32 2.69
CM3+V3 94.35 3.90 9.92 93.94 1.32 2.69
CM3+V4 **94.90** 3.90 9.92 **94.61** 1.32 2.69

Figure 3: This is the histogram of the gradient of the surrogate function for LIF neurons in the
attention module within the PM model. From the figure, we can see that the clamp operation ensures
that the variance in the attention map does not become too large, thus preventing the vanishing
gradient problem.

24

Figure 4: Visualization of CIFAR-10C. This figure showcases 19 columns corresponding to 19
different types of corruptions. Each column contains four images: the top image displays the original
CIFAR-10C image; the second image shows the visualization result of the baseline model; the third
image illustrates the first feedforward stage of the TDFormer model; the fourth image depicts the
second feedforward stage of the TDFormer model, demonstrating the model’s dynamic attention
adjustments across stages.

25

Figure 5: Visualization of ImageNet-C. This figure showcases 19 columns corresponding to 19
different types of corruptions. The layout and visualization style are similar to those shown in Figure
4.

26

Table 7: Robustness comparison on the CIFAR-10C dataset. The results in bold indicate superior
performance compared to the baseline. Average performance across different distortion types is
indicated by *.

Corruption
Type

SpikformerV1 SpikformerV1
Time Corruption
/TDFormer /TDFormer
Step Type

Acc (%) Acc (%)

91.32/91.27 (-0.05) 1 **76.23/76.97 (+0.74)**
Brightness **91.87/91.94 (+0.06)** 2 **77.00/78.30 (+1.30)** Motion Blur
**93.14/93.29 (+0.15)** 4 **79.44/80.01 (+0.57)**

**69.93/70.40 (+0.47)** 1 **79.31/79.51 (+0.20)**
Contrast **70.41/71.25 (+0.84)** 2 78.70/78.67 (-0.03) Pixelate
77.06/76.57 (-0.49) 4 **81.14/81.45 (+0.31)**

**80.59/80.83 (+0.24)** 1 87.33/87.10 (-0.23)
Defocus Blur **81.39/82.15 (+0.76)** 2 **88.30/88.44 (+0.14)** Saturate
82.88/82.75 (-0.13) 4 **90.58/90.60 (+0.02)**

**84.00/84.05 (+0.05)** 1 **69.63/70.68 (+1.05)**
Elastic Transform **84.10/84.63 (+0.53)** 2 **70.96/71.09 (+0.13)** Shot Noise
85.54/85.52 (-0.02) 4 **73.23/73.32 (+0.09)**

**84.29/85.22 (+0.93)** 1 **84.47/84.71 (+0.24)**
Fog **85.09/85.75 (+0.66)** 2 **84.72/84.72 (+0.00)** Snow
**87.25/87.53 (+0.28)** 4 **86.90/87.18 (+0.28)**

**82.35/82.66 (+0.31)** 1 88.20/88.03 (-0.17)
Frost **83.04/83.27 (+0.23)** 2 **87.58/87.71 (+0.13)** Spatter
**85.46/85.70 (+0.24)** 4 89.14/89.02 (-0.12)

**73.33/74.05 (+0.72)** 1 **71.77/72.66 (+0.89)**
Gaussian Blur **74.79/75.84 (+1.05)** 2 72.66/72.64 (-0.02) Speckle Noise
**76.08/76.25 (+0.17)** 4 **75.10/75.37 (+0.27)**

**61.35/62.71 (+1.36)** 1 **75.98/76.68 (+0.70)**
Gaussian Noise 63.05/62.71 (-0.34) 2 **77.60/78.75 (+1.15)** Zoom Blur
**64.34/64.89 (+0.55)** 4 **78.68/79.14 (+0.46)**

**67.84/68.10 (+0.26)** 1 **57.86/58.26 (+0.40)**
Impulse Noise 65.83/65.36 (-0.47) 2 56.09/55.81 (-0.28) Glass Blur
**65.98/66.93 (+0.95)** 4 **59.43/60.46 (+1.03)**

**83.32/83.53 (+0.21)** 1 **78.11/78.55 (+0.44)**
JPEG Compression **83.93/84.00 (+0.07)** 2 **78.52/78.84 (+0.32)**    - Avg
**84.60/84.76 (+0.16)** 4 **80.53/80.78 (+0.25)**

599 **D** **Limitations, Future Work, and Broader Impacts**

600 **D.1** **Limitations**

601 Despite the promising enhancements introduced by our proposed TDFormer with top-down feedback

602 structure for spiking neural networks, several limitations remain. First, the current feedback mecha
603 nism is specifically designed for Transformer-based architectures and may not be directly applicable

604 to CNN-based SNNs, limiting its architectural generalizability. Second, our evaluation has so far

605 been limited to image classification tasks, which may not fully reflect the method’s effectiveness in

606 other domains such as object detection[42], semantic segmentation[43], and NLP tasks[44].

607 **D.2** **Future Work**

608 Future work could focus on generalizing the proposed TDFormer architecture to other network back
609 bones, such as CNN-based spiking neural networks, thereby improving its architectural compatibility

27

610 and deployment flexibility. In addition, extending the evaluation of TDFormer to tasks such as object

611 detection, semantic segmentation, and natural language processing would provide deeper insights

612 into its generalization capacity across diverse domains and data modalities. Moreover, we observe

613 that the proposed top-down feedback structure increases the diversity of spike patterns[10], which

614 may contribute to the observed performance gains. Investigating the underlying relationship between

615 spike diversity and task performance remains an important direction for future research.

616 **D.3** **Broader Impacts**

617 This paper focuses on the fundamental research of spiking neural networks, introducing a top-down

618 feedback structure that aims to enhance their performance. Generally, there are no negative societal

619 impacts in this work.

28

620 **NeurIPS Paper Checklist**

621 The checklist is designed to encourage best practices for responsible machine learning research,

622 addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove

623 the checklist: **The papers not including the checklist will be desk rejected.** The checklist should

624 follow the references and follow the (optional) supplemental material. The checklist does NOT count

625 towards the page limit.

626 Please read the checklist guidelines carefully for information on how to answer these questions. For

627 each question in the checklist:

628 - You should answer [Yes], [No], or [NA] .

629 - [NA] means either that the question is Not Applicable for that particular paper or the

630 relevant information is Not Available.

631 - Please provide a short (1–2 sentence) justification right after your answer (even for NA).

632 **The checklist answers are an integral part of your paper submission.** They are visible to the

633 reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it

634 (after eventual revisions) with the final version of your paper, and its final version will be published

635 with the paper.

636 The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.

637 While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a

638 proper justification is given (e.g., "error bars are not reported because it would be too computationally

639 expensive" or "we were unable to find the license for the dataset we used"). In general, answering

640 "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we

641 acknowledge that the true answer is often more nuanced, so please just use your best judgment and

642 write a justification to elaborate. All supporting evidence can appear either in the main paper or the

643 supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification

644 please point to the section(s) where related material for the question can be found.

645 IMPORTANT, please:

646 - **Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist"**,

647 - **Keep the checklist subsection headings, questions/answers and guidelines below.**

648 - **Do not modify the questions and only use the provided macros for your answers** .

649 1. **Claims**

650 Question: Do the main claims made in the abstract and introduction accurately reflect the

651 paper’s contributions and scope?

652 Answer: [Yes]

653 Justification:The abstract and introduction clearly state our contributions in the field of

654 spiking neural networks, including the discovery of limitation caused by SNN dynamics and

655 the inspired improvement methods.

656 Guidelines:

657 - The answer NA means that the abstract and introduction do not include the claims

658 made in the paper.

659 - The abstract and/or introduction should clearly state the claims made, including the

660 contributions made in the paper and important assumptions and limitations. A No or

661 NA answer to this question will not be perceived well by the reviewers.

662 - The claims made should match theoretical and experimental results, and reflect how

663 much the results can be expected to generalize to other settings.

664 - It is fine to include aspirational goals as motivation as long as it is clear that these goals

665 are not attained by the paper.

666 2. **Limitations**

667 Question: Does the paper discuss the limitations of the work performed by the authors?

668 Answer: [Yes]

29

669 Justification: We discussed the limitations of the proposed method in the appendix.

670 Guidelines:

671 - The answer NA means that the paper has no limitation while the answer No means that

672 the paper has limitations, but those are not discussed in the paper.

673 - The authors are encouraged to create a separate "Limitations" section in their paper.

674 - The paper should point out any strong assumptions and how robust the results are to

675 violations of these assumptions (e.g., independence assumptions, noiseless settings,

676 model well-specification, asymptotic approximations only holding locally). The authors

677 should reflect on how these assumptions might be violated in practice and what the

678 implications would be.

679 - The authors should reflect on the scope of the claims made, e.g., if the approach was

680 only tested on a few datasets or with a few runs. In general, empirical results often

681 depend on implicit assumptions, which should be articulated.

682 - The authors should reflect on the factors that influence the performance of the approach.

683 For example, a facial recognition algorithm may perform poorly when image resolution

684 is low or images are taken in low lighting. Or a speech-to-text system might not be

685 used reliably to provide closed captions for online lectures because it fails to handle

686 technical jargon.

687 - The authors should discuss the computational efficiency of the proposed algorithms

688 and how they scale with dataset size.

689 - If applicable, the authors should discuss possible limitations of their approach to

690 address problems of privacy and fairness.

691 - While the authors might fear that complete honesty about limitations might be used by

692 reviewers as grounds for rejection, a worse outcome might be that reviewers discover

693 limitations that aren’t acknowledged in the paper. The authors should use their best

694 judgment and recognize that individual actions in favor of transparency play an impor
695 tant role in developing norms that preserve the integrity of the community. Reviewers

696 will be specifically instructed to not penalize honesty concerning limitations.

697 3. **Theory assumptions and proofs**

698 Question: For each theoretical result, does the paper provide the full set of assumptions and

699 a complete (and correct) proof?

700 Answer: [Yes]

701 Justification: The paper provides a complete proof of the proposed viewpoint and method.

702 Guidelines:

703 - The answer NA means that the paper does not include theoretical results.

704 - All the theorems, formulas, and proofs in the paper should be numbered and cross
705 referenced.

706 - All assumptions should be clearly stated or referenced in the statement of any theorems.

707 - The proofs can either appear in the main paper or the supplemental material, but if

708 they appear in the supplemental material, the authors are encouraged to provide a short

709 proof sketch to provide intuition.

710 - Inversely, any informal proof provided in the core of the paper should be complemented

711 by formal proofs provided in appendix or supplemental material.

712 - Theorems and Lemmas that the proof relies upon should be properly referenced.

713 4. **Experimental result reproducibility**

714 Question: Does the paper fully disclose all the information needed to reproduce the main ex
715 perimental results of the paper to the extent that it affects the main claims and/or conclusions

716 of the paper (regardless of whether the code and data are provided or not)?

717 Answer: [Yes]

718 Justification: The method section provides a detailed introduction to the method proposed in

719 this paper, which can be reproduced by referring to the experiment section and submitted

720 code.

721 Guidelines:

30

722 - The answer NA means that the paper does not include experiments.

723 - If the paper includes experiments, a No answer to this question will not be perceived

724 well by the reviewers: Making the paper reproducible is important, regardless of

725 whether the code and data are provided or not.

726 - If the contribution is a dataset and/or model, the authors should describe the steps taken

727 to make their results reproducible or verifiable.

728 - Depending on the contribution, reproducibility can be accomplished in various ways.

729 For example, if the contribution is a novel architecture, describing the architecture fully

730 might suffice, or if the contribution is a specific model and empirical evaluation, it may

731 be necessary to either make it possible for others to replicate the model with the same

732 dataset, or provide access to the model. In general. releasing code and data is often

733 one good way to accomplish this, but reproducibility can also be provided via detailed

734 instructions for how to replicate the results, access to a hosted model (e.g., in the case

735 of a large language model), releasing of a model checkpoint, or other means that are

736 appropriate to the research performed.

737 - While NeurIPS does not require releasing code, the conference does require all submis
738 sions to provide some reasonable avenue for reproducibility, which may depend on the

739 nature of the contribution. For example

740 (a) If the contribution is primarily a new algorithm, the paper should make it clear how

741 to reproduce that algorithm.

742 (b) If the contribution is primarily a new model architecture, the paper should describe

743 the architecture clearly and fully.

744 (c) If the contribution is a new model (e.g., a large language model), then there should

745 either be a way to access this model for reproducing the results or a way to reproduce

746 the model (e.g., with an open-source dataset or instructions for how to construct

747 the dataset).

748 (d) We recognize that reproducibility may be tricky in some cases, in which case

749 authors are welcome to describe the particular way they provide for reproducibility.

750 In the case of closed-source models, it may be that access to the model is limited in

751 some way (e.g., to registered users), but it should be possible for other researchers

752 to have some path to reproducing or verifying the results.

753 5. **Open access to data and code**

754 Question: Does the paper provide open access to the data and code, with sufficient instruc
755 tions to faithfully reproduce the main experimental results, as described in supplemental

756 material?

757 Answer: [Yes]

758 Justification: The dataset used in this article is publicly available, and the code will be made

759 public to ensure that others can reproduce the experimental results.

760 Guidelines:

761 - The answer NA means that paper does not include experiments requiring code.

762 - Please see the NeurIPS code and data submission guidelines ( `[https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy)`

763 `[public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)` ) for more details.

764 - While we encourage the release of code and data, we understand that this might not be

765 possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not

766 including code, unless this is central to the contribution (e.g., for a new open-source

767 benchmark).

768 - The instructions should contain the exact command and environment needed to run to

769 reproduce the results. See the NeurIPS code and data submission guidelines ( `[https:](https://nips.cc/public/guides/CodeSubmissionPolicy)`

770 `[//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)` ) for more details.

771 - The authors should provide instructions on data access and preparation, including how

772 to access the raw data, preprocessed data, intermediate data, and generated data, etc.

773 - The authors should provide scripts to reproduce all experimental results for the new

774 proposed method and baselines. If only a subset of experiments are reproducible, they

775 should state which ones are omitted from the script and why.

31

776 - At submission time, to preserve anonymity, the authors should release anonymized

777 versions (if applicable).

778 - Providing as much information as possible in supplemental material (appended to the

779 paper) is recommended, but including URLs to data and code is permitted.

780 6. **Experimental setting/details**

781 Question: Does the paper specify all the training and test details (e.g., data splits, hyper
782 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the

783 results?

784 Answer: [Yes]

785 Justification: The appendix of the paper provides detailed experimental settings.

786 Guidelines:

787 - The answer NA means that the paper does not include experiments.

788 - The experimental setting should be presented in the core of the paper to a level of detail

789 that is necessary to appreciate the results and make sense of them.

790 - The full details can be provided either with the code, in appendix, or as supplemental

791 material.

792 7. **Experiment statistical significance**

793 Question: Does the paper report error bars suitably and correctly defined or other appropriate

794 information about the statistical significance of the experiments?

795 Answer: [Yes]

796 Justification: The paper accurately presents error bars for the execution speed benchmark.

797 Notably, our experiments involved comparing our method’s optimal performance with other

798 approaches

799 Guidelines:

800 - The answer NA means that the paper does not include experiments.

801 - The authors should answer "Yes" if the results are accompanied by error bars, confi
802 dence intervals, or statistical significance tests, at least for the experiments that support

803 the main claims of the paper.

804 - The factors of variability that the error bars are capturing should be clearly stated (for

805 example, train/test split, initialization, random drawing of some parameter, or overall

806 run with given experimental conditions).

807 - The method for calculating the error bars should be explained (closed form formula,

808 call to a library function, bootstrap, etc.)

809 - The assumptions made should be given (e.g., Normally distributed errors).

810 - It should be clear whether the error bar is the standard deviation or the standard error

811 of the mean.

812 - It is OK to report 1-sigma error bars, but one should state it. The authors should

813 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis

814 of Normality of errors is not verified.

815 - For asymmetric distributions, the authors should be careful not to show in tables or

816 figures symmetric error bars that would yield results that are out of range (e.g. negative

817 error rates).

818 - If error bars are reported in tables or plots, The authors should explain in the text how

819 they were calculated and reference the corresponding figures or tables in the text.

820 8. **Experiments compute resources**

821 Question: For each experiment, does the paper provide sufficient information on the com
822 puter resources (type of compute workers, memory, time of execution) needed to reproduce

823 the experiments?

824 Answer: [Yes]

825 Justification: The article provides the resource cost required for conducting experiments,

826 further detailed information is provided in the code.

32

827 Guidelines:

828 - The answer NA means that the paper does not include experiments.

829 - The paper should indicate the type of compute workers CPU or GPU, internal cluster,

830 or cloud provider, including relevant memory and storage.

831 - The paper should provide the amount of compute required for each of the individual

832 experimental runs as well as estimate the total compute.

833 - The paper should disclose whether the full research project required more compute

834 than the experiments reported in the paper (e.g., preliminary or failed experiments that

835 didn’t make it into the paper).

836 9. **Code of ethics**

837 Question: Does the research conducted in the paper conform, in every respect, with the

838 NeurIPS Code of Ethics `[https://neurips.cc/public/EthicsGuidelines](https://neurips.cc/public/EthicsGuidelines)` ?

839 Answer: [Yes]

840 Justification: The research in this paper adheres to the NeurIPS Code of Ethics.

841 Guidelines:

842 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

843 - If the authors answer No, they should explain the special circumstances that require a

844 deviation from the Code of Ethics.

845 - The authors should make sure to preserve anonymity (e.g., if there is a special consid
846 eration due to laws or regulations in their jurisdiction).

847 10. **Broader impacts**

848 Question: Does the paper discuss both potential positive societal impacts and negative

849 societal impacts of the work performed?

850 Answer: [NA]

851 Justification: This paper focuses on the fundamental research of spiking neural networks,

852 there are no negative societal impacts in this work.

853 Guidelines:

854 - The answer NA means that there is no societal impact of the work performed.

855 - If the authors answer NA or No, they should explain why their work has no societal

856 impact or why the paper does not address societal impact.

857 - Examples of negative societal impacts include potential malicious or unintended uses

858 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations

859 (e.g., deployment of technologies that could make decisions that unfairly impact specific

860 groups), privacy considerations, and security considerations.

861 - The conference expects that many papers will be foundational research and not tied

862 to particular applications, let alone deployments. However, if there is a direct path to

863 any negative applications, the authors should point it out. For example, it is legitimate

864 to point out that an improvement in the quality of generative models could be used to

865 generate deepfakes for disinformation. On the other hand, it is not needed to point out

866 that a generic algorithm for optimizing neural networks could enable people to train

867 models that generate Deepfakes faster.

868 - The authors should consider possible harms that could arise when the technology is

869 being used as intended and functioning correctly, harms that could arise when the

870 technology is being used as intended but gives incorrect results, and harms following

871 from (intentional or unintentional) misuse of the technology.

872 - If there are negative societal impacts, the authors could also discuss possible mitigation

873 strategies (e.g., gated release of models, providing defenses in addition to attacks,

874 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from

875 feedback over time, improving the efficiency and accessibility of ML).

876 11. **Safeguards**

877 Question: Does the paper describe safeguards that have been put in place for responsible

878 release of data or models that have a high risk for misuse (e.g., pretrained language models,

879 image generators, or scraped datasets)?

33

880 Answer: [NA]

881 Justification: This paper focuses on the fundamental research of spiking neural networks,

882 which does not involve the development or release of data or models that have a high risk

883 for misuse.

884 Guidelines:

885 - The answer NA means that the paper poses no such risks.

886 - Released models that have a high risk for misuse or dual-use should be released with

887 necessary safeguards to allow for controlled use of the model, for example by requiring

888 that users adhere to usage guidelines or restrictions to access the model or implementing

889 safety filters.

890 - Datasets that have been scraped from the Internet could pose safety risks. The authors

891 should describe how they avoided releasing unsafe images.

892 - We recognize that providing effective safeguards is challenging, and many papers do

893 not require this, but we encourage authors to take this into account and make a best

894 faith effort.

895 12. **Licenses for existing assets**

896 Question: Are the creators or original owners of assets (e.g., code, data, models), used in

897 the paper, properly credited and are the license and terms of use explicitly mentioned and

898 properly respected?

899 Answer: [Yes]

900 Justification: The creators or original owners of the assets (such as code, data, and models)

901 used in this paper have been properly credited. Their contributions have been explicitly

902 mentioned in an appropriate manner. Additionally, the license and terms of use for each asset

903 have been explicitly stated and adhered to, including obtaining any necessary permissions or

904 authorizations.

905 Guidelines:

906 - The answer NA means that the paper does not use existing assets.

907 - The authors should cite the original paper that produced the code package or dataset.

908 - The authors should state which version of the asset is used and, if possible, include a

909 URL.

910 - The name of the license (e.g., CC-BY 4.0) should be included for each asset.

911 - For scraped data from a particular source (e.g., website), the copyright and terms of

912 service of that source should be provided.

913 - If assets are released, the license, copyright information, and terms of use in the

914 package should be provided. For popular datasets, `paperswithcode.com/datasets`

915 has curated licenses for some datasets. Their licensing guide can help determine the

916 license of a dataset.

917 - For existing datasets that are re-packaged, both the original license and the license of

918 the derived asset (if it has changed) should be provided.

919 - If this information is not available online, the authors are encouraged to reach out to

920 the asset’s creators.

921 13. **New assets**

922 Question: Are new assets introduced in the paper well documented and is the documentation

923 provided alongside the assets?

924 Answer: [Yes]

925 Justification: The experimental code will be made openly accessible, along with the neces
926 sary documents to facilitate reproducibility of the experimental results and utilization of the

927 code for future work.

928 Guidelines:

929 - The answer NA means that the paper does not release new assets.

930 - Researchers should communicate the details of the dataset/code/model as part of their

931 submissions via structured templates. This includes details about training, license,

932 limitations, etc.

34

933 - The paper should discuss whether and how consent was obtained from people whose

934 asset is used.

935 - At submission time, remember to anonymize your assets (if applicable). You can either

936 create an anonymized URL or include an anonymized zip file.

937 14. **Crowdsourcing and research with human subjects**

938 Question: For crowdsourcing experiments and research with human subjects, does the paper

939 include the full text of instructions given to participants and screenshots, if applicable, as

940 well as details about compensation (if any)?

941 Answer: [NA]

942 Justification: This paper does not involve crowdsourcing experiments or research with

943 human subjects.

944 Guidelines:

945 - The answer NA means that the paper does not involve crowdsourcing nor research with

946 human subjects.

947 - Including this information in the supplemental material is fine, but if the main contribu
948 tion of the paper involves human subjects, then as much detail as possible should be

949 included in the main paper.

950 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation,

951 or other labor should be paid at least the minimum wage in the country of the data

952 collector.

953 15. **Institutional** **review** **board** **(IRB)** **approvals** **or** **equivalent** **for** **research** **with** **human**

954 **subjects**

955 Question: Does the paper describe potential risks incurred by study participants, whether

956 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)

957 approvals (or an equivalent approval/review based on the requirements of your country or

958 institution) were obtained?

959 Answer: [Yes]

960 Justification: The paper does not involve crowdsourcing nor research with human subjects.

961 Guidelines:

962 - The answer NA means that the paper does not involve crowdsourcing nor research with

963 human subjects.

964 - Depending on the country in which research is conducted, IRB approval (or equivalent)

965 may be required for any human subjects research. If you obtained IRB approval, you

966 should clearly state this in the paper.

967 - We recognize that the procedures for this may vary significantly between institutions

968 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the

969 guidelines for their institution.

970 - For initial submissions, do not include any information that would break anonymity (if

971 applicable), such as the institution conducting the review.

972 16. **Declaration of LLM usage**

973 Question: Does the paper describe the usage of LLMs if it is an important, original, or

974 non-standard component of the core methods in this research? Note that if the LLM is used

975 only for writing, editing, or formatting purposes and does not impact the core methodology,

976 scientific rigorousness, or originality of the research, declaration is not required.

977 Answer: [NA]

978 Justification: The LLM was only used for translation purposes and did not affect the core

979 scientific methodology, analysis, or originality of the research.

980 Guidelines:

981 - The answer NA means that the core method development in this research does not

982 involve LLMs as any important, original, or non-standard components.

983 - Please refer to our LLM policy ( `[https://neurips.cc/Conferences/2025/LLM](https://neurips.cc/Conferences/2025/LLM)` )

984 for what should or should not be described.

35

