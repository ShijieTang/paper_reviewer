# Mechanistic Insights Into Grokking From The Embedding Layer

Anonymous Author(s)
Affiliation Address email

## Abstract

1 Grokking, a delayed generalization in neural networks after perfect training per2 formance, has been observed in Transformers and MLPs, but the components 3 driving it remain underexplored. We show that embeddings are central to grokking: 4 introducing them into MLPs induces delayed generalization in modular arithmetic 5 tasks, whereas MLPs without embeddings can generalize immediately. Our analy6 sis identifies two key mechanisms: (1) Embedding update dynamics, where rare 7 tokens stagnate due to sparse gradient updates and weight decay, and (2) Bilinear 8 coupling, where the interaction between embeddings and downstream weights 9 introduces saddle points and increases sensitivity to initialization. To confirm 10 these mechanisms, we investigate frequency-aware sampling, which balances token 11 updates by minimizing gradient variance, and embedding-specific learning rates, 12 derived from the asymmetric curvature of the bilinear loss landscape. We prove that an adaptive learning rate ratio, ηE
ηW
∝
σmax(E)
σmax(W)
·
fW
fE
13 , mitigates bilinear cou14 pling effects, accelerating convergence. Our methods not only improve grokking 15 dynamics but also extend to broader challenges in Transformer optimization, where 16 bilinear interactions hinder efficient training.

## 17 **1 Introduction**

18 The phenomenon of grokking, in which a neural network exhibits delayed generalization after 19 achieving close to or perfect training performance, has emerged as a compelling topic in deep learning. 20 Initially observed in Transformer architectures by [19], grokking presents a puzzling challenge 21 where models that seem to overfit to training data eventually demonstrate remarkable generalization 22 capabilities after extensive training. Subsequent research has identified this phenomenon across 23 various architectures, including convolutional neural networks (CNNs) and multi-layer perceptrons 24 (MLPs) [13, 12]. Despite growing interest, the underlying mechanisms of grokking remain elusive. 25 Existing studies have sought to unravel grokking by exploring its connection to delayed robustness, 26 local complexity, and model architecture [3, 6]. For instance, [6] suggest that grokking coincides with 27 a phase transition in the linear regions of a model's input space, leading to robust partitions that enable 28 generalization after extended training. Others have attributed grokking to emergent circuit behaviors 29 or optimization dynamics [17, 21]. However, these studies often focus on high-level phenomena, 30 overlooking the role of specific components, such as embedding layers, in shaping the dynamics of 31 grokking. 32 In this work, we argue that embedding layers are central to understanding the grokking phenomenon. 33 By introducing embedding layers into MLP architectures, we observe clear grokking patterns even in 34 simple modular arithmetic tasks, such as modular addition. Interestingly, MLPs without embedding 35 layers can often generalize without grokking, suggesting that embeddings introduce unique dynamics 36 that delay generalization. Our analysis identifies two critical factors that influence these dynamics:
Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

37 1. **Embedding update dynamics:** Embedding parameters are updated through gradient de38 scent and weight decay. However, embeddings corresponding to tokens not present in a 39 given batch are updated solely via weight decay or residual effects from previous gradi40 ents in optimizers like Adam. This imbalance delays stabilization and can hinder training, 41 particularly for low-probability tokens. 42 2. **Coupling with the first-layer weights:** When embeddings are multiplied with the weights 43 of the first layer, they form a bilinear interaction. This coupling introduces structural 44 complexity into the optimization landscape, making the process more susceptible to saddle 45 points and increasing the sensitivity to initialization.

60 - Highlighting the unique role of embedding layers in delaying generalization and their 61 coupling with the first layer in MLPs. 62 - Proposing strategies to accelerate grokking, including refined sampling and embedding63 specific learning rates. 64 - Connecting the challenges in embedding-based optimization to broader issues in Transformer 65 training, such as bilinearity, saddle points, and the effectiveness of adaptive optimizers like 66 Adam. 67 By bridging insights from grokking and Transformer optimization, we provide a unified perspective 68 on the interplay between embedding dynamics, optimization challenges, and generalization.

## 69 **2 Related Work**

70 The phenomenon of grokking, where generalization emerges abruptly after prolonged overfitting, was 71 first observed in transformers [19] and later extended to CNNs and ResNets [13, 12], indicating it is 72 architecture-agnostic. Various explanations have been proposed. [7] attribute it to phase transitions in 73 local complexity ("delayed robustness"), while others link it to circuit efficiency [17, 21, 11]. Though 74 insightful, these perspectives don't fully explain the delayed generalization. Connections to double 75 descent have also been explored [1, 16], but grokking's dynamics remain distinct. 76 The closest work to ours studies modular addition using permutation-equivariant models [15], where 77 one-hot inputs interact with the first layer as a fixed embedding. Their analysis, however, is limited to 78 modular tasks and specific activations. In contrast, we generalize across datasets and highlight how 79 embedding layers, especially when trainable, interact bilinearly with downstream weights, affecting 80 optimization dynamics. 81 Related studies like Tensor Programs IV [24] prescribe per-layer scaling based on width, assuming 82 independent layer evolution. Our setup differs: the embedding layer's updates depend on both its 83 own width and the spectrum of the coupled layer. Prieto et al. [20] connect delayed generalization to 84 numerical instability (Softmax Collapse), proposing solutions that complement our focus on structural 85 coupling and gradient imbalance.

86 Unlike works that focus on final representations [4], we analyze the embedding layer's evolving role 87 during training. Even with one-hot inputs, its interaction with the first linear layer forms a learnable 46 Building on these insights, we propose two strategies to address and prove the hypotheses introduced 47 for embedding layers. **First**: A refined sampling methodology that ensures more uniform updates 48 across all embeddings, mitigating frequency imbalance. **Second**: A learning rate adjustment for 49 embeddings, setting it higher than that of the rest of the model. This adjustment counteracts the 50 coupling effect with the first-layer weights, enabling faster stabilization and reducing the risk of 51 optimization stagnation. Our experiments demonstrate that these strategies not only accelerate the 52 grokking process but also enable generalization in scenarios where traditional approaches fail. 53 Additionally, the bilinear coupling observed in embedding-based MLPs highlights broader challenges 54 in optimizing Transformer architectures. Transformers, which rely on multiplicative interactions in 55 attention mechanisms, exhibit similar issues due to the bilinearity of query, key, and value projections. 56 While softmax attention and scaling by the dimensionality d help smooth the optimization landscape, 57 these mechanisms may still struggle with increased saddle points in certain layers [5]. In summary, 58 this work contributes to the understanding of grokking and its broader implications for deep learning 59 by:

## 96 **3 Preliminaries** 97 **3.1 Embedding Layers**

98 The Transformer model [22] utilizes a self-attention 99 mechanism to capture dependencies between tokens. In 100 this framework, embeddings map input tokens to high101 dimensional vectors, which are processed through atten102 tion layers. These embeddings help the model capture 103 contextualized representations. In contrast, MLPs rely on 104 fully connected layers without attention mechanisms. We 105 investigate the role of embeddings in MLPs, specifically 106 how they improve model generalization. The core contri107 bution of this work is to examine the role of embedding 108 layers in MLPs. These layers map discrete tokens to dense, 109 high-dimensional vectors, enabling models to handle non110 linear tasks like modular arithmetic. Even with one-hot 111 inputs—as studied in theoretical settings [2, 15]—the first 112 weight matrix effectively functions as a learned embed113 ding. Thus, embeddings, whether explicit or implicit, play 114 a central role in shaping model dynamics. While com115 monly associated with Transformers, we focus on MLPs 116 as a simpler and more interpretable setting. MLPs avoid 117 the added complexity of self-attention while still exhibit118 ing phenomena like grokking. Importantly, the bilinear 119 coupling between embeddings and downstream weights, 120 central to our analysis, also arises in Transformers but 121 is further complicated by attention. Studying MLPs al122 lows us to isolate and understand this coupling in a clean, 123 controlled environment.

0 Addition mod (6)
0 1 2 3 4 5 5 4 3 2 1 0 1 2 3 4 5
(a)
Multiplication mod (7)
1 2 3 4 5 6 6 5 4 3 2 1 1 2 3 4 5 6
(b)
Figure 1: Heatmaps for (a) additive group (mod 6) and (b) multiplicative group (mod 7). The two groups are isomorphic despite differing appearances.

## 124 **3.2 Algorithmic Datasets And Modular Arithmetic**

125 Algorithmic datasets are synthetic datasets carefully con126 structed with controlled mathematical properties, typically 127 involving operations over finite sets such as modular ad128 dition or multiplication. One well-known example is the 129 modular arithmetic dataset studied by [19], where the goal is to uncover relationships between binary 130 inputs and produce consistent outputs based on these operations. For instance, given inputs a and 131 b, the model is tasked to compute (a + b)mod P or (a × b)mod P, where P is a prime number, and 132 both inputs and outputs are constrained within {0, 1*, . . . , P* − 1} (refer to Figure 1).

133 This dataset highlights the challenging nature of generalization in grokking: the relationship between 134 inputs is defined purely by a deterministic operation, not by a probabilistic distribution. Unlike 135 typical machine learning datasets, where examples are drawn from an underlying (often unknown) 136 data distribution, algorithmic datasets consist of a finite and complete set of all possible input-output 137 combinations. In such cases, there is no statistical "distribution" in the conventional sense; instead, the 138 generalization task relies on uncovering the underlying relationship between inputs, which demands a 139 model to internalize the algorithm itself. Moreover, any hypothesis consistent with training examples 140 can initially seem plausible from a statistical perspective, as no known distribution governs the data.

88 embedding mechanism. Concurrent work shows that transferring embeddings from small to large 89 models can accelerate grokking [23]; while we share this motivation, we also observe in preliminary 90 trials that transferring other MLP layers may offer similar benefits. 91 Finally, the bilinear coupling we analyze in MLPs parallels challenges in Transformer architectures, 92 where attention mechanisms introduce similar multiplicative dynamics. Prior work highlights how 93 adaptive optimizers like Adam outperform SGD due to gradient noise and curvature heterogeneity 94 [25, 10, 26]. Our findings help bridge these perspectives by showing how embedding-layer coupling 95 shapes optimization and generalization.

141 The difficulty of generalization thus lies not in interpolating unseen samples but in discovering the 142 underlying relation, making it a fundamentally different task. 143 We note that there is an equivalence between modular addition and modular multiplication in certain 144 settings. Namely, given a prime number p, the groups (in mathematical sense) of modular addition
{0, 1*, . . . , p* − 2}, +
145 (where addition is performed modulo p − 1), and of modular multiplication
{1*, . . . , p* − 1}, ∗
146 (where multiplication is performed modulo p) are isomorphic. Both groups have 147 the same number of elements (which is p − 1), and are simple (meaning, there is an element g, called 148 generator, such that every other element is of the form g *∗ · · · ∗* g, where ∗ is the group operation and 149 the number of operations used is less than p. In the first group, any element different from 0 is the 150 group generator while in the second group, any element different from 1 is the generator (see Figure 151 1). 152 The embedding layer strips the input group elements of their numerical meanings, and assigns a 153 general, abstract vector to each element. In this way, training on modular addition or multiplication 154 presents no difference for MLP (or other architectures) with the embedding layer. In contrast to 155 this, the MLP without the embedding layer is able to fit and generalize on modular addition, while it 156 completely fails on modular multiplication.

## 157 **3.3 Problem Setup And Motivations**

Let D = {(xi, yi)}
N
i=1 represent an algorithmic dataset, where each xi 158 is an input token sequence
(e.g., *a, b,* operation, =), and yi 159 is the output derived from an operation modulo a positive integer P. 160 The task is to learn a mapping fθ : *X → Y* parameterized by θ, capable of generalizing to unseen samples from Dtest 161 .

162 To process inputs effectively, we tokenize them as sequences of their digit representations, as the 163 model does not inherently interpret numerical values. Each operand a and b is assigned a token in the 164 range 0 to P − 1, while the operation and equality symbols are represented by tokens P and P + 1, 165 respectively. For instance, the modular arithmetic expression (3 + 2)(mod 5) = 0 is tokenized as 166 [3, 5, 2, 6, 0]. 167 Embedding layers in models provide a dense representation of tokens. However, delayed updates to 168 embeddings for infrequent tokens can significantly impact convergence and generalization. Our work explores these dynamics, with a focus on the impact of pi, the i th 169 -token sampling probability, and 170 proposes adjustments to improve convergence. We investigate the use of embeddings in MLPs for 171 algorithmic tasks. We started by training a MLP on modular addition and multiplication datasets, 172 comparing setups with and without embedding layers.

173 **MLP Without Embeddings.** In this setup, input tokens (a, b, operation (P), and equality sign 174 (P + 1)) are encoded directly into a 4-dimensional input vector. The MLP processes these inputs as:

$\mathbf{h}_{1}=\sigma(\mathbf{W}_{1}x+\mathbf{b}_{1})$, $\mathbf{h}_{2}=\mathbf{W}_{2}\mathbf{h}_{1}+\mathbf{b}_{2}$, $\mathbf{\hat{y}}=\text{Softmax}(\mathbf{h}_{2})$.  
$\left(1\right)$. 
yˆ = Softmax(h2). (1)
where x ∈ R
4 175 is the encoded input vector (with first and third entry a and b, respectively), W1,W2 176 are weight matrices, b1, b2 are biases, σ is the ReLU activation function, and yˆ represents the 177 predicted output. 178 This configuration demonstrates that the MLP can fit the addition task with ease, but struggles to 179 generalize multiplication. This difficulty arises because multiplication modulo P is not linearly 180 separable, as evident in the non-trivial patterns in Figure 1. 181 **MLP With Embeddings.** To overcome the challenges of non-linear separability, we introduced 182 an embedding layer. Each token x is mapped to a dense vector ex through an embedding matrix E ∈ R
V ×d 183 , where d is the embedding dimension. Our input consists of 4 token embeddings of the form eˆ = [ei, e′∗′ , ek, e′=′ ]
⊤ 184 , and the modified forward pass is:

$$\begin{array}{l l}{{}}&{{\mathbf{h}_{1}=\sigma(\mathbf{W}\hat{\mathbf{e}}+\mathbf{b}_{1}),}}\\ {{}}&{{\mathbf{h}_{2}=\mathbf{W}_{2}\mathbf{h}_{1}+\mathbf{b}_{2},\quad\hat{\mathbf{y}}=\mathrm{Softmax}(\mathbf{h}_{2}),}}\end{array}$$
h2 = W2h1 + b2, yˆ = Softmax(h2), (2)
185 Adding embeddings allows the model to capture more expressive input representations. With this 186 setup, we observed that the model generalized well to both addition and multiplication tasks, but with 187 a delayed generalization for multiplication. This delay corresponds to the grokking phenomenon, 188 which appears as a "trapezoid pattern" in performance plots: a phase of memorization followed by a 189 sudden leap in test accuracy, as illustrated in figure 2 . 190 These observations motivate a deeper analysis of embedding dynamics during training. In particular, 191 we investigated the gradient heatmaps to understand the role of embeddings in delaying generalization. 192 By visualizing gradient magnitudes across training epochs, we point out that embeddings receive 193 smaller updates compared to other weights of the model, potentially causing grokking. This investi194 gation will help establish a connection between embedding behavior and the observed generalization 195 delays.

10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accur acy Train Test Accur acy Train Test Accur acy Train Test Accu racy Train Test 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8
Figure 2: Training and validation accuracies of the MLP model on modular arithmetic tasks, trained with Adam. *Left two:* Addition task, without (first) and with (second) embeddings. *Right two:* Multiplication task, without (third) and with (fourth) embeddings. In the embedding-free cases, training and validation accuracies increase together only for addition; multiplication fails to generalize. In contrast, models with embeddings reach 100% training accuracy in both tasks, but only begin generalizing after a delay exhibiting the grokking phenomenon.

## 196 **4 Main Results**

197 Our methodology investigates the dynamics of embedding layers within MLPs to address challenges 198 in generalization, particularly in the context of algorithmic tasks. The key contributions include: 199 (1) exploring the novel role of embedding layers attached to MLP architectures, (2) examining the 200 impact of embedding sampling probability pi on training dynamics, and (3) understanding how 201 initialization and the coupling of embedding and weight matrices affect learning efficiency. These 202 factors contribute to the grokking phenomenon, where generalization is delayed during training.

## 203 **4.1 Embedding Dynamics**

204 Let the loss function of the model be L(θ, E), where θ is model parameters other than embedding 205 weights. Let ei,t denote the embedding vector for token i at step t. Under stochastic gradient descent 206 (SGD) with weight decay λ, the embedding update rule is:
ei,t+1 − ei,t = −ηλei,t − η∇ei,tL, (3)
where η is the learning rate, and ∇eiL is the gradient1 207 . Token embeddings are updated using 208 corresponding gradients only when the associated tokens appear in a batch. Assume that token i being sampled in a batch with a probability pi 209 . Consequently, taking into account the randomness of 210 batch sampling, the expected update can be expressed as:
E[ei,t+1 − ei,t] = −ηλei,t − ηpi∇ei,tL. (4)
211 To summarize, the sampling probability pi directly influences the gradient dynamics of the embedding 212 layer. While gradients contribute to tokens only probabilistically, weight decay affects all embeddings 213 uniformly, leading to imbalances in parameter updates. This dynamic, visualized in Figure 3, 214 highlights the need for a deeper understanding of how pi affects convergence.

215 To analyze the reduction of the loss, we assume that the model's overall loss function L(θ, {ei}) is 216 β-smooth. This means it satisfies the following inequality for all updates:

$$\mathbb{E}[{\boldsymbol{e}}_{i,t+1}-{\boldsymbol{e}}_{i,t}]=-\eta\lambda{\boldsymbol{e}}_{i,t}-\eta p_{i}\nabla_{{\boldsymbol{e}}_{i,t}}{\mathcal{L}}.$$
$${\mathcal{L}}(\theta_{t+1},\{e_{i,t+1}\})\leq{\mathcal{L}}(\theta_{t},\{e_{i,t}\})+\langle\nabla{\mathcal{L}},\Delta\rangle+{\frac{\beta}{2}}\|\Delta\|^{2}.$$
$$({\mathfrak{I}})$$

217 where ∆ = (θt+1 − θt, ei,t+1 − ei,t). 218 Denote Lt := L(θt, {ei,t}) then taking expectations over randomness of batch sampling leads to the 219 following expected update:

E[Lt+1 − Lt] ≤ ∇θtL T(θt+1 + θt) −X V i=1 ∇ei,tL T E(ei,t+1 − ei,t) + β2 ∥∆∥ 2, (5)
$$\mathbf{\tau},\mathbf{e}_{i,t+1}-\mathbf{e}_{i,t}).$$
220 Substituting the embedding update based on equation 4 into the smoothness inequality,

## 225 **4.2 Dataset Splitting Strategies**

To further explore the role of pi 226 , we investigate how train-test splitting strategies affect its value 227 and, consequently, the grokking process. The train-test split determines the probability of token i 228 appearing in a batch. 229 We begin by assuming that the weight decay parameter λ is zero and that the learning rate η is uniform across all parameters. This reduces the optimization problem to focusing on pi 230 , under the constraints PV
i=1 231 pi = 1, pi ≥ 0 ∀i. Specifically, the optimal pi can be found by solving for the following:

$$\operatorname*{min}_{p_{i}|p_{i}\geq0,\sum p_{i}=1}-\eta\sum_{i=1}^{V}p_{i}\|\nabla_{\mathbf{e}_{i,t}}{\mathcal{L}}\|^{2}.$$
$$(7)$$

240 These splits enable us to regulate token sampling probabilities, offering a direct assessment of the 241 impact of pi on embedding convergence and grokking. Furthermore, Section 5.1 provides a detailed 242 experiments conducted on two algorithmic datasets.

222 and noting from the right hand side of the inequality above, pi plays important role in reduction of the expected loss. However, the dependence on pi 223 , is coupled with weight decay, which explains why 224 these two parameters are important to study more deeply to draw a conclusion about grokking.

$$-\eta\sum_{i=1}^{V}\left(p_{i}\|\nabla_{\mathbf{e}_{i,t}}\mathcal{L}\|^{2}+\lambda\mathbf{e}_{i,t}^{T}\nabla_{\mathbf{e}_{i,t}}\mathcal{L}\right)+\frac{\beta}{2}\|\Delta\|^{2},\tag{6}$$
$\hat{\varepsilon}$
221
235 1. **Uniform Sampling:** Distribute all combinations of a and b evenly across training and test 236 sets. 237 2. **Skewed Sampling:** Introduce a bias in the combinations of a that are distributed across 238 training and test sets. 239 3. **Random Sampling:** Randomly distribute the examples across training and test sets.

$$\mathbb{E}[{\mathcal{L}}_{t+1}-{\mathcal{L}}_{t}]\leq\nabla_{\theta_{t}}{\mathcal{L}}^{T}(\theta_{t+1}-\theta_{t})$$

232 However, solving this exactly is challenging in practice due to the need for estimating all embedding 233 gradient norms. Instead, we adopt approximate strategies for splitting the training data, guided by 234 various assumptions about the gradient structure (see Appendix A for details).

## 243 **4.3 Embedding Convergence And Initialization**

244 While the frequency of embedding updates plays 245 a crucial role in training dynamics, as demon246 strated in our experiments, it alone cannot fully 247 explain phenomena such as grokking after fit248 ting, its relationship to initialization, weight de249 cay, or the structure of the loss landscape. 250 Stabilization (or convergence) occurs when the
251 embedding ei reaches a steady state where the
252 updates become negligibly small, i.e., when the
253 change in the embedding ∥ei,t+1 − ei,t∥ is ap254 proximately zero. This condition implies that,
255 (ηλ)ei,t ≈ ηpi∇eiL. from equation 4.
256 For small learning rates (η ≪ 1), the embedding 257 updates behave like a continuous system, and we 258 can model this as a differential equation (along 259 every dimension):
$${\frac{d\mathbf{e}_{i}}{d t}}=-\lambda\mathbf{e}_{i}-p_{i}\nabla_{\mathbf{e}_{i}}{\mathcal{L}},$$
= −λei − pi∇eiL, (8)
260 where ∇eiL is the gradient of the loss function 261 with respect to the embedding i. Assuming that 262 the gradient ∇eiL stabilizes to a constant value 263 g, the solution to this equation is:

$$\mathbf{e}_{i}(t)=C e^{-\lambda t}-{\frac{\eta p g}{\lambda}},$$
, (9)
Batch Size = 64 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 64 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accu racy Accu racy Batch Size = 128 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 128 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accu racy Accu racy Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc urac y Acc urac y

(a) (a + b) mod p
(b) (a ÷ b) mod p

Figure 4: Sampling strategy comparison for two modular tasks—addition and division—across all batch sizes. Uniform sampling generalizes faster; skewed sampling fails to generalize due to token imbalance.

264 where C is an integration constant determined 265 by the initial conditions. As time t increases, 266 the embedding ei(t) converges to the equilibrium value ei(t) → −ηpg λ 267 . Thus, convergence 268 is achieved when ei(t) stabilizes around this equilibrium point. The time T to reach convergence is bounded as T ≥
1 λ ln  C
ϵ 269 , where ϵ is a small 270 threshold. In summary, convergence time is governed by the embedding gradient g, the weight decay 271 λ, and the initialization magnitude C: stronger gradients and larger λ accelerate convergence, while 272 larger initial values C slow it down. 273 In bilinear models such as MLPs and Transformers, embedding gradients are tightly coupled with 274 those of downstream weights (e.g., W), forming a feedback loop: poor updates to E degrade W, 275 and vice versa. To study the role of initialization in this dynamic, we tested two setups: frozen 276 embeddings, which led to slow convergence due to limited representational flexibility; and small 277 initial embeddings, which improved convergence by allowing stronger early gradients—an effect also 278 observed in prior work [26, 12], though without analyzing embedding-weight coupling. 279 Motivated by these observations, we propose the **Adam-LR Optimizer**, which adjusts the embedding 280 learning rate to balance update magnitudes between E and W. This coupling-aware scaling is 281 formalized below:
282 **Proposition 4.1.** Let E and W be the embedding matrix and first-layer weights. To equalize update scales under cross-entropy loss, the learning rate ratio c =
ηE
ηW
283 *should satisfy:*

$$c\propto{\frac{\sigma_{\operatorname*{max}}(\mathbf{E})}{\sigma_{\operatorname*{max}}(\mathbf{W})}}\cdot{\frac{f_{W}}{f_{E}}},$$

284 *where* σmax(·) denotes the largest singular value and fE, fW are the respective update frequen285 *cies,(see appendix B for details).*
286 In practice, we set c = 10, guided by empirical singular value trends and supported by sensitivity 287 analysis (see Fig. 7, §5.2). This adjustment improves convergence and stability, especially under 288 sparse embedding updates common in skewed token distributions.

10 1 10 2 10 3 10 4 Optimization steps 0.0 0.5 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.5 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.5 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.5 1.0 Acc uracy adam - Train adam - Test adam_lr - Train adam_lr - Test Acc uracy adam - Train adam - Test adam_lr - Train adam_lr - Test Acc uracyadam - Train adam - Test adam_lr - Train adam_lr - Test Acc uracyadam - Train adam - Test adam_lr - Train adam_lr - Test
(d) (a 2 + b 2) mod p
(a) (a + b) mod p
(b) (a ÷ b) mod p
(c) (a × b) mod p

## 289 **5 Experiments And Discussions** 295 **5.1 The Effect Of Embedding Probability** 317 **5.2 Comparison Of Optimizers**

296 The first set of experiments investigates various strategies for splitting the training and testing datasets. 297 Specifically, we explore three approaches, namely; uniform sampling, skewed sampling, and random 298 sampling. 299 The expression (a + b) mod p represents the sum of a and b modulo p. For our experiments, we 300 randomly set aside 20% of the data as a test set, ensuring that evaluation is performed on unseen 301 samples. From the remaining data, 30/80% (i.e. 30% from total set) is sampled as the training set 302 according to each sampling strategy. 303 Figure 4 compare the performance of the sampling methods (random, uniform, skew) across different 304 splits of the dataset (see appendix D.1 for further datasets and settings). Each represents a specific 305 datasets, while the rows compare batch sizes, and columns compare datasets. The x-axis is logarithmic 306 to emphasize the convergence trends.

307 Uniform sampling generally promotes faster generalization and convergence compared to random 308 sampling. However, its benefits diminish at larger batch sizes (e.g., beyond 512), where random 309 sampling becomes nearly as effective due to broader token coverage. Crucially, our results show 310 that skewed sampling—despite fitting the training data and preserving the overall train-test ra311 tio—consistently leads to suboptimal generalization. This suggests that models can converge to lower 312 subaccuracy plateaus when token probabilities are heavily imbalanced. Importantly, even uniform 313 sampling does not guarantee optimality: unless the batch size is sufficiently large, some tokens may 314 be consistently omitted from updates. These findings underscore that token probability, both in 315 expectation and in per-batch coverage, plays a central role in embedding dynamics and grokking 316 behavior. 318 To evaluate the effectiveness of our proposed optimizer, Adam-LR, which incorporates a simple yet 319 effective strategy for treating the embedding layer differently to avoid stagnation or saddle points, 320 we conducted experiments on four datasets. The results are shown in Figure 5, where we compare 321 the performance of the two optimizers, Adam-LR and the standard Adam optimizer, under identical 322 training settings (lr = 0.01, batch size = 512).

323 Using our proposed optimizer, Adam-LR, which scales the embedding learning rate by a factor of 10, 324 the results demonstrate a significant acceleration in the grokking process compared to the baseline 325 Adam optimizer across all datasets.

290 We begin our exploration with a MLP model. The architecture consists of two layers, where the 291 hidden dimension of the first layer is set to four times the embedding dimension (where four is the 292 sequence length), and embedding dimension is set to 128, as per prior work on grokking. The second 293 layer has a dimension of P = 97. The activation function used throughout is ReLU, and optimization 294 is performed using the Adam optimizer with a weight decay of 0.001.

0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 H

e s sia n Eig e n v alu e w.r.t. 

W

E
W Training 100%
Validation 100%
10 3 10 4 Optimizaton steps 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 H

e s sia n Eig e n v alu e w.r.t. 

W

E
W Training 100%
Validation 100%
0.0 0.2 0.4 0.6 0.8 1.0 H

e s sia n Eig e n v alu e w.r.t. 

t o E

10 3 10 4 Optimizaton steps 0.00 0.02 0.04 0.06 0.08 0.10 0.12 H

e s sia n Eig e n v alu e w.r.t. 

t o E

Figure 6: Maximum eigenvalues of the Hessian with respect to embedding weights (E) and downstream weights (W) during training. The left plot corresponds to the Adam optimizer, while the right plot uses Adam_lr optimizer (ours). With Adam (left), the eigenvalues for E are significantly smaller than those for W, reflecting differences in dimensionality and update frequency. In contrast, with Adam_lr (right), the eigenvalues of W are notably reduced and become closer to those of E, suggesting a more balanced optimization dynamic. Training accuracy reaches 100% when the eigenvalues of W begin to decrease, while validation accuracy improves as the eigenvalues of E decrease. This suggests that W drives early optimization progress, while E fine-tunes generalization. The Adam_lr optimizer (ours) appears to regularize W, leading to a more stable training process.

## 326 **5.3 Analysis Of Singular Values Of Embedding Layer**

327 Prior work attributes Adam's superiority over SGD in Transformers to factors like gradient noise, 328 descent direction, and Hessian block heterogeneity [25, 10, 18, 26]. However, these studies largely 329 overlook the role of embeddings and their bilinear interactions. Our analysis supports the view that 330 such bilinear structure, especially in embeddings, contributes significantly to the observed curvature 331 differences (see appendix C.1 for more discussion). 332 To analyze the curvature of the loss landscape, we compute the maximum eigenvalue 333 of the Hessian matrix using the power method with Hessian-vector products (HVPs). 334 Figure 6 shows the maximum eigenvalues of 335 the Hessian with respect to E and W during 336 training. The results highlight distinct curvature 337 properties for E and W, reflecting their roles in 338 the bilinear interaction.

## 339 **6 Discussions**

340 In this study, we explored the interplay between 341 embedding layers and downstream weights in 342 neural networks, highlighting how their bilin343 ear coupling influences optimization and drives 344 the grokking phenomenon. We demonstrated 345 that embedding layers play a central role in de346 layed generalization and introduced the Adam347 LR optimizer to address the imbalance in update 348 dynamics, scaling the embedding learning rate 349 based on singular values and update frequencies. 350 A key limitation of this work is its focus on 351 MLPs, which provide a simplified setting for 352 analyzing embedding-weight coupling. While 353 this enables controlled analysis, it leaves open 354 how these insights transfer to more complex ar355 chitectures such as Transformers, where similar 356 bilinear interactions appear in attention mechanisms but with added structural complexity. Extending 357 our framework to the Transformer setting is a promising direction for future work.

Modp c=0.5 c=1.0 c=4.0 c=8.0 c=10.0 c=16.0 c=24.0 c=32.0 Amodp c=0.5 c=1.0 c=4.0 c=8.0 c=10.0 c=16.0 c=24.0 c=32.0 10 0 10 1 10 2 10 3 Optimization Steps 0.0 0.2 0.4 0.6 0.8 1.0 10 0 10 1 10 2 10 3 Optimization Steps 0.0 0.2 0.4 0.6 0.8 1.0 Test Acc urac y Test Acc urac y Dmodp c=0.5 c=1.0 c=4.0 c=8.0 c=10.0 c=16.0 c=24.0 c=32.0 Sum_squares_modp c=0.5 c=1.0 c=4.0 c=8.0 c=10.0 c=16.0 c=24.0 c=32.0 10 0 10 1 10 2 10 3 Optimization Steps 0.0 0.2 0.4 0.6 0.8 1.0 10 0 10 1 10 2 10 3 Optimization Steps 0.0 0.2 0.4 0.6 0.8 1.0 Test A
ccurac y Test A
ccurac y

## 358 **References**

359 [1] X. Davies, L. Langosco, and D. Krueger. Unifying grokking and double descent. *arXiv preprint* 360 *arXiv:2303.06173*, 2023.

361 [2] D. Doshi, T. He, A. Das, and A. Gromov. Grokking modular polynomials. *arXiv preprint* 362 *arXiv:2406.03495*, 2024.

363 [3] S. Fan, R. Pascanu, and M. Jaggi. Deep grokking: Would deep neural networks generalize 364 better? *arXiv preprint arXiv:2405.19454*, 2024. 365 [4] A. Gromov. Grokking modular arithmetic. *arXiv preprint arXiv:2301.02679*, 2023. 366 [5] X. S. Huang, F. Perez, J. Ba, and M. Volkovs. Improving transformer optimization through 367 better initialization. In *International Conference on Machine Learning*, pages 4475–4483. 368 PMLR, 2020. 369 [6] A. I. Humayun, R. Balestriero, and R. Baraniuk. Deep networks always grok and here is why. 370 *arXiv preprint arXiv:2402.15555*, 2024. 371 [7] A. Jeffares, A. Curth, and M. van der Schaar. Deep learning through a telescoping lens: A
372 simple model provides empirical insights on grokking, gradient boosting & beyond. *Advances* 373 *in Neural Information Processing Systems*, 37:123498–123533, 2024.

374 [8] S. Kobayashi, Y. Akram, and J. Von Oswald. Weight decay induces low-rank attention layers. 375 *Advances in Neural Information Processing Systems*, 37:4481–4510, 2024. 376 [9] T. Kumar. *Grokking as the transition from lazy to rich training dynamics*. PhD thesis, none, 377 2024. 378 [10] F. Kunstner, J. Chen, J. W. Lavington, and M. Schmidt. Noise is not the main factor behind 379 the gap between sgd and adam on transformers, but sign descent might be. *arXiv preprint* 380 *arXiv:2304.13960*, 2023. 381 [11] J. Lee, B. G. Kang, K. Kim, and K. M. Lee. Grokfast: Accelerated grokking by amplifying 382 slow gradients. *arXiv preprint arXiv:2405.20233*, 2024.

383 [12] Z. Liu, O. Kitouni, N. S. Nolte, E. Michaud, M. Tegmark, and M. Williams. Towards un384 derstanding grokking: An effective theory of representation learning. *Advances in Neural* 385 *Information Processing Systems*, 35:34651–34663, 2022. 386 [13] Z. Liu, E. J. Michaud, and M. Tegmark. Omnigrok: Grokking beyond algorithmic data. In The 387 *Eleventh International Conference on Learning Representations*, 2022. 388 [14] K. Lyu, J. Jin, Z. Li, S. S. Du, J. D. Lee, and W. Hu. Dichotomy of early and late phase implicit 389 biases can provably induce grokking. *arXiv preprint arXiv:2311.18817*, 2023. 390 [15] M. A. Mohamadi, Z. Li, L. Wu, and D. J. Sutherland. Why do you grok? a theoretical analysis 391 of grokking modular addition. *arXiv preprint arXiv:2407.12332*, 2024.

392 [16] P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, and I. Sutskever. Deep double descent: 393 Where bigger models and more data hurt. *Journal of Statistical Mechanics: Theory and* 394 *Experiment*, 2021(12):124003, 2021. 397 [18] Y. Pan and Y. Li. Toward understanding why adam converges faster than sgd for transformers. 398 *arXiv preprint arXiv:2306.00204*, 2023. 399 [19] A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking: Generalization beyond 400 overfitting on small algorithmic datasets. *arXiv preprint arXiv:2201.02177*, 2022.

401 [20] L. Prieto, M. Barsbey, P. A. Mediano, and T. Birdal. Grokking at the edge of numerical stability.

402 *arXiv preprint arXiv:2501.04697*, 2025. 395 [17] N. Nanda, L. Chan, T. Lieberum, J. Smith, and J. Steinhardt. Progress measures for grokking 396 via mechanistic interpretability. *arXiv preprint arXiv:2301.05217*, 2023. 403 [21] V. Varma, R. Shah, Z. Kenton, J. Kramár, and R. Kumar. Explaining grokking through circuit 404 efficiency. *arXiv preprint arXiv:2309.02390*, 2023. 405 [22] A. Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 406 2017. 407 [23] Z. Xu, Z. Ni, Y. Wang, and W. Hu. Let me grok for you: Accelerating grokking via embedding 408 transfer from a weaker model. *arXiv preprint arXiv:2504.13292*, 2025. 409 [24] G. Yang and E. J. Hu. Tensor programs iv: Feature learning in infinite-width neural networks. In 410 M. Meila and T. Zhang, editors, *Proceedings of the 38th International Conference on Machine* 411 *Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 11727–11737.

412 PMLR, 18–24 Jul 2021. 413 [25] J. Zhang, S. P. Karimireddy, A. Veit, S. Kim, S. Reddi, S. Kumar, and S. Sra. Why are adaptive 414 methods good for attention models? *Advances in Neural Information Processing Systems*, 415 33:15383–15393, 2020. 416 [26] Y. Zhang, C. Chen, T. Ding, Z. Li, R. Sun, and Z.-Q. Luo. Why transformers need adam: A 417 hessian perspective. *arXiv preprint arXiv:2402.16788*, 2024.

## 418 **Appendix** 419 **A Optimizing For Sampling Porbability** 420 **Uniform Importance Assumption**

If we assume that all gradients are equally important, i.e., ∥∇Ei,tL∥2
421 is uniform across all embed422 dings:
$$\|\nabla_{\mathbf{E}_{i,t}}{\mathcal{L}}\|^{2}=c,\quad\forall i,$$
423 where c is a constant.
In this case, the optimization of −PV
i=1 pi∥∇Ei,tL∥2 becomes independent of pi 424 . To satisfy the
normalization constraint PV
i=1 425 pi = 1, the optimal solution is:
$$p_{i}={\frac{1}{V}},\quad\forall i.$$
$$(10)$$

426 This corresponds to a uniform distribution, where all embeddings are treated equally (see Figure 427 8). While computationally efficient, this approach may lead to suboptimal convergence if some 428 embeddings contribute disproportionately to the loss reduction.

## 429 **Gradient Norm Bounded By** Li

430 Now, let us assume that the gradient norm for each embedding is bounded,
∥∇Ei,t*L∥ ≤* Li, ∀i, (11)
where Li 431 is a known upper bound for embedding i. Using this bound, we approximate,

$$-\sum_{i=1}^{V}p_{i}\|\nabla_{{\bf E}_{i,t}}{\cal L}\|^{2}\geq-\sum_{i=1}^{V}p_{i}L_{i}^{2}.\tag{1}$$
$$(11)$$
$$(12)$$

$$(13)$$

To maximize PV
i=1 piL
2 isubject to the constraint PV
i=1 432 pi = 1, we note that the objective function 433 is linear in p. Therefore, the maximum is attained at a vertex of the probability simplex, meaning the 434 optimal solution is:
pk = 1, where k = arg max

$$\begin{array}{r l r l}{{k=\arg\operatorname*{max}_{i}L_{i}^{2},}}&{{\mathrm{~and~}}}&{{p_{i}=0,}}&{{\forall i\neq k.}}\end{array}$$
$$p_{k}=1,\quad\mathbf{v}$$

435 This result indicates that the optimal probability distribution assigns all weight to the embedding with 436 the highest gradient bound, ignoring all others. Therefore, to obtain a smooth probability distribution, 437 we introduce an entropy regularization term as follow,

$$H(\mathbf{p})=-\sum_{i=1}^{V}p_{i}\log p_{i}.$$

pilog pi. (14)
438 We now optimize the modified objective,

$\sum_{i=1}^{V}p_{i}L_{i}^{2}+\gamma H({\bf p})$, (10.1)
subject to the constraint PV
i=1 439 pi = 1, where γ > 0 controls the strength of the regularization.

440 The corresponding Lagrangian is as follow,

$${\mathcal{L}}_{p}=\sum_{i=1}^{V}p_{i}L_{i}^{2}+\gamma\left(-\sum_{i=1}^{V}p_{i}\log p_{i}\right)+\mu\left(\sum_{i=1}^{V}p_{i}-1\right).$$
$$(14)$$

$$(15)$$
$$(16)$$
$$(17)^{\frac{1}{2}}$$

441 Taking the derivative with respect to pi and setting it to zero, we get,

$$L_{i}^{-}-\gamma(1+$$
L
2
i − γ(1 + log pi) + µ = 0. (17)
442 Solving for pi gives:

$$\quad\quad\log p_{i}={\frac{L_{i}^{2}+\mu-\gamma}{\gamma}}\quad\implies\quad p_{i}=\exp\left({\frac{L_{i}^{2}+\mu-\gamma}{\gamma}}\right).$$

Applying the constraint PV
i=1 443 pi = 1, would results in the following solution,

$$p_{i}^{*}=\frac{\exp\left(L_{i}^{2}/\gamma\right)}{\sum_{j=1}^{V}\exp\left(L_{j}^{2}/\gamma\right)}.\tag{1}$$

444 This result smoothly distributes probabilities based on the gradient bounds, assigning higher probability to embeddings with larger L
2 445 i while ensuring a non-degenerate distribution.

0 20 40 60 80 Token id 0.005 0.010 0.015 0.020 0.025 Train Data Test Data Train Data Test Data

$$(18)$$
$$(19)$$

Train Data Test Data 0 20 40 60 80 Token id 0.007 0.008 0.009 0.010 0.011 0 20 40 60 80 Token id 0.008 0.010 0.012 Probabi lity Probabi lity Probabi lity
(a) Random Sampling
$$(21)$$
$\eqref{eq:walpha}$. 
Figure 8: Token probabilities in the training and test sets under different sampling strategies. Imbalanced sampling leads to uneven token occurrences in mini-batches, causing some tokens to be absent in multiple updates while others appear frequently. This results in highly variable gradient updates, where frequently seen tokens converge faster, while rare tokens stagnate due to sparse updates, affecting overall model generalization.

## 446 **B Dynamics Of Updates In Bilinear Systems With Initialization Effects**

We analyze the interaction between embeddings E ∈ R
p×dand weight matrix W ∈ R
4d×d 447 in a 448 bilinear term:
z(EW), (20)
449 where z is an activation function applied elementwise. The gradients of E and W are given as:

The grad $\sigma$. 
450 The gradient norms are influenced by the dominant singular values of W and E. Specifically:

$$\nabla_{\mathbf{E}}\propto\mathbf{W}^{\top}\nabla_{\mathrm{loss}},\quad\nabla_{\mathbf{W}}\propto\mathbf{E}^{\top}\nabla_{\mathrm{loss}}$$

⊤∇loss. (21)
$$\|\nabla_{\mathbf{E}}\|\propto\sigma_{\mathrm{max}}(\mathbf{W}),\quad\|\nabla_{\mathbf{W}}\|\propto\sigma_{\mathrm{max}}(\mathbf{E}).$$
∥∇E∥ ∝ σmax(W), ∥∇W∥ ∝ σmax(E). (22)
For proportional updates (∥∆E*∥ ∼ ∥*∆W∥), the ratio c =
ηE
$\mathbf{W}\left[\right]$), the ratio $c=\frac{\mu_{B}}{\eta_{W}}$ . 
460 must satisfy:
$$c\propto{\frac{\sigma_{\operatorname*{max}}(\mathbf{E})}{\sigma_{\operatorname*{max}}(\mathbf{W})}}\cdot{\frac{f_{W}}{f_{E}}}.$$
. (24)
$$(23)$$
$$(24)$$
451 At initialization, E and W are often drawn from distributions with variances that depend on their dimensions (e.g., PyTorch initializes weights with N (0,p 452 2/d) scaling). This initialization typically 453 ensures σmax(E) ≫ σmax(W), as W is higher-dimensional, amplifying the difference in gradient 454 magnitudes. 455 The embedding matrix E is updated less frequently than W because not all tokens appear in every 456 batch. Let fE and fW represent the update frequencies of E and W, respectively. Typically, 457 fW > fE, exacerbating the update disparity. 458 To balance the effective updates of E and W, the learning rates ηE and ηW must be scaled to account 459 for both their singular values and update frequencies. The effective update ratio is:

. (23)
$${\frac{\|\Delta\mathbf{E}\|}{\|\Delta\mathbf{W}\|}}\propto{\frac{\eta_{E}\cdot\sigma_{\operatorname*{max}}(\mathbf{W})\cdot f_{E}}{\eta_{W}\cdot\sigma_{\operatorname*{max}}(\mathbf{E})\cdot f_{W}}}.$$
The term σmax(E)
σmax(W)
461 reflects the imbalance in singular values due to initialization and structural properties. The term fW
fE
462 accounts for the frequency imbalance in updates between E and W, driven 463 by sparse token appearances in batches.

PyTorch initialization, which scales weights by O(p 464 2/d), ensures that σmax(W) and σmax(E) are 465 initially proportional to the dimensions d. This contributes to the observed imbalance in their singular 466 values at the start of training.

## 467 **C More Experiments** 468 **C.1 Analysis Of Singular Values Of Embedding Layer**

469 Previous studies (e.g., [25], [10], [18], [26]) have explored the gap between SGD and Adam in 470 optimizing Transformer models, but the specific role of embeddings and their bilinearity with down471 stream weights remains underexplored. For example, [25] attributes SGD's suboptimal performance 472 to the heavy-tailed distribution of stochastic gradient noise. This observation aligns with our findings 473 regarding the randomness in embedding updates for low-p tokens. 474 On the other hand, [10] argues that gradient noise alone cannot explain Adam's superiority. Their 475 experiments demonstrate that, even with full-batch training to eliminate stochastic noise, SGD 476 underperforms compared to Adam. They suggest that the sign of the gradient might be a more reliable 477 descent direction than its magnitude, and since Adam optimally balances both, it outperforms SGD, 478 particularly in small-batch settings.

479 Furthermore, [26] provides a novel explanation for Adam's advantage over SGD in Transformers 480 by analyzing the blockwise Hessian spectrum, introducing the concept of "block heterogeneity." 481 This refers to significant variations in the Hessian spectra across parameter blocks, a phenomenon 482 observed in Transformers but not in CNNs. However, the underlying source of this heterogeneity 483 is not explicitly discussed. We hypothesize that this stems from the bilinear nature of weights, 484 particularly in the embedding and attention mechanisms. To support this hypothesis, we analyze the 485 Hessian of embedding weights compared to other weight below. 486 To analyze the curvature of the loss landscape, we compute the maximum eigenvalue of the Hessian 487 matrix using the power method with Hessian-vector products (HVPs). This approach avoids explicitly 488 constructing the Hessian, making it computationally efficient for large-scale systems.

489 The power method iteratively approximates the maximum eigenvalue of the Hessian H as follows: 490 1. Initialize a random vector v0 with the same dimensionality as the parameters [E,W]. 491 2. Compute the Hessian-vector product Hvk using automatic differentiation:
Hvk = ∇θ (∇θL · vk),
492 where θ = [E,W]. 493 3. Normalize the vector and update the eigenvalue estimate:

$$\mathbf{v}_{k+1}={\frac{\mathbf{H}\mathbf{v}_{k}}{\|\mathbf{H}\mathbf{v}_{k}\|}},\quad\sigma_{\operatorname*{max}}\approx\mathbf{v}_{k}^{\top}\mathbf{H}\mathbf{v}_{k}.$$

494 Figure 9 shows the maximum eigenvalues of the Hessian with respect to E and W during training. 495 The results highlight distinct curvature properties for E and W, reflecting their roles in the bilinear 496 interaction.

497 Extending these insights to attention mechanisms highlights further challenges in bilinear optimization 498 and demonstrates how adaptive learning rates (e.g., Adam) help escape saddle points. This suggests 499 a deeper connection between the bilinearity of weight interactions and the optimization challenges 500 unique to Transformer models.

## 501 **C.2 Rank Evolution And Implicit Regularization**

502 Recent work has shown that weight decay in bilinear models (e.g., Z = EW) implicitly regularizes 503 the nuclear norm of the product matrix, promoting low-rank solutions and improved generalization

0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 H

e s sia n Eig e n v alu e w.r.t. 

W

E
W Training 100%
Validation 100%
10 3 10 4 Optimizaton steps 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 H

e s sia n Eig e n v alu e w.r.t. 

W

E
W Training 100%
Validation 100%
0.0 0.2 0.4 0.6 0.8 1.0 H

e s sia n Eig e n v alu e w.r.t. 

t o E

10 3 10 4 Optimizaton steps 0.00 0.02 0.04 0.06 0.08 0.10 0.12 H

e s sia n Eig e n v alu e w.r.t. 

t o E

504 [8]. This complements our focus on embedding dynamics, as both highlight the impact of bilinear 505 coupling on optimization. 506 To explore this in our setup, we track the rank evolution of E, W, and the product EW. As shown in 507 Figure 10, W exhibits three distinct phases: an early drop during training loss reduction, a plateau, 508 and a final decline aligned with grokking. In contrast, E's rank remains largely stable throughout.

509 Figure 10 compares three optimization setups: Adam (with weight decay 0.001), Adam-LR (our 510 proposed variant with a learning rate ratio), and Adam with stronger weight decay (0.005). All 511 configurations lead to a reduction in rank(EW), consistent with implicit nuclear norm regularization. 512 However, only Adam-LR shows continued rank changes after generalization, suggesting that rank 513 evolution alone does not capture the onset of grokking.

514 These findings reinforce that implicit regularization in bilinear systems depends not just on decay 515 strength, but also on the interplay between initialization, update frequency, and curvature.

Addition Modular P
Division Modular P
Addition Modular P with Adam_LR
0 20 40 60 80 100 Input Dimension (Vocab Index)
0 10 20 30 40 50 60 70 80 0 20 40 60 80 100 Input Dimension (Vocab Index)
0 10 20 30 40 50 60 70 80 0 20 40 60 80 100 Input Dimension (Vocab Index)
0 5 10 15 20 25 30 L2 N
orm L2 N
orm L2 N
orm

## 516 **D Fourier Analysis Of Embedding Representations** 531 **D.1 Additional Datasets And Learning Rate Sensitivity** 543 **Compute Resources**

544 All experiments were conducted using an NVIDIA A6000 GPU. Training runs were performed 545 using PyTorch, with each configuration fitting comfortably within the GPU's 48 GB memory. No 546 distributed training or multi-GPU setups were used. 526 The results for different tasks are shown in Figure 11. Clear frequency peaks indicate that the model 527 internally captures task-specific periodic structure. Notably, such structure emerges even without 528 explicit Fourier features, especially for modular addition and multiplication. However, in more 529 complex tasks, such as modular division, this frequency localization diminishes—suggesting the 530 limits of periodic encoding and the growing need for learned representations. 532 In addition to modular addition and division, we evaluate our methods on two further tasks: modular multiplication (a ÷ b) mod p and sum of squares (a 2 + b 2 533 ) mod p. These tasks share the same 534 architecture and tokenization as described in Section 5. 535 We emphasize that our experimental design is not centered on hyperparameter optimization. While 536 aggressive tuning of learning rates and batch sizes can suppress or delay grokking, our goal is to 537 study it where it naturally occurs. To that end, we identify configurations where grokking persists 538 and focus our analysis there. This approach aligns with prior work on mechanistic understanding 539 of grokking [9, 14], which likewise prioritize clarity of dynamics over benchmark performance.

540 For illustration, Figures 13 and 14 show learning rate sensitivity on four datasets, confirming the 541 robustness of our findings across reasonable settings (skewed distribution of embedding update delay 542 the generalization). 517 Fourier features offer a structured way to encode modular arithmetic directly into the input space. By 518 encoding periodicity into the representation, such features can bypass the need for learned embeddings 519 and mitigate challenges like sparse updates for rare tokens. However, this approach requires prior 520 knowledge of the task's structure—e.g., periodicity—which may not apply in more complex tasks 521 such as modular division or nonlinear compositions. 522 To investigate whether embedding layers naturally learn such structure, we analyze their frequency 523 characteristics. Following the approach in [12], we apply the Discrete Fourier Transform (DFT)
524 along the input dimension of the embedding matrix and compute the ℓ2-norm across the embedding 525 dimension. We then plot the first P/2 components, leveraging the symmetry of the DFT.

Batch Size = 64 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accu racy Batch Size = 128 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc urac y Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc uracy Batch Size = 64 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accu racy Batch Size = 128 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc urac y Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc uracy
(b) (a 2 + b 2) mod p

## 547 **Neurips Paper Checklist**

555 - You should answer [Yes] , [No] , or [NA] . 556 - [NA] means either that the question is Not Applicable for that particular paper or the 557 relevant information is Not Available. 558 - Please provide a short (1–2 sentence) justification right after your answer (even for NA). 563 The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. 564 While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a 565 proper justification is given (e.g., "error bars are not reported because it would be too computationally 566 expensive" or "we were unable to find the license for the dataset we used"). In general, answering 567 "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we 548 The checklist is designed to encourage best practices for responsible machine learning research, 549 addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove 550 the checklist: **The papers not including the checklist will be desk rejected.** The checklist should 551 follow the references and follow the (optional) supplemental material. The checklist does NOT count 552 towards the page limit. 553 Please read the checklist guidelines carefully for information on how to answer these questions. For 554 each question in the checklist:
559 **The checklist answers are an integral part of your paper submission.** They are visible to the 560 reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it 561 (after eventual revisions) with the final version of your paper, and its final version will be published 562 with the paper.

Batch Size = 64 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accurac y Batch Size = 128 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accu rac y Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc ura cy Batch Size = 256 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 256 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 256 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accurac y Accurac y Accurac y Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accu rac y Accu rac y Accu rac y Batch Size = 1024 Batch Size = 1024 Batch Size = 1024 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Acc ura cy Acc ura cy Acc ura cy random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 256 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Ac cur acy Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accuracy Batch Size = 1024 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accurac y Batch Size = 256 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Ac cur acy Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accuracy Batch Size = 1024 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accurac y Batch Size = 64 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 256 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Ac cur acy Ac cur acy Batch Size = 128 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accuracy Accuracy Batch Size = 512 random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test Batch Size = 1024 10 1 10 2 10 3 10 4 10 5 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 10 1 10 2 10 3 10 4 Optimization steps 0.0 0.2 0.4 0.6 0.8 1.0 Accurac y Accurac y random - Train random - Test skew - Train skew - Test uniform - Train uniform - Test
568 acknowledge that the true answer is often more nuanced, so please just use your best judgment and 569 write a justification to elaborate. All supporting evidence can appear either in the main paper or the 570 supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification 571 please point to the section(s) where related material for the question can be found. 572 IMPORTANT, please: 573 - **Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist"**,
574 - **Keep the checklist subsection headings, questions/answers and guidelines below.**
575 - **Do not modify the questions and only use the provided macros for your answers**.

## 576 1. **Claims**

577 Question: Do the main claims made in the abstract and introduction accurately reflect the 578 paper's contributions and scope? 579 Answer: [Yes] 580 Justification: Yes, the main claim made in the abstract and introduction are reflected in the 581 paper Sections 3,4 and 5. 582 Guidelines: 583 - The answer NA means that the abstract and introduction do not include the claims 584 made in the paper. 585 - The abstract and/or introduction should clearly state the claims made, including the 586 contributions made in the paper and important assumptions and limitations. A No or 587 NA answer to this question will not be perceived well by the reviewers. 588 - The claims made should match theoretical and experimental results, and reflect how 589 much the results can be expected to generalize to other settings. 590 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 591 are not attained by the paper. 592 2. **Limitations** 593 Question: Does the paper discuss the limitations of the work performed by the authors? 594 Answer: [Yes] 595 Justification: The limitations are discussed in section 5.3. 596 Guidelines: 597 - The answer NA means that the paper has no limitation while the answer No means that 598 the paper has limitations, but those are not discussed in the paper. 599 - The authors are encouraged to create a separate "Limitations" section in their paper. 600 - The paper should point out any strong assumptions and how robust the results are to 601 violations of these assumptions (e.g., independence assumptions, noiseless settings, 602 model well-specification, asymptotic approximations only holding locally). The authors 603 should reflect on how these assumptions might be violated in practice and what the 604 implications would be. 605 - The authors should reflect on the scope of the claims made, e.g., if the approach was 606 only tested on a few datasets or with a few runs. In general, empirical results often 607 depend on implicit assumptions, which should be articulated. 608 - The authors should reflect on the factors that influence the performance of the approach. 609 For example, a facial recognition algorithm may perform poorly when image resolution 610 is low or images are taken in low lighting. Or a speech-to-text system might not be 611 used reliably to provide closed captions for online lectures because it fails to handle 612 technical jargon. 613 - The authors should discuss the computational efficiency of the proposed algorithms 614 and how they scale with dataset size. 615 - If applicable, the authors should discuss possible limitations of their approach to 616 address problems of privacy and fairness. 617 - While the authors might fear that complete honesty about limitations might be used by 618 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 619 limitations that aren't acknowledged in the paper. The authors should use their best 620 judgment and recognize that individual actions in favor of transparency play an impor621 tant role in developing norms that preserve the integrity of the community. Reviewers 622 will be specifically instructed to not penalize honesty concerning limitations. 623 3. **Theory assumptions and proofs** 624 Question: For each theoretical result, does the paper provide the full set of assumptions and 625 a complete (and correct) proof? 626 Answer: [Yes]
627 Justification: The two main theories are supported by assumptions and proofs in the main 628 body and the appendix. 629 Guidelines: 630 - The answer NA means that the paper does not include theoretical results. 631 - All the theorems, formulas, and proofs in the paper should be numbered and cross632 referenced. 633 - All assumptions should be clearly stated or referenced in the statement of any theorems. 634 - The proofs can either appear in the main paper or the supplemental material, but if 635 they appear in the supplemental material, the authors are encouraged to provide a short 636 proof sketch to provide intuition. 637 - Inversely, any informal proof provided in the core of the paper should be complemented 638 by formal proofs provided in appendix or supplemental material. 639 - Theorems and Lemmas that the proof relies upon should be properly referenced.

## 640 4. **Experimental Result Reproducibility**

641 Question: Does the paper fully disclose all the information needed to reproduce the main ex642 perimental results of the paper to the extent that it affects the main claims and/or conclusions 643 of the paper (regardless of whether the code and data are provided or not)? 644 Answer: [Yes]
645 Justification: The details of the experiments are detailed in section 5.

646 Guidelines: 647 - The answer NA means that the paper does not include experiments. 648 - If the paper includes experiments, a No answer to this question will not be perceived 649 well by the reviewers: Making the paper reproducible is important, regardless of 650 whether the code and data are provided or not. 651 - If the contribution is a dataset and/or model, the authors should describe the steps taken 652 to make their results reproducible or verifiable. 653 - Depending on the contribution, reproducibility can be accomplished in various ways. 654 For example, if the contribution is a novel architecture, describing the architecture fully 655 might suffice, or if the contribution is a specific model and empirical evaluation, it may 656 be necessary to either make it possible for others to replicate the model with the same 657 dataset, or provide access to the model. In general. releasing code and data is often 658 one good way to accomplish this, but reproducibility can also be provided via detailed 659 instructions for how to replicate the results, access to a hosted model (e.g., in the case 660 of a large language model), releasing of a model checkpoint, or other means that are 661 appropriate to the research performed. 662 - While NeurIPS does not require releasing code, the conference does require all submis663 sions to provide some reasonable avenue for reproducibility, which may depend on the 664 nature of the contribution. For example 665 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 666 to reproduce that algorithm. 667 (b) If the contribution is primarily a new model architecture, the paper should describe 668 the architecture clearly and fully. 669 (c) If the contribution is a new model (e.g., a large language model), then there should 670 either be a way to access this model for reproducing the results or a way to reproduce 671 the model (e.g., with an open-source dataset or instructions for how to construct 672 the dataset). 673 (d) We recognize that reproducibility may be tricky in some cases, in which case 674 authors are welcome to describe the particular way they provide for reproducibility. 675 In the case of closed-source models, it may be that access to the model is limited in 676 some way (e.g., to registered users), but it should be possible for other researchers 677 to have some path to reproducing or verifying the results. 678 5. **Open access to data and code** 679 Question: Does the paper provide open access to the data and code, with sufficient instruc680 tions to faithfully reproduce the main experimental results, as described in supplemental 681 material? 682 Answer: [Yes]