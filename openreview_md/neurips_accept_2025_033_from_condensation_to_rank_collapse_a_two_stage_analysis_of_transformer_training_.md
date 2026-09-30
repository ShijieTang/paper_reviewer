# From Condensation To Rank Collapse: A Two-Stage Analysis Of Transformer Training Dynamics

Zheng-An Chen1, Tao Luo1,2∗
1School of Mathematical Sciences, Shanghai Jiao Tong University 2Institute of Natural Sciences, MOE-LSC, CMA-Shanghai, Shanghai Jiao Tong University

## Abstract

Although transformer-based models have shown exceptional empirical performance, the fundamental principles governing their training dynamics are inadequately characterized beyond configuration-specific studies. Inspired by empirical evidence showing improved reasoning capabilities under small initialization scales in language models, we employ the gradient flow analytical framework established in Zhou et al. [2022] to systematically investigate linearized Transformer training dynamics. Our theoretical analysis dissects the dynamics of attention modules into two distinct stages. In the first stage, asymmetric weight perturbations from random initialization sustain non-degenerate gradient dynamics in parameter matrices, facilitating systematic escape from small initialization regimes. Subsequently, these matrices undergo condensation, progressively aligning toward the target orientation. In the second stage, the previously static key-query matrices actively participate in training, driving the normalized matrices toward asymptotic rank collapse. This two-stage framework generalizes classical directional convergence results.

## 1 Introduction

The transformer-based models Vaswani et al. [2017] have achieved remarkable breakthroughs in various fields, with the successful application of large language models. However, the theoretical analysis of the transformer still remains in specific tasks, such as in-context learning settings Brown et al. [2020], Olsson et al. [2022], Bietti et al. [2023] or single attention block with reparameterization Tian et al. [2023]. The use of linear regression tasks Zhang et al. [2024a] and Markov chain tasks Ildiz et al. [2024] has provided highly interpretable theoretical analyses, but a crucial question still remains: Can we analyze the characteristics of the training dynamics of transformers independently of specific tasks? Meanwhile, small initialization has been increasingly shown to hold promise in the training process of large models, especially for reasoning tasks. Numerous studies Zhang et al. [2024b, 2025b], Yao et al. [2025] suggest that the implicit regularization effect of small initialization is still effective in large language models. This effectiveness is particularly significant in the context of modern large models, which are characterized by extreme overparameterization. In these regimes, where explicit regularization techniques like weight decay or dropout may prove insufficient on their own, implicit regularization becomes pivotal. It operates by imposing intrinsic constraints on the training dynamics and the resulting parameter space, effectively guiding the model towards solutions with good generalization properties despite the vast hypothesis space. This implicit bias is key to understanding how models with such immense capacity manage to avoid severe overfitting and achieve remarkable performance on unseen data.

∗Corresponding author: luotao41@sjtu.edu.cn.

Motivated by these observations, we propose to investigate the training dynamics of transformers under a small initialization setting. Leveraging the gradient flow theme similarly to Zhou et al. [2022], We delineate different training dynamics for outer parameters versus attention parameters WQ and WK in Transformers.

We dissect the dynamics of attention modules into two distinct stages. In the first stage, the core attention mechanism, softmax(QK⊺), remains nearly stagnant, as asymmetric weight perturbations from random initialization drive non-degenerate gradient dynamics in parameter matrices, particularly WV , facilitating escape from small initialization regimes. During this escape, the parameter matrix converges row-wise toward the target orientation, a process we term condensation. We theoretically prove that condensation is guaranteed under small initialization, and experimentally observe that it stabilizes without significant fluctuations.

In the second stage, after the outer parameters, such as WV , reach a quasi-steady state, the previously static key-query matrices, WQ and WK, begin to actively participate in training, driving their collapse. This two-stage framework not only elucidates the training dynamics but also generalizes classical directional convergence results, offering a robust theoretical foundation for Transformer optimization. To sum up, our contribution can be summarized as follows. 1. **Blow-up Dynamics**: We prove the blow-up property (Theorem 1) holds for measure-theoretically generic initializations, eliminating reliance on dichotomy assumptions while ensuring model nondegeneracy.

2. **Condensation Mechanism**: By introducing a condensation condition (Assumption 1), we establish theoretical guarantees for condensation emergence (Theorem 2).

3. **Key-Query Collapse**: After outer parameters stabilize in a quasi-steady state (Assumption 2), the key-query matrices begin active training, leading to asymptotic rank collapse of the normalized key-query matrices (Theorem 3).

4. **Experimental evidence**: We validate our hypotheses and theoretical predictions on both synthetic and real datasets with one and multi-layer Transformers, consistently observing two-stage dynamics marked by condensation and an eventual rank collapse of the normalized key-query matrices (Figure 1, 2, 3).

## 2 Related Works

Training dynamics of transformer. Given the scale of modern models and the complexity of optimizers, studying the training dynamics of Transformers is a challenging problem. Prior works have primarily investigated the optimization dynamics of a single attention layer Lu et al. [2021], Li et al. [2023], Snell et al. [2021]. However, these studies mainly focused on specific tasks, such as topic structure prediction and translation. Recently, the dynamics of in-context learning (ICL) has emerged as a prominent research area within Transformer dynamics, particularly given ICL's ability to solve novel tasks without parameter updates. Many works Mahankali et al. [2024], Zhang et al. [2024a], Huang et al. [2023], Collins et al. [2024] have focused on the linear regression setup to theoretically investigate the mechanism of ICL in single-layer Transformers, a line of work that has also informed algorithmic development Akyürek et al. [2023], Bai et al. [2023], Guo et al. [2024]. Another line of research investigates how specific structures within attention emerge during training, notably starting with studies on induction heads Olsson et al. [2022], Reddy [2024], Edelman et al. [2024], Zhang et al. [2025a], memory recall mechanisms Bietti et al. [2023], Cabannes et al. [2024], and even causal structure Nichani et al. [2024].

Despite the sophisticated structure of realistic Transformers, Tian et al. [2024] proposed a novel mathematical framework for analyzing the joint dynamics of MLP and attention blocks and successfully explained the sparsity of attention score matrices. Meanwhile, Chen et al. [2024a] provides a rigorous proof for the convergence of the ICL linear regression task using gradient flow with sufficiently small initialization. Small initialization and its applications The initialization of a neural network significantly affects its learning outcomes Arora et al. [2019b], Williams et al. [2019], Mei et al. [2018], Jacot et al. [2018], Rotskoff and Vanden-Eijnden [2018], Zhang et al. [2020]. Small initialization is a common setting investigated in the study of neural network optimization dynamics, which is different with the Neural Tangent Kernel (NTK) perspective in infinitely wide networks. For linear model, Ji and Telgarsky [2019] establish matrix alignment results theoretically. For nonlinear model, Zhou et al. [2022] found that small initialization can similarly promote parameter condensation, thereby reducing model complexity. Theoretically, Luo et al. [2021], Chen et al. [2024b], Zhou et al. [2023], Kumar and Haupt [2024] have deepened the understanding of this phenomenon. The recent survey article Xu et al. [2025] systematically synthesizes empirical and theoretical findings. The implicit bias induced by small initialization has also been discussed in the context of linear regression Saxe et al. [2013], Min et al. [2021], Varre et al. [2023] and matrix factorization tasks Li et al. [2018], Arora et al. [2019a], Stöger and Soltanolkotabi [2021], Soltanolkotabi et al. [2023], Bai et al. [2024]. More recently, many researchers have adopted small initialization settings to simplify the analysis of training dynamics in more complex models. From a theoretical perspective, Zhang et al. [2025a] applied small initialization to ICL tasks to analyze the behavior of linear attention. Yao et al. [2025] considered the training dynamics of the embedding space under small initialization using a synthetic dataset designed for reasoning and memorization. From an applied perspective, Zhang et al. [2019], Huang et al. [2020], Zhu et al. [2021] highlighted the importance of initialization in Transformers, while Bachlechner et al. [2021] combined zero-initialization with residual blocks in Transformers. Some research Zhang et al. [2024b], Yao et al. [2025] shows that small initialization helps Transformers learn the reasoning aspects of data rather than just memorization, a principle already applied in realistic LLM training Yin et al. [2025].

## 3 Preliminaries 3.1 Basic Notations

First, we introduce some notations that will be used in the rest of this paper. Let n and dm be the number of samples and the width of hidden layers, respectively. Let [n] denote the set of integers from 1 to n. Denote vector L
2 norm as *∥· ∥*2 and matrix Frobenius norm as *∥· ∥*F. Let ⟨·, ·⟩ represent standard inner product between two vectors. For a vector v, denote its k-th entry as vk. For a matrix A, denote the element in the k-th row and k′-th column as Akk′ . And denote k-th row as Ak and k′-th column as Ak′. Unless otherwise specified, summation 'P' is performed over the network width.

## 3.2 Classification Task

Binary classification: For decision tasks, the network produces a scalar output fθ(X) ∈ R. The predicted class assignment is determined by the sign of the output. The dataset is denoted by D = {(Xi, yi)}
n i=1 where Xi ∈ R
s×dm stands for input sequence in which s represents the sequence length and dm represents the hidden dimension, and yi *∈ {±*1} stands for label. For a loss function ℓ : R → R+, we define the empirical risk as L(θ) = 
1 n Pn i=1 ℓ (yifθ(Xi)).

Multi-class Classification: For probabilistic tasks, the network outputs logit vectors fθ(X) ∈ R
dv that parameterize a categorical distribution via the softmax transformation P(y = i|X; θ) =
exp(fθ(X)i)
∑dv j=1 exp(fθ(X)j )
where dv denotes the vocabulary size. For cross-entropy loss, we define the empirical risk as L(θ) = −
1 n Pn i=1 log P(y = yi|Xi; θ).

## 3.3 Condensation And Rank Collapse

We formalize the two geometric phenomena that will recur throughout our analysis.

Definition 1 (Condensation). Let W(t) be a matrix with rows Wk(t) (or columns Wk(t)*). We say* W condenses to a direction v *if, as* t → T,
D Wk(t)
∥Wk(t)∥2
, v E→ ±1 for every index k *with* ∥Wk(t)∥2 ̸= 0
(equivalently, the same holds columnwise).

Condensation is a directional notion and implies rank-1 collapse when a unique direction emerges. Rank collapse is a spectral notion and allows k > 1 when multiple top singular directions are tied.

Definition 2 (Asymptotic rank collapse). Let W(t) be a matrix. We say W *exhibits* rank-k collapse if the limit
$$W_{\infty}\;:=\;\operatorname*{lim}_{t\to T}{\frac{W(t)}{\|W(t)\|_{\mathrm{F}}}}$$

## Exists And Rank(W∞) ≤ K. 4 Theoretical Results 4.1 Problem Formulation

To analyze condensation phenomenon in transformers, we begin by formulating the problem. Specifically, we consider the following one-layer transformer model:
Definition 3 (One-layer transformer). Let X ∈ R
s×dm be an input sequence of length s *with model* dimension dm*. The Transformer function* fθ : R
s×dm → R
sis defined by the composition of attention and feed-forward operations:

$$f_{\theta}(X):=\mathrm{FFN}(\mathrm{Attn}(X))=\sigma\left(\mathrm{Attn}(X)W^{[1]}\right)W^{[2]}.$$

The attention sublayer Attn : R
s×dm → R
s×dm *is computed as:*

$$\operatorname{Attn}(\mathbf{X})=\operatorname{softmax}\left({\frac{\mathbf{X}\mathbf{W}_{Q}\mathbf{W}_{K}^{\mathsf{T}}\mathbf{X}^{\mathsf{T}}}{\sqrt{d_{m}}}}\right)\mathbf{X}\mathbf{W}_{V},$$

where parameter matrices satisfy WQ,WK,WV ,W[1] ∈ R
dm×dm and W[2] ∈ R
dm. The activation function σ : R → R *is tanh.* We use one-layer transformer fθ to solve binary classification tasks and take the last dimension of the output fθ(Xi)s as the output. So the empirical risk to be minimized is given by

$${\mathcal{L}}(\mathbf{\theta})={\frac{1}{n}}\sum_{i=1}^{n}\ell(y_{i}f_{\mathbf{\theta}}(\mathbf{X}_{i})_{s}).$$
$$(1)$$
$$\mathbf{(2)}$$
$$({\mathcal{I}})$$

For simplicity of presentation, we employ the exponential loss function ℓ(q) = e−q, which is commonly used in the analysis of classification tasks Lyu and Li [2020]. The analysis can be readily extended to other loss functions such as the logistic loss. The model parameters are initialized with Gaussian distributions scaled by a small perturbation parameter ε:

$$\begin{array}{l l l}{{W_{k}^{[2]}\sim{\mathcal{N}}(0,\varepsilon^{2}),}}&{{W_{k k^{\prime}}^{[1]}\sim{\mathcal{N}}(0,\varepsilon^{2}),}}&{{W_{Q,k k^{\prime}},W_{K,k k^{\prime}},W_{V,k k^{\prime}}\sim{\mathcal{N}}(0,\varepsilon^{2}),}}\end{array}$$

where ε ≪ 1 controls initialization magnitude. To analyze training dynamics, we adopt the gradient flow (GF) framework—the continuous-time limit of gradient descent. Given the small initialization scale, we derive effective dynamics through a perturbative expansion of the empirical risk L(θ) in powers of ε. First, we normalize parameters by absorbing the initialization scale:
W¯ [2] = ε
−1W[2], W¯ [1] = ε
−1W[1], W¯ Q = ε
−1WQ, W¯ K = ε
−1WK, W¯V = ε
−1WV .

Performing a Taylor expansion of L(θ) about ε = 0 yields the leading-order asymptotic form:

$$\mathcal{L}(\mathbf{\theta})=\frac{1}{2n}\sum_{i=1}^{n}\left[1-\varepsilon^{3}\left(\sum_{j=1}^{s}\frac{1}{s}y_{i}\mathbf{X}_{i,j}\mathbf{W}_{V}\mathbf{W}^{[1]}\mathbf{W}^{[2]}\right)+o(\varepsilon^{3})\right].$$  This expansion induces simplified gradient dynamics characterized by the following proposition:
$$(S)$$

Proposition 1 (Effective training dynamics). Given a binary dataset {(Xi, yi)}
n i=1, we define condensation direction v and rescaled time coordinate t¯*as follows:*

$$\mathbf{v}:={\frac{\sum_{i=1}^{n}y_{i}\left(\sum_{j=1}^{s}\mathbf{X}_{i,j}\right)}{\left\|\sum_{i=1}^{n}y_{i}\left(\sum_{j=1}^{s}\mathbf{X}_{i,j}\right)\right\|_{2}}},\quad{\bar{t}}:={\frac{\varepsilon}{n s}}\left\|\sum_{i=1}^{n}y_{i}\left(\sum_{j=1}^{s}\mathbf{X}_{i,j}\right)\right\|_{2}t.$$
$$(6)$$
t. (6)
Then, normalized parameters θ¯ *follow leading-order dynamics after rescaling:*

$${\frac{\mathrm{d}\bar{\theta}}{\mathrm{d}\bar{t}}}=\nabla_{\bar{\theta}}\left(\mathbf{v}\bar{W}_{V}\bar{W}^{[1]}\bar{W}^{[2]}\right).$$
$$(7)$$
$$E:=W_{v}W^{[1]}W^{[2]}.$$
$$(8)$$
vW¯V W¯ [1]W¯ [2]. (7)
Proposition 1 reveals a hierarchical learning mechanism: During initial training phases, the fullyconnected layers W[1],W[2] and value projection matrix WV exhibit substantial updates, while the query/key matrices WQ and WK in the self-attention module remain quasi-static. For subsequent analysis, we define the projection of WV onto v as Wv := vWV (omitting bar notation for simplicity) and introduce the energy functional:
E := WvW[1]W[2]. (8)
The effective dynamics can thus be interpreted as gradient ascent on this energy landscape.

## 4.2 Blow Up Dynamics

We first elucidate why Transformers with small initialization can successfully train and eventually escape the small initialization regime. This phenomenon emerges from the interplay between two fundamental mechanisms: 1. **Effective dynamics driving:** The parameter evolution governed by the effective dynamics exhibits remarkable symmetry, manifested through strict conservation laws that preserve key quantities during training.

2. **Random normal initialization:** While degenerate cases theoretically exist under Gaussian initialization, they occur with vanishing probability (measure zero in parameter space). Consequently, the dynamics almost surely demonstrate non-degenerate characteristics, ensuring stable training trajectories.

To preserve dynamical symmetry, we invoke the following proposition following the approach established in prior works Ji and Telgarsky [2019]: Proposition 2 (Conservation laws). *Under the gradient flow dynamics prescribed by system Eq. (7),* the following system of conservation laws emerges:

$$\frac{\mathrm{d}}{\mathrm{d}t}\left(\mathbf{W}_{\mathbf{v},k}^{2}-\sum_{k^{\prime}}\left(\mathbf{W}_{kk^{\prime}}^{[1]}\right)^{2}\right)=0\quad\text{and}\quad\frac{\mathrm{d}}{\mathrm{d}t}\left(\left(\mathbf{W}_{k}^{[2]}\right)^{2}-\sum_{k^{\prime}}\left(\mathbf{W}_{k^{\prime}k}^{[1]}\right)^{2}\right)=0.\tag{9}$$

We now analyze the non-symmetric property arising from Gaussian random initialization, with particular focus on the degeneracy mechanism. Crucially, we establish that degeneracy exclusively occurs when initialization violates the following non-degenerate initialization:
Definition 4 (Non-degenerate initialization). Let θ =WV ,W[1],W[2]denote parameters initialized from a Gaussian distribution. The initialization is called non-degenerate if ∥Wv∥
22 ̸= ∥W[2]∥
22
and
$\|\hat{\mathbf{W}}_{\mathbf{v}}\|_{2}^{2}-\|\hat{\mathbf{W}}^{[2]}\|_{2}^{2}+\min\bigl{\{}\|\mathbf{W}_{\mathbf{v}}\|_{2}^{2},\ \|\mathbf{W}^{[2]}\|_{2}^{2}\bigr{\}}\bigl{(}\|\mathbf{W}_{\mathbf{v}}\|_{2}^{2}-\|\mathbf{W}^{[2]}\|_{2}^{2}\bigr{)}\neq0.$
2̸= 0. (10)
Having clarified the definition of non-degenerate initialization, we present the following theorem that reveals the non-degeneracy property of effective training dynamics. Theorem 1 (Blow-up in finite time). *Let the parameters be initialized randomly as above from* a Gaussian distribution. Then, almost surely, the initialization is non-degenerate in the sense of Definition *4, and the effective training dynamics Eq. (7) blows up in finite time. That is, there exists* T∗ > 0 *such that*

$$\operatorname*{lim}_{t\to T^{*}}E(t)=+\infty.$$

Proof sketch. We prove finitetime blow-up via a Riccati-type differential inequality for the energy E(t). Full technical details are provided in Appendix A.1.

(1) Superlinear growth. A direct computation gives E˙(t) ≥ 3E(t)

$\zeta(t)\geq3E(t)^{4/3}$, hence $\partial_t E(t)^{-1/3}\leq-1$ and   1. 
$$E(t)\;\geq\;\frac{1}{\left(E(0)^{-1/3}-t\right)^{3}}.$$
. (11)
$$\mathbf{1}(0)$$
$$(11)$$

For E(0) > 0 this yields T∗ ≤ E(0)−1/3.

(2) Negative initial energy. If E(0) ≤ 0, then

$$E(t)\ \geq\ -\frac{1}{\left((-E(0))^{-1/3}+t\right)^{3}},$$
$$(12)^{\frac{1}{2}}$$
, (12)
so E is increasing and cannot remain negative indefinitely. Assuming E(t) ≤ 0 for all t leads to contradictions with (i) standard continuation at finite T∗, or (ii) monotone limits at T∗ = ∞, reducing to the borderline case E(t) ↑ 0.

(3) Borderline exclusion. In the regime T∗ = ∞ and E(t) ↑ 0, structural identities and conservation give ∥Wv(t)∥
22 ∥W[2](t)∥
22 → 0 and E˙(t) → 0. Under the non-degenerate initialization (Def. 4),
this forces a contradiction, since the limiting E˙ must be strictly positive.

## 4.3 Condensation Dynamics

We have proved that energy and parameters norm will blow up almost surely. It implies the effective dynamics drive parameters escape small initialization area in finite time. The next question is how the effective dynamics affects the emergence of condensation and whether there exist observables to help us characterize condensation. We propose a condition of condensation and verify its effectiveness using experimental and theoretical methods. In particular, we theoretically prove that the solution of the effective dynamics has specific properties, which is to some extent a sufficiency argument. The necessity argument is quite difficult in theory. But experimental results provide us a strong implication that this condition maybe also necessary. Assumption 1 (Condensation condition). The parameters satisfy the **condensation condition** at time t*. That is* 1. *For each index* i ∈ [dm] , W[2]
i WvW[1],i > 0 and Wv,iW[1]
i W[2] > 0.

2. For each pair *i, j* ∈ [dm], ⟨W[2]
i W[1],i,W[2]
j W[1],j ⟩ > 0*, and* ⟨Wv,iW[1]
i,Wv,jW[1]
j⟩ > 0.

This hypothesis can be verified experimentally in Sec. 5.1.2. Then, based on Assumption 1, we formalize the statement of the culminating theorem as follows: Theorem 2 (Condensation). Under Assumption 1, the effective dynamical system governed by Eq.

(7) drives the parameter matrix WV *to undergo condensation in the sense of Definition* 1.

This section gives a highlevel proof sketch; full details appear in Appendix A.2. Proof sketch. We establish finitetime directional convergence (condensation) via geometric propagation and twosided energy control.

(1) Geometric consistency and alignment dynamics. Under Assumption 1, Proposition 4 shows that once the alignment condition holds at some t0 < T ∗, it propagates throughout (t0, T ∗). Proposition 5 yields a structural dichotomy of the columns of WV into a condensing class C1 and a uniformly bounded class C2. Propositions 6 and 7 further establish dynamical alignment between W[2] and its time derivative, ensuring coherence of the evolving direction.

(2) Singularity structure and condensation. Proposition 8 supplies an energy upper bound which, combined with the lower bound in Eq. (11), furnishes a bilateral estimate on E(t). A telescopingintegral argument then proves that the condensing indices dominate in finite time, completing the proof of condensation via Theorem 2.

## 4.4 Key-Query Dynamics

Following the initial training stage, the parameter matrices WV , W[1] and W[2] exhibit substantial growth in magnitude, effectively escaping the small-initialization regime. In contrast, the key-query matrices WQ and WK demonstrate remarkable stability in scale. This separation phenomenon is fundamentally governed by the effective dynamics of the learning system. A pivotal question arises: Under what conditions do the key-query matrices become dynamically activated, thereby enabling the attention mechanism to exert its structural influence? We hypothesize that during early training, WV , W[1] and W[2] converge to a critical point where WQ and WK
almost vanish, temporarily stabilizing in this dormant state. The following analysis provides mechanistic insights into this dynamical freezing phenomenon. The final activation function is omitted from our analysis. This is justified because the layer's pre-activations consistently operate within the linear regime of the function. Furthermore, empirical results confirm that its inclusion does not alter the model's learning dynamics (refer to B.2). Now empirical loss L(θ) has the following decomposition:

$${\cal L}(\mathbf{\theta})\approx\frac{1}{n}\sum_{i=1}^{n}{\cal L}_{1,i}(\mathbf{\theta}){\cal L}_{2,i}(\mathbf{\theta}),\tag{13}$$

where

L1,i(θ) =exp n−yi Psj=1 1 sXi,jWV W[1]W[2]o, L2,i(θ) =exp n−yi Psj=1 1s Xi,sWQW⊺KX⊺ √ i,j dm− 1 s 2Ps l=1 Xi,sWQW⊺KX⊺ √ i,l dm Xi,jWV W[1]W[2]o.
Based on the above discussion, we now formalize the following assumption. Assumption 2 (Dynamics separation stage). After the breakdown of the effective dynamics in Eq. (7), let δ *denote a small parameter. The gradient flow subsequently enters a stage characterized by:*
1. **Criticality conditions:** *The outer parameters* (WV ,W[1],W[2]) converge to a quasi-stationary configuration such that ∇WV Le1 = ∇W[1]Le1 = ∇W[2]Le1 = O(δ 2)*, where* Le1 =
1 n Pi L1,i.

2. **Key-query stunting:** The attention parameters remain small, satisfying |WQij |, |WKij | = O(δ),
until their norms ∥WQ∥ and ∥WK∥ *exceed a critical scale.*
To facilitate empirical validation, we introduce a modified version of the basic equivalence of the first part of Assumption 2, denoted as Assumption 2*.

Assumption 2*. *The outer parameters* (WV ,W[1],W[2]) *reach a quasi-stationary state whose* directions vary negligibly over time, i.e., d dt WV
∥WV ∥
=d dt W[1]
∥W[1]∥
=d dt W[2]
∥W[2]∥
≈ 0, and the loss evolution satisfies dL1,i dt = O(δ 2) *for all* i.

Since we assume the small parameter δ is still relatively small, we get the following Proposition which illustrate evident dynamics separation and the leading order dynamics of key-query matrices. Proposition 3 (Effective dynamics during dynamics separation stage). Under Assumption 2 or Assumption 2*, the empirical risk L(θ) exhibits the following properties:
1. **Dynamics separation** *The gradients of the empirical risk with respect to* WV , W[1] and W[2]
are of order O(δ 2), while the gradients with respect to the query matrix WQ and key WK *are of* order O(δ).

2. **Key-query dynamics** *Treating* WV , W[1] and W[2] as fixed due to dynamics separation, the
leading-order dynamics of key-query matrices are given by
$${\frac{\mathrm{d}W_{Q}}{\mathrm{d}t}}=F W_{K},\quad{\frac{\mathrm{d}W_{K}}{\mathrm{d}t}}=F^{\intercal}W_{Q},$$
⊺WQ, (14)
where F *is defined as follows*

$$\mathbf{F}=\frac{1}{ns\sqrt{d_{m}}}\sum_{i=1}^{n}y_{i}\mathcal{L}_{1,i}\mathbf{X}_{i,s}^{\mathsf{T}}\mathbf{W}^{[2]\mathsf{T}}\mathbf{W}^{[1]\mathsf{T}}\mathbf{W}_{V}{}^{\mathsf{T}}\left(\sum_{j=1}^{s}\mathbf{X}_{i,j}^{\mathsf{T}}\left(\mathbf{X}_{i,j}-\frac{1}{s}\sum_{l=1}^{s}\mathbf{X}_{i,l}\right)\right).\tag{15}$$
$$(14)$$

Since the dynamics governing WQ and WK form a linear ordinary differential equation system in this context, we can rigorously establish the subsequent conclusions. Theorem 3 (Asymptotic rank collapse). Given the key-query dynamics governed by Eq. (14), the normalized key and query matrices exhibit rank collapse as Definition 2. Specifically, when F possesses a unique largest singular value, both normalized matrices asymptotically become rank 1. The detailed proofs of Proposition 3 and Theorem 3 can be found in the Appendix A.3.

## 5 Experimental Results

In this section, we first demonstrate the phenomena of cohesion and rank collapse using synthetic data and confirm the assumptions required for our theoretical analysis of the one-layer Transformer model. We then present experiments on natural language processing tasks to demonstrate the generality of our theoretical findings with respect to various datasets and network architectures.

## 5.1 Synthetic Dataset

We employ the concept of the anchor function Zhang et al. [2024c] to construct a synthetic dataset that simulates a simplified language modeling scenario. The model is a one-layer Transformer with tanh activation, trained using cross-entropy loss and the AdamW optimizer. Further experimental settings are detailed in Appendix B.1.

## 5.1.1 Phenomenon: Condensation And Rank Collapse

To dissect the learning dynamics, we visualize the training process through three complementary lenses: the cosine similarity of parameters (Calculation method refer to Sec. B.1), the relative change of norms, and the effective rank of weight matrices. As shown in Figure 1, these analyses collectively reveal a distinct three-stage training trajectory, which we characterize as Condensation, Key-Query Rank Collapse, and the further training. The training process begins with a rapid decrease in loss, driven almost exclusively by the outerlayer parameters since the relative change of the outer parameters far exceed those of the attention parameters (Fig.1(b)) during this initial phase. This intense optimization leads to the condensation phenomenon, where the initially random outer parameters organize into a low-rank configuration. This is visually evident from the emergence of block structures in their cosine similarity matrices (Fig.1(a)) and is quantified by a monotonic and significant decrease in their effective rank (Fig. 1(c)). Throughout this stage, the attention parameters remain largely static and unstructured. Following the initial phase, the training loss enters a prolonged plateau. This signals a critical transition in the learning dynamics, marked by the gray dashed line in Fig.1(b). At this stage, a clear dynamics separation occurs: the updates to the outer parameters subside, and the attention parameters become the primary focus of optimization. This empirical observation validates our theoretical framework, particularly Proposition 3. As the changes in the outer parameters become slower (supporting Assumption 2), the attention parameters begin to learn their specialized roles. This is characterized by a rank collapse, confirmed visually by the sudden formation of structure in their similarity matrices and quantitatively by a precipitous drop in their effective rank (Fig. 1(a),
1(c)).

## 5.1.2 Experimental Validation Of Key Assumptions

To ground our theoretical analysis in the observed dynamics, we now provide direct empirical validation for the key assumptions that underpin our framework: Assumption 1 and Assumption 2.

A

v e r a g e W

t F

(c)
WQ & WK WV & W[1]&W[2]
Di rect io n s im ila rity
(b)
Sa tisf ied Pro po rti on
(a)
0 1000 2000 3000 4000 Step 0.0 2.5 5.0 7.5 10.0 12.5 15.0 Condition 1 Condition 2 0 10 20 30 40 Step / 100 0.0 0.2 0.4 0.6 0.8 1.0 1.0 0 200 400 Step 0.5
First, we examine the condensation condition. Figure 2(a) plots the proportion of satisfied conditions in Assumption 1. The proportion rapidly approaches 1 within the first 200 training steps, confirming that the outer parameters quickly converge to a state where this assumption holds. Next, we validate the assumption of dynamics separation. As discussed in the previous section, our observation that the gradual change of outer parameters during Stage 2 and flat loss curve (often means a critical point has appeared) already provide strong qualitative support for the first part of Assumption 2. To analyze this more rigorously, we examine its empirical variant, Assumption 2*. This assumption points that the direction of parameters remains unchanged and the leading-order loss changes very slowly.

Figure 2(b) shows the cosine similarity between the singular vectors of the outer parameter matrices at adjacent time steps. The similarity for all outer parameters remains extremely close to 1 after the first stage. This indicates that the subspace spanned by these parameters is highly stable, meaning their directional structure is effectively frozen. This stability, combined with the flat loss curve observed in Stage 2, provides compelling evidence for Assumption 2*. Figure 2(c) validates the scale separation implied by the assumption. It shows that by the onset of Stage 2, the Frobenius norms of the outer parameters have grown significantly, while the norms of the attention parameters remain small and close to their initialization values. This confirms the expected scale difference between the two parameter groups, where outer parameters are O(1) and attention parameters are O(δ).

## 5.2 Real Task

We further validate our theoretical predictions on a real-world language modeling benchmark, Wiki- Text Merity et al. [2017]. Unlike the synthetic setup, where anchor functions are explicitly defined, WikiText provides natural linguistic dependencies and high distributional variability. This allows us to test whether the proposed two-stage dynamics, early condensation of outer parameters followed by attention-driven rank collapse, persist in realistic Transformer training. In this setting, we employ a two-layer transformer with GeLU activation and residual connections. To keep the consistency of architecture and focused on the core dynamics, layer normalization is omitted. Further experimental settings are provided in Appendix B.3.

As shown in Figure 3, the two-layer Transformer on WikiText exhibits the same stage-wise dynamics observed in the synthetic experiments. During the initial phase, the outer parameters (WV ,W[1],W[2]) in both layers undergo rapid condensation, while the attention weights

0 2000 4000 6000 8000 10000 12000 Step 9.4 9.6 9.8 10.0 10.2 10.4 Stage 1 Stage 2 Further Training Loss
(WQ,WK) remain largely unchanged. As training proceeds and the loss enters a plateau, the attention parameters begin to evolve, displaying a sharp rank collapse that reorganizes internal representations.

This empirical observation confirms that the separation between outer-parameter condensation and attention-driven rank reduction is not an artifact of the synthetic dataset but also emerges naturally in real-world text modeling. The consistent appearance of this two-stage dynamic across both synthetic and natural settings suggests that implicit regularization, first through low-rank condensation and then through targeted attention adaptation, may serve as a general mechanism underlying the emergence of structured representations in Transformer models.

## 6 Discussion 6.1 Conclusion

This work advances the theoretical understanding of transformer training dynamics by establishing a two-stage analytical framework. Through gradient flow analysis, we show that small initialization helps models escape degenerate regions via asymmetric weight updates, leading to condensation of parameter matrices toward task-relevant directions. In the subsequent stage, the key-query matrices undergo a coordinated collapse that further refines the learned representations. Together, these results clarify the mechanisms underlying the condensation and rank collapse phenomena, providing a principled foundation for future studies on Transformer optimization and generalization.

## 6.2 Limitations

While this work provides valuable theoretical insights, its most significant constraint stems from analyzing exclusively binary classification scenarios: a simplification dictated by technical barriers in gradient flow analysis. This narrow scope inherently precludes insights into transformers' dynamics in practical multi-class classification or sequence-to-sequence learning contexts, where complex interactions between multiple prediction targets and attention mechanisms likely emerge. Though focused theoretical simplification is methodologically justified, extending this framework to broader problem domains remains critical for unifying theory with real-world transformer optimization. Future work should prioritize overcoming these technical limitations to theoretically verify whether our conclusions hold true beyond binary settings.

## Acknowledgments And Disclosure Of Funding

This work is sponsored by the National Key R&D Program of China Grant No. 2022YFA1008200 (T. L.). We also thank Shanghai Institute for Mathematics and Interdisciplinary Sciences (SIMIS) for their financial support. This research was funded by SIMIS under grant number SIMIS-ID-2025- ST. The authors are grateful for the resources and facilities provided by SIMIS, which were essential for the completion of this work. We thank Pengxiao Lin for insightful discussions and support and encouragement to the authors.

## References

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=
0g0X4H8yN4I.

Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. *Advances in Neural Information Processing Systems*, 32, 2019a.

Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. Advances in neural information processing systems, 32, 2019b.

Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley. Rezero is all you need: fast convergence at large depth. In Cassio de Campos and Marloes H. Maathuis, editors, Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, volume 161 of *Proceedings of Machine Learning Research*, pages 1352–1361. PMLR, 27–30 Jul 2021. URL https://proceedings.mlr.press/v161/bachlechner21a.

html.

Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= liMSqUuVg9.

Zhiwei Bai, Jiajie Zhao, and Yaoyu Zhang. Connectivity shapes implicit regularization in matrix factorization models for matrix completion. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=9jgODkdH0F.

Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=3X2EbBLNsk.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Vivien Cabannes, Berfin Simsek, and Alberto Bietti. Learning associative memories with gradient descent, 2024. URL https://arxiv.org/abs/2402.18724.

Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. *arXiv preprint* arXiv:2402.19442, 2024a.

Zheng-An Chen, Yuqing Li, Tao Luo, Zhangchen Zhou, and Zhi-Qin John Xu. Phase diagram of initial condensation for two-layer neural networks. CSIAM Transactions on Applied Mathematics, 5(3):448–514, 2024b. ISSN 2708-0579. doi: https://doi. org/10.4208/csiam-am.SO-2023-0016. URL https://global-sci.com/article/91025/
phase-diagram-of-initial-condensation-for-two-layer-neural-networks.

Liam Collins, Advait Parulekar, Aryan Mokhtari, Sujay Sanghavi, and Sanjay Shakkottai. In-context learning with transformers: Softmax attention adapts to function lipschitzness. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 92638–92696. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ a8633d27d782f66fe660c2fb4bae446e-Paper-Conference.pdf.

Benjamin L. Edelman, Ezra Edelman, Surbhi Goel, Eran Malach, and Nikolaos Tsilivis. The evolution of statistical induction heads: In-context learning markov chains, 2024. URL https:
//arxiv.org/abs/2402.11004.

Tianyu Guo, Wei Hu, Song Mei, Huan Wang, Caiming Xiong, Silvio Savarese, and Yu Bai. How do transformers learn in-context beyond simple functions? a case study on learning with representations. In *The Twelfth International Conference on Learning Representations*, 2024. URL https://openreview.net/forum?id=ikwEDva1JZ.

Xiao Shi Huang, Felipe Perez, Jimmy Ba, and Maksims Volkovs. Improving transformer optimization through better initialization. In *International Conference on Machine Learning*, pages 4475–4483. PMLR, 2020.

Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers, 2023. URL
https://arxiv.org/abs/2310.05249.

M Emrullah Ildiz, Yixiao Huang, Yingcong Li, Ankit Singh Rawat, and Samet Oymak. From selfattention to markov models: Unveiling the dynamics of generative transformers. arXiv preprint arXiv:2402.13512, 2024.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. *Advances in neural information processing systems*, 31, 2018.

Ziwei Ji and Matus Telgarsky. Gradient descent aligns the layers of deep linear networks. In International Conference on Learning Representations, 2019. URL https://openreview.net/ forum?id=HJflg30qKX.

Akshay Kumar and Jarvis Haupt. Early directional convergence in deep homogeneous neural networks for small initializations. *arXiv preprint arXiv:2403.08121*, 2024.

Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. Algorithmic regularization in over-parameterized matrix sensing and neural networks with quadratic activations. In Conference On Learning Theory, pages 2–47. PMLR, 2018.

Yuchen Li, Yuanzhi Li, and Andrej Risteski. How do transformers learn topic structure: Towards a mechanistic understanding, 2023. URL https://arxiv.org/abs/2303.04245.

Haoye Lu, Yongyi Mao, and Amiya Nayak. On the dynamics of training attention models. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=1OCTOShAmqB.

Tao Luo, Zhi-Qin John Xu, Zheng Ma, and Yaoyu Zhang. Phase diagram for two-layer relu neural networks at infinite-width limit. *The Journal of Machine Learning Research*, 22(1):3327–3373, 2021.

Kaifeng Lyu and Jian Li. Gradient descent maximizes the margin of homogeneous neural networks.

In *International Conference on Learning Representations*, 2020. URL https://openreview. net/forum?id=SJeLIgBKPS.

Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=8p3fu56lKc.

Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of twolayer neural networks. *Proceedings of the National Academy of Sciences*, 115(33):E7665–E7671, 2018.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In *International Conference on Learning Representations*, 2017. URL https:
//openreview.net/forum?id=Byj72udxe.

Hancheng Min, Salma Tarmoun, Rene Vidal, and Enrique Mallada. On the explicit role of initialization on the convergence and implicit bias of overparametrized linear networks. In Marina Meila and Tong Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 7760–7768. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/min21c.html.

Eshaan Nichani, Alex Damian, and Jason D. Lee. How transformers learn causal structure with gradient descent. In *Forty-first International Conference on Machine Learning*, 2024. URL https://openreview.net/forum?id=jNM4imlHZv.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022. URL https://arxiv.org/ abs/2209.11895.

Gautam Reddy. The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In *The Twelfth International Conference on Learning Representations*, 2024. URL https://openreview.net/forum?id=aN4Jf6Cx69.

Grant Rotskoff and Eric Vanden-Eijnden. Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks. Advances in neural information processing systems, 31, 2018.

Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *arXiv preprint arXiv:1312.6120*, 2013.

Charles Burton Snell, Ruiqi Zhong, Dan Klein, and Jacob Steinhardt. Approximating how single head attention learns. *ArXiv*, abs/2103.07601, 2021. URL https://api.semanticscholar. org/CorpusID:232232786.

Mahdi Soltanolkotabi, Dominik Stöger, and Changzhi Xie. Implicit balancing and regularization:
Generalization and convergence guarantees for overparameterized asymmetric matrix sensing. In The Thirty Sixth Annual Conference on Learning Theory, pages 5140–5142. PMLR, 2023.

Dominik Stöger and Mahdi Soltanolkotabi. Small random initialization is akin to spectral learning:
Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction. *Advances in Neural Information Processing Systems*, 34:23831–23843, 2021.

Yuandong Tian, Yiping Wang, Beidi Chen, and Simon S Du. Scan and snap: Understanding training dynamics and token composition in 1-layer transformer. Advances in neural information processing systems, 36:71911–71947, 2023.

Yuandong Tian, Yiping Wang, Zhenyu Zhang, Beidi Chen, and Simon Shaolei Du. JoMA: Demystifying multilayer transformers via joint dynamics of MLP and attention. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=LbJqRGNYCf.

Aditya Vardhan Varre, Maria-Luiza Vladarean, Loucas Pillaud-Vivien, and Nicolas Flammarion. On the spectral bias of two-layer linear networks. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=FFdrXkm3Cz.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information* processing systems, 30, 2017.

Francis Williams, Matthew Trager, Daniele Panozzo, Claudio Silva, Denis Zorin, and Joan Bruna.

Gradient dynamics of shallow univariate relu networks. Advances in neural information processing systems, 32, 2019.

Zhi-Qin John Xu, Yaoyu Zhang, and Zhangchen Zhou. An overview of condensation phenomenon in deep learning. *arXiv preprint arXiv:2504.09484*, 2025.

Junjie Yao, Zhongwang Zhang, and Zhi-Qin John Xu. An analysis for reasoning bias of language models with small initialization, 2025. URL https://arxiv.org/abs/2502.04375.

Yichun Yin, Wenyong Huang, Kaikai Song, Yehui Tang, Xueyu Wu, Wei Guo, Peng Guo, Yaoyuan Wang, Xiaojun Meng, Yasheng Wang, Dong Li, Can Chen, Dandan Tu, Yin Li, Fisher Yu, Ruiming Tang, Yunhe Wang, Baojun Wang, Bin Wang, Bo Wang, Boxiao Liu, Changzheng Zhang, Duyu Tang, Fei Mi, Hui Jin, Jiansheng Wei, Jiarui Qin, Jinpeng Li, Jun Zhao, Liqun Deng, Lin Li, Minghui Xu, Naifu Zhang, Nianzu Zheng, Qiang Li, Rongju Ruan, Shengjun Cheng, Tianyu Guo, Wei He, Wei Li, Weiwen Liu, Wulong Liu, Xinyi Dai, Yonghan Dong, Yu Pan, Yue Li, Yufei Wang, Yujun Li, Yunsheng Ni, Zhe Liu, Zhenhe Zhang, and Zhicheng Liu.

Pangu ultra: Pushing the limits of dense large language models on ascend NPUs, 2025. URL https://arxiv.org/abs/2504.07866.

Biao Zhang, Ivan Titov, and Rico Sennrich. Improving deep transformer with depth-scaled initialization and merged attention. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 898–909, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1083. URL https://aclanthology.org/D19-1083/.

Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context.

Journal of Machine Learning Research, 25(49):1–55, 2024a.

Yaoyu Zhang, Zhi-Qin John Xu, Tao Luo, and Zheng Ma. A type of generalization error induced by initialization in deep neural networks. In *Mathematical and Scientific Machine Learning*, pages 144–164. PMLR, 2020.

Yedi Zhang, Aaditya K. Singh, Peter E. Latham, and Andrew Saxe. Training dynamics of in-context learning in linear attention, 2025a. URL https://arxiv.org/abs/2501.16265.

Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, and Zhi-Qin John Xu. Initialization is critical to whether transformers fit composite functions by inference or memorizing, 2024b. URL https://arxiv.org/abs/2405.05409.

Zhongwang Zhang, Zhiwei Wang, Junjie Yao, Zhangchen Zhou, Xiaolong Li, Zhi-Qin John Xu, et al. Anchor function: a type of benchmark functions for studying language models. arXiv preprint arXiv:2401.08309, 2024c.

Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, and Zhi-Qin John Xu. Complexity control facilitates reasoning-based compositional generalization in transformers. arXiv preprint arXiv:2501.08537, 2025b.

Hanxu Zhou, Zhou Qixuan, Tao Luo, Yaoyu Zhang, and Zhi-Qin Xu. Towards understanding the condensation of neural networks at initial training. *Advances in Neural Information Processing* Systems, 35:2184–2196, 2022.

Zhangchen Zhou, Hanxu Zhou, Yuqing Li, and Zhi-Qin John Xu. Understanding the initial condensation of convolutional neural networks. *arXiv preprint arXiv:2305.09947*, 2023.

Chen Zhu, Renkun Ni, Zheng Xu, Kezhi Kong, W Ronny Huang, and Tom Goldstein. Gradinit:
Learning to initialize neural networks for stable and efficient training. Advances in Neural Information Processing Systems, 34:16410–16422, 2021.

## Neurips Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: **The papers not including the checklist will be desk rejected.** The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit. Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
- You should answer [Yes] , [No] , or [NA] . - [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

- Please provide a short (12 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper. The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found. IMPORTANT, please:
- Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
- **Keep the checklist subsection headings, questions/answers and guidelines below.** - **Do not modify the questions and only use the provided macros for your answers**.

## 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We elaborate on our setups and contribution in the abstract and introduction, especially in the last paragraph of the introduction. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The limitations could be found at Sec.6.2. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We provide the definitions, assumptions and proofs at Sec. 4 and Appendix A. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems.

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We show the experiment setup in Sec. B. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code is provided in the supplementary materials. Guidelines:
- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We show the experiment setup in Sec. B. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We show the error bar in Fig. 1 for the anchor function experiments and Fig. 3 for WikiText task. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We provide the information of compute resources in Appendix. C. Guidelines:
- The answer NA means that the paper does not include experiments.

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This work is a phenomenological study, therefore, there is no societal impact of the work performed. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA]
Justification: Justification: This work is a phenomenological study, therefore, this work poses no such risks.

## Guidelines:

- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: All the assets mentioned in paper is open-sourced and properly cited. Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA] Justification: The paper does not release new assets. Guidelines:
- The answer NA means that the paper does not release new assets. - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?