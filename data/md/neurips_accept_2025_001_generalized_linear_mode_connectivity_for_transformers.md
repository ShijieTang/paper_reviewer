# Generalized Linear Mode Connectivity For Transformers

| Alexander Theus 1 2   | Alessandro Cabodi 1   | Sotiris Anagnostidis 1   |
|-----------------------|-----------------------|--------------------------|
| atheus@ethz.ch        | acabodi@ethz.ch       | sanagnos@ethz.ch         |
| Antonio Orvieto 3 4 5 | Sidak Pal Singh∗1 2   | Valentina Boeva∗1 6 7    |
| antonio@tue.ellis.eu  | contact@sidakpal.com  | vboeva@ethz.ch           |

## Abstract

Understanding the geometry of neural network loss landscapes is a central question in deep learning, with implications for generalization and optimization. A striking phenomenon is *linear mode connectivity* (LMC), where independently trained models can be connected by low- or zero-barrier paths, despite appearing to lie in separate loss basins. However, this is often obscured by symmetries in parameter space—such as neuron permutations—which make functionally equivalent models appear dissimilar. Prior work has predominantly focused on neuron reordering through permutations, but such approaches are limited in scope and fail to capture the richer symmetries exhibited by modern architectures such as Transformers. In this work, we introduce a unified framework that captures four symmetry classes—permutations, semi-permutations, orthogonal transformations, and general invertible maps—broadening the set of valid reparameterizations and subsuming many previous approaches as special cases. Crucially, this generalization enables, for the first time, the discovery of low- and zero-barrier linear interpolation paths between independently trained Vision Transformers and GPT-2 models. Furthermore, our framework extends beyond pairwise alignment, to multi-model and width-heterogeneous settings, enabling alignment across architectures of different sizes. These results reveal deeper structure in the loss landscape and underscore the importance of symmetry-aware analysis for understanding model space geometry. Our code is available here.

## 1 Introduction

Understanding the geometry of neural network loss landscapes is central to both theoretical and practical advances in deep learning. A key observation driving this line of research is that independently trained models, despite converging to different points in parameter space, can sometimes be connected by low-loss paths, suggesting an unexpected interrelation between seemingly isolated loss minima [Freeman and Bruna, 2016, Garipov et al., 2018, Tatro et al., 2020]. When these connecting paths are approximately linear and maintain low loss throughout, the phenomenon is known as *Linear Mode Connectivity* (LMC) [Frankle et al., 2020, Entezari et al., 2021]. LMC challenges the naive view of loss landscapes as consisting of distinct basins separated by high barriers. Instead, it hints at a reparameterization-induced redundancy: multiple minima that appear distant in weight space may correspond to functionally similar solutions, made to appear dissimilar due to symmetries in the parameterization. Recovering LMC between models thus requires aligning their parameters into symmetry-equivalent configurations that reveal their underlying functional equivalence. Several state-of-the-art approaches leverage discrete neuron permutations to achieve such alignments, demonstrating LMC for relatively simple architectures like shallow multilayer perceptrons (MLPs) and, under certain conditions, VGG networks and ResNets [Singh and Jaggi, 2020, Ainsworth et al., 2022]. Such works highlight the central role of symmetries in understanding neural network landscapes and inspired the development of so-called model *re-basin* techniques: transformations that port independently trained networks into a common valley of the loss landscape. Although foundational, permutation symmetries alone do not capture the full range of symmetries exhibited by modern architectures such as Transformers. Our empirical findings show that relying solely on discrete reordering often fails to reveal low-loss paths, as persistent barriers can remain even after permutation alignment [Verma and Elbayad, 2024]. In this work, we broaden the symmetry lens. We introduce a unified framework that formalizes four classes of transformations: permutations, semi-permutations, orthogonal transformations, and general invertible maps which, if appropriately used, can produce valid reparameterizations under which the original model functionality is preserved, while achieving low barriers. This generalization subsumes several existing alignment techniques as special cases and provides the means to uncover LMC in Transformers by allowing to identify and leverage their richer symmetry classes. Furthermore, this formalization also accommodates alignment between heterogeneous Transformers with differing architectural widths. Our empirical results demonstrate for the first time low- and zero-loss linear connections between independently trained Vision Transformers and GPT-2 models, even in multi-model settings. These findings suggest that the Transformers' loss landscape is more connected than previously thought, provided the symmetries at play are adequately modeled and exploited (see Figure 1). We discuss broader implications for ensembling, federated and continual learning, adversarial robustness, and the role of positional encodings in Appendix F. This paper advances the understanding of loss landscape geometry and model alignment through the following key contributions:
Figure 1: By considering network symmetries beyond permutations, we can teleport two independently trained Transformers to the same loss basin. ΘB is projected into a functionally equivalent representation π(ΘB).

i **Unified symmetry framework:** We formalize a broad class of parameter transformations —
including permutations, semi-permutations, orthogonal transformations, and general invertible maps - that preserve model functionality. This unifies and extends prior approaches under a single theoretical lens and enables alignment across both homogeneous and heterogeneous architectures (Section 3).

ii **LMC for Transformers:** We demonstrate, for the first time, low- and zero-loss linear paths between independently trained Vision Transformers and GPT-2 models using richer symmetry classes (Section 4 and 5).

iii **Multi-model mode connectivity:** Our framework extends to the multi-model setting, revealing that several independently trained transformers can be merged while maintaining a near-zero interpolation barrier (Section 4 and 5).

iv **Soft alignment via continuous symmetries:** We show that relaxing exact equivalence through differentiable, non-discrete transformations may improve interpolation outcomes (Appendix D).

## 2 Generalizing Linear Mode Connectivity

Let θ be network parameters and f[θ](·) the induced function. Let ℓ[θ](*x, y*) be the loss incurred by f[θ] on a data point (*x, y*). For a dataset D = {(xi, yi)}
N
i=1, define the empirical risk

$${\mathcal{L}}[\mathbf{\theta}]({\mathcal{D}})={\frac{1}{N}}\sum_{i=1}^{N}\ell[\mathbf{\theta}](x_{i},y_{i}).$$
$$(1)$$
Definition (Linear Mode Connectivity). Consider two models θA, θB, both pre-trained on the
same task (for simplicity). They are linearly mode connected if L[θA](D) ≃ L[θB](D) and the
interpolation barrier
$$\mathcal{B}_{\lambda}[\boldsymbol{\theta}_{A},\boldsymbol{\theta}_{B}](\mathcal{D})=\mathcal{L}[\lambda\boldsymbol{\theta}_{A}+(1-\lambda)\boldsymbol{\theta}_{B}](\mathcal{D})-\left(\lambda\,\mathcal{L}[\boldsymbol{\theta}_{A}](\mathcal{D})+(1-\lambda)\,\mathcal{L}[\boldsymbol{\theta}_{B}](\mathcal{D})\right),$$  is near-zero for all $\lambda\in[0,1]$. The empirical barrier is:  $$\mathcal{B}[\boldsymbol{\theta}_{A},\boldsymbol{\theta}_{B}](\mathcal{D})=\sup_{\lambda\in[0,1]}\mathcal{B}_{\lambda}[\boldsymbol{\theta}_{A},\boldsymbol{\theta}_{B}](\mathcal{D}),$$

and LMC is observed when B[θA, θB](D) ≈ 0.

Because neural networks admit non-unique parameterizations [Li et al., 2023], especially under different initializations, we seek invertible, function-preserving alignment mappings π : Θ → Θ that, when applied to θA and θB, encourage LMC:

$$\min_{\pi_{A},\pi_{B}}\ {\cal B}[\pi_{A}(\mathbf{\theta}_{A}),\pi_{B}(\mathbf{\theta}_{B})]({\cal D})\quad\mbox{s.t.}\quad f[\pi_{A}(\mathbf{\theta}_{A})]=f[\mathbf{\theta}_{A}],\ f[\pi_{B}(\mathbf{\theta}_{B})]=f[\mathbf{\theta}_{B}].$$

These mappings need not be mere permutations. As discussed in Section 3, modern architectures (e.g., Transformers) exhibit richer, component-specific symmetries that can more effectively reduce—or eliminate—the barrier (Appendix E). This broader view is not confined to networks with identical architecture and naturally extends to enable alignment across models of differing widths.

Extension to multi-model connectivity. For M independently trained models {θm}Mm=1, define the *multi-model barrier*

$$\mathcal{B}[\{\mathbf{\theta}_{m}\}_{m=1}^{M}](\mathcal{D})=\sup_{\mathbf{\lambda}\in\Delta^{M-1}}\left[\mathcal{L}\bigg{[}\sum_{m=1}^{M}\lambda_{m}\mathbf{\theta}_{m}\bigg{]}(\mathcal{D})-\sum_{m=1}^{M}\lambda_{m}\,\mathcal{L}[\mathbf{\theta}_{m}](\mathcal{D})\right],\tag{2}$$  ${}^{-1}=\{\mathbf{\lambda}\in\mathbb{R}_{\geq0}^{M}:\sum_{m}\lambda_{m}=1\}$. Multi-model linear connectivity holds when 
$$\pi_{A}(\mathbf{\theta}_{A})]=f[\mathbf{\theta}_{A}],\ f[\pi_{B}(\mathbf{\theta}_{B})]=f[\mathbf{\theta}_{B}].$$

where ∆M−1 = {λ ∈ RM
≥0:Pm λm = 1}. Multi-model linear connectivity holds when B[{θm}](D) ≈ 0, indicating a shared low-loss basin. As in the pairwise case, we search for symmetry-preserving mappings {πm}Mm=1 satisfying f[πm(θm)] = f[θm] that minimize B[{πm(θm)}](D), forming the basis for the merging procedure in Section 4.3.

## 3 Network Symmetries Under The Generalized Framework

To perform alignment and enable meaningful interpolation between independently trained models, we must first understand the underlying symmetries that govern neural network parameter spaces. While prior work has focused primarily on discrete permutations, these represent only a slice of the broader symmetry landscape. In this section, we introduce a hierarchy of network symmetries—permutation, semi-permutation, *orthogonal*, and *invertible*—each allowing progressively more flexible function-preserving transformations, as summarized in Table 1. We define each class, identify where it arises in neural architectures, and conclude by showing how these symmetries manifest in Transformer models, enabling their effective alignment and merging. More theoretical grounding and formal analysis of these symmetry classes can be found in Appendix E.

## 3.1 Symmetry Classes

Permutation. Permutation symmetry refers to transformations that reorder inputs while preserving the network's function. It arises when components treat each input dimension independently, allowing neuron reordering without affecting the output. This symmetry is characteristic of elementwise operations such as GELU, sigmoid, softmax, tanh, where output values correspond directly to input positions.

| Hierarchy   | Class            | Structure                        | Examples                     |
|-------------|------------------|----------------------------------|------------------------------|
| S1          | Permutation      | Permutation matrices (P)         | GELU, sigmoid, softmax, tanh |
| ⊂ S2        | Semi-permutation | Sparse, stochastic matrices (P˜) | RELU, LayerNorm, MHA         |
| ⊂ S3        | Orthogonal       | Orthogonal matrices (O)          | RMSNorm                      |
| ⊂ S4        | Invertible       | Full-rank matrices               | Linear layer                 |

Table 1: Hierarchical organization of symmetry classes in neural network components, illustrating their associated transformation structures and representative examples. Each class is a strict subset of the one below it. Semi-permutation. "Semi-permutation" symmetry extends permutation symmetry by allowing sparse, weighted mixing of input dimensions. It is defined by matrices P ∈ RM×N , where M ≥ N,
each column is a stochastic vector, and each row contains at most one positive entry. This symmetry arises in components that are *linearly decomposable*—that is, their functional output satisfies the following identity:
f(x) = f (αx) + f ((1 − α)x), ∀α ∈ [0, 1],
which holds for piecewise-linear functions such as RELU, PRELU, and the absolute value. These functions permit structured, non-permutative mixing of input channels while preserving functionality. As permutations are a subset of semi-permutations, many existing works focus on permutation symmetries of RELU based activation networks. Orthogonal. Orthogonal symmetry allows transformations that preserve vector norms and angles—such as rotations and reflections—without altering the network's behavior. This symmetry arises in components that normalize inputs across dimensions, irrespective of their orientation in space. *RMSNorm* is a key example, remaining invariant under orthogonal transformations. Invertible. Neural network components that preserve linearity admit functional equivalence under invertible transformations - the most general symmetry class in our hierarchy. This class includes layers whose learned transformations are unconstrained by structural restrictions such as sparsity or orthogonality. A key example is the *attention mechanism*, where the QK and OV circuits [Elhage et al., 2021] (i.e., the projection weights for queries, keys, and values) can be reparameterized via invertible maps without altering model behavior. Approximate invariance and soft symmetries. The strict requirement of functional equivalence f[θ
′](X) = f[θ](X) can be relaxed to approximate equality f[θ
′](X) ≈ f[θ](X), allowing continuous transformations to serve as soft symmetry operations. In this setting, soft permutations are represented by doubly stochastic matrices, computed via entropic optimal transport or Sinkhornbased projections. Such relaxations enable more flexible neuron alignments—e.g., many-to-one or one-to-many mappings— and can improve test-time performance (see Appendix D).

## 3.2 Symmetries In Transformers 3.2.1 Feed-Forward Layer

The Transformer's feed-forward layer applies two linear projections with a nonlinearity in between:
FF(x) = W2 ϕ(W1x + b1) + b2, where ϕ is an elementwise activation function, typically GELU or RELU. When ϕ is not linearly decomposable—such as GELU—the layer only admits a strict *permutation symmetry*. If ϕ instead is piecewise linear (e.g., RELU), the layer also admits a broader *semi-permutation symmetry* as described in Section 3.1. For any permutation matrix PFF ∈ R
h×h, the reparameterization

$\bf{W_{1}}$, $\bf{W_{2}^{\prime}=W_{2}P_{\rm{FF}}^{\top}}$, $\bf{b_{1}^{\prime}=P_{\rm{FF}}b_{1}}$
$$\mathbf{W}_{1}^{\prime}=\mathbf{P}_{\mathrm{{FF}}}\mathbf{W}_{1},$$
yields an equivalent function:
FF′(x) = FF(x).

+
+
W1 Activation Function W2 O⊤W1PFF Activation Function P⊤
FFW2O
FFN
FFN
+
x
∥x∥
+
x
∥x∥
Attention RMSNorm Attention RMSNorm Wq Wk Wv Multi-Head Attention Wo
(O⊤Wq) ⋄ P˜H (O⊤Wk) ⋄ P˜ H (O⊤Wv) ⋄ P˜ H Multi-Head Attention P˜ ⊤
H⋄ (WoO)
Inputs Inputs
(a) Transformer layer.

(b) Transformer layer after projection.
Figure 2: (a) illustrates a standard Transformer layer, and (b) shows its function-preserving equivalent after structured weight transformations. Attention heads are semi-permuted via blockwise multiplication with P˜ H using the ⋄ operator, feedforward weights are permuted via PFF, and residual stream weights are orthogonally transformed via O. The figures are inspired by Ashkboos et al. [2024].

## 3.2.2 Multi-Head Attention

The multi-head attention mechanism of Transformers is defined as:

$$\operatorname{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V})=\sum_{i=1}^{H}\underbrace{\operatorname{softmax}\left(\frac{(\mathbf{Q}\mathbf{W}_{i}^{Q})(\mathbf{K}\mathbf{W}_{i}^{K})^{\top}}{\sqrt{d_{k}}}\right)}_{\operatorname{head}_{i}(\mathbf{Q},\mathbf{K},\mathbf{V})}(\mathbf{V}\mathbf{W}_{i}^{V})\,\mathbf{W}_{i}^{O}\,.$$

Intra-head. Each attention head exhibits two *invertible symmetries*, as formally characterized in the QK and OV circuits by Elhage et al. [2021]. The first arises from the product WQ
i
(WK
i)
⊤,
which governs the attention scores (QK-circuit). The second appears in WV
i WO
i, which maps values to outputs (OV-circuit). Because any invertible transformation applied within the query, key, value, or output projections can be algebraically multiplied out, the behavior of the attention mechanism is uniquely determined by these two products. Consequently, we omit explicit invertible reparameterizations and treat the QK and OV circuits as canonical representations of each head. Inter-head. Multi-head attention exhibits a *semi-permutation symmetry* across heads. Since heads are summed independently, their order is irrelevant (permutation symmetry). Moreover, heads can be decomposed linearly:
head(X; QK, OV) = head(X; QK, α · OV) + head(X; QK,(1 − α) · OV),
for any α ∈ R. This enables structured reweighting and mixing of heads via sparse, stochastic matrices P˜ H, placing multi-head attention in the semi-permutation class.

## 3.2.3 Residual

Recently, Ashkboos et al. [2024] observed that Transformers exhibit an orthogonal symmetry along the residual path, where the only non-linear component is the normalization layer. When RMSNorm is employed, the model exhibits orthogonal symmetry directly (see Section 3.1). In contrast, when LayerNorm is used, it can be reformulated in terms of RMSNorm as follows:
LayerNorm(Z) = RMSNorm(ZM) · diag(α)
√
D + 1N β
⊤,
where α and β are learnable scale and offset parameters, respectively, specific to each LayerNorm instance. The matrix M = ID −
1 D
11⊤ centers each row of Z by subtracting its mean. The matrix M and diag(α)
√D can then be absorbed in preceding and subsequent linear layers.

Moreover, since the orthogonal transformation O can be chosen as a rectangular matrix with orthonormal columns (O ∈ RM×N , M ≥ N), this symmetry enables width-expanding transformations that preserve functionality. We provide a detailed derivation of this property in Appendix E, and demonstrate its empirical utility in Section 5. Figure 2 illustrates how applying any orthogonal matrix O to the residual stream of a Transformer—after RMSNorm reparameterization—yields a functionally equivalent model.

## 4 Method

Building on the symmetry framework from Section 3.1, we now describe methods for aligning two independently trained Transformer models. The goal is to find function-preserving transformations - constrained to the relevant symmetry classes - that bring the models into structural and representational agreement.

Concretely, given a Transformer with L layers, hidden dimension dh, residual embedding size dr, and H attention heads per layer, alignment involves estimating a set of symmetry-constrained matrices: (i) a global orthogonal matrix O ∈ R
dr×drfor the residual stream; (ii) L permutation matrices PF F ∈ R
dh×dh for aligning neurons in the feed-forward layers; and (iii) L semi-permutation matrices P˜ H ∈ R
H×H for aligning attention heads across layers.

We consider three strategies for estimating these transformations. *Activation matching* aligns layers by comparing intermediate activations on a shared dataset. *Weight matching* aligns parameters directly by minimizing distance under the symmetry constraints. *Learned matching* treats alignment as an optimization problem, learning the symmetry-aware re-parameterizations end-to-end. For activation matching, we use the method introduced by Verma and Elbayad [2024]. See Appendix C for an ablation study of our proposed algorithms.

## 4.1 Weight Matching

We adapt the weight-based alignment strategy introduced in Ainsworth et al. [2022], which formulates alignment as an optimization problem over permutation matrices that maximize weight similarity across networks. The core intuition is that if two units across models have similar incoming and outgoing weights, they will likely implement similar functions and thus be aligned. For Transformer feed-forward layers, we adopt a layerwise version of the "sum of bilinear assignments problem" (SOBLAP) proposed by Ainsworth et al. [2022]. Given weights W(A)
ℓand W(B)
ℓ in layer ℓ, we search for a permutation PFF
ℓthat maximizes alignment:

$$\mathbf{P}_{\ell}^{\mathrm{{FF}}}=\operatorname*{arg\,max}_{\mathbf{P}\in S_{d}}\;\langle\mathbf{W}_{\ell}^{(A)},\,\mathbf{PW}_{\ell}^{(B)}\mathbf{O}^{\top}\rangle_{F}+\langle\mathbf{W}_{\ell+1}^{(A)},\,\mathbf{OW}_{\ell+1}^{(B)}\mathbf{P}^{\top}\rangle_{F},$$

⊤⟩F , (3)
where Sd is the set of permutation matrices, and O is the orthogonal matrix for the residual stream.
This objective is NP-hard, but can be approximated using a coordinate descent strategy where each
PFF
ℓis updated by solving a linear assignment problem conditioned on adjacent layers.
For attention layers, we exploit the fact that QK and OV circuits are invariant under invertible reparameterizations [Elhage et al., 2021]. We define the **QK circuit** and **OV circuit** for each head i as:
$\mathbf{Q}\mathbf{K}_{i}:=(\mathbf{0}^{\top}\mathbf{W}_{i}^{Q})(\mathbf{0}^{\top}\mathbf{W}_{i}^{K})^{\top},\quad\mathbf{0}\mathbf{V}_{i}:=\mathbf{0}^{\top}\mathbf{W}_{i}^{V}\mathbf{W}_{i}^{Q}\mathbf{0},$  $\mathbf{0}$\(\mathbf{
We then define a cost matrix for aligning heads between models A and B using the Frobenius norm:
$$({\mathfrak{I}})$$
$$\mathbf{M}_{i,j}^{\mathrm{max}}=\|\mathbf{Q}\mathbf{K}_{i}^{(A)}-\mathbf{Q}\mathbf{K}_{j}^{(B)}\|_{F}^{2}+\|\mathbf{O}\mathbf{V}_{i}^{(A)}-\mathbf{O}\mathbf{V}_{j}^{(B)}\|_{F}^{2}.$$
We solve a linear assignment problem to obtain the head-level permutation matrix PH
ℓ.

For the residual stream, we estimate one global orthogonal matrix O by solving the Procrustes problem:

$$\begin{array}{c}\mathbf{O}=\operatorname*{arg\,min}_{\mathbf{O}\in\mathbb{R}^{d_{r}\times d_{r}},\,\mathbf{O}^{\top}\mathbf{O}=\mathbf{I}}\|\mathbf{R}^{(A)}-\mathbf{R}^{(B)}\mathbf{O}\|_{F}^{2},\end{array}\tag{1}$$

where R(A)and R(B)are the weights along the residual path collected from both models. The closed-form solution is given by the SVD of R(B)⊤R(A).

$$(4)$$

## 4.2 Learned Matching

While activation and weight matching rely on static alignment criteria, learned matching directly optimizes the alignment parameters using task loss as supervision. Rather than aligning weights or activations explicitly, we treat the symmetry transformations themselves as trainable parameters, to be learned end-to-end (see Algorithm 1). Algorithm 1 Learning matching via task loss Require: Base models θA, θB; Dataset D; Iterations Niter; Adam optimizer (lr = η).

1: Initialize latent matrices ZFF, ZH, ZO using weight matching. 2: for t = 1 to Niter do 4: Align: θ aligned B ← π(θB; PFF, PH, O). ▷ π applies transformations 5: Sample: λ ∼ U(0.4, 0.6)
6: Sample batch B = {(Xi,Yi)}
|B| i=1 from D.

7: Interpolate: θINTERP ← λ · θA + (1 − λ) · θ aligned B .

8: Objective: J ← 1 |B| P(X,Y )∈B LCE(θINTERP; X,Y ).

9: Gradients: (gZFF , gZH
, gZO
) ← ∇(ZFF,ZH,ZO)J . ▷ STE for ZFF, ZH
10: Update: For k ∈ {FF,H,O}, Zk ← Adam(Zk, gZk
, η).

11: **end for** 12: Final projections: P
∗
FF ← PROJPERM(ZFF), P
∗
H ← PROJPERM(ZH), O∗ ← PROJORTH(ZO).

13: **return** P
∗
FF, P
∗
H , O∗.

We introduce unconstrained latent matrices ZFF, ZH, and ZO, which are projected to the respective symmetry classes at each forward pass:
PFF = PROJPERM(ZFF), PH = PROJPERM(ZH), O = PROJORTH(ZO). (5)
We initialize these latent matrices using the weight matching procedure described in Section 4.1. For the permutation matrices, PROJPERM projects each matrix to the nearest permutation via the Hungarian algorithm; we use a straight-through estimator to allow gradients to flow through the relaxed Z parameters. The orthogonal projection PROJORTH computes UV⊤ from the SVD of ZO = UΣV⊤. This operation is fully differentiable.

At each step, we interpolate between model A and the reparameterized model B using a uniformly randomly sampled interpolation coefficient λ ∼ U(0.4, 0.6):
θINTERP = λ · θA + (1 − λ) · π(θB),
where π(·) denotes alignment of model B. We then compute the original training loss on θINTERP
and backpropagate through the alignment transformation. This approach encourages Transformer similarity through task performance, rather than explicit similarity of parameters or activation vectors, thus enabling the joint optimization of all symmetryaware transformations over all network layers.

## 4.3 Multi-Model Merging

Universe matching. Following Crisostomi et al. [2024], we build a shared *universe* U
(t)that serves as a common reference for all models. Given trained models θ1*, . . . ,* θM, initialize U
(0)←θs with any seed model (s ∈ {1*, . . . , M*}). For each iteration t = 1:N,

$$\pi_{m}^{(t)}\leftarrow\mathrm{ALIGN}(\mathbf{\theta}_{m},U^{(t-1)})\quad\forall m,\qquad U^{(t)}\leftarrow\frac{1}{M}\sum_{m=1}^{M}\pi_{m}^{(t)}(\mathbf{\theta}_{m}).$$

3: Project: {PFF, PH, O*} ← {*PROJPERM(ZFF), PROJPERM(ZH), PROJORTH(ZO)}.

Table 2: Loss barrier (lower is better) for each alignment method. Reported values are mean ± standard error, with rank in parentheses. Models are width-homogeneous, and only two models are aligned with each other. Rows highlighted in color correspond to our methods, using the same colors as in subsequent figures; bold text indicates the best performance for each dataset.

| ViT                             | GPT-2                                                                           |                 |                 |                  |                 |
|---------------------------------|---------------------------------------------------------------------------------|-----------------|-----------------|------------------|-----------------|
| Method                          | CIFAR-10                                                                        | CIFAR-100       | Tiny ImageNet   | Tiny Shakespeare | BookCorpus      |
| Vanilla averaging               | 1.69 ± 0.07 (5)                                                                 | 2.46 ± 0.04 (5) | 2.84 ± 0.02 (5) | 2.02 ± 0.12 (5)  | 4.34 ± 0.09 (5) |
| Activation matching             | 1.27 ± 0.13 (4)                                                                 | 2.11 ± 0.17 (4) | 1.86 ± 0.10 (4) | 1.43 ± 0.16 (4)  | 4.05 ± 0.13 (4) |
| Weight matching (ours)          | 0.36 ± 0.01 (2)                                                                 | 0.69 ± 0.21 (3) | 0.47 ± 0.04 (3) | 0.34 ± 0.01 (2)  | 1.56 ± 0.02 (2) |
| Learned matching (permutations) | 0.45 ± 0.02 (3)                                                                 | 0.53 ± 0.07 (2) | 0.29 ± 0.02 (2) | 0.63 ± 0.17 (3)  | 1.60 ± 0.04 (3) |
| Learned matching (ours)         | 0.00 ± 0.00 (1) 0.00 ± 0.00 (1) 0.00 ± 0.00 (1) 0.02 ± 0.00 (1) 0.42 ± 0.01 (1) |                 |                 |                  |                 |

A 0.2 0.4 0.6 0.8 ( B)
Interpolation coefficient ( )
5.5 6.0 6.5 7.0 7.5 0.2 0.4 0.6 0.8 ( )
Interpolation coefficient ( )
5.3 5.4 5.5 5.6 5.7 5.8 5.9 Learned Weight Vanilla Learned Weight Vanilla Size 1/16 th 1/8 th 1/4 th 1/2 th A 0.2 0.4 0.6 0.8 ( B)
Interpolation coefficient ( )
2.5 3.0 3.5 4.0 4.5 5.0 Lo ss Lo ss L

o s s

(a) Tiny ImageNet.

(b) Tiny Shakespeare.

(c) Tiny Shakespeare.
Each ALIGN step estimates a function-preserving transformation π
(t)
m constrained to the symmetry classes of Section 3.1. The evolving anchor U
(t)aggregates aligned parameters into a unified coordinate system, and after N iterations the resulting {π
(N)
m } approximately minimize the multi-model barrier in Eq. (2). Learned refinement. We refine {π
(N)
m } using the learned matching method from Section 4.2, extended to M-way mixtures. Sampling λ∼Dirichlet(α1M) with α = 0.1, we minimize

$$\mathcal{J}=\mathbb{E}_{\lambda}\left[\mathcal{L}\bigg{[}\sum_{m=1}^{M}\lambda_{m}\,\pi_{m}(\mathbf{\theta}_{m})\bigg{]}(\mathcal{D})-\sum_{m=1}^{M}\lambda_{m}\,\mathcal{L}[\pi_{m}(\mathbf{\theta}_{m})](\mathcal{D})\right],$$  which directly drives $\mathcal{B}[\{\pi_{m}(\mathbf{\theta}_{m})\}](\mathcal{D})$ toward zero. Gradients are backpropagated through $\pi_{m}$ via 
the projection operators of Section 4.2.

## 5 Results

We evaluate the proposed model alignment methods on two Transformer architectures: Vision Transformers (ViTs) and GPT-2, spanning vision and language tasks. To measure LMC, we compute the loss barrier between two models θA and θB as defined in Equation 1 on the test split. A
lower barrier indicates better connectivity; the optimal value is B = 0. See Appendix C and D for further results. Two-way model alignment. Table 2 and Figure 3 summarize alignment between two independently trained models. Our learned matching method consistently outperforms all alternative approaches, achieving zero or near-zero barriers on every dataset except BookCorpus. The weightmatching variant—fully unsupervised and training-free—also yields substantial reductions and often surpasses permutation-only learned matching (which is akin to the STE-based approach in

(a) Vanilla averaging. (b) Weight matching. (c) Learned matching.
Git Rebasin). The degree of connectivity among Transformer minima is striking: for comparison, achieving a zero barrier for ResNet-20 on CIFAR-10 in Git Rebasin required a 32× width increase [Ainsworth et al., 2022]. Figure 3c further considers width-heterogeneous alignment, aligning a larger model with embedding dimension 512 to smaller counterparts. Despite the architectural mismatch, the interpolation paths remain at or near zero barrier, indicating connected regions across Transformer sizes. While image tasks reliably attain B ≈ 0—including Tiny ImageNet—the language experiments, particularly on BookCorpus, exhibit higher barriers. This may reflect imperfect alignment, or genuinely disconnected minima. Juneja et al. [2023] show that, for NLP classifiers, fine-tuning can yield multiple basins associated with distinct generalization strategies (e.g., lexical-overlap vs. syntactic cues). Thus, while models using the same strategy might be linearly connected, linear paths across strategies exhibit barriers. This suggests the non-zero barriers on BookCorpus may reflect fundamentally different solutions rather than merely suboptimal alignment. Multi-way model alignment. Figure 4 extends the analysis to aligning three independently trained CIFAR-10 models. We visualize the loss over the simplex spanned by π(ΘA), π(ΘB), and π(ΘC ). Relative to vanilla averaging, both weight matching and learned matching flatten the surface toward the linear-interpolation baseline, with learned matching producing the broadest region of near-zero deviation. These results suggest that our procedures connect not only pairs of solutions but also carve out a shared low-loss manifold spanning multiple models.

## 6 Understanding The Gap Between Weight- And Learned Matching

Weight matching is an attractive, data-free alignment method:
it operates directly on parameters, requires no training data, and is computationally efficient—making it practical for settings such as federated learning where data sharing is limited. However, it typically yields higher interpolation barriers than learned matching, which optimizes alignment with task-loss supervision. This raises a key question: where does the gap arise, and can parameter-only methods close it? To locate the gap, we analyze the learned transformations. Across runs, the permutations of attention heads and feedforward blocks found by weight matching remain essentially unchanged under learned matching. The difference concentrates in the orthogonal map O that aligns the residual stream. As shown in Fig. 5, the eigen-angles of OWM are broadly distributed over [0, 2π], indicating nearly arbitrary rotations/reflections, whereas the relative correction Odiff =
OLMO⊤WM concentrates near 0 mod 2π, i.e., learned matching makes small, targeted refinements.

0 3¼
2
¼ 2
¼
In short, learned matching primarily *refines* the orthogonal alignment provided by weight matching. This suggests a promising avenue: better, data-free estimation of O (e.g., with structural priors or spectral objectives) could narrow most of the performance gap without full supervision.

## 7 Related Work

Linear Mode Connectivity (LMC). LMC describes the existence of low-loss linear paths between independently trained networks. Early work on mode connectivity [Garipov et al., 2018, Draxler et al., 2018] identified nearly constant-loss non-linear paths, suggesting SGD solutions lie on a connected manifold. LMC focuses on linear interpolations: Entezari et al. [2021] conjectured that permutation symmetries account for the observed disconnection, and once resolved, SGD solutions lie in a single basin. While a formal proof remains open, empirical evidence increasingly supports this view (see below). Recent work also links global LMC to layer-wise linearity [Zhou et al., 2024, Adilova et al., 2023].

Model merging and symmetry alignment. Several approaches have exploited the symmetry structure of neural networks to enable one-shot model merging. OT fusion [Singh and Jaggi, 2020] casts neuron alignment as an optimal transport (OT) problem, computing the Wasserstein barycenter between corresponding layers based on activation or weight similarity. This enables data-driven model fusion that outperforms naive averaging and can approximate ensemble performance after moderate finetuning. Related methods include Liu et al. [2022], Akash et al. [2022]. Subsequently, Git Re-Basin [Ainsworth et al., 2022] considers three alignment methods (two of which are highly similar to prior work of [Singh and Jaggi, 2020]): activation matching, weight matching (WM), and a learning-based variant using straight-through estimators (STE) [Bengio et al., 2013]. WM is data-free but loss landscape-agnostic; STE backpropagates through soft permutations and depends critically on WM for initialization. This approach achieves zero-barrier interpolation for modified (widened) ResNets with LayerNorm, a result later extended to BatchNorm networks via statistical recalibration [Jordan et al., 2022]. Ito et al. [2025] show that WM aligns dominant singular directions without altering singular values, thereby enabling LMC while preserving functionality. Consequently, it exhibits higher transitivity than STE, which may overfit local loss geometry. Sinkhorn Re-Basin [Peña et al., 2022] instead optimizes relaxed permutations using Sinkhorn operator and implicit differentiation [Eisenberger et al., 2022] to reduce interpolation barriers.

Transformer-specific extensions have also emerged: Imfeld et al. [2023] adapts OT fusion to handle multi-head attention and residual structures using Sinkhorn-based soft alignment, while Verma and Elbayad [2024] uses correlation-based matching to align BERT models. Both show reduced loss barriers compared to naive averaging, though non-zero barriers remain. Beyond merging identical-task models, Stoica et al. [2024] tackles multi-task model merging by aligning both inter- and intra-model features, revealing redundancy in neural representations. Cycleconsistent alignment across multiple models is explored in Crisostomi et al. [2024], enforcing consistency of neuron permutations to support multi-way merging.

## 8 Conclusion

We introduced a unified framework for symmetry-aware model alignment that captures a broad class of transformations—permutation, semi-permutation, orthogonal, and general invertible maps. This generalization subsumes prior re-basin techniques and enables, for the first time, the discovery of low- and zero-loss linear interpolation paths between independently trained Vision Transformers and GPT-2 models. Our empirical results show that broader symmetry classes are essential to uncovering the connectedness of modern neural network loss landscapes. These findings highlight the importance of modeling and leveraging richer symmetries in reparameterization to advance our understanding of neural network geometry, with potential implications for model ensembling, transfer, and interoperability. A more extensive discussion of the theoretical implications and broader impact of these results is provided in Appendix F, and the practical limitations of our framework are summarized in Appendix A.

## Acknowledgements

Alexander Theus and Sidak Pal Singh acknowledge the financial support from the Max Planck ETH Center for Learning Systems. Antonio Orvieto acknowledges the financial support of the Hector Foundation.

## References

Linara Adilova, Asja Fischer, and Martin Jaggi. Layerwise linear mode connectivity. arXiv preprint arXiv:2307.06966, 2023.

Samuel K Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. *arXiv preprint arXiv:2209.04836*, 2022.

Aditya Kumar Akash, Sixu Li, and Nicolás García Trillos. Wasserstein barycenter-based model fusion and linear mode connectivity of neural networks, 2022. URL https://arxiv.org/abs/ 2210.06671.

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. Slicegpt: Compress large language models by deleting rows and columns, 2024. URL https://arxiv.org/abs/2401.15024.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv preprint arXiv:1308.3432*, 2013.

Donato Crisostomi, Marco Fumero, Daniele Baieri, Florian Bernard, and Emanuele Rodolà. c 2m3:
Cycle-consistent multi-model merging, 2024. URL https://arxiv.org/abs/2405.17897.

Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred Hamprecht. Essentially no barriers in neural network energy landscape. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of *Proceedings of Machine* Learning Research, pages 1309–1318. PMLR, 10–15 Jul 2018. URL https://proceedings. mlr.press/v80/draxler18a.html.

Marvin Eisenberger, Aysim Toker, Laura Leal-Taixé, Florian Bernard, and Daniel Cremers. A unified framework for implicit sinkhorn differentiation, 2022. URL https://arxiv.org/abs/ 2205.06688.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. *Transformer Circuits Thread*, 1(1):12, 2021.

Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. *arXiv preprint arXiv:2110.06296*, 2021.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In *International Conference on Machine Learning*, pages 3259–3269. PMLR, 2020.

C Daniel Freeman and Joan Bruna. Topology and geometry of half-rectified network optimization.

arXiv preprint arXiv:1611.01540, 2016.

Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31, 2018.

Moritz Imfeld, Jacopo Graldi, Marco Giordano, Thomas Hofmann, Sotiris Anagnostidis, and Sidak Pal Singh. Transformer fusion with optimal transport, 2023.

Akira Ito, Masanori Yamada, and Atsutoshi Kumagai. Analysis of linear mode connectivity via permutation-based weight matching: With insights into other permutation search methods, 2025.

URL https://arxiv.org/abs/2402.04051.

Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur. Repair: Renormalizing permuted activations for interpolation repair. *arXiv preprint arXiv:2211.08403*, 2022.

Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies, 2023. URL https://arxiv.org/abs/2205.12411.

Weishi Li, Yong Peng, Miao Zhang, Liang Ding, Han Hu, and Li Shen. Deep model fusion: A
survey. *arXiv preprint arXiv:2309.15698*, 2023.

Chang Liu, Chenfei Lou, Runzhong Wang, Alan Yuhan Xi, Li Shen, and Junchi Yan. Deep neural network fusion via graph matching with applications to model ensemble and federated learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of Proceedings of Machine Learning Research, pages 13857–13869. PMLR, 17–23 Jul 2022. URL
https://proceedings.mlr.press/v162/liu22k.html.

Fidel A. Guerrero Peña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric Granger, and Marco Pedersoli. Re-basin via implicit sinkhorn differentiation, 2022. URL https://arxiv.org/abs/2212.12042.

Sidak Pal Singh and Martin Jaggi. Model fusion via optimal transport. Advances in Neural Information Processing Systems, 33:22045–22055, 2020.

George Stoica, Daniel Bolya, Jakob Bjorner, Pratik Ramesh, Taylor Hearn, and Judy Hoffman.

Zipit! merging models from different tasks without training, 2024. URL https://arxiv.org/ abs/2305.03053.

David Stutz, Matthias Hein, and Bernt Schiele. Relating adversarially robust generalization to flat minima, 2021. URL https://arxiv.org/abs/2104.04448.

N. Joseph Tatro, Pin-Yu Chen, Payel Das, Igor Melnyk, Prasanna Sattigeri, and Rongjie Lai. Optimizing mode connectivity via neuron alignment, 2020. URL https://arxiv.org/abs/2009. 02439.

Neha Verma and Maha Elbayad. Merging text transformer models from different initializations, 2024. URL https://arxiv.org/abs/2403.00986.

Zhanpeng Zhou, Yongyi Yang, Xiaojiang Yang, Junchi Yan, and Wei Hu. Going beyond linear mode connectivity: The layerwise linear feature connectivity. Advances in Neural Information Processing Systems, 36, 2024.

| Component              | ViT              | GPT-2            |                 |                 |
|------------------------|------------------|------------------|-----------------|-----------------|
| CIFAR-10/100           | Tiny ImageNet    | Tiny Shakespeare | BookCorpus      |                 |
| Transformer layers     | 6                | 8                | 6               | 6               |
| Attention heads        | 8                | 8                | 4               | 8               |
| Embedding dimension    | 256              | 384              | 256             | 512             |
| MLP hidden dimension   | 512              | 768              | 1024            | 2048            |
| Patch size             | 4 × 4            | 8 × 8            | -               | -               |
| Sequence length        | -                | -                | 256             | 512             |
| Training epochs        | 150              | 150              | 100             | 5               |
| Batch size             | 128              | 128              | 32              | 64              |
| Optimizer              | AdamW            | AdamW            | AdamW           | AdamW           |
| Learning rate          | 3 × 10−4         | 3 × 10−4         | 3 × 10−4        | 2.5 × 10−4      |
| Weight decay           | 1 × 10−3         | 0.05             | 0.01            | 0.01            |
| Learning rate schedule | Cosine annealing | Cosine (warmup)  | Cosine (warmup) | Cosine (warmup) |
| Hardware               | 1× RTX 2060      | 1x RTX 4090      | 1× RTX 4090     | 4× RTX 4090     |

## A Limitations

While the research introduces a unified framework for symmetry-aware model alignment in Transformers, it presents some limitations and areas for future exploration. Our methodology introduces a generalized notion of LMC that, in some cases, requires reparameterization (e.g., RMSNorm reparameterization or multiplying out intra-head dependencies). Although it preserves functional equivalence, it alters the underlying network architecture. The empirical results for language models were based on a smaller version of GPT-2 language models with reduced parameters due to resource constraints (Appendix B), indicating the need for evaluation on larger, more contemporary language models. The current scope is focused on aligning pairs of models that have been pretrained on the same task, and further investigation could extend these methodologies to models trained on (partially) different tasks [Stoica et al., 2024]. Additionally, while additional results (Appendix D) demonstrate that soft permutations can improve test-time performance of the merged model, more work is needed to fully refine soft relaxations of symmetry operations to improve the performance of aligned models. Finally, due to computational constraints, the study utilizes standard benchmarks such as CIFAR-10/100 and Tiny ImageNet for Vision Transformers and TinyShakespeare and Book- Corpus for GPT-2; exploring performance on a broader or more complex range of benchmarks could further validate the findings.

## B Experimental Details

We provide implementation and training details for the Vision Transformer (ViT) and GPT-2 models used in our experiments. Table 3 summarizes the key architectural parameters, optimization settings, and hardware configurations for both models. Additional details on data preprocessing, augmentation, and training protocols are provided in the following subsections.

## B.1 Vision Transformer (Vit)

We trained two Vision Transformer (ViT) configurations, one for CIFAR-10/100 and another for Tiny ImageNet. For CIFAR-10/100, the model consisted of 6 transformer layers and 8 attention heads, with an embedding dimension of 256 and a feedforward (MLP) hidden dimension of 512. Input images were divided into non-overlapping patches of size 4 × 4. For Tiny ImageNet, the model was scaled up to 8 transformer layers with the same number of attention heads (8). The embedding and hidden dimensions were increased to 384 and 768, respectively, and the patch size was enlarged to 8 × 8 to accommodate higher-resolution inputs.

All ViT models were trained for 150 epochs using the AdamW optimizer with an initial learning rate of 3 × 10−4. For CIFAR-10/100, we applied a weight decay of 1 × 10−3, while for Tiny ImageNet we used 0.05. Both configurations used cosine learning rate scheduling; the Tiny ImageNet setup additionally employed a short warmup phase at the start of training. For CIFAR-10/100, standard data augmentation was applied, including random cropping with padding (crop size 32, padding 4), horizontal flipping, and color jittering (brightness, contrast, and saturation adjustments of 0.4, and hue variation of 0.1). For Tiny ImageNet, a more advanced augmentation pipeline was used, consisting of random resized cropping to 64 × 64 (scale range
(0.8, 1.0), aspect ratio (0.9, 1.1)), random horizontal flipping with p = 0.5, and the AutoAugment policy for ImageNet. All images were normalized using the dataset-specific mean and standard deviation. Additionally, for Tiny ImageNet, we applied post-hoc temperature scaling to rescale the model logits and improve confidence calibration. Training for the CIFAR-10/100 model was performed on a single NVIDIA GeForce RTX 2060 GPU, while the Tiny ImageNet model was trained on a single NVIDIA RTX 4090 GPU. On the Tiny ImageNet test dataset, the models achieve an accuracy of 44.19 ± 0.17 and a calibrated loss of 2.54 ± 0.02. For the CIFAR-10 test dataset, they obtain an accuracy of 83.81 ± 0.44 and a loss of 0.57 ± 0.01.

## B.2 Gpt-2

Two small-scale GPT-2 models were trained: one on the Tiny Shakespeare corpus and another on the BookCorpus dataset. For Tiny Shakespeare, the model consisted of 6 transformer layers with 4 attention heads, an embedding dimension of 256, and an MLP hidden dimension of 1024. Sequences were truncated or padded to 256 tokens. Training was performed for 100 epochs with a batch size of 32 using the AdamW
optimizer. The learning rate was set to 3 × 10−4 with a cosine learning rate schedule and warmup phase, and a weight decay of 0.01. Additionally, early stopping was performed with a patience of 5 epochs to prevent overfitting. Training was conducted on a single NVIDIA RTX 4090 GPU. The model achieves a test loss of 5.28 ± 0.00. For BookCorpus, the model used 6 transformer layers and 8 attention heads, with an embedding dimension of 512 and an MLP hidden dimension of 2048. Tokenized sequences were limited to 512 tokens using the GPT-2 tokenizer, with end-of-sequence tokens used for padding. The model was trained for 5 epochs using the AdamW optimizer with an initial learning rate of 2.5 × 10−4, a 5%
warmup ratio, and a weight decay of 0.01. Mixed-precision (fp16) training was enabled to improve throughput. This model was trained across four NVIDIA RTX 4090 GPUs with an effective batch size of 64 (16 per device). On this dataset, the models obtain a loss of 3.55 ± 0.01 on the test split.

## B.3 Merging

To merge the ViT and GPT-2 models, we use the same setup as for training the individual models.

## C Ablation Study C.1 Learned Matching C.1.1 Coefficient Sampling

In Section 5, we already ablated the effect of using learned permutations in place of orthogonal maps, highlighting the importance of capturing more general alignment symmetries. Here, we further investigate how the choice of interpolation coefficient sampling strategy influences performance (see Line 5 in Algorithm 1). Specifically, we compare four sampling schemes:
- **Fixed interpolation (**λ = 0.5): A deterministic strategy where λ is always set to 0.5, representing a balanced average of the two models.

- **Uniform sampling [**0.4, 0.6]: A narrow uniform distribution centered at 0.5, introducing small random perturbations around equal weighting.

Gaussian sampling N(0:5; 0:1)
Uniform sampling [0:4; 0:6] Uniform sampling [0:0; 1:0] Fixed interpolation (® = 0:5)
Model A ¼(Model B)
Interpolation coefficient (¸)
0.50 0.55 0.60 0.65 L
os s 0 4 8 12 16 Iteration 0.30 0.60 0.90 1.20 1.50 1.80 Los s barri er Orthogonal matching Permutation matching Train Loss (WM init)
Test Loss (WM init)
Train Loss (Identity init) Test Loss (Identity init)
1 5 10 15 Epoch 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 L
o s s
 

- **Uniform sampling [**0.0, 1.0]: A broad uniform distribution over the entire interpolation range, allowing any convex combination of the two models.

- **Gaussian sampling** N (0.5, 0.1): A stochastic strategy that samples from a Gaussian centered at 0.5 with standard deviation 0.1, clipped to the interval [0, 1].

We visualize the impact of these sampling strategies in Figure 6, where we report the mean and standard deviation of loss values along the interpolation path for ViTs trained on CIFAR-10. Notably, both uniform and Gaussian sampling result in relatively high loss barriers, indicating unstable interpolations, particularly around λ = 0.5. In contrast, narrow uniform sampling and fixed interpolation produce lower loss near the midpoint. However, fixed interpolation exhibits significantly higher loss in the surrounding regions, leading to elevated barriers for certain random seeds and greater variance overall. Based on these observations, we adopt narrow uniform sampling in our implementation, as it offers more consistent performance across different random seeds.

## C.1.2 Initialization

The results reported in Section 5 use *weight matching* to initialize the function-preserving permutation and orthogonal transformations. In this section, we evaluate the effectiveness of weight matching as an initialization strategy by comparing it to a baseline with no prior matching.

0.60 0.80 1.00 1.20 Hard Permutations Soft Permutations (¸ » Unif(0:0; 1:0)) Soft Permutations (¸ » Unif(0:4; 0:6))
L
o ss
 

Model A ¼(Model B)
Interpolation coefficient (¸)
We train the learned matching procedure for 15 epochs and report both training and test losses for ViTs trained on CIFAR-10 across four random seeds (see Figure 8). Initializing with weight matching leads to significantly lower training and test losses. Notably, models reach zero test loss barriers within just one to two epochs of training. In contrast, training without any prior matching results in consistently higher losses, with positive loss barriers remaining even after 15 epochs. Evidently, learning the transformations alone is insufficient; a strong initialization provided by weight matching is essential for achieving low loss barriers and faster convergence.

## C.2 Weight Matching

Our proposed iterative weight matching algorithm, while not outperforming learned matching, requires significantly fewer computational resources and has the added advantage of being completely data-free. Nevertheless, several questions remain. In particular: (1) Does orthogonal matching provide improvements over permutation matching, as observed in the learned variant (see Table 2)? (2) How many iterations are needed for convergence? To address these questions, we evaluate the loss barrier across different numbers of iterations for both permutation and orthogonal matching. Results for ViTs trained on CIFAR-10 are shown in Figure 7. A stark contrast emerges: for orthogonal matching, the loss barrier steadily decreases, converging after five iterations with a substantially reduced barrier. In contrast, permutation-based matching shows no significant improvement across iterations. These findings confirm that orthogonal matching provides a more effective path to convergence than permutation matching in the data-free setting, reinforcing its role in our proposed algorithm.

## D Soft-Permutations

Soft permutations provide a continuous relaxation of hard (i.e., exact) permutations by allowing convex combinations of layer units. Formally, we define them as doubly stochastic matrices (i.e., matrices with non-negative entries whose rows and columns each sum to one) lying within the Birkhoff polytope, whose vertices correspond to the set of hard permutation matrices. This relaxation enables mappings that go beyond strict one-to-one neuron correspondences, offering greater flexibility in aligning network representations. We show results in Figure 9. This section details the methodology for learning such soft permutation matrices to align the parameters of two models, θA and θB. Unlike the hard permutations in Algorithm 1 which use a Straight-Through Estimator (STE), here soft permutations are derived from learnable latent parameters and made doubly stochastic using differentiable Sinkhorn normalization. The latter approach performed better than STE in our experiments.

## D.1 Objective

The primary goal is to learn a set of layer-wise latent matrices {Zl}l. These latent matrices are transformed into soft permutation matrices {Pl}l (where Pl = Sinkhorn(exp(Zl))) which are then used to construct a transformation π. This transformation aligns one model to another, e.g., yielding θ aligned B = π(θB; {Pl}). The optimal latent matrices {Z∗
l
} are those that minimize the empirical risk
(loss) of an interpolated model over a batch B drawn from the dataset D:

$$\{\mathbf{Z}_{i}^{*}\}=\operatorname*{arg\,min}_{\{\mathbf{Z}_{i}\}}\frac{1}{|B|}\sum_{(\mathbf{X},\mathbf{Y})\in B}[\mathcal{L}_{\mathrm{CE}}\left(\lambda\mathbf{\theta}_{A}+\left(1-\lambda\right)\pi(\mathbf{\theta}_{B};\left\{\mathrm{Sinkhorn}(\exp(\mathbf{Z}_{i}))\right\})\right)(\mathbf{X},\mathbf{Y})]$$

where λ is an interpolation coefficient, typically sampled uniformly at random (e.g., λ ∼ U[0.4, 0.6]
as in Algorithm 1 or λ ∼ U[0, 1]). The optimization is performed with respect to latent parameters Zl.

## D.2 Parametrization And Initialization Of Latent Matrices

Parametrization from latent matrices. The soft permutation matrices Pl (which must be doubly stochastic) are not optimized directly. Instead, we optimize underlying unconstrained latent matrices Zl. To obtain Pl from Zl:
1. First, a non-negative matrix Pelis generated using an element-wise exponential map:
Pel = exp(Zl)
This ensures all entries are positive, a requirement for the Sinkhorn algorithm, and allows unconstrained optimization of Zl as gradients can flow back through the exp function.

2. Second, Pelis projected onto the Birkhoff polytope B using the differentiable Sinkhorn-
Knopp normalization (detailed below in the learning process) to yield the doubly stochastic soft permutation matrix Pl.

Thus, Pl = Sinkhorn(exp(Zl)), and Zl are the parameters learned via gradient descent. Initialization of latent matrices Zl. The latent matrices Zl are initialized by first constructing a target matrix P
0 l
, which represents a desired initial state for exp(Z0 l
)—before Sinkhorn normalization. A baseline for random noise is established using a Xavier-scheme variance: let σ 2 =2 fan-in+fan-out .

A scaling factor for noise is defined as a = ε σ√3, where ε is a coefficient tuning the amount of noise to be injected. A random noise matrix P
rand lis then sampled, with entries, for example,
-P
rand lij ∼ Uniform(0, 2a). Note one can add a small positive constant to P
0 lto ensure all entries are strictly positive. The target matrix P
0 l is constructed based on one of the following strategies:
- **Random initialization:** The target matrix is formed directly from the random noise:
P
0 l = P
rand l
- **From pre-computed permutation:** If an initial hard permutation matrix P
init lis available
(e.g., from weight matching, as used for initializing ZFF, ZH in Algorithm 1), the target matrix is formed by perturbing this known permutation:

$$P_{l}^{\mathrm{c}}=$$

0 l = P
init l + P
rand l The initial latent parameters Z0 lare then set by inverting the exponential parametrization, i.e.,
Z0 l = log(P
0 l
), applied element-wise to the strictly positive target matrix P
0 l
. This ensures that exp(Z0 l
) = P
0 l at the start of the learning process.

## D.3 Learning Process

The latent matrices Zl for all relevant layers are optimized over Niter iterations. In each iteration t, using the Adam optimizer with learning rate η:

## 1. **Soft Permutation Matrix Computation:**

For each layer l:
(a) Obtain the non-negative matrix from the current latent parameters: Pel = exp(Zl).

(b) Project Pel onto the Birkhoff polytope using K iterations of the Sinkhorn-Knopp algorithm to get the soft permutation matrix Pl. Let Q
(0)
l = Pel. For k = 1*, . . . , K*:

$$\begin{array}{l}{{Q_{l}^{(k)}\leftarrow\mathrm{diag}\left(\frac{1}{Q_{l}^{(k-1)}{\bf1}}\right)Q_{l}^{(k-1)}}}\\ {{{}}}\\ {{Q_{l}^{(k)}\leftarrow Q_{l}^{(k)}\,\mathrm{diag}\left(\frac{1}{{\bf1}^{\top}{\cal Q}_{l}^{(k)}}\right)}}\end{array}$$
l(Normalize rows)
$$({\mathrm{Normalize~rows}})$$

(Normalize $\small\text{columns}$). 
The resulting soft permutation is Pl = Q
(K)
l. This Sinkhorn normalization process is differentiable with respect to Pel.

## 2. **Model Alignment And Interpolation:**

Align model θB using the computed soft permutations {Pl}l: θ aligned B ← π(θB; {Pl}l).

Sample an interpolation coefficient λ (e.g., λ ∼ U[0.4, 0.6] as in Algorithm 1, or from U[0, 1] based on desired outcomes, see discussion below). Form the interpolated model:
θINTERP ← λθA + (1 − λ)θ aligned B .

## 3. **Loss Computation And Parameter Update:**

Compute the empirical cross-entropy loss J on a sampled batch B = {(Xi,Yi)}
|B| i=1 from dataset D:

$${\mathcal{I}}\leftarrow{\frac{1}{|B|}}\sum_{(X,Y)\in B}{\mathcal{L}}_{\mathrm{CE}}(\theta_{\mathrm{INTERP}};X,Y)$$  I can't be seen as follows. 
Compute gradients of the loss with respect to the latent parameters: (. . . , ∇ZlJ *, . . .*). Update each latent matrix Zl using the Adam optimizer:
Zl ← Adam(Zl, ∇ZlJ , η)
Since the exponential mapping and Sinkhorn normalization are differentiable, gradients flow directly back to the latent parameters Zl.

After Niter training iterations, the final optimized latent matrices {Z∗
l
} are used to yield the learned soft permutation matrices {P
∗
l 
= Sinkhorn(exp(Z∗
l
))}.

## D.4 Remarks.

In our explorations, these learned soft permutations are applied to align components within Transformer architectures, specifically targeting attention heads and MLP layers between independently trained models. Empirically, this approach often yields meaningful improvements in the test-time performance of the merged (interpolated) model, particularly at the midpoint of the interpolation path (λ ≈ 0.5). However, a key issue of using soft permutations is that exact functional equivalence at the aligned endpoint (i.e., for θ aligned B compared to the original θB) is generally not maintained. We observe that the degree of functional equivalence at the endpoint can be improved by modifying the sampling strategy for the interpolation coefficient λ during the learning process—for instance, by sampling λ uniformly from the entire [0, 1] range, which allows for more direct optimization of the transformation of the aligned endpoint. Nevertheless, this adjustment typically involves a trade-off: while endpoint functional equivalence may improve, the performance gain observed at the midpoint of the interpolation path might be reduced compared to when λ is sampled more narrowly (e.g., from U[0.4, 0.6]). We leave further exploration of this approach to future work.

## E General Symmetries Of Network Components

Consider a feed-forward network with L layers, where the l-th layer computes activation vector al = σl(Wlal−1), with a0 = X being the input. The full parameter set is θ = {Wl}
L
l=1. Such a network implements the following composed mapping:

$Y=W_{L}\circ\sigma_{L}\circ W_{L-1}\circ\cdot\cdot\circ W_{1}X$
with:

$$\mathbf{\theta}={\left(\begin{array}{l l l l}{W_{1}}&{0}&{\cdots}&{0}\\ {0}&{W_{2}}&{\ddots}&{\vdots}\\ {\vdots}&{\ddots}&{\ddots}&{0}\\ {0}&{\cdots}&{0}&{W_{L}}\end{array}\right)}$$

We define transformation π as the *alignment* that reparametrizes the network to account for its inherent symmetries, mapping the original parameters θ to an equivalent (or approximately equivalent)
set θ
′ = {W′
l
}
L
l=1, with the goal of achieving *linear mode connectivity*—i.e. maintaining low loss barrier along the interpolation path. The reparameterization imposed by π is typically achieved in practice by defining a set of (approximately) invertible matrices {Sl}
L
l=0, where Sl ∈ R
dl×dl acts as a change of basis for the dl-dimensional hidden representation al. We fix the input and output bases by setting S0 = Id0and SL = IdL. The transformed weights are then given by:

$$W_{l}^{\prime}={\bf S}_{l}W_{l}{\bf S}_{l-1}^{-1}\quad\mathrm{for}\;l\in\{1,\ldots,L\}.$$
The aligned network function becomes:
$$f[\theta^{\prime}](\mathbf{X})$$

′](X) = σL(WLS
−1
σL−1(SL−1WL−1S
$${\mathcal{D}}L-1^{0}$$
$$L-1\left({\mathcal{L}}\right)$$
−1
L−2
. . . σ1(S1W1X)*. . .*))
Exact functional invariance, f[θ
′](X) = f[θ](X), is guaranteed if SL = I and the activation functions σl are equivariant with respect to their corresponding transformations Sl, i.e., Slσl(Z) = σl(SlZ) for l ∈ {1*, . . . , L* − 1}. A common case ensuring such equivariance is when σlis an element-wise activation function (e.g., RELU)—as seen in Section 3.1)—and Slis a *permutation matrix* Pl. In this scenario, the set of original parameters θ can be represented as a block diagonal matrix θdiag = diag(W1,W2*, . . . ,*WL).

The transformation θ
′diag = π(θdiag) with blocks W′
l = PlWlP
T
l−1 can be expressed by defining block diagonal transformation matrices:

$$\mathbf{P}_{\text{left}}=\begin{pmatrix}\mathbf{P}_{1}&0&\cdots&0\\ 0&\mathbf{P}_{2}&\ddots&\vdots\\ \vdots&\ddots&\ddots&0\\ 0&\cdots&0&\mathbf{I}_{L}\end{pmatrix},\qquad\mathbf{P}_{\text{right}}=\begin{pmatrix}\mathbf{I}_{0}&0&\cdots&0\\ 0&\mathbf{P}_{1}^{\top}&\ddots&\vdots\\ \vdots&\ddots&\ddots&0\\ 0&\cdots&0&\mathbf{P}_{L-1}^{\top}\end{pmatrix}.$$
$\sigma_1(\mathfrak{g})$
$$r_{1}W_{1}X)\ldots))$$

Then, the transformed parameters are obtained by block-wise operations: (θ
′
diag)ll =
(Pleft)ll(θdiag)ll(Pright)ll. This results in W′1 = P1W1, W′
l = PlWlP
T
l−1for l ∈ {2*, . . . , L* − 1},
and W′L = ILWLP
TL−1 = WLP
T
L−1.

It is important to note that the transformation matrices Sl are not fundamentally restricted to permutations. Many neural network components possess inherent symmetries richer than permutation symmetry, and modern architectures, such as Transformers, are often designed to effectively leverage such components. Recognizing and exploiting broader symmetry classes, when available, expands the set of valid reparameterization mappings. This, in turn, enhances alignment strategies by increasing the likelihood of discovering the low- or zero-barrier interpolation paths, thus allowing to achieve LMC. For instance, Root Mean Square Layer Normalization (RMSNorm) component inherently possesses orthogonal symmetry (see Section 3.2). This property is particularly significant in architectures such as Transformers, where RMSNorm is commonly applied as a standalone operation—typically preceding major components like the attention or MLP blocks, without an immediately following element-wise activation. The absence of such a non-linearity preserves the richer orthogonal symmetry, which would otherwise be reduced to permutation symmetry. We can leverage such an architectural design to exploit the full orthogonal symmetry of RMSNorm for alignment purposes. We now examine the symmetry properties of RMSNorm in more detail. Consider a network block defined as follows:

$$\mathbf{Y}=\mathbf{W}_{1}\,\mathrm{RMSNorm}\big(\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b};\,\beta,\gamma,\epsilon_{N}\big)=\mathbf{W}_{1}\left(\gamma\,\frac{\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b}}{\sqrt{\frac{1}{N}\|\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b}\|_{2}^{2}+\epsilon_{N}}}+\beta\right).$$

Introduce an orthogonal matrix O ∈ RM×N (M ≥ N, O⊤O = I). Since ∥Oz∥2 = ∥z∥2, inserting O⊤O = I gives

$$\mathbf{Y}=\mathbf{W}_{1}\mathbf{O}^{\mathsf{T}}\left(\gamma\,{\frac{\mathbf{O}(\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b})}{\sqrt{{\frac{1}{N}}\|\mathbf{O}(\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b})\|_{2}^{2}+\epsilon_{N}}}}+\mathbf{O}\beta\right).$$

To express the normalization over M elements, define c =
qN
M , ϵM =
N
M ϵN . Then

$$\frac{1}{\sqrt{\frac{1}{N}\|z\|^{2}+\epsilon_{N}}}=c\,\frac{1}{\sqrt{\frac{1}{M}\|z\|^{2}+\epsilon_{M}}},$$

so the block can be rewritten as

$$\mathbf{Y}=\mathbf{W}_{1}\,\gamma\,\mathbf{O}^{\mathsf{T}}c\left({\frac{\mathbf{O}(\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b})}{\sqrt{{\frac{1}{M}}\|\mathbf{O}(\mathbf{W}_{0}\,\mathbf{X}+\mathbf{b})\|_{2}^{2}+\epsilon_{M}}}}+{\frac{1}{c}}{\frac{\mathbf{O}\beta}{\gamma}}\right)\,.$$
$$(6)$$
$$\left(7\right)$$
$$(8)$$

Hence the network remains functionally equivalent when written as

$$\mathbf{Y}=\mathbf{W}_{1}^{\prime}\operatorname{RMSNorm}\left(\mathbf{W}_{0}^{\prime}\,\mathbf{X}+\mathbf{b}^{\prime};\,\beta^{\prime},\gamma^{\prime},\epsilon_{M}\right),$$
, (6)
with transformed parameters

$$W_{1}^{\prime}=W_{1}\,\gamma\,O^{\top}c,\qquad W_{0}^{\prime}=O\,W_{0},\qquad b^{\prime}=O\,b,$$

and normalization constants

$$\gamma^{\prime}={\bf1}_{M},\qquad\beta^{\prime}=\frac{1}{c}\,\frac{{\cal O}\,\beta}{\gamma}=\sqrt{\frac{M}{N}}\,\,\frac{{\cal O}\,\beta}{\gamma}.$$

This derivation for RMSNorm underscores a critical point: the nature of a component's inherent symmetries dictates the set of valid transformation matrices Slthat can be used for reparameterization while preserving its functionality. Let Sl denote the set of allowed transformation matrices for a given layer l of dimension dl. If a layer exclusively admits permutation symmetry, then Sl = Pdl, the finite set of dl × dl permutation matrices. However, if a component, such as RMSNorm blocks in Transformers, exhibit orthogonal symmetry, the set of permissible transformations expands to Sl = O(dl), the orthogonal group. For components allowing even more general transformations, this could be Sl = GL(dl, R), the general linear group of invertible matrices. These sets of transformations are nested, with Pdl ⊂ O(dl) *⊂ GL*(dl, R), where Pdl is a finite group, while O(dl)