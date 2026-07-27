# Transformers Provably Solve Parity Efficiently With Chain Of Thought

Juno Kim1,2∗ Taiji Suzuki1,2 1Department of Mathematical Informatics, University of Tokyo 2Center for Advanced Intelligence Project, RIKEN
∗junokim@berkeley.edu

## Abstract

This work provides the first theoretical analysis of training transformers to solve complex problems by recursively generating intermediate states, analogous to fine-tuning for chain-of-thought (CoT) reasoning. We consider training a one-layer transformer to solve the fundamental k-parity problem, extending the work on RNNs by Wies et al. (2023). We establish three key results: (1) any finite-precision gradient-based algorithm, without intermediate supervision, requires substantial iterations to solve parity with finite samples. (2) In contrast, when intermediate parities are incorporated into the loss function, our model can learn parity in one gradient update when aided by *teacher forcing*, where ground-truth labels of the reasoning chain are provided at each generation step. (3) Even without teacher forcing, where the model must generate CoT chains end-to-end, parity can be learned efficiently if augmented data is employed to internally verify the soundness of intermediate steps. Our findings, supported by numerical experiments, show that task decomposition and stepwise reasoning naturally arise from optimizing transformers with CoT; moreover, self-consistency checking can improve multistep reasoning ability, aligning with empirical studies of CoT.

## 1 Introduction

Large language models (LLMs) based on the transformer architecture (Vaswani et al., 2017) have achieved astounding success across a variety of natural language processing and machine learning tasks (see e.g. Wan et al., 2024; Minaee et al., 2024; Naveed et al., 2024; Zhao et al., 2024). However, they often struggle when tasked with solving complex reasoning problems, especially in a zero-shot setting without any form of intermediate guidance or supervision (Geva et al., 2021; Rae et al., 2022; Arkoudas, 2023; Wang et al., 2024). These failures are particularly evident in tasks requiring multi-hop reasoning or compounded logical steps (Sakarvadia et al., 2024). A promising approach to overcome these limitations is chain-of-thought (CoT) reasoning, where the model is prompted or fine-tuned to solve complex tasks step-by-step by explicitly making intermediate reasoning steps to arrive at the desired answers (Wei et al., 2022; Kojima et al., 2022). Since its discovery, CoT reasoning has been shown to significantly enhance the problem-solving capabilities of LLMs while also increasing the interpretability and trustworthiness of the reasoning process, and has spawned numerous prompting techniques (Liu et al., 2023; Qiao et al., 2023) and applications for a variety of downstream tasks including common-sense reasoning, mathematical problem-solving, and symbolic or multi-modal reasoning; see e.g. Zhang et al. (2023b); Yu et al. (2023); Chu et al. (2024) for surveys on CoT. In particular, besides being used as a prompting method, directly training or fine-tuning models to generate CoT has also been shown to significantly improve multi-step reasoning performance (Nye et al., 2021; Wei et al., 2022; Zelikman et al., 2022; Lightman et al., 2024).

Despite these empirical successes, however, the theoretical understanding of the CoT mechanism and task decomposition in transformers is still limited. Existing works focus on characterizing the expressivity of transformers equipped with CoT, providing constructions which can solve certain complexity classes (Feng et al., 2023; Merrill & Sabharwal, 2023; 2024; Li et al., 2024b), studying the class of functions that can be learned in-context with CoT (Li et al., 2023; Bhattamishra et al., 2024), or analyzing the estimation error of multi-step models (Hu et al., 2024). Nevertheless, such 1 approaches do not indicate how such capabilities might emerge when training transformers to generate reasoning chains. Li et al. (2024a) analyze the training dynamics of a one-layer transformer in an in-context learning setting and show that CoT ability may be acquired; however, they do not consider explicitly training with CoT chains, which is a more difficult problem since the objective depends on the recursive application of the transformer to itself. In this paper, we seek to formalize the mechanism through which stepwise reasoning emerges in transformers optimized to generate CoT chains. We focus on the specific problem of *bit subset parity* (learning the parity of an unknown subset of k bits from a d-bit input), which is known to be impossible to learn end-to-end with any finite-precision gradient-based algorithm in polynomial steps (Shalev- Shwartz et al., 2017; Shamir, 2018). In contrast, Wies et al. (2023) have demonstrated that recurrent neural networks (RNNs) can solve parity efficiently when provided with intermediate supervision. We build on this direction to establish positive optimization guarantees for the transformer architecture. Our object of study is a one-layer transformer incorporating a softmax attention layer, feedforward layer and positional encoding, that is recursively applied to its own output to generate a sequence of intermediate parity computations to arrive at the desired output, analogous to CoT generation. Our contributions are summarized as follows.

- We extend the impossibility result for parity (Theorem 1), which was established only for population gradient descent, to the more realistic finite-sample setting in Theorem 2. We prove that any iterative algorithm with access to an approximate gradient oracle for the end-to-end empirical loss cannot solve a random target parity within a specific polynomial number of steps.

- In contrast, we show that when the loss is summed over all intermediate states, by utilizing teacher forcing, a form of process supervision wherein ground-truth intermediate steps are provided during training,1 our model can learn any parity in a single gradient update (Theorem 5).

This shows the benefits of training directly with CoT chains to acquire task decomposition ability.

- We further consider training with CoT generated end-to-end without teacher forcing,2and show that parity can still be learned in a logarithmic number of steps if augmented data is employed to check the validity of intermediate steps (Theorem 7), thereby mimicking self-consistency checks often used in CoT reasoning (Zelikman et al., 2022; Wang et al., 2023; Huang et al., 2023a).

- We conduct numerical experiments supporting our findings (Section 4 and Appendix D).

Our results provide theoretical insights into how transformers can naturally and efficiently optimize to perform task decomposition, emphasizing the role of explicit intermediate supervision for complex tasks. Moreover, these findings corroborate recent empirical studies on CoT reasoning demonstrating improved performance through process supervision and internal validation of reasoning chains (Huang et al., 2023a; Tian et al., 2024; Lightman et al., 2024).

## 1.1 Related Works

Complexity of transformers. A line of work aims to understand the effectiveness of CoT from the perspective of complexity theory. Feng et al. (2023) show that autoregressive transformers of constant size can solve basic arithmetic tasks by recursively generating CoT reasoning steps, which is not possible when directly generating the solution; this separation arises because looping the generated outputs back to its inputs increases the 'effective depth' of the model. Works such as Chiang et al. (2023); Merrill & Sabharwal (2023) study the expressivity of fixed-precision transformer architectures in terms of classes of formal languages. Merrill & Sabharwal (2024); Li et al. (2024b) show that CoT reasoning enables recognizing wider language classes, and characterizes the increased expressivity depending on the length of the reasoning chain. Sanford et al. (2024) studies the relation between transformers and massively parallel computation protocols, showing that logarithmic depth suffices to solve multi-hop induction tasks that cannot be efficiently solved by other sequence models.

1Teacher forcing or process supervision is a training procedure for recurrent models in which the model receives the ground truth output at time t as input at time t + 1 during training (Goodfellow et al., 2016, p.377). Many fine-tuning methods with ground-truth CoT chains implement teacher forcing, being more effective than output supervision with chains generated end-to-end (Deng et al., 2023; Tian et al., 2024; Lightman et al., 2024).

2Teacher forcing can induce exposure bias where a model is not robust to its own errors. In practice, partial
(scheduled or random) teacher forcing methods are used to overcome this issue (Bengio et al., 2015; Goyal et al., 2017; Mihaylova & Martins, 2019).

Additionally, Li et al. (2023); Bhattamishra et al. (2024) study the class of functions that can be learned in context by transformers with CoT from the point of view of in-context learning.

Optimization and generalization of CoT. Zhu et al. (2024) study the 'reversal curse' via the training dynamics of a one-layer transformer and shows that the model fails to generalize from A → B, B → C to A → C as an argument for the necessity of explicit step-by-step reasoning. Hu et al. (2024) study CoT prompting from a statistical estimation perspective by introducing a multi-step latent variable model for CoT and analyzing its approximation, generalization and prompting-based errors. Notably, Li et al. (2024a) study the training dynamics of a one-layer attention-only transformer model in an in-context learning setting and show that CoT generalization capability can be obtained. However, this does not address the possibility or benefits of training with CoT chains. Lightman et al. (2024) empirically study training LLMs with either process or outcome supervision, showing that the former significantly outperforms the latter when training to solve challenging reasoning tasks.

Parity and task decomposition. The difficulty of learning parity without task decomposition is established in Shalev-Shwartz et al. (2017); Shamir (2018). The work most relevant to our paper is Wies et al. (2023), which study task decomposition for parity with classical Elman RNNs. They show that by incorporating intermediate states into the loss function and utilizing teacher forcing, parity can be solved with polynomial iterations and embedding size. Our Theorem 5 extends this positive result to autoregressive transformers, rigorously establishing the benefits of CoT-based training.

## 2 Problem Setup

Notation. We write [n] := {1, 2, · · · , n} for any integer n. Scalar operations apply componentwise to vectors, e.g. for z ∈ R
n we write ϕ(z) = (ϕ(z1), · · · , ϕ(zn))⊤, z 2 = z ⊙ z = (z 21, · · · , z2n) and |z| = (|z1|, *· · ·* , |zn|)
⊤. The 2-norm is always denoted by ∥·∥. The multi-linear inner product or contraction of z1, *· · ·* , zr ∈ R
nfor any r ∈ N is denoted as ⟨z1, *· · ·* , zr⟩ := Pn i=1 z1,i · · · zr,i. In particular, ⟨z1⟩ = z
⊤
1 1n and ⟨z1, z2⟩ = z
⊤
1 z2.

## 2.1 The Parity Problem

Let d ≥ k ≥ 2 be integers and let P denote the set of size k subsets of {1, · · · , d} equipped with the uniform distribution. In this paper, we study the k-parity problem for d-bit inputs x = (xj )
d j=1 ∼
Unif({±1}
d), where the output y =Qj∈p xj is determined by the parity of an unknown subset of bits p ∈ P. We abuse notation and identify the set of indices p with the corresponding parity mapping x 7→Qj∈p xj . Given n samples (x i, yi)i∈[n], our goal is to predict the parity of any test input.

It is known that parity is fundamentally difficult in the sense that it cannot be solved in polynomial time by any finite-precision gradient-based algorithm, such as neural networks. More precisely, let {fθ | θ ∈ Θ} be any differentiable (w.r.t. θ) parametrized model with polynomially bounded gradients, ∥∇fθ(x)∥ = O(poly(d)), and define the population loss L¯ = Ex-(y − fθ(x))2. We presume access to an ε-*approximate gradient oracle* ∇e for L, which takes any θ ∈ Θ as query and returns a vector ∇eL¯(θ) satisfying ∥∇eL¯(θ) − ∇L¯(θ)∥2 ≤ ε, potentially in an adversarial manner.

Then the following holds:
Theorem 1 (Wies et al. (2023), Theorem 4). Let ℓ0−1 *be the zero-one loss. There exists an* O(e
−d/3)-
approximate oracle ∇e such that3the output θ(A) of any iterative algorithm A which sequentially makes at most O(poly(d)) queries to ∇eL¯ *must satisfy* Ex-ℓ0−1(p(x), fθ(A)(x))≥
1 2 
− O(e
−d)
with probability at least 1 − O(e
−d/3), when the target parity p *is uniformly sampled from* P.

x23 = y h = 3 x21 x22 h = 2 x17 x18 x19 x20 h = 1 h = 0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16
each gradient call ∇L¯(θ) with respect to the target parity p is exponentially small (Shalev-Shwartz et al., 2017) and is drowned out by the noise from the adversarial oracle, so that no information can be gained on the target without exponentially many queries. See Section 3.1 for more details.

Task decomposition. As in Wies et al. (2023), we assume k = 2vfor an integer v for simplicity and decompose the problem into a hierarchy of 2-parity computations which can be efficiently learned in a sequential manner by our model. This is expressed as a complete binary tree T of height v and 2k−1 nodes. The lowest level contains k nodes representing the bits xjm for m ∈ [k]. The remaining nodes are labeled xd+1, · · · , xd+k−1 starting from the next lowest level and moving upwards, left to right. The largest index in level ℓ for 0 ≤ ℓ ≤ v is denoted as dℓ = d +Pℓj=1 2 v−j, d0 = d.

Also, for each *m > d*, the indices of the two child nodes of xm are denoted as c1[m], c2[m] where 1 ≤ c1[m] < c2[m] < m. In addition, the parent node index of xm is denoted as p[m] and the level or height of xm is denoted as h[m], so that dh[m]−1 < m ≤ dh[m].

e1 em e1 e2 ed ed+1 em
↓ ↓ ↓ ↓ ↓
x1 · · · *· · ·*
xˆm x i x1 x2
· · ·
xd xˆd+1
· · ·
xˆm
· · ·
xˆd+k−1 = yˆ
u1 uˆm
(a) Recursive generation of intermediate states.

ι ◦ ϕ ϕ
Figure 2: Illustration of the recursive data generation process by the transformer model. (a) Each token consists of a one-hot positional encoding ej and parity data xj . The d input tokens (blue)
are fixed. The token xˆm is generated at the (m − d)th step by computing attention scores based on position, combining the previous tokens and applying the feedforward layer ϕ. xˆd+k−1 is returned as the model prediction. (b) For the no teacher forcing setup in Section 3.3, data augmentation uj is implemented to check for self-consistency. If the augmented outputs from the previous generation
(red) are uninformative, a filter ι is applied to zero out the subsequent output.

We study a one-layer transformer architecture employing absolute positional encoding and a singlehead softmax attention layer followed by a shallow feedforward layer; skip connections are omitted for simplicity. See Figure 2 for a visualization of our setup.

Data encoding: Each input token xj = (x ij)
n i=1 for j ∈ [d] is the n-dimensional vector consisting of the jth bit of each sample x i. We also add dummy tokens xd+1, *· · ·* , xd+k−1 initially set to 0n, which will learn to sequentially generate the actual intermediate nodes. Each xj is concatenated with the one-hot positional encoding ej ∈ R
d+k−1for j ∈ [d + k − 1] to form the internal input pj = (x
⊤
j e
⊤ j
)
⊤ ∈ R
n+d+k−1to the attention layer.

Softmax attention layer: The attention layer is defined as in (1) in terms of key, query and value matrices K, Q, V. We fix the first n columns of K, Q to zero so that the attention scores are determined by only the positional encodings. This ensures that the transformer focuses on learning which positions contribute to the parity at each step. K, Q are then reparametrized by a single matrix W ∈ R
(d+k−1)2; conversely, the value matrix is set to only preserve the x component, as follows.

$$\mathbf{K}^{\top}\mathbf{Q}=\begin{pmatrix}\mathbf{0}_{n\times n}&\mathbf{0}_{n\times(d+k-1)}\\ \mathbf{0}_{(d+k-1)\times n}&\mathbf{W}\end{pmatrix},\quad\mathbf{V}=\begin{pmatrix}\mathbf{I}_{n\times n}&\mathbf{0}_{n\times(d+k-1)}\end{pmatrix}.$$  In a construction, we can write the literature to make a special case.  
This type of reparametrization is common in the literature to make dynamical analysis tractable (Zhang et al., 2023a; Huang et al., 2023b; Mahankali et al., 2023; Kim & Suzuki, 2024).

Feedforward layer: The feedforward layer realizes a fixed link function ϕ : [−1, 1] → [−1, 1],
applied elementwise and only to the xj component; the positional encodings are not affected. To exploit the decomposition of our task into 2-parities, we choose ϕ such that ϕ(0) = −1, ϕ(±1) = 1 so that sums are converted into parities, i.e. ϕ( a+b 2
) = ab for a, b *∈ {±*1}. Moreover, we require that ϕ
′(0) = ϕ
′(±1) = 0 and assume ϕ is symmetric and sufficiently regular, so that we may expand ϕ(t) = −1 + ct2 + O(|t| 4) and ϕ
′(t) = 2ct + O(|t| 3).

The transformer computes TF(x1, *· · ·* , xd+k−1;W) = (xˆ1, *· · ·* , xˆd+k−1) where the original data xˆj = xj , j ∈ [d] remain unchanged and tokens xˆd+1, *· · ·* , xˆd+k−1 are computed as

$\mathbf{\tau},\jmath\;\in\;\left|\theta\right|$
$$\hat{\mathbf{x}}_{m}=\phi(\hat{\mathbf{z}}_{m}),\quad\hat{\mathbf{z}}_{m}=\sum_{j=1}^{m-1}\mathbf{V}\hat{\mathbf{p}}_{j}\cdot\text{softmax}(\hat{\mathbf{p}}_{j}^{\top}\mathbf{K}^{\top}\mathbf{Q}\hat{\mathbf{p}}_{m})=\sum_{j=1}^{m-1}\sigma_{j}(\mathbf{w}_{m})\mathbf{x}_{j},$$  for softmax scores $\sigma_{j}(\mathbf{w}_{m})=e^{w_{j,m}/\sum_{j=1}^{m-1}e^{w_{\alpha,m}}}$. Here we have implicitly added 
$$(1)$$
where the softmax scores σj (wm) = e α=1 e wα,m. Here, we have implicitly added the causal mask wj,m *← −∞* to the attention layer for j ≥ m or m ≤ d. Note that each zˆm, xˆm will be contained in the cube [−1, 1]das long as the input tokens are also contained in [−1, 1]d.

Chain of thought. Consider repeatedly applying TF(·) to its own output to generate a 'reasoning chain.' Since the input tokens are fixed, the token xˆd+1 will be updated once and then always yield the same value afterwards. Next, since xˆd+2 depends on the input tokens and xˆd+1, it will be updated twice before becoming fixed. Repeating this, the entire chain stops updating after at most k − 1 steps, yielding the output TF(k−1)(x1, · · · , xd, 0n, *· · ·* , 0n;W) = (xˆ1, *· · ·* , xˆd+k−1)
where the intermediate predictions are recursively computed as xˆm = ϕ(Pm−1 j=1 σj (wm)xˆj ). Finally, the top node is returned as the model prediction yˆ = xˆd+k−1.

This process can be seen as a simplified version of CoT reasoning, albeit not in an in-context learning setting: instead of one-shot predicting y ifrom x i, the model starts by solving simpler subtasks and uses the information to attack compound problems, learning to generate intermediate reasoning steps xd+1 *→ · · · →* xd+k−1 to finally arrive at the desired solution. Importantly, this process is not possible if the model is only trained on the one-shot data (x i, yi)i∈[n] as we show in Section 3.1.

Instead, we incorporate the prediction error for all intermediate states directly into our loss function (Lightman et al., 2024). We also consider shortening the reasoning chain by using a different causal mask in Section 3.3, which will result in improved control of error and faster convergence.

## 3 Main Results 3.1 Hardness Of Parity Without Cot

Before analyzing our transformer model, we first prove a negative learning result in the absence of intermediate supervision that extends Theorem 1, which was stated with respect to the population objective L¯ and zero-one test loss ℓ0−1, to finite samples and mean squared loss.

Let fθ : {±1}
d → R be any differentiable parametrized model and suppose we select the target parity p uniformly at random from P. In the finite-sample setting, n i.i.d. samples (x i, yi)i∈[n] are generated as x i ∼ Unif({±1}
d), y i = p(x i) and we are given access to (approximate) gradients from the empirical loss

$$L_{n}(\theta)={\frac{1}{2n}}\sum_{i=1}^{n}(y^{i}-f_{\theta}(\mathbf{x}^{i}))^{2}={\frac{1}{2}}\|p-f_{\theta}\|_{n}^{2},$$

where ∥·∥n is the empirical norm. It is important that the model fθ is applied to each x iseparately and does not cross-reference between different samples, as there exist more efficient parity-learning algorithms if the data is allowed to be manipulated freely. For example, Gaussian elimination can solve parity with O(d) samples and O(d 3) iterations (Raz, 2018). Moreover, this implies that neural networks trained with stochastic gradient descent can also solve parity in polynomial time (Abbe & Sandon, 2020). Instead, in our setting the model is forced to learn from the averaged gradient signal and can only implicitly utilize the correlation between samples. We show the following result for learning parities with finite-samples in Appendix A:
Theorem 2 (hardness of finite-sample parity). *Suppose* k = Θ(d).

(1) If n = e Ω(d) and fθ *has polynomially bounded gradients, there exists an* e
−Ω(d)-approximate gradient oracle ∇e *such that with probability* 1 − e
−Ω(d) *over random sampling, the output* θ(A)
of any iterative (possibly randomized) algorithm which makes at most O(poly(d)) *queries to*
∇eLn has L2*-loss lower bounded as* Ep∈P,x-(p(x) − fθ(A)(x))2≥ 1 − e
−Ω(d).

(2) If n = Ω(d ν) and ∥∇fθ∥ = O(d ν1 )*, there exists an* O(d
−ν2 )*-approximate gradient oracle* ∇e such that with probability 1 − e
−Ω(d) over random sampling, the output θ(A) *of any iterative*
(possibly randomized) algorithm which makes at most O(d ν3 ) queries to ∇eLn has L2*-loss lower* bounded, where ν = 4ν1 + 4ν2 + 2ν3 + 2ν4 + 1*, as* Ep∈P,x-(p(x) − fθ(A)(x))2≥ 1 − O(d
−ν4).

We remark that the bounds are asymptotically optimal since fθ ≡ 0 is a valid estimator. Moreover, the expectation over p ∈ P can be replaced by the corresponding 'with high probability' statement.

A counter-intuitive aspect of the above result is that parity becomes potentially more difficult when the number of samples increases. Indeed, with exponential samples n = e Ω(d)(1) we basically recover the statement of Theorem 1, while the guarantees for n = poly(d) (2) are also polynomial in d. This is because the difficulty of parity (Theorem 1) fundamentally depends on the following result:
Proposition 3 (Shalev-Shwartz et al. (2017), Theorem 1). Suppose x *be a random variable in* R 
d. Let H *be a class of bounded real-valued functions on* R
d*such that* Ex[h(x)h
′(x)] = 0 for any two distinct h, h′ ∈ H and fθ *a differentiable parametric model with gradients bounded by* Ex[∥∇fθ∥
2] ≤ F(θ)
2*. Then for the loss* Fh(θ) := Ex[(h(x)−fθ(x))2] where h *is chosen uniformly* at random from H*, the gradient variance is bounded as*

$$\operatorname{Var}(\theta;\mathcal{H}):=\mathbb{E}_{h\in\mathcal{H}}\left[\|\nabla F_{h}(\theta)-\mathbb{E}_{h^{\prime}\in H}[\nabla F_{h^{\prime}}(\theta)]\|^{2}\right]\leq{\frac{F(\theta)^{2}}{|\mathcal{H}|}}.$$

Since all dk
= e Θ(d) parities in P are pairwise orthogonal with respect to the uniform distribution Unif({±1}
d), it follows that the variance of ∇L¯ is exponentially small and the target signal can be drowned out by a correspondingly small noise from the oracle. However, this is not true for the empirical distribution which cannot distinguish all elements in P with only poly(d) samples; the empirical correlation of two random parities will generally be Θ(n
−1/2). Therefore a more careful decorrelation argument is needed, resulting in the weaker guarantees of Theorem 2(2). Another technical difference is that Theorem 1 only considers the strong zero-one loss (more formally, their results can be seen to hold for any parity estimator pˆθ(A) ∈ P depending on the algorithm output), while we prove the L2 lower bound for any real-valued estimator fθ(A).

## 3.2 Cot With Teacher Forcing

When training with teacher forcing, at each position d + 1 ≤ m ≤ d + k − 1, the ground-truth labels of the preceding intermediate states x1, *· · ·* , xm−1 are fed into the transformer input to obtain the predictor xˆm at the mth position, xˆm = TF(x1, · · · , xm−1, 0n, *· · ·* , 0n;W)m.

The loss function then computes the squared error over all states,

$$L(\mathbf{W})=\frac{1}{2n}\sum_{m=d+1}^{d+k-1}\|\hat{\boldsymbol{x}}_{m}-\boldsymbol{x}_{m}\|^{2}.\tag{2}$$  Since each sequence of values $\hat{\boldsymbol{x}}_{d+1,i},\cdots,\hat{\boldsymbol{x}}_{d+k-1,i}$ are generated depending only on the correspond 
ing sample x iand the parameter matrix W, this can be rewritten in terms of the augmented labels y¯
i = (x i d+1, · · · , xid+k−1
)
⊤ as

$$L({\bf W})=\frac{1}{2n}\sum_{i=1}^{n}\|\vec{y}^{i}-f^{\circ}(\mathbf{x}^{i};{\bf W})\|^{2},\quad f_{m}^{\circ}(\mathbf{x}^{i};{\bf W})=\hat{x}_{m,i},\quad d+1\leq m\leq d+k-1$$

for a fixed mapping f
◦: {±1}
d × R
(d+k−1)2 → R
k−1, mirroring the setting of Theorem 2. Hence our model does not cross-reference between samples; moreover, the gradient of f
◦is bounded as Lemma 4. For all x,W *it holds uniformly that* ∥∇Wf
◦(x;W)∥ ≤ O(
√d).

At inference time, test inputs x1, *· · ·* , xd are randomly generated and the prediction for ytest =
p(x1, *· · ·* , xd) is computed by iterating TF to generate all k − 1 reasoning steps without reference to ground-truth labels; yˆtest = TF(k−1)(x1, · · · , xd, 0n, *· · ·* , 0n;W)d+k−1. Our positive learning result in this setting is as follows.

Theorem 5 (CoT with teacher forcing). Suppose n *= Ω(*d 2+ϵ) for ϵ > 0, d *is sufficiently large and* let ∇e *be any* O(d
−2−ϵ/8)-approximate gradient oracle.4 *Set initialization* W(0) = 0 *and learning* rate η *= Θ(*d 2+ϵ/16). Then for any target parity p ∈ P*, it holds with probability* 1 − exp(−d ϵ/2)
over random sampling that the one-step update W(1) = W(0) − η∇eL(W(0)) *w.r.t. the objective* (2)
with teacher forcing achieves loss ∥yˆtest − ytest∥∞ ≤ O(d
−ϵ/8).

On the other hand, Theorem 2(2) shows that when n = Ω(d 11+ϵ), any iterative algorithm querying an O(d
−2−ϵ/8)-approximate oracle, with gradients bounded as in Lemma 4, requires more than Ω( 
e d ϵ/4)
queries to attain a nontrivial (<
1 2
) loss. This establishes a strict separation between learning parities without intermediate supervision and our CoT transformer. The gap increases with more samples as ϵ increases; moreover, when n = e Ω(d), we have a much stronger separation by Theorem 2(1), where an exponential number of queries is required to learn p. Sketch of proof. The result is shown by explicitly calculating the gradient with respect to each weight wj,m and extracting the gradient signal. As the softmax scores are uniform at initialization, the gradient can be expanded to obtain multilinear contraction or 'interaction' terms between the tokens x1, *· · ·* , xm−1, one such example being

 In example being  $ \frac{1}{n}\left<\pmb{x}_m,\hat{\pmb{z}}_m,\hat{\pmb{z}}_m\right>=\frac{1}{n(m-1)^2}\sum_{\alpha,\beta}\left<\pmb{x}_m,\pmb{x}_\alpha,\pmb{x}_\beta\right>.$
In the above equation, if *α, β* are the two child nodes of m, the parity xαxβxm ≡ 1 will be trivial and ⟨xm, xα, xβ⟩ = n. On the other hand, for nontrivial parities the interaction strength will generally be O(
√n log d) due to sample concentration. For sufficiently large n, the trivial parities dominate, allowing us to extract the leading term. Performing these computations up to fourth order interaction terms, we show that the dominating signal of the gradient is Θ(d
−2) when j = c1[m], c2[m] and O(d
−2−ϵ/8) otherwise. Hence the transformer learns to increase only the weights at the relevant positions for each subtask, and is able to compute the desired 2-parity xˆm ≈ ϕ( 1 2
(xˆc1[m] + xˆc2[m])) ≈ xˆc1[m]xˆc2[m] at each node during its forward pass. The full proof is provided in Appendix B.

4In fact, we only require that each component of the gradient has error at most O(d
−2−ϵ/8) for Theorems 5, 7, which follows since the L∞ error is bounded above by L2.

## 3.3 Cot Without Teacher Forcing

In this section, we extend Theorem 5 to training a transformer without teacher forcing, which is employed alongside teacher forcing in practice to ensure robustness at test time (Bengio et al., 2015; Goyal et al., 2017; Mihaylova & Martins, 2019). The main difficulty in this setting is that wrong answers propagate to later generation steps, exponentially amplifying errors and drowning out the main gradient signals. Error accumulation is also a central practical issue of CoT (Zhang & Parkes, 2023; Wang et al., 2023). To solve this issue, we make some modifications to our transformer model.

First, we minimize the number of required reasoning steps by imposing a slightly stronger form of autoregressivity where each intermediate state x
+m depends on all tokens x
+ j
, j = 1, · · · , dh[m]−1 up to the previous level, rather than the immediately preceding token. This can be expressed as the causal mask wj,m *← −∞* for *j > d*h[m]−1 or m ≤ d; see Figure 3. This ensures that the model gradients are polynomially bounded as in Theorem 2 and that errors can propagate a logarithmic rather than a linear number of steps, and can be easily implemented as the indices dℓ are known.

m = d d0 d1 d2
Figure 3: Causal mask for W⊤ with teacher forcing (left); without teacher forcing (right). The gray entries are set to −∞.

Second, we implement a data augmentation technique where random d-bit strings u i ∼ Unif({±1}
d),
i ∈ [n
′] are appended to the original dataset (x i)i∈[n]. The resulting augmented tokens are denoted as x
+
j = (x
⊤
j u
⊤ j
)
⊤ ∈ R
n+n
′, uj = (u ij
)
n
′
i=1 so that pj = ((x
+ j
)
⊤ e
⊤ j
)
⊤ (the notation is extended to *j > d*), and the key, query and value matrices are appropriately enlarged. The ground truth labels as well as the intermediate states for the augmented data are unknown, so they are not included in the loss function. Nevertheless, unlabeled data can still suffice for self-consistency (Huang et al., 2023a); their purpose is to filter for 'faulty reasoning' in the following sense. If the weights are not sufficiently trained, the output of a node xj will consist of all nearly −1s and thus be uninformative for computing any parities. If the augmented tokens newly generated in the previous iteration of TF(·) (i.e. up to udℓ−1
) are uninformative, we zero out its output on the basis that all subsequent reasoning will be wrong. This is achieved by adding the following filter after the feedforward layer ϕ:

$\forall\mathbf{z}^{+}\in\mathbb{R}^{n+n^{\prime}},\quad\iota_{\ell}(\mathbf{z}^{+})=\begin{cases}\mathbf{0}&\|\mathbf{u}_{j}+\mathbf{1}_{n^{\prime}}\|_{\infty}<\varepsilon_{0}\;\;\text{for any}\;\;d_{\ell-2}<j\leq d_{\ell-1},\\ \mathbf{z}^{+}&\text{otherwise.}\end{cases}$
Without teacher forcing, during training the entire reasoning chain is generated by iteratively applying TF to its own output until convergence, which takes v = log2 k rather than k − 1 steps due to the imposed block autoregressivity. Hence TF(v)(x
+ 1
, *· · ·* , x
+ d
, 0n+n′ , *· · ·* ;W) = (xˆ
+ 1
, *· · ·* , xˆ
+ d+k−1
)
where the tokens xˆ
+
d+1, *· · ·* , xˆ
+ d+k−1 are recursively generated per level as

$$\hat{\mathbf{x}}_{m}^{+}=\iota_{\mathsf{h}[m]}\circ\phi(\hat{\mathbf{z}}_{m}^{+}),\quad\hat{\mathbf{z}}_{m}^{+}=\sum_{j=1}^{d_{\mathsf{h}[m]-1}}\sigma_{j}(\mathbf{w}_{m})\hat{\mathbf{x}}_{j}^{+}.\tag{3}$$

The loss is computed against the ground-truth labels as in (2). As before, each sequence of generated states depends only on each sample x iand the augmented data U = (u i)i∈[n′], so we may express

$$L({\bf W},{\bf U})=\frac{1}{2n}\sum_{i=1}^{n}\|\bar{p}^{i}-f^{\times}(\mathbf{x}^{i};{\bf W},{\bf U})\|^{2},\quad f_{m}^{\times}(\mathbf{x}^{i};{\bf W},{\bf U})=\hat{x}_{m,i}\tag{4}$$

for a fixed mapping f
×, so that the samples are again not cross-referenced. By considering the propagation of gradients up the chain, the gradient of f
× can be shown to be bounded as follows.

Lemma 6. For all x,W, U *we have* ∥∇Wf
×(x;W, U)∥ ≤ O(d g) *where* g = log2∥ϕ
′∥∞ + 1/2.

The exact exponent g depends on the shape of ϕ. Since ϕ(0) = −1 and ϕ(1) = 1, it must hold that ∥ϕ
′∥∞ > 2. Conversely, any such ∥ϕ
′∥∞ may be achieved by taking ϕ to be locally quadratic around 0, ±1 and smoothly joining the curve segments with straight lines of slope ±(2 + ϵ). Furthermore, such a link function can be realized by a simple shallow feedforward layer using e.g. O(1) ReQU neurons. Hence g can be taken to be arbitrarily close to 1.5. Finally, we implement a simple weight quantization method by rounding each entry of W to the nearest integer after every update; W(t+1) = r[W(t) − η∇eWL(W(t), U)], where r : R → Z is the nearest-integer operator. Equivalently, the gradients themselves are quantized. Integer-based quantization methods are widely used in practice to accelerate training and reduce memory usage (Wu et al., 2020; Jacob et al., 2018), and have been successfully implemented in LLMs to facilitate efficient fine-tuning (Dettmers et al., 2022; 2023). In our theoretical setting, quantization also allows us to simplify computations involving propagation of error. In this setting, we obtain the following learning result. Theorem 7 (CoT without teacher forcing). *Suppose* n = Ω(d 2+ϵ) for ϵ > 0, n
′ = poly(d),
5 d is sufficiently large and let ∇e *be any* O(d
−2−ϵ/8)*-approximate gradient oracle. Set* W(0) = 0 and η = Θ(d 2+ϵ/16). Then for any target parity p ∈ P*, it holds with probability* 1 − exp(−d
(ϵ∧1)/2)
over random sampling of (original and augmented) data that the sequence of updates W(t+1) =
r[W(t) − η∇eL(W(t), U)] *w.r.t. the objective* (4) without teacher forcing achieves exponentially small loss ∥yˆtest − ytest∥∞ ≤ exp(−Ω(d ϵ/16)) in log2 k *iterations.*
This gives the same order of separation from Theorem 2(2) as in Section 3.2. Hence transformers can learn parities even without teacher forcing, if the consistency of the chain of reasoning is suitably controlled for. Moreover, our result shows that logarithmic time suffices to learn parity by exploiting the hierarchical decomposition in Figure 1. This extends the circuit complexity result in Merrill & Sabharwal (2024), which states that bounded-depth transformers with a logarithmic number of CoT steps can express problems in log-space; Theorem 7 guarantees that transformers of depth one can learn by gradient descent any such function in the exponentially large class P.

Sketch of proof. The idea is to inductively show that each 2-parity subtask xm at level ℓ will become solved at time t = ℓ. When t ≤ ℓ − 2, xm cannot utilize its child nodes xc1[m], xc2[m] since they will also not be optimized, so the weights do not change. At time ℓ − 1, its child nodes learn to output their parities with high precision, so the objective is approximately equivalent to that of Theorem 5.

Then the gradient signal will similarly concentrate on wc1[m],m, wc2[m],m and xm will become solved in the next step. It remains to bound the gradients arising from the loss terms further down the chain xd+1 *→ · · · →* xd+k−1 (propagation of error), and verify that irrelevant weights wj,m (p[j] ̸= m)
and already optimized weights do not change. The full proof is provided in Appendix C.

## 4 Numerical Experiments

In this section, we present numerical experiments which support and complement our theoretical findings. Compared to the carefully calibrated step sizes and weight updates in Theorems 5 and 7, these experiments study a more realistic training scenario by taking relatively small learning rates and tracking the loss trajectories over a longer period of training. We train one-layer transformers based on the architecture described in Section 2 to solve a random k-parity problem with 64-bit inputs for k = 8, 16, 32. Specifically, we implement and compare the following four models.

- **Direct:** TF(·) is applied to itself k − 1 times to generate the reasoning chain end-to-end and the model prediction yˆ is directly compared to the ground truth y with the prediction loss 1 2n
∥yˆ−y∥
2.

- **CoT:** TF(·) is applied to itself to generate the reasoning chain end-to-end and the sequence of intermediate states is compared to the ground truth as in (2). Here, we also implement the causal mask in Figure 3 (right) so that only log2 k iterations are needed, for additional stability.

- **CoT + teacher forcing:** implements the model in Section 3.2 with teacher forcing.

- **CoT + self-consistency:** implements the model in Section 3.3 with the causal mask in Figure 3
(right) and data augmentation for consistency checks. Weight quantization is omitted.

5Any polynomial order suffices for the number of augmented data samples.

All models are optimized using full-batch gradient descent on 100K 64-bit samples with a single Tesla T4 GPU. The three CoT models are trained with the 'CoT loss' (2) scaled by 1 k−1 to match the prediction loss of the direct model. Figure 4 shows training curves for the CoT loss (left) and the prediction loss (right) over 350 epochs when k = 32; results for all k and more details are provided in Appendix D.

We first note that the direct model (red) completely fails to learn the target, plateauing almost immediately. We observed that the weights become nearly uniform so that yˆ ≈ 0n and the prediction error is stuck at 0.5. This was not improved by using a multilayer transformer instead of repeated composition. The basic CoT model (yellow) is able to significantly decrease CoT loss but fails to fully solve the problem and eventually becomes unstable. Moreover, the prediction loss never improves beyond 0.5. Indeed, due to the hierarchical structure of parity, the model has no chance of making an informative prediction at the last level xd+k−1 unless all preceding levels have been fully solved.

In contrast, we verify that CoT with teacher forcing (blue) solves parity efficiently as predicted in Section 3.2, even with a small learning rate. After a burn-in phase, the CoT loss steadily decreases to nearly zero, at which point the prediction loss also decreases rapidly as the final level is solved. CoT with self-consistency (green) is also able to solve parity efficiently as predicted. Furthermore, the corresponding CoT loss curve clearly exhibits multiple learning stages. In the beginning, the model is essentially optimizing only the first level as subsequent outputs are zeroed out. After a short burn-in phase, the weights are optimized so that the softmax scores concentrate on the relevant nodes, at which point the CoT loss sharply decreases and the filters for the next level are deactivated, unlocking the next learning stage. This phased optimization repeats until all levels are fully solved and is crucial to arriving at the correct answer (in essence, teacher forcing is doing this at all levels simultaneously). Notably, a similar behavior seems to arise in the basic CoT model as well but fails due to accumulating error, further justifying the use of the filtering mechanism. These results confirm that training explicitly for CoT generation can improve performance on multistep tasks, and that controlling error accumulation via teacher forcing or self-consistency is key to ensuring proper step-by-step learning.

## 5 Conclusion

In this paper, by focusing on the k-parity problem, we provide an initial theoretical foundation for training transformers with CoT to perform stepwise reasoning. Our results show that gradient-based learning of parity requires significant iterations without intermediate supervision, but task decomposition using teacher forcing enables efficient learning in a single gradient update. Furthermore, when transformers are trained to generate reasoning chains end-to-end, data augmentation and selfconsistency checks can enhance their ability to solve complex tasks. Our work takes the first steps towards understanding how CoT can be leveraged to improve multi-step reasoning capability of foundation models.

## Acknowledgments

JK was partially supported by JST CREST (JPMJCR2015). TS was partially supported by JSPS KAKENHI (24K02905, 20H00576) and JST CREST (JPMJCR2115).

## References

Emmanuel Abbe and Colin Sandon. On the universality of deep learning. In Advances in Neural Information Processing Systems, 2020.

Konstantine Arkoudas. GPT-4 Can't Reason. *arXiv preprint arXiv:2308.03762*, 2023. Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In *Advances in Neural Information Processing Systems*, 2015.

Satwik Bhattamishra, Arkil Patel, Phil Blunsom, and Varun Kanade. Understanding in-context learning in transformers and LLMs by learning to learn discrete functions. In International Conference on Learning Representations, 2024.

David Chiang, Peter Cholak, and Anand Pillay. Tighter bounds on the expressivity of transformer encoders. In *International Conference on Machine Learning*, 2023.

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. Navigate through enigmatic labyrinth: a survey of chain of thought reasoning: advances, frontiers and future. In *Association for Computational Linguistics*, 2024.

Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. arXiv preprint arXiv:2311.01460, 2023.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: efficient finetuning of quantized LLMs. In *Advances in Neural Information Processing Systems*, 2023.

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. In Advances in Neural Information Processing Systems, 2023.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle Use a laptop? A question answering benchmark with implicit reasoning strategies. *Transactions of* the Association for Computational Linguistics, 9:346–361, 2021.

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT Press, 2016. http:
//www.deeplearningbook.org.

Kartik Goyal, Chris Dyer, and Taylor Berg-Kirkpatrick. Differentiable scheduled sampling for credit assignment. In *Association for Computational Linguistics*, 2017.

Xinyang Hu, Fengzhuo Zhang, Siyu Chen, and Zhuoran Yang. Unveiling the statistical foundations of chain-of-thought prompting methods. *arXiv preprint arXiv:2408.14511*, 2024.

Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. In *Proceedings of the 2023 Conference on Empirical Methods* in Natural Language Processing, pp. 1051–1068, Singapore, December 2023a. Association for Computational Linguistics.

Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of Transformers. *arXiv preprint* arXiv:2310.05249, 2023b.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

Juno Kim and Taiji Suzuki. Transformers learn nonlinear features in context: nonconvex mean-field dynamics on the attention landscape. In *International Conference on Machine Learning*, 2024.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In *Advances in Neural Information Processing Systems*, 2022.

Hongkang Li, Meng Wang, Songtao Lu, Xiaodong Cui, and Pin-Yu Chen. How do nonlinear transformers acquire generalization-guaranteed CoT ability? In High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning, 2024a.

Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, and Samet Oymak. Dissecting chain-of-thought: compositionality through in-context filtering and learning. In *Advances* in Neural Information Processing Systems, 2023.

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. In *International Conference on Learning Representations*, 2024b.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In *International* Conference on Learning Representations, 2024.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig.

Pre-train, prompt, and predict: a systematic survey of prompting methods in natural language processing. *ACM Comput. Surv.*, 55(9), January 2023.

Arvind Mahankali, Tatsunori B. Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. arXiv preprint arXiv:2307.03576, 2023.

William Merrill and Ashish Sabharwal. A logic for expressing log-precision transformers. In Advances in Neural Information Processing Systems, 2023.

William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought.

In *International Conference on Learning Representations*, 2024.

Tsvetomila Mihaylova and Andre F. T. Martins. Scheduled sampling for transformers. In ´ Association for Computational Linguistics: Student Research Workshop, 2019.

Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: a survey, 2024.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A comprehensive overview of large language models, 2024.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: scratchpads for intermediate computation with language models. *arXiv preprint arXiv:2112.00114*, 2021.

Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen. Reasoning with language model prompting: a survey. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Proceedings of the 61st Annual Meeting of the* Association for Computational Linguistics (Volume 1: Long Papers), pp. 5368–5393, Toronto, Canada, July 2023. Association for Computational Linguistics.

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: methods, analysis and insights from training Gopher. *arXiv preprint arXiv:2112.11446*, 2022.

Ran Raz. Fast learning requires good memory: a time-space lower bound for parity learning. *J. ACM*,
66(1), 2018.

Mansi Sakarvadia, Aswathy Ajith, Arham Khan, Daniel Grzenda, Nathaniel Hudson, Andre Bauer, ´
Kyle Chard, and Ian Foster. Memory injections: correcting multi-hop reasoning failures during inference in transformer-based language models. *arXiv preprint arXiv:2309.05605*, 2024.

Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. *arXiv preprint arXiv:2402.09268*, 2024.

Shai Shalev-Shwartz, Ohad Shamir, and Shaked Shammah. Failures of gradient-based deep learning.

In *International Conference on Machine Learning*, 2017.

Ohad Shamir. Distribution-specific hardness of learning neural networks. Journal of Machine Learning Research, 19, August 2018.

Yijun Tian, Yikun Han, Xiusi Chen, Wei Wang, and Nitesh V. Chawla. TinyLLM: learning a small student from multiple large language models. *arXiv preprint arXiv:2402.04616*, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, and Mi Zhang. Efficient large language models: a survey. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. Survey Certification.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In *International Conference on Learning Representations*, 2023.

Zhiwei Wang, Yunji Wang, Zhongwang Zhang, Zhangchen Zhou, Hui Jin, Tianyang Hu, Jiacheng Sun, Zhenguo Li, Yaoyu Zhang, and Zhi-Qin John Xu. Towards understanding how transformer perform multi-step reasoning with matching operation. *arXiv preprint arXiv:2405.15302*, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems*, 2022.

Noam Wies, Yoav Levine, and Amnon Shashua. Sub-task decomposition enables learning in sequence to sequence tasks. In *International Conference on Learning Representations*, 2023.

Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev, and Paulius Micikevicius. Integer quantization for deep learning inference: principles and empirical evaluation. *arXiv preprint arXiv:2004.09602*, 2020.

Zihan Yu, Liang He, Zhen Wu, Xinyu Dai, and Jiajun Chen. Towards better chain-of-thought prompting strategies: a survey, 2023.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. STaR: bootstrapping reasoning with reasoning. In *Advances in Neural Information Processing Systems*, 2022.

Hugh Zhang and David C. Parkes. Chain-of-thought reasoning is a policy improvement operator, 2023.

Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained Transformers learn linear models in-context.

arXiv preprint arXiv:2306.09927, 2023a.

Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, and Hai Zhao. Igniting language intelligence: the hitchhiker's guide from chain-of-thought reasoning to language agents. *arXiv preprint arXiv:2311.11797*,
2023b.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models, 2024.

Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, and Stuart Russell. Towards a theoretical understanding of the 'reversal curse' via training dynamics. arXiv preprint arXiv:2405.04669, 2024.

## Appendix A Proof Of Theorem 2

Denote the empirical inner product on R
dby ⟨*f, g*⟩n =
1 n Pn i=1 f(x i)g(x i) and the corresponding norm as ∥f∥
2n = ⟨*f, f*⟩n. We also write

.$$ L_{n,p}(\theta)\,=\,\dfrac{1}{2}\|p\,-\,f_{\theta}\|_n^2\,=\,\dfrac{1}{2n}\sum_{i=1}^n(p(\color{blue}{x^i})\,-\,f_{\theta}(\color{blue}{x^i}))^2$$  . 
to emphasize the dependency of Ln on p. Note that (d/k)
k ≤dk
≤ (*ed/k*)
kso that |P| = e Θ(d).

Bounding gradient variance. Consider the variance of the empirical gradient ∇Ln,p w.r.t. the target parity p:
Varn(θ; P) := Ep∈P
-∥∇Ln,p(θ) − Ep′∈P [∇Ln,p′ (θ)]∥
2.

We proceed to evaluate the magnitude of Varn(θ; P). For *p, p*′ ∈ P with p ̸= p
′it holds that

$$\langle p,p^{\prime}\rangle_{n}=\frac{1}{n}\sum_{i=1}^{n}\left(\prod_{j\in p}x_{j}^{i}\prod_{j^{\prime}\in p^{\prime}}x_{j^{\prime}}^{i}\right)=\frac{1}{n}\sum_{i=1}^{n}\left(\prod_{j\in p\Delta p^{\prime}}x_{j}^{i}\right).$$

Since Qj∈p∆p′ x ijis i.i.d. Unif({±1}) for fixed *p, p*′, by applying a union bound over Hoeffding's inequality, it follows for δ := p4d/n that

$$\Pr\left(\sup_{p\neq p^{\prime}}|\langle p,p^{\prime}\rangle_{n}|\geq\delta\right)\leq|P|(|P|-1)\exp\left(-\frac{n\delta^{2}}{2}\right)\leq\binom{d}{k}^{2}e^{-2d}\leq\left(\frac{2}{e}\right)^{2d}.$$

Then with probability at least 1 − e
−Ω(d) over random sampling, every off-diagonal component of the Gram matrix GP := (⟨p, p′⟩n)p,p′∈P has magnitude at most δ, while the diagonal entries are equal to 1. By the Gershgorin circle theorem, the maximum eigenvalue of GP satisfies |λmax(GP ) − 1| ≤ (|P| − 1)δ, thus λmax(GP ) ≤ 2(1 ∨ |P|δ). This implies that P constitutes a partial frame for the empirical L
2 norm with the corresponding frame upper bound. More specifically, for f : R
d → R, decompose

$$f=\sum_{p\in P}c_{p}\cdot p+f_{0},\quad f_{0}\in(\operatorname{span}P)^{\perp},$$
$$\lambda_{\operatorname*{max}}(G_{P})-$$

for some coefficient sequence c = (cp)p∈P . It follows that

$$\|f\|_{n}^{2}\geq\|f-f_{0}\|_{n}^{2}=\sum_{p,p^{\prime}\in P}c_{p}c_{p^{\prime}}\langle p,p^{\prime}\rangle_{n}=\|G_{P}^{1/2}c\|^{2}$$

and

$$\sum_{p\in P}\langle f,p\rangle_{n}^{2}=\sum_{p\in P}\left(\sum_{p^{\prime}\in P}c_{p^{\prime}}\langle p,p^{\prime}\rangle\right)^{2}=\|G_{P}c\|^{2}\leq\lambda_{\max}(G_{P})\|f\|_{n}^{2}.$$

Denoting D = dim Θ, we can therefore bound Varn(θ; P) as

$$\operatorname{Var}_{n}(\theta;P)=\operatorname*{inf}_{\mu\in\mathbb{R}^{D}}\mathbb{E}_{p\in P}\left[\|\nabla L_{n,p}(\theta)-\mu\|^{2}\right]$$
$$\leq\mathbb{E}_{p\in P}\left[\left\|\frac{1}{n}\sum_{i=1}^{n}(f_{\theta}(\mathbf{x}^{i})-p(\mathbf{x}^{i}))\nabla f_{\theta}(\mathbf{x}^{i})-\frac{1}{n}\sum_{i=1}^{n}f_{\theta}(\mathbf{x}^{i})\nabla f_{\theta}(\mathbf{x}^{i})\right\|^{2}\right]$$ $$=\mathbb{E}_{p\in P}\left[\sum_{j=1}^{D}\langle\nabla_{\theta_{j}}f_{\theta},p\rangle_{n}^{2}\right]=\frac{1}{|P|}\sum_{p\in P}\sum_{j=1}^{D}\langle\nabla_{\theta_{j}}f_{\theta},p\rangle_{n}^{2}$$
$$\begin{array}{l}{{\leq\sum_{j=1}^{D}\frac{\lambda_{\operatorname*{max}}(G_{P})}{|P|}\|\nabla_{\theta_{j}}f_{\theta}\|_{n}^{2}}}\\ {{\leq2\left(\frac{1}{|P|}\lor\sqrt{\frac{4d}{n}}\right)\operatorname*{sup}_{\theta,{\bf x}}\|\nabla f_{\theta}({\bf x})\|^{2}.}}\end{array}$$

Now by Chebyshev's inequality, for any ε > 0 it holds that

$$\operatorname*{Pr}\left(\|\nabla L_{n,p}(\theta)-\mathbb{E}_{p^{\prime}\in P}[\nabla L_{n,p^{\prime}}(\theta)]\|>\varepsilon\right)\leq{\frac{\operatorname{Var}_{n}(\theta;P)}{\varepsilon^{2}}}.$$

Constructing the oracle. As in Shamir (2018), we define the ε-approximate oracle ∇e as

$$\tilde{\nabla}L_{n,p}(\theta)=\begin{cases}\mathbb{E}_{p^{\prime}\in P}[\nabla L_{n,p^{\prime}}(\theta)]&\|\nabla L_{n,p}(\theta)-\mathbb{E}_{p^{\prime}\in P}[\nabla L_{n,p^{\prime}}(\theta)]\|\leq\varepsilon,\\ \nabla L_{n,p}(\theta)&\mathrm{otherwise}.\end{cases}$$
By union bounding, we see that during T steps the oracle always defaults to the mean gradient and
does not reveal any information on the true parity p, with probability at least
$$\mathrm{Pr}(Q)\geq1-{\frac{2T}{\varepsilon^{2}}}\left({\frac{1}{|P|}}\lor{\sqrt{\frac{4d}{n}}}\right)\operatorname*{sup}_{\theta,\mathbf{x}}\|\nabla f_{\theta}(\mathbf{x})\|^{2},$$

where Q ⊆ P denotes the corresponding subset of the hypothesis space. Note that the argument can be extended to any randomized algorithm and random initialization in a straightforward manner by lifting to the product probability space, and so we consider Q to be fixed. Then for any target parity p ∈ Q, the output θ(A) of the algorithm after T steps does not depend on p, so the predictor f = fθ(A)is also fixed.

Lower bounding the loss. We first remark that a simpler proof can be given for the sup norm error, which is enough to establish a separation. Consider arbitrary *p, p*′ ∈ P with p ̸= p
′and let x *∈ {±*1}
d be such that p(x) ̸= p
′(x), then

$|p(\mathbf{x})-f(\mathbf{x})|+|p^{\prime}(\mathbf{x})-f(\mathbf{x})|\geq|1-f(\mathbf{x})|+|-1-f(\mathbf{x})|\geq2$,
Now let σ : Q → Q be any automorphism of Q with no fixed points. The L∞ error can be bounded below by restricting to the noninformative set Q as follows.

below by restricting to the noncommutative set $Q$ as follows.  $$\mathbb{E}_{p\in P}\left[\sup_{\mathbf{x}}\left|p(\mathbf{x})-f_{\theta(A)}(\mathbf{x})\right|\right]\geq\mathbb{E}_{p\in P}\left[1_{p(Q)}\sup_{\mathbf{x}}\left|p(\mathbf{x})-f(\mathbf{x})\right|\right]$$ $$=\frac{1}{2|P|}\sum_{p\in Q}\left(\sup_{\mathbf{x}}\left|p(\mathbf{x})-f(\mathbf{x})\right|+\sup_{\mathbf{x}}\left|\sigma\circ p(\mathbf{x})-f(\mathbf{x})\right|\right)$$ $$\geq\frac{1}{2|P|}\cdot2|Q|=\Pr(Q).$$
$$\vartheta(\mathcal{A})(\mathbf{x}))^{2}]\geq\mathbb{E}_{p\in P,n}$$
For mean squared error, we similarly restrict to Q so that Ep∈P,x-(p(x) − fθ(A)(x))2≥ Ep∈P,x-1{p∈Q}(p(x) − f(x))2.

Since the range of p is contained in [−1, 1], the above loss will not increase when f is replaced by its clipped version ¯f(x) = (f(x) ∧ 1) ∨ (−1). Moreover, in Lemma 8 (proved at the end of the section) we show that |Ep∈P [p(x)]| ≤ e
−Ω(d) holds with probability 1 − e
−Ω(d) over the sample space of x, so that

$$\left|\mathbb{E}_{p\in P,\mathbf{x}}\left[p(\mathbf{x}){\bar{f}}(\mathbf{x})\right]\right|\leq(1-e^{-\Omega(d)})\mathbb{E}_{p\in P}\left[p(\mathbf{x})\right]+e^{-\Omega(d)}\leq e^{-\Omega(d)}$$
and also
$$\mathbb{E}_{p\in P,\mathbf{x}}\left[1_{\{p\in Q\}}p(\mathbf{x})\bar{f}(\mathbf{x})\right]=\mathbb{E}_{p\in P,\mathbf{x}}\left[p(\mathbf{x})\bar{f}(\mathbf{x})\right]-\mathbb{E}_{p\in P,\mathbf{x}}\left[1_{\{p\notin Q\}}p(\mathbf{x})\bar{f}(\mathbf{x})\right]$$ $$\leq e^{-\Omega(d)}+(1-\Pr(Q))\mathbb{E}_{\mathbf{x}}\left[|\bar{f}(\mathbf{x})|\right]$$ $$\leq e^{-\Omega(d)}+\frac{(1-\Pr(Q))^{2}}{2\Pr(Q)}+\frac{\Pr(Q)}{2}\mathbb{E}_{\mathbf{x}}\left[\bar{f}(\mathbf{x})^{2}\right].$$

Therefore we may bound

Ep∈P,x-(p(x) − fθ(A)(x))2≥ Ep∈P,x-1{p∈Q}(p(x) − ¯f(x))2 = Pr(Q) − 2Ep∈P,x-1{p∈Q}p(x)¯f(x)+ Pr(Q) · Ex-¯f(x) 2 ≥ Pr(Q) − (1 − Pr(Q))2 Pr(Q)− 2e −Ω(d) ≥ 2 −1 Pr(Q) − 2e −Ω(d) ≥ 1 − 4T ε 2  1 |P| ∨ r4d n ! sup θ,x ∥∇fθ(x)∥ 2 − 2e −Ω(d), (5)
where we have used the inequality 2 − (1 − t)
−1 ≥ 1 − 2t, valid for t ∈ [0, 1 2
].

The proof is completed by evaluating the following cases. (1) If n = e Ω(d)and T, ∥∇fθ∥ = O(poly(d)), the gradient variance is bounded as Varn(θ; P) ≤
e
−Ω(d). By taking ε = Varn(θ; P)
1/3, it follows that Pr(Q) = 1 − e
−Ω(d)and (5) yields the lower bound 1 − e
−Ω(d).

(2) If n *= Ω(*d ν), ∥∇fθ∥ = O(d ν1 ), ε = Θ(d
−ν2 ) and T = O(d ν3 ), the gradient variance is bounded as Varn(θ; P) ≤ O(d 2ν1+ν3+1/2−ν/2) = O(d
−2ν2−ν4 ) and (5) yields the lower bound 1 − O(d
−ν4 ).

Lemma 8. If k = Θ(d)*, it holds with probability at least* 1 − e
−Ω(d) *over random sampling that*

$$|\mathbb{E}_{p\in P}[p(\mathbf{x})]|\leq e^{-\Omega(d)}.$$

Proof. Let m denote the number of −1s in x. By the Chernoff bound for the binomial distribution,

$$\operatorname*{Pr}\left(\left|m-{\frac{d}{2}}\right|\leq{\frac{\delta d}{2}}\right)\geq1-2\exp\left(-{\frac{\delta^{2}d}{6}}\right)$$

for a constant δ ∈ (0, 1) to be determined, so we assume the above event throughout the proof. Moreover denoting the complement parity p c = [d] \ p, it holds that p(x) = x1 *· · ·* xd · p c(x) and |Ep∈P [p(x)]| = |Ep∈P [p c(x)]|, so it suffices to consider the case where 2k ≤ d.

Without loss of generality, we may assume that x = (−1, · · · , −1, 1, *· · ·* , 1) so that p(x) is decided as (−1)|p∩[m]|. We bound the cardinality of the set P+ := {p ∈ P | p(x) = 1}. Each parity in P+
can be determined by choosing 2j elements from [m] and k − 2j elements from [d] \ [m]. Denoting by [t]j the *coefficient* of operator of order j, we can evaluate

|P+| = ⌊m/ X 2⌋ j=0 m 2j d − m k − 2j  = ⌊m/ X 2⌋ j=0 m 2j [t]k−2j (1 + t) d−m = ⌊m/ X 2⌋ j=0 m 2j [t]k(1 + t) d−mt 2j = [t]k(1 + t) d−m ⌊m/ X 2⌋ j=0 m 2j t 2j = 1 2 [t]k(1 + t) d−m((1 + t) m + (1 − t) m) = 1 2 d k + 1 2 [t]k(1 − t 2) m′(1 + st) d−2m′ = 1 2 d k + s k 2 ⌊ X k/2⌋ j=0 (−1)j m′ j d − 2m′ k − 2j ,
where m′ = m ∧ (d − m) and s = ±1. It further follows that

$$\left|\frac{|P_{+}|}{|P|}-\frac{1}{2}\right|\leq\frac{1}{2|P|}\sum_{j=0}^{|k/2|}\binom{m^{\prime}}{j}\binom{d-2m^{\prime}}{k-2j}\leq\frac{1}{2|P|}\sum_{j=0}^{|k/2|}\binom{|d/2|}{j}\binom{|\delta d|}{k-2j}$$ $$\leq\frac{|k/2|}{2}\binom{d}{k}^{-1}\binom{|d/2|}{|k/2|}\binom{|\delta d|}{|\delta d/2|}\leq\frac{d}{4}\binom{d-|d/2|-|\delta d|}{k-|k/2|-|\delta d/2|}^{-1}$$ $$\leq\frac{d}{4}\binom{|d/4|}{|k/4|}^{-1}\leq\frac{d}{4}\left(\frac{d}{k}\right)^{-k/4}=e^{-\Theta(d)}.$$
$$\square$$

Here, we have chosen δ =
1 4
∧
k 2d = Θ(1) and used the inequality a1+a2+a3 b1+b2+b3
≥a1 b1 a2 b2 a3 b3
.

From this, we conclude that

$$\left|\mathbb{E}_{p\in P}[p(\mathbf{x})]\right|=\left|{\frac{|P\setminus P_{+}|-|P_{+}|}{|P|}}\right|\leq e^{-\Omega(d)}$$

with probability 1 − e
−Ω(d).

## B Proof Of Theorem 5

Proof of Lemma 4. For each d + 1 ≤ m ≤ d + k − 1 and 1 ≤ *j < m*, the only component of f
◦
depending on wj,m is f
◦m and

 ∂f ◦m(x;W) ∂wj,m  = |ϕ ′(ˆzm)| ·  ∂zˆm ∂wj,m  ≤ ∥ϕ ′∥∞  ∂σj (wm) ∂wj,mxj +X α̸=j ∂σα(wm) ∂wj,mxα  = ∥ϕ ′∥∞  σj (wm)(1 − σj (wm))xj − σj (wm)X α̸=j σα(wm)xα  ≤ ∥ϕ ′∥∞σj (wm)(1 − σj (wm)) + ∥ϕ ′∥∞σj (wm)X α̸=j σα(wm) ≤ 2∥ϕ ′∥∞σj (wm).
$$\leq2\|\phi^{\prime}\|_{\infty}\sigma_{j}(\mathbf{w}_{m}).$$
Hence it follows that

$$\sum_{m=d+1}^{d+k-1}\|\nabla\mathbf{w}f_{m}^{\circ}\|^{2}\leq4\|\phi^{\prime}\|_{\infty}^{2}\sum_{m=d+1}^{d+k-1}\sum_{j=1}^{m-1}\sigma_{j}(\pmb{w}_{m})^{2}\leq4\|\phi^{\prime}\|_{\infty}^{2}(k-1)=O(d),$$

as desired.

We say that a parity xj1*· · ·* xjrfor 1 ≤ j1, · · · , jr ≤ d + k − 1 is *trivial* if it always equals 1, or equivalently if its reduction to the independent bits x1, · · · , xd cancel out mod 2. For example, the parity x1x4x17 in Figure 1 is trivial. Define Ir,m as the set of nontrivial index r-tuples less than m:

$$I_{r,m}=\{(j_{1},\cdots,j_{r})\mid1\leq j_{1},\cdots,j_{r}\leq m-1,\,x_{j_{1}}\cdots x_{j_{r}}\not\equiv1\}$$

In particular, I1,m = [m − 1] since no single parity is trivial.

Lemma 9 (concentration of interaction terms). *If each bit* x ijfor i ∈ [n], j ∈ [d] *is i.i.d. generated* from the uniform distribution on {±1}, for any p > 0 it holds with probability at least 1 − p *that*

$$\operatorname*{max}_{\begin{array}{c}{1\leq r\leq4}\\ {(j_{1},\cdots,j_{r})\in I_{r,m}}\end{array}}{\frac{\left|\left\langle\mathbf{x}_{j_{1}},\cdots,\mathbf{x}_{j_{r}}\right\rangle\right|}{n}}\leq\kappa:={\sqrt{\frac{2}{n}}}\log{\frac{32d^{4}}{p}}.$$

Proof. Each tuple (j1, · · · , jr) ∈ Ir,m computes a specific nontrivial parity xj1*· · ·* xjrfor which the bits x ij1
· · · x ijr
, i = 1, · · · , n are i.i.d. Unif({±1}) due to symmetry. By Hoeffding's inequality we have that

$$\operatorname*{Pr}{\big(}|\langle\pmb{x}_{j_{1}},\cdots,\pmb{x}_{j_{r}}\rangle|\geq\lambda{\big)}\leq2e^{-\lambda^{2}/2n}.$$
Moreover, |Ir,m| ≤ (d + k − 1)r ≤ (2d − 1)rso that
$$+\,k-1)^{r}\leq(2d-1)^{r}{\mathrm{~so~that~}}$$
$|I_{1,m}|+\cdots+|I_{4,m}|\leq(2d-1)+\cdots+(2d-1)^{4}<(2d)^{4}$.  
Therefore it follows by union bounding that

$$\mathrm{Pr}\left(\operatorname*{max}_{1\leq r\leq4,(j_{1},\cdots,j_{r})\in I_{r,m}}|\langle\mathbf{x}_{j_{1}},\cdots,\mathbf{x}_{j_{r}}\rangle|\geq\lambda\right)\leq32d^{4}e^{-\lambda^{2}/2n},$$

which implies the statement. In particular, we take n = Ω(d 2+ϵ) and p = exp(−d ϵ/2) so that κ = O(d
−1−ϵ/4). This will ensure that the informative gradient signals will dominate the irrelevant interaction terms.

We now proceed to the main proof of Theorem 5. The superscript (0) at initialization is omitted for simplicity. The loss can be written more explicitly as

$$L(\mathbf{W})={\frac{1}{2n}}\sum_{m=d+1}^{d+k-1}\|\phi({\hat{\mathbf{z}}}_{m})-\mathbf{x}_{m}\|^{2},\quad{\hat{\mathbf{z}}}_{m}=\sum_{j=1}^{m-1}\sigma_{j}(\mathbf{w}_{m})\mathbf{x}_{j}.$$
$$\square$$

It is straightforward to verify for 1 ≤ *α < m* that

$$\frac{\partial\sigma_{\alpha}(\mathbf{w}_{m})}{\partial w_{j,m}}=(\delta_{j\alpha}-\sigma_{\alpha}(\mathbf{w}_{m}))\sigma_{j}(\mathbf{w}_{m})=(\delta_{j\alpha}-\sigma_{j}(\mathbf{w}_{m}))\sigma_{\alpha}(\mathbf{w}_{m})$$

and

$$\frac{\partial\hat{\mathbf{z}}_{m}}{\partial w_{j,m}}=\sum_{\alpha=1}^{m-1}(\delta_{j\alpha}-\sigma_{j}(\mathbf{w}_{m}))\sigma_{\alpha}(\mathbf{w}_{m})\mathbf{x}_{\alpha}=\sigma_{j}(\mathbf{w}_{m})(\mathbf{x}_{j}-\hat{\mathbf{z}}_{m}).$$

Then the gradient of L with respect to each element wj,m at initialization can be computed as

∂L ∂wj,m (W) = 1n (ϕ(zˆm) − xm) ⊤ ∂ϕ(zˆm) ∂wj,m = σj (wm) n⟨ϕ(zˆm) − xm, ϕ′(zˆm), xj − zˆm⟩ (6) = −1 n(m − 1) ⟨xm, 2czˆm, xj − zˆm⟩ (7) +1 n(m − 1) −1n + czˆ 2 m, 2czˆm, xj − zˆm(8) +1 n(m − 1) O(|zˆm| 4), 2czˆm, xj − zˆm (9) +1 n(m − 1) ϕ(zˆm) − xm, O(|zˆm| 3), xj − zˆm. (10)
Computing interaction strengths. The term (7) will be shown to contain the dominating gradient
signal when j = c1[m], c2[m], while the other terms can be bounded as perturbations. Let ℓ = h2[m] so that xm computes a 2
ℓ-parity.
For term (7), we substitute zˆm =1
$\mathbf{x}_{m}(\cdot)$, we substitute $\mathbf{x}_{m}=\frac{1}{m-1}\sum_{\alpha}\mathbf{x}_{\alpha}$ at initialization to expand  $$\frac{1}{n}\left\langle\mathbf{x}_{m},\hat{\mathbf{z}}_{m},\mathbf{x}_{j}-\hat{\mathbf{z}}_{m}\right\rangle=\frac{1}{n(m-1)}\sum_{\alpha}\left\langle\mathbf{x}_{m},\mathbf{x}_{\alpha},\mathbf{x}_{j}\right\rangle-\frac{1}{n(m-1)^{2}}\sum_{\alpha,\beta}\left\langle\mathbf{x}_{m},\mathbf{x}_{\alpha},\mathbf{x}_{\beta}\right\rangle,$$
Pα xα at initialization to expand where the dummy indices α, β, *· · ·* are taken to run over [m − 1]. Let us evaluate the third-order interaction terms ⟨xm, xα, xβ⟩. If h[α] = ℓ, xmxα computes the parity of 2 ℓ+1 independent bits from x1, · · · , xd so xmxαxβ cannot be trivial, hence (*m, α, β*) ∈ I3,m and |⟨xm, xα, xβ*⟩| ≤* nκ by Lemma 9. Similarly, h[β] = ℓ implies that (*m, α, β*) ∈ I3,m. Suppose h[α], h[β] ≤ ℓ − 1; unless h[α] = h[β] = ℓ − 1, the combined parity xαxβ will not contain enough independent bits to cancel out the 2 ℓ bits in xm, so again (*m, α, β*) ∈ I3,m. Moreover if h[α] = h[β] = ℓ − 1, xmxαxβ will be trivial if and only if {*α, β*} = {c1[m], c2[m]}, in which case ⟨xm, xα, xβ⟩ = n. Thus we have that

$$\frac{1}{n}\sum_{\alpha}\langle\mathbf{x}_{m},\mathbf{x}_{\alpha},\mathbf{x}_{\beta}\rangle=2+\frac{1}{n}\sum_{(m,\alpha,\beta)\in I_{3,m}}\langle\mathbf{x}_{m},\mathbf{x}_{\alpha},\mathbf{x}_{\beta}\rangle=2+O((m-1)^{2}\kappa).$$

Similarly, the contraction ⟨xm, xα, xj ⟩ can be nontrivial only if p[j] = m and only when α is the other child node of xm, so that

$${\frac{1}{n}}\sum_{\alpha}\left\langle\mathbf{x}_{m},\mathbf{x}_{\alpha},\mathbf{x}_{j}\right\rangle={\begin{cases}1+O((m-1)\kappa)&\mathsf{p}[j]=m,\\ O((m-1)\kappa)&{\mathrm{otherwise}}.\end{cases}}$$

Since κ = O(d
−1−ϵ/4) and *d < m* ≤ 2d − 1, we can therefore isolate the leading term of order Θ(d
−2) as

$$-\frac{1}{n(m-1)}\left\langle\mathbf{x}_{m},2c\hat{\mathbf{z}}_{m},\mathbf{x}_{j}-\hat{\mathbf{z}}_{m}\right\rangle$$ $$=-\frac{2c}{(m-1)^{2}}(1_{\{p[j]=m\}}+O(d\kappa))+\frac{2c}{(m-1)^{3}}(2+O(d^{2}\kappa))$$ $$=-\frac{2c}{(m-1)^{2}}1_{\{p[j]=m\}}+O(d^{-2-\epsilon/4}).$$

Next, for term (8), we expand

$$\frac{1}{n}\left(-\mathbf{1}_{n}+c\hat{\mathbf{z}}_{m}^{2},2c\hat{\mathbf{z}}_{m},\mathbf{x}_{j}-\hat{\mathbf{z}}_{m}\right)=-\frac{2c}{n}\left\langle\hat{\mathbf{z}}_{m},\mathbf{x}_{j}\right\rangle+\frac{2c}{n}\left\langle\hat{\mathbf{z}}_{m}^{2}\right\rangle+\frac{2c^{2}}{n}\left\langle\hat{\mathbf{z}}_{m}^{3},\mathbf{x}_{j}\right\rangle-\frac{2c^{2}}{n}\left\langle\hat{\mathbf{z}}_{m}^{4}\right\rangle.$$  The second-order terms can be computed as 
$$\frac{1}{n}\left\langle\hat{\mathbf{z}}_{m},\mathbf{x}_{j}\right\rangle=\frac{1}{n(m-1)}\Bigg(\left\langle\mathbf{x}_{j},\mathbf{x}_{j}\right\rangle+\sum_{\alpha\neq j}\left\langle\mathbf{x}_{\alpha},\mathbf{x}_{j}\right\rangle\Bigg)=\frac{1}{m-1}+O(\kappa),$$ $$\frac{1}{n}\left\langle\hat{\mathbf{z}}_{m}^{2}\right\rangle=\frac{1}{n(m-1)^{2}}\Bigg(\sum_{\alpha}\left\langle\mathbf{x}_{\alpha},\mathbf{x}_{\alpha}\right\rangle+\sum_{\alpha\neq\beta}\left\langle\mathbf{x}_{\alpha},\mathbf{x}_{\beta}\right\rangle\Bigg)=\frac{1}{m-1}+O(\kappa).$$

We evaluate the fourth-order interaction terms by looking at when (*α, β, γ, δ*) ∈/ I4,m can occur. Without loss of generality, suppose h[α] ≤ h[β] ≤ h[γ] ≤ h[δ].

(i) If h[β] < h[γ] < h[δ], the parities of xα, xβ, xγ must combine without overlaps to cancel out xδ, so it must hold that xγ is a child of xδ and xα, xβ are the two children of the other child.

This subtree is fully determined by the choice of the index δ and one of its child nodes, so there are at most O(d) trivial 4-tuples in this case.

(ii) If h[β] = h[γ] < h[δ], it still must hold that h[γ] = h[δ] − 1. Moreover, both xβ, xγ must be children of xδ; otherwise, the bits of xδ and the non-child node cannot be canceled out by the remaining nodes. Then either xβ = xγ or xβxγ = xδ, and in both cases we see that xαxβxγxδ cannot be trivial.

(iii) If h[β] < h[γ] = h[δ], it must be that γ = δ, otherwise the bits of xγxδ cannot be canceled out by xαxβ. It follows that xαxβ ≡ 1 and α = β, so there are O(d 2) trivial 4-tuples in this case.

(iv) If h[β] = h[γ] = h[δ], it must again hold that two indices must be equal, and the remaining two indices must also be equal, so there are also O(d 2) trivial 4-tuples.

Hence it follows that
 we have  $ \frac{1}{n}\left<\hat{\mathbf{z}}_m^4\right>=\frac{1}{n(m-1)^4}\sum_{\alpha,\beta,\gamma,\delta}\left<\mathbf{x}_\alpha,\mathbf{x}_\beta,\mathbf{x}_\gamma,\mathbf{x}_\delta\right>$.