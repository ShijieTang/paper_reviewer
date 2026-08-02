# TRANSFORMERS PROVABLY SOLVE PARITY EFFICIENTLY WITH CHAIN OF THOUGHT

Juno Kim1,2<sup>∗</sup> Taiji Suzuki1,2

<sup>1</sup>Department of Mathematical Informatics, University of Tokyo

<sup>2</sup>Center for Advanced Intelligence Project, RIKEN

<sup>∗</sup>junokim@berkeley.edu

### ABSTRACT

This work provides the first theoretical analysis of training transformers to solve complex problems by recursively generating intermediate states, analogous to fine-tuning for chain-of-thought (CoT) reasoning. We consider training a one-layer transformer to solve the fundamental k-parity problem, extending the work on RNNs by [Wies et al.](#page-12-0) [\(2023\)](#page-12-0). We establish three key results: (1) any finite-precision gradient-based algorithm, without intermediate supervision, requires substantial iterations to solve parity with finite samples. (2) In contrast, when intermediate parities are incorporated into the loss function, our model can learn parity in one gradient update when aided by *teacher forcing*, where ground-truth labels of the reasoning chain are provided at each generation step. (3) Even without teacher forcing, where the model must generate CoT chains end-to-end, parity can be learned efficiently if augmented data is employed to internally verify the soundness of intermediate steps. Our findings, supported by numerical experiments, show that task decomposition and stepwise reasoning naturally arise from optimizing transformers with CoT; moreover, self-consistency checking can improve multistep reasoning ability, aligning with empirical studies of CoT.

# 1 INTRODUCTION

Large language models (LLMs) based on the transformer architecture [\(Vaswani et al.,](#page-12-1) [2017\)](#page-12-1) have achieved astounding success across a variety of natural language processing and machine learning tasks (see e.g. [Wan et al.,](#page-12-2) [2024;](#page-12-2) [Minaee et al.,](#page-11-0) [2024;](#page-11-0) [Naveed et al.,](#page-11-1) [2024;](#page-11-1) [Zhao et al.,](#page-13-0) [2024\)](#page-13-0). However, they often struggle when tasked with solving complex reasoning problems, especially in a zero-shot setting without any form of intermediate guidance or supervision [\(Geva et al.,](#page-10-0) [2021;](#page-10-0) [Rae et al.,](#page-12-3) [2022;](#page-12-3) [Arkoudas,](#page-10-1) [2023;](#page-10-1) [Wang et al.,](#page-12-4) [2024\)](#page-12-4). These failures are particularly evident in tasks requiring multi-hop reasoning or compounded logical steps [\(Sakarvadia et al.,](#page-12-5) [2024\)](#page-12-5).

A promising approach to overcome these limitations is chain-of-thought (CoT) reasoning, where the model is prompted or fine-tuned to solve complex tasks step-by-step by explicitly making intermediate reasoning steps to arrive at the desired answers [\(Wei et al.,](#page-12-6) [2022;](#page-12-6) [Kojima et al.,](#page-11-2) [2022\)](#page-11-2). Since its discovery, CoT reasoning has been shown to significantly enhance the problem-solving capabilities of LLMs while also increasing the interpretability and trustworthiness of the reasoning process, and has spawned numerous prompting techniques [\(Liu et al.,](#page-11-3) [2023;](#page-11-3) [Qiao et al.,](#page-11-4) [2023\)](#page-11-4) and applications for a variety of downstream tasks including common-sense reasoning, mathematical problem-solving, and symbolic or multi-modal reasoning; see e.g. [Zhang et al.](#page-13-1) [\(2023b\)](#page-13-1); [Yu et al.](#page-13-2) [\(2023\)](#page-13-2); [Chu et al.](#page-10-2) [\(2024\)](#page-10-2) for surveys on CoT. In particular, besides being used as a prompting method, directly training or fine-tuning models to generate CoT has also been shown to significantly improve multi-step reasoning performance [\(Nye et al.,](#page-11-5) [2021;](#page-11-5) [Wei et al.,](#page-12-6) [2022;](#page-12-6) [Zelikman et al.,](#page-13-3) [2022;](#page-13-3) [Lightman et al.,](#page-11-6) [2024\)](#page-11-6).

Despite these empirical successes, however, the theoretical understanding of the CoT mechanism and task decomposition in transformers is still limited. Existing works focus on characterizing the expressivity of transformers equipped with CoT, providing constructions which can solve certain complexity classes [\(Feng et al.,](#page-10-3) [2023;](#page-10-3) [Merrill & Sabharwal,](#page-11-7) [2023;](#page-11-7) [2024;](#page-11-8) [Li et al.,](#page-11-9) [2024b\)](#page-11-9), studying the class of functions that can be learned in-context with CoT [\(Li et al.,](#page-11-10) [2023;](#page-11-10) [Bhattamishra et al.,](#page-10-4) [2024\)](#page-10-4), or analyzing the estimation error of multi-step models [\(Hu et al.,](#page-10-5) [2024\)](#page-10-5). Nevertheless, such

approaches do not indicate how such capabilities might emerge when training transformers to generate reasoning chains. [Li et al.](#page-11-11) [\(2024a\)](#page-11-11) analyze the training dynamics of a one-layer transformer in an in-context learning setting and show that CoT ability may be acquired; however, they do not consider explicitly training with CoT chains, which is a more difficult problem since the objective depends on the recursive application of the transformer to itself.

In this paper, we seek to formalize the mechanism through which stepwise reasoning emerges in transformers optimized to generate CoT chains. We focus on the specific problem of *bit subset parity* (learning the parity of an unknown subset of k bits from a d-bit input), which is known to be impossible to learn end-to-end with any finite-precision gradient-based algorithm in polynomial steps [\(Shalev-](#page-12-7)[Shwartz et al.,](#page-12-7) [2017;](#page-12-7) [Shamir,](#page-12-8) [2018\)](#page-12-8). In contrast, [Wies et al.](#page-12-0) [\(2023\)](#page-12-0) have demonstrated that recurrent neural networks (RNNs) can solve parity efficiently when provided with intermediate supervision. We build on this direction to establish positive optimization guarantees for the transformer architecture. Our object of study is a one-layer transformer incorporating a softmax attention layer, feedforward layer and positional encoding, that is recursively applied to its own output to generate a sequence of intermediate parity computations to arrive at the desired output, analogous to CoT generation. Our contributions are summarized as follows.

- We extend the impossibility result for parity (Theorem [1\)](#page-2-0), which was established only for population gradient descent, to the more realistic finite-sample setting in Theorem [2.](#page-5-0) We prove that any iterative algorithm with access to an approximate gradient oracle for the end-to-end empirical loss cannot solve a random target parity within a specific polynomial number of steps.
- In contrast, we show that when the loss is summed over all intermediate states, by utilizing *teacher forcing*, a form of process supervision wherein ground-truth intermediate steps are provided during training,[<sup>1</sup>](#page-1-0) our model can learn any parity in a single gradient update (Theorem [5\)](#page-6-0). This shows the benefits of training directly with CoT chains to acquire task decomposition ability.
- We further consider training with CoT generated end-to-end without teacher forcing,[<sup>2</sup>](#page-1-1) and show that parity can still be learned in a logarithmic number of steps if augmented data is employed to check the validity of intermediate steps (Theorem [7\)](#page-8-0), thereby mimicking self-consistency checks often used in CoT reasoning [\(Zelikman et al.,](#page-13-3) [2022;](#page-13-3) [Wang et al.,](#page-12-9) [2023;](#page-12-9) [Huang et al.,](#page-10-6) [2023a\)](#page-10-6).
- We conduct numerical experiments supporting our findings (Section [4](#page-8-1) and Appendix [D\)](#page-25-0).

Our results provide theoretical insights into how transformers can naturally and efficiently optimize to perform task decomposition, emphasizing the role of explicit intermediate supervision for complex tasks. Moreover, these findings corroborate recent empirical studies on CoT reasoning demonstrating improved performance through process supervision and internal validation of reasoning chains [\(Huang et al.,](#page-10-6) [2023a;](#page-10-6) [Tian et al.,](#page-12-10) [2024;](#page-12-10) [Lightman et al.,](#page-11-6) [2024\)](#page-11-6).

### 1.1 RELATED WORKS

Complexity of transformers. A line of work aims to understand the effectiveness of CoT from the perspective of complexity theory. [Feng et al.](#page-10-3) [\(2023\)](#page-10-3) show that autoregressive transformers of constant size can solve basic arithmetic tasks by recursively generating CoT reasoning steps, which is not possible when directly generating the solution; this separation arises because looping the generated outputs back to its inputs increases the 'effective depth' of the model. Works such as [Chiang et al.](#page-10-7) [\(2023\)](#page-10-7); [Merrill & Sabharwal](#page-11-7) [\(2023\)](#page-11-7) study the expressivity of fixed-precision transformer architectures in terms of classes of formal languages. [Merrill & Sabharwal](#page-11-8) [\(2024\)](#page-11-8); [Li et al.](#page-11-9) [\(2024b\)](#page-11-9) show that CoT reasoning enables recognizing wider language classes, and characterizes the increased expressivity depending on the length of the reasoning chain. [Sanford et al.](#page-12-11) [\(2024\)](#page-12-11) studies the relation between transformers and massively parallel computation protocols, showing that logarithmic depth suffices to solve multi-hop induction tasks that cannot be efficiently solved by other sequence models.

<sup>1</sup>Teacher forcing or process supervision is a training procedure for recurrent models in which the model receives the ground truth output at time t as input at time t + 1 during training [\(Goodfellow et al.,](#page-10-8) [2016,](#page-10-8) p.377). Many fine-tuning methods with ground-truth CoT chains implement teacher forcing, being more effective than output supervision with chains generated end-to-end [\(Deng et al.,](#page-10-9) [2023;](#page-10-9) [Tian et al.,](#page-12-10) [2024;](#page-12-10) [Lightman et al.,](#page-11-6) [2024\)](#page-11-6).

<sup>2</sup>Teacher forcing can induce exposure bias where a model is not robust to its own errors. In practice, partial (scheduled or random) teacher forcing methods are used to overcome this issue [\(Bengio et al.,](#page-10-10) [2015;](#page-10-10) [Goyal et al.,](#page-10-11) [2017;](#page-10-11) [Mihaylova & Martins,](#page-11-12) [2019\)](#page-11-12).

Additionally, [Li et al.](#page-11-10) [\(2023\)](#page-11-10); [Bhattamishra et al.](#page-10-4) [\(2024\)](#page-10-4) study the class of functions that can be learned in context by transformers with CoT from the point of view of in-context learning.

Optimization and generalization of CoT. [Zhu et al.](#page-13-4) [\(2024\)](#page-13-4) study the 'reversal curse' via the training dynamics of a one-layer transformer and shows that the model fails to generalize from A → B, B → C to A → C as an argument for the necessity of explicit step-by-step reasoning. [Hu](#page-10-5) [et al.](#page-10-5) [\(2024\)](#page-10-5) study CoT prompting from a statistical estimation perspective by introducing a multi-step latent variable model for CoT and analyzing its approximation, generalization and prompting-based errors. Notably, [Li et al.](#page-11-11) [\(2024a\)](#page-11-11) study the training dynamics of a one-layer attention-only transformer model in an in-context learning setting and show that CoT generalization capability can be obtained. However, this does not address the possibility or benefits of training with CoT chains. [Lightman et al.](#page-11-6) [\(2024\)](#page-11-6) empirically study training LLMs with either process or outcome supervision, showing that the former significantly outperforms the latter when training to solve challenging reasoning tasks.

Parity and task decomposition. The difficulty of learning parity without task decomposition is established in [Shalev-Shwartz et al.](#page-12-7) [\(2017\)](#page-12-7); [Shamir](#page-12-8) [\(2018\)](#page-12-8). The work most relevant to our paper is [Wies et al.](#page-12-0) [\(2023\)](#page-12-0), which study task decomposition for parity with classical Elman RNNs. They show that by incorporating intermediate states into the loss function and utilizing teacher forcing, parity can be solved with polynomial iterations and embedding size. Our Theorem [5](#page-6-0) extends this positive result to autoregressive transformers, rigorously establishing the benefits of CoT-based training.

### 2 PROBLEM SETUP

Notation. We write [n] := {1, 2, · · · , n} for any integer n. Scalar operations apply componentwise to vectors, e.g. for z ∈ R <sup>n</sup> we write ϕ(z) = (ϕ(z1), · · · , ϕ(zn))<sup>⊤</sup>, z <sup>2</sup> = z ⊙ z = (z 2 1 , · · · , z<sup>2</sup> n ) and |z| = (|z1|, · · · , |zn|) <sup>⊤</sup>. The 2-norm is always denoted by ∥·∥. The multi-linear inner product or contraction of z1, · · · , z<sup>r</sup> ∈ <sup>R</sup> n for any <sup>r</sup> ∈ <sup>N</sup> is denoted as ⟨z1, · · · , <sup>z</sup>r⟩ := P<sup>n</sup> <sup>i</sup>=1 z1,i · · · zr,i. In particular, ⟨z1⟩ = z ⊤ <sup>1</sup> 1<sup>n</sup> and ⟨z1, z2⟩ = z ⊤ <sup>1</sup> z2.

### 2.1 THE PARITY PROBLEM

Let d ≥ k ≥ 2 be integers and let P denote the set of size k subsets of {1, · · · , d} equipped with the uniform distribution. In this paper, we study the k-parity problem for d-bit inputs x = (x<sup>j</sup> ) d <sup>j</sup>=1 ∼ Unif({±1} d ), where the output y = Q j∈p x<sup>j</sup> is determined by the parity of an unknown subset of bits p ∈ P. We abuse notation and identify the set of indices p with the corresponding parity mapping x 7→ Q j∈p x<sup>j</sup> . Given n samples (x i , y<sup>i</sup> )i∈[n] , our goal is to predict the parity of any test input.

It is known that parity is fundamentally difficult in the sense that it cannot be solved in polynomial time by any finite-precision gradient-based algorithm, such as neural networks. More precisely, let {f<sup>θ</sup> | θ ∈ Θ} be any differentiable (w.r.t. θ) parametrized model with polynomially bounded gradients, ∥∇fθ(x)∥ = O(poly(d)), and define the population loss L¯ = <sup>E</sup><sup>x</sup> -(y − fθ(x))<sup>2</sup> . We presume access to an <sup>ε</sup>-*approximate gradient oracle* ∇e for <sup>L</sup>, which takes any <sup>θ</sup> ∈ <sup>Θ</sup> as query and returns a vector ∇eL¯(θ) satisfying ∥∇eL¯(θ) − ∇L¯(θ)∥<sup>2</sup> ≤ <sup>ε</sup>, potentially in an adversarial manner. Then the following holds:

Theorem 1 [\(Wies et al.](#page-12-0) [\(2023\)](#page-12-0), Theorem 4). *Let* ℓ0−<sup>1</sup> *be the zero-one loss. There exists an* O(e −d/3 ) *approximate oracle* ∇e *such that*[<sup>3</sup>](#page-2-1) *the output* θ(A) *of any iterative algorithm* A *which sequentially makes at most* <sup>O</sup>(poly(d)) *queries to* ∇eL¯ *must satisfy*

$$\mathbb{E}_{\mathbf{x}} [\ell_{0-1}(p(\mathbf{x}), f_{\theta(\mathcal{A})}(\mathbf{x}))] \geq \frac{1}{2} - O(e^{-d})$$

*with probability at least* 1 − O(e −d/3 )*, when the target parity* p *is uniformly sampled from* P*.*

The intuition is that the set P of parity functions is exponentially large in the sense that all elements of P are pairwise orthogonal with respect to the data distribution. This implies that the variance of

<sup>3</sup>The original paper states that A can be any iterative gradient-based algorithm which receives an Ω(e −d/3 ) approximation of the gradient at each step. However, to be more precise, the result is only valid for certain adversarial perturbation schemes.

![](_page_3_Diagram_1.jpeg)

Figure 1: A hierarchical decomposition of an 8-parity problem for d = 16. Here x<sup>17</sup> = x1x<sup>4</sup> so that c1[17] = 1, c2[17] = 4, p[17] = 21 and h[17] = 1.

each gradient call ∇L¯(θ) with respect to the target parity p is exponentially small [\(Shalev-Shwartz](#page-12-7) [et al.,](#page-12-7) [2017\)](#page-12-7) and is drowned out by the noise from the adversarial oracle, so that no information can be gained on the target without exponentially many queries. See Section [3.1](#page-4-0) for more details.

Task decomposition. As in [Wies et al.](#page-12-0) [\(2023\)](#page-12-0), we assume k = 2<sup>v</sup> for an integer v for simplicity and decompose the problem into a hierarchy of 2-parity computations which can be efficiently learned in a sequential manner by our model. This is expressed as a complete binary tree T of height v and 2k−1 nodes. The lowest level contains k nodes representing the bits xj<sup>m</sup> for m ∈ [k]. The remaining nodes are labeled xd+1, · · · , xd+k−<sup>1</sup> starting from the next lowest level and moving upwards, left to right. The largest index in level ℓ for 0 ≤ ℓ ≤ v is denoted as d<sup>ℓ</sup> = d + P<sup>ℓ</sup> <sup>j</sup>=1 2 v−j , d<sup>0</sup> = d. Also, for each m > d, the indices of the two child nodes of x<sup>m</sup> are denoted as c1[m], c2[m] where 1 ≤ c1[m] < c2[m] < m. In addition, the parent node index of x<sup>m</sup> is denoted as p[m] and the level or height of x<sup>m</sup> is denoted as h[m], so that dh[m]−<sup>1</sup> < m ≤ dh[m] .

### 2.2 TRANSFORMER MODEL

![](_page_3_Diagram_6.jpeg)

Figure 2: Illustration of the recursive data generation process by the transformer model. (a) Each token consists of a one-hot positional encoding e<sup>j</sup> and parity data x<sup>j</sup> . The d input tokens (blue) are fixed. The token xˆ<sup>m</sup> is generated at the (m − d)th step by computing attention scores based on position, combining the previous tokens and applying the feedforward layer ϕ. xˆd+k−<sup>1</sup> is returned as the model prediction. (b) For the no teacher forcing setup in Section [3.3,](#page-7-0) data augmentation u<sup>j</sup> is implemented to check for self-consistency. If the augmented outputs from the previous generation (red) are uninformative, a filter ι is applied to zero out the subsequent output.

We study a one-layer transformer architecture employing absolute positional encoding and a singlehead softmax attention layer followed by a shallow feedforward layer; skip connections are omitted for simplicity. See Figure [2](#page-3-0) for a visualization of our setup.

*Data encoding*: Each input token x<sup>j</sup> = (x i j ) n <sup>i</sup>=1 for j ∈ [d] is the n-dimensional vector consisting of the jth bit of each sample x i . We also add dummy tokens xd+1, · · · , xd+k−<sup>1</sup> initially set to 0n, which will learn to sequentially generate the actual intermediate nodes. Each x<sup>j</sup> is concatenated with the one-hot positional encoding e<sup>j</sup> ∈ <sup>R</sup> d+k−1 for j ∈ [d + k − 1] to form the internal input p<sup>j</sup> = (x ⊤ <sup>j</sup> e ⊤ j ) <sup>⊤</sup> ∈ <sup>R</sup> n+d+k−1 to the attention layer.

*Softmax attention layer*: The attention layer is defined as in [\(1\)](#page-4-1) in terms of key, query and value matrices K, Q, V. We fix the first n columns of K, Q to zero so that the attention scores are determined by only the positional encodings. This ensures that the transformer focuses on learning which positions contribute to the parity at each step. K, Q are then reparametrized by a single matrix W ∈ R (d+k−1)<sup>2</sup> ; conversely, the value matrix is set to only preserve the x component, as follows.

$$\mathbf{K}^\top \mathbf{Q} = \begin{pmatrix} \mathbf{0}_{n \times n} & \mathbf{0}_{n \times (d+k-1)} \\ \mathbf{0}_{(d+k-1) \times n} & \mathbf{W} \end{pmatrix}, \quad \mathbf{V} = (\mathbf{I}_{n \times n} \quad \mathbf{0}_{n \times (d+k-1)}).$$

This type of reparametrization is common in the literature to make dynamical analysis tractable [\(Zhang et al.,](#page-13-5) [2023a;](#page-13-5) [Huang et al.,](#page-10-12) [2023b;](#page-10-12) [Mahankali et al.,](#page-11-13) [2023;](#page-11-13) [Kim & Suzuki,](#page-11-14) [2024\)](#page-11-14).

*Feedforward layer*: The feedforward layer realizes a fixed link function ϕ : [−1, 1] → [−1, 1], applied elementwise and only to the x<sup>j</sup> component; the positional encodings are not affected. To exploit the decomposition of our task into 2-parities, we choose ϕ such that ϕ(0) = −1, ϕ(±1) = 1 so that sums are converted into parities, i.e. ϕ( a+b 2 ) = ab for a, b ∈ {±1}. Moreover, we require that ϕ ′ (0) = ϕ ′ (±1) = 0 and assume ϕ is symmetric and sufficiently regular, so that we may expand ϕ(t) = −1 + ct<sup>2</sup> + O(|t| 4 ) and ϕ ′ (t) = 2ct + O(|t| 3 ).

The transformer computes TF(x1, · · · , xd+k−1;W) = (xˆ1, · · · , xˆd+k−1) where the original data xˆ<sup>j</sup> = x<sup>j</sup> , j ∈ [d] remain unchanged and tokens xˆd+1, · · · , xˆd+k−<sup>1</sup> are computed as

$$\hat{\mathbf{x}}_m = \phi(\hat{\mathbf{z}}_m), \quad \hat{\mathbf{z}}_m = \sum_{j=1}^{m-1} \mathbf{V}\hat{\mathbf{p}}_j \cdot \text{softmax}(\hat{\mathbf{p}}_j^\top \mathbf{K}^\top \mathbf{Q}\hat{\mathbf{p}}_m) = \sum_{j=1}^{m-1} \sigma_j(\mathbf{w}_m) \mathbf{x}_j, \quad (1)$$

where the softmax scores σ<sup>j</sup> (wm) = e <sup>w</sup>j,m/ P<sup>m</sup>−<sup>1</sup> <sup>α</sup>=1 e <sup>w</sup>α,m. Here, we have implicitly added the causal mask wj,m ← −∞ to the attention layer for j ≥ m or m ≤ d. Note that each zˆm, xˆ<sup>m</sup> will be contained in the cube [−1, 1]<sup>d</sup> as long as the input tokens are also contained in [−1, 1]<sup>d</sup> .

Chain of thought. Consider repeatedly applying TF(·) to its own output to generate a 'reasoning chain.' Since the input tokens are fixed, the token xˆd+1 will be updated once and then always yield the same value afterwards. Next, since xˆd+2 depends on the input tokens and xˆd+1, it will be updated twice before becoming fixed. Repeating this, the entire chain stops updating after at most k − 1 steps, yielding the output

$$\text{TF}^{(k-1)}(\mathbf{x}_1, \dots, \mathbf{x}_d, \mathbf{0}_n, \dots, \mathbf{0}_n; \mathbf{W}) = (\hat{\mathbf{x}}_1, \dots, \hat{\mathbf{x}}_{d+k-1})$$

where the intermediate predictions are recursively computed as xˆ<sup>m</sup> = ϕ( P<sup>m</sup>−<sup>1</sup> <sup>j</sup>=1 σ<sup>j</sup> (wm)xˆ<sup>j</sup> ). Finally, the top node is returned as the model prediction yˆ = xˆd+k−1.

This process can be seen as a simplified version of CoT reasoning, albeit not in an in-context learning setting: instead of one-shot predicting y i from x i , the model starts by solving simpler subtasks and uses the information to attack compound problems, learning to generate intermediate reasoning steps xd+1 → · · · → xd+k−<sup>1</sup> to finally arrive at the desired solution. Importantly, this process is not possible if the model is only trained on the one-shot data (x i , y<sup>i</sup> )i∈[n] as we show in Section [3.1.](#page-4-0) Instead, we incorporate the prediction error for all intermediate states directly into our loss function [\(Lightman et al.,](#page-11-6) [2024\)](#page-11-6). We also consider shortening the reasoning chain by using a different causal mask in Section [3.3,](#page-7-0) which will result in improved control of error and faster convergence.

# 3 MAIN RESULTS

### 3.1 HARDNESS OF PARITY WITHOUT COT

Before analyzing our transformer model, we first prove a negative learning result in the absence of intermediate supervision that extends Theorem [1,](#page-2-0) which was stated with respect to the population objective L¯ and zero-one test loss ℓ0−1, to finite samples and mean squared loss.

Let f<sup>θ</sup> : {±1} <sup>d</sup> → <sup>R</sup> be any differentiable parametrized model and suppose we select the target parity p uniformly at random from P. In the finite-sample setting, n i.i.d. samples (x i , y<sup>i</sup> )i∈[n] are generated as x <sup>i</sup> ∼ Unif({±1} d ), y <sup>i</sup> = p(x i ) and we are given access to (approximate) gradients from the empirical loss

$$L_n(\theta) = \frac{1}{2n} \sum_{i=1}^n (y^i - f_\theta(\mathbf{x}^i))^2 = \frac{1}{2} \|p - f_\theta\|_n^2,$$

where ∥·∥<sup>n</sup> is the empirical norm. It is important that the model f<sup>θ</sup> is applied to each x i separately and does not cross-reference between different samples, as there exist more efficient parity-learning algorithms if the data is allowed to be manipulated freely. For example, Gaussian elimination can solve parity with O(d) samples and O(d 3 ) iterations [\(Raz,](#page-12-12) [2018\)](#page-12-12). Moreover, this implies that neural networks trained with stochastic gradient descent can also solve parity in polynomial time [\(Abbe &](#page-10-13) [Sandon,](#page-10-13) [2020\)](#page-10-13). Instead, in our setting the model is forced to learn from the averaged gradient signal and can only implicitly utilize the correlation between samples.

We show the following result for learning parities with finite-samples in Appendix [A:](#page-14-0)

Theorem 2 (hardness of finite-sample parity). *Suppose* k = Θ(d)*.*

- (1) *If* n = e Ω(d) *and* f<sup>θ</sup> *has polynomially bounded gradients, there exists an* e −Ω(d) *-approximate gradient oracle* ∇e *such that with probability* <sup>1</sup> − <sup>e</sup> <sup>−</sup>Ω(d) *over random sampling, the output* θ(A) *of any iterative (possibly randomized) algorithm which makes at most* O(poly(d)) *queries to* ∇e<sup>L</sup><sup>n</sup> *has* <sup>L</sup>2*-loss lower bounded as*

$$\mathbb{E}_{p \in P, \mathbf{x}} \left[ (p(\mathbf{x}) - f_{\theta(\mathcal{A})}(\mathbf{x}))^2 \right] \geq 1 - e^{-\Omega(d)}.$$

- (2) *If* n = Ω(d ν ) *and* ∥∇fθ∥ = O(d <sup>ν</sup><sup>1</sup> )*, there exists an* O(d <sup>−</sup>ν<sup>2</sup> )*-approximate gradient oracle* ∇e *such that with probability* 1 − e <sup>−</sup>Ω(d) *over random sampling, the output* θ(A) *of any iterative (possibly randomized) algorithm which makes at most* O(d <sup>ν</sup><sup>3</sup> ) *queries to* ∇e<sup>L</sup><sup>n</sup> *has* <sup>L</sup>2*-loss lower bounded, where* ν = 4ν<sup>1</sup> + 4ν<sup>2</sup> + 2ν<sup>3</sup> + 2ν<sup>4</sup> + 1*, as*

$$\mathbb{E}_{p \in P, \mathbf{x}} \left[ (p(\mathbf{x}) - f_{\theta(\mathcal{A})}(\mathbf{x}))^2 \right] \geq 1 - O(d^{-\nu_4}).$$

We remark that the bounds are asymptotically optimal since f<sup>θ</sup> ≡ 0 is a valid estimator. Moreover, the expectation over p ∈ P can be replaced by the corresponding 'with high probability' statement.

A counter-intuitive aspect of the above result is that parity becomes potentially more difficult when the number of samples increases. Indeed, with exponential samples n = e Ω(d) [\(1\)](#page-5-1) we basically recover the statement of Theorem [1,](#page-2-0) while the guarantees for n = poly(d) [\(2\)](#page-5-2) are also polynomial in d. This is because the difficulty of parity (Theorem [1\)](#page-2-0) fundamentally depends on the following result:

Proposition 3 [\(Shalev-Shwartz et al.](#page-12-7) [\(2017\)](#page-12-7), Theorem 1). *Suppose* x *be a random variable in* R d *. Let* H *be a class of bounded real-valued functions on* R d *such that* <sup>E</sup>x[h(x)h ′ (x)] = 0 *for any two distinct* h, h′ ∈ H *and* f<sup>θ</sup> *a differentiable parametric model with gradients bounded by* <sup>E</sup>x[∥∇fθ∥ ] ≤ F(θ) 2 *. Then for the loss* Fh(θ) := <sup>E</sup>x[(h(x)−fθ(x))<sup>2</sup> ] *where* h *is chosen uniformly at random from* H*, the gradient variance is bounded as*

$$\text{Var}(\theta; \mathcal{H}) := \mathbb{E}_{h \in \mathcal{H}} [\|\nabla F_h(\theta) - \mathbb{E}_{h' \in H}[\nabla F_{h'}(\theta)]\|^2] \leq \frac{F(\theta)^2}{|\mathcal{H}|}.$$

Since all d k = e Θ(d) parities in P are pairwise orthogonal with respect to the uniform distribution Unif({±1} d ), it follows that the variance of ∇L¯ is exponentially small and the target signal can be drowned out by a correspondingly small noise from the oracle. However, this is not true for the empirical distribution which cannot distinguish all elements in P with only poly(d) samples; the empirical correlation of two random parities will generally be Θ(n −1/2 ). Therefore a more careful decorrelation argument is needed, resulting in the weaker guarantees of Theorem [2](#page-5-0)[\(2\).](#page-5-2) Another technical difference is that Theorem [1](#page-2-0) only considers the strong zero-one loss (more formally, their results can be seen to hold for any parity estimator pˆθ(A) ∈ P depending on the algorithm output), while we prove the L<sup>2</sup> lower bound for any real-valued estimator fθ(A) .

### 3.2 COT WITH TEACHER FORCING

When training with teacher forcing, at each position d + 1 ≤ m ≤ d + k − 1, the ground-truth labels of the preceding intermediate states x1, · · · , xm−<sup>1</sup> are fed into the transformer input to obtain the predictor xˆ<sup>m</sup> at the mth position,

$$\hat{\mathbf{x}}_m = \text{TF}(\mathbf{x}_1, \dots, \mathbf{x}_{m-1}, \mathbf{0}_n, \dots, \mathbf{0}_n; \mathbf{W})_m.$$

The loss function then computes the squared error over all states,

$$L(\mathbf{W}) = \frac{1}{2n} \sum_{m=d+1}^{d+k-1} \|\hat{\mathbf{x}}_m - \mathbf{x}_m\|^2. \quad (2)$$

Since each sequence of values xˆd+1,i, · · · , xˆd+k−1,i are generated depending only on the corresponding sample x i and the parameter matrix W, this can be rewritten in terms of the augmented labels y¯ <sup>i</sup> = (x i <sup>d</sup>+1, · · · , x<sup>i</sup> d+k−1 ) <sup>⊤</sup> as

$$L(\mathbf{W}) = \frac{1}{2n} \sum_{i=1}^n \|\bar{y}^i - f^\circ(\mathbf{x}^i; \mathbf{W})\|^2, \quad f_m^\circ(\mathbf{x}^i; \mathbf{W}) = \hat{x}_{m,i}, \quad d+1 \leq m \leq d+k-1$$

for a fixed mapping f ◦ : {±1} <sup>d</sup> × <sup>R</sup> (d+k−1)<sup>2</sup> → <sup>R</sup> k−1 , mirroring the setting of Theorem [2.](#page-5-0) Hence our model does not cross-reference between samples; moreover, the gradient of f ◦ is bounded as

Lemma 4. *For all* x,W *it holds uniformly that* ∥∇Wf ◦ (x;W)∥ ≤ O( √ d)*.*

At inference time, test inputs x1, · · · , x<sup>d</sup> are randomly generated and the prediction for ytest = p(x1, · · · , xd) is computed by iterating TF to generate all k − 1 reasoning steps without reference to ground-truth labels; yˆtest = TF(k−1)(x1, · · · , xd, 0n, · · · , 0n;W)d+k−1. Our positive learning result in this setting is as follows.

Theorem 5 (CoT with teacher forcing). *Suppose* n = Ω(d 2+ϵ ) *for* ϵ > 0*,* d *is sufficiently large and let* ∇e *be any* <sup>O</sup>(<sup>d</sup> −2−ϵ/8 )*-approximate gradient oracle.*[<sup>4</sup>](#page-6-1) *Set initialization* W(0) = 0 *and learning rate* η = Θ(d 2+ϵ/<sup>16</sup>)*. Then for any target parity* p ∈ P*, it holds with probability* 1 − exp(−d ϵ/2 ) *over random sampling that the one-step update* <sup>W</sup>(1) <sup>=</sup> <sup>W</sup>(0) − <sup>η</sup>∇eL(W(0)) *w.r.t. the objective* [\(2\)](#page-6-2) *with teacher forcing achieves loss* ∥yˆtest − ytest∥<sup>∞</sup> ≤ O(d −ϵ/8 )*.*

On the other hand, Theorem [2](#page-5-0)[\(2\)](#page-5-2) shows that when n = Ω(d 11+ϵ ), any iterative algorithm querying an O(d −2−ϵ/8 )-approximate oracle, with gradients bounded as in Lemma [4,](#page-6-3) requires more than Ω( e <sup>d</sup> ϵ/4 ) queries to attain a nontrivial (< 2 ) loss. This establishes a strict separation between learning parities without intermediate supervision and our CoT transformer. The gap increases with more samples as ϵ increases; moreover, when n = e Ω(d) , we have a much stronger separation by Theorem [2](#page-5-0)[\(1\),](#page-5-1) where an exponential number of queries is required to learn p.

*Sketch of proof.* The result is shown by explicitly calculating the gradient with respect to each weight wj,m and extracting the gradient signal. As the softmax scores are uniform at initialization, the gradient can be expanded to obtain multilinear contraction or 'interaction' terms between the tokens x1, · · · , xm−1, one such example being

$$\frac{1}{n} \langle \mathbf{x}_m, \hat{\mathbf{z}}_m, \hat{\mathbf{z}}_m \rangle = \frac{1}{n(m-1)^2} \sum_{\alpha, \beta} \langle \mathbf{x}_m, \mathbf{x}_\alpha, \mathbf{x}_\beta \rangle.$$

In the above equation, if α, β are the two child nodes of m, the parity xαxβx<sup>m</sup> ≡ 1 will be trivial and ⟨xm, xα, xβ⟩ = n. On the other hand, for nontrivial parities the interaction strength will generally be O( √ n log d) due to sample concentration. For sufficiently large n, the trivial parities dominate, allowing us to extract the leading term. Performing these computations up to fourth order interaction terms, we show that the dominating signal of the gradient is Θ(d −2 ) when j = c1[m], c2[m] and O(d −2−ϵ/8 ) otherwise. Hence the transformer learns to increase only the weights at the relevant positions for each subtask, and is able to compute the desired 2-parity xˆ<sup>m</sup> ≈ ϕ( 1 2 (xˆ<sup>c</sup>1[m] + xˆ<sup>c</sup>2[m])) ≈ xˆ<sup>c</sup>1[m]xˆ<sup>c</sup>2[m] at each node during its forward pass. The full proof is provided in Appendix [B.](#page-17-0)

<sup>4</sup> In fact, we only require that each component of the gradient has error at most O(d −2−ϵ/8 ) for Theorems [5,](#page-6-0) [7,](#page-8-0) which follows since the L<sup>∞</sup> error is bounded above by L2.

#### 3.3 COT WITHOUT TEACHER FORCING

In this section, we extend Theorem [5](#page-6-0) to training a transformer without teacher forcing, which is employed alongside teacher forcing in practice to ensure robustness at test time [\(Bengio et al.,](#page-10-10) [2015;](#page-10-10) [Goyal et al.,](#page-10-11) [2017;](#page-10-11) [Mihaylova & Martins,](#page-11-12) [2019\)](#page-11-12). The main difficulty in this setting is that wrong answers propagate to later generation steps, exponentially amplifying errors and drowning out the main gradient signals. Error accumulation is also a central practical issue of CoT [\(Zhang & Parkes,](#page-13-6) [2023;](#page-13-6) [Wang et al.,](#page-12-9) [2023\)](#page-12-9). To solve this issue, we make some modifications to our transformer model.

First, we minimize the number of required reasoning steps by imposing a slightly stronger form of autoregressivity where each intermediate state x + <sup>m</sup> depends on all tokens x + j , j = 1, · · · , dh[m]−<sup>1</sup> up to the previous level, rather than the immediately preceding token. This can be expressed as the causal mask wj,m ← −∞ for j > dh[m]−<sup>1</sup> or m ≤ d; see Figure [3.](#page-7-1) This ensures that the model gradients are polynomially bounded as in Theorem [2](#page-5-0) and that errors can propagate a logarithmic rather than a linear number of steps, and can be easily implemented as the indices d<sup>ℓ</sup> are known.

![](_page_7_Picture_4.jpeg)

![](_page_7_Diagram_5.jpeg)

Figure 3: Causal mask for W<sup>⊤</sup> with teacher forcing (left); without teacher forcing (right). The gray entries are set to −∞.

Second, we implement a data augmentation technique where random d-bit strings u <sup>i</sup> ∼ Unif({±1} d ), i ∈ [n ′ ] are appended to the original dataset (x i )i∈[n] . The resulting augmented tokens are denoted as x + <sup>j</sup> = (x ⊤ <sup>j</sup> u ⊤ j ) <sup>⊤</sup> ∈ <sup>R</sup> n+n , u<sup>j</sup> = (u i j ) n <sup>i</sup>=1 so that p<sup>j</sup> = ((x + j ) <sup>⊤</sup> e ⊤ j ) <sup>⊤</sup> (the notation is extended to j > d), and the key, query and value matrices are appropriately enlarged. The ground truth labels as well as the intermediate states for the augmented data are unknown, so they are not included in the loss function. Nevertheless, unlabeled data can still suffice for self-consistency [\(Huang et al.,](#page-10-6) [2023a\)](#page-10-6); their purpose is to filter for 'faulty reasoning' in the following sense. If the weights are not sufficiently trained, the output of a node x<sup>j</sup> will consist of all nearly −1s and thus be uninformative for computing any parities. If the augmented tokens newly generated in the previous iteration of TF(·) (i.e. up to u<sup>d</sup>ℓ−<sup>1</sup> ) are uninformative, we zero out its output on the basis that all subsequent reasoning will be wrong. This is achieved by adding the following filter after the feedforward layer ϕ:

$$\forall \mathbf{z}^+ \in \mathbb{R}^{n+n'}, \quad \iota_\ell(\mathbf{z}^+) = \begin{cases} \mathbf{0} & \|\mathbf{u}_j + \mathbf{1}\mathbf{n}'\|_\infty < \varepsilon_0 \text{ for any } d_{\ell-2} < j \leq d_{\ell-1}, \\ \mathbf{z}^+ & \text{otherwise.} \end{cases}$$

Without teacher forcing, during training the entire reasoning chain is generated by iteratively applying TF to its own output until convergence, which takes v = log<sup>2</sup> k rather than k − 1 steps due to the imposed block autoregressivity. Hence TF(v) (x + 1 , · · · , x + d , 0n+n′ , · · · ;W) = (xˆ + 1 , · · · , xˆ + d+k−1 ) where the tokens xˆ + <sup>d</sup>+1, · · · , xˆ + d+k−1 are recursively generated per level as

$$\hat{\mathbf{x}}_m^+ = \iota_{h[m]} \circ \phi(\hat{\mathbf{z}}_m^+), \quad \hat{\mathbf{z}}_m^+ = \sum_{j=1}^{d_{h[m]}-1} \sigma_j(\mathbf{w}_m) \hat{\mathbf{x}}_j^+. \quad (3)$$

The loss is computed against the ground-truth labels as in [\(2\)](#page-6-2). As before, each sequence of generated states depends only on each sample x i and the augmented data U = (u i )i∈[n′ , so we may express

$$L(\mathbf{W}, \mathbf{U}) = \frac{1}{2n} \sum_{i=1}^n \|\bar{y}^i - f^\times(\mathbf{x}^i; \mathbf{W}, \mathbf{U})\|^2, \quad f_m^\times(\mathbf{x}^i; \mathbf{W}, \mathbf{U}) = \hat{x}_{m,i} \quad (4)$$

for a fixed mapping f <sup>×</sup>, so that the samples are again not cross-referenced. By considering the propagation of gradients up the chain, the gradient of f <sup>×</sup> can be shown to be bounded as follows.

The exact exponent g depends on the shape of ϕ. Since ϕ(0) = −1 and ϕ(1) = 1, it must hold that ∥ϕ ′∥<sup>∞</sup> > 2. Conversely, any such ∥ϕ ′∥<sup>∞</sup> may be achieved by taking ϕ to be locally quadratic around 0, ±1 and smoothly joining the curve segments with straight lines of slope ±(2 + ϵ). Furthermore, such a link function can be realized by a simple shallow feedforward layer using e.g. O(1) ReQU neurons. Hence g can be taken to be arbitrarily close to 1.5.

Finally, we implement a simple weight quantization method by rounding each entry of W to the nearest integer after every update; <sup>W</sup>(t+1) <sup>=</sup> <sup>r</sup>[W(t) − <sup>η</sup>∇eWL(W(t) , U)], where r : <sup>R</sup> → <sup>Z</sup> is the nearest-integer operator. Equivalently, the gradients themselves are quantized. Integer-based quantization methods are widely used in practice to accelerate training and reduce memory usage [\(Wu et al.,](#page-12-13) [2020;](#page-12-13) [Jacob et al.,](#page-11-15) [2018\)](#page-11-15), and have been successfully implemented in LLMs to facilitate efficient fine-tuning [\(Dettmers et al.,](#page-10-14) [2022;](#page-10-14) [2023\)](#page-10-15). In our theoretical setting, quantization also allows us to simplify computations involving propagation of error.

In this setting, we obtain the following learning result.

Theorem 7 (CoT without teacher forcing). *Suppose* n = Ω(d 2+ϵ ) *for* ϵ > 0*,* n ′ = poly(d)*,* [<sup>5</sup>](#page-8-2) d *is sufficiently large and let* ∇e *be any* <sup>O</sup>(<sup>d</sup> −2−ϵ/8 )*-approximate gradient oracle. Set* W(0) = 0 *and* η = Θ(d 2+ϵ/<sup>16</sup>)*. Then for any target parity* p ∈ P*, it holds with probability* 1 − exp(−d (ϵ∧1)/2 ) *over random sampling of (original and augmented) data that the sequence of updates* W(t+1) = <sup>r</sup>[W(t) − <sup>η</sup>∇eL(W(t) , U)] *w.r.t. the objective* [\(4\)](#page-7-2) *without teacher forcing achieves exponentially small loss* ∥yˆtest − ytest∥<sup>∞</sup> ≤ exp(−Ω(d ϵ/<sup>16</sup>)) *in* log<sup>2</sup> k *iterations.*

This gives the same order of separation from Theorem [2](#page-5-0)[\(2\)](#page-5-2) as in Section [3.2.](#page-6-4) Hence transformers can learn parities even without teacher forcing, if the consistency of the chain of reasoning is suitably controlled for. Moreover, our result shows that logarithmic time suffices to learn parity by exploiting the hierarchical decomposition in Figure [1.](#page-3-1) This extends the circuit complexity result in [Merrill &](#page-11-8) [Sabharwal](#page-11-8) [\(2024\)](#page-11-8), which states that bounded-depth transformers with a logarithmic number of CoT steps can express problems in log-space; Theorem [7](#page-8-0) guarantees that transformers of depth one can *learn by gradient descent* any such function in the exponentially large class P.

*Sketch of proof.* The idea is to inductively show that each 2-parity subtask x<sup>m</sup> at level ℓ will become solved at time t = ℓ. When t ≤ ℓ − 2, x<sup>m</sup> cannot utilize its child nodes x<sup>c</sup>1[m] , x<sup>c</sup>2[m] since they will also not be optimized, so the weights do not change. At time ℓ − 1, its child nodes learn to output their parities with high precision, so the objective is approximately equivalent to that of Theorem [5.](#page-6-0) Then the gradient signal will similarly concentrate on w<sup>c</sup>1[m],m, w<sup>c</sup>2[m],m and x<sup>m</sup> will become solved in the next step. It remains to bound the gradients arising from the loss terms further down the chain xd+1 → · · · → xd+k−<sup>1</sup> (propagation of error), and verify that irrelevant weights wj,m (p[j] ̸= m) and already optimized weights do not change. The full proof is provided in Appendix [C.](#page-21-0)

# 4 NUMERICAL EXPERIMENTS

In this section, we present numerical experiments which support and complement our theoretical findings. Compared to the carefully calibrated step sizes and weight updates in Theorems [5](#page-6-0) and [7,](#page-8-0) these experiments study a more realistic training scenario by taking relatively small learning rates and tracking the loss trajectories over a longer period of training. We train one-layer transformers based on the architecture described in Section [2](#page-2-2) to solve a random k-parity problem with 64-bit inputs for k = 8, 16, 32. Specifically, we implement and compare the following four models.

- Direct: TF(·) is applied to itself k − 1 times to generate the reasoning chain end-to-end and the model prediction yˆ is directly compared to the ground truth y with the prediction loss <sup>1</sup> 2n ∥yˆ−y∥ 2 .
- CoT: TF(·) is applied to itself to generate the reasoning chain end-to-end and the sequence of intermediate states is compared to the ground truth as in [\(2\)](#page-6-2). Here, we also implement the causal mask in Figure [3](#page-7-1) (right) so that only log<sup>2</sup> k iterations are needed, for additional stability.
- CoT + teacher forcing: implements the model in Section [3.2](#page-6-4) with teacher forcing.
- CoT + self-consistency: implements the model in Section [3.3](#page-7-0) with the causal mask in Figure [3](#page-7-1) (right) and data augmentation for consistency checks. Weight quantization is omitted.

<sup>5</sup>Any polynomial order suffices for the number of augmented data samples.

![](_page_9_Figure_1.jpeg)

Figure 4: CoT loss (left) and prediction loss (right) curves for the four models when d = 64, k = 32. For the CoT+consistency model, dashed lines indicate when the filters of each level are deactivated.

All models are optimized using full-batch gradient descent on 100K 64-bit samples with a single Tesla T4 GPU. The three CoT models are trained with the 'CoT loss' [\(2\)](#page-6-2) scaled by <sup>1</sup> k−1 to match the prediction loss of the direct model. Figure [4](#page-9-0) shows training curves for the CoT loss (left) and the prediction loss (right) over 350 epochs when k = 32; results for all k and more details are provided in Appendix [D.](#page-25-0)

We first note that the direct model (red) completely fails to learn the target, plateauing almost immediately. We observed that the weights become nearly uniform so that yˆ ≈ 0<sup>n</sup> and the prediction error is stuck at 0.5. This was not improved by using a multilayer transformer instead of repeated composition. The basic CoT model (yellow) is able to significantly decrease CoT loss but fails to fully solve the problem and eventually becomes unstable. Moreover, the prediction loss never improves beyond 0.5. Indeed, due to the hierarchical structure of parity, the model has no chance of making an informative prediction at the last level xd+k−<sup>1</sup> unless all preceding levels have been fully solved. In contrast, we verify that CoT with teacher forcing (blue) solves parity efficiently as predicted in Section [3.2,](#page-6-4) even with a small learning rate. After a burn-in phase, the CoT loss steadily decreases to nearly zero, at which point the prediction loss also decreases rapidly as the final level is solved.

CoT with self-consistency (green) is also able to solve parity efficiently as predicted. Furthermore, the corresponding CoT loss curve clearly exhibits multiple learning stages. In the beginning, the model is essentially optimizing only the first level as subsequent outputs are zeroed out. After a short burn-in phase, the weights are optimized so that the softmax scores concentrate on the relevant nodes, at which point the CoT loss sharply decreases and the filters for the next level are deactivated, unlocking the next learning stage. This phased optimization repeats until all levels are fully solved and is crucial to arriving at the correct answer (in essence, teacher forcing is doing this at all levels simultaneously). Notably, a similar behavior seems to arise in the basic CoT model as well but fails due to accumulating error, further justifying the use of the filtering mechanism.

These results confirm that training explicitly for CoT generation can improve performance on multistep tasks, and that controlling error accumulation via teacher forcing or self-consistency is key to ensuring proper step-by-step learning.

## 5 CONCLUSION

In this paper, by focusing on the k-parity problem, we provide an initial theoretical foundation for training transformers with CoT to perform stepwise reasoning. Our results show that gradient-based learning of parity requires significant iterations without intermediate supervision, but task decomposition using teacher forcing enables efficient learning in a single gradient update. Furthermore, when transformers are trained to generate reasoning chains end-to-end, data augmentation and selfconsistency checks can enhance their ability to solve complex tasks. Our work takes the first steps towards understanding how CoT can be leveraged to improve multi-step reasoning capability of foundation models.

# ACKNOWLEDGMENTS

JK was partially supported by JST CREST (JPMJCR2015). TS was partially supported by JSPS KAKENHI (24K02905, 20H00576) and JST CREST (JPMJCR2115).

# REFERENCES


[1] Emmanuel Abbe and Colin Sandon. On the universality of deep learning. In *Advances in Neural Information Processing Systems*, 2020. Konstantine Arkoudas. GPT-4 Can't Reason. *arXiv preprint arXiv:2308.03762*, 2023. Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In *Advances in Neural Information Processing Systems*, 2015. Satwik Bhattamishra, Arkil Patel, Phil Blunsom, and Varun Kanade. Understanding in-context learning in transformers and LLMs by learning to learn discrete functions. In *International Conference on Learning Representations*, 2024. David Chiang, Peter Cholak, and Anand Pillay. Tighter bounds on the expressivity of transformer encoders. In *International Conference on Machine Learning*, 2023. Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. Navigate through enigmatic labyrinth: a survey of chain of thought reasoning: advances, frontiers and future. In *Association for Computational Linguistics*, 2024. Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. *arXiv preprint arXiv:2311.01460*, 2023. Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: efficient finetuning of quantized LLMs. In *Advances in Neural Information Processing Systems*, 2023. Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. In *Advances in Neural Information Processing Systems*, 2023. Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle Use a laptop? A question answering benchmark with implicit reasoning strategies. *Transactions of the Association for Computational Linguistics*, 9:346–361, 2021. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT Press, 2016. [http:](http://www.deeplearningbook.org) [//www.deeplearningbook.org](http://www.deeplearningbook.org). Kartik Goyal, Chris Dyer, and Taylor Berg-Kirkpatrick. Differentiable scheduled sampling for credit assignment. In *Association for Computational Linguistics*, 2017. Xinyang Hu, Fengzhuo Zhang, Siyu Chen, and Zhuoran Yang. Unveiling the statistical foundations of chain-of-thought prompting methods. *arXiv preprint arXiv:2408.14511*, 2024. Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 1051–1068, Singapore, December 2023a. Association for Computational Linguistics. Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of Transformers. *arXiv preprint arXiv:2310.05249*, 2023b.

[2] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2018. Juno Kim and Taiji Suzuki. Transformers learn nonlinear features in context: nonconvex mean-field dynamics on the attention landscape. In *International Conference on Machine Learning*, 2024. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In *Advances in Neural Information Processing Systems*, 2022. Hongkang Li, Meng Wang, Songtao Lu, Xiaodong Cui, and Pin-Yu Chen. How do nonlinear transformers acquire generalization-guaranteed CoT ability? In *High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning*, 2024a. Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, and Samet Oymak. Dissecting chain-of-thought: compositionality through in-context filtering and learning. In *Advances in Neural Information Processing Systems*, 2023. Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. In *International Conference on Learning Representations*, 2024b. Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In *International Conference on Learning Representations*, 2024. Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: a systematic survey of prompting methods in natural language processing. *ACM Comput. Surv.*, 55(9), January 2023. Arvind Mahankali, Tatsunori B. Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. *arXiv preprint arXiv:2307.03576*, 2023. William Merrill and Ashish Sabharwal. A logic for expressing log-precision transformers. In *Advances in Neural Information Processing Systems*, 2023. William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. In *International Conference on Learning Representations*, 2024. Tsvetomila Mihaylova and Andre F. T. Martins. Scheduled sampling for transformers. In ´ *Association for Computational Linguistics: Student Research Workshop*, 2019. Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: a survey, 2024. Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A comprehensive overview of large language models, 2024. Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: scratchpads for intermediate computation with language models. *arXiv preprint arXiv:2112.00114*, 2021. Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen. Reasoning with language model prompting: a survey. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 5368–5393, Toronto, Canada, July 2023. Association for Computational Linguistics.

[3] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: methods, analysis and insights from training Gopher. *arXiv preprint arXiv:2112.11446*, 2022. Ran Raz. Fast learning requires good memory: a time-space lower bound for parity learning. *J. ACM*, 66(1), 2018. Mansi Sakarvadia, Aswathy Ajith, Arham Khan, Daniel Grzenda, Nathaniel Hudson, Andre Bauer, ´ Kyle Chard, and Ian Foster. Memory injections: correcting multi-hop reasoning failures during inference in transformer-based language models. *arXiv preprint arXiv:2309.05605*, 2024. Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. *arXiv preprint arXiv:2402.09268*, 2024. Shai Shalev-Shwartz, Ohad Shamir, and Shaked Shammah. Failures of gradient-based deep learning. In *International Conference on Machine Learning*, 2017. Ohad Shamir. Distribution-specific hardness of learning neural networks. *Journal of Machine Learning Research*, 19, August 2018. Yijun Tian, Yikun Han, Xiusi Chen, Wei Wang, and Nitesh V. Chawla. TinyLLM: learning a small student from multiple large language models. *arXiv preprint arXiv:2402.04616*, 2024. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, 2017. Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, and Mi Zhang. Efficient large language models: a survey. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. Survey Certification. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In *International Conference on Learning Representations*, 2023. Zhiwei Wang, Yunji Wang, Zhongwang Zhang, Zhangchen Zhou, Hui Jin, Tianyang Hu, Jiacheng Sun, Zhenguo Li, Yaoyu Zhang, and Zhi-Qin John Xu. Towards understanding how transformer perform multi-step reasoning with matching operation. *arXiv preprint arXiv:2405.15302*, 2024. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems*, 2022. Noam Wies, Yoav Levine, and Amnon Shashua. Sub-task decomposition enables learning in sequence to sequence tasks. In *International Conference on Learning Representations*, 2023. Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev, and Paulius Micikevicius. Integer quantization for deep learning inference: principles and empirical evaluation. *arXiv preprint arXiv:2004.09602*, 2020.

[4] Zihan Yu, Liang He, Zhen Wu, Xinyu Dai, and Jiajun Chen. Towards better chain-of-thought prompting strategies: a survey, 2023. Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. STaR: bootstrapping reasoning with reasoning. In *Advances in Neural Information Processing Systems*, 2022. Hugh Zhang and David C. Parkes. Chain-of-thought reasoning is a policy improvement operator, 2023. Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained Transformers learn linear models in-context. *arXiv preprint arXiv:2306.09927*, 2023a. Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, and Hai Zhao. Igniting language intelligence: the hitchhiker's guide from chain-of-thought reasoning to language agents. *arXiv preprint arXiv:2311.11797*, 2023b. Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models, 2024. Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, and Stuart Russell. Towards a theoretical understanding of the 'reversal curse' via training dynamics. *arXiv preprint arXiv:2405.04669*, 2024.
# APPENDIX

# A PROOF OF THEOREM [2](#page-5-0)

Denote the empirical inner product on R d by ⟨f, g⟩<sup>n</sup> = 1 n P<sup>n</sup> <sup>i</sup>=1 f(x i )g(x i ) and the corresponding norm as ∥f∥ 2 <sup>n</sup> = ⟨f, f⟩n. We also write

$$L_{n,p}(\theta) = \frac{1}{2}\|p - f_\theta\|_n^2 = \frac{1}{2n} \sum_{i=1}^n (p(\mathbf{x}^i) - f_\theta(\mathbf{x}^i))^2$$

to emphasize the dependency of L<sup>n</sup> on p. Note that (d/k) <sup>k</sup> ≤ d k ≤ (ed/k) k so that |P| = e Θ(d) .

Bounding gradient variance. Consider the variance of the empirical gradient ∇Ln,p w.r.t. the target parity p:

$$\text{Var}_n(\theta; P) := \mathbb{E}_{p \in P} \left[ \|\nabla L_{n,p}(\theta) - \mathbb{E}_{p' \in P} [\nabla L_{n,p'}(\theta)]\|^2 \right].$$

We proceed to evaluate the magnitude of Varn(θ; P). For p, p′ ∈ P with p ̸= p ′ it holds that

$$\langle p, p' \rangle_n = \frac{1}{n} \sum_{i=1}^n \left( \prod_{j \in p} x_j^i \prod_{j' \in p'} x_{j'}^i \right) = \frac{1}{n} \sum_{i=1}^n \left( \prod_{j \in p \Delta p'} x_j^i \right).$$

Since Q <sup>j</sup>∈p∆p′ x i j is i.i.d. Unif({±1}) for fixed p, p′ , by applying a union bound over Hoeffding's inequality, it follows for δ := p 4d/n that

$$\Pr \left( \sup_{p \neq p'} |\langle p, p' \rangle_n| \geq \delta \right) \leq |P|(|P| - 1) \exp \left( -\frac{n\delta^2}{2} \right) \leq \binom{d}{k}^2 e^{-2d} \leq \left( \frac{2}{e} \right)^{2d}.$$

Then with probability at least 1 − e <sup>−</sup>Ω(d) over random sampling, every off-diagonal component of the Gram matrix G<sup>P</sup> := (⟨p, p′ ⟩n)p,p′∈<sup>P</sup> has magnitude at most δ, while the diagonal entries are equal to 1. By the Gershgorin circle theorem, the maximum eigenvalue of G<sup>P</sup> satisfies

$$|\lambda_{\max}(G_P) - 1| \leq (|P| - 1)\delta,$$

thus λmax(G<sup>P</sup> ) ≤ 2(1 ∨ |P|δ). This implies that P constitutes a partial frame for the empirical L 2 norm with the corresponding frame upper bound. More specifically, for f : R <sup>d</sup> → <sup>R</sup>, decompose

$$f = \sum_{p \in P} c_p \cdot p + f_0, \quad f_0 \in (\text{span } P)^\perp,$$

for some coefficient sequence c = (cp)p∈<sup>P</sup> . It follows that

$$\|f\|_n^2 \geq \|f - f_0\|_n^2 = \sum_{p, p' \in P} c_p c_{p'} \langle p, p' \rangle_n = \|G_P^{1/2} c\|^2$$

and

$$\sum_{p \in P} \langle f, p \rangle_n^2 = \sum_{p \in P} \left( \sum_{p' \in P} c_{p'} \langle p, p' \rangle \right)^2 = \|G_P c\|^2 \leq \lambda_{\max}(G_P) \|f\|_n^2.$$

Denoting D = dim Θ, we can therefore bound Varn(θ; P) as

$$\begin{aligned} \text{Var}_n(\theta; P) &= \inf_{\mu \in \mathbb{R}^D} \mathbb{E}_{p \in P} [\|\nabla L_{n,p}(\theta) - \mu\|^2] \\ &\leq \mathbb{E}_{p \in P} \left[ \left\| \frac{1}{n} \sum_{i=1}^n (f_\theta(\mathbf{x}^i) - p(\mathbf{x}^i)) \nabla f_\theta(\mathbf{x}^i) - \frac{1}{n} \sum_{i=1}^n f_\theta(\mathbf{x}^i) \nabla f_\theta(\mathbf{x}^i) \right\|^2 \right] \\ &= \mathbb{E}_{p \in P} \left[ \sum_{j=1}^D \langle \nabla_{\theta_j} f_\theta, p \rangle_n^2 \right] = \frac{1}{|P|} \sum_{p \in P} \sum_{j=1}^D \langle \nabla_{\theta_j} f_\theta, p \rangle_n^2 \end{aligned}$$

$$\begin{aligned} &\leq \sum_{j=1}^D \frac{\lambda_{\max}(G_P)}{|P|} \|\nabla_{\theta_j} f_\theta\|_n^2 \\ &\leq 2 \left( \frac{1}{|P|} \vee \sqrt{\frac{4d}{n}} \right) \sup_{\theta, \mathbf{x}} \|\nabla f_\theta(\mathbf{x})\|^2. \end{aligned}$$

Now by Chebyshev's inequality, for any ε > 0 it holds that

$$\Pr \left( \|\nabla L_{n,p}(\theta) - \mathbb{E}_{p' \in P}[\nabla L_{n,p'}(\theta)]\| > \varepsilon \right) \leq \frac{\text{Var}_n(\theta; P)}{\varepsilon^2}.$$

Constructing the oracle. As in [Shamir](#page-12-8) [\(2018\)](#page-12-8), we define the <sup>ε</sup>-approximate oracle ∇e as

$$\tilde{\nabla}_{L_{n,p}(\theta)} = \begin{cases} \mathbb{E}_{p' \in P}[\nabla_{L_{n,p'}(\theta)}] & \|\nabla_{L_{n,p}(\theta)} - \mathbb{E}_{p' \in P}[\nabla_{L_{n,p'}(\theta)}]\| \leq \varepsilon, \\ \nabla_{L_{n,p}(\theta)} & \text{otherwise.} \end{cases}$$

By union bounding, we see that during T steps the oracle always defaults to the mean gradient and does not reveal any information on the true parity p, with probability at least

$$\Pr(Q) \geq 1 - \frac{2T}{\varepsilon^2} \left( \frac{1}{|P|} \vee \sqrt{\frac{4d}{n}} \right) \sup_{\theta, \mathbf{x}} \|\nabla f_\theta(\mathbf{x})\|^2,$$

where Q ⊆ P denotes the corresponding subset of the hypothesis space. Note that the argument can be extended to any randomized algorithm and random initialization in a straightforward manner by lifting to the product probability space, and so we consider Q to be fixed. Then for any target parity p ∈ Q, the output θ(A) of the algorithm after T steps does not depend on p, so the predictor f = fθ(A) is also fixed.

Lower bounding the loss. We first remark that a simpler proof can be given for the sup norm error, which is enough to establish a separation. Consider arbitrary p, p′ ∈ P with p ̸= p ′ and let x ∈ {±1} <sup>d</sup> be such that p(x) ̸= p ′ (x), then

$$|p(\mathbf{x}) - f(\mathbf{x})| + |p'(\mathbf{x}) - f(\mathbf{x})| \geq |1 - f(\mathbf{x})| + |-1 - f(\mathbf{x})| \geq 2.$$

Now let σ : Q → Q be any automorphism of Q with no fixed points. The L<sup>∞</sup> error can be bounded below by restricting to the noninformative set Q as follows.

$$\begin{aligned}\mathbb{E}_{p \in P} \left[ \sup_{\mathbf{x}} |p(\mathbf{x}) - f_{\theta(\mathcal{A})}(\mathbf{x})| \right] &\geq \mathbb{E}_{p \in P} \left[ 1_{\{p \in Q\}} \sup_{\mathbf{x}} |p(\mathbf{x}) - f(\mathbf{x})| \right] \\ &= \frac{1}{2|P|} \sum_{p \in Q} \left( \sup_{\mathbf{x}} |p(\mathbf{x}) - f(\mathbf{x})| + \sup_{\mathbf{x}} |\sigma \circ p(\mathbf{x}) - f(\mathbf{x})| \right) \\ &\geq \frac{1}{2|P|} \cdot 2|Q| = \Pr(Q).\end{aligned}$$

For mean squared error, we similarly restrict to Q so that

$$\mathbb{E}_{p \in P, \mathbf{x}} [(p(\mathbf{x}) - f_{\theta(\mathcal{A})}(\mathbf{x}))^2] \geq \mathbb{E}_{p \in P, \mathbf{x}} [1_{\{p \in Q\}} (p(\mathbf{x}) - f(\mathbf{x}))^2] .$$

Since the range of p is contained in [−1, 1], the above loss will not increase when f is replaced by its clipped version ¯f(x) = (f(x) ∧ 1) ∨ (−1). Moreover, in Lemma [8](#page-16-0) (proved at the end of the section) we show that |<sup>E</sup>p∈<sup>P</sup> [p(x)]| ≤ e <sup>−</sup>Ω(d) holds with probability 1 − e <sup>−</sup>Ω(d) over the sample space of x, so that

$$|\mathbb{E}_{p \in P, \mathbf{x}} [p(\mathbf{x}) \bar{f}(\mathbf{x})]| \leq (1 - e^{-\Omega(d)}) \mathbb{E}_{p \in P} [p(\mathbf{x})] + e^{-\Omega(d)} \leq e^{-\Omega(d)}$$

and also

$$\begin{aligned} \mathbb{E}_{p \in P, \mathbf{x}} [1_{\{p \in Q\}} p(\mathbf{x}) \bar{f}(\mathbf{x})] &= \mathbb{E}_{p \in P, \mathbf{x}} [p(\mathbf{x}) \bar{f}(\mathbf{x})] - \mathbb{E}_{p \in P, \mathbf{x}} [1_{\{p \notin Q\}} p(\mathbf{x}) \bar{f}(\mathbf{x})] \\ &\leq e^{-\Omega(d)} + (1 - \Pr(Q)) \mathbb{E}_{\mathbf{x}} [|\bar{f}(\mathbf{x})|] \\ &\leq e^{-\Omega(d)} + \frac{(1 - \Pr(Q))^2}{2 \Pr(Q)} + \frac{\Pr(Q)}{2} \mathbb{E}_{\mathbf{x}} [\bar{f}(\mathbf{x})^2]. \end{aligned}$$

Therefore we may bound

$$\begin{aligned}\mathbb{E}_{p \in P, \mathbf{x}} [(p(\mathbf{x}) - f_{\theta(\mathcal{A})}(\mathbf{x}))^2] &\geq \mathbb{E}_{p \in P, \mathbf{x}} [1_{\{p \in Q\}} (p(\mathbf{x}) - \bar{f}(\mathbf{x}))^2] \\ &= \Pr(Q) - 2\mathbb{E}_{p \in P, \mathbf{x}} [1_{\{p \in Q\}} p(\mathbf{x}) \bar{f}(\mathbf{x})] + \Pr(Q) \cdot \mathbb{E}_{\mathbf{x}} [\bar{f}(\mathbf{x})^2] \\ &\geq \Pr(Q) - \frac{(1 - \Pr(Q))^2}{\Pr(Q)} - 2e^{-\Omega(d)} \\ &\geq 2 - \frac{1}{\Pr(Q)} - 2e^{-\Omega(d)} \\ &\geq 1 - \frac{4T}{\varepsilon^2} \left( \frac{1}{|P|} \vee \sqrt{\frac{4d}{n}} \right) \sup_{\theta, \mathbf{x}} \|\nabla f_{\theta}(\mathbf{x})\|^2 - 2e^{-\Omega(d)}, \quad (5)\end{aligned}$$

where we have used the inequality 2 − (1 − t) <sup>−</sup><sup>1</sup> ≥ 1 − 2t, valid for t ∈ [0, 2 ].

The proof is completed by evaluating the following cases.

- (1) If n = e Ω(d) and T, ∥∇fθ∥ = O(poly(d)), the gradient variance is bounded as Varn(θ; P) ≤ e −Ω(d) . By taking ε = Varn(θ; P) 1/3 , it follows that Pr(Q) = 1 − e −Ω(d) and [\(5\)](#page-16-1) yields the lower bound 1 − e −Ω(d) .
- (2) If n = Ω(d ν ), ∥∇fθ∥ = O(d <sup>ν</sup><sup>1</sup> ), ε = Θ(d <sup>−</sup>ν<sup>2</sup> ) and T = O(d <sup>ν</sup><sup>3</sup> ), the gradient variance is bounded as Varn(θ; P) ≤ O(d 2ν1+ν3+1/2−ν/2 ) = O(d <sup>−</sup>2ν2−ν<sup>4</sup> ) and [\(5\)](#page-16-1) yields the lower bound 1 − O(d <sup>−</sup>ν<sup>4</sup> ).

Lemma 8. *If* k = Θ(d)*, it holds with probability at least* 1 − e <sup>−</sup>Ω(d) *over random sampling that*

$$|\mathbb{E}_{p \in P}[p(\mathbf{x})]| \leq e^{-\Omega(d)}.$$

*Proof.* Let m denote the number of −1s in x. By the Chernoff bound for the binomial distribution,

$$\Pr \left( \left| m - \frac{d}{2} \right| \leq \frac{\delta d}{2} \right) \geq 1 - 2 \exp \left( -\frac{\delta^2 d}{6} \right)$$

for a constant δ ∈ (0, 1) to be determined, so we assume the above event throughout the proof. Moreover denoting the complement parity p <sup>c</sup> = [d] \ p, it holds that p(x) = x<sup>1</sup> · · · x<sup>d</sup> · p c (x) and |<sup>E</sup>p∈<sup>P</sup> [p(x)]| = |<sup>E</sup>p∈<sup>P</sup> [p c (x)]|, so it suffices to consider the case where 2k ≤ d.

Without loss of generality, we may assume that x = (−1, · · · , −1, 1, · · · , 1) so that p(x) is decided as (−1)<sup>|</sup>p∩[m]<sup>|</sup> . We bound the cardinality of the set P<sup>+</sup> := {p ∈ P | p(x) = 1}. Each parity in P<sup>+</sup> can be determined by choosing 2j elements from [m] and k − 2j elements from [d] \ [m]. Denoting by [t]<sup>j</sup> the *coefficient* of operator of order j, we can evaluate

$$\begin{aligned}
|P_+| &= \sum_{j=0}^{\lfloor m/2 \rfloor} \binom{m}{2j} \binom{d-m}{k-2j} \\
&= \sum_{j=0}^{\lfloor m/2 \rfloor} \binom{m}{2j} [t]_{k-2j} (1+t)^{d-m} = \sum_{j=0}^{\lfloor m/2 \rfloor} \binom{m}{2j} [t]_k (1+t)^{d-m} t^{2j} \\
&= [t]_k (1+t)^{d-m} \sum_{j=0}^{\lfloor m/2 \rfloor} \binom{m}{2j} t^{2j} = \frac{1}{2} [t]_k (1+t)^{d-m} ((1+t)^m + (1-t)^m) \\
&= \frac{1}{2} \binom{d}{k} + \frac{1}{2} [t]_k (1-t^2)^{m'} (1+st)^{d-2m'} \\
&= \frac{1}{2} \binom{d}{k} + \frac{s^k}{2} \sum_{j=0}^{\lfloor k/2 \rfloor} (-1)^j \binom{m'}{j} \binom{d-2m'}{k-2j},
\end{aligned}$$

where m′ = m ∧ (d − m) and s = ±1. It further follows that

$$\begin{aligned} \left| \frac{|P_+|}{|P|} - \frac{1}{2} \right| &\leq \frac{1}{2|P|} \sum_{j=0}^{\lfloor k/2 \rfloor} \binom{m'}{j} \binom{d-2m'}{k-2j} \leq \frac{1}{2|P|} \sum_{j=0}^{\lfloor k/2 \rfloor} \binom{\lfloor d/2 \rfloor}{j} \binom{\lfloor \delta d \rfloor}{k-2j} \\ &\leq \frac{\lfloor k/2 \rfloor}{2} \binom{d}{k}^{-1} \binom{\lfloor d/2 \rfloor}{\lfloor k/2 \rfloor} \binom{\lfloor \delta d \rfloor}{\lfloor \delta d/2 \rfloor} \leq \frac{d}{4} \binom{d-\lfloor d/2 \rfloor - \lfloor \delta d \rfloor}{k-\lfloor k/2 \rfloor - \lfloor \delta d/2 \rfloor}^{-1} \\ &\leq \frac{d}{4} \binom{\lfloor d/4 \rfloor}{\lfloor k/4 \rfloor}^{-1} \leq \frac{d}{4} \left( \frac{d}{k} \right)^{-k/4} = e^{-\Theta(d)}. \end{aligned}$$

Here, we have chosen δ = 4 ∧ k <sup>2</sup><sup>d</sup> = Θ(1) and used the inequality a1+a2+a<sup>3</sup> b1+b2+b<sup>3</sup> ≥ a<sup>1</sup> b<sup>1</sup> <sup>a</sup><sup>2</sup> b<sup>2</sup> <sup>a</sup><sup>3</sup> b<sup>3</sup> . From this, we conclude that

$$|\mathbb{E}_{p \in P}[p(\mathbf{x})]| = \left| \frac{|P \setminus P_+| - |P_+|}{|P|} \right| \leq e^{-\Omega(d)}$$

with probability 1 − e −Ω(d) .

# B PROOF OF THEOREM [5](#page-6-0)

Proof of Lemma [4.](#page-6-3) For each d + 1 ≤ m ≤ d + k − 1 and 1 ≤ j < m, the only component of f ◦ depending on wj,m is f ◦ <sup>m</sup> and

$$\begin{aligned} \left| \frac{\partial f_m^\circ(\mathbf{x}; \mathbf{W})}{\partial w_{j,m}} \right| &= |\phi'(\hat{z}_m)| \cdot \left| \frac{\partial \hat{z}_m}{\partial w_{j,m}} \right| \\ &\leq \|\phi'\|_\infty \left| \frac{\partial \sigma_j(\mathbf{w}_m)}{\partial w_{j,m}} x_j + \sum_{\alpha \neq j} \frac{\partial \sigma_\alpha(\mathbf{w}_m)}{\partial w_{j,m}} x_\alpha \right| \\ &= \|\phi'\|_\infty \left| \sigma_j(\mathbf{w}_m)(1 - \sigma_j(\mathbf{w}_m))x_j - \sigma_j(\mathbf{w}_m) \sum_{\alpha \neq j} \sigma_\alpha(\mathbf{w}_m)x_\alpha \right| \\ &\leq \|\phi'\|_\infty \sigma_j(\mathbf{w}_m)(1 - \sigma_j(\mathbf{w}_m)) + \|\phi'\|_\infty \sigma_j(\mathbf{w}_m) \sum_{\alpha \neq j} \sigma_\alpha(\mathbf{w}_m) \\ &\leq 2\|\phi'\|_\infty \sigma_j(\mathbf{w}_m). \end{aligned}$$

Hence it follows that

$$\sum_{m=d+1}^{d+k-1} \|\nabla_{\mathbf{w}} f_m^\circ\|^2 \leq 4\|\phi'\|_\infty^2 \sum_{m=d+1}^{d+k-1} \sum_{j=1}^{m-1} \sigma_j(\mathbf{w}_m)^2 \leq 4\|\phi'\|_\infty^2(k-1) = O(d),$$

as desired.

We say that a parity xj<sup>1</sup> · · · xj<sup>r</sup> for 1 ≤ j1, · · · , j<sup>r</sup> ≤ d + k − 1 is *trivial* if it always equals 1, or equivalently if its reduction to the independent bits x1, · · · , x<sup>d</sup> cancel out mod 2. For example, the parity x1x4x<sup>17</sup> in Figure [1](#page-3-1) is trivial. Define Ir,m as the set of nontrivial index r-tuples less than m:

$$I_{r,m} = \{(j_1, \dots, j_r) \mid 1 \leq j_1, \dots, j_r \leq m-1, x_{j_1} \cdots x_{j_r} \neq 1\}.$$

In particular, I1,m = [m − 1] since no single parity is trivial.

Lemma 9 (concentration of interaction terms). *If each bit* x i j *for* i ∈ [n]*,* j ∈ [d] *is i.i.d. generated from the uniform distribution on* {±1}*, for any* p > 0 *it holds with probability at least* 1 − p *that*

$$\max_{\substack{1 \leq r \leq 4 \\ (j_1, \dots, j_r) \in I_{r,m}}} \frac{|\langle \mathbf{x}_{j_1}, \dots, \mathbf{x}_{j_r} \rangle|}{n} \leq \kappa := \sqrt{\frac{2}{n} \log \frac{32d^4}{p}}.$$

*Proof.* Each tuple (j1, · · · , jr) ∈ Ir,m computes a specific nontrivial parity xj<sup>1</sup> · · · xj<sup>r</sup> for which the bits x i j1 · · · x i jr , i = 1, · · · , n are i.i.d. Unif({±1}) due to symmetry. By Hoeffding's inequality we have that

$$\Pr \left( |\langle \mathbf{x}_{j_1}, \dots, \mathbf{x}_{j_r} \rangle| \geq \lambda \right) \leq 2e^{-\lambda^2/2n}.$$

Moreover, |Ir,m| ≤ (d + k − 1)<sup>r</sup> ≤ (2d − 1)<sup>r</sup> so that

$$|I_{1,m}| + \cdots + |I_{4,m}| \leq (2d-1) + \cdots + (2d-1)^4 < (2d)^4.$$

Therefore it follows by union bounding that

$$\Pr \left( \max_{1 \leq r \leq 4, (j_1, \dots, j_r) \in I_{r,m}} |\langle \mathbf{x}_{j_1}, \dots, \mathbf{x}_{j_r} \rangle| \geq \lambda \right) \leq 32d^4 e^{-\lambda^2/2n},$$

which implies the statement.

In particular, we take n = Ω(d 2+ϵ ) and p = exp(−d ϵ/2 ) so that κ = O(d −1−ϵ/4 ). This will ensure that the informative gradient signals will dominate the irrelevant interaction terms.

We now proceed to the main proof of Theorem [5.](#page-6-0) The superscript (0) at initialization is omitted for simplicity. The loss can be written more explicitly as

$$L(\mathbf{W}) = \frac{1}{2n} \sum_{m=d+1}^{d+k-1} \|\phi(\hat{\mathbf{z}}_m) - \mathbf{x}_m\|^2, \quad \hat{\mathbf{z}}_m = \sum_{j=1}^{m-1} \sigma_j(\mathbf{w}_m) \mathbf{x}_j.$$

It is straightforward to verify for 1 ≤ α < m that

$$\frac{\partial \sigma_\alpha(\mathbf{w}_m)}{\partial w_{j,m}} = (\delta_{j\alpha} - \sigma_\alpha(\mathbf{w}_m))\sigma_j(\mathbf{w}_m) = (\delta_{j\alpha} - \sigma_j(\mathbf{w}_m))\sigma_\alpha(\mathbf{w}_m)$$

and

$$\frac{\partial \hat{z}_m}{\partial w_{j,m}} = \sum_{\alpha=1}^{m-1} (\delta_{j\alpha} - \sigma_j(\mathbf{w}_m)) \sigma_\alpha(\mathbf{w}_m) \mathbf{x}_\alpha = \sigma_j(\mathbf{w}_m) (\mathbf{x}_j - \hat{z}_m).$$

Then the gradient of L with respect to each element wj,m at initialization can be computed as

$$\begin{aligned} \frac{\partial L}{\partial w_{j,m}}(\mathbf{W}) &= \frac{1}{n} (\phi(\hat{\mathbf{z}}_m) - \mathbf{x}_m)^\top \frac{\partial \phi(\hat{\mathbf{z}}_m)}{\partial w_{j,m}} \\ &= \frac{\sigma_j(\mathbf{w}_m)}{n} \langle \phi(\hat{\mathbf{z}}_m) - \mathbf{x}_m, \phi'(\hat{\mathbf{z}}_m), \mathbf{x}_j - \hat{\mathbf{z}}_m \rangle \\ &= -\frac{1}{n(m-1)} \langle \mathbf{x}_m, 2c\hat{\mathbf{z}}_m, \mathbf{x}_j - \hat{\mathbf{z}}_m \rangle \end{aligned} \quad (6)$$
(7)

$$+ \frac{1}{n(m-1)} \langle -\mathbf{1}_n + c\hat{z}_m^2, 2c\hat{z}_m, \mathbf{x}_j - \hat{z}_m \rangle \quad (8)$$

$$+ \frac{1}{n(m-1)} \langle O(|\hat{z}_m|^4), 2c\hat{z}_m, \mathbf{x}_j - \hat{z}_m \rangle \quad (9)$$

$$+ \frac{1}{n(m-1)} \langle \phi(\hat{z}_m) - \mathbf{x}_m, O(|\hat{z}_m|^3), \mathbf{x}_j - \hat{z}_m \rangle. \quad (10)$$

Computing interaction strengths. The term [\(7\)](#page-18-0) will be shown to contain the dominating gradient signal when j = c1[m], c2[m], while the other terms can be bounded as perturbations. Let ℓ = h2[m] so that x<sup>m</sup> computes a 2 ℓ -parity.

For term [\(7\)](#page-18-0), we substitute zˆ<sup>m</sup> = m−1 P <sup>α</sup> x<sup>α</sup> at initialization to expand

$$\frac{1}{n} \langle \mathbf{x}_m, \hat{\mathbf{z}}_m, \mathbf{x}_j - \hat{\mathbf{z}}_m \rangle = \frac{1}{n(m-1)} \sum_{\alpha} \langle \mathbf{x}_m, \mathbf{x}_{\alpha}, \mathbf{x}_j \rangle - \frac{1}{n(m-1)^2} \sum_{\alpha, \beta} \langle \mathbf{x}_m, \mathbf{x}_{\alpha}, \mathbf{x}_{\beta} \rangle,$$

where the dummy indices α, β, · · · are taken to run over [m − 1]. Let us evaluate the third-order interaction terms ⟨xm, xα, xβ⟩. If h[α] = ℓ, xmx<sup>α</sup> computes the parity of 2 <sup>ℓ</sup>+1 independent bits from x1, · · · , x<sup>d</sup> so xmxαx<sup>β</sup> cannot be trivial, hence (m, α, β) ∈ I3,m and |⟨xm, xα, xβ⟩| ≤ nκ by Lemma [9.](#page-17-1) Similarly, h[β] = ℓ implies that (m, α, β) ∈ I3,m. Suppose h[α], h[β] ≤ ℓ − 1; unless h[α] = h[β] = ℓ − 1, the combined parity xαx<sup>β</sup> will not contain enough independent bits to cancel out the 2 <sup>ℓ</sup> bits in xm, so again (m, α, β) ∈ I3,m. Moreover if h[α] = h[β] = ℓ − 1, xmxαx<sup>β</sup> will be trivial if and only if {α, β} = {c1[m], c2[m]}, in which case ⟨xm, xα, xβ⟩ = n. Thus we have that

$$\frac{1}{n} \sum_{\alpha} \langle \mathbf{x}_m, \mathbf{x}_{\alpha}, \mathbf{x}_{\beta} \rangle = 2 + \frac{1}{n} \sum_{(m, \alpha, \beta) \in I_{3,m}} \langle \mathbf{x}_m, \mathbf{x}_{\alpha}, \mathbf{x}_{\beta} \rangle = 2 + O((m-1)^2 \kappa).$$

Similarly, the contraction ⟨xm, xα, x<sup>j</sup> ⟩ can be nontrivial only if p[j] = m and only when α is the other child node of xm, so that

$$\frac{1}{n} \sum_{\alpha} \langle \mathbf{x}_m, \mathbf{x}_{\alpha}, \mathbf{x}_j \rangle = \begin{cases} 1 + O((m-1)\kappa) & [p[j] = m, \\ O((m-1)\kappa) & \text{otherwise.} \end{cases}$$

Since κ = O(d −1−ϵ/4 ) and d < m ≤ 2d − 1, we can therefore isolate the leading term of order Θ(d −2 ) as

$$\begin{aligned} & -\frac{1}{n(m-1)} \langle \mathbf{x}_m, 2c\hat{\mathbf{z}}_m, \mathbf{x}_j - \hat{\mathbf{z}}_m \rangle \\ &= -\frac{2c}{(m-1)^2} (1_{\{\mathbf{p}[j]=m\}} + O(d\kappa)) + \frac{2c}{(m-1)^3} (2 + O(d^2\kappa)) \\ &= -\frac{2c}{(m-1)^2} 1_{\{\mathbf{p}[j]=m\}} + O(d^{-2-\epsilon/4}). \end{aligned}$$

Next, for term [\(8\)](#page-18-1), we expand

$$\frac{1}{n} \langle -\mathbf{1}_n + c\hat{z}_m^2, 2c\hat{z}_m, \mathbf{x}_j - \hat{z}_m \rangle = -\frac{2c}{n} \langle \hat{z}_m, \mathbf{x}_j \rangle + \frac{2c}{n} \langle \hat{z}_m^2 \rangle + \frac{2c^2}{n} \langle \hat{z}_m^3, \mathbf{x}_j \rangle - \frac{2c^2}{n} \langle \hat{z}_m^4 \rangle.$$

The second-order terms can be computed as

$$\frac{1}{n} \langle \hat{z}_m, \mathbf{x}_j \rangle = \frac{1}{n(m-1)} \left( \langle \mathbf{x}_j, \mathbf{x}_j \rangle + \sum_{\alpha \neq j} \langle \mathbf{x}_\alpha, \mathbf{x}_j \rangle \right) = \frac{1}{m-1} + O(\kappa),$$

$$\frac{1}{n} \langle \hat{z}_m^2 \rangle = \frac{1}{n(m-1)^2} \left( \sum_{\alpha} \langle \mathbf{x}_\alpha, \mathbf{x}_\alpha \rangle + \sum_{\alpha \neq \beta} \langle \mathbf{x}_\alpha, \mathbf{x}_\beta \rangle \right) = \frac{1}{m-1} + O(\kappa).$$

We evaluate the fourth-order interaction terms by looking at when (α, β, γ, δ) ∈/ I4,m can occur. Without loss of generality, suppose h[α] ≤ h[β] ≤ h[γ] ≤ h[δ].

- (i) If h[β] < h[γ] < h[δ], the parities of xα, xβ, x<sup>γ</sup> must combine without overlaps to cancel out xδ, so it must hold that x<sup>γ</sup> is a child of x<sup>δ</sup> and xα, x<sup>β</sup> are the two children of the other child. This subtree is fully determined by the choice of the index δ and one of its child nodes, so there are at most O(d) trivial 4-tuples in this case.
- (ii) If h[β] = h[γ] < h[δ], it still must hold that h[γ] = h[δ] − 1. Moreover, both xβ, x<sup>γ</sup> must be children of xδ; otherwise, the bits of x<sup>δ</sup> and the non-child node cannot be canceled out by the remaining nodes. Then either x<sup>β</sup> = x<sup>γ</sup> or xβx<sup>γ</sup> = xδ, and in both cases we see that xαxβxγx<sup>δ</sup> cannot be trivial.
- (iii) If h[β] < h[γ] = h[δ], it must be that γ = δ, otherwise the bits of xγx<sup>δ</sup> cannot be canceled out by xαxβ. It follows that xαx<sup>β</sup> ≡ 1 and α = β, so there are O(d 2 ) trivial 4-tuples in this case.
- (iv) If h[β] = h[γ] = h[δ], it must again hold that two indices must be equal, and the remaining two indices must also be equal, so there are also O(d 2 ) trivial 4-tuples.

$$\frac{1}{n} \langle \hat{\mathbf{z}}_m^4 \rangle = \frac{1}{n(m-1)^4} \sum_{\alpha, \beta, \gamma, \delta} \langle \mathbf{x}_\alpha, \mathbf{x}_\beta, \mathbf{x}_\gamma, \mathbf{x}_\delta \rangle$$

$$\begin{aligned} &= \frac{1}{n(m-1)^4} \sum_{(\alpha, \beta, \gamma, \delta) \notin I_{4,m}} n + \frac{1}{n(m-1)^4} \sum_{(\alpha, \beta, \gamma, \delta) \in I_{4,m}} O(n\kappa) \\ &= \frac{|[m-1]^4 \setminus I_{4,m}|}{(m-1)^4} + \frac{|I_{4,m}|}{(m-1)^4} O(\kappa) = O(d^{-2} + \kappa). \end{aligned}$$

Furthermore, suppose α, β, γ, δ are constrained to contain the index j. Then case (i) above counts O(1) nontrivial tuples, while case (i), while cases (iii),(iv) count at most O(d) tuples since there is only one free index to be determined. Hence we also have

$$\frac{1}{n} \langle \hat{z}_m^3, \mathbf{x}_j \rangle = \frac{1}{n(m-1)^3} \sum_{\alpha, \beta, \gamma} \langle \mathbf{x}_\alpha, \mathbf{x}_\beta, \mathbf{x}_\gamma, \mathbf{x}_j \rangle = \frac{O(d)}{(m-1)^3} + O(\kappa) = O(d^{-2} + \kappa).$$

Combining the above, we obtain that

$$\frac{1}{n(m-1)} \langle -\mathbf{1}_n + c\hat{z}_m^2, 2c\hat{z}_m, \mathbf{x}_j - \hat{z}_m \rangle = -\frac{2c}{(m-1)^2} + \frac{2c}{(m-1)^2} + \frac{O(\kappa)}{m-1} = O(d^{-2-\epsilon/4}).$$

For term [\(9\)](#page-18-2), we note that |zˆm| 4 = zˆ 4 m = O(nd−<sup>2</sup> + nκ) as derived above. Then since each component of zˆm, x<sup>j</sup> − zˆ<sup>m</sup> are contained in [−1, 1], [−2, 2], respectively, we have that

$$\frac{1}{n(m-1)} \langle O(|\hat{z}_m|^4), 2c\hat{z}_m, \mathbf{x}_j - \hat{z}_m \rangle = \frac{4c}{n(m-1)} O(\langle |\hat{z}_m|^4 \rangle) = O(d^{-2-\epsilon/4}).$$

Finally for term [\(10\)](#page-18-3), by the Cauchy-Schwarz inequality we have

$$\begin{aligned} \frac{1}{n} \langle |\hat{z}_m|^3 \rangle &= \frac{1}{n} \sum_{i=1}^n |\hat{z}_{m,i}|^3 \\ &\leq \frac{1}{n} \left( \sum_{i=1}^n \hat{z}_{m,i}^2 \right)^{1/2} \left( \sum_{i=1}^n \hat{z}_{m,i}^4 \right)^{1/2} = \frac{1}{n} \langle \hat{z}_m^2 \rangle^{1/2} \langle \hat{z}_m^4 \rangle^{1/2} \\ &= \frac{1}{n} O(nd^{-1})^{1/2} \cdot O(nd^{-2} + n\kappa)^{1/2} = O(d^{-1-\epsilon/8}), \end{aligned}$$

and so we may bound

$$\frac{1}{n(m-1)} \langle \phi(\hat{z}_m) - \mathbf{x}_m, O(|\hat{z}_m|^3), \mathbf{x}_j - \hat{z}_m \rangle = \frac{4}{n(m-1)} O(\langle |\hat{z}_m|^3 \rangle) = O(d^{-2-\epsilon/8}).$$

From [\(7\)](#page-18-0)-[\(10\)](#page-18-3) we conclude that

$$\frac{\partial L}{\partial w_{j,m}}(\mathbf{W}) = -\frac{2c}{(m-1)^2} 1_{\{\mathbf{p}[j]=m\}} + O(d^{-2-\epsilon/8}),$$

and the same result applies to the approximate gradient ∇e <sup>w</sup>j,m<sup>L</sup> at initialization since the cutoff does not apply and each component of the noise is bounded by O(d −2−ϵ/8 ).

Concentration of softmax scores. Taking η = d 2+ϵ/<sup>16</sup>, the updated weights <sup>W</sup>(1) <sup>=</sup> −η∇eL(W) become

$$w_{j,m}^{(1)} = \frac{2cd^{2+\epsilon/16}}{(m-1)^2} 1_{\{\mathfrak{p}[j]=m\}} + O(d^{-\epsilon/16}).$$

In particular, for each j ̸= c1[m], c2[m] the softmax scores satisfy

$$\sigma_j(\mathbf{w}_m^{(1)}) = e^{w_{j,m}^{(1)}} / \sum_{\alpha} e^{w_{\alpha,m}^{(1)}} \leq e^{w_{j,m}^{(1)} - w_{\mathbf{c}_1[m],m}^{(1)}} \leq \exp(-\Omega(d^{\epsilon/16})).$$

As softmax scores must sum to 1, it holds that σ<sup>c</sup>1[m](w (1) <sup>m</sup> ) + σ<sup>c</sup>2[m](w (1) <sup>m</sup> ) ≥ 1 − exp(−Ω(d ϵ/<sup>16</sup>)) and moreover

$$\frac{\sigma_{c_1[m]}(w_m^{(1)})}{\sigma_{c_2[m]}(w_m^{(1)})} = e^{w_{c_1[m],m}^{(1)} - w_{c_2[m],m}^{(1)}} \leq \exp(O(d^{-\epsilon/16})) \leq 1 + O(d^{-\epsilon/16})$$

from the inequality e <sup>t</sup> ≤ 1 + O(t) for small t > 0. By symmetry, σ<sup>c</sup>2[m](w (1) <sup>m</sup> )/σ<sup>c</sup>1[m](w (1) <sup>m</sup> ) ≤ 1 + O(d <sup>−</sup>ϵ/<sup>16</sup>). By simple algebraic manipulation, we can conclude that

$$\frac{1}{2} - O(d^{-\epsilon/16}) \leq \sigma_{c_1[m]}(w_m^{(1)}), \sigma_{c_2[m]}(w_m^{(1)}) \leq \frac{1}{2} + O(d^{-\epsilon/16}).$$

That is, the updated attention layer zˆ (1) <sup>m</sup> = P j σ<sup>j</sup> (w (1) <sup>m</sup> )x<sup>j</sup> has learned to take the average of the two child nodes and ignore the remaining input tokens at each step.

Evaluating the forward pass. Now to bound the updated prediction loss, we evaluate the error ∥xˆ (1) <sup>m</sup> − xm∥<sup>∞</sup> of each step of the forward pass for d + 1 ≤ m ≤ d + k − 1. More precisely, define the increasing sequence

$$\epsilon_m = \max_{d < j \leq m} \left\| \hat{\mathbf{x}}_j^{(1)} - \mathbf{x}_j \right\|_{\infty}, \quad \epsilon_d = 0.$$

Then

$$\left\| \hat{\mathbf{x}}_{c_1[m]}^{(1)} - \mathbf{x}_{c_1[m]} \right\|_{\infty}, \left\| \hat{\mathbf{x}}_{c_2[m]}^{(1)} - \mathbf{x}_{c_1[m]} \right\|_{\infty} \leq \epsilon_{c_1[m]}, \epsilon_{c_2[m]} \leq \epsilon_{m-1},$$

and for the intermediate values zˆ (1) <sup>m</sup> we have

$$\begin{aligned} \left\| \hat{\mathbf{z}}_m^{(1)} - \frac{\mathbf{x}_{c_1[m]} + \mathbf{x}_{c_2[m]}}{2} \right\|_{\infty} &\leq \left\| \hat{\mathbf{z}}_m^{(1)} - \frac{\hat{\mathbf{x}}_{c_1[m]}^{(1)} + \hat{\mathbf{x}}_{c_2[m]}^{(1)}}{2} \right\|_{\infty} + \epsilon_{m-1} \\ &\leq \sum_{\mathbf{p}[j] \neq m} \sigma_j(\mathbf{w}_m^{(1)}) + \left| \sigma_{c_1[m]}(\mathbf{w}_m^{(1)}) - \frac{1}{2} \right| + \left| \sigma_{c_2[m]}(\mathbf{w}_m^{(1)}) - \frac{1}{2} \right| + \epsilon_{m-1} \\ &\leq 2d \exp(-\Omega(d^{\epsilon/16})) + O(d^{-\epsilon/16}) + \epsilon_{m-1} \\ &\leq C_1 d^{-\epsilon/16} + \epsilon_{m-1}, \end{aligned}$$

for some constant C<sup>1</sup> > 0. Since ϕ behaves like a quadratic near 0, ±1, it follows that

$$\epsilon_m = \|\hat{\mathbf{x}}_m^{(1)} - \mathbf{x}_m\|_{\infty} = \left\| \phi(\hat{\mathbf{z}}_m^{(1)}) - \phi\left(\frac{\mathbf{x}_{c1[m]} + \mathbf{x}_{c2[m]}}{2}\right) \right\|_{\infty} \leq C_2(C_1 d^{-\epsilon/16} + \epsilon_{m-1})^2$$

for some constant C<sup>2</sup> > 0 depending only on ϕ. Then for sufficiently large d, by choosing C<sup>3</sup> such that C2(C<sup>1</sup> + C3d <sup>−</sup>ϵ/<sup>16</sup>) <sup>2</sup> ≤ C3, for ϵm−<sup>1</sup> ≤ C3d −ϵ/8 it follows that

$$\epsilon_m \leq C_2(C_1 d^{-\epsilon/16} + C_3 d^{-\epsilon/8})^2 \leq C_3 d^{-\epsilon/8},$$

thus ϵ<sup>m</sup> = O(d −ϵ/8 ) inductively for all m. We conclude that ∥yˆ − y∥<sup>∞</sup> = ∥xˆ (1) <sup>d</sup>+k−<sup>1</sup> − x (1) d+k−1 ∥<sup>∞</sup> is bounded for all inputs as O(d −ϵ/8 ).

# C PROOF OF THEOREM [7](#page-8-0)

Proof of Lemma [6.](#page-7-3) For the iterative generation scheme [\(3\)](#page-7-4), each wj,m affects xˆ<sup>m</sup> as well as all nodes xˆ<sup>α</sup> on higher levels h[α] > h[m] through xˆm. We bound the contribution of each term to the total gradient inductively with respect to the level. Define for each d < m ≤ d+k−1, 1 ≤ j ≤ m−1 and 0 < ℓ ≤ v the quantity

$$\xi_{j,m,\ell} := \max_{\alpha \leq d_\ell} \left| \frac{\partial \hat{x}_\alpha}{\partial w_{j,m}} \right|.$$

We denote κ := ∥ϕ ′∥<sup>∞</sup> for brevity. Clearly ξj,m,ℓ = 0 for ℓ < h[m] and

$$\xi_{j,m,h[m]} = \left| \frac{\partial \hat{x}_m}{\partial w_{j,m}} \right| \leq \kappa \sigma_j(\mathbf{w}_m) |x_j - \hat{z}_m| \leq 2\kappa \sigma_j(\mathbf{w}_m).$$

Moreover for any α with h[α] = ℓ > h[m], we can bound by the chain rule

$$\left| \frac{\partial \hat{x}_\alpha}{\partial w_{j,m}} \right| \leq |\phi'(\hat{z}_\alpha)| \cdot \left| \sum_{\beta=d_{h[m]}+1}^{d_{\ell-1}} \sigma_\beta(\mathbf{w}_\alpha) \frac{\partial \hat{x}_\beta}{\partial w_{j,m}} \right| \leq \kappa \xi_{j,m,\ell-1},$$

yielding the relation ξj,m,ℓ ≤ κξj,m,ℓ−1. Iterating, we obtain that ξj,m,ℓ ≤ 2κ <sup>ℓ</sup>−h[m]+1σ<sup>j</sup> (wm). Therefore we can bound the total gradient by the following.

$$\begin{aligned} \sum_{\alpha=d+1}^{d+k-1} \|\nabla_{\mathbf{w}} f_{\alpha}^{\times}\|^2 &= \sum_{\ell=1}^v \sum_{\alpha=d_{\ell-1}+1}^{d_{\ell}} \|\nabla_{\mathbf{w}} f_{\alpha}^{\times}\|^2 \\ &= \sum_{\ell=1}^v \sum_{\alpha=d_{\ell-1}+1}^{d_{\ell}} \sum_{m=d+1}^{d+k-1} \sum_{j=1}^{m-1} \left| \frac{\partial \hat{x}_{\alpha}}{\partial w_{j,m}} \right|^2 \\ &\leq \sum_{m=d+1}^{d+k-1} \sum_{j=1}^{m-1} \sum_{\ell=1}^v (d_{\ell} - d_{\ell-1}) \xi_{j,m,\ell}^2 \\ &\leq \sum_{m=d+1}^{d+k-1} \sum_{j=1}^{m-1} \sigma_j(\mathbf{w}_m)^2 \sum_{\ell=1}^v 2^{v-\ell} \cdot 4\kappa^{2\ell-2h[m]+2} \\ &\leq 4 \sum_{m=d+1}^{d+k-1} 2^v \kappa^{-2h[m]+2} \sum_{\ell=1}^v \left( \frac{\kappa^2}{2} \right)^{\ell} \leq \frac{4\kappa^2}{\kappa^2 - 2} \sum_{m=d+1}^{d+k-1} \kappa^{2v-2h[m]+2} \\ &\leq \frac{4\kappa^2}{\kappa^2 - 2} \sum_{\ell=1}^v (d_{\ell} - d_{\ell-1}) \kappa^{2v-2\ell+2} = \frac{4\kappa^4}{\kappa^2 - 2} \sum_{\ell=1}^v (2\kappa^2)^{v-\ell} \\ &\leq \frac{4\kappa^2}{(\kappa^2 - 2)(2\kappa^2 - 1)} (2\kappa^2)^v = O(d^{2\log_2 \kappa + 1}), \end{aligned}$$

since 2 <sup>v</sup> = k = O(d).

We first provide a concentration bound for the augmented data, which we take to hold throughout the proof by conditioning on the high probability event.

Lemma 10 (concentration of augmented data). *For* n ′ = poly(d)*, with probability* 1 − e −Ω(√ d) *over random sampling of the augmented data* u1, · · · ,ud*, it holds that* ∥u<sup>j</sup> + 1n′∥<sup>∞</sup> = 2 *for all* 1 ≤ j ≤ d + k − 1 *and*

$$\max_{0 \leq \ell \leq v} \left\| \frac{1}{d_\ell} \sum_{j=1}^{d_\ell} \mathbf{x}_j^+ \right\|_\infty \leq O(d^{-1/4}).$$

*Proof.* The nodes x<sup>d</sup>ℓ−1+1, · · · , x<sup>d</sup><sup>ℓ</sup> at each level ℓ compute independent parities, even though parities at different levels can be correlated. By Hoeffding's inequality and union bounding over coordinates, it follows that

$$\left\| \sum_{j=d_{\ell-1}+1}^{d_\ell} \mathbf{x}_j^+ \right\|_\infty \leq \sqrt{2(d_\ell - d_{\ell-1}) \log \frac{2n'}{p}} = 2^{\frac{v-\ell+1}{2}} \sqrt{\log \frac{2n'}{p}}$$

∞ with probability at least 1 − p. Again union bounding, the above holds for all levels 0 ≤ ℓ ≤ v simultaneously with probability at least 1 − vp, so that

$$\left\| \frac{1}{d_\ell} \sum_{j=1}^{d_\ell} \mathbf{x}_j^+ \right\|_\infty \leq \sum_{\ell'=0}^\ell \frac{2^{\frac{v-\ell'+1}{2}}}{d} \sqrt{\log \frac{2n'}{p}} \leq 2(\sqrt{2}+1) \sqrt{\frac{1}{d} \log \frac{2n'}{p}} = O(d^{-1/4})$$

for all ℓ if p = e − √ d . In addition, the probability that u<sup>j</sup> = −1n′ for some j ≤ d + k − 1 is bounded by 2d · 2 −n = e −Ω(d) ; otherwise, at least one entry is equal to 2.

Now to prove Theorem [7,](#page-8-0) we show by induction that with high probability, the weights can be written for constants C<sup>ℓ</sup> = Θ(1) as

$$w_{j,m}^{(t)} = \begin{cases} r[C_{h[m]}d^{\epsilon/16}] & h[m] \leq t, p[j] = m, \\ -\infty & j > d_{h[m]-1} \text{ or } m \leq d, \\ 0 & \text{otherwise.} \end{cases} \quad (11)$$

Evaluating the forward pass. We first evaluate the forward pass iteration of the transformer up to level h[m] ≤ t; fixing 0 < C < minℓ≤<sup>v</sup> Cℓ, it holds that

$$\sigma_j(\mathbf{w}_m^{(t)}) \leq \frac{1}{\exp(w_{c_{1[m],m}^{(t)}}) + \exp(w_{c_{2[m],m}^{(t)}}) + d_{h[m]-1} - 2} \leq \exp(-Cd^{\epsilon/16})$$

when p[j] ̸= m and

$$\frac{1 - d \exp(-C d^{\epsilon/16})}{2} \leq \sigma_{c_1[m]}(\mathbf{w}_m^{(t)}), \sigma_{c_2[m]}(\mathbf{w}_m^{(t)}) \leq \frac{1}{2}.$$

For the augmented tokens, define the increasing per-level error sequence

$$\epsilon_\ell = \max_{d < j \leq d_\ell} \left\| \hat{\mathbf{x}}_j^{+(t)} - \mathbf{x}_j^+ \right\|_\infty, \quad \epsilon_0 = 0.$$

We recursively bound ϵ<sup>ℓ</sup> as before up to ϵt; this will simultaneously verify that the filter ι is not applied for the first t + 1 levels since ∥u (t) <sup>j</sup> + 1n′∥<sup>∞</sup> ≥ 2 − ϵ<sup>t</sup> due to Lemma [10.](#page-22-1)

For each state zˆ +(t) <sup>m</sup> with h[m] = ℓ we have

$$\begin{aligned} & \left\| \hat{\mathbf{z}}_m^{+(t)} - \frac{\mathbf{x}_{c_1[m]}^+ + \mathbf{x}_{c_2[m]}^+}{2} \right\|_{\infty} \\ & \leq \sum_{\mathbf{p}[j] \neq m} \sigma_j(\mathbf{w}_m^{(t)}) + \left| \sigma_{c_1[m]}(\mathbf{w}_m^{(t)}) - \frac{1}{2} \right| + \left| \sigma_{c_2[m]}(\mathbf{w}_m^{(t)}) - \frac{1}{2} \right| + \epsilon_{\ell-1} \\ & \leq (2d - 2) \exp(-Cd^{\epsilon/16}) + \epsilon_{\ell-1}. \end{aligned}$$

Since ϕ behaves like a quadratic near 0, ±1, it follows that

$$\epsilon_\ell \leq C_2((2d-2) \exp(-Cd^{\epsilon/16}) + \epsilon_{\ell-1})^2,$$

and we can inductively verify ϵ<sup>ℓ</sup> ≤ exp(−Cdϵ/<sup>16</sup>) as well as ∥zˆ +(t) <sup>m</sup> − z + <sup>m</sup>∥<sup>∞</sup> ≤ <sup>2</sup><sup>d</sup> exp(−Cdϵ/<sup>16</sup>) holds for all ℓ ≤ t for sufficiently large d.

On the other hand, for the forward pass for levels h[m] > t the softmax scores are uniform over dh[m]−<sup>1</sup> tokens; moreover, the filter ι will be applied to all tokens on level t + 2 and higher. Indeed, the output of nodes on level h[m] = t + 1 reads

$$\hat{\mathbf{x}}_m^{+(t)} = \phi(\hat{\mathbf{z}}_m^{+(t)}), \quad \hat{\mathbf{z}}_m^{+(t)} = \frac{1}{d_t}(\hat{\mathbf{x}}_1^{+(t)} + \cdots + \hat{\mathbf{x}}_{d_t}^{+(t)}) = \frac{1}{d_t}(\mathbf{x}_1^+ + \cdots + \mathbf{x}_{d_t}^+) + O(\epsilon_\ell),$$

so that ∥zˆ +(t) <sup>m</sup> ∥<sup>∞</sup> ≤ O(d −1/4 ) by Lemma [10.](#page-22-1) Then

$$\|\hat{\mathbf{u}}_m^{(t)} + \mathbf{1}_{n'}\|_{\infty} \leq C_2 \||\hat{\mathbf{z}}_m^{+(t)}|^2\| \leq O(d^{-1/2})$$

so that if O(d −1/2 ) < ε0, the filter zeroes out the output of each node on level t + 2. Then the intermediate states of nodes xm′ on level t + 2 read

$$\hat{z}_{m'}^{+(t)} = \frac{1}{d_{t+1}} (\hat{x}_1^{+(t)} + \cdots + \hat{x}_{d_t}^{+(t)}) = \frac{d_t}{d_{t+1}} \hat{z}_m^{+(t)},$$

which again activates the filter. Repeating this process for the remaining levels, we conclude that ∥zˆ +(t) <sup>m</sup> ∥<sup>∞</sup> ≤ O(d −1/4 ) and so ∥xˆ +(t) <sup>m</sup> + 1n+n′∥<sup>∞</sup> ≤ O(d −1/2 ) holds simultaneously for all nodes h[m] > t (and all timesteps t for which [\(11\)](#page-22-0) is valid).

Evaluating the updates. Define z¯ (t) <sup>m</sup> = 1 d<sup>t</sup> P<sup>d</sup><sup>t</sup> <sup>j</sup>=1 x<sup>j</sup> so that

$$\|\hat{\mathbf{z}}_m^{(t)} - \bar{\mathbf{z}}_m^{(t)}\|_{\infty} \leq \frac{1}{d_t} \sum_{j=1}^{d_t} \|\hat{\mathbf{x}}_j^{(t)} - \mathbf{x}_j\|_{\infty} \leq \exp(-Cd^{\epsilon/16}).$$

We proceed to evaluate the gradient of L at [\(11\)](#page-22-0). For the weights wj,m with h[m] = t+ 1, by isolating the errors from the forward pass we have

$$\begin{aligned} \frac{\partial}{\partial w_{j,m}} \left( \frac{1}{2n} \|\hat{\mathbf{x}}_m^{(t)} - \mathbf{x}_m\|^2 \right) &= \frac{1}{n} \left\langle \phi(\hat{\mathbf{z}}_m^{(t)}) - \mathbf{x}_m, \frac{\partial \phi(\hat{\mathbf{z}}_m^{(t)})}{\partial w_{j,m}} \right\rangle \\ &= \frac{\sigma_j(\mathbf{w}_m^{(t)})}{n} \left\langle \phi(\hat{\mathbf{z}}_m^{(t)}) - \mathbf{x}_m, \phi'(\hat{\mathbf{z}}_m^{(t)}), \hat{\mathbf{x}}_j^{(t)} - \hat{\mathbf{z}}_m^{(t)} \right\rangle \\ &= \frac{1}{nd_t} \left\langle \phi(\bar{\mathbf{z}}_m^{(t)}) - \mathbf{x}_m, \phi'(\bar{\mathbf{z}}_m^{(t)}), \mathbf{x}_j - \bar{\mathbf{z}}_m^{(t)} \right\rangle \\ &\quad + O \left( \frac{4}{d_t} (1 + \|\phi'\|_\infty + \|\phi''\|_\infty) \|\hat{\mathbf{z}}_m^{(t)} - \bar{\mathbf{z}}_m^{(t)}\|_\infty \right) \\ &= \frac{1}{nd_t} \left\langle \phi(\bar{\mathbf{z}}_m^{(t)}) - \mathbf{x}_m, \phi'(\bar{\mathbf{z}}_m^{(t)}), \mathbf{x}_j - \bar{\mathbf{z}}_m^{(t)} \right\rangle + O(\exp(-Cd^{\epsilon/16})), \end{aligned}$$

Then the first term is identical to the initial gradient [\(6\)](#page-18-4) analyzed in the proof of Theorem [5](#page-6-0) except for the differences in indices, and from the same computation we obtain the leading term:

$$\frac{\partial}{\partial w_{j,m}} \left( \frac{1}{2n} \|\hat{\mathbf{x}}_m^{(t)} - \mathbf{x}_m\|^2 \right) = -\frac{2c}{d_t^2} 1_{\{\mathbf{p}[j]=m\}} + O(d^{-2-\epsilon/8}),$$

which holds with probability 1 − exp(−d ϵ/2 ) if n = Ω(d 2+ϵ ) under the same setting of Lemma [9.](#page-17-1)

For all other nodes on level t + 1 or below, the output does not depend on the weight wj,m, so the gradient of the squared error with respect to wj,m is zero. Moreover, all nodes on level t + 2 or above are zeroed out due to the filter and hence also has zero gradient. Then the oracle error is absorbed into the second term and the update after time t + 1 with learning rate fixed to η = d 2+ϵ/<sup>16</sup>η<sup>0</sup> reads

$$w_{j,m}^{(t)} - \eta \tilde{\nabla}_{w_{j,m}} L(\mathbf{W}^{(t)}, \mathbf{U}) = 2c\eta_0 \frac{d^{2+\epsilon/16}}{d_{\mathbf{h}[m]-1}^2} 1_{\{p[j]=m\}} + O(d^{-\epsilon/16}).$$

By choosing η<sup>0</sup> such that none of the leading terms lands exactly on a half-integer, we have that for sufficiently large d,

$$w_{c_1[m],m}^{(t+1)} = w_{c_2[m],m}^{(t+1)} = r \left[ 2c\eta_0 \frac{d^{2+\epsilon/16}}{d_h^{[m]-1}} \right], \quad w_{j,m}^{(t+1)} = r[O(d^{-\epsilon/16})] = 0$$

if p[j] ̸= m. We verify that

$$\frac{c\eta_0}{2} \leq C_\ell = 2c\eta_0 \frac{d^{2+\epsilon/16}}{d_{\ell-1}^2} \leq 2c\eta_0.$$

Also, the weights wj,m such that h[m] ≥ t + 2 only affect the nodes that are zeroed out, so that the gradient is also bounded as O(d <sup>−</sup>ϵ/<sup>16</sup>) and w (t+1) j,m = 0.

It remains to evaluate the gradient signal of weights wj,m with h[m] ≤ t, which have already been updated in previous steps. Define the error

$$\xi_{j,m,\ell} := \max_{1 \leq \alpha \leq d_\ell} \left\| \frac{\partial \hat{\mathbf{x}}_\alpha^{(t)}}{\partial w_{j,m}} \right\|_\infty.$$

This is similar to the error control in the proof of Lemma [6,](#page-7-3) but we exploit the fact that parities up to level t are solved to obtain a much tighter bound. Let us expand ϕ ′ (t) = 2c ′ (1 − t) + O((1 − t) 2 ) near 1 and 2c ′ (−1 − t) + O((1 + t) 2 ) near −1 for some positive constant c ′ . Recall

$$\|\hat{z}_\alpha^{+(t)} - z_\alpha^+\|_\infty \leq 2d \exp(-C d^{\epsilon/16}), \quad h[\alpha] \leq t$$

holds in the forward pass, so each component zˆ (t) α,i is <sup>O</sup>(exp(−Cdϵ/<sup>16</sup>))-close to either of ±1. It follows that |ϕ ′ (zˆ (t) α,i)| <sup>=</sup> <sup>O</sup>(exp(−Cdϵ/<sup>16</sup>)), so we can bound

$$\xi_{j,m,h[m]} = \left\| \frac{\partial \hat{\mathbf{x}}_m^{(t)}}{\partial w_{j,m}} \right\|_{\infty} \leq 2\|\phi'(\hat{\mathbf{z}}_m^{(t)})\|_{\infty} \sigma_j(\mathbf{w}_m) \leq O(\exp(-Cd^{\epsilon/16})).$$

Moreover, for any α on level h[α] = ℓ, h[m] < ℓ ≤ t the magnitude of the derivative of the output xˆ (t) <sup>α</sup> can be bounded as

$$\left\| \frac{\partial \hat{\mathbf{x}}_\alpha^{(t)}}{\partial w_{j,m}} \right\|_\infty \leq \left\| \phi'(\hat{\mathbf{z}}_\alpha^{(t)}) \right\|_\infty \sum_{\beta=1}^{d_{\ell-1}} \sigma_\beta(\mathbf{w}_\alpha^{(t)}) \left\| \frac{\partial \hat{\mathbf{x}}_\beta^{(t)}}{\partial w_{j,m}} \right\|_\infty \leq O(\exp(-Cd^{\epsilon/16}))\xi_{j,m,\ell-1}.$$

This implies that ξj,m,t = ξj,m,t−<sup>1</sup> = · · · = ξj,m,h[m] ≤ O(exp(−Cdϵ/<sup>16</sup>)). Furthermore, for any α on level h[α] = t + 1 it holds that

$$\left\| \frac{\partial \hat{\mathbf{x}}_\alpha^{(t)}}{\partial w_{j,m}} \right\|_\infty \leq \|\phi'\|_\infty \sum_{\beta=1}^{d_t} \sigma_\beta(\mathbf{w}_\alpha^{(t)}) \left\| \frac{\partial \hat{\mathbf{x}}_\beta^{(t)}}{\partial w_{j,m}} \right\|_\infty \leq \|\phi'\|_\infty \xi_{j,m,t} \leq O(\exp(-C d^{\epsilon/16})).$$

Thus we have for all α with h[α] ≤ t + 1,

$$\frac{\partial}{\partial w_{j,m}} \left( \frac{1}{2n} \|\hat{\mathbf{x}}_\alpha^{(t)} - \mathbf{x}_\alpha\|^2 \right) = \frac{1}{n} \left\langle \hat{\mathbf{x}}_\alpha^{(t)} - \mathbf{x}_\alpha, \frac{\partial \hat{\mathbf{x}}_\alpha^{(t)}}{\partial w_{j,m}} \right\rangle \leq 2 \left\| \frac{\partial \hat{\mathbf{x}}_\alpha^{(t)}}{\partial w_{j,m}} \right\|_\infty = O(\exp(-C d^{\epsilon/16})),$$

and the nodes on level t + 2 or above are zeroed out due to the filter. Hence the gradient signal is exponentially small and

$$\tilde{\nabla}_{w_{j,m}} L(\mathbf{W}^{(t)}, \mathbf{U}) = O(d \exp(-C d^{\epsilon/16})) + O(d^{-2-\epsilon/8}),$$

so that w (t+1) j,m = r[w (t) j,m + O(d <sup>−</sup>ϵ/<sup>16</sup>)] = w (t) j,m. This concludes the proof of [\(11\)](#page-22-0).

so that 
$$w_{j,m}^{(t+1)} = r[w_{j,m}^{(t)} + O(d^{-\epsilon/16})] = w_{j,m}^{(t)}$$
. This concludes the proof of (11).

Finally, after time t = v the weights at all levels have been updated, so that repeating the analysis of the forward pass yields that

$$\|\hat{\mathbf{y}}_{\text{test}} - \mathbf{y}_{\text{test}}\|_{\infty} = \left\| \hat{\mathbf{x}}_{d+k-1}^{(t)} - \mathbf{x}_{d+k-1} \right\|_{\infty} \leq \epsilon_v \leq \exp(-Cd^{\epsilon/16}),$$

as was to be shown.

### D EXPERIMENTAL DETAILS

For the transformer architecture, the feedforward layer was fixed to the following piecewise quadratic link function:

$$\phi(t) = \begin{cases} -4t^2 - 8t - 3 & t \in [-1, -0.5) \\ 4t^2 - 1 & t \in [-0.5, 0.5) \\ -4t^2 + 8t - 3 & t \in [0.5, 1]. \end{cases}$$

For all CoT models, learning rates were fixed to η = 15, 50, 100 for k = 8, 16, 32. For the direct model, the learning rate was scaled to 0.01η to ensure stability of training. For the self-consistency model, filtering was done through an equivalent weight-based filter, which checks if any softmax score exceeds a threshold value, here set to 0.4. Moreover, we found that adding a 10% fraction of the gradient from the prediction loss to that of the CoT loss resulted in more stable training.

Figure [5](#page-26-0) shows CoT loss and prediction loss curves for k = 8, 16, 32, extending Figure [4.](#page-9-0) The direct model fails to learn parity in all cases, while CoT with teacher forcing always learns efficiently. For CoT with self-consistency, a similar analysis as in Section [4](#page-8-1) can be applied for k = 8, 16 with two or three distinct learning stages. We also observe that the basic CoT model manages to fully solve the problem for k = 8 (two intermediate levels) but not for k = 16, 32 (three and four intermediate levels), indicating that assistance (teacher forcing or self-consistency checking) becomes necessary for more complex tasks.

![](_page_26_Figure_1.jpeg)

Figure 5: CoT loss (left) and prediction loss (right) curves for the four models when d = 64, k = 32 (top), k = 16 (middle) and k = 8 (bottom). For the CoT+consistency model, dashed lines indicate when the filters of each level are deactivated.