# Adas

Nuno Gonc¸alves 1 Marcos Treviso 2 **Andre F. T. Martins** ´
1 2 3

## Abstract

The computational cost of softmax-based attention in transformers limits their applicability to long-context tasks. Adaptive sparsity, of which α-entmax attention is an example, offers a flexible data-dependent alternative, but existing implementations are inefficient and do not leverage the sparsity to obtain runtime and memory gains. In this work, we propose ADASPLASH, which combines the efficiency of GPU-optimized algorithms with the sparsity benefits of α-entmax. We first introduce a hybrid Halley-bisection algorithm, resulting in a 7-fold reduction in the number of iterations needed to compute the α-entmax transformation. Then, we implement custom Triton kernels to efficiently handle adaptive sparsity. Experiments with RoBERTa and ModernBERT for text classification and single-vector retrieval, along with GPT-2 for language modeling, show that our method achieves substantial improvements in runtime and memory efficiency compared to existing α-entmax implementations. It approachesand in some cases surpasses—the efficiency of highly optimized softmax implementations like FlashAttention-2, enabling long-context training while maintaining strong task performance.1

## 1. Introduction

Central to the success of transformers (Vaswani et al., 2017) lies the attention mechanism, where each token in a sequence attends directly to every other token. Attention probabilities are computed through the **softmax** transformation, which always assigns a nonzero probability to every token. However, for long context inputs, the accumulation of small probabilities can lead to dispersion (Velickovi ˇ c et al. ´ , 2025).

1Instituto Superior Tecnico, Universidade de Lisboa, Portu- ´
gal 2Instituto de Telecomunicac¸oes, Lisbon, Portugal ˜
3Unbabel, Lisbon, Portugal. Correspondence to: Nuno Gonc¸alves <nuno.m.goncalves@tecnico.ulisboa.pt>.

Re lati ve Spe ed
 (vs
. Fl as hAtte ntion2)FlashAttention-2 (CUDA)
FlashAttention-2 (Triton) AdaSplash (Triton)
50 60 70 80 90 100 16x16 Block Sparsity (%)
0.4 0.6 0.8 1.0 1.2
In fact, previous research shows that attention probabilities tend to peak around a small number of tokens (Voita et al.,
2019; Treviso et al., 2022), which suggests that model performance and computational efficiency can be increased by leveraging attention sparsity. This has motivated methods that predefine sparse masks (Beltagy et al., 2020; Zaheer et al., 2020b), rely on clustering-based strategies (Kitaev et al., 2020), or low-rank approximate attention (Choromanski et al., 2021; Peng et al., 2021; Xiong et al., 2021; Chen et al., 2021). Some of these techniques show the potential of sparsity to mitigate memory and computation bottlenecks, but they often require architectural modifications or crude approximations, limiting their flexibility and generality. A related line of research explores adaptive and differentiable sparse activations as surrogates of softmax, such as sparsemax (Martins & Astudillo, 2016) and, more broadly, the α**-entmax** family (Peters et al., 2019; Correia et al., 2019). By assigning zero probability to irrelevant tokens, these activations eliminate their residual influence, reducing the dilution of attention scores and potentially improving both performance and interpretability. Unfortunately, existing algorithms and implementations for these adaptive sparse activations do not exploit the sparsity, being slower than softmax-based attention and struggling to scale 1 effectively with context length, primarily due to the lack of hardware-optimized implementations like FlashAttention2 (Dao, 2024) or support from programming models like FlexAttention (Dong et al., 2024). This paper addresses this problem by providing new algorithms and implementations to improve the computational efficiency of the family of α-entmax activations. Our main contributions include a faster and GPU-friendly algorithm for calculating α-entmax, alongside a Triton kernel (Tillet et al., 2019) for computing entmax-based attention, which we call ADASPLASH. In particular, ADASPLASH advances the goal of supporting training of adaptively sparse models with longer context lengths, as shown in Figure 1. We demonstrate the potential and scalability of our approach through experiments with synthetic data and with several natural language processing benchmarks for encoder-only and decoder-only models, achieving substantial improvements over previous α-entmax implementations and approaching (sometimes surpassing) the efficiency of softmaxbased attention with FlashAttention-2, with strong performance on downstream tasks.

## 2. Background 2.1. Hardware Performance

Modern GPUs, such as the Nvidia H100, are designed for efficient parallel computation using a hierarchical memory architecture, with high-bandwidth memory (HBM) providing large capacity but slower access compared to the smaller, faster on-chip SRAM. Efficient use of SRAM is critical to minimize the memory bottlenecks caused by frequent HBM accesses. GPUs execute operations (kernels) via thousands of threads organized into thread blocks, where data is loaded from HBM into SRAM for computation before being written back. Kernel fusion is a key optimization strategy that combines multiple operations into a single kernel, reducing intermediate HBM accesses by directly computing and storing final results. While compilers like torch.compile can automate fusion for simple operations (Ansel et al., 2024), complex tasks such as attention mechanisms require custom strategies to reorder operations and optimize memory usage effectively. Our method leverages this GPU memory organization by implementing block-wise computations, recomputation strategies, and kernel fusion specifically tailored for sparse attention, as detailed in §3.2.1 and §3.2.2.

## 2.2. Standard Attention

Given a set of matrices Q, K,V ∈ R
n×dcontaining ddimensional representations for n queries, keys and values, the *dot-product self-attention* at a single head is computed in the following way (Vaswani et al., 2017):

$$\mathbf{O}$$
$$(1)$$

O = π QK⊤
√d | {z }
S∈Rn×n
!

V ∈ R
n×d. (1)
The π transformation usually maps rows to distributions, with π(S)ij = softmax (si)jbeing a common choice. For decoder-only models, S is masked in order to ignore the contribution from future tokens. Notably, a naive implementation of Equation 1 leads to a On 2time and memory complexity for training.

## 2.3. Flashattention

To address the costs of naive attention implementations, Dao et al. (2022) introduced FlashAttention, an algorithm that avoids the materialization of quadratic matrices via a GPU-aware implementation of online softmax (Milakov & Gimelshein, 2018), bringing the overall memory complexity to O (n). Subsequent versions of FlashAttention further improved GPU usage by reordering the loops, reducing the number of non-GEMM (general matrix multiply) operations (Dao, 2024), and exploiting the asynchronicity and support for FP8 low-precision on the new Hopper GPUs (Shah et al., 2024). The key idea of FlashAttention is to split the inputs Q, K,V into blocks, load them from slow GPU high bandwidth memory (HBM) to the fast GPU on-chip SRAM, then compute the attention output regarding those blocks and, at the end, scale the output by the right normalization factor.

## 2.4. Sparse Attention

The original softmax-based attention is *dense*, i.e., it puts some probability mass on all tokens—not only a computational disadvantage, but also making interpretation and generalization harder (Voita et al., 2019; Treviso et al., 2022; Velickovi ˇ c et al. ´ , 2025). An alternative to softmax is the α**-entmax transformation** (Peters et al., 2019), which is differentiable and leads to sparse outputs:

$$\alpha\text{-entmax}\left(\mathbf{s}\right)=\left[(\alpha-1)\mathbf{s}-\tau\mathbf{1}\right]_{+}^{1/\alpha-1},\tag{2}$$

where [·]+ is the ReLU function, and τ ∈ R is a normalizing constant to ensure the output is a valid probability distribution. Importantly, entries with score si ≤τ α−1 get exactly zero probability. In the limit α → 1, α-entmax recovers the softmax function, while for any value of α > 1 this transformation returns increasingly sparser probability vectors.

When α = 2, we recover the sparsemax transformation
(Martins & Astudillo, 2016). However, in contrast to fixed sparse patterns, such as windowed sparse attention (Child et al., 2019; Beltagy et al., 2020) and block-sparse variants (Zaheer et al., 2020b; Dao et al., 2022), α-entmax's sparsity patterns are dynamic and hence difficult to exploit in order to reduce the quadratic burden of self-attention because we still need to materialize S = QK⊤ before applying the transformation. In the next section (§3), we outline ADASPLASH, our new method for computing α-entmax attention, along with a novel custom Triton kernel (Tillet et al., 2019) that enables efficient training of transformers for extremely long context lengths. As shown in §4, our implementation maintains competitiveness with state-of-the-art algorithms such as FlashAttention by leveraging the sparsity given by α-entmax, effectively exploiting the advantages of sparse attention at scale.

## 3. Adas**Plash**

We start by revisiting the computation of α-entmax for general values of α in §3.1, and proposing a new algorithm that has a fast empirical convergence. We design an efficient Triton kernel in §3.2, dubbed ADASPLASH, that effectively leverages adaptive sparsity patterns in both the forward and backward passes of α-entmax in order to minimize runtime.

## 3.1. Α**-Entmax Computation**

In order to compute Equation 2 for a given s ∈ R
n, we need to find the threshold τ ∈ R such that the resulting output sums to 1. Mathematically, this is equivalent to finding the root of the following equation:

$$f(\tau)=\sum_{i}\left[(\alpha-1)s_{i}-\tau\right]_{+}^{1/(\alpha-1)}-1.\qquad\mathrm{(3)}$$

Exact algorithms for α ∈ {1.5, 2}. In particular, for α = 2, the computation is reduced to an Euclidean projection onto the probability simplex, for which efficient algorithms have been extensively studied (Held et al., 1974; Duchi et al., 2008; Condat, 2016). Similarly, for α = 1.5, Peters et al. (2019) introduced an exact sort-based algorithm. However, these methods either require complex data structures that are not efficiently handled in GPUs, or sorting-based algorithms, which require the materialization of the entire input. Bisection algorithm for α > 1. For a general α, Blondel et al. (2019) introduced a bisection update rule to approximate τ by iteratively refining its lower (τlo) and higher (τhi)
bounds:

$$B_{f}(\tau)=\begin{cases}(\tau_{\text{lo}},\tau)&\text{if}f(\tau)<0,\\ (\tau,\tau_{\text{hi}})&\text{otherwise},\end{cases}\tag{4}$$

obtaining τ =
1 2
(τlo + τhi) after the last iteration. While the bisection algorithm is simple and effective, it converges at a linear rate (Kaufman & Lenker, 1986), meaning the absolute error decreases by approximately half at each iter-

Algorithm 1 Halley-bisection algorithm for α-entmax.
1: **Input:** logits s ∈ R
n, param. α ∈ R, iterations T
2: Define f(τ ) := Pi
[si − τ ]
1/(α−1)
+ − 1
3: Set s ← (α − 1)s
4: Initialize τlo = max(s) − 1 5: Initialize τhi = max(s) − n
1−α
6: Initialize τ = (τlo + τhi)/2
7: **repeat**
  **Peak**  Compute $\tau_{\rm lo},\tau_{\rm hi}=B_{f}(\tau)$ (Equation 4)  Compute $\tau_{H}=H_{f}(\tau)$ (Equation 5)  **if $\tau_{H}\in[\tau_{\rm lo},\tau_{\rm hi}]$ then**  $\tau\leftarrow\tau_{H}$  $\triangleright$ (Halle
12: **else**
13: τ ← 12
14: **end if** 16: **Output:** [s − τ1]
+
  **Example $\tau_{H}=\tau_{H}(\tau)$ (Equation 3)**  **if $\tau_{H}\in[\tau_{\rm lo},\tau_{\rm hi}]$ then**  $\tau\gets\tau_{H}$  **else**  $\tau\gets\frac{1}{2}(\tau_{\rm lo}+\tau_{\rm hi})$  $\triangleright$ (Bisection Update)  **end if**  **until $T$ iterations are completed**  **mutants $[\tau_{\rm lo},\tau_{\rm hi}]/(\alpha-1)$**
$$12{\dot{\cdot}}$$
ation. Achieving high precision often requires many iterations, resulting in frequent memory accesses. As a result, in memory-bound scenarios where the time taken is mostly determined by the number of memory accesses—such as in attention—the number of iterations can significantly impact the runtime cost. Halley-bisection algorithm. In order to obtain a faster runtime, we propose a hybrid algorithm for solving Equation 3 for any α > 1 that combines the convergence guarantee of bisection with the faster convergence of Halley's method (Scavo & Thoo, 1995). As we show in §4.1, this approach achieves significant wall-clock speed-ups while requiring fewer iterations to attain the same precision. The function defined in Equation 2 enjoys a cheap computation of its derivatives. Thus, methods that incorporate second-order information, such as Halley's method, can be leveraged to improve the approximation of τ at each iteration. Halley's method, which uses both the first and second derivatives, updates the solution using the following rule:

$$H_{f}(\tau)=\tau-\frac{2f(\tau)f^{\prime}(\tau)}{2f^{\prime}(\tau)^{2}-f(\tau)f^{\prime\prime}(\tau)},$$
$$({\boldsymbol{S}})$$
$$\mathbf{\Sigma}(\mathbf{6})$$
, (5)
where the derivatives are given as follows:

$$\begin{array}{l}{{f^{\prime}(\tau)=-\frac{1}{\alpha-1}\sum_{i}\left[(\alpha-1)s_{i}-\tau\right]_{+}^{1/(\alpha-1)-1},}}\\ {{f^{\prime\prime}(\tau)=\frac{2-\alpha}{(\alpha-1)^{2}}\sum_{i}\left[(\alpha-1)s_{i}-\tau\right]_{+}^{1/(\alpha-1)-2}.}}\end{array}$$

While Halley's method offers faster convergence under ideal conditions, it does not always converge, particularly when

Bisection (Output) Bisection (Gradient)
Halley-bisection (Output) Halley-bisection (Gradient)
5 10 15 20 25 Number of Iterations 10 10 10 9 10 8 10 7 10 6 10 5 10 4 Mean Ab solut e E
rror Magn itude
the initial guess is far from the solution. To ensure convergence, we introduce a fail-safe mechanism that integrates the convergence guarantee of bisection: whenever Halley's method produces an update that moves the solution out of the bisection bounds, the algorithm reverts to a bisection update Bf (τ ). This ensures that the algorithm converges, even in the worst cases, while leveraging the cubic convergence of Halley's method wherever possible. We outline our hybrid algorithm in Algorithm 1. Efficiency Benchmark. We compare the runtime of Halley-bisection against existing algorithms for computing α-entmax implemented in Torch. Specifically, we generate random tensors from a standard Gaussian distribution
(µ = 0, σ2 = 1) with a fixed sequence length of n = 8192.

For each configuration, we measure the average runtime over 1000 runs. Overall, we observe that Halley-bisection is significantly more efficient than the standard bisection algorithm implemented in Torch. Halley-bisection achieves a runtime of 2.38 ms, compared to 36.67 ms for the standard bisection algorithm, making it approximately 15× **faster**. In addition, Halley-bisection reduces memory usage by **1.75**×, requiring only 512 MB compared to 896.15 MB for bisection. Furthermore, in Figure 2 we show that Halley-bisection (α = 1.5) requires only 3 iterations to converge to machine precision for both the output and the gradient. On the other hand, the standard bisection algorithm takes 23 iterations to achieve the same precision for both cases.

## 3.2. Flash Α**-Entmax Attention**

Given an algorithm to compute the entmax mapping that requires T iteration steps, a naive implementation of entmax attention proceeds as follows: (1) multiply S = QK⊤ ∈
R 
n×n and write the result to slow HBM on the GPU; (2)
load S from HBM T times to compute τ ; (3) load S from HBM again, and write the result P = α-entmax (S) to HBM; (4) perform a matrix multiplication to get the output Algorithm 2 ADASPLASH forward pass (w/o masking)
1: **Require:** Matrices Q, K,V ∈ R
n×din HBM, block sizes Bc, Br, param. α ∈ R
2: Divide Q into Tr = ⌈n/Br⌉ blocks Q1*, . . . ,* QTrof size Br × d 3: Divide K,V into Tc = ⌈n/Bc⌉ blocks K1*, . . . ,* KTc, V1*, . . . ,* VTcof size Bc × d 4: Divide O ∈ R
n×dinto Tr blocks O1*, . . . ,* OTrof size Br × d 5: Divide τ into Tr blocks τ1*, . . . ,* τTrof size Br 6: for i = 1 to Tr do 7: Load Qi from HBM to on-chip SRAM 8: On chip, initialize Oi 9: On chip, compute τi using Hybrid Halley's with predefined α, using a block version of Algorithm 1.

10: for j = 1 to Tc do 11: Load Kj , Vj from HBM to on-chip SRAM
12: Compute S
(j)
i = QiK⊤
j ∈ R
Br×Bc 13: Compute P
(j)
i =
h(α − 1)S
(j)
i − τi i1/α−1
+
14: Accumulate Oi ← Oi + P
(j)
i Vj 15: **end for**
16: Write Oi and τito HBM
17: **end for** 18: **Return:** Output O and τ O = P V . However, since most of these operations are memory-bound, the excessive number of HBM accesses leads to slow wall-clock times. Moreover, having to materialize S and P in memory poses a major bottleneck, as their sizes quickly exceed GPU memory capacity when the sequence length n increases. To address these issues and speed up α-entmax attention on hardware accelerators like GPUs, we propose an algorithm that reduces HBM reads and writes while producing the same outputs as the naive implementation.

## 3.2.1. Forward Pass

We outline the forward pass in Algorithm 2 (without masking full-zero blocks, which we introduce later on this section). Concretely, given the inputs Q, K,V ∈ R
n×d stored in HBM, the goal is to compute the attention output O ∈ R
n×defficiently and write it back to HBM. Akin to the approach taken in FlashAttention (Dao et al., 2022),
we employ two well-known techniques—**tiling** and recomputation—to address the challenge of materializing the matrices S ∈ R
n×n and P ∈ R
n×n.

Tiling. The key idea involves splitting the inputs Q, K,V
into smaller blocks, and then computing attention block by block. We start by loading only Q and K from the slower HBM to the faster SRAM to compute τ ∈ R
n using the Halley-bisection algorithm (Alg. 1). In order to use the aforementioned algorithm, we need to accumulate three values: f(τ ), f
′(τ ), f
′′(τ ). Since f, as well as its derivatives, is additive over its inputs, their computation can also be computed in blocks. Let Br and Bc be the row and column block sizes, respectively, and define Tr = ⌈n/Br⌉ and Tc = ⌈n/Bc⌉. Divide Q into Q1*, ...,* QTrblocks, and K into K1*, ...,* KTc blocks. Then, f(τ ) can be computed as:

$$f(\mathbf{\tau}_{i})=\sum_{j=1}^{T_{c}}f(\mathbf{\tau}_{i};\mathbf{S}_{i}^{(j)})\qquad\qquad(\mathbf{8})$$

where S
(j)
i = QiK⊤
j ∈ R
Br×Bc and τi represents the i th sliced block of τ with size Tr. Thus, these quantities do not need to ever be materialized and can be accumulated directly in fast memory. Afterwards, we load V to compute the attention output O for those blocks. In contrast to FlashAttention, our approach requires loading K to compute S at least two additional times. Therefore, the forward pass is bound to always be slower than FlashAttention's due to the extra HBM reads and computation.

Recomputation. In order to avoid the materialization of the matrices S and P , we recompute them again in Algorithm 1, which is used to compute τ , and also recompute them for obtaining the gradients for the backward pass. By doing this we are increasing the required FLOPs to reduce the maximum amount of memory required. While this might suggest an increase in runtime, the opposite is observed (Dao et al., 2022). Despite the need for additional matrix multiplications, the reduction in total HBM reads and writes more than offsets the extra FLOPs, leading to improved performance overall.

Sparsity-aware implementation. The key challenge of α-entmax attention lies in finding the threshold τ , which requires multiple evaluations of the function f(τ ), which, in turn, depends on the score matrix S. While our proposed Halley-bisection algorithm alleviates the number of iterations needed to recompute S
(j)
iby providing a faster empirical convergence, our current implementation still iterates over all blocks of S, including **null blocks**—blocks where the corresponding entries of the sparse attention matrix P are zero. Furthermore, empirical evidence from Jiang et al. (2024) and (Xiao et al., 2024) suggests that for long inputs (e.g., 128k tokens in LLaMa-3-8b), approximately 3% of the entries in P suffice to capture over 96% of the total attention, which motivates an approach to leverage the adaptive and unstructured sparsity of α-entmax attention weights. To this end, we propose to only compute necessary blocks of P by skipping the null blocks. Concretely, let I(i) denote the set of all indices i
′such that ⌊i
′/Tr⌋ = i, and J (j) denote the set of all indices j
′such that ⌊j
′/Tc⌋ = j. We construct a block mask matrix M ∈ {0, 1}
Tr×Tc as follows:

$$M_{ij}=\begin{cases}1&\text{if}\exists_{i^{\prime}\in\mathcal{I}(i),j^{\prime}\in\mathcal{I}(j)}:S_{i^{\prime},j^{\prime}}>\tau_{i^{\prime}},\\ 0&\text{otherwise},\end{cases}\tag{9}$$

Importantly, M is created dynamically after a small predefined number of Halley-bisection iterations. While the introduction of M breaks the linear memory complexity of dense fused-attention by requiring Tr × Tc extra memory, the overhead is still manageable as it only contains binary values and is substantially smaller than the full P ∈ R
n×n matrix. Furthermore, M needs to be materialized only once and its memory can be shared across all attention layers. To leverage M in practice, we propose to create two **pointer-increment lookup tables**:
1. Kj = {i | Mij = 1}: A table containing the row indices i of M that lead to non-null blocks in P
(j)
i.

2. $\mathcal{Q}_{i}=\{j\mid M_{ij}=1\}$: A table containing the column indices $j$ of $M$ that lead to non-null blocks in $\boldsymbol{P}_{i}^{(j)}$.  
These tables enable efficient skipping of K and V blocks that do not contribute to the final attention output O, significantly reducing unnecessary computations. Moreover, the same mechanism can be extended to accelerate the backward pass, where gradients with respect to Q, K, and V are computed, which we describe next.

## 3.2.2. Backward Pass

In FlashAttention (Dao et al., 2022), the backward pass is executed using a single kernel that parallelizes computation across batch, head, and sequence dimensions. However, following Triton's official implementation of FlashAttention,2 we separate the backward pass into two kernels: one for dQ (the gradient w.r.t. Q) and another for dK and dV (the gradients w.r.t. K and V ). Sparse Jacobian of α**-entmax.** The sparsity in the Jacobian of α-entmax plays a crucial role in the backward pass. For p = α-entmax(s), the Jacobian is (Peters et al., 2019)

$$\partial\alpha\text{-entmax}(\mathbf{s})=\text{Diag}(\mathbf{u})-\frac{\mathbf{uu}^{\top}}{\|\mathbf{u}\|_{1}},\tag{10}$$

where uj = (pj )
2−α. Importantly, this Jacobian is sparse and only depends on p, which, in turn, is a function of τ computed during the forward pass. We denote by U ∈
R 
n×n the matrix defined element-wise as Ulk = P
2−α lk , and by U
(j)
i ∈ R
Br×Bcits (*i, j*)
th block. Using this information, 2https://github.com/triton-lang/triton/blob/
main/python/tutorials/06-fused-attention.py

Bisection (Torch) FlashAttention2 (Triton) FlashAttention2 (CUDA)
AdaSplash w/o masking (Triton) AdaSplash (Triton)
0.1 1.0 10.0 Ti m e (
s)
O
O M

O
O M

O
O M

O
O
M

1k 2k 4k 8k 16k 32k 64k
the gradient w.r.t. the score matrix S
(j)
i ∈ R
Br×Bc can be efficiently computed as:

$$\mathbf{dS}_{i}^{(j)}=\mathbf{U}_{i}^{(j)}\odot\mathbf{dP}_{i}^{(j)}-\text{Diag}(\mathbf{\delta}_{i})\mathbf{U}_{i}^{(j)},\tag{11}$$  where $\mathbf{dP}_{i}^{(j)}=\mathbf{dO}_{i}\mathbf{V}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$, with $\mathbf{dO}_{i}\in\mathbb{R}^{B_{r}\times n}$
and Vj ∈ R
Bc×n, and δi ∈ R
Br denotes the i th block of the vector δ ∈ R
n defined element-wise as δl =
(Pk UlkdPlk)/(Pk Ulk).

Efficient gradient computation. In ADASPLASH, instead of storing P , we store the lookup tables K and Q computed during the forward pass, allowing us to to efficiently skip the computations of null blocks during backpropagation. Given dSi, the gradients for Qi, Ki, Vi ∈ R
Br×dare computed as follows using the pointer-increment lookup tables:

$$\begin{array}{l}\mathbf{d}\mathbf{Q}_{i}=\sum_{j\in\mathbf{Q}_{i}}\mathbf{dS}_{i}^{(j)}\cdot\mathbf{K}_{j},\\ \mathbf{d}\mathbf{K}_{j}=\sum_{i\in\mathbf{K}_{j}}\mathbf{dS}_{i}^{(j)}\cdot\mathbf{Q}_{i},\\ \mathbf{d}\mathbf{V}_{j}=\sum_{i\in\mathbf{K}_{j}}\mathbf{P}_{i}^{(j)}\cdot\mathbf{d}\mathbf{O}_{i}.\end{array}\tag{14}$$

Hence, by splitting the backward pass into separate kernels and exploiting the sparsity of α-entmax through the Jacobian structure, we can achieve efficient gradient computation. Overall, ADASPLASH allows users to choose between memory efficiency (without block masking) and computational speed (with block masking) depending on the task requirements and hardware constraints. We provide a detailed derivation of α-entmax attention's backward pass and its implementation in Appendix A.2.

## 4. Experiments

We evaluate ADASPLASH across various scenarios to show its computational efficiency and impact on downstream tasks. Our experiments address the following questions:
- Performance efficiency: How does ADASPLASH compare with baseline methods in terms of runtime as sequence length and sparsity vary?

- Generalization to architectures: How does ADAS-
PLASH perform when integrated with encoder-only and decoder-only models?

- Effectiveness in finetuning: Can ADASPLASH-
pretrained models outperform or match their dense counterparts in short and long-context tasks?

## 4.1. Efficiency Benchmark

We compare the efficiency of ADASPLASH against FlashAttention-2 and naive implementations of α-entmax. For a fair comparison, we also include a variant of FlashAttention-2 implemented in Triton that follows closely our kernel implementation of ADASPLASH. We set the number of iterations of ADASPLASH to 3 and Bisection to 10. As input, we generate random tensors from a Gaussian distribution (µ = 0), simulating attention scores with a high level of sparsity by setting the Gaussian variance to σ 2 = 6 of query vectors. Sequence lengths range from 1k to 64k, with a fixed head size of d = 64.

Runtime. We show the average training step time for each method in Figure 3. ADASPLASH demonstrates superior scalability, efficiently handling sequences up to 64k, unlike the Bisection method implemented in Torch, which runs out of memory beyond 4k context length. We also note that, as context length increases, the amount of block sparsity naturally increases as well, leading to an advantage for our method over both implementations of FlashAttention-2.

## 4.2. Performance On Real Tasks

Encoder-only models, such as RoBERTa (Liu et al., 2019) and ModernBERT (Warner et al., 2024), exhibit higher attention sparsity than decoder-only models, making them well-suited for adaptive sparse attention mechanisms like ADASPLASH. Following ModernBERT's evaluation setup, we opt to evaluate these models on standard NLP tasks, such as text classification, natural language inference, textual similarity, and information retrieval. Moreover, following FlashAttention's evaluation setup (Dao et al., 2022), we also benchmark ADASPLASH with GPT-2, a decoder-only

| Model                     | Seq. SciFact NFC FiQA TREC-C   |      |      |      |      |
|---------------------------|--------------------------------|------|------|------|------|
| RoBERTa                   | 512                            | 51.7 | 23.1 | 27.8 | 60.1 |
| RoBERTa (α = 1.5)         | 512                            | 50.8 | 24.2 | 27.6 | 71.0 |
| RoBERTa (α = 2.0)         | 512                            | 52.2 | 23.8 | 25.7 | 65.5 |
| ModernBERT                | 8192                           | 57.7 | 22.4 | 25.7 | 67.6 |
| ModernBERT (α = 1.5) 8192 | 58.4                           | 25.7 | 29.6 | 75.2 |      |
| ModernBERT (α = 2.0) 8192 | 58.0                           | 25.4 | 29.3 | 71.1 |      |

Table 2. Long document classification performance (F1 micro) with softmax and α-entmax attention.

| Sequence Length   |
|-------------------|

Model 512 1024 2048 4096 8192

RoBERTa 71.5 74.4 75.1 77.9 **79.2**

RoBERTa (α = 1.5) **71.8 75.5 76.4 78.0** 78.6

model, to assess its efficiency in autoregressive settings where attention patterns are denser. This ensures a comprehensive comparison with optimized softmax-based methods while validating the benefits of sparsity across different architectures. We provide more training and evaluation details for each task in Appendix B. Continuous pretraining. We conducted continuous pretraining of RoBERTa-base and ModernBERT-base on 2B tokens of the English subset of Fineweb-edu (Lozhkov et al.,
2024) using ADASPLASH for α ∈ {1.5, 2}, and PyTorch's scaled dot product attention for α = 1.0. To ensure a smooth transition from dense to sparse attention, we linearly increased α from α = 1.0 to the target values α ∈ {1.5, 2.0} over the first 1B tokens and kept it fixed afterwards. We provide more details on the continuous pretraining phase in Appendix B.1, including efficiency results. Single-vector retrieval. We evaluate our pretrained models on single-vector retrieval performance using the BEIR benchmark (SciFact, NFCorpus, FiQA2018, TREC- COVID), following the setup in (Warner et al., 2024). Table 1 highlights the performance of RoBERTa and ModernBERT models using α-entmax attention in terms of the standard nDCG@10 metric. ModernBERT with α = 1.5 consistently outperformed its dense counterpart, achieving the highest scores on all tasks, demonstrating its ability to focus on relevant signals effectively. While ModernBERT
with α = 2.0 remained competitive, its higher sparsity might have excluded relevant information, affecting task performance. Finally, sparse versions of ModernBERT achieve better results than the sparse versions of RoBERTa on all tasks, highlighting the benefit of modeling long contexts.

Long document classification. We fine-tuned a pretrained RoBERTa model (Liu et al., 2019) on the ECtHR (Chalkidis et al., 2019; 2021) dataset while progressively increasing the sequence length up to 8192 tokens. Positional embeddings were extended by repetition, following the approach of Beltagy et al. (2020). As a baseline, we fine-tuned the model using standard softmax-based attention. For αentmax attention, we linearly increased the α from 1.0 to 1.5 during training to ensure smooth convergence. The results, summarized in Table 2, show a consistent improvement in model performance with longer context lengths. Notably, despite the base model being pretrained with standard attention, α-entmax attention was capable of effectively learning the task, achieving a slightly higher micro F1 score than the model fine-tuned with standard attention up to a sequence length of 4096 tokens.

| Runtime (hh:mm:ss)   | Sequence Length   |       |       |         |         |
|----------------------|-------------------|-------|-------|---------|---------|
| Model                | 512               | 1024  | 2048  | 4096    | 8192    |
| RoBERTa              | 2:39              | 5:00  | 9:35  | 18:36   | 35:51   |
| RoBERTa (α = 1.5)    | 2:45              | 5:20  | 10:24 | 19:54   | 38:08   |
| w/ Torch Bisect      | 4:51              | 8:44  | 22:48 | 1:11:53 | 4:12:34 |
| Memory (GB)          | Sequence Length   |       |       |         |         |
| 512                  | 1024              | 2048  | 4096  | 8192    |         |
| RoBERTa              | 6.75              | 11.43 | 20.35 | 37.49   | 75.00   |
| RoBERTa (α = 1.5)    | 6.75              | 11.45 | 20.38 | 39.17   | 79.88   |
| w/ Torch Bisect      | 7.75              | 16.92 | 44.06 | 142.76  | 508.16  |

Table 3 compares the runtime per epoch and peak memory usage for different sequence lengths on the long document classification task. We report results for RoBERTa with FlashAttention-2 (α = 1), RoBERTa with ADASPLASH (α = 1.5), and RoBERTa using Torch's bisection-based implementation. ADASPLASH enables scalable training with α-entmax attention. Prior to this, implementations had to resort to Torch's bisection, which leads to both extremely slow runtimes or even out-of-memory problems, rendering it infeasible for most realistic training setups. In contrast, our method brings the cost of α-entmax attention down to the level of existing dense attention implementations, as both runtime and memory usage with ADASPLASH remain well aligned with those of FlashAttention-2.

Language understanding. We also evaluate RoBERTa and ModernBERT models with α-entmax attention on the GLUE benchmark (Wang et al., 2018) in Appendix B.2.

Overall, the results indicate that models with sparse attention

| Model              | Val. Loss HS Acc. Runtime Memory   |      |      |      |
|--------------------|------------------------------------|------|------|------|
| GPT-2              | 3.283                              | 30.4 | 0.98 | 52.5 |
| GPT-2 (α = 1.5)    | 3.263                              | 30.6 | 1.03 | 52.5 |
| w/ Torch sorting   | -                                  | -    | 3.61 | 73.8 |
| w/ Torch bisection | -                                  | -    | 7.78 | 77.6 |

achieve comparable performance to their dense counterparts, which underscores the ability to efficiently train α-entmax models without sacrificing accuracy.

0.0 0.2 0.4 0.6 0.8 1.0 1 2 3 4 5 6 He ad 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12 Layer
Language modeling. Following (Dao et al., 2022), we trained a small 124M GPT-2 model (Radford et al., 2019) from scratch on 10B tokens of the FineWeb dataset (Penedo et al., 2024) with a context length of 1024 tokens. For a consistent evaluation between softmax and α-entmax attention, we also trained a softmax-based GPT-2 to serve as baseline. After training, we evaluated both models on the HellaSwag task (Zellers et al., 2019). Table 4 presents a side-by-side comparison of the final validation loss and accuracy on HellaSwag, along with runtime and memory usage numbers. Sparse GPT-2 achieves a slight improvement in validation loss (3.263 vs. 3.283) and final accuracy
(30.6% vs. 30.4%) compared to its softmax counterpart, while obtaining comparable runtime and memory efforts.

Furthermore, our approach achieves a runtime comparable to the GPT-2 using the highly optimized FA2 (1.03 s/step vs. 0.98 s/step) and matches its memory footprint (52.5 GB), while outperforming the sorting and bisection variants by large margins in both speed (1.03 s/step vs. 3.61 and 7.78 s/step) and memory usage (52.5 GB vs. 73.8 and 77.6 GB). In Appendix B.4, we report all training and evaluation details, including the validation loss curves of each method. Sparsity in attention heads. Figure 4 presents the sparsity observed in attention heads for all layers for an input of 1024 tokens for our sparse GPT-2 model (α = 1.5). Except for the first layer, all subsequent layers exhibit a high degree of sparsity, highlighting the potential efficiency gains from leveraging this property. Moreover, in Figure 5 (Appendix B.1), we illustrate the sparsity patterns in ModernBERT-base attention heads for α ∈ {1.5, 2.0},
reinforcing similar conclusions.

## 5. Related Works

Sparse Probability Transformations. The sparsity inherent to the α-entmax transformation, as demonstrated by Blondel et al. (2019), is directly controlled by the α parameter. For α = 2, the problem simplifies to a projection onto the probability simplex, a well-established optimization problem. Its solution forms the base of sparsemax (Martins & Astudillo, 2016), which can be efficiently computed using sorting and root-finding methods (Held et al., 1974; Condat, 2016; Liu & Ye, 2009). Moreover, for intermediate values such as α = 1.5, Peters et al. (2019) proposed an exact sorting-based algorithm along with an implementation of a bisection algorithm applicable to any α. However, these approaches remain suboptimal for long contexts due to slow convergence or reliance on complex data structures and sorting operations, which are difficult to optimize for hardware.

Sparse Attention Mechanisms. Efficient sparse attention mechanisms have been widely studied to reduce the quadratic cost of transformers. The Sparse Transformer (Child et al., 2019) introduces a fixed windowed attention that can be efficiently computed using CUDA kernels, a strategy also adopted by Longformer (Beltagy et al., 2020), and BigBird (Zaheer et al., 2020a). However, datadependent sparse attention methods, such as Reformer (Kitaev et al., 2020) and Routing Transformer (Roy et al., 2021), aimi to approximate softmax in return for efficiency, not leveraging the sparsity of attention weights. Other methods, such as Top-k attention (Gupta et al., 2021) and NSA (Yuan et al., 2025), provide sparsity but require a fixed, nonadaptable budget. In contrast, α-entmax attention provides natural, input-dependent sparsity patterns with an exact and differentiable transformation that generalizes softmax, making it more flexible for modeling attention distributions. Adaptively sparse transformers (Correia et al., 2019) uses α-entmax attention where attention heads can learn α dynamically, improving interpretability but without leveraging sparsity for efficiency. SparseFinder (Treviso et al., 2022)
aims to address efficiency issues by predicting the sparsity pattern of entmax attention a priori; however, it does not scale efficiently for long sequences.

Hardware-Aware Attention. Recent works have explored optimizing attention mechanisms with hardwareaware implementations. Flex Attention (Dong et al., 2024) provides an API for efficient attention computation, though they remain tied to softmax-based transformations and do not support more complex operations such as those considered in our work. Closely related to our approach, FlashAttention-1 and 2 (Dao et al., 2022; Dao, 2024) optimize softmax-based attention using tiling and recomputation techniques implemented in CUDA. While FlashAttention includes a sparse block variant, its sparsity pattern must be predefined, limiting adaptability. In this work, we compare our method, ADASPLASH, with FlashAttention-2 and demonstrate that our approach can outperform both its CUDA and Triton implementations at high input sparsity levels. Similarly, Sparse Flash Attention (Pagliardini et al., 2023) extends FlashAttention-1 with a sparse variant that reduces computational cost by either dropping queries and keys per head or grouping them using a hash-based bucketing approach. However, despite its efficiency improvements, it relies on slow sorting operations and is constrained to causal attention, making its sparsity a by-product of bucketing rather than an inherently adaptive feature, as in our case.

Efficiency at Inference Time. Another line of work focuses on optimizing transformers at inference time. Methods such as Paged Attention (Kwon et al., 2023) and KV
cache sparsification (Devoto et al., 2024; Luohe et al., 2024) aim to alleviate the linear complexity of inference by modifying key-value caching strategies. While our approach does not directly provide KV cache compression benefits, these methods are orthogonal and can be combined with our work to further improve inference efficiency.

## 6. Conclusion

In this work, we introduced ADASPLASH, a hardware-aware and efficient implementation of α-entmax attention, bridging the gap between adaptive sparse activations and efficient long-context modeling. Our approach leverages a hybrid Halley-bisection algorithm for faster empirical convergence and custom Triton kernels to exploit the inherent sparsity of α-entmax. Our experiments show that ADASPLASH not only achieves substantial computational improvements over existing α-entmax implementations, but can often match or even surpass the efficiency of highly optimized softmaxbased attention algorithms like FlashAttention-2. Moreover, ADASPLASH enables long-context training while maintaining strong task performance across diverse benchmarks, such as language understanding, information retrieval, document classification, and language modeling. Overall, our work unlocks the viability of dynamically sparse attention mechanisms in large-scale training, which was previously hindered by computational inefficiencies.

## Impact Statement

Efficient attention mechanisms are crucial for scaling transformers to long-context tasks. Our work provides a practical implementation by making adaptive sparse attention efficient, overcoming previous computational limitations of α-entmax. Therefore, the improved efficiency of ADAS-
PLASH has potential applications in large-scale NLP, where sparsity can be leveraged to reduce computational costs. We do not foresee direct societal consequences from sparsity itself, but its integration into decision-making models may still reflect biases in training data. As such, we encourage careful evaluation when deploying sparse attention mechanisms in high-stakes applications, ensuring that efficiency gains do not come at the cost of fairness or transparency.

## Acknowledgments

We thank Vlad Niculae for his insightful and constructive comments throughout this work. We also thank the SARDINE Lab members for reviewing this paper and providing helpful feedback. This work was supported by the Portuguese Recovery and Resilience Plan through project C645008882-00000055 (Center for ResponsibleAI), by the EU's Horizon Europe Research and Innovation Actions (UT- TER, contract 101070631), by the project DECOLLAGE (ERC-2022-CoG 101088763), and by FCT/MECI through national funds and when applicable co-funded EU funds under UID/50008: Instituto de Telecomunicac¸oes. ˜

## References

Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., Hirsh, B., Huang, S., Kalambarkar, K., Kirsch, L., Lazos, M., Lezcano, M., Liang, Y., Liang, J., Lu, Y., Luk, C. K., Maher, B., Pan, Y., Puhrsch, C., Reso, M., Saroufim, M., Siraichi, M. Y., Suk, H., Zhang, S., Suo, M., Tillet, P., Zhao, X., Wang, E., Zhou, K., Zou, R., Wang, X., Mathews, A., Wen, W., Chanan, G., Wu, P., and Chintala, S. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In *Proceedings of the 29th ACM* International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS '24, pp. 929–947, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400703850. doi: 10.1145/3620665.3640366. URL https://doi.org/10.1145/3620665.3640366.

Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T.,
et al. Ms marco: A human generated machine reading comprehension dataset. *arXiv preprint arXiv:1611.09268*, 2016.

Beltagy, I., Peters, M. E., and Cohan, A. Longformer:
The Long-Document Transformer. *arXiv:2004.05150*
[cs], April 2020. URL http://arxiv.org/abs/2004. 05150. arXiv: 2004.05150.

Blondel, M., Martins, A., and Niculae, V. Learning classifiers with fenchel-young losses: Generalized entropies, margins, and algorithms. In Chaudhuri, K. and Sugiyama, M. (eds.), Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of *Proceedings of Machine Learning Research*, pp. 606–615. PMLR, 16–18 Apr 2019. URL https:// proceedings.mlr.press/v89/blondel19a.html.

Chalkidis, I., Androutsopoulos, I., and Aletras, N. Neural legal judgment prediction in English. In Korhonen, A., Traum, D., and Marquez, L. (eds.), ` Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4317–4323, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1424. URL https://aclanthology. org/P19-1424/.

Chalkidis, I., Fergadiotis, M., Tsarapatsanis, D., Aletras, N.,
Androutsopoulos, I., and Malakasiotis, P. Paragraph-level rationale extraction through regularization: A case study on European court of human rights cases. In Toutanova, K., Rumshisky, A., Zettlemoyer, L., Hakkani-Tur, D., Beltagy, I., Bethard, S., Cotterell, R., Chakraborty, T., and Zhou, Y. (eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 226–241, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. naacl-main.22. URL https://aclanthology.org/ 2021.naacl-main.22/.

Chen, B., Dao, T., Winsor, E., Song, Z., Rudra, A., and Re,´
C. Scatterbrain: Unifying sparse and low-rank attention. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/ forum?id=SehIKudiIo1.

Child, R., Gray, S., Radford, A., and Sutskever, I.

Generating Long Sequences with Sparse Transformers.

arXiv:1904.10509 [cs, stat], April 2019. URL http:
//arxiv.org/abs/1904.10509. arXiv: 1904.10509 version: 1.

Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers.

In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=Ua6zuk0WRH.

Condat, L. Fast projection onto the simplex and the ℓ1 ball.

Math. Program., 158(1–2):575–585, July 2016. ISSN 0025-5610. doi: 10.1007/s10107-015-0946-6. URL https://doi.org/10.1007/s10107-015-0946-6.

Correia, G. M., Niculae, V., and Martins, A. F. T. Adaptively sparse transformers. In Inui, K., Jiang, J., Ng, V., and Wan, X. (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2174–2184, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1223. URL https://aclanthology.org/D19-1223/.

Dai, X., Chalkidis, I., Darkner, S., and Elliott, D. Revisiting transformer-based models for long document classification. In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 7212–7230, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. findings-emnlp.534. URL https://aclanthology. org/2022.findings-emnlp.534/.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. FlashAt- ´
tention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Devoto, A., Zhao, Y., Scardapane, S., and Minervini, P.

A simple and effective l 2 norm-based strategy for KV cache compression. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 18476–18499, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.1027. URL https: //aclanthology.org/2024.emnlp-main.1027/.

Dong, J., Feng, B., Guessous, D., Liang, Y., and He, H. Flex attention: A programming model for generating optimized attention kernels, 2024. URL https: //arxiv.org/abs/2412.05496.

Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T. Efficient projections onto the l1-ball for learning in high dimensions. In Proceedings of the 25th International Conference on Machine Learning, ICML '08, pp. 272–279, New York, NY, USA, 2008. Association for Computing Machinery. ISBN 9781605582054. doi:
10.1145/1390156.1390191. URL https://doi.org/ 10.1145/1390156.1390191.

Gupta, A., Dar, G., Goodman, S., Ciprut, D., and Berant, J. Memory-efficient transformers via top-k attention. In Moosavi, N. S., Gurevych, I., Fan, A., Wolf, T., Hou, Y., Marasovic, A., and Ravi, S. (eds.), ´ Proceedings of the Second Workshop on Simple and Efficient Natural Language Processing, pp. 39–52, Virtual, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.sustainlp-1.5. URL https: //aclanthology.org/2021.sustainlp-1.5/.

Held, M., Wolfe, P., and Crowder, H. P. Validation of subgradient optimization. *Mathematical Programming*, 6 (1):62–88, December 1974.

Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S.,
Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., Yang, Y., and Qiu, L. Minference 1.0: Accelerating pre-filling for longcontext llms via dynamic sparse attention. arXiv preprint arXiv:2407.02490, 2024.

Kaufman, E. H. and Lenker, T. D. Linear convergence and the bisection algorithm. The American Mathematical Monthly, 93(1):48–51, 1986. ISSN 00029890, 19300972. URL http://www.jstor.org/stable/2322546.

Kitaev, N., Kaiser, L., and Levskaya, A. Reformer: The efficient transformer. In International Conference on Learning Representations, 2020. URL https://openreview. net/forum?id=rkgNKkHtvB.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pp. 611–626, 2023.

Liu, J. and Ye, J. Efficient euclidean projections in linear time. In Proceedings of the 26th annual international conference on machine learning, pp. 657–664, 2009.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D.,
Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. Roberta: A robustly optimized bert pretraining approach, 2019. URL https://arxiv.org/abs/1907.11692.

Lozhkov, A., Ben Allal, L., von Werra, L., and Wolf, T.

Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface.co/datasets/
HuggingFaceFW/fineweb-edu.

Luohe, S., Zhang, H., Yao, Y., Li, Z., and hai zhao. Keep the cost down: A review on methods to optimize LLM's KV-cache consumption. In First Conference on Language Modeling, 2024. URL https://openreview.

net/forum?id=8tKjqqMM5z.

Martins, A. and Astudillo, R. From softmax to sparsemax:
A sparse model of attention and multi-label classification. In Balcan, M. F. and Weinberger, K. Q. (eds.),
International Conference on Machine Learning (ICML), volume 48 of *Proceedings of Machine Learning Research*, pp. 1614–1623, New York, New York, USA, 20–22 Jun 2016. PMLR. URL http://proceedings.mlr. press/v48/martins16.html.

Milakov, M. and Gimelshein, N. Online normalizer calculation for softmax. *arXiv preprint arXiv:1805.02867*, 2018.

Pagliardini, M., Paliotta, D., Jaggi, M., and Fleuret, F.

Fast attention over long sequences with dynamic sparse flash attention. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=UINHuKeWUa.

Penedo, G., Kydl´ıcek, H., allal, L. B., Lozhkov, A., Mitchell, ˇ
M., Raffel, C., Werra, L. V., and Wolf, T. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum? id=n6SCkn2QaG.

Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N., and Kong, L. Random feature attention. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= QtTKTdVrFBB.

Peters, B., Niculae, V., and Martins, A. F. T. Sparse sequence-to-sequence models. In *Proceedings of the* 57th Annual Meeting of the Association for Computational Linguistics, pp. 1504–1519, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1146. URL https://www.aclweb. org/anthology/P19-1146.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. 2019.

Roy, A., Saffar, M., Vaswani, A., and Grangier, D. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021. doi: 10.1162/tacl a 00353. URL https://aclanthology.org/2021.tacl-1.4.

Scavo, T. R. and Thoo, J. B. On the geometry of halley's method. *The American Mathematical Monthly*, 102(5):
417–426, 1995. ISSN 00029890, 19300972. URL http: //www.jstor.org/stable/2975033.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P.,
and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision, 2024. URL https: //arxiv.org/abs/2407.08608.

Thakur, N., Reimers, N., Ruckl ¨ e, A., Srivastava, A., and ´
Gurevych, I. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Vanschoren, J. and Yeung, S. (eds.), *Proceedings* of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL
https://datasets-benchmarks-proceedings. neurips.cc/paper_files/paper/2021/file/ 65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2. pdf.

Tillet, P., Kung, H. T., and Cox, D. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL 2019, pp. 10–19, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450367196. doi: 10.1145/3315508.3329973. URL https://doi.org/10.1145/3315508.3329973.

Treviso, M., Gois, A., Fernandes, P., Fonseca, E., and Mar- ´
tins, A. Predicting attention sparsity in transformers. In Proceedings of the Sixth Workshop on Structured Prediction for NLP, pp. 67–81, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.spnlp-1.7. URL https://aclanthology.org/ 2022.spnlp-1.7.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. *Advances in neural* information processing systems, 30, 2017. URL https://papers.nips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract. html.

Velickovi ˇ c, P., Perivolaropoulos, C., Barbero, F., and Pas- ´
canu, R. softmax is not enough (for sharp out-ofdistribution), 2025. URL https://openreview.net/ forum?id=wMj6PgKVuJ.

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I.

Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Korhonen, A., Traum, D., and Marquez, L. (eds.), ` Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 5797–5808, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1580. URL
https://aclanthology.org/P19-1580/.

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Linzen, T., Chrupała, G., and Alishahi, A. (eds.), Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 353– 355, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-5446. URL https://aclanthology.org/W18-5446/.

Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstr ´ om, ¨
O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., et al. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv preprint arXiv:2412.13663, 2024.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=NG7sS51zVF.

Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G.,
Li, Y., and Singh, V. Nystromformer: A nystr ¨ om-based ¨ algorithm for approximating self-attention. 2021.

Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z.,
Xie, Z., Wei, Y., Wang, L., Xiao, Z., et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. *arXiv preprint arXiv:2502.11089*, 2025.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big Bird: Transformers for Longer Sequences. *Advances in Neural Information* Processing Systems, 33:17283–17297, 2020a. URL
https://papers.nips.cc/paper/2020/hash/ c8512d142a2d849725f31a9a7a361ab9-Abstract. html.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q.,
Yang, L., et al. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33, 2020b.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In *Proceedings of the 57th Annual Meeting of the* Association for Computational Linguistics, 2019.

## A. Algorithm Details

We first derive a high-level view of the forward and backward passes of the entmax attention and then present the full algorithms for both mentioned versions. For consistency and ease of comparison, we follow the notation adopted by FlashAttention-1 (Dao et al., 2022).

## A.1. Α**-Entmax Attention Forward Pass**

We recall that given the input sequences Q, K,V ∈ R
n×d, we want to compute the attention output O ∈ R
n×das follows:

$$\mathbf{S}=\mathbf{Q}\mathbf{K}^{\mathsf{T}}\in\mathbb{R}^{n\times n},\ \mathbf{P}=\alpha\text{-ent}$$
$$\in\mathbb{R}^{n\times d}$$
$\in\mathbb{R}^{n\times n}$, $\mathbf{P}=\alpha$-entmax$(\mathbf{S})\in\mathbb{R}^{n\times n}$, $\mathbf{O}=\mathbf{PV}\in\mathbb{R}$.  
Therefore all we need is the τ ∈ R
n that solves Equation 2, for which we can use Algorithm 1. We note that we do not need to materialize S as we only need to accumulate the derivatives of f(τ ), defined in Equation 3. Once τ is computed, we can compute each row of O as follows:

$$\mathbf{O}_{i}=\mathbf{P}_{i}\mathbf{V}=\sum_{j}P_{ij}\mathbf{V}_{j}=\sum_{j=1}^{n}\max\left(0,(\alpha-1)\mathbf{Q}_{i}^{\top}\mathbf{K}_{j}-\tau_{i}\right)^{1/\alpha-1}\mathbf{V}_{j}\tag{1}$$

As in FlashAttention, we can compute Oi without extra memory by incrementally summing the contributions of each α-entmax(Q⊤
i Kj )Vj term. We can then compute the forward pass with O (n) extra memory as follows:
1. Compute τi for all 1 ≤ i ≤ n according to Algorithm 1, which takes O (n) extra memory.

2. Compute Oi for all 1 ≤ i ≤ n according to Equation 15 which takes O (n) extra memory.

## A.2. Α**-Entmax Attention Backward Pass**

For the α-entmax attention backward pass, we need to compute the gradients with respect to V , K, and Q. Let L be a scalar loss function, and dO ∈ R
n×d denote ∂L
∂O
. Our goal is to compute the input gradients dV , dK, dQ ∈ R
n×d.

## 1. Gradient Of V

Using reverse-mode autodifferentiation, we first compute dV :

$$(15)$$
$$d V=P^{\top}d O,$$
$$(16)$$
$$(17)$$
⊤dO, (16)
where P = α-entmax(S) is the output of the α-entmax transformation applied row-wise to the score matrix S = QK⊤.

Expressed element-wise, we obtain:

$dV_{j}=\sum_{i=1}^{n}P_{ij}dO_{i}$, (13.1)
i=1
which is analogous to the softmax case. Since Pij is sparse due to the nature of α-entmax, we can skip Qi blocks that leads to blocks of P full of zeros using the pointer increment tables, as shown in Equation 14.

## 2. Gradient Of P And S

The next step involves computing dP and dS. From O = P V , we have:

$${\mathfrak{P V}},{\mathrm{~we~have:}}$$
$$d P_{i j}=d{\cal O}_{i}^{\top}V_{j}.$$
i Vj . (18)
Next, let us recall the Jacobian of the α-entmax mapping (Peters et al., 2019). Defining p = α-entmax(s), the Jacobian is:

$${\frac{\partial\alpha\mathrm{-entmax}(\mathbf{s})}{\partial\mathbf{s}}}=\mathrm{Diag}(\mathbf{u})-{\frac{\mathbf{u}\mathbf{u}^{\top}}{\|\mathbf{u}\|_{1}}},$$
, (19)
$$(18)$$

$$(19)$$

13 where u is defined element-wise as:

$$u_{k}=\begin{cases}(p_{k})^{2-\alpha},&\text{if}p_{k}>0\\ 0,&\text{otherwise.}\end{cases}\tag{1}$$
$$(20)$$

Let U denote a stack of [u1*, ...,*un] for each row of P . From the relationship P = α-entmax(S), and the Jacobian of the α-entmax function, we can propagate the gradients back to S as follows:

$$\mathbf{dS}_{i}=\left[\text{Diag}(\mathbf{U}_{i})-\frac{\mathbf{U}_{i}\mathbf{U}_{i}^{\top}}{\|\mathbf{U}_{i}\|_{1}}\right]\mathbf{dP}_{i}$$ $$=\mathbf{U}_{i}\odot\mathbf{dP}_{i}-\left(\frac{\mathbf{U}_{i}^{\top}\mathbf{dP}_{i}}{\|\mathbf{U}_{i}\|_{1}}\right)\mathbf{U}_{i}.$$

We can further simplify by defining a new quantity δ ∈ R
n:

$$\delta_{i}=\frac{\boldsymbol{U}_{i}^{\top}\boldsymbol{d}\boldsymbol{P}_{i}}{\|\boldsymbol{U}_{i}\|_{1}}$$ $$=\frac{1}{\|\boldsymbol{U}_{i}^{\top}\|_{1}}\sum_{j=1}^{n}U_{ij}\left(\boldsymbol{d}\boldsymbol{O}_{i}^{\top}\boldsymbol{v}_{j}\right)$$ $$=\boldsymbol{d}\boldsymbol{O}_{i}^{\top}\underbrace{\left(\sum_{j=1}^{n}U_{ij}\boldsymbol{V}_{j}\right)}_{\|\boldsymbol{U}_{i}^{\top}\|_{1}}$$
(23)  $\binom{24}{24}$  . 
(21)  $\binom{22}{2}$  (22)  ... 
$$(25)$$
$$(26)$$
(27)  $\binom{28}{2}$  (28)  ... 
In standard softmax attention, instead of the right-side term in the above product, we would simply obtain Oi. Since this new quantity is required for the backward pass, and to avoid passing once more through Q, K and V , we compute and store this quantity during the forward pass solely during training. Unlike in softmax attention, however, the backward pass for α-entmax does not require saving the output matrix O; instead, we only require this new quantity, which we label O(2).

Then, we can simplify the computation of dS to:

$$d{\mathbf{S}}_{i}={\mathbf{U}}_{i}\odot(d{\mathbf{P}}_{i}-\delta_{i})$$
dSi = Ui ⊙ (dPi − δi) (26)
Again, we can use the sparsity stored in M (see Equation 9) from the forward pass to efficiently skip the computation of null blocks of P . 3. Gradients of Q and K
Using the definition of Sij = Q⊤
i Kj , the gradients for Q and K are:

$$d\mathbf{Q}_{i}=\sum_{j=1}^{n}dS_{ij}\mathbf{K}_{j},\tag{1}$$ $$d\mathbf{K}_{j}=\sum_{i=1}^{n}dS_{ij}\mathbf{Q}_{i}.$$

Substituting dSij , we get:

$$\begin{split}\boldsymbol{dQ}_{i}&=\sum_{j=1}^{n}U_{ij}\left(dP_{ij}-\delta_{i}\right)\boldsymbol{K}_{j}\\ \boldsymbol{dK}_{j}&=\sum_{i=1}^{n}U_{ij}\left(dP_{ij}-\delta_{i}\right)\boldsymbol{Q}_{i}\end{split}$$
$$(29)$$
$$(30)$$

Effectively, we can only iterate through the blocks that will result in Pij ̸= 0. As in FlashAttention, the backward pass can also be computed with O (n) extra memory:
1. Compute dVj for all j according to Equation 17, which takes O (d) extra memory. 2. Compute δi for all i according to Equation 23, which takes O (n) extra memory. 3. Compute O
(2)
ifor all i, as defined in Equation 25, which takes O (d) extra memory.

4. Compute dQi for all i according to Equation 29, which takes O (d) extra memory. 5. Compute dKj for all j according to Equation 30, which takes O (d) extra memory.

We note that the only extra memory requirement compared to FlashAttention is in having to additionally compute and storing O(2) ∈ R
n×d. When using block masking, we also need O (Tr × Tc) extra memory to store the binary mask M. However, we recall that this memory can be shared across attention layers, as it is merely a temporary matrix used to compute the pointer-increment tables.

## A.3. Adasplash**: Forward Pass (Without Block Masking)**

The full ADASPLASH's forward pass is presented in Algorithm 2. For completeness, we also provide in Algorithm 3 the steps for approximating τ without the need to materialize S in a block-wise manner. Algorithm 3 Halley-bisection for computing τ - Block Version Require: Matrices Q, K ∈ Rn×din HBM, block sizes Bc, Br and number of iterations M.

1: Divide Q into Tr = ⌈n/Br⌉ blocks Q1*, . . . ,* QTr of size Br × d 2: Divide K into Tc = ⌈n/Bc⌉ blocks K1*, . . . ,* KTcof size Bc × d 3: Divide τ into Tr blocks τ1*, . . . ,* τTrof size Br 4: for i = 1 to Tr do 5: Load Qi from HBM to on-chip SRAM
6: On chip, initialize τi, τloi, τhiiaccording to Algorithm 1. ▷ Note: this requires one pass over Kj for all j.

7: **repeat**
8: On chip, initialize f, f′, f′′= 0 ∈ R
Br 9: for j = 1 to Tc do 10: Load Kj , Vj from HBM to on-chip SRAM
11: Compute S
(j)
i = QiK⊤
j ∈ R
Br×Bc 12: Accumulate f, f′, f′′ according to Equations 3, 6 and 7, respectively.

13: **end for**
14: Update τi, τloi, τhiiaccording to Algorithm 1.

15: **until** M iterations are completed 16: Write τito HBM
17: **end for** 18: **Return:** τ

## A.4. Adasplash**: Backward Pass (Without Block Masking)**

As mentioned in §3.2.2, in contrast to FlashAttention, we propose to separate the kernels that compute the gradients dQ, dK, dV . However, as in FlashAttention, we need to compute δ before being able to compute the gradients, which we do in a separate kernel following Equation 25. We present the full steps for computing dK and dV in Algorithm 4, and for computing dQ in Algorithm 5.

## A.5. Adasplash**: Block Masked Version**

In this version, as outlined in Section 3, a boolean block mask M ∈ R
Tr×Tcis created dynamically in the forward pass, allowing the exploitation of the sparsity in the matrix P at the cost of linear memory complexity. The mask is populated Algorithm 4 ADAS Backward Pass for dK and dV
Require: Matrices Q, K,V , O, dO ∈ R
n×din HBM, vector τ ∈ R
n in HBM, block sizes Bc, Br, parameter α 1: Divide Q into Tr = ⌈n/Br⌉ blocks Q1*, . . . ,* QTrof size Br × d each, and divide K,V into Tc = ⌈n/Bc⌉ blocks K1, . . . , KTc, V1*, . . . ,* VTcof size Bc × d each.

2: Divide dO into Tr blocks dO1*, . . . ,* dOTrof size Br × d each. 3: Divide τ into Tr blocks τ1*, . . . ,* τTr of size Br each.

4: Initialize and divide dK, dV ∈ R
n×dinto Tc blocks dK1*, . . . ,* dKTcand dV1*, . . . ,* dVTc of size Bc × d each.

5: Divide δ into Tr blocks δ1*, . . . ,* δTrof size Br each. 6: for 1 ≤ j ≤ Tc do 7: Load Kj ,Vj from HBM to on-chip SRAM.

8: Initialize dKj = 0Bc×d on SRAM. 9: Initialize dVj = 0Bc×d on SRAM.

10: for 1 ≤ i ≤ Tr do 11: Load Qi, dOi, τi, δi from HBM to on-chip SRAM.

12: On chip, compute S
(j)
i = QiK⊤
j ∈ R
Br×Bc.

13: On chip, compute P
(j)
i = max(0,(α − 1)S
(j)
i − τi)
1/α−1 ∈ R
Br×Bc.

14: On chip, compute dVj ← dVj + (P
(j)
i)
⊤dOi ∈ R
Bc×d.

15: On chip, compute dPi = dOiV
⊤
j ∈ R
Br×Bc.

16: On chip, compute U
(j)
i = P
(j)
i 2−α∈ R
Br×Bc.

17: On chip, compute dS(j)
i = U
(j)
i ⊙ (dP 
(j)
i − δi) ∈ R
Br×Bc.

18: On chip, compute dKj ← dKj + (dS(j)
i)
⊤Qi ∈ R
Bc×d.

19: **end for**
20: Write dKj , dVj to HBM.

21: **end for** 22: **Return:** Gradients dK, dV .

during the final iteration of the Halley-bisection algorithm (Algorithm 3) by evaluating the condition any(S
(j)
i > τi) and storing the result as a boolean value. Thus, the mask M indicates whether a specific Q, K block pair contributes to the output. This process enables the creation of a lookup table that associates each query block with the set of key blocks that contribute non-zero values, thereby allowing to skip unnecessary computations for future computations. Similarly, a reverse lookup table can be created for each key block. Both tables can be used in the backward pass (Line 10 in Algorithm 4 and Line 9 in Algorithm 5) to avoid looping over unnecessary query/key blocks.

In practice, to create the lookup tables, we use the torch.argwhere function to extract the (*i, j*) indices of entries where Mij = 1. Combined with row-wise summation of non-zero entries, this approach efficiently skips computations for irrelevant blocks within the remaining kernels. Consequently, during the forward pass, only the K,V pairs identified in the lookup table are loaded, avoiding redundant memory and computational overhead. As mentioned, for the backward pass, given that we separated the computation of dQ and dK, dV , we can further use both tables (Q and K) to speedup the gradient computation.

## B. Experimental Setup B.1. Continuous Pre-Training

We conducted continuous pretraining of RoBERTa-base3and ModernBERT-base4 models with our custom sparse attention Triton kernel, ADASPLASH. The pretraining process was carried on 2B tokens of the FineWeb-Edu dataset,5 due to its high-quality, diverse and large-scale content. We used the HuggingFace Transformers library for model training and implementation and the Datasets library for data handling. Concretely, we used a batch size of 32 and a learning rate of 5 × 10−5, optimized with the AdamW optimizer. Training was conducted for 100,000 steps using mixed-precision (fp16).

3https://huggingface.co/FacebookAI/roberta-base 4https://huggingface.co/answerdotai/ModernBERT-base 5https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu Algorithm 5 ADAS Backward Pass for dQ
Require: Matrices Q, K,V , O, dO ∈ R
n×din HBM, vector τ ∈ R
n in HBM, block sizes Bc, Br, parameter α.

1: Divide Q into Tr = ⌈n/Br⌉ blocks Q1*, . . . ,* QTrof size Br × d each, and divide K,V into Tc = ⌈n/Bc⌉ blocks K1, . . . , KTc, V1*, . . . ,* VTcof size Bc × d each.

2: Divide dO into Tr blocks dO1*, . . . ,* dOTrof size Br × d each. 3: Divide τ into Tr blocks τ1*, . . . ,* τTr of size Br each.

4: Initialize dQ in HBM and divide it into Tr blocks dQ1*, . . . ,* dQTr of size Br × d each.

5: Divide δ into Tr blocks δ1*, . . . ,* δTrof size Br each.

6: for i = 1 to Tr do 7: Load Qi, dOi, δi, τi, from HBM to on-chip SRAM
8: Initialize dQi = 0Bc×d on SRAM.

9: for j = 1 to Tc do 10: On chip, compute S
(j)
i = QiK⊤
j ∈ R
Br×Bc.

11: On chip, compute P
(j)
i = max(0,(α − 1)S
(j)
i − τi)
1/α−1 ∈ R
Br×Bc.

12: On chip, compute dPi = dOiV
⊤
j ∈ R
Br×Bc.

13: On chip, compute U
(j)
i = P
(j)
i 2−α∈ R
Br×Bc.

14: On chip, compute dS(j)
i = U
(j)
i ⊙ (dP (j)
i − δi) ∈ R
Br×Bc.

15: On chip, compute dQi ← dQi + dS(j)
i Kj ∈ R
Br×d.

16: **end for**
17: Write dQito HBM
18: **end for** 19: **Return:** Gradient dQ
Table 5. Runtime (s) of ModernBERT-base (α *= 1.*5) for varying context lengths.

| Sequence Length           |      |      |      |      |      |
|---------------------------|------|------|------|------|------|
| Algorithm                 | 512  | 1024 | 2048 | 4096 | 8192 |
| Sorting (Torch)           | 0.09 | 0.11 | 0.26 | 0.76 | OOM  |
| Bisection (Torch)         | 0.11 | 0.15 | 0.42 | 1.35 | 4.99 |
| Halley-bisection (Triton) | 0.10 | 0.11 | 0.26 | 0.46 | 1.61 |
| ADASPLASH (Triton)        | 0.10 | 0.12 | 0.21 | 0.48 | 1.53 |

The sparsity parameter (α) was initialized at 1.01 and annealed linearly to a final value of 1.5 or 2.0 over 50,000 steps. We kept ModernBERT's window attention layers untouched, only replacing the full softmax layers by α-entmax. Finally, we also performed continuous pretraining of RoBERTa and ModernBERT with standard softmax attention with a fixed α = 1.0. As shown in Figure 5, the attention mechanisms of our sparse ModernBERT model (α = 1.5) obtain high sparsity levels in practice, with an overall sparsity of 95% for α = 1.5 and 99% for α = 2.0. For this reason, we used the version of ADASPLASH that leverages the pointer increment tables for training ModernBERT, which has a maximum sequence length of 8,192. For RoBERTa, which has a sequence length of 512, we opted to use the Halley-bisection algorithm implemented in Triton. In Table 5 we report efficiency results in terms of runtime and memory usage for different attention algorithms with ModernBERT-base. Overall, we observe that the sorting approach is slower than bisection, which is slower than our Halley-bisection and ADASPLASH, in that order.

## B.2. Glue And Bier Tasks

For GLUE tasks, we used the checkpoints of continuous pre-trained models for both RoBERTa-base and ModernBERT-base.

Then, we fine-tuned them on each GLUE task with the default hyperparameters from the Transformer library.6Importantly, we capped the maximum sequence length at 128 tokens to reduce computational cost while preserving task-relevant context and used fp16 for training.

6https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0 0 1 1 2 2 3 3 4 4 Hea ds Hea ds 5 5 6 6 7 7 8 8 9 9 10 10 11 11 0 3 6 9 12 15 18 21 Layers 0 3 6 9 12 15 18 21 Layers

Model Params Seq. CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE Avg.

| Single Sentence   | Paraphrase and Similarity   | Natural Language Inference   |
|-------------------|-----------------------------|------------------------------|

BERT 110M 512 58.6 91.9 86.9 89.0 89.3 84.0 91.0 69.3 82.5 RoBERTa 125M 512 59.8 93.7 89.5 89.6 89.8 87.7 92.3 69.3 83.9

RoBERTa (α = 1.5) 125M 512 58.5 93.2 91.5 90.2 89.7 87.3 92.5 68.6 83.9 RoBERTa (α = 2.0) 125M 512 56.8 93.0 90.9 88.8 89.0 86.7 91.9 67.2 83.0

ModernBERT 149M 8192 63.2 95.0 88.2 90.3 90.4 87.9 93.0 61.7 83.7 ModernBERT (α = 1.5) 149M 8192 62.2 96.1 87.7 89.4 90.2 87.9 92.6 61.7 83.5 ModernBERT (α *= 2.*0) 149M 8192 62.2 94.8 89.0 89.9 90.5 87.8 93.1 62.5 83.7

To evaluate the generalization of ADASPLASH in retrieval tasks, we fine-tuned ModernBERT-base and RoBERTa-base models on the MS MARCO dataset (Bajaj et al., 2016) and evaluated them on the BEIR benchmark (Thakur et al., 2021). This benchmark suite assesses performance across diverse information retrieval tasks, including SciFact, NFCorpus, FiQA- 2018, and TREC-COVID. The fine-tuning and evaluation process closely follows the approach proposed in the ModernBERT
paper (Warner et al., 2024). Fine-tuning was performed using the SentenceTransformers library.7 The models were evaluated on BEIR tasks using the MTEB benchmark toolkit.8 The evaluation metric for each task was nDCG@10 (Normalized Discounted Cumulative Gain), following standard information retrieval practices.

## B.3. Long Document Classification

The European Court of Human Rights (ECtHR) dataset comprises legal cases from the European Court of Human Rights, each associated with specific articles of the Convention on Human Rights allegedly violated. For this task, we fine-tuned the RoBERTa base model (Liu et al., 2019) with a classification head. Since this is a multi-label classification task, we used a binary cross-entropy loss. To accommodate longer contexts, we followed the approach proposed by (Beltagy et al., 2020), repeating the 512 position embeddings until the target context size was reached. We used the AdamW optimizer for training. For hyperparameters, we follow the recipe of Dai et al. (2022). For the attention mechanism, bfloat16 precision was used.

## B.4. Language Modeling

We trained both the standard GPT-2 model and sparse GPT-2 (α = 1.5) using the configuration provided in the llm.c repository,9following their training recipe. Specifically, we trained a GPT-2 (124M parameters) from scratch on 10B
tokens of the FineWeb dataset, with a maximum sequence length of 1024 tokens. Training was conducted using bfloat16 precision. We use an effective batch size of 512, and use gradient accumulation to fit into available GPU memory. We 7https://sbert.net/ 8https://github.com/embeddings-benchmark/mteb 9https://github.com/karpathy/llm.c use the AdamW optimizer, with learning rate 6 × 10−4and weight decay of 0.1. The learning rate followed a warm-up phase, linearly ramping from zero to a maximum of 6 × 10−4 over the first 700 iterations, equivalent to 350 million tokens.

Subsequently, the learning rate decayed to zero across the remaining training steps. We show the validation loss curves for both softmax and α-entmax (α = 1.5) in Figure 6. Given that, for this task, the context size was not high enough, for sparse attention we opted to use the algorithm that does not take advantage of the pointer increment tables. For the benchmarking of the time spent per step, we averaged across 50 steps after the model had trained for at least 100 steps.

0 2500 5000 7500 10000 12500 15000 17500 20000 Validation Step 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 GPT-2 Sparse GPT-2 Va lid ati on L
os s

## C. Computational Details

Experiments on masked language modeling, text classification, GLUE tasks and BIER tasks were carried on Nvidia RTX
A6000 GPUs with 48GB VRAM. Experiments with GPT-2 and the efficiency benchmark in Figures 1 and 3 were carried on a single Nvidia H100 GPU (80GB). The runtime experiments with ModernBERT were carried on a single A6000 GPU.