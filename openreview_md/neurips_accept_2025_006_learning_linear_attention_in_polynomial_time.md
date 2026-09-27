| Morris Yau MIT CSAIL                         |                                     |
|----------------------------------------------|-------------------------------------|
| morrisy@mit.edu                              | Ekin Akyürek MIT CSAIL              |
| akyurek@mit.edu                              | Jiayuan Mao MIT CSAIL               |
| jiayuanm@mit.edu                             |                                     |
| Joshua B. Tenenbaum                          |                                     |
| MIT Brain and Cognitive Sciences jbt@mit.edu | Stefanie Jegelka                    |
| TUM Munich, MCML, MIT CSAIL stefje@mit.edu   | Jacob Andreas MIT CSAIL jda@mit.edu |

# Learning Linear Attention In Polynomial Time

## Abstract

Previous research has explored the expressivity of Transformer models in simulating Boolean circuits or Turing machines. However, the efficient learnability of Transformers from data has remained an open question. Our study addresses this gap by providing the first polynomial-time learnability results (specifically strong, agnostic PAC learning) for single-layer Transformers with linear attention. We show that learning the optimal multi head linear attention can be recast as finding the optimal kernel predictor in a suitably defined RKHS. Moving to generalization, we construct an algorithm that, given a dataset, checks in polynomial time whether the set of best fit multi head linear attention networks on this data all perform an identical computation–a powerful notion for out of distribution generalization. We empirically validate our theoretical findings on several canonical tasks: learning random linear attention networks, key–value associations, and learning to execute finite automata. Our findings bridge a critical gap between theoretical expressivity and learnability of Transformer models.

## 1 Introduction

Transformers are the dominant neural architecture used in language modeling. A growing body of work seeks to explain the behavior of trained Transformers and characterize their learnability [Pérez et al., 2019, Edelman et al., 2022b, Hahn, 2020, Merrill and Sabharwal, 2023, Merrill et al., 2022, 2021, Liu et al., 2022, Feng et al., 2023, Edelman et al., 2022a, Wei et al., 2021, Zhang et al., 2024, Trauger and Tewari, 2023, Chen and Li, 2024]. While a large body of work shows that Transformers are *expressive* enough to implement important models of computation, it remains an open question whether these constructions may be efficiently *learned*. Even verifying that a trained model has successfully learned a computational procedure (uniform circuit family) has remained challenging.

Existing work shows positive results on how Transformer-like architectures can express diverse computations, including simulating universal Turing machines [Li et al., 2024], evaluating sentences of first-order logic [Barceló et al., 2020], and recognizing various formal languages [Strobl et al.,
2024]. On the other hand, results on learnability in polynomial time and samples with provable guarantees tend to rely on strong data-generating assumptions, e.g., Gaussian data, etc. [Zhang et al.,
2023, Jelassi et al., 2022, Tian et al., 2023, Oymak et al., 2023, Fu et al., 2023, Tarzanagh et al., 2024, Deora et al., 2023]. This brings us to our first motivating question.

Is there an efficient algorithm in time and samples that learns the optimal parameters of a class of Transformer models for any dataset?

In this paper, we establish the strong, agnostic PAC-learnability of linear attention. Linear attention variants (kernel, gated, flash, etc.) Yang et al. [2025, 2024], mLSTM in xLSTM Beck et al. [2024], Retnet Sun et al. [2023], Mamba2 Dao and Gu [2024], DeltaNet Schlag et al. [2021]) have recently matched or outperformed softmax attention in language and vision benchmarks, underscoring the practical value of their theory; Ahn et al., 2024, Katharopoulos et al., 2020). Despite its name, linear attention is not linear and its loss landscape is nonconvex. We focus our analysis on multi-head linear attention networks, or MHLAs for regression tasks. An MHLA is parameterized by two matrices (Vh, Qh) for each of H heads as such Θ = {(Vh, Qh)}h∈[H]. A one layer MHLA computes Y =Ph∈[H]VhZ(Z
T QhZ). Here key and query matrices are fused into one, as they multiply one another directly. We first show that the computation performed by MHLAs can be reformulated as an elementwise product between two larger matrices ⟨W, X (Z)⟩, where W =Ph∈[H]flatten(Vh)flatten(Qh)
Tand X (Z) is a fixed cubic polynomial function of Z. Consequently, optimizing over the class of H-head MHLA models is equivalent to optimizing over the class of rank-H matrices W. Furthermore, in the full-rank space of d 2 × d 2 matrices, optimization of W can be performed via linear regression with time polynomial in the inverse target error and size of the dataset. Finally, decomposing an optimal W via SVD recovers an MHLA model with no more than d 2 heads that is then guaranteed to compete against the best MHLA parameters—establishing our agnostic learning result (the learned model competes against the best choice of parameters in the hypothesis class). Next, achieving zero training and validation loss does not by itself certify that a model has learned a target computation well enough to generalize out of distribution. Imagine learning arithmetic from input output pairs alone. Many distinct parameter settings can fit the same data, and fail for larger length inputs. We therefore ask:
Is there a data-dependent, efficiently checkable condition that forces every empirical-risk minimiser to realise the same function? For MHLAs the answer is yes. Define the second-moment matrix of the cubic feature map X as

$=\;\mathbb{E}_{(Z,y)\in D}\big|\mathcal{X}(Z)$

## Λd = E(Z,Y)∈D -X (Z) X (Z) ⊤.

If ΛD is full rank—our *certifiable identifiability* criterion—then all empirical-risk minimisers of MHLA coincide on every input. The test runs in polynomial time and is unaffected by parameter redundancies such as rescaling V and Q. Combining this certificate with our expressivity result yields a polynomial-time procedure that (i) learns any circuit family implementable by MHLA whenever the training data satisfy the criterion, and (ii) provably recovers, for example, a bounded-history universal Turing machine from its input–output traces (Appendix C). Once learned, the MHLA simulates any such Turing machine on any input within the prescribed size budget. In the experimental section, we validate our theoretical findings. In Section 4.1, we train multiple models using stochastic gradient descent on a dataset generated by a single linear attention network's output. Our results demonstrate that multi-head linear attention outperforms both single-layer linear attention and multi-layer linear attention, achieving comparable results to our Algorithm 1. In Section 4.2, we show that our proposed certificate directly correlates with generalization error even for models trained using stochastic gradient descent. In summary:
- We provide a polynomial time algorithm that, given any dataset, finds the best fit parameters for multi head linear attention and generalizes with polynomial data, i.e., strong agnostic PAC learning (Section 2.1).

- We find an efficiently checkable condition (certifiable identifiability) on the training dataset that certifies every empirical risk minimizer of a MHLA is functionally equivalent, and therefore has the same behavior out of distribution (Appendix A see Lemma A.3).

Algorithm 1 MHLA Learning via Regression 1: **Input:** Data D := {(Zi, yi)}i∈[N] for Zi ∈ R
d×ni and y ∈ R
d 2: {Xi}i∈[N]:= ExtractFeature(D), generates

Xi :=   ⟨z1:, z1:⟩z1ni⟨z1:, z2:⟩z1ni· · · ⟨z1:, zd:⟩zdni ⟨z2:, z1:⟩z1ni⟨z2:, z2:⟩z1ni· · · ⟨z2:, zd:⟩zdni ............ ⟨zd:, z1:⟩z1ni⟨zd:, z2:⟩z1ni· · · ⟨zd:, zd:⟩zdni   2×d
. (1)
3: Create dataset {Xi,a}i∈[N],a∈[d]. Let Xi,a ∈ R
d 2be a matrix that is comprised of Xi in the a
′th block of d rows and 0 everywhere else:
4:
i*. . .* 0T(2)

$$X_{i,a}=\begin{bmatrix}0&\dots&\mathcal{X}_{i}^{T}&\dots&0\end{bmatrix}^{T}$$
 5: Let $\hat{W}\in\mathbb{R}^{d^2\times d^2}$ be regressor. 
where yi,a is the a'th coordinate of yi.

6: Take the SVD of Wˆ = ABT =Pi∈[Hˆ ] AiB
T
i where Hˆ is the rank of Wˆ .

7: Vh = Fold(Ah) and Qh = Fold(Bh) where Fold : R
d 2 → R
d×dtakes a vector p *:= [p*ij for i ∈
[d] and j ∈ [d]] and reshapes into a matrix P ∈ R
d×dsuch that Pij = pij .

8: **Return:** {Vh, Qh}h∈[Hˆ ]

## 2 Technical Overview

We start with basic definitions of a multi-head linear attention (MHLA) module, an attention module without the softmax activation.

Definition 2.1 (Multi-Head Linear Attention). Let Z ∈ R
d×n be a matrix of input data. Let Θ = {(Vh, Qh)}h∈[H] be a set of parameters where each Vh, Qh ∈ R
d×d denotes value and keyquery matrices for all heads h ∈ [H]. We say Θ ∈ ΩH where ΩH is the space of sets of H
ordered tuples of d × d matrices. We define *multi-head linear attention (MHLA)* to be the function MHLAΘ : R
d×n → R
d×n,

$$\hat{Y}=\mathrm{MHLA}_{\Theta}(Z)=\sum_{h\in[H]}V_{h}Z(Z^{T}Q_{h}Z)\;,$$
$$(4)$$
$$(5)$$

where Yˆ ∈ R
d×n is the output of the one layer linear attention. We will primarily be interested in the rightmost column vector output by MHLAΘ (e.g., as in auto-regressive language models), which is:

$$\hat{y}=\mathrm{MHLA}_{\Theta}(Z)=\sum_{h\in[H]}V_{h}Z(Z^{T}Q_{h}Z[:,n])\ ,$$

where Z[:, n] is the n'th column of Z.

## 2.1 Polynomial-Time Learnability

Our main result is that MHLA is learnable in polynomial time. Colloquially, Algorithm 1 returns an MHLA that attains the global minimum of the training loss and requires as few as poly(d, ϵ−1, *log(*δ
−1)) samples to achieve ϵ generalization error with probability 1 − δ. Our algorithmic guarantees do not require the data to be "realizable" (that is, the data need not be generated by an underlying MHLA).

Theorem 2.2 (Learnability of Linear Attention). Let D be a dataset D = {Zi, yi}i∈[N] drawn i.i.d.

from a distribution D *where each* Zi ∈ R
d×ni, yi ∈ R
d. The embedding dimension d is fixed across the dataset, whereas ni can be different for each datapoint. Let nmax = maxi∈[N] ni be the maximum sequence length, and let ΩH be the space of H pairs of value and key-query matrices {(Vh, Qh)}h∈[H]
for any H ∈ [1, ∞). Then there is an algorithm (Algorithm 1) that runs in time O(N d4nmaxϵ
−1)
and that, given input–output pairs {(Zi, yi)}i∈[N]*, returns* Θ =ˆ {(Vˆh, Qˆh)}h∈[Hˆ ] ∈ ΩHˆ for Hˆ ≤ d 2

$$\mathrm{(1)}$$

$$\left(2\right)$$

$\hat{W}:=\arg\min\limits_{W\in\mathbb{R}^{d^{2}}\times d^{2}}\sum\limits_{i\in[N]}\sum\limits_{a\in[d]}(\langle W,X_{i,a}\rangle-y_{i,a})^{2}$
$$({\mathfrak{I}})$$
2(3)
such that with probability 1 − δ,

$$\mathbb{E}_{(Z,y)\in{\mathcal{D}}}\left[\|M H L A_{\hat{\Theta}}(Z)-y\|^{2}\right]$$
$$-\min_{\Theta\in\Omega_{H}}\mathbb{E}_{(Z,y)\in\mathcal{D}}\left[\|MHLA_{\Theta}(Z)-y\|^{2}\right]\leq\epsilon\tag{6}$$

with sample complexity N = O1ϵ d 4 + log(δ
−1).

Below we describe the high-level ideas behind the algorithm; a formal proof is given in Appendix D. Note that if we are purely concerned with guaranteeing that we can find a global minimum of the training loss, we may remove the i.i.d. assumption: Algorithm 1 is always within error ϵ of the optimal training loss. This is also detailed in Appendix D. Specific issues related to generalization over autoregressive sequences rather than i.i.d. data are handled in the UTM learning result with a standard union bound on the sample complexity; see Section F.2.

The main idea behind Algorithm 1 is to construct a feature mapping X : R
d×n → R
d×d 2from the data covariates Z with entries zij for the entry in the i'th row and j'th column and rows z1:, z2:*, ..., z*d: ∈ R
n to a feature space of dimension d × d 2. The map X (Z) is defined as:
X (Z) :=

$$\begin{array}{ccccc}\left[\langle z_{1:},z_{1:}\rangle z_{1n}&\langle z_{1:},z_{2:}\rangle z_{1n}&\cdots&\langle z_{1:},z_{d:}\rangle z_{dn}\right]\\ \langle z_{2:},z_{1:}\rangle z_{1n}&\langle z_{2:},z_{2:}\rangle z_{1n}&\cdots&\langle z_{2:},z_{d:}\rangle z_{dn}\\ \vdots&\vdots&\ddots&\vdots\\ \langle z_{d:},z_{1:}\rangle z_{1n}&\langle z_{d:},z_{2:}\rangle z_{1n}&\cdots&\langle z_{d:},z_{d:}\rangle z_{dn}\end{array}\right].\tag{7}$$

Here, we index the rows of X (Z) by j ∈ [d] and the columns by all tuples (*k, ℓ*) ∈ [d]
2such that X (Z)j,(k,ℓ) = ⟨zj:, zk:⟩zℓn. At a high level, Algorithm 1 is a kernel method defined by the feature mapping X . The learned kernel predictor (a regressor) can be mapped back onto a set of parameters
{Vˆh, Qˆh}h∈Hˆ for an MHLA with no more than d 2 heads via SVD. Hence, the relaxation translates into more heads. Interestingly, in our experiments in Section 4.1, d 2 heads also benefit learning with SGD. Proof Idea: Much of the notation in this section is defined in Algorithm 1. First we write down the loss, and observe that a one-layer attention network is a quadratic polynomial in {Vh, Qh}h∈[H]
with input features Xi,a:

$${\mathcal{L}}_{\Theta}(\{(Z_{i},y_{i})\}_{i\in[N]})={\frac{1}{N}}\sum_{i\in[N]}\sum_{a\in[d]}(\langle{\mathcal{T}}_{\Theta},X_{i,a}\rangle-y_{i,a})^{2}$$

with

$${\mathcal{T}}_{\Theta}:=\sum_{h\in[H]}\mathrm{flatten}(V_{h})\mathrm{flatten}(Q_{h})^{T}$$
$$({\boldsymbol{8}})$$
$$=\sum_{h\in[H]}{\begin{bmatrix}V_{h,00}Q_{h,00}&V_{h,00}Q_{h,01}&\dots&V_{h,00}Q_{h,d d}\\ V_{h,01}Q_{h,00}&V_{h,01}Q_{h,01}&\dots&V_{h,01}Q_{h,d d}\\ \vdots&\vdots&\vdots\\ V_{h,d d}Q_{h,00}&V_{h,d d}Q_{h,01}&\dots&V_{h,d d}Q_{h,d d}\end{bmatrix}}$$

Now we relax this objective by replacing TΘ with an unconstrained matrix W ∈ R
d 2×d 2. While TΘ is a rank-H matrix, we allow W to be a general matrix, so this relaxation is guaranteed to have a smaller loss. Furthermore, the loss can be optimized via ordinary least squares. Finally, if we apply SVD to W we obtain a set of d 2left and right singular vectors scaled by the square root the magnitude of the singular value. Here the scaled left singular vectors correspond to Vˆh and the scaled right singular vectors correspond to Qˆh for h ∈ [Hˆ ]. Since the rank of W is no greater than d 2the resulting MHLA satisfies Hˆ ≤ d 2. The sample complexity follows from classical results in VC
theory [Kearns and Vazirani, 1994]. For a full proof see Appendix D.

## 2.2 Identifiability

A direct implication of our algorithmic result is the construction of an efficiently checkable condition on the data that guarantees every empirical risk minimizer in a family of MHLAs computes the
same function. Let ΛD be the second moment of a specific mapping H(Z) of the data, defined in
Lemma A.3.
$$\Lambda_{D}=\mathbb{E}[\mathcal{H}(Z)\,\mathcal{H}(Z)^{T}]=\frac{1}{N}\sum_{Z\in D}[\mathcal{H}(Z)\,\mathcal{H}(Z)^{T}].$$
T]. (9)
$$(9)$$
Then if ΛD is full rank or equivalently its minimum eigenvalue is greater than zero, then it is
guaranteed that MHLA is *identifiable with respect to the data*.
Lemma 2.3 (Certificate of Identifiability—Informal). Let dataset D = {(Zi, yi)}i∈[N] be realizable
(see Definition A.2) by an H-head MHLA for any H ≥ 1. Let H *be the uniform family of polynomials*
Hn : R
d×n → R
ψ for ψ :=d2
d + d
2 defined as in Algorithm 2. For convenience we drop the
subscript of n and write H(Z) to mean Hn(Z) for Z ∈ R
d×n*. Finally, define* ΛD ∈ R
ψ×ψ *to be the*
second moment of the data features:
$$\Lambda_{D}:=\mathbb{E}_{D}\left[\mathcal{H}(Z)\mathcal{H}(Z)^{T}\right]\ .$$
$$(10)$$
T. (10)
Then if the eigenvalue λmin (ΛD) > 0, we say that MHLAΘ is certifiably identifiable with respect to
D*. That is, for every pair of empirical risk minimizers* Θ, Θ′ ∈ ΩH
MHLAΘ = *MHLA*Θ′ (11)
i.e., the two models have the same outputs on all inputs.
Corollary 2.4. *There is a polynomial* p : ΩH → R
ψ *such that for any pair of parameters* Θ, Θ′ ∈ ΩH
we have MHLAΘ = MHLAΘ′ *if and only if* p(Θ) = p(Θ′).
$$(11)$$
$$M H L A_{\Theta}=M H L A_{\Theta}$$
The polynomial p defines the equivalence class of parameters that compute the same function. For a formal statement of Lemma 2.3 see Lemma A.3. For handling of errors for approximate empirical risk minimization see Lemma A.7. Moreover, the certificate given by Algorithm 2 is not the only choice of feature mapping H that would certify identifiability; Lemma E.1 gives a general certificate for identifiability. One way to interpret Corollary 2.4 is that two MHLA models parameterized by Θ
and Θ′compute the same function if and only if they are the same linear function in a specific feature space (akin to matching coefficients in polynomial regression), which in turn is true if p(Θ) = p(Θ′)
for the polynomial p given in Corollary A.4. Comparing distance between the coefficients in the range of p is essentially the only meaningful metric of distance that is agnostic to the choice of dataset. Finally, we answer a few natural questions related to identifiability which we briefly summarize here. Firstly, perfectly noisy input data is identifiable under weak assumptions on the moments of the noise (see Lemma A.5). Secondly, the model class of MHLA with at least d 2 heads is certifiably identifiable from the second moment condition alone, and does not require realizability of the data
(see Lemma A.6). Finally, we empirically verify the min eigenvalue of ΛD predicts the generalization behavior of SGD for MHLA for the problem of learning key–value memories (see Figure 2).

## 3 Application To Learning Universal Turing Machines.

In Appendix B, we demonstrate that MHLAs can (autoregressively) express universal Turing machines with polynomially bounded computation histories. In this context, our identifiability results imply that, given a certifiably identifiable dataset of Turing machines and their computation histories on input words, empirical risk minimization and in particular Algorithm 1 will learn the universal Turing machine in a strong sense (Lemma C.5 for learning, Lemma A.8 with identifiability). That means at test time the learned MHLA will simulate any Turing Machine on any input word up to a given size for a bounded number of steps. For more detail see C
Lemma 3.1 (Learning UTM from Certifiably Identifiable Data). Let D = {(Zi, yi)}i∈[N] be a dataset satisfying yi = MHLAΘ for Θ ∈ ΩH being the expressibility parameters of Lemma B.1 for the set of TM's/words (M, x) ∈ ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ . If D is certifiably identifiable with λmin(ΛD) > η, then there is a poly(d, N, Q, ˆ Σˆ, n, ˆ Φˆ, η−1) *time algorithm that outputs a set of parameters* Θˆ ∈ Ωd2 such that for all TM's M and input words x in ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ *, we have*

$$C H_{\hat{\Theta}}(M,x)^{c(t)}[:-k_{t}]=x^{t}\;.$$
t. (12)
The c(t) step of the autoregressive computation history of Θˆ is equal to the t'th step of the computation history of M on x.

## 4 Experiments

In our experiments, we validate our theoretical predictions in settings where Transformers are trained using stochastic gradient descent (SGD), as follows: Firstly, Theorem 2.2 exploits that adding a sufficient number of heads to an MHLA leads to a convex optimization problem after reparameterization. This suggests that over-parameterizing by adding heads may provide optimization benefits. We investigate the role of over-parameterization in multi-head and multi-layer linear attention networks. For random data generated from linear attention networks, we observe that adding more heads achieves faster convergence of training loss than adding more layers. This suggests that while depth is important for expressiveness, the number of heads is important for optimization (Figure 3). Secondly, we empirically verify the certificate of identifiability provided by Lemma A.3 on datasets for associative memory [Bietti et al., 2023, Cabannes et al., 2024] with different choices of embeddings, demonstrating convergence to the equivalence class of the true parameters when λmin(ΛD) > 0 and converging to spurious solutions when λmin(ΛD) = 0 (Figure 2).

## 4.1 Do Extra Heads Help Optimization With Sgd?

To probe whether more heads facilitate learning in general, we train our convex relaxation and different types of over-parameterized models with SGD on data generated from a single-layer linear attention network. For the data, we initialize a single-layer linear attention network with parameters V ∈ R
1×dand Q ∈ R
d×d, sampled from a Gaussian distribution N (0, √
I
d
). Input sequences Z
i ∈ R
T ×dare sampled from N (0, √
I
T
), where i = 1*, . . . , N*, T = 100 is the maximum number of time steps, and N is the dataset size. We generate outputs by running the ground-truth network auto-regressively: y it = V Zi1:t(Z
i[:, : t]QZi[:, t]), creating our dataset D = {(Z
i, yi)}
N
i=1.

In addition to learning with Algorithm 1, we train three types of models on this data using SGD: (1) multi-head linear attention as in Equation (4); (2) multi-layer linear attention with a single head; (3) an ordinary Transformer network [Vaswani et al., 2017] with softmax attention, multi-layer perceptron blocks, and layer normalization.

Figure 1 illustrates the results. For same experiment with d = 4 and N = 2048 see Figure 3a in the appendix. Detailed hyperparameters and optimization procedures are described in Appendix G.1. We observe that multi-head attention scales effectively with an increasing number of heads, resulting in improved performance. Notably, for d = 2 or 4 input dimensions, using d 2 heads yields the best performance and is empirically comparable to Algorithm 1, approaching floating-point error precision. Theoretically, d 2is the maximum rank in the relaxation in Algorithm 1. In contrast, multi-layer attention models show diminishing returns and perform worse than single-layer attention. Interestingly, adding more layers can sometimes degrade performance. The full transformer model, which incorporates softmax attention, MLP layers and layer normalization, does not significantly outperform the single-layer linear attention model on this task. These findings suggest that the type of over-parameterization matters significantly in learning linear attention networks. Interestingly, multi-head architectures appear to be particularly effective—aligned with the structure of Algorithm 1, where the relaxation corresponds to adding more heads.

## 4.2 Does Certifiable Identifiability Predict Generalization?

In Lemma A.3, we developed a certificate that provides a sufficient condition for identifiability. To assess the practical relevance of this certificate, we conducted an empirical analysis of convergence in cases where the condition is not satisfied. The results of this analysis are presented in Figure 2.

Associative Memory Associative Memory [Bietti et al., 2023, Cabannes et al., 2024] is a task of looking up a value in a table with a query. Via a single head one-layer linear attention model it can be

0 100 200 300 400 500 epochs 10 16 10 13 10 10 10 7 10 4 10 1 M

S

E

(y, y

)
m=1, n=1, linear m=2, n=1, linear m=4, n=1, linear m=8, n=1, linear m=16, n=1, linear m=1, n=2, linear m=1, n=4, linear m=1, n=1, full m=2, n=1, full m=1, n=2, full Convex Algorithm
(a) N = 512, d = 2 0 100 200 300 400 500 epochs 10 16 10 13 10 10 10 7 10 4 10 1 M

S

E

(y, y

)

(b) N = 2048, d = 4
represented with ground truth parameters Θ = {*V, Q*} where *V, Q* ∈ R
2d×2d:

$$V=\begin{bmatrix}0&0\\ 0&I_{d\times d}\end{bmatrix}\quad Q=\begin{bmatrix}I_{d\times d}&0\\ 0&0\end{bmatrix}.$$

The data Z is drawn as follows: let k1, k2*, ..., k*d ∈ R
d be random variables corresponding to keys in a lookup table, let v1, v2*, ..., v*d ∈ R
d be random variables corresponding to values in a lookup table, let q ∈ R
d be a random variable corresponding to a query to the lookup table, and ζ ∼ N (0, I) be random noise, such that Z and the output vector y are defined as:

$$Z=\begin{bmatrix}k_{1}&k_{2}&\dots&k_{d}&q\\ v_{1}&v_{2}&\dots&v_{d}&\zeta\end{bmatrix}\tag{13}$$  $y=\text{MHLA}_{\Theta}(Z)=\begin{bmatrix}0\\ \sum_{j\in[d]}\langle q,k_{j}\rangle v_{j}\end{bmatrix}.$ (14)
Mixture of distributions: We generate two datasets, one that has identifiable λmin(ΛD) > 0 and one that is nonidentifiable with λmin(ΛD) = 0. The identifiable dataset is generated with {kj}j∈[d] and {vj}j∈[d] drawn i.i.d N (0, I). The query q is chosen to be one of the {kj}j∈[d] uniformly at random. The non-identifiable dataset is drawn such that {kj}j∈[d]forms a random unitary matrix,

2 13 2 12 2 11 2 10 2 9 2 8 2 7 Eigenvalue 0 1 2 3 4 d i s t a n c e

(

p

(

), 
p

(

)

)

Convex1-head2-head4-head8-head 10 6 10 4 10 2 10 0 d i s t a n c e

(

p

(

), 
p

(

)

)

Certificate Full Rank Low Rank
Figure 2: **Impact of data distribution on the associative lookup task performance:** We generated training data for an associative lookup task [Bietti et al., 2023, Cabannes et al., 2024] using mixtures of two distributions: (1) Gaussian key and value vectors, and (2) random unitary key and value vectors. By adjusting the mixture probability, we can manipulate the certificate value (minimum eigenvalue of the data covariance matrix), as unitary key–value vectors give rank-deficient "certificates". (a) Algorithm 1: as the minimum eigenvalue increases, Algorithm 1 converges more closely to the true parameters. (b) SGD: SGD learns parameters that are equivalent to the ground truth parameters in p feature space for certifiably identifiable data, but for unidentifiable data, they are far apart in p feature space and therefore compute different functions.

i.e., ∥kj∥ = 1 for all j ∈ [d] and ⟨kj , kj
′ ⟩ = 0 for all j ̸= j
′. Similarly, {vj}j∈[d]is also drawn from a randomly generated unitary matrix. We draw new random unitary matrices for each datapoint, where q is again chosen to be one of the {kj}j∈[d] uniformly at random. We set d = 4 dimensions for both datasets, and draw N = 214 samples for each dataset. We mix the two datasets together with a mixing probability ranging from 95% unidentifiable to 100% unidentifiable. In this manner we generate a spread of datasets with different values for λmin(ΛD) that tend to zero. Certifiable Identifiability for Algorithm 1: For each dataset, we run Algorithm 1 which returns Θˆ . We compare Θˆ to the ground truth Θ in p feature space via the distance

$$d(\Theta,\hat{\Theta}):=\|p(\Theta)-p(\hat{\Theta})\|_{F}.$$
$$(15)$$

d(Θ, Θ) ˆ := ∥p(Θ) − p(Θ) ˆ ∥F . (15)
Here, p is the polynomial given in Lemma A.3. Recall from Corollary A.4 that p defines the equivalence class of parameters that compute the same function, i.e., MHLAΘ = MHLAΘˆ if and only if p(Θ) = p(Θ) ˆ . On each dataset, we measure the certificate value λmin(ΛD) on the x-axis vs.

d(Θ, Θ) ˆ on the y-axis. In Figure 2a, we see that as the certificate value increases, d(Θ, Θ) ˆ decreases, indicating that MHLAΘ and MHLAΘˆ compute the same function.

Certifiable Identifiability for MHLA: Our notion of certifiable identifiability in Lemma A.3 applies to any empirical risk minimizer. Therefore, it applies to popular optimizers like SGD and Adam if they achieve the minimum of the loss, which is in our synthetic case equal to zero. In Figure 2b, we train MHLA models via SGD with 1, 2, 4, and 8 heads. For identifiable data with minimum eigenvalue 0.06, we see that the learned parameters and ground truth parameters are the same in p feature space. However, for unidentifiable data with minimum eigenvalue 0, learned parameters and ground truth parameters are far apart in p feature space and therefore compute different functions.

## 5 Related Work 5.1 Formal Expressivity Of Transformers

A large body of work has been trying to tackle the problem of quantifying what algorithmic tasks can a Transformer do, in terms of various kinds of circuit families [Pérez et al., 2019, Edelman et al., 2022b, Hahn, 2020, Merrill and Sabharwal, 2023, Merrill et al., 2022, 2021, Liu et al., 2022, Feng et al., 2023].

In particular, researchers have studied how Transformers can realize specific DSLs [Weiss et al., 2021], logic expressions [Dong et al., 2019, Barceló et al., 2020, 2024], Turing machines [Dehghani et al., 2018, Giannou et al., 2023, Pérez et al., 2021], formal language recognition [Hao et al., 2022, Chiang et al., 2023], as well as automata and universal Turing machines [Liu et al., 2022, Li et al., 2024]. However, while these works primarily focus on determining the types of problems whose solutions a Transformer can express, they often overlook the crucial question of how these solutions can be learned from data. Moreover, there is limited discussion on the sufficiency of the dataset itself—whether the data available can identify the underlying "true" function or algorithm that we aim to capture.

## 5.2 Learning Transformers

We break down the literature on learning transformers. First, there is the literature on statistical learnability, where the focus is on the amount of data required to learn without considering whether there is a tractable algorithm for learning [Edelman et al., 2022a, Wei et al., 2021, Zhang et al., 2024, Trauger and Tewari, 2023].

Second, there are learnability results for single head transformers for data distributions under a variety of assumptions. In particular, Zhang et al. [2023] provide learnability results for in-context linear regression; Jelassi et al. [2022] show that data with spatial structure can be learned; the work of Tian et al. [2023] analyzes SGD training dynamics for a toy model for data; and Oymak et al. [2023] study the prompt attention model.

Third, the literature on provable guarantees for learning multi head attention is rather sparse. Fu et al.

[2023] give learnability results in a regime where attention matrices are fixed and only the projection matrices are trained. Tarzanagh et al. [2024] show connections between single layer attention optimization and SVM learning. Under a good gradient initialization condition, overparameterization condition, and a condition on the scores of optimal tokens the global convergence of gradient descent to a particular SVM problem can be established. Deora et al. [2023] analyze a setting of learning multi head attention with gradient descent under their Assumption 2. In the words of the authors "these conditions are related to the realizability condition, which guarantees obtaining small training error near initialization", which they instantiate with the separability of the data in an NTK space and a proximity of initialization to realizable parameters. Interestingly, they find that multi head attention has benign optimization properties. Finally, Chen and Li [2024] study learning for multi head attention for well structured data that is drawn independent Bernoulli or Gaussian. They provide an extensive discussion of lower bounds for learning multi head attention.

## 6 Conclusion And Limitations

In this work we tackle the fundamental problem of finding an efficient algorithm that provably learns the weights of a linear Transformer. Our key theoretical ingredient is to consider a model class that's sufficiently "wide" (scaling number of heads), and to find that the loss is convex under this scaling, with generalization guaranteed by the classical VC theory. This reinforces the empirical observation that scaling model size enables efficient optimization and can still result in successful generalization. Our theory extends trivially when arbitrary feature maps ϕ(·) are applied to keys and queries providing a natural avenue for extending our theory to models that can approximate softmax transformers with custom key-query kernels. Of course the model class we consider is far simpler than modern LLM's, but we consider our work an important step towards designing algorithms with provable guarantees for training neural sequence models.

## Acknowledgments

We gratefully acknowledge support from NSF grants IIS-2214177, IIS-2238240, CCF-2112665 and DMS-2134108; from AFOSR grant FA9550-22-1-0249; from ONR MURI grant N00014-22-1-2740; and from ARO grant W911NF-23-1-0034; from the OpenPhilanthropy Foundation; from MIT Quest for Intelligence; from the MIT-IBM Watson AI Lab; from ONR Science of AI; from Simons Center for the Social Brain; and from an Alexander von Humboldt professorship. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of our sponsors.

## References

Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Ali Jadbabaie, and Suvrit Sra. Linear attention is (maybe) all you need (to understand transformer optimization), 2024. URL https:
//arxiv.org/abs/2310.01082.

Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. In-context language learning: Architectures and algorithms, 2024. URL https://arxiv.org/abs/2401.12973.

Pablo Barceló, Egor V Kostylev, Mikael Monet, Jorge Pérez, Juan Reutter, and Juan-Pablo Silva. The logical expressiveness of graph neural networks. In ICLR, 2020.

Pablo Barceló, Alexander Kozachinskiy, Anthony Widjaja Lin, and Vladimir Podolskii. Logical languages accepted by transformer encoders with hard attention. 2024.

Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long short-term memory. Vancouver, Canada, December 2024.

Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint, 2023. URL https://arxiv.org/abs/2306.00802.

Vivien Cabannes, Berfin Simsek, and Alberto Bietti. Learning associative memories with gradient descent, 2024. URL https://arxiv.org/abs/2402.18724.

Sitan Chen and Yuanzhi Li. Provably learning a multi-head attention layer, 2024. URL https:
//arxiv.org/abs/2402.04084.

David Chiang, Peter Cholak, and Anand Pillay. Tighter bounds on the expressivity of transformer encoders. *arXiv preprint arXiv:2301.10743*, 2023.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024. URL https://arxiv.org/abs/2405.21060.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. *arXiv preprint arXiv:1807.03819*, 2018.

Puneesh Deora, Rouzbeh Ghaderi, Hossein Taheri, and Christos Thrampoulidis. On the optimization and generalization of multi-head attention, 2023. URL https://arxiv.org/abs/2310.12680.

Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou. Neural logic machines. In *ICLR*, 2019.

Benjamin L. Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms, 2022a. URL https://arxiv.org/abs/2110.10090.

Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms. In *International Conference on Machine Learning*, pages 5793–5831. PMLR, 2022b.

Guhao Feng, Yuntian Gu, Bohang Zhang, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. *arXiv preprint arXiv:2305.15408*, 2023.

Hengyu Fu, Tianyu Guo, Yu Bai, and Song Mei. What can a single attention layer learn? a study through the random features lens, 2023. URL https://arxiv.org/abs/2307.11353.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. *arXiv preprint arXiv:2301.13196*, 2023.

Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Transactions of the Association for Computational Linguistics, 8:156–171, 2020.

Yiding Hao, Dana Angluin, and Robert Frank. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. *Transactions of the Association for Computational* Linguistics, 10:800–810, 2022.

Samy Jelassi, Michael E. Sander, and Yuanzhi Li. Vision transformers provably learn spatial structure, 2022. URL https://arxiv.org/abs/2210.09221.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention, 2020. URL https://arxiv.org/ abs/2006.16236.

Michael J. Kearns and Umesh Vazirani. *An Introduction to Computational Learning Theory*. The MIT Press, 08 1994. ISBN 9780262276863. doi: 10.7551/mitpress/3897.001.0001. URL https://doi.org/10.7551/mitpress/3897.001.0001.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. *arXiv preprint arXiv:2402.12875*, 2024.

Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. *arXiv preprint arXiv:2210.10749*, 2022.

Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam, 2018. URL https:
//openreview.net/forum?id=rk6qdGgCZ.

William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. *Transactions of the Association for Computational Linguistics*, 11:531–545, 2023.

William Merrill, Yoav Goldberg, and Noah A Smith. On the power of saturated transformers: A view from circuit complexity. *arXiv preprint arXiv:2106.16213*, 2021.

William Merrill, Ashish Sabharwal, and Noah A Smith. Saturated transformers are constant-depth threshold circuits. *Transactions of the Association for Computational Linguistics*, 10:843–856, 2022.

Samet Oymak, Ankit Singh Rawat, Mahdi Soltanolkotabi, and Christos Thrampoulidis. On the role of attention in prompt-tuning, 2023. URL https://arxiv.org/abs/2306.03435.

Jorge Pérez, Javier Marinkovic, and Pablo Barceló. On the turing completeness of modern neural ´
network architectures. In *ICLR*, 2019.

Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing complete. *The Journal of* Machine Learning Research, 22(1):3463–3497, 2021.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear Transformers are secretly fast weight programmers. Virtual only, July 2021.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? a survey. *Transactions of the Association for Computational Linguistics*,
12:543–561, 2024.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. Preprint arXiv:2307.08621, 2023.

Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, and Samet Oymak. Transformers as support vector machines, 2024. URL https://arxiv.org/abs/2308.16898.

Yuandong Tian, Yiping Wang, Beidi Chen, and Simon Du. Scan and snap: Understanding training dynamics and token composition in 1-layer transformer, 2023. URL https://arxiv.org/abs/ 2305.16380.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

Jacob Trauger and Ambuj Tewari. Sequence length independent norm-based generalization bounds for transformers, 2023. URL https://arxiv.org/abs/2310.13088.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers. *CoRR*, abs/2107.13163, 2021. URL https:
//arxiv.org/abs/2107.13163.

Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In *International Conference* on Machine Learning, pages 11080–11090. PMLR, 2021.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. Vienna, Austria, July 2024.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length, 2025. URL https://arxiv.org/abs/2406.06484.

Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained transformers learn linear models in-context, 2023. URL https://arxiv.org/abs/2306.09927.

Yufeng Zhang, Boyi Liu, Qi Cai, Lingxiao Wang, and Zhaoran Wang. An analysis of attention via the lens of exchangeability and latent variable models, 2024. URL https://arxiv.org/abs/ 2212.14852.

## A Certificate For Identifiability Of Linear Attention

We begin by defining identifiability of a model class with respect to a dataset.

Definition A.1 (Identifiability). Let D = {(Zi, yi)}i∈[N]. Let UΘ denote a model class which is a uniform circuit family parameterized by parameters Θ ∈ Ω. Let L be a loss function and ΩERM be the set of empirical risk minimizers:

$$\Omega_{\Theta}=\{\hat{\Theta}\in\Omega\mid\hat{\Theta}=\arg\operatorname*{min}_{\Theta\in\Omega}{\mathcal{L}}({\mathcal{U}}_{\Theta},D)\}.$$

We say model class UΘ is *identifiable with respect to the dataset* D if for all Z ∈ R
d×n
′, and for all pairs of empirical risk minimizers Θ, Θ′ ∈ ΩERM we have UΘ and UΘ′ compute the same function, i.e., they agree on all inputs (are the same uniform circuit family):

$$(16)$$
$${\mathcal{U}}_{\Theta}(Z)={\mathcal{U}}_{\Theta^{\prime}}(Z).$$
$$(17)$$
UΘ(Z) = UΘ′ (Z). (17)
In establishing conditions for identifiability, it will be useful to refer to another condition relating models to datasets.

Definition A.2 (Realizability). Let Θ ∈ ΩH be an MHLA parameterization. We say a dataset D = {(Zi, yi)}i∈[N]is *realizable by a parameterization* Θ if yi = MHLAΘ(Zi).

The definition of realizability can be modified to include independent noise at the expense of adding some terms to our analyses. See Lemma A.7 for details. Next, we prove that for the model class MHLA there is an efficiently checkable condition (certificate) of the data D that guarantees the model class is identifiable with respect to D. Our results follow by reinterpreting the results of Theorem 2.2 with a focus on data conditions that uniquely determine the optimal regressor. In this section we denote the mapping from data to feature space to be H
and the mapping from parameters to feature space to be p which are analogous to the X and TΘ of Equation (8). We instantiate the feature mapping H and parameter mapping polynomial p as follows. Lemma A.3 (Certificate of Identifiability). Let dataset D = {(Zi, yi)}i∈[N] *be a realizable dataset.*
Let H = {Hn}∞
n=1 *be a family of polynomials* Hn : R
d×n → R
ψ for ψ =d2 d + d 2 defined as follows. We index the entries of H by taking the Kronecker product between all sets of pairs {*j, k*}
(for all j, k ∈ [d]) with with all ℓ ∈ [d]. We define H(Z){j,k}ℓ *as in Algorithm 2 to be*

$${\mathcal{H}}(Z)_{\{j,k\}\ell}:=\langle z_{j},z_{k}\rangle z_{\ell n_{i}}.$$
$$(18)^{\frac{1}{2}}$$

$[\mathfrak{A}(Z)]$? 
Then if λmin ED-H(Z)H(Z)
T > 0, we have that MHLAΘ *is identifiable with respect to* D.

Next we construct a mapping p : Ω → R
d×ψ that partitions the parameter space into equivalence classes of parameters that compute the same function. This is akin to matching coefficients in polynomial regression. This mapping defines a meaningful notion of "distance" between different attention parameters by constructing a feature space in which equivalent models have the same representation. We denote the a'th row of p to be pa : Ω → R
ψ and define it as follows.

Corollary A.4. Let {pa}a∈[d] *be a collection of polynomials such that* pa(Θ) : ΩH → R
ψ *is defined* as follows. Each pa(Θ) is indexed by pairs {j, k} for j, k ∈ [d] and ℓ ∈ [d] *defined to be*

$$p_{a}(\Theta)_{\{j,k\}\ell}=\sum_{h\in[H]}\,(V_{h,a j}Q_{k\ell}+V_{h,a k}Q_{j\ell})\ .$$
(Vh,ajQkℓ + V*h,ak*Qjℓ) . (19)
Let the polynomial p : Ω → R
d×ψ be p := (p1, p2, ..., pd)*. Then for any pair of parameters* Θ, Θ′ ∈ ΩH we have MHLAΘ = MHLAΘ′ *if and only if* p(Θ) = p(Θ′).

We give an overview of a few results building on our certifiable identifiability machinery:
First, data drawn from independent noise is certifiably identifiable. If the data matrices {Zi}i∈[N] are drawn with each entry being standard normal noise, then MHLAΘ for Θ ∈ ΩH is identifiable with respect to the data. The statement holds beyond standard normals to distributions satisfying weak moment conditions. The result is stated with population risk instead of empirical risk to simplify the statement.

$$(19)$$

Algorithm 2 Constructing Features for Certificates of Identifiability 1: **Input:** Data D := {Zi}i∈[N] for Zi ∈ R
d×ni 2: **Output:** feature vectors H(Zi) for i ∈ [N] 3: for Zi ∈ D do 4: Let z1:, z2:*, ...z*d: be the rows of Zi and let zab be the (*a, b*) entry of Zi 5: for sets {*j, k*} in Distinct Pairs of Indices in [d]
2do 6: for ℓ ∈ [d] do 7: H(Zi) = H(Zi) ◦ [⟨zj:, zk:⟩zℓni]
8: **end for** 9: **end for**
10: for j ∈ [d] do 11: for ℓ ∈ [d] do 12: H(Zi) = H(Zi) ◦-∥zj∥
2zℓni 13: **end for** 14: **end for** 15: **end for**
16: **Return:** {H(Zi)}i∈[N]
Lemma A.5 (Independent input noise yields identifiability). Let (Z, y) ∼ D be a realizable dataset. Let Z be drawn from a distribution Z where the (a, b)-th entry of Z denoted by Zab *is drawn i.i.d.*
from a distribution ν over R for all a ∈ [d] and b ∈ [n]*. Let the second and fourth moment of* ν be denoted m2 and m4 respectively. Let m2 > 0 and m4 > m22. Then MHLAΘ for Θ ∈ ΩH is identifiable with respect to D. That is to say, for any population risk minimizers Θ, Θ′ ∈ ΩPRM:
MHLAΘ = *MHLA*Θ′ . (20)
Second, when specialized to the case of Multi Head Linear Attention MHLAΘ with more than d 2 heads we can avoid the realizability assumption entirely. This is because the class of MHLA with an arbitrary number of heads is linear in the feature space H given in Lemma A.3. Lemma A.6 (Identifiability without realizability for MHLA with arbitrarily many heads). *Let dataset* D = {(Zi, yi)}i∈[N] be any dataset drawn i.i.d from a distribution D. Let H *be defined as in* Lemma A.3. Then if λmin ED[H(Z)H(Z)
T]> 0 then MHLAΘ for Θ ∈ ΩH for any H ∈ [d 2, ∞)
is identifiable with respect to the data D. That is, MHLAΘ = *MHLA*Θ′ (21)
for all pairs of empirical risk minimizers Θ, Θ′ ∈ ΩERM.

We also add a quantitative version of identifiability with precise treatment of issues related to error. (For a corresponding statement of realizability with noise see Lemma E.2.)
Lemma A.7 (Identifiability with Error). Let Ωϵ−ERM be the set of ϵ*-approximate empirical risk* minimizers,

Ωϵ−ERM =
$$\left\{\Theta\in\Omega_{H}\mid\mathbb{E}_{(Z,y_{i})\in D}\left[\left(\mathit{MHLA}_{\Theta}(Z_{i})-y_{i}\right)^{2}\right]\leq\epsilon\right\}.$$  _Then we have for any $\Theta,\Theta^{\prime}\in\Omega_{\epsilon-\mathit{text}}$ that for all inputs $Z\in\mathbb{R}^{d\times n}$_  $$\|\mathit{MHLA}_{\Theta}(Z)-\mathit{MHLA}_{\Theta^{\prime}}(Z)\|\leq\frac{\epsilon}{\lambda_{\min}\left(\Lambda_{D}\right)}\|Z\|_{F}^{\Theta}.\tag{22}$$
$$(21)^{\frac{1}{2}}$$
We prove all the above statements in Appendix E. Application to learning Universal Turing Machines. In Appendix B, we demonstrate that MHLAs can (autoregressively) express universal Turing machines with polynomially bounded computation histories. In this context, our identifiability results imply that, given a certifiably identifiable dataset of Turing machines and their computation histories on input words, empirical risk minimization and in particular Algorithm 1 will learn the universal Turing machine in a strong sense (Lemma C.5 for learning, Lemma A.8 with identifiability). That means at test time the learned MHLA will simulate any Turing Machine on any input word up to a given size for a bounded number of steps. For more detail see C

$$(222)$$

Lemma A.8 (Learning UTM from Certifiably Identifiable Data). Let D = {(Zi, yi)}i∈[N] be a dataset satisfying yi = MHLAΘ for Θ ∈ ΩH *being the expressibility parameters of Lemma B.1 for* the set of TM's/words (M, x) ∈ ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ . If D is certifiably identifiable with λmin(ΛD) > η, then there is a poly(d, N, Q, ˆ Σˆ, n, ˆ Φˆ, η−1) *time algorithm that outputs a set of parameters* Θˆ ∈ Ωd2 such that for all TM's M and input words x in ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ *, we have*

$$C H_{\hat{\Theta}}(M,x)^{c(t)}[:-k_{t}]=x^{t}\;.$$
$\eqref{eq:walpha}$. 
t. (23)
The c(t) step of the autoregressive computation history of Θˆ is equal to the t*'th step of the computation* history of M on x.

## B Realizability Of Universal Automata In Mhla

We also include an application of our theory on learnability and identifiability to the problem of learning a universal Turing machine (UTMs) with polynomially bounded computation length. We prove such a UTM is expressible via MHLA in Lemma B.1, and show that for certifiably identifiable data the learned MHLA generalizes to any TM M and input word x in Lemma A.8.

Lemma B.1 (UTM Expressibility). Let ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ *be the set of Turing machines* M =
{δ, Σ, Q, qstart, qaccept, qreject} *and words* x ∈ Σ
∗ *with number of states, size of alphabet, size* of input, and number of steps in computation history bounded by Qˆ, Σˆ, n, ˆ Φˆ respectively. For any (M, x) ∈ ∆*, let* {xt}t∈[Φ] be the computation history of the UTM on (M, x). Let the autoregressive computation history (see Definition C.2) of MHLAΘ on input (M, x) be denoted CHΘ(*M, x*) = {Z
1, Z2, ..., ZΦ}. Then there exists a set of parameters Θ ∈ ΩH for H = O(ˆnΦˆΣ) ˆ
and embedding dimension d = O(ˆnΦˆΣ max( ˆ Σˆ, Qˆ)), such that for all (M, x) ∈ ∆, the TM computation history at time step t *is equivalent to the autoregressive computation history at time step* c(t) where c(t) ≤ O((n + t)t) i.e Z
c(t)[: −*length*(x t))] = x t. Furthermore, this can be achieved with 2 bits of precision.

Our construction bears similarities to [Pérez et al., 2019, Hahn, 2020, Merrill and Sabharwal, 2023, Merrill et al., 2022, 2021, Liu et al., 2022, Feng et al., 2023]; the high-level idea is write down every letter in the computation history of M on x. If we use orthogonal vectors to encode every letter, state, and positional embedding we arrive at a natural construction involving a few basic primitives copy, lookup, and if-then-else. For details see discussion section F and Proof F.1

## C Application To Learning Universal Turing Machines

We apply our algorithmic and identifiability machinery to show that an important computational procedure is representable and learnable as an MHLA: namely, a restricted class of universal Turing machines (UTMs) with bounded computation history. We must first generalize our previous MHLA definition to enable multi-step computation: Definition C.1 (Autoregressive MHLA). Let Z
0 be an input matrix in dimension R
d×n. We define
the iterative process of Φ*-step autoregressive MHLA* as follows: starting from t = 0, let the next
and, for all $t\in[\Phi]$, let $Z^{t+1}\in\mathbb{R}^{d\times(n+1)}$ be the concatenation:  $$Z^{t+1}=Z^{t}\circ y^{t}\.$$
t) , (24)
$$(24)$$
$$Z^{t+1}=Z^{t}\circ y^{t}\;.$$
$$(25)$$
t. (25)
Next we define the computation history of an autoregressive model analogously to the computation history of a Turing machine.

Definition C.2 (Autoregressive Computation History). We refer to CHΘ(Z) = {Z
t}t∈[Φ] as the computation history of the Φ-step autoregressive MHLA. We denote the t-th step of the computation history as CHtΘ(Z) = Z
t.

We will often use the notation Zt[: −k] to denote the last k ∈ Z
+ tokens of Zt. Often, Z will be the embeddings corresponding to a word x in a language L, in which case we will use the notation CHΘ(x) and CHΘ(Z) interchangeably. For pedagogical discussion on how to map embeddings to letters in an alphabet, see Section G Although the theory derived in this paper applies to all functions expressible by MHLAs, we are particularly interested in the task of learning *universal Turing machines* (UTMs). Let Σ be an alphabet. Let Q be a set of states that includes {qstart, qaccept, q*reject*} a start, accept, and reject state respectively. Let δ : Q × Σ → Q × Σ × {L/R} be a transition function that takes an alphabet and state symbol and maps to a state transition, an output symbol, and a head movement left or right. Typically there is also a tape alphabet Γ for which the input alphabet Σ is a subset.

Definition C.3 (Accept TM). Let M = {δ, Σ, Γ, Q, qstart, qaccept, q*reject*} be a TM. Let x ∈ Σ
∗ be all strings in the alphabet Σ. Then let ATM be the language ATM = {(*M, x*) | M accepts x}. The UTM constructed in Turing's 1936 paper recognizes ATM. In practice, we are most often interested in the behavior of TMs that run in polynomial time, and focus below on implementing a universal simulator for this restricted class: Definition C.4. (Polynomially Bounded Universal Turing Machine) In general, a UTM is a recognizer for the language ATM. That is if x is in ATM, the UTM accepts, else, the UTM rejects or does not halt. Let ATM ∩ P be the language of input pairs (*M, x*) for TM M and word x ∈ Σ
∗such that M decides x in polynomial time. Here, we consider UTM to be the polynomial time decider for ATM ∩ P.

To define what it means for an autoregressive MHLA to perform the same computation as a TM, our main idea is to construct parameters for MHLA such that it executes the computation history of TM M on input x. Let the UTM computation history at step t include the contents x0*, . . . , x*kt on the tape after t transition steps of the Turing machine M, the current state qt, and the current head position ht. Here kt is the number of tokens at timestep t. Then, there is a single-layer MHLA
capable of simulating a UTM:
Lemma B.1 (UTM Expressibility). Let ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ *be the set of Turing machines* M =
{δ, Σ, Q, qstart, qaccept, qreject} *and words* x ∈ Σ
∗ with number of states, size of alphabet, size of input, and number of steps in computation history bounded by Qˆ, Σˆ, n, ˆ Φˆ *respectively. For* any (M, x) ∈ ∆*, let* {xt}t∈[Φ] *be the computation history of the UTM on* (M, x). Let the autoregressive computation history (see Definition C.2) of MHLAΘ *on input* (M, x) *be denoted* CHΘ(*M, x*) = {Z
1, Z2, ..., ZΦ}. Then there exists a set of parameters Θ ∈ ΩH for H = O(ˆnΦˆΣ) ˆ
and embedding dimension d = O(ˆnΦˆΣ max( ˆ Σˆ, Qˆ))*, such that for all* (M, x) ∈ ∆, the TM computation history at time step t *is equivalent to the autoregressive computation history at time step* c(t)
where c(t) ≤ O((n + t)t) i.e Z
c(t)[: −*length*(x t))] = x t*. Furthermore, this can be achieved with 2* bits of precision.

We include the full proof for the existence of Θ in the appendix. For simplicity, we adopt a naive embedding scheme that represents different letters in an alphabet as orthogonal unit vectors. This makes it easy to contrive embedding schemes that incorporate arbitrary polynomial-sized circuits which could compute whether x ∈ L(M). Moreover, we adopt positional encodings that are simply orthogonal unit vectors. Thus, in order to give each of T tokens a unique ID, we would require O(T) dimensional positional embeddings. This can be combined with the learnability results above to yield a specialized result for UTMs:
Lemma C.5 (Learning a UTM). Let Θ ∈ ΩH in dimension d be the MHLA parameters in Lemma B.1. Let {Mi, xi}i∈[N] be pairs of TM's M and words x of maximum length n drawn i.i.d. from a distribution D. Let Zi = Embed(Mi, xi). For each TM/word pair (Mi, xi) let CHΘ(Zi) = {Z
1 i
, Z2 i
, ..., ZΦ
i
} be the Φ-step autoregressive computation history of *MHLA*Θ on Zi. Let D *be the dataset* D := {(CHΘ(Zi)
t, yt+1 i}i∈[N],t∈[T] *where* y t+1 i = *MHLA*Θ(Z
t i). Then Algorithm 1 applied to input D returns Θˆ ∈ ΩH for H ≤ d 2*such that with probability* 1 − δ

$$\mathbb{E}_{(Z,y)\in\mathcal{D}}\left[\left(M H L A_{\Theta}(Z)-y\right)^{2}\right]\leq\epsilon$$

for sample complexity N = poly(*d, ϵ*−1, log(δ
−1)). Then with probability 1 − δ over the randomness in the data, the probability over D that the Φ*-step autoregressive computation history* CHΘˆ (M, x)
and CHΘ(M, x) *differ is upper bounded by* Pr(M,x)∼D[CHΘˆ (M, x) ̸= CHΘ(*M, x*)] ≤ O(ϵΦ). (27)

$$(26)^{\frac{1}{2}}$$

Finally, if the dataset D is certifiably identifiable, then generalization holds out-of-distribution. For proof see Appendix F.2.

Lemma A.8 (Learning UTM from Certifiably Identifiable Data). Let D = {(Zi, yi)}i∈[N] be a dataset satisfying yi = MHLAΘ for Θ ∈ ΩH *being the expressibility parameters of Lemma B.1 for* the set of TM's/words (M, x) ∈ ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ . If D is certifiably identifiable with λmin(ΛD) > η, then there is a poly(d, N, Q, ˆ Σˆ, n, ˆ Φˆ, η−1) *time algorithm that outputs a set of parameters* Θˆ ∈ Ωd2 such that for all TM's M and input words x in ∆(Qˆ, Σˆ, n, ˆ Φ) ˆ *, we have*

$$C H_{\hat{\Theta}}(M,x)^{c(t)}[:-k_{t}]=x^{t}\;.$$
$$(23)$$
t. (23)
The c(t) step of the autoregressive computation history of Θˆ is equal to the t*'th step of the computation* history of M on x.

## D Proof Of The Main Theorem

Theorem 2.2 (Learnability of Linear Attention). Let D be a dataset D = {Zi, yi}i∈[N] *drawn i.i.d.*
from a distribution D *where each* Zi ∈ R
d×ni, yi ∈ R
d. The embedding dimension d is fixed across the dataset, whereas ni can be different for each datapoint. Let nmax = maxi∈[N] ni be the maximum sequence length, and let ΩH be the space of H pairs of value and key-query matrices {(Vh, Qh)}h∈[H]
for any H ∈ [1, ∞). Then there is an algorithm (Algorithm 1) that runs in time O(N d4nmaxϵ
−1)
and that, given input–output pairs {(Zi, yi)}i∈[N]*, returns* Θ =ˆ {(Vˆh, Qˆh)}h∈[Hˆ ] ∈ ΩHˆ for Hˆ ≤ d 2 such that with probability 1 − δ,

$$\mathbb{E}_{(Z,y)\in{\mathcal{D}}}\left[\|M H L A_{\hat{\Theta}}(Z)-y\|^{2}\right]$$
$$-\min_{\Theta\in\Omega_{H}}\mathbb{E}_{(Z,y)\in\mathcal{D}}\left[\|MHLA_{\Theta}(Z)-y\|^{2}\right]\leq\epsilon\tag{6}$$  ($\delta^{-1}$))
with sample complexity N = O1ϵ d 4 + log(δ

$$\Xi(\delta^{-1})).$$

Proof. First we write down the loss:

$$\mathcal{L}_{\Theta}(\{(Z_{i},y_{i})\}_{i\in[N]}):=\frac{1}{N}\sum_{i\in[N]}\left\|\sum_{h\in[H]}V_{h}Z_{i}(Z_{i}^{T}Q_{h}Z[:,n_{i}])-y_{i}\right\|_{F}^{2}$$ $$=\frac{1}{N}\sum_{i\in[N]}\sum_{a\in[d]}\left(\sum_{h\in[H]}e_{a}^{T}V_{h}Z_{i}(Z_{i}^{T}Q_{h}Z[:,n_{i}])-y_{i,a}\right)^{2}$$
(28)  $$\begin{array}{l}\mathbf{(29)}\end{array}$$ . 
$$(30)$$

Observe that the one layer attention network is a quadratic polynomial in {Vh, Qh}h∈[H].

$$=\frac{1}{N}\sum_{i\in[N]}\sum_{a\in[d]}(\langle T_{\Theta},X_{i,a}\rangle-y_{i,a})^{2}$$
2(30)
Here

$$\mathcal{T}_{\Theta}:=\sum_{h\in[H]}\text{flatten}(V_{h})\text{flatten}(Q_{h})^{T}=\sum_{h\in[H]}\begin{bmatrix}V_{h,00}Q_{h,00}&V_{h,00}Q_{h,01}&\dots&V_{h,00}Q_{h,dd}\\ V_{h,01}Q_{h,00}&V_{h,01}Q_{h,01}&\dots&V_{h,01}Q_{h,dd}\\ \vdots&\vdots&\vdots&\vdots\\ V_{h,dd}Q_{h,00}&V_{h,dd}Q_{h,01}&\dots&V_{h,dd}Q_{h,dd}\end{bmatrix}\tag{31}$$

Now we relax the objective where we replace TΘ with an unconstrained matrix W ∈ R
d 2×d 2. Another way to put it is that TΘ is rank-H but W can be a general matrix. Because the space of general rank matrices is larger, we have written down a relaxation guaranteed to have a smaller loss. Furthermore the loss can be optimized via ordinary least squares.

$$\min_{W\in\mathbb{R}^{d^{2}\times d^{2}}}\mathcal{L}_{W}(\{(Z_{i},y_{i})\}_{i\in[N]}):=\frac{1}{N}\sum_{i\in[N]}\sum_{a\in[d]}(\langle W,X_{i,a}\rangle-y_{i,a})^{2}\\ \leq\min_{\Theta\in\mathbb{U}_{H}}\mathcal{L}_{\Theta}(\{(Z_{i},y_{i})\}_{i\in[N]})+\epsilon\tag{32}$$

Thus the optimum of the regression with respect to the data achieves optimum of the loss to error ϵ in time O(
1 ϵ d 4N). The sample complexity to achieve error ϵ is then O(
1 ϵ
(d 4 + log(δ
−1))) with probability 1 − δ over the data distribution. Furthermore, if we take the SVD of W =Pi∈[Hˆ ] AiBT
i where we absorb the singular values into the left and right singular vectors we have for Θ = ˆ
{Fold(Ah), Fold(Bh)}i∈[Hˆ ]
. Let Vˆh = Fold(Ah) and Qˆh = Fold(Bh)

$$\mathcal{L}_{\hat{\Theta}}(\{(Z_{i},y_{i})\}_{i\in[N]}):=\frac{1}{N}\sum_{i\in[N]}\left\|\sum_{h\in[\hat{H}]}\hat{V}_{h}Z_{i}(Z_{i}^{T}\hat{Q}_{h}Z_{i}[:,n_{i}])-y_{i}\right\|_{F}^{2}\tag{33}$$ $$=\frac{1}{N}\sum_{i\in[N]}\sum_{a\in[\hat{a}]}\left(\sum_{h\in[\hat{H}]}\hat{V}_{h}Z_{i}(Z_{i}^{T}\hat{Q}_{h}Z_{i}[:,n_{i}])-y_{i,a}\right)^{2}\leq\epsilon$$
$$\square$$

as desired.

## E Proofs From Identifiability Section

First, we start with a general lemma (Lemma E.1) which states a sufficient condition for identifiability of any model class that can be written as an inner product of a polynomial of parameters Θ with a polynomial feature mapping H. If the data is realizable by the model class and ΛD = ED
-H(Z)H(Z)
T
is full rank then the model class is identifiable with respect to D. The following is the certificate of identifiability written in an abstract form involving polynomials to map parameters to feature space and polynomials to map data to feature space. The proof does not require the model to be an MHLA, but we state it in MHLA terms for the sake of concreteness.

Lemma E.1 (General Certificate of Identifiability). Let dataset D = {(Zi, yi)}i∈[N] be a dataset realizable by Θ ∈ ΩH. Let p := {pa}a∈[d] *be a collection of polynomials* pa : Ω → R
ψ *mapping the* parameters Θ ∈ Ω *to a feature space of fixed dimension* ψ ∈ Z
+*. Let* H = {Hn}∞
n=1 be a uniform family of polynomials such that Hn : R
d×n → R
ψ. Let p and H satisfy

$$M H L A_{\Theta}(Z)[a]=\langle p_{a}(\Theta),{\mathcal{H}}_{n}(Z)\rangle$$
MHLAΘ(Z)[a] = ⟨pa(Θ), Hn(Z)⟩ (34)
$$\begin{array}{c}\mbox{\it mILL}_{\Theta}(Z)[a]-\langle p_{a}(G),R_{n}(Z)\rangle\end{array}$$  _for all $Z\in\mathbb{R}^{d\times n}$ for all $n\in[1,\infty)$. Then if $\lambda_{\min}\left(\mathbb{E}_{D}\left[\mathcal{H}(Z)\mathcal{H}(Z)^{T}\right]\right)>0$, we have_  $$\mbox{\it MILL}_{\Theta}=\mbox{\it MILL}_{\Theta^{\prime}}$$
$$(34)$$
$$(35)$$

for all empirical risk minimizers Θ, Θ′ ∈ ΩERM*. That is, all empirical risk minimizers compute the* same function.

Proof. We construct a map p : Ω → R
ψ such that MHLAΘ = MHLAΘ′ if and only if p(Θ) = p(Θ′).

Then we show that any empirical risk minimizer ΘERM and the ground truth Θ¯ satisfy p(ΘERM) =
p(Θ) ¯ .

In more detail, we construct some polynomials {pa}a∈[d] and family of polynomials H such that

$$\mathrm{MHLA}_{\Theta}(Z)|_{a}=\langle p_{a}(\Theta),{\mathcal{H}}(Z)\rangle$$
MHLAΘ(Z)|a = ⟨pa(Θ), H(Z)⟩ (36)
We construct a linear model class R that takes as parameters v ∈ R
ψ and data H(Z) ∈ R
ψ. such that

$${\mathcal{R}}_{v}({\mathcal{H}}(Z))=\langle v,{\mathcal{H}}(Z)\rangle$$
Rv(H(Z)) = ⟨v, H(Z)⟩ (37)
$${\mathcal{H}}(Z))$$

Let ΘERM be defined as

$$\Theta_{\mathrm{EM}}:=\{\Theta^{\prime}\in\Omega|\Theta^{\prime}=\operatorname*{arg\,min}_{\Theta\in\Omega}\mathbb{E}_{i\in[N]}\left[\mathcal{L}(\mathrm{MHLA}_{\Theta}(Z_{i}),y_{i})\right]\}$$  -
$$(38)$$

$$(39)$$

Let vERM be defined as
$$v_{\mathrm{ERM}}:=\{v^{\prime}\in\mathbb{R}^{\psi}|v^{\prime}=\operatorname*{arg\,min}_{v\in\mathbb{R}^{\psi}}\mathbb{E}_{i\in[N]}\left[\mathcal{L}(\mathcal{R}_{v}(\mathcal{H}(Z_{i})),y_{i})\right]\}$$
Ei∈[N][L(Rv(H(Zi)), yi)]} (39)
Observe that for all Θ ∈ ΘERM, we have p(Θ) ⊆ vERM. Here we use the fact that y is realizable by the ground truth Θ¯ . Therefore if we show that vERM is unique, i.e comprised of a single element then pERM := {p(Θ)|Θ ∈ ΘERM} is also unique. Therefore, MHLAΘ is the same function for any Θ ∈ ΘERM To show vERM is unique, all we need is that the second moment of the features ΛD =
ED-H(Z)H(Z)
Tis positive definite (the covariance has a minimum eigenvalue bounded away from zero).

Next we prove the main certifiable identifiability lemma by instantiating the polynomials H and p from Lemma E.1.

Lemma A.3 (Certificate of Identifiability). Let dataset D = {(Zi, yi)}i∈[N] be a realizable dataset.

Let H = {Hn}∞
n=1 *be a family of polynomials* Hn : R
d×n → R
ψ for ψ =d2 d + d 2 *defined as* follows. We index the entries of H by taking the Kronecker product between all sets of pairs {*j, k*}
(for all j, k ∈ [d]) with with all ℓ ∈ [d]. We define H(Z){j,k}ℓ *as in Algorithm 2 to be* H(Z){j,k}ℓ:= ⟨zj:, zk:⟩zℓni. (18)
Then if λmin ED-H(Z)H(Z)
T > 0, we have that MHLAΘ is identifiable with respect to D.

$$Then\;i f\,\lambda_{m i n}\;\bigl(\mathbb{E}_{D}\;\bigr)$$

Proof. First we construct a polynomial p : Ω → R
ψ and H : R
d×n → R
ψ for ψ =d2 d + d 2such that

$$\mathrm{MHLA}_{\Theta}(Z)[a]=\langle p_{a}(\Theta),{\mathcal{H}}(Z)\rangle$$

We begin by rewriting MHLAΘ(Z)[a]. We index the first d2 d entries of pa(Θ) by all pairs {*j, k*}
for *j, k* ∈ [d] and all ℓ ∈ [d].

$$p_{a}(\Theta)_{\{j,k\}},_{\ell}):=\sum_{h\in[H]}(V_{h,aj}Q_{h,k\ell}+V_{h,ak}Q_{h,j\ell})$$  We define the entries of $p_{a}(\Theta)$ from $[\binom{d}{2}d,\binom{d}{2}d+d^{2}$ as follows.  $$p_{a}(\Theta)_{\{j^{2}\}}(\ell):=\sum_{h\in[H]}V_{h,aj}Q_{h,j\ell}$$

Similarly, we define H(Z) be be the following d2
d + d
2features. H(Z){j,k}{ℓ} and H(Z){ℓ}.
H(Z){j,k}{ℓ}:= ⟨zj:, zk:⟩zℓn (43)
and
$${\mathcal{H}}(Z)_{\{j^{2}\}\{\ell\}}:=\|z_{j}.\|^{2}z_{\ell n}$$
2zℓn (44)
Thus we rewrite $\text{MHLA}_{\Theta}(Z)[a]$ as  $$\text{MHLA}_{\Theta}(Z)[a]=\sum_{\{j,k\}\in\mathcal{S}_{\ell}^{2}}\sum_{\ell\in[a]}p_{a}(\Theta)_{\{j,k\},\{\ell\}}\mathcal{H}(Z)_{\{j,k\}\{\ell\}}+\sum_{j,\ell\in[a]}p_{a}(\Theta)_{\{j^{2}\}\{\ell\}}\mathcal{H}(Z)_{\{j^{2}\}\{\ell\}}\tag{45}$$ $$=\langle p_{a}(\Theta),\mathcal{H}(Z)\rangle$$
$$(40)$$
$$(41)$$
$$(42)$$
$\left(44\right)$ . 
Here we introduce the notation S
d 2to denote the set of all pairs {*j, k*} for *j, k* ∈ [d]. We have constructed a polynomial pa(Θ) such that for any Θ, Θ′ ∈ Ω in the same equivalence class pa(Θ) = pa(Θ′), we have MHLAΘ = MHLAΘ′ . Furthermore, if there exists b ∈ [n] such that λmin ED-H(Z)H(Z)
T > 0 then OLS returns a unique solution for pa(Θ). Since the data is realizable, we conclude pa(Θ) = pa(Θ) ¯ for all Θ ∈ ΩERM.

Next we present the proof that realizability is not necessary to identify the function learned by MHLA with more than d 2 heads.

Lemma A.6 (Identifiability without realizability for MHLA with arbitrarily many heads). *Let dataset* D = {(Zi, yi)}i∈[N] be any dataset drawn i.i.d from a distribution D. Let H be defined as in Lemma A.3. Then if λmin ED[H(Z)H(Z)
T]> 0 then MHLAΘ for Θ ∈ ΩH *for any* H ∈ [d 2, ∞)
is identifiable with respect to the data D. That is, MHLAΘ = *MHLA*Θ′ (21)
for all pairs of empirical risk minimizers Θ, Θ′ ∈ ΩERM.

$$(21)$$
$HLA_{\Theta'}$  $M_{1}$
Proof. We know from [lemma main algorithm] there exists a surjective map pa(Θ) that takes Θ ∈ Ω
into v ∈ R
ψ. This implies that for all v ∈ R
ψ there exists a right inverse function p r(v) = Θ
satisfying p(Θ) = v given by SVD. Therefore, p(ΘERM) ∈ vERM i.e optimizing over v ∈ R
ψ does no better than optimizing over Θ ∈ Ω. To prove this consider the contrary that there exists v
′ ∈ vERM
and there is no Θ ∈ Ω that achieves the same empirical risk as v
′. However, p r(v) ∈ Ω is such a Θ, and we have a contradiction. The key point is that we avoid the assumption of realizability and replace it with surjectivity of the polynomials pa. Finally we prove that data drawn from independent noise is certifiably identifiable. A subtlety in the proof is that we use a somewhat different set of polynomials than Lemma A.3 as we center and normalize our features, which still satisfies the assumptions of the general certificate Lemma E.1 Lemma A.5 (Independent input noise yields identifiability). Let (Z, y) ∼ D be a realizable dataset.

Let Z be drawn from a distribution Z where the (a, b)-th entry of Z denoted by Zab is drawn i.i.d.

from a distribution ν over R for all a ∈ [d] and b ∈ [n]*. Let the second and fourth moment of* ν be denoted m2 and m4 respectively. Let m2 > 0 and m4 > m22. Then MHLAΘ for Θ ∈ ΩH is identifiable with respect to D. That is to say, for any population risk minimizers Θ, Θ′ ∈ ΩPRM:
MHLAΘ = *MHLA*Θ′ . (20)
Proof. We give the entries of Λ(Z) the following naming convention. Let the terms {*j, k*}{ℓ} and pairs {j
′, k′}{ℓ
′}. Terms that involve {j 2}{ℓ} and {j
′2}{ℓ
′} are referred to as 'singles'.

$$\mathbb{E}\left[\mathcal{H}_{b}(Z)_{\{j,k\}\{\ell\}}\mathcal{H}_{b}(Z)_{\{j^{\prime},k^{\prime}\}\{\ell^{\prime}\}}\right]=\frac{1}{n}\mathbb{E}\left[\langle z_{j},z_{k}\rangle\langle z_{j^{\prime}},z_{k^{\prime}}\rangle z_{\ell b}z_{\ell b}\right]$$  We give entries of the following form the name "sings to singles"  $$\mathbb{E}\left[\mathcal{H}_{b}(Z)_{\{j^{2}\}}\{\ell\}\,\mathcal{H}_{b}(Z)_{\{j^{2}\}}\{\ell\}\right]=\frac{1}{n}\mathbb{E}[(\|z_{j}\|^{2}-nm_{2})(\|z_{j^{\prime}}\|^{2}-nm_{2})z_{\ell b}^{2}]$$
′b] (46)
ℓb] (47)
For the case of Z drawn with each entry i.i.d ν we can proceed via case work.

Case 1: Pairs to Pairs, j ̸= k and j
′ ̸= k
′

$$(46)$$
1. **Subcase 1: $\{j,k\}\neq\{j^{\prime},k^{\prime}\}$ and $\ell=\ell^{\prime}$:** $$\frac{1}{n}\mathbb{E}[\langle z_{j},z_{k^{\prime}}\rangle\langle z_{j^{\prime}},z_{k^{\prime}}\rangle z_{\ell b}z_{\ell^{\prime}b}]=0$$
2. **Subcase 2: $\{j,k\}=\{j^{\prime},k^{\prime}\}$ and $\ell=\ell^{\prime}$:**  $$\frac{1}{n}\mathbb{E}[\langle z_{j},z_{k_{i}}\rangle^{2}z_{\ell b}^{2}]=m_{2}^{3}$$
ℓb] = m32(49)
Case 2: Singles to Singles, j = k and j
′ = k
′

 ## 1. Subcase 1: $j\neq j'$ and $\ell=1$
$$\begin{array}{l}{{\mathrm{and}\;e=e^{-i\theta}}}\\ {{\frac{1}{n}\mathbb{E}\left[\left(\|z_{j};\|^{2}-n m_{2}\right)\left(\|z_{j^{\prime}};\|^{2}-n m_{2}\right)z_{\theta b}^{2}\right]=0}}\end{array}$$
$${\boldsymbol{\ell}}=\ell^{\prime}{\boldsymbol{\cdot}}$$
2. **Subcase 2:**$j=j^{\prime}$**and**$\ell=\ell^{\prime}$**: $$\frac{1}{n}\mathbb{E}\left[\left(\left\|z_{j\cdot}\right\|^{2}-nm_{2}\right)^{2}z_{th}^{2}\right]=\frac{1}{n}\left((n^{2}-n)m_{2}^{2}+nm_{4}-n^{2}m_{2}^{2}\right)m_{2}=(m_{4}-m_{2}^{2})m_{2}$$ (51)

$$(47)$$
$$(48)$$
$$(49)$$
$$(50)$$