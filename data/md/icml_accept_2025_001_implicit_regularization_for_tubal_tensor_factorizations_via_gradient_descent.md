# Implicit Regularization For Tubal Tensor Factorizations Via Gradient Descent

Santhosh Karnik * 1 **Anna Veselovska** * 2 3 **Mark Iwen** 4 5 **Felix Krahmer** 2 3

## Abstract

We provide a rigorous analysis of implicit regularization in an overparametrized tensor factorization problem beyond the lazy training regime. For matrix factorization problems, this phenomenon has been studied in a number of works. A particular challenge has been to design universal initialization strategies which provably lead to implicit regularization in gradient-descent methods. At the same time, it has been argued by (Cohen et al., 2016) that more general classes of neural networks can be captured by considering tensor factorizations. However, in the tensor case, implicit regularization has only been rigorously established for gradient flow or in the lazy training regime. In this paper, we prove the first tensor result of its kind for gradient descent rather than gradient flow. We focus on the tubal tensor product and the associated notion of low tubal rank, encouraged by the relevance of this model for image data. We establish that gradient descent in an overparametrized tensor factorization model with a small random initialization exhibits an implicit bias towards solutions of low tubal rank. Our theoretical findings are illustrated in an extensive set of numerical simulations show-casing the dynamics predicted by our theory as well as the crucial role of using a small random initialization.

## 1. Introduction

1 overparametrization can give rise to superior generalization capability and lead to strong overall NN performance. Consequently, there has been a recent surge in research aimed at explaining how gradient-based methods interact with overparameterized models under nonconvex losses (see, e.g.,
(Ma et al., 2018; Ling & Strohmer, 2019)). Notably, recent empirical and theoretical studies have suggested that gradient-based methods with small random initializations exhibit a bias towards low-rank solutions in a variety of models. For matrix factorization models which represent linear neural networks, a rigorous analysis of implicit bias is available for both gradient descent (Gunasekar et al., 2018; Stoger & ¨ Soltanolkotabi, 2021) and gradient flow (its asymptotic limit for small step size) (Bah et al., 2022; Chou et al., 2024). In contrast, for neural networks with nonlinear activation, there has been a good deal of work done showing that fully connected layers can be represented by, e.g., tensor train factorizations in (Novikov et al., 2015; Razin et al., 2021). As a consequence, it has been argued that tensor factorizations should be considered instead of matrix factorizations (see, e.g., (Cohen et al., 2016)). For tensor factorization models, however, results predating 2024 were only available for the asymptotic regime, i.e., gradient flow. This is perhaps due to the many additional complications in the tensor setting beyond those in the matrix setting including, e.g, that there are many different valid notions of tensor rank, each of which motivates its own equally valid class of tensor factorizations. For gradient descent applied to the tensor recovery problem, only a very recent partial analysis by (Liu et al., 2024) currently exists for the tubal factorization model. This analysis requires that the initialization already well approximates the solution, only after which the convergence of gradient descent toward a low tubal-rank solution is shown. Herein we also focus on the tubal factorization, but establish the corresponding implicit regularization result without needing such a strong initialization assumption. Our work is motivated by recent research showing that the way neural networks are trained, especially with gradient descent, can lead to solutions with useful structure, even without adding explicit regularization terms. This phenomenon, known as implicit regularization, has been studied in contexts such as sparse recovery (Vaskevicius et al., 2019) and low-rank matrix completion (Li et al., 2020), where specific network architectures are designed to encourage certain types of structure in the solutions. However, for tensor recovery problems, most existing work either focuses only on gradient flow or provides only partial analysis. To the best of our knowledge, our paper is the first to analyze implicit bias under gradient descent with small random initialization for a tensor recovery problem. We focus on the tubal rank model, which is particularly relevant for applications like video representation. This opens the door to a broader investigation into how implicit regularization can be used for structured tensor recovery, how network architectures influence this bias, and what conditions ensure convergence. We see this work as a starting point for a larger line of research on implicit regularization in tensor problems. Related work: In deep learning it is common to use more network parameters than training points. In such overparameterized scenarios there are usually many networks that achieve zero training error so that the training algorithm effectively imposes an implicit regularization (bias) on the solution it computes. In practice, training networks with gradient descent is both common and tends to favor solutions that generalize well, offering the exploration of how gradient descent implicitly regularlizes in overparameterized regimes as one avenue for better understanding the success of deep learning more widely. As a result, a lot of recent work has been focussed on understanding the implicit regularization phenomena of gradient descent in multiple settings. The first theoretical works in this direction (Gunasekar et al., 2017; 2018; Geyer et al., 2020; Arora et al., 2019; Soudry et al., 2018) concentrated on training linear networks and suggested that during training (stochastic) gradient descent implicitly converges to a linear network (i.e., a linear function described by a matrix) that's low rank. Motivated by specific deep learning tasks, multiple works also investigated implicit bias phenomena in the special cases of sparse vector and low-rank matrix recovery from underdetermined measurements via an overparameterized square loss functional, where the vectors and matrices to be reconstructed were deeply factorized into several vector/matrix factors. In this setting, these works then showed that the dynamics of vanilla gradient descent are biased towards sparse/low-rank solutions, respectively (Chou et al., 2024; 2023; Li et al., 2022; Kolb et al., 2023). In the realm of optimization, a substantial body of work has also emerged that provides guarantees for gradient descent's convergence in the nonconvex setting for different problems such as phase retrieval, matrix completion, and blind deconvolution. Broadly, these findings can be categorized into two main approaches: smart initialization coupled with local convergence (demonstrating, e.g., local convergence of descent techniques starting from carefully designed spectral initializations) (Ma et al., 2018; Tu et al., 2016; Ling &
Strohmer, 2019; Candes et al., 2015); and landscape analysis paired with saddle-escaping algorithms which show, e.g., that all local minima are global and that saddle points exhibit strict negative curvature so that (stochastic) gradientbased methods can effectively escape saddles and ensure convergence to global minimizers (Jin et al., 2017; Ge et al., 2015; Raginsky et al., 2017). Notably, several studies (Woodworth et al., 2020; Ghorbani et al., 2020) have highlighted the importance of the scale of the training initialization for the generalization and test performance of modern machine learning architectures. In fact, a small random initialization followed by (stochastic) gradient descent is arguably the most widely used training algorithm in contemporary machine learning. And, stronger generalization performance is typically observed with smaller-scale initializations. Implicit bias for low-rank matrix recovery with small random initializations has been extensively studied in this setting as a result by, e.g., (Stoger ¨ & Soltanolkotabi, 2021; Soltanolkotabi et al., 2023; Wind, 2023; Kim & Chung, 2024). These studies have shown that a small random Gaussian initialization behaves similarly to a spectral initialization in overparameterized settings. Furthermore, they have shown that gradient descent algorithms with this initialization tend to converge towards low-rank solutions (i.e., that they demonstrate an implicit regularization towards low-rank solutions). Recently, numerous connections between tensor decompositions and training neural networks have also been established by, e.g., (Novikov et al., 2015; Razin et al., 2021; 2022). These studies argue that low-rank tensor factorization helps explain implicit regularization in deep learning, as well as how properties of real-world data translate this regularization to generalization. Similar to how matrix factorization can be viewed as a linear neural network (i.e., a fully connected network with linear activation), tensor factorizations correspond to a specific type of shallow (depthtwo) nonlinear convolutional neural network (Cohen et al., 2016; Razin et al., 2021). Additionally, (Novikov et al., 2015) demonstrated that the dense weight matrices of fully connected layers can be converted to tensor trains while preserving the layer's expressive power. These findings have positioned low-rank tensor factorizations as theoretical surrogates for various neural network learning settings, thereby enhancing our understanding of implicit regularization and overparameterization, and so further motivating investigation in this area. Since no unique definition of tensor rank is available, related literature concerning implicit bias has naturally split with respect to the notion of tensor rank being considered: CP- rank, Tucker-rank, and tubal-rank, in analogy to the analysis of algorithms specifically designed for tensor recovery and completion by, e.g., (Zhang et al., 2019; Hou et al., 2021; Kong et al., 2018; Ahmed et al., 2020; Liu et al., 2019; 2020; Haselby et al., 2024). For the CP-tensor factorization, several results are available for gradient-based methods (Wang et al., 2020; Ge & Ma, 2017). The first theoretical analysis of implicit regularization towards low tensor rank under arbitrarily small initialization was provided considering gradient flow in (Razin et al., 2021). In (Ge et al., 2015), it has been shown for the orthogonal tensor decomposition problem a simple variant of the stochastic gradient algorithm is able to leverage a low-rank structure from an arbitrary starting point. In addition, (Wang et al., 2020) shows that using gradient descent on an over-parametrized objective for the CP-rank tensor decomposition problem one could go beyond the lazy training regime and utilize certain low-rank structures. Perhaps most closely related to this paper, very recently (Liu et al., 2024) analyzed the convergence of factorized gradient descent for the low-tubal-rank sensing problem, showing that with carefully designed spectral initialization the gradient iterates converge to a low-tubal rank tensor. Although the authors in (Liu et al., 2024) allow for overparametrization, they argue the minimal recovery error can be achieved when knowing the true rank, thereby leaving questions concerning the advantages of overparametrization and small random initializations open. Our contribution: Motivated by connections between tensor rank and non-linear neural network representations, herein we study the implicit regularization phenomenon for low tubal-rank tensor recovery. Namely, our objective is to analyze the recovery process of a tensor with a low tubalrank factorization (Kilmer & Martin, 2011) (see Fig 1) from a limited number of random linear measurements. More specifically, we consider tensors of the form X ∗X
⊤ and employ a non-convex method based on the tensor factorization, minimizing the loss function using gradient descent with a small random initialization. To the best of our knowledge, we are the first to investigate the implicit bias phenomenon for gradient descent with a small random initialization applied to a tensor factorization. Namely, we demonstrate that, irrespective of the degree of overparameterization, vanilla gradient descent with a small random initialization applied to a tubal tensor factorization will consistently converge to a low tubal-rank solution. Inspired by recent results for the low-rank matrix sensing problem by (Stoger & Soltanolkotabi, 2021), we establish ¨ that gradient descent iterates with small random initializations can be closely approximated by power method iterations in (Gleich et al., 2013; Kilmer et al., 2013) modulo normalization, and deduce that after sufficient time the iterates approach a commonly used spectral initialization from the tubal-rank literature in (Liu et al., 2024). Along the way we must also overcome, e.g., a challenging intersection between the tensor slices during each gradient descent iterate which forces a non-trivial convergence analysis.

Organization: In Section 2, we define our notation and present a few basic facts regarding tubal tensors. In Section 3, we state our problem and our main result. In Section 4, we outline the steps of the proof in order to provide intuition. In Section 5, we show numerical experiments which demonstrate our theoretical findings. We conclude the paper in Section 6. The proof of our main result is broken up into several lemmas, which are stated and proven in the appendix.

## 2. Notation And Preliminaries

Every tensor in this paper will be an order-3 tensor whose third mode is length k. For such a tensor T ∈ R
m×n×k, we define a block-diagonal Fourier domain representation by

$${\overline{{\mathcal{T}}}}={\mathrm{blockdiag}}({\overline{{\mathcal{T}}}}^{(1)},\ldots,{\overline{{\mathcal{T}}}}^{(k)})\in\mathbb{C}^{m k\times n k}$$

where the j-th block T
(j)∈ C
m×n is defined by T
(j)(*i, i*′) = Pk j
′=1 T (i, i′, j′)e
−
√−12π(j−1)(j
′−1)/k. In other words, we take the FFT of each tube, and then arrange the resulting frontal slices into a block-diagonal matrix. The tubal product (or t-product) of two tubal tensors A ∈
R 
m×q×kand B ∈ R
q×n×kis a tubal tensor A ∗ B ∈
R 
m×n×k whose tubes are given by

$$({\mathcal{A}}*{\mathcal{B}})(i,i^{\prime},:)=\sum_{p=1}^{q}{\mathcal{A}}(i,p,:)*{\mathcal{B}}(p,i^{\prime},:).$$

Here, ∗ denotes the circular convolution operation, i.e., (x ∗
y)i =Pk j=1 xjyi−j (mod k). One can check that A ∗ B =
A B.

measurements of X ∗ X
⊤, that is For any tubal tensor T ∈ R
m×n×k, its tubal transpose T
⊤ ∈ R
n×m×kis given by (T
⊤)(*i, i*′, 1) = T (i
′*, i,* 1)
and (T
⊤)(i, i′, j) = T (i
′*, i, k* + 2 − j) for j = 2*, . . . , k*,
i.e., we take the transpose of each face, and then reverse the order of frontal slices j = 2*, . . . , k*. This ensures that T
⊤ = T
⊤.

For any n, the n × n × k identity tensor I ∈ R
n×n×k is defined by I(:, :, 1) = In×n (identity matrix), and I(: , :, j) = 0n×n (zero matrix). An orthogonal tensor Q ∈
R 
n×n×ksatisfies Q∗Q⊤ = Q⊤ ∗Q = I. An orthonormal tensor W ∈ R
m×n×k with m ≥ n satisfies W⊤ ∗W = I.

The tubal-SVD (Kilmer & Martin, 2011) (or t-SVD) of a tubal tensor T ∈ R
m×n×kis a factorization of the form

$${\mathcal{T}}={\mathcal{U}}*\Sigma*{\mathcal{V}}^{\top}$$
⊤ (2.1)
where U ∈ R
m×m×kand V ∈ R
n×n×kare orthogonal, and each frontal slice of Σ ∈ R
m×n×kis diagonal. The t-SVD
of a tensor T ∈ R
m×n×kcan be computed as follows: (1)
compute the FFT of each tube of T to get the frontal slices T
(j), j = 1*, . . . , k*, (2) compute the SVD of each resulting frontal slice T
(j)= U
(j)Σ
(j)V
(j)⊤, (3) concatenate the matrices {U
(j)}
k j=1 into a tubal tensor Ue ∈ C
m×m×kand take the inverse FFT along mode-3 to obtain U ∈ R
m×m×k
(and similarly to obtain Σ ∈ R
m×n×kand V ∈ R
n×n×k).

The tubal rank of a tensor T ∈ R
m×n×kis the number of non-zero diagonal tubes in the Σ tensor of its t-SVD, i.e., rank(T ) = \#{i : Σ(*i, i,* :) ̸= 0}. For an illustration of the t-SVD decomposition, see Figure 1. We also define the condition number κ(T ) of the tubal tensor T ∈ R
m×n×k by

$$\kappa({\mathcal{T}}):={\frac{\sigma_{1}({\overline{{{\mathcal{T}}}}})}{\sigma_{\operatorname*{min}\{m,n\}k}({\overline{{{\mathcal{T}}}}})}}.$$

Finally, for tubal tensors T ∈ R
m×n×k we define the tensor spectral norm ∥T ∥ := ∥T ∥ and the tensor nuclear norm ∥T ∥∗ := ∥T ∥∗ as the spectral and nuclear norm respectively of the block-diagonal Fourier domain representation T , and the tensor Frobenius norm
∥T ∥
2 F:= Pm i=1 Pn j=1 Pk ℓ=1 T (*i, j, ℓ*)
2 =
1 k
∥T ∥
2 Fas a scaled version of the Frobenius norm of the block-diagonal Fourier domain representation T .

## 3. Main Results

Problem Formulation Let X ∈ R
n×r×k have tubal rank r ≤ n so that X ∗ X
⊤ ∈ S
n×n×k
+ is a tubal positive semidefinite tensor with tubal rank r. Let κ = κ(X ) be the condition number of X . Suppose we observe m linear

$$y_{i}=\left\langle{\cal A}_{i},{\cal X}*{\cal X}^{\top}\right\rangle\quad\mbox{for}\quad i=1,\ldots,m\tag{3.1}$$

where each Ai ∈ S
n×n×kis a tubal-symmetric tensor.

We can write this compactly as y = A(X ∗ X
⊤) where A : S
n×n×k → R
m is the linear measurement operator. We aim to recover X ∗ X
⊤ from our measurements y by using gradient descent to learn an overparameterized factorization.

Specifically, we fix an R ≥ r and try to find a U ∈ R
n×R×k such that U ∗ U
⊤ = X ∗ X
⊤ by using gradient descent to minimize the loss function

$$\ell(\mathcal{U}):=\left\|\mathcal{A}\left(\mathcal{U}*\mathcal{U}^{\top}\right)-\mathcal{Y}\right\|_{2}^{2}$$ $$=\sum_{i=1}^{m}\left(\left\langle\mathcal{A}_{i},\mathcal{U}*\mathcal{U}^{\top}\right\rangle-y_{i}\right)^{2}.$$
$$(3.2)$$  $$(3.3)$$
2. (3.3)
$$(2.1)$$

We will start with a small random initialization U0 ∈
R 
n×R×k where each entry is i.i.d. N (0, α 2 R
) for some small α > 0. Then, the gradient descent iterations are given by

$$\mathcal{U}_{t+1}=\mathcal{U}_{t}-\mu\nabla\ell(\mathcal{U}_{t})$$ $$=\mathcal{U}_{t}+\mu\mathcal{A}^{*}\left[\boldsymbol{y}-\mathcal{A}\left(\mathcal{U}_{t}*\mathcal{U}_{t}^{\top}\right)\right]*\mathcal{U}_{t}$$ $$=\left[\mathcal{I}+\mu(\mathcal{A}^{*}\mathcal{A})\left(\mathcal{X}*\mathcal{X}^{\top}-\mathcal{U}_{t}*\mathcal{U}_{t}^{\top}\right)\right]*\mathcal{U}_{t}\tag{3.4}$$

for some suitably small stepsize µ > 0. Here A∗: R
m → S
n×n×k denotes the adjoint of A which is given by A∗z =Pm i=1 ziAi.

Moreover, we say that a measurement operator A : S
n×n×k → R
m satisfies the Restricted Isometry Property (RIP) of rank-r with constant δ > 0 (abbreviated RIP(r, δ)), if we have

$$(1-\delta)\|\mathbf{Z}\|_{F}^{2}\leq\|\mathcal{A}(\mathbf{Z})\|_{2}^{2}\leq(1+\delta)\|\mathbf{Z}\|_{F}^{2},$$

for all Z ∈ S
n×n×k with tubal-rank ≤ r. We note that an RIP condition is a standard condition in the literature, and is used in similar works such as (Li et al., 2018; Stoger & ¨
Soltanolkotabi, 2021). This condition is necessary to ensure that there is only one low tubal rank tensor for which the loss function is zero, and that this tensor could be recovered stably in the presence of noise.

Results We have analyzed the convergence process of the gradient descent iterates (3.4) in the scenario of small random initialization and overparametrization. Namely, with the ground truth tensor X ∈ R
n×r×k, we assume the initialization U0 ∈ R
n×R×kis such that each entry is i.i.d.

N (0, α 2 R
) with small scaling parameter α > 0 and the second dimension R exceeding three timesthe ground truth dimension r. Below, we present the direct results of our analysis.

Theorem 3.1. Suppose we have m *linear measurements* y = A(X ∗ X
⊤) *of a tubal positive semidefinite tensor* X ∗ X
⊤ ∈ S
n×n×k
+ *where* X ∈ R
n×r×k *has tubal rank* r ≤ n. We assume A satisfies RIP(2r + 1, δ) *with* δ ≤
cκ−4r
−1/2*. Suppose we fit a model* X ∗ X
⊤ = U ∗ U
⊤
where U ∈ R
n×R×k with R ≥ 3r and obtain U by running the gradient descent iterations

$${\mathcal{U}}_{t+1}=\left[{\mathcal{I}}+\mu({\mathcal{A}}^{*}{\mathcal{A}})\left({\mathcal{X}}*{\mathcal{X}}^{\top}-{\mathcal{U}}_{t}*{\mathcal{U}}_{t}^{\top}\right)\right]*{\mathcal{U}}_{t}$$

with a stepsize µ ≤ c
√kκ−4∥X ∥
2starting from the initialization U0 ∈ R
n×R×k *where each entry is i.i.d.* N (0, α 2 R
).

Then, if the scale of the initialization satisfies

$$\alpha\lesssim\frac{\sigma_{\mathrm{min}}(\mathcal{X})}{\kappa^{2}\operatorname*{min}\{n,R\}\sqrt{k}}\left(\frac{C_{2}\kappa^{2}\sqrt{n}}{\sqrt{\operatorname*{min}\{n,R\}}}\right)^{-16\kappa^{2}},$$

then after

$$\widehat{t}\lesssim{\frac{1}{\mu\sigma_{\mathrm{min}}(\mathbf{\mathcal{X}})^{2}}}\ln\left({\frac{C_{1}\kappa n}{\operatorname*{min}\{n,R\}}}\operatorname*{min}\left\{1,{\frac{\kappa r}{k(\operatorname*{min}\{n,R\}-r)}}\right\}{\frac{\|\mathbf{\mathcal{X}}\|}{k\alpha}}\right)$$

iterations, we have that

$$\frac{\|\mathcal{U}_{\hat{t}}*\mathcal{U}_{\hat{t}}^{\top}-\mathcal{X}*\mathcal{X}^{\top}\|_{F}^{2}}{\|\mathcal{X}\|^{2}}\leq$$ $$k^{\frac{61}{32}}r^{\frac{1}{8}}\kappa^{\frac{-3}{16}}\left(\operatorname*{min}\{n,R\}-r\right)^{\frac{3}{8}}\left[\frac{C_{2}\kappa^{2}\sqrt{n}}{\sqrt{\operatorname*{min}\{n,R\}}}\right]^{21\kappa^{2}}\left[\frac{\alpha}{\|\mathcal{X}\|}\right]^{\frac{21}{16}}$$

holds with probability at least 1 − Cke−cR˜*. Here,* c, c, C, C ˜ 1, C2 > 0 *are fixed numerical constants.*
Intuitively, this means that if the initialization is sufficiently small, gradient descent will approximately recover the low tubal rank tensor X ∗ X
⊤ after bt iterations. Note that the reconstruction error can be made arbitrarily small by making the size of the random initialization α arbitrarily small. This comes at the expense of requiring more iterations. However, this impact is mild as the number of iterations grows only logarithmically with respect to α. Although the above theorem holds for any R ≥ 3r, it is perhaps most interesting in the case where R ≥ n as then every n × n × k tubal positive semidefinite tensor can be expressed as U ∗ U
⊤ for some U ∈ R
n×R×k. Hence, the learner model does not assume that the ground truth tensor has low tubal rank, yet gradient descent is able to recover the ground truth tensor instead of any of the infinitely many high tubal rank tensors whose measurements match that of the ground truth tensor. We note that (Zhang et al., 2019) shows that a random sub-
Gaussian measurement operator A : R
n×n×k → R
m will satisfy the RIP for tubal rank-r tensors with RIP constant δ with high probability if m ≥ O(*rnk/δ*2). To obtain an RIP
constant of δ = O(κ
−4r
−1/2), one needs m ≥ O(κ 8r 2nk)
random sub-Gaussian measurements. Additionally, we acknowledge that the parameter dependence in Theorem 3.1 may initially seem unfamiliar. However, it aligns well with intuition and prior work: when the tensor is ill-conditioned - i.e., possesses a small tubal singular value - gradient descent without regularization naturally struggles to recover the rank-one component unless the initialization is sufficiently small. While our bound exhibits exponential dependence on the condition number, this is consistent with known results in the matrix setting (e.g., see Lemma 8.6 in (Stoger & Soltanolkotabi, 2021)).Although ¨ the necessity of exponential dependence remains an open question, it presents a compelling direction for future research. Moreover, our numerical experiments (see Figure 4) support a polynomial relationship between the test error and the initialization parameter α, and while the empirical degree may differ slightly, our theoretical exponent 21 16 appears to closely approximate the observed behavior.

## 4. Proof Outline

In this section, we turn our attention to giving an overview of the key ideas of the proof.

In our analysis, we demonstrate that the trajectory of gradient descent iterations can be approximately divided into two distinct stages: (I) a spectral stage and (II) a convergence stage described below. (I) The spectral stage. In the spectral stage, where we show that the gradient descent starting from random initialization behaves similarly to spectral initialization, enabling us to prove that by the end of this stage, the column spaces of the tensor iterates Ut (3.4) and the ground truth matrix X
are sufficiently aligned. Namely, we show that the first few iterations of the gradient descent algorithm Ut can be approximated by the iteration of the tensor power method modulo normalization (see, e.g.(Gleich et al., 2013)) defined as

$${\widetilde{\mathcal{U}}}_{t}=\left({\mathcal{I}}+\mu{\mathcal{A}}^{*}{\mathcal{A}}({\mathcal{X}}*{\mathcal{X}}^{\top})\right)^{*t}*{\mathcal{U}}_{0}\in\mathbb{R}^{n\times R\times k}.$$

We call this part of the evolution of the gradient descent iteration the "spectral stage" since, due to its similarity to the power method, at the end of this stage the iterates Ut will be closely aligned with the classical t-SVD spectral initialization of (Liu et al., 2024).

(II) The convergence stage. In the convergence stage, the gradient iterates converge approximately to the underlying low tubal-rank tensor X ∗ X
⊤ at a geometric rate until reaching a certain error floor which is dependent on the initialization scale.

The cornerstone of the analysis of this stage is the decomposition of the tensor gradient iterates Ut into two components, the so-called "signal" and "noise" terms. This is done by adapting similar decomposition methods used in recent works analyzing implicit bias phenomenon for gradient descent in the matrix setting (see (Stoger & ¨ Soltanolkotabi, 2021; Li et al., 2018)) to our tensor setting. Accordingly, let the tensor-column subspace of the ground truth tensor X ∈ R
n×r×k be denoted by VX with the corresponding basis VX ∈ R
n×r×k. Consider the tensor VX ∗ Ut ∈ R
r×R×k with its t-SVD decomposition VX ∗ Ut = Vt ∗ Σt ∗ W⊤
t. For Wt ∈ R
R×r×k, we denote by Wt,⊥ ∈ R
R×(n−r)×ka tensor whose tensorcolumn subspace is orthogonal to those of Wt, that is
∥W⊤
t,⊥ ∗ Wt∥ = 0 and its projection operator PWt,⊥
is defined as PWt,⊥ = Wt,⊥ ∗ W⊤
t,⊥ = I − Wt ∗ W⊤
t.

We then decompose the gradient descent iterates (3.4) as follows

$$\mathcal{U}_{t}=\mathcal{U}_{t}*\mathcal{W}_{t}*\mathcal{W}_{t}^{\top}+\mathcal{U}_{t}*\mathcal{W}_{t,\perp}*\mathcal{W}_{t,\perp}^{\top}\tag{4.1}$$

referring to the tensors Ut ∗ Wt ∗ W⊤
tas the signal term of the gradient descent iterates, and to the tensors Ut ∗
Wt,⊥ ∗ W⊤
t,⊥ as the noise term. The advantage of such a decomposition is that the tensor-column space of the noise term Ut ∗Wt,⊥ ∗W⊤
t,⊥ is orthogonal to the tensor-column subspace of the ground truth X allowing for a rigorous analysis of the convergence process of the two components separately.

At the convergence stage, we show that symmetric tensor Ut ∗Wt ∗W⊤
t ∗U
⊤
t built from the signal term converges towards the ground truth tensor X ∗ X
⊤, whereas the spectral norm of the noise term ∥Ut ∗ Wt,⊥∥, stays small. Additional challenges in the tensor setting vs. matrix setting When coming from the matrix case to the tensor setting com, there are several important differences and challenges, which need to be carefully considered and are described below.

- In contrast to the matrix case, the range and kernel of a third-order tubal tensor can include overlapping generator elements (we refrain from using the term basis, in the sense that knowledge of the multirank and complimentary tubal scalar of a tensor must be included to describe the range). Namely, if in the t-SVD (2.1) of a symmetric tensor X the tensor Σ contains q non-invertible tubes - tubes that have zero elements in the Fourier domain –, then there are q common generators for the range and the kernel of X , please see (Kilmer et al., 2013) for more details. With this phenomenon, the decomposition (C.1) of the gradient iterates into signal and noise term is not available for non-invertible tubes, which is why we need to work with a more intricate notion of condition number.

- As stated in (Gleich et al., 2013), running the power method for tubal tensors of dimensions n × n × k is equivalent to running in parallel k independent matrix power methods in Fourier domain. However, running gradient descent in the tubal tensor setting is not equivalent to running k gradient descent algorithms independently in Fourier space. This can be easily seen when transforming the measurement operator part of the gradient descent iterates.

Pkq=1 Ai
(q), X(q)X(q)H, j = 1*, . . . m* then A∗A(X ∗ X
⊤) = A∗(y) = Pm i=1 yiAi ∈
S
n×n×kand the for j-th slice in the Fourier domain, we get A∗A(X ∗ X
⊤)
(j) = 
Pm i=1 Pk j=1 Ai
(j)Ai
(q), X(q)X(q)H. This means that in each Fourier slice Ut
(j) of the gradient descent iterates (3.4) we have the full information about the ground truth tensor X ∗ X
⊤ and not only about its j-th slice. In the spectral stage, this fact does not cause significant difficulties. However, in the convergence stage, in order to get the global estimates, it requires a thorough and vigilant analysis of intersections between the slices in the Fourier domain. In particular, this required nontrivial estimations, such as those presented in Lemmas E.4 and E.5, to control these interactions and provide the respective bounds, which require control of proximity of the auxiliary parameter A∗A(X ∗ X
⊤ − Ut ∗ U
⊤
t)(j)to the corresponding jth Fourier slice of X ∗ X
⊤ −Ut ∗U
⊤
t via the RIP property of the measurement operator A and aligned matrix subspaces. Another important point is that one need to choose the learning rate µ and the initialization scale α carefully for the noise term Ut ∗ W⊥,t to grow slowly enough in each of the tensor slices in order to not allow overtaking the signal term Ut ∗ Wt in the norm, see, e.g., Theorem E.1 and the usage of Lemma E.3 in its proof.

## 5. Numerical Experiments

To verify our theoretical findings, we set multiple numerical tests: from showing two phases of the gradient descent algorithm to demonstrating the advantages of overparametrization. These experimental results showcase not only the implicit regularization for the gradient descent algorithm toward low-tubal-rank tensors but also demonstrate the firmness of our theoretical findings. Our experiments were conducted on a MacBook Pro equipped with an Apple M1 processor and 16GB of memory, using MATLAB 2023a software. The corresponding code is available in our GitHub repository, https://github.com/AnnaVeselovskaUA/tubal-tensorimplicit-reg-GD.git.

We generate the ground truth tensor T ∈ R
n×n×k with tubal rank r by T = X ∗ X
⊤ , where the entries of X ∈ R
n×r×kare i.i.d. sampled from a Gaussian distribution N (0, 1), and then X is normalized. The entries of measurement tensor Ai are i.i.d. sampled from a Gaussian distribution N (0, 1 m ). In the following, we describe different testing scenarios for recovery of T via the gradient descent algorithm and their outcome. For all the experiments, we set the dimensions to n = 10, k = 4, r = 3, the learning rate µ = 10−5, and the number of measurements m = 254. Illustration of the two convergence stages. To illustrate the convergence process of the gradient iterates, for the ground truth tensor X ∗ X
⊤ ∈ R
n×n×kand its counterpart Ut ∗ U
⊤
t ∈ R
n×n×k being learned by the gradient descent, we consider the training error ℓ(Ut), the test error
∥Ut∗U
⊤
t −X∗X ⊤∥F
∥X∗X ⊤∥F, and the test error for their rth singular tubes σr(Ut), σr(X ) ∈ R
k,
∥σr(Ut)−σr(X)∥2
∥σr(X)∥2. Moreover, we also take into our consideration the tensor subspace L spanned by the tensor-columns corresponding to the first r singular-tubes of the tensor A∗A(X ∗ X
⊤) and denote by Lt the tensor-column subspace spanned by the tensorcolumns corresponding to the first r singular tubes Ut ∗ U
⊤
t.

We note that although Theorem 3.1 bounded a relative error with ∥X ∥
2in the denominator, we use ∥X ∗ X
⊤∥F in the denominator of the relative error for our experiments as it is a more natural relative error to consider. Furthermore, since ∥X ∗ X
⊤∥F ≥ ∥X ∥
2, and ∥X ∗ X
⊤∥F could be much larger than ∥X ∥
2in cases where the singular values of X ∗ X
⊤ vary drastically, the result of Theorem 3.1 is stronger than if we bounded the more natural Frobenius norm error. Besides, the qualitative behavior in the numerical simulation will be the same for the two error measures as generically they will just differ by a dimensional factor. Figures 2 demonstrates that the convergence analysis can be divided into two stages: the spectral and the convergence stage. We see that in the first stage (1 ≤ t ≲ 3000), the first r tensor-columns of Ut ∗ U
⊤
tlearn the tensor column subspace corresponding to the first r singular-tubes of the tensor A∗A(X ∗ X
⊤), i.e. the principal angle between the tensor column subspaces Lt and L becomes small. Namely, as one can observe in Figure 2 (bottom), the principal angle between the two subspaces, ∥V
⊤
L⊥ ∗VLt ∥, decreases where as the principal angle between X and Lt reaches certain plateau, see the behavior of ∥V
⊤
X ⊥ ∗ VLt∥. At the same time, test errors 
∥Ut∗U
⊤
t −X∗X ⊤∥F
∥X∗X ⊤∥Fand 
∥σr(Ut)−σr(X)∥2
∥σr(X)∥2 stay large. In the second stage, we see that the test error
∥Ut∗U
⊤
t −X∗X ⊤∥F
∥X∗X ⊤∥Fstarts decreasing, meaning that the gradient descent iterates Ut ∗ U
⊤
tstart converging to X ∗ X
⊤
by learning more about the tensor-column subspace of the ground truth tensor. At the same time, the test error over rth singular tube 
∥σr(Ut)−σr(X)∥2
∥σr(X)∥2starts decreasing too and as a result converges to zero. We also see that in this stage the principal angle between Lt and L grows, which is also intuitive as the tensor-column subspace L does not have the full information about the tensor-column subspace of the ground truth tensor X ∗ X
⊤, and learning more about X ∗ X
⊤ leads to a larger error in terms of principal angles of the two. Depiction of the alignment stage. In this experiment, we illustrate that gradient descent with small initialization behaves similarly to the tensor-power method modulo normalization in the first few iterations, bringing the gradient iterates close to the spectral tubal initialization, used, e.g., in (Liu et al., 2024). Here, as before L denote the tensor subspace spanned by the tensor-columns corresponding to the first r singular-tubes of tensor A∗A(X ∗ X
⊤) and Lt is the tensor-column subspace corresponding to the first r singular tubes Ut ∗ U
⊤
t. Additionally, Let denotes the tensor-column subspace spanned by the first r singular-tubes of the tensor Uet ∗Ue
⊤
t, where Ue
⊤
t =
I + A∗AX ∗ X
⊤∗t∗ U0.

In Figure 2 (bottom), we see that Ut and Uet learn the subspace L almost at the same rate in the first iterations, 1 ≤ t ≲ 3000. In the same figure, we observe that also the angle between VX and Lt, respectively Let, decreases monotonically in the spectral stage. Then at the beginning of the convergence stage, 3000 ≲ t, the angle between VX
and Lt starts decreasing gradually and converges to zero, as expected since Ut ∗U
⊤
tconverges to X ∗ X
⊤. Whereas the principal angle between L and Lt growths until it reaches a certain plateau. DMS 2108479 and NSF EDU DGE 2152014.

Test and train error under different scales of initialization. In this experiment, we explore the influence of the initialization scale, denoted by α, on the training and the test error. With R = 200, we apply gradient descent for various values of α, halting the iterations at t = 3500 in each run. The results, presented in Figure 4, demonstrate a reduction in test error as α decreases. Notably, the figure indicates that the test error follows an almost polynomial relationship with the initialization scale α. This observation is consistent with our theoretical predictions, which also forecast a decrease in test error at a rate of α, see Theorem 3.1. Impact of different levels of overparameterization on the convergence. In this numerical analysis, we set α = 10−7 and examined the convergence speed of gradient descent to the ground truth tensor for various overparameterization rates R. We run the experiment twenty times for each value of R and plot the averaged values per each iteration. The results, shown in Figure 3, reveal that increasing the number of tensor columns R, that is, overparameterizing, accelerates the convergence rate, resulting in fewer iterations to reach the desired error level. Additionally, overparameterization reduces the test error and the training error by affecting the spectral stages.

## 6. Conclusion And Outlook

In this paper, we focused on studying the implicit regularization of tubal tensor factorizations via gradient descent by showing that with small random initialization and overparametrization, the gradient descent algorithm is biased towards a low-tubal-rank solution. We have shown that the first iterations of gradient descent with small random initialization behave similarly to the tensor power method, which leads to learning in these first iterations the tensor-column spaces close to the tensor-column space of the ground truth. We also demonstrate that the implicit regularization from small random initialization guides the gradient descent iterations toward low-tubal rank solutions that are not only globally optimal but also generalize well.

## Acknowledgments

AV and FK acknowledge support by the German Science Foundation (DFG) in the context of the collaborative research center TR-109, the Emmy Noether junior research group KR 4512/1-1 and the Bavarian Funding Program for Initiating International Research Cooperation, as well as by the Munich Data Science Institute and Munich Center for Machine Learning. SK acknowledges support by the United States National Science Foundation in the context of the Foundations of Data Science Institute funded by grant NSF DMS 2022205. MI acknowledges support by the United States National Science Foundation grants NSF

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, and more specifically, the theoretical understanding of implicit regularization as a tool for structured recovery problems. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Ahmed, T., Raja, H., and Bajwa, W. U. Tensor regression using low-rank and sparse tucker decompositions. SIAM Journal on Mathematics of Data Science, 2(4):944–966, 2020.

Arora, S., Cohen, N., Hu, W., and Luo, Y. Implicit regularization in deep matrix factorization. Advances in Neural Information Processing Systems, 32, 2019.

Bah, B., Rauhut, H., Terstiege, U., and Westdickenberg, M. Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers. Information and Inference: A Journal of the IMA, 11(1): 307–353, 2022.

Candes, E. J., Li, X., and Soltanolkotabi, M. Phase retrieval via wirtinger flow: Theory and algorithms. IEEE Transactions on Information Theory, 61(4):1985–2007, 2015.

Chou, H.-H., Maly, J., and Rauhut, H. More is less: inducing sparsity via overparameterization. *Information and* Inference: A Journal of the IMA, 12(3):1437–1460, 2023.

Chou, H.-H., Gieshoff, C., Maly, J., and Rauhut, H. Gradient descent for deep matrix factorization: Dynamics and implicit bias towards low rank. Applied and Computational Harmonic Analysis, 68:101595, 2024.

Cohen, N., Sharir, O., and Shashua, A. On the expressive power of deep learning: A tensor analysis. In *Conference* on learning theory, pp. 698–728. PMLR, 2016.

Ge, R. and Ma, T. On the optimization landscape of tensor decompositions. Advances in neural information processing systems, 30, 2017.

Ge, R., Huang, F., Jin, C., and Yuan, Y. Escaping from saddle points—online stochastic gradient for tensor decomposition. In *Conference on learning theory*, pp. 797–842. PMLR, 2015.

Geyer, K., Kyrillidis, A., and Kalev, A. Low-rank regularization and solution uniqueness in over-parameterized matrix sensing. In *International Conference on Artificial* Intelligence and Statistics, pp. 930–940. PMLR, 2020.

Li, Y., Ma, T., and Zhang, H. Algorithmic regularization in over-parameterized matrix sensing and neural networks with quadratic activations. In Conference On Learning Theory, pp. 2–47. PMLR, 2018.

Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A.

When do neural networks outperform kernel methods?

Advances in Neural Information Processing Systems, 33:
14820–14830, 2020.

Li, Z., Luo, Y., and Lyu, K. Towards resolving the implicit bias of gradient descent for matrix factorization: Greedy low-rank learning. *arXiv preprint arXiv:2012.09839*, 2020.

Gleich, D. F., Greif, C., and Varah, J. M. The power and arnoldi methods in an algebra of circulants. Numerical Linear Algebra with Applications, 20(5):809–831, 2013.

Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi, S. J., Ye, K., Chern, F., Yu, F., Guo, R., et al. The lazy neuron phenomenon: On emergence of activation sparsity in transformers. *arXiv preprint arXiv:2210.06313*, 2022.

Gunasekar, S., Woodworth, B. E., Bhojanapalli, S.,
Neyshabur, B., and Srebro, N. Implicit regularization in matrix factorization. *Advances in neural information* processing systems, 30, 2017.

Ling, S. and Strohmer, T. Regularized gradient descent: a non-convex recipe for fast joint blind deconvolution and demixing. *Information and Inference: A Journal of the* IMA, 8(1):1–49, 2019.

Gunasekar, S., Lee, J. D., Soudry, D., and Srebro, N. Implicit bias of gradient descent on linear convolutional networks. *Advances in neural information processing* systems, 31, 2018.

Liu, X.-Y., Aeron, S., Aggarwal, V., and Wang, X. Lowtubal-rank tensor completion using alternating minimization. *IEEE Transactions on Information Theory*, 66(3): 1714–1737, 2019.

Haselby, C., Iwen, M., Karnik, S., and Wang, R. Tensor deli:
Tensor completion for low cp-rank tensors via random sampling, 2024.

Hou, J., Zhang, F., Qiu, H., Wang, J., Wang, Y., and Meng, D. Robust low-tubal-rank tensor recovery from binary measurements. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4355–4373, 2021.

Liu, X.-Y., Aeron, S., Aggarwal, V., and Wang, X. Lowtubal-rank tensor completion using alternating minimization. *IEEE Transactions on Information Theory*, 66(3): 1714–1737, 2020. doi: 10.1109/TIT.2019.2959980.

Jin, C., Ge, R., Netrapalli, P., Kakade, S. M., and Jordan, M. I. How to escape saddle points efficiently. In International conference on machine learning, pp. 1724–1732.

PMLR, 2017.

Liu, Z., Han, Z., Tang, Y., Zhao, X.-L., and Wang, Y. Lowtubal-rank tensor recovery via factorized gradient descent. arXiv preprint arXiv:2401.11940, 2024.

Ma, C., Wang, K., Chi, Y., and Chen, Y. Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval and matrix completion. In *International Conference on Machine Learning*, pp. 3345–3354. PMLR, 2018.

Kilmer, M. E. and Martin, C. D. Factorization strategies for third-order tensors. *Linear Algebra and its Applications*, 435(3):641–658, 2011.

Kilmer, M. E., Braman, K., Hao, N., and Hoover, R. C.

Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. SIAM Journal on Matrix Analysis and Applications, 34(1):148–172, 2013.

Novikov, A., Podoprikhin, D., Osokin, A., and Vetrov, D. P.

Tensorizing neural networks. Advances in neural information processing systems, 28, 2015.

Raginsky, M., Rakhlin, A., and Telgarsky, M. Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis. In Conference on Learning Theory, pp. 1674–1703. PMLR, 2017.

Kim, D. and Chung, H. W. Rank-1 matrix completion with gradient descent and small random initialization. Advances in Neural Information Processing Systems, 36, 2024.

Razin, N., Maman, A., and Cohen, N. Implicit regularization in tensor factorization. In *International Conference* on Machine Learning, pp. 8913–8924. PMLR, 2021.

Kolb, C., Muller, C. L., Bischl, B., and R ¨ ugamer, D. Smooth- ¨
ing the edges: A general framework for smooth optimization in sparse regularization using hadamard overparametrization. *arXiv preprint arXiv:2307.03571*, 2023.

Razin, N., Maman, A., and Cohen, N. Implicit regularization in hierarchical tensor factorization and deep convolutional neural networks. In International Conference on Machine Learning, pp. 18422–18462. PMLR, 2022.

Kong, H., Xie, X., and Lin, Z. t-schatten-p norm for lowrank tensor recovery. *IEEE Journal of Selected Topics in* Signal Processing, 12(6):1405–1419, 2018.

Rudelson, M. and Vershynin, R. Smallest singular value of a random rectangular matrix. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, 62(12):1707–1739, 2009.

Soltanolkotabi, M., Stoger, D., and Xie, C. Implicit bal- ¨
ancing and regularization: Generalization and convergence guarantees for overparameterized asymmetric matrix sensing. In *The Thirty Sixth Annual Conference on* Learning Theory, pp. 5140–5142. PMLR, 2023.

Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N. The implicit bias of gradient descent on separable data. *Journal of Machine Learning Research*, 19 (70):1–57, 2018.

Stoger, D. and Soltanolkotabi, M. Small random initializa- ¨
tion is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction. Advances in Neural Information Processing Systems, 34:23831–23843, 2021.

Tao, T. and Vu, V. Random matrices: The distribution of the smallest singular values. Geometric And Functional Analysis, 20:260–297, 2010.

Tu, S., Boczar, R., Simchowitz, M., Soltanolkotabi, M.,
and Recht, B. Low-rank solutions of linear matrix equations via procrustes flow. In *International Conference on* Machine Learning, pp. 964–973. PMLR, 2016.

Vaskevicius, T., Kanade, V., and Rebeschini, P. Implicit regularization for optimal sparse recovery. Advances in Neural Information Processing Systems, 32, 2019.

Vershynin, R. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

Wang, X., Wu, C., Lee, J. D., Ma, T., and Ge, R. Beyond lazy training for over-parameterized tensor decomposition. *Advances in Neural Information Processing Systems*, 33:21934–21944, 2020.

Wedin, P.-A. Perturbation bounds in connection with sin- ˚
gular value decomposition. *BIT Numerical Mathematics*, 12:99–111, 1972.

Wind, J. S. Asymmetric matrix sensing by gradient descent with small random initialization. *arXiv preprint* arXiv:2309.01796, 2023.

Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E.,
Savarese, P., Golan, I., Soudry, D., and Srebro, N. Kernel and rich regimes in overparametrized models. In Conference on Learning Theory, pp. 3635–3673. PMLR,
2020.

Zhang, F., Wang, W., Hou, J., Wang, J., and Huang, J.

Tensor restricted isometry property analysis for a large class of random measurement ensembles. *arXiv preprint* arXiv:1906.01198, 2019.

## Supplementary Material A. Outline Of Appendices

For ease of organization, we divide the supplementary material into appendices as follows. In Appendix B, we define some additional notation, including the angles between two tensor-column subspaces. In Appendix C, we decompose the gradient descent iterates into a "signal" term and a "noise" term, which will aid us in our analysis. In Appendices D and E, we analyze the spectral and convergence stages, respectively, of the gradient descent iterations. In Appendix F, we prove our main result. To avoid breaking up the flow of our analysis, we put some technical lemmas in the last few appendices instead of in the previously mentioned appendices. In Appendix G, we prove some properties of measurement operators which satisfy the restricted isometry property. In Appendix H, we prove some properties of matrices and their subspaces. Finally, in Appendix I, we prove some properties of random Gaussian tubal tensors.

## B. Additional Notation

For a tensor Y ∈ R
n×r×k, we denote its t-SVD by Y = VY ∗ ΣY ∗ W⊤Y with the two orthogonal tensor VY ,WY ∈ R
n×r×k, and the f-diagonal tensor ΣY ∈ R
r×r×k. We will refer to VY as the tensor-column subspace of Y and by VY⊥ ∈ R
n×(n−r)×k we denote the tensor-column subspace orthogonal to VY with its projection operator VY⊥ ∗ V
⊤
Y⊥ = I − VY ∗ V
⊤
Y .

We measure the angles between two tensor-column subspaces Y1 and Y2 by the tensor-spectral norm ∥VY⊥
1
∗ VY2
∥ which according to (Liu et al., 2019; Gleich et al., 2013; Kilmer & Martin, 2011) is equal to

$\|\mathcal{V}_{\mathcal{Y}_{1}}^{\top}*\mathcal{V}_{\mathcal{Y}_{2}}\|=\|\mathcal{V}_{\mathcal{Y}_{1}}^{\top}*\mathcal{V}_{\mathcal{Y}_{2}}\|=\|\mathcal{V}_{\mathcal{Y}_{1}}^{\top}\mathcal{V}_{\mathcal{Y}_{2}}\|.$
which means that the largest principal angle between Y1 and Y2 equals to that of these two subspaces represented in the Fourier domain. In the Fourier domain, since V
⊤
Y⊥
1
∈ C
(n−r)k×nk and VY2 ∈ C
nk×nk are block diagonal matrices, it holds that

V
⊤
Y⊥
1
$$\mathbf{\bar{\nabla\!\!\!{\mathcal{Y}}_{2}}}\!\!\!\parallel=$$
V
⊤
Y⊥
1
(1)
V
⊤
Y⊥
1
(2)
...
V
⊤
Y⊥
1
(k)

   VY2 (1) VY2 (2)
...
VY2
$$\left.\overline{{{\mathcal{V}}_{\mathbf{\lambda}\mathbf{n}^{2}}}}(k)\right\|\left|\vphantom{\frac{\mathrm{T}}{\mathrm{T}}}\right.=\operatorname*{max}_{1\leq j\leq k}\left\|\,\overline{{{\mathcal{V}}_{\mathbf{\mathcal{V}}_{1}^{\top}}}}(j)\,\overline{{{\mathcal{V}}_{\mathbf{\mathcal{V}}_{2}^{\top}}}}(j)\,\right\|\,.$$

## C. Signal Decomposition

Recall that the gradient descent iterates are defined in (3.4) as

$$\mathcal{U}_{t+1}=\mathcal{U}_{t}-\mu\nabla\ell(\mathcal{U}_{t})$$ $$=\mathcal{U}_{t}+\mu\mathcal{A}^{*}\left[y-\mathcal{A}\left(\mathcal{U}_{t}*\mathcal{U}_{t}^{\top}\right)\right]*\mathcal{U}_{t}$$ $$=\left[\mathcal{I}+\mu(\mathcal{A}^{*}\mathcal{A})\left(\mathcal{X}*\mathcal{X}^{\top}-\mathcal{U}_{t}*\mathcal{U}_{t}^{\top}\right)\right]*\mathcal{U}_{t}.$$

For the ground truth tensor X ∈ R
n×r×k, consider its tensor-column subspace VX with the corresponding basis VX ∈
R 
n×r×k. Consider the tensor VX ∗ Ut ∈ R
r×R×k with its t-SVD decomposition VX ∗ Ut = Vt ∗ Σt ∗ W⊤
t. For
Wt ∈ R
R×r×k, we denote by Wt,⊥ ∈ R
R×(n−r)×ka tensor whose tensor-column subspace is orthogonal to those of Wt,
that is ∥W⊤
t,⊥ ∗ Wt∥ = 0 and its projection operator PWt,⊥ is defined as PWt,⊥ = Wt,⊥ ∗ W⊤
t,⊥ = I − Wt ∗ W⊤
t.
We then decompose the gradient descent iterates Ut as follows
$${\mathcal{U}}_{t}={\mathcal{U}}_{t}*{\mathcal{W}}_{t}*{\mathcal{W}}_{t}^{\top}+{\mathcal{U}}_{t}*{\mathcal{W}}_{t,\perp}*{\mathcal{W}}_{t,\perp}^{\top}$$
t,⊥ (C.1)
We will refer to the tensors Ut∗Wt∗W⊤
t as the signal term of the gradient descent iterates, and the tensors Ut∗Wt,⊥∗W⊤
t,⊥
will be named as the noise term.

$$(\mathbb{C}.1)$$

Lemma C.1. *The tensor-column space of the noise term* Ut ∗ Wt,⊥ ∗ W⊤
t,⊥ is orthogonal to the tensor-column subspace of the X *, namely* V
⊤
X ∗ Ut ∗ Wt,⊥ ∗ W⊤
t,⊥ = 0*. Moreover, if* V
⊤
X ∗ Ut is full tubal-rank with all invertible singular tubes, then the signal term Ut ∗ Wt ∗ W⊤
t has tubal-rank r *with all invertible singular tubes and the noise term has tubal rank at most* R − r. Proof. V
⊤
X ∗ Ut ∗ Wt,⊥ ∗ W⊤
t,⊥ = V
⊤
X ∗ Ut ∗ (I − Wt ∗ W⊤
t) = V
⊤
X ∗ Ut − V
⊤
X ∗ Ut ∗ Wt ∗ W⊤
t = 0 ∈ R
r×R×k.

The second part follows fact that if V
⊤
X ∗ Ut is full tubal rank with all invertible singular tubes then all the slices in the Fourier have full rank.

## D. Analysis Of The Spectral Stage

The goal of this section is to show that the first few iterations of the gradient descent algorithm can be approximated by the iteration of the tensor power method modulo normalization defined as

$${\tilde{\mathcal{U}}}_{t}=\left({\mathcal{I}}+\mu{\mathcal{A}}^{*}{\mathcal{A}}({\mathcal{X}}*{\mathcal{X}}^{\top})\right)^{*t}*{\mathcal{U}}_{0}={\mathcal{Z}}_{t}*{\mathcal{U}}_{0}\in\mathbb{R}^{n\times R\times k}.$$
$=:\left(\mathbb{Z}+\mu\mathcal{A}^*\mathcal{A}(\mathcal{Z})\right)$
with the tensor power method iteration Zt =: I + µA∗A(X ∗ X
$$\left({\mathcal{X}}^{\top}\right)^{*t}\in\mathbb{R}^{n\times n}$$
n×n×k. Moreover, this will result in the
feature that after the first few iterations, the tensor-column span of the signal term Ut ∗ Wt ∗ W⊤
t becomes aligned with
the tensor-column span of X , and that the noise term Ut ∗ Wt,⊥ is relatively small compared to signal term in terms of the
norm, indicating that the signal term dominates the noise term. For this, let us denote the difference between the power method and the gradient descent iterations by

$${\mathcal{E}}_{t}:={\mathcal{U}}_{t}-{\widetilde{\mathcal{U}}}_{t}.$$
$$(\mathbf{D}.1)$$
Et := Ut − Uet. (D.1)
For convenience, throughout this section, we will denote by M the tensor M := A∗A(X ∗ X
⊤) ∈ R
n×n×k, so that Uet = (I + µM)
∗t ∗ U0 and Zt = (I + µM)
∗t.

In the first result of this section, the following lemma, we show that Et can be made small via an appropriate initialization scale. Lemma D.1. *Suppose that* A : S
n×n×k → R
m satisfies RIP(2, δ1) *and let* t
⋆ *be defined as*

$$t^{*}=\min\Big{\{}j\in\mathbb{N}\colon\|\widetilde{\mathcal{U}}_{j-1}-\mathcal{U}_{j-1}\|>\|\widetilde{\mathcal{U}}_{j-1}\|\Big{\}}.\tag{1}$$

Then for all integers t *such that* 1 ≤ t ≤ t
⋆*it holds that*

$$\|\mathbf{\mathcal{E}}_{t}\|=\|\mathbf{\mathcal{U}}_{t}-\tilde{\mathbf{\mathcal{U}}}_{t}\|\leq8(1+\delta_{1}\sqrt{k})\sqrt{k\operatorname*{min}\left\{n,R\right\}}\frac{\alpha^{3}}{\|\mathbf{\mathcal{M}}\|}\|\mathbf{\mathcal{U}}\|^{3}(1+\mu\|\mathbf{\mathcal{M}}\|)^{3t}.$$
3t. (D.3)
Proof. Similarly to the matrix case in (Stoger & Soltanolkotabi, 2021), in the tubal tensor case it can be shown that for ¨
t ≥ 1, the difference tensor Et = Ut − Uet can be represented as

$$\mathcal{E}_{t}=\mathcal{U}_{t}-\widetilde{\mathcal{U}}_{t}=\sum_{j=1}^{t}(\mathcal{I}+\mu\mathcal{M})^{*(t-j)}\widehat{\mathcal{E}}_{j}\tag{1}$$
$$(\mathrm{D.2})$$
$$(\mathrm{D.3})$$
$$(\mathbf{D}.4)$$

with Ebj = µA∗AUj−1 ∗ U
⊤ j−1
∗ Uj−1. To estimate ∥Et∥, we will first estimate each summand in (D.4) separately. First, we can proceed with the following simple estimation

$\|(\mathcal{I}+\mu\mathcal{M})^{*(t-j)}\widehat{\mathcal{E}}_{j}\|\leq\|(\mathcal{I}+\mu\mathcal{M})\|^{(t-j)}\|\widehat{\mathcal{E}}_{j}\|\leq\left(1+\mu\|\mathcal{M}\|\right)^{(t-j)}\|\widehat{\mathcal{E}}_{j}\|.$
Now, for ∥Ebj∥, using the fact that the spectral norm of tubal tensors is sub-multiplicative, we get that

$\|\widehat{\mathcal{E}}_{j}\|=\mu\|\mathcal{A}^{*}\mathcal{A}(\mathcal{U}_{j-1}*\mathcal{U}_{j-1}^{\top})*\mathcal{U}_{j-1}\|\leq\mu\|\mathcal{A}^{*}\mathcal{A}(\mathcal{U}_{j-1}*\mathcal{U}_{j-1}^{\top})\|\cdot\|\mathcal{U}_{j-1}\|$.  
13 Since operator A satisfies RIP(2, δ1), by Lemma G.3, A also satisfies S2NRIP(δ1
√k), which provides the following estimate
∥A∗AUj−1 ∗ U
⊤
j−1
∥ ≤ (1 + δ1
√
k)∥Uj−1 ∗ U
⊤
j−1∥∗ = (1 + δ1
√
k)∥Uj−1∥
2F .

All this together leads to

$$\|\mathcal{E}_{t}\|=\|\mathcal{U}_{t}-\tilde{\mathcal{U}}_{t}\|\leq\mu(1+\delta_{1}\sqrt{k})\sum_{j=1}^{t}\left(1+\mu\|\mathcal{M}\|\right)^{(t-j)}\|\mathcal{U}_{j-1}\|_{F}^{2}\|\mathcal{U}_{j-1}\|.$$

From here, we want to bound ∥Et∥ in terms of the initialization scale α and the data-related norm ∥M∥. For this, we first use the fact that the tensor Frobenius norm above can be bounded as ∥Uj−1∥F ≤pk min {*n, R*}∥Uj−1∥. Then since for all 1 ≤ j ≤ t
⋆ we have ∥Uej−1 − Uj−1*∥ ≤ ∥*Uej−1∥, the spectral norm of Uj−1 can be bounded as

$$\|{\mathcal{U}}_{j-1}\|\leq\|{\widetilde{\mathcal{U}}}_{j-1}\|+\|{\mathcal{U}}_{j-1}-{\widetilde{\mathcal{U}}}_{j-1}\|\leq2\|{\widetilde{\mathcal{U}}}_{j-1}\|.$$

This gives us the following upper bound

$$\|\mathcal{E}_{t}\|\leq8\mu(1+\delta_{1}\sqrt{k})\sqrt{k\min\left\{n,R\right\}}\sum_{j=1}^{t}(1+\mu\|\mathcal{M}\|)^{t-j}\|\tilde{\mathcal{U}}_{j-1}\|^{3}.$$ (D.6)
$$(\mathbf{D}.5)$$

As for iterations of the tensor power method, it holds that
∥Uej−1∥ = ∥(I + µM)
∗(j−1) ∗ U0*∥ ≤ ∥*(I + µM)
∗(j−1)∥∥U0∥ ≤ (1 + µ∥M∥)
j−1∥U0∥ = α(1 + µ∥M∥)
j−1∥U∥,
we can proceed with (D.6) as follows

$$\|{\mathcal{E}}_{t}\|\leq8\mu(1+\delta_{1}{\sqrt{k}}){\sqrt{k\operatorname*{min}\left\{n,R\right\}}}\alpha^{3}\|{\mathcal{U}}\|^{3}\sum_{j=1}^{t}(1+\mu\|{\mathcal{M}}\|)^{t+2j-3}.$$

Now, the sum on the right-hand side can be estimated as

$$\sum_{j=1}^{t}(1+\mu\|\mathcal{M}\|)^{t+2j-3}=(1+\mu\|\mathcal{M}\|)^{t-1}\sum_{j=1}^{t}(1+\mu\|\mathcal{M}\|)^{2j-2}=(1+\mu\|\mathcal{M}\|)^{t-1}\frac{(1+\mu\|\mathcal{M}\|)^{2t}-1}{(1+\mu\|\mathcal{M}\|)^{2}-1}$$ $$=(1+\mu\|\mathcal{M}\|)^{t-1}\frac{(1+\mu\|\mathcal{M}\|)^{2t}-1}{\mu\|\mathcal{M}\|(2+\mu\|\mathcal{M}\|)}\leq\frac{(1+\mu\|\mathcal{M}\|)^{3t}}{\mu\|\mathcal{M}\|},$$

which gives us the final estimation for the norm of Et as follows

$$\|{\mathcal{E}}_{t}\|\leq8(1+\delta_{1}{\sqrt{k}}){\sqrt{k\operatorname*{min}\left\{n,R\right\}}}{\frac{\alpha^{3}}{\|{\mathcal{M}}\|}}\|{\mathcal{U}}\|^{3}(1+\mu\|{\mathcal{M}}\|)^{3t}$$

and finishes the proof. The following lemma provides a lower bound for t
⋆, indicating the duration for which the approximation in Lemma D.1 remains valid.

Lemma D.2. *Consider tensors* M := A∗A(X ∗ X
⊤) ∈ R
n×n×k and Uet := (I + µM)
∗t ∗ U0*. Let* M ∈ C
nk×nk be the corresponding block diagonal form of the tensor M *with the leading eigenvector* v1 ∈ C
nk*, then*

$$t^{\star}\geq\left\lfloor{\frac{\ln\left({\frac{\|{\mathcal{M}}\|\cdot\|{\overline{{{\mathcal{M}}}}}_{0}^{\mathrm{-H}}v_{1}\|_{\ell_{2}}}{8(1+\delta_{1}{\sqrt{k}}){\sqrt{k\operatorname*{min}\left\{n,R\right\}}\alpha^{3}\|{\mathcal{U}}\|^{3}\right\}}}\right)}{2\ln\left(1+\mu\|{\mathcal{M}}\|\right)}}\right\rfloor$$

$$(\mathbf{D}.7)$$
(D.7)
14 Proof. Let Uet ∈ C
nk×Rk be the corresponding block diagonal form of tensor Uet. By the definition of the spectral tensor norm, we have ∥Uet∥ = ∥Uet∥ and the definition of the matrix norm gives ∥Uet∥ ≥Uet H
v1ℓ2
. For the block diagonal version of Uet, the following properties (see, e.g., (Liu et al., 2019)) holds

$$\overline{{{\cal U}}}_{t}=\overline{{{({\cal I}+\mu{\cal M})^{*t}}}}^{t}*{\cal U}_{0}=\overline{{{({\cal I}+\mu{\cal M})^{*t}}}}\cdot\overline{{{\cal U}}}_{0}=\overline{{{({\cal I}+\mu{\cal M})}}}^{t}\cdot\overline{{{\cal U}}}_{0}.$$

This allows us to proceed as follows

$$\widetilde{\overline{{{\mathcal{U}}}}}_{t}^{\mathrm{H}}v_{1}=\left(\overline{{{\left(\mathcal{I}+\mu\mathcal{M}\right)}}}^{t}\cdot\overline{{{\mathcal{U}}}}_{0}\right)^{\mathrm{H}}v_{1}=\overline{{{\mathcal{U}}}}_{0}^{\mathrm{H}}\overline{{{\left(\mathcal{I}+\mu\mathcal{M}\right)}}}^{t}v_{1}=(1+\mu\|\mathcal{M}\|)^{t}\overline{{{\mathcal{U}}}}_{0}^{\mathrm{H}}v_{1},$$

where for the last equality we used the fact that block-diagonal matrix (I + µM) has the same set of eigenvectors as matrix M. From here, we get ∥Uet∥ ≥Uet H
v1 ℓ2
= (1 + µ∥M∥)
tU0 Hv1 ℓ2
. Then, applying Lemma D.1, the relative error in the spectral norm between Uet and Ut can be estimated as

$$\frac{\|\widetilde{\mathcal{U}}_{t}-\mathcal{U}_{t}\|}{\|\widetilde{\mathcal{U}}_{t}\|}\leq8(1+\delta_{1}\sqrt{k})\frac{\sqrt{k\operatorname*{min}\left\{n,R\right\}}\alpha^{3}}{\|\mathcal{M}\|\cdot\|\overline{{{\mathcal{U}}_{0}}}^{\mathrm{H}}v_{1}\|_{\ell_{2}}}\|\mathcal{U}\|^{3}(1+\mu\|\mathcal{M}\|\|)^{2t}.$$
$$(\mathbf{D}.8)$$

Setting the bound above to be smaller than 1 and solving for t, we get

$$t<\frac{\ln\left(\frac{\|\mathcal{M}\|\cdot\|\overline{{{\mathcal{M}}_{0}}}^{\mathrm{H}}v_{1}\|}{8(1+\delta_{1}\sqrt{k})\sqrt{k\operatorname*{min}\left\{n,R\right\}\alpha^{3}\|\mathcal{M}\|^{3}}}\right)}{2\ln\left(1+\mu\|\mathcal{M}\|\right)}.$$

Since t ∈ N with t ≤ t
⋆should be such that ∥Uet−1−Ut−1∥
∥Uet−1∥
< 1, we can choose t
⋆as the floor-value of the right-hand side above.

To show that the tensor column subspaces of the tensor power method iterates and the gradient descent iterates are aligned after the alignment phase, we use the largest principal angle between two tensor-column subspaces as the potential function for analysis. Borrowing the idea from (Gleich et al., 2013), we will show that the power method iteration in the tensor domain can be transformed to the classical subspace iteration in the frequency domain.

For this, consider the power method iterates Uet = (I + µM)
∗t ∗ U0, the iterates Zt = (I + µM)
∗tand the gradient descent iterates Ut represented as Ut = Uet + Et = Zt ∗ U0 + Et. All these tensors have their counterparts in the Fourier domain, which we will denote respectively as Uet, Zt and Ut.

As before, consider M = A∗A(X ∗ X
⊤) ∈ R
n×n×k with its t-SVD M = VM ∗ ΣM ∗ W⊤M and its Fourier domain representative M ∈ C
nk×nk. We denote by L ∈ R
n×r×kthe tensor column subspace spanned by the tensor columns corresponding to the first r singular tubes, that is L := VM(:, 1 : r, :) ∈ R
n×r×k. Note that L is also the subspace spanned by the tensor columns corresponding to the first r singular tubes of the tensor Zt ∈ R
n×n×k.

By Lt ∈ R
n×n×k we will donate the tensor-column subspace spanned by the tensor columns corresponding to the first r singular tubes of the gradient descent iterates Ut = Zt ∗ U0 + Et. More concretely, for Ut =PR
s=1 VUt(:, s, :) ∗ ΣUt(*s, s,* :) ∗ W⊤
Ut
(:, s, :) and the corresponding Fourier domain representation Ut =
diag(Ut
(1),Ut
(2)*, . . . ,U*t
(k)), where Ut
(j) =Pℓ σ
(j)
ℓv
(j)
ℓ w
(j) ℓ H= U
(j)
Ut Σ
(j)
Ut W
(j)
Ut H, we define the corresponding new tensors Lt := VUt
(:, 1 : r, :) ∈ R
n×r×kand their Fourier domain representations

$${\overline{{{\mathcal{L}}}}}_{t}=\mathrm{diag}({\overline{{{L}}}}_{t}^{(1)},{\overline{{{L}}}}_{t}^{(2)},\ldots,{\overline{{{L}}}}_{t}^{(k)})$$
$${\mathcal{L}}+\mu{\mathcal{M}})^{*i}$$
(k)) (D.9)
Lemma D.3. *Consider the tensor iterates* Zt = (I + µM)

*ith its block-matrix representation* $\mathbf{r}$. 
Zt = *bdiag*(Zt) = *diag*(Zt
(1),Zt
$$\overline{{{Z}}}_{t}^{(2)},\ldots,\overline{{{Z}}}_{t}^{(k)}).$$
$\overline{\epsilon}_{\text{in}}=$ . 
(k)). (D.10)
and the tensors

$$\begin{array}{l l}{{{\mathcal E}_{t}={\mathcal U}_{t}-\widetilde{\mathcal U}_{t}\in\mathbb{R}^{n\times R\times k}}}\\ {{{\mathcal U}_{0}=\alpha{\mathcal U}\in\mathbb{R}^{n\times R\times k},}}&{{\alpha>0.}}\end{array}$$
$$(\mathrm{D.9})$$

$$(\mathrm{D.}10)$$

Assume that for each 1 ≤ j ≤ k*, it holds that*

$$\sigma_{r+1}(\overline{{{Z}}}_{t}^{(j)})\|\mathcal{U}\|+\frac{\|\mathcal{E}_{t}\|}{\alpha}<\sigma_{r}(\overline{{{Z}}}_{t}^{(j)})\sigma_{m i n}(\overline{{{\mathcal{V}}}}_{\mathcal{L}}^{\top}\!*\!\mathcal{U}).$$
L ∗ U). (D.11)
Then for each 1 ≤ j ≤ k*, the following two inequalities hold*

σrUt (j)= σrZt (j)U0 (j) + Et (j)≥ ασr(Zt (j))σmin(V ⊤ L ∗ U) − ∥Et∥, (D.12) σr+1Ut (j)= σr+1Zt (j)U0 (j) + Et (j)≤ ασr+1(Zt (j))∥U∥ + ∥Et∥ (D.13)
Moreover, the principal angle between the tensor-column subspaces L and Lt *is bounded as follows*

$$\|\mathcal{V}_{\mathcal{L}^{\perp}}^{\top}\ast\mathcal{V}_{\mathcal{L}_{t}}\|\leq\operatorname*{max}_{1\leq j\leq k}\frac{\alpha\sigma_{r+1}(\overline{{{Z}}}_{t}^{(j)})\|\mathcal{U}\|+\|\mathcal{E}_{t}\|}{\sigma_{r}(\overline{{{Z}}}_{t}^{(j)})\sigma_{m i n}(\overline{{{\mathcal{V}}}}_{\mathcal{L}}^{\dagger}\ast\mathcal{U})-\alpha\sigma_{r+1}(\overline{{{Z}}}_{t}^{(j)})\|\mathcal{U}\|-\|\mathcal{E}_{t}\|}$$
$$(\mathrm{D.14})$$

Proof. For some t ∈ N, consider tensor Zt = (I + µM)
∗t with its block-matrix representation

$$\overline{\mathbf{Z}}_{t}=\text{bdiag}(\mathbf{Z}_{t})=\text{diag}(\overline{Z}_{t}{}^{(1)},\overline{Z}_{t}{}^{(2)},\ldots,\overline{Z}_{t}{}^{(k)})=\begin{pmatrix}\overline{Z}_{t}{}^{(1)}\\ \\ \\ \end{pmatrix}$$  The symmetric tensor case scenario, the block-diagonal matrix represent 
...

$$\overline{{{Z_{t}}}}^{(k)}\Big)$$
$$(\mathbb{D}.11)$$
$$(\mathrm{D.12})$$ $$(\mathrm{D.13})$$

.
As we assume the symmetric tensor case scenario, the block-diagonal matrix representation Zt consists of symmetric matrices Zt
(j) ∈ C
n×n. At the same time, according to (Gleich et al., 2013), the gradient descent tensors Ut = Zt ∗U0+Et

have their block-diagonal matrix representation  $$\mathcal{U}_{t}=\mathcal{Z}_{t}\ast\mathcal{U}_{0}+\mathcal{E}_{t}\ \Leftrightarrow\ \overline{\mathcal{Z}_{t}\mathcal{U}_{0}}+\overline{\mathcal{E}_{t}}=\left(\begin{array}{cccc}\overline{\mathcal{Z}}_{1}^{(1)}\overline{\mathcal{U}}_{0}^{(1)}&&&\\ &\overline{\mathcal{Z}}_{1}^{(2)}\overline{\mathcal{U}}_{0}^{(2)}&&\\ &&&\ddots&\\ &&&\overline{\mathcal{Z}_{t}^{(k)}\mathcal{U}_{0}^{(k)}}\end{array}\right)+\left(\begin{array}{cccc}\overline{\mathcal{E}}_{1}^{(1)}&&&\\ &\overline{\mathcal{E}}_{1}^{(2)}&&\\ &&\ddots&\\ &&&\overline{\mathcal{E}_{t}^{(k)}}\end{array}\right).\tag{15}$$
Using Weyl's inequality in each block, we have

$$\sigma_{r}\big{(}\overline{{{Z}}}_{t}^{(j)}\overline{{{U}}}_{0}^{(j)}+\overline{{{E}}}_{t}^{(j)}\big{)}\geq\sigma_{r}\big{(}\overline{{{Z}}}_{t}^{(j)}\overline{{{U}}}_{0}^{(j)}\big{)}-\|\overline{{{E}}}_{t}^{(j)}\|\geq\sigma_{r}\Big{(}\overline{{{V}}}_{\mathbf{\mathcal{L}}}^{(j)}\big{)}^{\mathrm{H}}\overline{{{Z}}}_{t}^{(j)}\overline{{{U}}}_{0}^{(j)}\Big{)}-\|\overline{{{E}}}_{t}^{(j)}\|.$$

Now, for the singular value above we get the following estimation

σr (VL (j)) HZt (j)U0 (j)= σminVL (j)HZt (j)V (j) L V (j) L HU0 (j) ≥ σminVL (j)HZt (j)VL (j)σminVL (j)HU0 (j) = σr(Zt (j))σminVL (j)HU0 (j)≥ ασr(Zt (j))σminVL (j)HU (j) = ασr(Zt (j))σminV H L (j)U (j)≥ ασr(Zt (j))σminV ⊤L ∗ U 
where in the last line we used that for each tensor it holds in the Fourier domain VL
(j)H = V
T L
(j).

To show inequality (D.13), we can use Weyl's bounds and then the Courant-Fisher theorem, which leads to

any $\delta\nu$, we can use $\nu_{j}$ to obtain an order in Count I must be seen that  $$\sigma_{r+1}\big{(}\overline{Z}_{t}^{(j)}\overline{U}_{0}^{(j)}+\overline{E}_{t}^{(j)}\big{)}\leq\sigma_{r+1}\big{(}\overline{Z}_{t}^{(j)}\overline{U}_{0}^{(j)}\big{)}+\|\overline{E}_{t}^{(j)}\|\leq\sigma_{r+1}\big{(}\overline{Z}_{t}^{(j)}\overline{U}_{0}^{(j)}\big{)}+\|\mathcal{E}_{t}\|$$ $$\leq\sigma_{r+1}\big{(}\overline{Z}_{t}^{(j)}\|\overline{U}_{0}^{(j)}\|+\|\mathcal{E}_{t}\|\leq\alpha\sigma_{r+1}\big{(}\overline{Z}_{t}^{(j)}\big{)}\|\boldsymbol{\mathcal{U}}\|+\|\mathcal{E}_{t}\|.$$
Now, for estimation of ∥V
⊥
L ∗ VLt ∥, let us recall that L is the tensor column subspace spanned by the tensor columns corresponding to the first r singular tubes of tensor Zt = (I − µM)
∗t ∈ R
n×n×k, and Lt is the tensor-column subspace spanned by the tensor-columns corresponding to the first r singular tubes of the gradient descent iterates Ut = Zt ∗U0 +Et, and consider Fourier-domain representation (D.15) of Ut. Here, for each 1 ≤ j ≤ k, the matrices Zt
(j)U0
(j) + Et
(j)can be represented as

$$\underbrace{\overline{Z}_{\epsilon}^{(i)}\overline{U}_{0}^{(i)}+\overline{E}_{t}^{(i)}}_{\tilde{A}^{(i)}}=\underbrace{\overline{Z}_{\epsilon}^{(i)}\overline{V}_{\epsilon}^{(i)}\overline{V}_{\epsilon}^{(i)}\overline{U}_{0}^{(i)}}_{A^{(i)}}+\underbrace{\overline{Z}_{\epsilon}^{(i)}\overline{V}_{\epsilon}^{(i)}\overline{V}_{\epsilon}^{(i)}\overline{U}_{0}^{(i)}+\overline{E}_{t}^{(i)}}_{C^{(i)}}.$$ (D.16)
$$(\mathrm{D.17})$$

As the tensor-column space VL is r-dimensional, each of matrices VL
(j) has rank r, see (Gleich et al., 2013). Since the matrices Zt
(j)can be decomposed as

Zt
$$\overline{{{r}}}_{t}^{(j)}=\overline{{{V_{\mathcal{L}}}}}^{(j)}\Sigma_{\mathcal{L}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}{}^{\mathrm{H}}+\overline{{{V_{\mathcal{L}}}}}^{\perp}{}^{(j)}\Sigma_{\mathcal{L}^{\perp}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}{}^{\mathrm{H}}$$
we have that
$$\overline{{{{Z}_{t}}}}^{(j)}\overline{{{{V}_{{\cal L}}}}}^{(j)}\overline{{{{V}_{{\cal L}}}}}^{(j)}{}^{\mathrm{H}}\overline{{{{U}_{0}}}}^{(j)}=\overline{{{{V}_{{\cal L}}}}}^{(j)}\Sigma_{{\cal L}}^{(j)}\overline{{{{V}_{{\cal L}}}}}^{(j)}{}^{\mathrm{H}}\overline{{{{U}_{0}}}}^{(j)}.$$
As U0
(j) ∈ C
r×R has rank r, VL
(j)HU0
(j) has rank r, which means that the product above has rank r too. Due to (D.17),

we see that
$$\overline{{{Z}}}_{t}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}{}^{\mathrm{H}}\overline{{{U_{0}}}}^{(j)}=\overline{{{V_{\mathcal{L}}}}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}{}^{\mathrm{H}}\overline{{{Z_{t}}}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}{}^{\mathrm{H}}\overline{{{U_{0}}}}^{(j)},$$
which makes VL
(j)to the column subspace of Zt
(j)VL
(j)VL
(j)HU0
(j). Considering the gap between the singular values of for matrices A(j)and Ae(j)in (D.16), namely δ
(j) = σr(A(j)) − σr+1(Ae(j)), and using Wedin's sin θ theorem (Wedin, 1972), for each 1 ≤ j ≤ k we get

$$\|\overline{{{V_{{\mathcal{L}}_{\perp}^{(j)}}}}}^{\mathrm{H}}\overline{{{V_{{\mathcal{L}}_{t}}}}}^{(j)}\|\leq\frac{\|C^{(j)}\|}{\delta^{(j)}}.$$

To conduct a further estimation of ∥VL⊥
(j)HVLt
(j)∥, we analyze lower and upper bounds for the denominator and the numerator above. We start with the denominator first

$$\begin{array}{l}{{\delta^{(j)}=\sigma_{r}(A^{(j)})-\sigma_{r+1}(\widetilde{A}^{(j)})}}\\ {{\qquad=\sigma_{r}(\overline{{{Z}}}_{t}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}\overline{{{V_{\mathcal{L}}}}}^{(j)}{}^{\mathrm{H}}\overline{{{U_{0}}}}^{(j)})-\sigma_{r+1}(\overline{{{Z}}}_{t}^{(j)}\overline{{{U_{0}}}}^{(j)}+\overline{{{E}}}_{t}^{(j)}).}}\end{array}$$

Using properties of singular values of the matrix product for the first term above and Weyl's bound for the second term, we get

$$\delta^{(j)}\geq\sigma_{r}(\overline{Z}_{t}^{(j)})\sigma_{min}\left(\overline{V}_{e}^{(j)}{}^{\mathrm{H}}\overline{U}_{0}^{(j)}\right)-\sigma_{r+1}\left(\overline{Z}_{t}^{(j)}\overline{U}_{0}^{(j)}\right)-\|\overline{E}_{t}^{(j)}\|\right.$$ $$\geq\sigma_{r}(\overline{Z}_{t}^{(j)})\sigma_{min}\left(\overline{V}_{e}^{\perp}\star\mathbf{\mathcal{U}}_{0}\right)-\sigma_{r+1}\left(\overline{Z}_{t}^{(j)}\overline{U}_{0}^{(j)}\right)-\|\mathbf{\mathcal{E}}_{t}\|.$$

For the norm of C
(j), the following upper bound can be established

$$\|C^{(j)}\|\leq\|\overline{Z_{t}}^{(j)}\overline{V_{\mathcal{L}^{-1}}}^{(j)}\overline{V_{\mathcal{L}^{-1}}}^{(j)}{}^{\mathrm{H}}\overline{U}_{0}^{(j)}\|+\|\overline{E}_{t}^{(j)}\|$$ $$\leq\|\overline{Z_{t}}^{(j)}\overline{V_{\mathcal{L}^{-1}}}^{(j)}\overline{V_{\mathcal{L}^{-1}}}^{(j)}{}^{\mathrm{H}}\|\|\overline{U}_{0}^{(j)}\|+\|\mathcal{E}_{t}\|$$ $$\leq\alpha\sigma_{r+1}(\overline{Z_{t}}^{(j)})\|\boldsymbol{\mathcal{U}}\|+\|\mathcal{E}_{t}\|$$
$$(\mathbb{D}.18)$$

Now, combining bounds (D.18) and (D.19), one obtains that

∥V ⊤ L⊥ ∗ VLt∥ = max 1≤j≤k ∥VL⊥ (j)HVLt (j)∥ ≤ max 1≤j≤k ασr+1(Zt (j))∥U∥ + ∥Et∥ σr(Zt (j))σminV ⊤L ∗ U− σr+1Zt (j)U(j)− ∥Et∥ :
Using in the denominator the fact that σr+1Zt
(j)U0
(j)≤ ασr+1Zt
(j)∥U
(j)∥ ≤ ασr+1Zt
(j))∥U∥ finishes the proof of this lemma.

$$(\mathrm{D.19})$$

Further, we consider the gradient descent iterates with its t-SVD

$$\mathcal{U}_{t}=\sum_{s=1}^{R}\mathcal{V}_{\mathcal{U}_{t}}(:,s,:)*\Sigma_{\mathcal{U}_{t}}(s,s,:)*\mathcal{W}_{\mathcal{U}_{t}}^{\mathsf{T}}(:,s,:)$$  Fourier domain representation $\overline{\mathcal{U}}_{t}=\operatorname{diag}(\overline{U}_{t}{}^{(1)},\overline{U}_{t}{}^{(2)},\ldots,\overline{U}_{t}{}^{(k)})$, $\mathcal{U}_{t}{}^{(i)}=\mathcal{U}_{t}{}^{(i)}$
and the corresponding Fourier domain representation Ut = diag(Ut
(k)), where
Ut
(j) =PR
ℓ=1 σ
(j)
ℓv
(j)
ℓ w
(j) ℓ H= V
(j)
Ut Σ
(j)
Ut W
(j)H
Utand its signal-noise term decomposition

$\mathcal{U}_{t}=\mathcal{U}_{t}*\mathcal{W}_{t}*\mathcal{W}_{t}^{\top}+\mathcal{U}_{t}*\mathcal{W}_{t,\perp}*\mathcal{W}_{t,\perp}^{\top}$.  
We also define the corresponding new tensors

$$\mathcal{L}_{t}=\sum_{s=1}^{r}\mathcal{V}_{\mathcal{U}_{t}}(:,s,:)*\Sigma_{\mathcal{U}_{t}}(s,s,:)*\mathcal{W}_{\mathcal{L}_{t}}^{\top}(:,s,:)$$ $$\mathcal{N}_{t}=\sum_{s=r+1}^{R}\mathcal{V}_{\mathcal{U}_{t}}(:,s,:)*\Sigma_{\mathcal{U}_{t}}(s,s,:)*\mathcal{W}_{\mathcal{U}_{t}}^{\top}(:,s,:)$$

and their Fourier domain representations

Lt = diag(Lt (1),Lt (2), . . . , Lt (k)), Lt (j) =X r ℓ=1 σ (j) ℓv (j) ℓ w (j) ℓ H= V (j) Lt Σ (j) LtW (j)H Lt(D.22) Nt = diag(Nt (1), Nt (2), . . . , Nt (k)), Nt (j) =X R ℓ=r+1 σ (j) ℓv (j) ℓ w (j) ℓ H= V (j) Nt Σ (j) NtW (j)H
$$(\mathrm{D.20})$$
$$(\mathrm{D.21})$$
$$(\mathbf{D}.22)$$  $$(\mathbf{D}.23)$$
$$(\mathrm{D}.24)$$
Nt(D.23)
Lemma D.4. *Assume* ∥V
⊤
X ⊥ ∗ VLt ∥ ≤ 12
. Then it holds that

$$\|\mathcal{W}_{\mathcal{L}_{t}^{\perp}}^{\top}*\mathcal{W}_{t}\|\leq2\max_{1\leq j\leq k}\frac{\sigma_{r+1}\left(\overline{U_{t}}^{(j)}\right)}{\sigma_{r}\left(\overline{U_{t}}^{(j)}\right)}\|\mathcal{V}_{\mathcal{X}^{\perp}}^{\top}*\mathcal{V}_{\mathcal{L}_{t}}\|.\tag{1}$$

Proof. Consider ∥WTL⊥
t
∗ Wt∥ = max1≤j≤k ∥WL⊥
t
(j)HWt
(j)∥. For each 1 ≤ j ≤ k, we can now exploit the results of Lemma A.1 in (Stoger & Soltanolkotabi, 2021), to get that ¨

$$\|(\overline{{{W}}}_{\widehat{{\cal L}_{t}^{+}}}^{\overline{{{\cal L}}}})^{(j)}\overline{{{W}}}_{t}^{(j)}\|\leq\frac{\|\Sigma_{{\cal M}}^{(j)}\|\|\overline{{{V}}}_{\overline{{{\cal M}}}_{t}^{(j)}}^{\overline{{{\cal M}}}}(j)\overline{{{V}}}_{\overline{{{\cal M}}}}^{(j)}\|}{\sigma_{m i n}\Big(\overline{{{V}}}_{\overline{{{\cal M}}}^{(j)}}\overline{{{U}}}_{t}^{(j)}\Big)}\quad\mathrm{and}\quad\sigma_{m i n}(\overline{{{V}}}_{\overline{{{\cal M}}}^{(j)}}^{(j)}\overline{{{U}}}_{t}^{(j)})\geq\frac{\sigma_{m i n}(\overline{{{L}}}_{t}^{(j)})}{2}.$$

From here, we can proceed as follows

∥W⊤L⊥ t ∗ Wt∥ = max 1≤j≤k ∥WH L⊥ t (j)Wt (j)∥ ≤ 2 max 1≤j≤k ∥Σ (j) Nt ∥∥V H Nt (j)VX (j)∥ σmin(Lt (j)) = 2 max 1≤j≤k σr+1(Ut (j))∥V H Nt (j)VX (j)∥ σr(Ut (j)) ≤ 2 max 1≤j≤k σr+1(Ut (j)) σr(Ut (j)) ∥V ⊤ L⊥ t ∗ VX ∥ = 2 max 1≤j≤k σr+1Ut (j) σrUt (j) ∥V ⊤ X ⊥ ∗ VLt∥,
which concludes the proof. Lemma D.5. *Assume that* ∥V
⊤
X ⊥ ∗ VLt∥ ≤ 18 for some t ≥ 1, t ∈ N. Then for each 1 ≤ j ≤ k*, it holds that*

$$\sigma_{r}\Big{(}\overline{\mathbf{\mathcal{U}}_{t}*\mathbf{\mathcal{W}}_{t}}^{(j)}\Big{)}\geq\frac{1}{2}\sigma_{r}\Big{(}\overline{\mathbf{\mathcal{U}}_{t}}^{(j)}\Big{)}$$ (D.25) $$\sigma_{1}(\overline{\mathbf{\mathcal{U}}}_{t}*\mathbf{\mathcal{W}}_{t,\perp}^{(j)})\leq2\sigma_{r+1}(\overline{\mathbf{\mathcal{U}}}_{t}^{(j)}).$$ (D.26)
Moreover, the principal angles between the tensor-column subspaces spanned by X and UtWt *can be estimated as follows*

$$\|\mathcal{V}_{\mathcal{X}^{\perp}}*\mathcal{V}_{\mathcal{U}_{t},\mathcal{W}_{t}}\|\leq7\|\mathcal{V}_{\mathcal{X}^{\perp}}^{\top}*\mathcal{V}_{\mathcal{L}_{t}}\|$$ $$\|\mathcal{U}_{t}*\mathcal{W}_{t,\perp}\|\leq2\max_{1\leq j\leq k}\sigma_{r+1}(\overline{U}_{t}^{(j)}).\tag{1}$$

Proof. We assume that ∥V
⊤
X ⊥ ∗ VLt∥ ≤ 
1 8
, then due to Lemma D.4, we obtain that

$$\|\mathcal{W}_{\mathcal{L}_{t}^{+}}^{\top}*\mathcal{W}_{t}\|\leq2\max_{1\leq j\leq k}\frac{\sigma_{r+1}\left(\overline{U_{j}}^{(j)}\right)}{\sigma_{r}\left(\overline{U_{j}}^{(j)}\right)}\|\mathcal{V}_{\mathcal{X}^{\perp}}^{\top}*\mathcal{V}_{\mathcal{L}_{t}}\|\leq\frac{1}{4}.\tag{1}$$
$$\begin{array}{l}{{\mathrm{(D.27)}}}\\ {{}}\end{array}$$ (D.28)
$$(\mathrm{D.29})$$
$$(\mathbb{D}.30)$$

Now, to estimate σr Ut ∗ Wt
(j), we see that for each 1 ≤ j ≤ k, it holds that

$$\sigma_{r}\Big(\overline{{{{\cal U}_{t}*{\cal W}}}}_{t}^{(j)}\Big)^{2}=\sigma_{r}\Big(\big(\overline{{{{\cal U}_{t}*{\cal W}}}}_{t}^{(j)}\big)^{\mathrm{H}}\overline{{{{\cal U}_{t}*{\cal W}}}}_{t}^{(j)}\Big)=\sigma_{r}\Big(\overline{{{{\cal W}_{t}}}}^{(i)\mathrm{H}}\overline{{{{\cal U}_{t}}}}^{(j)\mathrm{H}}\overline{{{{\cal U}_{t}}}}^{(j)}\Big)^{\mathrm{H}}\overline{{{{\cal W}_{t}}}}^{(j)}\Big)$$
(j)(D.30)
Since Ut
(j)HUt
(j) = Lt
(j)HLt
(j) + Nt
(j)HNt
(j), we get that

σr Ut ∗ Wt (j)2≥ σr Wt (j)HLt (j)HLt (j)Wt (j)= σr Wt (j)HLt (j)2 ≥ σr Wt (j)HWLt (j)2σr Lt (j)2≥ (1 − ∥WL⊥ t ∗ WT t ∥ 2σr Ut (j)2,
where in the last line we used the definition of the principal angle between tensor column subspaces and the corresponding properties in their Fourier domain slices, namely

$$\sigma_{r}\big{(}\overline{W_{t}}^{(j)^{\rm H}}W_{\overline{L}_{t}^{(j)}}\big{)}^{2}=1-\|\overline{W_{t}}^{(j)^{\rm H}}W_{\overline{L}_{t}^{(j)}}^{\perp}\|^{2}\geq1-\max_{1\leq j\leq k}\|\overline{W_{t}}^{(j)^{\rm H}}W_{\overline{L}_{t}^{(j)}}^{\perp}\|^{2}=1-\|\mathbf{\mathcal{W}}_{\mathbf{\mathcal{E}}_{t}^{+}}\ \mathbf{*}\ \mathbf{\mathcal{W}_{t}^{T}}^{T}\|^{2}.$$

Due to our assumption ∥V
⊤
X ⊥ ∗ VLt∥ ≤ 
1 8
, we can see that in the Fourier domain, the subspaces spanned by V
(j)
X ⊥
t and V
(j)
Lt
= VLt
(j) are close enough. Then, decomposing Ut
(j)into two different ways, namely as

$$\overline{{{U}}}_{t}^{(j)}=\sum_{\ell=1}^{R}\sigma_{\ell}^{(j)}v_{\ell}^{(j)}w_{\ell}^{(j)}{}^{\mathrm{H}}=\overline{{{L}}}_{t}^{(j)}+\overline{{{N}}}_{t}^{(j)}$$
and as
$$\overline{{{U}}}_{t}^{(j)}=\overline{{{U}}}_{t}^{(j)}\overline{{{W}}}_{t}^{(j)}\overline{{{W}}}_{t}^{(j)}{}^{\mathrm{H}}+\overline{{{U}}}_{t}^{(j)}\overline{{{W}}}_{t,\perp}{}^{(j)}\overline{{{W}}}_{t,\perp}{}^{(j)}{}^{\mathrm{H}},$$

according to Lemma H.1, one obtains for each 1 ≤ j ≤ k that

$$\begin{array}{c}{{\|\overline{{{V}}}_{\mathbf{\mathcal{X}}_{t}^{(j)}}^{(j)}\stackrel{\mathrm{H}}{V}\overline{{{U_{t}}}}_{(j)}\|\leq7\|\overline{{{V}}}_{\mathbf{\mathcal{X}}_{t}^{(j)}}^{(j)}\overline{{{V}}}_{\mathbf{\mathcal{L}}_{t}^{(j)}}^{(j)}\|}}\\ {{\|\overline{{{U}}}_{t}^{(j)}\overline{{{W_{t,\perp}}}}^{(j)}\|\leq2\sigma_{r+1}(\overline{{{U}}}_{t}^{(j)}),}}\end{array}$$

where the last inequality is equivalent to σ1(Ut ∗ Wt,⊥
(j)) ≤ 2σr+1(Ut
(j)). According to the definition of principal angles between tensor subspaces, this implies that

$\|\mathbf{\mathcal{V}}_{\mathbf{\mathcal{X}}^{\perp}}^{\top}*\mathcal{V}_{\mathbf{\mathcal{U}}_{t}*\mathbf{\mathcal{W}}_{t}}\|=\max_{j}\|\overline{\mathcal{V}}_{\mathbf{\mathcal{X}}_{t}^{j}}^{(j)}\|^{\mathrm{H}}V_{\overline{\mathcal{U}}_{t}(i)\,\overline{W}_{t}(j)}\|\leq7\max_{j}\|\overline{\mathcal{V}}_{\mathbf{\mathcal{X}}_{t}^{j}}^{(j)}\|\overline{\mathcal{V}}_{\mathbf{\mathcal{L}}_{t}}^{\mathrm{H}}\|=7\|\mathbf{\mathcal{V}}_{\mathbf{\mathcal{X}}^{\perp}}^{\top}*\mathcal{V}_{\mathbf{\mathcal{L}}_{t}}\|.$
In the same way, ∥Ut ∗ Wt,⊥∥ = maxj ∥Ut
(j)Wt,⊥
(j)∥ ≤ 2 maxj σr+1(Ut
(j)), which finishes the proof.

Lemma D.6. *Consider a tensor* T := X ∗ X
⊤ ∈ S
n×n×k
+ with tubal rank r ≤ n. Assume that measurement operator A is such that

$$\begin{array}{r l}{{\mathcal{M}}={\mathcal{A}}^{*}{\mathcal{A}}({\mathcal{T}})={\mathcal{T}}+{\mathcal{E}}}&{{}\in S_{+}^{n\times n\times k}}\end{array}$$

19 and for for each 1 ≤ j ≤ k one has ∥E(j)∥ ≤ δλr(T
(j)) *with* δ ≤14
. For the same M with its t-SVD M =
VM ∗ ΣM ∗ W⊤M*, let* L ∈ R
n×r×k *denote the tensor column subspace spanned by the tensor-columns corresponding to* the first r *singular tubes, that is* L := VM(:, 1 : r, :) ∈ R
n×r×k.

Then, in each Fourier slice j, 1 ≤ j ≤ k*, it holds that*

$\lambda\leq\lambda_{1}$, $\lambda$ holds and  $$(1-\delta)\lambda_{1}(\overline{T}^{(j)})\leq\lambda_{1}(\overline{M}^{(j)})\leq(1+\delta)\lambda_{1}(\overline{T}^{(j)})$$ $$\lambda_{r+1}(\overline{M}^{(j)})\leq\delta\lambda_{r}(\overline{T}^{(j)})$$ $$\lambda_{r}(\overline{M}^{(j)})\geq(1-\delta)\lambda_{r}(\overline{T}^{(j)}),$$
$$(1-\delta)\|{\mathcal{T}}\|\leq\|{\mathcal{M}}\|\leq(1+\delta)\|{\mathcal{T}}\|$$
and
(1 − δ)∥T ∥ ≤ ∥M∥ ≤ (1 + δ)∥T ∥ (D.34)
Moreover, the tensor-column subspaces of X and L *are aligned, namely*

 (D.31)  (D.32)  (D.33)
$$(\mathrm{D.34})$$
$$\|{\mathcal{V}}_{{\mathcal{X}}^{\perp}}^{\top}*{\mathcal{V}}{\mathcal{L}}\|\leq2\delta$$
$$(\mathbf{D}.35)$$
$$(\mathrm{D.36})$$
X ⊥ ∗ VL∥ ≤ 2δ (D.35)
Proof. Consider tensor T := X ∗ X
⊤ ∈ S
n×n×k
+ . Due to the definition of tensor transpose and conjugate symmetry of Fourier coefficients (Kilmer & Martin, 2011), the Fourier slices of T are defined as T
(j) = X(j)X(j)H. That is, each face of T is Hermitian and at least positive semidefinite. As we assume that for each j, 1 ≤ j ≤ k, one has ∥Et
(j)∥ ≤ δλr(T
(j))
using Weyl's inequality in each of the Fourier slices, we obtain the first three inequalities.

To show that the tensor subspace VX and VL are aligned, we use first the definition

$$\|\mathcal{V}_{\mathcal{X}^{\perp}}^{\top}*\mathcal{V}_{\mathcal{L}}\|=\max_{1\leq j\leq k}\|\overline{V}_{\mathcal{X}^{\perp}}^{(j)}\overline{V}_{\mathcal{L}}^{(j)}\|\tag{1}$$

For the estimation of ∥V
H
X ⊥
(j)V
(j)
L ∥ in each of the Fourier slices, we apply Wedin's sin Θ theorem. For this, denote L := VM(:, 1 : r, :) ∈ R
n×r×kand let V
(j)
L denote the corresponding Fourier slices of L ∈ R
n×r×k. Since in the Fourier space, it holds that M(j) = T
(j) +E(j)and V
(j)
L encompasses the first r eigenvectors of M(j), from Wedin's sin Θ theorem, we obtain

$$\|\overline{{{V_{\mathbf{x}^{\perp}}^{(j)}}}}^{\mathrm{H}}\overline{{{V}}}_{\mathbf{\mathcal{L}}}^{(j)}\|\leq\frac{\|\overline{{{E}}}^{(j)}\|}{\xi^{(j)}},$$

with ξ
(j):= λr(T
(j)) − λr+1(M(j)). Using estimate (D.32), ξ
(j)can be lower-bounded as

$$\xi^{(j)}:=\lambda_{r}(\overline{{{T}}}^{(j)})-\lambda_{r+1}(\overline{{{M}}}^{(j)})\geq\lambda_{r}(\overline{{{T}}}^{(j)})-\delta\lambda_{r}(\overline{{{T}}}^{(j)})=(1-\delta)\lambda_{r}$$
(j)).
Using the bound the the assumptions that ∥Et
(j)∥ ≤ δλr(T
(j)) and δ ≤
1 2
, we get

$$\|\overline{{{V_{\pm}^{(j)}}}}^{\mathrm{H}}\overline{{{V}}}_{\pm}^{(j)}\|\leq\frac{\delta}{1-\delta}\leq2\delta.$$

Coming back to equality (D.36), we obtain the stated bound for the principal angle between the two tensor column subspaces. Lemma D.7. *Consider a tensor* X ∗ X
⊤ ∈ S
n×n×k
+ with tubal rank r ≤ n. Assume that measurement operator A *is such* that

$${\mathcal{M}}={\mathcal{A}}^{*}{\mathcal{A}}({\mathcal{X}}*{\mathcal{X}}^{\top})={\mathcal{X}}*{\mathcal{X}}^{\top}+{\mathcal{E}}$$

and for each, j, 1 ≤ j ≤ k, one has ∥E(j)∥ ≤ δλr(X(j)X(j)H) with δ ≤ c1*. Moreover, assume that for difference tensor* Et = Ut − Uet *it holds that*

$$\gamma:=\frac{\alpha\max_{1\leq j\leq k}\sigma_{r+1}(\overline{Z}_{t}^{(j)})\|\mathbf{\mathcal{U}}\|+\|\mathbf{\mathcal{E}}_{t}\|}{\min_{1\leq j\leq k}\sigma_{r}(\overline{Z}_{t}^{(j)})}\frac{1}{\alpha\sigma_{min}(\overline{\mathcal{V}_{\mathcal{L}}^{2}*\mathbf{\mathcal{U}}})}\leq c_{2}\kappa^{-2},$$ (D.37)