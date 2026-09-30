# Implicit Regularization for Tubal Tensor Factorizations via Gradient Descent

Santhosh Karnik \* 1 Anna Veselovska \* 2 3 Mark Iwen 4 5 Felix Krahmer 2 3

## Abstract

We provide a rigorous analysis of implicit regularization in an overparametrized tensor factorization problem beyond the lazy training regime. For matrix factorization problems, this phenomenon has been studied in a number of works. A particular challenge has been to design universal initialization strategies which provably lead to implicit regularization in gradient-descent methods. At the same time, it has been argued by [\(Cohen](#page-8-0) [et al., 2016\)](#page-8-0) that more general classes of neural networks can be captured by considering tensor factorizations. However, in the tensor case, implicit regularization has only been rigorously established for gradient flow or in the lazy training regime. In this paper, we prove the first tensor result of its kind for gradient descent rather than gradient flow. We focus on the tubal tensor product and the associated notion of low tubal rank, encouraged by the relevance of this model for image data. We establish that gradient descent in an overparametrized tensor factorization model with a small random initialization exhibits an implicit bias towards solutions of low tubal rank. Our theoretical findings are illustrated in an extensive set of numerical simulations show-casing the dynamics predicted by our theory as well as the crucial role of using a small random initialization.

## 1. Introduction

Analyzing implicit regularization during Neural Network (NN) training is considered crucial for understanding why overparametrization can give rise to superior generalization capability and lead to strong overall NN performance. Consequently, there has been a recent surge in research aimed at explaining how gradient-based methods interact with overparameterized models under nonconvex losses (see, e.g., [\(Ma et al., 2018;](#page-9-0) [Ling & Strohmer, 2019\)](#page-9-1)). Notably, recent empirical and theoretical studies have suggested that gradient-based methods with small random initializations exhibit a bias towards low-rank solutions in a variety of models.

For matrix factorization models which represent linear neural networks, a rigorous analysis of implicit bias is available for both gradient descent [\(Gunasekar et al., 2018;](#page-9-2) [Stoger &](#page-10-0) ¨ [Soltanolkotabi, 2021\)](#page-10-0) and gradient flow (its asymptotic limit for small step size) [\(Bah et al., 2022;](#page-8-1) [Chou et al., 2024\)](#page-8-2). In contrast, for neural networks with nonlinear activation, there has been a good deal of work done showing that fully connected layers can be represented by, e.g., tensor train factorizations in [\(Novikov et al., 2015;](#page-9-3) [Razin et al., 2021\)](#page-9-4). As a consequence, it has been argued that tensor factorizations should be considered instead of matrix factorizations (see, e.g., [\(Cohen et al., 2016\)](#page-8-0)). For tensor factorization models, however, results predating 2024 were only available for the asymptotic regime, i.e., gradient flow. This is perhaps due to the many additional complications in the tensor setting beyond those in the matrix setting including, e.g, that there are many different valid notions of tensor rank, each of which motivates its own equally valid class of tensor factorizations. For gradient descent applied to the tensor recovery problem, only a very recent partial analysis by [\(Liu et al., 2024\)](#page-9-5) currently exists for the tubal factorization model. This analysis requires that the initialization already well approximates the solution, only after which the convergence of gradient descent toward a low tubal-rank solution is shown. Herein we also focus on the tubal factorization, but establish the corresponding implicit regularization result without needing such a strong initialization assumption.

Our work is motivated by recent research showing that the way neural networks are trained, especially with gradient descent, can lead to solutions with useful structure, even without adding explicit regularization terms. This phenomenon, known as implicit regularization, has been studied in contexts such as sparse recovery [\(Vaskevicius et al., 2019\)](#page-10-1) and low-rank matrix completion [\(Li et al., 2020\)](#page-9-6), where specific

<sup>\*</sup>Equal contribution <sup>1</sup>Department of Mathematics, Northeastern University, Boston, USA <sup>2</sup>Department of Mathematics and Munich Data Science Institute, Technical University of Munich, Munich, Germany <sup>3</sup>Munich Center for Machine Learning, Munich, Germany <sup>4</sup>Department of Mathematics, Michigan State University, East Lansing, USA <sup>5</sup>Department of Computational Mathematics Science and Engineering, Michigan State University, East Lansing, USA. Correspondence to: Anna Veselovska <anna.veselovska@tum.de>.

network architectures are designed to encourage certain types of structure in the solutions. However, for tensor recovery problems, most existing work either focuses only on gradient flow or provides only partial analysis. To the best of our knowledge, our paper is the first to analyze implicit bias under gradient descent with small random initialization for a tensor recovery problem. We focus on the tubal rank model, which is particularly relevant for applications like video representation. This opens the door to a broader investigation into how implicit regularization can be used for structured tensor recovery, how network architectures influence this bias, and what conditions ensure convergence. We see this work as a starting point for a larger line of research on implicit regularization in tensor problems.

Related work: In deep learning it is common to use more network parameters than training points. In such overparameterized scenarios there are usually many networks that achieve zero training error so that the training algorithm effectively imposes an implicit regularization (bias) on the solution it computes. In practice, training networks with gradient descent is both common and tends to favor solutions that generalize well, offering the exploration of how gradient descent implicitly regularlizes in overparameterized regimes as one avenue for better understanding the success of deep learning more widely. As a result, a lot of recent work has been focussed on understanding the implicit regularization phenomena of gradient descent in multiple settings. The first theoretical works in this direction [\(Gunasekar et al.,](#page-9-7) [2017;](#page-9-7) [2018;](#page-9-2) [Geyer et al., 2020;](#page-8-3) [Arora et al., 2019;](#page-8-4) [Soudry](#page-10-2) [et al., 2018\)](#page-10-2) concentrated on training linear networks and suggested that during training (stochastic) gradient descent implicitly converges to a linear network (i.e., a linear function described by a matrix) that's low rank. Motivated by specific deep learning tasks, multiple works also investigated implicit bias phenomena in the special cases of sparse vector and low-rank matrix recovery from underdetermined measurements via an overparameterized square loss functional, where the vectors and matrices to be reconstructed were deeply factorized into several vector/matrix factors. In this setting, these works then showed that the dynamics of vanilla gradient descent are biased towards sparse/low-rank solutions, respectively [\(Chou et al., 2024;](#page-8-2) [2023;](#page-8-5) [Li et al.,](#page-9-8) [2022;](#page-9-8) [Kolb et al., 2023\)](#page-9-9).

In the realm of optimization, a substantial body of work has also emerged that provides guarantees for gradient descent's convergence in the nonconvex setting for different problems such as phase retrieval, matrix completion, and blind deconvolution. Broadly, these findings can be categorized into two main approaches: smart initialization coupled with local convergence (demonstrating, e.g., local convergence of descent techniques starting from carefully designed spectral initializations) [\(Ma et al., 2018;](#page-9-0) [Tu et al., 2016;](#page-10-3) [Ling &](#page-9-1)

[Strohmer, 2019;](#page-9-1) [Candes et al., 2015\)](#page-8-6); and landscape analysis paired with saddle-escaping algorithms which show, e.g., that all local minima are global and that saddle points exhibit strict negative curvature so that (stochastic) gradientbased methods can effectively escape saddles and ensure convergence to global minimizers [\(Jin et al., 2017;](#page-9-10) [Ge et al.,](#page-8-7) [2015;](#page-8-7) [Raginsky et al., 2017\)](#page-9-11).

Notably, several studies [\(Woodworth et al., 2020;](#page-10-4) [Ghorbani](#page-9-12) [et al., 2020\)](#page-9-12) have highlighted the importance of the scale of the training initialization for the generalization and test performance of modern machine learning architectures. In fact, a small random initialization followed by (stochastic) gradient descent is arguably the most widely used training algorithm in contemporary machine learning. And, stronger generalization performance is typically observed with smaller-scale initializations. Implicit bias for low-rank matrix recovery with small random initializations has been extensively studied in this setting as a result by, e.g., [\(Stoger](#page-10-0) ¨ [& Soltanolkotabi, 2021;](#page-10-0) [Soltanolkotabi et al., 2023;](#page-10-5) [Wind,](#page-10-6) [2023;](#page-10-6) [Kim & Chung, 2024\)](#page-9-13). These studies have shown that a small random Gaussian initialization behaves similarly to a spectral initialization in overparameterized settings. Furthermore, they have shown that gradient descent algorithms with this initialization tend to converge towards low-rank solutions (i.e., that they demonstrate an implicit regularization towards low-rank solutions).

Recently, numerous connections between tensor decompositions and training neural networks have also been established by, e.g., [\(Novikov et al., 2015;](#page-9-3) [Razin et al., 2021;](#page-9-4) [2022\)](#page-9-14). These studies argue that low-rank tensor factorization helps explain implicit regularization in deep learning, as well as how properties of real-world data translate this regularization to generalization. Similar to how matrix factorization can be viewed as a linear neural network (i.e., a fully connected network with linear activation), tensor factorizations correspond to a specific type of shallow (depthtwo) nonlinear convolutional neural network [\(Cohen et al.,](#page-8-0) [2016;](#page-8-0) [Razin et al., 2021\)](#page-9-4). Additionally, [\(Novikov et al.,](#page-9-3) [2015\)](#page-9-3) demonstrated that the dense weight matrices of fully connected layers can be converted to tensor trains while preserving the layer's expressive power. These findings have positioned low-rank tensor factorizations as theoretical surrogates for various neural network learning settings, thereby enhancing our understanding of implicit regularization and overparameterization, and so further motivating investigation in this area.

Since no unique definition of tensor rank is available, related literature concerning implicit bias has naturally split with respect to the notion of tensor rank being considered: CPrank, Tucker-rank, and tubal-rank, in analogy to the analysis of algorithms specifically designed for tensor recovery and completion by, e.g., [\(Zhang et al., 2019;](#page-10-7) [Hou et al., 2021;](#page-9-15)

Figure 1: A low tubal-rank factorization of a threedimensional tensor. Using the (reduced) tubal-SVD, each three-dimensional tensor T ∈ R n×m×k can be decomposed into a tubal product of three tensors T = V ∗ Σ ∗W<sup>⊤</sup> with V ∈ R n×n×k , W ∈ R m×m×k and the frontal slice diagonal tensor Σ ∈ R n×m×k . Here, the tubal rank of a tensor is the number of non-zero singular tubes in Σ ∈ R n×m×k . For example, in the figure, the tubal rank of the tensor is equal to six.

[Kong et al., 2018;](#page-9-16) [Ahmed et al., 2020;](#page-8-8) [Liu et al., 2019;](#page-9-17) [2020;](#page-9-18) [Haselby et al., 2024\)](#page-9-19). For the CP-tensor factorization, several results are available for gradient-based methods [\(Wang](#page-10-8) [et al., 2020;](#page-10-8) [Ge & Ma, 2017\)](#page-8-9). The first theoretical analysis of implicit regularization towards low tensor rank under arbitrarily small initialization was provided considering gradient flow in [\(Razin et al., 2021\)](#page-9-4). In [\(Ge et al., 2015\)](#page-8-7), it has been shown for the orthogonal tensor decomposition problem a simple variant of the stochastic gradient algorithm is able to leverage a low-rank structure from an arbitrary starting point. In addition, [\(Wang et al., 2020\)](#page-10-8) shows that using gradient descent on an over-parametrized objective for the CP-rank tensor decomposition problem one could go beyond the lazy training regime and utilize certain low-rank structures.

Perhaps most closely related to this paper, very recently [\(Liu et al., 2024\)](#page-9-5) analyzed the convergence of factorized gradient descent for the low-tubal-rank sensing problem, showing that with carefully designed spectral initialization the gradient iterates converge to a low-tubal rank tensor. Although the authors in [\(Liu et al., 2024\)](#page-9-5) allow for overparametrization, they argue the minimal recovery error can be achieved when knowing the true rank, thereby leaving questions concerning the advantages of overparametrization and small random initializations open.

Our contribution: Motivated by connections between tensor rank and non-linear neural network representations, herein we study the implicit regularization phenomenon for low tubal-rank tensor recovery. Namely, our objective is to analyze the recovery process of a tensor with a low tubalrank factorization [\(Kilmer & Martin, 2011\)](#page-9-20) (see Fig 1) from a limited number of random linear measurements. More

specifically, we consider tensors of the form X ∗X <sup>⊤</sup> and employ a non-convex method based on the tensor factorization, minimizing the loss function using gradient descent with a small random initialization. To the best of our knowledge, we are the first to investigate the implicit bias phenomenon for gradient descent with a small random initialization applied to a tensor factorization. Namely, we demonstrate that, irrespective of the degree of overparameterization, vanilla gradient descent with a small random initialization applied to a tubal tensor factorization will consistently converge to a low tubal-rank solution.

![](_page_2_Diagram_2.jpeg)

Inspired by recent results for the low-rank matrix sensing problem by [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0), we establish ¨ that gradient descent iterates with small random initializations can be closely approximated by power method iterations in [\(Gleich et al., 2013;](#page-9-21) [Kilmer et al., 2013\)](#page-9-22) modulo normalization, and deduce that after sufficient time the iterates approach a commonly used spectral initialization from the tubal-rank literature in [\(Liu et al., 2024\)](#page-9-5). Along the way we must also overcome, e.g., a challenging intersection between the tensor slices during each gradient descent iterate which forces a non-trivial convergence analysis.

Organization: In Section [2,](#page-2-0) we define our notation and present a few basic facts regarding tubal tensors. In Section [3,](#page-3-0) we state our problem and our main result. In Section [4,](#page-4-0) we outline the steps of the proof in order to provide intuition. In Section [5,](#page-6-0) we show numerical experiments which demonstrate our theoretical findings. We conclude the paper in Section [6.](#page-8-10) The proof of our main result is broken up into several lemmas, which are stated and proven in the appendix.

#### 2. Notation and Preliminaries

Every tensor in this paper will be an order-3 tensor whose third mode is length k. For such a tensor T ∈ R m×n×k , we define a block-diagonal Fourier domain representation by

$$\overline{\mathcal{T}} = \text{blockdiag}(\overline{\mathcal{T}}^{(1)}, \dots, \overline{\mathcal{T}}^{(k)}) \in \mathbb{C}^{mk \times nk}$$

where the j-th block T (j) ∈ C <sup>m</sup>×<sup>n</sup> is defined by T (j) (i, i′ ) = P<sup>k</sup> j ′=1 T (i, i′ , j′ )e − √ −12π(j−1)(j ′−1)/k . In other words, we take the FFT of each tube, and then arrange the resulting frontal slices into a block-diagonal matrix.

The tubal product (or t-product) of two tubal tensors A ∈ R m×q×k and B ∈ R q×n×k is a tubal tensor A ∗ B ∈ R <sup>m</sup>×n×<sup>k</sup> whose tubes are given by

$$(\mathcal{A} * \mathcal{B})(i, i', :) = \sum_{p=1}^q \mathcal{A}(i, p, :) * \mathcal{B}(p, i', :).$$

Here, ∗ denotes the circular convolution operation, i.e., (x ∗

y)<sup>i</sup> = P<sup>k</sup> <sup>j</sup>=1 xjyi−<sup>j</sup> (mod <sup>k</sup>) . One can check that A ∗ B = A B.

For any tubal tensor T ∈ R m×n×k , its tubal transpose T <sup>⊤</sup> ∈ <sup>R</sup> n×m×k is given by (T <sup>⊤</sup>)(i, i′ , 1) = T (i ′ , i, 1) and (T <sup>⊤</sup>)(i, i′ , j) = T (i ′ , i, k + 2 − j) for j = 2, . . . , k, i.e., we take the transpose of each face, and then reverse the order of frontal slices j = 2, . . . , k. This ensures that T <sup>⊤</sup> = T ⊤ .

For any n, the n × n × k identity tensor I ∈ R n×n×k is defined by I(:, :, 1) = In×<sup>n</sup> (identity matrix), and I(: , :, j) = 0n×<sup>n</sup> (zero matrix). An orthogonal tensor Q ∈ R n×n×k satisfies Q∗Q<sup>⊤</sup> = Q<sup>⊤</sup> ∗Q = I. An orthonormal tensor W ∈ R <sup>m</sup>×n×<sup>k</sup> with m ≥ n satisfies W<sup>⊤</sup> ∗W = I.

The tubal-SVD [\(Kilmer & Martin, 2011\)](#page-9-20) (or t-SVD) of a tubal tensor T ∈ R m×n×k is a factorization of the form

$$\mathcal{T} = \mathcal{U} * \Sigma * \mathcal{V}^\top \quad (2.1)$$

where U ∈ R m×m×k and V ∈ R n×n×k are orthogonal, and each frontal slice of Σ ∈ R m×n×k is diagonal. The t-SVD of a tensor T ∈ R m×n×k can be computed as follows: (1) compute the FFT of each tube of T to get the frontal slices T (j) , j = 1, . . . , k, (2) compute the SVD of each resulting frontal slice T (j) = U (j) Σ (j) V (j)⊤ , (3) concatenate the matrices {U (j) } k <sup>j</sup>=1 into a tubal tensor <sup>U</sup>e <sup>∈</sup> <sup>C</sup> m×m×k and take the inverse FFT along mode-3 to obtain U ∈ R m×m×k (and similarly to obtain Σ ∈ R m×n×k and V ∈ R n×n×k ). The tubal rank of a tensor T ∈ R m×n×k is the number of non-zero diagonal tubes in the Σ tensor of its t-SVD, i.e., rank(T ) = #{i : Σ(i, i, :) ̸= 0}. For an illustration of the t-SVD decomposition, see Figure [1.](#page-2-1) We also define the condition number κ(T ) of the tubal tensor T ∈ <sup>R</sup> m×n×k by

$$\kappa(\mathcal{T}) := \frac{\sigma_1(\overline{\mathcal{T}})}{\sigma_{\min\{m,n\}k}(\overline{\mathcal{T}})}.$$

Finally, for tubal tensors T ∈ R <sup>m</sup>×n×<sup>k</sup> we define the tensor spectral norm ∥T ∥ := ∥T ∥ and the tensor nuclear norm ∥T ∥<sup>∗</sup> := ∥T ∥<sup>∗</sup> as the spectral and nuclear norm respectively of the block-diagonal Fourier domain representation T , and the tensor Frobenius norm ∥T ∥ 2 F := P<sup>m</sup> i=1 P<sup>n</sup> j=1 P<sup>k</sup> <sup>ℓ</sup>=1 T (i, j, ℓ) <sup>2</sup> = k ∥T ∥ 2 F as a scaled version of the Frobenius norm of the block-diagonal Fourier domain representation T .

## 3. Main Results

Problem Formulation Let X ∈ R <sup>n</sup>×r×<sup>k</sup> have tubal rank r ≤ n so that X ∗ X <sup>⊤</sup> ∈ S n×n×k <sup>+</sup> is a tubal positive semidefinite tensor with tubal rank r. Let κ = κ(X ) be the condition number of X . Suppose we observe m linear measurements of X ∗ X <sup>⊤</sup>, that is

$$y_i = \left\langle \mathbf{A}_i, \mathbf{X} * \mathbf{X}^\top \right\rangle \quad \text{for} \quad i = 1, \dots, m \quad (3.1)$$

where each A<sup>i</sup> ∈ S n×n×k is a tubal-symmetric tensor. We can write this compactly as y = A(X ∗ X <sup>⊤</sup>) where A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> is the linear measurement operator. We aim to recover X ∗ X <sup>⊤</sup> from our measurements y by using gradient descent to learn an overparameterized factorization. Specifically, we fix an R ≥ r and try to find a U ∈ R n×R×k such that U ∗ U <sup>⊤</sup> = X ∗ X <sup>⊤</sup> by using gradient descent to minimize the loss function

$$\ell(\mathbf{u}) := \left\| \mathcal{A}(\mathbf{u} * \mathbf{u}^\top) - \mathbf{y} \right\|_2^2 \quad (3.2)$$

$$= \sum_{i=1}^m \left( \left\langle \mathcal{A}_i, \mathcal{U} * \mathcal{U}^\top \right\rangle - y_i \right)^2. \quad (3.3)$$

We will start with a small random initialization U<sup>0</sup> ∈ R <sup>n</sup>×R×<sup>k</sup> where each entry is i.i.d. N (0, α 2 R ) for some small α > 0. Then, the gradient descent iterations are given by

$$\begin{aligned}
\mathcal{U}_{t+1} &= \mathcal{U}_t - \mu \nabla \ell(\mathcal{U}_t) \\
&= \mathcal{U}_t + \mu \mathcal{A}^* \left[ \mathbf{y} - \mathcal{A} \left( \mathcal{U}_t * \mathcal{U}_t^\top \right) \right] * \mathcal{U}_t \\
&= \left[ \mathcal{I} + \mu (\mathcal{A}^* \mathcal{A}) \left( \mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top \right) \right] * \mathcal{U}_t
\end{aligned} \tag{3.4}$$

for some suitably small stepsize µ > 0. Here A<sup>∗</sup> : R <sup>m</sup> → S <sup>n</sup>×n×<sup>k</sup> denotes the adjoint of A which is given by A<sup>∗</sup>z = P<sup>m</sup> <sup>i</sup>=1 ziA<sup>i</sup> .

Moreover, we say that a measurement operator A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> satisfies the Restricted Isometry Property (RIP) of rank-r with constant δ > 0 (abbreviated RIP(r, δ)), if we have

$$(1 - \delta)\|\mathbf{Z}\|_F^2 \leq \|\mathcal{A}(\mathbf{Z})\|_2^2 \leq (1 + \delta)\|\mathbf{Z}\|_F^2,$$

for all Z ∈ S <sup>n</sup>×n×<sup>k</sup> with tubal-rank ≤ r. We note that an RIP condition is a standard condition in the literature, and is used in similar works such as [\(Li et al., 2018;](#page-9-23) [Stoger &](#page-10-0) ¨ [Soltanolkotabi, 2021\)](#page-10-0). This condition is necessary to ensure that there is only one low tubal rank tensor for which the loss function is zero, and that this tensor could be recovered stably in the presence of noise.

Results We have analyzed the convergence process of the gradient descent iterates [\(3.4\)](#page-3-1) in the scenario of small random initialization and overparametrization. Namely, with the ground truth tensor X ∈ R n×r×k , we assume the initialization U<sup>0</sup> ∈ <sup>R</sup> n×R×k is such that each entry is i.i.d. N (0, α 2 R ) with small scaling parameter α > 0 and the second dimension R exceeding three timesthe ground truth dimension r. Below, we present the direct results of our analysis.

Theorem 3.1. *Suppose we have* m *linear measurements* y = A(X ∗ X <sup>⊤</sup>) *of a tubal positive semidefinite tensor* X ∗ X <sup>⊤</sup> ∈ S n×n×k <sup>+</sup> *where* X ∈ <sup>R</sup> <sup>n</sup>×r×<sup>k</sup> *has tubal rank* r ≤ n*. We assume* A *satisfies RIP*(2r + 1, δ) *with* δ ≤ cκ−<sup>4</sup> r −1/2 *. Suppose we fit a model* X ∗ X <sup>⊤</sup> = U ∗ U ⊤ *where* U ∈ R <sup>n</sup>×R×<sup>k</sup> *with* R ≥ 3r *and obtain* U *by running the gradient descent iterations*

$$u_{t+1} = \left[ \mathcal{I} + \mu(\mathcal{A}^* \mathcal{A}) \left( x * x^\top - u_t * u_t^\top \right) \right] * u_t$$

*with a stepsize* µ ≤ c √ kκ−<sup>4</sup>∥X ∥ 2 *starting from the initialization* U<sup>0</sup> ∈ <sup>R</sup> <sup>n</sup>×R×<sup>k</sup> *where each entry is i.i.d.* N (0, α 2 R )*. Then, if the scale of the initialization satisfies*

$$\alpha \lesssim \frac{\sigma_{\min}(\mathcal{X})}{\kappa^2 \min\{n, R\} \sqrt{k}} \left( \frac{C_2 \kappa^2 \sqrt{n}}{\sqrt{\min\{n, R\}}} \right)^{-16\kappa^2},$$

*then after*

$$\hat{t} \lesssim \frac{1}{\mu\sigma_{\min}(\mathcal{X})^2} \ln \left( \frac{C_1 \kappa n}{\min\{n, R\}} \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \right) \frac{\|\mathcal{X}\|}{k\alpha} )$$

*iterations, we have that*

$$\frac{\|\mathcal{U}_{\hat{t}} * \mathcal{U}_{\hat{t}}^\top - \mathcal{X} * \mathcal{X}^\top\|_F^2}{\|\mathcal{X}\|^2} \lesssim k \frac{61}{32} r^{\frac{1}{8}} \kappa^{-\frac{3}{16}} (\min\{n, R\} - r)^{\frac{3}{8}} \left[ \frac{C_2 \kappa^2 \sqrt{n}}{\sqrt{\min\{n, R\}}} \right]^{21 \kappa^2} \left[ \frac{\alpha}{\|\mathcal{X}\|} \right]^{\frac{21}{16}}$$

*holds with probability at least* 1 − Cke−cR˜ *. Here,* c, c, C, C ˜ <sup>1</sup>, C<sup>2</sup> > 0 *are fixed numerical constants.*

Intuitively, this means that if the initialization is sufficiently small, gradient descent will approximately recover the low tubal rank tensor X ∗ X <sup>⊤</sup> after b<sup>t</sup> iterations. Note that the reconstruction error can be made arbitrarily small by making the size of the random initialization α arbitrarily small. This comes at the expense of requiring more iterations. However, this impact is mild as the number of iterations grows only logarithmically with respect to α.

Although the above theorem holds for any R ≥ 3r, it is perhaps most interesting in the case where R ≥ n as then every n × n × k tubal positive semidefinite tensor can be expressed as U ∗ U <sup>⊤</sup> for some U ∈ <sup>R</sup> n×R×k . Hence, the learner model does not assume that the ground truth tensor has low tubal rank, yet gradient descent is able to recover the ground truth tensor instead of any of the infinitely many high tubal rank tensors whose measurements match that of the ground truth tensor.

We note that [\(Zhang et al., 2019\)](#page-10-7) shows that a random sub-Gaussian measurement operator A : R <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> will satisfy the RIP for tubal rank-r tensors with RIP constant δ with high probability if m ≥ O(rnk/δ<sup>2</sup> ). To obtain an RIP

constant of δ = O(κ −4 r −1/2 ), one needs m ≥ O(κ 8 r <sup>2</sup>nk) random sub-Gaussian measurements.

Additionally, we acknowledge that the parameter dependence in Theorem [3.1](#page-3-2) may initially seem unfamiliar. However, it aligns well with intuition and prior work: when the tensor is ill-conditioned – i.e., possesses a small tubal singular value – gradient descent without regularization naturally struggles to recover the rank-one component unless the initialization is sufficiently small. While our bound exhibits exponential dependence on the condition number, this is consistent with known results in the matrix setting (e.g., see Lemma 8.6 in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0)).Although ¨ the necessity of exponential dependence remains an open question, it presents a compelling direction for future research. Moreover, our numerical experiments (see Figure [4\)](#page-7-0) support a polynomial relationship between the test error and the initialization parameter α, and while the empirical degree may differ slightly, our theoretical exponent <sup>21</sup> <sup>16</sup> appears to closely approximate the observed behavior.

## 4. Proof Outline

In this section, we turn our attention to giving an overview of the key ideas of the proof.

In our analysis, we demonstrate that the trajectory of gradient descent iterations can be approximately divided into two distinct stages: (I) a spectral stage and (II) a convergence stage described below.

*(I) The spectral stage.* In the spectral stage, where we show that the gradient descent starting from random initialization behaves similarly to spectral initialization, enabling us to prove that by the end of this stage, the column spaces of the tensor iterates U<sup>t</sup> [\(3.4\)](#page-3-1) and the ground truth matrix X are sufficiently aligned. Namely, we show that the first few iterations of the gradient descent algorithm U<sup>t</sup> can be approximated by the iteration of the tensor power method modulo normalization (see, e.g.[\(Gleich et al., 2013\)](#page-9-21)) defined as

$$\tilde{\mathcal{U}}_t = \left( \mathcal{I} + \mu \mathcal{A}^* \mathcal{A}(\boldsymbol{x} * \boldsymbol{x}^\top) \right)^{*t} * \mathbf{u}_0 \in \mathbb{R}^{n \times R \times k}.$$

We call this part of the evolution of the gradient descent iteration the "spectral stage" since, due to its similarity to the power method, at the end of this stage the iterates U<sup>t</sup> will be closely aligned with the classical t-SVD spectral initialization of [\(Liu et al., 2024\)](#page-9-5).

*(II) The convergence stage*. In the convergence stage, the gradient iterates converge approximately to the underlying low tubal-rank tensor X ∗ X <sup>⊤</sup> at a geometric rate until reaching a certain error floor which is dependent on the initialization scale.

The cornerstone of the analysis of this stage is the de-

Figure 2: Illustration of (top figure) the two stages of gradient descent algorithm: the spectral alignment stage for 1 ≤ t ≲ 3000 and the convergence stage 3000 ≲ t and (bottom figure) more details on the alignment phase for the gradient descent progress. In the ground truth tensor X ∈ R n×r×k , we set n = 10, k = 4, r = 3.

composition of the tensor gradient iterates U<sup>t</sup> into two components, the so-called "signal" and "noise" terms. This is done by adapting similar decomposition methods used in recent works analyzing implicit bias phenomenon for gradient descent in the matrix setting (see [\(Stoger &](#page-10-0) ¨ [Soltanolkotabi, 2021;](#page-10-0) [Li et al., 2018\)](#page-9-23)) to our tensor setting. Accordingly, let the tensor-column subspace of the ground truth tensor X ∈ R <sup>n</sup>×r×<sup>k</sup> be denoted by V<sup>X</sup> with the corresponding basis V<sup>X</sup> ∈ <sup>R</sup> n×r×k . Consider the tensor V<sup>X</sup> ∗ U<sup>t</sup> ∈ <sup>R</sup> <sup>r</sup>×R×<sup>k</sup> with its t-SVD decomposition V<sup>X</sup> ∗ U<sup>t</sup> = V<sup>t</sup> ∗ Σ<sup>t</sup> ∗ W<sup>⊤</sup> t . For W<sup>t</sup> ∈ <sup>R</sup> R×r×k , we

denote by Wt,<sup>⊥</sup> ∈ <sup>R</sup> R×(n−r)×k a tensor whose tensorcolumn subspace is orthogonal to those of Wt, that is ∥W<sup>⊤</sup> t,<sup>⊥</sup> ∗ Wt∥ = 0 and its projection operator P<sup>W</sup>t,<sup>⊥</sup> is defined as P<sup>W</sup>t,<sup>⊥</sup> <sup>=</sup> Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> <sup>=</sup> I − W<sup>t</sup> ∗ W<sup>⊤</sup> t .

![](_page_5_Figure_2.jpeg)

We then decompose the gradient descent iterates [\(3.4\)](#page-3-1) as follows

$$\mathcal{U}_t = \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top + \mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top \quad (4.1)$$

referring to the tensors U<sup>t</sup> ∗ W<sup>t</sup> ∗ W<sup>⊤</sup> t as the signal term of the gradient descent iterates, and to the tensors U<sup>t</sup> ∗ Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> as the noise term. The advantage of such a decomposition is that the tensor-column space of the noise term U<sup>t</sup> ∗Wt,<sup>⊥</sup> ∗W<sup>⊤</sup> t,<sup>⊥</sup> is orthogonal to the tensor-column subspace of the ground truth X allowing for a rigorous analysis of the convergence process of the two components separately.

At the convergence stage, we show that symmetric tensor U<sup>t</sup> ∗W<sup>t</sup> ∗W<sup>⊤</sup> <sup>t</sup> ∗U ⊤ <sup>t</sup> built from the signal term converges towards the ground truth tensor X ∗ X <sup>⊤</sup>, whereas the spectral norm of the noise term ∥U<sup>t</sup> ∗ Wt,<sup>⊥</sup>∥, stays small.

Additional challenges in the tensor setting vs. matrix setting When coming from the matrix case to the tensor setting com, there are several important differences and challenges, which need to be carefully considered and are described below.

- In contrast to the matrix case, the range and kernel of a third-order tubal tensor can include overlapping generator elements (we refrain from using the term basis, in the sense that knowledge of the multirank and complimentary tubal scalar of a tensor must be included to describe the range). Namely, if in the t-SVD [\(2.1\)](#page-3-3) of a symmetric tensor X the tensor Σ contains q non-invertible tubes – tubes that have zero elements in the Fourier domain –, then there are q common generators for the range and the kernel of X , please see [\(Kilmer et al., 2013\)](#page-9-22) for more details. With this phenomenon, the decomposition [\(C.1\)](#page-11-0) of the gradient iterates into signal and noise term is not available for non-invertible tubes, which is why we need to work with a more intricate notion of condition number.
- As stated in [\(Gleich et al., 2013\)](#page-9-21), running the power method for tubal tensors of dimensions n × n × k is equivalent to running in parallel k independent matrix power methods in Fourier domain. However, running gradient descent in the tubal tensor setting is not equivalent to running k gradient descent algorithms independently in Fourier space. This can be easily seen when transforming the measurement operator part of the gradient descent iterates.

Figure 3: Outcomes of employing gradient descent to minimize the loss function [\(3.2\)](#page-3-4) with different overparametrization rates. We set n = 10, k = 4, r = 3 in the ground truth tensor X ∈ R n×r×k and for initialization U<sup>0</sup> ∈ <sup>R</sup> n×R×k , we set the over-rank to R = 10, 50, 100, 200, 400. For each R we plot the average over twenty experiments. The plots for <sup>∥</sup>Ut∗<sup>U</sup> ⊤ <sup>t</sup> <sup>−</sup>X∗<sup>X</sup> <sup>⊤</sup>∥<sup>F</sup> ∥X∗X <sup>⊤</sup>∥<sup>F</sup> , <sup>ℓ</sup>(Ut) and <sup>∥</sup>σr(Ut)−σr(X)∥<sup>2</sup> ∥σr(X)∥<sup>2</sup> are semi-log plots.

Namely, let as before y = A(X ∗ X <sup>⊤</sup>) ∈ <sup>R</sup> m with y<sup>i</sup> = D A<sup>i</sup> , X ∗ X ⊤ E = D A<sup>i</sup> , X ∗ X ⊤ E = P<sup>k</sup> <sup>q</sup>=1 A<sup>i</sup> (q) , X(q)X(q)H , j = 1, . . . m then A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>) = A<sup>∗</sup> (y) = P<sup>m</sup> <sup>i</sup>=1 yiA<sup>i</sup> ∈ S n×n×k and the for j-th slice in the Fourier domain, we get A<sup>∗</sup>A(X ∗ X ⊤) (j) = P<sup>m</sup> i=1 P<sup>k</sup> <sup>j</sup>=1 A<sup>i</sup> (j) A<sup>i</sup> (q) , X(q)X(q)H . This means that in each Fourier slice U<sup>t</sup> (j) of the gradient descent iterates [\(3.4\)](#page-3-1) we have the full information about the ground truth tensor X ∗ X <sup>⊤</sup> and not only about its j-th slice. In the spectral stage, this fact does not cause significant difficulties. However, in the convergence stage, in order to get the global estimates, it requires a thorough and vigilant analysis of intersections between the slices in the Fourier domain.

![](_page_6_Figure_2.jpeg)

In particular, this required nontrivial estimations, such as those presented in Lemmas E.4 and E.5, to control these interactions and provide the respective bounds, which require control of proximity of the auxiliary parameter A<sup>∗</sup>A(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ) (j) to the corresponding jth Fourier slice of X ∗ X <sup>⊤</sup> −U<sup>t</sup> ∗U ⊤ <sup>t</sup> via the RIP property of the measurement operator A and aligned matrix subspaces. Another important point is that one need to choose the learning rate µ and the initialization scale α carefully for the noise term U<sup>t</sup> ∗ W<sup>⊥</sup>,t to grow slowly enough in each of the tensor slices in order to not allow overtaking the signal term U<sup>t</sup> ∗ W<sup>t</sup> in the norm, see, e.g., Theorem [E.1](#page-40-0) and the usage of Lemma [E.3](#page-30-0) in its proof.

#### 5. Numerical Experiments

To verify our theoretical findings, we set multiple numerical tests: from showing two phases of the gradient descent algorithm to demonstrating the advantages of overparametrization. These experimental results showcase not only the implicit regularization for the gradient descent algorithm toward low-tubal-rank tensors but also demonstrate the firmness of our theoretical findings.

Our experiments were conducted on a MacBook Pro equipped with an Apple M1 processor and 16GB of memory, using MATLAB 2023a software. The corresponding code is available in our GitHub repository, [https://github.com/AnnaVeselovskaUA/tubal-tensor](https://github.com/AnnaVeselovskaUA/tubal-tensor-implicit-reg-GD.git)[implicit-reg-GD.git.](https://github.com/AnnaVeselovskaUA/tubal-tensor-implicit-reg-GD.git)

We generate the ground truth tensor T ∈ R <sup>n</sup>×n×<sup>k</sup> with tubal rank r by T = X ∗ X <sup>⊤</sup> , where the entries of X ∈ R n×r×k are i.i.d. sampled from a Gaussian distribution N (0, 1), and then X is normalized. The entries of measurement tensor A<sup>i</sup> are i.i.d. sampled from a Gaussian distribution N (0, <sup>m</sup> ). In the following, we describe different testing scenarios for recovery of T via the gradient descent algorithm and their outcome. For all the experiments, we set the dimensions to n = 10, k = 4, r = 3, the learning rate µ = 10−<sup>5</sup> , and the number of measurements m = 254.

Illustration of the two convergence stages. To illustrate the convergence process of the gradient iterates, for the ground truth tensor X ∗ X <sup>⊤</sup> ∈ <sup>R</sup> n×n×k and its counterpart U<sup>t</sup> ∗ U ⊤ <sup>t</sup> ∈ <sup>R</sup> <sup>n</sup>×n×<sup>k</sup> being learned by the gradient descent, we consider the training error ℓ(Ut), the test error ∥Ut∗U ⊤ <sup>t</sup> <sup>−</sup>X∗<sup>X</sup> <sup>⊤</sup>∥<sup>F</sup> ∥X∗X <sup>⊤</sup>∥<sup>F</sup> , and the test error for their rth singular tubes σr(Ut), σr(X ) ∈ <sup>R</sup> k , ∥σr(Ut)−σr(X)∥<sup>2</sup> ∥σr(X)∥<sup>2</sup> . Moreover, we also take into our consideration the tensor subspace L spanned by the tensor-columns corresponding to the first r singular-tubes of the tensor A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>) and denote by L<sup>t</sup> the tensor-column subspace spanned by the tensorcolumns corresponding to the first r singular tubes U<sup>t</sup> ∗ U ⊤ t . We note that although Theorem [3.1](#page-3-2) bounded a relative error with ∥X ∥ 2 in the denominator, we use ∥X ∗ X <sup>⊤</sup>∥<sup>F</sup> in the denominator of the relative error for our experiments as it is a more natural relative error to consider. Furthermore, since ∥X ∗ X <sup>⊤</sup>∥<sup>F</sup> ≥ ∥X ∥ , and ∥X ∗ X <sup>⊤</sup>∥<sup>F</sup> could be much larger than ∥X ∥ 2 in cases where the singular values of X ∗ X <sup>⊤</sup> vary drastically, the result of Theorem [3.1](#page-3-2) is stronger than if we bounded the more natural Frobenius norm error. Besides, the qualitative behavior in the numerical simulation will be the same for the two error measures as generically they will just differ by a dimensional factor.

![](_page_7_Figure_5.jpeg)

Figures [2](#page-5-0) demonstrates that the convergence analysis can be divided into two stages: the spectral and the convergence stage. We see that in the first stage (1 ≤ t ≲ 3000), the first r tensor-columns of U<sup>t</sup> ∗ U ⊤ t learn the tensor column subspace corresponding to the first r singular-tubes of the tensor A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>), i.e. the principal angle between the tensor column subspaces L<sup>t</sup> and L becomes small. Namely, as one can observe in Figure [2](#page-5-0) (bottom), the principal angle between the two subspaces, ∥V ⊤ <sup>L</sup><sup>⊥</sup> ∗V<sup>L</sup><sup>t</sup> ∥, decreases where as the principal angle between X and L<sup>t</sup> reaches certain plateau, see the behavior of ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>L</sup><sup>t</sup> ∥. At the same time, test errors <sup>∥</sup>Ut∗<sup>U</sup> ⊤ <sup>t</sup> <sup>−</sup>X∗<sup>X</sup> <sup>⊤</sup>∥<sup>F</sup> ∥X∗X <sup>⊤</sup>∥<sup>F</sup> and <sup>∥</sup>σr(Ut)−σr(X)∥<sup>2</sup> ∥σr(X)∥<sup>2</sup> stay large. In the second stage, we see that the test error ∥Ut∗U ⊤ <sup>t</sup> <sup>−</sup>X∗<sup>X</sup> <sup>⊤</sup>∥<sup>F</sup> ∥X∗X <sup>⊤</sup>∥<sup>F</sup> starts decreasing, meaning that the gradient descent iterates U<sup>t</sup> ∗ U ⊤ t start converging to X ∗ X ⊤ by learning more about the tensor-column subspace of the ground truth tensor. At the same time, the test error over rth singular tube <sup>∥</sup>σr(Ut)−σr(X)∥<sup>2</sup> ∥σr(X)∥<sup>2</sup> starts decreasing too and as a result converges to zero. We also see that in this stage the principal angle between L<sup>t</sup> and L grows, which is also intuitive as the tensor-column subspace L does not have the full information about the tensor-column subspace of

the ground truth tensor X ∗ X <sup>⊤</sup>, and learning more about X ∗ X <sup>⊤</sup> leads to a larger error in terms of principal angles of the two.

Depiction of the alignment stage. In this experiment, we illustrate that gradient descent with small initialization behaves similarly to the tensor-power method modulo normalization in the first few iterations, bringing the gradient iterates close to the spectral tubal initialization, used, e.g., in [\(Liu et al., 2024\)](#page-9-5). Here, as before L denote the tensor subspace spanned by the tensor-columns corresponding to the first r singular-tubes of tensor A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>) and L<sup>t</sup> is the tensor-column subspace corresponding to the first r singular tubes U<sup>t</sup> ∗ U ⊤ t . Additionally, Le<sup>t</sup> denotes the tensor-column subspace spanned by the first r singular-tubes of the tensor Ue<sup>t</sup> ∗Ue ⊤ t , where Ue ⊤ <sup>t</sup> = I + A<sup>∗</sup>A X ∗ X ⊤ ∗t ∗ U0. In Figure [2](#page-5-0) (bottom), we see that U<sup>t</sup> and Ue<sup>t</sup> learn the subspace L almost at the same rate in the first iterations, 1 ≤ t ≲ 3000. In the same figure, we observe that also the angle between V<sup>X</sup> and Lt, respectively Let, decreases monotonically in the spectral stage. Then at the beginning of the convergence stage, 3000 ≲ t, the angle between V<sup>X</sup> and L<sup>t</sup> starts decreasing gradually and converges to zero, as expected since U<sup>t</sup> ∗U ⊤ t converges to X ∗ X <sup>⊤</sup>. Whereas the principal angle between L and L<sup>t</sup> growths until it reaches a certain plateau.

Figure 4: Impact of different initialization scales on the test and the training error. The data are represented in the log-log plot. We set n = 10, k = 4, r = 3 in the ground truth tensor X ∈ R n×r×k and for initialization U<sup>0</sup> = α U ∈ <sup>R</sup> n×R×k with R = 200 and different scales of α. The plot depicts the averaged value for five runs and the bars represent the deviations from the mean value. For illustration, we also depict the theoretical test error bound obtained in Theorem [3.1.](#page-3-2) As one can see, the numerical error resembles the theoretical behavior of Cn,k,r,κ · α 21 <sup>16</sup> .

Test and train error under different scales of initialization. In this experiment, we explore the influence of the initialization scale, denoted by α, on the training and the test error. With R = 200, we apply gradient descent for various values of α, halting the iterations at t = 3500 in each run. The results, presented in Figure [4,](#page-7-0) demonstrate a reduction in test error as α decreases. Notably, the figure indicates that the test error follows an almost polynomial relationship with the initialization scale α. This observation is consistent with our theoretical predictions, which also forecast a decrease in test error at a rate of α, see Theorem [3.1.](#page-3-2)

Impact of different levels of overparameterization on the convergence. In this numerical analysis, we set α = 10−<sup>7</sup> and examined the convergence speed of gradient descent to the ground truth tensor for various overparameterization rates R. We run the experiment twenty times for each value of R and plot the averaged values per each iteration. The results, shown in Figure [3,](#page-6-1) reveal that increasing the number of tensor columns R, that is, overparameterizing, accelerates the convergence rate, resulting in fewer iterations to reach the desired error level. Additionally, overparameterization reduces the test error and the training error by affecting the spectral stages.

## 6. Conclusion and Outlook

In this paper, we focused on studying the implicit regularization of tubal tensor factorizations via gradient descent by showing that with small random initialization and overparametrization, the gradient descent algorithm is biased towards a low-tubal-rank solution. We have shown that the first iterations of gradient descent with small random initialization behave similarly to the tensor power method, which leads to learning in these first iterations the tensor-column spaces close to the tensor-column space of the ground truth. We also demonstrate that the implicit regularization from small random initialization guides the gradient descent iterations toward low-tubal rank solutions that are not only globally optimal but also generalize well.

## Acknowledgments

AV and FK acknowledge support by the German Science Foundation (DFG) in the context of the collaborative research center TR-109, the Emmy Noether junior research group KR 4512/1-1 and the Bavarian Funding Program for Initiating International Research Cooperation, as well as by the Munich Data Science Institute and Munich Center for Machine Learning. SK acknowledges support by the United States National Science Foundation in the context of the Foundations of Data Science Institute funded by grant NSF DMS 2022205. MI acknowledges support by the United States National Science Foundation grants NSF

DMS 2108479 and NSF EDU DGE 2152014.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, and more specifically, the theoretical understanding of implicit regularization as a tool for structured recovery problems. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References


[1] Ahmed, T., Raja, H., and Bajwa, W. U. Tensor regression using low-rank and sparse tucker decompositions. *SIAM Journal on Mathematics of Data Science*, 2(4):944–966, 2020. Arora, S., Cohen, N., Hu, W., and Luo, Y. Implicit regularization in deep matrix factorization. *Advances in Neural Information Processing Systems*, 32, 2019. Bah, B., Rauhut, H., Terstiege, U., and Westdickenberg,

[2] M. Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers. *Information and Inference: A Journal of the IMA*, 11(1): 307–353, 2022. Candes, E. J., Li, X., and Soltanolkotabi, M. Phase retrieval via wirtinger flow: Theory and algorithms. *IEEE Transactions on Information Theory*, 61(4):1985–2007, 2015. Chou, H.-H., Maly, J., and Rauhut, H. More is less: inducing sparsity via overparameterization. *Information and Inference: A Journal of the IMA*, 12(3):1437–1460, 2023. Chou, H.-H., Gieshoff, C., Maly, J., and Rauhut, H. Gradient descent for deep matrix factorization: Dynamics and implicit bias towards low rank. *Applied and Computational Harmonic Analysis*, 68:101595, 2024. Cohen, N., Sharir, O., and Shashua, A. On the expressive power of deep learning: A tensor analysis. In *Conference on learning theory*, pp. 698–728. PMLR, 2016. Ge, R. and Ma, T. On the optimization landscape of tensor decompositions. *Advances in neural information processing systems*, 30, 2017. Ge, R., Huang, F., Jin, C., and Yuan, Y. Escaping from saddle points—online stochastic gradient for tensor decomposition. In *Conference on learning theory*, pp. 797–842. PMLR, 2015. Geyer, K., Kyrillidis, A., and Kalev, A. Low-rank regularization and solution uniqueness in over-parameterized

[3] matrix sensing. In *International Conference on Artificial Intelligence and Statistics*, pp. 930–940. PMLR, 2020. Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A. When do neural networks outperform kernel methods? *Advances in Neural Information Processing Systems*, 33: 14820–14830, 2020. Gleich, D. F., Greif, C., and Varah, J. M. The power and arnoldi methods in an algebra of circulants. *Numerical Linear Algebra with Applications*, 20(5):809–831, 2013. Gunasekar, S., Woodworth, B. E., Bhojanapalli, S., Neyshabur, B., and Srebro, N. Implicit regularization in matrix factorization. *Advances in neural information processing systems*, 30, 2017. Gunasekar, S., Lee, J. D., Soudry, D., and Srebro, N. Implicit bias of gradient descent on linear convolutional networks. *Advances in neural information processing systems*, 31, 2018. Haselby, C., Iwen, M., Karnik, S., and Wang, R. Tensor deli: Tensor completion for low cp-rank tensors via random sampling, 2024. Hou, J., Zhang, F., Qiu, H., Wang, J., Wang, Y., and Meng,

[4] D. Robust low-tubal-rank tensor recovery from binary measurements. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(8):4355–4373, 2021. Jin, C., Ge, R., Netrapalli, P., Kakade, S. M., and Jordan,

[5] M. I. How to escape saddle points efficiently. In *International conference on machine learning*, pp. 1724–1732. PMLR, 2017. Kilmer, M. E. and Martin, C. D. Factorization strategies for third-order tensors. *Linear Algebra and its Applications*, 435(3):641–658, 2011. Kilmer, M. E., Braman, K., Hao, N., and Hoover, R. C. Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. *SIAM Journal on Matrix Analysis and Applications*, 34(1):148–172, 2013. Kim, D. and Chung, H. W. Rank-1 matrix completion with gradient descent and small random initialization. *Advances in Neural Information Processing Systems*, 36, 2024. Kolb, C., Muller, C. L., Bischl, B., and R ¨ ugamer, D. Smooth- ¨ ing the edges: A general framework for smooth optimization in sparse regularization using hadamard overparametrization. *arXiv preprint arXiv:2307.03571*, 2023. Kong, H., Xie, X., and Lin, Z. t-schatten-p norm for lowrank tensor recovery. *IEEE Journal of Selected Topics in Signal Processing*, 12(6):1405–1419, 2018. Li, Y., Ma, T., and Zhang, H. Algorithmic regularization in over-parameterized matrix sensing and neural networks with quadratic activations. In *Conference On Learning Theory*, pp. 2–47. PMLR, 2018. Li, Z., Luo, Y., and Lyu, K. Towards resolving the implicit bias of gradient descent for matrix factorization: Greedy low-rank learning. *arXiv preprint arXiv:2012.09839*, 2020. Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi,
  - S. J., Ye, K., Chern, F., Yu, F., Guo, R., et al. The lazy neuron phenomenon: On emergence of activation sparsity in transformers. *arXiv preprint arXiv:2210.06313*, 2022. Ling, S. and Strohmer, T. Regularized gradient descent: a non-convex recipe for fast joint blind deconvolution and demixing. *Information and Inference: A Journal of the IMA*, 8(1):1–49, 2019. Liu, X.-Y., Aeron, S., Aggarwal, V., and Wang, X. Lowtubal-rank tensor completion using alternating minimization. *IEEE Transactions on Information Theory*, 66(3): 1714–1737, 2019. Liu, X.-Y., Aeron, S., Aggarwal, V., and Wang, X. Lowtubal-rank tensor completion using alternating minimization. *IEEE Transactions on Information Theory*, 66(3): 1714–1737, 2020. doi: 10.1109/TIT.2019.2959980. Liu, Z., Han, Z., Tang, Y., Zhao, X.-L., and Wang, Y. Lowtubal-rank tensor recovery via factorized gradient descent. *arXiv preprint arXiv:2401.11940*, 2024. Ma, C., Wang, K., Chi, Y., and Chen, Y. Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval and matrix completion. In *International Conference on Machine Learning*, pp. 3345–3354. PMLR, 2018. Novikov, A., Podoprikhin, D., Osokin, A., and Vetrov, D. P. Tensorizing neural networks. *Advances in neural information processing systems*, 28, 2015. Raginsky, M., Rakhlin, A., and Telgarsky, M. Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis. In *Conference on Learning Theory*, pp. 1674–1703. PMLR, 2017. Razin, N., Maman, A., and Cohen, N. Implicit regularization in tensor factorization. In *International Conference on Machine Learning*, pp. 8913–8924. PMLR, 2021. Razin, N., Maman, A., and Cohen, N. Implicit regularization in hierarchical tensor factorization and deep convolutional neural networks. In *International Conference on Machine Learning*, pp. 18422–18462. PMLR, 2022.

[6] Rudelson, M. and Vershynin, R. Smallest singular value of a random rectangular matrix. *Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences*, 62(12):1707–1739, 2009. Soltanolkotabi, M., Stoger, D., and Xie, C. Implicit bal- ¨ ancing and regularization: Generalization and convergence guarantees for overparameterized asymmetric matrix sensing. In *The Thirty Sixth Annual Conference on Learning Theory*, pp. 5140–5142. PMLR, 2023. Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N. The implicit bias of gradient descent on separable data. *Journal of Machine Learning Research*, 19 (70):1–57, 2018. Stoger, D. and Soltanolkotabi, M. Small random initializa- ¨ tion is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction. *Advances in Neural Information Processing Systems*, 34:23831–23843, 2021. Tao, T. and Vu, V. Random matrices: The distribution of the smallest singular values. *Geometric And Functional Analysis*, 20:260–297, 2010. Tu, S., Boczar, R., Simchowitz, M., Soltanolkotabi, M., and Recht, B. Low-rank solutions of linear matrix equations via procrustes flow. In *International Conference on Machine Learning*, pp. 964–973. PMLR, 2016. Vaskevicius, T., Kanade, V., and Rebeschini, P. Implicit regularization for optimal sparse recovery. *Advances in Neural Information Processing Systems*, 32, 2019. Vershynin, R. *High-dimensional probability: An introduction with applications in data science*, volume 47. Cambridge university press, 2018. Wang, X., Wu, C., Lee, J. D., Ma, T., and Ge, R. Beyond lazy training for over-parameterized tensor decomposition. *Advances in Neural Information Processing Systems*, 33:21934–21944, 2020. Wedin, P.-A. Perturbation bounds in connection with sin- ˚ gular value decomposition. *BIT Numerical Mathematics*, 12:99–111, 1972. Wind, J. S. Asymmetric matrix sensing by gradient descent with small random initialization. *arXiv preprint arXiv:2309.01796*, 2023. Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D., and Srebro, N. Kernel and rich regimes in overparametrized models. In *Conference on Learning Theory*, pp. 3635–3673. PMLR, 2020. Zhang, F., Wang, W., Hou, J., Wang, J., and Huang, J. Tensor restricted isometry property analysis for a large class of random measurement ensembles. *arXiv preprint arXiv:1906.01198*, 2019.
# Supplementary Material

#### A. Outline of Appendices

For ease of organization, we divide the supplementary material into appendices as follows. In Appendix [B,](#page-11-1) we define some additional notation, including the angles between two tensor-column subspaces. In Appendix [C,](#page-11-2) we decompose the gradient descent iterates into a "signal" term and a "noise" term, which will aid us in our analysis. In Appendices [D](#page-12-0) and [E,](#page-27-0) we analyze the spectral and convergence stages, respectively, of the gradient descent iterations. In Appendix [F,](#page-48-0) we prove our main result.

To avoid breaking up the flow of our analysis, we put some technical lemmas in the last few appendices instead of in the previously mentioned appendices. In Appendix [G,](#page-49-0) we prove some properties of measurement operators which satisfy the restricted isometry property. In Appendix [H,](#page-51-0) we prove some properties of matrices and their subspaces. Finally, in Appendix [I,](#page-53-0) we prove some properties of random Gaussian tubal tensors.

#### B. Additional Notation

For a tensor Y ∈ R n×r×k , we denote its t-SVD by Y = V<sup>Y</sup> ∗ Σ<sup>Y</sup> ∗ W<sup>⊤</sup> <sup>Y</sup> with the two orthogonal tensor V<sup>Y</sup> ,W<sup>Y</sup> ∈ <sup>R</sup> n×r×k , and the f-diagonal tensor Σ<sup>Y</sup> ∈ <sup>R</sup> r×r×k . We will refer to V<sup>Y</sup> as the tensor-column subspace of Y and by VY<sup>⊥</sup> ∈ <sup>R</sup> <sup>n</sup>×(n−r)×<sup>k</sup> we denote the tensor-column subspace orthogonal to V<sup>Y</sup> with its projection operator VY<sup>⊥</sup> ∗ V ⊤ <sup>Y</sup><sup>⊥</sup> = I − V<sup>Y</sup> ∗ V ⊤ Y .

We measure the angles between two tensor-column subspaces Y<sup>1</sup> and Y<sup>2</sup> by the tensor-spectral norm ∥VY<sup>⊥</sup> ∗ V<sup>Y</sup><sup>2</sup> ∥ which according to [\(Liu et al., 2019;](#page-9-17) [Gleich et al., 2013;](#page-9-21) [Kilmer & Martin, 2011\)](#page-9-20) is equal to

$$\| \boldsymbol{v}_{\boldsymbol{y}_1^+}^\top * \boldsymbol{v}_{\boldsymbol{y}_2} \| = \| \overline{\boldsymbol{v}_{\boldsymbol{y}_1^+}^\top * \boldsymbol{v}_{\boldsymbol{y}_2}} \| = \| \overline{\boldsymbol{v}_{\boldsymbol{y}_1^+}^\top \overline{\boldsymbol{v}_{\boldsymbol{y}_2}}} \|.$$

which means that the largest principal angle between Y<sup>1</sup> and Y<sup>2</sup> equals to that of these two subspaces represented in the Fourier domain. In the Fourier domain, since V ⊤ Y<sup>⊥</sup> ∈ C (n−r)k×nk and V<sup>Y</sup><sup>2</sup> ∈ <sup>C</sup> nk×nk are block diagonal matrices, it holds that

$$\|\overline{\mathbf{v}_{\mathcal{Y}_1^\perp}^\top \overline{\mathbf{v}_{\mathcal{Y}_2}}\| = \left\| \begin{pmatrix} \overline{\mathbf{v}_{\mathcal{Y}_1^\perp}^{\top}(1)} & \overline{\mathbf{v}_{\mathcal{Y}_1^\perp}^{\top}(2) & \dots & \overline{\mathbf{v}_{\mathcal{Y}_1^\perp}^{\top}(k)} \\ & & \ddots & \\ & & & \overline{\mathbf{v}_{\mathcal{Y}_1^\perp}^{\top}(j)} \end{pmatrix} \begin{pmatrix} \overline{\mathbf{v}_{\mathcal{Y}_2}^{\top}(1)} & \overline{\mathbf{v}_{\mathcal{Y}_2}^{\top}(2) & \dots & \overline{\mathbf{v}_{\mathcal{Y}_2}^{\top}(k)} \end{pmatrix} \right\| = \max_{1 \leq j \leq k} \|\overline{\mathbf{v}_{\mathcal{Y}_1^\perp}^{\top}(j)} \overline{\mathbf{v}_{\mathcal{Y}_2}}(j)\|$$

### C. Signal Decomposition

Recall that the gradient descent iterates are defined in [\(3.4\)](#page-3-1) as

$$\begin{aligned}
\mathcal{U}_{t+1} &= \mathcal{U}_t - \mu \nabla \ell(\mathcal{U}_t) \\
&= \mathcal{U}_t + \mu \mathcal{A}^* \left[ \mathbf{y} - \mathcal{A} \left( \mathcal{U}_t * \mathcal{U}_t^\top \right) \right] * \mathcal{U}_t \\
&= \left[ \mathcal{I} + \mu (\mathcal{A}^* \mathcal{A}) \left( \mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top \right) \right] * \mathcal{U}_t.
\end{aligned}$$

For the ground truth tensor X ∈ R n×r×k , consider its tensor-column subspace V<sup>X</sup> with the corresponding basis V<sup>X</sup> ∈ R n×r×k . Consider the tensor V<sup>X</sup> ∗ U<sup>t</sup> ∈ <sup>R</sup> <sup>r</sup>×R×<sup>k</sup> with its t-SVD decomposition V<sup>X</sup> ∗ U<sup>t</sup> = V<sup>t</sup> ∗ Σ<sup>t</sup> ∗ W<sup>⊤</sup> t . For W<sup>t</sup> ∈ <sup>R</sup> R×r×k , we denote by Wt,<sup>⊥</sup> ∈ <sup>R</sup> R×(n−r)×k a tensor whose tensor-column subspace is orthogonal to those of Wt, that is ∥W<sup>⊤</sup> t,<sup>⊥</sup> ∗ Wt∥ = 0 and its projection operator P<sup>W</sup>t,<sup>⊥</sup> is defined as P<sup>W</sup>t,<sup>⊥</sup> <sup>=</sup> Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> <sup>=</sup> I − W<sup>t</sup> ∗ W<sup>⊤</sup> t . We then decompose the gradient descent iterates U<sup>t</sup> as follows

$$\mathcal{U}_t = \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top + \mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top \quad (\text{C.1})$$

We will refer to the tensors Ut∗Wt∗W<sup>⊤</sup> t as the signal term of the gradient descent iterates, and the tensors Ut∗Wt,<sup>⊥</sup>∗W<sup>⊤</sup> t,⊥ will be named as the noise term.

Lemma C.1. *The tensor-column space of the noise term* U<sup>t</sup> ∗ Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> *is orthogonal to the tensor-column subspace of the* X *, namely* V ⊤ <sup>X</sup> ∗ U<sup>t</sup> ∗ Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> = 0*. Moreover, if* V ⊤ <sup>X</sup> ∗ U<sup>t</sup> *is full tubal-rank with all invertible singular tubes, then the signal term*

$$u_t * w_t * w_t^\top$$

*has tubal-rank* r *with all invertible singular tubes and the noise term has tubal rank at most* R − r*.*

*Proof.* V ⊤ <sup>X</sup> ∗ U<sup>t</sup> ∗ Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> = V ⊤ <sup>X</sup> ∗ U<sup>t</sup> ∗ (I − W<sup>t</sup> ∗ W<sup>⊤</sup> t ) = V ⊤ <sup>X</sup> ∗ U<sup>t</sup> − V ⊤ <sup>X</sup> ∗ U<sup>t</sup> ∗ W<sup>t</sup> ∗ W<sup>⊤</sup> <sup>t</sup> = 0 ∈ <sup>R</sup> r×R×k . The second part follows fact that if V ⊤ <sup>X</sup> ∗ U<sup>t</sup> is full tubal rank with all invertible singular tubes then all the slices in the Fourier have full rank.

## D. Analysis of the Spectral Stage

The goal of this section is to show that the first few iterations of the gradient descent algorithm can be approximated by the iteration of the tensor power method modulo normalization defined as

$$\tilde{\mathbf{u}}_t = \left( \mathcal{I} + \mu \mathcal{A}^* \mathcal{A}(\mathbf{x} * \mathbf{x}^\top) \right)^{*t} * \mathbf{u}_0 = \mathbf{z}_t * \mathbf{u}_0 \in \mathbb{R}^{n \times R \times k}.$$

with the tensor power method iteration <sup>Z</sup><sup>t</sup> =: I + µA<sup>∗</sup>A(X ∗ X ⊤) ∗t ∈ R n×n×k . Moreover, this will result in the feature that after the first few iterations, the tensor-column span of the signal term U<sup>t</sup> ∗ W<sup>t</sup> ∗ W<sup>⊤</sup> <sup>t</sup> becomes aligned with the tensor-column span of X , and that the noise term U<sup>t</sup> ∗ Wt,<sup>⊥</sup> is relatively small compared to signal term in terms of the norm, indicating that the signal term dominates the noise term.

For this, let us denote the difference between the power method and the gradient descent iterations by

$$\mathcal{E}_t := \mathcal{U}_t - \tilde{\mathcal{U}}_t. \quad (\text{D.1})$$

For convenience, throughout this section, we will denote by M the tensor M := A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>) ∈ <sup>R</sup> n×n×k , so that Ue<sup>t</sup> = (I <sup>+</sup> <sup>µ</sup>M) <sup>∗</sup><sup>t</sup> ∗ U<sup>0</sup> and Z<sup>t</sup> = (I + µM) ∗t .

In the first result of this section, the following lemma, we show that E<sup>t</sup> can be made small via an appropriate initialization scale.

Lemma D.1. *Suppose that* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies RIP*(2, δ1) *and let* t <sup>⋆</sup> *be defined as*

$$t^* = \min \left\{ j \in \mathbb{N}: \|\tilde{\mathbf{u}}_{j-1} - \mathbf{u}_{j-1}\| > \|\tilde{\mathbf{u}}_{j-1}\| \right\}. \quad (\text{D.2})$$

*Then for all integers* t *such that* 1 ≤ t ≤ t ⋆ *it holds that*

$$\|\boldsymbol{\varepsilon}_t\| = \|\boldsymbol{u}_t - \tilde{\boldsymbol{u}}_t\| \leq 8(1 + \delta_1\sqrt{k})\sqrt{k \min\{n, R\}} \frac{\alpha^3}{\|\boldsymbol{\mathcal{M}}\|} \|\boldsymbol{u}\|^3 (1 + \mu\|\boldsymbol{\mathcal{M}}\|)^{3t}. \quad (\text{D.3})$$

*Proof.* Similarly to the matrix case in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0), in the tubal tensor case it can be shown that for ¨ <sup>t</sup> ≥ <sup>1</sup>, the difference tensor E<sup>t</sup> <sup>=</sup> U<sup>t</sup> − Ue<sup>t</sup> can be represented as

$$\mathcal{E}_t = \mathcal{U}_t - \tilde{\mathcal{U}}_t = \sum_{j=1}^t (\mathcal{I} + \mu \mathcal{M})^{*(t-j)} \hat{\mathcal{E}}_j \quad (\text{D.4})$$

with Eb<sup>j</sup> <sup>=</sup> <sup>µ</sup>A<sup>∗</sup>A Uj−<sup>1</sup> ∗ U ⊤ j−1 ∗ Uj−1. To estimate ∥Et∥, we will first estimate each summand in [\(D.4\)](#page-12-1) separately. First, we can proceed with the following simple estimation

$$\|(\mathcal{I} + \mu\mathcal{M})^{*(t-j)}\widehat{\mathcal{E}}_j\| \leq \|(\mathcal{I} + \mu\mathcal{M})\|^{(t-j)}\|\widehat{\mathcal{E}}_j\| \leq (1 + \mu\|\mathcal{M}\|)^{(t-j)}\|\widehat{\mathcal{E}}_j\|.$$

Now, for ∥Ebj∥, using the fact that the spectral norm of tubal tensors is sub-multiplicative, we get that

$$\|\widehat{\varepsilon}_j\| = \mu \|\mathcal{A}^*\mathcal{A}(\mathbf{u}_{j-1} * \mathbf{u}_{j-1}^\top) * \mathbf{u}_{j-1}\| \leq \mu \|\mathcal{A}^*\mathcal{A}(\mathbf{u}_{j-1} * \mathbf{u}_{j-1}^\top)\| \cdot \|\mathbf{u}_{j-1}\|.$$

Since operator A satisfies RIP(2, δ1), by Lemma [G.3,](#page-50-0) A also satisfies S2NRIP(δ<sup>1</sup> √ k), which provides the following estimate

$$\|\mathcal{A}^*\mathcal{A}(\mathbf{u}_{j-1} * \mathbf{u}_{j-1}^\top)\| \leq (1 + \delta_1 \sqrt{k}) \|\mathbf{u}_{j-1} * \mathbf{u}_{j-1}^\top\|_* = (1 + \delta_1 \sqrt{k}) \|\mathbf{u}_{j-1}\|_F^2.$$

All this together leads to

$$\|\boldsymbol{\mathcal{E}}_t\| = \|\boldsymbol{\mathcal{U}}_t - \tilde{\boldsymbol{\mathcal{U}}}_t\| \leq \mu(1 + \delta_1 \sqrt{k}) \sum_{j=1}^t (1 + \mu\|\boldsymbol{\mathcal{M}}\|)^{(t-j)} \|\boldsymbol{\mathcal{U}}_{j-1}\|_F^2 \|\boldsymbol{\mathcal{U}}_{j-1}\|. \quad (\text{D.5})$$

From here, we want to bound ∥Et∥ in terms of the initialization scale α and the data-related norm ∥M∥. For this, we first use the fact that the tensor Frobenius norm above can be bounded as ∥Uj−1∥<sup>F</sup> ≤ p k min {n, R}∥Uj−1∥. Then since for all 1 ≤ j ≤ t <sup>⋆</sup> we have ∥Uej−<sup>1</sup> − Uj−1∥ ≤ ∥Uej−1∥, the spectral norm of Uj−<sup>1</sup> can be bounded as

$$\|\boldsymbol{u}_{j-1}\| \leq \|\tilde{\boldsymbol{u}}_{j-1}\| + \|\boldsymbol{u}_{j-1} - \tilde{\boldsymbol{u}}_{j-1}\| \leq 2\|\tilde{\boldsymbol{u}}_{j-1}\|.$$

This gives us the following upper bound

$$\|\boldsymbol{\mathcal{E}}_t\| \leq 8\mu(1 + \delta_1\sqrt{k})\sqrt{k \min \{n, R\}} \sum_{j=1}^t (1 + \mu\|\boldsymbol{\mathcal{M}}\|)^{t-j} \|\tilde{\mathbf{u}}_{j-1}\|^3. \quad (\text{D.6})$$

As for iterations of the tensor power method, it holds that

$$\|\widetilde{\mathcal{U}}_{j-1}\| = \|(\mathcal{I} + \mu\mathcal{M})^{*(j-1)} * \mathcal{U}_0\| \leq \|(\mathcal{I} + \mu\mathcal{M})^{*(j-1)}\| \|\mathcal{U}_0\| \leq (1 + \mu\|\mathcal{M}\|)^{j-1} \|\mathcal{U}_0\| = \alpha(1 + \mu\|\mathcal{M}\|)^{j-1} \|\mathcal{U}\|,$$

we can proceed with [\(D.6\)](#page-13-0) as follows

$$\|\mathcal{E}_t\| \leq 8\mu(1 + \delta_1\sqrt{k})\sqrt{k \min \{n, R\}}\alpha^3\|\mathcal{U}\|^3 \sum_{j=1}^t (1 + \mu\|\mathcal{M}\|)^{t+2j-3}.$$

Now, the sum on the right-hand side can be estimated as

$$\begin{aligned} \sum_{j=1}^t (1 + \mu\|\mathcal{M}\|)^{t+2j-3} &= (1 + \mu\|\mathcal{M}\|)^{t-1} \sum_{j=1}^t (1 + \mu\|\mathcal{M}\|)^{2j-2} = (1 + \mu\|\mathcal{M}\|)^{t-1} \frac{(1 + \mu\|\mathcal{M}\|)^{2t-1}}{(1 + \mu\|\mathcal{M}\|)^2 - 1} \\ &= (1 + \mu\|\mathcal{M}\|)^{t-1} \frac{(1 + \mu\|\mathcal{M}\|)^{2t-1}}{\mu\|\mathcal{M}\|(2 + \mu\|\mathcal{M}\|)} \leq \frac{(1 + \mu\|\mathcal{M}\|)^{3t}}{\mu\|\mathcal{M}\|}, \end{aligned}$$

which gives us the final estimation for the norm of E<sup>t</sup> as follows

$$\|\mathcal{E}_t\| \leq 8(1 + \delta_1 \sqrt{k}) \sqrt{k \min \{n, R\}} \frac{\alpha^3}{\|\mathcal{M}\|} \|\mathcal{U}\|^3 (1 + \mu \|\mathcal{M}\|)^{3t}$$

and finishes the proof.

The following lemma provides a lower bound for t ⋆ , indicating the duration for which the approximation in Lemma [D.1](#page-12-2) remains valid.

Lemma D.2. *Consider tensors* M := A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>) ∈ <sup>R</sup> <sup>n</sup>×n×<sup>k</sup> *and* Ue<sup>t</sup> := (I <sup>+</sup> <sup>µ</sup>M) <sup>∗</sup><sup>t</sup> ∗ U0*. Let* M ∈ <sup>C</sup> nk×nk *be the corresponding block diagonal form of the tensor* M *with the leading eigenvector* v<sup>1</sup> ∈ <sup>C</sup> nk*, then*

$$t^* \geq \left\lceil \frac{\ln \left( \frac{\|\mathcal{M}\| \cdot \|\bar{\mathcal{U}}_0^H v_1\|_{\ell_2}}{8(1+\delta_1 \sqrt{k}) \sqrt{k} \min \{n, R\} \alpha^3 \|\mathcal{U}\|^3} \right)}{2 \ln (1 + \mu \|\mathcal{M}\|)} \right\rceil \quad (\text{D.7})$$

*Proof.* Let Ue<sup>t</sup> ∈ <sup>C</sup> nk×Rk be the corresponding block diagonal form of tensor Uet. By the definition of the spectral tensor norm, we have ∥Uet∥ <sup>=</sup> ∥Uet∥ and the definition of the matrix norm gives ∥Uet∥ ≥ Ue<sup>t</sup> v1 ℓ2 . For the block diagonal version of Uet, the following properties (see, e.g., [\(Liu et al., 2019\)](#page-9-17)) holds

$$\overline{\bar{\mathcal{U}}}_t = \overline{(\mathcal{I} + \mu\mathcal{M})^{*t} * \mathcal{U}_0} = \overline{(\mathcal{I} + \mu\mathcal{M})^{*t}} \cdot \overline{\mathcal{U}_0} = \overline{(\mathcal{I} + \mu\mathcal{M})^t} \cdot \overline{\mathcal{U}_0}. \quad (\text{D.8})$$

This allows us to proceed as follows

$$\widetilde{\mathcal{U}}_t^{\text{H}} v_1 = \left( \overline{(\mathcal{I} + \mu \mathcal{M})}^t \cdot \overline{\mathcal{U}}_0 \right)^{\text{H}} v_1 = \overline{\mathcal{U}}_0^{\text{H}} \overline{(\mathcal{I} + \mu \mathcal{M})}^{\text{tH}} v_1 = (1 + \mu \|\mathcal{M}\|)^t \overline{\mathcal{U}}_0^{\text{H}} v_1,$$

where for the last equality we used the fact that block-diagonal matrix (I + µM) has the same set of eigenvectors as matrix M. From here, we get ∥Uet∥ ≥ Ue<sup>t</sup> H v1 ℓ2 = (1 + µ∥M∥) t U<sup>0</sup> H v1 ℓ2 . Then, applying Lemma [D.1,](#page-12-2) the relative error in the spectral norm between Ue<sup>t</sup> and U<sup>t</sup> can be estimated as

$$\frac{\|\tilde{u}_t - u_t\|}{\|\tilde{u}_t\|} \leq 8(1 + \delta_1 \sqrt{k}) \frac{\sqrt{k \min \{n, R\}} \alpha^3}{\|\mathcal{M}\| \cdot \|\tilde{u}_0^{\text{H}} v_1\|_{\ell_2}} \|u\|^3 (1 + \mu \|\mathcal{M}\|)^{2t}.$$

Setting the bound above to be smaller than 1 and solving for t, we get

$$t < \frac{\ln \left( \frac{\|\mathcal{M}\| \cdot \|\bar{\mathcal{U}}_0^H v_1\|_{\ell_2}}{8(1+\delta_1 \sqrt{k}) \sqrt{k \min \{n, R\} \alpha^3 \|\mathcal{U}\|^3}} \right)}{2 \ln(1 + \mu \|\mathcal{M}\|)}.$$

Since t ∈ N with t ≤ t ⋆ should be such that <sup>∥</sup>Uet−1−Ut−1<sup>∥</sup> ∥Uet−1∥ < 1, we can choose t ⋆ as the floor-value of the right-hand side above.

To show that the tensor column subspaces of the tensor power method iterates and the gradient descent iterates are aligned after the alignment phase, we use the largest principal angle between two tensor-column subspaces as the potential function for analysis. Borrowing the idea from [\(Gleich et al., 2013\)](#page-9-21), we will show that the power method iteration in the tensor domain can be transformed to the classical subspace iteration in the frequency domain.

For this, consider the power method iterates Ue<sup>t</sup> = (I <sup>+</sup> <sup>µ</sup>M) <sup>∗</sup><sup>t</sup> ∗ U0, the iterates Z<sup>t</sup> = (I + µM) ∗t and the gradient descent iterates U<sup>t</sup> represented as U<sup>t</sup> <sup>=</sup> Ue<sup>t</sup> <sup>+</sup> E<sup>t</sup> <sup>=</sup> Z<sup>t</sup> ∗ U<sup>0</sup> <sup>+</sup> Et. All these tensors have their counterparts in the Fourier domain, which we will denote respectively as Uet, Z<sup>t</sup> and Ut.

As before, consider M = A<sup>∗</sup>A(X ∗ X <sup>⊤</sup>) ∈ <sup>R</sup> <sup>n</sup>×n×<sup>k</sup> with its t-SVD M = V<sup>M</sup> ∗ Σ<sup>M</sup> ∗ W<sup>⊤</sup> <sup>M</sup> and its Fourier domain representative M ∈ C nk×nk. We denote by L ∈ <sup>R</sup> n×r×k the tensor column subspace spanned by the tensor columns corresponding to the first r singular tubes, that is L := VM(:, 1 : r, :) ∈ <sup>R</sup> n×r×k . Note that L is also the subspace spanned by the tensor columns corresponding to the first r singular tubes of the tensor Z<sup>t</sup> ∈ <sup>R</sup> n×n×k .

By L<sup>t</sup> ∈ <sup>R</sup> <sup>n</sup>×n×<sup>k</sup> we will donate the tensor-column subspace spanned by the tensor columns corresponding to the first r singular tubes of the gradient descent iterates U<sup>t</sup> = Z<sup>t</sup> ∗ U<sup>0</sup> + Et. More concretely, for U<sup>t</sup> = P<sup>R</sup> <sup>s</sup>=1 V<sup>U</sup><sup>t</sup> (:, s, :) ∗ ΣU<sup>t</sup> (s, s, :) ∗ W<sup>⊤</sup> U<sup>t</sup> (:, s, :) and the corresponding Fourier domain representation U<sup>t</sup> = diag(U<sup>t</sup> (1) ,U<sup>t</sup> (2) , . . . ,U<sup>t</sup> (k) ), where U<sup>t</sup> (j) = P ℓ σ (j) ℓ v (j) <sup>ℓ</sup> w (j) ℓ H = U (j) U<sup>t</sup> Σ (j) <sup>U</sup><sup>t</sup> W (j) U<sup>t</sup> H , we define the corresponding new tensors L<sup>t</sup> := V<sup>U</sup><sup>t</sup> (:, 1 : r, :) ∈ <sup>R</sup> n×r×k and their Fourier domain representations

$$\bar{\mathbf{L}}_t = \text{diag}(\bar{L}_t^{(1)}, \bar{L}_t^{(2)}, \dots, \bar{L}_t^{(k)}) \quad (\text{D.9})$$

Lemma D.3. *Consider the tensor iterates* Z<sup>t</sup> = (I + µM) <sup>∗</sup><sup>t</sup> *with its block-matrix representation*

$$\overline{\mathcal{Z}}_t = bdiag(\mathcal{Z}_t) = diag(\overline{Z}_t^{(1)}, \overline{Z}_t^{(2)}, \dots, \overline{Z}_t^{(k)}). \quad (\text{D.10})$$

*and the tensors*

$$\begin{aligned} \mathcal{E}_t &= \mathcal{U}_t - \tilde{\mathcal{U}}_t \in \mathbb{R}^{n \times R \times k} \\ \mathcal{U}_0 &= \alpha \mathcal{U} \in \mathbb{R}^{n \times R \times k}, \quad \alpha > 0. \end{aligned}$$

*Assume that for each* 1 ≤ j ≤ k*, it holds that*

$$\sigma_{r+1}(\overline{Z}_t^{(j)})\|\mathbf{U}\| + \frac{\|\boldsymbol{\varepsilon}_t\|}{\alpha} < \sigma_r(\overline{Z}_t^{(j)})\sigma_{min}(\overline{\mathbf{y}_{\boldsymbol{L}}^\top * \mathbf{U}}). \quad (\text{D.11})$$

*Then for each* 1 ≤ j ≤ k*, the following two inequalities hold*

$$\sigma_r(\overline{U}_t^{(j)}) = \sigma_r(\overline{Z}_t^{(j)}\overline{U}_0^{(j)} + \overline{E}_t^{(j)}) \geq \alpha\sigma_r(\overline{Z}_t^{(j)})\sigma_{min}(\overline{\mathbf{V}}_{\mathcal{L}}^{\top} * \mathbf{U}) - \|\mathcal{E}_t\|, \quad (\text{D.12})$$

$$\sigma_{r+1}(\overline{U}_t^{(j)}) = \sigma_{r+1}(\overline{Z}_t^{(j)}\overline{U}_0^{(j)} + \overline{E}_t^{(j)}) \leq \alpha \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\mathbf{U}\| + \|\mathbf{E}_t\| \quad (\text{D.13})$$

*Moreover, the principal angle between the tensor-column subspaces* L *and* L<sup>t</sup> *is bounded as follows*

$$\|\mathbf{v}_{\mathcal{L}^\perp}^\top * \mathbf{v}_{\mathcal{L}_t}\| \leq \max_{1 \leq j \leq k} \frac{\alpha \sigma_{r+1}(\overline{\mathbf{Z}}_t^{(j)}) \|\mathbf{u}\| + \|\boldsymbol{\varepsilon}_t\|}{\sigma_r(\overline{\mathbf{Z}}_t^{(j)}) \sigma_{min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u}}) - \alpha \sigma_{r+1}(\overline{\mathbf{Z}}_t^{(j)}) \|\mathbf{u}\| - \|\boldsymbol{\varepsilon}_t\|} \quad (\text{D.14})$$

*Proof.* For some t ∈ <sup>N</sup>, consider tensor Z<sup>t</sup> = (I + µM) <sup>∗</sup><sup>t</sup> with its block-matrix representation

$$\overline{\mathcal{Z}}_t = \text{bdiag}(\mathcal{Z}_t) = \text{diag}(\overline{Z}_t^{(1)}, \overline{Z}_t^{(2)}, \dots, \overline{Z}_t^{(k)}) = \begin{pmatrix} \overline{Z}_t^{(1)} & & & \\ & \overline{Z}_t^{(2)} & & \\ & & \ddots & \\ & & & \overline{Z}_t^{(k)} \end{pmatrix}.$$

As we assume the symmetric tensor case scenario, the block-diagonal matrix representation Z<sup>t</sup> consists of symmetric matrices Z<sup>t</sup> (j) ∈ <sup>C</sup> <sup>n</sup>×<sup>n</sup>. At the same time, according to [\(Gleich et al., 2013\)](#page-9-21), the gradient descent tensors U<sup>t</sup> = Z<sup>t</sup> ∗U0+E<sup>t</sup> have their block-diagonal matrix representation

$$\mathcal{U}_t = \mathcal{Z}_t * \mathcal{U}_0 + \mathcal{E}_t \Leftrightarrow \overline{\mathcal{Z}}_t \overline{\mathcal{U}}_0 + \overline{\mathcal{E}}_t = \begin{pmatrix} \overline{\mathcal{Z}}_t^{(1)} \overline{\mathcal{U}}_0^{(1)} & & & & & & & & \\ & \overline{\mathcal{Z}}_t^{(2)} \overline{\mathcal{U}}_0^{(2)} & & & & & & & \\ & & \ddots & & & & & & \\ & & & \overline{\mathcal{Z}}_t^{(k)} \overline{\mathcal{U}}_0^{(k)} & & & & & \\ & & & & \ddots & & & & \\ & & & & & \ddots & & & \\ & & & & & & \ddots & & \\ & & & & & & & \ddots & \\ & & & & & & & & \overline{\mathcal{E}}_t^{(k)} \end{pmatrix}. \quad (\text{D.15})$$

Using Weyl's inequality in each block, we have

$$\sigma_r(\overline{Z}_t^{(j)}\overline{U}_0^{(j)} + \overline{E}_t^{(j)}) \geq \sigma_r(\overline{Z}_t^{(j)}\overline{U}_0^{(j)}) - \|\overline{E}_t^{(j)}\| \geq \sigma_r\left((\overline{V}_{\mathcal{L}}^{(j)})^{\text{H}}\overline{Z}_t^{(j)}\overline{U}_0^{(j)}\right) - \|\overline{E}_t^{(j)}\|.$$

Now, for the singular value above we get the following estimation

$$\begin{aligned} \sigma_r\left(\overline{(V_{\mathcal{L}}^{(j)})^H \overline{Z_t}^{(j)} \overline{U_0}^{(j)}}\right) &= \sigma_{min}\left(\overline{V_{\mathcal{L}}^{(j)}}^H \overline{Z_t}^{(j)} V_{\mathcal{L}}^{(j)} V_{\mathcal{L}}^{(j)} \overline{U_0}^{(j)}\right) \\ &\geq \sigma_{min}\left(\overline{V_{\mathcal{L}}^{(j)}}^H \overline{Z_t}^{(j)} \overline{V_{\mathcal{L}}^{(j)}}\right) \sigma_{min}\left(\overline{V_{\mathcal{L}}^{(j)}}^H \overline{U_0}^{(j)}\right) \\ &= \sigma_r(\overline{Z_t}^{(j)}) \sigma_{min}\left(\overline{V_{\mathcal{L}}^{(j)}}^H \overline{U_0}^{(j)}\right) \geq \alpha \sigma_r(\overline{Z_t}^{(j)}) \sigma_{min}\left(\overline{V_{\mathcal{L}}^{(j)}}^H \overline{U}^{(j)}\right) \\ &= \alpha \sigma_r(\overline{Z_t}^{(j)}) \sigma_{min}\left(\overline{V_{\mathcal{L}}^H}^{(j)} \overline{U}^{(j)}\right) \geq \alpha \sigma_r(\overline{Z_t}^{(j)}) \sigma_{min}\left(\overline{\mathbf{v}_{\mathcal{L}}^{\top} * \mathbf{U}}\right) \end{aligned}$$

where in the last line we used that for each tensor it holds in the Fourier domain V<sup>L</sup> (j)H = V T L (j) .

To show inequality [\(D.13\)](#page-15-0), we can use Weyl's bounds and then the Courant-Fisher theorem, which leads to

$$\begin{aligned} \sigma_{r+1}(\overline{Z}_t^{(j)}\overline{U}_0^{(j)} + \overline{E}_t^{(j)}) &\leq \sigma_{r+1}(\overline{Z}_t^{(j)}\overline{U}_0^{(j)}) + \|\overline{E}_t^{(j)}\| \leq \sigma_{r+1}(\overline{Z}_t^{(j)}\overline{U}_0^{(j)}) + \|\boldsymbol{\varepsilon}_t\| \\ &\leq \sigma_{r+1}(\overline{Z}_t^{(j)})\|\overline{U}_0^{(j)}\| + \|\boldsymbol{\varepsilon}_t\| \leq \alpha\sigma_{r+1}(\overline{Z}_t^{(j)})\|\boldsymbol{u}\| + \|\boldsymbol{\varepsilon}_t\|. \end{aligned}$$

Now, for estimation of ∥V ⊥ <sup>L</sup> ∗ V<sup>L</sup><sup>t</sup> ∥, let us recall that L is the tensor column subspace spanned by the tensor columns corresponding to the first r singular tubes of tensor Z<sup>t</sup> = (I − µM) <sup>∗</sup><sup>t</sup> ∈ <sup>R</sup> n×n×k , and L<sup>t</sup> is the tensor-column subspace spanned by the tensor-columns corresponding to the first r singular tubes of the gradient descent iterates U<sup>t</sup> = Z<sup>t</sup> ∗U<sup>0</sup> +Et, and consider Fourier-domain representation [\(D.15\)](#page-15-1) of Ut. Here, for each 1 ≤ j ≤ k, the matrices Z<sup>t</sup> (j)U<sup>0</sup> (j) + E<sup>t</sup> (j) can be represented as

$$\underbrace{\overline{Z}_t(j)\overline{U}_0(j) + \overline{E}_t(j)}_{\widetilde{A}(j)} = \underbrace{\overline{Z}_t(j)\overline{V}_{\mathcal{L}}(j)\overline{V}_{\mathcal{L}}(j)^H\overline{U}_0(j)}_{A(j)} + \underbrace{\overline{Z}_t(j)\overline{V}_{\mathcal{L}^\perp}(j)\overline{V}_{\mathcal{L}^\perp}(j)^H\overline{U}_0(j) + \overline{E}_t(j)}_{C(j)}. \quad (\text{D.16})$$

As the tensor-column space V<sup>L</sup> is r-dimensional, each of matrices V<sup>L</sup> (j) has rank r, see [\(Gleich et al., 2013\)](#page-9-21). Since the matrices Z<sup>t</sup> (j) can be decomposed as

$$\overline{Z}_t(j) = \overline{V}_{\mathcal{L}}(j) \Sigma_{\mathcal{L}}^{(j)} \overline{V}_{\mathcal{L}}(j)^{\text{H}} + \overline{V}_{\mathcal{L}^{\perp}}(j) \Sigma_{\mathcal{L}^{\perp}}^{(j)} \overline{V}_{\mathcal{L}^{\perp}}(j)^{\text{H}}$$

we have that

$$\overline{Z}_t^{(j)} \overline{V}_{\mathcal{L}}^{(j)} \overline{V}_{\mathcal{L}}^{(j)} {}^{\text{H}} \overline{U}_0^{(j)} = \overline{V}_{\mathcal{L}}^{(j)} \Sigma_{\mathcal{L}}^{(j)} \overline{V}_{\mathcal{L}}^{(j)} \overline{U}_0^{(j)}. \quad (\text{D.17})$$

As U<sup>0</sup> (j) ∈ <sup>C</sup> <sup>r</sup>×<sup>R</sup> has rank r, V<sup>L</sup> (j)<sup>H</sup> U<sup>0</sup> (j) has rank r, which means that the product above has rank r too. Due to [\(D.17\)](#page-16-0), we see that

$$\overline{Z}_t^{(j)} \overline{V}_{\mathcal{L}}^{(j)} \overline{V}_{\mathcal{L}}^{(j)^{\text{H}}} \overline{U}_0^{(j)} = \overline{V}_{\mathcal{L}}^{(j)} \overline{V}_{\mathcal{L}}^{(j)^{\text{H}}} \overline{Z}_t^{(j)} \overline{V}_{\mathcal{L}}^{(j)} \overline{V}_{\mathcal{L}}^{(j)^{\text{H}}} \overline{U}_0^{(j)},$$

which makes V<sup>L</sup> (j) to the column subspace of Z<sup>t</sup> (j)V<sup>L</sup> (j)V<sup>L</sup> (j)<sup>H</sup> U<sup>0</sup> (j) . Considering the gap between the singular values of for matrices A(j) and <sup>A</sup>e(j) in [\(D.16\)](#page-16-1), namely δ (j) = σr(A(j) ) − <sup>σ</sup>r+1(Ae(j) ), and using Wedin's sin θ theorem [\(Wedin,](#page-10-9) [1972\)](#page-10-9), for each 1 ≤ j ≤ k we get

$$\|\overline{V_{\mathcal{L}^{\perp}}(j)}^{\text{H}}\overline{V_{\mathcal{L}_t}(j)}\| \leq \frac{\|C^{(j)}\|}{\delta(j)}.$$

To conduct a further estimation of ∥VL<sup>⊥</sup> (j)<sup>H</sup> VL<sup>t</sup> (j)∥, we analyze lower and upper bounds for the denominator and the numerator above. We start with the denominator first

$$\begin{aligned}\delta^{(j)} &= \sigma_r(A^{(j)}) - \sigma_{r+1}(\tilde{A}^{(j)}) \\ &= \sigma_r(\overline{Z_t}^{(j)} \overline{V_{\mathcal{L}}}^{(j)} \overline{V_{\mathcal{L}}}^{(j)} \overline{U_0}^{(j)}) - \sigma_{r+1}(\overline{Z_t}^{(j)} \overline{U_0}^{(j)} + \overline{E_t}^{(j)}).\end{aligned}$$

Using properties of singular values of the matrix product for the first term above and Weyl's bound for the second term, we get

$$\begin{aligned} \delta^{(j)} &\geq \sigma_r(\overline{Z}_t^{(j)})\sigma_{min}\left(\overline{V}_{\mathcal{L}}^{(j)} \overline{U}_0^{(j)}\right) - \sigma_{r+1}\left(\overline{Z}_t^{(j)} \overline{U}_0^{(j)}\right) - \|\overline{E}_t^{(j)}\| \\ &\geq \sigma_r(\overline{Z}_t^{(j)})\sigma_{min}\left(\overline{\mathbf{v}}_{\mathcal{L}}^{\top} * \overline{\mathbf{u}}_0\right) - \sigma_{r+1}\left(\overline{Z}_t^{(j)} \overline{U}_0^{(j)}\right) - \|\mathbf{E}_t\|. \end{aligned} \quad (\text{D.18})$$

For the norm of C (j) , the following upper bound can be established

$$\begin{aligned}
\|C^{(j)}\| &\leq \|\overline{Z}_t^{(j)} \overline{V}_{\mathcal{L}^{\perp}}^{(j)} \overline{V}_{\mathcal{L}^{\perp}}^{(j)^{\text{H}}} \overline{U}_0^{(j)}\| + \|E_t^{(j)}\| \\
&\leq \|\overline{Z}_t^{(j)} \overline{V}_{\mathcal{L}^{\perp}}^{(j)} \overline{V}_{\mathcal{L}^{\perp}}^{(j)^{\text{H}}}\| \|\overline{U}_0^{(j)}\| + \|\boldsymbol{\varepsilon}_t\| \\
&\leq \alpha \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\boldsymbol{u}\| + \|\boldsymbol{\varepsilon}_t\| \tag{D.19}
\end{aligned}$$

Now, combining bounds [\(D.18\)](#page-16-2) and [\(D.19\)](#page-16-3), one obtains that

$$\|\mathbf{v}_{\mathcal{L}^\perp}^\top * \mathbf{v}_{\mathcal{L}_t}\| = \max_{1 \leq j \leq k} \|\overline{\mathbf{V}_{\mathcal{L}^\perp}^{(j)}} \overline{\mathbf{V}_{\mathcal{L}_t}^{(j)}}\| \leq \max_{1 \leq j \leq k} \frac{\alpha \sigma_{r+1}(\overline{Z_t}^{(j)}) \|\mathbf{u}\| + \|\boldsymbol{\varepsilon}_t\|}{\sigma_r(\overline{Z_t}^{(j)}) \sigma_{min}(\overline{\mathbf{v}_{\mathcal{L}^\perp}^\top * \mathbf{u}}) - \sigma_{r+1}(\overline{Z_t}^{(j)} \overline{U}^{(j)}) - \|\boldsymbol{\varepsilon}_t\|} :$$

Further, we consider the gradient descent iterates with its t-SVD

$$\mathcal{U}_t = \sum_{s=1}^R \mathcal{V}_{\mathcal{U}_t}(:, s, :) * \Sigma_{\mathcal{U}_t}(s, s, :) * \mathcal{W}_{\mathcal{U}_t}^\top(:, s, :)$$

and the corresponding Fourier domain representation U<sup>t</sup> = diag(U<sup>t</sup> (1) ,U<sup>t</sup> (2) , . . . ,U<sup>t</sup> (k) ), where U<sup>t</sup> (j) = P<sup>R</sup> <sup>ℓ</sup>=1 σ (j) ℓ v (j) <sup>ℓ</sup> w (j) ℓ H = V (j) U<sup>t</sup> Σ (j) <sup>U</sup><sup>t</sup> W (j)H U<sup>t</sup> and its signal-noise term decomposition

$$\mathcal{U}_t = \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top + \mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top.$$

We also define the corresponding new tensors

$$\mathcal{L}_t = \sum_{s=1}^r \mathcal{V}_{\mathcal{U}_t}(:, s, :) * \Sigma_{\mathcal{U}_t}(s, s, :) * \mathcal{W}_{\mathcal{L}_t}^\top(:, s, :) \quad (\text{D.20})$$

$$\mathcal{N}_t = \sum_{s=r+1}^R \mathcal{V}_{\mathcal{U}_t}(:, s, :) * \Sigma_{\mathcal{U}_t}(s, s, :) * \mathcal{W}_{\mathcal{U}_t}^\top(:, s, :) \quad (\text{D.21})$$

and their Fourier domain representations

$$\mathcal{E}_t = \text{diag}(\bar{L}_t^{(1)}, \bar{L}_t^{(2)}, \dots, \bar{L}_t^{(k)}), \quad \bar{L}_t^{(j)} = \sum_{\ell=1}^r \sigma_{\ell}^{(j)} v_{\ell}^{(j)} w_{\ell}^{(j)} \mathbf{1}^{\text{H}} = V_{\mathcal{E}_t}^{(j)} \Sigma_{\mathcal{E}_t}^{(j)} W_{\mathcal{E}_t}^{(j) \text{H}} \quad (\text{D.22})$$

$$\mathcal{M}_t = \text{diag}(\overline{N}_t^{(1)}, \overline{N}_t^{(2)}, \dots, \overline{N}_t^{(k)}), \quad \overline{N}_t^{(j)} = \sum_{\ell=r+1}^R \sigma_{\ell}^{(j)} v_{\ell}^{(j)} w_{\ell}^{(j)^{\text{H}}} = V_{\mathcal{M}_t}^{(j)} \Sigma_{\mathcal{M}_t}^{(j)} W_{\mathcal{M}_t}^{(j)^{\text{H}}} \quad (\text{D.23})$$

Lemma D.4. *Assume* ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>L</sup><sup>t</sup> ∥ ≤ <sup>1</sup> 2 *. Then it holds that*

$$\| \mathcal{W}_{\mathcal{L}_t^\perp}^\top * \mathcal{W}_t \| \leq 2 \max_{1 \leq j \leq k} \frac{\sigma_{r+1} \left( \overline{U_t}^{(j)} \right)}{\sigma_r \left( \overline{U_t}^{(j)} \right)} \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{L}_t} \|. \quad (\text{D.24})$$

*Proof.* Consider ∥W<sup>T</sup> L<sup>⊥</sup> ∗ Wt∥ = max1≤j≤<sup>k</sup> ∥WL<sup>⊥</sup> (j)<sup>H</sup>W<sup>t</sup> (j)∥. For each 1 ≤ j ≤ k, we can now exploit the results of Lemma A.1 in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0), to get that ¨

$$\|(\overline{W_{\mathcal{L}_t^\perp}^\top})^{(j)} \overline{W_t^{(j)}}\| \leq \frac{\|\Sigma_{\mathcal{N}_t}^{(j)} \| \|\overline{V_{\mathcal{N}_t^H}^{(j)} \overline{V_{\mathcal{X}}^{(j)}}\|}{\sigma_{\min}\left(\overline{V_{\mathcal{X}}^{(j)} \overline{U_t^{(j)}}}\right)} \quad \text{and} \quad \sigma_{\min}(\overline{V_{\mathcal{X}}^{(j)} \overline{U_t^{(j)}}}) \geq \frac{\sigma_{\min}(\overline{L_t^{(j)}})}{2}.$$

From here, we can proceed as follows

$$\begin{aligned} \|\mathcal{W}_{\mathcal{L}_t^\perp}^\top * \mathcal{W}_t\| &= \max_{1 \leq j \leq k} \left\| \overline{W_{\mathcal{L}_t^\perp}^{H_t^\top(j)} \overline{W_t^{(j)}}} \right\| \leq 2 \max_{1 \leq j \leq k} \frac{\left\| \frac{\Sigma_{\mathcal{N}_t^j}^{(j)}}{\sigma_{\min}(\overline{U_t^{(j)})}} \right\| \overline{V_{\mathcal{N}_t^j}^{(j)} \overline{V_{\mathcal{X}^{(j)}}}}}{\sigma_{\min}(\overline{U_t^{(j)}})} \\ &= 2 \max_{1 \leq j \leq k} \frac{\sigma_{r+1}(\overline{U_t^{(j)}}) \left\| \overline{V_{\mathcal{N}_t^j}^{(j)} \overline{V_{\mathcal{X}^{(j)}}}} \right\|}{\sigma_r(\overline{U_t^{(j)}})} \leq 2 \max_{1 \leq j \leq k} \frac{\sigma_{r+1}(\overline{U_t^{(j)}})}{\sigma_r(\overline{U_t^{(j)}})} \left\| \mathcal{V}_{\mathcal{L}_t^\perp}^\top * \mathcal{V}_{\mathcal{X}} \right\| \\ &= 2 \max_{1 \leq j \leq k} \frac{\sigma_{r+1}(\overline{U_t^{(j)}})}{\sigma_r(\overline{U_t^{(j)}})} \left\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{L}_t} \right\|, \end{aligned}$$

which concludes the proof.

Lemma D.5. *Assume that* ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>L</sup><sup>t</sup> ∥ ≤ <sup>1</sup> 8 *for some* t ≥ 1, t ∈ N*. Then for each* 1 ≤ j ≤ k*, it holds that*

$$\sigma_r\left(\overline{\boldsymbol{U}}_t * \overline{\boldsymbol{W}}_t^{(j)}\right) \geq \frac{1}{2}\sigma_r\left(\overline{\boldsymbol{U}}_t^{(j)}\right) \quad (\text{D.25})$$

$$\sigma_1(\overline{\mathcal{U}_t * \mathcal{W}_{t,\perp}}^{(j)}) \leq 2\sigma_{r+1}(\overline{U_t}^{(j)}). \quad (\text{D.26})$$

*Moreover, the principal angles between the tensor-column subspaces spanned by* X *and* UtW<sup>t</sup> *can be estimated as follows*

$$\|\boldsymbol{\nu}_{x^\perp} * \boldsymbol{\nu}_{u_t} w_t\| \leq 7 \|\boldsymbol{\nu}_{x^\perp}^\top * \boldsymbol{\nu}_{c_t}\| \quad (\text{D.27})$$

$$\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\| \leq 2 \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{U}_t^{(j)}). \quad (\text{D.28})$$

*Proof.* We assume that ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>L</sup><sup>t</sup> ∥ ≤ <sup>1</sup> 8 , then due to Lemma [D.4,](#page-17-0) we obtain that

$$\|\mathcal{W}_{\mathcal{L}_t^\top}^\top * \mathcal{W}_t\| \leq 2 \max_{1 \leq j \leq k} \frac{\sigma_{r+1}(\overline{U_j}^{(j)})}{\sigma_r(\overline{U_j}^{(j)})} \|\mathcal{V}_{\mathcal{X}^\top}^\top * \mathcal{V}_{\mathcal{L}_t}\| \leq \frac{1}{4}. \quad (\text{D.29})$$

Now, to estimate σ<sup>r</sup> U<sup>t</sup> ∗ W<sup>t</sup> (j) , we see that for each 1 ≤ j ≤ k, it holds that

$$\sigma_r\left(\overline{\mathbf{u}}_t * \overline{\mathbf{W}}_t^{(j)}\right)^2 = \sigma_r\left(\left(\overline{\mathbf{u}}_t * \overline{\mathbf{W}}_t^{(j)}\right)^H \overline{\mathbf{u}}_t * \overline{\mathbf{W}}_t^{(j)}\right) = \sigma_r\left(\overline{W}_t^{(j)H} \overline{U}_t^{(j)H} \overline{U}_t^{(j)} \overline{W}_t^{(j)}\right) \quad (\text{D.30})$$

Since U<sup>t</sup> (j)<sup>H</sup> U<sup>t</sup> (j) = L<sup>t</sup> (j)<sup>H</sup> L<sup>t</sup> (j) + N<sup>t</sup> (j)<sup>H</sup> N<sup>t</sup> (j) , we get that

$$\begin{aligned} \sigma_r \left( \overline{\mathbf{u}_t} * \overline{\mathbf{W}_t}^{(j)} \right)^2 &\geq \sigma_r \left( \overline{W_t}^{(j) \text{H}} \overline{L_t}^{(j) \text{H}} \overline{L_t}^{(j)} \overline{W_t}^{(j)} \right) = \sigma_r \left( \overline{W_t}^{(j) \text{H}} \overline{L_t}^{(j)} \right)^2 \\ &\geq \sigma_r \left( \overline{W_t}^{(j) \text{H}} W_{\overline{L_t}^{(j)}} \right)^2 \sigma_r \left( \overline{L_t}^{(j)} \right)^2 \geq (1 - \|\mathbf{W}_{\mathcal{L}_t^\perp} * \mathbf{W}_t^T\|^2) \sigma_r \left( \overline{U_t}^{(j)} \right)^2, \end{aligned}$$

where in the last line we used the definition of the principal angle between tensor column subspaces and the corresponding properties in their Fourier domain slices, namely

$$\sigma_r(\overline{W}_t^{(j)H}W_{\overline{L}_t^{(j)}})^2 = 1 - \|\overline{W}_t^{(j)H}W_{\overline{L}_t^{(j)}}^\perp\|^2 \geq 1 - \max_{1 \leq j \leq k} \|\overline{W}_t^{(j)H}W_{\overline{L}_t^{(j)}}^\perp\|^2 = 1 - \|\mathcal{W}_{\mathcal{L}_t^\perp} * \mathcal{W}_t^T\|^2.$$

Due to our assumption ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>L</sup><sup>t</sup> ∥ ≤ <sup>1</sup> 8 , we can see that in the Fourier domain, the subspaces spanned by V (j) X <sup>⊥</sup> and V (j) L<sup>t</sup> = V<sup>L</sup><sup>t</sup> (j) are close enough. Then, decomposing U<sup>t</sup> (j) into two different ways, namely as

$$\overline{U}_t^{(j)} = \sum_{\ell=1}^R \sigma_\ell^{(j)} v_\ell^{(j)} w_\ell^{(j)} \mathbf{1}^H = \overline{L}_t^{(j)} + \overline{N}_t^{(j)}$$

and as

$$\overline{U}_t^{(j)} = \overline{U}_t^{(j)} \overline{W}_t^{(j)} \overline{W}_t^{(j)}^{\text{H}} + \overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)} \overline{W}_{t,\perp}^{(j)}^{\text{H}},$$

according to Lemma [H.1,](#page-51-1) one obtains for each 1 ≤ j ≤ k that

$$\begin{aligned} \| \overline{V}_{\mathcal{X}_t^\perp}^{(j)} {}^{\text{H}} V_{\overline{U}_t^{(j)}} \overline{W}_{t^{(j)}} \| &\leq 7 \| \overline{V}_{\mathcal{X}_t^\perp}^{(j)} \overline{V}_{\mathcal{L}_t^{(j)}} \| \\ \| \overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)} \| &\leq 2 \sigma_{r+1} (\overline{U}_t^{(j)}), \end{aligned}$$

where the last inequality is equivalent to σ1(U<sup>t</sup> ∗ Wt,<sup>⊥</sup> (j) ) ≤ 2σr+1(U<sup>t</sup> (j) ). According to the definition of principal angles between tensor subspaces, this implies that

$$\| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{U}_t * \mathbf{w}_t} \| = \max_j \| \bar{V}_{\mathcal{X}_t^\perp}^{(j)} \mathbf{H} V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}} \| \leq 7 \max_j \| \bar{V}_{\mathcal{X}_t^\perp}^{(j)} \mathbf{H} \bar{V}_{\mathcal{L}_t}^{(j)} \| = 7 \| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{L}_t} \|.$$

In the same way, ∥U<sup>t</sup> ∗ Wt,<sup>⊥</sup>∥ = max<sup>j</sup> ∥U<sup>t</sup> (j)Wt,<sup>⊥</sup> (j)∥ ≤ 2 max<sup>j</sup> σr+1(U<sup>t</sup> (j) ), which finishes the proof.

Lemma D.6. *Consider a tensor* T := X ∗ X <sup>⊤</sup> ∈ S n×n×k <sup>+</sup> *with tubal rank* r ≤ n*. Assume that measurement operator* A *is such that*

$$\mathcal{M} = \mathcal{A}^* \mathcal{A}(\mathcal{T}) = \mathcal{T} + \mathcal{E} \quad \in S_+^{n \times n \times k}$$

*and for for each* 1 ≤ j ≤ k *one has* ∥E(j)∥ ≤ δλr(T (j) ) *with* δ ≤ 1 4 *. For the same* M *with its t-SVD* M = V<sup>M</sup> ∗ Σ<sup>M</sup> ∗ W<sup>⊤</sup> <sup>M</sup>*, let* L ∈ <sup>R</sup> <sup>n</sup>×r×<sup>k</sup> *denote the tensor column subspace spanned by the tensor-columns corresponding to the first* r *singular tubes, that is* L := VM(:, 1 : r, :) ∈ <sup>R</sup> n×r×k *.*

*Then, in each Fourier slice* j*,* 1 ≤ j ≤ k*, it holds that*

$$(1 - \delta)\lambda_1(\bar{T}^{(j)}) \leq \lambda_1(\bar{M}^{(j)}) \leq (1 + \delta)\lambda_1(\bar{T}^{(j)}) \quad (\text{D.31})$$

$$\lambda_{r+1}(\overline{M}^{(j)}) \leq \delta \lambda_r(\overline{T}^{(j)}) \quad (\text{D.32})$$

$$\lambda_r(\bar{M}^{(j)}) \geq (1 - \delta)\lambda_r(\bar{T}^{(j)}), \quad (\text{D.33})$$

*and*

$$(1 - \delta)\|\mathcal{T}\| \leq \|\mathcal{M}\| \leq (1 + \delta)\|\mathcal{T}\| \quad (\text{D.34})$$

*Moreover, the tensor-column subspaces of* X *and* L *are aligned, namely*

$$\| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{L}} \| \leq 2\delta \quad (\text{D.35})$$

*Proof.* Consider tensor T := X ∗ X <sup>⊤</sup> ∈ S n×n×k <sup>+</sup> . Due to the definition of tensor transpose and conjugate symmetry of Fourier coefficients [\(Kilmer & Martin, 2011\)](#page-9-20), the Fourier slices of T are defined as T (j) <sup>=</sup> <sup>X</sup>(j)X(j)<sup>H</sup> . That is, each face of T is Hermitian and at least positive semidefinite. As we assume that for each j, 1 ≤ j ≤ k, one has ∥E<sup>t</sup> (j)∥ ≤ δλr(T (j) ) using Weyl's inequality in each of the Fourier slices, we obtain the first three inequalities.

To show that the tensor subspace V<sup>X</sup> and V<sup>L</sup> are aligned, we use first the definition

$$\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{L}}\| = \max_{1 \leq j \leq k} \|\overline{\mathbf{V}_{\mathcal{X}^\perp}^{(j)}}^H \overline{\mathbf{V}_{\mathcal{L}}^{(j)}}\| \quad (\text{D.36})$$

For the estimation of ∥V H X <sup>⊥</sup> (j) V (j) <sup>L</sup> ∥ in each of the Fourier slices, we apply Wedin's sin Θ theorem. For this, denote L := VM(:, 1 : r, :) ∈ <sup>R</sup> n×r×k and let V (j) <sup>L</sup> denote the corresponding Fourier slices of L ∈ <sup>R</sup> n×r×k . Since in the Fourier space, it holds that M(j) = T (j) +E(j) and V (j) <sup>L</sup> encompasses the first <sup>r</sup> eigenvectors of <sup>M</sup>(j) , from Wedin's sin Θ theorem, we obtain

$$\|\overline{V}_{\mathcal{X}^\perp}^{(j)}\|^H \overline{V}_{\mathcal{L}}^{(j)} \leq \frac{\|\overline{E}^{(j)}\|}{\xi^{(j)}},$$

with ξ (j) := λr(T (j) ) − λr+1(M(j) ). Using estimate [\(D.32\)](#page-19-0), ξ (j) can be lower-bounded as

$$\xi^{(j)} := \lambda_r(\bar{T}^{(j)}) - \lambda_{r+1}(\bar{M}^{(j)}) \geq \lambda_r(\bar{T}^{(j)}) - \delta \lambda_r(\bar{T}^{(j)}) = (1 - \delta) \lambda_r(\bar{T}^{(j)}).$$

Using the bound the the assumptions that ∥E<sup>t</sup> (j)∥ ≤ δλr(T (j) ) and δ ≤ 1 , we get

$$\| \overline{V_{\mathcal{X}^\perp}^{(j)}}^{\text{H}} \overline{V_{\mathcal{L}}^{(j)}} \| \leq \frac{\delta}{1-\delta} \leq 2\delta.$$

Coming back to equality [\(D.36\)](#page-19-1), we obtain the stated bound for the principal angle between the two tensor column subspaces.

Lemma D.7. *Consider a tensor* X ∗ X <sup>⊤</sup> ∈ S n×n×k <sup>+</sup> *with tubal rank* r ≤ n*. Assume that measurement operator* A *is such that*

$$\mathcal{M} = \mathcal{A}^* \mathcal{A}(\boldsymbol{x} * \boldsymbol{x}^\top) = \boldsymbol{x} * \boldsymbol{x}^\top + \boldsymbol{\varepsilon}$$

*and for each,* <sup>j</sup>*,* <sup>1</sup> <sup>≤</sup> <sup>j</sup> <sup>≤</sup> <sup>k</sup>*, one has* ∥E(j)∥ ≤ δλr(X(j)X(j)<sup>H</sup> ) *with* δ ≤ c1*. Moreover, assume that for difference tensor* E<sup>t</sup> <sup>=</sup> U<sup>t</sup> − Ue<sup>t</sup> *it holds that*

$$\gamma := \frac{\alpha \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\mathbf{U}\| + \|\boldsymbol{\varepsilon}_t\|}{\min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)})} \frac{1}{\alpha \sigma_{\min}(\mathbf{V}_{\mathcal{U}}^{\top} * \mathbf{U})} \leq c_2 \kappa^{-2}, \quad (\text{D.37})$$

*where* c1, c<sup>2</sup> > 0 *are sufficiently small absolute constants. Then for the signal and noise term of the gradient descent* [\(C.1\)](#page-11-0)*, we have*

$$\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{U}_t * \mathbf{w}_t}\| \leq 14(\delta + \gamma) \quad (\text{D.38})$$

$$\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\| \leq \frac{\kappa^{-2}}{8} \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathcal{V}_{\mathcal{L}}^{\top}} * \mathcal{U}) \quad (\text{D.39})$$

*and for each* j*,* 1 ≤ j ≤ k*, it holds that*

$$\sigma_{min}(\overline{\mathcal{U}}_t * \overline{\mathcal{W}}_t^{(j)}) \geq \frac{1}{4} \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathcal{V}}_{\mathcal{L}}^{\top} * \mathcal{U}) \quad (\text{D.40})$$

$$\sigma_1(\overline{\mathcal{U}}_t * \overline{\mathcal{W}}_{t,\perp}^{(j)}) \leq \frac{\kappa^{-2}}{8} \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathcal{V}}_{\mathcal{L}}^{\top} * \mathcal{U}) \quad (\text{D.41})$$

*Proof.* To prove the above-stated properties, we will use Lemma [D.3.](#page-14-0) Therefore, we start by checking the conditions of this lemma. Sufficiently small c<sup>2</sup> and the assumption γ ≤ c2κ −2 allows for γ ≤ . This means that

$$\frac{\alpha \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{Z_t}^{(j)}) \|\mathbf{u}\| + \|\boldsymbol{\varepsilon}_t\|}{\min_{1 \leq j \leq k} \sigma_r(\overline{Z_t}^{(j)})} \frac{1}{\alpha \sigma_{\min}(\overline{\mathbf{v}_{\mathcal{L}}^{\top} * \mathbf{u}})} \leq \frac{1}{2}$$

and in each of the Fourier slices we have

$$\sigma_{r+1}(\overline{Z}_t^{(j)})\|\boldsymbol{u}\| + \frac{\|\boldsymbol{\varepsilon}_t\|}{\alpha} \leq \frac{1}{2}\sigma_r(\overline{Z}_t^{(j)})\sigma_{min}(\overline{\boldsymbol{v}_{\mathcal{L}}^\top * \boldsymbol{u}}),$$

fulfilling the assumption of Lemma [D.3.](#page-14-0) Hence, from Lemma [D.3,](#page-14-0) we conclude that

$$\|\mathbf{v}_{\mathcal{L}^\perp}^\top * \mathbf{v}_{\mathcal{L}_t}\| \leq \max_{1 \leq j \leq k} \frac{\alpha \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\mathbf{u}\| + \|\boldsymbol{\varepsilon}_t\|}{\alpha \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u}) - \alpha \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\mathbf{u}\| - \|\boldsymbol{\varepsilon}_t\|} \quad (\text{D.42})$$

$$\begin{aligned} & \frac{\alpha \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\boldsymbol{\mathcal{U}}\| + \|\boldsymbol{\mathcal{E}}_t\|}{\alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{\min}(\overline{\boldsymbol{\mathcal{V}}_{\mathcal{L}}} * \boldsymbol{\mathcal{U}}) - \alpha \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{Z}_t^{(j)}) \|\boldsymbol{\mathcal{U}}\| - \|\boldsymbol{\mathcal{E}}_t\|}, \\ & (D.43) \end{aligned}$$

(j)

and, moreover, together with Lemma [D.5](#page-17-1) and the assumption γ ≤ <sup>2</sup> we get

$$\min_{1 \leq j \leq k} \sigma_r(\overline{U}_t^{(j)}) \geq \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathbf{v}}_{\mathcal{L}}^{\top} * \overline{\mathbf{u}}) - \|\mathcal{E}_t\| \geq \frac{\alpha}{2} \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathbf{v}}_{\mathcal{L}}^{\top} * \overline{\mathbf{u}}) \quad (\text{D.44})$$

$$\max_{1 \leq j \leq k} \sigma_{r+1}(\overline{U}_t^{(j)}) \leq \alpha \min_{1 \leq j \leq k} \sigma_r \sigma_r(\overline{Z}_t^{(j)}) \|\mathbf{U}\| + \|\boldsymbol{\varepsilon}_t\| \leq \alpha \gamma \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathbf{V}}_{\mathcal{L}}^{\top} * \mathbf{U}) \quad (\text{D.45})$$

The last two inequalities, allow extend bound [\(D.42\)](#page-20-0) as follows

$$\|\mathbf{v}_{\mathcal{L}^\perp}^\top * \mathbf{v}_{\mathcal{L}_t}\| \leq \frac{\gamma}{1-\gamma} \quad (\text{D.46})$$

Now, consider the principal angle between X and L<sup>t</sup> using its definition

$$\begin{aligned} \|\boldsymbol{\nu}_{\boldsymbol{\mathcal{X}}^{\perp}}^{\top} * \boldsymbol{\nu}_{\boldsymbol{\mathcal{L}}}\| &= \max_{1 \leq j \leq k} \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)}\| = \max_{1 \leq j \leq k} \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)H} - \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H}\| \\ &\leq \max_{1 \leq j \leq k} \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)H} - \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H}\| \leq \max_{1 \leq j \leq k} \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)H} - \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H}\| + \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H} - \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H}\| \\ &\leq \max_{1 \leq j \leq k} \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{X}}^{\perp}}^{(j)H} - \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H}\| + \max_{1 \leq j \leq k} \|\bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H} - \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)} \bar{\boldsymbol{\nu}}_{\boldsymbol{\mathcal{L}}}^{(j)H}\| \\ &= \|\boldsymbol{\nu}_{\boldsymbol{\mathcal{X}}^{\perp}}^{\top} * \boldsymbol{\nu}_{\boldsymbol{\mathcal{L}}}\| + \|\boldsymbol{\nu}_{\boldsymbol{\mathcal{L}}^{\perp}}^{\top} * \boldsymbol{\nu}_{\boldsymbol{\mathcal{L}}}\| \end{aligned}$$

Using the last line above, and inequalities [\(D.35\)](#page-19-2) and [\(D.46\)](#page-20-1), we obtain

$$\| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{L}_t} \| \leq 2(\delta + \gamma).$$

From here, allowing δ and γ to be such that ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>L</sup><sup>t</sup> ∥ ≤ <sup>1</sup> 8 , we can use Lemma [D.5](#page-17-1) to get

$$\| \boldsymbol{\nu}_{\boldsymbol{x}^\perp} * \boldsymbol{\nu}_{\boldsymbol{u}_t \boldsymbol{w}_t} \| \leq 7 \| \boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{\nu}_{\boldsymbol{c}_t} \| \leq 14(\delta + \gamma).$$

Furthermore, Lemma [D.5](#page-17-1) together with inequality [\(D.45\)](#page-20-2) also results in

$$\begin{aligned} \sigma_1(\overline{\mathbf{U}}_t * \overline{\mathbf{W}}_{t,\perp}^{(j)}) &\leq 2\sigma_{r+1}(\overline{U_t}^{(j)}) \\ &\leq 2 \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{U_t}^{(j)}) \\ &\leq 2\gamma\alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z_t}^{(j)}) \sigma_{\min}(\overline{\mathbf{V}_{\mathcal{L}}^\top * \mathbf{U}}) \\ &\leq \frac{\kappa^{-2}}{8} \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z_t}^{(j)}) \sigma_{\min}(\overline{\mathbf{V}_{\mathcal{L}}^\top * \mathbf{U}}) \end{aligned}$$

and for the spectral norm of U<sup>t</sup> ∗ Wt,<sup>⊥</sup> we get

$$\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\| \leq 2 \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{U}_t^{(j)}) \leq \frac{\kappa^{-2}}{8} \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{\min}(\overline{\mathcal{V}_{\mathcal{L}}^{\top} * \mathcal{U}}).$$

To conclude the proof, we see that Lemma [D.5](#page-17-1) together with inequality [\(D.44\)](#page-20-3) provides for each j, 1 ≤ j ≤ k, the following lower bound

$$\sigma_r\left(\overline{\mathbf{u}_t * \mathbf{w}_t^{(j)}}\right) \geq \frac{1}{2}\sigma_r\left(\overline{\mathbf{u}_t^{(j)}}\right) \geq \frac{\alpha}{4}\sigma_r(\overline{Z_t^{(j)}})\sigma_{min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u}}) \geq \frac{\alpha}{4} \min_{1 \leq j \leq k} \sigma_r(\overline{Z_t^{(j)}})\sigma_{min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u}}).$$

The following lemma shows that for an appropriately chosen initialization, in the first new iteration, the tensor column subspaces between the signal term U<sup>t</sup> ∗ W<sup>t</sup> and the ground truth tensor X become aligned. Moreover, for each 1 ≤ j ≤ k there is a solid gap between the smallest singular values of the signal term and the largest singular values of the noise term.

Lemma D.8. *Assume* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies the S2NRIP*(δ1) *for some constant* δ<sup>1</sup> > 0*. Also, assume that*

$$\mathbf{M} := \mathcal{A}^* \mathcal{A}(\mathbf{x} * \mathbf{x}^\top) = \mathbf{x} * \mathbf{x}^\top + \boldsymbol{\varepsilon}$$

*with* ∥E(j)∥ ≤ δλr(X(j)X(j)<sup>H</sup> ) *for each* 1 ≤ j ≤ k *and* δ ≤ c1κ −2

*Denote by* L *the tensor-columns corresponding to the first* r *singular tubes in the t-SVD of*M*, that is,* L := VM(:, 1 : r, :) ∈ R n×r×k *, and define the initialization* U<sup>0</sup> = αU *with the coefficient* α *such that*

$$\alpha^2 \leq \frac{c \|\mathbf{x}\|^2}{12k\sqrt{\min\{n, R\}}\kappa^2 \|\mathbf{u}\|^3} \left( \frac{2\kappa^2 \|\mathbf{u}\|^3}{c_3 \sigma_{\min}(\mathbf{v}_{\mathcal{L}}^{\top} * \mathbf{u})} \right)^{-48\kappa^2} \min \{ \sigma_{\min}(\overline{\mathbf{v}_{\mathcal{L}}^{\top} * \mathbf{u}}), \|\overline{\mathbf{u}}_0^{\text{H}} v_1\|_{\ell_2} \} \quad (\text{D.47})$$

*where* v<sup>1</sup> ∈ <sup>C</sup> nk *is the leading eigenvector of matrix* M ∈ <sup>C</sup> nk×nk *.*

*Assume that learning rate* µ *fulfils* µ ≤ c3κ <sup>−</sup><sup>2</sup>∥X ∥ −2 *, then after* t<sup>⋆</sup> *iterations with*

$$t_* \asymp \frac{1}{\mu \min_{1 \leq j \leq k} \sigma_r(\overline{X}^{(j)})^2} \ln \left( \frac{2\kappa^2 \|\mathcal{U}\|}{c_3 \sigma_{\min}(\mathbf{y}_{\mathcal{L}}^\top * \mathcal{U})} \right) \quad (\text{D.48})$$

*it holds that*

$$\|\boldsymbol{u}_{t_*}\| \leq 3\|\boldsymbol{x}\| \quad (\text{D.49})$$

$$\| \mathcal{V}_{x^\perp} * \mathcal{V}_{u_{t_*} * \mathcal{W}_{t_*}} \| \leq c\kappa^{-2}. \quad (\text{D.50})$$

*and for each* 1 ≤ j ≤ k*, we have*

$$\sigma_r \left( \overline{\boldsymbol{u}_{t_*} * \boldsymbol{w}_{t_*}}^{(j)} \right) \geq \frac{1}{4} \alpha \beta \quad (\text{D.51})$$

$$\sigma_1\left(\overline{\boldsymbol{u}_{t_*} * \boldsymbol{w}_{t_*,\perp}}^{(j)}\right) \leq \frac{\kappa^{-2}}{8} \alpha \beta \quad (\text{D.52})$$

(D.53)

$$\text{where } \beta \text{ satisfies } \sigma_{\min}(\overline{\mathbf{v}}_{\mathcal{L}}^\top * \mathbf{u}) \leq \beta \leq \sigma_{\min}(\overline{\mathbf{v}}_{\mathcal{L}}^\top * \mathbf{u}) \left( \frac{2\kappa^2 \|\mathbf{u}\|}{c_3 \sigma_{\min}(\overline{\mathbf{v}}_{\mathcal{L}}^\top * \mathbf{u})} \right)^{16\kappa^2}.$$

*Proof.* For the proof of this lemma, we want to apply Lemma [D.7.](#page-19-3) The first condition of Lemma [D.7](#page-19-3) is the following

$$\gamma := \frac{\alpha \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{Z_t}^{(j)}) \|\mathcal{U}\| + \|\mathcal{E}_t\|}{\min_{1 \leq j \leq k} \sigma_r(\overline{Z_t}^{(j)})} \frac{1}{\alpha \sigma_{\min}(\overline{\mathcal{V}_{\mathcal{L}}^{\top} * \mathcal{U}})} \leq c_2 \kappa^{-2},$$

By the definition of γ, it is sufficient to show that

$$\max_{1 \leq j \leq k} \sigma_{r+1}(\overline{Z}_r^{(j)}) \|\boldsymbol{\mathcal{U}}\| \leq \frac{c_3}{2\kappa^2} \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_r^{(j)}) \sigma_{min}(\boldsymbol{\mathcal{V}}_{\mathcal{L}}^{\top} * \boldsymbol{\mathcal{U}}) \quad (\text{D.54})$$

and

$$\|\mathcal{E}_t\| \leq \frac{c_3}{2\kappa^2} \alpha \min_{1 \leq j \leq k} \sigma_r(\overline{Z}_t^{(j)}) \sigma_{min}(\overline{\mathcal{V}}_{\mathcal{L}}^\top * \mathcal{U}). \quad (\text{D.55})$$

Since for Z<sup>t</sup> = (I + µM) ∗t the transformation in the Fourier domain leads to the blocks

$$\overline{Z}_t^{(j)} = (\text{Id} + \mu \overline{M}^{(j)})^t,$$

this means that inequality [\(D.54\)](#page-22-0) is equivalent to

$$\frac{2\kappa^2\|\mathbf{u}\|}{c_3\sigma_{min}(\mathbf{v}_{\mathcal{U}}^\top * \mathbf{u})} \leq \left( \frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})} \right)^t,$$

which can be further modified as

$$\ln \left( \frac{2\kappa^2 \|\mathcal{U}\|}{\sigma_{\min}(\overline{\mathcal{V}_{\mathcal{L}}^\top \mathcal{U}})} \right) \leq t \ln \left( \frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})} \right).$$

Hence, if we take t<sup>⋆</sup> as follows

$$t_* := \left\lceil \ln \left( \frac{2\kappa^2 \|\mathbf{U}\|}{\sigma_{\min}(\overline{\mathbf{V}}_{\mathcal{L}}^\top * \mathbf{U})} \right) \right\rceil / \ln \left( \frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})} \right) \right] \quad (\text{D.56})$$

then condition [\(D.54\)](#page-22-0) will be satisfied in each block in the Fourier domain. For convenience, we will further denote

$$\psi := \ln \left( \frac{2\kappa^2 \|\mathbf{U}\|}{\sigma_{min}(\overline{\mathbf{V}_{\mathcal{L}}^\top * \mathbf{U}})} \right). \quad (\text{D.57})$$

For the second part of Lemma [D.7'](#page-19-3)s condition, inequality [\(D.55\)](#page-22-1), we will use Lemma [D.1.](#page-12-2) To apply this Lemma, the condition t<sup>⋆</sup> ≤ t <sup>⋆</sup> needs to be satisfied. According to Lemma [D.2](#page-13-1)

$$t^* \geq \left\lceil \frac{\ln \left( \frac{\|\mathcal{M}\| \cdot \|\mathcal{U}_0^{\text{H}} v_1\|_{\ell_2}}{8(1+\delta_1\sqrt{k})\sqrt{k \min \{n, R\} \alpha^3 \|\mathcal{U}\|^3}} \right)}{2 \ln (1 + \mu \|\mathcal{M}\|)} \right\rceil \quad (\text{D.58})$$

For t<sup>⋆</sup> ≤ t ⋆ to hold, it will be sufficient to check, e.g., the following condition

$$\frac{\psi}{\ln\left(\frac{1+\mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1+\mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})}\right)} \leq \frac{1}{2} \cdot \frac{\ln\left(\frac{\|\mathcal{M}\| \cdot \|\overline{\mathcal{U}}_0^H v_1\|_{\ell_2}}{8(1+\delta_1 \sqrt{k}) \sqrt{k \min \{n, R\}} \alpha^3 \|\mathcal{U}\|^3}\right)}{2 \ln(1+\mu \|\mathcal{M}\|)}.$$

To check this condition let us first analyze the expression ln (1 + <sup>µ</sup>∥M∥)/ln 1+µ min1≤j≤<sup>k</sup> <sup>σ</sup>r(M(j) ) 1+µ max1≤j≤<sup>k</sup> σr+1(M(j)) first. Using x 1+<sup>x</sup> ≤ ln(1 + x) ≤ x, we can upper bound the above expression as

$$\frac{\ln(1 + \mu\|\mathcal{M}\|)}{\ln\left(\frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})}\right)} \leq \frac{\|\mathcal{M}\|(1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)}))}{\min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)}) - \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})} \quad (\text{D.59})$$

From here, applying the PSD of the tensor representatives in the Fourier domain and the assumptions δ ≤ 3 and µ ≤ c3κ <sup>−</sup><sup>2</sup>∥X ∥ −2 and Lemma [D.6,](#page-18-0) we get

$$\begin{aligned} \frac{\|\mathcal{M}\|(1 + \min_{1 \leq j \leq k} \sigma_r(\bar{M}^{(j)}))}{\min_{1 \leq j \leq k} \sigma_r(\bar{M}^{(j)}) - \max_{1 \leq j \leq k} \sigma_{r+1}(\bar{M}^{(j)})} &\leq \frac{(1 + \delta)\|\mathcal{T}\|}{(1 - 2\delta)\lambda_r(\bar{T}^{(j)})} \left(1 + c_3(1 + \delta) \left(\frac{\lambda_1(\bar{X}^{(j)})}{\kappa\|\mathcal{X}\|}\right)^2\right) \\ &\leq \kappa^2 \frac{(1 + \delta)}{(1 - 2\delta)} (1 + c_3(1 + \delta)) \leq 8\kappa^2, \end{aligned}$$

in the last line, we used the bound on δ and that c<sup>3</sup> can be taken small enough. This means

$$\frac{\ln(1 + \mu\|\mathbf{M}\|)}{\ln\left(\frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})}\right)} \leq 8\kappa^2. \quad (\text{D.60})$$

Thus, to show that t<sup>⋆</sup> ≤ t ⋆ , it is sufficient to tune the initialization factor α so that

$$\psi \cdot 32\kappa^2 \leq \ln \left( \frac{\|\mathcal{M}\| \cdot \|\bar{\mathcal{U}}_0^H v_1\|_{\ell_2}}{8(1 + \delta_1 \sqrt{k}) \sqrt{k} \min \{n, R\} \alpha^3 \|\mathcal{U}\|^3} \right).$$

or using the notation for ϕ, this is equivalent to

$$\left( \frac{2\kappa^2 \|\mathbf{u}\|}{\sigma_{\min} \overline{\mathbf{v}_L^\top * \mathbf{u}}} \right)^{32\kappa^2} \leq \frac{\|\mathbf{M}\| \cdot \overline{\mathbf{u}_0}^\top v_1 \|_{\ell_2}}{8(1 + \delta_1 \sqrt{k}) \sqrt{k \min \{n, R\}} \alpha^3 \|\mathbf{u}\|^3}$$

Since ∥U<sup>0</sup> H v1∥<sup>ℓ</sup><sup>2</sup> /α = ∥U H v1∥<sup>ℓ</sup><sup>2</sup> , The last inequality is implied if

$$\alpha^2 \leq \left( \frac{2\kappa^2 \|\mathbf{U}\|}{\sigma_{\min}(\mathbf{V}_{\mathcal{L}}^{\top} * \mathbf{U})} \right)^{-32\kappa^2} \frac{\|\mathbf{M}\| \cdot \|\bar{\mathbf{U}}^{\text{H}} v_1\|_{\ell_2}}{8(1 + \delta_1 \sqrt{k}) \sqrt{k \min \{n, R\} \|\mathbf{U}\|^3}},$$

or if we set α even smaller using the fact that (1 + δ<sup>1</sup> √ k) √ <sup>k</sup> <sup>≤</sup> (1 + √ k) √ k ≤ 2k and ∥M∥ ≥ <sup>2</sup> 3 ∥X ∥ 2 and set the parameter α so that

$$\alpha^2 \leq \left( \frac{2\kappa^2 \|\mathbf{u}\|}{\sigma_{\min}(\mathbf{v}_{\mathcal{L}}^{\top} * \mathbf{u})} \right)^{-32\kappa^2} \frac{\|\mathbf{x}\|^2 \cdot \|\bar{\mathbf{u}}^{\text{H}} v_1\|_{\ell_2}}{24k\sqrt{\min\{n, R\}}\|\mathbf{u}\|^3}.$$

Hence t<sup>⋆</sup> ≤ t ⋆ is satisfied and applying Lemma [D.7,](#page-19-3) we get

$$\|\mathcal{E}_{*}\| \leq 8(1 + \delta_1 \sqrt{k}) \sqrt{k \min \{n, R\}} \frac{\alpha^3}{\|\mathcal{M}\|} \|\mathcal{U}\|^3 (1 + \mu \|\mathcal{M}\|)^{3t_*} \quad (\text{D.61})$$

Moreover, using ∥M∥ ≥ <sup>2</sup> 3 ∥X ∥ 2 from Lemma [D.6](#page-18-0) with δ ≤ 1/3 and (1 + δ<sup>1</sup> √ k) √ k ≤ 2k , we get

$$\|\mathcal{E}_{t*}\| \leq 12k\sqrt{\min\{n, R\}} \frac{\alpha^3}{\|\mathcal{X}\|^2} \|\mathcal{U}\|^3 (1 + \mu \|\mathcal{M}\|)^{3t*}$$

Hence, using that Z<sup>t</sup> (j) = (Id + µM(j) ) t inequality [\(D.55\)](#page-22-1) will be implied if

$$12k\sqrt{\min\{n, R\}} \frac{\alpha^3}{\|\mathbf{x}\|^2} \|\mathbf{u}\|^3 (1 + \mu \|\mathbf{M}\|)^{3t_*} \leq \frac{c_3}{2\kappa^2} \alpha \min_{1 \leq j \leq k} \sigma_r \left( (\text{Id} + \mu \overline{M}^{(j)})^{t_*} \right) \sigma_{min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u}}),$$

which is equivalent to

$$\alpha^2 \leq c_3 \frac{\|\boldsymbol{\mathcal{X}}\|^2 \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\mathcal{L}}^\top * \boldsymbol{\mathcal{U}})}{12k\sqrt{\min\{n, R\}}\kappa^2\|\boldsymbol{\mathcal{U}}\|^3} \frac{(1 + \mu\lambda_r(\overline{M}^{(j)}))^{t_\star}}{(1 + \mu\|\boldsymbol{\mathcal{M}}\|)^{3t_\star}}, \quad (\text{D.62})$$

for all j. To proceed further, let us analyze the last factor from above using the definition of t⋆. Note that

$$\frac{(1 + \mu\lambda_r(\overline{M}^{(j)}))^{t_\star}}{(1 + \mu\|\mathcal{M}\|)^{3t_\star}} = \exp\left(t_\star \ln\left(\frac{1 + \mu\lambda_r(\overline{M}^{(j)})}{(1 + \mu\|\mathcal{M}\|)^3}\right)\right) \geq \exp(-3t_\star \ln((1 + \mu\|\mathcal{M}\|)^3))$$

Now, using the definition of t⋆, that is t<sup>⋆</sup> = l ψ/ ln 1+µ min1≤j≤<sup>k</sup> <sup>σ</sup>r(M(j) 1+µ max1≤j≤<sup>k</sup> σr+1(M(j)) m and inequality [\(D.60\)](#page-23-0), we get

$$\exp\left(-3t_* \ln\left((1 + \mu\|\boldsymbol{\mathcal{M}}\|)^3\right)\right) \geq \exp\left(-48\psi\kappa^2\right) = \left(\frac{2\kappa^2\|\boldsymbol{\mathcal{U}}\|}{c_3\sigma_{min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{L}}}^\top * \boldsymbol{\mathcal{U}})}\right)^{-48\kappa^2} \quad (\text{D.63})$$

Inserting this into inequality [\(D.62\)](#page-24-0), we get

$$\alpha^2 \leq c_3 \frac{\|\mathcal{X}\|^2 \sigma_{\min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathcal{U}})}{12 k \sqrt{\min \{n, R\}} \kappa^2 \|\mathcal{U}\|^3} \left( \frac{2\kappa^2 \|\mathcal{U}\|}{c_3 \sigma_{\min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathcal{U}})} \right)^{-48\kappa^2}. \quad (\text{D.64})$$

For such α, we have shown that inequality [\(D.55\)](#page-22-1) holds, and the condition of Lemma [D.7](#page-19-3) is fulfilled, which gives us

$$\| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{u_t * \mathbf{w}_t} \| \leq 14(\delta + \gamma) \leq c\kappa^{-2}, \quad (\text{D.65})$$

where the last inequality follows from our assumption that δ ≤ c1κ −2 and µ ≤ c3κ <sup>−</sup><sup>2</sup>∥X ∥ −2 and from setting the constants c<sup>1</sup> and c<sup>3</sup> small enough.

Moreover, for each 1 ≤ j ≤ k, from Lemma [D.7](#page-19-3) it follows that

$$\sigma_{min}(\overline{\mathbf{u}}_t * \overline{\mathbf{W}}_t^{(j)}) \geq \frac{1}{4}\alpha\beta, \quad (\text{D.66})$$

$$\sigma_1(\overline{\mathcal{U}}_t * \overline{\mathcal{W}}_{t,\perp}^{(j)}) \leq \frac{\kappa^{-2}}{8} \alpha \beta. \quad (\text{D.67})$$

where β := min1≤j≤<sup>k</sup> σr(Z<sup>t</sup> (j) )σmin(V ⊤ <sup>L</sup> ∗ U).

In the remaining part, we will show that t⋆, β and ∥U<sup>t</sup><sup>⋆</sup> ∥ have the properties stated in the lemma.

Let us start with t⋆. Using the same inequalities for ln(1 + x) as above and Lemma [D.6,](#page-18-0) one can show

$$\ln \left( \frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})} \right) \geq \frac{\mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})} - \mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)}) \geq \frac{2}{3} \mu \min_{1 \leq j \leq k} \sigma_r(\overline{X}^{(j)})^2$$

and at the same time

$$\ln \left( \frac{1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1 + \mu \min_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})} \right) \leq \ln \left( 1 + \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)}) \right) \leq \mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})$$

$$\leq \mu(1 + \delta) \min_{1 \leq j \leq k} \sigma_r(\overline{X}^{(j)})^2 \leq 4/3\mu \min_{1 \leq j \leq k} \sigma_r(\overline{X}^{(j)})^2$$

which shows that, on the one hand,

$$\frac{1}{\ln\left(\frac{1+\mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}^{(j)})}{1+\mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}^{(j)})}\right)} \leq \frac{2}{3\mu} \max_{1 \leq j \leq k} \frac{1}{\sigma_r(\overline{X}^{(j)})^2} = \frac{2}{3\mu \min_{1 \leq j \leq k} \sigma_r(\overline{X}^{(j)})^2}$$

and on the other hand

$$\frac{1}{\ln\left(\frac{1+\mu \min_{1 \leq j \leq k} \sigma_r(\overline{M}(j))}{1+\mu \max_{1 \leq j \leq k} \sigma_{r+1}(\overline{M}(j))}\right)} \geq \frac{3}{4\mu \min_{1 \leq j \leq k} \sigma_r(\overline{X}(j))^2},$$

which shows the desired properties of t⋆.

Now, we consider β := min1≤j≤<sup>k</sup> σr(Z<sup>t</sup><sup>⋆</sup> (j) )σmin(V ⊤ <sup>L</sup> ∗ U). By the definition of Z<sup>t</sup> (j) and inequality [\(D.60\)](#page-23-0), we get

$$\begin{aligned} \left(1 + \mu\sigma_r(\overline{M}^{(j)})\right)^{t_\star} &= \exp\left(t_\star \ln(1 + \mu\sigma_r(\overline{M}^{(j)}))\right) \leq \exp\left(t_\star \ln(1 + \mu\|\boldsymbol{\mathcal{M}}\|)\right) \\ &\leq \exp\left(2\psi \max_{1 \leq j \leq k} \frac{\ln(1 + \mu\|\boldsymbol{\mathcal{M}}\|)}{\ln\left(\frac{1 + \mu\sigma_r(\overline{M}^{(j)})}{1 + \mu\sigma_{r+1}(\overline{M}^{(j)})}\right)}\right) \leq \exp(16\psi\kappa^2) = \left(\frac{2\kappa^2\|\boldsymbol{\mathcal{U}}\|}{c_3\sigma_{\min}(\overline{\boldsymbol{\mathcal{V}}_{\mathcal{L}}^\top} * \boldsymbol{\mathcal{U}})}\right)^{16\kappa^2}. \end{aligned} \quad (\text{D.68})$$

Since this holds for all j, we have

$$\beta \leq \sigma_{\min}(\overline{\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u}}) \left( \frac{2\kappa^2 \|\mathbf{u}\|}{c_3 \sigma_{\min}(\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u})} \right)^{16\kappa^2}.$$

Finally, we come to the properties of U<sup>t</sup> <sup>⋆</sup> . By the representation U<sup>t</sup><sup>⋆</sup> = Z<sup>t</sup><sup>⋆</sup> ∗ U<sup>0</sup> + E<sup>t</sup><sup>⋆</sup> , we get

$$\| \mathcal{U}_{t^*} \| \leq \alpha \| \mathcal{Z}_{t^*} \| \| \mathcal{U} \| + \| \mathcal{E}_{t^*} \|.$$

From [\(D.55\)](#page-22-1), we get

$$\|\boldsymbol{\varepsilon}_t\| \leq \frac{c_3}{2^{\kappa^2}} \alpha \|\boldsymbol{z}_t\| \sigma_{min}(\overline{\boldsymbol{V}}_{-}^{\text{H}} \overline{\boldsymbol{U}}) \leq \frac{c_3}{2^{\kappa^2}} \alpha \|\boldsymbol{z}_t\| \sigma_{min}(\overline{\boldsymbol{V}}_{-}^{\text{H}}) \sigma_{max}(\overline{\boldsymbol{U}}) \leq \alpha \|\boldsymbol{z}_t\| \|\boldsymbol{U}\|,$$

which allows us to proceed as follows

$$\begin{aligned} \|\mathbf{u}_{t^*}\| &\leq 2\alpha \|\mathbf{Z}_{t^*}\| \|\mathbf{u}\| \leq 2\alpha(1 + \mu \|\mathbf{M}\|)^{t^*} \|\mathbf{u}\|, \\ &= 2\alpha \ln \left( t_*(1 + \mu \|\mathbf{M}\|) \right) \|\mathbf{u}\| \leq 2\alpha \|\mathbf{u}\| \left( \frac{2\kappa^2 \|\mathbf{u}\|}{c_3 \sigma_{min}(\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u})} \right)^{16\kappa^2} \\ &\leq 2\|\mathbf{x}\| \sqrt{\frac{c_3 \sigma_{min}(\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u})}{12 k \sqrt{\min \{n, R\}} \kappa^2 \|\mathbf{u}\|}} \left( \frac{2\kappa^2 \|\mathbf{u}\|}{c_3 \sigma_{min}(\mathbf{v}_{\mathcal{L}}^\top * \mathbf{u})} \right)^{-8\kappa^2} \leq 3\|\mathbf{x}\|, \end{aligned}$$

where for the second inequality above we used [\(D.68\)](#page-25-0) and in the last one an upper bound on α from [\(D.64\)](#page-24-1) has been applied.

The results in Lemma [D.8](#page-21-0) hold for any initialization U. Below, we will use the fact that U is a tensor with Gaussian entries. This yields the following lemma, which shows that with initialization scale α > 0 chosen sufficiently small, the properties stated in Lemma [D.8](#page-21-0) hold with high probability.

Lemma D.9. *Fix a sufficiently small constant* c > 0*. Let* U ∈ R <sup>n</sup>×R×<sup>k</sup> *be a random tubal tensor with i.i.d.* N (0, R ) *entries, and let* ϵ ∈ (0, 1)*. Assume that* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies the S2NRIP*(δ1) *for some constant* δ<sup>1</sup> > 0*. Also, assume that*

$$\mathcal{M} := \mathcal{A}^* \mathcal{A}(\boldsymbol{x} * \boldsymbol{x}^\top) = \boldsymbol{x} * \boldsymbol{x}^\top + \boldsymbol{\varepsilon}$$

*with* ∥E(j)∥ ≤ δλr(X(j)X(j)<sup>H</sup> ) *for each* 1 ≤ j ≤ k*, where* δ ≤ c1κ −2 *. Let* U<sup>0</sup> = αU *where*

$$\alpha^2 \lesssim \begin{cases} \frac{\epsilon \min\{n, R\} \|\mathbf{X}\|^2}{k^2 n^{3/2} \kappa^2} \left( \frac{2\kappa^2 k n^{3/2}}{c_3 \min\{n, R\}^{3/2} \epsilon} \right)^{-24\kappa^2} & \text{if } R \geq 3r \\ \frac{\epsilon \|\mathbf{X}\|^2}{k^2 n^{3/2} \kappa^2} \left( \frac{2\kappa^2 k n^{3/2}}{c_3 r^{1/2} \epsilon} \right)^{-24\kappa^2} & \text{if } R < 3r \end{cases}$$

.

*Assume the step size satisfies* µ ≤ c2κ <sup>−</sup><sup>2</sup>∥X ∥ 2 *. Then, with probability at least* 1 − p *where*

$$p = \begin{cases} k(\tilde{C}\epsilon)^{R-r+1} + ke^{-\tilde{c}R} & \text{if } R \geq 2r \\ k\epsilon^2 + ke^{-\tilde{c}R} & \text{if } R < 2r \end{cases}$$

*the following statement holds. After*

$$t_* \lesssim \begin{cases} \frac{1}{\mu \min_{1 \leq j \leq k} \sigma_r(\bar{X}^{(j)})^2} \ln \left( \frac{2\kappa^2 \sqrt{n}}{c_3 \epsilon \sqrt{\min\{n; R\}}} \right) & \text{if } R \geq 3r \\ \frac{1}{\mu \min_{1 \leq j \leq k} \sigma_r(\bar{X}^{(j)})^2} \ln \left( \frac{2\kappa^2 \sqrt{n}}{c_3 \epsilon} \right) & \text{if } R < 3r \end{cases}$$

*iterations, it holds that*

$$\|\boldsymbol{u}_{t_*}\| \leq 3\|\boldsymbol{x}\| \quad (\text{D.69})$$

$$\| \mathcal{V}_{\mathcal{X}^\perp} * \mathcal{V}_{\mathcal{U}_{t_*} * \mathcal{W}_{t_*}} \| \leq c\kappa^{-2}. \quad (\text{D.70})$$

*and for each* 1 ≤ j ≤ k*, we have*

$$\sigma_r \left( \overline{\mathcal{U}_{t_*} * \mathcal{W}_{t_*}}^{(j)} \right) \geq \frac{1}{4} \alpha \beta \quad (\text{D.71})$$

$$\sigma_1 \left( \overline{\boldsymbol{u}_{t_*} * \boldsymbol{W}_{t_*,\perp}}^{(j)} \right) \leq \frac{\kappa^{-2}}{8} \alpha \beta \quad (\text{D.72})$$

(D.73)

*where*

$$\beta \lesssim \begin{cases} \epsilon \sqrt{k} \left( \frac{2\kappa^2 \sqrt{n}}{c_3 \epsilon \sqrt{\min\{n, R\}}} \right)^{16\kappa^2} & \text{if } R \geq 3r \\ \frac{\epsilon \sqrt{k}}{r} \left( \frac{2\kappa^2 \sqrt{rn}}{c_3 \epsilon} \right)^{16\kappa^2} & \text{if } R < 3r \end{cases}$$

*and*

$$\beta \gtrsim \begin{cases} \epsilon \sqrt{k} & \text{if } R \geq 3r \\ \frac{\epsilon \sqrt{k}}{r} & \text{if } R < 3r \end{cases}.$$

*Proof.* By Lemma [I.3,](#page-55-0) we have that ∥U∥ ≲ r k max{n, R} R = r kn min{n; R} with probability at least 1 − O(ke−<sup>c</sup> max{n,R} ). Also, by Lemma [I.4,](#page-56-0) we have that ∥U H v1∥<sup>ℓ</sup><sup>2</sup> = ∥U <sup>⊤</sup> ∗ V1∥<sup>F</sup> ≍ √ k with probability at least 1 − O(ke−cR). Since U ∈ <sup>R</sup> <sup>n</sup>×R×<sup>k</sup> has i.i.d. N (0, R ) entries and V ⊤ <sup>L</sup> ∗ V<sup>L</sup> = I, by rotational invariance, V ⊤ <sup>L</sup> ∗ U ∈ <sup>R</sup> r×R×k also has i.i.d. N (0, 1 R ) entries. Hence, the lower bound on σmin(V ⊤ <sup>L</sup> ∗ U) in Lemma [I.2](#page-55-1) applies. If r ≤ R ≤ 2r, we have

$$\sigma_{\min}(\mathcal{V}_{\mathcal{L}}^\top * \mathcal{U}) \geq \frac{\epsilon \sqrt{k}}{\sqrt{rR}} \gtrsim \frac{\epsilon \sqrt{k}}{r}$$

with probability at least 1 − kϵ<sup>2</sup> . If 2r < R < 3r, we have

$$\sigma_{\min}(\mathbf{v}_{\mathcal{L}}^{\top} * \mathbf{u}) \geq \frac{\epsilon\sqrt{k}(\sqrt{R} - \sqrt{2r-1})}{\sqrt{R}} \geq \frac{\epsilon\sqrt{k}(R - (2r-1))}{\sqrt{r}(\sqrt{R} + \sqrt{2r-1})} \gtrsim \frac{\epsilon\sqrt{k}}{r}$$

with probability at least 1 − k(Cϵ) <sup>R</sup>−2r+1 − ke−cR. If R ≥ 3r, we have

$$\sigma_{\min}(\mathbf{V}_{\mathcal{L}}^\top * \mathbf{U}) \geq \frac{\epsilon\sqrt{k}(\sqrt{R} - \sqrt{2r-1})}{\sqrt{R}} = \epsilon\sqrt{k} \left(1 - \sqrt{\frac{2r-1}{R}}\right) \gtrsim \epsilon\sqrt{k},$$

with probability at least 1 − k(Cϵ) <sup>R</sup>−2r+1 − ke−cR.

Therefore, the above bounds on ∥U∥, ∥U H v1∥<sup>ℓ</sup><sup>2</sup> , and σmin(V ⊤ <sup>L</sup> ∗ U) all hold simultaneously with probability at least 1 − p where

$$p = \begin{cases} k(\tilde{C}\epsilon)^{R-r+1} + ke^{-\tilde{c}R} & \text{if } R \geq 2r \\ k\epsilon^2 + ke^{-\tilde{c}R} & \text{if } R < 2r \end{cases}.$$

Provided that all three of these bounds hold, one can substitute these into Lemma [D.8](#page-21-0) to obtain the desired result.

## E. Analysis of Convergence Stage

In this section, we will prove that after passing the spectral stage, U<sup>t</sup> ∗ U ⊤ <sup>t</sup> goes into the convergence process towards the ground truth tensor X ∗ X <sup>⊤</sup> in the Frobenius norm. For this, we will first show that in each of the tensor slices σmin(V ⊤ <sup>X</sup> ∗ Ut+1 (j) ) grows exponentially, see Lemma [E.1,](#page-27-1) whereas the noise terms ∥Ut+1 ∗ Wt+1,<sup>⊥</sup> (j)∥, 1 ≤ j ≤ k, grow slower, see Lemma [E.3.](#page-30-0) Moreover, in Lemma [E.5,](#page-34-0) we show that the tensor column spaces of the signal term U<sup>t</sup> ∗ W<sup>t</sup> and the ground truth X stay aligned. With this, and several auxiliary lemmas in place, we show that

Lemma E.1. *Assume that the following conditions hold*

$$\begin{aligned} \mu &\leq c\|\boldsymbol{x}\|^{-2}\kappa^{-2} \\ \|u_t\| &\leq 3\|\boldsymbol{x}\| \\ \|\boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{\nu}_{u_t * \boldsymbol{w}_t}\| &\leq c\kappa^{-1} \end{aligned}$$

*and*

$$\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \leq c\sigma_{min}^2(\bar{\boldsymbol{x}}). \quad (\text{E.1})$$

*Moreover, assume that* V ⊤ <sup>X</sup> ∗ U<sup>t</sup> *has full tubal rank with all invertible t-SVD-singular tubes. Then, for each* j*,* 1 ≤ j ≤ k*, it holds that*

$$\sigma_{min}(\overline{\mathbf{v}_{\mathcal{X}}^\top * \mathcal{U}_{t+1}}^{(j)}) \geq \sigma_{min}(\overline{\mathbf{v}_{\mathcal{X}}^\top * \mathcal{U}_{t+1}} * \overline{\mathbf{W}_t}^{(j)}) \geq \sigma_{min}(\overline{\mathbf{v}_{\mathcal{X}}^\top * \mathcal{U}_t}^{(j)}) \left(1 + \frac{1}{4} \mu \sigma_{min}^2(\overline{\mathcal{X}}) - \mu \sigma_{min}^2(\overline{\mathbf{v}_{\mathcal{X}}^\top * \mathcal{U}_t}^{(j)})\right).$$

*Proof.* Consider the tensor V ⊤ <sup>X</sup> ∗ Ut+1 ∗ Wt. Using the definition of Ut+1 in terms of Ut, we can rewrite it as

$$\mathcal{V}_{\mathcal{X}}^\top * \mathcal{U}_{t+1} * \mathcal{W}_t = \mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} + \mu \mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)) * \mathcal{U}_t * \mathcal{W}_t.$$

This representation leads to the following representation of the RHS above in the Fourier domain

$$\overline{V}_{\boldsymbol{X}}^{(j)\text{ H}}(\text{Id} + \mu(\mathcal{A}^*\mathcal{A}(\boldsymbol{X} * \boldsymbol{X}^\top - \boldsymbol{U}_t * \boldsymbol{U}_t^\top))^{(j)})\overline{U}_t^{(j)}\overline{W}_t^{(j)} := H^{(j)}.$$

Note that here A<sup>∗</sup>A(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ) (j) can not be represented as an independent slice of measurements of X(j)X(j)H − U (j) <sup>t</sup> U (j)H t as it involved the information about all the slices 1 ≤ j ≤ k.

Due to our assumptions on ∥Ut∥ and the tensor spectral norm property, we get

$$\| \overline{V}_{\boldsymbol{x}}^{(j) \text{ H}} \overline{U}_t^{(j)} \| \leq \| \overline{U}_t^{(j)} \| \leq \| \boldsymbol{u}_t \| \leq 3 \| \boldsymbol{x} \|.$$

This in turn is leading to

$$\mu \leq c \| \boldsymbol{x} \|^{-2} \kappa^{-2} \leq \tilde{c} \| \overline{V}_{\boldsymbol{x}}^{(j)} {}^{\text{H}} \overline{U}_t^{(j)} \|^{-2}.$$

This property of µ together with the nature of W (j) t and V (j) <sup>X</sup> coming along from the signal-noise-term decomposition [\(C.1\)](#page-11-0) leads to the fulfilled conditions of Lemma [H.2.](#page-51-2) Applying Lemma [H.2](#page-51-2) to the matrix H(j) , the smallest singular value of matrix H(j) can be estimated as

$$\sigma_{min}(H^{(j)}) \geq (1 + \mu\sigma_{min}^2(\bar{X}^{(j)}) - \mu\|P_1^{(j)}\| - \mu\|P_2^{(j)}\| - \mu^2\|P_3^{(j)}\|) \sigma_{min}(\bar{V}_{\mathcal{X}}^{(j)\text{ H}} \bar{U}_t^{(j)}) (1 - \mu\sigma_{min}^2(\bar{V}_{\mathcal{X}}^{(j)\text{ H}} \bar{U}_t^{(j)})). \quad (\text{E.2})$$

with

$$\begin{aligned}\|P_1^{(j)}\| &\leq 4\|\overline{U}_t^{(j)}\overline{W}_t^{(j)}\|^2\|\overline{V}_{\mathcal{X}^\perp}^{(j)}V_{\overline{U}_t^{(j)}\overline{W}_t^{(j)}}\|^2 \\ \|P_2^{(j)}\| &\leq 4\left\|\overline{(\mathcal{A}^*\mathcal{A}(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top))^{(j)}} - \overline{\mathcal{X}^{(j)}}\overline{\mathcal{X}^{(j)}}^{\text{H}} + \overline{U}_t^{(j)}\overline{U}_t^{(j)}\right\| \\ \|P_3^{(j)}\| &\leq 2\|\overline{\mathcal{X}^{(j)}}\|^2\|\overline{U}_t^{(j)}\overline{W}_t^{(j)}\|^2.\end{aligned}$$

Further, we will make the above bounds for ∥P (j) i ∥, i ∈ {1, 2, 3}, more precise using information about the tensor setting. First of all since ∥U (j) <sup>t</sup> W (j) <sup>t</sup> ∥ ≤ ∥U (j) <sup>t</sup> ∥ ≤ ∥Ut∥ ≤ 3∥X ∥, we get ∥P (j) 1 ∥ ≤ 36∥X ∥ <sup>2</sup>∥V (j) <sup>X</sup> <sup>⊥</sup> V<sup>U</sup> <sup>t</sup> <sup>W</sup>(j) 2 . Moreover, since V (j) <sup>X</sup> <sup>⊥</sup> V<sup>U</sup> <sup>t</sup> <sup>W</sup>(j) = V ⊤ <sup>X</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> (j) and ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ cκ−<sup>1</sup> due to the assumption, it follows that for each j, 1 ≤ j ≤ k, it holds that ∥V (j) <sup>X</sup> <sup>⊥</sup> V<sup>U</sup> <sup>t</sup> <sup>W</sup>(j) t ∥ ≤ cκ−<sup>1</sup> . This allows for the following estimation

$$\|P_1^{(j)}\| \leq 36\|\boldsymbol{x}\|^2 c\kappa^{-1} \leq \frac{1}{4}\sigma_{min}^2(\bar{\boldsymbol{x}}),$$

where the last inequality follows from the fact that c > 0 is small enough.

Before proceeding with ∥P (j) 2 ∥, consider

$$(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) = (\mathcal{A}^*\mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) - (\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top).$$

The RHS from above has the following slices in the Fourier domain

$$\overline{(\mathcal{A}^*\mathcal{A})(\mathcal{X}^*\mathcal{X}^\top - \mathcal{U}_t^*\mathcal{U}_t^\top)}^{(j)} - (\overline{\mathcal{X}^{(j)}}\overline{\mathcal{X}^{(j)\text{H}}} - \overline{\mathcal{U}_t^{(j)}}\overline{\mathcal{U}_t^{(j)\text{H}}}),$$

the norm of which (due to assumption [\(E.1\)](#page-27-2) and the definition of the tensor spectral norm) can be bounded as

$$\|(\mathcal{A}^*\mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)^{(j)} - (\bar{\boldsymbol{x}}^{(j)}\bar{\boldsymbol{X}}^{(j)\text{H}} - \bar{\boldsymbol{U}}_t^{(j)}\bar{\boldsymbol{U}}_t^{(j)\text{H}})\| \leq \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \leq co_{min}^2(\bar{\boldsymbol{X}}).$$

This leads to the following estimation

$$\|P_2^{(j)}\| \leq 4c\sigma_{min}^2(\bar{\boldsymbol{\varkappa}})$$

To further assess ∥P (j) 3 ∥, we take into account that matrix W (j) t is an orthogonal matrix and the assumption ∥Ut∥ ≤ 3∥X ∥, which allows for the next bound

$$\|P_3^{(j)}\| \leq 2\|\bar{X}^{(j)}\|^2\|\bar{U}_t^{(j)}\bar{W}_t^{(j)}\|^2 \leq 2\|\boldsymbol{x}\|^2\|\bar{U}_t^{(j)}\|^2 \leq 2\|\boldsymbol{x}\|^2\|\boldsymbol{u}_t\|^2 \leq 18\|\boldsymbol{x}\|^4.$$

Inserting the newly obtained estimates for ∥P (j) i ∥, i ∈ {1, 2, 3}, into [\(E.2\)](#page-28-0), we get

$$\begin{aligned} \sigma_{min}(H^{(j)}) &\geq (1 + \mu\sigma_{min}^2(\bar{\mathcal{X}}^{(j)}) - \frac{\mu}{4}\sigma_{min}^2(\bar{\mathcal{X}}) - 4\mu c\sigma_{min}^2(\bar{\mathcal{X}}) - 18\mu^2\|\bar{\mathcal{X}}\|^4) \\ &\quad \cdot \sigma_{min}(\bar{V}_{\mathcal{X}}^{(j)}{}^{\text{H}}\bar{U}_t^{(j)})(1 - \mu\sigma_{min}^2(\bar{V}_{\mathcal{X}}^{(j)}{}^{\text{H}}\bar{U}_t^{(j)})) \\ &\geq (1 + \mu\sigma_{min}^2(\bar{\mathcal{X}}) - \frac{\mu}{4}\sigma_{min}^2(\bar{\mathcal{X}}) - 4\mu c\sigma_{min}^2(\bar{\mathcal{X}}) - 18\mu^2\|\bar{\mathcal{X}}\|^4)\sigma_{min}(\bar{V}_{\mathcal{X}}^{(j)}{}^{\text{H}}\bar{U}_t^{(j)})(1 - \mu\sigma_{min}^2(\bar{V}_{\mathcal{X}}^{(j)}{}^{\text{H}}\bar{U}_t^{(j)})). \end{aligned}$$

Now, according to the assumption on µ, we get

$$\mu^2 \|\boldsymbol{x}\|^4 \leq \mu c \kappa^{-2} \|\boldsymbol{x}\|^{-2} \|\boldsymbol{x}\|^4 = \mu c \frac{\sigma_{\min}^2(\overline{\boldsymbol{x}})}{\|\boldsymbol{x}\|^2} \|\boldsymbol{x}\|^{-2} \|\boldsymbol{x}\|^4 = c \mu \sigma_{\min}^2(\overline{\boldsymbol{x}})$$

Taking c small enough allows for the following estimation

$$\begin{aligned} \sigma_{min}(H^{(j)}) &\geq \sigma_{min}(\overline{V}_{\mathcal{X}}^{(j)H}\overline{U}_t^{(j)})(1 + \frac{1}{2}\mu\sigma_{min}^2(\overline{\mathcal{X}}))(1 - \mu\sigma_{min}^2(\overline{V}_{\mathcal{X}}^{(j)H}\overline{U}_t^{(j)})) \\ &= \sigma_{min}(\overline{V}_{\mathcal{X}}^{(j)H}\overline{U}_t^{(j)})\left(1 + \frac{1}{2}\mu\sigma_{min}^2(\overline{\mathcal{X}})(1 - \mu\sigma_{min}^2(\overline{V}_{\mathcal{X}}^{(j)H}\overline{U}_t^{(j)})) - \mu\sigma_{min}^2(\overline{V}_{\mathcal{X}}^{(j)H}\overline{U}_t^{(j)})\right) \end{aligned}$$

Now, since σmin(V (j) H <sup>X</sup> U (j) t ) ≤ σmin(U (j) t ) ≤ ∥Ut∥ ≤ 3∥X ∥, we have that

$$\mu\sigma_{min}^2(\overline{V}_{\mathbf{x}}^{(j)}{}^{\text{H}}\overline{U}_t^{(j)}) \leq \mu 9\|\mathbf{x}\|^2 \leq 9c\kappa^{-2} \leq \frac{1}{2}$$

due to the fact that c > 0 can be chosen small enough. The last part of Lemma's proof follows from σmin(V ⊤ <sup>X</sup> ∗ Ut+1 (j) ) ≥ σmin(V ⊤ <sup>X</sup> ∗ Ut+1 ∗ W<sup>t</sup> (j) ) and σmin(V ⊤ <sup>X</sup> ∗ Ut+1 ∗ W<sup>t</sup> (j) ) = σmin(H(j) ), which completes the argument.

The next two lemmas will allow us to show that in each of the Fourier slices the noise term part of the gradient descent iterates is growing slower than its signal term part.

Lemma E.2. *Assume that* <sup>µ</sup> <sup>≤</sup> <sup>c</sup> min n <sup>10</sup> ∥X ∥ −2 , ∥(A<sup>∗</sup>A − I)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t )∥ −1 o *and* ∥Ut∥ ≤ 3∥X ∥*. Moreover, suppose that* V ⊤ <sup>X</sup> ∗ U<sup>t</sup> *has full tubal rank with all invertible t-SVD-tubes and* ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ cκ−<sup>1</sup> *with a sufficiently small contact* c > 0*. Then, the principal angle between* V<sup>X</sup> <sup>⊥</sup> *and* V<sup>U</sup>t+1∗W<sup>t</sup> *can be bounded as follows*

$$\| \boldsymbol{\nu}_{\boldsymbol{X}^\perp}^\top * \boldsymbol{\nu}_{u_{t+1}*w_t} \| \leq 2 \| \boldsymbol{\nu}_{\boldsymbol{X}^\perp}^\top * \boldsymbol{\nu}_{u_t*w_t} \| + 2\mu \| (\mathcal{A}^* \mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top) \|.$$

*In particular, it holds that* ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t+1∗W<sup>t</sup> ∥ ≤ 1/50*.*

*Proof.* By the definition of Ut+1, we have

$$\mathcal{U}_{t+1} * \mathcal{W}_t = \left( \mathcal{I} + \mu \mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) \right) * \mathcal{U}_t * \mathcal{W}_t \quad \in \mathbb{R}^{n \times r \times k},$$

which allows for the following representation in the Fourier domain

$$\overline{\mathcal{U}_{t+1}} * \overline{\mathcal{W}_t}^{(j)} = \left( \text{Id} + \mu \mathcal{A} * \mathcal{A}(\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)^{(j)} \right) \overline{\mathcal{U}_t} * \overline{\mathcal{W}_t}^{(j)} \quad \in \mathbb{C}^{n \times r}, \quad 1 \leq j \leq k.$$

Consider the SVD decomposition U<sup>t</sup> ∗ W<sup>t</sup> (j) <sup>=</sup> <sup>V</sup>Ut∗W<sup>t</sup> (j)ΣUt∗W<sup>t</sup> (j)W<sup>H</sup> Ut∗W<sup>t</sup> and denote by Z (j) the matrix

$$Z^{(j)} := \left( \text{Id} + \mu \overline{\mathcal{A}^* \mathcal{A}(\boldsymbol{X} * \boldsymbol{X}^\top - \boldsymbol{U}_t * \boldsymbol{U}_t^\top)}^{(j)} \right) V_{\overline{\boldsymbol{U}_t * \boldsymbol{W}_t}^{(j)}} \in \mathbb{C}^{n \times r}.$$

Since by assumption U<sup>t</sup> ∗ W<sup>t</sup> (j) has full rank (due to full-rankness of V ⊤ <sup>X</sup> ∗ Ut, see Lemma [C.1\)](#page-11-3), matrix Z (j) has the same column space as Ut+1 ∗ W<sup>t</sup> (j) and the principal angle between tensor subspaces V<sup>X</sup> <sup>⊥</sup> and V<sup>U</sup>t+1∗W<sup>t</sup> can be computed via Z (j) as

$$\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{u_{t+1}*\mathbf{w}_t}\| = \max_{1 \leq j \leq k} \|\bar{\mathbf{v}}_{\mathcal{X}^\perp}^{(j)\text{H}} \bar{\mathbf{v}}_{u_{t+1}*\mathbf{w}_t}^{(j)}\| = \max_{1 \leq j \leq k} \|\bar{\mathbf{v}}_{\mathcal{X}^\perp}^{(j)\text{H}} \bar{\mathbf{v}}_{u_t*\mathbf{w}_{i^{(j)}}^\top}\| = \max_{1 \leq j \leq k} \|\bar{\mathbf{v}}_{\mathcal{X}^\perp}^{(j)\text{H}} V_{Z^{(j)}}\|.$$

Now, we will consider each of the terms ∥V (j)H <sup>X</sup> <sup>⊥</sup> <sup>V</sup>Z(j) ∥ separately and bound them as follows

$$\|\bar{V}_{\mathcal{X}^\perp}^{\text{H}} V_{Z(j)}\| \leq \|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}} V_{Z(j)} \Sigma_{Z(j)} W_{Z(j)}^\text{H}\| \|(\Sigma_{Z(j)} W_{Z(j)}^\text{H})^{-1}\| = \frac{\|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}} Z^{(j)}\|}{\sigma_{\min}(Z^{(j)})}. \quad (\text{E.3})$$

Using the definition of Z (j) , the norm in the numerator above can be estimated as

$$\begin{aligned} \|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}} Z^{(j)}\| &\leq \|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}} V_{\mathcal{U}_t * \mathcal{W}_t^{(j)}}\| + \mu \|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}} \mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)^{(j)}\| \\ &\leq \|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}} \bar{V}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)}\| + \mu \|\mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)^{(j)}\| \\ &\leq \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| + \mu \|\mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\|. \end{aligned}$$

Using again the definition of Z (j) and Weyl's inequality, the denominator in [\(E.3\)](#page-29-0) can be estimated from below as follows

$$\begin{aligned} \sigma_{\min}(Z^{(j)}) &\geq \sigma_{\min}(V_{\overline{\mathbf{u}_t * \mathbf{W}_t^{(j)}}}) - \mu \left\| \left( \overline{\mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)^{(j)}} \right) V_{\overline{\mathbf{u}_t * \mathbf{W}_t^{(j)}}} \right\| \\ &\geq 1 - \mu \left\| \overline{\mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)^{(j)}} \right\| \geq 1 - \mu \left\| \overline{\mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)} \right\| \\ &\geq 1 - \mu \left( \left\| (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top) \right\| + \left\| (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top) \right\| \right) \\ &\geq 1 - \mu \left( \left\| (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top) \right\| + \left\| \mathcal{X} \right\|^2 + \left\| \mathbf{u}_t \right\|^2 \right) \\ &\geq 1 - \mu \left( \left\| (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top) \right\| + 10 \left\| \mathcal{X} \right\|^2 \right) \geq \frac{1}{2}, \end{aligned}$$

where the last inequality follows from the assumption on µ. Now, we can come back to the estimation of ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗V<sup>U</sup>t+1∗W<sup>t</sup> ∥, which due to the combination of the above-carried estimated reads as

$$\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_{t+1}*w_t} \| \leq 2 \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t*w_t} \| + 2\mu \| \mathcal{A}^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) \|$$

providing the first result from the Lemma. The second bound stated in the Lemma follows from our assumption on ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ and µ and the fact that the constant c is chosen small enough to make ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t+1∗W<sup>t</sup> ∥ ≤ <sup>1</sup> <sup>50</sup> .

Lemma E.3. *Assume that* <sup>µ</sup> <sup>≤</sup> <sup>c</sup><sup>1</sup> min n <sup>10</sup> ∥X ∥ −2 , ∥(A<sup>∗</sup>A − I)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t )∥ −1 o *and* ∥Ut∥ ≤ 3∥X ∥*. Moreover, suppose that tensor* V ⊤ <sup>X</sup> ∗ Ut+1 ∗ W<sup>t</sup> *has all invertible t-SVD-tubes and that* ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ c1κ −1 *, with absolute constant* c<sup>1</sup> > 0 *chosen small enough. Then, it holds that*

$$\|\overline{\mathbf{u}_{t+1}} * \overline{\mathbf{w}_{t+1,\perp}}^{(j)}\| \leq \left(1 - \frac{\mu}{2} \|\overline{\mathbf{u}_t} * \overline{\mathbf{w}_{t,\perp}}^{(j)}\|^2 + 9\mu \|\overline{\mathbf{v}_{\mathcal{X}}^\top} * \overline{\mathbf{v}_{\mathcal{U}_t}} * \overline{\mathbf{w}_t}^{(j)}\| \|\mathcal{X}\|^2 + 2\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \right) \|\overline{\mathbf{u}_t} * \overline{\mathbf{w}_{t,\perp}}^{(j)}\|$$

*for each* j*, with* 1 ≤ j ≤ k*.*

*Proof.* First, we will consider tensor Ut+1 ∗ Wt+1,<sup>⊥</sup> splitting it into two different parts, and then will conduct the corresponding norm estimations of each Fourier slices.

To begin with, note that for the tensor-column space of X , that is V<sup>X</sup> , it holds that V<sup>X</sup> ∗ V ⊤ <sup>X</sup> + V<sup>X</sup> <sup>⊥</sup> ∗ V ⊤ <sup>X</sup> <sup>⊥</sup> = I (see, for example, [\(Liu et al., 2019\)](#page-9-17)). Using this, we can represent Ut+1 ∗ Wt+1,<sup>⊥</sup> as follows

$$\mathcal{U}_{t+1} * \mathcal{W}_{t+1,\perp} = \mathcal{V}_{\mathcal{X}} * \mathcal{V}_{\mathcal{X}}^\top * \mathcal{U}_{t+1} * \mathcal{W}_{t+1,\perp} + \mathcal{V}_{\mathcal{X}^\perp} * \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{U}_{t+1} * \mathcal{W}_{t+1,\perp} = \mathcal{V}_{\mathcal{X}^\perp} * \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{U}_{t+1} * \mathcal{W}_{t+1,\perp} \quad (\text{E.4})$$

where the last equality follows from Lemma [C.1](#page-11-3) due to the property V ⊤ <sup>X</sup> ∗ Ut+1 ∗ Wt+1,<sup>⊥</sup> = 0.

Now, we split the term V<sup>X</sup> <sup>⊥</sup> ∗ V ⊤ <sup>X</sup> <sup>⊥</sup> <sup>∗</sup> <sup>U</sup>t+1 <sup>∗</sup>Wt+1,<sup>⊥</sup> into two parts using <sup>W</sup><sup>t</sup> <sup>∗</sup> <sup>W</sup><sup>⊤</sup> <sup>t</sup> <sup>+</sup> Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> = I, which leads to

$$\nu_{\mathcal{X}^\perp} * \nu_{\mathcal{X}^\perp}^\top * \mathcal{U}_{t+1} * \mathcal{W}_{t+1,\perp} = \nu_{\mathcal{X}^\perp} * \nu_{\mathcal{X}^\perp}^\top * \mathcal{U}_{t+1} * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{W}_{t+1,\perp} + \nu_{\mathcal{X}^\perp} * \nu_{\mathcal{X}^\perp}^\top * \mathcal{U}_{t+1} * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top * \mathcal{W}_{t+1,\perp} \quad (\text{E.5})$$

To estimate the norm of V<sup>X</sup> <sup>⊥</sup> ∗ V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ U<sup>t</sup> ∗ Wt+1,<sup>⊥</sup> in each slice in the Fourier domain, we will use the above-given representation and estimate each of the summands individually. Let us start with the second one. Its jth slice in the Fourier domain reads as

$$(\mathcal{V}_{\mathcal{X}^\perp} * \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{U}_{t+1} * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top * \mathcal{W}_{t+1,\perp})^{(j)} = \overline{\mathcal{V}_{\mathcal{X}^\perp}^{(j)}} \overline{\mathcal{V}_{\mathcal{X}^\perp}^{(j)\text{H}}} \overline{\mathcal{U}_{t+1}^{(j)}} \overline{\mathcal{W}_{t,\perp}^{(j)}} \overline{\mathcal{W}_{t,\perp}^{(j)\text{H}}} \overline{\mathcal{W}_{t+1,\perp}^{(j)}}.$$

Due to the orthogonality of the columns of V (j) <sup>X</sup> <sup>⊥</sup> , it holds that ∥<sup>V</sup> (j) <sup>X</sup> <sup>⊥</sup> V (j)H <sup>X</sup> <sup>⊥</sup> U (j) <sup>t</sup>+1W (j) t,⊥W (j),H t,<sup>⊥</sup> W (j) <sup>t</sup>+1,<sup>⊥</sup>∥ = ∥V (j)H <sup>X</sup> <sup>⊥</sup> U (j) <sup>t</sup>+1W (j) t,⊥W (j),H t,<sup>⊥</sup> W (j) <sup>t</sup>+1,<sup>⊥</sup>∥. In the Fourier domain, this allows us to focus on jth slices of the last one

$$\overline{V}^{(j)H}_{\mathcal{X}^\perp} \overline{U}_{t+1}^{(j)} \overline{W}_{t,\perp}^{(j)} \overline{W}_{t,\perp}^{(j),H} \overline{W}_{t+1,\perp}^{(j)} := G_2^{(j)}.$$

Due to the definition of the gradient descent iterates Ut+1, we have the following representation for its blocks U (j) <sup>t</sup>+1 in the Fourier domain

$$\overline{U}_{t+1}^{(j)} = \left( \text{Id} + \mu (\overline{\mathcal{A}^* \mathcal{A}(\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)}) \right)^{(j)} \overline{U}_t^{(j)}$$

To upper bound the norm of G (j) 2 , we want to apply Lemma [H.3.](#page-52-0) Due to the assumptions in this lemma that V ⊤ <sup>X</sup> ∗Ut+1 ∗W<sup>t</sup> has full tubal rank with all invertible t-SVD-tubes and ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ cκ−<sup>1</sup> in addition to the conditions on µ and the decomposition of gradient descent iterates into the signal and noise term, the conditions of Lemma [H.3](#page-52-0) are satisfied for the choice Y<sup>1</sup> = U (j) <sup>t</sup>+1 and Y = U (j) t and Z as Z = A<sup>∗</sup>A(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ) (j) . This allows to upper-bound the norm of G (j) 2 as follows

$$\begin{aligned} \|G_2^{(j)}\| &\leq \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\| \left(1 - \mu \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^2 + \mu \|(\mathcal{A}^* \mathcal{A}(\boldsymbol{x} * \boldsymbol{X}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top))^{(j)} - (\bar{\boldsymbol{X}}^{(j)} \bar{\boldsymbol{X}}^{(j) \text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j) \text{H}})\|\right) \\ &\quad + \mu^2 \left( \|\bar{U}_t^{(j)} \bar{W}_t^{(j)}\|^2 + \|(\mathcal{A}^* \mathcal{A}(\boldsymbol{x} * \boldsymbol{X}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top))^{(j)} - (\bar{\boldsymbol{X}}^{(j)} \bar{\boldsymbol{X}}^{(j) \text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j) \text{H}})\|\right) \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^3 \end{aligned}$$

Using now the fact that for each j it holds that

$$\|(\mathcal{A}^*\mathcal{A}(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top))^{(j)} - (\bar{\boldsymbol{x}}^{(j)}\bar{\boldsymbol{X}}^{(j)\mathbf{H}} - \bar{\boldsymbol{U}}_t^{(j)}\bar{\boldsymbol{U}}_t^{(j)\mathbf{H}})\| \leq \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|$$

and that ∥U (j) <sup>t</sup> ∥ ≤ ∥Ut∥ ≤ 3∥X ∥, we can proceed with the bound for the norm of G (j) 2 as below

$$\begin{aligned} \|G_2^{(j)}\| &\leq \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\| \left(1 - \mu \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^2 + \mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|\right) \\ &\quad + \mu^2 \left(9 \|\boldsymbol{x}\|^2 + \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|\right) \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^3 \end{aligned}$$

Further, using the assumption <sup>µ</sup> <sup>≤</sup> <sup>c</sup><sup>1</sup> min n <sup>10</sup> ∥X ∥ −2 , ∥(A<sup>∗</sup>A − I)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t −1 o , we get

$$\begin{aligned} \|G_2^{(j)}\| &\leq \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\| \left(1 - \mu \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^2 + \mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|\right) + \frac{\mu}{2} \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^2 \\ &= \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\| \left(1 - \frac{\mu}{2} \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|^2 + \mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|\right). \end{aligned}$$

Now, let us return to the first summand in [\(E.5\)](#page-30-1), that is V ⊤ <sup>X</sup> ∗ Ut+1 ∗ W<sup>t</sup> ∗ W<sup>⊤</sup> <sup>t</sup> ∗ Wt+1,⊥. Using again the fact that V<sup>X</sup> ∗ Ut+1 ∗ Wt+1,<sup>⊥</sup> = 0 allows us to rewrite it as

$$\boldsymbol{\nu}_{\boldsymbol{x}}^{\top} * \boldsymbol{u}_{t+1} * \boldsymbol{w}_t * \boldsymbol{w}_t^{\top} * \boldsymbol{w}_{t+1,\perp} = -\boldsymbol{\nu}_{\boldsymbol{x}}^{\top} * \boldsymbol{u}_{t+1} * \boldsymbol{w}_{t,\perp} * \boldsymbol{w}_{t,\perp}^{\top} * \boldsymbol{w}_{t+1,\perp} \quad (\text{E.6})$$

Moreover, for the same summand, the corresponding jth slice in the Fourier domain reads as

$$\bar{V}_{\mathbf{x}^\perp}^{(j)\text{H}} \bar{U}_{t+1}^{(j)} \bar{W}_t^{(j)} \bar{W}_t^{(j)\text{H}} \bar{W}_{t+1,\perp}^{(j)} := G_1^{(j)}.$$

Due to relation [\(E.6\)](#page-31-0) in the tensor domain, in the Fourier domain it holds that

$$\bar{V}_{\boldsymbol{x}}^{(j)\text{H}}\bar{U}_{t+1}^{(j)}\bar{W}_t^{(j)}\bar{W}_t^{(j)\text{H}}\bar{W}_{t+1,\perp}^{(j)} = -\bar{V}_{\boldsymbol{x}}^{(j)\text{H}}\bar{U}_{t+1}^{(j)}\bar{W}_{t,\perp}^{(j)}\bar{W}_{t,\perp}^{(j)\text{H}}\bar{W}_{t+1,\perp}^{(j)}.$$

which allows to represent W (j)H <sup>t</sup> W (j) <sup>t</sup>+1,<sup>⊥</sup> as

$$\overline{W}_t^{(j)\text{H}} \overline{W}_{t+1,\perp}^{(j)} = -\left(\overline{V}_{\boldsymbol{x}}^{(j)\text{H}} \overline{U}_{t+1}^{(j)} \overline{W}_t^{(j)}\right)^{-1} \overline{V}_{\boldsymbol{x}}^{(j)\text{H}} \overline{U}_{t+1}^{(j)} \overline{W}_{t,\perp}^{(j)} \overline{W}_{t,\perp}^{(j)} \overline{W}_{t+1,\perp}^{(j)}.$$

Note that the matrix on the RHS above is invertible due to the assumption that V ⊤ <sup>X</sup> ∗ Ut+1 ∗ W<sup>t</sup> has full tubal rank with all invertible t-SVD-tubes. From here, G (j) 1 can be represented as

$$G_1^{(j)} = \overline{V}_{\boldsymbol{x}^\perp}^{(j)\text{H}} \overline{U}_{t+1}^{(j)} \overline{W}_t^{(j)} \left( \overline{V}_{\boldsymbol{x}}^{(j)\text{H}} \overline{U}_{t+1}^{(j)} \overline{W}_t^{(j)} \right)^{-1} \overline{V}_{\boldsymbol{x}}^{(j)\text{H}} \overline{U}_{t+1}^{(j)} \overline{W}_{t,\perp}^{(j)} \overline{W}_{t,\perp}^{(j)\text{H}} \overline{W}_{t+1,\perp}^{(j)}.$$

According to Lemma [H.3,](#page-52-0) the norm of G (j) 1 can be bounded from above as

$$\begin{aligned} \|G_1^{(j)}\| &\leq 2\mu\left(\|\overline{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\overline{U}_t^{(j)}\overline{W}_t^{(j)}}\|\|\overline{U}_t^{(j)}\overline{W}_t^{(j)}\|^2 + \|(\mathcal{A}^*\mathcal{A}(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top))^{(j)} - (\overline{X}^{(j)}\overline{X}^{(j)\text{H}} - \overline{U}_t^{(j)}\overline{U}_t^{(j)\text{H}})\|\right) \cdot \\ &\quad \cdot \|\overline{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\overline{U}_{t+1}^{(j)}\overline{W}_t^{(j)}}\|\|\overline{U}_t^{(j)}\overline{W}_{t,\perp}^{(j)}\| \\ &\leq 2\mu\left(\|\overline{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\overline{U}_t^{(j)}\overline{W}_t^{(j)}}\|\|\overline{U}_t^{(j)}\|^2 + \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top)\|\right) \cdot \|\overline{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\overline{U}_{t+1}^{(j)}\overline{W}_t^{(j)}}\|\|\overline{U}_t^{(j)}\overline{W}_{t,\perp}^{(j)}\| \\ &\leq 2\mu\left(\|\overline{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\overline{U}_t^{(j)}\overline{W}_t^{(j)}}\|\|\overline{U}_t^{(j)}\|^2 + \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top)\|\right) \cdot \|\mathbf{V}_{\mathcal{X}^\perp}^\top*\mathbf{V}_{\mathcal{U}_{t+1}*\mathbf{W}_t}\|\|\overline{U}_t^{(j)}\overline{W}_{t,\perp}^{(j)}\| \end{aligned}$$

Due to ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t+1∗W<sup>t</sup> ∥ ≤ <sup>1</sup> <sup>50</sup> from Lemma [E.2,](#page-29-1) the fact that ∥U (j) <sup>t</sup> ∥ ≤ ∥Ut∥, and our assumption that ∥Ut∥ ≤ 3∥X ∥, the norm of G (j) 1 can be further bounded as

$$\begin{aligned} \|G_1^{(j)}\| &\leq \mu \left( 9 \|\overline{V}_{\mathcal{X}^\perp}^{(j) \text{H}} V_{\overline{U}_t^{(j)} \overline{W}_t^{(j)}}\| \|\mathcal{X}\|^2 + \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \right) \|\overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)}\| \\ &= \mu \left( 9 \|\overline{\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}}\|^{(j)} \|\|\mathcal{X}\|^2 + \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \right) \|\overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)}\|. \end{aligned}$$

Since due to representation [\(E.4\)](#page-30-2), it holds that ∥ Ut+1 ∗ Wt+1,<sup>⊥</sup> (j)∥ = ∥ V<sup>X</sup> <sup>⊥</sup> ∗ Ut+1 ∗ Wt+1,<sup>⊥</sup> (j)∥, combining the inequalities for ∥G (j) 1 ∥ and ∥G (j) ∥ together with U (j) <sup>t</sup> W (j) t,<sup>⊥</sup> = U<sup>t</sup> ∗ Wt,<sup>⊥</sup> (j) leads to the final result

$$\begin{aligned} \|(\mathbf{u}_{t+1} * \mathbf{w}_{t+1,\perp})^{(j)}\| &\leq \left(1 - \frac{\mu}{2} \|(\mathbf{u}_t * \mathbf{w}_{t,\perp})^{(j)}\|^2 + 9\mu \|(\mathbf{v}_{\mathbf{x}^\perp} * \mathbf{v}_{\mathbf{u}_t * \mathbf{w}_t})^{(j)}\| \|\mathbf{x}\|^2 \right. \\ &\quad \left. + 2\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\| \right) \|(\mathbf{u}_t * \mathbf{w}_{t,\perp})^{(j)}\|. \end{aligned}$$

The next lemma shows that the tensors W<sup>t</sup> and Wt+1 span approximately the same tensor column space.

Lemma E.4. *Assume that the following conditions hold*

$$\|\boldsymbol{u}_t\| \leq 3\|\boldsymbol{x}\|, \quad (\text{E.7})$$

$$\mu \leq c \| \boldsymbol{\mathcal{X}} \|^{-2} \kappa^{-2} \quad (\text{E.8})$$

$$\| \mathcal{V}_{\boldsymbol{\chi}^\perp}^\top * \mathcal{V}_{\boldsymbol{u}_t * \boldsymbol{w}_t} \| \leq c \kappa^{-1} \quad (\text{E.9})$$

$$\|\overline{\mathcal{U}_t * \mathcal{W}_{t,\perp}}^{(j)}\| \leq 2\sigma_{\min}(\overline{\mathcal{U}_t * \mathcal{W}_t}^{(j)}), \quad (\text{E.10})$$

$$\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \leq c\sigma_{min}^2(\bar{\boldsymbol{x}}). \quad (\text{E.11})$$

*Then it holds that*

$$\| \mathcal{W}_{t,\perp}^\top * \mathcal{W}_{t+1} \| \leq \mu \left( \frac{1}{4800} \sigma_{min}^2(\bar{\mathcal{X}}) + \| \mathcal{U}_t * \mathcal{W}_t \| \| \mathcal{U}_t * \mathcal{W}_{t,\perp} \| \right) \| \mathcal{V}_{\mathcal{X}^\perp} * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} \| + 4\mu \| (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) \|$$

$$\|\mathcal{W}_{t,\perp}^\top * \mathcal{W}_{t+1}\| \leq \mu \left( \frac{1}{4800} \sigma_{\min}^2(\overline{\mathcal{X}}) + \|\mathcal{U}_t * \mathcal{W}_t\| \|\mathcal{U}_t * \mathcal{W}_{t,\perp}\| \right) \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| + 4\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\|$$
and  $\sigma_{\min}(\overline{\mathcal{W}_t^\top * \mathcal{W}_{t+1}}^{(j)}) \geq \frac{1}{2}$ ,  $1 \leq j \leq k$ .

*Proof.* To bound the norm of W<sup>⊤</sup> t,<sup>⊥</sup> ∗ Wt+1, we will rewrite W<sup>⊤</sup> t,<sup>⊥</sup> ∗ Wt+1 in the Fourier domain with the help of Fourier slices of V ⊤ <sup>X</sup> ∗ Ut. First, note that due to the decomposition of the gradient iterates into the noise and signal term, it holds V ⊤ <sup>X</sup> ∗Ut+1 = V ⊤ <sup>X</sup> ∗Ut+1 ∗Wt+1 ∗W<sup>⊤</sup> <sup>t</sup>+1. This allows us to represent the corresponding jth Fourier slices of V ⊤ <sup>X</sup> ∗Ut+1 as V (j)H <sup>X</sup> U (j) <sup>t</sup>+1 = V (j)H <sup>X</sup> U (j) <sup>t</sup>+1W (j) <sup>t</sup>+1W (j)H <sup>t</sup>+1 , which means that for each j, the matrices V (j)H <sup>X</sup> U (j) <sup>t</sup>+1 and V (j)H <sup>X</sup> U (j) <sup>t</sup>+1W (j) <sup>t</sup>+1W (j)H t+1 have the same kernel, and therefore U (j)H <sup>t</sup>+1 V (j) <sup>X</sup> spans the same subspace as W (j) <sup>t</sup>+1W (j)H <sup>t</sup>+1 U (j)H <sup>t</sup>+1 V (j) <sup>X</sup> . Due to this and the following representation of the matrices

$$\bar{U}_t^{(j)} = \bar{U}_t^{(j)} \bar{W}_t^{(j)} \bar{W}_t^{(j)\text{H}} + \bar{U}_t^{(j)} \bar{W}_t^{(j)} \bar{W}_t^{(j)\text{H}} \quad (\text{E.12})$$

$$\bar{U}_{t+1}^{(j)} = \bar{U}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)\text{H}} + \bar{U}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)\text{H}}, \quad (\text{E.13})$$

we can apply Lemma [H.4](#page-53-1) to estimate the norm of W<sup>H</sup> t,⊥W (j) <sup>t</sup>+1 taking Y<sup>1</sup> = U (j) <sup>t</sup>+1 and Y = U (j) t and Z as

$$Z^{(j)} := \overline{(\mathcal{A}^* \mathcal{A}(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top))^{(j)}}.$$

This gives us the following estimate

$$\begin{aligned} \|\bar{W}_{t,\perp}^{\text{H}} \bar{W}_{t+1}^{(j)}\| &\leq \mu \left( 1 + \mu \frac{\|Z^{(j)}\| \|\bar{U}_t^{(j)} \bar{W}_t^{(j)}\|}{\sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} \bar{U}_{t+1}^{(j)})} \right) \|\bar{U}_t^{(j)} \bar{W}_t^{(j)}\| \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\| \|\bar{V}_{\mathcal{X}}^{(j)\text{H}} V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}}\| \\ &\quad + \mu \frac{\|Z^{(j)} - (\bar{X}^{(j)} \bar{X}^{(j)\text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j)\text{H}})\|}{\sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} \bar{U}_{t+1}^{(j)})} \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|. \end{aligned} \quad (\text{E.14})$$

To proceed further with the upper bound above, we will first show that in each Fourier slice it holds that

$$\sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)H}\bar{U}_{t+1}^{(j)}) \geq \frac{1}{2}\sigma_{\min}(\bar{U}_t^{(j)}\bar{W}_t^{(j)}), \quad 1 \leq j \leq k. \quad (\text{E.15})$$

First, note that

$$\begin{aligned} \sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} \bar{U}_{t+1}^{(j)}) &\geq \sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} \bar{U}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)}) = \sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} (\text{Id} + \mu Z^{(j)}) \bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}) \\ &= \sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} (\text{Id} + \mu Z^{(j)}) V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}} V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}}^{\text{H}} \bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}) \\ &\geq \sigma_{\min}\left(\bar{V}_{\mathcal{X}}^{(j)\text{H}} (\text{Id} + \mu Z^{(j)}) V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}}\right) \cdot \sigma_{\min}(V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}}^{\text{H}} \bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}) \\ &\geq \left(\sigma_{\min}(\bar{V}_{\mathcal{X}}^{(j)\text{H}} V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}}) - \mu \|\bar{V}_{\mathcal{X}}^{(j)\text{H}} Z^{(j)} V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}}\|\right) \cdot \sigma_{\min}(V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}}^{\text{H}} \bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}). \end{aligned}$$

Due to our assumption [\(E.9\)](#page-32-0) on the principal angle ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ and the properties of the tensor slices, we have that

$$\sigma_{\min}\left(\overline{V}_{\mathbf{x}}^{(j)\text{H}}V_{\overline{U}_t^{(j)}\overline{W}_{t+1}^{(j)}}\right) \geq \sigma_{\min}\left(\overline{V}_{\mathbf{x}}^{\top} * \mathbf{v}_{u_t * \mathbf{w}_{t+1}}\right) = \sqrt{1 - \left\|\overline{\mathbf{v}}_{\mathbf{x}}^{\top} * \mathbf{v}_{u_t * \mathbf{w}_{t+1}}\right\|^2} \geq \frac{3}{4},$$

where that last inequality can be guaranteed by choosing c > 0 small enough. Thus, to show that relation [\(E.15\)](#page-33-0) holds we need to demonstrate that µ V (j)H <sup>X</sup> Z (j)V<sup>U</sup> <sup>t</sup> <sup>W</sup>(j) t+1 be bounded from above by <sup>1</sup> 4 . For this, we will proceed as follows

$$\|\mu\| \overline{V}_{\mathcal{X}}^{(j)\text{H}} Z^{(j)} V_{\overline{U}_t^{(j)} \overline{W}_{t+1}^{(j)}} \| \leq \mu \|Z^{(j)}\| \leq \mu \|Z^{(j)} - (\overline{X}^{(j)} \overline{X}^{(j)\text{H}} - \overline{U}_t^{(j)} \overline{U}_t^{(j)\text{H}})\| + \mu \|\overline{X}^{(j)} \overline{X}^{(j)\text{H}} - \overline{U}_t^{(j)} \overline{U}_t^{(j)\text{H}}\|. \quad (\text{E.16})$$

By the definition of Z (j) , for the first summand from above we have

$$\begin{aligned} \left\| Z^{(j)} - (\bar{X}^{(j)} \bar{X}^{(j)\text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j)\text{H}}) \right\| &= \left\| (\mathcal{A}^* \mathcal{A} (\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top))^{(j)} - (\bar{X}^{(j)} \bar{X}^{(j)\text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j)\text{H}}) \right\| \\ &= \left\| (\overline{\mathcal{I}} - \mathcal{A}^* \mathcal{A}) (\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)^{(j)} \right\| \\ &\leq \left\| (\overline{\mathcal{I}} - \mathcal{A}^* \mathcal{A}) (\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) \right\| \end{aligned}$$

and for the second summand, it holds that

$$\|\bar{X}^{(j)}\bar{X}^{(j)\text{H}} - \bar{U}_t^{(j)}\bar{U}_t^{(j)\text{H}}\| \leq \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\| \leq \|\boldsymbol{x}\|^2 + \|\boldsymbol{u}_t\|^2.$$

This allows us to proceed with inequality [\(E.16\)](#page-33-1) as

$$\begin{aligned} \mu \|\bar{V}_{\boldsymbol{X}}^{(j)\text{H}} Z^{(j)} V_{\bar{U}_t^{(j)} \bar{W}_{t+1}^{(j)}} \| &\leq \mu \|(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| + \mu(\|\boldsymbol{x}\|^2 + \|\boldsymbol{u}_t\|^2) \\ &\leq \mu \|(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| + 10\mu \|\boldsymbol{x}\|^2 \leq \mu c \sigma_{\min}^2(\bar{\boldsymbol{X}}) + 11\mu \|\boldsymbol{x}\|^2 \leq \frac{1}{2}, \end{aligned}$$

where in the first line we used assumption [\(E.7\)](#page-32-1), and in the second assumption[\(E.11\)](#page-32-2). The third inequality above follows from our assumption on µ and sufficiently small constant c > 0. This, in turn, shows that relation [\(E.15\)](#page-33-0) holds and we can proceed with [\(E.14\)](#page-33-2) in the following manner

$$\begin{aligned} \| \overline{W}_{t,\perp}^{\text{H}} \overline{W}_{t+1}^{(j)} \| &\leq \mu \left( 1 + 2\mu \frac{\| Z^{(j)} \| \| \overline{U}_t^{(j)} \overline{W}_t^{(j)} \|}{\sigma_{\min}(\overline{U}_t^{(j)} \overline{W}_t^{(j)})} \right) \| \overline{U}_t^{(j)} \overline{W}_t^{(j)} \| \| \| \overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)} \| \| \| \overline{V}_{\mathbf{X}}^{(j)\text{H}} V_{\overline{U}_t^{(j)} \overline{W}_t^{(j)}} \| \\ &\quad + 2\mu \frac{\| Z^{(j)} - (\overline{X}^{(j)} \overline{X}^{(j)\text{H}} - \overline{U}_t^{(j)} \overline{U}_t^{(j)\text{H}}) \|}{\sigma_{\min}(\overline{U}_t^{(j)} \overline{W}_t^{(j)})} \| \overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)} \|. \end{aligned}$$

Now, using assumption [\(E.10\)](#page-32-3) and the definition of Z (j) , we have

$$\begin{aligned} \|\bar{W}_{t,\perp}^{(j)}\bar{W}_{t+1}^{(j)}\| &\leq \mu\|\bar{V}_{\mathcal{X}^\perp}^{(j)}V_{\bar{U}_t^{(j)}\bar{W}_t^{(j)}}\| \|\bar{U}_t^{(j)}\bar{W}_t^{(j)}\| \|\bar{U}_t^{(j)}\bar{W}_{t,\perp}^{(j)}\| \\ &\quad + 4\mu\|(\mathcal{A}^*\mathcal{A}(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top))^{(j)} - (\bar{\mathcal{X}}^{(j)}\bar{\mathcal{X}}^{(j)\text{H}} - \bar{U}_t^{(j)}\bar{U}_t^{\text{H}})\| \\ &\quad + 4\mu^2\|(\mathcal{A}^*\mathcal{A}(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top))^{(j)}\| \|\bar{U}_t^{(j)}\bar{W}_t^{(j)}\|^2\|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\bar{U}_t^{(j)}\bar{W}_t^{(j)}}\| \\ &\leq \mu\|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\bar{U}_t^{(j)}\bar{W}_t^{(j)}}\| \|\bar{U}_t^{(j)}\bar{W}_t^{(j)}\| \|\bar{U}_t^{(j)}\bar{W}_{t,\perp}^{(j)}\| \\ &\quad + 4\mu\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top)\| \\ &\quad + 4\mu^2\|\mathcal{A}^*\mathcal{A}(\mathcal{X}*\mathcal{X}^\top - \mathcal{U}_t*\mathcal{U}_t^\top)\| \|\bar{U}_t^{(j)}\bar{W}_t^{(j)}\|^2\|\bar{V}_{\mathcal{X}^\perp}^{(j)\text{H}}V_{\bar{U}_t^{(j)}\bar{W}_t^{(j)}}\|. \end{aligned}$$

In the last inequality, we used the tensor norm as the maximum norm in each Fourier slice. Note that, similarly to one of the estimates above, we get

$$\begin{aligned} \|\mathcal{A}^*\mathcal{A}(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| &\leq \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\| + \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \\ &\leq \|\boldsymbol{x}\|^2 + \|\boldsymbol{u}_t\|^2 + c\sigma_{\min}^2(\bar{\boldsymbol{x}}) \leq 11\|\boldsymbol{x}\|^2 \end{aligned} \quad (\text{E.17})$$

where the last line holds due to the assumption ∥Ut∥ ≤ 3∥X ∥ and that c is small enough.

Now, since µ ≤ c∥X ∥ <sup>−</sup><sup>2</sup>κ −2 , ∥U (j) <sup>t</sup> W (j) <sup>t</sup> ∥ ≤ ∥Ut∥ ≤ 3∥X ∥ and ∥U (j) <sup>t</sup> W (j) t,<sup>⊥</sup>∥ ≤ ∥Ut∥ ≤ 3∥X ∥, constant c > 0 can be chosen so that 4µ · 11∥X ∥ <sup>2</sup> ≤ 1 <sup>4800</sup>σ min(X ), together with [\(E.17\)](#page-34-1) and [\(E.11\)](#page-32-2) we can proceed with the estimation of W<sup>H</sup> t,⊥W (j) <sup>t</sup>+1 as

$$\| \overline{W}_{t,\perp}^{(j)\text{H}} \overline{W}_{t+1}^{(j)} \| \leq \mu \left( \frac{1}{48000} \sigma_{\min}^2(\overline{\boldsymbol{X}}) + 9 \| \boldsymbol{x} \|^2 \right) \| \overline{V}_{\boldsymbol{X}^{\perp}}^{(j)\text{H}} \overline{V}_{\overline{U}_t^{(j)} \overline{W}_t^{(j)}} \| + 4\mu c \sigma_{\min}^2(\overline{\boldsymbol{X}}).$$

Using the assumption µ ≤ c∥X ∥ −2 and choosing c > 0 small enough, we obtain that ∥W (j)H t,<sup>⊥</sup> W (j) <sup>t</sup>+1∥ ≤ <sup>1</sup> 2 . Note that this implies that σmin(W<sup>⊤</sup> <sup>t</sup> ∗ Wt+1 (j) ) = q 1 − ∥W (j)H t,<sup>⊥</sup> W (j) <sup>t</sup>+1∥ <sup>2</sup> ≥ 1 2 , which finishes the proof.

Lemma E.5. *Assume that the following conditions hold*

$$\|\overline{\mathcal{U}_t * \mathcal{W}_{t,\perp}}^{(j)}\| \leq 2\sigma_{min}(\overline{\mathcal{U}_t * \mathcal{W}_t}^{(j)}), \quad (\text{E.18})$$

$$\|\boldsymbol{u}_t\| \leq 3\|\boldsymbol{x}\|, \quad (\text{E.19})$$

$$\|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| \leq \tilde{c} \quad (\text{E.20})$$

$$\mu \leq c \| \boldsymbol{X} \|^{-2} \kappa^{-2} \quad (\text{E.21})$$

$$\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\| \leq c\kappa^{-2} \|\mathcal{X}\| \quad (\text{E.22})$$

$$\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \leq c\sigma_{min}^2(\bar{\boldsymbol{X}}). \quad (\text{E.23})$$

*Then the angle between the column space of the signal term* U<sup>t</sup> ∗ W<sup>t</sup> *and column space of* X *stays sufficiently small from one iteration to another, namely*

$$\begin{aligned} \| \boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{\nu}_{\boldsymbol{u}_{t+1}*\boldsymbol{w}_{t+1}} \| &\leq \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\bar{\boldsymbol{x}}) \right) \| \boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{\nu}_{\boldsymbol{u}_t*\boldsymbol{w}_t} \| \\ &\quad + 150\mu \| (A^* \mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top) \| + 500\mu^2 \| \boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top \|^2. \end{aligned}$$

*Proof.* To estimate the principal angle ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t+1∗Wt+1 ∥, we first investigate the tensor-column subspace of Ut+1 ∗ Wt+1. By the definition of Ut+1 and W<sup>t</sup> ∗ W<sup>⊤</sup> <sup>t</sup> <sup>+</sup> Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> = I, we have

$$\begin{aligned} \mathcal{U}_{t+1} * \mathcal{W}_{t+1} &= \left( \mathcal{I} + \mu(\mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) \right) * \mathcal{U}_t * \mathcal{W}_{t+1} \\ &= (\mathcal{I} + \mu \mathcal{Z}) * \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{W}_{t+1} + (\mathcal{I} + \mu \mathcal{Z}) * \mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top * \mathcal{W}_{t+1}. \end{aligned}$$

where we use notation Z := (A<sup>∗</sup>A)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ). This allows to represent jth slice of Ut+1 ∗Wt+1 in the Fourier domain as

$$\bar{U}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)} = (\text{Id} + \mu \bar{Z}^{(j)}) \bar{U}_t^{(j)} \bar{W}_t^{(j)} \bar{W}_t^{(j)\text{H}} \bar{W}_{t+1}^{(j)} + (\text{Id} + \mu \bar{Z}^{(j)}) \bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)} \bar{W}_{t,\perp}^{(j)\text{H}} \bar{W}_{t+1}^{(j)}.$$

with Z (j) = (A<sup>∗</sup>A)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ) (j) . Because of this representation and decomposition [\(E.12\)](#page-32-4), to bound the principal angle between Ut+1 ∗ Wt+1 and X , we want to apply inequality [\(H.5\)](#page-53-2) from Lemma [H.4,](#page-53-1) but for this we first need to check whether for

$$P^{(j)} := \bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)} \bar{W}_{t,\perp}^{(j) \text{H}} \bar{W}_{t+1}^{(j)} \left( V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}}^{\text{H}} \bar{U}_t^{(j)} \bar{W}_t^{(j)} \bar{W}_t^{(j) \text{H}} \bar{W}_{t+1}^{(j)} \right)^{-1} V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}}^{\text{H}}$$

the following applies

$$\|\mu Z^{(j)} + P^{(j)} + \mu Z^{(j)} P^{(j)}\| \leq 1.$$

For convenience, we denote B(j) := µZ(j) + P (j) + µZ(j)P (j) . Using the triangular inequality and submultiplicativity of the norm, we bet the first simple bound on the norm of B(j)

$$\|B^{(j)}\| \leq \mu \|Z^{(j)}\| + (1 + \mu \|Z^{(j)}\|) \|P^{(j)}\| \quad (\text{E.24})$$

Note that P (j) can be rewritten as

$$P^{(j)} = \overline{U}_t^{(j)} \overline{W}_{t,\perp}^{(j)} \overline{W}_{t,\perp}^{(j) \text{H}} \overline{W}_{t+1}^{(j)} \left( \overline{W}_t^{(j) \text{H}} \overline{W}_{t+1}^{(j)} \right)^{-1} \left( V_{\overline{U}_t^{(j)} \overline{W}_t^{(j)}}^{\text{H}} \overline{U}_t^{(j)} \overline{W}_t^{(j)} \right)^{-1} V_{\overline{U}_t^{(j)} \overline{W}_t^{(j)}}^{\text{H}},$$

which allows for the following estimate of its norm

$$\begin{aligned} \|P^{(j)}\| &\leq \|\bar{U}_{t,\perp}^{(j)} \bar{W}_{t,\perp}^{(j)} \| \|\bar{W}_{t,\perp}^{(j) \text{H}} \bar{W}_{t,\perp}^{(j)}\| \left( \left\| \left( \bar{W}_{t,\perp}^{(j) \text{H}} \bar{W}_{t,\perp}^{(j)} \right)^{-1} \right\| \left\| \left( V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}}^{\text{H}} \bar{U}_t^{(j)} \bar{W}_t^{(j)} \right)^{-1} \right\| \| V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}}^{\text{H}} \right\| \\ &\leq \frac{\|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\| \|\bar{W}_{t,\perp}^{(j) \text{H}} \bar{W}_{t,\perp}^{(j)}\|}{\sigma_{\min}(\bar{W}_t^{(j) \text{H}} \bar{W}_{t,\perp}^{(j)}) \cdot \sigma_{\min}(\bar{U}_t^{(j)} \bar{W}_t^{(j)})}. \end{aligned}$$

From here, using assumption [\(E.18\)](#page-34-2) and a lower bound on σmin(W (j)H <sup>t</sup> W (j) <sup>t</sup>+1) from Lemma [E.4,](#page-32-5) we get

$$\|P^{(j)}\| \leq 4\|\overline{W}_{t,\perp}^{(j)\text{H}}\overline{W}_{t+1}^{(j)}\|. \quad (\text{E.25})$$

Using this and the definition of Z (j) , we have

$$\|B^{(j)}\| \leq \mu \|(\mathcal{A}^*\mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)^{(j)}\| + 4 \left(1 + \mu \|(\mathcal{A}^*\mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)^{(j)}\|\right) \|\overline{W}_{t,\perp}^{(j) \text{H}} \overline{W}_{t+1}^{(j)}\|. \quad (\text{E.26})$$

Due to the assumption on µ, we can bound µ∥(A<sup>∗</sup>A)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ) (j)∥ as follows

$$\begin{aligned} \mu \|(\mathcal{A}^*\mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)^{(j)}\| &\leq \mu \|(\mathcal{A}^*\mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)^{(j)}\| \\ &\leq \mu \|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| + \mu \|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\| \\ &\leq \mu (c\sigma_{\min}^2(\overline{\mathcal{X}}) + 10\|\mathcal{X}\|^2) \leq 1 \end{aligned}$$

where in the two last inequalities we use assumptions [\(E.23\)](#page-34-3), [\(E.19\)](#page-34-4) and [\(E.21\)](#page-34-5) with the fact for the learning rate constant c > 0 can be chosen sufficiently small.

This, in turn, allows us to proceed with inequality [\(E.26\)](#page-35-0) as

$$\|B^{(j)}\| \leq \mu \|(\mathcal{A} * \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)^{(j)}\| + 8 \|\overline{W}_{t,\perp}^{(j)H} \overline{W}_{t+1}^{(j)}\|. \quad (\text{E.27})$$

Now, applying the bound on ∥W (j)H t,<sup>⊥</sup> W (j) t+1∥ ≤ ∥W<sup>⊤</sup> t,<sup>⊥</sup> ∗ Wt+1∥ from Lemma [E.4](#page-32-5) and similar transformation for ∥(A<sup>∗</sup>A)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t ) (j)∥ as above, we come the following result in [\(E.27\)](#page-36-0)

$$\begin{aligned} \|B^{(j)}\| &\leq \mu \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\| + \mu \left( \frac{1}{600} \sigma_{\min}(\bar{\boldsymbol{X}})^2 + 8 \|\boldsymbol{u}_t * \boldsymbol{w}_t\| \|\boldsymbol{u}_t * \boldsymbol{w}_{t,\perp}\| \right) \|\boldsymbol{v}_{\boldsymbol{x}^\perp} * \boldsymbol{v}_{\boldsymbol{u}_t * \boldsymbol{w}_t}\| \\ &\quad + 33 \mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \end{aligned}$$

To show that this bound above can be made smaller than one, we use assumptions [\(E.22\)](#page-34-6), [\(E.23\)](#page-34-3) and that ∥U<sup>t</sup> ∗ Wt∥ ≤ ∥U∥ ≤ 2∥X ∥, which leads to

$$\begin{aligned} \|B^{(j)}\| &\leq \mu \|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\| + \mu \left( \frac{1}{600} \sigma_{\min}(\bar{\mathcal{X}})^2 + 8 c \frac{\sigma_{\min}(\bar{\mathcal{X}})}{\kappa^2} \cdot 3 \|\mathcal{X}\| \right) \|\mathcal{V}_{\mathcal{X}^\perp} * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| + 33 \mu c \sigma_{\min}^2(\bar{\mathcal{X}}) \\ &\leq \mu 10 \|\mathcal{X}\|^2 + \mu c \frac{1}{300} \sigma_{\min}^2(\bar{\mathcal{X}}) + 33 \mu c \sigma_{\min}^2(\bar{\mathcal{X}}) \leq 1, \end{aligned}$$

with the last inequality following from the assumption on µ. In such a way, we check the conditions of Lemma [H.4](#page-53-1) to be able to apply inequality [\(H.5\)](#page-53-2). This gives

$$\begin{aligned} \|V_{\mathcal{X}^\perp}^{(j)\text{H}} V_{\bar{U}_{t+1}^{(j)} \bar{W}_{t+1}^{(j)}}\| &\leq \|V_{\mathcal{X}^\perp}^{(j)\text{H}} V_{\bar{U}_t^{(j)} \bar{W}_t^{(j)}}\| \left(1 - \frac{\mu}{2} \sigma_{\min}^2(\bar{X}^{(j)}) + \mu \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|\right) \\ &\quad + \mu \|Z^{(j)} - (\bar{X}^{(j)} \bar{X}^{(j)\text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j)\text{H}})\| + \left(1 + \mu \|Z^{(j)}\|\right) \frac{2\|\bar{W}_{t,\perp}^{(j)\text{H}} \bar{W}_{t+1}^{(j)}\| \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|}{\sigma_{\min}(\bar{W}_t^{(j)\text{H}} \bar{W}_{t+1}^{(j)}) \sigma_{\min}(\bar{U}_t^{(j)} \bar{W}_t^{(j)})} \\ &\quad + 57 \left(\mu \|Z^{(j)}\| + (1 + \mu \|Z^{(j)}\|) \frac{\|\bar{W}_{t,\perp}^{(j)\text{H}} \bar{W}_{t+1}^{(j)}\| \|\bar{U}_t^{(j)} \bar{W}_{t,\perp}^{(j)}\|}{\sigma_{\min}(\bar{W}_t^{(j)\text{H}} \bar{W}_{t+1}^{(j)}) \sigma_{\min}(\bar{U}_t^{(j)} \bar{W}_t^{(j)})}\right)^2. \end{aligned}$$

Applying again assumption [\(E.18\)](#page-34-2) and a lower bound on σmin(W (j)H <sup>t</sup> W (j) <sup>t</sup>+1) from Lemma [E.4](#page-32-5) as for [\(E.25\)](#page-35-1), in addition to [\(E.22\)](#page-34-6), we get

$$\begin{aligned} \|V_{\mathcal{X}^\perp}^{(j)\text{H}} V_{\bar{U}_{t+1}}^{(j)} \bar{W}_{t+1}^{(j)}\| &\leq \|V_{\mathcal{X}^\perp}^{(j)\text{H}} V_{\bar{U}_t}^{(j)} \bar{W}_t^{(j)}\| \left(1 - \frac{\mu}{3} \sigma_{\min}^2(\bar{X}^{(j)})\right) + \mu \|Z^{(j)} - (\bar{X}^{(j)} \bar{X}^{(j)\text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j)\text{H}})\| \\ &\quad + 8(1 + \mu \|Z^{(j)}\|) \|\bar{W}_{t,\perp}^{(j)\text{H}} \bar{W}_{t+1}^{(j)}\| + 57 \left(\mu \|Z^{(j)}\| + 4(1 + \mu \|Z^{(j)}\|) \|\bar{W}_{t,\perp}^{(j)\text{H}} \bar{W}_{t+1}^{(j)}\|\right)^2. \end{aligned}$$

Now, making 1 + µ∥Z (j)∥ ≤ 3 by choosing c > 0 small enough and using the properties of the terms involved, the above inequality gets the following view

$$\begin{aligned} \|V_{\boldsymbol{x}^\perp}^{(j)\text{H}} V_{\overline{U}_{t+1}^{(j)} \overline{W}_{t+1}^{(j)}}\| &\leq \|V_{\boldsymbol{x}^\perp}^{(j)\text{H}} V_{\overline{U}_t^{(j)} \overline{W}_t^{(j)}}\| \left(1 - \frac{\mu}{3} \sigma_{\min}^2(\bar{\boldsymbol{X}})\right) + \mu \|(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \\ &\quad + 32 \|\overline{W}_{t,\perp}^{(j)\text{H}} \overline{W}_{t+1}^{(j)}\| + 57 \left(\mu \|Z^{(j)}\| + 12 \|\overline{W}_{t,\perp}^{(j)\text{H}} \overline{W}_{t+1}^{(j)}\|\right)^2. \end{aligned} \quad (\text{E.28})$$

To proceed further with [\(E.28\)](#page-36-1), we will first do several auxiliary estimates. We start by bounding the norm ∥W (j)H t,<sup>⊥</sup> W (j) <sup>t</sup>+1∥. Since it holds that ∥W (j)H t,<sup>⊥</sup> W (j) t+1∥ ≤ ∥W<sup>⊤</sup> t,<sup>⊥</sup> ∗ Wt+1∥, from Lemma [E.4,](#page-32-5) one gets

$$\begin{aligned} \|\bar{W}_{t,\perp}^{(j)H} \bar{W}_{t+1}^{(j)}\| &\leq \mu \left( \frac{1}{4800} \sigma_{\min}^2(\bar{\mathcal{X}}) + \|\mathcal{U}_t * \mathcal{W}_t\| \|\mathcal{U}_t * \mathcal{W}_{t,\perp}\| \right) \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| \\ &\quad + 4\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \\ &\leq \mu \left( \frac{1}{4800} \sigma_{\min}^2(\bar{\mathcal{X}}) + 3c\sigma_{\min}^2(\bar{\mathcal{X}}) \right) \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| + 4\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \\ &\leq \frac{1}{2400} \mu \sigma_{\min}^2(\bar{\mathcal{X}}) \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| + 4\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \end{aligned} \quad (\text{E.29})$$

where we use in the second inequality that ∥U<sup>t</sup> ∗ Wt∥ ≤ ∥Ut∥ ≤ 3∥X ∥ and ∥U<sup>t</sup> ∗ Wt,<sup>⊥</sup>∥ ≤ cκ−<sup>2</sup>∥X ∥ by assumption, and in the last line that c > 0 can be chosen small enough. Using this estimate, let us bound from above the squared term in [\(E.28\)](#page-36-1) as follows

$$\begin{aligned} \mu\|Z^{(j)}\| + 12\|\bar{W}_{t,\perp}^{(j)\text{H}}\bar{W}_{t+1}^{(j)}\| &\leq \mu\|Z^{(j)}\| + \mu\frac{\sigma_{\min}^2(\bar{\mathcal{X}})}{200}\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{U}_t * \mathbf{w}_t}\| + 48\mu\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \\ &\leq \mu\|\bar{\mathcal{X}}^{(j)}\bar{\mathcal{X}}^{(j)\text{H}} - \bar{U}_t^{(j)}\bar{U}_t^{(j)\text{H}}\| + \mu\frac{\sigma_{\min}^2(\bar{\mathcal{X}})}{200}\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{v}_{\mathcal{U}_t * \mathbf{w}_t}\| \\ &\quad + 49\mu\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\|. \end{aligned}$$

From here, using Jensen's inequality, we obtain

$$\begin{aligned} (\mu\|Z^{(j)}\| + 12\|\overline{W}_{t,\perp}^{(j)\text{H}}\overline{W}_{t+1}^{(j)}\|)^2 &\leq 3\mu^2\|\overline{X}^{(j)}\overline{X}^{(j)\text{H}} - \overline{U}_t^{(j)}\overline{U}_t^{(j)\text{H}}\|^2 + 3\mu^2\frac{\sigma_{\min}^4(\overline{\mathcal{X}})}{200^2}\|\boldsymbol{\nu}_{\mathcal{X}^\perp} * \boldsymbol{\nu}_{\mathcal{U}_t * \boldsymbol{\omega}_t}\|^2 \\ &\quad + 3 \cdot 49^2\mu^2\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \boldsymbol{\mathcal{X}}^\top - \boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{U}}_t^\top)\|^2. \end{aligned}$$

Now, we can come back to bounding [\(E.28\)](#page-36-1) proceeding as follows

$$\begin{aligned} \|V_{\mathcal{X}^\perp}^{(j)\text{H}} V_{\overline{U}_{t+1}}^{(j)} \overline{W}_{t+1}^{(j)}\| &\leq \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| \left(1 - \frac{\mu}{3} \sigma_{\min}^2(\bar{\mathcal{X}}) + \frac{4\mu}{300} \sigma_{\min}^2(\bar{\mathcal{X}})\right) \\ &\quad + 129\mu \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \\ &\quad + 171\mu^2 \|\bar{\mathcal{X}}^{(j)} \bar{\mathcal{X}}^{(j)\text{H}} - \bar{U}_t^{(j)} \bar{U}_t^{(j)\text{H}}\|^2 + \mu^2 \frac{171\sigma_{\min}^4(\bar{\mathcal{X}})}{200^2} \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\|^2 \\ &\quad + 171 \cdot 49^2 \mu^2 \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\|^2 \\ &\leq \|\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}\| \left(1 - \frac{\mu}{3} \sigma_{\min}^2(\bar{\mathcal{X}}) + \frac{4\mu}{300} \sigma_{\min}^2(\bar{\mathcal{X}}) + \frac{171}{200^2} \kappa^{-4} \tilde{c} \cdot c \mu \sigma_{\min}^2(\bar{\mathcal{X}})\right) \\ &\quad + 171\mu^2 \|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\|^2 \\ &\quad + \mu(129 + 171 \cdot 49^2 c^2 \kappa^{-4}) \|(\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\|, \end{aligned}$$

where for the last inequality we used assumptions [\(E.23\)](#page-34-3), [\(E.20\)](#page-34-7) and [\(E.21\)](#page-34-5), and the properties of the tubal tensor norm. Now choosing constant c > 0 sufficiently small, we obtain that

$$\begin{aligned} \|V_{\boldsymbol{x}^\perp}^{(j)\text{H}} V_{\overline{U}_{t+1}^{(j)} \overline{W}_{t+1}^{(j)}} \| &\leq \left(1 - \frac{\mu}{4} \sigma_{\min}^2(\overline{\boldsymbol{x}})\right) \|\boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{\nu}_{\boldsymbol{u}_t * \boldsymbol{w}_t}\| + 200\mu^2 \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\|^2 \\ &\quad + 150 \|(\boldsymbol{A}^* \boldsymbol{A} - \boldsymbol{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|. \end{aligned}$$

Since the right-hand side of the above inequality is independent of j, we obtain the lemma statement.

The following lemma shows that under a mild condition the technical assumption

$$\| \boldsymbol{u}_{t+1} \| \leq 3 \| \boldsymbol{x} \|$$

needed in the lemmas above holds.

Lemma E.6. *Assume that* ∥Ut∥ ≤ 3∥X ∥*,* µ ≤ 1 <sup>27</sup> ∥X ∥ <sup>−</sup><sup>2</sup> *and that linear measurement operator* A *is such that*

$$\|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\| \leq \|\boldsymbol{x}\|^2$$

*Then for the iteration* t + 1*, it also holds* ∥Ut+1∥ ≤ 3∥X ∥*.*

*Proof.* Consider the gradient iterate

$$\begin{aligned}
\mathcal{U}_{t+1} &= \mathcal{U}_t + \mu A^* \mathcal{A}(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) * \mathcal{U}_t \\
&= \mathcal{U}_t + \mu(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) * \mathcal{U}_t + \mu(A^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) * \mathcal{U}_t \\
&= (\mathcal{I} - \mu \mathcal{U}_t * \mathcal{U}_t^\top) * \mathcal{U}_t + \mu \mathcal{X} * \mathcal{X}^\top * \mathcal{U}_t + \mu(A^* \mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) * \mathcal{U}_t.
\end{aligned}$$

To estimate the norm of Ut+1, we will bound each summand above separately. Due to the assumption on µ and the norm of Ut, we have µ ≤ 1 <sup>27</sup> ∥X ∥ <sup>−</sup><sup>2</sup> ≤ 3 ∥Ut∥ −2 . This allows us to estimate the tensor norm of (I − µU<sup>t</sup> ∗ U ⊤ t ) ∗ U<sup>t</sup> via the norm of matrix block representation in the Fourier domain. Namely, assume that matrix U<sup>t</sup> has the SVD U<sup>t</sup> = V ΣW<sup>H</sup>. Then for matrix (I − µU<sup>t</sup> ∗ U ⊤ t ) ∗ Ut, we have

$$\overline{(\mathcal{X} - \mu \mathcal{U}_t * \mathcal{U}_t^\top) * \mathcal{U}_t} = V\Sigma W^H - \mu V\Sigma W^H \Sigma \Sigma V^H V\Sigma W^H = V\Sigma W^H - \mu V\Sigma^3 W^H = V(\Sigma - \mu \Sigma^3) W^H.$$

From here, since µ ≤ 1 <sup>27</sup> ∥X ∥ <sup>−</sup><sup>2</sup> ≤ 3 ∥U∥ −2 and ∥Ut∥ = ∥Ut∥, it holds that ∥(I − µU<sup>t</sup> ∗ U ⊤ t ) ∗ Ut∥ = ∥Ut∥ − µ∥Ut∥ <sup>3</sup> = ∥Ut∥(1 − µ∥Ut∥ 2 ). Besides, from the submultiplicativity of the tensor norm and the triangle inequality, we obtain that

$$\|\mathbf{u}_{t+1}\| \leq (1 - \mu\|\mathbf{u}_t\|^2 + \mu\|\mathbf{x}\|^2 + \mu\|(A^*\mathcal{A} - \mathcal{I})(\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\|)\|\mathbf{u}_t\| \quad (\text{E.30})$$

$$\leq (1 - \mu\|\mathbf{u}_t\|^2 + 2\mu\|\boldsymbol{x}\|^2)\|\mathbf{u}_t\|, \quad (\text{E.31})$$

where in the last line we used the assumption on ∥(A<sup>∗</sup>A − I)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t )∥. By combining inequality [\(E.31\)](#page-38-0) with the assumption µ ≤ 1 <sup>27</sup>∥X∥<sup>2</sup> ≤ 1 <sup>3</sup>∥U∥<sup>2</sup> , we obtain that ∥Ut+1∥ ≤ 3∥X ∥ , which finishes the proof.

The following lemma shows that U<sup>t</sup> ∗W<sup>t</sup> ∗W<sup>⊤</sup> <sup>t</sup> ∗ U ⊤ t converges towards X ∗ X T , when projected onto the tensor column space of X .

Lemma E.7. *Assume that the following conditions hold*

$$\|\boldsymbol{u}_t\| \leq 3\|\boldsymbol{x}\| \quad (\text{E.32})$$

$$\mu \leq c \cdot \frac{1}{\sqrt{nk}} \cdot \kappa^{-2} \|\boldsymbol{X}\|^{-2} \quad (\text{E.33})$$

$$\sigma_{min}(\overline{\mathbf{U}}_t * \overline{\mathbf{W}}_t) \geq \frac{1}{\sqrt{10}} \sigma_{min}(\overline{\mathbf{X}}) \quad (\text{E.34})$$

$$\|\mathbf{v}_{\mathcal{X}^\perp} * \mathbf{v}_{\mathcal{U}_t * \mathbf{w}_t}\| \leq c\kappa^{-2} \quad (\text{E.35})$$

*and*

$$\max \left\{ \| \mathcal{V}_{\mathcal{X}}^\top * (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{V}_t) \|_F, \| \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}^\top * (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{V}_t) \|_F, \| (\mathcal{A}^* \mathcal{A} - \mathcal{I})(\mathcal{V}_t) \| \right\} \leq \kappa^{-2} \| \mathcal{V}_t \|_F$$

*with* Y<sup>t</sup> := X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t *. Then it holds that*

$$\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top\|_F \leq 3\|\mathbf{v}_{\mathcal{X}^\perp}^\top * (\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\|_F + \|\mathbf{u}_t * \mathbf{w}_{t,\perp} * \mathbf{w}_{t,\perp}^\top * \mathbf{u}_t^\top\|_F \quad (\text{E.36})$$

*as well as*

$$\|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\|_F \leq 4 \|\boldsymbol{v}_{\boldsymbol{x}^\top} * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|_F + \|\boldsymbol{u}_t * \boldsymbol{w}_{t,\perp} * \boldsymbol{w}_{t,\perp}^\top * \boldsymbol{u}_t^\top\|_F \quad (\text{E.37})$$

*and*

$$\begin{aligned} \|\mathbf{v}_{\mathcal{X}^\perp}(\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_{t+1} * \mathbf{u}_{t+1}^\top)\|_F &\leq \left(1 - \frac{\mu}{200} \sigma_{min}^2(\bar{\mathbf{X}})\right) \|\mathbf{v}_{\mathcal{X}^\perp} * (\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\|_F \\ &\quad + \mu \frac{\sigma_{min}^2(\bar{\mathbf{X}})}{100} \|\mathbf{u}_t * \mathbf{w}_{t,\perp} * \mathbf{w}_{t,\perp}^\top * \mathbf{u}_t^\top\|_F \end{aligned} \quad (\text{E.38})$$

*Proof.* We start by proving the first inequality [\(E.38\)](#page-38-1). For this, let us decompose V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ U<sup>t</sup> ∗ U ⊤ t as follows

$$\boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{u}_t * \boldsymbol{u}_t^\top = \boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{u}_t * \boldsymbol{u}_t^\top * \boldsymbol{\nu}_{\boldsymbol{x}} * \boldsymbol{\nu}_{\boldsymbol{x}}^\top + \boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{u}_t * \boldsymbol{u}_t^\top * \boldsymbol{\nu}_{\boldsymbol{x}^\perp} * \boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top,$$

then using the triangle inequality and submultiplicativity of the Frobenius and the spectral norm, we obtain

$$\begin{aligned}\|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top\|_F &\leq \|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}}\|_F + \|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp}\|_F \\ &\leq \|\mathbf{v}_{\mathcal{X}^\perp}^\top * (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top) * \mathbf{v}_{\mathcal{X}}\|_F + \|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp}\|_F \\ &\leq \|\mathbf{v}_{\mathcal{X}^\perp}^\top * (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\|_F + \|\mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp}\|_F,\end{aligned}\tag{E.39}$$

where in the second line, we used the orthogonality of the decomposition. Now, we will work additionally on bounding the norm of V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ U<sup>t</sup> ∗ U ⊤ <sup>t</sup> ∗ V<sup>X</sup> <sup>⊥</sup> to obtain [\(E.38\)](#page-38-1). Here, we will use the orthogonal decomposition with respect to W<sup>t</sup> and Wt,⊥, which leads to

$$\begin{aligned} \| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp} \|_F &\leq \| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{w}_t * \mathbf{w}_t^\top * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp} \|_F + \| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{w}_{t,\perp} * \mathbf{w}_{t,\perp}^\top * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp} \|_F \\ &\leq \| \mathbf{v}_{\mathcal{X}^\perp}^\top * \mathbf{u}_t * \mathbf{w}_t * \mathbf{w}_t^\top * \mathbf{u}_t^\top * \mathbf{v}_{\mathcal{X}^\perp} \|_F + \| \mathbf{u}_t * \mathbf{w}_{t,\perp} * \mathbf{w}_{t,\perp}^\top * \mathbf{u}_t^\top \|_F \end{aligned}$$

Now, for the first term above, we get

$$\begin{aligned} & \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top * \mathcal{V}_{\mathcal{X}^\perp} \|_F = \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}^\top * \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top * \mathcal{V}_{\mathcal{X}^\perp} \|_F \\ &= \sum_{j=1}^k \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t}^\top * \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top * \mathcal{V}_{\mathcal{X}^\perp}^{(j)} \|_F \\ &= \sum_{j=1}^k \| \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j)} \bar{\mathcal{W}}_t^{(j)} \bar{\mathcal{W}}_t^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j)} \|_F \\ &= \sum_{j=1}^k \| \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \left( \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \right)^{-1} \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j)} \bar{\mathcal{W}}_t^{(j)} \bar{\mathcal{W}}_t^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j)} \|_F \\ &\leq \max_{1 \leq j \leq k} \| \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \|_{1 \leq j \leq k} \left\| \left( \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \right)^{-1} \right\| \sum_{j=1}^k \| \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j)} \bar{\mathcal{W}}_t^{(j)} \bar{\mathcal{W}}_t^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j)} \|_F \\ &= \frac{\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} \|}{\sigma_{\min}(\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t})} \sum_{j=1}^k \| \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j)} \bar{\mathcal{V}}_{\mathcal{U}_t * \mathcal{W}_t}^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j)} \bar{\mathcal{W}}_t^{(j)} \bar{\mathcal{W}}_t^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j)} \|_F \\ &= \frac{\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} \|}{\sigma_{\min}(\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t})} \sum_{j=1}^k \| \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j)} \bar{\mathcal{W}}_t^{(j)} \bar{\mathcal{W}}_t^{(j) \text{H}} \bar{\mathcal{U}}_t^{(j) \text{H}} \bar{\mathcal{V}}_{\mathcal{X}^\perp}^{(j)} \|_F \\ &= \frac{\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} \|}{\sigma_{\min}(\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t})} \| \mathcal{V}_{\mathcal{X}^\perp} * \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top * \mathcal{V}_{\mathcal{X}^\perp} \|_F \\ &= \frac{\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} \|}{\sigma_{\min}(\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t})} \| \mathcal{V}_{\mathcal{X}^\perp} * \mathcal{U}_t * \mathcal{U}_t^\top * \mathcal{V}_{\mathcal{X}^\perp} \|_F \\ &= \frac{\| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t * \mathcal{W}_t} \|}{\sigma_{\min}(\mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_$$

where in the last line we used the assumption [\(E.35\)](#page-38-2). Them, using just established bound together with [\(E.39\)](#page-38-3), we get

$$\|\mathbf{v}_{\mathcal{X}^\perp} * \mathbf{u}_t * \mathbf{u}_t^\top\|_F \leq 3 \|\mathbf{v}_{\mathcal{X}^\perp} * (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\|_F + \|\mathbf{u}_t * \mathbf{w}_{t,\perp} * \mathbf{w}_{t,\perp}^\top * \mathbf{u}_t^\top\|_F.$$

To get inequality [\(E.37\)](#page-38-4), we use the orthogonal decomposition of X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ <sup>t</sup> with respect to V<sup>X</sup> and V<sup>X</sup> <sup>⊥</sup> , which leads to

$$\begin{aligned}\|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\|_F &= \|\boldsymbol{\nu}_{\boldsymbol{x}}^\top * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|_F + \|\boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|_F \\ &= \|\boldsymbol{\nu}_{\boldsymbol{x}}^\top * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|_F + \|\boldsymbol{\nu}_{\boldsymbol{x}^\perp}^\top * \boldsymbol{u}_t * \boldsymbol{u}_t^\top\|_F \\ &\leq 4\|\boldsymbol{\nu}_{\boldsymbol{x}}^\top * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|_F + \|\boldsymbol{u}_t * \boldsymbol{w}_{t,\perp} * \boldsymbol{w}_{t,\perp}^\top * \boldsymbol{u}_t^\top\|_F.\end{aligned}$$

Inequality [\(E.38\)](#page-38-1) follows from the two inequalities proved here and Lemma 9.5 in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0). The ¨ building stones for this are the properties of the tubal tensor Frobenius norm. Namely, the Frobenius norm of any tubal tensor T can be represented as the sum of Frobenius norms of each slice in the domain, that is

$$\|\mathcal{T}\|_F = \sum_{j=1}^k \|\bar{T}^{(j)}\|_F$$

and ∥T ∥<sup>F</sup> ≤ √ n · k∥T ∥. Besides, the Frobenius norm of the product of two tensors T and P can be bounded as below

$$\|\mathcal{T} * \mathcal{P}\|_F = \sum_{j=1}^k \|\bar{T}^{(j)} \bar{P}^{(j)}\|_F \leq \max_{1 \leq j \leq k} \|\bar{T}^{(j)}\| \sum_{j=1}^k \|\bar{P}^{(j)}\|_F \leq \|\mathcal{T}\| \|\mathcal{P}\|_F.$$

Now, we have collected all the necessary ingredients to prove the main result of this section, which shows that after a sufficient number of interactions, the relative error between U<sup>t</sup> ∗ U ⊤ t and X ∗ X <sup>⊤</sup> becomes small.

Theorem E.1. *Suppose that the stepsize satisfies* µ ≤ c<sup>1</sup> √ kκ−<sup>4</sup>∥X ∥ −2 *for some small* c<sup>1</sup> > 0*, and* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> m *satisfies RIP*(2r + 1, δ) *for some constant* 0 < δ ≤ c1 κ 4 √ r *. Set* γ ∈ (0, 2 )*, and choose a number of iterations* t<sup>∗</sup>

*such that* σ*min*(U<sup>t</sup><sup>∗</sup> ∗ W<sup>t</sup><sup>∗</sup> ) ≥ γ*. Also, assume that* ∥U<sup>t</sup><sup>∗</sup> ∗ W<sup>t</sup>∗,<sup>⊥</sup>∥ ≤ 2γ*,* ∥U<sup>t</sup><sup>∗</sup> ∥ ≤ 3∥X ∥*,* γ ≤ c2σ*min*(X ) κ <sup>2</sup> min{n, R} *, and* ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup><sup>t</sup><sup>∗</sup> <sup>∗</sup>W<sup>t</sup><sup>∗</sup> ∥ ≤ c2κ −2 *for some small* c<sup>2</sup> > 0*. Then, after*

$$\hat{t} - t_* \lesssim \frac{1}{\mu\sigma_{\min}(\boldsymbol{X})^2} \ln \left( \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{\|\boldsymbol{X}\|}{\gamma} \right)$$

*additional iterations, we have*

$$\frac{\|\mathcal{U}_{\hat{t}} * \mathcal{U}_{\hat{t}}^\top - \mathcal{X} * \mathcal{X}^\top\|_F}{\|\mathcal{X}\|^2} \lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\mathcal{X}\|^{-21/16}.$$

*Proof.* First, we set

$$t_1 = \min \left\{ t \geq t_* : \sigma_{\min}(\mathbf{v}_{\boldsymbol{X}}^\top * \boldsymbol{u}_t) \geq \frac{1}{\sqrt{10}} \sigma_{\min}(\overline{\boldsymbol{X}}) \right\},$$

and then aim to prove that over the iterations t<sup>∗</sup> ≤ t ≤ t1, the following hold:

- σmin(V ⊤ <sup>X</sup> ∗ Ut) ≥ 1 2 γ 1 + <sup>1</sup> 8 µσmin(X ) 2 <sup>t</sup>−t<sup>∗</sup>
- ∥U<sup>t</sup> ∗ Wt,<sup>⊥</sup>∥ ≤ 2γ 1 + 80µc<sup>2</sup> √ kσmin(X ) 2 <sup>t</sup>−t<sup>∗</sup>
- ∥Ut∥ ≤ 3∥X ∥
- ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ c2κ −2 .

Intuitively, this means that over the range t<sup>∗</sup> ≤ t ≤ t1, the smallest singular value of the signal term V ⊤ <sup>X</sup> ∗ U<sup>t</sup> grows at a faster rate than the largest singular value of the noise term U<sup>t</sup> ∗ Wt,⊥.

For t = t∗, these inequalities hold due to the assumptions of this theorem. Now, suppose they hold for some t between t<sup>∗</sup> and t1. We'll show they also hold for t + 1.

First, note that we have:

$$\begin{aligned}
& \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \\
&= \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top - \mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top * \mathcal{U}_t^\top)\| \\
&\leq \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top)\| + \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp} * \mathcal{U}_t^\top)\| \\
(a) \quad &\leq \delta\sqrt{kr}\|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top\| + \delta\sqrt{k}\|\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp} * \mathcal{U}_t^\top\|_* \\
&\leq \delta\sqrt{kr} \left( \|\mathcal{X} * \mathcal{X}^\top\| + \|\mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top\| \right) + \delta\sqrt{k}\|\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp} * \mathcal{U}_t^\top\|_* \\
&= \delta\sqrt{kr} \left( \|\mathcal{X}\|^2 + \|\mathcal{U}_t * \mathcal{W}_t\|^2 \right) + \delta\sqrt{k}\|\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp} * \mathcal{U}_t^\top\|_* \\
&\leq \delta\sqrt{kr} \left( \|\mathcal{X}\|^2 + \|\mathcal{U}_t\|^2 \right) + \delta\sqrt{k}\|\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp} * \mathcal{U}_t^\top\|_* \\
(b) \quad &\leq \delta\sqrt{kr} \left( \|\mathcal{X}\|^2 + 9\|\mathcal{X}\|^2 \right) + \delta\sqrt{k}(\min\{n, R\} - r)\|\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp} * \mathcal{U}_t^\top\| \\
&\leq 10\delta\sqrt{kr}\|\mathcal{X}\|^2 + \delta\sqrt{k}(\min\{n, R\} - r)\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\|^2 \\
&\leq 10\delta\sqrt{kr}\kappa^2\sigma_{\min}(\mathcal{X})^2 + \delta\sqrt{k}(\min\{n, R\} - r)\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\|^2 \\
(c) \quad &\leq 10c_1\sqrt{k\kappa}^{-2}\sigma_{\min}(\mathcal{X})^2 + 4\delta\sqrt{k}(\min\{n, R\} - r)\gamma^2 \left( 1 + 80\mu c_2\sigma_{\min}(\mathcal{X})^2 \right)^{2(t-t_*)} \\
(d) \quad &\leq 10c_1\sqrt{k\kappa}^{-2}\sigma_{\min}(\mathcal{X})^2 + 8\delta\sqrt{k}(\min\{n, R\} - r)\gamma^{7/4}\sigma_{\min}(\mathcal{X})^{1/4} \\
(e) \quad &\leq 40c_1\sqrt{k\kappa}^{-2}\sigma_{\min}(\mathcal{X})^2.
\end{aligned}$$

In inequality (a), we used the fact that A satisfies RIP(2r + 1, δ) (and hence, RIP(r + 1, δ) and RIP(2, δ)), and thus, by Lemmas [G.2](#page-50-1) and [G.3,](#page-50-0) also satisfies S2SRIP(r, δ√ kr) and S2NRIP(δ √ k). Inequality (b) uses the assumption ∥Ut∥ ≤ 3∥X ∥ and the fact that U<sup>t</sup> ∗ Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> ∗ U ⊤ <sup>t</sup> has tubal rank at most min{n, R} − r. In inequality (c), we used the assumption δ ≤ c1 κ 4 √ r along with the second bulleted inequality assumed by the inductive step. Inequality (d) holds due to the definitions of t<sup>1</sup> and t<sup>∗</sup> and the fact that t<sup>∗</sup> ≤ t ≤ t1. Finally, inequality (e) holds due to the assumption γ ≤ c2σmin(X) .

κ<sup>2</sup> min{n,R} If c<sup>1</sup> is chosen small enough, the above bound is less than ∥X ∥. Then, along with our other assumptions, we can use Lemma [E.6](#page-37-0) to obtain ∥Ut+1∥ ≤ 3∥X ∥.

Next, we can use Lemma [E.1](#page-27-1) along with the bound σmin(V ⊤ <sup>X</sup> ∗ Ut) ≤ √ <sup>10</sup>σmin(X ) to obtain

$$\begin{aligned}\sigma_{\min}(\mathbf{v}_{\mathcal{X}}^\top * \mathbf{u}_{t+1}) &\geq \sigma_{\min}(\mathbf{v}_{\mathcal{X}}^\top * \mathbf{u}_{t+1} * \mathbf{w}_{t+1}) \\ &\geq \sigma_{\min}(\mathbf{v}_{\mathcal{X}}^\top * \mathbf{u}_t) \left( 1 + \frac{1}{4} \mu \sigma_{\min}(\mathcal{X})^2 - \mu \sigma_{\min}(\mathbf{v}_{\mathcal{X}}^\top * \mathbf{u}_t)^2 \right) \\ &\geq \sigma_{\min}(\mathbf{v}_{\mathcal{X}}^\top * \mathbf{u}_t) \left( 1 + \frac{1}{4} \mu \sigma_{\min}(\mathcal{X})^2 - \frac{1}{10} \mu \sigma_{\min}(\mathcal{X})^2 \right) \\ &\geq \sigma_{\min}(\mathbf{v}_{\mathcal{X}}^\top * \mathbf{u}_t) \left( 1 + \frac{1}{8} \mu \sigma_{\min}(\mathcal{X})^2 \right) \\ &\geq \frac{1}{2} \gamma \left( 1 + \frac{1}{8} \mu \sigma_{\min}(\mathcal{X})^2 \right)^{t-t_*} \cdot \left( 1 + \frac{1}{8} \mu \sigma_{\min}(\mathcal{X})^2 \right) \\ &= \frac{1}{2} \gamma \left( 1 + \frac{1}{8} \mu \sigma_{\min}(\mathcal{X})^2 \right)^{t-t_*+1}\end{aligned}$$

Since σmin(V ⊤ <sup>X</sup> ∗ Ut+1 ∗ Wt+1) = σmin(V ⊤ <sup>X</sup> ∗ Ut+1), which is positive by the above bound, all the singular tubes of V ⊤ <sup>X</sup> ∗ Ut+1 ∗ Wt+1 are invertible. Hence, we can apply Lemma [E.3](#page-30-0) to obtain

$$\begin{aligned}
\|\overline{\boldsymbol{\mathcal{U}}_{t+1}} * \overline{\boldsymbol{\mathcal{W}}_{t+1,\perp}}^{(j)}\| &\leq \left(1 - \frac{\mu}{2} \|\overline{\boldsymbol{\mathcal{U}}_t} * \overline{\boldsymbol{\mathcal{W}}_{t,\perp}}^{(j)}\|^2 + 9\mu \|\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}},\perp}^\top * \boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{U}}_t} * \boldsymbol{\mathcal{W}_t}^{(j)}\| \|\boldsymbol{\mathcal{X}}\|^2 \right. \\
&\quad \left. + 2\mu \|(\boldsymbol{\mathcal{A}}^* \boldsymbol{\mathcal{A}} - \boldsymbol{\mathcal{I}})(\boldsymbol{\mathcal{X}} * \boldsymbol{\mathcal{X}}^\top - \boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{U}}_t^\top)\| \right) \|\overline{\boldsymbol{\mathcal{U}}_t} * \overline{\boldsymbol{\mathcal{W}}_{t,\perp}}^{(j)}\| \\
&\leq \left(1 - \frac{\mu}{2} \cdot 4\gamma^2 \left(1 + 80\mu c_2 \sqrt{k} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{2(t-t_*)} + 9\mu c_2 \kappa^{-2} \|\boldsymbol{\mathcal{X}}\|^2 \right. \\
&\quad \left. + 2\mu \cdot 40c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2 \right) \|\overline{\boldsymbol{\mathcal{U}}_t} * \overline{\boldsymbol{\mathcal{W}}_{t,\perp}}^{(j)}\| \\
&\leq \left(1 - \frac{\mu}{2} \cdot 4\gamma^2 \left(1 + 80\mu c_2 \sqrt{k} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{2(t-t_*)} + 9\mu c_2 \sigma_{\min}(\boldsymbol{\mathcal{X}})^2 \right. \\
&\quad \left. + 80c_1 \mu \sqrt{k} \kappa^{-2} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2 \right) \|\overline{\boldsymbol{\mathcal{U}}_t} * \overline{\boldsymbol{\mathcal{W}}_{t,\perp}}^{(j)}\| \\
&\leq \left(1 + 80c_1 \mu \sqrt{k} \kappa^{-2} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right) \|\overline{\boldsymbol{\mathcal{U}}_t} * \overline{\boldsymbol{\mathcal{W}}_{t,\perp}}^{(j)}\| \\
&\leq \left(1 + 80c_1 \mu \sqrt{k} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right) \|\overline{\boldsymbol{\mathcal{U}}_t} * \overline{\boldsymbol{\mathcal{W}}_{t,\perp}}^{(j)}\| \\
&\leq 2\gamma \left(1 + 80c_1 \mu \sqrt{k} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{t-t_*+1},
\end{aligned}$$

where we have used the inductive assumption that the inequalities hold for t along with the fact that κ = ∥X ∥/σmin(X ) ≥ 1. Next, we will bound the term using Lemma [E.5](#page-34-0)

$$\begin{aligned} & \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_{t+1}*\mathcal{W}_{t+1}} \| \\ & \leq \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) \| \mathcal{V}_{\mathcal{X}^\perp}^\top * \mathcal{V}_{\mathcal{U}_t*\mathcal{W}_t} \| + 150\mu \| (\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top) \| + 500\mu^2 \| \mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top \|^2 \\ & \leq \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 150\mu \cdot 40c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 + 500\mu^2 \cdot (\|\mathcal{X}\|^2 + \|\mathcal{U}_t\|^2) \\ & \leq \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 6000\mu c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 + 500\mu^2 \cdot (\|\mathcal{X}\|^2 + 9\|\mathcal{X}\|^2)^2 \\ & = \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 6000\mu c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 + 50000\mu^2 \|\mathcal{X}\|^4 \\ & \leq \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 6000\mu c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 + 50000\mu \cdot c_1 \kappa^{-4} \|\mathcal{X}\|^{-2} \cdot \|\mathcal{X}\|^4 \\ & = \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 6000\mu c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 + 50000\mu \cdot c_1 \kappa^{-4} \|\mathcal{X}\|^2 \\ & = \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 6000\mu c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 + 50000\mu \cdot c_1 \kappa^{-4} \kappa^2 \sigma_{\min}(\mathcal{X})^2 \\ & = \left( 1 - \frac{\mu}{4} \sigma_{\min}^2(\mathcal{X}) \right) c_2 \kappa^{-2} + 56000\mu c_1 \sqrt{k} \kappa^{-2} \sigma_{\min}(\mathcal{X})^2 \end{aligned}$$

Here, we have again used the inductive assumptions along with the fact that κ = ∥X ∥/σmin(X ). If we choose c<sup>1</sup> sufficiently small, we will have ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t+1∗Wt+1 ∥ ≤ c2κ −2 .

Therefore, the four bullet points hold for t + 1, and thus, the induction is complete.

With the above bullet points in mind, we note that

$$\frac{1}{\sqrt{10}}\sigma_{\min}(\boldsymbol{x}) \geq \sigma_{\min}(\boldsymbol{\nu}_{\boldsymbol{x}}^{\top} * \boldsymbol{u}_{t_1}) \geq \frac{1}{2}\gamma \left(1 + \frac{1}{8}\mu\sigma_{\min}(\boldsymbol{x})^2\right)^{t_1-t_*},$$

and so,

$$t_1 - t_* \leq \frac{\log\left(\frac{2}{\gamma\sqrt{10}}\sigma_{\min}(\boldsymbol{\mathcal{X}})\right)}{\log\left(1 + \frac{1}{8}\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)} \leq \frac{16}{\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2} \log\left(\frac{2}{\gamma\sqrt{10}}\sigma_{\min}(\boldsymbol{\mathcal{X}})\right),$$

where we have used the inequality <sup>1</sup> log(1+x) ≤ 2 x for 0 < x < 1. Furthermore, we can bound the norm of the signal term at iteration t<sup>1</sup> by

$$\begin{aligned} \|\mathcal{U}_{t1} * \mathcal{W}_{t1,\perp}\| &\leq 2\gamma \left(1 + 80\mu c_2 \sqrt{k} \sigma_{\min}(\mathcal{X})^2\right)^{t_1-t_2} \\ &\leq 2\gamma \left(\frac{2}{\sqrt{10}} \cdot \frac{\sigma_{\min}(\mathcal{X})}{\gamma}\right)^{1280c_2} \\ &\leq 2\gamma \left(\frac{2}{\sqrt{10}} \cdot \frac{\sigma_{\min}(\mathcal{X})}{\gamma}\right)^{1/64} \\ &\leq 3\gamma^{63/64} \sigma_{\min}(\mathcal{X})^{1/64} \\ &\leq 3\gamma^{7/8} \sigma_{\min}(\mathcal{X})^{1/8}, \end{aligned}$$

where we have used the previous bound on t<sup>1</sup> − t∗, the fact that c<sup>2</sup> > 0 can be chosen to be sufficiently small, and the fact that σmin(X ) ≥ γ.

Next, we set

$$t_2 = t_1 + \left\lfloor \frac{300}{\mu\sigma_{\min}(\boldsymbol{X})^2} \ln \left( \frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\boldsymbol{X}\|^{7/4}}{\gamma^{7/4}} \right) \right\rfloor$$

$$t_3 = \min \left\{ t \geq t_1 : \left( \sqrt{k(\min\{n, R\} - r)} + 1 \right) \left\| \boldsymbol{U}_t * \boldsymbol{W}_{t,\perp} * \boldsymbol{W}_{t,\perp}^\top * \boldsymbol{U}_t^\top \right\|_F \geq \|\boldsymbol{X} * \boldsymbol{X}^\top - \boldsymbol{U}_t * \boldsymbol{U}_t^\top\|_F \right\}$$

$$\hat{t} = \min\{t_2, t_3\}.$$

We now aim to show that over the range <sup>t</sup><sup>1</sup> ≤ <sup>t</sup> ≤ b<sup>t</sup>, the following inequalities hold:

- σmin(U<sup>t</sup> ∗ Wt) ≥ σmin(V ⊤ <sup>X</sup> ∗ Ut) ≥ 1 √ 10 σmin(X )
- <sup>∥</sup>U<sup>t</sup> <sup>∗</sup> <sup>W</sup>t,<sup>⊥</sup>∥ ≤ 1 + 80µc<sup>2</sup> √ kσmin(X ) 2 <sup>t</sup>−t<sup>1</sup> ∥U<sup>t</sup><sup>1</sup> ∗ W<sup>t</sup>1,<sup>⊥</sup>∥
- ∥Ut∥ ≤ 3∥X ∥
- ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ c2κ −2
- ∥V ⊤ <sup>X</sup> ∗ (X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t )∥<sup>F</sup> <sup>≤</sup> <sup>10</sup>√ kr 1 − <sup>400</sup>µσmin(X ) 2 <sup>t</sup>−t<sup>1</sup> ∥X ∥ 2

For t = t1, the first four bullet points follow from what we previously proved via induction. The last one holds since we trivially have

$$\begin{aligned} \|\mathbf{v}_{\mathcal{X}}^\top * (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_{t_1} * \mathbf{u}_{t_1}^\top)\|_F &\leq \sqrt{kr} \|\mathbf{v}_{\mathcal{X}}^\top * (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_{t_1} * \mathbf{u}_{t_1}^\top)\| \\ &\leq \sqrt{kr} \|\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_{t_1} * \mathbf{u}_{t_1}^\top\| \\ &\leq \sqrt{kr} \|\mathcal{X} * \mathcal{X}^\top\| + \sqrt{kr} \|\mathbf{u}_{t_1} * \mathbf{u}_{t_1}^\top\| \\ &\leq \sqrt{kr} \|\mathcal{X}\|^2 + \sqrt{kr} \|\mathbf{u}_{t_1}\|^2 \\ &\leq 10\sqrt{kr} \|\mathcal{X}\|^2. \end{aligned}$$

Now suppose all the bullet points hold for some integer <sup>t</sup> ∈ [t1, b<sup>t</sup> − 1]. Again, we aim to show they all hold for <sup>t</sup> + 1. In a similar manner as done before, we can bound ∥(A<sup>∗</sup>A − I)(X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ U ⊤ t )∥ ≤ 10δ √ kr∥X ∥ <sup>2</sup> + δ √ k(min{n, R} − r)∥U<sup>t</sup> ∗ Wt,<sup>⊥</sup>∥ 2 , and then continue as follows

$$\begin{aligned} & \|(\mathcal{A}^*\mathcal{A} - \mathcal{I})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top)\| \\ & \leq 10\delta\sqrt{kr}\|\mathcal{X}\|^2 + \delta\sqrt{k}(\min\{n, R\} - r)\|\mathcal{U}_t * \mathcal{W}_{t,\perp}\|^2 \\ & \leq 10 \cdot \frac{c_1}{\kappa^4\sqrt{r}} \cdot \sqrt{kr} \cdot \kappa^2 \sigma_{\min}(\mathcal{X})^2 + \delta\sqrt{k}(\min\{n, R\} - r) \left(1 + 80\mu c_2\sqrt{k}\sigma_{\min}(\mathcal{X})^2\right)^{2(t-t_1)} \|\mathcal{U}_{t_1} * \mathcal{W}_{t_1,\perp}\|^2 \\ & \leq 10c_1\sqrt{k}\kappa^{-2}\sigma_{\min}(\mathcal{X})^2 + \delta\sqrt{k}(\min\{n, R\} - r) \left(1 + 80\mu c_2\sqrt{k}\sigma_{\min}(\mathcal{X})^2\right)^{2(t-t_1)} \cdot 9\gamma^{7/4}\sigma_{\min}(\mathcal{X})^{1/4} \\ & \leq 10c_1\sqrt{k}\kappa^{-2}\sigma_{\min}(\mathcal{X})^2 + 9\delta\sqrt{k}(\min\{n, R\} - r) \left(1 + 80\mu c_2\sqrt{k}\sigma_{\min}(\mathcal{X})^2\right)^{2(t_2-t_1)} \gamma^{7/4}\sigma_{\min}(\mathcal{X})^{1/4} \\ & \leq 10c_1\sqrt{k}\kappa^{-2}\sigma_{\min}(\mathcal{X})^2 + 9\delta\sqrt{k}(\min\{n, R\} - r) \left(\frac{5}{18}\kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\mathcal{X}\|^{7/4}}{\gamma^{7/4}}\right)^{O(c_2)} \gamma^{7/4}\sigma_{\min}(\mathcal{X})^{1/4} \\ & \leq 40c_1\sqrt{k}\kappa^{-2}\sigma_{\min}(\mathcal{X})^2 \end{aligned}$$

where we have used the bounds δ ≤ c<sup>1</sup> κ<sup>4</sup>√ r , ∥X ∥ = κσmin(X ), ∥U<sup>t</sup><sup>1</sup> ∗ W<sup>t</sup>1,<sup>⊥</sup>∥ ≤ 3γ <sup>7</sup>/<sup>8</sup>σmin(X ) 1/8 , along with the inductive assumptions and the definition of t1.

Next, we note that if σmin(V ⊤ <sup>X</sup> ∗ Ut) ≤ 2 σmin(X ), then we can use Lemma [E.1](#page-27-1) along with the inductive assumptions to obtain

$$\begin{aligned} \sigma_{\min}(\boldsymbol{\mathcal{U}}_{t+1} * \boldsymbol{\mathcal{W}}_{t+1}) &\geq \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}}}^\top * \boldsymbol{\mathcal{U}}_{t+1}) \\ &\geq \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}}}^\top * \boldsymbol{\mathcal{U}}_{t+1} * \boldsymbol{\mathcal{W}}_t) \\ &\geq \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}}}^\top * \boldsymbol{\mathcal{U}}_t) \left( 1 + \frac{1}{4} \mu \sigma_{\min}(\boldsymbol{\mathcal{X}})^2 - \mu \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}}}^\top * \boldsymbol{\mathcal{U}}_t)^2 \right) \\ &\geq \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}}}^\top * \boldsymbol{\mathcal{U}}_t) \left( 1 + \frac{1}{4} \mu \sigma_{\min}(\boldsymbol{\mathcal{X}})^2 - \mu \cdot \frac{1}{4} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2 \right) \\ &= \sigma_{\min}(\boldsymbol{\mathcal{V}}_{\boldsymbol{\mathcal{X}}}^\top * \boldsymbol{\mathcal{U}}_t) \\ &\geq \frac{1}{\sqrt{10}} \sigma_{\min}(\boldsymbol{\mathcal{X}}) \end{aligned}$$

Alternatively, if σmin(V ⊤ <sup>X</sup> ∗ Ut) ≥ 1 2 σmin(X ), then we can again use Lemma [E.1](#page-27-1) along with the inductive assumptions and the fact that µ ≤ c1κ <sup>−</sup><sup>2</sup>∥X ∥ 2 for sufficiently small c<sup>1</sup> to obtain

$$\begin{aligned}\sigma_{\min}(\mathcal{U}_{t+1} * \mathcal{W}_{t+1}) &\geq \sigma_{\min}(\mathcal{V}_{\mathcal{X}}^\top * \mathcal{U}_{t+1}) \\ &\geq \sigma_{\min}(\mathcal{V}_{\mathcal{X}}^\top * \mathcal{U}_{t+1} * \mathcal{W}_t) \\ &\geq \sigma_{\min}(\mathcal{V}_{\mathcal{X}}^\top * \mathcal{U}_t) \left( 1 + \frac{1}{4} \mu \sigma_{\min}(\mathcal{X})^2 - \mu \sigma_{\min}(\mathcal{V}_{\mathcal{X}}^\top * \mathcal{U}_t)^2 \right) \\ &\geq \frac{1}{2} \sigma_{\min}(\mathcal{X}) (1 - \mu \sigma_{\min}(\mathcal{U}_t)^2) \\ &\geq \frac{1}{2} \sigma_{\min}(\mathcal{X}) (1 - \mu \|\mathcal{U}_t\|^2) \\ &\geq \frac{1}{2} \sigma_{\min}(\mathcal{X}) (1 - 9\mu \|\mathcal{X}\|^2) \\ &\geq \frac{1}{2} \sigma_{\min}(\mathcal{X}) (1 - 9c_1\kappa^{-2}) \\ &\geq \frac{1}{\sqrt{10}} \sigma_{\min}(\mathcal{X})\end{aligned}$$

Again, since σmin(V ⊤ <sup>X</sup> ∗ Ut+1 ∗ Wt) ≥ √ 1 <sup>10</sup>σmin(X ) <sup>&</sup>gt; <sup>0</sup>, we have that V ⊤ <sup>X</sup> ∗ Ut+1 ∗ W<sup>t</sup> has full tubal rank with all invertible t-SVD singular tubes. Hence, by Lemma [E.3,](#page-30-0) we again can bound

$$\| \mathcal{U}_{t+1} * \mathcal{W}_{t+1,\perp} \| \leq \left(1 + 80 \mu c_2 \sqrt{k} \sigma_{\min}(\mathcal{X})^2\right)^{t+1-t_1} \| \mathcal{U}_{t_1} * \mathcal{W}_{t_1,\perp} \|.$$

In the exact same way as before, we can use Lemma [E.6](#page-37-0) to establish ∥Ut+1∥ ≤ 3∥X ∥, and use Lemma [E.7](#page-38-5) to establish ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t+1∗Wt+1 ∥ ≤ c2κ −2 .

To bound ∥V ⊤ <sup>X</sup> ∗ (X ∗ X <sup>⊤</sup> − Ut+1 ∗ U ⊤ <sup>t</sup>+1)∥<sup>F</sup> , we will aim to use Lemma [E.7.](#page-38-5) By the inductive assumptions, we already have ∥Ut∥ ≤ <sup>3</sup>∥X ∥, <sup>σ</sup>min(U<sup>t</sup> ∗ Wt) ≥ √ <sup>10</sup>σmin(X ), and ∥V ⊤ <sup>X</sup> <sup>⊥</sup> ∗ V<sup>U</sup>t∗W<sup>t</sup> ∥ ≤ c2κ −2 . To derive the remaining condition of Lemma [E.7,](#page-38-5) we first split

$$\begin{aligned}
& \| \mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U} * \mathcal{U}^\top) \|_F \\
&= \| \mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t \mathcal{W}_t^\top * \mathcal{U}_t^\top - \mathcal{U}_t * \mathcal{W}_{t,\perp} \mathcal{W}_{t,\perp}^\top * \mathcal{U}_t^\top) \|_F \\
&\leq \| \mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top) \|_F + \| \mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top * \mathcal{U}_t^\top) \|_F.
\end{aligned}$$

To bound the first term, we note that X ∗ X <sup>⊤</sup> − U<sup>t</sup> ∗ W<sup>t</sup> ∗ W<sup>⊤</sup> <sup>t</sup> ∗ U ⊤ t is tubal-symmetric with tubal rank at most 2r, so we can write it as the sum of two tubal-symmetric tensors Z1, Z<sup>2</sup> ∈ S <sup>n</sup>×n×<sup>k</sup> with tubal rank at most r, and then apply Lemma [G.4](#page-51-3) to obtain

$$\begin{aligned} \|\mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top)\|_F &= \|\mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}_1 + \mathcal{Z}_2)\|_F \\ &\leq \|\mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}_1)\|_F + \|\mathcal{V}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}_2)\|_F \\ &\leq \delta(\|\mathcal{Z}_1\|_F + \|\mathcal{Z}_2\|_F) \\ &\leq \delta\sqrt{2}\|\mathcal{Z}_1 + \mathcal{Z}_2\|_F \\ &= \delta\sqrt{2}\|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top\|_F \\ &\leq \delta\sqrt{2}\|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\|_F \end{aligned}$$

For the second piece, we use the symmetric t-SVD to write U<sup>t</sup> ∗ Wt,<sup>⊥</sup> ∗ W<sup>⊤</sup> t,<sup>⊥</sup> ∗ U ⊤ <sup>t</sup> = P <sup>i</sup> V<sup>i</sup> ∗ s<sup>i</sup> ∗ V ⊤ i . Then, we can bound

$$\begin{aligned}
\|\boldsymbol{\nu}_{\boldsymbol{\mathcal{X}}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp} * \boldsymbol{\mathcal{W}}_{t,\perp}^\top * \boldsymbol{\mathcal{U}}_t^\top)\|_F &= \left\| \boldsymbol{\nu}_{\boldsymbol{\mathcal{X}}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A}) \left( \sum_i \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right) \right\|_F \\
&\leq \sum_i \left\| \boldsymbol{\nu}_{\boldsymbol{\mathcal{X}}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A}) \left( \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right) \right\|_F \\
&\leq \sum_i \delta \left\| \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right\|_F \\
&= \sum_i \delta \|\boldsymbol{s}_i\|_2 \\
&= \delta \left\| \boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp} * \boldsymbol{\mathcal{W}}_{t,\perp}^\top * \boldsymbol{\mathcal{U}}_t^\top \right\|_* \\
&\leq \delta \sqrt{k(\min\{n, R\} - r)} \left\| \boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp} * \boldsymbol{\mathcal{W}}_{t,\perp}^\top * \boldsymbol{\mathcal{U}}_t^\top \right\|_F \\
&\leq \|\boldsymbol{\mathcal{X}} * \boldsymbol{\mathcal{X}}^\top - \boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{U}}_t^\top\|_F,
\end{aligned}$$

Hence,

$$\begin{aligned}
& \|\mathbf{v}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U} * \mathcal{U}^\top)\|_F \\
& \leq \|\mathbf{v}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{W}_t * \mathcal{W}_t^\top * \mathcal{U}_t^\top)\|_F + \|\mathbf{v}_{\mathcal{X}}^\top * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{U}_t * \mathcal{W}_{t,\perp} * \mathcal{W}_{t,\perp}^\top * \mathcal{U}_t^\top)\|_F \\
& \leq \delta \sqrt{2} \|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\|_F + \delta \|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\|_F \\
& \leq c\kappa^{-2} \|\mathcal{X} * \mathcal{X}^\top - \mathcal{U}_t * \mathcal{U}_t^\top\|_F,
\end{aligned}$$

where we have used the assumption that δ ≤ c<sup>1</sup> κ<sup>4</sup>√ <sup>r</sup> <sup>≤</sup> cκ−<sup>2</sup> .

Similarly, we can bound

$$\| \mathbf{v}_{\mathcal{U}_t * \mathbf{w}_t} * (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t) \|_F \leq c\kappa^{-2} \| \mathbf{x} * \mathbf{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top \|_F,$$

and

$$\|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t)\| \leq c\kappa^{-2} \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top\|_F.$$

Then, by Lemma [E.7,](#page-38-5) we have

$$\|\mathbf{v}_{\mathcal{X}^\perp}(\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_{t+1} * \mathbf{u}_{t+1}^\top)\|_F \leq \left(1 - \frac{\mu}{200}\sigma_{\min}^2(\mathcal{X})\right) \|\mathbf{v}_{\mathcal{X}^\perp}^\top * (\mathcal{X} * \mathcal{X}^\top - \mathbf{u}_t * \mathbf{u}_t^\top)\|_F + \mu \frac{\sigma_{\min}^2(\mathcal{X})}{100} \|\mathbf{u}_t * \mathbf{w}_{t,\perp} * \mathbf{w}_{t,\perp}^\top * \mathbf{u}_t^\top\|_F$$

By the inductive assumption,

$$\| \mathbf{v}_{\boldsymbol{x}\perp}^\top * (\boldsymbol{x} * \boldsymbol{x}^\top - \mathbf{u}_t * \mathbf{u}_t^\top) \|_F \leq 10\sqrt{kr} \left(1 - \frac{1}{400} \mu \sigma_{\min}(\boldsymbol{x})^2\right)^{t-t_1} \| \boldsymbol{x} \|^2.$$

Also, using the inductive assumption and the bound from the previous part, we can bound

$$\begin{aligned} \|\boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp} * \boldsymbol{\mathcal{W}}_{t,\perp}^\top * \boldsymbol{\mathcal{U}}_t^\top\|_F &\leq \sqrt{k(\min\{n, R\} - r)} \|\boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp} * \boldsymbol{\mathcal{W}}_{t,\perp}^\top * \boldsymbol{\mathcal{U}}_t^\top\| \\ &\leq \sqrt{k(\min\{n, R\} - r)} \|\boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp}\|^2 \\ &\leq \sqrt{k(\min\{n, R\} - r)} \left(1 + 80\mu_{C2}\sqrt{k}\sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{2(t-t_1)} \|\boldsymbol{\mathcal{U}}_{t_1} * \boldsymbol{\mathcal{W}}_{t_1,\perp}\|^2 \\ &\leq \sqrt{k(\min\{n, R\} - r)} \left(1 + 80\mu_{C2}\sqrt{k}\sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{2(t-t_1)} \cdot 9\gamma^{7/4}\sigma_{\min}(\boldsymbol{\mathcal{X}})^{1/4} \end{aligned}$$

Since t ≤ t2, we have

$$t - t_1 \leq t_2 - t_1 \leq \frac{300}{\mu\sqrt{k}\sigma_{\min}(\mathcal{X})^2} \ln \left( \frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{\min\{n, R\} - r}} \frac{\|\mathcal{X}\|^{7/4}}{\gamma^{7/4}} \right),$$

and thus,

$$\begin{aligned} \|\boldsymbol{\mathcal{U}}_t * \boldsymbol{\mathcal{W}}_{t,\perp} * \boldsymbol{\mathcal{W}}_{t,\perp}^\top * \boldsymbol{\mathcal{U}}_t^\top\|_F &\leq \sqrt{k(\min\{n, R\} - r)} \left(1 + 80\mu c_2 \sqrt{k} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{2(t-t_1)} \cdot 9\gamma^{7/4} \sigma_{\min}(\boldsymbol{\mathcal{X}})^{1/4} \\ &\leq \frac{5}{2} \sqrt{kr} \left(1 - \frac{\mu}{400} \sigma_{\min}(\boldsymbol{\mathcal{X}})^2\right)^{t-t_1} \|\boldsymbol{\mathcal{X}}\|^2. \end{aligned}$$

Combining these inequalities yields

$$\begin{aligned} \|\mathbf{v}_{\boldsymbol{X}^\perp}(\boldsymbol{X} * \boldsymbol{X}^\top - \boldsymbol{u}_{t+1} * \boldsymbol{u}_{t+1}^\top)\|_F &\leq \left(1 - \frac{\mu}{200} \sigma_{\min}^2(\boldsymbol{X})\right) \|\mathbf{v}_{\boldsymbol{X}^\perp} * (\boldsymbol{X} * \boldsymbol{X}^\top - \boldsymbol{u}_t * \boldsymbol{u}_t^\top)\|_F \\ &\quad + \mu \frac{\sigma_{\min}^2(\boldsymbol{X})}{100} \|\boldsymbol{u}_t * \boldsymbol{w}_{t,\perp} * \boldsymbol{w}_{t,\perp}^\top * \boldsymbol{u}_t^\top\|_F \\ &\leq \left(1 - \frac{\mu}{200} \sigma_{\min}^2(\boldsymbol{X})\right) \cdot 10\sqrt{kr} \left(1 - \frac{1}{400} \mu \sigma_{\min}(\boldsymbol{X})^2\right)^{t-t_1} \|\boldsymbol{X}\|^2 \\ &\quad + \mu \frac{\sigma_{\min}^2(\boldsymbol{X})}{100} \cdot \frac{5}{2} \sqrt{kr} \left(1 - \frac{\mu}{400} \sigma_{\min}(\boldsymbol{X})^2\right)^{t-t_1} \|\boldsymbol{X}\|^2 \\ &\leq 10\sqrt{kr} \left(1 - \frac{1}{400} \mu \sigma_{\min}(\boldsymbol{X})^2\right)^{t+1-t_1} \|\boldsymbol{X}\|^2 \end{aligned}$$

Hence, by induction, the five bullet points hold for t + 1.

If b<sup>t</sup> <sup>=</sup> <sup>t</sup>2, then, we can use Lemma [E.7,](#page-38-5) the previous bullet points, and the definition of <sup>t</sup><sup>2</sup> to bound

$$\begin{aligned} \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top\|_F &\leq 4 \|\boldsymbol{\nu}_{\boldsymbol{x}^\top} * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top)\|_F + \|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\|_F \\ &\leq 40 \sqrt{kr} \left(1 - \frac{1}{400} \mu \sigma_{\min}(\boldsymbol{x})^2\right)^{\hat{t}-t_1} \|\boldsymbol{x}\|^2 + \frac{5}{2} \sqrt{kr} \left(1 - \frac{1}{400} \mu \sigma_{\min}(\boldsymbol{x})^2\right)^{\hat{t}-t_1} \|\boldsymbol{x}\|^2 \\ &= \frac{85}{2} \sqrt{kr} \left(1 - \frac{1}{400} \mu \sigma_{\min}(\boldsymbol{x})^2\right)^{\hat{t}-t_1} \|\boldsymbol{x}\|^2 \\ &\lesssim \sqrt{kr} \left(\frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\boldsymbol{x}\|^{7/4}}{\gamma^{7/4}}\right)^{-3/4} \|\boldsymbol{x}\|^2 \\ &\lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\boldsymbol{x}\|^{11/16} \end{aligned}$$

If instead we have b<sup>t</sup> <sup>=</sup> <sup>t</sup>3, then

$$\begin{aligned} & \|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top\|_F \\ & \leq 4\|\boldsymbol{v}_{\boldsymbol{x},\perp}^\top * (\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top)\|_F + \|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\|_F \\ & \leq 4\|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top\|_F + \|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\|_F \\ & \leq 4(\sqrt{k(\min\{n, R\} - r)} + 1)\|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\|_F + \|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\|_F \\ & = 4(\sqrt{k(\min\{n, R\} - r)} + 5)\|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\|_F \\ & \leq 4(\sqrt{k(\min\{n, R\} - r)} + 5)\sqrt{\min\{n, R\} - r}\|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp} * \boldsymbol{w}_{\hat{t},\perp}^\top * \boldsymbol{u}_{\hat{t}}^\top\| \\ & \leq 4(\sqrt{k(\min\{n, R\} - r)} + 5)\sqrt{\min\{n, R\} - r}\|\boldsymbol{u}_{\hat{t}} * \boldsymbol{w}_{\hat{t},\perp}\|^2 \\ & \leq 4(\sqrt{k(\min\{n, R\} - r)} + 5)\sqrt{k(\min\{n, R\} - r)} \left(1 + 80\mu c_2 \sqrt{k} \sigma_{\min}(\boldsymbol{x})^2\right)^{2(\hat{t}-t_1)} \|\boldsymbol{u}_{t_1} * \boldsymbol{w}_{t_1,\perp}\|^2 \\ & \leq 4(\sqrt{k(\min\{n, R\} - r)} + 5)\sqrt{k(\min\{n, R\} - r)} \left(1 + 80\mu c_2 \sqrt{k} \sigma_{\min}(\boldsymbol{x})^2\right)^{2(\hat{t}-t_1)} \cdot 9\gamma^{63/32} \sigma_{\min}(\boldsymbol{x})^{1/32} \\ & \lesssim k(\min\{n, R\} - r) \left(\frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\boldsymbol{x}\|^{7/4}}{\gamma^{7/4}}\right)^{O(c_2)} \gamma^{63/32} \sigma_{\min}(\boldsymbol{x})^{1/32} \\ & \lesssim k(\min\{n, R\} - r) \left(\frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\boldsymbol{x}\|^{7/4}}{\gamma^{7/4}}\right)^{O(c_2)} \gamma^{21/16} \gamma^{21/32} \frac{\|\boldsymbol{x}\|^{1/32}}{\kappa^{1/32}} \\ & \lesssim k(\min\{n, R\} - r) \left(\frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\boldsymbol{x}\|^{7/4}}{\gamma^{7/4}}\right)^{O(c_2)} \gamma^{21/16} \left(\frac{\|\boldsymbol{x}\|}{\min\{n, R\} \kappa^3}\right)^{21/32} \frac{\|\boldsymbol{x}\|^{1/32}}{\kappa^{1/32}} \\ & \lesssim \frac{k(\min\{n, R\} - r)}{\min\{n, R\}^{21/32}} \left(\frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\boldsymbol{x}\|^{7/4}}{\gamma^{7/4}}\right)^{O(c_2)} \gamma^{21/16} \kappa^{-2} \|\boldsymbol{x}\|^{11/16} \\ & \lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\boldsymbol{x}\|^{11/16}. \end{aligned}$$

So in either case, we have

$$\|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top\|_F \lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\boldsymbol{x}\|^{11/16},$$

and thus,

$$\frac{\|\boldsymbol{x} * \boldsymbol{x}^\top - \boldsymbol{u}_{\hat{t}} * \boldsymbol{u}_{\hat{t}}^\top\|_F}{\|\boldsymbol{x}\|^2} \lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\boldsymbol{x}\|^{-21/16}.$$

Finally, by the definition of b<sup>t</sup>, we have that

$$\begin{aligned} \hat{t} - t_* &\leq t_2 - t_* \\ &\leq (t_2 - t_1) + (t_1 - t_*) \\ &\leq \frac{300}{\mu\sqrt{k}\sigma_{\min}(\mathcal{X})^2} \ln \left( \frac{5}{18} \kappa^{1/4} \sqrt{\frac{r}{k(\min\{n, R\} - r)}} \frac{\|\mathcal{X}\|^{7/4}}{\gamma^{7/4}} \right) + \frac{16}{\mu\sigma_{\min}(\mathcal{X})^2} \log \left( \frac{2}{\gamma\sqrt{10}} \sigma_{\min}(\mathcal{X}) \right) \\ &\lesssim \frac{1}{\mu\sigma_{\min}(\mathcal{X})^2} \ln \left( \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{\|\mathcal{X}\|}{\gamma} \right) \end{aligned}$$

## F. Proof of Main Result

Now that our analyses of the spectral stage and the convergence stage are complete, we are ready to combine these pieces to obtain the proof of our main result. Since <sup>A</sup> satisfies RIP(2<sup>r</sup> + 1, δ), by Lemma [G.2,](#page-50-1) <sup>A</sup> also satisfies S2SRIP(2r, √ 2krδ). Hence, E := (I − A<sup>∗</sup>A)(X ∗ X <sup>⊤</sup>) satisfies

$$\|\boldsymbol{\varepsilon}\| = \|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\boldsymbol{x} * \boldsymbol{x}^\top)\| \leq \sqrt{2kr}\delta\|\boldsymbol{x} * \boldsymbol{x}^\top\| \leq \sqrt{2kr} \cdot c\kappa^{-4} r^{-1/2} \cdot \|\boldsymbol{x}\|^2 = c\sqrt{k\kappa}^{-2} \sigma_{\min}(\boldsymbol{x})^2.$$

Then, by applying Lemma [D.9,](#page-25-1) with ϵ = 1 C˜ e −3˜c , we have that with probability at least 1 − k(Cϵ ˜ ) <sup>R</sup>−2r+1 − ke−cR˜ = 1 − ke−3˜c(R−2r+1) − ke−cR˜ ≥ 1 − ke−3˜c· 3 <sup>R</sup> − ke−cR˜ = 1 − O(ke−cR˜ ), after

$$t_* \lesssim \frac{1}{\mu\sigma_{\min}(\mathcal{X})^2} \ln \left( \frac{2\kappa^2 \sqrt{n}}{\tilde{c}_3 \sqrt{\min\{n; R\}}} \right)$$

iterations, we have

$$\|\mathcal{U}_{t_*}\| \leq 3\|\mathcal{X}\| \quad (\text{F.1})$$

$$\| \mathcal{V}_{x^\perp} * \mathcal{V}_{u_{t_*} * \mathcal{W}_{t_*}} \| \leq c\kappa^{-2}. \quad (\text{F.2})$$

and for each 1 ≤ j ≤ k, we have

$$\sigma_r \left( \overline{\mathcal{U}_{t_*} * \mathcal{W}_{t_*}}^{(j)} \right) \geq \frac{1}{4} \alpha \beta \quad (\text{F.3})$$

$$\sigma_1 \left( \overline{\mathcal{U}_{t_*} * \mathcal{W}_{t_*,\perp}}^{(j)} \right) \leq \frac{\kappa^{-2}}{8} \alpha \beta \quad (\text{F.4})$$

(F.5)

where (since R ≥ 3r and ϵ is a constant),

$$\sqrt{k} \lesssim \beta \lesssim \sqrt{k} \left( \frac{2\kappa^2 \sqrt{n}}{\tilde{c}_3 \sqrt{\min\{n; R\}}} \right)^{16\kappa^2}.$$

By choosing

$$\alpha \lesssim \frac{4c_2\sigma_{\min}(\mathcal{X})}{\kappa^2 \min\{n, R\}\sqrt{k}} \left( \frac{2\kappa^2\sqrt{n}}{\tilde{c}_3\sqrt{\min\{n, R\}}} \right)^{-16\kappa^2},$$

we have

$$\gamma = \frac{1}{4}\alpha\beta \lesssim \frac{c_2\sigma_{\min}(\boldsymbol{\chi})}{\kappa^2 \min\{n, R\}}.$$

Also, <sup>κ</sup> −2 8 αβ = <sup>2</sup>κ<sup>2</sup> γ ≤ 2γ holds. Therefore, we can apply Theorem [E.1,](#page-40-0) which gives us that after

$$\hat{t} - t_* \lesssim \frac{1}{\mu\sigma_{\min}(\boldsymbol{X})^2} \ln \left( \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{\|\boldsymbol{X}\|}{\gamma} \right)$$

iterations beyond the first phase, we have

$$\frac{\|\mathcal{U}_{\hat{t}} * \mathcal{U}_{\hat{t}}^\top - \mathcal{X} * \mathcal{X}^\top\|_F}{\|\mathcal{X}\|^2} \lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\mathcal{X}\|^{-21/16}.$$

The total amount of iterations is then bounded by

$$\begin{aligned} \hat{t} &= t_* + (\hat{t} - t_*) \\ &\lesssim \frac{1}{\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2} \ln \left( \frac{2\kappa^2\sqrt{n}}{\tilde{c}_3\sqrt{\min\{n, R\}}} \right) + \frac{1}{\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2} \ln \left( \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{\|\boldsymbol{\mathcal{X}}\|}{\gamma} \right) \\ &\lesssim \frac{1}{\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2} \ln \left( \frac{2\kappa^2\sqrt{n}}{\tilde{c}_3\sqrt{\min\{n, R\}}} \cdot \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{\|\boldsymbol{\mathcal{X}}\|}{\gamma} \right) \\ &\lesssim \frac{1}{\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2} \ln \left( \frac{2\kappa^2\sqrt{n}}{\tilde{c}_3\sqrt{\min\{n, R\}}} \cdot \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{4\|\boldsymbol{\mathcal{X}}\|}{\alpha\beta} \right) \\ &\lesssim \frac{1}{\mu\sigma_{\min}(\boldsymbol{\mathcal{X}})^2} \ln \left( \frac{C_1\kappa n}{\min\{n, R\}} \cdot \min \left\{ 1, \frac{\kappa r}{k(\min\{n, R\} - r)} \right\} \frac{\|\boldsymbol{\mathcal{X}}\|}{k\alpha} \right), \end{aligned}$$

where we have used the choice of γ = 1 4 αβ and the fact that β ≳ √ k. Finally, the error is bounded by

$$\begin{aligned} \frac{\|\mathcal{U}_{\hat{t}} * \mathcal{U}_{\hat{t}}^\top - \mathcal{X} * \mathcal{X}^\top\|_F}{\|\mathcal{X}\|^2} &\lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \gamma^{21/16} \|\mathcal{X}\|^{-21/16} \\ &\lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} (\alpha\beta)^{21/16} \|\mathcal{X}\|^{-21/16} \\ &\lesssim k^{5/4} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} k^{21/32} \left( \frac{2\kappa^2 \sqrt{n}}{\tilde{c}_3 \sqrt{\min\{n, R\}}} \right)^{21\kappa^2} \left( \frac{\alpha}{\|\mathcal{X}\|} \right)^{21/16} \\ &\lesssim k^{61/32} r^{1/8} \kappa^{-3/16} (\min\{n, R\} - r)^{3/8} \left( \frac{C_2 \kappa^2 \sqrt{n}}{\sqrt{\min\{n, R\}}} \right)^{21\kappa^2} \left( \frac{\alpha}{\|\mathcal{X}\|} \right)^{21/16}, \end{aligned}$$

as desired.

Remark: One could obtain similar results for the cases where r ≤ R < 2r and 2r ≤ R < 3r by choosing the parameter ϵ ∈ (0, 1) appropriately.

## G. Restricted Isometry Property

In this section, we show that a measurement operator which satisfies the standard restricted isometry property also satisfies two other variants of the restricted isometry property - a fact which we used in our analysis of the convergence stage.

We say that a measurement operator A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> satisfies the spectral-to-spectral Restricted Isometry Property of rank-r with constant δ > 0 (abbreviated S2SRIP(r, δ)) if for all tensors Z ∈ S <sup>n</sup>×n×<sup>k</sup> with tubal-rank ≤ r,

$$\|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\mathbf{Z})\| \leq \delta \|\mathbf{Z}\|.$$

We say that a measurement operator A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> satisfies the spectral-to-nuclear Restricted Isometry Property with constant δ > 0 (abbreviated S2NRIP(δ)) if for all tensors Z ∈ S <sup>n</sup>×n×<sup>k</sup> with tubal-rank ≤ r,

$$\|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\mathbf{Z})\| \leq \delta \|\mathbf{Z}\|_*.$$

Lemma G.1. *Suppose that* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies RIP*(r + r ′ , δ) *with* 0 < δ < 1*. Then, for any* Z, Y ∈ S n×n×k *with* rank(Z) ≤ r *and* rank(Y) ≤ r ′ *, we have*

$$|\langle (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}), \mathcal{Y} \rangle| \leq \delta \|\mathcal{Z}\|_F \|\mathcal{Y}\|_F.$$

*Proof.* Let Y ′ = ∥Z∥<sup>F</sup> ∥Y∥<sup>F</sup> Y so that ∥Y ′ ∥<sup>F</sup> = ∥Z∥<sup>F</sup> . Note that Z + Y ′ ∈ S n×n×k and Z − Y ′ ∈ S <sup>n</sup>×n×<sup>k</sup> both have tubal rank ≤ r + r ′ . Then, by using the identities ∥a + b∥ <sup>2</sup> − ∥a − b∥ <sup>2</sup> = 4 ⟨a, b⟩ and ∥a + b∥ <sup>2</sup> + ∥a − b∥ <sup>2</sup> = 2∥a∥ <sup>2</sup> + 2∥b∥ 2 (which both hold over any inner product space) along with the fact that A satisfies RIP(r + r ′ , δ), we have:

$$\begin{aligned} \langle \langle \mathcal{I} - \mathcal{A}^* \mathcal{A} \rangle (\mathcal{Z}), \mathcal{Y}' \rangle &= \langle \mathcal{Z}, \mathcal{Y}' \rangle - \langle \mathcal{A}^* \mathcal{A}(\mathcal{Z}), \mathcal{Y}' \rangle \\ &= \langle \mathcal{Z}, \mathcal{Y}' \rangle - \langle \mathcal{A}(\mathcal{Z}), \mathcal{A}(\mathcal{Y}') \rangle \\ &= \langle \mathcal{Z}, \mathcal{Y}' \rangle - \frac{1}{4} \|\mathcal{A}(\mathcal{Z} + \mathcal{Y}')\|_2^2 + \frac{1}{4} \|\mathcal{A}(\mathcal{Z} - \mathcal{Y}')\|_2^2 \\ &\leq \langle \mathcal{Z}, \mathcal{Y}' \rangle - \frac{1}{4} (1 - \delta) \|\mathcal{Z} + \mathcal{Y}'\|_F^2 + \frac{1}{4} (1 + \delta) \|\mathcal{Z} - \mathcal{Y}'\|_F^2 \\ &= \langle \mathcal{Z}, \mathcal{Y}' \rangle - \frac{1}{4} (\|\mathcal{Z} + \mathcal{Y}'\|_F^2 - \|\mathcal{Z} - \mathcal{Y}'\|_F^2) + \frac{1}{4} \delta (\|\mathcal{Z} + \mathcal{Y}'\|_F^2 + \|\mathcal{Z} - \mathcal{Y}'\|_F^2) \\ &= \frac{1}{2} \delta (\|\mathcal{Z}\|_F^2 + \|\mathcal{Y}'\|_F^2) \\ &= \delta \|\mathcal{Z}\|_F \|\mathcal{Y}'\|_F \end{aligned}$$

In a similar manner, (I − A<sup>∗</sup>A)(Z), Y ′ ≥ −δ∥Z∥<sup>F</sup> ∥Y ′ ∥<sup>F</sup> . Hence, (I − A<sup>∗</sup>A)(Z), Y ′   ≤ δ∥Z∥<sup>F</sup> ∥Y ′ ∥<sup>F</sup> . Then, since Y is a scalar multiple of Y ′ , we have

$$|\langle (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}), \mathbf{y} \rangle| = \frac{\| \mathbf{y} \|_F^F}{\| \mathbf{y} \|_F^F} |\langle (\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}), \mathbf{y}' \rangle| \leq \frac{\| \mathbf{y} \|_F^F}{\| \mathbf{y} \|_F^F} \delta \| \mathcal{Z} \|_F \| \mathbf{y}' \|_F = \delta \| \mathcal{Z} \|_F \| \mathbf{y} \|_F.$$

Lemma G.2. *Suppose that* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies RIP*(r + 1, δ1)*, where* 0 < δ<sup>1</sup> < 1*. Then,* A *also satisfies S2SRIP*(r, √ krδ1)*.*

*Proof.* Suppose Z ∈ S <sup>n</sup>×n×<sup>k</sup> has tubal-rank r. Since (I − A<sup>∗</sup>A)(Z) is symmetric, its t-SVD is of the form

$$(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z}) = \mathcal{V}_{(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z})} * \Sigma_{(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z})} * \mathcal{V}_{(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z})}^\top.$$

Now, define V = V(I−A∗A)(Z)(:, 1, :) ∈ <sup>R</sup> n×1×k and let s ∈ R <sup>1</sup>×1×<sup>k</sup> be defined by <sup>s</sup>(1, <sup>1</sup>, ℓ) = √ k e √ <sup>−</sup>12πjℓ where j = arg max<sup>j</sup> ′ |Σb(1, <sup>1</sup>, j′ )|. With this definition, one can check that D (I − A<sup>∗</sup>A)(Z), V ∗ s ∗ V ⊤ E  <sup>=</sup> <sup>∥</sup>(I − A<sup>∗</sup>A)(Z)∥. Then, since A satisfies RIP(r + 1, δ1) and rank(Z) ≤ r and rank(V ∗ s ∗ V <sup>⊤</sup>) = 1, by Lemma [G.1,](#page-49-1) we have

$$\begin{aligned} \|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\mathcal{Z})\| &= \left| \langle (\mathcal{I} - \mathcal{A}^*\mathcal{A})(\mathcal{Z}), \boldsymbol{\nu} * \boldsymbol{s} * \boldsymbol{\nu}^\top \rangle \right| \\ &\leq \delta_1 \|\boldsymbol{\nu} * \boldsymbol{s} * \boldsymbol{\nu}^\top\|_F \|\mathcal{Z}\|_F \\ &= \delta_1 \|\mathcal{Z}\|_F \\ &\leq \delta_1 \sqrt{kr} \|\mathcal{Z}\|. \end{aligned}$$

Since the bound ∥(I − A<sup>∗</sup>A)(Z)∥ ≤ δ<sup>1</sup> √ kr∥Z∥ holds for any Z ∈ S <sup>n</sup>×n×<sup>k</sup> with tubal rank ≤ r, we have that A satisfies S2SRIP(r, √ krδ1).

Lemma G.3. *Suppose that* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies RIP*(2, δ2) *where* 0 < δ<sup>2</sup> < 1*. Then,* A *also satisfies S2NRIP*( √ kδ2)*.*

*Proof.* Since A satisfies RIP(2, δ2), by Lemma [G.2](#page-50-1) for r = 1, A satisfies S2SRIP(1, √ kδ2). Now, suppose that Z ∈ S n×n×k . Since Z is symmetric, it has a t-SVD in the form

$$\mathcal{Z} = \sum_{i=1}^n \mathbf{v}_i * \mathbf{s}_i * \mathbf{v}_i^\top.$$

Then, since each term V<sup>i</sup> ∗ s<sup>i</sup> ∗ V ⊤ i is symmetric with tubal rank 1, we have

$$\begin{aligned} \|(\mathcal{I} - \mathcal{A}^*\mathcal{A})(\mathcal{Z})\| &= \left\| (\mathcal{I} - \mathcal{A}^*\mathcal{A}) \left( \sum_{i=1}^n \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right) \right\| \\ &= \left\| \sum_{i=1}^n (\mathcal{I} - \mathcal{A}^*\mathcal{A}) \left( \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right) \right\| \\ &\leq \sum_{i=1}^n \left\| (\mathcal{I} - \mathcal{A}^*\mathcal{A}) \left( \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right) \right\| \\ &\leq \sum_{i=1}^n \sqrt{k} \delta_2 \left\| \boldsymbol{\nu}_i * \boldsymbol{s}_i * \boldsymbol{\nu}_i^\top \right\| \\ &= \sum_{i=1}^n \sqrt{k} \delta_2 \|\boldsymbol{s}_i\| \\ &\leq \sqrt{k} \delta_2 \|\mathcal{Z}\|_* \end{aligned}$$

Since the bound <sup>∥</sup>(I −A<sup>∗</sup>A)(Z)∥ ≤ √ kδ2∥Z∥<sup>∗</sup> holds for any Z ∈ S n×n×k , we have that A satisfies S2NRIP( √ kδ2).

Lemma G.4. *Suppose* A : S <sup>n</sup>×n×<sup>k</sup> → <sup>R</sup> <sup>m</sup> *satisfies RIP*(2r, δ3)*, where* 0 < δ<sup>3</sup> < 1*, and* V ∈ <sup>R</sup> n×r×k *satisfies* V <sup>⊤</sup>∗V = I*. Then, for any* Z ∈ S <sup>n</sup>×n×<sup>k</sup> *with* rank(Z) ≤ r*, we have*

$$\left\| \boldsymbol{\nu}^\top * [(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\boldsymbol{Z})] \right\|_F \leq \delta_3 \|\boldsymbol{Z}\|_F.$$

*Proof.* Let Z ∈ S n×n×k , and let Y = V <sup>⊤</sup>∗[(I−A∗A)(Z)] ∥V⊤∗[(I−A∗A)(Z)]∥<sup>F</sup> ∈ R r×n×k . Trivially, ∥Y∥<sup>F</sup> = 1, and so, ∥V ∗ Y∥ 2 <sup>F</sup> = ⟨V ∗ Y, V ∗ Y⟩ = D Y, V <sup>⊤</sup> ∗ V ∗ Y E = ⟨Y, Y⟩ = ∥Y∥ 2 <sup>F</sup> = 1. Then, by using Lemma [G.1,](#page-49-1) we have that

$$\begin{aligned} \left\| \boldsymbol{\nu}^\top * [(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z})] \right\|_F &= \langle \boldsymbol{\nu}^\top * [(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z})], \boldsymbol{\nu} \rangle \\ &= \langle [(\mathcal{I} - \mathcal{A}^* \mathcal{A})(\mathcal{Z})], \boldsymbol{\nu} * \boldsymbol{\nu} \rangle \\ &\leq \delta_3 \| \mathcal{Z} \|_F \| \boldsymbol{\nu} * \boldsymbol{\nu} \|_F \\ &= \delta_3 \| \mathcal{Z} \|_F \end{aligned}$$

## H. Properties of Aligned Matrix Subspaces

In this section, we collect some properties of matrices and their subspaces, useful for the proof of the results in the tensor Fourier domain.

Lemma H.1. *([\(Stoger & Soltanolkotabi, 2021\)](#page-10-0)) For some orthogonal matrix ¨* X ∈ C <sup>n</sup>×<sup>r</sup> *and some full-rank matrix* Y ∈ C <sup>n</sup>×<sup>R</sup> *consider* X<sup>H</sup>Y = V ΣW<sup>H</sup>*, and the following decomposition of* Y

$$Y = YWW^H + YW_\perp W_\perp^H \quad (\text{H.1})$$

*with its SVD decomposition* Y = P<sup>R</sup> <sup>i</sup>=1 σiuiv i *and the best rank-*r *approximation* Y<sup>r</sup> = P<sup>r</sup> <sup>i</sup>=1 σiuiv i *. Then if the distance between the column subspace of* Y<sup>r</sup> *and the subspace spanned by the columns of* X *is small enough, that is* ∥X<sup>H</sup> <sup>⊥</sup>V<sup>Y</sup><sup>r</sup> ∥ ≤ <sup>1</sup> 8 *, then the decomposition* [\(H.1\)](#page-51-1) *follows some low-rank approximation properties, namely*

$$\|X_{\perp}^H V_{YW}\| \leq 7 \|X_{\perp}^H V_{Y_r}\| \quad (\text{H.2})$$

$$\|YW_\perp\| \leq 2\sigma_{r+1}(Y). \quad (\text{H.3})$$

Lemma H.2. *For a matrix* X ∈ C n×r *,* r ≤ n*, with its SVD-decomposition* X = VXΣXW<sup>H</sup> <sup>X</sup> *and some a full-rank matrix* Y ∈ C <sup>n</sup>×R*, consider* V H <sup>X</sup> Y = V ΣW<sup>H</sup>*, and the following decomposition of* Y

$$Y = YWW^H + YW_\perp W_\perp^H. \quad (\text{H.4})$$

*Let matrix* H ∈ C <sup>r</sup>×<sup>r</sup> *be defined as*

$$H = V_X^H (\text{Id} + \mu Z) Y W$$

*with some* Z ∈ C <sup>n</sup>×n*, parameter* <sup>µ</sup> <sup>≤</sup> √ 3 ∥V <sup>H</sup>Y ∥ <sup>−</sup><sup>2</sup> *and* ∥V H <sup>⊥</sup> VY W ∥ ≤ c<sup>2</sup> *with sufficiently small constants* c1, c<sup>2</sup> > 0*. Then* H *can be represented as follows*

$$H = (\text{Id} + \mu\Sigma_X^2 - \mu P_1 + \mu P_2 + \mu^2 P_3)V_X Y W (\text{Id} - \mu W^{\text{H}} Y^{\text{H}} V_X V_X^{\text{H}} Y W)$$

*with matrices* P1, P2, P<sup>3</sup> ∈ <sup>C</sup> r×r *such that*

$$\begin{aligned}
P_1 &:= V_X^H Y Y^H V_{X^\perp} V_{X^\perp}^H V_{YW} (V_{YW})^{-1} (\text{Id} - \mu V_X^H Y Y^H V_X)^{-1} \\
P_2 &:= V_X^H (Z - X X^H + Y Y^H) V_{YW} (V_X^H V_{YW})^{-1} (\text{Id} - \mu V_X^H Y W W^H Y^H V_X)^{-1} \\
P_3 &:= \Sigma_X^2 V_X^H Y W (\text{Id} - \mu W^H Y^H V_X V_X^H Y W)^{-1} W^H Y^H V_X
\end{aligned}$$

*with*

$$\begin{aligned}\|P_1\| &\leq 4\|YW\|^2\|V_{X^\perp}V_{YW}\|^2 \\ \|P_2\| &\leq 4\|Z - XX^H + YY^H\| \\ \|P_3\| &\leq 2\|X\|^2\|YW\|^2.\end{aligned}$$

*Moreover, it holds that*

$$\sigma_{min}(H) \geq (1 + \mu\sigma_{min}^2(X) - \mu\|P_1\| - \mu\|P_2\| - \mu^2\|P_3\|)\sigma_{min}(V_X^H Y)(1 - \mu\sigma_{min}^2(V_X^H Y)).$$

*Proof.* The proof of this Lemma follows from Lemma 9.1 in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0) by using an independent matrix ¨ Z ∈ C <sup>n</sup>×<sup>n</sup> instead of the matrix A<sup>∗</sup>A(XX<sup>H</sup> − Y Y <sup>H</sup>), omitting the assumption ∥Y ∥ ≤ 3∥X∥ and updating respectively the transformation steps.

Lemma H.3. *For a matrix* X ∈ C n×r *,* r ≤ n *with its SVD-decomposition* X = VXΣXW<sup>H</sup> <sup>X</sup> *and some full-rank matrix* Y ∈ C <sup>n</sup>×<sup>R</sup> *and* Y<sup>1</sup> = (Id + µZ)Y *consider* V H <sup>X</sup> Y = V ΣW<sup>H</sup>*,* V H <sup>X</sup> Y<sup>1</sup> = V1Σ1W<sup>H</sup> 1 *, and the following decomposition of* Y *and* Y<sup>1</sup>

$$Y = YWW^H + YW_\perp W_\perp^H,$$

$$Y_1 = Y_1W_1W_1^H + Y_1W_{1,\perp}W_{1,\perp}^H.$$

*Assume that* V <sup>X</sup> Y1W *is invertible, which also implies that* Y1W *is has full-rank, and that* ∥V <sup>X</sup><sup>⊥</sup> <sup>V</sup><sup>Y</sup>1<sup>W</sup> ∥ ≤ <sup>1</sup> <sup>50</sup> *and* µ ≤ min n √ 3 ∥V H <sup>X</sup><sup>⊥</sup> Y W⊥∥ −2 , 1 9 ∥X∥ −2 o *and moreover,* µ *is small enough so that* 0 ⪯ Id − µV <sup>H</sup> <sup>X</sup><sup>⊥</sup> Y WW<sup>H</sup><sup>Y</sup> <sup>H</sup>VX<sup>⊥</sup> ⪯ Id*. Consider two matrices*

$$\begin{aligned} G_1 &:= -V_{X\perp}^H Y_1 W (V_X^H Y_1 W)^{-1} V_X^H Y_1 W_\perp W_\perp^H W_{1,\perp} \\ G_2 &:= V_{X\perp}^H Y_1 W_\perp W_\perp^H W_{1,\perp}. \end{aligned}$$

*Then these matrices can be represented as*

$$G_1 = \mu V_{X\perp}^H V_{Y_1 W} (V_X^H V_{Y_1 W})^{-1} M_1 V_{X\perp}^H Y W_\perp W_\perp^H W_{1,\perp}$$

*with* M<sup>1</sup> := V H <sup>X</sup> (ZVX<sup>⊥</sup> − XX<sup>H</sup>VX<sup>⊥</sup> ) *and*

$$G_2 = \left( \text{Id} - \mu M_2 + \mu M_3 \right) V_{X_{\perp}}^H Y W_{\perp} (\text{Id} - \mu W_{\perp}^H Y^H Y W_{\perp}) - \mu^2 (M_2 - M_3) V_{X_{\perp}}^H Y W_{\perp} W_{\perp}^H Y^H Y W_{\perp}).$$

*with* M<sup>2</sup> = V H <sup>X</sup><sup>⊥</sup> Y WW<sup>H</sup><sup>Y</sup> <sup>H</sup>VX<sup>⊥</sup> *and* M<sup>3</sup> := V H <sup>X</sup><sup>⊥</sup> (<sup>Z</sup> − (XX<sup>H</sup> − Y Y <sup>H</sup>))VX<sup>⊥</sup> *. Moreover, the norm of* <sup>G</sup><sup>1</sup> *and* <sup>G</sup><sup>2</sup> *can be bounded respectively as*

$$\begin{aligned} \|G_1\| &\leq 2\mu(\|V_{X^{\perp}}^{\text{H}} V_{YW}\| \|YW\|^2 + \|Z - (XX^{\text{H}} - YY^{\text{H}})\|) \|V_{X^{\perp}}^{\text{H}} V_{Y_1W}\| \|YW_{\perp}\|, \\ \|G_2\| &\leq \|YW_{\perp}\| \left( 1 - \mu \|YW_{\perp}\|^2 + \mu \|Z - (XX^{\text{H}} - YY^{\text{H}})\| \right) \\ &\quad + \mu^2 \left( \|YW\|^2 + \|Z - (XX^{\text{H}} - YY^{\text{H}})\| \right) \|YW_{\perp}\|^3. \end{aligned}$$

*Proof.* The proof of this Lemma follows from Lemma 9.2 in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0) by changing the matrix ¨ A<sup>∗</sup>A(XX<sup>H</sup> − Y Y <sup>H</sup>) to the independent matrix Z ∈ <sup>C</sup> <sup>n</sup>×<sup>n</sup> and taking into account the respective changes without having the condition ∥Y ∥ ≤ 3∥X∥.

Lemma H.4. *For a matrix* X ∈ C n×r *,* r ≤ n *with its SVD-decomposition* X = VXΣXW<sup>H</sup> <sup>X</sup> *and some full-rank matrix* Y ∈ C <sup>n</sup>×<sup>R</sup> *and* Y<sup>1</sup> := (Id + µZ)Y *consider* V H <sup>X</sup> Y = V ΣW<sup>H</sup>*,* V <sup>X</sup> Y<sup>1</sup> = V1Σ1W<sup>H</sup> 1 *, and the following decomposition of* Y *and* Y<sup>1</sup>

$$Y = YWW^H + YW_\perp W_\perp^H,$$

$$Y_1 = Y_1W_1W_1^H + Y_1W_{1,\perp}W_{1,\perp}^H.$$

*Then it holds that*

$$\|W_{\perp}^H W_{\perp}\| \leq \mu \left( 1 + \mu \frac{\|Z\| \|YW\|}{\sigma_{\min}(V_X^H Y)} \right) \|YW\| \|YW_{\perp}\| \|V_X^H V_{YW}\| + \mu \frac{\|Z - (XX^H - YY^H)\|}{\sigma_{\min}(V_X^H Y)} \|YW_{\perp}\| \quad (\text{H.5})$$

*Moreover, if for* P := Y W⊥W<sup>H</sup> <sup>⊥</sup>W1(V H Y W Y WW<sup>H</sup>W1) <sup>−</sup><sup>1</sup>V H Y W *the following applies*

$$\|\mu Z + P + \mu ZP\| \leq 1,$$

*then it holds that*

$$\begin{aligned} \|V_{X^\perp}^H V_{Y_1 W_1}\| &\leq \|V_{X^\perp}^H V_{YW}\| \left(1 - \frac{\mu}{2} \sigma_{\min}^2(X) + \mu \|YW_\perp\|\right) + \mu \|Z - (XX^H - YY^H)\| \\ &+ (1 + \mu \|Z\|) \frac{2 \|W_\perp^H W_1\| \|YW_\perp\|}{\sigma_{\min}(W^H W_1) \sigma_{\min}(YW)} \\ &+ 57 \left( \mu \|Z\| + (1 + \mu \|Z\|) \frac{\|W_\perp^H W_1\| \|YW_\perp\|}{\sigma_{\min}(W^H W_1) \sigma_{\min}(YW)} \right)^2 \end{aligned} \quad (\text{H.6})$$

*Proof.* The proof of inequality [\(H.5\)](#page-53-2) follows from the first part of the proof of Lemma B.3 in [\(Stoger & Soltanolkotabi,](#page-10-0) ¨ [2021\)](#page-10-0). For this one needs to change the matrix A<sup>∗</sup>A(XX<sup>H</sup> − Y Y <sup>H</sup>) in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0) to an independent ¨ matrix Z ∈ C <sup>n</sup>×<sup>n</sup> and take into account the above-given decomposition of matrices Y and Y<sup>1</sup> and lack of assumptions on µ and the norm of matrix Z. Inequality [\(H.6\)](#page-53-3) follows from the proof of Lemma 9.3 in [\(Stoger & Soltanolkotabi, 2021\)](#page-10-0). ¨

### I. Random Tubal Tensors

In this section, we derive bounds on the minimum and maximum singular values as well as the Frobenius norm of a random tubal tensor with i.i.d. Gaussian random entries. In our analysis of the spectral stage, we applied these lemmas to the small random initialization.

We start with the following proposition from Rudelson and Vershynin (2009), which bounds the smallest singular value of an r × R random real Gaussian matrix.

Proposition I.1 ([\(Rudelson & Vershynin, 2009\)](#page-10-10)). *Let* G ∈ R <sup>r</sup>×<sup>R</sup> *with* r ≤ R *have i.i.d.* N (0, 1) *entries. Then, for every* ϵ > 0*, we have*

$$\sigma_{min}(\mathbf{G}) \geq \epsilon(\sqrt{R} - \sqrt{r-1})$$

Also, the following proposition from Tao and Vu (2010) bounds the smallest singular value of an r × r random complex Gaussian matrix.

Proposition I.2 ([\(Tao & Vu, 2010\)](#page-10-11)). *Let* G ∈ R <sup>r</sup>×<sup>r</sup> *have i.i.d.* CN (0, 1) *entries. Then, for every* ϵ > 0*, we have*

$$\sigma_{min}(\mathbf{G}) \geq \frac{\epsilon}{\sqrt{r}}$$

*with probability at least* 1 − ϵ 2 *.*

Using these propositions, we can obtain a bound on the smallest singular value of an r × R random complex Gaussian matrix, provided that r ≤ R.

Lemma I.1. *Let* G ∈ C <sup>r</sup>×<sup>R</sup> *with* r ≤ R *have i.i.d.* CN (0, 1) *entries. Then, for every* ϵ > 0*, we have*

$$\sigma_{min}(\mathbf{G}) \geq \begin{cases} \epsilon(\sqrt{R} - \sqrt{2r-1}) & \text{if } R > 2r \\ \frac{\epsilon}{\sqrt{r}} & \text{if } r \leq R \leq 2r \end{cases}$$

*with probability at least*

$$\begin{cases} 1 - (C\epsilon)^{R-2r+1} - e^{-cR} & \text{if } R > 2r \\ 1 - \epsilon^2 & \text{if } r \leq R \leq 2r \end{cases}$$

*The constants* C, c > 0 *are universal.*

*Proof.* First, suppose R > 2r. Let G = UΣV <sup>H</sup> be the SVD of G where U ∈ <sup>C</sup> r×r and V ∈ C <sup>R</sup>×<sup>R</sup> are unitary and Σ ∈ R <sup>r</sup>×<sup>R</sup>. Then, the following real 2r × 2R matrix has a real SVD of

$$\begin{bmatrix} \operatorname{Re}\{\mathbf{G}\} & -\operatorname{Im}\{\mathbf{G}\} \\ \operatorname{Im}\{\mathbf{G}\} & \operatorname{Re}\{\mathbf{G}\} \end{bmatrix} = \begin{bmatrix} \operatorname{Re}\{\mathbf{U}\} & -\operatorname{Im}\{\mathbf{U}\} \\ \operatorname{Im}\{\mathbf{U}\} & \operatorname{Re}\{\mathbf{U}\} \end{bmatrix} \begin{bmatrix} \Sigma & 0 \\ 0 & \Sigma \end{bmatrix} \begin{bmatrix} \operatorname{Re}\{\mathbf{V}\} & -\operatorname{Im}\{\mathbf{V}\} \\ \operatorname{Im}\{\mathbf{V}\} & \operatorname{Re}\{\mathbf{V}\} \end{bmatrix}^T.$$

By using the fact that for any A ∈ R <sup>p</sup>×<sup>q</sup> with p ≤ q, σmin(A) <sup>2</sup> = min x∈R p ∥x∥2=1 ∥A<sup>T</sup> x∥ 2 2 , we have

$$\begin{aligned}\sigma_{\min}(\mathbf{G})^2 &= \sigma_{\min} \left( \begin{bmatrix} \operatorname{Re}\{\mathbf{G}\} & -\operatorname{Im}\{\mathbf{G}\} \\ \operatorname{Im}\{\mathbf{G}\} & \operatorname{Re}\{\mathbf{G}\} \end{bmatrix} \right)^2 \\ &= \min_{\substack{\mathbf{x} \in \mathbb{R}^{2r} \\ \|\mathbf{x}\|_2=1}} \left\| \begin{bmatrix} \operatorname{Re}\{\mathbf{G}\}^T & \operatorname{Im}\{\mathbf{G}\}^T \\ -\operatorname{Im}\{\mathbf{G}\}^T & \operatorname{Re}\{\mathbf{G}\}^T \end{bmatrix} \mathbf{x} \right\|_2^2 \\ &= \min_{\substack{\mathbf{x} \in \mathbb{R}^{2r} \\ \|\mathbf{x}\|_2=1}} \left[ \left\| \begin{bmatrix} \operatorname{Re}\{\mathbf{G}\}^T & \operatorname{Im}\{\mathbf{G}\}^T \end{bmatrix} \mathbf{x} \right\|_2^2 + \left\| \begin{bmatrix} -\operatorname{Im}\{\mathbf{G}\}^T & \operatorname{Re}\{\mathbf{G}\}^T \end{bmatrix} \mathbf{x} \right\|_2^2 \right] \\ &\geq \min_{\substack{\mathbf{x} \in \mathbb{R}^{2r} \\ \|\mathbf{x}\|_2=1}} \left\| \begin{bmatrix} \operatorname{Re}\{\mathbf{G}\}^T & \operatorname{Im}\{\mathbf{G}\}^T \end{bmatrix} \mathbf{x} \right\|_2^2 + \min_{\substack{\mathbf{x} \in \mathbb{R}^{2r} \\ \|\mathbf{x}\|_2=1}} \left\| \begin{bmatrix} \operatorname{Im}\{\mathbf{G}\}^T & \operatorname{Re}\{\mathbf{G}\}^T \end{bmatrix} \mathbf{x} \right\|_2^2 \\ &= \sigma_{\min} \left( \begin{bmatrix} \operatorname{Re}\{\mathbf{G}\} \\ \operatorname{Im}\{\mathbf{G}\} \end{bmatrix} \right)^2 + \sigma_{\min} \left( \begin{bmatrix} -\operatorname{Im}\{\mathbf{G}\} \\ \operatorname{Re}\{\mathbf{G}\} \end{bmatrix} \right)^2 \\ &= 2\sigma_{\min} \left( \begin{bmatrix} \operatorname{Re}\{\mathbf{G}\} \\ \operatorname{Im}\{\mathbf{G}\} \end{bmatrix} \right)^2,\end{aligned}$$

where the last line follows since reordering the rows of a matrix or flipping the sign of some rows doesn't change the singular values.

Since G ∈ C <sup>r</sup>×<sup>R</sup> has i.i.d. CN (0, 1) entries, √ 2 Re {G} Im {G} ∈ R <sup>2</sup>r×<sup>R</sup> has i.i.d. N (0, 1) entries. Therefore, by Proposition [I.1,](#page-53-4) we have that

$$\sigma_{\min}(\mathbf{G}) \geq \sigma_{\min} \left( \sqrt{2} \begin{bmatrix} \text{Re}\{\mathbf{G}\} \\ \text{Im}\{\mathbf{G}\} \end{bmatrix} \right) \geq \epsilon(\sqrt{R} - \sqrt{2r-1})$$

with probability at least 1 − (Cϵ) <sup>R</sup>−2r+1 − e <sup>−</sup>cR, as desired.

Next, suppose r ≤ R ≤ 2r. Let Gr×<sup>r</sup> be an r × r submatrix of G. Then,

$$\sigma_{\min}(\mathbf{G})^2 = \min_{\substack{\mathbf{x} \in \mathbb{C}^r \\ \|\mathbf{x}\|_2 = 1}} \|\mathbf{G}^H \mathbf{x}\|_2^2 \geq \min_{\substack{\mathbf{x} \in \mathbb{C}^r \\ \|\mathbf{x}\|_2 = 1}} \|\mathbf{G}_{r \times r}^H \mathbf{x}\|_2^2 = \sigma_{\min}(\mathbf{G}_{r \times r})^2.$$

Hence, by Proposition [I.2,](#page-54-0) we have

$$\sigma_{\min}(\mathbf{G}) \geq \sigma_{\min}(\mathbf{G}_{r \times r}) \geq \frac{\epsilon}{\sqrt{r}}$$

with probability at least 1 − ϵ 2 .

Using the above lemma, we can bound the smallest singular value of an r × R × k tubal tensor.

Lemma I.2. *Let* G ∈ R <sup>r</sup>×R×<sup>k</sup> *with* r ≤ R *have i.i.d.* N (0, 1 R ) *entries. Then, for every* ϵ > 0*, we have*

$$\sigma_{min}(\mathbf{G}) \geq \begin{cases} \frac{\epsilon\sqrt{k}(\sqrt{R} - \sqrt{2r-1})}{\sqrt{R}} & \text{if } R > 2r \\ \frac{\epsilon\sqrt{k}}{\sqrt{rR}} & \text{if } r \leq R \leq 2r \end{cases}$$

*with probability at least*

$$\begin{cases} 1 - k(C\epsilon)^{R-2r+1} - ke^{-cR} & \text{if } R > 2r \\ 1 - k\epsilon^2 & \text{if } r \leq R \leq 2r \end{cases}$$

*Proof.* Since the entries of G are i.i.d. N (0, 1 R ), the entries of Ge are i.i.d. CN (0, k R ). Hence, each scaled slice q R k Ge (j) ∈ C <sup>r</sup>×<sup>R</sup> for j = 1, . . . , k has i.i.d. CN (0, 1) entries. By Lemma [I.1,](#page-54-1) each scaled slice satisfies

$$\sigma_{\min} \left( \sqrt{\frac{R}{k}} \tilde{\mathbf{G}}^{(j)} \right) \geq \begin{cases} \epsilon(\sqrt{R} - \sqrt{2r-1}) & \text{if } R > 2r \\ \frac{\epsilon}{\sqrt{r}} & \text{if } r \leq R \leq 2r \end{cases}$$

with probability at least

$$\begin{cases} 1 - (C\epsilon)^{R-2r+1} - e^{-cR} & \text{if } R > 2r \\ 1 - \epsilon^2 & \text{if } r \leq R \leq 2r \end{cases}$$

Then, by taking a union bound, we have that

$$\sigma_{\min}(\mathcal{G}) = \min_{1 \leq j \leq k} \sigma_{\min}\left(\tilde{\mathcal{G}}^{(j)}\right) \geq \begin{cases} \frac{\epsilon\sqrt{k}(\sqrt{R} - \sqrt{2r} - 1)}{\sqrt{R}} & \text{if } R > 2r \\ \frac{\epsilon\sqrt{k}}{\sqrt{rR}} & \text{if } r \leq R \leq 2r \end{cases}$$

with probability at least

$$\begin{cases} 1 - k(C\epsilon)^{R-2r+1} - k\epsilon^{-cR} & \text{if } R > 2r \\ 1 - k\epsilon^2 & \text{if } r \leq R \leq 2r \end{cases}$$

The following proposition bounds the operator norm of an r × R random Gaussian matrix.

Proposition I.3 ([\(Vershynin, 2018\)](#page-10-12)). *Let* U ∈ C <sup>n</sup>×<sup>R</sup> *have i.i.d.* CN (0, 1) *entries. Then, with probability at least* 1 − O(e −c max{n,R} )*, we have*

$$\|\mathbf{U}\| \lesssim \sqrt{\max\{n, R\}}.$$

Lemma I.3. *Let* U ∈ R <sup>n</sup>×R×<sup>k</sup> *have i.i.d.* N (0, 1 R ) *entries. Then, with probability at least* 1 − O(ke−<sup>c</sup> max{n,R} )*, we have*

$$\| \mathbf{u} \| \lesssim \sqrt{\frac{k \max\{n, R\}}{R}}.$$

*Proof.* Since the entries of U are i.i.d. N (0, R ), the entries of Ue are i.i.d. CN (0, k R ). Hence, each scaled slice q R k Ue (j) ∈ C <sup>r</sup>×<sup>R</sup> for j = 1, . . . , k has i.i.d CN (0, 1) entries. By Proposition [I.3,](#page-55-2) each scaled slice satisfies

$$\left\| \sqrt{\frac{R}{k}} \tilde{\mathbf{u}}^{(j)} \right\| \lesssim \sqrt{\max\{n, R\}}$$

with probability at least 1 − O(e −c max{n,R} ). Then, by taking a union bound, we have that

$$\|\mathbf{u}\| = \max_{1 \leq j \leq k} \left\| \tilde{\mathbf{u}}^{(j)} \right\| \lesssim \sqrt{\frac{k \max\{n, R\}}{R}}$$

with probability at least 1 − O(ke−<sup>c</sup> max{n,R} ).

Lemma I.4. *Let* U ∈ R <sup>n</sup>×R×<sup>k</sup> *have i.i.d.* N (0, R ) *entries. Then, for any fixed* V<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>×1×<sup>k</sup> *with* ∥V1∥ = 1*, we have*

$$\| \boldsymbol{u}^\top * \boldsymbol{v}_1 \|_F \asymp \sqrt{k}$$

*with probability at least* 1 − O(ke−cR)*.*

*Proof.* Since the entries of U are i.i.d. N (0, 1 R ), the entries of Ue are i.i.d. CN (0, k R ), and thus, the entries of Ue ⊤ are also i.i.d. CN (0, k R ). Then, for each slice <sup>j</sup> = 1, . . . , k, each entry of the matrix-vector product <sup>U</sup>g<sup>⊤</sup> (j) Ve (j) <sup>1</sup> ∈ <sup>C</sup> <sup>R</sup> is i.i.d. CN (0, k R ∥Ve (j) <sup>1</sup> ∥ 2 F ). Hence, the quantity

$$\frac{2R}{k} \frac{\left\| \widetilde{\mathbf{u}}^{\top(j)} \widetilde{\mathbf{v}}_1(j) \right\|^2}{\left\| \widetilde{\mathbf{v}}_1(j) \right\|^2} \frac{F}{F}$$

has a χ 2 (2R) distribution. It follows that

$$\left\| \widetilde{\mathbf{u}^\top}^{(j)} \widetilde{\mathbf{v}}_1^{(j)} \right\|_F^2 \asymp k \left\| \widetilde{\mathbf{v}}_1^{(j)} \right\|_F^2$$

holds with probability at least 1 − O(e <sup>−</sup>cR). By taking a union bound over all j = 1, . . . , k, we get that

$$\left\| \boldsymbol{u}^\top * \boldsymbol{v}_1 \right\|_F^2 = \frac{1}{k} \left\| \widetilde{\boldsymbol{u}}^\top \odot \widetilde{\boldsymbol{v}}_1 \right\|_F^2 = \frac{1}{k} \sum_{j=1}^k \left\| \widetilde{\boldsymbol{u}}^{\top(j)} \widetilde{\boldsymbol{v}}_1^{(j)} \right\|_F^2 \asymp \sum_{j=1}^k \left\| \widetilde{\boldsymbol{v}}_1^{(j)} \right\|_F^2 = \left\| \widetilde{\boldsymbol{v}}_1 \right\|_F^2 = k \left\| \boldsymbol{v}_1 \right\|_F^2 = k,$$

i.e., ∥U <sup>⊤</sup> ∗ V1∥<sup>F</sup> ≍ √ k with probability at least 1 − O(ke−cR).